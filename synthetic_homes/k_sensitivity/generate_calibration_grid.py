"""
Generate the EnergyPlus calibration grid used for k-sensitivity testing.

The calibration grid is constructed as the full Cartesian product of k ordered
levels across the four calibration features, yielding k^4 calibration points.

This experiment depends on the Synthetic Homes repository for IDF generation,
but this script does not need to live inside that repository.

Pass the location of a cloned Synthetic Homes repository using:

    --synthetic-homes-root /path/to/synthetic-homes

All persistent experiment data are written beneath one output directory:

    <output-dir>/
    └── data/
        ├── calibration_meta.json
        ├── calibration_levels.json
        ├── summary_stats.json
        └── calibration_runtime.json

During execution, temporary EnergyPlus files are written beneath:

    <output-dir>/work/

By default, each simulation's EnergyPlus outputs and generated IDF are deleted
immediately after the required summary statistics have been extracted and a
durable checkpoint has been written. This keeps disk usage bounded by the
number of simulations running concurrently rather than by the total number of
calibration points.

The script does NOT produce experimental_set.json because the k-sensitivity
analysis uses only the compact summary statistics.

If --output-dir is omitted, outputs are written to:

    calibration_grid/

next to this script.

For the k-sensitivity experiment, the intended workflow is to generate the
k=15 master calibration grid once. Smaller grids are then obtained by
subselecting approximately evenly spaced levels from this master grid.

Example:

    python generate_calibration_grid.py \
        --synthetic-homes-root ../synthetic-homes \
        --epw ../synthetic-homes/weather/KMSP.epw \
        --output-dir ./calibration_grid \
        --k 15 \
        --max-workers 8

To continue an interrupted run, use the same command with:

    --resume

To preserve the generated IDFs and raw EnergyPlus outputs for debugging, add:

    --keep-simulation-files

To rebuild the aggregate summary/runtime files from existing checkpoints:

    python generate_calibration_grid.py \
        --output-dir ./calibration_grid \
        --aggregate-only
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from eppy.modeleditor import IDF


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "calibration_grid"


def _resolve_cli_path(path: str | Path) -> Path:
    """
    Resolve a user-supplied path.

    Relative paths are interpreted relative to the shell's current working
    directory.
    """
    path = Path(path).expanduser()

    if not path.is_absolute():
        path = Path.cwd() / path

    return path.resolve()


def _resolve_output_dir(
    output_dir: str | Path | None,
) -> Path:
    """
    Resolve the experiment output directory.

    If --output-dir is omitted, use calibration_grid/ next to this script.
    """
    if output_dir is None:
        return DEFAULT_OUTPUT_DIR.resolve()

    return _resolve_cli_path(output_dir)


def _write_json(
    path: Path,
    data: Any,
) -> None:
    """
    Write JSON atomically.

    The temporary file is replaced only after serialization succeeds.
    """
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp_path = path.with_name(
        path.name + ".tmp"
    )

    with tmp_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            data,
            handle,
            indent=2,
        )

    os.replace(
        tmp_path,
        path,
    )


def _read_json_if_exists(
    path: Path,
    default: Any,
) -> Any:
    """Read JSON if it exists; otherwise return the supplied default."""
    if not path.exists():
        return default

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(
            handle
        )


# ---------------------------------------------------------------------------
# Dependency-free terminal progress bar
# ---------------------------------------------------------------------------

def _format_duration(
    seconds: float | None,
) -> str:
    """Format a duration for compact terminal display."""
    if (
        seconds is None
        or not np.isfinite(seconds)
    ):
        return "--:--:--"

    seconds = max(
        0,
        int(round(seconds)),
    )

    hours, remainder = divmod(
        seconds,
        3600,
    )

    minutes, seconds = divmod(
        remainder,
        60,
    )

    return (
        f"{hours:d}:"
        f"{minutes:02d}:"
        f"{seconds:02d}"
    )


class TerminalProgressBar:
    """
    Dependency-free progress bar fixed to the bottom row of the terminal.

    In an interactive ANSI-compatible terminal, rows 1..N-1 are used as the
    normal scrolling region and row N is reserved for the progress display.

    Only the parent process should write to the terminal while the bar is
    active. ProcessPool workers therefore have their stdout and stderr file
    descriptors redirected to /dev/null during initialization.

    Progress redraws are throttled to avoid excessive terminal refreshes.

    If stdout is redirected or is otherwise not a TTY, periodic ordinary
    progress lines are emitted instead.
    """

    def __init__(
        self,
        total: int,
        description: str = "Calibration grid",
        initial: int = 0,
        render_interval: float = 0.20,
    ) -> None:
        self.total = max(
            0,
            int(total),
        )

        self.description = description

        self.completed = max(
            0,
            min(
                int(initial),
                self.total,
            ),
        )

        self._initial_completed = self.completed

        # Previously completed jobs are known successful jobs.
        self.successful = self.completed
        self.failed = 0

        self.started = time.perf_counter()

        self.is_tty = sys.stdout.isatty()

        self.rows = 0
        self.columns = 0

        self.render_interval = max(
            0.05,
            float(render_interval),
        )

        self.last_render_time = 0.0

        # For redirected output, emit approximately one line per 1%.
        self.plain_interval = max(
            1,
            self.total // 100,
        )

        self.last_plain_report = -1

        self._closed = False

        if self.is_tty:
            self._configure_terminal()

        self.render(
            force=True
        )

    def _terminal_size(
        self,
    ) -> tuple[int, int]:
        size = shutil.get_terminal_size(
            fallback=(120, 24)
        )

        return (
            max(3, size.lines),
            max(20, size.columns),
        )

    def _configure_terminal(
        self,
    ) -> None:
        """
        Reserve the final terminal row for the progress bar.

        ANSI CSI <top>;<bottom> r defines the scrolling region.
        """
        rows, columns = self._terminal_size()

        self.rows = rows
        self.columns = columns

        # Restore the default region first.
        sys.stdout.write(
            "\033[r"
        )

        # Reserve the last row for the progress display.
        sys.stdout.write(
            f"\033[1;{rows - 1}r"
        )

        sys.stdout.flush()

    def _check_resize(
        self,
    ) -> None:
        """Reconfigure the scrolling region if the terminal was resized."""
        if not self.is_tty:
            return

        rows, columns = self._terminal_size()

        if (
            rows != self.rows
            or columns != self.columns
        ):
            self._configure_terminal()

    def _build_line(
        self,
    ) -> str:
        elapsed = (
            time.perf_counter()
            - self.started
        )

        newly_processed = max(
            0,
            self.completed
            - self._initial_completed,
        )

        if (
            elapsed > 0
            and newly_processed > 0
        ):
            rate = (
                newly_processed
                / elapsed
            )
        else:
            rate = 0.0

        remaining = max(
            0,
            self.total
            - self.completed,
        )

        eta = (
            remaining / rate
            if rate > 0
            else None
        )

        if self.total > 0:
            fraction = min(
                1.0,
                self.completed
                / self.total,
            )
        else:
            fraction = 1.0

        percent = (
            100.0
            * fraction
        )

        prefix = (
            f"{self.description} "
        )

        suffix = (
            f" {percent:6.2f}%"
            f" {self.completed}/{self.total}"
            f" | ok={self.successful}"
            f" failed={self.failed}"
            f" | {rate:.2f} job/s"
            f" | ETA {_format_duration(eta)}"
        )

        columns = (
            self.columns
            if self.is_tty
            else shutil.get_terminal_size(
                fallback=(120, 24)
            ).columns
        )

        available = (
            columns
            - len(prefix)
            - len(suffix)
            - 2
        )

        if available >= 10:
            bar_width = min(
                30,
                available,
            )

            filled = int(
                round(
                    fraction
                    * bar_width
                )
            )

            filled = max(
                0,
                min(
                    bar_width,
                    filled,
                ),
            )

            bar = (
                "["
                + "█" * filled
                + "░" * (
                    bar_width
                    - filled
                )
                + "]"
            )

            line = (
                prefix
                + bar
                + suffix
            )

        else:
            line = (
                f"{self.description}: "
                f"{percent:6.2f}% "
                f"{self.completed}/{self.total}"
                f" ok={self.successful}"
                f" failed={self.failed}"
                f" {rate:.2f} job/s"
                f" ETA {_format_duration(eta)}"
            )

        # Avoid wrapping onto a second terminal line.
        return line[
            : max(
                1,
                columns - 1,
            )
        ]

    def render(
        self,
        force: bool = False,
    ) -> None:
        """Render the current progress state."""
        if self._closed:
            return

        now = time.perf_counter()

        if (
            self.is_tty
            and not force
            and self.completed < self.total
            and (
                now
                - self.last_render_time
                < self.render_interval
            )
        ):
            return

        if self.is_tty:
            self._check_resize()

        line = self._build_line()

        if self.is_tty:
            # Move directly to the reserved final row.
            sys.stdout.write(
                f"\033[{self.rows};1H"
            )

            # Clear it before drawing.
            sys.stdout.write(
                "\033[2K"
            )

            sys.stdout.write(
                line
            )

            sys.stdout.flush()

            self.last_render_time = now

            return

        should_report = (
            force
            or self.completed == self.total
            or (
                self.completed
                - self.last_plain_report
                >= self.plain_interval
            )
        )

        if should_report:
            print(
                line,
                flush=True,
            )

            self.last_plain_report = (
                self.completed
            )

            self.last_render_time = now

    def advance(
        self,
        succeeded: bool,
    ) -> None:
        """Record one processed job and redraw the progress bar."""
        self.completed = min(
            self.total,
            self.completed + 1,
        )

        if succeeded:
            self.successful += 1
        else:
            self.failed += 1

        self.render()

    def write(
        self,
        message: str,
    ) -> None:
        """
        Print a permanent message in the scrolling region above the bar.

        The progress display remains on the reserved final row.
        """
        if not self.is_tty:
            print(
                message,
                flush=True,
            )
            return

        self._check_resize()

        lines = (
            str(message)
            .rstrip()
            .splitlines()
        )

        if not lines:
            lines = [""]

        # Move to the bottom row of the scrolling region.
        sys.stdout.write(
            f"\033[{self.rows - 1};1H"
        )

        for line in lines:
            sys.stdout.write(
                "\033[2K"
            )

            sys.stdout.write(
                line
            )

            # Because this cursor is at the bottom of the scrolling region,
            # newline scrolls only rows 1..N-1.
            sys.stdout.write(
                "\n"
            )

        sys.stdout.flush()

        self.render(
            force=True
        )

    def close(
        self,
    ) -> None:
        """Restore normal terminal scrolling."""
        if self._closed:
            return

        if not self.is_tty:
            self.render(
                force=True
            )

            self._closed = True
            return

        # Draw the final state one last time.
        self.render(
            force=True
        )

        final_line = self._build_line()

        # Restore the terminal's normal full-screen scrolling region.
        sys.stdout.write(
            "\033[r"
        )

        # Move to the final row and replace the reserved display with a normal
        # persistent final progress line.
        sys.stdout.write(
            f"\033[{self.rows};1H"
        )

        sys.stdout.write(
            "\033[2K"
        )

        sys.stdout.write(
            final_line
        )

        sys.stdout.write(
            "\n"
        )

        sys.stdout.flush()

        self._closed = True


# ---------------------------------------------------------------------------
# Synthetic Homes dependency
# ---------------------------------------------------------------------------

def _load_synthetic_homes(
    synthetic_homes_root: str | Path,
):
    """
    Make a cloned Synthetic Homes repository importable.

    Expected layout:

        <synthetic-homes-root>/
        └── src/
            ├── config.py
            └── pipeline/
                └── idf_generation.py
    """
    synthetic_homes_root = (
        _resolve_cli_path(
            synthetic_homes_root
        )
    )

    src_dir = (
        synthetic_homes_root
        / "src"
    )

    pipeline_dir = (
        src_dir
        / "pipeline"
    )

    if not synthetic_homes_root.is_dir():
        raise FileNotFoundError(
            "Synthetic Homes repository does not exist:\n"
            f"  {synthetic_homes_root}"
        )

    if not src_dir.is_dir():
        raise FileNotFoundError(
            "Could not find the Synthetic Homes src/ directory:\n"
            f"  {src_dir}\n\n"
            "--synthetic-homes-root should point to the root of the cloned "
            "Synthetic Homes repository."
        )

    config_path = (
        src_dir
        / "config.py"
    )

    idf_generation_path = (
        pipeline_dir
        / "idf_generation.py"
    )

    if not config_path.exists():
        raise FileNotFoundError(
            "Could not find Synthetic Homes config.py:\n"
            f"  {config_path}"
        )

    if not idf_generation_path.exists():
        raise FileNotFoundError(
            "Could not find Synthetic Homes IDF-generation module:\n"
            f"  {idf_generation_path}"
        )

    if str(src_dir) not in sys.path:
        sys.path.insert(
            0,
            str(src_dir),
        )

    try:
        import config
        from pipeline.idf_generation import (
            generate_idf_from_geojson,
        )

    except ImportError as exc:
        raise ImportError(
            "Synthetic Homes was found, but its Python modules could not be "
            "imported. Make sure its dependencies are installed.\n\n"
            f"Synthetic Homes src directory:\n  {src_dir}"
        ) from exc

    return (
        synthetic_homes_root,
        src_dir,
        pipeline_dir,
        config,
        generate_idf_from_geojson,
    )


def _resolve_idd_path(
    raw_idd_path: str | Path,
    synthetic_homes_root: Path,
    synthetic_homes_src: Path,
) -> Path:
    """Resolve Synthetic Homes' configured EnergyPlus IDD path."""
    raw_path = Path(
        raw_idd_path
    ).expanduser()

    if raw_path.is_absolute():
        resolved = (
            raw_path.resolve()
        )

        if not resolved.exists():
            raise FileNotFoundError(
                "Synthetic Homes config.IDD_FILE_PATH points to a file that "
                "does not exist:\n"
                f"  {resolved}"
            )

        return resolved

    candidates = [
        synthetic_homes_src
        / raw_path,
        synthetic_homes_root
        / raw_path,
        Path.cwd()
        / raw_path,
    ]

    for candidate in candidates:
        resolved = (
            candidate.resolve()
        )

        if resolved.exists():
            return resolved

    checked = "\n".join(
        f"  {candidate.resolve()}"
        for candidate
        in candidates
    )

    raise FileNotFoundError(
        "Could not resolve config.IDD_FILE_PATH from Synthetic Homes.\n"
        f"Configured value: {raw_idd_path}\n\n"
        "Checked:\n"
        f"{checked}"
    )


# ---------------------------------------------------------------------------
# Calibration grid
# ---------------------------------------------------------------------------

ANCHOR_LEVELS = {
    "wall_r_value": (
        4.0,
        7.0,
        13.0,
        20.0,
        30.0,
    ),
    "roof_r_value": (
        10.0,
        20.0,
        30.0,
        40.0,
        50.0,
    ),
    "hvac_heating_cop": (
        0.70,
        0.80,
        0.90,
        0.95,
        1.00,
    ),
    "hvac_cooling_cop": (
        1.0,
        2.0,
        3.0,
        3.5,
        4.0,
    ),
}


def evenly_spaced_level_indices(
    k: int,
    max_k: int = 15,
) -> tuple[int, ...]:
    """
    Return k approximately evenly spaced one-based indices from 1..max_k.
    """
    if not 2 <= k <= max_k:
        raise ValueError(
            f"k must satisfy 2 <= k <= {max_k}; got {k}."
        )

    indices = tuple(
        int(
            round(
                1
                + i
                * (max_k - 1)
                / (k - 1)
            )
        )
        for i in range(k)
    )

    if len(set(indices)) != k:
        raise RuntimeError(
            f"Could not construct {k} unique levels "
            f"from 1..{max_k}: {indices}"
        )

    return indices


def build_parameter_levels(
    max_k: int = 15,
) -> dict[str, tuple[float, ...]]:
    """
    Expand the original five physical anchor levels to a max_k-level grid.

    Intermediate values are obtained using piecewise-linear interpolation.
    """
    if max_k < 5:
        raise ValueError(
            "max_k must be at least 5 so the original five anchors "
            "are retained."
        )

    anchor_positions = np.array(
        [
            index - 1
            for index
            in evenly_spaced_level_indices(
                5,
                max_k,
            )
        ],
        dtype=float,
    )

    target_positions = np.arange(
        max_k,
        dtype=float,
    )

    decimal_places = {
        "wall_r_value": 3,
        "roof_r_value": 3,
        "hvac_heating_cop": 4,
        "hvac_cooling_cop": 4,
    }

    levels: dict[
        str,
        tuple[float, ...],
    ] = {}

    for (
        parameter_name,
        anchors,
    ) in ANCHOR_LEVELS.items():
        interpolated = np.interp(
            target_positions,
            anchor_positions,
            np.asarray(
                anchors,
                dtype=float,
            ),
        )

        rounded = tuple(
            float(
                round(
                    value,
                    decimal_places[
                        parameter_name
                    ],
                )
            )
            for value
            in interpolated
        )

        if any(
            right <= left
            for left, right
            in zip(
                rounded,
                rounded[1:],
            )
        ):
            raise RuntimeError(
                f"Interpolated levels for {parameter_name} are not "
                f"strictly increasing: {rounded}"
            )

        levels[
            parameter_name
        ] = rounded

    return levels


@dataclass(frozen=True)
class CalibrationJob:
    """One point in the calibration grid."""

    name: str
    params: Mapping[str, Any]


# ---------------------------------------------------------------------------
# Worker initialization
# ---------------------------------------------------------------------------

_WORKER_GENERATE_IDF = None
_WORKER_PIPELINE_DIR: Path | None = None


def _initialize_worker(
    synthetic_homes_root: str,
    idd_path: str,
) -> None:
    """Initialize EnergyPlus/Synthetic Homes state in the current process."""
    global _WORKER_GENERATE_IDF
    global _WORKER_PIPELINE_DIR

    (
        _,
        _,
        pipeline_dir,
        _,
        generate_idf_from_geojson,
    ) = _load_synthetic_homes(
        synthetic_homes_root
    )

    _WORKER_GENERATE_IDF = (
        generate_idf_from_geojson
    )

    _WORKER_PIPELINE_DIR = (
        pipeline_dir
    )

    try:
        IDF.setiddname(
            str(
                idd_path
            )
        )

    except Exception:
        existing_idd = getattr(
            IDF,
            "iddname",
            None,
        )

        if (
            existing_idd is None
            or Path(
                str(
                    existing_idd
                )
            ).resolve()
            != Path(
                idd_path
            ).resolve()
        ):
            raise


def _silence_worker_terminal_output() -> None:
    """
    Permanently redirect a ProcessPool worker's stdout and stderr to /dev/null.

    Redirecting the operating-system file descriptors, rather than only
    Python's sys.stdout/sys.stderr objects, also suppresses output produced by
    native libraries and subprocesses that inherit those descriptors.

    ProcessPool result communication does not use stdout or stderr, so normal
    future/result handling is unaffected.
    """
    devnull_fd = os.open(
        os.devnull,
        os.O_WRONLY,
    )

    try:
        os.dup2(
            devnull_fd,
            1,
        )

        os.dup2(
            devnull_fd,
            2,
        )

    finally:
        os.close(
            devnull_fd
        )


def _initialize_worker_process(
    synthetic_homes_root: str,
    idd_path: str,
) -> None:
    """
    Initialize a ProcessPool worker without allowing it to write to the TTY.
    """
    _silence_worker_terminal_output()

    _initialize_worker(
        synthetic_homes_root,
        idd_path,
    )


# ---------------------------------------------------------------------------
# EnergyPlus result extraction
# ---------------------------------------------------------------------------

def _summarize_eplusout_csv(
    csv_path: Path,
) -> dict[str, dict[str, float]]:
    """
    Extract compact summary statistics from one EnergyPlus CSV.

    Hourly arrays are deliberately not retained.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            "EnergyPlus did not produce the expected CSV:\n"
            f"  {csv_path}"
        )

    df = pd.read_csv(
        csv_path
    )

    summary: dict[
        str,
        dict[str, float],
    ] = {}

    for column_name in df.columns:
        series = df[
            column_name
        ]

        if not pd.api.types.is_numeric_dtype(
            series
        ):
            continue

        numeric = pd.to_numeric(
            series,
            errors="coerce",
        ).dropna()

        if numeric.empty:
            continue

        values = numeric.to_numpy(
            dtype=float
        )

        summary[
            column_name
        ] = {
            "min": float(
                np.min(values)
            ),
            "max": float(
                np.max(values)
            ),
            "mean": float(
                np.mean(values)
            ),
            "std": float(
                np.std(values)
            ),
        }

    if not summary:
        raise RuntimeError(
            "EnergyPlus output CSV contained no usable numeric columns:\n"
            f"  {csv_path}"
        )

    return summary


def _read_text_tail(
    path: Path,
    max_chars: int = 20000,
) -> str:
    """Read the end of a diagnostic text file."""
    if not path.exists():
        return ""

    try:
        text = path.read_text(
            encoding="utf-8",
            errors="replace",
        )

    except OSError:
        return ""

    return text[
        -max_chars:
    ]


def _build_geojson(
    name: str,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the Synthetic Homes input GeoJSON for one calibration point."""
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [0, 0],
                            [0, 10],
                            [10, 10],
                            [10, 0],
                            [0, 0],
                        ]
                    ],
                },
                "properties": {
                    "name": name,
                    "Total Square Feet Living Area": 1000,
                    "height_ft": 10,
                    "conditioned": True,
                    **dict(params),
                },
            }
        ],
    }


# ---------------------------------------------------------------------------
# EnergyPlus worker
# ---------------------------------------------------------------------------

def _run_energyplus_job(
    job: CalibrationJob,
    epw_path: str,
    work_root: str,
    checkpoint_dir: str,
    keep_simulation_files: bool,
) -> tuple[str, bool, float, str]:
    """
    Generate, simulate, summarize, checkpoint, and optionally clean one job.

    ProcessPool workers are silenced at the operating-system descriptor level,
    so only the parent process controls the terminal.

    The checkpoint is written before raw EnergyPlus files are removed.
    """
    if _WORKER_GENERATE_IDF is None:
        raise RuntimeError(
            "EnergyPlus worker was not initialized."
        )

    work_root_path = Path(
        work_root
    )

    checkpoint_dir_path = Path(
        checkpoint_dir
    )

    sim_dir = (
        work_root_path
        / job.name
    )

    checkpoint_path = (
        checkpoint_dir_path
        / f"{job.name}.json"
    )

    total_started = (
        time.perf_counter()
    )

    energyplus_runtime = 0.0
    expand_warning = ""
    error_message = ""
    error_tail = ""

    summary_stats: dict[
        str,
        dict[str, float],
    ] = {}

    succeeded = False

    try:
        # A work directory without a successful checkpoint is disposable.
        if sim_dir.exists():
            shutil.rmtree(
                sim_dir
            )

        sim_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        idf_path = (
            sim_dir
            / "in.idf"
        )

        # Preserve compatibility with Synthetic Homes code that may assume
        # execution from its src/pipeline directory.
        original_cwd = Path.cwd()

        try:
            if _WORKER_PIPELINE_DIR is not None:
                os.chdir(
                    _WORKER_PIPELINE_DIR
                )

            geojson = _build_geojson(
                job.name,
                job.params,
            )

            # This suppresses Python-level output in the max_workers=1 case.
            # ProcessPool workers are additionally silenced at the OS file-
            # descriptor level by _initialize_worker_process().
            with open(
                os.devnull,
                "w",
                encoding="utf-8",
            ) as devnull:
                with redirect_stdout(
                    devnull
                ), redirect_stderr(
                    devnull
                ):
                    _WORKER_GENERATE_IDF(
                        geojson,
                        str(idf_path),
                    )

        finally:
            os.chdir(
                original_cwd
            )

        expanded = (
            sim_dir
            / "expanded.idf"
        )

        expanded.unlink(
            missing_ok=True
        )

        for executable in (
            "expandobjects",
            "ExpandObjects",
        ):
            try:
                subprocess.run(
                    [
                        executable
                    ],
                    cwd=sim_dir,
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT,
                )

                expand_warning = ""
                break

            except FileNotFoundError as exc:
                expand_warning = str(
                    exc
                )

            except subprocess.CalledProcessError as exc:
                expand_warning = str(
                    exc
                )
                break

        input_idf = (
            expanded.name
            if expanded.exists()
            else idf_path.name
        )

        energyplus_started = (
            time.perf_counter()
        )

        try:
            subprocess.run(
                [
                    "energyplus",
                    "-w",
                    epw_path,
                    "-d",
                    "simulation_output",
                    "-r",
                    input_idf,
                ],
                cwd=sim_dir,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )

        finally:
            energyplus_runtime = (
                time.perf_counter()
                - energyplus_started
            )

        csv_path = (
            sim_dir
            / "simulation_output"
            / "eplusout.csv"
        )

        summary_stats = (
            _summarize_eplusout_csv(
                csv_path
            )
        )

        succeeded = True

    except Exception as exc:
        error_message = (
            f"{type(exc).__name__}: {exc}"
        )

        error_tail = (
            _read_text_tail(
                sim_dir
                / "simulation_output"
                / "eplusout.err"
            )
        )

    total_runtime = (
        time.perf_counter()
        - total_started
    )

    checkpoint = {
        "simulation_name": job.name,
        "succeeded": succeeded,
        "runtime_seconds": float(
            energyplus_runtime
        ),
        "total_wall_time_seconds": float(
            total_runtime
        ),
        "summary_stats": summary_stats,
        "expandobjects_warning": (
            expand_warning
        ),
        "error": (
            error_message
        ),
        "energyplus_error_tail": (
            error_tail
        ),
    }

    try:
        _write_json(
            checkpoint_path,
            checkpoint,
        )

    except Exception as checkpoint_error:
        # Do not delete raw outputs if the durable checkpoint could not be
        # written. Those files may be the only recoverable result.
        return (
            job.name,
            False,
            energyplus_runtime,
            (
                "Simulation completed but checkpoint could not be written: "
                f"{checkpoint_error}"
            ),
        )

    if not keep_simulation_files:
        shutil.rmtree(
            sim_dir,
            ignore_errors=True,
        )

    return (
        job.name,
        succeeded,
        energyplus_runtime,
        error_message,
    )


# ---------------------------------------------------------------------------
# Checkpoint and aggregate handling
# ---------------------------------------------------------------------------

def _load_successful_checkpoint_names(
    checkpoint_dir: Path,
) -> set[str]:
    """Return simulation names with valid successful checkpoints."""
    completed: set[str] = set()

    if not checkpoint_dir.is_dir():
        return completed

    for path in checkpoint_dir.glob(
        "cal_*.json"
    ):
        try:
            record = _read_json_if_exists(
                path,
                {},
            )

        except (
            OSError,
            json.JSONDecodeError,
        ):
            continue

        if (
            isinstance(record, dict)
            and record.get("succeeded") is True
            and isinstance(
                record.get("summary_stats"),
                dict,
            )
            and record["summary_stats"]
        ):
            name = record.get(
                "simulation_name"
            )

            if name is not None:
                completed.add(
                    str(name)
                )

    return completed


def _load_successful_aggregate_names(
    data_dir: Path,
) -> set[str]:
    """Return simulations known to be successful in aggregate files."""
    summary_path = (
        data_dir
        / "summary_stats.json"
    )

    runtime_path = (
        data_dir
        / "calibration_runtime.json"
    )

    summary = _read_json_if_exists(
        summary_path,
        {},
    )

    runtime = _read_json_if_exists(
        runtime_path,
        {},
    )

    if not isinstance(
        summary,
        dict,
    ):
        return set()

    jobs = (
        runtime.get(
            "jobs",
            {},
        )
        if isinstance(
            runtime,
            dict,
        )
        else {}
    )

    if not isinstance(
        jobs,
        dict,
    ):
        return set()

    completed = set()

    for (
        name,
        job_record,
    ) in jobs.items():
        if (
            isinstance(
                job_record,
                dict,
            )
            and job_record.get(
                "succeeded"
            ) is True
            and name in summary
        ):
            completed.add(
                str(name)
            )

    return completed


def _completed_job_names(
    data_dir: Path,
    checkpoint_dir: Path,
) -> set[str]:
    """Return successful jobs known from checkpoints or aggregates."""
    return (
        _load_successful_checkpoint_names(
            checkpoint_dir
        )
        | _load_successful_aggregate_names(
            data_dir
        )
    )


def _aggregate_results(
    data_dir: Path,
    checkpoint_dir: Path,
) -> tuple[int, int]:
    """
    Merge per-job checkpoints into summary_stats.json and runtime JSON.

    Existing aggregate files are retained and updated, allowing aggregation
    to continue correctly across resumed runs.
    """
    data_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        data_dir
        / "summary_stats.json"
    )

    runtime_path = (
        data_dir
        / "calibration_runtime.json"
    )

    summary_stats = _read_json_if_exists(
        summary_path,
        {},
    )

    runtime_payload = _read_json_if_exists(
        runtime_path,
        {},
    )

    if not isinstance(
        summary_stats,
        dict,
    ):
        summary_stats = {}

    if not isinstance(
        runtime_payload,
        dict,
    ):
        runtime_payload = {}

    runtime_jobs = runtime_payload.get(
        "jobs",
        {},
    )

    if not isinstance(
        runtime_jobs,
        dict,
    ):
        runtime_jobs = {}

    if checkpoint_dir.is_dir():
        for checkpoint_path in sorted(
            checkpoint_dir.glob(
                "cal_*.json"
            )
        ):
            try:
                record = _read_json_if_exists(
                    checkpoint_path,
                    {},
                )

            except (
                OSError,
                json.JSONDecodeError,
            ):
                print(
                    "Warning: could not read checkpoint "
                    f"{checkpoint_path}"
                )
                continue

            if not isinstance(
                record,
                dict,
            ):
                continue

            name = record.get(
                "simulation_name"
            )

            if name is None:
                continue

            name = str(name)

            succeeded = bool(
                record.get(
                    "succeeded",
                    False,
                )
            )

            runtime_jobs[
                name
            ] = {
                "runtime_seconds": float(
                    record.get(
                        "runtime_seconds",
                        0.0,
                    )
                ),
                "total_wall_time_seconds": float(
                    record.get(
                        "total_wall_time_seconds",
                        0.0,
                    )
                ),
                "succeeded": succeeded,
                "error": str(
                    record.get(
                        "error",
                        "",
                    )
                ),
                "expandobjects_warning": str(
                    record.get(
                        "expandobjects_warning",
                        "",
                    )
                ),
            }

            if succeeded:
                job_summary = record.get(
                    "summary_stats",
                    {},
                )

                if isinstance(
                    job_summary,
                    dict,
                ):
                    summary_stats[
                        name
                    ] = job_summary

    successful_jobs = sum(
        1
        for record
        in runtime_jobs.values()
        if (
            isinstance(
                record,
                dict,
            )
            and record.get(
                "succeeded"
            ) is True
        )
    )

    failed_jobs = sum(
        1
        for record
        in runtime_jobs.values()
        if (
            isinstance(
                record,
                dict,
            )
            and record.get(
                "succeeded"
            ) is False
        )
    )

    _write_json(
        summary_path,
        summary_stats,
    )

    _write_json(
        runtime_path,
        {
            "jobs": runtime_jobs,
            "summary": {
                "timed_jobs": len(
                    runtime_jobs
                ),
                "successful_jobs": (
                    successful_jobs
                ),
                "failed_jobs": (
                    failed_jobs
                ),
                "total_energyplus_compute_seconds": float(
                    sum(
                        float(
                            record.get(
                                "runtime_seconds",
                                0.0,
                            )
                        )
                        for record
                        in runtime_jobs.values()
                        if isinstance(
                            record,
                            dict,
                        )
                    )
                ),
            },
        },
    )

    print()
    print(
        f"Wrote: {summary_path}"
    )
    print(
        f"Wrote: {runtime_path}"
    )
    print(
        "Successful jobs in aggregate: "
        f"{successful_jobs}"
    )

    if failed_jobs:
        print(
            "Failed jobs in aggregate: "
            f"{failed_jobs}"
        )

    return (
        successful_jobs,
        failed_jobs,
    )


# ---------------------------------------------------------------------------
# Batch execution
# ---------------------------------------------------------------------------

def _run_energyplus_batch(
    jobs: Sequence[CalibrationJob],
    epw_path: str,
    work_root: Path,
    checkpoint_dir: Path,
    synthetic_homes_root: Path,
    idd_path: Path,
    max_workers: int,
    keep_simulation_files: bool,
    total_jobs: int,
    initial_completed: int = 0,
) -> tuple[int, int]:
    """
    Run pending EnergyPlus calibration jobs.

    The parent process exclusively owns terminal output. ProcessPool workers
    have stdout and stderr redirected to /dev/null before Synthetic Homes is
    imported.

    The progress bar represents the entire calibration grid, including
    simulations completed before a --resume invocation.
    """
    if max_workers < 1:
        raise ValueError(
            "max_workers must be at least 1."
        )

    pending_count = len(
        jobs
    )

    print()
    print(
        f"EnergyPlus jobs to run: {pending_count}"
    )
    print(
        f"Worker processes:       {max_workers}"
    )

    if initial_completed:
        print(
            f"Already completed:       {initial_completed}"
        )

    print()

    if pending_count == 0:
        print(
            "No EnergyPlus jobs need to be run."
        )

        return (
            0,
            0,
        )

    successful = 0
    failed = 0

    progress = TerminalProgressBar(
        total=total_jobs,
        description="Calibration grid",
        initial=initial_completed,
        render_interval=0.20,
    )

    try:
        if max_workers == 1:
            # This runs in the parent process, so do NOT use the permanently
            # silent ProcessPool initializer.
            _initialize_worker(
                str(
                    synthetic_homes_root
                ),
                str(
                    idd_path
                ),
            )

            for job in jobs:
                (
                    name,
                    succeeded,
                    _runtime_seconds,
                    message,
                ) = _run_energyplus_job(
                    job,
                    epw_path,
                    str(
                        work_root
                    ),
                    str(
                        checkpoint_dir
                    ),
                    keep_simulation_files,
                )

                if succeeded:
                    successful += 1
                else:
                    failed += 1

                    progress.write(
                        f"FAILED {name}: {message}"
                    )

                progress.advance(
                    succeeded
                )

            return (
                successful,
                failed,
            )

        executor = ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_initialize_worker_process,
            initargs=(
                str(
                    synthetic_homes_root
                ),
                str(
                    idd_path
                ),
            ),
        )

        try:
            futures = {
                executor.submit(
                    _run_energyplus_job,
                    job,
                    epw_path,
                    str(
                        work_root
                    ),
                    str(
                        checkpoint_dir
                    ),
                    keep_simulation_files,
                ): job.name
                for job in jobs
            }

            for future in as_completed(
                futures
            ):
                name = futures[
                    future
                ]

                try:
                    (
                        _,
                        succeeded,
                        _runtime_seconds,
                        message,
                    ) = future.result()

                except Exception as exc:
                    succeeded = False

                    message = (
                        f"{type(exc).__name__}: {exc}"
                    )

                    _write_json(
                        checkpoint_dir
                        / f"{name}.json",
                        {
                            "simulation_name": name,
                            "succeeded": False,
                            "runtime_seconds": 0.0,
                            "total_wall_time_seconds": 0.0,
                            "summary_stats": {},
                            "expandobjects_warning": "",
                            "error": message,
                            "energyplus_error_tail": "",
                        },
                    )

                if succeeded:
                    successful += 1
                else:
                    failed += 1

                    progress.write(
                        f"FAILED {name}: {message}"
                    )

                progress.advance(
                    succeeded
                )

        except BaseException:
            executor.shutdown(
                wait=False,
                cancel_futures=True,
            )
            raise

        else:
            executor.shutdown(
                wait=True
            )

        return (
            successful,
            failed,
        )

    finally:
        progress.close()


# ---------------------------------------------------------------------------
# Main generation workflow
# ---------------------------------------------------------------------------

def generate(
    synthetic_homes_root: str | Path,
    epw: str | Path,
    output_dir: str | Path | None = None,
    k: int = 15,
    max_workers: int = 1,
    resume: bool = False,
    keep_simulation_files: bool = False,
) -> None:
    """
    Generate and simulate the k^4 calibration grid.

    Each successful simulation is summarized immediately. Raw EnergyPlus
    output is then deleted unless keep_simulation_files=True.

    Resume status is determined from durable per-job checkpoints and, when
    available, existing aggregate result files.
    """
    if k < 5:
        raise ValueError(
            "This generator requires k >= 5 because the original five "
            "physical anchor levels must be retained."
        )

    if max_workers < 1:
        raise ValueError(
            "max_workers must be at least 1."
        )

    (
        synthetic_homes_root,
        synthetic_homes_src,
        _,
        config,
        _,
    ) = _load_synthetic_homes(
        synthetic_homes_root
    )

    idd_path = _resolve_idd_path(
        config.IDD_FILE_PATH,
        synthetic_homes_root,
        synthetic_homes_src,
    )

    epw_path = _resolve_cli_path(
        epw
    )

    if not epw_path.exists():
        raise FileNotFoundError(
            "EPW file does not exist:\n"
            f"  {epw_path}"
        )

    output_dir = _resolve_output_dir(
        output_dir
    )

    data_dir = (
        output_dir
        / "data"
    )

    work_dir = (
        output_dir
        / "work"
    )

    checkpoint_dir = (
        output_dir
        / "checkpoints"
    )

    data_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    work_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        "=" * 72
    )
    print(
        "EnergyPlus calibration-grid generation"
    )
    print(
        "=" * 72
    )
    print(
        f"Script location:        "
        f"{Path(__file__).resolve()}"
    )
    print(
        f"Synthetic Homes root:   "
        f"{synthetic_homes_root}"
    )
    print(
        f"Synthetic Homes src:    "
        f"{synthetic_homes_src}"
    )
    print(
        f"EnergyPlus IDD:          "
        f"{idd_path}"
    )
    print(
        f"Weather file:            "
        f"{epw_path}"
    )
    print(
        f"Experiment output:       "
        f"{output_dir}"
    )
    print(
        f"Calibration k:           "
        f"{k}"
    )
    print(
        f"Worker processes:        "
        f"{max_workers}"
    )
    print(
        f"Resume existing run:     "
        f"{resume}"
    )
    print(
        f"Keep simulation files:   "
        f"{keep_simulation_files}"
    )
    print(
        "=" * 72
    )
    print()

    baseline = {
        "air_change_rate": 2.0,
        "hvac_heating_cop": 0.8,
        "hvac_cooling_cop": 3.0,
        "window_u_value": 2.0,
        "wall_r_value": 13.0,
        "roof_r_value": 30.0,
        "hvac_system_type": "gas_furnace",
    }

    levels = build_parameter_levels(
        k
    )

    keys = list(
        levels.keys()
    )

    total_combinations = (
        k ** len(keys)
    )

    name_width = max(
        4,
        len(
            str(
                total_combinations
            )
        ),
    )

    # Detect existing successful work BEFORE overwriting experiment metadata.
    existing_results = _completed_job_names(
        data_dir,
        checkpoint_dir,
    )

    if (
        existing_results
        and not resume
    ):
        raise RuntimeError(
            "This output directory already contains completed calibration "
            "jobs.\n\n"
            f"  {output_dir}\n\n"
            "Use --resume to continue the existing experiment, or choose "
            "a different --output-dir."
        )

    calibration_levels_path = (
        data_dir
        / "calibration_levels.json"
    )

    if (
        resume
        and calibration_levels_path.exists()
    ):
        existing_levels = _read_json_if_exists(
            calibration_levels_path,
            {},
        )

        existing_k = (
            existing_levels.get(
                "max_k"
            )
            if isinstance(
                existing_levels,
                dict,
            )
            else None
        )

        if (
            existing_k is not None
            and int(existing_k) != k
        ):
            raise RuntimeError(
                "Cannot resume this output directory with a different "
                "calibration resolution.\n\n"
                f"Existing k: {existing_k}\n"
                f"Requested k: {k}\n"
            )

    meta: dict[
        str,
        dict[str, Any],
    ] = {}

    jobs: list[
        CalibrationJob
    ] = []

    print(
        f"Preparing metadata for "
        f"{total_combinations} calibration points..."
    )

    for (
        i,
        level_indices,
    ) in enumerate(
        product(
            range(k),
            repeat=len(keys),
        ),
        1,
    ):
        values = tuple(
            levels[key][index]
            for key, index
            in zip(
                keys,
                level_indices,
            )
        )

        params = (
            baseline
            | dict(
                zip(
                    keys,
                    values,
                )
            )
        )

        params[
            "air_change_rate"
        ] = baseline[
            "air_change_rate"
        ]

        name = (
            f"cal_{i:0{name_width}d}"
        )

        meta[
            name
        ] = {
            "wall_r_value": params[
                "wall_r_value"
            ],
            "roof_r_value": params[
                "roof_r_value"
            ],
            "hvac_heating_cop": params[
                "hvac_heating_cop"
            ],
            "hvac_cooling_cop": params[
                "hvac_cooling_cop"
            ],
            "air_change_rate": params[
                "air_change_rate"
            ],
            "calibration_k": k,
            "ordinal_levels": {
                key: int(
                    index + 1
                )
                for key, index
                in zip(
                    keys,
                    level_indices,
                )
            },
        }

        jobs.append(
            CalibrationJob(
                name=name,
                params=params,
            )
        )

    calibration_meta_path = (
        data_dir
        / "calibration_meta.json"
    )

    _write_json(
        calibration_meta_path,
        meta,
    )

    _write_json(
        calibration_levels_path,
        {
            "max_k": k,
            "parameter_levels": {
                key: list(values)
                for key, values
                in levels.items()
            },
        },
    )

    print(
        f"Wrote: {calibration_meta_path}"
    )
    print(
        f"Wrote: {calibration_levels_path}"
    )

    expected_names = {
        job.name
        for job
        in jobs
    }

    # Ignore any unrelated names that might somehow exist in the directory.
    existing_results = (
        existing_results
        & expected_names
    )

    if resume:
        pending_jobs = [
            job
            for job in jobs
            if job.name
            not in existing_results
        ]

        print()
        print(
            "Resume mode: found "
            f"{len(existing_results)} completed jobs."
        )
        print(
            "Resume mode: "
            f"{len(pending_jobs)} jobs remain."
        )

    else:
        pending_jobs = jobs

    try:
        (
            newly_successful,
            newly_failed,
        ) = _run_energyplus_batch(
            pending_jobs,
            str(epw_path),
            work_dir,
            checkpoint_dir,
            synthetic_homes_root,
            idd_path,
            max_workers,
            keep_simulation_files,
            total_jobs=total_combinations,
            initial_completed=len(
                existing_results
            ),
        )

    except BaseException:
        print()
        print(
            "Execution was interrupted."
        )
        print(
            "Completed simulations have durable checkpoints."
        )
        print(
            "Aggregating all checkpoints written so far..."
        )

        try:
            _aggregate_results(
                data_dir,
                checkpoint_dir,
            )

        except Exception as aggregation_error:
            print(
                "Warning: aggregation after interruption failed:"
            )
            print(
                f"  {aggregation_error}"
            )

        raise

    successful_jobs, failed_jobs = (
        _aggregate_results(
            data_dir,
            checkpoint_dir,
        )
    )

    completed_names = (
        _completed_job_names(
            data_dir,
            checkpoint_dir,
        )
        & expected_names
    )

    missing_names = (
        expected_names
        - completed_names
    )

    if (
        not missing_names
        and successful_jobs
        >= total_combinations
    ):
        # Once the aggregate files contain every successful job, individual
        # checkpoints are no longer required for resume.
        shutil.rmtree(
            checkpoint_dir,
            ignore_errors=True,
        )

        if not keep_simulation_files:
            shutil.rmtree(
                work_dir,
                ignore_errors=True,
            )

    print()
    print(
        "=" * 72
    )

    if missing_names:
        print(
            "Calibration-grid generation INCOMPLETE."
        )
    else:
        print(
            "Calibration-grid generation complete."
        )

    print(
        "=" * 72
    )
    print(
        f"Calibration points:      "
        f"{total_combinations}"
    )
    print(
        f"Successful points:       "
        f"{len(completed_names)}"
    )
    print(
        f"Remaining/failed points: "
        f"{len(missing_names)}"
    )
    print(
        f"New successes this run:  "
        f"{newly_successful}"
    )
    print(
        f"New failures this run:   "
        f"{newly_failed}"
    )
    print(
        f"Experiment output:       "
        f"{output_dir}"
    )
    print()
    print(
        "Persistent data files:"
    )
    print(
        f"  {calibration_meta_path}"
    )
    print(
        f"  {calibration_levels_path}"
    )
    print(
        f"  {data_dir / 'summary_stats.json'}"
    )
    print(
        f"  {data_dir / 'calibration_runtime.json'}"
    )
    print(
        "=" * 72
    )

    if missing_names:
        print()
        print(
            "Some simulations did not complete successfully."
        )
        print(
            "Rerun the same command with --resume to retry only "
            "the missing/failed calibration points."
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Generate the Synthetic Homes EnergyPlus calibration grid "
            "used for k-sensitivity testing."
        ),
        formatter_class=(
            argparse.RawDescriptionHelpFormatter
        ),
        epilog="""
Examples:

  New k=15 run:

    python generate_calibration_grid.py \\
        --synthetic-homes-root ../synthetic-homes \\
        --epw ../synthetic-homes/weather/KMSP.epw \\
        --output-dir ./calibration_grid \\
        --k 15 \\
        --max-workers 8

  Resume an interrupted run:

    python generate_calibration_grid.py \\
        --synthetic-homes-root ../synthetic-homes \\
        --epw ../synthetic-homes/weather/KMSP.epw \\
        --output-dir ./calibration_grid \\
        --k 15 \\
        --max-workers 8 \\
        --resume

  Rebuild aggregate files from existing checkpoints:

    python generate_calibration_grid.py \\
        --output-dir ./calibration_grid \\
        --aggregate-only
""",
    )

    parser.add_argument(
        "--synthetic-homes-root",
        help=(
            "Path to the root of the cloned Synthetic Homes repository. "
            "Required unless --aggregate-only is used."
        ),
    )

    parser.add_argument(
        "--epw",
        help=(
            "Path to the EnergyPlus weather file. Relative paths are "
            "interpreted relative to the current working directory. "
            "Required unless --aggregate-only is used."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory for experiment outputs. Relative paths are interpreted "
            "relative to the current working directory. If omitted, defaults "
            "to calibration_grid/ next to this script."
        ),
    )

    parser.add_argument(
        "--k",
        type=int,
        default=15,
        help=(
            "Calibration-grid resolution. Default: 15."
        ),
    )

    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help=(
            "Maximum number of concurrent EnergyPlus workers. Default: 1."
        ),
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Continue an existing run. Successfully checkpointed simulations "
            "are skipped."
        ),
    )

    parser.add_argument(
        "--keep-simulation-files",
        action="store_true",
        help=(
            "Keep generated IDFs and raw EnergyPlus outputs after each job. "
            "By default they are deleted after summary statistics have been "
            "safely checkpointed."
        ),
    )

    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help=(
            "Do not load Synthetic Homes or run EnergyPlus. Rebuild "
            "summary_stats.json and calibration_runtime.json from existing "
            "per-job checkpoints."
        ),
    )

    args = parser.parse_args()

    output_dir = _resolve_output_dir(
        args.output_dir
    )

    data_dir = (
        output_dir
        / "data"
    )

    checkpoint_dir = (
        output_dir
        / "checkpoints"
    )

    if args.aggregate_only:
        print()
        print(
            "=" * 72
        )
        print(
            "Aggregate calibration checkpoints"
        )
        print(
            "=" * 72
        )
        print(
            f"Experiment output: {output_dir}"
        )
        print(
            f"Data directory:    {data_dir}"
        )
        print(
            f"Checkpoint dir:    {checkpoint_dir}"
        )
        print(
            "=" * 72
        )

        _aggregate_results(
            data_dir,
            checkpoint_dir,
        )

        return 0

    if not args.synthetic_homes_root:
        parser.error(
            "--synthetic-homes-root is required unless "
            "--aggregate-only is used."
        )

    if not args.epw:
        parser.error(
            "--epw is required unless --aggregate-only is used."
        )

    generate(
        synthetic_homes_root=(
            args.synthetic_homes_root
        ),
        epw=(
            args.epw
        ),
        output_dir=(
            args.output_dir
        ),
        k=(
            args.k
        ),
        max_workers=(
            args.max_workers
        ),
        resume=(
            args.resume
        ),
        keep_simulation_files=(
            args.keep_simulation_files
        ),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )