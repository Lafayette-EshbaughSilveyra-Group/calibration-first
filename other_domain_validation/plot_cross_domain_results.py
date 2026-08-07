#!/usr/bin/env python3
"""
Plot every scorer produced by the cross-domain validation experiments.

Run this script from the directory that contains ``results/``::

    python3 plot_cross_domain_scorers.py

By default, it searches recursively under ``results/`` for:

- ames_housing_observations.csv
- breast_cancer_wisconsin_observations.csv

It writes publication-ready PDF and PNG figures to ``paper_figs/``.

For each experiment, the script produces:

1. A calibrated-score figure containing every calibrated scorer and kappa.
2. A raw-score figure containing every raw scorer and the uncalibrated average.

Regression targets are shown as score-versus-target scatter plots with a
quantile-binned median trend. Binary targets are shown as jittered class-wise
score distributions with median and interquartile-range summaries.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExperimentSpec:
    """Plotting metadata for one validation experiment."""

    filename: str
    slug: str
    title: str
    target_column: str
    target_label: str
    task: str
    class_labels: tuple[str, str] | None = None


EXPERIMENTS: tuple[ExperimentSpec, ...] = (
    ExperimentSpec(
        filename="ames_housing_observations.csv",
        slug="ames_housing",
        title="Ames Housing",
        target_column="SalePrice",
        target_label="Sale price",
        task="regression",
    ),
    ExperimentSpec(
        filename="breast_cancer_wisconsin_observations.csv",
        slug="breast_cancer_wisconsin",
        title="Breast Cancer Wisconsin",
        target_column="malignant",
        target_label="Diagnosis",
        task="classification",
        class_labels=("Benign", "Malignant"),
    ),
)


DISPLAY_NAME_OVERRIDES = {
    "kappa": r"Calibrated fusion ($\kappa$)",
    "uncalibrated_average": "Uncalibrated average",
    "physical_scale": "Physical scale",
    "quality_condition": "Quality and condition",
    "age_modernization": "Age and modernization",
    "garage_capacity": "Garage capacity",
    "size": "Size",
    "shape_irregularity": "Shape irregularity",
    "texture_smoothness": "Texture and smoothness",
    "worst_case": "Worst-case features",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Plot every raw and calibrated scorer from the cross-domain "
            "validation result tables."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help=(
            "Directory beneath which observation CSV files are located. "
            "The default is ./results."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper_figs"),
        help="Figure output directory. The default is ./paper_figs.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("pdf", "png", "svg"),
        default=("pdf", "png"),
        help="Output formats. The default is PDF and PNG.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Raster output resolution. The default is 300 DPI.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=20,
        help=(
            "Number of quantile bins for regression trend lines. "
            "The default is 20."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed used only for classification-point jitter.",
    )
    return parser.parse_args()


def configure_matplotlib() -> None:
    """Apply restrained, publication-oriented Matplotlib defaults."""
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
            "lines.linewidth": 1.6,
        }
    )


def find_unique_file(root: Path, filename: str) -> Path:
    """Find exactly one named file recursively beneath root."""
    if not root.exists():
        raise FileNotFoundError(
            f"Results directory does not exist: {root.resolve()}"
        )

    matches = sorted(path for path in root.rglob(filename) if path.is_file())

    if not matches:
        raise FileNotFoundError(
            f"Could not find {filename!r} beneath {root.resolve()}."
        )

    if len(matches) > 1:
        formatted = "\n".join(f"  - {path}" for path in matches)
        raise RuntimeError(
            f"Found multiple copies of {filename!r}. Pass a narrower "
            f"--results-dir. Matches:\n{formatted}"
        )

    return matches[0]


def scorer_name_from_column(column: str, prefix: str) -> str:
    """Convert a persisted scorer column into its short scorer name."""
    if not column.startswith(prefix):
        raise ValueError(f"Column {column!r} does not start with {prefix!r}.")
    return column[len(prefix) :]


def display_name(name: str) -> str:
    """Return a paper-friendly label for a scorer or aggregate."""
    return DISPLAY_NAME_OVERRIDES.get(
        name,
        name.replace("_", " ").strip().title(),
    )


def discover_score_columns(
    frame: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """Discover and validate raw and calibrated scorer columns."""
    raw_columns = [
        column
        for column in frame.columns
        if column.startswith("raw_scorer_")
    ]
    calibrated_columns = [
        column
        for column in frame.columns
        if column.startswith("calibrated_scorer_")
    ]

    if not raw_columns:
        raise ValueError("No columns beginning with 'raw_scorer_' were found.")
    if not calibrated_columns:
        raise ValueError(
            "No columns beginning with 'calibrated_scorer_' were found."
        )

    raw_names = {
        scorer_name_from_column(column, "raw_scorer_")
        for column in raw_columns
    }
    calibrated_names = {
        scorer_name_from_column(column, "calibrated_scorer_")
        for column in calibrated_columns
    }

    if raw_names != calibrated_names:
        missing_raw = sorted(calibrated_names - raw_names)
        missing_calibrated = sorted(raw_names - calibrated_names)
        raise ValueError(
            "Raw and calibrated scorer columns do not match. "
            f"Missing raw scorers: {missing_raw}; "
            f"missing calibrated scorers: {missing_calibrated}."
        )

    # Preserve the source-table order rather than alphabetizing scorer names.
    calibrated_by_name = {
        scorer_name_from_column(column, "calibrated_scorer_"): column
        for column in calibrated_columns
    }
    calibrated_columns = [
        calibrated_by_name[
            scorer_name_from_column(column, "raw_scorer_")
        ]
        for column in raw_columns
    ]

    return raw_columns, calibrated_columns


def validate_experiment_frame(
    frame: pd.DataFrame,
    spec: ExperimentSpec,
) -> tuple[list[str], list[str]]:
    """Validate required target, aggregate, and scorer columns."""
    required = {
        spec.target_column,
        "kappa",
        "uncalibrated_average",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"{spec.filename} is missing required columns: {missing}."
        )

    raw_columns, calibrated_columns = discover_score_columns(frame)

    relevant_columns = [
        spec.target_column,
        "kappa",
        "uncalibrated_average",
        *raw_columns,
        *calibrated_columns,
    ]
    missing_counts = frame[relevant_columns].isna().sum()
    nonzero_missing = missing_counts[missing_counts > 0]
    if not nonzero_missing.empty:
        details = ", ".join(
            f"{column}={count}"
            for column, count in nonzero_missing.items()
        )
        raise ValueError(
            f"{spec.filename} contains missing plotting values: {details}."
        )

    if spec.task == "classification":
        classes = set(pd.unique(frame[spec.target_column]))
        if not classes.issubset({0, 1}):
            raise ValueError(
                f"Expected a binary 0/1 target in {spec.target_column!r}; "
                f"found {sorted(classes)}."
            )

    return raw_columns, calibrated_columns


def subplot_grid(n_panels: int, max_columns: int = 3) -> tuple[int, int]:
    """Choose a compact subplot grid."""
    columns = min(max_columns, n_panels)
    rows = math.ceil(n_panels / columns)
    return rows, columns


def quantile_binned_medians(
    target: pd.Series,
    score: pd.Series,
    bins: int,
) -> pd.DataFrame:
    """Compute median target and score within approximately equal-count bins."""
    data = pd.DataFrame(
        {
            "target": pd.to_numeric(target, errors="coerce"),
            "score": pd.to_numeric(score, errors="coerce"),
        }
    ).dropna()

    if data.empty:
        return data

    unique_targets = int(data["target"].nunique())
    effective_bins = max(2, min(bins, unique_targets, len(data)))

    try:
        data["bin"] = pd.qcut(
            data["target"],
            q=effective_bins,
            duplicates="drop",
        )
    except ValueError:
        return pd.DataFrame(columns=["target", "score"])

    grouped = (
        data.groupby("bin", observed=True)[["target", "score"]]
        .median()
        .reset_index(drop=True)
        .sort_values("target")
    )
    return grouped



def compact_currency(value: float, _position: int) -> str:
    """Format large currency values compactly for figure axes."""
    absolute = abs(value)
    if absolute >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    if absolute >= 1_000:
        return f"${value / 1_000:.0f}k"
    return f"${value:.0f}"

def plot_regression_panel(
    axis: plt.Axes,
    target: pd.Series,
    score: pd.Series,
    target_label: str,
    score_label: str,
    bins: int,
) -> None:
    """Plot every score value against a continuous target."""
    x = pd.to_numeric(target, errors="coerce")
    y = pd.to_numeric(score, errors="coerce")
    valid = x.notna() & y.notna()

    axis.scatter(
        x[valid],
        y[valid],
        s=10,
        alpha=0.24,
        linewidths=0,
        rasterized=True,
    )

    trend = quantile_binned_medians(
        x[valid],
        y[valid],
        bins=bins,
    )
    if not trend.empty:
        axis.plot(
            trend["target"],
            trend["score"],
            marker="o",
            markersize=3,
            label="Binned median",
        )

    axis.set_title(score_label)
    axis.set_xlabel(target_label)
    axis.set_ylabel("Score")
    if target_label.lower() == "sale price":
        axis.xaxis.set_major_formatter(FuncFormatter(compact_currency))


def plot_classification_panel(
    axis: plt.Axes,
    target: pd.Series,
    score: pd.Series,
    score_label: str,
    class_labels: tuple[str, str],
    rng: np.random.Generator,
) -> None:
    """Plot every score value by binary target class."""
    target_values = pd.to_numeric(target, errors="coerce")
    score_values = pd.to_numeric(score, errors="coerce")

    for class_value in (0, 1):
        class_scores = score_values[target_values == class_value].dropna()
        jitter = rng.normal(
            loc=0.0,
            scale=0.055,
            size=len(class_scores),
        )
        axis.scatter(
            np.full(len(class_scores), class_value) + jitter,
            class_scores,
            s=11,
            alpha=0.28,
            linewidths=0,
            rasterized=True,
        )

        if len(class_scores):
            q1, median, q3 = np.quantile(
                class_scores.to_numpy(),
                [0.25, 0.50, 0.75],
            )
            axis.errorbar(
                class_value,
                median,
                yerr=np.array([[median - q1], [q3 - median]]),
                fmt="o",
                markersize=5,
                capsize=4,
                linewidth=1.5,
            )

    axis.set_title(score_label)
    axis.set_xticks([0, 1], labels=class_labels)
    axis.set_xlabel("Diagnosis")
    axis.set_ylabel("Score")
    axis.set_xlim(-0.35, 1.35)


def plot_score_collection(
    frame: pd.DataFrame,
    spec: ExperimentSpec,
    columns: Sequence[str],
    labels: Sequence[str],
    figure_title: str,
    output_stem: Path,
    formats: Iterable[str],
    dpi: int,
    bins: int,
    seed: int,
) -> None:
    """Plot one panel per score column and save the completed figure."""
    if len(columns) != len(labels):
        raise ValueError("columns and labels must have the same length.")

    rows, columns_per_row = subplot_grid(len(columns))
    figure, axes = plt.subplots(
        rows,
        columns_per_row,
        figsize=(4.0 * columns_per_row, 3.15 * rows),
        squeeze=False,
    )
    flattened_axes = list(axes.flat)
    rng = np.random.default_rng(seed)

    for axis, column, label in zip(flattened_axes, columns, labels):
        if spec.task == "regression":
            plot_regression_panel(
                axis=axis,
                target=frame[spec.target_column],
                score=frame[column],
                target_label=spec.target_label,
                score_label=label,
                bins=bins,
            )
        elif spec.task == "classification":
            if spec.class_labels is None:
                raise ValueError(
                    f"No class labels were configured for {spec.title}."
                )
            plot_classification_panel(
                axis=axis,
                target=frame[spec.target_column],
                score=frame[column],
                score_label=label,
                class_labels=spec.class_labels,
                rng=rng,
            )
        else:
            raise ValueError(f"Unsupported task: {spec.task!r}.")

    for unused_axis in flattened_axes[len(columns) :]:
        unused_axis.remove()

    figure.suptitle(figure_title, fontsize=12, y=1.01)
    figure.tight_layout()

    for extension in formats:
        destination = output_stem.with_suffix(f".{extension}")
        figure.savefig(destination, dpi=dpi)
        print(f"Wrote {destination}")

    plt.close(figure)


def plot_experiment(
    csv_path: Path,
    spec: ExperimentSpec,
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
    bins: int,
    seed: int,
) -> None:
    """Create calibrated and raw scorer figures for one experiment."""
    frame = pd.read_csv(csv_path)
    raw_columns, calibrated_columns = validate_experiment_frame(frame, spec)

    calibrated_labels = [
        display_name(
            scorer_name_from_column(column, "calibrated_scorer_")
        )
        for column in calibrated_columns
    ]
    calibrated_plot_columns = [*calibrated_columns, "kappa"]
    calibrated_plot_labels = [
        *calibrated_labels,
        display_name("kappa"),
    ]

    plot_score_collection(
        frame=frame,
        spec=spec,
        columns=calibrated_plot_columns,
        labels=calibrated_plot_labels,
        figure_title=f"{spec.title}: calibrated scorer values",
        output_stem=(
            output_dir / f"{spec.slug}_calibrated_scorers"
        ),
        formats=formats,
        dpi=dpi,
        bins=bins,
        seed=seed,
    )

    raw_labels = [
        display_name(scorer_name_from_column(column, "raw_scorer_"))
        for column in raw_columns
    ]
    raw_plot_columns = [*raw_columns, "uncalibrated_average"]
    raw_plot_labels = [
        *raw_labels,
        display_name("uncalibrated_average"),
    ]

    plot_score_collection(
        frame=frame,
        spec=spec,
        columns=raw_plot_columns,
        labels=raw_plot_labels,
        figure_title=f"{spec.title}: raw scorer values",
        output_stem=output_dir / f"{spec.slug}_raw_scorers",
        formats=formats,
        dpi=dpi,
        bins=bins,
        seed=seed,
    )


def main() -> None:
    """Locate result tables and generate all paper figures."""
    args = parse_args()
    configure_matplotlib()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for spec in EXPERIMENTS:
        csv_path = find_unique_file(args.results_dir, spec.filename)
        print(f"Reading {csv_path}")
        plot_experiment(
            csv_path=csv_path,
            spec=spec,
            output_dir=args.output_dir,
            formats=args.formats,
            dpi=args.dpi,
            bins=args.bins,
            seed=args.seed,
        )

    print(f"\nAll figures written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
