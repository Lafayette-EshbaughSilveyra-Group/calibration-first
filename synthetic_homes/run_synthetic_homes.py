#!/usr/bin/env python3
"""Synthetic Homes calibration and proposition experiments.

Each home contains one whole-home inspection note. The note is evaluated once
by a text scorer that returns two concept-specific judgments:

1. text_hvac
2. text_insulation

The structured representation is evaluated by two additional scorers:

3. structured_hvac: combines heating and cooling COP
4. structured_insulation: combines wall and roof R-values

All four scorer outputs are calibrated independently over the same 5^4 = 625
reference space.

Larger scores always indicate greater retrofit need.
"""

from __future__ import annotations

from dotenv import load_dotenv
import argparse
import csv
import importlib
import importlib.util
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import combinations, product
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Final


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FEATURE_LEVELS: Final[dict[str, tuple[float, ...]]] = {
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
        0.7,
        0.8,
        0.9,
        0.95,
        1.0,
    ),
    "hvac_cooling_cop": (
        1.0,
        2.0,
        3.0,
        3.5,
        4.0,
    ),
}

CHANNELS: Final[tuple[str, ...]] = (
    "text_hvac",
    "text_insulation",
    "structured_hvac",
    "structured_insulation",
)

CONCEPT_CHANNELS: Final[dict[str, tuple[str, str]]] = {
    "hvac": (
        "text_hvac",
        "structured_hvac",
    ),
    "insulation": (
        "text_insulation",
        "structured_insulation",
    ),
}

RAW_COLUMN: Final[dict[str, str]] = {
    channel: f"{channel}_raw"
    for channel in CHANNELS
}

Z_COLUMN: Final[dict[str, str]] = {
    channel: f"{channel}_z"
    for channel in CHANNELS
}

CONCEPT_KAPPA_COLUMN: Final[dict[str, str]] = {
    "hvac": "kappa_hvac",
    "insulation": "kappa_insulation",
}

DEFAULT_OUTPUT_DIR: Final[Path] = Path(
    "results/synthetic_homes"
)

# The scorer may return:
#
# - TextScores;
# - {"hvac": ..., "insulation": ...};
# - {"hvac_retrofit_need": ..., "insulation_retrofit_need": ...};
# - a two-item sequence: (hvac_score, insulation_score).
TextScorer = Callable[
    [str, Mapping[str, Any]],
    Any,
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CalibrationStats:
    mean: float
    std: float
    n: int

    def __post_init__(self) -> None:
        if self.n < 2:
            raise ValueError(
                "Calibration requires at least two outputs."
            )

        if not math.isfinite(self.mean):
            raise ValueError(
                "Calibration mean must be finite."
            )

        if not math.isfinite(self.std) or self.std <= 0:
            raise ValueError(
                "Calibration standard deviation must be greater than zero."
            )

    def to_dict(
        self,
    ) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class NoteTemplates:
    """Components used to construct one whole-home inspection note."""

    hvac: dict[int, str]
    insulation: dict[int, str]


@dataclass(frozen=True)
class TextScores:
    """Two judgments obtained from one whole-home inspection note."""

    hvac: float
    insulation: float


# ---------------------------------------------------------------------------
# Framework operations
# ---------------------------------------------------------------------------

def fit_calibration(
    values: Sequence[float],
) -> CalibrationStats:
    numeric_values = [
        float(value)
        for value in values
    ]

    if len(numeric_values) < 2:
        raise ValueError(
            "Calibration requires at least two outputs."
        )

    if not all(
        math.isfinite(value)
        for value in numeric_values
    ):
        raise ValueError(
            "Calibration outputs must all be finite."
        )

    return CalibrationStats(
        mean=fmean(numeric_values),
        std=pstdev(numeric_values),
        n=len(numeric_values),
    )


def standardize(
    value: float,
    stats: CalibrationStats,
) -> float:
    value = float(value)

    if not math.isfinite(value):
        raise ValueError(
            "Scorer output must be finite."
        )

    return (
        value - stats.mean
    ) / stats.std


def fuse(
    calibrated_scores: Mapping[str, float],
) -> float:
    if not calibrated_scores:
        raise ValueError(
            "At least one calibrated score is required."
        )

    values = [
        float(value)
        for value in calibrated_scores.values()
    ]

    if not all(
        math.isfinite(value)
        for value in values
    ):
        raise ValueError(
            "Calibrated scores must all be finite."
        )

    return fmean(values)


def exact_relation(
    left: float,
    right: float,
) -> int:
    """Return -1, 0, or 1 for left < right, left == right, or left > right."""

    if left < right:
        return -1

    if left > right:
        return 1

    return 0


def tolerant_sign(
    value: float,
    atol: float = 1e-12,
) -> int:
    if value > atol:
        return 1

    if value < -atol:
        return -1

    return 0


def verify_proposition_1(
    raw_scores: Sequence[float],
    calibrated_scores: Sequence[float],
) -> dict[str, int | bool]:
    """Verify that standardization preserves every pairwise ordering."""

    if len(raw_scores) != len(calibrated_scores):
        raise ValueError(
            "Raw and calibrated sequences must align."
        )

    comparisons = 0
    failures = 0

    for left_index, right_index in combinations(
        range(len(raw_scores)),
        2,
    ):
        comparisons += 1

        raw_relation = exact_relation(
            float(raw_scores[left_index]),
            float(raw_scores[right_index]),
        )

        calibrated_relation = exact_relation(
            float(calibrated_scores[left_index]),
            float(calibrated_scores[right_index]),
        )

        if raw_relation != calibrated_relation:
            failures += 1

    return {
        "comparisons": comparisons,
        "failures": failures,
        "passed": failures == 0,
    }


def verify_proposition_2(
    calibrated_a: Mapping[str, float],
    calibrated_b: Mapping[str, float],
) -> dict[str, Any]:
    """Verify fused ordering against aggregate calibrated support."""

    if set(calibrated_a) != set(calibrated_b):
        raise ValueError(
            "Both instances must contain the same channels."
        )

    deltas = {
        channel: (
            float(calibrated_b[channel])
            - float(calibrated_a[channel])
        )
        for channel in calibrated_a
    }

    positive_support = sum(
        delta
        for delta in deltas.values()
        if delta > 0
    )

    negative_support = sum(
        abs(delta)
        for delta in deltas.values()
        if delta < 0
    )

    support_balance = (
        positive_support
        - negative_support
    )

    fused_a = fuse(calibrated_a)
    fused_b = fuse(calibrated_b)
    fused_delta = fused_b - fused_a

    return {
        "deltas": deltas,
        "positive_support": positive_support,
        "negative_support": negative_support,
        "support_balance": support_balance,
        "fused_a": fused_a,
        "fused_b": fused_b,
        "fused_delta": fused_delta,
        "passed": (
            tolerant_sign(fused_delta)
            == tolerant_sign(support_balance)
        ),
    }


def monotone_decreasing(
    values: Sequence[float],
    atol: float = 1e-12,
) -> bool:
    """Check that retrofit need does not rise as efficiency rises."""

    return all(
        float(right)
        <= float(left) + atol
        for left, right in zip(
            values,
            values[1:],
        )
    )


# ---------------------------------------------------------------------------
# Whole-home note construction
# ---------------------------------------------------------------------------

def load_note_templates(
    path: Path,
) -> NoteTemplates:
    """Load five HVAC and five insulation note components.

    Expected JSON:

    {
      "hvac": {
        "1": "...",
        "2": "...",
        "3": "...",
        "4": "...",
        "5": "..."
      },
      "insulation": {
        "1": "...",
        "2": "...",
        "3": "...",
        "4": "...",
        "5": "..."
      }
    }

    One HVAC component and one insulation component are joined into one
    whole-home inspection note.
    """

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        payload = json.load(handle)

    if not isinstance(payload, Mapping):
        raise ValueError(
            "The note-template file must contain a JSON object."
        )

    try:
        hvac = {
            int(level): str(text).strip()
            for level, text in payload["hvac"].items()
        }

        insulation = {
            int(level): str(text).strip()
            for level, text in payload["insulation"].items()
        }

    except (
        KeyError,
        TypeError,
        ValueError,
        AttributeError,
    ) as error:
        raise ValueError(
            "Notes JSON must contain 'hvac' and 'insulation' objects."
        ) from error

    expected_levels = {
        1,
        2,
        3,
        4,
        5,
    }

    for name, templates in {
        "hvac": hvac,
        "insulation": insulation,
    }.items():
        if set(templates) != expected_levels:
            raise ValueError(
                f"{name} templates must contain exactly levels 1-5. "
                f"Received: {sorted(templates)}"
            )

        empty_levels = [
            level
            for level, text in templates.items()
            if not text
        ]

        if empty_levels:
            raise ValueError(
                f"{name} templates contain empty text at levels "
                f"{empty_levels}."
            )

    return NoteTemplates(
        hvac=hvac,
        insulation=insulation,
    )


def collapse_levels(
    first_level: int,
    second_level: int,
) -> int:
    """Collapse two subfeature levels into one concept-description level."""

    for level in (
        first_level,
        second_level,
    ):
        if level not in {
            1,
            2,
            3,
            4,
            5,
        }:
            raise ValueError(
                f"Level must be in 1-5; got {level}."
            )

    average = (
        first_level
        + second_level
    ) / 2.0

    # Half-up rounding: 2.5 becomes 3.
    return int(
        math.floor(
            average + 0.5
        )
    )


def compose_inspection_note(
    notes: NoteTemplates,
    *,
    hvac_level: int,
    insulation_level: int,
) -> str:
    """Return one complete inspection note for one home."""

    return (
        f"{notes.insulation[insulation_level]} "
        f"{notes.hvac[hvac_level]}"
    ).strip()


# ---------------------------------------------------------------------------
# Home construction and calibration space
# ---------------------------------------------------------------------------

def make_home(
    notes: NoteTemplates,
    *,
    wall_level: int,
    roof_level: int,
    heating_level: int,
    cooling_level: int,
    record_id: str,
    hvac_note_level: int | None = None,
    insulation_note_level: int | None = None,
) -> dict[str, Any]:
    """Construct one home with one whole-home inspection note.

    Normally, each note-component level is derived from its corresponding
    structured subfeatures.

    Experiments may override either note level to construct agreement or
    conflict between textual and structured evidence.
    """

    for level in (
        wall_level,
        roof_level,
        heating_level,
        cooling_level,
    ):
        if level not in {
            1,
            2,
            3,
            4,
            5,
        }:
            raise ValueError(
                f"Structured levels must be in 1-5; got {level}."
            )

    if hvac_note_level is None:
        hvac_note_level = collapse_levels(
            heating_level,
            cooling_level,
        )

    if insulation_note_level is None:
        insulation_note_level = collapse_levels(
            wall_level,
            roof_level,
        )

    for level in (
        hvac_note_level,
        insulation_note_level,
    ):
        if level not in {
            1,
            2,
            3,
            4,
            5,
        }:
            raise ValueError(
                f"Note levels must be in 1-5; got {level}."
            )

    inspection_note = compose_inspection_note(
        notes,
        hvac_level=hvac_note_level,
        insulation_level=insulation_note_level,
    )

    return {
        "record_id": record_id,

        "wall_r_value_level": wall_level,
        "wall_r_value": FEATURE_LEVELS[
            "wall_r_value"
        ][wall_level - 1],

        "roof_r_value_level": roof_level,
        "roof_r_value": FEATURE_LEVELS[
            "roof_r_value"
        ][roof_level - 1],

        "hvac_heating_cop_level": heating_level,
        "hvac_heating_cop": FEATURE_LEVELS[
            "hvac_heating_cop"
        ][heating_level - 1],

        "hvac_cooling_cop_level": cooling_level,
        "hvac_cooling_cop": FEATURE_LEVELS[
            "hvac_cooling_cop"
        ][cooling_level - 1],

        "hvac_note_level": hvac_note_level,
        "insulation_note_level": insulation_note_level,

        # There is exactly one note for the whole home.
        "inspection_note": inspection_note,
    }


def build_calibration_space(
    notes: NoteTemplates,
) -> list[dict[str, Any]]:
    """Build all 5^4 = 625 structured reference homes."""

    records: list[dict[str, Any]] = []

    for index, (
        wall_level,
        roof_level,
        heating_level,
        cooling_level,
    ) in enumerate(
        product(
            range(1, 6),
            repeat=4,
        ),
        start=1,
    ):
        records.append(
            make_home(
                notes,
                wall_level=wall_level,
                roof_level=roof_level,
                heating_level=heating_level,
                cooling_level=cooling_level,
                record_id=f"shc_{index:04d}",
            )
        )

    if len(records) != 625:
        raise RuntimeError(
            f"Expected 625 calibration records; got {len(records)}."
        )

    unique_points = {
        (
            record["wall_r_value_level"],
            record["roof_r_value_level"],
            record["hvac_heating_cop_level"],
            record["hvac_cooling_cop_level"],
        )
        for record in records
    }

    if len(unique_points) != 625:
        raise RuntimeError(
            "Calibration space contains duplicate structured points."
        )

    return records


# ---------------------------------------------------------------------------
# Structured evidence scorers
# ---------------------------------------------------------------------------

def efficiency_level(
    value: float,
    feature: str,
) -> float:
    """Interpolate a physical value onto its 1-5 efficiency scale."""

    anchors = FEATURE_LEVELS[feature]
    value = float(value)

    if value <= anchors[0]:
        return 1.0

    if value >= anchors[-1]:
        return 5.0

    for lower_level, (
        lower,
        upper,
    ) in enumerate(
        zip(
            anchors,
            anchors[1:],
        ),
        start=1,
    ):
        if lower <= value <= upper:
            fraction = (
                value - lower
            ) / (
                upper - lower
            )

            return (
                float(lower_level)
                + fraction
            )

    raise RuntimeError(
        f"Could not map {feature}={value} to an efficiency level."
    )


def retrofit_need(
    value: float,
    feature: str,
) -> float:
    """Convert increasing efficiency into decreasing retrofit need."""

    return 6.0 - efficiency_level(
        value,
        feature,
    )


def score_structured_hvac(
    record: Mapping[str, Any],
) -> float:
    """One scorer over heating COP and cooling COP."""

    heating_need = retrofit_need(
        float(
            record["hvac_heating_cop"]
        ),
        "hvac_heating_cop",
    )

    cooling_need = retrofit_need(
        float(
            record["hvac_cooling_cop"]
        ),
        "hvac_cooling_cop",
    )

    # Do not round. Both subfeatures retain equal influence.
    return fmean(
        (
            heating_need,
            cooling_need,
        )
    )


def score_structured_insulation(
    record: Mapping[str, Any],
) -> float:
    """One scorer over wall R-value and roof R-value."""

    wall_need = retrofit_need(
        float(
            record["wall_r_value"]
        ),
        "wall_r_value",
    )

    roof_need = retrofit_need(
        float(
            record["roof_r_value"]
        ),
        "roof_r_value",
    )

    # Do not round. Both subfeatures retain equal influence.
    return fmean(
        (
            wall_need,
            roof_need,
        )
    )


# ---------------------------------------------------------------------------
# Text scorer loading and caching
# ---------------------------------------------------------------------------

def demo_text_scorer(
    note: str,
    record: Mapping[str, Any],
) -> TextScores:
    """Mechanical smoke test only; do not report these results."""

    del note

    hvac_need = (
        5.0
        - float(
            record["hvac_note_level"]
        )
    ) / 4.0

    insulation_need = (
        5.0
        - float(
            record["insulation_note_level"]
        )
    ) / 4.0

    return TextScores(
        hvac=hvac_need,
        insulation=insulation_need,
    )


def coerce_text_scores(
    result: Any,
) -> TextScores:
    """Accept TextScores, a mapping, or a two-item sequence."""

    if isinstance(
        result,
        TextScores,
    ):
        scores = result

    elif isinstance(
        result,
        Mapping,
    ):
        try:
            hvac = (
                result["hvac"]
                if "hvac" in result
                else result[
                    "hvac_retrofit_need"
                ]
            )

            insulation = (
                result["insulation"]
                if "insulation" in result
                else result[
                    "insulation_retrofit_need"
                ]
            )

            scores = TextScores(
                hvac=float(hvac),
                insulation=float(insulation),
            )

        except (
            KeyError,
            TypeError,
            ValueError,
        ) as error:
            raise ValueError(
                "Text-score mappings must contain either "
                "('hvac', 'insulation') or "
                "('hvac_retrofit_need', "
                "'insulation_retrofit_need')."
            ) from error

    elif (
        isinstance(
            result,
            Sequence,
        )
        and not isinstance(
            result,
            (
                str,
                bytes,
                bytearray,
            ),
        )
        and len(result) == 2
    ):
        try:
            scores = TextScores(
                hvac=float(
                    result[0]
                ),
                insulation=float(
                    result[1]
                ),
            )

        except (
            TypeError,
            ValueError,
        ) as error:
            raise ValueError(
                "Two-item text-score sequences must contain numbers."
            ) from error

    else:
        raise TypeError(
            "The text scorer must return TextScores, a supported mapping, "
            "or a two-item numeric sequence."
        )

    if not math.isfinite(
        scores.hvac
    ):
        raise ValueError(
            "The HVAC text score must be finite."
        )

    if not math.isfinite(
        scores.insulation
    ):
        raise ValueError(
            "The insulation text score must be finite."
        )

    return scores


def load_text_scorer(
    specification: str,
) -> TextScorer:
    """Load a text scorer from a Python file or importable module.

    Accepted formats:

        demo
        text_score_adapter:score_text
        ./text_score_adapter.py:score_text
        /absolute/path/text_score_adapter.py:score_text
    """

    if specification == "demo":
        return demo_text_scorer

    location, separator, function_name = specification.rpartition(":")

    if not separator or not location or not function_name:
        raise ValueError(
            "Text scorer must be supplied as "
            "'file.py:function', 'module:function', or 'demo'."
        )

    file_path = Path(location)

    if file_path.suffix == ".py" or file_path.exists():
        file_path = file_path.resolve()

        if not file_path.is_file():
            raise FileNotFoundError(
                f"Text scorer file does not exist: {file_path}"
            )

        module_name = (
            f"_text_scorer_"
            f"{abs(hash(str(file_path)))}"
        )

        spec = importlib.util.spec_from_file_location(
            module_name,
            file_path,
        )

        if spec is None or spec.loader is None:
            raise ImportError(
                f"Could not load Python file: {file_path}"
            )

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

    else:
        module = importlib.import_module(
            location
        )

    try:
        function = getattr(
            module,
            function_name,
        )
    except AttributeError as error:
        raise AttributeError(
            f"Could not find function {function_name!r} "
            f"in {location!r}."
        ) from error

    if not callable(function):
        raise TypeError(
            f"{function_name!r} in {location!r} is not callable."
        )

    return function


class CachedTextScorer:
    """Call the text scorer once for each distinct whole-home note."""

    def __init__(
        self,
        scorer: TextScorer,
    ) -> None:
        self.scorer = scorer
        self.cache: dict[
            str,
            TextScores,
        ] = {}

    def __call__(
        self,
        note: str,
        record: Mapping[str, Any],
    ) -> TextScores:
        if note not in self.cache:
            self.cache[note] = (
                coerce_text_scores(
                    self.scorer(
                        note,
                        record,
                    )
                )
            )

        return self.cache[note]


# ---------------------------------------------------------------------------
# Raw scoring, calibration, and fusion
# ---------------------------------------------------------------------------

def add_raw_scores(
    record: Mapping[str, Any],
    text_scorer: CachedTextScorer,
) -> dict[str, Any]:
    result = dict(record)

    # One call on the entire note; two judgments returned.
    text_scores = text_scorer(
        str(
            result["inspection_note"]
        ),
        result,
    )

    result[
        "text_hvac_raw"
    ] = text_scores.hvac

    result[
        "text_insulation_raw"
    ] = text_scores.insulation

    result[
        "structured_hvac_raw"
    ] = score_structured_hvac(
        result
    )

    result[
        "structured_insulation_raw"
    ] = score_structured_insulation(
        result
    )

    return result


def fit_channel_statistics(
    records: Sequence[
        Mapping[str, Any]
    ],
) -> dict[
    str,
    CalibrationStats,
]:
    return {
        channel: fit_calibration(
            [
                float(
                    record[
                        RAW_COLUMN[channel]
                    ]
                )
                for record in records
            ]
        )
        for channel in CHANNELS
    }


def calibrate_record(
    record: Mapping[str, Any],
    statistics: Mapping[
        str,
        CalibrationStats,
    ],
) -> dict[str, Any]:
    result = dict(record)
    calibrated: dict[
        str,
        float,
    ] = {}

    for channel in CHANNELS:
        value = standardize(
            float(
                result[
                    RAW_COLUMN[channel]
                ]
            ),
            statistics[channel],
        )

        result[
            Z_COLUMN[channel]
        ] = value

        calibrated[
            channel
        ] = value

    result[
        "kappa_hvac"
    ] = fuse(
        {
            channel: calibrated[channel]
            for channel in CONCEPT_CHANNELS[
                "hvac"
            ]
        }
    )

    result[
        "kappa_insulation"
    ] = fuse(
        {
            channel: calibrated[channel]
            for channel in CONCEPT_CHANNELS[
                "insulation"
            ]
        }
    )

    result[
        "kappa"
    ] = fuse(
        calibrated
    )

    return result


def score_case(
    record: Mapping[str, Any],
    text_scorer: CachedTextScorer,
    statistics: Mapping[
        str,
        CalibrationStats,
    ],
) -> dict[str, Any]:
    return calibrate_record(
        add_raw_scores(
            record,
            text_scorer,
        ),
        statistics,
    )


def calibrated_channel_scores(
    record: Mapping[str, Any],
    concept: str,
) -> dict[str, float]:
    try:
        channels = CONCEPT_CHANNELS[
            concept
        ]

    except KeyError as error:
        raise ValueError(
            f"Unknown concept: {concept}"
        ) from error

    return {
        channel: float(
            record[
                Z_COLUMN[channel]
            ]
        )
        for channel in channels
    }


# ---------------------------------------------------------------------------
# Proposition 1: full-space check
# ---------------------------------------------------------------------------

def check_proposition_1_over_calibration_space(
    calibration_records: Sequence[
        Mapping[str, Any]
    ],
) -> dict[
    str,
    dict[str, int | bool],
]:
    """Check all calibration-space pairs for all four scorers."""

    return {
        channel: verify_proposition_1(
            raw_scores=[
                float(
                    record[
                        RAW_COLUMN[channel]
                    ]
                )
                for record in calibration_records
            ],
            calibrated_scores=[
                float(
                    record[
                        Z_COLUMN[channel]
                    ]
                )
                for record in calibration_records
            ],
        )
        for channel in CHANNELS
    }


# ---------------------------------------------------------------------------
# Ablation experiment
# ---------------------------------------------------------------------------

def make_ablation_case(
    notes: NoteTemplates,
    *,
    varied_channel: str,
    level: int,
) -> dict[str, Any]:
    """Vary one evidence channel while all others remain at level 3."""

    arguments: dict[
        str,
        Any,
    ] = {
        "wall_level": 3,
        "roof_level": 3,
        "heating_level": 3,
        "cooling_level": 3,
        "hvac_note_level": 3,
        "insulation_note_level": 3,
        "record_id": (
            f"ablation_{varied_channel}_{level}"
        ),
    }

    if varied_channel == "text_hvac":
        arguments[
            "hvac_note_level"
        ] = level

    elif varied_channel == "text_insulation":
        arguments[
            "insulation_note_level"
        ] = level

    elif varied_channel == "structured_hvac":
        arguments[
            "heating_level"
        ] = level

        arguments[
            "cooling_level"
        ] = level

    elif varied_channel == "structured_insulation":
        arguments[
            "wall_level"
        ] = level

        arguments[
            "roof_level"
        ] = level

    else:
        raise ValueError(
            f"Unknown ablation channel: {varied_channel}"
        )

    return make_home(
        notes,
        **arguments,
    )


def run_ablation(
    notes: NoteTemplates,
    text_scorer: CachedTextScorer,
    statistics: Mapping[
        str,
        CalibrationStats,
    ],
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
]:
    rows: list[
        dict[str, Any]
    ] = []

    checks: dict[
        str,
        Any,
    ] = {}

    ablations = (
        (
            "text_hvac",
            "text_hvac",
            "kappa_hvac",
        ),
        (
            "text_insulation",
            "text_insulation",
            "kappa_insulation",
        ),
        (
            "structured_hvac",
            "structured_hvac",
            "kappa_hvac",
        ),
        (
            "structured_insulation",
            "structured_insulation",
            "kappa_insulation",
        ),
    )

    for (
        varied_channel,
        scorer_channel,
        fused_column,
    ) in ablations:
        group: list[
            dict[str, Any]
        ] = []

        for level in range(
            1,
            6,
        ):
            row = score_case(
                make_ablation_case(
                    notes,
                    varied_channel=varied_channel,
                    level=level,
                ),
                text_scorer,
                statistics,
            )

            row.update(
                {
                    "experiment": "ablation",
                    "varied_channel": varied_channel,
                    "efficiency_level": level,
                }
            )

            rows.append(
                row
            )

            group.append(
                row
            )

        raw_values = [
            float(
                row[
                    RAW_COLUMN[
                        scorer_channel
                    ]
                ]
            )
            for row in group
        ]

        calibrated_values = [
            float(
                row[
                    Z_COLUMN[
                        scorer_channel
                    ]
                ]
            )
            for row in group
        ]

        fused_values = [
            float(
                row[
                    fused_column
                ]
            )
            for row in group
        ]

        checks[
            varied_channel
        ] = {
            "scorer_channel": scorer_channel,
            "fused_column": fused_column,

            # Direct check of Proposition 1 within this ablation.
            "proposition_1": verify_proposition_1(
                raw_values,
                calibrated_values,
            ),

            # These are application-level properties, not guaranteed
            # merely by Proposition 1.
            "raw_scorer_monotone_with_efficiency": (
                monotone_decreasing(
                    raw_values
                )
            ),
            "calibrated_scorer_monotone_with_efficiency": (
                monotone_decreasing(
                    calibrated_values
                )
            ),
            "concept_fusion_monotone_with_efficiency": (
                monotone_decreasing(
                    fused_values
                )
            ),

            "raw_values": raw_values,
            "calibrated_values": calibrated_values,
            "concept_kappa_values": fused_values,
        }

    return (
        rows,
        checks,
    )


# ---------------------------------------------------------------------------
# Proposition 2: combined variation
# ---------------------------------------------------------------------------

def make_combined_case(
    notes: NoteTemplates,
    *,
    concept: str,
    text_level: int,
    structured_level: int,
) -> dict[str, Any]:
    """Vary text and structured evidence independently for one concept."""

    arguments: dict[
        str,
        Any,
    ] = {
        "wall_level": 3,
        "roof_level": 3,
        "heating_level": 3,
        "cooling_level": 3,
        "hvac_note_level": 3,
        "insulation_note_level": 3,
        "record_id": (
            f"combined_{concept}_"
            f"text{text_level}_"
            f"structured{structured_level}"
        ),
    }

    if concept == "hvac":
        arguments[
            "hvac_note_level"
        ] = text_level

        arguments[
            "heating_level"
        ] = structured_level

        arguments[
            "cooling_level"
        ] = structured_level

    elif concept == "insulation":
        arguments[
            "insulation_note_level"
        ] = text_level

        arguments[
            "wall_level"
        ] = structured_level

        arguments[
            "roof_level"
        ] = structured_level

    else:
        raise ValueError(
            f"Unknown concept: {concept}"
        )

    return make_home(
        notes,
        **arguments,
    )


def run_combined_variation(
    notes: NoteTemplates,
    text_scorer: CachedTextScorer,
    statistics: Mapping[
        str,
        CalibrationStats,
    ],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    case_rows: list[
        dict[str, Any]
    ] = []

    pairwise_rows: list[
        dict[str, Any]
    ] = []

    checks: dict[
        str,
        Any,
    ] = {}

    for concept in (
        "hvac",
        "insulation",
    ):
        concept_cases: list[
            dict[str, Any]
        ] = []

        for text_level in range(
            1,
            6,
        ):
            for structured_level in range(
                1,
                6,
            ):
                row = score_case(
                    make_combined_case(
                        notes,
                        concept=concept,
                        text_level=text_level,
                        structured_level=structured_level,
                    ),
                    text_scorer,
                    statistics,
                )

                row.update(
                    {
                        "experiment": "combined_variation",
                        "concept": concept,
                        "text_efficiency_level": text_level,
                        "structured_efficiency_level": (
                            structured_level
                        ),
                    }
                )

                case_rows.append(
                    row
                )

                concept_cases.append(
                    row
                )

        failures = 0
        comparisons_checked = 0
        concept_channels = CONCEPT_CHANNELS[
            concept
        ]

        for first, second in combinations(
            concept_cases,
            2,
        ):
            comparisons_checked += 1

            check = verify_proposition_2(
                calibrated_channel_scores(
                    first,
                    concept,
                ),
                calibrated_channel_scores(
                    second,
                    concept,
                ),
            )

            if not check[
                "passed"
            ]:
                failures += 1

            pairwise_row: dict[
                str,
                Any,
            ] = {
                "concept": concept,
                "a_record_id": first[
                    "record_id"
                ],
                "b_record_id": second[
                    "record_id"
                ],
                "a_text_efficiency_level": first[
                    "text_efficiency_level"
                ],
                "a_structured_efficiency_level": first[
                    "structured_efficiency_level"
                ],
                "b_text_efficiency_level": second[
                    "text_efficiency_level"
                ],
                "b_structured_efficiency_level": second[
                    "structured_efficiency_level"
                ],
                "positive_support": check[
                    "positive_support"
                ],
                "negative_support": check[
                    "negative_support"
                ],
                "support_balance": check[
                    "support_balance"
                ],
                "fused_a": check[
                    "fused_a"
                ],
                "fused_b": check[
                    "fused_b"
                ],
                "fused_delta": check[
                    "fused_delta"
                ],
                "passed": check[
                    "passed"
                ],
            }

            for channel in concept_channels:
                pairwise_row[
                    f"delta_{channel}"
                ] = check[
                    "deltas"
                ][channel]

            pairwise_rows.append(
                pairwise_row
            )

        checks[
            concept
        ] = {
            "comparisons": comparisons_checked,
            "failures": failures,
            "passed": failures == 0,
        }

    return (
        case_rows,
        pairwise_rows,
        checks,
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def collect_fieldnames(
    records: Sequence[
        Mapping[str, Any]
    ],
) -> list[str]:
    fieldnames: list[
        str
    ] = []

    seen: set[
        str
    ] = set()

    for record in records:
        for key in record:
            if key not in seen:
                seen.add(
                    key
                )

                fieldnames.append(
                    key
                )

    return fieldnames


def write_csv(
    records: Sequence[
        Mapping[str, Any]
    ],
    path: Path,
) -> None:
    if not records:
        raise ValueError(
            f"Cannot write empty CSV: {path}"
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=collect_fieldnames(
                records
            ),
            extrasaction="raise",
        )

        writer.writeheader()
        writer.writerows(
            records
        )


def write_json(
    payload: object,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            payload,
            handle,
            indent=2,
        )

        handle.write(
            "\n"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build, calibrate, and test the Synthetic Homes "
            "evidence-channel framework."
        )
    )

    parser.add_argument(
        "--notes",
        type=Path,
        required=True,
        help=(
            "JSON file containing level-1-through-level-5 "
            "HVAC and insulation note components."
        ),
    )

    parser.add_argument(
        "--text-scorer",
        required=True,
        help=(
            "Use 'module:function' for the real whole-note scorer, "
            "or 'demo' for a mechanical smoke test."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    return parser.parse_args()


def main() -> int:
    load_dotenv()

    arguments = parse_args()

    notes = load_note_templates(
        arguments.notes
    )

    text_scorer = CachedTextScorer(
        load_text_scorer(
            arguments.text_scorer
        )
    )

    if arguments.text_scorer == "demo":
        print(
            "WARNING: demo text scores are placeholders. "
            "Do not report them as experimental results."
        )

    # 1. Build and score the common 625-point calibration space.
    calibration_space = build_calibration_space(
        notes
    )

    calibration_raw = [
        add_raw_scores(
            record,
            text_scorer,
        )
        for record in calibration_space
    ]

    # 2. Fit one calibration distribution per scorer.
    statistics = fit_channel_statistics(
        calibration_raw
    )

    calibration_scored = [
        calibrate_record(
            record,
            statistics,
        )
        for record in calibration_raw
    ]

    # 3. Proposition 1: direct full-space check.
    proposition_1_checks = (
        check_proposition_1_over_calibration_space(
            calibration_scored
        )
    )

    # 4. Proposition 1: ablation illustration.
    (
        ablation_rows,
        ablation_checks,
    ) = run_ablation(
        notes,
        text_scorer,
        statistics,
    )

    # 5. Proposition 2: combined variation and pairwise decomposition.
    (
        combined_rows,
        combined_pairwise_rows,
        proposition_2_checks,
    ) = run_combined_variation(
        notes,
        text_scorer,
        statistics,
    )

    proposition_1_passed = all(
        check[
            "passed"
        ]
        for check in proposition_1_checks.values()
    )

    proposition_2_passed = all(
        check[
            "passed"
        ]
        for check in proposition_2_checks.values()
    )

    # 6. Write outputs.
    write_csv(
        calibration_space,
        arguments.output_dir
        / "calibration_space.csv",
    )

    write_csv(
        calibration_scored,
        arguments.output_dir
        / "calibration_scored.csv",
    )

    write_csv(
        ablation_rows,
        arguments.output_dir
        / "ablation_results.csv",
    )

    write_csv(
        combined_rows,
        arguments.output_dir
        / "combined_variation_results.csv",
    )

    write_csv(
        combined_pairwise_rows,
        arguments.output_dir
        / "combined_variation_pairwise_checks.csv",
    )

    write_json(
        {
            "framework_unit": (
                "evidence-channel scorer"
            ),
            "score_direction": (
                "larger values indicate greater retrofit need"
            ),
            "calibration_cardinality": (
                len(
                    calibration_scored
                )
            ),
            "distinct_whole_home_notes_scored": (
                len(
                    text_scorer.cache
                )
            ),
            "channels": {
                channel: statistics[
                    channel
                ].to_dict()
                for channel in CHANNELS
            },
        },
        arguments.output_dir
        / "calibration_stats.json",
    )

    write_json(
        {
            "proposition_1": {
                "description": (
                    "Every pairwise raw scorer ordering is preserved "
                    "after standardization."
                ),
                "calibration_space_checks": proposition_1_checks,
                "passed": proposition_1_passed,
            },
            "ablation": ablation_checks,
            "proposition_2": {
                "description": (
                    "Fused ordering matches the balance of positive "
                    "and negative calibrated support."
                ),
                "combined_variation_checks": proposition_2_checks,
                "passed": proposition_2_passed,
            },
            "all_proposition_checks_passed": (
                proposition_1_passed
                and proposition_2_passed
            ),
        },
        arguments.output_dir
        / "experiment_summary.json",
    )

    print(
        f"Calibration records: {len(calibration_scored)}"
    )

    print(
        "Distinct whole-home notes scored: "
        f"{len(text_scorer.cache)}"
    )

    print(
        "Proposition 1 passed: "
        f"{proposition_1_passed}"
    )

    print(
        "Proposition 2 passed: "
        f"{proposition_2_passed}"
    )

    print(
        f"Results written to: {arguments.output_dir}"
    )

    return (
        0
        if (
            proposition_1_passed
            and proposition_2_passed
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(
        main()
    )