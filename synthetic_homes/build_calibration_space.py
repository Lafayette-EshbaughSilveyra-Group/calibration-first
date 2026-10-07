#!/usr/bin/env python3
"""Build the Synthetic Homes calibration reference space.

This script constructs the full Cartesian product of the four ordered
Synthetic Homes calibration dimensions:

- wall R-value
- roof R-value
- HVAC heating COP
- HVAC cooling COP

Each dimension contains five ordered levels, producing 5^4 = 625 calibration
points. Level 1 is the least efficient condition and level 5 is the most
efficient condition.

The script only constructs and serializes the reference space. It does not run
EnergyPlus, invoke an LLM, or evaluate any modality-specific scorer.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Mapping, Sequence
from itertools import product
from pathlib import Path
from typing import Final


FEATURE_LEVELS: Final[dict[str, tuple[float, ...]]] = {
    "wall_r_value": (4.0, 7.0, 13.0, 20.0, 30.0),
    "roof_r_value": (10.0, 20.0, 30.0, 40.0, 50.0),
    "hvac_heating_cop": (0.7, 0.8, 0.9, 0.95, 1.0),
    "hvac_cooling_cop": (1.0, 2.0, 3.0, 3.5, 4.0),
}

DEFAULT_OUTPUT_PATH: Final[Path] = Path(
    "calibration_space/calibration_space.csv"
)

Record = dict[str, str | int | float]


def validate_feature_levels(
    feature_levels: Mapping[str, Sequence[float]],
) -> None:
    """Validate that every calibration dimension is ordered and non-degenerate."""
    if not feature_levels:
        raise ValueError("At least one calibration dimension is required.")

    for feature, levels in feature_levels.items():
        if not feature:
            raise ValueError("Feature names must be non-empty.")

        if len(levels) < 2:
            raise ValueError(
                f"{feature!r} must contain at least two levels."
            )

        numeric_levels = [float(value) for value in levels]

        if not all(math.isfinite(value) for value in numeric_levels):
            raise ValueError(
                f"{feature!r} contains a non-finite value."
            )

        if any(
            right <= left
            for left, right in zip(
                numeric_levels,
                numeric_levels[1:],
                strict=False,
            )
        ):
            raise ValueError(
                f"{feature!r} levels must be strictly increasing: "
                f"{numeric_levels!r}"
            )


def expected_space_size(
    feature_levels: Mapping[str, Sequence[float]] = FEATURE_LEVELS,
) -> int:
    """Return the cardinality of the Cartesian calibration space."""
    return math.prod(
        len(levels)
        for levels in feature_levels.values()
    )


def build_calibration_space(
    feature_levels: Mapping[str, Sequence[float]] = FEATURE_LEVELS,
) -> list[Record]:
    """Construct the deterministic full-factorial calibration space."""
    validate_feature_levels(feature_levels)

    feature_names = tuple(feature_levels.keys())

    indexed_dimensions = tuple(
        tuple(enumerate(levels, start=1))
        for levels in feature_levels.values()
    )

    records: list[Record] = []
    total_points = expected_space_size(feature_levels)
    id_width = max(4, len(str(total_points)))

    for position, combination in enumerate(
        product(*indexed_dimensions),
        start=1,
    ):
        record: Record = {
            "calibration_id": f"shc_{position:0{id_width}d}",
        }

        for feature, (level_index, value) in zip(
            feature_names,
            combination,
            strict=True,
        ):
            record[f"{feature}_level"] = level_index
            record[feature] = float(value)

        records.append(record)

    validate_calibration_space(records, feature_levels)
    return records


def validate_calibration_space(
    records: Sequence[Record],
    feature_levels: Mapping[str, Sequence[float]] = FEATURE_LEVELS,
) -> None:
    """Validate cardinality, uniqueness, and level-value consistency."""
    expected_size = expected_space_size(feature_levels)

    if len(records) != expected_size:
        raise ValueError(
            f"Expected {expected_size} calibration points; "
            f"got {len(records)}."
        )

    calibration_ids = [
        str(record["calibration_id"])
        for record in records
    ]

    if len(calibration_ids) != len(set(calibration_ids)):
        raise ValueError("Calibration IDs are not unique.")

    observed_points: set[tuple[float, ...]] = set()

    for record in records:
        point: list[float] = []

        for feature, levels in feature_levels.items():
            level_key = f"{feature}_level"

            if feature not in record or level_key not in record:
                raise ValueError(
                    f"Record {record.get('calibration_id')!r} is missing "
                    f"{feature!r} or {level_key!r}."
                )

            level_index = int(record[level_key])

            if not 1 <= level_index <= len(levels):
                raise ValueError(
                    f"Record {record['calibration_id']!r} has invalid "
                    f"{level_key}={level_index}."
                )

            observed_value = float(record[feature])
            expected_value = float(levels[level_index - 1])

            if observed_value != expected_value:
                raise ValueError(
                    f"Record {record['calibration_id']!r} maps "
                    f"{level_key}={level_index} to {observed_value}, "
                    f"but expected {expected_value}."
                )

            point.append(observed_value)

        observed_points.add(tuple(point))

    if len(observed_points) != expected_size:
        raise ValueError(
            "The calibration space does not contain every Cartesian "
            "point exactly once."
        )


def write_calibration_space(
    records: Sequence[Record],
    output_path: Path,
    *,
    overwrite: bool = False,
) -> None:
    """Write the calibration space to CSV."""
    if not records:
        raise ValueError("Cannot write an empty calibration space.")

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} already exists. "
            "Pass --overwrite to replace it."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(records[0].keys())

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Construct the 625-point Synthetic Homes calibration space."
        )
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=(
            "Output CSV path. "
            f"Default: {DEFAULT_OUTPUT_PATH}"
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output file if it already exists.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        records = build_calibration_space()

        write_calibration_space(
            records,
            args.output,
            overwrite=args.overwrite,
        )
    except (FileExistsError, OSError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error

    print(
        f"Wrote {len(records)} calibration points to "
        f"{args.output}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())