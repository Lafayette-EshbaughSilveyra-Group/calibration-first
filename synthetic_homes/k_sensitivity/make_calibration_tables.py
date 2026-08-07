#!/usr/bin/env python3

"""
Generate LaTeX tables describing calibration-grid level selection.

Inputs:
    calibration_grid/data/calibration_levels.json

Outputs:
    calibration_level_selection.tex
    calibration_parameter_values.tex
"""

from pathlib import Path
import json


SCRIPT_DIR = Path(__file__).resolve().parent

LEVEL_FILE = (
    SCRIPT_DIR
    / "calibration_grid"
    / "data"
    / "calibration_levels.json"
)

OUTPUT_DIR = (
    SCRIPT_DIR
    / "tables"
)


K_VALUES = (
    2,
    3,
    5,
    7,
    9,
    11,
    13,
    15,
)


def evenly_spaced_level_indices(k, max_k):
    """
    Return k approximately evenly spaced one-based levels from 1..max_k.
    """

    return tuple(
        int(
            round(
                i
                * (max_k - 1)
                / (k - 1)
            )
        )
        + 1
        for i in range(k)
    )


def write_selection_table(levels, output):

    max_k = len(
        next(iter(levels.values()))
    )

    lines = []

    lines.append(
        r"""
\begin{table}[t]
\centering
\caption{Selected ordinal levels from the $k=15$ master calibration grid for each calibration resolution.}
\label{tab:k_level_selection}
\begin{tabular}{cc}
\toprule
Resolution & Selected levels \\
\midrule
"""
    )

    for k in K_VALUES:
        selected = evenly_spaced_level_indices(
            k,
            max_k,
        )

        selected = [
            x + 1
            for x in selected
        ]

        lines.append(
            f"$k={k}$ & "
            + ", ".join(
                map(
                    str,
                    selected,
                )
            )
            + r" \\"
            + "\n"
        )

    lines.append(
        r"""
\bottomrule
\end{tabular}
\end{table}
"""
    )

    output.write_text(
        "".join(lines),
        encoding="utf-8",
    )


def write_parameter_table(levels, output):

    names = {
        "wall_r_value": r"Wall $R$-value",
        "roof_r_value": r"Roof $R$-value",
        "hvac_heating_cop": r"Heating COP",
        "hvac_cooling_cop": r"Cooling COP",
    }

    n_levels = len(
        next(iter(levels.values()))
    )

    lines = []

    lines.append(
        r"""
\begin{table}[t]
\centering
\caption{Physical values associated with the $k=15$ calibration levels.}
\label{tab:calibration_parameter_values}
\begin{tabular}{ccccc}
\toprule
Level & Wall $R$ & Roof $R$ & Heating COP & Cooling COP \\
\midrule
"""
    )

    for i in range(n_levels):

        lines.append(
            f"{i+1} & "
            f"{levels['wall_r_value'][i]:g} & "
            f"{levels['roof_r_value'][i]:g} & "
            f"{levels['hvac_heating_cop'][i]:g} & "
            f"{levels['hvac_cooling_cop'][i]:g}"
            r" \\"
            "\n"
        )

    lines.append(
        r"""
\bottomrule
\end{tabular}
\end{table}
"""
    )

    output.write_text(
        "".join(lines),
        encoding="utf-8",
    )


def main():

    with LEVEL_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:
        payload = json.load(f)

    levels = payload[
        "parameter_levels"
    ]

    write_selection_table(
        levels,
        OUTPUT_DIR
        / "calibration_level_selection.tex",
    )

    write_parameter_table(
        levels,
        OUTPUT_DIR
        / "calibration_parameter_values.tex",
    )

    print(
        "Generated:"
    )
    print(
        "  calibration_level_selection.tex"
    )
    print(
        "  calibration_parameter_values.tex"
    )


if __name__ == "__main__":
    main()