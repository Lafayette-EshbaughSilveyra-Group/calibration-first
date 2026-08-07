#!/usr/bin/env python3

"""
Plot calibration-resolution sensitivity results.

Reads:
    k_sensitivity_results/k_sensitivity_summary.csv

Produces:
    k_sensitivity_plots/
        spearman_vs_resolution.png
        top10_overlap_vs_resolution.png
        stability_vs_compute.png
        mean_absolute_difference.png
        calibration_cost_scaling.png
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent

INPUT = (
    SCRIPT_DIR
    / "k_sensitivity_results"
    / "k_sensitivity_summary.csv"
)

OUTPUT = (
    SCRIPT_DIR
    / "k_sensitivity_plots"
)


def save_plot(name):
    path = OUTPUT / name
    plt.tight_layout()
    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print(f"Wrote {path}")


def main():

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(INPUT)

    df["k"] = (
        df["calibration_resolution"]
        .str.replace("k", "")
        .astype(int)
    )

    # ----------------------------------------------------------
    # 1. Spearman stability
    # ----------------------------------------------------------

    plt.figure(figsize=(6,4))

    plt.plot(
        df["k"],
        df["spearman_vs_reference"],
        marker="o",
    )

    plt.xlabel(
        "Calibration resolution k"
    )

    plt.ylabel(
        "Spearman correlation vs k=15"
    )

    plt.ylim(
        0.95,
        1.005,
    )

    plt.grid(True)

    save_plot(
        "spearman_vs_resolution.png"
    )


    # ----------------------------------------------------------
    # 2. Top-10% overlap
    # ----------------------------------------------------------

    plt.figure(figsize=(6,4))

    plt.plot(
        df["k"],
        df["top_10_percent_overlap_vs_reference"],
        marker="o",
    )

    plt.xlabel(
        "Calibration resolution k"
    )

    plt.ylabel(
        "Top-10% overlap vs k=15"
    )

    plt.ylim(
        0.75,
        1.05,
    )

    plt.grid(True)

    save_plot(
        "top10_overlap_vs_resolution.png"
    )


    # ----------------------------------------------------------
    # 3. Pareto: compute vs stability
    # ----------------------------------------------------------

    plt.figure(figsize=(6,4))

    plt.plot(
        df["energyplus_compute_seconds"] / 3600,
        df["top_10_percent_overlap_vs_reference"],
        marker="o",
    )

    for _, row in df.iterrows():
        plt.annotate(
            row["calibration_resolution"],
            (
                row["energyplus_compute_seconds"] / 3600,
                row["top_10_percent_overlap_vs_reference"],
            ),
            xytext=(5,5),
            textcoords="offset points",
        )

    plt.xlabel(
        "EnergyPlus compute time (hours)"
    )

    plt.ylabel(
        "Top-10% overlap"
    )

    plt.grid(True)

    save_plot(
        "stability_vs_compute.png"
    )


    # ----------------------------------------------------------
    # 4. Mean absolute deviation
    # ----------------------------------------------------------

    plt.figure(figsize=(6,4))

    plt.plot(
        df["k"],
        df["mean_abs_diff_vs_reference"],
        marker="o",
    )

    plt.xlabel(
        "Calibration resolution k"
    )

    plt.ylabel(
        "Mean |Δκ| vs k=15"
    )

    plt.grid(True)

    save_plot(
        "mean_absolute_difference.png"
    )


    # ----------------------------------------------------------
    # 5. k^4 scaling
    # ----------------------------------------------------------

    plt.figure(figsize=(6,4))

    plt.plot(
        df["k"],
        df["num_calibration_points"],
        marker="o",
    )

    plt.yscale(
        "log"
    )

    plt.xlabel(
        "Calibration resolution k"
    )

    plt.ylabel(
        "Number of calibration points (log)"
    )

    plt.grid(True)

    save_plot(
        "calibration_cost_scaling.png"
    )

    # ----------------------------------------------------------
    # Spearman vs EnergyPlus compute cost
    # ----------------------------------------------------------

    plt.figure(figsize=(6, 4))

    plt.plot(
        df["spearman_vs_reference"],
        df["energyplus_compute_seconds"] / 3600,
        marker="o",
    )

    for _, row in df.iterrows():
        plt.annotate(
            row["calibration_resolution"],
            (
                row["spearman_vs_reference"],
                row["energyplus_compute_seconds"] / 3600,
            ),
            xytext=(5, 5),
            textcoords="offset points",
        )

    plt.yscale(
        "log"
    )

    plt.xlabel(
        "Spearman correlation vs k=15"
    )

    plt.ylabel(
        "EnergyPlus compute time (hours, log scale)"
    )

    plt.xlim(
        0.95,
        1.005,
    )

    plt.grid(
        True
    )

    save_plot(
        "spearman_vs_energyplus_cost.png"
    )


if __name__ == "__main__":
    main()