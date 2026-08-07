#!/usr/bin/env python3

"""
Plot calibration-resolution sensitivity results.

Reads:
    k_sensitivity_results/k_sensitivity_summary.csv
    k_sensitivity_results/k_sensitivity_labels.csv

Produces:
    k_sensitivity_plots/
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent

RESULTS = (
    SCRIPT_DIR
    / "k_sensitivity_results"
)

SUMMARY_INPUT = (
    RESULTS
    / "k_sensitivity_summary.csv"
)

LABEL_INPUT = (
    RESULTS
    / "k_sensitivity_labels.csv"
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

    print(
        f"Wrote {path}"
    )


def annotate_points(
    x,
    y,
    labels,
):
    for xi, yi, label in zip(
        x,
        y,
        labels,
    ):
        plt.annotate(
            label,
            (xi, yi),
            xytext=(5, 5),
            textcoords="offset points",
        )


def main():

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(
        SUMMARY_INPUT
    )

    df["k"] = (
        df["calibration_resolution"]
        .str.replace("k", "")
        .astype(int)
    )

    labels = pd.read_csv(
        LABEL_INPUT
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
    # 2. Top-10 overlap
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
    # 3. Top-10 overlap vs compute
    # ----------------------------------------------------------

    plt.figure(figsize=(6,4))

    x = (
        df["energyplus_compute_seconds"]
        / 3600
    )

    y = (
        df["top_10_percent_overlap_vs_reference"]
    )

    plt.plot(
        x,
        y,
        marker="o",
    )

    annotate_points(
        x,
        y,
        df["calibration_resolution"],
    )

    plt.xlabel(
        "EnergyPlus compute time (hours)"
    )

    plt.ylabel(
        "Top-10% overlap"
    )

    plt.grid(True)

    save_plot(
        "top10_vs_compute.png"
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
    # 5. Calibration cost scaling
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
    # 6. Spearman vs EnergyPlus compute
    # ----------------------------------------------------------

    plt.figure(figsize=(6,4))

    x = (
        df["spearman_vs_reference"]
    )

    y = (
        df["energyplus_compute_seconds"]
        / 3600
    )

    plt.plot(
        x,
        y,
        marker="o",
    )

    annotate_points(
        x,
        y,
        df["calibration_resolution"],
    )

    plt.yscale(
        "log"
    )

    plt.xlabel(
        "Spearman correlation vs k=15"
    )

    plt.ylabel(
        "EnergyPlus compute time (hours, log)"
    )

    plt.xlim(
        0.95,
        1.005,
    )

    plt.grid(True)

    save_plot(
        "spearman_vs_energyplus_cost.png"
    )


    # ----------------------------------------------------------
    # 7. Score agreement scatter plots
    # ----------------------------------------------------------

    reference = labels[
        labels["calibration_resolution"] == "k15"
    ][
        [
            "example_id",
            "overall_label",
        ]
    ].rename(
        columns={
            "overall_label": "reference_label"
        }
    )


    for k_name in (
        "k5",
        "k7",
        "k9",
    ):

        candidate = labels[
            labels["calibration_resolution"]
            == k_name
        ][
            [
                "example_id",
                "overall_label",
            ]
        ].rename(
            columns={
                "overall_label": "candidate_label"
            }
        )


        merged = candidate.merge(
            reference,
            on="example_id",
        )


        plt.figure(
            figsize=(5,5)
        )

        plt.scatter(
            merged["reference_label"],
            merged["candidate_label"],
            s=5,
            alpha=0.25,
        )


        minimum = min(
            merged["reference_label"].min(),
            merged["candidate_label"].min(),
        )

        maximum = max(
            merged["reference_label"].max(),
            merged["candidate_label"].max(),
        )

        plt.plot(
            [
                minimum,
                maximum,
            ],
            [
                minimum,
                maximum,
            ],
            linestyle="--",
        )


        plt.xlabel(
            "k=15 supervision score"
        )

        plt.ylabel(
            f"{k_name} supervision score"
        )

        plt.axis(
            "equal"
        )

        plt.grid(True)

        save_plot(
            f"{k_name}_score_vs_k15.png"
        )


    # ----------------------------------------------------------
    # 8. Marginal Spearman gain per compute hour
    # ----------------------------------------------------------

    temp = df.sort_values(
        "k"
    ).copy()

    temp["hours"] = (
        temp["energyplus_compute_seconds"]
        / 3600
    )

    temp["delta_spearman"] = (
        temp["spearman_vs_reference"]
        .diff()
    )

    temp["delta_hours"] = (
        temp["hours"]
        .diff()
    )

    temp["gain_per_hour"] = (
        temp["delta_spearman"]
        /
        temp["delta_hours"]
    )


    plt.figure(figsize=(6,4))

    plt.plot(
        temp["k"],
        temp["gain_per_hour"],
        marker="o",
    )

    plt.yscale(
        "log"
    )

    plt.xlabel(
        "Calibration resolution k"
    )

    plt.ylabel(
        "ΔSpearman / additional EnergyPlus hour"
    )

    plt.grid(True)

    save_plot(
        "marginal_gain_per_compute.png"
    )


    # ----------------------------------------------------------
    # 9. Fraction of calibration space vs stability
    # ----------------------------------------------------------

    plt.figure(figsize=(6,4))

    x = df["fraction_of_reference_grid"]

    y = df["spearman_vs_reference"]

    plt.plot(
        x,
        y,
        marker="o",
    )

    annotate_points(
        x,
        y,
        df["calibration_resolution"],
    )

    plt.xscale(
        "log"
    )

    plt.xlabel(
        "Fraction of k=15 calibration grid"
    )

    plt.ylabel(
        "Spearman correlation vs k=15"
    )

    plt.grid(True)

    save_plot(
        "fraction_grid_vs_stability.png"
    )


    # ----------------------------------------------------------
    # 10. Fraction of calibration space vs top-10 overlap
    # ----------------------------------------------------------

    plt.figure(figsize=(6,4))

    x = df["fraction_of_reference_grid"]

    y = (
        df["top_10_percent_overlap_vs_reference"]
    )

    plt.plot(
        x,
        y,
        marker="o",
    )

    annotate_points(
        x,
        y,
        df["calibration_resolution"],
    )

    plt.xscale(
        "log"
    )

    plt.xlabel(
        "Fraction of k=15 calibration grid"
    )

    plt.ylabel(
        "Top-10% overlap"
    )

    plt.grid(True)

    save_plot(
        "fraction_grid_vs_top10.png"
    )


    # ----------------------------------------------------------
    # 11. Mean difference vs compute
    # ----------------------------------------------------------

    plt.figure(figsize=(6,4))

    x = (
        df["energyplus_compute_seconds"]
        / 3600
    )

    y = (
        df["mean_abs_diff_vs_reference"]
    )

    plt.plot(
        x,
        y,
        marker="o",
    )

    annotate_points(
        x,
        y,
        df["calibration_resolution"],
    )

    plt.xscale(
        "log"
    )

    plt.yscale(
        "log"
    )

    plt.xlabel(
        "EnergyPlus compute time (hours, log)"
    )

    plt.ylabel(
        "Mean |Δκ| vs k=15 (log)"
    )

    plt.grid(True)

    save_plot(
        "difference_vs_compute.png"
    )


if __name__ == "__main__":
    main()