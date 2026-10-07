#!/usr/bin/env python3
"""
statistical_analysis.py

Statistical analysis for the cross-domain calibration experiment.

Expected files, by default in the same directory as this script:

    ames_housing_observations.csv
    breast_cancer_wisconsin_observations.csv
    wine_quality_observations.csv

Outputs are written to:

    statistical_analysis/

Main outputs:
    dataset_summary.csv
    point_estimates_and_cis.csv
    paired_differences.csv
    primary_pca_comparison.csv
    analysis_summary.md
    analysis_config.json
    bootstrap_replicates/*.npz

The bootstrap is PAIRED:
within each dataset and bootstrap replicate, the exact same sampled
observation indices are used for every method. This is essential for
valid CIs on method differences.

Dependencies:
    numpy
    pandas
    scipy
    scikit-learn

Example:
    python statistical_analysis.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

METHODS = {
    "kappa": "Calibrated Fusion (κ)",
    "uncalibrated_average": "Uncalibrated Avg",
    "empirical_zscore_average": "Empirical z-Score",
    "borda_count_average": "Borda Rank Avg",
    "pca_first_component": "Unsupervised PCA",
}

REFERENCE_METHOD = "kappa"

DATASETS = {
    "ames": {
        "filename": "ames_housing_observations.csv",
        "name": "Ames Housing",
        "target": "SalePrice",
        "task": "regression",
        "primary_metric": "spearman_rho",
        "metrics": [
            "spearman_rho",
            "kendall_tau",
            "top_10pct_overlap",
        ],
    },
    "bcw": {
        "filename": "breast_cancer_wisconsin_observations.csv",
        "name": "Breast Cancer Wisconsin",
        "target": "malignant",
        "task": "classification",
        "primary_metric": "auroc",
        "metrics": [
            "auroc",
            "average_precision",
        ],
    },
    "wine": {
        "filename": "wine_quality_observations.csv",
        "name": "Wine Quality",
        "target": "quality",
        "task": "regression",
        "primary_metric": "spearman_rho",
        "metrics": [
            "spearman_rho",
            "kendall_tau",
            "top_10pct_overlap",
        ],
    },
}


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def spearman_rho(y: np.ndarray, score: np.ndarray) -> float:
    result = spearmanr(y, score, nan_policy="omit")
    return float(result.statistic)


def kendall_tau(y: np.ndarray, score: np.ndarray) -> float:
    result = kendalltau(y, score, nan_policy="omit")
    return float(result.statistic)


def auroc(y: np.ndarray, score: np.ndarray) -> float:
    # AUROC is undefined if a bootstrap sample contains only one class.
    if np.unique(y).size < 2:
        return np.nan
    return float(roc_auc_score(y, score))


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    if np.unique(y).size < 2:
        return np.nan
    return float(average_precision_score(y, score))


def top_10pct_overlap(y: np.ndarray, score: np.ndarray) -> float:
    """
    Fractional overlap between the observations in the top 10% according
    to the target and the top 10% according to the method score.

    This matches the usual fixed-k top-decile definition. Note that
    discrete targets such as Wine Quality can have ties at the cutoff.
    """
    n = len(y)
    k = max(1, int(np.ceil(0.10 * n)))

    # Stable sorting makes tie behavior deterministic.
    target_top = np.argsort(-y, kind="stable")[:k]
    score_top = np.argsort(-score, kind="stable")[:k]

    return float(
        len(np.intersect1d(target_top, score_top, assume_unique=False)) / k
    )


METRIC_FUNCTIONS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "spearman_rho": spearman_rho,
    "kendall_tau": kendall_tau,
    "top_10pct_overlap": top_10pct_overlap,
    "auroc": auroc,
    "average_precision": average_precision,
}

METRIC_LABELS = {
    "spearman_rho": "Spearman's rho",
    "kendall_tau": "Kendall's tau",
    "top_10pct_overlap": "Top-10% overlap",
    "auroc": "AUROC",
    "average_precision": "Average precision",
}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def percentile_ci(
    values: np.ndarray,
    confidence: float = 0.95,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan, np.nan

    alpha = 1.0 - confidence
    lo = np.quantile(values, alpha / 2.0)
    hi = np.quantile(values, 1.0 - alpha / 2.0)

    return float(lo), float(hi)


def alpha_percentile_ci(
    values: np.ndarray,
    alpha: float,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan, np.nan

    return (
        float(np.quantile(values, alpha / 2.0)),
        float(np.quantile(values, 1.0 - alpha / 2.0)),
    )


def metric_value(
    metric: str,
    y: np.ndarray,
    score: np.ndarray,
) -> float:
    return METRIC_FUNCTIONS[metric](y, score)


def validate_columns(df: pd.DataFrame, cfg: dict) -> None:
    required = {cfg["target"], *METHODS.keys()}
    missing = sorted(required - set(df.columns))

    if missing:
        raise ValueError(
            f"{cfg['name']}: missing required columns: {missing}"
        )


def load_dataset(path: Path, cfg: dict) -> pd.DataFrame:
    df = pd.read_csv(path)
    validate_columns(df, cfg)

    needed = [cfg["target"], *METHODS.keys()]

    before = len(df)
    df = df.dropna(subset=needed).reset_index(drop=True)
    after = len(df)

    if after != before:
        print(
            f"{cfg['name']}: dropped {before - after} rows with missing "
            f"target/method values."
        )

    return df


# ---------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------

def bootstrap_dataset(
    df: pd.DataFrame,
    cfg: dict,
    B: int,
    rng: np.random.Generator,
) -> tuple[
    dict[tuple[str, str], np.ndarray],
    dict[tuple[str, str], float],
]:
    """
    Returns
    -------
    bootstrap_values:
        (metric, method) -> array of B bootstrap estimates

    point_estimates:
        (metric, method) -> estimate on the original sample
    """

    n = len(df)
    y = df[cfg["target"]].to_numpy()

    scores = {
        method: df[method].to_numpy()
        for method in METHODS
    }

    point_estimates = {}
    for metric in cfg["metrics"]:
        for method, score in scores.items():
            point_estimates[(metric, method)] = metric_value(
                metric, y, score
            )

    bootstrap_values = {
        (metric, method): np.full(B, np.nan, dtype=float)
        for metric in cfg["metrics"]
        for method in METHODS
    }

    # IMPORTANT:
    # One set of sampled indices per replicate, shared by every method.
    for b in range(B):
        idx = rng.integers(0, n, size=n)

        y_b = y[idx]

        for metric in cfg["metrics"]:
            for method, score in scores.items():
                bootstrap_values[(metric, method)][b] = metric_value(
                    metric,
                    y_b,
                    score[idx],
                )

    return bootstrap_values, point_estimates


# ---------------------------------------------------------------------
# Output construction
# ---------------------------------------------------------------------

def build_point_estimate_table(
    dataset_key: str,
    cfg: dict,
    bootstrap_values: dict,
    point_estimates: dict,
    confidence: float,
    B: int,
) -> list[dict]:
    rows = []

    for metric in cfg["metrics"]:
        for method in METHODS:
            reps = bootstrap_values[(metric, method)]
            lo, hi = percentile_ci(reps, confidence)

            rows.append({
                "dataset_key": dataset_key,
                "dataset": cfg["name"],
                "task": cfg["task"],
                "metric": metric,
                "metric_label": METRIC_LABELS[metric],
                "primary_metric": metric == cfg["primary_metric"],
                "method": method,
                "method_label": METHODS[method],
                "estimate": point_estimates[(metric, method)],
                "ci_level": confidence,
                "ci_low": lo,
                "ci_high": hi,
                "bootstrap_replicates": B,
                "valid_bootstrap_replicates": int(
                    np.isfinite(reps).sum()
                ),
            })

    return rows


def build_paired_table(
    dataset_key: str,
    cfg: dict,
    bootstrap_values: dict,
    point_estimates: dict,
    confidence: float,
    B: int,
) -> list[dict]:
    """
    Paired differences:
        Calibrated Fusion (kappa) - comparison method

    Positive values favor kappa on all metrics used here.
    """
    rows = []

    for metric in cfg["metrics"]:
        ref_reps = bootstrap_values[(metric, REFERENCE_METHOD)]
        ref_est = point_estimates[(metric, REFERENCE_METHOD)]

        for comparison in METHODS:
            if comparison == REFERENCE_METHOD:
                continue

            comp_reps = bootstrap_values[(metric, comparison)]
            comp_est = point_estimates[(metric, comparison)]

            diff_reps = ref_reps - comp_reps
            diff = ref_est - comp_est

            lo, hi = percentile_ci(diff_reps, confidence)

            if lo > 0:
                sign = "kappa_higher"
            elif hi < 0:
                sign = "kappa_lower"
            else:
                sign = "includes_zero"

            rows.append({
                "dataset_key": dataset_key,
                "dataset": cfg["name"],
                "metric": metric,
                "metric_label": METRIC_LABELS[metric],
                "primary_metric": metric == cfg["primary_metric"],
                "reference_method": REFERENCE_METHOD,
                "reference_label": METHODS[REFERENCE_METHOD],
                "comparison_method": comparison,
                "comparison_label": METHODS[comparison],
                "reference_estimate": ref_est,
                "comparison_estimate": comp_est,
                "difference_reference_minus_comparison": diff,
                "absolute_difference": abs(diff),
                "ci_level": confidence,
                "difference_ci_low": lo,
                "difference_ci_high": hi,
                "ci_excludes_zero": bool(lo > 0 or hi < 0),
                "ci_direction": sign,
                "bootstrap_replicates": B,
                "valid_bootstrap_replicates": int(
                    np.isfinite(diff_reps).sum()
                ),
                # Descriptive only: not a frequentist p-value.
                "bootstrap_fraction_difference_gt_zero": float(
                    np.nanmean(diff_reps > 0)
                ),
                "bootstrap_fraction_difference_lt_zero": float(
                    np.nanmean(diff_reps < 0)
                ),
            })

    return rows


def build_primary_baseline_table(
    all_results: dict,
    confidence: float,
) -> pd.DataFrame:
    """
    Primary metric only:
        kappa - comparison baseline

    For each baseline separately, includes:
      - ordinary paired bootstrap CI
      - Bonferroni-adjusted familywise CI across the three datasets

    Thus each baseline defines its own family of three comparisons.
    """
    m = len(DATASETS)
    family_alpha = (1.0 - confidence) / m
    family_confidence = 1.0 - family_alpha

    rows = []

    for comparison in METHODS:
        if comparison == REFERENCE_METHOD:
            continue

        for dataset_key, result in all_results.items():
            cfg = result["cfg"]
            metric = cfg["primary_metric"]

            point = result["point_estimates"]
            boot = result["bootstrap_values"]

            kappa_est = point[(metric, REFERENCE_METHOD)]
            comparison_est = point[(metric, comparison)]

            diff = kappa_est - comparison_est

            diff_reps = (
                boot[(metric, REFERENCE_METHOD)]
                - boot[(metric, comparison)]
            )

            # Ordinary paired bootstrap CI
            lo, hi = percentile_ci(diff_reps, confidence)

            # Bonferroni-adjusted CI across the three datasets
            fw_lo, fw_hi = alpha_percentile_ci(
                diff_reps,
                family_alpha,
            )

            rows.append({
                "dataset_key": dataset_key,
                "dataset": cfg["name"],
                "primary_metric": metric,
                "metric_label": METRIC_LABELS[metric],

                "reference_method": REFERENCE_METHOD,
                "reference_label": METHODS[REFERENCE_METHOD],
                "comparison_method": comparison,
                "comparison_label": METHODS[comparison],

                "reference_estimate": kappa_est,
                "comparison_estimate": comparison_est,
                "difference_reference_minus_comparison": diff,

                "ci_level": confidence,
                "ci_low": lo,
                "ci_high": hi,
                "ci_excludes_zero": bool(lo > 0 or hi < 0),

                "familywise_method": "Bonferroni",
                "family_definition": (
                    f"{METHODS[REFERENCE_METHOD]} vs "
                    f"{METHODS[comparison]} across datasets"
                ),
                "family_size": m,
                "familywise_ci_level_per_comparison": family_confidence,
                "familywise_ci_low": fw_lo,
                "familywise_ci_high": fw_hi,
                "familywise_ci_excludes_zero": bool(
                    fw_lo > 0 or fw_hi < 0
                ),

                "bootstrap_fraction_reference_gt_comparison": float(
                    np.nanmean(diff_reps > 0)
                ),
                "bootstrap_fraction_reference_lt_comparison": float(
                    np.nanmean(diff_reps < 0)
                ),
            })

    return pd.DataFrame(rows)


def dataset_summary_row(
    dataset_key: str,
    cfg: dict,
    df: pd.DataFrame,
) -> dict:
    row = {
        "dataset_key": dataset_key,
        "dataset": cfg["name"],
        "task": cfg["task"],
        "target": cfg["target"],
        "n": len(df),
        "primary_metric": cfg["primary_metric"],
    }

    if cfg["task"] == "classification":
        y = df[cfg["target"]]
        row["n_positive"] = int((y == 1).sum())
        row["n_negative"] = int((y == 0).sum())
        row["positive_fraction"] = float((y == 1).mean())

    return row


# ---------------------------------------------------------------------
# Markdown summary
# ---------------------------------------------------------------------

def fmt(x: float, digits: int = 4) -> str:
    if pd.isna(x):
        return "NA"
    return f"{x:.{digits}f}"


def write_markdown_summary(
    path: Path,
    summary_df: pd.DataFrame,
    points_df: pd.DataFrame,
    paired_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    B: int,
    confidence: float,
    seed: int,
) -> None:
    lines = []

    lines.append("# Cross-domain statistical analysis")
    lines.append("")
    lines.append(
        f"- Bootstrap replicates: **{B:,}**"
    )
    lines.append(
        f"- Nominal confidence level: **{confidence:.1%}**"
    )
    lines.append(f"- Random seed: **{seed}**")
    lines.append(
        "- All method comparisons use **paired bootstrap resampling**: "
        "each bootstrap replicate uses the same sampled observations "
        "for every method."
    )
    lines.append("")

    lines.append("## Dataset summary")
    lines.append("")
    lines.append(
        "| Dataset | n | Task | Primary metric |"
    )
    lines.append(
        "|---|---:|---|---|"
    )

    for _, r in summary_df.iterrows():
        lines.append(
            f"| {r['dataset']} | {int(r['n'])} | "
            f"{r['task']} | {METRIC_LABELS[r['primary_metric']]} |"
        )

    lines.append("")
    lines.append("## Primary paired comparisons")
    lines.append("")
    lines.append(
        "Differences are **Calibrated Fusion − baseline**. "
        "For each baseline, Bonferroni correction controls the "
        "family-wise error rate across the three benchmark datasets."
    )
    lines.append("")
    lines.append(
        "| Dataset | Baseline | κ | Baseline estimate | Difference | "
        f"{confidence:.0%} paired CI | "
        "Bonferroni-adjusted CI |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|")

    for _, r in baseline_df.iterrows():
        lines.append(
            f"| {r['dataset']} | "
            f"{r['comparison_label']} | "
            f"{fmt(r['reference_estimate'])} | "
            f"{fmt(r['comparison_estimate'])} | "
            f"{fmt(r['difference_reference_minus_comparison'])} | "
            f"[{fmt(r['ci_low'])}, {fmt(r['ci_high'])}] | "
            f"[{fmt(r['familywise_ci_low'])}, "
            f"{fmt(r['familywise_ci_high'])}] |"
        )

    lines.append("")
    lines.append("## All paired comparisons")
    lines.append("")
    lines.append(
        "The full machine-readable results are in "
        "`paired_differences.csv`."
    )
    lines.append("")
    lines.append(
        "**Important:** `bootstrap_fraction_*` columns are descriptive "
        "fractions of bootstrap replicates. They are not reported as "
        "frequentist p-values."
    )
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bootstrap analysis for cross-domain validation."
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing the observation CSVs. "
            "Default: directory containing this script."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Default: "
            "<script directory>/statistical_analysis"
        ),
    )

    parser.add_argument(
        "--bootstrap",
        "-B",
        type=int,
        default=10000,
        help=(
            "Number of bootstrap replicates. "
            "Default: 10000."
        ),
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Nominal confidence level. Default: 0.95",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260922,
        help="Random seed for reproducibility.",
    )

    args = parser.parse_args()

    if args.bootstrap < 100:
        raise ValueError("Use at least 100 bootstrap replicates.")

    if not (0 < args.confidence < 1):
        raise ValueError("--confidence must be between 0 and 1.")

    script_dir = Path(__file__).resolve().parent
    input_dir = (
        args.input_dir.resolve()
        if args.input_dir is not None
        else script_dir / "results" / "cross_domain_validation"
    )

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else script_dir / "statistical_analysis"
    )

    replicate_dir = output_dir / "bootstrap_replicates"

    output_dir.mkdir(parents=True, exist_ok=True)
    replicate_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    all_results = {}
    summary_rows = []
    point_rows = []
    paired_rows = []

    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Bootstrap B:      {args.bootstrap:,}")
    print(f"Confidence:       {args.confidence:.1%}")
    print(f"Seed:             {args.seed}")
    print()

    for dataset_key, cfg in DATASETS.items():
        path = input_dir / cfg["filename"]

        if not path.exists():
            raise FileNotFoundError(
                f"Could not find required file:\n  {path}"
            )

        print(f"Analyzing {cfg['name']} ...")

        df = load_dataset(path, cfg)
        summary_rows.append(
            dataset_summary_row(dataset_key, cfg, df)
        )

        bootstrap_values, point_estimates = bootstrap_dataset(
            df=df,
            cfg=cfg,
            B=args.bootstrap,
            rng=rng,
        )

        all_results[dataset_key] = {
            "cfg": cfg,
            "df": df,
            "bootstrap_values": bootstrap_values,
            "point_estimates": point_estimates,
        }

        point_rows.extend(
            build_point_estimate_table(
                dataset_key=dataset_key,
                cfg=cfg,
                bootstrap_values=bootstrap_values,
                point_estimates=point_estimates,
                confidence=args.confidence,
                B=args.bootstrap,
            )
        )

        paired_rows.extend(
            build_paired_table(
                dataset_key=dataset_key,
                cfg=cfg,
                bootstrap_values=bootstrap_values,
                point_estimates=point_estimates,
                confidence=args.confidence,
                B=args.bootstrap,
            )
        )

        # Save raw bootstrap draws so no future rerun is needed merely
        # to construct another table or plot.
        npz_payload = {}

        for (metric, method), values in bootstrap_values.items():
            key = f"{metric}__{method}"
            npz_payload[key] = values

        np.savez_compressed(
            replicate_dir / f"{dataset_key}_bootstrap.npz",
            **npz_payload,
        )

    summary_df = pd.DataFrame(summary_rows)
    points_df = pd.DataFrame(point_rows)
    paired_df = pd.DataFrame(paired_rows)

    baseline_df = build_primary_baseline_table(
        all_results=all_results,
        confidence=args.confidence,
    )

    summary_df.to_csv(
        output_dir / "dataset_summary.csv",
        index=False,
    )

    points_df.to_csv(
        output_dir / "point_estimates_and_cis.csv",
        index=False,
    )

    paired_df.to_csv(
        output_dir / "paired_differences.csv",
        index=False,
    )

    baseline_df.to_csv(
        output_dir / "primary_baseline_comparisons.csv",
        index=False,
    )

    config = {
        "bootstrap_replicates": args.bootstrap,
        "confidence": args.confidence,
        "seed": args.seed,
        "bootstrap_type": "paired nonparametric percentile bootstrap",
        "reference_method": REFERENCE_METHOD,
        "methods": METHODS,
        "datasets": DATASETS,
        "primary_comparisons": (
            "kappa versus each comparison method on the primary metric"
        ),
        "familywise_adjustment": {
            "method": "Bonferroni",
            "family": (
                "For each baseline separately, the primary-metric "
                "kappa-vs-baseline comparisons across the three datasets"
            ),
            "family_size_per_baseline": len(DATASETS),
        },
        "notes": [
            (
                "All methods within a bootstrap replicate use the same "
                "sampled observation indices."
            ),
            (
                "bootstrap_fraction_* values are descriptive bootstrap "
                "fractions, not frequentist p-values."
            ),
            (
                "No non-inferiority margin is tested. Such a margin should "
                "be scientifically justified rather than selected after "
                "observing the results."
            ),
        ],
    }

    with open(
        output_dir / "analysis_config.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(config, f, indent=2)

    write_markdown_summary(
        path=output_dir / "analysis_summary.md",
        summary_df=summary_df,
        points_df=points_df,
        paired_df=paired_df,
        baseline_df=baseline_df,
        B=args.bootstrap,
        confidence=args.confidence,
        seed=args.seed,
    )

    print()
    print("Done.")
    print()
    print("Primary kappa-vs-baseline results:")
    print(
        baseline_df[
            [
                "dataset",
                "metric_label",
                "comparison_label",
                "reference_estimate",
                "comparison_estimate",
                "difference_reference_minus_comparison",
                "ci_low",
                "ci_high",
                "familywise_ci_low",
                "familywise_ci_high",
            ]
        ].to_string(index=False)
    )
    print()
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()