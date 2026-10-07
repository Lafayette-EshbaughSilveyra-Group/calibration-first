"""
Cross-domain validation experiments for Align Before You Combine.

This script evaluates calibration-first supervision construction on three labeled
benchmark datasets while withholding labels during supervision construction:

1. Ames Housing / OpenML house_prices
2. Breast Cancer Wisconsin / sklearn breast_cancer
3. UCI Wine Quality (Red) / OpenML wine-quality-red

Labels are used only after kappa has been constructed, for external validation.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, rankdata, spearmanr
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml, load_breast_cancer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import OrdinalEncoder


ScoreFn = Callable[[pd.DataFrame], pd.Series]


@dataclass(frozen=True)
class ValidationResult:
    """Container for validation metrics."""

    dataset: str
    method: str
    target: str
    metrics: Mapping[str, float]


@dataclass(frozen=True)
class ScoreArtifacts:
    """All intermediate and final outputs from supervision construction."""

    raw_data_scores: pd.DataFrame
    raw_reference_scores: pd.DataFrame
    calibrated_data_scores: pd.DataFrame
    calibrated_reference_scores: pd.DataFrame
    empirical_zscore_data_scores: pd.DataFrame
    calibration_statistics: pd.DataFrame
    kappa: pd.Series
    uncalibrated_average: pd.Series
    empirical_zscore_average: pd.Series
    borda_count_average: pd.Series
    pca_first_component: pd.Series


@dataclass(frozen=True)
class DatasetArtifacts:
    """Serializable tables produced for one validation dataset."""

    slug: str
    dataset: str
    target: str
    task: str
    k: int
    observations: pd.DataFrame
    calibration_reference: pd.DataFrame
    calibration_statistics: pd.DataFrame


def min_max_score(
    values: pd.Series,
    higher_is_better: bool = True,
) -> pd.Series:
    """Map a numeric series to [0, 1], preserving or reversing order."""
    values = pd.to_numeric(values, errors="coerce")
    v_min = values.min()
    v_max = values.max()

    if pd.isna(v_min) or pd.isna(v_max) or np.isclose(v_min, v_max):
        return pd.Series(
            np.zeros(len(values)),
            index=values.index,
        )

    scaled = (values - v_min) / (v_max - v_min)
    return scaled if higher_is_better else 1.0 - scaled


def zscore_from_reference(
    raw_scores: pd.Series,
    reference_scores: pd.Series,
) -> pd.Series:
    """Standardize raw scores using calibration-reference statistics."""
    mu = reference_scores.mean()
    sigma = reference_scores.std(ddof=0)

    if pd.isna(sigma) or np.isclose(sigma, 0.0):
        return pd.Series(
            np.zeros(len(raw_scores)),
            index=raw_scores.index,
        )

    return (raw_scores - mu) / sigma


def evaluate_all_scorers(
    frame: pd.DataFrame,
    scorers: Mapping[str, ScoreFn],
) -> pd.DataFrame:
    """
    Evaluate every scorer once.
    """
    score_columns: Dict[str, pd.Series] = {}

    for scorer_name, scorer in scorers.items():
        values = scorer(frame)

        if isinstance(values, pd.Series):
            series = values.reindex(frame.index)
        else:
            series = pd.Series(values, index=frame.index)

        score_columns[scorer_name] = pd.to_numeric(
            series,
            errors="coerce",
        ).rename(scorer_name)

    return pd.DataFrame(
        score_columns,
        index=frame.index,
    )


def compute_score_artifacts(
    data: pd.DataFrame,
    calibration_grid: pd.DataFrame,
    scorers: Mapping[str, ScoreFn],
) -> ScoreArtifacts:
    """
    Compute and retain every raw score, calibrated score, normalization
    statistic, fused score, uncalibrated average, and empirical-z-score/Borda baselines.
    """
    raw_data_scores = evaluate_all_scorers(
        data,
        scorers,
    )

    raw_reference_scores = evaluate_all_scorers(
        calibration_grid,
        scorers,
    )

    calibrated_data_scores = pd.DataFrame(
        index=data.index,
    )

    calibrated_reference_scores = pd.DataFrame(
        index=calibration_grid.index,
    )

    empirical_zscore_data_scores = pd.DataFrame(
        index=data.index,
    )

    statistics_rows: List[Dict[str, object]] = []
    included_calibrated_scorers: List[str] = []
    included_empirical_scorers: List[str] = []

    for scorer_name in scorers:
        data_scores = raw_data_scores[scorer_name]
        reference_scores = raw_reference_scores[scorer_name]

        # Proposed method: normalize against the synthetic reference space.
        reference_mean = reference_scores.mean()
        reference_std = reference_scores.std(ddof=0)

        included_in_calibrated_fusion = not (
            pd.isna(reference_std)
            or np.isclose(reference_std, 0.0)
        )

        if included_in_calibrated_fusion:
            included_calibrated_scorers.append(scorer_name)

            calibrated_data_scores[scorer_name] = (
                data_scores - reference_mean
            ) / reference_std

            calibrated_reference_scores[scorer_name] = (
                reference_scores - reference_mean
            ) / reference_std
        else:
            calibrated_data_scores[scorer_name] = np.nan
            calibrated_reference_scores[scorer_name] = np.nan

        # Baseline: normalize against the unlabeled observed dataset itself.
        empirical_mean = data_scores.mean()
        empirical_std = data_scores.std(ddof=0)

        included_in_empirical_zscore = not (
            pd.isna(empirical_std)
            or np.isclose(empirical_std, 0.0)
        )

        if included_in_empirical_zscore:
            included_empirical_scorers.append(scorer_name)

            empirical_zscore_data_scores[scorer_name] = (
                data_scores - empirical_mean
            ) / empirical_std
        else:
            empirical_zscore_data_scores[scorer_name] = np.nan

        statistics_rows.append(
            {
                "scorer": scorer_name,
                "reference_mean": (
                    float(reference_mean)
                    if not pd.isna(reference_mean)
                    else np.nan
                ),
                "reference_std": (
                    float(reference_std)
                    if not pd.isna(reference_std)
                    else np.nan
                ),
                "included_in_fusion": bool(
                    included_in_calibrated_fusion
                ),
                "empirical_data_mean": (
                    float(empirical_mean)
                    if not pd.isna(empirical_mean)
                    else np.nan
                ),
                "empirical_data_std": (
                    float(empirical_std)
                    if not pd.isna(empirical_std)
                    else np.nan
                ),
                "included_in_empirical_zscore": bool(
                    included_in_empirical_zscore
                ),
                "n_data_values": int(data_scores.notna().sum()),
                "n_reference_values": int(
                    reference_scores.notna().sum()
                ),
            }
        )

    if not included_calibrated_scorers:
        raise ValueError(
            "Fusion requires at least one scorer with nonzero variance "
            "over the calibration-reference space."
        )

    if not included_empirical_scorers:
        raise ValueError(
            "Empirical z-score averaging requires at least one scorer "
            "with nonzero variance over the observed dataset."
        )

    kappa = (
        calibrated_data_scores[included_calibrated_scorers]
        .mean(axis=1)
        .rename("kappa")
    )

    uncalibrated_average = (
        raw_data_scores
        .mean(axis=1)
        .rename("uncalibrated_average")
    )

    empirical_zscore_average = (
        empirical_zscore_data_scores[included_empirical_scorers]
        .mean(axis=1)
        .rename("empirical_zscore_average")
    )

    # Borda Rank Averaging Baseline
    n_obs = len(raw_data_scores)
    ranks = pd.DataFrame(index=data.index)
    for col in included_empirical_scorers:
        ranks[col] = (rankdata(raw_data_scores[col]) - 1.0) / max(1, n_obs - 1)
    borda_count_average = ranks.mean(axis=1).rename("borda_count_average")

    # PCA Baseline (First Principal Component on empirical z-scores)
    pca = PCA(n_components=1)
    pc1_vals = pca.fit_transform(
        empirical_zscore_data_scores[included_empirical_scorers].fillna(0)
    ).ravel()

    # Align directionality with the average scorer
    if np.corrcoef(pc1_vals, empirical_zscore_average)[0, 1] < 0:
        pc1_vals = -pc1_vals

    pca_first_component = pd.Series(
        pc1_vals,
        index=data.index,
        name="pca_first_component",
    )

    return ScoreArtifacts(
        raw_data_scores=raw_data_scores,
        raw_reference_scores=raw_reference_scores,
        calibrated_data_scores=calibrated_data_scores,
        calibrated_reference_scores=calibrated_reference_scores,
        empirical_zscore_data_scores=empirical_zscore_data_scores,
        calibration_statistics=pd.DataFrame(statistics_rows),
        kappa=kappa,
        uncalibrated_average=uncalibrated_average,
        empirical_zscore_average=empirical_zscore_average,
        borda_count_average=borda_count_average,
        pca_first_component=pca_first_component,
    )


def fuse_calibrated_scores(
    data: pd.DataFrame,
    calibration_grid: pd.DataFrame,
    scorers: Mapping[str, ScoreFn],
) -> pd.Series:
    """Compute kappa by calibrating each scorer against a synthetic ordinal reference grid."""
    return compute_score_artifacts(
        data=data,
        calibration_grid=calibration_grid,
        scorers=scorers,
    ).kappa


def average_uncalibrated_scores(
    data: pd.DataFrame,
    scorers: Mapping[str, ScoreFn],
) -> pd.Series:
    """Average raw scorer outputs without calibration."""
    return (
        evaluate_all_scorers(data, scorers)
        .mean(axis=1)
        .rename("uncalibrated_average")
    )


def top_decile_overlap(
    score: pd.Series,
    target: pd.Series,
) -> float:
    """Return overlap between the top 10% by score and target."""
    n = len(score)
    k = max(1, int(np.ceil(0.10 * n)))

    top_score = set(score.nlargest(k).index)
    top_target = set(target.nlargest(k).index)

    return len(top_score & top_target) / k


def regression_metrics(
    score: pd.Series,
    target: pd.Series,
) -> Dict[str, float]:
    """Rank-oriented metrics for continuous external validation targets."""
    valid = pd.concat(
        [score, target],
        axis=1,
    ).dropna()

    valid_score = valid.iloc[:, 0]
    valid_target = valid.iloc[:, 1]

    return {
        "spearman_rho": float(
            spearmanr(
                valid_score,
                valid_target,
            ).correlation
        ),
        "kendall_tau": float(
            kendalltau(
                valid_score,
                valid_target,
            ).correlation
        ),
        "top_10pct_overlap": float(
            top_decile_overlap(
                valid_score,
                valid_target,
            )
        ),
    }


def classification_metrics(
    score: pd.Series,
    target: pd.Series,
) -> Dict[str, float]:
    """Ranking metrics for binary external validation targets."""
    valid = pd.concat(
        [score, target],
        axis=1,
    ).dropna()

    valid_score = valid.iloc[:, 0]
    valid_target = valid.iloc[:, 1].astype(int)

    return {
        "auroc": float(
            roc_auc_score(
                valid_target,
                valid_score,
            )
        ),
        "average_precision": float(
            average_precision_score(
                valid_target,
                valid_score,
            )
        ),
        "spearman_rho": float(
            spearmanr(
                valid_score,
                valid_target,
            ).correlation
        ),
        "mean_score_positive": float(
            valid_score[valid_target == 1].mean()
        ),
        "mean_score_negative": float(
            valid_score[valid_target == 0].mean()
        ),
    }


def encode_ames_categories(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Ordinally encode categorical columns for simple scorers."""
    encoded = df.copy()

    categorical_columns = (
        encoded.select_dtypes(
            include=["object", "category", "string"]
        )
        .columns
        .tolist()
    )

    if not categorical_columns:
        return encoded

    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )

    encoded[categorical_columns] = encoder.fit_transform(
        encoded[categorical_columns].astype(str)
    )

    return encoded


def load_ames_housing() -> Tuple[pd.DataFrame, pd.Series]:
    """Load Ames Housing from OpenML."""
    bunch = fetch_openml(
        name="house_prices",
        as_frame=True,
        parser="auto",
    )

    frame = bunch.frame.copy()

    target = pd.to_numeric(
        frame.pop("SalePrice"),
        errors="coerce",
    ).rename("SalePrice")

    features = encode_ames_categories(frame)

    features = features.fillna(
        features.median(numeric_only=True)
    )

    return features, target


def make_ames_calibration_grid(
    k: int = 5,
) -> pd.DataFrame:
    """Construct a synthetic ordinal grid for housing quality and value."""
    levels = np.linspace(0.0, 1.0, k)
    rows = []

    for living_area in levels:
        for overall_quality in levels:
            for age_quality in levels:
                for garage_capacity in levels:
                    rows.append(
                        {
                            "GrLivArea": (
                                600.0
                                + living_area
                                * (3500.0 - 600.0)
                            ),
                            "TotalBsmtSF": (
                                300.0
                                + living_area
                                * (1800.0 - 300.0)
                            ),
                            "OverallQual": (
                                2.0
                                + overall_quality
                                * (10.0 - 2.0)
                            ),
                            "OverallCond": (
                                3.0
                                + overall_quality
                                * (9.0 - 3.0)
                            ),
                            "YearBuilt": (
                                1920.0
                                + age_quality
                                * (2010.0 - 1920.0)
                            ),
                            "YearRemodAdd": (
                                1950.0
                                + age_quality
                                * (2010.0 - 1950.0)
                            ),
                            "GarageCars": (
                                garage_capacity * 4.0
                            ),
                            "GarageArea": (
                                garage_capacity * 1000.0
                            ),
                        }
                    )

    return pd.DataFrame(rows)


def ames_scorers() -> Mapping[str, ScoreFn]:
    """Housing scorers constructed without SalePrice."""

    def physical_scale(df: pd.DataFrame) -> pd.Series:
        return pd.concat(
            [
                pd.to_numeric(
                    df["GrLivArea"],
                    errors="coerce",
                ),
                pd.to_numeric(
                    df["TotalBsmtSF"],
                    errors="coerce",
                ),
            ],
            axis=1,
        ).mean(axis=1)

    def quality_condition(df: pd.DataFrame) -> pd.Series:
        return pd.concat(
            [
                pd.to_numeric(
                    df["OverallQual"],
                    errors="coerce",
                ),
                pd.to_numeric(
                    df["OverallCond"],
                    errors="coerce",
                ),
            ],
            axis=1,
        ).mean(axis=1)

    def age_modernization(df: pd.DataFrame) -> pd.Series:
        return pd.concat(
            [
                pd.to_numeric(
                    df["YearBuilt"],
                    errors="coerce",
                ),
                pd.to_numeric(
                    df["YearRemodAdd"],
                    errors="coerce",
                ),
            ],
            axis=1,
        ).mean(axis=1)

    def garage_capacity(df: pd.DataFrame) -> pd.Series:
        return pd.concat(
            [
                pd.to_numeric(
                    df["GarageCars"],
                    errors="coerce",
                ),
                pd.to_numeric(
                    df["GarageArea"],
                    errors="coerce",
                ),
            ],
            axis=1,
        ).mean(axis=1)

    return {
        "physical_scale": physical_scale,
        "quality_condition": quality_condition,
        "age_modernization": age_modernization,
        "garage_capacity": garage_capacity,
    }


def load_bcw() -> Tuple[pd.DataFrame, pd.Series]:
    """Load Breast Cancer Wisconsin."""
    bunch = load_breast_cancer(as_frame=True)

    features = bunch.frame.drop(
        columns=["target"]
    )

    malignant = (
        1 - bunch.frame["target"]
    ).rename("malignant")

    return features, malignant


def make_bcw_calibration_grid(
    k: int = 5,
) -> pd.DataFrame:
    """Construct a synthetic ordinal grid for malignancy risk."""
    levels = np.linspace(0.0, 1.0, k)
    rows = []

    for size in levels:
        for shape in levels:
            for texture in levels:
                for worst_case in levels:
                    rows.append(
                        {
                            "mean radius": (
                                7.0
                                + size
                                * (28.0 - 7.0)
                            ),
                            "mean perimeter": (
                                45.0
                                + size
                                * (190.0 - 45.0)
                            ),
                            "mean area": (
                                150.0
                                + size
                                * (2500.0 - 150.0)
                            ),
                            "mean compactness": (
                                0.02
                                + shape
                                * (0.35 - 0.02)
                            ),
                            "mean concavity": (
                                shape * 0.45
                            ),
                            "mean concave points": (
                                shape * 0.20
                            ),
                            "mean texture": (
                                9.0
                                + texture
                                * (40.0 - 9.0)
                            ),
                            "mean smoothness": (
                                0.05
                                + texture
                                * (0.16 - 0.05)
                            ),
                            "mean symmetry": (
                                0.10
                                + texture
                                * (0.30 - 0.10)
                            ),
                            "worst radius": (
                                8.0
                                + worst_case
                                * (36.0 - 8.0)
                            ),
                            "worst perimeter": (
                                50.0
                                + worst_case
                                * (250.0 - 50.0)
                            ),
                            "worst area": (
                                200.0
                                + worst_case
                                * (4300.0 - 200.0)
                            ),
                            "worst concavity": (
                                worst_case * 1.25
                            ),
                            "worst concave points": (
                                worst_case * 0.30
                            ),
                        }
                    )

    return pd.DataFrame(rows)


def bcw_scorers() -> Mapping[str, ScoreFn]:
    """Malignancy-risk scorers constructed without diagnosis labels."""

    def size(df: pd.DataFrame) -> pd.Series:
        return pd.concat(
            [
                pd.to_numeric(
                    df["mean radius"],
                    errors="coerce",
                ),
                pd.to_numeric(
                    df["mean perimeter"],
                    errors="coerce",
                ),
                pd.to_numeric(
                    df["mean area"],
                    errors="coerce",
                ),
            ],
            axis=1,
        ).mean(axis=1)

    def shape_irregularity(df: pd.DataFrame) -> pd.Series:
        return pd.concat(
            [
                pd.to_numeric(
                    df["mean compactness"],
                    errors="coerce",
                ),
                pd.to_numeric(
                    df["mean concavity"],
                    errors="coerce",
                ),
                pd.to_numeric(
                    df["mean concave points"],
                    errors="coerce",
                ),
            ],
            axis=1,
        ).mean(axis=1)

    def texture_smoothness(df: pd.DataFrame) -> pd.Series:
        return pd.concat(
            [
                pd.to_numeric(
                    df["mean texture"],
                    errors="coerce",
                ),
                pd.to_numeric(
                    df["mean smoothness"],
                    errors="coerce",
                ),
                pd.to_numeric(
                    df["mean symmetry"],
                    errors="coerce",
                ),
            ],
            axis=1,
        ).mean(axis=1)

    def worst_case(df: pd.DataFrame) -> pd.Series:
        return pd.concat(
            [
                pd.to_numeric(
                    df["worst radius"],
                    errors="coerce",
                ),
                pd.to_numeric(
                    df["worst perimeter"],
                    errors="coerce",
                ),
                pd.to_numeric(
                    df["worst area"],
                    errors="coerce",
                ),
                pd.to_numeric(
                    df["worst concavity"],
                    errors="coerce",
                ),
                pd.to_numeric(
                    df["worst concave points"],
                    errors="coerce",
                ),
            ],
            axis=1,
        ).mean(axis=1)

    return {
        "size": size,
        "shape_irregularity": shape_irregularity,
        "texture_smoothness": texture_smoothness,
        "worst_case": worst_case,
    }


def load_wine_quality() -> Tuple[pd.DataFrame, pd.Series]:
    """Load Wine Quality (Red) from OpenML."""
    bunch = fetch_openml(name="wine-quality-red", version=1, as_frame=True, parser="auto")
    frame = bunch.frame.copy()

    # Standardize column names (lowercase and replace underscores with spaces)
    frame.columns = frame.columns.str.lower().str.replace('_', ' ')

    # OpenML renames the target to 'class' in version 1 of this dataset
    target_col = "class" if "class" in frame.columns else "quality"

    target = pd.to_numeric(frame.pop(target_col), errors="coerce").rename("quality")
    features = frame.apply(pd.to_numeric, errors="coerce")
    features = features.fillna(features.median())

    return features, target


def make_wine_calibration_grid(k: int = 5) -> pd.DataFrame:
    """Construct a synthetic ordinal grid for wine quality."""
    levels = np.linspace(0.0, 1.0, k)
    rows = []

    for acidity in levels:
        for additives in levels:
            for preservatives in levels:
                for profile in levels:
                    rows.append({
                        # Acidity: Lower volatile acidity, higher citric acid is better
                        "volatile acidity": 1.58 - acidity * (1.58 - 0.12),
                        "citric acid": 0.0 + acidity * (1.0 - 0.0),

                        # Additives: Lower chlorides, higher sulphates is better
                        "chlorides": 0.611 - additives * (0.611 - 0.012),
                        "sulphates": 0.33 + additives * (2.0 - 0.33),

                        # Preservatives: Lower SO2 is better
                        "free sulfur dioxide": 72.0 - preservatives * (72.0 - 1.0),
                        "total sulfur dioxide": 289.0 - preservatives * (289.0 - 6.0),

                        # Profile: Higher alcohol, lower density is better
                        "alcohol": 8.0 + profile * (14.9 - 8.0),
                        "density": 1.00369 - profile * (1.00369 - 0.99007),
                    })

    return pd.DataFrame(rows)


def wine_scorers() -> Mapping[str, ScoreFn]:
    """Wine scorers constructed using monotonic domain intuition."""

    def acidity(df: pd.DataFrame) -> pd.Series:
        v_acid_inv = -pd.to_numeric(df["volatile acidity"], errors="coerce")
        c_acid = pd.to_numeric(df["citric acid"], errors="coerce")
        return pd.concat([v_acid_inv, c_acid], axis=1).mean(axis=1)

    def additives(df: pd.DataFrame) -> pd.Series:
        chlor_inv = -pd.to_numeric(df["chlorides"], errors="coerce")
        sulph = pd.to_numeric(df["sulphates"], errors="coerce")
        return pd.concat([chlor_inv, sulph], axis=1).mean(axis=1)

    def preservatives(df: pd.DataFrame) -> pd.Series:
        free_inv = -pd.to_numeric(df["free sulfur dioxide"], errors="coerce")
        total_inv = -pd.to_numeric(df["total sulfur dioxide"], errors="coerce")
        return pd.concat([free_inv, total_inv], axis=1).mean(axis=1)

    def profile(df: pd.DataFrame) -> pd.Series:
        alc = pd.to_numeric(df["alcohol"], errors="coerce")
        dens_inv = -pd.to_numeric(df["density"], errors="coerce")
        return pd.concat([alc, dens_inv], axis=1).mean(axis=1)

    return {
        "acidity": acidity,
        "additives": additives,
        "preservatives": preservatives,
        "profile": profile,
    }


def validation_results_from_artifacts(
    dataset: str,
    target_name: str,
    target: pd.Series,
    artifacts: ScoreArtifacts,
    task: str,
) -> List[ValidationResult]:
    """
    Evaluate calibrated fusion, empirical baselines (z-score, borda, pca), raw averaging,
    and every individual scorer.
    """
    if task == "regression":
        metric_function = regression_metrics
    elif task == "classification":
        metric_function = classification_metrics
    else:
        raise ValueError(
            "task must be either 'regression' or 'classification'."
        )

    results = [
        ValidationResult(
            dataset=dataset,
            method="calibrated_fusion_kappa",
            target=target_name,
            metrics=metric_function(artifacts.kappa, target),
        ),
        ValidationResult(
            dataset=dataset,
            method="empirical_zscore_average",
            target=target_name,
            metrics=metric_function(artifacts.empirical_zscore_average, target),
        ),
        ValidationResult(
            dataset=dataset,
            method="borda_count_average",
            target=target_name,
            metrics=metric_function(artifacts.borda_count_average, target),
        ),
        ValidationResult(
            dataset=dataset,
            method="pca_first_component",
            target=target_name,
            metrics=metric_function(artifacts.pca_first_component, target),
        ),
        ValidationResult(
            dataset=dataset,
            method="uncalibrated_average",
            target=target_name,
            metrics=metric_function(artifacts.uncalibrated_average, target),
        ),
    ]

    for scorer_name in artifacts.raw_data_scores.columns:
        results.append(
            ValidationResult(
                dataset=dataset,
                method=f"single_source_{scorer_name}",
                target=target_name,
                metrics=metric_function(
                    artifacts.raw_data_scores[scorer_name],
                    target,
                ),
            )
        )

    return results


def build_dataset_artifacts(
    *,
    slug: str,
    dataset: str,
    target_name: str,
    task: str,
    k: int,
    data: pd.DataFrame,
    target: pd.Series,
    calibration_grid: pd.DataFrame,
    scores: ScoreArtifacts,
) -> DatasetArtifacts:
    """Build wide, plot-friendly tables containing all features and scores."""
    observations = data.copy().add_prefix("feature_")

    observations.insert(0, "row_id", data.index)
    observations[target_name] = target.reindex(data.index)

    observations["kappa"] = scores.kappa.reindex(data.index)
    observations["uncalibrated_average"] = scores.uncalibrated_average.reindex(data.index)
    observations["empirical_zscore_average"] = scores.empirical_zscore_average.reindex(data.index)
    observations["borda_count_average"] = scores.borda_count_average.reindex(data.index)
    observations["pca_first_component"] = scores.pca_first_component.reindex(data.index)

    for scorer_name in scores.raw_data_scores.columns:
        observations[f"raw_scorer_{scorer_name}"] = scores.raw_data_scores[scorer_name].reindex(data.index)
        observations[f"calibrated_scorer_{scorer_name}"] = scores.calibrated_data_scores[scorer_name].reindex(data.index)
        observations[f"empirical_zscore_scorer_{scorer_name}"] = scores.empirical_zscore_data_scores[scorer_name].reindex(data.index)

    observations = observations.reset_index(drop=True)

    calibration_reference = calibration_grid.copy().add_prefix("feature_")
    calibration_reference.insert(0, "calibration_id", np.arange(len(calibration_reference)))

    for scorer_name in scores.raw_reference_scores.columns:
        calibration_reference[f"raw_reference_scorer_{scorer_name}"] = (
            scores.raw_reference_scores[scorer_name].reindex(calibration_grid.index).to_numpy()
        )
        calibration_reference[f"calibrated_reference_scorer_{scorer_name}"] = (
            scores.calibrated_reference_scores[scorer_name].reindex(calibration_grid.index).to_numpy()
        )

    return DatasetArtifacts(
        slug=slug, dataset=dataset, target=target_name, task=task, k=k,
        observations=observations, calibration_reference=calibration_reference,
        calibration_statistics=(scores.calibration_statistics.copy()),
    )


def run_ames_validation_with_artifacts(k: int = 5) -> Tuple[List[ValidationResult], DatasetArtifacts]:
    """Run Ames validation and retain artifacts."""
    x, y = load_ames_housing()
    scorers = ames_scorers()
    calibration_grid = make_ames_calibration_grid(k=k)

    score_artifacts = compute_score_artifacts(data=x, calibration_grid=calibration_grid, scorers=scorers)

    results = validation_results_from_artifacts(
        dataset="Ames Housing", target_name="SalePrice", target=y,
        artifacts=score_artifacts, task="regression",
    )

    dataset_artifacts = build_dataset_artifacts(
        slug="ames_housing", dataset="Ames Housing", target_name="SalePrice",
        task="regression", k=k, data=x, target=y,
        calibration_grid=calibration_grid, scores=score_artifacts,
    )
    return results, dataset_artifacts


def run_bcw_validation_with_artifacts(k: int = 5) -> Tuple[List[ValidationResult], DatasetArtifacts]:
    """Run BCW validation and retain artifacts."""
    x, y = load_bcw()
    scorers = bcw_scorers()
    calibration_grid = make_bcw_calibration_grid(k=k)

    score_artifacts = compute_score_artifacts(data=x, calibration_grid=calibration_grid, scorers=scorers)

    results = validation_results_from_artifacts(
        dataset="Breast Cancer Wisconsin", target_name="malignant", target=y,
        artifacts=score_artifacts, task="classification",
    )

    dataset_artifacts = build_dataset_artifacts(
        slug="breast_cancer_wisconsin", dataset="Breast Cancer Wisconsin", target_name="malignant",
        task="classification", k=k, data=x, target=y,
        calibration_grid=calibration_grid, scores=score_artifacts,
    )
    return results, dataset_artifacts


def run_wine_validation_with_artifacts(k: int = 5) -> Tuple[List[ValidationResult], DatasetArtifacts]:
    """Run Wine Quality validation and retain artifacts."""
    x, y = load_wine_quality()
    scorers = wine_scorers()
    calibration_grid = make_wine_calibration_grid(k=k)

    score_artifacts = compute_score_artifacts(data=x, calibration_grid=calibration_grid, scorers=scorers)

    results = validation_results_from_artifacts(
        dataset="Wine Quality", target_name="quality", target=y,
        artifacts=score_artifacts, task="regression",
    )

    dataset_artifacts = build_dataset_artifacts(
        slug="wine_quality", dataset="Wine Quality", target_name="quality",
        task="regression", k=k, data=x, target=y,
        calibration_grid=calibration_grid, scores=score_artifacts,
    )
    return results, dataset_artifacts


def results_to_frame(results: Iterable[ValidationResult]) -> pd.DataFrame:
    """Convert validation results into a flat dataframe."""
    rows = []
    for result in results:
        row = {
            "dataset": result.dataset,
            "method": result.method,
            "target": result.target,
        }
        row.update(result.metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def run_all_validations_with_artifacts(k: int = 5) -> Tuple[pd.DataFrame, Dict[str, DatasetArtifacts]]:
    """Run all experiments and retain complete per-dataset artifacts."""
    results: List[ValidationResult] = []
    datasets: Dict[str, DatasetArtifacts] = {}

    ames_results, ames_artifacts = run_ames_validation_with_artifacts(k=k)
    results.extend(ames_results)
    datasets[ames_artifacts.slug] = ames_artifacts

    bcw_results, bcw_artifacts = run_bcw_validation_with_artifacts(k=k)
    results.extend(bcw_results)
    datasets[bcw_artifacts.slug] = bcw_artifacts

    wine_results, wine_artifacts = run_wine_validation_with_artifacts(k=k)
    results.extend(wine_results)
    datasets[wine_artifacts.slug] = wine_artifacts

    return results_to_frame(results), datasets


def dataframe_to_json_records(frame: pd.DataFrame) -> List[Dict[str, object]]:
    """Convert a dataframe to JSON-compatible records."""
    serialized = frame.to_json(orient="records", double_precision=15)
    return json.loads(serialized)


def save_validation_outputs(
    summary: pd.DataFrame,
    datasets: Mapping[str, DatasetArtifacts],
    output_dir: Path | str,
    k: int,
    write_csv: bool = True,
) -> Path:
    """Save one complete JSON artifact and optional plot-friendly CSV files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, object] = {
        "schema_version": 1,
        "experiment": "cross_domain_validation",
        "calibration_resolution_k": int(k),
        "summary_metrics": dataframe_to_json_records(summary),
        "datasets": {},
    }

    dataset_payloads: Dict[str, object] = {}

    for slug, artifacts in datasets.items():
        dataset_metrics = summary.loc[
            summary["dataset"] == artifacts.dataset
        ].reset_index(drop=True)

        dataset_payloads[slug] = {
            "dataset": artifacts.dataset,
            "target": artifacts.target,
            "task": artifacts.task,
            "k": int(artifacts.k),
            "n_observations": int(len(artifacts.observations)),
            "n_calibration_points": int(len(artifacts.calibration_reference)),
            "metrics": dataframe_to_json_records(dataset_metrics),
            "calibration_statistics": dataframe_to_json_records(artifacts.calibration_statistics),
            "observations": dataframe_to_json_records(artifacts.observations),
            "calibration_reference": dataframe_to_json_records(artifacts.calibration_reference),
        }

        if write_csv:
            artifacts.observations.to_csv(output_path / f"{slug}_observations.csv", index=False)
            artifacts.calibration_reference.to_csv(output_path / f"{slug}_calibration_reference.csv", index=False)
            artifacts.calibration_statistics.to_csv(output_path / f"{slug}_calibration_statistics.csv", index=False)

    payload["datasets"] = dataset_payloads

    if write_csv:
        summary.to_csv(output_path / "validation_metrics.csv", index=False)

    json_path = output_path / f"cross_domain_validation_k{k}.json"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False, allow_nan=False)

    return json_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run cross-domain calibration validation with standard unsupervised baselines."
    )
    parser.add_argument("--k", type=int, default=5, help="Number of calibration levels per dimension.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/cross_domain_validation"))
    parser.add_argument("--json-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    summary, dataset_artifacts = run_all_validations_with_artifacts(k=args.k)

    json_path = save_validation_outputs(
        summary=summary,
        datasets=dataset_artifacts,
        output_dir=args.output_dir,
        k=args.k,
        write_csv=not args.json_only,
    )

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)

    print(summary.round(4).to_string(index=False))
    print(f"\nSaved complete JSON artifact to: {json_path}")
    if not args.json_only:
        print(f"Saved companion CSV files to: {args.output_dir}")
