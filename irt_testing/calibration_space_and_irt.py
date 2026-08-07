"""
calibration_space_and_psychometrics.py

Cross-domain psychometric validation for Align Before You Combine.

This script runs calibration-first fusion on three OpenPsychometrics-style
datasets contained in one archive:

    data/Archive.zip
        BIG5/data.csv
        HEXACO/data.csv
        MGKT_data/data.csv

Datasets:
    1. BIG5
       - Five personality domains as modalities.
       - Target: first principal component of item-level responses.

    2. HEXACO
       - Six personality domains as modalities.
       - Target: first principal component of item-level responses.

    3. MGKT
       - General-knowledge domains as modalities.
       - Target: first principal component of question-level scores.

The goal is to test whether calibration-first fusion produces a latent ranking
comparable to a standard item-level latent-variable summary.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping
import zipfile

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


ScoreFn = Callable[[pd.DataFrame], pd.Series]


@dataclass(frozen=True)
class ValidationResult:
    dataset: str
    method: str
    target: str
    metrics: Mapping[str, float]


def zscore_from_reference(raw_scores: pd.Series, reference_scores: pd.Series) -> pd.Series:
    mu = reference_scores.mean()
    sigma = reference_scores.std(ddof=0)

    if pd.isna(sigma) or np.isclose(sigma, 0.0):
        return pd.Series(np.zeros(len(raw_scores)), index=raw_scores.index)

    return (raw_scores - mu) / sigma


def fuse_calibrated_scores(
    data: pd.DataFrame,
    calibration_grid: pd.DataFrame,
    scorers: Mapping[str, ScoreFn],
) -> pd.Series:
    calibrated_scores: List[pd.Series] = []

    for scorer_name, scorer in scorers.items():
        raw_data_scores = scorer(data)
        raw_reference_scores = scorer(calibration_grid)
        calibrated = zscore_from_reference(raw_data_scores, raw_reference_scores)
        calibrated_scores.append(calibrated.rename(scorer_name))

    return pd.concat(calibrated_scores, axis=1).mean(axis=1).rename("kappa")


def average_uncalibrated_scores(
    data: pd.DataFrame,
    scorers: Mapping[str, ScoreFn],
) -> pd.Series:
    raw_scores = [scorer(data).rename(name) for name, scorer in scorers.items()]
    return pd.concat(raw_scores, axis=1).mean(axis=1).rename("uncalibrated_average")


def top_decile_overlap(score: pd.Series, target: pd.Series) -> float:
    n = len(score)
    k = max(1, int(np.ceil(0.10 * n)))

    top_score = set(score.nlargest(k).index)
    top_target = set(target.nlargest(k).index)

    return len(top_score & top_target) / k


def regression_metrics(score: pd.Series, target: pd.Series) -> Dict[str, float]:
    valid = pd.concat([score, target], axis=1).dropna()

    if len(valid) < 2:
        return {
            "spearman_rho": np.nan,
            "kendall_tau": np.nan,
            "top_10pct_overlap": np.nan,
        }

    valid_score = valid.iloc[:, 0]
    valid_target = valid.iloc[:, 1]

    rho = spearmanr(valid_score, valid_target).correlation
    tau = kendalltau(valid_score, valid_target).correlation

    # PCA direction is arbitrary. Use absolute rank agreement.
    return {
        "spearman_rho": float(abs(rho)),
        "kendall_tau": float(abs(tau)),
        "top_10pct_overlap": float(
            max(
                top_decile_overlap(valid_score, valid_target),
                top_decile_overlap(-valid_score, valid_target),
            )
        ),
    }


def read_csv_from_archive_or_folder(
    archive_or_folder: str | Path,
    internal_path: str,
    sep: str,
) -> pd.DataFrame:
    archive_or_folder = Path(archive_or_folder)

    if archive_or_folder.is_file() and archive_or_folder.suffix.lower() == ".zip":
        with zipfile.ZipFile(archive_or_folder) as archive:
            with archive.open(internal_path) as handle:
                return pd.read_csv(handle, sep=sep, low_memory=False)

    path = archive_or_folder / internal_path
    if not path.exists():
        raise FileNotFoundError(f"Could not find {path}")

    return pd.read_csv(path, sep=sep, low_memory=False)


def clean_likert_matrix(
    df: pd.DataFrame,
    item_columns: Iterable[str],
    min_value: int,
    max_value: int,
) -> pd.DataFrame:
    items = df[list(item_columns)].copy()

    for col in items.columns:
        items[col] = pd.to_numeric(items[col], errors="coerce")

    items = items.replace(0, np.nan)
    items = items.where((items >= min_value) & (items <= max_value))
    items = items.dropna(axis=0)

    return items.astype(float)


def reverse_score(
    series: pd.Series,
    min_value: int,
    max_value: int,
) -> pd.Series:
    return (min_value + max_value) - series


def first_pc_score(item_matrix: pd.DataFrame) -> pd.Series:
    valid = item_matrix.dropna(axis=0)

    scaled = StandardScaler().fit_transform(valid)
    pc1 = PCA(n_components=1).fit_transform(scaled).ravel()

    return pd.Series(pc1, index=valid.index, name="item_pc1")


def make_scorers(modality_columns: Iterable[str]) -> Mapping[str, ScoreFn]:
    scorers: Dict[str, ScoreFn] = {}

    for modality_name in modality_columns:

        def scorer(df: pd.DataFrame, col: str = modality_name) -> pd.Series:
            return pd.to_numeric(df[col], errors="coerce")

        scorers[modality_name] = scorer

    return scorers


def make_calibration_grid(
    modality_ranges: Mapping[str, tuple[float, float]],
    k: int = 5,
) -> pd.DataFrame:
    levels_by_modality = {
        modality: np.linspace(lo, hi, k)
        for modality, (lo, hi) in modality_ranges.items()
    }

    rows = []
    names = list(levels_by_modality.keys())

    for values in product(*[levels_by_modality[name] for name in names]):
        rows.append(dict(zip(names, values)))

    return pd.DataFrame(rows)


def results_to_frame(results: Iterable[ValidationResult]) -> pd.DataFrame:
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


def evaluate_dataset(
    dataset_name: str,
    modality_scores: pd.DataFrame,
    item_matrix: pd.DataFrame,
    modality_ranges: Mapping[str, tuple[float, float]],
    k: int = 5,
) -> pd.DataFrame:
    target = first_pc_score(item_matrix)

    common_index = modality_scores.index.intersection(target.index)
    modality_scores = modality_scores.loc[common_index]
    target = target.loc[common_index]

    scorers = make_scorers(modality_scores.columns)
    calibration_grid = make_calibration_grid(modality_ranges, k=k)

    kappa = fuse_calibrated_scores(modality_scores, calibration_grid, scorers)
    uncalibrated = average_uncalibrated_scores(modality_scores, scorers)
    total_score = modality_scores.mean(axis=1).rename("total_score")

    results: List[ValidationResult] = [
        ValidationResult(
            dataset=dataset_name,
            method="calibrated_fusion_kappa",
            target="item_pc1",
            metrics=regression_metrics(kappa, target),
        ),
        ValidationResult(
            dataset=dataset_name,
            method="uncalibrated_average",
            target="item_pc1",
            metrics=regression_metrics(uncalibrated, target),
        ),
        ValidationResult(
            dataset=dataset_name,
            method="total_modality_score",
            target="item_pc1",
            metrics=regression_metrics(total_score, target),
        ),
    ]

    for name, scorer in scorers.items():
        results.append(
            ValidationResult(
                dataset=dataset_name,
                method=f"single_source_{name}",
                target="item_pc1",
                metrics=regression_metrics(scorer(modality_scores), target),
            )
        )

    summary = results_to_frame(results)

    summary["n_subjects"] = len(common_index)
    summary["n_items"] = item_matrix.shape[1]
    summary["n_modalities"] = modality_scores.shape[1]
    summary["calibration_grid_size"] = len(calibration_grid)

    return summary


def prepare_big5(
    archive_or_folder: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, Mapping[str, tuple[float, float]]]:
    df = read_csv_from_archive_or_folder(
        archive_or_folder,
        "BIG5/data.csv",
        sep="\t",
    )

    factors = {
        "extraversion": [f"E{i}" for i in range(1, 11)],
        "neuroticism": [f"N{i}" for i in range(1, 11)],
        "agreeableness": [f"A{i}" for i in range(1, 11)],
        "conscientiousness": [f"C{i}" for i in range(1, 11)],
        "openness": [f"O{i}" for i in range(1, 11)],
    }

    reverse_items = {
        "E2", "E4", "E6", "E8", "E10",
        "N2", "N4",
        "A1", "A3", "A5", "A7",
        "C2", "C4", "C6", "C8",
        "O2", "O4", "O6",
    }

    item_columns = [col for cols in factors.values() for col in cols]
    items = clean_likert_matrix(df, item_columns, min_value=1, max_value=5)

    keyed_items = items.copy()
    for col in reverse_items:
        if col in keyed_items.columns:
            keyed_items[col] = reverse_score(keyed_items[col], 1, 5)

    modality_scores = pd.DataFrame(index=keyed_items.index)
    for factor, columns in factors.items():
        modality_scores[factor] = keyed_items[columns].mean(axis=1)

    modality_ranges = {factor: (1.0, 5.0) for factor in factors}

    return modality_scores, keyed_items, modality_ranges


def prepare_hexaco(
    archive_or_folder: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, Mapping[str, tuple[float, float]]]:
    df = read_csv_from_archive_or_folder(
        archive_or_folder,
        "HEXACO/data.csv",
        sep="\t",
    )

    domain_prefixes = {
        "honesty_humility": ["HSinc", "HFair", "HGree", "HMode"],
        "emotionality": ["EFear", "EAnxi", "EDepe", "ESent"],
        "extraversion": ["XExpr", "XSocB", "XSoci", "XLive"],
        "agreeableness": ["AForg", "AGent", "AFlex", "APati"],
        "conscientiousness": ["COrga", "CDili", "CPerf", "CPrud"],
        "openness": ["OAesA", "OInqu", "OCrea", "OUnco"],
    }

    item_columns = []
    for prefixes in domain_prefixes.values():
        for prefix in prefixes:
            item_columns.extend([f"{prefix}{i}" for i in range(1, 11)])

    items = clean_likert_matrix(df, item_columns, min_value=1, max_value=7)

    # Full HEXACO keying is more detailed than the file names alone reveal.
    # For this validation, use domain-level raw response summaries and let the
    # item-level PCA target absorb the dominant latent direction.
    keyed_items = items.copy()

    modality_scores = pd.DataFrame(index=keyed_items.index)
    for domain, prefixes in domain_prefixes.items():
        columns = []
        for prefix in prefixes:
            columns.extend([f"{prefix}{i}" for i in range(1, 11)])
        modality_scores[domain] = keyed_items[columns].mean(axis=1)

    modality_ranges = {domain: (1.0, 7.0) for domain in domain_prefixes}

    return modality_scores, keyed_items, modality_ranges


def prepare_mgkt(
    archive_or_folder: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, Mapping[str, tuple[float, float]]]:
    df = read_csv_from_archive_or_folder(
        archive_or_folder,
        "MGKT_data/data.csv",
        sep=",",
    )

    question_columns = [f"Q{i}S" for i in range(1, 33)]

    items = df[question_columns].copy()
    for col in question_columns:
        items[col] = pd.to_numeric(items[col], errors="coerce")

    # QXS is right answers selected minus wrong answers selected.
    # Valid values can range approximately from -5 to 5.
    items = items.where((items >= -5) & (items <= 5))
    items = items.dropna(axis=0)

    domains = {
        "arts_literature_culture": ["Q1S", "Q2S", "Q3S", "Q4S", "Q17S", "Q19S", "Q25S", "Q29S", "Q32S"],
        "health_biology_medicine": ["Q5S", "Q6S", "Q27S", "Q31S"],
        "geopolitics_history": ["Q9S", "Q10S", "Q11S", "Q12S", "Q23S", "Q24S"],
        "technology_computing": ["Q13S", "Q14S", "Q15S", "Q16S", "Q21S", "Q22S", "Q28S", "Q30S"],
        "practical_general_knowledge": ["Q7S", "Q8S", "Q18S", "Q20S", "Q26S"],
    }

    modality_scores = pd.DataFrame(index=items.index)
    for domain, columns in domains.items():
        modality_scores[domain] = items[columns].mean(axis=1)

    modality_ranges = {domain: (-5.0, 5.0) for domain in domains}

    return modality_scores, items, modality_ranges


def run_all_validations(
    archive_or_folder: str | Path = "data",
    k: int = 5,
) -> pd.DataFrame:
    results = []

    big5_scores, big5_items, big5_ranges = prepare_big5(archive_or_folder)
    results.append(
        evaluate_dataset(
            dataset_name="BIG5",
            modality_scores=big5_scores,
            item_matrix=big5_items,
            modality_ranges=big5_ranges,
            k=k,
        )
    )

    hexaco_scores, hexaco_items, hexaco_ranges = prepare_hexaco(archive_or_folder)
    results.append(
        evaluate_dataset(
            dataset_name="HEXACO",
            modality_scores=hexaco_scores,
            item_matrix=hexaco_items,
            modality_ranges=hexaco_ranges,
            k=k,
        )
    )

    mgkt_scores, mgkt_items, mgkt_ranges = prepare_mgkt(archive_or_folder)
    results.append(
        evaluate_dataset(
            dataset_name="MGKT",
            modality_scores=mgkt_scores,
            item_matrix=mgkt_items,
            modality_ranges=mgkt_ranges,
            k=k,
        )
    )

    return pd.concat(results, axis=0, ignore_index=True)


if __name__ == "__main__":
    summary = run_all_validations(
        archive_or_folder="data",
        k=5,
    )

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 240)

    print(summary.round(4).to_string(index=False))