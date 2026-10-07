# Align Before You Combine: A Calibration-First Framework for Supervision Without Ground Truth

This repository contains the reference implementation and experimental code for *Align Before You Combine: A Calibration-First Framework for Supervision Without Ground Truth*.

## Overview

Many applications require constructing supervision signals from heterogeneous evidence sources without access to ground-truth labels or a shared annotation space. Directly combining scorer outputs can be problematic when those outputs differ in scale, offset, or interpretation.

Our framework takes a **calibration-first** approach. We construct an externally defined synthetic ordinal reference space from ordered domain features and evaluate each scorer over that common reference. Scorer-specific calibration statistics are then estimated from the reference space and used to place heterogeneous scorer outputs on a common unitless scale before fusion.

The framework is scorer-agnostic: scorers may be heuristic, simulation-based, statistical, machine-learning, or language-model based, provided that they can be evaluated over the calibration reference space and their directionality can be aligned.

## Repository Structure

```text
.
├── synthetic_homes/
│   ├── build_calibration_space.py
│   ├── run_synthetic_homes.py
│   ├── text_score_adapter.py
│   └── k_sensitivity/
├── other_domain_validation/
│   ├── validations.py
│   ├── statistics.py
│   ├── results/
│   └── statistical_analysis/
├── irt_testing/
│   ├── calibration_space_and_irt.py
│   ├── data/
│   └── irt_exp_results.md
├── dataset_licenses.md
├── requirements.txt
└── README.md
```

`synthetic_homes/` contains the residential energy-retrofit case study and calibration-resolution analysis. `other_domain_validation/` contains the cross-domain benchmark experiments and paired bootstrap analysis reported in the main paper. `irt_testing/` contains the psychometric validation experiments reported in the appendix.

## Setup

Create and activate a Python virtual environment, then install the required dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The residential case study additionally requires an OpenAI API key for the language-model scorer:

```bash
export OPENAI_API_KEY="..."
```

The API key should not be committed to the repository.

## Residential Energy-Retrofit Case Study

The Synthetic Homes case study constructs supervision for **residential retrofit priority** from four heterogeneous scoring channels:

- text-based HVAC evidence;
- structured HVAC evidence;
- text-based insulation evidence; and
- structured insulation evidence.

The shared calibration space is formed from four ordered building characteristics:

1. heating COP;
2. cooling COP;
3. roof R-value; and
4. wall R-value.

At the default resolution of five levels per feature, this produces a \(5^4 = 625\)-point synthetic ordinal reference space.

### Constructing the Calibration Space

To construct the 625-point reference space:

```bash
python3 synthetic_homes/build_calibration_space.py
```

### Synthetic Homes Data

The residential case study and calibration-resolution experiments use the Synthetic Homes pipeline described in the accompanying work. The dataset is generated externally using the [Synthetic Homes repository](https://github.com/Lafayette-EshbaughSilveyra-Group/synthetic-homes) and is not included in this repository.

Follow the Synthetic Homes repository instructions to generate the residential dataset and EnergyPlus outputs, then provide the resulting dataset directory to the scripts in this repository using the `--dataset-dir` argument.

### Running the Case Study

The reported text scorer is implemented in `synthetic_homes/text_score_adapter.py`. To run the case-study pipeline:

```bash
python3 synthetic_homes/run_synthetic_homes.py \
    --notes synthetic_homes/note_templates.json \
    --text-scorer synthetic_homes.text_score_adapter:score_text \
    --output-dir synthetic_homes/results
```

A mechanical smoke test can instead be run with:

```bash
python3 synthetic_homes/run_synthetic_homes.py \
    --notes synthetic_homes/note_templates.json \
    --text-scorer demo \
    --output-dir synthetic_homes/results
```

The demo scorer is provided only for testing the pipeline and does not reproduce the reported experimental results.

## Calibration-Resolution Sensitivity

The paper also evaluates sensitivity to the resolution \(k\) of the synthetic calibration space.

The high-resolution Synthetic Homes experiment uses a \(k=15\) master grid, corresponding to

\[
15^4 = 50{,}625
\]

calibration points. Lower-resolution spaces are obtained by selecting approximately evenly spaced levels from this master grid.

This experiment requires the companion Synthetic Homes repository and EnergyPlus.

From the repository root, generate the \(k=15\) master grid with:

```bash
python3 synthetic_homes/k_sensitivity/generate_calibration_grid.py \
    --synthetic-homes-root ../synthetic-homes \
    --epw ../synthetic-homes/weather/KMSP.epw \
    --output-dir synthetic_homes/k_sensitivity/calibration_grid \
    --k 15 \
    --max-workers 8
```

Because this step runs 50,625 EnergyPlus simulations, execution on a multicore or high-performance computing system is recommended. Interrupted runs can be resumed by adding `--resume`.

Once the calibration grid and target Synthetic Homes dataset are available, run:

```bash
python3 synthetic_homes/k_sensitivity/test_k_sensitivity.py \
    --calibration-dir synthetic_homes/k_sensitivity/calibration_grid \
    --dataset-dir ../synthetic-homes/path/to/dataset \
    --output-dir synthetic_homes/k_sensitivity/k_sensitivity_results
```

The analysis evaluates

\[
k \in \{2,3,5,7,9,11,13,15\},
\]

using \(k=15\) as the high-resolution reference.

## Cross-Domain Validation

The framework is additionally evaluated on three labeled benchmark datasets:

- **Ames Housing** — regression, evaluated primarily using Spearman rank correlation;
- **Breast Cancer Wisconsin** — classification, evaluated primarily using AUROC; and
- **Wine Quality** — regression, evaluated primarily using Spearman rank correlation.

Ground-truth outcomes are **not used to construct the supervision signal**. They are used only afterward to evaluate the resulting signal.

The calibrated fusion method is compared against uncalibrated averaging and sample-dependent unsupervised baselines.

### Run the Benchmark Experiments

From the repository root:

```bash
python3 other_domain_validation/validations.py \
    --k 5 \
    --output-dir other_domain_validation/results/cross_domain_validation
```

This produces per-dataset observation files, calibration references, calibration statistics, and summary metrics.

### Run the Statistical Analysis

The paired bootstrap analysis reported in the paper can then be reproduced with:

```bash
python3 other_domain_validation/statistics.py \
    --input-dir other_domain_validation/results/cross_domain_validation \
    --output-dir other_domain_validation/statistical_analysis \
    --bootstrap 50000 \
    --seed 20260922
```

All method comparisons use paired bootstrap resampling so that each method is evaluated on the same resampled observations within each replicate. Primary comparisons are adjusted across the three benchmark datasets using Bonferroni correction.

Machine-readable results are written to `other_domain_validation/statistical_analysis/`.

## Psychometric Validation

The appendix evaluates the framework in a psychometric setting using the experiments in `irt_testing/`.

The primary experiment script is:

```text
irt_testing/calibration_space_and_irt.py
```

and the corresponding experimental results are documented in:

```text
irt_testing/irt_exp_results.md
```

The datasets used by these experiments are stored under `irt_testing/data/`.

To run the psychometric validation from the repository root:

```bash
python3 irt_testing/calibration_space_and_irt.py
```

These experiments provide an additional evaluation of calibration-first fusion outside the main cross-domain benchmark suite and are reported in the appendix rather than the primary experimental results.

## Data

The cross-domain datasets are obtained from OpenML, scikit-learn, or the UCI Machine Learning Repository at runtime. Dataset sources and licenses are documented in [`dataset_licenses.md`](dataset_licenses.md).

The Synthetic Homes calibration-resolution experiment relies on the companion Synthetic Homes pipeline for building simulation and dataset generation.

Psychometric-validation data used for the appendix experiments are contained under `irt_testing/data/`.

## Reproducibility

The repository includes the random seeds, calibration resolutions, scorer definitions, statistical-analysis settings, and precomputed outputs used for the reported experiments where practical. These artifacts permit inspection of the reported results without rerunning computationally expensive simulations or bootstrap analyses.

The principal reproduction paths are:

- `synthetic_homes/` for the residential energy-retrofit case study;
- `synthetic_homes/k_sensitivity/` for calibration-resolution sensitivity;
- `other_domain_validation/` for the three cross-domain benchmark experiments and paired statistical analysis; and
- `irt_testing/` for the psychometric validation reported in the appendix.