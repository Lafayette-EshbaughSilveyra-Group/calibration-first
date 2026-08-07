# Align Before You Combine: A Calibration-First Framework for Supervision Without Ground Truth

This repository contains the reference implementation and experimental code accompanying the forthcoming paper *Align Before You Combine: A Calibration-First Framework for Supervision Without Ground Truth*.

## Framework Synopsis

Many real-world problems require constructing supervision signals from heterogeneous sources of evidence without access to ground-truth labels. Directly combining outputs from different scorers can be problematic when those scorers differ in scale, offset, or interpretation.

This framework introduces a calibration-first approach to supervision construction. A synthetic ordinal reference space is first constructed from domain knowledge, and subset-specific scorers are independently evaluated over this shared semantic reference. Their outputs are then standardized using calibration statistics derived from the reference space, placing heterogeneous scorers on a common unitless scale before fusion.

The framework is scorer-agnostic and can accommodate heuristic, simulation-based, statistical, machine-learning, and large language model (LLM) scorers, provided that each scorer can be evaluated over the calibration reference space and that scorer directionality is aligned.

For a complete theoretical treatment, see the accompanying paper (forthcoming).

## Experiments

### Synthetic Homes Testing (Sec. 4.1)

Implements the residential energy retrofit case study presented in the paper. The framework is used to construct supervision signals from textual inspection reports and EnergyPlus simulation outputs without requiring retrofit-priority labels. This directory contains the calibration pipeline, subset-specific scorers, fusion procedures, ablation studies, agreement and conflict experiments, and calibration-resolution sensitivity testing.

#### $k$ Sensitivity Testing

The sensitivity of the calibration-grid resolution $k$ is evaluated using the Synthetic Homes case study. A $k=15$ master calibration grid is generated once, and lower-resolution grids are obtained by selecting approximately evenly spaced subsets of the master grid.

To reproduce the experiment:

1. Clone the Synthetic Homes repository ([`Lafayette-EshbaughSilveyra-Group/synthetic-homes`](https://github.com/Lafayette-EshbaughSilveyra-Group/synthetic-homes)) alongside this repository.

2. Follow the setup instructions in the Synthetic Homes `README.md`, including installation of its Python dependencies and EnergyPlus, and activate the corresponding virtual environment.

3. From this repository, generate the $k=15$ master calibration grid using `generate_calibration_grid.py`. For example:

```zsh
python3 synthetic_homes/k_sensitivity/generate_calibration_grid.py \
    --synthetic-homes-root ../synthetic-homes \
    --epw ../synthetic-homes/weather/KMSP.epw \
    --output-dir synthetic_homes/k_sensitivity/calibration_grid \
    --k 15 \
    --max-workers 8
```

The generator constructs $15^4 = 50{,}625$ EnergyPlus calibration points. Because this step is computationally intensive, execution on a remote or high-performance computing system is recommended.

The generator stores compact calibration artifacts under:

```text
synthetic_homes/k_sensitivity/calibration_grid/
└── data/
    ├── calibration_meta.json
    ├── calibration_levels.json
    ├── summary_stats.json
    └── calibration_runtime.json
```

Raw hourly EnergyPlus outputs are summarized during execution and are not retained by default. If execution is interrupted, rerun the same command with `--resume` to continue from the successfully completed calibration points.

4. Run the Synthetic Homes pipeline normally, following the instructions in the Synthetic Homes repository, to produce the target Synthetic Homes dataset used for evaluation.

5. Run the $k$-sensitivity analysis. If the generated Synthetic Homes dataset has been copied to `synthetic_homes/k_sensitivity/dataset/`, the default invocation is:

```zsh
python3 synthetic_homes/k_sensitivity/test_k_sensitivity.py
```

Alternatively, the dataset may remain in the Synthetic Homes repository and be supplied explicitly:

```zsh
python3 synthetic_homes/k_sensitivity/test_k_sensitivity.py \
    --calibration-dir synthetic_homes/k_sensitivity/calibration_grid \
    --dataset-dir ../synthetic-homes/path/to/dataset \
    --output-dir synthetic_homes/k_sensitivity/k_sensitivity_results
```

The analysis evaluates $k \in \{2,3,5,7,9,11,13,15\}$, using $k=15$ as the high-resolution reference. It reports rank stability, absolute changes in the resulting supervision signal, top-ranked-set overlap, and computational cost across calibration resolutions.

### Validations in Other Domains (Sec. 4.2)

Evaluates the generality of the proposed framework on labeled benchmark datasets outside the residential retrofit domain. The calibration-first supervision signal is constructed without observing ground-truth labels; labels are used only afterward for external validation.

The experiments include Ames Housing and Breast Cancer Wisconsin. Across these datasets, calibrated fusion is compared with naïve averaging and individual subset-specific scorers using task-appropriate external validation metrics.

### Psychometric Evaluations (App. B)

Evaluates the proposed framework on psychometric datasets in which multiple observed subscales serve as heterogeneous evidence sources for an underlying latent construct.

Experiments include:

- **BIG5:** Five personality domain scores (Extraversion, Neuroticism, Agreeableness, Conscientiousness, and Openness) are treated as subset-specific scorers and fused into a single supervision signal. The resulting ranking is compared against the first principal component computed from all 50 questionnaire items.

- **HEXACO:** Six personality domain scores (Honesty–Humility, Emotionality, Extraversion, Agreeableness, Conscientiousness, and Openness) are treated as subset-specific scorers and fused. The resulting ranking is evaluated against the first principal component of the complete 240-item inventory.

- **MGKT:** Five domain-level scores from the Multifactor General Knowledge Test are treated as subset-specific scorers and fused into a single supervision signal. The resulting ranking is compared against the first principal component computed from all 32 question-level scores.

For each dataset, calibrated fusion is compared with naïve averaging, the aggregate domain score, and individual subset-specific scorers. Agreement with the item-level latent reference is evaluated using Spearman rank correlation, Kendall rank correlation, and top-10% overlap.

Collectively, these experiments examine whether calibration-first supervision recovers latent structure identified from substantially finer-grained item-level information without using that item-level reference during supervision construction.