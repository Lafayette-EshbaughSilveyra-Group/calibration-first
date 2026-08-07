# Exploratory Psychometric Validation

## Motivation

In addition to the Ames Housing and Breast Cancer Wisconsin validation experiments presented in the main paper, we conducted an exploratory analysis on several psychometric datasets to investigate potential connections between calibration-first supervision construction and latent-variable modeling.

Unlike the main validation tasks, these datasets do not provide naturally heterogeneous modalities operating on incompatible scales. Instead, they contain multiple subscales intended to measure related psychological or cognitive constructs. As a result, these experiments should be interpreted as exploratory rather than as primary evidence for the proposed framework.

Specifically, we evaluated whether calibration-first fusion could recover latent structure extracted directly from item-level responses. For each dataset, subscale scores were treated as modalities, a calibrated fusion score $\kappa$ was constructed, and the resulting ranking was compared against the first principal component computed from the underlying item responses.

## Experimental Setup

Three datasets were considered:

### BIG5

The BIG5 dataset contains 50 personality items organized into five domains:

- Extraversion
- Neuroticism
- Agreeableness
- Conscientiousness
- Openness

Each domain score was treated as a modality. A fused score $\kappa$ was constructed from the five domain scores and compared against the first principal component derived from all 50 questionnaire items.

### HEXACO

The HEXACO dataset contains 240 personality items organized into six domains:

- Honesty–Humility
- Emotionality
- Extraversion
- Agreeableness
- Conscientiousness
- Openness

As with BIG5, domain scores were treated as modalities and compared against the first principal component extracted from the complete item set.

### MGKT

The Multifactor General Knowledge Test (MGKT) contains objective knowledge questions spanning multiple knowledge domains.

Domain-level scores were treated as modalities and fused into $\kappa$. The resulting ranking was then compared against the first principal component computed from all item-level responses.

## Results

| Dataset | Method                | Spearman ρ with Item-Level PC1 |
|---------|-----------------------|--------------------------------|
| BIG5    | Calibrated Fusion ($\kappa$) | 0.615                          |
| HEXACO  | Calibrated Fusion ($\kappa$) | 0.252                          |
| MGKT    | Calibrated Fusion ($\kappa$) | 0.987                          |

Several observations emerge.

First, MGKT exhibits an extremely strong correspondence between $\kappa$ and the dominant latent factor extracted from the item responses. The resulting Spearman correlation of 0.987 indicates that calibrated fusion almost perfectly reproduces the ranking implied by the item-level latent structure.

Second, BIG5 shows a moderate relationship between $\kappa$ and the first principal component. This is unsurprising because the Big Five model was explicitly designed to represent several distinct personality dimensions rather than a single latent trait.

Third, HEXACO exhibits substantially weaker agreement. HEXACO was developed as a multidimensional personality framework containing six conceptually distinct domains. Consequently, a single fused score is not expected to align strongly with a one-dimensional latent representation.

## Interpretation

These results are consistent with established psychometric theory.

Datasets characterized by a strong dominant latent factor should exhibit higher agreement between fused scores and latent-variable estimates. Conversely, datasets designed to measure multiple independent constructs should exhibit weaker agreement because no single latent dimension adequately summarizes the entire instrument.

The MGKT results are particularly notable because they suggest that calibration-first fusion can recover a latent ranking similar to one obtained from item-level analysis, despite operating only on aggregated domain-level scores.

## Why Calibration Had Little Effect

An important observation is that calibrated fusion and uncalibrated averaging produced nearly identical results across all three psychometric datasets.

This behavior is expected. In contrast to the Ames Housing and Breast Cancer Wisconsin experiments, the psychometric modalities were already expressed on highly comparable scales. Personality and knowledge subscale scores were measured using similar response formats and exhibited similar numerical ranges.

Because little scale mismatch existed between modalities, calibration had little work to perform. Consequently, calibrated fusion collapsed to approximately the same solution as uncalibrated averaging.

This should not be interpreted as a failure of the calibration procedure. Rather, it demonstrates that the method behaves sensibly when sources are already aligned.

## Conclusion

The psychometric experiments provide exploratory evidence that calibration-first fusion is connected to latent-variable modeling. In particular, the MGKT results show that domain-level fusion can closely recover latent structure derived from item-level responses.

However, because the psychometric modalities were already measured on comparable scales, these experiments do not directly evaluate the primary contribution of the paper: calibration across heterogeneous sources. For that reason, they are presented as supplementary evidence rather than as the primary validation experiments.