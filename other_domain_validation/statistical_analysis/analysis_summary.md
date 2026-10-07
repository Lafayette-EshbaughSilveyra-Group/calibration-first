# Cross-domain statistical analysis

- Bootstrap replicates: **50,000**
- Nominal confidence level: **95.0%**
- Random seed: **20260922**
- All method comparisons use **paired bootstrap resampling**: each bootstrap replicate uses the same sampled observations for every method.

## Dataset summary

| Dataset | n | Task | Primary metric |
|---|---:|---|---|
| Ames Housing | 1460 | regression | Spearman's rho |
| Breast Cancer Wisconsin | 569 | classification | AUROC |
| Wine Quality | 1599 | regression | Spearman's rho |

## Primary paired comparisons

Differences are **Calibrated Fusion − baseline**. For each baseline, Bonferroni correction controls the family-wise error rate across the three benchmark datasets.

| Dataset | Baseline | κ | Baseline estimate | Difference | 95% paired CI | Bonferroni-adjusted CI |
|---|---|---:|---:|---:|---:|---:|
| Ames Housing | Uncalibrated Avg | 0.8905 | 0.8593 | 0.0311 | [0.0143, 0.0485] | [0.0109, 0.0524] |
| Breast Cancer Wisconsin | Uncalibrated Avg | 0.9843 | 0.9575 | 0.0268 | [0.0140, 0.0414] | [0.0115, 0.0449] |
| Wine Quality | Uncalibrated Avg | 0.5557 | 0.1916 | 0.3641 | [0.3173, 0.4112] | [0.3062, 0.4221] |
| Ames Housing | Empirical z-Score | 0.8905 | 0.9077 | -0.0172 | [-0.0225, -0.0122] | [-0.0238, -0.0111] |
| Breast Cancer Wisconsin | Empirical z-Score | 0.9843 | 0.9827 | 0.0017 | [0.0004, 0.0030] | [0.0002, 0.0034] |
| Wine Quality | Empirical z-Score | 0.5557 | 0.5464 | 0.0093 | [0.0012, 0.0176] | [-0.0005, 0.0194] |
| Ames Housing | Borda Rank Avg | 0.8905 | 0.9053 | -0.0148 | [-0.0214, -0.0085] | [-0.0230, -0.0072] |
| Breast Cancer Wisconsin | Borda Rank Avg | 0.9843 | 0.9841 | 0.0002 | [-0.0034, 0.0041] | [-0.0041, 0.0051] |
| Wine Quality | Borda Rank Avg | 0.5557 | 0.5488 | 0.0069 | [-0.0050, 0.0189] | [-0.0077, 0.0215] |
| Ames Housing | Unsupervised PCA | 0.8905 | 0.9085 | -0.0181 | [-0.0233, -0.0131] | [-0.0245, -0.0120] |
| Breast Cancer Wisconsin | Unsupervised PCA | 0.9843 | 0.9867 | -0.0024 | [-0.0049, -0.0001] | [-0.0055, 0.0003] |
| Wine Quality | Unsupervised PCA | 0.5557 | 0.5460 | 0.0097 | [-0.0061, 0.0257] | [-0.0094, 0.0293] |

## All paired comparisons

The full machine-readable results are in `paired_differences.csv`.

**Important:** `bootstrap_fraction_*` columns are descriptive fractions of bootstrap replicates. They are not reported as frequentist p-values.
