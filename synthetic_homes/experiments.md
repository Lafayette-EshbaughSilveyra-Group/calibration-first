# Synthetic Homes Experiments

## Overview

The Synthetic Homes experiments evaluate whether the calibration-first framework behaves as theoretically expected when applied to a realistic multimodal evidence setting.

These experiments do not attempt to prove the framework propositions mathematically; those proofs are provided separately. Instead, they evaluate whether the implementation exhibits the properties predicted by the framework when using practical scorers, including a large language model scorer and heuristic scorers.

The experiments address two questions:

1. **Does calibration preserve the ordinal behavior of individual scorers?**
   - Corresponds to Proposition 1 (Ordinal Preservation).

2. **Does fusion behave according to the aggregate calibrated support of multiple scorers?**
   - Corresponds to Proposition 2 (Fusion Consistency).

The latent concept in this case study is **residential retrofit priority**.

---

# Scoring Channels

The framework is evaluated using four scoring channels.

A scoring channel consists of an evidence representation and a scorer that maps that evidence to a scalar estimate of the latent concept.

| Scoring Channel | Evidence | Scorer |
|---|---|---|
| Text HVAC | Whole-home inspection note | LLM-based HVAC retrofit-need scorer |
| Text Insulation | Whole-home inspection note | LLM-based insulation retrofit-need scorer |
| Structured HVAC | HVAC heating COP and cooling COP | Heuristic scorer |
| Structured Insulation | Wall R-value and roof R-value | Heuristic scorer |

The text channels share the same whole-home inspection note. The LLM receives the complete note and produces separate HVAC and insulation retrofit-need estimates.

The structured channels operate on grouped calibration features:

- HVAC uses heating and cooling COP jointly.
- Insulation uses wall and roof R-values jointly.

---

# Calibration Space

The synthetic ordinal reference space contains:

$$
5^4 = 625
$$

calibration points.

The four calibration features are:

1. HVAC heating COP
2. HVAC cooling COP
3. Roof R-value
4. Wall R-value

Each feature is discretized into five ordered levels.

- Level 1 represents inefficient residential characteristics.
- Level 5 represents efficient residential characteristics.

The calibration levels define an interpretable ordinal progression rather than an empirical distribution of existing homes.

For each calibration point:

1. The corresponding evidence representation is constructed.
2. Each scoring channel evaluates that evidence.
3. Calibration statistics are computed:

$$
\mu_j,\sigma_j
$$

for every scorer.

Future scorer outputs are standardized:

$$
\hat{s}_j=\frac{s_j-\mu_j}{\sigma_j}
$$

before fusion.

---

# Experiment 1: Ablation by Scoring Channel

## Objective

The ablation experiment evaluates whether individual scoring channels exhibit the intended ordinal relationship with residential retrofit priority and whether calibration preserves this relationship.

The expected relationship is:

- inefficient homes should receive higher retrofit-priority scores;
- efficient homes should receive lower retrofit-priority scores.

---

## Procedure

Each scoring channel is varied independently across five ordered efficiency levels.

The remaining scoring channels are held fixed at the neutral third level.

The four ablation conditions are:

1. Text HVAC
2. Structured HVAC
3. Text Insulation
4. Structured Insulation

For text channels:

- the whole-home inspection note is generated from the calibration condition;
- only the relevant component is varied;
- the unrelated component remains neutral.

For structured channels:

- HVAC features are varied jointly.
- Insulation features are varied jointly.

The resulting concept-specific supervision signal is recorded at each efficiency level.

---

## Results

### Text HVAC

LLM raw scores:
`[1.0, 0.6, 0.5, 0.0, 0.0]`

After calibration and fusion:
`[1.10, 0.43, 0.26, -0.57, -0.57]`


The signal decreases as HVAC efficiency increases.

The tie between levels 4 and 5 reflects identical LLM judgments rather than a calibration failure.

---

### Structured HVAC

Raw heuristic scores: `[5, 4, 3, 2, 1]`


After calibration and fusion:
`[1.26, 0.76, 0.26, -0.24, -0.74]`


The resulting supervision signal is strictly decreasing.

---

### Text Insulation

LLM raw scores:
`[1.0, 1.0, 0.3, 0.2, 0.0]`


After calibration and fusion:
`[0.94, 0.94, -0.09, -0.24, -0.53]`

The two least efficient conditions receive identical scores because the LLM treats both as requiring definite insulation upgrades.

---

### Structured Insulation

Raw heuristic scores:
`[5, 4, 3, 2, 1]`


After calibration and fusion:
`[0.91, 0.41, -0.09, -0.59, -1.09]`


The resulting supervision signal decreases monotonically.

---

## Proposition 1 Evaluation

For each scorer:

- all 625 calibration points were compared pairwise;
- the number of unordered pairs was:

$$
\binom{625}{2}=195000
$$

- each raw-score ordering was compared with its standardized-score ordering.

Results:

| Scorer | Pairwise Comparisons | Violations |
|---|---:|---:|
| Text HVAC | 195,000 | 0 |
| Structured HVAC | 195,000 | 0 |
| Text Insulation | 195,000 | 0 |
| Structured Insulation | 195,000 | 0 |

Calibration introduced no ordering reversals.

---

# Experiment 2: Scoring-Channel Agreement and Conflict

## Objective

The second experiment evaluates how the framework behaves when multiple scoring channels agree or disagree.

The ablation experiment evaluates individual channels in isolation. This experiment evaluates the fusion mechanism itself.

The experiment tests Proposition 2.

---

## Procedure

For each retrofit concept, two related scoring channels are varied simultaneously.

### HVAC

Channels:

- Text HVAC
- Structured HVAC

### Insulation

Channels:

- Text Insulation
- Structured Insulation

Each channel is varied across five efficiency levels, producing a:

$$
5 \times 5
$$

combined-variation grid.

This produces:

$$
25
$$

configurations per concept.

The full grid is evaluated by comparing all pairwise fused outputs.

---

# Expected Behavior

The fused supervision signal is:

$$
\kappa=
\frac{1}{m}
\sum_j \hat{s}_j
$$

Therefore:

- when channels agree on inefficiency, calibrated support accumulates and the score increases;
- when channels agree on efficiency, calibrated support decreases and the score decreases;
- when channels conflict, the final score depends on the relative magnitude of calibrated evidence.

The framework does not arbitrarily prefer one channel.

---

# Results

Across both concepts:

- HVAC grid: 300 pairwise comparisons.
- Insulation grid: 300 pairwise comparisons.

No fusion-consistency violations occurred.

Agreement cases produced the most extreme supervision signals.

Conflict cases produced intermediate signals.

---

## Interpretation

Example:

Both channels indicate inefficient HVAC:

- **Text HVAC:** inefficient
- **Structured HVAC:** inefficient


Both calibrated contributions are positive.

The fused score is high.

---

Example:

Channels disagree:

- **Text HVAC:** inefficient
- **Structured HVAC:** efficient


One channel contributes positive evidence while the other contributes negative evidence.

The fused signal falls between the two agreement extremes.

---

# Overall Conclusions

The Synthetic Homes experiments demonstrate that the calibration-first framework behaves as expected when instantiated with heterogeneous scoring channels.

The experiments show:

- LLM-based and heuristic scorers can be aligned through calibration.
- Multiple scorers can operate on the same evidence source.
- Feature groups can be evaluated jointly.
- Standardization preserves scorer-implied ordinal structure.
- Fusion combines calibrated evidence without introducing arbitrary ordering rules.

The experiments do not establish that the LLM predictions are ground truth. Instead, they demonstrate that when scorers provide monotonic evidence about a latent concept, the framework preserves and combines that evidence according to its theoretical properties.