# FRDA Composite MRI Biomarker

This repository implements a notebook-first, function-backed pipeline for building and evaluating MRI-derived longitudinal biomarkers in Friedreich ataxia (FRDA).

The core goal is to learn an interpretable composite MRI score whose within-participant change is sensitive to disease progression.

Clinical scores such as FARS and SARA are used for comparison and validation. They are not used as primary imaging-model inputs for the composite biomarker.

## Contents

- [Project Context](#project-context)
- [Data And Preprocessing](#data-and-preprocessing)
- [Analysis Populations](#analysis-populations)
- [Cross-Validation And Leakage Control](#cross-validation-and-leakage-control)
- [Models](#models)
- [Model Selection](#model-selection)
- [Methodological Decisions](#methodological-decisions)
- [Final Evaluation](#final-evaluation)
- [Result Display](#result-display)
- [Current Result Snapshot](#current-result-snapshot)
- [Code Map](#code-map)
- [How To Run](#how-to-run)
- [Current Known Gaps](#current-known-gaps)

## Project Context

The biomarker target is 12-month longitudinal sensitivity. The two annual progression intervals are:

```text
V1 -> V2
V2 -> V3
```

For a visit-level composite score `score`, each interval uses:

```text
delta = score(follow-up visit) - score(baseline visit)
d_z   = mean(delta) / std(delta, ddof=1)
```

`mean(delta)` measures the average longitudinal change in the composite score.
`d_z` standardises that mean change by the participant-to-participant
variability in change, so it measures how strong the longitudinal signal is
relative to its noise.

The pipeline also reports:

```text
P(delta > 0)
```

This is the proportion of participants whose score changed in the expected
disease-progression direction. It is used because a large `d_z` alone does not
show whether most participants move consistently in the expected direction.
For example, `P(delta > 0) = 0.88` means that 88% of participants showed
positive progression-direction change.

The 24-month interval:

```text
V1 -> V3
```

is reported as cumulative 24-month sensitivity, not as the primary one-year tuning target.

The primary tuning summary uses the mean of the two annual interval effects:

```text
S_annual = (d_z(V1->V2) + d_z(V2->V3)) / 2
```

This is used because the scientific claim is annual progression sensitivity.
Using both V1->V2 and V2->V3 asks whether the biomarker works across two
observed 12-month windows, not only in a single interval. Because the current
annual counts are similar, equal weighting gives a simple and transparent
summary while still keeping `d12` and `d23` visible separately for temporal
consistency review.

## Data And Preprocessing

The pipeline prepares two main processed datasets under `data/processed/`.

| Dataset | Role | Main use |
|---|---|---|
| `trackfa_long.csv` | Subject-level longitudinal visit table | Final interval evaluation across V1, V2, and V3. |
| `trackfa_pairs_drop3poms.csv` | Main paired annual modelling table | Training/tuning on annual V1->V2 and V2->V3 progression pairs. |
| `trackfa_pairs.csv` | Strict complete-sheet paired table | Audit/comparison; more restrictive than the Drop 3 POMs modelling table. |
| `trackfa_pairs_miss.csv` | Missingness export | Missingness review and data-audit support. |

Data processing is run from:

```text
notebooks/trackfa_merge.ipynb
```

Reusable data functions live in:

```text
src/data/
```

The merge and audit workflow includes:

1. load raw TRACK-FA clinical and imaging tables;
2. merge to subject-level visits;
3. construct paired annual progression rows;
4. export `trackfa_long.csv`;
5. export `trackfa_pairs_drop3poms.csv`;
6. display visit-pattern counts;
7. display feature missingness;
8. display Feature x Visit x Site missingness;
9. run MRI QC by site;
10. document harmonisation leakage policy.

Current paired annual modelling counts are:

| Interval | N |
|---|---:|
| V1->V2 | 108 |
| V2->V3 | 99 |

## Analysis Populations

The current workflow distinguishes these populations:

| Population | Meaning | Use |
|---|---|---|
| V1->V2 cohort | Participants with a usable V1 and V2 interval | Primary 12-month sensitivity estimate. |
| V2->V3 cohort | Participants with a usable V2 and V3 interval | Temporal replication of annual sensitivity. |
| V1->V3 cohort | Participants with usable V1 and V3 visits | Secondary 24-month cumulative sensitivity. |
| All available paired intervals | All usable annual pairs | Model tuning and comparator summaries. |

V1->V2 and V2->V3 are kept separately in tuning and reporting. They are not concatenated and treated as independent patients.

## Cross-Validation And Leakage Control

All visits and intervals from the same participant must remain in the same fold.

In the paired modelling table:

| Column | Meaning |
|---|---|
| `pair_id` | One annual interval, such as a participant's V1V2 or V2V3 pair. |
| `subject` | Participant identifier used for grouped splitting. |

The modelling notebooks therefore use:

```python
subject_col = "pair_id"
split_group_col = "subject"
```

This lets the model calculate paired deltas by interval while keeping the same participant out of both training and held-out folds.

The intended validation structure is:

| Layer | Role |
|---|---|
| Outer subject-level/grouped CV | Produces out-of-fold predictions for unbiased performance estimation. |
| Inner subject-level/grouped CV | Tunes hyperparameters inside the outer-training participants only. |

All learned preprocessing is fitted inside the relevant training fold only:

- imputation;
- scaling;
- outlier clipping if enabled;
- feature selection;
- harmonisation if later enabled;
- model fitting;
- hyperparameter tuning.

## Models

| Model | Role | Input | Output | Current interpretation |
|---|---|---|---|---|
| SRM Global Linear Composite | Main classical biomarker | MRI features | One score per visit | Current best-performing classical model. |
| Patient-Adaptive interaction model | Exploratory adaptive biomarker | MRI features plus participant modulators | One adaptive score per visit | Tests whether participant factors should modulate MRI weights. |
| LDA comparator | Classical comparator | MRI visit rows | Projection score | Reference model, not the main biomarker. |
| ElasticNet regression | Clinical-score prediction comparator | MRI visit rows | Predicted clinical score | Single retained clinical-score comparator. |
| FusionMLP | Exploratory deep-learning model | MRI feature groups | Progression score | Clinical heads disabled by default. |
| PairModel | Exploratory paired deep-learning model | Paired MRI visits | Progression score | Uses paired visit inputs. |
| Single MRI feature baselines | Baseline | One MRI feature at a time | Feature delta | Used to compare the composite against strongest single MRI feature. |

## Model Selection

The model-selection target is annual progression sensitivity.

For each candidate hyperparameter configuration:

```text
d12 = d_z(V1->V2)
d23 = d_z(V2->V3)
S_annual = (d12 + d23) / 2
gap = |d12 - d23|
```

Primary tuning metric:

```text
mean_validation_annual_dz = (d12 + d23) / 2
```

Candidate selection uses a one-standard-error style hierarchy:

1. identify candidates near the best mean annual validation d_z;
2. prefer smaller `|d12 - d23|`;
3. prefer higher `P(delta > 0)`;
4. prefer fewer selected features;
5. prefer more stable feature selection and coefficient signs where available.

Feature-selection method is now treated as a model-selection component rather
than only a separate audit. The active candidate family can include:

| Selection method | Objective | Status |
|---|---|---|
| `none` | Full prespecified MRI feature panel. | Required reference candidate. |
| `mi_visit` | Historical mutual information ranking against visit/progression label; `mi` remains a compatibility alias. | Sensitivity comparator. |
| `mml` | Existing MML linear-regression complexity score against the visit-label target. | Generic complexity comparator; not progression-aligned MML. |
| `progression_univariate` | Feature-wise annual paired effects: V1->V2, V2->V3, mean annual d_z, and interval gap. | Progression-aligned filter. |
| `progression_mrmr` | Progression relevance minus absolute-correlation redundancy penalty. | Progression-aware redundancy filter. |
| `sparse_srm` | Embedded sparse SRM-style selection from interval-balanced annual change statistics with ElasticNet-style shrinkage. | Sparse progression selector. |

All non-`none` selectors are fitted inside the relevant training fold only.
Selection frequency, Jaccard stability, and represented MRI domains are
displayed for review but are not used as causal feature-importance claims.

The tuning notebooks display:

- all tested parameter values;
- validation `d12`;
- validation `d23`;
- mean annual d_z;
- `|d12 - d23|`;
- `P(delta > 0)`;
- selected feature count;
- feature/coefficient stability if available;
- standard error or variability proxy;
- numerically best row;
- one-standard-error candidate set;
- recommended row;
- reason for recommendation.

## Methodological Decisions

| Decision | What was chosen | Reason |
|---|---|---|
| Biomarker target | Optimise longitudinal progression sensitivity rather than clinical-score prediction. | The scientific aim is to detect disease progression from MRI change; clinical scores are important benchmarks but should not define the imaging biomarker. |
| Primary tuning interval | Tune on V1->V2 and V2->V3 annual intervals. | The intended context of use is 12-month sensitivity, so the training/tuning metric should match one-year progression. |
| V1->V3 role | Report V1->V3 as secondary 24-month cumulative sensitivity. | The 24-month signal is useful and often stronger, but it should not be described as the primary one-year sensitivity metric. |
| Annual interval weighting | Use equal weighting: `(d12 + d23) / 2`. | Current annual counts are similar enough for equal interval weighting, and equal weighting keeps the interpretation simple. |
| Interval handling | Keep V1->V2 and V2->V3 separate rather than concatenating them as independent patients. | The same participant may contribute multiple intervals, so pooling intervals as independent rows can overstate effective sample size and temporal consistency. |
| Split unit | Split by participant using `split_group_col="subject"`. | This prevents leakage where one participant's V1->V2 interval is in training while the same participant's V2->V3 interval is held out. |
| Pair unit | Use `pair_id` as the paired-delta unit. | The paired table stores one annual interval per `pair_id`; paired deltas must be computed within each interval. |
| Outer validation | Use subject-level/grouped outer CV for OOF performance. | Out-of-fold predictions provide a less biased estimate of model performance than training-set d_z. |
| Inner tuning | Use grouped inner CV inside each outer-training fold. | Hyperparameters must be selected without using outer-test participants. |
| Preprocessing | Fit scaling, feature selection, clipping, and any future harmonisation inside training folds only. | Preprocessing fitted on all data would leak test-fold information into model training. |
| Primary tuning metric | Use `mean_validation_annual_dz`. | It directly measures average annual progression sensitivity across V1->V2 and V2->V3. |
| Do not select raw maximum blindly | Use a one-standard-error style hierarchy. | A numerically highest candidate may be unstable or work in only one annual interval; near-optimal but more consistent models are scientifically preferable. |
| First tie-breaker | Prefer smaller `|d12 - d23|`. | A useful biomarker should show temporal replication, not just a large effect in one interval. |
| Directional consistency | Report and use `P(delta > 0)` as a diagnostic/tie-breaker. | It shows the proportion of participants changing in the expected disease-progression direction. |
| Model simplicity | Prefer fewer features among near-equivalent models. | Simpler models are easier to interpret and less likely to overfit in a small longitudinal dataset. |
| Stability diagnostics | Report feature-selection, sign, and ranking stability where available. | Stable selected features and coefficient directions make the biomarker more credible and interpretable. |
| Clinical scores | Use FARS/SARA as comparators and validation outcomes, not as primary biomarker-selection inputs. | Tuning on clinical correlations would shift the objective from MRI progression sensitivity to clinical-score prediction. |
| Clinical-score prediction comparator | Keep ElasticNet regression as the single clinical-score prediction comparator. | ElasticNet gives a sparse linear clinical-score prediction benchmark, which is easier to interpret than retaining several side models. Keeping one comparator avoids diluting the main analysis and lets the clinical-prediction result remain a benchmark rather than a competing biomarker objective. |
| P(delta > 0) reporting | Report direction consistency alongside `d_z`. | `d_z` measures standardised mean change, while `P(delta > 0)` shows whether most participants move in the expected progression direction. This guards against a model whose average effect is driven by a subset of participants. |
| Mean annual change/effect | Summarise V1->V2 and V2->V3 using their mean annual effect while displaying each interval separately. | The project target is 12-month progression sensitivity. Averaging the two annual intervals gives one tuning summary, while separate reporting of `d12`, `d23`, and their gap checks whether the signal is temporally consistent. |
| Single-feature baselines | Compare against the strongest individual MRI feature. | A composite is useful only if it improves on simple, interpretable single-feature alternatives. |
| Healthy controls | Reserve healthy-control change for specificity analysis. | Controls should test disease specificity after the biomarker is trained, not influence model selection. |
| Harmonisation | Do not apply harmonisation by default; if added, fit it only within training subjects. | Harmonisation can reduce scanner/site effects but may also remove true longitudinal change if applied carelessly. |
| Notebook-first execution | Keep functions in `src/`, run and display results in notebooks. | This keeps code reusable while making the analysis transparent for supervisor review and thesis reporting. |
| Consolidated reporting | Save one final model-performance CSV. | A single reporting table avoids scattered results and makes model comparison easier to audit. |

## Final Evaluation

Final evaluation uses out-of-fold predictions.

Primary 12-month evaluation:

| Question | Metric |
|---|---|
| 12-month sensitivity V1->V2 | d_z, 95% bootstrap CI, N, P(delta > 0). |
| 12-month sensitivity V2->V3 | d_z, 95% bootstrap CI, N, P(delta > 0). |
| 12-month pooled annual sensitivity | Pooled V1->V2 + V2->V3 d_z, 95% bootstrap CI, N, P(delta > 0), using subject-grouped OOF annual pair predictions. |

Secondary evaluation:

| Question | Metric |
|---|---|
| 24-month cumulative sensitivity | V1->V3 d_z, 95% bootstrap CI, N. |
| Direction consistency | Annual P(delta > 0). |
| Better than clinical scales? | Composite d_z vs FARS/SARA change d_z. |
| Better than MRI alone? | Composite d_z vs strongest individual MRI feature. |
| Clinically meaningful? | Spearman score vs FARS/SARA. |
| Tracks clinical change? | Spearman delta composite vs delta FARS/SARA. |
| Disease specific? | FRDA vs healthy-control change, once control analysis is complete. |
| Feature robustness | Feature-selection, coefficient-sign, and ranking stability diagnostics. |

## Result Display

The pipeline is notebook-first:

```text
src/       reusable functions
notebooks/ execution and result display
results/  exported tables/logs
```

| Notebook | Purpose |
|---|---|
| `trackfa_merge.ipynb` | Data merge, processed exports, visit pattern, missingness, site/QC audit. |
| `feature_selection_pipeline.ipynb` | Feature-selection comparison and tuning-review tables. |
| `srm_composite.ipynb` | Main SRM Global Linear composite model. |
| `interaction_term.ipynb` | Patient-Adaptive interaction model using MRI features plus demographic/genetic modulators. |
| `comparator_table.ipynb` | LDA and regression comparator models. |
| `progression_dl.ipynb` | Exploratory deep-learning models; skips gracefully if PyTorch is unavailable. |
| `model_performance.ipynb` | Final consolidated performance display and CSV export. |

The consolidated model-performance file is:

```text
results/model_performance_summary.csv
```

## Current Result Snapshot

The current linear composite model is performing well in the preliminary analysis.

Current out-of-fold summary for the main SRM Global Linear Composite:

| Metric | Current value |
|---|---:|
| V1->V2 Cohen's d_z | 1.13 |
| V1->V2 95% CI | [0.93, 1.40] |
| V1->V2 N | 108 |
| V1->V2 P(delta > 0) | 0.89 |
| V2->V3 Cohen's d_z | 0.78 |
| V2->V3 95% CI | [0.58, 1.02] |
| V2->V3 N | 99 |
| V2->V3 P(delta > 0) | 0.76 |
| Pooled annual Cohen's d_z | 0.94 |
| Pooled annual 95% CI | [0.78, 1.12] |
| Pooled annual N | 207 |
| Pooled annual P(delta > 0) | 0.83 |
| V1->V3 cumulative Cohen's d_z | 1.46 |
| V1->V3 95% CI | [1.24, 1.79] |
| V1->V3 N | 90 |

`P(delta > 0)` measures the proportion of participants whose composite score changed in the expected disease-progression direction. A value of 0.89 means that approximately 89% of participants changed in the expected direction for the stronger annual interval.

These results keep SRM Global Linear as the primary model. Patient-Adaptive, LDA/regression, and deep-learning models remain comparator or exploratory analyses because their annual progression sensitivity is lower in the current out-of-fold results. These results are preliminary and should be interpreted together with the model-selection table, annual interval consistency, clinical benchmarks, and remaining specificity analysis.

## Code Map

| Path | Role |
|---|---|
| `src/data/` | Loading, merging, reshaping, audit, QC, missingness, model-safety checks. |
| `src/features/` | Feature registry, mutual information ranking, MML selection, stability helpers. |
| `src/models/` | SRM Global, Patient-Adaptive, LDA, ElasticNet, DL model definitions. |
| `src/eval/` | Cross-validation, interval effects, metrics, clinical validity, specificity, stability. |
| `src/training/` | FusionMLP and PairModel training loops. |
| `src/reporting/` | Tuning-review tables, final performance tables, CSV export. |
| `notebooks/` | Human-readable execution and result display. |
| `tests/` | Unit tests for audit, leakage, intervals, model selection, and reporting. |

## How To Run

Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run notebooks in this order:

```text
1. notebooks/trackfa_merge.ipynb
2. notebooks/feature_selection_pipeline.ipynb
3. notebooks/srm_composite.ipynb
4. notebooks/interaction_term.ipynb
5. notebooks/comparator_table.ipynb
6. notebooks/progression_dl.ipynb
7. notebooks/model_performance.ipynb
```

Run quick checks:

```bash
python3 -m compileall -q src
python3 -m pytest -q
```

## Current Known Gaps

| Gap | Current status |
|---|---|
| Healthy-control specificity | Reporting structure exists, but final control-specific result is still missing. |
| Harmonisation | Leakage policy is documented; implementation should only be added if needed and must fit parameters inside training folds. |
| DL results | Notebook is executable, but DL training depends on PyTorch availability. |
| Thesis wording | Must describe V1->V2 and V2->V3 as annual sensitivity, and V1->V3 as cumulative 24-month sensitivity. |
