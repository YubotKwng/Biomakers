# FRDA Composite MRI Biomarker

Paired, leakage-safe modelling pipeline for evaluating MRI-derived progression scores in Friedreich ataxia (FRDA).

> **Objective:** Build interpretable imaging composites whose visit-to-visit change is sensitive to disease progression.
> The primary metric is paired Cohen's `d_z`, also reported as Standardized Response Mean.

## Contents

- [Project Idea](#project-idea)
- [Quick Start](#quick-start)
- [Workflow](#workflow)
- [Code Map](#code-map)
- [Models](#models)
- [Metrics and Validation](#metrics-and-validation)
- [Configuration](#configuration)
- [Usage Example](#usage-example)
- [Contributing](#contributing)

## Project Idea

The pipeline turns TRACK-FA imaging and clinical visit data into paired progression intervals such as `V1V2` and `V2V3`.
Models learn a scalar MRI score per visit. Training and tuning target
12-month longitudinal sensitivity using the two observed annual intervals:
`V1->V2` and `V2->V3`.

```text
delta = score(visit2) - score(visit1)
d_z   = mean(delta) / std(delta, ddof=1)
```

Clinical scales such as FARS and SARA are used as benchmarks, not imaging-model inputs.

For hyperparameter tuning, candidate models are evaluated by:

```text
S_annual = (d_z(V1->V2) + d_z(V2->V3)) / 2
```

The two annual intervals are kept separate, not concatenated as independent
patients. Candidate selection uses a one-standard-error style hierarchy:
near-optimal annual sensitivity first, then smaller `|d12-d23|`, higher
`P(delta > 0)`, fewer selected features, and available feature/coefficient
stability diagnostics. `V1->V3` is reported as 24-month cumulative sensitivity,
not as the one-year tuning objective.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/
```

Run a quick import check:

```bash
python3 -m compileall -q src
```

### Data Expectation

Processed TRACK-FA files are expected under `data/processed/`, especially `trackfa_pairs_drop3poms.csv` or `trackfa_pairs.csv`.
Raw data paths are configured in `src/config.py`.

## Workflow

| Step | Notebook | Purpose |
|---|---|---|
| 1 | `trackfa_merge.ipynb` | Merge raw TRACK-FA tables and quality-control paired rows. |
| 2 | `feature_selection_pipeline.ipynb` | Compare `none`, mutual information, and MML feature selection. |
| 3 | `srm_composite.ipynb` | Primary SRM Global and Patient-Adaptive composites. |
| 4 | `comparator_table.ipynb` | LDA and regression references. |
| 5 | `progression_dl.ipynb` | Exploratory deep-learning progression models. |

## Code Map

| Path | Role |
|---|---|
| `src/data/` | Loading, merging, reshaping, QC, and leakage guardrails. |
| `src/features/` | Feature registry, mutual information ranking, MML selection, stability reports. |
| `src/models/` | SRM Global, Patient-Adaptive, LDA, Ridge, PLS, ElasticNet CD, DL model definitions. |
| `src/eval/` | Cross-validation, paired effect-size metrics, and SHAP helpers. |
| `src/training/` | Fusion and pair-model training loops. |

## Models

| Model | Input | Training idea | Output |
|---|---|---|---|
| SRM Global Linear | all imaging features | `w = solve(cov(delta) + ridge I, mean(delta))` with covariance shrinkage | visit score |
| Patient-Adaptive | imaging + demographics | ElasticNet on imaging-by-demographic interactions | adaptive visit score |
| LDA comparator | imaging visits | regularised visit-separation direction | projection score |
| Regression reference | imaging visits | Ridge / PLS clinical-target prediction | predicted clinical score |
| DL exploratory | paired imaging visits | leakage-safe progression loss, clinical heads off by default | progression score |

## Metrics and Validation

> **Leakage rule:** `pair_id` identifies one progression interval, while `subject` is the split group.
> In Leave-One-Out or grouped cross-validation, all intervals for the same participant stay together.

- Primary tuning metric: mean annual validation `d_z` from held-out `V1->V2` and `V2->V3`.
- Primary final evaluation: genuine OOF `V1->V2` and `V2->V3` `d_z`, bootstrap CI, N, and `P(delta > 0)`.
- Secondary final evaluation: OOF `V1->V3` as 24-month cumulative sensitivity.
- Confidence intervals: bootstrap over paired deltas.
- Feature selection and hyperparameter tuning: fit inside each relevant training fold only.
- Clinical benchmarks: computed from adjacent changes only, e.g. `FARS2-FARS1` and `FARS3-FARS2`.

## Configuration

Main defaults live in `src/config.py`.

| Setting | Meaning |
|---|---|
| `random_state` | Reproducibility seed. |
| `cv_n_splits` | Grouped CV folds when Leave-One-Out is too slow. |
| `n_boot_ci` | Bootstrap draws for effect-size intervals. |
| `dl_weight_decay` | DL regularisation default. |

No `.env` file is required by the current codebase.

## Usage Example

```python
import pandas as pd
from src.data.trackfa_pairs import infer_trackfa_feature_groups, trackfa_pairs_to_long
from src.models.srm_global import srm_global_loocv

pairs = pd.read_csv("data/processed/trackfa_pairs_drop3poms.csv")
long_df = trackfa_pairs_to_long(pairs)
features = [c for c in infer_trackfa_feature_groups(pairs).all_neuroimaging if c in long_df]

result = srm_global_loocv(
    long_df,
    features,
    subject_col="pair_id",
    split_group_col="subject",
    selection_method="none",
    ridge=1e-8,
    covariance_shrinkage=0.45,
)

print(result["d_score"], result["d_ci_low"], result["d_ci_high"])
```

## Contributing

1. Keep model fitting fold-safe.
2. Keep clinical scores out of imaging features.
3. Prefer all imaging features unless a notebook explicitly studies selection.
4. Run a quick compile check before sharing results.

## Support

For project questions, contact the repository owner or project team.

## License

No license file is currently included.
