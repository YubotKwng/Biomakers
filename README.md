# Refactored baseline pipeline

Bit-exact refactor of the four notebooks under `../notebooks/` into a clean
`src/` package with thin orchestration notebooks under `notebooks/`.

> The original `notebooks/` folder is **read-only** — it must never be
> edited or moved. This refactor reads the originals and reproduces them.

## Layout (mirrors `draft/`)

```
biomarkers/
├── src/
│   ├── config.py
│   ├── data/        # ids, loading, merge, reshape, qc
│   ├── features/    # registry, entropy (MI), selection
│   ├── models/      # elasticnet (sklearn + CD), lda, ridge, pls, mlp, fusion, pair
│   ├── training/    # loss, fusion training, pair training
│   ├── eval/        # metrics, cv, importance, composite, shap
│   └── viz/
├── notebooks/
│   ├── data_merge.ipynb
│   ├── paper_reproduction.ipynb
│   ├── entropy_methods.ipynb
│   └── progression_dl.ipynb
├── tests/
│   ├── test_metrics.py
│   ├── test_cv.py
│   ├── test_reshape.py
│   ├── test_selection.py
│   └── test_equivalence/
└── results/         # gitignored
```

## How to run

```bash
# from biomarkers/
pip install -r requirements.txt
pytest tests/                           # unit + equivalence tests
jupyter lab notebooks/                  # the four refactored notebooks
```

## Notes
- `src/config.py::set_global_seeds(42)` is called as the first cell of every
  notebook for determinism.
- See `REFACTOR_PLAN.md` for the migration plan and `MIGRATION_LOG.md` for
  what has been migrated and any deviations.
- The interaction-term template under `../draft/` is independent and is not
  touched by this refactor.
# Biomakers
