# Open Approach (Approach 6) - submission notes

## Files
| File | Goes in submission ZIP? | Purpose |
|---|---|---|
| `submission/approach6.py` | yes (ZIP root) | Entrypoint: `load_model(weights_dir, config, assets)` |
| `submission/open_features.py` | yes (ZIP root) | Feature extraction from q and r (imported by approach6) |
| `submission/weights/approach6.joblib` | yes (`weights/`) | Saved RandomForest + Ridge (plain sklearn objects) |
| `train_open.py` | **no** (local only) | `cv` = 5-fold CV + writes `folds.csv`; `train` = fit + save weights |
| `folds.csv` | no | Shared fold assignment (`id`, `fold`) for teammates |
| `analysis.py`, `open_approach.py`, `results_*.csv` | no | Earlier analysis for the report (see note below) |

## Retraining
Use the grader's pinned versions (numpy 1.26.4, scipy 1.12.0, scikit-learn 1.6.1, pandas 2.2.3,
joblib 1.5.3, sqlglot 26.16.4, rapidfuzz 3.13.0), e.g. via `local_environment/create_environment.py`.
Saved weights from other versions may not load, and sqlglot's diff output can change between versions,
so features must be computed with the same sqlglot at training and inference.

    python train_open.py cv      # needs training_data_Part2.csv in the same folder
    python train_open.py train   # rewrites submission/weights/approach6.joblib

## Method
0.7 x RandomForest(300 trees, min_samples_leaf=5, max_features=0.4) + 0.3 x Ridge(RidgeCV, standardized),
clipped to [0, 100]. 44 features from q and r only: character/token Levenshtein, token lengths,
SQLGlot diff counts (insert/remove/update/move/keep), AST sizes, and per-clause construct-count changes.
No query execution, no LLM, no `question_prompt`/hw/question ids.

## Fallbacks (document in report)
- SQL that fails to parse: text features kept, AST/clause features 0, `parse_fail = 1`.
- A row whose features raise, or any model error: that row gets the training mean grade (73.1).
- Every id returned exactly once as `{"id", "score"}` with a float in [0, 100].

## Validation (5-fold, seed 42, grader-pinned versions)
| Model | RMSE | MAE | Pearson | Under / Over |
|---|---|---|---|---|
| Constant mean | 21.47 | 16.41 | -0.02 | 104 / 70 |
| Ridge | 16.90 | 13.22 | 0.62 | 99 / 75 |
| Random forest | 16.47 | 12.84 | 0.64 | 98 / 76 |
| Blend (submitted) | 16.26 | 12.64 | 0.66 | 101 / 73 |
