# CS473 Part 2: Open Approach (Approach 6)

Learned regression (random forest blended with Ridge) that predicts partial credit from
features of the original query `q` and the repaired query `r`.

Uses only `q` and `r`. No query execution, no result-set comparison, no LLM, no question IDs.

## Setup

```
pip install -r requirements.txt
```

Place `training_data_Part2.csv` in the same folder as the scripts.

## Usage

| Command | What it does |
|---|---|
| `python open_approach.py cv` | 5-fold CV (fixed seed, writes `folds.csv`), prints RMSE / MAE / Pearson / under / over |
| `python open_approach.py train` | Fits the final model on all training data, saves `open_model.joblib` |
| `python open_approach.py predict in.csv out.csv` | Adds a `predicted_grade` column (input needs `q` and `r`) |
| `python analysis.py` | Repeated-seed CV, ablations, residuals, importances; writes `results_*.csv` |

In code: `from open_approach import predict_df; predict_df(df)` returns predictions in [0, 100].
If `open_model.joblib` is missing or incompatible with the installed library versions, the model
is retrained automatically from `training_data_Part2.csv`.

## Features (about 44)

- **Text:** character and token Levenshtein (raw and normalized), token lengths of q and r, length ratio.
  Tokenization: identifiers, numbers, string literals, multi-character operators, single symbols.
- **AST (SQLGlot, SQLite dialect):** counts of Insert / Remove / Update / Move / Keep from `sqlglot.diff`,
  node counts of both trees, edits normalized by tree size.
- **Clause-level change counts:** absolute difference between q and r in the number of SELECT, JOIN, WHERE,
  GROUP BY, HAVING, subqueries, set operations, aggregates, LIKE / IN / BETWEEN, and so on.
- **Fallback:** if either query fails to parse, AST and clause features are zero and `parse_fail = 1`.

## Model

0.7 x RandomForest (300 trees, `min_samples_leaf=5`, `max_features=0.4`) + 0.3 x Ridge (`RidgeCV`,
standardized features), clipped to [0, 100]. `UNDER_SHIFT` in `open_approach.py` adds a constant to
predictions to trade a little RMSE for fewer underpredictions (default 0).

## Validation

5-fold stratified CV on score bins (<=50, 50-70, 70-85, >85), seed 42. `folds.csv` maps each `id`
to its fold so other approaches can use identical folds. Nothing is fit outside the training fold.

## Current results (5 seeds, from `results_*.csv`)

| Model | RMSE | MAE | Pearson | Under / Over |
|---|---|---|---|---|
| Constant mean | 21.46 | 16.41 | n/a | 104 / 70 |
| Ridge | 17.70 | 13.95 | 0.57 | 99.6 / 74.4 |
| Random forest | 16.68 | 12.94 | 0.63 | 96.0 / 78.0 |
| Blend (0.7 RF + 0.3 Ridge) | 16.59 | 12.88 | 0.64 | 97.4 / 76.6 |

Differences between the learned models are small relative to the label noise (174 rows).
