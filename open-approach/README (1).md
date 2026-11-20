# Approach 6 (Open Approach): how we get our RMSE

Predicts partial credit (0-100) for a student's SQL query by comparing the original query `q`
with its repaired version `r`. A learned regression model maps features of the q-to-r change to
the manual grade. It uses only `q` and `r`: no query execution, no LLM, no question ids.

## Results

| Where | RMSE | MAE | Pearson | Under / Over |
|---|---|---|---|---|
| Constant prediction (training mean, 73.1) | 21.47 | 16.41 | -0.02 | 104 / 70 |
| 5-fold CV: Ridge only | 16.90 | 13.22 | 0.62 | 99 / 75 |
| 5-fold CV: Random forest only | 16.47 | 12.84 | 0.64 | 98 / 76 |
| **5-fold CV: Blend (submitted)** | **16.26** | **12.64** | **0.66** | 101 / 73 |
| **Leaderboard (held-out set, submission #1)** | **18.55** | **13.76** | **0.677** | not reported |

Runtime on the grader: 2.97 s for load + predict.

<!-- TODO: add a row each time you resubmit (date, what changed, RMSE) -->

**Why the leaderboard RMSE is higher than CV.** RMSE is dominated by a few large errors. In
our training data, the 10 worst rows account for about 42% of the total squared error, so a
different set of rows gives a different handful of big misses. Pearson matches CV (0.68 vs 0.66),
so the model ranks students about as well on the held-out set. The leaderboard was used only as
a sanity check, not for tuning.

## How the score is produced, step by step

1. **Parse** both queries with SQLGlot (SQLite dialect). If either fails to parse, keep the text
   features and set `parse_fail = 1`; AST and clause features become 0.
2. **Extract about 44 features** (`submission/open_features.py`):
   - *Text edit:* character and token Levenshtein (raw and normalized), token length of q and r,
     length ratio q/r. Tokens are identifiers, numbers, string literals, multi-character
     operators and single symbols, lower-cased.
   - *AST edit:* counts of Insert / Remove / Update / Move / Keep from `sqlglot.diff`, node
     counts of both trees, and edits normalized by tree size.
   - *Clause changes:* absolute difference between q and r in the number of SELECT, JOIN, WHERE,
     GROUP BY, HAVING, subqueries, set operations, aggregates, LIKE / IN / BETWEEN, and so on.
3. **Predict** with a blend: `0.7 x RandomForest + 0.3 x Ridge`, clipped to [0, 100].
   - Random forest: 300 trees, `min_samples_leaf=5`, `max_features=0.4`, `random_state=42`.
   - Ridge: `RidgeCV` (alphas 0.1 to 10^4) on standardized features.
4. **Fallbacks:** a row whose features fail, or any model error, gets the training mean (73.1).
   Every id is returned exactly once as `{"id", "score"}` with a float in [0, 100].

## How we validated (no leakage)

- 5-fold CV stratified on score bins (<=50, 50-70, 70-85, >85), seed 42. The fold of each row is
  in `folds.csv` (`id`, `fold`) so every approach can use identical folds.
- Nothing is fit outside the training fold: the forest, the Ridge and its scaler are refit per fold.
- We chose the blend weight (0.7) and forest settings by CV RMSE only. Differences between
  Ridge, forest and blend are small relative to label noise (174 rows), so treat them as close.
- The final weights are fit on all 174 rows and saved to `submission/weights/approach6.joblib`.

## Why a learned model beats a single similarity score

A raw similarity (edit distance, cosine) is not on the grade scale, so using it directly gives
a large RMSE; it has to be mapped to 0-100 using training data. Here the regression learns
that mapping and combines several signals. A single feature (normalized AST edits through
Ridge) gave about 20.2 RMSE in an earlier test, while combining features gave about 16.5.

## What the model does well and badly

- Ranks students well (Pearson about 0.66).
- Pulls extreme grades toward the middle: low grades (0-50) are overpredicted by about 25 points
  and high grades (85+) underpredicted by about 13 in CV. This is typical of regression to the
  mean and is the main source of error.
- Cannot see meaning: near-identical query pairs have received very different manual grades
  (for example 0 vs 89 on the same question), which no q-to-r edit feature can separate.
- Things we tried that did not improve CV RMSE: homework / database / question-number features,
  per-question mean grade, directional add/delete clause features, per-clause edit locations,
  and tuning forest settings.

<!-- TODO: fill in for the report -->
<!-- - Explainability: top features (RF importance: len_ratio, len_q, ast_edit_norm, ...) -->
<!-- - Robustness by homework / database (RMSE and bias) -->
<!-- - Possible exploits: padding a wrong query so the repair looks like a small edit -->

## Reproduce

Use the grader's pinned versions (numpy 1.26.4, scipy 1.12.0, scikit-learn 1.6.1, pandas 2.2.3,
joblib 1.5.3, sqlglot 26.16.4, rapidfuzz 3.13.0, Python 3.10-3.12), for example the
`.part2-venv` made by `local_environment/create_environment.py`. Saved weights built on other
versions may fail to load, and SQLGlot's diff output can differ between versions.

```bash
# needs training_data_Part2.csv in this folder
python train_open.py cv      # prints the CV table above and writes folds.csv
python train_open.py train   # refits on all rows, writes submission/weights/approach6.joblib
```

## Files

| File | In the submission ZIP? |
|---|---|
| `submission/approach6.py` (entrypoint) | yes, ZIP root |
| `submission/open_features.py` (features) | yes, ZIP root |
| `submission/weights/approach6.joblib` | yes, in `weights/` |
| `train_open.py`, `folds.csv`, `README.md` | no |
| `analysis.py`, `open_approach.py`, `results_*.csv` | no (earlier analysis for the report) |

## How to update this README

When you change the model, rerun `python train_open.py cv`, paste the new numbers into the
Results table, add a row for each leaderboard submission, and note what changed in the step list.
