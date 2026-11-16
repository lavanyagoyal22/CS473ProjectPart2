"""Train + cross-validate Approach 3 (TF-IDF cosine similarity).

Usage:  python train_approach3.py training_data_Part2.csv [weights_dir]

* 5-fold GroupKFold, grouped by (hw_num, question_number) so every question's
  submissions stay in one fold (IDF and calibration are re-fit inside each fold).
* Reports RMSE / MAE / Pearson / under- and over-prediction counts.
* Fits the best setting on ALL training data and saves it to weights_dir.
"""
import itertools
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from approach3 import TfidfSqlModel


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {"RMSE": float(np.sqrt(np.mean((p - y) ** 2))),
            "MAE": float(np.mean(np.abs(p - y))),
            "Pearson": float(np.corrcoef(y, p)[0, 1]),
            "under": int((p < y - 0.5).sum()),
            "over": int((p > y + 0.5).sum())}


def get_folds(df, n_splits=5, path="folds.csv"):
    """Deterministic folds shared by ALL approaches: reuse folds.csv if present
    (copy it to the other approaches' folders), else create it. Grouped by
    (hw_num, question_number) so a question never spans train and validation."""
    import os
    if os.path.exists(path):
        f = pd.read_csv(path).set_index("id")["fold"]
        return df["id"].map(f).to_numpy()
    groups = df["hw_num"].astype(str) + ":" + df["question_number"].astype(str)
    fold = np.zeros(len(df), int)
    for k, (_, va) in enumerate(GroupKFold(n_splits=n_splits).split(df, groups=groups)):
        fold[va] = k
    pd.DataFrame({"id": df["id"], "fold": fold}).to_csv(path, index=False)
    return fold


def cross_validate(df, params, n_splits=5):
    fold = get_folds(df, n_splits)
    oof = np.zeros(len(df))
    for k in range(n_splits):
        tr, va = np.where(fold != k)[0], np.where(fold == k)[0]
        m = TfidfSqlModel(**params).fit(df.q.iloc[tr], df.r.iloc[tr],
                                        df.manual_grade.iloc[tr])
        oof[va] = m.predict_pairs(df.q.iloc[va], df.r.iloc[va])
    return metrics(df.manual_grade, oof)


def main():
    csv = sys.argv[1] if len(sys.argv) > 1 else "training_data_Part2.csv"
    out = sys.argv[2] if len(sys.argv) > 2 else "weights"
    df = pd.read_csv(csv, encoding="utf-8-sig")

    rows = []
    for fs, alpha in itertools.product(["cosine", "tfidf", "full"],
                                       [1.0, 10.0, 30.0]):
        params = dict(feature_set=fs, alpha=alpha)
        rows.append({**params, **cross_validate(df, params)})
    res = pd.DataFrame(rows).sort_values("RMSE")
    print(res.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    best = res.iloc[0]
    params = dict(feature_set=best.feature_set, alpha=float(best.alpha))
    print("\nBest setting:", params)
    model = TfidfSqlModel(**params).fit(df.q, df.r, df.manual_grade)
    model.save(out)
    print(f"Saved to {out}/")
    from approach3 import FEATURE_NAMES, FEATURE_SETS
    names = [FEATURE_NAMES[k] for k in FEATURE_SETS[model.feature_set]]
    print("Standardised ridge coefficients (grade points per 1 SD):")
    for n, c in zip(names, model.coef):
        print(f"  {n:20s} {c:+.2f}")
    res.to_csv("approach3_cv_results.csv", index=False)


if __name__ == "__main__":
    main()