"""Train / tune / save Approach 2 (text edit distance).  Run LOCALLY, never in the grader.

    python train_approach2.py --csv training_data_Part2.csv

  1. Computes Levenshtein stats for every (q, r) pair for each unit/case combination (cached).
  2. Uses 5 stratified folds (stratified on grade bins, seed 42).  This is the SAME fold function as
     train_approach1.py, so teammates get identical folds; pass --folds-csv folds_5fold.csv to force it.
  3. Cross-validates every configuration in text_models.config_grid(); mapping parameters are refitted
     on the training part of each fold only.  Selection is by validation RMSE only.
  4. Refits the best configuration on all labelled data -> weights/approach2.joblib.
  5. Writes cv_results_approach2.csv, oof_predictions_approach2.csv, approach2_summary.json.
  6. Also reports question-grouped CV (GroupKFold) as an "unseen question" check.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold

from text_models import (ARTIFACT_NAME, cfg_name, config_grid, fit_artifact, pair_stats,
                         score_stats)


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    err = p - y
    pear = float(np.corrcoef(y, p)[0, 1]) if np.std(p) > 0 and np.std(y) > 0 else float("nan")
    return {"rmse": float(np.sqrt(np.mean(err ** 2))), "mae": float(np.mean(np.abs(err))),
            "pearson": pear, "under": int((p < y).sum()), "over": int((p > y).sum()),
            "exact": int((p == y).sum()), "mean_signed_error": float(err.mean())}


def make_folds(y, n_splits, seed):
    bins = pd.qcut(pd.Series(y), q=5, labels=False, duplicates="drop").to_numpy()
    fold = np.zeros(len(y), dtype=int)
    for k, (_, va) in enumerate(StratifiedKFold(n_splits, shuffle=True, random_state=seed)
                                .split(np.zeros(len(y)), bins)):
        fold[va] = k
    return fold


def cv_predict(cfg, stats, y, fold):
    pred = np.zeros(len(y))
    for k in sorted(set(fold)):
        tr, va = np.where(fold != k)[0], np.where(fold == k)[0]
        art = fit_artifact(cfg, [stats[i] for i in tr], y[tr])
        for i in va:
            pred[i] = score_stats(art, stats[i])
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds-csv", default=None, help="optional id,fold file to reuse")
    a = ap.parse_args()

    out = Path(a.out_dir)
    (out / "weights").mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(a.csv)
    y = df["manual_grade"].to_numpy(float)
    keys = (df["hw_num"].astype(str) + ":" + df["question_number"].astype(str)).tolist()
    print(f"{len(df)} rows, {len(set(keys))} questions")

    if a.folds_csv:
        fmap = pd.read_csv(a.folds_csv).set_index("id")["fold"]
        fold = df["id"].map(fmap).to_numpy(int)
    else:
        fold = make_folds(y, a.folds, a.seed)
    pd.DataFrame({"id": df["id"], "fold": fold}).to_csv(out / "folds_5fold.csv", index=False)
    gfold = np.zeros(len(y), dtype=int)
    for k, (_, va) in enumerate(GroupKFold(n_splits=a.folds).split(y, groups=keys)):
        gfold[va] = k

    base_p = np.zeros(len(y))
    for k in sorted(set(fold)):
        base_p[fold == k] = y[fold != k].mean()
    base = metrics(y, base_p)
    print(f"Baseline (predict training mean): RMSE={base['rmse']:.2f} MAE={base['mae']:.2f}")

    print("computing Levenshtein distances ...")
    cache = {(u, c): [pair_stats(q, r, u, c) for q, r in zip(df["q"], df["r"])]
             for u in ("char", "token") for c in ("keep", "lower")}

    grid = config_grid()
    print(f"cross-validating {len(grid)} configurations ...")
    res, rows = [], []
    for cfg in grid:
        p = cv_predict(cfg, cache[(cfg["unit"], cfg["case"])], y, fold)
        m = metrics(y, p)
        res.append((m["rmse"], cfg, m, p))
        rows.append({"config": cfg_name(cfg), **cfg, **m})
    res.sort(key=lambda t: t[0])
    print("\nTop 8 configurations:")
    for rmse, cfg, m, _ in res[:8]:
        print(f"  RMSE {rmse:6.2f}  MAE {m['mae']:6.2f}  r={m['pearson']:.3f}  {cfg_name(cfg)}")

    best_rmse, best_cfg, best_m, best_p = res[0]
    ident = [t for t in res if t[1]["mapping"] == "identity"][0]   # naive 100*(1-d) rule, no fitting
    gp = cv_predict(best_cfg, cache[(best_cfg["unit"], best_cfg["case"])], y, gfold)
    gm = metrics(y, gp)
    art = fit_artifact(best_cfg, cache[(best_cfg["unit"], best_cfg["case"])], y)
    joblib.dump(art, out / "weights" / ARTIFACT_NAME, compress=3)

    pd.DataFrame(rows).sort_values("rmse").to_csv(out / "cv_results_approach2.csv", index=False)
    o = df[["id", "hw_num", "question_number", "manual_grade"]].copy()
    o["fold"], o["pred_approach2"] = fold, best_p
    o.to_csv(out / "oof_predictions_approach2.csv", index=False)
    summary = {"baseline_mean": base, "best_config": best_cfg, "cv": best_m, "grouped_cv": gm,
               "mapping": art["mapping"], "best_unfitted_100_minus_100d": {"config": ident[1], "cv": ident[2]},
               "n_configs_tried": len(grid)}
    (out / "approach2_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print("\n================ SUMMARY (5-fold CV) ================")
    print(f"{'method':38s} {'RMSE':>6s} {'MAE':>6s} {'Pearson':>8s} {'under':>6s} {'over':>5s}")
    print(f"{'baseline: training mean':38s} {base['rmse']:6.2f} {base['mae']:6.2f} {'-':>8s} {base['under']:6d} {base['over']:5d}")
    print(f"{'naive 100*(1-d), best normalisation':38s} {ident[2]['rmse']:6.2f} {ident[2]['mae']:6.2f} {ident[2]['pearson']:8.3f} {ident[2]['under']:6d} {ident[2]['over']:5d}   [{cfg_name(ident[1])}]")
    print(f"{'Approach 2 (best fitted config)':38s} {best_m['rmse']:6.2f} {best_m['mae']:6.2f} {best_m['pearson']:8.3f} {best_m['under']:6d} {best_m['over']:5d}")
    print(f"question-grouped CV RMSE of best config: {gm['rmse']:.2f}")
    print("best config:", cfg_name(best_cfg))
    print("mapping:", art["mapping"] if art["mapping"]["type"] != "isotonic" else "isotonic curve (see json)")
    print(f"saved weights/{ARTIFACT_NAME}\nNext: python test_approach2.py --submission-dir . --csv <csv>")


if __name__ == "__main__":
    main()
