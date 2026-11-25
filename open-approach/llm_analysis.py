"""LOCAL. Step 2 of the LLM experiment: does the LLM score add anything to the current model?
  python llm_analysis.py llm_scores_gpt_oss_120b.csv
Uses the same folds (folds.csv) and the same blend as train_open.py. Nothing here touches the leaderboard.
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent / "submission"))
from open_features import feature_matrix
import train_open as T

path = sys.argv[1]
df = pd.read_csv(T.CSV); y = df.manual_grade.values.astype(float)
M, names = feature_matrix(df.q, df.r)
fold = pd.read_csv("folds.csv").set_index("id").loc[df["id"], "fold"].values
L = pd.read_csv(path).drop_duplicates("id").set_index("id").reindex(df["id"])["llm_score"].values.astype(float)
have = ~np.isnan(L)
print(f"LLM scores available for {have.sum()}/{len(df)} rows")
if have.sum() < 100: print("WARNING: fewer than 100 rows scored; results will be unreliable")

def cv(X, w=0.7):
    p = np.zeros(len(y))
    for k in range(5):
        tr, te = fold != k, fold == k
        Xtr, Xte = X[tr].copy(), X[te].copy()
        if X.shape[1] > M.shape[1]:                       # impute missing LLM score with the TRAIN-fold mean
            mu = np.nanmean(Xtr[:, -1]); Xtr[np.isnan(Xtr[:, -1]), -1] = mu; Xte[np.isnan(Xte[:, -1]), -1] = mu
        rf = T.make_rf().fit(Xtr, y[tr]); rd = T.make_ridge().fit(Xtr, y[tr])
        p[te] = np.clip(w * rf.predict(Xte) + (1 - w) * rd.predict(Xte), 0, 100)
    return p

base = cv(M); resid = y - base
m = have
print("\n== Signal check (rows with an LLM score) ==")
print(f"corr(LLM score, manual grade)         Pearson={np.corrcoef(L[m], y[m])[0,1]:.2f}  Spearman={spearmanr(L[m], y[m])[0]:.2f}")
print(f"corr(LLM score, current model's error) Pearson={np.corrcoef(L[m], resid[m])[0,1]:.2f}   <- the key number")
print(f"corr(LLM score, current model's pred)  Pearson={np.corrcoef(L[m], base[m])[0,1]:.2f}   (high = mostly repeats what we have)")

X2 = np.hstack([M, L[:, None]]); p2 = cv(X2)
only = np.zeros(len(y))
for k in range(5):
    tr, te = (fold != k) & m, (fold == k) & m
    only[(fold == k)] = np.clip(T.make_ridge().fit(L[tr][:, None], y[tr]).predict(np.nan_to_num(L[fold == k], nan=np.nanmean(L))[:, None]), 0, 100)
res = {"constant mean": T.metrics(y, np.full(len(y), y.mean())), "LLM score only (calibrated by Ridge)": T.metrics(y, only),
       "current blend (no LLM)": T.metrics(y, base), "current blend + LLM score": T.metrics(y, p2)}
print("\n== 5-fold CV, same folds ==")
print(pd.DataFrame(res).T.round(2).to_string())
d = res["current blend (no LLM)"]["RMSE"] - res["current blend + LLM score"]["RMSE"]
print(f"\nRMSE change from adding the LLM score: {-d:+.2f}")
print("Rule of thumb: promising only if the change is about -0.3 or better AND corr(LLM, error) is clearly above ~0.2.")
