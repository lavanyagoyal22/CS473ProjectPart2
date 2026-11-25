"""LOCAL. Is the LLM-score gain real, or just luck in one fold split?
Repeats 5-fold CV over 10 different random splits and compares the blend with and without the LLM score.
  python llm_seed_check.py llm_scores_gpt_oss_120b.csv
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent / "submission"))
from open_features import feature_matrix
from sklearn.model_selection import StratifiedKFold
import train_open as T

df = pd.read_csv(T.CSV); y = df.manual_grade.values.astype(float); M, _ = feature_matrix(df.q, df.r)
L = pd.read_csv(sys.argv[1]).drop_duplicates("id").set_index("id").reindex(df["id"])["llm_score"].values.astype(float)
L = np.where(np.isnan(L), np.nanmean(L), L)
X2 = np.hstack([M, L[:, None]]); strat = pd.cut(y, [-1, 50, 70, 85, 100], labels=False)

def rmse(X, seed):
    p = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, strat):
        rf = T.make_rf().fit(X[tr], y[tr]); rd = T.make_ridge().fit(X[tr], y[tr])
        p[te] = np.clip(0.7 * rf.predict(X[te]) + 0.3 * rd.predict(X[te]), 0, 100)
    return np.sqrt(np.mean((p - y) ** 2))

rows = []
for s in range(10):
    a, b = rmse(M, s), rmse(X2, s); rows.append((s, a, b, b - a)); print(f"seed {s}: without={a:.2f}  with LLM={b:.2f}  change={b-a:+.2f}", flush=True)
d = np.array([r[3] for r in rows])
print(f"\nmean change {d.mean():+.2f}  (std {d.std():.2f}); LLM score helped in {(d<0).sum()}/10 splits")
print("Interpretation: a gain you can trust helps in at least 8/10 splits with a mean change of about -0.25 or better.")
