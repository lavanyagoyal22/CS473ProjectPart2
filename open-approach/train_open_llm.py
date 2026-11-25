"""LOCAL ONLY. Trains the Open Approach WITH the LLM score and writes the weights the grader loads.
Needs llm_scores_<model>.csv from llm_score_train.py (all training rows scored).

  python train_open_llm.py llm_scores_gpt_oss_120b.csv            # CV check + fit + save
  python train_open_llm.py llm_scores_gpt_oss_120b.csv --out /tmp/w/approach6.joblib   # save elsewhere (testing)

The artifact holds two blends: one on the 44 edit features (used when no LLM score is available for a row)
and one on the edit features + the LLM score (used when it is).
"""
import argparse, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, joblib
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "submission"))
from open_features import feature_matrix
import open_llm as OL
import train_open as T

ap = argparse.ArgumentParser(); ap.add_argument("scores"); ap.add_argument("--out", default=str(T.OUT)); a = ap.parse_args()
df = pd.read_csv(T.CSV); y = df.manual_grade.values.astype(float); M, names = feature_matrix(df.q, df.r)
assert not np.isnan(M).any()
L = pd.read_csv(a.scores).drop_duplicates("id").set_index("id").reindex(df["id"])["llm_score"].values.astype(float)
missing = int(np.isnan(L).sum()); print(f"LLM scores: {len(L) - missing}/{len(L)} rows" + ("  (missing rows get the mean)" if missing else ""))
mean_score = float(np.nanmean(L)); L = np.where(np.isnan(L), mean_score, L); X2 = np.hstack([M, L[:, None]])

fold = pd.read_csv("folds.csv").set_index("id").loc[df["id"], "fold"].values
def cv(X):
    p = np.zeros(len(y))
    for k in range(5):
        tr, te = fold != k, fold == k
        rf = T.make_rf().fit(X[tr], y[tr]); rd = T.make_ridge().fit(X[tr], y[tr])
        p[te] = np.clip(0.7 * rf.predict(X[te]) + 0.3 * rd.predict(X[te]), 0, 100)
    return T.metrics(y, p)
print(pd.DataFrame({"blend (edit features only)": cv(M), "blend + LLM score": cv(X2)}).T.round(2).to_string())

art = dict(rf=T.make_rf().fit(M, y), ridge=T.make_ridge().fit(M, y), names=names, w_rf=T.W_RF, shift=T.SHIFT,
           train_mean=float(y.mean()),
           llm=dict(rf=T.make_rf().fit(X2, y), ridge=T.make_ridge().fit(X2, y), mean_score=mean_score,
                    model_id=OL.MODEL_ID, prompt_id=OL.PROMPT_ID, temperature=OL.TEMPERATURE, max_tokens=OL.MAX_TOKENS))
out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True); joblib.dump(art, out, compress=3)
print("saved", out, out.stat().st_size, "bytes")
