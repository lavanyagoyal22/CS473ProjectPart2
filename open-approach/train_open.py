"""LOCAL ONLY (do not put in the submission ZIP). Trains the Open Approach and writes
submission/weights/approach6.joblib. Run with the grader-pinned environment.

  python train_open.py cv          # 5-fold CV, writes folds.csv, prints metrics
  python train_open.py train       # fit on all rows and save weights
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, joblib
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent / "submission"))
from open_features import feature_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import ExtraTreesRegressor

CSV = "training_data_Part2.csv"; SEED = 42; W_RF = 0.7; SHIFT = 0.0
OUT = Path(__file__).resolve().parent / "submission" / "weights" / "approach6.joblib"

def make_rf(): return RandomForestRegressor(300, min_samples_leaf=5, max_features=0.4, random_state=SEED, n_jobs=1)
def make_ridge(): return make_pipeline(StandardScaler(), RidgeCV(np.logspace(-1, 4, 25)))

def metrics(y, p):
    return dict(RMSE=np.sqrt(np.mean((p - y) ** 2)), MAE=np.mean(abs(p - y)), Pearson=np.corrcoef(p, y)[0, 1],
                under=int((p < y).sum()), over=int((p > y).sum()))

def load():
    df = pd.read_csv(CSV); M, names = feature_matrix(df.q, df.r)
    assert not np.isnan(M).any(); return df, M, names, df.manual_grade.values.astype(float)

def blend_pred(rf, rd, X): return np.clip(W_RF * rf.predict(X) + (1 - W_RF) * rd.predict(X) + SHIFT, 0, 100)

def cv():
    df, M, names, y = load()
    strat = pd.cut(y, [-1, 50, 70, 85, 100], labels=False); fold = np.zeros(len(y), int)
    for k, (_, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(M, strat)): fold[te] = k
    pd.DataFrame({"id": df["id"], "fold": fold}).to_csv("folds.csv", index=False)
    rows = {"constant": None, "ridge": "ridge", "rf": "rf", "blend": "blend"}; res = {}
    for name, kind in rows.items():
        p = np.zeros(len(y))
        for k in range(5):
            tr, te = fold != k, fold == k
            if kind is None: p[te] = y[tr].mean(); continue
            rf = make_rf().fit(M[tr], y[tr]); rd = make_ridge().fit(M[tr], y[tr])
            p[te] = {"ridge": np.clip(rd.predict(M[te]), 0, 100), "rf": np.clip(rf.predict(M[te]), 0, 100),
                     "blend": blend_pred(rf, rd, M[te])}[kind]
        res[name] = metrics(y, p)
    print(pd.DataFrame(res).T.round(2).to_string())

def train():
    df, M, names, y = load()
    art = dict(rf=make_rf().fit(M, y), ridge=make_ridge().fit(M, y), names=names, w_rf=W_RF, shift=SHIFT,
               train_mean=float(y.mean()))
    OUT.parent.mkdir(exist_ok=True); joblib.dump(art, OUT, compress=3); print("saved", OUT, OUT.stat().st_size, "bytes")

if __name__ == "__main__":
    {"cv": cv, "train": train}[sys.argv[1] if len(sys.argv) > 1 else "cv"]()
