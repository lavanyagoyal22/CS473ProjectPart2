"""
CS473 Part 2 - Approach 6 (Open Approach): learned regression on AST + token + complexity features.
Uses ONLY q and r (no query execution, no manual_explanation, no question id, no LLM).

  python open_approach.py cv       # 5-fold CV (fixed folds written to folds.csv), prints metrics
  python open_approach.py train    # fit final model on all data -> open_model.joblib
  python open_approach.py predict in.csv out.csv   # in.csv needs columns q, r
"""
import sys, re, warnings
import numpy as np, pandas as pd, joblib
warnings.filterwarnings("ignore")
import sqlglot
from sqlglot import exp
from sqlglot.diff import diff, Insert, Remove, Update, Move
from rapidfuzz.distance import Levenshtein
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

TRAIN_CSV = "training_data_Part2.csv"
MODEL_PATH = "open_model.joblib"
FOLDS_PATH = "folds.csv"
SEED = 42
UNDER_SHIFT = 0.0   # points added to predictions to reduce underprediction (tune; report tradeoff)

CLAUSES = [exp.Select, exp.From, exp.Join, exp.Where, exp.Group, exp.Having, exp.Order, exp.Limit,
           exp.Subquery, exp.Union, exp.Intersect, exp.Except, exp.Distinct, exp.Count, exp.Avg,
           exp.Sum, exp.Min, exp.Max, exp.Like, exp.In, exp.Between, exp.Is, exp.Case, exp.Alias,
           exp.Column, exp.Literal]

def parse(s):
    try:
        return sqlglot.parse_one(str(s).strip().rstrip(";"), read="sqlite")
    except Exception:
        return None

def tokenize(s):
    return re.findall(r"[A-Za-z_][A-Za-z_0-9\.]*|\d+\.?\d*|'[^']*'|<=|>=|<>|!=|\|\||\S", s)

def clause_counts(t):
    c = np.zeros(len(CLAUSES))
    if t is not None:
        for n in t.walk():
            for i, k in enumerate(CLAUSES):
                if type(n) is k:
                    c[i] += 1
    return c

def features(q, r):
    qt, rt = parse(q), parse(r)
    f = {"parse_fail": float(qt is None or rt is None)}
    qs, rs = " ".join(str(q).split()).lower(), " ".join(str(r).split()).lower()
    tq, tr = tokenize(qs), tokenize(rs)
    f["char_lev_norm"] = Levenshtein.normalized_distance(qs, rs)
    f["tok_lev"] = Levenshtein.distance(tq, tr)
    f["tok_lev_norm"] = Levenshtein.normalized_distance(tq, tr)
    f.update(len_q=len(tq), len_r=len(tr), len_ratio=len(tq) / max(len(tr), 1))
    ins = rem = upd = mov = keep = 0
    if qt is not None and rt is not None:
        try:
            for e in diff(qt, rt, delta_only=False):
                if isinstance(e, Insert): ins += 1
                elif isinstance(e, Remove): rem += 1
                elif isinstance(e, Update): upd += 1
                elif isinstance(e, Move): mov += 1
                else: keep += 1
        except Exception:
            pass
    nq = len(list(qt.walk())) if qt is not None else 0
    nr = len(list(rt.walk())) if rt is not None else 0
    ed = ins + rem + upd + mov
    f.update(ins=ins, rem=rem, upd=upd, mov=mov, keep=keep, nq=nq, nr=nr, ast_edits=ed,
             ast_edit_norm=ed / max(nq + nr, 1), ast_edit_frac_r=ed / max(nr, 1))
    d = clause_counts(rt) - clause_counts(qt)
    for i, k in enumerate(CLAUSES):
        f["d_" + k.__name__] = abs(d[i])
    f["d_total"] = np.abs(d).sum()
    return f

def build_X(df):
    return pd.DataFrame([features(a, b) for a, b in zip(df.q, df.r)])

def make_rf():
    return RandomForestRegressor(300, min_samples_leaf=5, max_features=0.4, random_state=SEED, n_jobs=-1)

def make_ridge():
    return make_pipeline(StandardScaler(), RidgeCV(np.logspace(-1, 4, 25)))

class Blend:
    """Average of RF and Ridge; weight w on RF."""
    def __init__(self, w=0.7): self.w = w
    def fit(self, X, y):
        self.cols = list(X.columns); self.rf = make_rf().fit(X, y); self.rd = make_ridge().fit(X, y); return self
    def predict(self, X):
        X = X.reindex(columns=self.cols, fill_value=0)
        return np.clip(self.w * self.rf.predict(X) + (1 - self.w) * self.rd.predict(X) + UNDER_SHIFT, 0, 100)

def metrics(y, p):
    return dict(RMSE=np.sqrt(np.mean((p - y) ** 2)), MAE=np.mean(abs(p - y)),
                Pearson=np.corrcoef(p, y)[0, 1], under=int((p < y).sum()), over=int((p > y).sum()))

def get_folds(df):
    y = df.manual_grade.values
    strat = pd.cut(y, [-1, 50, 70, 85, 100], labels=False)
    fold = np.zeros(len(df), int)
    for k, (_, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(df, strat)):
        fold[te] = k
    pd.DataFrame({"id": df["id"], "fold": fold}).to_csv(FOLDS_PATH, index=False)
    return fold

def cv():
    df = pd.read_csv(TRAIN_CSV); X = build_X(df); y = df.manual_grade.values
    fold = get_folds(df)
    models = {"constant": None, "ridge": make_ridge, "rf": make_rf, "blend(rf .7 + ridge .3)": Blend}
    rows = {}
    for name, mk in models.items():
        p = np.zeros(len(y))
        for k in range(5):
            tr, te = fold != k, fold == k
            if mk is None:
                p[te] = y[tr].mean()
            else:
                m = mk().fit(X[tr], y[tr]); p[te] = np.clip(m.predict(X[te]), 0, 100)
        rows[name] = metrics(y, p)
    print(pd.DataFrame(rows).T.round(2).to_string())
    print(f"\nfolds written to {FOLDS_PATH}")

def train(path=MODEL_PATH):
    df = pd.read_csv(TRAIN_CSV)
    m = Blend().fit(build_X(df), df.manual_grade.values)
    joblib.dump(m, path); print("saved", path)
    return m

def load_or_train():
    """Load the saved model; if it is missing or incompatible with installed library
    versions, retrain from the training CSV (fallback for the autograder)."""
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        print("could not load saved model (%s); retraining from %s" % (type(e).__name__, TRAIN_CSV))
        return train(MODEL_PATH)

def predict_df(df, model=None):
    """Main entry point. df needs columns q and r (other columns are ignored).
    Returns a numpy array of predicted grades in [0, 100]."""
    model = model or load_or_train()
    return model.predict(build_X(df))

def predict(inp, out):
    df = pd.read_csv(inp)
    df["predicted_grade"] = predict_df(df)
    df.to_csv(out, index=False); print("wrote", out)

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "cv"
    if cmd == "cv": cv()
    elif cmd == "train": train()
    elif cmd == "predict": predict(sys.argv[2], sys.argv[3])
    else: print(__doc__)
