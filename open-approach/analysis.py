"""
Report analyses for the Open Approach (RQ1 robustness / explainability, RQ3 ablation).
Run:  python analysis.py        (writes results_*.csv next to this file)
"""
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
import open_approach as oa

df = pd.read_csv(oa.TRAIN_CSV); X = oa.build_X(df); y = df.manual_grade.values
strat = pd.cut(y, [-1, 50, 70, 85, 100], labels=False)
SEEDS = [0, 1, 2, 3, 4]

TEXT = ["char_lev_norm", "tok_lev", "tok_lev_norm", "len_q", "len_r", "len_ratio"]
AST = ["ins", "rem", "upd", "mov", "keep", "nq", "nr", "ast_edits", "ast_edit_norm", "ast_edit_frac_r"]
CLAUSE = [c for c in X.columns if c.startswith("d_")]
GROUPS = {"text only": TEXT, "AST only": AST + ["parse_fail"], "clause-diff only": CLAUSE,
          "text + AST": TEXT + AST + ["parse_fail"], "all features": list(X.columns)}

def oof(make, cols, seed):
    p = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, strat):
        m = make().fit(X.iloc[tr][cols], y[tr]); p[te] = np.clip(m.predict(X.iloc[te][cols]), 0, 100)
    return p

def avg_metrics(make, cols):
    rows = [oa.metrics(y, oof(make, cols, s)) for s in SEEDS]
    return pd.DataFrame(rows).mean()

# 1) model comparison, repeated seeds
models = {"ridge": oa.make_ridge, "random forest": oa.make_rf, "blend 0.7 RF + 0.3 Ridge": oa.Blend}
comp = pd.DataFrame({n: avg_metrics(mk, list(X.columns)) for n, mk in models.items()}).T
comp.loc["constant mean"] = pd.DataFrame([oa.metrics(y, np.full(len(y), y.mean()))]).iloc[0]
print("== Model comparison (mean of %d seeds) ==" % len(SEEDS)); print(comp.round(2).to_string())
comp.round(3).to_csv("results_models.csv")

# 2) feature-group ablation with the blend (RQ3 candidate)
abl = pd.DataFrame({n: avg_metrics(oa.Blend, c) for n, c in GROUPS.items()}).T
print("\n== Feature-group ablation (blend) =="); print(abl.round(2).to_string())
abl.round(3).to_csv("results_ablation.csv")

# 3) blend weight sweep
sw = {}
for w in [0.0, 0.3, 0.5, 0.7, 1.0]:
    sw[f"w_rf={w}"] = avg_metrics(lambda w=w: oa.Blend(w), list(X.columns))
sw = pd.DataFrame(sw).T
print("\n== RF weight in blend =="); print(sw.round(2).to_string()); sw.round(3).to_csv("results_blend_weight.csv")

# 4) underprediction shift sweep (out-of-fold predictions of the blend)
P = np.mean([oof(oa.Blend, list(X.columns), s) for s in SEEDS], axis=0)
rows = {}
for shift in [0, 1, 2, 3, 5, 8]:
    rows[f"+{shift}"] = oa.metrics(y, np.clip(P + shift, 0, 100))
shf = pd.DataFrame(rows).T
print("\n== Underprediction shift (OOF, seed-averaged predictions) =="); print(shf.round(2).to_string())
shf.round(3).to_csv("results_shift.csv")

# 5) residuals by homework and database
res = df[["hw_num", "db_instance_path"]].copy(); res["err"] = P - y
res["db"] = res.db_instance_path.astype(str).str.split("/").str[-1]
by = pd.concat([res.groupby("hw_num").err.agg(n="count", bias="mean", rmse=lambda e: np.sqrt((e**2).mean())),
                ], axis=1)
bydb = res.groupby("db").err.agg(n="count", bias="mean", rmse=lambda e: np.sqrt((e**2).mean()))
print("\n== Residuals by homework =="); print(by.round(2).to_string())
print("\n== Residuals by database =="); print(bydb.round(2).to_string())
by.round(3).to_csv("results_by_hw.csv"); bydb.round(3).to_csv("results_by_db.csv")

# 6) explainability: RF importances and Ridge coefficients
rf = oa.make_rf().fit(X, y); rd = oa.make_ridge().fit(X, y)
imp = pd.Series(rf.feature_importances_, X.columns).sort_values(ascending=False)
coef = pd.Series(rd[-1].coef_, X.columns)
expl = pd.DataFrame({"rf_importance": imp, "ridge_coef_per_std": coef.reindex(imp.index)})
print("\n== Top 12 features =="); print(expl.head(12).round(3).to_string())
expl.round(4).to_csv("results_explain.csv")
