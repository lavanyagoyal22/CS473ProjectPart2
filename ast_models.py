"""Models for Approach 1 (AST edit distance, IDF-weighted AST edits, clause-weighted AST edits).

Pipeline for every (q, r) pair:
  1. ast_utils.extract_pair  -> AST edit script (ins / rem / upd / mov, one edit per node)
  2. weight each edit        -> Option A: every node costs 1
                                Option B: node cost = IDF of its label (rare => expensive)
                                Option C: like B/A, but costs are kept separate per SQL clause and
                                          a *learned per-clause penalty* is applied (rubric-like)
  3. normalise by total tree weight of q + r  (percent of the trees that had to change)
  4. map to a grade with a small model fitted on training folds only
Saved artifacts are plain dicts (numbers / lists), so no custom classes are pickled.
Inference needs only numpy + joblib + sqlglot.
"""
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ast_utils import extract_pair

OPS = ["ins", "rem", "upd", "mov"]
CLAUSES = ["select", "from", "join", "where", "group", "having", "order",
           "limit", "distinct", "setop", "other"]
ARTIFACT_NAMES = {"option1": "approach1_option1.joblib",
                  "option2": "approach1_option2.joblib",
                  "option3": "approach1_option3.joblib"}
VERSION = 1


# ============================================================================ configs
def config_grid(option):
    """All configurations tried by cross-validation for a given option."""
    grid = []
    if option == "option1":
        for norm in ("raw", "size"):
            for mapping, mv in (("sum_linear", 0.0), ("sum_linear", 1.0),
                                ("sum_isotonic", 0.0), ("sum_isotonic", 1.0),
                                ("vec_poslin", 1.0)):
                grid.append(dict(wmode="unit", unit=None, gran="c", norm=norm,
                                 mapping=mapping, move_w=mv, alpha=0.0))
    elif option == "option2":
        for unit in ("q", "d"):          # IDF document = question  /  single query
            for gran in ("c", "f"):      # label = node type  /  node type + leaf value
                for norm in ("raw", "size"):
                    for mapping, mv in (("sum_linear", 0.0), ("sum_linear", 1.0),
                                        ("sum_isotonic", 0.0), ("sum_isotonic", 1.0),
                                        ("vec_poslin", 1.0)):
                        grid.append(dict(wmode="idf", unit=unit, gran=gran, norm=norm,
                                         mapping=mapping, move_w=mv, alpha=0.0))
    elif option == "option3":
        wm = [("unit", None, "c")] + [("idf", u, g) for u in ("q", "d") for g in ("c", "f")]
        for wmode, unit, gran in wm:
            for norm in ("raw", "size"):
                for alpha in (0.01, 0.1, 1.0, 10.0):
                    grid.append(dict(wmode=wmode, unit=unit, gran=gran, norm=norm,
                                     mapping="clause_ridge", move_w=1.0, alpha=alpha))
    else:
        raise ValueError(option)
    return grid


def cfg_name(c):
    return (f"{c['wmode']}/{c['unit'] or '-'}/{c['gran']} norm={c['norm']} "
            f"map={c['mapping']} move_w={c['move_w']} alpha={c['alpha']}")


# ============================================================================ IDF
def fit_idf(recs, keys, unit, gran):
    """IDF of node labels.  Fit on TRAINING rows only.

    unit="q": a document = all labels of all queries of one question (hw:q)
              -> labels that only occur in some questions (special join, aggregate, ...) get high IDF
    unit="d": a document = a single query (q and r counted separately)
    idf = ln((N+1)/(df+1)) + 1 ; labels never seen in training get the maximum (df = 0).
    """
    docs = []
    if unit == "q":
        by = defaultdict(set)
        for rec, k in zip(recs, keys):
            if rec["ok"]:
                by[k].update(rec["q_" + gran])
                by[k].update(rec["r_" + gran])
        docs = list(by.values())
    else:
        for rec in recs:
            if rec["ok"]:
                docs.append(set(rec["q_" + gran]))
                docs.append(set(rec["r_" + gran]))
    n = len(docs)
    df = Counter()
    for d in docs:
        df.update(d)
    table = {lab: math.log((n + 1) / (c + 1)) + 1.0 for lab, c in df.items()}
    return {"table": table, "default": math.log(n + 1) + 1.0, "n_docs": n}


# ============================================================================ features
def _weight_fn(idf):
    if idf is None:
        return lambda lab: 1.0
    tab, dflt = idf["table"], idf["default"]
    return lambda lab: tab.get(lab, dflt)


def edit_costs(rec, cfg, idf):
    """Per-edit costs. Returns (list of (op, clause, labelA, labelB, cost)), total tree weight W."""
    g = cfg["gran"]
    w = _weight_fn(idf)
    rows = []
    for op, lc, lf, lc2, lf2, clause in rec["edits"]:
        a, b = (lc, lc2) if g == "c" else (lf, lf2)
        cost = w(a) if op != "upd" else 0.5 * (w(a) + w(b))
        if op == "mov":
            cost *= cfg["move_w"]
        rows.append((op, clause if clause in CLAUSES else "other", a, b, cost))
    total = sum(w(l) for l in rec["q_" + g]) + sum(w(l) for l in rec["r_" + g])
    return rows, total


def featurize(rec, cfg, idf):
    rows, total = edit_costs(rec, cfg, idf)
    ops, cl = dict.fromkeys(OPS, 0.0), dict.fromkeys(CLAUSES, 0.0)
    for op, clause, _, _, cost in rows:
        ops[op] += cost
        cl[clause] += cost
    den = (total / 100.0) if (cfg["norm"] == "size" and total > 0) else 1.0
    m = cfg["mapping"]
    if m.startswith("sum"):
        return np.array([sum(ops.values()) / den])
    if m == "vec_poslin":
        return np.array([ops[o] / den for o in OPS])
    if m == "clause_ridge":
        return np.array([cl[c] / den for c in CLAUSES])
    raise ValueError(m)


# ============================================================================ mapping
def fit_mapping(X, y, cfg):
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LinearRegression, Ridge
    m = cfg["mapping"]
    if m == "sum_linear":
        mod = LinearRegression().fit(X, y)
        return {"type": "linear", "coef": [float(c) for c in mod.coef_], "intercept": float(mod.intercept_)}
    if m == "sum_isotonic":
        iso = IsotonicRegression(increasing=False, out_of_bounds="clip", y_min=0.0, y_max=100.0)
        iso.fit(X[:, 0], y)
        return {"type": "isotonic", "xs": [float(v) for v in iso.X_thresholds_],
                "ys": [float(v) for v in iso.y_thresholds_]}
    # penalties must be >= 0:  grade = intercept - sum(penalty_i * cost_i)
    if m == "vec_poslin":
        mod = LinearRegression(positive=True).fit(-X, y)
    elif m == "clause_ridge":
        mod = Ridge(alpha=cfg["alpha"], positive=True).fit(-X, y)
    else:
        raise ValueError(m)
    return {"type": "linear", "coef": [float(-c) for c in mod.coef_], "intercept": float(mod.intercept_)}


def apply_mapping(mp, x):
    if mp["type"] == "isotonic":
        return float(np.interp(x[0], mp["xs"], mp["ys"]))
    return float(mp["intercept"] + np.dot(mp["coef"], x))


def _clip(v):
    if v is None or not np.isfinite(v):
        return None
    return float(min(100.0, max(0.0, v)))


# ============================================================================ fit / score
def fit_artifact(option, cfg, recs, keys, y):
    """Fit everything on the given (training) records only."""
    y = np.asarray(y, dtype=float)
    ok = np.array([r["ok"] for r in recs])
    idf = None
    if cfg["wmode"] == "idf":
        idf = fit_idf(recs, keys, cfg["unit"], cfg["gran"])
    X = np.array([featurize(r, cfg, idf) for r, o in zip(recs, ok) if o])
    mapping = fit_mapping(X, y[ok], cfg)
    # documented fallback for unparsable queries: grade ~ a + b * token_distance
    tok = np.array([[r["tok"]] for r in recs])
    from sklearn.linear_model import LinearRegression
    fb = LinearRegression().fit(tok, y)
    return {"version": VERSION, "option": option, "cfg": cfg, "idf": idf, "mapping": mapping,
            "fallback": {"coef": float(fb.coef_[0]), "intercept": float(fb.intercept_)},
            "train_mean": float(y.mean()), "n_train": int(len(y)), "n_parsed": int(ok.sum())}


def score_record(art, rec):
    """(score in [0,100], used_fallback)"""
    if rec["ok"]:
        try:
            s = _clip(apply_mapping(art["mapping"], featurize(rec, art["cfg"], art["idf"])))
            if s is not None:
                return s, False
        except Exception:
            pass
    fb = art["fallback"]
    s = _clip(fb["intercept"] + fb["coef"] * rec["tok"])
    return (s if s is not None else _clip(art["train_mean"])), True


# ============================================================================ model class
class ASTModel:
    """Object returned by load_model(); implements predict(test_rows, questions)."""

    def __init__(self, artifact):
        self.art = artifact

    def predict(self, test_rows, questions=None):
        out = []
        for row in test_rows:
            q = row.get("q", row.get("text", "")) or ""
            r = row.get("r", "") or ""
            try:
                score, _ = score_record(self.art, extract_pair(q, r))
            except Exception:
                score = _clip(self.art["train_mean"])
            out.append({"id": row["id"], "score": float(score)})
        return out

    def explain(self, q, r):
        """Human-readable breakdown of one prediction (for the report / syllabus-style rules)."""
        rec = extract_pair(q, r)
        art = self.art
        score, fb = score_record(art, rec)
        info = {"score": score, "used_fallback": fb, "parse_error": rec["err"], "edits": []}
        if rec["ok"]:
            idf = art["idf"]
            rows, total = edit_costs(rec, art["cfg"], idf)
            info["total_tree_weight"] = total
            info["edits"] = [dict(op=o, clause=c, node=a if o != "upd" else f"{a} -> {b}",
                                  cost=round(k, 3)) for o, c, a, b, k in rows]
        return info


def load_ast_model(option, weights_dir):
    import joblib
    name = ARTIFACT_NAMES[option]
    for base in (Path(weights_dir), Path(__file__).resolve().parent / "weights"):
        p = base / name
        if p.exists():
            return ASTModel(joblib.load(p))
    raise FileNotFoundError(f"{name} not found in {weights_dir}; run train_approach1.py first")
