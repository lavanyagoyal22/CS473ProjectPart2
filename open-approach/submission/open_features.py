"""Feature extraction for the Open Approach (Approach 6). Uses only q and r.
Pure functions, no training, no file access. Must be identical at training and inference time."""
import re
import numpy as np
import sqlglot
from sqlglot import exp
from sqlglot.diff import diff, Insert, Remove, Update, Move
from rapidfuzz.distance import Levenshtein

CLAUSES = [exp.Select, exp.From, exp.Join, exp.Where, exp.Group, exp.Having, exp.Order, exp.Limit,
           exp.Subquery, exp.Union, exp.Intersect, exp.Except, exp.Distinct, exp.Count, exp.Avg,
           exp.Sum, exp.Min, exp.Max, exp.Like, exp.In, exp.Between, exp.Is, exp.Case, exp.Alias,
           exp.Column, exp.Literal]

FEATURE_NAMES = None  # filled on first call to features()

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
    q = "" if q is None else str(q)
    r = "" if r is None else str(r)
    qt, rt = parse(q), parse(r)
    f = {"parse_fail": float(qt is None or rt is None)}
    qs, rs = " ".join(q.split()).lower(), " ".join(r.split()).lower()
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

def feature_matrix(qs, rs):
    """Returns (matrix, names). A row that raises is replaced by NaN and later imputed by the caller."""
    rows, names = [], None
    for q, r in zip(qs, rs):
        try:
            f = features(q, r)
        except Exception:
            f = None
        rows.append(f)
        if f is not None and names is None:
            names = list(f.keys())
    if names is None:
        names = []
    M = np.full((len(rows), len(names)), np.nan)
    for i, f in enumerate(rows):
        if f is not None:
            M[i] = [f[n] for n in names]
    return M, names
