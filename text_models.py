"""Approach 2: Text edit distance (Levenshtein) between q and r  ->  partial-credit grade.

Pipeline per (q, r) pair
  1. preprocess both queries  (strip comments / trailing ';', collapse whitespace, optional lower-casing)
  2. Levenshtein distance on  characters  OR  SQL tokens   (insert / delete / substitute, cost 1 each)
  3. normalise the distance   (raw | by max(len) | by len(q)+len(r) | by len(q) | by len(r))
  4. map the normalised distance to a grade with a tiny model fitted on training folds only
Saved artifact = plain dict (numbers/lists).  Inference needs only numpy + joblib
(+ python-Levenshtein if installed; otherwise a pure-Python fallback with identical results).
"""
import re
from pathlib import Path

import numpy as np

try:  # fast C implementation (in the course environment)
    import Levenshtein as _lev
except Exception:  # pragma: no cover
    _lev = None

ARTIFACT_NAME = "approach2.joblib"
VERSION = 1

_COMMENT_RE = re.compile(r"--[^\n]*|/\*.*?\*/", re.S)
_TOKEN_RE = re.compile(
    r"[A-Za-z_][A-Za-z_0-9]*|\d+\.?\d*|'[^']*'|\"[^\"]*\"|`[^`]*`|<=|>=|<>|!=|\|\||[^\s\w]"
)


# ============================================================================ preprocessing
def preprocess(sql, case):
    s = _COMMENT_RE.sub(" ", str(sql) if sql is not None else "")
    s = s.strip()
    while s.endswith(";"):
        s = s[:-1].rstrip()
    s = re.sub(r"\s+", " ", s)
    return s.lower() if case == "lower" else s


def tokenize(s):
    return _TOKEN_RE.findall(s)


# ============================================================================ distance
def _lev_py(a, b):
    """Pure-Python Levenshtein on two sequences (fallback; trims common prefix/suffix)."""
    i = 0
    while i < len(a) and i < len(b) and a[i] == b[i]:
        i += 1
    a, b = a[i:], b[i:]
    while a and b and a[-1] == b[-1]:
        a, b = a[:-1], b[:-1]
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for x, ca in enumerate(a, 1):
        cur = [x]
        for y, cb in enumerate(b, 1):
            cur.append(min(prev[y] + 1, cur[y - 1] + 1, prev[y - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def lev(a, b):
    """Levenshtein distance of two strings (python-Levenshtein if available)."""
    return _lev.distance(a, b) if _lev is not None else _lev_py(a, b)


def pair_stats(q, r, unit, case):
    """(distance, len_q, len_r) in units of characters or SQL tokens."""
    pq, pr = preprocess(q, case), preprocess(r, case)
    if unit == "char":
        return lev(pq, pr), len(pq), len(pr)
    tq, tr = tokenize(pq), tokenize(pr)
    vocab = {}
    enc = lambda toks: "".join(chr(0x100 + vocab.setdefault(t, len(vocab))) for t in toks)
    a, b = enc(tq), enc(tr)          # one unicode char per distinct token -> token-level Levenshtein
    return lev(a, b), len(tq), len(tr)


def normalise(d, lq, lr, norm):
    den = {"raw": 1.0, "max": max(lq, lr), "sum": lq + lr, "q": lq, "r": lr}[norm]
    return float(d) / den if den > 0 else 0.0


# ============================================================================ configs
def config_grid():
    grid = []
    for unit in ("char", "token"):
        for case in ("keep", "lower"):
            for norm in ("raw", "max", "sum", "q", "r"):
                maps = ["linear", "isotonic", "sqrt_linear"] + ([] if norm == "raw" else ["identity"])
                for mapping in maps:
                    grid.append(dict(unit=unit, case=case, norm=norm, mapping=mapping))
    return grid


def cfg_name(c):
    return f"unit={c['unit']} case={c['case']} norm={c['norm']} map={c['mapping']}"


# ============================================================================ mapping
def _feats(x, feat):
    return [x] if feat == "x" else [x, float(np.sqrt(max(x, 0.0)))]


def fit_mapping(x, y, cfg):
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LinearRegression
    x, y, m = np.asarray(x, float), np.asarray(y, float), cfg["mapping"]
    if m == "identity":                       # no fitting: grade = 100 * (1 - normalised distance)
        return {"type": "identity"}
    if m == "isotonic":
        iso = IsotonicRegression(increasing=False, out_of_bounds="clip", y_min=0.0, y_max=100.0).fit(x, y)
        return {"type": "isotonic", "xs": [float(v) for v in iso.X_thresholds_],
                "ys": [float(v) for v in iso.y_thresholds_]}
    feat = "x" if m == "linear" else "x_sqrt"
    X = np.array([_feats(v, feat) for v in x])
    mod = LinearRegression().fit(X, y)
    return {"type": "linear", "feat": feat, "coef": [float(c) for c in mod.coef_],
            "intercept": float(mod.intercept_)}


def apply_mapping(mp, x):
    if mp["type"] == "identity":
        return 100.0 * (1.0 - x)
    if mp["type"] == "isotonic":
        return float(np.interp(x, mp["xs"], mp["ys"]))
    return float(mp["intercept"] + np.dot(mp["coef"], _feats(x, mp["feat"])))


def _clip(v):
    return float(min(100.0, max(0.0, v))) if v is not None and np.isfinite(v) else None


# ============================================================================ fit / score
def fit_artifact(cfg, stats, y):
    """stats: list of (d, lq, lr) already computed for cfg['unit'], cfg['case'] (training rows only)."""
    x = [normalise(d, lq, lr, cfg["norm"]) for d, lq, lr in stats]
    return {"version": VERSION, "cfg": cfg, "mapping": fit_mapping(x, y, cfg),
            "train_mean": float(np.mean(y)), "n_train": int(len(y))}


def score_stats(art, st):
    cfg = art["cfg"]
    s = _clip(apply_mapping(art["mapping"], normalise(*st, cfg["norm"])))
    return s if s is not None else _clip(art["train_mean"])


# ============================================================================ model class
class TextEditModel:
    def __init__(self, artifact):
        self.art = artifact

    def _score(self, q, r):
        cfg = self.art["cfg"]
        return score_stats(self.art, pair_stats(q, r, cfg["unit"], cfg["case"]))

    def predict(self, test_rows, questions=None):
        out = []
        for row in test_rows:
            q = row.get("q", row.get("text", ""))
            try:
                score = self._score(q, row.get("r", ""))
            except Exception:                       # never fail a row
                score = _clip(self.art["train_mean"])
            out.append({"id": row["id"], "score": float(score)})
        return out

    def explain(self, q, r):
        """Distance, normalised distance and predicted grade + the token-level edits."""
        import difflib
        cfg = self.art["cfg"]
        d, lq, lr = pair_stats(q, r, cfg["unit"], cfg["case"])
        a = tokenize(preprocess(q, cfg["case"]))
        b = tokenize(preprocess(r, cfg["case"]))
        ops = [(t, a[i1:i2], b[j1:j2]) for t, i1, i2, j1, j2 in
               difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes() if t != "equal"]
        return {"score": self._score(q, r), "distance": d, "len_q": lq, "len_r": lr,
                "normalised": normalise(d, lq, lr, cfg["norm"]),
                "edits": [{"op": t, "from": " ".join(x), "to": " ".join(y)} for t, x, y in ops]}


def load_text_model(weights_dir):
    import joblib
    for base in (Path(weights_dir), Path(__file__).resolve().parent / "weights"):
        p = base / ARTIFACT_NAME
        if p.exists():
            return TextEditModel(joblib.load(p))
    raise FileNotFoundError(f"{ARTIFACT_NAME} not found in {weights_dir}; run train_approach2.py first")
