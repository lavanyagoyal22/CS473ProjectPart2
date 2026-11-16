"""Approach 3: TF-IDF Vector Similarity for SQL partial-credit prediction.

Pipeline
  1. Tokenize q and r as SQL (comments/semicolons dropped, keywords and
     identifiers case-folded, identifier quotes stripped, string/number
     literals kept as distinct tokens, operators such as >= kept whole).
  2. Build word n-grams (default 1-2) from the token stream.
  3. IDF is fit ONLY on the training queries (both q and r of every training
     row). Unseen n-grams at inference get the maximum IDF (df = 0).
  4. Each query -> sublinear-TF * IDF vector, L2-normalised; the raw score
     is cosine(q, r) in [0, 1].
  5. Extra TF-IDF-derived features (IDF mass of tokens/keywords removed or
     added by the repair, query-length ratios) feed a ridge regression fit on
     the training data that maps them to a grade in [0, 100].

Artifacts are stored as plain JSON (no pickle), so inference never needs
the training code or the test data to be fit on.
"""
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

ARTIFACT_NAME = "approach3_tfidf.json"

# ----------------------------------------------------------------------------
# Tokenization
# ----------------------------------------------------------------------------
_COMMENT_RE = re.compile(r"/\*.*?\*/|--[^\n]*", re.S)
_TOKEN_RE = re.compile(
    r"""
    '(?:[^']|'')*'            # string literal
  | "(?:[^"]|"")*"            # "quoted identifier"
  | `[^`]*`                   # `quoted identifier`
  | \[[^\]]*\]                # [quoted identifier]
  | \d+(?:\.\d+)?             # number
  | [A-Za-z_][A-Za-z_0-9$]*   # keyword / identifier
  | <=>|<>|!=|>=|<=|==|\|\|   # multi-char operators
  | [^\sA-Za-z0-9_]           # any other single symbol
    """,
    re.X,
)


def tokenize_sql(sql):
    """SQL string -> list of normalised tokens."""
    if sql is None or (isinstance(sql, float) and math.isnan(sql)):
        return []
    text = _COMMENT_RE.sub(" ", str(sql))
    tokens = []
    for tok in _TOKEN_RE.findall(text):
        c = tok[0]
        if c == ";":
            continue
        if c == "'":                       # literal: keep case, mark as literal
            tokens.append("'" + tok[1:-1].replace("''", "'"))
        elif c in '"`[':                   # quoted identifier -> plain identifier
            tokens.append(tok[1:-1].lower())
        elif tok == "<>":
            tokens.append("!=")
        elif tok == "==":
            tokens.append("=")
        else:
            tokens.append(tok.lower())
    return tokens


def ngrams(tokens, ngram_range=(1, 2)):
    lo, hi = ngram_range
    out = []
    for n in range(lo, hi + 1):
        for i in range(len(tokens) - n + 1):
            out.append(" ".join(tokens[i:i + n]))
    return out


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
SQL_KEYWORDS = frozenset("""select from where group by having order join inner left right
outer natural on using distinct limit union intersect except not in exists like between
and or is null as case when then else end count sum avg min max desc asc all any cross
with""".split())

FEATURE_NAMES = ["cos_unigram", "cos_uni_bigram", "idf_mass_removed",
                 "idf_mass_added", "idf_keyword_change", "length_diff_ratio",
                 "log_len_repaired"]
FEATURE_SETS = {"cosine": [0], "tfidf": [0, 1, 2, 3, 4], "full": list(range(7))}


class _Tfidf:
    """Plain TF-IDF cosine helper (one n-gram range). IDF fit on training only."""

    def __init__(self, ngram_range=(1, 1), sublinear_tf=True):
        self.ngram_range = tuple(ngram_range)
        self.sublinear_tf = sublinear_tf
        self.idf, self.n_docs, self.default_idf = {}, 0, 1.0

    def fit_idf(self, docs):
        df = Counter()
        for d in docs:
            df.update(set(ngrams(tokenize_sql(d), self.ngram_range)))
        n = len(docs)
        self.n_docs = n
        self.idf = {t: math.log((1 + n) / (1 + c)) + 1.0 for t, c in df.items()}
        self.default_idf = math.log(1 + n) + 1.0      # unseen term (df = 0)
        return self

    def vector(self, tokens):
        tf = Counter(ngrams(tokens, self.ngram_range))
        vec = {t: ((1.0 + math.log(c)) if self.sublinear_tf else float(c))
                  * self.idf.get(t, self.default_idf) for t, c in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values()))
        return {t: v / norm for t, v in vec.items()} if norm > 0 else {}

    def cosine(self, tq, tr):
        vq, vr = self.vector(tq), self.vector(tr)
        if not vq and not vr:
            return 1.0
        if len(vq) > len(vr):
            vq, vr = vr, vq
        return float(min(1.0, max(0.0, sum(v * vr.get(t, 0.0)
                                           for t, v in vq.items()))))


class TfidfSqlModel:
    """TF-IDF similarity features of (q, r) -> ridge regression -> grade.

    Every feature is derived from TF-IDF weights of the q/r tokens, so the
    grade stays explainable: grade = intercept + sum(coef_i * z_i), with
    z_i the standardised feature (coefficients are saved in the JSON).
    feature_set: 'cosine' (cosine only), 'tfidf' (+ IDF-mass of changed tokens
    and keywords), 'full' (+ two query-length features).
    """

    def __init__(self, feature_set="full", alpha=10.0, sublinear_tf=True):
        self.feature_set = feature_set
        self.alpha = alpha
        self.sublinear_tf = sublinear_tf
        self.uni = _Tfidf((1, 1), sublinear_tf)
        self.bi = _Tfidf((1, 2), sublinear_tf)
        self.mu = self.sd = self.coef = None
        self.intercept = 73.0
        self.mean_grade = 73.0
        self.n_docs = 0

    @property
    def cols(self):
        return FEATURE_SETS[self.feature_set]

    # --- features -----------------------------------------------------------
    def features(self, q, r):
        tq, tr = tokenize_sql(q), tokenize_sql(r)
        cq, cr = Counter(tq), Counter(tr)
        w = lambda t: self.uni.idf.get(t, self.uni.default_idf)
        total = sum(w(t) * c for t, c in (cq + cr).items()) or 1.0
        removed = sum(w(t) * max(0, cq[t] - cr[t]) for t in cq) / total
        added = sum(w(t) * max(0, cr[t] - cq[t]) for t in cr) / total
        kq = Counter(t for t in tq if t in SQL_KEYWORDS)
        kr = Counter(t for t in tr if t in SQL_KEYWORDS)
        ktot = sum(w(t) * c for t, c in (kq + kr).items()) or 1.0
        kw = sum(w(t) * abs(kq[t] - kr[t]) for t in set(kq) | set(kr)) / ktot
        n = max(len(tq), len(tr), 1)
        f = [self.uni.cosine(tq, tr), self.bi.cosine(tq, tr), removed, added, kw,
             abs(len(tq) - len(tr)) / n, math.log1p(len(tr))]
        return np.array([f[i] for i in self.cols], float)

    def similarity(self, q, r):
        """Raw unigram TF-IDF cosine similarity of q and r, in [0, 1]."""
        return self.uni.cosine(tokenize_sql(q), tokenize_sql(r))

    # --- fit / predict --------------------------------------------------------
    def fit(self, q_list, r_list, grades):
        q_list, r_list = list(q_list), list(r_list)
        docs = q_list + r_list
        self.uni.fit_idf(docs)
        self.bi.fit_idf(docs)
        self.n_docs = len(docs)
        X = np.array([self.features(q, r) for q, r in zip(q_list, r_list)])
        y = np.asarray(grades, float)
        self.mean_grade = float(y.mean())
        self.mu = X.mean(0)
        self.sd = X.std(0) + 1e-9
        Z = (X - self.mu) / self.sd
        # closed-form ridge with unpenalised intercept
        A = Z.T @ Z + self.alpha * np.eye(Z.shape[1])
        self.coef = np.linalg.solve(A, Z.T @ (y - self.mean_grade))
        self.intercept = self.mean_grade
        return self

    def predict_pair(self, q, r):
        z = (self.features(q, r) - self.mu) / self.sd
        return float(min(100.0, max(0.0, self.intercept + z @ self.coef)))

    def predict_pairs(self, q_list, r_list):
        return [self.predict_pair(q, r) for q, r in zip(q_list, r_list)]

    def explain(self, q, r):
        """Per-feature contribution (grade points) to the prediction."""
        z = (self.features(q, r) - self.mu) / self.sd
        names = [FEATURE_NAMES[i] for i in self.cols]
        d = {"intercept": self.intercept}
        d.update({n: float(c * v) for n, c, v in zip(names, self.coef, z)})
        return d

    # --- persistence ----------------------------------------------------------
    def to_dict(self):
        def pack(t):
            return {"ngram_range": list(t.ngram_range), "n_docs": t.n_docs,
                    "default_idf": t.default_idf, "idf": t.idf}
        return {"version": 2, "feature_set": self.feature_set,
                "alpha": self.alpha, "sublinear_tf": self.sublinear_tf,
                "uni": pack(self.uni), "bi": pack(self.bi),
                "mu": self.mu.tolist(), "sd": self.sd.tolist(),
                "coef": self.coef.tolist(), "intercept": self.intercept,
                "mean_grade": self.mean_grade, "n_docs": self.n_docs}

    @classmethod
    def from_dict(cls, d):
        m = cls(d["feature_set"], d["alpha"], d["sublinear_tf"])
        for name, t in (("uni", m.uni), ("bi", m.bi)):
            t.ngram_range = tuple(d[name]["ngram_range"])
            t.n_docs, t.default_idf = d[name]["n_docs"], d[name]["default_idf"]
            t.idf = d[name]["idf"]
        m.mu, m.sd = np.array(d["mu"]), np.array(d["sd"])
        m.coef, m.intercept = np.array(d["coef"]), d["intercept"]
        m.mean_grade, m.n_docs = d["mean_grade"], d["n_docs"]
        return m

    def save(self, weights_dir):
        p = Path(weights_dir)
        p.mkdir(parents=True, exist_ok=True)
        (p / ARTIFACT_NAME).write_text(json.dumps(self.to_dict()))

    @classmethod
    def load(cls, weights_dir):
        return cls.from_dict(
            json.loads((Path(weights_dir) / ARTIFACT_NAME).read_text()))


# ----------------------------------------------------------------------------
# Grader interface (interface_version 4)
# ----------------------------------------------------------------------------
def _safe_score(x, fallback):
    """Python float, finite, clipped to [0, 100]."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return float(fallback)
    if not math.isfinite(x):
        return float(fallback)
    return float(min(100.0, max(0.0, x)))


class Approach3Predictor:
    def __init__(self, weights_dir, config, assets):
        self.config = config
        self.assets = assets
        here = Path(__file__).resolve().parent
        candidates = [Path(weights_dir) if weights_dir else here / "weights",
                      here / "weights"]
        self.model = None
        for d in candidates:
            if (d / ARTIFACT_NAME).exists():
                self.model = TfidfSqlModel.load(d)
                break
        if self.model is not None:
            self.fallback = _safe_score(self.model.mean_grade, 73.0)
            self.diagnostics = {"fitted": True, "n_idf_docs": self.model.n_docs,
                                "feature_set": self.model.feature_set,
                                "alpha": self.model.alpha}
        else:   # no artifacts: unweighted unigram cosine * 100
            self.fallback = 73.0
            self._plain = _Tfidf((1, 1))
            self.diagnostics = {"fitted": False}

    def _score(self, q, r):
        tq, tr = tokenize_sql(q), tokenize_sql(r)
        if not tq or not tr:
            return self.fallback            # missing/empty query
        if tq == tr:
            return 86.0   # q == r after normalisation; training mean of such rows (n=4: 80-95)
        if self.model is not None:
            return self.model.predict_pair(q, r)
        return 100.0 * self._plain.cosine(tokenize_sql(q), tokenize_sql(r))

    def predict(self, test_rows, questions=None):
        # Inference only: uses q and r. Nothing is fit on test_rows, and
        # question ids / row order / row ids are never used to infer grades.
        out = []
        for row in test_rows:
            try:
                score = _safe_score(self._score(row.get("q"), row.get("r")),
                                    self.fallback)
            except Exception:                   # never fail a row
                score = self.fallback
            out.append({"id": row["id"], "score": score})
        return out


def load_model(weights_dir, config, assets):
    # Load fitted artifacts (IDF + calibration); never train here.
    return Approach3Predictor(weights_dir, config, assets)