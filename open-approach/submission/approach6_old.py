"""Approach 6 (Open Approach): learned regression on AST + token + clause features of (q, r).

Model: 0.7 * RandomForest + 0.3 * Ridge on ~44 features, clipped to [0, 100].
Inputs used: only the q and r columns. No query execution, no LLM, no question ids.
Fallbacks (documented): if a row's features cannot be computed, or the model call fails, the
row gets the training-set mean grade. SQL parse failures keep the text-only features
(AST/clause features are 0 and `parse_fail` = 1).
"""
import math
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from open_features import feature_matrix  # noqa: E402


class OpenModel:
    def __init__(self, weights_dir, config, assets):
        art = joblib.load(Path(weights_dir) / "approach6.joblib")
        self.rf, self.ridge = art["rf"], art["ridge"]
        self.names, self.w = art["names"], art["w_rf"]
        self.shift = art.get("shift", 0.0)
        self.fallback = float(art["train_mean"])
        self.diagnostics = {"features": len(self.names), "w_rf": self.w, "shift": self.shift}

    def predict(self, test_rows, questions):
        qs = [row.get("q") for row in test_rows]
        rs = [row.get("r") for row in test_rows]
        scores = [self.fallback] * len(test_rows)
        try:
            M, names = feature_matrix(qs, rs)
            if names != self.names:
                raise ValueError("feature mismatch")
            ok = ~np.isnan(M).any(axis=1)
            if ok.any():
                p = self.w * self.rf.predict(M[ok]) + (1 - self.w) * self.ridge.predict(M[ok]) + self.shift
                p = np.clip(p, 0.0, 100.0)
                for i, v in zip(np.flatnonzero(ok), p):
                    scores[i] = float(v)
        except Exception:
            pass
        out = []
        for row, s in zip(test_rows, scores):
            s = float(s)
            if not math.isfinite(s):
                s = self.fallback
            out.append({"id": row["id"], "score": min(100.0, max(0.0, s))})
        return out


def load_model(weights_dir, config, assets):
    return OpenModel(weights_dir, config, assets)
