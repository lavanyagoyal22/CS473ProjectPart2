"""Approach 6 (Open Approach): learned regression on edit features of (q, r), plus an optional LLM score.

Edit features (about 44, from q and r only) -> 0.7 RandomForest + 0.3 Ridge, clipped to [0, 100].
LLM score (optional): for up to a capped number of rows, one Purdue GenAI Studio call (see open_llm.py for the
model id, prompt and settings) returns a 0-100 partial-credit estimate for the (q, r) pair. That number is used
only as ONE extra input feature of a second blend trained the same way - it is never the grade itself.

Fallbacks (the prediction is always valid):
  - no LLM score for a row (API disabled, call failed, unparseable reply, budget/time cap, 5 failures in a row)
        -> the edit-feature blend alone is used for that row
  - a row whose features cannot be computed, or any model error -> the training-set mean grade
  - SQL that fails to parse keeps the text features (AST/clause features are 0, parse_fail = 1)
"""
import math
import re
import sys
import time
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from open_features import feature_matrix  # noqa: E402
import open_llm  # noqa: E402

BUDGET_FRACTION = 0.8       # use at most 80% of the API attempts we are given (retries included)
MAX_LLM_SECONDS = 1200      # stop calling the API after 20 minutes of the 38-minute overall budget
MAX_CONSECUTIVE_FAILS = 5   # API is probably down: stop calling and fall back


class OpenModel:
    def __init__(self, weights_dir, config, assets):
        art = joblib.load(Path(weights_dir) / "approach6.joblib")
        self.rf, self.ridge = art["rf"], art["ridge"]
        self.names, self.w = art["names"], art["w_rf"]
        self.shift = art.get("shift", 0.0)
        self.fallback = float(art["train_mean"])
        self.config = dict(config or {})
        self.llm = art.get("llm")
        if self.llm and self.llm.get("prompt_id") != open_llm.PROMPT_ID:
            self.llm = None  # prompt changed since training: the saved LLM model would be miscalibrated
        self.diagnostics = {"features": len(self.names), "llm_model_available": bool(self.llm)}

    def _log(self, msg):
        """One short diagnostic line on stderr (never contains the API key)."""
        try:
            msg = re.sub(r"sk-[A-Za-z0-9_-]+", "[REDACTED]", str(msg))[:300]
            sys.stderr.write("[approach6] " + msg + "\n")
            sys.stderr.flush()
        except Exception:
            pass

    # ---- LLM scores -------------------------------------------------------------------------------
    def _llm_scores(self, qs, rs, idx):
        """Return {row_index: score} for the rows that could be scored. Never raises."""
        out = {}
        try:
            if not self.llm:
                self._log("LLM skipped: no LLM model in weights or prompt id mismatch")
                return out
            if self.config.get("genai_enabled") is False:      # explicit opt-out only; a missing key means "try"
                self._log("LLM skipped: config genai_enabled is False")
                return out
            try:
                from credentials import API_KEY
            except Exception:
                API_KEY = ""
            import genai_wrapper
            client = genai_wrapper.OpenAI(api_key=API_KEY)
            budget = int(self.config.get("api_max_requests", 150))
            cap = max(0, int(BUDGET_FRACTION * budget))
            attempts, fails, start, seen = 0, 0, time.monotonic(), {}
            first_error = None
            for i in idx:
                if not (isinstance(qs[i], str) and qs[i].strip() and isinstance(rs[i], str) and rs[i].strip()):
                    continue                              # nothing to score: no call, no budget used
                key = (qs[i], rs[i])
                if key in seen:                       # identical pair already scored: no new call
                    if seen[key] is not None:
                        out[i] = seen[key]
                    continue
                if attempts >= cap or fails >= MAX_CONSECUTIVE_FAILS or time.monotonic() - start > MAX_LLM_SECONDS:
                    break
                score = None
                for _ in range(2):                    # one retry at most, and only while budget remains
                    if attempts >= cap:
                        break
                    attempts += 1
                    try:
                        resp = client.chat.completions.create(
                            model=self.llm["model_id"],
                            messages=[{"role": "user", "content": open_llm.build_prompt(qs[i], rs[i])}],
                            temperature=self.llm["temperature"], max_tokens=self.llm["max_tokens"])
                        score, _ = open_llm.parse_response(resp.choices[0].message.content)
                    except Exception as e:
                        score = None
                        if first_error is None:
                            first_error = "%s: %s" % (type(e).__name__, str(e)[:120])
                    if score is not None:
                        break
                seen[key] = score
                if score is None:
                    fails += 1
                else:
                    fails = 0
                    out[i] = float(score)
                if attempts and attempts % 20 == 0:   # trust the grader's own attempt counter if it exposes one
                    try:
                        used = int(genai_wrapper.get_stats(API_KEY).get("attempts", 0))
                        attempts = max(attempts, used)
                    except Exception:
                        pass
        except Exception as e:
            self._log("LLM path failed: %s: %s" % (type(e).__name__, str(e)[:120]))
        self.diagnostics["llm_scored_rows"] = len(out)
        self._log("LLM scored %d/%d rows; first_error=%s" % (len(out), len(idx), locals().get("first_error")))
        return out

    # ---- prediction --------------------------------------------------------------------------------
    def predict(self, test_rows, questions):
        qs = [row.get("q") for row in test_rows]
        rs = [row.get("r") for row in test_rows]
        scores = [self.fallback] * len(test_rows)
        try:
            M, names = feature_matrix(qs, rs)
            if names != self.names:
                raise ValueError("feature mismatch")
            ok = ~np.isnan(M).any(axis=1)
            ok_idx = list(np.flatnonzero(ok))
            if ok_idx:
                base = self.w * self.rf.predict(M[ok]) + (1 - self.w) * self.ridge.predict(M[ok]) + self.shift
                for i, v in zip(ok_idx, base):
                    scores[i] = float(np.clip(v, 0.0, 100.0))
                llm = self._llm_scores(qs, rs, ok_idx)
                if llm:
                    li = sorted(llm)
                    X2 = np.hstack([M[li], np.array([[llm[i]] for i in li])])
                    p = (self.w * self.llm["rf"].predict(X2) + (1 - self.w) * self.llm["ridge"].predict(X2)
                         + self.shift)
                    for i, v in zip(li, p):
                        scores[i] = float(np.clip(v, 0.0, 100.0))
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
