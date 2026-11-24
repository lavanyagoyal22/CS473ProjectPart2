"""Approach 2: text edit distance (Levenshtein) between q and r -> grade.

Loads the fitted artifact saved by train_approach2.py (weights/approach2.joblib).
No training happens here.  Pure text method: no SQL parsing, so nothing can fail to parse.
"""
from text_models import load_text_model


def load_model(weights_dir, config, assets):
    return load_text_model(weights_dir)
