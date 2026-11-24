"""Approach 1 Option A: SQLGlot AST edit distance (unit cost per edited node).

Loads the fitted artifact saved by train_approach1.py (weights/approach1_option1.joblib).
No training happens here.  Unparsable SQL -> token-distance fallback (see ast_utils.py).
"""
from ast_models import load_ast_model


def load_model(weights_dir, config, assets):
    return load_ast_model("option1", weights_dir)
