"""Interface-only placeholder shared by starter entrypoints; no grading algorithm."""
from pathlib import Path


class PlaceholderModel:
    def __init__(self, weights_dir, config, assets):
        self.weights_dir = Path(weights_dir)
        self.config = config
        self.assets = assets
        self.diagnostics = {'placeholder': True}

    def predict(self, test_rows, questions):
        # Replace with inference from your trained method. Never fit on test_rows.
        # Each row includes q, r, hw_num, question_number, question_prompt,
        # db_instance_path, question='hw:question', and text=q. No gold/explanation.
        # IDs are freshly randomized by the grader; do not infer grades from IDs.
        return [{'id': row['id'], 'score': 50.0} for row in test_rows]
