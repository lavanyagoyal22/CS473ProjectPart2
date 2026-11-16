#!/usr/bin/env python3
"""Small local interface checker. It never contains course evaluation data."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path
import sys
import traceback

APPROACHES = (
    "approach1_option1", "approach1_option2", "approach1_option3",
    "approach2", "approach3", "approach4", "approach5", "approach6",
)
REQUIRED_APPROACHES = ('approach1_option1', 'approach1_option2', 'approach6')
EXECUTION_ORDER = tuple(name for name in APPROACHES if name != 'approach1_option3') + ('approach1_option3',)
BASELINE_APPROACHES = ('approach2', 'approach3', 'approach4', 'approach5')
MIN_BASELINES = 2


def submission_requirements(successful_approaches):
    successful = set(successful_approaches)
    missing = [ident for ident in REQUIRED_APPROACHES if ident not in successful]
    baselines = [ident for ident in BASELINE_APPROACHES if ident in successful]
    return {
        'valid_submission': int(not missing and len(baselines) >= MIN_BASELINES),
        'missing_required_approaches': missing,
        'successful_baselines': baselines,
        'additional_baselines_needed': max(0, MIN_BASELINES - len(baselines)),
    }


def check_structure(root):
    files = [p for p in root.rglob("*") if p.is_file()]
    if len(files) > 256:
        raise ValueError("Submission exceeds 256 files.")
    if sum(p.stat().st_size for p in files) > 100_000_000:
        raise ValueError("Submission exceeds 100 MB uncompressed.")
    present = [name + ".py" for name in APPROACHES if (root / (name + ".py")).is_file()]
    if not present:
        raise ValueError("Include at least one approach file at the ZIP root.")
    print("Structure OK. Grader entrypoints: " + ", ".join(present) + ". Extra helper/artifact files are allowed.")


def load_csv(path):
    """Load Part 2 inputs only, excluding grades and manual explanations."""
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        first_line = f.readline()
        f.seek(0)
        delimiter = "\t" if Path(path).suffix.lower() == ".tsv" or "\t" in first_line else ","
        reader = csv.DictReader(f, delimiter=delimiter)
        norm = lambda name: "".join(c for c in name.lower() if c.isalnum())
        headers = {norm(name): name for name in (reader.fieldnames or [])}
        aliases = {
            "id": ("id", "qpair_id", "submissionid"),
            "hw_num": ("hw_num",),
            "question_number": ("question_number",),
            "question_prompt": ("question_prompt", "question_text"),
            "q": ("q", "sq", "query"),
            "r": ("r", "corrected_sql"),
            "db_instance_path": ("db_instance_path",),
        }
        mapping = {}
        for key, names in aliases.items():
            mapping[key] = next((headers[norm(n)] for n in names if norm(n) in headers), None)
            if mapping[key] is None:
                raise ValueError("Dataset is missing required column: " + key)
        rows = []
        for item in reader:
            row = {key: str(item.get(column, "") or "") for key, column in mapping.items()}
            for key in ("id", "hw_num", "question_number"):
                row[key] = row[key].strip()
                if not row[key] or (key != "id" and ":" in row[key]):
                    raise ValueError("Invalid or empty " + key)
            row["question"] = row["hw_num"] + ":" + row["question_number"]
            row["text"] = row["q"]
            rows.append(row)
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("CSV is empty or has duplicate IDs.")
    return rows


def build_questions(rows):
    """Generate metadata using the same public fields as the autograder."""
    questions = {}
    for row in rows:
        questions.setdefault(row["question"], {
            "hw_num": row["hw_num"],
            "question_number": row["question_number"],
            "context": row["question_prompt"],
            "allowed_grades": [0, 100],
        })
    return questions


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def validate(predictions, rows, questions):
    if not isinstance(predictions, list) or len(predictions) != len(rows):
        raise ValueError("predict() must return one item per row.")
    expected = {r["id"]: r for r in rows}
    if len(expected) != len(rows):
        raise ValueError("Input data contains duplicate IDs.")
    seen = set()
    for item in predictions:
        if not isinstance(item, dict) or set(item) != {"id", "score"}:
            raise ValueError("Each prediction must contain only id and score.")
        ident, score = item["id"], item["score"]
        if not isinstance(ident, str) or ident not in expected or ident in seen:
            raise ValueError("Unknown or duplicate prediction id.")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError("score must be an ordinary Python int or float.")
        if not math.isfinite(score):
            raise ValueError("score must be finite; NaN and infinity are not allowed.")
        allowed = questions[expected[ident]["question"]]["allowed_grades"]
        if not allowed[0] <= score <= allowed[-1]:
            raise ValueError(f"Score {score} is outside the permitted range for question {expected[ident]['question']}.")
        seen.add(ident)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission-dir", type=Path, default=Path("."))
    ap.add_argument("--structure-only", action="store_true")
    ap.add_argument("--csv", help="your local training/validation subset; labels are removed before predict()")
    ap.add_argument("--assets", type=Path, help="optional local_assets/assets.json")
    ap.add_argument("--allow-api", action="store_true", help="allow Open Approach (approach6) to use your local GenAI key")
    args = ap.parse_args()

    root = args.submission_dir.resolve()
    check_structure(root)
    if args.structure_only:
        present = [name for name in APPROACHES if (root / (name + '.py')).is_file()]
        print('Required file combination complete: ' +
              str(submission_requirements(present)['valid_submission']))
        print('File presence alone does not verify valid predictions or algorithms.')
        return 0
    if not args.csv:
        ap.error("--csv is required unless --structure-only is used")

    rows = load_csv(args.csv)
    questions = build_questions(rows)
    assets = json.loads(args.assets.read_text(encoding="utf-8-sig")) if args.assets else {"models": {}, "precomputed": {}}
    sys.path.insert(0, str(root))

    print("\nExact input row shape passed to predict():")
    print(json.dumps(rows[0], indent=2, ensure_ascii=False))
    print("(Grades and manual_explanation are not passed to predict.)\n")

    successful = []
    for name in EXECUTION_ORDER:
        path = root / f"{name}.py"
        if not path.is_file():
            continue
        print(f"Testing {name} ...")
        try:
            module = load_module(name, path)
            model = module.load_model(root / "weights", {
                "threads": 4,
                "interface_version": 4,
                "genai_enabled": bool(args.allow_api and name == "approach6"),
                "api_max_requests": 150,
            }, assets)
            predictions = model.predict(rows, questions)
            validate(predictions, rows, questions)
            successful.append(name)
            print(f"  OK: {len(predictions)} predictions")
            print("  Example returned item:", predictions[0])
        except Exception:
            print(f"  FAILED: {name}")
            traceback.print_exc()
    requirements = submission_requirements(successful)
    print('\nValid submission: ' + str(requirements['valid_submission']))
    if requirements['missing_required_approaches']:
        print('Required approaches not successful: ' + ', '.join(requirements['missing_required_approaches']))
    if requirements['additional_baselines_needed']:
        print('Additional successful baselines needed: ' + str(requirements['additional_baselines_needed']))
    print('This checks interfaces and method combinations, not algorithm correctness or grader isolation.')
    return 0 if requirements['valid_submission'] else 1


if __name__ == "__main__":
    raise SystemExit(main())
