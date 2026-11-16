# Project Part 2: Student Submission Instructions

Use these instructions together with the **CS473 Project Part 2: SQL Partial Credit Prediction** handout. Train and tune locally using `training_data_Part2.csv`, then submit your inference code and saved artifacts. The autograder loads your models and predicts scores on a held-out evaluation set; it does **not** train or tune your models.

The handout lists the code/leaderboard deadline as **December 7, 2026**, and the report and presentation deadline as **December 9, 2026**. Follow the assignment page and course announcements for submission times and updates.

## 1. Before you submit

Use your group's assigned anonymous leaderboard name exactly, and add all group members to each Gradescope submission. Do not use real names or change submitters to bypass the daily limit.

The project requires four approaches: AST comparison, two selected baselines, and an Open Approach. Both required AST options need separate entrypoint files, so a minimally complete submission has **five method files**.

| File | Method | Requirement |
|---|---|---|
| `approach1_option1.py` | Option A: library-provided AST edit distance | Required |
| `approach1_option2.py` | Option B: IDF-weighted AST edits | Required |
| `approach1_option3.py` | Option C: rubric/custom-weighted AST edits | Optional |
| `approach2.py` | Text edit distance | Choose two of Approaches 2-5 |
| `approach3.py` | SQL TF-IDF vector similarity | Choose two of Approaches 2-5 |
| `approach4.py` | Dense embedding similarity | Choose two of Approaches 2-5 |
| `approach5.py` | Graph2Vec of SQL ASTs | Choose two of Approaches 2-5 |
| `approach6.py` | Additional meaningful Open Approach | Required |

Use these exact filenames. For example, Option C is named `approach1_option3.py`, not `approach1_optionc.py`. Remove unselected placeholder methods from your final ZIP. Implementing additional methods does not replace any required method.

Use the same validation folds across methods and settings; five-fold cross-validation is one suitable procedure. Select configurations using validation RMSE, not leaderboard feedback. Fit IDF, vocabulary, distance-to-grade mappings, regressors and graph embeddings only on training folds. Report RMSE, MAE, Pearson correlation and underprediction/overprediction counts, and complete the research questions, explainability analysis and other deliverables in the handout.

## 2. Starter files and submission layout

Start with the supplied `demo_submission` package. All eight starter entrypoints are clearly marked **interface placeholders returning 50.0**. They are not implementations of the required methods. Run the demo once, then replace the selected placeholders with your own methods.

Example final ZIP layout, choosing Approaches 2 and 3:

```text
submission.zip
+-- approach1_option1.py
+-- approach1_option2.py
+-- approach2.py
+-- approach3.py
+-- approach6.py
+-- weights/
|   +-- your_saved_artifacts.joblib
+-- your_helpers.py
+-- embedding_assets.py       # if used by your model
+-- genai_wrapper.py          # optional; replaced by the official grader wrapper
+-- credentials.py            # only needed for optional GenAI use
```

- Put entrypoint files at the ZIP root, without an enclosing `demo_submission/` folder.
- Include every helper module and saved artifact that your code imports or reads.
- Keep `demo_model.py` only while a placeholder still imports it; replace placeholders before final submission.
- Do not include virtual environments, downloaded staff encoders, private evaluation data, evaluation caches, `__pycache__`, or `.part2-validation/`.
- Your submission's `requirements.txt` is **not** automatically installed. Use the supported packages or arrange any additional dependency with course staff before submission.
- Do not assume the training CSV is present during grading. Save everything required for inference.

The grader directly loads only registered approach entrypoints. Other files are used only if your methods import or read them.

## 3. Training data and evaluation input

The training CSV has nine columns:

| Column | Meaning | Available during prediction? |
|---|---|---|
| `id` | Record identifier | Yes, but replaced by a fresh random ID |
| `hw_num` | Homework identifier | Yes |
| `question_number` | Question number within the homework | Yes |
| `question_prompt` | Question text | Yes |
| `q` | Original student SQL query | Yes |
| `r` | Repaired SQL query | Yes |
| `manual_grade` | Ground-truth score, 0-100 | **No** |
| `manual_explanation` | Training-only explanation of the manual grade | **No** |
| `db_instance_path` | Associated database path metadata | Yes, as metadata |

Use `manual_grade` as your target. Training explanations are not necessarily a complete reusable rubric and cannot be required at prediction time.

Unlike Part 1, the question identifier is the pair **`(hw_num, question_number)`**. Question 3 in homework 1 and question 3 in homework 4 are different questions. The grader creates the compatibility field `question` as `"hw_num:question_number"`, and `text` as an alias of `q`:

```python
test_rows = [{
    "id": "r_random_identifier",
    "hw_num": "4",
    "question_number": "3",
    "question_prompt": "Find the team with the highest average salary ...",
    "q": "SELECT ...",       # original query
    "r": "SELECT ...",       # repaired query
    "db_instance_path": "./DB/spy.db",
    "question": "4:3",
    "text": "SELECT ...",    # same original query as q
}]
```

IDs, homework numbers and question numbers are strings in this interface. Evaluation IDs and row order change between runs. Do not look up scores by original IDs, infer scores from row positions or assume every evaluation question has same-question training examples.

`db_instance_path` does not guarantee a readable database file in the worker. The default interface supplies SQL pairs, not a running database service. Do not depend on those paths without an explicitly supported deployment arrangement. The handout also states that result-set differences should not be the main basis of partial-credit prediction.

No `questions.json` file is required. The grader generates metadata from the input CSV:

```python
questions["4:3"] = {
    "hw_num": "4",
    "question_number": "3",
    "context": "Find the team with the highest average salary ...",
    "allowed_grades": [0, 100],
}
```

Here `[0, 100]` gives the **range endpoints**, not two allowed categories. Any finite numeric score within the range is valid.

## 4. Required Python interface

Each entrypoint exports:

```python
def load_model(weights_dir, config, assets):
    # Load fitted artifacts; do not train here.
    return model
```

The returned object implements:

```python
def predict(self, test_rows, questions):
    return [
        {"id": row["id"], "score": 85.5}
        for row in test_rows
    ]
```

The example score is illustrative, not a grading algorithm. Return an ordinary Python list in memory; do not write a prediction CSV.

- Preserve every supplied ID exactly once. Return only `id` and `score` in each prediction dictionary.
- Scores must be ordinary Python `int` or `float` values in **[0, 100]**. Convert NumPy scalar values to Python numbers.
- Missing, duplicate or unknown IDs, boolean/string scores, NaN, infinity or out-of-range values invalidate the method.
- Floating scores are accepted. RMSE and MAE use raw predictions without rounding or clipping by the grader.
- Return a prediction even when parsing a SQL query fails; document your fallback strategy.
- `config` includes `interface_version=4`, `threads=4`, `genai_enabled` and `api_max_requests`.

The grader calls `load_model()` once and `predict()` once per executed approach. Both phases count toward runtime.

## 5. Save everything needed for inference

Depending on your method, save fitted tokenization/preprocessing settings, TF-IDF vocabulary and weights, AST edit weights, normalization parameters, score calibration/regression models, graph labels and Graph2Vec/Doc2Vec artifacts.

```python
# After training locally:
from pathlib import Path
import joblib

Path("weights").mkdir(exist_ok=True)
joblib.dump(fitted_pipeline, "weights/approach3.joblib", compress=3)

# Inside load_model():
saved_model = joblib.load(Path(weights_dir) / "approach3.joblib")
```

Use `weights_dir` or `Path(__file__).resolve().parent` to locate files. Your local absolute paths will not exist on Gradescope. Submitted files are read-only, and the worker's working directory is a separate writable output directory. Do not create caches beside submitted code.

For custom PyTorch models, include architecture code and saved parameters, load on CPU, call `eval()`, and use `torch.inference_mode()`. Where appropriate, load a state dictionary with `weights_only=True`. Define serialized custom classes in importable modules included in the ZIP. Keep serialization library versions consistent with the grader.

## 6. Local environment and testing

The following commands assume the distributed folders are extracted side by side:

```bash
python local_environment/create_environment.py
# Optional: download and offline-validate the five staff encoders.
python local_environment/create_environment.py --download-encoders
```

The default virtual environment is `.part2-venv`. Activate it before running models:

```powershell
# Windows PowerShell
.\.part2-venv\Scripts\Activate.ps1
```

```bash
# Linux/macOS
source .part2-venv/bin/activate
```

Test the starter interface:

```bash
python demo_submission/local_test.py --submission-dir demo_submission --csv demo_submission/demo_training_sample.csv
```

The sample has eight public training records, two per homework. It is for interface checks, not model training or performance measurement. Local packages contain no private test labels or evaluation caches.

Test your final model on your own held-out validation CSV:

```bash
python local_environment/local_test.py --submission-dir demo_submission --csv your_validation.csv --assets local_assets/assets.json
```

Omit `--assets` if you do not use staff encoders. Add `--allow-api` only when deliberately testing the Open Approach with your own key. You do **not** need `--questions`.

The checker accepts the Part 2 columns and strips grades and explanations before prediction. It continues after an individual method fails, then prints `Valid submission`. Exit code 0 means the required method combination produced valid predictions; 1 means it did not. `--structure-only` checks files without executing models. Local success does not prove correct algorithms, good prediction performance or compliance with the grader's Linux isolation and time limits.

## 7. Supported environment

Use Python 3.10-3.12; the environment has been checked with Python 3.12 on Windows and Ubuntu 24.04. The grader runs on CPU only.

| Library | Version |
|---|---|
| NumPy / SciPy | 1.26.4 / 1.12.0 |
| pandas / scikit-learn | 2.2.3 / 1.6.1 |
| joblib / psutil | 1.5.3 / 7.0.0 |
| SQLGlot / RapidFuzz | 26.16.4 / 3.13.0 |
| APTED / zss / six | 1.0.3 / 1.2.0 / 1.17.0 |
| NetworkX / gensim / smart-open | 3.4.2 / 4.4.0 / 7.3.0.post1 |
| Karate Club | 1.3.4, pinned official source commit |
| decorator / python-louvain / PyGSP | 5.1.1 / 0.16 / 0.6.1 |
| python-Levenshtein / Levenshtein | 0.27.1 / 0.27.1 |
| PyTorch | 2.8.0 CPU |
| sentence-transformers / transformers | 5.1.1 / 4.55.4 |
| huggingface-hub / ONNX Runtime | 0.34.4 / 1.22.1 CPU |
| tokenizers / safetensors | 0.21.4 / 0.6.2 |
| SentencePiece / protobuf / Pillow | 0.2.1 / 6.32.0 / 11.3.0 |

The supplied requirements files list supported direct dependencies. NLTK and its corpora are not installed. Karate Club 1.3.4 is installed from the official upstream commit `cb46a91df8dcbeb2570debcf6a9d0c518107a2de`. NumPy and SciPy are selected for its compatibility. Use `from karateclub import Graph2Vec`; this pinned version uses `use_node_attribute="feature"` for labeled graph nodes. A documented alternative implementation is also permitted. Staff do not supply a trained grade predictor or pretrained Graph2Vec model.

During grading, general outbound networking and new subprocess creation are blocked; threads are allowed. Do not call `pip`, download models/corpora, launch external programs, or rely on unrestricted HTTP. Staff model files are installed during setup.

## 8. Staff-supplied encoders and q/r caches

| Asset key | Model | Dimensions | Maximum tokens | Evaluation cache |
|---|---|---:|---:|---|
| `minilm` | `sentence-transformers/all-MiniLM-L6-v2` | 384 | 512 | q and r |
| `multiqa_minilm` | `sentence-transformers/multi-qa-MiniLM-L6-cos-v1` | 384 | 512 | None |
| `e5_small` | `intfloat/e5-small-v2` | 384 | 512 | None |
| `bge_small` | `BAAI/bge-small-en-v1.5` | 384 | 512 | None |
| `jina_code` | `jinaai/jina-embeddings-v2-base-code` | 768 | 2048 | q and r |

Exact full model revisions are pinned in `local_environment/assets_config.json` and available in `assets["models"][name]`. The four general encoders use CPU Sentence Transformers; Jina uses the published quantized CPU ONNX model. Tokenizers and pooling configuration are included. Use the supplied helper for consistent loading:

The general encoders use batch size 16; Jina uses batch size 2. The token limits include special tokens. A 512-token limit still truncates longer SQL; increasing it beyond 512 is not supported by these general encoders. Jina's 2048 limit covers the current public training queries. If you implement chunking or another preprocessing strategy, use it consistently and generate your own compatible vectors rather than reusing the staff cache.

**Updating an older environment:** re-run `create_environment.py --download-encoders` from the same parent folder to refresh `local_assets/assets.json`. The current settings are 512 tokens for the general encoders and 2048 for Jina. Existing model weights may be reused, but old manifests/caches describe different preprocessing and must not be mixed with newly encoded training vectors.

```python
from embedding_assets import load_encoder

spec = assets["models"]["jina_code"]
encoder = load_encoder(spec, threads=config.get("threads", 4))
q_vectors = encoder.encode(
    [row["q"] for row in test_rows], batch_size=spec["batch_size"],
    normalize_embeddings=True, convert_to_numpy=True,
)
r_vectors = encoder.encode(
    [row["r"] for row in test_rows], batch_size=spec["batch_size"],
    normalize_embeddings=True, convert_to_numpy=True,
)
```

Include `embedding_assets.py` in your submission if you import it. Use the same encoder and preprocessing for training and prediction. Follow E5's prefix convention consistently when using that model. No GPU or remote model code is needed by the provided loaders.

For MiniLM and Jina, frozen evaluation vectors can avoid repeated encoding:

```python
import numpy as np

cache_path = assets.get("precomputed", {}).get("minilm")
if cache_path:
    with np.load(cache_path, allow_pickle=False) as cache:
        lookup = {str(value): i for i, value in enumerate(cache["ids"])}
        indices = [lookup[row["id"]] for row in test_rows]
        q_vectors = cache["q_vectors"][indices]
        r_vectors = cache["r_vectors"][indices]
    cosine_similarity = (q_vectors * r_vectors).sum(axis=1)
else:
    # Encode q and r locally using the selected staff model.
    ...
```

Cosine similarity alone is not necessarily a calibrated grade: learn your mapping using training data. Cache arrays include `ids`, `q_vectors`, `r_vectors`, `q_hashes`, `r_hashes`, `encoder_config` and `cache_version=2`. Vectors are L2-normalized. IDs are remapped for each grading run, so always align by the supplied IDs.

Caches encode **only q and r**, never manual grades, explanations or question prompts. They apply only to the recorded frozen model, pooling, normalization and truncation settings. If you change these settings or fine-tune the encoder, compute compatible vectors yourself. Do not mix cached evaluation vectors with training vectors from a different configuration. The local environment downloads public encoders only; private evaluation caches are not distributed.

## 9. Optional Purdue GenAI Studio use

The handout permits Purdue GenAI Studio as the only LLM API and discourages relying solely on opaque LLM grading. Justify robustness and transparency if you use an LLM. Under this autograder interface, API access is enabled **only for `approach6`**.

Put your key in `credentials.py`, following `credentials_example.py`. Use the supplied text-only wrapper:

```python
from credentials import API_KEY
from genai_wrapper import OpenAI

client = OpenAI(api_key=API_KEY)
response = client.chat.completions.create(
    model="gpt-oss:120b",
    messages=[{"role": "user", "content": "Your SQL-pair grading prompt"}],
    temperature=0,
    max_tokens=1024,
)
text = response.choices[0].message.content
```

The grader replaces submitted wrappers with its trusted wrapper. Direct OpenAI clients and HTTP requests are blocked. Parse the response and return valid numeric predictions; a model response is not itself a prediction dictionary.

The wrapper supports ordinary text chat, not tools, function calls, attachments, alternate endpoints or custom model IDs. The allowlist is available as `genai_wrapper.ALLOWED_MODELS`; account availability may be a subset. Examples include `gpt-oss:120b`, `qwen3.8:27b`, `codellama:latest` and `qwen3-coder:latest`.

Requests are paced at 20 per minute. The grading budget is 150 HTTP attempts, including retries and model-list requests. Batch SQL pairs where appropriate, and leave time for parsing and validation. Never print your key or include it in reports. Document your model ID, prompt and relevant settings for reproducibility. Running the separate wrapper example makes a real request; the interface demo does not.

## 10. Leaderboard, runtime and submission limits

The leaderboard contains **10 columns**: Best RMSE, each of the eight methods' RMSE, and `Valid submission`. RMSE columns sort ascending; lower is better. `Valid submission` sorts descending.

- RMSE and MAE are computed across all evaluation records with equal weight per record, not by averaging question-level errors.
- MAE and Pearson correlation appear in `results.json` feedback/details, not as leaderboard columns. Undefined Pearson is reported as undefined/null.
- Best RMSE is the lowest valid RMSE among methods actually evaluated, even when the submission is incomplete.
- A missing, failed or unexecuted method has leaderboard RMSE **101**, an unavailable-result sentinel beyond the maximum valid RMSE of 100. With no successful method, Best RMSE is also 101.
- `Valid submission=1` requires successful AST Options A and B, successful Approach 6, and at least two successful baselines from Approaches 2-5. Otherwise it is 0. Additional optional-method failures do not invalidate an otherwise complete combination.
- This binary check verifies interfaces and method combinations, not whether your algorithms meet the handout. Placeholder code is not a completed project.

| Limit | Current rule |
|---|---|
| Daily admitted submissions | 6 per day under the current Part 2 configuration; no overall submission cap |
| Daily reset | Midnight in `America/New_York` |
| Uncompressed submission size | At most 100,000,000 bytes |
| File count | At most 256 files |
| CPU | Up to four CPU cores/threads configured per method; no GPU |
| Memory | 4,800 MiB process-tree limit |
| Per-method CPU execution | Up to 600 seconds, subject to remaining overall time |
| Overall grading budget | 2,280 seconds (38 minutes), including initialization and executed methods |
| Approach 6 with API enabled | May use remaining overall time, up to the configured overall budget |
| API budget | 150 HTTP attempts, including retries and model-list requests |
| Captured output / worker result | 4 MiB each |

Methods execute **sequentially**, not concurrently: AST Option A, AST Option B, Approaches 2, 3, 4, 5, 6, then optional **AST Option C last**. Missing methods are skipped. Native libraries may use multiple threads within one method; pure Python code does not automatically become parallel. New subprocesses are prohibited.

Loading time counts. Earlier methods can consume the overall budget and prevent later methods from running. Timeouts and methods not started due to the overall cutoff receive unavailable RMSE 101. Each method's runtime is reported.

Use the same designated submitting members and add the full group. Do not launch overlapping submissions or switch submitters to evade quotas. Admitted submissions count even if code crashes or times out. A recognized GenAI service outage is quota-exempt; genuine Gradescope platform errors are excluded from admission history. Do not assume every API error qualifies as a service outage. Uploads rejected by the quota may display retained earlier feedback, which does not evaluate the newly uploaded files.

## 11. Before uploading

1. Create and activate the supplied environment, then run the placeholder demo once.
2. Replace the selected placeholders with your actual methods and fit/tune only on training data.
3. Save fitted artifacts under `weights/` and include all required helper modules. Remove unselected methods and unnecessary files.
4. Test using your own held-out validation CSV. Add local encoder assets or API access only when needed. Inspect `Valid submission` and each method's status.
5. Test CPU-only behavior, parse failures, unseen question pairs, artifact loading and runtime. Do not rely on original IDs, test order, gold explanations or accessible database paths.
6. Create a ZIP with method files at its root. Exclude environments, staff model downloads, private data/caches, test outputs and unrelated files.
7. Upload to the **Part 2** Gradescope assignment, enter the assigned anonymous leaderboard name and add all group members.
8. Inspect both performance and completeness: lower Best RMSE is better, but your final method combination must also satisfy `Valid submission=1` and the actual algorithm requirements.
9. Submit the report, presentation, slides and reviews separately as directed in the handout and course announcements.

AI tools may be used as permitted by the handout. Your team remains responsible for understanding and explaining its work. For questions or deployment issues, make a private post on Ed or contact course staff through the announced channel.