# Part 2: Set Up Your Environment and Test Your Submission

Follow this guide to install the supported Python environment, run the starter
submission, and check your own prediction code before uploading to Gradescope.
Start with the unchanged demo; you do not need to train a model, download encoder
weights, or supply an API key to run it.

## 1. Extract the two packages

Extract `local_environment.zip` and `demo_submission.zip` into the same parent
folder. If you received folders instead of ZIPs, place them side by side:

```text
part2-work/
+-- local_environment/
|   +-- create_environment.py
|   +-- requirements.txt
|   +-- requirements-embeddings.txt
|   +-- assets_config.json
|   +-- embedding_assets.py
|   +-- validate_environment.py
|   +-- local_test.py
|   +-- ...
+-- demo_submission/
    +-- approach1_option1.py
    +-- approach1_option2.py
    +-- approach1_option3.py
    +-- approach2.py ... approach6.py
    +-- demo_model.py
    +-- demo_training_sample.csv
    +-- local_test.py
    +-- ...
```

Open a terminal in **`part2-work/`**, the parent of both folders. Run the commands
below from this parent folder unless a command explicitly changes directories.
Keep the files inside `local_environment/` together; the setup script reads its
requirements and configuration from that folder.

## 2. Install Python and create the environment

Use **Python 3.12** if possible. Python 3.10-3.12 is supported. You need an
internet connection for initial package installation. CPU PyTorch and the other
libraries take time to download and require several GB of free disk space. No GPU
or CUDA installation is needed.

### Windows PowerShell

Install Python 3.12 if it is not already available, then run:

```powershell
py -3.12 --version
py -3.12 local_environment/create_environment.py
```

### Linux or macOS

Check that `python3` is version 3.10-3.12, then run:

```bash
python3 --version
python3 local_environment/create_environment.py
```

On Ubuntu, if environment creation reports that `ensurepip` or `venv` is missing,
install the venv package for your Python version (for example,
`sudo apt install python3-venv`), then retry.

The script creates **`.part2-venv/` in your current directory**. It installs the
supported numeric, SQL, tree-edit, graph and embedding libraries, runs `pip check`,
and exercises the installed libraries. Successful setup prints a validation JSON
object with `"status": "passed"`, followed by `Environment ready:` and the path to
the environment's Python executable. Wait for the command to finish successfully
before continuing.

The default setup installs model-running libraries but **does not download the
five encoders' weights**. Those weights are optional for the first demo test.
Use the setup script from this package with its matching requirements/configuration
files; do not combine files from older Part 1 or earlier Part 2 packages.

## 3. Activate the environment

### Windows PowerShell

```powershell
.\.part2-venv\Scripts\Activate.ps1
python --version
```

### Linux or macOS

```bash
source .part2-venv/bin/activate
python --version
```

Run subsequent `python` commands in this activated terminal. When opening a new
terminal, activate the environment again.

If PowerShell blocks activation, you can use the environment's Python directly
without changing execution policy. For example, replace `python` in the next
command with `.\.part2-venv\Scripts\python.exe`. On Linux/macOS, the equivalent
executable is `./.part2-venv/bin/python`.

## 4. Run the unchanged demo first

From the parent folder, run:

```bash
python local_environment/local_test.py --submission-dir demo_submission --csv demo_submission/demo_training_sample.csv
```

This command should work with the distributed files exactly as supplied. It does
not require encoder downloads, saved weights, a database, `questions.json`, or an
API key. Do not add `--assets` or `--allow-api` for this first test.

The checker prints the input format, then tests all eight demo entrypoints.
You should see output like:

```text
Testing approach1_option1 ...
  OK: 8 predictions
  Example returned item: {'id': '...', 'score': 50.0}
...
Testing approach6 ...
  OK: 8 predictions
Testing approach1_option3 ...
  OK: 8 predictions

Valid submission: 1
```

Expect **eight** `OK: 8 predictions` messages, no `FAILED:` messages, and a process
exit code of **0**. Option C (`approach1_option3`) runs last.

You can also run the checker included in the demo; it behaves the same way:

```bash
python demo_submission/local_test.py --submission-dir demo_submission --csv demo_submission/demo_training_sample.csv
```

**The demo is an interface example, not a completed project.** Every demo method
returns 50.0. Its successful result confirms that the package can run and return
valid predictions, not that it implements the required algorithms or predicts
accurate grades. Keep `demo_model.py` while the supplied placeholders import it.

The sample contains eight public training records, two per homework. It is too
small for meaningful model training or performance evaluation.

## 5. Replace the placeholders and test your own code

Implement the selected methods and save any trained artifacts under your
submission's `weights/` directory. Include helper modules that your methods import.
The grader loads saved models; it does not invoke training.

Your final methods must include:

- `approach1_option1.py`: library-provided AST edit distance.
- `approach1_option2.py`: IDF-weighted AST edits.
- `approach6.py`: your Open Approach.
- At least two of `approach2.py` (text edit distance), `approach3.py` (TF-IDF),
  `approach4.py` (dense embeddings), and `approach5.py` (Graph2Vec).

`approach1_option3.py` is optional. Remove the unselected placeholder files from
your final submission. Each retained method must be your real implementation.

Create a held-out validation CSV from the full training dataset, preserving the
Part 2 column names. Assuming your folder is named `my_submission/` and the CSV is
`my_validation.csv`, run:

```bash
python local_environment/local_test.py --submission-dir my_submission --csv my_validation.csv
```

Use the full project handout for cross-validation, algorithm and report
requirements. Train and tune on training folds only.

### What the local checker verifies

Each method must export `load_model(weights_dir, config, assets)` and return an
object with `predict(test_rows, questions)`. Prediction must return an ordinary
Python list containing exactly one dictionary per input record:

```python
[
    {'id': 'the_supplied_id', 'score': 85.5},
    # ...one item for every supplied record
]
```

Each dictionary contains only `id` and `score`. Preserve every input ID exactly
once. Scores must be ordinary Python integers or floats in **[0, 100]**; booleans,
strings, NumPy scalars, NaN and infinity are invalid. Convert numeric library
scalars with `float(...)` before returning them.

The input CSV must include these seven input columns:

```text
id,hw_num,question_number,question_prompt,q,r,db_instance_path
```

A labeled validation CSV can also contain `manual_grade` and
`manual_explanation`. The checker removes both before calling `predict()`.
It generates metadata automatically: `question` is `hw_num:question_number`, and
`text` is an alias of `q`. There is no `--questions` argument.

Database paths are metadata; they do not guarantee access to a database during
grading. Do not require local database files merely to run prediction.

### How to interpret the result

- `OK: N predictions` means that method returned valid predictions for all N rows.
- `FAILED:` includes a traceback. The checker continues with the remaining methods.
- `Valid submission: 1` means the three required entrypoints and at least two
  baselines succeeded. Its exit code is 0.
- `Valid submission: 0` means the required combination did not succeed. Its exit
  code is 1, and the output lists missing required methods or additional baselines.
- A failed extra optional method does not change an otherwise valid combination,
  so inspect individual statuses as well as the final binary result.

To check file layout without loading models:

```bash
python local_environment/local_test.py --submission-dir my_submission --structure-only
```

This reports file presence only. It does not verify valid predictions or method
implementations, and its exit code does not certify a complete submission.

The checker **does not compute RMSE, MAE or Pearson**, train your model, verify
algorithm correctness, or reproduce Gradescope's Linux isolation, memory and time
limits. Evaluate accuracy separately using your held-out labels. The leaderboard
shows RMSE and submission completeness; MAE and Pearson remain in result feedback.

## 6. Optional: download the staff embedding encoders

Skip this section if your methods do not use the supplied embedding models. From
the parent folder, with the environment activated, run:

```bash
python local_environment/create_environment.py --download-encoders
```

The script reuses `.part2-venv`, checks its packages, downloads the five pinned
encoders and their tokenizers into `local_assets/`, and validates local CPU
encoding. This requires internet access and additional disk space. It does not
run your grading methods or make GenAI API calls.

| Asset key | Encoder | Dimensions | Configured maximum tokens |
|---|---|---:|---:|
| `minilm` | `sentence-transformers/all-MiniLM-L6-v2` | 384 | 512 |
| `multiqa_minilm` | `sentence-transformers/multi-qa-MiniLM-L6-cos-v1` | 384 | 512 |
| `e5_small` | `intfloat/e5-small-v2` | 384 | 512 |
| `bge_small` | `BAAI/bge-small-en-v1.5` | 384 | 512 |
| `jina_code` | `jinaai/jina-embeddings-v2-base-code` | 768 | 2048 |

Once downloading finishes, pass the generated manifest to your local test:

```bash
python local_environment/local_test.py --submission-dir my_submission --csv my_validation.csv --assets local_assets/assets.json
```

Your model can access specifications through `assets['models'][asset_name]`.
Use the supplied `embedding_assets.load_encoder()` helper, including a copy of
`embedding_assets.py` in your submission if you import it. Jina uses CPU quantized
ONNX with masked mean pooling and L2 normalization; the other models use CPU
Sentence Transformers. Keep model, pooling, tokenization and truncation settings
consistent between training and prediction.
The limits above include special tokens. General encoders use batch size 16 and
may still truncate SQL longer than 512 tokens. Jina uses batch size 2 and a
2048-token limit. If you previously downloaded encoders with the old 128/512
limits, rerun the download command to refresh the manifest, then rebuild your
training representations consistently. Do not mix vectors from different
tokenization or truncation settings. Use your own vectors if you add chunking.

Local assets contain **no private evaluation vectors or labels**. On Gradescope,
MiniLM and Jina have optional q/r caches; your code should also support encoding
without those caches so it can run locally. See the demo README for cache loading.
Do not include downloaded staff encoders or `local_assets/` in your submission ZIP.

## 7. Optional: test GenAI in your Open Approach

The unchanged demo never calls the API. For your own `approach6`, put your key in
`my_submission/credentials.py`, use the supplied `genai_wrapper.py`, then add
`--allow-api` to the local test command:

```bash
python local_environment/local_test.py --submission-dir my_submission --csv my_validation.csv --allow-api
```

This can consume real API requests. Only use it intentionally. Never print or
publish your key. The local flag enables the Open Approach's API configuration;
it does not enforce the grader's network restrictions or API budgets. Follow the
project submission instructions for permitted API use and limits.

## 8. Common problems

| Problem | What to check |
|---|---|
| `local_environment/...` or the sample CSV cannot be found | Run from the parent of the two extracted folders; check for an extra enclosing folder. |
| Python version error | Use Python 3.10-3.12, preferably 3.12, to create the environment. |
| `ModuleNotFoundError` for an installed library | Activate `.part2-venv`, or invoke its Python executable directly. Confirm setup completed. |
| A demo method cannot import `demo_model` | Restore `demo_model.py` beside the unchanged demo entrypoints. |
| Your model or helper cannot be found | Include it in the submission and locate artifacts with `weights_dir` or `Path(__file__).resolve().parent`. Avoid local absolute paths. |
| Missing input column | Use Part 2 data; the old `EssaySet/EssayText` format is not supported. |
| Missing `assets.json` | Finish optional encoder downloading, or omit `--assets` for methods that do not use encoders. |
| A method returns invalid predictions | Check IDs, output length, ordinary numeric types and the 0-100 range. |
| Local checks pass but Gradescope fails | Check CPU-only behavior, read-only submission files, subprocess/network use, artifact versions and resource limits. |

You can rerun the library checks independently:

```bash
python -m pip check
python local_environment/validate_environment.py
```

When ready to submit, ZIP the **contents** of your submission folder so the method
files are at the ZIP root. Include the saved artifacts and required helpers;
exclude `.part2-venv/`, `local_assets/`, test outputs and unrelated files. Use the
main Part 2 submission instructions and handout for all final requirements.

## Graph2Vec support

The environment installs **Karate Club 1.3.4** from the official pinned upstream
commit, including its dependencies. NumPy is 1.26.4 and SciPy is 1.12.0. Use Python
3.10-3.12; older environments should be rebuilt or updated with the supplied
setup script. Refit saved training artifacts in this environment to avoid mixing
NumPy/library versions. Git is not required: pip downloads the pinned source ZIP.

```python
from karateclub import Graph2Vec

model = Graph2Vec(dimensions=128, workers=1, min_count=1,
                  use_node_attribute="feature")
# Each NetworkX graph needs integer node IDs 0..n-1 and string node attribute "feature".
model.fit(training_graphs)       # training data only
training_vectors = model.get_embedding()
validation_vectors = model.infer(validation_graphs)
```

The names above are your prepared SQL-AST graph lists. Save the fitted model with
joblib and load it for inference; do not call fit on evaluation graphs. The sample
demo remains an interface placeholder, not an implementation of Graph2Vec.
