# Part 2 submission interface demo

All eight entrypoints are runnable **placeholders returning 50.0**. They do not
implement the required algorithms and are not a completed project submission.
Replace each selected entrypoint's `load_model()` and model with your real method.
`demo_model.py` is only a shared interface example and may be removed when unused.

Required methods: `approach1_option1.py` (AST baseline),
`approach1_option2.py` (IDF-weighted AST), and `approach6.py` (Open Approach).
Choose at least two of `approach2.py` (text edit distance), `approach3.py` (TF-IDF),
`approach4.py` (dense embeddings), and `approach5.py` (Graph2Vec).
`approach1_option3.py` is optional custom-weighted AST and cannot replace a required method.
Remove unselected placeholder entrypoints from your final submission.
Methods run sequentially, with `approach1_option3` last. The leaderboard retains
its method-number display order.

The grader calls:

```python
model = load_model(weights_dir, config, assets)
predictions = model.predict(test_rows, questions)
```

Train and tune on the labeled training data beforehand; save any necessary state
under `weights/`. The grader never invokes training. The shared weights directory
may contain separate artifacts for each method. SQL parse failures require a
documented fallback so your method still returns a prediction for every row.

Input rows contain `id`, `hw_num`, `question_number`, `question_prompt`, `q`, `r`,
`db_instance_path`, and compatibility fields `question` (`hw_num:question_number`)
and `text` (`q`). Metadata is generated from the CSV, not from `questions.json`.
Scores and manual explanations are hidden from prediction. Preserve each input ID
exactly once in a returned list of `{"id": "...", "score": 85.5}` dictionaries.
Scores must be finite Python numbers in [0, 100]. Do not round for RMSE/MAE.

The leaderboard reports Best RMSE, each method's RMSE (lower is better), and
Valid submission. MAE and Pearson correlation remain in results.json feedback
and details, not in leaderboard columns.
`Valid submission` checks the successful method combination, not whether the
algorithms are implemented: the placeholders pass the interface check only.

From the project root, after creating/activating the local environment:

```bash
python demo_submission/local_test.py --submission-dir demo_submission --csv demo_submission/demo_training_sample.csv
```

The sample is eight public training rows, two per homework; it is not test data.
The checker strips grades and manual explanations before prediction and continues
past individual method failures. Exit code 0 means the required combination
succeeded; 1 means it did not. `--structure-only` checks file presence only.

API use is optional and enabled for `approach6` in grading. Put your key in
`credentials.py` (see `credentials_example.py`). The supplied wrapper is replaced
by the trusted wrapper during grading. The separate `genai_wrapper_example.py`
only checks connectivity; it never runs automatically. Use `--allow-api` to set
the local checker's Open Approach API flag. Local checks do not enforce the
grader's Linux isolation or API restrictions. The demo never calls the API.

Frozen encoders are available through `assets['models']` when installed.
`assets['precomputed']` may contain MiniLM and Jina Code q/r caches. Load with
`numpy.load(path, allow_pickle=False)`; arrays are `ids`, `q_vectors`, `r_vectors`,
`q_hashes`, `r_hashes`, plus `cache_version` and `encoder_config`. Match rows by
`ids`, never by position or by original CSV IDs. IDs are randomized per run.
Vectors use L2 normalization; dot products give cosine similarity. No grades,
manual explanations or question prompts are encoded. Cache vectors are valid
only for the recorded frozen encoder configuration; use your own encoding if
you change tokenization, max length, pooling or model weights.

To encode training data or run without caches, use the same helper:

```python
from embedding_assets import load_encoder
spec = assets['models']['jina_code']
encoder = load_encoder(spec, threads=4)
q_vectors = encoder.encode([row['q'] for row in test_rows],
                          batch_size=spec['batch_size'],
                          normalize_embeddings=True, convert_to_numpy=True)
```

Jina Code uses its published quantized ONNX model on CPU, masked mean pooling,
2048-token truncation and L2 normalization; it does not require remote model code.
Use identical preprocessing for training and inference. Local environment
downloads do not include private evaluation caches.
The general encoders use a 512-token limit and batch size 16; Jina uses 2048 and
batch size 2. These limits include special tokens. General encoders still truncate
longer SQL. If updating an older package, refresh the local assets manifest by
re-running encoder downloading and rebuild fitted training features with the
same settings. Do not reuse old 128/512-length staff caches with new settings.
Database paths are dataset metadata; they do not by themselves install databases
in the worker. Do not assume they point to accessible files during grading.

Karate Club Graph2Vec is available in the supported environment (official pinned
1.3.4 source). Use `use_node_attribute="feature"` for string SQL-AST node labels,
train on training folds only, and save the fitted model for `infer()` at grading
time. Keep node IDs consecutive integers. NumPy/SciPy are 1.26.4/1.12.0; recreate
older saved artifacts in the matching environment. See the local environment
README for an API example; `approach5.py` is still only an interface placeholder.
