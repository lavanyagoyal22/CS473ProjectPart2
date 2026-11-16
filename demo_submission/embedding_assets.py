"""Load frozen local encoders using the same preprocessing as cached vectors."""
from pathlib import Path


def download_patterns(spec):
    patterns = ['config.json', 'config_sentence_transformers.json', 'modules.json',
                'sentence_bert_config.json', 'tokenizer*.json', 'special_tokens_map.json',
                'added_tokens.json', 'vocab.*', 'merges.txt', '*.model',
                '1_Pooling/config.json']
    if spec.get('backend') == 'onnx':
        patterns.append(spec['onnx_file'])
    else:
        patterns.append('*.safetensors')
    return patterns


class OnnxEncoder:
    """Jina's published CPU ONNX model; no remote Python model code is needed."""
    def __init__(self, spec, threads):
        import onnxruntime as ort
        from transformers import AutoTokenizer
        self.max_seq_length = spec.get('max_length', 2048)
        self.tokenizer = AutoTokenizer.from_pretrained(
            spec['path'], local_files_only=True, trust_remote_code=False)
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(Path(spec['path']) / spec['onnx_file']), sess_options=options,
            providers=['CPUExecutionProvider'])
        self.dimensions = spec['dimensions']

    def encode(self, texts, batch_size=4, **kwargs):
        import numpy as np
        if batch_size < 1:
            raise ValueError('batch_size must be positive.')
        batches = []
        for start in range(0, len(texts), batch_size):
            tokens = self.tokenizer(texts[start:start + batch_size], padding=True,
                                    truncation=True, max_length=self.max_seq_length,
                                    return_tensors='np')
            feeds = {}
            for item in self.session.get_inputs():
                if item.name == 'token_type_ids' and item.name not in tokens:
                    value = np.zeros_like(tokens['input_ids'])
                else:
                    value = tokens[item.name]
                dtypes = {'tensor(int64)': np.int64, 'tensor(int32)': np.int32,
                          'tensor(bool)': np.bool_}
                if item.type not in dtypes:
                    raise ValueError('Unsupported ONNX input type: ' + item.type)
                feeds[item.name] = value.astype(dtypes[item.type], copy=False)
            outputs = self.session.run(None, feeds)
            hidden = next((value for value in outputs if value.ndim == 3), None)
            if hidden is None:
                raise ValueError('Expected token embeddings from the Jina ONNX encoder.')
            mask = tokens['attention_mask'][..., None].astype(np.float32)
            vectors = (hidden.astype(np.float32) * mask).sum(axis=1) / mask.sum(axis=1).clip(min=1)
            vectors /= np.linalg.norm(vectors, axis=1, keepdims=True).clip(min=1e-12)
            if vectors.shape[1] != self.dimensions:
                raise ValueError('Unexpected embedding dimensions.')
            batches.append(vectors)
        return np.concatenate(batches) if batches else np.empty((0, self.dimensions), dtype=np.float32)


def load_encoder(spec, threads=4):
    """spec is assets['models'][name]; all inference is local and CPU-only."""
    if spec.get('backend') == 'onnx':
        return OnnxEncoder(spec, threads)
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(threads)
    model = SentenceTransformer(spec['path'], device='cpu', local_files_only=True,
                                trust_remote_code=False)
    model.max_seq_length = spec.get('max_length', 512)
    return model


def validate_encoder(model, spec):
    """A real encode validates local tokenizer, weights, runtime and output format."""
    import numpy as np
    vectors = model.encode(['SELECT a FROM t WHERE a > 1;',
                            'SELECT a FROM t WHERE a >= 1;'],
                           batch_size=2, normalize_embeddings=True, convert_to_numpy=True,
                           show_progress_bar=False)
    if vectors.shape != (2, spec['dimensions']) or not np.isfinite(vectors).all():
        raise ValueError('Encoder smoke check returned invalid vectors.')
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4):
        raise ValueError('Encoder smoke check returned unnormalized vectors.')
    return {'offline_validated': True, 'smoke_shape': list(vectors.shape)}
