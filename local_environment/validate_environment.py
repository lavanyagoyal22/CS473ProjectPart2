"""Exercise the installed Part 2 libraries without network access or model downloads."""
import importlib.metadata
import json


def validate():
    import numpy as np
    import pandas as pd
    import sqlglot
    from sqlglot.diff import diff
    from rapidfuzz.distance import Levenshtein
    from apted import APTED
    from apted.helpers import Tree
    import zss
    import networkx as nx
    from karateclub import Graph2Vec
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge
    import joblib
    import psutil
    import torch
    import onnxruntime as ort
    import sentence_transformers
    import transformers
    import sentencepiece
    import safetensors
    import tokenizers
    import google.protobuf
    from PIL import Image

    q, r = sqlglot.parse_one('SELECT a FROM t'), sqlglot.parse_one('SELECT b FROM t')
    assert diff(q, r)
    assert Levenshtein.distance('a', 'b') == 1
    assert APTED(Tree.from_text('{a}'), Tree.from_text('{b}')).compute_edit_distance() == 1
    assert zss.simple_distance(zss.Node('a'), zss.Node('b')) == 1
    graph = nx.path_graph(3)
    assert nx.weisfeiler_lehman_graph_hash(graph)
    documents = [TaggedDocument(['SELECT', 'COLUMN_a'], ['q']),
                 TaggedDocument(['SELECT', 'COLUMN_b'], ['r'])]
    model = Doc2Vec(documents, vector_size=8, min_count=1, workers=1, epochs=2, seed=7)
    assert model.infer_vector(['SELECT', 'COLUMN_a']).shape == (8,)
    graphs = [nx.path_graph(3), nx.path_graph(4)]
    for item in graphs:
        nx.set_node_attributes(item, {node: ('SELECT' if node == 0 else 'COLUMN')
                                     for node in item}, 'feature')
    graph_model = Graph2Vec(dimensions=8, workers=1, epochs=3, min_count=1,
                           down_sampling=0, use_node_attribute='feature', seed=7)
    graph_model.fit(graphs)
    assert graph_model.get_embedding().shape == (2, 8)
    import io
    buffer = io.BytesIO()
    joblib.dump(graph_model, buffer)
    buffer.seek(0)
    inferred = joblib.load(buffer).infer(graphs[:1])
    assert inferred.shape == (1, 8) and np.isfinite(inferred).all()
    source = json.loads(importlib.metadata.distribution('karateclub').read_text('direct_url.json'))
    assert 'cb46a91df8dcbeb2570debcf6a9d0c518107a2de' in source['url'], \
        'Karate Club must be installed from the pinned official source commit.'
    x = TfidfVectorizer().fit_transform(['select a from table', 'select b from table'])
    assert np.isfinite(Ridge().fit(x, [50., 75.]).predict(x)).all()
    assert 'CPUExecutionProvider' in ort.get_available_providers()
    assert torch.tensor([1., 2.], device='cpu').sum().item() == 3
    packages = ['numpy', 'scipy', 'pandas', 'scikit-learn', 'gensim', 'networkx',
                'sqlglot', 'rapidfuzz', 'apted', 'zss', 'torch', 'onnxruntime',
                'sentence-transformers', 'transformers', 'huggingface-hub',
                'tokenizers', 'safetensors', 'sentencepiece', 'protobuf', 'Pillow',
                'karateclub', 'decorator', 'python-louvain', 'pygsp', 'python-Levenshtein']
    return {'status': 'passed', 'karateclub_source': source['url'],
            'graph2vec_fit_save_load_infer': 'passed',
            'packages': {name: importlib.metadata.version(name)
                                            for name in packages}}


if __name__ == '__main__':
    print(json.dumps(validate(), indent=2))
