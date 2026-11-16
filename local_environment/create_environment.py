#!/usr/bin/env python3
"""Build the Part 2 CPU environment; download only public frozen model resources."""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import venv

HERE = Path(__file__).resolve().parent

def requirements(name):
    return [line.strip() for line in (HERE / name).read_text().splitlines()
            if line.strip() and not line.lstrip().startswith('#')]

SQL_PACKAGES = [line for line in requirements('requirements.txt')
                if line.split('==')[0] in ('sqlglot', 'rapidfuzz', 'networkx', 'apted', 'zss')]
BASIC = [line for line in requirements('requirements.txt') if line not in SQL_PACKAGES]
EMBEDDINGS = requirements('requirements-embeddings.txt')
ASSETS_CONFIG = json.loads((HERE / 'assets_config.json').read_text())
ENCODERS = ASSETS_CONFIG['models']


def run(command):
    print('+', ' '.join(map(str, command)), flush=True)
    subprocess.check_call([str(value) for value in command])


def venv_python(folder):
    return folder / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--venv', default='.part2-venv')
    parser.add_argument('--download-encoders', action='store_true',
                        help='Download and offline-validate five CPU encoders; no private data/caches.')
    parser.add_argument('--assets-dir', type=Path, default=Path('local_assets'))
    args = parser.parse_args()
    if not (3, 10) <= sys.version_info[:2] <= (3, 12):
        parser.error('Use Python 3.10-3.12; Python 3.12 is recommended for the Karate Club stack.')
    folder = Path(args.venv).resolve()
    if not venv_python(folder).exists():
        venv.EnvBuilder(with_pip=True).create(folder)
    py = venv_python(folder)
    run([py, '-m', 'pip', 'install', '--upgrade', 'pip'])
    if platform.system() in ('Linux', 'Windows'):
        run([py, '-m', 'pip', 'install', 'torch==2.8.0+cpu',
             '--index-url', 'https://download.pytorch.org/whl/cpu'])
    else:
        run([py, '-m', 'pip', 'install', 'torch==2.8.0'])
    run([py, '-m', 'pip', 'install', '-r', HERE / 'requirements.txt',
         '-r', HERE / 'requirements-embeddings.txt'])
    run([py, '-m', 'pip', 'check'])
    run([py, HERE / 'validate_environment.py'])
    if args.download_encoders:
        code = r"""
import json, os, sys
from pathlib import Path
sys.path.insert(0, os.environ['PART2_HELPER_DIR'])
from huggingface_hub import snapshot_download
from embedding_assets import download_patterns, load_encoder, validate_encoder
config = json.loads(Path(os.environ['PART2_CONFIG']).read_text())
root = Path(os.environ['PART2_ASSETS']).resolve()
root.mkdir(parents=True, exist_ok=True)
manifest = {'models': {}, 'precomputed': {}, 'measurements': {}}
for name, spec in config['models'].items():
    folder = root / name
    snapshot_download(repo_id=spec['repo_id'], revision=spec['revision'],
                      local_dir=str(folder), allow_patterns=download_patterns(spec), max_workers=2)
    metadata = {'path': str(folder), **spec}
    metadata.setdefault('max_length', config['max_length'])
    metadata.setdefault('batch_size', config['batch_size'])
    encoder = load_encoder(metadata, threads=config['threads'])
    manifest['measurements'][name] = validate_encoder(encoder, metadata)
    del encoder
    manifest['models'][name] = metadata
(root / 'assets.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print('Five encoders validated; no evaluation caches downloaded:', root)
"""
        env = dict(os.environ, PART2_HELPER_DIR=str(HERE),
                   PART2_CONFIG=str(HERE / 'assets_config.json'),
                   PART2_ASSETS=str(args.assets_dir.resolve()),
                   HF_HUB_DISABLE_XET='1', TOKENIZERS_PARALLELISM='false')
        subprocess.check_call([str(py), '-c', code], env=env)
    print('Environment ready:', py)
    print('Use a Part 2 training/validation CSV with local_test.py; no question JSON required.')


if __name__ == '__main__':
    main()
