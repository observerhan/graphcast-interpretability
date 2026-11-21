"""Prepare Colab's separate GraphCast environment and verify its activation hook.

Run with Colab's Python: python scripts/setup_graphcast.py
The --check-hooks mode is used internally with the GraphCast environment's Python.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

from ams691_extract import GRAPHCAST_COMMIT


def patch_hooks_and_check_gpu():
    import importlib.util
    from pathlib import Path
    root = Path(next(iter(importlib.util.find_spec('graphcast').submodule_search_locations)))
    path = root / 'deep_typed_graph_net.py'
    old = '            tag = f"{self.name}/post_res/nodes/{nset}"\n            jdebug.callback('
    new = '            tag = f"{self.name}/post_res/nodes/{nset}"\n            if not am._should_save(tag, step_idx, nset):\n                continue\n            jdebug.callback('
    source = path.read_text()
    if old in source:
        assert source.count(old) == 1
        path.write_text(source.replace(old, new))
    elif new not in source:
        raise RuntimeError('Unexpected GraphCast hook source; cannot apply the memory-saving change.')
    from graphcast import graphcast, data_utils
    import jax
    assert jax.__version__ == '0.4.38'
    assert any(device.platform == 'gpu' for device in jax.devices()), 'JAX cannot see a GPU'
    print('GraphCast imports and GPU check passed.')


def main():
    if not shutil.which('nvidia-smi'):
        raise RuntimeError('Choose Runtime > Change runtime type > A100 GPU, then run again.')
    gpu = subprocess.run(
        ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'],
        capture_output=True, text=True,
    )
    if gpu.returncode != 0 or not gpu.stdout.strip():
        raise RuntimeError('Choose Runtime > Change runtime type > A100 GPU, then run again.')
    print('GPU:', gpu.stdout.strip())
    memory_mb = float(gpu.stdout.strip().splitlines()[0].rsplit(',', 1)[1])
    if memory_mb < 30000:
        raise RuntimeError('This notebook uses full GraphCast. Select an A100 or H100 GPU; the usual T4 has too little memory.')

    # GraphCast runs in its own automatically prepared Python environment.
    # This avoids changing Colab's imported packages or requiring a kernel restart.
    ENV_DIR = Path('/content/ams691_graphcast_env')
    PYTHON = str(ENV_DIR / 'bin/python')

    def setup_command(command):
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            print(result.stdout[-4000:])
            print(result.stderr[-8000:])
            raise RuntimeError('Setup stopped. The installation error is printed above.')

    if not Path(PYTHON).exists():
        setup_command([sys.executable, '-m', 'pip', 'install', '-q', 'virtualenv==20.31.2'])
        setup_command([sys.executable, '-m', 'virtualenv', '--python', sys.executable, str(ENV_DIR)])

    print('Installing the packages GraphCast needs...')
    packages = [
        'jax[cuda12]==0.4.38', 'flax', 'dm-haiku==0.0.13', 'chex==0.1.88',
        'numpy==2.2.6', 'pandas==2.2.3', 'scipy==1.15.3', 'xarray==2025.1.2',
        'zarr==2.18.7', 'numcodecs==0.15.1', 'gcsfs==2025.3.0',
        'google-cloud-storage==3.1.0', 'dask[array]==2025.3.0',
        'dm-tree==0.1.9', 'jraph==0.0.6.dev0', 'trimesh==4.6.8',
        'rtree==1.4.0', 'netCDF4==1.7.2',
    ]
    setup_command([PYTHON, '-m', 'pip', 'install', '-q', *packages])
    setup_command([PYTHON, '-m', 'pip', 'install', '-q', '--no-deps',
                   f'git+https://github.com/theodoremacmillan/graphcast.git@{GRAPHCAST_COMMIT}'])

    # The authors' callback copies every layer to the host before filtering.
    # Filter first so only the selected layer-8 mesh array is transferred.
    setup_command([PYTHON, str(Path(__file__).resolve()), '--check-hooks'])
    print('Setup complete. Continue to the next cell; no restart is needed.')

if __name__ == "__main__":
    if sys.argv[1:] == ['--check-hooks']:
        patch_hooks_and_check_gpu()
    elif not sys.argv[1:]:
        main()
    else:
        raise SystemExit('Usage: python scripts/setup_graphcast.py [--check-hooks]')
