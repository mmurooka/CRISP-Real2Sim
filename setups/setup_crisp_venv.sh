#!/usr/bin/env bash
# Alternative to setup_crisp.sh when a system CUDA toolkit is available.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.4}"
export CUB_HOME="${CUB_HOME:-$CUDA_HOME/include}"
export PATH="$ROOT/.venv/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MAX_JOBS="${MAX_JOBS:-4}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
[[ -x "$CUDA_HOME/bin/nvcc" ]] || { echo "Set CUDA_HOME to your CUDA 12.4 toolkit." >&2; exit 1; }
command -v uv >/dev/null || { echo "uv is required." >&2; exit 1; }
uv venv --allow-existing --python 3.10 .venv
uv pip install --python .venv/bin/python -r requirements-crisp-video.txt
uv pip install --python .venv/bin/python 'setuptools<81' wheel ninja cmake
uv pip install --python .venv/bin/python --no-deps nerfstudio==0.3.3
uv pip install --python .venv/bin/python --no-build-isolation \
    'git+https://github.com/facebookresearch/pytorch3d.git@stable' \
    'git+https://github.com/mattloper/chumpy.git'
SAM2_BUILD_ALLOW_ERRORS=1 uv pip install --python .venv/bin/python --no-deps --no-build-isolation \
    -e prep/AutoMask -e prep/HMR -e prep/UFM/UniCeption -e prep/UFM -e vis_scripts/viser_m
uv pip install --python .venv/bin/python --no-deps --no-build-isolation ./prep/MogeSAM/third_party/pointops2
# This upstream setup.py defines two distributions; install both in place.
(
    cd prep/MogeSAM/third_party/megasam/base
    "$ROOT/.venv/bin/python" setup.py install
)
uv pip install --python .venv/bin/python -e prep/GeoCalib
uv pip install --python .venv/bin/python numpy==1.26.4 opencv-python==4.6.0.66
PYTHONPATH="$ROOT/runtime_shims${PYTHONPATH:+:$PYTHONPATH}" \
    bash setups/validate_crisp_video_env.sh "$ROOT/.venv/bin/python"
