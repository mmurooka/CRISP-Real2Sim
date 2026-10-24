#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.4}"
export PATH="$ROOT/.venv-contact/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MAX_JOBS="${MAX_JOBS:-4}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
command -v uv >/dev/null || { echo "uv is required." >&2; exit 1; }
[[ -f prep/Contact-Predictor/requirements.txt ]] || { echo "Initialize Contact-Predictor submodule first." >&2; exit 1; }
uv venv --allow-existing --python 3.10 .venv-contact
uv pip install --python .venv-contact/bin/python \
    torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121
uv pip install --python .venv-contact/bin/python pip 'setuptools<81' wheel ninja 'cython<3' numpy==1.24.2
TMP_REQ="$(mktemp)"
trap 'rm -f "$TMP_REQ"' EXIT
# Build PyTorch3D only after installing torch and the Python dependencies.
rg -v '^git\+https://github.com/facebookresearch/pytorch3d.git@stable$' \
    prep/Contact-Predictor/requirements.txt > "$TMP_REQ"
uv pip install --python .venv-contact/bin/python --no-build-isolation \
    -r "$TMP_REQ" torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 accelerate==0.24.1
uv pip install --python .venv-contact/bin/python --no-build-isolation \
    'git+https://github.com/facebookresearch/pytorch3d.git@stable'
.venv-contact/bin/python - <<'PY'
import torch
import pytorch3d
from transformers import AutoTokenizer, BitsAndBytesConfig
assert torch.cuda.is_available(), 'CUDA is unavailable in the contact environment'
print('Contact environment ready:', torch.cuda.get_device_name(0))
PY
