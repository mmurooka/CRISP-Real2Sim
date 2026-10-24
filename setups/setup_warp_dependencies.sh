#!/usr/bin/env bash
# Optional physics benchmark environment; never modifies the main reconstruction venv.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_PYTHON="${1:-$ROOT/.venv/bin/python}"
WARP_ENV="$ROOT/.venv-warp"
WARP_SOURCE="$ROOT/.cache/mujoco-warp"
WARP_COMMIT="09ec1da65ac44bd68703f9507c9c9e8f22ae7c71"
export UV_CACHE_DIR="$ROOT/.cache/uv"
if [[ ! -d "$WARP_SOURCE/.git" ]]; then
    git clone --filter=blob:none https://github.com/google-deepmind/mujoco_warp.git "$WARP_SOURCE"
    git -C "$WARP_SOURCE" checkout "$WARP_COMMIT"
fi
[[ "$(git -C "$WARP_SOURCE" rev-parse HEAD)" == "$WARP_COMMIT" ]] || {
    echo 'The cached Warp source has a different revision; use the pinned revision in a separate checkout.' >&2
    exit 1
}
if [[ ! -x "$WARP_ENV/bin/python" ]]; then
    uv venv --python "$BASE_PYTHON" "$WARP_ENV"
fi
# Exact official PyPI artifacts avoid the upstream project's nightly-index override.
# These wheels target the project's Python 3.10 / Linux x86_64 environment.
MUJOCO_WHEEL='https://files.pythonhosted.org/packages/3e/59/d22fef0352a7e7a8e79f5580611a62015c47a37838fa7ae887caba81df92/mujoco-3.3.7-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl#sha256=771923e71faa19749fed8dae1329b70b77d1cd9cef699d0e0ad984e00ee05c0a'
WARP_WHEEL='https://files.pythonhosted.org/packages/5f/75/d99bca2f89baa0fdcaa42d2dd8ceb2cb9806467f18a27515df1f73eef2e8/warp_lang-1.10.0-py3-none-manylinux_2_28_x86_64.whl#sha256=1bdff31e170b00c89fb9d8b647e906fefdcdcf8741925126d1c0be42783174fa'
uv pip install --no-config --python "$WARP_ENV/bin/python" --index-url https://pypi.org/simple \
    "$MUJOCO_WHEEL" "$WARP_WHEEL" numpy==1.26.4 "$WARP_SOURCE"
"$WARP_ENV/bin/python" -c 'import mujoco, warp; print("MuJoCo", mujoco.__version__, "Warp", warp.__version__)'
