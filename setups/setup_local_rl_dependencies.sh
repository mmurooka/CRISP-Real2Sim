#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${1:-$ROOT/.venv/bin/python}"
CONSTRAINTS="$(mktemp)"
trap 'rm -f "$CONSTRAINTS"' EXIT
"$PYTHON" - "$CONSTRAINTS" <<'PY'
from importlib.metadata import version
from pathlib import Path
import sys
Path(sys.argv[1]).write_text('\n'.join(f'{name}=={version(name)}' for name in ('numpy','torch','mujoco'))+'\n')
PY
uv pip install --python "$PYTHON" --constraint "$CONSTRAINTS" \
    stable-baselines3==2.4.1 gymnasium==1.0.0 loguru==0.7.3
