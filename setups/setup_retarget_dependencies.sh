#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${1:-$ROOT/.venv/bin/python}"
CONSTRAINTS="$(mktemp)"
trap 'rm -f "$CONSTRAINTS"' EXIT
# Do not upgrade the working video reconstruction environment when adding IK.
"$PYTHON" - "$CONSTRAINTS" <<'PY'
import importlib.metadata as metadata
from pathlib import Path
import sys
packages=['numpy','scipy','torch','trimesh','viser','yourdfpy','tyro']
Path(sys.argv[1]).write_text('\n'.join(f'{p}=={metadata.version(p)}' for p in packages)+'\n')
PY
uv pip install --python "$PYTHON" --constraint "$CONSTRAINTS" \
    mujoco==3.2.7 cvxpy==1.6.7 libigl==2.5.1
"$PYTHON" - <<'PY'
import mujoco, cvxpy, igl
assert 'CLARABEL' in cvxpy.installed_solvers()
print('Retargeting dependencies:', mujoco.__version__, cvxpy.__version__)
PY
