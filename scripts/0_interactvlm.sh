#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    cat <<'EOF' >&2
Usage:
  bash scripts/0_interactvlm.sh <SEQ_ROOT> [OBJECT_NAME]

Example:
  bash scripts/0_interactvlm.sh /abs/path/to/data/demo/wall-kicking stairs

Notes:
  - Pass the sequence root without the `_img` suffix.
  - Uses the separate Contact-Predictor env `crisp_contact` by default.
  - Outputs are written under `results/init/contacts` and `results/init/contact_vis`.
EOF
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CP_DIR="$REPO_ROOT/prep/Contact-Predictor"
CONTACT_ENV="${CONTACT_ENV:-crisp_contact}"
CONTACT_PYTHON="${CONTACT_PYTHON:-}"
CONTACT_QUANTIZATION="${CONTACT_QUANTIZATION:-none}"
if [[ -z "$CONTACT_PYTHON" && -x "$REPO_ROOT/.venv-contact/bin/python" ]]; then
    CONTACT_PYTHON="$REPO_ROOT/.venv-contact/bin/python"
fi

ROOT_INPUT="$(realpath -m "$1")"
OBJ="${2:-stairs}"

if [[ "$ROOT_INPUT" == *_img ]]; then
    DATA_PATH="$ROOT_INPUT"
else
    DATA_PATH="${ROOT_INPUT%/}_img"
fi

[[ -d "$CP_DIR" ]] || { echo "Contact-Predictor not found: $CP_DIR" >&2; exit 1; }
[[ -d "$DATA_PATH" ]] || { echo "Image folder not found: $DATA_PATH" >&2; exit 1; }

if [[ -n "$CONTACT_PYTHON" ]]; then
    CONTACT_PYTHON="$(realpath "$CONTACT_PYTHON")"
    [[ -x "$CONTACT_PYTHON" ]] || { echo "Python not executable: $CONTACT_PYTHON" >&2; exit 1; }
else
    command -v conda >/dev/null 2>&1 || { echo "Set CONTACT_PYTHON to the contact venv Python, or install conda." >&2; exit 1; }
    conda run -n "$CONTACT_ENV" python -V >/dev/null 2>&1 || {
    echo "Conda env '$CONTACT_ENV' not found. Run: bash setups/setup_crisp_contact.sh $CONTACT_ENV" >&2
    exit 1
    }
fi

CONTACT_ARGS=()
case "$CONTACT_QUANTIZATION" in
    none) CONTACT_ARGS+=(--precision "${CONTACT_PRECISION:-bf16}") ;;
    4bit|8bit)
        # process.py's bf16 path calls .cuda(), which quantized models reject.
        CONTACT_ARGS+=(--precision fp16 "--load_in_${CONTACT_QUANTIZATION}") ;;
    *) echo "CONTACT_QUANTIZATION must be none, 4bit, or 8bit." >&2; exit 1 ;;
esac

MODEL_PATH="${CONTACT_MODEL_PATH:-$CP_DIR/trained_models/interactvlm-3d-hcontact-wScene-damon-lemon-rich}"
[[ -f "$MODEL_PATH/config.json" ]] || {
    echo "Contact model missing: $MODEL_PATH (register at https://interactvlm.is.tue.mpg.de and fetch hcontact-wScene)." >&2
    exit 1
}
[[ -f "$CP_DIR/data/smpl_to_smplx.pkl" && -d "$CP_DIR/data/hcontact_vitruvian" ]] || {
    echo "Contact data missing: extract InteractVLM data.zip in $CP_DIR." >&2
    exit 1
}
CONTACT_ARGS+=(--version "$(realpath "$MODEL_PATH")")

GPU_LIST=$(nvidia-smi --query-gpu=index --format=csv,noheader)
mapfile -t GPU_IDS <<< "$GPU_LIST"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] && IFS=',' read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
GPU_COUNT=${#GPU_IDS[@]}
[[ -n "$GPU_LIST" && "$GPU_COUNT" -gt 0 ]] || { echo "No NVIDIA GPUs available." >&2; exit 1; }

echo "Found $GPU_COUNT GPUs -> ${GPU_IDS[*]}"
echo "Scanning $DATA_PATH"
echo "Using Contact-Predictor Python: ${CONTACT_PYTHON:-conda:$CONTACT_ENV}"

mapfile -d '' DIRS < <(find "$DATA_PATH" -mindepth 1 -maxdepth 1 -type d -print0)
NUM_DIRS=${#DIRS[@]}
echo "$NUM_DIRS folders to process"
(( NUM_DIRS > 0 )) || { echo "No sequences found in $DATA_PATH." >&2; exit 1; }

worker() {
    local gpu_id="$1"
    shift
    local folders=("$@")

    local python_bin="$CONTACT_PYTHON"
    if [[ -z "$python_bin" ]]; then
        eval "$(conda shell.bash hook)"
        conda activate "$CONTACT_ENV"
        python_bin="$(command -v python)"
    fi
    cd "$CP_DIR"

    for cam_folder in "${folders[@]}"; do
        local seq
        seq="$(basename "$cam_folder")"
        local parent_dir
        parent_dir="$(dirname "$cam_folder")"

        echo "GPU $gpu_id | $seq"
        CUDA_VISIBLE_DEVICES="$gpu_id" \
            "$python_bin" process.py \
            --base_dir "$parent_dir" \
            --tgt_name "$seq" \
            --obj "$OBJ" "${CONTACT_ARGS[@]}" || return $?
    done
}

pids=()
for (( slot=0; slot<GPU_COUNT; slot++ )); do
    gpu_id="${GPU_IDS[slot]}"
    gpu_dirs=()
    for (( idx=slot; idx<NUM_DIRS; idx+=GPU_COUNT )); do
        gpu_dirs+=("${DIRS[idx]}")
    done
    worker "$gpu_id" "${gpu_dirs[@]}" &
    pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
done
(( failed == 0 )) || { echo "Contact inference failed; see worker errors above." >&2; exit 1; }
echo "All jobs finished."
