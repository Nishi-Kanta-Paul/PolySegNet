#!/usr/bin/env bash
set -euo pipefail

python src/main.py --mode infer "$@"

EXP_NAME=${EXPERIMENT_NAME:-polysegnet}
INFER_DIR="experiments/${EXP_NAME}/inference"
FIG_DIR="outputs/figures"

if [[ -d "${INFER_DIR}" ]]; then
	mkdir -p "${FIG_DIR}"
	cp -f "${INFER_DIR}"/*_overlay.png "${FIG_DIR}"/ 2>/dev/null || true
fi
