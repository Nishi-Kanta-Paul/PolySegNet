#!/usr/bin/env bash
set -euo pipefail

usage() {
	cat <<'EOF'
Run multi-seed BGD-SF PolySegNet training + evaluation.

Usage:
  bash scripts/run_multiseed_bgdsf.sh [extra args passed to src/main.py]

Example:
  bash scripts/run_multiseed_bgdsf.sh \
    --dataset-root data/Kvasir-SEG \
    --image-dir images \
    --mask-dir masks \
    --epochs 50 \
    --batch-size 8 \
    --image-size 352
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
	usage
	exit 0
fi

EXTRA_ARGS=("$@")
CONFIG_PATH="configs/baselines/bgdsf_full.yaml"

run_seed() {
	local seed="$1"
	local exp_name="bgdsf_polysegnet_seed${seed}"
	python src/main.py --mode train --config "$CONFIG_PATH" --seed "$seed" --experiment-name "$exp_name" "${EXTRA_ARGS[@]}"
	python src/main.py --mode eval --config "$CONFIG_PATH" --seed "$seed" --experiment-name "$exp_name" "${EXTRA_ARGS[@]}"
}

run_seed 42
run_seed 123
run_seed 2025
