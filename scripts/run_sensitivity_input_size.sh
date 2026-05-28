#!/usr/bin/env bash
set -euo pipefail

usage() {
	cat <<'EOF'
Run input size sensitivity for BGD-SF PolySegNet.

Usage:
  bash scripts/run_sensitivity_input_size.sh [--skip-eval] [extra args passed to src/main.py]

Example:
  bash scripts/run_sensitivity_input_size.sh \
    --dataset-root data/Kvasir-SEG --image-dir images --mask-dir masks \
    --epochs 50 --batch-size 8
EOF
}

SKIP_EVAL=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
	case "$1" in
		--skip-eval)
			SKIP_EVAL=1
			shift
			;;
		-h|--help)
			usage
			exit 0
			;;
		*)
			EXTRA_ARGS+=("$1")
			shift
			;;
	esac
done

BASE_ARGS=(
	--model-name bgdsf_polysegnet
	--use-msca
	--use-csaf
	--use-mbgh
	--use-boundary-loss
	--use-dynamic-weighting
	--use-boundary-guidance
	--use-multilevel-boundary
	--use-freq-aug
)

run_train_eval() {
	local exp_name="$1"
	local image_size="$2"

	python src/main.py --mode train --experiment-name "$exp_name" \
		"${BASE_ARGS[@]}" --image-size "$image_size" \
		"${EXTRA_ARGS[@]}" --seed 42

	if [[ "$SKIP_EVAL" -eq 0 ]]; then
		python src/main.py --mode eval --experiment-name "$exp_name" \
			"${BASE_ARGS[@]}" --image-size "$image_size" \
			"${EXTRA_ARGS[@]}" --seed 42
	fi
}

run_train_eval "bgdsf_size_256" "256"
run_train_eval "bgdsf_size_352" "352"
run_train_eval "bgdsf_size_512" "512"
