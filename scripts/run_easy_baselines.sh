#!/usr/bin/env bash
set -euo pipefail

usage() {
	cat <<'EOF'
Run easy baselines (SMP models + BGD-SF PolySegNet).

Usage:
  bash scripts/run_easy_baselines.sh [--skip-eval] [extra args passed to src/main.py]

Example:
  bash scripts/run_easy_baselines.sh --dataset-root data/Kvasir-SEG --image-dir images --mask-dir masks
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

run_train_eval() {
	local cfg="$1"
	python src/main.py --mode train --config "$cfg" "${EXTRA_ARGS[@]}"
	if [[ "$SKIP_EVAL" -eq 0 ]]; then
		python src/main.py --mode eval --config "$cfg" "${EXTRA_ARGS[@]}"
	fi
}

run_train_eval "configs/baselines/smp_unet.yaml"
run_train_eval "configs/baselines/smp_unetpp.yaml"
run_train_eval "configs/baselines/smp_deeplabv3plus.yaml"
run_train_eval "configs/baselines/smp_fpn.yaml"
run_train_eval "configs/baselines/smp_pspnet.yaml"
run_train_eval "configs/baselines/smp_linknet.yaml"
run_train_eval "configs/baselines/bgdsf_full.yaml"
