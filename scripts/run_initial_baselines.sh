#!/usr/bin/env bash
set -euo pipefail

usage() {
	cat <<'EOF'
Run initial baselines (U-Net, U-Net++, BGD-SF PolySegNet).

Usage:
  bash scripts/run_initial_baselines.sh [options] [extra args passed to src/main.py]

Examples:
  bash scripts/run_initial_baselines.sh --dataset-root data/Kvasir-SEG --image-dir images --mask-dir masks
  bash scripts/run_initial_baselines.sh --dataset-root data/Kvasir-SEG --image-dir images --mask-dir masks \
    --pranet-pred-dir /path/to/pranet/preds --polyp-pvt-pred-dir /path/to/polyp-pvt/preds
EOF
}

SKIP_EVAL=0
DATASET_ROOT=""
IMAGE_DIR=""
MASK_DIR=""
IMAGE_SIZE=""
SPLIT="test"
SPLIT_DIR=""
PRANET_PRED_DIR=""
POLYP_PVT_PRED_DIR=""
PRANET_PRED_SUFFIX=""
POLYP_PVT_PRED_SUFFIX=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
	case "$1" in
		--skip-eval)
			SKIP_EVAL=1
			shift
			;;
		--dataset-root)
			DATASET_ROOT="$2"
			EXTRA_ARGS+=("$1" "$2")
			shift 2
			;;
		--image-dir|--image_dir)
			IMAGE_DIR="$2"
			EXTRA_ARGS+=("$1" "$2")
			shift 2
			;;
		--mask-dir)
			MASK_DIR="$2"
			EXTRA_ARGS+=("$1" "$2")
			shift 2
			;;
		--image-size)
			IMAGE_SIZE="$2"
			EXTRA_ARGS+=("$1" "$2")
			shift 2
			;;
		--split)
			SPLIT="$2"
			shift 2
			;;
		--split-dir)
			SPLIT_DIR="$2"
			shift 2
			;;
		--pranet-pred-dir)
			PRANET_PRED_DIR="$2"
			shift 2
			;;
		--pranet-pred-suffix)
			PRANET_PRED_SUFFIX="$2"
			shift 2
			;;
		--polyp-pvt-pred-dir)
			POLYP_PVT_PRED_DIR="$2"
			shift 2
			;;
		--polyp-pvt-pred-suffix)
			POLYP_PVT_PRED_SUFFIX="$2"
			shift 2
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

run_train_eval "configs/baselines/unet.yaml"
run_train_eval "configs/baselines/unetpp.yaml"
run_train_eval "configs/baselines/bgdsf_full.yaml"

run_prediction_eval() {
	local method_name="$1"
	local pred_dir="$2"
	local output_dir="$3"
	local pred_suffix="$4"

	if [[ -z "$pred_dir" ]]; then
		return
	fi
	if [[ -z "$DATASET_ROOT" || -z "$IMAGE_DIR" || -z "$MASK_DIR" ]]; then
		echo "Skipping ${method_name} eval: --dataset-root/--image-dir/--mask-dir required."
		return
	fi

	CMD=(python scripts/evaluate_baseline_predictions.py
		--method-name "$method_name"
		--dataset-root "$DATASET_ROOT"
		--image-dir "$IMAGE_DIR"
		--mask-dir "$MASK_DIR"
		--pred-dir "$pred_dir"
		--output-dir "$output_dir")

	if [[ -n "$IMAGE_SIZE" ]]; then
		CMD+=(--image-size "$IMAGE_SIZE")
	fi
	if [[ -n "$SPLIT" ]]; then
		CMD+=(--split "$SPLIT")
	fi
	if [[ -n "$SPLIT_DIR" ]]; then
		CMD+=(--split-dir "$SPLIT_DIR")
	fi
	if [[ -n "$pred_suffix" ]]; then
		CMD+=(--pred-suffix "$pred_suffix")
	fi

	"${CMD[@]}"
}

run_prediction_eval "PraNet" "$PRANET_PRED_DIR" "outputs/official_baselines/pranet" "$PRANET_PRED_SUFFIX"
run_prediction_eval "Polyp-PVT" "$POLYP_PVT_PRED_DIR" "outputs/official_baselines/polyp_pvt" "$POLYP_PVT_PRED_SUFFIX"
