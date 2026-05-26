#!/usr/bin/env bash
set -euo pipefail

usage() {
	cat <<'EOF'
Generate all paper figures (Figure 1-8).

Required arguments:
  --fig1-cases-file PATH
  --fig1-prev-ckpt PATH
  --fig1-ours-ckpt PATH
  --fig2-image-dir PATH
  --fig2-mask-dir PATH
  --fig2-ckpt PATH
  --fig3-ckpt PATH
  --fig3-images PATHS (comma-separated)
  --fig4-csv PATH
  --fig5-csv PATH
  --fig5-train-dataset NAME
  --fig6-images PATHS (comma-separated)
  --fig6-ckpt-without PATH
  --fig6-ckpt-with PATH
  --fig7-ckpt PATH
  --fig7-image-dir PATH
  --fig7-mask-dir PATH
  --fig8-csv PATH

Optional arguments:
  --datasets PATHS (comma-separated dataset roots)
  --device DEVICE (default: cuda)
  --image-size SIZE (default: 352)
  --fig2-indices LIST (default: 0,1,2,3)
  --fig7-count N (default: 4)

Example:
  bash scripts/make_all_figures.sh \
    --datasets /data/CVC-ClinicDB,/data/Kvasir-SEG \
    --fig1-cases-file cases.json \
    --fig1-prev-ckpt experiments/prev/checkpoints/best.pth \
    --fig1-ours-ckpt experiments/bgdsf_polysegnet_full/checkpoints/best.pth \
    --fig2-image-dir data/Kvasir-SEG/images --fig2-mask-dir data/Kvasir-SEG/masks \
    --fig2-ckpt experiments/bgdsf_polysegnet_full/checkpoints/best.pth \
    --fig3-ckpt experiments/bgdsf_polysegnet_full/checkpoints/best.pth \
    --fig3-images img1.png,img2.png \
    --fig4-csv outputs/tables/ablation.csv \
    --fig5-csv outputs/tables/generalization.csv --fig5-train-dataset Kvasir-SEG \
    --fig6-images img1.png,img2.png \
    --fig6-ckpt-without experiments/bgdsf_polysegnet_no_freqaug/checkpoints/best.pth \
    --fig6-ckpt-with experiments/bgdsf_polysegnet_full/checkpoints/best.pth \
    --fig7-ckpt experiments/bgdsf_polysegnet_full/checkpoints/best.pth \
    --fig7-image-dir data/Kvasir-SEG/images --fig7-mask-dir data/Kvasir-SEG/masks \
    --fig8-csv outputs/tables/comparison_long.csv
EOF
}

DEVICE="cuda"
IMAGE_SIZE="352"
FIG2_INDICES="0,1,2,3"
FIG7_COUNT="4"

DATASETS=()
FIG3_IMAGES=()
FIG6_IMAGES=()

FIG1_CASES_FILE=""
FIG1_PREV_CKPT=""
FIG1_OURS_CKPT=""
FIG2_IMAGE_DIR=""
FIG2_MASK_DIR=""
FIG2_CKPT=""
FIG3_CKPT=""
FIG4_CSV=""
FIG5_CSV=""
FIG5_TRAIN_DATASET=""
FIG6_CKPT_WITHOUT=""
FIG6_CKPT_WITH=""
FIG7_CKPT=""
FIG7_IMAGE_DIR=""
FIG7_MASK_DIR=""
FIG8_CSV=""

while [[ $# -gt 0 ]]; do
	case "$1" in
		--datasets)
			IFS=',' read -r -a DATASETS <<< "$2"
			shift 2
			;;
		--device)
			DEVICE="$2"
			shift 2
			;;
		--image-size)
			IMAGE_SIZE="$2"
			shift 2
			;;
		--fig1-cases-file)
			FIG1_CASES_FILE="$2"
			shift 2
			;;
		--fig1-prev-ckpt)
			FIG1_PREV_CKPT="$2"
			shift 2
			;;
		--fig1-ours-ckpt)
			FIG1_OURS_CKPT="$2"
			shift 2
			;;
		--fig2-image-dir)
			FIG2_IMAGE_DIR="$2"
			shift 2
			;;
		--fig2-mask-dir)
			FIG2_MASK_DIR="$2"
			shift 2
			;;
		--fig2-ckpt)
			FIG2_CKPT="$2"
			shift 2
			;;
		--fig2-indices)
			FIG2_INDICES="$2"
			shift 2
			;;
		--fig3-ckpt)
			FIG3_CKPT="$2"
			shift 2
			;;
		--fig3-images)
			IFS=',' read -r -a FIG3_IMAGES <<< "$2"
			shift 2
			;;
		--fig4-csv)
			FIG4_CSV="$2"
			shift 2
			;;
		--fig5-csv)
			FIG5_CSV="$2"
			shift 2
			;;
		--fig5-train-dataset)
			FIG5_TRAIN_DATASET="$2"
			shift 2
			;;
		--fig6-images)
			IFS=',' read -r -a FIG6_IMAGES <<< "$2"
			shift 2
			;;
		--fig6-ckpt-without)
			FIG6_CKPT_WITHOUT="$2"
			shift 2
			;;
		--fig6-ckpt-with)
			FIG6_CKPT_WITH="$2"
			shift 2
			;;
		--fig7-ckpt)
			FIG7_CKPT="$2"
			shift 2
			;;
		--fig7-image-dir)
			FIG7_IMAGE_DIR="$2"
			shift 2
			;;
		--fig7-mask-dir)
			FIG7_MASK_DIR="$2"
			shift 2
			;;
		--fig7-count)
			FIG7_COUNT="$2"
			shift 2
			;;
		--fig8-csv)
			FIG8_CSV="$2"
			shift 2
			;;
		-h|--help)
			usage
			exit 0
			;;
		*)
			echo "Unknown argument: $1"
			usage
			exit 1
			;;
	esac
done

require_arg() {
	local name="$1"
	local value="$2"
	if [[ -z "$value" ]]; then
		echo "Missing required argument: --$name"
		usage
		exit 1
	fi
}

require_arg fig1-cases-file "$FIG1_CASES_FILE"
require_arg fig1-prev-ckpt "$FIG1_PREV_CKPT"
require_arg fig1-ours-ckpt "$FIG1_OURS_CKPT"
require_arg fig2-image-dir "$FIG2_IMAGE_DIR"
require_arg fig2-mask-dir "$FIG2_MASK_DIR"
require_arg fig2-ckpt "$FIG2_CKPT"
require_arg fig3-ckpt "$FIG3_CKPT"
require_arg fig4-csv "$FIG4_CSV"
require_arg fig5-csv "$FIG5_CSV"
require_arg fig5-train-dataset "$FIG5_TRAIN_DATASET"
require_arg fig6-ckpt-without "$FIG6_CKPT_WITHOUT"
require_arg fig6-ckpt-with "$FIG6_CKPT_WITH"
require_arg fig7-ckpt "$FIG7_CKPT"
require_arg fig7-image-dir "$FIG7_IMAGE_DIR"
require_arg fig7-mask-dir "$FIG7_MASK_DIR"
require_arg fig8-csv "$FIG8_CSV"

if [[ ${#FIG3_IMAGES[@]} -eq 0 ]]; then
	echo "Missing required argument: --fig3-images"
	exit 1
fi

if [[ ${#FIG6_IMAGES[@]} -eq 0 ]]; then
	echo "Missing required argument: --fig6-images"
	exit 1
fi

dataset_args=()
if [[ ${#DATASETS[@]} -gt 0 ]]; then
	dataset_args=(--datasets "${DATASETS[@]}")
fi

python scripts/make_figure1_qualitative.py \
	"${dataset_args[@]}" \
	--cases-file "$FIG1_CASES_FILE" \
	--prev-checkpoint "$FIG1_PREV_CKPT" \
	--ours-checkpoint "$FIG1_OURS_CKPT" \
	--image-size "$IMAGE_SIZE" \
	--device "$DEVICE"

python scripts/make_figure2_boundary.py \
	"${dataset_args[@]}" \
	--image-dir "$FIG2_IMAGE_DIR" \
	--mask-dir "$FIG2_MASK_DIR" \
	--checkpoint "$FIG2_CKPT" \
	--image-size "$IMAGE_SIZE" \
	--device "$DEVICE" \
	--indices "$FIG2_INDICES"

python scripts/make_figure3_featuremaps.py \
	"${dataset_args[@]}" \
	--checkpoint "$FIG3_CKPT" \
	--images "${FIG3_IMAGES[@]}" \
	--image-size "$IMAGE_SIZE" \
	--device "$DEVICE"

python scripts/make_figure4_ablation.py \
	--csv "$FIG4_CSV"

python scripts/make_figure5_generalization.py \
	--csv "$FIG5_CSV" \
	--train-dataset "$FIG5_TRAIN_DATASET"

python scripts/make_figure6_freqaug.py \
	"${dataset_args[@]}" \
	--images "${FIG6_IMAGES[@]}" \
	--checkpoint-without "$FIG6_CKPT_WITHOUT" \
	--checkpoint-with "$FIG6_CKPT_WITH" \
	--image-size "$IMAGE_SIZE" \
	--device "$DEVICE"

python scripts/make_figure7_failure.py \
	"${dataset_args[@]}" \
	--checkpoint "$FIG7_CKPT" \
	--image-dir "$FIG7_IMAGE_DIR" \
	--mask-dir "$FIG7_MASK_DIR" \
	--image-size "$IMAGE_SIZE" \
	--device "$DEVICE" \
	--count "$FIG7_COUNT"

python scripts/make_figure8_table.py \
	--csv "$FIG8_CSV"

printf "All figures generated.\n"
