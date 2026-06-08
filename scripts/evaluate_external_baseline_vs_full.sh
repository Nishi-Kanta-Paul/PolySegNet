#!/usr/bin/env bash
# Evaluate EffB4 baseline and Full BGD-SF on external datasets.
#
# Usage:
#   ./scripts/evaluate_external_baseline_vs_full.sh \
#       /data/CVC-ClinicDB \
#       /data/CVC-ColonDB \
#       /data/ETIS-Larib \
#       /data/CVC-300
#
# Optional environment overrides:
#   BASELINE_EXP   experiment name for the baseline  (default: effb4_unet_baseline)
#   FULL_EXP       experiment name for full BGD-SF   (default: bgdsf_polysegnet_full)
#   DEVICE         pytorch device                    (default: cuda)
#   BATCH_SIZE     evaluation batch size             (default: 8)
#
# Each experiment is evaluated once with all dataset roots passed together,
# producing per-dataset subdirectories under:
#   experiments/<exp>/evaluation/<dataset_name>/
#
# Checkpoint selection: best.pth only (selected by validation Dice during training).
# This script does NOT train anything and does NOT use test data for selection.

set -euo pipefail

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
BASELINE_EXP="${BASELINE_EXP:-effb4_unet_baseline}"
FULL_EXP="${FULL_EXP:-bgdsf_polysegnet_full}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-8}"

EXPERIMENTS=("$BASELINE_EXP" "$FULL_EXP")

# --------------------------------------------------------------------------
# Validate arguments
# --------------------------------------------------------------------------
if [[ $# -eq 0 ]]; then
    echo "ERROR: No dataset roots provided."
    echo ""
    echo "Usage: $0 <dataset_root1> [dataset_root2] ..."
    echo ""
    echo "Example:"
    echo "  $0 /data/CVC-ClinicDB /data/CVC-ColonDB /data/ETIS-Larib /data/CVC-300"
    exit 1
fi

DATASET_ROOTS=("$@")

echo "============================================================"
echo " External baseline-vs-full evaluation"
echo "============================================================"
echo " Baseline : $BASELINE_EXP"
echo " Full     : $FULL_EXP"
echo " Device   : $DEVICE"
echo " Datasets : ${DATASET_ROOTS[*]}"
echo "============================================================"
echo ""

# --------------------------------------------------------------------------
# Helper: extract model config flags from checkpoint
# --------------------------------------------------------------------------
extract_cfg_args() {
    local ckpt_path="$1"
    python - "$ckpt_path" <<'PY'
import sys
import torch

ckpt_path = sys.argv[1]
checkpoint = torch.load(ckpt_path, map_location="cpu")
cfg = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}

keys = [
    "model_name",
    "bgdsf_encoder_name",
    "pretrained",
    "unified_channels",
    "use_msca",
    "use_csaf",
    "use_mbgh",
    "use_boundary_loss",
    "use_dynamic_weighting",
    "use_boundary_guidance",
    "use_multilevel_boundary",
    "use_freq_aug",
    "boundary_kernel_size",
    "aux_boundary_weight",
]

args = []
for key in keys:
    if key not in cfg:
        continue
    value = cfg[key]
    if value is None:
        continue
    flag = key.replace("_", "-")
    if isinstance(value, bool):
        prefix = "" if value else "no-"
        args.append(f"--{prefix}{flag}")
    else:
        args.append(f"--{flag}={value}")

for arg in args:
    print(arg)
PY
}

# --------------------------------------------------------------------------
# Main loop: evaluate each experiment on all external datasets at once
# --------------------------------------------------------------------------
missing_ckpt=()

for EXP in "${EXPERIMENTS[@]}"; do
    CKPT="experiments/${EXP}/checkpoints/best.pth"

    if [[ ! -f "$CKPT" ]]; then
        echo "WARNING: best.pth missing for '${EXP}' (expected: ${CKPT})."
        echo "         Skipping. Run training or provide the checkpoint before re-running."
        missing_ckpt+=("$EXP")
        echo ""
        continue
    fi

    echo "------------------------------------------------------------"
    echo " Evaluating: $EXP"
    echo " Checkpoint: $CKPT"
    echo " Datasets  : ${DATASET_ROOTS[*]}"
    echo "------------------------------------------------------------"

    # Extract model-specific config flags saved inside the checkpoint
    mapfile -t CFG_ARGS < <(extract_cfg_args "$CKPT")

    python src/main.py \
        --mode eval \
        --experiment-name "$EXP" \
        --checkpoint "$CKPT" \
        --dataset-roots "${DATASET_ROOTS[@]}" \
        --device "$DEVICE" \
        --batch-size "$BATCH_SIZE" \
        "${CFG_ARGS[@]}"

    echo ""
    echo " Done: $EXP"
    echo ""
done

# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
echo "============================================================"
echo " Evaluation complete."
echo "============================================================"

if (( ${#missing_ckpt[@]} > 0 )); then
    echo ""
    echo "WARNING: The following experiments had no best.pth and were skipped:"
    for EXP in "${missing_ckpt[@]}"; do
        echo "  - $EXP  (expected: experiments/${EXP}/checkpoints/best.pth)"
    done
fi

echo ""
echo "Output locations:"
for EXP in "${EXPERIMENTS[@]}"; do
    echo "  experiments/${EXP}/evaluation/<dataset_name>/results.json"
    echo "  experiments/${EXP}/evaluation/<dataset_name>/results.csv"
done
echo ""
echo "Next step — generate diagnostic tables:"
echo "  python scripts/analyze_external_baseline_vs_full.py \\"
echo "      --datasets $(IFS=' '; echo "${DATASET_ROOTS[*]}" | \
        xargs -n1 basename | tr '\n' ' ')"
