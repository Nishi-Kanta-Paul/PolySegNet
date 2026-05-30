#!/usr/bin/env bash
set -euo pipefail

EXPERIMENTS=(
  effb4_unet_baseline
  effb4_unet_original_msca
  effb4_unet_cmsca_static
  effb4_unet_bgd_cmsca
  bgdsf_cmsca_sagf_plain
  bgdsf_cmsca_bgsagf
  bgdsf_polysegnet_no_freqaug
  bgdsf_polysegnet_full
)

missing_ckpt=()

for exp in "${EXPERIMENTS[@]}"; do
  ckpt="experiments/${exp}/checkpoints/best.pth"
  if [[ ! -f "$ckpt" ]]; then
    echo "Warning: best checkpoint missing for ${exp} (${ckpt})."
    missing_ckpt+=("$exp")
    continue
  fi

  echo "Evaluating ${exp} using ${ckpt}"
  python src/main.py --mode eval --experiment-name "$exp" --checkpoint "$ckpt" "$@"
done

if (( ${#missing_ckpt[@]} > 0 )); then
  echo "Missing best.pth for: ${missing_ckpt[*]}"
fi
