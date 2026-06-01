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

  mapfile -t cfg_args < <(
    python - "$ckpt" <<'PY'
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
    value = cfg.get(key)
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
  )

  echo "Evaluating ${exp} using ${ckpt}"
  python src/main.py --mode eval --experiment-name "$exp" --checkpoint "$ckpt" \
    "${cfg_args[@]}" "$@"
done

if (( ${#missing_ckpt[@]} > 0 )); then
  echo "Missing best.pth for: ${missing_ckpt[*]}"
fi
