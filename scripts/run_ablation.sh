#!/usr/bin/env bash
set -euo pipefail

python baselines/train_unet.py "$@"
python baselines/train_effb4_unet.py --variant effb4_unet "$@"
python baselines/train_effb4_unet.py --variant effb4_unet_msca "$@"
python baselines/train_effb4_unet.py --variant effb4_unet_msca_csaf "$@"
python baselines/train_effb4_unet.py --variant polysegnet_full "$@"

python baselines/compare_results.py
