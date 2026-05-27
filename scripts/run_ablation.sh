#!/usr/bin/env bash
set -euo pipefail

# 1) EfficientNet-B4 U-Net baseline
python src/main.py --mode train --experiment-name effb4_unet_baseline \
	--model-name bgdsf_polysegnet --no-use-msca --no-use-csaf --no-use-mbgh --no-use-boundary-loss \
	--no-use-dynamic-weighting --no-use-boundary-guidance --no-use-multilevel-boundary \
	--no-use-freq-aug "$@"

# 2) + Original MSCA (parallel dilations)
python src/main.py --mode train --experiment-name effb4_unet_original_msca \
	--model-name original_msca --use-msca --no-use-csaf --no-use-mbgh --no-use-boundary-loss \
	--no-use-dynamic-weighting --no-use-boundary-guidance --no-use-multilevel-boundary \
	--no-use-freq-aug "$@"

# 3) + Cascaded CMSCA without dynamic weighting
python src/main.py --mode train --experiment-name effb4_unet_cmsca_static \
	--model-name bgdsf_polysegnet --use-msca --no-use-csaf --no-use-mbgh --no-use-boundary-loss \
	--no-use-dynamic-weighting --no-use-boundary-guidance --no-use-multilevel-boundary \
	--no-use-freq-aug "$@"

# 4) + Full BGD-CMSCA with dynamic weighting
python src/main.py --mode train --experiment-name effb4_unet_bgd_cmsca \
	--model-name bgdsf_polysegnet --use-msca --no-use-csaf --no-use-mbgh --no-use-boundary-loss \
	--use-dynamic-weighting --no-use-boundary-guidance --no-use-multilevel-boundary \
	--no-use-freq-aug "$@"

# 5) + plain SAGF skip fusion (no boundary guidance)
python src/main.py --mode train --experiment-name bgdsf_cmsca_sagf_plain \
	--model-name bgdsf_polysegnet --use-msca --use-csaf --no-use-mbgh --no-use-boundary-loss \
	--use-dynamic-weighting --no-use-boundary-guidance --no-use-multilevel-boundary \
	--no-use-freq-aug "$@"

# 6) + BG-SAGF with boundary guidance
python src/main.py --mode train --experiment-name bgdsf_cmsca_bgsagf \
	--model-name bgdsf_polysegnet --use-msca --use-csaf --no-use-mbgh --no-use-boundary-loss \
	--use-dynamic-weighting --use-boundary-guidance --no-use-multilevel-boundary \
	--no-use-freq-aug "$@"

# 7) + MBGH (multi-level boundary head)
python src/main.py --mode train --experiment-name bgdsf_polysegnet_no_freqaug \
	--model-name bgdsf_polysegnet --use-msca --use-csaf --use-mbgh --use-boundary-loss \
	--use-dynamic-weighting --use-boundary-guidance --use-multilevel-boundary \
	--no-use-freq-aug "$@"

# 8) Full BGD-SF PolySegNet with frequency augmentation
python src/main.py --mode train --experiment-name bgdsf_polysegnet_full \
	--model-name bgdsf_polysegnet --use-msca --use-csaf --use-mbgh --use-boundary-loss \
	--use-dynamic-weighting --use-boundary-guidance --use-multilevel-boundary \
	--use-freq-aug "$@"

python baselines/compare_results.py
