# PolySegNet

PolySegNet: A Hybrid Encoder-Decoder Architecture with Attention-Guided Feature Fusion for Robust Polyp Segmentation in Colonoscopy Images.

## Overview

This project implements a PyTorch-based binary polyp segmentation system with:
- EfficientNet-B4 encoder (timm)
- U-Net-style decoder
- MSCA bottleneck for multi-scale context aggregation
- CSAF for attention-guided skip fusion
- Composite loss (BCE + Soft Dice + optional boundary loss)

## Folder Structure

```
PolySegNet/
├── baselines/                  # Baselines and ablations
├── data/                       # Datasets (ignored by Git)
├── experiments/                # Training outputs (ignored by Git)
├── outputs/                    # Final paper-ready outputs only
├── notebooks/
├── scripts/
├── src/
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset Preparation

Expected dataset format:

```
data/<DatasetName>/
	images/
		xxx.png
	masks/
		xxx.png
```

Optional split files (if missing, deterministic splits are created):

```
data/<DatasetName>/
	train.txt
	val.txt
	test.txt
```

## Training

```bash
python src/main.py --mode train --dataset-root data/<DatasetName> --image-dir images --mask-dir masks
```

Debug training (no real data needed):

```bash
python src/main.py --mode train --debug
```

## Evaluation

```bash
python src/main.py --mode eval --dataset-root data/<DatasetName> --image-dir images --mask-dir masks
```

## Inference

Single image:

```bash
python src/main.py --mode infer --checkpoint experiments/polysegnet/checkpoints/best.pth --image path/to/image.png
```

Folder:

```bash
python src/main.py --mode infer --checkpoint experiments/polysegnet/checkpoints/best.pth --image_dir path/to/images
```

Inference outputs are saved to:

```
experiments/<experiment_name>/inference/
```

## Baselines and Ablations

```bash
python baselines/train_unet.py
python baselines/train_effb4_unet.py --variant effb4_unet
python baselines/train_effb4_unet.py --variant effb4_unet_msca
python baselines/train_effb4_unet.py --variant effb4_unet_msca_csaf
python baselines/train_effb4_unet.py --variant polysegnet_full
```

Run all ablations:

```bash
bash scripts/run_ablation.sh
```

Compare results:

```bash
python baselines/compare_results.py
```

Comparison tables are saved to:

```
outputs/tables/
```

## Experiments vs Outputs

- experiments/ contains raw runs: checkpoints, logs, visualizations, and results.json.
- outputs/ is only for final paper-ready figures and tables.
- Do not place checkpoints inside outputs/.

## Notes

- Configuration can be overridden via CLI or YAML (src/config.py).
- Debug mode runs with synthetic data.
- Pretrained EfficientNet weights download only when pretrained=True and cache is missing.

## Citation / Notes

If you use this code in your research, please cite the original dataset sources and mention PolySegNet as the segmentation framework.
