# PolySegNet

PolySegNet: A Hybrid Encoder-Decoder Architecture with Attention-Guided Feature Fusion for Robust Polyp Segmentation in Colonoscopy Images.

## Architecture: BGD-SF PolySegNet

- Encoder: EfficientNet-B4 pretrained on ImageNet
- Bottleneck: BGD-CMSCA — Boundary-Guided Dynamic Cascaded Multi-Scale
	Context Aggregation with internal boundary-sensitive dynamic scale weighting
- Skip Fusion: BG-SAGF — Boundary-Guided Selective Attention Fusion
	with internally generated boundary prior at each decoder level
- Boundary Head: MBGH — Multi-Level Boundary Guidance Head
	with auxiliary supervision at D2, D3, D4
- Final Mask: Boundary-refined via D1_bg = D1 + D1 * sigmoid(boundary_logits)
- Loss: BCE + Soft Dice + Boundary Dice + Auxiliary Boundary + Multi-Level Boundary
- Optional: Frequency-domain style augmentation for cross-dataset robustness

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

## Training Commands

```bash
# Full model
python src/main.py --mode train \
	--experiment-name bgdsf_polysegnet_full

# Debug mode
python src/main.py --mode train --debug

# Run all 8 ablation variants
bash scripts/run_ablation.sh
```

## Inference

Single image:

```bash
python src/main.py --mode infer \
	--checkpoint experiments/bgdsf_polysegnet_full/checkpoints/best.pth \
	--image path/to/image.png
```

Generate ablation comparison table:

```bash
python baselines/compare_results.py
```

## Ablation Table

| Variant | CMSCA | BG-SAGF | MBGH | FreqAug | Experiment Name |
| --- | --- | --- | --- | --- | --- |
| EfficientNet-B4 U-Net Baseline | – | – | – | – | effb4_unet_baseline |
| + Original MSCA (parallel) | – | – | – | – | effb4_unet_original_msca |
| + Cascaded CMSCA (static) | ✓ | – | – | – | effb4_unet_cmsca_static |
| + BGD-CMSCA (dynamic) | ✓ | – | – | – | effb4_unet_bgd_cmsca |
| + SAGF (plain) | ✓ | – | – | – | bgdsf_cmsca_sagf_plain |
| + BG-SAGF | ✓ | ✓ | – | – | bgdsf_cmsca_bgsagf |
| + MBGH | ✓ | ✓ | ✓ | – | bgdsf_polysegnet_no_freqaug |
| Full BGD-SF (FreqAug) | ✓ | ✓ | ✓ | ✓ | bgdsf_polysegnet_full |

## Experiments vs Outputs

experiments/<name>/checkpoints/   — model checkpoints
experiments/<name>/logs/          — train/val loss and metric logs
experiments/<name>/logs/visuals/  — validation mask visualizations
experiments/<name>/logs/visuals/boundary/ — boundary heatmaps
experiments/<name>/inference/     — inference outputs (_mask, _prob, _boundary, _overlay)
experiments/<name>/explainability/boundary_heatmaps/ — Grad-CAM or boundary heatmaps
outputs/tables/                   — final paper-ready comparison tables
outputs/figures/                  — final paper-ready qualitative figures

## Notes

- Configuration can be overridden via CLI or YAML (src/config.py).
- Debug mode runs with synthetic data.
- Pretrained EfficientNet weights download only when pretrained=True and cache is missing.

## Statistical Testing

- Wilcoxon signed-rank test is used because comparisons are paired per image.
- p < 0.05 is treated as statistically significant.
- Interpret p-values alongside effect size rather than alone.

## Multi-Seed Experiments

- Multi-seed experiments test training stability.
- Reported mean ± standard deviation is computed across three independent runs with seeds 42, 123, and 2025.

## Citation / Notes

If you use this code in your research, please cite the original dataset sources and mention PolySegNet as the segmentation framework.
