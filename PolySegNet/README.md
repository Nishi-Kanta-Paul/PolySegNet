# PolySegNet

PolySegNet: A Hybrid Encoder-Decoder Architecture with Attention-Guided Feature Fusion for Robust Polyp Segmentation in Colonoscopy Images.

## Scope
- PyTorch-based binary polyp segmentation
- Supports Kvasir-SEG, CVC-ClinicDB, CVC-ColonDB, and custom datasets

## Quickstart
1. Create a virtual environment
2. Install dependencies: pip install -r requirements.txt
3. Put datasets under data/ (kept out of Git)
4. Run training: python -m src.main --mode train

## Notes
- Configuration is loaded from YAML via src/config.py (optional).
- Debug mode can be enabled in config to run without real data.
