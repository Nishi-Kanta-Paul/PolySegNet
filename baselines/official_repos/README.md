# Official Baselines

This folder is reserved for official baseline repositories that are evaluated with the same
metrics used in this project. The code below does not copy or rewrite the original
implementations.

## PraNet (pending)

Repo: https://github.com/DengPingFan/PraNet

Clone:

```bash
git clone https://github.com/DengPingFan/PraNet baselines/official_repos/pranet
```

Train / infer: follow the official repo instructions to produce prediction masks for the
same test split used here.

Evaluate with our metrics:

```bash
python scripts/evaluate_baseline_predictions.py \
  --method-name PraNet \
  --dataset-root data/Kvasir-SEG \
  --image-dir images \
  --mask-dir masks \
  --pred-dir /path/to/pranet/predictions \
  --output-dir outputs/official_baselines/pranet
```

## Polyp-PVT (pending)

Repo: https://github.com/DengPingFan/Polyp-PVT

Clone:

```bash
git clone https://github.com/DengPingFan/Polyp-PVT baselines/official_repos/polyp_pvt
```

Train / infer: follow the official repo instructions to produce prediction masks for the
same test split used here.

Evaluate with our metrics:

```bash
python scripts/evaluate_baseline_predictions.py \
  --method-name "Polyp-PVT" \
  --dataset-root data/Kvasir-SEG \
  --image-dir images \
  --mask-dir masks \
  --pred-dir /path/to/polyp-pvt/predictions \
  --output-dir outputs/official_baselines/polyp_pvt
```
