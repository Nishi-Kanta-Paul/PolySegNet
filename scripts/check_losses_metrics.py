import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.evaluate import CompositeLoss, calculate_metrics


def main() -> None:
    torch.manual_seed(42)
    logits = torch.randn(2, 1, 352, 352)
    masks = torch.randint(0, 2, (2, 1, 352, 352)).float()

    criterion = CompositeLoss(
        bce_weight=1.0,
        dice_weight=1.0,
        boundary_weight=1.0,
        use_boundary_loss=True,
    )
    loss = criterion(logits, masks)

    metrics = calculate_metrics(logits, masks)

    print(f"Composite loss: {loss.item():.6f}")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")

    assert torch.isfinite(loss).item()
    for value in metrics.values():
        assert value == value


if __name__ == "__main__":
    main()
