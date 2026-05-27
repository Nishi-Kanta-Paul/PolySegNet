import os
import sys
from types import SimpleNamespace

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.evaluate import CompositeLoss, calculate_metrics


def main() -> None:
    torch.manual_seed(42)
    logits = torch.randn(2, 1, 352, 352)
    masks = torch.randint(0, 2, (2, 1, 352, 352)).float()

    cfg = SimpleNamespace(
        loss_weights={"bce": 1.0, "dice": 1.0, "boundary": 1.0},
        aux_boundary_weight=0.2,
        boundary_kernel_size=3,
        use_boundary_loss=True,
        use_multilevel_boundary=True,
    )
    criterion = CompositeLoss(cfg)

    outputs = {
        "mask_logits": logits,
        "boundary_logits": torch.randn(2, 1, 352, 352),
        "aux_boundary_logits": [
            torch.randn(2, 1, 176, 176),
            torch.randn(2, 1, 88, 88),
        ],
    }

    loss, loss_dict = criterion(outputs, masks)
    metrics = calculate_metrics(outputs, masks)

    print(f"Composite loss: {loss.item():.6f}")
    print(f"Loss breakdown: {loss_dict}")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")

    assert torch.isfinite(loss).item()
    for value in metrics.values():
        assert value == value


if __name__ == "__main__":
    main()
