#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time
from typing import Dict, List

import torch

from src.config import load_config
from src.model import build_model
from src.utils import count_parameters, ensure_dir, save_json


def _safe_device_name(device: torch.device) -> str:
    if device.type == "cuda" and torch.cuda.is_available():
        index = device.index if device.index is not None else torch.cuda.current_device()
        return torch.cuda.get_device_name(index)
    return "cpu"


def _extract_output(outputs):
    if isinstance(outputs, dict):
        if "mask_logits" in outputs:
            return outputs["mask_logits"]
        if "out" in outputs:
            return outputs["out"]
        return outputs
    return outputs


def _try_flops(model: torch.nn.Module, dummy: torch.Tensor) -> float | None:
    try:
        from thop import profile
    except ImportError:
        print("Warning: thop is not installed. FLOPs will be blank.")
        return None

    try:
        flops, _params = profile(model, inputs=(dummy,), verbose=False)
        return float(flops) / 1e9
    except Exception as exc:
        print(f"Warning: FLOPs calculation failed: {exc}")
        return None


def _measure_speed(
    model: torch.nn.Module,
    dummy: torch.Tensor,
    device: torch.device,
    warmup: int,
    iters: int,
) -> tuple[float | None, float | None]:
    if iters <= 0:
        return None, None

    model.eval()
    with torch.no_grad():
        for _ in range(max(warmup, 0)):
            _ = _extract_output(model(dummy))
        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(iters):
            _ = _extract_output(model(dummy))
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()

    total = end - start
    ms_per_image = (total / iters) * 1000.0 if iters > 0 else None
    fps = 1000.0 / ms_per_image if ms_per_image and ms_per_image > 0 else None
    return fps, ms_per_image


def _format_float(value: float | None, decimals: int = 6) -> str:
    if value is None:
        return ""
    return f"{value:.{decimals}f}"


def _format_params(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute model complexity for baselines.")
    parser.add_argument("--configs", nargs="+", required=True, help="List of config paths.")
    parser.add_argument("--image-size", type=int, default=352)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument(
        "--output-csv",
        default="outputs/tables/model_complexity.csv",
    )
    parser.add_argument(
        "--output-json",
        default="outputs/tables/model_complexity.json",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available. Falling back to CPU.")
        device = torch.device("cpu")

    rows: List[Dict[str, str]] = []
    details: List[Dict[str, object]] = []

    for config_path in args.configs:
        cfg = load_config(config_path)
        cfg.image_size = args.image_size
        cfg.device = str(device)

        method_name = cfg.experiment_name or os.path.splitext(os.path.basename(config_path))[0]
        model_name = cfg.model_name

        record = {
            "method_name": method_name,
            "model_name": model_name,
            "config_path": config_path,
            "input_size": f"{args.image_size}x{args.image_size}",
            "params_m": "",
            "flops_g": "",
            "fps": "",
            "ms_per_image": "",
            "device": str(device),
            "gpu_name": _safe_device_name(device),
        }

        try:
            model = build_model(cfg).to(device)
        except Exception as exc:
            print(f"Warning: failed to build {model_name} from {config_path}: {exc}")
            rows.append(record)
            details.append(record)
            continue

        params_m = count_parameters(model) / 1e6
        dummy = torch.randn(1, 3, args.image_size, args.image_size, device=device)

        flops_g = _try_flops(model, dummy)
        fps, ms_per_image = _measure_speed(
            model,
            dummy,
            device,
            warmup=args.warmup,
            iters=args.iters,
        )

        record["params_m"] = _format_params(params_m)
        record["flops_g"] = _format_float(flops_g, 4)
        record["fps"] = _format_float(fps, 2)
        record["ms_per_image"] = _format_float(ms_per_image, 3)

        rows.append(record)
        details.append(
            {
                **record,
                "params_m": params_m,
                "flops_g": flops_g,
                "fps": fps,
                "ms_per_image": ms_per_image,
            }
        )

    output_csv = args.output_csv
    ensure_dir(os.path.dirname(output_csv))
    fieldnames = [
        "method_name",
        "model_name",
        "config_path",
        "input_size",
        "params_m",
        "flops_g",
        "fps",
        "ms_per_image",
        "device",
        "gpu_name",
    ]
    with open(output_csv, "w", encoding="utf-8", newline="") as handle:
        handle.write(",".join(fieldnames) + "\n"
        )
        for row in rows:
            handle.write(",".join(str(row.get(name, "")) for name in fieldnames) + "\n"
            )

    output_json = args.output_json
    ensure_dir(os.path.dirname(output_json))
    save_json(output_json, {"results": details})

    print(f"Saved CSV to {output_csv}")
    print(f"Saved JSON to {output_json}")


if __name__ == "__main__":
    main()
