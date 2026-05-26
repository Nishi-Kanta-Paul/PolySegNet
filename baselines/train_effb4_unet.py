import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


VARIANTS = {
    "effb4_unet_baseline": {
        "experiment": "effb4_unet_baseline",
        "flags": [
            "--model-name",
            "bgdsf_polysegnet",
            "--no-use-msca",
            "--no-use-csaf",
            "--no-use-boundary-loss",
            "--no-use-dynamic-weighting",
            "--no-use-boundary-guidance",
            "--no-use-multilevel-boundary",
            "--no-use-freq-aug",
        ],
    },
    "effb4_unet_original_msca": {
        "experiment": "effb4_unet_original_msca",
        "flags": [
            "--model-name",
            "original_msca",
            "--use-msca",
            "--no-use-csaf",
            "--no-use-boundary-loss",
            "--no-use-dynamic-weighting",
            "--no-use-boundary-guidance",
            "--no-use-multilevel-boundary",
            "--no-use-freq-aug",
        ],
    },
    "effb4_unet_cmsca_static": {
        "experiment": "effb4_unet_cmsca_static",
        "flags": [
            "--model-name",
            "bgdsf_polysegnet",
            "--use-msca",
            "--no-use-csaf",
            "--no-use-boundary-loss",
            "--no-use-dynamic-weighting",
            "--no-use-boundary-guidance",
            "--no-use-multilevel-boundary",
            "--no-use-freq-aug",
        ],
    },
    "effb4_unet_bgd_cmsca": {
        "experiment": "effb4_unet_bgd_cmsca",
        "flags": [
            "--model-name",
            "bgdsf_polysegnet",
            "--use-msca",
            "--no-use-csaf",
            "--no-use-boundary-loss",
            "--use-dynamic-weighting",
            "--no-use-boundary-guidance",
            "--no-use-multilevel-boundary",
            "--no-use-freq-aug",
        ],
    },
    "bgdsf_cmsca_sagf_plain": {
        "experiment": "bgdsf_cmsca_sagf_plain",
        "flags": [
            "--model-name",
            "bgdsf_polysegnet",
            "--use-msca",
            "--use-csaf",
            "--no-use-boundary-loss",
            "--use-dynamic-weighting",
            "--no-use-boundary-guidance",
            "--no-use-multilevel-boundary",
            "--no-use-freq-aug",
        ],
    },
    "bgdsf_cmsca_bgsagf": {
        "experiment": "bgdsf_cmsca_bgsagf",
        "flags": [
            "--model-name",
            "bgdsf_polysegnet",
            "--use-msca",
            "--use-csaf",
            "--no-use-boundary-loss",
            "--use-dynamic-weighting",
            "--use-boundary-guidance",
            "--no-use-multilevel-boundary",
            "--no-use-freq-aug",
        ],
    },
    "bgdsf_polysegnet_no_freqaug": {
        "experiment": "bgdsf_polysegnet_no_freqaug",
        "flags": [
            "--model-name",
            "bgdsf_polysegnet",
            "--use-msca",
            "--use-csaf",
            "--use-boundary-loss",
            "--use-dynamic-weighting",
            "--use-boundary-guidance",
            "--use-multilevel-boundary",
            "--no-use-freq-aug",
        ],
    },
    "bgdsf_polysegnet_full": {
        "experiment": "bgdsf_polysegnet_full",
        "flags": [
            "--model-name",
            "bgdsf_polysegnet",
            "--use-msca",
            "--use-csaf",
            "--use-boundary-loss",
            "--use-dynamic-weighting",
            "--use-boundary-guidance",
            "--use-multilevel-boundary",
            "--use-freq-aug",
        ],
    },
}


def _run_variant(variant_key: str, extra_args: list[str]) -> None:
    variant = VARIANTS[variant_key]
    cmd = [
        sys.executable,
        os.path.join(ROOT, "src", "main.py"),
        "--mode",
        "train",
        "--experiment-name",
        variant["experiment"],
    ]
    cmd.extend(variant["flags"])
    cmd.extend(extra_args)
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="EfficientNet-B4 ablations")
    parser.add_argument(
        "--variant",
        choices=sorted(VARIANTS.keys()),
        default="effb4_unet_baseline",
    )
    args, extra_args = parser.parse_known_args()
    _run_variant(args.variant, extra_args)


if __name__ == "__main__":
    main()
