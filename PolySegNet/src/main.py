import argparse

from .config import load_config
from .evaluate import evaluate
from .explainability import run_explainability
from .inference import run_inference
from .train import train


def main() -> None:
    parser = argparse.ArgumentParser(description="PolySegNet runner")
    parser.add_argument("--mode", choices=["train", "eval", "infer", "explain"], required=True)
    parser.add_argument("--config", default=None, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.mode == "train":
        train(cfg)
    elif args.mode == "eval":
        evaluate(cfg)
    elif args.mode == "infer":
        run_inference(cfg)
    else:
        run_explainability(cfg)


if __name__ == "__main__":
    main()
