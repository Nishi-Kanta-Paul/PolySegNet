try:
    from .config import build_arg_parser, config_from_args
    from .evaluate import evaluate
    from .explainability import run_explainability
    from .inference import run_inference
    from .train import train
except ImportError:
    import os
    import sys

    ROOT = os.path.dirname(os.path.abspath(__file__))
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)

    from config import build_arg_parser, config_from_args
    from evaluate import evaluate
    from explainability import run_explainability
    from inference import run_inference
    from train import train


def main() -> None:
    parser = build_arg_parser()
    parser.description = "PolySegNet runner"
    parser.add_argument("--mode", choices=["train", "eval", "infer", "explain"], required=True)
    args = parser.parse_args()

    cfg = config_from_args(args)

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
