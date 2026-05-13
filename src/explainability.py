import os
import sys

try:
    from .config import Config
except ImportError:
    ROOT = os.path.dirname(os.path.abspath(__file__))
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from config import Config


def run_explainability(cfg: Config) -> None:
    raise NotImplementedError("Explainability tooling not implemented yet.")
