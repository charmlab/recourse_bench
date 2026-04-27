from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
if ROOT.as_posix() not in sys.path:
    sys.path.insert(0, ROOT.as_posix())

from experiment import Experiment


def resolve_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise TypeError("Smoke config must parse to a dictionary")
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--path",
        default=Path(__file__)
        .with_name("credit_linear_claproar_smoke.yaml")
        .as_posix(),
    )
    args = parser.parse_args()

    config_path = Path(args.path)
    config = load_config(config_path)
    device = resolve_device()
    config["model"]["device"] = device
    config["method"]["device"] = device

    metrics = Experiment(config).run()
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
