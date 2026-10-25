"""Train an encoder model from a YAML config and evaluate it on dev.

Usage: uv run python scripts/train.py --config configs/b_distilroberta.yaml [--seed 2]
       [--device mps] [--purpose encoder_check]

Only train and dev are loaded; the test split is used once, in M6.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from entsent.data import REPO_ROOT

# Keep downloaded models inside the project (gitignored), before transformers is imported.
os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".cache" / "hf"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from entsent.config import TrainConfig, load_config
from entsent.data import load_split
from entsent.experiment import run_training
from entsent.training import select_device


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, help="override the config seed")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--purpose", default="eval")
    args = parser.parse_args()

    cfg = load_config(args.config)
    overrides = {k: v for k, v in (("seed", args.seed), ("device", args.device)) if v is not None}
    if overrides:
        cfg = TrainConfig.model_validate({**cfg.model_dump(), **overrides})
    device = select_device(cfg.device)
    print(f"{cfg.name} seed {cfg.seed} on {device.type}")

    path = run_training(
        cfg,
        load_split("train"),
        load_split("dev"),
        device=device,
        purpose=args.purpose,
        config_path=str(args.config),
    )
    print(f"wrote {path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
