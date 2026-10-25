"""Training configs: YAML files in configs/, validated with pydantic so typos fail loudly."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from entsent.data import REPO_ROOT

CONFIG_DIR = REPO_ROOT / "configs"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EncoderConfig(_Strict):
    name: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")  # pinned Hub commit


class OptimConfig(_Strict):
    epochs: int = Field(ge=1, le=50)
    batch_size: int = Field(ge=1, le=1024)
    lr: float = Field(gt=0, lt=1e-2)
    weight_decay: float = Field(ge=0, le=1)
    warmup_ratio: float = Field(ge=0, lt=1)
    max_grad_norm: float = Field(gt=0)
    dropout: float = Field(ge=0, lt=1)


class TrainConfig(_Strict):
    name: str = Field(pattern=r"^[a-z0-9_]+$")
    architecture: Literal["per_entity", "single_pass"]
    encoder: EncoderConfig
    max_length: int = Field(ge=8, le=512)
    optim: OptimConfig
    seed: int = Field(ge=0)
    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    save_checkpoint: bool = False


def load_config(path: Path) -> TrainConfig:
    return TrainConfig.model_validate(yaml.safe_load(path.read_text()))
