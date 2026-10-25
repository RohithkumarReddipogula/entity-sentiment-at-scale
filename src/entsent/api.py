"""M7: FastAPI service for the chosen model (C seed 1): POST /predict and GET /health.

Configuration (environment):
  MODEL_DIR      checkpoint folder (model.pt, tokenizer/, encoder/, config.json). Required.
  MODEL_SHA256   expected SHA-256 of model.pt; if set, the server refuses mismatching weights.
  TORCH_THREADS  torch threads per worker. Default 1, which M5 found most efficient on macOS
                 (Accelerate BLAS); the Docker image sets 4, measured faster on Linux arm64.

Request bodies are never logged: headlines may be sensitive.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator

from entsent.data import LABELS, sha256_file
from entsent.single_pass import HeadlineInstance, collate_single_pass

logger = logging.getLogger("entsent.api")

MAX_TEXT_CHARS = 1000
MAX_ENTITIES = 20
MAX_MENTIONS = 10

Span = Annotated[list[int], Field(min_length=2, max_length=2)]


class EntityIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str | None = Field(default=None, max_length=200)
    spans: list[Span] = Field(min_length=1, max_length=MAX_MENTIONS)


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)
    entities: list[EntityIn] = Field(min_length=1, max_length=MAX_ENTITIES)

    @model_validator(mode="after")
    def check_spans(self) -> PredictRequest:
        if not self.text.strip():
            raise ValueError("text is empty")
        owned: list[tuple[int, int, int]] = []
        for i, ent in enumerate(self.entities):
            for start, end in ent.spans:
                if not 0 <= start < end <= len(self.text):
                    raise ValueError(
                        f"entity {i}: span [{start}, {end}) is outside the text "
                        f"(length {len(self.text)}) or empty"
                    )
                if not self.text[start:end].strip():
                    raise ValueError(f"entity {i}: span [{start}, {end}) is only whitespace")
                owned.append((start, end, i))
        for n, (start_a, end_a, a) in enumerate(owned):  # at most 200 spans: check every pair
            for start_b, end_b, b in owned[n + 1 :]:
                if start_a < end_b and start_b < end_a:
                    raise ValueError(f"spans of entities {a} and {b} overlap")
        return self


class EntityOut(BaseModel):
    name: str | None
    spans: list[list[int]]
    label: str
    confidence: float
    probabilities: dict[str, float]


class ModelInfo(BaseModel):
    name: str
    architecture: str
    sha256: str


class PredictResponse(BaseModel):
    model: ModelInfo
    entities: list[EntityOut]


@dataclass
class LoadedModel:
    model: Any
    tokenizer: Any
    max_length: int
    info: ModelInfo


def load_model(model_dir: Path, expected_sha256: str | None) -> LoadedModel:
    from entsent.experiment import load_checkpoint

    if not (model_dir / "model.pt").exists():
        raise RuntimeError(f"no model.pt in MODEL_DIR {model_dir}")
    digest = sha256_file(model_dir / "model.pt")
    if expected_sha256 and digest != expected_sha256:
        raise RuntimeError(f"model.pt sha256 {digest} does not match MODEL_SHA256")
    model, tokenizer, cfg = load_checkpoint(model_dir)
    if cfg.architecture != "single_pass":
        raise RuntimeError(f"the API serves model C (single_pass), got {cfg.architecture}")
    # Identity comes from the checkpoint itself, not from where it is mounted (e.g. /model).
    info = ModelInfo(name=f"{cfg.name}_s{cfg.seed}", architecture=cfg.architecture, sha256=digest)
    return LoadedModel(model.eval(), tokenizer, cfg.max_length, info)


def predict(loaded: LoadedModel, request: PredictRequest) -> PredictResponse:
    instance = HeadlineInstance(
        headline_id=0,
        text=request.text,
        labels=tuple(0 for _ in request.entities),  # unused at inference
        spans=tuple(tuple((s, e) for s, e in ent.spans) for ent in request.entities),
    )
    try:
        batch = collate_single_pass([instance], loaded.tokenizer, loaded.max_length)
    except ValueError as exc:  # too many tokens, or a mention that covers no token
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    inputs = {k: v for k, v in batch.items() if k != "labels"}
    with torch.inference_mode():
        probs = torch.softmax(loaded.model(**inputs), dim=-1).tolist()
    entities = []
    for ent, p in zip(request.entities, probs, strict=True):
        best = max(range(len(LABELS)), key=lambda i: p[i])
        entities.append(
            EntityOut(
                name=ent.name,
                spans=[list(s) for s in ent.spans],
                label=LABELS[best],
                confidence=p[best],
                probabilities=dict(zip(LABELS, p, strict=True)),
            )
        )
    return PredictResponse(model=loaded.info, entities=entities)


def create_app(model_dir: Path | None = None, expected_sha256: str | None = None) -> FastAPI:
    state: dict[str, Any] = {"model": None, "error": None}

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "1")))
        directory = model_dir or Path(os.environ.get("MODEL_DIR", ""))
        sha = expected_sha256 or os.environ.get("MODEL_SHA256") or None
        try:
            if not str(directory):
                raise RuntimeError("MODEL_DIR is not set")
            state["model"] = load_model(directory, sha)
            logger.info("model loaded: %s", state["model"].info.name)
        except Exception as exc:  # report through /health instead of crashing silently
            state["error"] = str(exc)
            logger.error("model failed to load: %s", exc)
        yield

    app = FastAPI(title="entity-sentiment-at-scale", version="1.0", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, Any]:
        loaded: LoadedModel | None = state["model"]
        if loaded is None:
            raise HTTPException(status_code=503, detail=state["error"] or "model not loaded")
        return {"status": "ok", "model": loaded.info.name, "sha256": loaded.info.sha256}

    @app.post("/predict", response_model=PredictResponse)
    def predict_endpoint(request: PredictRequest) -> PredictResponse:
        loaded: LoadedModel | None = state["model"]
        if loaded is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        start = time.perf_counter()
        response = predict(loaded, request)
        logger.info(
            "predict: %d entities in %.1f ms",
            len(request.entities),
            (time.perf_counter() - start) * 1000,
        )
        return response

    return app


app = create_app()
