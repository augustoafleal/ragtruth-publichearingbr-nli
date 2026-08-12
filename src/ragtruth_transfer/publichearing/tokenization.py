from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from .config import PublicHearingConfig


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def token_cache_key(frame: pd.DataFrame, dataset_sha256: str, config: PublicHearingConfig) -> str:
    return _stable_hash({
        "dataset_sha256": dataset_sha256, "sample_ids": frame.sample_id.tolist(),
        "model_id": config.model_id, "model_revision": config.model_revision,
        "max_length": config.max_length,
    })


def _torch_load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def build_or_load_token_cache(frame: pd.DataFrame, dataset_sha256: str, config: PublicHearingConfig, cache_dir: Path, force: bool = False) -> tuple[dict[str, Any], Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = token_cache_key(frame, dataset_sha256, config)
    path = cache_dir / f"token_bags_{key[:16]}.pt"
    if path.is_file() and not force:
        cache = _torch_load(path)
        validate_token_cache(cache, frame, key, config)
        return cache, path
    tokenizer = AutoTokenizer.from_pretrained(config.model_id, revision=config.model_revision, use_fast=True)
    premises: list[str] = []
    hypotheses: list[str] = []
    for row in frame.itertuples(index=False):
        if len(row.context_chunks) != 4 or any(not str(chunk).strip() for chunk in row.context_chunks):
            raise RuntimeError(f"Bag inválida em {row.sample_id}")
        premises.extend(row.context_chunks)
        hypotheses.extend([row.opinion] * 4)
    batches: dict[str, list[torch.Tensor]] = {"input_ids": [], "attention_mask": []}
    token_types: list[torch.Tensor] = []
    for start in range(0, len(premises), config.tokenization_batch_size):
        encoded = tokenizer(premises[start:start + config.tokenization_batch_size], hypotheses[start:start + config.tokenization_batch_size], truncation="only_first", max_length=config.max_length, padding="max_length", return_tensors="pt")
        batches["input_ids"].append(encoded["input_ids"].to(torch.int32))
        batches["attention_mask"].append(encoded["attention_mask"].to(torch.uint8))
        if "token_type_ids" in encoded:
            token_types.append(encoded["token_type_ids"].to(torch.uint8))
    cache: dict[str, Any] = {
        "cache_key": key, "sample_ids": frame.sample_id.tolist(),
        "input_ids": torch.cat(batches["input_ids"]).reshape(len(frame), 4, config.max_length),
        "attention_mask": torch.cat(batches["attention_mask"]).reshape(len(frame), 4, config.max_length),
        "labels": torch.tensor(frame.label.to_numpy(dtype=np.int8), dtype=torch.int8),
        "metadata": {"dataset_sha256": dataset_sha256, "model_id": config.model_id, "model_revision": config.model_revision, "max_length": config.max_length},
    }
    if token_types:
        cache["token_type_ids"] = torch.cat(token_types).reshape(len(frame), 4, config.max_length)
    validate_token_cache(cache, frame, key, config)
    stage = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        torch.save(cache, stage)
        os.replace(stage, path)
    finally:
        if stage.exists():
            stage.unlink()
    return cache, path


def validate_token_cache(cache: dict[str, Any], frame: pd.DataFrame, key: str, config: PublicHearingConfig) -> None:
    if cache.get("cache_key") != key or cache.get("sample_ids") != frame.sample_id.tolist():
        raise RuntimeError("Cache de tokenização incompatível com dataset ou ordenação.")
    expected = (len(frame), 4, config.max_length)
    for name in ("input_ids", "attention_mask"):
        if tuple(cache[name].shape) != expected:
            raise RuntimeError(f"Shape inválido de {name}: {tuple(cache[name].shape)} != {expected}")


class TokenBagDataset(Dataset):
    def __init__(self, cache: dict[str, Any], indices: Sequence[int]) -> None:
        self.cache, self.indices = cache, np.asarray(indices, dtype=int)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> dict[str, torch.Tensor]:
        index = int(self.indices[position])
        item = {
            "row_index": torch.tensor(index, dtype=torch.long),
            "input_ids": self.cache["input_ids"][index].long(),
            "attention_mask": self.cache["attention_mask"][index].long(),
            "evidence_mask": torch.ones(4, dtype=torch.bool),
            "labels": self.cache["labels"][index].float(),
        }
        if "token_type_ids" in self.cache:
            item["token_type_ids"] = self.cache["token_type_ids"][index].long()
        return item


def make_loader(cache: dict[str, Any], indices: Sequence[int], batch_size: int, shuffle: bool, num_workers: int = 0) -> DataLoader:
    return DataLoader(TokenBagDataset(cache, indices), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=torch.cuda.is_available(), drop_last=False)
