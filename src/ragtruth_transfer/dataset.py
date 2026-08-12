from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from .io_utils import read_jsonl


class EvidenceBagDataset(Dataset):
    def __init__(self, path: Path, limit: int | None = None) -> None:
        self.rows = read_jsonl(path)
        if limit is not None:
            self.rows = self.rows[:limit]
        if not self.rows:
            raise ValueError(f"Dataset vazio: {path}")
        for row in self.rows:
            evidence = row.get("evidence")
            mask = row.get("evidence_mask")
            if not isinstance(evidence, list) or not isinstance(mask, list) or len(evidence) != len(mask):
                raise ValueError(f"Exemplo inválido: {row.get('example_id')}")
            if not any(bool(value) for value in mask):
                raise ValueError(f"Exemplo sem evidência válida: {row.get('example_id')}")

    @classmethod
    def from_rows(cls, rows: list[dict[str, Any]], limit: int | None = None, allow_empty: bool = False) -> "EvidenceBagDataset":
        instance = cls.__new__(cls)
        instance.rows = list(rows[:limit] if limit is not None else rows)
        if not instance.rows and not allow_empty:
            raise ValueError("Dataset vazio")
        for row in instance.rows:
            evidence = row.get("evidence")
            mask = row.get("evidence_mask")
            if not isinstance(evidence, list) or not isinstance(mask, list) or len(evidence) != len(mask):
                raise ValueError(f"Exemplo inválido: {row.get('example_id')}")
            if not any(bool(value) for value in mask):
                raise ValueError(f"Exemplo sem evidência válida: {row.get('example_id')}")
        return instance

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]

    @property
    def labels(self) -> np.ndarray:
        return np.asarray([bool(row["label"]) for row in self.rows], dtype=bool)

    @property
    def tasks(self) -> list[str]:
        return [str(row.get("task_type", "unknown")) for row in self.rows]

    @property
    def sources(self) -> list[str]:
        return [str(row.get("source_id", "")) for row in self.rows]

    def task_balanced_sampler(
        self,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> WeightedRandomSampler:
        counts = Counter(self.tasks)
        weights = torch.tensor([1.0 / counts[task] for task in self.tasks], dtype=torch.double)
        if generator is None:
            if seed is None:
                raise ValueError("seed ou generator é obrigatório")
            generator = torch.Generator().manual_seed(seed)
        return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True, generator=generator)


class RagTruthParquetDataset(EvidenceBagDataset):

    @classmethod
    def from_parquet(cls, path: Path, **kwargs: Any) -> "RagTruthParquetDataset":
        from .ragtruth_parquet import load_ragtruth_parquet

        rows, _ = load_ragtruth_parquet(path, **kwargs)
        return cls.from_rows(rows)  # type: ignore[return-value]


class BagCollator:
    def __init__(self, tokenizer, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        n_evidence = len(rows[0]["evidence"])
        premises: list[str] = []
        claims: list[str] = []
        evidence_masks: list[list[bool]] = []
        for row in rows:
            if len(row["evidence"]) != n_evidence:
                raise ValueError("Número variável de evidências dentro do batch.")
            premises.extend(str(value) for value in row["evidence"])
            claims.extend([str(row["claim"])] * n_evidence)
            evidence_masks.append([bool(value) for value in row["evidence_mask"]])

        encoded = self.tokenizer(
            premises,
            claims,
            truncation="only_first",
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )
        batch_size = len(rows)
        result: dict[str, Any] = {
            key: value.reshape(batch_size, n_evidence, value.shape[-1])
            for key, value in encoded.items()
        }
        result["evidence_mask"] = torch.tensor(evidence_masks, dtype=torch.bool)
        result["labels"] = torch.tensor([float(bool(row["label"])) for row in rows], dtype=torch.float32)
        result["example_ids"] = [str(row["example_id"]) for row in rows]
        result["source_ids"] = [str(row["source_id"]) for row in rows]
        result["task_types"] = [str(row.get("task_type", "")) for row in rows]
        return result
