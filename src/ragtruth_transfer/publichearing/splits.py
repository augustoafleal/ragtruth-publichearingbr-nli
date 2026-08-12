from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .config import PublicHearingConfig


@dataclass(frozen=True)
class FoldSplit:
    outer_fold: int
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def make_splits(frame: pd.DataFrame, config: PublicHearingConfig) -> list[FoldSplit]:
    labels = frame.label.to_numpy(dtype=int)
    groups = frame.hearing_id.astype(str).to_numpy()
    outer = StratifiedGroupKFold(config.effective_outer_splits, shuffle=True, random_state=config.split_seed)
    folds: list[FoldSplit] = []
    for fold, (train_dev, test) in enumerate(outer.split(np.zeros(len(frame)), labels, groups)):
        inner = StratifiedGroupKFold(config.inner_splits, shuffle=True, random_state=config.split_seed + 1000 + fold)
        target_prev, target_size = labels[train_dev].mean(), 1 / config.inner_splits
        candidates = []
        for candidate, (train_local, validation_local) in enumerate(inner.split(np.zeros(len(train_dev)), labels[train_dev], groups[train_dev])):
            validation = train_dev[validation_local]
            score = abs(labels[validation].mean() - target_prev) + .25 * abs(len(validation) / len(train_dev) - target_size)
            candidates.append((score, candidate, train_dev[train_local], validation))
        _, _, train, validation = min(candidates, key=lambda item: (item[0], item[1]))
        assert_no_group_leakage(groups, train, validation, test)
        for name, indices in (("train", train), ("validation", validation), ("test", test)):
            if np.unique(labels[indices]).size != 2:
                raise RuntimeError(f"{name} do fold {fold} não contém as duas classes")
        folds.append(FoldSplit(fold, train, validation, test))
    coverage = np.concatenate([fold.test for fold in folds])
    if sorted(coverage.tolist()) != list(range(len(frame))):
        raise RuntimeError("Cobertura de folds externos inválida.")
    return folds


def assert_no_group_leakage(groups: np.ndarray, train: np.ndarray, validation: np.ndarray, test: np.ndarray) -> None:
    sets = [set(groups[index]) for index in (train, validation, test)]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise RuntimeError("Leakage de hearing_id entre treino, validação e teste.")


def split_frames(frame: pd.DataFrame, folds: list[FoldSplit]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, membership = [], []
    for fold in folds:
        for role, indices in (("train", fold.train), ("validation", fold.validation), ("test", fold.test)):
            part = frame.iloc[indices]
            rows.append({"outer_fold": fold.outer_fold, "role": role, "examples": len(part), "hearings": part.hearing_id.nunique(), "positives": int(part.label.sum()), "prevalence": float(part.label.mean())})
            membership.extend({"sample_id": row.sample_id, "hearing_id": row.hearing_id, "outer_fold": fold.outer_fold, "role": role, "label": int(row.label)} for row in part.itertuples())
    return pd.DataFrame(rows), pd.DataFrame(membership)


def splits_hash(frame: pd.DataFrame, folds: list[FoldSplit]) -> str:
    payload = [{"fold": fold.outer_fold, **{name: frame.iloc[idx].sample_id.tolist() for name, idx in (("train", fold.train), ("validation", fold.validation), ("test", fold.test))}} for fold in folds]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
