from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

from ..io_utils import sha256_file


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        stage.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(stage, path)
    finally:
        if stage.exists(): stage.unlink()


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        frame.to_csv(stage, index=False)
        os.replace(stage, path)
    finally:
        if stage.exists(): stage.unlink()


def file_manifest(root: Path) -> list[dict[str, Any]]:
    return [{"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in sorted(root.rglob("*")) if path.is_file() and not path.name.startswith(".")]


def make_archive(run_dir: Path, include_checkpoints: bool) -> Path:
    staging = run_dir / f".zip-staging-{uuid.uuid4().hex}"
    try:
        shutil.copytree(run_dir / "outputs", staging / "outputs")
        for name in ("run_config.json", "experiment_signature.json"):
            if (run_dir / name).is_file(): shutil.copy2(run_dir / name, staging / name)
        if include_checkpoints and (run_dir / "folds").is_dir(): shutil.copytree(run_dir / "folds", staging / "folds")
        return Path(shutil.make_archive(str(run_dir / "publichearing_results"), "zip", root_dir=staging))
    finally:
        if staging.exists(): shutil.rmtree(staging)
