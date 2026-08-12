
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol

import numpy as np
import torch

from .ragtruth_top4_config import RetrieverSettings


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def signature_for(value: Any, length: int = 16) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()[:length]


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


class OffsetTokenizer(Protocol):
    model_id: str
    revision: str | None

    def encode_offsets(self, text: str) -> tuple[list[int], list[tuple[int, int]]]: ...


class WhitespaceTokenizer:
    model_id = "mock-whitespace"
    revision = "local-v1"

    def encode_offsets(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        offsets = [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]
        return list(range(len(offsets))), offsets


class HFOffsetTokenizer:
    def __init__(self, tokenizer: Any, model_id: str, revision: str | None) -> None:
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.revision = revision

    def encode_offsets(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
            return_offsets_mapping=True,
            truncation=False,
        )
        offsets = [tuple(int(x) for x in pair) for pair in encoded["offset_mapping"]]
        ids = [int(x) for x in encoded["input_ids"]]
        keep = [(token_id, offset) for token_id, offset in zip(ids, offsets) if offset[1] > offset[0]]
        return [item[0] for item in keep], [item[1] for item in keep]


def build_tokenizer(settings: RetrieverSettings) -> OffsetTokenizer:
    if settings.encoder_backend == "mock":
        return WhitespaceTokenizer()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        settings.tokenizer_model_id,
        revision=settings.tokenizer_revision,
        use_fast=True,
    )
    return HFOffsetTokenizer(tokenizer, settings.tokenizer_model_id, settings.tokenizer_revision)


@dataclass(frozen=True)
class Chunk:
    chunk_key: str
    source_index: int
    window_index: int
    token_start: int
    token_end: int
    char_start: int
    char_end: int
    text: str
    text_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_key": self.chunk_key,
            "source_index": self.source_index,
            "window_index": self.window_index,
            "token_start": self.token_start,
            "token_end": self.token_end,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "text": self.text,
            "text_sha256": self.text_sha256,
        }


def chunk_passage(
    text: str,
    source_index: int,
    tokenizer: OffsetTokenizer,
    chunk_size_tokens: int,
    chunk_overlap_tokens: int,
    source_id: str,
) -> list[Chunk]:
    clean = normalize_text(text)
    if not clean:
        return []
    token_ids, offsets = tokenizer.encode_offsets(clean)
    if not offsets:
        return []
    step = chunk_size_tokens - chunk_overlap_tokens
    chunks: list[Chunk] = []
    window_index = 0
    for token_start in range(0, len(offsets), step):
        token_end = min(token_start + chunk_size_tokens, len(offsets))
        char_start = offsets[token_start][0]
        char_end = offsets[token_end - 1][1]
        chunk_text = clean[char_start:char_end].strip()
        if chunk_text:
            text_hash = sha256_text(chunk_text)
            chunks.append(
                Chunk(
                    chunk_key=f"{source_id}:{source_index}:{window_index}:{text_hash[:16]}",
                    source_index=source_index,
                    window_index=window_index,
                    token_start=token_start,
                    token_end=token_end,
                    char_start=char_start,
                    char_end=char_end,
                    text=chunk_text,
                    text_sha256=text_hash,
                )
            )
        window_index += 1
        if token_end >= len(offsets):
            break
    return chunks


class TextEncoder(Protocol):
    dimension: int
    model_id: str
    revision: str | None
    device: str
    dtype: str

    def encode(self, texts: list[str], prefix: str, batch_size: int) -> np.ndarray: ...


class MockTextEncoder:
    def __init__(self, dimension: int = 32) -> None:
        self.dimension = dimension
        self.model_id = "mock-hash-encoder"
        self.revision = "local-v1"
        self.device = "cpu"
        self.dtype = "float32"

    def encode(self, texts: list[str], prefix: str, batch_size: int) -> np.ndarray:
        rows: list[np.ndarray] = []
        for text in texts:
            digest = hashlib.sha256((prefix + text).encode("utf-8")).digest()
            seed = int.from_bytes(digest[:8], "little", signed=False)
            rng = np.random.default_rng(seed)
            vector = rng.standard_normal(self.dimension).astype(np.float32)
            norm = float(np.linalg.norm(vector)) or 1.0
            rows.append(vector / norm)
        return np.vstack(rows) if rows else np.empty((0, self.dimension), dtype=np.float32)


class TransformersTextEncoder:
    def __init__(self, settings: RetrieverSettings) -> None:
        from transformers import AutoModel, AutoTokenizer

        if not settings.revision or not settings.tokenizer_revision:
            raise ValueError("Encoder E5 requer revision e tokenizer_revision imutáveis.")
        self.settings = settings
        selected_device = settings.device
        if selected_device == "auto":
            selected_device = "cuda" if torch.cuda.is_available() else "cpu"
        if selected_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA foi solicitado, mas torch.cuda.is_available() é falso.")
        self.device = selected_device
        self.dtype = settings.dtype
        if self.device == "cpu" and self.dtype in {"float16", "bfloat16", "auto"}:
            self.dtype = "float32"
        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.tokenizer_model_id,
            revision=settings.tokenizer_revision,
            use_fast=True,
        )
        self.model = AutoModel.from_pretrained(settings.model_id, revision=settings.revision)
        self.model.to(self.device)
        self.model.eval()
        self.dimension = int(getattr(self.model.config, "hidden_size", settings.embedding_dim))
        self.model_id = settings.model_id
        self.revision = settings.revision or getattr(self.model.config, "_commit_hash", None)

    def encode(self, texts: list[str], prefix: str, batch_size: int) -> np.ndarray:
        output: list[np.ndarray] = []
        max_length = self.settings.max_length
        autocast_dtype = None
        if self.device.startswith("cuda"):
            if self.settings.dtype == "float16":
                autocast_dtype = torch.float16
            elif self.settings.dtype == "bfloat16":
                autocast_dtype = torch.bfloat16
        with torch.inference_mode():
            for start in range(0, len(texts), max(1, batch_size)):
                batch = [prefix + text for text in texts[start : start + max(1, batch_size)]]
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=max_length is not None,
                    max_length=max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                context = (
                    torch.autocast(device_type="cuda", dtype=autocast_dtype)
                    if autocast_dtype is not None
                    else torch.autocast(device_type="cpu", enabled=False)
                )
                with context:
                    hidden = self.model(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
                pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=1)
                output.append(pooled.cpu().numpy().astype(np.float32, copy=False))
        return np.vstack(output) if output else np.empty((0, self.dimension), dtype=np.float32)


def build_encoder(settings: RetrieverSettings) -> TextEncoder:
    if settings.encoder_backend == "mock":
        return MockTextEncoder(settings.embedding_dim)
    return TransformersTextEncoder(settings)


class EmbeddingCache:

    def __init__(self, root: Path, kind: str, signature: str) -> None:
        self.root = root
        self.kind = kind
        self.signature = signature
        self.root.mkdir(parents=True, exist_ok=True)
        self.data_path = self.root / f"{kind}-{signature}.npz"
        self.meta_path = self.root / f"{kind}-{signature}.json"

    @staticmethod
    def _keys_hash(keys: list[str]) -> str:
        return hashlib.sha256("\n".join(keys).encode("utf-8")).hexdigest()

    def load(self, keys: list[str], dimension: int) -> np.ndarray | None:
        if len(keys) != len(set(keys)):
            return None
        if not self.data_path.is_file() or not self.meta_path.is_file():
            return None
        try:
            meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
            if (
                meta.get("schema_version") != "ragtruth-embedding-cache-v1"
                or meta.get("signature") != self.signature
                or meta.get("keys_hash") != self._keys_hash(keys)
                or int(meta.get("rows", -1)) != len(keys)
                or int(meta.get("dimension", -1)) != dimension
                or meta.get("dtype") != "float32"
            ):
                return None
            if meta.get("data_sha256") != _sha256_file(self.data_path):
                return None
            with np.load(self.data_path, allow_pickle=False) as archive:
                cached_keys = [str(x) for x in archive["keys"].tolist()]
                embeddings = np.asarray(archive["embeddings"], dtype=np.float32)
            if cached_keys != keys or embeddings.shape != (len(keys), dimension):
                return None
            if not np.isfinite(embeddings).all():
                return None
            if len(embeddings) and not np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=2e-3):
                return None
            return embeddings
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return None

    def save(self, keys: list[str], embeddings: np.ndarray) -> None:
        if len(keys) != len(set(keys)):
            raise ValueError("Chaves duplicadas não podem ser gravadas no cache.")
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(keys) or not np.isfinite(embeddings).all():
            raise ValueError("Embeddings inválidos para cache.")
        self.root.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=self.root, prefix=f".{self.kind}-", suffix=".npz", delete=False) as handle:
            temp_data = Path(handle.name)
        try:
            np.savez_compressed(temp_data, keys=np.asarray(keys), embeddings=embeddings)
            os.replace(temp_data, self.data_path)
            meta = {
                "schema_version": "ragtruth-embedding-cache-v1",
                "signature": self.signature,
                "kind": self.kind,
                "keys_hash": self._keys_hash(keys),
                "rows": len(keys),
                "dimension": int(embeddings.shape[1]),
                "dtype": "float32",
            }
            meta["data_sha256"] = _sha256_file(self.data_path)
            with tempfile.NamedTemporaryFile(
                dir=self.root, prefix=f".{self.kind}-", suffix=".json", mode="w", encoding="utf-8", delete=False
            ) as handle:
                json.dump(meta, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
                temp_meta = Path(handle.name)
            os.replace(temp_meta, self.meta_path)
        finally:
            if temp_data.exists():
                temp_data.unlink()
            if "temp_meta" in locals() and temp_meta.exists():
                temp_meta.unlink()
