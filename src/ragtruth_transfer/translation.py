from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import yaml

TRANSLATION_CACHE_SCHEMA = "ragtruth-translation-cache-v1"
TRANSLATION_MANIFEST_SCHEMA = "ragtruth-translated-v1"
_SPLIT_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


class BatchTranslator(Protocol):
    def translate_batch(self, texts: list[str]) -> list[str]:
        ...


@dataclass(frozen=True)
class TranslationSettings:
    translator: str
    model_name: str
    source_language: str
    target_language: str
    batch_size: int = 16
    num_beams: int = 4
    max_input_tokens: int = 480
    max_new_tokens: int = 512
    device: str = "auto"
    model_revision: str | None = None

    def __post_init__(self) -> None:
        if self.translator not in {"nllb", "madlad"}:
            raise ValueError(f"Tradutor desconhecido: {self.translator}. Use nllb ou madlad.")
        if self.batch_size < 1:
            raise ValueError("translation.batch_size deve ser positivo")
        if self.num_beams < 1:
            raise ValueError("translation.num_beams deve ser positivo")
        if self.max_input_tokens < 1 or self.max_new_tokens < 1:
            raise ValueError("Limites de tokens devem ser positivos")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("translation.device deve ser auto, cpu ou cuda")


@dataclass(frozen=True)
class TranslationConfig:
    translation: TranslationSettings
    input_dir: Path
    output_dir: Path
    splits: tuple[str, ...]
    cache_path: Path | None = None
    sample_fraction: float | None = None
    sample_seed: int = 42
    max_examples_per_split: int | None = None

    def __post_init__(self) -> None:
        if self.sample_fraction is not None and not 0.0 < self.sample_fraction <= 1.0:
            raise ValueError("data.sample_fraction deve estar no intervalo (0, 1]")
        if self.max_examples_per_split is not None and self.max_examples_per_split < 1:
            raise ValueError("data.max_examples_per_split deve ser positivo")
        if self.sample_fraction is not None and self.max_examples_per_split is not None:
            raise ValueError(
                "data.sample_fraction e data.max_examples_per_split são incompatíveis; use apenas um"
            )

    @property
    def effective_sample_fraction(self) -> float:
        return 1.0 if self.sample_fraction is None else self.sample_fraction

    @classmethod
    def from_yaml(cls, path: Path) -> "TranslationConfig":
        path = path.expanduser().resolve()
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"Configuração YAML inválida: {path}")
        translation_raw = raw.get("translation")
        data_raw = raw.get("data")
        if not isinstance(translation_raw, dict) or not isinstance(data_raw, dict):
            raise ValueError("A configuração deve conter os mapas translation e data")

        translator = str(translation_raw.get("translator", "")).lower()
        defaults = {
            "nllb": {
                "model_name": "facebook/nllb-200-distilled-600M",
                "source_language": "eng_Latn",
                "target_language": "por_Latn",
            },
            "madlad": {
                "model_name": "google/madlad400-3b-mt",
                "source_language": "en",
                "target_language": "pt",
            },
        }
        if translator not in defaults:
            raise ValueError(f"Tradutor desconhecido: {translator}. Use nllb ou madlad.")
        backend_defaults = defaults[translator]

        def resolve(value: Any, default: str | None = None) -> Path | None:
            if value is None:
                value = default
            if value is None:
                return None
            result = Path(str(value)).expanduser()
            if not result.is_absolute():
                result = path.parent / result
            return result.resolve()

        splits_raw = data_raw.get("splits", ["train", "validation", "test"])
        if not isinstance(splits_raw, list) or not splits_raw:
            raise ValueError("data.splits deve ser uma lista não vazia")
        splits = tuple(str(value) for value in splits_raw)
        if any(not _SPLIT_NAME.fullmatch(value) for value in splits):
            raise ValueError("data.splits contém nome inválido")
        if len(set(splits)) != len(splits):
            raise ValueError("data.splits não pode conter duplicatas")

        input_dir = resolve(data_raw.get("input_dir"))
        output_dir = resolve(data_raw.get("output_dir"))
        if input_dir is None or output_dir is None:
            raise ValueError("data.input_dir e data.output_dir são obrigatórios")
        cache_path = resolve(data_raw.get("cache_path"))
        has_fraction = "sample_fraction" in data_raw
        has_smoke_limit = "max_examples_per_split" in data_raw and data_raw.get("max_examples_per_split") is not None
        if has_fraction and has_smoke_limit:
            raise ValueError(
                "data.sample_fraction e data.max_examples_per_split são incompatíveis; use apenas um"
            )
        sample_fraction = float(data_raw["sample_fraction"]) if has_fraction else None
        max_examples_per_split = (
            int(data_raw["max_examples_per_split"]) if has_smoke_limit else None
        )
        settings = TranslationSettings(
            translator=translator,
            model_name=str(translation_raw.get("model_name", backend_defaults["model_name"])),
            source_language=str(translation_raw.get("source_language", backend_defaults["source_language"])),
            target_language=str(translation_raw.get("target_language", backend_defaults["target_language"])),
            batch_size=int(translation_raw.get("batch_size", 16)),
            num_beams=int(translation_raw.get("num_beams", 4)),
            max_input_tokens=int(translation_raw.get("max_input_tokens", 480)),
            max_new_tokens=int(translation_raw.get("max_new_tokens", 512)),
            device=str(translation_raw.get("device", "auto")).lower(),
            model_revision=(str(translation_raw["model_revision"]) if translation_raw.get("model_revision") else None),
        )
        return cls(
            settings,
            input_dir,
            output_dir,
            splits,
            cache_path,
            sample_fraction,
            int(data_raw.get("sample_seed", 42)),
            max_examples_per_split,
        )


def _resolved_device(requested: str) -> Any:
    import torch

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("translation.device=cuda, mas CUDA não está disponível")
    return torch.device(requested)


def _preferred_dtype(device: Any) -> Any:
    import torch

    if device.type != "cuda":
        return torch.float32
    if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


class _TransformersTranslator:
    def __init__(self, settings: TranslationSettings) -> None:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.settings = settings
        self._torch = torch
        self.device = _resolved_device(settings.device)
        self.dtype = _preferred_dtype(self.device)
        model_load_kwargs: dict[str, Any] = {"torch_dtype": self.dtype}
        if settings.model_revision:
            model_load_kwargs["revision"] = settings.model_revision
        tokenizer_load_kwargs = {"revision": settings.model_revision} if settings.model_revision else {}
        self.tokenizer = AutoTokenizer.from_pretrained(settings.model_name, **tokenizer_load_kwargs)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(settings.model_name, **model_load_kwargs)
        self.model.to(self.device)
        self.model.eval()

    def _encode(self, texts: list[str]) -> dict[str, Any]:
        tokenized = self.tokenizer(texts, truncation=False, padding=False, add_special_tokens=True)
        lengths = [len(ids) for ids in tokenized["input_ids"]]
        too_long = [index for index, length in enumerate(lengths) if length > self.settings.max_input_tokens]
        if too_long:
            raise ValueError(
                "Texto excede translation.max_input_tokens "
                f"({self.settings.max_input_tokens}); índices no batch: {too_long}"
            )
        encoded = self.tokenizer(
            texts,
            truncation=False,
            max_length=self.settings.max_input_tokens,
            padding=True,
            return_tensors="pt",
            add_special_tokens=True,
        )
        return {key: value.to(self.device) for key, value in encoded.items()}

    def _generate(self, texts: list[str], **kwargs: Any) -> list[str]:
        encoded = self._encode(texts)
        with self._torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                num_beams=self.settings.num_beams,
                max_new_tokens=self.settings.max_new_tokens,
                **kwargs,
            )
        result = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        if len(result) != len(texts) or any(not isinstance(value, str) for value in result):
            raise RuntimeError("O modelo retornou quantidade inválida de traduções")
        return result

    def translate_batch(self, texts: list[str]) -> list[str]:
        if not texts:
            return []
        try:
            return self._translate_batch(texts)
        except RuntimeError as error:
            is_oom = self.device.type == "cuda" and "out of memory" in str(error).lower()
            if not is_oom:
                raise
            if len(texts) == 1:
                raise RuntimeError("CUDA OOM ao traduzir um único texto; reduza os limites de tokens") from error
            self._torch.cuda.empty_cache()
            middle = len(texts) // 2
            return self.translate_batch(texts[:middle]) + self.translate_batch(texts[middle:])

    def _translate_batch(self, texts: list[str]) -> list[str]:
        raise NotImplementedError


class NLLBTranslator(_TransformersTranslator):
    def __init__(self, settings: TranslationSettings) -> None:
        super().__init__(settings)
        self.tokenizer.src_lang = settings.source_language
        self.target_token_id = self.tokenizer.convert_tokens_to_ids(settings.target_language)
        if self.target_token_id is None or self.target_token_id == self.tokenizer.unk_token_id:
            raise ValueError(f"Idioma alvo NLLB desconhecido: {settings.target_language}")

    def _translate_batch(self, texts: list[str]) -> list[str]:
        return self._generate(texts, forced_bos_token_id=self.target_token_id)


class MADLADTranslator(_TransformersTranslator):
    def _translate_batch(self, texts: list[str]) -> list[str]:
        prefixed = [f"<2{self.settings.target_language}> {text}" for text in texts]
        return self._generate(prefixed)


def create_translator(settings: TranslationSettings) -> BatchTranslator:
    if settings.translator == "nllb":
        return NLLBTranslator(settings)
    if settings.translator == "madlad":
        return MADLADTranslator(settings)
    raise ValueError(f"Tradutor desconhecido: {settings.translator}. Use nllb ou madlad.")


def _config_signature(settings: TranslationSettings) -> str:
    payload = {
        "translator": settings.translator,
        "model_name": settings.model_name,
        "model_revision": settings.model_revision,
        "source_language": settings.source_language,
        "target_language": settings.target_language,
        "num_beams": settings.num_beams,
        "max_input_tokens": settings.max_input_tokens,
        "max_new_tokens": settings.max_new_tokens,
        "device": settings.device,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _input_signature(input_files: dict[str, str], splits: tuple[str, ...]) -> str:
    payload = {"splits": list(splits), "files": input_files}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


class TranslationCache:
    """Small SQLite cache with atomic per-batch commits for safe resume."""

    def __init__(self, path: Path, config_signature: str, input_signature: str) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS translations (text_hash TEXT PRIMARY KEY, translation TEXT NOT NULL)"
        )
        self.connection.commit()
        expected = {
            "schema": TRANSLATION_CACHE_SCHEMA,
            "config_signature": config_signature,
            "input_signature": input_signature,
        }
        found = dict(self.connection.execute("SELECT key, value FROM metadata"))
        if found and found != expected:
            self.close()
            raise ValueError(f"Cache de tradução incompatível: {path}")
        if not found:
            if self.connection.execute("SELECT 1 FROM translations LIMIT 1").fetchone() is not None:
                self.close()
                raise ValueError(f"Cache de tradução sem metadata: {path}")
            self.connection.executemany("INSERT INTO metadata(key, value) VALUES (?, ?)", expected.items())
            self.connection.commit()

    def get_many(self, text_hashes: list[str]) -> dict[str, str]:
        if not text_hashes:
            return {}
        result: dict[str, str] = {}
        for start in range(0, len(text_hashes), 900):
            batch = text_hashes[start : start + 900]
            placeholders = ",".join("?" for _ in batch)
            rows = self.connection.execute(
                f"SELECT text_hash, translation FROM translations WHERE text_hash IN ({placeholders})", batch
            )
            result.update(dict(rows.fetchall()))
        return result

    def put_many(self, values: dict[str, str]) -> None:
        self.connection.executemany(
            "INSERT OR REPLACE INTO translations(text_hash, translation) VALUES (?, ?)", values.items()
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "TranslationCache":
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        self.close()


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _dataset_rows(config: TranslationConfig) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    if config.input_dir.resolve() == config.output_dir.resolve():
        raise ValueError("data.input_dir e data.output_dir devem ser diferentes")
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    input_files: dict[str, str] = {}
    for split in config.splits:
        path = config.input_dir / f"{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Split ausente: {path}")
        input_files[split] = _sha256_file(path)
        rows = _read_jsonl(path)
        for row_index, row in enumerate(rows):
            if not isinstance(row.get("claim"), str):
                raise ValueError(f"{path}:{row_index + 1} tem claim que não é string")
            evidence = row.get("evidence")
            mask = row.get("evidence_mask")
            if not isinstance(evidence, list) or not isinstance(mask, list) or len(evidence) != len(mask):
                raise ValueError(f"{path}:{row_index + 1} tem evidence/evidence_mask incompatíveis")
            if any(not isinstance(value, bool) for value in mask):
                raise ValueError(f"{path}:{row_index + 1} tem evidence_mask que não é booleana")
            if any(mask[index] and not isinstance(evidence[index], str) for index in range(len(mask))):
                raise ValueError(f"{path}:{row_index + 1} tem evidência válida que não é string")
        rows_by_split[split] = _select_rows(rows, split, config)
    return rows_by_split, input_files


def _select_rows(rows: list[dict[str, Any]], split: str, config: TranslationConfig) -> list[dict[str, Any]]:
    if config.max_examples_per_split is not None:
        return rows[: config.max_examples_per_split]
    fraction = config.effective_sample_fraction
    if fraction >= 1.0:
        return rows
    count = min(len(rows), max(1, math.ceil(len(rows) * fraction)))
    ranked = sorted(
        range(len(rows)),
        key=lambda index: hashlib.sha256(
            f"{config.sample_seed}:{split}:{index}:{rows[index].get('example_id', '')}".encode("utf-8")
        ).hexdigest(),
    )
    selected = set(ranked[:count])
    return [row for index, row in enumerate(rows) if index in selected]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Esperava objeto JSON em {path}, linha {line_number}")
            rows.append(value)
    return rows


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _translatable_texts(rows_by_split: dict[str, list[dict[str, Any]]]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for rows in rows_by_split.values():
        for row in rows:
            candidates = [row["claim"]]
            candidates.extend(
                value for value, is_valid in zip(row["evidence"], row["evidence_mask"]) if is_valid
            )
            for text in candidates:
                if text and text not in seen:
                    seen.add(text)
                    result.append(text)
    return result


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def translate_ragtruth(
    config: TranslationConfig,
    translator: BatchTranslator | None = None,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    rows_by_split, input_files = _dataset_rows(config)
    input_signature = _input_signature(input_files, config.splits)
    config_signature = _config_signature(config.translation)
    cache_path = config.cache_path or config.output_dir / ".translation_cache.sqlite3"
    config.output_dir.mkdir(parents=True, exist_ok=True)
    existing_manifest_path = config.output_dir / "manifest.json"
    selection_payload = {
        "sample_fraction": config.effective_sample_fraction,
        "sample_seed": config.sample_seed,
        "max_examples_per_split": config.max_examples_per_split,
    }
    selection_signature = hashlib.sha256(
        json.dumps(selection_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    run_signature = hashlib.sha256(
        f"{config_signature}:{input_signature}:{selection_signature}".encode("utf-8")
    ).hexdigest()
    if existing_manifest_path.is_file():
        existing = json.loads(existing_manifest_path.read_text(encoding="utf-8"))
        if existing.get("run_signature") != run_signature:
            raise ValueError(f"Output existente é incompatível com a configuração: {config.output_dir}")
        if resume and all((config.output_dir / f"{split}.jsonl").is_file() for split in config.splits):
            return existing

    texts = _translatable_texts(rows_by_split)
    hashes = {_text_hash(text): text for text in texts}
    translations: dict[str, str] = {}
    with TranslationCache(cache_path, config_signature, input_signature) as cache:
        translations.update(cache.get_many(list(hashes)))
        missing = [text for text in texts if _text_hash(text) not in translations]
        if missing:
            active_translator = translator or create_translator(config.translation)
            for start in range(0, len(missing), config.translation.batch_size):
                batch = missing[start : start + config.translation.batch_size]
                translated = active_translator.translate_batch(batch)
                if len(translated) != len(batch) or any(not isinstance(value, str) for value in translated):
                    raise RuntimeError("Tradutor retornou resultado incompleto ou inválido")
                batch_values = {_text_hash(text): value for text, value in zip(batch, translated)}
                cache.put_many(batch_values)
                translations.update(batch_values)

        output_rows: dict[str, list[dict[str, Any]]] = {}
        for split, rows in rows_by_split.items():
            converted: list[dict[str, Any]] = []
            for row in rows:
                result = dict(row)
                result["claim"] = translations.get(_text_hash(row["claim"]), row["claim"])
                result["evidence"] = [
                    translations.get(_text_hash(value), value) if is_valid else value
                    for value, is_valid in zip(row["evidence"], row["evidence_mask"])
                ]
                converted.append(result)
            output_rows[split] = converted

    for split, rows in output_rows.items():
        _write_jsonl_atomic(config.output_dir / f"{split}.jsonl", rows)

    mode = (
        "smoke"
        if config.max_examples_per_split is not None
        else ("full" if config.effective_sample_fraction >= 1.0 else "sample")
    )
    manifest = {
        "schema_version": TRANSLATION_MANIFEST_SCHEMA,
        "status": "completed",
        "run_signature": run_signature,
        "mode": mode,
        "sample_fraction": config.effective_sample_fraction,
        "sample_seed": config.sample_seed,
        "max_examples_per_split": config.max_examples_per_split,
        "is_full_dataset": mode == "full",
        "backend": config.translation.translator,
        "model_name": config.translation.model_name,
        "model_revision": config.translation.model_revision,
        "source_language": config.translation.source_language,
        "target_language": config.translation.target_language,
        "generation": {
            "num_beams": config.translation.num_beams,
            "max_input_tokens": config.translation.max_input_tokens,
            "max_new_tokens": config.translation.max_new_tokens,
        },
        "device": config.translation.device,
        "splits": list(config.splits),
        "counts": {
            "examples": sum(len(rows) for rows in rows_by_split.values()),
            "unique_texts": len(texts),
            "examples_by_split": {split: len(rows) for split, rows in rows_by_split.items()},
        },
        "input": {
            "directory": str(config.input_dir),
            "files_sha256": input_files,
            "signature": input_signature,
            "selection_signature": selection_signature,
        },
    }
    _write_text_atomic(existing_manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest
