from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import yaml
import pandas as pd

from .ragtruth_parquet import load_ragtruth_parquet, split_ragtruth_parquet, validate_training_view_manifest
from .translation_qa import classify_translation_pair

TRANSLATION_CACHE_SCHEMA = "ragtruth-translation-cache-v1"
TRANSLATION_MANIFEST_SCHEMA = "ragtruth-translated-v1"
PARQUET_TRANSLATION_MANIFEST_SCHEMA = "ragtruth-qa-training-view-deduplicated-v1"


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
    input_path: Path
    output_dir: Path
    cache_path: Path | None = None
    max_examples_per_split: int | None = None
    manifest_path: Path | None = None
    expected_source_sha256: str | None = None
    expected_source_signature: str | None = None
    expected_source_schema: str | None = None
    expected_source_rows: int | None = None
    reference_split_assignments: Path | None = None
    expected_split_signature: str | None = None

    def __post_init__(self) -> None:
        if self.max_examples_per_split is not None and self.max_examples_per_split < 1:
            raise ValueError("data.max_examples_per_split deve ser positivo")

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

        output_dir = resolve(data_raw.get("output_dir"))
        input_path = resolve(data_raw.get("input_path"))
        if output_dir is None or input_path is None:
            raise ValueError("O pipeline oficial exige data.input_path e data.output_dir Parquet")
        if "input_dir" in data_raw or "splits" in data_raw or "sample_fraction" in data_raw:
            raise ValueError("O pipeline oficial não aceita input_dir, splits ou sample_fraction JSONL")
        cache_path = resolve(data_raw.get("cache_path"))
        has_smoke_limit = "max_examples_per_split" in data_raw and data_raw.get("max_examples_per_split") is not None
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
            input_path,
            output_dir,
            cache_path,
            max_examples_per_split,
            resolve(data_raw.get("manifest_path")),
            str(data_raw["expected_source_sha256"]) if data_raw.get("expected_source_sha256") else None,
            str(data_raw["expected_source_signature"]) if data_raw.get("expected_source_signature") else None,
            str(data_raw["expected_source_schema"]) if data_raw.get("expected_source_schema") else None,
            int(data_raw["expected_source_rows"]) if data_raw.get("expected_source_rows") is not None else None,
            resolve(data_raw.get("reference_split_assignments")),
            str(data_raw["expected_split_signature"]) if data_raw.get("expected_split_signature") else None,
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
        self.resolved_model_revision = getattr(self.model.config, "_commit_hash", None)
        self.resolved_dtype = str(self.dtype)
        self.resolved_device = str(self.device)

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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_parquet_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _parquet_rows(config: TranslationConfig) -> tuple[pd.DataFrame, dict[str, Any], str]:
    if config.input_path is None:
        raise ValueError("A origem Parquet não está configurada")
    if config.input_path.resolve() == (config.output_dir / "dataset.parquet").resolve():
        raise ValueError("O Parquet de entrada e saída devem ser diferentes")
    lineage = validate_training_view_manifest(
        config.input_path,
        manifest_path=config.manifest_path,
        expected_signature=config.expected_source_signature,
        expected_schema=config.expected_source_schema,
    )
    if config.expected_source_sha256 and lineage["dataset_sha256"] != config.expected_source_sha256:
        raise ValueError("SHA-256 do Parquet de origem diverge da configuração")
    frame = pd.read_parquet(config.input_path)
    required = {"example_id", "source_id", "response_id", "split", "claim", "label", "evidence_mask", "chunk_1", "chunk_2", "chunk_3", "chunk_4"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Parquet de origem sem colunas obrigatórias: {missing}")
    if frame["response_id"].isna().any() or (frame["response_id"].astype(str).str.strip() == "").any():
        raise ValueError("response_id não pode estar vazio no Parquet de origem")
    load_ragtruth_parquet(
        config.input_path,
        manifest_path=config.manifest_path,
        expected_signature=config.expected_source_signature,
        expected_schema=config.expected_source_schema,
    )
    if config.expected_source_rows is not None and len(frame) != config.expected_source_rows:
        raise ValueError("Número de linhas do Parquet de origem diverge da configuração")
    lineage = {**lineage, "input_rows": int(len(frame))}
    if config.max_examples_per_split is not None:
        frame = frame.iloc[: config.max_examples_per_split].copy()
    return frame, lineage, _input_signature({"dataset.parquet": lineage["dataset_sha256"]}, ())


def _parquet_texts(frame: pd.DataFrame) -> list[str]:
    texts: list[str] = []
    seen: set[str] = set()
    for _, row in frame.iterrows():
        candidates = [str(row["claim"])]
        mask = list(row["evidence_mask"])
        candidates.extend(str(row[f"chunk_{index}"]) for index, valid in enumerate(mask, start=1) if bool(valid))
        for text in candidates:
            if not text:
                raise ValueError("Texto traduzível vazio no Parquet canônico")
            if text not in seen:
                seen.add(text)
                texts.append(text)
    return texts


def _translation_signature(config: TranslationConfig, input_signature: str) -> tuple[str, str, str]:
    config_signature = _config_signature(config.translation)
    selection = {"max_examples": config.max_examples_per_split}
    selection_signature = hashlib.sha256(json.dumps(selection, sort_keys=True).encode("utf-8")).hexdigest()
    return config_signature, selection_signature, hashlib.sha256(
        f"{config_signature}:{input_signature}:{selection_signature}".encode("utf-8")
    ).hexdigest()


def _assert_same_split_assignments(config: TranslationConfig, output_path: Path, manifest_path: Path) -> dict[str, Any] | None:
    if config.reference_split_assignments is None:
        return None
    if config.max_examples_per_split is not None:
        return {"checked": False, "reason": "smoke_subset"}
    if not config.reference_split_assignments.is_file():
        raise FileNotFoundError(f"Split assignments de referência não encontrados: {config.reference_split_assignments}")
    rows, metadata = load_ragtruth_parquet(
        output_path,
        manifest_path=manifest_path,
        expected_schema=PARQUET_TRANSLATION_MANIFEST_SCHEMA,
    )
    split = split_ragtruth_parquet(rows, metadata, split_seed=42, validation_fraction=0.15, max_test_sources=None)
    expected = pd.read_parquet(config.reference_split_assignments)
    actual = split.assignments
    columns = ["source_id", "partition"]
    if any(column not in expected.columns for column in columns):
        raise ValueError("Split assignments de referência não contém source_id/partition")
    expected = expected[columns].sort_values(columns).reset_index(drop=True)
    actual = actual[columns].sort_values(columns).reset_index(drop=True)
    if not expected.equals(actual):
        raise ValueError("As atribuições train/validation/test do Parquet traduzido divergem da referência EN")
    return {"checked": True, "translated_signature": split.metadata["signature"], "reference": str(config.reference_split_assignments), "reference_signature": config.expected_split_signature}


def _translation_qa(source: pd.DataFrame, translated: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    flags: list[dict[str, Any]] = []
    pairs = 0
    counts = {"low_ratio": 0, "high_ratio": 0, "high_confidence_repetition": 0, "empty_translation": 0, "control_character_issue": 0, "exclusion_candidate": 0}
    for source_row, translated_row in zip(source.to_dict("records"), translated.to_dict("records")):
        values = [("claim", None, source_row["claim"], translated_row["claim"])]
        mask = list(source_row["evidence_mask"])
        values.extend((f"chunk_{index}", index, source_row[f"chunk_{index}"], translated_row[f"chunk_{index}"]) for index, valid in enumerate(mask, start=1) if bool(valid))
        for field, slot, original, rendered in values:
            pairs += 1
            metrics = classify_translation_pair(original, rendered)
            for name in counts:
                counts[name] += int(bool(metrics[name]))
            if metrics["exclusion_candidate"] or metrics["empty_translation"] or metrics["control_character_issue"]:
                flags.append({"example_id": str(source_row["example_id"]), "field": field, "slot": slot, **metrics})
    columns = ["example_id", "field", "slot", "source_normalized", "translated_normalized", "source_chars", "translated_chars", "source_words", "translated_words", "length_ratio", "empty_translation", "identical_to_source", "low_ratio", "high_ratio", "source_has_repetition", "translation_has_repetition", "source_repetition_metric", "translation_repetition_metric", "repetition_excess", "repetition_severity", "translation_added_repetition", "high_confidence_repetition", "control_character_issue", "exclusion_candidate"]
    return {"pairs": pairs, "flags": counts, "flagged_pairs": len(flags), "rows_removed": 0, "filtering_applied": False}, pd.DataFrame(flags, columns=columns)


def _runtime_manifest_fields(translator: BatchTranslator | None) -> dict[str, Any]:
    return {
        "resolved_model_revision": getattr(translator, "resolved_model_revision", None),
        "resolved_dtype": getattr(translator, "resolved_dtype", None),
        "resolved_device": getattr(translator, "resolved_device", None),
    }


def _translate_parquet(
    config: TranslationConfig,
    translator: BatchTranslator | None,
    *,
    resume: bool,
) -> dict[str, Any]:
    frame, lineage, input_signature = _parquet_rows(config)
    config_signature, selection_signature, run_signature = _translation_signature(config, input_signature)
    output_path = config.output_dir / "dataset.parquet"
    manifest_path = config.output_dir / "manifest.json"
    if output_path.exists() or manifest_path.exists():
        if not resume:
            raise FileExistsError(f"Output Parquet já existe: {config.output_dir}. Use --resume apenas para output compatível.")
        if not output_path.is_file() or not manifest_path.is_file():
            raise ValueError("Output Parquet parcialmente existente; não é seguro retomar")
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        status = existing.get("status")
        if existing.get("run_signature") != run_signature or status not in {"completed", "validating"}:
            raise ValueError("Output Parquet existente é incompatível com a configuração")
        for relative, expected in dict(existing.get("artifacts", {})).items():
            candidate = config.output_dir / relative
            if not candidate.is_file() or _sha256_file(candidate) != str(expected):
                raise ValueError("Artefato do Output Parquet existente diverge do manifesto")
        split_check = _assert_same_split_assignments(config, output_path, manifest_path)
        if status == "validating":
            if split_check is not None:
                existing["split_assignment_validation"] = split_check
            existing["status"] = "completed"
            _write_text_atomic(manifest_path, json.dumps(existing, ensure_ascii=False, indent=2) + "\n")
        return existing

    texts = _parquet_texts(frame)
    hashes = {_text_hash(text): text for text in texts}
    cache_path = config.cache_path or config.output_dir / ".translation_cache.sqlite3"
    translations: dict[str, str] = {}
    active: BatchTranslator | None = None
    cache_hits = 0
    cache_misses = 0
    with TranslationCache(cache_path, config_signature, input_signature) as cache:
        translations.update(cache.get_many(list(hashes)))
        cache_hits = len(translations)
        missing = [text for text in texts if _text_hash(text) not in translations]
        cache_misses = len(missing)
        if missing:
            active = translator or create_translator(config.translation)
            for start in range(0, len(missing), config.translation.batch_size):
                batch = missing[start : start + config.translation.batch_size]
                translated = active.translate_batch(batch)
                if len(translated) != len(batch) or any(not isinstance(value, str) or not value.strip() for value in translated):
                    raise RuntimeError("Tradutor retornou resultado incompleto ou vazio")
                values = {_text_hash(source): target for source, target in zip(batch, translated)}
                cache.put_many(values)
                translations.update(values)
    missing_hashes = sorted(set(hashes) - set(translations))
    if missing_hashes:
        raise RuntimeError(f"Traduções ausentes para {len(missing_hashes)} textos; nenhuma cópia em inglês é permitida")

    result = frame.copy(deep=True)
    result["claim"] = [translations[_text_hash(str(value))] for value in frame["claim"]]
    for index in range(1, 5):
        column = f"chunk_{index}"
        values: list[Any] = []
        for text, mask in zip(frame[column], frame["evidence_mask"]):
            values.append(translations[_text_hash(str(text))] if bool(list(mask)[index - 1]) else text)
        result[column] = values
    _write_parquet_atomic(output_path, result)
    qa_summary, qa_flags = _translation_qa(frame, result)
    qa_summary_path = config.output_dir / "translation_qa.json"
    qa_flags_path = config.output_dir / "translation_qa_flags.parquet"
    _write_text_atomic(qa_summary_path, json.dumps(qa_summary, ensure_ascii=False, indent=2) + "\n")
    _write_parquet_atomic(qa_flags_path, qa_flags)

    manifest = {
        "schema_version": PARQUET_TRANSLATION_MANIFEST_SCHEMA,
        "status": "validating",
        "signature": run_signature[:16],
        "run_signature": run_signature,
        "translation_schema": "ragtruth-confirmatory-parquet-translation-v1",
        "created_at": time.time(),
        "mode": "smoke" if config.max_examples_per_split is not None else "full",
        "is_full_dataset": config.max_examples_per_split is None,
        "backend": config.translation.translator,
        "model_name": config.translation.model_name,
        "requested_model_revision": config.translation.model_revision,
        **_runtime_manifest_fields(active),
        "source_language": config.translation.source_language,
        "target_language": config.translation.target_language,
        "generation": {"num_beams": config.translation.num_beams, "max_input_tokens": config.translation.max_input_tokens, "max_new_tokens": config.translation.max_new_tokens, "batch_size": config.translation.batch_size, "sampling": False},
        "device": config.translation.device,
        "source": {"path": str(config.input_path), "dataset_sha256": lineage["dataset_sha256"], "signature": lineage["signature"], "schema_version": lineage["schema_version"], "input_rows": lineage["input_rows"]},
        "artifacts": {"dataset.parquet": _sha256_file(output_path), "translation_qa.json": _sha256_file(qa_summary_path), "translation_qa_flags.parquet": _sha256_file(qa_flags_path)},
        "columns": list(frame.columns),
        "translation_contract": {"translated": ["claim", "chunk_i where evidence_mask[i] is true"], "preserved": "all other cells, row order, columns, IDs, labels, masks, splits and retrieval metadata", "english_chunk_hashes_and_offsets": "preserved as provenance of the canonical EN source; not recomputed after translation", "retrieval_rerun": False, "top_k_preserved": True, "labels_preserved": True, "row_order_preserved": True, "fallback_to_source_text": False},
        "counts": {"input_rows": lineage["input_rows"], "output_rows": int(len(frame)), "unique_texts": len(texts), "valid_evidence_texts": int(sum(sum(bool(value) for value in list(mask)) for mask in frame["evidence_mask"]))},
        "cache": {"path": str(cache_path), "hits": cache_hits, "misses": cache_misses},
        "qa": qa_summary,
        "input_signature": input_signature,
        "config_signature": config_signature,
        "selection_signature": selection_signature,
    }
    _write_text_atomic(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    split_check = _assert_same_split_assignments(config, output_path, manifest_path)
    if split_check is not None:
        manifest["split_assignment_validation"] = split_check
    manifest["status"] = "completed"
    _write_text_atomic(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def validate_translation_input(config: TranslationConfig) -> dict[str, Any]:
    frame, lineage, input_signature = _parquet_rows(config)
    return {"status": "valid", "format": "parquet", "model_loaded": False, "rows": len(frame), "unique_texts": len(_parquet_texts(frame)), "source": lineage, "input_signature": input_signature}


def translate_ragtruth(
    config: TranslationConfig,
    translator: BatchTranslator | None = None,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    return _translate_parquet(config, translator, resume=resume)
