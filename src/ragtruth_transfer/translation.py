from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
import time
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import yaml
import pandas as pd

from .io_utils import read_jsonl
from .ragtruth_parquet import load_ragtruth_parquet, split_ragtruth_parquet, validate_training_view_manifest
from .translation_qa import classify_translation_pair

TRANSLATION_CACHE_SCHEMA = "ragtruth-translation-cache-v1"
TRANSLATION_MANIFEST_SCHEMA = "ragtruth-translated-v1"
PARQUET_TRANSLATION_MANIFEST_SCHEMA = "ragtruth-qa-training-view-deduplicated-v1"
PUBLICHEARING_TRANSLATION_MANIFEST_SCHEMA = "publichearing-nli-translation-v1"


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
    expected_source_revision: str | None = None
    expected_top_level_rows: int | None = None
    expected_modelable_examples: int | None = None
    expected_positives: int | None = None
    expected_hearings: int | None = None
    dataset_adapter: str = "ragtruth_parquet"
    output_filename: str = "dataset.parquet"

    def __post_init__(self) -> None:
        if self.dataset_adapter not in {"ragtruth_parquet", "publichearing_nli_jsonl"}:
            raise ValueError(f"Adapter de dataset desconhecido: {self.dataset_adapter}")
        if not self.output_filename or Path(self.output_filename).name != self.output_filename:
            raise ValueError("data.output_filename deve ser um nome de arquivo simples")
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
            raise ValueError("O pipeline oficial exige data.input_path e data.output_dir")
        dataset_adapter = str(data_raw.get("dataset_adapter", data_raw.get("adapter", "ragtruth_parquet"))).lower()
        default_filename = "PublicHearingBR_NLI.jsonl" if dataset_adapter == "publichearing_nli_jsonl" else "dataset.parquet"
        output_filename = str(data_raw.get("output_filename", default_filename))
        if dataset_adapter == "ragtruth_parquet" and ("input_dir" in data_raw or "splits" in data_raw or "sample_fraction" in data_raw):
            raise ValueError("O pipeline Parquet não aceita input_dir, splits ou sample_fraction")
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
            translation=settings,
            input_path=input_path,
            output_dir=output_dir,
            cache_path=cache_path,
            max_examples_per_split=max_examples_per_split,
            manifest_path=resolve(data_raw.get("manifest_path")),
            expected_source_sha256=str(data_raw["expected_source_sha256"]) if data_raw.get("expected_source_sha256") else None,
            expected_source_signature=str(data_raw["expected_source_signature"]) if data_raw.get("expected_source_signature") else None,
            expected_source_schema=str(data_raw["expected_source_schema"]) if data_raw.get("expected_source_schema") else None,
            expected_source_rows=int(data_raw["expected_source_rows"]) if data_raw.get("expected_source_rows") is not None else None,
            reference_split_assignments=resolve(data_raw.get("reference_split_assignments")),
            expected_split_signature=str(data_raw["expected_split_signature"]) if data_raw.get("expected_split_signature") else None,
            expected_source_revision=str(data_raw["expected_source_revision"]) if data_raw.get("expected_source_revision") else None,
            expected_top_level_rows=int(data_raw["expected_top_level_rows"]) if data_raw.get("expected_top_level_rows") is not None else None,
            expected_modelable_examples=int(data_raw["expected_modelable_examples"]) if data_raw.get("expected_modelable_examples") is not None else None,
            expected_positives=int(data_raw["expected_positives"]) if data_raw.get("expected_positives") is not None else None,
            expected_hearings=int(data_raw["expected_hearings"]) if data_raw.get("expected_hearings") is not None else None,
            dataset_adapter=dataset_adapter,
            output_filename=output_filename,
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


def _config_signature(settings: TranslationSettings, dataset_adapter: str | None = None) -> str:
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
    if dataset_adapter and dataset_adapter != "ragtruth_parquet":
        payload["dataset_adapter"] = dataset_adapter
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
    config_signature = _config_signature(config.translation, config.dataset_adapter)
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


def _publichearing_modelable_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        hearing_id = str(record.get("id", ""))
        metadata = record.get("metadados_extraidos")
        if not isinstance(metadata, dict):
            raise ValueError("Registro PublicHearingBR sem metadados_extraidos")
        people = metadata.get("envolvidos", [])
        if not isinstance(people, list):
            raise ValueError("metadados_extraidos.envolvidos deve ser uma lista")
        for person_index, person in enumerate(people):
            if not isinstance(person, dict):
                raise ValueError("Envolvido PublicHearingBR inválido")
            opinions = person.get("opinioes", [])
            if not isinstance(opinions, list):
                raise ValueError("envolvidos[].opinioes deve ser uma lista")
            for opinion_index, opinion in enumerate(opinions):
                if not isinstance(opinion, dict):
                    raise ValueError("Opinião PublicHearingBR inválida")
                chunks = opinion.get("chunks_proximos") or []
                claim = opinion.get("opiniao", "")
                if not isinstance(chunks, list) or len(chunks) != 4:
                    continue
                if not isinstance(claim, str) or not claim.strip() or any(not isinstance(chunk, str) or not chunk.strip() for chunk in chunks):
                    continue
                rows.append({
                    "example_id": f"{hearing_id}:{person_index}:{opinion_index}",
                    "hearing_id": hearing_id,
                    "person_index": person_index,
                    "opinion_index": opinion_index,
                    "claim": claim,
                    "chunks": list(chunks),
                    "label": int(bool((opinion.get("verificacao_alucinacao") or {}).get("verificacao_manual"))),
                })
    return rows


def _publichearing_source(config: TranslationConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], str]:
    if not config.input_path.is_file():
        raise FileNotFoundError(f"PublicHearingBR não encontrado: {config.input_path}")
    source_sha = _sha256_file(config.input_path)
    if config.expected_source_sha256 and source_sha != config.expected_source_sha256:
        raise ValueError("SHA-256 do PublicHearingBR de origem diverge da configuração")
    records = read_jsonl(config.input_path)
    if not records:
        raise ValueError("PublicHearingBR de origem vazio")
    ids = [str(record.get("id", "")) for record in records]
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("IDs de hearings PublicHearingBR ausentes ou duplicados")
    rows = _publichearing_modelable_rows(records)
    positives = int(sum(row["label"] for row in rows))
    hearings = len({row["hearing_id"] for row in rows})
    counts = {
        "top_level_rows": len(records),
        "modelable_examples": len(rows),
        "positives": positives,
        "hearings": hearings,
        "unique_example_ids": len({row["example_id"] for row in rows}),
    }
    expected = {
        "top_level_rows": config.expected_top_level_rows,
        "modelable_examples": config.expected_modelable_examples,
        "positives": config.expected_positives,
        "hearings": config.expected_hearings,
    }
    for key, value in expected.items():
        if value is not None and counts[key] != value:
            raise ValueError(f"Contagem PublicHearingBR inesperada em {key}: {counts[key]} != {value}")
    lineage = {
        "path": str(config.input_path.resolve()),
        "sha256": source_sha,
        "revision": config.expected_source_revision,
        **counts,
    }
    input_signature = _input_signature({config.input_path.name: source_sha}, ())
    return records, rows, lineage, input_signature


def _publichearing_texts(rows: list[dict[str, Any]]) -> list[str]:
    texts: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for text in [row["claim"], *row["chunks"]]:
            if text not in seen:
                seen.add(text)
                texts.append(text)
    return texts


def _publichearing_pairs(records: list[dict[str, Any]]) -> list[tuple[str, str, int | None, str, str]]:
    pairs: list[tuple[str, str, int | None, str, str]] = []
    for row in _publichearing_modelable_rows(records):
        pairs.append((row["example_id"], "claim", None, row["claim"], row["claim"]))
        pairs.extend((row["example_id"], f"chunk_{index}", index, text, text) for index, text in enumerate(row["chunks"], start=1))
    return pairs


def _qa_pairs(pairs: list[tuple[str, str, int | None, str, str]]) -> tuple[dict[str, Any], pd.DataFrame]:
    flags: list[dict[str, Any]] = []
    counts = {"low_ratio": 0, "high_ratio": 0, "high_confidence_repetition": 0, "empty_translation": 0, "control_character_issue": 0, "exclusion_candidate": 0}
    for example_id, field, slot, original, rendered in pairs:
        metrics = classify_translation_pair(original, rendered)
        for name in counts:
            counts[name] += int(bool(metrics[name]))
        if metrics["exclusion_candidate"] or metrics["empty_translation"] or metrics["control_character_issue"]:
            flags.append({"example_id": example_id, "field": field, "slot": slot, **metrics})
    columns = ["example_id", "field", "slot", "source_normalized", "translated_normalized", "source_chars", "translated_chars", "source_words", "translated_words", "length_ratio", "empty_translation", "identical_to_source", "low_ratio", "high_ratio", "source_has_repetition", "translation_has_repetition", "source_repetition_metric", "translation_repetition_metric", "repetition_excess", "repetition_severity", "translation_added_repetition", "high_confidence_repetition", "control_character_issue", "exclusion_candidate"]
    return {"pairs": len(pairs), "flags": counts, "flagged_pairs": len(flags), "rows_removed": 0, "filtering_applied": False}, pd.DataFrame(flags, columns=columns)


def _translate_publichearing_records(records: list[dict[str, Any]], rows: list[dict[str, Any]], translations: dict[str, str]) -> list[dict[str, Any]]:
    result = copy.deepcopy(records)
    for row in _publichearing_modelable_rows(result):
        opinion = result[[str(item.get("id")) for item in result].index(row["hearing_id"])]
        people = opinion["metadados_extraidos"]["envolvidos"]
        entry = people[row["person_index"]]["opinioes"][row["opinion_index"]]
        entry["opiniao"] = translations[_text_hash(row["claim"])]
        entry["chunks_proximos"] = [translations[_text_hash(text)] for text in row["chunks"]]
    return result


def _publichearing_alignment(source: list[dict[str, Any]], translated: list[dict[str, Any]], expected: dict[str, int | None]) -> dict[str, Any]:
    if len(source) != len(translated):
        raise ValueError("Quantidade de hearings alterada pela tradução")
    source_rows = _publichearing_modelable_rows(source)
    translated_rows = _publichearing_modelable_rows(translated)
    if [(row["hearing_id"], row["person_index"], row["opinion_index"]) for row in source_rows] != [(row["hearing_id"], row["person_index"], row["opinion_index"]) for row in translated_rows]:
        raise ValueError("IDs ou ordenação dos exemplos alterados pela tradução")
    translated_by_id = {row["example_id"]: row for row in translated_rows}
    for source_record, translated_record in zip(source, translated):
        if source_record.get("id") != translated_record.get("id"):
            raise ValueError("IDs de hearings alterados pela tradução")
        source_metadata = source_record["metadados_extraidos"]
        translated_metadata = translated_record["metadados_extraidos"]
        source_metadata_untranslated = {key: value for key, value in source_metadata.items() if key != "envolvidos"}
        translated_metadata_untranslated = {key: value for key, value in translated_metadata.items() if key != "envolvidos"}
        if source_metadata_untranslated != translated_metadata_untranslated:
            raise ValueError("Metadados de hearing alterados pela tradução")
        source_people = source_metadata.get("envolvidos", [])
        translated_people = translated_metadata.get("envolvidos", [])
        if len(source_people) != len(translated_people):
            raise ValueError("Número de envolvidos alterado pela tradução")
        for source_person, translated_person in zip(source_people, translated_people):
            source_person_untranslated = {key: value for key, value in source_person.items() if key != "opinioes"}
            translated_person_untranslated = {key: value for key, value in translated_person.items() if key != "opinioes"}
            if source_person_untranslated != translated_person_untranslated:
                raise ValueError("Metadados de envolvidos alterados pela tradução")
            source_opinions = source_person.get("opinioes", [])
            translated_opinions = translated_person.get("opinioes", [])
            if len(source_opinions) != len(translated_opinions):
                raise ValueError("Número de opiniões alterado pela tradução")
            for source_opinion, translated_opinion in zip(source_opinions, translated_opinions):
                if source_opinion.get("verificacao_alucinacao") != translated_opinion.get("verificacao_alucinacao"):
                    raise ValueError("Labels alterados pela tradução")
                source_opinion_untranslated = {key: value for key, value in source_opinion.items() if key not in {"opiniao", "chunks_proximos"}}
                translated_opinion_untranslated = {key: value for key, value in translated_opinion.items() if key not in {"opiniao", "chunks_proximos"}}
                if source_opinion_untranslated != translated_opinion_untranslated:
                    raise ValueError("Campos de opinião alterados pela tradução")
                source_chunks = source_opinion.get("chunks_proximos")
                translated_chunks = translated_opinion.get("chunks_proximos")
                if not isinstance(source_chunks, list) or source_chunks != translated_chunks and len(source_chunks) != len(translated_chunks or []):
                    raise ValueError("Estrutura de chunks alterada pela tradução")
                if len(source_chunks) != 4 or not str(source_opinion.get("opiniao", "")).strip() or any(not isinstance(chunk, str) or not chunk.strip() for chunk in source_chunks):
                    if source_opinion != translated_opinion:
                        raise ValueError("Entrada inválida/empty foi alterada pela tradução")
    for row in source_rows:
        candidate = translated_by_id[row["example_id"]]
        if not str(candidate["claim"]).strip() or any(not str(chunk).strip() for chunk in candidate["chunks"]):
            raise ValueError("Tradução gerou campo vazio")
    counts = {"top_level_rows": len(translated), "modelable_examples": len(translated_rows), "positives": sum(row["label"] for row in translated_rows), "hearings": len({row["hearing_id"] for row in translated_rows}), "unique_example_ids": len(translated_by_id)}
    for key, value in expected.items():
        if value is not None and counts[key] != value:
            raise ValueError(f"Contagem de alinhamento inesperada em {key}: {counts[key]} != {value}")
    return {"status": "passed", "top_level_order_preserved": True, "metadata_order_preserved": True, "labels_preserved": True, "invalid_slots_preserved": True, "chunk_order_preserved": True, "counts": counts}


def validate_publichearing_translation(config: TranslationConfig) -> dict[str, Any]:
    if config.dataset_adapter != "publichearing_nli_jsonl":
        raise ValueError("validate_publichearing_translation exige o adapter JSONL PublicHearingBR")
    source, rows, lineage, input_signature = _publichearing_source(config)
    output_path = config.output_dir / config.output_filename
    manifest_path = config.output_dir / "manifest.json"
    if not output_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("Output ou manifesto da tradução PublicHearingBR não encontrado")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError("Manifesto da tradução PublicHearingBR não está completed")
    if _sha256_file(output_path) != manifest.get("output", {}).get("sha256"):
        raise ValueError("SHA do output não coincide com o manifesto")
    translated = read_jsonl(output_path)
    audit = _publichearing_alignment(source, translated, {"top_level_rows": config.expected_top_level_rows, "modelable_examples": config.expected_modelable_examples, "positives": config.expected_positives, "hearings": config.expected_hearings})
    return {"status": "valid", "model_loaded": False, "source": lineage, "output": {"path": str(output_path), "sha256": _sha256_file(output_path)}, "alignment_audit": audit, "manifest": manifest}


def _translate_publichearing(config: TranslationConfig, translator: BatchTranslator | None, *, resume: bool) -> dict[str, Any]:
    source, rows, lineage, input_signature = _publichearing_source(config)
    config_signature, selection_signature, run_signature = _translation_signature(config, input_signature)
    output_path = config.output_dir / config.output_filename
    manifest_path = config.output_dir / "manifest.json"
    if output_path.exists() or manifest_path.exists():
        if not resume:
            raise FileExistsError(f"Output JSONL já existe: {config.output_dir}. Use --resume apenas para output compatível.")
        existing = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
        if existing.get("run_signature") != run_signature or existing.get("status") != "completed":
            raise ValueError("Output JSONL existente é incompatível ou incompleto")
        return existing
    texts = _publichearing_texts(rows)
    hashes = {_text_hash(text): text for text in texts}
    cache_path = config.cache_path or config.output_dir / ".translation_cache.sqlite3"
    translations: dict[str, str] = {}
    cache_hits = 0
    cache_misses = 0
    active: BatchTranslator | None = None
    with TranslationCache(cache_path, config_signature, input_signature) as cache:
        translations.update(cache.get_many(list(hashes)))
        cache_hits = len(translations)
        missing = [text for text in texts if _text_hash(text) not in translations]
        cache_misses = len(missing)
        if missing:
            active = translator or create_translator(config.translation)
            for start in range(0, len(missing), config.translation.batch_size):
                batch = missing[start : start + config.translation.batch_size]
                rendered = active.translate_batch(batch)
                if len(rendered) != len(batch) or any(not isinstance(value, str) or not value.strip() for value in rendered):
                    raise RuntimeError("Tradutor retornou resultado incompleto ou vazio")
                values = {_text_hash(source_text): target for source_text, target in zip(batch, rendered)}
                cache.put_many(values)
                translations.update(values)
    if set(hashes) != set(translations):
        raise RuntimeError("Traduções ausentes; nenhuma cópia silenciosa do texto fonte é permitida")
    translated = _translate_publichearing_records(source, rows, translations)
    audit = _publichearing_alignment(source, translated, {"top_level_rows": config.expected_top_level_rows, "modelable_examples": config.expected_modelable_examples, "positives": config.expected_positives, "hearings": config.expected_hearings})
    output_text = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in translated)
    _write_text_atomic(output_path, output_text)
    qa_pairs = []
    for source_row, translated_row in zip(_publichearing_modelable_rows(source), _publichearing_modelable_rows(translated)):
        qa_pairs.append((source_row["example_id"], "claim", None, source_row["claim"], translated_row["claim"]))
        qa_pairs.extend((source_row["example_id"], f"chunk_{index}", index, original, rendered) for index, (original, rendered) in enumerate(zip(source_row["chunks"], translated_row["chunks"]), start=1))
    qa_summary, qa_flags = _qa_pairs(qa_pairs)
    qa_path = config.output_dir / "translation_qa.json"
    flags_path = config.output_dir / "translation_qa_flags.parquet"
    alignment_path = config.output_dir / "alignment_audit.json"
    _write_text_atomic(qa_path, json.dumps(qa_summary, ensure_ascii=False, indent=2) + "\n")
    _write_parquet_atomic(flags_path, qa_flags)
    _write_text_atomic(alignment_path, json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    manifest = {
        "schema_version": PUBLICHEARING_TRANSLATION_MANIFEST_SCHEMA,
        "status": "completed",
        "signature": run_signature[:16],
        "run_signature": run_signature,
        "dataset_adapter": config.dataset_adapter,
        "backend": config.translation.translator,
        "model_name": config.translation.model_name,
        "requested_model_revision": config.translation.model_revision,
        **_runtime_manifest_fields(active),
        "source_language": config.translation.source_language,
        "target_language": config.translation.target_language,
        "target_prefix": f"<2{config.translation.target_language}>" if config.translation.translator == "madlad" else None,
        "source_revision": config.expected_source_revision,
        "generation": {"num_beams": config.translation.num_beams, "max_input_tokens": config.translation.max_input_tokens, "max_new_tokens": config.translation.max_new_tokens, "batch_size": config.translation.batch_size, "sampling": False},
        "source": lineage,
        "output": {"path": str(output_path), "sha256": _sha256_file(output_path)},
        "counts": {key: lineage[key] for key in ("top_level_rows", "modelable_examples", "positives", "hearings")},
        "translated_fields": ["metadados_extraidos.envolvidos[].opinioes[].opiniao", "metadados_extraidos.envolvidos[].opinioes[].chunks_proximos[i] for valid four-slot entries"],
        "translation_contract": {"retrieval_rerun": False, "chunking_rerun": False, "labels_preserved": True, "hearing_ids_preserved": True, "invalid_slots_preserved": True, "fallback_to_source_text": False},
        "cache": {"path": str(cache_path), "hits": cache_hits, "misses": cache_misses, "config_signature": config_signature, "input_signature": input_signature},
        "qa": qa_summary,
        "alignment_audit": {"path": str(alignment_path), "sha256": _sha256_file(alignment_path), **audit},
        "artifacts": {config.output_filename: _sha256_file(output_path), "translation_qa.json": _sha256_file(qa_path), "translation_qa_flags.parquet": _sha256_file(flags_path), "alignment_audit.json": _sha256_file(alignment_path)},
    }
    _write_text_atomic(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def validate_translation_input(config: TranslationConfig) -> dict[str, Any]:
    if config.dataset_adapter == "publichearing_nli_jsonl":
        source, rows, lineage, input_signature = _publichearing_source(config)
        output_path = config.output_dir / config.output_filename
        manifest_path = config.output_dir / "manifest.json"
        return {"status": "valid", "format": "jsonl", "dataset_adapter": config.dataset_adapter, "model_loaded": False, "rows": lineage["top_level_rows"], "modelable_examples": len(rows), "unique_texts": len(_publichearing_texts(rows)), "source": lineage, "input_signature": input_signature, "output": {"path": str(output_path), "collision": output_path.exists() or manifest_path.exists(), "manifest_path": str(manifest_path)}}
    frame, lineage, input_signature = _parquet_rows(config)
    return {"status": "valid", "format": "parquet", "model_loaded": False, "rows": len(frame), "unique_texts": len(_parquet_texts(frame)), "source": lineage, "input_signature": input_signature}


def translate_ragtruth(
    config: TranslationConfig,
    translator: BatchTranslator | None = None,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    if config.dataset_adapter == "publichearing_nli_jsonl":
        return _translate_publichearing(config, translator, resume=resume)
    return _translate_parquet(config, translator, resume=resume)
