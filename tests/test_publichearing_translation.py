from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
from pathlib import Path

from ragtruth_transfer.translation import MADLADTranslator, TranslationConfig, TranslationSettings, translate_ragtruth, validate_publichearing_translation, validate_translation_input


class FakeTranslator:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def translate_batch(self, texts: list[str]) -> list[str]:
        self.calls.append(texts)
        return [f"EN:{text}" for text in texts]


def _config(tmp_path: Path, source: Path) -> TranslationConfig:
    return TranslationConfig(
        translation=TranslationSettings("nllb", "fake", "por_Latn", "eng_Latn", batch_size=2, device="cpu"),
        input_path=source,
        output_dir=tmp_path / "translated",
        dataset_adapter="publichearing_nli_jsonl",
        output_filename="PublicHearingBR_NLI.jsonl",
        expected_top_level_rows=1,
        expected_modelable_examples=1,
        expected_positives=1,
        expected_hearings=1,
    )


def test_publichearing_adapter_translates_only_valid_fields_and_preserves_structure(tmp_path: Path) -> None:
    source = tmp_path / "snapshot" / "PublicHearingBR_NLI.jsonl"
    source.parent.mkdir()
    record = {
        "id": 7,
        "metadados_extraidos": {
            "assunto": "intacto",
            "tl_dr": "intacto",
            "envolvidos": [{
                "nome": "Pessoa",
                "cargo": "Cargo",
                "opinioes": [
                    {"opiniao": "claim", "chunks_proximos": ["a", "b", "c", "d"], "verificacao_alucinacao": {"verificacao_manual": True}, "extra": 3},
                    {"opiniao": "invalid", "chunks_proximos": ["a", "", "c", "d"], "verificacao_alucinacao": {"verificacao_manual": False}},
                ],
            }],
        },
    }
    source.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    source_bytes = source.read_bytes()
    config = _config(tmp_path, source)
    assert validate_translation_input(config)["model_loaded"] is False
    fake = FakeTranslator()
    manifest = translate_ragtruth(config, translator=fake)
    output = json.loads((config.output_dir / config.output_filename).read_text(encoding="utf-8").splitlines()[0])
    opinions = output["metadados_extraidos"]["envolvidos"][0]["opinioes"]
    assert opinions[0]["opiniao"] == "EN:claim"
    assert opinions[0]["chunks_proximos"] == ["EN:a", "EN:b", "EN:c", "EN:d"]
    assert opinions[0]["verificacao_alucinacao"] == {"verificacao_manual": True}
    assert opinions[0]["extra"] == 3
    assert opinions[1] == record["metadados_extraidos"]["envolvidos"][0]["opinioes"][1]
    assert manifest["alignment_audit"]["status"] == "passed"
    assert manifest["counts"] == {"top_level_rows": 1, "modelable_examples": 1, "positives": 1, "hearings": 1}
    assert sum(len(batch) for batch in fake.calls) == 5
    assert validate_publichearing_translation(config)["status"] == "valid"
    assert source.read_bytes() == source_bytes


def test_publichearing_translation_cache_and_resume_are_model_free(tmp_path: Path) -> None:
    source = tmp_path / "PublicHearingBR_NLI.jsonl"
    record = {"id": "h", "metadados_extraidos": {"envolvidos": [{"opinioes": [{"opiniao": "c", "chunks_proximos": ["a", "b", "c", "d"], "verificacao_alucinacao": {"verificacao_manual": True}}]}]}}
    source.write_text(json.dumps(record) + "\n", encoding="utf-8")
    config = _config(tmp_path, source)
    first = translate_ragtruth(config, translator=FakeTranslator())
    resumed = translate_ragtruth(config, translator=FakeTranslator(), resume=True)
    assert resumed["run_signature"] == first["run_signature"]
    assert resumed["cache"]["misses"] == first["cache"]["misses"]


def test_publichearing_madlad_config_uses_pt_to_en_and_shared_adapter() -> None:
    config = TranslationConfig.from_yaml(Path("configs/publichearing_pt_to_en_madlad.yaml"))
    assert config.dataset_adapter == "publichearing_nli_jsonl"
    assert config.translation.translator == "madlad"
    assert config.translation.model_name == "google/madlad400-3b-mt"
    assert config.translation.source_language == "pt"
    assert config.translation.target_language == "en"
    assert config.output_dir.name == "publichearing_nli_pt_to_en_madlad"


def test_madlad_backend_builds_pt_to_en_target_prefix_without_model() -> None:
    translator = MADLADTranslator.__new__(MADLADTranslator)
    translator.settings = SimpleNamespace(target_language="en")
    captured: list[list[str]] = []
    translator._generate = lambda texts: captured.append(texts) or ["translated"]
    assert translator._translate_batch(["texto português"]) == ["translated"]
    assert captured == [["<2en> texto português"]]
