from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import torch

from ragtruth_transfer.ragtruth_zero_shot import (
    ZeroShotConfig,
    _infer,
    load_publichearing_inputs,
    run_zero_shot,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def _source_fixture(
    tmp_path: Path,
    architecture: str = "gated_attention",
) -> tuple[Path, Path]:
    run = tmp_path / "run"
    checkpoint = run / "checkpoints" / "epoch_03"
    adapter = checkpoint / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text(json.dumps({"r": 8, "lora_alpha": 16, "lora_dropout": 0.1, "target_modules": ["query_proj", "value_proj"]}), encoding="utf-8")
    torch.save({"projection.0.weight": torch.zeros((2, 2)), "projection.0.bias": torch.zeros(2), "classifier.1.weight": torch.zeros((1, 2)), "classifier.1.bias": torch.zeros(1)}, checkpoint / "head.pt")
    validation = checkpoint / "validation_predictions.csv"
    pd.DataFrame({"label": [0, 1, 0, 1], "score": [0.1, 0.9, 0.2, 0.8]}).to_csv(validation, index=False)
    artifacts = {}
    for path in (adapter / "adapter_config.json", checkpoint / "head.pt", validation):
        artifacts[str(path.relative_to(checkpoint))] = hashlib.sha256(path.read_bytes()).hexdigest()
    checkpoint_manifest = {"format_version": 2, "resumable": True, "epoch": 3, "model_revision": "b5113eb38ab63efdd7f280f8c144ea8b13f978ce", "data_split_sha256": {"split": "525edec2966a4fac"}, "artifacts": artifacts}
    (checkpoint / "checkpoint_manifest.json").write_text(json.dumps(checkpoint_manifest), encoding="utf-8")
    (checkpoint / "CHECKPOINT_COMPLETE").write_text(hashlib.sha256((checkpoint / "checkpoint_manifest.json").read_bytes()).hexdigest() + "\n", encoding="ascii")
    model_config = {"run_name": "ragtruth_lora_attention_mil_parquet", "model_id": "dummy", "model_revision": "b5113eb38ab63efdd7f280f8c144ea8b13f978ce", "architecture": architecture, "pooling_type": "attention" if architecture == "gated_attention" else architecture, "encoder_mode": "lora", "max_length": 8, "projection_size": 2, "attention_size": 2, "dropout": 0.0, "lora": {"r": 8, "alpha": 16, "dropout": 0.1, "target_modules": ["query_proj", "value_proj"]}, "training": {"max_epochs": 3, "planned_total_epochs": 3, "train_batch_size": 1, "eval_batch_size": 1, "gradient_accumulation_steps": 1, "head_learning_rate": 1e-3, "encoder_learning_rate": 1e-3, "save_epoch_checkpoints": True}}
    run_manifest = {"status": "completed", "config": model_config, "config_fingerprint": "source-signature", "best_epoch": 3, "best_checkpoint": "checkpoints/epoch_03", "thresholds": None, "dataset": {"signature": "0cdf598fa866741d", "schema_version": "ragtruth-qa-training-view-deduplicated-v1"}, "data_split_sha256": {"split": "525edec2966a4fac"}}
    (run / "run_manifest.json").write_text(json.dumps(run_manifest), encoding="utf-8")
    return run, checkpoint


def _public_fixture(tmp_path: Path) -> Path:
    path = tmp_path / "PublicHearingBR_NLI.jsonl"
    record = {"id": 1, "metadados_extraidos": {"envolvidos": [{"nome": "p", "opinioes": [{"opiniao": "claim", "chunks_proximos": ["a", "b", "c", "d"], "verificacao_alucinacao": {"verificacao_manual": False}}]}]}}
    _write_jsonl(path, [record])
    return path


def test_zero_shot_validate_only_is_model_free_and_uses_four_slots(tmp_path: Path) -> None:
    run, checkpoint = _source_fixture(tmp_path)
    public = _public_fixture(tmp_path)
    config = ZeroShotConfig(ragtruth_run_dir=run, publichearing_path=public, expected_publichearing_examples=1, expected_publichearing_positives=0, publichearing_dataset_sha256=hashlib.sha256(public.read_bytes()).hexdigest(), output_root=tmp_path / "out")
    result = run_zero_shot(config, validate_only=True)
    assert result["model_loaded"] is False
    assert result["inference_executed"] is False
    assert result["checkpoint_epoch"] == 3
    assert load_publichearing_inputs(config)[0]["evidence_mask"] == [True, True, True, True]


def test_zero_shot_validation_accepts_mean_and_max_source_models(tmp_path: Path) -> None:
    public = _public_fixture(tmp_path)
    for architecture in ("mean", "max"):
        run, _ = _source_fixture(tmp_path / architecture, architecture)
        config = ZeroShotConfig(
            ragtruth_run_dir=run,
            publichearing_path=public,
            expected_publichearing_examples=1,
            expected_publichearing_positives=0,
            publichearing_dataset_sha256=hashlib.sha256(public.read_bytes()).hexdigest(),
            output_root=tmp_path / f"out_{architecture}",
        )
        result = run_zero_shot(config, validate_only=True)
        assert result["pooling_type"] == architecture


def test_infer_does_not_change_model_and_has_normalized_attention() -> None:
    class DummyTokenizer:
        def __call__(self, premises, claims, **kwargs):
            return {"input_ids": torch.ones((len(premises), 3), dtype=torch.long), "attention_mask": torch.ones((len(premises), 3), dtype=torch.long)}

    class DummyModel(torch.nn.Module):
        def forward(self, input_ids, attention_mask, evidence_mask):
            return torch.zeros(input_ids.shape[0]), evidence_mask.float() / evidence_mask.sum(dim=1, keepdim=True)

    rows = [{"example_id": "1:0:0", "source_id": "1", "claim": "c", "evidence": ["a", "b", "c", "d"], "evidence_mask": [True] * 4, "label": 0, "task_type": "PublicHearingBR"}]
    model = DummyModel()
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    scores, attention = _infer(model, DummyTokenizer(), rows, 1, 8, torch.device("cpu"))
    assert scores.tolist() == [0.5]
    assert attention.tolist() == [[0.25, 0.25, 0.25, 0.25]]
    assert not model.training
    assert all(torch.equal(value, before[key]) for key, value in model.state_dict().items())
