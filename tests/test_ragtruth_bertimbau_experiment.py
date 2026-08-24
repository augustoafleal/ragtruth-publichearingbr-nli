from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from transformers import BertConfig, BertModel, BertTokenizer, get_linear_schedule_with_warmup

from ragtruth_transfer.dataset import BagCollator
from ragtruth_transfer.modeling import (
    HierarchicalEncoderClassifier,
    _validate_lora_targets,
)
from ragtruth_transfer.ragtruth_confirmatory import ConfirmatoryConfig


ROOT = Path(__file__).parents[1]
BASELINE_CONFIG = ROOT / "configs/ragtruth_pt_nllb_filtered_lora_attention_mil_confirmatory.yaml"
BERTIMBAU_CONFIG = ROOT / "configs/ragtruth_pt_nllb_bertimbau_lora_attention_mil_confirmatory.yaml"
EXPECTED_REVISION = "74364c8dbc30e651fee36aa714a772dcaae83815"
EXPECTED_DATASET_SIGNATURE = "ee8b7c597e51e120"
EXPECTED_SPLIT_SIGNATURE = "611cd01cd5d41e18"


def _configs() -> tuple[ConfirmatoryConfig, ConfirmatoryConfig]:
    return ConfirmatoryConfig.from_yaml(BASELINE_CONFIG), ConfirmatoryConfig.from_yaml(BERTIMBAU_CONFIG)


def _tiny_bert() -> BertModel:
    return BertModel(
        BertConfig(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=32,
            type_vocab_size=2,
        )
    )


def _bert_lora_model() -> HierarchicalEncoderClassifier:
    encoder = _tiny_bert()
    _validate_lora_targets(encoder, ("query", "value"))
    encoder = get_peft_model(
        encoder,
        LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=8,
            lora_alpha=16,
            lora_dropout=0.10,
            target_modules=["query", "value"],
            bias="none",
        ),
    )
    return HierarchicalEncoderClassifier(
        encoder=encoder,
        hidden_size=32,
        projection_size=16,
        dropout=0.30,
        architecture="gated_attention",
        attention_size=8,
    )


def test_bertimbau_config_is_a_new_protocol_and_baseline_is_unchanged() -> None:
    baseline, bertimbau = _configs()
    base = baseline.experiment
    candidate = bertimbau.experiment

    assert base.model_id == "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    assert base.model_revision == "b5113eb38ab63efdd7f280f8c144ea8b13f978ce"
    assert base.lora.target_modules == ("query_proj", "value_proj")
    assert candidate.model_id == "neuralmind/bert-base-portuguese-cased"
    assert candidate.model_revision == EXPECTED_REVISION
    assert candidate.lora.target_modules == ("query", "value")
    assert bertimbau.output_root != baseline.output_root
    assert bertimbau.protocol_name != baseline.protocol_name

    for left, right in (
        (base.dataset.path, candidate.dataset.path),
        (base.dataset.manifest_path, candidate.dataset.manifest_path),
    ):
        assert left == right
    assert candidate.dataset.expected_signature == EXPECTED_DATASET_SIGNATURE
    assert bertimbau.expected_split_signature == EXPECTED_SPLIT_SIGNATURE
    assert candidate.canonical_pooling_type == base.canonical_pooling_type == "attention"
    assert candidate.max_length == base.max_length == 512
    assert candidate.truncation == base.truncation == "longest_first"
    assert candidate.training.max_epochs == base.training.max_epochs == 6
    assert candidate.training.planned_total_epochs == base.training.planned_total_epochs == 6
    assert candidate.training.early_stopping_patience == base.training.early_stopping_patience == 2
    assert candidate.training.train_batch_size == base.training.train_batch_size == 1
    assert candidate.training.eval_batch_size == base.training.eval_batch_size == 4
    assert candidate.training.gradient_accumulation_steps == base.training.gradient_accumulation_steps == 16
    assert candidate.training.encoder_learning_rate == base.training.encoder_learning_rate == 2.0e-4
    assert candidate.training.head_learning_rate == base.training.head_learning_rate == 5.0e-4
    assert candidate.training.weight_decay == base.training.weight_decay == 0.01
    assert candidate.training.warmup_ratio == base.training.warmup_ratio == 0.10
    assert bertimbau.seeds == baseline.seeds == (0, 1, 2)


def test_bert_lora_targets_are_real_and_deberta_targets_fail() -> None:
    encoder = _tiny_bert()
    matches = _validate_lora_targets(encoder, ("query", "value"))
    assert len(matches["query"]) == 2
    assert len(matches["value"]) == 2
    with pytest.raises(RuntimeError, match="Módulos LoRA ausentes"):
        _validate_lora_targets(encoder, ("query_proj", "value_proj"))


def test_synthetic_bert_forward_preserves_pair_fields_and_four_evidence_slots() -> None:
    model = _bert_lora_model()
    input_ids = torch.randint(0, 128, (1, 4, 12))
    attention_mask = torch.ones((1, 4, 12), dtype=torch.long)
    token_type_ids = torch.zeros((1, 4, 12), dtype=torch.long)
    token_type_ids[:, :, 6:] = 1
    evidence_mask = torch.tensor([[True, True, True, False]])
    logits, weights = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        evidence_mask=evidence_mask,
    )
    assert logits.shape == (1,)
    assert weights.shape == (1, 4)
    assert torch.isfinite(logits).all()
    assert torch.allclose(weights.sum(dim=1), torch.ones(1), atol=1e-6)
    assert weights[0, 3].item() == pytest.approx(0.0)


def test_bert_tokenizer_pair_contract_without_pretrained_download(tmp_path: Path) -> None:
    vocab = tmp_path / "vocab.txt"
    vocab.write_text(
        "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nevidência\nclaim\nlonga\n",
        encoding="utf-8",
    )
    tokenizer = BertTokenizer(str(vocab), do_lower_case=False)
    rows = [{
        "evidence": ["evidência longa", "evidência", "evidência", ""],
        "claim": "claim longa",
        "evidence_mask": [True, True, True, False],
        "label": 0,
        "example_id": "example",
        "source_id": "source",
    }]
    batch = BagCollator(tokenizer, max_length=8, truncation="longest_first")(rows)
    assert batch["input_ids"].shape[:2] == (1, 4)
    assert batch["input_ids"].shape[-1] <= 8
    assert batch["token_type_ids"].shape == batch["input_ids"].shape
    assert batch["attention_mask"].shape == batch["input_ids"].shape
    assert batch["token_type_ids"][0, 0].tolist().count(1) > 0
    assert batch["evidence_mask"].tolist() == [[True, True, True, False]]


def test_scheduler_protocol_has_same_horizon_and_optimizer_steps() -> None:
    baseline, bertimbau = _configs()
    base = baseline.experiment.training
    candidate = bertimbau.experiment.training
    train_examples = 25_347
    steps_baseline = math.ceil(math.ceil(train_examples / base.train_batch_size) / base.gradient_accumulation_steps)
    steps_candidate = math.ceil(math.ceil(train_examples / candidate.train_batch_size) / candidate.gradient_accumulation_steps)
    assert steps_baseline == steps_candidate == 1_585
    total_baseline = steps_baseline * base.planned_total_epochs
    total_candidate = steps_candidate * candidate.planned_total_epochs
    warmup_baseline = int(total_baseline * base.warmup_ratio)
    warmup_candidate = int(total_candidate * candidate.warmup_ratio)
    assert total_baseline == total_candidate == 9_510
    assert warmup_baseline == warmup_candidate == 951

    parameter = torch.nn.Parameter(torch.ones(1))
    schedules = []
    for settings in (base, candidate):
        optimizer = AdamW([parameter], lr=settings.head_learning_rate, weight_decay=settings.weight_decay)
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_baseline, total_baseline)
        values = []
        for _ in range(steps_baseline * 3):
            values.append(tuple(group["lr"] for group in optimizer.param_groups))
            optimizer.step()
            scheduler.step()
        schedules.append(values)
    assert schedules[0] == schedules[1]
