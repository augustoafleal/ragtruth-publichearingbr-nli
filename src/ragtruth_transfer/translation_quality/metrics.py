from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from ..translation_qa import classify_translation_pair
from .config import resolve_device

Pair = tuple[str, str]


def _resolve_comet_checkpoint(snapshot_dir: str | Path) -> Path:
    """Find the checkpoint file inside a Hugging Face COMET snapshot."""
    snapshot_path = Path(snapshot_dir)
    if snapshot_path.is_file():
        return snapshot_path

    preferred = snapshot_path / "checkpoints" / "model.ckpt"
    if preferred.is_file():
        return preferred

    candidates = sorted(snapshot_path.rglob("*.ckpt"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"Nenhum checkpoint .ckpt encontrado no snapshot COMET: {snapshot_path}"
        )
    raise RuntimeError(
        "Snapshot COMET contém múltiplos checkpoints e nenhum é o esperado "
        f"({preferred}): {', '.join(str(path) for path in candidates)}"
    )


def _clean(text: Any) -> str:
    return "" if text is None else str(text)


class HeuristicScorer:
    name = "heuristics"

    def score_pairs(self, pairs: Sequence[Pair]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for source, translated in pairs:
            metrics = classify_translation_pair(_clean(source), _clean(translated))
            results.append(
                {
                    "heur_length_ratio": metrics["length_ratio"],
                    "heur_repetition_severity": metrics["repetition_severity"],
                    "heur_empty_translation": bool(metrics["empty_translation"]),
                    "heur_identical_to_source": bool(metrics["identical_to_source"]),
                    "heur_low_ratio": bool(metrics["low_ratio"]),
                    "heur_high_ratio": bool(metrics["high_ratio"]),
                    "heur_control_char": bool(metrics["control_character_issue"]),
                    "heur_exclusion_candidate": bool(metrics["exclusion_candidate"]),
                }
            )
        return results


@dataclass
class CometKiwiScorer:
    name = "cometkiwi"
    model: Any
    batch_size: int
    device: str

    @classmethod
    def load(
        cls, model_id: str, revision: str, device: str, batch_size: int
    ) -> "CometKiwiScorer":
        try:
            from comet import load_from_checkpoint
            from huggingface_hub import snapshot_download
        except ImportError as error:  # pragma: no cover - optional extra
            raise RuntimeError(
                "COMETKiwi requer unbabel-comet. Instale com `pip install -e '.[quality]'`."
            ) from error
        checkpoint_dir = snapshot_download(repo_id=model_id, revision=revision)
        checkpoint_path = _resolve_comet_checkpoint(checkpoint_dir)
        model = load_from_checkpoint(str(checkpoint_path))
        return cls(model=model, batch_size=batch_size, device=resolve_device(device))

    def score_pairs(self, pairs: Sequence[Pair]) -> list[dict[str, Any]]:
        data = [{"src": _clean(s), "mt": _clean(t)} for s, t in pairs]
        gpus = 1 if self.device.startswith("cuda") else 0
        output = self.model.predict(
            data, batch_size=self.batch_size, gpus=gpus, progress_bar=False
        )
        scores = output["scores"] if isinstance(output, dict) else output.scores
        return [{"cometkiwi": float(value)} for value in scores]


@dataclass
class NLIEntailmentScorer:
    name = "nli_consistency"
    model: Any
    tokenizer: Any
    device: str
    max_length: int
    batch_size: int
    entail_index: int

    @classmethod
    def load(
        cls, model_id: str, revision: str, device: str, batch_size: int, max_length: int
    ) -> "NLIEntailmentScorer":
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as error:  # pragma: no cover - optional extra
            raise RuntimeError("NLI-consistency requer transformers e torch.") from error
        resolved = resolve_device(device)
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        model = AutoModelForSequenceClassification.from_pretrained(model_id, revision=revision)
        model.to(resolved)
        model.eval()
        return cls(
            model=model,
            tokenizer=tokenizer,
            device=resolved,
            max_length=max_length,
            batch_size=batch_size,
            entail_index=_entailment_index(model.config),
        )

    def entailment_probs(self, pairs: Sequence[Pair]) -> list[float]:
        import torch

        probs: list[float] = []
        with torch.inference_mode():
            for start in range(0, len(pairs), self.batch_size):
                batch = pairs[start : start + self.batch_size]
                encoded = self.tokenizer(
                    [_clean(p) for p, _ in batch],
                    [_clean(h) for _, h in batch],
                    truncation=True,
                    max_length=self.max_length,
                    padding=True,
                    return_tensors="pt",
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                logits = self.model(**encoded).logits
                softmax = torch.softmax(logits.float(), dim=-1)
                probs.extend(softmax[:, self.entail_index].cpu().tolist())
        return probs


def _entailment_index(config: Any) -> int:
    id2label = getattr(config, "id2label", None) or {}
    for index, label in id2label.items():
        if str(label).lower().startswith("entail"):
            return int(index)
    return 0


def detector_pair_tokens(tokenizer: Any, premise: str, hypothesis: str) -> int:
    encoded = tokenizer(premise, hypothesis, truncation=False, return_attention_mask=False)
    return len(encoded["input_ids"])


def detector_truncation_flags(
    tokenizer: Any,
    en_pairs: Sequence[Pair],
    pt_pairs: Sequence[Pair],
    max_length: int,
) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []
    for (en_premise, en_hyp), (pt_premise, pt_hyp) in zip(en_pairs, pt_pairs):
        en_tokens = detector_pair_tokens(tokenizer, en_premise, en_hyp)
        pt_tokens = detector_pair_tokens(tokenizer, pt_premise, pt_hyp)
        flags.append(
            {
                "detector_pair_tokens_en": en_tokens,
                "detector_pair_tokens_pt": pt_tokens,
                "detector_truncated_en": bool(en_tokens > max_length),
                "detector_truncated_pt": bool(pt_tokens > max_length),
                "detector_truncation_introduced": bool(
                    pt_tokens > max_length and en_tokens <= max_length
                ),
            }
        )
    return flags
