import pandas as pd
import torch

from ragtruth_transfer.publichearing.config import PublicHearingConfig
from ragtruth_transfer.publichearing.tokenization import build_or_load_token_cache, make_loader


class Tokenizer:
    def __call__(self, premises, hypotheses, **kwargs):
        width = kwargs["max_length"]
        return {"input_ids": torch.ones((len(premises), width), dtype=torch.long), "attention_mask": torch.ones((len(premises), width), dtype=torch.long)}


def test_token_cache_has_compact_four_bag_shapes(tmp_path, monkeypatch):
    monkeypatch.setattr("ragtruth_transfer.publichearing.tokenization.AutoTokenizer.from_pretrained", lambda *args, **kwargs: Tokenizer())
    frame = pd.DataFrame({"sample_id": ["a", "b"], "opinion": ["one", "two"], "context_chunks": [["a", "b", "c", "d"], ["e", "f", "g", "h"]], "label": [0, 1]})
    config = PublicHearingConfig(mode="smoke", outer_splits=2, inner_splits=2, max_epochs=1, seeds=(101,), max_length=7)
    cache, _ = build_or_load_token_cache(frame, "dataset", config, tmp_path)
    assert cache["input_ids"].shape == (2, 4, 7)
    batch = next(iter(make_loader(cache, [0, 1], 2, False)))
    assert batch["input_ids"].dtype == torch.long and batch["input_ids"].shape == (2, 4, 7)
