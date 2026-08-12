import json

from ragtruth_transfer.publichearing.data import normalize_publichearing


def _record(opinions, hearing="h1"):
    return {"id": hearing, "metadados_extraidos": {"assunto": "tema", "envolvidos": [{"nome": "Pessoa", "cargo": "Cargo", "opinioes": opinions}]}}


def test_normalizes_complete_four_chunk_bag_and_rejects_invalid_rows(tmp_path):
    rows = [_record([
        {"opiniao": "opinião", "chunks_proximos": ["a", "b", "c", "d"], "verificacao_alucinacao": {"verificacao_manual": True}},
        {"opiniao": "", "chunks_proximos": ["a", "b", "c"], "verificacao_alucinacao": {}},
    ])]
    path = tmp_path / "data.jsonl"; path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    accepted, rejected, audit = normalize_publichearing(path, validate_expected=False)
    assert accepted.loc[0, "sample_id"] == "h1:0:0"
    assert accepted.loc[0, "context_chunks"] == ["a", "b", "c", "d"]
    assert accepted.loc[0, "label"] == 1
    assert set(rejected.loc[0, "rejection_reason"].split("|")) == {"empty_opinion", "chunk_count_3", "missing_manual_label"}
    assert audit["modelable_examples"] == 1
