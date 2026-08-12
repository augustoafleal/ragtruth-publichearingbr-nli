from ragtruth_transfer.ragtruth_data import (
    merge_sentence_spans_for_labels,
    parse_qa_passages,
    response_claims,
    select_top_evidence,
    sentence_spans,
)


def test_sentence_spans_and_cross_sentence_label_merge():
    text = "First fact. Second hallucinated sentence. Third fact."
    labels = [{"start": 7, "end": 25, "text": text[7:25]}]
    merged = merge_sentence_spans_for_labels(text, labels)
    assert merged[0] == (0, text.index(" Third"))
    assert merged[1][0] > merged[0][1]


def test_response_claims_labels_overlap():
    text = "Supported claim. Invented claim."
    start = text.index("Invented")
    response = {
        "response": text,
        "labels": [{"start": start, "end": len(text), "text": text[start:]}],
    }
    claims = response_claims(response, granularity="claim", min_words=2)
    assert [row["label"] for row in claims] == [False, True]


def test_parse_qa_passages():
    value = "passage 1: alpha\n\n passage 2: beta\n\npassage 3: gamma"
    assert parse_qa_passages(value) == ["alpha", "beta", "gamma"]


def test_select_top_evidence_pads():
    evidence, mask = select_top_evidence("cats", ["cats are mammals", "dogs bark"], top_k=4)
    assert len(evidence) == 4
    assert mask == [True, True, False, False]
