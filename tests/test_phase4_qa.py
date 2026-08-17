import numpy as np

from qkef.qa import EvidenceQA
from qkef.qa.extractive import EvidenceRecord


def test_qa_returns_valid_evidence_and_provenance():
    records = [EvidenceRecord("mortgage", "Mortgage rates affect monthly payments and total borrowing cost.", ["source-1"])]
    result = EvidenceQA(records).answer("What affects mortgage payments?", minimum_score=0.01)
    assert "Mortgage" in result.answer
    assert result.provenance == [["source-1"]]
    assert result.supporting_passages


def test_qa_returns_insufficient_evidence_without_hallucinating():
    records = [EvidenceRecord("tax", "Tax documentation must be retained.", ["source-2"])]
    result = EvidenceQA(records).answer("How do stars form?", minimum_score=0.2)
    assert result.answer == "Insufficient evidence in the current knowledge base."
    assert not result.supporting_passages


def test_qa_excludes_archived_and_superseded_records():
    records = [
        EvidenceRecord("old", "Obsolete mortgage claim.", ["old"], "superseded"),
        EvidenceRecord("archived", "Archived mortgage claim.", ["archive"], "archived"),
        EvidenceRecord("new", "Current mortgage evidence.", ["new"], "active"),
    ]
    qa = EvidenceQA(records)
    assert [record.identifier for record in qa.records] == ["new"]
