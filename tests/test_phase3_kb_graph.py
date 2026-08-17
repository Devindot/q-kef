import numpy as np

from qkef.evolution.knowledge_base import EvolutionKnowledgeBase, KBRecord
from qkef.graph.semantic_graph import SemanticGraph


def record(identifier, text="text"):
    return KBRecord(identifier, text, np.array([1.0, 0.0]), [identifier])


def test_replace_archives_target_from_active_search():
    kb = EvolutionKnowledgeBase(2); kb.add(record("old"))
    kb.execute("REPLACE", record("new"), ["old"])
    assert kb.records["old"].status == "superseded"
    assert [item[0] for item in kb.search(np.array([1.0, 0.0]), 5)] == ["new"]


def test_merge_preserves_provenance_and_sources():
    kb = EvolutionKnowledgeBase(2); kb.add(record("old", "one"))
    result = kb.execute("MERGE", record("new", "two"), ["old"])
    merged = kb.records[result[0]]
    assert merged.provenance == ["old", "new"] and set(merged.source_ids) == {"old", "new"}


def test_archive_coexist_and_split_semantics():
    kb = EvolutionKnowledgeBase(2); kb.add(record("old"))
    kb.execute("COEXIST", record("peer"), ["old"])
    kb.execute("ARCHIVE", record("notice"), ["old"])
    children = kb.execute("SPLIT", record("parent"), [], split_texts=["first", "second"])
    assert kb.records["old"].status == "archived" and len(children) == 2
    assert kb.graph.graph.number_of_edges() >= 5


def test_graph_round_trip(tmp_path):
    graph = SemanticGraph(); graph.add_node("a", status="active"); graph.add_node("b"); graph.add_edge("a", "b", "supersedes")
    path = tmp_path / "graph.json"; graph.save(path); loaded = SemanticGraph.load(path)
    assert loaded.graph.nodes["a"]["status"] == "active"
    assert loaded.graph.edges["a", "b", 0]["relation"] == "supersedes"
