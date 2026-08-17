"""Package boundary smoke tests."""


def test_package_imports() -> None:
    import qkef
    import qkef.chunking
    import qkef.chunking.fixed_window
    import qkef.chunking.identity
    import qkef.chunking.tfidf_boundary
    import qkef.datasets
    import qkef.datasets.evolution
    import qkef.datasets.fiqa
    import qkef.embeddings
    import qkef.evaluation
    import qkef.evolution
    import qkef.graph
    import qkef.ingestion
    import qkef.ingestion.corpus
    import qkef.quantum_inspired
    import qkef.retrieval
    import qkef.schemas

    assert qkef.__version__ == "0.1.0"
