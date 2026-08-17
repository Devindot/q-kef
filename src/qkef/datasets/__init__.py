"""Dataset acquisition, loading, and controlled benchmark construction."""

from qkef.datasets.fiqa import (
    FiqaDataset,
    FiqaDocument,
    FiqaQrel,
    FiqaQuery,
    load_fiqa_dataset,
)

__all__ = [
    "FiqaDataset",
    "FiqaDocument",
    "FiqaQrel",
    "FiqaQuery",
    "load_fiqa_dataset",
]
