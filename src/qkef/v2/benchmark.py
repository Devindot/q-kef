"""Leakage-isolated ancestry partitioning for the Q-KEF v2 benchmark."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Mapping

from qkef.datasets.fiqa import FiqaDataset, FiqaQrel
from qkef.v2.canonical import sha256_payload


@dataclass(frozen=True)
class AncestryPartition:
    dataset: FiqaDataset
    query_assignments: dict[str, str]
    document_assignments: dict[str, str]
    manifest: dict[str, object]


def _stable_component_key(seed: int, queries: set[str], documents: set[str]) -> str:
    payload = f"{seed}|{'|'.join(sorted(queries))}|{'|'.join(sorted(documents))}"
    return hashlib.sha256(payload.encode()).hexdigest()


def partition_by_ancestry(
    dataset: FiqaDataset,
    split_weights: Mapping[str, int],
    *,
    seed: int,
) -> AncestryPartition:
    """Assign complete positive-qrel query/document components to one split.

    Whole connected components are indivisible, so neither a source document nor
    a query can occur in more than one output partition. Assignment is a stable
    largest-first greedy balance against the requested per-split event weights.
    """

    weights = {name: int(weight) for name, weight in split_weights.items() if int(weight) > 0}
    if not weights:
        raise ValueError("at least one positive split weight is required")
    if any(weight < 0 for weight in split_weights.values()):
        raise ValueError("split weights must be non-negative")

    adjacency: dict[str, set[str]] = defaultdict(set)
    eligible_qrels = []
    original_splits: dict[str, set[str]] = defaultdict(set)
    for qrel in dataset.qrels:
        if qrel.relevance <= 0 or not dataset.documents[qrel.document_id].text.strip():
            continue
        query_node = "q:" + qrel.query_id
        document_node = "d:" + qrel.document_id
        adjacency[query_node].add(document_node)
        adjacency[document_node].add(query_node)
        eligible_qrels.append(qrel)
        original_splits[qrel.query_id].add(qrel.source_split)

    components: list[tuple[set[str], set[str]]] = []
    visited: set[str] = set()
    for root in sorted(adjacency):
        if root in visited:
            continue
        stack = [root]
        visited.add(root)
        queries: set[str] = set()
        documents: set[str] = set()
        while stack:
            node = stack.pop()
            (queries if node.startswith("q:") else documents).add(node[2:])
            for neighbor in adjacency[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        components.append((queries, documents))
    components.sort(key=lambda item: (-len(item[1]), _stable_component_key(seed, item[0], item[1])))

    assigned_documents = Counter({split: 0 for split in weights})
    assigned_queries = Counter({split: 0 for split in weights})
    assigned_components = Counter({split: 0 for split in weights})
    split_order = {split: index for index, split in enumerate(weights)}
    query_assignments: dict[str, str] = {}
    document_assignments: dict[str, str] = {}
    for queries, documents in components:
        destination = min(
            weights,
            key=lambda split: (
                assigned_documents[split] / weights[split],
                assigned_queries[split] / weights[split],
                split_order[split],
            ),
        )
        for query_id in queries:
            query_assignments[query_id] = destination
        for document_id in documents:
            document_assignments[document_id] = destination
        assigned_documents[destination] += len(documents)
        assigned_queries[destination] += len(queries)
        assigned_components[destination] += 1

    partitioned_qrels = tuple(
        FiqaQrel(qrel.query_id, qrel.document_id, qrel.relevance, query_assignments[qrel.query_id])
        for qrel in eligible_qrels
    )
    partitioned = FiqaDataset(dataset.root, dataset.documents, dataset.queries, partitioned_qrels)
    overlaps = {}
    for left, right in combinations(weights, 2):
        left_queries = {key for key, value in query_assignments.items() if value == left}
        right_queries = {key for key, value in query_assignments.items() if value == right}
        left_documents = {key for key, value in document_assignments.items() if value == left}
        right_documents = {key for key, value in document_assignments.items() if value == right}
        overlaps[f"{left}_{right}"] = {
            "query_count": len(left_queries & right_queries),
            "document_count": len(left_documents & right_documents),
        }
    manifest = {
        "partition_version": "ancestry-components-v1",
        "seed": seed,
        "algorithm": "stable largest-first component allocation by document/query load per requested event weight",
        "requested_split_weights": weights,
        "component_count": len(components),
        "eligible_positive_qrel_count": len(eligible_qrels),
        "eligible_query_count": len(query_assignments),
        "eligible_document_count": len(document_assignments),
        "counts_by_split": {
            split: {
                "components": assigned_components[split],
                "queries": assigned_queries[split],
                "documents": assigned_documents[split],
            }
            for split in weights
        },
        "original_qrels_split_counts": {
            split: dict(sorted(Counter(source for query, destination in query_assignments.items() if destination == split for source in original_splits[query]).items()))
            for split in weights
        },
        "cross_split_overlaps": overlaps,
        "assignment_sha256": sha256_payload({"queries": query_assignments, "documents": document_assignments}),
    }
    if any(value["query_count"] or value["document_count"] for value in overlaps.values()):
        raise RuntimeError("ancestry component partitioning produced cross-split overlap")
    return AncestryPartition(partitioned, query_assignments, document_assignments, manifest)
