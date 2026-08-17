"""Serializable evolution lineage graph."""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
from networkx.readwrite import json_graph


class SemanticGraph:
    def __init__(self):
        self.graph = nx.MultiDiGraph()

    def add_node(self, identifier: str, **attributes: object) -> None:
        self.graph.add_node(identifier, **attributes)

    def add_edge(self, source: str, target: str, relation: str) -> None:
        self.graph.add_edge(source, target, relation=relation)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(json_graph.node_link_data(self.graph, edges="edges"), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "SemanticGraph":
        value = cls()
        value.graph = json_graph.node_link_graph(json.loads(path.read_text(encoding="utf-8")), edges="edges", directed=True, multigraph=True)
        return value
