"""Cardinality-aware transition planning and non-live overlay views."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterable

from qkef.schemas.knowledge import EvolutionAction
from qkef.v2.canonical import sha256_payload
from qkef.v2.state import GraphEdge, KnowledgeState, V2KnowledgeRecord
from qkef.v2.types import LifecycleState, RetrievalState


@dataclass(frozen=True)
class TransitionPlan:
    transition_id: str
    proposed_action: EvolutionAction
    input_unit_id: str
    predecessor_ids: tuple[str, ...]
    record_additions: tuple[V2KnowledgeRecord, ...] = ()
    record_updates: tuple[V2KnowledgeRecord, ...] = ()
    edge_additions: tuple[GraphEdge, ...] = ()
    edge_removals: tuple[GraphEdge, ...] = ()
    index_additions: tuple[tuple[str, tuple[float, ...]], ...] = ()
    index_removals: tuple[str, ...] = ()
    preconditions: tuple[str, ...] = ()
    postconditions: tuple[str, ...] = ()
    affected_lineages: tuple[str, ...] = ()
    estimated_churn: int = 0
    configuration_hash: str = ""

    @property
    def payload_hash(self) -> str:
        payload = {name: value for name, value in self.__dict__.items() if name != "transition_id"}
        return sha256_payload(payload)


class ShadowStateView:
    """Read-only delta overlay. Construction never mutates the live base state."""

    def __init__(self, base: KnowledgeState, plan: TransitionPlan):
        self.base = base
        self.plan = plan
        self._updates = {record.identifier: record for record in plan.record_updates}
        self._additions = {record.identifier: record for record in plan.record_additions}
        self._index_additions = dict(plan.index_additions)

    @property
    def records(self) -> dict[str, V2KnowledgeRecord]:
        result = dict(self.base.records)
        result.update(self._updates)
        result.update(self._additions)
        return result

    @property
    def edges(self) -> tuple[GraphEdge, ...]:
        removed = set(self.plan.edge_removals)
        return tuple(sorted((set(self.base.edges) - removed) | set(self.plan.edge_additions)))

    @property
    def index_entries(self) -> dict[str, tuple[float, ...]]:
        result = {key: value for key, value in self.base.index_entries.items() if key not in self.plan.index_removals}
        result.update(self._index_additions)
        return result

    def materialize(self, *, epoch_id: int | None = None) -> KnowledgeState:
        return KnowledgeState(
            records=self.records,
            edges=self.edges,
            index_entries=self.index_entries,
            epoch_id=self.base.epoch_id if epoch_id is None else epoch_id,
            transition_id=self.plan.transition_id,
        )


class TransitionPlanner:
    """Map every lifecycle hypothesis to an explicit deterministic state delta."""

    def __init__(self, configuration: dict[str, Any] | None = None):
        self.configuration = configuration or {"version": "v2.0", "archive_notice": True}
        self.configuration_hash = sha256_payload(self.configuration)

    def plan(
        self,
        state: KnowledgeState,
        action: EvolutionAction | str,
        incoming: V2KnowledgeRecord,
        predecessor_ids: Iterable[str] = (),
        *,
        split_children: Iterable[V2KnowledgeRecord] = (),
    ) -> TransitionPlan:
        action = EvolutionAction(action)
        predecessors = tuple(sorted(dict.fromkeys(predecessor_ids)))
        missing = tuple(identifier for identifier in predecessors if identifier not in state.records)
        additions: list[V2KnowledgeRecord] = []
        updates: list[V2KnowledgeRecord] = []
        edges: list[GraphEdge] = []
        index_additions: list[tuple[str, tuple[float, ...]]] = []
        index_removals: list[str] = []
        preconditions: list[str] = []

        def add_active(record: V2KnowledgeRecord, lifecycle: LifecycleState) -> V2KnowledgeRecord:
            value = replace(record, lifecycle_state=lifecycle, retrieval_state=RetrievalState.ACTIVE)
            additions.append(value)
            index_additions.append((value.identifier, value.normalized_vector()))
            return value

        def retire(identifier: str, lifecycle: LifecycleState) -> None:
            current = state.records[identifier]
            updates.append(replace(current, lifecycle_state=lifecycle, retrieval_state=RetrievalState.HISTORICAL_ONLY))
            index_removals.append(identifier)

        if missing:
            preconditions.append("missing_predecessor:" + ",".join(missing))
        elif action is EvolutionAction.NEW:
            add_active(incoming, LifecycleState.CURRENT)
        elif action is EvolutionAction.REPLACE:
            if len(predecessors) != 1:
                preconditions.append("REPLACE_requires_exactly_one_predecessor")
            else:
                retire(predecessors[0], LifecycleState.SUPERSEDED)
                add_active(incoming, LifecycleState.CURRENT)
                edges.append(GraphEdge(incoming.identifier, predecessors[0], "supersedes"))
        elif action is EvolutionAction.ARCHIVE:
            if not predecessors:
                preconditions.append("ARCHIVE_requires_predecessor")
            else:
                for identifier in predecessors:
                    retire(identifier, LifecycleState.ARCHIVED)
                notice = replace(incoming, lifecycle_state=LifecycleState.ARCHIVED, retrieval_state=RetrievalState.HISTORICAL_ONLY)
                additions.append(notice)
                edges.extend(GraphEdge(identifier, incoming.identifier, "archived_by") for identifier in predecessors)
        elif action is EvolutionAction.MERGE:
            if len(predecessors) < 2:
                preconditions.append("MERGE_requires_multiple_predecessors")
            else:
                for identifier in predecessors:
                    retire(identifier, LifecycleState.SUPERSEDED)
                    edges.append(GraphEdge(incoming.identifier, identifier, "merged_from"))
                add_active(incoming, LifecycleState.CONSOLIDATED)
        elif action is EvolutionAction.COEXIST:
            if not predecessors:
                preconditions.append("COEXIST_requires_predecessor")
            else:
                add_active(incoming, LifecycleState.COEXISTING)
                for identifier in predecessors:
                    edges.append(GraphEdge(incoming.identifier, identifier, "coexists_with"))
                    edges.append(GraphEdge(identifier, incoming.identifier, "coexists_with"))
        elif action is EvolutionAction.SPLIT:
            children = tuple(sorted(split_children, key=lambda item: item.identifier))
            if len(children) < 2:
                preconditions.append("SPLIT_requires_two_meaningful_children")
            else:
                parent = replace(incoming, lifecycle_state=LifecycleState.DERIVED, retrieval_state=RetrievalState.HISTORICAL_ONLY)
                additions.append(parent)
                for child in children:
                    add_active(child, LifecycleState.DERIVED)
                    edges.append(GraphEdge(parent.identifier, child.identifier, "split_into"))

        plan_body = {
            "action": action.value,
            "additions": additions,
            "configuration_hash": self.configuration_hash,
            "edges": edges,
            "incoming": incoming.identifier,
            "predecessors": predecessors,
            "updates": updates,
        }
        transition_id = "tr_" + sha256_payload(plan_body)[:24]
        return TransitionPlan(
            transition_id=transition_id,
            proposed_action=action,
            input_unit_id=incoming.identifier,
            predecessor_ids=predecessors,
            record_additions=tuple(additions),
            record_updates=tuple(updates),
            edge_additions=tuple(sorted(edges)),
            index_additions=tuple(sorted(index_additions)),
            index_removals=tuple(sorted(index_removals)),
            preconditions=tuple(preconditions),
            postconditions=("graph_index_same_epoch", "historical_lineage_retained"),
            affected_lineages=tuple(sorted(set(predecessors) | {incoming.identifier} | {record.identifier for record in additions})),
            estimated_churn=len(index_additions) + len(index_removals),
            configuration_hash=self.configuration_hash,
        )
