"""Epoch-coherent in-memory transaction, certificate, failure injection, rollback."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from qkef.v2.canonical import canonical_value, sha256_payload
from qkef.v2.planning import ShadowStateView, TransitionPlan
from qkef.v2.risk import TransitionRisk
from qkef.v2.state import GraphEdge, KnowledgeState, V2KnowledgeRecord


class FailureStage(str, Enum):
    BEFORE_GRAPH_PREPARE = "BEFORE_GRAPH_PREPARE"
    AFTER_GRAPH_PREPARE = "AFTER_GRAPH_PREPARE"
    BEFORE_INDEX_PREPARE = "BEFORE_INDEX_PREPARE"
    AFTER_INDEX_PREPARE = "AFTER_INDEX_PREPARE"
    BEFORE_CERTIFICATE = "BEFORE_CERTIFICATE"
    BEFORE_EPOCH_COMMIT = "BEFORE_EPOCH_COMMIT"
    AFTER_EPOCH_COMMIT_METADATA = "AFTER_EPOCH_COMMIT_METADATA"


class InjectedTransitionFailure(RuntimeError):
    pass


@dataclass(frozen=True)
class RollbackDelta:
    transition_id: str
    pre_epoch: int
    pre_transition_id: str
    prior_records: tuple[V2KnowledgeRecord, ...]
    added_record_ids: tuple[str, ...]
    added_edges: tuple[GraphEdge, ...]
    removed_edges: tuple[GraphEdge, ...]
    prior_index_entries: tuple[tuple[str, tuple[float, ...] | None], ...]
    pre_state_hash: str


@dataclass(frozen=True)
class TransitionCertificate:
    payload: dict[str, Any]
    payload_hash: str
    status: str

    def as_dict(self) -> dict[str, Any]:
        return {"payload": canonical_value(self.payload), "payload_hash": self.payload_hash, "status": self.status}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n"
        descriptor, temporary = tempfile.mkstemp(prefix=path.name, dir=path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


@dataclass(frozen=True)
class CommitResult:
    committed: bool
    certificate: TransitionCertificate
    state_hash: str
    error: str | None = None


class EpochTransactionManager:
    def __init__(self, state: KnowledgeState):
        self.live_state = state
        self.rollback_records: dict[str, RollbackDelta] = {}
        self.certificates: dict[str, TransitionCertificate] = {}

    @staticmethod
    def _fail(stage: FailureStage | None, expected: FailureStage) -> None:
        if stage is expected:
            raise InjectedTransitionFailure(expected.value)

    def _delta(self, plan: TransitionPlan) -> RollbackDelta:
        changed_ids = {record.identifier for record in plan.record_updates}
        changed_index_ids = set(plan.index_removals) | {identifier for identifier, _ in plan.index_additions}
        return RollbackDelta(
            transition_id=plan.transition_id,
            pre_epoch=self.live_state.epoch_id,
            pre_transition_id=self.live_state.transition_id,
            prior_records=tuple(self.live_state.records[identifier] for identifier in sorted(changed_ids)),
            added_record_ids=tuple(sorted(record.identifier for record in plan.record_additions)),
            added_edges=plan.edge_additions,
            removed_edges=plan.edge_removals,
            prior_index_entries=tuple((identifier, self.live_state.index_entries.get(identifier)) for identifier in sorted(changed_index_ids)),
            pre_state_hash=self.live_state.state_hash,
        )

    def _certificate(self, plan: TransitionPlan, risk: TransitionRisk, before: KnowledgeState, after: KnowledgeState, context: dict[str, Any], *, status: str) -> TransitionCertificate:
        payload = {
            "affected_lineages": plan.affected_lineages,
            "chosen_action": plan.proposed_action.value if status == "COMMITTED" else None,
            "configuration_hash": plan.configuration_hash,
            "context": {key: value for key, value in context.items() if key != "timestamp"},
            "input_unit_id": plan.input_unit_id,
            "invariant_results": [asdict(value) for value in risk.invariant_results],
            "post_epoch": after.epoch_id,
            "post_graph_hash": after.graph_state_hash,
            "post_index_hash": after.index_state_hash,
            "pre_epoch": before.epoch_id,
            "pre_graph_hash": before.graph_state_hash,
            "pre_index_hash": before.index_state_hash,
            "predecessor_candidate_ids": plan.predecessor_ids,
            "proposed_action": plan.proposed_action.value,
            "risk_vector": risk.vector(),
            "soft_risk": risk.soft_score,
            "transition_id": plan.transition_id,
        }
        return TransitionCertificate(payload, sha256_payload(payload), status)

    def commit(self, plan: TransitionPlan, risk: TransitionRisk, *, context: dict[str, Any] | None = None, fail_at: FailureStage | None = None) -> CommitResult:
        before = self.live_state.clone()
        context = context or {}
        try:
            if not risk.feasible:
                raise ValueError("transition did not pass hard invariants")
            self._fail(fail_at, FailureStage.BEFORE_GRAPH_PREPARE)
            shadow = ShadowStateView(before, plan)
            prepared_records, prepared_edges = shadow.records, shadow.edges
            self._fail(fail_at, FailureStage.AFTER_GRAPH_PREPARE)
            self._fail(fail_at, FailureStage.BEFORE_INDEX_PREPARE)
            prepared_index = shadow.index_entries
            self._fail(fail_at, FailureStage.AFTER_INDEX_PREPARE)
            prepared = KnowledgeState(prepared_records, prepared_edges, prepared_index, before.epoch_id + 1, plan.transition_id)
            if prepared.active_graph_ids() != set(prepared.index_entries):
                raise ValueError("prepared graph/index mismatch")
            self._fail(fail_at, FailureStage.BEFORE_CERTIFICATE)
            certificate = self._certificate(plan, risk, before, prepared, context, status="COMMITTED")
            self._fail(fail_at, FailureStage.BEFORE_EPOCH_COMMIT)
            delta = self._delta(plan)
            self.live_state = prepared
            self._fail(fail_at, FailureStage.AFTER_EPOCH_COMMIT_METADATA)
            self.rollback_records[plan.transition_id] = delta
            self.certificates[plan.transition_id] = certificate
            return CommitResult(True, certificate, prepared.state_hash)
        except Exception as error:
            self.live_state = before
            failed_payload = {
                "error": str(error), "post_epoch": before.epoch_id, "pre_epoch": before.epoch_id,
                "risk_vector": risk.vector(), "transition_id": plan.transition_id,
            }
            certificate = TransitionCertificate(failed_payload, sha256_payload(failed_payload), "FAILED")
            self.certificates[plan.transition_id] = certificate
            return CommitResult(False, certificate, before.state_hash, str(error))

    def rollback(self, transition_id: str) -> KnowledgeState:
        if transition_id not in self.rollback_records:
            raise KeyError(f"no rollback delta for {transition_id}")
        delta = self.rollback_records[transition_id]
        if self.live_state.transition_id != transition_id:
            raise ValueError("rollback is only supported for the current transition")
        state = self.live_state.clone()
        for identifier in delta.added_record_ids:
            state.records.pop(identifier, None)
            state.index_entries.pop(identifier, None)
        for record in delta.prior_records:
            state.records[record.identifier] = record
        edges = (set(state.edges) - set(delta.added_edges)) | set(delta.removed_edges)
        state.edges = tuple(sorted(edges))
        for identifier, prior in delta.prior_index_entries:
            if prior is None:
                state.index_entries.pop(identifier, None)
            else:
                state.index_entries[identifier] = prior
        state.epoch_id = delta.pre_epoch
        state.transition_id = delta.pre_transition_id
        if state.state_hash != delta.pre_state_hash:
            raise RuntimeError("rollback did not restore exact state hash")
        self.live_state = state
        return state
