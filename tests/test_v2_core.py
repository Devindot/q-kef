from __future__ import annotations

from dataclasses import replace

import pytest

from qkef.schemas.knowledge import EvolutionAction
from qkef.v2.planning import ShadowStateView, TransitionPlanner
from qkef.v2.risk import CounterfactualEvaluator, InvariantChecker, SafeTransitionSelector, witness_queries
from qkef.v2.runtime import QKEFV2Runtime
from qkef.v2.state import AuthorityMetadata, KnowledgeState, V2KnowledgeRecord
from qkef.v2.transaction import EpochTransactionManager, FailureStage
from qkef.v2.types import Decision, LifecycleState, RetrievalState


def record(identifier: str, vector=(1.0, 0.0), **kwargs) -> V2KnowledgeRecord:
    return V2KnowledgeRecord(identifier, f"policy evidence for {identifier}", tuple(vector), (identifier,), **kwargs)


@pytest.fixture
def base() -> KnowledgeState:
    return KnowledgeState.from_records([record("old"), record("other", (0.0, 1.0))], epoch_id=12)


@pytest.mark.parametrize(
    ("action", "predecessors", "children", "active", "relation"),
    [
        ("NEW", [], [], {"old", "other", "incoming"}, None),
        ("REPLACE", ["old"], [], {"other", "incoming"}, "supersedes"),
        ("MERGE", ["old", "other"], [], {"incoming"}, "merged_from"),
        ("ARCHIVE", ["old"], [], {"other"}, "archived_by"),
        ("COEXIST", ["old"], [], {"old", "other", "incoming"}, "coexists_with"),
        ("SPLIT", [], [record("child-a"), record("child-b", (0.0, 1.0))], {"old", "other", "child-a", "child-b"}, "split_into"),
    ],
)
def test_every_operator_is_explicit_and_consistent(base, action, predecessors, children, active, relation):
    incoming = record("incoming", (0.7, 0.7))
    plan = TransitionPlanner().plan(base, action, incoming, predecessors, split_children=children)
    shadow = ShadowStateView(base, plan).materialize()
    assert set(shadow.index_entries) == active
    assert shadow.active_graph_ids() == active
    assert not plan.preconditions
    if relation:
        assert any(edge.relation == relation for edge in shadow.edges)


def test_transition_plan_is_deterministic(base):
    planner = TransitionPlanner()
    first = planner.plan(base, "REPLACE", record("incoming"), ["old"])
    second = planner.plan(base, "REPLACE", record("incoming"), ["old"])
    assert first == second
    assert first.transition_id == second.transition_id
    assert first.payload_hash == second.payload_hash


def test_shadow_evaluation_never_mutates_live_state(base):
    before = base.state_hash
    incoming = record("incoming")
    plan = TransitionPlanner().plan(base, "REPLACE", incoming, ["old"])
    risk = CounterfactualEvaluator().evaluate(base, plan, incoming, 0.95, witness_queries(incoming.text, [base.records["old"].text]))
    assert base.state_hash == before
    assert risk.feasible


def test_historical_only_is_excluded_from_active_but_available_as_of(base):
    old = replace(base.records["old"], lifecycle_state=LifecycleState.SUPERSEDED, retrieval_state=RetrievalState.HISTORICAL_ONLY)
    state = KnowledgeState.from_records([old, base.records["other"]])
    assert "old" not in dict(state.search((1.0, 0.0), 5))
    assert "old" in dict(state.search((1.0, 0.0), 5, historical=True))


def test_bitemporal_retrieval_filters_valid_and_system_time():
    temporal = record("temporal", valid_from="2025-01-01T00:00:00Z", valid_to="2026-01-01T00:00:00Z", system_from="2025-02-01T00:00:00Z")
    state = KnowledgeState.from_records([temporal])
    assert state.search((1.0, 0.0), 1, as_of_valid_time="2025-06-01T00:00:00Z")
    assert not state.search((1.0, 0.0), 1, as_of_valid_time="2026-06-01T00:00:00Z")
    assert not state.search((1.0, 0.0), 1, as_of_system_time="2025-01-15T00:00:00Z")


def test_authority_conflict_is_hard_rejection():
    high = record("old", authority=AuthorityMetadata(source_authority="regulator", source_priority=10))
    low = record("incoming", authority=AuthorityMetadata(source_authority="blog", source_priority=1))
    state = KnowledgeState.from_records([high])
    plan = TransitionPlanner().plan(state, "REPLACE", low, ["old"])
    risk = CounterfactualEvaluator().evaluate(state, plan, low, 0.99, ["policy evidence"])
    assert risk.authority_violation == 1.0
    assert not risk.feasible


def test_selector_quarantines_ambiguous_and_accepts_singleton(base):
    incoming = record("incoming")
    plan = TransitionPlanner().plan(base, "REPLACE", incoming, ["old"])
    risk = CounterfactualEvaluator().evaluate(base, plan, incoming, 0.99, ["policy evidence"])
    selector = SafeTransitionSelector(maximum_risk=10)
    assert selector.select([risk], ["REPLACE", "COEXIST"]).decision is Decision.QUARANTINE
    assert selector.select([risk], ["REPLACE"]).decision is Decision.AUTO_COMMIT


def test_epoch_commit_certificate_and_exact_rollback(base):
    before = base.state_hash
    incoming = record("incoming")
    plan = TransitionPlanner().plan(base, "REPLACE", incoming, ["old"])
    risk = CounterfactualEvaluator().evaluate(base, plan, incoming, 0.99, ["policy evidence"])
    manager = EpochTransactionManager(base)
    result = manager.commit(plan, risk, context={"model_version": "test"})
    assert result.committed
    assert manager.live_state.epoch_id == 13
    assert manager.live_state.transition_id == plan.transition_id
    assert result.certificate.payload_hash == manager.certificates[plan.transition_id].payload_hash
    restored = manager.rollback(plan.transition_id)
    assert restored.state_hash == before


@pytest.mark.parametrize("stage", list(FailureStage))
def test_failure_injection_never_publishes_mixed_epoch(base, stage):
    before = base.state_hash
    incoming = record("incoming")
    plan = TransitionPlanner().plan(base, "REPLACE", incoming, ["old"])
    risk = CounterfactualEvaluator().evaluate(base, plan, incoming, 0.99, ["policy evidence"])
    manager = EpochTransactionManager(base)
    result = manager.commit(plan, risk, fail_at=stage)
    assert not result.committed
    assert manager.live_state.state_hash == before
    assert manager.live_state.epoch_id == 12
    assert manager.live_state.active_graph_ids() == set(manager.live_state.index_entries)


def test_runtime_separates_analysis_from_mutation_and_rejects_leakage(base):
    runtime = QKEFV2Runtime(base, selector=SafeTransitionSelector(maximum_risk=10))
    before = runtime.state.state_hash
    analysis = runtime.analyze_incoming_knowledge(record("incoming"), ["old"], {"REPLACE": 0.99}, ["REPLACE"])
    assert runtime.state.state_hash == before
    assert analysis.selection.decision is Decision.AUTO_COMMIT
    assert runtime.commit_transition(analysis).committed
    leaked = record("leaked", metadata={"expected_action": "NEW"})
    with pytest.raises(ValueError, match="forbidden"):
        runtime.analyze_incoming_knowledge(leaked, [], {"NEW": 1.0}, ["NEW"])


def test_invalid_plans_fail_invariants(base):
    incoming = record("incoming")
    plan = TransitionPlanner().plan(base, "MERGE", incoming, ["old"])
    shadow = ShadowStateView(base, plan).materialize()
    results = InvariantChecker().check(base, shadow, plan, incoming)
    assert not next(item for item in results if item.name == "preconditions").passed


def test_witnesses_are_anchored_to_incoming_and_contrastive_candidates():
    probes = witness_queries(
        "The revised contribution ceiling is 9000 dollars",
        ["The prior contribution ceiling was 7000 dollars"],
    )
    assert probes[0].startswith("revised")
    assert "9000" in probes[0]
    assert any("7000" in probe and "9000" in probe for probe in probes)


def test_split_children_are_active_affected_lineage(base):
    incoming = record("incoming")
    children = [record("child-a"), record("child-b", (0.0, 1.0))]
    plan = TransitionPlanner().plan(base, "SPLIT", incoming, [], split_children=children)
    assert {"incoming", "child-a", "child-b"} <= set(plan.affected_lineages)
    risk = CounterfactualEvaluator().evaluate(base, plan, incoming, 0.99, ["policy evidence child-a"])
    assert risk.current_evidence_miss == 0.0


def test_archive_without_active_successor_has_no_current_evidence_miss(base):
    incoming = record("archive-notice")
    plan = TransitionPlanner().plan(base, "ARCHIVE", incoming, ["old"])
    risk = CounterfactualEvaluator().evaluate(base, plan, incoming, 0.99, ["policy evidence old"])
    assert risk.current_evidence_miss == 0.0


def test_witness_search_uses_query_coverage_not_document_length():
    distractor = V2KnowledgeRecord("distractor", "alpha", (0.0, 1.0), ("d",))
    state = KnowledgeState.from_records([distractor])
    long_text = "alpha beta gamma " + " ".join(f"context{index}" for index in range(100))
    incoming = V2KnowledgeRecord("incoming", long_text, (1.0, 0.0), ("i",))
    plan = TransitionPlanner().plan(state, "NEW", incoming)
    risk = CounterfactualEvaluator(top_k=1).evaluate(state, plan, incoming, 0.99, ["alpha beta gamma"])
    assert risk.current_evidence_miss == 0.0
