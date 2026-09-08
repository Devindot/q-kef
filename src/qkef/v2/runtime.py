"""Analysis/mutation separation for Q-KEF v2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qkef.schemas.knowledge import EvolutionAction
from qkef.v2.planning import TransitionPlan, TransitionPlanner
from qkef.v2.risk import FORBIDDEN_RUNTIME_FIELDS, CounterfactualEvaluator, SafeTransitionSelector, SelectionResult, TransitionRisk, witness_queries
from qkef.v2.state import KnowledgeState, V2KnowledgeRecord
from qkef.v2.transaction import CommitResult, EpochTransactionManager, FailureStage


@dataclass(frozen=True)
class V2Analysis:
    predecessor_candidate_ids: tuple[str, ...]
    lifecycle_probabilities: dict[str, float]
    conformal_action_set: tuple[str, ...]
    plans: tuple[TransitionPlan, ...]
    risks: tuple[TransitionRisk, ...]
    selection: SelectionResult
    witness_queries: tuple[str, ...]
    transition_certificate_preview: dict[str, Any]


class QKEFV2Runtime:
    def __init__(self, state: KnowledgeState, *, planner: TransitionPlanner | None = None, evaluator: CounterfactualEvaluator | None = None, selector: SafeTransitionSelector | None = None):
        self.transactions = EpochTransactionManager(state)
        self.planner = planner or TransitionPlanner()
        self.evaluator = evaluator or CounterfactualEvaluator()
        self.selector = selector or SafeTransitionSelector()

    @property
    def state(self) -> KnowledgeState:
        return self.transactions.live_state

    def analyze_incoming_knowledge(
        self,
        incoming: V2KnowledgeRecord,
        predecessor_candidate_ids: list[str],
        lifecycle_probabilities: dict[str, float],
        conformal_action_set: list[str],
        *,
        split_children: list[V2KnowledgeRecord] | None = None,
    ) -> V2Analysis:
        forbidden = FORBIDDEN_RUNTIME_FIELDS & set(incoming.metadata)
        if forbidden:
            raise ValueError(f"oracle benchmark fields are forbidden at runtime: {sorted(forbidden)}")
        before_hash = self.state.state_hash
        candidate_texts = [self.state.records[identifier].text for identifier in predecessor_candidate_ids if identifier in self.state.records]
        witnesses = witness_queries(incoming.text, candidate_texts)
        plans, risks = [], []
        for action_name in sorted(set(conformal_action_set)):
            action = EvolutionAction(action_name)
            if action is EvolutionAction.NEW:
                predecessors = []
            elif action is EvolutionAction.MERGE:
                predecessors = predecessor_candidate_ids[:2]
            else:
                predecessors = predecessor_candidate_ids[:1]
            plan = self.planner.plan(self.state, action, incoming, predecessors, split_children=split_children or [])
            plans.append(plan)
            risks.append(self.evaluator.evaluate(self.state, plan, incoming, lifecycle_probabilities.get(action_name, 0.0), witnesses))
        if self.state.state_hash != before_hash:
            raise RuntimeError("analysis mutated live state")
        selection = self.selector.select(risks, conformal_action_set)
        preview = {
            "decision": selection.decision.value,
            "input_content_hash": __import__("hashlib").sha256(incoming.text.encode()).hexdigest(),
            "pre_epoch": self.state.epoch_id,
            "proposed_action_set": sorted(set(conformal_action_set)),
            "reason": selection.reason,
            "risk_vectors": {risk.action: risk.vector() for risk in risks},
        }
        return V2Analysis(tuple(predecessor_candidate_ids), dict(lifecycle_probabilities), tuple(sorted(set(conformal_action_set))), tuple(plans), tuple(risks), selection, witnesses, preview)

    def commit_transition(self, analysis: V2Analysis, *, fail_at: FailureStage | None = None) -> CommitResult:
        if analysis.selection.chosen_action is None:
            raise ValueError("quarantined analysis cannot be committed")
        plan = next(plan for plan in analysis.plans if plan.proposed_action.value == analysis.selection.chosen_action)
        risk = next(risk for risk in analysis.risks if risk.action == analysis.selection.chosen_action)
        return self.transactions.commit(plan, risk, context={"probability_distribution": analysis.lifecycle_probabilities, "conformal_prediction_set": analysis.conformal_action_set}, fail_at=fail_at)
