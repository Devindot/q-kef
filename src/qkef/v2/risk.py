"""Leakage-safe witnesses, hard invariants, risk calculation, and selection."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

from qkef.schemas.knowledge import EvolutionAction
from qkef.v2.planning import ShadowStateView, TransitionPlan
from qkef.v2.state import KnowledgeState, V2KnowledgeRecord
from qkef.v2.types import ACTIVE_LIFECYCLE_STATES, Decision, RetrievalState


FORBIDDEN_RUNTIME_FIELDS = {"expected_action", "true_action", "expected_target_ids", "mutation_method", "benchmark_split", "relation_type"}

_WITNESS_STOPWORDS = {
    "and", "are", "but", "for", "from", "has", "have", "into", "not", "that", "the", "their", "this", "was", "were", "with",
}


def _salient_tokens(text: str, document_frequency: Counter[str], limit: int = 6) -> list[str]:
    tokens = [token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 2 and token not in _WITNESS_STOPWORDS]
    counts = Counter(tokens)
    first_position = {token: tokens.index(token) for token in counts}
    selected = sorted(
        counts,
        key=lambda token: (document_frequency[token], -counts[token], -len(token), first_position[token], token),
    )[:limit]
    selected_set = set(selected)
    return list(dict.fromkeys(token for token in tokens if token in selected_set))


def witness_queries(incoming_text: str, candidate_texts: Iterable[str] = (), *, historical_queries: Iterable[str] = (), mode: str = "STRICT_CONTENT_DERIVED") -> tuple[str, ...]:
    if mode not in {"STRICT_CONTENT_DERIVED", "HISTORICAL_QUERY_LOG"}:
        raise ValueError("unsupported witness mode")
    candidates = tuple(candidate_texts)
    documents = (incoming_text, *candidates)
    token_sets = [set(re.findall(r"[a-z0-9]+", text.lower())) for text in documents]
    document_frequency = Counter(token for tokens in token_sets for token in tokens)
    incoming_tokens = _salient_tokens(incoming_text, document_frequency)
    probes = [" ".join(incoming_tokens)] if incoming_tokens else []
    for candidate_text in candidates:
        candidate_tokens = _salient_tokens(candidate_text, document_frequency, limit=2)
        contrastive = list(dict.fromkeys([*incoming_tokens, *candidate_tokens]))
        if contrastive:
            probes.append(" ".join(contrastive))
    if mode == "HISTORICAL_QUERY_LOG":
        probes.extend(query.strip() for query in historical_queries if query.strip())
    return tuple(dict.fromkeys(probe for probe in probes if probe)) or (incoming_text[:160],)


@dataclass(frozen=True)
class InvariantResult:
    name: str
    passed: bool
    detail: str


class InvariantChecker:
    def check(self, base: KnowledgeState, shadow: KnowledgeState, plan: TransitionPlan, incoming: V2KnowledgeRecord) -> tuple[InvariantResult, ...]:
        active_graph = shadow.active_graph_ids()
        indexed = set(shadow.index_entries)
        results = [
            InvariantResult("preconditions", not plan.preconditions, ";".join(plan.preconditions) or "satisfied"),
            InvariantResult("active_graph_iff_searchable_index", active_graph == indexed, f"graph_only={sorted(active_graph-indexed)};index_only={sorted(indexed-active_graph)}"),
            InvariantResult("historical_records_retained", set(base.records) <= set(shadow.records), "no base record deleted"),
            InvariantResult("coherent_epoch", shadow.epoch_id == base.epoch_id, "shadow overlays the committed base epoch"),
        ]
        edge_set = set(shadow.edges)
        if plan.proposed_action is EvolutionAction.REPLACE and len(plan.predecessor_ids) == 1:
            expected = (incoming.identifier, plan.predecessor_ids[0], "supersedes")
            results.append(InvariantResult("replace_lineage", any((edge.source, edge.target, edge.relation) == expected for edge in edge_set), str(expected)))
        if plan.proposed_action is EvolutionAction.MERGE:
            missing = [identifier for identifier in plan.predecessor_ids if not any(edge.source == incoming.identifier and edge.target == identifier and edge.relation == "merged_from" for edge in edge_set)]
            results.append(InvariantResult("merge_all_parents", not missing, f"missing={missing}"))
        if plan.proposed_action is EvolutionAction.SPLIT:
            children = [record.identifier for record in plan.record_additions if record.identifier != incoming.identifier]
            missing = [identifier for identifier in children if not any(edge.source == incoming.identifier and edge.target == identifier and edge.relation == "split_into" for edge in edge_set)]
            results.append(InvariantResult("split_child_lineage", len(children) >= 2 and not missing, f"children={children};missing={missing}"))
        temporal_ok = all(record.visible_at(valid_time=record.valid_from, system_time=record.system_from) for record in shadow.records.values())
        results.append(InvariantResult("temporal_intervals", temporal_ok, "interval bounds validated"))
        return tuple(results)


@dataclass(frozen=True)
class TransitionRisk:
    action: str
    obsolete_exposure: float
    current_evidence_miss: float
    index_churn: float
    cross_view_violation: float
    lineage_violation: float
    temporal_violation: float
    authority_violation: float
    probability_penalty: float
    soft_score: float
    feasible: bool
    invariant_results: tuple[InvariantResult, ...]

    def vector(self) -> dict[str, float]:
        return {"OE": self.obsolete_exposure, "CM": self.current_evidence_miss, "IC": self.index_churn, "CV": self.cross_view_violation, "LV": self.lineage_violation, "TV": self.temporal_violation, "AV": self.authority_violation}


@dataclass
class CounterfactualEvaluator:
    weights: dict[str, float] = field(default_factory=lambda: {"OE": 3.0, "CM": 3.0, "IC": 0.5, "PROB": 0.25})
    top_k: int = 5

    def _lexical_search(self, state: KnowledgeState, query: str) -> list[str]:
        terms = set(re.findall(r"[a-z0-9]+", query.lower()))
        document_terms = {
            identifier: set(re.findall(r"[a-z0-9]+", state.records[identifier].text.lower()))
            for identifier in sorted(state.index_entries)
        }
        document_frequency = Counter(term for words in document_terms.values() for term in words)
        document_count = max(1, len(document_terms))
        term_weights = {
            term: math.log((document_count + 1) / (document_frequency[term] + 1)) + 1.0
            for term in terms
        }
        total_weight = sum(term_weights.values()) or 1.0
        scored = []
        for identifier, words in document_terms.items():
            score = sum(term_weights[term] for term in terms & words) / total_weight
            scored.append((identifier, score))
        return [identifier for identifier, _ in sorted(scored, key=lambda item: (-item[1], item[0]))[:self.top_k]]

    def evaluate(self, base: KnowledgeState, plan: TransitionPlan, incoming: V2KnowledgeRecord, probability: float, witnesses: Iterable[str]) -> TransitionRisk:
        live_hash = base.state_hash
        shadow = ShadowStateView(base, plan).materialize()
        invariants = InvariantChecker().check(base, shadow, plan, incoming)
        if base.state_hash != live_hash:
            raise RuntimeError("counterfactual evaluation mutated live state")
        rankings = [self._lexical_search(shadow, query) for query in witnesses]
        returned = [identifier for ranking in rankings for identifier in ranking]
        obsolete = sum(shadow.records[identifier].lifecycle_state not in ACTIVE_LIFECYCLE_STATES for identifier in returned) / max(1, len(returned))
        active_affected = set(plan.affected_lineages) & set(shadow.index_entries)
        misses = 0.0 if not active_affected else sum(not any(identifier in active_affected for identifier in ranking) for ranking in rankings) / max(1, len(rankings))
        churn = len(set(base.index_entries) ^ set(shadow.index_entries)) / max(1, len(base.index_entries))
        failed = {result.name for result in invariants if not result.passed}
        cross = float("active_graph_iff_searchable_index" in failed or "coherent_epoch" in failed)
        lineage = float(any(name in failed for name in {"replace_lineage", "merge_all_parents", "split_child_lineage", "historical_records_retained", "preconditions"}))
        temporal = float("temporal_intervals" in failed)
        authority = 0.0
        if plan.proposed_action is EvolutionAction.REPLACE and len(plan.predecessor_ids) == 1:
            old_priority = base.records[plan.predecessor_ids[0]].authority.source_priority
            new_priority = incoming.authority.source_priority
            authority = float(old_priority is not None and new_priority is not None and new_priority < old_priority)
        probability_penalty = -math.log(max(probability, 1e-12))
        score = self.weights["OE"] * obsolete + self.weights["CM"] * misses + self.weights["IC"] * churn + self.weights["PROB"] * probability_penalty
        feasible = not failed and not authority
        return TransitionRisk(plan.proposed_action.value, obsolete, misses, churn, cross, lineage, temporal, authority, probability_penalty, score, feasible, invariants)


@dataclass(frozen=True)
class SelectionResult:
    decision: Decision
    chosen_action: str | None
    reason: str


@dataclass
class SafeTransitionSelector:
    maximum_risk: float = 1.5
    minimum_risk_gap: float = 0.05
    require_singleton_conformal: bool = True

    def select(self, risks: Iterable[TransitionRisk], conformal_set: Iterable[str]) -> SelectionResult:
        plausible = set(conformal_set)
        feasible = sorted((risk for risk in risks if risk.feasible and risk.action in plausible and risk.soft_score <= self.maximum_risk), key=lambda risk: (risk.soft_score, risk.action))
        if self.require_singleton_conformal and len(plausible) != 1:
            return SelectionResult(Decision.QUARANTINE, None, "conformal prediction set is not a singleton")
        if not feasible:
            return SelectionResult(Decision.QUARANTINE, None, "no plausible transition passes hard invariants and risk threshold")
        if len(feasible) > 1 and feasible[1].soft_score - feasible[0].soft_score < self.minimum_risk_gap:
            return SelectionResult(Decision.QUARANTINE, None, "candidate transition risks are not uniquely separated")
        return SelectionResult(Decision.AUTO_COMMIT, feasible[0].action, "unique calibrated feasible transition")
