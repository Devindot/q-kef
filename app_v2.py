"""Offline Q-KEF v2 retrieval-safe lifecycle-transition demonstration."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from qkef.v2.risk import SafeTransitionSelector
from qkef.v2.runtime import QKEFV2Runtime
from qkef.v2.state import AuthorityMetadata, KnowledgeState, V2KnowledgeRecord
from qkef.v2.transaction import FailureStage


ROOT = Path(__file__).resolve().parent
NOTICE = "Q-KEF v2 investigates a retrieval-safe lifecycle-transition architecture. Patent-oriented technical differentiation is under evaluation and subject to professional review."


def new_runtime() -> QKEFV2Runtime:
    old = V2KnowledgeRecord(
        "policy-v1", "The annual management fee is 1.5 percent.", (1.0, 0.0), ("policy",),
        valid_from="2025-01-01T00:00:00Z", system_from="2025-01-02T00:00:00Z",
        authority=AuthorityMetadata(source_authority="approved policy", source_priority=5),
    )
    context = V2KnowledgeRecord("account-context", "Account fees and disclosures.", (0.8, 0.2), ("context",))
    state = KnowledgeState.from_records([old, context], epoch_id=12)
    return QKEFV2Runtime(state, selector=SafeTransitionSelector(maximum_risk=5.0))


def results() -> dict:
    confirmatory = ROOT / "reports/v2/confirmatory/confirmatory_results.json"
    pilot = ROOT / "reports/v2/v2_results.json"
    if confirmatory.is_file():
        return json.loads(confirmatory.read_text(encoding="utf-8"))
    return json.loads(pilot.read_text(encoding="utf-8")) if pilot.is_file() else {}


def incoming(text: str, priority: int) -> V2KnowledgeRecord:
    return V2KnowledgeRecord(
        "policy-v2", text, (1.0, 0.0), ("policy-update",), valid_from="2026-01-01T00:00:00Z",
        system_from="2026-01-02T00:00:00Z", authority=AuthorityMetadata(source_authority="approved update", source_priority=priority),
    )


def analysis_frames(analysis):
    plan_rows = [{"action": plan.proposed_action.value, "transition_id": plan.transition_id, "predecessors": ", ".join(plan.predecessor_ids), "record_additions": len(plan.record_additions), "record_updates": len(plan.record_updates), "index_churn_operations": plan.estimated_churn, "preconditions": "; ".join(plan.preconditions)} for plan in analysis.plans]
    risk_rows = [{"action": risk.action, **risk.vector(), "soft_score": risk.soft_score, "feasible": risk.feasible} for risk in analysis.risks]
    invariant_rows = [{"action": risk.action, "invariant": item.name, "passed": item.passed, "detail": item.detail} for risk in analysis.risks for item in risk.invariant_results]
    return pd.DataFrame(plan_rows), pd.DataFrame(risk_rows), pd.DataFrame(invariant_rows)


def main() -> None:
    st.set_page_config(page_title="Q-KEF v2", page_icon="🛡️", layout="wide")
    if "v2_runtime" not in st.session_state:
        st.session_state.v2_runtime = new_runtime()
    runtime = st.session_state.v2_runtime
    data = results()
    st.title("Q-KEF v2")
    st.caption("Quantum-Inspired Knowledge Evolution Framework — retrieval-safe transition research prototype")
    st.info(NOTICE)
    tabs = st.tabs(["Overview", "Incoming Knowledge", "Candidate Predecessors", "Lifecycle Probabilities", "Shadow Comparison", "Retrieval Risk", "Invariants", "Epoch Commit / Rollback", "Certificate", "Temporal / Authority", "Quantum Ablation", "Technical Effects"])

    with tabs[0]:
        st.header("Predict → plan → simulate → retrieve → validate → publish")
        st.code("S_t=(G_t,I_t,E_t) → {T_a(S_t)} → witness risk + invariants → quarantine or E_t+1", language="text")
        st.write("The live state is never changed during analysis. Normal retrieval uses ACTIVE records; historical retrieval can include HISTORICAL_ONLY records at a requested valid/system time.")
        st.metric("Current committed epoch", runtime.state.epoch_id)
        st.code(runtime.state.state_hash)
        if data.get("status") == "CONFIRMATORY_TEST_EXECUTED":
            st.subheader("Confirmatory evaluation")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("TEST events", data["test_event_count"])
            c2.metric("B0 macro-F1", f"{data['model_metrics']['B0']['macro_f1']:.4f}")
            c3.metric("Q-full macro-F1", f"{data['model_metrics']['Q_FULL']['macro_f1']:.4f}")
            c4.metric("Safe precision", f"{data['technical_effects']['auto_commit_precision']:.4f}")
            st.warning(f"Conformal coverage was {data['conformal']['marginal_coverage']:.4f}, below the nominal 0.90 target; this negative result is retained.")

    with tabs[1]:
        text = st.text_area("Incoming update", "Effective 2026, the annual management fee is 1.2 percent.", height=120)
        priority = st.number_input("Source priority", 0, 10, 5)
        plausible = st.multiselect("Calibrated plausible action set", ["NEW", "REPLACE", "MERGE", "ARCHIVE", "COEXIST", "SPLIT"], default=["REPLACE", "COEXIST", "MERGE"])
        if st.button("Analyze non-live candidate states", type="primary"):
            probabilities = {"NEW": 0.01, "REPLACE": 0.82, "MERGE": 0.07, "ARCHIVE": 0.01, "COEXIST": 0.08, "SPLIT": 0.01}
            st.session_state.v2_incoming = incoming(text, int(priority))
            st.session_state.v2_analysis = runtime.analyze_incoming_knowledge(st.session_state.v2_incoming, ["policy-v1", "account-context"], probabilities, plausible)
        if st.button("Use calibrated singleton REPLACE"):
            probabilities = {"NEW": 0.01, "REPLACE": 0.92, "MERGE": 0.02, "ARCHIVE": 0.01, "COEXIST": 0.03, "SPLIT": 0.01}
            st.session_state.v2_incoming = incoming(text, int(priority))
            st.session_state.v2_analysis = runtime.analyze_incoming_knowledge(st.session_state.v2_incoming, ["policy-v1", "account-context"], probabilities, ["REPLACE"])
        if "v2_analysis" in st.session_state:
            st.write(st.session_state.v2_analysis.selection)

    analysis = st.session_state.get("v2_analysis")
    plan_frame, risk_frame, invariant_frame = analysis_frames(analysis) if analysis else (pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    with tabs[2]:
        st.header("Deterministic predecessor candidates")
        st.dataframe(pd.DataFrame([{"rank": index + 1, "identifier": identifier, "text": runtime.state.records[identifier].text if identifier in runtime.state.records else "historical"} for index, identifier in enumerate(analysis.predecessor_candidate_ids if analysis else [])]), hide_index=True, use_container_width=True)
        st.caption("Dense retrieval was selected on DEV. In the confirmatory TEST run it remained stronger than the locked weighted-RRF hybrid; TEST was not used for tuning.")

    with tabs[3]:
        st.header("Lifecycle probabilities and conformal set")
        if analysis:
            st.bar_chart(pd.Series(analysis.lifecycle_probabilities))
            st.write("Prediction set:", analysis.conformal_action_set)
        else: st.warning("Run analysis first.")

    with tabs[4]:
        st.header("Non-live TransitionPlan overlays")
        st.dataframe(plan_frame, hide_index=True, use_container_width=True)
        st.caption("Multiple hypotheses are visible for comparison; the singleton rule quarantines ambiguous sets.")

    with tabs[5]:
        st.header("Witness-query retrieval risk")
        if analysis: st.write("Witness queries:", analysis.witness_queries)
        st.dataframe(risk_frame, hide_index=True, use_container_width=True)

    with tabs[6]:
        st.header("Hard invariant results")
        st.dataframe(invariant_frame, hide_index=True, use_container_width=True)

    with tabs[7]:
        st.header("Epoch commit and exact rollback")
        c1, c2, c3 = st.columns(3)
        if c1.button("Commit accepted transition", disabled=not analysis or analysis.selection.chosen_action is None):
            st.session_state.v2_commit = runtime.commit_transition(analysis)
        if c2.button("Inject graph-prepare failure"):
            demo = new_runtime(); candidate = incoming("Effective 2026, the fee is 1.2 percent.", 5)
            safe = demo.analyze_incoming_knowledge(candidate, ["policy-v1"], {"REPLACE": 0.99}, ["REPLACE"])
            st.session_state.v2_failure = demo.commit_transition(safe, fail_at=FailureStage.AFTER_GRAPH_PREPARE)
            st.session_state.v2_failure_epoch = demo.state.epoch_id
        if c3.button("Rollback current transition", disabled=runtime.state.transition_id == "GENESIS"):
            runtime.transactions.rollback(runtime.state.transition_id); st.success("Exact prior state hash restored.")
        st.metric("Live epoch", runtime.state.epoch_id)
        if "v2_failure" in st.session_state:
            st.error(f"Injected commit published={st.session_state.v2_failure.committed}; readable epoch={st.session_state.v2_failure_epoch}")

    with tabs[8]:
        st.header("Deterministic transition certificate")
        if "v2_commit" in st.session_state: st.json(st.session_state.v2_commit.certificate.as_dict())
        elif analysis: st.json(analysis.transition_certificate_preview)

    with tabs[9]:
        st.header("Bitemporal and authority-aware view")
        st.dataframe(pd.DataFrame([{"id": key, "lifecycle": value.lifecycle_state.value, "retrieval": value.retrieval_state.value, "valid_from": value.valid_from, "valid_to": value.valid_to, "system_from": value.system_from, "priority": value.authority.source_priority} for key, value in sorted(runtime.state.records.items())]), hide_index=True, use_container_width=True)
        st.write("Active retrieval:", runtime.state.search((1.0, 0.0), 5))
        st.write("Historical retrieval:", runtime.state.search((1.0, 0.0), 5, historical=True))

    with tabs[10]:
        st.header("Matched feature ablation")
        if data.get("status") == "CONFIRMATORY_TEST_EXECUTED":
            frame = pd.DataFrame([{"model": name, "macro_f1": values["macro_f1"], "COEXIST_f1": values["per_class"]["COEXIST"]["f1-score"]} for name, values in data["model_metrics"].items()])
            st.dataframe(frame, hide_index=True, use_container_width=True); st.bar_chart(frame.set_index("model"))
        elif data:
            frame = pd.DataFrame([{"model": name, "macro_f1": values["TEST"]["macro_f1"], "COEXIST_f1": values["TEST"]["per_class"]["COEXIST"]["f1-score"]} for name, values in data["flat_ablation"].items()])
            st.dataframe(frame, hide_index=True, use_container_width=True); st.bar_chart(frame.set_index("model"))
        st.warning("All calculations are classical. The confirmatory run did not establish Q-specific advantage.")

    with tabs[11]:
        st.header("Local academic technical effects")
        if data: st.json(data["technical_effects"])
        st.caption("Dashboard HTTP availability is a smoke check, not a research result.")


if __name__ == "__main__":
    main()
