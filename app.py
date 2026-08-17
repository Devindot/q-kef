"""Q-KEF final academic Streamlit application."""

from __future__ import annotations

import hashlib
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from qkef.datasets.evolution import load_jsonl_models
from qkef.qa import EvidenceQA
from qkef.qa.extractive import EvidenceRecord
from qkef.runtime.demo import initial_demo_records, load_demo_examples, runtime_input
from qkef.runtime.results import load_final_results
from qkef.runtime.schemas import CLASS_ORDER
from qkef.runtime.service import DemoKnowledgeBase, QKEFService
from qkef.schemas import EvolutionEvent, IngestedKnowledgeUnit


ROOT = Path(__file__).resolve().parent
DISCLAIMER = "Academic research prototype. Results are limited to the controlled Q-KEF FiQA evolution benchmark. Quantum-inspired calculations are classical. This demo is not financial advice."


@st.cache_data
def results_data() -> dict:
    return load_final_results(ROOT)


@st.cache_data
def demo_data() -> list[dict]:
    return load_demo_examples(ROOT / "data/demo/demo_examples.json")


@st.cache_resource
def runtime_service() -> QKEFService:
    return QKEFService.load_default(local_files_only=True)


def probability_frame(result) -> pd.DataFrame:
    return pd.DataFrame({"Action": CLASS_ORDER, "Probability": [result.class_probabilities[name] for name in CLASS_ORDER]})


def graph_figure(graph: nx.MultiDiGraph, *, limit: int = 70) -> go.Figure:
    nodes = sorted(graph.nodes)[:limit]
    subgraph = graph.subgraph(nodes)
    positions = nx.spring_layout(subgraph, seed=42)
    edge_x, edge_y = [], []
    for source, target in subgraph.edges():
        x0, y0 = positions[source]; x1, y1 = positions[target]
        edge_x += [x0, x1, None]; edge_y += [y0, y1, None]
    edges = go.Scatter(x=edge_x, y=edge_y, mode="lines", line={"width": 1, "color": "#9aa4b2"}, hoverinfo="none")
    statuses = [str(subgraph.nodes[node].get("status", "unknown")) for node in subgraph.nodes]
    symbols = {"active": "circle", "superseded": "x", "archived": "square", "container": "diamond", "audit_notice": "triangle-up", "lineage": "star"}
    traces = [edges]
    for status in sorted(set(statuses)):
        selected = [node for node in subgraph.nodes if str(subgraph.nodes[node].get("status", "unknown")) == status]
        traces.append(go.Scatter(x=[positions[node][0] for node in selected], y=[positions[node][1] for node in selected], mode="markers", name=status.upper(), text=selected, hovertemplate="%{text}<extra>"+status+"</extra>", marker={"size": 10, "symbol": symbols.get(status, "circle-open")}))
    return go.Figure(traces, layout=go.Layout(height=560, showlegend=True, xaxis={"visible": False}, yaxis={"visible": False}, margin={"l": 10, "r": 10, "t": 30, "b": 10}))


def render_analysis(result, title: str) -> None:
    st.subheader(title)
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted action", result.predicted_action)
    c2.metric("Confidence", f"{result.confidence:.2%}")
    c3.metric("Top semantic similarity", f"{(result.semantic_similarity or 0):.3f}")
    for warning in result.warnings: st.warning(warning)
    st.plotly_chart(px.bar(probability_frame(result), x="Action", y="Probability", range_y=[0, 1], title="Six-class model probabilities"), use_container_width=True)
    if result.candidates:
        best = result.candidates[0]
        st.write(f"Best candidate: `{best.identifier}`")
        st.caption(best.excerpt)
    if result.quantum_features_enabled:
        q1, q2, q3 = st.columns(3)
        q1.metric("Fidelity-like", f"{(result.fidelity_like_similarity or 0):.3f}")
        q2.metric("State entropy", f"{(result.state_entropy or 0):.3f}")
        q3.metric("Coherence-like", f"{(result.coherence_like_score or 0):.3f}")
    st.info(result.kb_effect_preview)


def benchmark_error_frame() -> pd.DataFrame:
    predictions = pd.read_csv(ROOT / "reports/phase3/test_predictions.csv")
    features = pd.read_csv(ROOT / "data/processed/qkef_phase3/conventional_features.csv")
    labels = pd.read_csv(ROOT / "data/processed/qkef_phase3/evaluation_labels.csv")
    features = features.merge(labels, on="sample_id").query("split == 'test'")
    events = load_jsonl_models(ROOT / "data/processed/qkef_fiqa_evolution/events.jsonl", EvolutionEvent)
    event_by_sample = {"sample_" + hashlib.sha256(event.event_id.encode()).hexdigest()[:20]: event for event in events}
    units = load_jsonl_models(ROOT / "data/processed/qkef_fiqa_chunks/ingested_t0.jsonl", IngestedKnowledgeUnit)
    units += load_jsonl_models(ROOT / "data/processed/qkef_fiqa_chunks/ingested_t1.jsonl", IngestedKnowledgeUnit)
    unit_by_id = {unit.knowledge_id: unit for unit in units}
    merged = predictions.merge(features, on="sample_id", how="left")
    merged["safe_incoming_excerpt"] = [unit_by_id[event_by_sample[sample].incoming_knowledge_ids[0]].model_text[:280] for sample in merged.sample_id]
    return merged


def main() -> None:
    st.set_page_config(page_title="Q-KEF Final Academic Release", page_icon="🧭", layout="wide")
    results = results_data()
    examples = demo_data()
    try:
        service, runtime_error = runtime_service(), None
    except Exception as error:
        service, runtime_error = None, str(error)
    st.title("Q-KEF")
    st.caption("Quantum-Inspired Knowledge Evolution Framework for Dynamic Enterprise Semantic Graphs")
    tabs = st.tabs(["Overview", "Evolution Demo", "B vs C", "Retrieval", "Semantic Graph", "Q-State", "Evidence QA", "Research Results", "Error Analysis", "Methodology & Limits"])

    with tabs[0]:
        st.header("A pre-indexing lifecycle layer for changing knowledge")
        st.write("Append-only RAG can retain obsolete or contradictory information. Q-KEF predicts and executes one of six lifecycle actions before final retrieval indexing.")
        st.write(" → ".join(["Incoming Knowledge", "Sanitize", "MiniLM", "Candidate Search", "Conventional + Q-State Features", "Evolution Engine", "Evolution-Aware KB", "Vector + Graph Retrieval", "Evidence"]))
        st.write("Actions: " + " · ".join(CLASS_ORDER))
        c1,c2,c3,c4,c5=st.columns(5)
        c1.metric("Candidate R@5", f"{results['candidate_retrieval']['recall@5']:.2%}")
        c2.metric("B TEST Macro F1", f"{results['lifecycle_classification']['conventional']['TEST']['macro_f1']:.2%}")
        c3.metric("C TEST Macro F1", f"{results['lifecycle_classification']['qkef']['TEST']['macro_f1']:.2%}")
        c4.metric("Append-only obsolete@5", f"{results['retrieval']['append_only']['obsolete_rate@5']:.2%}")
        c5.metric("Q-KEF obsolete@5", f"{results['retrieval']['qkef']['obsolete_rate@5']:.2%}")
        st.warning("Results are from a controlled 300-event temporal FiQA benchmark.")

    with tabs[1]:
        st.header("Interactive Knowledge Evolution")
        selected = st.selectbox("Stored benchmark-backed example", range(len(examples)), format_func=lambda index: examples[index]["name"])
        text = st.text_area("Incoming knowledge", runtime_input(examples[selected]), height=220, max_chars=20_000)
        system = st.radio("System", ["conventional", "qkef"], horizontal=True)
        if runtime_error: st.error("Interactive MiniLM inference is unavailable offline on this machine. Stored results remain fully available. " + runtime_error)
        if st.button("Analyze Evolution", type="primary", disabled=service is None):
            st.session_state["analysis"] = service.analyze_incoming_knowledge(text, system)
            st.session_state["analysis_text"] = text
        if "analysis" in st.session_state:
            render_analysis(st.session_state["analysis"], "Prediction")
            if "demo_kb" not in st.session_state:
                st.session_state["demo_kb"] = DemoKnowledgeBase.from_records(initial_demo_records(examples))
            a,b=st.columns(2)
            if a.button("Apply to Demo KB"):
                st.session_state["demo_kb"].apply(st.session_state["analysis"], st.session_state["analysis_text"]); st.success("Applied to this session's isolated demo KB.")
            if b.button("Reset Demo"):
                st.session_state["demo_kb"].reset(initial_demo_records(examples)); st.success("Demo KB reset.")
            st.dataframe(pd.DataFrame(st.session_state["demo_kb"].records.values()), use_container_width=True)
        st.caption(examples[selected]["disclosure"])

    with tabs[2]:
        st.header("Identical input, controlled B/C comparison")
        compare_text=st.text_area("Comparison input",runtime_input(examples[1]),height=180,key="comparison")
        if st.button("Compare Conventional and Q-KEF",disabled=service is None):
            st.session_state["comparison_results"]=(service.analyze_incoming_knowledge(compare_text,"conventional"),service.analyze_incoming_knowledge(compare_text,"qkef"))
        if "comparison_results" in st.session_state:
            left,right=st.columns(2)
            with left: render_analysis(st.session_state["comparison_results"][0],"Conventional B")
            with right: render_analysis(st.session_state["comparison_results"][1],"Q-KEF C")

    with tabs[3]:
        st.header("Frozen retrieval comparison")
        names={"append_only":"Append-only A","conventional":"Conventional B","qkef":"Q-KEF C","oracle":"Oracle"}
        frame=pd.DataFrame([{"System":names[name],**values} for name,values in results["retrieval"].items()])
        st.dataframe(frame,use_container_width=True,hide_index=True)
        long=frame.melt(id_vars="System",value_vars=["recall@5","ndcg@5","obsolete_rate@5"],var_name="Metric",value_name="Value")
        st.plotly_chart(px.bar(long,x="System",y="Value",color="Metric",barmode="group",range_y=[0,1]),use_container_width=True)
        st.info("Oracle eliminates obsolete retrieval but achieves lower source-record Recall@5 because lifecycle consolidation reduces independently retrievable source records under the benchmark's source-lineage relevance definition.")
        st.subheader("Supplementary lifecycle-aware metrics")
        st.dataframe(pd.DataFrame([{"System":names[name],**values} for name,values in results["supplementary_retrieval"]["systems"].items()]),use_container_width=True,hide_index=True)

    with tabs[4]:
        st.header("Evolution-aware semantic graph")
        graph_system=st.selectbox("Graph system",["qkef","conventional","oracle","append_only"])
        if service:
            graph=service.artifacts.graphs[graph_system].graph.copy()
            statuses=sorted({str(data.get("status","unknown")) for _,data in graph.nodes(data=True)})
            status=st.multiselect("Status filter",statuses,default=statuses)
            edge_types=sorted({str(data.get("relation","unknown")) for *_,data in graph.edges(data=True)})
            selected_edges=st.multiselect("Edge type",edge_types,default=edge_types)
            demo_nodes=[item["existing"]["identifier"] for item in examples if item.get("existing") and item["existing"]["identifier"] in graph]
            anchor=st.selectbox("Example/event neighborhood",demo_nodes or sorted(graph.nodes)[:20])
            depth=st.slider("Neighborhood depth",1,3,1)
            neighborhood=nx.ego_graph(graph.to_undirected(),anchor,radius=depth).nodes
            selected_nodes=[node for node in neighborhood if str(graph.nodes[node].get("status","unknown")) in status]
            graph=graph.subgraph(selected_nodes).copy()
            remove=[(u,v,k) for u,v,k,data in graph.edges(keys=True,data=True) if str(data.get("relation","unknown")) not in selected_edges]
            graph.remove_edges_from(remove)
            st.plotly_chart(graph_figure(graph),use_container_width=True)
            st.caption("Symbols supplement color; only a deterministic manageable subgraph is displayed.")

    with tabs[5]:
        st.header("Classical quantum-inspired state")
        st.write("A frozen TRAIN-only PCA maps MiniLM embeddings to a 16-dimensional normalized real state. Squared amplitudes form a probability-like distribution.")
        st.info("No quantum hardware or quantum simulator is used. These values are classical descriptors, not qubits.")
        qtext=st.text_area("Knowledge for state inspection",runtime_input(examples[1]),height=150,key="qtext")
        if st.button("Compute Q-State",disabled=service is None): st.session_state["qstate"]=service.analyze_incoming_knowledge(qtext,"qkef")
        if "qstate" in st.session_state:
            result=st.session_state["qstate"]; frame=pd.DataFrame({"Component":list(range(1,17)),"Amplitude":result.state_amplitudes,"Squared amplitude":result.state_probabilities})
            st.plotly_chart(px.bar(frame,x="Component",y=["Amplitude","Squared amplitude"],barmode="group"),use_container_width=True)
            st.dataframe(frame,use_container_width=True,hide_index=True)

    with tabs[6]:
        st.header("Deterministic evidence-grounded QA")
        st.write("This is an extractive evidence composer, not a generative LLM.")
        question=st.text_input("Financial-domain question",examples[1]["qa_question"],help="This question comes from the stored successful REPLACE benchmark case.")
        if st.button("Compare Append-only and Q-KEF Evidence",disabled=service is None):
            answers=[]
            for system_name in ("append_only","qkef"):
                records=[]; graph=service.artifacts.graphs[system_name].graph
                for unit in service.artifacts.units:
                    if unit.knowledge_id in graph and graph.nodes[unit.knowledge_id].get("status")=="active":
                        records.append(EvidenceRecord(unit.knowledge_id,unit.model_text,list(unit.source_document_ids),"active",service.artifacts.embedding_vectors[service.artifacts.embedding_index[unit.knowledge_id]]))
                answers.append(EvidenceQA(records,service.encoder,system=system_name).answer(question))
            st.session_state["qa_compare"]=answers
        if "qa_compare" in st.session_state:
            for column,qa,title in zip(st.columns(2),st.session_state["qa_compare"],["Append-only A","Q-KEF C"]):
                with column:
                    st.subheader(title); st.success(qa.answer)
                    for score,passage,provenance in zip(qa.retrieval_scores,qa.supporting_passages,qa.provenance): st.write(f"Score {score:.3f} · provenance {provenance}"); st.caption(passage)
                    for warning in qa.warnings: st.warning(warning)

    with tabs[7]:
        st.header("Frozen research results")
        st.json({"candidate_retrieval":results["candidate_retrieval"],"classification":results["lifecycle_classification"],"ablation":results["ablation"]},expanded=False)
        per=pd.DataFrame([{"Action":action,"Conventional":results["lifecycle_classification"]["conventional"]["TEST"]["per_class"][action]["f1-score"],"Q-KEF":results["lifecycle_classification"]["qkef"]["TEST"]["per_class"][action]["f1-score"]} for action in CLASS_ORDER])
        st.plotly_chart(px.bar(per,x="Action",y=["Conventional","Q-KEF"],barmode="group",range_y=[0,1]),use_container_width=True)
        st.write(f"Ablation: **{results['ablation']['absolute_macro_f1_difference']:+.4f} absolute macro-F1 on this controlled held-out benchmark.**")

    with tabs[8]:
        st.header("Benchmark-only error analysis")
        errors=benchmark_error_frame(); true_filter=st.multiselect("True action",CLASS_ORDER,default=CLASS_ORDER)
        filtered=errors[errors.true_action.isin(true_filter)]
        f1,f2,f3=st.columns(3)
        b_filter=f1.multiselect("B prediction",CLASS_ORDER,default=CLASS_ORDER)
        c_filter=f2.multiselect("C prediction",CLASS_ORDER,default=CLASS_ORDER)
        minimum_confidence=f3.slider("Minimum confidence",0.0,1.0,0.0,0.05)
        filtered=filtered[filtered.conventional_prediction.isin(b_filter)&filtered.qkef_prediction.isin(c_filter)]
        filtered=filtered[(filtered.conventional_confidence>=minimum_confidence)|(filtered.qkef_confidence>=minimum_confidence)]
        correctness=st.selectbox("Correctness",["All","B incorrect","C incorrect","Either incorrect"])
        if correctness=="B incorrect": filtered=filtered[filtered.true_action!=filtered.conventional_prediction]
        elif correctness=="C incorrect": filtered=filtered[filtered.true_action!=filtered.qkef_prediction]
        elif correctness=="Either incorrect": filtered=filtered[(filtered.true_action!=filtered.conventional_prediction)|(filtered.true_action!=filtered.qkef_prediction)]
        st.dataframe(filtered,use_container_width=True,hide_index=True)
        st.caption("Ground truth is displayed only in this benchmark-results mode and is never used by interactive inference.")

    with tabs[9]:
        st.header("Methodology, limitations, and responsible use")
        st.write("TRAIN fit → DEV selection → locked configuration → one held-out TEST report. The Q-state basis uses TRAIN ancestry only. Conventional and Q-KEF share candidates, labels, splits, and classifier family.")
        st.write("Limitations include a controlled 300-event benchmark, synthetic temporal mutations, generic MiniLM embeddings, COEXIST ambiguity, SPLIT/segmentation overlap, and no production validation.")
        st.warning("The system does not establish quantum advantage, eliminate hallucinations, provide financial advice, prove patent novelty, or authorize autonomous deletion. Human review and audit retention remain necessary.")

    st.divider(); st.caption(DISCLAIMER)


if __name__ == "__main__":
    main()
