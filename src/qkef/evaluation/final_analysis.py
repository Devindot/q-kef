"""Frozen Phase 4 post-hoc analysis. This module never imports model fitting code."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "qkef-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score

from qkef.datasets.evolution import load_jsonl_models, sha256_file
from qkef.evaluation.phase3 import _hybrid_search
from qkef.evolution.knowledge_base import EvolutionKnowledgeBase, KBRecord
from qkef.evolution.models import CLASS_ORDER
from qkef.runtime.artifacts import FinalArtifactLoader, LoadedArtifacts
from qkef.schemas import EvolutionEvent, KnowledgeChunk


def supplementary_rates_for_rankings(
    rankings: list[list[str]], obsolete: set[str], lineage_relevant: list[set[str]], current_relevant: list[set[str]], k: int,
) -> dict[str, float]:
    """Exact small-example metric formulas used by regression tests and documentation."""
    top = [ranking[:k] for ranking in rankings]
    returned = sum(len(items) for items in top)
    return {
        "active_valid_hit_rate": sum(item not in obsolete for items in top for item in items) / max(1, returned),
        "obsolete_free_query_rate": sum(not set(items) & obsolete for items in top) / max(1, len(top)),
        "lineage_normalized_hit_rate": sum(bool(set(items) & relevant) for items, relevant in zip(top, lineage_relevant)) / max(1, len(top)),
        "current_version_hit_rate": sum(bool(set(items) & relevant) for items, relevant in zip(top, current_relevant)) / max(1, len(top)),
    }


def paired_bootstrap(predictions: pd.DataFrame, *, iterations: int = 5000, seed: int = 42) -> dict[str, Any]:
    truth = predictions["true_action"].to_numpy()
    conventional = predictions["conventional_prediction"].to_numpy()
    qkef = predictions["qkef_prediction"].to_numpy()
    rng = np.random.default_rng(seed)
    samples = {name: [] for name in ("b_accuracy", "c_accuracy", "accuracy_difference", "b_macro_f1", "c_macro_f1", "macro_f1_difference")}
    for _ in range(iterations):
        indices = rng.integers(0, len(truth), len(truth))
        y, b, c = truth[indices], conventional[indices], qkef[indices]
        b_acc, c_acc = accuracy_score(y, b), accuracy_score(y, c)
        b_f1 = f1_score(y, b, labels=CLASS_ORDER, average="macro", zero_division=0)
        c_f1 = f1_score(y, c, labels=CLASS_ORDER, average="macro", zero_division=0)
        for name, value in (("b_accuracy", b_acc), ("c_accuracy", c_acc), ("accuracy_difference", c_acc-b_acc), ("b_macro_f1", b_f1), ("c_macro_f1", c_f1), ("macro_f1_difference", c_f1-b_f1)):
            samples[name].append(float(value))
    intervals = {name: {"lower": float(np.percentile(values, 2.5)), "upper": float(np.percentile(values, 97.5))} for name, values in samples.items()}
    b_correct, c_correct = conventional == truth, qkef == truth
    b_only = int(np.sum(b_correct & ~c_correct)); c_only = int(np.sum(~b_correct & c_correct))
    discordant = b_only + c_only
    p_value = float(binomtest(min(b_only, c_only), discordant, 0.5, alternative="two-sided").pvalue) if discordant else 1.0
    return {"seed": seed, "iterations": iterations, "confidence_level": 0.95, "intervals": intervals, "mcnemar": {"b_correct_c_wrong": b_only, "b_wrong_c_correct": c_only, "discordant_total": discordant, "exact_p_value": p_value}}


def _rebuild_systems(artifacts: LoadedArtifacts) -> tuple[dict[str, EvolutionKnowledgeBase], list[EvolutionEvent], dict[str, np.ndarray]]:
    root = artifacts.root
    events = sorted(load_jsonl_models(root / "data/processed/qkef_fiqa_evolution/events.jsonl", EvolutionEvent), key=lambda value: value.event_id)
    units = {unit.knowledge_id: unit for unit in artifacts.units}
    dimensions = artifacts.manifest["embedding"]["dimension"]
    systems = {name: EvolutionKnowledgeBase(dimensions, append_only=name == "append_only") for name in ("append_only", "conventional", "qkef", "oracle")}
    for kb in systems.values():
        for unit in artifacts.units:
            if unit.temporal_state.value == "T0":
                vector = artifacts.embedding_vectors[artifacts.embedding_index[unit.knowledge_id]].copy()
                kb.add(KBRecord(unit.knowledge_id, unit.model_text, vector, list(unit.source_document_ids)))
    conventional_features = pd.read_csv(root / "data/processed/qkef_phase3/conventional_features.csv").drop(columns="sample_id").to_numpy()
    qkef_features = pd.read_csv(root / "data/processed/qkef_phase3/qkef_features.csv").drop(columns="sample_id").to_numpy()
    conventional_predictions = artifacts.conventional_model.predict(conventional_features)
    qkef_predictions = artifacts.qkef_model.predict(qkef_features)
    chunks = load_jsonl_models(root / "data/processed/qkef_fiqa_chunks/chunks_tfidf_boundary.jsonl", KnowledgeChunk)
    chunk_texts: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for chunk in chunks:
        chunk_texts[chunk.knowledge_id].append((chunk.chunk_index, chunk.text))
    t0_by_split = defaultdict(list)
    for unit in artifacts.units:
        if unit.temporal_state.value == "T0":
            t0_by_split[unit.benchmark_split.value].append(unit)
    for index, event in enumerate(events):
        incoming = units[event.incoming_knowledge_ids[0]]
        vector = artifacts.embedding_vectors[artifacts.embedding_index[incoming.knowledge_id]]
        candidates = sorted(
            ((unit.knowledge_id, float(np.dot(vector, artifacts.embedding_vectors[artifacts.embedding_index[unit.knowledge_id]]))) for unit in t0_by_split[event.benchmark_split.value]),
            key=lambda item: (-item[1], item[0]),
        )[:artifacts.locked_config["candidate_top_k"]]
        segments = [text for _, text in sorted(chunk_texts[incoming.knowledge_id])]
        def record() -> KBRecord:
            return KBRecord(incoming.knowledge_id, incoming.model_text, vector.copy(), list(incoming.source_document_ids))
        systems["append_only"].execute("NEW", record(), [])
        systems["conventional"].execute(str(conventional_predictions[index]), record(), [item[0] for item in candidates[:1]], split_texts=segments)
        systems["qkef"].execute(str(qkef_predictions[index]), record(), [item[0] for item in candidates[:1]], split_texts=segments)
        systems["oracle"].execute(event.expected_action.value, record(), list(event.expected_target_ids), split_texts=segments)
    query_vectors = {}
    for event in events:
        for query_id in event.source_query_ids:
            opaque = "qry_" + hashlib.sha256(query_id.encode()).hexdigest()[:20]
            query_vectors[query_id] = artifacts.embedding_vectors[artifacts.embedding_index[opaque]]
    return systems, events, query_vectors


def supplementary_retrieval(artifacts: LoadedArtifacts) -> dict[str, Any]:
    from qkef.datasets.fiqa import load_fiqa_dataset
    systems, events, query_vectors = _rebuild_systems(artifacts)
    fiqa = load_fiqa_dataset(artifacts.root / artifacts.config["dataset"]["raw_dir"] / "fiqa")
    oracle_obsolete = {identifier for identifier, record in systems["oracle"].records.items() if record.status != "active"}
    evaluated = [event for event in events if event.source_query_ids]
    results = {}
    for system_name, kb in systems.items():
        totals = Counter()
        replacement_queries = 0
        for event in evaluated:
            query_id = event.source_query_ids[0]
            ranked5 = _hybrid_search(kb, query_vectors[query_id], fiqa.queries[query_id].text, 5, artifacts.locked_config["hybrid_weights"])
            ranked1 = ranked5[:1]
            for k, ranked in ((1, ranked1), (5, ranked5)):
                totals[f"active_valid_items@{k}"] += sum(identifier not in oracle_obsolete for identifier in ranked)
                totals[f"returned_items@{k}"] += len(ranked)
                totals[f"obsolete_free_queries@{k}"] += int(not set(ranked) & oracle_obsolete)
                lineage_hit = any(set(kb.records[identifier].source_ids) & set(event.source_document_ids) for identifier in ranked if identifier in kb.records)
                totals[f"lineage_hit_queries@{k}"] += int(lineage_hit)
            if event.expected_action.value == "REPLACE":
                replacement_queries += 1
                for k, ranked in ((1, ranked1), (5, ranked5)):
                    current_hit = any(kb.records[identifier].status == "active" and set(kb.records[identifier].source_ids) & set(event.source_document_ids) for identifier in ranked if identifier in kb.records)
                    totals[f"current_version_queries@{k}"] += int(current_hit)
        values = {}
        for k in (1, 5):
            values[f"active_valid_hit_rate@{k}"] = totals[f"active_valid_items@{k}"] / max(1, totals[f"returned_items@{k}"])
            values[f"obsolete_free_query_rate@{k}"] = totals[f"obsolete_free_queries@{k}"] / len(evaluated)
            values[f"current_version_hit_rate@{k}"] = totals[f"current_version_queries@{k}"] / max(1, replacement_queries)
            values[f"lineage_normalized_hit_rate@{k}"] = totals[f"lineage_hit_queries@{k}"] / len(evaluated)
        values["query_denominator"] = len(evaluated)
        values["replacement_query_denominator"] = replacement_queries
        results[system_name] = values
    return {"definitions": {
        "active_valid_hit_rate": "Retrieved-item fraction not marked archived/superseded by the oracle lifecycle state.",
        "obsolete_free_query_rate": "Query fraction with no oracle-obsolete identifier in top-k.",
        "current_version_hit_rate": "Among REPLACE queries, fraction whose top-k contains an active record representing the event source lineage.",
        "lineage_normalized_hit_rate": "Query fraction with at least one top-k active record whose source lineage intersects the query event's source documents.",
    }, "systems": results}


def _bar(path: Path, labels: list[str], values: list[float], title: str, ylabel: str, *, ylim: tuple[float, float] = (0, 1)) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8)); bars = ax.bar(labels, values, color="#315c8c")
    ax.set_title(title); ax.set_ylabel(ylabel); ax.set_ylim(*ylim); ax.grid(axis="y", alpha=.25)
    for bar, value in zip(bars, values): ax.text(bar.get_x()+bar.get_width()/2, value+.015, f"{value:.3f}", ha="center", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def generate_figures(root: Path, final_results: dict[str, Any]) -> None:
    directory = root / "reports/final/figures"; directory.mkdir(parents=True, exist_ok=True)
    candidate = final_results["candidate_retrieval"]
    _bar(directory/"candidate_retrieval.png", ["R@1", "R@3", "R@5", "MRR"], [candidate["recall@1"], candidate["recall@3"], candidate["recall@5"], candidate["mrr"]], "Candidate predecessor retrieval", "Score")
    models = final_results["lifecycle_classification"]
    _bar(directory/"lifecycle_model_comparison.png", ["Conventional B", "Q-KEF C"], [models["conventional"]["TEST"]["macro_f1"], models["qkef"]["TEST"]["macro_f1"]], "Held-out lifecycle macro F1", "Macro F1", ylim=(0.7, 1.0))
    per_class = models["qkef"]["TEST"]["per_class"]
    fig, ax = plt.subplots(figsize=(10, 5)); x=np.arange(len(CLASS_ORDER)); width=.36
    ax.bar(x-width/2, [models["conventional"]["TEST"]["per_class"][name]["f1-score"] for name in CLASS_ORDER], width, label="Conventional B")
    ax.bar(x+width/2, [per_class[name]["f1-score"] for name in CLASS_ORDER], width, label="Q-KEF C")
    ax.set_xticks(x, CLASS_ORDER); ax.set_ylim(0,1.05); ax.set_ylabel("F1"); ax.set_title("Per-class held-out F1"); ax.legend(); ax.grid(axis="y",alpha=.25); fig.tight_layout(); fig.savefig(directory/"per_class_f1.png",dpi=180); plt.close(fig)
    for name, filename, title in (("conventional", "conventional_confusion_matrix.png", "Conventional B confusion matrix"), ("qkef", "qkef_confusion_matrix.png", "Q-KEF C confusion matrix")):
        matrix = pd.read_csv(root/f"reports/phase3/{name}_test_confusion_matrix.csv", index_col=0).to_numpy()
        fig, ax=plt.subplots(figsize=(7,6)); image=ax.imshow(matrix,cmap="Blues",vmin=0,vmax=10)
        ax.set_xticks(range(6),CLASS_ORDER,rotation=35,ha="right"); ax.set_yticks(range(6),CLASS_ORDER); ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
        for i in range(6):
            for j in range(6): ax.text(j,i,str(matrix[i,j]),ha="center",va="center")
        fig.colorbar(image,ax=ax); fig.tight_layout(); fig.savefig(directory/filename,dpi=180); plt.close(fig)
    retrieval=final_results["retrieval"]
    labels=["Append-only A","Conventional B","Q-KEF C","Oracle"]; keys=["append_only","conventional","qkef","oracle"]
    fig, ax=plt.subplots(figsize=(9,5)); x=np.arange(4); width=.35
    ax.bar(x-width/2,[retrieval[k]["recall@5"] for k in keys],width,label="Recall@5"); ax.bar(x+width/2,[retrieval[k]["ndcg@5"] for k in keys],width,label="nDCG@5")
    ax.set_xticks(x,labels,rotation=15); ax.set_ylim(0,1); ax.set_title("Retrieval comparison"); ax.legend(); ax.grid(axis="y",alpha=.25); fig.tight_layout(); fig.savefig(directory/"retrieval_comparison.png",dpi=180); plt.close(fig)
    _bar(directory/"obsolete_retrieval_rate.png", labels, [retrieval[k]["obsolete_rate@5"] for k in keys], "Obsolete retrieval rate at 5", "Rate")
    _bar(directory/"quantum_ablation.png", ["Without Q features", "With Q features"], [final_results["ablation"]["conventional_macro_f1"], final_results["ablation"]["qkef_macro_f1"]], "Quantum-inspired feature ablation", "TEST macro F1", ylim=(0.7,1.0))


def build_final_results(root: Path, *, iterations: int = 5000) -> dict[str, Any]:
    artifacts = FinalArtifactLoader(root).load()
    predictions = pd.read_csv(root / "reports/phase3/test_predictions.csv")
    statistical = paired_bootstrap(predictions, iterations=iterations, seed=42)
    supplementary = supplementary_retrieval(artifacts)
    metrics = artifacts.model_metrics
    result = {
        "release": "Phase 4 Final Academic Release", "scientific_results_frozen": True,
        "dataset": {"name": "BEIR FiQA", "corpus_documents": 57638, "queries": 6648, "evolution_events": 300, "train": 180, "dev": 60, "test": 60},
        "embedding": artifacts.manifest["embedding"], "candidate_retrieval": artifacts.manifest["candidate_metrics"],
        "model_metadata": {"classifier_family": "StandardScaler + class-balanced LogisticRegression", "conventional_C": .1, "qkef_C": 1.0, "state_dimension": 16, "pca_fit_split": "TRAIN"},
        "lifecycle_classification": metrics,
        "ablation": {"conventional_macro_f1": metrics["conventional"]["TEST"]["macro_f1"], "qkef_macro_f1": metrics["qkef"]["TEST"]["macro_f1"], "absolute_macro_f1_difference": metrics["qkef"]["TEST"]["macro_f1"]-metrics["conventional"]["TEST"]["macro_f1"]},
        "retrieval": artifacts.retrieval_metrics, "supplementary_retrieval": supplementary, "statistical_analysis": statistical,
        "graph_statistics": artifacts.manifest["graph_statistics"],
        "known_warnings": ["Three SPLIT executions per learned/oracle system used safe fallback.", "COEXIST is the weakest held-out class.", "Joblib emits harmless NumPy 2.5 deprecation warnings during reload."],
    }
    return result


def write_outputs(root: Path, results: dict[str, Any]) -> None:
    final_dir=root/"reports/final"; tables=final_dir/"tables"; tables.mkdir(parents=True,exist_ok=True)
    (final_dir/"final_results.json").write_text(json.dumps(results,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    (final_dir/"statistical_analysis.json").write_text(json.dumps(results["statistical_analysis"],indent=2,sort_keys=True)+"\n",encoding="utf-8")
    stats=results["statistical_analysis"]; ci=stats["intervals"]; mc=stats["mcnemar"]
    interpretation = "The paired accuracy difference was not statistically significant at the 0.05 level under McNemar's test." if mc["exact_p_value"] >= .05 else "The paired accuracy difference was statistically significant at 0.05 only for this controlled benchmark."
    (final_dir/"STATISTICAL_ANALYSIS.md").write_text(f"# Statistical Analysis\n\nPaired bootstrap: {stats['iterations']} iterations, seed {stats['seed']}, 95% percentile intervals.\n\n| Quantity | 95% CI |\n|---|---|\n"+"\n".join(f"| {name} | [{value['lower']:.4f}, {value['upper']:.4f}] |" for name,value in ci.items())+f"\n\nExact McNemar/binomial discordance: B-only correct={mc['b_correct_c_wrong']}, C-only correct={mc['b_wrong_c_correct']}, p={mc['exact_p_value']:.6f}. {interpretation}\n\nThe 60-item TEST set is modest; intervals quantify resampling uncertainty rather than external generalizability.\n",encoding="utf-8")
    pd.DataFrame([{"metric":key,"value":value} for key,value in results["candidate_retrieval"].items()]).to_csv(tables/"candidate_retrieval.csv",index=False)
    pd.DataFrame([{"system":s,"accuracy":results["lifecycle_classification"][s]["TEST"]["accuracy"],"macro_f1":results["lifecycle_classification"][s]["TEST"]["macro_f1"]} for s in ("conventional","qkef")]).to_csv(tables/"lifecycle_summary.csv",index=False)
    pd.DataFrame([{"action":a,"conventional_f1":results["lifecycle_classification"]["conventional"]["TEST"]["per_class"][a]["f1-score"],"qkef_f1":results["lifecycle_classification"]["qkef"]["TEST"]["per_class"][a]["f1-score"]} for a in CLASS_ORDER]).to_csv(tables/"per_class_f1.csv",index=False)
    pd.DataFrame([{"system":name,**values} for name,values in results["retrieval"].items()]).to_csv(tables/"retrieval_comparison.csv",index=False)
    pd.DataFrame([results["ablation"]]).to_csv(tables/"quantum_ablation.csv",index=False)
    pd.DataFrame([{"metric":name,**value} for name,value in ci.items()]).to_csv(tables/"bootstrap_analysis.csv",index=False)
    pd.DataFrame([{"system":name,**values} for name,values in results["graph_statistics"].items()]).to_csv(tables/"graph_statistics.csv",index=False)
    pd.DataFrame([{"system":name,**values} for name,values in results["supplementary_retrieval"]["systems"].items()]).to_csv(tables/"supplementary_lifecycle_retrieval.csv",index=False)
    generate_figures(root,results)
    ci=results["statistical_analysis"]["intervals"]; mc=results["statistical_analysis"]["mcnemar"]
    supp=results["supplementary_retrieval"]["systems"]
    report=f"""# Final Experiment Report

## 1. Research question

Can pre-indexing lifecycle management reduce obsolete retrieval, and does a classical quantum-inspired feature augmentation add measurable held-out value beyond conventional evolution features?

## 2. Dataset and controlled temporal benchmark

Canonical BEIR FiQA contains 57,638 documents and 6,648 queries. The reproducible Q-KEF benchmark contains 300 controlled/synthetic temporal events—50 per action—with 180 TRAIN, 60 DEV, and 60 held-out TEST events. These updates are not historical enterprise events.

## 3. Ingestion, chunking, and leakage controls

Phase 2 preserves raw text/provenance and provides sanitized `model_text`, identity representations for event classification, and TF-IDF boundary chunks for retrieval/SPLIT. Source ancestry overlap is zero. Action, relation, mutation, target, split, and provenance identifiers are absent from model X. PCA uses TRAIN ancestry only; TEST was not used for selection.

## 4. Systems

- **A append-only:** retains all T0/T1 knowledge.
- **B conventional:** MiniLM candidates plus safe conventional features and balanced logistic regression (C=0.1).
- **C Q-KEF:** same candidates/classifier family plus a TRAIN-only PCA-16 normalized state and fidelity-like, entropy, coherence-like, and Hellinger features (C=1.0).
- **Oracle:** executes benchmark lifecycle actions for descriptive comparison.

All quantum-inspired calculations are classical. No LLM, quantum hardware, or simulator is involved.

## 5. Candidate retrieval

For 150 eligible REPLACE/MERGE/COEXIST events: R@1={results['candidate_retrieval']['recall@1']:.4f}, R@3={results['candidate_retrieval']['recall@3']:.4f}, R@5={results['candidate_retrieval']['recall@5']:.4f}, MRR={results['candidate_retrieval']['mrr']:.4f}. NEW/SPLIT have no target; ARCHIVE is an explicit-reference administrative action.

## 6. Held-out lifecycle classification

| System | Accuracy | Balanced accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|
| Conventional B | 0.8500 | 0.8500 | 0.8455 | 0.8455 |
| Q-KEF C | 0.9000 | 0.9000 | 0.8933 | 0.8933 |

The quantum-inspired ablation difference is **+{results['ablation']['absolute_macro_f1_difference']:.4f} absolute macro F1** on this controlled held-out benchmark.

## 7. Per-class and error analysis

Q-KEF F1 is NEW 0.7826, REPLACE 1.0000, MERGE 1.0000, ARCHIVE 1.0000, COEXIST 0.6250, SPLIT 0.9524. COEXIST is weakest: Conventional classified six of ten COEXIST cases as NEW; Q-KEF classified four as NEW and one as SPLIT. This supports boundary ambiguity, not a causal explanation.

## 8. Frozen retrieval results

| System | Recall@5 | MRR | nDCG@5 | Obsolete@5 |
|---|---:|---:|---:|---:|
| Append-only A | 0.6717 | 0.6503 | 0.6109 | 0.3393 |
| Conventional B | 0.7218 | 0.6494 | 0.6612 | 0.1173 |
| Q-KEF C | 0.7225 | 0.6506 | 0.6613 | 0.1180 |
| Oracle | 0.5893 | 0.5422 | 0.5466 | 0.0000 |

Oracle's lower source-record recall follows consolidation under the fixed relevance definition; relevance was not changed post hoc.

## 9. Supplementary lifecycle-aware retrieval

Denominators are 300 evaluable queries and 50 REPLACE queries. Q-KEF active-valid hit rate is {supp['qkef']['active_valid_hit_rate@1']:.4f}@1/{supp['qkef']['active_valid_hit_rate@5']:.4f}@5; obsolete-free query rate is {supp['qkef']['obsolete_free_query_rate@1']:.4f}@1/{supp['qkef']['obsolete_free_query_rate@5']:.4f}@5; current-version hit is {supp['qkef']['current_version_hit_rate@1']:.4f}@1/{supp['qkef']['current_version_hit_rate@5']:.4f}@5. The lineage-normalized metric gives binary query credit when any active retrieved record represents source lineage: Q-KEF {supp['qkef']['lineage_normalized_hit_rate@1']:.4f}@1/{supp['qkef']['lineage_normalized_hit_rate@5']:.4f}@5 and oracle {supp['oracle']['lineage_normalized_hit_rate@1']:.4f}@1/{supp['oracle']['lineage_normalized_hit_rate@5']:.4f}@5. These are descriptive additions, not replacements for frozen metrics.

## 10. Statistical uncertainty

Using 5,000 paired bootstrap resamples (seed 42), B accuracy CI is [{ci['b_accuracy']['lower']:.4f}, {ci['b_accuracy']['upper']:.4f}], C accuracy [{ci['c_accuracy']['lower']:.4f}, {ci['c_accuracy']['upper']:.4f}], difference [{ci['accuracy_difference']['lower']:.4f}, {ci['accuracy_difference']['upper']:.4f}]. B macro-F1 CI is [{ci['b_macro_f1']['lower']:.4f}, {ci['b_macro_f1']['upper']:.4f}], C [{ci['c_macro_f1']['lower']:.4f}, {ci['c_macro_f1']['upper']:.4f}], difference [{ci['macro_f1_difference']['lower']:.4f}, {ci['macro_f1_difference']['upper']:.4f}]. Exact McNemar discordance is B-only={mc['b_correct_c_wrong']}, C-only={mc['b_wrong_c_correct']}, p={mc['exact_p_value']:.4f}. The paired accuracy difference is not statistically significant at 0.05.

## 11. Limitations

The controlled benchmark is modest; transformations are synthetic; MiniLM is generic; COEXIST is difficult; SPLIT overlaps segmentation and has three safe fallbacks per learned/oracle system; retrieval depends on FiQA provenance; target resolution differs from classification; broader longitudinal and human evaluation is required.

## 12. Conclusion

On the controlled temporal FiQA benchmark, evolution-aware management substantially reduced observed obsolete retrieval relative to append-only storage. Q-KEF achieved higher descriptive held-out macro F1 than B, but its paired confidence interval includes zero and McNemar p=0.3750. The result therefore supports further investigation—not quantum advantage, statistical superiority, or production generalizability.
"""
    (root/"reports/FINAL_EXPERIMENT_REPORT.md").write_text(report,encoding="utf-8")


def release_hashes(root: Path) -> dict[str, str]:
    paths=[root/"app.py",root/"reports/FINAL_EXPERIMENT_REPORT.md",root/"reports/final/final_results.json",root/"reports/final/statistical_analysis.json",root/"data/demo/demo_examples.json",root/"FINAL_PROJECT_STATUS.md"]
    paths += sorted((root/"reports/final/figures").glob("*.png"))
    paths += sorted((root/"docs").glob("*.md"))
    return {str(path.relative_to(root)).replace("\\","/"):release_digest(path) for path in paths}


def release_digest(path: Path) -> str:
    """Cross-platform digest: normalize line endings for text, preserve binary bytes."""
    data = path.read_bytes()
    if path.suffix.lower() in {".md", ".json", ".py", ".csv", ".yaml", ".yml", ".txt"}:
        data = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(data).hexdigest()
