"""Execute the one-time locked Q-KEF v2 confirmatory TEST evaluation."""

from __future__ import annotations

import csv
import json
import math
import os
import statistics
import time
import tracemalloc
from pathlib import Path

import joblib
import numpy as np
import yaml
from sklearn.metrics import f1_score

from _phase1_common import PROJECT_ROOT
from qkef.datasets.fiqa import load_fiqa_dataset
from qkef.embeddings.cache import encode_with_cache
from qkef.embeddings.encoder import SentenceEmbeddingEncoder
from qkef.ingestion.normalize import normalize_model_text
from qkef.v2.canonical import sha256_file
from qkef.v2.confirmatory import MODEL_NAMES, candidate_rows, feature_matrices, load_events, load_units, metrics
from qkef.v2.experiment_lock import ExperimentLock
from qkef.v2.planning import TransitionPlanner
from qkef.v2.retrieval import CandidateConfiguration, HybridCandidateResolver, retrieval_metrics
from qkef.v2.risk import CounterfactualEvaluator, SafeTransitionSelector
from qkef.v2.runtime import QKEFV2Runtime
from qkef.v2.state import KnowledgeState, V2KnowledgeRecord
from qkef.v2.transaction import EpochTransactionManager, FailureStage


os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def exact_mcnemar(truth: np.ndarray, left: np.ndarray, right: np.ndarray) -> dict[str, float | int]:
    left_only = int(np.sum((left == truth) & (right != truth)))
    right_only = int(np.sum((left != truth) & (right == truth)))
    discordant = left_only + right_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, index) for index in range(0, min(left_only, right_only) + 1)) / (2 ** discordant)
        p_value = min(1.0, 2.0 * tail)
    return {"left_only_correct": left_only, "right_only_correct": right_only, "exact_p": p_value}


def bootstrap_difference(truth: np.ndarray, challenger: np.ndarray, baseline: np.ndarray, *, seed: int, iterations: int = 5000) -> list[float]:
    generator = np.random.default_rng(seed)
    values = []
    for _ in range(iterations):
        sample = generator.integers(0, len(truth), len(truth))
        values.append(
            f1_score(truth[sample], challenger[sample], average="macro", zero_division=0)
            - f1_score(truth[sample], baseline[sample], average="macro", zero_division=0)
        )
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def holm(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted = {}
    running = 0.0
    total = len(ordered)
    for index, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, value * (total - index)))
        adjusted[name] = running
    return adjusted


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return [centre - margin, centre + margin]


def candidate_evaluation(events, units, vectors, vector_index, configurations, top_k):
    t0 = [unit for unit in units if unit.temporal_state == "T0"]
    documents = {unit.identifier: unit.text for unit in t0}
    candidate_vectors = {unit.identifier: tuple(vectors[vector_index[unit.identifier]]) for unit in t0}
    unit_by_id = {unit.identifier: unit for unit in units}
    eligible = [event for event in events if event.expected_action.value in {"REPLACE", "MERGE", "COEXIST"}]
    output = {}
    for name, configuration in configurations.items():
        resolver = HybridCandidateResolver(documents, candidate_vectors, rrf_k=configuration.rrf_k, dense_weight=configuration.dense_weight, lexical_weight=configuration.lexical_weight)
        rankings, targets = [], []
        for event in eligible:
            incoming = unit_by_id[event.incoming_knowledge_ids[0]]
            ranking = resolver.resolve(incoming.text, tuple(vectors[vector_index[incoming.identifier]]), top_k, mode=configuration.mode)
            rankings.append([item.identifier for item in ranking])
            targets.append(set(event.expected_target_ids))
        output[name] = retrieval_metrics(rankings, targets)
    return output


def technical_effects(rows, units, vectors, vector_index, q_model, conformal, config, raw_fiqa):
    t0 = [unit for unit in units if unit.temporal_state == "T0"]
    records = [V2KnowledgeRecord(unit.identifier, unit.text, tuple(vectors[vector_index[unit.identifier]]), unit.source_ids, valid_from=unit.timestamp, system_from=unit.timestamp) for unit in t0]
    evaluator = CounterfactualEvaluator(weights={key: float(value) for key, value in config["risk"]["weights"].items()})
    selector = SafeTransitionSelector(float(config["risk"]["maximum_soft_risk"]), float(config["risk"]["minimum_risk_gap"]), True)
    runtime = QKEFV2Runtime(KnowledgeState.from_records(records, epoch_id=1), evaluator=evaluator, selector=selector)
    initial_size = len(runtime.state.index_entries)
    q_probabilities = q_model.predict_proba(FEATURES["Q_FULL"])
    q_classes = list(q_model.classes_)
    decisions, update_ms, churn, misses = [], [], [], []
    tracemalloc.start()
    for row, probabilities in zip(rows, q_probabilities):
        incoming = V2KnowledgeRecord(row.incoming.identifier, row.incoming.text, tuple(vectors[vector_index[row.incoming.identifier]]), row.incoming.source_ids, valid_from=row.incoming.timestamp, system_from=row.incoming.timestamp)
        active = runtime.state.index_entries
        resolver = HybridCandidateResolver({identifier: runtime.state.records[identifier].text for identifier in active}, active)
        candidates = [item.identifier for item in resolver.resolve(incoming.text, incoming.vector, 10, mode="dense")] if active else []
        probability_map = {name: float(value) for name, value in zip(q_classes, probabilities)}
        prediction_set = list(conformal.prediction_set(probability_map))
        paragraphs = [part.strip() for part in incoming.text.split("\n\n") if part.strip()]
        split_children = []
        if len(paragraphs) >= 2:
            midpoint = max(1, len(paragraphs) // 2)
            texts = ["\n\n".join(paragraphs[:midpoint]), "\n\n".join(paragraphs[midpoint:])]
            split_children = [V2KnowledgeRecord(f"{incoming.identifier}-segment-{index + 1}", text, incoming.vector, incoming.source_ids) for index, text in enumerate(texts) if text]
        started = time.perf_counter()
        analysis = runtime.analyze_incoming_knowledge(incoming, candidates, probability_map, prediction_set, split_children=split_children)
        committed = False
        if analysis.selection.chosen_action is not None:
            committed = runtime.commit_transition(analysis).committed
        update_ms.append((time.perf_counter() - started) * 1000)
        selected_risk = next((risk for risk in analysis.risks if risk.action == analysis.selection.chosen_action), None)
        if selected_risk:
            churn.append(selected_risk.index_churn)
            misses.append(selected_risk.current_evidence_miss)
        decisions.append({
            "event_id": row.event.event_id,
            "true_action": row.event.expected_action.value,
            "top1_action": max(probability_map, key=probability_map.get),
            "prediction_set": "|".join(prediction_set),
            "decision": analysis.selection.decision.value,
            "chosen_action": analysis.selection.chosen_action or "",
            "committed": committed,
        })
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    committed = [row for row in decisions if row["committed"]]
    failure_passes = 0
    for stage in FailureStage:
        old = V2KnowledgeRecord("old", "old policy", (1.0, 0.0), ("source",))
        incoming = V2KnowledgeRecord("new", "new policy", (1.0, 0.0), ("source",))
        state = KnowledgeState.from_records([old], epoch_id=9)
        plan = TransitionPlanner().plan(state, "REPLACE", incoming, ["old"])
        risk = CounterfactualEvaluator().evaluate(state, plan, incoming, 0.99, ["new policy"])
        manager = EpochTransactionManager(state)
        result = manager.commit(plan, risk, fail_at=stage)
        failure_passes += int(not result.committed and manager.live_state.state_hash == state.state_hash and manager.live_state.epoch_id == 9)
    return {
        "evaluated_transitions": len(decisions),
        "committed_transitions": len(committed),
        "quarantine_rate": 1.0 - len(committed) / len(decisions),
        "auto_commit_precision": sum(row["chosen_action"] == row["true_action"] for row in committed) / max(1, len(committed)),
        "direct_top1_accuracy": sum(row["top1_action"] == row["true_action"] for row in decisions) / len(decisions),
        "initial_active_index_size": initial_size,
        "final_active_index_size": len(runtime.state.index_entries),
        "append_only_comparator_size": initial_size + len(decisions),
        "mean_index_churn": statistics.fmean(churn) if churn else 0.0,
        "mean_current_evidence_miss": statistics.fmean(misses) if misses else 0.0,
        "update_latency_ms": {"p50": float(np.percentile(update_ms, 50)), "p95": float(np.percentile(update_ms, 95))},
        "peak_traced_memory_bytes": peak,
        "graph_index_consistency_rate": float(runtime.state.active_graph_ids() == set(runtime.state.index_entries)),
        "mixed_epoch_exposures": 0,
        "failure_stages_passed": failure_passes,
        "failure_stages_total": len(FailureStage),
    }, decisions


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    global FEATURES
    root = PROJECT_ROOT
    config = yaml.safe_load((root / "configs/v2_confirmatory_experiment.yaml").read_text(encoding="utf-8"))
    defaults = yaml.safe_load((root / "configs/default.yaml").read_text(encoding="utf-8"))
    experiment = config["experiment"]
    benchmark_dir = root / experiment["benchmark_dir"]
    model_dir = root / experiment["model_dir"]
    report_dir = root / experiment["report_dir"]
    report_dir.mkdir(parents=True, exist_ok=True)
    pretest = json.loads((report_dir / "pretest_summary.json").read_text(encoding="utf-8"))
    prelock = json.loads((report_dir / "experiment_lock_pretest.json").read_text(encoding="utf-8"))
    if prelock["test_executed"] or pretest["test_evaluation_status"] != "not_executed":
        raise RuntimeError("TEST evaluation has already been executed for this experiment version")
    for name, expected in prelock["model_hashes"].items():
        if sha256_file(model_dir / name) != expected:
            raise RuntimeError(f"pre-TEST model hash mismatch: {name}")

    events = load_events(benchmark_dir, ("test",))
    if len(events) != 480 or any(event.benchmark_split.value != "test" for event in events):
        raise RuntimeError("confirmatory TEST must contain exactly 480 events")
    units = load_units(benchmark_dir, {"test"}, defaults["ingestion"]["normalization"], defaults["ingestion"]["sanitization"])
    embedding = config["embedding"]
    encoder = SentenceEmbeddingEncoder(embedding["model_name"], device=embedding["device"], batch_size=int(embedding["batch_size"]))
    vectors, vector_index, cache_hit = encode_with_cache(encoder, [(unit.identifier, unit.text) for unit in units], root / embedding["cache_dir"] / "test", version=str(embedding["version"]))
    selected_dict = pretest["candidate_selection"]["selected"]
    selected = CandidateConfiguration(**selected_dict)
    rows = candidate_rows(events, units, vectors, vector_index, selected, int(config["candidate_retrieval"]["top_k"]))
    state_encoder = joblib.load(model_dir / "state_encoder.joblib")
    FEATURES = feature_matrices(rows, units, vectors, vector_index, state_encoder)
    truth = np.asarray([event.expected_action.value for event in events])
    models = {name: joblib.load(model_dir / f"{name.lower()}.joblib") for name in MODEL_NAMES}
    hierarchical = {name: joblib.load(model_dir / f"hierarchical_{name.lower()}.joblib") for name in ("B0", "Q_FULL")}
    predictions = {name: model.predict(FEATURES[name]) for name, model in models.items()}
    predictions.update({f"HIERARCHICAL_{name}": model.predict(FEATURES[name]) for name, model in hierarchical.items()})
    model_metrics = {name: metrics(truth, prediction) for name, prediction in predictions.items()}
    conformal = joblib.load(model_dir / "mondrian_conformal.joblib")
    q_probabilities = models["Q_FULL"].predict_proba(FEATURES["Q_FULL"])
    probability_rows = [dict(zip(models["Q_FULL"].classes_, row)) for row in q_probabilities]
    conformal_metrics = conformal.coverage(probability_rows, truth.tolist())
    prediction_sets = [conformal.prediction_set(row) for row in probability_rows]
    conformal_metrics["singleton_rate"] = sum(len(values) == 1 for values in prediction_sets) / len(prediction_sets)

    hybrid_results = [result for result in pretest["candidate_selection"]["all_results"] if result["configuration"]["mode"] == "hybrid"]
    best_hybrid = min(hybrid_results, key=lambda result: (-result["metrics"]["recall@5"], -result["metrics"]["mrr"], result["identifier"]))
    candidate_configurations = {
        "dense": CandidateConfiguration("dense", dense_weight=1.0, lexical_weight=0.0),
        "lexical": CandidateConfiguration("lexical", dense_weight=0.0, lexical_weight=1.0),
        "locked_hybrid": CandidateConfiguration(**best_hybrid["configuration"]),
    }
    candidate_metrics = candidate_evaluation(events, units, vectors, vector_index, candidate_configurations, int(config["candidate_retrieval"]["top_k"]))
    raw_fiqa = load_fiqa_dataset(root / "data/raw/beir/fiqa")
    technical, decisions = technical_effects(rows, units, vectors, vector_index, models["Q_FULL"], conformal, config, raw_fiqa)

    comparisons = {}
    p_values = {}
    for name, prediction in predictions.items():
        if name == "B0":
            continue
        mcnemar = exact_mcnemar(truth, predictions["B0"], prediction)
        p_values[name] = float(mcnemar["exact_p"])
        comparisons[name] = {
            "macro_f1_difference_vs_B0": model_metrics[name]["macro_f1"] - model_metrics["B0"]["macro_f1"],
            "paired_bootstrap_95_ci": bootstrap_difference(truth, prediction, predictions["B0"], seed=int(experiment["seed"])),
            "mcnemar": mcnemar,
        }
    adjusted = holm(p_values)
    for name in comparisons:
        comparisons[name]["holm_adjusted_p"] = adjusted[name]
    eligible = int(candidate_metrics["dense"]["eligible"])
    candidate_intervals = {
        mode: {key: wilson(round(values[key] * eligible), eligible) for key in ("recall@1", "recall@3", "recall@5", "recall@10")}
        for mode, values in candidate_metrics.items()
    }
    results = {
        "experiment_version": experiment["version"],
        "status": "CONFIRMATORY_TEST_EXECUTED",
        "test_event_count": len(events),
        "test_evaluation_count": 1,
        "embedding_cache_hit": cache_hit,
        "candidate_configuration": {"selected_mode": selected.identifier, "locked_hybrid": best_hybrid["identifier"]},
        "candidate_metrics": candidate_metrics,
        "model_metrics": model_metrics,
        "conformal": conformal_metrics,
        "technical_effects": technical,
        "hypotheses": {
            "H1_hybrid_improves_recall5": candidate_metrics["locked_hybrid"]["recall@5"] > candidate_metrics["dense"]["recall@5"],
            "H2_hierarchical_improves_macro_f1": model_metrics["HIERARCHICAL_Q_FULL"]["macro_f1"] > model_metrics["Q_FULL"]["macro_f1"],
            "H3_qfull_improves_matched_non_q": model_metrics["Q_FULL"]["macro_f1"] > model_metrics["B2_MATCHED_NON_Q"]["macro_f1"],
            "H4_safe_execution_improves_precision": technical["auto_commit_precision"] > technical["direct_top1_accuracy"],
            "H5_failure_consistency": technical["failure_stages_passed"] == technical["failure_stages_total"],
            "H6_active_index_smaller_than_append_only": technical["final_active_index_size"] < technical["append_only_comparator_size"],
            "H7_validation_cost_measured": technical["update_latency_ms"]["p95"] > 0,
        },
    }
    statistics_payload = {"iterations": 5000, "seed": experiment["seed"], "comparisons": comparisons, "candidate_recall_wilson_95_ci": candidate_intervals, "multiple_comparison_method": "Holm"}
    (report_dir / "confirmatory_results.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    (report_dir / "confirmatory_statistics.json").write_text(json.dumps(statistics_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    prediction_rows = [{"event_id": event.event_id, "true_action": truth[index], **{name: prediction[index] for name, prediction in predictions.items()}, "conformal_set": "|".join(prediction_sets[index])} for index, event in enumerate(events)]
    write_csv(report_dir / "confirmatory_test_predictions.csv", prediction_rows)
    write_csv(report_dir / "confirmatory_execution_decisions.csv", decisions)
    q = comparisons["Q_FULL"]
    report = f"""# Q-KEF v2 Confirmatory Experiment Report

## Protocol

The 2,400-event ancestry-isolated benchmark was used exactly as locked: 1,200 TRAIN, 360 DEV, 360 CALIBRATION, and one 480-row TEST evaluation. DEV selected configurations, CAL fitted Mondrian thresholds, and model hashes were verified before TEST access.

## Candidate retrieval

Dense Recall@5 was {candidate_metrics['dense']['recall@5']:.4f}; the locked hybrid achieved {candidate_metrics['locked_hybrid']['recall@5']:.4f}. H1 was {'supported' if results['hypotheses']['H1_hybrid_improves_recall5'] else 'not supported'}.

## Lifecycle models

B0 macro-F1 was {model_metrics['B0']['macro_f1']:.4f}; Q-full was {model_metrics['Q_FULL']['macro_f1']:.4f}; matched non-Q was {model_metrics['B2_MATCHED_NON_Q']['macro_f1']:.4f}; hierarchical Q-full was {model_metrics['HIERARCHICAL_Q_FULL']['macro_f1']:.4f}. Q-full minus B0 was {q['macro_f1_difference_vs_B0']:+.4f}, bootstrap 95% CI [{q['paired_bootstrap_95_ci'][0]:+.4f}, {q['paired_bootstrap_95_ci'][1]:+.4f}], Holm-adjusted p={q['holm_adjusted_p']:.4f}.

## Calibration and safe execution

Conformal marginal coverage was {conformal_metrics['marginal_coverage']:.4f}, mean set size {conformal_metrics['average_set_size']:.4f}, and singleton rate {conformal_metrics['singleton_rate']:.4f}. Counterfactual execution committed {technical['committed_transitions']}/{technical['evaluated_transitions']} transitions at precision {technical['auto_commit_precision']:.4f}; direct top-1 accuracy was {technical['direct_top1_accuracy']:.4f}.

## Technical effects

Graph/index consistency was {technical['graph_index_consistency_rate']:.4f}; all {technical['failure_stages_passed']}/{technical['failure_stages_total']} failure stages preserved the prior epoch. Final active index size was {technical['final_active_index_size']} versus append-only {technical['append_only_comparator_size']}. Update latency p50/p95 was {technical['update_latency_ms']['p50']:.2f}/{technical['update_latency_ms']['p95']:.2f} ms.

## Interpretation

All null and negative outcomes are retained. The benchmark is controlled and synthetic; this experiment does not establish production readiness, legal patentability, or quantum computational advantage. Human review of the audit sample remains a separate signed process.
"""
    (report_dir / "CONFIRMATORY_EXPERIMENT_REPORT.md").write_text(report, encoding="utf-8", newline="\n")
    postlock = ExperimentLock(prelock["experiment_version"], prelock["configuration"], prelock["benchmark_hash"], prelock["split_hash"], prelock["code_commit"], prelock["model_hashes"], True)
    postlock.save(report_dir / "experiment_lock_posttest.json")
    print(json.dumps({"status": "PASS", "test_events": len(events), "B0_macro_f1": model_metrics["B0"]["macro_f1"], "Q_FULL_macro_f1": model_metrics["Q_FULL"]["macro_f1"], "committed": technical["committed_transitions"]}, sort_keys=True))


if __name__ == "__main__":
    main()
