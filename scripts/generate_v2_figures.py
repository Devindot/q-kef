"""Generate monochrome patent-drafting-friendly Q-KEF v2 technical figures."""

from __future__ import annotations

import os
import tempfile
import textwrap
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "qkef-v2-mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from _phase1_common import PROJECT_ROOT


FIGURES = {
    "figure_01_overall_architecture.png": ("Q-KEF v2 Overall Architecture", ["Incoming knowledge", "Lifecycle hypotheses", "Shadow plans", "Witness retrieval + risk", "Invariant gate", "Epoch publication"]),
    "figure_02_candidate_retrieval.png": ("Candidate Predecessor Retrieval", ["Incoming text/vector", "Dense ranking", "Lexical ranking", "Deterministic RRF", "Filtered candidates"]),
    "figure_03_cardinality_tree.png": ("Cardinality-Aware Lifecycle Decision", ["Meaningful predecessor?", "0→1 NEW / relation", "1→1 / m→1 / 1→r", "REPLACE · ARCHIVE · COEXIST / MERGE / SPLIT"]),
    "figure_04_multiple_hypotheses.png": ("Calibrated Lifecycle Hypotheses", ["Probability distribution", "Conformal set Γα(x)", "Plan REPLACE", "Plan COEXIST", "Plan MERGE"]),
    "figure_05_shadow_state.png": ("Non-Live Shadow State Generation", ["Committed Sₜ", "TransitionPlan delta", "ShadowStateView", "Candidate Sₜ⁽ᵃ⁾", "Live hash unchanged"]),
    "figure_06_witness_risk.png": ("Witness-Query Retrieval-Risk Evaluation", ["Content-derived probes", "Candidate retrieval", "OE · CM · IC", "CV · LV · TV · AV", "Risk vector"]),
    "figure_07_safe_selection.png": ("Safe Transition Selection", ["Plausible actions", "Reject invariant failures", "Reject authority/temporal conflict", "Compare frozen risk", "Unique safe action / quarantine"]),
    "figure_08_epoch_commit.png": ("Epoch-Synchronized Graph/Index Commit", ["Prepare graph delta", "Prepare index delta", "Certificate hash", "Validate same transition ID", "Publish Eₜ₊₁ atomically"]),
    "figure_09_failed_commit.png": ("Failed Commit and Restoration", ["Readable epoch Eₜ", "Injected prepare failure", "Discard candidate state", "Restore exact pre-hash", "Queries remain on Eₜ"]),
    "figure_10_certificate.png": ("Deterministic Transition Certificate", ["Input + candidates", "Probabilities + Γ", "Plans + risk + invariants", "Pre/post hashes + rollback ref", "Canonical JSON → SHA-256"]),
    "figure_11_bitemporal.png": ("Bitemporal Active/Historical Retrieval", ["Valid time", "System time", "ACTIVE current view", "HISTORICAL_ONLY as-of view", "QUARANTINED/BLOCKED excluded"]),
    "figure_12_quantum_optional.png": ("Optional Quantum-Inspired Embodiment", ["MiniLM embedding", "TRAIN-only PCA state", "Fidelity · entropy · coherence", "Matched non-Q ablation", "Safety architecture encoder-independent"]),
}


def draw(path: Path, title: str, labels: list[str]) -> None:
    figure, axis = plt.subplots(figsize=(12, 3.4))
    axis.set_xlim(0, len(labels)); axis.set_ylim(0, 1); axis.axis("off")
    for index, label in enumerate(labels):
        x = index + 0.08
        box = FancyBboxPatch((x, 0.35), 0.84, 0.30, boxstyle="round,pad=0.02", facecolor="white", edgecolor="black", linewidth=1.4)
        axis.add_patch(box); axis.text(x + 0.42, 0.50, textwrap.fill(label, width=22), ha="center", va="center", fontsize=8.5, linespacing=1.25)
        if index < len(labels) - 1:
            axis.annotate("", xy=(index + 1.08, 0.50), xytext=(index + 0.92, 0.50), arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.4})
    axis.set_title(title, fontsize=13, fontweight="bold", pad=18)
    axis.text(0, 0.08, "Technical research figure — not a legal drawing-format representation", fontsize=8, color="#444444")
    figure.tight_layout(); figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white"); plt.close(figure)


def main() -> None:
    destination = PROJECT_ROOT / "reports/v2/patent_figures"
    destination.mkdir(parents=True, exist_ok=True)
    for filename, (title, labels) in FIGURES.items():
        draw(destination / filename, title, labels)
    print(f"PASS: generated {len(FIGURES)} Q-KEF v2 technical figures")


if __name__ == "__main__":
    main()
