"""Print a concise Phase 3 result summary."""

from __future__ import annotations

import json

from _phase1_common import PROJECT_ROOT


def main() -> None:
    manifest = json.loads((PROJECT_ROOT / "data/processed/qkef_phase3/phase3_manifest.json").read_text(encoding="utf-8"))
    print(f"Embedding: {manifest['embedding']['model']} ({manifest['embedding']['dimension']}d, {manifest['embedding']['backend']})")
    print(f"Candidate Recall@5: {manifest['candidate_metrics']['recall@5']:.4f}; MRR: {manifest['candidate_metrics']['mrr']:.4f}")
    for system in ("conventional", "qkef"):
        values = manifest["model_metrics"][system]["TEST"]
        print(f"{system}: TEST macro-F1={values['macro_f1']:.4f}, balanced accuracy={values['balanced_accuracy']:.4f}")
    for system, values in manifest["retrieval_metrics"].items():
        print(f"retrieval/{system}: Recall@5={values['recall@5']:.4f}, MRR={values['mrr']:.4f}, obsolete@5={values['obsolete_rate@5']:.4f}")


if __name__ == "__main__":
    main()
