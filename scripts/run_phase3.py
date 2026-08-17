"""Run the complete Phase 3 experiment."""

from __future__ import annotations

import argparse
import json

from _phase1_common import PROJECT_ROOT, load_config
from qkef.evaluation.phase3 import run_phase3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--force-embeddings", action="store_true")
    parser.add_argument("--offline-fallback", action="store_true", help="Explicit deterministic test/offline encoder; never used silently.")
    args = parser.parse_args()
    config, _ = load_config(args.config)
    result = run_phase3(config, PROJECT_ROOT, force_embeddings=args.force_embeddings, offline_fallback=args.offline_fallback)
    print(json.dumps({"status": "PASS", "backend": result["embedding"]["backend"], "model": result["embedding"]["model"]}, sort_keys=True))


if __name__ == "__main__":
    main()
