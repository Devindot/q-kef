# Reproducibility Guide

The final reference environment is Python 3.12.13.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/final_release_check.py
pytest -q
streamlit run app.py
```

The app does not download FiQA, rebuild the benchmark, regenerate embeddings, retrain, or retune. It needs trusted Phase 3 artifacts and the local MiniLM cache for interactive inference. Stored results remain available if interactive MiniLM is unavailable.

For a clean scientific rebuild, verify `configs/default.yaml`, then run `download_fiqa.py`, `build_evolution_benchmark.py`, `validate_evolution_benchmark.py`, `build_chunks.py`, `validate_chunks.py`, `run_phase3.py`, and `validate_phase3.py` in order. Never select parameters from TEST.

Phase 4 analysis is non-training:

```powershell
python scripts/build_demo_data.py
python scripts/run_final_analysis.py
python scripts/final_release_check.py
```

It uses stored predictions, seed 42, 5,000 paired bootstrap samples, and SHA-256 release hashes.
