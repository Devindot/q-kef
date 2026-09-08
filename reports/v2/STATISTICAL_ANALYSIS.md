# Q-KEF v2 Statistical Analysis

These are retrospective pilot results on the frozen 60-row v1 TEST partition, not confirmatory Q-KEF v2 evidence. Configuration selection used TRAIN/DEV and a dedicated 60-row TRAIN-derived calibration subset; TEST was not used for tuning.

## Q-full versus B0

- Macro-F1 difference: -0.0337
- Paired bootstrap 95% CI: [-0.0876, +0.0000]
- Exact McNemar p: 0.5000
- Holm-adjusted p across planned ablations: 1.0000

The interval crosses zero and the corrected comparison is not statistically significant at 0.05. No quantum-specific advantage is established. Full paired results and candidate-recall Wilson intervals are in `statistical_analysis.json`.
