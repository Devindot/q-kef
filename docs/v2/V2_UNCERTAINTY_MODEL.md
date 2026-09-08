# Calibrated Uncertainty and Quarantine

`MondrianConformalClassifier` uses class-conditional scores `1-p_y(x)` on a dedicated CAL partition and finite-sample quantiles at configurable alpha (pilot: 0.10). It reports marginal/per-class coverage and average set size.

Auto-commit requires a singleton conformal set, a feasible plan, risk below the frozen threshold, and a unique risk minimum. Ambiguous sets, near-tied risks, missing cardinality preconditions, authority conflicts, temporal ambiguity, and invariant failures produce QUARANTINE/HUMAN REVIEW. Conformal prediction is a known safety mechanism and is not claimed as a novelty.

The retrospective pilot observed 0.95 marginal coverage, average set size 1.417, and a 43.3% non-singleton quarantine rate. These estimates use only 60 calibration and 60 test rows and are not production guarantees.
