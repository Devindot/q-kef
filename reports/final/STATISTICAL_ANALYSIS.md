# Statistical Analysis

Paired bootstrap: 5000 iterations, seed 42, 95% percentile intervals.

| Quantity | 95% CI |
|---|---|
| b_accuracy | [0.7500, 0.9333] |
| c_accuracy | [0.8167, 0.9667] |
| accuracy_difference | [-0.0167, 0.1333] |
| b_macro_f1 | [0.7491, 0.9241] |
| c_macro_f1 | [0.8057, 0.9606] |
| macro_f1_difference | [-0.0376, 0.1414] |

Exact McNemar/binomial discordance: B-only correct=1, C-only correct=4, p=0.375000. The paired accuracy difference was not statistically significant at the 0.05 level under McNemar's test.

The 60-item TEST set is modest; intervals quantify resampling uncertainty rather than external generalizability.
