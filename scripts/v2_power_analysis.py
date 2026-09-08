"""Planning-only power analysis; no observed v2 claim is produced."""

from __future__ import annotations

import json

import math

from scipy.stats import norm

from _phase1_common import PROJECT_ROOT


def power(total: int, discordance_rate: float = 5 / 60, challenger_share: float = 4 / 5, alpha: float = 0.05) -> float:
    """Normal planning approximation for a two-sided paired McNemar contrast."""
    challenger = discordance_rate * challenger_share
    baseline = discordance_rate * (1 - challenger_share)
    noncentrality = (challenger - baseline) * math.sqrt(total) / math.sqrt(challenger + baseline)
    critical = norm.ppf(1 - alpha / 2)
    return float(norm.cdf(-critical - noncentrality) + 1 - norm.cdf(critical - noncentrality))


def minimum_sample(target: float) -> int:
    return next(total for total in range(60, 2001) if power(total) >= target)


def main() -> None:
    values = {"assumptions": {"v1_discordance_rate": 5 / 60, "challenger_share_among_discordances": 4 / 5, "alpha": 0.05}, "minimum_test_n_for_80_percent_power": minimum_sample(0.8), "minimum_test_n_for_90_percent_power": minimum_sample(0.9), "planned_v2_test_n": 480, "power_at_planned_n": power(480)}
    path = PROJECT_ROOT / "reports/v2/POWER_ANALYSIS.md"
    path.write_text(f"""# Q-KEF v2 Power Analysis

This is planning analysis, not an experimental result. Under the v1 observed discordance rate ({values['assumptions']['v1_discordance_rate']:.4f}) and challenger-correct share ({values['assumptions']['challenger_share_among_discordances']:.2f}), exact-binomial McNemar planning gives:

- Approximate minimum TEST n for 80% power: {values['minimum_test_n_for_80_percent_power']}
- Approximate minimum TEST n for 90% power: {values['minimum_test_n_for_90_percent_power']}
- Planned confirmatory TEST n: 480
- Estimated power at n=480 under unchanged assumptions: {values['power_at_planned_n']:.3f}

The assumptions come from a small prior test and are uncertain. The 2,400-event benchmark remains a future confirmatory target; the current v2 run is explicitly retrospective.
""", encoding="utf-8", newline="\n")
    (PROJECT_ROOT / "reports/v2/power_analysis.json").write_text(json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", **{key: value for key, value in values.items() if key != "assumptions"}}, sort_keys=True))


if __name__ == "__main__":
    main()
