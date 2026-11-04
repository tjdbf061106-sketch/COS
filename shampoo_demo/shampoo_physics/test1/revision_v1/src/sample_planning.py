"""Illustrative class-specific precision planning; never an automatic acceptance rule."""
import math
import pandas as pd
from research_core import ROOT, wilson_interval, write_json


def minimum_n_for_scenario(expected_rate, desired_lower, confidence=.95, maximum=10000):
    if not 0 < desired_lower < expected_rate <= 1: raise ValueError("Need 0 < desired lower bound < assumed rate <= 1")
    for n in range(2, maximum + 1):
        successes = math.floor(expected_rate * n + 1e-10)
        lower, upper = wilson_interval(successes, n, confidence)
        if lower >= desired_lower:
            return {"independent_units_per_class": n, "assumed_successes": successes, "wilson_lower": lower, "wilson_upper": upper}
    return {"independent_units_per_class": None}


if __name__ == "__main__":
    rows = []
    for expected in [.8, .9, .95, 1.0]:
        for lower in [.7, .8, .9]:
            if lower < expected:
                rows.append({"assumed_class_recall": expected, "desired_lower_bound": lower, "confidence": .95,
                             **minimum_n_for_scenario(expected, lower)})
    pd.DataFrame(rows).to_csv(ROOT / "results/sample_size_scenarios.csv", index=False, encoding="utf-8-sig")
    write_json(ROOT / "results/sample_size_notes.json", {
        "illustrative_only": True, "team_acceptance_targets_selected": False,
        "interpretation": "Counts are per independent evaluation class, not total recipes and not training samples.",
        "assumptions": ["Independent Bernoulli units", "Assumed future success rate", "Two-sided 95% Wilson lower bound"],
        "limitations": ["Not a power or assurance calculation", "No guarantee the assumed success rate will occur",
                        "Technical repeats do not increase independent n", "Manufacturing-batch clustering requires a clustered design/analysis",
                        "Stable/unstable prevalence and invalid measurements can increase the number of recipes needed",
                        "Regression requires pilot residual/variance data and an application-specific tolerance; no universal minimum is supplied"]})
    print("Saved illustrative precision scenarios; registration targets remain unset.")
