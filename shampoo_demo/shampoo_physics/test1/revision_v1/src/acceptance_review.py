"""Validate a declared use-case policy and emit a review decision, never register a model."""
import argparse
from pathlib import Path
from research_core import ROOT, read_json, write_json, assess_registration


def validate_policy(criteria):
    if criteria.get("positive_class") != "stable": raise ValueError("Policy positive class must match stable=1")
    confidence = criteria.get("confidence_level", .95)
    if not isinstance(confidence, (int, float)) or not 0 < confidence < 1: raise ValueError("Invalid confidence level")
    for key in ["sensitivity_lower_bound_min", "specificity_lower_bound_min", "max_sensitivity_specificity_gap", "coverage_min"]:
        value = criteria.get(key)
        if value is not None and (not isinstance(value, (int, float)) or not 0 <= value <= 1): raise ValueError(key)
    for key in ["min_independent_stable_units", "min_independent_unstable_units", "min_viscosity_test_units", "min_turbidity_test_units"]:
        value = criteria.get(key)
        if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 1): raise ValueError(key)
    for key in ["viscosity_log_mae_max", "turbidity_log_mae_max"]:
        value = criteria.get(key)
        if value is not None and (not isinstance(value, (int, float)) or not 0 <= value < float("inf")): raise ValueError(key)


def review(evaluation, criteria):
    validate_policy(criteria)
    evidence = evaluation.get("independence_audit", {})
    independence = (evidence.get("independent_evaluation_units_verified") is True
                    and evidence.get("prospective_independence_confirmed") is True
                    and bool(evidence.get("audit_reference")) and bool(evidence.get("reviewer")))
    result = assess_registration(evaluation.get("classification", {}), evaluation.get("regression", {}), criteria, independence)
    result["registration_performed"] = False
    result["scope"] = "Decision support only. PASS_LIMITED_SCOPE still requires a named material/grade/condition scope and documented review."
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--criteria", type=Path, default=ROOT / "config/acceptance_criteria.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError("Do not overwrite a decision record")
    decision = review(read_json(args.evaluation), read_json(args.criteria))
    write_json(args.output, decision)
    print(decision)
