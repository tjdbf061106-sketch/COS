"""Lock an A/B split for real new-material data only when grouping identifiers exist."""
import argparse
from pathlib import Path
import pandas as pd
from research_core import ROOT, linked_groups, split_group_sets, read_json, write_json, sha


def create_intake_split(frame, features, seed, test_fraction):
    if "contains_new_material" not in frame: raise ValueError("contains_new_material required")
    if not frame.contains_new_material.isin([True, False, 0, 1]).all(): raise ValueError("New-material indicator must be Boolean")
    new = frame.contains_new_material.astype(bool)
    if not new.any(): raise ValueError("No new-material formulations")
    known_features = read_json(ROOT / "inputs/feature_names.json")
    novel_features = [f for f in features if f not in known_features]
    if not novel_features: raise ValueError("A new material/grade requires its own new stock column")
    if not frame[novel_features].gt(0).any(axis=1).equals(new):
        raise ValueError("contains_new_material conflicts with actual new-stock quantities")
    for col in ["record_id", "source_id", "formulation_id", "preparation_batch_id", "measurement_event_id"]:
        if col in frame and frame[col].astype(str).str.strip().str.lower().isin(["unknown", "not_reported", "none", "nan", "미확인"]).any():
            raise ValueError(f"Unresolved identifier: {col}")
    groups = linked_groups(frame, features, require_experimental_ids=True)
    new_groups = set(groups[new])
    development, final = split_group_sets(new_groups, test_fraction, seed)
    result = frame[["record_id", "source_id", "formulation_id", "preparation_batch_id", "measurement_event_id"]].copy()
    result["connected_group"] = groups
    # Any linked record follows the entire connected group into the final set.
    result["partition"] = ["B_final_evaluation" if g in final else "A_development" for g in groups]
    assert result.groupby("connected_group").partition.nunique().eq(1).all()
    assert (new & result.partition.eq("A_development")).any()
    assert (new & result.partition.eq("B_final_evaluation")).any()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--feature-list", type=Path, required=True, help="JSON list including each new stock column")
    parser.add_argument("--output-dir", type=Path, required=True, help="A new, empty output folder")
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--test-fraction", type=float, default=.3)
    args = parser.parse_args()
    features = read_json(args.feature_list)
    if not isinstance(features, list) or not features or len(set(features)) != len(features): raise ValueError("Unique feature list required")
    frame = pd.read_csv(args.input, float_precision="round_trip")
    split = create_intake_split(frame, features, args.seed, args.test_fraction)
    if args.output_dir.exists() and any(args.output_dir.iterdir()): raise FileExistsError("Locked split must not be overwritten")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split.to_csv(args.output_dir / "locked_split.csv", index=False, encoding="utf-8-sig")
    write_json(args.output_dir / "split_lock.json", {"input_sha256": sha(args.input), "feature_list_sha256": sha(args.feature_list),
        "split_sha256": sha(args.output_dir / "locked_split.csv"), "seed": args.seed, "test_fraction": args.test_fraction,
        "groups": split.connected_group.nunique(), "requires_manual_identifier_audit": True,
        "external_independence_confirmed": False, "next_step": "Only A may be used for fitting/selection; freeze the model before opening B outcomes"})
    print("A/B split written and locked. Identifiers are declared metadata, not independently verified external evidence.")
