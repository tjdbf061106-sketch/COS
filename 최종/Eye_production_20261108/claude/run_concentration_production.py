"""Paired full-grid EPA validation: baseline versus four concentration features.

Only the existing 606 rows are evaluated. Augmentation is train-only and is
blocked from training when connected to either held-out partition. Original
production CSVs and experimental caches are never overwritten.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib.metadata
from pathlib import Path
import sys

import numpy as np
import pandas as pd

import experiment_utils as E
from concentration_features import NAMES, concentration_block, fit_concentration_median

u = E.u
ROOT = Path(__file__).resolve().parent
_DATA = None


def production_splits(df, groups, seed):
    roles = df.evaluation_role
    if not roles.isin(["existing_evaluation_pool", "augmentation_train_only"]).all():
        raise ValueError("Unknown evaluation role")
    evaluation = np.flatnonzero(roles.eq("existing_evaluation_pool"))
    augmentation = np.flatnonzero(roles.eq("augmentation_train_only"))
    groups = np.asarray(groups)
    folds = u.nested_splits(
        df.final_label.to_numpy()[evaluation], groups[evaluation], seed
    )
    output = []
    for fold in folds:
        calibration = evaluation[fold["train"]]
        test = evaluation[fold["test"]]
        blocked_outer = groups[test]
        allowed = augmentation[~np.isin(groups[augmentation], blocked_outer)]
        inner = []
        for part in fold["inner"]:
            train, valid = evaluation[part["train"]], evaluation[part["validation"]]
            blocked = np.union1d(blocked_outer, groups[valid])
            added = augmentation[~np.isin(groups[augmentation], blocked)]
            train = np.concatenate([train, added])
            if set(groups[train]) & set(blocked):
                raise ValueError("Inner group leakage")
            inner.append({"train": train.tolist(), "validation": valid.tolist()})
        train = np.concatenate([calibration, allowed])
        if set(groups[train]) & set(blocked_outer):
            raise ValueError("Outer group leakage")
        if sorted(i for p in inner for i in p["validation"]) != sorted(
            calibration.tolist()
        ):
            raise ValueError("Invalid calibration coverage")
        output.append(
            {
                "train": train.tolist(),
                "test": test.tolist(),
                "calibration": calibration.tolist(),
                "inner": inner,
            }
        )
    if sorted(i for f in output for i in f["test"]) != evaluation.tolist():
        raise ValueError("Invalid evaluation coverage")
    return output


def feature_pair(df, matrix, train, heldout, arm):
    train, heldout = np.asarray(train, dtype=int), np.asarray(heldout, dtype=int)
    if arm == "baseline":
        return matrix[train], matrix[heldout], None
    if arm != "concentration":
        raise ValueError(arm)
    median = fit_concentration_median(df.iloc[train])
    a, _, _ = concentration_block(df.iloc[train], median)
    b, _, _ = concentration_block(df.iloc[heldout], median)
    return (
        np.column_stack([matrix[train], a]),
        np.column_stack([matrix[heldout], b]),
        median,
    )


def worker_data(expected_hash):
    global _DATA
    if _DATA is None:
        if u.file_hash(E.DATA) != expected_hash:
            raise ValueError("Production source changed during validation")
        _DATA = E.load_eye()
    return _DATA


def run_fold(job):
    folder = Path(job["folder"])
    identity = job["identity"]
    done = folder / "outer.json"
    if done.exists():
        return u.load_record(done, identity)
    df, matrix, _ = worker_data(job["source_hash"])
    y = E.labels_of(df)[job["label"]]
    fold, model, arm = job["split"], job["model"], job["arm"]
    inner_data = []
    for part in fold["inner"]:
        a, b, median = feature_pair(df, matrix, part["train"], part["validation"], arm)
        inner_data.append((part, a, b, median))
    candidates = []
    for candidate in u.candidate_grid(model):
        path = folder / "candidates" / (candidate["candidate_id"] + ".json")
        if path.exists():
            result = u.load_record(path, identity)
            if result["candidate"] != candidate:
                raise ValueError("Candidate checkpoint mismatch")
        else:
            scores = np.full(len(df), np.nan)
            aucs = []
            for part, a, b, _ in inner_data:
                est = u.fit_weighted(
                    u.build_estimator(model, candidate["params"]), a, y[part["train"]]
                )
                s = u.predict_scores(est, b, model)
                scores[part["validation"]] = s
                aucs.append(float(u.roc_auc_score(y[part["validation"]], s)))
            calibration = fold["calibration"]
            if not np.isfinite(scores[calibration]).all():
                raise ValueError("Incomplete inner OOF")
            result = {
                "candidate": candidate,
                "mean_auc": float(np.mean(aucs)),
                "fold_aucs": aucs,
                "calibration_rows": calibration,
                "calibration_scores": scores[calibration].tolist(),
            }
            u.save_record(path, identity, result)
        candidates.append(result)
    # Stable ordering resolves tied mean AUC exactly as in the EPA harness.
    best = max(candidates, key=lambda r: r["mean_auc"])
    cut = u.threshold(best["calibration_scores"], y[fold["calibration"]])
    a, b, median = feature_pair(df, matrix, fold["train"], fold["test"], arm)
    estimator = u.fit_weighted(
        u.build_estimator(model, best["candidate"]["params"]), a, y[fold["train"]]
    )
    scores = u.predict_scores(estimator, b, model)
    result = {k: job[k] for k in ["arm", "label", "model", "seed", "fold"]}
    result.update(
        test=fold["test"],
        y_true=y[fold["test"]].tolist(),
        scores=scores.tolist(),
        predictions=(scores >= cut).astype(int).tolist(),
        threshold=float(cut),
        selected_candidate=best["candidate"],
        candidates_evaluated=len(candidates),
        outer_concentration_median=median,
        inner_concentration_medians=[item[3] for item in inner_data],
        train_rows=fold["train"],
        calibration_rows=fold["calibration"],
    )
    u.save_record(done, identity, result)
    return result


def summarize(results, evaluation_rows):
    rows = []
    frame = pd.DataFrame(results)
    for keys, parts in frame.groupby(["arm", "label", "model", "seed"], sort=True):
        parts = parts.sort_values("fold")
        indices = np.concatenate(parts.test.to_list())
        if sorted(indices.tolist()) != list(evaluation_rows):
            raise ValueError(
                "Outer predictions must cover only the evaluation pool exactly once"
            )
        y = np.concatenate(parts.y_true.to_list())
        scores = np.concatenate(parts.scores.to_list())
        predictions = np.concatenate(parts.predictions.to_list())
        rows.append(
            dict(
                zip(["arm", "label", "model", "seed"], keys),
                **u.compute_metrics(y, scores, predictions),
            )
        )
    metrics = pd.DataFrame(rows)
    paired = metrics.pivot(
        index=["label", "model", "seed"], columns="arm", values="MCC"
    )
    paired["delta_MCC"] = paired.concentration - paired.baseline
    return metrics, paired.reset_index()


def export_features(df, matrix, columns, output, identity, source_hash):
    """Write a full-training export, never used for cross-validation."""
    median = fit_concentration_median(df)
    block, names, _ = concentration_block(df, median)
    exported = df.copy()
    exported[names] = block
    csv = output / "eye_vectorized_concentration.csv"
    manifest = output / "export_manifest.json"
    if csv.exists():
        if not manifest.exists():
            raise FileExistsError(f"Unverified existing export preserved: {csv}")
        prior = u.load_record(manifest, identity)
        if u.file_hash(csv) != prior["output_sha256"]:
            raise ValueError("Existing export checksum mismatch; file preserved")
    else:
        output.mkdir(parents=True, exist_ok=True)
        exported.to_csv(csv, index=False, encoding="utf-8-sig", mode="x")
    reread = pd.read_csv(csv, low_memory=False)
    pd.testing.assert_frame_equal(
        reread[df.columns], df, check_dtype=False, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        reread[columns].to_numpy(float), matrix, rtol=1e-12, atol=1e-12, equal_nan=True
    )
    np.testing.assert_allclose(
        reread[names].to_numpy(float), block, rtol=1e-12, atol=1e-12
    )
    u.save_record(
        manifest,
        identity,
        {
            "output_sha256": u.file_hash(csv),
            "source_sha256": source_hash,
            "features": columns + names,
            "median_pct": median,
            "rows": len(df),
            "feature_count": len(columns) + 4,
            "usage": "Full-training export only. CV must use original metadata and fit concentration median separately within every training fold. Original production CSV unchanged.",
        },
    )
    return csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=E.WORKERS)
    parser.add_argument("--models", nargs="+", choices=E.MODELS, default=E.MODELS)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or len(args.models) != len(set(args.models)):
        parser.error("Use positive workers and distinct models")
    source_hash = u.file_hash(E.DATA)
    skin = E.DATA.with_name("skin_vectorized.csv")
    skin_hash = u.file_hash(skin)
    df, matrix, columns = E.load_eye()
    if len(df) != 673 or df.evaluation_role.value_counts().to_dict() != {
        "existing_evaluation_pool": 606,
        "augmentation_train_only": 67,
    }:
        raise ValueError("Unexpected production population")
    if not df.formulation_id.is_unique or set(columns) & set(NAMES):
        raise ValueError("Invalid row IDs or feature schema")
    groups = E.merged_groups(df, matrix)
    splits = {str(seed): production_splits(df, groups, seed) for seed in E.SEEDS}
    config = {
        "input": str(E.DATA),
        "input_sha256": source_hash,
        "skin_sha256": skin_hash,
        "code_hashes": {
            name: u.file_hash(ROOT / name)
            for name in [
                "run_concentration_production.py",
                "concentration_features.py",
                "experiment_utils.py",
            ]
        },
        "epa_code_sha256": u.file_hash(u.__file__),
        "environment": {
            "python": sys.version,
            "packages": {
                p: importlib.metadata.version(p)
                for p in ["numpy", "pandas", "scikit-learn", "lightgbm", "xgboost"]
            },
        },
        "features": {"baseline": columns, "concentration": columns + NAMES},
        "seeds": E.SEEDS,
        "grids": {m: u.candidate_grid(m) for m in args.models},
        "groups": groups.tolist(),
        "splits": splits,
        "protocol": "EPA 5x3 nested grouped CV; full 20-candidate grids; estimator seed 42; balanced weights; inner mean AUC; sensitivity/specificity-gap threshold with MCC tie-break; median fitted on training rows only; 606 evaluation rows; augmentation blocked by held-out groups",
    }
    identity = u.hash_json(config)
    output = E.RESULTS / "concentration_production" / identity[:16]
    u.save_record(output / "config.json", identity, config)
    print(f"Artifacts: {output}", flush=True)
    if args.prepare_only:
        print("Prepared and validated splits; no training performed", flush=True)
        return
    jobs = []
    for arm in ["baseline", "concentration"]:
        for label in ["A", "B"]:
            for model in args.models:
                for seed in E.SEEDS:
                    for index, fold in enumerate(splits[str(seed)]):
                        jobs.append(
                            {
                                "arm": arm,
                                "label": label,
                                "model": model,
                                "seed": seed,
                                "fold": index,
                                "split": fold,
                                "identity": identity,
                                "source_hash": source_hash,
                                "folder": str(
                                    output
                                    / arm
                                    / label
                                    / model
                                    / str(seed)
                                    / f"fold-{index}"
                                ),
                            }
                        )
    results = []
    print(f"Full-grid jobs: {len(jobs)}; 20 candidates each", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_fold, job): job for job in jobs}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:
                for pending in futures:
                    pending.cancel()
                raise RuntimeError(
                    f"Failed production fold: {futures[future]['folder']}"
                ) from exc
            print(f"Completed {len(results)}/{len(jobs)}", flush=True)
    evaluation = np.flatnonzero(
        df.evaluation_role.eq("existing_evaluation_pool")
    ).tolist()
    metrics, paired = summarize(results, evaluation)
    if u.file_hash(E.DATA) != source_hash or u.file_hash(skin) != skin_hash:
        raise ValueError("Source CSV changed during run")
    u.save_record(output / "metrics.json", identity, metrics.to_dict("records"))
    u.save_record(
        output / "paired_comparison.json", identity, paired.to_dict("records")
    )
    export_features(df, matrix, columns, output, identity, source_hash)
    if u.file_hash(E.DATA) != source_hash or u.file_hash(skin) != skin_hash:
        raise ValueError("Source CSV changed during export")
    print(
        paired.groupby(["label", "model"]).delta_MCC.agg(["mean", "std"]).to_string(),
        flush=True,
    )


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    main()
