"""Eye production v2: frozen evaluation protocol, fitted features and model bundles.

Inference needs this module, concentration_features.py and the recorded Python
packages. It does not read training data, labels, groups or legacy experiments.
Only load trusted locally produced joblib bundles; hashes detect corruption,
not an untrusted producer.
"""

from __future__ import annotations

from contextlib import contextmanager
import csv
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import tempfile

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import concentration_features as C

ROOT = Path(__file__).resolve().parent
VERSION = "eye-production-v2.1"
MODELS = ("lr", "rf", "et", "svm", "xgb", "lgbm")
ROLES = {"existing_evaluation_pool": 606, "augmentation_train_only": 67}
LABELS = {"A": "EPA I/II=1, III/IV=0", "B": "EPA I/II/III=1, IV=0"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def hash_json(value):
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
            default=plain,
        ).encode("utf-8")
    ).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish_new(temporary, destination):
    """Publish atomically without replacing any existing destination."""
    os.link(temporary, destination)


def write_json_new(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=".v2-", delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(
            json.dumps(
                payload, ensure_ascii=False, indent=2, allow_nan=False, default=plain
            ).encode("utf-8")
        )
        stream.flush()
        os.fsync(stream.fileno())
    try:
        publish_new(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_record(path, identity, payload):
    path = Path(path)
    if path.exists():
        require(
            hash_json(load_record(path, identity)) == hash_json(payload),
            f"Existing result differs; preserved: {path}",
        )
        return
    write_json_new(
        path,
        {
            "identity": identity,
            "payload": payload,
            "payload_sha256": hash_json(payload),
        },
    )


def load_record(path, identity=None):
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if identity is not None:
        require(record.get("identity") == identity, f"Identity mismatch: {path}")
    require(
        record.get("payload_sha256") == hash_json(record["payload"]),
        f"Checksum mismatch: {path}",
    )
    return record["payload"]


def read_raw_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream))
    require(len(header) == len(set(header)), "Duplicate CSV column names")
    return pd.read_csv(path, low_memory=False)


def base_columns(df):
    require(isinstance(df, pd.DataFrame), "Raw feature input must be a DataFrame")
    require(df.columns.is_unique, "Duplicate DataFrame column names")
    require(all(isinstance(c, str) for c in df.columns), "Column names must be strings")
    require(
        not any(c.startswith("conc_") for c in df.columns),
        "Use original raw metadata, not a precomputed conc_* export",
    )
    fp = [c for c in df.columns if c.startswith("fp_")]
    desc = [c for c in df.columns if c.startswith("desc_")]
    require(len(fp) == 2048 and len(desc) == 217, "Expected fp2048 + desc217")
    return fp + desc


def concentration_metadata(df):
    names = ["known_pct", "pct_sum_used", "undeclared_pct"]
    require(
        set(names) <= set(df.columns), f"Missing raw concentration metadata: {names}"
    )
    return df[names].apply(pd.to_numeric, errors="raise")


class FeatureTransform:
    """Fit the concentration median once; select persisted names at inference."""

    def __init__(self, arm):
        require(arm in ("baseline", "concentration"), "Unknown feature arm")
        self.arm = arm

    def fit(self, training):
        self.base_columns_ = base_columns(training)
        self.median_pct_ = (
            C.fit_concentration_median(concentration_metadata(training))
            if self.arm == "concentration"
            else None
        )
        self.feature_names_ = self.base_columns_ + (
            list(C.NAMES) if self.arm == "concentration" else []
        )
        self.transform(training)
        return self

    def transform(self, frame):
        require(hasattr(self, "feature_names_"), "FeatureTransform has not been fitted")
        incoming = base_columns(frame)
        require(
            set(incoming) == set(self.base_columns_),
            "Missing or unexpected vector feature names",
        )
        matrix = (
            frame[self.base_columns_]
            .apply(pd.to_numeric, errors="raise")
            .to_numpy(dtype=np.float64)
        )
        require(not np.isinf(matrix).any(), "Infinite vector features")
        if self.arm == "concentration":
            block, _, _ = C.concentration_block(
                concentration_metadata(frame), self.median_pct_
            )
            matrix = np.column_stack([matrix, block])
        require(
            matrix.shape[1] == len(self.feature_names_), "Feature dimension mismatch"
        )
        return matrix

    def schema(self):
        return {
            "arm": self.arm,
            "base_features": list(self.base_columns_),
            "features": list(self.feature_names_),
            "feature_count": len(self.feature_names_),
            "median_pct": self.median_pct_,
            "pct_fallback": "known_pct -> pct_sum_used -> fitted training median",
            "undeclared_missing": 0,
            "original_vectors_rescaled_by_concentration": False,
        }


def score_estimator(estimator, matrix, model):
    require(model in MODELS, "Unsupported model")
    require(list(estimator.classes_) == [0, 1], "Incorrect class orientation")
    transformed = estimator[:-1].transform(matrix)
    require(
        transformed.shape == matrix.shape and np.isfinite(transformed).all(),
        "Invalid preprocessing output",
    )
    with threadpool_limits(limits=1):
        scores = (
            estimator.decision_function(matrix)
            if model == "svm"
            else estimator.predict_proba(matrix)[:, 1]
        )
    scores = np.asarray(scores, dtype=float)
    require(
        scores.shape == (len(matrix),) and np.isfinite(scores).all(),
        "Invalid prediction scores",
    )
    return scores


@dataclass
class EyeModel:
    features: FeatureTransform
    estimator: object
    model: str
    label: str
    threshold: float
    provenance: dict

    def describe(self):
        require(self.model in MODELS and self.label in LABELS, "Invalid model or label")
        require(np.isfinite(self.threshold), "Nonfinite threshold")
        require(
            self.estimator.n_features_in_ == len(self.features.feature_names_),
            "Estimator/schema mismatch",
        )
        require(list(self.estimator.classes_) == [0, 1], "Invalid estimator classes")
        return {
            "version": VERSION,
            "model": self.model,
            "label": self.label,
            "label_definition": LABELS[self.label],
            "threshold": float(self.threshold),
            "comparison": ">=",
            "score_kind": "decision_function"
            if self.model == "svm"
            else "uncalibrated_class_1_score",
            "schema": self.features.schema(),
            "provenance": self.provenance,
        }

    def predict(self, frame):
        self.describe()
        scores = score_estimator(
            self.estimator, self.features.transform(frame), self.model
        )
        output = pd.DataFrame(
            {
                "input_row": np.arange(len(frame)),
                "score": scores,
                "prediction": (scores >= self.threshold).astype(int),
            }
        )
        if "formulation_id" in frame:
            output.insert(1, "formulation_id", frame.formulation_id.to_numpy())
        output["threshold"] = self.threshold
        output["label"] = self.label
        return output


def runtime_environment(model):
    names = ["numpy", "pandas", "scipy", "scikit-learn", "joblib", "threadpoolctl"]
    if model in ("xgb", "lgbm"):
        names.append({"xgb": "xgboost", "lgbm": "lightgbm"}[model])
    return {
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in names},
    }


def runtime_hashes():
    return {
        "production_pipeline_v2.py": file_hash(__file__),
        "concentration_features.py": file_hash(C.__file__),
    }


def save_model(bundle, directory, probe):
    """Manifest is the completion marker; partial files are never overwritten."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    description = bundle.describe()
    for name in ("model.joblib", "model_manifest.json"):
        if (directory / name).exists():
            raise FileExistsError(
                f"Model artifact exists; preserved: {directory / name}"
            )
    before = bundle.predict(probe)
    with tempfile.NamedTemporaryFile(
        dir=directory, prefix=".model-", delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        joblib.dump(bundle, temporary, compress=3)
        restored = joblib.load(temporary)  # Only our just-written local object.
        pd.testing.assert_frame_equal(before, restored.predict(probe), check_exact=True)
        require(restored.describe() == description, "Reloaded metadata differs")
        publish_new(temporary, directory / "model.joblib")
    finally:
        temporary.unlink(missing_ok=True)
    payload = {
        "description": description,
        "environment": runtime_environment(bundle.model),
        "runtime_code_sha256": runtime_hashes(),
        "model_sha256": file_hash(directory / "model.joblib"),
        "serialization_prediction_parity": True,
        "probe_rows": len(probe),
        "deployed": False,
        "trust": "Only load trusted bundles; checksums are not publisher authentication",
    }
    save_record(directory / "model_manifest.json", hash_json(description), payload)
    return load_model(directory)


def load_model(directory):
    directory = Path(directory)
    manifest = load_record(directory / "model_manifest.json")
    description = manifest["description"]
    load_record(directory / "model_manifest.json", hash_json(description))
    require(description["version"] == VERSION, "Unsupported model bundle version")
    require(
        manifest["environment"] == runtime_environment(description["model"]),
        "Model runtime package/Python version mismatch",
    )
    require(
        manifest["runtime_code_sha256"] == runtime_hashes(),
        "Model runtime source hash mismatch",
    )
    require(
        manifest["model_sha256"] == file_hash(directory / "model.joblib"),
        "Model checksum mismatch",
    )
    bundle = joblib.load(directory / "model.joblib")
    require(isinstance(bundle, EyeModel), "Unexpected bundle type")
    require(bundle.describe() == description, "Model/manifest metadata mismatch")
    return bundle


def legacy():
    # Lazy: loading/predicting a bundle never imports training/evaluation code.
    import run_concentration_production as production

    return production, production.E, production.u


def fit_model(training, y, *, arm, model, params, label, threshold, provenance):
    _, _, u = legacy()
    require(label in LABELS and model in MODELS, "Invalid model/label")
    features = FeatureTransform(arm).fit(training)
    estimator = u.fit_weighted(
        u.build_estimator(model, params), features.transform(training), np.asarray(y)
    )
    return EyeModel(features, estimator, model, label, float(threshold), provenance)


def load_training_data(path):
    _, E, u = legacy()
    frame = read_raw_csv(path)
    columns = base_columns(frame)
    required = {
        "formulation_id",
        "evaluation_role",
        "comp_group_id",
        "split_group_id",
        "final_label",
    }
    require(
        required <= set(frame.columns),
        "Missing production role/ID/group/label metadata",
    )
    require(
        frame.evaluation_role.value_counts(dropna=False).to_dict() == ROLES
        and len(frame) == 673,
        "Expected exactly 606 evaluation + 67 augmentation rows",
    )
    require(
        frame.formulation_id.notna().all() and frame.formulation_id.is_unique,
        "Missing or duplicate formulation IDs",
    )
    require(
        frame.formulation_id.astype(str).str.strip().ne("").all(),
        "Empty formulation IDs",
    )
    u.epa_labels(frame.final_label)
    matrix = FeatureTransform("baseline").fit(frame).transform(frame)
    require(
        columns == u.feature_columns(frame.columns), "Legacy feature schema mismatch"
    )
    groups = E.merged_groups(frame, matrix)
    return frame, groups


def calibration_splits(frame, groups, seed):
    """Three grouped folds over existing rows; augmentation never calibrates."""
    from sklearn.model_selection import StratifiedGroupKFold

    _, _, u = legacy()
    evaluation = np.flatnonzero(frame.evaluation_role.eq("existing_evaluation_pool"))
    augmentation = np.flatnonzero(frame.evaluation_role.eq("augmentation_train_only"))
    groups = np.asarray(groups)
    require(frame.evaluation_role.isin(ROLES).all(), "Unknown evaluation role")
    epa = frame.final_label.to_numpy()
    labels = u.epa_labels(epa)
    parts = []
    for a, b in StratifiedGroupKFold(3, shuffle=True, random_state=seed).split(
        evaluation, epa[evaluation], groups[evaluation]
    ):
        valid = evaluation[b]
        allowed = augmentation[~np.isin(groups[augmentation], groups[valid])]
        train = np.concatenate([evaluation[a], allowed])
        u._check_partition(train, valid, groups, labels)
        parts.append({"train": train.tolist(), "validation": valid.tolist()})
    require(
        sorted(i for p in parts for i in p["validation"]) == evaluation.tolist(),
        "Calibration OOF coverage mismatch",
    )
    return parts, evaluation.tolist()


def scientific_hashes():
    P, E, u = legacy()
    return {
        **runtime_hashes(),
        "run_concentration_production.py": file_hash(P.__file__),
        "experiment_utils.py": file_hash(E.__file__),
        "epa_experiment_utils.py": file_hash(u.__file__),
    }


def train_final(*, label, model, arm, seed=42, output=None):
    """Tune on training-only grouped OOF, select EPA threshold, fit all 673.

    Calibration OOF is selection/calibration data, NOT an unbiased performance
    estimate. Model family and feature arm are explicit user choices.
    """
    _, E, u = legacy()
    require(
        label in LABELS and model in MODELS and arm in ("baseline", "concentration"),
        "Invalid training selection",
    )
    source_hash = file_hash(E.DATA)
    skin_hash = file_hash(E.DATA.with_name("skin_vectorized.csv"))
    frame, groups = load_training_data(E.DATA)
    parts, evaluation = calibration_splits(frame, groups, seed)
    grid = u.candidate_grid(model)
    require(len(grid) == 20, "Expected the complete 20-candidate EPA grid")
    config = {
        "version": VERSION,
        "purpose": "final_training_not_performance_evaluation",
        "input": str(E.DATA),
        "input_sha256": source_hash,
        "skin_sha256": skin_hash,
        "label": label,
        "model": model,
        "arm": arm,
        "seed": seed,
        "grid": grid,
        "groups": groups.tolist(),
        "calibration_splits": parts,
        "calibration_rows": evaluation,
        "code_sha256": scientific_hashes(),
        "environment": runtime_environment(model),
        "selection": "mean inner ROC-AUC, stable grid ordering; EPA gap/MCC threshold on existing-row OOF",
        "final_fit_rows": len(frame),
        "calibration_evaluation_rows": len(evaluation),
        "augmentation_evaluated": 0,
    }
    identity = hash_json(config)
    output = (
        Path(output)
        if output
        else ROOT / "results/production_pipeline_v2/models" / identity[:16]
    )
    save_record(output / "training_config.json", identity, config)
    if (output / "model_manifest.json").exists():
        bundle = load_model(output)
        require(
            bundle.provenance["training_identity"] == identity,
            "Existing model belongs to another training run",
        )
        return output
    if (output / "model.joblib").exists():
        raise FileExistsError(
            "Incomplete model artifact preserved; use another output directory"
        )
    y = u.epa_labels(frame.final_label)[label]
    prepared = []
    for part in parts:
        transform = FeatureTransform(arm).fit(frame.iloc[part["train"]])
        prepared.append(
            (
                part,
                transform.transform(frame.iloc[part["train"]]),
                transform.transform(frame.iloc[part["validation"]]),
                transform.median_pct_,
            )
        )
    candidates = []
    for number, candidate in enumerate(grid, 1):
        path = output / "candidates" / (candidate["candidate_id"] + ".json")
        if path.exists():
            result = load_record(path, identity)
            require(
                result["candidate"] == candidate
                and result["calibration_rows"] == evaluation,
                "Candidate checkpoint does not match final-training configuration",
            )
        else:
            scores = np.full(len(frame), np.nan)
            aucs = []
            for part, train, valid, _ in prepared:
                estimator = u.fit_weighted(
                    u.build_estimator(model, candidate["params"]),
                    train,
                    y[part["train"]],
                )
                values = score_estimator(estimator, valid, model)
                scores[part["validation"]] = values
                aucs.append(float(u.roc_auc_score(y[part["validation"]], values)))
            require(
                np.isfinite(scores[evaluation]).all(),
                "Incomplete calibration predictions",
            )
            result = {
                "candidate": candidate,
                "mean_auc": float(np.mean(aucs)),
                "fold_aucs": aucs,
                "calibration_rows": evaluation,
                "calibration_scores": scores[evaluation].tolist(),
            }
            save_record(path, identity, result)
        candidates.append(result)
        print(
            f"Final-training candidate {number}/20: {candidate['candidate_id']}",
            flush=True,
        )
    best = max(candidates, key=lambda r: r["mean_auc"])
    cut = u.threshold(best["calibration_scores"], y[evaluation])
    save_record(
        output / "calibration.json",
        identity,
        {
            **best,
            "threshold": float(cut),
            "inner_training_medians": [p[3] for p in prepared],
            "usage": "Selection/calibration only; not held-out performance",
        },
    )
    bundle = fit_model(
        frame,
        y,
        arm=arm,
        model=model,
        params=best["candidate"]["params"],
        label=label,
        threshold=cut,
        provenance={
            "training_identity": identity,
            "input_sha256": source_hash,
            "selected_candidate": best["candidate"],
            "training_rows": len(frame),
            "calibration_rows": len(evaluation),
            "augmentation_training_rows": int(
                frame.evaluation_role.eq("augmentation_train_only").sum()
            ),
            "scope": "full_training_candidate_not_deployed",
        },
    )
    require(
        file_hash(E.DATA) == source_hash
        and file_hash(E.DATA.with_name("skin_vectorized.csv")) == skin_hash,
        "Input CSV changed during final training",
    )
    require(
        scientific_hashes() == config["code_sha256"],
        "Source code changed during training",
    )
    save_model(bundle, output, frame)
    return output


@contextmanager
def training_lock():
    """Keep v2 calls exclusive and refuse overlap with the original runner."""
    import psutil

    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        if (
            process.pid == os.getpid()
            or "python" not in (process.info["name"] or "").lower()
        ):
            continue
        args = process.info["cmdline"] or []
        if any(
            Path(a).name.lower()
            in ("run_concentration_production.py", "run_concentration_production")
            for a in args
        ):
            raise RuntimeError(
                f"Original production runner is active: PID {process.pid}, {args}"
            )
    path = ROOT / "results/.eye_production_v2.lock"
    path.parent.mkdir(exist_ok=True)
    with path.open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"1")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise RuntimeError(
                    "Another v2 training/evaluation call is active"
                ) from error
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def evaluate(*, workers=6, models=MODELS, prepare_only=False):
    """Canonical evaluation entry: delegate to the immutable, audited harness.

    Its identity and completed checkpoints remain reusable; the historical EPA
    all-row evaluator is never invoked. It uses the same C concentration
    formulas and fitted-training-median policy as FeatureTransform.
    """
    P, E, _ = legacy()
    load_training_data(E.DATA)  # Reject derived exports, bad roles and invalid IDs.
    previous = sys.argv
    try:
        sys.argv = [str(P.__file__), "--workers", str(workers), "--models", *models]
        if prepare_only:
            sys.argv.append("--prepare-only")
        P.main()
    finally:
        sys.argv = previous
