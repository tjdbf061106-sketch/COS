"""Contract tests for the EPA helper API (run by the integration owner)."""

import json
import numpy as np
import pandas as pd
import pytest
from epa_experiment_utils import (
    MODELS,
    epa_labels,
    candidate_grid,
    balanced_weights,
    threshold,
    compute_metrics,
    connected_groups,
    nested_splits,
    feature_columns,
    load_csv,
    build_estimator,
    predict_scores,
    fit_weighted,
    save_record,
    load_record,
)


def test_epa_mapping_uses_final_labels_and_all_classes():
    out = epa_labels(np.array([1, 2, 3, 4]))
    assert out["A"].tolist() == [1, 1, 0, 0]
    assert out["B"].tolist() == [1, 1, 1, 0]


def test_exact_candidate_grids():
    assert MODELS == ("lr", "rf", "et", "svm", "xgb", "lgbm")
    for model in MODELS:
        grid = candidate_grid(model)
        assert len(grid) == 20
        assert [x["candidate_id"] for x in grid] == [
            f"{model}-{i:03d}" for i in range(20)
        ]


def test_balanced_weights_and_threshold_contract():
    y = np.array([0, 0, 1, 1])
    scores = np.array([-2.0, -0.1, 0.2, 3.0])
    assert np.allclose(balanced_weights(y), 1.0)
    t = threshold(scores, y)
    assert np.isfinite(t) and scores.min() < t < scores.max()
    pred = (scores >= t).astype(int)
    result = compute_metrics(y, scores, pred)
    assert {"sensitivity", "specificity", "MCC", "ROC-AUC"} <= set(result)


def test_grouped_connected_components_and_shared_splits():
    df = pd.DataFrame(
        {
            "comp_group_id": ["a", "a", "b", "c", "d", "e", "f", "g"],
            "final_label": [1, 2, 3, 4, 1, 2, 3, 4],
            "split_group_id": ["x", "x", "y", "z", "q", "r", "s", "t"],
        }
    )
    groups = connected_groups(df, "eye")
    assert len(groups) == len(df) and groups[0] == groups[1]
    epa = np.tile([1, 2, 3, 4], 30)
    group_ids = np.arange(len(epa))
    folds = nested_splits(epa, group_ids, seed=42)
    assert len(folds) == 5
    assert sorted(i for fold in folds for i in fold["test"]) == list(range(len(epa)))
    for fold in folds:
        assert not set(fold["train"]) & set(fold["test"])
        assert len(fold["inner"]) == 3
        for inner in fold["inner"]:
            assert not set(inner["train"]) & set(inner["validation"])
            assert set(inner["train"] + inner["validation"]) == set(fold["train"])


def test_feature_allowlist_exact_counts():
    header = [f"fp_{i}" for i in range(2048)] + [f"desc_{i}" for i in range(217)]
    assert len(feature_columns(header)) == 2265
    with pytest.raises(ValueError):
        feature_columns(header[:-1])


def test_fit_and_prediction_api_smoke():
    est = build_estimator("lr", candidate_grid("lr")[0]["params"])
    X = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 2.0], [2.0, 0.0]])
    y = np.array([0, 1, 0, 1])
    fit_weighted(est, X, y)
    scores = predict_scores(est, X, "lr")
    assert scores.shape == (4,) and np.isfinite(scores).all()


def test_atomic_checkpoint_identity_and_tamper_rejection(tmp_path):
    path = tmp_path / "record.json"
    save_record(path, "identity-a", {"fold": 1, "value": "ok"})
    assert load_record(path, "identity-a")["value"] == "ok"
    payload = json.loads(path.read_text())
    payload["payload"]["value"] = "tampered"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_record(path, "identity-a")
    with pytest.raises(ValueError):
        load_record(path, "identity-b")


@pytest.mark.parametrize("labels", [[1.5, 2], [0, 4], [1, np.nan], [1, np.inf]])
def test_invalid_epa_rejected(labels):
    with pytest.raises(ValueError):
        epa_labels(labels)


def test_threshold_extreme_and_adjacent_values():
    assert threshold([-1e308, 1e308], [0, 1]) == 0
    a = 1.0
    b = np.nextafter(a, np.inf)
    assert threshold([a, b], [0, 1]) == b
    for scores, labels in [([1, 1], [0, 1]), ([np.inf, 2], [0, 1]), ([1, 2], [1, 1])]:
        with pytest.raises(ValueError):
            threshold(scores, labels)


def test_zero_imputation_does_not_learn_heldout_values():
    X = np.array([[np.nan, 1], [np.nan, 2], [np.nan, 3], [np.nan, 4]])
    est = fit_weighted(
        build_estimator("lr", candidate_grid("lr")[0]["params"]),
        X,
        np.array([0, 1, 0, 1]),
    )
    assert est.named_steps["imputer"].statistics_.tolist() == [0, 0]
    assert est[:-1].transform([[1000.0, 1000.0]]).shape == (1, 2)
    assert est.named_steps["scaler"].mean_.tolist() == [0, 2.5]


def test_pooled_metrics_use_actual_fold_predictions():
    y = np.array([0, 1, 0, 1])
    scores = np.array([0.8, 0.9, 0.1, 0.2])
    predictions = np.array([0, 1, 0, 1])
    result = compute_metrics(y, scores, predictions)
    assert result["sensitivity"] == result["specificity"] == result["MCC"] == 1
    assert result["ROC-AUC"] < 1


def test_connected_groups_merge_transitive_links():
    df = pd.DataFrame(
        {"comp_group_id": ["a", "a", "b", "c"], "split_group_id": ["x", "y", "y", "z"]}
    )
    groups = connected_groups(df, "eye")
    assert groups[0] == groups[1] == groups[2]
    assert groups[3] != groups[0]


def test_checkpoint_refuses_changed_payload(tmp_path):
    path = tmp_path / "record.json"
    save_record(path, "id", {"x": 1})
    with pytest.raises(ValueError):
        save_record(path, "id", {"x": 2})


def test_inner_oof_excludes_each_predicted_row(monkeypatch):
    import epa_experiment_utils as utils

    epa = np.tile([1, 2, 3, 4], 30)
    X = np.random.default_rng(42).normal(size=(len(epa), 4))
    X[:, 0] = np.arange(len(epa))
    y = epa_labels(epa)["A"]
    fold = nested_splits(epa, np.arange(len(epa)))[0]
    seen = []
    original_fit = utils.fit_weighted
    original_predict = utils.predict_scores

    def audited_fit(est, train, target):
        est._audited_rows = set(train[:, 0])
        seen.append(est)
        return original_fit(est, train, target)

    def audited_predict(est, heldout, model):
        assert not est._audited_rows & set(heldout[:, 0])
        return original_predict(est, heldout, model)

    monkeypatch.setattr(utils, "fit_weighted", audited_fit)
    monkeypatch.setattr(utils, "predict_scores", audited_predict)
    result = utils._inner_scores(
        "lr",
        candidate_grid("lr")[0]["params"],
        X,
        y,
        np.array(fold["train"]),
        fold["inner"],
    )
    assert len({id(est) for est in seen}) == 3
    assert len(result["scores"]) == len(fold["train"])
    assert np.isfinite(result["scores"]).all()


def test_target_infeasibility_rejected_even_with_four_epa_values():
    epa = np.array([1, 2] + [3, 4] * 20)
    with pytest.raises(ValueError):
        nested_splits(epa, np.arange(len(epa)))


def test_missing_eye_group_rejected():
    df = pd.DataFrame({"comp_group_id": ["a", "b"], "split_group_id": ["x", None]})
    with pytest.raises(ValueError):
        connected_groups(df, "eye")


def test_feature_family_substitution_rejected():
    header = [f"fp_{i}" for i in range(2047)] + [f"desc_{i}" for i in range(218)]
    with pytest.raises(ValueError):
        feature_columns(header)


def test_completed_pointer_rejects_other_run(tmp_path):
    from epa_experiment_utils import completed_pointer

    root = tmp_path / "requested-run"
    root.mkdir()
    pointer = {
        "identity": "a" * 64,
        "output_dir": str(tmp_path / "other-run" / "skin" / ("a" * 16)),
    }
    (root / "skin_completed.json").write_text(json.dumps(pointer), encoding="utf-8")
    with pytest.raises(ValueError, match="escapes"):
        completed_pointer(root, "skin")


def test_completed_pointer_checks_endpoint_and_run(tmp_path):
    from epa_experiment_utils import completed_pointer, hash_json

    root = tmp_path / "requested-run"
    config = {"endpoint": "eye", "run_id": root.name}
    identity = hash_json(config)
    output = root / "skin" / identity[:16]
    save_record(output / "config.json", identity, config)
    (root / "skin_completed.json").write_text(
        json.dumps({"identity": identity, "output_dir": str(output)}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="identity mismatch"):
        completed_pointer(root, "skin")
