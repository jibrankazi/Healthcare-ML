import numpy as np
import pytest
import sys
import types

from hchealth.data import load_tabular


def test_load_tabular_returns_four_splits():
    result = load_tabular()
    assert len(result) == 4, "Expected (X_train, X_test, y_train, y_test)"


def test_load_tabular_shapes():
    X_train, X_test, y_train, y_test = load_tabular()
    assert X_train.shape[0] == len(y_train)
    assert X_test.shape[0] == len(y_test)
    assert X_train.shape[0] + X_test.shape[0] == 569  # breast cancer total


def test_load_tabular_features():
    X_train, X_test, y_train, y_test = load_tabular()
    assert X_train.shape[1] == 30, "Breast cancer has 30 features"
    assert X_test.shape[1] == 30


def test_load_tabular_stratified():
    X_train, X_test, y_train, y_test = load_tabular()
    train_ratio = y_train.mean()
    test_ratio = y_test.mean()
    assert abs(train_ratio - test_ratio) < 0.05, "Splits should be roughly stratified"


def test_load_tabular_deterministic():
    a = load_tabular()
    b = load_tabular()
    assert np.array_equal(a[0].values, b[0].values), "Same seed should give same split"


def test_external_source_failure_does_not_silently_switch_dataset(monkeypatch):
    """A failing HF request must never be presented as a different dataset."""
    def unavailable(*args, **kwargs):
        raise ConnectionError("Source unavailable")

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=unavailable))
    with pytest.raises(RuntimeError, match="refusing to substitute"):
        load_tabular(hf_id="nonexistent/fake_dataset_999")


def test_missing_requested_target_is_rejected(monkeypatch):
    """HF schema mismatches must be errors, not guessed target labels."""
    import pandas as pd

    class ExternalDataset:
        def to_pandas(self):
            return pd.DataFrame({"feature": [1, 2, 3, 4], "wrong_target": [0, 1, 0, 1]})

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=lambda *args, **kwargs: ExternalDataset()))
    with pytest.raises(ValueError, match="Target 'target' absent"):
        load_tabular(hf_id="org/dataset", target_column="target")


def test_build_models():
    """Verify build_models returns all 5 expected models."""
    from hchealth.train import build_models

    cfg = {"seed": 42, "model": {"n_estimators": 10, "max_depth": 3}}
    models = build_models(cfg)
    assert len(models) == 5
    expected = {"LogReg", "RF", "RF+Isotonic", "RF+Platt", "XGBoost"}
    assert set(models.keys()) == expected


def test_bootstrap_metric():
    """Verify bootstrap_metric returns point estimate and CIs."""
    from hchealth.evaluate import bootstrap_metric
    from sklearn.metrics import roc_auc_score

    rng = np.random.RandomState(42)
    y_true = np.concatenate([np.zeros(30), np.ones(30)])
    y_score = np.concatenate([rng.beta(2, 5, size=30), rng.beta(5, 2, size=30)])

    point, lo, hi = bootstrap_metric(y_true, y_score, roc_auc_score, n_boot=500, seed=0)

    assert 0.0 <= lo <= point <= hi <= 1.0
    assert hi - lo > 0, "CI should have non-zero width"


def test_wisconsin_positive_class_means_malignant():
    from sklearn.datasets import load_breast_cancer

    source = load_breast_cancer(as_frame=True)
    assert list(source.target_names) == ["malignant", "benign"]
    X_train, X_test, y_train, y_test = load_tabular()
    for X, y in ((X_train, y_train), (X_test, y_test)):
        expected = (source.frame.loc[X.index, "target"] == 0).astype(int)
        assert np.array_equal(y.to_numpy(), expected.to_numpy())
    assert int(y_train.sum() + y_test.sum()) == 212


def test_external_nonbinary_target_rejected(monkeypatch):
    import pandas as pd

    class ExternalDataset:
        def to_pandas(self):
            return pd.DataFrame({
                "feature": [1, 2, 3, 4],
                "target": ["benign", "malignant", "benign", "malignant"],
            })

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(
        load_dataset=lambda *args, **kwargs: ExternalDataset()
    ))
    with pytest.raises(ValueError, match="binary 0/1"):
        load_tabular(hf_id="org/string_label_data")


def test_explanation_model_selected_from_training_cv():
    from hchealth.evaluate import choose_explanation_model

    meta = {
        "models": ["LogReg", "RF"],
        "cv_results": {
            "LogReg": {"cv_auroc_mean": 0.96},
            "RF": {"cv_auroc_mean": 0.93},
        },
    }
    assert choose_explanation_model(meta) == "LogReg"
