import numpy as np

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


def test_fallback_on_bad_hf_id():
    """An invalid HF dataset ID should fall back to breast cancer."""
    X_train, X_test, y_train, y_test = load_tabular(hf_id="nonexistent/fake_dataset_999")
    total = len(y_train) + len(y_test)
    assert total == 569


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
