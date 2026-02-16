import numpy as np
from sklearn.datasets import load_breast_cancer

from hchealth.data import load_tabular


def test_load_tabular_returns_four_splits():
    result = load_tabular()
    assert len(result) == 4, "Expected (X_train, X_test, y_train, y_test)"


def test_load_tabular_shapes():
    X_train, X_test, y_train, y_test = load_tabular()
    assert X_train.shape[0] == len(y_train)
    assert X_test.shape[0] == len(y_test)
    assert X_train.shape[0] + X_test.shape[0] == 569  # breast cancer total


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
