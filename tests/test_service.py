"""Real HTTP serving contract and independent aggregate accounting, fixture models only."""
import json

from fastapi.testclient import TestClient
import joblib
import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from hchealth import service


@pytest.fixture()
def client(tmp_path, monkeypatch):
    original = load_breast_cancer(as_frame=True)
    X = original.data
    y = (original.target == 0).astype(int)
    # Fast unit fixture, not the real 5-model CV training or heldout benchmark.
    model = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=250)),
    ]).fit(X.iloc[::6], y.iloc[::6])
    out = tmp_path
    joblib.dump(model, out / "model_LogReg.joblib")
    (out / "train_meta.json").write_text(json.dumps({
        "data_source": "sklearn Wisconsin Breast Cancer",
        "positive_label": 1, "positive_outcome": "malignant",
        "n_features": 30, "models": ["LogReg"],
        "cv_results": {"LogReg": {"cv_auroc_mean": 0.5}},
    }))
    training = X.iloc[::6]
    (out / "monitoring_reference.json").write_text(json.dumps({
        "features": list(X.columns),
        "n_training_rows": len(training),
        "lower_training_percentile_01": {
            c: float(np.quantile(training[c], .01)) for c in X.columns},
        "upper_training_percentile_99": {
            c: float(np.quantile(training[c], .99)) for c in X.columns},
    }))
    monkeypatch.setenv("HC_RUN_DIR", str(out))
    service.get_serving_model.cache_clear()
    yield TestClient(service.app), model, X
    service.get_serving_model.cache_clear()


def test_train_save_serve_exact_predict_proba_and_monitor_counts(client):
    app, model, source = client
    assert app.get("/health").status_code == 200
    schema = app.get("/schema").json()
    assert schema["positive_label"] == 1
    assert len(schema["features_in_order"]) == 30
    rows = source.iloc[200:204].to_dict(orient="records")
    predicted = app.post("/predict", json={"records": rows})
    assert predicted.status_code == 200, predicted.text
    result = predicted.json()
    assert result["research_only"] and not result["clinical_decision_support"]
    expected = model.predict_proba(source.iloc[200:204])[:, 1]
    assert np.allclose(expected, result["research_probability_malignant"], atol=1e-12)
    metrics = app.get("/metrics").json()
    assert metrics["requests"] == 1
    assert metrics["observations"] == 4
    assert sum(metrics["probability_bins_0_20_40_60_80_100"]) == 4
    assert metrics["feature_cells_evaluated"] == 120
    assert metrics["monitoring_excludes_raw_records"]
    assert "records" not in metrics and "features" not in metrics
    assert metrics["reference_rows"] == len(source.iloc[::6])


@pytest.mark.parametrize("invalid", [
    {"mean radius": 10},
    {},
    {c: 1.0 for c in load_breast_cancer(as_frame=True).feature_names[:-1]},
])
def test_rejects_missing_or_unmatched_original_features(client, invalid):
    app, _, _ = client
    result = app.post("/predict", json={"records": [invalid]})
    assert result.status_code == 422
    assert app.get("/metrics").json()["observations"] == 0


def test_refuses_extra_or_nonfinite_numeric_payload(client):
    app, _, x = client
    row = x.iloc[0].to_dict()
    row["outcome"] = 1
    assert app.post("/predict", json={"records": [row]}).status_code == 422
    row.pop("outcome")
    row["mean radius"] = "NaN"
    assert app.post("/predict", json={"records": [row]}).status_code == 422
    assert app.post("/predict", json={"records": []}).status_code == 422


def test_fails_closed_without_fitted_local_artifact(tmp_path, monkeypatch):
    monkeypatch.setenv("HC_RUN_DIR", str(tmp_path))
    service.get_serving_model.cache_clear()
    try:
        app = TestClient(service.app)
        assert app.get("/health").status_code == 503
        assert app.get("/metrics").status_code == 503
    finally:
        service.get_serving_model.cache_clear()
