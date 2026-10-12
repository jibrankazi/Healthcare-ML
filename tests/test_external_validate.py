"""Software fixtures are not independent patient observations."""
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from hchealth.external_validate import FEATURES, evaluate_external


@pytest.fixture
def fitted(tmp_path):
    bundle = load_breast_cancer(as_frame=True)
    features = bundle.data.loc[:, FEATURES]
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(features, (bundle.target == 0).astype(int))
    location = tmp_path / "trusted_model.joblib"
    joblib.dump(model, location)
    return location, features


def make_fixture(tmp_path, source, *, change=True):
    # Synthetic feature perturbation is a software test, NOT new clinical data.
    external = source.iloc[:50].copy().reset_index(drop=True)
    if change:
        external.iloc[:, :] = external.to_numpy() + .000123
    bundle = load_breast_cancer()
    external["diagnosis"] = ["M" if int(v)==0 else "B" for v in bundle.target[:50]]
    external["patient_id"] = ["SOFTWARE_FIXTURE_" + str(i) for i in range(50)]
    path = tmp_path / "software_fixture.csv"
    external.to_csv(path, index=False)
    return path


def run(path, model):
    return evaluate_external(path, model, source_url="https://example.org/fixture",
                             collection_site="TEST ONLY", collection_period="TEST ONLY")


def test_rehosted_original_wdbc_is_rejected(fitted, tmp_path):
    model, source = fitted
    path = make_fixture(tmp_path, source, change=False)
    with pytest.raises(ValueError, match="overlap original Wisconsin"):
        run(path, model)


def test_external_gate_computes_only_exploratory_metrics(fitted, tmp_path):
    model, source = fitted
    path = make_fixture(tmp_path, source)
    result = run(path, model)
    assert result["n_records"] == 50
    assert 0 <= result["auroc"] <= 1
    assert result["clinical_external_validation_established"] is False
    assert result["source_site_patient_disjointness_independently_audited"] is False


def test_feature_schema_patient_ids_and_provenance_fail_closed(fitted, tmp_path):
    model, source = fitted
    path = make_fixture(tmp_path, source)
    frame = pd.read_csv(path)
    frame.loc[0, "patient_id"] = frame.loc[1, "patient_id"]
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="unique"):
        run(path, model)
    frame.loc[0, "patient_id"] = "FIXED"
    frame = frame.drop(columns=[FEATURES[0]])
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing required"):
        run(path, model)
    with pytest.raises(ValueError, match="HTTPS source"):
        evaluate_external(path, model, source_url="", collection_site="", collection_period="")
