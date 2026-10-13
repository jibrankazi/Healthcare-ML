"""Regression tests for the deployable research inference boundary.

The tiny fitting fixture checks the software contract, not heldout performance.
The full original-data training and inference lifecycle is a separate CI step.
"""
import json

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from hchealth.infer import run_inference


@pytest.fixture
def research_paths(tmp_path):
    source = load_breast_cancer(as_frame=True)
    X = source.data
    y = (source.target == 0).astype(int)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=250))
    model.fit(X.iloc[:120], y.iloc[:120])
    run_dir = tmp_path / "trained"
    run_dir.mkdir()
    joblib.dump(model, run_dir / "model_LogReg.joblib")
    metadata = {
        "data_source": "sklearn Wisconsin Breast Cancer",
        "positive_label": 1, "positive_outcome": "malignant",
        "n_features": 30, "models": ["LogReg"],
        "cv_results": {"LogReg": {"cv_auroc_mean": .91}},
    }
    (run_dir / "train_meta.json").write_text(json.dumps(metadata))
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({
        "data": {"hf_dataset_id": "none", "target_column": "target"},
        "outputs": {"run_dir": str(run_dir)},
    }))
    input_path = tmp_path / "unlabeled.csv"
    X.iloc[120:133].to_csv(input_path, index=False)
    return config, input_path, tmp_path / "output.json", run_dir


def test_inference_reloads_model_and_validates_probability_contract(research_paths):
    config, csv, output, run_dir = research_paths
    result = run_inference(config, csv, output)
    assert result["positive_outcome"] == "malignant"
    assert result["positive_label"] == 1
    assert result["selected_model"] == "LogReg"
    assert result["n_rows"] == 13
    assert result["n_features"] == 30
    assert not result["clinical_decision_support"]
    assert not result["external_clinical_validation_established"]
    assert all(len(result[k]) == 64 for k in (
        "input_sha256", "model_sha256", "training_metadata_sha256"
    ))
    selected = joblib.load(run_dir / "model_LogReg.joblib")
    expected = selected.predict_proba(pd.read_csv(csv))[:, 1]
    assert np.array_equal(np.asarray(result["probabilities_malignant"]), expected)
    assert json.loads(output.read_text()) == result
    assert run_inference(config, csv, output) == result


@pytest.mark.parametrize("corruption", ["extra_target", "shuffled_columns", "missing_col", "infinity", "nonnumeric"])
def test_refuses_unseen_or_invalid_feature_schema(research_paths, corruption):
    config, csv, output, _ = research_paths
    df = pd.read_csv(csv)
    if corruption == "extra_target":
        df["target"] = 1
    elif corruption == "shuffled_columns":
        df = df[df.columns[::-1]]
    elif corruption == "missing_col":
        df = df.drop(columns=df.columns[-1])
    elif corruption == "infinity":
        df.iloc[0, 0] = np.inf
    elif corruption == "nonnumeric":
        df.iloc[0, 0] = "not-a-measurement"
    df.to_csv(csv, index=False)
    with pytest.raises(ValueError):
        run_inference(config, csv, output)
    assert not output.exists()


def test_fails_closed_if_label_or_source_metadata_disagree(research_paths):
    config, csv, output, run_dir = research_paths
    path = run_dir / "train_meta.json"
    original = json.loads(path.read_text())
    original["positive_outcome"] = "benign"
    path.write_text(json.dumps(original))
    with pytest.raises(ValueError, match="source or malignant-positive"):
        run_inference(config, csv, output)
    assert not output.exists()


def test_fails_closed_on_unfitted_or_wrong_schema_model(research_paths):
    config, csv, output, run_dir = research_paths
    joblib.dump(StandardScaler(), run_dir / "model_LogReg.joblib")
    with pytest.raises(ValueError):
        run_inference(config, csv, output)
    assert not output.exists()
