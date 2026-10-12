"""Check real UCI candidate-cohort metadata before claiming external validation.

This screens 3 genuine UCI datasets. NO new data are passed to the fitted
WDBC diagnostic classifier, because incompatible outcomes/features must not
be silently relabeled, padded, or treated as new benign/malignant patients.
"""
import argparse
import json
from pathlib import Path

import requests

API = "https://archive.ics.uci.edu/api/dataset"
CANDIDATES = {
    15: {"name": "Wisconsin Original", "expected_features": 9,
         "outcome": "benign_vs_malignant",
         "reason": "Only nine ordinal cytology variables; not the 30 WDBC image-derived features."},
    16: {"name": "Wisconsin Prognostic", "expected_features": None,
         "outcome": "recurrence_after_cancer",
         "reason": "Recurrence outcome among diagnosed cancer patients is not a benign/malignant diagnosis target; likely related Wisconsin clinical source."},
    451: {"name": "Coimbra", "expected_features": 9,
          "outcome": "patient_vs_healthy_control",
          "reason": "Blood biomarkers/age/BMI versus WDBC image-derived nuclear morphology; not interchangeable predictor measurements."},
}


def assess_dataset(dataset_id, metadata):
    item = CANDIDATES[dataset_id]
    if not isinstance(metadata, dict) or not metadata.get("name"):
        raise ValueError("Missing official UCI dataset identity")
    features = metadata.get("variables")
    if not isinstance(features, list):
        raise ValueError("Missing official UCI variable role schema")
    predictors = [x["name"] for x in features if x.get("role") == "Feature"]
    targets = [x["name"] for x in features if x.get("role") == "Target"]
    if not predictors or not targets:
        raise ValueError("UCI source metadata lacks documented predictors or outcome")
    return {
        "uci_id": dataset_id,
        "official_name": metadata["name"],
        "official_feature_count_by_roles": len(predictors),
        "official_feature_names": predictors,
        "official_target_names": targets,
        "documented_study_outcome": item["outcome"],
        "30_wdbc_predictors_and_same_diagnostic_endpoint_compatible": False,
        "may_evaluate_trained_wdbc_model_on_this_dataset": False,
        "clinical_external_validation_established": False,
        "rejection_reason": item["reason"],
        "uci_source_url": f"https://archive.ics.uci.edu/dataset/{dataset_id}",
        "official_metadata_verified": True,
    }


def screen(session=None):
    client = session or requests
    outcomes = []
    for dataset_id, item in CANDIDATES.items():
        try:
            response = client.get(API, params={"id": dataset_id}, timeout=30)
            response.raise_for_status()
            body = response.json()
            if body.get("status") != 200 or not isinstance(body.get("data"), dict):
                raise ValueError("UCI metadata endpoint did not return confirmed dataset")
            result = assess_dataset(dataset_id, body["data"])
        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            result = {
                "uci_id": dataset_id,
                "uci_source_url": f"https://archive.ics.uci.edu/dataset/{dataset_id}",
                "official_metadata_verified": False,
                "30_wdbc_predictors_and_same_diagnostic_endpoint_compatible": False,
                "may_evaluate_trained_wdbc_model_on_this_dataset": False,
                "clinical_external_validation_established": False,
                "metadata_retrieval_issue": type(exc).__name__ + ": " + str(exc)[:180],
            }
        outcomes.append(result)
    return {
        "original_training_dataset_uci_id": 17,
        "original_training_outcome": "malignant vs benign (WDBC 30 FNA-image features)",
        "candidate_uci_ids_reviewed": list(CANDIDATES),
        "official_metadata_verified_count": sum(x["official_metadata_verified"] for x in outcomes),
        "external_diagnostic_model_validation_completed": False,
        "no_synthetic_or_rehosted_patient_data_used": True,
        "no_model_predictions_on_incompatible_cohorts": True,
        "candidates": outcomes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/uci_external_cohort_suitability.json")
    args = parser.parse_args()
    report = screen()
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
