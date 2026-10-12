"""Faked metadata is used only to test schema decisions, not as patient data."""
import pytest

from hchealth.external_source_screening import CANDIDATES, assess_dataset, screen


def info(name, predictors, target="Class"):
    return {"name": name, "variables": (
        [{"name": f"feature_{j}", "role": "Feature"} for j in range(predictors)] +
        [{"name": target, "role": "Target"}])}


@pytest.mark.parametrize("dataset_id,predictors", [(15, 9), (16, 30), (451, 9)])
def test_incompatible_public_candidates_are_not_falsely_validated(dataset_id, predictors):
    record = assess_dataset(dataset_id, info("software fixture", predictors))
    assert record["official_feature_count_by_roles"] == predictors
    assert record["may_evaluate_trained_wdbc_model_on_this_dataset"] is False
    assert record["clinical_external_validation_established"] is False
    assert len(record["rejection_reason"]) > 20


def test_incomplete_metadata_cannot_be_promoted_to_external_cohort():
    with pytest.raises(ValueError, match="variable role"):
        assess_dataset(15, {"name": "Wisconsin Original"})
    with pytest.raises(ValueError, match="predictors or outcome"):
        assess_dataset(16, {"name": "Prognostic", "variables": []})


def test_official_service_error_does_not_become_fake_verification():
    class Mock:
        def get(self, *args, **kwargs):
            raise ConnectionError("offline fixture")
    report = screen(Mock())
    assert report["official_metadata_verified_count"] == 0
    assert not report["external_diagnostic_model_validation_completed"]
    assert all(not r["official_metadata_verified"] for r in report["candidates"])
    assert set(report["candidate_uci_ids_reviewed"]) == set(CANDIDATES)
