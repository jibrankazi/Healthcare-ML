# External clinical cohort validation — not yet established

The 569-record Wisconsin Diagnostic Breast Cancer dataset was collected in one
research context with 30 cytology-image-derived predictors. A second download
or rehost of those same patients **is not external validation**. UCI documents
the original collection: https://archive.ics.uci.edu/dataset/17/breast-cancer-wisconsin-diagnostic

This repository now has an **executable fail-closed evaluator** for a genuinely
separate clinical cohort with the **same 30 measured predictors** and a binary
diagnosis label. It rejects any exact feature-row overlap with the 569 WDBC
records and checks model feature schema, finite values, patient IDs, diagnosis
mapping and supplied source metadata. It does not retrain/tune the model.

Example *only when a suitably licensed independent cohort exists*:

```sh
python -m hchealth.external_validate --csv /secure/cohort.csv \
  --model runs/ci_demo/model_LogReg.joblib \
  --source-url https://your-authorized-source.example/dataset \
  --site "Documented external clinical site" \
  --period "Documented collection dates" \
  --output runs/external_cohort_review.json
```

Never commit clinical records, identifiers, credentials or protected data.
This script does not download or invent patients. Supplying a site name is NOT
independent proof: a clinical data custodian must verify patient separation,
collection and outcome definitions, permissions, feature-unit equivalence,
sampling, calibration, prevalence, subgroups, and intended use. The output
intentionally sets `clinical_external_validation_established=false` until
an independently documented clinical verification has occurred.

Tests exercise only synthetic software fixtures; they are not external cohort
results. No diagnostic model from this research repo is approved for patient care.
