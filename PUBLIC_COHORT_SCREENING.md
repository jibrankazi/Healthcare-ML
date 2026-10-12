# Independent clinical dataset source screening

**Outcome:** the fitted model requires the same 30 fine-needle-aspirate
image-derived measurements and **benign versus malignant diagnosis** labels
in a new patient cohort; merely having "breast cancer" in the title does not
make an external dataset applicable.

A new executable checks live official UCI repository metadata for:
- [UCI 15: Wisconsin Original](https://archive.ics.uci.edu/dataset/15): 699 cases, nine graded cytology features; cannot be passed to the 30-feature WDBC model.
- [UCI 16: Wisconsin Prognostic](https://archive.ics.uci.edu/dataset/16): 198 follow-up records of already diagnosed cancer cases; its target is recurrence, **not malignant vs benign** (and may involve related Wisconsin cases).
- [UCI 451: Breast Cancer Coimbra](https://archive.ics.uci.edu/dataset/451): 116 records of clinical/blood biomarkers and controls, not the WDBC cytology-image features.

Run `python -m hchealth.external_source_screening`; inspect
`runs/uci_external_cohort_suitability.json`. The script queries the genuine
UCI API metadata and records retrieval failures openly; tests use synthetic
metadata fixtures only, never fabricated clinical validation cases.

**No clinical external validation can be claimed from these three candidates.**
No patient-level records are altered, and the trained diagnostic model is not
run with missing, renamed, converted or padded predictors. To finish an actual
independent diagnostic validation, we must obtain a legitimately collected,
authorized separate cohort with the **same 30 measured predictors**, the
appropriate diagnosis outcome, reliable units and documented patient/site
separation, then use the existing `hchealth.external_validate` evaluator.
