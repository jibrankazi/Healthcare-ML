# Healthcare-ML — reproducible public research dataset

Python pipeline for binary classification using **the real, publicly distributed scikit-learn Wisconsin Breast Cancer dataset** (569 records; 30 measured features). These are historical diagnostic feature measurements, **not live electronic health records, a clinical API, or evidence of clinical deployment**.

The code implements five algorithms, stratified train/test splitting, cross-validation, held-out AUROC/AUPRC and Brier scores, bootstrap intervals, plots including SHAP interpretation, and a LaTeX results table. The validity of any specific metric is established by a completed execution and its output artifact, **not by an example metric printed in this README**.

## Interview-ready engineering enhancement: saved-model inference

A new bounded research-only `hchealth.infer` CLI scores a **feature-only CSV**
using the model selected from training cross-validation rather than selecting
whichever model looks best on the held-out test. It reloads the serialized local
model, verifies the exact original 30-column feature schema and order, refuses
invalid numeric values or wrongly labeled model metadata, and writes positive
class probabilities together with source, model and input SHA-256 identifiers.

After you run the existing training pipeline, prepare a demonstration using
**original public Wisconsin records**, never real patient records:

```bash
python -m hchealth.run_pipeline --config configs/ci_demo.yaml
python - <<'PY'
from pathlib import Path
from hchealth.data import load_tabular
_, X_test, _, _ = load_tabular()
Path("runs/ci_demo").mkdir(parents=True, exist_ok=True)
X_test.to_csv("runs/ci_demo/heldout_unlabeled.csv", index=False)
PY
python -m hchealth.infer --config configs/ci_demo.yaml \
  --input-csv runs/ci_demo/heldout_unlabeled.csv \
  --output runs/ci_demo/heldout_inference.json
```

This exercise answers common engineering interview questions: *Which fitted
model is used for inference? Was the heldout test used to pick it? How is the
input schema enforced? How do you reproduce an output? Why is the positive
class malignant rather than benign?* The CI workflow independently reloads
the selected classifier and compares every computed probability on the heldout
public measurements. The unit tests explicitly reject incorrectly ordered,
missing, nonnumeric, infinite or extra columns, and inconsistent class-label
metadata.

**Security and validity:** Load only trusted locally produced model files;
`joblib` is pickle-based and must not deserialize untrusted uploads. The
trained data are historical public research records. These results **do not**
constitute a patient risk score, diagnosis, externally validated clinical
performance or a production medical device. The CLI is a reproducible offline
research inference boundary, not an authenticated HTTP service.

---

## End-to-end lifecycle

```bash
pip install -e ".[dev]"
python -m pytest -q
python -m hchealth.run_pipeline --config configs/ci_demo.yaml
```

Stages executed by `hchealth.run_pipeline`:

1. Select the actual Wisconsin Breast Cancer dataset explicitly (`hf_dataset_id: "none"` in `configs/ci_demo.yaml`).
2. Create a reproducible stratified training/holdout split, cross-validate five models, train and serialize fitted models.
3. Evaluate every fitted model on the held-out set; report AUROC, AUPRC, Brier score, and bootstrap confidence intervals.
4. Save ROC, PR, calibration and SHAP plots.
5. Export the comparison table as LaTeX.

The [October 11, 2026 corrected-target end-to-end execution](https://github.com/jibrankazi/Healthcare-ML/actions/runs/38164590942) **passed 12 tests** and all four stages (train, held-out evaluation, figures including SHAP, and LaTeX) on the real sklearn-distributed Wisconsin Breast Cancer dataset. Its GitHub Actions workflow validated data shape, split, five model results and required files, then uploaded the **wisconsin-breast-cancer-complete-research-lifecycle** artifact. This supports the configured *research demonstration*, not hospital integration or clinical validity.

The research-sized configuration `configs/clinical_demo.yaml` requests 5-fold cross-validation and 1,000 bootstrap resamples and takes longer. CI uses a deliberately smaller compute budget (2 folds, 30 bootstrap resamples), **not a substitute for evaluating the research configuration**.

## Positive outcome and holdout integrity

The original scikit-learn Wisconsin Breast Cancer dataset encodes **0 = malignant and 1 = benign**. For this research classifier the loader explicitly recodes to **1 = malignant, 0 = benign** (212 malignant records; 357 benign records). The original measured features and rows are unchanged. Positive-class AUPRC, model probabilities, calibration and class-1 SHAP explanations now refer to malignant diagnoses.

**Old execution artifacts generated before this correction used class 1 = benign.** They must not be interpreted as malignancy-risk measurements. Re-run the current pipeline and inspect its generated artifacts for the new outcome convention. This is historical diagnostic research data, not a clinical decision system.

The model explained using SHAP is selected from *training-set cross-validation AUROC* rather than using held-out AUROC for model selection. The output metadata makes the positive-label meaning, source, class counts, and selection basis explicit.

## Dataset integrity

- `hf_dataset_id: "none"` means the public Wisconsin Breast Cancer data included with scikit-learn; it is an explicit, real-data selection, not a randomly generated fixture.
- When a Hugging Face dataset ID is supplied, retrieval **must succeed** and the specified target column **must exist**. An error will stop the pipeline instead of silently switching to the sklearn dataset.
- Training and evaluation call the same loader and fixed split. CI checks 569 observations, 30 input features and 212 malignant positive labels.
- External Hugging Face datasets must use binary 0/1 target labels; the meaning of an external label 1 remains source-specific and must be independently verified.
- There are **no independently verified patient-level outcomes beyond the included research dataset**, no prospective clinical trial, and no validated medical decision support deployment.

## Observed corrected-target results (October 11, 2026)

In the passing [GitHub Actions run](https://github.com/jibrankazi/Healthcare-ML/actions/runs/38164590942), the 114-record holdout contained 42 malignant diagnoses. The observed metrics under the **CI demo configuration only** (2-fold CV, 10 estimators, 30 bootstrap samples) were:

| Model | Held-out AUROC | Held-out AUPRC | Brier |
| --- | ---: | ---: | ---: |
| Logistic Regression | 0.9874 | 0.9883 | 0.0289 |
| Random Forest | 0.9876 | 0.9848 | 0.0425 |
| RF + Isotonic | 0.9879 | 0.9870 | 0.0417 |
| RF + Platt | 0.9864 | 0.9870 | 0.0404 |
| XGBoost | 0.9864 | 0.9827 | 0.0683 |

The training-CV-selected SHAP explanation model was Logistic Regression (2-fold training CV AUROC 0.9964). The score differences do not establish superiority or clinical value. The earlier October 9 workflow artifact used the original benign-positive label convention and has been superseded for malignancy-risk interpretation.

The tracked `paper/main.pdf` and `paper/figures/` are **legacy binary artifacts not regenerated for the corrected outcome**. They must not be used as evidence of current results. The corrected `paper/main.tex` and `paper/results.tex` source documents describe verified CI output; current generated plots and LaTeX are in the October 11 workflow artifact. The larger five-fold / 300-tree research configuration remains unverified.

## Outputs

- `runs/ci_demo/train_meta.json`: dataset shape, split, model names and CV metrics
- `runs/ci_demo/results.json`: held-out results for each model
- `paper/ci_figures/*.png`: generated interpretation and evaluation figures
- `paper/ci_results.tex`: LaTeX comparison table

Check the specific CI run's uploaded artifacts rather than treating historical approximate AUROC scores as replicated measurements.

## Scope of the project

The implementation uses scikit-learn, XGBoost, SHAP, matplotlib, and optional Hugging Face Datasets access. This is a public-data research demonstration; it should **not** be used to make patient care decisions. The repository does not demonstrate a connected hospital system, externally validated clinical performance, or a regulatory authorization.


## External-cohort diagnostic validation gate (October 2026)

The new [external validation protocol](EXTERNAL_VALIDATION.md) and executable
`hchealth.external_validate` evaluate a **separate** 30-feature breast-cancer
cohort without refitting the model, reject exact reused Wisconsin source
observations, and check source metadata, outcomes, schema and patient IDs.
[GitHub Actions passed the new regression tests](https://github.com/jibrankazi/Healthcare-ML/actions/runs/38279855171).

**Independent clinical verification is still outstanding:** no new external
patient cohort was introduced by these tests, and source/site independence
requires clinical-custodian review. Research results are not approved for care.
