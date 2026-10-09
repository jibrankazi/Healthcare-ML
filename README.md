# Healthcare-ML — reproducible public research dataset

Python pipeline for binary classification using **the real, publicly distributed scikit-learn Wisconsin Breast Cancer dataset** (569 records; 30 measured features). These are historical diagnostic feature measurements, **not live electronic health records, a clinical API, or evidence of clinical deployment**.

The code implements five algorithms, stratified train/test splitting, cross-validation, held-out AUROC/AUPRC and Brier scores, bootstrap intervals, plots including SHAP interpretation, and a LaTeX results table. The validity of any specific metric is established by a completed execution and its output artifact, **not by an example metric printed in this README**.

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

The [October 9 verified end-to-end execution](https://github.com/jibrankazi/Healthcare-ML/actions/runs/37940182428) **passed** all four stages (train, held-out evaluation, figures including SHAP, and LaTeX) on the real sklearn-distributed Wisconsin Breast Cancer dataset. Its GitHub Actions workflow validated data shape, split, five model results and required files, then uploaded the **wisconsin-breast-cancer-complete-research-lifecycle** artifact. This supports the configured *research demonstration*, not hospital integration or clinical validity.

The research-sized configuration `configs/clinical_demo.yaml` requests 5-fold cross-validation and 1,000 bootstrap resamples and takes longer. CI uses a deliberately smaller compute budget (2 folds, 30 bootstrap resamples), **not a substitute for evaluating the research configuration**.

## Dataset integrity

- `hf_dataset_id: "none"` means the public Wisconsin Breast Cancer data included with scikit-learn; it is an explicit, real-data selection, not a randomly generated fixture.
- When a Hugging Face dataset ID is supplied, retrieval **must succeed** and the specified target column **must exist**. An error will stop the pipeline instead of silently switching to the sklearn dataset.
- Training and evaluation call the same loader and fixed split. CI checks the expected 569 observations and 30 input features.
- There are **no independently verified patient-level outcomes beyond the included research dataset**, no prospective clinical trial, and no validated medical decision support deployment.

## Outputs

- `runs/ci_demo/train_meta.json`: dataset shape, split, model names and CV metrics
- `runs/ci_demo/results.json`: held-out results for each model
- `paper/ci_figures/*.png`: generated interpretation and evaluation figures
- `paper/ci_results.tex`: LaTeX comparison table

Check the specific CI run's uploaded artifacts rather than treating historical approximate AUROC scores as replicated measurements.

## Scope of the project

The implementation uses scikit-learn, XGBoost, SHAP, matplotlib, and optional Hugging Face Datasets access. This is a public-data research demonstration; it should **not** be used to make patient care decisions. The repository does not demonstrate a connected hospital system, externally validated clinical performance, or a regulatory authorization.
