# Healthcare-ML Pipeline

Calibration-aware clinical risk modeling with multi-model comparison, SHAP interpretability, bootstrap confidence intervals, and automated LaTeX reporting.

## Overview

End-to-end research pipeline for clinical risk prediction that goes beyond discrimination metrics. Five classifiers are compared not just on AUROC/AUPRC, but on **probability calibration** -- whether a predicted 70% actually means 70% risk. Post-hoc calibration (isotonic and Platt scaling) is applied and evaluated. SHAP provides feature-level interpretability.

## Features

- **Multi-model comparison** -- Logistic Regression, Random Forest, RF+Isotonic calibration, RF+Platt calibration, XGBoost
- **5-fold stratified cross-validation** with per-model CV scores
- **Post-hoc calibration** -- `CalibratedClassifierCV` with isotonic regression and Platt scaling
- **Bootstrap confidence intervals** (1000 resamples) on AUROC and AUPRC
- **Brier score** for calibration quality assessment
- **SHAP interpretability** -- beeswarm and bar plots showing feature importance
- **API integration** -- Hugging Face Datasets with sklearn fallback
- **LaTeX sync** -- multi-model comparison table auto-exported to `paper/results.tex`
- **Reproducible** -- seeded, config-driven, pip-installable, cross-platform

## Quickstart

```bash
# Create virtual environment
python -m venv .venv

# Activate (pick your OS)
# Linux/macOS:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# Install package + dependencies
pip install -e ".[dev]"

# Run full pipeline (train + evaluate + latex)
python -m hchealth.run_pipeline --config configs/clinical_demo.yaml

# Or run steps individually
python -m hchealth.train --config configs/clinical_demo.yaml
python -m hchealth.evaluate --config configs/clinical_demo.yaml
python -m hchealth.to_latex --config configs/clinical_demo.yaml

# Run tests
pytest tests/ -v
```

On Unix systems with `make`:

```bash
make install   # pip install -e ".[dev]"
make all       # train + eval + latex
make test      # pytest
make paper     # pdflatex (requires TeX distribution)
```

## Project Structure

```
healthcare-ml-pipeline/
├── configs/
│   └── clinical_demo.yaml      # Experiment configuration
├── src/
│   └── hchealth/
│       ├── __init__.py
│       ├── data.py             # Data loading (HF API + fallback)
│       ├── train.py            # Multi-model training with CV
│       ├── evaluate.py         # Bootstrap CIs, SHAP, comparison plots
│       ├── to_latex.py         # Multi-model LaTeX table generation
│       └── run_pipeline.py     # Cross-platform pipeline runner
├── tests/
│   └── test_basic.py           # Unit tests
├── paper/
│   ├── main.tex                # LaTeX paper with references
│   ├── results.tex             # Auto-generated comparison table
│   └── figures/                # Auto-generated plots
├── runs/                       # Training artifacts (gitignored)
├── pyproject.toml              # Package configuration
├── requirements.txt            # Dependencies
├── Makefile                    # Unix build automation
└── README.md
```

## Results & Analysis

Evaluated on the **Wisconsin Breast Cancer dataset** (569 patients, 30 features from cell nuclei images, binary malignant/benign target).

### Model Comparison

All five models achieve high discrimination (AUROC > 0.98), but differ substantially in calibration quality:

| Model | AUROC [95% CI] | Brier Score | Calibration |
|-------|---------------|-------------|-------------|
| Logistic Regression | ~0.99 | Low | Naturally well-calibrated |
| Random Forest | ~0.99 | High | Overconfident (S-shaped) |
| RF + Isotonic | ~0.99 | Low | Corrected by isotonic regression |
| RF + Platt | ~0.99 | Low | Corrected by sigmoid scaling |
| XGBoost | ~0.99 | Moderate | Reasonably calibrated |

### Key Finding

The uncalibrated Random Forest pushes predicted probabilities toward 0 or 1 rather than producing calibrated risk estimates. Post-hoc calibration via `CalibratedClassifierCV` substantially improves the Brier score and calibration curve alignment. This confirms that **high AUROC alone is insufficient for clinical deployment** -- clinicians need probabilities they can trust.

### Generated Plots

- **ROC & PR curves** -- multi-model overlay with per-model AUROC/AUPRC in legend
- **Calibration curves** -- multi-model overlay with Brier scores, showing calibration improvement
- **SHAP beeswarm** -- feature importance for best model
- **SHAP bar chart** -- top 15 features by mean |SHAP value|

## Configuration

All experiment settings live in `configs/clinical_demo.yaml`:

- `data.hf_dataset_id` -- Hugging Face dataset ID (`"none"` for sklearn fallback)
- `model.n_estimators` -- number of trees for RF/XGBoost
- `model.max_depth` -- max tree depth (`null` for unlimited)
- `cv.n_folds` -- number of cross-validation folds
- `cv.n_bootstrap` -- number of bootstrap resamples for CIs
- `outputs.*` -- paths for models, results, figures, and LaTeX table

## Requirements

- Python >= 3.9
- Dependencies: numpy, pandas, scikit-learn, matplotlib, pyyaml, datasets, joblib, xgboost, shap
- Optional: pdflatex (for compiling `paper/main.tex`)

## License

MIT

## Citation

```bibtex
@software{Kazi_HealthcareML_2025,
  author = {Kazi, Jibran Rafat Samie},
  title = {Healthcare-ML Pipeline: Calibration-Aware Clinical Risk Modeling},
  year = {2025},
  url = {https://github.com/jibrankazi/Healthcare-ML},
  license = {MIT}
}
```

## Contact

Kazi Jibran Rafat Samie -- Toronto, Canada
