# Healthcare-ML Pipeline

Clinical risk modeling with AUROC/AUPRC evaluation, calibration analysis, and automated LaTeX reporting.

## Overview

End-to-end, API-driven research pipeline for clinical risk prediction using tabular healthcare data.
Demonstrates reproducible ML methodology, calibration analysis, and automated LaTeX sync.

Developed for University of Toronto PhD in Computer Science (Fall 2026) portfolio.

## Features

- **API integration** -- loads tabular datasets via Hugging Face Datasets (falls back to sklearn breast cancer)
- **Baseline modeling** -- Random Forest classifier with StandardScaler preprocessing
- **Evaluation** -- AUROC, AUPRC, and calibration curves
- **Visualization** -- ROC, PR, and calibration plots (matplotlib)
- **LaTeX sync** -- metrics and figures auto-exported into `paper/results.tex`

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

# Run full pipeline
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
│       ├── train.py            # Model training
│       ├── evaluate.py         # Evaluation & plotting
│       ├── to_latex.py         # LaTeX table generation
│       └── run_pipeline.py     # Cross-platform pipeline runner
├── tests/
│   └── test_basic.py           # Unit tests
├── paper/
│   ├── main.tex                # LaTeX document
│   ├── results.tex             # Auto-generated metrics table
│   └── figures/                # Auto-generated plots
├── runs/                       # Training artifacts (gitignored)
├── pyproject.toml              # Package configuration
├── requirements.txt            # Dependencies
├── Makefile                    # Unix build automation
└── README.md
```

## Results & Analysis

The current demo runs on the **Wisconsin Breast Cancer dataset** (569 patients, 30 features from cell nuclei images, binary malignant/benign target).

| Metric | Value | Meaning |
|--------|-------|---------|
| AUROC | 0.999 | Near-perfect ranking of malignant vs. benign |
| AUPRC | 0.999 | High precision across all recall thresholds |

**ROC curve** -- achieves ~97% true positive rate at essentially 0% false positive rate. The model correctly identifies nearly all malignant tumors without mislabeling benign ones.

**PR curve** -- precision stays at 1.0 across nearly the full recall range. When the model predicts malignant, it is almost always correct.

**Calibration curve** -- the most clinically relevant finding. The S-shaped deviation from the diagonal shows the Random Forest is **overconfident**: it pushes predicted probabilities toward 0 or 1 rather than producing calibrated risk estimates. A raw prediction of 0.7 does not mean "70% chance of malignancy." This motivates the next phase of work: integrating probability calibration (isotonic/Platt scaling via `CalibratedClassifierCV`) so predicted probabilities can be trusted for patient-facing risk communication.

## Configuration

All experiment settings live in `configs/clinical_demo.yaml`:

- `data.hf_dataset_id` -- Hugging Face dataset ID (`"none"` for sklearn fallback)
- `model.n_estimators` -- number of Random Forest trees
- `model.max_depth` -- max tree depth (`null` for unlimited)
- `outputs.*` -- paths for model, results, figures, and LaTeX table

## Requirements

- Python >= 3.9
- Dependencies: numpy, pandas, scikit-learn, matplotlib, pyyaml, datasets, joblib
- Optional: pdflatex (for compiling `paper/main.tex`)

## License

MIT

## Citation

```bibtex
@software{Kazi_HealthcareML_2025,
  author = {Kazi, Jibran Rafat Samie},
  title = {Healthcare-ML Pipeline: Clinical Risk Modeling with API Data and LaTeX Sync},
  year = {2025},
  url = {https://github.com/jibrankazi/healthcare-ml-pipeline},
  license = {MIT}
}
```

## Contact

Kazi Jibran Rafat Samie -- Toronto, Canada
