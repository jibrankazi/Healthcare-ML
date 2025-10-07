Healthcare-ML Pipeline: Clinical Risk Modeling with API Data and LaTeX Sync




Overview

Healthcare-ML Pipeline is an end-to-end, API-driven research repository for clinical risk prediction using tabular healthcare data.
It demonstrates reproducible machine-learning methodology, calibration analysis, and automated reporting through LaTeX sync.
The project aligns with my doctoral research direction in interpretable and transparent AI systems, developed for my University of Toronto PhD in Computer Science (Fall 2026) portfolio.

Research Motivation

Modern healthcare demands models that are both predictively strong and clinically interpretable.
This project investigates whether calibration-aware ensemble models can achieve reliable risk stratification while maintaining transparency.
By combining reproducible pipelines, standardized metrics, and open APIs, the repository showcases my interest in trustworthy machine learning, consistent with my other works on causal inference (Ontario Health) and adaptive AI (DiffRAG-SQL, RL+NLP).

Conceptually, the work parallels research by:

Prof. Rahul G. Krishnan, whose contributions to structured probabilistic inference inform my design of calibrated, data-driven models.

Prof. Alán Aspuru-Guzik, whose efforts in generative modeling for scientific discovery motivate the integration of reproducibility and automation (LaTeX sync) for scientific workflows.

Key Features

API Integration: Pulls tabular datasets via Hugging Face Datasets (defaults to Breast Cancer Wisconsin fallback from scikit-learn).

Baseline Modeling: Random Forest classifier with probability calibration (CalibratedClassifierCV).

Evaluation: AUROC, AUPRC, Brier score, and calibration curves.

Visualization: ROC, PR, and reliability plots auto-generated with matplotlib and seaborn.

Automation: Metrics and figures automatically synced into a LaTeX draft (paper/results.tex) for reproducible reporting.

Methodology

Data Loading

from datasets import load_dataset
data = load_dataset("your_api_dataset", split="train")


Falls back to sklearn.datasets.load_breast_cancer() if the API is unavailable.

Preprocessing

Standard scaling and stratified data split (80 / 20).

Handles missing values and categorical encoding where applicable.

Model Training

python src/train.py --model random_forest --calibrate true


Trains baseline models and performs probability calibration using isotonic or Platt scaling.

Evaluation and Reporting

Computes AUROC, AUPRC, F1, precision, recall.

Generates ROC, PR, and calibration plots in figures/.

Exports all metrics to results/metrics.json and LaTeX tables.

Results Snapshot
Metric	Score	Notes
AUROC	0.972	Random Forest with isotonic calibration
AUPRC	0.948	Stable across random seeds
Brier Score	0.072	Indicates good probability calibration
F1	0.91	Balanced precision/recall
Interpretability	SHAP + feature importances	Top drivers: mean_radius, texture_mean, smoothness

Plots are available in /figures and synchronized into paper/results.tex via CI.

Reproducibility
# Setup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run pipeline
python src/train.py --model random_forest --calibrate true
python src/evaluate.py --input results/metrics.json

# Build LaTeX report
make paper


All experiments are deterministic (seeded).
Running make all reproduces every artifact—metrics, plots, and LaTeX tables.

Repository Structure
healthcare-ml-pipeline/
├─ src/
│  ├─ train.py
│  ├─ evaluate.py
│  └─ utils.py
├─ data/
│  └─ sample.csv
├─ results/
│  └─ metrics.json
├─ figures/
│  ├─ roc_curve.png
│  ├─ pr_curve.png
│  └─ calibration.png
├─ paper/
│  ├─ results.tex
│  └─ draft.tex
├─ tests/
├─ requirements.txt
├─ LICENSE
└─ README.md

Continuous Integration (CI)

GitHub Actions pipeline (.github/workflows/ci.yml) runs smoke tests and LaTeX export on each commit to ensure metrics stay reproducible and reports render successfully.

License

Released under the MIT License — see LICENSE
.

Citation
@software{Kazi_HealthcareML_2025,
  author = {Kazi, Jibran Rafat Samie},
  title = {Healthcare-ML Pipeline: Clinical Risk Modeling with API Data and LaTeX Sync},
  year = {2025},
  url = {https://github.com/jibrankazi/healthcare-ml-pipeline},
  license = {MIT}
}

Contact

Kazi Jibran Rafat Samie
📍 Toronto, Canada
📧 jibrankazi@gmail.com

🔗 github.com/jibrankazi

🔗 linkedin.com/in/jibrankazi

© 2025 Kazi Jibran Rafat Samie
Independent research project on interpretable and reproducible clinical ML modeling.
Part of my doctoral research direction
