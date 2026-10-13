# Reproducible training → Docker inference → aggregate monitoring

**Scope:** educational / research-only Wisconsin Breast Cancer dataset from
scikit-learn (569 historic measurements, 30 numerical inputs). This is
**not** a clinically validated diagnosis, health-risk estimate, patient
decision-support device, or live hospital integration.

## Architecture

\`\`\`mermaid
flowchart TD
    A[scikit-learn original Wisconsin research dataset] --> B[Fixed 80/20 stratified split]
    B --> C[Training data only]
    B --> D[Untouched 114-row research holdout]
    C --> E[5 estimators and 2-fold CI-demo cross-validation]
    C --> F[Training 1st/99th percentile feature reference]
    E --> G[Best selected by training CV AUROC]
    G --> H[Trusted local serialized model]
    H --> I[FastAPI container]
    F --> I
    I --> J[POST /predict with exact 30-feature contract]
    I --> K[GET /metrics: counts, score bins, envelope rate]
    I --> L[GET /schema and /health]
    D --> M[Independent heldout evaluation, never selection]
\`\`\`

The full existing pipeline also evaluates AUROC, AUPRC, Brier, calibration
and SHAP on public research records. For the **fast example CI configuration**,
the best model is chosen by *training cross-validation AUROC*, never by
peeking at heldout test score.

Monitoring is deliberately **aggregate-only in memory**, and represents
an engineering demonstration rather than formal statistical drift monitoring:
\`requests\`, \`observations\`, positive class score histogram/average,
and rate of values beyond a reference range learned **only from training
features**. It does not log individual measurements, store request rows, infer
demographics, promise clinical safety, or establish accuracy after deployment.

## Build and run the actual trained model in Docker

\`\`\`bash
git clone https://github.com/jibrankazi/Healthcare-ML.git
cd Healthcare-ML
docker build -t wisconsin-research-api .
docker run --rm -p 127.0.0.1:8002:8000 wisconsin-research-api
\`\`\`

The Docker image **trains all five research estimators** from the explicitly
chosen public Wisconsin dataset **at image build**, saves the models and
\`monitoring_reference.json\`, and starts an unprivileged user running
FastAPI. This is not a hosted web deployment; bind to **127.0.0.1** only.
See http://localhost:8002/docs for interactive API routes.

Inspect the exact original schema:

\`\`\`bash
curl http://127.0.0.1:8002/health
curl http://127.0.0.1:8002/schema
curl http://127.0.0.1:8002/metrics
\`\`\`

Example request from genuine public records (never patient data):

\`\`\`bash
python - <<'PY'
import json
from pathlib import Path
from sklearn.datasets import load_breast_cancer
from urllib.request import Request, urlopen

x = load_breast_cancer(as_frame=True).data.iloc[:2].to_dict(orient="records")
request = Request(
    "http://127.0.0.1:8002/predict",
    data=json.dumps({"records": x}).encode("utf-8"),
    headers={"Content-Type": "application/json"}, method="POST",
)
with urlopen(request, timeout=30) as response:
    print(response.read().decode())
with urlopen("http://127.0.0.1:8002/metrics", timeout=5) as response:
    print(response.read().decode())
PY
\`\`\`

**Do not expose this unauthenticated medical-data-shaped API to the internet.**
It is a local demonstration; it lacks end-user authentication, audit
authorization, persistent metrics, privacy and retention controls,
systematic data-shift statistical evaluation and external clinical validation.
The probability for class \`1\` is a model output associated with
\`malignant\` **in this historical research dataset**; it is not a clinical
estimate for anyone.

## CI and evidence

The [separate training-to-container workflow](https://github.com/jibrankazi/Healthcare-ML/actions/workflows/inference-container.yml)
does all of the following: run HTTP contract tests against fixture-trained
models (not fabricated outcome metrics), train on genuine public records,
assert the training-only feature envelope, score 12 untouched original test
records with the serialized model, build the Docker image, launch it,
POST original public research measurements and verify \`GET /metrics\`
independently. Training metadata and feature reference are archived as
run artifacts.

The full model-quality lifecycle is checked independently by
[the original train/heldout/SHAP/LaTeX workflow](https://github.com/jibrankazi/Healthcare-ML/actions/workflows/tests.yml).

The [container workflow](https://github.com/jibrankazi/Healthcare-ML/actions/workflows/inference-container.yml)
also opens the **actual running FastAPI Swagger UI** at
`http://127.0.0.1:8002/docs` in headless Chrome when a browser is
available, and uploads `docs/screenshots/real-wisconsin-fastapi-docs.png`
in the `genuine-wisconsin-train-serve-monitor-contract` Actions artifact.
This is a real local browser capture, **not a publicly deployed clinical
dashboard**. The screenshot is a UI demonstration, not model evaluation
evidence.

## Local development

\`\`\`bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-serving.txt pytest
export PYTHONPATH=src
python -m pytest -q tests/test_service.py
python -m hchealth.train --config configs/ci_demo.yaml
python -m uvicorn hchealth.service:app --host 127.0.0.1 --port 8002
\`\`\`

Model artifact files are trusted local training outputs. \`joblib\` is
pickle-based; do not load untrusted uploads.
