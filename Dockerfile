FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src
WORKDIR /app
COPY requirements-serving.txt .
RUN python -m pip install --no-cache-dir -r requirements-serving.txt
COPY configs/ci_demo.yaml configs/ci_demo.yaml
COPY src/hchealth/ src/hchealth/
# Train only on original public dataset *during image build*. This guarantees
# reproducible local saved artifacts without an external model registry.
RUN python -m hchealth.train --config configs/ci_demo.yaml
RUN useradd --uid 10001 --create-home modeluser && chown -R modeluser:modeluser /app/runs
USER modeluser
EXPOSE 8000
CMD ["uvicorn","hchealth.service:app","--host","0.0.0.0","--port","8000","--workers","1"]
