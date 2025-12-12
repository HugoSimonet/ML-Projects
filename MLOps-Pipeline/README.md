# MLOps Pipeline

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![Docker](https://img.shields.io/badge/Docker-enabled-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

End-to-end ML pipeline for model training, deployment, monitoring, and A/B testing with Docker and Kubernetes support.

## Overview

This project implements an MLOps pipeline covering the complete ML lifecycle. It includes data processing, model training, versioning, deployment via REST API, monitoring with Prometheus/Grafana, and CI/CD integration.

## Features

- Data pipeline with validation and versioning
- Experiment tracking with MLflow
- Model registry and versioning
- Docker containerization
- Kubernetes deployment
- REST API for model serving
- Monitoring and drift detection
- A/B testing framework
- CI/CD integration

## Architecture

**Data Pipeline** - Ingestion, validation, feature engineering, versioning with DVC

**Training Pipeline** - Experiment tracking, hyperparameter tuning, model evaluation, checkpointing

**Model Registry** - Model versioning, metadata storage, artifact management

**Deployment** - Docker containers, Kubernetes orchestration, load balancing, blue-green/canary deployments

**Monitoring** - Performance metrics, data drift detection, logging, alerting

**API Server** - REST endpoints, request validation, model inference, health checks

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Initialize DVC
dvc init

# Pull data (if configured)
dvc pull
```

Requirements: Python 3.8+, Docker, Kubernetes (optional), MLflow, DVC

## Quick Start

### Train Model

```bash
python train.py \
    --data data/processed \
    --model xgboost \
    --experiment-name my_experiment \
    --tracking-uri http://localhost:5000
```

### Start API Server

```bash
# Local development
python api/serve.py --model-path models/best_model.pkl --port 8000

# Docker
docker build -t ml-api:latest .
docker run -p 8000:8000 ml-api:latest

# Kubernetes
kubectl apply -f k8s/deployment.yaml
```

### Make Predictions

```bash
curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"features": [1.0, 2.0, 3.0, 4.0]}'
```

## Project Structure

```
MLOps-Pipeline/
├── data/                # Data storage
├── models/              # Trained models
├── api/                 # REST API server
├── pipelines/           # Training and inference pipelines
├── monitoring/          # Monitoring and drift detection
├── tests/               # Unit and integration tests
├── k8s/                 # Kubernetes manifests
├── docker/              # Dockerfiles
├── configs/             # Configuration files
└── train.py             # Main training script
```

## Data Pipeline

```python
from pipelines import DataPipeline

pipeline = DataPipeline(config_path='configs/data.yaml')
pipeline.ingest(source='s3://bucket/data')
pipeline.validate()
pipeline.transform()
pipeline.split(train=0.7, val=0.15, test=0.15)
```

## Model Training

```python
from pipelines import TrainingPipeline
import mlflow

with mlflow.start_run(experiment_name='my_experiment'):
    pipeline = TrainingPipeline(config_path='configs/training.yaml')
    model = pipeline.train()
    metrics = pipeline.evaluate(model)
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(model, 'model')
```

## Deployment

### Docker

```dockerfile
FROM python:3.9-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . /app
WORKDIR /app
CMD ["python", "api/serve.py"]
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-api
  template:
    metadata:
      labels:
        app: ml-api
    spec:
      containers:
      - name: api
        image: ml-api:latest
        ports:
        - containerPort: 8000
```

## Monitoring

```python
from monitoring import ModelMonitor

monitor = ModelMonitor(
    model_path='models/production',
    metrics=['accuracy', 'latency', 'drift']
)

# Log predictions
monitor.log_prediction(features, prediction, ground_truth)

# Check for drift
drift_detected = monitor.check_drift(
    reference_data='data/train',
    current_data='data/recent'
)
```

## A/B Testing

```python
from deployment import ABTest

ab_test = ABTest(
    model_a='models/v1',
    model_b='models/v2',
    traffic_split=0.5
)

ab_test.start()
results = ab_test.get_results()
```

## CI/CD

GitHub Actions workflow for automated testing and deployment:

```yaml
name: MLOps Pipeline
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run tests
        run: pytest tests/
      - name: Build Docker
        run: docker build -t ml-api:${{ github.sha }} .
      - name: Deploy
        run: kubectl set image deployment/ml-api api=ml-api:${{ github.sha }}
```

## Configuration

```yaml
data:
  source: s3://bucket/data
  format: parquet
  validation:
    schema: schemas/data.json
    quality_checks: true

training:
  model: xgboost
  hyperparameters:
    max_depth: 6
    learning_rate: 0.1
  tracking:
    experiment_name: production
    tracking_uri: http://mlflow:5000

deployment:
  replicas: 3
  resources:
    cpu: "500m"
    memory: "1Gi"
  autoscaling:
    min_replicas: 2
    max_replicas: 10
    target_cpu: 70

monitoring:
  metrics:
    - accuracy
    - latency
    - drift
  alerting:
    email: alerts@example.com
    threshold:
      accuracy: 0.9
      latency: 100ms
```

## Testing

```bash
# Unit tests
pytest tests/unit/

# Integration tests
pytest tests/integration/

# End-to-end tests
pytest tests/e2e/
```

## Metrics

**Model Performance**: Accuracy, precision, recall, F1, AUC
**System Performance**: Latency, throughput, error rate
**Data Quality**: Drift detection, schema validation
**Business Impact**: Conversion rate, revenue impact

## Implementation Notes

Uses MLflow for experiment tracking and model registry. DVC for data versioning. Docker for containerization. Kubernetes for orchestration. Prometheus for metrics collection. Grafana for visualization.

API built with FastAPI for high performance and automatic OpenAPI documentation. Health checks at `/health` and `/ready` endpoints. Request validation using Pydantic models.

Model serving uses caching for frequently requested predictions. Batch prediction endpoint available for bulk inference.

## References

- MLflow: Model Management and Experiment Tracking
- DVC: Data Version Control
- Kubernetes: Container Orchestration
- Prometheus: Monitoring and Alerting
- FastAPI: Modern Python Web Framework

## License

MIT License - see LICENSE file for details.
