# MLOps Pipeline - Quick Start Guide

Get started with the MLOps Pipeline in 5 minutes!

## Quick Installation

### 1. Clone and Setup
```bash
cd MLOps-Pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Option A: Minimal installation (fastest, for quick start)
pip install -r requirements-minimal.txt

# Option B: Full installation (includes all optional dependencies)
pip install -r requirements.txt

# Option C: Development installation (includes testing and linting tools)
pip install -r requirements-minimal.txt -r requirements-dev.txt

# Option D: Production installation (includes Kubernetes and monitoring)
pip install -r requirements-minimal.txt -r requirements-prod.txt
```

### Installation Options Explained:

- **requirements-minimal.txt**: Core dependencies only (~50MB download)
  - Perfect for trying out the examples
  - Includes: pandas, scikit-learn, FastAPI, basic monitoring

- **requirements.txt**: All dependencies with optional features
  - Includes everything for full functionality
  - Use this for production deployments

- **requirements-dev.txt**: Development tools
  - Testing frameworks (pytest)
  - Code quality tools (black, flake8, mypy)
  - Jupyter notebooks

- **requirements-prod.txt**: Production extras
  - Kubernetes client
  - Docker integration
  - Advanced monitoring tools

### 2. Run Example
```bash
cd examples
python full_pipeline_example.py
```

That's it! The pipeline will:
1. Ingest and validate data
2. Engineer features
3. Train a model
4. Register the model
5. Deploy it
6. Monitor performance
7. Check for drift

## What Just Happened?

The example pipeline:
- ✅ Created a sample dataset
- ✅ Validated data quality
- ✅ Transformed features
- ✅ Trained a Random Forest model
- ✅ Registered the model with version control
- ✅ Simulated a deployment
- ✅ Monitored predictions
- ✅ Checked for data drift
- ✅ Setup A/B testing
- ✅ Generated audit logs

## Project Structure

```
MLOps-Pipeline/
├── data/                   # Data pipeline components
│   ├── data_ingestion.py
│   ├── data_validation.py
│   ├── feature_engineering.py
│   └── data_versioning.py
├── registry/               # Model registry
│   └── model_registry.py
├── deployment/             # Model deployment
│   ├── model_serving.py
│   └── deployment_manager.py
├── monitoring/             # Monitoring & drift detection
│   ├── performance_monitoring.py
│   └── drift_detection.py
├── testing/                # A/B testing
│   └── ab_testing.py
├── infrastructure/         # Kubernetes management
│   └── kubernetes.py
├── security/               # Security & compliance
│   └── security_manager.py
├── pipeline/               # Main orchestration
│   └── mlops_pipeline.py
├── examples/               # Usage examples
│   ├── full_pipeline_example.py
│   └── simple_deployment.py
├── configs/                # Configuration
│   └── pipeline_config.yaml
└── k8s/                    # Kubernetes manifests
    ├── deployment.yaml
    ├── configmap.yaml
    └── ingress.yaml
```

## Key Features

### 🔄 Complete ML Lifecycle
- Data ingestion and validation
- Feature engineering and versioning
- Model training and registration
- Automated deployment
- Continuous monitoring

### 📊 Model Management
- Version control for models and data
- Model registry with metadata
- Stage transitions (dev → staging → production)
- Model comparison and rollback

### 🚀 Deployment Strategies
- Blue-green deployment
- Canary deployment
- Rolling updates
- Shadow deployment

### 📈 Monitoring & Observability
- Performance metrics tracking
- Data drift detection
- Model drift detection
- Prometheus integration
- Custom alerts

### 🧪 A/B Testing
- Multi-variant testing
- Statistical significance testing
- Traffic splitting
- Automated winner selection

### 🔒 Security & Compliance
- Role-based access control (RBAC)
- Audit logging
- Data encryption
- PII detection and anonymization

### ☸️ Cloud-Native
- Kubernetes deployment
- Horizontal pod autoscaling
- Service mesh ready
- Ingress configuration

## Quick Examples

### Example 1: Simple Model Deployment
```python
from pipeline import MLOpsPipeline
from sklearn.ensemble import RandomForestClassifier

# Initialize
pipeline = MLOpsPipeline(config)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Register
version = pipeline.train_and_register_model(
    model=model,
    model_name="my_model",
    data_version="v1",
    metrics={'accuracy': 0.95}
)

# Deploy
deployment = pipeline.deploy_model(
    model_name="my_model",
    model_version="v1",
    strategy="blue_green"
)
```

### Example 2: Monitor Deployment
```python
# Monitor performance
results = pipeline.monitor_deployment(
    deployment_id=deployment['deployment_id'],
    check_drift=True
)

print(f"Accuracy: {results['performance']['accuracy']}")
print(f"Latency: {results['performance']['avg_latency_ms']}ms")
```

### Example 3: A/B Testing
```python
# Setup A/B test
test_config = {
    'control_model': 'v1',
    'treatment_model': 'v2',
    'success_metric': 'accuracy',
    'traffic_split': 0.5
}

results = pipeline.run_ab_test(test_config)
```

## Configuration

Edit `configs/pipeline_config.yaml` to customize:

```yaml
# Data configuration
data:
  ingestion:
    source_type: local
    source_path: ./data

# Model registry
registry:
  registry_dir: ./model_registry

# Deployment
deployment:
  platform: local  # or 'kubernetes'
  replicas: 3

# Monitoring
monitoring:
  alert_thresholds:
    error_rate: 0.1
    latency_ms: 1000
```

## Docker Deployment

```bash
# Build image
docker build -t ml-model:v1 .

# Run container
docker run -p 8000:8000 ml-model:v1

# Test API
curl http://localhost:8000/health
```

## Kubernetes Deployment

```bash
# Create namespace
kubectl create namespace mlops

# Deploy
kubectl apply -f k8s/

# Check status
kubectl get pods -n mlops
```

## API Usage

### Health Check
```bash
curl http://localhost:8000/health
```

### Make Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "feature_1": 0.5,
      "feature_2": 1.2
    }
  }'
```

### Get Model Info
```bash
curl http://localhost:8000/model/info
```

## Monitoring Dashboards

### Prometheus Metrics
Access at: http://localhost:9090

Available metrics:
- `model_predictions_total`
- `model_prediction_latency_seconds`
- `model_accuracy`
- `model_error_rate`

### Grafana Dashboards
Access at: http://localhost:3000

Pre-built dashboards:
- Model Performance
- System Metrics
- Drift Detection
- A/B Test Results

## Next Steps

1. **Customize Configuration**: Edit `configs/pipeline_config.yaml`
2. **Add Your Models**: Integrate your own ML models
3. **Setup Monitoring**: Configure Prometheus and Grafana
4. **Deploy to Production**: Use Kubernetes for production deployment
5. **Enable Security**: Configure RBAC and audit logging

## Common Commands

```bash
# List registered models
python -c "from registry import ModelRegistry; r = ModelRegistry({}); print(r.list_models())"

# Check deployment status
kubectl get deployments -n mlops

# View logs
kubectl logs -f deployment/ml-model-server -n mlops

# Scale deployment
kubectl scale deployment ml-model-server --replicas=5 -n mlops

# Rollback deployment
kubectl rollout undo deployment/ml-model-server -n mlops
```

## Troubleshooting

### Issue: Import errors
```bash
# Solution: Make sure you're in the right directory
cd MLOps-Pipeline
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Issue: Docker build fails
```bash
# Solution: Check Docker is running
docker --version
docker ps
```

### Issue: Kubernetes deployment fails
```bash
# Solution: Check cluster connection
kubectl cluster-info
kubectl get nodes
```

## Resources

- 📖 [Full Documentation](README.md)
- 🚀 [Deployment Guide](DEPLOYMENT_GUIDE.md)
- 💻 [Examples](examples/)
- 🐛 [GitHub Issues](https://github.com/yourusername/mlops-pipeline/issues)

## Support

Need help?
- Check the [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
- Review example scripts in `examples/`
- Open an issue on GitHub

---

**Happy MLOps! 🚀**
