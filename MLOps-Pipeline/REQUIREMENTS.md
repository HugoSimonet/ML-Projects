# Requirements Guide

This project provides multiple requirements files for different use cases. Choose the one that fits your needs.

## Installation Options

### 1. Quick Start (Recommended for Beginners)
```bash
pip install -r requirements-minimal.txt
```

**Includes:**
- Core data processing (numpy, pandas, scipy)
- Machine learning (scikit-learn)
- API server (FastAPI, uvicorn)
- Basic monitoring (prometheus-client, psutil)
- Configuration (pyyaml)

**Use when:**
- You want to try the examples quickly
- You're on a laptop or have limited bandwidth
- You don't need deep learning frameworks

**Size:** ~50MB download

---

### 2. Development Setup
```bash
pip install -r requirements-minimal.txt -r requirements-dev.txt
```

**Additional includes:**
- Testing frameworks (pytest, pytest-cov)
- Code quality tools (black, flake8, mypy, pylint)
- Development tools (ipython, jupyter)
- Documentation tools (sphinx)

**Use when:**
- You're developing or contributing to the project
- You want to run tests
- You need code linting and formatting

**Size:** ~100MB download

---

### 3. Production Deployment
```bash
pip install -r requirements-minimal.txt -r requirements-prod.txt
```

**Additional includes:**
- Kubernetes client
- Docker SDK
- Production web server (gunicorn)
- Advanced monitoring (OpenTelemetry ready)
- Database support (optional, commented out)

**Use when:**
- Deploying to production
- Using Kubernetes orchestration
- Need enterprise monitoring

**Size:** ~150MB download

---

### 4. Full Installation
```bash
pip install -r requirements.txt
```

**Includes:**
- All core dependencies
- All optional features (commented out)
- Deep learning frameworks (commented out)
- MLOps tools (commented out)

**Use when:**
- You want everything available
- You're not concerned about install size
- You may use various optional features

**Size:** ~200MB+ download (depending on what you uncomment)

---

## Requirements Files Breakdown

### requirements-minimal.txt
Core dependencies only. This is what you need to run the basic pipeline.

```
numpy, pandas, scipy          # Data processing
scikit-learn                  # Machine learning
fastapi, uvicorn             # API server
prometheus-client, psutil    # Monitoring
pyyaml                       # Configuration
```

### requirements-dev.txt
Development and testing tools. Combine with minimal for development.

```
pytest*                      # Testing framework
black, flake8, mypy         # Code quality
jupyter, notebook           # Interactive development
sphinx                      # Documentation
```

### requirements-prod.txt
Production deployment tools. Combine with minimal for production.

```
kubernetes, docker          # Container orchestration
gunicorn                   # Production WSGI server
python-json-logger         # Structured logging
opentelemetry (optional)   # Distributed tracing
```

### requirements.txt
Complete list with all optional dependencies commented out.

```
All of the above plus:
- Deep learning frameworks (PyTorch, TensorFlow)
- MLOps tools (MLflow, Weights & Biases)
- Database adapters (SQLAlchemy, Redis)
- Advanced monitoring (Jaeger, Grafana)
- And more...
```

---

## Deep Learning Frameworks

If you're working with deep learning models, uncomment the relevant section in `requirements.txt`:

### For PyTorch:
```bash
pip install torch>=1.9.0 torchvision>=0.10.0
```

### For TensorFlow:
```bash
pip install tensorflow>=2.8.0
```

### For Transformers (Hugging Face):
```bash
pip install transformers>=4.20.0 datasets>=2.0.0
```

---

## Optional MLOps Tools

### MLflow (Experiment Tracking)
```bash
pip install mlflow>=2.0.0
```

### Weights & Biases
```bash
pip install wandb>=0.12.0
```

### DVC (Data Version Control)
```bash
pip install dvc>=2.10.0
```

---

## Database Support

### PostgreSQL
```bash
pip install sqlalchemy>=1.4.0 psycopg2-binary>=2.9.0
```

### Redis
```bash
pip install redis>=4.3.0
```

---

## GPU Support

For GPU monitoring and metrics:
```bash
pip install GPUtil>=1.4.0 nvidia-ml-py3>=7.352.0
```

---

## Common Installation Scenarios

### Scenario 1: Data Scientist on Laptop
**Goal:** Try examples, experiment with models
```bash
pip install -r requirements-minimal.txt
# Add scikit-learn models, run examples
```

### Scenario 2: ML Engineer Developing
**Goal:** Develop features, run tests, ensure code quality
```bash
pip install -r requirements-minimal.txt -r requirements-dev.txt
# Write code, run tests, commit with pre-commit hooks
```

### Scenario 3: DevOps Engineer Deploying
**Goal:** Deploy to Kubernetes, set up monitoring
```bash
pip install -r requirements-minimal.txt -r requirements-prod.txt
# Deploy to K8s, configure monitoring
```

### Scenario 4: Full Stack ML Engineer
**Goal:** End-to-end pipeline with all features
```bash
pip install -r requirements.txt
# Uncomment needed features in requirements.txt first
```

---

## Troubleshooting

### Issue: Installation takes too long
**Solution:** Use requirements-minimal.txt instead:
```bash
pip install -r requirements-minimal.txt
```

### Issue: Kubernetes client fails to install
**Solution:** Skip it if not deploying to K8s:
```bash
pip install -r requirements-minimal.txt
# Skip requirements-prod.txt
```

### Issue: PyTorch too large
**Solution:** Install CPU-only version:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Issue: Version conflicts
**Solution:** Use a fresh virtual environment:
```bash
python -m venv fresh_env
source fresh_env/bin/activate  # Windows: fresh_env\Scripts\activate
pip install -r requirements-minimal.txt
```

---

## Version Constraints

All requirements use version ranges for compatibility:
- `>=X.Y.0`: Minimum version
- `<Z.0.0`: Maximum major version (prevents breaking changes)

Example: `numpy>=1.21.0,<2.0.0`
- Minimum: 1.21.0
- Maximum: Any 1.x version
- Blocked: 2.0.0+ (to prevent breaking changes)

---

## Updating Dependencies

To update to latest compatible versions:
```bash
pip install --upgrade -r requirements-minimal.txt
```

To check for outdated packages:
```bash
pip list --outdated
```

---

## Docker Considerations

The Dockerfile uses requirements.txt by default. To optimize Docker builds:

1. Copy only minimal requirements:
```dockerfile
COPY requirements-minimal.txt .
RUN pip install -r requirements-minimal.txt
```

2. Add production requirements:
```dockerfile
COPY requirements-prod.txt .
RUN pip install -r requirements-prod.txt
```

3. Add your model-specific requirements last:
```dockerfile
COPY requirements-model.txt .
RUN pip install -r requirements-model.txt
```

This creates better Docker layer caching.

---

## Summary

| File | Size | Use Case | Install Time |
|------|------|----------|--------------|
| requirements-minimal.txt | ~50MB | Quick start, examples | ~2 min |
| requirements-dev.txt | ~50MB | Development | +1 min |
| requirements-prod.txt | ~100MB | Production | +3 min |
| requirements.txt | ~200MB+ | Full features | ~10 min+ |

**Recommendation:** Start with `requirements-minimal.txt` and add others as needed!
