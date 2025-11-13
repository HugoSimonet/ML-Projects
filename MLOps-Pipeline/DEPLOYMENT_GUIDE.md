# MLOps Pipeline Deployment Guide

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Local Deployment](#local-deployment)
3. [Docker Deployment](#docker-deployment)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Configuration](#configuration)
6. [Monitoring Setup](#monitoring-setup)
7. [Security Setup](#security-setup)
8. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements
- Python 3.8+
- Docker 20.10+
- Kubernetes 1.20+ (for K8s deployment)
- 16GB+ RAM recommended
- 50GB+ disk space

### Python Dependencies
```bash
pip install -r requirements.txt
```

### Required Tools
- kubectl (for Kubernetes)
- helm (optional, for package management)
- terraform (optional, for infrastructure provisioning)

## Local Deployment

### 1. Setup Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Pipeline
Edit `configs/pipeline_config.yaml` to match your environment:

```yaml
data:
  ingestion:
    source_path: ./data  # Your data directory

registry:
  registry_dir: ./model_registry  # Model storage

deployment:
  platform: local  # Use 'local' for development
```

### 3. Run Example
```bash
cd examples
python full_pipeline_example.py
```

## Docker Deployment

### 1. Build Docker Image
```bash
# Build the image
docker build -t ml-model-server:v1 .

# Verify image
docker images | grep ml-model-server
```

### 2. Run Container
```bash
# Run the container
docker run -d \
  --name ml-server \
  -p 8000:8000 \
  -e MODEL_NAME=random_forest_classifier \
  -e MODEL_VERSION=v1 \
  -v $(pwd)/model_registry:/app/model_registry \
  ml-model-server:v1

# Check logs
docker logs -f ml-server

# Test API
curl http://localhost:8000/health
```

### 3. Docker Compose (Optional)
Create `docker-compose.yml`:

```yaml
version: '3.8'
services:
  ml-server:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_NAME=random_forest_classifier
      - MODEL_VERSION=v1
    volumes:
      - ./model_registry:/app/model_registry
      - ./logs:/app/logs
```

Run with:
```bash
docker-compose up -d
```

## Kubernetes Deployment

### 1. Create Namespace
```bash
kubectl create namespace mlops
```

### 2. Apply ConfigMaps and Secrets
```bash
kubectl apply -f k8s/configmap.yaml
```

### 3. Deploy Application
```bash
# Apply deployment
kubectl apply -f k8s/deployment.yaml

# Verify deployment
kubectl get deployments -n mlops
kubectl get pods -n mlops

# Check logs
kubectl logs -f deployment/ml-model-server -n mlops
```

### 4. Setup Ingress
```bash
# Install nginx ingress controller (if not installed)
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/cloud/deploy.yaml

# Apply ingress
kubectl apply -f k8s/ingress.yaml

# Get ingress IP
kubectl get ingress -n mlops
```

### 5. Setup Monitoring

#### Prometheus
```bash
# Add Prometheus Helm repo
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Install Prometheus
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace mlops

# Verify
kubectl get pods -n mlops | grep prometheus
```

#### Grafana
Grafana is included with kube-prometheus-stack:
```bash
# Get Grafana password
kubectl get secret -n mlops prometheus-grafana \
  -o jsonpath="{.data.admin-password}" | base64 --decode

# Port forward to access Grafana
kubectl port-forward -n mlops svc/prometheus-grafana 3000:80
```

Access Grafana at http://localhost:3000

### 6. Auto-scaling Setup
```bash
# Install metrics server (if not installed)
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml

# Verify HPA
kubectl get hpa -n mlops

# Test auto-scaling
kubectl run -it --rm load-generator --image=busybox --restart=Never -- /bin/sh
# Inside the pod:
while true; do wget -q -O- http://ml-model-service.mlops.svc.cluster.local/health; done
```

## Configuration

### Environment Variables
Key environment variables for deployment:

| Variable | Description | Default |
|----------|-------------|---------|
| MODEL_NAME | Name of the model | default_model |
| MODEL_VERSION | Version of the model | v1 |
| LOG_LEVEL | Logging level | INFO |
| PROMETHEUS_ENABLED | Enable Prometheus metrics | true |

### Resource Configuration
Edit deployment resource limits in `k8s/deployment.yaml`:

```yaml
resources:
  requests:
    memory: "512Mi"
    cpu: "500m"
  limits:
    memory: "2Gi"
    cpu: "2000m"
```

### Scaling Configuration
Adjust HPA settings:

```yaml
spec:
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        averageUtilization: 70
```

## Monitoring Setup

### 1. Access Prometheus
```bash
kubectl port-forward -n mlops svc/prometheus-kube-prometheus-prometheus 9090:9090
```

Access at http://localhost:9090

### 2. Create Dashboards
Import Grafana dashboards:
1. Go to Grafana UI
2. Click "+" -> Import
3. Use dashboard IDs:
   - 315 (Kubernetes cluster monitoring)
   - 6417 (Kubernetes pod monitoring)

### 3. Setup Alerts
Create alert rules in Prometheus:

```yaml
groups:
- name: ml_model_alerts
  rules:
  - alert: HighErrorRate
    expr: rate(model_errors_total[5m]) > 0.1
    for: 5m
    annotations:
      summary: "High error rate detected"
```

## Security Setup

### 1. RBAC Configuration
Service account and roles are defined in `k8s/configmap.yaml`:

```bash
# Verify RBAC
kubectl get serviceaccount -n mlops
kubectl get role -n mlops
kubectl get rolebinding -n mlops
```

### 2. Network Policies
Apply network policies:

```bash
kubectl apply -f k8s/ingress.yaml
# This includes NetworkPolicy definitions
```

### 3. Secrets Management
Create secrets for sensitive data:

```bash
kubectl create secret generic ml-secrets \
  --from-literal=api-key=your-api-key \
  --from-literal=db-password=your-db-password \
  -n mlops
```

### 4. TLS/SSL Setup
Using cert-manager for automatic TLS:

```bash
# Install cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Create ClusterIssuer
kubectl apply -f - <<EOF
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: your-email@example.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
EOF
```

## Troubleshooting

### Common Issues

#### 1. Pod CrashLoopBackOff
```bash
# Check pod logs
kubectl logs -n mlops <pod-name>

# Describe pod for events
kubectl describe pod -n mlops <pod-name>

# Common causes:
# - Missing ConfigMap
# - Insufficient resources
# - Image pull errors
```

#### 2. High Memory Usage
```bash
# Check resource usage
kubectl top pods -n mlops

# Adjust resource limits in deployment.yaml
```

#### 3. Model Loading Fails
```bash
# Verify PVC is mounted
kubectl describe pod -n mlops <pod-name> | grep Volumes -A 10

# Check model files
kubectl exec -it -n mlops <pod-name> -- ls -la /app/model_registry
```

#### 4. Ingress Not Working
```bash
# Check ingress status
kubectl get ingress -n mlops

# Verify ingress controller
kubectl get pods -n ingress-nginx

# Check ingress logs
kubectl logs -n ingress-nginx <ingress-controller-pod>
```

### Health Checks

#### API Health Check
```bash
# Local
curl http://localhost:8000/health

# Kubernetes
kubectl exec -it -n mlops <pod-name> -- curl localhost:8000/health
```

#### Model Predictions
```bash
# Test prediction endpoint
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"data": {"feature_1": 0.5, "feature_2": 1.2}}'
```

### Debugging Commands

```bash
# Get all resources in namespace
kubectl get all -n mlops

# View events
kubectl get events -n mlops --sort-by='.lastTimestamp'

# Get pod logs
kubectl logs -f -n mlops deployment/ml-model-server

# Execute commands in pod
kubectl exec -it -n mlops <pod-name> -- /bin/bash

# Port forward for debugging
kubectl port-forward -n mlops <pod-name> 8000:8000
```

## Performance Tuning

### 1. Optimize Resource Allocation
Monitor and adjust based on actual usage:
```bash
kubectl top pods -n mlops --containers
```

### 2. Enable Horizontal Pod Autoscaling
Already configured in `k8s/deployment.yaml`

### 3. Use Pod Disruption Budgets
Ensures availability during updates:
```yaml
spec:
  minAvailable: 1
```

### 4. Configure Readiness/Liveness Probes
Tune probe settings based on model loading time:
```yaml
livenessProbe:
  initialDelaySeconds: 30  # Increase for slow-loading models
  periodSeconds: 10
```

## Rollback Strategy

### Rolling Back Deployment
```bash
# View deployment history
kubectl rollout history deployment/ml-model-server -n mlops

# Rollback to previous version
kubectl rollout undo deployment/ml-model-server -n mlops

# Rollback to specific revision
kubectl rollout undo deployment/ml-model-server -n mlops --to-revision=2

# Check rollout status
kubectl rollout status deployment/ml-model-server -n mlops
```

## Maintenance

### Regular Tasks
1. **Monitor Resource Usage**: Daily
2. **Check Logs**: Daily
3. **Review Alerts**: Daily
4. **Update Dependencies**: Weekly
5. **Security Patches**: As needed
6. **Backup Models**: Weekly

### Backup
```bash
# Backup model registry
tar -czf model_registry_backup_$(date +%Y%m%d).tar.gz model_registry/

# Backup configurations
kubectl get configmap -n mlops -o yaml > configmap_backup.yaml
```

## Support

For issues and questions:
- Check logs: `kubectl logs`
- Review documentation
- Open GitHub issue
- Contact MLOps team

## Additional Resources

- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
