# Docker Quick Start Guide

## Prerequisites

- Docker installed (version 20.10+)
- Docker Compose installed (version 2.0+)
- 4GB RAM minimum
- 10GB disk space

## Quick Start (Single Command)

```bash
# Start the entire stack
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f api
```

That's it! Your MLOps platform is now running.

---

## Access Points

Once running, access these services:

| Service | URL | Credentials |
|---------|-----|-------------|
| ML API | http://localhost:8000 | None (public) |
| API Docs | http://localhost:8000/docs | None |
| Grafana | http://localhost:3000 | admin/admin |
| Prometheus | http://localhost:9090 | None |
| PostgreSQL | localhost:5432 | mlops/mlops_password |
| Redis | localhost:6379 | None |

---

## Step-by-Step Setup

### 1. Build the Docker Image

```bash
cd MLOps-Pipeline

# Build the image
docker build -t mlops-pipeline:v1 .

# Check image size
docker images mlops-pipeline
```

**Expected output**:
```
REPOSITORY         TAG       SIZE
mlops-pipeline     v1        ~800MB
```

### 2. Run Single Container (API only)

```bash
# Run API container
docker run -d \
  --name mlops-api \
  -p 8000:8000 \
  -v $(pwd)/model_registry:/app/model_registry \
  -v $(pwd)/data_versions:/app/data_versions \
  mlops-pipeline:v1

# Check logs
docker logs -f mlops-api

# Test API
curl http://localhost:8000/health
```

### 3. Run Full Stack with Docker Compose

```bash
# Start all services
docker-compose up -d

# Check all services are running
docker-compose ps

# Expected output:
# NAME                  STATUS    PORTS
# mlops-api             Up        0.0.0.0:8000->8000/tcp
# mlops-postgres        Up        0.0.0.0:5432->5432/tcp
# mlops-redis           Up        0.0.0.0:6379->6379/tcp
# mlops-prometheus      Up        0.0.0.0:9090->9090/tcp
# mlops-grafana         Up        0.0.0.0:3000->3000/tcp
# mlops-node-exporter   Up        0.0.0.0:9100->9100/tcp
```

### 4. Verify Everything Works

```bash
# Test API health
curl http://localhost:8000/health

# Test API docs (in browser)
open http://localhost:8000/docs

# Test Prometheus (in browser)
open http://localhost:9090

# Test Grafana (in browser)
open http://localhost:3000
```

---

## Testing the API

### Health Check
```bash
curl http://localhost:8000/health
```

**Expected response**:
```json
{
  "status": "healthy",
  "version": "v1",
  "timestamp": "2025-11-12T10:00:00Z"
}
```

### Predict Endpoint
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "feature_1": 0.5,
      "feature_2": 1.2,
      "feature_3": -0.8
    }
  }'
```

**Expected response**:
```json
{
  "prediction": 1,
  "model_version": "v1",
  "timestamp": "2025-11-12T10:00:00Z",
  "latency_ms": 15.3
}
```

### Metrics Endpoint
```bash
curl http://localhost:8000/metrics
```

**Expected response**: Prometheus-formatted metrics

---

## Viewing Logs

### View all logs
```bash
docker-compose logs
```

### Follow specific service logs
```bash
# API logs
docker-compose logs -f api

# Database logs
docker-compose logs -f postgres

# Prometheus logs
docker-compose logs -f prometheus
```

### Last 100 lines
```bash
docker-compose logs --tail=100 api
```

---

## Database Access

### Connect to PostgreSQL

```bash
# Using docker exec
docker-compose exec postgres psql -U mlops -d mlops_db

# Using psql client locally
psql -h localhost -U mlops -d mlops_db
# Password: mlops_password
```

### Common SQL queries
```sql
-- Check tables
\dt

-- View models
SELECT name, version, stage, created_at FROM models;

-- View deployments
SELECT * FROM active_deployments;

-- View recent predictions
SELECT model_version, COUNT(*), AVG(latency_ms)
FROM predictions
WHERE created_at > NOW() - INTERVAL '1 hour'
GROUP BY model_version;

-- View drift detections
SELECT * FROM drift_detections ORDER BY detected_at DESC LIMIT 10;
```

---

## Grafana Setup

### 1. First Login

1. Go to http://localhost:3000
2. Login: `admin` / `admin`
3. (Optional) Change password

### 2. Verify Datasource

1. Settings → Data Sources
2. You should see "Prometheus" already configured
3. Click "Test" to verify connection

### 3. Import Dashboards

```bash
# Import pre-built dashboard (if available)
# Or create custom dashboards for:
# - API Performance
# - Model Performance
# - Infrastructure Metrics
# - Business Metrics
```

### 4. Create Your First Dashboard

1. Click "+" → Dashboard
2. Add Panel
3. Query: `rate(predictions_total[5m])`
4. Title: "Predictions per Second"
5. Save Dashboard

---

## Prometheus Setup

### 1. Access Prometheus

Go to http://localhost:9090

### 2. Check Targets

1. Status → Targets
2. Verify all targets are "UP":
   - mlops-api
   - prometheus
   - node-exporter

### 3. Run Sample Queries

```promql
# Request rate
rate(http_requests_total[5m])

# Latency P95
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))

# Error rate
rate(http_requests_total{status=~"5.."}[5m]) /
rate(http_requests_total[5m])

# Memory usage
process_resident_memory_bytes / 1024 / 1024
```

---

## Common Operations

### Stop All Services
```bash
docker-compose down
```

### Stop and Remove Volumes (Clean Slate)
```bash
docker-compose down -v
```

### Restart Single Service
```bash
docker-compose restart api
```

### Rebuild After Code Changes
```bash
docker-compose up -d --build api
```

### Scale API Service
```bash
docker-compose up -d --scale api=3
```

### View Resource Usage
```bash
docker stats
```

---

## Troubleshooting

### Issue: Container won't start

**Check logs**:
```bash
docker-compose logs api
```

**Common causes**:
- Port already in use
- Missing dependencies
- Database not ready

**Solutions**:
```bash
# Check port usage
netstat -an | grep 8000

# Restart database first
docker-compose restart postgres
sleep 5
docker-compose restart api
```

### Issue: API returns 500 errors

**Check API logs**:
```bash
docker-compose logs --tail=50 api
```

**Check database connection**:
```bash
docker-compose exec api env | grep DATABASE
```

**Restart API**:
```bash
docker-compose restart api
```

### Issue: Can't connect to database

**Verify database is running**:
```bash
docker-compose ps postgres
```

**Check database logs**:
```bash
docker-compose logs postgres
```

**Verify connection from API container**:
```bash
docker-compose exec api nc -zv postgres 5432
```

### Issue: Prometheus has no data

**Check Prometheus targets**:
```bash
curl http://localhost:9090/api/v1/targets
```

**Verify API is exposing metrics**:
```bash
curl http://localhost:8000/metrics
```

**Check network connectivity**:
```bash
docker-compose exec prometheus nc -zv api 8000
```

### Issue: High memory usage

**Check resource usage**:
```bash
docker stats --no-stream
```

**Limit container resources**:
```yaml
# In docker-compose.yml
services:
  api:
    deploy:
      resources:
        limits:
          memory: 1G
          cpus: '1'
```

---

## Production Considerations

### Current Setup: Development
The docker-compose.yml is configured for **development**, not production.

### For Production:

#### 1. Use Secrets
```yaml
# Don't use plain text passwords
environment:
  - POSTGRES_PASSWORD_FILE=/run/secrets/db_password
secrets:
  db_password:
    external: true
```

#### 2. Add Health Checks
Already configured! But verify they work:
```bash
docker inspect mlops-api | grep -A 10 Healthcheck
```

#### 3. Set Resource Limits
```yaml
deploy:
  resources:
    limits:
      cpus: '2'
      memory: 2G
    reservations:
      cpus: '1'
      memory: 1G
```

#### 4. Use Production Database
Don't use containerized PostgreSQL in production:
- Use managed database (AWS RDS, GCP Cloud SQL, Azure Database)
- Backup strategy
- High availability

#### 5. Persistent Storage
```yaml
volumes:
  model_registry:
    driver: local
    driver_opts:
      type: nfs
      o: addr=10.0.0.1,rw
      device: ":/models"
```

#### 6. SSL/TLS
Add nginx reverse proxy:
```yaml
nginx:
  image: nginx:alpine
  ports:
    - "443:443"
  volumes:
    - ./nginx.conf:/etc/nginx/nginx.conf
    - ./ssl:/etc/nginx/ssl
```

---

## Cleanup

### Remove Everything
```bash
# Stop and remove containers
docker-compose down

# Remove volumes (data will be lost!)
docker-compose down -v

# Remove images
docker rmi mlops-pipeline:v1

# Remove all unused Docker resources
docker system prune -a
```

### Keep Data, Remove Containers
```bash
# Stop and remove containers only
docker-compose down

# Data persists in named volumes
docker volume ls
```

---

## Next Steps

After verifying Docker works:

1. **Load Test**
   ```bash
   pip install locust
   locust -f tests/load/locustfile.py --host=http://localhost:8000
   ```

2. **Deploy to Kubernetes**
   ```bash
   kubectl apply -f k8s/
   ```

3. **Set Up CI/CD**
   - GitHub Actions for automated builds
   - Automated testing
   - Deploy to staging/production

4. **Add Authentication**
   - JWT tokens
   - API keys
   - Rate limiting

5. **Production Deployment**
   - AWS EKS / GCP GKE / Azure AKS
   - Managed database
   - CDN for static assets
   - Monitoring and alerting

---

## Performance Benchmarks

Expected performance on local Docker:

| Metric | Target | Notes |
|--------|--------|-------|
| API latency (P50) | < 50ms | Without complex models |
| API latency (P95) | < 100ms | |
| Throughput | > 100 req/s | Single container |
| Memory usage | < 500MB | Per container |
| Cold start | < 10s | First request |

---

## Useful Commands Cheat Sheet

```bash
# Quick status check
docker-compose ps && docker-compose logs --tail=5 api

# Full restart
docker-compose down && docker-compose up -d

# Watch logs
docker-compose logs -f --tail=100

# Execute command in container
docker-compose exec api python examples/simple_deployment.py

# Shell access
docker-compose exec api /bin/bash

# Database backup
docker-compose exec postgres pg_dump -U mlops mlops_db > backup.sql

# Database restore
docker-compose exec -T postgres psql -U mlops mlops_db < backup.sql

# Check disk usage
docker system df

# Monitor resources
watch docker stats --no-stream
```

---

## Support

If you encounter issues:

1. Check logs: `docker-compose logs`
2. Verify all containers are running: `docker-compose ps`
3. Check GitHub issues
4. Review DEBUGGING_LOG.md

---

**Last Updated**: November 12, 2025
**Docker Version**: 20.10+
**Docker Compose Version**: 2.0+
**Status**: Ready for local testing ✅
