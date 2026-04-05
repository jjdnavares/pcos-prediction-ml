# Deployment Guide

> Step-by-step instructions for deploying the PCOS Prediction API locally and to cloud platforms.

---

## Prerequisites

- Python 3.10+
- Docker (for containerized deployment)
- Trained model artifacts in `models/` (run `python scripts/train_pipeline.py` first)

---

## 1. Local Development

### 1.1 Direct (Uvicorn)

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model (if not already done)
python scripts/train_pipeline.py

# Start the API
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API is now available at:
- Swagger UI: http://localhost:8000/docs
- Health check: http://localhost:8000/health/
- Predictions: POST http://localhost:8000/api/v1/predict

### 1.2 Docker

```bash
# Build the image
docker build -t pcos-api:latest .

# Run the container
docker run -p 8000:8000 pcos-api:latest
```

### 1.3 Docker Compose

```bash
# Start in detached mode
docker-compose up -d

# Check logs
docker-compose logs -f

# Stop
docker-compose down
```

The `docker-compose.yml` mounts `models/` and `data/` as read-only volumes, so you can update the model without rebuilding the image.

---

## 2. Verify Deployment

After starting the service (any method), verify it works:

```bash
# 1. Health check
curl http://localhost:8000/health/
# Expected: {"status": "healthy", "model_loaded": true, ...}

# 2. Readiness probe
curl http://localhost:8000/health/ready
# Expected: {"status": "ready"}

# 3. Test prediction
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 28,
    "vit_d3": 22.0,
    "follicle_no_l": 13,
    "follicle_no_r": 14,
    "avg_f_size_r": 18.0,
    "endometrium": 7.5,
    "skin_darkening": 1,
    "hair_growth": 1,
    "weight_gain": 1,
    "hair_loss": 0,
    "fast_food": 1,
    "cycle_regularity": 4,
    "cycle_length": 42
  }'
```

---

## 3. Cloud Deployment

### 3.1 Render

Render auto-detects the `Dockerfile` in the repo root.

1. Go to [render.com](https://render.com) and create a new **Web Service**
2. Connect your GitHub repo (`jjdnavares/pcos-prediction-ml`)
3. Settings:
   - **Environment:** Docker
   - **Branch:** main
   - **Instance Type:** Free (or Starter for production)
   - **Health Check Path:** `/health/live`
4. Click **Deploy**

A `deployment/render.yaml` is included for blueprint deploys:

```bash
# Or deploy via Render Blueprint
# Push render.yaml to repo root, then connect in Render dashboard
```

### 3.2 Railway

1. Go to [railway.app](https://railway.app) and create a new project
2. Connect your GitHub repo
3. Railway auto-detects the Dockerfile
4. Set environment variables:
   - `PORT=8000`
   - `ENV=production`
5. Deploy

A `deployment/railway.json` is included for configuration.

### 3.3 Fly.io

```bash
# Install flyctl
curl -L https://fly.io/install.sh | sh

# Launch (creates fly.toml)
fly launch --name pcos-prediction-api

# Deploy
fly deploy

# Check status
fly status
```

### 3.4 AWS (ECS / App Runner)

```bash
# Build and push to ECR
aws ecr get-login-password --region ap-southeast-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.ap-southeast-1.amazonaws.com

docker tag pcos-api:latest <account-id>.dkr.ecr.ap-southeast-1.amazonaws.com/pcos-api:latest
docker push <account-id>.dkr.ecr.ap-southeast-1.amazonaws.com/pcos-api:latest

# Then create an ECS service or App Runner service pointing to this image
```

---

## 4. Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ENV` | `development` | Environment name (`development`, `production`) |
| `LOG_LEVEL` | `INFO` | Logging level |
| `PORT` | `8000` | API port (used by cloud platforms) |

---

## 5. Production Checklist

Before deploying to production:

- [ ] Model trained and tested (`pytest tests/ -v` passes)
- [ ] `models/best_model.pkl`, `scaler.pkl`, and `model_config.json` present
- [ ] Health endpoints responding (`/health/`, `/health/ready`, `/health/live`)
- [ ] Docker build succeeds (`docker build -t pcos-api:latest .`)
- [ ] Container starts and serves predictions
- [ ] Environment variables configured
- [ ] CORS origins restricted (update `app/main.py` for production domains)
- [ ] Model version in `model_config.json` matches the release tag

---

## 6. Monitoring

### Health Checks

The API exposes three health endpoints for orchestrators:

| Endpoint | Purpose | Used By |
|----------|---------|---------|
| `/health/` | Full health status with model info | Dashboards |
| `/health/ready` | Readiness probe (model loaded) | Kubernetes / ECS |
| `/health/live` | Liveness probe (process alive) | Docker HEALTHCHECK |

### Logs

```bash
# Docker logs
docker-compose logs -f api

# Or direct
uvicorn app.main:app --log-level info
```

### MLflow Experiment History

```bash
# Review past training runs
mlflow ui --backend-store-uri mlruns
# Open http://localhost:5000
```
