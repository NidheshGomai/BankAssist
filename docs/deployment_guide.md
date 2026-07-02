# BankAssist RAG — Deployment Guide

## Prerequisites

- **Docker** 24+ and **Docker Compose** v2
- **NVIDIA GPU** with 6GB+ VRAM (CUDA 12.x drivers, `nvidia-container-toolkit`)
- **Python 3.12** (for local development)
- **HuggingFace token** with access to `Qwen/Qwen3-4B` (gated model)
- **Google service account** with Drive API access (for Drive ingestion)

## Quick Start (Docker Compose)

### 1. Clone and configure

```bash
git clone <repo-url> BankAssist
cd BankAssist

# Copy environment template and fill in your values
cp .env.example .env
```

### 2. Configure `.env`

```env
# Required
HUGGINGFACE_TOKEN=hf_your_token_here
GOOGLE_DRIVE_FOLDER_ID=your_drive_folder_id
GOOGLE_APPLICATION_CREDENTIALS=config/google_service_account.json

# Optional overrides
APP_ENVIRONMENT=production
API_PORT=8000
LOG_LEVEL=INFO
```

### 3. Place service account JSON

```bash
# Place your Google service account key at:
config/google_service_account.json
```

### 4. Build and start

```bash
# With GPU support (default)
docker compose up -d --build

# CPU-only mode
CUDA_VISIBLE_DEVICES="" docker compose up -d --build

# With monitoring stack (Prometheus + Grafana)
docker compose --profile monitoring up -d

# Monitor logs
docker compose logs -f bankassist-api
```

### 5. Verify

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Expected response: {"status":"HEALTHY","version":"1.0.0",...}
```

## Local Development (without Docker)

### 1. Install system dependencies

**Windows:**
```powershell
# Install CUDA 12.x from NVIDIA website
# Install Visual Studio Build Tools with C++ workload
```

**Linux:**
```bash
sudo apt-get update && sudo apt-get install -y \
    build-essential \
    python3-dev \
    libgomp1
```

### 2. Create virtual environment

```bash
python3.12 -m venv venv
source venv/bin/activate  # Linux/Mac
# .\venv\Scripts\Activate  # Windows PowerShell
```

### 3. Install dependencies

```bash
# For GPU training (via uv — recommended)
pip install uv
uv sync

# For RAG API
pip install -r requirements_rag.txt

# Install PyTorch with CUDA 12.4
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 4. Run the API

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Run tests

```bash
pytest tests/ -v --asyncio-mode=auto
pytest tests/evaluation/ -v
```

## GPU Configuration

### Check GPU availability

```python
# From the Python shell
from app.utils.device import detect_device
info = detect_device()
print(info)
# Expected: DeviceInfo(device='cuda', device_name='NVIDIA ...', total_vram_gb=6.0, ...)
```

### VRAM requirements (minimum)

| Component | VRAM |
|-----------|------|
| Qwen3-4B (4-bit) | ~4.5 GB |
| BGE-M3 embedder | ~1.0 GB |
| BGE-reranker-large | ~1.5 GB |
| **Total** | **~6.0 GB** |

### Troubleshooting GPU

- **CUDA out of memory**: Reduce `models.llm.max_new_tokens` in `config/config.yaml`
- **CUDA not detected**: Verify drivers: `nvidia-smi`; verify toolkit: `nvcc --version`
- **bitsandbytes errors**: Install appropriate version for your CUDA: `pip install bitsandbytes-cuda124`

## Google Drive Setup

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a project → Enable Drive API
3. Create a service account → download JSON key
4. Share your Google Drive folder with the service account email
5. Set `GOOGLE_DRIVE_FOLDER_ID` in `.env`
6. Place the JSON key at `config/google_service_account.json`

## Production Hardening

### Environment variables

```env
APP_ENVIRONMENT=production
LOG_LEVEL=WARNING
API_SECRET_KEY=<openssl rand -hex 32>
ALLOWED_ORIGINS=https://yourfrontend.com
RATE_LIMIT_ENABLED=true
RATE_LIMIT_PER_MINUTE=60
RATE_LIMIT_BURST=10
```

### Docker recommendations

```yaml
# docker-compose.yml additions for production:
services:
  bankassist-api:
    restart: always
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
    deploy:
      resources:
        limits:
          memory: 8G
```

### Security checklist

- [ ] Rotate HuggingFace token to a machine-account token
- [ ] Set a strong `API_SECRET_KEY` in production
- [ ] Restrict `ALLOWED_ORIGINS` to known frontend domains
- [ ] Enable HTTPS via reverse proxy (Caddy, nginx, Traefik)
- [ ] Run Docker containers with non-root user
- [ ] Regularly rotate Google service account keys
- [ ] Set `HALLUCINATION_GUARD_ENABLED=true` and `CONFIDENCE_THRESHOLD=0.40`

## API Endpoints

See [API Reference](api_reference.md) for full documentation.

## Monitoring

```bash
# When started with --profile monitoring:
# Prometheus: http://localhost:9090
# Grafana:    http://localhost:3000 (admin/admin)

# Available metrics (at /metrics):
# - rag_retrieval_latency_ms
# - rag_generation_latency_ms
# - rag_confidence_score
# - rag_refusal_count
# - rag_chunk_count
```
