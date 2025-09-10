# Docker Deployment Guide

This guide explains how to build and deploy the SAN backend using Docker and AWS App Runner.

## 📦 Files Created

- **`Dockerfile`** - Multi-stage build for optimized image (~350MB)
- **`requirements-docker.txt`** - CPU-only PyTorch dependencies
- **`.dockerignore`** - Excludes unnecessary files from image
- **`apprunner.yaml`** - AWS App Runner configuration
- **Updated `server.py`** - Now supports PORT environment variable

## 🏗️ Local Testing

### Build the Docker Image

**IMPORTANT**: After reorganization, use the build script:

```bash
cd app/backend
./build-docker.sh
```

Or manually:
```bash
cd app/backend
docker build -f Dockerfile -t san-backend:latest ../..
```

Note: The context is set to the project root (`../..`) to access `san_model/`, `data_tools/`, etc.

### Run Locally
```bash
docker run -p 8080:8080 san-backend
```

Test the API:
```bash
curl http://localhost:8080/health
```

## 🚀 AWS App Runner Deployment

### Prerequisites
1. AWS CLI configured
2. AWS ECR repository created
3. AWS App Runner service ready

### Step 1: Create ECR Repository
```bash
aws ecr create-repository --repository-name san-backend --region eu-west-1
```

### Step 2: Build and Push to ECR
```bash
# Get ECR login
aws ecr get-login-password --profile algebrakit-dev | docker login --username AWS --password-stdin 802678058659.dkr.ecr.eu-west-1.amazonaws.com

# Build image
docker build -f Dockerfile -t san-backend:latest ../..

# Tag for ECR
docker tag san-backend:latest 802678058659.dkr.ecr.eu-west-1.amazonaws.com/san-backend:latest

# Push to ECR
docker push 802678058659.dkr.ecr.eu-west-1.amazonaws.com/san-backend:latest
```

### Step 3: Create App Runner Service

#### Option A: AWS Console
1. Go to AWS App Runner console
2. Click "Create service"
3. Choose "Container registry" → "Amazon ECR"
4. Select your pushed image
5. Configure using settings from `apprunner.yaml`
6. Deploy!

#### Option B: AWS CLI
```bash
aws apprunner create-service \
  --service-name "san-backend-demo" \
  --source-configuration '{
    "ImageRepository": {
      "ImageIdentifier": "[YOUR_ACCOUNT_ID].dkr.ecr.us-east-1.amazonaws.com/san-backend:latest",
      "ImageConfiguration": {
        "Port": "8080",
        "RuntimeEnvironmentVariables": {
          "FLASK_ENV": "production",
          "PORT": "8080"
        }
      },
      "ImageRepositoryType": "ECR"
    },
    "AutoDeploymentsEnabled": false
  }' \
  --instance-configuration '{
    "Cpu": "0.25 vCPU",
    "Memory": "0.5 GB"
  }' \
  --auto-scaling-configuration '{
    "MinSize": 0,
    "MaxSize": 1
  }'
```

## 📊 Configuration Details

### Instance Size
- **CPU**: 0.25 vCPU (sufficient for demo)
- **Memory**: 0.5 GB (model is only 31MB)
- **Cost**: ~$5-8/month with idle scaling

### Auto-scaling
- **Min instances**: 0 (scales to zero)
- **Max instances**: 1 (demo only needs one)
- **Idle timeout**: 5 minutes
- **Cold start**: ~5-10 seconds

### Health Check
- **Endpoint**: `/health`
- **Interval**: 30 seconds
- **Timeout**: 10 seconds

## 🔧 Optimization Notes

### Image Size Optimizations
- Using CPU-only PyTorch (saves ~600MB)
- Multi-stage build (reduces by ~40%)
- opencv-python-headless (saves ~50MB)
- Final image: ~350MB (vs 1GB+ with full PyTorch)

### Performance Optimizations
- Pre-compiled Python files
- Model loaded once at startup
- Health check endpoint for monitoring
- Non-root user for security

## 🐛 Troubleshooting

### Architecture Issues (exec format error)
If you get "exec format error" on AWS App Runner, you need to build for AMD64:

```bash
# Build specifically for AMD64 architecture
docker buildx build --platform linux/amd64 -t san-backend .

# Or set up buildx and push directly to ECR
docker buildx build --platform linux/amd64 -t [ECR_URL]:latest --push .
```

The Dockerfile has been updated to target `linux/amd64` explicitly for App Runner compatibility.

### Docker Build Issues
```bash
# Clear Docker cache
docker system prune -a

# Build with no cache
docker build --no-cache -t san-backend .

# If PyTorch downloads full CUDA version, ensure requirements-docker.txt uses CPU index:
# --index-url https://download.pytorch.org/whl/cpu
```

### Container Runtime Issues

**ModuleNotFoundError: No module named 'flask'**
- Fixed: Updated Dockerfile to properly copy Python packages to `/home/appuser/.local`

**ModuleNotFoundError: No module named 'infer'**
- Fixed: Added `infer/` directory to Dockerfile COPY commands

**FileNotFoundError: data/word.txt**
- Fixed: Copy `word.txt` from parent directory to backend folder, then include in Docker build

### Port Issues
```bash
# Check if port 8080 is in use
lsof -i :8080

# Use different port
docker run -p 3000:8080 san-backend
```

### Memory Issues
If the container crashes, increase memory in `apprunner.yaml`:
```yaml
instance:
  memory: 1.0  # Increase to 1GB
```

### ARM64 (Apple Silicon) Development
For local development on ARM64 Macs:
- PyTorch requirements changed from `torch==2.0.1+cpu` to `torch==2.0.1` 
- Debian packages changed from `libgl1-mesa-glx` to `libgl1`
- Production deployment targets `linux/amd64` for App Runner compatibility

## 🔄 Updating the Service

When you make changes:

1. Rebuild the image:
```bash
docker build -t san-backend .
```

2. Push to ECR:
```bash
docker tag san-backend:latest [YOUR_ACCOUNT_ID].dkr.ecr.us-east-1.amazonaws.com/san-backend:latest
docker push [YOUR_ACCOUNT_ID].dkr.ecr.us-east-1.amazonaws.com/san-backend:latest
```

3. Update App Runner:
```bash
aws apprunner update-service --service-arn [YOUR_SERVICE_ARN] --source-configuration ...
```

## 📈 Monitoring

View logs in AWS Console:
- App Runner → Your Service → Logs
- CloudWatch Logs → `/aws/apprunner/san-backend-demo`

## 🎯 Next Steps

For production deployment on Kubernetes:
1. Use this same Docker image
2. Create Kubernetes manifests
3. Deploy to your EKS cluster
4. Scale horizontally as needed

The Docker image is designed to work in both App Runner (demo) and Kubernetes (production) environments!