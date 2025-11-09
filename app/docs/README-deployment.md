# SAN App - Complete Deployment Guide

This guide covers the complete deployment of the SAN (Syntax-Aware Network) application with both backend and frontend components.

## 🏗️ Architecture

```
Users → CloudFront → S3 (Frontend) + App Runner (Backend)
                 ├── /                → S3 (React/StencilJS app)  
                 ├── /convert         → App Runner (API)
                 └── /health          → App Runner (API)
```

## 📦 Prerequisites

1. AWS CLI configured with appropriate profile
2. Docker installed (for backend)
3. Node.js and npm (for frontend)

## 🚀 Backend Deployment (App Runner)

### 1. Build and Push Docker Image

```bash
cd backend

# Build for AMD64 (App Runner compatibility)
docker buildx build --platform linux/amd64 -t san-backend .

# Push to ECR (replace with your ECR repository)
docker tag san-backend:latest [YOUR_ACCOUNT_ID].dkr.ecr.eu-west-1.amazonaws.com/san-backend:latest
docker push [YOUR_ACCOUNT_ID].dkr.ecr.eu-west-1.amazonaws.com/san-backend:latest
```

### 2. Deploy to App Runner

Use AWS Console or CLI to create App Runner service with:
- Image: Your ECR image
- Port: 8080
- Instance: 0.25 vCPU, 0.5 GB RAM
- Auto-scaling: 0-1 instances

Note the App Runner URL (e.g., `https://abc123.eu-west-1.awsapprunner.com`)

## 🌐 Frontend Deployment (S3 + CloudFront)

### 1. Deploy Frontend to S3

```bash
cd frontend

# Configure deployment settings
# Edit deploy-s3.sh to set:
# - BUCKET_NAME
# - REGION
# - PROFILE

# Deploy
./deploy-s3.sh
```

### 2. Create CloudFront Distribution

```bash
# Edit setup-cloudfront.sh to set:
# - BACKEND_URL (your App Runner URL)
# - Other configuration values

# Create CloudFront distribution
./setup-cloudfront.sh
```

This will create a CloudFront distribution that:
- Routes `/` to S3 (frontend)
- Routes `/convert` and `/health` to App Runner (backend)
- Uses HTTPS with proper caching
- Avoids CORS issues by using same domain

### 3. Wait for Deployment

CloudFront deployment takes 15-20 minutes. Check status:

```bash
aws cloudfront get-distribution --id [DISTRIBUTION_ID] --profile [PROFILE] --query 'Distribution.Status'
```

## ✅ Testing Deployment

### Backend API Test
```bash
curl -X POST https://[CLOUDFRONT_URL]/convert \
  -H "Content-Type: application/json" \
  -d '{"strokes": [[[0,0],[10,10]]], "stroke_length": 25}'
```

### Frontend Test
Open `https://[CLOUDFRONT_URL]/` in browser and test drawing functionality.

## 🔧 Configuration Files

### Backend Files Created:
- `Dockerfile` - Multi-stage build for App Runner
- `requirements-docker.txt` - CPU-only PyTorch dependencies
- `apprunner.yaml` - App Runner service configuration
- `DOCKER_DEPLOYMENT.md` - Backend deployment guide

### Frontend Files Created:
- `deploy-s3.sh` - S3 deployment script
- `setup-cloudfront.sh` - CloudFront creation script
- `DEPLOYMENT.md` - Frontend deployment options

## 🐛 Common Issues & Solutions

### Architecture Mismatch
**Problem**: `exec format error` on App Runner
**Solution**: Use `--platform linux/amd64` when building Docker image

### Host Header Issues
**Problem**: API returns 404 through CloudFront
**Solution**: CloudFront only forwards specific headers (Content-Type, Accept, etc.), not Host header

### Content-Type Issues
**Problem**: Browser downloads files instead of displaying
**Solution**: Set correct Content-Type headers when uploading to S3

### CORS Issues
**Problem**: Frontend can't call backend API
**Solution**: Use CloudFront to serve both on same domain

## 📊 Cost Estimation

- **App Runner**: ~$8-12/month (with auto-scaling to 0)
- **S3**: ~$1-2/month (storage + requests)
- **CloudFront**: ~$1-3/month (data transfer)
- **Total**: ~$10-17/month

## 🔄 Updates & Maintenance

### Backend Updates:
1. Build new Docker image
2. Push to ECR
3. App Runner auto-deploys latest image

### Frontend Updates:
1. Run `./deploy-s3.sh` to upload new files
2. Invalidate CloudFront cache if needed

### Cache Invalidation:
```bash
aws cloudfront create-invalidation \
  --distribution-id [ID] \
  --paths "/*" \
  --profile [PROFILE]
```

## 🔒 Security Notes

- App Runner service uses non-root user
- S3 bucket has minimal public permissions
- CloudFront provides DDoS protection
- All communication uses HTTPS

## 🎯 Production Considerations

For production deployment:
1. Use custom domain with SSL certificate
2. Set up monitoring and alerts
3. Configure backup strategies
4. Implement CI/CD pipeline
5. Consider scaling to multiple regions

## 📈 Monitoring

- **App Runner**: AWS Console → App Runner → Logs
- **CloudFront**: AWS Console → CloudFront → Monitoring
- **S3**: AWS Console → S3 → Metrics

## 🆘 Troubleshooting

### Check Deployment Status:
```bash
# App Runner
aws apprunner describe-service --service-arn [ARN]

# CloudFront
aws cloudfront get-distribution --id [ID]

# S3
aws s3 ls s3://[BUCKET_NAME]/
```

### View Logs:
- App Runner logs in AWS Console
- CloudFront access logs (if enabled)
- Browser Developer Tools for frontend issues