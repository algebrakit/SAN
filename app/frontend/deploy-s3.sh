#!/bin/bash

# Frontend deployment script for AWS S3 + CloudFront

# Configuration
BUCKET_NAME="san-frontend-demo"  # Change this to your bucket name
REGION="eu-west-1"
CLOUDFRONT_DISTRIBUTION_ID="E3NVBGQJRUD54B"  # Add this after creating CloudFront
PROFILE="algebrakit-dev"  # AWS CLI profile to use

# Build the frontend
echo "Building frontend..."
npm run build

# Create S3 bucket if it doesn't exist
echo "Creating S3 bucket..."
aws s3api create-bucket \
  --bucket $BUCKET_NAME \
  --region $REGION \
  --create-bucket-configuration LocationConstraint=$REGION \
  --profile $PROFILE \
  2>/dev/null || echo "Bucket already exists"

# Enable static website hosting
echo "Configuring bucket for static hosting..."
aws s3 website s3://$BUCKET_NAME/ \
  --index-document index.html \
  --error-document index.html \
  --profile $PROFILE

# Set bucket policy for public read
echo "Setting bucket policy..."
cat > /tmp/bucket-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "PublicReadGetObject",
      "Effect": "Allow",
      "Principal": "*",
      "Action": "s3:GetObject",
      "Resource": "arn:aws:s3:::$BUCKET_NAME/*"
    }
  ]
}
EOF

aws s3api put-bucket-policy \
  --bucket $BUCKET_NAME \
  --policy file:///tmp/bucket-policy.json \
  --profile $PROFILE

# Upload files with proper content types
echo "Uploading files to S3..."

# Upload HTML files with correct content type
aws s3 sync www/ s3://$BUCKET_NAME/ \
  --delete \
  --exclude "*" \
  --include "*.html" \
  --content-type "text/html" \
  --cache-control max-age=0,no-cache,no-store,must-revalidate \
  --profile $PROFILE

# Upload CSS files
aws s3 sync www/ s3://$BUCKET_NAME/ \
  --exclude "*" \
  --include "*.css" \
  --content-type "text/css" \
  --cache-control max-age=31536000 \
  --profile $PROFILE

# Upload JS files
aws s3 sync www/ s3://$BUCKET_NAME/ \
  --exclude "*" \
  --include "*.js" \
  --content-type "application/javascript" \
  --cache-control max-age=31536000 \
  --profile $PROFILE

# Upload JSON files
aws s3 sync www/ s3://$BUCKET_NAME/ \
  --exclude "*" \
  --include "*.json" \
  --content-type "application/json" \
  --cache-control max-age=31536000 \
  --profile $PROFILE

# Upload images
aws s3 sync www/ s3://$BUCKET_NAME/ \
  --exclude "*" \
  --include "*.png" \
  --include "*.jpg" \
  --include "*.jpeg" \
  --include "*.gif" \
  --include "*.svg" \
  --include "*.ico" \
  --cache-control max-age=31536000 \
  --profile $PROFILE

# Upload any remaining files
aws s3 sync www/ s3://$BUCKET_NAME/ \
  --exclude "*.html" \
  --exclude "*.css" \
  --exclude "*.js" \
  --exclude "*.json" \
  --exclude "*.png" \
  --exclude "*.jpg" \
  --exclude "*.jpeg" \
  --exclude "*.gif" \
  --exclude "*.svg" \
  --exclude "*.ico" \
  --cache-control max-age=31536000 \
  --profile $PROFILE

# Invalidate CloudFront cache (if distribution ID is set)
if [ ! -z "$CLOUDFRONT_DISTRIBUTION_ID" ]; then
  echo "Invalidating CloudFront cache..."
  aws cloudfront create-invalidation \
    --distribution-id $CLOUDFRONT_DISTRIBUTION_ID \
    --paths "/*" \
    --profile $PROFILE
fi

echo "Deployment complete!"
echo "Website URL: http://$BUCKET_NAME.s3-website-$REGION.amazonaws.com"