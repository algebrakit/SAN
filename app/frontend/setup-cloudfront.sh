#!/bin/bash

# CloudFront setup script for SAN App

# Configuration
BUCKET_NAME="san-frontend-demo"
BACKEND_URL="egxud3kv2k.eu-west-1.awsapprunner.com"
PROFILE="algebrakit-dev"
REGION="eu-west-1"

echo "Creating CloudFront distribution..."

# Create CloudFront distribution configuration
cat > /tmp/cloudfront-config.json << EOF
{
  "CallerReference": "san-app-$(date +%s)",
  "Comment": "SAN App Distribution",
  "DefaultRootObject": "index.html",
  "Origins": {
    "Quantity": 2,
    "Items": [
      {
        "Id": "S3-Frontend",
        "DomainName": "${BUCKET_NAME}.s3.amazonaws.com",
        "S3OriginConfig": {
          "OriginAccessIdentity": ""
        }
      },
      {
        "Id": "AppRunner-Backend",
        "DomainName": "${BACKEND_URL}",
        "CustomOriginConfig": {
          "HTTPPort": 80,
          "HTTPSPort": 443,
          "OriginProtocolPolicy": "https-only",
          "OriginSslProtocols": {
            "Quantity": 3,
            "Items": ["TLSv1", "TLSv1.1", "TLSv1.2"]
          }
        }
      }
    ]
  },
  "DefaultCacheBehavior": {
    "TargetOriginId": "S3-Frontend",
    "ViewerProtocolPolicy": "redirect-to-https",
    "TrustedSigners": {
      "Enabled": false,
      "Quantity": 0
    },
    "ForwardedValues": {
      "QueryString": false,
      "Cookies": {
        "Forward": "none"
      },
      "Headers": {
        "Quantity": 0
      }
    },
    "MinTTL": 0,
    "AllowedMethods": {
      "Quantity": 2,
      "Items": ["GET", "HEAD"],
      "CachedMethods": {
        "Quantity": 2,
        "Items": ["GET", "HEAD"]
      }
    },
    "Compress": true
  },
  "CacheBehaviors": {
    "Quantity": 2,
    "Items": [
      {
        "PathPattern": "/convert",
        "TargetOriginId": "AppRunner-Backend",
        "ViewerProtocolPolicy": "redirect-to-https",
        "TrustedSigners": {
          "Enabled": false,
          "Quantity": 0
        },
        "ForwardedValues": {
          "QueryString": true,
          "Cookies": {
            "Forward": "all"
          },
          "Headers": {
            "Quantity": 4,
            "Items": ["Content-Type", "Accept", "Authorization", "Origin"]
          }
        },
        "MinTTL": 0,
        "DefaultTTL": 0,
        "MaxTTL": 0,
        "AllowedMethods": {
          "Quantity": 7,
          "Items": ["GET", "HEAD", "OPTIONS", "PUT", "POST", "PATCH", "DELETE"],
          "CachedMethods": {
            "Quantity": 2,
            "Items": ["GET", "HEAD"]
          }
        },
        "Compress": false
      },
      {
        "PathPattern": "/health",
        "TargetOriginId": "AppRunner-Backend",
        "ViewerProtocolPolicy": "redirect-to-https",
        "TrustedSigners": {
          "Enabled": false,
          "Quantity": 0
        },
        "ForwardedValues": {
          "QueryString": false,
          "Cookies": {
            "Forward": "none"
          },
          "Headers": {
            "Quantity": 0
          }
        },
        "MinTTL": 0,
        "DefaultTTL": 0,
        "MaxTTL": 0,
        "AllowedMethods": {
          "Quantity": 2,
          "Items": ["GET", "HEAD"],
          "CachedMethods": {
            "Quantity": 2,
            "Items": ["GET", "HEAD"]
          }
        },
        "Compress": false
      }
    ]
  },
  "CustomErrorResponses": {
    "Quantity": 0,
    "Items": []
  },
  "Enabled": true,
  "PriceClass": "PriceClass_100",
  "ViewerCertificate": {
    "CloudFrontDefaultCertificate": true
  }
}
EOF

# Create the distribution
echo "Creating CloudFront distribution..."
DISTRIBUTION_ID=$(aws cloudfront create-distribution \
  --distribution-config file:///tmp/cloudfront-config.json \
  --profile $PROFILE \
  --query 'Distribution.Id' \
  --output text)

if [ $? -eq 0 ]; then
  echo "✅ CloudFront distribution created successfully!"
  echo "Distribution ID: $DISTRIBUTION_ID"
  
  # Get the CloudFront domain name
  DOMAIN_NAME=$(aws cloudfront get-distribution \
    --id $DISTRIBUTION_ID \
    --profile $PROFILE \
    --query 'Distribution.DomainName' \
    --output text)
  
  echo ""
  echo "========================================="
  echo "CloudFront Setup Complete!"
  echo "========================================="
  echo ""
  echo "Distribution ID: $DISTRIBUTION_ID"
  echo "CloudFront URL: https://$DOMAIN_NAME"
  echo ""
  echo "⏳ Note: It takes 15-20 minutes for the distribution to be fully deployed."
  echo ""
  echo "Next steps:"
  echo "1. Wait for deployment to complete (check status in AWS Console)"
  echo "2. Update deploy-s3.sh with the Distribution ID: $DISTRIBUTION_ID"
  echo "3. Test the app at: https://$DOMAIN_NAME"
  echo ""
  echo "To check deployment status:"
  echo "aws cloudfront get-distribution --id $DISTRIBUTION_ID --profile $PROFILE --query 'Distribution.Status'"
  echo ""
  
  # Update the deploy script with the distribution ID
  echo "Would you like to update deploy-s3.sh with the CloudFront distribution ID? (y/n)"
  read -r response
  if [[ "$response" == "y" ]]; then
    sed -i '' "s/CLOUDFRONT_DISTRIBUTION_ID=\"\"/CLOUDFRONT_DISTRIBUTION_ID=\"$DISTRIBUTION_ID\"/" deploy-s3.sh
    echo "✅ Updated deploy-s3.sh with CloudFront distribution ID"
  fi
  
else
  echo "❌ Failed to create CloudFront distribution"
  echo "Please check your AWS credentials and try again"
fi

# Clean up
rm -f /tmp/cloudfront-config.json