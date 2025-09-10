#!/bin/bash

# Build Docker image for SAN backend from the reorganized structure
# This script should be run from the app/backend directory

echo "Building SAN backend Docker image..."
echo "Context: $(pwd)"

# Build from the app/backend directory but with context at the project root
# so we can access the san_model, data_tools, etc.
docker build -f Dockerfile -t san-backend:latest ../..

if [ $? -eq 0 ]; then
    echo "Docker image built successfully!"
    echo "Run with: docker run -p 8080:8080 san-backend:latest"
else
    echo "Docker build failed!"
    exit 1
fi