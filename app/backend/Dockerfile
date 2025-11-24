# Multi-stage build for optimized image size
# Explicitly target AMD64 for AWS App Runner compatibility
FROM --platform=linux/amd64 python:3.9-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY app/backend/requirements-docker.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements-docker.txt

# --- Runtime Stage ---
FROM --platform=linux/amd64 python:3.9-slim

# Install runtime dependencies only
# Updated for compatibility with newer Debian and ARM64
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 appuser

# Set working directory
WORKDIR /app

# Create appuser and set up directories
RUN mkdir -p /home/appuser/.local

# Copy Python packages from builder
COPY --from=builder /root/.local /home/appuser/.local

# Fix ownership
RUN chown -R appuser:appuser /home/appuser/.local

# Copy application code
COPY --chown=appuser:appuser app/backend/*.py app/backend/*.yaml ./

# Copy utils from project root
COPY --chown=appuser:appuser utils/ ./utils/

# Copy the entire project structure for imports
COPY --chown=appuser:appuser model/ ./model/
COPY --chown=appuser:appuser data_tools/ ./data_tools/

# Copy only essential data files
RUN mkdir -p resources
COPY --chown=appuser:appuser app/backend/resources/ ./resources/

# Ensure local pip packages are in PATH
ENV PATH=/home/appuser/.local/bin:$PATH
ENV PYTHONPATH=/app:$PYTHONPATH

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PORT=8080 \
    FLASK_APP=server.py \
    CONFIG_PATH=/app/config.yaml

# Pre-compile Python files for faster startup
RUN python -m compileall .

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health')" || exit 1

# Expose port (App Runner uses 8080 by default)
EXPOSE 8080

# Start the Flask server
# Using exec form for proper signal handling
CMD ["python", "server.py", "8080"]