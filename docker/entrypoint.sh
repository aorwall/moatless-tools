#!/bin/bash

echo "Starting API..."

# Install custom dependencies if custom_requirements.txt exists
if [ -f /app/custom_requirements.txt ]; then
    echo "Installing custom dependencies from /app/custom_requirements.txt..."
    uv pip install -r /app/custom_requirements.txt
fi

# Use WORKERS_COUNT env var if set, otherwise default to 1
WORKERS=${WORKERS_COUNT:-1}
echo "Starting with $WORKERS workers"

# Run with Gunicorn using Uvicorn workers - optimized for k8s networking
exec uv run gunicorn moatless.api.api:create_api \
    --worker-class uvicorn.workers.UvicornWorker \
    --workers $WORKERS \
    --bind 0.0.0.0:8000 \
    --timeout 300 \
    --keep-alive 75 \
    --log-level info \
    --access-logfile - \
    --error-logfile - \
    --forwarded-allow-ips="*" \
    --worker-tmp-dir /dev/shm \
    --graceful-timeout 60 \
    --preload \
    --threads 4 \
    --enable-stdio-inheritance \
    --proxy-protocol \
    --worker-connections 1000 