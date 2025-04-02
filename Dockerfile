FROM --platform=${TARGETPLATFORM:-linux/amd64} python:3.11-slim

# Add ARG for target platform
ARG TARGETPLATFORM="linux/amd64"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install UV package manager
RUN curl -fsSL https://github.com/astral-sh/uv/releases/download/0.1.25/uv-installer.sh | sh && \
    mv ~/.cargo/bin/uv /usr/local/bin/

# Install Bun properly
RUN curl https://bun.sh/install | bash && \
    ln -s /root/.bun/bin/bun /usr/local/bin/bun

# Copy only dependency files first to leverage layer caching
COPY pyproject.toml uv.lock ./

# Install Python dependencies - this layer will be cached unless dependencies change
RUN uv pip install --no-deps -e ".[api]"

# Explicitly install websockets for uvicorn
RUN uv pip install websockets 

# TODO: Remove 
RUN uv pip install azure-monitor-opentelemetry

# Copy frontend dependency files only (package.json and lock files)
COPY moatless-ui/package.json moatless-ui/bun.lock ./moatless-ui/

# Install frontend dependencies - this layer will be cached unless dependencies change
WORKDIR /app/moatless-ui
RUN bun install
WORKDIR /app

# Copy build script and Python application files
COPY build_ui.py ./
COPY moatless ./moatless

# Copy frontend configuration files
COPY moatless-ui/vite.config.ts moatless-ui/tsconfig.json moatless-ui/tsconfig.app.json \
    moatless-ui/tsconfig.node.json moatless-ui/postcss.config.js moatless-ui/tailwind.config.js \
    moatless-ui/eslint.config.js moatless-ui/components.json moatless-ui/index.html ./moatless-ui/

# Copy frontend source code
COPY moatless-ui/src/ ./moatless-ui/src/
COPY moatless-ui/public/ ./moatless-ui/public/

# Set environment variable to preserve source maps in production build
ENV GENERATE_SOURCEMAP=true
# Set default API URL for production builds
ENV VITE_API_BASE_URL=/api

# Build the UI
RUN python build_ui.py

# Create directories for mounted volumes and set proper permissions
RUN mkdir -p /data/moatless /data/repos /data/index_stores && \
    chmod -R 777 /data

# Set environment variables
ENV PYTHONPATH=/app
ENV MOATLESS_DIR=/data/moatless
ENV MOATLESS_REPO_DIR=/data/repos
ENV INDEX_STORE_DIR=/data/index_stores

# Create non-root user with UID 1000 (same as in Kubernetes pod)
RUN groupadd -g 1000 appuser && \
    useradd -u 1000 -g 1000 -m -s /bin/bash appuser

# Configure git to trust all directories to avoid dubious ownership issues
# Set it both globally and for the non-root user
RUN git config --global --add safe.directory '*' && \
    mkdir -p /home/appuser/.config/git && \
    echo "[safe]" > /home/appuser/.gitconfig && \
    echo "	directory = *" >> /home/appuser/.gitconfig && \
    chown -R appuser:appuser /home/appuser/.gitconfig

# Expose the API port
EXPOSE 8000

# Create entrypoint script
RUN echo '#!/bin/bash\n\
    # Set git safe.directory at runtime to ensure it works with mounted volumes\n\
    git config --global --add safe.directory "*"\n\
    if [ "$1" = "api" ]; then\n\
    # Run using uvicorn directly instead of python -m\n\
    uvicorn moatless.api.api:create_api --host 0.0.0.0 --port 8000\n\
    elif [ "$1" = "worker" ]; then\n\
    rq worker --url $REDIS_URL\n\
    else\n\
    exec "$@"\n\
    fi' > /entrypoint.sh && chmod +x /entrypoint.sh

# Make sure the entrypoint script is executable by the non-root user
RUN chown appuser:appuser /entrypoint.sh && \
    chmod 755 /app && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

ENTRYPOINT ["/entrypoint.sh"]
CMD ["api"] 