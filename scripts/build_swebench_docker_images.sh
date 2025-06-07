#!/bin/bash

# Check if dataset_file or instance_id is provided
if [ $# -lt 1 ] || [ $# -gt 5 ]; then
    echo "Usage: $0 <dataset_file_or_instance_id> [swegym] [--no-push] [--no-base] [--arm]"
    echo ""
    echo "Arguments:"
    echo "  dataset_file_or_instance_id: Either a JSON dataset file or a single instance ID"
    echo "  swegym:                      Use swegym base images instead of swebench"
    echo "  --no-push:                   Build images but don't push to registry"
    echo "  --no-base:                   Skip building the base image"
    echo "  --arm:                       Build for ARM64 architecture instead of x86_64"
    echo ""
    echo "Examples:"
    echo "  # Build all instances from dataset file:"
    echo "  $0 moatless/evaluation/datasets/verified_mini_dataset.json"
    echo "  $0 moatless/evaluation/datasets/verified_mini_dataset.json swegym"
    echo "  $0 moatless/evaluation/datasets/verified_mini_dataset.json --no-push"
    echo "  $0 moatless/evaluation/datasets/verified_mini_dataset.json --no-base"
    echo "  $0 moatless/evaluation/datasets/verified_mini_dataset.json --arm"
    echo "  $0 moatless/evaluation/datasets/verified_mini_dataset.json swegym --no-push --no-base --arm"
    echo ""
    echo "  # Build single instance:"
    echo "  $0 django__django-11099"
    echo "  $0 django__django-11099 swegym"
    echo "  $0 django__django-11099 --no-push"
    echo "  $0 django__django-11099 --arm"
    echo "  $0 django__django-11099 swegym --no-base --no-push --arm"
    exit 1
fi

DATASET_FILE_OR_INSTANCE=$1
SWEGYM_MODE=false
PUSH_IMAGES=true
BUILD_BASE=true
ARM_MODE=false

# Parse arguments
for arg in "$@"; do
    if [ "$arg" = "swegym" ]; then
        SWEGYM_MODE=true
        echo "Running in swegym mode"
    elif [ "$arg" = "--no-push" ]; then
        PUSH_IMAGES=false
        echo "Images will be built but not pushed"
    elif [ "$arg" = "--no-base" ]; then
        BUILD_BASE=false
        echo "Base image will not be built (assuming it already exists)"
    elif [ "$arg" = "--arm" ]; then
        ARM_MODE=true
        echo "Building for ARM64 architecture"
    fi
done

# Set architecture-specific variables
if [ "$ARM_MODE" = true ]; then
    ARCH_SUFFIX="arm64"
    DOCKER_PLATFORM="linux/arm64"
else
    ARCH_SUFFIX="x86_64"
    DOCKER_PLATFORM="linux/amd64"
fi

echo "Building for architecture: $ARCH_SUFFIX (platform: $DOCKER_PLATFORM)"

# Check if INDEX_STORE_DIR environment variable is set
if [ -z "$INDEX_STORE_DIR" ]; then
    echo "Error: INDEX_STORE_DIR environment variable is not set"
    exit 1
fi

# Print INDEX_STORE_DIR for debugging
echo "Using INDEX_STORE_DIR: $INDEX_STORE_DIR"

# Check if the index store directory exists
if [ ! -d "$INDEX_STORE_DIR" ]; then
    echo "Creating index store directory $INDEX_STORE_DIR"
    mkdir -p "$INDEX_STORE_DIR"
fi

# Set the index store URL, use environment variable if set
if [ -z "$INDEX_STORE_URL" ]; then
    INDEX_STORE_URL="https://stmoatless.blob.core.windows.net/indexstore/20250118-voyage-code-3/"
    echo "INDEX_STORE_URL not set, using default: $INDEX_STORE_URL"
else
    echo "Using INDEX_STORE_URL: $INDEX_STORE_URL"
fi

# Function to download and extract index if it doesn't exist
download_index() {
    local instance_id=$1
    local index_dir="${INDEX_STORE_DIR}/${instance_id}"
    
    if [ ! -d "$index_dir" ] || [ -z "$(ls -A $index_dir 2>/dev/null)" ]; then
        echo "Index for $instance_id not found in $INDEX_STORE_DIR, downloading..."
        local temp_dir=$(mktemp -d)
        local zip_file="${temp_dir}/${instance_id}.zip"
        local download_url="${INDEX_STORE_URL}${instance_id}.zip"
        
        echo "Downloading from: $download_url"
        if ! curl -L -o "$zip_file" "$download_url"; then
            echo "Error: Failed to download index for $instance_id"
            rm -rf "$temp_dir"
            return 1
        fi
        
        echo "Extracting index to $index_dir"
        mkdir -p "$index_dir"
        if ! unzip -q "$zip_file" -d "$index_dir"; then
            echo "Error: Failed to extract index for $instance_id"
            rm -rf "$temp_dir"
            return 1
        fi
        
        rm -rf "$temp_dir"
        echo "Successfully downloaded and extracted index for $instance_id"
    else
        echo "Index for $instance_id already exists in $INDEX_STORE_DIR"
    fi
    
    return 0
}

# Check if instances directory exists, if not, run create_dataset_index.py
if [ ! -d "instances" ]; then
    echo "instances directory not found, running create_dataset_index.py"
    python3 scripts/create_dataset_index.py
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create dataset index"
        exit 1
    fi
fi

# Determine if the first argument is a dataset file or an instance ID
if [ -f "$DATASET_FILE_OR_INSTANCE" ] && [[ "$DATASET_FILE_OR_INSTANCE" == *.json ]]; then
    # Check if jq is installed (only needed for JSON parsing)
    if ! command -v jq &> /dev/null; then
        echo "Error: jq is required for parsing JSON dataset files but not installed. Please install jq with 'sudo apt-get install jq' or equivalent"
        exit 1
    fi
    
    # It's a JSON dataset file
    echo "Processing dataset file: $DATASET_FILE_OR_INSTANCE"
    
    # Extract instance IDs using jq for reliable JSON parsing
    INSTANCE_IDS=$(jq -r '.instance_ids[]? // empty' "$DATASET_FILE_OR_INSTANCE" 2>/dev/null)

    # If the above format fails, try alternate JSON formats
    if [ -z "$INSTANCE_IDS" ]; then
        # Try another common format (array of objects with instance_id field)
        INSTANCE_IDS=$(jq -r '.[]?.instance_id? // empty' "$DATASET_FILE_OR_INSTANCE" 2>/dev/null)
    fi

    if [ -z "$INSTANCE_IDS" ]; then
        # Try inspecting the file structure for debugging
        echo "Error: Could not parse instance IDs from $DATASET_FILE_OR_INSTANCE"
        echo "File content preview:"
        head -n 20 "$DATASET_FILE_OR_INSTANCE"
        exit 1
    fi
    
    echo "Found $(echo "$INSTANCE_IDS" | wc -l) instances to build"
else
    # It's a single instance ID
    INSTANCE_IDS="$DATASET_FILE_OR_INSTANCE"
    echo "Building single instance: $INSTANCE_IDS"
fi

# Build and push the base image first
if [ "$BUILD_BASE" = true ]; then
    BASE_IMAGE_NAME="aorwall/moatless.swebench.base.${ARCH_SUFFIX}"
    echo "Building base image for ${ARCH_SUFFIX}: ${BASE_IMAGE_NAME}"
    echo "Setting up buildx builder for ${ARCH_SUFFIX} builds..."
    docker buildx create --use --name multiarch-builder --driver docker-container --platform ${DOCKER_PLATFORM} || true
    
    # Create architecture-specific Dockerfile.base
    TEMP_DOCKERFILE_BASE="Dockerfile.base.temp"
    
    # Set the appropriate base image based on architecture
    if [ "$ARM_MODE" = true ]; then
        # Check if ARM-specific base image exists, otherwise use a fallback
        BASE_SWEB_IMAGE="aorwall/sweb.base.py.arm64:latest"
        echo "Checking if ARM base image ${BASE_SWEB_IMAGE} exists..."
        if ! docker manifest inspect ${BASE_SWEB_IMAGE} >/dev/null 2>&1; then
            echo "ARM base image not found, using fallback approach with ubuntu:22.04"
            BASE_SWEB_IMAGE="ubuntu:22.04"
            USE_FALLBACK_BASE=true
        else
            echo "ARM base image found: ${BASE_SWEB_IMAGE}"
            USE_FALLBACK_BASE=false
        fi
    else
        BASE_SWEB_IMAGE="aorwall/sweb.base.py.x86_64:latest"
        USE_FALLBACK_BASE=false
    fi
    
    # Create temporary Dockerfile with correct base image and UV handling
    if [ "$USE_FALLBACK_BASE" = true ]; then
        cat > ${TEMP_DOCKERFILE_BASE} << EOL
FROM ${BASE_SWEB_IMAGE}

# Install basic dependencies for fallback base
RUN apt-get update && apt-get install -y \\
    git \\
    curl \\
    python3 \\
    python3-pip \\
    python3-venv \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Ensure the installed binary is on the PATH
ENV PATH="/root/.local/bin/:$PATH"

RUN uv python install 3.12 --managed-python

RUN mkdir -p /data/nltk_data
ENV NLTK_DATA=/data/nltk_data

# Add a build arg that can be used to invalidate cache for git operations
RUN git clone https://github.com/aorwall/moatless-tools.git -b main /opt/moatless
RUN git clone https://github.com/aorwall/moatless-tree-search.git -b docker /opt/components

COPY docker/update-moatless.sh /usr/local/bin/update-moatless.sh
COPY docker/update-components.sh /usr/local/bin/update-components.sh

ENV MOATLESS_DIR=/data/moatless
ENV MOATLESS_COMPONENTS_PATH=/opt/components

WORKDIR /opt/moatless

# Install dependencies with better error handling for different architectures
RUN --mount=type=cache,target=/root/.cache/uv \\
    uv sync --frozen --compile-bytecode --all-extras || \\
    (echo "Initial sync failed, trying without frozen..." && uv sync --compile-bytecode --all-extras)
EOL
    else
        cat > ${TEMP_DOCKERFILE_BASE} << EOL
FROM ${BASE_SWEB_IMAGE}

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Ensure the installed binary is on the PATH
ENV PATH="/root/.local/bin/:$PATH"

RUN uv python install 3.12 --managed-python

RUN mkdir -p /data/nltk_data
ENV NLTK_DATA=/data/nltk_data

# Add a build arg that can be used to invalidate cache for git operations
RUN git clone https://github.com/aorwall/moatless-tools.git -b main /opt/moatless
RUN git clone https://github.com/aorwall/moatless-tree-search.git -b docker /opt/components

COPY docker/update-moatless.sh /usr/local/bin/update-moatless.sh
COPY docker/update-components.sh /usr/local/bin/update-components.sh

ENV MOATLESS_DIR=/data/moatless
ENV MOATLESS_COMPONENTS_PATH=/opt/components

WORKDIR /opt/moatless

# Install dependencies with better error handling for different architectures
RUN --mount=type=cache,target=/root/.cache/uv \\
    uv sync --frozen --compile-bytecode --all-extras || \\
    (echo "Initial sync failed, trying without frozen..." && uv sync --compile-bytecode --all-extras)
EOL
    fi
    
    if [ "$PUSH_IMAGES" = true ]; then
        docker buildx build --platform ${DOCKER_PLATFORM} -t ${BASE_IMAGE_NAME} -f ${TEMP_DOCKERFILE_BASE} --push --no-cache .
    else
        docker buildx build --platform ${DOCKER_PLATFORM} -t ${BASE_IMAGE_NAME} -f ${TEMP_DOCKERFILE_BASE} --load --no-cache .
    fi

    # Clean up temporary file
    rm -f ${TEMP_DOCKERFILE_BASE}

    if [ $? -ne 0 ]; then
        echo "Error: Failed to build base image"
        exit 1
    fi

    if [ "$PUSH_IMAGES" = true ]; then
        echo "Successfully built and pushed ${ARCH_SUFFIX} base image"
    else
        echo "${ARCH_SUFFIX} base image built but not pushed"
    fi
else
    echo "Skipping base image build (--no-base specified)"
    # Still ensure buildx is set up for instance builds
    echo "Setting up buildx builder for ${ARCH_SUFFIX} builds..."
    docker buildx create --use --name multiarch-builder --driver docker-container --platform ${DOCKER_PLATFORM} || true
fi

# Create directory for temporary files
mkdir -p .moatless_index_store

# Process each instance
for INSTANCE_ID in $INSTANCE_IDS; do
    echo "Processing instance: $INSTANCE_ID"
    
    # Check if the instance JSON file exists
    if [ ! -f "instances/${INSTANCE_ID}.json" ]; then
        echo "Error: Instance JSON file not found: instances/${INSTANCE_ID}.json"
        exit 1
    fi
    
    # Download the index if it doesn't exist
    download_index "$INSTANCE_ID"
    if [ $? -ne 0 ]; then
        echo "Warning: Failed to ensure index for $INSTANCE_ID, skipping..."
        continue
    fi
    
    # Set base image name based on mode
    if [ "$SWEGYM_MODE" = true ]; then
        DOCKER_BASE_IMAGE=$(echo $INSTANCE_ID | sed 's/__/_s_/g' | tr '[:upper:]' '[:lower:]')
        DOCKER_BASE_PREFIX="xingyaoww/sweb.eval.${ARCH_SUFFIX}."
    else
        DOCKER_BASE_IMAGE=$(echo $INSTANCE_ID | sed 's/__/_1776_/g' | tr '[:upper:]' '[:lower:]')
        DOCKER_BASE_PREFIX="swebench/sweb.eval.${ARCH_SUFFIX}."
    fi

    # Create the target image name in the required format - using first part before __
    DOCKER_TARGET_IMAGE=$(echo "aorwall/sweb.eval.${ARCH_SUFFIX}.${INSTANCE_ID%%__*}_moatless_${INSTANCE_ID#*__}" | tr '[:upper:]' '[:lower:]')

    # Create temporary Dockerfile
    BASE_IMAGE_NAME="aorwall/moatless.swebench.base.${ARCH_SUFFIX}"
    cat > Dockerfile.temp << EOL
FROM ${DOCKER_BASE_PREFIX}$DOCKER_BASE_IMAGE

WORKDIR /opt/moatless
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy libraries and moatless directory from base image
COPY --from=${BASE_IMAGE_NAME} /root/.local/share/uv /root/.local/share/uv
COPY --from=${BASE_IMAGE_NAME} /opt/moatless /opt/moatless

# Git configuration
RUN git config --global --add safe.directory /testbed

# Create necessary directories
RUN mkdir -p /testbed/.moatless

ENV VIRTUAL_ENV=""

# Copy instance specific files
COPY instances/${INSTANCE_ID}.json /data/instance.json
COPY .moatless_index_store/${INSTANCE_ID} /data/index_store

ENV REPO_PATH=/testbed
ENV INDEX_STORE_DIR=/data/index_store/${INSTANCE_ID}
ENV INSTANCE_PATH=/data/instance.json

EOL

    # Copy index store files from environment variable directory
    echo "Copying index store files for $INSTANCE_ID from $INDEX_STORE_DIR"
    mkdir -p ".moatless_index_store/${INSTANCE_ID}"
    cp -r "${INDEX_STORE_DIR}/${INSTANCE_ID}"/* .moatless_index_store/${INSTANCE_ID}/ 2>/dev/null
    
    if [ ! -d ".moatless_index_store/${INSTANCE_ID}" ] || [ -z "$(ls -A .moatless_index_store/${INSTANCE_ID} 2>/dev/null)" ]; then
        echo "Warning: No index store files found for $INSTANCE_ID after copying, skipping..."
        continue
    fi

    # Build the Docker image
    echo "Building Docker image: ${DOCKER_TARGET_IMAGE}"
    docker build --platform ${DOCKER_PLATFORM} -t ${DOCKER_TARGET_IMAGE} -f Dockerfile.temp .
    
    if [ $? -ne 0 ]; then
        echo "Error: Failed to build image for $INSTANCE_ID"
        exit 1
    fi

    # Push the Docker image
    if [ "$PUSH_IMAGES" = true ]; then
        echo "Pushing Docker image: ${DOCKER_TARGET_IMAGE}"
        docker push ${DOCKER_TARGET_IMAGE}
        
        if [ $? -ne 0 ]; then
            echo "Error: Failed to push image for $INSTANCE_ID"
            exit 1
        else
            echo "Successfully built and pushed image for $INSTANCE_ID"
        fi
    else
        echo "Image for $INSTANCE_ID built but not pushed"
    fi
done

# Clean up temporary files
rm -f Dockerfile.temp
rm -f Dockerfile.base.temp
rm -rf .moatless_index_store

if [ "$PUSH_IMAGES" = true ]; then
    echo "All instances processed and images pushed"
else
    echo "All instances processed (images built but not pushed)"
fi 