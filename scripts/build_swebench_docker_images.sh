#!/bin/bash

# Check if dataset_file is provided
if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo "Usage: $0 <dataset_file> [swegym]"
    echo "Example: $0 moatless/evaluation/datasets/verified_mini_dataset.json"
    echo "         $0 moatless/evaluation/datasets/verified_mini_dataset.json swegym"
    exit 1
fi

DATASET_FILE=$1
SWEGYM_MODE=false

if [ $# -eq 2 ] && [ "$2" = "swegym" ]; then
    SWEGYM_MODE=true
    echo "Running in swegym mode"
fi

# Check if the dataset file exists
if [ ! -f "$DATASET_FILE" ]; then
    echo "Error: Dataset file $DATASET_FILE not found"
    exit 1
fi

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

# Check if jq is installed
if ! command -v jq &> /dev/null; then
    echo "Error: jq is required but not installed. Please install jq with 'sudo apt-get install jq' or equivalent"
    exit 1
fi

# Build and push the base image first
echo "Building base image: aorwall/moatless.swebench.base.x86_64"
docker build --platform linux/amd64 -t aorwall/moatless.swebench.base.x86_64 -f docker/Dockerfile.base --no-cache .

if [ $? -ne 0 ]; then
    echo "Error: Failed to build base image"
    exit 1
fi

echo "Pushing base image: aorwall/moatless.swebench.base.x86_64"
docker push aorwall/moatless.swebench.base.x86_64

if [ $? -ne 0 ]; then
    echo "Error: Failed to push base image"
    exit 1
fi

echo "Successfully built and pushed base image"

# Extract instance IDs using jq for reliable JSON parsing
INSTANCE_IDS=$(jq -r '.instance_ids[]? // empty' "$DATASET_FILE" 2>/dev/null)

# If the above format fails, try alternate JSON formats
if [ -z "$INSTANCE_IDS" ]; then
    # Try another common format (array of objects with instance_id field)
    INSTANCE_IDS=$(jq -r '.[]?.instance_id? // empty' "$DATASET_FILE" 2>/dev/null)
fi

if [ -z "$INSTANCE_IDS" ]; then
    # Try inspecting the file structure for debugging
    echo "Error: Could not parse instance IDs from $DATASET_FILE"
    echo "File content preview:"
    head -n 20 "$DATASET_FILE"
    exit 1
fi

echo "Found $(echo "$INSTANCE_IDS" | wc -l) instances to build"

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
        DOCKER_BASE_PREFIX="xingyaoww/sweb.eval.x86_64."
    else
        DOCKER_BASE_IMAGE=$(echo $INSTANCE_ID | sed 's/__/_1776_/g' | tr '[:upper:]' '[:lower:]')
        DOCKER_BASE_PREFIX="swebench/sweb.eval.x86_64."
    fi

    # Create the target image name in the required format - using first part before __
    DOCKER_TARGET_IMAGE=$(echo "aorwall/sweb.eval.x86_64.${INSTANCE_ID%%__*}_moatless_${INSTANCE_ID#*__}" | tr '[:upper:]' '[:lower:]')

    # Create temporary Dockerfile
    cat > Dockerfile.temp << EOL
FROM ${DOCKER_BASE_PREFIX}$DOCKER_BASE_IMAGE

WORKDIR /opt/moatless
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy libraries and moatless directory from base image
COPY --from=aorwall/moatless.swebench.base.x86_64 /root/.local/share/uv /root/.local/share/uv
COPY --from=aorwall/moatless.swebench.base.x86_64 /opt/moatless /opt/moatless

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
    docker build --platform linux/amd64 -t ${DOCKER_TARGET_IMAGE} -f Dockerfile.temp .
    
    if [ $? -ne 0 ]; then
        echo "Error: Failed to build image for $INSTANCE_ID"
        exit 1
    fi

    # Push the Docker image
    echo "Pushing Docker image: ${DOCKER_TARGET_IMAGE}"
    docker push ${DOCKER_TARGET_IMAGE}
    
    if [ $? -ne 0 ]; then
        echo "Error: Failed to push image for $INSTANCE_ID"
        exit 1
    else
        echo "Successfully built and pushed image for $INSTANCE_ID"
    fi
done

# Clean up temporary files
rm -f Dockerfile.temp
rm -rf .moatless_index_store

echo "All instances processed" 