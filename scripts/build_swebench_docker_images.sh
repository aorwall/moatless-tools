#!/bin/bash

# Check if dataset_file is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <dataset_file>"
    echo "Example: $0 moatless/evaluation/datasets/verified_mini_dataset.json"
    exit 1
fi

DATASET_FILE=$1

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
    echo "Error: Index store directory $INDEX_STORE_DIR does not exist"
    exit 1
fi

# Check if instances directory exists, if not, run create_dataset_index.py
if [ ! -d "instances" ]; then
    echo "instances directory not found, running create_dataset_index.py"
    python scripts/create_dataset_index.py
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
docker build -t aorwall/moatless.swebench.base.x86_64 -f docker/Dockerfile.base .

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
    
    # Replace __ with _1776_ in instance_id for the base image name
    DOCKER_BASE_IMAGE=$(echo $INSTANCE_ID | sed 's/__/_1776_/g')
    # Create the target image name in the required format - using first part before __
    DOCKER_TARGET_IMAGE="aorwall/sweb.eval.x86_64.${INSTANCE_ID%%__*}_moatless_${INSTANCE_ID#*__}"

    # Create temporary Dockerfile
    cat > Dockerfile.temp << EOL
FROM swebench/sweb.eval.x86_64.$DOCKER_BASE_IMAGE

# Set pip cache directory and configuration
ENV PIP_CACHE_DIR=/root/.cache/pip
ENV PIP_NO_CACHE_DIR=false
ENV PYTHONUNBUFFERED=1

# Copy Python libraries and moatless directory from base image
COPY --from=aorwall/moatless.swebench.base.x86_64 /usr/local/lib/python3.10/ /usr/local/lib/python3.10/

# Git configuration
RUN git config --global --add safe.directory /testbed

# Create necessary directories
RUN mkdir -p /testbed/.moatless

# Copy instance specific files
COPY instances/${INSTANCE_ID}.json /data/instance.json
COPY .moatless_index_store/${INSTANCE_ID} /data/index_store

ENV REPO_PATH=/testbed
ENV INDEX_STORE_DIR=/data/index_store/${INSTANCE_ID}
ENV INSTANCE_PATH=/data/instance.json

RUN mkdir -p /opt/moatless
WORKDIR /opt/moatless
EOL

    # Copy index store files from environment variable directory
    echo "Copying index store files for $INSTANCE_ID from $INDEX_STORE_DIR"
    cp -r "${INDEX_STORE_DIR}/${INSTANCE_ID}" .moatless_index_store/ 2>/dev/null
    
    if [ ! -d ".moatless_index_store/${INSTANCE_ID}" ]; then
        echo "Warning: No index store files found for $INSTANCE_ID, skipping..."
        continue
    fi

    # Build the Docker image
    echo "Building Docker image: ${DOCKER_TARGET_IMAGE}"
    docker build -t ${DOCKER_TARGET_IMAGE} -f Dockerfile.temp .
    
    if [ $? -ne 0 ]; then
        echo "Error: Failed to build image for $INSTANCE_ID"
        continue
    fi

    # Push the Docker image
    echo "Pushing Docker image: ${DOCKER_TARGET_IMAGE}"
    docker push ${DOCKER_TARGET_IMAGE}
    
    if [ $? -ne 0 ]; then
        echo "Error: Failed to push image for $INSTANCE_ID"
    else
        echo "Successfully built and pushed image for $INSTANCE_ID"
    fi
done

# Clean up temporary files
rm -f Dockerfile.temp
rm -rf .moatless_index_store

echo "All instances processed" 