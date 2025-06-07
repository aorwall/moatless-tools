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

# Check if jq is installed
if ! command -v jq &> /dev/null; then
    echo "Error: jq is required but not installed. Please install jq with 'sudo apt-get install jq' or equivalent"
    exit 1
fi

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

echo "Found $(echo "$INSTANCE_IDS" | wc -l) instances to download indices for"

# Download indices for each instance
for INSTANCE_ID in $INSTANCE_IDS; do
    echo "Processing instance: $INSTANCE_ID"
    
    # Download the index if it doesn't exist
    download_index "$INSTANCE_ID"
    if [ $? -ne 0 ]; then
        echo "Warning: Failed to download index for $INSTANCE_ID, continuing with next..."
        continue
    fi
done

echo "Index store download completed" 