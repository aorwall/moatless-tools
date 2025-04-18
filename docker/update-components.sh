#!/bin/bash
set -e

# Default values
BRANCH="docker"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --branch)
      BRANCH="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

echo "🔄 Updating moatless-tree-search repository to latest from branch '$BRANCH'"

# Navigate to components directory
cd /opt/components

# Update the repository
echo "📂 Fetching latest changes..."
git fetch origin
git checkout $BRANCH
git pull origin $BRANCH

# Update dependencies if needed
echo "📦 Updating dependencies..."
uv sync --frozen --compile-bytecode

echo "✅ Update complete! Repository is now at: $(git rev-parse --short HEAD)" 