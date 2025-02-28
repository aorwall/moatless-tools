#!/bin/bash

# Docker setup script for Moatless

set -e

echo "Setting up Moatless Docker environment..."

# Create data directories
echo "Creating data directories..."
mkdir -p data/moatless data/repos data/index_stores

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.docker .env
    echo "Please edit the .env file to set your API keys and configuration."
else
    echo ".env file already exists, skipping..."
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. Please install Docker and Docker Compose."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "Docker Compose is not installed. Please install Docker Compose."
    exit 1
fi

echo "Environment setup complete!"
echo ""
echo "To start the services, run:"
echo "  docker-compose up -d"
echo ""
echo "To view logs, run:"
echo "  docker-compose logs -f"
echo ""
echo "To stop the services, run:"
echo "  docker-compose down"
echo ""
echo "The API will be available at http://localhost:8000" 