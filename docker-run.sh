#!/bin/bash

# Docker run script for Moatless

set -e

# Function to display help
show_help() {
    echo "Moatless Docker Environment"
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  start       Start all services"
    echo "  stop        Stop all services"
    echo "  restart     Restart all services"
    echo "  logs        View logs from all services"
    echo "  status      Check status of services"
    echo "  scale N     Scale workers to N instances"
    echo "  build       Rebuild Docker images"
    echo "  help        Show this help message"
    echo ""
}

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running or not accessible."
    exit 1
fi

# Process commands
case "$1" in
    start)
        echo "Starting Moatless services..."
        docker-compose up -d
        echo "Services started. API available at http://localhost:8000"
        ;;
    stop)
        echo "Stopping Moatless services..."
        docker-compose down
        echo "Services stopped."
        ;;
    restart)
        echo "Restarting Moatless services..."
        docker-compose restart
        echo "Services restarted."
        ;;
    logs)
        echo "Showing logs (press Ctrl+C to exit)..."
        docker-compose logs -f
        ;;
    status)
        echo "Service status:"
        docker-compose ps
        ;;
    scale)
        if [ -z "$2" ]; then
            echo "Error: Please specify the number of worker instances."
            echo "Usage: $0 scale N"
            exit 1
        fi
        echo "Scaling workers to $2 instances..."
        docker-compose up -d --scale worker=$2
        echo "Workers scaled to $2 instances."
        ;;
    build)
        echo "Rebuilding Docker images..."
        docker-compose build
        echo "Images rebuilt."
        ;;
    help|*)
        show_help
        ;;
esac 