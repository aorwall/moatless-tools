#!/bin/bash

# Export current user and group IDs as environment variables
export UID=$(id -u)
export GID=$(id -g)

# Print user information
echo "Running Docker containers as UID:GID = $UID:$GID"

# Run docker-compose with all arguments passed to this script
docker compose "$@" 