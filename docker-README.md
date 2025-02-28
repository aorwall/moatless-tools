# Moatless Docker Setup

This directory contains Docker configuration files to run the Moatless API and workers in a containerized environment.

## Prerequisites

- Docker and Docker Compose installed on your system
- Git (to clone the repository)

## Quick Start

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd moatless-tools
   ```

2. Set up environment variables:
   ```bash
   cp .env.docker .env
   ```
   
   Edit the `.env` file to set your API keys and other configuration options.

3. Create data directories:
   ```bash
   mkdir -p data/moatless data/repos data/index_stores
   ```

4. Start the services:
   ```bash
   docker-compose up -d
   ```

   This will start:
   - Redis server
   - Moatless API server
   - Moatless workers (default: 2 workers)

5. Access the API at http://localhost:8000

## Configuration

### Environment Variables

The following environment variables can be configured in the `.env` file:

- `MOATLESS_DIR`: Directory for Moatless data (mounted as volume)
- `MOATLESS_REPO_DIR`: Directory for repositories (mounted as volume)
- `INDEX_STORE_DIR`: Directory for index stores (mounted as volume)
- `WORKER_COUNT`: Number of worker instances to run
- `REDIS_URL`: URL for Redis connection
- API keys and other configuration options

### Scaling Workers

You can adjust the number of workers by changing the `WORKER_COUNT` environment variable in the `.env` file.

Alternatively, you can scale workers dynamically:

```bash
docker-compose up -d --scale worker=4
```

## Directory Structure

- `Dockerfile`: Defines the Docker image for both API and workers
- `docker-compose.yml`: Defines the services and their configuration
- `.env.docker`: Template for environment variables
- `data/`: Directory for mounted volumes
  - `moatless/`: Moatless data directory
  - `repos/`: Repository directory
  - `index_stores/`: Index stores directory

## Logs

To view logs from the containers:

```bash
# All services
docker-compose logs

# Specific service
docker-compose logs api
docker-compose logs worker
```

## Stopping the Services

```bash
docker-compose down
```

To remove volumes as well:

```bash
docker-compose down -v
```

## Troubleshooting

### Redis Connection Issues

If the API or workers can't connect to Redis, ensure that:
1. Redis container is running: `docker-compose ps`
2. The `REDIS_URL` environment variable is set correctly
3. The Redis service is healthy: `docker-compose logs redis`

### Volume Mounting Issues

If you encounter issues with volume mounting:
1. Ensure the host directories exist
2. Check permissions on the host directories
3. Verify the paths in the `.env` file

### Worker Not Processing Jobs

If workers are not processing jobs:
1. Check worker logs: `docker-compose logs worker`
2. Ensure Redis connection is working
3. Verify that the API is submitting jobs correctly 