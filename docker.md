# Moatless Docker Setup

This directory contains Docker configuration files to run the Moatless API in a containerized environment.

## Prerequisites

- Docker and Docker Compose installed on your system
- Git (to clone the repository)

## Quick Start

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd moatless-api
   ```

2. Set up environment variables:
   ```bash
   cp .env.example .env
   ```
   
   Edit the `.env` file to set your API keys and other configuration options, including the required `MOATLESS_DIR` variable:
   ```
   MOATLESS_DIR=/path/to/your/moatless/data
   ```
   
   **Note**: `MOATLESS_DIR` specifies the directory where Moatless will store configuration files and trajectory data. This directory will be mounted as a volume in the Docker containers.

3. Start the services:
   ```bash
   make run
   ```

   This command will:
   - Start Redis (for event bus communication)
   - Start the Moatless API server (with Docker runner configured)

4. Access the API at http://localhost

## Source Code Configuration

The Moatless setup can override existing source code in two different places:

1. **API Container**: Needs moatless module code at `/app/moatless`
   - Set by `MOATLESS_SOURCE_DIR` (default: `./moatless`)

2. **Job Containers**: Optionally uses moatless source at `/opt/moatless`
   - Set by `MOATLESS_RUNNER_MOUNT_SOURCE_DIR` or `MOATLESS_HOST_RUNNER_SOURCE_DIR`

This separation allows you to use different source code for the API and the spawned job containers.

## Configuration

### Environment Variables

The following environment variables can be configured in the `.env` file:

- `MOATLESS_DIR`: **[REQUIRED]** Directory where Moatless stores configuration files and trajectory data (mounted as volume)
- `MOATLESS_REPO_DIR`: Directory for repository data
- `MOATLESS_SOURCE_DIR`: Path to moatless module code for API container
- `MOATLESS_RUNNER`: Set to `docker` to use the Docker runner
- `MOATLESS_RUNNER_MOUNT_SOURCE_DIR`: Path to source code for job containers
- `MOATLESS_COMPONENTS_PATH`: Path to components (optional)
- `REDIS_URL`: URL for Redis connection (used for event bus)
- API keys and other configuration options

### Docker Runner Configuration

The Docker runner configuration allows the API service to spawn Docker containers for jobs:

1. The API container needs access to the Docker socket (`/var/run/docker.sock`)
2. Source code mounting for job containers is **optional**:
   - You can set `MOATLESS_RUNNER_MOUNT_SOURCE_DIR` in your `.env` file to point to your source code
   - Job containers will have this code available at `/opt/moatless`

### Redis Event Bus

The Redis event bus enables communication between the API and job containers:

1. Redis stores events and provides pub/sub functionality
2. The API connects to Redis using the `REDIS_URL` environment variable
3. Job containers inherit the same Redis connection information

## Directory Structure

- `Dockerfile`: Defines the Docker image for the API
- `docker-compose.yml`: Defines the services and their configuration
- `.env.example`: Template for environment variables
- `data/`: Directory for mounted volumes
  - `moatless/`: Moatless data directory
- `moatless/`: Source code directory mounted into the API container

## Makefile Targets

- `make setup`: Set up directories and create `.env` file
- `make run`: Start services in detached mode
- `make dev`: Start services in interactive mode with logs
- `make stop`: Stop all services
- `make logs`: View logs for all services
- `make status`: Check status of all services

## Logs

To view logs from the containers:

```bash
# All services
docker-compose logs

# Specific service
docker-compose logs api
docker-compose logs redis
```

## Docker Job Containers

The API service creates job containers using the Docker runner:

1. Each job gets its own container with isolated execution environment
2. Job containers can access mounted volumes:
   - Moatless data directory at `/data/moatless`
   - Components directory at `/opt/components` (if configured)
   - Source code at `/opt/moatless` (if configured)
3. Job status and logs can be retrieved through the API
4. Containers are labeled with project ID and trajectory ID for easy filtering

To see running job containers:

```bash
docker ps --filter "label=moatless.managed=true"
```

## Stopping the Services

```bash
make stop
```