# Docker Runner for Moatless

This module provides a Docker-based implementation of the Moatless BaseRunner interface. It allows running Moatless jobs in Docker containers without requiring a Kubernetes cluster.

## Features

- Starts jobs as Docker containers
- Uses the same Docker images as the Kubernetes runner
- Minimal dependencies (only requires Docker to be installed)
- Compatible with the Moatless BaseRunner interface
- Supports mounting local source code for development and testing

## Prerequisites

- Docker installed and running on the host machine
- Python 3.8+ with asyncio support

## Usage

### Basic Usage

```python
import asyncio
from moatless.runner.docker_runner import DockerRunner

async def run_example():
    # Create Docker runner
    runner = DockerRunner()
    
    # Start a job
    project_id = "my_project"
    trajectory_id = "django__django-11099"
    func_name = "moatless.runner.job_wrappers.run_evaluation_instance"
    
    success = await runner.start_job(project_id, trajectory_id, func_name)
    
    if success:
        print("Job started successfully!")
        
        # Check job status
        status = await runner.get_job_status(project_id, trajectory_id)
        print(f"Job status: {status}")
        
        # Get logs
        logs = await runner.get_job_logs(project_id, trajectory_id)
        if logs:
            print(f"Logs: {logs}")

# Run the example
asyncio.run(run_example())
```

### Example Script

See the `examples/docker_runner_example.py` script for a complete example of how to use the DockerRunner.

```bash
# Basic usage
python examples/docker_runner_example.py my_project django__django-11099

# Use local moatless source code (for development)
python examples/docker_runner_example.py my_project django__django-11099 --use-local-source
```

### CLI Script

A command-line script is available to quickly run Docker evaluations:

```bash
# Basic usage
./moatless/cli/docker_run.py my_project django__django-11099

# Use local moatless source code (for development)
./moatless/cli/docker_run.py my_project django__django-11099 --use-local

# Pull the Docker image before running
./moatless/cli/docker_run.py my_project django__django-11099 --pull

# Specify the model and flow
./moatless/cli/docker_run.py my_project django__django-11099 --model gpt-4 --flow code
```

## Configuration

The DockerRunner can be configured with the following parameters:

- `job_ttl_seconds`: Time-to-live for completed jobs (default: 3600)
- `timeout_seconds`: Timeout for jobs (default: 3600)
- `volume_mappings`: List of volume mappings in the format "host_path:container_path"
- `moatless_source_dir`: Path to local moatless source code to mount at /opt/moatless in the container

### Using Local Source Code

For development and testing, you can mount your local moatless source code into the container:

```python
# Mount the current directory as the moatless source code
source_dir = os.path.abspath(os.path.dirname(__file__))
runner = DockerRunner(moatless_source_dir=source_dir)
```

This will:
1. Mount your local source directory at `/opt/moatless` in the container
2. Update the `PYTHONPATH` in the container to include `/opt/moatless`

This allows you to make changes to your local code and test them without rebuilding the Docker image.

## Docker Image

The Docker runner uses the same Docker image format as the Kubernetes runner:

```
aorwall/sweb.eval.x86_64.{repo_name}_moatless_{instance_id}
```

For example, for a trajectory ID of `django__django-11099`, the Docker image would be:

```
aorwall/sweb.eval.x86_64.django_moatless_django-11099
```

## Environment Variables

The Docker runner passes the following environment variables to the container:

- `PROJECT_ID`: The project ID
- `TRAJECTORY_ID`: The trajectory ID
- `JOB_FUNC`: The fully qualified function name to execute
- `MOATLESS_DIR`: Path to Moatless data directory
- `MOATLESS_COMPONENTS_PATH`: Path to Moatless components
- `NLTK_DATA`: Path to NLTK data
- `INDEX_STORE_DIR`: Path to index store
- `REPO_DIR`: Path to repository
- `INSTANCE_PATH`: Path to instance file
- `PYTHONPATH`: Updated to include `/opt/moatless` when using local source code

It also passes any environment variables from the host that have names ending with `API_KEY` or starting with `MOATLESS_` or `AWS_`.

## Volume Mappings

The Docker runner supports the following volume mappings:

- `{moatless_dir}:/data/moatless`: Moatless data directory
- `{index_store_dir}:/data/index_store`: Index store directory
- `{nltk_data_dir}:/data/nltk_data`: NLTK data directory
- `{instance_json_path}:/data/instance.json`: Instance file
- `{moatless_source_dir}:/opt/moatless`: Local moatless source code (when moatless_source_dir is specified)

You can add custom volume mappings by setting the `volume_mappings` parameter:

```python
runner = DockerRunner(volume_mappings=[
    "/path/on/host:/path/in/container",
    "/another/path:/another/container/path"
])
```

## Limitations

- The Docker runner stores job information in memory, so it will lose track of jobs if the process is restarted.
- The retry_job method is not fully implemented and always returns False.

## Troubleshooting

### Container Not Starting

If the container is not starting, check the Docker logs:

```bash
docker logs <container_name>
```

### Image Not Found

If the Docker image is not found, you may need to pull it first:

```bash
docker pull aorwall/sweb.eval.x86_64.django_moatless_django-11099
```

### Permission Issues

If you encounter permission issues, make sure the Docker daemon is accessible to the user running the script:

```bash
# Add current user to docker group
sudo usermod -aG docker $USER
# Then log out and log back in
``` 