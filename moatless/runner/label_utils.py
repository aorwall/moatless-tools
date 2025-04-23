"""Utility functions for creating and managing container labels."""

import asyncio
import hashlib
from typing import Callable
from typing import Dict, List, Optional


def create_job_args(project_id: str, trajectory_id: str, job_func: Callable, node_id: Optional[int] = None):
    func_module = job_func.__module__
    func_name = job_func.__name__

    if node_id:
        args = f"(project_id='{project_id}', trajectory_id='{trajectory_id}', node_id={node_id})"
    else:
        args = f"(project_id='{project_id}', trajectory_id='{trajectory_id}')"

    is_async = asyncio.iscoroutinefunction(job_func)

    # Create the appropriate Python code for uv run
    if is_async:
        return (
            f"import logging\n"
            f"logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')\n"
            f"logging.debug('Starting job execution')\n"
            f"import asyncio\n"
            f"from {func_module} import {func_name}\n"
            f"logging.debug('Imports completed, starting execution')\n"
            f"asyncio.run({func_name}{args})\n"
            f"logging.debug('Job execution completed')"
        )
    else:
        return (
            f"import logging\n"
            f"logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')\n"
            f"logging.debug('Starting job execution')\n"
            f"from {func_module} import {func_name}\n"
            f"logging.debug('Imports completed, starting execution')\n"
            f"{func_name}{args}\n"
            f"logging.debug('Job execution completed')"
        )


def create_resource_id(project_id: str, trajectory_id: str, prefix: str = "run") -> str:
    """Create a resource ID that's valid for both Kubernetes jobs and Docker containers.

    This will generate an ID with the format: {prefix}-{proj_prefix}-{traj_suffix}-{hash}

    Args:
        project_id: The project ID (required, must not be empty)
        trajectory_id: The trajectory ID (required, must not be empty)
        prefix: Prefix for the resource name ("run" for Kubernetes jobs, "moatless" for Docker containers)

    Returns:
        A valid resource ID string

    Raises:
        ValueError: If project_id or trajectory_id is empty
    """
    # Validate required parameters
    if not project_id:
        raise ValueError("project_id must be provided and not empty")
    if not trajectory_id:
        raise ValueError("trajectory_id must be provided and not empty")

    # Pre-process: convert to lowercase and handle non-ASCII characters
    project_lower = project_id.lower()
    project_ascii = "".join(c if c.isascii() else "-" for c in project_lower)

    traj_lower = trajectory_id.lower()
    traj_ascii = "".join(c if c.isascii() else "-" for c in traj_lower)

    # Use sanitize_label for remaining sanitization
    sanitized_project = sanitize_label(project_ascii)
    sanitized_traj = sanitize_label(traj_ascii)

    # Replace underscores and dots with dashes (as sanitize_label keeps them)
    sanitized_project = sanitized_project.replace("_", "-").replace(".", "-")
    sanitized_traj = sanitized_traj.replace("_", "-").replace(".", "-")

    # Create hash from original IDs
    hash_input = f"{project_id}:{trajectory_id}"
    hash_value = hashlib.md5(hash_input.encode()).hexdigest()[:8]

    # Determine the parts of the ID
    proj_prefix = sanitized_project[:16]  # First 16 chars of project id

    # Calculate remaining space
    # Format: prefix-proj_prefix-traj_suffix-hash
    # Count separators: 3 hyphens
    max_total_length = 63  # Maximum length for both Kubernetes and Docker
    remaining_space = max_total_length - len(prefix) - len(proj_prefix) - len(hash_value) - 3

    # Determine how many chars we can use for trajectory suffix
    traj_chars = min(remaining_space, len(sanitized_traj))
    traj_suffix = sanitized_traj[:traj_chars]

    # Construct the resource ID
    resource_id = f"{prefix}-{proj_prefix}-{traj_suffix}-{hash_value}"

    # Clean up the resource ID
    # Handle potential double hyphens from concatenation
    resource_id = resource_id.replace("--", "-x-")

    # Ensure it doesn't start or end with a dash
    if resource_id.startswith(f"{prefix}--"):
        resource_id = f"{prefix}-x{resource_id[len(prefix)+2:]}"
    if resource_id.endswith("-"):
        resource_id = f"{resource_id[:-1]}x"

    # Maximum length check (should already be handled by our calculations, but as a safeguard)
    resource_id = resource_id[:63]

    return resource_id


def sanitize_label(value: str) -> str:
    """Sanitize a string to be a valid container label.

    Args:
        value: The string to sanitize

    Returns:
        A valid label string (max 63 chars, alphanumeric, '-', '_', or '.')
    """
    # Replace invalid characters with dashes
    clean_id = "".join(c if c.isalnum() or c in ["-", "_", "."] else "-" for c in value)

    # Normalize multiple dashes into a single dash
    while "--" in clean_id:
        clean_id = clean_id.replace("--", "-")

    # Ensure it starts and ends with alphanumeric character
    if clean_id and not clean_id[0].isalnum():
        clean_id = "x" + clean_id[1:]
    if clean_id and not clean_id[-1].isalnum():
        clean_id = clean_id[:-1] + "x"

    # Ensure it's not too long (63 character limit)
    clean_id = clean_id[:63]

    # Final check to ensure the truncated string ends with an alphanumeric character
    if clean_id and not clean_id[-1].isalnum():
        clean_id = clean_id[:-1] + "x"

    return clean_id


def get_project_label(project_id: str) -> str:
    """Get a valid label for a project ID.

    Args:
        project_id: The project ID

    Returns:
        A valid label string
    """
    return sanitize_label(project_id)


def get_trajectory_label(trajectory_id: str) -> str:
    """Get a valid label for a trajectory ID.

    Args:
        trajectory_id: The trajectory ID

    Returns:
        A valid label string
    """
    return sanitize_label(trajectory_id)


def create_labels(
    project_id: str,
    trajectory_id: str,
    func_name: Optional[str] = None,
) -> Dict[str, str]:
    """Create labels for a job."""
    project_label = get_project_label(project_id)
    trajectory_label = get_trajectory_label(trajectory_id)

    labels = {
        "app": "moatless-worker",
        "project_id": project_label,
        "trajectory_id": trajectory_label,
        "moatless.managed": "true",
    }

    if func_name:
        labels["function"] = func_name

    return labels


def create_annotations(project_id: str, trajectory_id: str, func_name: Optional[str] = None) -> Dict[str, str]:
    """Create annotations for a job (labels and annotations).

    Args:
        project_id: The project ID
        trajectory_id: The trajectory ID
        func_name: Function name to execute (optional)

    Returns:
        Dictionary of annotations
    """
    # Store original values in annotations
    annotations = {
        "moatless.ai/project-id": project_id,
        "moatless.ai/trajectory-id": trajectory_id,
    }

    return annotations


def create_docker_label_args(labels: Dict[str, str]) -> List[str]:
    """Convert a dictionary of labels to Docker command-line arguments.

    Args:
        labels: Dictionary of label key-value pairs

    Returns:
        List of Docker command-line arguments for labels
    """
    args = []
    for key, value in labels.items():
        args.extend(["--label", f"{key}={value}"])
    return args
