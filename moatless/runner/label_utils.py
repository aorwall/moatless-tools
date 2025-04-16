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
    
    This will generate an ID with the format: {prefix}-{sanitized_project_id}-{sanitized_trajectory_id}
    
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
        
    # Sanitize project_id: replace special chars and Unicode with dashes
    sanitized_project = project_id.lower()
    # Handle non-ASCII characters by replacing them with dashes
    sanitized_project = ''.join(c if c.isascii() and (c.isalnum() or c == '-') else '-' for c in sanitized_project)
    # Replace underscores, dots, and other special chars with dashes
    sanitized_project = sanitized_project.replace('_', '-').replace('.', '-')
    # Normalize multiple dashes into a single dash
    while '--' in sanitized_project:
        sanitized_project = sanitized_project.replace('--', '-')
    
    # Sanitize trajectory_id using the same approach
    sanitized_traj = trajectory_id.lower()
    # Handle non-ASCII characters by replacing them with dashes
    sanitized_traj = ''.join(c if c.isascii() and (c.isalnum() or c == '-') else '-' for c in sanitized_traj)
    # Replace underscores, dots, and other special chars with dashes
    sanitized_traj = sanitized_traj.replace('_', '-').replace('.', '-')
    # Normalize multiple dashes into a single dash
    while '--' in sanitized_traj:
        sanitized_traj = sanitized_traj.replace('--', '-')
    
    # Calculate max available length
    max_total_length = 63  # Maximum length for both Kubernetes and Docker
    max_id_length = max_total_length - len(prefix) - 2  # -2 for the hyphens
    
    # Determine how much space to give each ID part
    if len(sanitized_project) + len(sanitized_traj) + 1 <= max_id_length:  # +1 for hyphen
        # Both fit completely
        resource_id = f"{prefix}-{sanitized_project}-{sanitized_traj}"
    else:
        # Need to truncate - allocate space fairly between project and trajectory IDs
        # Aim for roughly equal portions but reserve at least 10 chars for each if possible
        min_chars = 10
        
        # Total space available for both IDs together
        available_space = max_id_length - 1  # -1 for the hyphen between them
        
        if available_space >= min_chars * 2:
            # We have enough space for minimum requirements
            # Divide remaining space proportionally based on original lengths
            total_original_length = len(sanitized_project) + len(sanitized_traj)
            
            # Allocate space based on proportions
            project_proportion = len(sanitized_project) / total_original_length
            traj_proportion = len(sanitized_traj) / total_original_length
            
            max_project_length = max(min_chars, int(available_space * project_proportion))
            max_traj_length = available_space - max_project_length
            
            # Adjust if needed to ensure minimum sizes
            if max_traj_length < min_chars:
                max_traj_length = min_chars
                max_project_length = available_space - max_traj_length
            
            # Ensure we haven't exceeded available space
            if max_project_length + max_traj_length > available_space:
                max_project_length = available_space - max_traj_length
        else:
            # Not enough space - just divide available space roughly equally
            max_project_length = available_space // 2
            max_traj_length = available_space - max_project_length
        
        # Truncate if needed
        truncated_project = sanitized_project[:max_project_length]
        truncated_traj = sanitized_traj[:max_traj_length]
        
        resource_id = f"{prefix}-{truncated_project}-{truncated_traj}"
    
    # Clean up the resource ID
    # If starts or ends with a dash, replace with 'x'
    if resource_id.startswith(f"{prefix}--"):
        resource_id = f"{prefix}-x{resource_id[len(prefix)+2:]}"
    if resource_id.endswith('-'):
        resource_id = f"{resource_id[:-1]}x"
    
    # Maximum length check
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

    if func_name:
        annotations["moatless.ai/function"] = func_name

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
