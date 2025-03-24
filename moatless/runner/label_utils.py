"""Utility functions for creating and managing container labels."""

from datetime import datetime
from typing import Dict, List, Tuple


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
    return clean_id[:63]


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


def create_job_labels(project_id: str, trajectory_id: str, prefix: str = "moatless") -> Dict[str, str]:
    """Create standard labels for a job.

    Args:
        project_id: The project ID
        trajectory_id: The trajectory ID
        prefix: The prefix for label keys (default: "moatless")

    Returns:
        Dictionary of labels
    """
    project_label = get_project_label(project_id)
    trajectory_label = get_trajectory_label(trajectory_id)

    return {
        f"{prefix}.project_id": project_id,
        f"{prefix}.trajectory_id": trajectory_id,
        f"{prefix}.managed": "true",
        f"{prefix}.started_at": datetime.now().isoformat(),
    }


def create_metadata(
    project_id: str, trajectory_id: str, func_name: str = None
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Create metadata for a job (labels and annotations).

    Args:
        project_id: The project ID
        trajectory_id: The trajectory ID
        func_name: Function name to execute (optional)

    Returns:
        Tuple of (labels, annotations)
    """
    # Prepare labels
    project_label = get_project_label(project_id)
    trajectory_label = get_trajectory_label(trajectory_id)

    labels = {
        "app": "moatless-worker",
        "project_id": project_label,
        "trajectory_id": trajectory_label,
    }

    # Store original values in annotations
    annotations = {
        "moatless.ai/project-id": project_id,
        "moatless.ai/trajectory-id": trajectory_id,
    }

    if func_name:
        annotations["moatless.ai/function"] = func_name

    return labels, annotations


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
