import os
from pathlib import Path

from moatless.context_data import current_project_id


def get_moatless_dir() -> Path:
    """Get the moatless directory."""
    return Path(os.getenv("MOATLESS_DIR", ".moatless"))


def get_moatless_trajectories_dir(evaluation_name: str | None = None) -> Path:
    """Get the moatless trajectories directory."""

    if evaluation_name:
        eval_name = evaluation_name
    else:
        eval_name = current_project_id.get()

    if eval_name:
        # If evaluation context exists, use projects/eval_name/trajs
        trajectories_dir = get_moatless_dir() / "projects" / eval_name / "trajs"
    else:
        # Otherwise use default trajs directory
        trajectories_dir = get_moatless_dir() / "projects" / "default" / "trajs"

    if not trajectories_dir.exists():
        trajectories_dir.mkdir(parents=True, exist_ok=True)
    return trajectories_dir


def get_moatless_trajectory_dir(trajectory_id: str | None = None, project_id: str | None = None) -> Path:
    """Get the moatless trajectory directory."""
    trajectory_base_dir = get_moatless_trajectories_dir(project_id)
    if not trajectory_id:
        return trajectory_base_dir
    trajectory_dir = trajectory_base_dir / trajectory_id
    if not trajectory_dir.exists():
        trajectory_dir.mkdir(parents=True, exist_ok=True)
    return trajectory_dir
