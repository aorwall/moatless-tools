import os
from pathlib import Path

from moatless.context_data import moatless_dir, current_project_id


def get_moatless_dir() -> Path:
    """Get the moatless directory."""
    if not moatless_dir.get():
        dir_path = Path(os.getenv("MOATLESS_DIR", ".moatless"))
    else:
        dir_path = Path(moatless_dir.get())
        
    if not dir_path.exists():
        dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path

def get_moatless_trajectories_dir(evaluation_name: str | None = None) -> Path:
    """Get the moatless trajectories directory."""

    if evaluation_name:
        eval_name = evaluation_name
    else:
        eval_name = current_project_id.get()

    if eval_name:
        # If evaluation context exists, use evals/eval_name/trajs
        trajectories_dir = get_moatless_dir() / "evals" / eval_name / "trajs"
    else:
        # Otherwise use default trajs directory
        trajectories_dir = get_moatless_dir() / "trajs"
    
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

