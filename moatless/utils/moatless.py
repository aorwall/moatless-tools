import os
from pathlib import Path
from moatless.context_data import current_evaluation_name


def get_moatless_dir() -> Path:
    """Get the moatless directory."""
    moatless_dir = Path(os.getenv("MOATLESS_DIR", ".moatless"))
    if not moatless_dir.exists():
        moatless_dir.mkdir(parents=True, exist_ok=True)
    return moatless_dir

def get_moatless_trajectories_dir() -> Path:
    """Get the moatless trajectories directory."""
    eval_name = current_evaluation_name.get()
    if eval_name:
        # If evaluation context exists, use evals/eval_name/trajs
        trajectories_dir = get_moatless_dir() / "evals" / eval_name / "trajs"
    else:
        # Otherwise use default trajs directory
        trajectories_dir = get_moatless_dir() / "trajs"
    
    if not trajectories_dir.exists():
        trajectories_dir.mkdir(parents=True, exist_ok=True)
    return trajectories_dir


def get_moatless_trajectory_dir(trajectory_id: str | None = None) -> Path:
    """Get the moatless trajectory directory."""
    trajectory_base_dir = get_moatless_trajectories_dir()
    if not trajectory_id:
        return trajectory_base_dir
    trajectory_dir = trajectory_base_dir / trajectory_id
    if not trajectory_dir.exists():
        trajectory_dir.mkdir(parents=True, exist_ok=True)
    return trajectory_dir

