import os
from pathlib import Path


def get_moatless_dir() -> Path:
    """Get the moatless directory."""
    moatless_dir = Path(os.getenv("MOATLESS_DIR", ".moetless"))
    if not moatless_dir.exists():
        moatless_dir.mkdir(parents=True, exist_ok=True)
    return moatless_dir

def get_moatless_trajectory_dir() -> Path:
    """Get the moatless trajectory directory."""
    trajectory_dir = get_moatless_dir() / "trajs"
    if not trajectory_dir.exists():
        trajectory_dir.mkdir(parents=True, exist_ok=True)
    return trajectory_dir
