import contextvars
import os
from pathlib import Path

# A context variable to hold the current node id (or any other context you need)
moatless_dir = contextvars.ContextVar[str | None]("moatless_dir", default=None)
current_node_id = contextvars.ContextVar[int | None]("current_node_id", default=None)
current_trajectory_id = contextvars.ContextVar[str | None]("current_trajectory_id", default=None)
current_project_id = contextvars.ContextVar[str | None]("current_project_id", default=None)

current_trajectory_dir = contextvars.ContextVar[Path | None]("current_trajectory_dir", default=None)


def get_moatless_dir() -> Path:
    """Get the moatless directory."""
    dir_value = moatless_dir.get()
    if not dir_value:
        dir_path = Path(os.getenv("MOATLESS_DIR", ".moatless"))
    else:
        dir_path = Path(dir_value)

    if not dir_path.exists():
        dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_projects_dir() -> Path:
    """Get the moatless projects directory."""
    return get_moatless_dir() / "projects"


def get_project_dir(project_id: str | None = None) -> Path:
    """Get the moatless project directory."""

    if project_id:
        project_id = project_id
    else:
        project_id = current_project_id.get()

    if project_id:
        project_dir = get_moatless_dir() / "projects" / project_id
    else:
        project_dir = get_moatless_dir() / "projects" / "default"

    if not project_dir.exists():
        project_dir.mkdir(parents=True, exist_ok=True)
    return project_dir


def get_trajectory_dir(trajectory_id: str | None = None, project_id: str | None = None) -> Path:
    """Get the moatless trajectory directory."""

    trajectory_dir_value = current_trajectory_dir.get()
    if trajectory_dir_value:
        return trajectory_dir_value

    trajectory_base_dir = get_project_dir(project_id)
    if not trajectory_id:
        return trajectory_base_dir
    trajectory_dir = trajectory_base_dir / trajectory_id
    if not trajectory_dir.exists():
        trajectory_dir.mkdir(parents=True, exist_ok=True)
    return trajectory_dir
