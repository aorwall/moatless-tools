import contextvars

# A context variable to hold the current node id (or any other context you need)
moatless_dir = contextvars.ContextVar("moatless_dir", default=None)
current_node_id = contextvars.ContextVar("current_node_id", default=None)
current_trajectory_id = contextvars.ContextVar("current_trajectory_id", default=None)
current_project_id = contextvars.ContextVar("current_project_id", default=None)
