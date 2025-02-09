import contextvars

# A context variable to hold the current node id (or any other context you need)
current_node_id = contextvars.ContextVar("current_node_id", default=None)
current_trajectory_id = contextvars.ContextVar("current_trajectory_id", default=None)
current_evaluation_name = contextvars.ContextVar("current_evaluation_name", default=None)
