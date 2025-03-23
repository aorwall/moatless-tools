import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from opentelemetry import trace

from moatless.context_data import current_node_id, current_project_id, current_trajectory_id, get_moatless_dir
from moatless.events import BaseEvent

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("moatless.runner")


def emit_event(evaluation_name: str, instance_id: str, scope: str, event_type: str, data: Any = None):
    """Emit evaluation event."""
    event = BaseEvent(
        project_id=evaluation_name, trajectory_id=instance_id, scope=scope, event_type=event_type, data=data
    )

    from moatless.settings import event_bus

    try:
        run_async(event_bus.publish(event))
    except Exception as e:
        logger.error(f"Failed to publish event {event_type}: {e}")


def run_async(coro, span_name: str | None = None):
    """
    Helper to run coroutines in synchronous context while preserving trace context.
    """
    current_span = trace.get_current_span()

    context_data = {
        "moatless_dir": get_moatless_dir(),
        "current_node_id": current_node_id.get(),
        "current_trajectory_id": current_trajectory_id.get(),
        "current_project_id": current_project_id.get(),
    }

    # Create wrapper to run coroutine with trace context
    async def _run_with_context():
        try:
            # Restore context data
            tokens = []
            if "current_trajectory_id" in context_data:
                current_trajectory_id.set(context_data["current_trajectory_id"])
            if "current_project_id" in context_data:
                current_project_id.set(context_data["current_project_id"])

            try:
                # Create a new span if name is provided, otherwise just run with current context
                if span_name:
                    with tracer.start_as_current_span(span_name) as span:
                        return await coro
                else:
                    return await coro
            finally:
                # Reset context vars
                for var, token in tokens:
                    var.reset(token)

        except Exception as e:
            if current_span:
                current_span.record_exception(e)
            raise

    # Get or create event loop
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # Run coroutine with context
    return loop.run_until_complete(_run_with_context())


def setup_job_logging(log_path: Path) -> list[logging.Handler]:
    """Set up logging for a job and return the original handlers for cleanup.

    Args:
        log_path: Path to the log file


    Returns:
        List of original handlers that should be restored after job completion
    """

    log_dir = log_path.parent
    log_dir.mkdir(parents=True, exist_ok=True)

    # Set root logger to WARN to filter out INFO from non-moatless packages
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    logging.getLogger("azure").setLevel(logging.WARNING)
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)

    original_handlers = root_logger.handlers[:]
    for handler in original_handlers:
        root_logger.removeHandler(handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    file_handler = logging.FileHandler(str(log_path))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    return original_handlers


def cleanup_job_logging(original_handlers: list[logging.Handler]) -> None:
    """Clean up job-specific logging and restore original handlers.

    Args:
        original_handlers: List of handlers to restore
    """
    if original_handlers:
        root_logger = logging.getLogger()
        # Remove job-specific handlers
        for handler in root_logger.handlers[:]:
            handler.close()
            root_logger.removeHandler(handler)
        # Restore original handlers
        for handler in original_handlers:
            root_logger.addHandler(handler)
