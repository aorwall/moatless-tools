import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, List

from moatless.events import BaseEvent, event_bus

logger = logging.getLogger(__name__)


def emit_event(evaluation_name: str, instance_id: str, scope: str, event_type: str, data: Any = None):
    """Emit evaluation event."""
    event = BaseEvent(
        project_id=evaluation_name, trajectory_id=instance_id, scope=scope, event_type=event_type, data=data
    )

    try:
        run_async(event_bus.publish(event))
    except Exception as e:
        logger.error(f"Failed to publish event {event_type}: {e}")


def run_async(coro):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(coro)


def setup_job_logging(job_type: str, trajectory_dir: Path) -> list[logging.Handler]:
    """Set up logging for a job and return the original handlers for cleanup.

    Args:
        instance_id: The ID of the instance being processed
        job_type: Type of job (run/eval) for log file naming
        trajectory_dir: Directory for trajectory-specific logs

    Returns:
        List of original handlers that should be restored after job completion
    """

    # Set up trajectory-specific logging
    log_dir = trajectory_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    original_handlers = root_logger.handlers[:]
    for handler in original_handlers:
        root_logger.removeHandler(handler)

    # Add console handler for WARN and ERROR
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Add file handler for INFO and above
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{timestamp}_{job_type}.log"
    file_handler = logging.FileHandler(str(log_dir / log_filename))
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
