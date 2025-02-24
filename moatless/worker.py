from asyncio import Queue
import os
import logging
from pathlib import Path
from moatless.logging_config import setup_logging
from opentelemetry import trace
from rq import Queue, Worker
from rq.job import Job

from moatless.telemetry import (
    instrument,
    restore_trace_context,
    instrument_span,
    setup_telemetry,
)
from moatless.context_data import (
    moatless_dir,
    current_node_id,
    current_trajectory_id,
    current_project_id,
)

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class MoatlessWorker(Worker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        load_dotenv()
        setup_logging(log_dir=Path(f"logs/{self.name}"))
        setup_telemetry()
        
    def execute_job(self, job: Job, queue: Queue):
        """Execute a job with OpenTelemetry context propagation."""
        carrier = job.meta.get("otel_context", {})
        context_data = job.meta.get("context_data", {})

        logger.info(f"job.func_name {job.func_name}")
        logger.info(f"carrier {carrier}")
        logger.info(f"context_data: {context_data}")

        # Restore trace context
        restore_trace_context(carrier)

        # Restore context variables
        context_tokens = []
        # Set context variables and store their tokens
        if "moatless_dir" in context_data:
            context_tokens.append(("moatless_dir", moatless_dir.set(context_data["moatless_dir"])))
        if "current_node_id" in context_data:
            context_tokens.append(("current_node_id", current_node_id.set(context_data["current_node_id"])))
        if "current_trajectory_id" in context_data:
            context_tokens.append(("current_trajectory_id", current_trajectory_id.set(context_data["current_trajectory_id"])))
        if "current_project_id" in context_data:
            context_tokens.append(("current_project_id", current_project_id.set(context_data["current_project_id"])))

        span_attributes = {
            "queue.name": queue.name,
            "job.id": job.id,
            "job.func_name": job.func_name,
            "job.description": job.description
        }

        with instrument_span(
            name=job.func_name,
            attributes=span_attributes
        ) as span:
            try:
                return super().execute_job(job, queue)
            except Exception as exc:
                span.record_exception(
                    exception=exc,
                    attributes={
                        "exception.type": exc.__class__.__name__,
                        "exception.message": str(exc),
                    }
                )
                span.set_status(trace.Status(trace.StatusCode.ERROR))
                raise
