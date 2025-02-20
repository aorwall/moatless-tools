"""Validation manager for SWEBench validations."""

import logging
import os
import uuid
from dataclasses import dataclass
from typing import Dict, Optional, Callable, Awaitable

from moatless.benchmark.swebench import create_repository
from moatless.evaluation.utils import get_moatless_instances
from moatless.config.agent_config import get_agent
from moatless.config.model_config import create_completion_model
from moatless.index import CodeIndex
from moatless.flow.loop import AgenticLoop
from moatless.runtime.testbed import TestbedEnvironment
from moatless.workspace import Workspace

logger = logging.getLogger(__name__)


@dataclass
class ValidationState:
    """State of a validation."""

    validation_id: str
    instance_id: str
    model_id: str
    agent_id: str
    status: str = "pending"  # pending, running, completed, error
    error: Optional[str] = None
    loop: Optional[AgenticLoop] = None


class ValidationManager:
    """Manager for SWEBench validations."""

    def __init__(self):
        self.validations: Dict[str, ValidationState] = {}

    def create_validation(self, instance_id: str, model_id: str, agent_id: str) -> str:
        """Create a new validation and return its ID."""
        validation_id = str(uuid.uuid4())
        self.validations[validation_id] = ValidationState(
            validation_id=validation_id,
            instance_id=instance_id,
            model_id=model_id,
            agent_id=agent_id,
        )
        return validation_id

    def get_validation(self, validation_id: str) -> Optional[ValidationState]:
        """Get validation state by ID."""
        return self.validations.get(validation_id)

    def set_loop(self, validation_id: str, loop: AgenticLoop):
        """Set the loop for a validation."""
        if validation := self.validations.get(validation_id):
            validation.loop = loop
            validation.status = "running"

    def set_error(self, validation_id: str, error: str):
        """Set error state for a validation."""
        if validation := self.validations.get(validation_id):
            validation.error = error
            validation.status = "error"

    def set_completed(self, validation_id: str):
        """Mark a validation as completed."""
        if validation := self.validations.get(validation_id):
            validation.status = "completed"

    async def run_validation(
        self,
        validation_id: str,
        progress_callback: Callable[[int, str], Awaitable[None]],
    ):
        """Run a validation."""
        validation = self.validations.get(validation_id)
        if not validation:
            logger.error(f"Validation {validation_id} not found")
            return

        try:
            instances = get_moatless_instances()
            instance = instances.get(validation.instance_id)
            if not instance:
                self.set_error(validation_id, "Instance not found")
                return

            # Create model and agent
            completion_model = create_completion_model(validation.model_id)
            repository = create_repository(instance)

            index_store_dir = os.getenv("INDEX_STORE_DIR", "/tmp/index_store")
            code_index = CodeIndex.from_index_name(
                instance["instance_id"],
                index_store_dir=index_store_dir,
                file_repo=repository,
            )

            if os.getenv("TESTBED_API_KEY") and os.getenv("TESTBED_BASE_URL"):
                runtime = TestbedEnvironment(repository=repository, instance=instance)
            else:
                logger.warning("TESTBED_API_KEY and TESTBED_BASE_URL not set, wont use testbed runtime")
                runtime = None

            agent = get_agent(validation.agent_id)

            workspace = Workspace(repository=repository, code_index=code_index, runtime=runtime)
            agent.workspace = workspace
            agent.completion_model = completion_model


            # Create and run loop with progress callback
            loop = AgenticLoop.create(
                f"<task>\n{instance['problem_statement']}\n</task>",
                agent=agent,
                repository=repository,
                runtime=runtime,
                max_iterations=15,
                progress_callback=progress_callback,
            )

            # Set the loop in validation manager
            self.set_loop(validation_id, loop)

            # Run the loop
            loop.run()

            # Mark validation as completed
            self.set_completed(validation_id)

        except Exception as e:
            logger.exception(f"Failed to run validation: {str(e)}")
            self.set_error(validation_id, str(e))

    def cleanup_validation(self, validation_id: str):
        """Clean up a validation's resources."""
        if validation := self.validations.get(validation_id):
            if validation.loop:
                # Clean up any resources associated with the loop
                pass
            del self.validations[validation_id]


# Global validation manager instance
validation_manager = ValidationManager()
