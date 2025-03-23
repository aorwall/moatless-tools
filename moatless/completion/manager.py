"""Model configuration management."""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from moatless import settings
from moatless.completion.base import BaseCompletionModel, LLMResponseFormat
from moatless.completion.log_handler import LogHandler
from moatless.completion.schema import FewShotExample, ResponseSchema
from moatless.exceptions import CompletionRuntimeError
from moatless.storage.base import BaseStorage

logger = logging.getLogger(__name__)


class ModelTestResult(BaseModel):
    """Result of a model configuration test"""

    success: bool = Field(..., description="Whether the model test passed")
    message: str = Field(..., description="Human-readable test result message")
    model_id: str = Field(..., description="ID of the tested model")
    model_response: Optional[str] = Field(None, description="Raw response message from the model if available")
    error_type: Optional[str] = Field(None, description="Type of error if test failed")
    error_details: Optional[str] = Field(None, description="Detailed error information if test failed")
    response_time_ms: Optional[float] = Field(None, description="Response time in milliseconds")
    test_timestamp: str = Field(..., description="ISO timestamp when test was performed")


class TestResponseSchema(ResponseSchema):
    """A simple schema for testing model configurations."""

    message: str = Field(..., description="A test message from the model")
    success: bool = Field(..., description="Whether the model understood the request")

    @classmethod
    def get_few_shot_examples(cls) -> list[FewShotExample]:
        return [
            FewShotExample.create(
                user_input="This is a test message. Please respond in the correct format to verify you understand the schema.",
                action=TestResponseSchema(
                    message="I understand the format and can respond correctly",
                    success=True,
                ),
            )
        ]


class ModelConfigManager:
    """Manages model configurations and their runtime overrides."""

    _instance: Optional["ModelConfigManager"] = None

    def __init__(self, storage: BaseStorage):
        """Initialize the model config manager."""
        self._base_configs = self._load_base_configs()
        if not storage:
            raise ValueError("Storage is required")
        self._storage = storage
        self._user_configs: dict[str, BaseCompletionModel] = {}

    async def initialize(self):
        """Initialize the model config manager."""
        await self._load_user_configs()

    def _get_base_config_path(self) -> Path:
        """Get the path to the base model config file."""
        return Path(__file__).parent / "model_config.json"

    def _load_base_configs(self) -> dict[str, BaseCompletionModel]:
        """Load base model configurations from JSON file."""
        config_path = self._get_base_config_path()
        with open(config_path) as f:
            config_data = json.load(f)

        base_config = config_data["base_config"]
        models_config = config_data["models"]

        configs = {}
        for model_id, model_config in models_config.items():
            model_config.update({k: v for k, v in base_config.items() if k not in model_config})
            model_config["model_id"] = model_id
            try:
                configs[model_id] = BaseCompletionModel.model_validate(model_config)
            except Exception as e:
                logger.error(f"Failed to validate model config: {model_config}")
                raise e

        return configs

    async def _load_user_configs(self) -> None:
        """Load user model configurations from JSON file."""

        if not await self._storage.exists("models.json"):
            return

        models_configs = await self._storage.read("models.json")

        logger.info(f"Loading {len(models_configs)} model configs from {self._storage}")

        self._user_configs = {}
        for model_config in models_configs:
            self._user_configs[model_config["model_id"]] = BaseCompletionModel.model_validate(model_config)

    async def _save_user_configs(self) -> None:
        """Save user model configurations to file."""
        try:
            logger.info(f"Saving user model configs to {self._storage}")
            dumps = [v.model_dump() for k, v in self._user_configs.items()]
            await self._storage.write("models.json", dumps)
        except Exception as e:
            raise Exception(f"Failed to save user model configs: {e}") from e

    def get_base_model_config(self, model_id: str) -> BaseCompletionModel:
        if model_id not in self._base_configs:
            raise ValueError(f"Base model {model_id} not found")
        return self._base_configs[model_id].copy()

    def get_model_config(self, model_id: str) -> BaseCompletionModel:
        if not model_id:
            raise ValueError("Model ID is required")

        if model_id in self._user_configs:
            return self._user_configs[model_id]
        elif model_id in self._base_configs:
            return self._base_configs[model_id]
        raise ValueError(f"Model {model_id} not found, available models: {list(self._base_configs.keys())}")

    def get_all_base_configs(self) -> list[BaseCompletionModel]:
        return list(self._base_configs.values())

    def get_all_configs(self) -> list[BaseCompletionModel]:
        return list(self._user_configs.values())

    async def add_model_from_base(
        self,
        base_model_id: str,
        new_model_id: str,
        updates: Optional[dict[str, Any]] = None,
    ) -> BaseCompletionModel:
        """Add a new model configuration based on a base model.

        Args:
            base_model_id: The ID of the base model to copy from.
            new_model_id: The ID for the new model.
            updates: Optional dictionary of configuration updates to apply.

        Returns:
            The new model configuration.

        Raises:
            ValueError: If the base model ID is not found or if the new model ID already exists.
        """
        if base_model_id not in self._base_configs:
            raise ValueError(f"Base model {base_model_id} not found")
        if new_model_id in self._user_configs:
            raise ValueError(f"Model {new_model_id} already exists")

        config = self._base_configs[base_model_id].model_copy()
        self._user_configs[new_model_id] = config
        await self._save_user_configs()
        return config

    async def update_model_config(self, model_id: str, updates: BaseCompletionModel) -> BaseCompletionModel:
        """Update configuration for a specific model.

        Args:
            model_id: The ID of the model to update.
            updates: BaseCompletionModel

        Returns:
            The updated model configuration.

        Raises:
            ValueError: If the model ID is not found.
        """
        if model_id not in self._user_configs:
            raise ValueError(f"Model {model_id} not found in user configs")

        config = self._user_configs[model_id]
        self._user_configs[model_id] = updates
        await self._save_user_configs()

        return config

    async def delete_model_config(self, model_id: str) -> None:
        """Delete a user model configuration.

        Args:
            model_id: The ID of the model to delete.

        Raises:
            ValueError: If the model ID is not found in user configs.
        """
        if model_id not in self._user_configs:
            raise ValueError(f"Model {model_id} not found in user configs")

        del self._user_configs[model_id]
        await self._save_user_configs()

    def create_completion_model(self, model_id: str) -> BaseCompletionModel:
        """Create a BaseCompletionModel instance for a model ID.

        Args:
            model_id: The ID of the model to create.

        Returns:
            A configured BaseCompletionModel instance.

        Raises:
            ValueError: If the model ID is not found.
        """
        config = self.get_model_config(model_id)
        if not config:
            raise ValueError(f"Model {model_id} not found")

        logger.debug(f"Creating completion model for {model_id} with config: {config}")

        config_dict = config.model_dump()
        config_dict["model_id"] = model_id
        return BaseCompletionModel.model_validate(config_dict)

    async def test_model_setup(self, model_id: str) -> dict[str, Any]:
        """Test if a model configuration works correctly.

        This method creates a simple completion model with a basic schema and tests
        if the model can understand and respond in the expected format.

        Args:
            model_id: The ID of the model to test.

        Returns:
            Dictionary containing detailed test results including success status,
            timing information, and any error details.
        """
        start_time = time.time()
        result = {
            "success": False,
            "message": "",
            "model_id": model_id,
            "model_response": None,
            "error_type": None,
            "error_details": None,
            "response_time_ms": None,
            "test_timestamp": datetime.utcnow().isoformat(),
        }

        try:
            model = self.create_completion_model(model_id)

            model.initialize(
                response_schema=TestResponseSchema,
                system_prompt="You are a helpful AI assistant. Please respond to this test message to verify you can understand and follow the required response format.",
            )

            messages = [
                {
                    "role": "user",
                    "content": "This is a test message. Please respond in the correct format to verify you understand the schema.",
                }
            ]

            response = await model.create_completion(messages=messages)

            if response and isinstance(response.structured_output, TestResponseSchema):
                test_response = response.structured_output
                result["model_response"] = test_response.message

                if test_response.success:
                    result["success"] = True
                    result["message"] = "Model setup test passed."
                else:
                    result["message"] = f"Model indicated failure. Message from LLM: {test_response.message}"
            else:
                result["message"] = str(response)
                result["error_type"] = "ValidationError"
                result["error_details"] = "Response did not match expected schema"

        except CompletionRuntimeError as e:
            result.update(
                {
                    "message": f"Runtime error during test: {str(e)}",
                    "error_type": "CompletionRuntimeError",
                    "error_details": str(e),
                }
            )
        except Exception as e:
            result.update(
                {
                    "message": f"Unexpected error during test: {str(e)}",
                    "error_type": type(e).__name__,
                    "error_details": str(e),
                }
            )
        finally:
            result["response_time_ms"] = (time.time() - start_time) * 1000

        return result

    async def create_model(self, model_config: BaseCompletionModel) -> BaseCompletionModel:
        """Create a new model configuration from scratch.

        Args:
            model_config: The complete model configuration.

        Returns:
            The new model configuration.

        Raises:
            ValueError: If a model with the same ID already exists.
        """
        if model_config.model_id in self._user_configs:
            raise ValueError(f"Model {model_config.model_id} already exists")
        self._user_configs[model_config.model_id] = model_config
        await self._save_user_configs()
        return model_config
