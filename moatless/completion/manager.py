"""Model configuration management."""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from moatless.completion.base import BaseCompletionModel, LLMResponseFormat
from moatless.completion.log_handler import LogHandler
from moatless.completion.schema import FewShotExample, ResponseSchema
from moatless.exceptions import CompletionRuntimeError
from moatless.schema import MessageHistoryType
from moatless.utils.moatless import get_moatless_dir

logger = logging.getLogger(__name__)


class ModelConfig(BaseModel):
    id: str = Field(..., description="Unique identifier for the model")
    model: str = Field(..., description="Model identifier used in LiteLLM")
    model_base_url: Optional[str] = Field(None, description="Base URL for the model API")
    model_api_key: Optional[str] = Field(None, description="API key for the model")
    temperature: Optional[float] = Field(..., description="Temperature for model sampling")
    max_tokens: Optional[int] = Field(..., description="Maximum number of tokens to generate")
    timeout: float = Field(..., description="Timeout in seconds for model requests")
    thoughts_in_action: bool = Field(..., description="Whether to include thoughts in actions")
    disable_thoughts: bool = Field(..., description="Whether to disable thoughts completely")
    merge_same_role_messages: bool = Field(..., description="Whether to merge consecutive messages with same role")
    message_cache: bool = Field(..., description="Whether to enable message caching")
    few_shot_examples: bool = Field(..., description="Whether to use few-shot examples")
    response_format: LLMResponseFormat = Field(..., description="Format for model responses")
    message_history_type: MessageHistoryType = Field(..., description="Type of message history to use")
    headers: dict[str, Any] = Field(default_factory=dict, description="Additional headers provided to LiteLLM")
    params: dict[str, Any] = Field(default_factory=dict, description="Additional parameters provided to LiteLLM")

    @classmethod
    def model_validate(cls, data: Any) -> "ModelConfig":
        if "response_format" in data:
            data["response_format"] = LLMResponseFormat(data["response_format"])
        if "message_history_type" in data:
            data["message_history_type"] = MessageHistoryType(data["message_history_type"])
        return super().model_validate(data)

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        data = super().model_dump(*args, **kwargs)
        data["response_format"] = data["response_format"].value
        data["message_history_type"] = data["message_history_type"].value
        return data


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
                action=TestResponseSchema(message="I understand the format and can respond correctly", success=True),
            )
        ]


class ModelConfigManager:
    """Manages model configurations and their runtime overrides."""

    def __init__(self):
        """Initialize the model config manager."""
        self._base_configs = self._load_base_configs()
        self._user_configs: dict[str, ModelConfig] = self._load_user_configs()

    def _get_base_config_path(self) -> Path:
        """Get the path to the base model config file."""
        return Path(__file__).parent / "model_config.json"

    def _get_user_config_path(self) -> Path:
        """Get the path to the user model config file."""
        base_dir = get_moatless_dir()
        if base_dir:
            base_path = Path(base_dir)
        else:
            base_path = Path.cwd()
        return base_path / "models.json"

    def _load_base_configs(self) -> dict[str, ModelConfig]:
        """Load base model configurations from JSON file."""
        config_path = self._get_base_config_path()
        with open(config_path) as f:
            config_data = json.load(f)

        # Convert string response formats to enum
        base_config = config_data["base_config"]
        models_config = config_data["models"]

        configs = {}
        for model_id, model_config in models_config.items():
            if "response_format" in model_config:
                model_config["response_format"] = LLMResponseFormat(model_config["response_format"])
            if "message_history_type" in model_config:
                model_config["message_history_type"] = MessageHistoryType(model_config["message_history_type"])
            # Merge base config with model specific config
            model_config.update({k: v for k, v in base_config.items() if k not in model_config})
            configs[model_id] = ModelConfig(**model_config, id=model_id)

        return configs

    def _load_user_configs(self) -> dict[str, ModelConfig]:
        """Load user model configurations from JSON file."""
        path = self._get_user_config_path()
        if not path.exists():
            return {}
        try:
            with open(path) as f:
                models_configs = json.load(f)

            configs = {}
            for model_config in models_configs:
                configs[model_config["id"]] = ModelConfig(**model_config)
            return configs
        except Exception as e:
            raise Exception(f"Failed to load user model configs: {e}") from e

    def _save_user_configs(self) -> None:
        """Save user model configurations to file."""
        path = self._get_user_config_path()
        logger.info(f"Saving user model configs to {path}")
        try:
            # Ensure directory exists
            path.parent.mkdir(parents=True, exist_ok=True)
            dumps = [v.model_dump() for k, v in self._user_configs.items()]
            with open(path, "w") as f:
                json.dump(dumps, f, indent=2)
        except Exception as e:
            raise Exception(f"Failed to save user model configs: {e}") from e

    def get_base_model_config(self, model_id: str) -> ModelConfig:
        """Get base configuration for a specific model.

        Args:
            model_id: The ID of the model to get configuration for.

        Returns:
            The model configuration.

        Raises:
            ValueError: If the model ID is not found.
        """
        if model_id not in self._base_configs:
            raise ValueError(f"Base model {model_id} not found")
        return self._base_configs[model_id].copy()

    def get_model_config(self, model_id: str) -> ModelConfig:
        """Get configuration for a specific model.

        Args:
            model_id: The ID of the model to get configuration for.

        Returns:
            The model configuration.

        Raises:
            ValueError: If the model ID is not found.
        """
        if model_id in self._user_configs:
            return self._user_configs[model_id]
        elif model_id in self._base_configs:
            return self._base_configs[model_id]
        raise ValueError(f"Model {model_id} not found")

    def get_all_base_configs(self) -> list[ModelConfig]:
        """Get all base model configurations.

        Returns:
            List of all base model configurations.
        """

        return list(self._base_configs.values())

    def get_all_configs(self) -> list[ModelConfig]:
        """Get all model configurations.

        Returns:
            List of all model configurations.
        """
        return list(self._user_configs.values())

    def add_model_from_base(
        self, base_model_id: str, new_model_id: str, updates: Optional[dict[str, Any]] = None
    ) -> ModelConfig:
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
        config.id = new_model_id
        # if updates:
        #    config = config.model_copy(update=updates)

        self._user_configs[new_model_id] = config
        self._save_user_configs()
        return config

    def update_model_config(self, model_id: str, updates: ModelConfig) -> ModelConfig:
        """Update configuration for a specific model.

        Args:
            model_id: The ID of the model to update.
            updates: ModelConfig

        Returns:
            The updated model configuration.

        Raises:
            ValueError: If the model ID is not found.
        """
        if model_id not in self._user_configs:
            raise ValueError(f"Model {model_id} not found in user configs")

        config = self._user_configs[model_id]
        self._user_configs[model_id] = updates
        self._save_user_configs()

        return config

    def delete_model_config(self, model_id: str) -> None:
        """Delete a user model configuration.

        Args:
            model_id: The ID of the model to delete.

        Raises:
            ValueError: If the model ID is not found in user configs.
        """
        if model_id not in self._user_configs:
            raise ValueError(f"Model {model_id} not found in user configs")

        del self._user_configs[model_id]
        self._save_user_configs()

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

        # Convert string enums to proper enum types

        logger.debug(f"Creating completion model for {model_id} with config: {config}")

        return BaseCompletionModel.create(**config.model_dump(), model_id=model_id)

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
            import litellm

            litellm.set_verbose = True
            log_dir = Path.cwd() / "logs"
            if not log_dir.exists():
                log_dir.mkdir(parents=True, exist_ok=True)

            litellm.callbacks = [LogHandler(log_dir=log_dir)]
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

    def create_model(self, model_config: ModelConfig) -> ModelConfig:
        """Create a new model configuration from scratch.

        Args:
            model_config: The complete model configuration.

        Returns:
            The new model configuration.

        Raises:
            ValueError: If a model with the same ID already exists.
        """
        if model_config.id in self._user_configs:
            raise ValueError(f"Model {model_config.id} already exists")
        self._user_configs[model_config.id] = model_config
        self._save_user_configs()
        return model_config


# Create singleton instance
_manager = ModelConfigManager()

# Export convenience functions
get_model_config = _manager.get_model_config
get_all_configs = _manager.get_all_configs
get_base_model_config = _manager.get_base_model_config
get_all_base_configs = _manager.get_all_base_configs
add_model_from_base = _manager.add_model_from_base
update_model_config = _manager.update_model_config
delete_model_config = _manager.delete_model_config
create_completion_model = _manager.create_completion_model


def create_model(model_config: ModelConfig) -> ModelConfig:
    """Create a new model configuration from scratch.

    Args:
        model_config: The complete model configuration.

    Returns:
        The new model configuration.

    Raises:
        ValueError: If a model with the same ID already exists.
    """
    return _manager.create_model(model_config)
