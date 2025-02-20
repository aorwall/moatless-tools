"""Model configuration management."""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

from pydantic import BaseModel, Field

from moatless.completion.base import BaseCompletionModel, LLMResponseFormat
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


class ModelConfigManager:
    """Manages model configurations and their runtime overrides."""

    def __init__(self):
        """Initialize the model config manager."""
        self._base_configs = self._load_model_configs()
        self._runtime_overrides = self._load_overrides()

    def _get_overrides_path(self) -> Path:
        """Get the path to the overrides file.

        Uses MOATLESS_DIR environment variable if set, otherwise uses current working directory.
        The overrides file will be stored in the 'config' subdirectory.
        """
        base_dir = get_moatless_dir()
        if base_dir:
            base_path = Path(base_dir)
        else:
            base_path = Path.cwd()

        return base_path / "config" / "model_overrides.json"

    def _load_model_configs(self) -> Dict[str, ModelConfig]:
        """Load model configurations from JSON file."""
        config_path = Path(__file__).parent / "model_config.json"
        with open(config_path) as f:
            config_data = json.load(f)

        # Convert string response formats to enum
        base_config = config_data["base_config"]

        models_config = config_data["models"]
        for model_id, model_config in models_config.items():
            
            if "response_format" in model_config:
                model_config["response_format"] = LLMResponseFormat(model_config["response_format"])
            if "message_history_type" in model_config:
                model_config["message_history_type"] = MessageHistoryType(model_config["message_history_type"])
            # Merge base config with model specific config
            model_config.update({k: v for k, v in base_config.items() if k not in model_config})

            models_config[model_id] = ModelConfig(**model_config, id=model_id)

        return models_config

    def _load_overrides(self) -> Dict[str, Dict[str, Any]]:
        """Load model overrides from file."""
        path = self._get_overrides_path()
        if not path.exists():
            return {}
        try:
            with open(path) as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to load model overrides: {e}")
            return {}

    def _save_overrides(self) -> None:
        """Save current runtime overrides to file."""
        path = self._get_overrides_path()
        logger.info(f"Saving model overrides to {path}")
        try:
            # Ensure directory exists
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(self._runtime_overrides, f, indent=2)
        except Exception as e:
            print(f"Failed to save model overrides: {e}")

    def get_model_config(self, model_id: str) -> ModelConfig:
        """Get configuration for a specific model with any runtime overrides applied.

        Args:
            model_id: The ID of the model to get configuration for.

        Returns:
            The model configuration with any runtime overrides applied.

        Raises:
            ValueError: If the model ID is not found.
        """
        if model_id not in self._base_configs:
            raise ValueError(f"Model {model_id} not found")

        config = self._base_configs[model_id].copy()
        if model_id in self._runtime_overrides:
            config.update(self._runtime_overrides[model_id])

        config.id = model_id
        return config

    def get_all_configs(self) -> List[ModelConfig]:
        """Get all model configurations with runtime overrides applied.

        Returns:
            Dictionary mapping model IDs to their configurations with overrides applied.
        """
        return list(self._base_configs.values())

    def update_model_config(self, model_id: str, updates: ModelConfig) -> ModelConfig:
        """Update configuration for a specific model.

        Args:
            model_id: The ID of the model to update.
            updates: Dictionary of configuration updates to apply.

        Returns:
            The updated model configuration.

        Raises:
            ValueError: If the model ID is not found.
        """
        if model_id not in self._base_configs:
            raise ValueError(f"Model {model_id} not found")

        # Get current overrides or create new
        current_overrides = self._runtime_overrides.get(model_id, {})

        logger.info(f"Updating model config for {model_id} with {updates}")

        # Update overrides with new values
        if updates:
            current_overrides.update(updates)
            self._runtime_overrides[model_id] = current_overrides
            self._save_overrides()

        # Return merged configuration
        config = self._base_configs[model_id].copy()
        config.update(current_overrides)
        return ModelConfig.model_validate(config)

    def reset_model_config(self, model_id: str) -> Dict[str, Any]:
        """Reset model configuration to defaults by removing overrides.

        Args:
            model_id: The ID of the model to reset.

        Returns:
            The base model configuration without overrides.

        Raises:
            ValueError: If the model ID is not found.
        """
        if model_id not in self._base_configs:
            raise ValueError(f"Model {model_id} not found")

        # Remove overrides if they exist
        if model_id in self._runtime_overrides:
            del self._runtime_overrides[model_id]
            self._save_overrides()

        return self._base_configs[model_id].copy()

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


# Create singleton instance
_manager = ModelConfigManager()

# Export convenience functions
get_model_config = _manager.get_model_config
get_all_configs = _manager.get_all_configs
update_model_config = _manager.update_model_config
reset_model_config = _manager.reset_model_config
create_completion_model = _manager.create_completion_model
