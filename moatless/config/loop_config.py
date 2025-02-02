"""Loop configuration management."""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from pydantic import BaseModel, Field

from moatless.config.agent_config import get_agent as get_agent_config

logger = logging.getLogger(__name__)


class LoopConfig(BaseModel):
    """Configuration for an agentic loop."""

    agent_config_id: str = Field(..., description="The ID of the agent configuration to use")
    max_iterations: int = Field(10, description="The maximum number of iterations to run")
    max_cost: Optional[float] = Field(None, description="The maximum cost spent on tokens before finishing")


class LoopConfigManager:
    """Manages loop configurations."""

    def __init__(self):
        """Initialize the loop config manager."""
        self._configs = {}
        self._load_configs()

    def _get_config_path(self) -> Path:
        """Get the path to the config file."""
        return Path(__file__).parent / "loop_config.json"

    def _load_configs(self):
        """Load configurations from JSON file."""
        path = self._get_config_path()
        if not path.exists():
            return

        try:
            with open(path) as f:
                configs = json.load(f)
                for config_id, config in configs.items():
                    self._configs[config_id] = LoopConfig(**config)
        except Exception as e:
            logger.error(f"Failed to load loop configs: {e}")

    def _save_configs(self):
        """Save configurations to JSON file."""
        path = self._get_config_path()
        try:
            configs = {k: v.model_dump() for k, v in self._configs.items()}
            with open(path, "w") as f:
                json.dump(configs, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save loop configs: {e}")

    def get_config(self, config_id: str) -> LoopConfig:
        """Get a loop configuration by ID."""
        if config_id not in self._configs:
            raise ValueError(f"Loop config {config_id} not found")
        return self._configs[config_id]

    def get_all_configs(self) -> Dict[str, LoopConfig]:
        """Get all loop configurations."""
        return self._configs.copy()

    def create_config(self, config_id: str, config: Dict[str, Any]) -> LoopConfig:
        """Create a new loop configuration."""
        if config_id in self._configs:
            raise ValueError(f"Loop config {config_id} already exists")

        # Validate that agent config exists
        get_agent_config(config["agent_config_id"])

        loop_config = LoopConfig(**config)
        self._configs[config_id] = loop_config
        self._save_configs()
        return loop_config

    def update_config(self, config_id: str, updates: Dict[str, Any]) -> LoopConfig:
        """Update an existing loop configuration."""
        if config_id not in self._configs:
            raise ValueError(f"Loop config {config_id} not found")

        # Validate that agent config exists if being updated
        if "agent_config_id" in updates:
            get_agent_config(updates["agent_config_id"])

        config = self._configs[config_id]
        updated = config.model_copy(update=updates)
        self._configs[config_id] = updated
        self._save_configs()
        return updated

    def delete_config(self, config_id: str):
        """Delete a loop configuration."""
        if config_id not in self._configs:
            raise ValueError(f"Loop config {config_id} not found")

        del self._configs[config_id]
        self._save_configs()


# Create singleton instance
_manager = LoopConfigManager()

# Export convenience functions
get_config = _manager.get_config
get_all_configs = _manager.get_all_configs
create_config = _manager.create_config
update_config = _manager.update_config
delete_config = _manager.delete_config
