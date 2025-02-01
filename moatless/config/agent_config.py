"""Agent configuration management."""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

from pydantic import BaseModel, Field

from moatless.actions.action import Action
from moatless.agent.agent import ActionAgent
from moatless.index.code_index import CodeIndex
from moatless.message_history import MessageHistoryGenerator
from moatless.completion import BaseCompletionModel
from moatless.repository.repository import Repository
from moatless.workspace import Workspace
from moatless.exceptions import RuntimeError

logger = logging.getLogger(__name__)


class AgentConfigManager:
    """Manages agent configurations."""

    def __init__(self):
        """Initialize the agent config manager."""
        self._configs = {}
        logger.info("Loading agent configs")
        self._load_configs()

    def _get_config_path(self) -> Path:
        """Get the path to the config file."""
        return Path(__file__).parent / "agent_config.json"

    def _load_configs(self):
        """Load configurations from JSON file."""
        path = self._get_config_path()
        if not path.exists():
            logger.warning(f"Agent config file {path} not found")
            return

        try:
            with open(path) as f:
                configs = json.load(f)
                logger.info(f"Loaded {len(configs)} agent configs")
                for config_id, config in configs.items():
                    logger.info(f"Loaded agent config {config_id}")
                    config["agent_id"] = config_id
                    self._configs[config_id] = ActionAgent.model_validate(config)

            if not len(self._configs):
                raise RuntimeError("No agent configs found")
        except Exception as e:
            logger.error(f"Failed to load agent configs: {e}")
            raise e

    def _save_configs(self):
        """Save configurations to JSON file."""
        path = self._get_config_path()
        try:
            configs = {k: v.model_dump() for k, v in self._configs.items()}
            with open(path, "w") as f:
                json.dump(configs, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save agent configs: {e}")

    def get_config(self, config_id: str) -> ActionAgent:
        """Get an agent configuration by ID."""
        logger.info(f"Getting agent config {config_id}")
        if config_id not in self._configs:
            raise ValueError(f"Agent config {config_id} not found. Available configs: {self._configs.keys()}")
        return self._configs[config_id]

    def get_all_configs(self) -> Dict[str, ActionAgent]:
        """Get all agent configurations."""
        return self._configs.copy()

    def create_config(self, config_id: str, config: Dict[str, Any]) -> ActionAgent:
        """Create a new agent configuration."""
        if config_id in self._configs:
            raise ValueError(f"Agent config {config_id} already exists")

        agent_config = ActionAgent(**config)
        self._configs[config_id] = agent_config
        self._save_configs()
        return agent_config

    def update_config(self, config_id: str, updates: Dict[str, Any]) -> ActionAgent:
        """Update an existing agent configuration."""
        if config_id not in self._configs:
            raise ValueError(f"Agent config {config_id} not found")

        config = self._configs[config_id]
        updated = config.model_copy(update=updates)
        self._configs[config_id] = updated
        self._save_configs()
        return updated

    def delete_config(self, config_id: str):
        """Delete an agent configuration."""
        if config_id not in self._configs:
            raise ValueError(f"Agent config {config_id} not found")

        del self._configs[config_id]
        self._save_configs()

    def create_agent(
        self,
        config_id: str,
        completion_model: BaseCompletionModel,
        repository: Repository,
        code_index: CodeIndex,
        runtime: Optional[str] = None,
    ) -> ActionAgent:
        """Create an ActionAgent instance from a configuration.

        Args:
            config_id: ID of the agent configuration to use
            completion_model: The completion model to use

        Returns:
            Configured ActionAgent instance

        Raises:
            ValueError: If the agent config is not found
        """
        config = self.get_config(config_id)
        if not config:
            raise RuntimeError(f"Agent config {config_id} not found. Available configs: {self._configs.keys()}")

        logger.info(
            f"Creating agent with config id {config_id}, completion model {completion_model}, repository {repository}, code index {code_index}, runtime {runtime}"
        )

        agent = ActionAgent.model_validate(config.model_dump())
        workspace = Workspace(repository=repository, code_index=code_index, runtime=runtime)
        agent.workspace = workspace
        agent.completion_model = completion_model

        return agent


# Create singleton instance
_manager = AgentConfigManager()

# Export convenience functions
get_config = _manager.get_config
get_all_configs = _manager.get_all_configs
create_config = _manager.create_config
update_config = _manager.update_config
delete_config = _manager.delete_config
create_agent = _manager.create_agent
