"""Agent configuration management."""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Optional

from pydantic import BaseModel, Field

from moatless.actions.action import Action
from moatless.agent.agent import ActionAgent
from moatless.index.code_index import CodeIndex
from moatless.message_history import MessageHistoryGenerator
from moatless.completion import BaseCompletionModel
from moatless.repository.repository import Repository
from moatless.utils.moatless import get_moatless_dir
from moatless.workspace import Workspace
from moatless.exceptions import RuntimeError

logger = logging.getLogger(__name__)


class AgentConfigManager:
    """Manages agent configurations."""

    def __init__(self):
        """Initialize the agent config manager."""
        self._global_config = {}
        self._configs = {}
        logger.info("Loading agent configs")
        self._load_configs()

    def _get_config_path(self) -> Path:
        """Get the path to the config file."""
        return get_moatless_dir() / "agents.json"

    def _get_global_config_path(self) -> Path:
        """Get the path to the config file."""
        return Path(__file__).parent / "agents.json"

    def _load_configs(self):
        """Load configurations from JSON file."""

        with open(self._get_global_config_path()) as f:
            global_config = json.load(f)
            for agent_id, agent_config in global_config.items():
                self._global_config[agent_id] = agent_config

        if os.path.exists(self._get_config_path()):
            try:
                with open(self._get_config_path()) as f:
                    configs = json.load(f)
                logger.info(f"Loaded {len(configs)} agent configs")
                for config in configs:
                    try: 
                        self._configs[config["id"]] = config
                        logger.info(f"Loaded agent config {config['id']}")
                    except Exception as e:
                        logger.error(f"Failed to load agent config {config['id']}: {e}")

            except Exception as e:
                logger.error(f"Failed to load agent configs: {e}")
                raise e

    def _save_configs(self):
        """Save configurations to JSON file."""
        path = self._get_config_path()
        try:
            configs = [c.model_dump() for c in self._configs.values()]
            with open(path, "w") as f:
                json.dump(configs, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save agent configs: {e}")

    def get_agent(self, agent_id: str) -> ActionAgent:
        """Get an agent configuration by ID."""
        logger.info(f"Getting agent config {agent_id}")
        if agent_id in self._configs:
            return ActionAgent.model_validate(self._configs[agent_id])
        elif agent_id in self._global_config:
            return ActionAgent.model_validate(self._global_config[agent_id])
        else:
            raise ValueError(f"Agent config {agent_id} not found. Available configs: {self._configs.keys()}")

    def get_all_agents(self) -> List[ActionAgent]:
        agents = []
        for config in self._configs.values():
            agents.append(ActionAgent.model_validate(config))

        for agent_id, config in self._global_config.items():
            if agent_id not in self._configs:
                try:
                    config["agent_id"] = agent_id
                    agents.append(ActionAgent.model_validate(config))
                except Exception as e:
                    logger.exception(f"Failed to load global agent config {config}")

        agents.sort(key=lambda x: x.agent_id)
        return agents

    def create_agent(self, agent: ActionAgent) -> ActionAgent:
        """Create a new agent configuration."""
        if agent.agent_id in self._configs:
            raise ValueError(f"Agent config {agent.agent_id} already exists")

        self._configs[agent.agent_id] = agent.model_dump()
        self._save_configs()
        return agent

    def update_agent(self, agent: ActionAgent):
        """Update an existing agent configuration."""
        self._configs[agent.agent_id] = agent.model_dump()
        self._save_configs()

    def delete_agent(self, agent_id: str):
        """Delete an agent configuration."""
        if agent_id not in self._configs:
            raise ValueError(f"Agent config {agent_id} not found")

        del self._configs[agent_id]
        self._save_configs()

    def create_agent(
        self,
        agent_id: str,
        completion_model: BaseCompletionModel,
        repository: Repository,
        code_index: CodeIndex,
        runtime: Optional[str] = None,
    ) -> ActionAgent:
        """Create an ActionAgent instance from a configuration.

        Args:
            agent_id: ID of the agent configuration to use
            completion_model: The completion model to use

        Returns:
            Configured ActionAgent instance

        Raises:
            ValueError: If the agent config is not found
        """
        agent = self.get_agent(agent_id)
        if not agent:
            raise RuntimeError(f"Agent config {agent_id} not found. Available configs: {self._configs.keys()}")

        logger.info(
            f"Creating agent with config id {agent.agent_id}, completion model {completion_model}, repository {repository}, code index {code_index}, runtime {runtime}"
        )

        workspace = Workspace(repository=repository, code_index=code_index, runtime=runtime)
        agent.workspace = workspace
        agent.completion_model = completion_model

        return agent


# Create singleton instance
_manager = AgentConfigManager()

# Export convenience functions
get_agent = _manager.get_agent
get_all_agents = _manager.get_all_agents
create_agent = _manager.create_agent
update_agent = _manager.update_agent
delete_agent = _manager.delete_agent
