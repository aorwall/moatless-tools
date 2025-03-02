"""Agent configuration management."""

import json
import logging
from pathlib import Path

from moatless.agent.agent import ActionAgent
from moatless.utils.moatless import get_moatless_dir

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
        return get_moatless_dir() / "agents.json"

    def _get_global_config_path(self) -> Path:
        """Get the path to the config file."""
        return Path(__file__).parent / "agents.json"

    def _load_configs(self):
        """Load configurations from JSON file."""
        config_path = self._get_config_path()

        # Copy global config to local path if it doesn't exist
        if not config_path.exists():
            try:
                global_path = self._get_global_config_path()
                if global_path.exists():
                    # Copy global config to local path
                    config_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(global_path) as f:
                        global_config = json.load(f)
                        with open(config_path, "w") as local_f:
                            json.dump(global_config, local_f, indent=2)
                    logger.info("Copied global config to local path")
                else:
                    logger.info("No global agent configs found")
            except Exception as e:
                logger.error(f"Failed to copy global agent configs: {e}")

        # Load configs from local path
        try:
            if config_path.exists():
                with open(config_path) as f:
                    configs = json.load(f)
                logger.info(f"Loaded {len(configs)} agent configs")
                for config in configs:
                    try:
                        self._configs[config["agent_id"]] = config
                        logger.info(f"Loaded agent config {config['agent_id']}")
                    except Exception as e:
                        logger.error(f"Failed to load agent config {config['agent_id']}: {e}")
            else:
                logger.info("No local agent configs found")
        except Exception as e:
            logger.error(f"Failed to load agent configs: {e}")
            raise e

    def _save_configs(self):
        """Save configurations to JSON file."""
        path = self._get_config_path()
        try:
            configs = list(self._configs.values())
            logger.info(f"Saving agent configs to {path}")
            with open(path, "w") as f:
                json.dump(configs, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save agent configs: {e}")

    def get_agent(self, agent_id: str) -> ActionAgent:
        """Get an agent configuration by ID."""
        logger.debug(f"Getting agent config {agent_id}")
        if agent_id in self._configs:
            return ActionAgent.model_validate(self._configs[agent_id])
        else:
            raise ValueError(f"Agent config {agent_id} not found. Available configs: {self._configs.keys()}")

    def get_all_agents(self) -> list[ActionAgent]:
        agents = []
        for config in self._configs.values():
            try:
                agents.append(ActionAgent.model_validate(config))
            except Exception as e:
                logger.exception(f"Failed to load agent config {config['agent_id']}: {e}")

        agents.sort(key=lambda x: x.agent_id)
        return agents

    def create_agent(self, agent: ActionAgent) -> ActionAgent:
        """Create a new agent configuration."""
        logger.debug(f"Creating agent config {agent.agent_id}")
        if agent.agent_id in self._configs:
            raise ValueError(f"Agent config {agent.agent_id} already exists")

        self._configs[agent.agent_id] = agent.model_dump()
        self._save_configs()
        return agent

    def update_agent(self, agent: ActionAgent):
        """Update an existing agent configuration."""
        logger.debug(f"Updating agent config {agent.agent_id}")
        self._configs[agent.agent_id] = agent.model_dump()
        self._save_configs()

    def delete_agent(self, agent_id: str):
        """Delete an agent configuration."""
        logger.debug(f"Deleting agent config {agent_id}")
        if agent_id not in self._configs:
            raise ValueError(f"Agent config {agent_id} not found")

        del self._configs[agent_id]
        self._save_configs()


# Create singleton instance
_manager = AgentConfigManager()

# Export convenience functions
get_agent = _manager.get_agent
get_all_agents = _manager.get_all_agents
create_agent = _manager.create_agent
update_agent = _manager.update_agent
delete_agent = _manager.delete_agent
