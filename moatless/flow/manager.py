"""Tree search configuration management."""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Literal, Optional, List

from pydantic import BaseModel

from moatless.agent.agent import ActionAgent
from moatless.discriminator.base import BaseDiscriminator
from moatless.expander import Expander
from moatless.feedback.base import BaseFeedbackGenerator
from moatless.flow import AgenticFlow, AgenticLoop, SearchTree
from moatless.flow.schema import FlowConfig
from moatless.selector.base import BaseSelector
from moatless.utils.moatless import get_moatless_dir
from moatless.value_function.base import BaseValueFunction

from moatless.config.agent_config import get_agent
from moatless.completion.manager import create_completion_model

logger = logging.getLogger(__name__)

class FlowManager:
    """Manages tree search configurations."""

    def __init__(self):
        """Initialize the tree config manager."""
        self._configs = {}
        logger.info("Loading flow configs")
        self._load_configs()

    def create_flow(self,
                    id: str, 
                    model_id: str,
                    message: str | None = None,
                    trajectory_id: str | None = None,
                    persist_dir: str | None = None,
                    metadata: Dict[str, Any] | None = None,
                    **kwargs) -> AgenticFlow:
        
        """Create a SearchTree instance from this configuration.
        
        Args:
            root_node: Optional root node for the search tree. If not provided, 
                      the tree will need to be initialized with a root node later.
        
        Returns:
            SearchTree: A configured search tree instance
        """
        # Create expander if not specified

        config = self.get_flow_config(id)

        agent = get_agent(agent_id=config.agent_id)
        completion_model = create_completion_model(model_id)
        agent.completion_model = completion_model

        if config.flow_type == "loop":
            return AgenticLoop.create(
                message=message,
                trajectory_id=trajectory_id,
                agent=agent,
                max_iterations=config.max_iterations,
                max_cost=config.max_cost,
                persist_dir=persist_dir,
                metadata=metadata,
                **kwargs
            )
        elif config.flow_type == "tree":
            expander = config.expander or Expander(max_expansions=config.max_expansions)

            if hasattr(config.value_function, "model_id") and not config.value_function.model_id:
                config.value_function.model_id = model_id

            if hasattr(config.feedback_generator, "model_id") and not config.feedback_generator.model_id:
                config.feedback_generator.model_id = model_id

            if hasattr(config.discriminator, "model_id") and not config.discriminator.model_id:
                config.discriminator.model_id = model_id

            if hasattr(config.selector, "model_id") and not config.selector.model_id:
                config.selector.model_id = model_id

            tree = SearchTree.create(
                message=message,
                trajectory_id=trajectory_id,
                agent=agent,
                selector=config.selector,
                expander=expander,
                value_function=config.value_function,
                feedback_generator=config.feedback_generator,
                discriminator=config.discriminator,
                max_expansions=config.max_expansions,
                max_iterations=config.max_iterations,
                max_cost=config.max_cost,
                min_finished_nodes=config.min_finished_nodes,
                max_finished_nodes=config.max_finished_nodes,
                reward_threshold=config.reward_threshold,
                max_depth=config.max_depth,
                persist_dir=persist_dir,
                metadata=metadata,
                **kwargs
            )
        
        return tree

    def _get_config_path(self) -> Path:
        """Get the path to the config file."""
        return get_moatless_dir() / "flows.json"

    def _get_global_config_path(self) -> Path:
        """Get the path to the global config file."""
        return Path(__file__).parent / "flows.json"

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
                        with open(config_path, 'w') as local_f:
                            json.dump(global_config, local_f, indent=2)
                    logger.info("Copied global config to local path")
                else:
                    logger.info("No global tree configs found")
            except Exception as e:
                logger.error(f"Failed to copy global tree configs: {e}")

        # Load configs from local path
        try:
            if config_path.exists():
                with open(config_path) as f:
                    configs = json.load(f)
                logger.info(f"Loaded {len(configs)} tree configs")
                for config in configs:
                    try:
                        self._configs[config["id"]] = config
                        logger.info(f"Loaded tree config {config['id']}")
                    except Exception as e:
                        logger.error(f"Failed to load tree config {config['id']}: {e}")
            else:
                logger.info("No local tree configs found")
        except Exception as e:
            logger.error(f"Failed to load tree configs: {e}")
            raise e

    def _save_configs(self):
        """Save configurations to JSON file."""
        path = self._get_config_path()
        try:
            configs = list(self._configs.values())
            logger.info(f"Saving tree configs to {path}")
            with open(path, "w") as f:
                json.dump(configs, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save tree configs: {e}")

    def get_flow_config(self, id: str) -> FlowConfig:
        """Get a tree configuration by ID."""
        logger.debug(f"Getting tree config {id}")

        if id in self._configs:
            config = self._configs[id]
            # Ensure the id is set in the config
            config["id"] = id
            return FlowConfig.model_validate(config)
        else:
            raise ValueError(f"Tree config {id} not found. Available configs: {list(self._configs.keys())}")

    def get_all_configs(self) -> List[FlowConfig]:
        """Get all tree configurations."""
        configs = []
        for config in self._configs.values():
            try:
                configs.append(FlowConfig.model_validate(config))
            except Exception as e:
                logger.exception(f"Failed to load tree config {config['id']}: {e}")

        configs.sort(key=lambda x: x.id)
        return configs

    def create_config(self, config: FlowConfig) -> FlowConfig:
        """Create a new tree configuration."""
        logger.debug(f"Creating tree config {config.id}")
        if config.id in self._configs:
            raise ValueError(f"Tree config {config.id} already exists")

        self._configs[config.id] = config.model_dump()
        self._save_configs()
        return config

    def update_config(self, config: FlowConfig):
        """Update an existing tree configuration."""
        logger.debug(f"Updating tree config {config.id}")
        if config.id not in self._configs:
            raise ValueError(f"Tree config {config.id} not found")

        self._configs[config.id] = config.model_dump()
        self._save_configs()

    def delete_config(self, id: str):
        """Delete a tree configuration."""
        logger.debug(f"Deleting tree config {id}")
        if id not in self._configs:
            raise ValueError(f"Tree config {id} not found")

        del self._configs[id]
        self._save_configs()


_manager = FlowManager()

get_flow_config = _manager.get_flow_config
get_all_configs = _manager.get_all_configs
create_flow_config = _manager.create_config
update_flow_config = _manager.update_config
delete_flow_config = _manager.delete_config
create_flow = _manager.create_flow