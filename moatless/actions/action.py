import importlib
import logging
import pkgutil
from abc import ABC, abstractmethod
from typing import List, Type, Tuple, Any, Dict, Optional, ClassVar
import inspect
import os
import sys

from docstring_parser import parse
from pydantic import BaseModel, ConfigDict, PrivateAttr

from moatless.actions.schema import (
    ActionArguments,
    ActionProperty,
    ActionSchema,
    Observation,
    RewardScaleEntry,
    FewShotExample,
)
from moatless.completion.base import BaseCompletionModel
from moatless.file_context import FileContext
from moatless.index import CodeIndex
from moatless.repository.repository import Repository
from moatless.runtime.runtime import RuntimeEnvironment
from moatless.workspace import Workspace
from moatless.utils.class_loading import DynamicClassLoadingMixin

logger = logging.getLogger(__name__)

_actions: Dict[str, Type["Action"]] = {}

class CompletionModelMixin:
    """Mixin to provide completion model functionality to actions that need it"""

    _completion_model: Optional[BaseCompletionModel] = None

    @property
    def completion_model(self):
        return self._completion_model

    @completion_model.setter
    def completion_model(self, value: Optional[BaseCompletionModel]):
        if value is None:
            self._completion_model = None
        else:
            self._completion_model = value.clone()
            self._initialize_completion_model()

    @abstractmethod
    def _initialize_completion_model(self):
        """Override this method to customize completion model initialization"""
        pass


class Action(BaseModel, ABC, DynamicClassLoadingMixin):
    args_schema: ClassVar[Type[ActionArguments]]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    _workspace: Workspace = PrivateAttr(default=None)

    async def execute(self, args: ActionArguments, file_context: FileContext | None = None) -> Observation:
        """
        Execute the action.
        """

        if not self._workspace:
            raise RuntimeError("No workspace set")

        message = self._execute(args, file_context=file_context)
        return Observation.create(message)

    async def _execute(self, args: ActionArguments, file_context: FileContext | None = None) -> str | None:
        """
        Execute the action and return the updated FileContext.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @property
    def workspace(self) -> Workspace:
        if not self._workspace:
            raise ValueError("Workspace is not set")
        return self._workspace

    @workspace.setter
    def workspace(self, value: Workspace):
        self._workspace = value

    @property
    def _repository(self) -> Repository:
        return self.workspace.repository

    @property
    def _code_index(self) -> CodeIndex:
        return self.workspace.code_index

    @property
    def _runtime(self) -> RuntimeEnvironment:
        return self.workspace.runtime

    @property
    def name(self) -> str:
        """Returns the name of the action class as a string."""
        return self.__class__.__name__

    @classmethod
    def get_name(cls) -> str:
        """Returns the name of the action class as a string."""
        return cls.__name__

    @classmethod
    def get_evaluation_criteria(cls, trajectory_length: int | None = None) -> List[str]:
        if trajectory_length < 3:
            return [
                "Exploratory Actions: Recognize that initial searches and information-gathering steps are essential and should not be heavily penalized if they don't yield immediate results.",
                "Appropriateness of Action: Evaluate if the action is logical given the agent's current knowledge and the early stage of problem-solving.",
            ]

        else:
            return [
                "Solution Quality: Assess the logical changes, contextual fit, and overall improvement without introducing new issues.",
                "Progress Assessment: Evaluate the agent's awareness of solution history, detection of repetitive actions, and planned next steps.",
                "Repetitive or Redundant Actions: Detect if the agent is repeating the same unsuccessful or redundant actions without making progress. Pay close attention to the agent's history and outputs indicating lack of progress.",
            ]

    @staticmethod
    def generate_reward_scale_entries(
        descriptions: List[Tuple[int, int, str]],
    ) -> List[RewardScaleEntry]:
        """
        Generate a list of RewardScaleEntry objects based on the provided descriptions.

        Args:
            descriptions: A list of tuples, each containing (min_value, max_value, description)

        Returns:
            A list of RewardScaleEntry objects
        """
        return [
            RewardScaleEntry(min_value=min_val, max_value=max_val, description=desc)
            for min_val, max_val, desc in descriptions
        ]

    @classmethod
    def get_reward_range(cls, trajectory_length: int) -> Tuple[int, int]:
        """
        Get the minimum and maximum reward values for this action.

        Args:
            trajectory_length: The length of the current trajectory

        Returns:
            A tuple containing the minimum and maximum reward values
        """
        reward_scale = cls.get_reward_scale(trajectory_length)
        min_reward = min(entry.min_value for entry in reward_scale)
        max_reward = max(entry.max_value for entry in reward_scale)
        return min_reward, max_reward

    @classmethod
    def get_value_function_prompt(cls) -> str:
        """
        Get the base prompt for the value function.
        This method can be overridden in subclasses to provide action-specific prompts.
        """
        pass

    @classmethod
    def get_few_shot_examples(cls) -> List[FewShotExample]:
        """
        Returns a list of few-shot examples specific to this action.
        Override this method in subclasses to provide custom examples.
        """
        return []

    @classmethod
    def get_action_by_args_class(cls, args_class: Type[ActionArguments]) -> Optional[Type["Action"]]:
        """
        Get the Action subclass corresponding to the given ActionArguments subclass.

        Args:
            args_class: The ActionArguments subclass to look up.

        Returns:
            The Action subclass if found, None otherwise.
        """

        def search_subclasses(current_class):
            if hasattr(current_class, "args_schema") and current_class.args_schema == args_class:
                return current_class
            for subclass in current_class.__subclasses__():
                result = search_subclasses(subclass)
                if result:
                    return result
            return None

        return search_subclasses(cls)

    @classmethod
    def get_action_by_name(cls, action_name: str) -> Type["Action"]:
        """
        Dynamically import and return the appropriate Action class for the given action name.
        """
        cls._initialize_actions_map()
        return _actions.get(action_name)
    
    @classmethod
    def _initialize_actions_map(cls):
        global _actions
        if not _actions:
            _actions = cls._load_classes("moatless.actions", Action)

    @classmethod
    def model_validate(cls, obj: Any) -> "Action":
        if isinstance(obj, dict):
            obj = obj.copy()

            if obj.get("action_class"):
                action_class_path = obj["action_class"]
                # TODO: Keep backwards compatibility for old claude text editor package
                if action_class_path == "moatless.actions.edit":
                    action_class_path = "moatless.actions.claude_text_editor"

                try:
                    # Ensure actions are loaded
                    cls._initialize_actions_map()
                    
                    # Try to find action by full class path first
                    module_name, class_name = action_class_path.rsplit(".", 1)
                    action_class = cls.get_action_by_name(class_name)
                    
                    if not action_class:
                        logger.warning(f"Invalid action class: {class_name}. Available actions: {_actions.keys()}")
                        raise ValueError(f"Invalid action class: {class_name}")
                    
                    return action_class(**obj)
                except Exception as e:
                    logger.warning(f"Failed to load action class {action_class_path}: {e}")
                    raise ValueError(f"Failed to load action class {action_class_path}") from e
            else:
                raise ValueError(f"action_class is required in {obj}")

        return super().model_validate(obj)
    
    @classmethod
    def create_by_name(cls, name: str, **kwargs) -> "Action":
        cls._initialize_actions_map()

        action_class = cls.get_action_by_name(name)
        if not action_class:
            raise ValueError(f"Unknown action: {name}")
        return action_class(**kwargs)

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        dump = super().model_dump(**kwargs)
        dump["action_class"] = self.get_class_name()
        return dump

    @classmethod
    def get_action_schema(cls) -> ActionSchema:
        """Generate an ActionSchema for this action."""
        schema = cls.model_json_schema()
        properties = {}
        for prop_name, prop_data in schema.get('properties', {}).items():
            properties[prop_name] = ActionProperty(
                type=prop_data.get('type', 'string'),
                title=prop_data.get('title', prop_name),
                description=prop_data.get('description', ''),
                default=prop_data.get('default')
            )

        description = schema.get('description', '')
        if not description:
            description = cls.args_schema.model_json_schema().get('description', '')

        return ActionSchema(
            title=cls.__name__,
            description=description,
            properties=properties,
            action_class=cls.get_class_name()
        )

    @classmethod
    def get_available_actions(cls) -> List[ActionSchema]:
        """Get all available actions with their schema."""
        if not cls._registered_classes:
            cls._load_classes("moatless.actions", Action)

        return [
            action_class.get_action_schema() 
            for name, action_class in cls._registered_classes.items()
        ]

    @classmethod
    def get_class_name(cls) -> str:
        return f"{cls.__module__}.{cls.__name__}"
    