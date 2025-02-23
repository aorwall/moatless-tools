import logging
from abc import ABC, abstractmethod
from typing import List, Type, Tuple, Any, Dict, Optional, ClassVar

from moatless.telemetry import instrument
from pydantic import BaseModel, ConfigDict, PrivateAttr

from moatless.actions.schema import (
    ActionArguments,
    ActionProperty,
    ActionSchema,
    Observation,
    RewardScaleEntry,
)
from moatless.completion.base import BaseCompletionModel
from moatless.file_context import FileContext
from moatless.index import CodeIndex
from moatless.repository.repository import Repository
from moatless.runtime.runtime import RuntimeEnvironment
from moatless.component import MoatlessComponent
from moatless.workspace import Workspace
from moatless.completion.schema import FewShotExample

logger = logging.getLogger(__name__)

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

    def _initialize_completion_model(self):
        """Override this method to customize completion model initialization"""
        pass


class Action(MoatlessComponent):
    """Base class for all actions."""
    args_schema: ClassVar[Type[ActionArguments]]
    model_config = ConfigDict(arbitrary_types_allowed=True)
    _workspace: Workspace = PrivateAttr(default=None)
    
    @classmethod
    def get_component_type(cls) -> str:
        return "action"
        
    @classmethod
    def _get_package(cls) -> str:
        return "moatless.actions"
        
    @classmethod
    def _get_base_class(cls) -> Type:
        return Action

    @instrument(name=lambda self: f"{self.name}")
    async def execute(self, args: ActionArguments, file_context: FileContext | None = None) -> Observation:
        """Execute the action."""
        if not self._workspace:
            raise RuntimeError("No workspace set")

        message = await self._execute(args, file_context=file_context)
        return Observation.create(message)

    async def _execute(self, args: ActionArguments, file_context: FileContext | None = None) -> str | None:
        """Execute the action and return the updated FileContext."""
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
    def get_value_function_prompt(cls) -> str | None:
        """
        Get the base prompt for the value function.
        This method can be overridden in subclasses to provide action-specific prompts.
        """
        return None

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
        cls._initialize_components()
        return cls._get_components().get(action_name)
    
    @classmethod
    def create_by_name(cls, name: str, **kwargs) -> "Action":
        cls._initialize_components()

        action_class = cls.get_action_by_name(name)
        if not action_class:
            raise ValueError(f"Unknown action: {name}, available actions: {cls._get_components().keys()}")
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
        cls._initialize_components()

        return [
            action_class.get_action_schema() 
            for action_class in cls._get_components().values()
        ]

    @classmethod
    def get_class_name(cls) -> str:
        return f"{cls.__module__}.{cls.__name__}"
    