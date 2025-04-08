import logging
from abc import ABC
from typing import Any, ClassVar, Optional

from opentelemetry import trace
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from moatless.actions.schema import (
    ActionArguments,
    ActionProperty,
    ActionSchema,
    Observation,
    RewardScaleEntry,
)
from moatless.completion.base import BaseCompletionModel
from moatless.component import MoatlessComponent
from moatless.file_context import FileContext
from moatless.index import CodeIndex
from moatless.repository.repository import Repository
from moatless.runtime.runtime import RuntimeEnvironment
from moatless.workspace import Workspace

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


class CompletionModelMixin(BaseModel, ABC):
    """Mixin to provide completion model functionality to actions that need it"""

    completion_model: Optional[BaseCompletionModel] = Field(
        default=None,
        description="Completion model to be used for generating completions",
    )

    def _initialize_completion_model(self):
        """Override this method to customize completion model initialization"""
        pass

    @model_validator(mode="after")
    def initialize_completion_model(self):
        """Automatically initialize the completion model if set"""
        if self.completion_model:
            self._initialize_completion_model()
        return self

    def model_completion_dump(self, dump: dict[str, Any]) -> dict[str, Any]:
        if self.completion_model:
            dump["completion_model"] = self.completion_model.model_dump()
        return dump

    @classmethod
    def model_completion_validate(cls, obj: dict[str, Any]) -> dict[str, Any]:
        if "completion_model" in obj:
            obj["completion_model"] = BaseCompletionModel.model_validate(obj["completion_model"])
        return obj


class Action(MoatlessComponent, ABC):
    """Base class for all actions."""

    args_schema: ClassVar[type[ActionArguments]]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    is_terminal: bool = Field(default=False, description="Whether the action will finish the flow")

    _workspace: Workspace | None = PrivateAttr(default=None)

    @classmethod
    def get_component_type(cls) -> str:
        return "action"

    @classmethod
    def _get_package(cls) -> str:
        return "moatless.actions"

    @classmethod
    def _get_base_class(cls) -> type:
        return Action

    @tracer.start_as_current_span("execute")
    async def execute(self, args: ActionArguments, file_context: FileContext | None = None) -> Observation:
        """Execute the action."""
        if not self._workspace:
            raise RuntimeError("No workspace set")

        message = await self._execute(args, file_context=file_context)
        return Observation.create(message=message or "")

    async def _execute(self, args: ActionArguments, file_context: FileContext | None = None) -> str | None:
        """Execute the action and return the updated FileContext."""
        raise NotImplementedError("Subclasses must implement this method.")

    @property
    def workspace(self) -> Workspace:
        if not self._workspace:
            raise ValueError("Workspace is not set")
        return self._workspace

    async def initialize(self, workspace: Workspace):
        self._workspace = workspace
        # No need to check for CompletionModelMixin - validation hooks will handle it

    # TODO: Replace this with initialize method
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
        if not self.workspace.runtime:
            raise RuntimeError("Runtime is not set")
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
    def get_evaluation_criteria(cls, trajectory_length: int | None = None) -> list[str]:
        if trajectory_length is None or trajectory_length < 2:
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
        descriptions: list[tuple[int, int, str]],
    ) -> list[RewardScaleEntry]:
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
    def get_value_function_prompt(cls) -> str | None:
        """
        Get the base prompt for the value function.
        This method can be overridden in subclasses to provide action-specific prompts.
        """
        return None

    @classmethod
    def get_action_by_args_class(cls, args_class: type[ActionArguments]) -> type["Action"] | None:
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
    def get_action_by_name(cls, action_name: str) -> type["Action"]:
        """
        Dynamically import and return the appropriate Action class for the given action name.
        """
        cls._initialize_components()
        components = cls._get_components()
        
        # Find the component where the class name matches action_name
        for qualified_name, component_class in components.items():
            if qualified_name.split('.')[-1] == action_name:
                return component_class
        
        # Get just the class names for the error message
        available_actions = [name.split('.')[-1] for name in components.keys()]
        raise ValueError(f"Unknown action: {action_name}, available actions: {available_actions}")

    @classmethod
    def create_by_name(cls, name: str, **kwargs) -> "Action":
        cls._initialize_components()

        action_class = cls.get_action_by_name(name)
        if not action_class:
            raise ValueError(f"Unknown action: {name}, available actions: {cls._get_components().keys()}")
        return action_class(**kwargs)

    @classmethod
    def get_action_schema(cls) -> ActionSchema:
        """Generate an ActionSchema for this action."""
        schema = cls.model_json_schema()
        properties = {}
        for prop_name, prop_data in schema.get("properties", {}).items():
            properties[prop_name] = ActionProperty(
                type=prop_data.get("type", "string"),
                title=prop_data.get("title", prop_name),
                description=prop_data.get("description", ""),
                default=prop_data.get("default"),
            )

        description = schema.get("description", "")
        if not description:
            description = cls.args_schema.model_json_schema().get("description", "")

        return ActionSchema(
            title=cls.__name__,
            description=description,
            properties=properties,
            action_class=cls.get_class_name(),
        )

    @classmethod
    def get_available_actions(cls) -> list[ActionSchema]:
        """Get all available actions with their schema."""
        cls._initialize_components()

        return [action_class.get_action_schema() for action_class in cls._get_components().values()]

    @classmethod
    def get_class_name(cls) -> str:
        return f"{cls.__module__}.{cls.__name__}"

    def model_dump(self, **kwargs) -> dict[str, Any]:
        dump = super().model_dump(**kwargs)

        if isinstance(self, CompletionModelMixin):
            dump = self.model_completion_dump(dump)

        return dump

    @classmethod
    def model_validate(cls, obj: Any) -> "Action":
        if isinstance(obj, dict):
            obj = obj.copy()

            if issubclass(cls, CompletionModelMixin):
                obj = cls.model_completion_validate(obj)

        return super().model_validate(obj)
