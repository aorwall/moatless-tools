import importlib
import logging
import pkgutil
from abc import ABC
from typing import List, Type, Tuple, Any, Dict, Optional, ClassVar

from pydantic import BaseModel, ConfigDict

from moatless.actions.schema import (
    ActionArguments,
    Observation, RewardScaleEntry, FewShotExample
)
from moatless.file_context import FileContext
from moatless.index import CodeIndex
from moatless.repository.repository import Repository
from moatless.workspace import Workspace

logger = logging.getLogger(__name__)

_actions: Dict[str, Type["Action"]] = {}


class Action(BaseModel, ABC):
    args_schema: ClassVar[Type[ActionArguments]]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data):
        super().__init__(**data)

    def execute(
        self,
        args: ActionArguments,
        file_context: FileContext | None = None,
        workspace: Workspace | None = None,
    ) -> Observation:
        """
        Execute the action.
        """

        message = self._execute(args, file_context=file_context, workspace=workspace)
        return Observation.create(message)

    def _execute(
        self,
        args: ActionArguments,
        file_context: FileContext | None = None,
        workspace: Workspace | None = None,
    ) -> str | None:
        """
        Execute the action and return the updated FileContext.
        """
        raise NotImplementedError("Subclasses must implement this method.")

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
    def get_action_by_args_class(
        cls, args_class: Type[ActionArguments]
    ) -> Optional[Type["Action"]]:
        """
        Get the Action subclass corresponding to the given ActionArguments subclass.

        Args:
            args_class: The ActionArguments subclass to look up.

        Returns:
            The Action subclass if found, None otherwise.
        """

        def search_subclasses(current_class):
            if (
                hasattr(current_class, "args_schema")
                and current_class.args_schema == args_class
            ):
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
        if not _actions:
            cls._load_actions()

        action = _actions.get(action_name)
        if action:
            return action

        raise ValueError(f"Unknown action: {action_name}")

    @classmethod
    def _load_actions(cls):
        actions_package = importlib.import_module("moatless.actions")

        for _, module_name, _ in pkgutil.iter_modules(actions_package.__path__):
            full_module_name = f"moatless.actions.{module_name}"
            module = importlib.import_module(full_module_name)
            for name, obj in module.__dict__.items():
                if isinstance(obj, type) and issubclass(obj, Action) and obj != Action:
                    _actions[name] = obj

    @classmethod
    def model_validate(
        cls,
        obj: Any,
        repository: Repository = None,
        runtime: Any = None,
        code_index: CodeIndex = None,
    ) -> "Action":
        if isinstance(obj, dict):
            obj = obj.copy()
            action_class_path = obj.pop("action_class", None)

            if action_class_path:
                module_name, class_name = action_class_path.rsplit(".", 1)
                module = importlib.import_module(module_name)
                action_class = getattr(module, class_name)

                if repository and hasattr(action_class, "_repository"):
                    obj["repository"] = repository
                if code_index and hasattr(action_class, "_code_index"):
                    obj["code_index"] = code_index
                if runtime and hasattr(action_class, "_runtime"):
                    obj["runtime"] = runtime

                return action_class(**obj)

        return cls(**obj)

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        dump = super().model_dump(**kwargs)
        dump["action_class"] = f"{self.__class__.__module__}.{self.__class__.__name__}"
        return dump
