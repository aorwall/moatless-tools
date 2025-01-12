import importlib
import logging
import pkgutil
from abc import ABC
from typing import List, Type, Tuple, Any, Dict, Optional, ClassVar

from pydantic import BaseModel, ConfigDict

from moatless.actions.model import (
    ActionArguments,
    Observation,
    RewardScaleEntry,
    FewShotExample,
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

    @classmethod
    def get_reward_scale(cls, trajectory_length) -> List[RewardScaleEntry]:
        return [
            RewardScaleEntry(
                min_value=75,
                max_value=100,
                description="The action significantly advances the solution.",
            ),
            RewardScaleEntry(
                min_value=50,
                max_value=74,
                description="The action contributes positively towards solving the problem.",
            ),
            RewardScaleEntry(
                min_value=25,
                max_value=49,
                description="The action is acceptable but may have some issues.",
            ),
            RewardScaleEntry(
                min_value=0,
                max_value=24,
                description="The action has minimal impact or minor negative consequences.",
            ),
            RewardScaleEntry(
                min_value=-49,
                max_value=-1,
                description="The code change is inappropriate, unhelpful, introduces new issues, or redundantly repeats previous changes without making further progress. The Git diff does not align with instructions or is unnecessary.",
            ),
            RewardScaleEntry(
                min_value=-100,
                max_value=-50,
                description="The code change is counterproductive, causing significant setbacks or demonstrating persistent repetition without learning. The agent fails to recognize completed tasks and continues to attempt redundant actions.",
            ),
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
        return """Your role is to evaluate the **last executed action** of the search tree that our AI agents are traversing, to help us determine the best trajectory to solve a programming issue. The agent is responsible for identifying and modifying the correct file(s) in response to the problem statement.

Important: While line numbers may be referenced in the initial problem description, they can shift as changes are made to the file. Focus on whether the agent is modifying the correct logical parts of the code, rather than strictly matching the initially mentioned line numbers. What matters is that the right section of code is being modified, even if its current line number differs from what was originally specified.

At this stage, the agent is still working on the solution. Your task is twofold:
1. **Evaluation**: Assess whether the change done by the **last executed action** is appropriate for addressing the problem and whether the agent is on the right path to resolving the issue. Verify that the correct sections of code are being modified, regardless of their current line numbers.
2. **Alternative Feedback**: Independently of your evaluation, provide guidance for an alternative problem-solving branch. This ensures parallel exploration of different solution paths.
"""

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
