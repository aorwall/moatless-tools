import importlib
import logging
import pkgutil
from abc import ABC
from typing import List, Type, Tuple, Any, Dict, Optional, ClassVar

from pydantic import BaseModel, ConfigDict

from moatless.actions.model import (
    ActionArguments,
    Observation,
    FewShotExample,
)
from moatless.file_context import FileContext
from moatless.index import CodeIndex
from moatless.repository.repository import Repository

logger = logging.getLogger(__name__)

_actions: Dict[str, Type["Action"]] = {}


class Action(BaseModel, ABC):
    args_schema: ClassVar[Type[ActionArguments]]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data):
        super().__init__(**data)

    def execute(self, args: ActionArguments, file_context: FileContext) -> Observation:
        """
        Execute the action.
        """

        message = self._execute(file_context=file_context)
        return Observation.create(message)

    def _execute(self, file_context: FileContext) -> str | None:
        """
        Execute the action and return the updated FileContext.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @property
    def name(self) -> str:
        return self.__class__.__name__


    @classmethod
    def from_dict(
        cls,
        obj: dict,
        repository: Repository = None,
        runtime: Any = None,
        code_index: CodeIndex = None,
    ) -> "Action":
        obj = obj.copy()
        obj.pop("args_schema", None)
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

            return action_class.model_validate(obj)

        raise ValueError(f"Unknown action: {obj}")

    @classmethod
    def model_validate(cls, obj: Any) -> "Action":
        return cls(**obj)

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
    def get_few_shot_examples(cls) -> List[FewShotExample]:
        """
        Returns a list of few-shot examples specific to this action.
        Override this method in subclasses to provide custom examples.
        """
        return []
