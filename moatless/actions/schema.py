import importlib
import logging
import pkgutil
from abc import ABC
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator

from moatless.artifacts.artifact import ArtifactChange
from moatless.completion.model import Completion
from moatless.completion.schema import ResponseSchema

logger = logging.getLogger(__name__)


_action_args: dict[str, type["ActionArguments"]] = {}


class ActionArguments(ResponseSchema, ABC):
    thoughts: Optional[str] = Field(None, description="Your reasoning for the action.")

    def format_for_llm(self) -> str:
        """Format the action name for LLM consumption"""
        return str(self.name)

    @classmethod
    def format_name_for_llm(cls) -> str:
        """Format the class name for LLM consumption"""
        return str(cls.name)

    @classmethod
    def from_tool_call(cls, tool_args: dict[str, Any], tool_name: str | None = None):
        return cls(**tool_args)

    def equals(self, other: "ActionArguments") -> bool:
        return self.model_dump(exclude={"thoughts"}) == other.model_dump(exclude={"thoughts"})

    def to_prompt(self):
        prompt = f"Action: {self.name}\n"
        prompt += "\n".join([f"  {k}: {v}" for k, v in self.model_dump(exclude={"thoughts"}).items()])
        return prompt

    def short_summary(self) -> str:
        return f"{self.name}()"

    @model_validator(mode="before")
    @classmethod
    def fix_null_fields(cls, data: Any) -> Any:
        """Allow thoughts to be null."""
        if isinstance(data, dict):
            for key, value in data.items():
                if value == "null":
                    data[key] = None

        return data

    @classmethod
    def get_action_args(cls, action_name: str) -> type["ActionArguments"]:
        """
        Dynamically import and return the appropriate ActionArguments class for the given action.
        """
        if not _action_args:
            cls._load_action_args()

        action_args = _action_args.get(action_name)
        if action_args:
            return action_args

        raise ValueError(f"Unknown action: {action_name}")

    @classmethod
    def _load_action_args(cls):
        actions_package = importlib.import_module("moatless.actions")

        for _, module_name, _ in pkgutil.iter_modules(actions_package.__path__):
            full_module_name = f"moatless.actions.{module_name}"
            module = importlib.import_module(full_module_name)
            for name, obj in module.__dict__.items():
                if isinstance(obj, type) and issubclass(obj, ActionArguments) and obj != ActionArguments:
                    _action_args[name] = obj

    @classmethod
    def model_validate(cls, obj: Any) -> "ActionArguments":
        if isinstance(obj, dict):
            obj = obj.copy()
            action_args_class_path = obj.pop("action_args_class", None)

            if action_args_class_path:
                if action_args_class_path == "moatless.actions.request_context.RequestMoreContextArgs":
                    action_args_class_path = "moatless.actions.view_code.ViewCodeArgs"

                if action_args_class_path.startswith("moatless.actions.edit"):
                    action_args_class_path = "moatless.actions.claude_text_editor.EditActionArguments"

                module_name, class_name = action_args_class_path.rsplit(".", 1)
                module = importlib.import_module(module_name)
                action_args_class = getattr(module, class_name)
                return action_args_class.model_validate(obj)
        return super().model_validate(obj)


class Observation(BaseModel):
    message: Optional[str] = Field(
        None,
        description="The message returned to the agent, will be displayed in message history.",
    )
    summary: Optional[str] = Field(
        None,
        description="Summary of the observation, will be displayed in summarised message history.",
    )
    terminal: bool = Field(False, description="Indicates if this action results in a terminal state")

    properties: Optional[dict[str, Any]] = Field(default_factory=dict, description="Additional properties")
    execution_completion: Optional[Completion] = Field(None, description="Completion created when executing the action")
    artifact_changes: Optional[list[ArtifactChange]] = Field(
        default_factory=list, description="Artifact changes created when executing the action"
    )

    @classmethod
    def create(
        cls,
        message: str,
        summary: Optional[str] = None,
        terminal: bool = False,
        properties: Optional[dict[str, Any]] = None,
        execution_completion: Optional[Completion] = None,
        artifact_changes: Optional[list[ArtifactChange]] = None,
    ):
        return cls(
            message=message,
            terminal=terminal,
            summary=summary,
            properties=properties,
            execution_completion=execution_completion,
            artifact_changes=artifact_changes,
        )


class RewardScaleEntry(BaseModel):
    min_value: int
    max_value: int
    description: str


class ActionProperty(BaseModel):
    type: str
    title: str
    description: str
    default: Optional[Any] = None


class ActionSchema(BaseModel):
    title: str
    description: str
    type: str = "object"
    action_class: str
    properties: dict[str, ActionProperty]
