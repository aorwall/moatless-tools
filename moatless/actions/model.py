import importlib
import logging
import pkgutil
from abc import ABC
from typing import Dict, Type, Any, Optional

from pydantic import Field, BaseModel, model_validator

from moatless.completion.model import ToolCall, Completion, StructuredOutput

logger = logging.getLogger(__name__)


_action_args: Dict[str, Type["ActionArguments"]] = {}


class ActionArguments(StructuredOutput, ABC):
    thoughts: str = Field(..., description="Your reasoning for the action.")

    class Config:
        title = "Action"

    @classmethod
    def get_name(cls) -> str:
        """Returns the action name for the class based on Config title."""
        return str(getattr(cls.Config, "title", cls.__name__))

    def format_for_llm(self) -> str:
        """Format the action name for LLM consumption"""
        return str(self.name)

    @classmethod
    def format_name_for_llm(cls) -> str:
        """Format the class name for LLM consumption"""
        return str(cls.get_name())

    def to_tool_call(self) -> ToolCall:
        tool_input = self.model_dump()

        return ToolCall(name=self.name, input=tool_input)

    @classmethod
    def from_tool_call(cls, tool_args: dict[str, Any], tool_name: str | None = None):
        return cls(**tool_args)

    def equals(self, other: "ActionArguments") -> bool:
        return self.model_dump(exclude={"thoughts"}) == other.model_dump(
            exclude={"thoughts"}
        )

    def to_prompt(self):
        prompt = f"Action: {self.name}\n"
        prompt += "\n".join(
            [f"  {k}: {v}" for k, v in self.model_dump(exclude={"thoughts"}).items()]
        )
        return prompt

    def short_summary(self) -> str:
        return f"{self.name}()"

    @model_validator(mode="before")
    @classmethod
    def fix_thoughts(cls, data: Any) -> Any:
        """Allow thoughts to be null."""
        if isinstance(data, dict):
            if "scratch_pad" in data:
                data["thoughts"] = data["scratch_pad"]
                del data["scratch_pad"]

            if not data.get("thoughts"):
                data["thoughts"] = ""

        return data

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
    def get_action_args(cls, action_name: str) -> Type["ActionArguments"]:
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
                if (
                    isinstance(obj, type)
                    and issubclass(obj, ActionArguments)
                    and obj != ActionArguments
                ):
                    _action_args[name] = obj

    @classmethod
    def model_validate(cls, obj: Any) -> "ActionArguments":
        if isinstance(obj, dict):
            obj = obj.copy()
            action_args_class_path = obj.pop("action_args_class", None)
            if (
                action_args_class_path
                == "moatless.actions.request_context.RequestMoreContextArgs"
            ):
                action_args_class_path = "moatless.actions.view_code.ViewCodeArgs"

            if action_args_class_path:
                module_name, class_name = action_args_class_path.rsplit(".", 1)
                module = importlib.import_module(module_name)
                action_args_class = getattr(module, class_name)
                return action_args_class.model_validate(obj)
        return super().model_validate(obj)


class RewardScaleEntry(BaseModel):
    min_value: int
    max_value: int
    description: str


class Observation(BaseModel):
    message: Optional[str] = Field(
        None,
        description="The message returned to the agent, will be displayed in message history.",
    )
    summary: Optional[str] = Field(
        None,
        description="Summary of the observation, will be displayed in summarised message history.",
    )
    terminal: bool = Field(
        False, description="Indicates if this action results in a terminal state"
    )
    expect_correction: bool = Field(
        False,
        description="Indicates that a the action arguments was inccorect and we expect a correction",
    )
    properties: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Additional properties"
    )
    execution_completion: Optional[Completion] = Field(
        None, description="Completion created when executing the action"
    )

    @classmethod
    def create(cls, message: str, terminal: bool = False):
        return cls(message=message, terminal=terminal)


class FewShotExample(BaseModel):
    user_input: str = Field(..., description="The user's input/question")
    action: ActionArguments = Field(
        ..., description="The expected response as ActionArguments"
    )

    @classmethod
    def create(cls, user_input: str, action: ActionArguments) -> "FewShotExample":
        return cls(user_input=user_input, action=action)


class ActionError(ActionArguments):
    """Error"""

    error: str = Field(..., description="Error.")

    class Config:
        title = "Error"

    def to_prompt(self):
        return f"Error: {self.error}"


class RetryException(Exception):
    """Exception raised when an action needs to be retried with corrected arguments."""

    def __init__(self, message: str, action_args: ActionArguments):
        super().__init__(message)
        self.message = message
        self.action_args = action_args
