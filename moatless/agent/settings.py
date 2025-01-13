from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from moatless.completion import CompletionModel
from moatless.schema import MessageHistoryType


class AgentSettings(BaseModel):
    model_config = {"frozen": True}

    completion_model: CompletionModel = Field(
        ..., description="Completion model to be used for generating completions"
    )
    system_prompt: Optional[str] = Field(
        None, description="System prompt to be used for generating completions"
    )
    actions: List[str] = Field(default_factory=list)
    message_history_type: MessageHistoryType = Field(
        default=MessageHistoryType.MESSAGES,
        description="Determines how message history is generated",
    )
    thoughts_in_action: bool = Field(
        default=False,
        description="Whether to include thoughts in the action or in the message",
    )

    def __eq__(self, other):
        if not isinstance(other, AgentSettings):
            return False
        return (
            self.completion_model == other.completion_model
            and self.system_prompt == other.system_prompt
            and self.actions == other.actions
            and self.message_history_type == other.message_history_type
        )

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        dump = super().model_dump(**kwargs)
        dump["message_history_type"] = self.message_history_type.value
        return dump

    @classmethod
    def model_validate(cls, obj: Any) -> "AgentSettings":
        if isinstance(obj, dict):
            if "message_history_type" in obj:
                obj["message_history_type"] = MessageHistoryType(
                    obj["message_history_type"]
                )

        return super().model_validate(obj)
