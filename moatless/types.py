from typing import Any

from pydantic import BaseModel, Field


class FileWithSpans(BaseModel):
    file_path: str = Field(
        description="The file path where the relevant code is found."
    )
    span_ids: list[str] = Field(
        default_factory=list,
        description="The span ids of the relevant code in the file",
    )

    def add_span_id(self, span_id):
        if span_id not in self.span_ids:
            self.span_ids.append(span_id)

    def add_span_ids(self, span_ids: list[str]):
        for span_id in span_ids:
            self.add_span_id(span_id)


class ActionRequest(BaseModel):
    pass

    @property
    def action_name(self):
        return self.__class__.__name__


class EmptyRequest(ActionRequest):
    pass


class Finish(ActionRequest):
    thoughts: str = Field(..., description="The reason to finishing the request.")


class Reject(ActionRequest):
    thoughts: str = Field(..., description="The reason for rejecting the request.")


class Content(ActionRequest):
    content: str


class Message(BaseModel):
    role: str
    content: str | None = None
    action: ActionRequest | None = Field(default=None)


class AssistantMessage(Message):
    role: str = "assistant"
    content: str | None = None
    action: ActionRequest | None = Field(default=None)


class UserMessage(Message):
    role: str = "user"
    content: str | None = None


class ActionResponse(BaseModel):
    trigger: str | None = None
    output: dict[str, Any] | None = None
    retry_message: str | None = None

    @classmethod
    def retry(cls, retry_message: str):
        return cls(trigger="retry", retry_message=retry_message)

    @classmethod
    def transition(cls, trigger: str, output: dict[str, Any] | None = None):
        output = output or {}
        return cls(trigger=trigger, output=output)

    @classmethod
    def no_transition(cls, output: dict[str, Any]):
        return cls(output=output)


class Response(BaseModel):
    status: str
    message: str
    output: dict[str, Any] | None = None


class VerificationError(BaseModel):
    code: str
    file_path: str
    message: str
    line: int


class CodeChange(BaseModel):
    instructions: str = Field(..., description="Instructions to do the code change.")
    file_path: str = Field(..., description="The file path of the code to be updated.")
    span_id: str = Field(..., description="The span id of the code to be updated.")
