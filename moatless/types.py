import json
from typing import List, Optional, Dict, Any, Type

from pydantic import BaseModel, Field
from pydantic.main import Model


class FileWithSpans(BaseModel):
    file_path: str = Field(
        description="The file path where the relevant code is found."
    )
    span_ids: List[str] = Field(
        default_factory=list,
        description="The span ids of the relevant code in the file",
    )

    def add_span_id(self, span_id):
        if span_id not in self.span_ids:
            self.span_ids.append(span_id)

    def add_span_ids(self, span_ids: List[str]):
        for span_id in span_ids:
            self.add_span_id(span_id)


class ActionResponse(BaseModel):
    message: Optional[str] = None


class ActionRequest(BaseModel):
    pass


class EmptyRequest(ActionRequest):
    pass


class ActionSpec(BaseModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def name(cls) -> str:
        return ""

    @classmethod
    def description(cls) -> str:
        return ""

    @classmethod
    def request_class(cls) -> Type[ActionRequest]:
        return EmptyRequest

    @classmethod
    def validate_request(cls, args: Dict[str, Any]) -> ActionRequest:
        return cls.request_class().model_validate(args, strict=True)

    @classmethod
    def create_request_from_json(cls, call_id: str, args: str) -> ActionRequest:
        args_dict = json.loads(args)
        args_dict["call_id"] = call_id
        return cls.request_class().model_validate(args_dict, strict=True)

    # TODO: Do generic solution to get parameters from ActionRequest
    @classmethod
    def openai_tool_spec(cls) -> Dict[str, Any]:
        parameters = cls.request_class().openai_tool_parameters()
        return {
            "type": "function",
            "function": {
                "name": cls.name(),
                "description": cls.description(),
                "parameters": parameters,
            },
        }


class Finish(ActionRequest):
    thoughts: str = Field(..., description="The reason to finishing the request.")


class Reject(ActionRequest):
    reason: str = Field(..., description="The reason for rejecting the request.")


class Content(ActionRequest):
    content: str


class Message(BaseModel):
    role: str
    content: Optional[str] = None
    action: Optional[ActionRequest] = Field(default=None)


class AssistantMessage(Message):
    role: str = "assistant"
    content: Optional[str] = None
    action: Optional[ActionRequest] = Field(default=None)


class UserMessage(Message):
    role: str = "user"
    content: Optional[str] = None
