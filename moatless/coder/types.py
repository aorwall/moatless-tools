from abc import abstractmethod
from typing import Optional, List, Dict, Any

from instructor import OpenAISchema
from pydantic import BaseModel, Extra

from moatless.types import BaseResponse


class WriteCodeResult(BaseModel):
    content: Optional[str] = None
    error: Optional[str] = None
    change: Optional[str] = None


class CoderResponse(BaseResponse):
    file_path: str
    diff: Optional[str] = None
    error: Optional[str] = None
    change: Optional[str] = None


class FunctionResponse(BaseModel):
    message: Optional[str] = None
    error: Optional[str] = None
    error_type: Optional[str] = None


class Function(OpenAISchema):

    @classmethod
    @property
    def openai_tool_spec(cls) -> Dict[str, Any]:
        return {"type": "function", "function": cls.openai_schema}

    @classmethod
    @property
    def name(cls) -> str:
        return cls.openai_schema["name"]

    # TODO: @abstractmethod
    def call(self) -> FunctionResponse:
        pass

    class Config:
        extra = Extra.forbid
        arbitrary_types_allowed = True


class CodeFunction(Function):
    file_path: str
