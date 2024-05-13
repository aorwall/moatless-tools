from typing import Optional, Dict, Any, TypeVar, Annotated, Type

from instructor import OpenAISchema
from pydantic import BaseModel, Extra, GetCoreSchemaHandler
from pydantic_core import core_schema

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

class NotNullable:
    def __get_pydantic_core_schema__(self, source: Type[Any], handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        schema = handler(source)
        assert schema["type"] == "nullable"
        return schema["schema"]


T = TypeVar("T")
Omissible = Annotated[Optional[T], NotNullable()]

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
