from typing import Optional, List, Dict, Any

from instructor import OpenAISchema
from pydantic import BaseModel

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


class CodeFunction(OpenAISchema):
    file_path: str

    @classmethod
    @property
    def openai_tool_spec(cls) -> Dict[str, Any]:
        return {"type": "function", "function": cls.openai_schema}


class WriteCodeResult(BaseModel):
    content: Optional[str] = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    change: Optional[str] = None
