from collections import namedtuple
from typing import List, Optional

from pydantic import BaseModel

BlockPath = List[str]

Span = namedtuple("Span", ["start_line", "end_line"])


class ContextFile(BaseModel):
    file_path: str
    spans: List[Span] = None


class CodingTask(BaseModel):
    file_path: str
    instructions: str
    span_id: Optional[str] = None
    action: str = "update"  # add, remove, update
    state: str = "planned"  # completed, planned, rejected, failed


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class BaseResponse(BaseModel):
    thoughts: Optional[str] = None
    error: Optional[str] = None
    usage_stats: List[Usage] = []
