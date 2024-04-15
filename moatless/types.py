from collections import namedtuple
from typing import List, Optional

from pydantic import BaseModel

BlockPath = List[str]


class Span(BaseModel):
    start_line: int
    end_line: int
    block_path: List[str] = None
    is_partial: bool = False

    @property
    def span_id(self):
        _span_id = ""
        if self.block_path:
            _span_id += f"{'.'.join(self.block_path)}"

        if _span_id:
            _span_id += "_"
        _span_id += f"L{self.start_line}"

        if self.end_line:
            _span_id += f"_L{self.end_line}"

        return _span_id

    def __hash__(self):
        return hash((self.start_line, self.end_line)) + hash(str(self.block_path))


class ContextFile(BaseModel):
    file_path: str
    spans: List[Span] = None


class CodingTask(BaseModel):
    file_path: str
    instructions: str
    span: Optional[Span] = None
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
