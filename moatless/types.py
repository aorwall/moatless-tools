from collections import namedtuple
from typing import List, Optional

from pydantic import BaseModel

from moatless.codeblocks.codeblocks import BlockSpan


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

        if self.is_partial:
            _span_id += f"_L{self.start_line}-L{self.end_line}"

        return _span_id

    def __hash__(self):
        return hash((self.start_line, self.end_line)) + hash(str(self.block_path))


class ContextFile(BaseModel):
    file_path: str
    span_ids: List[str] = None  # TODO: Define relevant spans in a central place


class CodingTask(BaseModel):
    file_path: str
    instructions: str
    span: Optional[BlockSpan] = None
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
