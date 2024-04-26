from typing import Optional, List

from pydantic import BaseModel

from moatless.codeblocks import CodeBlock


class CodingTask(BaseModel):
    file_path: str
    instructions: str
    span_id: Optional[str] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    action: str = "update"  # add, remove, update
    state: str = "planned"  # completed, planned, rejected, failed


class UpdateCodeTask(CodingTask):
    action: str = "update"
    file_path: Optional[str] = None
    block_path: Optional[List[str]] = None
    start_index: Optional[int] = None
    end_index: Optional[int] = None
