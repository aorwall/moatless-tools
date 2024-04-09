from collections import namedtuple
from typing import List, Optional

from pydantic import BaseModel

BlockPath = List[str]




class ContextFile(BaseModel):
    file_path: str
    block_paths: List[BlockPath] = None


class DevelopmentTask(BaseModel):
    file_path: str
    instructions: str
    block_path: BlockPath = None
    action: Optional[str] = "update"  # add, remove, update
    state: Optional[str] = "planned"  # completed, planned, rejected, failed


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class BaseResponse(BaseModel):
    thoughts: Optional[str] = None
    error: Optional[str] = None
    usage_stats: List[Usage] = []

