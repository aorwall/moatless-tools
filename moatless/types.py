from typing import List

from pydantic import BaseModel

BlockPath = List[str]

class CodeFile(BaseModel):
    file_path: str
    content: str = None
    is_complete: bool = False


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
