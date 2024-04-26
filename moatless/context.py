from typing import List

from pydantic import BaseModel

from moatless.codeblocks.module import Module


class ContextFile(BaseModel):
    file_path: str
    module: Module
    span_ids: List[str] = None
