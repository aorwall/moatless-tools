from pydantic import BaseModel


class CodeFile(BaseModel):
    file_path: str
    content: str = None
    is_complete: bool = False
