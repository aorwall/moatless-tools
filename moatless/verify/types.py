from pydantic import BaseModel


class VerificationError(BaseModel):
    code: str
    file_path: str
    message: str
    line: int
