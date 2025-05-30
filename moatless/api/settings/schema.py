from typing import Optional

from moatless.flow.flow import AgenticFlow
from pydantic import BaseModel


class FlowConfigUpdateDTO(BaseModel):
    description: Optional[str] = None


class FlowConfigsResponseDTO(BaseModel):
    configs: list[AgenticFlow]
