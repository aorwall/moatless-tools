from typing import Optional

from moatless.flow.schema import FlowConfig
from pydantic import BaseModel


class FlowConfigUpdateDTO(BaseModel):
    description: Optional[str] = None


class FlowConfigsResponseDTO(BaseModel):
    configs: list[FlowConfig]
