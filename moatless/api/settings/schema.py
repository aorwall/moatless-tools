from typing import Optional

from pydantic import BaseModel

from moatless.flow.schema import FlowConfig


class FlowConfigUpdateDTO(BaseModel):
    description: Optional[str] = None


class FlowConfigsResponseDTO(BaseModel):
    configs: list[FlowConfig]
