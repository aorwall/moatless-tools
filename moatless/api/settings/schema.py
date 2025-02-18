from pydantic import BaseModel
from typing import Optional, List

from moatless.flow.schema import FlowConfig

class FlowConfigUpdateDTO(BaseModel):
    description: Optional[str] = None

class FlowConfigsResponseDTO(BaseModel):
    configs: List[FlowConfig]

