from typing import Optional

from moatless.completion.manager import BaseCompletionModel
from pydantic import BaseModel, Field


class ModelsResponseDTO(BaseModel):
    """Response model for listing all models"""

    models: list[dict] = Field(..., description="List of model configurations")


class AddModelFromBaseDTO(BaseModel):
    """Request model for adding a new model from a base model"""

    base_model_id: str = Field(..., description="ID of the base model to copy from")
    new_model_id: str = Field(..., description="ID for the new model")
    updates: Optional[BaseCompletionModel] = Field(None, description="Optional configuration updates to apply")
