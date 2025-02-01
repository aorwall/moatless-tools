from typing import List, Optional
from pydantic import BaseModel, Field
from moatless.api.trajectory.schema import TrajectoryDTO


class SWEBenchInstanceDTO(BaseModel):
    """Schema for a SWEBench instance"""

    instance_id: str = Field(..., description="Unique identifier for the instance")
    problem_statement: str = Field(..., description="Problem statement for the instance")
    resolved_count: int = Field(..., description="Number of agents that have resolved this instance")


class SWEBenchValidationRequestDTO(BaseModel):
    """Schema for validation request"""

    instance_id: str = Field(..., description="ID of the instance to validate")
    model_id: str = Field(..., description="ID of the model to use")
    agent_id: str = Field(..., description="ID of the agent to use")
    max_iterations: int = Field(15, description="Maximum number of iterations")
    max_cost: Optional[float] = Field(1.0, description="Maximum cost of the validation")

class SWEBenchValidationResponseDTO(BaseModel):
    """Schema for validation response"""
    run_id: str = Field(..., description="Unique identifier for the validation")


class SWEBenchInstancesResponseDTO(BaseModel):
    """Response containing all available SWEBench instances"""

    instances: List[SWEBenchInstanceDTO] = Field(..., description="List of available instances")
