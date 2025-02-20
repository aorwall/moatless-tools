from pydantic import BaseModel, Field
from typing import Any, Dict, Optional, Literal
import logging
from moatless.discriminator.base import BaseDiscriminator
from moatless.expander import Expander
from moatless.feedback.base import BaseFeedbackGenerator
from moatless.selector.base import BaseSelector
from moatless.value_function.base import BaseValueFunction
from pydantic import model_validator

logger = logging.getLogger(__name__)

class FlowConfig(BaseModel):
    """Configuration for a tree search instance."""
    id: str = Field(..., description="Unique identifier for the flow")
    description: Optional[str] = Field(None, description="Optional description of the flow")
    
    flow_type: Literal["tree", "loop"] = Field(..., description="Type of flow - tree or loop")
    
    # Common fields for both types
    max_iterations: int = Field(100, description="Maximum number of iterations")
    max_cost: float = Field(4.0, description="Maximum cost allowed in USD")
    agent_id: Optional[str] = Field(None, description="ID of the agent to use")
    
    # Tree-specific fields
    max_expansions: Optional[int] = Field(3, description="Maximum number of expansions per iteration")
    max_depth: Optional[int] = Field(20, description="Maximum depth of the tree")
    min_finished_nodes: Optional[int] = Field(None, description="Minimum number of finished nodes required")
    max_finished_nodes: Optional[int] = Field(None, description="Maximum number of finished nodes allowed")
    reward_threshold: Optional[float] = Field(None, description="Minimum reward threshold for accepting nodes")
    
    # Component references
    selector: Optional[BaseSelector] = None
    expander: Optional[Expander] = None
    value_function: Optional[BaseValueFunction] = None
    feedback_generator: Optional[BaseFeedbackGenerator] = None
    discriminator: Optional[BaseDiscriminator] = None

    def __str__(self) -> str:
        """Return a nice string representation of the flow config."""
        components = []
        if self.description:
            components.append(f"Description: {self.description}")

        components.extend([
            f"Type: {self.flow_type}",
            f"Max iterations: {self.max_iterations}",
            f"Max cost: ${self.max_cost:.2f}"
        ])

        if self.flow_type == "tree":
            components.extend([
                f"Max expansions: {self.max_expansions}",
                f"Max depth: {self.max_depth}"
            ])
            
            if self.min_finished_nodes:
                components.append(f"Min finished nodes: {self.min_finished_nodes}")
            if self.max_finished_nodes:
                components.append(f"Max finished nodes: {self.max_finished_nodes}")
            if self.reward_threshold:
                components.append(f"Reward threshold: {self.reward_threshold}")

        if self.agent_id:
            components.append(f"Agent: {self.agent_id}")

        # Add component names if present
        if self.selector:
            components.append(f"Selector: {self.selector.__class__.__name__}")
        if self.expander:
            components.append(f"Expander: {self.expander.__class__.__name__}")
        if self.value_function:
            components.append(f"Value function: {self.value_function.__class__.__name__}")
        if self.feedback_generator:
            components.append(f"Feedback generator: {self.feedback_generator.__class__.__name__}")
        if self.discriminator:
            components.append(f"Discriminator: {self.discriminator.__class__.__name__}")

        return f"Flow Config '{self.id}':\n" + "\n".join(f"- {c}" for c in components)

    model_config = {
        "json_encoders": {
            BaseSelector: lambda v: v.model_dump(),
            BaseValueFunction: lambda v: v.model_dump(),
            BaseFeedbackGenerator: lambda v: v.model_dump(),
            BaseDiscriminator: lambda v: v.model_dump(),
        }
    }

    @model_validator(mode='before')
    @classmethod
    def validate_components(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(data, dict):
            data = data.copy()
            
            if "selector" in data and data["selector"]:
                data["selector"] = BaseSelector.model_validate(data["selector"])
            if "value_function" in data and data["value_function"]:
                data["value_function"] = BaseValueFunction.model_validate(data["value_function"])
            if "feedback_generator" in data and data["feedback_generator"]:
                data["feedback_generator"] = BaseFeedbackGenerator.model_validate(data["feedback_generator"])
            if "discriminator" in data and data["discriminator"]:
                data["discriminator"] = BaseDiscriminator.model_validate(data["discriminator"])

        return data

    def model_dump(self, *args, **kwargs) -> Dict[str, Any]:
        data = super().model_dump(*args, **kwargs)
        
        if self.selector:
            selector_data = self.selector.model_dump()
            selector_data["selector_class"] = f"{self.selector.__class__.__module__}.{self.selector.__class__.__name__}"
            data["selector"] = selector_data
            
        if self.value_function:
            value_data = self.value_function.model_dump()
            value_data["value_function_class"] = f"{self.value_function.__class__.__module__}.{self.value_function.__class__.__name__}"
            data["value_function"] = value_data
            
        if self.feedback_generator:
            feedback_data = self.feedback_generator.model_dump()
            feedback_data["feedback_generator_class"] = f"{self.feedback_generator.__class__.__module__}.{self.feedback_generator.__class__.__name__}"
            data["feedback_generator"] = feedback_data
            
        return data
