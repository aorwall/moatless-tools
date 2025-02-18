import importlib
import logging
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple, Type

from pydantic import BaseModel

from moatless.completion.base import BaseCompletionModel
from moatless.completion.model import Completion
from moatless.node import Node, Reward
from moatless.flow.base import FlowComponentMixin
from moatless.component import MoatlessComponent

logger = logging.getLogger(__name__)


class BaseValueFunction(MoatlessComponent):
    # Add this class variable to tell Pydantic to use model_validate
    model_config = {
        "from_attributes": True,
    }

    @abstractmethod
    async def get_reward(self, node: Node) -> Tuple[Reward, Optional[Completion]]:
        raise NotImplementedError("get_reward method must be implemented")

    @classmethod
    def get_component_type(cls) -> str:
        return "value_function"

    @classmethod
    def _get_package(cls) -> str:
        return "moatless.value_function"

    @classmethod
    def _get_base_class(cls) -> Type:
        return BaseValueFunction
    
    @classmethod
    def model_validate(cls, data: Any) -> "BaseValueFunction":
        return super().model_validate(data)

    def model_dump(self, *args, **kwargs):
        data = super().model_dump(*args, **kwargs)
        data["value_function_class"] = self.__class__.__module__ + "." + self.__class__.__name__
        return data
