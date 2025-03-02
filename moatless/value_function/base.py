import importlib
import logging
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple, Type, TypeVar, cast

from pydantic import BaseModel

from moatless.completion.base import BaseCompletionModel
from moatless.completion.model import Completion
from moatless.node import Node, Reward
from moatless.component import MoatlessComponent

logger = logging.getLogger(__name__)

# TypeVar for ValueFunction types
VF = TypeVar("VF", bound="BaseValueFunction")


class BaseValueFunction(MoatlessComponent[VF]):
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
    def _get_base_class(cls) -> Type["BaseValueFunction"]:
        return BaseValueFunction
