import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Type

from pydantic import BaseModel

from moatless.node import Node
from moatless.component import MoatlessComponent

logger = logging.getLogger(__name__)


class BaseDiscriminator(MoatlessComponent, ABC):
    @abstractmethod
    def select(self, nodes: List[Node]) -> Optional[Node]:
        raise NotImplementedError

    @classmethod
    def get_component_type(cls) -> str:
        return "discriminator"

    @classmethod
    def _get_package(cls) -> str:
        return "moatless.discriminator"

    @classmethod
    def _get_base_class(cls) -> Type:
        return BaseDiscriminator
