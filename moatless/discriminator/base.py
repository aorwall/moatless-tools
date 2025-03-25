import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Type

from moatless.component import MoatlessComponent
from moatless.node import Node
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class BaseDiscriminator(MoatlessComponent, ABC):
    @abstractmethod
    def select(self, nodes: list[Node]) -> Optional[Node]:
        raise NotImplementedError

    @classmethod
    def get_component_type(cls) -> str:
        return "discriminator"

    @classmethod
    def _get_package(cls) -> str:
        return "moatless.discriminator"

    @classmethod
    def _get_base_class(cls) -> type:
        return BaseDiscriminator
