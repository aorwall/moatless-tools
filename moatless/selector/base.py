import logging
from abc import abstractmethod

from moatless.component import MoatlessComponent
from moatless.node import Node

logger = logging.getLogger(__name__)


class BaseSelector(MoatlessComponent):
    @abstractmethod
    async def select(self, expandable_nodes: list[Node]) -> Node | None:
        if not expandable_nodes:
            return None

        return expandable_nodes[0]

    @classmethod
    def get_component_type(cls) -> str:
        return "selector"

    @classmethod
    def _get_package(cls) -> str:
        return "moatless.selector"

    @classmethod
    def _get_base_class(cls) -> type:
        return BaseSelector
