import logging
from abc import ABC
from typing import List

from pydantic import BaseModel

from moatless.node import Node

logger = logging.getLogger(__name__)

class BaseSelector(BaseModel, ABC):
    def select(self, expandable_nodes: List[Node]) -> Node | None:
        if not expandable_nodes:
            return None

        return expandable_nodes[0]
