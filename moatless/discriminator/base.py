import logging
from abc import ABC
from typing import List, Optional

from pydantic import BaseModel

from moatless.node import Node

logger = logging.getLogger(__name__)


class BaseDiscriminator(BaseModel, ABC):
    def select(self, nodes: List[Node]) -> Optional[Node]:
        raise NotImplementedError
