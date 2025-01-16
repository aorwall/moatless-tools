import logging
from abc import ABC
from typing import Optional, Tuple

from pydantic import BaseModel

from moatless.completion.model import Completion
from moatless.node import Node, Reward

logger = logging.getLogger(__name__)


class BaseValueFunction(BaseModel, ABC):

    def get_reward(self, node: Node) -> Tuple[Reward, Optional[Completion]]:
        pass
