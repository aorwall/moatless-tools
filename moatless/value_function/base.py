import logging
from abc import ABC
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from moatless.completion.model import Completion
from moatless.node import Node

logger = logging.getLogger(__name__)


class BaseValueFunction(BaseModel, ABC):

    def get_reward(self, node: Node) -> Tuple[Reward, Optional[Completion]]:
        pass
