import importlib
import logging
from abc import ABC, abstractmethod
from typing import Any, List, Type

from pydantic import BaseModel

from moatless.actions.schema import ActionArguments
from moatless.completion.base import BaseCompletionModel
from moatless.node import Node, FeedbackData
from moatless.component import MoatlessComponent

logger = logging.getLogger(__name__)


class BaseFeedbackGenerator(MoatlessComponent):
    @abstractmethod
    async def generate_feedback(self, node: Node) -> FeedbackData | None:
        """Generate feedback based on the node."""
        pass

    @classmethod
    def get_component_type(cls) -> str:
        return "feedback_generator"

    @classmethod
    def _get_package(cls) -> str:
        return "moatless.feedback"

    @classmethod
    def _get_base_class(cls) -> Type:
        return BaseFeedbackGenerator
