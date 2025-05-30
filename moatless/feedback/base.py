import logging
from abc import abstractmethod

from moatless.component import MoatlessComponent
from moatless.node import FeedbackData, Node

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
    def _get_base_class(cls) -> type:
        return BaseFeedbackGenerator
