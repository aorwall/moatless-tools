import importlib
import logging
from abc import ABC, abstractmethod
from typing import Any, List

from pydantic import BaseModel

from moatless.actions.model import ActionArguments
from moatless.node import Node, FeedbackData

logger = logging.getLogger(__name__)


class BaseFeedbackGenerator(BaseModel, ABC):
    @abstractmethod
    def generate_feedback(
        self, node: Node, actions: List[ActionArguments] | None = None
    ) -> FeedbackData | None:
        """Generate feedback based on the node."""
        pass

    def model_dump(self, **kwargs) -> dict[str, Any]:
        dump = super().model_dump(**kwargs)
        dump["feedback_class"] = (
            f"{self.__class__.__module__}.{self.__class__.__name__}"
        )
        return dump

    @classmethod
    def model_validate(cls, obj: Any) -> "FeedbackGenerator":
        if isinstance(obj, dict):
            obj = obj.copy()
            feedback_class_path = obj.pop(
                "feedback_class"
            )

            try:
                module_name, class_name = feedback_class_path.rsplit(".", 1)
                module = importlib.import_module(module_name)
                feedback_class = getattr(module, class_name)
                return feedback_class(**obj)
            except (ImportError, AttributeError) as e:
                logger.warning(
                    f"Failed to load feedback generator class {feedback_class_path}, defaulting to RewardFeedbackGenerator: {e}"
                )
                raise e

        return super().model_validate(obj)
