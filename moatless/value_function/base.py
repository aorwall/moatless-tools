import importlib
import logging
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

from pydantic import BaseModel

from moatless.completion.base import BaseCompletionModel
from moatless.completion.model import Completion
from moatless.node import Node, Reward

logger = logging.getLogger(__name__)


class BaseValueFunction(BaseModel, ABC):
    @abstractmethod
    def get_reward(self, node: Node) -> Tuple[Reward, Optional[Completion]]:
        raise NotImplementedError("get_reward method must be implemented")

    @classmethod
    def model_validate(cls, obj: Any):
        if isinstance(obj, dict):
            obj = obj.copy()
            value_function_class_path = obj.pop("value_function_class", None)

            if value_function_class_path:
                module_name, class_name = value_function_class_path.rsplit(".", 1)
                module = importlib.import_module(module_name)
                value_function_class = getattr(module, class_name)
                logger.info(f"Value function class: {value_function_class}")

                if "completion_model" in obj:
                    obj["completion_model"] = BaseCompletionModel.model_validate(obj["completion_model"])

                instance = value_function_class.model_validate(obj)
            else:
                return None
                # raise ValueError("value_function_class is required in {obj}")

            return instance

        return obj

    def model_dump(self, *args, **kwargs):
        data = super().model_dump(*args, **kwargs)
        data["value_function_class"] = self.__class__.__module__ + "." + self.__class__.__name__
        return data
