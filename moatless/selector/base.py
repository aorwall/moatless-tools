import importlib
import logging
from abc import ABC
from typing import Any, List, Type

from pydantic import BaseModel

from moatless.node import Node
from moatless.flow.base import FlowComponentMixin
from moatless.component import MoatlessComponent

logger = logging.getLogger(__name__)


class BaseSelector(MoatlessComponent):
    async def select(self, expandable_nodes: List[Node]) -> Node | None:
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
    def _get_base_class(cls) -> Type:
        return BaseSelector

    @classmethod
    def model_validate(cls, obj: Any):
        if isinstance(obj, dict):
            obj = obj.copy()
            selector_class_path = obj.pop("selector_class", None)

            if selector_class_path:
                module_name, class_name = selector_class_path.rsplit(".", 1)
                module = importlib.import_module(module_name)
                selector_class = getattr(module, class_name)
                instance = selector_class(**obj)
            else:
                instance = cls(**obj)

            return instance

        return super().model_validate(obj)

    def model_dump(self, *args, **kwargs):
        data = super().model_dump(*args, **kwargs)
        data["selector_class"] = self.__class__.__module__ + "." + self.__class__.__name__
        return data
