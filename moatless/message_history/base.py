from abc import abstractmethod

from moatless.completion.schema import AllMessageValues
from moatless.component import MoatlessComponent
from moatless.node import Node
from moatless.workspace import Workspace


class BaseMemory(MoatlessComponent):
    @abstractmethod
    async def generate_messages(self, node: Node, workspace: Workspace) -> list[AllMessageValues]:
        pass

    @classmethod
    def get_component_type(cls) -> str:
        return "memory"

    @classmethod
    def _get_package(cls) -> str:
        return "moatless.message_history"

    @classmethod
    def _get_base_class(cls) -> type:
        return BaseMemory
