import logging

from pydantic import Field

from moatless.coder.types import FunctionResponse, CoderResponse, Function
from moatless.file_context import FileContext

logger = logging.getLogger(__name__)


class RemoveCode(Function):
    """
    Remove all code in one span. If parts of the code should be removed you should use the function 'update'.
    """

    file_path: str = Field(description="Path to the file that should be updated")
    span_id: str = Field(description="The ID of the span that should be removed")


class RemoveCodeAction:

    def __init__(self, file_context: FileContext = None):
        self._file_context = file_context

    def execute(self, task: RemoveCode):
        file = self._file_context.get_file(task.file_path)
        if not file:
            logger.error(f"File {task.file_path} not found in file context.")
            return CoderResponse(
                file_path=task.file_path,
                error="The provided file isn't found in the file context.",
            )

        original_module = file.module
        first_block = original_module.find_first_by_span_id(task.span_id)

        parent_block = first_block.parent
        logger.info(
            f"Will remove span {task.span_id} from block path {parent_block.full_path()}"
        )

        remaining_children = []
        for child in parent_block.children:
            if (
                not child.belongs_to_span
                or child.belongs_to_span.span_id != task.span_id
            ):
                remaining_children.append(child)

        logger.info(
            f"Will remove {len(parent_block.children) - len(remaining_children)} children from {parent_block.full_path()}"
        )

        parent_block.children = remaining_children

        return FunctionResponse()
