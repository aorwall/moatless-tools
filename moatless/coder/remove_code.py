import logging

from instructor import OpenAISchema
from pydantic import Field

from moatless.coder.types import WriteCodeResult, CoderResponse, CodeFunction
from moatless.file_context import FileContext

logger = logging.getLogger(__name__)


class RemoveCode(CodeFunction):
    """
    Remove all code in one span. If parts of the code should be removed you should use the function 'update'.
    """

    file_path: str = Field(description="Path to the file that should be updated")
    span_id: str = Field(description="The ID of the span that should be removed")


class RemoveCodeAction:

    def __init__(self, file_context: FileContext = None):
        self._file_context = file_context

    def execute(self, task: RemoveCode):
        if self._file_context.is_in_context(task.file_path):
            logger.error(f"File {task.file_path} not found in file context.")
            return CoderResponse(
                file_path=task.file_path,
                error="The provided file isn't found in the file context.",
            )

        original_module = self._file_context.get_module(task.file_path)
        span = original_module.find_by_span_id(task.span_id)

        first_path_in_span = span.block_paths[0]
        last_path_in_span = span.block_paths[-1]

        logger.info(
            f"Will remove span {span.span_id} from block path {first_path_in_span} to {last_path_in_span}"
        )

        if len(first_path_in_span) > 1:
            parent_block = original_module.find_by_path(first_path_in_span[:-1])
        else:
            parent_block = original_module

        remaining_children = []
        for child in parent_block.children:
            if (
                not child.belongs_to_span
                or child.belongs_to_span.span_id != span.span_id
            ):
                remaining_children.append(child)

        parent_block.children = remaining_children

        return WriteCodeResult()
