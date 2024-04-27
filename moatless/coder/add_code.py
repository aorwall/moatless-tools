import logging

from instructor import OpenAISchema
from pydantic import Field

from moatless.codeblocks.codeblocks import CodeBlockTypeGroup, CodeBlockType
from moatless.codeblocks.module import Module
from moatless.coder.code_action import CodeAction, respond_with_invalid_block
from moatless.coder.code_utils import find_start_and_end_index
from moatless.coder.types import WriteCodeResult, CodeFunction
from moatless.file_context import FileContext
from moatless.settings import Settings

logger = logging.getLogger(__name__)


class AddCode(CodeFunction):
    """
    Add new code after one span. You can provide a start line number if the new code should be added
    after a specific line.
    """

    instructions: str = Field(description="Instructions for what code to add")
    file_path: str = Field(description="Path to the file that should be updated")
    after_span_id: str = Field(
        description="The span ID after which the new code should be added"
    )
    # TODO: before_span_id: Optional[str]
    # TODO: line_number?


class AddCodeAction(CodeAction):

    def __init__(self, file_context: FileContext = None):
        super().__init__(file_context)

    def _execute(
        self, task: AddCode, original_module: Module, updated_module: Module
    ) -> WriteCodeResult:
        last_block = original_module.find_last_by_span_id(task.after_span_id)

        last_parent_structure = last_block.find_type_group_in_parents(
            CodeBlockTypeGroup.STRUCTURE
        )

        updated_parent_block = updated_module.find_by_path(
            last_parent_structure.full_path()
        )

        if not updated_parent_block:
            last_parent_structure = last_parent_structure.parent

            print(
                f"Couldn't find parent block with path {last_parent_structure.full_path()}, will try {last_parent_structure.full_path()[:-1]}"
            )
            updated_parent_block = updated_module.find_by_path(
                last_parent_structure.full_path()
            )

            if not updated_parent_block:
                logger.warning(
                    f"Couldn't find parent block with path {last_parent_structure.full_path()}"
                )
                return respond_with_invalid_block(
                    f"Couldn't find the parent block with path {last_parent_structure.full_path()}",
                    error_type="parent_block_not_found",
                )

        if Settings.coder.debug_mode:
            logger.debug(
                f"Adding new block: \n{updated_parent_block.to_tree(show_spans=True, show_tokens=True)}"
            )

        updated_child_blocks = [
            child
            for child in updated_parent_block.children
            if child.type not in [CodeBlockType.COMMENTED_OUT_CODE]
        ]

        if not updated_child_blocks:
            logger.warning(
                f"Couldn't find expected block with path {last_block.path_string()}"
            )
            return respond_with_invalid_block(
                f"Couldn't find the expected block with path {last_block.path_string()}",
                error_type="block_not_found",
            )

        start_index, end_index = find_start_and_end_index(
            last_parent_structure, task.span_id
        )
        end_index = min(end_index + 1, len(last_parent_structure.children))

        for changed_block in updated_child_blocks:
            last_parent_structure.children.insert(end_index, changed_block)
            end_index += 1

        return WriteCodeResult()

    def _task_instructions(self, task: AddCode, module: Module) -> str:
        first_block = module.find_first_by_span_id(task.after_span_id)

        if first_block.parent.parent:
            trimmed_parent = first_block.parent.copy_with_trimmed_parents()
            trimmed_parent.children = [
                trimmed_parent.create_comment_block("Write implementation here...")
            ]
            code = trimmed_parent.to_prompt()
        else:
            code = "# Write implementation here..."

        code = code.strip()

        return f"""{task.instructions}

    # Write the new code. Leave all placeholder comments as is.

    ```python
    {code}
    ```"""
