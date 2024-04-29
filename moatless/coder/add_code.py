import logging

from pydantic import Field

from moatless.codeblocks.codeblocks import (
    CodeBlockTypeGroup,
    CodeBlockType,
    CodeBlock,
    BlockPath,
)
from moatless.codeblocks.module import Module
from moatless.coder.code_action import CodeAction, respond_with_invalid_block
from moatless.coder.code_utils import find_start_and_end_index
from moatless.coder.types import FunctionResponse, CodeFunction
from moatless.settings import Settings

logger = logging.getLogger(__name__)


class AddCodeBlock(CodeFunction):
    """
    Add a new function or class after one span. You can provide a start line number if the new code should be added
    after a specific line.
    """

    instructions: str = Field(description="Instructions for what code to add")
    file_path: str = Field(description="Path to the file that should be updated")
    span_id: str = Field(description="The span ID")
    position: str = Field(
        description="Where to add the new code in relation to the provided span id.",
        enum=["after", "before", "inside"],
    )
    block_type: str = Field(
        description="The type of block to add", enum=["function", "class"]
    )
    # TODO: before_span_id: Optional[str]
    # TODO: line_number?


class AddCodeAction(CodeAction):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _execute(
        self, task: AddCodeBlock, original_module: Module, updated_module: Module
    ) -> FunctionResponse:
        logger.info(
            f"Add code in file {task.file_path}, {task.position} span {task.span_id}"
        )

        first_block_in_span = original_module.find_first_by_span_id(task.span_id)

        if task.position == "inside":
            parent_block = first_block_in_span
            if parent_block.type not in [CodeBlockType.FUNCTION, CodeBlockType.CLASS]:
                return respond_with_invalid_block(
                    f"Can't add a {task.block_type} inside a {parent_block.type.value}",
                    error_type="invalid_block_type",
                )
        else:
            parent_block = first_block_in_span.parent

        updated_parent_block = updated_module.find_by_path(parent_block.full_path())

        if not updated_parent_block:
            logger.warning(
                f"Couldn't find parent block with path {parent_block.path_string()}"
            )
            return respond_with_invalid_block(
                f"Couldn't find the parent block on {parent_block.path_string()}",
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
            and child.belongs_to_span.span_id != task.span_id
        ]

        if not updated_child_blocks:
            logger.warning(
                f"Couldn't find expected block in {updated_parent_block.path_string()}"
            )
            return respond_with_invalid_block(
                f"Couldn't find expected block in {updated_parent_block.path_string()}",
                error_type="block_not_found",
            )

        if task.position == "inside":
            parent_block.append_children(updated_child_blocks)
        else:
            start_index, end_index = find_start_and_end_index(
                parent_block, task.span_id
            )

            end_index = min(end_index, len(parent_block.children))

            if task.position == "after":
                end_index += 1

            parent_block.insert_children(end_index, updated_child_blocks)

        return FunctionResponse()

    def _to_instruction_code_prompt(
        self, code_block: CodeBlock, relative_block_path: BlockPath, position: str
    ):
        contents = ""
        if len(relative_block_path) == 0 and position == "inside":
            contents += (
                code_block.children[0]
                .create_comment_block("Write the new code here...", pre_lines=2)
                .to_string()
            )
        elif len(relative_block_path) == 1 and position != "inside":
            relative_block = code_block.find_by_path(relative_block_path)
            comment = relative_block.create_comment_block(
                "Write the new code here...", pre_lines=2
            ).to_string()

            relative_block_content = relative_block._to_prompt_string()
            relative_block_content += (
                relative_block.children[0]
                .create_comment_block("... other code")
                .to_string()
            )

            if position == "before":
                contents += comment
                contents += relative_block_content
            else:
                contents += relative_block_content
                contents += comment
        else:
            for i, child in enumerate(code_block.children):
                show_child = child.identifier == relative_block_path[0]
                if show_child:
                    contents += child._to_prompt_string()
                    contents += self._to_instruction_code_prompt(
                        code_block=child,
                        relative_block_path=relative_block_path[1:],
                        position=position,
                    )

        return contents

    def _task_instructions(self, task: AddCodeBlock, module: Module) -> str:
        first_block_in_span = module.find_first_by_span_id(task.span_id)

        code = self._to_instruction_code_prompt(
            code_block=module,
            relative_block_path=first_block_in_span.full_path(),
            position=task.position,
        )
        code = code.strip()

        return f"""{task.instructions}

# Write the new code. Leave all placeholder comments as is.

```python
{code}
```
"""
