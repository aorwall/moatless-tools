import logging
from enum import Enum
from typing import Optional, List, Union, Tuple, Any, Type, ClassVar

from pydantic import Field, PrivateAttr, model_validator

from moatless.actions.action import Action
from moatless.actions.model import (
    ActionArguments,
    FewShotExample,
    Observation,
)
from moatless.codeblocks import CodeBlock, get_parser_by_path, PythonParser
from moatless.codeblocks.codeblocks import CodeBlockTypeGroup, CodeBlockType
from moatless.codeblocks.module import Module
from moatless.completion.completion import CompletionModel
from moatless.completion.model import AssistantMessage, UserMessage, Completion
from moatless.file_context import FileContext, ContextFile
from moatless.repository.file import do_diff, remove_duplicate_lines
from moatless.repository.repository import Repository
from moatless.utils.tokenizer import count_tokens

logger = logging.getLogger(__name__)

ROLE_PROMPT = "You are autonomous AI assisistant with superior programming skills."

SEARCH_REPLACE_PROMPT = """# Objective:
Your task is to update the code within the `<search>` tags based on the provided `<instructions>` and `<pseudo_code>`. Follow these rules meticulously:

1. **Understanding Instructions and Pseudo Code:**
   - **Instructions:** Describe the specific changes that need to be made to the code.
   - **Pseudo Code:** Provides a code snippet illustrating the proposed modification or addition. It serves as a guide for how to implement the changes.
   - **Full Replacement:** Use the pseudo code and instructions to **completely replace** the entire content within the `<search>` tags. The `<pseudo_code>` may address only a subset of the block, but your replacement should ensure that the entire block reflects the necessary updates.

2. **Update Rules:**
   - **Implement Changes Fully:** Integrate all changes as specified in `<instructions>` and `<pseudo_code>`, ensuring the entire `<search>` block is updated accordingly.
   - **No Additional Changes:** Do not modify any part of the code outside the `<search>` tags or make changes not explicitly requested.

3. **Formatting and Indentation:**
   - **CRITICAL: Preserve Exact Indentation:** Maintain the EXACT indentation level of the original code within the `<search>` tags, including the very first line. Do not add any extra indentation to the entire block.
   - **Consistent Formatting:** Ensure that the formatting (spaces, line breaks) matches the original code structure precisely.

4. **Comments and Placeholders:**
   - **Retain Existing Comments:** Keep any existing placeholder comments (e.g., `# ... other code`) intact within the updated block.
   - **No New Comments:** Do not add comments describing your changes.

5. **Response Formatting:**
   - **Replacement Code:** Return the **entire updated code block** within `<replace>` tags, reflecting all necessary modifications.
   - **Empty Replacement:** If all code within `<search>` should be removed, return empty `<replace>` tags.
   - **Rejection:** If unable to make the changes or if instructions are unclear/incorrect, use `<reject>` tags with a clear reason.

# Response Format:

**Successful Replacement:**
<replace>
[Entire updated code block with all modifications applied, maintaining original indentation]
</replace>

**Empty Replacement:**
<replace>
</replace>

**Rejection:**
<reject>
[Reason for rejection]
</reject>

# IMPORTANT:

 * Do not include any code outside the <replace> tags.
 * Ensure the indentation matches EXACTLY with the original code inside <search> tags, including the first line.
 * Completely replace the entire <search> block with the updated code based on the instructions and pseudo code.
 * Do not add or remove any lines unless instructed.
 * Double-check that the first line of your <replace> block has the same indentation as the first line of the <search> block.
"""


class ChangeType(str, Enum):
    addition = "addition"
    modification = "modification"
    deletion = "deletion"


class RequestCodeChangeArgs(ActionArguments):
    """
    Apply a code change through an AI agent. This action instructs an AI assistant to
    modify code based on provided instructions and pseudo-code. The AI will analyze the existing code within
    the specified line range and apply changes while maintaining proper syntax, indentation, and context.
    """

    file_path: str = Field(..., description="The file path of the code to be updated.")
    instructions: str = Field(
        ...,
        description="Natural language instructions for the AI assistant describing the required code changes.",
    )
    pseudo_code: str = Field(
        ...,
        description="Example code snippet illustrating the desired changes. The AI will use this as a reference for implementing the modifications.",
    )
    change_type: ChangeType = Field(
        ...,
        description="Type of change to perform: 'addition' (insert new code), 'modification' (update existing code), or 'deletion' (remove code).",
    )
    start_line: int = Field(
        ...,
        description="The line number where the code change should begin. For additions, specifies the insertion point.",
    )
    end_line: int = Field(
        ...,
        description="The line number where the code change should end. For additions, specifies the insertion point.",
    )

    class Config:
        title = "RequestCodeChange"

    @model_validator(mode="before")
    @classmethod
    def set_missing_end_line(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if not data.get("end_line") and data.get("start_line"):
                data["end_line"] = data["start_line"]

        return data

    def equals(self, other: "RequestCodeChangeArgs") -> bool:
        if not isinstance(other, RequestCodeChangeArgs):
            return False

        return (
            self.file_path == other.file_path
            and self.pseudo_code == other.pseudo_code
            and self.change_type == other.change_type
            and self.start_line == other.start_line
            and self.end_line == other.end_line
        )


class RequestCodeChange(Action):
    args_schema: ClassVar[Type[ActionArguments]] = RequestCodeChangeArgs

    max_tokens_in_edit_prompt: int = Field(
        default=750,
        description="The maximum number of tokens allowed in the edit prompt.",
    )
    show_file_context: bool = Field(
        default=True, description="Whether to show the file context in the prompt."
    )

    _repository: Repository = PrivateAttr()
    _completion_model: CompletionModel = PrivateAttr()

    def __init__(
        self,
        repository: Repository | None = None,
        completion_model: CompletionModel | None = None,
        **data,
    ):
        super().__init__(**data)
        self._repository = repository
        self._completion_model = completion_model

    def execute(
        self, args: RequestCodeChangeArgs, file_context: FileContext
    ) -> Observation:
        logger.info(
            f"RequestCodeChange: file_path={args.file_path}, start_line={args.start_line}, end_line={args.end_line}, change_type={args.change_type}"
        )

        if not args.instructions:
            return Observation(
                message="Please provide instructions for the code change.",
                properties={"fail_reason": "no_instructions"},
                expect_correction=True,
            )

        if not args.pseudo_code:
            return Observation(
                message="Please provide pseudo code for the code change.",
                properties={"fail_reason": "no_pseudo_code"},
                expect_correction=True,
            )

        if not args.file_path.endswith(".py"):
            return Observation(
                message="Only Python files can be edited. Please provide a file path to a Python file.",
                properties={"fail_reason": "not_python_file"},
                expect_correction=True,
            )

        if args.change_type != ChangeType.addition and not self._repository.file_exists(
            args.file_path
        ):
            return Observation(
                message=f"File {args.file_path} not found.",
                properties={"fail_reason": "file_not_found"},
            )

        if self._repository.is_directory(args.file_path):
            return Observation(
                message=f"{args.file_path} is a directory. Please provide a file path.",
                properties={"fail_reason": "is_directory"},
            )

        if (
            not file_context.has_file(args.file_path)
            and args.change_type != ChangeType.addition
        ):
            context_file = file_context.get_file(args.file_path)
            message = f"File {args.file_path} is not in context."
            if context_file.module:
                message += f"At least one span must be added. Use RequestMoreContext to one ore more of the available spans: {self.span_id_list(context_file.module.span_ids)}"

            return Observation(
                message=message,
                properties={"fail_reason": "file_not_in_context"},
            )

        context_file = file_context.get_file(args.file_path)
        if not context_file:
            logger.info(
                f"File {args.file_path} is not found in the file repository. Will create it and add to context."
            )

            context_file = file_context.add_file(args.file_path)
            updated_content = args.pseudo_code
            return self._apply_changes(context_file, updated_content, args.file_path)
        else:
            # TODO: Verify if the code span is in context
            # TODO: Check for hallucinations
            retry_message = self.verify_request(context_file, args)
            if retry_message:
                extra = context_file.to_prompt(
                    show_line_numbers=True,
                    show_span_ids=True,
                    exclude_comments=False,
                    show_outcommented_code=False,
                    outcomment_code_comment="... other code",
                )
                return Observation(
                    message=retry_message, extra=extra, expect_correction=True
                )

            completion = None
            replace_block, args = self._get_replace_block_from_pseudo_code(
                args, context_file
            )
            if not replace_block:
                if context_file.module:
                    args.start_line, args.end_line, args.change_type = (
                        self.get_line_span(
                            args.change_type,
                            context_file,
                            args.start_line,
                            args.end_line,
                            self.max_tokens_in_edit_prompt,
                        )
                    )

                response, completion = self._generate_replace_block(
                    context_file, args, args.start_line, args.end_line
                )

                if "<reject>" in response:
                    rejection_message = response.split("<reject>")[1].split(
                        "</reject>"
                    )[0]
                    logger.info(
                        f"Rejected the instructions. Reason: {rejection_message}"
                    )
                    return Observation(
                        message=f"Failed to apply changes using search/replace blocks. {rejection_message}",
                        properties={"fail_reason": "rejected"},
                        execution_completion=completion,
                    )

                replace_block = response.split("<replace>")[1].split("</replace>")[0]

            observation = self._update_content(
                context_file,
                replace_block,
                args.start_line,
                args.end_line,
                args.change_type,
            )

            observation.execution_completion = completion
            return observation

    def create_replacement_block(
        self, messages: List[Union[UserMessage, AssistantMessage]]
    ) -> Tuple[str, Any]:
        try:
            replace_code, completion = self._completion_model.create_text_completion(
                messages=messages,
                system_prompt=self._system_prompt(),
            )

            return replace_code, completion
        except Exception as e:
            logger.exception(f"Error applying change. Retrying...")
            raise e

    def _system_prompt(self) -> str:
        system_prompt = ROLE_PROMPT

        system_prompt += "\n\n"
        system_prompt += SEARCH_REPLACE_PROMPT

        return system_prompt

    def _generate_replace_block(
        self,
        context_file: ContextFile,
        args: RequestCodeChangeArgs,
        start_line: int,
        end_line: int,
    ) -> Tuple[str, Completion]:
        span_ids = []
        span_to_update = context_file.module.find_spans_by_line_numbers(
            start_line, end_line
        )
        if span_to_update:
            # Pin the spans that are planned to be updated to context
            for span in span_to_update:
                if span.span_id not in span_ids:
                    span_ids.append(span.span_id)
            context_file.add_spans(span_ids=set(span_ids), pinned=True)

        logger.info(
            f"Requesting code change in {args.file_path} from {start_line} to {end_line}"
        )

        messages = []
        search_block = self.create_search_block(
            context_file, start_line, end_line, args.change_type
        )

        user_message = self.create_message(
            context_file,
            search_block,
            start_line,
            end_line,
            args.instructions,
            args.pseudo_code,
        )

        messages.append(UserMessage(content=user_message))
        response, completion = self._completion_model.create_text_completion(
            messages=messages,
            system_prompt=self._system_prompt(),
        )
        return response, completion

    def _update_content(
        self,
        context_file: ContextFile,
        replace_block: str,
        start_line: int,
        end_line: int | None,
        change_type: ChangeType,
    ) -> Observation:
        if replace_block:
            updated_content = self._update_content_by_line_numbers(
                context_file, start_line - 1, end_line, replace_block
            )

            updated_module = self._parse_module(context_file, updated_content)
            if not updated_module:
                invalid_response = "Code is invalid."
                invalid_reason = "invalid_syntax"
            else:
                indentation_fix = self._check_indentation(
                    context_file, updated_module, start_line, end_line
                )
                if indentation_fix:
                    replace_block = self._apply_indentation_fix(
                        replace_block, indentation_fix
                    )
                    updated_content = self._update_content_by_line_numbers(
                        context_file, start_line - 1, end_line, replace_block
                    )
                    updated_module = self._parse_module(context_file, updated_content)

                invalid_response, invalid_reason = self._verify_change(
                    updated_module, context_file, start_line, end_line, change_type
                )
                if not invalid_response:
                    output = self._apply_changes(
                        context_file, updated_content, context_file.file_path
                    )
                    return output

        else:
            invalid_response = "The code in the replace tag is empty."
            invalid_reason = "empty_replace_tag"

        logger.warning(f"Failed to apply changes. Reason: {invalid_response}")
        return Observation(
            message=f"Failed to apply changes using search/replace blocks. Reason: {invalid_response}"
            f"Verify that the right lines are provided and that the code that should changed is in the context.",
            properties={"fail_reason": invalid_reason},
        )

    def _parse_pseudo_code_block(self, args: RequestCodeChangeArgs):
        try:
            parser = PythonParser(apply_gpt_tweaks=True)
            return parser.parse(args.pseudo_code, file_path=args.file_path)
        except Exception as e:
            logger.warning(
                f"Failed to parse pseudo code with error {e}. Pseudo code:\n{args.pseudo_code}"
            )
            return None

    def create_message(
        self,
        file: ContextFile,
        search_block: str,
        start_line: int,
        end_line: int,
        instructions: str,
        pseudo_code: str,
    ) -> str:
        content = ""

        # TODO: Be able to include intial problem statement?
        # if self.show_initial_message:
        #    content = f"<main_objective>\n{self.initial_message}\n</main_objective>\n\n"

        if self.show_file_context:
            file_context = FileContext(repo=self._repository, max_tokens=3000)
            file_context.add_line_span_to_context(file.file_path, start_line, end_line)
            # file_context.expand_context_with_related_spans(self.max_prompt_file_tokens)

            file_context_str = file_context.create_prompt(
                show_line_numbers=True,
                show_span_ids=False,
                exclude_comments=False,
                show_outcommented_code=False,
                outcomment_code_comment="... other code",
            )

            content += f"\n<file_context>\n{file_context_str}\n</file_context>\n"

        content += f"\n<instructions>\n{instructions}\n</instructions>\n"

        if pseudo_code:
            content += f"\n<pseudo_code>\n{pseudo_code}\n</pseudo_code>\n"

        if file:
            content += f"<search>\n{search_block}\n</search>\n"
            if self.show_file_context:
                content += f"\nCode found on line numbers {start_line} to {end_line} in {file.file_path}:\n"
        else:
            content += "\n<search>\n# No content...\n</search>\n"

        return content

    def create_search_block(
        self, file: ContextFile, start_line: int, end_line: int, change_type: ChangeType
    ):
        code_lines = file.content.split("\n")
        lines_to_replace = code_lines[start_line - 1 : end_line]
        code_to_replace = "\n".join(lines_to_replace)
        if not code_to_replace and change_type != ChangeType.addition:
            logger.warning(
                f"No code found to replace in {file.file_path} from line {start_line} to {end_line}."
            )
        return code_to_replace

    def verify_request(
        self, context_file: ContextFile, args: RequestCodeChangeArgs
    ) -> Optional[str]:
        if not args.start_line:
            message = "You must specify the start line and end line of the code change in the variables start_line and end_line. If you want to update the first line in the file, set start line to 1. If you believe that the lines you want to edit isn't in the file context, you can request more context by providing the file path and the line numbers or span ids to the RequestMoreContext function."
            return message

        if not args.end_line:
            if args.change_type != ChangeType.addition:
                return f"If your intention is to modify an existing code span you must provide the end line for the code change in end_line."

            logger.info(f"End line not set, set to start line {args.start_line}")
            args.end_line = args.start_line

        pseudo_code_block = self._parse_pseudo_code_block(args)
        existing_hallucinated_spans = self.find_hallucinated_spans(
            pseudo_code_block, context_file, args.start_line, args.end_line
        )
        if existing_hallucinated_spans:
            context_file.add_spans(existing_hallucinated_spans)
            return f"""There where code in the pseudo code that wasn't present in the file context.
The following code spans where added to file context: {', '.join(existing_hallucinated_spans)}.
Please provide instructions for the code change again."""

        # Verify that the code that is supposed to be changed is in the context
        if context_file.module and args.change_type != ChangeType.addition:
            code_block = self.find_smallest_covering_block(
                context_file.module, args.start_line, args.end_line
            )
            if (
                code_block
                and code_block.belongs_to_span
                and code_block.belongs_to_span.span_id not in context_file.span_ids
            ):
                return f"The code span {code_block.belongs_to_span.span_id} between lines {args.start_line} - {args.end_line} is not in the context. Please use the RequestMoreContext to add the correct line numbers or span ids to context."
            # TODO: Handle if no code block is found

        code_lines = context_file.content.split("\n")
        lines_to_edit = code_lines[args.start_line - 1 : args.end_line + 1]
        code_to_edit = "\n".join(lines_to_edit)

        # Verify if change has already been implemented
        existing_content = code_to_edit.expandtabs()
        new_content = args.pseudo_code.expandtabs()
        occurrences = existing_content.count(new_content)
        if occurrences:
            return f"The pseudo code does already exist in the existing code base indicating that the change has already been implemented. Existing content at lines {args.start_line} : {args.end_line}\n{existing_content}"

        tokens = count_tokens(code_to_edit)
        if tokens > self.max_tokens_in_edit_prompt:
            clarify_msg = (
                f"The code span between lines {args.start_line} - {args.end_line} has {tokens} tokens, which is higher than the "
                f"maximum allowed {self.max_tokens_in_edit_prompt} tokens. "
            )
            logger.info(f"{clarify_msg}. Ask for clarification.")
            return f"The change request was rejected! {clarify_msg}. Narrow down the instructions and specify the exact part of the code that needs to be updated to fulfill the change. "

        return None

    def _get_replace_block_from_pseudo_code(
        self, args: RequestCodeChangeArgs, context_file: ContextFile
    ) -> Tuple[Optional[str], RequestCodeChangeArgs]:
        pseudo_code_block = self._parse_pseudo_code_block(args)
        if not pseudo_code_block:
            return None, args

        if (
            len(pseudo_code_block.has_placeholders()) > 0
            or len(pseudo_code_block.children) > 1
            or pseudo_code_block.children[0].type.group != CodeBlockTypeGroup.STRUCTURE
        ):
            return None, args

        pseudo_code_block = pseudo_code_block.children[0]

        existing_block = None
        if args.change_type == ChangeType.modification:
            block_to_replace = self._get_block_to_replace(
                context_file, args.start_line, args.end_line
            )
            if (
                block_to_replace
                and block_to_replace.type == pseudo_code_block.type
                and block_to_replace.identifier == block_to_replace.identifier
            ):
                logger.info(
                    f"Found existing block {block_to_replace.path_string()} at start line {args.start_line} to be replaced by the pseudo code."
                )
                existing_block = block_to_replace
        elif args.change_type == ChangeType.addition:
            block_at_start_line = context_file.module.find_first_by_start_line(
                args.start_line
            )
            if block_at_start_line:
                if block_at_start_line.type.group == CodeBlockTypeGroup.STRUCTURE:
                    logger.info(
                        f"Found existing block {block_at_start_line.path_string()} at start line {args.start_line}. Add the new block after this block on line {block_at_start_line.end_line}."
                    )
                    args.start_line = block_at_start_line.end_line + 1
                    args.end_line = args.start_line
                    existing_block = block_at_start_line
                else:
                    logger.warning(
                        f"Existing block {block_at_start_line.path_string()} at start line {args.start_line} has a different type than the pseudo code block {pseudo_code_block.path_string()}."
                    )
            else:
                structure_block = self.find_smallest_covering_block(
                    context_file.module, args.start_line, args.start_line
                )

                if structure_block and structure_block.type in [
                    CodeBlockType.CLASS,
                    CodeBlockType.MODULE,
                ]:
                    existing_blocks = structure_block.find_blocks_with_type(
                        pseudo_code_block.type
                    )
                    if existing_blocks:
                        # TODO: Pick the closest one
                        existing_block = existing_blocks[0]
                    else:
                        logger.warning(
                            f"No existing block of type {pseudo_code_block.type.display_name} found in {structure_block.path_string()}."
                        )

        if existing_block:
            indentation_diff = existing_block.compare_indentation(pseudo_code_block)
            if indentation_diff:
                replace_block = self._apply_indentation_fix(
                    args.pseudo_code, indentation_diff
                )
            else:
                replace_block = args.pseudo_code

            replace_block = "\n" + replace_block + "\n"
            return replace_block, args
        else:
            return None, args

    def _apply_changes(
        self, file: ContextFile, updated_content: str, file_path: str
    ) -> Observation:
        diff = do_diff(file_path, file.content, updated_content)

        if file.module:
            existing_span_ids = file.module.get_all_span_ids()

        if not diff:
            logger.info(f"No changes in {file_path}.")
            return Observation(
                message="Code wasn't updated, was the request code change the same as the existing code?",
                properties={"fail_reason": "no_changes"},
            )

        file.apply_changes(updated_content)

        return Observation(
            message=f"Applied the change to {file_path}\n\n```diff\n{diff}\n```",
            properties={"diff": diff},
        )

    def _update_content_by_line_numbers(
        self,
        file: ContextFile,
        start_line_index: int,
        end_line_index: int,
        replacement_content: str,
    ) -> str:
        replacement_lines = replacement_content.split("\n")
        while replacement_lines and replacement_lines[0].strip() == "":
            replacement_lines.pop(0)
        while replacement_lines and replacement_lines[-1].strip() == "":
            replacement_lines.pop()

        original_lines = file.content.split("\n")
        replacement_lines = remove_duplicate_lines(
            replacement_lines, original_lines[end_line_index:]
        )
        updated_lines = (
            original_lines[:start_line_index]
            + replacement_lines
            + original_lines[end_line_index:]
        )
        return "\n".join(updated_lines)

    def _parse_module(self, file: ContextFile, updated_content: str) -> Module | None:
        parser = get_parser_by_path(file.file_path)
        if not parser:
            raise ValueError(f"Parser not found for {file.file_path}")

        try:
            return parser.parse(updated_content)
        except Exception as e:
            logger.warning(
                f"Failed to parse updated content in {file.file_path}: {e}. Content:\n{updated_content}"
            )
            return None

    def _verify_change(
        self,
        updated_module: Module,
        file: ContextFile,
        start_line: int,
        end_line: int,
        change_type: Optional[ChangeType],
    ) -> Tuple[str, str]:
        existing_placeholders = file.module.find_blocks_with_type(
            CodeBlockType.COMMENTED_OUT_CODE
        )
        new_placeholders = (
            updated_module.find_blocks_with_type(CodeBlockType.COMMENTED_OUT_CODE)
            if not existing_placeholders
            else []
        )

        if new_placeholders:
            error_response = ""
            for new_placeholder in new_placeholders:
                parent_block = new_placeholder.find_type_group_in_parents(
                    CodeBlockTypeGroup.STRUCTURE
                )
                if parent_block and parent_block.type != CodeBlockType.MODULE:
                    error_response += f"{parent_block.identifier} has a placeholder `{new_placeholder.content}` indicating that it's not fully implemented."
                else:
                    error_response += f"There is a placeholder in the replace block indicating that it's not fully implemented. : \n```{new_placeholder.to_string()}\n```. \n"
            return error_response, "placeholders"

        if change_type == ChangeType.modification:
            existing_block = self._get_block_to_replace(file, start_line, end_line)
            if existing_block:
                new_block = updated_module.find_first_by_start_line(start_line)
                if existing_block.indentation != new_block.indentation:
                    return (
                        f"The code in the <replace> tag has an indentation of {len(new_block.indentation)} spaces while the code in the <search> tag has {len(existing_block.indentation)} spaces.",
                        "indentation",
                    )

                block_in_updated_code = file.module.find_by_path(
                    existing_block.full_path()
                )
                if existing_block.type != new_block.type and not (
                    block_in_updated_code
                    or block_in_updated_code.type != existing_block.type
                ):
                    return (
                        f"The code block {existing_block.identifier} in the <search> tag with the type {existing_block.type.display_name} was expected to be replaced. But the code provided in the <replace> tag has the type {new_block.type.display_name}.",
                        "block_type",
                    )

        return None, None

    def _get_block_to_replace(self, file: ContextFile, start_line: int, end_line: int):
        code_block = file.module.find_first_by_start_line(start_line)
        if (
            code_block
            and code_block.start_line == start_line
            and code_block.end_line == end_line
            and code_block.type.group == CodeBlockTypeGroup.STRUCTURE
        ):
            return code_block
        return None

    def find_hallucinated_spans(
        self,
        code_block: CodeBlock,
        context_file: ContextFile,
        start_line: int,
        end_line: int,
    ) -> set[str]:
        """
        Find out if the suggested code block contains any identifiers that are not present in the context.
        """

        existing_hallucinated_spans = set()
        for child_block in code_block.children:
            # Only verify structure blocks like classed and functions
            if child_block.type.group != CodeBlockTypeGroup.STRUCTURE:
                continue

            if child_block.type == CodeBlockType.CLASS:
                existing_hallucinated_spans.update(
                    self.find_hallucinated_spans(
                        child_block, context_file, start_line, end_line
                    )
                )

            # Check if the pseudo code identifier is part of any existing span_id
            if any(
                child_block.identifier in span_id for span_id in context_file.span_ids
            ):
                continue

            span_id = child_block.belongs_to_span.span_id
            existing_block = context_file.module.find_first_by_span_id(
                child_block.belongs_to_span.span_id
            )
            if existing_block:
                logger.info(
                    f"Checking if {span_id} is in context. Found {existing_block}"
                )
                existing_hallucinated_spans.add(span_id)
            else:
                if "." not in span_id:
                    # Check if there is child blocks with the span_id as identifier
                    child_blocks = context_file.module.find_blocks_with_identifier(
                        span_id
                    )

                    for child_block in child_blocks:
                        if context_file.has_span(child_block.belongs_to_span.span_id):
                            continue

                        parent_block = child_block.find_type_group_in_parents(
                            CodeBlockTypeGroup.STRUCTURE
                        )
                        if (
                            parent_block
                            and parent_block.type
                            in [CodeBlockType.CLASS, CodeBlockType.FUNCTION]
                            and parent_block.has_lines(start_line, end_line)
                        ) or child_block.is_within_lines(start_line, end_line):
                            logger.info(
                                f"Found child block {child_block.identifier} with {child_block.belongs_to_span.span_id} of {span_id} in context."
                            )
                            existing_hallucinated_spans.add(
                                child_block.belongs_to_span.span_id
                            )

        return existing_hallucinated_spans

    def find_smallest_covering_block(
        self, code_block: CodeBlock, start_line: int, end_line: int
    ) -> Optional[CodeBlock]:
        # If the code_block doesn't cover the lines, return None
        if code_block.start_line > start_line or code_block.end_line < end_line:
            return None

        # Check if any child block covers the lines
        for child in code_block.children:
            if child.start_line <= start_line and child.end_line >= end_line:
                # Found a smaller block that covers the lines
                smaller_block = self.find_smallest_covering_block(
                    child, start_line, end_line
                )

                if child.type.group == CodeBlockTypeGroup.STRUCTURE:
                    return smaller_block or child

        # No smaller block found, return the current block
        return code_block

    def find_lines_within_blocks(
        self, code_block: CodeBlock, start_line: int, end_line: int
    ) -> List[int]:
        # Collect lines from code blocks within max_tokens
        lines = []

        def traverse_blocks(block: CodeBlock):
            if block.end_line < start_line or block.start_line > end_line:
                return

            for child in block.children:
                traverse_blocks(child)

            # It's a code block within the line range
            if block.start_line >= start_line and block.end_line <= end_line:
                lines.extend(range(block.start_line, block.end_line + 1))
            elif (
                not block.children
                and block.end_line >= start_line
                and block.start_line <= end_line
            ):
                lines.extend(range(block.start_line, block.end_line + 1))

        traverse_blocks(code_block)
        return sorted(set(lines))

    def get_line_span(
        self,
        change_type: ChangeType,
        file: ContextFile,
        start_line: int,
        end_line: int,
        max_tokens: int,
    ) -> tuple[Optional[int], Optional[int], Optional[ChangeType]]:
        if not end_line:
            end_line = start_line

        structure_block = self.find_smallest_covering_block(
            file.module, start_line, end_line
        )
        if structure_block:
            logger.info(
                f"Found smallest covering block {structure_block.display_name} (start_line: {structure_block.start_line}, end_line: {structure_block.end_line}, tokens: {structure_block.sum_tokens()})"
            )

            if structure_block.type == CodeBlockType.CLASS:
                class_start_line, init_end_line, tokens = self.get_class_init_span(
                    structure_block
                )

                if (
                    class_start_line <= start_line <= end_line <= init_end_line
                    and tokens < max_tokens
                ):
                    logger.info(
                        f"Return class init block {structure_block.display_name} (start_line: {class_start_line}, end_line: {init_end_line}, tokens: {tokens})"
                    )
                    return class_start_line, init_end_line, change_type

            if structure_block.sum_tokens() < max_tokens:
                logger.info(
                    f"Return block {structure_block.display_name} (start_line: {structure_block.start_line}, end_line: {structure_block.end_line}, tokens: {structure_block.sum_tokens()}"
                )

                return structure_block.start_line, structure_block.end_line, change_type

        lines = self.find_lines_within_blocks(
            file.module, max(0, start_line - 5), min(file.module.end_line, end_line + 5)
        )
        if lines and len(lines) > 1:
            logger.info(
                f"Could not find start and end block for lines {start_line}-{end_line}. Return {lines[0]}-{lines[-1]}"
            )
            return lines[0], lines[-1], change_type
        else:
            logger.info(
                f"Could not find any lines within blocks for lines {start_line}-{end_line}. Returning original start and end lines."
            )
            return start_line, end_line, change_type

    def get_class_init_span(self, class_block: CodeBlock):
        """
        Get end line of the class initation span by including all lines until the first function or class
        """
        end_line = class_block.start_line + len(class_block.content_lines) - 1
        tokens = class_block.tokens
        for child in class_block.children:
            if (
                child.type.group == CodeBlockTypeGroup.STRUCTURE
                and child.type != CodeBlockType.CONSTRUCTOR
            ):
                break

            end_line = child.end_line
            tokens += child.tokens

        return class_block.start_line, end_line, tokens

    def _check_indentation(
        self,
        context_file: ContextFile,
        updated_module: Module,
        start_line: int,
        end_line: int,
    ) -> Optional[int]:
        existing_block = self._get_block_to_replace(context_file, start_line, end_line)
        if existing_block:
            new_block = updated_module.find_first_by_start_line(start_line)
            indentation_diff = existing_block.compare_indentation(new_block)
            if (
                indentation_diff != 0
                and new_block.identifier == existing_block.identifier
            ):
                logger.info(
                    f"Indentation difference detected: {indentation_diff} spaces on updated block {existing_block.identifier}"
                )
                return indentation_diff

        return None

    def _apply_indentation_fix(self, content: str, indentation_diff: int) -> str:
        lines = content.split("\n")
        if indentation_diff > 0:
            return "\n".join(" " * indentation_diff + line for line in lines)
        else:
            return "\n".join(line[-indentation_diff:] for line in lines)

    def span_id_list(self, span_ids: set[str]) -> str:
        list_str = ""
        for span_id in span_ids:
            list_str += f" * {span_id}\n"
        return list_str

    def model_dump(self, **kwargs):
        dump = super().model_dump(**kwargs)
        dump["completion_model"] = self._completion_model.model_dump(**kwargs)
        return dump

    @classmethod
    def model_validate(cls, obj: dict):
        if "completion_model" in obj and obj["completion_model"]:
            obj["completion_model"] = CompletionModel.model_validate(
                obj["completion_model"]
            )

        return cls(**obj)

    @classmethod
    def get_few_shot_examples(cls) -> List[FewShotExample]:
        return [
            FewShotExample.create(
                user_input="Add error handling to the process_payment method in the PaymentProcessor class",
                action=RequestCodeChangeArgs(
                    scratch_pad="We need to add try-catch blocks to handle potential payment processing errors.",
                    file_path="payment/processor.py",
                    instructions="Add error handling to catch and handle payment processing exceptions",
                    pseudo_code="""try:
    result = self._process_transaction(payment_data)
    return result
except PaymentError as e:
    logger.error(f"Payment processing failed: {e}")
    raise PaymentProcessingError(f"Failed to process payment: {e}")""",
                    change_type=ChangeType.modification,
                    start_line=45,
                    end_line=47,
                ),
            ),
            FewShotExample.create(
                user_input="Add import for the logging module",
                action=RequestCodeChangeArgs(
                    scratch_pad="We need to add the logging import at the top of the file.",
                    file_path="utils/helper.py",
                    instructions="Add import for the logging module",
                    pseudo_code="import logging",
                    change_type=ChangeType.addition,
                    start_line=1,
                    end_line=1,
                ),
            ),
        ]
