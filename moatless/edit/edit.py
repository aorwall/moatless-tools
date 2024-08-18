import logging
from typing import Optional

from pydantic import BaseModel, Field, PrivateAttr

from moatless.codeblocks import get_parser_by_path, CodeBlockType
from moatless.codeblocks.codeblocks import CodeBlockTypeGroup
from moatless.file_context import ContextFile
from moatless.repository.file import remove_duplicate_lines, do_diff, CodeFile
from moatless.state import AgenticState, ActionRequest, StateOutcome, Content, AssistantMessage, Message, UserMessage
from moatless.schema import (
    VerificationError,
)

logger = logging.getLogger(__name__)

ROLE_PROMPT = "You are autonomous AI assisistant with superior programming skills."

MAIN_OBJECTIVE_PROMPT = "The main objective is to solve a bigger task specified by the user, this is wrapped in a <main_objective> tag."

SEARCH_REPLACE_PROMPT = """Your task is to solve a smaller task within the main objective. This task is wrapped in a <task> tag.

The surrounding code context is wrapped in a <file_context> tag.

The code to that should be modified is wrapped in a <search> tag, like this:
<search>
{{CODE}}
</search>

Your task is to update the code inside the <search> tags based on the current task.

When updating the code, please adhere to the following important rules:
- Fully implement the requested change, but do not make any other changes that were not directly asked for
- Do not add any comments describing your changes 
- Indentation and formatting should be the same in the replace code as in the search code
- Ensure the modified code is complete - do not leave any TODOs, placeholder, or missing pieces
- Keep any existing placeholder comments in the <search> block (e.g. # ... other code) - do not remove or implement them

After updating the code, please format your response like this:

<replace>
put the updated code here
</replace>

ONLY return the code that was inside the original <search> tags, but with the requested modifications made. 
Do not include any of the surrounding code.

If all code in the search tag should be removed you can return an empty <replace> tag like this:
<replace>
</replace>

If you can't do any changes and want to reject the instructions return the rejection reason wrapped in a <reject> tag, like this:
<reject>
{{REASON}}
</reject>

Here is an example of what the user's request might look like:

<search>
from flask import Flask 
</search>

And here is how you should format your response:

<replace>
import math
from flask import Flask
</replace>

Remember, only put the updated version of the code from inside the <search> tags in your response, wrapped in <replace>
tags. DO NOT include any other surrounding code than the code in the <search> tag! DO NOT leave out any code that was inside the <search> tag!
"""


CHAIN_OF_THOUGHT_PROMPT = "Please provide your thoughts on the code change, if any, in the tag <scratch_pad>, and then the code change itself."


class CodeChange(ActionRequest):
    scratch_pad: Optional[str] = Field(
        default=None, description="The thoughts on the code change."
    )
    replace: str = Field(..., description="The code to replace the existing code with.")
    rejected: bool = Field(..., description="Whether the code change was rejected.")


class EditCode(AgenticState):
    instructions: str = Field(..., description="The instructions for the code change.")
    file_path: str = Field(..., description="The path to the file to be updated.")
    span_id: Optional[str] = Field(
        None, description="The ID of the span to be updated."
    )
    start_line: int = Field(
        ..., description="The start line of the code to be updated."
    )
    end_line: int = Field(..., description="The end line of the code to be updated.")

    show_initial_message: bool = Field(
        True, description="Whether to show the initial message."
    )
    show_file_context: bool = Field(
        True, description="Whether to show the file context."
    )
    verify: bool = Field(True, description="Whether to verify the code change.")
    chain_of_thought: bool = Field(
        False, description="Whether to use chain of thought reasoning."
    )

    max_prompt_file_tokens: int = Field(
        4000,
        description="The maximum number of tokens in the file context to show in the prompt.",
    )

    _code_to_replace: Optional[str] = PrivateAttr(default=None)
    _retry: int = PrivateAttr(default=0)
    _messages: list[Message] = PrivateAttr(default_factory=list)

    def init(self):
        file = self.file_context.get_file(self.file_path)
        if not file:
            raise ValueError(f"File not found: {self.file_path}")

        code_lines = file.file.content.split("\n")
        lines_to_replace = code_lines[self.start_line - 1 : self.end_line]
        self._code_to_replace = "\n".join(lines_to_replace)

    def _execute_action(self, content: Content) -> StateOutcome:
        self._messages.append(AssistantMessage(content=content.content))

        scratch_pad = None

        if "<scratch_pad>" in content.content:
            scratch_pad = content.content.split("<scratch_pad>")[1].split(
                "</scratch_pad>"
            )[0]

        if "<reject>" in content.content:
            rejection_message = content.content.split("<reject>")[1].split("</reject>")[
                0
            ]
            return StateOutcome.transition(
                "reject",
                output={"message": rejection_message},
            )

        msg_split = content.content.split("<replace>")
        if len(msg_split) == 1:
            if not self._add_prepared_response:
                logger.warning(
                    f"No <replace> tag found in response without prepped tag: {msg_split[0]}"
                )
                return StateOutcome.retry(
                    "You did not provide any code in the replace tag. If you want to reject the instructions, use the reject function."
                )

            replacement_code = msg_split[0]
        else:
            if msg_split[0] and not scratch_pad:
                scratch_pad = msg_split[0]

            if (
                "</replace" in msg_split[1]
            ):  # Skip last > to support deepseek responses where it's missing
                replacement_code = msg_split[1].split("</replace")[0]
            else:
                replacement_code = msg_split[1]

        file = self.file_context.get_file(self.file_path)

        updated_content = update_content_by_line_numbers(
            file, self.start_line - 1, self.end_line, replacement_code
        )

        diff = do_diff(file.file_path, file.content, updated_content)
        if not diff:
            return self.retry(
                "The code in the replace tag is the same as in the search. Use the reject function if you "
                "can't do any changes and want to reject the instructions.",
                scratch_pad,
            )

        updated_file = CodeFile(file_path=self.file_path, content=updated_content)

        invalid_update_str = self.verify_change(file, replacement_code, updated_file)
        if invalid_update_str:
            logger.warning(
                f"Invalid update in {self.file_path}: {invalid_update_str}.\nDiff:\n{diff}"
            )
            return self.retry(invalid_update_str, scratch_pad)

        file.file.update_content(updated_content)
        self.file_repo.save_file(file_path=file.file_path, updated_content=file.content)

        logger.info(f"Updated file {self.file_path} with diff:\n{diff}")

        message = f"Applied the change to {self.file_path}."

        if scratch_pad:
            message += f"\n\n<scratch_pad>\n{scratch_pad}</scratch_pad>"

        original_verification_errors = []
        if self.verify:
            logger.info(f"Verifying original code in {self.file_path}.")
            original_verification_errors = self.workspace.verify(file.file)

        self.file_repo.save_file(file_path=file.file_path)

        verification_errors = []
        if self.verify:
            logger.info(f"Verifying updated code in {self.file_path}.")
            verification_errors_in_update = self.workspace.verify(file.file)

            if len(verification_errors_in_update) > len(original_verification_errors):
                logger.info(
                    f"Found {len(verification_errors_in_update)} verification errors in updated code. Which differs from the original {len(original_verification_errors)}."
                )

                for error in verification_errors_in_update:
                    logger.info(f"Verification error: {error.code}, {error.message}")
            else:
                logger.info(
                    f"Found {len(verification_errors_in_update)} verification errors in updated code."
                )

            original_error_set = set(
                (msg.code, msg.message) for msg in original_verification_errors
            )

            updated_error_set = set(
                (msg.code, msg.message) for msg in verification_errors_in_update
            )
            added_messages_set = updated_error_set - original_error_set

            verification_errors = [
                VerificationError(
                    code=msg.code,
                    file_path=file.file_path,
                    message=msg.message,
                    line=msg.line,
                )
                for msg in verification_errors_in_update
                if (msg.code, msg.message) in added_messages_set
            ]

            for error in verification_errors:
                logger.info(f"New verification error: {error.code}, {error.message}")

        return StateOutcome.transition(
            "finish",
            output={
                "message": message,
                "diff": diff,
                "verification_errors": verification_errors,
            },
        )

    def retry(self, message: str, scratch_pad: str | None) -> StateOutcome:
        if self._retry > 2:
            logger.warning(f"Failed after {self._retry} retries. Will reject change.")
            message = ""
            if scratch_pad:
                message += f"<scratch_pad>\n{scratch_pad}</scratch_pad>\n\n"
            message = "Failed to apply changes. Please try again."
            return StateOutcome.transition("reject", output={"message": message})

        self._retry += 1
        return StateOutcome.retry(message)

    def verify_change(
        self, file: ContextFile, replacement_code: str, updated_file: CodeFile
    ) -> str | None:
        if not updated_file.module:
            return None

        module = updated_file.module
        error_blocks = module.find_errors()
        validation_errors = module.find_validation_errors()
        existing_placeholders = file.module.find_blocks_with_type(
            CodeBlockType.COMMENTED_OUT_CODE
        )

        new_placeholders = (
            module.find_blocks_with_type(CodeBlockType.COMMENTED_OUT_CODE)
            if not existing_placeholders
            else []
        )

        if self.span_id:
            existing_span = file.module.find_span_by_id(self.span_id)
            is_full_block = (
                existing_span.initiating_block.start_line == self.start_line
                and existing_span.initiating_block.end_line == self.end_line
            )

            # Check if the intended change was for a whole code block
            if (
                is_full_block
                and existing_span.initiating_block.type.group
                == CodeBlockTypeGroup.STRUCTURE
            ):
                # Empty updated_content would indicate a removal of a code block, but otherwise we don't except it to be replaced by another type of block
                new_block = module.find_first_by_start_line(self.start_line)

                if (
                    replacement_code
                    and existing_span.initiating_block.type != new_block.type
                ):
                    logger.warning(
                        f"Full block change: {existing_span.initiating_block.type.value} -> {new_block.type.value}"
                    )
                    return (
                        f"The code block {self.span_id} in the <search> tag with the type {existing_span.initiating_block.type.value} was expected to be replaced. But the code provided in the <replace> tag has the type  {new_block.type.value}."
                        f"You must provide the full contents of {self.span_id}. "
                        f"If you shouldn't do any changes to {self.span_id}, reject the request and explain why."
                    )

        if error_blocks or validation_errors or new_placeholders:
            error_response = ""
            if error_blocks:
                for error_block in error_blocks:
                    parent_block = error_block.find_type_group_in_parents(
                        CodeBlockTypeGroup.STRUCTURE
                    )
                    if parent_block and parent_block.type != CodeBlockType.MODULE:
                        logger.info(f"Invalid code {parent_block.to_tree()}")
                        error_response += f"{parent_block.type.name} has invalid code:\n\n```{parent_block.to_string()}\n```.\n"
                    else:
                        error_response += f"This code is invalid: \n```{error_block.to_string()}\n```.\n"

            if new_placeholders:
                for new_placeholder in new_placeholders:
                    parent_block = new_placeholder.find_type_group_in_parents(
                        CodeBlockTypeGroup.STRUCTURE
                    )
                    if parent_block:
                        error_response += f"{parent_block.identifier} has a placeholder `{new_placeholder.content}` indicating that it's not fully implemented. Implement the full {parent_block.type.name} or reject the request.: \n\n```{parent_block.to_string()}```\n\n"
                    else:
                        error_response += f"There is a placeholder indicating out commented code : \n```{new_placeholder.to_string()}\n```. Do the full implementation or reject the request.\n"

            for validation_error in validation_errors:
                error_response += f"{validation_error}\n"

            logger.warning(
                f"Errors in updated file {file.file_path}:\n{error_response}"
            )

            return error_response

        new_span_ids = module.get_all_span_ids() - set(file.module.get_all_span_ids())

        logger.info(
            f"Updated content for {file.file_path} with {len(new_span_ids)} new span ids."
        )

    @classmethod
    def required_fields(cls) -> set[str]:
        return {"instructions", "file_path", "span_id", "start_line", "end_line"}

    def system_prompt(self) -> str:
        system_prompt = ROLE_PROMPT

        if self.show_initial_message:
            system_prompt += "\n\n"
            system_prompt += MAIN_OBJECTIVE_PROMPT

        system_prompt += "\n\n"
        system_prompt += SEARCH_REPLACE_PROMPT

        if self.chain_of_thought:
            system_prompt += "\n\n"
            system_prompt += CHAIN_OF_THOUGHT_PROMPT

        return system_prompt

    def messages(self) -> list[Message]:
        if not self._code_to_replace:
            self.init()

        content = ""
        if self.show_initial_message:
            content = f"<main_objective>\n{self.initial_message}\n</main_objective>\n\n"

        content += f"<instructions>\n{self.instructions}\n</instructions>\n"

        if self.show_file_context:
            file_context_str = self.file_context.create_prompt(
                show_line_numbers=False,
                show_span_ids=False,
                exclude_comments=False,
                show_outcommented_code=True,
                outcomment_code_comment="... other code",
            )
        else:
            file_context = self.create_file_context(max_tokens=self.max_prompt_file_tokens)
            file_context.add_span_to_context(self.file_path, self.span_id)
            file_context.expand_context_with_init_spans()
            file_context.expand_context_with_related_spans(self.max_prompt_file_tokens)
            file_context.expand_classes(self.max_prompt_file_tokens)
            file_context_str = file_context.create_prompt(
                show_line_numbers=False,
                show_span_ids=False,
                exclude_comments=False,
                show_outcommented_code=True,
                outcomment_code_comment="... other code",
            )
        content += f"<file_context>\n{file_context_str}\n</file_context>\n"

        content += f"<search>\n{self._code_to_replace}\n</search>"

        messages = [UserMessage(content=content)]

        messages.extend(self.retry_messages())

        if self._add_prepared_response:
            messages.append(AssistantMessage(content="<replace>"))

        return messages

    @property
    def _add_prepared_response(self):
        return False  # FIXME? "claude" in self.model and not self.chain_of_thought

    def action_type(self) -> type[BaseModel] | None:
        return None

    def stop_words(self):
        return ["</replace>"]


def update_content_by_line_numbers(
    file: ContextFile,
    start_line_index: int,
    end_line_index: int,
    replacement_content: str,
) -> str:
    replacement_lines = replacement_content.split("\n")

    # Strip empty lines from the start and end
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
