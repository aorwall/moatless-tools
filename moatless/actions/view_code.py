import logging
from typing import List, Optional

from pydantic import Field, BaseModel, PrivateAttr

from moatless.actions.action import Action
from moatless.actions.identify_mixin import IdentifyMixin
from moatless.actions.model import (
    ActionArguments,
    FewShotExample,
    Observation,
    RewardScaleEntry,
)
from moatless.codeblocks import CodeBlockType
from moatless.completion import CompletionModel
from moatless.file_context import FileContext, ContextFile
from moatless.repository.repository import Repository
from moatless.workspace import Workspace

logger = logging.getLogger(__name__)


class CodeSpan(BaseModel):
    file_path: str = Field(
        description="The file path where the relevant code is found."
    )
    start_line: Optional[int] = Field(
        None, description="The start line of the code to add to context."
    )
    end_line: Optional[int] = Field(
        None, description="The end line of the code to add to context."
    )
    span_ids: list[str] = Field(
        default_factory=list,
        description="Span IDs identiying the relevant code spans. A span id is a unique identifier for a code sippet. It can be a class name or function name. For functions in classes separete with a dot like 'class.function'.",
    )

    @property
    def log_name(self):
        log = self.file_path

        if self.start_line and self.end_line:
            log += f" {self.start_line}-{self.end_line}"

        if self.span_ids:
            log += f" {', '.join(self.span_ids)}"

        return log


class ViewCodeArgs(ActionArguments):
    """View the code in a file or a specific code span."""

    thoughts: str = Field(..., description="Your thoughts on the code change.")
    files: List[CodeSpan] = Field(
        ..., description="The code that should be provided in the file context."
    )

    class Config:
        title = "ViewCode"

    @property
    def log_name(self):
        if len(self.files) == 1:
            return f"ViewCode({self.files[0].log_name})"
        else:
            logs = []
            for i, file in enumerate(self.files):
                logs.append(f"{i}=[{file.log_name}]")
            return f"ViewCode(" + ", ".join(logs) + ")"

    def to_prompt(self):
        prompt = "Show the following code:\n"
        for file in self.files:
            prompt += f"* {file.file_path}\n"
            if file.start_line and file.end_line:
                prompt += f"  Lines: {file.start_line}-{file.end_line}\n"
            if file.span_ids:
                prompt += f"  Spans: {', '.join(file.span_ids)}\n"
        return prompt

    def short_summary(self) -> str:
        param_strs = []
        for file in self.files:
            param_strs.append(f"path={file.file_path}")
        param_str = ", ".join(param_strs)
        return f"{self.name}({param_str})"


class ViewCode(Action, IdentifyMixin):
    args_schema = ViewCodeArgs

    _repository: Repository = PrivateAttr()

    def __init__(
        self,
        repository: Repository = None,
        completion_model: CompletionModel | None = None,
        **data,
    ):
        super().__init__(completion_model=completion_model, **data)
        self._repository = repository

    max_tokens: int = Field(
        3000,
        description="The maximum number of tokens in the requested code.",
    )

    def execute(
        self,
        args: ViewCodeArgs,
        file_context: FileContext | None = None,
        workspace: Workspace | None = None,
    ) -> Observation:
        if file_context is None:
            raise ValueError(
                "File context must be provided to execute the view action."
            )

        properties = {"files": {}}

        # Group files by filepath and combine span_ids
        grouped_files = {}
        for file_with_spans in args.files:
            if file_with_spans.file_path not in grouped_files:
                grouped_files[file_with_spans.file_path] = file_with_spans
            else:
                grouped_files[file_with_spans.file_path].span_ids.extend(
                    file_with_spans.span_ids
                )

        # Validate all file spans before processing
        for file_path, file_span in grouped_files.items():
            logger.info(
                f"Processing file {file_path} with span_ids {file_span.span_ids}"
            )
            file = file_context.get_file(file_path)

            if not file:
                message = f"The requested file {file_path} is not found in the file repository. Use the search functions to search for the code if you are unsure of the file path."
                properties["fail_reason"] = "file_not_found"
                return Observation(
                    message=message, properties=properties, expect_correction=False
                )

            if self._repository.is_directory(file_path):
                message = f"The requested file {file_path} is a directory and not a file. Use the search functions to search for the code if you are unsure of the file path."
                properties["fail_reason"] = "is_directory"
                return Observation(
                    message=message, properties=properties, expect_correction=False
                )

        view_context = FileContext(repo=self._repository)
        completion = None

        for file_path, file_span in grouped_files.items():
            file = file_context.get_file(file_path)

            if file_span.span_ids:
                missing_span_ids = set()
                found_span_ids = set()
                if file_span.span_ids and not file.module:
                    logger.warning(
                        f"Tried to add span ids {file_span.span_ids} to not parsed file {file.file_path}."
                    )
                    message = self.create_retry_message(
                        file, f"No span ids found. Is it empty?"
                    )
                    properties["fail_reason"] = "invalid_file"
                    return Observation(
                        message=message,
                        properties=properties,
                        expect_correction=True,
                    )

                for span_id in file_span.span_ids:
                    block_span = file.module.find_span_by_id(span_id)
                    if not block_span:
                        # Try to find the relevant code block by code block identifier
                        block_identifier = span_id.split(".")[-1]
                        blocks = file.module.find_blocks_with_identifier(
                            block_identifier
                        )

                        if not blocks:
                            missing_span_ids.add(span_id)

                        for block in blocks:
                            view_context.add_span_to_context(
                                file_path,
                                block.belongs_to_span.span_id,
                                add_extra=False,
                            )

                    elif block_span.initiating_block.type == CodeBlockType.CLASS:
                        class_block = block_span.initiating_block
                        found_span_ids.add(block_span.span_id)
                        view_context.add_spans_to_context(
                            file_path, class_block.get_all_span_ids()
                        )
                    else:
                        view_context.add_span_to_context(
                            file_path, block_span.span_id, add_extra=False
                        )

            if file_span.start_line:
                view_context.add_line_span_to_context(
                    file_path, file_span.start_line, file_span.end_line, add_extra=False
                )

            if not file_span.start_line and not file_span.span_ids:
                view_context.add_file(file_path, show_all_spans=True)

            if file.patch:
                view_file = view_context.get_file(file_path)
                if view_file:
                    view_file.set_patch(file.patch)

            if view_context.context_size() > self.max_tokens:
                view_context, completion = self._identify_code(
                    args, view_context, self.max_tokens
                )

            new_span_ids = file_context.add_file_context(view_context)
            properties["files"][file_path] = {
                "new_span_ids": list(new_span_ids),
            }

        added_new_spans = any(
            len(file["new_span_ids"]) > 0 for file in properties["files"].values()
        )

        if view_context.is_empty():
            message = f"\nThe specified code spans wasn't found."
            properties["fail_reason"] = "no_spans_found"
            summary = "The specified code spans wasn't found."
        else:
            message = "Here's the contents of the file where the not requested code spans have been commented out:\n"
            message += view_context.create_prompt(
                show_span_ids=False,
                show_line_numbers=True,
                exclude_comments=False,
                show_outcommented_code=True,
                outcomment_code_comment="Rest of the code...",
            )

            if added_new_spans:
                summary = (
                    f"Showed the following code spans:\n"
                    + view_context.create_summary()
                )
            else:
                summary = "The specified code spans has already been viewed in a previous action."

        if not added_new_spans:
            properties["flags"] = ["no_spans_added"]

        return Observation(
            message=message,
            summary=summary,
            properties=properties,
            execution_completion=completion,
        )

    def _search_for_alternative_suggestion(self, args: ViewCodeArgs):
        return None

    def _select_span_instructions(self, search_result) -> str:
        return "The requested code is too large. You must identify the most relevant code sections to view."

    @classmethod
    def get_few_shot_examples(cls) -> List[FewShotExample]:
        return [
            FewShotExample.create(
                user_input="Show me the implementation of the authenticate method in the AuthenticationService class",
                action=ViewCodeArgs(
                    thoughts="To understand the authentication implementation, we need to examine the authenticate method within the AuthenticationService class.",
                    files=[
                        CodeSpan(
                            file_path="auth/service.py",
                            span_ids=["AuthenticationService.authenticate"],
                        )
                    ],
                ),
            ),
            FewShotExample.create(
                user_input="Show me lines 50-75 of the database configuration file",
                action=ViewCodeArgs(
                    thoughts="To examine the database configuration settings, we'll look at the specified line range in the config file.",
                    files=[
                        CodeSpan(
                            file_path="config/database.py", start_line=50, end_line=75
                        )
                    ],
                ),
            ),
        ]

    def create_retry_message(self, file: ContextFile, message: str):
        retry_message = f"\n\nProblems when trying to find spans in {file.file_path}. "
        retry_message += message

        hint = self.create_hint(file)
        if hint:
            retry_message += f"\n\n{hint}"

        if file.module and file.span_ids:
            search_result_context = FileContext(repo=self._repository)
            search_result_context.add_file(file.file_path, show_all_spans=True)

            search_result_str = search_result_context.create_prompt(
                show_span_ids=False,
                show_line_numbers=False,
                exclude_comments=False,
                show_outcommented_code=True,
                outcomment_code_comment="...",
                only_signatures=True,
            )
            retry_message += f"\n\nHere's the code structure:\n{search_result_str}"

        return retry_message

    def create_hint(self, file: ContextFile):
        if "test" in file.file_path:
            return "If you want to write a new test method, start by adding one of the existing ones that might relevant for reference."

        return None

    def span_id_list(self, span_ids: set[str]) -> str:
        list_str = ""
        for span_id in span_ids:
            list_str += f" * {span_id}\n"
        return list_str

    @classmethod
    def get_evaluation_criteria(cls, trajectory_length) -> List[str]:
        criteria = [
            "Relevance of Requested Context: Ensure that the requested context is directly related to the problem and necessary for making progress.",
            "Avoiding Hallucinations: Verify that the agent is requesting context for code that actually exists in the codebase.",
            "Efficiency: Assess whether the agent is requesting an appropriate amount of context without overloading unnecessary information.",
            "Appropriateness of Action: Evaluate if requesting more context is logical at this point in the problem-solving process.",
        ]
        return criteria

    @classmethod
    def get_reward_scale(cls, trajectory_length) -> List[RewardScaleEntry]:
        return [
            RewardScaleEntry(
                min_value=75,
                max_value=100,
                description="The requested context is highly relevant, precise, and necessary for solving the problem; the agent avoids hallucinations.",
            ),
            RewardScaleEntry(
                min_value=50,
                max_value=74,
                description="The requested context is relevant and helpful, with minor issues in specificity or relevance.",
            ),
            RewardScaleEntry(
                min_value=25,
                max_value=49,
                description="The requested context is somewhat relevant but may include unnecessary information or lacks specificity.",
            ),
            RewardScaleEntry(
                min_value=0,
                max_value=24,
                description="The requested context has minimal relevance or includes excessive unnecessary information.",
            ),
            RewardScaleEntry(
                min_value=-49,
                max_value=-1,
                description="The requested context is irrelevant, demonstrates misunderstanding, or the agent is hallucinating code that doesn't exist.",
            ),
        ]
