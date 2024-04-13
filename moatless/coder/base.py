import json
import logging
import os
import uuid
from typing import List, Optional

from litellm import completion
from llama_index.core import get_tokenizer
from pydantic import BaseModel

from moatless.analytics import send_event
from moatless.codeblocks import CodeBlock
from moatless.codeblocks.print_block import (
    print_by_block_path,
    SpanMarker,
    print_by_spans,
)
from moatless.codeblocks.parser.python import PythonParser
from moatless.coder.code_utils import extract_response_parts, do_diff, CodePart
from moatless.coder.prompt import (
    CREATE_PLAN_SYSTEM_PROMPT,
    JSON_SCHEMA,
    CODER_SYSTEM_PROMPT,
)
from moatless.coder.write_code import write_code
from moatless.types import Usage, BaseResponse, ContextFile, CodingTask, Span

logger = logging.getLogger(__name__)


class WriteCodeResult(BaseModel):
    content: Optional[str] = None
    error: Optional[str] = None
    change: Optional[str] = None


class CodeRequest(BaseModel):
    request: str
    files: List[ContextFile]


class CoderResponse(BaseResponse):
    file_path: str
    diff: Optional[str] = None
    error: Optional[str] = None
    change: Optional[str] = None


class Coder:

    def __init__(
        self,
        repo_path: str,
        requirement: str,
        files: List[ContextFile],
        tags: List[str] = None,
        trace_id: str = None,
        session_id: str = None,
        one_file_mode: bool = True,
        model_name: str = "gpt-4-turbo-2024-04-09",
    ):

        if one_file_mode and len(files) > 1:
            raise ValueError(
                "One file mode is enabled, but multiple files are provided."
            )

        self._model_name = model_name
        self._repo_path = repo_path
        self._parser = PythonParser(apply_gpt_tweaks=True)
        self._one_file_mode = one_file_mode
        self._tokenizer = get_tokenizer()

        self._min_tokens_for_planning = 0

        self._trace_id = trace_id or uuid.uuid4().hex
        self._session_id = session_id or uuid.uuid4().hex
        self._tags = tags or []

        self._file_context = files
        self._requirement = requirement

        self._codeblocks = {}

        self._tasks = []

        for file in files:
            full_file_path = os.path.join(self._repo_path, file.file_path)
            with open(full_file_path, "r") as f:
                content = f.read()

            codeblock = self._parser.parse(content)
            self._codeblocks[file.file_path] = codeblock

    def run(self):
        result = self.plan()
        if not result:
            logger.warning("Failed to plan.")
            return

        for task in self._tasks:
            if task.state == "planned":
                coding_result = self.code(task)
                # TODO: Handle result and iterate

        logger.info("All tasks completed.")

    def plan(self):
        file_context_content = self._file_context_content()
        tokens = len(self._tokenizer(file_context_content))

        if tokens < self._min_tokens_for_planning:
            self._tasks.append(
                CodingTask(
                    file_path=file.file_path,
                    instructions=self._requirement,
                    state="planned",
                    action="update",
                )
                for file in self._file_context
            )
            return True

        system_prompt = f"""# FILES:
{file_context_content}

========================

INSTRUCTIONS:
{CREATE_PLAN_SYSTEM_PROMPT}

Respond ONLY in JSON that follows the schema below:"
{JSON_SCHEMA}
"""

        system_message = {"content": system_prompt, "role": "system"}
        instruction_message = {
            "content": f"# Requirement\n{self._requirement}",
            "role": "user",
        }

        messages = [
            system_message,
            instruction_message,
        ]

        retry = 0
        while retry < 2:
            response = completion(
                model=self._model_name,
                max_tokens=750,
                temperature=0.0,
                response_format={"type": "json_object"},
                # tools=tools,
                metadata={
                    "generation_name": "coder-plan",
                    "session_id": self._session_id,
                    "trace_id": self._trace_id,
                    "tags": self._tags,
                },
                messages=messages,
            )

            if response.choices[0].message.content:
                try:
                    devplan_json = json.loads(response.choices[0].message.content)
                    # TODO: Verify that blocks exists
                    # TODO: Handle rejection of request

                    if "thoughts" in devplan_json:
                        logger.info(f"Thoughts: {devplan_json['thoughts']}")

                    for task_json in devplan_json["tasks"]:
                        send_event(
                            session_id=self._session_id,
                            event="planned_tasks",
                            properties={
                                "num_of_tasks": len(devplan_json["tasks"]),
                                "tags": self._tags,
                            },
                        )

                        task = CodingTask.parse_obj(task_json)
                        logger.info(f"Task: {task.file_path} - {task.instructions}")
                        self._tasks.append(task)
                        return True

                except Exception as e:
                    logger.warning(
                        f"Error parsing message {response.choices[0].message.content}. Error {type(e)}: {e} "
                    )
                    correction = f"Error parsing message: {e}. Please respond with JSON that follows the schema below:\n{JSON_SCHEMA}"
                    send_event(
                        session_id=self._session_id,
                        event="task_planning_failed",
                        properties={"tags": self._tags},
                    )

                    messages.append(
                        {
                            "content": response.choices[0].message.content,
                            "role": "assistant",
                        }
                    )
                    messages.append({"content": correction, "role": "user"})
                    retry += 1
                    logger.warning(f"No tasks found in response. Retrying {retry}/3...")
            else:
                logger.warning("No message found in response.")
                send_event(
                    session_id=self._session_id,
                    event="task_planning_failed",
                    properties={"error": "no_message", "tags": self._tags},
                )

                raise ValueError("No message found in response.")

        # TODO: Return error?
        return False

    def code(self, task: CodingTask):
        if task.file_path not in self._codeblocks:
            logger.error(f"File {task.file_path} not found in code blocks.")
            return False

        logger.debug(f"Write code to {task.file_path} and span id {task.span_id}")

        codeblock = self._codeblocks[task.file_path]

        if not task.span_id:
            expected_span = None
            expected_block = codeblock
            instruction_code = codeblock.to_string()
        elif task.span_id[0].isdigit():
            spans = task.span_id.split("_")
            expected_span = Span(int(spans[0]), int(spans[1]))
            expected_block = codeblock.find_block_with_span(expected_span)
            instruction_code = print_by_spans(codeblock, [expected_span])
        else:
            expected_span = None
            block_path = task.span_id.split(".")
            expected_block = codeblock.find_by_path(block_path)
            instruction_code = print_by_block_path(codeblock, block_path)

        if task.span_id and len(self._file_context) > 1:
            file_context_content = f"File context:\n\n{self._file_context_content()}"
        else:
            file_context_content = ""

        # TODO: Adjust system prompt depending on action and the type of code span

        system_prompt = f"""
{CODER_SYSTEM_PROMPT}

{file_context_content}
"""

        system_message = {"content": system_prompt, "role": "system"}

        instruction_message = {
            "role": "user",
            "content": f"""# Instructions:

{task.instructions}

# Code to be updated:

```python
{instruction_code}
```
""",
        }
        messages = [system_message, instruction_message]

        usage_stats = []
        thoughts = None

        retry = 0
        while retry < 3:
            llm_response = completion(
                model=self._model_name,
                temperature=0.0,
                max_tokens=2000,
                messages=messages,
                metadata={
                    "generation_name": "coder-write-code",
                    "session_id": self._session_id,
                    "trace_id": self._trace_id,
                    "tags": self._tags,
                },
            )

            if llm_response.usage:
                usage_stats.append(Usage.parse_obj(llm_response.usage.dict()))

            change = llm_response.choices[0].message.content

            extracted_parts = extract_response_parts(change)

            changes = [part for part in extracted_parts if isinstance(part, CodePart)]

            thoughts = [part for part in extracted_parts if isinstance(part, str)]
            thoughts = "\n".join([thought for thought in thoughts])

            if thoughts:
                logger.info(f"Thoughts: {thoughts}")

            if not changes:
                logger.info("No code changes found in response.")
                send_event(
                    session_id=self._session_id,
                    event="code_update_failed",
                    properties={"error": "no changes", "tags": self._tags},
                )

                return False

            if len(changes) > 1:
                logger.warning(
                    f"Multiple code blocks found in response, ignoring all but the first one."
                )

            try:
                updated_block = self._parser.parse(changes[0].content)
            except Exception as e:
                logger.error(f"Failed to parse block content: {e}")
                corrections = f"There was a syntax error in the code block. Please correct it and try again. Error {e}"
                send_event(
                    session_id=self._session_id,
                    event="code_update_failed",
                    properties={"error": "syntax_error", "tags": self._tags},
                )
                updated_block = None

            if updated_block:
                response = write_code(
                    codeblock,
                    updated_block,
                    expected_block=expected_block,
                    expected_span=expected_span,
                    action=task.action,
                )

                if not response.error and response.content:
                    full_file_path = os.path.join(self._repo_path, task.file_path)
                    with open(full_file_path, "r") as f:
                        original_content = f.read()

                    diff = do_diff(task.file_path, original_content, response.content)
                    if diff:
                        logger.debug(f"Writing updated content to {full_file_path}")
                        # TODO: Write directly to file??
                        with open(full_file_path, "w") as file:
                            file.write(response.content)
                    else:
                        logger.info("No changes detected.")

                    added_lines = [
                        line for line in diff.split("\n") if line.startswith("+")
                    ]
                    removed_lines = [
                        line for line in diff.split("\n") if line.startswith("-")
                    ]

                    send_event(
                        session_id=self._session_id,
                        event="updated_code",
                        properties={
                            "added_lines": added_lines,
                            "removed_lines": removed_lines,
                            "file": task.file_path,
                            "expected_block": expected_block.path_string(),
                            "expected_span": expected_span,
                            "tags": self._tags,
                        },
                    )

                    return CoderResponse(
                        thoughts=thoughts,
                        usage_stats=usage_stats,
                        file_path=task.file_path,
                        diff=diff,
                        change=response.change,
                    )
                elif response.error:
                    send_event(
                        session_id=self._session_id,
                        event="code_update_failed",
                        properties={"error": response.error, "tags": self._tags},
                    )
                    corrections = f"The code isn't correct.\n{response.error}\n"
                else:
                    send_event(
                        session_id=self._session_id,
                        event="code_update_failed",
                        properties={"error": "no changes", "tags": self._tags},
                    )
                    corrections = "No changes detected."

            change = f"```\n{changes[0].content}\n```\n"

            assistant_message = {"role": "assistant", "content": change}
            messages.append(assistant_message)
            correction_message = {"role": "user", "content": corrections}

            logger.info(
                f"Ask to the LLM to retry with the correction message: {correction_message}"
            )

            retry += 1

        return CoderResponse(
            thoughts=thoughts,
            file_path=task.file_path,
            error="Failed to update code blocks.",
        )

    def _file_context_content(self):
        file_context_content = ""
        for file in self._file_context:
            codeblock = self._codeblocks[file.file_path]
            content = print_by_spans(
                codeblock, file.spans, block_marker=SpanMarker.COMMENT
            )
            file_context_content += f"{file.file_path}\n```\n{content}\n```\n"
        return file_context_content
