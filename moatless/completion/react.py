import json
import logging
from textwrap import dedent
from typing import Any, Dict, List, Optional, Type

from pydantic import ValidationError

from moatless.actions.schema import ActionArguments
from moatless.actions.think import Think, ThinkArgs
from moatless.completion.base import CompletionRetryError
from moatless.completion.json import JsonCompletionModel
from moatless.completion.schema import ChatCompletionUserMessage, ResponseSchema
from moatless.exceptions import CompletionRuntimeError

logger = logging.getLogger(__name__)


class ReActCompletionModel(JsonCompletionModel):
    """ReAct-specific implementation of the completion model.

    This class handles:
    1. Converting response schemas into ReAct format instructions
    2. Parsing and validating ReAct format responses
    3. Managing thought inclusion in actions
    4. Validating action sequence format
    """

    def _prepare_system_prompt(self, system_prompt: str, response_schema: list[type[ResponseSchema]]) -> str:
        """Add ReAct format instructions to system prompt.

        This method appends the ReAct format instructions and available
        actions with their schemas to the base system prompt.

        Args:
            system_prompt: Base system prompt
            response_schema: List of response schemas

        Returns:
            System prompt with ReAct format instructions
        """

        # Fall back to JSON completion model if the action is not an ActionArguments
        if not self._supports_react_format():
            return super()._prepare_system_prompt(system_prompt, response_schema)

        action_input_schemas = []
        for action in response_schema:
            action_input_schemas.append(f" * {action.name} {action.format_schema_for_llm()}")

        input_schemas = "\n\n".join(action_input_schemas)

        system_prompt += dedent(f"""\n# Response format

Use the following format:
{'' if not self.disable_thoughts else '''
Thoughts: You should always think about what to do'''}
Action: The action to take followed by the input arguments based on the schema below

Use one of the following actions and provide input arguments matching the schema.
                            
{input_schemas}

Important: You can include multiple Action blocks to perform sequential actions. The first Action must be preceded by a Thought section{'' if self.disable_thoughts else ''}.
""")
        return system_prompt

    def _supports_react_format(self) -> bool:
        if not isinstance(self._response_schema, list):
            return False
        for schema in self._response_schema:
            if not issubclass(schema, ActionArguments):
                return False
        return True

    async def _validate_completion(
        self,
        completion_response: Any,
    ) -> tuple[list[ResponseSchema], Optional[str]]:
        """Validate and parse ReAct format responses.

        This method:
        1. Validates ReAct format structure
        2. Extracts thoughts and actions
        3. Parses action parameters
        4. Validates against schema

        Args:
            completion_response: Raw response from the LLM

        Returns:
            Tuple of:
            - List of validated ResponseSchema instances
            - Optional text response string

        Raises:
            CompletionRejectError: For invalid format that should be retried
        """

        # Fall back to JSON completion model if the action is not an ActionArguments
        if not self._supports_react_format():
            return await super()._validate_completion(completion_response)

        try:
            response_text = completion_response.choices[0].message.content
            self._validate_react_format(response_text)

            # Get all action blocks
            thought, action_blocks = self._extract_action_blocks(response_text)
            validated_actions = []

            for i, action_input in enumerate(action_blocks):
                action_name, action_input = self._parse_action(action_input)
                action_class = self._get_action_class(action_name)

                if action_input.strip().startswith("<") or action_input.strip().startswith("```"):
                    try:
                        action_request = action_class.model_validate_xml(action_input)
                    except Exception as e:
                        if validated_actions:
                            logger.warning(
                                f"Invalid XML format for {action_name}. Error: {e}. Will break flow and return the already validated actions"
                            )
                            break

                        format_example = (
                            action_class.format_schema_for_llm()
                            if hasattr(action_class, "format_schema_for_llm")
                            else ""
                        )
                        raise CompletionRetryError(
                            f"Invalid XML format for {action_name}. Error: {e}\n\n"
                            f"Expected format:\n{format_example}"
                        )
                else:
                    try:
                        action_request = action_class.model_validate_json(action_input)
                    except Exception as e:
                        if validated_actions:
                            logger.warning(
                                f"Invalid JSON format for {action_name}. Error: {e}. Will break flow and return the already validated actions"
                            )
                            break

                        raise CompletionRetryError(
                            f"Invalid format for {action_name}. Error: {e}\n\n"
                            f"Expected schema:\n{action_class.format_schema_for_llm()}"
                        )

                validated_actions.append(action_request)

            if thought:
                validated_actions.insert(0, ThinkArgs(thought=thought))  # type: ignore

            return validated_actions, None
        except CompletionRetryError as e:
            raise e
        except (ValueError, ValidationError) as e:
            logger.warning(f"ReAct parsing failed. Response: {response_text}")
            raise CompletionRetryError(str(e)) from e
        except Exception as e:
            logger.exception(f"ReAct parsing failed. Response: {response_text}")
            raise CompletionRuntimeError(str(e)) from e

    def _validate_react_format(self, response_text: str) -> None:
        """Validate the ReAct format structure.

        Args:
            response_text: Raw response to validate

        Raises:
            ValueError: If format is invalid
        """
        lines = [line.strip() for line in response_text.split("\n") if line.strip()]

        action_count = sum(1 for line in lines if line.lower().startswith("action:"))

        if action_count < 1:
            raise CompletionRetryError("Response must have one 'Action:' section")

        if not self.disable_thoughts:
            thought_line = next(
                (i for i, line in enumerate(lines) if line.lower().startswith(("thought:", "thoughts:"))), -1
            )
            action_line = next((i for i, line in enumerate(lines) if line.lower().startswith("action:")), -1)

            # Check if sections are in correct order
            if not (thought_line < action_line):
                raise CompletionRetryError(
                    "Your response is incorrect. The Thought section must come before the Action section. Please try the same request again with the correct format."
                )

    def _extract_action_blocks(self, response_text: str) -> tuple[Optional[str], list[str]]:
        """Extract multiple action blocks from response text.

        Args:
            response_text: Raw response text

        Returns:
            Tuple of:
            - Optional thought if it exists
            - List of action blocks
        """
        # Find all action markers in the response
        response_lower = response_text.lower()
        action_positions = []
        pos = 0

        while True:
            pos = response_lower.find("action:", pos)
            if pos == -1:
                break
            action_positions.append(pos)
            pos += 1

        if not action_positions:
            raise ValueError("Missing Action section")

        first_action_pos = action_positions[0]
        thought = ""

        thought_start = min(
            (pos for pos in (response_lower.find("thought:"), response_lower.find("thoughts:")) if pos != -1),
            default=-1,
        )

        if thought_start >= 0:
            thought_prefix_end = response_text.find(":", thought_start) + 1
            thought = response_text[thought_prefix_end:first_action_pos].strip()

        action_blocks = []

        for i, action_pos in enumerate(action_positions):
            # Extract the action text up to the next action or end of text
            if i < len(action_positions) - 1:
                next_action_pos = action_positions[i + 1]
                action_text = response_text[action_pos + 7 : next_action_pos].strip()
            else:
                action_text = response_text[action_pos + 7 :].strip()

            action_blocks.append(action_text)

        return thought, action_blocks

    def _extract_thought_action(self, response_text: str) -> tuple[str, str]:
        """Extract thought and action from response text.

        Args:
            response_text: Raw response text

        Returns:
            Tuple of (thought, action_input)

        Raises:
            ValueError: If sections can't be extracted
        """
        thought = ""
        if not self.disable_thoughts:
            # Find the first occurrence of either "thought:" or "thoughts:" case-insensitive
            response_lower = response_text.lower()
            thought_start = min(
                (pos for pos in (response_lower.find("thought:"), response_lower.find("thoughts:")) if pos != -1),
                default=-1,
            )
            action_start = response_lower.find("action:")

            if thought_start == -1 or action_start == -1:
                raise ValueError("Missing Thought or Action sections")

            # Find the actual length of the thought prefix to skip
            thought_prefix_end = response_text.find(":", thought_start) + 1
            thought = response_text[thought_prefix_end:action_start].strip()
            action_input = response_text[action_start + 7 :].strip()
        else:
            action_start = response_text.lower().find("action:")
            if action_start == -1:
                raise ValueError("Missing Action section")
            action_input = response_text[action_start + 7 :].strip()

        return thought, action_input

    def _parse_action(self, action_input: str) -> tuple[str, str]:
        """Parse action name and input from action text.

        Args:
            action_input: Raw action text

        Returns:
            Tuple of (action_name, action_parameters)

        Raises:
            ValueError: If action format is invalid
        """
        action_lines = action_input.split("\n", 1)
        if len(action_lines) < 2:
            raise ValueError("Missing action name and input")

        return action_lines[0].strip(), action_lines[1].strip()

    def _get_action_class(self, action_name: str) -> type[ResponseSchema]:
        """Get the action class for an action name.

        Args:
            action_name: Name of the action

        Returns:
            Matching ResponseSchema class

        Raises:
            ValueError: If action name is invalid
        """
        if not isinstance(self._response_schema, list):
            schemas = [self._response_schema]
        else:
            schemas = self._response_schema

        # Find the matching action class
        matching_actions = [a for a in schemas if hasattr(a, "name") and a.name == action_name]
        if not matching_actions:
            action_names = [a.name for a in schemas if hasattr(a, "name")]
            raise ValueError(f"Unknown action: {action_name}. Available actions: {', '.join(action_names)}")

        return matching_actions[0]

    def _generate_few_shot_examples(self) -> str:
        """Generate few-shot examples in ReAct format"""
        base_prompt = super()._generate_few_shot_examples()
        if not base_prompt:
            return ""

        if not self._response_schema:
            return ""

        few_shot_examples = []
        for schema in self._response_schema:
            if hasattr(schema, "get_few_shot_examples"):
                examples = schema.get_few_shot_examples()
                if examples:
                    for i, example in enumerate(examples):
                        prompt = f"\n**Example {i + 1}**"
                        action_data = example.action.model_dump()
                        thoughts = action_data.pop("thoughts", "")

                        prompt += f"\nTask: {example.user_input}\n"
                        if not self.disable_thoughts:
                            prompt += f"\nThoughts: {thoughts}\n"
                        prompt += f"Action: {example.action.name}\n"

                        # Handle special action types
                        # TODO: Move to the action implementations
                        if example.action.__class__.__name__ in [
                            "StringReplaceArgs",
                            "CreateFileArgs",
                            "AppendStringArgs",
                            "InsertLinesArgs",
                            "FindCodeSnippetArgs",
                        ]:
                            if "path" in action_data:
                                prompt += f"<path>{action_data['path']}</path>\n"
                            if "old_str" in action_data:
                                prompt += f"<old_str>\n{action_data['old_str']}\n</old_str>\n"
                            if "new_str" in action_data:
                                prompt += f"<new_str>\n{action_data['new_str']}\n</new_str>\n"
                            if "file_text" in action_data:
                                prompt += f"<file_text>\n{action_data['file_text']}\n</file_text>\n"
                            if "insert_line" in action_data:
                                prompt += f"<insert_line>{action_data['insert_line']}</insert_line>\n"
                            if "code_snippet" in action_data:
                                prompt += f"<code_snippet>{action_data['code_snippet']}</code_snippet>\n"
                        else:
                            prompt += f"{json.dumps(action_data)}\n"

                        few_shot_examples.append(prompt)

        return base_prompt + "\n".join(few_shot_examples)
