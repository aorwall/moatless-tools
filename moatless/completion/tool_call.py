import json
import logging
from typing import Any, Dict, List, Optional, Type, Union

from moatless.completion import BaseCompletionModel
from moatless.completion.base import CompletionRetryError
from moatless.completion.schema import ChatCompletionToolMessage, ResponseSchema
from moatless.exceptions import CompletionRuntimeError
from pydantic import ValidationError

logger = logging.getLogger(__name__)


class ToolCallCompletionModel(BaseCompletionModel):
    """Tool call-specific implementation of the completion model.

    This class handles:
    1. Converting response schemas into OpenAI function/tool schemas
    2. Configuring the LLM to use tool calls
    3. Validating and parsing tool call responses
    4. Managing thought inclusion in tool calls
    """

    def _get_completion_params(self, schema: list[type[ResponseSchema]]) -> dict[str, Union[str, dict, list]]:
        """Get tool call-specific completion parameters.

        This method:
        1. Converts schemas to OpenAI tool format
        2. Configures tool choice behavior
        3. Handles thought inclusion settings

        Args:
            schema: List of prepared tool schemas

        Returns:
            Parameters for tool-based completion including:
            - tools: List of available tools/functions
            - tool_choice: How tools should be selected
        """
        if not schema:
            raise CompletionRuntimeError("At least one response schema must be provided when using tool calls")

        return {
            "tools": [s.tool_schema(thoughts_in_action=self.thoughts_in_action) for s in schema],
            "tool_choice": "auto",
        }

    def _create_retry_message(self, tool_call: Any, error: str):
        return ChatCompletionToolMessage(role="tool", tool_call_id=tool_call.id, content=error)

    async def _validate_completion(
        self,
        completion_response: Any,
    ) -> tuple[list[ResponseSchema], Optional[str], Optional[str]]:
        """Validate tool call completion response.

        Args:
            completion_response: Raw response from the LLM

        Returns:
            Tuple of:
            - List of validated structured outputs (only valid toolcalls; invalid toolcalls are ignored)
            - Optional text response
            - Optional thought string (always None for tool call model)

        Raises:
            CompletionRetryError: If all toolcalls are invalid (invalid name, bad JSON, or validation error)
        """
        message = completion_response.choices[0].message
        content = message.content or ""

        # If no tool calls, return just the content
        if not hasattr(message, "tool_calls") or not message.tool_calls:
            if "SendMessage" in self._response_schema:
                return [], content, None
            else:
                raise CompletionRetryError(
                    message="No tool calls found in response, respond with a valid tool call.",
                )

        # Track seen arguments to detect duplicates
        seen_arguments = set()
        structured_outputs = []
        valid_names = [s.name for s in self._response_schema] if self._response_schema else []

        retry_messages = []
        num_toolcalls = len(message.tool_calls)
        num_invalid = 0

        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name

            if tool_name not in valid_names:
                retry_message = self._create_retry_message(
                    tool_call, f"Tool {tool_name} not found. Available tools: {valid_names}"
                )
                retry_messages.append(retry_message)
                num_invalid += 1
                continue

            # Check for duplicate arguments
            if tool_call.function.arguments in seen_arguments:
                logger.warning(f"Duplicate tool call arguments found for {tool_call.function.name}")
                continue

            seen_arguments.add(tool_call.function.arguments)

            # Parse and validate tool arguments
            try:
                args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as e:
                retry_message = self._create_retry_message(tool_call, f"Invalid JSON in tool args: {e}")
                retry_messages.append(retry_message)
                num_invalid += 1
                continue

            # Find matching schema for tool
            schema = None
            if self._response_schema:
                for s in self._response_schema:
                    if s.name == tool_name:
                        schema = s
                        break

            if not schema:
                retry_message = self._create_retry_message(tool_call, f"Tool {tool_name} not found.")
                retry_messages.append(retry_message)
                num_invalid += 1
                continue

            try:
                validated = schema.model_validate(args)
                structured_outputs.append(validated)
            except ValidationError as e:
                retry_message = self._create_retry_message(tool_call, f"Tool arguments is invalid. Error: {e}")
                retry_messages.append(retry_message)
                num_invalid += 1
                continue

        if num_toolcalls > 0 and num_invalid == num_toolcalls:
            # All toolcalls failed
            raise CompletionRetryError(
                message="All toolcalls failed validation or had invalid names.",
                retry_messages=retry_messages,
            )

        # If at least one valid, ignore the invalids and return only valid ones
        return structured_outputs, content, None

    def _get_response_model(self, tool_name: str) -> type[ResponseSchema]:
        """Get the response model for a tool name.

        Args:
            tool_name: Name of the tool to find schema for

        Returns:
            Matching ResponseSchema class

        Raises:
            ValueError: If no matching schema is found
        """
        if self._response_schema:
            for r in self._response_schema:
                if r.name == tool_name:
                    return r

        raise ValueError(f"No response schema found for tool name: {tool_name}")

    def _generate_few_shot_examples(self) -> str:
        """Generate few-shot examples in tool call format"""
        base_prompt = super()._generate_few_shot_examples()
        if not base_prompt:
            return ""

        few_shot_examples = []
        if self._response_schema:
            for schema in self._response_schema:
                if hasattr(schema, "get_few_shot_examples"):
                    examples = schema.get_few_shot_examples()
                    if examples:
                        for example in examples:
                            action_json = {
                                "action": example.action.model_dump(),
                                "action_type": example.action.name,
                            }
                            prompt = f"User: {example.user_input}\nAssistant:\n```json\n{json.dumps(action_json, indent=2)}\n```\n\n"
                            few_shot_examples.append(prompt)

        return base_prompt + "\n".join(few_shot_examples)
