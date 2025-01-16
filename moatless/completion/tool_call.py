import json
import logging
from typing import List, Dict, Any, Type, Union, Optional

from litellm import ChatCompletionToolMessage
from pydantic import ValidationError

from moatless.completion import BaseCompletionModel
from moatless.completion.base import CompletionRetryError
from moatless.completion.schema import ResponseSchema

logger = logging.getLogger(__name__)


class ToolCallCompletionModel(BaseCompletionModel):
    """Tool call-specific implementation of the completion model.
    
    This class handles:
    1. Converting response schemas into OpenAI function/tool schemas
    2. Configuring the LLM to use tool calls
    3. Validating and parsing tool call responses
    4. Managing thought inclusion in tool calls
    """

    def _get_completion_params(
        self,
        schema: List[Type[ResponseSchema]]
    ) -> Dict[str, Union[str, Dict, List]]:
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
        return {
            "tools": [
                s.tool_schema(thoughts_in_action=self.thoughts_in_action)
                for s in schema
            ],
            "tool_choice": "required" if self.thoughts_in_action else "auto"
        }
    
    def _create_retry_message(self, tool_call: Any, error: str):
        return ChatCompletionToolMessage(
            role="tool",
            tool_call_id=tool_call.id,
            content=error
        )

    def _validate_completion(
        self,
        completion_response: Any,
    ) -> tuple[List[ResponseSchema], Optional[str], List[str]]:
        """Validate tool call completion response.
        
        Args:
            completion_response: Raw response from the LLM
            
        Returns:
            Tuple of:
            - List of validated structured outputs
            - Optional text response
            - List of flags
            
        Raises:
            CompletionRejectError: If validation fails
        """
        message = completion_response.choices[0].message
        content = message.content or ""

        # If no tool calls, return just the content
        if not hasattr(message, "tool_calls") or not message.tool_calls:
            return [], content, []

        # Track seen arguments to detect duplicates
        seen_arguments = set()
        flags = []
        structured_outputs = []
        valid_names = [s.name for s in self.response_schema]
        invalid_function_names = []

        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name

            if tool_name not in valid_names:
                retry_message = self._create_retry_message(tool_call, f"Tool {tool_name} not found. Available tools: {valid_names}")
                raise CompletionRetryError(
                    message=f"Tool {tool_name} not found. Available tools: {valid_names}",
                    retry_message=retry_message,
                )

            # Check for duplicate arguments
            if tool_call.function.arguments in seen_arguments:
                logger.warning(
                    f"Duplicate tool call arguments found for {tool_call.function.name}"
                )
                flags.append("duplicate_tool_call")
                continue

            seen_arguments.add(tool_call.function.arguments)

            # Parse and validate tool arguments
            try:
                args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as e:
                retry_message = self._create_retry_message(tool_call, f"Invalid JSON in tool args: {e}")
                raise CompletionRetryError(
                    message=f"Invalid JSON in tool args: {e}",
                    retry_message=retry_message,
                ) from e

            # Find matching schema for tool
            schema = None
            for s in self.response_schema:
                if s.name == tool_name:
                    schema = s
                    break

            if not schema:
                retry_message = self._create_retry_message(tool_call, f"Tool {tool_name} not found.")
                raise CompletionRetryError(
                    message=f"Tool {tool_name} not found.",
                    retry_message=retry_message,
                )

            try:
                validated = schema.model_validate(args)
                structured_outputs.append(validated)
            except ValidationError as e:
                retry_message = self._create_retry_message(tool_call, f"Tool arguments is invalid. Error: {e}")
                raise CompletionRetryError(
                    message=f"Tool arguments is invalid. Error: {e}",
                    retry_message=retry_message,
                ) from e


        return structured_outputs, content, flags

    def _get_response_model(
        self,
        tool_name: str
    ) -> Type[ResponseSchema]:
        """Get the response model for a tool name.
        
        Args:
            tool_name: Name of the tool to find schema for
        
        Returns:
            Matching ResponseSchema class or None if not found
        """
        for r in self.response_schema:
            if r.name == tool_name:
                return r
    
