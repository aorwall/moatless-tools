import json
import logging
from typing import Any, Dict, List, Optional, Type, Union

from moatless.completion.base import BaseCompletionModel, CompletionRetryError
from moatless.completion.schema import ChatCompletionToolMessage, ResponseSchema
from moatless.exceptions import CompletionRuntimeError
from pydantic import Field, ValidationError

logger = logging.getLogger(__name__)


class MultiToolCallSchema(ResponseSchema):
    """A wrapper schema for multiple tool calls.
    
    This schema wraps multiple tool calls in a single response, allowing models
    that can only return a single tool call to return multiple actions.
    """
    
    tool_calls: List[Dict[str, Any]] = Field(
        ..., 
        description="List of tool calls to execute. Each must have 'name' and 'args' fields."
    )
    
    model_config = {"title": "MultiToolCall"}


class MultiToolCallCompletionModel(BaseCompletionModel):
    """Multi-tool call-specific implementation of the completion model.
    
    This class is designed to handle multiple tool calls in a single response by:
    1. Processing a specialized schema format that wraps multiple tools
    2. Converting tool responses into a structured multi-tool format
    3. Validating and parsing multiple tool call responses
    """

    def _get_completion_params(self, schema: list[type[ResponseSchema]]) -> dict[str, Union[str, dict, list]]:
        """Get multi-tool call specific completion parameters.

        This implementation wraps the actual tools in a MultiToolCall wrapper
        that can contain multiple individual tool calls.

        Args:
            schema: List of prepared tool schemas

        Returns:
            Parameters for tool-based completion
        """
        if not schema:
            raise CompletionRuntimeError("At least one response schema must be provided when using tool calls")

        tools_schema = {
            "type": "function",
            "function": {
                "name": "MultiToolCall",
                "description": "Use this when you can run multiple tools in the same response",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tool_calls": {
                            "type": "array",
                            "description": "List of tool calls to execute",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {
                                        "type": "string",
                                        "description": "The name of the tool to call",
                                        "enum": [s.name for s in schema],
                                    },
                                    "args": {
                                        "type": "object",
                                        "description": "The arguments for the tool call",
                                    }
                                },
                                "required": ["name", "args"]
                            }
                        }
                    },
                    "required": ["tool_calls"]
                }
            }
        }
        
        tools = [s.tool_schema(thoughts_in_action=self.thoughts_in_action) for s in schema]
        tools.append(tools_schema)
        
        return {
            "tools": tools,
            "tool_choice": "auto",
        }

    def _create_retry_message(self, tool_call: Any, error: str):
        """Create a retry message for a failed tool call."""
        return ChatCompletionToolMessage(role="tool", tool_call_id=tool_call.id, content=error)

    def _prepare_system_prompt(
        self,
        system_prompt: str,
        response_schema: Union[list[type[ResponseSchema]], type[ResponseSchema]],
    ) -> str:
        """Prepare the system prompt by adding format-specific instructions.

        Enhances the system prompt with explanations about the multi-tool call format
        and details about available tools.

        Args:
            system_prompt: The base system prompt
            response_schema: The response schema to use for completion

        Returns:
            The modified system prompt with multi-tool call instructions
        """
        tool_descriptions = []
        
        if isinstance(response_schema, list):
            schemas = response_schema
        else:
            schemas = [response_schema]
            
        # Add descriptions of each tool
        for schema in schemas:
            tool_info = f"Tool: {schema.name}\nDescription: {schema.description()}\n"
            
            # Get parameters from schema
            json_schema = schema.model_json_schema()
            if "properties" in json_schema:
                tool_info += "Parameters:\n"
                for param_name, param_info in json_schema["properties"].items():
                    if param_name == "thoughts":
                        continue  # Skip thoughts parameter
                    description = param_info.get("description", "")
                    required = param_name in json_schema.get("required", [])
                    tool_info += f"  - {param_name}: {description} {'(required)' if required else '(optional)'}\n"
            
            tool_descriptions.append(tool_info)
        
        # Create the multi-tool instructions
        multi_tool_instructions = (
            "\n\nYou can execute multiple tools in a single response using the MultiToolCall format. "
            "Each tool call should include the tool name and the arguments for that tool. "
            "Format your response as a single MultiToolCall with an array of tool_calls inside it.\n\n"
            "Available tools:\n\n" + "\n".join(tool_descriptions)
        )
        
        return system_prompt + multi_tool_instructions

    async def _validate_completion(
        self,
        completion_response: Any,
    ) -> tuple[list[ResponseSchema], Optional[str], Optional[str]]:
        """Validate multi-tool call completion response.

        This method validates either:
        1. A MultiToolCall wrapper containing multiple individual tool calls
        2. Direct tool calls (similar to ToolCallCompletionModel)

        Args:
            completion_response: Raw response from the LLM

        Returns:
            Tuple of:
            - List of validated structured outputs (one per successful tool call)
            - Optional text response
            - Optional thought string (always None for tool call model)

        Raises:
            CompletionRetryError: If validation fails
        """
        message = completion_response.choices[0].message
        content = message.content or ""

        # If no tool calls, return just the content
        if not hasattr(message, "tool_calls") or not message.tool_calls:
            raise CompletionRetryError(
                message="No tool calls found in response, respond with a valid tool call.",
            )

        # Check if this is a MultiToolCall wrapper or direct tool calls
        is_wrapper = len(message.tool_calls) == 1 and message.tool_calls[0].function.name == "MultiToolCall"
        
        if is_wrapper:
            # Handle MultiToolCall wrapper
            wrapper_tool_call = message.tool_calls[0]
            
            # Parse the wrapper tool call
            try:
                wrapper_args = json.loads(wrapper_tool_call.function.arguments)
            except json.JSONDecodeError as e:
                raise CompletionRetryError(
                    message=f"Invalid JSON in MultiToolCall args: {e}",
                    retry_message=self._create_retry_message(wrapper_tool_call, f"Invalid JSON in MultiToolCall args: {e}")
                )

            if not isinstance(wrapper_args, dict) or "tool_calls" not in wrapper_args:
                raise CompletionRetryError(
                    message="MultiToolCall must contain a 'tool_calls' array.",
                    retry_message=self._create_retry_message(
                        wrapper_tool_call, "MultiToolCall must contain a 'tool_calls' array."
                    )
                )

            if not isinstance(wrapper_args["tool_calls"], list) or not wrapper_args["tool_calls"]:
                raise CompletionRetryError(
                    message="MultiToolCall 'tool_calls' must be a non-empty array.",
                    retry_message=self._create_retry_message(
                        wrapper_tool_call, "MultiToolCall 'tool_calls' must be a non-empty array."
                    )
                )

            # Process each tool call in the wrapper
            structured_outputs = []
            valid_names = [s.name for s in self._response_schema] if self._response_schema else []
            retry_messages = []
            retry = False

            for tool_call_data in wrapper_args["tool_calls"]:
                if not isinstance(tool_call_data, dict):
                    logger.warning("Tool call item is not a dictionary")
                    retry = True
                    continue

                if "name" not in tool_call_data:
                    logger.warning("Tool call is missing 'name' field")
                    retry = True
                    continue

                if "args" not in tool_call_data:
                    logger.warning(f"Tool call {tool_call_data.get('name')} is missing 'args' field")
                    retry = True
                    continue

                tool_name = tool_call_data["name"]
                args = tool_call_data["args"]

                if tool_name not in valid_names:
                    retry_message = self._create_retry_message(
                        wrapper_tool_call, 
                        f"Tool {tool_name} not found in {valid_names}"
                    )
                    retry_messages.append(retry_message)
                    retry = True
                    continue

                # Find matching schema for tool
                schema = None
                if self._response_schema:
                    for s in self._response_schema:
                        if s.name == tool_name:
                            schema = s
                            break

                if not schema:
                    retry_message = self._create_retry_message(
                        wrapper_tool_call, 
                        f"Tool {tool_name} not found."
                    )
                    retry_messages.append(retry_message)
                    retry = True
                    continue

                # Include global thoughts in each tool if available
                if "thoughts" in wrapper_args and self.thoughts_in_action:
                    if isinstance(args, dict) and "thoughts" not in args:
                        args["thoughts"] = wrapper_args["thoughts"]

                try:
                    # Convert args to string if it's a dict
                    if isinstance(args, dict):
                        args_str = json.dumps(args)
                    else:
                        args_str = str(args)
                        
                    # Parse JSON if it's a string
                    if isinstance(args_str, str):
                        try:
                            args_dict = json.loads(args_str)
                        except json.JSONDecodeError:
                            args_dict = {"value": args_str}
                    else:
                        args_dict = args
                    
                    validated = schema.model_validate(args_dict)
                    structured_outputs.append(validated)
                except ValidationError as e:
                    retry_message = self._create_retry_message(
                        wrapper_tool_call, 
                        f"Tool {tool_name} arguments are invalid: {e}"
                    )
                    retry_messages.append(retry_message)
                    retry = True
                    continue
                except Exception as e:
                    retry_message = self._create_retry_message(
                        wrapper_tool_call, 
                        f"Error processing tool {tool_name}: {str(e)}"
                    )
                    retry_messages.append(retry_message)
                    retry = True
                    continue

            # If all tool calls failed validation, retry the entire completion
            if retry and not structured_outputs:
                raise CompletionRetryError(
                    message="All tool calls in MultiToolCall failed validation.",
                    retry_messages=retry_messages,
                )
            
            # If some tool calls passed validation but others failed, we can still use the valid ones
            if retry and structured_outputs:
                logger.warning(f"Some tool calls failed validation but {len(structured_outputs)} valid ones were found.")

            # Extract thoughts if not included in individual tool calls
            thought = wrapper_args.get("thoughts") if not self.thoughts_in_action else None

            return structured_outputs, content, thought
        
        else:
            # Handle direct tool calls (similar to ToolCallCompletionModel)
            # Track seen arguments to detect duplicates
            seen_arguments = set()
            structured_outputs = []
            valid_names = [s.name for s in self._response_schema] if self._response_schema else []

            retry_messages = []
            retry = False

            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name

                if tool_name not in valid_names:
                    retry_message = self._create_retry_message(
                        tool_call, f"Tool {tool_name} not found. Available tools: {valid_names}"
                    )
                    retry_messages.append(retry_message)
                    retry = True
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
                    retry = True
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
                    retry = True
                    continue

                try:
                    validated = schema.model_validate(args)
                    structured_outputs.append(validated)
                except ValidationError as e:
                    retry_message = self._create_retry_message(tool_call, f"Tool arguments is invalid. Error: {e}")
                    retry_messages.append(retry_message)
                    retry = True
                    continue

            if retry:
                raise CompletionRetryError(
                    message="One or more tool calls could not be executed.",
                    retry_messages=retry_messages,
                )

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
        if tool_name == "MultiToolCall":
            return MultiToolCallSchema
            
        if self._response_schema:
            for r in self._response_schema:
                if r.name == tool_name:
                    return r

        raise ValueError(f"No response schema found for tool name: {tool_name}")

    def _generate_few_shot_examples(self) -> str:
        """Generate few-shot examples specifically formatted for multi-tool calls."""
        base_prompt = super()._generate_few_shot_examples()
        if not base_prompt:
            return ""

        few_shot_examples = []
        if self._response_schema:
            # Create example with multiple tool calls
            example = "User: Run a command to list files and check disk space\n\nAssistant:"
            example += "\n```json\n"
            example += json.dumps({
                "tool_calls": [
                    {
                        "name": self._response_schema[0].name,
                        "args": {
                            "command": "ls", 
                            "args": ["-la"],
                            "thoughts": "Listing files with details"
                        }
                    },
                    {
                        "name": self._response_schema[0].name if len(self._response_schema) == 1 else self._response_schema[1].name,
                        "args": {
                            "command" if len(self._response_schema) == 1 else "path": "df",
                            "args" if len(self._response_schema) == 1 else "recursive": ["-h"] if len(self._response_schema) == 1 else True,
                            "thoughts": "Checking disk space"
                        }
                    }
                ],
                "thoughts": "Need to run two commands to fulfill the request"
            }, indent=2)
            example += "\n```\n\n"
            few_shot_examples.append(example)

        return base_prompt + "\n".join(few_shot_examples) 