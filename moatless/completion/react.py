import logging
from textwrap import dedent
from typing import List, Dict, Any, Type, Optional
import json

from moatless.actions.schema import ActionArguments
from moatless.completion.base import CompletionRetryError
from moatless.completion.json import JsonCompletionModel
from moatless.completion.schema import ResponseSchema, ChatCompletionUserMessage

logger = logging.getLogger(__name__)


class ReActCompletionModel(JsonCompletionModel):
    """ReAct-specific implementation of the completion model.

    This class handles:
    1. Converting response schemas into ReAct format instructions
    2. Parsing and validating ReAct format responses
    3. Managing thought inclusion in actions
    4. Validating action sequence format
    """

    def _prepare_system_prompt(self, system_prompt: str, response_schema: List[Type[ResponseSchema]]) -> str:
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

Important: Do not include multiple{' Thought-' if self.disable_thoughts else ''} Action blocks. Do not include code blocks or additional text outside of this format.
""")
        return system_prompt

    def _get_completion_params(self, schema: ResponseSchema) -> Dict[str, str | Dict | List]:
        #params = super()._get_completion_params(schema)
        #params["stop"] = ["Observation:"]
        #return params
        return {}
    def _supports_react_format(self) -> bool:
        if not isinstance(self._response_schema, list):
            return False
        for schema in self._response_schema:
            from moatless.actions.schema import ActionArguments
            if not issubclass(schema, ActionArguments):
                return False
        return True

    async def _validate_completion(
        self,
        completion_response: Any,
    ) -> tuple[List[ResponseSchema], Optional[str], List[str]]:
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
            - List of flags indicating any special conditions

        Raises:
            CompletionRejectError: For invalid format that should be retried
            CompletionRuntimeError: For fundamentally invalid responses
        """

        # Fall back to JSON completion model if the action is not an ActionArguments
        if not self._supports_react_format():
            return await super()._validate_completion(completion_response)
        
        try:
            response_text = completion_response.choices[0].message.content

            self._validate_react_format(response_text)

            thought, action_input = self._extract_thought_action(response_text)
            action_name, action_input = self._parse_action(action_input)
            action_class = self._get_action_class(action_name)

            if action_input.strip().startswith("<") or action_input.strip().startswith("```"):
                try:
                    action_request = action_class.model_validate_xml(action_input)
                except Exception as e:
                    format_example = (
                        action_class.format_schema_for_llm() if hasattr(action_class, "format_schema_for_llm") else ""
                    )
                    raise ValueError(
                        f"Invalid XML format for {action_name}. Error: {e}\n\n" f"Expected format:\n{format_example}"
                    )
            else:
                try:
                    action_request = action_class.model_validate_json(action_input)
                except Exception as e:
                    raise ValueError(
                        f"Invalid format for {action_name}. Error: {e}\n\n"
                        f"Expected schema:\n{action_class.format_schema_for_llm()}"
                    )

            action_request.thoughts = thought
            return [action_request], None, []

        except Exception as e:
            logger.warning(f"ReAct parsing failed: {e}. Response: {response_text}")
            retry_message = ChatCompletionUserMessage(role="user", content=str(e))
            raise CompletionRetryError(
                message=str(e),
                retry_messages=[retry_message],
            ) from e

    def _validate_react_format(self, response_text: str) -> None:
        """Validate the ReAct format structure.

        Args:
            response_text: Raw response to validate

        Raises:
            ValueError: If format is invalid
        """
        # Split into lines and remove empty ones
        lines = [line.strip() for line in response_text.split("\n") if line.strip()]

        # Count occurrences of each section using case-insensitive matching
        thought_count = sum(1 for line in lines if line.lower().startswith(("thought:", "thoughts:")))
        action_count = sum(1 for line in lines if line.lower().startswith("action:"))

        # Check for multiple action blocks
        if thought_count > 1 or action_count > 1:
            logger.warning(f"Multiple Thought or Action sections found in response: {response_text}")

        # Check if all sections exist
        if not self.disable_thoughts and thought_count < 1:
            raise ValueError("The response is incorrect, it should start with 'Thoughts:'")
        if action_count < 1:
            raise ValueError("Response must have one 'Action:' section")

        if not self.disable_thoughts:
            # Find the starting lines for each section
            thought_line = next((i for i, line in enumerate(lines) if line.lower().startswith(("thought:", "thoughts:"))), -1)
            action_line = next((i for i, line in enumerate(lines) if line.lower().startswith("action:")), -1)

            # Check if sections are in correct order
            if not (thought_line < action_line):
                raise ValueError(
                    "Your response is incorrect. The Thought section must come before the Action section. Please try the same request again with the correct format."
                )

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
                (pos for pos in (response_lower.find("thought:"), response_lower.find("thoughts:"))
                if pos != -1),
                default=-1
            )
            action_start = response_lower.find("action:")

            if thought_start == -1 or action_start == -1:
                raise ValueError("Missing Thought or Action sections")

            # Find the actual length of the thought prefix to skip
            thought_prefix_end = response_text.find(":", thought_start) + 1
            thought = response_text[thought_prefix_end:action_start].strip()
            action_input = response_text[action_start + 7:].strip()
        else:
            action_start = response_text.lower().find("action:")
            if action_start == -1:
                raise ValueError("Missing Action section")
            action_input = response_text[action_start + 7:].strip()

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

    def _get_action_class(self, action_name: str) -> Optional[Type[ResponseSchema]]:
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

        action_class = next((a for a in schemas if a.name == action_name), None)
        if not action_class:
            action_names = [a.name for a in schemas]
            raise ValueError(f"Unknown action: {action_name}. Available actions: {', '.join(action_names)}")

        return action_class

    def _generate_few_shot_examples(self) -> str:
        """Generate few-shot examples in ReAct format"""
        base_prompt = super()._generate_few_shot_examples()
        if not base_prompt:
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
