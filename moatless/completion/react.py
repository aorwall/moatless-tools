import json
import logging
from textwrap import dedent
from typing import List

import tenacity
from litellm import APIError, BadRequestError, NotFoundError, AuthenticationError

from moatless.completion import CompletionModel
from moatless.completion.completion import CompletionResponse
from moatless.completion.model import Completion, StructuredOutput, Usage
from moatless.exceptions import CompletionRejectError

logger = logging.getLogger(__name__)


class ReActCompletionModel(CompletionModel):
    def create_completion(
        self,
        messages: List[dict],
        system_prompt: str,
        response_model: List[type[StructuredOutput]],
    ) -> CompletionResponse:
        action_input_schemas = []

        total_usage = Usage()
        retry_count = 0

        for action in response_model:
            action_input_schemas.append(
                f" * {action.name} {action.format_schema_for_llm()}"
            )

        system_prompt += dedent(f"""\n# Response format

Use the following format:

Thought: You should always think about what to do
Action: The action to take followed by the input arguments based on the schema below

Use one of the following actions and provide input arguments matching the schema.
                            
{'\n\n'.join(action_input_schemas)}

Important: Do not include multiple Thought-Action blocks. Do not include code blocks or additional text outside of this format.
""")

        messages.insert(0, {"role": "system", "content": system_prompt})

        retries = tenacity.Retrying(
            retry=tenacity.retry_if_not_exception_type(
                (APIError, BadRequestError, NotFoundError, AuthenticationError)
            ),
            stop=tenacity.stop_after_attempt(3),
        )

        def _do_completion():
            nonlocal total_usage, retry_count
            completion_response = self._litellm_base_completion(messages=messages)
            response_text = completion_response.choices[0].message.content

            total_usage += Usage.from_completion_response(
                completion_response, self.model
            )

            try:
                self._validate_react_format(response_text)

                thought_start = response_text.find("Thought:")
                action_start = response_text.find("Action:")

                if thought_start == -1 or action_start == -1:
                    raise ValueError("Missing Thought or Action sections")

                thought = response_text[thought_start + 8 : action_start].strip()
                action_input = response_text[action_start + 7 :].strip()

                # Extract action name and input
                action_lines = action_input.split("\n", 1)
                if len(action_lines) < 2:
                    raise ValueError("Missing action name and input")

                action_name = action_lines[0].strip()
                action_input = action_lines[1].strip()

                # Find the matching action class
                action_class = next(
                    (a for a in response_model if a.name == action_name), None
                )
                if not action_class:
                    action_names = [a.name for a in response_model]
                    raise ValueError(
                        f"Unknown action: {action_name}. Available actions: {', '.join(action_names)}"
                    )

                # Check if input appears to be XML format
                if action_input.strip().startswith(
                    "<"
                ) or action_input.strip().startswith("```xml"):
                    try:
                        action_request = action_class.model_validate_xml(action_input)
                    except Exception as e:
                        format_example = (
                            action_class.format_schema_for_llm()
                            if hasattr(action_class, "format_schema_for_llm")
                            else ""
                        )
                        raise ValueError(
                            f"Invalid XML format for {action_name}. Error: {e}\n\n"
                            f"Expected format:\n{format_example}"
                        )
                else:
                    # Otherwise, try to validate as JSON
                    try:
                        action_request = action_class.model_validate_json(action_input)
                    except Exception as e:
                        schema = action_class.model_json_schema()
                        if "thoughts" in schema["properties"]:
                            del schema["properties"]["thoughts"]
                        raise ValueError(
                            f"Invalid format for {action_name}. Error: {e}\n\n"
                            f"Expected JSON schema:\n{json.dumps(schema, indent=2)}"
                        )

                action_request.thoughts = thought
                completion = Completion.from_llm_completion(
                    input_messages=messages,
                    completion_response=completion_response,
                    model=self.model,
                    retries=retry_count,
                    usage=total_usage,
                )

                return CompletionResponse(
                    structured_outputs=[action_request], completion=completion
                )

            except Exception as e:
                logger.warning(f"ReAct parsing failed: {e}. Response: {response_text}")
                messages.append({"role": "assistant", "content": response_text})

                messages.append(
                    {
                        "role": "user",
                        "content": f"The response was invalid. {e}",
                    }
                )

                retry_count += 1

                raise CompletionRejectError(
                    message=str(e),
                    last_completion=completion_response,
                    messages=messages,
                ) from e

        try:
            return retries(_do_completion)
        except tenacity.RetryError as e:
            raise e.reraise()

    def _validate_react_format(self, response_text: str):
        # Split into lines and remove empty ones
        lines = [line.strip() for line in response_text.split("\n") if line.strip()]

        # Count occurrences of each section
        thought_count = sum(1 for line in lines if line.startswith("Thought:"))
        action_count = sum(1 for line in lines if line.startswith("Action:"))

        # Check for multiple action blocks
        if thought_count > 1 or action_count > 1:
            logger.warning(
                f"Multiple Thought or Action sections found in response: {response_text}"
            )

        # Check if all sections exist
        if thought_count < 1 or action_count < 1:
            raise ValueError("Response must have one 'Thought:' and 'Action:' section")

        # Find the starting lines for each section
        thought_line = next(
            (i for i, line in enumerate(lines) if line.startswith("Thought:")), -1
        )
        action_line = next(
            (i for i, line in enumerate(lines) if line.startswith("Action:")), -1
        )

        # Check if sections are in correct order
        if not (thought_line < action_line):
            raise ValueError("Sections must be in order: Thought, Action")
