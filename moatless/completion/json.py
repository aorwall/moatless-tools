import json
import logging
from textwrap import dedent
from typing import List, Any, Type, Optional

from pydantic import ValidationError

from moatless.completion import BaseCompletionModel
from moatless.completion.base import CompletionRetryError
from moatless.completion.schema import ChatCompletionUserMessage, ResponseSchema

logger = logging.getLogger(__name__)


class JsonCompletionModel(BaseCompletionModel):
    """JSON-specific implementation of the completion model.

    This class handles:
    1. Converting multiple response schemas into a single JSON schema
    2. Configuring the LLM to output valid JSON
    3. Validating and parsing JSON responses
    """

    def _prepare_system_prompt(self, system_prompt: str, response_schema: List[Type[ResponseSchema]]) -> str:
        """Add JSON schema instructions to system prompt.

        This method appends the JSON schema and format instructions
        to the base system prompt.

        Args:
            system_prompt: Base system prompt
            prepared_schema: The prepared ResponseSchema

        Returns:
            System prompt with JSON format instructions
        """

        if len(response_schema) > 1:
            raise ValueError("JSON Completion Model only handles one response schema")

        system_prompt += dedent(f"""\n# Response format
You must respond with only a JSON object that match the following json_schema:\n

{json.dumps(response_schema[0].model_json_schema(), indent=2, ensure_ascii=False)}

Make sure to return an instance of the JSON, not the schema itself.""")
        return system_prompt

    def _validate_completion(self, completion_response: Any) -> tuple[List[ResponseSchema], Optional[str], List[str]]:
        """Validate and parse JSON completion response.

        This method:
        1. Extracts JSON content from the response
        2. Validates JSON syntax
        3. Validates against the response schema
        4. Handles both single and multi-action responses

        Args:
            completion_response: Raw response from the LLM

        Returns:
            Tuple of:
            - List of validated ResponseSchema instances
            - Optional text response string
            - List of flags indicating any special conditions

        Raises:
            CompletionRejectError: For invalid JSON that should be retried
        """

        if len(self.response_schema) > 1:
            raise ValueError("JSON Completion Model only handles one response schema")

        try:
            assistant_message = completion_response.choices[0].message.content

            response = self.response_schema[0].model_validate_json(assistant_message)
            return [response], None, []

        except (ValidationError, json.JSONDecodeError) as e:
            logger.warning(f"JSON validation failed with error: {e}")
            retry_message = ChatCompletionUserMessage(
                role="user", content=f"The response was invalid. Fix these errors:\n{e}"
            )
            raise CompletionRetryError(
                message=str(e),
                retry_message=retry_message,
            ) from e
