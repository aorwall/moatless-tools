import json
import logging
from textwrap import dedent
from typing import Any, List, Optional, Type, Tuple

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

    @property
    def response_schema(self) -> list[type[ResponseSchema]]:
        """Get the response schema."""
        return self._response_schema if self._response_schema else []

    def _get_completion_params(self, schema: list[type[ResponseSchema]]) -> dict[str, Any]:
        """Get the completion parameters for JSON completion."""
        return {"response_format": {"type": "json_object"}}

    def _prepare_system_prompt(
        self, system_prompt: str, response_schema: list[type[ResponseSchema]] | type[ResponseSchema]
    ) -> str:
        """Add JSON schema instructions to system prompt.

        This method appends the JSON schema and format instructions
        to the base system prompt.

        Args:
            system_prompt: Base system prompt
            prepared_schema: The prepared ResponseSchema

        Returns:
            System prompt with JSON format instructions
        """
        schemas = [response_schema] if not isinstance(response_schema, list) else response_schema

        if len(schemas) > 1:
            raise ValueError("JSON Completion Model only handles one response schema")

        system_prompt += dedent(f"""\n# Response format
You must respond with only a JSON object that match the following json_schema:\n

{json.dumps(schemas[0].model_json_schema(), indent=2, ensure_ascii=False)}

Make sure to return an instance of the JSON, not the schema itself.""")
        return system_prompt

    async def _validate_completion(
        self, completion_response: Any
    ) -> tuple[list[ResponseSchema], Optional[str], Optional[str]]:
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
            - Optional thought string (always None for JSON model)

        Raises:
            CompletionRetryError: For invalid JSON that should be retried
        """

        if not self._response_schema:
            raise ValueError("No response schema provided")

        if len(self._response_schema) > 1:
            raise ValueError("JSON Completion Model only handles one response schema")

        try:
            assistant_message = completion_response.choices[0].message.content
            response = self._response_schema[0].model_validate_json(assistant_message)
            return [response], None, None

        except (ValidationError, json.JSONDecodeError) as e:
            logger.warning(f"JSON validation failed with error: {e}")
            raise CompletionRetryError(f"The response was invalid. Fix these errors:\n{e}") from e
