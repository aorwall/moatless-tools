import hashlib
import json
import logging
from typing import Optional, Any, Union

import litellm
from instructor import OpenAISchema
from litellm import cost_per_token, NotFoundError
from pydantic import BaseModel, model_validator, Field, ValidationError

logger = logging.getLogger(__name__)


class Message(BaseModel):
    role: str = Field(..., description="The role of the sender")
    content: Optional[str] = Field(None, description="The message content")


class ToolCall(BaseModel):
    name: str = Field(..., description="The name of the tool being called")
    type: Optional[str] = Field(None, description="The type of tool call")
    input: Optional[dict[str, Any]] = Field(
        None, description="The input parameters for the tool"
    )


class AssistantMessage(Message):
    role: str = Field("assistant", description="The role of the assistant")
    content: Optional[str] = Field(None, description="The assistant's message content")
    tool_call: Optional[ToolCall] = Field(
        None, description="Tool call made by the assistant"
    )

    @property
    def tool_call_id(self) -> Optional[str]:
        """Generate a deterministic tool call ID based on the tool call content"""
        if not self.tool_call:
            return None

        # Create a string combining name and input for hashing
        tool_str = (
            f"{self.tool_call.name}:{json.dumps(self.tool_call.input, sort_keys=True)}"
        )
        # Generate SHA-256 hash and take first 8 characters
        hash_id = hashlib.sha256(tool_str.encode()).hexdigest()[:8]
        return f"call_{hash_id}"


class UserMessage(Message):
    role: str = Field("user", description="The role of the user")
    content: str = Field(..., description="The user's message content")


class Usage(BaseModel):
    completion_cost: float = 0
    completion_tokens: int = 0
    prompt_tokens: int = 0
    cached_tokens: int = 0

    @classmethod
    def from_completion_response(
        cls, completion_response: dict | BaseModel, model: str
    ) -> Union["Usage", None]:
        if isinstance(completion_response, BaseModel) and hasattr(
            completion_response, "usage"
        ):
            usage = completion_response.usage.model_dump()
        elif isinstance(completion_response, dict) and "usage" in completion_response:
            usage = completion_response["usage"]
        else:
            logger.warning(
                f"No usage info available in completion response: {completion_response}"
            )
            return None

        logger.debug(f"Usage: {json.dumps(usage, indent=2)}")

        prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens", 0)

        if usage.get("cache_creation_input_tokens"):
            prompt_tokens += usage["cache_creation_input_tokens"]

        completion_tokens = usage.get("completion_tokens") or usage.get(
            "output_tokens", 0
        )

        if usage.get("prompt_cache_hit_tokens"):
            cached_tokens = usage["prompt_cache_hit_tokens"]
        elif usage.get("cache_read_input_tokens"):
            cached_tokens = usage["cache_read_input_tokens"]
        else:
            cached_tokens = 0

        try:
            cost = litellm.completion_cost(
                completion_response=completion_response, model=model
            )
        except Exception:
            # If cost calculation fails, fall back to calculating it manually
            try:
                prompt_cost, completion_cost = cost_per_token(
                    model=model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
                cost = prompt_cost + completion_cost
            except NotFoundError as e:
                logger.debug(
                    f"Failed to calculate cost for completion response: {completion_response}. Error: {e}"
                )
                cost = 0
            except Exception as e:
                logger.debug(
                    f"Failed to calculate cost for completion response: {completion_response}. Error: {e}"
                )
                cost = 0

        return cls(
            completion_cost=cost,
            completion_tokens=completion_tokens,
            prompt_tokens=prompt_tokens,
            cached_tokens=cached_tokens,
        )

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            completion_cost=self.completion_cost + other.completion_cost,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
        )

    def __str__(self) -> str:
        return (
            f"Usage(cost: ${self.completion_cost:.4f}, "
            f"completion tokens: {self.completion_tokens}, "
            f"prompt tokens: {self.prompt_tokens}, "
            f"cached tokens: {self.cached_tokens})"
        )

    @model_validator(mode="before")
    @classmethod
    def fix_null_tokens(cls, data: Any) -> Any:
        if isinstance(data, dict):
            for key, value in data.items():
                if not value:
                    data[key] = 0

        return data


class Completion(BaseModel):
    model: str
    input: list[dict] | None = None
    response: dict[str, Any] | None = None
    usage: Usage | None = None

    @classmethod
    def from_llm_completion(
        cls, input_messages: list[dict], completion_response: Any, model: str
    ) -> Optional["Completion"]:
        if isinstance(completion_response, BaseModel):
            response = completion_response.model_dump()
        elif isinstance(completion_response, dict):
            response = completion_response
        else:
            logger.error(
                f"Unexpected completion response type: {type(completion_response)}"
            )
            return None

        usage = Usage.from_completion_response(completion_response, model)

        return cls(
            model=model,
            input=input_messages,
            response=response,
            usage=usage,
        )


class StructuredOutput(OpenAISchema):
    @classmethod
    def model_validate_json(
        cls,
        json_data: str | bytes | bytearray,
        **kwarg,
    ):
        if not json_data:
            raise ValidationError("Message is empty")

        try:
            parsed_data = json.loads(json_data, strict=False)
            cleaned_json = json.dumps(parsed_data)
            return super().model_validate_json(cleaned_json, **kwarg)

        except (json.JSONDecodeError, ValidationError) as e:
            # If direct parsing fails, try more aggressive cleanup
            logger.warning(f"Initial JSON parse failed, attempting alternate cleanup")

            message = json_data

            cleaned_message = "".join(
                char for char in message if ord(char) >= 32 or char in "\n\r\t"
            )
            if cleaned_message != message:
                logger.info(
                    f"parse_json() Cleaned control chars: {repr(message)} -> {repr(cleaned_message)}"
                )
            message = cleaned_message

            # Replace None with null
            message = message.replace(": None", ": null").replace(":None", ":null")

            # Extract JSON and try parsing again
            message, all_jsons = extract_json_from_message(message)
            if all_jsons:
                if len(all_jsons) > 1:
                    logger.warning(
                        f"Found multiple JSON objects, using the first one. All found: {all_jsons}"
                    )
                message = all_jsons[0]

            # Normalize line endings
            if isinstance(message, str):
                message = message.replace("\r\n", "\n").replace("\r", "\n")

            logger.debug(f"Final message to validate: {repr(message)}")

            return super().model_validate_json(
                message if isinstance(message, str) else json.dumps(message), **kwarg
            )


def extract_json_from_message(message: str) -> tuple[dict | str, list[dict]]:
    """
    Extract JSON from a message, handling both code blocks and raw JSON.
    Returns a tuple of (selected_json_dict, all_found_json_dicts).

    Args:
        message: The message to parse

    Returns:
        tuple[dict | str, list[dict]]: (The selected JSON dict to use or original message, List of all JSON dicts found)
    """

    def clean_json_string(json_str: str) -> str:
        # Remove single-line comments and clean control characters
        lines = []
        for line in json_str.split("\n"):
            # Remove everything after // or #
            line = line.split("//")[0].split("#")[0].rstrip()
            # Clean control characters but preserve newlines and spaces
            line = "".join(char for char in line if ord(char) >= 32 or char in "\n\t")
            if line:  # Only add non-empty lines
                lines.append(line)
        return "\n".join(lines)

    all_found_jsons = []

    # First try to find ```json blocks
    try:
        current_pos = 0
        while True:
            start = message.find("```json", current_pos)
            if start == -1:
                break
            start += 7  # Move past ```json
            end = message.find("```", start)
            if end == -1:
                break
            potential_json = clean_json_string(message[start:end].strip())
            try:
                json_dict = json.loads(potential_json)
                all_found_jsons.append(json_dict)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON from code block: {e}")
                pass
            current_pos = end + 3

        if all_found_jsons:
            return all_found_jsons[0], all_found_jsons
    except Exception as e:
        logger.warning(f"Failed to extract JSON from code blocks: {e}")

    # If no ```json blocks found, try to find raw JSON objects
    try:
        current_pos = 0
        while True:
            start = message.find("{", current_pos)
            if start == -1:
                break
            # Try to parse JSON starting from each { found
            for end in range(len(message), start, -1):
                try:
                    potential_json = clean_json_string(message[start:end])
                    json_dict = json.loads(potential_json)
                    all_found_jsons.append(json_dict)
                    break
                except json.JSONDecodeError:
                    continue
            if not all_found_jsons:  # If no valid JSON found, move past this {
                current_pos = start + 1
            else:
                current_pos = end

        if all_found_jsons:
            return all_found_jsons[0], all_found_jsons
    except Exception as e:
        logger.warning(f"Failed to extract raw JSON objects: {e}")

    return message, all_found_jsons
