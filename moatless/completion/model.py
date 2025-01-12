import hashlib
import json
import logging
from typing import Optional, Any, Union, Self, ClassVar

from docstring_parser import parse
from instructor.utils import classproperty
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

    def __post_init__(self):
        # Ensure name is always a string
        self.name = str(self.name)


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
        tool_str = f"{str(self.tool_call.name)}:{json.dumps(self.tool_call.input, sort_keys=True)}"
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
            import litellm

            cost = litellm.completion_cost(
                completion_response=completion_response, model=model
            )
        except Exception:
            # If cost calculation fails, fall back to calculating it manually
            try:
                from litellm import cost_per_token, NotFoundError

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
        # Get completion cost, defaulting to 0 if not available
        other_cost = getattr(other, "completion_cost", 0)
        other_completion = getattr(other, "completion_tokens", 0)
        other_prompt = getattr(other, "prompt_tokens", 0)
        other_cached = getattr(other, "cached_tokens", 0)

        return Usage(
            completion_cost=self.completion_cost + other_cost,
            completion_tokens=self.completion_tokens + other_completion,
            prompt_tokens=self.prompt_tokens + other_prompt,
            cached_tokens=self.cached_tokens + other_cached,
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
    retries: int | None = None
    usage: Usage | None = None
    flags: list[str] = Field(
        default_factory=list,
        description="List of flags indicating special conditions or states during completion",
    )

    @classmethod
    def from_llm_completion(
        cls,
        input_messages: list[dict],
        completion_response: Any,
        model: str,
        usage: Usage | None = None,
        retries: int | None = None,
        flags: list[str] | None = None,
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

        if not usage:
            usage = Usage.from_completion_response(completion_response, model)

        return cls(
            model=model,
            input=input_messages,
            response=response,
            retries=retries,
            usage=usage,
            flags=flags or [],
        )


class NameDescriptor:
    def __get__(self, obj, cls=None) -> str:
        if hasattr(cls, "Config") and hasattr(cls.Config, "title") and cls.Config.title:
            return cls.Config.title
        return cls.__name__


class StructuredOutput(BaseModel):
    name: ClassVar[NameDescriptor] = NameDescriptor()

    class Config:
        ignored_types = (classproperty,)

    @classproperty
    def description(cls):
        return cls.model_json_schema().get("description", "")

    @classmethod
    def openai_schema(cls, thoughts_in_action: bool = False) -> dict[str, Any]:
        """
        Return the schema in the format of OpenAI's schema as jsonschema
        """
        schema = cls.model_json_schema()
        docstring = parse(cls.__doc__ or "")
        parameters = {
            k: v
            for k, v in schema.items()
            if k not in ("title", "description")
            and (thoughts_in_action or k != "thoughts")
        }

        if not thoughts_in_action and parameters["properties"].get("thoughts"):
            del parameters["properties"]["thoughts"]

        def remove_defaults(obj: dict) -> None:
            """Recursively remove default fields from a schema object"""
            if isinstance(obj, dict):
                if "default" in obj:
                    del obj["default"]
                # Recurse into nested properties
                for value in obj.values():
                    remove_defaults(value)
            elif isinstance(obj, list):
                for item in obj:
                    remove_defaults(item)

        def resolve_refs(obj: dict, defs: dict) -> dict:
            """Recursively resolve $ref references in the schema"""
            if not isinstance(obj, dict):
                return obj

            result = {}
            for k, v in obj.items():
                if k == "items" and isinstance(v, dict) and "$ref" in v:
                    # Handle array items that use $ref
                    ref_path = v["$ref"]
                    if ref_path.startswith("#/$defs/"):
                        ref_name = ref_path.split("/")[-1]
                        if ref_name in defs:
                            result[k] = defs[ref_name].copy()
                            continue
                elif k == "$ref":
                    ref_path = v
                    if ref_path.startswith("#/$defs/"):
                        ref_name = ref_path.split("/")[-1]
                        if ref_name in defs:
                            # Create a new dict with all properties except $ref
                            resolved = {
                                k2: v2 for k2, v2 in obj.items() if k2 != "$ref"
                            }
                            # Merge with the referenced definition
                            referenced = defs[ref_name].copy()
                            referenced.update(resolved)
                            return resolve_refs(referenced, defs)

                # Recursively resolve nested objects/arrays
                if isinstance(v, dict):
                    result[k] = resolve_refs(v, defs)
                elif isinstance(v, list):
                    result[k] = [
                        resolve_refs(item, defs) if isinstance(item, dict) else item
                        for item in v
                    ]
                else:
                    result[k] = v

            return result

        # Remove default field from all properties recursively
        remove_defaults(parameters)

        # Resolve all $ref references
        if "$defs" in parameters:
            defs = parameters.pop("$defs")
            parameters = resolve_refs(parameters, defs)

        for param in docstring.params:
            if (name := param.arg_name) in parameters["properties"] and (
                description := param.description
            ):
                if "description" not in parameters["properties"][name]:
                    parameters["properties"][name]["description"] = description

        parameters["required"] = sorted(
            k
            for k, v in parameters["properties"].items()
            if "default" not in v and (thoughts_in_action or k != "thoughts")
        )

        if "description" not in schema:
            if docstring.short_description:
                schema["description"] = docstring.short_description
            else:
                schema["description"] = (
                    f"Correctly extracted `{cls.__name__}` with all "
                    f"the required parameters with correct types"
                )
        name = cls.name
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": schema["description"],
                "parameters": parameters,
            },
        }

    @classmethod
    def anthropic_schema(cls) -> dict[str, Any]:
        schema = cls.model_json_schema()
        del schema["title"]

        if "description" in schema:
            description = schema["description"]
            del schema["description"]
        else:
            description = None

        response = {
            "name": cls.name,
            "input_schema": schema,
        }

        if description:
            response["description"] = description

        # Exclude thoughts field from properties and required if it exists
        if "thoughts" in schema.get("properties", {}):
            del schema["properties"]["thoughts"]
            if "required" in schema and "thoughts" in schema["required"]:
                schema["required"].remove("thoughts")

        return response

    @classmethod
    def model_validate_xml(cls, xml_text: str) -> Self:
        """Parse XML format into model fields."""
        parsed_input = {}
        # Fields that can be parsed from XML format
        xml_fields = ["path", "old_str", "new_str", "file_text", "insert_line"]

        for field in xml_fields:
            start_tag = f"<{field}>"
            end_tag = f"</{field}>"
            if start_tag in xml_text and end_tag in xml_text:
                start_idx = xml_text.index(start_tag) + len(start_tag)
                end_idx = xml_text.index(end_tag)
                content = xml_text[start_idx:end_idx]

                # Handle both single-line and multi-line block content
                if content:
                    # If content starts/ends with newlines, preserve the inner content
                    if content.startswith("\n") and content.endswith("\n"):
                        # Remove first and last newline but preserve internal formatting
                        content = content[1:-1].rstrip("\n")
                    parsed_input[field] = content

        return cls.model_validate(parsed_input)

    @classmethod
    def model_validate_json(
        cls,
        json_data: str | bytes | bytearray,
        **kwarg,
    ) -> Self:
        if not json_data:
            raise ValidationError("Message is empty")

        try:
            parsed_data = json.loads(json_data, strict=False)

            def unescape_values(obj):
                if isinstance(obj, dict):
                    return {k: unescape_values(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [unescape_values(v) for v in obj]
                elif isinstance(obj, str) and "\\" in obj:
                    return obj.encode().decode("unicode_escape")
                return obj

            cleaned_data = unescape_values(parsed_data)
            cleaned_json = json.dumps(cleaned_data)
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

    def format_args_for_llm(self) -> str:
        """
        Format the input arguments for LLM completion calls. Override in subclasses for custom formats.
        Default implementation returns JSON format.
        """
        return json.dumps(
            self.model_dump(
                exclude={"thoughts"} if hasattr(self, "thoughts") else None
            ),
            indent=2,
        )

    @classmethod
    def format_schema_for_llm(cls, thoughts_in_action: bool = False) -> str:
        """
        Format the schema description for LLM completion calls.
        Default implementation returns JSON schema.
        """
        schema = cls.model_json_schema()

        if not thoughts_in_action and schema["properties"].get("thoughts"):
            del schema["properties"]["thoughts"]
            schema["required"] = sorted(
                k
                for k, v in schema["properties"].items()
                if "default" not in v and (thoughts_in_action or k != "thoughts")
            )

        return f"Requires a JSON response with the following schema: {json.dumps(schema, ensure_ascii=False)}"

    @classmethod
    def format_xml_schema(cls, xml_fields: dict[str, str]) -> str:
        """
        Format XML schema description.
        Used by actions that require XML-formatted input.

        Args:
            xml_fields: Dictionary mapping field names to their descriptions
        """
        schema = [f"Requires the following XML format:"]

        # Build example XML structure
        example = []
        for field_name, field_desc in xml_fields.items():
            example.append(f"<{field_name}>{field_desc}</{field_name}>")

        return "\n".join(schema + example)


def extract_json_from_message(message: str) -> tuple[dict | str, list[dict]]:
    """
    Extract JSON from a message, handling both code blocks and raw JSON.
    Returns a tuple of (selected_json_dict, all_found_json_dicts).
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
                # Validate that this is a complete, non-truncated JSON object
                if isinstance(json_dict, dict) and all(
                    isinstance(k, str) for k in json_dict.keys()
                ):
                    all_found_jsons.append(json_dict)
            except json.JSONDecodeError:
                pass
            current_pos = end + 3

        if all_found_jsons:
            # Return the most complete JSON object (one with the most fields)
            return max(all_found_jsons, key=lambda x: len(x)), all_found_jsons
    except Exception as e:
        logger.warning(f"Failed to extract JSON from code blocks: {e}")

    # If no ```json blocks found or they failed, try to find raw JSON objects
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
                    # Validate that this is a complete, non-truncated JSON object
                    if isinstance(json_dict, dict) and all(
                        isinstance(k, str) for k in json_dict.keys()
                    ):
                        all_found_jsons.append(json_dict)
                    break
                except json.JSONDecodeError:
                    continue
            if not all_found_jsons:  # If no valid JSON found, move past this {
                current_pos = start - +1
            else:
                current_pos = end

        if all_found_jsons:
            # Return the most complete JSON object (one with the most fields)
            return max(all_found_jsons, key=lambda x: len(x)), all_found_jsons
    except Exception as e:
        logger.warning(f"Failed to extract raw JSON objects: {e}")

    return message, all_found_jsons
