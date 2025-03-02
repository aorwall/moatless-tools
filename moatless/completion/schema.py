import json
from typing import (
    ClassVar,
    Self,
    Union,
    TypedDict,
    Literal,
    Iterable,
    Required,
    Optional,
    List,
)

from docstring_parser import parse
from pydantic import BaseModel, ValidationError, Field

from moatless.completion.model import logger


class ChatCompletionCachedContent(TypedDict):
    type: Literal["ephemeral"]


class ChatCompletionToolParamFunctionChunk(TypedDict, total=False):
    name: Required[str]
    description: str
    parameters: dict


class ChatCompletionToolParam(TypedDict, total=False):
    type: Union[Literal["function"], str]
    function: ChatCompletionToolParamFunctionChunk
    cache_control: Optional[ChatCompletionCachedContent]


class ChatCompletionTextObject(TypedDict):
    type: Literal["text"]
    text: str
    cache_control: Optional[ChatCompletionCachedContent]


class ChatCompletionThinkingObject(TypedDict):
    type: Literal["thinking"]
    thinking: Optional[str]
    signature: Optional[str]
    data: Optional[str]


class ChatCompletionImageUrlObject(TypedDict, total=False):
    url: Required[str]
    detail: str


class ChatCompletionImageObject(TypedDict):
    type: Literal["image_url"]
    image_url: Union[str, ChatCompletionImageUrlObject]


MessageContentListBlock = Union[ChatCompletionTextObject, ChatCompletionImageObject]

MessageContent = Union[
    str,
    Iterable[MessageContentListBlock],
]

ValidUserMessageContentTypes = [
    "text",
    "image_url",
    "input_audio",
]  # used for validating user messages. Prevent users from accidentally sending anthropic messages.


class ChatCompletionUserMessage(TypedDict):
    role: Literal["user"]
    content: MessageContent
    cache_control: Optional[ChatCompletionCachedContent]


class ChatCompletionToolCallFunctionChunk(TypedDict, total=False):
    name: Optional[str]
    arguments: str


class ChatCompletionAssistantToolCall(TypedDict):
    id: Optional[str]
    type: Literal["function"]
    function: ChatCompletionToolCallFunctionChunk


class ChatCompletionMessage(TypedDict, total=False):
    role: Required[Literal["assistant", "user", "tool"]]
    content: Optional[Union[str, Iterable[ChatCompletionTextObject]]]


class ChatCompletionAssistantMessage(ChatCompletionMessage, total=False):
    name: Optional[str]
    tool_calls: Optional[List[ChatCompletionAssistantToolCall]]
    function_call: Optional[ChatCompletionToolCallFunctionChunk]
    cache_control: Optional[ChatCompletionCachedContent]


class ChatCompletionToolMessage(ChatCompletionMessage):
    tool_call_id: str


class ChatCompletionSystemMessage(TypedDict, total=False):
    role: Required[Literal["system"]]
    content: Required[Union[str, List]]
    name: str
    cache_control: Optional[ChatCompletionCachedContent]


AllMessageValues = Union[
    ChatCompletionUserMessage,
    ChatCompletionAssistantMessage,
    ChatCompletionToolMessage,
    ChatCompletionSystemMessage,
]


class NameDescriptor:
    def __get__(self, obj, cls=None) -> str:
        if hasattr(cls, "model_config") and "title" in cls.model_config:
            return cls.model_config["title"]
        return cls.__name__


class ResponseSchema(BaseModel):
    name: ClassVar[NameDescriptor] = NameDescriptor()

    @classmethod
    def description(cls):
        return cls.model_json_schema().get("description", "")

    @classmethod
    def tool_schema(cls, thoughts_in_action: bool = False) -> ChatCompletionToolParam:
        return cls.openai_schema(thoughts_in_action=thoughts_in_action)

    @classmethod
    def openai_schema(cls, thoughts_in_action: bool = False) -> ChatCompletionToolParam:
        """
        Return the schema in the format of OpenAI's schema as jsonschema
        """
        schema = cls.model_json_schema()
        docstring = parse(cls.__doc__ or "")
        parameters = {
            k: v
            for k, v in schema.items()
            if k not in ("title", "description") and (thoughts_in_action or k != "thoughts")
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
                            resolved = {k2: v2 for k2, v2 in obj.items() if k2 != "$ref"}
                            # Merge with the referenced definition
                            referenced = defs[ref_name].copy()
                            referenced.update(resolved)
                            return resolve_refs(referenced, defs)
                elif k == "allOf" and isinstance(v, list):
                    # Merge all objects in allOf into a single object
                    merged = {}
                    for item in v:
                        resolved_item = resolve_refs(item, defs)
                        merged.update(resolved_item)
                    # Copy over any other properties from the parent object
                    for other_k, other_v in obj.items():
                        if other_k != "allOf":
                            merged[other_k] = other_v
                    return merged

                # Recursively resolve nested objects/arrays
                if isinstance(v, dict):
                    result[k] = resolve_refs(v, defs)
                elif isinstance(v, list):
                    result[k] = [resolve_refs(item, defs) if isinstance(item, dict) else item for item in v]
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
            if (name := param.arg_name) in parameters["properties"] and (description := param.description):
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
                    f"Correctly extracted `{cls.__name__}` with all " f"the required parameters with correct types"
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
    def model_validate_xml(cls, xml_text: str) -> Self:
        """Parse XML format into model fields."""
        parsed_input = {}
        # Get fields from the class's schema
        schema = cls.model_json_schema()
        properties = schema.get("properties", {})

        if "thoughts" in properties:
            del properties["thoughts"]

        xml_fields = list(properties.keys())

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

            cleaned_message = "".join(char for char in message if ord(char) >= 32 or char in "\n\r\t")
            if cleaned_message != message:
                logger.info(f"parse_json() Cleaned control chars: {repr(message)} -> {repr(cleaned_message)}")
            message = cleaned_message

            # Replace None with null
            message = message.replace(": None", ": null").replace(":None", ":null")

            # Extract JSON and try parsing again
            message, all_jsons = extract_json_from_message(message)
            if all_jsons:
                if len(all_jsons) > 1:
                    logger.warning(f"Found multiple JSON objects, using the first one. All found: {all_jsons}")
                message = all_jsons[0]

            # Normalize line endings
            if isinstance(message, str):
                message = message.replace("\r\n", "\n").replace("\r", "\n")

            return super().model_validate_json(message if isinstance(message, str) else json.dumps(message), **kwarg)

    def format_args_for_llm(self) -> str:
        """
        Format the input arguments for LLM completion calls. Override in subclasses for custom formats.
        Default implementation returns JSON format.
        """
        return json.dumps(
            self.model_dump(exclude={"thoughts"} if hasattr(self, "thoughts") else None),
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

    @classmethod
    def get_few_shot_examples(cls) -> List["FewShotExample"]:
        """
        Returns a list of few-shot examples specific to this action.
        Override this method in subclasses to provide custom examples.
        """
        return []


class FewShotExample(BaseModel):
    user_input: str = Field(..., description="The user's input/question")
    action: ResponseSchema = Field(..., description="The expected response")

    @classmethod
    def create(cls, user_input: str, action: ResponseSchema) -> "FewShotExample":
        return cls(user_input=user_input, action=action)


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
                if isinstance(json_dict, dict) and all(isinstance(k, str) for k in json_dict.keys()):
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
                    if isinstance(json_dict, dict) and all(isinstance(k, str) for k in json_dict.keys()):
                        all_found_jsons.append(json_dict)
                    break
                except json.JSONDecodeError:
                    continue
            if not all_found_jsons:  # If no valid JSON found, move past this {
                current_pos = start + 1
            else:
                current_pos = end

        if all_found_jsons:
            # Return the most complete JSON object (one with the most fields)
            return max(all_found_jsons, key=lambda x: len(x)), all_found_jsons
    except Exception as e:
        logger.warning(f"Failed to extract raw JSON objects: {e}")

    return message, all_found_jsons
