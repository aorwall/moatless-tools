from typing import Optional, ClassVar, List

import pytest
import json
from moatless.actions.create_file import CreateFileArgs
from moatless.actions.string_replace import StringReplaceArgs
from moatless.completion.schema import ResponseSchema, FewShotExample
from pydantic import ValidationError, BaseModel, Field, ConfigDict


def test_string_replace_xml_validation():
    # Test valid XML format
    valid_xml = """
<path>test/file.py</path>
<old_str>def old_function():
    pass</old_str>
<new_str>def new_function():
    return True</new_str>
"""

    args = StringReplaceArgs.model_validate_xml(valid_xml)
    assert args.path == "test/file.py"
    assert args.old_str == "def old_function():\n    pass"
    assert args.new_str == "def new_function():\n    return True"

    # Test invalid XML - missing closing tag
    invalid_xml = """
<path>test/file.py</path>
<old_str>some code</old_str>
<new_str>new code
"""
    with pytest.raises(ValidationError):
        StringReplaceArgs.model_validate_xml(invalid_xml)

    # Test invalid XML - wrong tag names
    wrong_tags_xml = """
<path>test/file.py</path>
<old_str>some code</old_str>
<file_text>new code</file_text>
"""
    with pytest.raises(ValidationError):
        StringReplaceArgs.model_validate_xml(wrong_tags_xml)


def test_create_file_xml_validation():
    # Test valid XML format
    valid_xml = """
<path>new/test/file.py</path>
<file_text>def test_function():
    return True</file_text>
"""

    args = CreateFileArgs.model_validate_xml(valid_xml)
    assert args.path == "new/test/file.py"
    assert args.file_text == "def test_function():\n    return True"


def test_string_replace_indentation():
    # Test indentation preservation in XML format
    xml_with_indentation = """
<path>test/file.py</path>
<old_str>        data = StringIO(data)
        for obj in serializers.deserialize("json", data, using=self.connection.alias):
            obj.save()</old_str>
<new_str>        data = StringIO(data)
        with transaction.atomic(using=self.connection.alias):
            for obj in serializers.deserialize("json", data, using=self.connection.alias):
                obj.save()</new_str>
"""

    args = StringReplaceArgs.model_validate_xml(xml_with_indentation)
    assert args.path == "test/file.py"
    assert (
        args.old_str
        == '        data = StringIO(data)\n        for obj in serializers.deserialize("json", data, using=self.connection.alias):\n            obj.save()'
    )
    assert (
        args.new_str
        == '        data = StringIO(data)\n        with transaction.atomic(using=self.connection.alias):\n            for obj in serializers.deserialize("json", data, using=self.connection.alias):\n                obj.save()'
    )


def test_structured_output_name_and_description():
    # Test class with Config.title and docstring
    class TestWithConfig(ResponseSchema):
        """This is a test description."""

        model_config = {"title": "CustomTitle"}

    assert TestWithConfig.name == "CustomTitle"
    # Debug: Print raw docstring and description method result
    print(f"Raw docstring: {TestWithConfig.__doc__!r}")
    print(f"Description method result: {TestWithConfig.description()!r}")
    assert TestWithConfig.description() == "This is a test description."

    # Test class without Config.title but with docstring
    class TestWithoutConfig(ResponseSchema):
        """Another test description."""

        pass

    assert TestWithoutConfig.name == "TestWithoutConfig"
    assert TestWithoutConfig.description() == "Another test description."

    # Test class without Config.title or docstring
    class TestBare(ResponseSchema):
        pass

    assert TestBare.name == "TestBare"
    assert TestBare.description() == ""


def test_schema_without_refs():
    # Define a nested model
    class NestedModel(BaseModel):
        nested_field: str = Field(..., description="A nested field")
        optional_field: Optional[int] = Field(None, description="An optional nested field")

    # Define the main model using StructuredOutput
    class MainSchema(ResponseSchema):
        """A test schema with nested model."""

        main_field: str = Field(..., description="A main field")
        nested: NestedModel = Field(..., description="A nested model field")

    # Get the OpenAI schema
    schema = MainSchema.openai_schema()

    # Print schema for debugging
    print(json.dumps(schema, indent=2))

    # Verify schema structure - using get() to avoid TypedDict access issues
    assert schema.get("type") == "function"
    assert "function" in schema
    function_dict = schema.get("function", {})
    assert function_dict.get("name") == "MainSchema"
    assert function_dict.get("description") == "A test schema with nested model."

    # Verify parameters
    params = function_dict.get("parameters", {})
    assert "properties" in params
    properties = params.get("properties", {})

    # Check main field
    assert "main_field" in properties
    main_field_props = properties.get("main_field", {})
    assert main_field_props.get("description") == "A main field"
    assert main_field_props.get("type") == "string"

    # Check nested field structure - should be expanded, not using $ref
    assert "nested" in properties
    nested_props = properties.get("nested", {})
    assert "$ref" not in nested_props
    assert nested_props.get("description") == "A nested model field"
    assert nested_props.get("type") == "object"

    # Verify nested model properties are expanded inline
    assert "properties" in nested_props
    nested_obj_props = nested_props.get("properties", {})

    assert "nested_field" in nested_obj_props
    nested_field_props = nested_obj_props.get("nested_field", {})
    assert nested_field_props.get("description") == "A nested field"
    assert nested_field_props.get("type") == "string"

    assert "optional_field" in nested_obj_props
    optional_field_props = nested_obj_props.get("optional_field", {})
    assert optional_field_props.get("description") == "An optional nested field"

    # For Optional[int], Pydantic creates an anyOf schema with integer and null types
    if "type" in optional_field_props:
        assert optional_field_props.get("type") == "integer"
    elif "anyOf" in optional_field_props:
        # Check that anyOf contains both integer and null types
        type_options = optional_field_props.get("anyOf", [])
        has_integer = any(opt.get("type") == "integer" for opt in type_options)
        has_null = any(opt.get("type") == "null" for opt in type_options)
        assert has_integer and has_null, "Optional field should have both integer and null types in anyOf"

    # Verify no $ref is used anywhere in the schema
    def check_no_refs(obj):
        if isinstance(obj, dict):
            assert "$ref" not in obj
            for value in obj.values():
                check_no_refs(value)
        elif isinstance(obj, list):
            for item in obj:
                check_no_refs(item)

    check_no_refs(schema)


def test_schema_with_array_refs():
    # Define a nested model
    class ItemModel(BaseModel):
        item_field: str = Field(..., description="An item field")

    # Define the main model using StructuredOutput
    class ArraySchema(ResponseSchema):
        """A test schema with array of nested models."""

        items: list[ItemModel] = Field(..., description="A list of items")

    # Get the OpenAI schema
    schema = ArraySchema.openai_schema()

    # Print schema for debugging
    print(json.dumps(schema, indent=2))

    # Verify schema structure - using get() to avoid TypedDict access issues
    assert schema.get("type") == "function"
    assert "function" in schema
    function_dict = schema.get("function", {})
    assert function_dict.get("name") == "ArraySchema"
    assert function_dict.get("description") == "A test schema with array of nested models."

    # Verify parameters
    params = function_dict.get("parameters", {})
    assert "properties" in params
    properties = params.get("properties", {})

    # Check items field
    assert "items" in properties
    items_props = properties.get("items", {})
    assert items_props.get("description") == "A list of items"
    assert items_props.get("type") == "array"

    # Check items schema
    items_schema = items_props.get("items", {})
    assert "$ref" not in items_schema
    assert items_schema.get("type") == "object"
    assert "properties" in items_schema

    item_props = items_schema.get("properties", {})
    assert "item_field" in item_props
    item_field_props = item_props.get("item_field", {})
    assert item_field_props.get("description") == "An item field"
    assert item_field_props.get("type") == "string"

    # Verify no $ref is used anywhere in the schema
    def check_no_refs(obj):
        if isinstance(obj, dict):
            assert "$ref" not in obj
            for value in obj.values():
                check_no_refs(value)
        elif isinstance(obj, list):
            for item in obj:
                check_no_refs(item)

    check_no_refs(schema)


def test_response_schema_name_behavior():
    # Test class with model_config title
    class SchemaWithConfigTitle(ResponseSchema):
        field: str
        model_config = ConfigDict(title="CustomName")

    assert SchemaWithConfigTitle.name == "CustomName"

    # Test class without model_config title - should use class name
    class SchemaWithoutConfigTitle(ResponseSchema):
        field: str

    assert SchemaWithoutConfigTitle.name == "SchemaWithoutConfigTitle"

    # Test class with empty model_config - should use class name
    class SchemaWithEmptyConfig(ResponseSchema):
        field: str
        model_config = ConfigDict()

    assert SchemaWithEmptyConfig.name == "SchemaWithEmptyConfig"

    # Test class with model_config but no title - should use class name
    class SchemaWithConfigNoTitle(ResponseSchema):
        field: str
        model_config = ConfigDict(arbitrary_types_allowed=True)

    assert SchemaWithConfigNoTitle.name == "SchemaWithConfigNoTitle"


def test_resolving_nested_refs_in_schema():
    """Test that nested objects with $ref are properly resolved in the schema"""

    # Define a test schema that contains a nested object with $ref
    class TestFileLocation(BaseModel):
        file_path: str = Field(description="Path to the file")
        start_line: Optional[int] = Field(default=None, description="Starting line number")
        end_line: Optional[int] = Field(default=None, description="Ending line number")

    class TestNestedItem(BaseModel):
        id: str = Field(description="ID of the item")
        title: str = Field(description="Title of the item")
        related_files: list[TestFileLocation] = Field(default_factory=list, description="List of related files")

    class TestSchema(ResponseSchema):
        """Test schema with nested references"""

        items: list[TestNestedItem] = Field(description="List of items")

    # Get the OpenAI schema
    schema = TestSchema.openai_schema()

    # Print schema for debugging
    print(json.dumps(schema, indent=2))

    # Verify the schema properly resolves nested objects
    function_dict = schema.get("function", {})
    params = function_dict.get("parameters", {})
    properties = params.get("properties", {})

    # Check items array
    assert "items" in properties, "Items property missing in schema"
    items_prop = properties["items"]
    assert items_prop["type"] == "array", "Items should be an array"

    # Check nested item schema
    nested_item_schema = items_prop["items"]
    assert nested_item_schema["type"] == "object", "Nested item should be an object"
    nested_item_properties = nested_item_schema["properties"]

    # Check related_files array
    assert "related_files" in nested_item_properties, "related_files property missing in nested item"
    related_files_prop = nested_item_properties["related_files"]
    assert related_files_prop["type"] == "array", "related_files should be an array"

    # Check FileLocation schema - it should be expanded, not a $ref
    file_location_schema = related_files_prop["items"]
    assert "$ref" not in file_location_schema, "FileLocation should not be a $ref but fully expanded"
    assert file_location_schema["type"] == "object", "FileLocation should be an object"

    # Verify FileLocation properties are present
    file_location_properties = file_location_schema["properties"]
    assert "file_path" in file_location_properties, "file_path property missing in FileLocation"
    assert "start_line" in file_location_properties, "start_line property missing in FileLocation"
    assert "end_line" in file_location_properties, "end_line property missing in FileLocation"


def test_nested_refs_in_real_schema():
    """Test that nested objects with $ref are properly resolved in a real schema"""

    # Create a simplified version of what's in add_coding_tasks.py
    class FileRelationType(str):
        CREATE = "create"
        UPDATE = "update"
        REFERENCE = "reference"
        DEPENDENCY = "dependency"

    class FileLocation(BaseModel):
        """Represents a file location with path and line range"""

        file_path: str = Field(description="Path to the file")
        start_line: Optional[int] = Field(default=None, description="Starting line number")
        end_line: Optional[int] = Field(default=None, description="Ending line number")
        relation_type: str = Field(default="reference", description="How this file relates to the task")

    class CodingTaskItem(BaseModel):
        """A single coding task item to be created."""

        id: str = Field(
            ...,
            description="Identifier or short name for the task. This will be used as the task's ID in the system.",
        )

        title: str = Field(
            ...,
            description="Short title or description of the task.",
        )

        instructions: str = Field(
            ...,
            description="Detailed instructions for completing the task.",
        )

        related_files: List[FileLocation] = Field(
            default_factory=list,
            description="List of files related to this task with their locations and relationship types.",
        )

        priority: int = Field(
            default=100,
            description="Execution priority - lower numbers = higher priority.",
        )

    class AddCodingTasksArgs(ResponseSchema):
        """Create new coding tasks with the given descriptions and file locations."""

        tasks: List[CodingTaskItem] = Field(
            ...,
            description="List of coding tasks to create.",
        )

        model_config = ConfigDict(title="AddCodingTasks")

    # Generate the schema
    schema = AddCodingTasksArgs.openai_schema()

    # Print schema for debugging
    print(json.dumps(schema, indent=2))

    # Check that the schema has no $ref fields anywhere
    def check_no_refs(obj):
        if isinstance(obj, dict):
            assert "$ref" not in obj, f"Found $ref in: {obj}"
            for value in obj.values():
                check_no_refs(value)
        elif isinstance(obj, list):
            for item in obj:
                check_no_refs(item)

    check_no_refs(schema)

    # Verify the schema properly expands nested objects
    function_dict = schema.get("function", {})
    params = function_dict.get("parameters", {})
    properties = params.get("properties", {})

    # Check tasks array
    assert "tasks" in properties, "Tasks property missing in schema"
    tasks_prop = properties["tasks"]
    assert tasks_prop["type"] == "array", "Tasks should be an array"

    # Check CodingTaskItem schema
    task_item_schema = tasks_prop["items"]
    assert task_item_schema["type"] == "object", "Task item should be an object"
    task_item_properties = task_item_schema["properties"]

    # Check related_files array
    assert "related_files" in task_item_properties, "related_files property missing in task item"
    related_files_prop = task_item_properties["related_files"]
    assert related_files_prop["type"] == "array", "related_files should be an array"

    # Check FileLocation schema - it should be expanded, not a $ref
    file_location_schema = related_files_prop["items"]
    assert "$ref" not in file_location_schema, "FileLocation should not be a $ref but fully expanded"
    assert file_location_schema["type"] == "object", "FileLocation should be an object"

    # Verify FileLocation properties are present
    file_location_properties = file_location_schema["properties"]
    assert "file_path" in file_location_properties, "file_path property missing in FileLocation"
    assert "start_line" in file_location_properties, "start_line property missing in FileLocation"
    assert "end_line" in file_location_properties, "end_line property missing in FileLocation"
    assert "relation_type" in file_location_properties, "relation_type property missing in FileLocation"
