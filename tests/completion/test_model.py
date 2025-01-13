import pytest
from pydantic import ValidationError, BaseModel, Field

from moatless.actions.create_file import CreateFileArgs
from moatless.actions.string_replace import StringReplaceArgs
from moatless.completion.model import StructuredOutput


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
    assert args.old_str == "        data = StringIO(data)\n        for obj in serializers.deserialize(\"json\", data, using=self.connection.alias):\n            obj.save()"
    assert args.new_str == "        data = StringIO(data)\n        with transaction.atomic(using=self.connection.alias):\n            for obj in serializers.deserialize(\"json\", data, using=self.connection.alias):\n                obj.save()"

def test_structured_output_name_and_description():
    # Test class with Config.title and docstring
    class TestWithConfig(StructuredOutput):
        """This is a test description."""
        class Config:
            title = "CustomTitle"

    assert TestWithConfig.name == "CustomTitle"
    assert TestWithConfig.description == "This is a test description."

    # Test class without Config.title but with docstring
    class TestWithoutConfig(StructuredOutput):
        """Another test description."""
        pass

    assert TestWithoutConfig.name == "TestWithoutConfig"
    assert TestWithoutConfig.description == "Another test description."

    # Test class without Config.title or docstring
    class TestBare(StructuredOutput):
        pass

    assert TestBare.name == "TestBare"
    assert TestBare.description == ""

def test_schema_without_refs():
    # Define a nested model
    class NestedModel(BaseModel):
        nested_field: str = Field(..., description="A nested field")
        optional_field: int = Field(None, description="An optional nested field")

    # Define the main model using StructuredOutput
    class MainSchema(StructuredOutput):
        """A test schema with nested model."""
        main_field: str = Field(..., description="A main field")
        nested: NestedModel = Field(..., description="A nested model field")

    # Get the OpenAI schema
    schema = MainSchema.openai_schema()

    # Print schema for debugging
    import json
    print(json.dumps(schema, indent=2))

    # Verify schema structure
    assert schema["type"] == "function"
    assert "function" in schema
    assert schema["function"]["name"] == "MainSchema"
    assert schema["function"]["description"] == "A test schema with nested model."

    # Verify parameters
    params = schema["function"]["parameters"]
    assert "properties" in params
    properties = params["properties"]

    # Check main field
    assert "main_field" in properties
    assert properties["main_field"]["description"] == "A main field"
    assert properties["main_field"]["type"] == "string"

    # Check nested field structure - should be expanded, not using $ref
    assert "nested" in properties
    assert "$ref" not in properties["nested"]
    assert properties["nested"]["description"] == "A nested model field"
    assert properties["nested"]["type"] == "object"
    
    # Verify nested model properties are expanded inline
    assert "properties" in properties["nested"]
    nested_props = properties["nested"]["properties"]
    
    assert "nested_field" in nested_props
    assert nested_props["nested_field"]["description"] == "A nested field"
    assert nested_props["nested_field"]["type"] == "string"

    assert "optional_field" in nested_props
    assert nested_props["optional_field"]["description"] == "An optional nested field"
    assert nested_props["optional_field"]["type"] == "integer"

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
    class ArraySchema(StructuredOutput):
        """A test schema with array of nested models."""
        items: list[ItemModel] = Field(..., description="A list of items")

    # Get the OpenAI schema
    schema = ArraySchema.openai_schema()

    # Print schema for debugging
    import json
    print(json.dumps(schema, indent=2))

    # Verify schema structure
    assert schema["type"] == "function"
    assert "function" in schema
    assert schema["function"]["name"] == "ArraySchema"
    assert schema["function"]["description"] == "A test schema with array of nested models."

    # Verify parameters
    params = schema["function"]["parameters"]
    assert "properties" in params
    properties = params["properties"]

    # Check items field
    assert "items" in properties
    assert properties["items"]["description"] == "A list of items"
    assert properties["items"]["type"] == "array"
    
    # Check items schema
    items_schema = properties["items"]["items"]
    assert "$ref" not in items_schema
    assert items_schema["type"] == "object"
    assert "properties" in items_schema
    
    item_props = items_schema["properties"]
    assert "item_field" in item_props
    assert item_props["item_field"]["description"] == "An item field"
    assert item_props["item_field"]["type"] == "string"

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
