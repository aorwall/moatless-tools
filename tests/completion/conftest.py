from typing import Optional, List

import pytest
from pydantic import Field

from moatless.completion.schema import ResponseSchema


class TestAction(ResponseSchema):
    """Test action schema for completion tests"""
    name = "test_action"
    command: str = Field(..., description="Command to execute")
    args: List[str] = Field(default_factory=list, description="Command arguments")
    thoughts: Optional[str] = None

    @classmethod
    def format_schema_for_llm(cls) -> str:
        return """
        command: The command to execute
        args: Optional list of command arguments
        """
    
    @classmethod
    def tool_schema(cls, thoughts_in_action: bool = False) -> dict:
        return {
            "type": "function",
            "function": {
                "name": cls.name,
                "description": "Execute a test command",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Command to execute"},
                        "args": {"type": "array", "items": {"type": "string"}, "description": "Command arguments"},
                        **({"thoughts": {"type": "string"}} if thoughts_in_action else {})
                    },
                    "required": ["command"]
                }
            }
        }


@pytest.fixture
def test_schema():
    """Basic test action schema"""
    return TestAction


@pytest.fixture
def test_messages():
    """Sample conversation history"""
    return [
        {"role": "user", "content": "Run the test command"},
    ]
