from typing import Optional, List

import pytest
from pydantic import Field

from moatless.actions.schema import ActionArguments
from moatless.completion.schema import ResponseSchema


class TestAction(ActionArguments):
    """Test action schema for completion tests"""
    name = "test_action"
    command: str = Field(..., description="Command to execute")
    args: List[str] = Field(default_factory=list, description="Command arguments")
    thoughts: Optional[str] = Field(default=None)

    @classmethod
    def format_schema_for_llm(cls) -> str:
        return """
        command: The command to execute
        args: Optional list of command arguments
        """


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
