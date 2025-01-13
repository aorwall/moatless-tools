import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from moatless.actions import Finish
from moatless.actions.finish import FinishArgs
from moatless.agent.agent import ActionAgent
from moatless.artifacts.file import FileArtifactHandler
from moatless.chat import Chat
from moatless.completion.completion import CompletionModel, CompletionResponse
from moatless.schema import Attachment, UserMessage, AssistantMessage


@pytest.fixture
def mock_completion_model():
    model = Mock(spec=CompletionModel)

    def mock_create_completion(*args, **kwargs):
        return CompletionResponse(
            text_response="I'll help with that",
        )
    
    model.create_completion = mock_create_completion
    model.model = "mock-model"
    model.temperature = 0.0
    return model


@pytest.fixture
def temp_dir(tmp_path):
    return tmp_path


def test_chat_flow_and_dump(mock_completion_model, temp_dir):
    # Setup
    system_prompt = "You're an AI assistant"
    
    agent = ActionAgent(
        completion=mock_completion_model,
        system_prompt=system_prompt,
        actions=[Finish()]
    )

    artifact_handlers = [
        FileArtifactHandler(directory_path=temp_dir),
    ]

    chat = Chat(
        agent=agent,
        artifact_handlers=artifact_handlers
    )

    # Test sending a message
    response = chat.send_message("Hello, how are you?")
    assert response is not None

    # Test sending a message with attachment
    test_file_content = b"print('hello world')"
    attachment = Attachment(
        file_name="test.py",
        content=test_file_content,
        mime_type="text/plain"
    )
    response = chat.send_message("Here's a Python file", attachments=[attachment])
    assert response is not None

    # Test getting messages
    messages = chat.get_messages()
    assert len(messages) >= 2  # Should have at least user and assistant messages
    
    # Validate message types and content
    user_messages = [m for m in messages if isinstance(m, UserMessage)]
    assistant_messages = [m for m in messages if isinstance(m, AssistantMessage)]
    
    assert len(user_messages) >= 2  # Initial message and file message
    assert len(assistant_messages) >= 2  # Responses to both messages
    
    # Verify first message content
    assert user_messages[0].content == "Hello, how are you?"
    assert user_messages[1].content == "Here's a Python file"
    assert user_messages[1].artifact_ids == ["test.py"]

    # Test getting artifacts
    artifacts = chat.get_artifacts()
    assert len(artifacts) == 1  # Should have our test.py file
    
    # Verify the file artifact was saved correctly
    test_file_path = temp_dir / "test.py"
    assert test_file_path.exists()
    assert test_file_path.read_bytes() == test_file_content

    # Test dumping chat state
    dump = chat.model_dump()
    
    # Validate dump structure
    assert "current_node_id" in dump
    assert "nodes" in dump
    assert "agent" in dump
    assert "artifact_handlers" in dump

