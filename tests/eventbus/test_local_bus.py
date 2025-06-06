import asyncio
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from moatless.eventbus.base import BaseEventBus
from moatless.eventbus.local_bus import LocalEventBus
from moatless.events import BaseEvent
from moatless.storage.base import BaseStorage
from moatless.storage.file_storage import FileStorage


@pytest.fixture
def temp_dir():
    """Fixture to create a temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    import shutil

    shutil.rmtree(temp_dir)


@pytest.fixture
def file_storage(temp_dir):
    """Fixture to create a FileStorage instance and set it as the singleton."""
    return FileStorage(base_dir=temp_dir)


@pytest.fixture
def local_event_bus(file_storage):
    """Fixture to create a LocalEventBus instance."""
    # First, reset the singleton to ensure we're starting fresh
    BaseEventBus.reset_instance()
    # Create and return the new instance as the singleton
    bus = BaseEventBus.get_instance(eventbus_impl=LocalEventBus, storage=file_storage)
    return bus


@pytest.mark.asyncio
async def test_publish_and_read_events(local_event_bus, file_storage):
    """Test basic publish and read operations."""
    # Create a test event
    event1 = BaseEvent(
        project_id="test-project", trajectory_id="test-trajectory", event_type="test-event-1", data={"key1": "value1"}
    )

    event2 = BaseEvent(
        project_id="test-project", trajectory_id="test-trajectory", event_type="test-event-2", data={"key2": "value2"}
    )

    # Publish the events
    await local_event_bus.publish(event1)
    await local_event_bus.publish(event2)

    # Read the events back as dictionaries
    events = await local_event_bus.read_events("test-project", "test-trajectory")

    # Verify the events
    assert len(events) == 2
    assert events[0].event_type == "test-event-1"
    assert events[0].data["key1"] == "value1"
    assert events[1].event_type == "test-event-2"
    assert events[1].data["key2"] == "value2"

    # Convert to BaseEvent objects for more testing
    events = [BaseEvent.from_dict(d) for d in events]
    assert len(events) == 2
    assert events[0].event_type == "test-event-1"
    assert events[0].data["key1"] == "value1"

    # Verify the events were saved to storage
    key = "projects/test-project/trajs/test-trajectory/events.jsonl"
    assert await file_storage.exists(key)


@pytest.mark.asyncio
async def test_event_subscription(local_event_bus):
    """Test event subscription and notification."""
    # Create a mock subscriber
    mock_subscriber = AsyncMock()

    # Subscribe to events
    await local_event_bus.subscribe(mock_subscriber)

    # Create a test event
    event = BaseEvent(
        project_id="test-project", trajectory_id="test-trajectory", event_type="test-event", data={"key": "value"}
    )

    # Publish the event
    await local_event_bus.publish(event)

    # Verify the subscriber was called
    mock_subscriber.assert_called_once()

    # Verify the event was passed to the subscriber
    args, _ = mock_subscriber.call_args
    passed_event = args[0]
    assert passed_event.event_type == "test-event"
    assert passed_event.data["key"] == "value"

    # Unsubscribe
    await local_event_bus.unsubscribe(mock_subscriber)

    # Reset the mock
    mock_subscriber.reset_mock()

    # Publish another event
    event2 = BaseEvent(project_id="test-project", trajectory_id="test-trajectory", event_type="test-event-2")
    await local_event_bus.publish(event2)

    # Verify the subscriber wasn't called
    mock_subscriber.assert_not_called()
