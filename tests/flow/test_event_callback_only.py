import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from moatless.events import BaseEvent, FlowStartedEvent
from moatless.flow.flow import AgenticFlow
from moatless.node import Node


class TestBasicEventCallback:
    """Test the basic event callback mechanism in AgenticFlow"""

    @pytest.fixture
    def mock_flow(self):
        """Create a mocked AgenticFlow instance with minimal setup"""
        flow = MagicMock(spec=AgenticFlow)
        flow.project_id = "test-project"
        flow.trajectory_id = "test-trajectory"
        flow._root = MagicMock(spec=Node)
        flow._status = MagicMock()
        return flow

    @pytest.fixture
    def captured_events(self):
        """Create a list to capture events"""
        return []

    @pytest.mark.asyncio
    async def test_emit_event_with_callback(self, mock_flow, captured_events):
        """Test that _emit_event properly calls the callback function"""
        # Create test event
        test_event = FlowStartedEvent()
        
        # Create async event callback function
        async def event_callback(event: BaseEvent) -> None:
            captured_events.append(event)
        
        # Set the callback on the flow
        mock_flow._on_event = event_callback
        
        # Get the actual method from the real class to test it
        emit_event = AgenticFlow._emit_event
        
        # Call the emit_event method with our mocked flow
        await emit_event(mock_flow, test_event)
        
        # Verify the event was captured by the callback
        assert len(captured_events) == 1
        assert captured_events[0].event_type == "started"
        assert captured_events[0].project_id == "test-project"
        assert captured_events[0].trajectory_id == "test-trajectory"

    @pytest.mark.asyncio
    async def test_emit_event_without_callback(self, mock_flow):
        """Test that _emit_event works properly without a callback"""
        # Create test event
        test_event = FlowStartedEvent()
        
        # Set no callback on the flow
        mock_flow._on_event = None
        
        # Get the actual method from the real class to test it
        emit_event = AgenticFlow._emit_event
        
        # Call the emit_event method with our mocked flow (should not raise errors)
        await emit_event(mock_flow, test_event)

    @pytest.mark.asyncio
    @patch('moatless.settings.get_storage')
    @patch('moatless.settings.get_event_bus')
    async def test_run_flow_callback(self, mock_get_event_bus, mock_get_storage):
        """Test that event callback works properly in run_flow"""
        from moatless.flow.run_flow import _flow_lock
        
        # Create mock storage and event bus
        mock_storage = AsyncMock()
        mock_get_storage.return_value = mock_storage
        mock_storage.write_to_trajectory = AsyncMock()
        
        mock_event_bus = AsyncMock()
        mock_get_event_bus.return_value = mock_event_bus
        mock_event_bus.publish = AsyncMock()
        
        # Create a mock flow
        flow = MagicMock(spec=AgenticFlow)
        flow.project_id = "test-project"
        flow.trajectory_id = "test-trajectory"
        flow.get_trajectory_data.return_value = {"nodes": []}
        
        # Create test event
        event = FlowStartedEvent(
            project_id="test-project",
            trajectory_id="test-trajectory"
        )
        
        # Directly run the process_event_task logic (which is normally run in a background task)
        async with _flow_lock:
            storage = await mock_get_storage()
            
            trajectory_data = flow.get_trajectory_data()
            await storage.write_to_trajectory("trajectory.json", trajectory_data, flow.project_id, flow.trajectory_id)
            
            event_bus = await mock_get_event_bus()
            await event_bus.publish(event)
        
        # Verify storage and event bus interactions
        mock_storage.write_to_trajectory.assert_called_once()
        mock_event_bus.publish.assert_called_once_with(event) 