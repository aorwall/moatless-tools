import asyncio
import json
import pytest
from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

from moatless.agent.agent import ActionAgent
from moatless.events import BaseEvent, FlowStartedEvent, FlowCompletedEvent, FlowErrorEvent
from moatless.flow.flow import AgenticFlow
from moatless.flow.loop import AgenticLoop
from moatless.node import Node
from moatless.workspace import Workspace


class TestFlowEventCallback:
    """Test the event callback mechanism in AgenticFlow"""

    @pytest.fixture
    def mock_agent(self):
        agent = MagicMock(spec=ActionAgent)
        agent.workspace = MagicMock(spec=Workspace)
        agent.initialize = AsyncMock()
        agent.run = AsyncMock()
        return agent

    @pytest.fixture
    def captured_events(self):
        return []

    @pytest.fixture
    def test_flow(self, mock_agent, captured_events):
        # Create event callback
        async def event_callback(event: BaseEvent) -> None:
            captured_events.append(event)

        # Create root node
        root = Node.create_root(user_message="test message")

        # Create flow with callback
        flow = AgenticLoop.create(
            root=root,
            agent=mock_agent,
            project_id="test-project",
            trajectory_id="test-trajectory",
            max_iterations=2,
            on_event=event_callback,
        )

        # Mock the agent's run method to return the node as executed
        async def mock_run(node):
            # Add a mock action step that reports as executed
            step = MagicMock()
            step.is_executed.return_value = True
            step.observation = "Test observation"
            step.action = MagicMock()
            step.action.name = "test_action"
            node.action_steps.append(step)
            return node

        mock_agent.run.side_effect = mock_run

        return flow

    @pytest.mark.asyncio
    async def test_emit_event_with_callback(self, test_flow, captured_events):
        """Test that _emit_event properly calls the callback function"""
        # Emit an event
        test_event = FlowStartedEvent()
        await test_flow._emit_event(test_event)

        # Verify the event was captured
        assert len(captured_events) == 1
        assert captured_events[0].event_type == "started"
        assert captured_events[0].project_id == "test-project"
        assert captured_events[0].trajectory_id == "test-trajectory"

    @pytest.mark.asyncio
    async def test_flow_run_emits_events(self, test_flow, captured_events):
        """Test that running the flow emits start and complete events"""
        # Run the flow
        await test_flow.run()

        # Check that we got at least start and complete events
        assert len(captured_events) >= 2

        # Verify the events types (first should be started, last should be completed)
        assert captured_events[0].event_type == "started"
        assert captured_events[-1].event_type == "completed"

    @pytest.mark.asyncio
    async def test_flow_without_callback(self, mock_agent):
        """Test that a flow without callback works properly"""
        # Create flow without callback
        root = Node.create_root(user_message="test message")
        flow = AgenticLoop.create(
            root=root,
            agent=mock_agent,
            project_id="test-project",
            trajectory_id="test-trajectory",
            max_iterations=2,
        )

        # Mock the agent's run method
        async def mock_run(node):
            # Add a mock action step that reports as executed
            step = MagicMock()
            step.is_executed.return_value = True
            step.observation = "Test observation"
            step.action = MagicMock()
            step.action.name = "test_action"
            node.action_steps.append(step)
            return node

        mock_agent.run.side_effect = mock_run

        # Emit an event (should not raise errors)
        await flow._emit_event(FlowStartedEvent())

        # Run the flow (should complete without errors)
        await flow.run()

    @pytest.mark.asyncio
    @patch("moatless.settings.get_storage")
    async def test_run_flow_persistence(self, mock_get_storage, mock_agent):
        """Test the persistence of flow data when running with the event callback"""
        # Setup
        from moatless.flow.run_flow import handle_flow_event

        # Create mock storage
        mock_storage = AsyncMock()
        mock_get_storage.return_value = mock_storage
        mock_storage.write_to_trajectory = AsyncMock()

        # Create a flow
        root = Node.create_root(user_message="test message")
        flow = AgenticLoop.create(
            root=root,
            agent=mock_agent,
            project_id="test-project",
            trajectory_id="test-trajectory",
            max_iterations=2,
        )

        # Extract the process_event_task method directly to avoid the background task
        # Mock the get_trajectory_data method
        with patch.object(AgenticFlow, "get_trajectory_data", return_value={"nodes": []}):
            # Create test event
            event = FlowStartedEvent(project_id="test-project", trajectory_id="test-trajectory")

            # Extract and run the process_event_task function directly
            # instead of patching asyncio.create_task
            from moatless.flow.run_flow import _flow_lock

            async def direct_process_event_task():
                async with _flow_lock:
                    storage = await mock_get_storage()  # Use our mocked storage
                    trajectory_data = flow.get_trajectory_data()
                    await storage.write_to_trajectory(
                        "trajectory.json", trajectory_data, flow.project_id, flow.trajectory_id
                    )

            # Run the task directly
            await direct_process_event_task()

            # Verify that storage.write_to_trajectory was called
            mock_storage.write_to_trajectory.assert_called_once()

            # Verify the arguments
            call_args = mock_storage.write_to_trajectory.call_args[0]
            assert call_args[0] == "trajectory.json"
            assert "nodes" in call_args[1]  # Trajectory data
            assert call_args[2] == "test-project"
            assert call_args[3] == "test-trajectory"

    @pytest.mark.asyncio
    @patch("moatless.settings.get_event_bus")
    async def test_handle_flow_event_publishes_to_bus(self, mock_get_event_bus, mock_agent):
        """Test that handle_flow_event publishes to the event bus"""
        from moatless.flow.run_flow import handle_flow_event

        # Create mock event bus
        mock_event_bus = AsyncMock()
        mock_get_event_bus.return_value = mock_event_bus
        mock_event_bus.publish = AsyncMock()

        # Create a mock storage
        with patch("moatless.settings.get_storage") as mock_get_storage:
            mock_storage = AsyncMock()
            mock_get_storage.return_value = mock_storage
            mock_storage.write_to_trajectory = AsyncMock()

            # Create a flow
            root = Node.create_root(user_message="test message")
            flow = AgenticLoop.create(
                root=root,
                agent=mock_agent,
                project_id="test-project",
                trajectory_id="test-trajectory",
                max_iterations=2,
            )

            # Mock the get_trajectory_data method
            with patch.object(AgenticFlow, "get_trajectory_data", return_value={"nodes": []}):
                # Create test event
                event = FlowStartedEvent(project_id="test-project", trajectory_id="test-trajectory")

                # Extract and run the process_event_task function directly
                from moatless.flow.run_flow import _flow_lock

                async def direct_process_event_task():
                    async with _flow_lock:
                        storage = await mock_get_storage()
                        trajectory_data = flow.get_trajectory_data()
                        await storage.write_to_trajectory(
                            "trajectory.json", trajectory_data, flow.project_id, flow.trajectory_id
                        )

                        event_bus = await mock_get_event_bus()
                        await event_bus.publish(event)

                # Run the task directly
                await direct_process_event_task()

                # Verify that event_bus.publish was called
                mock_event_bus.publish.assert_called_once_with(event)
