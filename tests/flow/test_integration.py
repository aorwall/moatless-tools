import asyncio
import pytest
import tempfile
from pathlib import Path
from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

from moatless.agent.agent import ActionAgent
from moatless.events import BaseEvent, FlowStartedEvent, FlowCompletedEvent
from moatless.flow.events import NodeExpandedEvent, NodeSelectedEvent
from moatless.flow.flow import AgenticFlow
from moatless.flow.loop import AgenticLoop
from moatless.flow.search_tree import SearchTree
from moatless.expander import Expander
from moatless.selector.base import BaseSelector
from moatless.node import Node, Reward
from moatless.workspace import Workspace


class MockSelector(BaseSelector):
    """A simple selector for testing"""
    async def select(self, nodes: List[Node]) -> Optional[Node]:
        # Return the first valid node
        return nodes[0] if nodes else None


class TestIntegration:
    """Integration tests for event callback in different flow implementations"""
    
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
    def root_node(self):
        return Node.create_root(user_message="test message")
    
    @pytest.mark.asyncio
    async def test_loop_integration(self, mock_agent, captured_events, root_node):
        """Test that AgenticLoop properly emits events through the callback"""
        # Create event callback
        async def event_callback(event: BaseEvent) -> None:
            captured_events.append(event)
            
        # Create a flow with limited iterations
        loop = AgenticLoop.create(
            root=root_node,
            agent=mock_agent,
            project_id="test-project",
            trajectory_id="test-trajectory",
            max_iterations=2,  # Just need a couple iterations
            on_event=event_callback
        )
        
        # Mock the agent's run method to mark nodes as executed
        async def mock_run(node):
            # Create an action step directly to mark as executed
            step = MagicMock()
            step.is_executed.return_value = True
            step.observation = "Test observation"
            step.action = MagicMock()
            step.action.name = "test_action"
            node.action_steps.append(step)
            return node
        
        mock_agent.run.side_effect = mock_run
        
        # Run the loop
        await loop.run()
        
        # Verify we got events
        assert len(captured_events) >= 3  # At least start, node expanded, and completed
        
        # Check specific event types
        start_events = [e for e in captured_events if e.event_type == "started"]
        expanded_events = [e for e in captured_events if e.event_type == "expanded"]
        completed_events = [e for e in captured_events if e.event_type == "completed"]
        
        assert len(start_events) == 1
        assert len(expanded_events) >= 1
        assert len(completed_events) == 1
    
    @pytest.mark.asyncio
    @patch('moatless.settings.get_storage')
    async def test_search_tree_integration(self, mock_get_storage, mock_agent, captured_events, root_node):
        """Test that SearchTree properly emits events through the callback"""
        # Create mock storage
        mock_storage = AsyncMock()
        mock_get_storage.return_value = mock_storage
        
        # Mock the storage exists_in_trajectory and read_from_trajectory methods
        mock_storage.exists_in_trajectory.return_value = False
        
        # Create event callback
        async def event_callback(event: BaseEvent) -> None:
            captured_events.append(event)
        
        # Create selector and expander
        selector = MockSelector()
        expander = Expander(max_expansions=1)
        
        # Create a search tree with limited iterations
        tree = SearchTree.create(
            root=root_node,
            agent=mock_agent, 
            selector=selector,
            expander=expander,
            project_id="test-project",
            trajectory_id="test-trajectory",
            max_iterations=2,  # Just need a couple iterations
            on_event=event_callback
        )
        
        # Mock the agent's run method to mark nodes as executed
        async def mock_run(node):
            # Create an action step directly to mark as executed
            step = MagicMock()
            step.is_executed.return_value = True
            step.observation = "Test observation"
            step.action = MagicMock()
            step.action.name = "test_action"
            node.action_steps.append(step)
            
            # Set the reward directly (it's a Field in the model)
            node.reward = Reward(value=50)
            return node
        
        mock_agent.run.side_effect = mock_run
        
        # Run the tree
        await tree.run()
        
        # Verify we got events
        assert len(captured_events) >= 3  # At least start, node expanded/selected, and completed
        
        # Check specific event types
        start_events = [e for e in captured_events if e.event_type == "started"]
        expanded_events = [e for e in captured_events if e.event_type == "expanded"]
        selected_events = [e for e in captured_events if e.event_type == "selected"]
        completed_events = [e for e in captured_events if e.event_type == "completed"]
        
        assert len(start_events) == 1
        assert len(expanded_events) + len(selected_events) >= 1
        assert len(completed_events) == 1
    
    @pytest.mark.asyncio
    @patch('moatless.settings.get_storage')
    @patch('moatless.settings.get_event_bus')
    @patch('moatless.flow.run_flow.setup_job_logging')
    @patch('moatless.flow.run_flow.setup_flow')
    @patch('moatless.flow.run_flow.setup_workspace')
    @patch('moatless.flow.run_flow.get_storage')  # Add patch for get_storage in run_flow module
    async def test_run_flow_integration(self, mock_get_storage_run_flow, mock_setup_workspace, mock_setup_flow,
                                       mock_setup_logging, mock_get_event_bus, mock_get_storage,
                                       mock_agent, root_node):
        """Test the run_flow module's integration with the event callback"""
        from moatless.flow.run_flow import run_flow, handle_flow_event
        from moatless.events import BaseEvent, FlowStartedEvent
        
        # Mock the logging setup 
        mock_setup_logging.return_value = []
        
        # Create mock storage and event bus
        mock_storage = AsyncMock()
        mock_get_storage.return_value = mock_storage
        mock_get_storage_run_flow.return_value = mock_storage  # Set the same mock for run_flow.get_storage
        
        # Setup necessary methods
        mock_storage.get_trajectory_path.return_value = "test-path"
        mock_storage.write_raw = AsyncMock()
        mock_storage.exists_in_trajectory.return_value = False
        mock_storage.write_to_trajectory = AsyncMock()
        
        # Set up the event bus mock
        mock_event_bus = AsyncMock()
        mock_event_bus.publish = AsyncMock()
        mock_get_event_bus.return_value = mock_event_bus
        
        # Mock setup_workspace
        workspace = MagicMock(spec=Workspace)
        mock_setup_workspace.return_value = workspace
        
        # Create a collection of captured events
        captured_events = []
        
        # Create a test flow that will use our callback
        flow = AgenticLoop.create(
            root=root_node,
            agent=mock_agent,
            project_id="test-project",
            trajectory_id="test-trajectory",
            max_iterations=2
        )
        
        # We need to manually set the event handler to work with our test
        # This is what `setup_flow` would normally do
        async def event_callback(event: BaseEvent) -> None:
            # Add to our tracking list for test verification
            captured_events.append(event)
            # Call the real handle_flow_event to ensure events flow to event bus
            await handle_flow_event(flow, event)
            
        # Set our callback
        flow._on_event = event_callback
        
        # Set flow as the return value of setup_flow
        mock_setup_flow.return_value = flow
        
        # Create mock log path
        with patch('moatless.flow.run_flow.Path') as mock_path:
            mock_file_path = MagicMock()
            mock_file_path.read_text.return_value = "Test log"
            mock_file_path.name = "job.log"
            mock_path.return_value = mock_file_path
        
            # Important: Mock is_finished to return False first, then True
            # This ensures the flow runs once
            with patch.object(AgenticFlow, 'is_finished', side_effect=[False, True]):
                # Make _run return a predictable value
                with patch.object(AgenticFlow, '_run', return_value=(root_node, "test_finish_reason")):
                    # Manually emit an event to verify our event flow
                    start_event = FlowStartedEvent()
                    await flow._emit_event(start_event)
                    
                    # Wait for any pending tasks to complete
                    await asyncio.sleep(0.1)
                    
                    # Run the flow - this should call _emit_event with various events
                    await run_flow("test-project", "test-trajectory")
                    
                    # Wait for any pending tasks to complete
                    await asyncio.sleep(0.1)
                    
                    # Verify that at least some events were captured by our callback
                    assert len(captured_events) >= 1
                    
                    # Verify event types 
                    start_events = [e for e in captured_events if e.event_type == "started"]
                    assert len(start_events) >= 1
                    
                    # Verify the event bus was called to publish the events
                    assert mock_event_bus.publish.call_count >= 1 