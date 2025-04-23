import pytest
from unittest.mock import AsyncMock, MagicMock

from moatless.actions.action import Action
from moatless.actions.schema import ActionArguments, Observation
from moatless.agent.agent import ActionAgent
from moatless.completion import BaseCompletionModel
from moatless.expander import Expander
from moatless.file_context import FileContext
from moatless.flow.search_tree import SearchTree
from moatless.node import ActionStep, Node, Thoughts
from moatless.selector.simple import SimpleSelector
from moatless.workspace import Workspace
from moatless.actions.finish import Finish, FinishArgs

# Mock Actions for testing purposes
class MockActionArgs(ActionArguments):
    arg1: str = "test_arg"
    thoughts: str = "mock thoughts"

    @property
    def name(self) -> str:
        return "MockAction"

class MockAction(Action):
    args_schema = MockActionArgs

    async def _execute(self, args: MockActionArgs, file_context=None) -> str:
        # This won't be called due to mocking _execute_action_step
        return "Mock action executed"

# Test Fixtures
@pytest.fixture
def mock_workspace():
    # Simple mock workspace sufficient for these tests
    workspace = MagicMock(spec=Workspace)
    workspace.repository = MagicMock()
    # Provide a way to get a file context if needed, though cloning in Node handles it
    workspace.get_file_context = MagicMock(return_value=FileContext(shadow_mode=True))
    return workspace

@pytest.fixture
def simple_agent(mock_workspace):
    # Agent needs at least one action for generation logic to work
    agent = ActionAgent(
        agent_id="test_agent",
        actions=[MockAction()],
        system_prompt="Test system prompt",
        completion_model=MagicMock(spec=BaseCompletionModel)
    )
    # Initialize the agent with the workspace (replaces setter)
    # Use a synchronous approach for fixture setup if initialize is async
    # For simplicity here, directly set the internal workspace if needed by mocks,
    # but agent.run itself needs workspace.
    agent._workspace = mock_workspace

    # Mock the execution part to return a simple observation
    agent._execute_action_step = AsyncMock(
        return_value=Observation(message="Mock observation")  # type: ignore
    )

    # Mock the generation part to create a predictable action
    async def mock_generate_actions(node: Node):
         if not node.action_steps: # Only generate if not already present
            node.action_steps = [ActionStep(action=MockActionArgs())]
            node.assistant_message = "Assistant message for mock action."
            node.thoughts = Thoughts(text="Mock thoughts during generation.")

    agent._generate_actions = AsyncMock(side_effect=mock_generate_actions)
    return agent

@pytest.fixture
def search_tree_factory(simple_agent): # Renamed to indicate it's a factory
    def _create_tree(root: Node, max_iterations=5, max_depth=10, auto_expand_root=False, max_expansions=1):
        selector = SimpleSelector()
        # Allow overriding expander settings per test
        expander = Expander(max_expansions=max_expansions, auto_expand_root=auto_expand_root)
        # Use SearchTree.create() instead of direct constructor
        tree = SearchTree.create(
            root=root, # Pass root during initialization
            selector=selector,
            expander=expander,
            agent=simple_agent,
            value_function=None,      # Skip value function
            feedback_generator=None,  # Skip feedback generator
            discriminator=None,       # Skip discriminator
            max_iterations=max_iterations,
            max_depth=max_depth,
            max_expansions=max_expansions # Pass max_expansions here as well if create uses it
        )
        return tree
    return _create_tree # Return the factory

@pytest.fixture
def root_node():
    node = Node.create_root(user_message="Initial task")
    node.file_context = FileContext(shadow_mode=True)
    node.action_steps = []
    return node


# Test Cases
@pytest.mark.asyncio
async def test_search_tree_single_run(search_tree_factory, root_node, simple_agent):
    # Arrange
    # Expect 1 cycle, need max_iterations = 1 (start) + 1 = 2
    tree = search_tree_factory(root=root_node, max_iterations=2)

    # Act
    final_node, reason = await tree._run()

    # Assert
    assert reason == "max_iterations"
    assert len(root_node.get_all_nodes()) == 2 # Root + 1 child
    child_node = root_node.children[0]
    assert child_node is not None
    assert child_node.node_id == 1

    assert final_node == child_node

    # Verify agent methods were called for the child node
    simple_agent._generate_actions.assert_awaited_once_with(child_node)
    simple_agent._execute_action_step.assert_awaited_once()
    # Check the arguments passed to _execute_action_step
    call_args, _ = simple_agent._execute_action_step.call_args
    assert call_args[0] == child_node # First arg is node
    assert isinstance(call_args[1], ActionStep) # Second arg is action_step
    assert isinstance(call_args[1].action, MockActionArgs)

    # Verify node state
    assert child_node.is_executed()
    assert child_node.action_steps is not None
    assert len(child_node.action_steps) == 1
    assert isinstance(child_node.action_steps[0].action, MockActionArgs)
    assert child_node.action_steps[0].observation is not None
    assert child_node.action_steps[0].observation.message == "Mock observation"


@pytest.mark.asyncio
async def test_search_tree_max_iterations(search_tree_factory, root_node, simple_agent):
    # Arrange
    # Expect 3 cycles, need max_iterations = 1 (start) + 3 = 4
    tree = search_tree_factory(root=root_node, max_iterations=4)

    # Act
    final_node, reason = await tree._run()

    # Assert
    assert reason == "max_iterations"
    # Iteration 1: Select root(0), expand -> node 1, simulate node 1. Nodes: [0, 1]
    # Iteration 2: Select node 1, expand -> node 2, simulate node 2. Nodes: [0, 1, 2]
    # Iteration 3: Select node 2, expand -> node 3, simulate node 3. Nodes: [0, 1, 2, 3]
    # Loop finishes because node count (4) >= max_iterations (4) before starting iteration 4
    assert len(root_node.get_all_nodes()) == 4 # root(0) + node(1) + node(2) + node(3)
    assert final_node is not None
    assert final_node.node_id == 3 # Last created and simulated node

    # Verify agent calls
    assert simple_agent._generate_actions.call_count == 3 # Called for nodes 1, 2, 3
    assert simple_agent._execute_action_step.call_count == 3 # Called for nodes 1, 2, 3


@pytest.mark.asyncio
async def test_search_tree_max_depth(search_tree_factory, root_node, simple_agent):
    # Test that the search stops when max_depth is reached
    tree = search_tree_factory(root=root_node, max_depth=1, max_iterations=5)
    final_node = await tree.run()

    assert final_node.node_id == 1
    assert final_node.terminal is True
    assert len(root_node.get_all_nodes()) == 2 # Root + 1 child


@pytest.mark.asyncio
async def test_search_tree_terminal_action_observation(search_tree_factory, root_node, simple_agent):
    # Arrange
    # Expect 1 cycle before hitting terminal observation, need max_iterations > 1
    tree = search_tree_factory(root=root_node, max_iterations=3) # iterations > 2 to ensure terminal is the limit

    # Store original mocks
    original_generate = simple_agent._generate_actions
    original_execute = simple_agent._execute_action_step

    # Mock execute to set terminal=True in Observation
    simple_agent._execute_action_step = AsyncMock(
        return_value=Observation(message="Mock observation", terminal=True)
    )
    # Ensure generate still produces the mock action
    async def mock_generate_actions(node: Node):
         if not node.action_steps:
            node.action_steps = [ActionStep(action=MockActionArgs())]
            node.assistant_message = "Assistant message for mock action."
            node.thoughts = "Mock thoughts during generation."
    simple_agent._generate_actions = AsyncMock(side_effect=mock_generate_actions)

    try:
        # Act
        final_node, reason = await tree._run()

        # Assert
        # Iteration 1: Select root(0), expand -> node 1, simulate node 1. Observation marks terminal=True. Node 1 becomes terminal.
        # Iteration 2: Select node 1 (terminal). Selector finds no expandable nodes.
        # Loop breaks. is_finished returns 'no_expandable_nodes'.
        assert reason == "no_expandable_nodes" # Should stop due to no expandable nodes
        assert len(root_node.get_all_nodes()) == 2 # Root + 1 child
        child_node = root_node.children[0]
        assert child_node.terminal is True # Terminal state set from observation
        assert child_node.is_executed()
        assert final_node == child_node

        # Verify agent calls
        assert simple_agent._generate_actions.call_count == 1
        simple_agent._generate_actions.assert_awaited_with(child_node)
        assert simple_agent._execute_action_step.call_count == 1
    finally:
        # Restore original mocks
        simple_agent._generate_actions = original_generate
        simple_agent._execute_action_step = original_execute


@pytest.mark.asyncio
async def test_search_tree_finish_action(search_tree_factory, root_node, simple_agent):
    # Arrange
    # Expect 1 cycle before hitting FinishAction, need max_iterations > 1
    tree = search_tree_factory(root=root_node, max_iterations=3) # iterations > 2 to ensure finish action is the limit

    # Store original mocks and agent state
    original_generate = simple_agent._generate_actions
    original_execute = simple_agent._execute_action_step
    original_actions = list(simple_agent.actions)
    original_action_map = dict(simple_agent._action_map)

    # Add FinishAction to the agent's known actions for this test
    finish_action_instance = Finish()
    simple_agent.actions.append(finish_action_instance)
    simple_agent._action_map[FinishArgs] = finish_action_instance

    # Mock generation to produce FinishActionArgs
    async def mock_generate_finish_action(node: Node):
        if not node.action_steps:
            node.action_steps = [ActionStep(action=FinishArgs(finish_reason="Test Finished"))]  # type: ignore
            node.assistant_message = "Assistant chose FinishAction."
            node.thoughts = Thoughts(text="Thoughts for FinishAction.")
    simple_agent._generate_actions = AsyncMock(side_effect=mock_generate_finish_action)

    # Mock execute to handle FinishActionArgs and return terminal observation
    async def mock_execute_finish(node: Node, action_step: ActionStep):
        if isinstance(action_step.action, FinishArgs):
            obs = Observation(message="Finished task", terminal=True)
            return obs
        else:
            return Observation(message="Mock observation")
    simple_agent._execute_action_step = AsyncMock(side_effect=mock_execute_finish)

    try:
        # Act
        final_node, reason = await tree._run()

        # Assert
        # Iteration 1: Select root(0), expand -> node 1, simulate node 1. Action is FinishActionArgs.
        # Mocked execution returns terminal=True observation, setting node.terminal = True.
        # Iteration 2: Select node 1 (terminal). Selector finds no expandable nodes.
        # Loop breaks. is_finished returns 'no_expandable_nodes'.
        assert reason == "no_expandable_nodes" # Should stop due to no expandable nodes
        assert len(root_node.get_all_nodes()) == 2 # Root + 1 child
        child_node = root_node.children[0]

        assert child_node.is_executed()
        assert isinstance(child_node.action_steps[0].action, FinishArgs)
        # Check the specific 'is_finished' method on the node relies on action name
        assert child_node.is_finished() is True # Node method checks action name - Fixed in node.py
        assert child_node.terminal is True # State set by mocked execution observation
        assert final_node == child_node
    finally:
         # Restore original mocks and agent state
        simple_agent._generate_actions = original_generate
        simple_agent._execute_action_step = original_execute
        simple_agent.actions = original_actions
        simple_agent._action_map = original_action_map


@pytest.mark.asyncio
async def test_search_tree_auto_expand_root(search_tree_factory, root_node, simple_agent):
    # Arrange
    # Expect 1 cycle, need max_iterations = 1 (start) + 1 = 2
    tree = search_tree_factory(root=root_node, max_iterations=2, auto_expand_root=True, max_expansions=3)

    # Act
    final_node, reason = await tree._run()

    # Assert
    # Iteration 1:
    # - Select: Selects root node (0) as first unexecuted.
    # - Expand: Expander has auto_expand_root=True, max_expansions=3. It creates nodes 1, 2, 3. Returns node 3.
    # - Simulate: Simulates node 3.
    # Loop finishes due to max_iterations=2 (node count becomes 4).
    assert reason == "max_iterations"
    assert len(root_node.get_all_nodes()) == 4 # Root + 3 children
    assert len(root_node.children) == 3
    node1, node2, node3 = root_node.children

    assert final_node == node3 # Expander returns the last created child, which is then simulated.

    # Verify agent calls only for the last expanded node (node 3)
    assert simple_agent._generate_actions.call_count == 1
    simple_agent._generate_actions.assert_awaited_with(node3)
    assert simple_agent._execute_action_step.call_count == 1
    # Check execute was called on node 3
    call_args, _ = simple_agent._execute_action_step.call_args
    assert call_args[0] == node3

    # Verify states of other children (should not be executed)
    assert not node1.is_executed() # node1 has no action steps
    assert not node1.action_steps
    assert not node2.is_executed() # node2 has no action steps
    assert not node2.action_steps
    assert node3.is_executed() # Node 3 was simulated
    assert node3.action_steps is not None


@pytest.mark.asyncio
async def test_search_tree_run_on_existing_tree(search_tree_factory, root_node, simple_agent):
    # Arrange
    # Create an existing tree structure: root -> child1
    child1 = Node(node_id=1, parent=root_node, max_expansions=1, file_context=root_node.file_context.clone())
    # Mark child1 as executed (as if it ran before and is ready for expansion)
    child1.action_steps = [ActionStep(action=MockActionArgs(), observation=Observation(message="prev obs"))]
    root_node.add_child(child1)

    # Expect 1 cycle, need max_iterations = 2 (start) + 1 = 3
    tree = search_tree_factory(root=root_node, max_iterations=3)

    # Act
    # Iteration 1:
    # - Select: Finds child1 (executed). Calls selector. Selector picks child1.
    # - Expand: Expands child1 -> creates node 2. Returns node 2.
    # - Simulate: Simulates node 2.
    # Loop finishes (max_iterations=3, node count becomes 3).
    final_node, reason = await tree._run()

    # Assert
    assert reason == "max_iterations"
    assert len(root_node.get_all_nodes()) == 3 # Root(0) + Child1(1) + Child1's Child(2)
    child2 = child1.children[0]
    assert child2 is not None
    assert child2.node_id == 2
    assert child2.parent == child1
    assert final_node == child2 # Last simulated node

    # Verify agent methods were called only for the new node (child2)
    assert simple_agent._generate_actions.call_count == 1
    simple_agent._generate_actions.assert_awaited_with(child2)
    assert simple_agent._execute_action_step.call_count == 1
    call_args, _ = simple_agent._execute_action_step.call_args
    assert call_args[0] == child2 # Check execute called on child2

@pytest.mark.asyncio
async def test_search_tree_no_expandable_nodes_finish(search_tree_factory, root_node, simple_agent):
     # Arrange
     # Scenario: Root -> Child1 (Terminal)
     # Expect 1 cycle before hitting terminal, need max_iterations > 1
     tree = search_tree_factory(root=root_node, max_iterations=3) # Iterations > 2 to ensure termination reason

     # Store original mocks
     original_generate = simple_agent._generate_actions
     original_execute = simple_agent._execute_action_step

     # Mock execute to make the first child terminal
     simple_agent._execute_action_step = AsyncMock(
         return_value=Observation(message="Mock observation", terminal=True)
     )
     # Ensure generate still produces the mock action for the first step
     async def mock_generate_actions(node: Node):
         if not node.action_steps:
             node.action_steps = [ActionStep(action=MockActionArgs())]
             node.assistant_message = "Assistant message for mock action."
             node.thoughts = "Mock thoughts during generation."
     simple_agent._generate_actions = AsyncMock(side_effect=mock_generate_actions)

     try:
         # Act
         final_node, reason = await tree._run()

         # Assert
         # Iteration 1: Select root(0), expand -> node 1, simulate node 1. Node 1 becomes terminal.
         # Iteration 2: Select node 1 (terminal). Selector gets empty list.
         # Selection returns node_id=None. Loop breaks.
         # is_finished checks expandable_nodes - finds none. Returns 'no_expandable_nodes'.
         assert reason == "no_expandable_nodes"
         assert len(root_node.get_all_nodes()) == 2 # Root + 1 child
         child_node = root_node.children[0]
         assert child_node.terminal is True
         assert final_node == child_node # Last processed node

         # Verify agent calls (only for the child node)
         assert simple_agent._generate_actions.call_count == 1
         simple_agent._generate_actions.assert_awaited_with(child_node)
         assert simple_agent._execute_action_step.call_count == 1
     finally:
        # Restore original mocks
        simple_agent._generate_actions = original_generate
        simple_agent._execute_action_step = original_execute

@pytest.mark.asyncio
async def test_search_tree_selects_unexecuted_node(search_tree_factory, root_node, simple_agent):
    # Arrange
    # Scenario: Root (executed) -> Child1 (executed), Child2 (NOT executed)
    child1_fc = root_node.file_context.clone()
    child2_fc = root_node.file_context.clone()
    child1 = Node(node_id=1, parent=root_node, max_expansions=1, file_context=child1_fc)
    child2 = Node(node_id=2, parent=root_node, max_expansions=1, file_context=child2_fc)

    # Mark child1 as executed
    child1.action_steps = [ActionStep(action=MockActionArgs(), observation=Observation(message="child1 obs"))]
    # Child2 has NO action_steps yet

    root_node.add_child(child1)
    root_node.add_child(child2)

    # Node 1 executed, Node 2 is not
    child2.action_steps = []

    # Expected cycles = 1 (select Node 2, simulate)
    tree = search_tree_factory(root=root_node, max_iterations=4) # Start with 3 nodes, need 1 more iter

    original_generate = simple_agent._generate_actions
    original_execute = simple_agent._execute_action_step
    try:
        final_node = await tree.run(node_id=1)

        # Assertions
        assert final_node.node_id == 2 # Node 2 was the last processed node
        assert final_node.is_duplicate is True # Marked as duplicate
        assert len(root_node.get_all_nodes()) == 3 # Root + child1 + child2 (no child3 created)

        # Mock calls
        # Root was expanded before run, child1 was executed before run
        # Run(node_id=1) -> _select -> selects unexecuted Node 2
        # _expand(Node 2) -> returns Node 2 (unexecuted)
        # _simulate(Node 2) -> agent.run(Node 2)
        # agent.run(Node 2) -> _generate_actions(Node 2) -> finds duplicate -> returns early
        simple_agent._generate_actions.assert_called_once_with(child2)
        simple_agent._execute_action_step.assert_not_called()
    finally:
        simple_agent._generate_actions = original_generate
        simple_agent._execute_action_step = original_execute