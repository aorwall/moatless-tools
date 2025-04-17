import pytest
from unittest.mock import AsyncMock, MagicMock

from moatless.actions.action import Action
from moatless.actions.schema import ActionArguments, Observation
from moatless.agent.agent import ActionAgent
from moatless.expander import Expander
from moatless.file_context import FileContext
from moatless.flow.search_tree import SearchTree
from moatless.node import ActionStep, Node
from moatless.selector.simple import SimpleSelector
from moatless.workspace import Workspace

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

# Attempt to import real FinishAction, otherwise use a mock
try:
    from moatless.actions.finish import FinishAction, FinishActionArgs
except ImportError:
    class FinishActionArgs(ActionArguments):
        message: str = "Task finished"
        status: str = "success"
        thoughts: str = "finish thoughts"

        @property
        def name(self) -> str:
            # Ensure name property exists and matches class name for lookup
            return "FinishAction"

    class FinishAction(Action):
        args_schema = FinishActionArgs
        is_terminal = True

        async def _execute(self, args: FinishActionArgs, file_context=None) -> str:
            # This won't be called due to mocking _execute_action_step
            return f"Finished with status {args.status}: {args.message}"

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
        actions=[MockAction()],
        system_prompt="Test system prompt",
    )
    # Initialize the agent with the workspace (replaces setter)
    # Use a synchronous approach for fixture setup if initialize is async
    # For simplicity here, directly set the internal workspace if needed by mocks,
    # but agent.run itself needs workspace.
    agent._workspace = mock_workspace

    # Mock the execution part to return a simple observation
    agent._execute_action_step = AsyncMock(
        return_value=Observation(message="Mock observation")
    )

    # Mock the generation part to create a predictable action
    async def mock_generate_actions(node: Node):
         if not node.action_steps: # Only generate if not already present
            node.action_steps = [ActionStep(action=MockActionArgs())]
            node.assistant_message = "Assistant message for mock action."
            node.thoughts = "Mock thoughts during generation."

    agent._generate_actions = AsyncMock(side_effect=mock_generate_actions)
    return agent

@pytest.fixture
def search_tree(simple_agent):
    selector = SimpleSelector()
    expander = Expander(max_expansions=1, auto_expand_root=False) # Default auto_expand=False for clarity
    tree = SearchTree(
        selector=selector,
        expander=expander,
        agent=simple_agent,
        value_function=None,        # Skip value function
        feedback_generator=None,  # Skip feedback generator
        discriminator=None,       # Skip discriminator
        max_iterations=5,           # Limit iterations for tests
        max_depth=10,               # Default reasonable depth
    )
    return tree

@pytest.fixture
def root_node():
    # Create root with a real FileContext
    node = Node.create_root(user_message="Initial task")
    node.file_context = FileContext(shadow_mode=True)
    # Root node is considered executed initially
    node.action_steps = [] # Explicitly empty, not None
    return node


# Test Cases
@pytest.mark.asyncio
async def test_search_tree_single_run(search_tree, root_node, simple_agent):
    # Arrange
    search_tree.root = root_node
    search_tree.max_iterations = 1 # Run only one select/expand/simulate cycle

    # Act
    final_node, reason = await search_tree._run()

    # Assert
    assert reason == "max_iterations"
    assert len(root_node.get_all_nodes()) == 2 # Root + 1 child
    child_node = root_node.children[0]
    assert child_node is not None
    assert child_node.node_id == 1
    # In MCTS structure, final_node is the last *processed* node in the loop
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
async def test_search_tree_max_iterations(search_tree, root_node, simple_agent):
    # Arrange
    search_tree.root = root_node
    search_tree.max_iterations = 3

    # Act
    final_node, reason = await search_tree._run()

    # Assert
    assert reason == "max_iterations"
    # Iteration 1: Select root(0), expand -> node 1, simulate node 1. Nodes: [0, 1]
    # Iteration 2: Select node 1, expand -> node 2, simulate node 2. Nodes: [0, 1, 2]
    # Iteration 3: Select node 2, expand -> node 3, simulate node 3. Nodes: [0, 1, 2, 3]
    # Loop stops because node count (4) > max_iterations (3)
    assert len(root_node.get_all_nodes()) == 4 # root(0) + node(1) + node(2) + node(3)
    assert final_node is not None
    assert final_node.node_id == 3 # Last created and simulated node

    # Verify agent calls
    assert simple_agent._generate_actions.call_count == 3
    assert simple_agent._execute_action_step.call_count == 3


@pytest.mark.asyncio
async def test_search_tree_max_depth(search_tree, root_node, simple_agent):
    # Arrange
    search_tree.root = root_node
    search_tree.max_depth = 1 # Only root (depth 0) and its direct children (depth 1)
    search_tree.max_iterations = 5 # Allow enough iterations

    # Act
    final_node, reason = await search_tree._run()

    # Assert
    # Iteration 1: Select root(0), expand -> node 1 (depth 1), simulate node 1.
    # Inside simulate, depth check (1 >= 1) marks node 1 as terminal.
    # Iteration 2: Select node 1. Selector gets empty list as node 1 is terminal.
    # Selection returns node_id=None. Loop breaks. _run checks finished nodes.
    # is_finished checks expandable_nodes - finds none. Returns 'no_expandable_nodes'.
    assert reason == "no_expandable_nodes"
    assert len(root_node.get_all_nodes()) == 2 # Root + 1 child
    child_node = root_node.children[0]
    assert child_node.terminal is True # Max depth should mark as terminal
    assert child_node.is_executed()
    assert final_node == child_node # Last node processed in the loop

    # Verify agent calls (only for the child node)
    assert simple_agent._generate_actions.call_count == 1
    simple_agent._generate_actions.assert_awaited_with(child_node)
    assert simple_agent._execute_action_step.call_count == 1


@pytest.mark.asyncio
async def test_search_tree_terminal_action_observation(search_tree, root_node, simple_agent):
    # Arrange
    search_tree.root = root_node
    search_tree.max_iterations = 5 # Allow enough iterations

    # Mock execute to set terminal=True in Observation
    simple_agent._execute_action_step = AsyncMock(
        return_value=Observation(message="Mock observation", terminal=True)
    )
    # Ensure generate still produces the mock action
    async def mock_generate_actions(node: Node):
         if not node.action_steps:
            node.action_steps = [ActionStep(action=MockActionArgs())]
    simple_agent._generate_actions = AsyncMock(side_effect=mock_generate_actions)


    # Act
    final_node, reason = await search_tree._run()

    # Assert
    # Iteration 1: Select root(0), expand -> node 1, simulate node 1. Observation marks terminal=True. Node 1 becomes terminal.
    # Iteration 2: Select node 1 (terminal). Selector finds no expandable nodes.
    # is_finished returns 'no_expandable_nodes'.
    assert reason == "no_expandable_nodes"
    assert len(root_node.get_all_nodes()) == 2 # Root + 1 child
    child_node = root_node.children[0]
    assert child_node.terminal is True # Terminal state set from observation
    assert child_node.is_executed()
    assert final_node == child_node

    # Verify agent calls
    assert simple_agent._generate_actions.call_count == 1
    simple_agent._generate_actions.assert_awaited_with(child_node)
    assert simple_agent._execute_action_step.call_count == 1


@pytest.mark.asyncio
async def test_search_tree_finish_action(search_tree, root_node, simple_agent):
    # Arrange
    # Add FinishAction to the agent's known actions for generation check (if needed)
    finish_action_instance = FinishAction()
    # Ensure FinishAction's args_schema is correctly mapped
    simple_agent.actions.append(finish_action_instance) # Add instance
    simple_agent._action_map[FinishActionArgs] = finish_action_instance # Manually update map


    # Mock generation to produce FinishActionArgs
    async def mock_generate_finish_action(node: Node):
        if not node.action_steps:
            node.action_steps = [ActionStep(action=FinishActionArgs(message="Test Finished"))]
    simple_agent._generate_actions = AsyncMock(side_effect=mock_generate_finish_action)

    # Mock execute to handle FinishActionArgs and set terminal
    async def mock_execute_finish(node: Node, action_step: ActionStep):
        if isinstance(action_step.action, FinishActionArgs):
            # Simulate that executing FinishAction sets the node to terminal
            obs = Observation(message="Finished task", terminal=True)
            node.terminal = True # Critical step: Execution marks the node terminal
            return obs
        else:
            # Fallback for other actions if any were added
            return Observation(message="Mock observation")
    simple_agent._execute_action_step = AsyncMock(side_effect=mock_execute_finish)


    search_tree.root = root_node
    search_tree.max_iterations = 5

    # Act
    final_node, reason = await search_tree._run()

    # Assert
    # Iteration 1: Select root(0), expand -> node 1, simulate node 1. Action is FinishActionArgs.
    # Execution mock marks node 1 as terminal.
    # Iteration 2: Select node 1 (terminal). Selector finds no expandable nodes.
    # is_finished returns 'no_expandable_nodes'.
    assert reason == "no_expandable_nodes"
    assert len(root_node.get_all_nodes()) == 2 # Root + 1 child
    child_node = root_node.children[0]

    assert child_node.is_executed()
    assert isinstance(child_node.action_steps[0].action, FinishActionArgs)
    # Check the specific 'is_finished' method on the node relies on action name
    assert child_node.is_finished() is True # Node method checks action name
    assert child_node.terminal is True # State set by mocked execution
    assert final_node == child_node


@pytest.mark.asyncio
async def test_search_tree_auto_expand_root(search_tree, root_node, simple_agent):
    # Arrange
    search_tree.expander.auto_expand_root = True
    search_tree.expander.max_expansions = 3
    search_tree.max_iterations = 1 # Only allow one iteration (select/expand/simulate)
    search_tree.root = root_node

    # Act
    final_node, reason = await search_tree._run()

    # Assert
    # Iteration 1:
    # - Select: Selects root node (0) because it's the start.
    # - Expand: Expander has auto_expand_root=True, max_expansions=3. It creates nodes 1, 2, 3. Returns node 3.
    # - Simulate: Simulates node 3.
    # Loop finishes due to max_iterations=1.
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
    assert not node1.is_executed()
    assert not node1.action_steps
    assert not node2.is_executed()
    assert not node2.action_steps
    assert node3.is_executed() # Node 3 was simulated
    assert node3.action_steps is not None


@pytest.mark.asyncio
async def test_search_tree_run_on_existing_tree(search_tree, root_node, simple_agent):
    # Arrange
    # Create an existing tree structure: root -> child1
    child1 = Node(node_id=1, parent=root_node, max_expansions=1, file_context=root_node.file_context.clone())
    # Mark child1 as executed (as if it ran before and is ready for expansion)
    child1.action_steps = [ActionStep(action=MockActionArgs(), observation=Observation(message="prev obs"))]
    root_node.add_child(child1)

    search_tree.root = root_node
    search_tree.max_iterations = 1 # Allow one more iteration

    # Act
    # Iteration 1:
    # - Select: SimpleSelector selects first expandable (child1).
    # - Expand: Expands child1 -> creates node 2. Returns node 2.
    # - Simulate: Simulates node 2.
    # Loop finishes (max_iterations=1).
    final_node, reason = await search_tree._run()

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
async def test_search_tree_run_with_node_id(search_tree, root_node, simple_agent):
    # Arrange
    # Create a tree structure: root -> child1, child2
    child1_fc = root_node.file_context.clone()
    child2_fc = root_node.file_context.clone()
    child1 = Node(node_id=1, parent=root_node, max_expansions=1, file_context=child1_fc)
    child2 = Node(node_id=2, parent=root_node, max_expansions=1, file_context=child2_fc)
    # Mark child1 as executed so it can be expanded from if selected
    child1.action_steps = [ActionStep(action=MockActionArgs(), observation=Observation(message="child1 obs"))]
    root_node.add_child(child1)
    root_node.add_child(child2) # child2 is not executed

    search_tree.root = root_node
    search_tree.max_iterations = 5

    # Act
    # Run starting specifically from child1 (node_id=1)
    final_node, reason = await search_tree._run(node_id=1)

    # Assert
    # The loop should execute for the specified node_id and then break immediately after simulation.
    # - Select: Overridden by node_id=1. Effective node is child1.
    # - Expand: Expands child1 -> creates node 3. Returns node 3.
    # - Simulate: Simulates node 3.
    # Loop condition `if node_id:` then breaks.
    assert reason is None # Loop breaks due to specified node_id, not a standard finish condition
    assert len(root_node.get_all_nodes()) == 4 # Root(0), child1(1), child2(2), child1's child(3)

    child3 = child1.children[0]
    assert child3 is not None
    assert child3.node_id == 3
    assert final_node == child3 # The newly created and simulated node

    # Verify agent calls for the new node 3
    assert simple_agent._generate_actions.call_count == 1
    simple_agent._generate_actions.assert_awaited_with(child3)
    assert simple_agent._execute_action_step.call_count == 1
    call_args, _ = simple_agent._execute_action_step.call_args
    assert call_args[0] == child3 # Check execute called on child3

    # Verify child2 was not touched
    assert not child2.action_steps
    assert not child2.children
    assert not child2.is_executed()


@pytest.mark.asyncio
async def test_search_tree_no_expandable_nodes_finish(search_tree, root_node, simple_agent):
     # Arrange
     # Scenario: Root -> Child1 (Terminal)
     search_tree.root = root_node
     search_tree.max_iterations = 5 # Allow enough iterations

     # Mock execute to make the first child terminal
     async def mock_execute_make_terminal(node: Node, action_step: ActionStep):
         obs = Observation(message="Mock observation", terminal=True)
         node.terminal = True # Mark node as terminal upon execution
         return obs
     simple_agent._execute_action_step = AsyncMock(side_effect=mock_execute_make_terminal)
     # Ensure generate still produces the mock action
     async def mock_generate_actions(node: Node):
         if not node.action_steps:
             node.action_steps = [ActionStep(action=MockActionArgs())]
     simple_agent._generate_actions = AsyncMock(side_effect=mock_generate_actions)

     # Act
     final_node, reason = await search_tree._run()

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


@pytest.mark.asyncio
async def test_search_tree_selects_unexecuted_node(search_tree, root_node, simple_agent):
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

    search_tree.root = root_node
    search_tree.max_iterations = 1 # Allow one select/expand/simulate cycle

    # Act
    # Iteration 1:
    # - Select: Should find Child2 as the first unexecuted node.
    # - Expand: Child2 hasn't been executed, so _expand returns Child2 itself without creating new children.
    # - Simulate: Simulates Child2.
    # Loop finishes (max_iterations=1).
    final_node, reason = await search_tree._run()

    # Assert
    assert reason == "max_iterations"
    assert len(root_node.get_all_nodes()) == 3 # Root, Child1, Child2
    assert final_node == child2 # Node 2 was selected and simulated

    # Verify agent methods were called for Child2
    assert simple_agent._generate_actions.call_count == 1
    simple_agent._generate_actions.assert_awaited_with(child2)
    assert simple_agent._execute_action_step.call_count == 1
    call_args, _ = simple_agent._execute_action_step.call_args
    assert call_args[0] == child2 # Check execute called on child2

    # Verify Child1 was not expanded further
    assert not child1.children 