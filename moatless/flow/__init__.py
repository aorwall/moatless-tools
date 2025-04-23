"""Flow module for managing agentic flows and search trees.

This module provides the core functionality for managing different types of agentic flows:
- Simple flows that execute a single action sequence
- Looping flows that iterate until a condition is met
- Search tree flows that explore multiple action paths

The flow system is built around the concept of nodes that represent states in the
action sequence. Each node can be expanded to create child nodes, forming a tree
of possible action sequences.

Example:
    ```python
    from moatless.flow import AgenticFlow, SimpleFlow, AgenticLoop, SearchTree

    # Create a simple flow
    flow = SimpleFlow.create(
        message="Do something",
        agent=my_agent,
        project_id="my_project",
        trajectory_id="my_trajectory"
    )

    # Run the flow
    await flow.run()
    ```
"""

from moatless.flow.flow import AgenticFlow
from moatless.flow.oneoff import OneOffFlow
from moatless.flow.loop import AgenticLoop
from moatless.flow.search_tree import SearchTree
from moatless.flow.schema import FlowStatus
from moatless.flow.manager import FlowManager
from moatless.flow.run_flow import run_flow

__all__ = [
    # Core flow classes
    "AgenticFlow",
    "OneOffFlow",
    "AgenticLoop",
    "SearchTree",
    # Flow management
    "FlowManager",
    "run_flow",
    # Types and schemas
    "FlowStatus",
]
