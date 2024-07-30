from moatless import Transitions, AgenticLoop
from moatless.benchmark.swebench import create_workspace


def create_loop(transitions: Transitions, instance: dict):
    workspace = create_workspace(instance)
    trajectory_path = f"search_{instance['name']}.json"
    return AgenticLoop(
        transitions,
        workspace=workspace,
        trajectory_path=trajectory_path,
    )
