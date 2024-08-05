import os
from datetime import datetime

import litellm
import pytest
from dotenv import load_dotenv

from moatless import AgenticLoop
from moatless.benchmark.swebench import load_instance, create_workspace
from moatless.benchmark.utils import trace_metadata
from moatless.edit import EditCode
from moatless.transitions import (
    search_transitions,
    code_transitions,
    search_and_code_transitions,
)

load_dotenv()
moatless_dir = os.getenv("MOATLESS_DIR", "/tmp/moatless")

global_params = {
    "model": "gpt-4o-mini-2024-07-18",  # "azure/gpt-4o",
    "temperature": 0.5,
    "max_tokens": 2000,
    "max_prompt_file_tokens": 8000,
}

pytest.mark.llm_integration = pytest.mark.skipif(
    "not config.getoption('--run-llm-integration')",
    reason="need --run-llm-integration option to run tests that call LLMs",
)


@pytest.mark.llm_integration
def test_run_and_reload_django_16379():
    instance = load_instance("django__django-16379")
    workspace = create_workspace(instance)

    datestr = datetime.now().strftime("%Y%m%d-%H%M%S")
    dir = f"{moatless_dir}/{datestr}_test_django_16379_search_for_small_change"
    trajectory_path = f"{dir}/trajectory.json"

    loop = AgenticLoop(
        search_and_code_transitions(global_params=global_params),
        workspace=workspace,
        trajectory_path=trajectory_path,
        prompt_log_dir=dir,
    )

    response = loop.run(message=instance["problem_statement"])
    print("Response")
    print(response)

    diff = loop.workspace.file_repo.diff()
    print("Diff")
    print(diff)

    assert workspace.file_context.has_span(
        "django/core/cache/backends/filebased.py", "FileBasedCache.has_key"
    )

    saved_loop = AgenticLoop.from_trajectory_file(trajectory_path=trajectory_path)

    saved_response = saved_loop.run(message=instance["problem_statement"])

    assert saved_response.status == response.status
    assert saved_response.message == response.message

    assert saved_loop.workspace.file_context.has_span(
        "django/core/cache/backends/filebased.py", "FileBasedCache.has_key"
    )

    assert saved_loop.workspace.file_repo.diff() == diff


@pytest.mark.llm_integration
def test_different_edit_models():
    instance = load_instance("django__django-16379")
    workspace = create_workspace(instance)

    datestr = datetime.now().strftime("%Y%m%d-%H%M%S")
    dir = f"{moatless_dir}/{datestr}_test_django_16379_search_for_small_change"
    trajectory_path = f"{dir}/trajectory.json"

    metadata = trace_metadata(
        instance_id=instance["instance_id"],
        session_id="integration_test",
        trace_name="search",
    )

    state_params = {
        EditCode: {
            "model": "gpt-4o-2024-05-13",
        }
    }

    loop = AgenticLoop(
        search_and_code_transitions(
            global_params=global_params, state_params=state_params
        ),
        workspace=workspace,
        trajectory_path=trajectory_path,
        prompt_log_dir=dir,
        metadata=metadata,
    )

    response = loop.run(message=instance["problem_statement"])

    print("Response")
    print(response)

    diff = loop.workspace.file_repo.diff()
    print("Diff")
    print(diff)

    assert workspace.file_context.has_span(
        "django/core/cache/backends/filebased.py", "FileBasedCache.has_key"
    )

    first_commit = loop.workspace.file_repo._current_commit
    assert first_commit != loop.workspace.file_repo._initial_commit

    # Reverts to PlanToCode state and set LLM to GPT-4o-mini in the EditCode state
    response_mini = loop.retry_from_transition(
        transition_id=4,  # PlanToCode
        state_params={
            EditCode: {
                "model": "gpt-4o-mini-2024-07-18",
            }
        },
    )

    print("Response")
    print(response_mini)

    diff = loop.workspace.file_repo.diff()
    print("Diff")
    print(diff)

    assert loop.workspace.file_repo._current_commit != first_commit
