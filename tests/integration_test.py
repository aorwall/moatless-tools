import os

import litellm
import pytest
from dotenv import load_dotenv

from moatless import AgenticLoop
from moatless.benchmark.swebench import load_instance, create_workspace
from moatless.benchmark.utils import trace_metadata
from moatless.transitions import search_transitions, code_transitions

load_dotenv()
trajectory_dir = os.getenv("TRAJECTORY_DIR", "/tmp/trajectories")
prompt_log_dir = os.getenv("PROMPT_LOG_DIR", "/tmp/prompt_logs")

global_params = {
    "model": "azure/gpt-4o",
    "temperature": 0.2,
    "max_tokens": 2000,
    "max_prompt_file_tokens": 8000,
}

pytest.mark.llm_integration = pytest.mark.skipif(
    "not config.getoption('--run-llm-integration')",
    reason="need --run-llm-integration option to run tests that call LLMs"
)


@pytest.mark.llm_integration
def test_django_16379():
    instance = load_instance("django__django-16379")
    workspace = create_workspace(instance)

    trajectory_path = f"{trajectory_dir}/test_django_16379_search_for_small_change.json"
    prompt_log_path = f"{prompt_log_dir}/test_django_16379_search_for_small_change"

    metadata = trace_metadata(
        instance_id=instance["instance_id"],
        session_id="integration_test",
        trace_name="search",
    )

    litellm.success_callback = ["langfuse"]
    litellm.failure_callback = ["langfuse"]

    search_loop = AgenticLoop(
        search_transitions(global_params=global_params),
        workspace=workspace,
        trajectory_path=trajectory_path,
        prompt_log_dir=prompt_log_path,
        metadata=metadata
    )

    response = search_loop.run(message=instance["problem_statement"])
    print(response)
    print(workspace.file_context.create_prompt())
    assert workspace.file_context.has_span("django/core/cache/backends/filebased.py", "FileBasedCache.has_key")

    code_loop = AgenticLoop(
        code_transitions(global_params=global_params),
        workspace=workspace,
        trajectory_path=trajectory_path,
        prompt_log_dir=prompt_log_path,
        metadata=metadata
    )

    response = code_loop.run(message=instance["problem_statement"])
    print(response)
    print(workspace.file_context.create_prompt())
    assert workspace.file_context.has_span("django/core/cache/backends/filebased.py", "FileBasedCache.has_key")
