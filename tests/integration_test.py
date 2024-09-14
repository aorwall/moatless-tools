import os
from datetime import datetime

import litellm
import pytest
from dotenv import load_dotenv

from moatless import AgenticLoop
from moatless.benchmark.swebench import load_instance, create_workspace
from moatless.benchmark.utils import trace_metadata, get_moatless_instance
from moatless.edit import EditCode, PlanToCode
from moatless.edit.expand import ExpandContext
from moatless.find import SearchCode, IdentifyCode, DecideRelevance
from moatless.state import Finished, Pending
from moatless.transition_rules import TransitionRule, TransitionRules
from moatless.transitions import (
    search_transitions,
    code_transitions,
    search_and_code_transitions,
    edit_code_transitions,
)
from moatless.utils.llm_utils import response_format_by_model

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


@pytest.mark.parametrize(
    "model",
    [
        "claude-3-5-sonnet-20240620",
        "gpt-4o-mini",
        "gpt-4o",
        "deepseek/deepseek-coder",
        "openrouter/anthropic/claude-3.5-sonnet",
    ],
)
@pytest.mark.llm_integration
def test_simple_search(model):
    global_params = {"model": model, "temperature": 0.0}

    transitions = TransitionRules(
        global_params=global_params,
        initial_state=SearchCode,
        transition_rules=[
            TransitionRule(source=Pending, dest=SearchCode, trigger="init"),
            TransitionRule(source=SearchCode, dest=Finished, trigger="did_search"),
            TransitionRule(source=SearchCode, dest=Finished, trigger="finish"),
        ],
    )

    instance = load_instance("django__django-16379")
    workspace = create_workspace(instance)
    loop = AgenticLoop(
        transitions,
        workspace=workspace,
        prompt_log_dir=dir,
    )

    response = loop.run(message=instance["problem_statement"])
    assert response.status == "finished"


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


@pytest.mark.llm_integration
def test_plan_and_code_add_method():
    model = "deepseek/deepseek-coder"
    litellm.drop_params = True

    instance_id = "scikit-learn__scikit-learn-13439"

    global_params = {
        "model": model,
        "temperature": 0.5,
        "max_tokens": 2000,
        "max_prompt_file_tokens": 8000,
    }

    instance = load_instance(instance_id)
    workspace = create_workspace(instance)

    workspace.file_context.add_spans_to_context("sklearn/pipeline.py", ["Pipeline"])

    datestr = datetime.now().strftime("%Y%m%d-%H%M%S")
    dir = f"{moatless_dir}/{datestr}_test_scikit_learn_13439__{model.replace('/', '_')}"
    trajectory_path = f"{dir}/trajectory.json"

    loop = AgenticLoop(
        code_transitions(global_params=global_params),
        initial_message=instance["problem_statement"],
        workspace=workspace,
        trajectory_path=trajectory_path,
        prompt_log_dir=dir,
    )

    response = loop.run()
    print("Response")
    print(response)
    print("Response Output")
    print(response.output)

    assert response.status == "finished"

    diff = loop.workspace.file_repo.diff()
    print("Diff")
    print(diff)
    assert diff


@pytest.mark.llm_integration
def test_deepseek_coder_django_12286_edit_code():
    model = "deepseek/deepseek-coder"
    instance_id = "django__django-12286"
    litellm.drop_params = True

    global_params = {
        "model": model,
        "temperature": 0.5,
        "max_tokens": 2000,
        "max_prompt_file_tokens": 8000,
    }

    state_params = {
        EditCode: {
            "model": model,
            "instructions": "Modify the check_language_settings_consistent function to split the LANGUAGE_CODE into its base language and check if the base language is in the available_tags.",
            "file_path": "django/core/checks/translation.py",
            "span_id": "check_language_settings_consistent",
            "start_line": 55,
            "end_line": 61,
        }
    }

    instance = load_instance(instance_id)
    workspace = create_workspace(instance)

    workspace.file_context.add_spans_to_context(
        "django/core/checks/translation.py",
        ["imports", "check_language_settings_consistent"],
    )

    datestr = datetime.now().strftime("%Y%m%d-%H%M%S")
    dir = f"{moatless_dir}/{datestr}_test_django_12286_deepseek_coder"
    trajectory_path = f"{dir}/trajectory.json"

    loop = AgenticLoop(
        edit_code_transitions(global_params=global_params, state_params=state_params),
        initial_message=instance["problem_statement"],
        workspace=workspace,
        trajectory_path=trajectory_path,
        prompt_log_dir=dir,
    )

    response = loop.run()
    print("Response")
    print(response)
    print("Response Output")
    print(response.output)

    assert response.status == "finished"

    diff = loop.workspace.file_repo.diff()
    print("Diff")
    print(diff)
    assert diff
