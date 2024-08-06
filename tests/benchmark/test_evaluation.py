import os
from datetime import datetime

import pytest
from dotenv import load_dotenv

from moatless.benchmark.evaluation import Evaluation
from moatless.edit import PlanToCode, EditCode
from moatless.find import SearchCode, IdentifyCode, DecideRelevance
from moatless.transitions import search_and_code_transitions

load_dotenv()
moatless_dir = os.getenv("MOATLESS_DIR", "/tmp/moatless")
index_store_dir = os.getenv("INDEX_STORE_DIR", "/tmp/index_store")
repo_dir = os.getenv("REPO_DIR", "/tmp/repo")

global_params = {
    "model": "gpt-4o-mini-2024-07-18",  # "azure/gpt-4o",
    "temperature": 0.5,
    "max_tokens": 2000,
    "max_prompt_file_tokens": 8000,
}

state_params = {
    SearchCode: {
        "provide_initial_context": True,
        "max_search_results": 75,
        "initial_context_tokens": 6000,
        "initial_search_results": 100,
        "initial_context_spans_per_file": 5,
    },
    IdentifyCode: {"expand_context": True},
    DecideRelevance: {
        "finish_after_relevant_count": 1,
    },
    PlanToCode: {
        "max_tokens_in_edit_prompt": 750,
        "expand_context_with_related_spans": False,
        "finish_on_review": True,
    },
    EditCode: {
        "chain_of_thought": False,
        "show_file_context": False,
        "max_prompt_file_tokens": 8000,
    },
}

search_and_code = search_and_code_transitions(
    global_params=global_params, state_params=state_params
)

pytest.mark.llm_integration = pytest.mark.skipif(
    "not config.getoption('--run-llm-integration')",
    reason="need --run-llm-integration option to run tests that call LLMs",
)


@pytest.mark.llm_integration
def test_run_single_evaluation_mcts():
    datestr = datetime.now().strftime("%Y%m%d-%H%M%S")
    dir = f"{moatless_dir}/eval_test"
    evaluation_name = f"{datestr}_mcts"

    evaluation = Evaluation(
        transitions=search_and_code,
        evaluations_dir=dir,
        evaluation_name=evaluation_name,
        index_store_dir=index_store_dir,
        repo_base_dir=repo_dir,
        max_file_context_tokens=16000,
        num_workers=1,
        detailed_report=True,
    )

    result = evaluation.run_single_instance("django__django-16379")

    assert result["instance_id"] == "django__django-16379"
    assert result["status"] == "edited"
    assert result["edited"]
    assert result["identified"]
    assert result["found_in_search"]
    assert result["file_identified"]
