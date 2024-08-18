import os
from datetime import datetime

import pytest
from dotenv import load_dotenv

from moatless.benchmark.evaluation import Evaluation
from moatless.edit import PlanToCode, EditCode
from moatless.find.search import SearchCode
from moatless.find.identify import IdentifyCode
from moatless.find.decide import DecideRelevance
from moatless.transitions import search_and_code_transitions

import pickle
import pytest
from moatless.transitions import search_and_code_transitions
from moatless.edit import PlanToCode, EditCode
from moatless.find.search import SearchCode
from moatless.find.identify import IdentifyCode
from moatless.find.decide import DecideRelevance

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

def test_pickle_search_and_code_transitions():
    global_params = {
        "model": "gpt-4o-mini-2024-07-18",
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

    transitions = search_and_code_transitions(
        global_params=global_params, state_params=state_params
    )

    evaluation = Evaluation(
        transitions=transitions,
        evaluations_dir="/tmp",
        evaluation_name="test",
        max_file_context_tokens=16000,
        reset_from_state=PlanToCode.name,
        num_workers=4,
        detailed_report=True,
        report_modes=["search_and_identify"],
    )

    # Attempt to pickle the transitions object
    pickled_evaluation = pickle.dumps(evaluation)

    # Attempt to unpickle the transitions object
    unpickled_evaluation = pickle.loads(pickled_evaluation)

    # Check if the unpickled object is an instance of TransitionRules
    assert isinstance(unpickled_evaluation, type(evaluation))



@pytest.mark.llm_integration
def test_run_single_evaluation_mcts():
    datestr = datetime.now().strftime("%Y%m%d-%H%M%S")
    dir = f"{moatless_dir}/eval_test"
    evaluation_name = f"{datestr}_dsadsdads"

    evaluation = Evaluation(
        transitions=search_and_code,
        evaluations_dir=dir,
        evaluation_name=evaluation_name,
        index_store_dir=index_store_dir,
        repo_base_dir=repo_dir,
        max_file_context_tokens=16000,
        num_workers=4,
        detailed_report=True,
        report_modes=["search_and_identify"],
    )

    result = evaluation.run_evaluation()

    assert result["instance_id"] == "django__django-16379"
    assert result["status"] == "edited"
    assert result["edited"]
    assert result["identified"]
    assert result["found_in_search"]
    assert result["file_identified"]
