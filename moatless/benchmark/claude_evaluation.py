import json
import logging
from typing import Optional

import instructor

from moatless import Transitions
from moatless.benchmark.evaluation import create_evaluation_name, Evaluation
from moatless.edit.edit import EditCode
from moatless.edit.plan import PlanToCode
from moatless.find.decide import DecideRelevance
from moatless.find.identify import IdentifyCode
from moatless.find.search_v2 import SearchCode
from moatless.loop import Transition
from moatless.state import Finished, Rejected
from moatless.transitions import (
    search_and_code_transitions,
    search_transitions,
    code_transitions,
)

# model = "claude-3-5-sonnet-20240620"

# model = "gpt-4o-2024-05-13"
model = "azure/gpt-4o"

# model = "openrouter/anthropic/claude-3.5-sonnet"

global_params = {
    "model": model,
    "temperature": 0.2,
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

index_store_dir = f"/home/albert/20240522-voyage-code-2"
repo_base_dir = "/tmp/repos"
evaluations_dir = "/home/albert/repos/albert/moatless/evaluations"

search_and_code = search_and_code_transitions(
    global_params=global_params, state_params=state_params
)

identified_spans_but_failed_implementation = [
    "django__django-11583",
    "django__django-11179",
    "django__django-12286",
    "django__django-12700",
    "django__django-12708",
    "django__django-13315",
    "django__django-13933",
    "django__django-14382",
    "django__django-14608",
    "django__django-14787",
    "django__django-14999",
    "django__django-15347",
    "django__django-15789",
    "django__django-16041",
    "django__django-16046",
    "django__django-16595",
    "matplotlib__matplotlib-26020",
    "matplotlib__matplotlib-24149",
    "mwaskom__seaborn-3190",
    "psf__requests-3362",
    "pytest-dev__pytest-5692",
    "scikit-learn__scikit-learn-11281",
    "django__django-2708",
    "scikit-learn__scikit-learn-13241",
    "scikit-learn__scikit-learn-13779",
    "scikit-learn__scikit-learn-14894",
    "scikit-learn__scikit-learn-15535",
    "scikit-learn__scikit-learn-25570",
    "sympy__sympy-18621",
    "sympy__sympy-23117",
    "sympy__sympy-22714",
    "sympy__sympy-24213",
]

coding_test_set = [
    "django__django-11848",
    "django__django-12308",
    "django__django-12497",
    "django__django-13551",
    "django__django-13660",
    "django__django-14238",
    "django__django-14411",
    "django__django-14787",
    "django__django-16041",
    "django__django-17051",
    "matplotlib__matplotlib-24149",
    "mwaskom__seaborn-3190",
    "psf__requests-1963",
    "pylint-dev__pylint-6506",
    "pylint-dev__pylint-7993",
    "pytest-dev__pytest-7432",
    "scikit-learn__scikit-learn-13142",
    "scikit-learn__scikit-learn-25570",
    "sphinx-doc__sphinx-7975",
    "sympy__sympy-12481",
    "sympy__sympy-14396",
    "sympy__sympy-14817",
    "sympy__sympy-15609",
    "sympy__sympy-16988",
    "sympy__sympy-18189",
    "sympy__sympy-18532",
    "sympy__sympy-21847",
    "sympy__sympy-22005",
    "sympy__sympy-22714",
    "sympy__sympy-24066",
]

search_and_identify_set = [
    "matplotlib__matplotlib-25442",
    "matplotlib__matplotlib-23562",
    "pytest-dev__pytest-11148",
    "sphinx-doc__sphinx-8721",
    "sphinx-doc__sphinx-10325",
    "scikit-learn__scikit-learn-15535",
    "scikit-learn__scikit-learn-11281",
    "astropy__astropy-6938",
    "sympy__sympy-17022",
    "sympy__sympy-17139",
    "sympy__sympy-13031",
    "django__django-15814",
    "django__django-15498",
    "django__django-12125",
    "django__django-13964",
    "django__django-11964",
    "django__django-14580",
    "django__django-17087",
]


def run_evaluation():
    max_file_context_lines = 1000

    transitions = search_and_code_transitions(
        state_params={
            PlanToCode: {
                "max_prompt_file_tokens": 16000,
                "max_tokens_in_edit_prompt": 500,
                "max_file_context_lines": max_file_context_lines,
            }
        },
    )


def evaluate_search():
    transitions = Transitions(
        global_params=global_params,
        state_params={
            SearchCode: {"max_search_results": 50, "provide_initial_context": True},
        },
        initial_state=SearchCode,
        transitions=[
            Transition(source=SearchCode, dest=Finished, trigger="did_search"),
            Transition(source=SearchCode, dest=Finished, trigger="finish"),
        ],
    )

    evaluation_name = create_evaluation_name(model, "search")

    evaluation = Evaluation(
        transitions=transitions,
        evaluations_dir=evaluations_dir + "/search",
        evaluation_name=evaluation_name,
        index_store_dir=index_store_dir,
        repo_base_dir=repo_base_dir,
        max_file_context_tokens=16000,
        litellm_callback="langfuse",
        detailed_report=True,
    )

    evaluation.run_evaluation_with_moatless_dataset(use_test_subset=True)


def evaluate_search_and_identify(
    resolved_by: Optional[int] = 4,
    previous_trajectory_dir: Optional[str] = None,
    instance_ids: Optional[list] = None,
):
    transitions = search_transitions(
        global_params=global_params,
        state_params=state_params,
    )

    evaluation_name = create_evaluation_name("search_and_identify_3", model)
    # evaluation_name = "20240624_search_and_identify_claude-3-5-sonnet-20240620"

    evaluation = Evaluation(
        transitions=transitions,
        evaluations_dir=evaluations_dir + "/search_and_identify",
        evaluation_name=evaluation_name,
        index_store_dir=index_store_dir,
        repo_base_dir=repo_base_dir,
        previous_trajectory_dir=previous_trajectory_dir,
        max_file_context_tokens=16000,
        litellm_callback="langfuse",
        detailed_report=True,
    )

    evaluation.run_evaluation_with_moatless_dataset(
        resolved_by=resolved_by, instance_ids=instance_ids
    )


def evaluate_search_and_code(
    resolved_by: Optional[int],
    previous_trajectory_dir: Optional[str] = None,
    retry_state: Optional[str] = None,
    instance_ids: Optional[list] = None,
):
    evaluation_name = create_evaluation_name("search_and_code", model)
    # evaluation_name = "20240624_search_and_code_2_claude-3-5-sonnet-20240620"
    # evaluation_name = "20240623_moatless_claude-3.5-sonnet"

    evaluation = Evaluation(
        transitions=search_and_code,
        evaluations_dir=evaluations_dir + "/search_and_code",
        evaluation_name=evaluation_name,
        index_store_dir=index_store_dir,
        repo_base_dir=repo_base_dir,
        previous_trajectory_dir=previous_trajectory_dir,
        retry_state=retry_state,
        max_file_context_tokens=16000,
        num_workers=3,
        litellm_callback="langfuse",
        detailed_report=True,
    )

    evaluation.run_evaluation_with_moatless_dataset(
        resolved_by=resolved_by,
        instance_ids=instance_ids,
    )


def evaluate_coding():
    evaluation_name = create_evaluation_name("coding", model)
    # evaluation_name = "20240623_coding_2_claude-3.5-sonnet"

    evaluation = Evaluation(
        transitions=code_transitions(
            global_params=global_params, state_params=state_params
        ),
        use_expected_file_context=True,
        evaluations_dir=evaluations_dir + "/coding",
        evaluation_name=evaluation_name,
        index_store_dir=index_store_dir,
        repo_base_dir=repo_base_dir,
        max_file_context_tokens=16000,
        litellm_callback="langfuse",
        detailed_report=True,
    )

    df = evaluation.run_evaluation_with_moatless_dataset(instance_ids=coding_test_set)


def evaluate_plan(previous_trajectory_dir: Optional[str] = None):
    transitions = Transitions(
        global_params=global_params,
        state_params={
            SearchCode: {
                "provide_initial_context": True,
                "max_search_results": 75,
                "initial_context_tokens": 6000,
                "initial_search_results": 100,
                "initial_context_spans_per_file": 5,
            },
            PlanToCode: {
                "max_prompt_file_tokens": 16000,
                "max_tokens_in_edit_prompt": 750,
                "expand_context_with_related_spans": False,
            },
        },
        initial_state=SearchCode,
        transitions=[
            Transition(source=SearchCode, dest=IdentifyCode, trigger="did_search"),
            Transition(source=IdentifyCode, dest=SearchCode, trigger="search"),
            Transition(source=IdentifyCode, dest=DecideRelevance, trigger="finish"),
            Transition(source=DecideRelevance, dest=SearchCode, trigger="search"),
            Transition(
                source=DecideRelevance,
                dest=PlanToCode,
                trigger="finish",
                exclude_fields={"message"},
            ),
            Transition(source=PlanToCode, dest=Finished, trigger="edit_code"),
            Transition(source=PlanToCode, dest=Rejected, trigger="finish"),
            Transition(source=PlanToCode, dest=Rejected, trigger="reject"),
        ],
    )

    evaluation_name = create_evaluation_name("search_and_plan_2", model)

    evaluation = Evaluation(
        transitions=transitions,
        evaluations_dir=evaluations_dir + "/search_and_plan",
        evaluation_name=evaluation_name,
        index_store_dir=index_store_dir,
        repo_base_dir=repo_base_dir,
        previous_trajectory_dir=previous_trajectory_dir,
        retry_state="PlanToCode",
        max_file_context_tokens=16000,
        litellm_callback="langfuse",
        detailed_report=True,
    )

    df = evaluation.run_evaluation_with_moatless_dataset(
        instance_ids=identified_spans_but_failed_implementation
    )

    # print out instance id and if planned
    for instance_id in df.index:
        print(df.loc[instance_id, "instance_id"], df.loc[instance_id, "planned"])


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("Evaluator").setLevel(logging.INFO)

    # evaluate_coding()
    # evaluate_search_and_identify()
    evaluate_search_and_code(
        1,
        "/home/albert/repos/albert/moatless/evaluations/20240623_moatless_claude-3.5-sonnet/trajs",
        retry_state="PlanToCode",
    )
    # evaluate_search_and_code()
    # evaluate_search_and_code(
    #    # "/home/albert/repos/albert/moatless/evaluations/search_and_code/20240622_search_and_code_6_claude-3.5-sonnet/trajs"
    # )
