import argparse
import logging
import os
from typing import Optional, List

from dotenv import load_dotenv

from moatless.edit import ClarifyCodeChange
from moatless.benchmark.evaluation import create_evaluation_name, Evaluation
from moatless.edit.edit import EditCode
from moatless.edit.expand import ExpandContext
from moatless.edit.plan import PlanToCode
from moatless.find.decide import DecideRelevance
from moatless.find.identify import IdentifyCode
from moatless.find.search import SearchCode
from moatless.settings import Settings
from moatless.transition_rules import TransitionRule, TransitionRules
from moatless.state import Finished, Rejected
from moatless.transitions import (
    search_and_code_transitions,
    search_transitions,
    code_transitions,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run evaluation for Moatless")
    parser.add_argument("--model", default="gpt-4o", help="Model to use for evaluation")
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="Temperature for model sampling"
    )
    parser.add_argument(
        "--mode",
        default="search",
        choices=["search", "identify", "search_and_identify", "code"],
        help="Evaluation mode",
    )
    parser.add_argument(
        "--resolved_by", type=int, default=None, help="Resolved by parameter"
    )
    parser.add_argument(
        "--instance_ids", nargs="*", default=[], help="List of instance IDs"
    )
    parser.add_argument("--split", default="verified", help="Data split to use")
    parser.add_argument(
        "--evaluation_dir", default=None, help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--max_workers", type=int, default=1, help="Maximum number of worker processes"
    )
    parser.add_argument(
        "--previous_trajectory_dir",
        default=None,
        help="Directory containing previous trajectory data",
    )
    parser.add_argument(
        "--evaluation_name", default=None, help="Custom name for the evaluation"
    )
    return parser.parse_args()

search_model = "openrouter/anthropic/claude-3.5-sonnet"
plan_model = "claude-3-5-sonnet-20240620" # "openrouter/anthropic/claude-3.5-sonnet"
edit_model = "azure/gpt-4o"

DEFAULT_STATE_PARAMS = {
    SearchCode: {
        "model": "claude-3-5-sonnet-20240620",
        "temperature": 0.2,
        "provide_initial_context": True,
        "max_search_results": 75,
        "initial_context_tokens": 6000,
        "initial_search_results": 100,
        "initial_context_spans_per_file": 5,
    },
    IdentifyCode: {"model": "azure/gpt-4o", "temperature": 0.2, "expand_context": True},
    DecideRelevance: {
        "model": "azure/gpt-4o",
        "temperature": 0.2,
        "finish_after_relevant_count": 1,
    },
    PlanToCode: {
        "model": plan_model,
        "temperature": 0.2,
        "max_tokens_in_edit_prompt": 750,
        "write_code_suggestions": False,
        "finish_on_review": True,
    },
    ExpandContext: {
        "expand_to_max_tokens": 8000
    },
    ClarifyCodeChange: {
        "model": "azure/gpt-4o",
        "temperature": 0.0,
        "max_tokens_in_edit_prompt": 750,
    },
    EditCode: {
        "model": edit_model,
        "temperature": 0.0,
        "chain_of_thought": False,
        "show_file_context": False,
        "max_prompt_file_tokens": 8000,
    }
}


def evaluate(
    model: str,
    temperature: float,
    split: str,
    mode: str | None = None,
    previous_trajectory_dir: Optional[str] = None,
    evaluation_dir: Optional[str] = None,
    resolved_by: Optional[int] = None,
    instance_ids: Optional[List[str]] = None,
    max_workers: int = 1,
    evaluation_name: Optional[str] = None,
):
    global_params = {
        "model": model,
        "temperature": temperature,
        "max_tokens": 2000,
        "max_prompt_file_tokens": 12000,
    }
    state_params = DEFAULT_STATE_PARAMS
    reset_from_state = None

    if mode == "search":
        transitions = TransitionRules(
            global_params=global_params,
            state_params=state_params,
            initial_state=SearchCode,
            transition_rules=[
                TransitionRule(source=SearchCode, dest=Finished, trigger="did_search"),
                TransitionRule(source=SearchCode, dest=Finished, trigger="finish"),
            ],
        )
    elif mode == "identify":
        if previous_trajectory_dir is None:
            raise ValueError(
                "Previous trajectory directory must be provided for identify mode"
            )
        reset_from_state = IdentifyCode.name
        transitions = TransitionRules(
            global_params=global_params,
            state_params=state_params,
            initial_state=SearchCode,
            transition_rules=[
                TransitionRule(
                    source=SearchCode, dest=IdentifyCode, trigger="did_search"
                ),
                TransitionRule(source=IdentifyCode, dest=SearchCode, trigger="search"),
                TransitionRule(source=IdentifyCode, dest=Finished, trigger="finish"),
            ],
        )
    elif mode == "search_and_identify":
        reset_from_state = DecideRelevance.name
        state_params[SearchCode]["model"] = (
            "claude-3-5-sonnet-20240620"  # FIXME: Make configurable
        )
        transitions = search_transitions(
            global_params=global_params,
            state_params=state_params,
        )
    elif mode == "code":
        if previous_trajectory_dir:
            reset_from_state = PlanToCode.name
            transitions = search_and_code_transitions(
                global_params=global_params,
                state_params=state_params,
            )
        else:
            transitions = code_transitions(
                global_params=global_params,
                state_params=state_params,
            )
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    if evaluation_name is None:
        evaluation_name = create_evaluation_name(mode, global_params["model"])

    evaluation = Evaluation(
        transitions=transitions,
        evaluations_dir=evaluation_dir or os.path.join(os.getenv("MOATLESS_DIR"), "evaluations"),
        previous_trajectory_dir=previous_trajectory_dir,
        evaluation_name=evaluation_name,
        retry_state=reset_from_state,
        max_file_context_tokens=16000,
        report_mode=mode,
        num_workers=max_workers,
    )

    evaluation.run_evaluation(
        resolved_by=resolved_by, instance_ids=instance_ids, split=split
    )


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("moatless").setLevel(logging.INFO)

    load_dotenv()

    Settings.include_completions_in_trajectories = True

    args = parse_args()

    if not args.evaluation_dir and not os.getenv("MOATLESS_DIR"):
        raise ValueError("Evaluation directory ot the MOATLESS_DIR env var must be provided")

    evaluate(
        mode=args.mode,
        model=args.model,
        temperature=args.temperature,
        split=args.split,
        resolved_by=args.resolved_by,
        evaluation_dir=args.evaluation_dir,
        previous_trajectory_dir=args.previous_trajectory_dir,
        instance_ids=args.instance_ids if args.instance_ids else None,
        max_workers=args.max_workers,
        evaluation_name=args.evaluation_name,
    )
