import logging
from typing import Optional

from moatless.edit.clarify import ClarifyCodeChange
from moatless.edit.edit import EditCode
from moatless.edit.expand import ExpandContext
from moatless.edit.plan import PlanToCode
from moatless.edit.plan_lines import PlanToCodeWithLines
from moatless.find.decide import DecideRelevance
from moatless.find.identify import IdentifyCode
from moatless.find.search import SearchCode
from moatless.state import Finished, Rejected, Pending
from moatless.transition_rules import TransitionRule, TransitionRules

CODE_TRANSITIONS = [
    TransitionRule(
        source=PlanToCode,
        dest=EditCode,
        trigger="edit_code",
        required_fields=EditCode.required_fields(),
    ),
    TransitionRule(
        source=PlanToCode,
        dest=ClarifyCodeChange,
        trigger="edit_code",
        required_fields=ClarifyCodeChange.required_fields(),
    ),
    TransitionRule(source=PlanToCode, dest=Finished, trigger="finish"),
    TransitionRule(source=PlanToCode, dest=Rejected, trigger="reject"),
    TransitionRule(
        source=ClarifyCodeChange,
        dest=EditCode,
        trigger="edit_code",
        required_fields=EditCode.required_fields(),
    ),
    TransitionRule(source=ClarifyCodeChange, dest=PlanToCode, trigger="reject"),
    TransitionRule(source=EditCode, dest=PlanToCode, trigger="finish"),
    TransitionRule(source=EditCode, dest=PlanToCode, trigger="reject"),
]


logger = logging.getLogger(__name__)


def code_transitions(
    global_params: Optional[dict] = None,
    state_params: Optional[dict] = None,
    max_prompt_file_tokens: Optional[int] = 16000,
    max_tokens_in_edit_prompt: Optional[int] = 500,
) -> TransitionRules:
    state_params = state_params or {}
    state_params.setdefault(
        PlanToCode,
        {
            "max_prompt_file_tokens": max_prompt_file_tokens,
            "max_tokens_in_edit_prompt": max_tokens_in_edit_prompt,
        },
    )

    return TransitionRules(
        global_params=global_params or {},
        state_params=state_params,
        initial_state=PlanToCode,
        transition_rules=CODE_TRANSITIONS,
    )


def code_transitions_use_line_numbers(
    global_params: Optional[dict] = None, state_params: Optional[dict] = None
) -> TransitionRules:
    return TransitionRules(
        global_params=global_params or {},
        state_params=state_params or {},
        initial_state=PlanToCodeWithLines,
        transition_rules=[
            TransitionRule(
                source=PlanToCodeWithLines,
                dest=EditCode,
                trigger="edit_code",
                required_fields=PlanToCodeWithLines.required_fields(),
            ),
            TransitionRule(source=PlanToCodeWithLines, dest=Finished, trigger="finish"),
            TransitionRule(source=PlanToCodeWithLines, dest=Rejected, trigger="reject"),
            TransitionRule(source=EditCode, dest=PlanToCodeWithLines, trigger="finish"),
            TransitionRule(source=EditCode, dest=PlanToCodeWithLines, trigger="reject"),
        ],
    )


def edit_code_transitions(
    global_params: Optional[dict] = None, state_params: Optional[dict] = None
) -> TransitionRules:
    return TransitionRules(
        global_params=global_params or {},
        state_params=state_params or {},
        initial_state=EditCode,
        transition_rules=[
            TransitionRule(source=EditCode, dest=Finished, trigger="finish"),
            TransitionRule(source=EditCode, dest=Rejected, trigger="reject"),
        ],
    )


def search_transitions(
    model: Optional[str] = None,
    max_prompt_file_tokens: Optional[int] = None,
    max_search_results: Optional[int] = None,
    max_maybe_finish_iterations: int = 5,
    global_params: Optional[dict] = None,
    state_params: Optional[dict] = None,
) -> TransitionRules:
    global_params = global_params or {}

    if model is not None:
        global_params["model"] = model

    if state_params is None:
        state_params = {}

    if max_search_results is not None:
        state_params.setdefault(SearchCode, {"max_search_results": max_search_results})

    if max_prompt_file_tokens is not None:
        state_params.setdefault(
            IdentifyCode, {"max_prompt_file_tokens": max_prompt_file_tokens}
        )

    state_params.setdefault(
        DecideRelevance, {"max_iterations": max_maybe_finish_iterations}
    )

    logger.info(state_params)

    return TransitionRules(
        global_params=global_params,
        state_params=state_params,
        initial_state=SearchCode,
        transition_rules=[
            TransitionRule(source=SearchCode, dest=IdentifyCode, trigger="did_search"),
            TransitionRule(source=SearchCode, dest=Finished, trigger="finish"),
            TransitionRule(source=IdentifyCode, dest=SearchCode, trigger="search"),
            TransitionRule(source=IdentifyCode, dest=DecideRelevance, trigger="finish"),
            TransitionRule(source=DecideRelevance, dest=SearchCode, trigger="search"),
            TransitionRule(source=DecideRelevance, dest=Finished, trigger="finish"),
        ],
    )


def identify_directly_transition(
    model: Optional[str] = None,
    max_prompt_file_tokens: Optional[int] = 30000,
    max_search_results: Optional[int] = 100,
    global_params: Optional[dict] = None,
    state_params: Optional[dict] = None,
) -> TransitionRules:
    global_params = global_params or {}

    if model is not None:
        global_params["model"] = model

    if state_params is None:
        state_params = {}

    if max_search_results is not None:
        state_params.setdefault(SearchCode, {"max_search_results": max_search_results})

    if max_prompt_file_tokens is not None:
        state_params.setdefault(
            IdentifyCode, {"max_prompt_file_tokens": max_prompt_file_tokens}
        )

    logger.info(state_params)

    return TransitionRules(
        global_params=global_params,
        state_params=state_params,
        initial_state=IdentifyCode,
        transition_rules=[
            TransitionRule(source=IdentifyCode, dest=Finished, trigger="search"),
            TransitionRule(source=IdentifyCode, dest=Finished, trigger="finish"),
        ],
    )


def search_and_code_transitions(
    global_params: Optional[dict] = None,
    state_params: Optional[dict] = None,
) -> TransitionRules:
    state_params = state_params or {}
    return TransitionRules(
        global_params=global_params,
        state_params=state_params,
        transition_rules=[
            TransitionRule(source=Pending, dest=SearchCode, trigger="init"),
            TransitionRule(source=SearchCode, dest=IdentifyCode, trigger="did_search"),
            TransitionRule(source=SearchCode, dest=PlanToCode, trigger="finish"),
            TransitionRule(source=IdentifyCode, dest=SearchCode, trigger="search"),
            TransitionRule(source=IdentifyCode, dest=DecideRelevance, trigger="finish"),
            TransitionRule(source=DecideRelevance, dest=SearchCode, trigger="search"),
            TransitionRule(source=DecideRelevance, dest=ExpandContext, trigger="finish"),
            TransitionRule(
                source=ExpandContext,
                dest=PlanToCode,
                trigger="finish",
                exclude_fields={"added_spans", "original_tokens","expanded_tokens"},
            ),
        ]
        + CODE_TRANSITIONS,
    )


def identify_and_code_transitions(
    model: Optional[str] = None,
    max_prompt_file_tokens: Optional[int] = 16000,
    max_tokens_in_edit_prompt: Optional[int] = 500,
    max_search_results: Optional[int] = 100,
    global_params: Optional[dict] = None,
    state_params: Optional[dict] = None,
) -> TransitionRules:
    global_params = global_params or {}

    if model is not None:
        global_params["model"] = model

    if state_params is None:
        state_params = {}

    if max_search_results is not None:
        state_params.setdefault(SearchCode, {"max_search_results": max_search_results})

    if max_prompt_file_tokens is not None:
        state_params.setdefault(
            IdentifyCode, {"max_prompt_file_tokens": max_prompt_file_tokens}
        )

    if max_tokens_in_edit_prompt is not None:
        state_params.setdefault(
            PlanToCode,
            {
                "max_prompt_file_tokens": max_prompt_file_tokens,
                "max_tokens_in_edit_prompt": max_tokens_in_edit_prompt,
            },
        )

    return TransitionRules(
        global_params=global_params,
        state_params=state_params or {},
        initial_state=IdentifyCode,
        transition_rules=[
            TransitionRule(source=IdentifyCode, dest=SearchCode, trigger="search"),
            TransitionRule(source=IdentifyCode, dest=PlanToCode, trigger="finish"),
        ]
        + CODE_TRANSITIONS,
    )
