import logging
from typing import Optional

from moatless.edit.clarify import ClarifyCodeChange
from moatless.edit.edit import EditCode
from moatless.edit.plan import PlanToCode
from moatless.edit.plan_lines import PlanToCodeWithLines
from moatless.find.identify import IdentifyCode
from moatless.find.decide import DecideRelevance
from moatless.find.search_v2 import SearchCode
from moatless.loop import Transitions, Transition
from moatless.state import Rejected, Finished

CODE_TRANSITIONS = [
    Transition(
        source=PlanToCode,
        dest=EditCode,
        trigger="edit_code",
        required_fields=EditCode.required_fields(),
    ),
    Transition(
        source=PlanToCode,
        dest=ClarifyCodeChange,
        trigger="edit_code",
        required_fields=ClarifyCodeChange.required_fields(),
    ),
    Transition(source=PlanToCode, dest=Finished, trigger="finish"),
    Transition(source=PlanToCode, dest=Rejected, trigger="reject"),
    Transition(
        source=ClarifyCodeChange,
        dest=EditCode,
        trigger="edit_code",
        required_fields=EditCode.required_fields(),
    ),
    Transition(source=ClarifyCodeChange, dest=PlanToCode, trigger="reject"),
    Transition(source=EditCode, dest=PlanToCode, trigger="finish"),
    Transition(source=EditCode, dest=PlanToCode, trigger="reject"),
]


logger = logging.getLogger(__name__)


def code_transitions(
    global_params: Optional[dict] = None,
    state_params: Optional[dict] = None,
    max_prompt_file_tokens: Optional[int] = 16000,
    max_tokens_in_edit_prompt: Optional[int] = 500,
) -> Transitions:
    state_params = state_params or {}
    state_params.setdefault(
        PlanToCode,
        {
            "max_prompt_file_tokens": max_prompt_file_tokens,
            "max_tokens_in_edit_prompt": max_tokens_in_edit_prompt,
        },
    )

    return Transitions(
        global_params=global_params or {},
        state_params=state_params,
        initial_state=PlanToCode,
        transitions=CODE_TRANSITIONS,
    )


def code_transitions_use_line_numbers(
    global_params: Optional[dict] = None, state_params: Optional[dict] = None
) -> Transitions:
    return Transitions(
        global_params=global_params or {},
        state_params=state_params or {},
        initial_state=PlanToCodeWithLines,
        transitions=[
            Transition(
                source=PlanToCodeWithLines,
                dest=EditCode,
                trigger="edit_code",
                required_fields=PlanToCodeWithLines.required_fields(),
            ),
            Transition(source=PlanToCodeWithLines, dest=Finished, trigger="finish"),
            Transition(source=PlanToCodeWithLines, dest=Rejected, trigger="reject"),
            Transition(source=EditCode, dest=PlanToCodeWithLines, trigger="finish"),
            Transition(source=EditCode, dest=PlanToCodeWithLines, trigger="reject"),
        ],
    )


def edit_code_transitions(
    global_params: Optional[dict] = None, state_params: Optional[dict] = None
) -> Transitions:
    return Transitions(
        global_params=global_params or {},
        state_params=state_params or {},
        initial_state=EditCode,
        transitions=[
            Transition(source=EditCode, dest=Finished, trigger="finish"),
            Transition(source=EditCode, dest=Rejected, trigger="reject"),
        ],
    )


def search_transitions(
    model: Optional[str] = None,
    max_prompt_file_tokens: Optional[int] = None,
    max_search_results: Optional[int] = None,
    max_maybe_finish_iterations: int = 5,
    global_params: Optional[dict] = None,
    state_params: Optional[dict] = None,
) -> Transitions:
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

    return Transitions(
        global_params=global_params,
        state_params=state_params,
        initial_state=SearchCode,
        transitions=[
            Transition(source=SearchCode, dest=IdentifyCode, trigger="did_search"),
            Transition(source=SearchCode, dest=Finished, trigger="finish"),
            Transition(source=IdentifyCode, dest=SearchCode, trigger="search"),
            Transition(source=IdentifyCode, dest=DecideRelevance, trigger="finish"),
            Transition(source=DecideRelevance, dest=SearchCode, trigger="search"),
            Transition(source=DecideRelevance, dest=Finished, trigger="finish"),
        ],
    )


def identify_directly_transition(
    model: Optional[str] = None,
    max_prompt_file_tokens: Optional[int] = 30000,
    max_search_results: Optional[int] = 100,
    global_params: Optional[dict] = None,
    state_params: Optional[dict] = None,
) -> Transitions:
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

    return Transitions(
        global_params=global_params,
        state_params=state_params,
        initial_state=IdentifyCode,
        transitions=[
            Transition(source=IdentifyCode, dest=Finished, trigger="search"),
            Transition(source=IdentifyCode, dest=Finished, trigger="finish"),
        ],
    )


def search_and_code_transitions(
    max_tokens_in_edit_prompt: Optional[int] = 500,
    global_params: Optional[dict] = None,
    state_params: Optional[dict] = None,
) -> Transitions:
    state_params = state_params or {}
    if max_tokens_in_edit_prompt is not None:
        state_params.setdefault(
            PlanToCode, {"max_tokens_in_edit_prompt": max_tokens_in_edit_prompt}
        )
    return Transitions(
        global_params=global_params,
        state_params=state_params,
        initial_state=SearchCode,
        transitions=[
            Transition(source=SearchCode, dest=IdentifyCode, trigger="did_search"),
            Transition(source=SearchCode, dest=PlanToCode, trigger="finish"),
            Transition(source=IdentifyCode, dest=SearchCode, trigger="search"),
            Transition(source=IdentifyCode, dest=DecideRelevance, trigger="finish"),
            Transition(source=DecideRelevance, dest=SearchCode, trigger="search"),
            Transition(
                source=DecideRelevance,
                dest=PlanToCode,
                trigger="finish",
                exclude_fields={"message"},
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
) -> Transitions:
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

    return Transitions(
        global_params=global_params,
        state_params=state_params or {},
        initial_state=IdentifyCode,
        transitions=[
            Transition(source=IdentifyCode, dest=SearchCode, trigger="search"),
            Transition(source=IdentifyCode, dest=PlanToCode, trigger="finish"),
        ]
        + CODE_TRANSITIONS,
    )
