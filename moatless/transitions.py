from typing import Optional

from moatless.edit.clarify import ClarifyCodeChange
from moatless.edit.edit import EditCode
from moatless.edit.plan import PlanToCode
from moatless.find.identify import IdentifyCode
from moatless.find.search import SearchCode
from moatless.loop import Transitions, Transition
from moatless.state import Rejected, Finished

CODE_TRANSITIONS = \
    [Transition(
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
            Transition(source=EditCode, dest=PlanToCode, trigger="reject")
]

def code_transitions(global_params: Optional[dict] = None, state_params: Optional[dict] = None) -> Transitions:
    return Transitions(
        global_params=global_params or {},
        state_params=state_params or {},
        initial_state=PlanToCode,
        transitions=CODE_TRANSITIONS,
    )


def search_transitions(global_params: dict, state_params: Optional[dict] = None) -> Transitions:
    return Transitions(
        global_params=global_params,
        state_params=state_params or {},
        initial_state=SearchCode,
        transitions=[
            Transition(source=SearchCode, dest=IdentifyCode, trigger="did_search"),
            Transition(source=IdentifyCode, dest=SearchCode, trigger="search"),
            Transition(source=IdentifyCode, dest=Finished, trigger="finish"),
        ],
    )


def search_and_code_transitions(global_params: Optional[dict] = None, state_params: Optional[dict] = None) -> Transitions:
    return Transitions(
        global_params=global_params,
        state_params=state_params or {},
        initial_state=SearchCode,
        transitions=[
            Transition(source=SearchCode, dest=IdentifyCode, trigger="did_search"),
            Transition(source=IdentifyCode, dest=SearchCode, trigger="search"),
            Transition(source=IdentifyCode, dest=PlanToCode, trigger="finish"),
        ] + CODE_TRANSITIONS,
    )