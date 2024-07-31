import logging

from pydantic import BaseModel, Field, PrivateAttr, model_validator
from typing import Any, Type, Optional
from moatless.state import AgenticState, get_state_class


logger = logging.getLogger(__name__)


class TransitionRule(BaseModel):
    trigger: str = Field(
        ...,
        description="The trigger from the current state that causes the transition to fire.",
    )
    source: type[AgenticState] = Field(
        ..., description="The source state that the transition rule is defined for."
    )
    dest: type[AgenticState] = Field(
        ...,
        description="The destination state that the transition rule is defined for.",
    )
    required_fields: Optional[set[str]] = Field(
        default=None,
        description="The fields that are required for the transition to fire.",
    )
    excluded_fields: Optional[set[str]] = Field(
        default=None, description="The fields that are excluded from the transition."
    )

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def model_dump(self, **kwargs):
        data = super().model_dump(**kwargs)
        data["source"] = self.source.__name__
        data["dest"] = self.dest.__name__

        if data.get("required_fields"):
            data["required_fields"] = list(data.get("required_fields"))

        if data.get("excluded_fields"):
            data["excluded_fields"] = list(data.get("excluded_fields"))

        return data

    @model_validator(mode="before")
    @classmethod
    def validate_state_classes(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if isinstance(data.get("source"), str):
                data["source"] = get_state_class(data["source"])
            if isinstance(data.get("dest"), str):
                data["dest"] = get_state_class(data["dest"])
        return data


class TransitionRules(BaseModel):
    initial_state: type[AgenticState] = Field(
        ..., description="The initial state of the loop."
    )
    transition_rules: list[TransitionRule] = Field(
        ..., description="The transition rules for the loop."
    )
    global_params: dict[str, Any] = Field(
        default_factory=dict, description="Global parameters used by all transitions."
    )
    state_params: dict[type[AgenticState], dict[str, Any]] = Field(
        default_factory=dict, description="State-specific parameters."
    )

    _source_trigger_index: dict[
        tuple[type[AgenticState], str], list[TransitionRule]
    ] = PrivateAttr(default_factory=dict)

    def __init__(self, **data):
        super().__init__(**data)
        self._build_source_trigger_index()

    def model_dump(self, **kwargs):
        return {
            "initial_state": self.initial_state.__name__,
            "transition_rules": [
                rule.model_dump(**kwargs) for rule in self.transition_rules
            ],
            "global_params": self.global_params,
            "state_params": {k.__name__: v for k, v in self.state_params.items()},
        }

    @model_validator(mode="before")
    @classmethod
    def validate_state_classes(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if isinstance(data.get("initial_state"), str):
                data["initial_state"] = get_state_class(data["initial_state"])

            if "state_params" in data:
                data["state_params"] = {
                    get_state_class(k) if isinstance(k, str) else k: v
                    for k, v in data["state_params"].items()
                }

        return data

    def _build_source_trigger_index(self):
        for rule in self.transition_rules:
            key = (rule.source, rule.trigger)
            if key not in self._source_trigger_index:
                self._source_trigger_index[key] = []
            self._source_trigger_index[key].append(rule)

    def find_transition_rule_by_source_and_trigger(
        self, source: type[AgenticState], trigger: str
    ) -> list[TransitionRule]:
        return self._source_trigger_index.get((source, trigger), [])

    def create_initial_state(self, **data) -> AgenticState:
        params = {}
        params.update(self.global_params)
        params.update(self.state_params.get(self.initial_state, {}))
        params.update(data)
        print(f"initial_state,{params}")
        return self.initial_state(**params)

    def next_state(
        self, source: AgenticState, trigger: str, data: dict[str, Any]
    ) -> AgenticState | None:
        transition_rules = self.find_transition_rule_by_source_and_trigger(
            source.__class__, trigger
        )
        for transition_rule in transition_rules:
            if (
                transition_rule.required_fields
                and not transition_rule.required_fields.issubset(data.keys())
            ):
                logger.info(f"Missing required fields for transition {transition_rule}")
                continue

            params = {}
            params.update(self.global_params)
            params.update(self.state_params.get(transition_rule.dest, {}))

            if transition_rule.excluded_fields:
                data = {
                    k: v
                    for k, v in data.items()
                    if k not in transition_rule.excluded_fields
                }

            params.update(data)
            return transition_rule.dest(**params)
        return None
