from abc import ABC
from pydantic import Field, model_validator
from typing import Any, Optional
from moatless.completion.base import BaseCompletionModel
from moatless.component import MoatlessComponent


class CompletionModelMixin(MoatlessComponent, ABC):
    """Mixin to provide completion model functionality to actions that need it"""

    completion_model: Optional[BaseCompletionModel] = Field(
        default=None,
        description="Completion model to be used for generating completions",
    )

    def _initialize_completion_model(self):
        """Override this method to customize completion model initialization"""
        pass

    @model_validator(mode="after")
    def initialize_completion_model(self):
        """Automatically initialize the completion model if set"""
        if self.completion_model:
            self._initialize_completion_model()
        return self

    def model_completion_dump(self, dump: dict[str, Any]) -> dict[str, Any]:
        if self.completion_model:
            dump["completion_model"] = self.completion_model.model_dump()
        return dump

    @classmethod
    def model_completion_validate(cls, obj: dict[str, Any]) -> dict[str, Any]:
        if "completion_model" in obj:
            obj["completion_model"] = BaseCompletionModel.model_validate(obj["completion_model"])
        return obj
