import logging
from typing import Any, Optional, Union, List, Set

from instructor import OpenAISchema
from litellm import completion_cost, cost_per_token

from pydantic import BaseModel, Field

from moatless.settings import Settings

logger = logging.getLogger(__name__)


class FileWithSpans(BaseModel):
    file_path: str = Field(
        description="The file path where the relevant code is found."
    )
    span_ids: list[str] = Field(
        default_factory=list,
        description="The span ids of the relevant code in the file",
    )

    def add_span_id(self, span_id):
        if span_id not in self.span_ids:
            self.span_ids.append(span_id)

    def add_span_ids(self, span_ids: list[str]):
        for span_id in span_ids:
            self.add_span_id(span_id)

class Usage(BaseModel):
    completion_cost: float = 0
    completion_tokens: int = 0
    prompt_tokens: int = 0

    @classmethod
    def from_completion_response(
        cls, completion_response: dict | BaseModel, model: str
    ) -> Union["Usage", None]:
        if isinstance(completion_response, BaseModel) and hasattr(
            completion_response, "usage"
        ):
            usage = completion_response.usage.model_dump()
        elif isinstance(completion_response, dict) and "usage" in completion_response:
            usage = completion_response["usage"]
        else:
            logger.warning(
                f"No usage info available in completion response: {completion_response}"
            )
            return None

        prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens", 0)
        completion_tokens = usage.get("completion_tokens") or usage.get(
            "output_tokens", 0
        )

        try:
            cost = completion_cost(completion_response=completion_response, model=model)
        except Exception:
            # If cost calculation fails, fall back to calculating it manually
            try:
                prompt_cost, completion_cost = cost_per_token(
                    model=model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
                cost = prompt_cost + completion_cost
            except Exception:
                logger.warning(
                    f"Failed to calculate cost for completion response: {completion_response}"
                )
                cost = 0

        return cls(
            completion_cost=cost,
            completion_tokens=completion_tokens,
            prompt_tokens=prompt_tokens,
        )


class Completion(BaseModel):
    model: str
    input: list[dict] | None = None
    response: dict[str, Any] | None = None
    usage: Usage | None = None

    @classmethod
    def from_llm_completion(
        cls, input_messages: list[dict], completion_response: Any, model: str
    ) -> Optional["Completion"]:
        if isinstance(completion_response, BaseModel):
            response = completion_response.model_dump()
        elif isinstance(completion_response, dict):
            response = completion_response
        else:
            logger.error(
                f"Unexpected completion response type: {type(completion_response)}"
            )
            return None

        usage = Usage.from_completion_response(completion_response, model)

        return cls(
            model=model,
            input=input_messages,
            response=response,
            usage=usage,
        )



class Response(BaseModel):
    status: str
    message: str
    output: Optional[dict[str, Any]] = None


class VerificationError(BaseModel):
    code: str
    file_path: str
    message: str
    line: int


class CodeChange(BaseModel):
    instructions: str = Field(..., description="Instructions to do the code change.")
    file_path: str = Field(..., description="The file path of the code to be updated.")
    span_id: str = Field(..., description="The span id of the code to be updated.")
