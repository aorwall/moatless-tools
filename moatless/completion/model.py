import json
import logging
from typing import Optional, Any, Union

from pydantic import BaseModel, model_validator, Field

logger = logging.getLogger(__name__)

# Model costs per million tokens
MODEL_COSTS = {
    "claude-3-5-haiku-20241022": {
        "input": 0.80,
        "output": 4.0,
        "cache": 0.08,
        "cached_included": False
    },
    "claude-3-5-sonnet-20241022": {
        "input": 3.0,
        "output": 15.0,
        "cache": 0.30,
        "cached_included": False
    },
    "deepseek/deepseek-chat": {
        "input": 0.14,
        "output": 0.28,
        "cache": 0.014,
        "cached_included": True
    },
    "o1-mini-2024-09-12": {
        "input": 3.0,
        "output": 12.0,
        "cache": 1.5,
        "cached_included": True
    },
    "o1-preview-2024-09-12": {
        "input": 15.0,
        "output": 60.0,
        "cache": 7.5,
        "cached_included": True
    }
}


class Usage(BaseModel):
    version: int = Field(default=2, description="Version of the usage model")
    completion_cost: float = Field(default=0, description="Total cost of the completion in USD")
    completion_tokens: int = Field(default=0, description="Number of tokens in the completion/response")
    prompt_tokens: int = Field(default=0, description="Total number of tokens in the prompt, including both cached and non-cached tokens")
    cache_read_tokens: int = Field(default=0, description="Number of tokens read from cache, included in prompt_tokens")
    cache_write_tokens: int = Field(default=0, description="Number of tokens written to cache, included in prompt_tokens")

    def get_total_prompt_tokens(self, model: str) -> int:
        """Get total prompt tokens based on model's token counting behavior."""
        if model not in MODEL_COSTS:
            return self.prompt_tokens
        
        # All tokens are already included in prompt_tokens
        return self.prompt_tokens

    def get_calculated_cost(self, model: str) -> float:
        """Get the calculated cost based on instance token counts and model."""
        if self.completion_cost > 0:
            return self.completion_cost
        return self.calculate_cost(
            model,
            self.prompt_tokens,
            self.completion_tokens,
            self.cache_read_tokens
        )

    @staticmethod
    def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int, cache_read_tokens: int = 0) -> float:
        """Calculate cost based on token counts and model."""
        if model not in MODEL_COSTS:
            return 0.0
                
        rates = MODEL_COSTS[model]
        non_cached_tokens = prompt_tokens - cache_read_tokens
        input_cost = non_cached_tokens * rates["input"] / 1_000_000
        cache_cost = cache_read_tokens * rates["cache"] / 1_000_000 if cache_read_tokens else 0
        output_cost = completion_tokens * rates["output"] / 1_000_000
        return input_cost + output_cost + cache_cost

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

        logger.debug(f"Usage: {json.dumps(usage, indent=2)}")

        prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens", 0)

        if usage.get("cache_creation_input_tokens"):
            prompt_tokens += usage["cache_creation_input_tokens"]

        completion_tokens = usage.get("completion_tokens") or usage.get(
            "output_tokens", 0
        )

        if usage.get("prompt_cache_hit_tokens"):
            cache_read_tokens = usage["prompt_cache_hit_tokens"]
        elif usage.get("cache_read_input_tokens"):
            cache_read_tokens = usage["cache_read_input_tokens"]
        elif usage.get("prompt_tokens_details") and usage["prompt_tokens_details"].get("cached_tokens"):
            cache_read_tokens = usage["prompt_tokens_details"]["cached_tokens"]
        else:
            cache_read_tokens = 0

        cache_write_tokens = usage.get("cache_creation_input_tokens", 0)

        try:
            import litellm
            cost = litellm.completion_cost(
                completion_response=completion_response, model=model
            )
        except Exception:
            # If cost calculation fails, fall back to calculating it manually
            try:
                from litellm import cost_per_token, NotFoundError

                prompt_cost, completion_cost = cost_per_token(
                    model=model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
                cost = prompt_cost + completion_cost
            except NotFoundError as e:
                logger.debug(
                    f"Failed to calculate cost for completion response: {completion_response}. Error: {e}"
                )
                # Use our own cost calculation if litellm fails
                cost = cls.calculate_cost(model, prompt_tokens, completion_tokens, cache_read_tokens)
            except Exception as e:
                logger.debug(
                    f"Failed to calculate cost for completion response: {completion_response}. Error: {e}"
                )
                cost = 0

        return cls(
            completion_cost=cost,
            completion_tokens=completion_tokens,
            prompt_tokens=prompt_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
        )

    def __add__(self, other: "Usage") -> "Usage":
        # Get completion cost, defaulting to 0 if not available
        other_cost = getattr(other, "completion_cost", 0)
        other_completion = getattr(other, "completion_tokens", 0)
        other_prompt = getattr(other, "prompt_tokens", 0)
        other_cache_read = getattr(other, "cache_read_tokens", 0)
        other_cache_write = getattr(other, "cache_write_tokens", 0)

        return Usage(
            completion_cost=self.completion_cost + other_cost,
            completion_tokens=self.completion_tokens + other_completion,
            prompt_tokens=self.prompt_tokens + other_prompt,
            cache_read_tokens=self.cache_read_tokens + other_cache_read,
            cache_write_tokens=self.cache_write_tokens + other_cache_write,
        )

    def __str__(self) -> str:
        return (
            f"Usage(cost: ${self.completion_cost:.4f}, "
            f"completion tokens: {self.completion_tokens}, "
            f"prompt tokens: {self.prompt_tokens}, "
            f"cache read tokens: {self.cache_read_tokens}, "
            f"cache write tokens: {self.cache_write_tokens})"
        )

    @model_validator(mode="before")
    @classmethod
    def fix_backward_compatibility(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Handle backward compatibility for cached_tokens
            if "cached_tokens" in data:
                data["cache_read_tokens"] = data.pop("cached_tokens")
            
            # Set any null values to 0
            for key, value in data.items():
                if not value:
                    data[key] = 0

            # Handle older versions for cache-excluded models
            version = data.get("version")
            model = data.get("model")
            if (version is None or version == 1) and model in MODEL_COSTS:
                if not MODEL_COSTS[model]["cached_included"]:
                    # For older versions with cache-excluded models like Claude
                    # cache tokens were not included in prompt_tokens, so we need to add them
                    data["prompt_tokens"] = data["prompt_tokens"] + data["cache_read_tokens"]

        return data


class Completion(BaseModel):
    model: str
    input: list[dict] | None = None
    response: dict[str, Any] | None = None
    retries: int | None = None
    usage: Usage | None = None
    flags: list[str] = Field(
        default_factory=list,
        description="List of flags indicating special conditions or states during completion",
    )

    @model_validator(mode="before")
    @classmethod
    def fix_usage(cls, data: Any) -> Any:
        """Allow thoughts to be null."""
        if isinstance(data, dict):
            if "response" in data:
                # Check if we need to reparse usage for version 1 with retries

                if isinstance(data.get("usage"), dict):
                    version = data.get("usage", {}).get("version")
                    retries = data.get("retries", 0)
                elif isinstance(data.get("usage"), Usage):
                    version = data.get("usage").version
                    retries = data.get("retries", 0)
                else:
                    version = 1
                    retries = data.get("retries", 0)

                if version == 1 and retries > 0:
                    data["usage"] = Usage.from_completion_response(data["response"], data["model"])
                elif "usage" not in data:
                    data["usage"] = Usage.from_completion_response(data["response"], data["model"])
        return data

    @classmethod
    def from_llm_completion(
        cls,
        input_messages: list[dict],
        completion_response: Any,
        model: str,
        usage: Usage | None = None,
        retries: int | None = None,
        flags: list[str] | None = None,
    ) -> Optional["Completion"]:
        if completion_response is None:
            raise ValueError("Completion response is None")
        if isinstance(completion_response, BaseModel):
            response = completion_response.model_dump()
        elif isinstance(completion_response, dict):
            response = completion_response
        else:
            logger.error(
                f"Unexpected completion response type: {type(completion_response)}"
            )
            return None

        if not usage:
            usage = Usage.from_completion_response(completion_response, model)

        return cls(
            model=model,
            input=input_messages,
            response=response,
            retries=retries,
            usage=usage,
            flags=flags or [],
        )

