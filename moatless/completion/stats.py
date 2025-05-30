import json
import logging
import time
from token import OP
from typing import Any, Optional, Union, List

from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = logging.getLogger(__name__)

# Model costs per million tokens
MODEL_COSTS = {
    "claude-3-5-haiku-20241022": {
        "input": 0.80,
        "output": 4.0,
        "cache": 0.08,
        "cached_included": False,
    },
    "claude-3-5-sonnet-20241022": {
        "input": 3.0,
        "output": 15.0,
        "cache": 0.30,
        "cached_included": False,
    },
    "deepseek/deepseek-chat": {
        "input": 0.14,
        "output": 0.28,
        "cache": 0.014,
        "cached_included": True,
    },
    "deepseek/deepseek-reasoner": {
        "input": 0.55,
        "output": 2.19,
        "cache": 0.14,
        "cached_included": True,
    },
    "o1-mini-2024-09-12": {
        "input": 3.0,
        "output": 12.0,
        "cache": 1.5,
        "cached_included": True,
    },
    "o1-preview-2024-09-12": {
        "input": 15.0,
        "output": 60.0,
        "cache": 7.5,
        "cached_included": True,
    },
}


class Usage(BaseModel):
    """Class to track usage statistics for LLM completions."""

    completion_cost: float = Field(default=0, description="Total cost of the completion in USD")
    completion_tokens: int = Field(default=0, description="Number of tokens in the completion/response")
    prompt_tokens: int = Field(
        default=0,
        description="Total number of tokens in the prompt, including both cached and non-cached tokens",
    )
    cache_read_tokens: int = Field(
        default=0,
        description="Number of tokens read from cache, included in prompt_tokens",
    )
    cache_write_tokens: int = Field(
        default=0,
        description="Number of tokens written to cache, included in prompt_tokens",
    )
    reasoning_tokens: int = Field(
        default=0,
        description="Number of tokens in the reasoning content",
    )

    def __add__(self, other: "Usage") -> "Usage":
        """Add two Usage objects together."""
        if not isinstance(other, Usage):
            return self

        return Usage(
            completion_cost=self.completion_cost + other.completion_cost,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
        )

    @staticmethod
    def calculate_cost(
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cache_read_tokens: int = 0,
    ) -> float:
        """Calculate cost based on token counts and model."""
        if model not in MODEL_COSTS:
            return 0.0

        rates = MODEL_COSTS[model]
        non_cached_tokens = prompt_tokens - cache_read_tokens
        input_cost = non_cached_tokens * rates["input"] / 1_000_000
        cache_cost = cache_read_tokens * rates["cache"] / 1_000_000 if cache_read_tokens else 0
        output_cost = completion_tokens * rates["output"] / 1_000_000
        return input_cost + output_cost + cache_cost

    def get_total_prompt_tokens(self, model: str) -> int:
        """Get total prompt tokens based on model's token counting behavior."""
        if model not in MODEL_COSTS:
            return self.prompt_tokens

        # All tokens are already included in prompt_tokens
        return self.prompt_tokens

    def update_from_response(self, completion_response: dict | BaseModel, model: str) -> "Usage":
        """Update this usage data with token counts and other data from a response."""
        if isinstance(completion_response, BaseModel) and hasattr(completion_response, "usage"):
            usage_obj = getattr(completion_response, "usage")
            if isinstance(usage_obj, BaseModel):
                usage = usage_obj.model_dump()
            elif isinstance(usage_obj, dict):
                usage = usage_obj
            else:
                usage = {}
        elif isinstance(completion_response, dict) and "usage" in completion_response:
            usage = completion_response["usage"]
        else:
            logger.warning(f"No usage info available in completion response: {completion_response}")
            return self

        logger.debug(f"Usage: {json.dumps(usage, indent=2)}")

        self.prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens", 0)

        if usage.get("cache_creation_input_tokens"):
            self.prompt_tokens += usage["cache_creation_input_tokens"]

        self.completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens", 0)

        if usage.get("prompt_cache_hit_tokens"):
            self.cache_read_tokens = usage["prompt_cache_hit_tokens"]
        elif usage.get("cache_read_input_tokens"):
            self.cache_read_tokens = usage["cache_read_input_tokens"]
        elif usage.get("prompt_tokens_details") and usage["prompt_tokens_details"].get("cached_tokens"):
            self.cache_read_tokens = usage["prompt_tokens_details"]["cached_tokens"]
        else:
            self.cache_read_tokens = 0
            
        if usage.get("completion_tokens_details") and usage["completion_tokens_details"].get("reasoning_tokens"):
            self.reasoning_tokens = usage["completion_tokens_details"]["reasoning_tokens"]

        self.cache_write_tokens = usage.get("cache_creation_input_tokens", 0)

        try:
            from litellm.cost_calculator import completion_cost

            self.completion_cost = completion_cost(completion_response=completion_response, model=model)
        except Exception:
            # If cost calculation fails, fall back to calculating it manually
            try:
                from litellm.cost_calculator import cost_per_token

                prompt_cost, completion_cost = cost_per_token(
                    model=model,
                    prompt_tokens=self.prompt_tokens,
                    completion_tokens=self.completion_tokens,
                )
                self.completion_cost = prompt_cost + completion_cost
            except Exception as e:
                logger.debug(f"Failed to calculate cost for completion response: {completion_response}. Error: {e}")
                # Use our own cost calculation if litellm fails
                self.completion_cost = self.calculate_cost(
                    model,
                    self.prompt_tokens,
                    self.completion_tokens,
                    self.cache_read_tokens,
                )

        return self

    @classmethod
    def from_completion_response(cls, completion_response: dict | BaseModel, model: str) -> "Usage":
        """Create a new usage instance from a completion response."""
        instance = cls()
        return instance.update_from_response(completion_response, model)

    def __str__(self) -> str:
        return (
            f"Usage(cost: ${self.completion_cost:.4f}, "
            f"completion tokens: {self.completion_tokens}, "
            f"prompt tokens: {self.prompt_tokens}, "
            f"cache read tokens: {self.cache_read_tokens})"
        )


class CompletionAttempt(BaseModel):
    start_time: float = Field(default=0, description="The start time of the completion in milliseconds")
    end_time: float = Field(default=0, description="The end time of the completion in milliseconds")
    usage: Usage = Field(default_factory=Usage, description="Usage statistics for this invocation")
    success: bool = Field(default=True, description="Whether the completion attempt was successful")
    failure_reason: Optional[str] = Field(
        default=None, description="Reason for failure if the attempt was unsuccessful"
    )
    attempt_number: int = Field(default=1, description="The attempt number for this completion")

    @staticmethod
    def _current_time_ms() -> float:
        """Get current time in milliseconds."""
        return time.time() * 1000

    def update_from_response(self, completion_response: dict | BaseModel, model: str) -> "CompletionAttempt":
        """Update this invocation with token counts and other data from a response."""
        self.usage.update_from_response(completion_response, model)
        return self

    @property
    def completion_cost(self) -> float:
        """Get the completion cost from the usage."""
        return self.usage.completion_cost

    @property
    def completion_tokens(self) -> int:
        """Get the completion tokens from the usage."""
        return self.usage.completion_tokens

    @property
    def prompt_tokens(self) -> int:
        """Get the prompt tokens from the usage."""
        return self.usage.prompt_tokens

    @property
    def cache_read_tokens(self) -> int:
        """Get the cache read tokens from the usage."""
        return self.usage.cache_read_tokens

    @property
    def cache_write_tokens(self) -> int:
        """Get the cache write tokens from the usage."""
        return self.usage.cache_write_tokens
    
    @property
    def reasoning_tokens(self) -> int:
        """Get the reasoning tokens from the usage."""
        return self.usage.reasoning_tokens

    def get_total_prompt_tokens(self, model: str) -> int:
        """Get total prompt tokens based on model's token counting behavior."""
        return self.usage.get_total_prompt_tokens(model)

    def get_calculated_cost(self, model: str) -> float:
        """Get the calculated cost based on instance token counts and model."""
        if self.usage.completion_cost > 0:
            return self.usage.completion_cost
        return Usage.calculate_cost(
            model,
            self.usage.prompt_tokens,
            self.usage.completion_tokens,
            self.usage.cache_read_tokens,
        )

    @classmethod
    def from_completion_response(cls, completion_response: dict | BaseModel, model: str) -> "CompletionAttempt":
        """Create a new invocation instance from a completion response."""
        usage = Usage.from_completion_response(completion_response, model)
        return cls(usage=usage)

    @classmethod
    def create_failed_invocation(cls, failure_reason: str, attempt_number: int = 1) -> "CompletionAttempt":
        """Create a CompletionAttempt instance for a failed attempt."""
        current_time_ms = cls._current_time_ms()
        return cls(
            start_time=current_time_ms,
            end_time=current_time_ms,
            success=False,
            failure_reason=failure_reason,
            attempt_number=attempt_number,
        )

    def __add__(self, other: "CompletionAttempt") -> "CompletionAttempt":
        """Add two invocations together by combining their usage data."""
        if not isinstance(other, CompletionAttempt):
            return self

        return CompletionAttempt(usage=self.usage + other.usage)

    def __str__(self) -> str:
        status = "SUCCESS" if self.success else f"FAILED: {self.failure_reason}"
        duration_ms = (self.end_time - self.start_time) if self.end_time and self.start_time else 0
        duration_sec = duration_ms / 1000
        return f"Completion[{status}] " f"(duration: {duration_sec:.2f}s, {self.usage})"

    def model_dump(self, **kwargs) -> dict:
        data = super().model_dump(**kwargs)
        data["usage"] = self.usage.model_dump(**kwargs)
        return data

    @classmethod
    def model_validate(cls, data: Any, **kwargs):
        if isinstance(data, dict):
            data["usage"] = Usage.model_validate(data["usage"], **kwargs)
            return super().model_validate(data, **kwargs)
        return data


class CompletionInvocation(BaseModel):
    model: Optional[str] = None
    attempts: list[CompletionAttempt] = Field(
        default_factory=list,
        description="The usage information for each completion attempt",
    )
    start_time: float = Field(default=0, description="The start time of the entire invocation in milliseconds")
    end_time: float = Field(default=0, description="The end time of the entire invocation in milliseconds")
    current_attempt: Optional[CompletionAttempt] = Field(default=None, exclude=True)
    model_config = ConfigDict(extra="ignore")

    @staticmethod
    def _current_time_ms() -> float:
        """Get current time in milliseconds."""
        return time.time() * 1000

    def __enter__(self) -> "CompletionInvocation":
        """Start timing when entering the context manager."""
        self.start_time = self._current_time_ms()
        # Create a new attempt and store it as the current attempt
        self.current_attempt = CompletionAttempt(attempt_number=len(self.attempts) + 1)
        self.current_attempt.start_time = self.start_time
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """End timing when exiting the context manager."""
        end_time = self._current_time_ms()
        self.end_time = end_time

        if self.current_attempt:
            # Automatically mark as failed if there was an exception
            if exc_type is not None:
                self.current_attempt.success = False
                self.current_attempt.failure_reason = str(exc_val)

            # Set the end time and add the attempt to our list
            self.current_attempt.end_time = end_time
            self.attempts.append(self.current_attempt)
            self.current_attempt = None

    @classmethod
    def from_llm_completion(
        cls,
        completion_response: Any,
        model: str,
        completion_metrics: CompletionAttempt | None = None,
    ) -> "CompletionInvocation":
        if completion_response is None:
            raise ValueError("Completion response is None")

        if not completion_metrics:
            completion_metrics = CompletionAttempt.from_completion_response(completion_response, model)

        return cls(model=model, attempts=[completion_metrics])

    def add_attempt(self, attempt: CompletionAttempt) -> None:
        """Add a completion attempt to this invocation."""
        self.attempts.append(attempt)

    @property
    def usage(self) -> Usage:
        """Return the total usage across all completions."""
        if not self.attempts:
            return Usage()

        total_usage = Usage()
        for attempt in self.attempts:
            total_usage = total_usage + attempt.usage

        return total_usage

    @property
    def last_attempt(self) -> Optional[CompletionAttempt]:
        """Return the last attempt, or None if there are no attempts."""
        if not self.attempts:
            return None
        return self.attempts[-1]

    @property
    def duration_ms(self) -> float:
        """Return the duration of the invocation in milliseconds."""
        if not self.start_time or not self.end_time:
            return 0
        return self.end_time - self.start_time

    @property
    def duration_sec(self) -> float:
        """Return the duration of the invocation in seconds."""
        return self.duration_ms / 1000

    def __str__(self) -> str:
        num_attempts = len(self.attempts)
        success = any(attempt.success for attempt in self.attempts) if self.attempts else False
        status = "SUCCESS" if success else "FAILED"
        return (
            f"CompletionInvocation[{status}] "
            f"(model: {self.model}, attempts: {num_attempts}, duration: {self.duration_sec:.2f}s, {self.usage})"
        )

    def model_dump(self, **kwargs) -> dict:
        data = super().model_dump(**kwargs)
        data["attempts"] = [attempt.model_dump(**kwargs) for attempt in self.attempts]
        return data

    @classmethod
    def model_validate(cls, data: Any, **kwargs):
        if isinstance(data, dict):
            if "attempts" in data:
                data["attempts"] = [CompletionAttempt.model_validate(attempt, **kwargs) for attempt in data["attempts"]]
            return super().model_validate(data, **kwargs)
        return data
