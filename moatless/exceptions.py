from typing import Any, List


class MoatlessError(Exception):
    """Base exception class for all Moatless exceptions."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class RuntimeError(MoatlessError):
    """Exception raised when an operation encounters a fundamental error that prevents further execution."""

    pass


class RejectError(MoatlessError):
    """Exception raised when an operation should be rejected but the flow can continue."""

    pass


class CompletionError(MoatlessError):
    """Base exception for completion-related errors."""

    def __init__(
        self, message: str, last_completion: Any = None, messages: List[dict] = None
    ):
        super().__init__(message)
        self.last_completion = last_completion
        self.messages = messages or []


class CompletionRuntimeError(RuntimeError, CompletionError):
    """Exception raised when completion encounters an unrecoverable error."""

    pass


class CompletionRejectError(RejectError, CompletionError):
    """Exception raised when completion should reject the current node but continue search."""

    pass


class CompletionValidationError(MoatlessError):
    """Exception raised when completion encounters a validation error."""

    pass
