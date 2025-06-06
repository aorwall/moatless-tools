# Import CompletionInvocation for type hint
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


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
        self,
        message: str,
        completion_invocation: "CompletionInvocation | None" = None,
    ):
        super().__init__(message)
        self.completion_invocation = completion_invocation


class CompletionRuntimeError(RuntimeError, CompletionError):
    """Exception raised when completion encounters an unrecoverable error."""

    pass


class CompletionRejectError(RejectError, CompletionError):
    """Raised when a completion response is rejected due to validation failure."""

    pass


class CompletionValidationError(MoatlessError):
    """Exception raised when completion encounters a validation error."""

    pass
