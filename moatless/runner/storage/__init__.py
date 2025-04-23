"""Storage implementations for the job scheduler.

This package provides storage interfaces and implementations for the job scheduler,
including in-memory and Redis-based storage options.
"""

from moatless.runner.storage.storage import JobStorage
from moatless.runner.storage.memory import InMemoryJobStorage

__all__ = ["JobStorage", "InMemoryJobStorage"]