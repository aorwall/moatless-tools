from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from moatless.schema import RankedFileSpan


class TestStatus(str, Enum):
    FAILED = "FAILED"
    PASSED = "PASSED"
    SKIPPED = "SKIPPED"
    ERROR = "ERROR"


class TestResult(BaseModel):
    name: Optional[str] = None
    status: TestStatus
    message: Optional[str] = None
    file_path: Optional[str] = None
    span_id: Optional[str] = None
    line: Optional[int] = None
    relevant_files: list[RankedFileSpan] = Field(
        default_factory=list,
        description="List of spans that are relevant to the issue",
    )


class RuntimeEnvironment(ABC):
    @abstractmethod
    async def run_tests(self, patch: str | None = None, test_files: list[str] | None = None) -> list[TestResult]:
        pass


class NoEnvironment(RuntimeEnvironment):
    async def run_tests(self, patch: str | None = None, test_files: list[str] | None = None) -> list[TestResult]:
        return []
