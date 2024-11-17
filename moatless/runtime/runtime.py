from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

from moatless.schema import RankedFileSpan


class TestStatus(str, Enum):
    FAILED = "FAILED"
    PASSED = "PASSED"
    SKIPPED = "SKIPPED"
    ERROR = "ERROR"


class TestResult(BaseModel):
    status: TestStatus
    message: Optional[str] = None
    file_path: Optional[str] = None
    span_id: Optional[str] = None
    line: Optional[int] = None
    relevant_files: List[RankedFileSpan] = Field(
        default_factory=list,
        description="List of spans that are relevant to the issue",
    )


class RuntimeEnvironment(ABC):
    @abstractmethod
    def run_tests(
        self, patch: str, test_files: List[str] | None = None
    ) -> list[TestResult]:
        pass
