from abc import ABC, abstractmethod

from moatless.repository import CodeFile
from moatless.schema import VerificationError


class Verifier(ABC):
    @abstractmethod
    def verify(self, file: CodeFile | None = None) -> list[VerificationError]:
        pass
