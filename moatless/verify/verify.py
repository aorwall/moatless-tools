from abc import ABC, abstractmethod
from typing import Optional

from moatless.repository import CodeFile
from moatless.types import VerificationError


class Verifier(ABC):

    @abstractmethod
    def verify(self, file: Optional[CodeFile] = None) -> list[VerificationError]:
        pass
