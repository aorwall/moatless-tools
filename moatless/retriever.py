from dataclasses import dataclass
from typing import List, Optional


@dataclass
class CodeSnippet:
    id: str
    file_path: str
    content: str
    distance: float = 0.0
    tokens: int = 0
    language: str = "python"
    start_line: Optional[int] = None
    end_line: Optional[int] = None


class CodeSnippetRetriever:

    def retrieve(self, query: str) -> List[CodeSnippet]:
        pass
