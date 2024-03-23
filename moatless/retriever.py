from dataclasses import dataclass
from typing import List, Optional


@dataclass
class CodeSnippet:
    path: str
    content: str
    start_line: Optional[int] = None
    end_line: Optional[int] = None


class CodeSnippetRetriever:

    def retrieve(self, query: str) -> List[CodeSnippet]:
        pass
