from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel, Field


@dataclass
class CodeSnippet:
    id: str
    file_path: str
    content: str = None
    distance: float = 0.0
    tokens: int = None
    language: str = "python"
    span_ids: list[str] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    start_block: Optional[str] = None
    end_block: Optional[str] = None


class SpanHit(BaseModel):
    span_id: str = Field(description="The span id of the relevant code in the file")
    rank: int = Field(
        default=0,
        description="The rank of relevance of the span in the file. 0 is highest.",
    )
    tokens: int = Field(default=0, description="The number of tokens in the span.")


class SearchCodeHit(BaseModel):
    file_path: str = Field(
        description="The file path where the relevant code is found."
    )
    spans: list[SpanHit] = Field(
        default_factory=list,
        description="The spans of the relevant code in the file",
    )

    @property
    def span_ids(self):
        return [span.span_id for span in self.spans]

    def add_span(self, span_id: str, rank: int = 0, tokens: int = 0):
        if span_id not in [span.span_id for span in self.spans]:
            self.spans.append(SpanHit(span_id=span_id, rank=rank, tokens=tokens))

    def contains_span(self, span_id: str) -> bool:
        return span_id in [span.span_id for span in self.spans]

    def add_spans(self, span_ids: list[str], rank: int = 0):
        for span_id in span_ids:
            self.add_span(span_id, rank)

    def __str__(self):
        return f"{self.file_path}: {', '.join([span.span_id for span in self.spans])}"


class SearchCodeResponse(BaseModel):
    message: Optional[str] = Field(
        default=None, description="A message to return to the user."
    )

    hits: list[SearchCodeHit] = Field(
        default_factory=list,
        description="Search results.",
    )

    def sum_tokens(self):
        return sum([sum([span.tokens for span in hit.spans]) for hit in self.hits])
