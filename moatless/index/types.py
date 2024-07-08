from dataclasses import dataclass

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
    start_line: int | None = None
    end_line: int | None = None
    start_block: str | None = None
    end_block: str | None = None


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


class SearchCodeResponse(BaseModel):
    message: str | None = Field(
        default=None, description="A message to return to the user."
    )

    hits: list[SearchCodeHit] = Field(
        default_factory=list,
        description="Search results.",
    )
