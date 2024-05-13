from typing import List, Dict

from pydantic import BaseModel

from moatless.codeblocks.module import Module
from moatless.file_context import ContextFile


class SummarizedSpan:
    span_id: str
    type: str
    summary: str

class Summary(BaseModel):
    summarized_files: Dict[str, List[SummarizedSpan]]



class Summarize:

    def summarize(self, files: List[ContextFile]):

        for file in files:





    def _summarize_spans(self, module: Module):
        pass