from moatless.index.code_graph import CodeGraph
from moatless.index.retriever import CodeSnippetRetriever


class CodeIndex:

    def __init__(self, graph: CodeGraph, retriever: CodeSnippetRetriever):
        self.graph = graph
        self.retriever = retriever
