from moatless.code_graph import CodeGraph
from moatless.retriever import CodeSnippetRetriever


class CodeIndex:

    def __init__(self, graph: CodeGraph, retriever: CodeSnippetRetriever):
        self.graph = graph
        self.retriever = retriever
