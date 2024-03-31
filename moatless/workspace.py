from moatless.retriever import CodeSnippetRetriever


class Workspace:

    def __init__(self, retriever: CodeSnippetRetriever):
        self._retriever = retriever

    def run(self, instructions: str):
        snippets = self._retriever.retrieve(instructions)

        # TODO: Select the right files

        # TODO: Select blocks to edit, remove or add

        # TODO: Edit blocks
