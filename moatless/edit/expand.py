import logging
from typing import Optional

from pydantic import Field

from moatless.codeblocks import CodeBlockType
from moatless.state import State, ActionRequest, StateOutcome


logger = logging.getLogger(__name__)


class ExpandContext(State):
    """ """

    expand_to_max_tokens: int = Field(
        12000,
        description="The maximum number of tokens to expand context with.",
    )

    expand_classes: bool = Field(
        False,
        description="Whether to expand with class blocks.",
    )

    expand_relations: bool = Field(
        True,
        description="Whether to expand with related spans.",
    )

    expand_other: bool = Field(
        False,
        description="Whether to expand with related spans.",
    )

    def execute(
        self, mocked_action_request: ActionRequest | None = None
    ) -> StateOutcome:
        # TODO: Provide more info to use in the search query?
        results = self.workspace.code_index.semantic_search(
            query=self.initial_message, max_results=1000
        )

        # Flatten and sort the search results by rank
        flattened_results = []
        for hit in results.hits:
            for span in hit.spans:
                flattened_results.append((hit.file_path, span.span_id, span.rank))

        flattened_results.sort(key=lambda x: (x[2]))

        span_ids = set()

        if self.expand_classes:
            span_ids.update(self.get_class_spans())

        if self.expand_relations:
            span_ids.update(self.get_related_spans())

        added_spans = 0
        original_tokens = self.file_context.context_size()

        for file_path, span_id, rank in flattened_results:
            if span_id not in span_ids:
                continue

            # TODO: Check the sum of the tokens in the context and the tokens in the span
            if self.file_context.context_size() > self.expand_to_max_tokens:
                break

            added_spans += 1
            self.file_context.add_span_to_context(file_path, span_id)

        if self.expand_other:
            # Add possibly relevant spans from the same file
            for file_path, span_id, rank in flattened_results:
                if self.file_context.context_size() > self.expand_to_max_tokens:
                    break

                added_spans += 1
                self.file_context.add_span_to_context(file_path, span_id)

        logger.debug(
            f"Expanded context with {added_spans} spans. Original tokens: {original_tokens}, Expanded tokens: {self.file_context.context_size()}"
        )

        return StateOutcome.finish(
            {
                "added_spans": added_spans,
                "original_tokens": original_tokens,
                "expanded_tokens": self.file_context.context_size(),
            }
        )

    def get_class_spans(self) -> set[str]:
        expanded_classes = set()

        span_ids = set()
        for file in self.file_context.files:
            if not file.file.supports_codeblocks:
                continue

            for span in file.spans:
                block_span = file.module.find_span_by_id(span.span_id)
                if not block_span:
                    continue

                if block_span.initiating_block.type != CodeBlockType.CLASS:
                    class_block = block_span.initiating_block.find_type_in_parents(
                        CodeBlockType.CLASS
                    )
                elif block_span.initiating_block.type == CodeBlockType.CLASS:
                    class_block = block_span.initiating_block
                else:
                    continue

                if class_block.belongs_to_span.span_id in expanded_classes:
                    continue

                span_ids.add(class_block.belongs_to_span.span_id)
                for span_id in class_block.get_all_span_ids():
                    span_ids.add(span_id)

        return span_ids

    def get_related_spans(
        self,
    ):
        spans = []
        for file in self.file_context.files:
            if not file.file.supports_codeblocks:
                continue
            if not file.span_ids:
                continue

            for span in file.spans:
                spans.append((file, span))

        spans.sort(key=lambda x: x[1].tokens or 0, reverse=True)

        all_related_span_ids = set()
        for file, span in spans:
            span_id = span.span_id
            related_span_ids = file.module.find_related_span_ids(span_id)
            all_related_span_ids.update(related_span_ids)

        return all_related_span_ids
