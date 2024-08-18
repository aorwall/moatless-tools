import logging
from dataclasses import field, dataclass
from typing import Optional, Dict

from networkx import DiGraph

from moatless.codeblocks import CodeBlock, CodeBlockType
from moatless.codeblocks.codeblocks import BlockSpan, SpanType

logger = logging.getLogger(__name__)


@dataclass
class Module(CodeBlock):
    file_path: Optional[str] = None
    content: str = ""
    spans_by_id: Dict[str, BlockSpan] = field(default_factory=dict)
    language: Optional[str] = None
    code_block: CodeBlock = field(
        default_factory=lambda: CodeBlock(content="", type=CodeBlockType.MODULE)
    )
    _graph: DiGraph = field(
        default_factory=DiGraph, init=False
    )  # TODO: Move to central CodeGraph

    def __post_init__(self):
        if not self.code_block.type == CodeBlockType.MODULE:
            self.code_block.type = CodeBlockType.MODULE

    # Delegate other methods to self.code_block as needed
    def __getattr__(self, name):
        return getattr(self.code_block, name)

    @property
    def module(self) -> "Module":  # noqa: F821
        return self

    def find_span_by_id(self, span_id: str) -> BlockSpan | None:
        return self.spans_by_id.get(span_id)

    def sum_tokens(self, span_ids: set[str] | None = None):
        tokens = self.tokens
        if span_ids:
            for span_id in span_ids:
                span = self.spans_by_id.get(span_id)
                if span:
                    tokens += span.tokens
            return tokens

        tokens += sum([child.sum_tokens() for child in self.children])
        return tokens

    def show_spans(
        self,
        span_ids: list[str] | None = None,
        show_related: bool = False,
        max_tokens: int = 2000,
    ) -> bool:
        for span in self.spans_by_id.values():
            span.visible = False

        checked_span_ids = set()
        span_ids_to_check = []

        tokens = 0
        for span_id in span_ids:
            span = self.spans_by_id.get(span_id)
            if not span:
                return False

            tokens += span.tokens
            checked_span_ids.add(span_id)
            span_ids_to_check.append(span_id)
            span.visible = True

        if not show_related:
            return True

        # Add imports from module
        for span in self.spans.values():
            if (
                span.span_type == SpanType.INITATION
                and span.span_id not in checked_span_ids
            ):
                span_ids_to_check.append(span.span_id)

        while span_ids_to_check:
            span_id = span_ids_to_check.pop(0)
            related_spans = self.find_related_spans(span_id)

            logger.info(f"Related spans: {len(related_spans)} for {span_id}")

            # TODO: Go through priotiized related spans to make sure that the most relevant are added first
            # TODO: Verify span token size
            for span in related_spans:
                if span.tokens + tokens > max_tokens:
                    logger.info(
                        f"Max tokens reached: {span.tokens} + {tokens} > {max_tokens}"
                    )
                    return True

                span.visible = True
                tokens += span.tokens

                if span.span_id not in checked_span_ids:
                    checked_span_ids.add(span.span_id)
                    span_ids_to_check.append(span.span_id)

        logger.info(f"Max tokens reached {tokens} < {max_tokens}")

        return True

    def find_related_span_ids(self, span_id: Optional[str] = None) -> set[str]:
        related_span_ids = set()

        blocks = self.find_blocks_by_span_id(span_id)
        for block in blocks:
            # Find successors (outgoing relationships)
            successors = list(self._graph.successors(block.path_string()))
            for succ in successors:
                node_data = self._graph.nodes[succ]
                if "block" in node_data:
                    span = node_data["block"].belongs_to_span
                    related_span_ids.add(span.span_id)

            # Find predecessors (incoming relationships)
            predecessors = list(self._graph.predecessors(block.path_string()))
            for pred in predecessors:
                node_data = self._graph.nodes[pred]
                if "block" in node_data:
                    span = node_data["block"].belongs_to_span
                    related_span_ids.add(span.span_id)

            # Always add parent class initation span
            if block.parent and block.parent.type == CodeBlockType.CLASS:
                related_span_ids.add(block.belongs_to_span.span_id)
                for class_child in block.parent.children:
                    if class_child.belongs_to_span.span_type == SpanType.INITATION:
                        related_span_ids.add(class_child.belongs_to_span.span_id)

        # Always add module initation span
        for span in self.spans_by_id.values():
            if (
                span.block_type == CodeBlockType.MODULE
                and span.span_type == SpanType.INITATION
            ):
                related_span_ids.add(span.span_id)

        return related_span_ids
