import logging
from typing import List

import networkx as nx
from pydantic import BaseModel

from moatless.codeblocks import CodeBlock
from moatless.codeblocks.codeblocks import CodeBlockType, Span
from moatless.types import BlockPath

logger = logging.getLogger(__name__)


class CodeSearchHit(BaseModel):
    file_path: str
    block_paths: List[BlockPath] = []
    spans: List[Span] = []


class CodeGraph:
    """A graph representation of the codebase. TODO: WIP!"""

    def __init__(self):
        self._graph = nx.DiGraph()
        self._blocks_by_file_path = {}

    def add_to_graph(self, file_path: str, codeblock: CodeBlock):
        node = self._create_node(file_path, codeblock.full_path())
        self._graph.add_node(node, block=codeblock, file_path=file_path)

        if codeblock.type == CodeBlockType.MODULE:
            self._blocks_by_file_path[file_path] = codeblock

        #if codeblock.type == CodeBlockType.CLASS:
        #    self._add_node_with_instance_vars(file_path, codeblock, codeblock)

        self._add_relationships(file_path, codeblock, node)

    def find(self, file_name: str = None, class_name: str = None, function_name: str = None) -> List[CodeSearchHit]:
        results = {}

        if not file_name and not class_name and not function_name:
            return []

        # TODO: Just to verify the idea to find by class/file/function name, needs to be optimized
        for node in self._graph.nodes:
            node_data = self._graph.nodes[node]

            if "block" not in node_data or "file_path" not in node_data:
                continue

            file_path = node_data["file_path"]

            block = node_data["block"]
            if file_name and file_name.lower() not in file_path.lower():
                continue

            if class_name and block.type == CodeBlockType.CLASS and class_name.lower() not in block.identifier.lower():
                continue

            if function_name and block.type == CodeBlockType.FUNCTION and function_name.lower() not in block.identifier.lower():
                continue

            if file_path not in results:
                results[file_path] = CodeSearchHit(
                    file_path=file_path,
                    block_paths=[],
                    spans=[]
                )

            results[file_path].block_paths.append(block.full_path())
            results[file_path].spans.append(Span(block.start_line, block.end_line))

        return list(results.values())

    def find_relationships(self, file_path: str, block_path: List[str]):
        node_id = self._create_node(file_path, block_path)

        if node_id not in self._graph:
            logger.warning(f"Code block {node_id} not found in graph.")
            return

        related_blocks = []

        # Find predecessors (incoming relationships)
        predecessors = list(self._graph.predecessors(node_id))
        for pred in predecessors:
            node_data = self._graph.nodes[pred]
            if "block" in node_data:
                related_blocks.append(node_data["block"])

        # Find successors (outgoing relationships)
        successors = list(self._graph.successors(node_id))
        for succ in successors:
            node_data = self._graph.nodes[succ]
            if "block" in node_data:
                related_blocks.append(node_data["block"])

        return related_blocks

    def _add_node_with_instance_vars(self, file_path: str, codeblock: CodeBlock, class_block: CodeBlock):
        # TODO: Just experimantal workaround to point all instance variables to the class
        for child in codeblock.children:
            if child.type == CodeBlockType.ASSIGNMENT and child.identifier.startswith("self."):
                identifier = child.identifier.split(".")[1]
                node = self._create_node(file_path, class_block.full_path() + [identifier])
                self._graph.add_node(node, block=class_block)
            else:
                self._add_node_with_instance_vars(file_path, child, class_block)

    def _add_relationships(self, file_path: str, codeblock: CodeBlock, from_node: str):
        for rel in codeblock.references:
            self._add_edge(from_node, file_path, rel.path)

        for child in codeblock.children:
            if not child.is_indexed:
                self._add_relationships(file_path, child, from_node)

    def _add_block_edge(self, from_node: str, file_path: str, codeblock: CodeBlock):
        self._add_edge(from_node, file_path, codeblock.full_path())

    def _add_edge(self, from_node: str, file_path: str, block_path: List[str]):
        to_node = self._create_node(file_path, block_path)
        self._graph.add_edge(from_node, to_node)

    def _create_node(self, file_path: str, block_path: List[str]):
        block_path = ".".join(block_path)
        return f"{file_path}::{block_path}"

    def _create_from_node(self, file_path: str, codeblock: CodeBlock):
        if codeblock.is_indexed:
            return f"{file_path}::{codeblock.full_path()}"
        elif codeblock.parent:
            # TODO: Check scope!
            return self._create_from_node(file_path, codeblock.parent)

        return None
