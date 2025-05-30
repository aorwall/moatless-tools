import fnmatch
import json
import logging
import os
from typing import Optional

import aiofiles
from opentelemetry import trace

logger = logging.getLogger(__name__)

tracer = trace.get_tracer(__name__)


class CodeBlockIndex:
    """
    Manages code block indexes (by class and function name) and their associated file path indexes.
    In this implementation we also build a tree index to efficiently match glob patterns.
    """

    def __init__(
        self,
        blocks_by_class_name: Optional[dict[str, list[tuple[str, str]]]] = None,
        blocks_by_function_name: Optional[dict[str, list[tuple[str, str]]]] = None,
    ):
        self._blocks_by_class_name = blocks_by_class_name or {}
        self._blocks_by_function_name = blocks_by_function_name or {}

        # Our tree index for file paths
        self._file_tree: dict = {}

    async def _build_indexes(self):
        """Build the tree index from blocks."""
        # Process class blocks
        for blocks in self._blocks_by_class_name.values():
            for file_path, _ in blocks:
                self._insert_into_tree(file_path)

        # Process function blocks
        for blocks in self._blocks_by_function_name.values():
            for file_path, _ in blocks:
                self._insert_into_tree(file_path)

        logger.debug(f"Built file tree index: {self._file_tree}")

    def _insert_into_tree(self, file_path: str):
        """Insert a file path into the tree.
        Directories are represented as dicts;
        files are stored as a terminal key with value None."""
        parts = file_path.split("/")
        node = self._file_tree
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                node[part] = {}
            node = node[part]
        # Mark the file by setting its value to None.
        node[parts[-1]] = None

    def _search_tree(self, node: dict, parts: list[str], cur_path: str) -> set[str]:
        """
        Recursively search the tree for files matching the given pattern parts.
        - If a part is '**', match zero or more directories.
        - Otherwise, use fnmatch to match the current level.
        """
        if not parts:
            return set()

        part = parts[0]
        rest = parts[1:]
        results = set()

        if all(c == "*" for c in part):
            # Option 1: Skip '**' (match zero directories)
            results |= self._search_tree(node, rest, cur_path)
            # Option 2: Recurse into every subdirectory keeping the '**'
            for name, child in node.items():
                if isinstance(child, dict):
                    new_path = os.path.join(cur_path, name) if cur_path else name
                    results |= self._search_tree(child, parts, new_path)
            return results
        else:
            # For normal glob parts
            for name, child in node.items():
                if fnmatch.fnmatchcase(name, part):
                    new_path = os.path.join(cur_path, name) if cur_path else name
                    if rest:
                        if isinstance(child, dict):
                            results |= self._search_tree(child, rest, new_path)
                    else:
                        # Last part: add if it's a file (leaf node)
                        if child is None:
                            results.add(new_path)
            return results

    async def match_glob_pattern(self, file_pattern: str) -> set[str]:
        """
        Match files against a glob pattern using the tree index.

        If the query does not include a directory separator, assume
        a recursive search (i.e. prepend '**/').
        """
        logger.debug(f"Matching pattern using tree index: {file_pattern}")
        # If no slash is in the pattern, search recursively from the root.
        if "/" not in file_pattern:
            file_pattern = "**/" + file_pattern

        # Split the pattern into parts.
        pattern_parts = file_pattern.split("/")
        # Search starting from the file tree root.
        matches = self._search_tree(self._file_tree, pattern_parts, "")
        logger.debug(f"Pattern '{file_pattern}' matched files: {matches}")
        return matches

    async def get_blocks_by_class(self, class_name: str) -> list[tuple[str, str]]:
        """Get all blocks for a given class name."""
        return self._blocks_by_class_name.get(class_name, [])

    async def get_blocks_by_function(self, function_name: str) -> list[tuple[str, str]]:
        """Get all blocks for a given function name."""
        return self._blocks_by_function_name.get(function_name, [])

    def persist(self, persist_dir: str):
        """Save all indexes to disk."""
        # Convert tuples to lists for JSON serialization
        blocks_by_class_name = {k: [list(t) for t in v] for k, v in self._blocks_by_class_name.items()}

        blocks_by_function_name = {k: [list(t) for t in v] for k, v in self._blocks_by_function_name.items()}

        with open(os.path.join(persist_dir, "blocks_by_class_name.json"), "w") as f:
            f.write(json.dumps(blocks_by_class_name, indent=2))

        with open(os.path.join(persist_dir, "blocks_by_function_name.json"), "w") as f:
            f.write(json.dumps(blocks_by_function_name, indent=2))

        # Save tree index
        inverted_indexes = {"file_tree": self._file_tree}
        with open(os.path.join(persist_dir, "inverted_indexes.json"), "w") as f:
            f.write(json.dumps(inverted_indexes, indent=2))

    @classmethod
    def from_persist_dir(cls, persist_dir: str) -> "CodeBlockIndex":
        """Load indexes from disk."""
        blocks_by_class_name = {}
        blocks_by_function_name = {}

        if os.path.exists(os.path.join(persist_dir, "blocks_by_class_name.json")):
            with open(os.path.join(persist_dir, "blocks_by_class_name.json"), "r") as f:
                blocks_by_class_name = json.load(f)

        if os.path.exists(os.path.join(persist_dir, "blocks_by_function_name.json")):
            with open(os.path.join(persist_dir, "blocks_by_function_name.json"), "r") as f:
                blocks_by_function_name = json.load(f)

        instance = cls(
            blocks_by_class_name=blocks_by_class_name,
            blocks_by_function_name=blocks_by_function_name,
        )

        # Load tree index if it exists
        if os.path.exists(os.path.join(persist_dir, "inverted_indexes.json")):
            with open(os.path.join(persist_dir, "inverted_indexes.json"), "r") as f:
                inverted_indexes = json.load(f)
                instance._file_tree = inverted_indexes.get("file_tree", {})

        return instance

    @classmethod
    async def from_persist_dir_async(cls, persist_dir: str) -> "CodeBlockIndex":
        """Load indexes from disk."""
        blocks_by_class_name = {}
        blocks_by_function_name = {}

        if os.path.exists(os.path.join(persist_dir, "blocks_by_class_name.json")):
            async with aiofiles.open(os.path.join(persist_dir, "blocks_by_class_name.json")) as f:
                content = await f.read()
                data = json.loads(content)
                blocks_by_class_name = {k: [tuple(t) for t in v] for k, v in data.items()}

        if os.path.exists(os.path.join(persist_dir, "blocks_by_function_name.json")):
            async with aiofiles.open(os.path.join(persist_dir, "blocks_by_function_name.json")) as f:
                content = await f.read()
                data = json.loads(content)
                blocks_by_function_name = {k: [tuple(t) for t in v] for k, v in data.items()}

        instance = cls(
            blocks_by_class_name=blocks_by_class_name,
            blocks_by_function_name=blocks_by_function_name,
        )

        # Load tree index if it exists
        if os.path.exists(os.path.join(persist_dir, "inverted_indexes.json")):
            async with aiofiles.open(os.path.join(persist_dir, "inverted_indexes.json")) as f:
                content = await f.read()
                inverted_indexes = json.loads(content)
                instance._file_tree = inverted_indexes.get("file_tree", {})
        else:
            # Build tree index from blocks
            await instance._build_indexes()

        return instance
