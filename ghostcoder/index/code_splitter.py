import re
from dataclasses import dataclass
from typing import Optional, Any, List, Tuple

import tree_sitter_languages
from langchain.text_splitter import TextSplitter
from llama_index.callbacks import CallbackManager, CBEventType
from llama_index.callbacks.schema import EventPayload
from tree_sitter import Node


def count_length_without_whitespace(s: str):
    string_without_whitespace = re.sub(r'\s', '', s)
    return len(string_without_whitespace)


def find_node(node: Node, node_type: str):
    for child in node.children:
        if child.type == node_type:
            return child
    return None


def get_definition_end(node: Node) -> Tuple[int, Node]:
    if node.type in ["function_definition", "class_definition"]:
        return find_node(node, ":").start_byte + 1, find_node(node, "block")
    elif node.type in ["class_declaration"]:
        return find_node(node, "class_body").start_byte, find_node(node, "class_body")
    elif node.type in ["constructor_declaration"]:
        return find_node(node, "constructor_body").start_byte, find_node(node, "constructor_body")
    elif node.type in ["method_declaration"]:
        block = find_node(node, "block")
        if block:
            return block.start_byte, find_node(node, "block")
    return None, None


def get_text(node: Node, text: str):
    return text[node.start_byte:node.end_byte]


class Chunk:
    text: str
    definitions: List[str]
    first: bool = False

    def __init__(self, text: str, definitions: List[str], first: bool = False):
        self.text = text
        self.definitions = definitions
        self.first = first

    def __str__(self):
        pre_def = ""

        for definition in self.definitions:
            pre_def += f"{definition}"
            if not self.first or definition != self.definitions[-1]:
                pre_def += "\n...\n"

        return pre_def + self.text

    def equal_names(self, other):
        if self.definitions != other.definitions:
            return False
        return True


class CodeSplitter(TextSplitter):
    """Split code using a AST parser.

    Thank you to Kevin Lu / SweepAI for suggesting this elegant code splitting solution.
    https://docs.sweep.dev/blogs/chunking-2m-files
    """

    def __init__(
            self,
            language: str,
            chunk_lines: int = 40,
            chunk_lines_overlap: int = 15,
            max_chars: int = 1500,
            callback_manager: Optional[CallbackManager] = None,
    ):
        self.language = language
        self.chunk_lines = chunk_lines
        self.chunk_lines_overlap = chunk_lines_overlap
        self.max_chars = max_chars
        self.callback_manager = callback_manager or CallbackManager([])
        self.parser = None  # Set a default value

        try:
            self.parser = tree_sitter_languages.get_parser(language)
        except Exception as e:
            print(
                f"Could not get parser for language {language}. Check "
                "https://github.com/grantjenks/py-tree-sitter-languages#license "
                "for a list of valid languages."
            )
            raise e

    def _chunk_node(self,
                    node: Any,
                    text: str,
                    last_end: int = 0,
                    definitions: List[str] = []) -> List[Chunk]:
        new_chunks = []
        current_chunk = ""
        for child in node.children:
            definition_end, block = get_definition_end(child)
            if definition_end and block:
                definition = text[child.start_byte:definition_end]
                new_chunks.extend(self._chunk_node(block, text, definition_end, definitions + [definition]))
            elif child.end_byte - child.start_byte > self.max_chars:
                # Child is too big, recursively chunk the child
                if len(current_chunk) > 0:
                    new_chunks.append(Chunk(current_chunk, definitions, len(new_chunks) == 0))
                current_chunk = ""

                new_chunks.extend(self._chunk_node(child, text, last_end, definitions))
            elif (
                    len(current_chunk) + child.end_byte - child.start_byte > self.max_chars
            ):
                # Child would make the current chunk too big, so start a new chunk
                new_chunks.append(Chunk(current_chunk, definitions, len(new_chunks) == 0))
                current_chunk = text[last_end: child.end_byte]
            else:
                current_chunk += text[last_end: child.end_byte]
            last_end = child.end_byte
        if len(current_chunk) > 0:
            new_chunks.append(Chunk(current_chunk, definitions, len(new_chunks) == 0))
        return new_chunks

    def split_text(self, text: str) -> List[str]:
        """Split incoming code and return chunks using the AST."""
        with self.callback_manager.event(
                CBEventType.CHUNKING, payload={EventPayload.CHUNKS: [text]}
        ) as event:
            tree = self.parser.parse(bytes(text, "utf-8"))

            if (
                    not tree.root_node.children
                    or tree.root_node.children[0].type != "ERROR"
            ):
                chunks = self._chunk_node(tree.root_node, text)

                # combining small chunks with bigger ones
                new_chunks = []
                if len(chunks) > 0:
                    current_chunk = chunks[0]
                    i = 1
                    while i < len(chunks):
                        if count_length_without_whitespace(current_chunk.text) \
                                + count_length_without_whitespace(chunks[i].text) > self.max_chars \
                                or not chunks[i].equal_names(current_chunk):
                            new_chunks.append(current_chunk)
                            current_chunk = chunks[i]
                        else:
                            current_chunk.text += chunks[i].text
                        i += 1

                    if len(current_chunk.text) > 0:
                        new_chunks.append(current_chunk)

                event.on_end(
                    payload={EventPayload.CHUNKS: chunks},
                )

                return [str(chunk) for chunk in new_chunks]
            else:
                raise ValueError(f"Could not parse code with language {self.language}.")

        # TODO: set up auto-language detection using something like https://github.com/yoeo/guesslang.


def print_chunks(file_path: str):
    splitter = CodeSplitter(language="java")

    with open(file_path, "r") as file:
        code = file.read()

    chunks = splitter.split_text(code)

    i = 1
    for chunk in chunks:
        print("--- Chunk", i, "---")
        print(chunk)
        i += 1


if __name__ == "__main__":
    # TODO: Set from args and detect language
    print_chunks("/home/albert/repos/albert/ghostcoder/ghostcoder/filerepository.py")
