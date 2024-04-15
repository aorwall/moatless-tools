"""Wrapper for testing the Langchain Recursive Character Text Splitter.

."""

from typing import List, Optional

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.callbacks import CallbackManager
from llama_index.core.node_parser import TextSplitter


class RecursiveCharacterTextSplitterLangchainWrapper(TextSplitter):

    _langchain_splitter = PrivateAttr()

    def __init__(
        self,
        language,
        chunk_size: int = 4000,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        try:
            from langchain_text_splitters import (
                RecursiveCharacterTextSplitter,
                Language,
            )
        except ImportError:
            raise ImportError(
                "To use the RecursiveCharacterTextSplitterLangchainWrapper, you must install the langchain package."
            )

        self._langchain_splitter = RecursiveCharacterTextSplitter.from_language(
            language, chunk_size=chunk_size, chunk_overlap=0, keep_separator=True
        )
        callback_manager = callback_manager or CallbackManager([])

        super().__init__(callback_manager=callback_manager)

    @classmethod
    def class_name(cls) -> str:
        return "RecursiveCharacterTextSplitterLangchainWrapper"

    def split_text(self, text: str) -> List[str]:
        return self._langchain_splitter.split_text(text)
