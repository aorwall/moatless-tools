import os
from dataclasses import dataclass


@dataclass
class _Settings:
    _agent_model: str = os.environ.get("AGENT_MODEL", "gpt-4o-2024-05-13")
    _cheap_model: str = os.environ.get("CHEAP_MODEL", "gpt-4o-mini-2024-07-18")
    _embed_model: str = "text-embedding-3-small"

    _max_context_tokens: int = 8000
    _max_message_tokens: int = 16000

    @property
    def agent_model(self) -> str:
        return self._agent_model

    @agent_model.setter
    def agent_model(self, agent_model: str) -> None:
        self._agent_model = agent_model

    @property
    def cheap_model(self) -> str:
        return self._cheap_model

    @cheap_model.setter
    def cheap_model(self, cheap_model: str) -> None:
        self._cheap_model = cheap_model

    @property
    def embed_model(self) -> str:
        return self._embed_model

    @embed_model.setter
    def embed_model(self, embed_model: str) -> None:
        self._embed_model = embed_model

    @property
    def max_context_tokens(self) -> int:
        return self._max_context_tokens

    @max_context_tokens.setter
    def max_context_tokens(self, max_context_tokens: int) -> None:
        self._max_context_tokens = max_context_tokens

    @property
    def max_message_tokens(self) -> int:
        return self._max_message_tokens

    @max_message_tokens.setter
    def max_message_tokens(self, max_message_tokens: int) -> None:
        self._max_message_tokens = max_message_tokens


Settings = _Settings()
