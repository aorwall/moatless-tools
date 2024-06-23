from dataclasses import dataclass


@dataclass
class _Settings:

    _coder: _CoderSettings = _CoderSettings()
    _search: _SearchSettings = _SearchSettings()

    _agent_model: str = "gpt-4o-2024-05-13"
    _embed_model: str = "text-embedding-3-small"

    _max_context_tokens: int = 8000
    _max_message_tokens: int = 16000

    # TODO: Remove _one_file_mode
    _one_file_mode = True

    @property
    def coder(self) -> _CoderSettings:
        return self._coder

    @property
    def agent_model(self) -> str:
        return self._agent_model

    @agent_model.setter
    def agent_model(self, agent_model: str) -> None:
        self._agent_model = agent_model

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
