import os
from dataclasses import dataclass


@dataclass
class _Settings:

    # Default model used if not provided in global params
    _default_model: str = os.environ.get("DEFAULT_MODEL", "gpt-4o-2024-05-13")

    # Cheaper model used for supporting tasks like creating commit messages
    _cheap_model: str | None = os.environ.get("CHEAP_MODEL", "gpt-4o-mini-2024-07-18")

    # Model used for embedding to index and search vector indexes
    _embed_model: str = "text-embedding-3-small"

    # Flag to determine if llm completions should be included when trajectories are saved
    _include_completions_in_trajectories: bool = True

    @property
    def default_model(self) -> str:
        return self._default_model

    @default_model.setter
    def default_model(self, default_model: str) -> None:
        self._default_model = default_model

    @property
    def cheap_model(self) -> str | None:
        return self._cheap_model

    @cheap_model.setter
    def cheap_model(self, cheap_model: str | None) -> None:
        self._cheap_model = cheap_model

    @property
    def embed_model(self) -> str:
        return self._embed_model

    @embed_model.setter
    def embed_model(self, embed_model: str) -> None:
        self._embed_model = embed_model

    @property
    def include_completions_in_trajectories(self) -> bool:
        return self._include_completions_in_trajectories

    @include_completions_in_trajectories.setter
    def include_completions_in_trajectories(self, include: bool) -> None:
        self._include_completions_in_trajectories = include


Settings = _Settings()