from dataclasses import dataclass


class _ContextSettings:
    _include_imports = False

    @property
    def include_imports(self) -> bool:
        return self._include_imports

    @include_imports.setter
    def include_imports(self, include_imports: bool) -> None:
        self._include_imports = include_imports


class _CoderSettings:

    _planning_model = "gpt-4-turbo-2024-04-09"
    _coding_model = "gpt-4-turbo-2024-04-09"
    _enable_chain_of_thought = True
    _min_tokens_for_split_span = 200
    _debug_mode = False
    _use_line_numbers = False
    _use_spans = True

    _context: _ContextSettings = _ContextSettings()

    @property
    def planning_model(self) -> str:
        return self._planning_model

    @planning_model.setter
    def planning_model(self, planning_model: str) -> None:
        self._planning_model = planning_model

    @property
    def coding_model(self) -> str:
        return self._coding_model

    @coding_model.setter
    def coding_model(self, coding_model: str) -> None:
        self._coding_model = coding_model

    @property
    def enable_chain_of_thought(self) -> bool:
        return self._enable_chain_of_thought

    @enable_chain_of_thought.setter
    def enable_chain_of_thought(self, enable_chain_of_thought: bool) -> None:
        """Set the LLM."""
        self._enable_chain_of_thought = enable_chain_of_thought

    @property
    def use_spans(self) -> bool:
        return self._use_spans

    @use_spans.setter
    def use_spans(self, use_spans: bool) -> None:
        self._use_spans = use_spans

    @property
    def use_line_numbers(self) -> bool:
        return self._use_line_numbers

    @use_line_numbers.setter
    def use_line_numbers(self, use_line_numbers: bool) -> None:
        self._use_line_numbers = use_line_numbers

    @property
    def min_tokens_for_split_span(self) -> int:
        return self._min_tokens_for_split_span

    @min_tokens_for_split_span.setter
    def min_tokens_for_split_span(self, min_tokens_for_split_span: int) -> None:
        self._min_tokens_for_split_span = min_tokens_for_split_span

    @property
    def debug_mode(self) -> bool:
        return self._debug_mode

    @debug_mode.setter
    def debug_mode(self, debug_mode: bool) -> None:
        self._debug_mode = debug_mode


@dataclass
class _Settings:

    _one_file_mode = True
    _coder: _CoderSettings = _CoderSettings()
    _analytics_file: str = None

    @property
    def coder(self) -> _CoderSettings:
        return self._coder

    @property
    def one_file_mode(self) -> bool:
        return self._one_file_mode

    @one_file_mode.setter
    def one_file_mode(self, one_file_mode: bool) -> None:
        self._one_file_mode = one_file_mode

    @property
    def analytics_file(self) -> str:
        return self._analytics_file

    @analytics_file.setter
    def analytics_file(self, analytics_file: str) -> None:
        self._analytics_file = analytics_file


Settings = _Settings()
