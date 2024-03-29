

## EpicSplitter

1 chunks

#### Split 1
167 tokens, line: 7 - 34

```python
from __future__ import annotations

from typing import TYPE_CHECKING

from pylint import utils
from pylint.reporters.base_reporter import BaseReporter
from pylint.reporters.collecting_reporter import CollectingReporter
from pylint.reporters.json_reporter import JSONReporter
from pylint.reporters.multi_reporter import MultiReporter
from pylint.reporters.reports_handler_mix_in import ReportsHandlerMixIn

if TYPE_CHECKING:
    from pylint.lint.pylinter import PyLinter


def initialize(linter: PyLinter) -> None:
    """Initialize linter with reporters in this package."""
    utils.register_plugins(linter, __path__[0])


__all__ = [
    "BaseReporter",
    "ReportsHandlerMixIn",
    "JSONReporter",
    "CollectingReporter",
    "MultiReporter",
]
```

