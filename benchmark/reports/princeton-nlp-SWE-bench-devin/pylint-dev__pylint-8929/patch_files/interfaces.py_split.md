

## EpicSplitter

1 chunks

#### Split 1
209 tokens, line: 5 - 38

```python
from __future__ import annotations

from typing import NamedTuple

__all__ = (
    "HIGH",
    "CONTROL_FLOW",
    "INFERENCE",
    "INFERENCE_FAILURE",
    "UNDEFINED",
    "CONFIDENCE_LEVELS",
    "CONFIDENCE_LEVEL_NAMES",
)


class Confidence(NamedTuple):
    name: str
    description: str


# Warning Certainties
HIGH = Confidence("HIGH", "Warning that is not based on inference result.")
CONTROL_FLOW = Confidence(
    "CONTROL_FLOW", "Warning based on assumptions about control flow."
)
INFERENCE = Confidence("INFERENCE", "Warning based on inference result.")
INFERENCE_FAILURE = Confidence(
    "INFERENCE_FAILURE", "Warning based on inference with failures."
)
UNDEFINED = Confidence("UNDEFINED", "Warning without any associated confidence level.")

CONFIDENCE_LEVELS = [HIGH, CONTROL_FLOW, INFERENCE, INFERENCE_FAILURE, UNDEFINED]
CONFIDENCE_LEVEL_NAMES = [i.name for i in CONFIDENCE_LEVELS]
```

