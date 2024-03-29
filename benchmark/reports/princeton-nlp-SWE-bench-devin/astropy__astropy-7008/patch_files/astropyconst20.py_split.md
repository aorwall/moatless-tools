

## EpicSplitter

1 chunks

#### Split 1
82 tokens, line: 9 - 18

```python
import itertools

from .constant import Constant
from . import codata2014, iau2015

for _nm, _c in itertools.chain(sorted(vars(codata2014).items()),
                               sorted(vars(iau2015).items())):
    if (isinstance(_c, Constant) and _c.abbrev not in locals()):
        locals()[_c.abbrev] = _c
```

