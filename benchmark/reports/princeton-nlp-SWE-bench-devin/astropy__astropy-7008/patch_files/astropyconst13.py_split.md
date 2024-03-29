

## EpicSplitter

1 chunks

#### Split 1
82 tokens, line: 10 - 19

```python
import itertools

from .constant import Constant
from . import codata2010, iau2012

for _nm, _c in itertools.chain(sorted(vars(codata2010).items()),
                               sorted(vars(iau2012).items())):
    if (isinstance(_c, Constant) and _c.abbrev not in locals()):
        locals()[_c.abbrev] = _c
```

