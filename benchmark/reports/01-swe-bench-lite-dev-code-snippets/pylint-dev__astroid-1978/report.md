# pylint-dev__astroid-1978

| **pylint-dev/astroid** | `0c9ab0fe56703fa83c73e514a1020d398d23fa7f` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 11519 |
| **Any found context length** | 3295 |
| **Avg pos** | 58.0 |
| **Min pos** | 10 |
| **Max pos** | 38 |
| **Top file pos** | 9 |
| **Missing snippets** | 3 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/astroid/raw_building.py b/astroid/raw_building.py
--- a/astroid/raw_building.py
+++ b/astroid/raw_building.py
@@ -10,11 +10,14 @@
 
 import builtins
 import inspect
+import io
+import logging
 import os
 import sys
 import types
 import warnings
 from collections.abc import Iterable
+from contextlib import redirect_stderr, redirect_stdout
 from typing import Any, Union
 
 from astroid import bases, nodes
@@ -22,6 +25,9 @@
 from astroid.manager import AstroidManager
 from astroid.nodes import node_classes
 
+logger = logging.getLogger(__name__)
+
+
 _FunctionTypes = Union[
     types.FunctionType,
     types.MethodType,
@@ -471,7 +477,26 @@ def imported_member(self, node, member, name: str) -> bool:
             # check if it sounds valid and then add an import node, else use a
             # dummy node
             try:
-                getattr(sys.modules[modname], name)
+                with redirect_stderr(io.StringIO()) as stderr, redirect_stdout(
+                    io.StringIO()
+                ) as stdout:
+                    getattr(sys.modules[modname], name)
+                    stderr_value = stderr.getvalue()
+                    if stderr_value:
+                        logger.error(
+                            "Captured stderr while getting %s from %s:\n%s",
+                            name,
+                            sys.modules[modname],
+                            stderr_value,
+                        )
+                    stdout_value = stdout.getvalue()
+                    if stdout_value:
+                        logger.info(
+                            "Captured stdout while getting %s from %s:\n%s",
+                            name,
+                            sys.modules[modname],
+                            stdout_value,
+                        )
             except (KeyError, AttributeError):
                 attach_dummy_node(node, name, member)
             else:

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| astroid/raw_building.py | 10 | - | 10 | 9 | 3295
| astroid/raw_building.py | 22 | - | 10 | 9 | 3295
| astroid/raw_building.py | 474 | 476 | 38 | 9 | 11519


## Problem Statement

```
Deprecation warnings from numpy
### Steps to reproduce

1. Run pylint over the following test case:

\`\`\`
"""Test case"""

import numpy as np
value = np.random.seed(1234)
\`\`\`

### Current behavior
\`\`\`
/home/bje/source/nemo/myenv/lib/python3.10/site-packages/astroid/raw_building.py:470: FutureWarning: In the future `np.long` will be defined as the corresponding NumPy scalar.  (This may have returned Python scalars in past versions.
  getattr(sys.modules[modname], name)
/home/bje/source/nemo/myenv/lib/python3.10/site-packages/astroid/raw_building.py:470: FutureWarning: In the future `np.long` will be defined as the corresponding NumPy scalar.  (This may have returned Python scalars in past versions.
  getattr(sys.modules[modname], name)
\`\`\`

### Expected behavior
There should be no future warnings.

### python -c "from astroid import __pkginfo__; print(__pkginfo__.version)" output
2.12.13

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- |
| 1 | 1 astroid/decorators.py | 161 | 287| 1066 | 1066 | 
| 2 | 2 astroid/brain/brain_numpy_utils.py | 6 | 38| 241 | 1307 | 
| 3 | 3 astroid/util.py | 4 | 28| 130 | 1437 | 
| 4 | 4 astroid/exceptions.py | 6 | 46| 240 | 1677 | 
| 5 | 5 astroid/brain/brain_numpy_ndarray.py | 15 | 12| 200 | 1877 | 
| 6 | 5 astroid/exceptions.py | 378 | 398| 140 | 2017 | 
| 7 | 6 astroid/mixins.py | 6 | 30| 212 | 2229 | 
| 8 | 7 astroid/bases.py | 7 | 73| 424 | 2653 | 
| 9 | 8 astroid/nodes/scoped_nodes/scoped_nodes.py | 10 | 70| 426 | 3079 | 
| **-> 10 <-** | **9 astroid/raw_building.py** | 8 | 43| 216 | 3295 | 
| 11 | 10 astroid/_ast.py | 4 | 27| 138 | 3433 | 
| 12 | 11 astroid/nodes/node_classes.py | 6 | 81| 511 | 3944 | 
| 13 | 11 astroid/exceptions.py | 361 | 375| 113 | 4057 | 
| 14 | 12 astroid/const.py | 4 | 38| 221 | 4278 | 
| 15 | 12 astroid/util.py | 104 | 154| 377 | 4655 | 
| 16 | **12 astroid/raw_building.py** | 303 | 334| 241 | 4896 | 
| 17 | 13 astroid/modutils.py | 82 | 163| 696 | 5592 | 
| 18 | 14 astroid/brain/brain_numpy_ma.py | 6 | 31| 140 | 5732 | 
| 19 | 15 astroid/brain/brain_gi.py | 209 | 249| 248 | 5980 | 
| 20 | 16 astroid/brain/brain_argparse.py | 12 | 9| 286 | 6266 | 
| 21 | 16 astroid/exceptions.py | 278 | 300| 192 | 6458 | 
| 22 | 17 astroid/brain/brain_unittest.py | 11 | 8| 201 | 6659 | 
| 23 | 18 astroid/interpreter/_import/spec.py | 289 | 314| 189 | 6848 | 
| 24 | 19 astroid/brain/brain_namedtuple_enum.py | 6 | 62| 324 | 7172 | 
| 25 | 19 astroid/exceptions.py | 228 | 275| 438 | 7610 | 
| 26 | 19 astroid/interpreter/_import/spec.py | 4 | 56| 310 | 7920 | 
| 27 | 20 astroid/brain/brain_typing.py | 6 | 121| 654 | 8574 | 
| 28 | 20 astroid/brain/brain_namedtuple_enum.py | 279 | 309| 251 | 8825 | 
| 29 | 20 astroid/bases.py | 454 | 485| 254 | 9079 | 
| 30 | 20 astroid/decorators.py | 137 | 134| 209 | 9288 | 
| 31 | 20 astroid/_ast.py | 30 | 49| 187 | 9475 | 
| 32 | 21 astroid/interpreter/objectmodel.py | 700 | 792| 621 | 10096 | 
| 33 | 21 astroid/util.py | 69 | 101| 220 | 10316 | 
| 34 | 22 astroid/nodes/scoped_nodes/__init__.py | 11 | 45| 186 | 10502 | 
| 35 | 23 astroid/brain/brain_functools.py | 59 | 66| 115 | 10617 | 
| 36 | 24 astroid/helpers.py | 6 | 41| 252 | 10869 | 
| 37 | 24 astroid/exceptions.py | 73 | 98| 195 | 11064 | 
| **-> 38 <-** | **24 astroid/raw_building.py** | 442 | 489| 455 | 11519 | 
| 39 | 25 astroid/inference.py | 1205 | 1243| 370 | 11889 | 
| 40 | 25 astroid/exceptions.py | 140 | 137| 221 | 12110 | 
| 41 | 25 astroid/exceptions.py | 105 | 102| 173 | 12283 | 
| 42 | 26 astroid/brain/brain_random.py | 27 | 24| 263 | 12546 | 


### Hint

```
This seems very similar to https://github.com/PyCQA/astroid/pull/1514 that was fixed in 2.12.0.
I'm running 2.12.13 (> 2.12.0), so the fix isn't working in this case?
I don't know why #1514 did not fix this, I think we were capturing both stdout and stderr, so this will need some investigation. My guess would be that there's somewhere else to apply the same method to.
Hello, 
I see the same error with pylint on our tool [demcompare](https://github.com/CNES/demcompare). Pylint version:
\`\`\`
pylint --version
pylint 2.15.9
astroid 2.12.13
Python 3.8.10 (default, Nov 14 2022, 12:59:47) 
[GCC 9.4.0]
\`\`\`
I confirm the weird astroid lower warning and I don't know how to bypass it with pylint checking. 

\`\`\`
pylint demcompare 
/home/duboise/work/src/demcompare/venv/lib/python3.8/site-packages/astroid/raw_building.py:470: FutureWarning: In the future `np.long` will be defined as the corresponding NumPy scalar.  (This may have returned Python scalars in past versions.
  getattr(sys.modules[modname], name)
... (four times)
\`\`\`

Thanks in advance if there is a solution
Cordially

> Thanks in advance if there is a solution

while annoying the warning does not make pylint fail. Just ignore it. In a CI you can just check pylint return code. It will return 0 as expected
I agree, even if annoying because it feels our code as a problem somewhere, the CI with pylint doesn't fail indeed. Thanks for the answer that confirm to not bother for now. 
That might be fine in a CI environment, but for users, ultimately, ignoring warnings becomes difficult when there are too many such warnings. I would like to see this fixed.
Oh, it was not an argument in favour of not fixing it. It was just to point out that it is not a breaking problem. It is "just" a lot of quite annoying warnings. I am following the issue because it annoys me too. So I am in the same "I hope they will fix it" boat
> I don't know why https://github.com/PyCQA/astroid/pull/1514 did not fix this, I think we were capturing both stdout and stderr, so this will need some investigation. My guess would be that there's somewhere else to apply the same method to.

That PR only addressed import-time. This `FutureWarning` is emitted by numpy's package-level `__getattr__` method, not during import.
```

## Patch

```diff
diff --git a/astroid/raw_building.py b/astroid/raw_building.py
--- a/astroid/raw_building.py
+++ b/astroid/raw_building.py
@@ -10,11 +10,14 @@
 
 import builtins
 import inspect
+import io
+import logging
 import os
 import sys
 import types
 import warnings
 from collections.abc import Iterable
+from contextlib import redirect_stderr, redirect_stdout
 from typing import Any, Union
 
 from astroid import bases, nodes
@@ -22,6 +25,9 @@
 from astroid.manager import AstroidManager
 from astroid.nodes import node_classes
 
+logger = logging.getLogger(__name__)
+
+
 _FunctionTypes = Union[
     types.FunctionType,
     types.MethodType,
@@ -471,7 +477,26 @@ def imported_member(self, node, member, name: str) -> bool:
             # check if it sounds valid and then add an import node, else use a
             # dummy node
             try:
-                getattr(sys.modules[modname], name)
+                with redirect_stderr(io.StringIO()) as stderr, redirect_stdout(
+                    io.StringIO()
+                ) as stdout:
+                    getattr(sys.modules[modname], name)
+                    stderr_value = stderr.getvalue()
+                    if stderr_value:
+                        logger.error(
+                            "Captured stderr while getting %s from %s:\n%s",
+                            name,
+                            sys.modules[modname],
+                            stderr_value,
+                        )
+                    stdout_value = stdout.getvalue()
+                    if stdout_value:
+                        logger.info(
+                            "Captured stdout while getting %s from %s:\n%s",
+                            name,
+                            sys.modules[modname],
+                            stdout_value,
+                        )
             except (KeyError, AttributeError):
                 attach_dummy_node(node, name, member)
             else:

```

## Test Patch

```diff
diff --git a/tests/unittest_raw_building.py b/tests/unittest_raw_building.py
--- a/tests/unittest_raw_building.py
+++ b/tests/unittest_raw_building.py
@@ -8,8 +8,15 @@
 # For details: https://github.com/PyCQA/astroid/blob/main/LICENSE
 # Copyright (c) https://github.com/PyCQA/astroid/blob/main/CONTRIBUTORS.txt
 
+from __future__ import annotations
+
+import logging
+import os
+import sys
 import types
 import unittest
+from typing import Any
+from unittest import mock
 
 import _io
 import pytest
@@ -117,5 +124,45 @@ def test_module_object_with_broken_getattr(self) -> None:
         AstroidBuilder().inspect_build(fm_getattr, "test")
 
 
+@pytest.mark.skipif(
+    "posix" not in sys.builtin_module_names, reason="Platform doesn't support posix"
+)
+def test_build_module_getattr_catch_output(
+    capsys: pytest.CaptureFixture[str],
+    caplog: pytest.LogCaptureFixture,
+) -> None:
+    """Catch stdout and stderr in module __getattr__ calls when building a module.
+
+    Usually raised by DeprecationWarning or FutureWarning.
+    """
+    caplog.set_level(logging.INFO)
+    original_sys = sys.modules
+    original_module = sys.modules["posix"]
+    expected_out = "INFO (TEST): Welcome to posix!"
+    expected_err = "WARNING (TEST): Monkey-patched version of posix - module getattr"
+
+    class CustomGetattr:
+        def __getattr__(self, name: str) -> Any:
+            print(f"{expected_out}")
+            print(expected_err, file=sys.stderr)
+            return getattr(original_module, name)
+
+    def mocked_sys_modules_getitem(name: str) -> types.ModuleType | CustomGetattr:
+        if name != "posix":
+            return original_sys[name]
+        return CustomGetattr()
+
+    with mock.patch("astroid.raw_building.sys.modules") as sys_mock:
+        sys_mock.__getitem__.side_effect = mocked_sys_modules_getitem
+        builder = AstroidBuilder()
+        builder.inspect_build(os)
+
+    out, err = capsys.readouterr()
+    assert expected_out in caplog.text
+    assert expected_err in caplog.text
+    assert not out
+    assert not err
+
+
 if __name__ == "__main__":
     unittest.main()

```


## Code snippets

### 1 - astroid/decorators.py:

Start line: 161, End line: 287

```python
if util.check_warnings_filter():  # noqa: C901

    def deprecate_default_argument_values(
        astroid_version: str = "3.0", **arguments: str
    ) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
        """Decorator which emits a DeprecationWarning if any arguments specified
        are None or not passed at all.

        Arguments should be a key-value mapping, with the key being the argument to check
        and the value being a type annotation as string for the value of the argument.

        To improve performance, only used when DeprecationWarnings other than
        the default one are enabled.
        """
        # Helpful links
        # Decorator for DeprecationWarning: https://stackoverflow.com/a/49802489
        # Typing of stacked decorators: https://stackoverflow.com/a/68290080

        def deco(func: Callable[_P, _R]) -> Callable[_P, _R]:
            """Decorator function."""

            @functools.wraps(func)
            def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
                """Emit DeprecationWarnings if conditions are met."""

                keys = list(inspect.signature(func).parameters.keys())
                for arg, type_annotation in arguments.items():
                    try:
                        index = keys.index(arg)
                    except ValueError:
                        raise ValueError(
                            f"Can't find argument '{arg}' for '{args[0].__class__.__qualname__}'"
                        ) from None
                    if (
                        # Check kwargs
                        # - if found, check it's not None
                        (arg in kwargs and kwargs[arg] is None)
                        # Check args
                        # - make sure not in kwargs
                        # - len(args) needs to be long enough, if too short
                        #   arg can't be in args either
                        # - args[index] should not be None
                        or arg not in kwargs
                        and (
                            index == -1
                            or len(args) <= index
                            or (len(args) > index and args[index] is None)
                        )
                    ):
                        warnings.warn(
                            f"'{arg}' will be a required argument for "
                            f"'{args[0].__class__.__qualname__}.{func.__name__}'"
                            f" in astroid {astroid_version} "
                            f"('{arg}' should be of type: '{type_annotation}')",
                            DeprecationWarning,
                        )
                return func(*args, **kwargs)

            return wrapper

        return deco

    def deprecate_arguments(
        astroid_version: str = "3.0", **arguments: str
    ) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
        """Decorator which emits a DeprecationWarning if any arguments specified
        are passed.

        Arguments should be a key-value mapping, with the key being the argument to check
        and the value being a string that explains what to do instead of passing the argument.

        To improve performance, only used when DeprecationWarnings other than
        the default one are enabled.
        """

        def deco(func: Callable[_P, _R]) -> Callable[_P, _R]:
            @functools.wraps(func)
            def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
                keys = list(inspect.signature(func).parameters.keys())
                for arg, note in arguments.items():
                    try:
                        index = keys.index(arg)
                    except ValueError:
                        raise ValueError(
                            f"Can't find argument '{arg}' for '{args[0].__class__.__qualname__}'"
                        ) from None
                    if arg in kwargs or len(args) > index:
                        warnings.warn(
                            f"The argument '{arg}' for "
                            f"'{args[0].__class__.__qualname__}.{func.__name__}' is deprecated "
                            f"and will be removed in astroid {astroid_version} ({note})",
                            DeprecationWarning,
                        )
                return func(*args, **kwargs)

            return wrapper

        return deco

else:

    def deprecate_default_argument_values(
        astroid_version: str = "3.0", **arguments: str
    ) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
        """Passthrough decorator to improve performance if DeprecationWarnings are
        disabled.
        """

        def deco(func: Callable[_P, _R]) -> Callable[_P, _R]:
            """Decorator function."""
            return func

        return deco

    def deprecate_arguments(
        astroid_version: str = "3.0", **arguments: str
    ) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
        """Passthrough decorator to improve performance if DeprecationWarnings are
        disabled.
        """

        def deco(func: Callable[_P, _R]) -> Callable[_P, _R]:
            """Decorator function."""
            return func

        return deco
```
### 2 - astroid/brain/brain_numpy_utils.py:

Start line: 6, End line: 38

```python
from __future__ import annotations

from astroid.builder import extract_node
from astroid.context import InferenceContext
from astroid.nodes.node_classes import Attribute, Import, Name, NodeNG

# Class subscript is available in numpy starting with version 1.20.0
NUMPY_VERSION_TYPE_HINTS_SUPPORT = ("1", "20", "0")


def numpy_supports_type_hints() -> bool:
    """Returns True if numpy supports type hints."""
    np_ver = _get_numpy_version()
    return np_ver and np_ver > NUMPY_VERSION_TYPE_HINTS_SUPPORT


def _get_numpy_version() -> tuple[str, str, str]:
    """
    Return the numpy version number if numpy can be imported.

    Otherwise returns ('0', '0', '0')
    """
    try:
        import numpy  # pylint: disable=import-outside-toplevel

        return tuple(numpy.version.version.split("."))
    except (ImportError, AttributeError):
        return ("0", "0", "0")


def infer_numpy_member(src, node, context: InferenceContext | None = None):
    node = extract_node(src)
    return node.infer(context=context)
```
### 3 - astroid/util.py:

Start line: 4, End line: 28

```python
import importlib
import sys
import warnings
from typing import Any

import lazy_object_proxy

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal


def lazy_descriptor(obj):
    class DescriptorProxy(lazy_object_proxy.Proxy):
        def __get__(self, instance, owner=None):
            return self.__class__.__get__(self, instance)

    return DescriptorProxy(obj)


def lazy_import(module_name: str) -> lazy_object_proxy.Proxy:
    return lazy_object_proxy.Proxy(
        lambda: importlib.import_module("." + module_name, "astroid")
    )
```
### 4 - astroid/exceptions.py:

Start line: 6, End line: 46

```python
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from astroid import util

if TYPE_CHECKING:
    from astroid import arguments, bases, nodes, objects
    from astroid.context import InferenceContext

__all__ = (
    "AstroidBuildingError",
    "AstroidBuildingException",
    "AstroidError",
    "AstroidImportError",
    "AstroidIndexError",
    "AstroidSyntaxError",
    "AstroidTypeError",
    "AstroidValueError",
    "AttributeInferenceError",
    "BinaryOperationError",
    "DuplicateBasesError",
    "InconsistentMroError",
    "InferenceError",
    "InferenceOverwriteError",
    "MroError",
    "NameInferenceError",
    "NoDefault",
    "NotFoundError",
    "OperationError",
    "ParentMissingError",
    "ResolveError",
    "StatementMissing",
    "SuperArgumentTypeError",
    "SuperError",
    "TooManyLevelsError",
    "UnaryOperationError",
    "UnresolvableName",
    "UseInferenceDefault",
)
```
### 5 - astroid/brain/brain_numpy_ndarray.py:

Start line: 15, End line: 12

```python
from __future__ import annotations

from astroid.brain.brain_numpy_utils import numpy_supports_type_hints
from astroid.builder import extract_node
from astroid.context import InferenceContext
from astroid.inference_tip import inference_tip
from astroid.manager import AstroidManager
from astroid.nodes.node_classes import Attribute


def infer_numpy_ndarray(node, context: InferenceContext | None = None):
    ndarray =
 # ... other code
    if numpy_supports_type_hints():
        ndarray += """
        @classmethod
        def __class_getitem__(cls, value):
            return cls
        """
    node = extract_node(ndarray)
    return node.infer(context=context)


def _looks_like_numpy_ndarray(node) -> bool:
    return isinstance(node, Attribute) and node.attrname == "ndarray"


AstroidManager().register_transform(
    Attribute,
    inference_tip(infer_numpy_ndarray),
    _looks_like_numpy_ndarray,
)
```
### 6 - astroid/exceptions.py:

Start line: 378, End line: 398

```python
class AstroidValueError(AstroidError):
    """Raised when a ValueError would be expected in Python code."""


class InferenceOverwriteError(AstroidError):
    """Raised when an inference tip is overwritten.

    Currently only used for debugging.
    """


class ParentMissingError(AstroidError):
    """Raised when a node which is expected to have a parent attribute is missing one.

    Standard attributes:
        target: The node for which the parent lookup failed.
    """

    def __init__(self, target: nodes.NodeNG) -> None:
        self.target = target
        super().__init__(message=f"Parent not found on {target!r}.")
```
### 7 - astroid/mixins.py:

Start line: 6, End line: 30

```python
import warnings

from astroid.nodes._base_nodes import AssignTypeNode as AssignTypeMixin
from astroid.nodes._base_nodes import FilterStmtsBaseNode as FilterStmtsMixin
from astroid.nodes._base_nodes import ImportNode as ImportFromMixin
from astroid.nodes._base_nodes import MultiLineBlockNode as MultiLineBlockMixin
from astroid.nodes._base_nodes import MultiLineWithElseBlockNode as BlockRangeMixIn
from astroid.nodes._base_nodes import NoChildrenNode as NoChildrenMixin
from astroid.nodes._base_nodes import ParentAssignNode as ParentAssignTypeMixin

__all__ = (
    "AssignTypeMixin",
    "BlockRangeMixIn",
    "FilterStmtsMixin",
    "ImportFromMixin",
    "MultiLineBlockMixin",
    "NoChildrenMixin",
    "ParentAssignTypeMixin",
)

warnings.warn(
    "The 'astroid.mixins' module is deprecated and will become private in astroid 3.0.0",
    DeprecationWarning,
)
```
### 8 - astroid/bases.py:

Start line: 7, End line: 73

```python
from __future__ import annotations

import collections
import collections.abc
import sys
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar

from astroid import decorators, nodes
from astroid.const import PY310_PLUS
from astroid.context import (
    CallContext,
    InferenceContext,
    bind_context_to_node,
    copy_context,
)
from astroid.exceptions import (
    AstroidTypeError,
    AttributeInferenceError,
    InferenceError,
    NameInferenceError,
)
from astroid.typing import InferBinaryOp, InferenceErrorInfo, InferenceResult
from astroid.util import Uninferable, lazy_descriptor, lazy_import

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

if TYPE_CHECKING:
    from astroid.constraint import Constraint

objectmodel = lazy_import("interpreter.objectmodel")
helpers = lazy_import("helpers")
manager = lazy_import("manager")


# TODO: check if needs special treatment
BOOL_SPECIAL_METHOD = "__bool__"
BUILTINS = "builtins"  # TODO Remove in 2.8

PROPERTIES = {"builtins.property", "abc.abstractproperty"}
if PY310_PLUS:
    PROPERTIES.add("enum.property")

# List of possible property names. We use this list in order
# to see if a method is a property or not. This should be
# pretty reliable and fast, the alternative being to check each
# decorator to see if its a real property-like descriptor, which
# can be too complicated.
# Also, these aren't qualified, because each project can
# define them, we shouldn't expect to know every possible
# property-like decorator!
POSSIBLE_PROPERTIES = {
    "cached_property",
    "cachedproperty",
    "lazyproperty",
    "lazy_property",
    "reify",
    "lazyattribute",
    "lazy_attribute",
    "LazyProperty",
    "lazy",
    "cache_readonly",
    "DynamicClassAttribute",
}
```
### 9 - astroid/nodes/scoped_nodes/scoped_nodes.py:

Start line: 10, End line: 70

```python
from __future__ import annotations

import io
import itertools
import os
import sys
import warnings
from collections.abc import Generator, Iterator
from functools import lru_cache
from typing import TYPE_CHECKING, ClassVar, NoReturn, TypeVar, overload

from astroid import bases
from astroid import decorators as decorators_mod
from astroid import util
from astroid.const import IS_PYPY, PY38, PY38_PLUS, PY39_PLUS
from astroid.context import (
    CallContext,
    InferenceContext,
    bind_context_to_node,
    copy_context,
)
from astroid.exceptions import (
    AstroidBuildingError,
    AstroidTypeError,
    AttributeInferenceError,
    DuplicateBasesError,
    InconsistentMroError,
    InferenceError,
    MroError,
    StatementMissing,
    TooManyLevelsError,
)
from astroid.interpreter.dunder_lookup import lookup
from astroid.interpreter.objectmodel import ClassModel, FunctionModel, ModuleModel
from astroid.manager import AstroidManager
from astroid.nodes import Arguments, Const, NodeNG, _base_nodes, node_classes
from astroid.nodes.scoped_nodes.mixin import ComprehensionScope, LocalsDictNodeNG
from astroid.nodes.scoped_nodes.utils import builtin_lookup
from astroid.nodes.utils import Position
from astroid.typing import InferBinaryOp, InferenceResult, SuccessfulInferenceResult

if sys.version_info >= (3, 8):
    from functools import cached_property
    from typing import Literal
else:
    from typing_extensions import Literal

    from astroid.decorators import cachedproperty as cached_property

if TYPE_CHECKING:
    from astroid import nodes


ITER_METHODS = ("__iter__", "__getitem__")
EXCEPTION_BASE_CLASSES = frozenset({"Exception", "BaseException"})
objects = util.lazy_import("objects")
BUILTIN_DESCRIPTORS = frozenset(
    {"classmethod", "staticmethod", "builtins.classmethod", "builtins.staticmethod"}
)

_T = TypeVar("_T")
```
### 10 - astroid/raw_building.py:

Start line: 8, End line: 43

```python
from __future__ import annotations

import builtins
import inspect
import os
import sys
import types
import warnings
from collections.abc import Iterable
from typing import Any, Union

from astroid import bases, nodes
from astroid.const import _EMPTY_OBJECT_MARKER, IS_PYPY
from astroid.manager import AstroidManager
from astroid.nodes import node_classes

_FunctionTypes = Union[
    types.FunctionType,
    types.MethodType,
    types.BuiltinFunctionType,
    types.WrapperDescriptorType,
    types.MethodDescriptorType,
    types.ClassMethodDescriptorType,
]

# the keys of CONST_CLS eg python builtin types
_CONSTANTS = tuple(node_classes.CONST_CLS)
_BUILTINS = vars(builtins)
TYPE_NONE = type(None)
TYPE_NOTIMPLEMENTED = type(NotImplemented)
TYPE_ELLIPSIS = type(...)


def _attach_local_node(parent, node, name: str) -> None:
    node.name = name  # needed by add_local_node
    parent.add_local_node(node)
```
### 16 - astroid/raw_building.py:

Start line: 303, End line: 334

```python
def _build_from_function(
    node: nodes.Module | nodes.ClassDef,
    name: str,
    member: _FunctionTypes,
    module: types.ModuleType,
) -> None:
    # verify this is not an imported function
    try:
        code = member.__code__  # type: ignore[union-attr]
    except AttributeError:
        # Some implementations don't provide the code object,
        # such as Jython.
        code = None
    filename = getattr(code, "co_filename", None)
    if filename is None:
        assert isinstance(member, object)
        object_build_methoddescriptor(node, member, name)
    elif filename != getattr(module, "__file__", None):
        attach_dummy_node(node, name, member)
    else:
        object_build_function(node, member, name)


def _safe_has_attribute(obj, member: str) -> bool:
    """Required because unexpected RunTimeError can be raised.

    See https://github.com/PyCQA/astroid/issues/1958
    """
    try:
        return hasattr(obj, member)
    except Exception:  # pylint: disable=broad-except
        return False
```
### 38 - astroid/raw_building.py:

Start line: 442, End line: 489

```python
class InspectBuilder:

    def imported_member(self, node, member, name: str) -> bool:
        """verify this is not an imported class or handle it"""
        # /!\ some classes like ExtensionClass doesn't have a __module__
        # attribute ! Also, this may trigger an exception on badly built module
        # (see http://www.logilab.org/ticket/57299 for instance)
        try:
            modname = getattr(member, "__module__", None)
        except TypeError:
            modname = None
        if modname is None:
            if name in {"__new__", "__subclasshook__"}:
                # Python 2.5.1 (r251:54863, Sep  1 2010, 22:03:14)
                # >>> print object.__new__.__module__
                # None
                modname = builtins.__name__
            else:
                attach_dummy_node(node, name, member)
                return True

        # On PyPy during bootstrapping we infer _io while _module is
        # builtins. In CPython _io names itself io, see http://bugs.python.org/issue18602
        # Therefore, this basically checks whether we are not in PyPy.
        if modname == "_io" and not self._module.__name__ == "builtins":
            return False

        real_name = {"gtk": "gtk_gtk"}.get(modname, modname)

        if real_name != self._module.__name__:
            # check if it sounds valid and then add an import node, else use a
            # dummy node
            try:
                getattr(sys.modules[modname], name)
            except (KeyError, AttributeError):
                attach_dummy_node(node, name, member)
            else:
                attach_import_node(node, modname, name)
            return True
        return False


# astroid bootstrapping ######################################################

_CONST_PROXY: dict[type, nodes.ClassDef] = {}


def _set_proxied(const) -> nodes.ClassDef:
    # TODO : find a nicer way to handle this situation;
    return _CONST_PROXY[const.value.__class__]
```
