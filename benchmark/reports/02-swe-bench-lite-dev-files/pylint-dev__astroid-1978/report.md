# pylint-dev__astroid-1978

| **pylint-dev/astroid** | `0c9ab0fe56703fa83c73e514a1020d398d23fa7f` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 45004 |
| **Any found context length** | 45004 |
| **Avg pos** | 30.0 |
| **Min pos** | 10 |
| **Max pos** | 10 |
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
| astroid/raw_building.py | 10 | - | 10 | 9 | 45004
| astroid/raw_building.py | 22 | - | 10 | 9 | 45004
| astroid/raw_building.py | 474 | 476 | 10 | 9 | 45004


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
| 1 | 1 astroid/decorators.py | 0 | 288| 2164 | 2164 | 
| 2 | 2 astroid/brain/brain_numpy_utils.py | 0 | 87| 635 | 2799 | 
| 3 | 3 astroid/util.py | 0 | 155| 1023 | 3822 | 
| 4 | 4 astroid/exceptions.py | 0 | 427| 3041 | 6863 | 
| 5 | 5 astroid/brain/brain_numpy_ndarray.py | 0 | 163| 2434 | 9297 | 
| 6 | 5 astroid/exceptions.py | 0 | 427| 3041 | 12338 | 
| 7 | 6 astroid/mixins.py | 0 | 31| 291 | 12629 | 
| 8 | 7 astroid/bases.py | 0 | 672| 5280 | 17909 | 
| 9 | 8 astroid/nodes/scoped_nodes/scoped_nodes.py | 0 | 3077| 22484 | 40393 | 
| **-> 10 <-** | **9 astroid/raw_building.py** | 0 | 580| 4611 | 45004 | 


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

```python
# Licensed under the LGPL: https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html
# For details: https://github.com/PyCQA/astroid/blob/main/LICENSE
# Copyright (c) https://github.com/PyCQA/astroid/blob/main/CONTRIBUTORS.txt

"""A few useful function/method decorators."""

from __future__ import annotations

import functools
import inspect
import sys
import warnings
from collections.abc import Callable, Generator
from typing import TypeVar

import wrapt

from astroid import _cache, util
from astroid.context import InferenceContext
from astroid.exceptions import InferenceError

if sys.version_info >= (3, 10):
    from typing import ParamSpec
else:
    from typing_extensions import ParamSpec

_R = TypeVar("_R")
_P = ParamSpec("_P")


@wrapt.decorator
def cached(func, instance, args, kwargs):
    """Simple decorator to cache result of method calls without args."""
    cache = getattr(instance, "__cache", None)
    if cache is None:
        instance.__cache = cache = {}
        _cache.CACHE_MANAGER.add_dict_cache(cache)
    try:
        return cache[func]
    except KeyError:
        cache[func] = result = func(*args, **kwargs)
        return result


# TODO: Remove when support for 3.7 is dropped
# TODO: astroid 3.0 -> move class behind sys.version_info < (3, 8) guard
class cachedproperty:
    """Provides a cached property equivalent to the stacking of
    @cached and @property, but more efficient.

    After first usage, the <property_name> becomes part of the object's
    __dict__. Doing:

      del obj.<property_name> empties the cache.

    Idea taken from the pyramid_ framework and the mercurial_ project.

    .. _pyramid: http://pypi.python.org/pypi/pyramid
    .. _mercurial: http://pypi.python.org/pypi/Mercurial
    """

    __slots__ = ("wrapped",)

    def __init__(self, wrapped):
        if sys.version_info >= (3, 8):
            warnings.warn(
                "cachedproperty has been deprecated and will be removed in astroid 3.0 for Python 3.8+. "
                "Use functools.cached_property instead.",
                DeprecationWarning,
            )
        try:
            wrapped.__name__
        except AttributeError as exc:
            raise TypeError(f"{wrapped} must have a __name__ attribute") from exc
        self.wrapped = wrapped

    @property
    def __doc__(self):
        doc = getattr(self.wrapped, "__doc__", None)
        return "<wrapped by the cachedproperty decorator>%s" % (
            "\n%s" % doc if doc else ""
        )

    def __get__(self, inst, objtype=None):
        if inst is None:
            return self
        val = self.wrapped(inst)
        setattr(inst, self.wrapped.__name__, val)
        return val


def path_wrapper(func):
    """Return the given infer function wrapped to handle the path.

    Used to stop inference if the node has already been looked
    at for a given `InferenceContext` to prevent infinite recursion
    """

    @functools.wraps(func)
    def wrapped(
        node, context: InferenceContext | None = None, _func=func, **kwargs
    ) -> Generator:
        """Wrapper function handling context."""
        if context is None:
            context = InferenceContext()
        if context.push(node):
            return

        yielded = set()

        for res in _func(node, context, **kwargs):
            # unproxy only true instance, not const, tuple, dict...
            if res.__class__.__name__ == "Instance":
                ares = res._proxied
            else:
                ares = res
            if ares not in yielded:
                yield res
                yielded.add(ares)

    return wrapped


@wrapt.decorator
def yes_if_nothing_inferred(func, instance, args, kwargs):
    generator = func(*args, **kwargs)

    try:
        yield next(generator)
    except StopIteration:
        # generator is empty
        yield util.Uninferable
        return

    yield from generator


@wrapt.decorator
def raise_if_nothing_inferred(func, instance, args, kwargs):
    generator = func(*args, **kwargs)
    try:
        yield next(generator)
    except StopIteration as error:
        # generator is empty
        if error.args:
            # pylint: disable=not-a-mapping
            raise InferenceError(**error.args[0]) from error
        raise InferenceError(
            "StopIteration raised without any error information."
        ) from error
    except RecursionError as error:
        raise InferenceError(
            f"RecursionError raised with limit {sys.getrecursionlimit()}."
        ) from error

    yield from generator


# Expensive decorators only used to emit Deprecation warnings.
# If no other than the default DeprecationWarning are enabled,
# fall back to passthrough implementations.
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

```python
# Licensed under the LGPL: https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html
# For details: https://github.com/PyCQA/astroid/blob/main/LICENSE
# Copyright (c) https://github.com/PyCQA/astroid/blob/main/CONTRIBUTORS.txt

"""Different utilities for the numpy brains."""

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


def _is_a_numpy_module(node: Name) -> bool:
    """
    Returns True if the node is a representation of a numpy module.

    For example in :
        import numpy as np
        x = np.linspace(1, 2)
    The node <Name.np> is a representation of the numpy module.

    :param node: node to test
    :return: True if the node is a representation of the numpy module.
    """
    module_nickname = node.name
    potential_import_target = [
        x for x in node.lookup(module_nickname)[1] if isinstance(x, Import)
    ]
    return any(
        ("numpy", module_nickname) in target.names or ("numpy", None) in target.names
        for target in potential_import_target
    )


def looks_like_numpy_member(member_name: str, node: NodeNG) -> bool:
    """
    Returns True if the node is a member of numpy whose
    name is member_name.

    :param member_name: name of the member
    :param node: node to test
    :return: True if the node is a member of numpy
    """
    if (
        isinstance(node, Attribute)
        and node.attrname == member_name
        and isinstance(node.expr, Name)
        and _is_a_numpy_module(node.expr)
    ):
        return True
    if (
        isinstance(node, Name)
        and node.name == member_name
        and node.root().name.startswith("numpy")
    ):
        return True
    return False

```
### 3 - astroid/util.py:

```python
# Licensed under the LGPL: https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html
# For details: https://github.com/PyCQA/astroid/blob/main/LICENSE
# Copyright (c) https://github.com/PyCQA/astroid/blob/main/CONTRIBUTORS.txt

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


@object.__new__
class Uninferable:
    """Special inference object, which is returned when inference fails."""

    def __repr__(self) -> str:
        return "Uninferable"

    __str__ = __repr__

    def __getattribute__(self, name: str) -> Any:
        if name == "next":
            raise AttributeError("next method should not be called")
        if name.startswith("__") and name.endswith("__"):
            return object.__getattribute__(self, name)
        if name == "accept":
            return object.__getattribute__(self, name)
        return self

    def __call__(self, *args, **kwargs):
        return self

    def __bool__(self) -> Literal[False]:
        return False

    __nonzero__ = __bool__

    def accept(self, visitor):
        return visitor.visit_uninferable(self)


class BadOperationMessage:
    """Object which describes a TypeError occurred somewhere in the inference chain.

    This is not an exception, but a container object which holds the types and
    the error which occurred.
    """


class BadUnaryOperationMessage(BadOperationMessage):
    """Object which describes operational failures on UnaryOps."""

    def __init__(self, operand, op, error):
        self.operand = operand
        self.op = op
        self.error = error

    @property
    def _object_type_helper(self):
        helpers = lazy_import("helpers")
        return helpers.object_type

    def _object_type(self, obj):
        objtype = self._object_type_helper(obj)
        if objtype is Uninferable:
            return None

        return objtype

    def __str__(self) -> str:
        if hasattr(self.operand, "name"):
            operand_type = self.operand.name
        else:
            object_type = self._object_type(self.operand)
            if hasattr(object_type, "name"):
                operand_type = object_type.name
            else:
                # Just fallback to as_string
                operand_type = object_type.as_string()

        msg = "bad operand type for unary {}: {}"
        return msg.format(self.op, operand_type)


class BadBinaryOperationMessage(BadOperationMessage):
    """Object which describes type errors for BinOps."""

    def __init__(self, left_type, op, right_type):
        self.left_type = left_type
        self.right_type = right_type
        self.op = op

    def __str__(self) -> str:
        msg = "unsupported operand type(s) for {}: {!r} and {!r}"
        return msg.format(self.op, self.left_type.name, self.right_type.name)


def _instancecheck(cls, other) -> bool:
    wrapped = cls.__wrapped__
    other_cls = other.__class__
    is_instance_of = wrapped is other_cls or issubclass(other_cls, wrapped)
    warnings.warn(
        "%r is deprecated and slated for removal in astroid "
        "2.0, use %r instead" % (cls.__class__.__name__, wrapped.__name__),
        PendingDeprecationWarning,
        stacklevel=2,
    )
    return is_instance_of


def proxy_alias(alias_name, node_type):
    """Get a Proxy from the given name to the given node type."""
    proxy = type(
        alias_name,
        (lazy_object_proxy.Proxy,),
        {
            "__class__": object.__dict__["__class__"],
            "__instancecheck__": _instancecheck,
        },
    )
    return proxy(lambda: node_type)


def check_warnings_filter() -> bool:
    """Return True if any other than the default DeprecationWarning filter is enabled.

    https://docs.python.org/3/library/warnings.html#default-warning-filter
    """
    return any(
        issubclass(DeprecationWarning, filter[2])
        and filter[0] != "ignore"
        and filter[3] != "__main__"
        for filter in warnings.filters
    )

```
### 4 - astroid/exceptions.py:

```python
# Licensed under the LGPL: https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html
# For details: https://github.com/PyCQA/astroid/blob/main/LICENSE
# Copyright (c) https://github.com/PyCQA/astroid/blob/main/CONTRIBUTORS.txt

"""This module contains exceptions used in the astroid library."""

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


class AstroidError(Exception):
    """Base exception class for all astroid related exceptions.

    AstroidError and its subclasses are structured, intended to hold
    objects representing state when the exception is thrown.  Field
    values are passed to the constructor as keyword-only arguments.
    Each subclass has its own set of standard fields, but use your
    best judgment to decide whether a specific exception instance
    needs more or fewer fields for debugging.  Field values may be
    used to lazily generate the error message: self.message.format()
    will be called with the field names and values supplied as keyword
    arguments.
    """

    def __init__(self, message: str = "", **kws: Any) -> None:
        super().__init__(message)
        self.message = message
        for key, value in kws.items():
            setattr(self, key, value)

    def __str__(self) -> str:
        return self.message.format(**vars(self))


class AstroidBuildingError(AstroidError):
    """Exception class when we are unable to build an astroid representation.

    Standard attributes:
        modname: Name of the module that AST construction failed for.
        error: Exception raised during construction.
    """

    def __init__(
        self,
        message: str = "Failed to import module {modname}.",
        modname: str | None = None,
        error: Exception | None = None,
        source: str | None = None,
        path: str | None = None,
        cls: type | None = None,
        class_repr: str | None = None,
        **kws: Any,
    ) -> None:
        self.modname = modname
        self.error = error
        self.source = source
        self.path = path
        self.cls = cls
        self.class_repr = class_repr
        super().__init__(message, **kws)


class AstroidImportError(AstroidBuildingError):
    """Exception class used when a module can't be imported by astroid."""


class TooManyLevelsError(AstroidImportError):
    """Exception class which is raised when a relative import was beyond the top-level.

    Standard attributes:
        level: The level which was attempted.
        name: the name of the module on which the relative import was attempted.
    """

    def __init__(
        self,
        message: str = "Relative import with too many levels "
        "({level}) for module {name!r}",
        level: int | None = None,
        name: str | None = None,
        **kws: Any,
    ) -> None:
        self.level = level
        self.name = name
        super().__init__(message, **kws)


class AstroidSyntaxError(AstroidBuildingError):
    """Exception class used when a module can't be parsed."""

    def __init__(
        self,
        message: str,
        modname: str | None,
        error: Exception,
        path: str | None,
        source: str | None = None,
    ) -> None:
        super().__init__(message, modname, error, source, path)


class NoDefault(AstroidError):
    """Raised by function's `default_value` method when an argument has
    no default value.

    Standard attributes:
        func: Function node.
        name: Name of argument without a default.
    """

    def __init__(
        self,
        message: str = "{func!r} has no default for {name!r}.",
        func: nodes.FunctionDef | None = None,
        name: str | None = None,
        **kws: Any,
    ) -> None:
        self.func = func
        self.name = name
        super().__init__(message, **kws)


class ResolveError(AstroidError):
    """Base class of astroid resolution/inference error.

    ResolveError is not intended to be raised.

    Standard attributes:
        context: InferenceContext object.
    """

    def __init__(
        self, message: str = "", context: InferenceContext | None = None, **kws: Any
    ) -> None:
        self.context = context
        super().__init__(message, **kws)


class MroError(ResolveError):
    """Error raised when there is a problem with method resolution of a class.

    Standard attributes:
        mros: A sequence of sequences containing ClassDef nodes.
        cls: ClassDef node whose MRO resolution failed.
        context: InferenceContext object.
    """

    def __init__(
        self,
        message: str,
        mros: list[nodes.ClassDef],
        cls: nodes.ClassDef,
        context: InferenceContext | None = None,
        **kws: Any,
    ) -> None:
        self.mros = mros
        self.cls = cls
        self.context = context
        super().__init__(message, **kws)

    def __str__(self) -> str:
        mro_names = ", ".join(f"({', '.join(b.name for b in m)})" for m in self.mros)
        return self.message.format(mros=mro_names, cls=self.cls)


class DuplicateBasesError(MroError):
    """Error raised when there are duplicate bases in the same class bases."""


class InconsistentMroError(MroError):
    """Error raised when a class's MRO is inconsistent."""


class SuperError(ResolveError):
    """Error raised when there is a problem with a *super* call.

    Standard attributes:
        *super_*: The Super instance that raised the exception.
        context: InferenceContext object.
    """

    def __init__(self, message: str, super_: objects.Super, **kws: Any) -> None:
        self.super_ = super_
        super().__init__(message, **kws)

    def __str__(self) -> str:
        return self.message.format(**vars(self.super_))


class InferenceError(ResolveError):  # pylint: disable=too-many-instance-attributes
    """Raised when we are unable to infer a node.

    Standard attributes:
        node: The node inference was called on.
        context: InferenceContext object.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        message: str = "Inference failed for {node!r}.",
        node: nodes.NodeNG | bases.Instance | None = None,
        context: InferenceContext | None = None,
        target: nodes.NodeNG | bases.Instance | None = None,
        targets: nodes.Tuple | None = None,
        attribute: str | None = None,
        unknown: nodes.NodeNG | bases.Instance | None = None,
        assign_path: list[int] | None = None,
        caller: nodes.Call | None = None,
        stmts: Sequence[nodes.NodeNG | bases.Instance] | None = None,
        frame: nodes.LocalsDictNodeNG | None = None,
        call_site: arguments.CallSite | None = None,
        func: nodes.FunctionDef | None = None,
        arg: str | None = None,
        positional_arguments: list | None = None,
        unpacked_args: list | None = None,
        keyword_arguments: dict | None = None,
        unpacked_kwargs: dict | None = None,
        **kws: Any,
    ) -> None:
        self.node = node
        self.context = context
        self.target = target
        self.targets = targets
        self.attribute = attribute
        self.unknown = unknown
        self.assign_path = assign_path
        self.caller = caller
        self.stmts = stmts
        self.frame = frame
        self.call_site = call_site
        self.func = func
        self.arg = arg
        self.positional_arguments = positional_arguments
        self.unpacked_args = unpacked_args
        self.keyword_arguments = keyword_arguments
        self.unpacked_kwargs = unpacked_kwargs
        super().__init__(message, **kws)


# Why does this inherit from InferenceError rather than ResolveError?
# Changing it causes some inference tests to fail.
class NameInferenceError(InferenceError):
    """Raised when a name lookup fails, corresponds to NameError.

    Standard attributes:
        name: The name for which lookup failed, as a string.
        scope: The node representing the scope in which the lookup occurred.
        context: InferenceContext object.
    """

    def __init__(
        self,
        message: str = "{name!r} not found in {scope!r}.",
        name: str | None = None,
        scope: nodes.LocalsDictNodeNG | None = None,
        context: InferenceContext | None = None,
        **kws: Any,
    ) -> None:
        self.name = name
        self.scope = scope
        self.context = context
        super().__init__(message, **kws)


class AttributeInferenceError(ResolveError):
    """Raised when an attribute lookup fails, corresponds to AttributeError.

    Standard attributes:
        target: The node for which lookup failed.
        attribute: The attribute for which lookup failed, as a string.
        context: InferenceContext object.
    """

    def __init__(
        self,
        message: str = "{attribute!r} not found on {target!r}.",
        attribute: str = "",
        target: nodes.NodeNG | bases.Instance | None = None,
        context: InferenceContext | None = None,
        mros: list[nodes.ClassDef] | None = None,
        super_: nodes.ClassDef | None = None,
        cls: nodes.ClassDef | None = None,
        **kws: Any,
    ) -> None:
        self.attribute = attribute
        self.target = target
        self.context = context
        self.mros = mros
        self.super_ = super_
        self.cls = cls
        super().__init__(message, **kws)


class UseInferenceDefault(Exception):
    """Exception to be raised in custom inference function to indicate that it
    should go back to the default behaviour.
    """


class _NonDeducibleTypeHierarchy(Exception):
    """Raised when is_subtype / is_supertype can't deduce the relation between two
    types.
    """


class AstroidIndexError(AstroidError):
    """Raised when an Indexable / Mapping does not have an index / key."""

    def __init__(
        self,
        message: str = "",
        node: nodes.NodeNG | bases.Instance | None = None,
        index: nodes.Subscript | None = None,
        context: InferenceContext | None = None,
        **kws: Any,
    ) -> None:
        self.node = node
        self.index = index
        self.context = context
        super().__init__(message, **kws)


class AstroidTypeError(AstroidError):
    """Raised when a TypeError would be expected in Python code."""

    def __init__(
        self,
        message: str = "",
        node: nodes.NodeNG | bases.Instance | None = None,
        index: nodes.Subscript | None = None,
        context: InferenceContext | None = None,
        **kws: Any,
    ) -> None:
        self.node = node
        self.index = index
        self.context = context
        super().__init__(message, **kws)


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


class StatementMissing(ParentMissingError):
    """Raised when a call to node.statement() does not return a node.

    This is because a node in the chain does not have a parent attribute
    and therefore does not return a node for statement().

    Standard attributes:
        target: The node for which the parent lookup failed.
    """

    def __init__(self, target: nodes.NodeNG) -> None:
        super(ParentMissingError, self).__init__(
            message=f"Statement not found on {target!r}"
        )


# Backwards-compatibility aliases
OperationError = util.BadOperationMessage
UnaryOperationError = util.BadUnaryOperationMessage
BinaryOperationError = util.BadBinaryOperationMessage

SuperArgumentTypeError = SuperError
UnresolvableName = NameInferenceError
NotFoundError = AttributeInferenceError
AstroidBuildingException = AstroidBuildingError

```
### 5 - astroid/brain/brain_numpy_ndarray.py:

```python
# Licensed under the LGPL: https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html
# For details: https://github.com/PyCQA/astroid/blob/main/LICENSE
# Copyright (c) https://github.com/PyCQA/astroid/blob/main/CONTRIBUTORS.txt

"""Astroid hooks for numpy ndarray class."""
from __future__ import annotations

from astroid.brain.brain_numpy_utils import numpy_supports_type_hints
from astroid.builder import extract_node
from astroid.context import InferenceContext
from astroid.inference_tip import inference_tip
from astroid.manager import AstroidManager
from astroid.nodes.node_classes import Attribute


def infer_numpy_ndarray(node, context: InferenceContext | None = None):
    ndarray = """
    class ndarray(object):
        def __init__(self, shape, dtype=float, buffer=None, offset=0,
                     strides=None, order=None):
            self.T = numpy.ndarray([0, 0])
            self.base = None
            self.ctypes = None
            self.data = None
            self.dtype = None
            self.flags = None
            # Should be a numpy.flatiter instance but not available for now
            # Putting an array instead so that iteration and indexing are authorized
            self.flat = np.ndarray([0, 0])
            self.imag = np.ndarray([0, 0])
            self.itemsize = None
            self.nbytes = None
            self.ndim = None
            self.real = np.ndarray([0, 0])
            self.shape = numpy.ndarray([0, 0])
            self.size = None
            self.strides = None

        def __abs__(self): return numpy.ndarray([0, 0])
        def __add__(self, value): return numpy.ndarray([0, 0])
        def __and__(self, value): return numpy.ndarray([0, 0])
        def __array__(self, dtype=None): return numpy.ndarray([0, 0])
        def __array_wrap__(self, obj): return numpy.ndarray([0, 0])
        def __contains__(self, key): return True
        def __copy__(self): return numpy.ndarray([0, 0])
        def __deepcopy__(self, memo): return numpy.ndarray([0, 0])
        def __divmod__(self, value): return (numpy.ndarray([0, 0]), numpy.ndarray([0, 0]))
        def __eq__(self, value): return numpy.ndarray([0, 0])
        def __float__(self): return 0.
        def __floordiv__(self): return numpy.ndarray([0, 0])
        def __ge__(self, value): return numpy.ndarray([0, 0])
        def __getitem__(self, key): return uninferable
        def __gt__(self, value): return numpy.ndarray([0, 0])
        def __iadd__(self, value): return numpy.ndarray([0, 0])
        def __iand__(self, value): return numpy.ndarray([0, 0])
        def __ifloordiv__(self, value): return numpy.ndarray([0, 0])
        def __ilshift__(self, value): return numpy.ndarray([0, 0])
        def __imod__(self, value): return numpy.ndarray([0, 0])
        def __imul__(self, value): return numpy.ndarray([0, 0])
        def __int__(self): return 0
        def __invert__(self): return numpy.ndarray([0, 0])
        def __ior__(self, value): return numpy.ndarray([0, 0])
        def __ipow__(self, value): return numpy.ndarray([0, 0])
        def __irshift__(self, value): return numpy.ndarray([0, 0])
        def __isub__(self, value): return numpy.ndarray([0, 0])
        def __itruediv__(self, value): return numpy.ndarray([0, 0])
        def __ixor__(self, value): return numpy.ndarray([0, 0])
        def __le__(self, value): return numpy.ndarray([0, 0])
        def __len__(self): return 1
        def __lshift__(self, value): return numpy.ndarray([0, 0])
        def __lt__(self, value): return numpy.ndarray([0, 0])
        def __matmul__(self, value): return numpy.ndarray([0, 0])
        def __mod__(self, value): return numpy.ndarray([0, 0])
        def __mul__(self, value): return numpy.ndarray([0, 0])
        def __ne__(self, value): return numpy.ndarray([0, 0])
        def __neg__(self): return numpy.ndarray([0, 0])
        def __or__(self, value): return numpy.ndarray([0, 0])
        def __pos__(self): return numpy.ndarray([0, 0])
        def __pow__(self): return numpy.ndarray([0, 0])
        def __repr__(self): return str()
        def __rshift__(self): return numpy.ndarray([0, 0])
        def __setitem__(self, key, value): return uninferable
        def __str__(self): return str()
        def __sub__(self, value): return numpy.ndarray([0, 0])
        def __truediv__(self, value): return numpy.ndarray([0, 0])
        def __xor__(self, value): return numpy.ndarray([0, 0])
        def all(self, axis=None, out=None, keepdims=False): return np.ndarray([0, 0])
        def any(self, axis=None, out=None, keepdims=False): return np.ndarray([0, 0])
        def argmax(self, axis=None, out=None): return np.ndarray([0, 0])
        def argmin(self, axis=None, out=None): return np.ndarray([0, 0])
        def argpartition(self, kth, axis=-1, kind='introselect', order=None): return np.ndarray([0, 0])
        def argsort(self, axis=-1, kind='quicksort', order=None): return np.ndarray([0, 0])
        def astype(self, dtype, order='K', casting='unsafe', subok=True, copy=True): return np.ndarray([0, 0])
        def byteswap(self, inplace=False): return np.ndarray([0, 0])
        def choose(self, choices, out=None, mode='raise'): return np.ndarray([0, 0])
        def clip(self, min=None, max=None, out=None): return np.ndarray([0, 0])
        def compress(self, condition, axis=None, out=None): return np.ndarray([0, 0])
        def conj(self): return np.ndarray([0, 0])
        def conjugate(self): return np.ndarray([0, 0])
        def copy(self, order='C'): return np.ndarray([0, 0])
        def cumprod(self, axis=None, dtype=None, out=None): return np.ndarray([0, 0])
        def cumsum(self, axis=None, dtype=None, out=None): return np.ndarray([0, 0])
        def diagonal(self, offset=0, axis1=0, axis2=1): return np.ndarray([0, 0])
        def dot(self, b, out=None): return np.ndarray([0, 0])
        def dump(self, file): return None
        def dumps(self): return str()
        def fill(self, value): return None
        def flatten(self, order='C'): return np.ndarray([0, 0])
        def getfield(self, dtype, offset=0): return np.ndarray([0, 0])
        def item(self, *args): return uninferable
        def itemset(self, *args): return None
        def max(self, axis=None, out=None): return np.ndarray([0, 0])
        def mean(self, axis=None, dtype=None, out=None, keepdims=False): return np.ndarray([0, 0])
        def min(self, axis=None, out=None, keepdims=False): return np.ndarray([0, 0])
        def newbyteorder(self, new_order='S'): return np.ndarray([0, 0])
        def nonzero(self): return (1,)
        def partition(self, kth, axis=-1, kind='introselect', order=None): return None
        def prod(self, axis=None, dtype=None, out=None, keepdims=False): return np.ndarray([0, 0])
        def ptp(self, axis=None, out=None): return np.ndarray([0, 0])
        def put(self, indices, values, mode='raise'): return None
        def ravel(self, order='C'): return np.ndarray([0, 0])
        def repeat(self, repeats, axis=None): return np.ndarray([0, 0])
        def reshape(self, shape, order='C'): return np.ndarray([0, 0])
        def resize(self, new_shape, refcheck=True): return None
        def round(self, decimals=0, out=None): return np.ndarray([0, 0])
        def searchsorted(self, v, side='left', sorter=None): return np.ndarray([0, 0])
        def setfield(self, val, dtype, offset=0): return None
        def setflags(self, write=None, align=None, uic=None): return None
        def sort(self, axis=-1, kind='quicksort', order=None): return None
        def squeeze(self, axis=None): return np.ndarray([0, 0])
        def std(self, axis=None, dtype=None, out=None, ddof=0, keepdims=False): return np.ndarray([0, 0])
        def sum(self, axis=None, dtype=None, out=None, keepdims=False): return np.ndarray([0, 0])
        def swapaxes(self, axis1, axis2): return np.ndarray([0, 0])
        def take(self, indices, axis=None, out=None, mode='raise'): return np.ndarray([0, 0])
        def tobytes(self, order='C'): return b''
        def tofile(self, fid, sep="", format="%s"): return None
        def tolist(self, ): return []
        def tostring(self, order='C'): return b''
        def trace(self, offset=0, axis1=0, axis2=1, dtype=None, out=None): return np.ndarray([0, 0])
        def transpose(self, *axes): return np.ndarray([0, 0])
        def var(self, axis=None, dtype=None, out=None, ddof=0, keepdims=False): return np.ndarray([0, 0])
        def view(self, dtype=None, type=None): return np.ndarray([0, 0])
    """
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

```python
# Licensed under the LGPL: https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html
# For details: https://github.com/PyCQA/astroid/blob/main/LICENSE
# Copyright (c) https://github.com/PyCQA/astroid/blob/main/CONTRIBUTORS.txt

"""This module contains exceptions used in the astroid library."""

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


class AstroidError(Exception):
    """Base exception class for all astroid related exceptions.

    AstroidError and its subclasses are structured, intended to hold
    objects representing state when the exception is thrown.  Field
    values are passed to the constructor as keyword-only arguments.
    Each subclass has its own set of standard fields, but use your
    best judgment to decide whether a specific exception instance
    needs more or fewer fields for debugging.  Field values may be
    used to lazily generate the error message: self.message.format()
    will be called with the field names and values supplied as keyword
    arguments.
    """

    def __init__(self, message: str = "", **kws: Any) -> None:
        super().__init__(message)
        self.message = message
        for key, value in kws.items():
            setattr(self, key, value)

    def __str__(self) -> str:
        return self.message.format(**vars(self))


class AstroidBuildingError(AstroidError):
    """Exception class when we are unable to build an astroid representation.

    Standard attributes:
        modname: Name of the module that AST construction failed for.
        error: Exception raised during construction.
    """

    def __init__(
        self,
        message: str = "Failed to import module {modname}.",
        modname: str | None = None,
        error: Exception | None = None,
        source: str | None = None,
        path: str | None = None,
        cls: type | None = None,
        class_repr: str | None = None,
        **kws: Any,
    ) -> None:
        self.modname = modname
        self.error = error
        self.source = source
        self.path = path
        self.cls = cls
        self.class_repr = class_repr
        super().__init__(message, **kws)


class AstroidImportError(AstroidBuildingError):
    """Exception class used when a module can't be imported by astroid."""


class TooManyLevelsError(AstroidImportError):
    """Exception class which is raised when a relative import was beyond the top-level.

    Standard attributes:
        level: The level which was attempted.
        name: the name of the module on which the relative import was attempted.
    """

    def __init__(
        self,
        message: str = "Relative import with too many levels "
        "({level}) for module {name!r}",
        level: int | None = None,
        name: str | None = None,
        **kws: Any,
    ) -> None:
        self.level = level
        self.name = name
        super().__init__(message, **kws)


class AstroidSyntaxError(AstroidBuildingError):
    """Exception class used when a module can't be parsed."""

    def __init__(
        self,
        message: str,
        modname: str | None,
        error: Exception,
        path: str | None,
        source: str | None = None,
    ) -> None:
        super().__init__(message, modname, error, source, path)


class NoDefault(AstroidError):
    """Raised by function's `default_value` method when an argument has
    no default value.

    Standard attributes:
        func: Function node.
        name: Name of argument without a default.
    """

    def __init__(
        self,
        message: str = "{func!r} has no default for {name!r}.",
        func: nodes.FunctionDef | None = None,
        name: str | None = None,
        **kws: Any,
    ) -> None:
        self.func = func
        self.name = name
        super().__init__(message, **kws)


class ResolveError(AstroidError):
    """Base class of astroid resolution/inference error.

    ResolveError is not intended to be raised.

    Standard attributes:
        context: InferenceContext object.
    """

    def __init__(
        self, message: str = "", context: InferenceContext | None = None, **kws: Any
    ) -> None:
        self.context = context
        super().__init__(message, **kws)


class MroError(ResolveError):
    """Error raised when there is a problem with method resolution of a class.

    Standard attributes:
        mros: A sequence of sequences containing ClassDef nodes.
        cls: ClassDef node whose MRO resolution failed.
        context: InferenceContext object.
    """

    def __init__(
        self,
        message: str,
        mros: list[nodes.ClassDef],
        cls: nodes.ClassDef,
        context: InferenceContext | None = None,
        **kws: Any,
    ) -> None:
        self.mros = mros
        self.cls = cls
        self.context = context
        super().__init__(message, **kws)

    def __str__(self) -> str:
        mro_names = ", ".join(f"({', '.join(b.name for b in m)})" for m in self.mros)
        return self.message.format(mros=mro_names, cls=self.cls)


class DuplicateBasesError(MroError):
    """Error raised when there are duplicate bases in the same class bases."""


class InconsistentMroError(MroError):
    """Error raised when a class's MRO is inconsistent."""


class SuperError(ResolveError):
    """Error raised when there is a problem with a *super* call.

    Standard attributes:
        *super_*: The Super instance that raised the exception.
        context: InferenceContext object.
    """

    def __init__(self, message: str, super_: objects.Super, **kws: Any) -> None:
        self.super_ = super_
        super().__init__(message, **kws)

    def __str__(self) -> str:
        return self.message.format(**vars(self.super_))


class InferenceError(ResolveError):  # pylint: disable=too-many-instance-attributes
    """Raised when we are unable to infer a node.

    Standard attributes:
        node: The node inference was called on.
        context: InferenceContext object.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        message: str = "Inference failed for {node!r}.",
        node: nodes.NodeNG | bases.Instance | None = None,
        context: InferenceContext | None = None,
        target: nodes.NodeNG | bases.Instance | None = None,
        targets: nodes.Tuple | None = None,
        attribute: str | None = None,
        unknown: nodes.NodeNG | bases.Instance | None = None,
        assign_path: list[int] | None = None,
        caller: nodes.Call | None = None,
        stmts: Sequence[nodes.NodeNG | bases.Instance] | None = None,
        frame: nodes.LocalsDictNodeNG | None = None,
        call_site: arguments.CallSite | None = None,
        func: nodes.FunctionDef | None = None,
        arg: str | None = None,
        positional_arguments: list | None = None,
        unpacked_args: list | None = None,
        keyword_arguments: dict | None = None,
        unpacked_kwargs: dict | None = None,
        **kws: Any,
    ) -> None:
        self.node = node
        self.context = context
        self.target = target
        self.targets = targets
        self.attribute = attribute
        self.unknown = unknown
        self.assign_path = assign_path
        self.caller = caller
        self.stmts = stmts
        self.frame = frame
        self.call_site = call_site
        self.func = func
        self.arg = arg
        self.positional_arguments = positional_arguments
        self.unpacked_args = unpacked_args
        self.keyword_arguments = keyword_arguments
        self.unpacked_kwargs = unpacked_kwargs
        super().__init__(message, **kws)


# Why does this inherit from InferenceError rather than ResolveError?
# Changing it causes some inference tests to fail.
class NameInferenceError(InferenceError):
    """Raised when a name lookup fails, corresponds to NameError.

    Standard attributes:
        name: The name for which lookup failed, as a string.
        scope: The node representing the scope in which the lookup occurred.
        context: InferenceContext object.
    """

    def __init__(
        self,
        message: str = "{name!r} not found in {scope!r}.",
        name: str | None = None,
        scope: nodes.LocalsDictNodeNG | None = None,
        context: InferenceContext | None = None,
        **kws: Any,
    ) -> None:
        self.name = name
        self.scope = scope
        self.context = context
        super().__init__(message, **kws)


class AttributeInferenceError(ResolveError):
    """Raised when an attribute lookup fails, corresponds to AttributeError.

    Standard attributes:
        target: The node for which lookup failed.
        attribute: The attribute for which lookup failed, as a string.
        context: InferenceContext object.
    """

    def __init__(
        self,
        message: str = "{attribute!r} not found on {target!r}.",
        attribute: str = "",
        target: nodes.NodeNG | bases.Instance | None = None,
        context: InferenceContext | None = None,
        mros: list[nodes.ClassDef] | None = None,
        super_: nodes.ClassDef | None = None,
        cls: nodes.ClassDef | None = None,
        **kws: Any,
    ) -> None:
        self.attribute = attribute
        self.target = target
        self.context = context
        self.mros = mros
        self.super_ = super_
        self.cls = cls
        super().__init__(message, **kws)


class UseInferenceDefault(Exception):
    """Exception to be raised in custom inference function to indicate that it
    should go back to the default behaviour.
    """


class _NonDeducibleTypeHierarchy(Exception):
    """Raised when is_subtype / is_supertype can't deduce the relation between two
    types.
    """


class AstroidIndexError(AstroidError):
    """Raised when an Indexable / Mapping does not have an index / key."""

    def __init__(
        self,
        message: str = "",
        node: nodes.NodeNG | bases.Instance | None = None,
        index: nodes.Subscript | None = None,
        context: InferenceContext | None = None,
        **kws: Any,
    ) -> None:
        self.node = node
        self.index = index
        self.context = context
        super().__init__(message, **kws)


class AstroidTypeError(AstroidError):
    """Raised when a TypeError would be expected in Python code."""

    def __init__(
        self,
        message: str = "",
        node: nodes.NodeNG | bases.Instance | None = None,
        index: nodes.Subscript | None = None,
        context: InferenceContext | None = None,
        **kws: Any,
    ) -> None:
        self.node = node
        self.index = index
        self.context = context
        super().__init__(message, **kws)


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


class StatementMissing(ParentMissingError):
    """Raised when a call to node.statement() does not return a node.

    This is because a node in the chain does not have a parent attribute
    and therefore does not return a node for statement().

    Standard attributes:
        target: The node for which the parent lookup failed.
    """

    def __init__(self, target: nodes.NodeNG) -> None:
        super(ParentMissingError, self).__init__(
            message=f"Statement not found on {target!r}"
        )


# Backwards-compatibility aliases
OperationError = util.BadOperationMessage
UnaryOperationError = util.BadUnaryOperationMessage
BinaryOperationError = util.BadBinaryOperationMessage

SuperArgumentTypeError = SuperError
UnresolvableName = NameInferenceError
NotFoundError = AttributeInferenceError
AstroidBuildingException = AstroidBuildingError

```
### 7 - astroid/mixins.py:

```python
# Licensed under the LGPL: https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html
# For details: https://github.com/PyCQA/astroid/blob/main/LICENSE
# Copyright (c) https://github.com/PyCQA/astroid/blob/main/CONTRIBUTORS.txt

"""This module contains some mixins for the different nodes."""

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

```python
# Licensed under the LGPL: https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html
# For details: https://github.com/PyCQA/astroid/blob/main/LICENSE
# Copyright (c) https://github.com/PyCQA/astroid/blob/main/CONTRIBUTORS.txt

"""This module contains base classes and functions for the nodes and some
inference utils.
"""
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


def _is_property(meth, context: InferenceContext | None = None) -> bool:
    decoratornames = meth.decoratornames(context=context)
    if PROPERTIES.intersection(decoratornames):
        return True
    stripped = {
        name.split(".")[-1] for name in decoratornames if name is not Uninferable
    }
    if any(name in stripped for name in POSSIBLE_PROPERTIES):
        return True

    # Lookup for subclasses of *property*
    if not meth.decorators:
        return False
    for decorator in meth.decorators.nodes or ():
        inferred = helpers.safe_infer(decorator, context=context)
        if inferred is None or inferred is Uninferable:
            continue
        if inferred.__class__.__name__ == "ClassDef":
            for base_class in inferred.bases:
                if base_class.__class__.__name__ != "Name":
                    continue
                module, _ = base_class.lookup(base_class.name)
                if module.name == "builtins" and base_class.name == "property":
                    return True

    return False


class Proxy:
    """A simple proxy object.

    Note:

    Subclasses of this object will need a custom __getattr__
    if new instance attributes are created. See the Const class
    """

    _proxied: nodes.ClassDef | nodes.Lambda | Proxy | None = (
        None  # proxied object may be set by class or by instance
    )

    def __init__(
        self, proxied: nodes.ClassDef | nodes.Lambda | Proxy | None = None
    ) -> None:
        if proxied is None:
            # This is a hack to allow calling this __init__ during bootstrapping of
            # builtin classes and their docstrings.
            # For Const and Generator nodes the _proxied attribute is set during bootstrapping
            # as we first need to build the ClassDef that they can proxy.
            # Thus, if proxied is None self should be a Const or Generator
            # as that is the only way _proxied will be correctly set as a ClassDef.
            assert isinstance(self, (nodes.Const, Generator))
        else:
            self._proxied = proxied

    def __getattr__(self, name):
        if name == "_proxied":
            return self.__class__._proxied
        if name in self.__dict__:
            return self.__dict__[name]
        return getattr(self._proxied, name)

    def infer(  # type: ignore[return]
        self, context: InferenceContext | None = None, **kwargs: Any
    ) -> collections.abc.Generator[InferenceResult, None, InferenceErrorInfo | None]:
        yield self


def _infer_stmts(
    stmts: Sequence[nodes.NodeNG | type[Uninferable] | Instance],
    context: InferenceContext | None,
    frame: nodes.NodeNG | Instance | None = None,
) -> collections.abc.Generator[InferenceResult, None, None]:
    """Return an iterator on statements inferred by each statement in *stmts*."""
    inferred = False
    constraint_failed = False
    if context is not None:
        name = context.lookupname
        context = context.clone()
        constraints = context.constraints.get(name, {})
    else:
        name = None
        constraints = {}
        context = InferenceContext()

    for stmt in stmts:
        if stmt is Uninferable:
            yield stmt
            inferred = True
            continue
        # 'context' is always InferenceContext and Instances get '_infer_name' from ClassDef
        context.lookupname = stmt._infer_name(frame, name)  # type: ignore[union-attr]
        try:
            stmt_constraints: set[Constraint] = set()
            for constraint_stmt, potential_constraints in constraints.items():
                if not constraint_stmt.parent_of(stmt):
                    stmt_constraints.update(potential_constraints)
            # Mypy doesn't recognize that 'stmt' can't be Uninferable
            for inf in stmt.infer(context=context):  # type: ignore[union-attr]
                if all(constraint.satisfied_by(inf) for constraint in stmt_constraints):
                    yield inf
                    inferred = True
                else:
                    constraint_failed = True
        except NameInferenceError:
            continue
        except InferenceError:
            yield Uninferable
            inferred = True

    if not inferred and constraint_failed:
        yield Uninferable
    elif not inferred:
        raise InferenceError(
            "Inference failed for all members of {stmts!r}.",
            stmts=stmts,
            frame=frame,
            context=context,
        )


def _infer_method_result_truth(instance, method_name, context):
    # Get the method from the instance and try to infer
    # its return's truth value.
    meth = next(instance.igetattr(method_name, context=context), None)
    if meth and hasattr(meth, "infer_call_result"):
        if not meth.callable():
            return Uninferable
        try:
            context.callcontext = CallContext(args=[], callee=meth)
            for value in meth.infer_call_result(instance, context=context):
                if value is Uninferable:
                    return value
                try:
                    inferred = next(value.infer(context=context))
                except StopIteration as e:
                    raise InferenceError(context=context) from e
                return inferred.bool_value()
        except InferenceError:
            pass
    return Uninferable


class BaseInstance(Proxy):
    """An instance base class, which provides lookup methods for potential
    instances.
    """

    special_attributes = None

    def display_type(self) -> str:
        return "Instance of"

    def getattr(self, name, context: InferenceContext | None = None, lookupclass=True):
        try:
            values = self._proxied.instance_attr(name, context)
        except AttributeInferenceError as exc:
            if self.special_attributes and name in self.special_attributes:
                return [self.special_attributes.lookup(name)]

            if lookupclass:
                # Class attributes not available through the instance
                # unless they are explicitly defined.
                return self._proxied.getattr(name, context, class_context=False)

            raise AttributeInferenceError(
                target=self, attribute=name, context=context
            ) from exc
        # since we've no context information, return matching class members as
        # well
        if lookupclass:
            try:
                return values + self._proxied.getattr(
                    name, context, class_context=False
                )
            except AttributeInferenceError:
                pass
        return values

    def igetattr(self, name, context: InferenceContext | None = None):
        """Inferred getattr."""
        if not context:
            context = InferenceContext()
        try:
            context.lookupname = name
            # avoid recursively inferring the same attr on the same class
            if context.push(self._proxied):
                raise InferenceError(
                    message="Cannot infer the same attribute again",
                    node=self,
                    context=context,
                )

            # XXX frame should be self._proxied, or not ?
            get_attr = self.getattr(name, context, lookupclass=False)
            yield from _infer_stmts(
                self._wrap_attr(get_attr, context), context, frame=self
            )
        except AttributeInferenceError:
            try:
                # fallback to class.igetattr since it has some logic to handle
                # descriptors
                # But only if the _proxied is the Class.
                if self._proxied.__class__.__name__ != "ClassDef":
                    raise
                attrs = self._proxied.igetattr(name, context, class_context=False)
                yield from self._wrap_attr(attrs, context)
            except AttributeInferenceError as error:
                raise InferenceError(**vars(error)) from error

    def _wrap_attr(self, attrs, context: InferenceContext | None = None):
        """Wrap bound methods of attrs in a InstanceMethod proxies."""
        for attr in attrs:
            if isinstance(attr, UnboundMethod):
                if _is_property(attr):
                    yield from attr.infer_call_result(self, context)
                else:
                    yield BoundMethod(attr, self)
            elif hasattr(attr, "name") and attr.name == "<lambda>":
                if attr.args.arguments and attr.args.arguments[0].name == "self":
                    yield BoundMethod(attr, self)
                    continue
                yield attr
            else:
                yield attr

    def infer_call_result(
        self, caller: nodes.Call | Proxy, context: InferenceContext | None = None
    ):
        """Infer what a class instance is returning when called."""
        context = bind_context_to_node(context, self)
        inferred = False

        # If the call is an attribute on the instance, we infer the attribute itself
        if isinstance(caller, nodes.Call) and isinstance(caller.func, nodes.Attribute):
            for res in self.igetattr(caller.func.attrname, context):
                inferred = True
                yield res

        # Otherwise we infer the call to the __call__ dunder normally
        for node in self._proxied.igetattr("__call__", context):
            if node is Uninferable or not node.callable():
                continue
            for res in node.infer_call_result(caller, context):
                inferred = True
                yield res
        if not inferred:
            raise InferenceError(node=self, caller=caller, context=context)


class Instance(BaseInstance):
    """A special node representing a class instance."""

    _proxied: nodes.ClassDef

    # pylint: disable=unnecessary-lambda
    special_attributes = lazy_descriptor(lambda: objectmodel.InstanceModel())

    def __init__(self, proxied: nodes.ClassDef | None) -> None:
        super().__init__(proxied)

    infer_binary_op: ClassVar[InferBinaryOp[Instance]]

    def __repr__(self) -> str:
        return "<Instance of {}.{} at 0x{}>".format(
            self._proxied.root().name, self._proxied.name, id(self)
        )

    def __str__(self) -> str:
        return f"Instance of {self._proxied.root().name}.{self._proxied.name}"

    def callable(self) -> bool:
        try:
            self._proxied.getattr("__call__", class_context=False)
            return True
        except AttributeInferenceError:
            return False

    def pytype(self) -> str:
        return self._proxied.qname()

    def display_type(self) -> str:
        return "Instance of"

    def bool_value(self, context: InferenceContext | None = None):
        """Infer the truth value for an Instance.

        The truth value of an instance is determined by these conditions:

           * if it implements __bool__ on Python 3 or __nonzero__
             on Python 2, then its bool value will be determined by
             calling this special method and checking its result.
           * when this method is not defined, __len__() is called, if it
             is defined, and the object is considered true if its result is
             nonzero. If a class defines neither __len__() nor __bool__(),
             all its instances are considered true.
        """
        context = context or InferenceContext()
        context.boundnode = self

        try:
            result = _infer_method_result_truth(self, BOOL_SPECIAL_METHOD, context)
        except (InferenceError, AttributeInferenceError):
            # Fallback to __len__.
            try:
                result = _infer_method_result_truth(self, "__len__", context)
            except (AttributeInferenceError, InferenceError):
                return True
        return result

    def getitem(self, index, context: InferenceContext | None = None):
        new_context = bind_context_to_node(context, self)
        if not context:
            context = new_context
        method = next(self.igetattr("__getitem__", context=context), None)
        # Create a new CallContext for providing index as an argument.
        new_context.callcontext = CallContext(args=[index], callee=method)
        if not isinstance(method, BoundMethod):
            raise InferenceError(
                "Could not find __getitem__ for {node!r}.", node=self, context=context
            )
        if len(method.args.arguments) != 2:  # (self, index)
            raise AstroidTypeError(
                "__getitem__ for {node!r} does not have correct signature",
                node=self,
                context=context,
            )
        return next(method.infer_call_result(self, new_context), None)


class UnboundMethod(Proxy):
    """A special node representing a method not bound to an instance."""

    # pylint: disable=unnecessary-lambda
    special_attributes = lazy_descriptor(lambda: objectmodel.UnboundMethodModel())

    def __repr__(self) -> str:
        frame = self._proxied.parent.frame(future=True)
        return "<{} {} of {} at 0x{}".format(
            self.__class__.__name__, self._proxied.name, frame.qname(), id(self)
        )

    def implicit_parameters(self) -> Literal[0]:
        return 0

    def is_bound(self) -> Literal[False]:
        return False

    def getattr(self, name, context: InferenceContext | None = None):
        if name in self.special_attributes:
            return [self.special_attributes.lookup(name)]
        return self._proxied.getattr(name, context)

    def igetattr(self, name, context: InferenceContext | None = None):
        if name in self.special_attributes:
            return iter((self.special_attributes.lookup(name),))
        return self._proxied.igetattr(name, context)

    def infer_call_result(self, caller, context):
        """
        The boundnode of the regular context with a function called
        on ``object.__new__`` will be of type ``object``,
        which is incorrect for the argument in general.
        If no context is given the ``object.__new__`` call argument will
        be correctly inferred except when inside a call that requires
        the additional context (such as a classmethod) of the boundnode
        to determine which class the method was called from
        """

        # If we're unbound method __new__ of a builtin, the result is an
        # instance of the class given as first argument.
        if self._proxied.name == "__new__":
            qname = self._proxied.parent.frame(future=True).qname()
            # Avoid checking builtins.type: _infer_type_new_call() does more validation
            if qname.startswith("builtins.") and qname != "builtins.type":
                return self._infer_builtin_new(caller, context)
        return self._proxied.infer_call_result(caller, context)

    def _infer_builtin_new(
        self,
        caller: nodes.Call,
        context: InferenceContext,
    ) -> collections.abc.Generator[
        nodes.Const | Instance | type[Uninferable], None, None
    ]:
        if not caller.args:
            return
        # Attempt to create a constant
        if len(caller.args) > 1:
            value = None
            if isinstance(caller.args[1], nodes.Const):
                value = caller.args[1].value
            else:
                inferred_arg = next(caller.args[1].infer(), None)
                if isinstance(inferred_arg, nodes.Const):
                    value = inferred_arg.value
            if value is not None:
                yield nodes.const_factory(value)
                return

        node_context = context.extra_context.get(caller.args[0])
        for inferred in caller.args[0].infer(context=node_context):
            if inferred is Uninferable:
                yield inferred
            if isinstance(inferred, nodes.ClassDef):
                yield Instance(inferred)
            raise InferenceError

    def bool_value(self, context: InferenceContext | None = None) -> Literal[True]:
        return True


class BoundMethod(UnboundMethod):
    """A special node representing a method bound to an instance."""

    # pylint: disable=unnecessary-lambda
    special_attributes = lazy_descriptor(lambda: objectmodel.BoundMethodModel())

    def __init__(self, proxy, bound):
        super().__init__(proxy)
        self.bound = bound

    def implicit_parameters(self) -> Literal[0, 1]:
        if self.name == "__new__":
            # __new__ acts as a classmethod but the class argument is not implicit.
            return 0
        return 1

    def is_bound(self) -> Literal[True]:
        return True

    def _infer_type_new_call(self, caller, context):  # noqa: C901
        """Try to infer what type.__new__(mcs, name, bases, attrs) returns.

        In order for such call to be valid, the metaclass needs to be
        a subtype of ``type``, the name needs to be a string, the bases
        needs to be a tuple of classes
        """
        # pylint: disable=import-outside-toplevel; circular import
        from astroid.nodes import Pass

        # Verify the metaclass
        try:
            mcs = next(caller.args[0].infer(context=context))
        except StopIteration as e:
            raise InferenceError(context=context) from e
        if mcs.__class__.__name__ != "ClassDef":
            # Not a valid first argument.
            return None
        if not mcs.is_subtype_of("builtins.type"):
            # Not a valid metaclass.
            return None

        # Verify the name
        try:
            name = next(caller.args[1].infer(context=context))
        except StopIteration as e:
            raise InferenceError(context=context) from e
        if name.__class__.__name__ != "Const":
            # Not a valid name, needs to be a const.
            return None
        if not isinstance(name.value, str):
            # Needs to be a string.
            return None

        # Verify the bases
        try:
            bases = next(caller.args[2].infer(context=context))
        except StopIteration as e:
            raise InferenceError(context=context) from e
        if bases.__class__.__name__ != "Tuple":
            # Needs to be a tuple.
            return None
        try:
            inferred_bases = [next(elt.infer(context=context)) for elt in bases.elts]
        except StopIteration as e:
            raise InferenceError(context=context) from e
        if any(base.__class__.__name__ != "ClassDef" for base in inferred_bases):
            # All the bases needs to be Classes
            return None

        # Verify the attributes.
        try:
            attrs = next(caller.args[3].infer(context=context))
        except StopIteration as e:
            raise InferenceError(context=context) from e
        if attrs.__class__.__name__ != "Dict":
            # Needs to be a dictionary.
            return None
        cls_locals = collections.defaultdict(list)
        for key, value in attrs.items:
            try:
                key = next(key.infer(context=context))
            except StopIteration as e:
                raise InferenceError(context=context) from e
            try:
                value = next(value.infer(context=context))
            except StopIteration as e:
                raise InferenceError(context=context) from e
            # Ignore non string keys
            if key.__class__.__name__ == "Const" and isinstance(key.value, str):
                cls_locals[key.value].append(value)

        # Build the class from now.
        cls = mcs.__class__(
            name=name.value,
            lineno=caller.lineno,
            col_offset=caller.col_offset,
            parent=caller,
        )
        empty = Pass()
        cls.postinit(
            bases=bases.elts,
            body=[empty],
            decorators=[],
            newstyle=True,
            metaclass=mcs,
            keywords=[],
        )
        cls.locals = cls_locals
        return cls

    def infer_call_result(self, caller, context: InferenceContext | None = None):
        context = bind_context_to_node(context, self.bound)
        if (
            self.bound.__class__.__name__ == "ClassDef"
            and self.bound.name == "type"
            and self.name == "__new__"
            and len(caller.args) == 4
        ):
            # Check if we have a ``type.__new__(mcs, name, bases, attrs)`` call.
            new_cls = self._infer_type_new_call(caller, context)
            if new_cls:
                return iter((new_cls,))

        return super().infer_call_result(caller, context)

    def bool_value(self, context: InferenceContext | None = None) -> Literal[True]:
        return True


class Generator(BaseInstance):
    """A special node representing a generator.

    Proxied class is set once for all in raw_building.
    """

    _proxied: nodes.ClassDef

    special_attributes = lazy_descriptor(objectmodel.GeneratorModel)

    def __init__(
        self, parent=None, generator_initial_context: InferenceContext | None = None
    ):
        super().__init__()
        self.parent = parent
        self._call_context = copy_context(generator_initial_context)

    @decorators.cached
    def infer_yield_types(self):
        yield from self.parent.infer_yield_result(self._call_context)

    def callable(self) -> Literal[False]:
        return False

    def pytype(self) -> Literal["builtins.generator"]:
        return "builtins.generator"

    def display_type(self) -> str:
        return "Generator"

    def bool_value(self, context: InferenceContext | None = None) -> Literal[True]:
        return True

    def __repr__(self) -> str:
        return f"<Generator({self._proxied.name}) l.{self.lineno} at 0x{id(self)}>"

    def __str__(self) -> str:
        return f"Generator({self._proxied.name})"


class AsyncGenerator(Generator):
    """Special node representing an async generator."""

    def pytype(self) -> Literal["builtins.async_generator"]:
        return "builtins.async_generator"

    def display_type(self) -> str:
        return "AsyncGenerator"

    def __repr__(self) -> str:
        return f"<AsyncGenerator({self._proxied.name}) l.{self.lineno} at 0x{id(self)}>"

    def __str__(self) -> str:
        return f"AsyncGenerator({self._proxied.name})"

```
### 9 - astroid/nodes/scoped_nodes/scoped_nodes.py:

```python
# Licensed under the LGPL: https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html
# For details: https://github.com/PyCQA/astroid/blob/main/LICENSE
# Copyright (c) https://github.com/PyCQA/astroid/blob/main/CONTRIBUTORS.txt

"""
This module contains the classes for "scoped" node, i.e. which are opening a
new local scope in the language definition : Module, ClassDef, FunctionDef (and
Lambda, GeneratorExp, DictComp and SetComp to some extent).
"""

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


def _c3_merge(sequences, cls, context):
    """Merges MROs in *sequences* to a single MRO using the C3 algorithm.

    Adapted from http://www.python.org/download/releases/2.3/mro/.

    """
    result = []
    while True:
        sequences = [s for s in sequences if s]  # purge empty sequences
        if not sequences:
            return result
        for s1 in sequences:  # find merge candidates among seq heads
            candidate = s1[0]
            for s2 in sequences:
                if candidate in s2[1:]:
                    candidate = None
                    break  # reject the current head, it appears later
            else:
                break
        if not candidate:
            # Show all the remaining bases, which were considered as
            # candidates for the next mro sequence.
            raise InconsistentMroError(
                message="Cannot create a consistent method resolution order "
                "for MROs {mros} of class {cls!r}.",
                mros=sequences,
                cls=cls,
                context=context,
            )

        result.append(candidate)
        # remove the chosen candidate
        for seq in sequences:
            if seq[0] == candidate:
                del seq[0]
    return None


def clean_typing_generic_mro(sequences: list[list[ClassDef]]) -> None:
    """A class can inherit from typing.Generic directly, as base,
    and as base of bases. The merged MRO must however only contain the last entry.
    To prepare for _c3_merge, remove some typing.Generic entries from
    sequences if multiple are present.

    This method will check if Generic is in inferred_bases and also
    part of bases_mro. If true, remove it from inferred_bases
    as well as its entry the bases_mro.

    Format sequences: [[self]] + bases_mro + [inferred_bases]
    """
    bases_mro = sequences[1:-1]
    inferred_bases = sequences[-1]
    # Check if Generic is part of inferred_bases
    for i, base in enumerate(inferred_bases):
        if base.qname() == "typing.Generic":
            position_in_inferred_bases = i
            break
    else:
        return
    # Check if also part of bases_mro
    # Ignore entry for typing.Generic
    for i, seq in enumerate(bases_mro):
        if i == position_in_inferred_bases:
            continue
        if any(base.qname() == "typing.Generic" for base in seq):
            break
    else:
        return
    # Found multiple Generics in mro, remove entry from inferred_bases
    # and the corresponding one from bases_mro
    inferred_bases.pop(position_in_inferred_bases)
    bases_mro.pop(position_in_inferred_bases)


def clean_duplicates_mro(sequences, cls, context):
    for sequence in sequences:
        names = [
            (node.lineno, node.qname()) if node.name else None for node in sequence
        ]
        last_index = dict(map(reversed, enumerate(names)))
        if names and names[0] is not None and last_index[names[0]] != 0:
            raise DuplicateBasesError(
                message="Duplicates found in MROs {mros} for {cls!r}.",
                mros=sequences,
                cls=cls,
                context=context,
            )
        yield [
            node
            for i, (node, name) in enumerate(zip(sequence, names))
            if name is None or last_index[name] == i
        ]


def function_to_method(n, klass):
    if isinstance(n, FunctionDef):
        if n.type == "classmethod":
            return bases.BoundMethod(n, klass)
        if n.type == "property":
            return n
        if n.type != "staticmethod":
            return bases.UnboundMethod(n)
    return n


class Module(LocalsDictNodeNG):
    """Class representing an :class:`ast.Module` node.

    >>> import astroid
    >>> node = astroid.extract_node('import astroid')
    >>> node
    <Import l.1 at 0x7f23b2e4e5c0>
    >>> node.parent
    <Module l.0 at 0x7f23b2e4eda0>
    """

    _astroid_fields = ("doc_node", "body")

    fromlineno: Literal[0] = 0
    """The first line that this node appears on in the source code."""

    lineno: Literal[0] = 0
    """The line that this node appears on in the source code."""

    # attributes below are set by the builder module or by raw factories

    file_bytes: str | bytes | None = None
    """The string/bytes that this ast was built from."""

    file_encoding: str | None = None
    """The encoding of the source file.

    This is used to get unicode out of a source file.
    Python 2 only.
    """

    special_attributes = ModuleModel()
    """The names of special attributes that this module has."""

    # names of module attributes available through the global scope
    scope_attrs = {"__name__", "__doc__", "__file__", "__path__", "__package__"}
    """The names of module attributes available through the global scope."""

    _other_fields = (
        "name",
        "doc",
        "file",
        "path",
        "package",
        "pure_python",
        "future_imports",
    )
    _other_other_fields = ("locals", "globals")

    col_offset: None
    end_lineno: None
    end_col_offset: None
    parent: None

    @decorators_mod.deprecate_arguments(doc="Use the postinit arg 'doc_node' instead")
    def __init__(
        self,
        name: str,
        doc: str | None = None,
        file: str | None = None,
        path: list[str] | None = None,
        package: bool | None = None,
        parent: None = None,
        pure_python: bool | None = True,
    ) -> None:
        """
        :param name: The name of the module.

        :param doc: The module docstring.

        :param file: The path to the file that this ast has been extracted from.

        :param path:

        :param package: Whether the node represents a package or a module.

        :param parent: The parent node in the syntax tree.

        :param pure_python: Whether the ast was built from source.
        """
        self.name = name
        """The name of the module."""

        self._doc = doc
        """The module docstring."""

        self.file = file
        """The path to the file that this ast has been extracted from.

        This will be ``None`` when the representation has been built from a
        built-in module.
        """

        self.path = path

        self.package = package
        """Whether the node represents a package or a module."""

        self.pure_python = pure_python
        """Whether the ast was built from source."""

        self.globals: dict[str, list[node_classes.NodeNG]]
        """A map of the name of a global variable to the node defining the global."""

        self.locals = self.globals = {}
        """A map of the name of a local variable to the node defining the local."""

        self.body: list[node_classes.NodeNG] | None = []
        """The contents of the module."""

        self.doc_node: Const | None = None
        """The doc node associated with this node."""

        self.future_imports: set[str] = set()
        """The imports from ``__future__``."""

        super().__init__(lineno=0, parent=parent)

    # pylint: enable=redefined-builtin

    def postinit(self, body=None, *, doc_node: Const | None = None):
        """Do some setup after initialisation.

        :param body: The contents of the module.
        :type body: list(NodeNG) or None
        :param doc_node: The doc node associated with this node.
        """
        self.body = body
        self.doc_node = doc_node
        if doc_node:
            self._doc = doc_node.value

    @property
    def doc(self) -> str | None:
        """The module docstring."""
        warnings.warn(
            "The 'Module.doc' attribute is deprecated, "
            "use 'Module.doc_node' instead.",
            DeprecationWarning,
        )
        return self._doc

    @doc.setter
    def doc(self, value: str | None) -> None:
        warnings.warn(
            "Setting the 'Module.doc' attribute is deprecated, "
            "use 'Module.doc_node' instead.",
            DeprecationWarning,
        )
        self._doc = value

    def _get_stream(self):
        if self.file_bytes is not None:
            return io.BytesIO(self.file_bytes)
        if self.file is not None:
            # pylint: disable=consider-using-with
            stream = open(self.file, "rb")
            return stream
        return None

    def stream(self):
        """Get a stream to the underlying file or bytes.

        :type: file or io.BytesIO or None
        """
        return self._get_stream()

    def block_range(self, lineno):
        """Get a range from where this node starts to where this node ends.

        :param lineno: Unused.
        :type lineno: int

        :returns: The range of line numbers that this node belongs to.
        :rtype: tuple(int, int)
        """
        return self.fromlineno, self.tolineno

    def scope_lookup(self, node, name, offset=0):
        """Lookup where the given variable is assigned.

        :param node: The node to look for assignments up to.
            Any assignments after the given node are ignored.
        :type node: NodeNG

        :param name: The name of the variable to find assignments for.
        :type name: str

        :param offset: The line offset to filter statements up to.
        :type offset: int

        :returns: This scope node and the list of assignments associated to the
            given name according to the scope where it has been found (locals,
            globals or builtin).
        :rtype: tuple(str, list(NodeNG))
        """
        if name in self.scope_attrs and name not in self.locals:
            try:
                return self, self.getattr(name)
            except AttributeInferenceError:
                return self, ()
        return self._scope_lookup(node, name, offset)

    def pytype(self) -> Literal["builtins.module"]:
        """Get the name of the type that this node represents.

        :returns: The name of the type.
        """
        return "builtins.module"

    def display_type(self) -> str:
        """A human readable type of this node.

        :returns: The type of this node.
        :rtype: str
        """
        return "Module"

    def getattr(
        self, name, context: InferenceContext | None = None, ignore_locals=False
    ):
        if not name:
            raise AttributeInferenceError(target=self, attribute=name, context=context)

        result = []
        name_in_locals = name in self.locals

        if name in self.special_attributes and not ignore_locals and not name_in_locals:
            result = [self.special_attributes.lookup(name)]
        elif not ignore_locals and name_in_locals:
            result = self.locals[name]
        elif self.package:
            try:
                result = [self.import_module(name, relative_only=True)]
            except (AstroidBuildingError, SyntaxError) as exc:
                raise AttributeInferenceError(
                    target=self, attribute=name, context=context
                ) from exc
        result = [n for n in result if not isinstance(n, node_classes.DelName)]
        if result:
            return result
        raise AttributeInferenceError(target=self, attribute=name, context=context)

    def igetattr(self, name, context: InferenceContext | None = None):
        """Infer the possible values of the given variable.

        :param name: The name of the variable to infer.
        :type name: str

        :returns: The inferred possible values.
        :rtype: iterable(NodeNG) or None
        """
        # set lookup name since this is necessary to infer on import nodes for
        # instance
        context = copy_context(context)
        context.lookupname = name
        try:
            return bases._infer_stmts(self.getattr(name, context), context, frame=self)
        except AttributeInferenceError as error:
            raise InferenceError(
                str(error), target=self, attribute=name, context=context
            ) from error

    def fully_defined(self) -> bool:
        """Check if this module has been build from a .py file.

        If so, the module contains a complete representation,
        including the code.

        :returns: Whether the module has been built from a .py file.
        """
        return self.file is not None and self.file.endswith(".py")

    @overload
    def statement(self, *, future: None = ...) -> Module:
        ...

    @overload
    def statement(self, *, future: Literal[True]) -> NoReturn:
        ...

    def statement(self, *, future: Literal[None, True] = None) -> Module | NoReturn:
        """The first parent node, including self, marked as statement node.

        When called on a :class:`Module` with the future parameter this raises an error.

        TODO: Deprecate the future parameter and only raise StatementMissing

        :raises StatementMissing: If no self has no parent attribute and future is True
        """
        if future:
            raise StatementMissing(target=self)
        warnings.warn(
            "In astroid 3.0.0 NodeNG.statement() will return either a nodes.Statement "
            "or raise a StatementMissing exception. nodes.Module will no longer be "
            "considered a statement. This behaviour can already be triggered "
            "by passing 'future=True' to a statement() call.",
            DeprecationWarning,
        )
        return self

    def previous_sibling(self):
        """The previous sibling statement.

        :returns: The previous sibling statement node.
        :rtype: NodeNG or None
        """

    def next_sibling(self):
        """The next sibling statement node.

        :returns: The next sibling statement node.
        :rtype: NodeNG or None
        """

    _absolute_import_activated = True

    def absolute_import_activated(self) -> bool:
        """Whether :pep:`328` absolute import behaviour has been enabled.

        :returns: Whether :pep:`328` has been enabled.
        """
        return self._absolute_import_activated

    def import_module(
        self,
        modname: str | None,
        relative_only: bool = False,
        level: int | None = None,
        use_cache: bool = True,
    ) -> Module:
        """Get the ast for a given module as if imported from this module.

        :param modname: The name of the module to "import".

        :param relative_only: Whether to only consider relative imports.

        :param level: The level of relative import.

        :param use_cache: Whether to use the astroid_cache of modules.

        :returns: The imported module ast.
        """
        if relative_only and level is None:
            level = 0
        absmodname = self.relative_to_absolute_name(modname, level)

        try:
            return AstroidManager().ast_from_module_name(
                absmodname, use_cache=use_cache
            )
        except AstroidBuildingError:
            # we only want to import a sub module or package of this module,
            # skip here
            if relative_only:
                raise
        return AstroidManager().ast_from_module_name(modname)

    def relative_to_absolute_name(
        self, modname: str | None, level: int | None
    ) -> str | None:
        """Get the absolute module name for a relative import.

        The relative import can be implicit or explicit.

        :param modname: The module name to convert.

        :param level: The level of relative import.

        :returns: The absolute module name.

        :raises TooManyLevelsError: When the relative import refers to a
            module too far above this one.
        """
        # XXX this returns non sens when called on an absolute import
        # like 'pylint.checkers.astroid.utils'
        # XXX doesn't return absolute name if self.name isn't absolute name
        if self.absolute_import_activated() and level is None:
            return modname
        if level:
            if self.package:
                level = level - 1
                package_name = self.name.rsplit(".", level)[0]
            elif (
                self.path
                and not os.path.exists(os.path.dirname(self.path[0]) + "/__init__.py")
                and os.path.exists(
                    os.path.dirname(self.path[0]) + "/" + modname.split(".")[0]
                )
            ):
                level = level - 1
                package_name = ""
            else:
                package_name = self.name.rsplit(".", level)[0]
            if level and self.name.count(".") < level:
                raise TooManyLevelsError(level=level, name=self.name)

        elif self.package:
            package_name = self.name
        else:
            package_name = self.name.rsplit(".", 1)[0]

        if package_name:
            if not modname:
                return package_name
            return f"{package_name}.{modname}"
        return modname

    def wildcard_import_names(self):
        """The list of imported names when this module is 'wildcard imported'.

        It doesn't include the '__builtins__' name which is added by the
        current CPython implementation of wildcard imports.

        :returns: The list of imported names.
        :rtype: list(str)
        """
        # We separate the different steps of lookup in try/excepts
        # to avoid catching too many Exceptions
        default = [name for name in self.keys() if not name.startswith("_")]
        try:
            all_values = self["__all__"]
        except KeyError:
            return default

        try:
            explicit = next(all_values.assigned_stmts())
        except (InferenceError, StopIteration):
            return default
        except AttributeError:
            # not an assignment node
            # XXX infer?
            return default

        # Try our best to detect the exported name.
        inferred = []
        try:
            explicit = next(explicit.infer())
        except (InferenceError, StopIteration):
            return default
        if not isinstance(explicit, (node_classes.Tuple, node_classes.List)):
            return default

        def str_const(node) -> bool:
            return isinstance(node, node_classes.Const) and isinstance(node.value, str)

        for node in explicit.elts:
            if str_const(node):
                inferred.append(node.value)
            else:
                try:
                    inferred_node = next(node.infer())
                except (InferenceError, StopIteration):
                    continue
                if str_const(inferred_node):
                    inferred.append(inferred_node.value)
        return inferred

    def public_names(self):
        """The list of the names that are publicly available in this module.

        :returns: The list of public names.
        :rtype: list(str)
        """
        return [name for name in self.keys() if not name.startswith("_")]

    def bool_value(self, context: InferenceContext | None = None) -> bool:
        """Determine the boolean value of this node.

        :returns: The boolean value of this node.
            For a :class:`Module` this is always ``True``.
        """
        return True

    def get_children(self):
        yield from self.body

    def frame(self: _T, *, future: Literal[None, True] = None) -> _T:
        """The node's frame node.

        A frame node is a :class:`Module`, :class:`FunctionDef`,
        :class:`ClassDef` or :class:`Lambda`.

        :returns: The node itself.
        """
        return self


class GeneratorExp(ComprehensionScope):
    """Class representing an :class:`ast.GeneratorExp` node.

    >>> import astroid
    >>> node = astroid.extract_node('(thing for thing in things if thing)')
    >>> node
    <GeneratorExp l.1 at 0x7f23b2e4e400>
    """

    _astroid_fields = ("elt", "generators")
    _other_other_fields = ("locals",)
    elt = None
    """The element that forms the output of the expression.

    :type: NodeNG or None
    """

    def __init__(
        self,
        lineno=None,
        col_offset=None,
        parent=None,
        *,
        end_lineno=None,
        end_col_offset=None,
    ):
        """
        :param lineno: The line that this node appears on in the source code.
        :type lineno: int or None

        :param col_offset: The column that this node appears on in the
            source code.
        :type col_offset: int or None

        :param parent: The parent node in the syntax tree.
        :type parent: NodeNG or None

        :param end_lineno: The last line this node appears on in the source code.
        :type end_lineno: Optional[int]

        :param end_col_offset: The end column this node appears on in the
            source code. Note: This is after the last symbol.
        :type end_col_offset: Optional[int]
        """
        self.locals = {}
        """A map of the name of a local variable to the node defining the local."""

        super().__init__(
            lineno=lineno,
            col_offset=col_offset,
            end_lineno=end_lineno,
            end_col_offset=end_col_offset,
            parent=parent,
        )

    def postinit(self, elt=None, generators: list[nodes.Comprehension] | None = None):
        """Do some setup after initialisation.

        :param elt: The element that forms the output of the expression.
        :type elt: NodeNG or None

        :param generators: The generators that are looped through.
        """
        self.elt = elt
        if generators is None:
            self.generators = []
        else:
            self.generators = generators

    def bool_value(self, context: InferenceContext | None = None) -> Literal[True]:
        """Determine the boolean value of this node.

        :returns: The boolean value of this node.
            For a :class:`GeneratorExp` this is always ``True``.
        """
        return True

    def get_children(self):
        yield self.elt

        yield from self.generators


class DictComp(ComprehensionScope):
    """Class representing an :class:`ast.DictComp` node.

    >>> import astroid
    >>> node = astroid.extract_node('{k:v for k, v in things if k > v}')
    >>> node
    <DictComp l.1 at 0x7f23b2e41d68>
    """

    _astroid_fields = ("key", "value", "generators")
    _other_other_fields = ("locals",)
    key = None
    """What produces the keys.

    :type: NodeNG or None
    """
    value = None
    """What produces the values.

    :type: NodeNG or None
    """

    def __init__(
        self,
        lineno=None,
        col_offset=None,
        parent=None,
        *,
        end_lineno=None,
        end_col_offset=None,
    ):
        """
        :param lineno: The line that this node appears on in the source code.
        :type lineno: int or None

        :param col_offset: The column that this node appears on in the
            source code.
        :type col_offset: int or None

        :param parent: The parent node in the syntax tree.
        :type parent: NodeNG or None

        :param end_lineno: The last line this node appears on in the source code.
        :type end_lineno: Optional[int]

        :param end_col_offset: The end column this node appears on in the
            source code. Note: This is after the last symbol.
        :type end_col_offset: Optional[int]
        """
        self.locals = {}
        """A map of the name of a local variable to the node defining the local."""

        super().__init__(
            lineno=lineno,
            col_offset=col_offset,
            end_lineno=end_lineno,
            end_col_offset=end_col_offset,
            parent=parent,
        )

    def postinit(
        self,
        key=None,
        value=None,
        generators: list[nodes.Comprehension] | None = None,
    ):
        """Do some setup after initialisation.

        :param key: What produces the keys.
        :type key: NodeNG or None

        :param value: What produces the values.
        :type value: NodeNG or None

        :param generators: The generators that are looped through.
        """
        self.key = key
        self.value = value
        if generators is None:
            self.generators = []
        else:
            self.generators = generators

    def bool_value(self, context: InferenceContext | None = None):
        """Determine the boolean value of this node.

        :returns: The boolean value of this node.
            For a :class:`DictComp` this is always :class:`Uninferable`.
        :rtype: Uninferable
        """
        return util.Uninferable

    def get_children(self):
        yield self.key
        yield self.value

        yield from self.generators


class SetComp(ComprehensionScope):
    """Class representing an :class:`ast.SetComp` node.

    >>> import astroid
    >>> node = astroid.extract_node('{thing for thing in things if thing}')
    >>> node
    <SetComp l.1 at 0x7f23b2e41898>
    """

    _astroid_fields = ("elt", "generators")
    _other_other_fields = ("locals",)
    elt = None
    """The element that forms the output of the expression.

    :type: NodeNG or None
    """

    def __init__(
        self,
        lineno=None,
        col_offset=None,
        parent=None,
        *,
        end_lineno=None,
        end_col_offset=None,
    ):
        """
        :param lineno: The line that this node appears on in the source code.
        :type lineno: int or None

        :param col_offset: The column that this node appears on in the
            source code.
        :type col_offset: int or None

        :param parent: The parent node in the syntax tree.
        :type parent: NodeNG or None

        :param end_lineno: The last line this node appears on in the source code.
        :type end_lineno: Optional[int]

        :param end_col_offset: The end column this node appears on in the
            source code. Note: This is after the last symbol.
        :type end_col_offset: Optional[int]
        """
        self.locals = {}
        """A map of the name of a local variable to the node defining the local."""

        super().__init__(
            lineno=lineno,
            col_offset=col_offset,
            end_lineno=end_lineno,
            end_col_offset=end_col_offset,
            parent=parent,
        )

    def postinit(self, elt=None, generators: list[nodes.Comprehension] | None = None):
        """Do some setup after initialisation.

        :param elt: The element that forms the output of the expression.
        :type elt: NodeNG or None

        :param generators: The generators that are looped through.
        """
        self.elt = elt
        if generators is None:
            self.generators = []
        else:
            self.generators = generators

    def bool_value(self, context: InferenceContext | None = None):
        """Determine the boolean value of this node.

        :returns: The boolean value of this node.
            For a :class:`SetComp` this is always :class:`Uninferable`.
        :rtype: Uninferable
        """
        return util.Uninferable

    def get_children(self):
        yield self.elt

        yield from self.generators


class ListComp(ComprehensionScope):
    """Class representing an :class:`ast.ListComp` node.

    >>> import astroid
    >>> node = astroid.extract_node('[thing for thing in things if thing]')
    >>> node
    <ListComp l.1 at 0x7f23b2e418d0>
    """

    _astroid_fields = ("elt", "generators")
    _other_other_fields = ("locals",)

    elt = None
    """The element that forms the output of the expression.

    :type: NodeNG or None
    """

    def __init__(
        self,
        lineno=None,
        col_offset=None,
        parent=None,
        *,
        end_lineno=None,
        end_col_offset=None,
    ):
        self.locals = {}
        """A map of the name of a local variable to the node defining it."""

        super().__init__(
            lineno=lineno,
            col_offset=col_offset,
            end_lineno=end_lineno,
            end_col_offset=end_col_offset,
            parent=parent,
        )

    def postinit(self, elt=None, generators: list[nodes.Comprehension] | None = None):
        """Do some setup after initialisation.

        :param elt: The element that forms the output of the expression.
        :type elt: NodeNG or None

        :param generators: The generators that are looped through.
        :type generators: list(Comprehension) or None
        """
        self.elt = elt
        if generators is None:
            self.generators = []
        else:
            self.generators = generators

    def bool_value(self, context: InferenceContext | None = None):
        """Determine the boolean value of this node.

        :returns: The boolean value of this node.
            For a :class:`ListComp` this is always :class:`Uninferable`.
        :rtype: Uninferable
        """
        return util.Uninferable

    def get_children(self):
        yield self.elt

        yield from self.generators


def _infer_decorator_callchain(node):
    """Detect decorator call chaining and see if the end result is a
    static or a classmethod.
    """
    if not isinstance(node, FunctionDef):
        return None
    if not node.parent:
        return None
    try:
        result = next(node.infer_call_result(node.parent), None)
    except InferenceError:
        return None
    if isinstance(result, bases.Instance):
        result = result._proxied
    if isinstance(result, ClassDef):
        if result.is_subtype_of("builtins.classmethod"):
            return "classmethod"
        if result.is_subtype_of("builtins.staticmethod"):
            return "staticmethod"
    if isinstance(result, FunctionDef):
        if not result.decorators:
            return None
        # Determine if this function is decorated with one of the builtin descriptors we want.
        for decorator in result.decorators.nodes:
            if isinstance(decorator, node_classes.Name):
                if decorator.name in BUILTIN_DESCRIPTORS:
                    return decorator.name
            if (
                isinstance(decorator, node_classes.Attribute)
                and isinstance(decorator.expr, node_classes.Name)
                and decorator.expr.name == "builtins"
                and decorator.attrname in BUILTIN_DESCRIPTORS
            ):
                return decorator.attrname
    return None


class Lambda(_base_nodes.FilterStmtsBaseNode, LocalsDictNodeNG):
    """Class representing an :class:`ast.Lambda` node.

    >>> import astroid
    >>> node = astroid.extract_node('lambda arg: arg + 1')
    >>> node
    <Lambda.<lambda> l.1 at 0x7f23b2e41518>
    """

    _astroid_fields = ("args", "body")
    _other_other_fields = ("locals",)
    name = "<lambda>"
    is_lambda = True
    special_attributes = FunctionModel()
    """The names of special attributes that this function has."""

    def implicit_parameters(self) -> Literal[0]:
        return 0

    @property
    def type(self) -> Literal["method", "function"]:
        """Whether this is a method or function.

        :returns: 'method' if this is a method, 'function' otherwise.
        """
        if self.args.arguments and self.args.arguments[0].name == "self":
            if isinstance(self.parent.scope(), ClassDef):
                return "method"
        return "function"

    def __init__(
        self,
        lineno=None,
        col_offset=None,
        parent=None,
        *,
        end_lineno=None,
        end_col_offset=None,
    ):
        """
        :param lineno: The line that this node appears on in the source code.
        :type lineno: int or None

        :param col_offset: The column that this node appears on in the
            source code.
        :type col_offset: int or None

        :param parent: The parent node in the syntax tree.
        :type parent: NodeNG or None

        :param end_lineno: The last line this node appears on in the source code.
        :type end_lineno: Optional[int]

        :param end_col_offset: The end column this node appears on in the
            source code. Note: This is after the last symbol.
        :type end_col_offset: Optional[int]
        """
        self.locals = {}
        """A map of the name of a local variable to the node defining it."""

        self.args: Arguments
        """The arguments that the function takes."""

        self.body = []
        """The contents of the function body.

        :type: list(NodeNG)
        """

        self.instance_attrs: dict[str, list[NodeNG]] = {}

        super().__init__(
            lineno=lineno,
            col_offset=col_offset,
            end_lineno=end_lineno,
            end_col_offset=end_col_offset,
            parent=parent,
        )

    def postinit(self, args: Arguments, body):
        """Do some setup after initialisation.

        :param args: The arguments that the function takes.

        :param body: The contents of the function body.
        :type body: list(NodeNG)
        """
        self.args = args
        self.body = body

    def pytype(self) -> Literal["bultins.instancemethod", "builtins.function"]:
        """Get the name of the type that this node represents.

        :returns: The name of the type.
        """
        if "method" in self.type:
            return "builtins.instancemethod"
        return "builtins.function"

    def display_type(self) -> str:
        """A human readable type of this node.

        :returns: The type of this node.
        :rtype: str
        """
        if "method" in self.type:
            return "Method"
        return "Function"

    def callable(self) -> Literal[True]:
        """Whether this node defines something that is callable.

        :returns: Whether this defines something that is callable
            For a :class:`Lambda` this is always ``True``.
        """
        return True

    def argnames(self) -> list[str]:
        """Get the names of each of the arguments, including that
        of the collections of variable-length arguments ("args", "kwargs",
        etc.), as well as positional-only and keyword-only arguments.

        :returns: The names of the arguments.
        :rtype: list(str)
        """
        if self.args.arguments:  # maybe None with builtin functions
            names = _rec_get_names(self.args.arguments)
        else:
            names = []
        if self.args.vararg:
            names.append(self.args.vararg)
        names += [elt.name for elt in self.args.kwonlyargs]
        if self.args.kwarg:
            names.append(self.args.kwarg)
        return names

    def infer_call_result(self, caller, context: InferenceContext | None = None):
        """Infer what the function returns when called.

        :param caller: Unused
        :type caller: object
        """
        # pylint: disable=no-member; github.com/pycqa/astroid/issues/291
        # args is in fact redefined later on by postinit. Can't be changed
        # to None due to a strong interaction between Lambda and FunctionDef.
        return self.body.infer(context)

    def scope_lookup(self, node, name, offset=0):
        """Lookup where the given names is assigned.

        :param node: The node to look for assignments up to.
            Any assignments after the given node are ignored.
        :type node: NodeNG

        :param name: The name to find assignments for.
        :type name: str

        :param offset: The line offset to filter statements up to.
        :type offset: int

        :returns: This scope node and the list of assignments associated to the
            given name according to the scope where it has been found (locals,
            globals or builtin).
        :rtype: tuple(str, list(NodeNG))
        """
        if node in self.args.defaults or node in self.args.kw_defaults:
            frame = self.parent.frame(future=True)
            # line offset to avoid that def func(f=func) resolve the default
            # value to the defined function
            offset = -1
        else:
            # check this is not used in function decorators
            frame = self
        return frame._scope_lookup(node, name, offset)

    def bool_value(self, context: InferenceContext | None = None) -> Literal[True]:
        """Determine the boolean value of this node.

        :returns: The boolean value of this node.
            For a :class:`Lambda` this is always ``True``.
        """
        return True

    def get_children(self):
        yield self.args
        yield self.body

    def frame(self: _T, *, future: Literal[None, True] = None) -> _T:
        """The node's frame node.

        A frame node is a :class:`Module`, :class:`FunctionDef`,
        :class:`ClassDef` or :class:`Lambda`.

        :returns: The node itself.
        """
        return self

    def getattr(
        self, name: str, context: InferenceContext | None = None
    ) -> list[NodeNG]:
        if not name:
            raise AttributeInferenceError(target=self, attribute=name, context=context)

        found_attrs = []
        if name in self.instance_attrs:
            found_attrs = self.instance_attrs[name]
        if name in self.special_attributes:
            found_attrs.append(self.special_attributes.lookup(name))
        if found_attrs:
            return found_attrs
        raise AttributeInferenceError(target=self, attribute=name)


class FunctionDef(_base_nodes.MultiLineBlockNode, _base_nodes.Statement, Lambda):
    """Class representing an :class:`ast.FunctionDef`.

    >>> import astroid
    >>> node = astroid.extract_node('''
    ... def my_func(arg):
    ...     return arg + 1
    ... ''')
    >>> node
    <FunctionDef.my_func l.2 at 0x7f23b2e71e10>
    """

    _astroid_fields = ("decorators", "args", "returns", "doc_node", "body")
    _multi_line_block_fields = ("body",)
    returns = None
    decorators: node_classes.Decorators | None = None
    """The decorators that are applied to this method or function."""

    is_function = True
    """Whether this node indicates a function.

    For a :class:`FunctionDef` this is always ``True``.

    :type: bool
    """
    type_annotation = None
    """If present, this will contain the type annotation passed by a type comment

    :type: NodeNG or None
    """
    type_comment_args = None
    """
    If present, this will contain the type annotation for arguments
    passed by a type comment
    """
    type_comment_returns = None
    """If present, this will contain the return type annotation, passed by a type comment"""
    # attributes below are set by the builder module or by raw factories
    _other_fields = ("name", "doc", "position")
    _other_other_fields = (
        "locals",
        "_type",
        "type_comment_returns",
        "type_comment_args",
    )
    _type = None

    @decorators_mod.deprecate_arguments(doc="Use the postinit arg 'doc_node' instead")
    def __init__(
        self,
        name=None,
        doc: str | None = None,
        lineno=None,
        col_offset=None,
        parent=None,
        *,
        end_lineno=None,
        end_col_offset=None,
    ):
        """
        :param name: The name of the function.
        :type name: str or None

        :param doc: The function docstring.

        :param lineno: The line that this node appears on in the source code.
        :type lineno: int or None

        :param col_offset: The column that this node appears on in the
            source code.
        :type col_offset: int or None

        :param parent: The parent node in the syntax tree.
        :type parent: NodeNG or None

        :param end_lineno: The last line this node appears on in the source code.
        :type end_lineno: Optional[int]

        :param end_col_offset: The end column this node appears on in the
            source code. Note: This is after the last symbol.
        :type end_col_offset: Optional[int]
        """
        self.name = name
        """The name of the function.

        :type name: str or None
        """

        self._doc = doc
        """The function docstring."""

        self.doc_node: Const | None = None
        """The doc node associated with this node."""

        self.instance_attrs = {}
        super().__init__(
            lineno=lineno,
            col_offset=col_offset,
            end_lineno=end_lineno,
            end_col_offset=end_col_offset,
            parent=parent,
        )
        if parent:
            frame = parent.frame(future=True)
            frame.set_local(name, self)

    def postinit(
        self,
        args: Arguments,
        body,
        decorators: node_classes.Decorators | None = None,
        returns=None,
        type_comment_returns=None,
        type_comment_args=None,
        *,
        position: Position | None = None,
        doc_node: Const | None = None,
    ):
        """Do some setup after initialisation.

        :param args: The arguments that the function takes.

        :param body: The contents of the function body.
        :type body: list(NodeNG)

        :param decorators: The decorators that are applied to this
            method or function.
        :type decorators: Decorators or None
        :params type_comment_returns:
            The return type annotation passed via a type comment.
        :params type_comment_args:
            The args type annotation passed via a type comment.
        :params position:
            Position of function keyword(s) and name.
        :param doc_node:
            The doc node associated with this node.
        """
        self.args = args
        self.body = body
        self.decorators = decorators
        self.returns = returns
        self.type_comment_returns = type_comment_returns
        self.type_comment_args = type_comment_args
        self.position = position
        self.doc_node = doc_node
        if doc_node:
            self._doc = doc_node.value

    @property
    def doc(self) -> str | None:
        """The function docstring."""
        warnings.warn(
            "The 'FunctionDef.doc' attribute is deprecated, "
            "use 'FunctionDef.doc_node' instead.",
            DeprecationWarning,
        )
        return self._doc

    @doc.setter
    def doc(self, value: str | None) -> None:
        warnings.warn(
            "Setting the 'FunctionDef.doc' attribute is deprecated, "
            "use 'FunctionDef.doc_node' instead.",
            DeprecationWarning,
        )
        self._doc = value

    @cached_property
    def extra_decorators(self) -> list[node_classes.Call]:
        """The extra decorators that this function can have.

        Additional decorators are considered when they are used as
        assignments, as in ``method = staticmethod(method)``.
        The property will return all the callables that are used for
        decoration.
        """
        frame = self.parent.frame(future=True)
        if not isinstance(frame, ClassDef):
            return []

        decorators: list[node_classes.Call] = []
        for assign in frame._get_assign_nodes():
            if isinstance(assign.value, node_classes.Call) and isinstance(
                assign.value.func, node_classes.Name
            ):
                for assign_node in assign.targets:
                    if not isinstance(assign_node, node_classes.AssignName):
                        # Support only `name = callable(name)`
                        continue

                    if assign_node.name != self.name:
                        # Interested only in the assignment nodes that
                        # decorates the current method.
                        continue
                    try:
                        meth = frame[self.name]
                    except KeyError:
                        continue
                    else:
                        # Must be a function and in the same frame as the
                        # original method.
                        if (
                            isinstance(meth, FunctionDef)
                            and assign_node.frame(future=True) == frame
                        ):
                            decorators.append(assign.value)
        return decorators

    @cached_property
    def type(self) -> str:  # pylint: disable=too-many-return-statements # noqa: C901
        """The function type for this node.

        Possible values are: method, function, staticmethod, classmethod.
        """
        for decorator in self.extra_decorators:
            if decorator.func.name in BUILTIN_DESCRIPTORS:
                return decorator.func.name

        frame = self.parent.frame(future=True)
        type_name = "function"
        if isinstance(frame, ClassDef):
            if self.name == "__new__":
                return "classmethod"
            if self.name == "__init_subclass__":
                return "classmethod"
            if self.name == "__class_getitem__":
                return "classmethod"

            type_name = "method"

        if not self.decorators:
            return type_name

        for node in self.decorators.nodes:
            if isinstance(node, node_classes.Name):
                if node.name in BUILTIN_DESCRIPTORS:
                    return node.name
            if (
                isinstance(node, node_classes.Attribute)
                and isinstance(node.expr, node_classes.Name)
                and node.expr.name == "builtins"
                and node.attrname in BUILTIN_DESCRIPTORS
            ):
                return node.attrname

            if isinstance(node, node_classes.Call):
                # Handle the following case:
                # @some_decorator(arg1, arg2)
                # def func(...)
                #
                try:
                    current = next(node.func.infer())
                except (InferenceError, StopIteration):
                    continue
                _type = _infer_decorator_callchain(current)
                if _type is not None:
                    return _type

            try:
                for inferred in node.infer():
                    # Check to see if this returns a static or a class method.
                    _type = _infer_decorator_callchain(inferred)
                    if _type is not None:
                        return _type

                    if not isinstance(inferred, ClassDef):
                        continue
                    for ancestor in inferred.ancestors():
                        if not isinstance(ancestor, ClassDef):
                            continue
                        if ancestor.is_subtype_of("builtins.classmethod"):
                            return "classmethod"
                        if ancestor.is_subtype_of("builtins.staticmethod"):
                            return "staticmethod"
            except InferenceError:
                pass
        return type_name

    @cached_property
    def fromlineno(self) -> int | None:
        """The first line that this node appears on in the source code."""
        # lineno is the line number of the first decorator, we want the def
        # statement lineno. Similar to 'ClassDef.fromlineno'
        lineno = self.lineno
        if self.decorators is not None:
            lineno += sum(
                node.tolineno - node.lineno + 1 for node in self.decorators.nodes
            )

        return lineno

    @cached_property
    def blockstart_tolineno(self):
        """The line on which the beginning of this block ends.

        :type: int
        """
        return self.args.tolineno

    def implicit_parameters(self) -> Literal[0, 1]:
        return 1 if self.is_bound() else 0

    def block_range(self, lineno):
        """Get a range from the given line number to where this node ends.

        :param lineno: Unused.
        :type lineno: int

        :returns: The range of line numbers that this node belongs to,
        :rtype: tuple(int, int)
        """
        return self.fromlineno, self.tolineno

    def igetattr(self, name, context: InferenceContext | None = None):
        """Inferred getattr, which returns an iterator of inferred statements."""
        try:
            return bases._infer_stmts(self.getattr(name, context), context, frame=self)
        except AttributeInferenceError as error:
            raise InferenceError(
                str(error), target=self, attribute=name, context=context
            ) from error

    def is_method(self) -> bool:
        """Check if this function node represents a method.

        :returns: Whether this is a method.
        """
        # check we are defined in a ClassDef, because this is usually expected
        # (e.g. pylint...) when is_method() return True
        return self.type != "function" and isinstance(
            self.parent.frame(future=True), ClassDef
        )

    @decorators_mod.cached
    def decoratornames(self, context: InferenceContext | None = None):
        """Get the qualified names of each of the decorators on this function.

        :param context:
            An inference context that can be passed to inference functions
        :returns: The names of the decorators.
        :rtype: set(str)
        """
        result = set()
        decoratornodes = []
        if self.decorators is not None:
            decoratornodes += self.decorators.nodes
        decoratornodes += self.extra_decorators
        for decnode in decoratornodes:
            try:
                for infnode in decnode.infer(context=context):
                    result.add(infnode.qname())
            except InferenceError:
                continue
        return result

    def is_bound(self) -> bool:
        """Check if the function is bound to an instance or class.

        :returns: Whether the function is bound to an instance or class.
        """
        return self.type in {"method", "classmethod"}

    def is_abstract(self, pass_is_abstract=True, any_raise_is_abstract=False) -> bool:
        """Check if the method is abstract.

        A method is considered abstract if any of the following is true:
        * The only statement is 'raise NotImplementedError'
        * The only statement is 'raise <SomeException>' and any_raise_is_abstract is True
        * The only statement is 'pass' and pass_is_abstract is True
        * The method is annotated with abc.astractproperty/abc.abstractmethod

        :returns: Whether the method is abstract.
        """
        if self.decorators:
            for node in self.decorators.nodes:
                try:
                    inferred = next(node.infer())
                except (InferenceError, StopIteration):
                    continue
                if inferred and inferred.qname() in {
                    "abc.abstractproperty",
                    "abc.abstractmethod",
                }:
                    return True

        for child_node in self.body:
            if isinstance(child_node, node_classes.Raise):
                if any_raise_is_abstract:
                    return True
                if child_node.raises_not_implemented():
                    return True
            return pass_is_abstract and isinstance(child_node, node_classes.Pass)
        # empty function is the same as function with a single "pass" statement
        if pass_is_abstract:
            return True

        return False

    def is_generator(self) -> bool:
        """Check if this is a generator function.

        :returns: Whether this is a generator function.
        """
        return bool(next(self._get_yield_nodes_skip_lambdas(), False))

    def infer_yield_result(self, context: InferenceContext | None = None):
        """Infer what the function yields when called

        :returns: What the function yields
        :rtype: iterable(NodeNG or Uninferable) or None
        """
        # pylint: disable=not-an-iterable
        # https://github.com/PyCQA/astroid/issues/1015
        for yield_ in self.nodes_of_class(node_classes.Yield):
            if yield_.value is None:
                const = node_classes.Const(None)
                const.parent = yield_
                const.lineno = yield_.lineno
                yield const
            elif yield_.scope() == self:
                yield from yield_.value.infer(context=context)

    def infer_call_result(self, caller=None, context: InferenceContext | None = None):
        """Infer what the function returns when called.

        :returns: What the function returns.
        :rtype: iterable(NodeNG or Uninferable) or None
        """
        if self.is_generator():
            if isinstance(self, AsyncFunctionDef):
                generator_cls = bases.AsyncGenerator
            else:
                generator_cls = bases.Generator
            result = generator_cls(self, generator_initial_context=context)
            yield result
            return
        # This is really a gigantic hack to work around metaclass generators
        # that return transient class-generating functions. Pylint's AST structure
        # cannot handle a base class object that is only used for calling __new__,
        # but does not contribute to the inheritance structure itself. We inject
        # a fake class into the hierarchy here for several well-known metaclass
        # generators, and filter it out later.
        if (
            self.name == "with_metaclass"
            and len(self.args.args) == 1
            and self.args.vararg is not None
        ):
            metaclass = next(caller.args[0].infer(context), None)
            if isinstance(metaclass, ClassDef):
                try:
                    class_bases = [next(arg.infer(context)) for arg in caller.args[1:]]
                except StopIteration as e:
                    raise InferenceError(node=caller.args[1:], context=context) from e
                new_class = ClassDef(name="temporary_class")
                new_class.hide = True
                new_class.parent = self
                new_class.postinit(
                    bases=[base for base in class_bases if base != util.Uninferable],
                    body=[],
                    decorators=[],
                    metaclass=metaclass,
                )
                yield new_class
                return
        returns = self._get_return_nodes_skip_functions()

        first_return = next(returns, None)
        if not first_return:
            if self.body:
                if self.is_abstract(pass_is_abstract=True, any_raise_is_abstract=True):
                    yield util.Uninferable
                else:
                    yield node_classes.Const(None)
                return

            raise InferenceError("The function does not have any return statements")

        for returnnode in itertools.chain((first_return,), returns):
            if returnnode.value is None:
                yield node_classes.Const(None)
            else:
                try:
                    yield from returnnode.value.infer(context)
                except InferenceError:
                    yield util.Uninferable

    def bool_value(self, context: InferenceContext | None = None) -> bool:
        """Determine the boolean value of this node.

        :returns: The boolean value of this node.
            For a :class:`FunctionDef` this is always ``True``.
        """
        return True

    def get_children(self):
        if self.decorators is not None:
            yield self.decorators

        yield self.args

        if self.returns is not None:
            yield self.returns

        yield from self.body

    def scope_lookup(self, node, name, offset=0):
        """Lookup where the given name is assigned."""
        if name == "__class__":
            # __class__ is an implicit closure reference created by the compiler
            # if any methods in a class body refer to either __class__ or super.
            # In our case, we want to be able to look it up in the current scope
            # when `__class__` is being used.
            frame = self.parent.frame(future=True)
            if isinstance(frame, ClassDef):
                return self, [frame]
        return super().scope_lookup(node, name, offset)

    def frame(self: _T, *, future: Literal[None, True] = None) -> _T:
        """The node's frame node.

        A frame node is a :class:`Module`, :class:`FunctionDef`,
        :class:`ClassDef` or :class:`Lambda`.

        :returns: The node itself.
        """
        return self


class AsyncFunctionDef(FunctionDef):
    """Class representing an :class:`ast.FunctionDef` node.

    A :class:`AsyncFunctionDef` is an asynchronous function
    created with the `async` keyword.

    >>> import astroid
    >>> node = astroid.extract_node('''
    async def func(things):
        async for thing in things:
            print(thing)
    ''')
    >>> node
    <AsyncFunctionDef.func l.2 at 0x7f23b2e416d8>
    >>> node.body[0]
    <AsyncFor l.3 at 0x7f23b2e417b8>
    """


def _rec_get_names(args, names: list[str] | None = None) -> list[str]:
    """return a list of all argument names"""
    if names is None:
        names = []
    for arg in args:
        if isinstance(arg, node_classes.Tuple):
            _rec_get_names(arg.elts, names)
        else:
            names.append(arg.name)
    return names


def _is_metaclass(klass, seen=None) -> bool:
    """Return if the given class can be
    used as a metaclass.
    """
    if klass.name == "type":
        return True
    if seen is None:
        seen = set()
    for base in klass.bases:
        try:
            for baseobj in base.infer():
                baseobj_name = baseobj.qname()
                if baseobj_name in seen:
                    continue

                seen.add(baseobj_name)
                if isinstance(baseobj, bases.Instance):
                    # not abstract
                    return False
                if baseobj is util.Uninferable:
                    continue
                if baseobj is klass:
                    continue
                if not isinstance(baseobj, ClassDef):
                    continue
                if baseobj._type == "metaclass":
                    return True
                if _is_metaclass(baseobj, seen):
                    return True
        except InferenceError:
            continue
    return False


def _class_type(klass, ancestors=None):
    """return a ClassDef node type to differ metaclass and exception
    from 'regular' classes
    """
    # XXX we have to store ancestors in case we have an ancestor loop
    if klass._type is not None:
        return klass._type
    if _is_metaclass(klass):
        klass._type = "metaclass"
    elif klass.name.endswith("Exception"):
        klass._type = "exception"
    else:
        if ancestors is None:
            ancestors = set()
        klass_name = klass.qname()
        if klass_name in ancestors:
            # XXX we are in loop ancestors, and have found no type
            klass._type = "class"
            return "class"
        ancestors.add(klass_name)
        for base in klass.ancestors(recurs=False):
            name = _class_type(base, ancestors)
            if name != "class":
                if name == "metaclass" and not _is_metaclass(klass):
                    # don't propagate it if the current class
                    # can't be a metaclass
                    continue
                klass._type = base.type
                break
    if klass._type is None:
        klass._type = "class"
    return klass._type


def get_wrapping_class(node):
    """Get the class that wraps the given node.

    We consider that a class wraps a node if the class
    is a parent for the said node.

    :returns: The class that wraps the given node
    :rtype: ClassDef or None
    """

    klass = node.frame(future=True)
    while klass is not None and not isinstance(klass, ClassDef):
        if klass.parent is None:
            klass = None
        else:
            klass = klass.parent.frame(future=True)
    return klass


# pylint: disable=too-many-instance-attributes
class ClassDef(
    _base_nodes.FilterStmtsBaseNode, LocalsDictNodeNG, _base_nodes.Statement
):
    """Class representing an :class:`ast.ClassDef` node.

    >>> import astroid
    >>> node = astroid.extract_node('''
    class Thing:
        def my_meth(self, arg):
            return arg + self.offset
    ''')
    >>> node
    <ClassDef.Thing l.2 at 0x7f23b2e9e748>
    """

    # some of the attributes below are set by the builder module or
    # by a raw factories

    # a dictionary of class instances attributes
    _astroid_fields = ("decorators", "bases", "keywords", "doc_node", "body")  # name

    decorators = None
    """The decorators that are applied to this class.

    :type: Decorators or None
    """
    special_attributes = ClassModel()
    """The names of special attributes that this class has.

    :type: objectmodel.ClassModel
    """

    _type = None
    _metaclass: NodeNG | None = None
    _metaclass_hack = False
    hide = False
    type = property(
        _class_type,
        doc=(
            "The class type for this node.\n\n"
            "Possible values are: class, metaclass, exception.\n\n"
            ":type: str"
        ),
    )
    _other_fields = ("name", "doc", "is_dataclass", "position")
    _other_other_fields = ("locals", "_newstyle")
    _newstyle = None

    @decorators_mod.deprecate_arguments(doc="Use the postinit arg 'doc_node' instead")
    def __init__(
        self,
        name=None,
        doc: str | None = None,
        lineno=None,
        col_offset=None,
        parent=None,
        *,
        end_lineno=None,
        end_col_offset=None,
    ):
        """
        :param name: The name of the class.
        :type name: str or None

        :param doc: The class docstring.

        :param lineno: The line that this node appears on in the source code.
        :type lineno: int or None

        :param col_offset: The column that this node appears on in the
            source code.
        :type col_offset: int or None

        :param parent: The parent node in the syntax tree.
        :type parent: NodeNG or None

        :param end_lineno: The last line this node appears on in the source code.
        :type end_lineno: Optional[int]

        :param end_col_offset: The end column this node appears on in the
            source code. Note: This is after the last symbol.
        :type end_col_offset: Optional[int]
        """
        self.instance_attrs = {}
        self.locals = {}
        """A map of the name of a local variable to the node defining it."""

        self.keywords = []
        """The keywords given to the class definition.

        This is usually for :pep:`3115` style metaclass declaration.

        :type: list(Keyword) or None
        """

        self.bases: list[NodeNG] = []
        """What the class inherits from."""

        self.body = []
        """The contents of the class body.

        :type: list(NodeNG)
        """

        self.name = name
        """The name of the class.

        :type name: str or None
        """

        self._doc = doc
        """The class docstring."""

        self.doc_node: Const | None = None
        """The doc node associated with this node."""

        self.is_dataclass: bool = False
        """Whether this class is a dataclass."""

        super().__init__(
            lineno=lineno,
            col_offset=col_offset,
            end_lineno=end_lineno,
            end_col_offset=end_col_offset,
            parent=parent,
        )
        if parent is not None:
            parent.frame(future=True).set_local(name, self)

        for local_name, node in self.implicit_locals():
            self.add_local_node(node, local_name)

    infer_binary_op: ClassVar[InferBinaryOp[ClassDef]]

    @property
    def doc(self) -> str | None:
        """The class docstring."""
        warnings.warn(
            "The 'ClassDef.doc' attribute is deprecated, "
            "use 'ClassDef.doc_node' instead.",
            DeprecationWarning,
        )
        return self._doc

    @doc.setter
    def doc(self, value: str | None) -> None:
        warnings.warn(
            "Setting the 'ClassDef.doc' attribute is deprecated, "
            "use 'ClassDef.doc_node.value' instead.",
            DeprecationWarning,
        )
        self._doc = value

    def implicit_parameters(self) -> Literal[1]:
        return 1

    def implicit_locals(self):
        """Get implicitly defined class definition locals.

        :returns: the the name and Const pair for each local
        :rtype: tuple(tuple(str, node_classes.Const), ...)
        """
        locals_ = (("__module__", self.special_attributes.attr___module__),)
        # __qualname__ is defined in PEP3155
        locals_ += (("__qualname__", self.special_attributes.attr___qualname__),)
        return locals_

    # pylint: disable=redefined-outer-name
    def postinit(
        self,
        bases,
        body,
        decorators,
        newstyle=None,
        metaclass: NodeNG | None = None,
        keywords=None,
        *,
        position: Position | None = None,
        doc_node: Const | None = None,
    ):
        """Do some setup after initialisation.

        :param bases: What the class inherits from.
        :type bases: list(NodeNG)

        :param body: The contents of the class body.
        :type body: list(NodeNG)

        :param decorators: The decorators that are applied to this class.
        :type decorators: Decorators or None

        :param newstyle: Whether this is a new style class or not.
        :type newstyle: bool or None

        :param metaclass: The metaclass of this class.

        :param keywords: The keywords given to the class definition.
        :type keywords: list(Keyword) or None

        :param position: Position of class keyword and name.

        :param doc_node: The doc node associated with this node.
        """
        if keywords is not None:
            self.keywords = keywords
        self.bases = bases
        self.body = body
        self.decorators = decorators
        if newstyle is not None:
            self._newstyle = newstyle
        if metaclass is not None:
            self._metaclass = metaclass
        self.position = position
        self.doc_node = doc_node
        if doc_node:
            self._doc = doc_node.value

    def _newstyle_impl(self, context: InferenceContext | None = None):
        if context is None:
            context = InferenceContext()
        if self._newstyle is not None:
            return self._newstyle
        for base in self.ancestors(recurs=False, context=context):
            if base._newstyle_impl(context):
                self._newstyle = True
                break
        klass = self.declared_metaclass()
        # could be any callable, we'd need to infer the result of klass(name,
        # bases, dict).  punt if it's not a class node.
        if klass is not None and isinstance(klass, ClassDef):
            self._newstyle = klass._newstyle_impl(context)
        if self._newstyle is None:
            self._newstyle = False
        return self._newstyle

    _newstyle = None
    newstyle = property(
        _newstyle_impl,
        doc=("Whether this is a new style class or not\n\n" ":type: bool or None"),
    )

    @cached_property
    def fromlineno(self) -> int | None:
        """The first line that this node appears on in the source code."""
        if not PY38_PLUS or PY38 and IS_PYPY:
            # For Python < 3.8 the lineno is the line number of the first decorator.
            # We want the class statement lineno. Similar to 'FunctionDef.fromlineno'
            lineno = self.lineno
            if self.decorators is not None:
                lineno += sum(
                    node.tolineno - node.lineno + 1 for node in self.decorators.nodes
                )

            return lineno
        return super().fromlineno

    @cached_property
    def blockstart_tolineno(self):
        """The line on which the beginning of this block ends.

        :type: int
        """
        if self.bases:
            return self.bases[-1].tolineno

        return self.fromlineno

    def block_range(self, lineno):
        """Get a range from the given line number to where this node ends.

        :param lineno: Unused.
        :type lineno: int

        :returns: The range of line numbers that this node belongs to,
        :rtype: tuple(int, int)
        """
        return self.fromlineno, self.tolineno

    def pytype(self) -> Literal["builtins.type", "builtins.classobj"]:
        """Get the name of the type that this node represents.

        :returns: The name of the type.
        """
        if self.newstyle:
            return "builtins.type"
        return "builtins.classobj"

    def display_type(self) -> str:
        """A human readable type of this node.

        :returns: The type of this node.
        :rtype: str
        """
        return "Class"

    def callable(self) -> bool:
        """Whether this node defines something that is callable.

        :returns: Whether this defines something that is callable.
            For a :class:`ClassDef` this is always ``True``.
        """
        return True

    def is_subtype_of(self, type_name, context: InferenceContext | None = None) -> bool:
        """Whether this class is a subtype of the given type.

        :param type_name: The name of the type of check against.
        :type type_name: str

        :returns: Whether this class is a subtype of the given type.
        """
        if self.qname() == type_name:
            return True

        return any(anc.qname() == type_name for anc in self.ancestors(context=context))

    def _infer_type_call(self, caller, context):
        try:
            name_node = next(caller.args[0].infer(context))
        except StopIteration as e:
            raise InferenceError(node=caller.args[0], context=context) from e
        if isinstance(name_node, node_classes.Const) and isinstance(
            name_node.value, str
        ):
            name = name_node.value
        else:
            return util.Uninferable

        result = ClassDef(name)

        # Get the bases of the class.
        try:
            class_bases = next(caller.args[1].infer(context))
        except StopIteration as e:
            raise InferenceError(node=caller.args[1], context=context) from e
        if isinstance(class_bases, (node_classes.Tuple, node_classes.List)):
            bases = []
            for base in class_bases.itered():
                inferred = next(base.infer(context=context), None)
                if inferred:
                    bases.append(
                        node_classes.EvaluatedObject(original=base, value=inferred)
                    )
            result.bases = bases
        else:
            # There is currently no AST node that can represent an 'unknown'
            # node (Uninferable is not an AST node), therefore we simply return Uninferable here
            # although we know at least the name of the class.
            return util.Uninferable

        # Get the members of the class
        try:
            members = next(caller.args[2].infer(context))
        except (InferenceError, StopIteration):
            members = None

        if members and isinstance(members, node_classes.Dict):
            for attr, value in members.items:
                if isinstance(attr, node_classes.Const) and isinstance(attr.value, str):
                    result.locals[attr.value] = [value]

        result.parent = caller.parent
        return result

    def infer_call_result(self, caller, context: InferenceContext | None = None):
        """infer what a class is returning when called"""
        if self.is_subtype_of("builtins.type", context) and len(caller.args) == 3:
            result = self._infer_type_call(caller, context)
            yield result
            return

        dunder_call = None
        try:
            metaclass = self.metaclass(context=context)
            if metaclass is not None:
                # Only get __call__ if it's defined locally for the metaclass.
                # Otherwise we will find ObjectModel.__call__ which will
                # return an instance of the metaclass. Instantiating the class is
                # handled later.
                if "__call__" in metaclass.locals:
                    dunder_call = next(metaclass.igetattr("__call__", context))
        except (AttributeInferenceError, StopIteration):
            pass

        if dunder_call and dunder_call.qname() != "builtins.type.__call__":
            # Call type.__call__ if not set metaclass
            # (since type is the default metaclass)
            context = bind_context_to_node(context, self)
            context.callcontext.callee = dunder_call
            yield from dunder_call.infer_call_result(caller, context)
        else:
            yield self.instantiate_class()

    def scope_lookup(self, node, name, offset=0):
        """Lookup where the given name is assigned.

        :param node: The node to look for assignments up to.
            Any assignments after the given node are ignored.
        :type node: NodeNG

        :param name: The name to find assignments for.
        :type name: str

        :param offset: The line offset to filter statements up to.
        :type offset: int

        :returns: This scope node and the list of assignments associated to the
            given name according to the scope where it has been found (locals,
            globals or builtin).
        :rtype: tuple(str, list(NodeNG))
        """
        # If the name looks like a builtin name, just try to look
        # into the upper scope of this class. We might have a
        # decorator that it's poorly named after a builtin object
        # inside this class.
        lookup_upper_frame = (
            isinstance(node.parent, node_classes.Decorators)
            and name in AstroidManager().builtins_module
        )
        if (
            any(node == base or base.parent_of(node) for base in self.bases)
            or lookup_upper_frame
        ):
            # Handle the case where we have either a name
            # in the bases of a class, which exists before
            # the actual definition or the case where we have
            # a Getattr node, with that name.
            #
            # name = ...
            # class A(name):
            #     def name(self): ...
            #
            # import name
            # class A(name.Name):
            #     def name(self): ...

            frame = self.parent.frame(future=True)
            # line offset to avoid that class A(A) resolve the ancestor to
            # the defined class
            offset = -1
        else:
            frame = self
        return frame._scope_lookup(node, name, offset)

    @property
    def basenames(self):
        """The names of the parent classes

        Names are given in the order they appear in the class definition.

        :type: list(str)
        """
        return [bnode.as_string() for bnode in self.bases]

    def ancestors(
        self, recurs: bool = True, context: InferenceContext | None = None
    ) -> Generator[ClassDef, None, None]:
        """Iterate over the base classes in prefixed depth first order.

        :param recurs: Whether to recurse or return direct ancestors only.

        :returns: The base classes
        """
        # FIXME: should be possible to choose the resolution order
        # FIXME: inference make infinite loops possible here
        yielded = {self}
        if context is None:
            context = InferenceContext()
        if not self.bases and self.qname() != "builtins.object":
            yield builtin_lookup("object")[1][0]
            return

        for stmt in self.bases:
            with context.restore_path():
                try:
                    for baseobj in stmt.infer(context):
                        if not isinstance(baseobj, ClassDef):
                            if isinstance(baseobj, bases.Instance):
                                baseobj = baseobj._proxied
                            else:
                                continue
                        if not baseobj.hide:
                            if baseobj in yielded:
                                continue
                            yielded.add(baseobj)
                            yield baseobj
                        if not recurs:
                            continue
                        for grandpa in baseobj.ancestors(recurs=True, context=context):
                            if grandpa is self:
                                # This class is the ancestor of itself.
                                break
                            if grandpa in yielded:
                                continue
                            yielded.add(grandpa)
                            yield grandpa
                except InferenceError:
                    continue

    def local_attr_ancestors(self, name, context: InferenceContext | None = None):
        """Iterate over the parents that define the given name.

        :param name: The name to find definitions for.
        :type name: str

        :returns: The parents that define the given name.
        :rtype: iterable(NodeNG)
        """
        # Look up in the mro if we can. This will result in the
        # attribute being looked up just as Python does it.
        try:
            ancestors = self.mro(context)[1:]
        except MroError:
            # Fallback to use ancestors, we can't determine
            # a sane MRO.
            ancestors = self.ancestors(context=context)
        for astroid in ancestors:
            if name in astroid:
                yield astroid

    def instance_attr_ancestors(self, name, context: InferenceContext | None = None):
        """Iterate over the parents that define the given name as an attribute.

        :param name: The name to find definitions for.
        :type name: str

        :returns: The parents that define the given name as
            an instance attribute.
        :rtype: iterable(NodeNG)
        """
        for astroid in self.ancestors(context=context):
            if name in astroid.instance_attrs:
                yield astroid

    def has_base(self, node) -> bool:
        """Whether this class directly inherits from the given node.

        :param node: The node to check for.
        :type node: NodeNG

        :returns: Whether this class directly inherits from the given node.
        """
        return node in self.bases

    def local_attr(self, name, context: InferenceContext | None = None):
        """Get the list of assign nodes associated to the given name.

        Assignments are looked for in both this class and in parents.

        :returns: The list of assignments to the given name.
        :rtype: list(NodeNG)

        :raises AttributeInferenceError: If no attribute with this name
            can be found in this class or parent classes.
        """
        result = []
        if name in self.locals:
            result = self.locals[name]
        else:
            class_node = next(self.local_attr_ancestors(name, context), None)
            if class_node:
                result = class_node.locals[name]
        result = [n for n in result if not isinstance(n, node_classes.DelAttr)]
        if result:
            return result
        raise AttributeInferenceError(target=self, attribute=name, context=context)

    def instance_attr(self, name, context: InferenceContext | None = None):
        """Get the list of nodes associated to the given attribute name.

        Assignments are looked for in both this class and in parents.

        :returns: The list of assignments to the given name.
        :rtype: list(NodeNG)

        :raises AttributeInferenceError: If no attribute with this name
            can be found in this class or parent classes.
        """
        # Return a copy, so we don't modify self.instance_attrs,
        # which could lead to infinite loop.
        values = list(self.instance_attrs.get(name, []))
        # get all values from parents
        for class_node in self.instance_attr_ancestors(name, context):
            values += class_node.instance_attrs[name]
        values = [n for n in values if not isinstance(n, node_classes.DelAttr)]
        if values:
            return values
        raise AttributeInferenceError(target=self, attribute=name, context=context)

    def instantiate_class(self) -> bases.Instance:
        """Get an :class:`Instance` of the :class:`ClassDef` node.

        :returns: An :class:`Instance` of the :class:`ClassDef` node
        """
        try:
            if any(cls.name in EXCEPTION_BASE_CLASSES for cls in self.mro()):
                # Subclasses of exceptions can be exception instances
                return objects.ExceptionInstance(self)
        except MroError:
            pass
        return bases.Instance(self)

    def getattr(
        self,
        name: str,
        context: InferenceContext | None = None,
        class_context: bool = True,
    ) -> list[SuccessfulInferenceResult]:
        """Get an attribute from this class, using Python's attribute semantic.

        This method doesn't look in the :attr:`instance_attrs` dictionary
        since it is done by an :class:`Instance` proxy at inference time.
        It may return an :class:`Uninferable` object if
        the attribute has not been
        found, but a ``__getattr__`` or ``__getattribute__`` method is defined.
        If ``class_context`` is given, then it is considered that the
        attribute is accessed from a class context,
        e.g. ClassDef.attribute, otherwise it might have been accessed
        from an instance as well. If ``class_context`` is used in that
        case, then a lookup in the implicit metaclass and the explicit
        metaclass will be done.

        :param name: The attribute to look for.

        :param class_context: Whether the attribute can be accessed statically.

        :returns: The attribute.

        :raises AttributeInferenceError: If the attribute cannot be inferred.
        """
        if not name:
            raise AttributeInferenceError(target=self, attribute=name, context=context)

        # don't modify the list in self.locals!
        values: list[SuccessfulInferenceResult] = list(self.locals.get(name, []))
        for classnode in self.ancestors(recurs=True, context=context):
            values += classnode.locals.get(name, [])

        if name in self.special_attributes and class_context and not values:
            result = [self.special_attributes.lookup(name)]
            if name == "__bases__":
                # Need special treatment, since they are mutable
                # and we need to return all the values.
                result += values
            return result

        if class_context:
            values += self._metaclass_lookup_attribute(name, context)

        # Remove AnnAssigns without value, which are not attributes in the purest sense.
        for value in values.copy():
            if isinstance(value, node_classes.AssignName):
                stmt = value.statement(future=True)
                if isinstance(stmt, node_classes.AnnAssign) and stmt.value is None:
                    values.pop(values.index(value))

        if not values:
            raise AttributeInferenceError(target=self, attribute=name, context=context)

        return values

    @lru_cache(maxsize=1024)  # noqa
    def _metaclass_lookup_attribute(self, name, context):
        """Search the given name in the implicit and the explicit metaclass."""
        attrs = set()
        implicit_meta = self.implicit_metaclass()
        context = copy_context(context)
        metaclass = self.metaclass(context=context)
        for cls in (implicit_meta, metaclass):
            if cls and cls != self and isinstance(cls, ClassDef):
                cls_attributes = self._get_attribute_from_metaclass(cls, name, context)
                attrs.update(set(cls_attributes))
        return attrs

    def _get_attribute_from_metaclass(self, cls, name, context):
        try:
            attrs = cls.getattr(name, context=context, class_context=True)
        except AttributeInferenceError:
            return

        for attr in bases._infer_stmts(attrs, context, frame=cls):
            if not isinstance(attr, FunctionDef):
                yield attr
                continue

            if isinstance(attr, objects.Property):
                yield attr
                continue
            if attr.type == "classmethod":
                # If the method is a classmethod, then it will
                # be bound to the metaclass, not to the class
                # from where the attribute is retrieved.
                # get_wrapping_class could return None, so just
                # default to the current class.
                frame = get_wrapping_class(attr) or self
                yield bases.BoundMethod(attr, frame)
            elif attr.type == "staticmethod":
                yield attr
            else:
                yield bases.BoundMethod(attr, self)

    def igetattr(
        self,
        name: str,
        context: InferenceContext | None = None,
        class_context: bool = True,
    ) -> Iterator[InferenceResult]:
        """Infer the possible values of the given variable.

        :param name: The name of the variable to infer.

        :returns: The inferred possible values.
        """
        # set lookup name since this is necessary to infer on import nodes for
        # instance
        context = copy_context(context)
        context.lookupname = name

        metaclass = self.metaclass(context=context)
        try:
            attributes = self.getattr(name, context, class_context=class_context)
            # If we have more than one attribute, make sure that those starting from
            # the second one are from the same scope. This is to account for modifications
            # to the attribute happening *after* the attribute's definition (e.g. AugAssigns on lists)
            if len(attributes) > 1:
                first_attr, attributes = attributes[0], attributes[1:]
                first_scope = first_attr.scope()
                attributes = [first_attr] + [
                    attr
                    for attr in attributes
                    if attr.parent and attr.parent.scope() == first_scope
                ]

            for inferred in bases._infer_stmts(attributes, context, frame=self):
                # yield Uninferable object instead of descriptors when necessary
                if not isinstance(inferred, node_classes.Const) and isinstance(
                    inferred, bases.Instance
                ):
                    try:
                        inferred._proxied.getattr("__get__", context)
                    except AttributeInferenceError:
                        yield inferred
                    else:
                        yield util.Uninferable
                elif isinstance(inferred, objects.Property):
                    function = inferred.function
                    if not class_context:
                        # Through an instance so we can solve the property
                        yield from function.infer_call_result(
                            caller=self, context=context
                        )
                    # If we're in a class context, we need to determine if the property
                    # was defined in the metaclass (a derived class must be a subclass of
                    # the metaclass of all its bases), in which case we can resolve the
                    # property. If not, i.e. the property is defined in some base class
                    # instead, then we return the property object
                    elif metaclass and function.parent.scope() is metaclass:
                        # Resolve a property as long as it is not accessed through
                        # the class itself.
                        yield from function.infer_call_result(
                            caller=self, context=context
                        )
                    else:
                        yield inferred
                else:
                    yield function_to_method(inferred, self)
        except AttributeInferenceError as error:
            if not name.startswith("__") and self.has_dynamic_getattr(context):
                # class handle some dynamic attributes, return a Uninferable object
                yield util.Uninferable
            else:
                raise InferenceError(
                    str(error), target=self, attribute=name, context=context
                ) from error

    def has_dynamic_getattr(self, context: InferenceContext | None = None) -> bool:
        """Check if the class has a custom __getattr__ or __getattribute__.

        If any such method is found and it is not from
        builtins, nor from an extension module, then the function
        will return True.

        :returns: Whether the class has a custom __getattr__ or __getattribute__.
        """

        def _valid_getattr(node):
            root = node.root()
            return root.name != "builtins" and getattr(root, "pure_python", None)

        try:
            return _valid_getattr(self.getattr("__getattr__", context)[0])
        except AttributeInferenceError:
            # if self.newstyle: XXX cause an infinite recursion error
            try:
                getattribute = self.getattr("__getattribute__", context)[0]
                return _valid_getattr(getattribute)
            except AttributeInferenceError:
                pass
        return False

    def getitem(self, index, context: InferenceContext | None = None):
        """Return the inference of a subscript.

        This is basically looking up the method in the metaclass and calling it.

        :returns: The inferred value of a subscript to this class.
        :rtype: NodeNG

        :raises AstroidTypeError: If this class does not define a
            ``__getitem__`` method.
        """
        try:
            methods = lookup(self, "__getitem__")
        except AttributeInferenceError as exc:
            if isinstance(self, ClassDef):
                # subscripting a class definition may be
                # achieved thanks to __class_getitem__ method
                # which is a classmethod defined in the class
                # that supports subscript and not in the metaclass
                try:
                    methods = self.getattr("__class_getitem__")
                    # Here it is assumed that the __class_getitem__ node is
                    # a FunctionDef. One possible improvement would be to deal
                    # with more generic inference.
                except AttributeInferenceError:
                    raise AstroidTypeError(node=self, context=context) from exc
            else:
                raise AstroidTypeError(node=self, context=context) from exc

        method = methods[0]

        # Create a new callcontext for providing index as an argument.
        new_context = bind_context_to_node(context, self)
        new_context.callcontext = CallContext(args=[index], callee=method)

        try:
            return next(method.infer_call_result(self, new_context), util.Uninferable)
        except AttributeError:
            # Starting with python3.9, builtin types list, dict etc...
            # are subscriptable thanks to __class_getitem___ classmethod.
            # However in such case the method is bound to an EmptyNode and
            # EmptyNode doesn't have infer_call_result method yielding to
            # AttributeError
            if (
                isinstance(method, node_classes.EmptyNode)
                and self.pytype() == "builtins.type"
                and PY39_PLUS
            ):
                return self
            raise
        except InferenceError:
            return util.Uninferable

    def methods(self):
        """Iterate over all of the method defined in this class and its parents.

        :returns: The methods defined on the class.
        :rtype: iterable(FunctionDef)
        """
        done = {}
        for astroid in itertools.chain(iter((self,)), self.ancestors()):
            for meth in astroid.mymethods():
                if meth.name in done:
                    continue
                done[meth.name] = None
                yield meth

    def mymethods(self):
        """Iterate over all of the method defined in this class only.

        :returns: The methods defined on the class.
        :rtype: iterable(FunctionDef)
        """
        for member in self.values():
            if isinstance(member, FunctionDef):
                yield member

    def implicit_metaclass(self):
        """Get the implicit metaclass of the current class.

        For newstyle classes, this will return an instance of builtins.type.
        For oldstyle classes, it will simply return None, since there's
        no implicit metaclass there.

        :returns: The metaclass.
        :rtype: builtins.type or None
        """
        if self.newstyle:
            return builtin_lookup("type")[1][0]
        return None

    def declared_metaclass(
        self, context: InferenceContext | None = None
    ) -> NodeNG | None:
        """Return the explicit declared metaclass for the current class.

        An explicit declared metaclass is defined
        either by passing the ``metaclass`` keyword argument
        in the class definition line (Python 3) or (Python 2) by
        having a ``__metaclass__`` class attribute, or if there are
        no explicit bases but there is a global ``__metaclass__`` variable.

        :returns: The metaclass of this class,
            or None if one could not be found.
        """
        for base in self.bases:
            try:
                for baseobj in base.infer(context=context):
                    if isinstance(baseobj, ClassDef) and baseobj.hide:
                        self._metaclass = baseobj._metaclass
                        self._metaclass_hack = True
                        break
            except InferenceError:
                pass

        if self._metaclass:
            # Expects this from Py3k TreeRebuilder
            try:
                return next(
                    node
                    for node in self._metaclass.infer(context=context)
                    if node is not util.Uninferable
                )
            except (InferenceError, StopIteration):
                return None

        return None

    def _find_metaclass(
        self, seen: set[ClassDef] | None = None, context: InferenceContext | None = None
    ) -> NodeNG | None:
        if seen is None:
            seen = set()
        seen.add(self)

        klass = self.declared_metaclass(context=context)
        if klass is None:
            for parent in self.ancestors(context=context):
                if parent not in seen:
                    klass = parent._find_metaclass(seen)
                    if klass is not None:
                        break
        return klass

    def metaclass(self, context: InferenceContext | None = None) -> NodeNG | None:
        """Get the metaclass of this class.

        If this class does not define explicitly a metaclass,
        then the first defined metaclass in ancestors will be used
        instead.

        :returns: The metaclass of this class.
        """
        return self._find_metaclass(context=context)

    def has_metaclass_hack(self):
        return self._metaclass_hack

    def _islots(self):
        """Return an iterator with the inferred slots."""
        if "__slots__" not in self.locals:
            return None
        for slots in self.igetattr("__slots__"):
            # check if __slots__ is a valid type
            for meth in ITER_METHODS:
                try:
                    slots.getattr(meth)
                    break
                except AttributeInferenceError:
                    continue
            else:
                continue

            if isinstance(slots, node_classes.Const):
                # a string. Ignore the following checks,
                # but yield the node, only if it has a value
                if slots.value:
                    yield slots
                continue
            if not hasattr(slots, "itered"):
                # we can't obtain the values, maybe a .deque?
                continue

            if isinstance(slots, node_classes.Dict):
                values = [item[0] for item in slots.items]
            else:
                values = slots.itered()
            if values is util.Uninferable:
                continue
            if not values:
                # Stop the iteration, because the class
                # has an empty list of slots.
                return values

            for elt in values:
                try:
                    for inferred in elt.infer():
                        if inferred is util.Uninferable:
                            continue
                        if not isinstance(
                            inferred, node_classes.Const
                        ) or not isinstance(inferred.value, str):
                            continue
                        if not inferred.value:
                            continue
                        yield inferred
                except InferenceError:
                    continue

        return None

    def _slots(self):
        if not self.newstyle:
            raise NotImplementedError(
                "The concept of slots is undefined for old-style classes."
            )

        slots = self._islots()
        try:
            first = next(slots)
        except StopIteration as exc:
            # The class doesn't have a __slots__ definition or empty slots.
            if exc.args and exc.args[0] not in ("", None):
                return exc.args[0]
            return None
        return [first] + list(slots)

    # Cached, because inferring them all the time is expensive
    @decorators_mod.cached
    def slots(self):
        """Get all the slots for this node.

        :returns: The names of slots for this class.
            If the class doesn't define any slot, through the ``__slots__``
            variable, then this function will return a None.
            Also, it will return None in the case the slots were not inferred.
        :rtype: list(str) or None
        """

        def grouped_slots(
            mro: list[ClassDef],
        ) -> Iterator[node_classes.NodeNG | None]:
            for cls in mro:
                # Not interested in object, since it can't have slots.
                if cls.qname() == "builtins.object":
                    continue
                try:
                    cls_slots = cls._slots()
                except NotImplementedError:
                    continue
                if cls_slots is not None:
                    yield from cls_slots
                else:
                    yield None

        if not self.newstyle:
            raise NotImplementedError(
                "The concept of slots is undefined for old-style classes."
            )

        try:
            mro = self.mro()
        except MroError as e:
            raise NotImplementedError(
                "Cannot get slots while parsing mro fails."
            ) from e

        slots = list(grouped_slots(mro))
        if not all(slot is not None for slot in slots):
            return None

        return sorted(set(slots), key=lambda item: item.value)

    def _inferred_bases(self, context: InferenceContext | None = None):
        # Similar with .ancestors, but the difference is when one base is inferred,
        # only the first object is wanted. That's because
        # we aren't interested in superclasses, as in the following
        # example:
        #
        # class SomeSuperClass(object): pass
        # class SomeClass(SomeSuperClass): pass
        # class Test(SomeClass): pass
        #
        # Inferring SomeClass from the Test's bases will give
        # us both SomeClass and SomeSuperClass, but we are interested
        # only in SomeClass.

        if context is None:
            context = InferenceContext()
        if not self.bases and self.qname() != "builtins.object":
            yield builtin_lookup("object")[1][0]
            return

        for stmt in self.bases:
            try:
                # Find the first non-None inferred base value
                baseobj = next(
                    b
                    for b in stmt.infer(context=context.clone())
                    if not (isinstance(b, Const) and b.value is None)
                )
            except (InferenceError, StopIteration):
                continue
            if isinstance(baseobj, bases.Instance):
                baseobj = baseobj._proxied
            if not isinstance(baseobj, ClassDef):
                continue
            if not baseobj.hide:
                yield baseobj
            else:
                yield from baseobj.bases

    def _compute_mro(self, context: InferenceContext | None = None):
        inferred_bases = list(self._inferred_bases(context=context))
        bases_mro = []
        for base in inferred_bases:
            if base is self:
                continue

            try:
                mro = base._compute_mro(context=context)
                bases_mro.append(mro)
            except NotImplementedError:
                # Some classes have in their ancestors both newstyle and
                # old style classes. For these we can't retrieve the .mro,
                # although in Python it's possible, since the class we are
                # currently working is in fact new style.
                # So, we fallback to ancestors here.
                ancestors = list(base.ancestors(context=context))
                bases_mro.append(ancestors)

        unmerged_mro = [[self]] + bases_mro + [inferred_bases]
        unmerged_mro = list(clean_duplicates_mro(unmerged_mro, self, context))
        clean_typing_generic_mro(unmerged_mro)
        return _c3_merge(unmerged_mro, self, context)

    def mro(self, context: InferenceContext | None = None) -> list[ClassDef]:
        """Get the method resolution order, using C3 linearization.

        :returns: The list of ancestors, sorted by the mro.
        :rtype: list(NodeNG)
        :raises DuplicateBasesError: Duplicate bases in the same class base
        :raises InconsistentMroError: A class' MRO is inconsistent
        """
        return self._compute_mro(context=context)

    def bool_value(self, context: InferenceContext | None = None) -> Literal[True]:
        """Determine the boolean value of this node.

        :returns: The boolean value of this node.
            For a :class:`ClassDef` this is always ``True``.
        """
        return True

    def get_children(self):
        if self.decorators is not None:
            yield self.decorators

        yield from self.bases
        if self.keywords is not None:
            yield from self.keywords
        yield from self.body

    @decorators_mod.cached
    def _get_assign_nodes(self):
        children_assign_nodes = (
            child_node._get_assign_nodes() for child_node in self.body
        )
        return list(itertools.chain.from_iterable(children_assign_nodes))

    def frame(self: _T, *, future: Literal[None, True] = None) -> _T:
        """The node's frame node.

        A frame node is a :class:`Module`, :class:`FunctionDef`,
        :class:`ClassDef` or :class:`Lambda`.

        :returns: The node itself.
        """
        return self

```
### 10 - astroid/raw_building.py:

```python
# Licensed under the LGPL: https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html
# For details: https://github.com/PyCQA/astroid/blob/main/LICENSE
# Copyright (c) https://github.com/PyCQA/astroid/blob/main/CONTRIBUTORS.txt

"""this module contains a set of functions to create astroid trees from scratch
(build_* functions) or from living object (object_build_* functions)
"""

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


def _add_dunder_class(func, member) -> None:
    """Add a __class__ member to the given func node, if we can determine it."""
    python_cls = member.__class__
    cls_name = getattr(python_cls, "__name__", None)
    if not cls_name:
        return
    cls_bases = [ancestor.__name__ for ancestor in python_cls.__bases__]
    ast_klass = build_class(cls_name, cls_bases, python_cls.__doc__)
    func.instance_attrs["__class__"] = [ast_klass]


def attach_dummy_node(node, name: str, runtime_object=_EMPTY_OBJECT_MARKER) -> None:
    """create a dummy node and register it in the locals of the given
    node with the specified name
    """
    enode = nodes.EmptyNode()
    enode.object = runtime_object
    _attach_local_node(node, enode, name)


def attach_const_node(node, name: str, value) -> None:
    """create a Const node and register it in the locals of the given
    node with the specified name
    """
    if name not in node.special_attributes:
        _attach_local_node(node, nodes.const_factory(value), name)


def attach_import_node(node, modname: str, membername: str) -> None:
    """create a ImportFrom node and register it in the locals of the given
    node with the specified name
    """
    from_node = nodes.ImportFrom(modname, [(membername, None)])
    _attach_local_node(node, from_node, membername)


def build_module(name: str, doc: str | None = None) -> nodes.Module:
    """create and initialize an astroid Module node"""
    node = nodes.Module(name, pure_python=False, package=False)
    node.postinit(
        body=[],
        doc_node=nodes.Const(value=doc) if doc else None,
    )
    return node


def build_class(
    name: str, basenames: Iterable[str] = (), doc: str | None = None
) -> nodes.ClassDef:
    """Create and initialize an astroid ClassDef node."""
    node = nodes.ClassDef(name)
    node.postinit(
        bases=[nodes.Name(name=base, parent=node) for base in basenames],
        body=[],
        decorators=None,
        doc_node=nodes.Const(value=doc) if doc else None,
    )
    return node


def build_function(
    name: str,
    args: list[str] | None = None,
    posonlyargs: list[str] | None = None,
    defaults: list[Any] | None = None,
    doc: str | None = None,
    kwonlyargs: list[str] | None = None,
) -> nodes.FunctionDef:
    """create and initialize an astroid FunctionDef node"""
    # first argument is now a list of decorators
    func = nodes.FunctionDef(name)
    argsnode = nodes.Arguments(parent=func)

    # If args is None we don't have any information about the signature
    # (in contrast to when there are no arguments and args == []). We pass
    # this to the builder to indicate this.
    if args is not None:
        arguments = [nodes.AssignName(name=arg, parent=argsnode) for arg in args]
    else:
        arguments = None

    default_nodes: list[nodes.NodeNG] | None = []
    if defaults is not None:
        for default in defaults:
            default_node = nodes.const_factory(default)
            default_node.parent = argsnode
            default_nodes.append(default_node)
    else:
        default_nodes = None

    argsnode.postinit(
        args=arguments,
        defaults=default_nodes,
        kwonlyargs=[
            nodes.AssignName(name=arg, parent=argsnode) for arg in kwonlyargs or ()
        ],
        kw_defaults=[],
        annotations=[],
        posonlyargs=[
            nodes.AssignName(name=arg, parent=argsnode) for arg in posonlyargs or ()
        ],
    )
    func.postinit(
        args=argsnode,
        body=[],
        doc_node=nodes.Const(value=doc) if doc else None,
    )
    if args:
        register_arguments(func)
    return func


def build_from_import(fromname: str, names: list[str]) -> nodes.ImportFrom:
    """create and initialize an astroid ImportFrom import statement"""
    return nodes.ImportFrom(fromname, [(name, None) for name in names])


def register_arguments(func: nodes.FunctionDef, args: list | None = None) -> None:
    """add given arguments to local

    args is a list that may contains nested lists
    (i.e. def func(a, (b, c, d)): ...)
    """
    # If no args are passed in, get the args from the function.
    if args is None:
        if func.args.vararg:
            func.set_local(func.args.vararg, func.args)
        if func.args.kwarg:
            func.set_local(func.args.kwarg, func.args)
        args = func.args.args
        # If the function has no args, there is nothing left to do.
        if args is None:
            return
    for arg in args:
        if isinstance(arg, nodes.AssignName):
            func.set_local(arg.name, arg)
        else:
            register_arguments(func, arg.elts)


def object_build_class(
    node: nodes.Module | nodes.ClassDef, member: type, localname: str
) -> nodes.ClassDef:
    """create astroid for a living class object"""
    basenames = [base.__name__ for base in member.__bases__]
    return _base_class_object_build(node, member, basenames, localname=localname)


def _get_args_info_from_callable(
    member: _FunctionTypes,
) -> tuple[list[str], list[str], list[Any], list[str]]:
    """Returns args, posonlyargs, defaults, kwonlyargs.

    :note: currently ignores the return annotation.
    """
    signature = inspect.signature(member)
    args: list[str] = []
    defaults: list[Any] = []
    posonlyargs: list[str] = []
    kwonlyargs: list[str] = []

    for param_name, param in signature.parameters.items():
        if param.kind == inspect.Parameter.POSITIONAL_ONLY:
            posonlyargs.append(param_name)
        elif param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
            args.append(param_name)
        elif param.kind == inspect.Parameter.VAR_POSITIONAL:
            args.append(param_name)
        elif param.kind == inspect.Parameter.VAR_KEYWORD:
            args.append(param_name)
        elif param.kind == inspect.Parameter.KEYWORD_ONLY:
            kwonlyargs.append(param_name)
        if param.default is not inspect._empty:
            defaults.append(param.default)

    return args, posonlyargs, defaults, kwonlyargs


def object_build_function(
    node: nodes.Module | nodes.ClassDef, member: _FunctionTypes, localname: str
) -> None:
    """create astroid for a living function object"""
    args, posonlyargs, defaults, kwonlyargs = _get_args_info_from_callable(member)

    func = build_function(
        getattr(member, "__name__", None) or localname,
        args,
        posonlyargs,
        defaults,
        member.__doc__,
        kwonlyargs=kwonlyargs,
    )

    node.add_local_node(func, localname)


def object_build_datadescriptor(
    node: nodes.Module | nodes.ClassDef, member: type, name: str
) -> nodes.ClassDef:
    """create astroid for a living data descriptor object"""
    return _base_class_object_build(node, member, [], name)


def object_build_methoddescriptor(
    node: nodes.Module | nodes.ClassDef,
    member: _FunctionTypes,
    localname: str,
) -> None:
    """create astroid for a living method descriptor object"""
    # FIXME get arguments ?
    func = build_function(
        getattr(member, "__name__", None) or localname, doc=member.__doc__
    )
    node.add_local_node(func, localname)
    _add_dunder_class(func, member)


def _base_class_object_build(
    node: nodes.Module | nodes.ClassDef,
    member: type,
    basenames: list[str],
    name: str | None = None,
    localname: str | None = None,
) -> nodes.ClassDef:
    """create astroid for a living class object, with a given set of base names
    (e.g. ancestors)
    """
    class_name = name or getattr(member, "__name__", None) or localname
    assert isinstance(class_name, str)
    klass = build_class(
        class_name,
        basenames,
        member.__doc__,
    )
    klass._newstyle = isinstance(member, type)
    node.add_local_node(klass, localname)
    try:
        # limit the instantiation trick since it's too dangerous
        # (such as infinite test execution...)
        # this at least resolves common case such as Exception.args,
        # OSError.errno
        if issubclass(member, Exception):
            instdict = member().__dict__
        else:
            raise TypeError
    except TypeError:
        pass
    else:
        for item_name, obj in instdict.items():
            valnode = nodes.EmptyNode()
            valnode.object = obj
            valnode.parent = klass
            valnode.lineno = 1
            klass.instance_attrs[item_name] = [valnode]
    return klass


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


class InspectBuilder:
    """class for building nodes from living object

    this is actually a really minimal representation, including only Module,
    FunctionDef and ClassDef nodes and some others as guessed.
    """

    def __init__(self, manager_instance: AstroidManager | None = None) -> None:
        self._manager = manager_instance or AstroidManager()
        self._done: dict[types.ModuleType | type, nodes.Module | nodes.ClassDef] = {}
        self._module: types.ModuleType

    def inspect_build(
        self,
        module: types.ModuleType,
        modname: str | None = None,
        path: str | None = None,
    ) -> nodes.Module:
        """build astroid from a living module (i.e. using inspect)
        this is used when there is no python source code available (either
        because it's a built-in module or because the .py is not available)
        """
        self._module = module
        if modname is None:
            modname = module.__name__
        try:
            node = build_module(modname, module.__doc__)
        except AttributeError:
            # in jython, java modules have no __doc__ (see #109562)
            node = build_module(modname)
        if path is None:
            node.path = node.file = path
        else:
            node.path = [os.path.abspath(path)]
            node.file = node.path[0]
        node.name = modname
        self._manager.cache_module(node)
        node.package = hasattr(module, "__path__")
        self._done = {}
        self.object_build(node, module)
        return node

    def object_build(
        self, node: nodes.Module | nodes.ClassDef, obj: types.ModuleType | type
    ) -> None:
        """recursive method which create a partial ast from real objects
        (only function, class, and method are handled)
        """
        if obj in self._done:
            return None
        self._done[obj] = node
        for name in dir(obj):
            # inspect.ismethod() and inspect.isbuiltin() in PyPy return
            # the opposite of what they do in CPython for __class_getitem__.
            pypy__class_getitem__ = IS_PYPY and name == "__class_getitem__"
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    member = getattr(obj, name)
            except AttributeError:
                # damned ExtensionClass.Base, I know you're there !
                attach_dummy_node(node, name)
                continue
            if inspect.ismethod(member) and not pypy__class_getitem__:
                member = member.__func__
            if inspect.isfunction(member):
                _build_from_function(node, name, member, self._module)
            elif inspect.isbuiltin(member) or pypy__class_getitem__:
                if self.imported_member(node, member, name):
                    continue
                object_build_methoddescriptor(node, member, name)
            elif inspect.isclass(member):
                if self.imported_member(node, member, name):
                    continue
                if member in self._done:
                    class_node = self._done[member]
                    assert isinstance(class_node, nodes.ClassDef)
                    if class_node not in node.locals.get(name, ()):
                        node.add_local_node(class_node, name)
                else:
                    class_node = object_build_class(node, member, name)
                    # recursion
                    self.object_build(class_node, member)
                if name == "__class__" and class_node.parent is None:
                    class_node.parent = self._done[self._module]
            elif inspect.ismethoddescriptor(member):
                object_build_methoddescriptor(node, member, name)
            elif inspect.isdatadescriptor(member):
                object_build_datadescriptor(node, member, name)
            elif isinstance(member, _CONSTANTS):
                attach_const_node(node, name, member)
            elif inspect.isroutine(member):
                # This should be called for Jython, where some builtin
                # methods aren't caught by isbuiltin branch.
                _build_from_function(node, name, member, self._module)
            elif _safe_has_attribute(member, "__all__"):
                module = build_module(name)
                _attach_local_node(node, module, name)
                # recursion
                self.object_build(module, member)
            else:
                # create an empty node so that the name is actually defined
                attach_dummy_node(node, name, member)
        return None

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


def _astroid_bootstrapping() -> None:
    """astroid bootstrapping the builtins module"""
    # this boot strapping is necessary since we need the Const nodes to
    # inspect_build builtins, and then we can proxy Const
    builder = InspectBuilder()
    astroid_builtin = builder.inspect_build(builtins)

    for cls, node_cls in node_classes.CONST_CLS.items():
        if cls is TYPE_NONE:
            proxy = build_class("NoneType")
            proxy.parent = astroid_builtin
        elif cls is TYPE_NOTIMPLEMENTED:
            proxy = build_class("NotImplementedType")
            proxy.parent = astroid_builtin
        elif cls is TYPE_ELLIPSIS:
            proxy = build_class("Ellipsis")
            proxy.parent = astroid_builtin
        else:
            proxy = astroid_builtin.getattr(cls.__name__)[0]
            assert isinstance(proxy, nodes.ClassDef)
        if cls in (dict, list, set, tuple):
            node_cls._proxied = proxy
        else:
            _CONST_PROXY[cls] = proxy

    # Set the builtin module as parent for some builtins.
    nodes.Const._proxied = property(_set_proxied)

    _GeneratorType = nodes.ClassDef(types.GeneratorType.__name__)
    _GeneratorType.parent = astroid_builtin
    generator_doc_node = (
        nodes.Const(value=types.GeneratorType.__doc__)
        if types.GeneratorType.__doc__
        else None
    )
    _GeneratorType.postinit(
        bases=[],
        body=[],
        decorators=None,
        doc_node=generator_doc_node,
    )
    bases.Generator._proxied = _GeneratorType
    builder.object_build(bases.Generator._proxied, types.GeneratorType)

    if hasattr(types, "AsyncGeneratorType"):
        _AsyncGeneratorType = nodes.ClassDef(types.AsyncGeneratorType.__name__)
        _AsyncGeneratorType.parent = astroid_builtin
        async_generator_doc_node = (
            nodes.Const(value=types.AsyncGeneratorType.__doc__)
            if types.AsyncGeneratorType.__doc__
            else None
        )
        _AsyncGeneratorType.postinit(
            bases=[],
            body=[],
            decorators=None,
            doc_node=async_generator_doc_node,
        )
        bases.AsyncGenerator._proxied = _AsyncGeneratorType
        builder.object_build(bases.AsyncGenerator._proxied, types.AsyncGeneratorType)
    builtin_types = (
        types.GetSetDescriptorType,
        types.GeneratorType,
        types.MemberDescriptorType,
        TYPE_NONE,
        TYPE_NOTIMPLEMENTED,
        types.FunctionType,
        types.MethodType,
        types.BuiltinFunctionType,
        types.ModuleType,
        types.TracebackType,
    )
    for _type in builtin_types:
        if _type.__name__ not in astroid_builtin:
            klass = nodes.ClassDef(_type.__name__)
            klass.parent = astroid_builtin
            klass.postinit(
                bases=[],
                body=[],
                decorators=None,
                doc_node=nodes.Const(value=_type.__doc__) if _type.__doc__ else None,
            )
            builder.object_build(klass, _type)
            astroid_builtin[_type.__name__] = klass


_astroid_bootstrapping()

```
