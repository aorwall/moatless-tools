# pylint-dev__astroid-1866

| **pylint-dev/astroid** | `6cf238d089cf4b6753c94cfc089b4a47487711e5` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 17428 |
| **Any found context length** | 17428 |
| **Avg pos** | 62.0 |
| **Min pos** | 62 |
| **Max pos** | 62 |
| **Top file pos** | 3 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/astroid/brain/brain_builtin_inference.py b/astroid/brain/brain_builtin_inference.py
--- a/astroid/brain/brain_builtin_inference.py
+++ b/astroid/brain/brain_builtin_inference.py
@@ -954,8 +954,10 @@ def _infer_str_format_call(
 
     try:
         formatted_string = format_template.format(*pos_values, **keyword_values)
-    except (IndexError, KeyError):
-        # If there is an IndexError there are too few arguments to interpolate
+    except (IndexError, KeyError, TypeError, ValueError):
+        # IndexError: there are too few arguments to interpolate
+        # TypeError: Unsupported format string
+        # ValueError: Unknown format code
         return iter([util.Uninferable])
 
     return iter([nodes.const_factory(formatted_string)])

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| astroid/brain/brain_builtin_inference.py | 957 | 959 | 62 | 3 | 17428


## Problem Statement

```
"TypeError: unsupported format string passed to NoneType.__format__" while running type inference in version 2.12.x
### Steps to reproduce

I have no concise reproducer. Exception happens every time I run pylint on some internal code, with astroid 2.12.10 and 2.12.12 (debian bookworm). It does _not_ happen with earlier versions of astroid (not with version 2.9). The pylinted code itself is "valid", it runs in production here.

### Current behavior

When running pylint on some code, I get this exception:
\`\`\`
Traceback (most recent call last):
  File "/usr/lib/python3/dist-packages/pylint/utils/ast_walker.py", line 90, in walk
    callback(astroid)
  File "/usr/lib/python3/dist-packages/pylint/checkers/classes/special_methods_checker.py", line 183, in visit_functiondef
    inferred = _safe_infer_call_result(node, node)
  File "/usr/lib/python3/dist-packages/pylint/checkers/classes/special_methods_checker.py", line 42, in _safe_infer_call_result
    value = next(inferit)
  File "/usr/lib/python3/dist-packages/astroid/nodes/scoped_nodes/scoped_nodes.py", line 1749, in infer_call_result
    yield from returnnode.value.infer(context)
  File "/usr/lib/python3/dist-packages/astroid/nodes/node_ng.py", line 159, in infer
    results = list(self._explicit_inference(self, context, **kwargs))
  File "/usr/lib/python3/dist-packages/astroid/inference_tip.py", line 45, in _inference_tip_cached
    result = _cache[func, node] = list(func(*args, **kwargs))
  File "/usr/lib/python3/dist-packages/astroid/brain/brain_builtin_inference.py", line 956, in _infer_str_format_call
    formatted_string = format_template.format(*pos_values, **keyword_values)
TypeError: unsupported format string passed to NoneType.__format__
\`\`\`

### Expected behavior

TypeError exception should not happen

### `python -c "from astroid import __pkginfo__; print(__pkginfo__.version)"` output

2.12.10,
2.12.12

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- |
| 1 | 1 astroid/exceptions.py | 229 | 276| 438 | 438 | 
| 2 | 1 astroid/exceptions.py | 7 | 47| 240 | 678 | 
| 3 | 1 astroid/exceptions.py | 360 | 374| 113 | 791 | 
| 4 | 2 astroid/__init__.py | 32 | 180| 791 | 1582 | 
| 5 | **3 astroid/brain/brain_builtin_inference.py** | 897 | 923| 201 | 1783 | 
| 6 | 4 astroid/_ast.py | 4 | 26| 131 | 1914 | 
| 7 | 5 astroid/nodes/node_classes.py | 6 | 79| 492 | 2406 | 
| 8 | 6 astroid/brain/brain_argparse.py | 9 | 6| 260 | 2666 | 
| 9 | 7 astroid/interpreter/objectmodel.py | 23 | 53| 178 | 2844 | 
| 10 | 8 astroid/bases.py | 7 | 70| 400 | 3244 | 
| 11 | 9 astroid/inference.py | 628 | 661| 327 | 3571 | 
| 12 | 10 astroid/modutils.py | 16 | 80| 651 | 4222 | 
| 13 | 11 astroid/helpers.py | 46 | 72| 253 | 4475 | 
| 14 | 11 astroid/__init__.py | 182 | 197| 152 | 4627 | 
| 15 | 12 astroid/nodes/node_ng.py | 4 | 55| 333 | 4960 | 
| 16 | 12 astroid/exceptions.py | 377 | 397| 140 | 5100 | 
| 17 | 12 astroid/exceptions.py | 343 | 340| 183 | 5283 | 
| 18 | 13 astroid/manager.py | 9 | 46| 260 | 5543 | 
| 19 | 14 astroid/inference_tip.py | 6 | 27| 138 | 5681 | 
| 20 | 15 astroid/nodes/scoped_nodes/scoped_nodes.py | 10 | 69| 412 | 6093 | 
| 21 | 15 astroid/inference_tip.py | 71 | 90| 175 | 6268 | 
| 22 | 15 astroid/inference.py | 1190 | 1228| 378 | 6646 | 
| 23 | **15 astroid/brain/brain_builtin_inference.py** | 963 | 1002| 326 | 6972 | 
| 24 | 16 astroid/brain/brain_typing.py | 6 | 111| 607 | 7579 | 
| 25 | 17 astroid/brain/brain_pytest.py | 10 | 7| 455 | 8034 | 
| 26 | **17 astroid/brain/brain_builtin_inference.py** | 6 | 106| 678 | 8712 | 
| 27 | 18 astroid/rebuilder.py | 8 | 52| 351 | 9063 | 
| 28 | 19 astroid/typing.py | 4 | 57| 328 | 9391 | 
| 29 | 20 astroid/objects.py | 13 | 40| 154 | 9545 | 
| 30 | 20 astroid/brain/brain_typing.py | 404 | 431| 208 | 9753 | 
| 31 | 21 astroid/node_classes.py | 6 | 98| 403 | 10156 | 
| 32 | 22 astroid/interpreter/_import/util.py | 4 | 12| 58 | 10214 | 
| 33 | 23 astroid/brain/brain_fstrings.py | 30 | 48| 194 | 10408 | 
| 34 | 23 astroid/exceptions.py | 279 | 301| 192 | 10600 | 
| 35 | 24 astroid/raw_building.py | 8 | 43| 206 | 10806 | 
| 36 | 24 astroid/inference.py | 373 | 379| 104 | 10910 | 
| 37 | 24 astroid/inference.py | 382 | 400| 152 | 11062 | 
| 38 | 24 astroid/inference.py | 1070 | 1095| 211 | 11273 | 
| 39 | 25 astroid/brain/brain_functools.py | 6 | 22| 141 | 11414 | 
| 40 | 25 astroid/inference.py | 1116 | 1116| 141 | 11555 | 
| 41 | 25 astroid/inference.py | 932 | 965| 281 | 11836 | 
| 42 | 26 astroid/brain/brain_gi.py | 11 | 53| 204 | 12040 | 
| 43 | 27 astroid/brain/brain_numpy_ndarray.py | 12 | 9| 174 | 12214 | 
| 44 | 27 astroid/inference.py | 524 | 524| 188 | 12402 | 
| 45 | 27 astroid/helpers.py | 8 | 43| 252 | 12654 | 
| 46 | 27 astroid/nodes/node_classes.py | 1005 | 1034| 244 | 12898 | 
| 47 | 28 astroid/util.py | 62 | 59| 262 | 13160 | 
| 48 | 28 astroid/brain/brain_gi.py | 209 | 249| 245 | 13405 | 
| 49 | 29 astroid/brain/brain_pathlib.py | 4 | 51| 285 | 13690 | 
| 50 | 29 astroid/inference.py | 7 | 85| 497 | 14187 | 
| 51 | 30 astroid/brain/brain_namedtuple_enum.py | 6 | 46| 243 | 14430 | 
| 52 | 30 astroid/brain/brain_typing.py | 383 | 380| 214 | 14644 | 
| 53 | 31 astroid/builder.py | 10 | 60| 359 | 15003 | 
| 54 | 32 astroid/const.py | 4 | 35| 215 | 15218 | 
| 55 | 32 astroid/inference.py | 110 | 128| 140 | 15358 | 
| 56 | 33 astroid/brain/brain_numpy_core_function_base.py | 6 | 29| 246 | 15604 | 
| 57 | 34 astroid/decorators.py | 135 | 132| 209 | 15813 | 
| 58 | 34 astroid/interpreter/objectmodel.py | 692 | 784| 621 | 16434 | 
| 59 | 34 astroid/helpers.py | 196 | 216| 181 | 16615 | 
| 60 | 34 astroid/util.py | 24 | 21| 258 | 16873 | 
| 61 | 34 astroid/exceptions.py | 141 | 138| 221 | 17094 | 
| **-> 62 <-** | **34 astroid/brain/brain_builtin_inference.py** | 926 | 960| 334 | 17428 | 


### Hint

```
Hi @crosser, thanks for the report.

> I have no concise reproducer. 

We might be able to help you distill one.

`pylint` produces a crash report, and shows the link in your terminal, like this:
\`\`\`shell
************* Module a
a.py:1:0: F0002: a.py: Fatal error while checking 'a.py'. Please open an issue in our bug tracker so we address this. There is a pre-filled template that you can use in '/Users/.../Library/Caches/pylint/pylint-crash-2022-10-29-08-48-25.txt'. (astroid-error)
\`\`\`
The offending file is at the top of the crash report. If the code is too long, or contains sensitive information, you can use the knowledge that the crash happened in `_infer_str_format_call` to look for calls to `.format()` on strings. You should be able to then just provide us those calls--and enough surrounding code to rebuild the objects you provided to `format()`. 

Doing this would be a tremendous help!
> `pylint` produces a crash report, and shows the link in your terminal, like this:

No, not really, it does not. I am attaching a (censored) stderr from running the test. The line in the source code that apparently triggers the problem is pretty innocuous:

\`\`\`
    @property
    def vnet_id(self):  # <---- this is the line 266 that is mentioned in the "Exception on node" message
        if ...:
\`\`\`
There is very similar property definition right before this one, that does not trigger the problem.

[pyerr.txt](https://github.com/PyCQA/astroid/files/9900190/pyerr.txt)

Pylint command was `python3 -m pylint --jobs=0 --rcfile=test/style/pylint.conf <project-dir>`

\`\`\`
$ pylint --version
pylint 2.15.5
astroid 2.12.12
Python 3.10.8 (main, Oct 24 2022, 10:07:16) [GCC 12.2.0]
\`\`\`

edit:
> enough surrounding code to rebuild the objects you provided to format().

_I_ did not provide any objects to `format()`, astroid did...
Thanks for providing the traceback.

> No, not really, it does not. I am attaching a (censored) stderr from running the test. 

I see now that it's because you're invoking pylint from a unittest, so your test is managing the output.

> The line in the source code that apparently triggers the problem is pretty innocuous:

The deeper failure is on the call in line 268, not the function def on line 266. Is there anything you can sanitize and tell us about line 268? Thanks again for providing the help.
> I see now that it's because you're invoking pylint from a unittest, so your test is managing the output.

When I run pylint by hand

\`\`\`
pylint --jobs=0 --rcfile=test/style/pylint.conf <module-name> | tee /tmp/pyerr.txt
\`\`\`
there is still no "Fatal error while checking ..." message in the output

> > The line in the source code that apparently triggers the problem is pretty innocuous:
> 
> The deeper failure is on the call in line 268, not the function def on line 266. Is there anything you can sanitize and tell us about line 268? Thanks again for providing the help.

Oh yes, there is a `something.format()` in that line! But the "something" is a literal string:
\`\`\`
    @property
    def vnet_id(self):
        if self.backend == "something":
            return "{:04x}{:04x}n{:d}".format(  # <---- this is line 268
                self.<some-attr>, self.<another-attr>, self.<third-attr>
            )
        if self.backend == "somethingelse":
            return "h{:08}n{:d}".format(self.<more-attr>, self.<and more>)
        return None
\`\`\`

Thanks, that was very helpful. Here is a reproducer:
\`\`\`python
x = "{:c}".format(None)
\`\`\`
```

## Patch

```diff
diff --git a/astroid/brain/brain_builtin_inference.py b/astroid/brain/brain_builtin_inference.py
--- a/astroid/brain/brain_builtin_inference.py
+++ b/astroid/brain/brain_builtin_inference.py
@@ -954,8 +954,10 @@ def _infer_str_format_call(
 
     try:
         formatted_string = format_template.format(*pos_values, **keyword_values)
-    except (IndexError, KeyError):
-        # If there is an IndexError there are too few arguments to interpolate
+    except (IndexError, KeyError, TypeError, ValueError):
+        # IndexError: there are too few arguments to interpolate
+        # TypeError: Unsupported format string
+        # ValueError: Unknown format code
         return iter([util.Uninferable])
 
     return iter([nodes.const_factory(formatted_string)])

```

## Test Patch

```diff
diff --git a/tests/unittest_brain_builtin.py b/tests/unittest_brain_builtin.py
--- a/tests/unittest_brain_builtin.py
+++ b/tests/unittest_brain_builtin.py
@@ -103,6 +103,12 @@ def test_string_format(self, format_string: str) -> None:
             """
             "My name is {fname}, I'm {age}".format(fsname = "Daniel", age = 12)
             """,
+            """
+            "My unicode character is {:c}".format(None)
+            """,
+            """
+            "My hex format is {:4x}".format('1')
+            """,
         ],
     )
     def test_string_format_uninferable(self, format_string: str) -> None:

```


## Code snippets

### 1 - astroid/exceptions.py:

Start line: 229, End line: 276

```python
class InferenceError(ResolveError):  # pylint: disable=too-many-instance-attributes
    """raised when we are unable to infer a node

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
```
### 2 - astroid/exceptions.py:

Start line: 7, End line: 47

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
### 3 - astroid/exceptions.py:

Start line: 360, End line: 374

```python
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
```
### 4 - astroid/__init__.py:

Start line: 32, End line: 180

```python
import functools
import tokenize
from importlib import import_module

# isort: off
# We have an isort: off on '__version__' because the packaging need to access
# the version before the dependencies are installed (in particular 'wrapt'
# that is imported in astroid.inference)
from astroid.__pkginfo__ import __version__, version
from astroid.nodes import node_classes, scoped_nodes

# isort: on

from astroid import inference, raw_building
from astroid.astroid_manager import MANAGER
from astroid.bases import BaseInstance, BoundMethod, Instance, UnboundMethod
from astroid.brain.helpers import register_module_extender
from astroid.builder import extract_node, parse
from astroid.const import BRAIN_MODULES_DIRECTORY, PY310_PLUS, Context, Del, Load, Store
from astroid.exceptions import (
    AstroidBuildingError,
    AstroidBuildingException,
    AstroidError,
    AstroidImportError,
    AstroidIndexError,
    AstroidSyntaxError,
    AstroidTypeError,
    AstroidValueError,
    AttributeInferenceError,
    BinaryOperationError,
    DuplicateBasesError,
    InconsistentMroError,
    InferenceError,
    InferenceOverwriteError,
    MroError,
    NameInferenceError,
    NoDefault,
    NotFoundError,
    OperationError,
    ParentMissingError,
    ResolveError,
    StatementMissing,
    SuperArgumentTypeError,
    SuperError,
    TooManyLevelsError,
    UnaryOperationError,
    UnresolvableName,
    UseInferenceDefault,
)
from astroid.inference_tip import _inference_tip_cached, inference_tip
from astroid.objects import ExceptionInstance

# isort: off
# It's impossible to import from astroid.nodes with a wildcard, because
# there is a cyclic import that prevent creating an __all__ in astroid/nodes
# and we need astroid/scoped_nodes and astroid/node_classes to work. So
# importing with a wildcard would clash with astroid/nodes/scoped_nodes
# and astroid/nodes/node_classes.
from astroid.nodes import (  # pylint: disable=redefined-builtin (Ellipsis)
    CONST_CLS,
    AnnAssign,
    Arguments,
    Assert,
    Assign,
    AssignAttr,
    AssignName,
    AsyncFor,
    AsyncFunctionDef,
    AsyncWith,
    Attribute,
    AugAssign,
    Await,
    BinOp,
    BoolOp,
    Break,
    Call,
    ClassDef,
    Compare,
    Comprehension,
    ComprehensionScope,
    Const,
    Continue,
    Decorators,
    DelAttr,
    Delete,
    DelName,
    Dict,
    DictComp,
    DictUnpack,
    Ellipsis,
    EmptyNode,
    EvaluatedObject,
    ExceptHandler,
    Expr,
    ExtSlice,
    For,
    FormattedValue,
    FunctionDef,
    GeneratorExp,
    Global,
    If,
    IfExp,
    Import,
    ImportFrom,
    Index,
    JoinedStr,
    Keyword,
    Lambda,
    List,
    ListComp,
    Match,
    MatchAs,
    MatchCase,
    MatchClass,
    MatchMapping,
    MatchOr,
    MatchSequence,
    MatchSingleton,
    MatchStar,
    MatchValue,
    Module,
    Name,
    NamedExpr,
    NodeNG,
    Nonlocal,
    Pass,
    Raise,
    Return,
    Set,
    SetComp,
    Slice,
    Starred,
    Subscript,
    TryExcept,
    TryFinally,
    Tuple,
    UnaryOp,
    Unknown,
    While,
    With,
    Yield,
    YieldFrom,
    are_exclusive,
    builtin_lookup,
    unpack_infer,
    function_to_method,
)

# isort: on
```
### 5 - astroid/brain/brain_builtin_inference.py:

Start line: 897, End line: 923

```python
def _infer_copy_method(
    node: nodes.Call, context: InferenceContext | None = None
) -> Iterator[nodes.NodeNG]:
    assert isinstance(node.func, nodes.Attribute)
    inferred_orig, inferred_copy = itertools.tee(node.func.expr.infer(context=context))
    if all(
        isinstance(
            inferred_node, (nodes.Dict, nodes.List, nodes.Set, objects.FrozenSet)
        )
        for inferred_node in inferred_orig
    ):
        return inferred_copy

    raise UseInferenceDefault()


def _is_str_format_call(node: nodes.Call) -> bool:
    """Catch calls to str.format()."""
    if not isinstance(node.func, nodes.Attribute) or not node.func.attrname == "format":
        return False

    if isinstance(node.func.expr, nodes.Name):
        value = helpers.safe_infer(node.func.expr)
    else:
        value = node.func.expr

    return isinstance(value, nodes.Const) and isinstance(value.value, str)
```
### 6 - astroid/_ast.py:

Start line: 4, End line: 26

```python
from __future__ import annotations

import ast
import sys
import types
from functools import partial
from typing import NamedTuple

from astroid.const import PY38_PLUS, Context

if sys.version_info >= (3, 8):
    # On Python 3.8, typed_ast was merged back into `ast`
    _ast_py3: types.ModuleType | None = ast
else:
    try:
        import typed_ast.ast3 as _ast_py3
    except ImportError:
        _ast_py3 = None


class FunctionType(NamedTuple):
    argtypes: list[ast.expr]
    returns: ast.expr
```
### 7 - astroid/nodes/node_classes.py:

Start line: 6, End line: 79

```python
from __future__ import annotations

import abc
import itertools
import sys
import typing
import warnings
from collections.abc import Generator, Iterable, Mapping
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Optional, TypeVar, Union

from astroid import decorators, util
from astroid.bases import Instance, _infer_stmts
from astroid.const import Context
from astroid.context import InferenceContext
from astroid.exceptions import (
    AstroidIndexError,
    AstroidTypeError,
    InferenceError,
    NoDefault,
    ParentMissingError,
)
from astroid.manager import AstroidManager
from astroid.nodes import _base_nodes
from astroid.nodes.const import OP_PRECEDENCE
from astroid.nodes.node_ng import NodeNG
from astroid.typing import (
    ConstFactoryResult,
    InferenceErrorInfo,
    InferenceResult,
    SuccessfulInferenceResult,
)

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

if TYPE_CHECKING:
    from astroid import nodes
    from astroid.nodes import LocalsDictNodeNG

if sys.version_info >= (3, 8):
    from functools import cached_property
else:
    from astroid.decorators import cachedproperty as cached_property


def _is_const(value):
    return isinstance(value, tuple(CONST_CLS))


_NodesT = TypeVar("_NodesT", bound=NodeNG)
_BadOpMessageT = TypeVar("_BadOpMessageT", bound=util.BadOperationMessage)

AssignedStmtsPossibleNode = Union["List", "Tuple", "AssignName", "AssignAttr", None]
AssignedStmtsCall = Callable[
    [
        _NodesT,
        AssignedStmtsPossibleNode,
        Optional[InferenceContext],
        Optional[typing.List[int]],
    ],
    Any,
]
InferBinaryOperation = Callable[
    [_NodesT, Optional[InferenceContext]],
    typing.Generator[Union[InferenceResult, _BadOpMessageT], None, None],
]
InferLHS = Callable[
    [_NodesT, Optional[InferenceContext]],
    typing.Generator[InferenceResult, None, Optional[InferenceErrorInfo]],
]
InferUnaryOp = Callable[[_NodesT, str], ConstFactoryResult]
```
### 8 - astroid/brain/brain_argparse.py:

Start line: 9, End line: 6

```python
from astroid import arguments, inference_tip, nodes
from astroid.exceptions import UseInferenceDefault
from astroid.manager import AstroidManager


def infer_namespace(node, context=None):
    callsite = arguments.CallSite.from_call(node, context=context)
    if not callsite.keyword_arguments:
        # Cannot make sense of it.
        raise UseInferenceDefault()

    class_node = nodes.ClassDef("Namespace")
    # Set parent manually until ClassDef constructor fixed:
    # https://github.com/PyCQA/astroid/issues/1490
    class_node.parent = node.parent
    for attr in set(callsite.keyword_arguments):
        fake_node = nodes.EmptyNode()
        fake_node.parent = class_node
        fake_node.attrname = attr
        class_node.instance_attrs[attr] = [fake_node]
    return iter((class_node.instantiate_class(),))


def _looks_like_namespace(node):
    func = node.func
    if isinstance(func, nodes.Attribute):
        return (
            func.attrname == "Namespace"
            and isinstance(func.expr, nodes.Name)
            and func.expr.name == "argparse"
        )
    return False


AstroidManager().register_transform(
    nodes.Call, inference_tip(infer_namespace), _looks_like_namespace
)
```
### 9 - astroid/interpreter/objectmodel.py:

Start line: 23, End line: 53

```python
from __future__ import annotations

import itertools
import os
import pprint
import sys
import types
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import astroid
from astroid import bases, nodes, util
from astroid.context import InferenceContext, copy_context
from astroid.exceptions import AttributeInferenceError, InferenceError, NoDefault
from astroid.manager import AstroidManager
from astroid.nodes import node_classes

objects = util.lazy_import("objects")
builder = util.lazy_import("builder")

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

if TYPE_CHECKING:
    from astroid import builder
    from astroid.objects import Property

IMPL_PREFIX = "attr_"
LEN_OF_IMPL_PREFIX = len(IMPL_PREFIX)
```
### 10 - astroid/bases.py:

Start line: 7, End line: 70

```python
from __future__ import annotations

import collections
import collections.abc
import sys
from collections.abc import Sequence
from typing import Any

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
from astroid.typing import InferenceErrorInfo, InferenceResult
from astroid.util import Uninferable, lazy_descriptor, lazy_import

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

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
### 23 - astroid/brain/brain_builtin_inference.py:

Start line: 963, End line: 1002

```python
# Builtins inference
register_builtin_transform(infer_bool, "bool")
register_builtin_transform(infer_super, "super")
register_builtin_transform(infer_callable, "callable")
register_builtin_transform(infer_property, "property")
register_builtin_transform(infer_getattr, "getattr")
register_builtin_transform(infer_hasattr, "hasattr")
register_builtin_transform(infer_tuple, "tuple")
register_builtin_transform(infer_set, "set")
register_builtin_transform(infer_list, "list")
register_builtin_transform(infer_dict, "dict")
register_builtin_transform(infer_frozenset, "frozenset")
register_builtin_transform(infer_type, "type")
register_builtin_transform(infer_slice, "slice")
register_builtin_transform(infer_isinstance, "isinstance")
register_builtin_transform(infer_issubclass, "issubclass")
register_builtin_transform(infer_len, "len")
register_builtin_transform(infer_str, "str")
register_builtin_transform(infer_int, "int")
register_builtin_transform(infer_dict_fromkeys, "dict.fromkeys")


# Infer object.__new__ calls
AstroidManager().register_transform(
    nodes.ClassDef,
    inference_tip(_infer_object__new__decorator),
    _infer_object__new__decorator_check,
)

AstroidManager().register_transform(
    nodes.Call,
    inference_tip(_infer_copy_method),
    lambda node: isinstance(node.func, nodes.Attribute)
    and node.func.attrname == "copy",
)

AstroidManager().register_transform(
    nodes.Call, inference_tip(_infer_str_format_call), _is_str_format_call
)
```
### 26 - astroid/brain/brain_builtin_inference.py:

Start line: 6, End line: 106

```python
from __future__ import annotations

import itertools
from collections.abc import Iterator
from functools import partial

from astroid import arguments, helpers, inference_tip, nodes, objects, util
from astroid.builder import AstroidBuilder
from astroid.context import InferenceContext
from astroid.exceptions import (
    AstroidTypeError,
    AttributeInferenceError,
    InferenceError,
    MroError,
    UseInferenceDefault,
)
from astroid.manager import AstroidManager
from astroid.nodes import scoped_nodes

OBJECT_DUNDER_NEW = "object.__new__"

STR_CLASS = """
class whatever(object):
    def join(self, iterable):
        return {rvalue}
    def replace(self, old, new, count=None):
        return {rvalue}
    def format(self, *args, **kwargs):
        return {rvalue}
    def encode(self, encoding='ascii', errors=None):
        return b''
    def decode(self, encoding='ascii', errors=None):
        return u''
    def capitalize(self):
        return {rvalue}
    def title(self):
        return {rvalue}
    def lower(self):
        return {rvalue}
    def upper(self):
        return {rvalue}
    def swapcase(self):
        return {rvalue}
    def index(self, sub, start=None, end=None):
        return 0
    def find(self, sub, start=None, end=None):
        return 0
    def count(self, sub, start=None, end=None):
        return 0
    def strip(self, chars=None):
        return {rvalue}
    def lstrip(self, chars=None):
        return {rvalue}
    def rstrip(self, chars=None):
        return {rvalue}
    def rjust(self, width, fillchar=None):
        return {rvalue}
    def center(self, width, fillchar=None):
        return {rvalue}
    def ljust(self, width, fillchar=None):
        return {rvalue}
"""


BYTES_CLASS = """
class whatever(object):
    def join(self, iterable):
        return {rvalue}
    def replace(self, old, new, count=None):
        return {rvalue}
    def decode(self, encoding='ascii', errors=None):
        return u''
    def capitalize(self):
        return {rvalue}
    def title(self):
        return {rvalue}
    def lower(self):
        return {rvalue}
    def upper(self):
        return {rvalue}
    def swapcase(self):
        return {rvalue}
    def index(self, sub, start=None, end=None):
        return 0
    def find(self, sub, start=None, end=None):
        return 0
    def count(self, sub, start=None, end=None):
        return 0
    def strip(self, chars=None):
        return {rvalue}
    def lstrip(self, chars=None):
        return {rvalue}
    def rstrip(self, chars=None):
        return {rvalue}
    def rjust(self, width, fillchar=None):
        return {rvalue}
    def center(self, width, fillchar=None):
        return {rvalue}
    def ljust(self, width, fillchar=None):
        return {rvalue}
"""
```
### 62 - astroid/brain/brain_builtin_inference.py:

Start line: 926, End line: 960

```python
def _infer_str_format_call(
    node: nodes.Call, context: InferenceContext | None = None
) -> Iterator[nodes.Const | type[util.Uninferable]]:
    """Return a Const node based on the template and passed arguments."""
    call = arguments.CallSite.from_call(node, context=context)
    if isinstance(node.func.expr, nodes.Name):
        value: nodes.Const = helpers.safe_infer(node.func.expr)
    else:
        value = node.func.expr

    format_template = value.value

    # Get the positional arguments passed
    inferred_positional = [
        helpers.safe_infer(i, context) for i in call.positional_arguments
    ]
    if not all(isinstance(i, nodes.Const) for i in inferred_positional):
        return iter([util.Uninferable])
    pos_values: list[str] = [i.value for i in inferred_positional]

    # Get the keyword arguments passed
    inferred_keyword = {
        k: helpers.safe_infer(v, context) for k, v in call.keyword_arguments.items()
    }
    if not all(isinstance(i, nodes.Const) for i in inferred_keyword.values()):
        return iter([util.Uninferable])
    keyword_values: dict[str, str] = {k: v.value for k, v in inferred_keyword.items()}

    try:
        formatted_string = format_template.format(*pos_values, **keyword_values)
    except (IndexError, KeyError):
        # If there is an IndexError there are too few arguments to interpolate
        return iter([util.Uninferable])

    return iter([nodes.const_factory(formatted_string)])
```
