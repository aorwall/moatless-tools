# pylint-dev__astroid-1196

| **pylint-dev/astroid** | `39c2a9805970ca57093d32bbaf0e6a63e05041d8` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 468 |
| **Any found context length** | 468 |
| **Avg pos** | 2.0 |
| **Min pos** | 2 |
| **Max pos** | 2 |
| **Top file pos** | 2 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/astroid/nodes/node_classes.py b/astroid/nodes/node_classes.py
--- a/astroid/nodes/node_classes.py
+++ b/astroid/nodes/node_classes.py
@@ -2346,24 +2346,33 @@ def itered(self):
         """
         return [key for (key, _) in self.items]
 
-    def getitem(self, index, context=None):
+    def getitem(
+        self, index: Const | Slice, context: InferenceContext | None = None
+    ) -> NodeNG:
         """Get an item from this node.
 
         :param index: The node to use as a subscript index.
-        :type index: Const or Slice
 
         :raises AstroidTypeError: When the given index cannot be used as a
             subscript index, or if this node is not subscriptable.
         :raises AstroidIndexError: If the given index does not exist in the
             dictionary.
         """
+        # pylint: disable-next=import-outside-toplevel; circular import
+        from astroid.helpers import safe_infer
+
         for key, value in self.items:
             # TODO(cpopa): no support for overriding yet, {1:2, **{1: 3}}.
             if isinstance(key, DictUnpack):
+                inferred_value = safe_infer(value, context)
+                if not isinstance(inferred_value, Dict):
+                    continue
+
                 try:
-                    return value.getitem(index, context)
+                    return inferred_value.getitem(index, context)
                 except (AstroidTypeError, AstroidIndexError):
                     continue
+
             for inferredkey in key.infer(context):
                 if inferredkey is util.Uninferable:
                     continue

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| astroid/nodes/node_classes.py | 2349 | 2364 | 2 | 2 | 468


## Problem Statement

```
getitem does not infer the actual unpacked value
When trying to call `Dict.getitem()` on a context where we have a dict unpacking of anything beside a real dict, astroid currently raises an `AttributeError: 'getitem'`, which has 2 problems:

- The object might be a reference against something constant, this pattern is usually seen when we have different sets of dicts that extend each other, and all of their values are inferrable. 
- We can have something that is uninferable, but in that case instead of an `AttributeError` I think it makes sense to raise the usual `AstroidIndexError` which is supposed to be already handled by the downstream.


Here is a short reproducer;

\`\`\`py
from astroid import parse


source = """
X = {
    'A': 'B'
}

Y = {
    **X
}

KEY = 'A'
"""

tree = parse(source)

first_dict = tree.body[0].value
second_dict = tree.body[1].value
key = tree.body[2].value

print(f'{first_dict.getitem(key).value = }')
print(f'{second_dict.getitem(key).value = }')


\`\`\`

The current output;

\`\`\`
 $ python t1.py                                                                                                 3ms
first_dict.getitem(key).value = 'B'
Traceback (most recent call last):
  File "/home/isidentical/projects/astroid/t1.py", line 23, in <module>
    print(f'{second_dict.getitem(key).value = }')
  File "/home/isidentical/projects/astroid/astroid/nodes/node_classes.py", line 2254, in getitem
    return value.getitem(index, context)
AttributeError: 'Name' object has no attribute 'getitem'
\`\`\`

Expeceted output;
\`\`\`
 $ python t1.py                                                                                                 4ms
first_dict.getitem(key).value = 'B'
second_dict.getitem(key).value = 'B'

\`\`\`


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- |
| 1 | 1 astroid/bases.py | 320 | 337| 190 | 190 | 
| **-> 2 <-** | **2 astroid/nodes/node_classes.py** | 2348 | 2381| 278 | 468 | 
| 3 | 3 astroid/inference.py | 145 | 163| 185 | 653 | 
| 4 | 4 astroid/brain/brain_typing.py | 6 | 111| 607 | 1260 | 
| 5 | 5 astroid/brain/brain_namedtuple_enum.py | 6 | 46| 236 | 1496 | 
| 6 | **5 astroid/nodes/node_classes.py** | 166 | 205| 307 | 1803 | 
| 7 | **5 astroid/nodes/node_classes.py** | 1941 | 1975| 270 | 2073 | 
| 8 | 6 astroid/__init__.py | 32 | 180| 791 | 2864 | 
| 9 | 7 astroid/brain/brain_builtin_inference.py | 338 | 378| 324 | 3188 | 
| 10 | **7 astroid/nodes/node_classes.py** | 208 | 231| 201 | 3389 | 
| 11 | 7 astroid/brain/brain_builtin_inference.py | 830 | 894| 547 | 3936 | 
| 12 | 7 astroid/brain/brain_builtin_inference.py | 309 | 335| 235 | 4171 | 
| 13 | 8 astroid/arguments.py | 83 | 125| 298 | 4469 | 
| 14 | 9 astroid/typing.py | 4 | 41| 198 | 4667 | 
| 15 | 10 astroid/nodes/scoped_nodes/scoped_nodes.py | 2720 | 2771| 459 | 5126 | 
| 16 | 11 astroid/brain/brain_collections.py | 68 | 123| 414 | 5540 | 
| 17 | 11 astroid/inference.py | 347 | 406| 418 | 5958 | 
| 18 | 12 astroid/exceptions.py | 211 | 265| 363 | 6321 | 
| 19 | 13 astroid/interpreter/objectmodel.py | 712 | 757| 332 | 6653 | 
| 20 | 13 astroid/brain/brain_builtin_inference.py | 436 | 457| 188 | 6841 | 
| 21 | 14 astroid/rebuilder.py | 1050 | 1068| 193 | 7034 | 
| 22 | 14 astroid/brain/brain_builtin_inference.py | 6 | 106| 678 | 7712 | 
| 23 | 14 astroid/exceptions.py | 7 | 45| 215 | 7927 | 
| 24 | 14 astroid/interpreter/objectmodel.py | 23 | 45| 128 | 8055 | 
| 25 | 14 astroid/nodes/scoped_nodes/scoped_nodes.py | 10 | 68| 393 | 8448 | 
| 26 | 15 astroid/protocols.py | 8 | 108| 765 | 9213 | 
| 27 | 15 astroid/rebuilder.py | 1070 | 1082| 136 | 9349 | 
| 28 | 16 astroid/_ast.py | 4 | 26| 131 | 9480 | 
| 29 | 17 astroid/context.py | 26 | 23| 698 | 10178 | 
| 30 | **17 astroid/nodes/node_classes.py** | 2309 | 2346| 240 | 10418 | 
| 31 | 17 astroid/bases.py | 8 | 59| 330 | 10748 | 
| 32 | 17 astroid/inference.py | 107 | 119| 122 | 10870 | 
| 33 | 18 astroid/brain/brain_functools.py | 6 | 22| 141 | 11011 | 
| 34 | 18 astroid/__init__.py | 182 | 197| 152 | 11163 | 
| 35 | 18 astroid/brain/brain_typing.py | 145 | 181| 393 | 11556 | 
| 36 | 18 astroid/rebuilder.py | 8 | 51| 336 | 11892 | 
| 37 | 18 astroid/brain/brain_namedtuple_enum.py | 504 | 538| 292 | 12184 | 
| 38 | 19 astroid/manager.py | 300 | 342| 346 | 12530 | 
| 39 | 20 astroid/brain/brain_gi.py | 11 | 53| 204 | 12734 | 
| 40 | 20 astroid/bases.py | 206 | 235| 262 | 12996 | 


## Patch

```diff
diff --git a/astroid/nodes/node_classes.py b/astroid/nodes/node_classes.py
--- a/astroid/nodes/node_classes.py
+++ b/astroid/nodes/node_classes.py
@@ -2346,24 +2346,33 @@ def itered(self):
         """
         return [key for (key, _) in self.items]
 
-    def getitem(self, index, context=None):
+    def getitem(
+        self, index: Const | Slice, context: InferenceContext | None = None
+    ) -> NodeNG:
         """Get an item from this node.
 
         :param index: The node to use as a subscript index.
-        :type index: Const or Slice
 
         :raises AstroidTypeError: When the given index cannot be used as a
             subscript index, or if this node is not subscriptable.
         :raises AstroidIndexError: If the given index does not exist in the
             dictionary.
         """
+        # pylint: disable-next=import-outside-toplevel; circular import
+        from astroid.helpers import safe_infer
+
         for key, value in self.items:
             # TODO(cpopa): no support for overriding yet, {1:2, **{1: 3}}.
             if isinstance(key, DictUnpack):
+                inferred_value = safe_infer(value, context)
+                if not isinstance(inferred_value, Dict):
+                    continue
+
                 try:
-                    return value.getitem(index, context)
+                    return inferred_value.getitem(index, context)
                 except (AstroidTypeError, AstroidIndexError):
                     continue
+
             for inferredkey in key.infer(context):
                 if inferredkey is util.Uninferable:
                     continue

```

## Test Patch

```diff
diff --git a/tests/unittest_python3.py b/tests/unittest_python3.py
--- a/tests/unittest_python3.py
+++ b/tests/unittest_python3.py
@@ -5,7 +5,9 @@
 import unittest
 from textwrap import dedent
 
-from astroid import nodes
+import pytest
+
+from astroid import exceptions, nodes
 from astroid.builder import AstroidBuilder, extract_node
 from astroid.test_utils import require_version
 
@@ -285,6 +287,33 @@ def test_unpacking_in_dict_getitem(self) -> None:
             self.assertIsInstance(value, nodes.Const)
             self.assertEqual(value.value, expected)
 
+    @staticmethod
+    def test_unpacking_in_dict_getitem_with_ref() -> None:
+        node = extract_node(
+            """
+        a = {1: 2}
+        {**a, 2: 3}  #@
+        """
+        )
+        assert isinstance(node, nodes.Dict)
+
+        for key, expected in ((1, 2), (2, 3)):
+            value = node.getitem(nodes.Const(key))
+            assert isinstance(value, nodes.Const)
+            assert value.value == expected
+
+    @staticmethod
+    def test_unpacking_in_dict_getitem_uninferable() -> None:
+        node = extract_node("{**a, 2: 3}")
+        assert isinstance(node, nodes.Dict)
+
+        with pytest.raises(exceptions.AstroidIndexError):
+            node.getitem(nodes.Const(1))
+
+        value = node.getitem(nodes.Const(2))
+        assert isinstance(value, nodes.Const)
+        assert value.value == 3
+
     def test_format_string(self) -> None:
         code = "f'{greetings} {person}'"
         node = extract_node(code)

```


## Code snippets

### 1 - astroid/bases.py:

Start line: 320, End line: 337

```python
class Instance(BaseInstance):

    def getitem(self, index, context=None):
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
```
### 2 - astroid/nodes/node_classes.py:

Start line: 2348, End line: 2381

```python
class Dict(NodeNG, Instance):

    def getitem(self, index, context=None):
        """Get an item from this node.

        :param index: The node to use as a subscript index.
        :type index: Const or Slice

        :raises AstroidTypeError: When the given index cannot be used as a
            subscript index, or if this node is not subscriptable.
        :raises AstroidIndexError: If the given index does not exist in the
            dictionary.
        """
        for key, value in self.items:
            # TODO(cpopa): no support for overriding yet, {1:2, **{1: 3}}.
            if isinstance(key, DictUnpack):
                try:
                    return value.getitem(index, context)
                except (AstroidTypeError, AstroidIndexError):
                    continue
            for inferredkey in key.infer(context):
                if inferredkey is util.Uninferable:
                    continue
                if isinstance(inferredkey, Const) and isinstance(index, Const):
                    if inferredkey.value == index.value:
                        return value

        raise AstroidIndexError(index)

    def bool_value(self, context=None):
        """Determine the boolean value of this node.

        :returns: The boolean value of this node.
        :rtype: bool
        """
        return bool(self.items)
```
### 3 - astroid/inference.py:

Start line: 145, End line: 163

```python
def _infer_map(node, context):
    """Infer all values based on Dict.items"""
    values = {}
    for name, value in node.items:
        if isinstance(name, nodes.DictUnpack):
            double_starred = helpers.safe_infer(value, context)
            if not double_starred:
                raise InferenceError
            if not isinstance(double_starred, nodes.Dict):
                raise InferenceError(node=node, context=context)
            unpack_items = _infer_map(double_starred, context)
            values = _update_with_replacement(values, unpack_items)
        else:
            key = helpers.safe_infer(name, context=context)
            value = helpers.safe_infer(value, context=context)
            if any(not elem for elem in (key, value)):
                raise InferenceError(node=node, context=context)
            values = _update_with_replacement(values, {key: value})
    return values
```
### 4 - astroid/brain/brain_typing.py:

Start line: 6, End line: 111

```python
from __future__ import annotations

import typing
from collections.abc import Iterator
from functools import partial

from astroid import context, extract_node, inference_tip
from astroid.builder import _extract_single_node
from astroid.const import PY38_PLUS, PY39_PLUS
from astroid.exceptions import (
    AttributeInferenceError,
    InferenceError,
    UseInferenceDefault,
)
from astroid.manager import AstroidManager
from astroid.nodes.node_classes import (
    Assign,
    AssignName,
    Attribute,
    Call,
    Const,
    JoinedStr,
    Name,
    NodeNG,
    Subscript,
    Tuple,
)
from astroid.nodes.scoped_nodes import ClassDef, FunctionDef
from astroid.util import Uninferable

TYPING_NAMEDTUPLE_BASENAMES = {"NamedTuple", "typing.NamedTuple"}
TYPING_TYPEVARS = {"TypeVar", "NewType"}
TYPING_TYPEVARS_QUALIFIED = {"typing.TypeVar", "typing.NewType"}
TYPING_TYPE_TEMPLATE = """
class Meta(type):
    def __getitem__(self, item):
        return self

    @property
    def __args__(self):
        return ()

class {0}(metaclass=Meta):
    pass
"""
TYPING_MEMBERS = set(getattr(typing, "__all__", []))

TYPING_ALIAS = frozenset(
    (
        "typing.Hashable",
        "typing.Awaitable",
        "typing.Coroutine",
        "typing.AsyncIterable",
        "typing.AsyncIterator",
        "typing.Iterable",
        "typing.Iterator",
        "typing.Reversible",
        "typing.Sized",
        "typing.Container",
        "typing.Collection",
        "typing.Callable",
        "typing.AbstractSet",
        "typing.MutableSet",
        "typing.Mapping",
        "typing.MutableMapping",
        "typing.Sequence",
        "typing.MutableSequence",
        "typing.ByteString",
        "typing.Tuple",
        "typing.List",
        "typing.Deque",
        "typing.Set",
        "typing.FrozenSet",
        "typing.MappingView",
        "typing.KeysView",
        "typing.ItemsView",
        "typing.ValuesView",
        "typing.ContextManager",
        "typing.AsyncContextManager",
        "typing.Dict",
        "typing.DefaultDict",
        "typing.OrderedDict",
        "typing.Counter",
        "typing.ChainMap",
        "typing.Generator",
        "typing.AsyncGenerator",
        "typing.Type",
        "typing.Pattern",
        "typing.Match",
    )
)

CLASS_GETITEM_TEMPLATE = """
@classmethod
def __class_getitem__(cls, item):
    return cls
"""


def looks_like_typing_typevar_or_newtype(node):
    func = node.func
    if isinstance(func, Attribute):
        return func.attrname in TYPING_TYPEVARS
    if isinstance(func, Name):
        return func.name in TYPING_TYPEVARS
    return False
```
### 5 - astroid/brain/brain_namedtuple_enum.py:

Start line: 6, End line: 46

```python
from __future__ import annotations

import functools
import keyword
from collections.abc import Iterator
from textwrap import dedent

import astroid
from astroid import arguments, inference_tip, nodes, util
from astroid.builder import AstroidBuilder, extract_node
from astroid.context import InferenceContext
from astroid.exceptions import (
    AstroidTypeError,
    AstroidValueError,
    InferenceError,
    MroError,
    UseInferenceDefault,
)
from astroid.manager import AstroidManager

TYPING_NAMEDTUPLE_BASENAMES = {"NamedTuple", "typing.NamedTuple"}
ENUM_BASE_NAMES = {
    "Enum",
    "IntEnum",
    "enum.Enum",
    "enum.IntEnum",
    "IntFlag",
    "enum.IntFlag",
}


def _infer_first(node, context):
    if node is util.Uninferable:
        raise UseInferenceDefault
    try:
        value = next(node.infer(context=context))
    except StopIteration as exc:
        raise InferenceError from exc
    if value is util.Uninferable:
        raise UseInferenceDefault()
    return value
```
### 6 - astroid/nodes/node_classes.py:

Start line: 166, End line: 205

```python
# getitem() helpers.

_SLICE_SENTINEL = object()


def _slice_value(index, context=None):
    """Get the value of the given slice index."""

    if isinstance(index, Const):
        if isinstance(index.value, (int, type(None))):
            return index.value
    elif index is None:
        return None
    else:
        # Try to infer what the index actually is.
        # Since we can't return all the possible values,
        # we'll stop at the first possible value.
        try:
            inferred = next(index.infer(context=context))
        except (InferenceError, StopIteration):
            pass
        else:
            if isinstance(inferred, Const):
                if isinstance(inferred.value, (int, type(None))):
                    return inferred.value

    # Use a sentinel, because None can be a valid
    # value that this function can return,
    # as it is the case for unspecified bounds.
    return _SLICE_SENTINEL


def _infer_slice(node, context=None):
    lower = _slice_value(node.lower, context)
    upper = _slice_value(node.upper, context)
    step = _slice_value(node.step, context)
    if all(elem is not _SLICE_SENTINEL for elem in (lower, upper, step)):
        return slice(lower, upper, step)

    raise AstroidTypeError(
        message="Could not infer slice used in subscript",
        node=node,
        index=node.parent,
        context=context,
    )
```
### 7 - astroid/nodes/node_classes.py:

Start line: 1941, End line: 1975

```python
class Const(mixins.NoChildrenMixin, NodeNG, Instance):

    def getitem(self, index, context=None):
        """Get an item from this node if subscriptable.

        :param index: The node to use as a subscript index.
        :type index: Const or Slice

        :raises AstroidTypeError: When the given index cannot be used as a
            subscript index, or if this node is not subscriptable.
        """
        if isinstance(index, Const):
            index_value = index.value
        elif isinstance(index, Slice):
            index_value = _infer_slice(index, context=context)

        else:
            raise AstroidTypeError(
                f"Could not use type {type(index)} as subscript index"
            )

        try:
            if isinstance(self.value, (str, bytes)):
                return Const(self.value[index_value])
        except IndexError as exc:
            raise AstroidIndexError(
                message="Index {index!r} out of range",
                node=self,
                index=index,
                context=context,
            ) from exc
        except TypeError as exc:
            raise AstroidTypeError(
                message="Type error {error!r}", node=self, index=index, context=context
            ) from exc

        raise AstroidTypeError(f"{self!r} (value={self.value})")
```
### 8 - astroid/__init__.py:

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
### 9 - astroid/brain/brain_builtin_inference.py:

Start line: 338, End line: 378

```python
def infer_dict(node, context=None):
    """Try to infer a dict call to a Dict node.

    The function treats the following cases:

        * dict()
        * dict(mapping)
        * dict(iterable)
        * dict(iterable, **kwargs)
        * dict(mapping, **kwargs)
        * dict(**kwargs)

    If a case can't be inferred, we'll fallback to default inference.
    """
    call = arguments.CallSite.from_call(node, context=context)
    if call.has_invalid_arguments() or call.has_invalid_keywords():
        raise UseInferenceDefault

    args = call.positional_arguments
    kwargs = list(call.keyword_arguments.items())

    if not args and not kwargs:
        # dict()
        return nodes.Dict()
    if kwargs and not args:
        # dict(a=1, b=2, c=4)
        items = [(nodes.Const(key), value) for key, value in kwargs]
    elif len(args) == 1 and kwargs:
        # dict(some_iterable, b=2, c=4)
        elts = _get_elts(args[0], context)
        keys = [(nodes.Const(key), value) for key, value in kwargs]
        items = elts + keys
    elif len(args) == 1:
        items = _get_elts(args[0], context)
    else:
        raise UseInferenceDefault()
    value = nodes.Dict(
        col_offset=node.col_offset, lineno=node.lineno, parent=node.parent
    )
    value.postinit(items)
    return value
```
### 10 - astroid/nodes/node_classes.py:

Start line: 208, End line: 231

```python
def _container_getitem(instance, elts, index, context=None):
    """Get a slice or an item, using the given *index*, for the given sequence."""
    try:
        if isinstance(index, Slice):
            index_slice = _infer_slice(index, context=context)
            new_cls = instance.__class__()
            new_cls.elts = elts[index_slice]
            new_cls.parent = instance.parent
            return new_cls
        if isinstance(index, Const):
            return elts[index.value]
    except IndexError as exc:
        raise AstroidIndexError(
            message="Index {index!s} out of range",
            node=instance,
            index=index,
            context=context,
        ) from exc
    except TypeError as exc:
        raise AstroidTypeError(
            message="Type error {error!r}", node=instance, index=index, context=context
        ) from exc

    raise AstroidTypeError(f"Could not use {index} as subscript index")
```
### 30 - astroid/nodes/node_classes.py:

Start line: 2309, End line: 2346

```python
class Dict(NodeNG, Instance):

    def pytype(self):
        """Get the name of the type that this node represents.

        :returns: The name of the type.
        :rtype: str
        """
        return "builtins.dict"

    def get_children(self):
        """Get the key and value nodes below this node.

        Children are returned in the order that they are defined in the source
        code, key first then the value.

        :returns: The children.
        :rtype: iterable(NodeNG)
        """
        for key, value in self.items:
            yield key
            yield value

    def last_child(self):
        """An optimized version of list(get_children())[-1]

        :returns: The last child, or None if no children exist.
        :rtype: NodeNG or None
        """
        if self.items:
            return self.items[-1][1]
        return None

    def itered(self):
        """An iterator over the keys this node contains.

        :returns: The keys of this node.
        :rtype: iterable(NodeNG)
        """
        return [key for (key, _) in self.items]
```
