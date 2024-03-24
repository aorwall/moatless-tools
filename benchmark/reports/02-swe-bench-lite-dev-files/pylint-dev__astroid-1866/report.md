# pylint-dev__astroid-1866

| **pylint-dev/astroid** | `6cf238d089cf4b6753c94cfc089b4a47487711e5` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 17853 |
| **Any found context length** | 17853 |
| **Avg pos** | 5.0 |
| **Min pos** | 5 |
| **Max pos** | 5 |
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
| astroid/brain/brain_builtin_inference.py | 957 | 959 | 5 | 3 | 17853


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
| 1 | 1 astroid/exceptions.py | 0 | 425| 3039 | 3039 | 
| 2 | 1 astroid/exceptions.py | 0 | 425| 3039 | 6078 | 
| 3 | 1 astroid/exceptions.py | 0 | 425| 3039 | 9117 | 
| 4 | 2 astroid/__init__.py | 0 | 198| 1227 | 10344 | 
| **-> 5 <-** | **3 astroid/brain/brain_builtin_inference.py** | 0 | 1003| 7509 | 17853 | 


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

```python
# Licensed under the LGPL: https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html
# For details: https://github.com/PyCQA/astroid/blob/main/LICENSE
# Copyright (c) https://github.com/PyCQA/astroid/blob/main/CONTRIBUTORS.txt

"""this module contains exceptions used in the astroid library
"""

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
    """base exception class for all astroid related exceptions

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
    """exception class when we are unable to build an astroid representation

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
    """raised by function's `default_value` method when an argument has
    no default value

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
    """exception to be raised in custom inference function to indicate that it
    should go back to the default behaviour
    """


class _NonDeducibleTypeHierarchy(Exception):
    """Raised when is_subtype / is_supertype can't deduce the relation between two types."""


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
    """Raised when an inference tip is overwritten

    Currently only used for debugging.
    """


class ParentMissingError(AstroidError):
    """Raised when a node which is expected to have a parent attribute is missing one

    Standard attributes:
        target: The node for which the parent lookup failed.
    """

    def __init__(self, target: nodes.NodeNG) -> None:
        self.target = target
        super().__init__(message=f"Parent not found on {target!r}.")


class StatementMissing(ParentMissingError):
    """Raised when a call to node.statement() does not return a node. This is because
    a node in the chain does not have a parent attribute and therefore does not
    return a node for statement().

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
### 2 - astroid/exceptions.py:

```python
# Licensed under the LGPL: https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html
# For details: https://github.com/PyCQA/astroid/blob/main/LICENSE
# Copyright (c) https://github.com/PyCQA/astroid/blob/main/CONTRIBUTORS.txt

"""this module contains exceptions used in the astroid library
"""

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
    """base exception class for all astroid related exceptions

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
    """exception class when we are unable to build an astroid representation

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
    """raised by function's `default_value` method when an argument has
    no default value

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
    """exception to be raised in custom inference function to indicate that it
    should go back to the default behaviour
    """


class _NonDeducibleTypeHierarchy(Exception):
    """Raised when is_subtype / is_supertype can't deduce the relation between two types."""


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
    """Raised when an inference tip is overwritten

    Currently only used for debugging.
    """


class ParentMissingError(AstroidError):
    """Raised when a node which is expected to have a parent attribute is missing one

    Standard attributes:
        target: The node for which the parent lookup failed.
    """

    def __init__(self, target: nodes.NodeNG) -> None:
        self.target = target
        super().__init__(message=f"Parent not found on {target!r}.")


class StatementMissing(ParentMissingError):
    """Raised when a call to node.statement() does not return a node. This is because
    a node in the chain does not have a parent attribute and therefore does not
    return a node for statement().

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
### 3 - astroid/exceptions.py:

```python
# Licensed under the LGPL: https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html
# For details: https://github.com/PyCQA/astroid/blob/main/LICENSE
# Copyright (c) https://github.com/PyCQA/astroid/blob/main/CONTRIBUTORS.txt

"""this module contains exceptions used in the astroid library
"""

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
    """base exception class for all astroid related exceptions

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
    """exception class when we are unable to build an astroid representation

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
    """raised by function's `default_value` method when an argument has
    no default value

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
    """exception to be raised in custom inference function to indicate that it
    should go back to the default behaviour
    """


class _NonDeducibleTypeHierarchy(Exception):
    """Raised when is_subtype / is_supertype can't deduce the relation between two types."""


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
    """Raised when an inference tip is overwritten

    Currently only used for debugging.
    """


class ParentMissingError(AstroidError):
    """Raised when a node which is expected to have a parent attribute is missing one

    Standard attributes:
        target: The node for which the parent lookup failed.
    """

    def __init__(self, target: nodes.NodeNG) -> None:
        self.target = target
        super().__init__(message=f"Parent not found on {target!r}.")


class StatementMissing(ParentMissingError):
    """Raised when a call to node.statement() does not return a node. This is because
    a node in the chain does not have a parent attribute and therefore does not
    return a node for statement().

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
### 4 - astroid/__init__.py:

```python
# Licensed under the LGPL: https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html
# For details: https://github.com/PyCQA/astroid/blob/main/LICENSE
# Copyright (c) https://github.com/PyCQA/astroid/blob/main/CONTRIBUTORS.txt

"""Python Abstract Syntax Tree New Generation

The aim of this module is to provide a common base representation of
python source code for projects such as pychecker, pyreverse,
pylint... Well, actually the development of this library is essentially
governed by pylint's needs.

It mimics the class defined in the python's _ast module with some
additional methods and attributes. New nodes instances are not fully
compatible with python's _ast.

Instance attributes are added by a
builder object, which can either generate extended ast (let's call
them astroid ;) by visiting an existent ast tree or by inspecting living
object. Methods are added by monkey patching ast classes.

Main modules are:

* nodes and scoped_nodes for more information about methods and
  attributes added to different node classes

* the manager contains a high level object to get astroid trees from
  source files and living objects. It maintains a cache of previously
  constructed tree for quick access

* builder contains the class responsible to build astroid trees
"""

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

from astroid.util import Uninferable

# Performance hack for tokenize. See https://bugs.python.org/issue43014
# Adapted from https://github.com/PyCQA/pycodestyle/pull/993
if (
    not PY310_PLUS
    and callable(getattr(tokenize, "_compile", None))
    and getattr(tokenize._compile, "__wrapped__", None) is None  # type: ignore[attr-defined]
):
    tokenize._compile = functools.lru_cache()(tokenize._compile)  # type: ignore[attr-defined]

# load brain plugins
for module in BRAIN_MODULES_DIRECTORY.iterdir():
    if module.suffix == ".py":
        import_module(f"astroid.brain.{module.stem}")

```
### 5 - astroid/brain/brain_builtin_inference.py:

```python
# Licensed under the LGPL: https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html
# For details: https://github.com/PyCQA/astroid/blob/main/LICENSE
# Copyright (c) https://github.com/PyCQA/astroid/blob/main/CONTRIBUTORS.txt

"""Astroid hooks for various builtins."""

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


def _extend_string_class(class_node, code, rvalue):
    """function to extend builtin str/unicode class"""
    code = code.format(rvalue=rvalue)
    fake = AstroidBuilder(AstroidManager()).string_build(code)["whatever"]
    for method in fake.mymethods():
        method.parent = class_node
        method.lineno = None
        method.col_offset = None
        if "__class__" in method.locals:
            method.locals["__class__"] = [class_node]
        class_node.locals[method.name] = [method]
        method.parent = class_node


def _extend_builtins(class_transforms):
    builtin_ast = AstroidManager().builtins_module
    for class_name, transform in class_transforms.items():
        transform(builtin_ast[class_name])


_extend_builtins(
    {
        "bytes": partial(_extend_string_class, code=BYTES_CLASS, rvalue="b''"),
        "str": partial(_extend_string_class, code=STR_CLASS, rvalue="''"),
    }
)


def _builtin_filter_predicate(node, builtin_name):
    if (
        builtin_name == "type"
        and node.root().name == "re"
        and isinstance(node.func, nodes.Name)
        and node.func.name == "type"
        and isinstance(node.parent, nodes.Assign)
        and len(node.parent.targets) == 1
        and isinstance(node.parent.targets[0], nodes.AssignName)
        and node.parent.targets[0].name in {"Pattern", "Match"}
    ):
        # Handle re.Pattern and re.Match in brain_re
        # Match these patterns from stdlib/re.py
        # ```py
        # Pattern = type(...)
        # Match = type(...)
        # ```
        return False
    if isinstance(node.func, nodes.Name) and node.func.name == builtin_name:
        return True
    if isinstance(node.func, nodes.Attribute):
        return (
            node.func.attrname == "fromkeys"
            and isinstance(node.func.expr, nodes.Name)
            and node.func.expr.name == "dict"
        )
    return False


def register_builtin_transform(transform, builtin_name):
    """Register a new transform function for the given *builtin_name*.

    The transform function must accept two parameters, a node and
    an optional context.
    """

    def _transform_wrapper(node, context=None):
        result = transform(node, context=context)
        if result:
            if not result.parent:
                # Let the transformation function determine
                # the parent for its result. Otherwise,
                # we set it to be the node we transformed from.
                result.parent = node

            if result.lineno is None:
                result.lineno = node.lineno
            # Can be a 'Module' see https://github.com/PyCQA/pylint/issues/4671
            # We don't have a regression test on this one: tread carefully
            if hasattr(result, "col_offset") and result.col_offset is None:
                result.col_offset = node.col_offset
        return iter([result])

    AstroidManager().register_transform(
        nodes.Call,
        inference_tip(_transform_wrapper),
        partial(_builtin_filter_predicate, builtin_name=builtin_name),
    )


def _container_generic_inference(node, context, node_type, transform):
    args = node.args
    if not args:
        return node_type()
    if len(node.args) > 1:
        raise UseInferenceDefault()

    (arg,) = args
    transformed = transform(arg)
    if not transformed:
        try:
            inferred = next(arg.infer(context=context))
        except (InferenceError, StopIteration) as exc:
            raise UseInferenceDefault from exc
        if inferred is util.Uninferable:
            raise UseInferenceDefault
        transformed = transform(inferred)
    if not transformed or transformed is util.Uninferable:
        raise UseInferenceDefault
    return transformed


def _container_generic_transform(  # pylint: disable=inconsistent-return-statements
    arg, context, klass, iterables, build_elts
):
    if isinstance(arg, klass):
        return arg
    if isinstance(arg, iterables):
        if all(isinstance(elt, nodes.Const) for elt in arg.elts):
            elts = [elt.value for elt in arg.elts]
        else:
            # TODO: Does not handle deduplication for sets.
            elts = []
            for element in arg.elts:
                if not element:
                    continue
                inferred = helpers.safe_infer(element, context=context)
                if inferred:
                    evaluated_object = nodes.EvaluatedObject(
                        original=element, value=inferred
                    )
                    elts.append(evaluated_object)
    elif isinstance(arg, nodes.Dict):
        # Dicts need to have consts as strings already.
        if not all(isinstance(elt[0], nodes.Const) for elt in arg.items):
            raise UseInferenceDefault()
        elts = [item[0].value for item in arg.items]
    elif isinstance(arg, nodes.Const) and isinstance(arg.value, (str, bytes)):
        elts = arg.value
    else:
        return
    return klass.from_elements(elts=build_elts(elts))


def _infer_builtin_container(
    node, context, klass=None, iterables=None, build_elts=None
):
    transform_func = partial(
        _container_generic_transform,
        context=context,
        klass=klass,
        iterables=iterables,
        build_elts=build_elts,
    )

    return _container_generic_inference(node, context, klass, transform_func)


# pylint: disable=invalid-name
infer_tuple = partial(
    _infer_builtin_container,
    klass=nodes.Tuple,
    iterables=(
        nodes.List,
        nodes.Set,
        objects.FrozenSet,
        objects.DictItems,
        objects.DictKeys,
        objects.DictValues,
    ),
    build_elts=tuple,
)

infer_list = partial(
    _infer_builtin_container,
    klass=nodes.List,
    iterables=(
        nodes.Tuple,
        nodes.Set,
        objects.FrozenSet,
        objects.DictItems,
        objects.DictKeys,
        objects.DictValues,
    ),
    build_elts=list,
)

infer_set = partial(
    _infer_builtin_container,
    klass=nodes.Set,
    iterables=(nodes.List, nodes.Tuple, objects.FrozenSet, objects.DictKeys),
    build_elts=set,
)

infer_frozenset = partial(
    _infer_builtin_container,
    klass=objects.FrozenSet,
    iterables=(nodes.List, nodes.Tuple, nodes.Set, objects.FrozenSet, objects.DictKeys),
    build_elts=frozenset,
)


def _get_elts(arg, context):
    def is_iterable(n):
        return isinstance(n, (nodes.List, nodes.Tuple, nodes.Set))

    try:
        inferred = next(arg.infer(context))
    except (InferenceError, StopIteration) as exc:
        raise UseInferenceDefault from exc
    if isinstance(inferred, nodes.Dict):
        items = inferred.items
    elif is_iterable(inferred):
        items = []
        for elt in inferred.elts:
            # If an item is not a pair of two items,
            # then fallback to the default inference.
            # Also, take in consideration only hashable items,
            # tuples and consts. We are choosing Names as well.
            if not is_iterable(elt):
                raise UseInferenceDefault()
            if len(elt.elts) != 2:
                raise UseInferenceDefault()
            if not isinstance(elt.elts[0], (nodes.Tuple, nodes.Const, nodes.Name)):
                raise UseInferenceDefault()
            items.append(tuple(elt.elts))
    else:
        raise UseInferenceDefault()
    return items


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


def infer_super(node, context=None):
    """Understand super calls.

    There are some restrictions for what can be understood:

        * unbounded super (one argument form) is not understood.

        * if the super call is not inside a function (classmethod or method),
          then the default inference will be used.

        * if the super arguments can't be inferred, the default inference
          will be used.
    """
    if len(node.args) == 1:
        # Ignore unbounded super.
        raise UseInferenceDefault

    scope = node.scope()
    if not isinstance(scope, nodes.FunctionDef):
        # Ignore non-method uses of super.
        raise UseInferenceDefault
    if scope.type not in ("classmethod", "method"):
        # Not interested in staticmethods.
        raise UseInferenceDefault

    cls = scoped_nodes.get_wrapping_class(scope)
    if not node.args:
        mro_pointer = cls
        # In we are in a classmethod, the interpreter will fill
        # automatically the class as the second argument, not an instance.
        if scope.type == "classmethod":
            mro_type = cls
        else:
            mro_type = cls.instantiate_class()
    else:
        try:
            mro_pointer = next(node.args[0].infer(context=context))
        except (InferenceError, StopIteration) as exc:
            raise UseInferenceDefault from exc
        try:
            mro_type = next(node.args[1].infer(context=context))
        except (InferenceError, StopIteration) as exc:
            raise UseInferenceDefault from exc

    if mro_pointer is util.Uninferable or mro_type is util.Uninferable:
        # No way we could understand this.
        raise UseInferenceDefault

    super_obj = objects.Super(
        mro_pointer=mro_pointer, mro_type=mro_type, self_class=cls, scope=scope
    )
    super_obj.parent = node
    return super_obj


def _infer_getattr_args(node, context):
    if len(node.args) not in (2, 3):
        # Not a valid getattr call.
        raise UseInferenceDefault

    try:
        obj = next(node.args[0].infer(context=context))
        attr = next(node.args[1].infer(context=context))
    except (InferenceError, StopIteration) as exc:
        raise UseInferenceDefault from exc

    if obj is util.Uninferable or attr is util.Uninferable:
        # If one of the arguments is something we can't infer,
        # then also make the result of the getattr call something
        # which is unknown.
        return util.Uninferable, util.Uninferable

    is_string = isinstance(attr, nodes.Const) and isinstance(attr.value, str)
    if not is_string:
        raise UseInferenceDefault

    return obj, attr.value


def infer_getattr(node, context=None):
    """Understand getattr calls

    If one of the arguments is an Uninferable object, then the
    result will be an Uninferable object. Otherwise, the normal attribute
    lookup will be done.
    """
    obj, attr = _infer_getattr_args(node, context)
    if (
        obj is util.Uninferable
        or attr is util.Uninferable
        or not hasattr(obj, "igetattr")
    ):
        return util.Uninferable

    try:
        return next(obj.igetattr(attr, context=context))
    except (StopIteration, InferenceError, AttributeInferenceError):
        if len(node.args) == 3:
            # Try to infer the default and return it instead.
            try:
                return next(node.args[2].infer(context=context))
            except (StopIteration, InferenceError) as exc:
                raise UseInferenceDefault from exc

    raise UseInferenceDefault


def infer_hasattr(node, context=None):
    """Understand hasattr calls

    This always guarantees three possible outcomes for calling
    hasattr: Const(False) when we are sure that the object
    doesn't have the intended attribute, Const(True) when
    we know that the object has the attribute and Uninferable
    when we are unsure of the outcome of the function call.
    """
    try:
        obj, attr = _infer_getattr_args(node, context)
        if (
            obj is util.Uninferable
            or attr is util.Uninferable
            or not hasattr(obj, "getattr")
        ):
            return util.Uninferable
        obj.getattr(attr, context=context)
    except UseInferenceDefault:
        # Can't infer something from this function call.
        return util.Uninferable
    except AttributeInferenceError:
        # Doesn't have it.
        return nodes.Const(False)
    return nodes.Const(True)


def infer_callable(node, context=None):
    """Understand callable calls

    This follows Python's semantics, where an object
    is callable if it provides an attribute __call__,
    even though that attribute is something which can't be
    called.
    """
    if len(node.args) != 1:
        # Invalid callable call.
        raise UseInferenceDefault

    argument = node.args[0]
    try:
        inferred = next(argument.infer(context=context))
    except (InferenceError, StopIteration):
        return util.Uninferable
    if inferred is util.Uninferable:
        return util.Uninferable
    return nodes.Const(inferred.callable())


def infer_property(
    node: nodes.Call, context: InferenceContext | None = None
) -> objects.Property:
    """Understand `property` class

    This only infers the output of `property`
    call, not the arguments themselves.
    """
    if len(node.args) < 1:
        # Invalid property call.
        raise UseInferenceDefault

    getter = node.args[0]
    try:
        inferred = next(getter.infer(context=context))
    except (InferenceError, StopIteration) as exc:
        raise UseInferenceDefault from exc

    if not isinstance(inferred, (nodes.FunctionDef, nodes.Lambda)):
        raise UseInferenceDefault

    prop_func = objects.Property(
        function=inferred,
        name=inferred.name,
        lineno=node.lineno,
        parent=node,
        col_offset=node.col_offset,
    )
    prop_func.postinit(
        body=[],
        args=inferred.args,
        doc_node=getattr(inferred, "doc_node", None),
    )
    return prop_func


def infer_bool(node, context=None):
    """Understand bool calls."""
    if len(node.args) > 1:
        # Invalid bool call.
        raise UseInferenceDefault

    if not node.args:
        return nodes.Const(False)

    argument = node.args[0]
    try:
        inferred = next(argument.infer(context=context))
    except (InferenceError, StopIteration):
        return util.Uninferable
    if inferred is util.Uninferable:
        return util.Uninferable

    bool_value = inferred.bool_value(context=context)
    if bool_value is util.Uninferable:
        return util.Uninferable
    return nodes.Const(bool_value)


def infer_type(node, context=None):
    """Understand the one-argument form of *type*."""
    if len(node.args) != 1:
        raise UseInferenceDefault

    return helpers.object_type(node.args[0], context)


def infer_slice(node, context=None):
    """Understand `slice` calls."""
    args = node.args
    if not 0 < len(args) <= 3:
        raise UseInferenceDefault

    infer_func = partial(helpers.safe_infer, context=context)
    args = [infer_func(arg) for arg in args]
    for arg in args:
        if not arg or arg is util.Uninferable:
            raise UseInferenceDefault
        if not isinstance(arg, nodes.Const):
            raise UseInferenceDefault
        if not isinstance(arg.value, (type(None), int)):
            raise UseInferenceDefault

    if len(args) < 3:
        # Make sure we have 3 arguments.
        args.extend([None] * (3 - len(args)))

    slice_node = nodes.Slice(
        lineno=node.lineno, col_offset=node.col_offset, parent=node.parent
    )
    slice_node.postinit(*args)
    return slice_node


def _infer_object__new__decorator(node, context=None):
    # Instantiate class immediately
    # since that's what @object.__new__ does
    return iter((node.instantiate_class(),))


def _infer_object__new__decorator_check(node):
    """Predicate before inference_tip

    Check if the given ClassDef has an @object.__new__ decorator
    """
    if not node.decorators:
        return False

    for decorator in node.decorators.nodes:
        if isinstance(decorator, nodes.Attribute):
            if decorator.as_string() == OBJECT_DUNDER_NEW:
                return True
    return False


def infer_issubclass(callnode, context=None):
    """Infer issubclass() calls

    :param nodes.Call callnode: an `issubclass` call
    :param InferenceContext context: the context for the inference
    :rtype nodes.Const: Boolean Const value of the `issubclass` call
    :raises UseInferenceDefault: If the node cannot be inferred
    """
    call = arguments.CallSite.from_call(callnode, context=context)
    if call.keyword_arguments:
        # issubclass doesn't support keyword arguments
        raise UseInferenceDefault("TypeError: issubclass() takes no keyword arguments")
    if len(call.positional_arguments) != 2:
        raise UseInferenceDefault(
            f"Expected two arguments, got {len(call.positional_arguments)}"
        )
    # The left hand argument is the obj to be checked
    obj_node, class_or_tuple_node = call.positional_arguments

    try:
        obj_type = next(obj_node.infer(context=context))
    except (InferenceError, StopIteration) as exc:
        raise UseInferenceDefault from exc
    if not isinstance(obj_type, nodes.ClassDef):
        raise UseInferenceDefault("TypeError: arg 1 must be class")

    # The right hand argument is the class(es) that the given
    # object is to be checked against.
    try:
        class_container = _class_or_tuple_to_container(
            class_or_tuple_node, context=context
        )
    except InferenceError as exc:
        raise UseInferenceDefault from exc
    try:
        issubclass_bool = helpers.object_issubclass(obj_type, class_container, context)
    except AstroidTypeError as exc:
        raise UseInferenceDefault("TypeError: " + str(exc)) from exc
    except MroError as exc:
        raise UseInferenceDefault from exc
    return nodes.Const(issubclass_bool)


def infer_isinstance(callnode, context=None):
    """Infer isinstance calls

    :param nodes.Call callnode: an isinstance call
    :param InferenceContext context: context for call
        (currently unused but is a common interface for inference)
    :rtype nodes.Const: Boolean Const value of isinstance call

    :raises UseInferenceDefault: If the node cannot be inferred
    """
    call = arguments.CallSite.from_call(callnode, context=context)
    if call.keyword_arguments:
        # isinstance doesn't support keyword arguments
        raise UseInferenceDefault("TypeError: isinstance() takes no keyword arguments")
    if len(call.positional_arguments) != 2:
        raise UseInferenceDefault(
            f"Expected two arguments, got {len(call.positional_arguments)}"
        )
    # The left hand argument is the obj to be checked
    obj_node, class_or_tuple_node = call.positional_arguments
    # The right hand argument is the class(es) that the given
    # obj is to be check is an instance of
    try:
        class_container = _class_or_tuple_to_container(
            class_or_tuple_node, context=context
        )
    except InferenceError as exc:
        raise UseInferenceDefault from exc
    try:
        isinstance_bool = helpers.object_isinstance(obj_node, class_container, context)
    except AstroidTypeError as exc:
        raise UseInferenceDefault("TypeError: " + str(exc)) from exc
    except MroError as exc:
        raise UseInferenceDefault from exc
    if isinstance_bool is util.Uninferable:
        raise UseInferenceDefault
    return nodes.Const(isinstance_bool)


def _class_or_tuple_to_container(node, context=None):
    # Move inferences results into container
    # to simplify later logic
    # raises InferenceError if any of the inferences fall through
    try:
        node_infer = next(node.infer(context=context))
    except StopIteration as e:
        raise InferenceError(node=node, context=context) from e
    # arg2 MUST be a type or a TUPLE of types
    # for isinstance
    if isinstance(node_infer, nodes.Tuple):
        try:
            class_container = [
                next(node.infer(context=context)) for node in node_infer.elts
            ]
        except StopIteration as e:
            raise InferenceError(node=node, context=context) from e
        class_container = [
            klass_node for klass_node in class_container if klass_node is not None
        ]
    else:
        class_container = [node_infer]
    return class_container


def infer_len(node, context=None):
    """Infer length calls

    :param nodes.Call node: len call to infer
    :param context.InferenceContext: node context
    :rtype nodes.Const: a Const node with the inferred length, if possible
    """
    call = arguments.CallSite.from_call(node, context=context)
    if call.keyword_arguments:
        raise UseInferenceDefault("TypeError: len() must take no keyword arguments")
    if len(call.positional_arguments) != 1:
        raise UseInferenceDefault(
            "TypeError: len() must take exactly one argument "
            "({len}) given".format(len=len(call.positional_arguments))
        )
    [argument_node] = call.positional_arguments

    try:
        return nodes.Const(helpers.object_len(argument_node, context=context))
    except (AstroidTypeError, InferenceError) as exc:
        raise UseInferenceDefault(str(exc)) from exc


def infer_str(node, context=None):
    """Infer str() calls

    :param nodes.Call node: str() call to infer
    :param context.InferenceContext: node context
    :rtype nodes.Const: a Const containing an empty string
    """
    call = arguments.CallSite.from_call(node, context=context)
    if call.keyword_arguments:
        raise UseInferenceDefault("TypeError: str() must take no keyword arguments")
    try:
        return nodes.Const("")
    except (AstroidTypeError, InferenceError) as exc:
        raise UseInferenceDefault(str(exc)) from exc


def infer_int(node, context=None):
    """Infer int() calls

    :param nodes.Call node: int() call to infer
    :param context.InferenceContext: node context
    :rtype nodes.Const: a Const containing the integer value of the int() call
    """
    call = arguments.CallSite.from_call(node, context=context)
    if call.keyword_arguments:
        raise UseInferenceDefault("TypeError: int() must take no keyword arguments")

    if call.positional_arguments:
        try:
            first_value = next(call.positional_arguments[0].infer(context=context))
        except (InferenceError, StopIteration) as exc:
            raise UseInferenceDefault(str(exc)) from exc

        if first_value is util.Uninferable:
            raise UseInferenceDefault

        if isinstance(first_value, nodes.Const) and isinstance(
            first_value.value, (int, str)
        ):
            try:
                actual_value = int(first_value.value)
            except ValueError:
                return nodes.Const(0)
            return nodes.Const(actual_value)

    return nodes.Const(0)


def infer_dict_fromkeys(node, context=None):
    """Infer dict.fromkeys

    :param nodes.Call node: dict.fromkeys() call to infer
    :param context.InferenceContext context: node context
    :rtype nodes.Dict:
        a Dictionary containing the values that astroid was able to infer.
        In case the inference failed for any reason, an empty dictionary
        will be inferred instead.
    """

    def _build_dict_with_elements(elements):
        new_node = nodes.Dict(
            col_offset=node.col_offset, lineno=node.lineno, parent=node.parent
        )
        new_node.postinit(elements)
        return new_node

    call = arguments.CallSite.from_call(node, context=context)
    if call.keyword_arguments:
        raise UseInferenceDefault("TypeError: int() must take no keyword arguments")
    if len(call.positional_arguments) not in {1, 2}:
        raise UseInferenceDefault(
            "TypeError: Needs between 1 and 2 positional arguments"
        )

    default = nodes.Const(None)
    values = call.positional_arguments[0]
    try:
        inferred_values = next(values.infer(context=context))
    except (InferenceError, StopIteration):
        return _build_dict_with_elements([])
    if inferred_values is util.Uninferable:
        return _build_dict_with_elements([])

    # Limit to a couple of potential values, as this can become pretty complicated
    accepted_iterable_elements = (nodes.Const,)
    if isinstance(inferred_values, (nodes.List, nodes.Set, nodes.Tuple)):
        elements = inferred_values.elts
        for element in elements:
            if not isinstance(element, accepted_iterable_elements):
                # Fallback to an empty dict
                return _build_dict_with_elements([])

        elements_with_value = [(element, default) for element in elements]
        return _build_dict_with_elements(elements_with_value)
    if isinstance(inferred_values, nodes.Const) and isinstance(
        inferred_values.value, (str, bytes)
    ):
        elements = [
            (nodes.Const(element), default) for element in inferred_values.value
        ]
        return _build_dict_with_elements(elements)
    if isinstance(inferred_values, nodes.Dict):
        keys = inferred_values.itered()
        for key in keys:
            if not isinstance(key, accepted_iterable_elements):
                # Fallback to an empty dict
                return _build_dict_with_elements([])

        elements_with_value = [(element, default) for element in keys]
        return _build_dict_with_elements(elements_with_value)

    # Fallback to an empty dictionary
    return _build_dict_with_elements([])


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
