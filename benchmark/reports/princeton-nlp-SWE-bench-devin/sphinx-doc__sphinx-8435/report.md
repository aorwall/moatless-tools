# sphinx-doc__sphinx-8435

| **sphinx-doc/sphinx** | `5d8d6275a54f2c5fb72b82383b5712c22d337634` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | 5776 |
| **Avg pos** | 8.0 |
| **Min pos** | 8 |
| **Max pos** | 8 |
| **Top file pos** | 1 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sphinx/ext/autodoc/__init__.py b/sphinx/ext/autodoc/__init__.py
--- a/sphinx/ext/autodoc/__init__.py
+++ b/sphinx/ext/autodoc/__init__.py
@@ -1702,7 +1702,8 @@ def add_directive_header(self, sig: str) -> None:
         if not self.options.annotation:
             # obtain annotation for this data
             try:
-                annotations = get_type_hints(self.parent)
+                annotations = get_type_hints(self.parent, None,
+                                             self.config.autodoc_type_aliases)
             except NameError:
                 # Failed to evaluate ForwardRef (maybe TYPE_CHECKING)
                 annotations = safe_getattr(self.parent, '__annotations__', {})
@@ -2093,7 +2094,8 @@ def add_directive_header(self, sig: str) -> None:
         if not self.options.annotation:
             # obtain type annotation for this attribute
             try:
-                annotations = get_type_hints(self.parent)
+                annotations = get_type_hints(self.parent, None,
+                                             self.config.autodoc_type_aliases)
             except NameError:
                 # Failed to evaluate ForwardRef (maybe TYPE_CHECKING)
                 annotations = safe_getattr(self.parent, '__annotations__', {})

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sphinx/ext/autodoc/__init__.py | 1705 | 1705 | - | 1 | -
| sphinx/ext/autodoc/__init__.py | 2096 | 2096 | 8 | 1 | 5776


## Problem Statement

```
autodoc_type_aliases does not effect to variables and attributes
**Describe the bug**
autodoc_type_aliases does not effect to variables and attributes

**To Reproduce**

\`\`\`
# example.py
from __future__ import annotations


#: blah blah blah
var: String


class MyString:
    "mystring"

    #: blah blah blah
    var: String
\`\`\`
\`\`\`
# index.rst
.. automodule:: example
   :members:
   :undoc-members:
\`\`\`
\`\`\`
# conf.py
autodoc_type_aliases = {
    'String': 'example.MyString'
}
\`\`\`

**Expected behavior**
`autodoc_type_aliases` should be applied to `example.var` and `example.MyString.var`.

**Your project**
N/A

**Screenshots**
N/A

**Environment info**
- OS: Mac
- Python version: 3.9.0
- Sphinx version: HEAD of 3.x branch
- Sphinx extensions: sphinx.ext.autodoc
- Extra tools: Nothing

**Additional context**
N/A

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 sphinx/ext/autodoc/__init__.py** | 1842 | 1852| 128 | 128 | 19989 | 
| 2 | **1 sphinx/ext/autodoc/__init__.py** | 1812 | 1840| 237 | 365 | 19989 | 
| 3 | **1 sphinx/ext/autodoc/__init__.py** | 13 | 114| 775 | 1140 | 19989 | 
| 4 | **1 sphinx/ext/autodoc/__init__.py** | 1787 | 1809| 191 | 1331 | 19989 | 
| 5 | 2 sphinx/ext/autodoc/mock.py | 71 | 93| 200 | 1531 | 21096 | 
| 6 | **2 sphinx/ext/autodoc/__init__.py** | 1388 | 1665| 2641 | 4172 | 21096 | 
| 7 | **2 sphinx/ext/autodoc/__init__.py** | 2304 | 2342| 478 | 4650 | 21096 | 
| **-> 8 <-** | **2 sphinx/ext/autodoc/__init__.py** | 2016 | 2149| 1126 | 5776 | 21096 | 
| 9 | **2 sphinx/ext/autodoc/__init__.py** | 1024 | 1052| 232 | 6008 | 21096 | 
| 10 | **2 sphinx/ext/autodoc/__init__.py** | 2183 | 2224| 321 | 6329 | 21096 | 
| 11 | **2 sphinx/ext/autodoc/__init__.py** | 1855 | 2000| 1279 | 7608 | 21096 | 
| 12 | 3 sphinx/ext/autodoc/typehints.py | 82 | 138| 460 | 8068 | 22138 | 
| 13 | 4 sphinx/ext/autodoc/type_comment.py | 115 | 140| 257 | 8325 | 23362 | 
| 14 | **4 sphinx/ext/autodoc/__init__.py** | 285 | 359| 784 | 9109 | 23362 | 
| 15 | **4 sphinx/ext/autodoc/__init__.py** | 1241 | 1341| 836 | 9945 | 23362 | 
| 16 | **4 sphinx/ext/autodoc/__init__.py** | 681 | 782| 829 | 10774 | 23362 | 
| 17 | **4 sphinx/ext/autodoc/__init__.py** | 1749 | 1784| 279 | 11053 | 23362 | 
| 18 | **4 sphinx/ext/autodoc/__init__.py** | 117 | 150| 207 | 11260 | 23362 | 
| 19 | **4 sphinx/ext/autodoc/__init__.py** | 2281 | 2301| 226 | 11486 | 23362 | 
| 20 | 5 sphinx/ext/autodoc/importer.py | 244 | 263| 180 | 11666 | 25390 | 
| 21 | 5 sphinx/ext/autodoc/mock.py | 11 | 68| 451 | 12117 | 25390 | 
| 22 | **5 sphinx/ext/autodoc/__init__.py** | 569 | 581| 133 | 12250 | 25390 | 
| 23 | 5 sphinx/ext/autodoc/importer.py | 73 | 136| 597 | 12847 | 25390 | 
| 24 | **5 sphinx/ext/autodoc/__init__.py** | 1200 | 1220| 241 | 13088 | 25390 | 
| 25 | **5 sphinx/ext/autodoc/__init__.py** | 2227 | 2243| 126 | 13214 | 25390 | 
| 26 | 6 sphinx/ext/autodoc/directive.py | 9 | 49| 298 | 13512 | 26651 | 
| 27 | **6 sphinx/ext/autodoc/__init__.py** | 987 | 1009| 223 | 13735 | 26651 | 
| 28 | 6 sphinx/ext/autodoc/type_comment.py | 11 | 35| 239 | 13974 | 26651 | 
| 29 | 7 sphinx/domains/python.py | 11 | 78| 526 | 14500 | 38556 | 
| 30 | 8 sphinx/util/typing.py | 11 | 63| 400 | 14900 | 42905 | 
| 31 | **8 sphinx/ext/autodoc/__init__.py** | 248 | 282| 279 | 15179 | 42905 | 
| 32 | 8 sphinx/domains/python.py | 639 | 667| 230 | 15409 | 42905 | 
| 33 | 9 sphinx/ext/autosummary/generate.py | 175 | 194| 176 | 15585 | 48173 | 
| 34 | **9 sphinx/ext/autodoc/__init__.py** | 1054 | 1071| 177 | 15762 | 48173 | 
| 35 | **9 sphinx/ext/autodoc/__init__.py** | 1011 | 1022| 126 | 15888 | 48173 | 
| 36 | 9 sphinx/ext/autosummary/generate.py | 20 | 57| 275 | 16163 | 48173 | 
| 37 | 10 sphinx/transforms/references.py | 11 | 55| 266 | 16429 | 48491 | 
| 38 | 11 doc/usage/extensions/example_numpy.py | 320 | 334| 109 | 16538 | 50599 | 
| 39 | 11 sphinx/ext/autodoc/importer.py | 11 | 36| 175 | 16713 | 50599 | 
| 40 | 12 sphinx/domains/c.py | 3414 | 3837| 3704 | 20417 | 81802 | 
| 41 | 12 sphinx/ext/autodoc/typehints.py | 11 | 37| 197 | 20614 | 81802 | 
| 42 | 13 sphinx/application.py | 1079 | 1092| 174 | 20788 | 93092 | 
| 43 | 13 sphinx/ext/autodoc/typehints.py | 40 | 79| 324 | 21112 | 93092 | 
| 44 | 13 sphinx/domains/python.py | 266 | 284| 214 | 21326 | 93092 | 
| 45 | 14 sphinx/deprecation.py | 11 | 36| 164 | 21490 | 93804 | 
| 46 | 15 sphinx/pycode/ast.py | 11 | 44| 201 | 21691 | 95744 | 
| 47 | 15 doc/usage/extensions/example_numpy.py | 336 | 356| 120 | 21811 | 95744 | 
| 48 | **15 sphinx/ext/autodoc/__init__.py** | 372 | 407| 315 | 22126 | 95744 | 
| 49 | 15 sphinx/util/typing.py | 255 | 278| 188 | 22314 | 95744 | 
| 50 | **15 sphinx/ext/autodoc/__init__.py** | 1074 | 1092| 179 | 22493 | 95744 | 
| 51 | 16 sphinx/directives/__init__.py | 267 | 321| 517 | 23010 | 98479 | 
| 52 | 17 sphinx/util/inspect.py | 325 | 337| 144 | 23154 | 105051 | 
| 53 | **17 sphinx/ext/autodoc/__init__.py** | 1095 | 1125| 287 | 23441 | 105051 | 
| 54 | 17 sphinx/ext/autosummary/generate.py | 87 | 117| 288 | 23729 | 105051 | 
| 55 | 18 doc/usage/extensions/example_google.py | 261 | 275| 109 | 23838 | 107036 | 
| 56 | **18 sphinx/ext/autodoc/__init__.py** | 1357 | 1665| 189 | 24027 | 107036 | 
| 57 | 19 sphinx/util/pycompat.py | 91 | 110| 158 | 24185 | 107904 | 
| 58 | **19 sphinx/ext/autodoc/__init__.py** | 1223 | 1238| 166 | 24351 | 107904 | 
| 59 | 19 doc/usage/extensions/example_google.py | 277 | 297| 120 | 24471 | 107904 | 
| 60 | 19 sphinx/ext/autosummary/generate.py | 228 | 243| 184 | 24655 | 107904 | 
| 61 | 19 sphinx/ext/autodoc/importer.py | 160 | 180| 157 | 24812 | 107904 | 
| 62 | 19 sphinx/domains/python.py | 1009 | 1027| 143 | 24955 | 107904 | 
| 63 | 20 sphinx/domains/cpp.py | 2393 | 2450| 494 | 25449 | 171008 | 
| 64 | **20 sphinx/ext/autodoc/__init__.py** | 430 | 478| 359 | 25808 | 171008 | 
| 65 | 20 sphinx/ext/autodoc/directive.py | 109 | 159| 453 | 26261 | 171008 | 
| 66 | 21 doc/conf.py | 59 | 124| 668 | 26929 | 172546 | 
| 67 | 21 sphinx/util/inspect.py | 11 | 41| 263 | 27192 | 172546 | 
| 68 | **21 sphinx/ext/autodoc/__init__.py** | 2152 | 2180| 247 | 27439 | 172546 | 
| 69 | 22 sphinx/config.py | 386 | 438| 474 | 27913 | 176947 | 
| 70 | 23 sphinx/util/__init__.py | 480 | 493| 123 | 28036 | 183249 | 
| 71 | 24 sphinx/pycode/parser.py | 47 | 86| 391 | 28427 | 187895 | 
| 72 | 24 sphinx/domains/python.py | 1369 | 1407| 304 | 28731 | 187895 | 
| 73 | 24 sphinx/ext/autosummary/generate.py | 272 | 285| 202 | 28933 | 187895 | 
| 74 | **24 sphinx/ext/autodoc/__init__.py** | 2245 | 2264| 170 | 29103 | 187895 | 
| 75 | 25 sphinx/util/cfamily.py | 179 | 230| 341 | 29444 | 191397 | 
| 76 | 25 sphinx/directives/__init__.py | 74 | 86| 128 | 29572 | 191397 | 
| 77 | 25 sphinx/domains/python.py | 727 | 781| 530 | 30102 | 191397 | 
| 78 | 26 sphinx/directives/other.py | 9 | 39| 235 | 30337 | 194568 | 


## Patch

```diff
diff --git a/sphinx/ext/autodoc/__init__.py b/sphinx/ext/autodoc/__init__.py
--- a/sphinx/ext/autodoc/__init__.py
+++ b/sphinx/ext/autodoc/__init__.py
@@ -1702,7 +1702,8 @@ def add_directive_header(self, sig: str) -> None:
         if not self.options.annotation:
             # obtain annotation for this data
             try:
-                annotations = get_type_hints(self.parent)
+                annotations = get_type_hints(self.parent, None,
+                                             self.config.autodoc_type_aliases)
             except NameError:
                 # Failed to evaluate ForwardRef (maybe TYPE_CHECKING)
                 annotations = safe_getattr(self.parent, '__annotations__', {})
@@ -2093,7 +2094,8 @@ def add_directive_header(self, sig: str) -> None:
         if not self.options.annotation:
             # obtain type annotation for this attribute
             try:
-                annotations = get_type_hints(self.parent)
+                annotations = get_type_hints(self.parent, None,
+                                             self.config.autodoc_type_aliases)
             except NameError:
                 # Failed to evaluate ForwardRef (maybe TYPE_CHECKING)
                 annotations = safe_getattr(self.parent, '__annotations__', {})

```

## Test Patch

```diff
diff --git a/tests/roots/test-ext-autodoc/target/annotations.py b/tests/roots/test-ext-autodoc/target/annotations.py
--- a/tests/roots/test-ext-autodoc/target/annotations.py
+++ b/tests/roots/test-ext-autodoc/target/annotations.py
@@ -4,6 +4,9 @@
 
 myint = int
 
+#: docstring
+variable: myint
+
 
 def sum(x: myint, y: myint) -> myint:
     """docstring"""
@@ -23,3 +26,10 @@ def mult(x: float, y: float) -> float:
 def mult(x, y):
     """docstring"""
     return x, y
+
+
+class Foo:
+    """docstring"""
+
+    #: docstring
+    attr: myint
diff --git a/tests/test_ext_autodoc_configs.py b/tests/test_ext_autodoc_configs.py
--- a/tests/test_ext_autodoc_configs.py
+++ b/tests/test_ext_autodoc_configs.py
@@ -700,6 +700,19 @@ def test_autodoc_type_aliases(app):
         '.. py:module:: target.annotations',
         '',
         '',
+        '.. py:class:: Foo()',
+        '   :module: target.annotations',
+        '',
+        '   docstring',
+        '',
+        '',
+        '   .. py:attribute:: Foo.attr',
+        '      :module: target.annotations',
+        '      :type: int',
+        '',
+        '      docstring',
+        '',
+        '',
         '.. py:function:: mult(x: int, y: int) -> int',
         '                 mult(x: float, y: float) -> float',
         '   :module: target.annotations',
@@ -712,6 +725,13 @@ def test_autodoc_type_aliases(app):
         '',
         '   docstring',
         '',
+        '',
+        '.. py:data:: variable',
+        '   :module: target.annotations',
+        '   :type: int',
+        '',
+        '   docstring',
+        '',
     ]
 
     # define aliases
@@ -722,6 +742,19 @@ def test_autodoc_type_aliases(app):
         '.. py:module:: target.annotations',
         '',
         '',
+        '.. py:class:: Foo()',
+        '   :module: target.annotations',
+        '',
+        '   docstring',
+        '',
+        '',
+        '   .. py:attribute:: Foo.attr',
+        '      :module: target.annotations',
+        '      :type: myint',
+        '',
+        '      docstring',
+        '',
+        '',
         '.. py:function:: mult(x: myint, y: myint) -> myint',
         '                 mult(x: float, y: float) -> float',
         '   :module: target.annotations',
@@ -734,6 +767,13 @@ def test_autodoc_type_aliases(app):
         '',
         '   docstring',
         '',
+        '',
+        '.. py:data:: variable',
+        '   :module: target.annotations',
+        '   :type: myint',
+        '',
+        '   docstring',
+        '',
     ]
 
 

```


## Code snippets

### 1 - sphinx/ext/autodoc/__init__.py:

Start line: 1842, End line: 1852

```python
class TypeVarDocumenter(DataDocumenter):

    def add_content(self, more_content: Any, no_docstring: bool = False) -> None:
        attrs = [repr(self.object.__name__)]
        for constraint in self.object.__constraints__:
            attrs.append(stringify_typehint(constraint))
        if self.object.__covariant__:
            attrs.append("covariant=True")
        if self.object.__contravariant__:
            attrs.append("contravariant=True")

        content = StringList([_('alias of TypeVar(%s)') % ", ".join(attrs)], source='')
        super().add_content(content)
```
### 2 - sphinx/ext/autodoc/__init__.py:

Start line: 1812, End line: 1840

```python
class TypeVarDocumenter(DataDocumenter):
    """
    Specialized Documenter subclass for TypeVars.
    """

    objtype = 'typevar'
    directivetype = 'data'
    priority = DataDocumenter.priority + 1

    @classmethod
    def can_document_member(cls, member: Any, membername: str, isattr: bool, parent: Any
                            ) -> bool:
        return isinstance(member, TypeVar) and isattr

    def add_directive_header(self, sig: str) -> None:
        self.options = Options(self.options)
        self.options['annotation'] = SUPPRESS
        super().add_directive_header(sig)

    def get_doc(self, encoding: str = None, ignore: int = None) -> List[List[str]]:
        if ignore is not None:
            warnings.warn("The 'ignore' argument to autodoc.%s.get_doc() is deprecated."
                          % self.__class__.__name__,
                          RemovedInSphinx50Warning, stacklevel=2)

        if self.object.__doc__ != TypeVar.__doc__:
            return super().get_doc()
        else:
            return []
```
### 3 - sphinx/ext/autodoc/__init__.py:

Start line: 13, End line: 114

```python
import importlib
import re
import warnings
from inspect import Parameter, Signature
from types import ModuleType
from typing import (Any, Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Type,
                    TypeVar, Union, get_type_hints)

from docutils.statemachine import StringList

import sphinx
from sphinx.application import Sphinx
from sphinx.config import ENUM, Config
from sphinx.deprecation import RemovedInSphinx40Warning, RemovedInSphinx50Warning
from sphinx.environment import BuildEnvironment
from sphinx.ext.autodoc.importer import get_module_members, get_object_members, import_object
from sphinx.ext.autodoc.mock import mock
from sphinx.locale import _, __
from sphinx.pycode import ModuleAnalyzer, PycodeError
from sphinx.util import inspect, logging
from sphinx.util.docstrings import extract_metadata, prepare_docstring
from sphinx.util.inspect import (evaluate_signature, getdoc, object_description, safe_getattr,
                                 stringify_signature)
from sphinx.util.typing import restify
from sphinx.util.typing import stringify as stringify_typehint

if False:
    # For type annotation
    from typing import Type  # NOQA # for python3.5.1

    from sphinx.ext.autodoc.directive import DocumenterBridge


logger = logging.getLogger(__name__)


# This type isn't exposed directly in any modules, but can be found
# here in most Python versions
MethodDescriptorType = type(type.__subclasses__)


#: extended signature RE: with explicit module name separated by ::
py_ext_sig_re = re.compile(
    r'''^ ([\w.]+::)?            # explicit module name
          ([\w.]+\.)?            # module and/or class name(s)
          (\w+)  \s*             # thing name
          (?: \((.*)\)           # optional: arguments
           (?:\s* -> \s* (.*))?  #           return annotation
          )? $                   # and nothing more
          ''', re.VERBOSE)
special_member_re = re.compile(r'^__\S+__$')


def identity(x: Any) -> Any:
    return x


class _All:
    """A special value for :*-members: that matches to any member."""

    def __contains__(self, item: Any) -> bool:
        return True


class _Empty:
    """A special value for :exclude-members: that never matches to any member."""

    def __contains__(self, item: Any) -> bool:
        return False


ALL = _All()
EMPTY = _Empty()
UNINITIALIZED_ATTR = object()
INSTANCEATTR = object()
SLOTSATTR = object()


def members_option(arg: Any) -> Union[object, List[str]]:
    """Used to convert the :members: option to auto directives."""
    if arg is None or arg is True:
        return ALL
    elif arg is False:
        return None
    else:
        return [x.strip() for x in arg.split(',') if x.strip()]


def members_set_option(arg: Any) -> Union[object, Set[str]]:
    """Used to convert the :members: option to auto directives."""
    warnings.warn("members_set_option() is deprecated.",
                  RemovedInSphinx50Warning, stacklevel=2)
    if arg is None:
        return ALL
    return {x.strip() for x in arg.split(',') if x.strip()}


def exclude_members_option(arg: Any) -> Union[object, Set[str]]:
    """Used to convert the :exclude-members: option."""
    if arg is None:
        return EMPTY
    return {x.strip() for x in arg.split(',') if x.strip()}
```
### 4 - sphinx/ext/autodoc/__init__.py:

Start line: 1787, End line: 1809

```python
class GenericAliasDocumenter(DataDocumenter):
    """
    Specialized Documenter subclass for GenericAliases.
    """

    objtype = 'genericalias'
    directivetype = 'data'
    priority = DataDocumenter.priority + 1

    @classmethod
    def can_document_member(cls, member: Any, membername: str, isattr: bool, parent: Any
                            ) -> bool:
        return inspect.isgenericalias(member)

    def add_directive_header(self, sig: str) -> None:
        self.options = Options(self.options)
        self.options['annotation'] = SUPPRESS
        super().add_directive_header(sig)

    def add_content(self, more_content: Any, no_docstring: bool = False) -> None:
        name = stringify_typehint(self.object)
        content = StringList([_('alias of %s') % name], source='')
        super().add_content(content)
```
### 5 - sphinx/ext/autodoc/mock.py:

Start line: 71, End line: 93

```python
def _make_subclass(name: str, module: str, superclass: Any = _MockObject,
                   attributes: Any = None) -> Any:
    attrs = {'__module__': module, '__display_name__': module + '.' + name}
    attrs.update(attributes or {})

    return type(name, (superclass,), attrs)


class _MockModule(ModuleType):
    """Used by autodoc_mock_imports."""
    __file__ = os.devnull
    __sphinx_mock__ = True

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.__all__ = []  # type: List[str]
        self.__path__ = []  # type: List[str]

    def __getattr__(self, name: str) -> _MockObject:
        return _make_subclass(name, self.__name__)()

    def __repr__(self) -> str:
        return self.__name__
```
### 6 - sphinx/ext/autodoc/__init__.py:

Start line: 1388, End line: 1665

```python
class ClassDocumenter(DocstringSignatureMixin, ModuleLevelDocumenter):  # type: ignore
    """
    Specialized Documenter subclass for classes.
    """
    objtype = 'class'
    member_order = 20
    option_spec = {
        'members': members_option, 'undoc-members': bool_option,
        'noindex': bool_option, 'inherited-members': inherited_members_option,
        'show-inheritance': bool_option, 'member-order': member_order_option,
        'exclude-members': exclude_members_option,
        'private-members': members_option, 'special-members': members_option,
    }  # type: Dict[str, Callable]

    _signature_class = None  # type: Any
    _signature_method_name = None  # type: str

    def __init__(self, *args: Any) -> None:
        super().__init__(*args)
        merge_members_option(self.options)

    @classmethod
    def can_document_member(cls, member: Any, membername: str, isattr: bool, parent: Any
                            ) -> bool:
        return isinstance(member, type)

    def import_object(self, raiseerror: bool = False) -> bool:
        ret = super().import_object(raiseerror)
        # if the class is documented under another name, document it
        # as data/attribute
        if ret:
            if hasattr(self.object, '__name__'):
                self.doc_as_attr = (self.objpath[-1] != self.object.__name__)
            else:
                self.doc_as_attr = True
        return ret

    def _get_signature(self) -> Tuple[Optional[Any], Optional[str], Optional[Signature]]:
        def get_user_defined_function_or_method(obj: Any, attr: str) -> Any:
            """ Get the `attr` function or method from `obj`, if it is user-defined. """
            if inspect.is_builtin_class_method(obj, attr):
                return None
            attr = self.get_attr(obj, attr, None)
            if not (inspect.ismethod(attr) or inspect.isfunction(attr)):
                return None
            return attr

        # This sequence is copied from inspect._signature_from_callable.
        # ValueError means that no signature could be found, so we keep going.

        # First, we check the obj has a __signature__ attribute
        if (hasattr(self.object, '__signature__') and
                isinstance(self.object.__signature__, Signature)):
            return None, None, self.object.__signature__

        # Next, let's see if it has an overloaded __call__ defined
        # in its metaclass
        call = get_user_defined_function_or_method(type(self.object), '__call__')

        if call is not None:
            if "{0.__module__}.{0.__qualname__}".format(call) in _METACLASS_CALL_BLACKLIST:
                call = None

        if call is not None:
            self.env.app.emit('autodoc-before-process-signature', call, True)
            try:
                sig = inspect.signature(call, bound_method=True,
                                        type_aliases=self.config.autodoc_type_aliases)
                return type(self.object), '__call__', sig
            except ValueError:
                pass

        # Now we check if the 'obj' class has a '__new__' method
        new = get_user_defined_function_or_method(self.object, '__new__')

        if new is not None:
            if "{0.__module__}.{0.__qualname__}".format(new) in _CLASS_NEW_BLACKLIST:
                new = None

        if new is not None:
            self.env.app.emit('autodoc-before-process-signature', new, True)
            try:
                sig = inspect.signature(new, bound_method=True,
                                        type_aliases=self.config.autodoc_type_aliases)
                return self.object, '__new__', sig
            except ValueError:
                pass

        # Finally, we should have at least __init__ implemented
        init = get_user_defined_function_or_method(self.object, '__init__')
        if init is not None:
            self.env.app.emit('autodoc-before-process-signature', init, True)
            try:
                sig = inspect.signature(init, bound_method=True,
                                        type_aliases=self.config.autodoc_type_aliases)
                return self.object, '__init__', sig
            except ValueError:
                pass

        # None of the attributes are user-defined, so fall back to let inspect
        # handle it.
        # We don't know the exact method that inspect.signature will read
        # the signature from, so just pass the object itself to our hook.
        self.env.app.emit('autodoc-before-process-signature', self.object, False)
        try:
            sig = inspect.signature(self.object, bound_method=False,
                                    type_aliases=self.config.autodoc_type_aliases)
            return None, None, sig
        except ValueError:
            pass

        # Still no signature: happens e.g. for old-style classes
        # with __init__ in C and no `__text_signature__`.
        return None, None, None

    def format_args(self, **kwargs: Any) -> str:
        if self.config.autodoc_typehints in ('none', 'description'):
            kwargs.setdefault('show_annotation', False)

        try:
            self._signature_class, self._signature_method_name, sig = self._get_signature()
        except TypeError as exc:
            # __signature__ attribute contained junk
            logger.warning(__("Failed to get a constructor signature for %s: %s"),
                           self.fullname, exc)
            return None

        if sig is None:
            return None

        return stringify_signature(sig, show_return_annotation=False, **kwargs)

    def format_signature(self, **kwargs: Any) -> str:
        if self.doc_as_attr:
            return ''

        sig = super().format_signature()
        sigs = []

        overloads = self.get_overloaded_signatures()
        if overloads and self.config.autodoc_typehints == 'signature':
            # Use signatures for overloaded methods instead of the implementation method.
            method = safe_getattr(self._signature_class, self._signature_method_name, None)
            __globals__ = safe_getattr(method, '__globals__', {})
            for overload in overloads:
                overload = evaluate_signature(overload, __globals__,
                                              self.config.autodoc_type_aliases)

                parameters = list(overload.parameters.values())
                overload = overload.replace(parameters=parameters[1:],
                                            return_annotation=Parameter.empty)
                sig = stringify_signature(overload, **kwargs)
                sigs.append(sig)
        else:
            sigs.append(sig)

        return "\n".join(sigs)

    def get_overloaded_signatures(self) -> List[Signature]:
        if self._signature_class and self._signature_method_name:
            for cls in self._signature_class.__mro__:
                try:
                    analyzer = ModuleAnalyzer.for_module(cls.__module__)
                    analyzer.parse()
                    qualname = '.'.join([cls.__qualname__, self._signature_method_name])
                    if qualname in analyzer.overloads:
                        return analyzer.overloads.get(qualname)
                    elif qualname in analyzer.tagorder:
                        # the constructor is defined in the class, but not overrided.
                        return []
                except PycodeError:
                    pass

        return []

    def add_directive_header(self, sig: str) -> None:
        sourcename = self.get_sourcename()

        if self.doc_as_attr:
            self.directivetype = 'attribute'
        super().add_directive_header(sig)

        if self.analyzer and '.'.join(self.objpath) in self.analyzer.finals:
            self.add_line('   :final:', sourcename)

        # add inheritance info, if wanted
        if not self.doc_as_attr and self.options.show_inheritance:
            sourcename = self.get_sourcename()
            self.add_line('', sourcename)

            if hasattr(self.object, '__orig_bases__') and len(self.object.__orig_bases__):
                # A subclass of generic types
                # refs: PEP-560 <https://www.python.org/dev/peps/pep-0560/>
                bases = [restify(cls) for cls in self.object.__orig_bases__]
                self.add_line('   ' + _('Bases: %s') % ', '.join(bases), sourcename)
            elif hasattr(self.object, '__bases__') and len(self.object.__bases__):
                # A normal class
                bases = [restify(cls) for cls in self.object.__bases__]
                self.add_line('   ' + _('Bases: %s') % ', '.join(bases), sourcename)

    def get_doc(self, encoding: str = None, ignore: int = None) -> List[List[str]]:
        if encoding is not None:
            warnings.warn("The 'encoding' argument to autodoc.%s.get_doc() is deprecated."
                          % self.__class__.__name__,
                          RemovedInSphinx40Warning, stacklevel=2)
        lines = getattr(self, '_new_docstrings', None)
        if lines is not None:
            return lines

        content = self.config.autoclass_content

        docstrings = []
        attrdocstring = self.get_attr(self.object, '__doc__', None)
        if attrdocstring:
            docstrings.append(attrdocstring)

        # for classes, what the "docstring" is can be controlled via a
        # config value; the default is only the class docstring
        if content in ('both', 'init'):
            __init__ = self.get_attr(self.object, '__init__', None)
            initdocstring = getdoc(__init__, self.get_attr,
                                   self.config.autodoc_inherit_docstrings,
                                   self.parent, self.object_name)
            # for new-style classes, no __init__ means default __init__
            if (initdocstring is not None and
                (initdocstring == object.__init__.__doc__ or  # for pypy
                 initdocstring.strip() == object.__init__.__doc__)):  # for !pypy
                initdocstring = None
            if not initdocstring:
                # try __new__
                __new__ = self.get_attr(self.object, '__new__', None)
                initdocstring = getdoc(__new__, self.get_attr,
                                       self.config.autodoc_inherit_docstrings,
                                       self.parent, self.object_name)
                # for new-style classes, no __new__ means default __new__
                if (initdocstring is not None and
                    (initdocstring == object.__new__.__doc__ or  # for pypy
                     initdocstring.strip() == object.__new__.__doc__)):  # for !pypy
                    initdocstring = None
            if initdocstring:
                if content == 'init':
                    docstrings = [initdocstring]
                else:
                    docstrings.append(initdocstring)

        tab_width = self.directive.state.document.settings.tab_width
        return [prepare_docstring(docstring, ignore, tab_width) for docstring in docstrings]

    def add_content(self, more_content: Any, no_docstring: bool = False) -> None:
        if self.doc_as_attr:
            classname = safe_getattr(self.object, '__qualname__', None)
            if not classname:
                classname = safe_getattr(self.object, '__name__', None)
            if classname:
                module = safe_getattr(self.object, '__module__', None)
                parentmodule = safe_getattr(self.parent, '__module__', None)
                if module and module != parentmodule:
                    classname = str(module) + '.' + str(classname)
                content = StringList([_('alias of :class:`%s`') % classname], source='')
                super().add_content(content, no_docstring=True)
        else:
            super().add_content(more_content)

    def document_members(self, all_members: bool = False) -> None:
        if self.doc_as_attr:
            return
        super().document_members(all_members)

    def generate(self, more_content: Any = None, real_modname: str = None,
                 check_module: bool = False, all_members: bool = False) -> None:
        # Do not pass real_modname and use the name from the __module__
        # attribute of the class.
        # If a class gets imported into the module real_modname
        # the analyzer won't find the source of the class, if
        # it looks in real_modname.
        return super().generate(more_content=more_content,
                                check_module=check_module,
                                all_members=all_members)
```
### 7 - sphinx/ext/autodoc/__init__.py:

Start line: 2304, End line: 2342

```python
def setup(app: Sphinx) -> Dict[str, Any]:
    app.add_autodocumenter(ModuleDocumenter)
    app.add_autodocumenter(ClassDocumenter)
    app.add_autodocumenter(ExceptionDocumenter)
    app.add_autodocumenter(DataDocumenter)
    app.add_autodocumenter(DataDeclarationDocumenter)
    app.add_autodocumenter(GenericAliasDocumenter)
    app.add_autodocumenter(TypeVarDocumenter)
    app.add_autodocumenter(FunctionDocumenter)
    app.add_autodocumenter(DecoratorDocumenter)
    app.add_autodocumenter(MethodDocumenter)
    app.add_autodocumenter(AttributeDocumenter)
    app.add_autodocumenter(PropertyDocumenter)
    app.add_autodocumenter(InstanceAttributeDocumenter)
    app.add_autodocumenter(SlotsAttributeDocumenter)

    app.add_config_value('autoclass_content', 'class', True, ENUM('both', 'class', 'init'))
    app.add_config_value('autodoc_member_order', 'alphabetical', True,
                         ENUM('alphabetic', 'alphabetical', 'bysource', 'groupwise'))
    app.add_config_value('autodoc_default_options', {}, True)
    app.add_config_value('autodoc_docstring_signature', True, True)
    app.add_config_value('autodoc_mock_imports', [], True)
    app.add_config_value('autodoc_typehints', "signature", True,
                         ENUM("signature", "description", "none"))
    app.add_config_value('autodoc_type_aliases', {}, True)
    app.add_config_value('autodoc_warningiserror', True, True)
    app.add_config_value('autodoc_inherit_docstrings', True, True)
    app.add_event('autodoc-before-process-signature')
    app.add_event('autodoc-process-docstring')
    app.add_event('autodoc-process-signature')
    app.add_event('autodoc-skip-member')

    app.connect('config-inited', migrate_autodoc_member_order, priority=800)

    app.setup_extension('sphinx.ext.autodoc.type_comment')
    app.setup_extension('sphinx.ext.autodoc.typehints')

    return {'version': sphinx.__display_version__, 'parallel_read_safe': True}
```
### 8 - sphinx/ext/autodoc/__init__.py:

Start line: 2016, End line: 2149

```python
class AttributeDocumenter(DocstringStripSignatureMixin, ClassLevelDocumenter):  # type: ignore
    """
    Specialized Documenter subclass for attributes.
    """
    objtype = 'attribute'
    member_order = 60
    option_spec = dict(ModuleLevelDocumenter.option_spec)
    option_spec["annotation"] = annotation_option

    # must be higher than the MethodDocumenter, else it will recognize
    # some non-data descriptors as methods
    priority = 10

    @staticmethod
    def is_function_or_method(obj: Any) -> bool:
        return inspect.isfunction(obj) or inspect.isbuiltin(obj) or inspect.ismethod(obj)

    @classmethod
    def can_document_member(cls, member: Any, membername: str, isattr: bool, parent: Any
                            ) -> bool:
        if inspect.isattributedescriptor(member):
            return True
        elif (not isinstance(parent, ModuleDocumenter) and
              not inspect.isroutine(member) and
              not isinstance(member, type)):
            return True
        else:
            return False

    def document_members(self, all_members: bool = False) -> None:
        pass

    def isinstanceattribute(self) -> bool:
        """Check the subject is an instance attribute."""
        try:
            analyzer = ModuleAnalyzer.for_module(self.modname)
            attr_docs = analyzer.find_attr_docs()
            if self.objpath:
                key = ('.'.join(self.objpath[:-1]), self.objpath[-1])
                if key in attr_docs:
                    return True

            return False
        except PycodeError:
            return False

    def import_object(self, raiseerror: bool = False) -> bool:
        try:
            ret = super().import_object(raiseerror=True)
            if inspect.isenumattribute(self.object):
                self.object = self.object.value
            if inspect.isattributedescriptor(self.object):
                self._datadescriptor = True
            else:
                # if it's not a data descriptor
                self._datadescriptor = False
        except ImportError as exc:
            if self.isinstanceattribute():
                self.object = INSTANCEATTR
                self._datadescriptor = False
                ret = True
            elif raiseerror:
                raise
            else:
                logger.warning(exc.args[0], type='autodoc', subtype='import_object')
                self.env.note_reread()
                ret = False

        return ret

    def get_real_modname(self) -> str:
        return self.get_attr(self.parent or self.object, '__module__', None) \
            or self.modname

    def add_directive_header(self, sig: str) -> None:
        super().add_directive_header(sig)
        sourcename = self.get_sourcename()
        if not self.options.annotation:
            # obtain type annotation for this attribute
            try:
                annotations = get_type_hints(self.parent)
            except NameError:
                # Failed to evaluate ForwardRef (maybe TYPE_CHECKING)
                annotations = safe_getattr(self.parent, '__annotations__', {})
            except TypeError:
                annotations = {}
            except KeyError:
                # a broken class found (refs: https://github.com/sphinx-doc/sphinx/issues/8084)
                annotations = {}
            except AttributeError:
                # AttributeError is raised on 3.5.2 (fixed by 3.5.3)
                annotations = {}

            if self.objpath[-1] in annotations:
                objrepr = stringify_typehint(annotations.get(self.objpath[-1]))
                self.add_line('   :type: ' + objrepr, sourcename)
            else:
                key = ('.'.join(self.objpath[:-1]), self.objpath[-1])
                if self.analyzer and key in self.analyzer.annotations:
                    self.add_line('   :type: ' + self.analyzer.annotations[key],
                                  sourcename)

            # data descriptors do not have useful values
            if not self._datadescriptor:
                try:
                    if self.object is INSTANCEATTR:
                        pass
                    else:
                        objrepr = object_description(self.object)
                        self.add_line('   :value: ' + objrepr, sourcename)
                except ValueError:
                    pass
        elif self.options.annotation is SUPPRESS:
            pass
        else:
            self.add_line('   :annotation: %s' % self.options.annotation, sourcename)

    def get_doc(self, encoding: str = None, ignore: int = None) -> List[List[str]]:
        try:
            # Disable `autodoc_inherit_docstring` temporarily to avoid to obtain
            # a docstring from the value which descriptor returns unexpectedly.
            # ref: https://github.com/sphinx-doc/sphinx/issues/7805
            orig = self.config.autodoc_inherit_docstrings
            self.config.autodoc_inherit_docstrings = False  # type: ignore
            return super().get_doc(encoding, ignore)
        finally:
            self.config.autodoc_inherit_docstrings = orig  # type: ignore

    def add_content(self, more_content: Any, no_docstring: bool = False) -> None:
        if not self._datadescriptor:
            # if it's not a data descriptor, its docstring is very probably the
            # wrong thing to display
            no_docstring = True
        super().add_content(more_content, no_docstring)
```
### 9 - sphinx/ext/autodoc/__init__.py:

Start line: 1024, End line: 1052

```python
class ModuleDocumenter(Documenter):

    def get_object_members(self, want_all: bool) -> Tuple[bool, ObjectMembers]:
        if want_all:
            members = get_module_members(self.object)
            if not self.__all__:
                # for implicit module members, check __module__ to avoid
                # documenting imported objects
                return True, members
            else:
                ret = []
                for name, value in members:
                    if name in self.__all__:
                        ret.append(ObjectMember(name, value))
                    else:
                        ret.append(ObjectMember(name, value, skipped=True))

                return False, ret
        else:
            memberlist = self.options.members or []
            ret = []
            for name in memberlist:
                try:
                    value = safe_getattr(self.object, name)
                    ret.append(ObjectMember(name, value))
                except AttributeError:
                    logger.warning(__('missing attribute mentioned in :members: option: '
                                      'module %s, attribute %s') %
                                   (safe_getattr(self.object, '__name__', '???'), name),
                                   type='autodoc')
            return False, ret
```
### 10 - sphinx/ext/autodoc/__init__.py:

Start line: 2183, End line: 2224

```python
class InstanceAttributeDocumenter(AttributeDocumenter):
    """
    Specialized Documenter subclass for attributes that cannot be imported
    because they are instance attributes (e.g. assigned in __init__).
    """
    objtype = 'instanceattribute'
    directivetype = 'attribute'
    member_order = 60

    # must be higher than AttributeDocumenter
    priority = 11

    @classmethod
    def can_document_member(cls, member: Any, membername: str, isattr: bool, parent: Any
                            ) -> bool:
        """This documents only INSTANCEATTR members."""
        return (not isinstance(parent, ModuleDocumenter) and
                isattr and
                member is INSTANCEATTR)

    def import_parent(self) -> Any:
        try:
            parent = importlib.import_module(self.modname)
            for name in self.objpath[:-1]:
                parent = self.get_attr(parent, name)

            return parent
        except (ImportError, AttributeError):
            return None

    def import_object(self, raiseerror: bool = False) -> bool:
        """Never import anything."""
        # disguise as an attribute
        self.objtype = 'attribute'
        self.object = INSTANCEATTR
        self.parent = self.import_parent()
        self._datadescriptor = False
        return True

    def add_content(self, more_content: Any, no_docstring: bool = False) -> None:
        """Never try to get a docstring from the object."""
        super().add_content(more_content, no_docstring=True)
```
### 11 - sphinx/ext/autodoc/__init__.py:

Start line: 1855, End line: 2000

```python
class MethodDocumenter(DocstringSignatureMixin, ClassLevelDocumenter):  # type: ignore
    """
    Specialized Documenter subclass for methods (normal, static and class).
    """
    objtype = 'method'
    directivetype = 'method'
    member_order = 50
    priority = 1  # must be more than FunctionDocumenter

    @classmethod
    def can_document_member(cls, member: Any, membername: str, isattr: bool, parent: Any
                            ) -> bool:
        return inspect.isroutine(member) and \
            not isinstance(parent, ModuleDocumenter)

    def import_object(self, raiseerror: bool = False) -> bool:
        ret = super().import_object(raiseerror)
        if not ret:
            return ret

        # to distinguish classmethod/staticmethod
        obj = self.parent.__dict__.get(self.object_name)
        if obj is None:
            obj = self.object

        if (inspect.isclassmethod(obj) or
                inspect.isstaticmethod(obj, cls=self.parent, name=self.object_name)):
            # document class and static members before ordinary ones
            self.member_order = self.member_order - 1

        return ret

    def format_args(self, **kwargs: Any) -> str:
        if self.config.autodoc_typehints in ('none', 'description'):
            kwargs.setdefault('show_annotation', False)

        try:
            if self.object == object.__init__ and self.parent != object:
                # Classes not having own __init__() method are shown as no arguments.
                #
                # Note: The signature of object.__init__() is (self, /, *args, **kwargs).
                #       But it makes users confused.
                args = '()'
            else:
                if inspect.isstaticmethod(self.object, cls=self.parent, name=self.object_name):
                    self.env.app.emit('autodoc-before-process-signature', self.object, False)
                    sig = inspect.signature(self.object, bound_method=False,
                                            type_aliases=self.config.autodoc_type_aliases)
                else:
                    self.env.app.emit('autodoc-before-process-signature', self.object, True)
                    sig = inspect.signature(self.object, bound_method=True,
                                            type_aliases=self.config.autodoc_type_aliases)
                args = stringify_signature(sig, **kwargs)
        except TypeError as exc:
            logger.warning(__("Failed to get a method signature for %s: %s"),
                           self.fullname, exc)
            return None
        except ValueError:
            args = ''

        if self.config.strip_signature_backslash:
            # escape backslashes for reST
            args = args.replace('\\', '\\\\')
        return args

    def add_directive_header(self, sig: str) -> None:
        super().add_directive_header(sig)

        sourcename = self.get_sourcename()
        obj = self.parent.__dict__.get(self.object_name, self.object)
        if inspect.isabstractmethod(obj):
            self.add_line('   :abstractmethod:', sourcename)
        if inspect.iscoroutinefunction(obj):
            self.add_line('   :async:', sourcename)
        if inspect.isclassmethod(obj):
            self.add_line('   :classmethod:', sourcename)
        if inspect.isstaticmethod(obj, cls=self.parent, name=self.object_name):
            self.add_line('   :staticmethod:', sourcename)
        if self.analyzer and '.'.join(self.objpath) in self.analyzer.finals:
            self.add_line('   :final:', sourcename)

    def document_members(self, all_members: bool = False) -> None:
        pass

    def format_signature(self, **kwargs: Any) -> str:
        sigs = []
        if (self.analyzer and
                '.'.join(self.objpath) in self.analyzer.overloads and
                self.config.autodoc_typehints == 'signature'):
            # Use signatures for overloaded methods instead of the implementation method.
            overloaded = True
        else:
            overloaded = False
            sig = super().format_signature(**kwargs)
            sigs.append(sig)

        meth = self.parent.__dict__.get(self.objpath[-1])
        if inspect.is_singledispatch_method(meth):
            # append signature of singledispatch'ed functions
            for typ, func in meth.dispatcher.registry.items():
                if typ is object:
                    pass  # default implementation. skipped.
                else:
                    self.annotate_to_first_argument(func, typ)

                    documenter = MethodDocumenter(self.directive, '')
                    documenter.parent = self.parent
                    documenter.object = func
                    documenter.objpath = [None]
                    sigs.append(documenter.format_signature())
        if overloaded:
            __globals__ = safe_getattr(self.object, '__globals__', {})
            for overload in self.analyzer.overloads.get('.'.join(self.objpath)):
                overload = evaluate_signature(overload, __globals__,
                                              self.config.autodoc_type_aliases)

                if not inspect.isstaticmethod(self.object, cls=self.parent,
                                              name=self.object_name):
                    parameters = list(overload.parameters.values())
                    overload = overload.replace(parameters=parameters[1:])
                sig = stringify_signature(overload, **kwargs)
                sigs.append(sig)

        return "\n".join(sigs)

    def annotate_to_first_argument(self, func: Callable, typ: Type) -> None:
        """Annotate type hint to the first argument of function if needed."""
        try:
            sig = inspect.signature(func, type_aliases=self.config.autodoc_type_aliases)
        except TypeError as exc:
            logger.warning(__("Failed to get a method signature for %s: %s"),
                           self.fullname, exc)
            return
        except ValueError:
            return
        if len(sig.parameters) == 1:
            return

        params = list(sig.parameters.values())
        if params[1].annotation is Parameter.empty:
            params[1] = params[1].replace(annotation=typ)
            try:
                func.__signature__ = sig.replace(parameters=params)  # type: ignore
            except TypeError:
                # failed to update signature (ex. built-in or extension types)
                return
```
### 14 - sphinx/ext/autodoc/__init__.py:

Start line: 285, End line: 359

```python
class Documenter:
    """
    A Documenter knows how to autodocument a single object type.  When
    registered with the AutoDirective, it will be used to document objects
    of that type when needed by autodoc.

    Its *objtype* attribute selects what auto directive it is assigned to
    (the directive name is 'auto' + objtype), and what directive it generates
    by default, though that can be overridden by an attribute called
    *directivetype*.

    A Documenter has an *option_spec* that works like a docutils directive's;
    in fact, it will be used to parse an auto directive's options that matches
    the documenter.
    """
    #: name by which the directive is called (auto...) and the default
    #: generated directive name
    objtype = 'object'
    #: indentation by which to indent the directive content
    content_indent = '   '
    #: priority if multiple documenters return True from can_document_member
    priority = 0
    #: order if autodoc_member_order is set to 'groupwise'
    member_order = 0
    #: true if the generated content may contain titles
    titles_allowed = False

    option_spec = {'noindex': bool_option}  # type: Dict[str, Callable]

    def get_attr(self, obj: Any, name: str, *defargs: Any) -> Any:
        """getattr() override for types such as Zope interfaces."""
        return autodoc_attrgetter(self.env.app, obj, name, *defargs)

    @classmethod
    def can_document_member(cls, member: Any, membername: str, isattr: bool, parent: Any
                            ) -> bool:
        """Called to see if a member can be documented by this documenter."""
        raise NotImplementedError('must be implemented in subclasses')

    def __init__(self, directive: "DocumenterBridge", name: str, indent: str = '') -> None:
        self.directive = directive
        self.config = directive.env.config
        self.env = directive.env    # type: BuildEnvironment
        self.options = directive.genopt
        self.name = name
        self.indent = indent
        # the module and object path within the module, and the fully
        # qualified name (all set after resolve_name succeeds)
        self.modname = None         # type: str
        self.module = None          # type: ModuleType
        self.objpath = None         # type: List[str]
        self.fullname = None        # type: str
        # extra signature items (arguments and return annotation,
        # also set after resolve_name succeeds)
        self.args = None            # type: str
        self.retann = None          # type: str
        # the object to document (set after import_object succeeds)
        self.object = None          # type: Any
        self.object_name = None     # type: str
        # the parent/owner of the object to document
        self.parent = None          # type: Any
        # the module analyzer to get at attribute docs, or None
        self.analyzer = None        # type: ModuleAnalyzer

    @property
    def documenters(self) -> Dict[str, "Type[Documenter]"]:
        """Returns registered Documenter classes"""
        return self.env.app.registry.documenters

    def add_line(self, line: str, source: str, *lineno: int) -> None:
        """Append one line of generated reST to the output."""
        if line.strip():  # not a blank line
            self.directive.result.append(self.indent + line, source, *lineno)
        else:
            self.directive.result.append('', source, *lineno)
```
### 15 - sphinx/ext/autodoc/__init__.py:

Start line: 1241, End line: 1341

```python
class FunctionDocumenter(DocstringSignatureMixin, ModuleLevelDocumenter):  # type: ignore
    """
    Specialized Documenter subclass for functions.
    """
    objtype = 'function'
    member_order = 30

    @classmethod
    def can_document_member(cls, member: Any, membername: str, isattr: bool, parent: Any
                            ) -> bool:
        # supports functions, builtins and bound methods exported at the module level
        return (inspect.isfunction(member) or inspect.isbuiltin(member) or
                (inspect.isroutine(member) and isinstance(parent, ModuleDocumenter)))

    def format_args(self, **kwargs: Any) -> str:
        if self.config.autodoc_typehints in ('none', 'description'):
            kwargs.setdefault('show_annotation', False)

        try:
            self.env.app.emit('autodoc-before-process-signature', self.object, False)
            sig = inspect.signature(self.object, type_aliases=self.config.autodoc_type_aliases)
            args = stringify_signature(sig, **kwargs)
        except TypeError as exc:
            logger.warning(__("Failed to get a function signature for %s: %s"),
                           self.fullname, exc)
            return None
        except ValueError:
            args = ''

        if self.config.strip_signature_backslash:
            # escape backslashes for reST
            args = args.replace('\\', '\\\\')
        return args

    def document_members(self, all_members: bool = False) -> None:
        pass

    def add_directive_header(self, sig: str) -> None:
        sourcename = self.get_sourcename()
        super().add_directive_header(sig)

        if inspect.iscoroutinefunction(self.object):
            self.add_line('   :async:', sourcename)

    def format_signature(self, **kwargs: Any) -> str:
        sigs = []
        if (self.analyzer and
                '.'.join(self.objpath) in self.analyzer.overloads and
                self.config.autodoc_typehints == 'signature'):
            # Use signatures for overloaded functions instead of the implementation function.
            overloaded = True
        else:
            overloaded = False
            sig = super().format_signature(**kwargs)
            sigs.append(sig)

        if inspect.is_singledispatch_function(self.object):
            # append signature of singledispatch'ed functions
            for typ, func in self.object.registry.items():
                if typ is object:
                    pass  # default implementation. skipped.
                else:
                    self.annotate_to_first_argument(func, typ)

                    documenter = FunctionDocumenter(self.directive, '')
                    documenter.object = func
                    documenter.objpath = [None]
                    sigs.append(documenter.format_signature())
        if overloaded:
            __globals__ = safe_getattr(self.object, '__globals__', {})
            for overload in self.analyzer.overloads.get('.'.join(self.objpath)):
                overload = evaluate_signature(overload, __globals__,
                                              self.config.autodoc_type_aliases)

                sig = stringify_signature(overload, **kwargs)
                sigs.append(sig)

        return "\n".join(sigs)

    def annotate_to_first_argument(self, func: Callable, typ: Type) -> None:
        """Annotate type hint to the first argument of function if needed."""
        try:
            sig = inspect.signature(func, type_aliases=self.config.autodoc_type_aliases)
        except TypeError as exc:
            logger.warning(__("Failed to get a function signature for %s: %s"),
                           self.fullname, exc)
            return
        except ValueError:
            return

        if len(sig.parameters) == 0:
            return

        params = list(sig.parameters.values())
        if params[0].annotation is Parameter.empty:
            params[0] = params[0].replace(annotation=typ)
            try:
                func.__signature__ = sig.replace(parameters=params)  # type: ignore
            except TypeError:
                # failed to update signature (ex. built-in or extension types)
                return
```
### 16 - sphinx/ext/autodoc/__init__.py:

Start line: 681, End line: 782

```python
class Documenter:

    def filter_members(self, members: ObjectMembers, want_all: bool
                       ) -> List[Tuple[str, Any, bool]]:
        # ... other code
        for obj in members:
            membername, member = obj
            # if isattr is True, the member is documented as an attribute
            if member is INSTANCEATTR:
                isattr = True
            else:
                isattr = False

            doc = getdoc(member, self.get_attr, self.config.autodoc_inherit_docstrings,
                         self.parent, self.object_name)
            if not isinstance(doc, str):
                # Ignore non-string __doc__
                doc = None

            # if the member __doc__ is the same as self's __doc__, it's just
            # inherited and therefore not the member's doc
            cls = self.get_attr(member, '__class__', None)
            if cls:
                cls_doc = self.get_attr(cls, '__doc__', None)
                if cls_doc == doc:
                    doc = None
            has_doc = bool(doc)

            metadata = extract_metadata(doc)
            if 'private' in metadata:
                # consider a member private if docstring has "private" metadata
                isprivate = True
            elif 'public' in metadata:
                # consider a member public if docstring has "public" metadata
                isprivate = False
            else:
                isprivate = membername.startswith('_')

            keep = False
            if safe_getattr(member, '__sphinx_mock__', False):
                # mocked module or object
                pass
            elif self.options.exclude_members and membername in self.options.exclude_members:
                # remove members given by exclude-members
                keep = False
            elif want_all and special_member_re.match(membername):
                # special __methods__
                if self.options.special_members and membername in self.options.special_members:
                    if membername == '__doc__':
                        keep = False
                    elif is_filtered_inherited_member(membername):
                        keep = False
                    else:
                        keep = has_doc or self.options.undoc_members
                else:
                    keep = False
            elif (namespace, membername) in attr_docs:
                if want_all and isprivate:
                    if self.options.private_members is None:
                        keep = False
                    else:
                        keep = membername in self.options.private_members
                else:
                    # keep documented attributes
                    keep = True
                isattr = True
            elif want_all and isprivate:
                if has_doc or self.options.undoc_members:
                    if self.options.private_members is None:
                        keep = False
                    elif is_filtered_inherited_member(membername):
                        keep = False
                    else:
                        keep = membername in self.options.private_members
                else:
                    keep = False
            else:
                if self.options.members is ALL and is_filtered_inherited_member(membername):
                    keep = False
                else:
                    # ignore undocumented members if :undoc-members: is not given
                    keep = has_doc or self.options.undoc_members

            if isinstance(obj, ObjectMember) and obj.skipped:
                # forcedly skipped member (ex. a module attribute not defined in __all__)
                keep = False

            # give the user a chance to decide whether this member
            # should be skipped
            if self.env.app:
                # let extensions preprocess docstrings
                try:
                    skip_user = self.env.app.emit_firstresult(
                        'autodoc-skip-member', self.objtype, membername, member,
                        not keep, self.options)
                    if skip_user is not None:
                        keep = not skip_user
                except Exception as exc:
                    logger.warning(__('autodoc: failed to determine %r to be documented, '
                                      'the following exception was raised:\n%s'),
                                   member, exc, type='autodoc')
                    keep = False

            if keep:
                ret.append((membername, member, isattr))

        return ret
```
### 17 - sphinx/ext/autodoc/__init__.py:

Start line: 1749, End line: 1784

```python
class DataDeclarationDocumenter(DataDocumenter):
    """
    Specialized Documenter subclass for data that cannot be imported
    because they are declared without initial value (refs: PEP-526).
    """
    objtype = 'datadecl'
    directivetype = 'data'
    member_order = 60

    # must be higher than AttributeDocumenter
    priority = 11

    @classmethod
    def can_document_member(cls, member: Any, membername: str, isattr: bool, parent: Any
                            ) -> bool:
        """This documents only INSTANCEATTR members."""
        return (isinstance(parent, ModuleDocumenter) and
                isattr and
                member is INSTANCEATTR)

    def import_object(self, raiseerror: bool = False) -> bool:
        """Never import anything."""
        # disguise as a data
        self.objtype = 'data'
        self.object = UNINITIALIZED_ATTR
        try:
            # import module to obtain type annotation
            self.parent = importlib.import_module(self.modname)
        except ImportError:
            pass

        return True

    def add_content(self, more_content: Any, no_docstring: bool = False) -> None:
        """Never try to get a docstring from the object."""
        super().add_content(more_content, no_docstring=True)
```
### 18 - sphinx/ext/autodoc/__init__.py:

Start line: 117, End line: 150

```python
def inherited_members_option(arg: Any) -> Union[object, Set[str]]:
    """Used to convert the :members: option to auto directives."""
    if arg is None:
        return 'object'
    else:
        return arg


def member_order_option(arg: Any) -> Optional[str]:
    """Used to convert the :members: option to auto directives."""
    if arg is None:
        return None
    elif arg in ('alphabetical', 'bysource', 'groupwise'):
        return arg
    else:
        raise ValueError(__('invalid value for member-order option: %s') % arg)


SUPPRESS = object()


def annotation_option(arg: Any) -> Any:
    if arg is None:
        # suppress showing the representation of the object
        return SUPPRESS
    else:
        return arg


def bool_option(arg: Any) -> bool:
    """Used to convert flag options to auto directives.  (Instead of
    directives.flag(), which returns None).
    """
    return True
```
### 19 - sphinx/ext/autodoc/__init__.py:

Start line: 2281, End line: 2301

```python
def get_documenters(app: Sphinx) -> Dict[str, "Type[Documenter]"]:
    """Returns registered Documenter classes"""
    warnings.warn("get_documenters() is deprecated.", RemovedInSphinx50Warning, stacklevel=2)
    return app.registry.documenters


def autodoc_attrgetter(app: Sphinx, obj: Any, name: str, *defargs: Any) -> Any:
    """Alternative getattr() for types"""
    for typ, func in app.registry.autodoc_attrgettrs.items():
        if isinstance(obj, typ):
            return func(obj, name, *defargs)

    return safe_getattr(obj, name, *defargs)


def migrate_autodoc_member_order(app: Sphinx, config: Config) -> None:
    if config.autodoc_member_order == 'alphabetic':
        # RemovedInSphinx50Warning
        logger.warning(__('autodoc_member_order now accepts "alphabetical" '
                          'instead of "alphabetic". Please update your setting.'))
        config.autodoc_member_order = 'alphabetical'  # type: ignore
```
### 22 - sphinx/ext/autodoc/__init__.py:

Start line: 569, End line: 581

```python
class Documenter:

    def get_sourcename(self) -> str:
        if (getattr(self.object, '__module__', None) and
                getattr(self.object, '__qualname__', None)):
            # Get the correct location of docstring from self.object
            # to support inherited methods
            fullname = '%s.%s' % (self.object.__module__, self.object.__qualname__)
        else:
            fullname = self.fullname

        if self.analyzer:
            return '%s:docstring of %s' % (self.analyzer.srcname, fullname)
        else:
            return 'docstring of %s' % fullname
```
### 24 - sphinx/ext/autodoc/__init__.py:

Start line: 1200, End line: 1220

```python
class DocstringSignatureMixin:

    def get_doc(self, encoding: str = None, ignore: int = None) -> List[List[str]]:
        if encoding is not None:
            warnings.warn("The 'encoding' argument to autodoc.%s.get_doc() is deprecated."
                          % self.__class__.__name__,
                          RemovedInSphinx40Warning, stacklevel=2)
        if self._new_docstrings is not None:
            return self._new_docstrings
        return super().get_doc(None, ignore)  # type: ignore

    def format_signature(self, **kwargs: Any) -> str:
        if self.args is None and self.config.autodoc_docstring_signature:  # type: ignore
            # only act if a signature is not explicitly given already, and if
            # the feature is enabled
            result = self._find_signature()
            if result is not None:
                self.args, self.retann = result
        sig = super().format_signature(**kwargs)  # type: ignore
        if self._signatures:
            return "\n".join([sig] + self._signatures)
        else:
            return sig
```
### 25 - sphinx/ext/autodoc/__init__.py:

Start line: 2227, End line: 2243

```python
class SlotsAttributeDocumenter(AttributeDocumenter):
    """
    Specialized Documenter subclass for attributes that cannot be imported
    because they are attributes in __slots__.
    """
    objtype = 'slotsattribute'
    directivetype = 'attribute'
    member_order = 60

    # must be higher than AttributeDocumenter
    priority = 11

    @classmethod
    def can_document_member(cls, member: Any, membername: str, isattr: bool, parent: Any
                            ) -> bool:
        """This documents only SLOTSATTR members."""
        return member is SLOTSATTR
```
### 27 - sphinx/ext/autodoc/__init__.py:

Start line: 987, End line: 1009

```python
class ModuleDocumenter(Documenter):

    def import_object(self, raiseerror: bool = False) -> bool:
        def is_valid_module_all(__all__: Any) -> bool:
            """Check the given *__all__* is valid for a module."""
            if (isinstance(__all__, (list, tuple)) and
                    all(isinstance(e, str) for e in __all__)):
                return True
            else:
                return False

        ret = super().import_object(raiseerror)

        if not self.options.ignore_module_all:
            __all__ = getattr(self.object, '__all__', None)
            if is_valid_module_all(__all__):
                # valid __all__ found. copy it to self.__all__
                self.__all__ = __all__
            elif __all__:
                # invalid __all__ found.
                logger.warning(__('__all__ should be a list of strings, not %r '
                                  '(in module %s) -- ignoring __all__') %
                               (__all__, self.fullname), type='autodoc')

        return ret
```
### 31 - sphinx/ext/autodoc/__init__.py:

Start line: 248, End line: 282

```python
# This class is used only in ``sphinx.ext.autodoc.directive``,
# But we define this class here to keep compatibility (see #4538)
class Options(dict):
    """A dict/attribute hybrid that returns None on nonexisting keys."""
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name.replace('_', '-')]
        except KeyError:
            return None


class ObjectMember(tuple):
    """A member of object.

    This is used for the result of `Documenter.get_object_members()` to
    represent each member of the object.

    .. Note::

       An instance of this class behaves as a tuple of (name, object)
       for compatibility to old Sphinx.  The behavior will be dropped
       in the future.  Therefore extensions should not use the tuple
       interface.
    """

    def __new__(cls, name: str, obj: Any, **kwargs: Any) -> Any:
        return super().__new__(cls, (name, obj))  # type: ignore

    def __init__(self, name: str, obj: Any, skipped: bool = False) -> None:
        self.__name__ = name
        self.object = obj
        self.skipped = skipped


ObjectMembers = Union[List[ObjectMember], List[Tuple[str, Any]]]
```
### 34 - sphinx/ext/autodoc/__init__.py:

Start line: 1054, End line: 1071

```python
class ModuleDocumenter(Documenter):

    def sort_members(self, documenters: List[Tuple["Documenter", bool]],
                     order: str) -> List[Tuple["Documenter", bool]]:
        if order == 'bysource' and self.__all__:
            # Sort alphabetically first (for members not listed on the __all__)
            documenters.sort(key=lambda e: e[0].name)

            # Sort by __all__
            def keyfunc(entry: Tuple[Documenter, bool]) -> int:
                name = entry[0].name.split('::')[1]
                if name in self.__all__:
                    return self.__all__.index(name)
                else:
                    return len(self.__all__)
            documenters.sort(key=keyfunc)

            return documenters
        else:
            return super().sort_members(documenters, order)
```
### 35 - sphinx/ext/autodoc/__init__.py:

Start line: 1011, End line: 1022

```python
class ModuleDocumenter(Documenter):

    def add_directive_header(self, sig: str) -> None:
        Documenter.add_directive_header(self, sig)

        sourcename = self.get_sourcename()

        # add some module-specific options
        if self.options.synopsis:
            self.add_line('   :synopsis: ' + self.options.synopsis, sourcename)
        if self.options.platform:
            self.add_line('   :platform: ' + self.options.platform, sourcename)
        if self.options.deprecated:
            self.add_line('   :deprecated:', sourcename)
```
### 48 - sphinx/ext/autodoc/__init__.py:

Start line: 372, End line: 407

```python
class Documenter:

    def parse_name(self) -> bool:
        """Determine what module to import and what attribute to document.

        Returns True and sets *self.modname*, *self.objpath*, *self.fullname*,
        *self.args* and *self.retann* if parsing and resolving was successful.
        """
        # first, parse the definition -- auto directives for classes and
        # functions can contain a signature which is then used instead of
        # an autogenerated one
        try:
            explicit_modname, path, base, args, retann = \
                py_ext_sig_re.match(self.name).groups()
        except AttributeError:
            logger.warning(__('invalid signature for auto%s (%r)') % (self.objtype, self.name),
                           type='autodoc')
            return False

        # support explicit module and class name separation via ::
        if explicit_modname is not None:
            modname = explicit_modname[:-2]
            parents = path.rstrip('.').split('.') if path else []
        else:
            modname = None
            parents = []

        with mock(self.config.autodoc_mock_imports):
            self.modname, self.objpath = self.resolve_name(modname, parents, path, base)

        if not self.modname:
            return False

        self.args = args
        self.retann = retann
        self.fullname = (self.modname or '') + \
                        ('.' + '.'.join(self.objpath) if self.objpath else '')
        return True
```
### 50 - sphinx/ext/autodoc/__init__.py:

Start line: 1074, End line: 1092

```python
class ModuleLevelDocumenter(Documenter):
    """
    Specialized Documenter subclass for objects on module level (functions,
    classes, data/constants).
    """
    def resolve_name(self, modname: str, parents: Any, path: str, base: Any
                     ) -> Tuple[str, List[str]]:
        if modname is None:
            if path:
                modname = path.rstrip('.')
            else:
                # if documenting a toplevel object without explicit module,
                # it can be contained in another auto directive ...
                modname = self.env.temp_data.get('autodoc:module')
                # ... or in the scope of a module directive
                if not modname:
                    modname = self.env.ref_context.get('py:module')
                # ... else, it stays None, which means invalid
        return modname, parents + [base]
```
### 53 - sphinx/ext/autodoc/__init__.py:

Start line: 1095, End line: 1125

```python
class ClassLevelDocumenter(Documenter):
    """
    Specialized Documenter subclass for objects on class level (methods,
    attributes).
    """
    def resolve_name(self, modname: str, parents: Any, path: str, base: Any
                     ) -> Tuple[str, List[str]]:
        if modname is None:
            if path:
                mod_cls = path.rstrip('.')
            else:
                mod_cls = None
                # if documenting a class-level object without path,
                # there must be a current class, either from a parent
                # auto directive ...
                mod_cls = self.env.temp_data.get('autodoc:class')
                # ... or from a class directive
                if mod_cls is None:
                    mod_cls = self.env.ref_context.get('py:class')
                # ... if still None, there's no way to know
                if mod_cls is None:
                    return None, []
            modname, sep, cls = mod_cls.rpartition('.')
            parents = [cls]
            # if the module name is still missing, get it like above
            if not modname:
                modname = self.env.temp_data.get('autodoc:module')
            if not modname:
                modname = self.env.ref_context.get('py:module')
            # ... else, it stays None, which means invalid
        return modname, parents + [base]
```
### 56 - sphinx/ext/autodoc/__init__.py:

Start line: 1357, End line: 1665

```python
class DecoratorDocumenter(FunctionDocumenter):
    """
    Specialized Documenter subclass for decorator functions.
    """
    objtype = 'decorator'

    # must be lower than FunctionDocumenter
    priority = -1

    def format_args(self, **kwargs: Any) -> Any:
        args = super().format_args(**kwargs)
        if ',' in args:
            return args
        else:
            return None


# Types which have confusing metaclass signatures it would be best not to show.
# These are listed by name, rather than storing the objects themselves, to avoid
# needing to import the modules.
_METACLASS_CALL_BLACKLIST = [
    'enum.EnumMeta.__call__',
]


# Types whose __new__ signature is a pass-thru.
_CLASS_NEW_BLACKLIST = [
    'typing.Generic.__new__',
]


class ClassDocumenter(DocstringSignatureMixin, ModuleLevelDocumenter):
```
### 58 - sphinx/ext/autodoc/__init__.py:

Start line: 1223, End line: 1238

```python
class DocstringStripSignatureMixin(DocstringSignatureMixin):
    """
    Mixin for AttributeDocumenter to provide the
    feature of stripping any function signature from the docstring.
    """
    def format_signature(self, **kwargs: Any) -> str:
        if self.args is None and self.config.autodoc_docstring_signature:  # type: ignore
            # only act if a signature is not explicitly given already, and if
            # the feature is enabled
            result = self._find_signature()
            if result is not None:
                # Discarding _args is a only difference with
                # DocstringSignatureMixin.format_signature.
                # Documenter.format_signature use self.args value to format.
                _args, self.retann = result
        return super().format_signature(**kwargs)
```
### 64 - sphinx/ext/autodoc/__init__.py:

Start line: 430, End line: 478

```python
class Documenter:

    def get_real_modname(self) -> str:
        """Get the real module name of an object to document.

        It can differ from the name of the module through which the object was
        imported.
        """
        return self.get_attr(self.object, '__module__', None) or self.modname

    def check_module(self) -> bool:
        """Check if *self.object* is really defined in the module given by
        *self.modname*.
        """
        if self.options.imported_members:
            return True

        subject = inspect.unpartial(self.object)
        modname = self.get_attr(subject, '__module__', None)
        if modname and modname != self.modname:
            return False
        return True

    def format_args(self, **kwargs: Any) -> str:
        """Format the argument signature of *self.object*.

        Should return None if the object does not have a signature.
        """
        return None

    def format_name(self) -> str:
        """Format the name of *self.object*.

        This normally should be something that can be parsed by the generated
        directive, but doesn't need to be (Sphinx will display it unparsed
        then).
        """
        # normally the name doesn't contain the module (except for module
        # directives of course)
        return '.'.join(self.objpath) or self.modname

    def _call_format_args(self, **kwargs: Any) -> str:
        if kwargs:
            try:
                return self.format_args(**kwargs)
            except TypeError:
                # avoid chaining exceptions, by putting nothing here
                pass

        # retry without arguments for old documenters
        return self.format_args()
```
### 68 - sphinx/ext/autodoc/__init__.py:

Start line: 2152, End line: 2180

```python
class PropertyDocumenter(DocstringStripSignatureMixin, ClassLevelDocumenter):  # type: ignore
    """
    Specialized Documenter subclass for properties.
    """
    objtype = 'property'
    directivetype = 'method'
    member_order = 60

    # before AttributeDocumenter
    priority = AttributeDocumenter.priority + 1

    @classmethod
    def can_document_member(cls, member: Any, membername: str, isattr: bool, parent: Any
                            ) -> bool:
        return inspect.isproperty(member) and isinstance(parent, ClassDocumenter)

    def document_members(self, all_members: bool = False) -> None:
        pass

    def get_real_modname(self) -> str:
        return self.get_attr(self.parent or self.object, '__module__', None) \
            or self.modname

    def add_directive_header(self, sig: str) -> None:
        super().add_directive_header(sig)
        sourcename = self.get_sourcename()
        if inspect.isabstractmethod(self.object):
            self.add_line('   :abstractmethod:', sourcename)
        self.add_line('   :property:', sourcename)
```
### 74 - sphinx/ext/autodoc/__init__.py:

Start line: 2245, End line: 2264

```python
class SlotsAttributeDocumenter(AttributeDocumenter):

    def import_object(self, raiseerror: bool = False) -> bool:
        """Never import anything."""
        # disguise as an attribute
        self.objtype = 'attribute'
        self._datadescriptor = True

        with mock(self.config.autodoc_mock_imports):
            try:
                ret = import_object(self.modname, self.objpath[:-1], 'class',
                                    attrgetter=self.get_attr,
                                    warningiserror=self.config.autodoc_warningiserror)
                self.module, _, _, self.parent = ret
                return True
            except ImportError as exc:
                if raiseerror:
                    raise
                else:
                    logger.warning(exc.args[0], type='autodoc', subtype='import_object')
                    self.env.note_reread()
                    return False
```
