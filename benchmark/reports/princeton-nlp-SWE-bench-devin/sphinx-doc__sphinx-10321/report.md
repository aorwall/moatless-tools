# sphinx-doc__sphinx-10321

| **sphinx-doc/sphinx** | `4689ec6de1241077552458ed38927c0e713bb85d` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 6480 |
| **Any found context length** | 6480 |
| **Avg pos** | 5.0 |
| **Min pos** | 5 |
| **Max pos** | 5 |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sphinx/ext/autodoc/preserve_defaults.py b/sphinx/ext/autodoc/preserve_defaults.py
--- a/sphinx/ext/autodoc/preserve_defaults.py
+++ b/sphinx/ext/autodoc/preserve_defaults.py
@@ -79,7 +79,11 @@ def update_defvalue(app: Sphinx, obj: Any, bound_method: bool) -> None:
             kw_defaults = list(function.args.kw_defaults)
             parameters = list(sig.parameters.values())
             for i, param in enumerate(parameters):
-                if param.default is not param.empty:
+                if param.default is param.empty:
+                    if param.kind == param.KEYWORD_ONLY:
+                        # Consume kw_defaults for kwonly args
+                        kw_defaults.pop(0)
+                else:
                     if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD):
                         default = defaults.pop(0)
                         value = get_default_value(lines, default)

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sphinx/ext/autodoc/preserve_defaults.py | 82 | 82 | 5 | 1 | 6480


## Problem Statement

```
autodoc_preserve_defaults=True does not work for mixture of keyword only arguments with/without defaults
### Describe the bug

If I understand [PEP 0570](https://peps.python.org/pep-0570/) correctly, the following is a valid signature of a class method:

\`\`\`
class Thing:
    def __init__(
            self, 
            kw_or_pos_without_default, 
            kw_or_pos_with_default=None, 
            *,
            kw_without_default,
            kw_with_default="Foo"
    ):
        pass
\`\`\`

When documenting this with _autodoc_ and `autodoc_preserve_defaults=True`, `sphinx.ext.autodoc.preserve_defaults.update_defvalue` generates a `DefaultValue` with `name=None` for the `kw_with_default` arguments. This later raises an exception in `sphinx.util.inspect.object_description` since the `DefaultValue.__repr__` dunder method now returns `None` instead of a string.

Basically what happens is that _ast_ generates a `None` value in the `kw_defaults` of the `arguments` since the first keyword argument is required, but `update_defvalue` simply ignores that argument because the `default` is empty. This leaves the `None` in the `kw_defaults` to be picked up when the keyword argument _with_ default value is processed -- instead of the actual default.
This can't be resolved by the `unparse` call which therefore simply returns `None`, which ends up as the `name` of the `DefaultValue`.

Imo this could simply be resolved by `pop`ing the corresponding `None` from the `kw_defaults` if a `KW_ONLY` parameter with empty `default` is encountered.





### How to Reproduce

Create a module with contents 

\`\`\`
class Thing:
    def __init__(
            self, 
            kw_or_pos_without_default, 
            kw_or_pos_with_default=None, 
            *,
            kw_without_default,
            kw_with_default="Foo"
    ):
        pass

\`\`\`

and auto-document while setting  `autodoc_preserve_defaults=True` in your `conf.py`

Make sure sphinx tries to document all parameters, (since it's a `__init__` method, they will be documented when the _autodoc_ directive has `:undoc-members:`, if you try the same with a module level method you need to document the parameters)

[test.zip](https://github.com/sphinx-doc/sphinx/files/8253301/test.zip)


### Expected behavior

The correct default value should be documented. The Warning Message also is pretty worthless (probably the value should not be
formatted with a simple `%s` but instead with a `%r`?)

### Your project

https://github.com/sphinx-doc/sphinx/files/8253301/test.zip

### OS

Any

### Python version

Tested with versions > 3.8

### Sphinx version

4.4.0

### Sphinx extensions

sphinx.ext.autodoc

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 sphinx/ext/autodoc/preserve_defaults.py** | 1 | 26| 139 | 139 | 868 | 
| 2 | 2 sphinx/ext/autodoc/__init__.py | 2859 | 2904| 557 | 696 | 25334 | 
| 3 | 2 sphinx/ext/autodoc/__init__.py | 2125 | 2335| 1913 | 2609 | 25334 | 
| 4 | 2 sphinx/ext/autodoc/__init__.py | 1437 | 1817| 3418 | 6027 | 25334 | 
| **-> 5 <-** | **2 sphinx/ext/autodoc/preserve_defaults.py** | 62 | 112| 453 | 6480 | 25334 | 
| 6 | 2 sphinx/ext/autodoc/__init__.py | 1 | 102| 778 | 7258 | 25334 | 
| 7 | 2 sphinx/ext/autodoc/__init__.py | 1279 | 1403| 1054 | 8312 | 25334 | 
| 8 | 2 sphinx/ext/autodoc/__init__.py | 2638 | 2661| 229 | 8541 | 25334 | 
| 9 | 2 sphinx/ext/autodoc/__init__.py | 1083 | 1107| 209 | 8750 | 25334 | 
| 10 | 3 sphinx/ext/autodoc/directive.py | 1 | 39| 314 | 9064 | 26742 | 
| 11 | 4 sphinx/ext/autodoc/deprecated.py | 24 | 39| 140 | 9204 | 27654 | 
| 12 | 4 sphinx/ext/autodoc/__init__.py | 1029 | 1041| 124 | 9328 | 27654 | 
| 13 | 5 sphinx/util/inspect.py | 457 | 483| 176 | 9504 | 34163 | 
| 14 | 5 sphinx/ext/autodoc/__init__.py | 1950 | 1988| 307 | 9811 | 34163 | 
| 15 | 5 sphinx/ext/autodoc/__init__.py | 708 | 818| 907 | 10718 | 34163 | 
| 16 | 5 sphinx/ext/autodoc/__init__.py | 2067 | 2104| 320 | 11038 | 34163 | 
| 17 | 5 sphinx/ext/autodoc/__init__.py | 2706 | 2731| 316 | 11354 | 34163 | 
| 18 | 5 sphinx/ext/autodoc/__init__.py | 1820 | 1855| 245 | 11599 | 34163 | 
| 19 | 5 sphinx/ext/autodoc/__init__.py | 294 | 370| 751 | 12350 | 34163 | 
| 20 | 5 sphinx/ext/autodoc/__init__.py | 2734 | 2808| 624 | 12974 | 34163 | 
| 21 | 5 sphinx/ext/autodoc/deprecated.py | 1 | 21| 165 | 13139 | 34163 | 
| 22 | 5 sphinx/ext/autodoc/__init__.py | 1406 | 1817| 188 | 13327 | 34163 | 
| 23 | 6 sphinx/ext/autodoc/mock.py | 65 | 90| 217 | 13544 | 35532 | 
| 24 | 6 sphinx/ext/autodoc/deprecated.py | 106 | 119| 114 | 13658 | 35532 | 
| 25 | 6 sphinx/ext/autodoc/__init__.py | 2689 | 2704| 182 | 13840 | 35532 | 
| 26 | 6 sphinx/ext/autodoc/__init__.py | 2829 | 2859| 363 | 14203 | 35532 | 
| 27 | 6 sphinx/ext/autodoc/__init__.py | 582 | 594| 139 | 14342 | 35532 | 
| 28 | 6 sphinx/ext/autodoc/deprecated.py | 70 | 85| 139 | 14481 | 35532 | 
| 29 | 6 sphinx/ext/autodoc/__init__.py | 2504 | 2521| 134 | 14615 | 35532 | 
| 30 | 6 sphinx/ext/autodoc/__init__.py | 1150 | 1180| 287 | 14902 | 35532 | 
| 31 | 6 sphinx/ext/autodoc/__init__.py | 1242 | 1258| 184 | 15086 | 35532 | 
| 32 | 6 sphinx/ext/autodoc/__init__.py | 1109 | 1126| 182 | 15268 | 35532 | 
| 33 | 6 sphinx/ext/autodoc/__init__.py | 2338 | 2366| 248 | 15516 | 35532 | 
| 34 | 6 sphinx/ext/autodoc/__init__.py | 2398 | 2421| 210 | 15726 | 35532 | 
| 35 | 6 sphinx/ext/autodoc/__init__.py | 2040 | 2065| 268 | 15994 | 35532 | 
| 36 | 6 sphinx/ext/autodoc/__init__.py | 1129 | 1147| 179 | 16173 | 35532 | 
| 37 | 6 sphinx/ext/autodoc/__init__.py | 1991 | 2038| 365 | 16538 | 35532 | 
| 38 | 7 sphinx/ext/autodoc/importer.py | 69 | 139| 649 | 17187 | 37945 | 
| 39 | 7 sphinx/ext/autodoc/deprecated.py | 57 | 67| 108 | 17295 | 37945 | 
| 40 | 8 sphinx/ext/autosummary/generate.py | 81 | 93| 163 | 17458 | 43562 | 
| 41 | 8 sphinx/ext/autodoc/deprecated.py | 42 | 54| 113 | 17571 | 43562 | 
| 42 | 8 sphinx/ext/autodoc/__init__.py | 2663 | 2687| 295 | 17866 | 43562 | 
| 43 | 8 sphinx/ext/autodoc/__init__.py | 1043 | 1054| 126 | 17992 | 43562 | 
| 44 | 9 sphinx/application.py | 1108 | 1129| 263 | 18255 | 55371 | 
| 45 | 9 sphinx/ext/autodoc/__init__.py | 2616 | 2636| 231 | 18486 | 55371 | 
| 46 | 9 sphinx/ext/autodoc/__init__.py | 443 | 491| 359 | 18845 | 55371 | 
| 47 | 9 sphinx/ext/autosummary/generate.py | 169 | 188| 176 | 19021 | 55371 | 
| 48 | **9 sphinx/ext/autodoc/preserve_defaults.py** | 29 | 59| 274 | 19295 | 55371 | 
| 49 | 9 sphinx/ext/autosummary/generate.py | 237 | 272| 343 | 19638 | 55371 | 
| 50 | 9 sphinx/ext/autodoc/mock.py | 1 | 62| 468 | 20106 | 55371 | 
| 51 | 10 doc/conf.py | 165 | 187| 269 | 20375 | 57092 | 
| 52 | 10 sphinx/ext/autodoc/__init__.py | 2523 | 2556| 289 | 20664 | 57092 | 
| 53 | 11 sphinx/ext/apidoc.py | 298 | 363| 752 | 21416 | 61294 | 
| 54 | 11 sphinx/ext/autodoc/importer.py | 1 | 32| 219 | 21635 | 61294 | 
| 55 | 11 sphinx/ext/autodoc/__init__.py | 372 | 381| 121 | 21756 | 61294 | 
| 56 | 11 sphinx/ext/autodoc/__init__.py | 984 | 1027| 387 | 22143 | 61294 | 
| 57 | 11 sphinx/ext/autodoc/__init__.py | 890 | 981| 857 | 23000 | 61294 | 
| 58 | 11 sphinx/ext/autodoc/deprecated.py | 88 | 103| 133 | 23133 | 61294 | 
| 59 | 11 sphinx/ext/autodoc/__init__.py | 251 | 291| 327 | 23460 | 61294 | 
| 60 | 11 sphinx/ext/autodoc/directive.py | 117 | 167| 453 | 23913 | 61294 | 
| 61 | 12 sphinx/deprecation.py | 31 | 56| 236 | 24149 | 61956 | 
| 62 | 12 sphinx/ext/autodoc/__init__.py | 1881 | 1901| 169 | 24318 | 61956 | 
| 63 | 12 sphinx/ext/autodoc/__init__.py | 2559 | 2593| 291 | 24609 | 61956 | 
| 64 | 12 sphinx/ext/autodoc/__init__.py | 2595 | 2614| 224 | 24833 | 61956 | 
| 65 | 12 sphinx/ext/autodoc/__init__.py | 1928 | 1947| 195 | 25028 | 61956 | 
| 66 | 12 sphinx/ext/autosummary/generate.py | 380 | 477| 833 | 25861 | 61956 | 
| 67 | 12 sphinx/ext/autosummary/generate.py | 495 | 517| 254 | 26115 | 61956 | 
| 68 | 13 doc/development/tutorials/examples/autodoc_intenum.py | 30 | 55| 199 | 26314 | 62344 | 
| 69 | 14 sphinx/ext/autodoc/typehints.py | 127 | 182| 460 | 26774 | 63785 | 
| 70 | 15 sphinx/ext/napoleon/docstring.py | 686 | 727| 418 | 27192 | 74774 | 
| 71 | 15 sphinx/ext/autodoc/__init__.py | 383 | 418| 317 | 27509 | 74774 | 
| 72 | 16 sphinx/directives/other.py | 1 | 31| 240 | 27749 | 77863 | 
| 73 | 16 sphinx/ext/autodoc/directive.py | 42 | 71| 250 | 27999 | 77863 | 
| 74 | 16 sphinx/ext/autodoc/__init__.py | 2107 | 2335| 136 | 28135 | 77863 | 
| 75 | 17 sphinx/cmd/quickstart.py | 1 | 111| 767 | 28902 | 83390 | 
| 76 | 17 sphinx/ext/autosummary/generate.py | 299 | 312| 202 | 29104 | 83390 | 
| 77 | 17 sphinx/ext/autodoc/__init__.py | 1858 | 1878| 162 | 29266 | 83390 | 
| 78 | 18 sphinx/domains/python.py | 1032 | 1050| 140 | 29406 | 95988 | 
| 79 | 18 sphinx/ext/autodoc/__init__.py | 105 | 153| 342 | 29748 | 95988 | 
| 80 | 19 doc/usage/extensions/example_google.py | 273 | 292| 120 | 29868 | 98073 | 
| 81 | 19 sphinx/ext/autodoc/directive.py | 74 | 97| 255 | 30123 | 98073 | 
| 82 | 19 sphinx/ext/autodoc/typehints.py | 1 | 34| 256 | 30379 | 98073 | 
| 83 | 19 sphinx/deprecation.py | 1 | 28| 166 | 30545 | 98073 | 
| 84 | 19 doc/development/tutorials/examples/autodoc_intenum.py | 1 | 28| 198 | 30743 | 98073 | 
| 85 | 20 sphinx/transforms/__init__.py | 93 | 111| 178 | 30921 | 101354 | 
| 86 | 21 setup.py | 174 | 251| 651 | 31572 | 103128 | 
| 87 | 22 doc/usage/extensions/example_numpy.py | 332 | 352| 120 | 31692 | 105208 | 
| 88 | 22 sphinx/ext/autodoc/__init__.py | 1904 | 1926| 197 | 31889 | 105208 | 
| 89 | 22 sphinx/ext/autodoc/__init__.py | 2811 | 2826| 126 | 32015 | 105208 | 
| 90 | 23 sphinx/directives/patches.py | 1 | 33| 256 | 32271 | 107113 | 
| 91 | 24 sphinx/directives/__init__.py | 41 | 63| 187 | 32458 | 109258 | 
| 92 | 24 sphinx/ext/autodoc/__init__.py | 1183 | 1240| 434 | 32892 | 109258 | 
| 93 | 24 sphinx/ext/autodoc/__init__.py | 820 | 863| 451 | 33343 | 109258 | 
| 94 | 24 sphinx/ext/apidoc.py | 364 | 392| 374 | 33717 | 109258 | 
| 95 | 24 doc/usage/extensions/example_google.py | 176 | 255| 563 | 34280 | 109258 | 
| 96 | 25 sphinx/domains/c.py | 3614 | 3926| 2931 | 37211 | 141680 | 


## Patch

```diff
diff --git a/sphinx/ext/autodoc/preserve_defaults.py b/sphinx/ext/autodoc/preserve_defaults.py
--- a/sphinx/ext/autodoc/preserve_defaults.py
+++ b/sphinx/ext/autodoc/preserve_defaults.py
@@ -79,7 +79,11 @@ def update_defvalue(app: Sphinx, obj: Any, bound_method: bool) -> None:
             kw_defaults = list(function.args.kw_defaults)
             parameters = list(sig.parameters.values())
             for i, param in enumerate(parameters):
-                if param.default is not param.empty:
+                if param.default is param.empty:
+                    if param.kind == param.KEYWORD_ONLY:
+                        # Consume kw_defaults for kwonly args
+                        kw_defaults.pop(0)
+                else:
                     if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD):
                         default = defaults.pop(0)
                         value = get_default_value(lines, default)

```

## Test Patch

```diff
diff --git a/tests/roots/test-ext-autodoc/target/preserve_defaults.py b/tests/roots/test-ext-autodoc/target/preserve_defaults.py
--- a/tests/roots/test-ext-autodoc/target/preserve_defaults.py
+++ b/tests/roots/test-ext-autodoc/target/preserve_defaults.py
@@ -8,7 +8,10 @@
 def foo(name: str = CONSTANT,
         sentinel: Any = SENTINEL,
         now: datetime = datetime.now(),
-        color: int = 0xFFFFFF) -> None:
+        color: int = 0xFFFFFF,
+        *,
+        kwarg1,
+        kwarg2 = 0xFFFFFF) -> None:
     """docstring"""
 
 
@@ -16,5 +19,6 @@ class Class:
     """docstring"""
 
     def meth(self, name: str = CONSTANT, sentinel: Any = SENTINEL,
-             now: datetime = datetime.now(), color: int = 0xFFFFFF) -> None:
+             now: datetime = datetime.now(), color: int = 0xFFFFFF,
+             *, kwarg1, kwarg2 = 0xFFFFFF) -> None:
         """docstring"""
diff --git a/tests/test_ext_autodoc_preserve_defaults.py b/tests/test_ext_autodoc_preserve_defaults.py
--- a/tests/test_ext_autodoc_preserve_defaults.py
+++ b/tests/test_ext_autodoc_preserve_defaults.py
@@ -29,14 +29,16 @@ def test_preserve_defaults(app):
         '',
         '',
         '   .. py:method:: Class.meth(name: str = CONSTANT, sentinel: ~typing.Any = '
-        'SENTINEL, now: ~datetime.datetime = datetime.now(), color: int = %s) -> None' % color,
+        'SENTINEL, now: ~datetime.datetime = datetime.now(), color: int = %s, *, '
+        'kwarg1, kwarg2=%s) -> None' % (color, color),
         '      :module: target.preserve_defaults',
         '',
         '      docstring',
         '',
         '',
         '.. py:function:: foo(name: str = CONSTANT, sentinel: ~typing.Any = SENTINEL, '
-        'now: ~datetime.datetime = datetime.now(), color: int = %s) -> None' % color,
+        'now: ~datetime.datetime = datetime.now(), color: int = %s, *, kwarg1, '
+        'kwarg2=%s) -> None' % (color, color),
         '   :module: target.preserve_defaults',
         '',
         '   docstring',

```


## Code snippets

### 1 - sphinx/ext/autodoc/preserve_defaults.py:

Start line: 1, End line: 26

```python
"""Preserve function defaults.

Preserve the default argument values of function signatures in source code
and keep them not evaluated for readability.
"""

import ast
import inspect
import sys
from typing import Any, Dict, List, Optional

from sphinx.application import Sphinx
from sphinx.locale import __
from sphinx.pycode.ast import parse as ast_parse
from sphinx.pycode.ast import unparse as ast_unparse
from sphinx.util import logging

logger = logging.getLogger(__name__)


class DefaultValue:
    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return self.name
```
### 2 - sphinx/ext/autodoc/__init__.py:

Start line: 2859, End line: 2904

```python
# NOQA


def setup(app: Sphinx) -> Dict[str, Any]:
    app.add_autodocumenter(ModuleDocumenter)
    app.add_autodocumenter(ClassDocumenter)
    app.add_autodocumenter(ExceptionDocumenter)
    app.add_autodocumenter(DataDocumenter)
    app.add_autodocumenter(NewTypeDataDocumenter)
    app.add_autodocumenter(FunctionDocumenter)
    app.add_autodocumenter(DecoratorDocumenter)
    app.add_autodocumenter(MethodDocumenter)
    app.add_autodocumenter(AttributeDocumenter)
    app.add_autodocumenter(PropertyDocumenter)
    app.add_autodocumenter(NewTypeAttributeDocumenter)

    app.add_config_value('autoclass_content', 'class', True, ENUM('both', 'class', 'init'))
    app.add_config_value('autodoc_member_order', 'alphabetical', True,
                         ENUM('alphabetic', 'alphabetical', 'bysource', 'groupwise'))
    app.add_config_value('autodoc_class_signature', 'mixed', True, ENUM('mixed', 'separated'))
    app.add_config_value('autodoc_default_options', {}, True)
    app.add_config_value('autodoc_docstring_signature', True, True)
    app.add_config_value('autodoc_mock_imports', [], True)
    app.add_config_value('autodoc_typehints', "signature", True,
                         ENUM("signature", "description", "none", "both"))
    app.add_config_value('autodoc_typehints_description_target', 'all', True,
                         ENUM('all', 'documented'))
    app.add_config_value('autodoc_type_aliases', {}, True)
    app.add_config_value('autodoc_typehints_format', "short", 'env',
                         ENUM("fully-qualified", "short"))
    app.add_config_value('autodoc_warningiserror', True, True)
    app.add_config_value('autodoc_inherit_docstrings', True, True)
    app.add_event('autodoc-before-process-signature')
    app.add_event('autodoc-process-docstring')
    app.add_event('autodoc-process-signature')
    app.add_event('autodoc-skip-member')
    app.add_event('autodoc-process-bases')

    app.connect('config-inited', migrate_autodoc_member_order, priority=800)

    app.setup_extension('sphinx.ext.autodoc.preserve_defaults')
    app.setup_extension('sphinx.ext.autodoc.type_comment')
    app.setup_extension('sphinx.ext.autodoc.typehints')

    return {'version': sphinx.__display_version__, 'parallel_read_safe': True}
```
### 3 - sphinx/ext/autodoc/__init__.py:

Start line: 2125, End line: 2335

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
        return inspect.isroutine(member) and not isinstance(parent, ModuleDocumenter)

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
        if self.config.autodoc_typehints_format == "short":
            kwargs.setdefault('unqualified_typehints', True)

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
        if inspect.iscoroutinefunction(obj) or inspect.isasyncgenfunction(obj):
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
        if self.config.autodoc_typehints_format == "short":
            kwargs.setdefault('unqualified_typehints', True)

        sigs = []
        if (self.analyzer and
                '.'.join(self.objpath) in self.analyzer.overloads and
                self.config.autodoc_typehints != 'none'):
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
                    dispatchmeth = self.annotate_to_first_argument(func, typ)
                    if dispatchmeth:
                        documenter = MethodDocumenter(self.directive, '')
                        documenter.parent = self.parent
                        documenter.object = dispatchmeth
                        documenter.objpath = [None]
                        sigs.append(documenter.format_signature())
        if overloaded:
            if inspect.isstaticmethod(self.object, cls=self.parent, name=self.object_name):
                actual = inspect.signature(self.object, bound_method=False,
                                           type_aliases=self.config.autodoc_type_aliases)
            else:
                actual = inspect.signature(self.object, bound_method=True,
                                           type_aliases=self.config.autodoc_type_aliases)

            __globals__ = safe_getattr(self.object, '__globals__', {})
            for overload in self.analyzer.overloads.get('.'.join(self.objpath)):
                overload = self.merge_default_value(actual, overload)
                overload = evaluate_signature(overload, __globals__,
                                              self.config.autodoc_type_aliases)

                if not inspect.isstaticmethod(self.object, cls=self.parent,
                                              name=self.object_name):
                    parameters = list(overload.parameters.values())
                    overload = overload.replace(parameters=parameters[1:])
                sig = stringify_signature(overload, **kwargs)
                sigs.append(sig)

        return "\n".join(sigs)

    def merge_default_value(self, actual: Signature, overload: Signature) -> Signature:
        """Merge default values of actual implementation to the overload variants."""
        parameters = list(overload.parameters.values())
        for i, param in enumerate(parameters):
            actual_param = actual.parameters.get(param.name)
            if actual_param and param.default == '...':
                parameters[i] = param.replace(default=actual_param.default)

        return overload.replace(parameters=parameters)

    def annotate_to_first_argument(self, func: Callable, typ: Type) -> Optional[Callable]:
        """Annotate type hint to the first argument of function if needed."""
        try:
            sig = inspect.signature(func, type_aliases=self.config.autodoc_type_aliases)
        except TypeError as exc:
            logger.warning(__("Failed to get a method signature for %s: %s"),
                           self.fullname, exc)
            return None
        except ValueError:
            return None

        if len(sig.parameters) == 1:
            return None

        def dummy():
            pass

        params = list(sig.parameters.values())
        if params[1].annotation is Parameter.empty:
            params[1] = params[1].replace(annotation=typ)
            try:
                dummy.__signature__ = sig.replace(parameters=params)  # type: ignore
                return dummy
            except (AttributeError, TypeError):
                # failed to update signature (ex. built-in or extension types)
                return None
        else:
            return None

    def get_doc(self, ignore: int = None) -> Optional[List[List[str]]]:
        if self._new_docstrings is not None:
            # docstring already returned previously, then modified by
            # `DocstringSignatureMixin`.  Just return the previously-computed
            # result, so that we don't lose the processing done by
            # `DocstringSignatureMixin`.
            return self._new_docstrings
        if self.objpath[-1] == '__init__':
            docstring = getdoc(self.object, self.get_attr,
                               self.config.autodoc_inherit_docstrings,
                               self.parent, self.object_name)
            if (docstring is not None and
                (docstring == object.__init__.__doc__ or  # for pypy
                 docstring.strip() == object.__init__.__doc__)):  # for !pypy
                docstring = None
            if docstring:
                tab_width = self.directive.state.document.settings.tab_width
                return [prepare_docstring(docstring, tabsize=tab_width)]
            else:
                return []
        elif self.objpath[-1] == '__new__':
            docstring = getdoc(self.object, self.get_attr,
                               self.config.autodoc_inherit_docstrings,
                               self.parent, self.object_name)
            if (docstring is not None and
                (docstring == object.__new__.__doc__ or  # for pypy
                 docstring.strip() == object.__new__.__doc__)):  # for !pypy
                docstring = None
            if docstring:
                tab_width = self.directive.state.document.settings.tab_width
                return [prepare_docstring(docstring, tabsize=tab_width)]
            else:
                return []
        else:
            return super().get_doc()
```
### 4 - sphinx/ext/autodoc/__init__.py:

Start line: 1437, End line: 1817

```python
class ClassDocumenter(DocstringSignatureMixin, ModuleLevelDocumenter):  # type: ignore
    """
    Specialized Documenter subclass for classes.
    """
    objtype = 'class'
    member_order = 20
    option_spec: OptionSpec = {
        'members': members_option, 'undoc-members': bool_option,
        'noindex': bool_option, 'inherited-members': inherited_members_option,
        'show-inheritance': bool_option, 'member-order': member_order_option,
        'exclude-members': exclude_members_option,
        'private-members': members_option, 'special-members': members_option,
        'class-doc-from': class_doc_from_option,
    }

    _signature_class: Any = None
    _signature_method_name: str = None

    def __init__(self, *args: Any) -> None:
        super().__init__(*args)

        if self.config.autodoc_class_signature == 'separated':
            self.options = self.options.copy()

            # show __init__() method
            if self.options.special_members is None:
                self.options['special-members'] = ['__new__', '__init__']
            else:
                self.options.special_members.append('__new__')
                self.options.special_members.append('__init__')

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
        if self.config.autodoc_typehints_format == "short":
            kwargs.setdefault('unqualified_typehints', True)

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

    def _find_signature(self) -> Tuple[str, str]:
        result = super()._find_signature()
        if result is not None:
            # Strip a return value from signature of constructor in docstring (first entry)
            result = (result[0], None)

        for i, sig in enumerate(self._signatures):
            if sig.endswith(' -> None'):
                # Strip a return value from signatures of constructor in docstring (subsequent
                # entries)
                self._signatures[i] = sig[:-8]

        return result

    def format_signature(self, **kwargs: Any) -> str:
        if self.doc_as_attr:
            return ''
        if self.config.autodoc_class_signature == 'separated':
            # do not show signatures
            return ''

        if self.config.autodoc_typehints_format == "short":
            kwargs.setdefault('unqualified_typehints', True)

        sig = super().format_signature()
        sigs = []

        overloads = self.get_overloaded_signatures()
        if overloads and self.config.autodoc_typehints != 'none':
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
                    analyzer.analyze()
                    qualname = '.'.join([cls.__qualname__, self._signature_method_name])
                    if qualname in analyzer.overloads:
                        return analyzer.overloads.get(qualname)
                    elif qualname in analyzer.tagorder:
                        # the constructor is defined in the class, but not overridden.
                        return []
                except PycodeError:
                    pass

        return []

    def get_canonical_fullname(self) -> Optional[str]:
        __modname__ = safe_getattr(self.object, '__module__', self.modname)
        __qualname__ = safe_getattr(self.object, '__qualname__', None)
        if __qualname__ is None:
            __qualname__ = safe_getattr(self.object, '__name__', None)
        if __qualname__ and '<locals>' in __qualname__:
            # No valid qualname found if the object is defined as locals
            __qualname__ = None

        if __modname__ and __qualname__:
            return '.'.join([__modname__, __qualname__])
        else:
            return None

    def add_directive_header(self, sig: str) -> None:
        sourcename = self.get_sourcename()

        if self.doc_as_attr:
            self.directivetype = 'attribute'
        super().add_directive_header(sig)

        if self.analyzer and '.'.join(self.objpath) in self.analyzer.finals:
            self.add_line('   :final:', sourcename)

        canonical_fullname = self.get_canonical_fullname()
        if not self.doc_as_attr and canonical_fullname and self.fullname != canonical_fullname:
            self.add_line('   :canonical: %s' % canonical_fullname, sourcename)

        # add inheritance info, if wanted
        if not self.doc_as_attr and self.options.show_inheritance:
            if inspect.getorigbases(self.object):
                # A subclass of generic types
                # refs: PEP-560 <https://peps.python.org/pep-0560/>
                bases = list(self.object.__orig_bases__)
            elif hasattr(self.object, '__bases__') and len(self.object.__bases__):
                # A normal class
                bases = list(self.object.__bases__)
            else:
                bases = []

            self.env.events.emit('autodoc-process-bases',
                                 self.fullname, self.object, self.options, bases)

            if self.config.autodoc_typehints_format == "short":
                base_classes = [restify(cls, "smart") for cls in bases]
            else:
                base_classes = [restify(cls) for cls in bases]

            sourcename = self.get_sourcename()
            self.add_line('', sourcename)
            self.add_line('   ' + _('Bases: %s') % ', '.join(base_classes), sourcename)

    def get_object_members(self, want_all: bool) -> Tuple[bool, ObjectMembers]:
        members = get_class_members(self.object, self.objpath, self.get_attr)
        if not want_all:
            if not self.options.members:
                return False, []  # type: ignore
            # specific members given
            selected = []
            for name in self.options.members:  # type: str
                if name in members:
                    selected.append(members[name])
                else:
                    logger.warning(__('missing attribute %s in object %s') %
                                   (name, self.fullname), type='autodoc')
            return False, selected
        elif self.options.inherited_members:
            return False, list(members.values())
        else:
            return False, [m for m in members.values() if m.class_ == self.object]

    def get_doc(self, ignore: int = None) -> Optional[List[List[str]]]:
        if self.doc_as_attr:
            # Don't show the docstring of the class when it is an alias.
            comment = self.get_variable_comment()
            if comment:
                return []
            else:
                return None

        lines = getattr(self, '_new_docstrings', None)
        if lines is not None:
            return lines

        classdoc_from = self.options.get('class-doc-from', self.config.autoclass_content)

        docstrings = []
        attrdocstring = getdoc(self.object, self.get_attr)
        if attrdocstring:
            docstrings.append(attrdocstring)

        # for classes, what the "docstring" is can be controlled via a
        # config value; the default is only the class docstring
        if classdoc_from in ('both', 'init'):
            __init__ = self.get_attr(self.object, '__init__', None)
            initdocstring = getdoc(__init__, self.get_attr,
                                   self.config.autodoc_inherit_docstrings,
                                   self.object, '__init__')
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
                                       self.object, '__new__')
                # for new-style classes, no __new__ means default __new__
                if (initdocstring is not None and
                    (initdocstring == object.__new__.__doc__ or  # for pypy
                     initdocstring.strip() == object.__new__.__doc__)):  # for !pypy
                    initdocstring = None
            if initdocstring:
                if classdoc_from == 'init':
                    docstrings = [initdocstring]
                else:
                    docstrings.append(initdocstring)

        tab_width = self.directive.state.document.settings.tab_width
        return [prepare_docstring(docstring, ignore, tab_width) for docstring in docstrings]

    def get_variable_comment(self) -> Optional[List[str]]:
        try:
            key = ('', '.'.join(self.objpath))
            if self.doc_as_attr:
                analyzer = ModuleAnalyzer.for_module(self.modname)
            else:
                analyzer = ModuleAnalyzer.for_module(self.get_real_modname())
            analyzer.analyze()
            return list(analyzer.attr_docs.get(key, []))
        except PycodeError:
            return None

    def add_content(self, more_content: Optional[StringList], no_docstring: bool = False
                    ) -> None:
        if self.doc_as_attr and self.modname != self.get_real_modname():
            try:
                # override analyzer to obtain doccomment around its definition.
                self.analyzer = ModuleAnalyzer.for_module(self.modname)
                self.analyzer.analyze()
            except PycodeError:
                pass

        if self.doc_as_attr and not self.get_variable_comment():
            try:
                if self.config.autodoc_typehints_format == "short":
                    alias = restify(self.object, "smart")
                else:
                    alias = restify(self.object)
                more_content = StringList([_('alias of %s') % alias], source='')
            except AttributeError:
                pass  # Invalid class object is passed.

        super().add_content(more_content)

    def document_members(self, all_members: bool = False) -> None:
        if self.doc_as_attr:
            return
        super().document_members(all_members)

    def generate(self, more_content: Optional[StringList] = None, real_modname: str = None,
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
### 5 - sphinx/ext/autodoc/preserve_defaults.py:

Start line: 62, End line: 112

```python
def update_defvalue(app: Sphinx, obj: Any, bound_method: bool) -> None:
    """Update defvalue info of *obj* using type_comments."""
    if not app.config.autodoc_preserve_defaults:
        return

    try:
        lines = inspect.getsource(obj).splitlines()
        if lines[0].startswith((' ', r'\t')):
            lines.insert(0, '')  # insert a dummy line to follow what get_function_def() does.
    except (OSError, TypeError):
        lines = []

    try:
        function = get_function_def(obj)
        if function.args.defaults or function.args.kw_defaults:
            sig = inspect.signature(obj)
            defaults = list(function.args.defaults)
            kw_defaults = list(function.args.kw_defaults)
            parameters = list(sig.parameters.values())
            for i, param in enumerate(parameters):
                if param.default is not param.empty:
                    if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD):
                        default = defaults.pop(0)
                        value = get_default_value(lines, default)
                        if value is None:
                            value = ast_unparse(default)  # type: ignore
                        parameters[i] = param.replace(default=DefaultValue(value))
                    else:
                        default = kw_defaults.pop(0)
                        value = get_default_value(lines, default)
                        if value is None:
                            value = ast_unparse(default)  # type: ignore
                        parameters[i] = param.replace(default=DefaultValue(value))
            sig = sig.replace(parameters=parameters)
            obj.__signature__ = sig
    except (AttributeError, TypeError):
        # failed to update signature (ex. built-in or extension types)
        pass
    except NotImplementedError as exc:  # failed to ast.unparse()
        logger.warning(__("Failed to parse a default argument value for %r: %s"), obj, exc)


def setup(app: Sphinx) -> Dict[str, Any]:
    app.add_config_value('autodoc_preserve_defaults', False, True)
    app.connect('autodoc-before-process-signature', update_defvalue)

    return {
        'version': '1.0',
        'parallel_read_safe': True
    }
```
### 6 - sphinx/ext/autodoc/__init__.py:

Start line: 1, End line: 102

```python
"""Extension to create automatic documentation from code docstrings.

Automatically insert docstrings for functions, classes or whole modules into
the doctree, thus avoiding duplication between docstrings and documentation
for those who like elaborate docstrings.
"""

import re
import warnings
from inspect import Parameter, Signature
from types import ModuleType
from typing import (TYPE_CHECKING, Any, Callable, Dict, Iterator, List, Optional, Sequence,
                    Set, Tuple, Type, TypeVar, Union)

from docutils.statemachine import StringList

import sphinx
from sphinx.application import Sphinx
from sphinx.config import ENUM, Config
from sphinx.deprecation import RemovedInSphinx50Warning, RemovedInSphinx60Warning
from sphinx.environment import BuildEnvironment
from sphinx.ext.autodoc.importer import (get_class_members, get_object_members, import_module,
                                         import_object)
from sphinx.ext.autodoc.mock import ismock, mock, undecorate
from sphinx.locale import _, __
from sphinx.pycode import ModuleAnalyzer, PycodeError
from sphinx.util import inspect, logging
from sphinx.util.docstrings import prepare_docstring, separate_metadata
from sphinx.util.inspect import (evaluate_signature, getdoc, object_description, safe_getattr,
                                 stringify_signature)
from sphinx.util.typing import OptionSpec, get_type_hints, restify
from sphinx.util.typing import stringify as stringify_typehint

if TYPE_CHECKING:
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

    def append(self, item: Any) -> None:
        pass  # nothing


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
    if arg in (None, True):
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
```
### 7 - sphinx/ext/autodoc/__init__.py:

Start line: 1279, End line: 1403

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
        if self.config.autodoc_typehints_format == "short":
            kwargs.setdefault('unqualified_typehints', True)

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

        if inspect.iscoroutinefunction(self.object) or inspect.isasyncgenfunction(self.object):
            self.add_line('   :async:', sourcename)

    def format_signature(self, **kwargs: Any) -> str:
        if self.config.autodoc_typehints_format == "short":
            kwargs.setdefault('unqualified_typehints', True)

        sigs = []
        if (self.analyzer and
                '.'.join(self.objpath) in self.analyzer.overloads and
                self.config.autodoc_typehints != 'none'):
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
                    dispatchfunc = self.annotate_to_first_argument(func, typ)
                    if dispatchfunc:
                        documenter = FunctionDocumenter(self.directive, '')
                        documenter.object = dispatchfunc
                        documenter.objpath = [None]
                        sigs.append(documenter.format_signature())
        if overloaded:
            actual = inspect.signature(self.object,
                                       type_aliases=self.config.autodoc_type_aliases)
            __globals__ = safe_getattr(self.object, '__globals__', {})
            for overload in self.analyzer.overloads.get('.'.join(self.objpath)):
                overload = self.merge_default_value(actual, overload)
                overload = evaluate_signature(overload, __globals__,
                                              self.config.autodoc_type_aliases)

                sig = stringify_signature(overload, **kwargs)
                sigs.append(sig)

        return "\n".join(sigs)

    def merge_default_value(self, actual: Signature, overload: Signature) -> Signature:
        """Merge default values of actual implementation to the overload variants."""
        parameters = list(overload.parameters.values())
        for i, param in enumerate(parameters):
            actual_param = actual.parameters.get(param.name)
            if actual_param and param.default == '...':
                parameters[i] = param.replace(default=actual_param.default)

        return overload.replace(parameters=parameters)

    def annotate_to_first_argument(self, func: Callable, typ: Type) -> Optional[Callable]:
        """Annotate type hint to the first argument of function if needed."""
        try:
            sig = inspect.signature(func, type_aliases=self.config.autodoc_type_aliases)
        except TypeError as exc:
            logger.warning(__("Failed to get a function signature for %s: %s"),
                           self.fullname, exc)
            return None
        except ValueError:
            return None

        if len(sig.parameters) == 0:
            return None

        def dummy():
            pass

        params = list(sig.parameters.values())
        if params[0].annotation is Parameter.empty:
            params[0] = params[0].replace(annotation=typ)
            try:
                dummy.__signature__ = sig.replace(parameters=params)  # type: ignore
                return dummy
            except (AttributeError, TypeError):
                # failed to update signature (ex. built-in or extension types)
                return None
        else:
            return None
```
### 8 - sphinx/ext/autodoc/__init__.py:

Start line: 2638, End line: 2661

```python
class AttributeDocumenter(GenericAliasMixin, NewTypeMixin, SlotsMixin,  # type: ignore
                          TypeVarMixin, RuntimeInstanceAttributeMixin,
                          UninitializedInstanceAttributeMixin, NonDataDescriptorMixin,
                          DocstringStripSignatureMixin, ClassLevelDocumenter):

    def import_object(self, raiseerror: bool = False) -> bool:
        ret = super().import_object(raiseerror)
        if inspect.isenumattribute(self.object):
            self.object = self.object.value
        if self.parent:
            self.update_annotations(self.parent)

        return ret

    def get_real_modname(self) -> str:
        real_modname = self.get_attr(self.parent or self.object, '__module__', None)
        return real_modname or self.modname

    def should_suppress_value_header(self) -> bool:
        if super().should_suppress_value_header():
            return True
        else:
            doc = self.get_doc()
            if doc:
                docstring, metadata = separate_metadata('\n'.join(sum(doc, [])))
                if 'hide-value' in metadata:
                    return True

        return False
```
### 9 - sphinx/ext/autodoc/__init__.py:

Start line: 1083, End line: 1107

```python
class ModuleDocumenter(Documenter):

    def get_object_members(self, want_all: bool) -> Tuple[bool, ObjectMembers]:
        members = self.get_module_members()
        if want_all:
            if self.__all__ is None:
                # for implicit module members, check __module__ to avoid
                # documenting imported objects
                return True, list(members.values())
            else:
                for member in members.values():
                    if member.__name__ not in self.__all__:
                        member.skipped = True

                return False, list(members.values())
        else:
            memberlist = self.options.members or []
            ret = []
            for name in memberlist:
                if name in members:
                    ret.append(members[name])
                else:
                    logger.warning(__('missing attribute mentioned in :members: option: '
                                      'module %s, attribute %s') %
                                   (safe_getattr(self.object, '__name__', '???'), name),
                                   type='autodoc')
            return False, ret
```
### 10 - sphinx/ext/autodoc/directive.py:

Start line: 1, End line: 39

```python
import warnings
from typing import Any, Callable, Dict, List, Set, Type

from docutils import nodes
from docutils.nodes import Element, Node
from docutils.parsers.rst.states import RSTState
from docutils.statemachine import StringList
from docutils.utils import Reporter, assemble_option_dict

from sphinx.config import Config
from sphinx.deprecation import RemovedInSphinx50Warning, RemovedInSphinx60Warning
from sphinx.environment import BuildEnvironment
from sphinx.ext.autodoc import Documenter, Options
from sphinx.util import logging
from sphinx.util.docutils import SphinxDirective, switch_source_input
from sphinx.util.nodes import nested_parse_with_titles

logger = logging.getLogger(__name__)


# common option names for autodoc directives
AUTODOC_DEFAULT_OPTIONS = ['members', 'undoc-members', 'inherited-members',
                           'show-inheritance', 'private-members', 'special-members',
                           'ignore-module-all', 'exclude-members', 'member-order',
                           'imported-members', 'class-doc-from', 'no-value']

AUTODOC_EXTENDABLE_OPTIONS = ['members', 'private-members', 'special-members',
                              'exclude-members']


class DummyOptionSpec(dict):
    """An option_spec allows any options."""

    def __bool__(self) -> bool:
        """Behaves like some options are defined."""
        return True

    def __getitem__(self, key: str) -> Callable[[str], str]:
        return lambda x: x
```
### 48 - sphinx/ext/autodoc/preserve_defaults.py:

Start line: 29, End line: 59

```python
def get_function_def(obj: Any) -> ast.FunctionDef:
    """Get FunctionDef object from living object.
    This tries to parse original code for living object and returns
    AST node for given *obj*.
    """
    try:
        source = inspect.getsource(obj)
        if source.startswith((' ', r'\t')):
            # subject is placed inside class or block.  To read its docstring,
            # this adds if-block before the declaration.
            module = ast_parse('if True:\n' + source)
            return module.body[0].body[0]  # type: ignore
        else:
            module = ast_parse(source)
            return module.body[0]  # type: ignore
    except (OSError, TypeError):  # failed to load source code
        return None


def get_default_value(lines: List[str], position: ast.AST) -> Optional[str]:
    try:
        if sys.version_info < (3, 8):  # only for py38+
            return None
        elif position.lineno == position.end_lineno:
            line = lines[position.lineno - 1]
            return line[position.col_offset:position.end_col_offset]
        else:
            # multiline value is not supported now
            return None
    except (AttributeError, IndexError):
        return None
```
