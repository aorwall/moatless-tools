# sphinx-doc__sphinx-9128

| **sphinx-doc/sphinx** | `dfdc7626b5dd06bff3d326e6efddc492ef00c471` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | 15850 |
| **Avg pos** | 22.0 |
| **Min pos** | 22 |
| **Max pos** | 22 |
| **Top file pos** | 6 |
| **Missing snippets** | 4 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sphinx/domains/python.py b/sphinx/domains/python.py
--- a/sphinx/domains/python.py
+++ b/sphinx/domains/python.py
@@ -68,7 +68,7 @@ class ObjectEntry(NamedTuple):
     docname: str
     node_id: str
     objtype: str
-    canonical: bool
+    aliased: bool
 
 
 class ModuleEntry(NamedTuple):
@@ -505,7 +505,7 @@ def add_target_and_index(self, name_cls: Tuple[str, str], sig: str,
 
         canonical_name = self.options.get('canonical')
         if canonical_name:
-            domain.note_object(canonical_name, self.objtype, node_id, canonical=True,
+            domain.note_object(canonical_name, self.objtype, node_id, aliased=True,
                                location=signode)
 
         if 'noindexentry' not in self.options:
@@ -1138,17 +1138,25 @@ def objects(self) -> Dict[str, ObjectEntry]:
         return self.data.setdefault('objects', {})  # fullname -> ObjectEntry
 
     def note_object(self, name: str, objtype: str, node_id: str,
-                    canonical: bool = False, location: Any = None) -> None:
+                    aliased: bool = False, location: Any = None) -> None:
         """Note a python object for cross reference.
 
         .. versionadded:: 2.1
         """
         if name in self.objects:
             other = self.objects[name]
-            logger.warning(__('duplicate object description of %s, '
-                              'other instance in %s, use :noindex: for one of them'),
-                           name, other.docname, location=location)
-        self.objects[name] = ObjectEntry(self.env.docname, node_id, objtype, canonical)
+            if other.aliased and aliased is False:
+                # The original definition found. Override it!
+                pass
+            elif other.aliased is False and aliased:
+                # The original definition is already registered.
+                return
+            else:
+                # duplicated
+                logger.warning(__('duplicate object description of %s, '
+                                  'other instance in %s, use :noindex: for one of them'),
+                               name, other.docname, location=location)
+        self.objects[name] = ObjectEntry(self.env.docname, node_id, objtype, aliased)
 
     @property
     def modules(self) -> Dict[str, ModuleEntry]:
@@ -1326,8 +1334,8 @@ def get_objects(self) -> Iterator[Tuple[str, str, str, str, str, int]]:
             yield (modname, modname, 'module', mod.docname, mod.node_id, 0)
         for refname, obj in self.objects.items():
             if obj.objtype != 'module':  # modules are already handled
-                if obj.canonical:
-                    # canonical names are not full-text searchable.
+                if obj.aliased:
+                    # aliased names are not full-text searchable.
                     yield (refname, refname, obj.objtype, obj.docname, obj.node_id, -1)
                 else:
                     yield (refname, refname, obj.objtype, obj.docname, obj.node_id, 1)

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sphinx/domains/python.py | 71 | 71 | 22 | 6 | 15850
| sphinx/domains/python.py | 508 | 508 | - | 6 | -
| sphinx/domains/python.py | 1141 | 1151 | - | 6 | -
| sphinx/domains/python.py | 1329 | 1330 | - | 6 | -


## Problem Statement

```
autodoc: duplication warning on documenting aliased object
**Describe the bug**
autodoc: duplication warning on documenting aliased object

**To Reproduce**
\`\`\`
# example.py
from io import StringIO
\`\`\`
\`\`\`
# index.rst
.. autoclass:: example.StringIO
.. autoclass:: io.StringIO
\`\`\`
\`\`\`
Removing everything under '_build'...
Running Sphinx v4.0.0+/dfdc7626b
making output directory... done
[autosummary] generating autosummary for: index.rst
building [mo]: targets for 0 po files that are out of date
building [html]: targets for 1 source files that are out of date
updating environment: [new config] 1 added, 0 changed, 0 removed
reading sources... [100%] index
docstring of _io.StringIO:1: WARNING: duplicate object description of _io.StringIO, other instance in index, use :noindex: for one of them
looking for now-outdated files... none found
pickling environment... done
checking consistency... done
preparing documents... done
writing output... [100%] index
generating indices... genindex done
writing additional pages... search done
copying static files... done
copying extra files... done
dumping search index in English (code: en)... done
dumping object inventory... done
build succeeded, 1 warning.

The HTML pages are in _build/html.
\`\`\`

**Expected behavior**
No warning

**Your project**
N/A

**Screenshots**
N/A

**Environment info**
- OS: Mac
- Python version: 3.9.4
- Sphinx version: HEAD of 4.0.x
- Sphinx extensions: sphinx.ext.autodoc
- Extra tools: No

**Additional context**
No


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 sphinx/ext/autodoc/deprecated.py | 114 | 127| 114 | 114 | 960 | 
| 2 | 2 sphinx/ext/autodoc/__init__.py | 2440 | 2463| 220 | 334 | 23401 | 
| 3 | 2 sphinx/ext/autodoc/__init__.py | 1927 | 1949| 245 | 579 | 23401 | 
| 4 | 3 sphinx/ext/autosummary/__init__.py | 55 | 103| 372 | 951 | 29856 | 
| 5 | 3 sphinx/ext/autodoc/__init__.py | 1758 | 1773| 129 | 1080 | 29856 | 
| 6 | 4 sphinx/ext/autosummary/generate.py | 20 | 53| 257 | 1337 | 35008 | 
| 7 | 4 sphinx/ext/autodoc/__init__.py | 1408 | 1717| 2902 | 4239 | 35008 | 
| 8 | 4 sphinx/ext/autodoc/__init__.py | 2465 | 2486| 272 | 4511 | 35008 | 
| 9 | 4 sphinx/ext/autosummary/generate.py | 350 | 440| 756 | 5267 | 35008 | 
| 10 | 4 sphinx/ext/autodoc/deprecated.py | 32 | 47| 140 | 5407 | 35008 | 
| 11 | 4 sphinx/ext/autodoc/__init__.py | 1065 | 1089| 209 | 5616 | 35008 | 
| 12 | 4 sphinx/ext/autodoc/__init__.py | 2505 | 2530| 316 | 5932 | 35008 | 
| 13 | 4 sphinx/ext/autodoc/__init__.py | 2624 | 2665| 488 | 6420 | 35008 | 
| 14 | 4 sphinx/ext/autodoc/__init__.py | 1951 | 1988| 314 | 6734 | 35008 | 
| 15 | 4 sphinx/ext/autosummary/generate.py | 225 | 240| 184 | 6918 | 35008 | 
| 16 | 4 sphinx/ext/autosummary/__init__.py | 237 | 283| 457 | 7375 | 35008 | 
| 17 | 4 sphinx/ext/autodoc/__init__.py | 1011 | 1023| 124 | 7499 | 35008 | 
| 18 | 5 sphinx/domains/c.py | 2779 | 3576| 6651 | 14150 | 67085 | 
| 19 | 5 sphinx/ext/autosummary/generate.py | 172 | 191| 176 | 14326 | 67085 | 
| 20 | 5 sphinx/ext/autodoc/deprecated.py | 11 | 29| 155 | 14481 | 67085 | 
| 21 | 5 sphinx/ext/autodoc/__init__.py | 699 | 806| 865 | 15346 | 67085 | 
| **-> 22 <-** | **6 sphinx/domains/python.py** | 11 | 79| 504 | 15850 | 78879 | 
| 23 | 7 sphinx/ext/autodoc/importer.py | 77 | 147| 649 | 16499 | 81341 | 
| 24 | 7 sphinx/ext/autodoc/importer.py | 11 | 40| 211 | 16710 | 81341 | 
| 25 | 7 sphinx/ext/autodoc/__init__.py | 2488 | 2503| 182 | 16892 | 81341 | 
| 26 | 8 sphinx/domains/std.py | 11 | 46| 311 | 17203 | 91606 | 
| 27 | 8 sphinx/ext/autodoc/__init__.py | 2236 | 2250| 152 | 17355 | 91606 | 
| 28 | 8 sphinx/ext/autodoc/__init__.py | 573 | 585| 133 | 17488 | 91606 | 
| 29 | 9 doc/conf.py | 1 | 82| 731 | 18219 | 93070 | 
| 30 | 9 sphinx/ext/autosummary/__init__.py | 285 | 306| 200 | 18419 | 93070 | 
| 31 | 10 sphinx/io.py | 10 | 39| 234 | 18653 | 94475 | 
| 32 | 11 sphinx/builders/html/__init__.py | 11 | 62| 432 | 19085 | 106655 | 
| 33 | 12 sphinx/domains/cpp.py | 6825 | 7557| 6257 | 25342 | 172312 | 
| 34 | 12 sphinx/ext/autodoc/__init__.py | 13 | 111| 770 | 26112 | 172312 | 
| 35 | 13 sphinx/deprecation.py | 11 | 36| 156 | 26268 | 173014 | 
| 36 | 13 sphinx/ext/autodoc/deprecated.py | 78 | 93| 139 | 26407 | 173014 | 
| 37 | 13 sphinx/ext/autosummary/generate.py | 242 | 267| 292 | 26699 | 173014 | 
| 38 | 14 sphinx/directives/__init__.py | 269 | 294| 167 | 26866 | 175266 | 
| 39 | 15 sphinx/ext/autodoc/mock.py | 11 | 69| 452 | 27318 | 176614 | 
| 40 | 15 sphinx/ext/autodoc/__init__.py | 1025 | 1036| 126 | 27444 | 176614 | 
| 41 | 15 sphinx/ext/autodoc/__init__.py | 285 | 361| 751 | 28195 | 176614 | 
| 42 | 15 sphinx/domains/c.py | 3579 | 3891| 2923 | 31118 | 176614 | 
| 43 | 15 sphinx/ext/autosummary/__init__.py | 756 | 783| 376 | 31494 | 176614 | 
| 44 | 16 sphinx/ext/autodoc/directive.py | 9 | 47| 305 | 31799 | 178059 | 
| 45 | 17 sphinx/builders/__init__.py | 532 | 540| 120 | 31919 | 183401 | 
| 46 | 17 sphinx/ext/autosummary/generate.py | 299 | 347| 516 | 32435 | 183401 | 
| 47 | 17 sphinx/ext/autodoc/deprecated.py | 65 | 75| 108 | 32543 | 183401 | 
| 48 | 18 sphinx/util/inspect.py | 11 | 45| 277 | 32820 | 189302 | 
| 49 | 18 sphinx/ext/autodoc/__init__.py | 1878 | 1925| 362 | 33182 | 189302 | 
| 50 | 18 sphinx/ext/autodoc/__init__.py | 2325 | 2358| 289 | 33471 | 189302 | 
| 51 | 18 sphinx/ext/autodoc/__init__.py | 2397 | 2416| 224 | 33695 | 189302 | 
| 52 | 18 sphinx/ext/autosummary/__init__.py | 132 | 178| 355 | 34050 | 189302 | 
| 53 | 18 sphinx/ext/autosummary/generate.py | 84 | 96| 163 | 34213 | 189302 | 
| 54 | 18 sphinx/directives/__init__.py | 50 | 72| 187 | 34400 | 189302 | 
| 55 | 19 sphinx/builders/singlehtml.py | 11 | 26| 128 | 34528 | 191173 | 
| 56 | 19 sphinx/ext/autosummary/__init__.py | 181 | 213| 279 | 34807 | 191173 | 
| 57 | 19 sphinx/ext/autodoc/directive.py | 125 | 175| 453 | 35260 | 191173 | 
| 58 | 19 sphinx/ext/autodoc/__init__.py | 1837 | 1875| 307 | 35567 | 191173 | 
| 59 | 19 sphinx/ext/autosummary/generate.py | 458 | 479| 217 | 35784 | 191173 | 
| 60 | 19 sphinx/ext/autodoc/__init__.py | 2175 | 2203| 248 | 36032 | 191173 | 
| 61 | 19 sphinx/ext/autosummary/__init__.py | 718 | 753| 325 | 36357 | 191173 | 
| 62 | 20 sphinx/directives/patches.py | 9 | 34| 192 | 36549 | 193017 | 
| 63 | 20 sphinx/ext/autodoc/__init__.py | 1819 | 1834| 160 | 36709 | 193017 | 
| 64 | 20 sphinx/domains/std.py | 1112 | 1136| 205 | 36914 | 193017 | 
| 65 | 20 sphinx/ext/autosummary/generate.py | 284 | 297| 198 | 37112 | 193017 | 
| 66 | 20 sphinx/ext/autodoc/deprecated.py | 96 | 111| 133 | 37245 | 193017 | 
| 67 | 21 sphinx/ext/autodoc/typehints.py | 130 | 185| 454 | 37699 | 194461 | 
| 68 | **21 sphinx/domains/python.py** | 954 | 972| 140 | 37839 | 194461 | 
| 69 | 22 sphinx/ext/inheritance_diagram.py | 38 | 67| 243 | 38082 | 198327 | 


### Hint

```
I noticed the example is not good. Both `example.StringIO` and `io.StringIO` are aliases of `_io.StringIO`. So they're surely conflicted.

It would be better to not emit a warning for this case:
\`\`\`
.. autoclass:: _io.StringIO
.. autoclass:: io.StringIO
\`\`\`

The former one is a canonical name of the `io.StringIO`. So this should not be conflicted.
```

## Patch

```diff
diff --git a/sphinx/domains/python.py b/sphinx/domains/python.py
--- a/sphinx/domains/python.py
+++ b/sphinx/domains/python.py
@@ -68,7 +68,7 @@ class ObjectEntry(NamedTuple):
     docname: str
     node_id: str
     objtype: str
-    canonical: bool
+    aliased: bool
 
 
 class ModuleEntry(NamedTuple):
@@ -505,7 +505,7 @@ def add_target_and_index(self, name_cls: Tuple[str, str], sig: str,
 
         canonical_name = self.options.get('canonical')
         if canonical_name:
-            domain.note_object(canonical_name, self.objtype, node_id, canonical=True,
+            domain.note_object(canonical_name, self.objtype, node_id, aliased=True,
                                location=signode)
 
         if 'noindexentry' not in self.options:
@@ -1138,17 +1138,25 @@ def objects(self) -> Dict[str, ObjectEntry]:
         return self.data.setdefault('objects', {})  # fullname -> ObjectEntry
 
     def note_object(self, name: str, objtype: str, node_id: str,
-                    canonical: bool = False, location: Any = None) -> None:
+                    aliased: bool = False, location: Any = None) -> None:
         """Note a python object for cross reference.
 
         .. versionadded:: 2.1
         """
         if name in self.objects:
             other = self.objects[name]
-            logger.warning(__('duplicate object description of %s, '
-                              'other instance in %s, use :noindex: for one of them'),
-                           name, other.docname, location=location)
-        self.objects[name] = ObjectEntry(self.env.docname, node_id, objtype, canonical)
+            if other.aliased and aliased is False:
+                # The original definition found. Override it!
+                pass
+            elif other.aliased is False and aliased:
+                # The original definition is already registered.
+                return
+            else:
+                # duplicated
+                logger.warning(__('duplicate object description of %s, '
+                                  'other instance in %s, use :noindex: for one of them'),
+                               name, other.docname, location=location)
+        self.objects[name] = ObjectEntry(self.env.docname, node_id, objtype, aliased)
 
     @property
     def modules(self) -> Dict[str, ModuleEntry]:
@@ -1326,8 +1334,8 @@ def get_objects(self) -> Iterator[Tuple[str, str, str, str, str, int]]:
             yield (modname, modname, 'module', mod.docname, mod.node_id, 0)
         for refname, obj in self.objects.items():
             if obj.objtype != 'module':  # modules are already handled
-                if obj.canonical:
-                    # canonical names are not full-text searchable.
+                if obj.aliased:
+                    # aliased names are not full-text searchable.
                     yield (refname, refname, obj.objtype, obj.docname, obj.node_id, -1)
                 else:
                     yield (refname, refname, obj.objtype, obj.docname, obj.node_id, 1)

```

## Test Patch

```diff
diff --git a/tests/test_domain_py.py b/tests/test_domain_py.py
--- a/tests/test_domain_py.py
+++ b/tests/test_domain_py.py
@@ -870,6 +870,39 @@ def test_canonical(app):
     assert domain.objects['_io.StringIO'] == ('index', 'io.StringIO', 'class', True)
 
 
+def test_canonical_definition_overrides(app, warning):
+    text = (".. py:class:: io.StringIO\n"
+            "   :canonical: _io.StringIO\n"
+            ".. py:class:: _io.StringIO\n")
+    restructuredtext.parse(app, text)
+    assert warning.getvalue() == ""
+
+    domain = app.env.get_domain('py')
+    assert domain.objects['_io.StringIO'] == ('index', 'id0', 'class', False)
+
+
+def test_canonical_definition_skip(app, warning):
+    text = (".. py:class:: _io.StringIO\n"
+            ".. py:class:: io.StringIO\n"
+            "   :canonical: _io.StringIO\n")
+
+    restructuredtext.parse(app, text)
+    assert warning.getvalue() == ""
+
+    domain = app.env.get_domain('py')
+    assert domain.objects['_io.StringIO'] == ('index', 'io.StringIO', 'class', False)
+
+
+def test_canonical_duplicated(app, warning):
+    text = (".. py:class:: mypackage.StringIO\n"
+            "   :canonical: _io.StringIO\n"
+            ".. py:class:: io.StringIO\n"
+            "   :canonical: _io.StringIO\n")
+
+    restructuredtext.parse(app, text)
+    assert warning.getvalue() != ""
+
+
 def test_info_field_list(app):
     text = (".. py:module:: example\n"
             ".. py:class:: Class\n"

```


## Code snippets

### 1 - sphinx/ext/autodoc/deprecated.py:

Start line: 114, End line: 127

```python
class GenericAliasDocumenter(DataDocumenter):
    """
    Specialized Documenter subclass for GenericAliases.
    """

    objtype = 'genericalias'
    directivetype = 'data'
    priority = DataDocumenter.priority + 1  # type: ignore

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        warnings.warn("%s is deprecated." % self.__class__.__name__,
                      RemovedInSphinx50Warning, stacklevel=2)
        super().__init__(*args, **kwargs)
```
### 2 - sphinx/ext/autodoc/__init__.py:

Start line: 2440, End line: 2463

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
        return self.get_attr(self.parent or self.object, '__module__', None) \
            or self.modname

    def should_suppress_value_header(self) -> bool:
        if super().should_suppress_value_header():
            return True
        else:
            doc = self.get_doc()
            if doc:
                metadata = extract_metadata('\n'.join(sum(doc, [])))
                if 'hide-value' in metadata:
                    return True

        return False
```
### 3 - sphinx/ext/autodoc/__init__.py:

Start line: 1927, End line: 1949

```python
class DataDocumenter(GenericAliasMixin, NewTypeMixin, TypeVarMixin,
                     UninitializedGlobalVariableMixin, ModuleLevelDocumenter):

    def add_directive_header(self, sig: str) -> None:
        super().add_directive_header(sig)
        sourcename = self.get_sourcename()
        if self.options.annotation is SUPPRESS or self.should_suppress_directive_header():
            pass
        elif self.options.annotation:
            self.add_line('   :annotation: %s' % self.options.annotation,
                          sourcename)
        else:
            # obtain annotation for this data
            annotations = get_type_hints(self.parent, None, self.config.autodoc_type_aliases)
            if self.objpath[-1] in annotations:
                objrepr = stringify_typehint(annotations.get(self.objpath[-1]))
                self.add_line('   :type: ' + objrepr, sourcename)

            try:
                if self.options.no_value or self.should_suppress_value_header():
                    pass
                else:
                    objrepr = object_description(self.object)
                    self.add_line('   :value: ' + objrepr, sourcename)
            except ValueError:
                pass
```
### 4 - sphinx/ext/autosummary/__init__.py:

Start line: 55, End line: 103

```python
import inspect
import os
import posixpath
import re
import sys
import warnings
from os import path
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple, Type, cast

from docutils import nodes
from docutils.nodes import Element, Node, system_message
from docutils.parsers.rst import directives
from docutils.parsers.rst.states import RSTStateMachine, Struct, state_classes
from docutils.statemachine import StringList

import sphinx
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.config import Config
from sphinx.deprecation import RemovedInSphinx50Warning
from sphinx.environment import BuildEnvironment
from sphinx.environment.adapters.toctree import TocTree
from sphinx.ext.autodoc import INSTANCEATTR, Documenter
from sphinx.ext.autodoc.directive import DocumenterBridge, Options
from sphinx.ext.autodoc.importer import import_module
from sphinx.ext.autodoc.mock import mock
from sphinx.locale import __
from sphinx.pycode import ModuleAnalyzer, PycodeError
from sphinx.util import logging, rst
from sphinx.util.docutils import (NullReporter, SphinxDirective, SphinxRole, new_document,
                                  switch_source_input)
from sphinx.util.matching import Matcher
from sphinx.util.typing import OptionSpec
from sphinx.writers.html import HTMLTranslator

logger = logging.getLogger(__name__)


periods_re = re.compile(r'\.(?:\s+)')
literal_re = re.compile(r'::\s*$')

WELL_KNOWN_ABBREVIATIONS = ('et al.', ' i.e.',)


# -- autosummary_toc node ------------------------------------------------------

class autosummary_toc(nodes.comment):
    pass
```
### 5 - sphinx/ext/autodoc/__init__.py:

Start line: 1758, End line: 1773

```python
class GenericAliasMixin(DataDocumenterMixinBase):
    """
    Mixin for DataDocumenter and AttributeDocumenter to provide the feature for
    supporting GenericAliases.
    """

    def should_suppress_directive_header(self) -> bool:
        return (inspect.isgenericalias(self.object) or
                super().should_suppress_directive_header())

    def update_content(self, more_content: StringList) -> None:
        if inspect.isgenericalias(self.object):
            more_content.append(_('alias of %s') % restify(self.object), '')
            more_content.append('', '')

        super().update_content(more_content)
```
### 6 - sphinx/ext/autosummary/generate.py:

Start line: 20, End line: 53

```python
import argparse
import inspect
import locale
import os
import pkgutil
import pydoc
import re
import sys
import warnings
from gettext import NullTranslations
from os import path
from typing import Any, Dict, List, NamedTuple, Set, Tuple, Type, Union

from jinja2 import TemplateNotFound
from jinja2.sandbox import SandboxedEnvironment

import sphinx.locale
from sphinx import __display_version__, package_dir
from sphinx.application import Sphinx
from sphinx.builders import Builder
from sphinx.config import Config
from sphinx.deprecation import RemovedInSphinx50Warning
from sphinx.ext.autodoc import Documenter
from sphinx.ext.autodoc.importer import import_module
from sphinx.ext.autosummary import get_documenter, import_by_name, import_ivar_by_name
from sphinx.locale import __
from sphinx.pycode import ModuleAnalyzer, PycodeError
from sphinx.registry import SphinxComponentRegistry
from sphinx.util import logging, rst, split_full_qualified_name
from sphinx.util.inspect import safe_getattr
from sphinx.util.osutil import ensuredir
from sphinx.util.template import SphinxTemplateLoader

logger = logging.getLogger(__name__)
```
### 7 - sphinx/ext/autodoc/__init__.py:

Start line: 1408, End line: 1717

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
    }

    _signature_class: Any = None
    _signature_method_name: str = None

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
                    analyzer.analyze()
                    qualname = '.'.join([cls.__qualname__, self._signature_method_name])
                    if qualname in analyzer.overloads:
                        return analyzer.overloads.get(qualname)
                    elif qualname in analyzer.tagorder:
                        # the constructor is defined in the class, but not overrided.
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
            return None

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

    def add_content(self, more_content: Optional[StringList], no_docstring: bool = False
                    ) -> None:
        if self.doc_as_attr:
            try:
                more_content = StringList([_('alias of %s') % restify(self.object)], source='')
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
### 8 - sphinx/ext/autodoc/__init__.py:

Start line: 2465, End line: 2486

```python
class AttributeDocumenter(GenericAliasMixin, NewTypeMixin, SlotsMixin,  # type: ignore
                          TypeVarMixin, RuntimeInstanceAttributeMixin,
                          UninitializedInstanceAttributeMixin, NonDataDescriptorMixin,
                          DocstringStripSignatureMixin, ClassLevelDocumenter):

    def add_directive_header(self, sig: str) -> None:
        super().add_directive_header(sig)
        sourcename = self.get_sourcename()
        if self.options.annotation is SUPPRESS or self.should_suppress_directive_header():
            pass
        elif self.options.annotation:
            self.add_line('   :annotation: %s' % self.options.annotation, sourcename)
        else:
            # obtain type annotation for this attribute
            annotations = get_type_hints(self.parent, None, self.config.autodoc_type_aliases)
            if self.objpath[-1] in annotations:
                objrepr = stringify_typehint(annotations.get(self.objpath[-1]))
                self.add_line('   :type: ' + objrepr, sourcename)

            try:
                if self.options.no_value or self.should_suppress_value_header():
                    pass
                else:
                    objrepr = object_description(self.object)
                    self.add_line('   :value: ' + objrepr, sourcename)
            except ValueError:
                pass
```
### 9 - sphinx/ext/autosummary/generate.py:

Start line: 350, End line: 440

```python
def generate_autosummary_docs(sources: List[str], output_dir: str = None,
                              suffix: str = '.rst', base_path: str = None,
                              builder: Builder = None, template_dir: str = None,
                              imported_members: bool = False, app: Any = None,
                              overwrite: bool = True, encoding: str = 'utf-8') -> None:
    if builder:
        warnings.warn('builder argument for generate_autosummary_docs() is deprecated.',
                      RemovedInSphinx50Warning, stacklevel=2)

    if template_dir:
        warnings.warn('template_dir argument for generate_autosummary_docs() is deprecated.',
                      RemovedInSphinx50Warning, stacklevel=2)

    showed_sources = list(sorted(sources))
    if len(showed_sources) > 20:
        showed_sources = showed_sources[:10] + ['...'] + showed_sources[-10:]
    logger.info(__('[autosummary] generating autosummary for: %s') %
                ', '.join(showed_sources))

    if output_dir:
        logger.info(__('[autosummary] writing to %s') % output_dir)

    if base_path is not None:
        sources = [os.path.join(base_path, filename) for filename in sources]

    template = AutosummaryRenderer(app)

    # read
    items = find_autosummary_in_files(sources)

    # keep track of new files
    new_files = []

    if app:
        filename_map = app.config.autosummary_filename_map
    else:
        filename_map = {}

    # write
    for entry in sorted(set(items), key=str):
        if entry.path is None:
            # The corresponding autosummary:: directive did not have
            # a :toctree: option
            continue

        path = output_dir or os.path.abspath(entry.path)
        ensuredir(path)

        try:
            name, obj, parent, modname = import_by_name(entry.name)
            qualname = name.replace(modname + ".", "")
        except ImportError as e:
            try:
                # try to importl as an instance attribute
                name, obj, parent, modname = import_ivar_by_name(entry.name)
                qualname = name.replace(modname + ".", "")
            except ImportError:
                logger.warning(__('[autosummary] failed to import %r: %s') % (entry.name, e))
                continue

        context: Dict[str, Any] = {}
        if app:
            context.update(app.config.autosummary_context)

        content = generate_autosummary_content(name, obj, parent, template, entry.template,
                                               imported_members, app, entry.recursive, context,
                                               modname, qualname)

        filename = os.path.join(path, filename_map.get(name, name) + suffix)
        if os.path.isfile(filename):
            with open(filename, encoding=encoding) as f:
                old_content = f.read()

            if content == old_content:
                continue
            elif overwrite:  # content has changed
                with open(filename, 'w', encoding=encoding) as f:
                    f.write(content)
                new_files.append(filename)
        else:
            with open(filename, 'w', encoding=encoding) as f:
                f.write(content)
            new_files.append(filename)

    # descend recursively to new files
    if new_files:
        generate_autosummary_docs(new_files, output_dir=output_dir,
                                  suffix=suffix, base_path=base_path,
                                  builder=builder, template_dir=template_dir,
                                  imported_members=imported_members, app=app,
                                  overwrite=overwrite)
```
### 10 - sphinx/ext/autodoc/deprecated.py:

Start line: 32, End line: 47

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

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        warnings.warn("%s is deprecated." % self.__class__.__name__,
                      RemovedInSphinx50Warning, stacklevel=2)
        super().__init__(*args, **kwargs)
```
### 22 - sphinx/domains/python.py:

Start line: 11, End line: 79

```python
import builtins
import inspect
import re
import sys
import typing
import warnings
from inspect import Parameter
from typing import Any, Dict, Iterable, Iterator, List, NamedTuple, Tuple, Type, cast

from docutils import nodes
from docutils.nodes import Element, Node
from docutils.parsers.rst import directives

from sphinx import addnodes
from sphinx.addnodes import desc_signature, pending_xref, pending_xref_condition
from sphinx.application import Sphinx
from sphinx.builders import Builder
from sphinx.deprecation import RemovedInSphinx50Warning
from sphinx.directives import ObjectDescription
from sphinx.domains import Domain, Index, IndexEntry, ObjType
from sphinx.environment import BuildEnvironment
from sphinx.locale import _, __
from sphinx.pycode.ast import ast
from sphinx.pycode.ast import parse as ast_parse
from sphinx.roles import XRefRole
from sphinx.util import logging
from sphinx.util.docfields import Field, GroupedField, TypedField
from sphinx.util.docutils import SphinxDirective
from sphinx.util.inspect import signature_from_str
from sphinx.util.nodes import find_pending_xref_condition, make_id, make_refnode
from sphinx.util.typing import OptionSpec, TextlikeNode

logger = logging.getLogger(__name__)


# REs for Python signatures
py_sig_re = re.compile(
    r'''^ ([\w.]*\.)?            # class name(s)
          (\w+)  \s*             # thing name
          (?: \(\s*(.*)\s*\)     # optional: arguments
           (?:\s* -> \s* (.*))?  #           return annotation
          )? $                   # and nothing more
          ''', re.VERBOSE)


pairindextypes = {
    'module':    _('module'),
    'keyword':   _('keyword'),
    'operator':  _('operator'),
    'object':    _('object'),
    'exception': _('exception'),
    'statement': _('statement'),
    'builtin':   _('built-in function'),
}


class ObjectEntry(NamedTuple):
    docname: str
    node_id: str
    objtype: str
    canonical: bool


class ModuleEntry(NamedTuple):
    docname: str
    node_id: str
    synopsis: str
    platform: str
    deprecated: bool
```
### 68 - sphinx/domains/python.py:

Start line: 954, End line: 972

```python
class PyCurrentModule(SphinxDirective):
    """
    This directive is just to tell Sphinx that we're documenting
    stuff in module foo, but links to module foo won't lead here.
    """

    has_content = False
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = False
    option_spec: OptionSpec = {}

    def run(self) -> List[Node]:
        modname = self.arguments[0].strip()
        if modname == 'None':
            self.env.ref_context.pop('py:module', None)
        else:
            self.env.ref_context['py:module'] = modname
        return []
```
