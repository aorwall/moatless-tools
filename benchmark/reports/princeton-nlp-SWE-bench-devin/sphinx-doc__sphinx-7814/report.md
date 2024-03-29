# sphinx-doc__sphinx-7814

| **sphinx-doc/sphinx** | `55fc097833ee1e0efc689ddc85bd2af2e77f4af7` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | - |
| **Avg pos** | - |
| **Min pos** | - |
| **Max pos** | - |
| **Top file pos** | 1 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sphinx/domains/python.py b/sphinx/domains/python.py
--- a/sphinx/domains/python.py
+++ b/sphinx/domains/python.py
@@ -623,7 +623,8 @@ def handle_signature(self, sig: str, signode: desc_signature) -> Tuple[str, str]
 
         typ = self.options.get('type')
         if typ:
-            signode += addnodes.desc_annotation(typ, '', nodes.Text(': '), type_to_xref(typ))
+            annotations = _parse_annotation(typ)
+            signode += addnodes.desc_annotation(typ, '', nodes.Text(': '), *annotations)
 
         value = self.options.get('value')
         if value:
@@ -868,7 +869,8 @@ def handle_signature(self, sig: str, signode: desc_signature) -> Tuple[str, str]
 
         typ = self.options.get('type')
         if typ:
-            signode += addnodes.desc_annotation(typ, '', nodes.Text(': '), type_to_xref(typ))
+            annotations = _parse_annotation(typ)
+            signode += addnodes.desc_annotation(typ, '', nodes.Text(': '), *annotations)
 
         value = self.options.get('value')
         if value:

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sphinx/domains/python.py | 626 | 626 | - | 1 | -
| sphinx/domains/python.py | 871 | 871 | - | 1 | -


## Problem Statement

```
Warnings raised on variable and attribute type annotations
**Describe the bug**

autodoc signature for non-builtin types raises warning and thus fails nitpicking:

\`\`\`
/path/to/foo.py:docstring of foo.Foo.a:: WARNING: py:class reference target not found: Optional[str]
\`\`\`

**To Reproduce**

Steps to reproduce the behavior:

Create a file `foo.py` with the following content:
\`\`\`python
from typing import Optional


class Foo:
    a: Optional[str] = None
\`\`\`

Use sphinx-apidoc to generate an rst file, while enabling autodoc and intersphinx: `sphinx-apidoc --ext-autodoc --ext-intersphinx`

Make sure the `intersphinx_mapping` in the Sphinx `conf.py` contains `"python": ("https://docs.python.org/3.8/", None),`

Run `make html` with loud warnings and nitpicking: `SPHINXOPTS="-n -v -W --keep-going" make html`.

You will get an error message
\`\`\`
/path/to/foo.py:docstring of foo.Foo.a:: WARNING: py:class reference target not found: Optional[str]
\`\`\`

**Expected behavior**

I'd expect Sphinx to resolve the type annotation `Optional[str]` and possibly link both classes.

**Environment info**
- OS: Linux
- Python version: 3.8.3
- Sphinx version: 3.1.0
- Sphinx extensions:  sphinx.ext.autodoc, sphinx.ext.intersphinx

**Additional context**

I think the issue stems from the change in 88e8ebbe199c151a14d7df814807172f7565a073 which appears to try to lookup the entire type annotation as a single class.

Using `_parse_annotation()` instead of `type_to_xref()` solves this particular issue:
\`\`\`diff
diff --git a/sphinx/domains/python.py b/sphinx/domains/python.py
index fc1136ae2..6101de56a 100644
--- a/sphinx/domains/python.py
+++ b/sphinx/domains/python.py
@@ -623,7 +623,7 @@ class PyVariable(PyObject):
 
         typ = self.options.get('type')
         if typ:
-            signode += addnodes.desc_annotation(typ, '', nodes.Text(': '), type_to_xref(typ))
+            signode += addnodes.desc_annotation(typ, '', nodes.Text(': '), *_parse_annotation(typ))
 
         value = self.options.get('value')
         if value:
@@ -868,7 +868,7 @@ class PyAttribute(PyObject):
 
         typ = self.options.get('type')
         if typ:
-            signode += addnodes.desc_annotation(typ, '', nodes.Text(': '), type_to_xref(typ))
+            signode += addnodes.desc_annotation(typ, '', nodes.Text(': '), *_parse_annotation(typ))
 
         value = self.options.get('value')
         if value:
\`\`\`

However, it doesn't seem to work with custom classes. Take this snippet for example:
\`\`\`python
class Bar:
    i: int


class Foo:
    a: Bar
\`\`\`
This causes the following warning:
\`\`\`
foo.py:docstring of foo.Foo.a:: WARNING: py:class reference target not found: Bar
\`\`\`

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 sphinx/domains/python.py** | 1340 | 1378| 304 | 304 | 11637 | 
| 2 | 2 sphinx/deprecation.py | 11 | 34| 145 | 449 | 12239 | 
| 3 | **2 sphinx/domains/python.py** | 11 | 88| 598 | 1047 | 12239 | 
| 4 | 3 sphinx/transforms/post_transforms/__init__.py | 154 | 177| 279 | 1326 | 14234 | 
| 5 | 4 sphinx/util/inspect.py | 11 | 46| 257 | 1583 | 19986 | 
| 6 | 5 sphinx/ext/autodoc/directive.py | 9 | 49| 298 | 1881 | 21247 | 
| 7 | 6 sphinx/ext/autosummary/__init__.py | 412 | 438| 217 | 2098 | 27715 | 
| 8 | 7 sphinx/parsers.py | 11 | 28| 134 | 2232 | 28583 | 
| 9 | 8 sphinx/domains/std.py | 11 | 48| 323 | 2555 | 38792 | 
| 10 | 9 sphinx/directives/code.py | 9 | 31| 146 | 2701 | 42689 | 
| 11 | 9 sphinx/ext/autosummary/__init__.py | 55 | 106| 366 | 3067 | 42689 | 
| 12 | 10 sphinx/ext/autodoc/__init__.py | 1276 | 1522| 2333 | 5400 | 60757 | 
| 13 | 11 sphinx/ext/todo.py | 14 | 44| 207 | 5607 | 63474 | 
| 14 | 12 sphinx/writers/latex.py | 14 | 73| 453 | 6060 | 83178 | 
| 15 | 13 sphinx/ext/autodoc/typehints.py | 78 | 134| 460 | 6520 | 84198 | 
| 16 | 13 sphinx/ext/autodoc/__init__.py | 13 | 122| 791 | 7311 | 84198 | 
| 17 | 14 sphinx/ext/autosummary/generate.py | 20 | 60| 283 | 7594 | 89315 | 
| 18 | 15 sphinx/transforms/__init__.py | 252 | 272| 192 | 7786 | 92521 | 
| 19 | 15 sphinx/ext/autodoc/__init__.py | 1697 | 1836| 1215 | 9001 | 92521 | 
| 20 | 16 sphinx/directives/other.py | 9 | 40| 238 | 9239 | 95678 | 
| 21 | 16 sphinx/deprecation.py | 37 | 52| 145 | 9384 | 95678 | 
| 22 | 17 sphinx/io.py | 10 | 46| 275 | 9659 | 97371 | 
| 23 | 17 sphinx/ext/autodoc/__init__.py | 1684 | 1694| 128 | 9787 | 97371 | 
| 24 | 18 sphinx/util/nodes.py | 11 | 41| 227 | 10014 | 102792 | 
| 25 | **18 sphinx/domains/python.py** | 242 | 260| 214 | 10228 | 102792 | 
| 26 | 19 sphinx/directives/__init__.py | 269 | 307| 273 | 10501 | 105287 | 
| 27 | 19 sphinx/ext/autodoc/__init__.py | 1147 | 1240| 770 | 11271 | 105287 | 
| 28 | 20 sphinx/errors.py | 38 | 67| 213 | 11484 | 106029 | 
| 29 | 20 sphinx/ext/autodoc/__init__.py | 1593 | 1628| 272 | 11756 | 106029 | 
| 30 | 21 sphinx/util/cfamily.py | 11 | 80| 764 | 12520 | 109529 | 
| 31 | 21 sphinx/deprecation.py | 55 | 82| 261 | 12781 | 109529 | 
| 32 | 21 sphinx/domains/std.py | 51 | 66| 140 | 12921 | 109529 | 
| 33 | 22 sphinx/domains/math.py | 11 | 40| 207 | 13128 | 111039 | 
| 34 | 23 sphinx/ext/autodoc/type_comment.py | 117 | 139| 226 | 13354 | 112240 | 
| 35 | 23 sphinx/directives/other.py | 187 | 208| 141 | 13495 | 112240 | 
| 36 | 24 sphinx/events.py | 13 | 54| 352 | 13847 | 113258 | 
| 37 | 24 sphinx/ext/autodoc/type_comment.py | 11 | 37| 247 | 14094 | 113258 | 
| 38 | 24 sphinx/directives/__init__.py | 52 | 74| 206 | 14300 | 113258 | 
| 39 | 25 sphinx/ext/autodoc/mock.py | 71 | 93| 200 | 14500 | 114364 | 
| 40 | 25 sphinx/ext/autodoc/__init__.py | 926 | 947| 192 | 14692 | 114364 | 
| 41 | 25 sphinx/ext/autodoc/__init__.py | 913 | 924| 126 | 14818 | 114364 | 
| 42 | 25 sphinx/ext/autodoc/__init__.py | 1655 | 1682| 237 | 15055 | 114364 | 
| 43 | 26 sphinx/ext/autodoc/importer.py | 39 | 101| 582 | 15637 | 115913 | 
| 44 | **26 sphinx/domains/python.py** | 980 | 998| 143 | 15780 | 115913 | 
| 45 | 27 sphinx/util/pycompat.py | 11 | 53| 350 | 16130 | 116698 | 
| 46 | **27 sphinx/domains/python.py** | 300 | 308| 126 | 16256 | 116698 | 
| 47 | 28 sphinx/registry.py | 11 | 50| 307 | 16563 | 121244 | 
| 48 | 28 sphinx/ext/autodoc/typehints.py | 11 | 38| 200 | 16763 | 121244 | 
| 49 | 29 sphinx/roles.py | 11 | 47| 285 | 17048 | 126894 | 
| 50 | 30 sphinx/ext/doctest.py | 12 | 51| 286 | 17334 | 131849 | 
| 51 | 30 sphinx/registry.py | 211 | 233| 283 | 17617 | 131849 | 
| 52 | 30 sphinx/ext/autodoc/__init__.py | 1106 | 1126| 242 | 17859 | 131849 | 
| 53 | 31 sphinx/util/typing.py | 11 | 45| 255 | 18114 | 133859 | 
| 54 | 32 sphinx/domains/c.py | 2699 | 3478| 6630 | 24744 | 162556 | 
| 55 | 32 sphinx/ext/autodoc/__init__.py | 1963 | 1993| 247 | 24991 | 162556 | 


### Hint

```
We have a similar problem in the Trio project where we use annotations like "str or list", ThisType or None" or even "bytes-like" in a number of place. Here's an example: https://github.com/python-trio/trio/blob/dependabot/pip/sphinx-3.1.0/trio/_subprocess.py#L75-L96
To clarify a bit on the Trio issue: we don't expect sphinx to magically do anything with `bytes-like`, but currently you can use something like this in a Google-style docstring:

\`\`\`
    Attributes:
      args (str or list): The ``command`` passed at construction time,
          specifying the process to execute and its arguments.
\`\`\`

And with previous versions of Sphinx, it renders like this:

![image](https://user-images.githubusercontent.com/609896/84207125-73d55100-aa65-11ea-98e6-4b3b8f619be9.png)

https://trio.readthedocs.io/en/v0.15.1/reference-io.html#trio.Process.args

Notice that `str` and `list` are both hyperlinked appropriately.

So Sphinx used to be able to cope with this kind of "or" syntax, and if it can't anymore it's a regression.
This also occurs with built-in container classes and 'or'-types (in nitpick mode):
\`\`\`
WARNING: py:class reference target not found: tuple[str]
WARNING: py:class reference target not found: str or None
\`\`\`

Unfortunately this breaks my CI pipeline at the moment. Does anyone know a work-around other than disabling nitpick mode?

```

## Patch

```diff
diff --git a/sphinx/domains/python.py b/sphinx/domains/python.py
--- a/sphinx/domains/python.py
+++ b/sphinx/domains/python.py
@@ -623,7 +623,8 @@ def handle_signature(self, sig: str, signode: desc_signature) -> Tuple[str, str]
 
         typ = self.options.get('type')
         if typ:
-            signode += addnodes.desc_annotation(typ, '', nodes.Text(': '), type_to_xref(typ))
+            annotations = _parse_annotation(typ)
+            signode += addnodes.desc_annotation(typ, '', nodes.Text(': '), *annotations)
 
         value = self.options.get('value')
         if value:
@@ -868,7 +869,8 @@ def handle_signature(self, sig: str, signode: desc_signature) -> Tuple[str, str]
 
         typ = self.options.get('type')
         if typ:
-            signode += addnodes.desc_annotation(typ, '', nodes.Text(': '), type_to_xref(typ))
+            annotations = _parse_annotation(typ)
+            signode += addnodes.desc_annotation(typ, '', nodes.Text(': '), *annotations)
 
         value = self.options.get('value')
         if value:

```

## Test Patch

```diff
diff --git a/tests/test_domain_py.py b/tests/test_domain_py.py
--- a/tests/test_domain_py.py
+++ b/tests/test_domain_py.py
@@ -679,7 +679,7 @@ def test_pyattribute(app):
     text = (".. py:class:: Class\n"
             "\n"
             "   .. py:attribute:: attr\n"
-            "      :type: str\n"
+            "      :type: Optional[str]\n"
             "      :value: ''\n")
     domain = app.env.get_domain('py')
     doctree = restructuredtext.parse(app, text)
@@ -692,7 +692,10 @@ def test_pyattribute(app):
                 entries=[('single', 'attr (Class attribute)', 'Class.attr', '', None)])
     assert_node(doctree[1][1][1], ([desc_signature, ([desc_name, "attr"],
                                                      [desc_annotation, (": ",
-                                                                        [pending_xref, "str"])],
+                                                                        [pending_xref, "Optional"],
+                                                                        [desc_sig_punctuation, "["],
+                                                                        [pending_xref, "str"],
+                                                                        [desc_sig_punctuation, "]"])],
                                                      [desc_annotation, " = ''"])],
                                    [desc_content, ()]))
     assert 'Class.attr' in domain.objects

```


## Code snippets

### 1 - sphinx/domains/python.py:

Start line: 1340, End line: 1378

```python
def builtin_resolver(app: Sphinx, env: BuildEnvironment,
                     node: pending_xref, contnode: Element) -> Element:
    """Do not emit nitpicky warnings for built-in types."""
    def istyping(s: str) -> bool:
        if s.startswith('typing.'):
            s = s.split('.', 1)[1]

        return s in typing.__all__  # type: ignore

    if node.get('refdomain') != 'py':
        return None
    elif node.get('reftype') in ('class', 'obj') and node.get('reftarget') == 'None':
        return contnode
    elif node.get('reftype') in ('class', 'exc'):
        reftarget = node.get('reftarget')
        if inspect.isclass(getattr(builtins, reftarget, None)):
            # built-in class
            return contnode
        elif istyping(reftarget):
            # typing class
            return contnode

    return None


def setup(app: Sphinx) -> Dict[str, Any]:
    app.setup_extension('sphinx.directives')

    app.add_domain(PythonDomain)
    app.connect('object-description-transform', filter_meta_fields)
    app.connect('missing-reference', builtin_resolver, priority=900)

    return {
        'version': 'builtin',
        'env_version': 2,
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
```
### 2 - sphinx/deprecation.py:

Start line: 11, End line: 34

```python
import sys
import warnings
from importlib import import_module
from typing import Any, Dict

if False:
    # For type annotation
    from typing import Type  # for python3.5.1


class RemovedInSphinx40Warning(DeprecationWarning):
    pass


class RemovedInSphinx50Warning(PendingDeprecationWarning):
    pass


RemovedInNextVersionWarning = RemovedInSphinx40Warning


def deprecated_alias(modname: str, objects: Dict, warning: "Type[Warning]") -> None:
    module = import_module(modname)
    sys.modules[modname] = _ModuleWrapper(module, modname, objects, warning)  # type: ignore
```
### 3 - sphinx/domains/python.py:

Start line: 11, End line: 88

```python
import builtins
import inspect
import re
import typing
import warnings
from inspect import Parameter
from typing import Any, Dict, Iterable, Iterator, List, NamedTuple, Tuple
from typing import cast

from docutils import nodes
from docutils.nodes import Element, Node
from docutils.parsers.rst import directives

from sphinx import addnodes
from sphinx.addnodes import pending_xref, desc_signature
from sphinx.application import Sphinx
from sphinx.builders import Builder
from sphinx.deprecation import RemovedInSphinx40Warning, RemovedInSphinx50Warning
from sphinx.directives import ObjectDescription
from sphinx.domains import Domain, ObjType, Index, IndexEntry
from sphinx.environment import BuildEnvironment
from sphinx.locale import _, __
from sphinx.pycode.ast import ast, parse as ast_parse
from sphinx.roles import XRefRole
from sphinx.util import logging
from sphinx.util.docfields import Field, GroupedField, TypedField
from sphinx.util.docutils import SphinxDirective
from sphinx.util.inspect import signature_from_str
from sphinx.util.nodes import make_id, make_refnode
from sphinx.util.typing import TextlikeNode

if False:
    # For type annotation
    from typing import Type  # for python3.5.1


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

ObjectEntry = NamedTuple('ObjectEntry', [('docname', str),
                                         ('node_id', str),
                                         ('objtype', str)])
ModuleEntry = NamedTuple('ModuleEntry', [('docname', str),
                                         ('node_id', str),
                                         ('synopsis', str),
                                         ('platform', str),
                                         ('deprecated', bool)])


def type_to_xref(text: str) -> addnodes.pending_xref:
    """Convert a type string to a cross reference node."""
    if text == 'None':
        reftype = 'obj'
    else:
        reftype = 'class'

    return pending_xref('', nodes.Text(text),
                        refdomain='py', reftype=reftype, reftarget=text)
```
### 4 - sphinx/transforms/post_transforms/__init__.py:

Start line: 154, End line: 177

```python
class ReferencesResolver(SphinxPostTransform):

    def warn_missing_reference(self, refdoc: str, typ: str, target: str,
                               node: pending_xref, domain: Domain) -> None:
        warn = node.get('refwarn')
        if self.config.nitpicky:
            warn = True
            if self.config.nitpick_ignore:
                dtype = '%s:%s' % (domain.name, typ) if domain else typ
                if (dtype, target) in self.config.nitpick_ignore:
                    warn = False
                # for "std" types also try without domain name
                if (not domain or domain.name == 'std') and \
                   (typ, target) in self.config.nitpick_ignore:
                    warn = False
        if not warn:
            return
        if domain and typ in domain.dangling_warnings:
            msg = domain.dangling_warnings[typ]
        elif node.get('refdomain', 'std') not in ('', 'std'):
            msg = (__('%s:%s reference target not found: %%(target)s') %
                   (node['refdomain'], typ))
        else:
            msg = __('%r reference target not found: %%(target)s') % typ
        logger.warning(msg % {'target': target},
                       location=node, type='ref', subtype=typ)
```
### 5 - sphinx/util/inspect.py:

Start line: 11, End line: 46

```python
import builtins
import enum
import inspect
import re
import sys
import types
import typing
import warnings
from functools import partial, partialmethod
from inspect import (  # NOQA
    Parameter, isclass, ismethod, ismethoddescriptor
)
from io import StringIO
from typing import Any, Callable, Mapping, List, Optional, Tuple
from typing import cast

from sphinx.deprecation import RemovedInSphinx40Warning, RemovedInSphinx50Warning
from sphinx.pycode.ast import ast  # for py35-37
from sphinx.pycode.ast import unparse as ast_unparse
from sphinx.util import logging
from sphinx.util.typing import stringify as stringify_annotation

if sys.version_info > (3, 7):
    from types import (
        ClassMethodDescriptorType,
        MethodDescriptorType,
        WrapperDescriptorType
    )
else:
    ClassMethodDescriptorType = type(object.__init__)
    MethodDescriptorType = type(str.join)
    WrapperDescriptorType = type(dict.__dict__['fromkeys'])

logger = logging.getLogger(__name__)

memory_address_re = re.compile(r' at 0x[0-9a-f]{8,16}(?=>)', re.IGNORECASE)
```
### 6 - sphinx/ext/autodoc/directive.py:

Start line: 9, End line: 49

```python
import warnings
from typing import Any, Callable, Dict, List, Set

from docutils import nodes
from docutils.nodes import Element, Node
from docutils.parsers.rst.states import RSTState, Struct
from docutils.statemachine import StringList
from docutils.utils import Reporter, assemble_option_dict

from sphinx.config import Config
from sphinx.deprecation import RemovedInSphinx40Warning
from sphinx.environment import BuildEnvironment
from sphinx.ext.autodoc import Documenter, Options
from sphinx.util import logging
from sphinx.util.docutils import SphinxDirective, switch_source_input
from sphinx.util.nodes import nested_parse_with_titles

if False:
    # For type annotation
    from typing import Type  # for python3.5.1


logger = logging.getLogger(__name__)


# common option names for autodoc directives
AUTODOC_DEFAULT_OPTIONS = ['members', 'undoc-members', 'inherited-members',
                           'show-inheritance', 'private-members', 'special-members',
                           'ignore-module-all', 'exclude-members', 'member-order',
                           'imported-members']


class DummyOptionSpec(dict):
    """An option_spec allows any options."""

    def __bool__(self) -> bool:
        """Behaves like some options are defined."""
        return True

    def __getitem__(self, key: str) -> Callable[[str], str]:
        return lambda x: x
```
### 7 - sphinx/ext/autosummary/__init__.py:

Start line: 412, End line: 438

```python
class Autosummary(SphinxDirective):

    def warn(self, msg: str) -> None:
        warnings.warn('Autosummary.warn() is deprecated',
                      RemovedInSphinx40Warning, stacklevel=2)
        logger.warning(msg)

    @property
    def genopt(self) -> Options:
        warnings.warn('Autosummary.genopt is deprecated',
                      RemovedInSphinx40Warning, stacklevel=2)
        return self.bridge.genopt

    @property
    def warnings(self) -> List[Node]:
        warnings.warn('Autosummary.warnings is deprecated',
                      RemovedInSphinx40Warning, stacklevel=2)
        return []

    @property
    def result(self) -> StringList:
        warnings.warn('Autosummary.result is deprecated',
                      RemovedInSphinx40Warning, stacklevel=2)
        return self.bridge.result


def strip_arg_typehint(s: str) -> str:
    """Strip a type hint from argument definition."""
    return s.split(':')[0].strip()
```
### 8 - sphinx/parsers.py:

Start line: 11, End line: 28

```python
import warnings
from typing import Any, Dict, List, Union

import docutils.parsers
import docutils.parsers.rst
from docutils import nodes
from docutils.parsers.rst import states
from docutils.statemachine import StringList
from docutils.transforms.universal import SmartQuotes

from sphinx.deprecation import RemovedInSphinx50Warning
from sphinx.util.rst import append_epilog, prepend_prolog

if False:
    # For type annotation
    from docutils.transforms import Transform  # NOQA
    from typing import Type  # NOQA # for python3.5.1
    from sphinx.application import Sphinx
```
### 9 - sphinx/domains/std.py:

Start line: 11, End line: 48

```python
import re
import unicodedata
import warnings
from copy import copy
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple, Union
from typing import cast

from docutils import nodes
from docutils.nodes import Element, Node, system_message
from docutils.parsers.rst import Directive, directives
from docutils.statemachine import StringList

from sphinx import addnodes
from sphinx.addnodes import desc_signature, pending_xref
from sphinx.deprecation import RemovedInSphinx40Warning, RemovedInSphinx50Warning
from sphinx.directives import ObjectDescription
from sphinx.domains import Domain, ObjType
from sphinx.locale import _, __
from sphinx.roles import XRefRole
from sphinx.util import ws_re, logging, docname_join
from sphinx.util.docutils import SphinxDirective
from sphinx.util.nodes import clean_astext, make_id, make_refnode
from sphinx.util.typing import RoleFunction

if False:
    # For type annotation
    from typing import Type  # for python3.5.1
    from sphinx.application import Sphinx
    from sphinx.builders import Builder
    from sphinx.environment import BuildEnvironment

logger = logging.getLogger(__name__)


# RE for option descriptions
option_desc_re = re.compile(r'((?:/|--|-|\+)?[^\s=[]+)(=?\s*.*)')
# RE for grammar tokens
token_re = re.compile(r'`(\w+)`', re.U)
```
### 10 - sphinx/directives/code.py:

Start line: 9, End line: 31

```python
import sys
import warnings
from difflib import unified_diff
from typing import Any, Dict, List, Tuple

from docutils import nodes
from docutils.nodes import Element, Node
from docutils.parsers.rst import directives
from docutils.statemachine import StringList

from sphinx import addnodes
from sphinx.config import Config
from sphinx.deprecation import RemovedInSphinx40Warning
from sphinx.locale import __
from sphinx.util import logging
from sphinx.util import parselinenos
from sphinx.util.docutils import SphinxDirective

if False:
    # For type annotation
    from sphinx.application import Sphinx

logger = logging.getLogger(__name__)
```
### 25 - sphinx/domains/python.py:

Start line: 242, End line: 260

```python
# This override allows our inline type specifiers to behave like :class: link
# when it comes to handling "." and "~" prefixes.
class PyXrefMixin:
    def make_xref(self, rolename: str, domain: str, target: str,
                  innernode: "Type[TextlikeNode]" = nodes.emphasis,
                  contnode: Node = None, env: BuildEnvironment = None) -> Node:
        result = super().make_xref(rolename, domain, target,  # type: ignore
                                   innernode, contnode, env)
        result['refspecific'] = True
        if target.startswith(('.', '~')):
            prefix, result['reftarget'] = target[0], target[1:]
            if prefix == '.':
                text = target[1:]
            elif prefix == '~':
                text = target.split('.')[-1]
            for node in result.traverse(nodes.Text):
                node.parent[node.parent.index(node)] = nodes.Text(text)
                break
        return result
```
### 44 - sphinx/domains/python.py:

Start line: 980, End line: 998

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
    option_spec = {}  # type: Dict

    def run(self) -> List[Node]:
        modname = self.arguments[0].strip()
        if modname == 'None':
            self.env.ref_context.pop('py:module', None)
        else:
            self.env.ref_context['py:module'] = modname
        return []
```
### 46 - sphinx/domains/python.py:

Start line: 300, End line: 308

```python
class PyTypedField(PyXrefMixin, TypedField):
    def make_xref(self, rolename: str, domain: str, target: str,
                  innernode: "Type[TextlikeNode]" = nodes.emphasis,
                  contnode: Node = None, env: BuildEnvironment = None) -> Node:
        if rolename == 'class' and target == 'None':
            # None is not a type, so use obj role instead.
            rolename = 'obj'

        return super().make_xref(rolename, domain, target, innernode, contnode, env)
```
