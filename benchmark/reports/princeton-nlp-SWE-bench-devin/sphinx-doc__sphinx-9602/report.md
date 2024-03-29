# sphinx-doc__sphinx-9602

| **sphinx-doc/sphinx** | `6c38f68dae221e8cfc70c137974b8b88bd3baaab` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | - |
| **Avg pos** | - |
| **Min pos** | - |
| **Max pos** | - |
| **Top file pos** | 2 |
| **Missing snippets** | 3 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sphinx/domains/python.py b/sphinx/domains/python.py
--- a/sphinx/domains/python.py
+++ b/sphinx/domains/python.py
@@ -123,7 +123,7 @@ def unparse(node: ast.AST) -> List[Node]:
             if node.value is Ellipsis:
                 return [addnodes.desc_sig_punctuation('', "...")]
             else:
-                return [nodes.Text(node.value)]
+                return [nodes.Text(repr(node.value))]
         elif isinstance(node, ast.Expr):
             return unparse(node.value)
         elif isinstance(node, ast.Index):
@@ -149,6 +149,12 @@ def unparse(node: ast.AST) -> List[Node]:
             result.append(addnodes.desc_sig_punctuation('', '['))
             result.extend(unparse(node.slice))
             result.append(addnodes.desc_sig_punctuation('', ']'))
+
+            # Wrap the Text nodes inside brackets by literal node if the subscript is a Literal
+            if result[0] in ('Literal', 'typing.Literal'):
+                for i, subnode in enumerate(result[1:], start=1):
+                    if isinstance(subnode, nodes.Text):
+                        result[i] = nodes.literal('', '', subnode)
             return result
         elif isinstance(node, ast.Tuple):
             if node.elts:
@@ -179,7 +185,9 @@ def unparse(node: ast.AST) -> List[Node]:
         tree = ast_parse(annotation)
         result = unparse(tree)
         for i, node in enumerate(result):
-            if isinstance(node, nodes.Text) and node.strip():
+            if isinstance(node, nodes.literal):
+                result[i] = node[0]
+            elif isinstance(node, nodes.Text) and node.strip():
                 result[i] = type_to_xref(str(node), env)
         return result
     except SyntaxError:

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sphinx/domains/python.py | 126 | 126 | - | 2 | -
| sphinx/domains/python.py | 152 | 152 | - | 2 | -
| sphinx/domains/python.py | 182 | 182 | - | 2 | -


## Problem Statement

```
Nitpick flags Literal annotation values as missing py:class
### Describe the bug

When a value is present in a type annotation as `Literal`, sphinx will treat the value as a `py:class`. With nitpick enabled, values like `Literal[True]` end up failing, because `True` is not a class.

This is a problem for builds which want to use `-n -W` to catch doc errors.

### How to Reproduce

Setup a simple function which uses Literal, then attempt to autodoc it. e.g.
\`\`\`python
import typing
@typing.overload
def foo(x: "typing.Literal[True]") -> int: ...
@typing.overload
def foo(x: "typing.Literal[False]") -> str: ...
def foo(x: bool):
    """a func"""
    return 1 if x else "foo"
\`\`\`

I've pushed an example [failing project](https://github.com/sirosen/repro/tree/master/sphinxdoc/literal) to [my repro repo](https://github.com/sirosen/repro). Just run `./doc.sh` with `sphinx-build` available to see the failing build.

### Expected behavior

`Literal[True]` (or whatever literal value) should be present in the type annotation but should not trigger the nitpick warning.

### Your project

https://github.com/sirosen/repro/tree/master/sphinxdoc/literal

### Screenshots

_No response_

### OS

Linux

### Python version

3.8, 3.9

### Sphinx version

4.1.2

### Sphinx extensions

autodoc

### Extra tools

_No response_

### Additional context

_No response_

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 sphinx/directives/code.py | 372 | 405| 288 | 288 | 3851 | 
| 2 | **2 sphinx/domains/python.py** | 1383 | 1405| 211 | 499 | 15974 | 
| 3 | **2 sphinx/domains/python.py** | 11 | 80| 518 | 1017 | 15974 | 
| 4 | 2 sphinx/directives/code.py | 407 | 470| 642 | 1659 | 15974 | 
| 5 | 3 sphinx/util/inspect.py | 11 | 47| 290 | 1949 | 22676 | 
| 6 | 4 sphinx/domains/changeset.py | 49 | 107| 516 | 2465 | 23930 | 
| 7 | 5 utils/checks.py | 33 | 109| 545 | 3010 | 24836 | 
| 8 | 6 sphinx/ext/autodoc/__init__.py | 13 | 114| 788 | 3798 | 48671 | 
| 9 | 6 sphinx/directives/code.py | 180 | 209| 248 | 4046 | 48671 | 
| 10 | 7 sphinx/ext/doctest.py | 12 | 43| 227 | 4273 | 53689 | 
| 11 | 8 sphinx/ext/napoleon/docstring.py | 13 | 67| 578 | 4851 | 64689 | 
| 12 | 9 sphinx/testing/util.py | 10 | 45| 270 | 5121 | 66440 | 
| 13 | 9 sphinx/directives/code.py | 251 | 267| 162 | 5283 | 66440 | 
| 14 | 10 sphinx/directives/__init__.py | 11 | 47| 253 | 5536 | 68692 | 
| 15 | 11 sphinx/ext/autodoc/type_comment.py | 11 | 35| 239 | 5775 | 69916 | 
| 16 | 12 sphinx/environment/__init__.py | 11 | 82| 508 | 6283 | 75413 | 
| 17 | 13 sphinx/application.py | 13 | 58| 361 | 6644 | 87035 | 
| 18 | 14 sphinx/directives/other.py | 9 | 38| 229 | 6873 | 90162 | 
| 19 | 14 sphinx/ext/autodoc/__init__.py | 2397 | 2432| 313 | 7186 | 90162 | 
| 20 | 14 sphinx/ext/autodoc/__init__.py | 2569 | 2592| 229 | 7415 | 90162 | 
| 21 | 15 sphinx/ext/autosummary/generate.py | 20 | 53| 257 | 7672 | 95461 | 
| 22 | 16 sphinx/errors.py | 77 | 134| 297 | 7969 | 96257 | 
| 23 | 17 sphinx/ext/autodoc/importer.py | 11 | 40| 211 | 8180 | 98719 | 
| 24 | 18 sphinx/cmd/quickstart.py | 11 | 119| 756 | 8936 | 104288 | 
| 25 | 19 doc/conf.py | 1 | 82| 731 | 9667 | 105752 | 
| 26 | 20 sphinx/highlighting.py | 11 | 68| 620 | 10287 | 107298 | 
| 27 | 21 sphinx/builders/__init__.py | 11 | 48| 302 | 10589 | 112648 | 
| 28 | 21 sphinx/directives/code.py | 269 | 291| 243 | 10832 | 112648 | 
| 29 | **21 sphinx/domains/python.py** | 287 | 321| 382 | 11214 | 112648 | 
| 30 | 22 sphinx/registry.py | 11 | 51| 314 | 11528 | 117288 | 
| 31 | 23 sphinx/ext/autodoc/typehints.py | 130 | 185| 454 | 11982 | 118736 | 
| 32 | 24 sphinx/pycode/ast.py | 11 | 44| 214 | 12196 | 120823 | 
| 33 | 25 sphinx/ext/autodoc/directive.py | 9 | 47| 310 | 12506 | 122274 | 
| 34 | 26 sphinx/parsers.py | 11 | 26| 113 | 12619 | 123121 | 
| 35 | 27 sphinx/domains/std.py | 11 | 46| 311 | 12930 | 133418 | 
| 36 | 28 sphinx/io.py | 10 | 39| 234 | 13164 | 134824 | 
| 37 | 29 sphinx/config.py | 11 | 55| 325 | 13489 | 139284 | 
| 38 | 29 sphinx/config.py | 408 | 460| 474 | 13963 | 139284 | 
| 39 | 30 setup.py | 176 | 252| 638 | 14601 | 141044 | 
| 40 | 30 sphinx/directives/code.py | 351 | 369| 169 | 14770 | 141044 | 
| 41 | 30 sphinx/ext/doctest.py | 481 | 504| 264 | 15034 | 141044 | 
| 42 | 30 sphinx/directives/code.py | 227 | 249| 196 | 15230 | 141044 | 
| 43 | 31 sphinx/transforms/post_transforms/code.py | 114 | 143| 208 | 15438 | 142061 | 
| 44 | 31 sphinx/directives/code.py | 324 | 349| 193 | 15631 | 142061 | 
| 45 | 32 sphinx/util/typing.py | 11 | 73| 454 | 16085 | 146492 | 
| 46 | 32 sphinx/config.py | 91 | 150| 740 | 16825 | 146492 | 
| 47 | 33 sphinx/ext/autodoc/mock.py | 11 | 69| 452 | 17277 | 147840 | 
| 48 | 34 sphinx/domains/rst.py | 11 | 32| 170 | 17447 | 150319 | 
| 49 | 35 sphinx/util/__init__.py | 11 | 64| 445 | 17892 | 155197 | 
| 50 | 36 sphinx/writers/latex.py | 14 | 70| 441 | 18333 | 174587 | 
| 51 | 36 sphinx/ext/autodoc/__init__.py | 2789 | 2832| 528 | 18861 | 174587 | 
| 52 | 37 sphinx/domains/citation.py | 11 | 29| 125 | 18986 | 175868 | 
| 53 | 38 sphinx/util/cfamily.py | 11 | 62| 749 | 19735 | 179327 | 
| 54 | 39 sphinx/transforms/__init__.py | 11 | 44| 231 | 19966 | 182498 | 
| 55 | 40 sphinx/builders/dummy.py | 11 | 53| 223 | 20189 | 182774 | 
| 56 | 40 sphinx/ext/doctest.py | 153 | 197| 292 | 20481 | 182774 | 
| 57 | 40 sphinx/directives/code.py | 293 | 322| 212 | 20693 | 182774 | 
| 58 | 41 sphinx/builders/latex/__init__.py | 11 | 42| 331 | 21024 | 188486 | 
| 59 | 41 sphinx/domains/std.py | 1113 | 1138| 209 | 21233 | 188486 | 
| 60 | 41 sphinx/application.py | 332 | 381| 410 | 21643 | 188486 | 
| 61 | 42 sphinx/ext/inheritance_diagram.py | 38 | 67| 243 | 21886 | 192352 | 
| 62 | 42 sphinx/ext/autodoc/__init__.py | 2526 | 2545| 224 | 22110 | 192352 | 
| 63 | 42 setup.py | 1 | 78| 478 | 22588 | 192352 | 
| 64 | 43 sphinx/domains/javascript.py | 11 | 33| 199 | 22787 | 196416 | 
| 65 | 43 sphinx/directives/code.py | 211 | 225| 159 | 22946 | 196416 | 
| 66 | 43 sphinx/util/typing.py | 398 | 461| 638 | 23584 | 196416 | 


## Patch

```diff
diff --git a/sphinx/domains/python.py b/sphinx/domains/python.py
--- a/sphinx/domains/python.py
+++ b/sphinx/domains/python.py
@@ -123,7 +123,7 @@ def unparse(node: ast.AST) -> List[Node]:
             if node.value is Ellipsis:
                 return [addnodes.desc_sig_punctuation('', "...")]
             else:
-                return [nodes.Text(node.value)]
+                return [nodes.Text(repr(node.value))]
         elif isinstance(node, ast.Expr):
             return unparse(node.value)
         elif isinstance(node, ast.Index):
@@ -149,6 +149,12 @@ def unparse(node: ast.AST) -> List[Node]:
             result.append(addnodes.desc_sig_punctuation('', '['))
             result.extend(unparse(node.slice))
             result.append(addnodes.desc_sig_punctuation('', ']'))
+
+            # Wrap the Text nodes inside brackets by literal node if the subscript is a Literal
+            if result[0] in ('Literal', 'typing.Literal'):
+                for i, subnode in enumerate(result[1:], start=1):
+                    if isinstance(subnode, nodes.Text):
+                        result[i] = nodes.literal('', '', subnode)
             return result
         elif isinstance(node, ast.Tuple):
             if node.elts:
@@ -179,7 +185,9 @@ def unparse(node: ast.AST) -> List[Node]:
         tree = ast_parse(annotation)
         result = unparse(tree)
         for i, node in enumerate(result):
-            if isinstance(node, nodes.Text) and node.strip():
+            if isinstance(node, nodes.literal):
+                result[i] = node[0]
+            elif isinstance(node, nodes.Text) and node.strip():
                 result[i] = type_to_xref(str(node), env)
         return result
     except SyntaxError:

```

## Test Patch

```diff
diff --git a/tests/test_domain_py.py b/tests/test_domain_py.py
--- a/tests/test_domain_py.py
+++ b/tests/test_domain_py.py
@@ -342,6 +342,27 @@ def test_parse_annotation(app):
     assert_node(doctree[0], pending_xref, refdomain="py", reftype="obj", reftarget="None")
 
 
+@pytest.mark.skipif(sys.version_info < (3, 8), reason='python 3.8+ is required.')
+def test_parse_annotation_Literal(app):
+    doctree = _parse_annotation("Literal[True, False]", app.env)
+    assert_node(doctree, ([pending_xref, "Literal"],
+                          [desc_sig_punctuation, "["],
+                          "True",
+                          [desc_sig_punctuation, ", "],
+                          "False",
+                          [desc_sig_punctuation, "]"]))
+
+    doctree = _parse_annotation("typing.Literal[0, 1, 'abc']", app.env)
+    assert_node(doctree, ([pending_xref, "typing.Literal"],
+                          [desc_sig_punctuation, "["],
+                          "0",
+                          [desc_sig_punctuation, ", "],
+                          "1",
+                          [desc_sig_punctuation, ", "],
+                          "'abc'",
+                          [desc_sig_punctuation, "]"]))
+
+
 def test_pyfunction_signature(app):
     text = ".. py:function:: hello(name: str) -> str"
     doctree = restructuredtext.parse(app, text)

```


## Code snippets

### 1 - sphinx/directives/code.py:

Start line: 372, End line: 405

```python
class LiteralInclude(SphinxDirective):
    """
    Like ``.. include:: :literal:``, but only warns if the include file is
    not found, and does not raise errors.  Also has several options for
    selecting what to include.
    """

    has_content = False
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = True
    option_spec: OptionSpec = {
        'dedent': optional_int,
        'linenos': directives.flag,
        'lineno-start': int,
        'lineno-match': directives.flag,
        'tab-width': int,
        'language': directives.unchanged_required,
        'force': directives.flag,
        'encoding': directives.encoding,
        'pyobject': directives.unchanged_required,
        'lines': directives.unchanged_required,
        'start-after': directives.unchanged_required,
        'end-before': directives.unchanged_required,
        'start-at': directives.unchanged_required,
        'end-at': directives.unchanged_required,
        'prepend': directives.unchanged_required,
        'append': directives.unchanged_required,
        'emphasize-lines': directives.unchanged_required,
        'caption': directives.unchanged,
        'class': directives.class_option,
        'name': directives.unchanged,
        'diff': directives.unchanged_required,
    }
```
### 2 - sphinx/domains/python.py:

Start line: 1383, End line: 1405

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
```
### 3 - sphinx/domains/python.py:

Start line: 11, End line: 80

```python
import builtins
import inspect
import re
import sys
import typing
import warnings
from inspect import Parameter
from typing import Any, Dict, Iterable, Iterator, List, NamedTuple, Optional, Tuple, Type, cast

from docutils import nodes
from docutils.nodes import Element, Node
from docutils.parsers.rst import directives
from docutils.parsers.rst.states import Inliner

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
    aliased: bool


class ModuleEntry(NamedTuple):
    docname: str
    node_id: str
    synopsis: str
    platform: str
    deprecated: bool
```
### 4 - sphinx/directives/code.py:

Start line: 407, End line: 470

```python
class LiteralInclude(SphinxDirective):

    def run(self) -> List[Node]:
        document = self.state.document
        if not document.settings.file_insertion_enabled:
            return [document.reporter.warning('File insertion disabled',
                                              line=self.lineno)]
        # convert options['diff'] to absolute path
        if 'diff' in self.options:
            _, path = self.env.relfn2path(self.options['diff'])
            self.options['diff'] = path

        try:
            location = self.state_machine.get_source_and_line(self.lineno)
            rel_filename, filename = self.env.relfn2path(self.arguments[0])
            self.env.note_dependency(rel_filename)

            reader = LiteralIncludeReader(filename, self.options, self.config)
            text, lines = reader.read(location=location)

            retnode: Element = nodes.literal_block(text, text, source=filename)
            retnode['force'] = 'force' in self.options
            self.set_source_info(retnode)
            if self.options.get('diff'):  # if diff is set, set udiff
                retnode['language'] = 'udiff'
            elif 'language' in self.options:
                retnode['language'] = self.options['language']
            if ('linenos' in self.options or 'lineno-start' in self.options or
                    'lineno-match' in self.options):
                retnode['linenos'] = True
            retnode['classes'] += self.options.get('class', [])
            extra_args = retnode['highlight_args'] = {}
            if 'emphasize-lines' in self.options:
                hl_lines = parselinenos(self.options['emphasize-lines'], lines)
                if any(i >= lines for i in hl_lines):
                    logger.warning(__('line number spec is out of range(1-%d): %r') %
                                   (lines, self.options['emphasize-lines']),
                                   location=location)
                extra_args['hl_lines'] = [x + 1 for x in hl_lines if x < lines]
            extra_args['linenostart'] = reader.lineno_start

            if 'caption' in self.options:
                caption = self.options['caption'] or self.arguments[0]
                retnode = container_wrapper(self, retnode, caption)

            # retnode will be note_implicit_target that is linked from caption and numref.
            # when options['name'] is provided, it should be primary ID.
            self.add_name(retnode)

            return [retnode]
        except Exception as exc:
            return [document.reporter.warning(exc, line=self.lineno)]


def setup(app: "Sphinx") -> Dict[str, Any]:
    directives.register_directive('highlight', Highlight)
    directives.register_directive('code-block', CodeBlock)
    directives.register_directive('sourcecode', CodeBlock)
    directives.register_directive('literalinclude', LiteralInclude)

    return {
        'version': 'builtin',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
```
### 5 - sphinx/util/inspect.py:

Start line: 11, End line: 47

```python
import builtins
import contextlib
import enum
import inspect
import re
import sys
import types
import typing
import warnings
from functools import partial, partialmethod
from importlib import import_module
from inspect import Parameter, isclass, ismethod, ismethoddescriptor, ismodule  # NOQA
from io import StringIO
from types import ModuleType
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Type, cast

from sphinx.deprecation import RemovedInSphinx50Warning
from sphinx.pycode.ast import ast  # for py36-37
from sphinx.pycode.ast import unparse as ast_unparse
from sphinx.util import logging
from sphinx.util.typing import ForwardRef
from sphinx.util.typing import stringify as stringify_annotation

if sys.version_info > (3, 7):
    from types import ClassMethodDescriptorType, MethodDescriptorType, WrapperDescriptorType
else:
    ClassMethodDescriptorType = type(object.__init__)
    MethodDescriptorType = type(str.join)
    WrapperDescriptorType = type(dict.__dict__['fromkeys'])

if False:
    # For type annotation
    from typing import Type  # NOQA

logger = logging.getLogger(__name__)

memory_address_re = re.compile(r' at 0x[0-9a-f]{8,16}(?=>)', re.IGNORECASE)
```
### 6 - sphinx/domains/changeset.py:

Start line: 49, End line: 107

```python
class VersionChange(SphinxDirective):
    """
    Directive to describe a change/addition/deprecation in a specific version.
    """
    has_content = True
    required_arguments = 1
    optional_arguments = 1
    final_argument_whitespace = True
    option_spec: OptionSpec = {}

    def run(self) -> List[Node]:
        node = addnodes.versionmodified()
        node.document = self.state.document
        self.set_source_info(node)
        node['type'] = self.name
        node['version'] = self.arguments[0]
        text = versionlabels[self.name] % self.arguments[0]
        if len(self.arguments) == 2:
            inodes, messages = self.state.inline_text(self.arguments[1],
                                                      self.lineno + 1)
            para = nodes.paragraph(self.arguments[1], '', *inodes, translatable=False)
            self.set_source_info(para)
            node.append(para)
        else:
            messages = []
        if self.content:
            self.state.nested_parse(self.content, self.content_offset, node)
        classes = ['versionmodified', versionlabel_classes[self.name]]
        if len(node) > 0 and isinstance(node[0], nodes.paragraph):
            # the contents start with a paragraph
            if node[0].rawsource:
                # make the first paragraph translatable
                content = nodes.inline(node[0].rawsource, translatable=True)
                content.source = node[0].source
                content.line = node[0].line
                content += node[0].children
                node[0].replace_self(nodes.paragraph('', '', content, translatable=False))

            para = cast(nodes.paragraph, node[0])
            para.insert(0, nodes.inline('', '%s: ' % text, classes=classes))
        elif len(node) > 0:
            # the contents do not starts with a paragraph
            para = nodes.paragraph('', '',
                                   nodes.inline('', '%s: ' % text, classes=classes),
                                   translatable=False)
            node.insert(0, para)
        else:
            # the contents are empty
            para = nodes.paragraph('', '',
                                   nodes.inline('', '%s.' % text, classes=classes),
                                   translatable=False)
            node.append(para)

        domain = cast(ChangeSetDomain, self.env.get_domain('changeset'))
        domain.note_changeset(node)

        ret: List[Node] = [node]
        ret += messages
        return ret
```
### 7 - utils/checks.py:

Start line: 33, End line: 109

```python
@flake8ext
def sphinx_has_header(physical_line, filename, lines, line_number):
    if line_number != 1 or len(lines) < 10:
        return
    if os.path.samefile(filename, './sphinx/util/smartypants.py'):
        return

    # if the top-level package or not inside the package, ignore
    mod_name = os.path.splitext(filename)[0].strip('./\\').replace(
        '/', '.').replace('.__init__', '')
    if mod_name == 'sphinx' or not mod_name.startswith('sphinx.'):
        return

    # line number correction
    offset = 1
    if lines[0:1] == ['#!/usr/bin/env python3\n']:
        lines = lines[1:]
        offset = 2

    llist = []
    doc_open = False

    for lno, line in enumerate(lines):
        llist.append(line)
        if lno == 0:
            if line != '"""\n' and line != 'r"""\n':
                return 0, 'X101 missing docstring begin (""")'
            else:
                doc_open = True
        elif doc_open:
            if line == '"""\n':
                # end of docstring
                if lno <= 3:
                    return 0, 'X101 missing module name in docstring'
                break

            if line != '\n' and line[:4] != '    ' and doc_open:
                return 0, 'X101 missing correct docstring indentation'

            if lno == 1:
                mod_name_len = len(line.strip())
                if line.strip() != mod_name:
                    return 2, 'X101 wrong module name in docstring heading'
            elif lno == 2:
                if line.strip() != mod_name_len * '~':
                    return (3, 'X101 wrong module name underline, should be '
                            '~~~...~')
    else:
        return 0, 'X101 missing end and/or start of docstring...'
    license = llist[-2:-1]
    if not license or not license_re.match(license[0]):
        return 0, 'X101 no correct license info'

    offset = -3
    copyright = llist[offset:offset + 1]
    while copyright and copyright_2_re.match(copyright[0]):
        offset -= 1
        copyright = llist[offset:offset + 1]
    if not copyright or not copyright_re.match(copyright[0]):
        return 0, 'X101 no correct copyright info'
```
### 8 - sphinx/ext/autodoc/__init__.py:

Start line: 13, End line: 114

```python
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


def exclude_members_option(arg: Any) -> Union[object, Set[str]]:
    """Used to convert the :exclude-members: option."""
    if arg in (None, True):
        return EMPTY
    return {x.strip() for x in arg.split(',') if x.strip()}
```
### 9 - sphinx/directives/code.py:

Start line: 180, End line: 209

```python
class LiteralIncludeReader:
    INVALID_OPTIONS_PAIR = [
        ('lineno-match', 'lineno-start'),
        ('lineno-match', 'append'),
        ('lineno-match', 'prepend'),
        ('start-after', 'start-at'),
        ('end-before', 'end-at'),
        ('diff', 'pyobject'),
        ('diff', 'lineno-start'),
        ('diff', 'lineno-match'),
        ('diff', 'lines'),
        ('diff', 'start-after'),
        ('diff', 'end-before'),
        ('diff', 'start-at'),
        ('diff', 'end-at'),
    ]

    def __init__(self, filename: str, options: Dict, config: Config) -> None:
        self.filename = filename
        self.options = options
        self.encoding = options.get('encoding', config.source_encoding)
        self.lineno_start = self.options.get('lineno-start', 1)

        self.parse_options()

    def parse_options(self) -> None:
        for option1, option2 in self.INVALID_OPTIONS_PAIR:
            if option1 in self.options and option2 in self.options:
                raise ValueError(__('Cannot use both "%s" and "%s" options') %
                                 (option1, option2))
```
### 10 - sphinx/ext/doctest.py:

Start line: 12, End line: 43

```python
import doctest
import re
import sys
import time
from io import StringIO
from os import path
from typing import (TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Sequence, Set, Tuple,
                    Type)

from docutils import nodes
from docutils.nodes import Element, Node, TextElement
from docutils.parsers.rst import directives
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import Version

import sphinx
from sphinx.builders import Builder
from sphinx.locale import __
from sphinx.util import logging
from sphinx.util.console import bold  # type: ignore
from sphinx.util.docutils import SphinxDirective
from sphinx.util.osutil import relpath
from sphinx.util.typing import OptionSpec

if TYPE_CHECKING:
    from sphinx.application import Sphinx


logger = logging.getLogger(__name__)

blankline_re = re.compile(r'^\s*<BLANKLINE>', re.MULTILINE)
doctestopt_re = re.compile(r'#\s*doctest:.+$', re.MULTILINE)
```
### 29 - sphinx/domains/python.py:

Start line: 287, End line: 321

```python
# This override allows our inline type specifiers to behave like :class: link
# when it comes to handling "." and "~" prefixes.
class PyXrefMixin:
    def make_xref(self, rolename: str, domain: str, target: str,
                  innernode: Type[TextlikeNode] = nodes.emphasis,
                  contnode: Node = None, env: BuildEnvironment = None,
                  inliner: Inliner = None, location: Node = None) -> Node:
        # we use inliner=None to make sure we get the old behaviour with a single
        # pending_xref node
        result = super().make_xref(rolename, domain, target,  # type: ignore
                                   innernode, contnode,
                                   env, inliner=None, location=None)
        result['refspecific'] = True
        result['py:module'] = env.ref_context.get('py:module')
        result['py:class'] = env.ref_context.get('py:class')
        if target.startswith(('.', '~')):
            prefix, result['reftarget'] = target[0], target[1:]
            if prefix == '.':
                text = target[1:]
            elif prefix == '~':
                text = target.split('.')[-1]
            for node in result.traverse(nodes.Text):
                node.parent[node.parent.index(node)] = nodes.Text(text)
                break
        elif isinstance(result, pending_xref) and env.config.python_use_unqualified_type_names:
            children = result.children
            result.clear()

            shortname = target.split('.')[-1]
            textnode = innernode('', shortname)
            contnodes = [pending_xref_condition('', '', textnode, condition='resolved'),
                         pending_xref_condition('', '', *children, condition='*')]
            result.extend(contnodes)

        return result
```
