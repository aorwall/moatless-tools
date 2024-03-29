# sphinx-doc__sphinx-9799

| **sphinx-doc/sphinx** | `2b5c55e45a0fc4e2197a9b8edb482b77c2fa3f85` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | 7698 |
| **Avg pos** | 28.0 |
| **Min pos** | 14 |
| **Max pos** | 14 |
| **Top file pos** | 11 |
| **Missing snippets** | 3 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sphinx/ext/autodoc/preserve_defaults.py b/sphinx/ext/autodoc/preserve_defaults.py
--- a/sphinx/ext/autodoc/preserve_defaults.py
+++ b/sphinx/ext/autodoc/preserve_defaults.py
@@ -11,7 +11,8 @@
 
 import ast
 import inspect
-from typing import Any, Dict
+import sys
+from typing import Any, Dict, List, Optional
 
 from sphinx.application import Sphinx
 from sphinx.locale import __
@@ -49,11 +50,32 @@ def get_function_def(obj: Any) -> ast.FunctionDef:
         return None
 
 
+def get_default_value(lines: List[str], position: ast.AST) -> Optional[str]:
+    try:
+        if sys.version_info < (3, 8):  # only for py38+
+            return None
+        elif position.lineno == position.end_lineno:
+            line = lines[position.lineno - 1]
+            return line[position.col_offset:position.end_col_offset]
+        else:
+            # multiline value is not supported now
+            return None
+    except (AttributeError, IndexError):
+        return None
+
+
 def update_defvalue(app: Sphinx, obj: Any, bound_method: bool) -> None:
     """Update defvalue info of *obj* using type_comments."""
     if not app.config.autodoc_preserve_defaults:
         return
 
+    try:
+        lines = inspect.getsource(obj).splitlines()
+        if lines[0].startswith((' ', r'\t')):
+            lines.insert(0, '')  # insert a dummy line to follow what get_function_def() does.
+    except OSError:
+        lines = []
+
     try:
         function = get_function_def(obj)
         if function.args.defaults or function.args.kw_defaults:
@@ -64,11 +86,17 @@ def update_defvalue(app: Sphinx, obj: Any, bound_method: bool) -> None:
             for i, param in enumerate(parameters):
                 if param.default is not param.empty:
                     if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD):
-                        value = DefaultValue(ast_unparse(defaults.pop(0)))  # type: ignore
-                        parameters[i] = param.replace(default=value)
+                        default = defaults.pop(0)
+                        value = get_default_value(lines, default)
+                        if value is None:
+                            value = ast_unparse(default)  # type: ignore
+                        parameters[i] = param.replace(default=DefaultValue(value))
                     else:
-                        value = DefaultValue(ast_unparse(kw_defaults.pop(0)))  # type: ignore
-                        parameters[i] = param.replace(default=value)
+                        default = kw_defaults.pop(0)
+                        value = get_default_value(lines, default)
+                        if value is None:
+                            value = ast_unparse(default)  # type: ignore
+                        parameters[i] = param.replace(default=DefaultValue(value))
             sig = sig.replace(parameters=parameters)
             obj.__signature__ = sig
     except (AttributeError, TypeError):

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sphinx/ext/autodoc/preserve_defaults.py | 14 | 14 | - | 11 | -
| sphinx/ext/autodoc/preserve_defaults.py | 52 | 52 | 14 | 11 | 7698
| sphinx/ext/autodoc/preserve_defaults.py | 67 | 71 | 14 | 11 | 7698


## Problem Statement

```
Re-opening #8255: hexadecimal default arguments are changed to decimal
### Describe the bug

I am experiencing the exact same problem as described in #8255: hexadecimal default arguments are changed to decimal.

### How to Reproduce

Autodoc the following function:

\`\`\`python3
def some_function(
        param_a,
        param_b,
        *,  # enforce keyword arguments from this point onwards
        background_colour: int = 0xFFFFFFFF,
        # ... other optional parameters
    ):
    pass
\`\`\`

HTML result looks like this
\`\`\`
background_colour: int = 4294967295
\`\`\`

### Expected behavior

Hexadecimal defaults should not be converted to decimal, or at least there should be an option to enforce this behaviour.

### Your project

I'm afraid this is private

### Screenshots

_No response_

### OS

Linux Ubuntu 20.04

### Python version

3.8.10

### Sphinx version

4.2.0

### Sphinx extensions

autodoc, intersphinx, napoleon

### Extra tools

Chromium 94

### Additional context

_No response_

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 sphinx/ext/autodoc/__init__.py | 2798 | 2841| 528 | 528 | 23912 | 
| 2 | 2 doc/conf.py | 1 | 82| 732 | 1260 | 25377 | 
| 3 | 2 sphinx/ext/autodoc/__init__.py | 13 | 114| 788 | 2048 | 25377 | 
| 4 | 3 sphinx/util/inspect.py | 521 | 547| 176 | 2224 | 32255 | 
| 5 | 3 sphinx/ext/autodoc/__init__.py | 1281 | 1400| 1004 | 3228 | 32255 | 
| 6 | 4 sphinx/writers/html5.py | 156 | 212| 521 | 3749 | 39550 | 
| 7 | 5 sphinx/builders/html/__init__.py | 1305 | 1386| 1000 | 4749 | 51913 | 
| 8 | 6 sphinx/writers/html.py | 185 | 241| 516 | 5265 | 59683 | 
| 9 | 7 sphinx/config.py | 371 | 405| 296 | 5561 | 64142 | 
| 10 | 8 sphinx/ext/autodoc/directive.py | 9 | 47| 310 | 5871 | 65593 | 
| 11 | 9 sphinx/builders/latex/constants.py | 74 | 124| 537 | 6408 | 67838 | 
| 12 | 9 sphinx/config.py | 91 | 150| 740 | 7148 | 67838 | 
| 13 | 10 doc/development/tutorials/examples/autodoc_intenum.py | 27 | 53| 200 | 7348 | 68212 | 
| **-> 14 <-** | **11 sphinx/ext/autodoc/preserve_defaults.py** | 52 | 89| 350 | 7698 | 68906 | 
| 15 | 12 sphinx/cmd/quickstart.py | 191 | 266| 675 | 8373 | 74476 | 
| 16 | 12 sphinx/cmd/quickstart.py | 11 | 119| 756 | 9129 | 74476 | 
| 17 | 13 sphinx/environment/__init__.py | 11 | 85| 532 | 9661 | 79997 | 
| 18 | 13 doc/development/tutorials/examples/autodoc_intenum.py | 1 | 25| 183 | 9844 | 79997 | 
| 19 | 13 sphinx/util/inspect.py | 11 | 48| 298 | 10142 | 79997 | 
| 20 | 14 sphinx/ext/imgmath.py | 11 | 82| 506 | 10648 | 83182 | 
| 21 | 15 sphinx/ext/autodoc/deprecated.py | 32 | 47| 140 | 10788 | 84142 | 
| 22 | 15 sphinx/ext/autodoc/__init__.py | 117 | 158| 284 | 11072 | 84142 | 
| 23 | 16 sphinx/cmd/build.py | 33 | 98| 647 | 11719 | 86804 | 
| 24 | 17 sphinx/directives/__init__.py | 269 | 294| 167 | 11886 | 89056 | 
| 25 | 18 sphinx/ext/napoleon/__init__.py | 271 | 294| 298 | 12184 | 93084 | 
| 26 | 19 sphinx/builders/manpage.py | 110 | 129| 166 | 12350 | 94047 | 
| 27 | 20 sphinx/util/__init__.py | 272 | 289| 207 | 12557 | 98925 | 
| 28 | 21 sphinx/ext/autodoc/importer.py | 11 | 40| 211 | 12768 | 101387 | 
| 29 | 22 sphinx/ext/napoleon/docstring.py | 13 | 67| 578 | 13346 | 112387 | 
| 30 | 22 doc/conf.py | 141 | 162| 255 | 13601 | 112387 | 
| 31 | 22 sphinx/ext/napoleon/docstring.py | 695 | 736| 418 | 14019 | 112387 | 
| 32 | 22 sphinx/ext/autodoc/__init__.py | 2070 | 2275| 1863 | 15882 | 112387 | 
| 33 | 22 sphinx/ext/imgmath.py | 345 | 365| 289 | 16171 | 112387 | 
| 34 | 22 sphinx/ext/autodoc/deprecated.py | 11 | 29| 155 | 16326 | 112387 | 
| 35 | 23 sphinx/ext/autosummary/generate.py | 20 | 53| 257 | 16583 | 117687 | 
| 36 | 23 sphinx/ext/autodoc/__init__.py | 1434 | 1776| 3113 | 19696 | 117687 | 
| 37 | 24 sphinx/util/pycompat.py | 11 | 46| 328 | 20024 | 118180 | 
| 38 | 25 sphinx/testing/fixtures.py | 65 | 110| 391 | 20415 | 120004 | 
| 39 | 26 sphinx/util/cfamily.py | 11 | 62| 749 | 21164 | 123463 | 
| 40 | 27 sphinx/highlighting.py | 11 | 68| 620 | 21784 | 125009 | 
| 41 | 27 sphinx/writers/html.py | 721 | 819| 769 | 22553 | 125009 | 
| 42 | 28 sphinx/util/rst.py | 11 | 81| 556 | 23109 | 125877 | 
| 43 | 29 sphinx/ext/apidoc.py | 303 | 368| 752 | 23861 | 130111 | 
| 44 | 29 sphinx/ext/napoleon/__init__.py | 334 | 350| 158 | 24019 | 130111 | 
| 45 | 30 sphinx/domains/std.py | 259 | 276| 122 | 24141 | 140408 | 
| 46 | 31 sphinx/builders/texinfo.py | 198 | 220| 227 | 24368 | 142421 | 
| 47 | 32 doc/usage/extensions/example_numpy.py | 336 | 356| 120 | 24488 | 144529 | 
| 48 | 33 doc/usage/extensions/example_google.py | 277 | 296| 120 | 24608 | 146639 | 
| 49 | 33 sphinx/ext/autodoc/directive.py | 82 | 105| 255 | 24863 | 146639 | 
| 50 | 33 sphinx/writers/html5.py | 241 | 262| 208 | 25071 | 146639 | 
| 51 | 34 sphinx/util/console.py | 11 | 29| 123 | 25194 | 147635 | 
| 52 | 34 sphinx/builders/latex/constants.py | 126 | 217| 1040 | 26234 | 147635 | 
| 53 | 35 sphinx/ext/autodoc/typehints.py | 130 | 185| 460 | 26694 | 149089 | 
| 54 | 36 sphinx/pygments_styles.py | 11 | 35| 135 | 26829 | 149781 | 
| 55 | 36 sphinx/writers/html5.py | 657 | 743| 686 | 27515 | 149781 | 
| 56 | 37 sphinx/directives/other.py | 9 | 38| 229 | 27744 | 152908 | 
| 57 | 37 sphinx/cmd/quickstart.py | 138 | 163| 234 | 27978 | 152908 | 
| 58 | 38 sphinx/transforms/post_transforms/code.py | 114 | 143| 208 | 28186 | 153926 | 
| 59 | 38 sphinx/ext/autodoc/__init__.py | 1986 | 2010| 259 | 28445 | 153926 | 
| 60 | 38 sphinx/ext/autodoc/__init__.py | 2578 | 2601| 229 | 28674 | 153926 | 
| 61 | 38 sphinx/directives/__init__.py | 246 | 266| 162 | 28836 | 153926 | 
| 62 | 38 sphinx/cmd/quickstart.py | 544 | 611| 491 | 29327 | 153926 | 
| 63 | 39 sphinx/domains/c.py | 3606 | 3919| 2940 | 32267 | 186369 | 
| 64 | 40 sphinx/directives/patches.py | 9 | 44| 292 | 32559 | 188307 | 
| 65 | 40 sphinx/ext/napoleon/docstring.py | 478 | 519| 345 | 32904 | 188307 | 
| 66 | 40 sphinx/util/console.py | 101 | 142| 296 | 33200 | 188307 | 
| 67 | 40 sphinx/ext/autodoc/__init__.py | 1403 | 1776| 188 | 33388 | 188307 | 
| 68 | 41 sphinx/util/logging.py | 11 | 54| 262 | 33650 | 192102 | 
| 69 | 41 sphinx/writers/html5.py | 760 | 817| 550 | 34200 | 192102 | 
| 70 | 41 doc/usage/extensions/example_numpy.py | 101 | 162| 378 | 34578 | 192102 | 
| 71 | 42 sphinx/builders/changes.py | 125 | 168| 438 | 35016 | 193615 | 
| 72 | 42 sphinx/directives/__init__.py | 11 | 47| 253 | 35269 | 193615 | 
| 73 | 42 sphinx/util/__init__.py | 145 | 172| 228 | 35497 | 193615 | 
| 74 | 43 utils/checks.py | 33 | 109| 545 | 36042 | 194521 | 
| 75 | 43 sphinx/ext/napoleon/__init__.py | 20 | 270| 2007 | 38049 | 194521 | 
| 76 | 44 sphinx/util/smartypants.py | 380 | 392| 137 | 38186 | 198664 | 
| 77 | 44 doc/usage/extensions/example_google.py | 78 | 129| 381 | 38567 | 198664 | 
| 78 | 44 sphinx/writers/html5.py | 11 | 50| 307 | 38874 | 198664 | 


### Hint

```
Does `autodoc_preserve_defaults` help you?
https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#confval-autodoc_preserve_defaults
I had already added
\`\`\`python3
autodoc_preserve_defaults = True
\`\`\`
to my `conf.py` file but it didn't change the output. So no, it doesn't help.
```

## Patch

```diff
diff --git a/sphinx/ext/autodoc/preserve_defaults.py b/sphinx/ext/autodoc/preserve_defaults.py
--- a/sphinx/ext/autodoc/preserve_defaults.py
+++ b/sphinx/ext/autodoc/preserve_defaults.py
@@ -11,7 +11,8 @@
 
 import ast
 import inspect
-from typing import Any, Dict
+import sys
+from typing import Any, Dict, List, Optional
 
 from sphinx.application import Sphinx
 from sphinx.locale import __
@@ -49,11 +50,32 @@ def get_function_def(obj: Any) -> ast.FunctionDef:
         return None
 
 
+def get_default_value(lines: List[str], position: ast.AST) -> Optional[str]:
+    try:
+        if sys.version_info < (3, 8):  # only for py38+
+            return None
+        elif position.lineno == position.end_lineno:
+            line = lines[position.lineno - 1]
+            return line[position.col_offset:position.end_col_offset]
+        else:
+            # multiline value is not supported now
+            return None
+    except (AttributeError, IndexError):
+        return None
+
+
 def update_defvalue(app: Sphinx, obj: Any, bound_method: bool) -> None:
     """Update defvalue info of *obj* using type_comments."""
     if not app.config.autodoc_preserve_defaults:
         return
 
+    try:
+        lines = inspect.getsource(obj).splitlines()
+        if lines[0].startswith((' ', r'\t')):
+            lines.insert(0, '')  # insert a dummy line to follow what get_function_def() does.
+    except OSError:
+        lines = []
+
     try:
         function = get_function_def(obj)
         if function.args.defaults or function.args.kw_defaults:
@@ -64,11 +86,17 @@ def update_defvalue(app: Sphinx, obj: Any, bound_method: bool) -> None:
             for i, param in enumerate(parameters):
                 if param.default is not param.empty:
                     if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD):
-                        value = DefaultValue(ast_unparse(defaults.pop(0)))  # type: ignore
-                        parameters[i] = param.replace(default=value)
+                        default = defaults.pop(0)
+                        value = get_default_value(lines, default)
+                        if value is None:
+                            value = ast_unparse(default)  # type: ignore
+                        parameters[i] = param.replace(default=DefaultValue(value))
                     else:
-                        value = DefaultValue(ast_unparse(kw_defaults.pop(0)))  # type: ignore
-                        parameters[i] = param.replace(default=value)
+                        default = kw_defaults.pop(0)
+                        value = get_default_value(lines, default)
+                        if value is None:
+                            value = ast_unparse(default)  # type: ignore
+                        parameters[i] = param.replace(default=DefaultValue(value))
             sig = sig.replace(parameters=parameters)
             obj.__signature__ = sig
     except (AttributeError, TypeError):

```

## Test Patch

```diff
diff --git a/tests/roots/test-ext-autodoc/target/preserve_defaults.py b/tests/roots/test-ext-autodoc/target/preserve_defaults.py
--- a/tests/roots/test-ext-autodoc/target/preserve_defaults.py
+++ b/tests/roots/test-ext-autodoc/target/preserve_defaults.py
@@ -7,7 +7,8 @@
 
 def foo(name: str = CONSTANT,
         sentinel: Any = SENTINEL,
-        now: datetime = datetime.now()) -> None:
+        now: datetime = datetime.now(),
+        color: int = 0xFFFFFF) -> None:
     """docstring"""
 
 
@@ -15,5 +16,5 @@ class Class:
     """docstring"""
 
     def meth(self, name: str = CONSTANT, sentinel: Any = SENTINEL,
-             now: datetime = datetime.now()) -> None:
+             now: datetime = datetime.now(), color: int = 0xFFFFFF) -> None:
         """docstring"""
diff --git a/tests/test_ext_autodoc_preserve_defaults.py b/tests/test_ext_autodoc_preserve_defaults.py
--- a/tests/test_ext_autodoc_preserve_defaults.py
+++ b/tests/test_ext_autodoc_preserve_defaults.py
@@ -8,6 +8,8 @@
     :license: BSD, see LICENSE for details.
 """
 
+import sys
+
 import pytest
 
 from .test_ext_autodoc import do_autodoc
@@ -16,6 +18,11 @@
 @pytest.mark.sphinx('html', testroot='ext-autodoc',
                     confoverrides={'autodoc_preserve_defaults': True})
 def test_preserve_defaults(app):
+    if sys.version_info < (3, 8):
+        color = "16777215"
+    else:
+        color = "0xFFFFFF"
+
     options = {"members": None}
     actual = do_autodoc(app, 'module', 'target.preserve_defaults', options)
     assert list(actual) == [
@@ -30,14 +37,14 @@ def test_preserve_defaults(app):
         '',
         '',
         '   .. py:method:: Class.meth(name: str = CONSTANT, sentinel: Any = SENTINEL, '
-        'now: datetime.datetime = datetime.now()) -> None',
+        'now: datetime.datetime = datetime.now(), color: int = %s) -> None' % color,
         '      :module: target.preserve_defaults',
         '',
         '      docstring',
         '',
         '',
         '.. py:function:: foo(name: str = CONSTANT, sentinel: Any = SENTINEL, now: '
-        'datetime.datetime = datetime.now()) -> None',
+        'datetime.datetime = datetime.now(), color: int = %s) -> None' % color,
         '   :module: target.preserve_defaults',
         '',
         '   docstring',

```


## Code snippets

### 1 - sphinx/ext/autodoc/__init__.py:

Start line: 2798, End line: 2841

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
### 2 - doc/conf.py:

Start line: 1, End line: 82

```python
# Sphinx documentation build configuration file

import re

import sphinx

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.doctest', 'sphinx.ext.todo',
              'sphinx.ext.autosummary', 'sphinx.ext.extlinks',
              'sphinx.ext.intersphinx',
              'sphinx.ext.viewcode', 'sphinx.ext.inheritance_diagram']

root_doc = 'contents'
templates_path = ['_templates']
exclude_patterns = ['_build']

project = 'Sphinx'
copyright = '2007-2021, Georg Brandl and the Sphinx team'
version = sphinx.__display_version__
release = version
show_authors = True

html_theme = 'sphinx13'
html_theme_path = ['_themes']
modindex_common_prefix = ['sphinx.']
html_static_path = ['_static']
html_sidebars = {'index': ['indexsidebar.html', 'searchbox.html']}
html_title = 'Sphinx documentation'
html_additional_pages = {'index': 'index.html'}
html_use_opensearch = 'https://www.sphinx-doc.org/en/master'
html_baseurl = 'https://www.sphinx-doc.org/en/master/'
html_favicon = '_static/favicon.svg'

htmlhelp_basename = 'Sphinxdoc'

epub_theme = 'epub'
epub_basename = 'sphinx'
epub_author = 'Georg Brandl'
epub_publisher = 'https://www.sphinx-doc.org/'
epub_uid = 'web-site'
epub_scheme = 'url'
epub_identifier = epub_publisher
epub_pre_files = [('index.xhtml', 'Welcome')]
epub_post_files = [('usage/installation.xhtml', 'Installing Sphinx'),
                   ('develop.xhtml', 'Sphinx development')]
epub_exclude_files = ['_static/opensearch.xml', '_static/doctools.js',
                      '_static/jquery.js', '_static/searchtools.js',
                      '_static/underscore.js', '_static/basic.css',
                      '_static/language_data.js',
                      'search.html', '_static/websupport.js']
epub_fix_images = False
epub_max_image_width = 0
epub_show_urls = 'inline'
epub_use_index = False
epub_guide = (('toc', 'contents.xhtml', 'Table of Contents'),)
epub_description = 'Sphinx documentation generator system manual'

latex_documents = [('contents', 'sphinx.tex', 'Sphinx Documentation',
                    'Georg Brandl', 'manual', 1)]
latex_logo = '_static/sphinx.png'
latex_elements = {
    'fontenc': r'\usepackage[LGR,X2,T1]{fontenc}',
    'passoptionstopackages': r'''
\PassOptionsToPackage{svgnames}{xcolor}
''',
    'preamble': r'''
\DeclareUnicodeCharacter{229E}{\ensuremath{\boxplus}}
\setcounter{tocdepth}{3}%    depth of what main TOC shows (3=subsubsection)
\setcounter{secnumdepth}{1}% depth of section numbering
''',
    # fix missing index entry due to RTD doing only once pdflatex after makeindex
    'printindex': r'''
\IfFileExists{\jobname.ind}
             {\footnotesize\raggedright\printindex}
             {\begin{sphinxtheindex}\end{sphinxtheindex}}
''',
}
latex_show_urls = 'footnote'
latex_use_xindy = True

autodoc_member_order = 'groupwise'
autosummary_generate = False
todo_include_todos = True
```
### 3 - sphinx/ext/autodoc/__init__.py:

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
### 4 - sphinx/util/inspect.py:

Start line: 521, End line: 547

```python
class DefaultValue:
    """A simple wrapper for default value of the parameters of overload functions."""

    def __init__(self, value: str) -> None:
        self.value = value

    def __eq__(self, other: object) -> bool:
        return self.value == other

    def __repr__(self) -> str:
        return self.value


class TypeAliasForwardRef:
    """Pseudo typing class for autodoc_type_aliases.

    This avoids the error on evaluating the type inside `get_type_hints()`.
    """
    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self) -> None:
        # Dummy method to imitate special typing classes
        pass

    def __eq__(self, other: Any) -> bool:
        return self.name == other
```
### 5 - sphinx/ext/autodoc/__init__.py:

Start line: 1281, End line: 1400

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

        if inspect.iscoroutinefunction(self.object) or inspect.isasyncgenfunction(self.object):
            self.add_line('   :async:', sourcename)

    def format_signature(self, **kwargs: Any) -> str:
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
### 6 - sphinx/writers/html5.py:

Start line: 156, End line: 212

```python
class HTML5Translator(SphinxTranslator, BaseTranslator):

    def visit_desc_parameterlist(self, node: Element) -> None:
        self.body.append('<span class="sig-paren">(</span>')
        self.first_param = 1
        self.optional_param_level = 0
        # How many required parameters are left.
        self.required_params_left = sum([isinstance(c, addnodes.desc_parameter)
                                         for c in node.children])
        self.param_separator = node.child_text_separator

    def depart_desc_parameterlist(self, node: Element) -> None:
        self.body.append('<span class="sig-paren">)</span>')

    # If required parameters are still to come, then put the comma after
    # the parameter.  Otherwise, put the comma before.  This ensures that
    # signatures like the following render correctly (see issue #1001):
    #
    #     foo([a, ]b, c[, d])
    #
    def visit_desc_parameter(self, node: Element) -> None:
        if self.first_param:
            self.first_param = 0
        elif not self.required_params_left:
            self.body.append(self.param_separator)
        if self.optional_param_level == 0:
            self.required_params_left -= 1
        if not node.hasattr('noemph'):
            self.body.append('<em class="sig-param">')

    def depart_desc_parameter(self, node: Element) -> None:
        if not node.hasattr('noemph'):
            self.body.append('</em>')
        if self.required_params_left:
            self.body.append(self.param_separator)

    def visit_desc_optional(self, node: Element) -> None:
        self.optional_param_level += 1
        self.body.append('<span class="optional">[</span>')

    def depart_desc_optional(self, node: Element) -> None:
        self.optional_param_level -= 1
        self.body.append('<span class="optional">]</span>')

    def visit_desc_annotation(self, node: Element) -> None:
        self.body.append(self.starttag(node, 'em', '', CLASS='property'))

    def depart_desc_annotation(self, node: Element) -> None:
        self.body.append('</em>')

    ##############################################

    def visit_versionmodified(self, node: Element) -> None:
        self.body.append(self.starttag(node, 'div', CLASS=node['type']))

    def depart_versionmodified(self, node: Element) -> None:
        self.body.append('</div>\n')

    # overwritten
```
### 7 - sphinx/builders/html/__init__.py:

Start line: 1305, End line: 1386

```python
# NOQA


def setup(app: Sphinx) -> Dict[str, Any]:
    # builders
    app.add_builder(StandaloneHTMLBuilder)

    # config values
    app.add_config_value('html_theme', 'alabaster', 'html')
    app.add_config_value('html_theme_path', [], 'html')
    app.add_config_value('html_theme_options', {}, 'html')
    app.add_config_value('html_title',
                         lambda self: _('%s %s documentation') % (self.project, self.release),
                         'html', [str])
    app.add_config_value('html_short_title', lambda self: self.html_title, 'html')
    app.add_config_value('html_style', None, 'html', [str])
    app.add_config_value('html_logo', None, 'html', [str])
    app.add_config_value('html_favicon', None, 'html', [str])
    app.add_config_value('html_css_files', [], 'html')
    app.add_config_value('html_js_files', [], 'html')
    app.add_config_value('html_static_path', [], 'html')
    app.add_config_value('html_extra_path', [], 'html')
    app.add_config_value('html_last_updated_fmt', None, 'html', [str])
    app.add_config_value('html_sidebars', {}, 'html')
    app.add_config_value('html_additional_pages', {}, 'html')
    app.add_config_value('html_domain_indices', True, 'html', [list])
    app.add_config_value('html_add_permalinks', UNSET, 'html')
    app.add_config_value('html_permalinks', True, 'html')
    app.add_config_value('html_permalinks_icon', '¶', 'html')
    app.add_config_value('html_use_index', True, 'html')
    app.add_config_value('html_split_index', False, 'html')
    app.add_config_value('html_copy_source', True, 'html')
    app.add_config_value('html_show_sourcelink', True, 'html')
    app.add_config_value('html_sourcelink_suffix', '.txt', 'html')
    app.add_config_value('html_use_opensearch', '', 'html')
    app.add_config_value('html_file_suffix', None, 'html', [str])
    app.add_config_value('html_link_suffix', None, 'html', [str])
    app.add_config_value('html_show_copyright', True, 'html')
    app.add_config_value('html_show_sphinx', True, 'html')
    app.add_config_value('html_context', {}, 'html')
    app.add_config_value('html_output_encoding', 'utf-8', 'html')
    app.add_config_value('html_compact_lists', True, 'html')
    app.add_config_value('html_secnumber_suffix', '. ', 'html')
    app.add_config_value('html_search_language', None, 'html', [str])
    app.add_config_value('html_search_options', {}, 'html')
    app.add_config_value('html_search_scorer', '', None)
    app.add_config_value('html_scaled_image_link', True, 'html')
    app.add_config_value('html_baseurl', '', 'html')
    app.add_config_value('html_codeblock_linenos_style', 'inline', 'html',  # RemovedInSphinx60Warning  # NOQA
                         ENUM('table', 'inline'))
    app.add_config_value('html_math_renderer', None, 'env')
    app.add_config_value('html4_writer', False, 'html')

    # events
    app.add_event('html-collect-pages')
    app.add_event('html-page-context')

    # event handlers
    app.connect('config-inited', convert_html_css_files, priority=800)
    app.connect('config-inited', convert_html_js_files, priority=800)
    app.connect('config-inited', migrate_html_add_permalinks, priority=800)
    app.connect('config-inited', validate_html_extra_path, priority=800)
    app.connect('config-inited', validate_html_static_path, priority=800)
    app.connect('config-inited', validate_html_logo, priority=800)
    app.connect('config-inited', validate_html_favicon, priority=800)
    app.connect('builder-inited', validate_math_renderer)
    app.connect('html-page-context', setup_css_tag_helper)
    app.connect('html-page-context', setup_js_tag_helper)
    app.connect('html-page-context', setup_resource_paths)

    # load default math renderer
    app.setup_extension('sphinx.ext.mathjax')

    # load transforms for HTML builder
    app.setup_extension('sphinx.builders.html.transforms')

    return {
        'version': 'builtin',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
```
### 8 - sphinx/writers/html.py:

Start line: 185, End line: 241

```python
class HTMLTranslator(SphinxTranslator, BaseTranslator):

    def visit_desc_parameterlist(self, node: Element) -> None:
        self.body.append('<span class="sig-paren">(</span>')
        self.first_param = 1
        self.optional_param_level = 0
        # How many required parameters are left.
        self.required_params_left = sum([isinstance(c, addnodes.desc_parameter)
                                         for c in node.children])
        self.param_separator = node.child_text_separator

    def depart_desc_parameterlist(self, node: Element) -> None:
        self.body.append('<span class="sig-paren">)</span>')

    # If required parameters are still to come, then put the comma after
    # the parameter.  Otherwise, put the comma before.  This ensures that
    # signatures like the following render correctly (see issue #1001):
    #
    #     foo([a, ]b, c[, d])
    #
    def visit_desc_parameter(self, node: Element) -> None:
        if self.first_param:
            self.first_param = 0
        elif not self.required_params_left:
            self.body.append(self.param_separator)
        if self.optional_param_level == 0:
            self.required_params_left -= 1
        if not node.hasattr('noemph'):
            self.body.append('<em>')

    def depart_desc_parameter(self, node: Element) -> None:
        if not node.hasattr('noemph'):
            self.body.append('</em>')
        if self.required_params_left:
            self.body.append(self.param_separator)

    def visit_desc_optional(self, node: Element) -> None:
        self.optional_param_level += 1
        self.body.append('<span class="optional">[</span>')

    def depart_desc_optional(self, node: Element) -> None:
        self.optional_param_level -= 1
        self.body.append('<span class="optional">]</span>')

    def visit_desc_annotation(self, node: Element) -> None:
        self.body.append(self.starttag(node, 'em', '', CLASS='property'))

    def depart_desc_annotation(self, node: Element) -> None:
        self.body.append('</em>')

    ##############################################

    def visit_versionmodified(self, node: Element) -> None:
        self.body.append(self.starttag(node, 'div', CLASS=node['type']))

    def depart_versionmodified(self, node: Element) -> None:
        self.body.append('</div>\n')

    # overwritten
```
### 9 - sphinx/config.py:

Start line: 371, End line: 405

```python
def convert_highlight_options(app: "Sphinx", config: Config) -> None:
    """Convert old styled highlight_options to new styled one.

    * old style: options
    * new style: a dict which maps from language name to options
    """
    options = config.highlight_options
    if options and not all(isinstance(v, dict) for v in options.values()):
        # old styled option detected because all values are not dictionary.
        config.highlight_options = {config.highlight_language: options}  # type: ignore


def init_numfig_format(app: "Sphinx", config: Config) -> None:
    """Initialize :confval:`numfig_format`."""
    numfig_format = {'section': _('Section %s'),
                     'figure': _('Fig. %s'),
                     'table': _('Table %s'),
                     'code-block': _('Listing %s')}

    # override default labels by configuration
    numfig_format.update(config.numfig_format)
    config.numfig_format = numfig_format  # type: ignore


def correct_copyright_year(app: "Sphinx", config: Config) -> None:
    if getenv('SOURCE_DATE_EPOCH') is not None:
        for k in ('copyright', 'epub_copyright'):
            if k in config:
                replace = r'\g<1>%s' % format_date('%Y')
                config[k] = copyright_year_re.sub(replace, config[k])
```
### 10 - sphinx/ext/autodoc/directive.py:

Start line: 9, End line: 47

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
                           'imported-members', 'class-doc-from']

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
### 14 - sphinx/ext/autodoc/preserve_defaults.py:

Start line: 52, End line: 89

```python
def update_defvalue(app: Sphinx, obj: Any, bound_method: bool) -> None:
    """Update defvalue info of *obj* using type_comments."""
    if not app.config.autodoc_preserve_defaults:
        return

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
                        value = DefaultValue(ast_unparse(defaults.pop(0)))  # type: ignore
                        parameters[i] = param.replace(default=value)
                    else:
                        value = DefaultValue(ast_unparse(kw_defaults.pop(0)))  # type: ignore
                        parameters[i] = param.replace(default=value)
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
