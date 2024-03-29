# sphinx-doc__sphinx-9798

| **sphinx-doc/sphinx** | `4c91c038b220d07bbdfe0c1680af42fe897f342c` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | - |
| **Avg pos** | - |
| **Min pos** | - |
| **Max pos** | - |
| **Top file pos** | 6 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sphinx/domains/python.py b/sphinx/domains/python.py
--- a/sphinx/domains/python.py
+++ b/sphinx/domains/python.py
@@ -353,17 +353,21 @@ def make_xrefs(self, rolename: str, domain: str, target: str,
 
         split_contnode = bool(contnode and contnode.astext() == target)
 
+        in_literal = False
         results = []
         for sub_target in filter(None, sub_targets):
             if split_contnode:
                 contnode = nodes.Text(sub_target)
 
-            if delims_re.match(sub_target):
+            if in_literal or delims_re.match(sub_target):
                 results.append(contnode or innernode(sub_target, sub_target))
             else:
                 results.append(self.make_xref(rolename, domain, sub_target,
                                               innernode, contnode, env, inliner, location))
 
+            if sub_target in ('Literal', 'typing.Literal'):
+                in_literal = True
+
         return results
 
 

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sphinx/domains/python.py | 356 | 356 | - | 6 | -


## Problem Statement

```
Nitpick flags Literal annotation values as missing py:class (with type hints in description)
### Describe the bug

This is basically the same issue as #9576, which was fixed in #9602.  However, I still get this issue when using `autodoc_typehints = 'description'`.

### How to Reproduce

\`\`\`
$ unzip attachment.zip
$ python3.9 -m venv .venv
$ . .venv/bin/activate
$ pip install sphinx
$ sphinx-build -b html -n -W docs docs/_build
Running Sphinx v4.2.0
making output directory... done
[autosummary] generating autosummary for: index.rst, rst/api.rst
[autosummary] generating autosummary for: <snip>/docs/rst/generated/dummy.foo.bar.rst
building [mo]: targets for 0 po files that are out of date
building [html]: targets for 2 source files that are out of date
updating environment: [new config] 3 added, 0 changed, 0 removed
reading sources... [100%] rst/generated/dummy.foo.bar                                                                                                                                                                                                     
looking for now-outdated files... none found
pickling environment... done
checking consistency... done
preparing documents... done
writing output... [100%] rst/generated/dummy.foo.bar                                                                                                                                                                                                      

Warning, treated as error:
<snip>/src/dummy/foo.py:docstring of dummy.foo.bar::py:class reference target not found: ''
\`\`\`

Comment out the line `autodoc_typehints = 'description'` in docs/conf.py and it is successful, as shown below (and removing the build artifacts to start fresh).

\`\`\`
$ sphinx-build -b html -n -W docs docs/_build
Running Sphinx v4.2.0
making output directory... done
[autosummary] generating autosummary for: index.rst, rst/api.rst
[autosummary] generating autosummary for: <snip>/docs/rst/generated/dummy.foo.bar.rst
building [mo]: targets for 0 po files that are out of date
building [html]: targets for 2 source files that are out of date
updating environment: [new config] 3 added, 0 changed, 0 removed
reading sources... [100%] rst/generated/dummy.foo.bar                                                                                                                                                                                                     
looking for now-outdated files... none found
pickling environment... done
checking consistency... done
preparing documents... done
writing output... [100%] rst/generated/dummy.foo.bar                                                                                                                                                                                                      
generating indices... genindex py-modindex done
writing additional pages... search done
copying static files... done
copying extra files... done
dumping search index in English (code: en)... done
dumping object inventory... done
build succeeded.

The HTML pages are in docs/_build.
\`\`\`

[attachment.zip](https://github.com/sphinx-doc/sphinx/files/7416418/attachment.zip)


### Expected behavior

No error, the build should succeed.

### Your project

See attachment in "How to Reproduce" section

### Screenshots

N/A - output is shown in "How to Reproduce" section

### OS

Linux and Windows

### Python version

3.9

### Sphinx version

4.2.0

### Sphinx extensions

sphinx.ext.autodoc, sphinx.ext.autosummary

### Extra tools

N/A

### Additional context

This re-produces for me on both Linux and Windows.  I think the source of it issue is probably from [this line](https://github.com/sphinx-doc/sphinx/blob/2be9d6b092965a2f9354da66b645bf5ea76ce288/sphinx/ext/autodoc/typehints.py#L43) in `merge_typehints` since this function would otherwise be skipped if the type hints are left in the signature.  But I haven't yet been able to track it all the way down to the error.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 doc/conf.py | 1 | 82| 732 | 732 | 1465 | 
| 2 | 2 sphinx/ext/autosummary/generate.py | 20 | 53| 257 | 989 | 6765 | 
| 3 | 3 setup.py | 176 | 253| 651 | 1640 | 8538 | 
| 4 | 4 sphinx/ext/autodoc/__init__.py | 2798 | 2841| 528 | 2168 | 32450 | 
| 5 | 5 sphinx/environment/__init__.py | 11 | 85| 532 | 2700 | 37971 | 
| 6 | **6 sphinx/domains/python.py** | 11 | 80| 518 | 3218 | 50561 | 
| 7 | 7 sphinx/builders/__init__.py | 11 | 48| 302 | 3520 | 55911 | 
| 8 | 8 sphinx/ext/autodoc/typehints.py | 130 | 185| 460 | 3980 | 57365 | 
| 9 | 9 sphinx/errors.py | 77 | 134| 297 | 4277 | 58161 | 
| 10 | 10 utils/checks.py | 33 | 109| 545 | 4822 | 59067 | 
| 11 | 11 sphinx/application.py | 13 | 58| 361 | 5183 | 70824 | 
| 12 | 11 sphinx/ext/autosummary/generate.py | 172 | 191| 176 | 5359 | 70824 | 
| 13 | 11 sphinx/application.py | 332 | 381| 410 | 5769 | 70824 | 
| 14 | 12 sphinx/builders/html/__init__.py | 1305 | 1386| 1000 | 6769 | 83187 | 
| 15 | 13 sphinx/setup_command.py | 142 | 190| 415 | 7184 | 84733 | 
| 16 | 13 doc/conf.py | 83 | 138| 476 | 7660 | 84733 | 
| 17 | 14 sphinx/io.py | 10 | 39| 234 | 7894 | 86139 | 
| 18 | 15 sphinx/cmd/build.py | 11 | 30| 132 | 8026 | 88801 | 
| 19 | 15 setup.py | 1 | 78| 478 | 8504 | 88801 | 
| 20 | 15 sphinx/cmd/build.py | 101 | 168| 763 | 9267 | 88801 | 
| 21 | 15 sphinx/ext/autosummary/generate.py | 595 | 634| 361 | 9628 | 88801 | 
| 22 | 16 sphinx/cmd/make_mode.py | 17 | 54| 532 | 10160 | 90503 | 
| 23 | 16 sphinx/setup_command.py | 14 | 89| 415 | 10575 | 90503 | 
| 24 | 16 sphinx/ext/autosummary/generate.py | 368 | 458| 756 | 11331 | 90503 | 
| 25 | 16 sphinx/builders/html/__init__.py | 11 | 63| 443 | 11774 | 90503 | 
| 26 | 17 sphinx/registry.py | 11 | 51| 314 | 12088 | 95143 | 
| 27 | 18 sphinx/util/inspect.py | 11 | 48| 298 | 12386 | 102004 | 
| 28 | 19 sphinx/domains/changeset.py | 49 | 107| 516 | 12902 | 103258 | 
| 29 | 19 sphinx/ext/autodoc/__init__.py | 2578 | 2601| 229 | 13131 | 103258 | 
| 30 | 20 sphinx/ext/doctest.py | 12 | 43| 227 | 13358 | 108276 | 
| 31 | 20 sphinx/ext/autosummary/generate.py | 637 | 658| 162 | 13520 | 108276 | 
| 32 | 20 sphinx/ext/autodoc/typehints.py | 40 | 80| 333 | 13853 | 108276 | 
| 33 | 20 sphinx/setup_command.py | 91 | 118| 229 | 14082 | 108276 | 
| 34 | 20 sphinx/cmd/make_mode.py | 96 | 140| 375 | 14457 | 108276 | 
| 35 | 21 sphinx/__init__.py | 14 | 60| 476 | 14933 | 108826 | 
| 36 | 22 sphinx/builders/_epub_base.py | 11 | 118| 687 | 15620 | 115139 | 
| 37 | 23 sphinx/builders/latex/__init__.py | 11 | 42| 331 | 15951 | 120851 | 
| 38 | 23 sphinx/ext/autodoc/typehints.py | 11 | 37| 210 | 16161 | 120851 | 
| 39 | 24 sphinx/ext/autodoc/importer.py | 11 | 40| 211 | 16372 | 123313 | 
| 40 | 25 sphinx/ext/apidoc.py | 369 | 397| 374 | 16746 | 127547 | 
| 41 | 25 sphinx/ext/apidoc.py | 303 | 368| 752 | 17498 | 127547 | 
| 42 | 26 sphinx/cmd/quickstart.py | 11 | 119| 756 | 18254 | 133117 | 
| 43 | 27 sphinx/builders/changes.py | 125 | 168| 438 | 18692 | 134630 | 
| 44 | 28 sphinx/builders/manpage.py | 11 | 29| 154 | 18846 | 135593 | 
| 45 | 29 sphinx/builders/xml.py | 11 | 26| 116 | 18962 | 136432 | 
| 46 | 29 sphinx/ext/autodoc/__init__.py | 1434 | 1776| 3113 | 22075 | 136432 | 
| 47 | 30 sphinx/domains/javascript.py | 11 | 33| 199 | 22274 | 140585 | 
| 48 | 30 sphinx/ext/autodoc/__init__.py | 13 | 114| 788 | 23062 | 140585 | 
| 49 | 30 sphinx/ext/autodoc/importer.py | 77 | 147| 649 | 23711 | 140585 | 
| 50 | 31 sphinx/builders/latex/constants.py | 74 | 124| 537 | 24248 | 142830 | 
| 51 | 31 sphinx/ext/autodoc/__init__.py | 2645 | 2670| 316 | 24564 | 142830 | 
| 52 | 32 sphinx/domains/citation.py | 11 | 29| 125 | 24689 | 144111 | 
| 53 | 33 sphinx/builders/singlehtml.py | 11 | 25| 112 | 24801 | 145907 | 
| 54 | 34 sphinx/builders/texinfo.py | 11 | 36| 226 | 25027 | 147920 | 
| 55 | 34 sphinx/ext/autodoc/__init__.py | 2406 | 2441| 313 | 25340 | 147920 | 
| 56 | 34 sphinx/ext/autosummary/generate.py | 225 | 260| 339 | 25679 | 147920 | 
| 57 | 35 sphinx/transforms/__init__.py | 11 | 44| 231 | 25910 | 151092 | 
| 58 | 35 sphinx/cmd/build.py | 169 | 199| 383 | 26293 | 151092 | 
| 59 | 36 sphinx/directives/patches.py | 9 | 44| 292 | 26585 | 153030 | 
| 60 | 36 sphinx/ext/autodoc/__init__.py | 587 | 599| 139 | 26724 | 153030 | 
| 61 | 37 sphinx/testing/util.py | 140 | 154| 188 | 26912 | 154781 | 
| 62 | 38 sphinx/domains/std.py | 11 | 46| 311 | 27223 | 165078 | 
| 63 | 38 sphinx/ext/doctest.py | 246 | 273| 277 | 27500 | 165078 | 
| 64 | 38 sphinx/ext/doctest.py | 315 | 330| 130 | 27630 | 165078 | 
| 65 | 39 sphinx/builders/linkcheck.py | 286 | 304| 203 | 27833 | 171031 | 
| 66 | 40 sphinx/domains/rst.py | 11 | 32| 170 | 28003 | 173510 | 
| 67 | 40 sphinx/cmd/quickstart.py | 459 | 525| 752 | 28755 | 173510 | 
| 68 | 40 sphinx/setup_command.py | 120 | 140| 172 | 28927 | 173510 | 
| 69 | 41 sphinx/util/__init__.py | 11 | 64| 445 | 29372 | 178388 | 
| 70 | 42 sphinx/builders/epub3.py | 12 | 55| 288 | 29660 | 180975 | 
| 71 | 42 sphinx/application.py | 300 | 313| 132 | 29792 | 180975 | 
| 72 | 43 sphinx/ext/inheritance_diagram.py | 38 | 67| 243 | 30035 | 184841 | 
| 73 | 43 sphinx/ext/autodoc/__init__.py | 2628 | 2643| 182 | 30217 | 184841 | 
| 74 | 43 sphinx/ext/autosummary/generate.py | 287 | 300| 202 | 30419 | 184841 | 
| 75 | 43 sphinx/ext/autodoc/__init__.py | 1986 | 2010| 259 | 30678 | 184841 | 
| 76 | 43 sphinx/testing/util.py | 10 | 45| 270 | 30948 | 184841 | 
| 77 | 44 sphinx/ext/autodoc/directive.py | 9 | 47| 310 | 31258 | 186292 | 
| 78 | 45 sphinx/ext/graphviz.py | 12 | 44| 243 | 31501 | 190028 | 
| 79 | 46 sphinx/config.py | 91 | 150| 740 | 32241 | 194487 | 
| 80 | 46 sphinx/ext/doctest.py | 481 | 504| 264 | 32505 | 194487 | 
| 81 | 47 sphinx/ext/autodoc/type_comment.py | 11 | 35| 239 | 32744 | 195711 | 
| 82 | 48 sphinx/util/smartypants.py | 380 | 392| 137 | 32881 | 199854 | 


### Hint

```
@tk0miya is there any additional info I can provide?  Or any suggestions you can make to help me narrow down the source of this issue within the code base.  I ran it also with -vvv and it provide the traceback, but it doesn't really provide any additional insight to me.
Thank you for reporting. I reproduce the same error on my local. The example is expanded to the following mark-up on memory:

\`\`\`
.. function:: bar(bar='')

   :param bar:
   :type bar: Literal['', 'f', 'd']
\`\`\`

And the function directive failed to handle the `Literal` type.
```

## Patch

```diff
diff --git a/sphinx/domains/python.py b/sphinx/domains/python.py
--- a/sphinx/domains/python.py
+++ b/sphinx/domains/python.py
@@ -353,17 +353,21 @@ def make_xrefs(self, rolename: str, domain: str, target: str,
 
         split_contnode = bool(contnode and contnode.astext() == target)
 
+        in_literal = False
         results = []
         for sub_target in filter(None, sub_targets):
             if split_contnode:
                 contnode = nodes.Text(sub_target)
 
-            if delims_re.match(sub_target):
+            if in_literal or delims_re.match(sub_target):
                 results.append(contnode or innernode(sub_target, sub_target))
             else:
                 results.append(self.make_xref(rolename, domain, sub_target,
                                               innernode, contnode, env, inliner, location))
 
+            if sub_target in ('Literal', 'typing.Literal'):
+                in_literal = True
+
         return results
 
 

```

## Test Patch

```diff
diff --git a/tests/test_domain_py.py b/tests/test_domain_py.py
--- a/tests/test_domain_py.py
+++ b/tests/test_domain_py.py
@@ -1110,6 +1110,42 @@ def test_info_field_list_piped_type(app):
                 **{"py:module": "example", "py:class": "Class"})
 
 
+def test_info_field_list_Literal(app):
+    text = (".. py:module:: example\n"
+            ".. py:class:: Class\n"
+            "\n"
+            "   :param age: blah blah\n"
+            "   :type age: Literal['foo', 'bar', 'baz']\n")
+    doctree = restructuredtext.parse(app, text)
+
+    assert_node(doctree,
+                (nodes.target,
+                 addnodes.index,
+                 addnodes.index,
+                 [desc, ([desc_signature, ([desc_annotation, ("class", desc_sig_space)],
+                                           [desc_addname, "example."],
+                                           [desc_name, "Class"])],
+                         [desc_content, nodes.field_list, nodes.field, (nodes.field_name,
+                                                                        nodes.field_body)])]))
+    assert_node(doctree[3][1][0][0][1],
+                ([nodes.paragraph, ([addnodes.literal_strong, "age"],
+                                    " (",
+                                    [pending_xref, addnodes.literal_emphasis, "Literal"],
+                                    [addnodes.literal_emphasis, "["],
+                                    [addnodes.literal_emphasis, "'foo'"],
+                                    [addnodes.literal_emphasis, ", "],
+                                    [addnodes.literal_emphasis, "'bar'"],
+                                    [addnodes.literal_emphasis, ", "],
+                                    [addnodes.literal_emphasis, "'baz'"],
+                                    [addnodes.literal_emphasis, "]"],
+                                    ")",
+                                    " -- ",
+                                    "blah blah")],))
+    assert_node(doctree[3][1][0][0][1][0][2], pending_xref,
+                refdomain="py", reftype="class", reftarget="Literal",
+                **{"py:module": "example", "py:class": "Class"})
+
+
 def test_info_field_list_var(app):
     text = (".. py:class:: Class\n"
             "\n"

```


## Code snippets

### 1 - doc/conf.py:

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
### 2 - sphinx/ext/autosummary/generate.py:

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
### 3 - setup.py:

Start line: 176, End line: 253

```python
setup(
    name='Sphinx',
    version=sphinx.__version__,
    url='https://www.sphinx-doc.org/',
    download_url='https://pypi.org/project/Sphinx/',
    license='BSD',
    author='Georg Brandl',
    author_email='georg@python.org',
    description='Python documentation generator',
    long_description=long_desc,
    long_description_content_type='text/x-rst',
    project_urls={
        "Code": "https://github.com/sphinx-doc/sphinx",
        "Issue tracker": "https://github.com/sphinx-doc/sphinx/issues",
    },
    zip_safe=False,
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Environment :: Web Environment',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Science/Research',
        'Intended Audience :: System Administrators',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy',
        'Framework :: Setuptools Plugin',
        'Framework :: Sphinx',
        'Framework :: Sphinx :: Extension',
        'Framework :: Sphinx :: Theme',
        'Topic :: Documentation',
        'Topic :: Documentation :: Sphinx',
        'Topic :: Internet :: WWW/HTTP :: Site Management',
        'Topic :: Printing',
        'Topic :: Software Development',
        'Topic :: Software Development :: Documentation',
        'Topic :: Text Processing',
        'Topic :: Text Processing :: General',
        'Topic :: Text Processing :: Indexing',
        'Topic :: Text Processing :: Markup',
        'Topic :: Text Processing :: Markup :: HTML',
        'Topic :: Text Processing :: Markup :: LaTeX',
        'Topic :: Utilities',
    ],
    platforms='any',
    packages=find_packages(exclude=['tests', 'utils']),
    package_data = {
        'sphinx': ['py.typed'],
    },
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'sphinx-build = sphinx.cmd.build:main',
            'sphinx-quickstart = sphinx.cmd.quickstart:main',
            'sphinx-apidoc = sphinx.ext.apidoc:main',
            'sphinx-autogen = sphinx.ext.autosummary.generate:main',
        ],
        'distutils.commands': [
            'build_sphinx = sphinx.setup_command:BuildDoc',
        ],
    },
    python_requires=">=3.6",
    install_requires=install_requires,
    extras_require=extras_require,
    cmdclass=cmdclass,
)
```
### 4 - sphinx/ext/autodoc/__init__.py:

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
### 5 - sphinx/environment/__init__.py:

Start line: 11, End line: 85

```python
import os
import pickle
import warnings
from collections import defaultdict
from copy import copy
from datetime import datetime
from os import path
from typing import (TYPE_CHECKING, Any, Callable, Dict, Generator, Iterator, List, Optional,
                    Set, Tuple, Union)

from docutils import nodes
from docutils.nodes import Node

from sphinx import addnodes
from sphinx.config import Config
from sphinx.deprecation import RemovedInSphinx60Warning
from sphinx.domains import Domain
from sphinx.environment.adapters.toctree import TocTree
from sphinx.errors import BuildEnvironmentError, DocumentError, ExtensionError, SphinxError
from sphinx.events import EventManager
from sphinx.locale import __
from sphinx.project import Project
from sphinx.transforms import SphinxTransformer
from sphinx.util import DownloadFiles, FilenameUniqDict, logging
from sphinx.util.docutils import LoggingReporter
from sphinx.util.i18n import CatalogRepository, docname_to_domain
from sphinx.util.nodes import is_translatable
from sphinx.util.osutil import canon_path, os_path

if TYPE_CHECKING:
    from sphinx.application import Sphinx
    from sphinx.builders import Builder


logger = logging.getLogger(__name__)

default_settings: Dict[str, Any] = {
    'auto_id_prefix': 'id',
    'embed_images': False,
    'embed_stylesheet': False,
    'cloak_email_addresses': True,
    'pep_base_url': 'https://www.python.org/dev/peps/',
    'pep_references': None,
    'rfc_base_url': 'https://tools.ietf.org/html/',
    'rfc_references': None,
    'input_encoding': 'utf-8-sig',
    'doctitle_xform': False,
    'sectsubtitle_xform': False,
    'section_self_link': False,
    'halt_level': 5,
    'file_insertion_enabled': True,
    'smartquotes_locales': [],
}

# This is increased every time an environment attribute is added
# or changed to properly invalidate pickle files.
ENV_VERSION = 56

# config status
CONFIG_OK = 1
CONFIG_NEW = 2
CONFIG_CHANGED = 3
CONFIG_EXTENSIONS_CHANGED = 4

CONFIG_CHANGED_REASON = {
    CONFIG_NEW: __('new config'),
    CONFIG_CHANGED: __('config changed'),
    CONFIG_EXTENSIONS_CHANGED: __('extensions changed'),
}


versioning_conditions: Dict[str, Union[bool, Callable]] = {
    'none': False,
    'text': is_translatable,
}
```
### 6 - sphinx/domains/python.py:

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
### 7 - sphinx/builders/__init__.py:

Start line: 11, End line: 48

```python
import pickle
import time
from os import path
from typing import (TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple,
                    Type, Union)

from docutils import nodes
from docutils.nodes import Node

from sphinx.config import Config
from sphinx.environment import CONFIG_CHANGED_REASON, CONFIG_OK, BuildEnvironment
from sphinx.environment.adapters.asset import ImageAdapter
from sphinx.errors import SphinxError
from sphinx.events import EventManager
from sphinx.io import read_doc
from sphinx.locale import __
from sphinx.util import import_object, logging, progress_message, rst, status_iterator
from sphinx.util.build_phase import BuildPhase
from sphinx.util.console import bold  # type: ignore
from sphinx.util.docutils import sphinx_domains
from sphinx.util.i18n import CatalogInfo, CatalogRepository, docname_to_domain
from sphinx.util.osutil import SEP, ensuredir, relative_uri, relpath
from sphinx.util.parallel import ParallelTasks, SerialTasks, make_chunks, parallel_available
from sphinx.util.tags import Tags

# side effect: registers roles and directives
from sphinx import directives  # NOQA isort:skip
from sphinx import roles  # NOQA isort:skip
try:
    import multiprocessing
except ImportError:
    multiprocessing = None

if TYPE_CHECKING:
    from sphinx.application import Sphinx


logger = logging.getLogger(__name__)
```
### 8 - sphinx/ext/autodoc/typehints.py:

Start line: 130, End line: 185

```python
def augment_descriptions_with_types(
    node: nodes.field_list,
    annotations: Dict[str, str],
) -> None:
    fields = cast(Iterable[nodes.field], node)
    has_description = set()  # type: Set[str]
    has_type = set()  # type: Set[str]
    for field in fields:
        field_name = field[0].astext()
        parts = re.split(' +', field_name)
        if parts[0] == 'param':
            if len(parts) == 2:
                # :param xxx:
                has_description.add(parts[1])
            elif len(parts) > 2:
                # :param xxx yyy:
                name = ' '.join(parts[2:])
                has_description.add(name)
                has_type.add(name)
        elif parts[0] == 'type':
            name = ' '.join(parts[1:])
            has_type.add(name)
        elif parts[0] in ('return', 'returns'):
            has_description.add('return')
        elif parts[0] == 'rtype':
            has_type.add('return')

    # Add 'type' for parameters with a description but no declared type.
    for name in annotations:
        if name in ('return', 'returns'):
            continue
        if name in has_description and name not in has_type:
            field = nodes.field()
            field += nodes.field_name('', 'type ' + name)
            field += nodes.field_body('', nodes.paragraph('', annotations[name]))
            node += field

    # Add 'rtype' if 'return' is present and 'rtype' isn't.
    if 'return' in annotations:
        if 'return' in has_description and 'return' not in has_type:
            field = nodes.field()
            field += nodes.field_name('', 'rtype')
            field += nodes.field_body('', nodes.paragraph('', annotations['return']))
            node += field


def setup(app: Sphinx) -> Dict[str, Any]:
    app.connect('autodoc-process-signature', record_typehints)
    app.connect('object-description-transform', merge_typehints)

    return {
        'version': 'builtin',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
```
### 9 - sphinx/errors.py:

Start line: 77, End line: 134

```python
class BuildEnvironmentError(SphinxError):
    """BuildEnvironment error."""
    category = 'BuildEnvironment error'


class ConfigError(SphinxError):
    """Configuration error."""
    category = 'Configuration error'


class DocumentError(SphinxError):
    """Document error."""
    category = 'Document error'


class ThemeError(SphinxError):
    """Theme error."""
    category = 'Theme error'


class VersionRequirementError(SphinxError):
    """Incompatible Sphinx version error."""
    category = 'Sphinx version error'


class SphinxParallelError(SphinxError):
    """Sphinx parallel build error."""

    category = 'Sphinx parallel build error'

    def __init__(self, message: str, traceback: Any) -> None:
        self.message = message
        self.traceback = traceback

    def __str__(self) -> str:
        return self.message


class PycodeError(Exception):
    """Pycode Python source code analyser error."""

    def __str__(self) -> str:
        res = self.args[0]
        if len(self.args) > 1:
            res += ' (exception was: %r)' % self.args[1]
        return res


class NoUri(Exception):
    """Raised by builder.get_relative_uri() or from missing-reference handlers
    if there is no URI available."""
    pass


class FiletypeNotFoundError(Exception):
    """Raised by get_filetype() if a filename matches no source suffix."""
    pass
```
### 10 - utils/checks.py:

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
