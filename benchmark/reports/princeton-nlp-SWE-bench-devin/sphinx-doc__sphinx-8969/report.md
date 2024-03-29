# sphinx-doc__sphinx-8969

| **sphinx-doc/sphinx** | `ae413e95ed6fd2b3a9d579a3d802e38846906b54` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | 115 |
| **Avg pos** | 1.0 |
| **Min pos** | 1 |
| **Max pos** | 1 |
| **Top file pos** | 1 |
| **Missing snippets** | 4 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sphinx/directives/patches.py b/sphinx/directives/patches.py
--- a/sphinx/directives/patches.py
+++ b/sphinx/directives/patches.py
@@ -6,7 +6,9 @@
     :license: BSD, see LICENSE for details.
 """
 
+import os
 import warnings
+from os import path
 from typing import TYPE_CHECKING, Any, Dict, List, Tuple, cast
 
 from docutils import nodes
@@ -18,13 +20,19 @@
 from sphinx.deprecation import RemovedInSphinx60Warning
 from sphinx.directives import optional_int
 from sphinx.domains.math import MathDomain
+from sphinx.locale import __
+from sphinx.util import logging
 from sphinx.util.docutils import SphinxDirective
 from sphinx.util.nodes import set_source_info
+from sphinx.util.osutil import SEP, os_path, relpath
 
 if TYPE_CHECKING:
     from sphinx.application import Sphinx
 
 
+logger = logging.getLogger(__name__)
+
+
 class Figure(images.Figure):
     """The figure directive which applies `:name:` option to the figure node
     instead of the image node.
@@ -87,21 +95,25 @@ def make_title(self) -> Tuple[nodes.title, List[system_message]]:
 
 
 class CSVTable(tables.CSVTable):
-    """The csv-table directive which sets source and line information to its caption.
-
-    Only for docutils-0.13 or older version."""
+    """The csv-table directive which searches a CSV file from Sphinx project's source
+    directory when an absolute path is given via :file: option.
+    """
 
     def run(self) -> List[Node]:
-        warnings.warn('CSVTable is deprecated.',
-                      RemovedInSphinx60Warning)
-        return super().run()
+        if 'file' in self.options and self.options['file'].startswith((SEP, os.sep)):
+            env = self.state.document.settings.env
+            filename = self.options['file']
+            if path.exists(filename):
+                logger.warning(__('":file:" option for csv-table directive now recognizes '
+                                  'an absolute path as a relative path from source directory. '
+                                  'Please update your document.'),
+                               location=(env.docname, self.lineno))
+            else:
+                abspath = path.join(env.srcdir, os_path(self.options['file'][1:]))
+                docdir = path.dirname(env.doc2path(env.docname))
+                self.options['file'] = relpath(abspath, docdir)
 
-    def make_title(self) -> Tuple[nodes.title, List[system_message]]:
-        title, message = super().make_title()
-        if title:
-            set_source_info(self, title)
-
-        return title, message
+        return super().run()
 
 
 class ListTable(tables.ListTable):
@@ -224,6 +236,7 @@ def add_target(self, ret: List[Node]) -> None:
 def setup(app: "Sphinx") -> Dict[str, Any]:
     directives.register_directive('figure', Figure)
     directives.register_directive('meta', Meta)
+    directives.register_directive('csv-table', CSVTable)
     directives.register_directive('code', Code)
     directives.register_directive('math', MathDirective)
 

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sphinx/directives/patches.py | 9 | 9 | - | 1 | -
| sphinx/directives/patches.py | 21 | 21 | - | 1 | -
| sphinx/directives/patches.py | 90 | 104 | 1 | 1 | 115
| sphinx/directives/patches.py | 227 | 227 | - | 1 | -


## Problem Statement

```
Absolute/"source-relative" imports for csv-table :file:
**Describe the bug**
To be honest, I am not sure if this can be considered a bug, or if it is rather a feature request. Sorry about that.

When using the `csv-table` directive, the use of `:file:` with absolute paths are really absolute, unlike with (eg) the `figure` directive, where absolute paths are treated relative to the source directory (herein called "source-relative").

I do understand that there is a difference in the 2 cases, because with `figure` the path is not specified in `:file:`. Yet, I do not see a possibility to mimic this behavior in the `cvs-tables` directive.

**To Reproduce**
A `phone_list.rst` file in `source/resources`:

- Relative imports:
\`\`\`rst
.. csv-table:: The group's phone and room list
   :align: center
   :file: _tables/phone_list.csv
   :header-rows: 1
\`\`\`
are treated, as expected, relative to the `.rst` file:
\`\`\`
C:\Users\lcnittl\project\docs\source\resources\phone_list.rst:13: WARNING: Problems with "csv-table" directive path:
[Errno 2] No such file or directory: 'source/resources/_tables/phone_list.csv'.

.. csv-table:: The group's phone and room list
   :align: center
   :file: _tables/phone_list.csv
   :header-rows: 1
\`\`\`

- Absolute imports:
\`\`\`rst
.. csv-table:: The group's phone and room list
   :align: center
   :file: /_tables/phone_list.csv
   :header-rows: 1
\`\`\`
are treated, opposed to my expectations, like real absolute paths:
\`\`\`
C:\Users\lcnittl\project\docs\source\resources\phone_list.rst:13: WARNING: Problems with "csv-table" directive path:
[Errno 2] No such file or directory: 'C:/_tables/phone_list.csv'.

.. csv-table:: The group's phone and room list
   :align: center
   :file: /_tables/phone_list.csv
   :header-rows: 1
\`\`\`
and not like relative-to-source paths.

**Expected behavior**
I would expect this to work like absolute paths in the (eg) `figure` directive.

But as stated in the beginning, probably I am wrong with my expectation, and this should be a feature request to add an option to use "source-relative" paths with the `csv-table` directive.

**Environment info**
- OS: Win
- Python version: 3.8.5
- Sphinx version: 3.2.1


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 sphinx/directives/patches.py** | 89 | 104| 115 | 115 | 1690 | 
| 2 | 2 sphinx/directives/other.py | 351 | 365| 136 | 251 | 4839 | 
| 3 | **2 sphinx/directives/patches.py** | 71 | 86| 116 | 367 | 4839 | 
| 4 | 3 sphinx/util/osutil.py | 71 | 171| 718 | 1085 | 6476 | 
| 5 | 4 sphinx/directives/code.py | 406 | 469| 646 | 1731 | 10316 | 
| 6 | **4 sphinx/directives/patches.py** | 107 | 122| 114 | 1845 | 10316 | 
| 7 | 5 sphinx/writers/text.py | 131 | 176| 425 | 2270 | 19291 | 
| 8 | 6 sphinx/testing/path.py | 32 | 81| 269 | 2539 | 20954 | 
| 9 | 7 sphinx/builders/html/__init__.py | 1175 | 1192| 160 | 2699 | 33195 | 
| 10 | 8 sphinx/writers/texinfo.py | 864 | 968| 788 | 3487 | 45508 | 
| 11 | 8 sphinx/writers/text.py | 766 | 873| 830 | 4317 | 45508 | 
| 12 | **8 sphinx/directives/patches.py** | 28 | 51| 187 | 4504 | 45508 | 
| 13 | 9 sphinx/util/nodes.py | 121 | 175| 683 | 5187 | 50979 | 
| 14 | 10 sphinx/ext/autosummary/__init__.py | 717 | 752| 325 | 5512 | 57436 | 
| 15 | 10 sphinx/directives/other.py | 190 | 211| 141 | 5653 | 57436 | 
| 16 | 11 sphinx/environment/__init__.py | 330 | 351| 192 | 5845 | 62990 | 
| 17 | 12 sphinx/util/__init__.py | 11 | 64| 440 | 6285 | 67748 | 
| 18 | 12 sphinx/util/osutil.py | 11 | 45| 214 | 6499 | 67748 | 
| 19 | 12 sphinx/writers/texinfo.py | 970 | 1087| 846 | 7345 | 67748 | 
| 20 | 13 sphinx/domains/c.py | 3552 | 3867| 2965 | 10310 | 99350 | 
| 21 | 13 sphinx/testing/path.py | 123 | 236| 802 | 11112 | 99350 | 
| 22 | 14 sphinx/errors.py | 77 | 134| 297 | 11409 | 100146 | 
| 23 | 15 sphinx/writers/html.py | 694 | 797| 827 | 12236 | 107557 | 
| 24 | 16 sphinx/builders/__init__.py | 153 | 173| 195 | 12431 | 112935 | 
| 25 | 16 sphinx/directives/code.py | 179 | 208| 248 | 12679 | 112935 | 
| 26 | 17 sphinx/transforms/i18n.py | 226 | 380| 1653 | 14332 | 117579 | 
| 27 | 18 sphinx/util/cfamily.py | 11 | 76| 747 | 15079 | 120899 | 
| 28 | 18 sphinx/directives/other.py | 40 | 59| 151 | 15230 | 120899 | 
| 29 | 18 sphinx/writers/text.py | 227 | 252| 227 | 15457 | 120899 | 
| 30 | 19 sphinx/builders/latex/constants.py | 74 | 124| 540 | 15997 | 123098 | 
| 31 | 19 sphinx/directives/other.py | 368 | 392| 228 | 16225 | 123098 | 
| 32 | 19 sphinx/writers/texinfo.py | 1089 | 1188| 839 | 17064 | 123098 | 
| 33 | 20 sphinx/builders/latex/__init__.py | 44 | 99| 825 | 17889 | 128816 | 
| 34 | 20 sphinx/directives/code.py | 226 | 248| 196 | 18085 | 128816 | 
| 35 | 21 sphinx/io.py | 10 | 39| 234 | 18319 | 130225 | 
| 36 | 21 sphinx/writers/text.py | 929 | 1042| 873 | 19192 | 130225 | 
| 37 | 22 sphinx/util/smartypants.py | 28 | 127| 1450 | 20642 | 134372 | 
| 38 | 23 sphinx/environment/adapters/toctree.py | 11 | 47| 324 | 20966 | 137673 | 
| 39 | 23 sphinx/writers/texinfo.py | 1302 | 1385| 678 | 21644 | 137673 | 
| 40 | 23 sphinx/writers/text.py | 54 | 118| 512 | 22156 | 137673 | 
| 41 | 24 doc/development/tutorials/examples/todo.py | 30 | 53| 175 | 22331 | 138581 | 
| 42 | 24 sphinx/writers/texinfo.py | 754 | 862| 813 | 23144 | 138581 | 


### Hint

```
+1: I agree this is inconsistent behavior. It should behave like the figure directive. But changing the behavior is an incompatible change. So we have to change it carefully...
I'd greatly appreciate this change too. Changing the behavior of `:file:` flag for `csv-table` to mimic that of `figure` directive would be great. As it stands with `csv-table`, even when using an absolute path and the file fails to load, it fails silently.  
```

## Patch

```diff
diff --git a/sphinx/directives/patches.py b/sphinx/directives/patches.py
--- a/sphinx/directives/patches.py
+++ b/sphinx/directives/patches.py
@@ -6,7 +6,9 @@
     :license: BSD, see LICENSE for details.
 """
 
+import os
 import warnings
+from os import path
 from typing import TYPE_CHECKING, Any, Dict, List, Tuple, cast
 
 from docutils import nodes
@@ -18,13 +20,19 @@
 from sphinx.deprecation import RemovedInSphinx60Warning
 from sphinx.directives import optional_int
 from sphinx.domains.math import MathDomain
+from sphinx.locale import __
+from sphinx.util import logging
 from sphinx.util.docutils import SphinxDirective
 from sphinx.util.nodes import set_source_info
+from sphinx.util.osutil import SEP, os_path, relpath
 
 if TYPE_CHECKING:
     from sphinx.application import Sphinx
 
 
+logger = logging.getLogger(__name__)
+
+
 class Figure(images.Figure):
     """The figure directive which applies `:name:` option to the figure node
     instead of the image node.
@@ -87,21 +95,25 @@ def make_title(self) -> Tuple[nodes.title, List[system_message]]:
 
 
 class CSVTable(tables.CSVTable):
-    """The csv-table directive which sets source and line information to its caption.
-
-    Only for docutils-0.13 or older version."""
+    """The csv-table directive which searches a CSV file from Sphinx project's source
+    directory when an absolute path is given via :file: option.
+    """
 
     def run(self) -> List[Node]:
-        warnings.warn('CSVTable is deprecated.',
-                      RemovedInSphinx60Warning)
-        return super().run()
+        if 'file' in self.options and self.options['file'].startswith((SEP, os.sep)):
+            env = self.state.document.settings.env
+            filename = self.options['file']
+            if path.exists(filename):
+                logger.warning(__('":file:" option for csv-table directive now recognizes '
+                                  'an absolute path as a relative path from source directory. '
+                                  'Please update your document.'),
+                               location=(env.docname, self.lineno))
+            else:
+                abspath = path.join(env.srcdir, os_path(self.options['file'][1:]))
+                docdir = path.dirname(env.doc2path(env.docname))
+                self.options['file'] = relpath(abspath, docdir)
 
-    def make_title(self) -> Tuple[nodes.title, List[system_message]]:
-        title, message = super().make_title()
-        if title:
-            set_source_info(self, title)
-
-        return title, message
+        return super().run()
 
 
 class ListTable(tables.ListTable):
@@ -224,6 +236,7 @@ def add_target(self, ret: List[Node]) -> None:
 def setup(app: "Sphinx") -> Dict[str, Any]:
     directives.register_directive('figure', Figure)
     directives.register_directive('meta', Meta)
+    directives.register_directive('csv-table', CSVTable)
     directives.register_directive('code', Code)
     directives.register_directive('math', MathDirective)
 

```

## Test Patch

```diff
diff --git a/tests/roots/test-directive-csv-table/conf.py b/tests/roots/test-directive-csv-table/conf.py
new file mode 100644
diff --git a/tests/roots/test-directive-csv-table/example.csv b/tests/roots/test-directive-csv-table/example.csv
new file mode 100644
--- /dev/null
+++ b/tests/roots/test-directive-csv-table/example.csv
@@ -0,0 +1 @@
+foo,bar,baz
diff --git a/tests/roots/test-directive-csv-table/subdir/example.csv b/tests/roots/test-directive-csv-table/subdir/example.csv
new file mode 100644
--- /dev/null
+++ b/tests/roots/test-directive-csv-table/subdir/example.csv
@@ -0,0 +1 @@
+FOO,BAR,BAZ
diff --git a/tests/test_directive_patch.py b/tests/test_directive_patch.py
--- a/tests/test_directive_patch.py
+++ b/tests/test_directive_patch.py
@@ -8,6 +8,7 @@
     :license: BSD, see LICENSE for details.
 """
 
+import pytest
 from docutils import nodes
 
 from sphinx.testing import restructuredtext
@@ -54,6 +55,37 @@ def test_code_directive(app):
     assert_node(doctree[0], language="python", linenos=True, highlight_args={'linenostart': 5})
 
 
+@pytest.mark.sphinx(testroot='directive-csv-table')
+def test_csv_table_directive(app):
+    # relative path from current document
+    text = ('.. csv-table::\n'
+            '   :file: example.csv\n')
+    doctree = restructuredtext.parse(app, text, docname="subdir/index")
+    assert_node(doctree,
+                ([nodes.table, nodes.tgroup, (nodes.colspec,
+                                              nodes.colspec,
+                                              nodes.colspec,
+                                              [nodes.tbody, nodes.row])],))
+    assert_node(doctree[0][0][3][0],
+                ([nodes.entry, nodes.paragraph, "FOO"],
+                 [nodes.entry, nodes.paragraph, "BAR"],
+                 [nodes.entry, nodes.paragraph, "BAZ"]))
+
+    # absolute path from source directory
+    text = ('.. csv-table::\n'
+            '   :file: /example.csv\n')
+    doctree = restructuredtext.parse(app, text, docname="subdir/index")
+    assert_node(doctree,
+                ([nodes.table, nodes.tgroup, (nodes.colspec,
+                                              nodes.colspec,
+                                              nodes.colspec,
+                                              [nodes.tbody, nodes.row])],))
+    assert_node(doctree[0][0][3][0],
+                ([nodes.entry, nodes.paragraph, "foo"],
+                 [nodes.entry, nodes.paragraph, "bar"],
+                 [nodes.entry, nodes.paragraph, "baz"]))
+
+
 def test_math_directive(app):
     # normal case
     text = '.. math:: E = mc^2'

```


## Code snippets

### 1 - sphinx/directives/patches.py:

Start line: 89, End line: 104

```python
class CSVTable(tables.CSVTable):
    """The csv-table directive which sets source and line information to its caption.

    Only for docutils-0.13 or older version."""

    def run(self) -> List[Node]:
        warnings.warn('CSVTable is deprecated.',
                      RemovedInSphinx60Warning)
        return super().run()

    def make_title(self) -> Tuple[nodes.title, List[system_message]]:
        title, message = super().make_title()
        if title:
            set_source_info(self, title)

        return title, message
```
### 2 - sphinx/directives/other.py:

Start line: 351, End line: 365

```python
class Include(BaseInclude, SphinxDirective):
    """
    Like the standard "Include" directive, but interprets absolute paths
    "correctly", i.e. relative to source directory.
    """

    def run(self) -> List[Node]:
        if self.arguments[0].startswith('<') and \
           self.arguments[0].endswith('>'):
            # docutils "standard" includes, do not do path processing
            return super().run()
        rel_filename, filename = self.env.relfn2path(self.arguments[0])
        self.arguments[0] = filename
        self.env.note_included(filename)
        return super().run()
```
### 3 - sphinx/directives/patches.py:

Start line: 71, End line: 86

```python
class RSTTable(tables.RSTTable):
    """The table directive which sets source and line information to its caption.

    Only for docutils-0.13 or older version."""

    def run(self) -> List[Node]:
        warnings.warn('RSTTable is deprecated.',
                      RemovedInSphinx60Warning)
        return super().run()

    def make_title(self) -> Tuple[nodes.title, List[system_message]]:
        title, message = super().make_title()
        if title:
            set_source_info(self, title)

        return title, message
```
### 4 - sphinx/util/osutil.py:

Start line: 71, End line: 171

```python
def ensuredir(path: str) -> None:
    """Ensure that a path exists."""
    os.makedirs(path, exist_ok=True)


def mtimes_of_files(dirnames: List[str], suffix: str) -> Iterator[float]:
    for dirname in dirnames:
        for root, dirs, files in os.walk(dirname):
            for sfile in files:
                if sfile.endswith(suffix):
                    try:
                        yield path.getmtime(path.join(root, sfile))
                    except OSError:
                        pass


def movefile(source: str, dest: str) -> None:
    """Move a file, removing the destination if it exists."""
    warnings.warn('sphinx.util.osutil.movefile() is deprecated for removal. '
                  'Please use os.replace() instead.',
                  RemovedInSphinx50Warning, stacklevel=2)
    if os.path.exists(dest):
        try:
            os.unlink(dest)
        except OSError:
            pass
    os.rename(source, dest)


def copytimes(source: str, dest: str) -> None:
    """Copy a file's modification times."""
    st = os.stat(source)
    if hasattr(os, 'utime'):
        os.utime(dest, (st.st_atime, st.st_mtime))


def copyfile(source: str, dest: str) -> None:
    """Copy a file and its modification times, if possible.

    Note: ``copyfile`` skips copying if the file has not been changed"""
    if not path.exists(dest) or not filecmp.cmp(source, dest):
        shutil.copyfile(source, dest)
        try:
            # don't do full copystat because the source may be read-only
            copytimes(source, dest)
        except OSError:
            pass


no_fn_re = re.compile(r'[^a-zA-Z0-9_-]')
project_suffix_re = re.compile(' Documentation$')


def make_filename(string: str) -> str:
    return no_fn_re.sub('', string) or 'sphinx'


def make_filename_from_project(project: str) -> str:
    return make_filename(project_suffix_re.sub('', project)).lower()


def relpath(path: str, start: str = os.curdir) -> str:
    """Return a relative filepath to *path* either from the current directory or
    from an optional *start* directory.

    This is an alternative of ``os.path.relpath()``.  This returns original path
    if *path* and *start* are on different drives (for Windows platform).
    """
    try:
        return os.path.relpath(path, start)
    except ValueError:
        return path


safe_relpath = relpath  # for compatibility
fs_encoding = sys.getfilesystemencoding() or sys.getdefaultencoding()


def abspath(pathdir: str) -> str:
    if Path is not None and isinstance(pathdir, Path):
        return pathdir.abspath()
    else:
        pathdir = path.abspath(pathdir)
        if isinstance(pathdir, bytes):
            try:
                pathdir = pathdir.decode(fs_encoding)
            except UnicodeDecodeError as exc:
                raise UnicodeDecodeError('multibyte filename not supported on '
                                         'this filesystem encoding '
                                         '(%r)' % fs_encoding) from exc
        return pathdir


@contextlib.contextmanager
def cd(target_dir: str) -> Generator[None, None, None]:
    cwd = os.getcwd()
    try:
        os.chdir(target_dir)
        yield
    finally:
        os.chdir(cwd)
```
### 5 - sphinx/directives/code.py:

Start line: 406, End line: 469

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

            retnode = nodes.literal_block(text, text, source=filename)  # type: Element
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
### 6 - sphinx/directives/patches.py:

Start line: 107, End line: 122

```python
class ListTable(tables.ListTable):
    """The list-table directive which sets source and line information to its caption.

    Only for docutils-0.13 or older version."""

    def run(self) -> List[Node]:
        warnings.warn('ListTable is deprecated.',
                      RemovedInSphinx60Warning)
        return super().run()

    def make_title(self) -> Tuple[nodes.title, List[system_message]]:
        title, message = super().make_title()
        if title:
            set_source_info(self, title)

        return title, message
```
### 7 - sphinx/writers/text.py:

Start line: 131, End line: 176

```python
class Table:

    def __getitem__(self, pos: Tuple[int, int]) -> Cell:
        line, col = pos
        self._ensure_has_line(line + 1)
        self._ensure_has_column(col + 1)
        return self.lines[line][col]

    def __setitem__(self, pos: Tuple[int, int], cell: Cell) -> None:
        line, col = pos
        self._ensure_has_line(line + cell.rowspan)
        self._ensure_has_column(col + cell.colspan)
        for dline in range(cell.rowspan):
            for dcol in range(cell.colspan):
                self.lines[line + dline][col + dcol] = cell
                cell.row = line
                cell.col = col

    def _ensure_has_line(self, line: int) -> None:
        while len(self.lines) < line:
            self.lines.append([])

    def _ensure_has_column(self, col: int) -> None:
        for line in self.lines:
            while len(line) < col:
                line.append(None)

    def __repr__(self) -> str:
        return "\n".join(repr(line) for line in self.lines)

    def cell_width(self, cell: Cell, source: List[int]) -> int:
        """Give the cell width, according to the given source (either
        ``self.colwidth`` or ``self.measured_widths``).
        This take into account cells spanning on multiple columns.
        """
        width = 0
        for i in range(self[cell.row, cell.col].colspan):
            width += source[cell.col + i]
        return width + (cell.colspan - 1) * 3

    @property
    def cells(self) -> Generator[Cell, None, None]:
        seen = set()  # type: Set[Cell]
        for lineno, line in enumerate(self.lines):
            for colno, cell in enumerate(line):
                if cell and cell not in seen:
                    yield cell
                    seen.add(cell)
```
### 8 - sphinx/testing/path.py:

Start line: 32, End line: 81

```python
class path(str):
    """
    Represents a path which behaves like a string.
    """

    @property
    def parent(self) -> "path":
        """
        The name of the directory the file or directory is in.
        """
        return self.__class__(os.path.dirname(self))

    def basename(self) -> str:
        return os.path.basename(self)

    def abspath(self) -> "path":
        """
        Returns the absolute path.
        """
        return self.__class__(os.path.abspath(self))

    def isabs(self) -> bool:
        """
        Returns ``True`` if the path is absolute.
        """
        return os.path.isabs(self)

    def isdir(self) -> bool:
        """
        Returns ``True`` if the path is a directory.
        """
        return os.path.isdir(self)

    def isfile(self) -> bool:
        """
        Returns ``True`` if the path is a file.
        """
        return os.path.isfile(self)

    def islink(self) -> bool:
        """
        Returns ``True`` if the path is a symbolic link.
        """
        return os.path.islink(self)

    def ismount(self) -> bool:
        """
        Returns ``True`` if the path is a mount point.
        """
        return os.path.ismount(self)
```
### 9 - sphinx/builders/html/__init__.py:

Start line: 1175, End line: 1192

```python
def setup_resource_paths(app: Sphinx, pagename: str, templatename: str,
                         context: Dict, doctree: Node) -> None:
    """Set up relative resource paths."""
    pathto = context.get('pathto')

    # favicon_url
    favicon = context.get('favicon')
    if favicon and not isurl(favicon):
        context['favicon_url'] = pathto('_static/' + favicon, resource=True)
    else:
        context['favicon_url'] = favicon

    # logo_url
    logo = context.get('logo')
    if logo and not isurl(logo):
        context['logo_url'] = pathto('_static/' + logo, resource=True)
    else:
        context['logo_url'] = logo
```
### 10 - sphinx/writers/texinfo.py:

Start line: 864, End line: 968

```python
class TexinfoTranslator(SphinxTranslator):

    def visit_citation(self, node: Element) -> None:
        self.body.append('\n')
        for id in node.get('ids'):
            self.add_anchor(id, node)
        self.escape_newlines += 1

    def depart_citation(self, node: Element) -> None:
        self.escape_newlines -= 1

    def visit_citation_reference(self, node: Element) -> None:
        self.body.append('@w{[')

    def depart_citation_reference(self, node: Element) -> None:
        self.body.append(']}')

    # -- Lists

    def visit_bullet_list(self, node: Element) -> None:
        bullet = node.get('bullet', '*')
        self.body.append('\n\n@itemize %s\n' % bullet)

    def depart_bullet_list(self, node: Element) -> None:
        self.ensure_eol()
        self.body.append('@end itemize\n')

    def visit_enumerated_list(self, node: Element) -> None:
        # doesn't support Roman numerals
        enum = node.get('enumtype', 'arabic')
        starters = {'arabic': '',
                    'loweralpha': 'a',
                    'upperalpha': 'A'}
        start = node.get('start', starters.get(enum, ''))
        self.body.append('\n\n@enumerate %s\n' % start)

    def depart_enumerated_list(self, node: Element) -> None:
        self.ensure_eol()
        self.body.append('@end enumerate\n')

    def visit_list_item(self, node: Element) -> None:
        self.body.append('\n@item ')

    def depart_list_item(self, node: Element) -> None:
        pass

    # -- Option List

    def visit_option_list(self, node: Element) -> None:
        self.body.append('\n\n@table @option\n')

    def depart_option_list(self, node: Element) -> None:
        self.ensure_eol()
        self.body.append('@end table\n')

    def visit_option_list_item(self, node: Element) -> None:
        pass

    def depart_option_list_item(self, node: Element) -> None:
        pass

    def visit_option_group(self, node: Element) -> None:
        self.at_item_x = '@item'

    def depart_option_group(self, node: Element) -> None:
        pass

    def visit_option(self, node: Element) -> None:
        self.escape_hyphens += 1
        self.body.append('\n%s ' % self.at_item_x)
        self.at_item_x = '@itemx'

    def depart_option(self, node: Element) -> None:
        self.escape_hyphens -= 1

    def visit_option_string(self, node: Element) -> None:
        pass

    def depart_option_string(self, node: Element) -> None:
        pass

    def visit_option_argument(self, node: Element) -> None:
        self.body.append(node.get('delimiter', ' '))

    def depart_option_argument(self, node: Element) -> None:
        pass

    def visit_description(self, node: Element) -> None:
        self.body.append('\n')

    def depart_description(self, node: Element) -> None:
        pass

    # -- Definitions

    def visit_definition_list(self, node: Element) -> None:
        self.body.append('\n\n@table @asis\n')

    def depart_definition_list(self, node: Element) -> None:
        self.ensure_eol()
        self.body.append('@end table\n')

    def visit_definition_list_item(self, node: Element) -> None:
        self.at_item_x = '@item'

    def depart_definition_list_item(self, node: Element) -> None:
        pass
```
### 12 - sphinx/directives/patches.py:

Start line: 28, End line: 51

```python
class Figure(images.Figure):
    """The figure directive which applies `:name:` option to the figure node
    instead of the image node.
    """

    def run(self) -> List[Node]:
        name = self.options.pop('name', None)
        result = super().run()
        if len(result) == 2 or isinstance(result[0], nodes.system_message):
            return result

        assert len(result) == 1
        figure_node = cast(nodes.figure, result[0])
        if name:
            # set ``name`` to figure_node if given
            self.options['name'] = name
            self.add_name(figure_node)

        # copy lineno from image node
        if figure_node.line is None and len(figure_node) == 2:
            caption = cast(nodes.caption, figure_node[1])
            figure_node.line = caption.line

        return [figure_node]
```
