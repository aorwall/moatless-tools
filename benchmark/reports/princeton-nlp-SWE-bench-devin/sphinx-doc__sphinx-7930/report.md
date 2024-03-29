# sphinx-doc__sphinx-7930

| **sphinx-doc/sphinx** | `2feb0b43b64012ac982a9d07af85002b43b59226` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | 10717 |
| **Avg pos** | 33.0 |
| **Min pos** | 33 |
| **Max pos** | 33 |
| **Top file pos** | 4 |
| **Missing snippets** | 7 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sphinx/domains/python.py b/sphinx/domains/python.py
--- a/sphinx/domains/python.py
+++ b/sphinx/domains/python.py
@@ -77,18 +77,24 @@
                                          ('deprecated', bool)])
 
 
-def type_to_xref(text: str) -> addnodes.pending_xref:
+def type_to_xref(text: str, env: BuildEnvironment = None) -> addnodes.pending_xref:
     """Convert a type string to a cross reference node."""
     if text == 'None':
         reftype = 'obj'
     else:
         reftype = 'class'
 
+    if env:
+        kwargs = {'py:module': env.ref_context.get('py:module'),
+                  'py:class': env.ref_context.get('py:class')}
+    else:
+        kwargs = {}
+
     return pending_xref('', nodes.Text(text),
-                        refdomain='py', reftype=reftype, reftarget=text)
+                        refdomain='py', reftype=reftype, reftarget=text, **kwargs)
 
 
-def _parse_annotation(annotation: str) -> List[Node]:
+def _parse_annotation(annotation: str, env: BuildEnvironment = None) -> List[Node]:
     """Parse type annotation."""
     def unparse(node: ast.AST) -> List[Node]:
         if isinstance(node, ast.Attribute):
@@ -130,18 +136,22 @@ def unparse(node: ast.AST) -> List[Node]:
         else:
             raise SyntaxError  # unsupported syntax
 
+    if env is None:
+        warnings.warn("The env parameter for _parse_annotation becomes required now.",
+                      RemovedInSphinx50Warning, stacklevel=2)
+
     try:
         tree = ast_parse(annotation)
         result = unparse(tree)
         for i, node in enumerate(result):
             if isinstance(node, nodes.Text):
-                result[i] = type_to_xref(str(node))
+                result[i] = type_to_xref(str(node), env)
         return result
     except SyntaxError:
-        return [type_to_xref(annotation)]
+        return [type_to_xref(annotation, env)]
 
 
-def _parse_arglist(arglist: str) -> addnodes.desc_parameterlist:
+def _parse_arglist(arglist: str, env: BuildEnvironment = None) -> addnodes.desc_parameterlist:
     """Parse a list of arguments using AST parser"""
     params = addnodes.desc_parameterlist(arglist)
     sig = signature_from_str('(%s)' % arglist)
@@ -167,7 +177,7 @@ def _parse_arglist(arglist: str) -> addnodes.desc_parameterlist:
             node += addnodes.desc_sig_name('', param.name)
 
         if param.annotation is not param.empty:
-            children = _parse_annotation(param.annotation)
+            children = _parse_annotation(param.annotation, env)
             node += addnodes.desc_sig_punctuation('', ':')
             node += nodes.Text(' ')
             node += addnodes.desc_sig_name('', '', *children)  # type: ignore
@@ -415,7 +425,7 @@ def handle_signature(self, sig: str, signode: desc_signature) -> Tuple[str, str]
         signode += addnodes.desc_name(name, name)
         if arglist:
             try:
-                signode += _parse_arglist(arglist)
+                signode += _parse_arglist(arglist, self.env)
             except SyntaxError:
                 # fallback to parse arglist original parser.
                 # it supports to represent optional arguments (ex. "func(foo [, bar])")
@@ -430,7 +440,7 @@ def handle_signature(self, sig: str, signode: desc_signature) -> Tuple[str, str]
                 signode += addnodes.desc_parameterlist()
 
         if retann:
-            children = _parse_annotation(retann)
+            children = _parse_annotation(retann, self.env)
             signode += addnodes.desc_returns(retann, '', *children)
 
         anno = self.options.get('annotation')
@@ -626,7 +636,7 @@ def handle_signature(self, sig: str, signode: desc_signature) -> Tuple[str, str]
 
         typ = self.options.get('type')
         if typ:
-            annotations = _parse_annotation(typ)
+            annotations = _parse_annotation(typ, self.env)
             signode += addnodes.desc_annotation(typ, '', nodes.Text(': '), *annotations)
 
         value = self.options.get('value')
@@ -872,7 +882,7 @@ def handle_signature(self, sig: str, signode: desc_signature) -> Tuple[str, str]
 
         typ = self.options.get('type')
         if typ:
-            annotations = _parse_annotation(typ)
+            annotations = _parse_annotation(typ, self.env)
             signode += addnodes.desc_annotation(typ, '', nodes.Text(': '), *annotations)
 
         value = self.options.get('value')

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sphinx/domains/python.py | 80 | 91 | - | 4 | -
| sphinx/domains/python.py | 133 | 139 | - | 4 | -
| sphinx/domains/python.py | 170 | 170 | - | 4 | -
| sphinx/domains/python.py | 418 | 418 | - | 4 | -
| sphinx/domains/python.py | 433 | 433 | - | 4 | -
| sphinx/domains/python.py | 629 | 629 | 33 | 4 | 10717
| sphinx/domains/python.py | 875 | 875 | - | 4 | -


## Problem Statement

```
Regression: autodoc Dataclass variables reference target not found
**Describe the bug**

When I use `sphinx.ext.autodoc` and `nitpicky = True` with my code which includes a dataclass with a variable of a custom type, I get a warning.

**To Reproduce**

Open the attached project [sphinx-example.zip](https://github.com/sphinx-doc/sphinx/files/4890646/sphinx-example.zip).
Install Sphinx.
Run `sphinx-build -M html source/ build/`.

**Expected behavior**

I expect there to be no warning, or a clear message saying how I can avoid this warning.

**Your project**

[sphinx-example.zip](https://github.com/sphinx-doc/sphinx/files/4890646/sphinx-example.zip)

**Environment info**

macOS latest
Python 3.7.7
Sphinx 3.1.2 (reproducible also with 3.1.0 but not 3.0.4)

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 sphinx/ext/autodoc/__init__.py | 1678 | 1688| 128 | 128 | 18159 | 
| 2 | 2 sphinx/deprecation.py | 11 | 34| 145 | 273 | 18761 | 
| 3 | 2 sphinx/ext/autodoc/__init__.py | 1587 | 1622| 272 | 545 | 18761 | 
| 4 | 3 sphinx/ext/autosummary/__init__.py | 414 | 440| 217 | 762 | 25315 | 
| 5 | **4 sphinx/domains/python.py** | 1345 | 1383| 304 | 1066 | 36995 | 
| 6 | 5 sphinx/ext/autosummary/generate.py | 20 | 60| 283 | 1349 | 42178 | 
| 7 | 6 sphinx/ext/autodoc/directive.py | 9 | 49| 298 | 1647 | 43439 | 
| 8 | 6 sphinx/ext/autosummary/__init__.py | 55 | 108| 382 | 2029 | 43439 | 
| 9 | 6 sphinx/ext/autodoc/__init__.py | 1649 | 1676| 237 | 2266 | 43439 | 
| 10 | 6 sphinx/deprecation.py | 55 | 82| 261 | 2527 | 43439 | 
| 11 | 7 sphinx/transforms/post_transforms/__init__.py | 154 | 177| 279 | 2806 | 45439 | 
| 12 | 7 sphinx/ext/autodoc/__init__.py | 1270 | 1516| 2333 | 5139 | 45439 | 
| 13 | 7 sphinx/deprecation.py | 37 | 52| 145 | 5284 | 45439 | 
| 14 | 8 sphinx/directives/code.py | 9 | 31| 146 | 5430 | 49347 | 
| 15 | 8 sphinx/ext/autodoc/__init__.py | 2074 | 2111| 463 | 5893 | 49347 | 
| 16 | 9 setup.py | 1 | 75| 443 | 6336 | 51072 | 
| 17 | 9 sphinx/ext/autodoc/__init__.py | 930 | 951| 192 | 6528 | 51072 | 
| 18 | 10 sphinx/util/inspect.py | 11 | 47| 264 | 6792 | 56961 | 
| 19 | 11 sphinx/errors.py | 38 | 67| 213 | 7005 | 57703 | 
| 20 | 12 sphinx/parsers.py | 11 | 28| 134 | 7139 | 58571 | 
| 21 | 13 sphinx/ext/todo.py | 14 | 44| 207 | 7346 | 61288 | 
| 22 | 13 sphinx/ext/autodoc/__init__.py | 917 | 928| 126 | 7472 | 61288 | 
| 23 | 14 sphinx/ext/autodoc/importer.py | 39 | 101| 584 | 8056 | 62843 | 
| 24 | 15 sphinx/ext/autodoc/mock.py | 71 | 93| 200 | 8256 | 63950 | 
| 25 | 16 sphinx/environment/__init__.py | 11 | 82| 489 | 8745 | 69792 | 
| 26 | 17 sphinx/transforms/__init__.py | 252 | 272| 192 | 8937 | 72998 | 
| 27 | 18 sphinx/writers/latex.py | 14 | 73| 453 | 9390 | 92702 | 
| 28 | 19 sphinx/domains/std.py | 11 | 48| 323 | 9713 | 102916 | 
| 29 | 20 sphinx/io.py | 10 | 46| 275 | 9988 | 104609 | 
| 30 | 21 sphinx/directives/__init__.py | 269 | 307| 273 | 10261 | 107104 | 
| 31 | 22 doc/usage/extensions/example_numpy.py | 320 | 334| 109 | 10370 | 109212 | 
| 32 | 22 doc/usage/extensions/example_numpy.py | 336 | 356| 120 | 10490 | 109212 | 
| **-> 33 <-** | **22 sphinx/domains/python.py** | 615 | 643| 227 | 10717 | 109212 | 
| 34 | 22 sphinx/ext/autosummary/generate.py | 177 | 196| 176 | 10893 | 109212 | 
| 35 | 22 sphinx/ext/autodoc/__init__.py | 591 | 678| 759 | 11652 | 109212 | 
| 36 | 23 sphinx/util/nodes.py | 11 | 41| 227 | 11879 | 114633 | 
| 37 | 24 sphinx/ext/doctest.py | 12 | 51| 286 | 12165 | 119588 | 
| 38 | 25 sphinx/builders/devhelp.py | 13 | 39| 153 | 12318 | 119815 | 
| 39 | 25 sphinx/directives/__init__.py | 52 | 74| 206 | 12524 | 119815 | 
| 40 | 25 sphinx/ext/autodoc/__init__.py | 1099 | 1119| 242 | 12766 | 119815 | 
| 41 | 25 sphinx/ext/autodoc/__init__.py | 1519 | 1584| 521 | 13287 | 119815 | 
| 42 | 26 sphinx/util/logging.py | 11 | 56| 279 | 13566 | 123649 | 
| 43 | 27 sphinx/domains/math.py | 11 | 40| 207 | 13773 | 125159 | 
| 44 | 27 sphinx/ext/autodoc/__init__.py | 994 | 1024| 287 | 14060 | 125159 | 
| 45 | 28 sphinx/util/__init__.py | 482 | 495| 123 | 14183 | 131464 | 
| 46 | 28 sphinx/ext/autodoc/directive.py | 109 | 159| 453 | 14636 | 131464 | 
| 47 | 29 sphinx/application.py | 13 | 60| 378 | 15014 | 142232 | 
| 48 | 30 sphinx/ext/autodoc/typehints.py | 11 | 38| 200 | 15214 | 143277 | 
| 49 | 30 sphinx/ext/autosummary/__init__.py | 184 | 216| 281 | 15495 | 143277 | 
| 50 | 31 sphinx/ext/inheritance_diagram.py | 38 | 66| 214 | 15709 | 147132 | 
| 51 | 32 sphinx/events.py | 13 | 54| 352 | 16061 | 148150 | 
| 52 | 32 sphinx/ext/autodoc/importer.py | 11 | 36| 207 | 16268 | 148150 | 
| 53 | 33 sphinx/__init__.py | 14 | 65| 494 | 16762 | 148718 | 
| 54 | 33 sphinx/ext/autodoc/__init__.py | 1967 | 1997| 247 | 17009 | 148718 | 
| 55 | 33 sphinx/ext/autodoc/__init__.py | 2051 | 2071| 226 | 17235 | 148718 | 
| 56 | **33 sphinx/domains/python.py** | 985 | 1003| 143 | 17378 | 148718 | 
| 57 | 33 sphinx/ext/autodoc/__init__.py | 973 | 991| 179 | 17557 | 148718 | 
| 58 | 33 sphinx/ext/autodoc/typehints.py | 83 | 139| 460 | 18017 | 148718 | 
| 59 | 33 sphinx/ext/autodoc/__init__.py | 214 | 287| 776 | 18793 | 148718 | 
| 60 | **33 sphinx/domains/python.py** | 11 | 88| 598 | 19391 | 148718 | 
| 61 | 34 sphinx/search/__init__.py | 10 | 30| 132 | 19523 | 152760 | 
| 62 | 34 sphinx/application.py | 1013 | 1031| 233 | 19756 | 152760 | 
| 63 | **34 sphinx/domains/python.py** | 527 | 546| 147 | 19903 | 152760 | 
| 64 | 35 sphinx/util/smartypants.py | 376 | 388| 137 | 20040 | 156871 | 
| 65 | 36 sphinx/directives/other.py | 9 | 40| 238 | 20278 | 160028 | 
| 66 | 37 sphinx/ext/napoleon/docstring.py | 13 | 40| 327 | 20605 | 168875 | 
| 67 | 38 doc/conf.py | 59 | 126| 693 | 21298 | 170429 | 
| 68 | 38 sphinx/directives/code.py | 383 | 416| 284 | 21582 | 170429 | 
| 69 | 38 sphinx/ext/autosummary/__init__.py | 240 | 284| 431 | 22013 | 170429 | 
| 70 | 39 doc/usage/extensions/example_google.py | 277 | 297| 120 | 22133 | 172414 | 
| 71 | 39 sphinx/ext/autodoc/__init__.py | 2018 | 2034| 154 | 22287 | 172414 | 
| 72 | 39 sphinx/ext/autodoc/__init__.py | 1691 | 1829| 1193 | 23480 | 172414 | 
| 73 | 39 doc/usage/extensions/example_google.py | 261 | 275| 109 | 23589 | 172414 | 
| 74 | 39 sphinx/directives/other.py | 187 | 208| 141 | 23730 | 172414 | 
| 75 | 40 sphinx/ext/jsmath.py | 12 | 37| 145 | 23875 | 172621 | 
| 76 | 40 sphinx/ext/autosummary/generate.py | 89 | 119| 289 | 24164 | 172621 | 
| 77 | 41 sphinx/builders/htmlhelp.py | 12 | 43| 218 | 24382 | 172904 | 
| 78 | 41 sphinx/ext/doctest.py | 157 | 191| 209 | 24591 | 172904 | 
| 79 | 42 sphinx/util/cfamily.py | 11 | 80| 766 | 25357 | 176415 | 
| 80 | 42 sphinx/ext/autodoc/__init__.py | 893 | 915| 213 | 25570 | 176415 | 
| 81 | 43 sphinx/cmd/quickstart.py | 11 | 119| 772 | 26342 | 181975 | 
| 82 | 43 sphinx/ext/autodoc/__init__.py | 13 | 121| 780 | 27122 | 181975 | 
| 83 | 44 sphinx/roles.py | 11 | 47| 285 | 27407 | 187630 | 
| 84 | 44 sphinx/ext/autosummary/__init__.py | 769 | 795| 359 | 27766 | 187630 | 
| 85 | 45 sphinx/builders/applehelp.py | 11 | 43| 204 | 27970 | 187884 | 
| 86 | 45 sphinx/ext/autosummary/generate.py | 230 | 245| 184 | 28154 | 187884 | 
| 87 | 46 sphinx/util/docutils.py | 314 | 339| 195 | 28349 | 192004 | 
| 88 | 46 sphinx/ext/autosummary/__init__.py | 219 | 238| 131 | 28480 | 192004 | 
| 89 | 47 sphinx/registry.py | 11 | 50| 307 | 28787 | 196576 | 
| 90 | 48 sphinx/setup_command.py | 94 | 111| 145 | 28932 | 198281 | 
| 91 | 48 doc/conf.py | 1 | 58| 502 | 29434 | 198281 | 
| 92 | 48 sphinx/ext/doctest.py | 309 | 324| 130 | 29564 | 198281 | 
| 93 | 48 sphinx/ext/doctest.py | 72 | 154| 788 | 30352 | 198281 | 
| 94 | 48 sphinx/directives/code.py | 418 | 482| 658 | 31010 | 198281 | 
| 95 | 48 sphinx/errors.py | 70 | 127| 297 | 31307 | 198281 | 


### Hint

```
Thank you for reporting.

Note: Internally, the autodoc generates the following code:

\`\`\`
.. py:module:: example


.. py:class:: Report(status: example.Statuses)
   :module: example


   .. py:attribute:: Report.status
      :module: example
      :type: Statuses


.. py:class:: Statuses()
   :module: example
\`\`\`

It seems the intermediate code is good. But `py:attribute` class does not process `:type: Statuses` option well.
A Dockerfile to reproduce the error:
\`\`\`
FROM python:3.7-slim

RUN apt update; apt install -y build-essential curl git make unzip vim
RUN curl -LO https://github.com/sphinx-doc/sphinx/files/4890646/sphinx-example.zip
RUN unzip sphinx-example.zip
WORKDIR /sphinx-example
RUN pip install -U sphinx
RUN sphinx-build -NTvv source/ build/html/
\`\`\`
@tk0miya - It is great to see such open and quick progress. Thank you for your hard work on this.
```

## Patch

```diff
diff --git a/sphinx/domains/python.py b/sphinx/domains/python.py
--- a/sphinx/domains/python.py
+++ b/sphinx/domains/python.py
@@ -77,18 +77,24 @@
                                          ('deprecated', bool)])
 
 
-def type_to_xref(text: str) -> addnodes.pending_xref:
+def type_to_xref(text: str, env: BuildEnvironment = None) -> addnodes.pending_xref:
     """Convert a type string to a cross reference node."""
     if text == 'None':
         reftype = 'obj'
     else:
         reftype = 'class'
 
+    if env:
+        kwargs = {'py:module': env.ref_context.get('py:module'),
+                  'py:class': env.ref_context.get('py:class')}
+    else:
+        kwargs = {}
+
     return pending_xref('', nodes.Text(text),
-                        refdomain='py', reftype=reftype, reftarget=text)
+                        refdomain='py', reftype=reftype, reftarget=text, **kwargs)
 
 
-def _parse_annotation(annotation: str) -> List[Node]:
+def _parse_annotation(annotation: str, env: BuildEnvironment = None) -> List[Node]:
     """Parse type annotation."""
     def unparse(node: ast.AST) -> List[Node]:
         if isinstance(node, ast.Attribute):
@@ -130,18 +136,22 @@ def unparse(node: ast.AST) -> List[Node]:
         else:
             raise SyntaxError  # unsupported syntax
 
+    if env is None:
+        warnings.warn("The env parameter for _parse_annotation becomes required now.",
+                      RemovedInSphinx50Warning, stacklevel=2)
+
     try:
         tree = ast_parse(annotation)
         result = unparse(tree)
         for i, node in enumerate(result):
             if isinstance(node, nodes.Text):
-                result[i] = type_to_xref(str(node))
+                result[i] = type_to_xref(str(node), env)
         return result
     except SyntaxError:
-        return [type_to_xref(annotation)]
+        return [type_to_xref(annotation, env)]
 
 
-def _parse_arglist(arglist: str) -> addnodes.desc_parameterlist:
+def _parse_arglist(arglist: str, env: BuildEnvironment = None) -> addnodes.desc_parameterlist:
     """Parse a list of arguments using AST parser"""
     params = addnodes.desc_parameterlist(arglist)
     sig = signature_from_str('(%s)' % arglist)
@@ -167,7 +177,7 @@ def _parse_arglist(arglist: str) -> addnodes.desc_parameterlist:
             node += addnodes.desc_sig_name('', param.name)
 
         if param.annotation is not param.empty:
-            children = _parse_annotation(param.annotation)
+            children = _parse_annotation(param.annotation, env)
             node += addnodes.desc_sig_punctuation('', ':')
             node += nodes.Text(' ')
             node += addnodes.desc_sig_name('', '', *children)  # type: ignore
@@ -415,7 +425,7 @@ def handle_signature(self, sig: str, signode: desc_signature) -> Tuple[str, str]
         signode += addnodes.desc_name(name, name)
         if arglist:
             try:
-                signode += _parse_arglist(arglist)
+                signode += _parse_arglist(arglist, self.env)
             except SyntaxError:
                 # fallback to parse arglist original parser.
                 # it supports to represent optional arguments (ex. "func(foo [, bar])")
@@ -430,7 +440,7 @@ def handle_signature(self, sig: str, signode: desc_signature) -> Tuple[str, str]
                 signode += addnodes.desc_parameterlist()
 
         if retann:
-            children = _parse_annotation(retann)
+            children = _parse_annotation(retann, self.env)
             signode += addnodes.desc_returns(retann, '', *children)
 
         anno = self.options.get('annotation')
@@ -626,7 +636,7 @@ def handle_signature(self, sig: str, signode: desc_signature) -> Tuple[str, str]
 
         typ = self.options.get('type')
         if typ:
-            annotations = _parse_annotation(typ)
+            annotations = _parse_annotation(typ, self.env)
             signode += addnodes.desc_annotation(typ, '', nodes.Text(': '), *annotations)
 
         value = self.options.get('value')
@@ -872,7 +882,7 @@ def handle_signature(self, sig: str, signode: desc_signature) -> Tuple[str, str]
 
         typ = self.options.get('type')
         if typ:
-            annotations = _parse_annotation(typ)
+            annotations = _parse_annotation(typ, self.env)
             signode += addnodes.desc_annotation(typ, '', nodes.Text(': '), *annotations)
 
         value = self.options.get('value')

```

## Test Patch

```diff
diff --git a/tests/test_domain_py.py b/tests/test_domain_py.py
--- a/tests/test_domain_py.py
+++ b/tests/test_domain_py.py
@@ -236,18 +236,18 @@ def test_get_full_qualified_name():
     assert domain.get_full_qualified_name(node) == 'module1.Class.func'
 
 
-def test_parse_annotation():
-    doctree = _parse_annotation("int")
+def test_parse_annotation(app):
+    doctree = _parse_annotation("int", app.env)
     assert_node(doctree, ([pending_xref, "int"],))
     assert_node(doctree[0], pending_xref, refdomain="py", reftype="class", reftarget="int")
 
-    doctree = _parse_annotation("List[int]")
+    doctree = _parse_annotation("List[int]", app.env)
     assert_node(doctree, ([pending_xref, "List"],
                           [desc_sig_punctuation, "["],
                           [pending_xref, "int"],
                           [desc_sig_punctuation, "]"]))
 
-    doctree = _parse_annotation("Tuple[int, int]")
+    doctree = _parse_annotation("Tuple[int, int]", app.env)
     assert_node(doctree, ([pending_xref, "Tuple"],
                           [desc_sig_punctuation, "["],
                           [pending_xref, "int"],
@@ -255,14 +255,14 @@ def test_parse_annotation():
                           [pending_xref, "int"],
                           [desc_sig_punctuation, "]"]))
 
-    doctree = _parse_annotation("Tuple[()]")
+    doctree = _parse_annotation("Tuple[()]", app.env)
     assert_node(doctree, ([pending_xref, "Tuple"],
                           [desc_sig_punctuation, "["],
                           [desc_sig_punctuation, "("],
                           [desc_sig_punctuation, ")"],
                           [desc_sig_punctuation, "]"]))
 
-    doctree = _parse_annotation("Callable[[int, int], int]")
+    doctree = _parse_annotation("Callable[[int, int], int]", app.env)
     assert_node(doctree, ([pending_xref, "Callable"],
                           [desc_sig_punctuation, "["],
                           [desc_sig_punctuation, "["],
@@ -275,12 +275,11 @@ def test_parse_annotation():
                           [desc_sig_punctuation, "]"]))
 
     # None type makes an object-reference (not a class reference)
-    doctree = _parse_annotation("None")
+    doctree = _parse_annotation("None", app.env)
     assert_node(doctree, ([pending_xref, "None"],))
     assert_node(doctree[0], pending_xref, refdomain="py", reftype="obj", reftarget="None")
 
 
-
 def test_pyfunction_signature(app):
     text = ".. py:function:: hello(name: str) -> str"
     doctree = restructuredtext.parse(app, text)
@@ -458,14 +457,22 @@ def test_pyobject_prefix(app):
 
 
 def test_pydata(app):
-    text = ".. py:data:: var\n"
+    text = (".. py:module:: example\n"
+            ".. py:data:: var\n"
+            "   :type: int\n")
     domain = app.env.get_domain('py')
     doctree = restructuredtext.parse(app, text)
-    assert_node(doctree, (addnodes.index,
-                          [desc, ([desc_signature, desc_name, "var"],
+    assert_node(doctree, (nodes.target,
+                          addnodes.index,
+                          addnodes.index,
+                          [desc, ([desc_signature, ([desc_addname, "example."],
+                                                    [desc_name, "var"],
+                                                    [desc_annotation, (": ",
+                                                                       [pending_xref, "int"])])],
                                   [desc_content, ()])]))
-    assert 'var' in domain.objects
-    assert domain.objects['var'] == ('index', 'var', 'data')
+    assert_node(doctree[3][0][2][1], pending_xref, **{"py:module": "example"})
+    assert 'example.var' in domain.objects
+    assert domain.objects['example.var'] == ('index', 'example.var', 'data')
 
 
 def test_pyfunction(app):
@@ -698,6 +705,8 @@ def test_pyattribute(app):
                                                                         [desc_sig_punctuation, "]"])],
                                                      [desc_annotation, " = ''"])],
                                    [desc_content, ()]))
+    assert_node(doctree[1][1][1][0][1][1], pending_xref, **{"py:class": "Class"})
+    assert_node(doctree[1][1][1][0][1][3], pending_xref, **{"py:class": "Class"})
     assert 'Class.attr' in domain.objects
     assert domain.objects['Class.attr'] == ('index', 'Class.attr', 'attribute')
 

```


## Code snippets

### 1 - sphinx/ext/autodoc/__init__.py:

Start line: 1678, End line: 1688

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
### 3 - sphinx/ext/autodoc/__init__.py:

Start line: 1587, End line: 1622

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

    def import_object(self) -> bool:
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
### 4 - sphinx/ext/autosummary/__init__.py:

Start line: 414, End line: 440

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
### 5 - sphinx/domains/python.py:

Start line: 1345, End line: 1383

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
### 6 - sphinx/ext/autosummary/generate.py:

Start line: 20, End line: 60

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
from typing import Any, Callable, Dict, List, NamedTuple, Set, Tuple, Union

from jinja2 import TemplateNotFound
from jinja2.sandbox import SandboxedEnvironment

import sphinx.locale
from sphinx import __display_version__
from sphinx import package_dir
from sphinx.application import Sphinx
from sphinx.builders import Builder
from sphinx.config import Config
from sphinx.deprecation import RemovedInSphinx40Warning, RemovedInSphinx50Warning
from sphinx.ext.autodoc import Documenter
from sphinx.ext.autosummary import import_by_name, get_documenter
from sphinx.locale import __
from sphinx.pycode import ModuleAnalyzer, PycodeError
from sphinx.registry import SphinxComponentRegistry
from sphinx.util import logging
from sphinx.util import rst
from sphinx.util import split_full_qualified_name
from sphinx.util.inspect import safe_getattr
from sphinx.util.osutil import ensuredir
from sphinx.util.template import SphinxTemplateLoader

if False:
    # For type annotation
    from typing import Type  # for python3.5.1


logger = logging.getLogger(__name__)
```
### 7 - sphinx/ext/autodoc/directive.py:

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
### 8 - sphinx/ext/autosummary/__init__.py:

Start line: 55, End line: 108

```python
import inspect
import os
import posixpath
import re
import sys
import warnings
from os import path
from types import ModuleType
from typing import Any, Dict, List, Tuple
from typing import cast

from docutils import nodes
from docutils.nodes import Element, Node, system_message
from docutils.parsers.rst import directives
from docutils.parsers.rst.states import Inliner, RSTStateMachine, Struct, state_classes
from docutils.statemachine import StringList

import sphinx
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.deprecation import RemovedInSphinx40Warning, RemovedInSphinx50Warning
from sphinx.environment import BuildEnvironment
from sphinx.environment.adapters.toctree import TocTree
from sphinx.ext.autodoc import Documenter
from sphinx.ext.autodoc.directive import DocumenterBridge, Options
from sphinx.ext.autodoc.importer import import_module
from sphinx.ext.autodoc.mock import mock
from sphinx.locale import __
from sphinx.pycode import ModuleAnalyzer, PycodeError
from sphinx.util import rst, logging
from sphinx.util.docutils import (
    NullReporter, SphinxDirective, SphinxRole, new_document, switch_source_input
)
from sphinx.util.matching import Matcher
from sphinx.writers.html import HTMLTranslator

if False:
    # For type annotation
    from typing import Type  # for python3.5.1


logger = logging.getLogger(__name__)


periods_re = re.compile(r'\.(?:\s+)')
literal_re = re.compile(r'::\s*$')

WELL_KNOWN_ABBREVIATIONS = (' i.e.',)


# -- autosummary_toc node ------------------------------------------------------

class autosummary_toc(nodes.comment):
    pass
```
### 9 - sphinx/ext/autodoc/__init__.py:

Start line: 1649, End line: 1676

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
        return isinstance(member, TypeVar) and isattr  # type: ignore

    def add_directive_header(self, sig: str) -> None:
        self.options.annotation = SUPPRESS  # type: ignore
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
### 10 - sphinx/deprecation.py:

Start line: 55, End line: 82

```python
class DeprecatedDict(dict):
    """A deprecated dict which warns on each access."""

    def __init__(self, data: Dict, message: str, warning: "Type[Warning]") -> None:
        self.message = message
        self.warning = warning
        super().__init__(data)

    def __setitem__(self, key: str, value: Any) -> None:
        warnings.warn(self.message, self.warning, stacklevel=2)
        super().__setitem__(key, value)

    def setdefault(self, key: str, default: Any = None) -> Any:
        warnings.warn(self.message, self.warning, stacklevel=2)
        return super().setdefault(key, default)

    def __getitem__(self, key: str) -> None:
        warnings.warn(self.message, self.warning, stacklevel=2)
        return super().__getitem__(key)

    def get(self, key: str, default: Any = None) -> Any:
        warnings.warn(self.message, self.warning, stacklevel=2)
        return super().get(key, default)

    def update(self, other: Dict) -> None:  # type: ignore
        warnings.warn(self.message, self.warning, stacklevel=2)
        super().update(other)
```
### 33 - sphinx/domains/python.py:

Start line: 615, End line: 643

```python
class PyVariable(PyObject):
    """Description of a variable."""

    option_spec = PyObject.option_spec.copy()
    option_spec.update({
        'type': directives.unchanged,
        'value': directives.unchanged,
    })

    def handle_signature(self, sig: str, signode: desc_signature) -> Tuple[str, str]:
        fullname, prefix = super().handle_signature(sig, signode)

        typ = self.options.get('type')
        if typ:
            annotations = _parse_annotation(typ)
            signode += addnodes.desc_annotation(typ, '', nodes.Text(': '), *annotations)

        value = self.options.get('value')
        if value:
            signode += addnodes.desc_annotation(value, ' = ' + value)

        return fullname, prefix

    def get_index_text(self, modname: str, name_cls: Tuple[str, str]) -> str:
        name, cls = name_cls
        if modname:
            return _('%s (in module %s)') % (name, modname)
        else:
            return _('%s (built-in variable)') % name
```
### 56 - sphinx/domains/python.py:

Start line: 985, End line: 1003

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
### 60 - sphinx/domains/python.py:

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
### 63 - sphinx/domains/python.py:

Start line: 527, End line: 546

```python
class PyModulelevel(PyObject):
    """
    Description of an object on module level (functions, data).
    """

    def run(self) -> List[Node]:
        for cls in self.__class__.__mro__:
            if cls.__name__ != 'DirectiveAdapter':
                warnings.warn('PyModulelevel is deprecated. '
                              'Please check the implementation of %s' % cls,
                              RemovedInSphinx40Warning, stacklevel=2)
                break
        else:
            warnings.warn('PyModulelevel is deprecated',
                          RemovedInSphinx40Warning, stacklevel=2)

        return super().run()

    def needs_arglist(self) -> bool:
        return self.objtype == 'function'
```
