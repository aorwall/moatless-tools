# sphinx-doc__sphinx-8075

| **sphinx-doc/sphinx** | `487b8436c6e8dc596db4b8d4d06e9145105a2ac2` |
| ---- | ---- |
| **No of patches** | 3 |
| **All found context length** | 36134 |
| **Any found context length** | 36134 |
| **Avg pos** | 25.0 |
| **Min pos** | 75 |
| **Max pos** | 75 |
| **Top file pos** | 39 |
| **Missing snippets** | 4 |
| **Missing patch files** | 1 |


## Expected patch

```diff
diff --git a/sphinx/domains/std.py b/sphinx/domains/std.py
--- a/sphinx/domains/std.py
+++ b/sphinx/domains/std.py
@@ -610,8 +610,6 @@ class StandardDomain(Domain):
 
     dangling_warnings = {
         'term': 'term not in glossary: %(target)s',
-        'ref':  'undefined label: %(target)s (if the link has no caption '
-                'the label must precede a section header)',
         'numref':  'undefined label: %(target)s',
         'keyword': 'unknown keyword: %(target)s',
         'doc': 'unknown document: %(target)s',
@@ -1107,8 +1105,23 @@ def note_labels(self, env: "BuildEnvironment", docname: str, document: nodes.doc
                       RemovedInSphinx40Warning, stacklevel=2)
 
 
+def warn_missing_reference(app: "Sphinx", domain: Domain, node: pending_xref) -> bool:
+    if domain.name != 'std' or node['reftype'] != 'ref':
+        return None
+    else:
+        target = node['reftarget']
+        if target not in domain.anonlabels:  # type: ignore
+            msg = __('undefined label: %s')
+        else:
+            msg = __('Failed to create a cross reference. A title or caption not found: %s')
+
+        logger.warning(msg % target, location=node, type='ref', subtype=node['reftype'])
+        return True
+
+
 def setup(app: "Sphinx") -> Dict[str, Any]:
     app.add_domain(StandardDomain)
+    app.connect('warn-missing-reference', warn_missing_reference)
 
     return {
         'version': 'builtin',
diff --git a/sphinx/events.py b/sphinx/events.py
--- a/sphinx/events.py
+++ b/sphinx/events.py
@@ -46,6 +46,7 @@
     'doctree-read': 'the doctree before being pickled',
     'env-merge-info': 'env, read docnames, other env instance',
     'missing-reference': 'env, node, contnode',
+    'warn-missing-reference': 'domain, node',
     'doctree-resolved': 'doctree, docname',
     'env-updated': 'env',
     'html-collect-pages': 'builder',
diff --git a/sphinx/transforms/post_transforms/__init__.py b/sphinx/transforms/post_transforms/__init__.py
--- a/sphinx/transforms/post_transforms/__init__.py
+++ b/sphinx/transforms/post_transforms/__init__.py
@@ -166,7 +166,10 @@ def warn_missing_reference(self, refdoc: str, typ: str, target: str,
                     warn = False
         if not warn:
             return
-        if domain and typ in domain.dangling_warnings:
+
+        if self.app.emit_firstresult('warn-missing-reference', domain, node):
+            return
+        elif domain and typ in domain.dangling_warnings:
             msg = domain.dangling_warnings[typ]
         elif node.get('refdomain', 'std') not in ('', 'std'):
             msg = (__('%s:%s reference target not found: %%(target)s') %

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sphinx/domains/std.py | 613 | 614 | - | 41 | -
| sphinx/domains/std.py | 1110 | 1110 | - | 41 | -
| sphinx/events.py | 49 | 49 | - | - | -
| sphinx/transforms/post_transforms/__init__.py | 169 | 169 | 75 | 39 | 36134


## Problem Statement

```
References to figures without captions: errors in both HTML and LaTeX

**Describe the bug**
Using figures without captions causes errors in both HTML (though these are properly reported when source is processed) and in LaTeX (they are not reported until LaTeX says there were undefined references).

This was the test document, compiled with sphinx 2.2.2 from pypi; `numfig=True` was added to conf.py, the project was otherwise generated with sphinx-build with no other changes. It is attached here: [sphinx-captions.zip](https://github.com/sphinx-doc/sphinx/files/3947135/sphinx-captions.zip)

\`\`\`
Welcome to foo's documentation!
===============================

References:

* figure without caption

   * plain reference :ref:`fig-sample-nocaption` (error: HTML, LaTeX)
   * named reference :ref:`figure without caption <fig-sample-nocaption>` (error: LaTeX)
   * numbered reference :numref:`fig-sample-nocaption` (error: LaTeX)

* figure with caption

   * plain reference :ref:`fig-sample-caption`
   * named reference :ref:`figure without caption <fig-sample-caption>`
   * numbered reference :numref:`fig-sample-caption`

.. _fig-sample-nocaption:
.. figure:: sample.png


.. _fig-sample-caption:
.. figure:: sample.png
   
   This is some caption.
\`\`\`

and these are the results:

1. misleading warning: **index.rst:8: WARNING: undefined label: fig-sample-nocaption (if the link has no caption the label must precede a section header)**
2. this is HTML output (the error highlighted corresponds to the warning mentioned above):
![html output](https://user-images.githubusercontent.com/1029876/70568432-2b150c00-1b98-11ea-98ac-67e7fbc23927.png)
3. this is LaTeX (pdflatex) output:
\`\`\`
LaTeX Warning: Hyper reference `index:fig-sample-nocaption' on page 1 undefined
 on input line 99.
LaTeX Warning: Hyper reference `index:fig-sample-nocaption' on page 1 undefined
 on input line 102.
\`\`\`
![latex output](https://user-images.githubusercontent.com/1029876/70568602-7fb88700-1b98-11ea-85bd-b7b6fec93e41.png)

**Expected behavior**
I expect
1. sphinx to produce valid LaTeX input without undefined references;
2. uncaptioned figures to be referencable in LaTeX (this could be an optional setting perhaps causing uncaptioned figured to produce only "Figure 4.1." caption);
3. warning about figure not being captioned to be more meaningful -- I understand that non-numbered figure cannot be referenced via :ref:`label` (as the label will not resolve to any text) but the warning is not pointing to how to fix the issue.

**Environment info**
- OS: Ubuntu 18.04 LTS
- Python version: 3.6.8
- Sphinx version: 2.2.2
- Sphinx extensions: none
- Extra tools: pdflatex TeXLive


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 sphinx/writers/latex.py | 1330 | 1398| 667 | 667 | 19878 | 
| 2 | 1 sphinx/writers/latex.py | 2089 | 2124| 413 | 1080 | 19878 | 
| 3 | 1 sphinx/writers/latex.py | 14 | 73| 453 | 1533 | 19878 | 
| 4 | 2 doc/conf.py | 60 | 125| 668 | 2201 | 21416 | 
| 5 | 3 utils/checks.py | 33 | 109| 545 | 2746 | 22322 | 
| 6 | 4 sphinx/io.py | 10 | 47| 282 | 3028 | 24072 | 
| 7 | 5 sphinx/writers/texinfo.py | 1090 | 1189| 839 | 3867 | 36407 | 
| 8 | 6 sphinx/util/nodes.py | 11 | 41| 227 | 4094 | 41833 | 
| 9 | 7 sphinx/writers/html.py | 460 | 482| 238 | 4332 | 49222 | 
| 10 | 8 sphinx/builders/html/__init__.py | 11 | 67| 462 | 4794 | 60592 | 
| 11 | 8 sphinx/writers/latex.py | 1588 | 1611| 290 | 5084 | 60592 | 
| 12 | 8 sphinx/writers/latex.py | 289 | 443| 1717 | 6801 | 60592 | 
| 13 | 9 sphinx/builders/_epub_base.py | 11 | 102| 662 | 7463 | 67332 | 
| 14 | 9 sphinx/writers/latex.py | 1294 | 1328| 421 | 7884 | 67332 | 
| 15 | 9 sphinx/writers/latex.py | 1685 | 1739| 513 | 8397 | 67332 | 
| 16 | 9 sphinx/writers/latex.py | 796 | 858| 567 | 8964 | 67332 | 
| 17 | 10 sphinx/writers/html5.py | 412 | 434| 239 | 9203 | 74258 | 
| 18 | 11 sphinx/builders/latex/constants.py | 70 | 119| 523 | 9726 | 76338 | 
| 19 | 11 sphinx/writers/latex.py | 507 | 545| 384 | 10110 | 76338 | 
| 20 | 12 sphinx/directives/__init__.py | 269 | 329| 527 | 10637 | 79087 | 
| 21 | 12 sphinx/writers/latex.py | 692 | 794| 825 | 11462 | 79087 | 
| 22 | 12 sphinx/writers/latex.py | 1522 | 1586| 621 | 12083 | 79087 | 
| 23 | 13 sphinx/util/__init__.py | 11 | 75| 509 | 12592 | 85392 | 
| 24 | 13 sphinx/writers/latex.py | 1613 | 1672| 496 | 13088 | 85392 | 
| 25 | 14 sphinx/builders/latex/__init__.py | 11 | 42| 336 | 13424 | 91390 | 
| 26 | 14 sphinx/writers/latex.py | 1082 | 1152| 538 | 13962 | 91390 | 
| 27 | 14 sphinx/writers/texinfo.py | 865 | 969| 788 | 14750 | 91390 | 
| 28 | 15 sphinx/ext/imgmath.py | 11 | 110| 678 | 15428 | 94791 | 
| 29 | 15 sphinx/writers/latex.py | 2026 | 2046| 200 | 15628 | 94791 | 
| 30 | 16 sphinx/util/pycompat.py | 11 | 54| 357 | 15985 | 95659 | 
| 31 | 17 sphinx/ext/graphviz.py | 12 | 43| 233 | 16218 | 99265 | 
| 32 | 17 doc/conf.py | 1 | 59| 511 | 16729 | 99265 | 
| 33 | 18 sphinx/builders/latex/transforms.py | 363 | 441| 666 | 17395 | 103549 | 
| 34 | 19 sphinx/builders/singlehtml.py | 105 | 124| 300 | 17695 | 105431 | 
| 35 | 20 sphinx/ext/autosummary/generate.py | 20 | 60| 289 | 17984 | 110714 | 
| 36 | 21 sphinx/directives/patches.py | 71 | 107| 247 | 18231 | 112327 | 
| 37 | 22 sphinx/builders/epub3.py | 12 | 53| 276 | 18507 | 115020 | 
| 38 | 22 sphinx/writers/latex.py | 2070 | 2087| 170 | 18677 | 115020 | 
| 39 | 22 sphinx/writers/latex.py | 579 | 642| 521 | 19198 | 115020 | 
| 40 | 23 doc/development/tutorials/examples/todo.py | 30 | 53| 175 | 19373 | 115928 | 
| 41 | 23 sphinx/writers/texinfo.py | 755 | 863| 813 | 20186 | 115928 | 
| 42 | 24 sphinx/writers/manpage.py | 313 | 411| 757 | 20943 | 119439 | 
| 43 | 24 sphinx/writers/html.py | 697 | 796| 803 | 21746 | 119439 | 
| 44 | 24 sphinx/builders/latex/__init__.py | 307 | 323| 170 | 21916 | 119439 | 
| 45 | 25 sphinx/domains/python.py | 11 | 78| 522 | 22438 | 131340 | 
| 46 | 26 sphinx/util/cfamily.py | 11 | 80| 766 | 23204 | 134845 | 
| 47 | 27 sphinx/environment/__init__.py | 11 | 82| 489 | 23693 | 140708 | 
| 48 | 28 sphinx/transforms/__init__.py | 252 | 272| 192 | 23885 | 143957 | 
| 49 | 28 sphinx/writers/texinfo.py | 1218 | 1283| 457 | 24342 | 143957 | 
| 50 | 28 sphinx/writers/latex.py | 1400 | 1454| 463 | 24805 | 143957 | 
| 51 | 29 sphinx/util/texescape.py | 11 | 64| 623 | 25428 | 145726 | 
| 52 | 29 sphinx/writers/latex.py | 1456 | 1520| 863 | 26291 | 145726 | 
| 53 | 29 sphinx/builders/latex/transforms.py | 184 | 360| 800 | 27091 | 145726 | 
| 54 | 30 sphinx/directives/code.py | 9 | 31| 146 | 27237 | 149636 | 
| 55 | 30 sphinx/writers/latex.py | 883 | 922| 308 | 27545 | 149636 | 
| 56 | 31 sphinx/ext/linkcode.py | 11 | 80| 470 | 28015 | 150162 | 
| 57 | 32 sphinx/addnodes.py | 281 | 380| 580 | 28595 | 152908 | 
| 58 | 33 sphinx/highlighting.py | 11 | 54| 379 | 28974 | 154216 | 
| 59 | 34 sphinx/util/inspect.py | 11 | 48| 276 | 29250 | 160655 | 
| 60 | 34 sphinx/writers/texinfo.py | 1304 | 1387| 678 | 29928 | 160655 | 
| 61 | 34 sphinx/writers/html.py | 354 | 409| 472 | 30400 | 160655 | 
| 62 | 35 sphinx/cmd/quickstart.py | 11 | 119| 772 | 31172 | 166251 | 
| 63 | 35 sphinx/writers/html.py | 309 | 333| 271 | 31443 | 166251 | 
| 64 | 35 sphinx/writers/texinfo.py | 1191 | 1216| 275 | 31718 | 166251 | 
| 65 | 35 sphinx/writers/latex.py | 1051 | 1080| 328 | 32046 | 166251 | 
| 66 | 35 sphinx/writers/html5.py | 281 | 305| 272 | 32318 | 166251 | 
| 67 | 36 sphinx/errors.py | 70 | 127| 297 | 32615 | 166993 | 
| 68 | 36 sphinx/builders/latex/__init__.py | 325 | 366| 467 | 33082 | 166993 | 
| 69 | 37 setup.py | 172 | 248| 638 | 33720 | 168718 | 
| 70 | 37 sphinx/writers/html.py | 260 | 282| 214 | 33934 | 168718 | 
| 71 | 37 sphinx/transforms/__init__.py | 11 | 45| 234 | 34168 | 168718 | 
| 72 | 37 sphinx/writers/html.py | 171 | 231| 562 | 34730 | 168718 | 
| 73 | 38 sphinx/util/logging.py | 11 | 56| 279 | 35009 | 172552 | 
| 74 | 38 sphinx/writers/texinfo.py | 971 | 1088| 846 | 35855 | 172552 | 
| **-> 75 <-** | **39 sphinx/transforms/post_transforms/__init__.py** | 154 | 177| 279 | 36134 | 174552 | 
| 76 | 40 sphinx/transforms/references.py | 11 | 55| 266 | 36400 | 174870 | 
| 77 | **41 sphinx/domains/std.py** | 11 | 48| 323 | 36723 | 185211 | 
| 78 | 41 sphinx/util/texescape.py | 65 | 126| 712 | 37435 | 185211 | 
| 79 | 42 sphinx/__init__.py | 14 | 65| 495 | 37930 | 185780 | 
| 80 | 43 sphinx/environment/collectors/toctree.py | 214 | 259| 456 | 38386 | 188649 | 
| 81 | 43 sphinx/writers/latex.py | 1741 | 1794| 561 | 38947 | 188649 | 
| 82 | 43 sphinx/ext/graphviz.py | 126 | 170| 405 | 39352 | 188649 | 
| 83 | 44 sphinx/parsers.py | 11 | 28| 134 | 39486 | 189517 | 
| 84 | 45 sphinx/ext/inheritance_diagram.py | 38 | 70| 238 | 39724 | 193384 | 
| 85 | 45 sphinx/writers/html5.py | 634 | 720| 686 | 40410 | 193384 | 
| 86 | 45 sphinx/writers/latex.py | 2048 | 2068| 221 | 40631 | 193384 | 
| 87 | 45 sphinx/writers/texinfo.py | 1479 | 1554| 549 | 41180 | 193384 | 
| 88 | 45 sphinx/writers/latex.py | 1974 | 1981| 127 | 41307 | 193384 | 
| 89 | 45 sphinx/ext/imgmath.py | 382 | 402| 289 | 41596 | 193384 | 
| 90 | 45 sphinx/writers/latex.py | 1209 | 1280| 729 | 42325 | 193384 | 
| 91 | 46 sphinx/ext/doctest.py | 12 | 51| 286 | 42611 | 198472 | 
| 92 | 46 sphinx/builders/html/__init__.py | 1195 | 1267| 886 | 43497 | 198472 | 
| 93 | 46 sphinx/environment/collectors/toctree.py | 283 | 306| 206 | 43703 | 198472 | 
| 94 | 46 sphinx/ext/graphviz.py | 314 | 348| 289 | 43992 | 198472 | 


## Missing Patch Files

 * 1: sphinx/domains/std.py
 * 2: sphinx/events.py
 * 3: sphinx/transforms/post_transforms/__init__.py

## Patch

```diff
diff --git a/sphinx/domains/std.py b/sphinx/domains/std.py
--- a/sphinx/domains/std.py
+++ b/sphinx/domains/std.py
@@ -610,8 +610,6 @@ class StandardDomain(Domain):
 
     dangling_warnings = {
         'term': 'term not in glossary: %(target)s',
-        'ref':  'undefined label: %(target)s (if the link has no caption '
-                'the label must precede a section header)',
         'numref':  'undefined label: %(target)s',
         'keyword': 'unknown keyword: %(target)s',
         'doc': 'unknown document: %(target)s',
@@ -1107,8 +1105,23 @@ def note_labels(self, env: "BuildEnvironment", docname: str, document: nodes.doc
                       RemovedInSphinx40Warning, stacklevel=2)
 
 
+def warn_missing_reference(app: "Sphinx", domain: Domain, node: pending_xref) -> bool:
+    if domain.name != 'std' or node['reftype'] != 'ref':
+        return None
+    else:
+        target = node['reftarget']
+        if target not in domain.anonlabels:  # type: ignore
+            msg = __('undefined label: %s')
+        else:
+            msg = __('Failed to create a cross reference. A title or caption not found: %s')
+
+        logger.warning(msg % target, location=node, type='ref', subtype=node['reftype'])
+        return True
+
+
 def setup(app: "Sphinx") -> Dict[str, Any]:
     app.add_domain(StandardDomain)
+    app.connect('warn-missing-reference', warn_missing_reference)
 
     return {
         'version': 'builtin',
diff --git a/sphinx/events.py b/sphinx/events.py
--- a/sphinx/events.py
+++ b/sphinx/events.py
@@ -46,6 +46,7 @@
     'doctree-read': 'the doctree before being pickled',
     'env-merge-info': 'env, read docnames, other env instance',
     'missing-reference': 'env, node, contnode',
+    'warn-missing-reference': 'domain, node',
     'doctree-resolved': 'doctree, docname',
     'env-updated': 'env',
     'html-collect-pages': 'builder',
diff --git a/sphinx/transforms/post_transforms/__init__.py b/sphinx/transforms/post_transforms/__init__.py
--- a/sphinx/transforms/post_transforms/__init__.py
+++ b/sphinx/transforms/post_transforms/__init__.py
@@ -166,7 +166,10 @@ def warn_missing_reference(self, refdoc: str, typ: str, target: str,
                     warn = False
         if not warn:
             return
-        if domain and typ in domain.dangling_warnings:
+
+        if self.app.emit_firstresult('warn-missing-reference', domain, node):
+            return
+        elif domain and typ in domain.dangling_warnings:
             msg = domain.dangling_warnings[typ]
         elif node.get('refdomain', 'std') not in ('', 'std'):
             msg = (__('%s:%s reference target not found: %%(target)s') %

```

## Test Patch

```diff
diff --git a/tests/roots/test-domain-py-xref-warning/conf.py b/tests/roots/test-domain-py-xref-warning/conf.py
new file mode 100644
diff --git a/tests/roots/test-domain-py-xref-warning/index.rst b/tests/roots/test-domain-py-xref-warning/index.rst
new file mode 100644
--- /dev/null
+++ b/tests/roots/test-domain-py-xref-warning/index.rst
@@ -0,0 +1,7 @@
+test-domain-py-xref-warning
+===========================
+
+.. _existing-label:
+
+:ref:`no-label`
+:ref:`existing-label`
diff --git a/tests/test_domain_py.py b/tests/test_domain_py.py
--- a/tests/test_domain_py.py
+++ b/tests/test_domain_py.py
@@ -859,3 +859,11 @@ def test_noindexentry(app):
     assert_node(doctree, (addnodes.index, desc, addnodes.index, desc))
     assert_node(doctree[0], addnodes.index, entries=[('single', 'f (built-in class)', 'f', '', None)])
     assert_node(doctree[2], addnodes.index, entries=[])
+
+
+@pytest.mark.sphinx('dummy', testroot='domain-py-xref-warning')
+def test_warn_missing_reference(app, status, warning):
+    app.build()
+    assert 'index.rst:6: WARNING: undefined label: no-label' in warning.getvalue()
+    assert ('index.rst:6: WARNING: Failed to create a cross reference. A title or caption not found: existing-label'
+            in warning.getvalue())

```


## Code snippets

### 1 - sphinx/writers/latex.py:

Start line: 1330, End line: 1398

```python
class LaTeXTranslator(SphinxTranslator):

    def depart_figure(self, node: Element) -> None:
        self.body.append(self.context.pop())

    def visit_caption(self, node: Element) -> None:
        self.in_caption += 1
        if isinstance(node.parent, captioned_literal_block):
            self.body.append('\\sphinxSetupCaptionForVerbatim{')
        elif self.in_minipage and isinstance(node.parent, nodes.figure):
            self.body.append('\\captionof{figure}{')
        elif self.table and node.parent.tagname == 'figure':
            self.body.append('\\sphinxfigcaption{')
        else:
            self.body.append('\\caption{')

    def depart_caption(self, node: Element) -> None:
        self.body.append('}')
        if isinstance(node.parent, nodes.figure):
            labels = self.hypertarget_to(node.parent)
            self.body.append(labels)
        self.in_caption -= 1

    def visit_legend(self, node: Element) -> None:
        self.body.append('\n\\begin{sphinxlegend}')

    def depart_legend(self, node: Element) -> None:
        self.body.append('\\end{sphinxlegend}\n')

    def visit_admonition(self, node: Element) -> None:
        self.body.append('\n\\begin{sphinxadmonition}{note}')
        self.no_latex_floats += 1

    def depart_admonition(self, node: Element) -> None:
        self.body.append('\\end{sphinxadmonition}\n')
        self.no_latex_floats -= 1

    def _visit_named_admonition(self, node: Element) -> None:
        label = admonitionlabels[node.tagname]
        self.body.append('\n\\begin{sphinxadmonition}{%s}{%s:}' %
                         (node.tagname, label))
        self.no_latex_floats += 1

    def _depart_named_admonition(self, node: Element) -> None:
        self.body.append('\\end{sphinxadmonition}\n')
        self.no_latex_floats -= 1

    visit_attention = _visit_named_admonition
    depart_attention = _depart_named_admonition
    visit_caution = _visit_named_admonition
    depart_caution = _depart_named_admonition
    visit_danger = _visit_named_admonition
    depart_danger = _depart_named_admonition
    visit_error = _visit_named_admonition
    depart_error = _depart_named_admonition
    visit_hint = _visit_named_admonition
    depart_hint = _depart_named_admonition
    visit_important = _visit_named_admonition
    depart_important = _depart_named_admonition
    visit_note = _visit_named_admonition
    depart_note = _depart_named_admonition
    visit_tip = _visit_named_admonition
    depart_tip = _depart_named_admonition
    visit_warning = _visit_named_admonition
    depart_warning = _depart_named_admonition

    def visit_versionmodified(self, node: Element) -> None:
        pass

    def depart_versionmodified(self, node: Element) -> None:
        pass
```
### 2 - sphinx/writers/latex.py:

Start line: 2089, End line: 2124

```python
class LaTeXTranslator(SphinxTranslator):

    def generate_numfig_format(self, builder: "LaTeXBuilder") -> str:
        warnings.warn('generate_numfig_format() is deprecated.',
                      RemovedInSphinx40Warning, stacklevel=2)
        ret = []  # type: List[str]
        figure = self.builder.config.numfig_format['figure'].split('%s', 1)
        if len(figure) == 1:
            ret.append('\\def\\fnum@figure{%s}\n' % self.escape(figure[0]).strip())
        else:
            definition = escape_abbr(self.escape(figure[0]))
            ret.append(self.babel_renewcommand('\\figurename', definition))
            ret.append('\\makeatletter\n')
            ret.append('\\def\\fnum@figure{\\figurename\\thefigure{}%s}\n' %
                       self.escape(figure[1]))
            ret.append('\\makeatother\n')

        table = self.builder.config.numfig_format['table'].split('%s', 1)
        if len(table) == 1:
            ret.append('\\def\\fnum@table{%s}\n' % self.escape(table[0]).strip())
        else:
            definition = escape_abbr(self.escape(table[0]))
            ret.append(self.babel_renewcommand('\\tablename', definition))
            ret.append('\\makeatletter\n')
            ret.append('\\def\\fnum@table{\\tablename\\thetable{}%s}\n' %
                       self.escape(table[1]))
            ret.append('\\makeatother\n')

        codeblock = self.builder.config.numfig_format['code-block'].split('%s', 1)
        if len(codeblock) == 1:
            pass  # FIXME
        else:
            definition = self.escape(codeblock[0]).strip()
            ret.append(self.babel_renewcommand('\\literalblockname', definition))
            if codeblock[1]:
                pass  # FIXME

        return ''.join(ret)
```
### 3 - sphinx/writers/latex.py:

Start line: 14, End line: 73

```python
import re
import warnings
from collections import defaultdict
from os import path
from typing import Any, Dict, Iterable, Iterator, List, Tuple, Set, Union
from typing import cast

from docutils import nodes, writers
from docutils.nodes import Element, Node, Text

from sphinx import addnodes
from sphinx import highlighting
from sphinx.deprecation import (
    RemovedInSphinx40Warning, RemovedInSphinx50Warning, deprecated_alias
)
from sphinx.domains import IndexEntry
from sphinx.domains.std import StandardDomain
from sphinx.errors import SphinxError
from sphinx.locale import admonitionlabels, _, __
from sphinx.util import split_into, logging, texescape
from sphinx.util.docutils import SphinxTranslator
from sphinx.util.nodes import clean_astext, get_prev_node
from sphinx.util.template import LaTeXRenderer
from sphinx.util.texescape import tex_replace_map

try:
    from docutils.utils.roman import toRoman
except ImportError:
    # In Debain/Ubuntu, roman package is provided as roman, not as docutils.utils.roman
    from roman import toRoman  # type: ignore

if False:
    # For type annotation
    from sphinx.builders.latex import LaTeXBuilder
    from sphinx.builders.latex.theming import Theme


logger = logging.getLogger(__name__)

MAX_CITATION_LABEL_LENGTH = 8
LATEXSECTIONNAMES = ["part", "chapter", "section", "subsection",
                     "subsubsection", "paragraph", "subparagraph"]
ENUMERATE_LIST_STYLE = defaultdict(lambda: r'\arabic',
                                   {
                                       'arabic': r'\arabic',
                                       'loweralpha': r'\alph',
                                       'upperalpha': r'\Alph',
                                       'lowerroman': r'\roman',
                                       'upperroman': r'\Roman',
                                   })

EXTRA_RE = re.compile(r'^(.*\S)\s+\(([^()]*)\)\s*$')


class collected_footnote(nodes.footnote):
    """Footnotes that are collected are assigned this class."""


class UnsupportedError(SphinxError):
    category = 'Markup is unsupported in LaTeX'
```
### 4 - doc/conf.py:

Start line: 60, End line: 125

```python
latex_elements = {
    'fontenc': r'\usepackage[LGR,X2,T1]{fontenc}',
    'fontpkg': r'''
\usepackage[sc]{mathpazo}
\usepackage[scaled]{helvet}
\usepackage{courier}
\substitutefont{LGR}{\rmdefault}{cmr}
\substitutefont{LGR}{\sfdefault}{cmss}
\substitutefont{LGR}{\ttdefault}{cmtt}
\substitutefont{X2}{\rmdefault}{cmr}
\substitutefont{X2}{\sfdefault}{cmss}
\substitutefont{X2}{\ttdefault}{cmtt}
''',
    'passoptionstopackages': '\\PassOptionsToPackage{svgnames}{xcolor}',
    'preamble': '\\DeclareUnicodeCharacter{229E}{\\ensuremath{\\boxplus}}',
    'fvset': '\\fvset{fontsize=auto}',
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
todo_include_todos = True
extlinks = {'duref': ('http://docutils.sourceforge.net/docs/ref/rst/'
                      'restructuredtext.html#%s', ''),
            'durole': ('http://docutils.sourceforge.net/docs/ref/rst/'
                       'roles.html#%s', ''),
            'dudir': ('http://docutils.sourceforge.net/docs/ref/rst/'
                      'directives.html#%s', '')}

man_pages = [
    ('contents', 'sphinx-all', 'Sphinx documentation generator system manual',
     'Georg Brandl', 1),
    ('man/sphinx-build', 'sphinx-build', 'Sphinx documentation generator tool',
     '', 1),
    ('man/sphinx-quickstart', 'sphinx-quickstart', 'Sphinx documentation '
     'template generator', '', 1),
    ('man/sphinx-apidoc', 'sphinx-apidoc', 'Sphinx API doc generator tool',
     '', 1),
    ('man/sphinx-autogen', 'sphinx-autogen', 'Generate autodoc stub pages',
     '', 1),
]

texinfo_documents = [
    ('contents', 'sphinx', 'Sphinx Documentation', 'Georg Brandl',
     'Sphinx', 'The Sphinx documentation builder.', 'Documentation tools',
     1),
]

intersphinx_mapping = {'python': ('https://docs.python.org/3/', None)}

# Sphinx document translation with sphinx gettext feature uses these settings:
locale_dirs = ['locale/']
gettext_compact = False


# -- Extension interface -------------------------------------------------------

from sphinx import addnodes  # noqa

event_sig_re = re.compile(r'([a-zA-Z-]+)\s*\((.*)\)')
```
### 5 - utils/checks.py:

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
### 6 - sphinx/io.py:

Start line: 10, End line: 47

```python
import codecs
import warnings
from typing import Any, List

from docutils import nodes
from docutils.core import Publisher
from docutils.frontend import Values
from docutils.io import FileInput, Input, NullOutput
from docutils.parsers import Parser
from docutils.parsers.rst import Parser as RSTParser
from docutils.readers import standalone
from docutils.transforms import Transform
from docutils.transforms.references import DanglingReferences
from docutils.writers import UnfilteredWriter

from sphinx import addnodes
from sphinx.deprecation import RemovedInSphinx40Warning, deprecated_alias
from sphinx.environment import BuildEnvironment
from sphinx.errors import FiletypeNotFoundError
from sphinx.transforms import (
    AutoIndexUpgrader, DoctreeReadEvent, FigureAligner, SphinxTransformer
)
from sphinx.transforms.i18n import (
    PreserveTranslatableMessages, Locale, RemoveTranslatableInline,
)
from sphinx.transforms.references import SphinxDomains
from sphinx.util import logging, get_filetype
from sphinx.util import UnicodeDecodeErrorHandler
from sphinx.util.docutils import LoggingReporter
from sphinx.versioning import UIDTransform

if False:
    # For type annotation
    from typing import Type  # for python3.5.1
    from sphinx.application import Sphinx


logger = logging.getLogger(__name__)
```
### 7 - sphinx/writers/texinfo.py:

Start line: 1090, End line: 1189

```python
class TexinfoTranslator(SphinxTranslator):

    def _visit_named_admonition(self, node: Element) -> None:
        label = admonitionlabels[node.tagname]
        self.body.append('\n@cartouche\n@quotation %s ' % label)

    def depart_admonition(self, node: Element) -> None:
        self.ensure_eol()
        self.body.append('@end quotation\n'
                         '@end cartouche\n')

    visit_attention = _visit_named_admonition
    depart_attention = depart_admonition
    visit_caution = _visit_named_admonition
    depart_caution = depart_admonition
    visit_danger = _visit_named_admonition
    depart_danger = depart_admonition
    visit_error = _visit_named_admonition
    depart_error = depart_admonition
    visit_hint = _visit_named_admonition
    depart_hint = depart_admonition
    visit_important = _visit_named_admonition
    depart_important = depart_admonition
    visit_note = _visit_named_admonition
    depart_note = depart_admonition
    visit_tip = _visit_named_admonition
    depart_tip = depart_admonition
    visit_warning = _visit_named_admonition
    depart_warning = depart_admonition

    # -- Misc

    def visit_docinfo(self, node: Element) -> None:
        raise nodes.SkipNode

    def visit_generated(self, node: Element) -> None:
        raise nodes.SkipNode

    def visit_header(self, node: Element) -> None:
        raise nodes.SkipNode

    def visit_footer(self, node: Element) -> None:
        raise nodes.SkipNode

    def visit_container(self, node: Element) -> None:
        if node.get('literal_block'):
            self.body.append('\n\n@float LiteralBlock\n')

    def depart_container(self, node: Element) -> None:
        if node.get('literal_block'):
            self.body.append('\n@end float\n\n')

    def visit_decoration(self, node: Element) -> None:
        pass

    def depart_decoration(self, node: Element) -> None:
        pass

    def visit_topic(self, node: Element) -> None:
        # ignore TOC's since we have to have a "menu" anyway
        if 'contents' in node.get('classes', []):
            raise nodes.SkipNode
        title = cast(nodes.title, node[0])
        self.visit_rubric(title)
        self.body.append('%s\n' % self.escape(title.astext()))
        self.depart_rubric(title)

    def depart_topic(self, node: Element) -> None:
        pass

    def visit_transition(self, node: Element) -> None:
        self.body.append('\n\n%s\n\n' % ('_' * 66))

    def depart_transition(self, node: Element) -> None:
        pass

    def visit_attribution(self, node: Element) -> None:
        self.body.append('\n\n@center --- ')

    def depart_attribution(self, node: Element) -> None:
        self.body.append('\n\n')

    def visit_raw(self, node: Element) -> None:
        format = node.get('format', '').split()
        if 'texinfo' in format or 'texi' in format:
            self.body.append(node.astext())
        raise nodes.SkipNode

    def visit_figure(self, node: Element) -> None:
        self.body.append('\n\n@float Figure\n')

    def depart_figure(self, node: Element) -> None:
        self.body.append('\n@end float\n\n')

    def visit_caption(self, node: Element) -> None:
        if (isinstance(node.parent, nodes.figure) or
           (isinstance(node.parent, nodes.container) and
                node.parent.get('literal_block'))):
            self.body.append('\n@caption{')
        else:
            logger.warning(__('caption not inside a figure.'),
                           location=node)
```
### 8 - sphinx/util/nodes.py:

Start line: 11, End line: 41

```python
import re
import unicodedata
import warnings
from typing import Any, Callable, Iterable, List, Set, Tuple
from typing import cast

from docutils import nodes
from docutils.nodes import Element, Node
from docutils.parsers.rst import Directive
from docutils.parsers.rst.states import Inliner
from docutils.statemachine import StringList

from sphinx import addnodes
from sphinx.deprecation import RemovedInSphinx40Warning
from sphinx.locale import __
from sphinx.util import logging

if False:
    # For type annotation
    from typing import Type  # for python3.5.1
    from sphinx.builders import Builder
    from sphinx.domain import IndexEntry
    from sphinx.environment import BuildEnvironment
    from sphinx.util.tags import Tags

logger = logging.getLogger(__name__)


# \x00 means the "<" was backslash-escaped
explicit_title_re = re.compile(r'^(.+?)\s*(?<!\x00)<([^<]*?)>$', re.DOTALL)
caption_ref_re = explicit_title_re
```
### 9 - sphinx/writers/html.py:

Start line: 460, End line: 482

```python
class HTMLTranslator(SphinxTranslator, BaseTranslator):

    def visit_caption(self, node: Element) -> None:
        if isinstance(node.parent, nodes.container) and node.parent.get('literal_block'):
            self.body.append('<div class="code-block-caption">')
        else:
            super().visit_caption(node)
        self.add_fignumber(node.parent)
        self.body.append(self.starttag(node, 'span', '', CLASS='caption-text'))

    def depart_caption(self, node: Element) -> None:
        self.body.append('</span>')

        # append permalink if available
        if isinstance(node.parent, nodes.container) and node.parent.get('literal_block'):
            self.add_permalink_ref(node.parent, _('Permalink to this code'))
        elif isinstance(node.parent, nodes.figure):
            self.add_permalink_ref(node.parent, _('Permalink to this image'))
        elif node.parent.get('toctree'):
            self.add_permalink_ref(node.parent.parent, _('Permalink to this toctree'))

        if isinstance(node.parent, nodes.container) and node.parent.get('literal_block'):
            self.body.append('</div>\n')
        else:
            super().depart_caption(node)
```
### 10 - sphinx/builders/html/__init__.py:

Start line: 11, End line: 67

```python
import html
import posixpath
import re
import sys
import warnings
from os import path
from typing import Any, Dict, IO, Iterable, Iterator, List, Set, Tuple
from urllib.parse import quote

from docutils import nodes
from docutils.core import publish_parts
from docutils.frontend import OptionParser
from docutils.io import DocTreeInput, StringOutput
from docutils.nodes import Node
from docutils.utils import relative_path

from sphinx import package_dir, __display_version__
from sphinx.application import Sphinx
from sphinx.builders import Builder
from sphinx.config import Config, ENUM
from sphinx.deprecation import RemovedInSphinx40Warning
from sphinx.domains import Domain, Index, IndexEntry
from sphinx.environment.adapters.asset import ImageAdapter
from sphinx.environment.adapters.indexentries import IndexEntries
from sphinx.environment.adapters.toctree import TocTree
from sphinx.errors import ConfigError, ThemeError
from sphinx.highlighting import PygmentsBridge
from sphinx.locale import _, __
from sphinx.search import js_index
from sphinx.theming import HTMLThemeFactory
from sphinx.util import logging, progress_message, status_iterator, md5
from sphinx.util.docutils import is_html5_writer_available, new_document
from sphinx.util.fileutil import copy_asset
from sphinx.util.i18n import format_date
from sphinx.util.inventory import InventoryFile
from sphinx.util.matching import patmatch, Matcher, DOTFILES
from sphinx.util.osutil import os_path, relative_uri, ensuredir, movefile, copyfile
from sphinx.util.tags import Tags
from sphinx.writers.html import HTMLWriter, HTMLTranslator

if False:
    # For type annotation
    from typing import Type  # for python3.5.1


# HTML5 Writer is available or not
if is_html5_writer_available():
    from sphinx.writers.html5 import HTML5Translator
    html5_ready = True
else:
    html5_ready = False

#: the filename for the inventory of objects
INVENTORY_FILENAME = 'objects.inv'

logger = logging.getLogger(__name__)
return_codes_re = re.compile('[\r\n]+')
```
### 75 - sphinx/transforms/post_transforms/__init__.py:

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
### 77 - sphinx/domains/std.py:

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
