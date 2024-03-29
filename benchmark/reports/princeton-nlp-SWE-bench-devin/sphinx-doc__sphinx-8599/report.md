# sphinx-doc__sphinx-8599

| **sphinx-doc/sphinx** | `3a0a6556c59a7b31586dd97b43101f8dbfd2ef63` |
| ---- | ---- |
| **No of patches** | 3 |
| **All found context length** | 37660 |
| **Any found context length** | 507 |
| **Avg pos** | 173.66666666666666 |
| **Min pos** | 1 |
| **Max pos** | 84 |
| **Top file pos** | 1 |
| **Missing snippets** | 15 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sphinx/builders/html/__init__.py b/sphinx/builders/html/__init__.py
--- a/sphinx/builders/html/__init__.py
+++ b/sphinx/builders/html/__init__.py
@@ -1205,6 +1205,16 @@ def validate_html_favicon(app: Sphinx, config: Config) -> None:
         config.html_favicon = None  # type: ignore
 
 
+def migrate_html_add_permalinks(app: Sphinx, config: Config) -> None:
+    """Migrate html_add_permalinks to html_permalinks*."""
+    if config.html_add_permalinks:
+        if (isinstance(config.html_add_permalinks, bool) and
+                config.html_add_permalinks is False):
+            config.html_permalinks = False  # type: ignore
+        else:
+            config.html_permalinks_icon = html.escape(config.html_add_permalinks)  # type: ignore  # NOQA
+
+
 # for compatibility
 import sphinxcontrib.serializinghtml  # NOQA
 
@@ -1235,7 +1245,9 @@ def setup(app: Sphinx) -> Dict[str, Any]:
     app.add_config_value('html_sidebars', {}, 'html')
     app.add_config_value('html_additional_pages', {}, 'html')
     app.add_config_value('html_domain_indices', True, 'html', [list])
-    app.add_config_value('html_add_permalinks', '¶', 'html')
+    app.add_config_value('html_add_permalinks', None, 'html')
+    app.add_config_value('html_permalinks', True, 'html')
+    app.add_config_value('html_permalinks_icon', '¶', 'html')
     app.add_config_value('html_use_index', True, 'html')
     app.add_config_value('html_split_index', False, 'html')
     app.add_config_value('html_copy_source', True, 'html')
@@ -1267,6 +1279,7 @@ def setup(app: Sphinx) -> Dict[str, Any]:
     # event handlers
     app.connect('config-inited', convert_html_css_files, priority=800)
     app.connect('config-inited', convert_html_js_files, priority=800)
+    app.connect('config-inited', migrate_html_add_permalinks, priority=800)
     app.connect('config-inited', validate_html_extra_path, priority=800)
     app.connect('config-inited', validate_html_static_path, priority=800)
     app.connect('config-inited', validate_html_logo, priority=800)
diff --git a/sphinx/writers/html.py b/sphinx/writers/html.py
--- a/sphinx/writers/html.py
+++ b/sphinx/writers/html.py
@@ -22,7 +22,7 @@
 
 from sphinx import addnodes
 from sphinx.builders import Builder
-from sphinx.deprecation import RemovedInSphinx40Warning
+from sphinx.deprecation import RemovedInSphinx40Warning, RemovedInSphinx50Warning
 from sphinx.locale import _, __, admonitionlabels
 from sphinx.util import logging
 from sphinx.util.docutils import SphinxTranslator
@@ -100,11 +100,6 @@ def __init__(self, *args: Any) -> None:
         self.docnames = [self.builder.current_docname]  # for singlehtml builder
         self.manpages_url = self.config.manpages_url
         self.protect_literal_text = 0
-        self.permalink_text = self.config.html_add_permalinks
-        # support backwards-compatible setting to a bool
-        if not isinstance(self.permalink_text, str):
-            self.permalink_text = '¶' if self.permalink_text else ''
-        self.permalink_text = self.encode(self.permalink_text)
         self.secnumber_suffix = self.config.html_secnumber_suffix
         self.param_separator = ''
         self.optional_param_level = 0
@@ -333,9 +328,10 @@ def append_fignumber(figtype: str, figure_id: str) -> None:
                 append_fignumber(figtype, node['ids'][0])
 
     def add_permalink_ref(self, node: Element, title: str) -> None:
-        if node['ids'] and self.permalink_text and self.builder.add_permalinks:
+        if node['ids'] and self.config.html_permalinks and self.builder.add_permalinks:
             format = '<a class="headerlink" href="#%s" title="%s">%s</a>'
-            self.body.append(format % (node['ids'][0], title, self.permalink_text))
+            self.body.append(format % (node['ids'][0], title,
+                                       self.config.html_permalinks_icon))
 
     def generate_targets_for_listing(self, node: Element) -> None:
         """Generate hyperlink targets for listings.
@@ -410,7 +406,7 @@ def visit_title(self, node: Element) -> None:
 
     def depart_title(self, node: Element) -> None:
         close_tag = self.context[-1]
-        if (self.permalink_text and self.builder.add_permalinks and
+        if (self.config.html_permalinks and self.builder.add_permalinks and
            node.parent.hasattr('ids') and node.parent['ids']):
             # add permalink anchor
             if close_tag.startswith('</h'):
@@ -420,7 +416,7 @@ def depart_title(self, node: Element) -> None:
                                  node.parent['ids'][0] +
                                  'title="%s">%s' % (
                                      _('Permalink to this headline'),
-                                     self.permalink_text))
+                                     self.config.html_permalinks_icon))
             elif isinstance(node.parent, nodes.table):
                 self.body.append('</span>')
                 self.add_permalink_ref(node.parent, _('Permalink to this table'))
@@ -838,3 +834,9 @@ def depart_math_block(self, node: Element, math_env: str = '') -> None:
 
     def unknown_visit(self, node: Node) -> None:
         raise NotImplementedError('Unknown node: ' + node.__class__.__name__)
+
+    @property
+    def permalink_text(self) -> str:
+        warnings.warn('HTMLTranslator.permalink_text is deprecated.',
+                      RemovedInSphinx50Warning, stacklevel=2)
+        return self.config.html_permalinks_icon
diff --git a/sphinx/writers/html5.py b/sphinx/writers/html5.py
--- a/sphinx/writers/html5.py
+++ b/sphinx/writers/html5.py
@@ -20,7 +20,7 @@
 
 from sphinx import addnodes
 from sphinx.builders import Builder
-from sphinx.deprecation import RemovedInSphinx40Warning
+from sphinx.deprecation import RemovedInSphinx40Warning, RemovedInSphinx50Warning
 from sphinx.locale import _, __, admonitionlabels
 from sphinx.util import logging
 from sphinx.util.docutils import SphinxTranslator
@@ -71,11 +71,6 @@ def __init__(self, *args: Any) -> None:
         self.docnames = [self.builder.current_docname]  # for singlehtml builder
         self.manpages_url = self.config.manpages_url
         self.protect_literal_text = 0
-        self.permalink_text = self.config.html_add_permalinks
-        # support backwards-compatible setting to a bool
-        if not isinstance(self.permalink_text, str):
-            self.permalink_text = '¶' if self.permalink_text else ''
-        self.permalink_text = self.encode(self.permalink_text)
         self.secnumber_suffix = self.config.html_secnumber_suffix
         self.param_separator = ''
         self.optional_param_level = 0
@@ -304,9 +299,10 @@ def append_fignumber(figtype: str, figure_id: str) -> None:
                 append_fignumber(figtype, node['ids'][0])
 
     def add_permalink_ref(self, node: Element, title: str) -> None:
-        if node['ids'] and self.permalink_text and self.builder.add_permalinks:
+        if node['ids'] and self.config.html_permalinks and self.builder.add_permalinks:
             format = '<a class="headerlink" href="#%s" title="%s">%s</a>'
-            self.body.append(format % (node['ids'][0], title, self.permalink_text))
+            self.body.append(format % (node['ids'][0], title,
+                                       self.config.html_permalinks_icon))
 
     # overwritten
     def visit_bullet_list(self, node: Element) -> None:
@@ -361,8 +357,8 @@ def visit_title(self, node: Element) -> None:
 
     def depart_title(self, node: Element) -> None:
         close_tag = self.context[-1]
-        if (self.permalink_text and self.builder.add_permalinks and
-           node.parent.hasattr('ids') and node.parent['ids']):
+        if (self.config.html_permalinks and self.builder.add_permalinks and
+                node.parent.hasattr('ids') and node.parent['ids']):
             # add permalink anchor
             if close_tag.startswith('</h'):
                 self.add_permalink_ref(node.parent, _('Permalink to this headline'))
@@ -371,7 +367,7 @@ def depart_title(self, node: Element) -> None:
                                  node.parent['ids'][0] +
                                  'title="%s">%s' % (
                                      _('Permalink to this headline'),
-                                     self.permalink_text))
+                                     self.config.html_permalinks_icon))
             elif isinstance(node.parent, nodes.table):
                 self.body.append('</span>')
                 self.add_permalink_ref(node.parent, _('Permalink to this table'))
@@ -786,3 +782,9 @@ def depart_math_block(self, node: Element, math_env: str = '') -> None:
 
     def unknown_visit(self, node: Node) -> None:
         raise NotImplementedError('Unknown node: ' + node.__class__.__name__)
+
+    @property
+    def permalink_text(self) -> str:
+        warnings.warn('HTMLTranslator.permalink_text is deprecated.',
+                      RemovedInSphinx50Warning, stacklevel=2)
+        return self.config.html_permalinks_icon

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sphinx/builders/html/__init__.py | 1208 | 1208 | 84 | 6 | 37660
| sphinx/builders/html/__init__.py | 1238 | 1238 | 7 | 6 | 3674
| sphinx/builders/html/__init__.py | 1270 | 1270 | 7 | 6 | 3674
| sphinx/writers/html.py | 25 | 25 | - | 5 | -
| sphinx/writers/html.py | 103 | 107 | 6 | 5 | 2765
| sphinx/writers/html.py | 336 | 338 | 36 | 5 | 17102
| sphinx/writers/html.py | 413 | 413 | 70 | 5 | 31126
| sphinx/writers/html.py | 423 | 423 | 70 | 5 | 31126
| sphinx/writers/html.py | 841 | 841 | 29 | 5 | 14787
| sphinx/writers/html5.py | 23 | 23 | - | 1 | -
| sphinx/writers/html5.py | 74 | 78 | 12 | 1 | 5971
| sphinx/writers/html5.py | 307 | 309 | 1 | 1 | 507
| sphinx/writers/html5.py | 364 | 365 | 63 | 1 | 28347
| sphinx/writers/html5.py | 374 | 374 | 63 | 1 | 28347
| sphinx/writers/html5.py | 789 | 789 | 73 | 1 | 32267


## Problem Statement

```
Allow custom link texts for permalinks and links to source code.
I'd like to be able to customize the content of the HTML links that Sphinx generates for permalinks and links to source code (generated by the `viewcode` extension).

E.g. instead of ``<a class="headerlink" href="..." title="Permalink to this definition">¶</a>``, I'd like to have ``<a class="headerlink" href="..." title="Permalink to this definition"><i class="fas fa-link"></i></a>`` to use [FontAwesome](https://fontawesome.com/) icons.

Note that the "Read The Docs" theme does this by fidling with the CSS to hide the text of the link and add's the icon via some `:after:` CSS rules.

IMHO it would be much clearer if this was customizable via configuration options.

This patch adds two configuration options:

`html_add_permalinks_html` which does the same as `html_add_permalinks`, but interprets the value as HTML, not as text.

`viewcode_source_html` which will be used as the link content for source code links generated by the `viewcode` extension. (The default is `<span class="viewcode-link">[source]</span>`).

This is my first attempt to work with the Sphinx source code, and I'm not exactly sure, whether defining a new node for the viewcode link text and replacing that node with the configured HTML in the HTML writer is the correct approach.

However with this patch I can now put
\`\`\`python
html_add_permalinks_html = '<i class="fa fa-link"></i>'

viewcode_source_html = '<span class="viewcode-link"><i class="fa fa-code"></i></span>'
\`\`\`

in my `conf.py` to get nice Font Awesome icons.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 sphinx/writers/html5.py** | 306 | 360| 507 | 507 | 6895 | 
| 2 | 2 sphinx/directives/__init__.py | 269 | 323| 517 | 1024 | 9645 | 
| 3 | 3 doc/conf.py | 59 | 127| 688 | 1712 | 11203 | 
| 4 | 4 sphinx/directives/patches.py | 53 | 67| 143 | 1855 | 12813 | 
| 5 | 4 sphinx/directives/patches.py | 9 | 24| 121 | 1976 | 12813 | 
| **-> 6 <-** | **5 sphinx/writers/html.py** | 82 | 169| 789 | 2765 | 20181 | 
| **-> 7 <-** | **6 sphinx/builders/html/__init__.py** | 1212 | 1288| 909 | 3674 | 31732 | 
| 8 | **6 sphinx/writers/html5.py** | 407 | 429| 239 | 3913 | 31732 | 
| 9 | **6 sphinx/builders/html/__init__.py** | 11 | 68| 462 | 4375 | 31732 | 
| 10 | **6 sphinx/writers/html.py** | 171 | 231| 562 | 4937 | 31732 | 
| 11 | **6 sphinx/writers/html.py** | 456 | 478| 238 | 5175 | 31732 | 
| **-> 12 <-** | **6 sphinx/writers/html5.py** | 53 | 140| 796 | 5971 | 31732 | 
| 13 | 7 sphinx/ext/linkcode.py | 11 | 80| 470 | 6441 | 32258 | 
| 14 | **7 sphinx/writers/html5.py** | 142 | 202| 567 | 7008 | 32258 | 
| 15 | 8 sphinx/ext/extlinks.py | 26 | 71| 400 | 7408 | 32877 | 
| 16 | **8 sphinx/writers/html5.py** | 629 | 715| 686 | 8094 | 32877 | 
| 17 | 8 doc/conf.py | 1 | 58| 511 | 8605 | 32877 | 
| 18 | 9 sphinx/builders/latex/__init__.py | 11 | 42| 338 | 8943 | 38877 | 
| 19 | **9 sphinx/writers/html.py** | 693 | 792| 803 | 9746 | 38877 | 
| 20 | 10 sphinx/highlighting.py | 55 | 164| 876 | 10622 | 40179 | 
| 21 | 11 sphinx/builders/_epub_base.py | 11 | 101| 657 | 11279 | 46914 | 
| 22 | 12 sphinx/builders/latex/theming.py | 11 | 48| 256 | 11535 | 47941 | 
| 23 | 13 sphinx/directives/code.py | 9 | 32| 154 | 11689 | 51873 | 
| 24 | **13 sphinx/writers/html.py** | 354 | 409| 472 | 12161 | 51873 | 
| 25 | **13 sphinx/writers/html.py** | 625 | 671| 362 | 12523 | 51873 | 
| 26 | 14 sphinx/ext/graphviz.py | 265 | 314| 547 | 13070 | 55523 | 
| 27 | 15 sphinx/writers/latex.py | 2119 | 2158| 467 | 13537 | 75363 | 
| 28 | 16 sphinx/writers/texinfo.py | 865 | 969| 788 | 14325 | 87676 | 
| **-> 29 <-** | **16 sphinx/writers/html.py** | 794 | 841| 462 | 14787 | 87676 | 
| 30 | 16 sphinx/ext/graphviz.py | 317 | 351| 289 | 15076 | 87676 | 
| 31 | 17 sphinx/ext/viewcode.py | 320 | 338| 202 | 15278 | 90546 | 
| 32 | 17 sphinx/directives/code.py | 419 | 483| 658 | 15936 | 90546 | 
| 33 | 18 sphinx/builders/latex/transforms.py | 79 | 100| 214 | 16150 | 94826 | 
| 34 | 18 sphinx/writers/latex.py | 1609 | 1668| 496 | 16646 | 94826 | 
| 35 | 19 sphinx/transforms/__init__.py | 11 | 44| 229 | 16875 | 98078 | 
| **-> 36 <-** | **19 sphinx/writers/html.py** | 335 | 352| 227 | 17102 | 98078 | 
| 37 | 20 sphinx/ext/todo.py | 304 | 318| 143 | 17245 | 100792 | 
| 38 | 21 sphinx/application.py | 981 | 1036| 582 | 17827 | 112488 | 
| 39 | **21 sphinx/writers/html5.py** | 565 | 607| 321 | 18148 | 112488 | 
| 40 | 21 sphinx/directives/patches.py | 109 | 151| 304 | 18452 | 112488 | 
| 41 | 22 sphinx/writers/text.py | 1044 | 1156| 813 | 19265 | 121461 | 
| 42 | 22 sphinx/ext/viewcode.py | 11 | 45| 228 | 19493 | 121461 | 
| 43 | **22 sphinx/writers/html5.py** | 231 | 252| 208 | 19701 | 121461 | 
| 44 | 22 sphinx/builders/latex/transforms.py | 102 | 127| 277 | 19978 | 121461 | 
| 45 | 23 sphinx/environment/__init__.py | 11 | 81| 485 | 20463 | 127301 | 
| 46 | **23 sphinx/writers/html.py** | 260 | 282| 214 | 20677 | 127301 | 
| 47 | 24 sphinx/builders/changes.py | 125 | 168| 438 | 21115 | 128823 | 
| 48 | 25 sphinx/util/console.py | 101 | 142| 296 | 21411 | 129816 | 
| 49 | **25 sphinx/writers/html5.py** | 203 | 229| 319 | 21730 | 129816 | 
| 50 | 25 sphinx/writers/text.py | 647 | 764| 839 | 22569 | 129816 | 
| 51 | 25 sphinx/writers/latex.py | 1519 | 1582| 617 | 23186 | 129816 | 
| 52 | 25 sphinx/builders/_epub_base.py | 293 | 311| 205 | 23391 | 129816 | 
| 53 | **25 sphinx/writers/html.py** | 232 | 258| 318 | 23709 | 129816 | 
| 54 | 26 sphinx/addnodes.py | 281 | 380| 580 | 24289 | 132562 | 
| 55 | **26 sphinx/writers/html.py** | 480 | 505| 238 | 24527 | 132562 | 
| 56 | 26 sphinx/writers/texinfo.py | 1303 | 1386| 678 | 25205 | 132562 | 
| 57 | 27 sphinx/builders/latex/constants.py | 69 | 118| 523 | 25728 | 134642 | 
| 58 | **27 sphinx/writers/html5.py** | 608 | 627| 201 | 25929 | 134642 | 
| 59 | 28 sphinx/config.py | 97 | 155| 726 | 26655 | 139243 | 
| 60 | **28 sphinx/writers/html5.py** | 431 | 456| 239 | 26894 | 139243 | 
| 61 | 28 sphinx/writers/latex.py | 793 | 856| 573 | 27467 | 139243 | 
| 62 | 28 sphinx/writers/latex.py | 1327 | 1395| 667 | 28134 | 139243 | 
| **-> 63 <-** | **28 sphinx/writers/html5.py** | 362 | 383| 213 | 28347 | 139243 | 
| 64 | 28 sphinx/application.py | 1038 | 1054| 141 | 28488 | 139243 | 
| 65 | **28 sphinx/writers/html.py** | 55 | 79| 262 | 28750 | 139243 | 
| 66 | **28 sphinx/writers/html.py** | 560 | 582| 203 | 28953 | 139243 | 
| 67 | 28 sphinx/writers/latex.py | 689 | 791| 825 | 29778 | 139243 | 
| 68 | 28 sphinx/writers/texinfo.py | 971 | 1088| 846 | 30624 | 139243 | 
| 69 | 28 sphinx/ext/viewcode.py | 141 | 171| 290 | 30914 | 139243 | 
| **-> 70 <-** | **28 sphinx/writers/html.py** | 411 | 432| 212 | 31126 | 139243 | 
| 71 | **28 sphinx/writers/html5.py** | 500 | 522| 204 | 31330 | 139243 | 
| 72 | 29 sphinx/util/__init__.py | 11 | 71| 506 | 31836 | 145545 | 
| **-> 73 <-** | **29 sphinx/writers/html5.py** | 745 | 789| 431 | 32267 | 145545 | 
| 74 | 29 sphinx/writers/text.py | 929 | 1042| 873 | 33140 | 145545 | 
| 75 | 30 sphinx/transforms/i18n.py | 229 | 384| 1673 | 34813 | 150145 | 
| 76 | 30 sphinx/directives/patches.py | 154 | 182| 218 | 35031 | 150145 | 
| 77 | 30 sphinx/builders/latex/transforms.py | 11 | 49| 327 | 35358 | 150145 | 
| 78 | 31 sphinx/writers/manpage.py | 308 | 406| 757 | 36115 | 153654 | 
| 79 | **31 sphinx/writers/html.py** | 672 | 691| 200 | 36315 | 153654 | 
| 80 | 32 sphinx/io.py | 10 | 45| 275 | 36590 | 155397 | 
| 81 | **32 sphinx/writers/html.py** | 527 | 558| 248 | 36838 | 155397 | 
| 82 | 33 sphinx/ext/imgmath.py | 382 | 402| 289 | 37127 | 158795 | 
| 83 | 34 sphinx/domains/rst.py | 139 | 170| 356 | 37483 | 161263 | 
| **-> 84 <-** | **34 sphinx/builders/html/__init__.py** | 1194 | 1212| 177 | 37660 | 161263 | 
| 85 | 34 sphinx/ext/todo.py | 14 | 43| 204 | 37864 | 161263 | 
| 86 | 34 sphinx/writers/latex.py | 286 | 440| 1717 | 39581 | 161263 | 
| 87 | 34 sphinx/builders/latex/transforms.py | 52 | 77| 168 | 39749 | 161263 | 
| 88 | 34 sphinx/directives/code.py | 35 | 58| 173 | 39922 | 161263 | 
| 89 | 34 sphinx/transforms/__init__.py | 142 | 163| 214 | 40136 | 161263 | 
| 90 | 34 sphinx/ext/graphviz.py | 397 | 413| 196 | 40332 | 161263 | 
| 91 | 35 sphinx/builders/singlehtml.py | 190 | 213| 141 | 40473 | 163140 | 
| 92 | 36 sphinx/util/pycompat.py | 11 | 53| 357 | 40830 | 164008 | 
| 93 | 36 sphinx/addnodes.py | 383 | 428| 326 | 41156 | 164008 | 
| 94 | 36 sphinx/writers/text.py | 528 | 627| 640 | 41796 | 164008 | 
| 95 | 36 sphinx/config.py | 370 | 404| 294 | 42090 | 164008 | 
| 96 | 36 sphinx/writers/latex.py | 1397 | 1451| 463 | 42553 | 164008 | 
| 97 | **36 sphinx/writers/html5.py** | 478 | 498| 156 | 42709 | 164008 | 
| 98 | 36 sphinx/directives/patches.py | 27 | 50| 187 | 42896 | 164008 | 
| 99 | 36 sphinx/builders/_epub_base.py | 334 | 359| 301 | 43197 | 164008 | 
| 100 | 36 sphinx/transforms/i18n.py | 385 | 451| 765 | 43962 | 164008 | 
| 101 | 37 sphinx/pygments_styles.py | 38 | 96| 506 | 44468 | 164700 | 
| 102 | 37 sphinx/writers/latex.py | 442 | 502| 569 | 45037 | 164700 | 
| 103 | 38 sphinx/search/__init__.py | 10 | 29| 128 | 45165 | 168701 | 
| 104 | 38 sphinx/config.py | 254 | 289| 326 | 45491 | 168701 | 
| 105 | 38 sphinx/addnodes.py | 150 | 261| 710 | 46201 | 168701 | 
| 106 | **38 sphinx/builders/html/__init__.py** | 915 | 968| 492 | 46693 | 168701 | 
| 107 | 38 sphinx/transforms/__init__.py | 274 | 315| 309 | 47002 | 168701 | 
| 108 | 39 sphinx/ext/autodoc/__init__.py | 519 | 539| 235 | 47237 | 191238 | 
| 109 | 39 sphinx/writers/texinfo.py | 1090 | 1189| 839 | 48076 | 191238 | 
| 110 | 39 sphinx/writers/texinfo.py | 1477 | 1552| 549 | 48625 | 191238 | 
| 111 | 39 sphinx/builders/latex/constants.py | 120 | 202| 925 | 49550 | 191238 | 
| 112 | 40 doc/development/tutorials/examples/todo.py | 30 | 53| 175 | 49725 | 192146 | 
| 113 | 40 sphinx/writers/texinfo.py | 755 | 863| 813 | 50538 | 192146 | 
| 114 | 40 sphinx/writers/texinfo.py | 1217 | 1282| 457 | 50995 | 192146 | 
| 115 | 41 sphinx/util/logging.py | 11 | 57| 279 | 51274 | 195980 | 
| 116 | **41 sphinx/writers/html5.py** | 280 | 304| 271 | 51545 | 195980 | 
| 117 | 41 sphinx/ext/todo.py | 321 | 344| 214 | 51759 | 195980 | 
| 118 | 41 sphinx/application.py | 64 | 126| 494 | 52253 | 195980 | 
| 119 | 42 sphinx/transforms/post_transforms/code.py | 112 | 141| 208 | 52461 | 197005 | 


### Hint

```
Thank you for proposal. I agree with your idea. But it would better to customize it via CSS to me. How about wrapping these labels with `<span>`? Adding a custom node will cause incompatibility to writers. Please take a look errors in Travis CI.
I agree that adding a custom node might be to invasive, but I don't know how else it would be possible to influence the generated HTML.

Tests seem to pass now, but I don't know whether `getattr` is the correct approach for that.
Hmm, the unused import that https://travis-ci.org/sphinx-doc/sphinx/jobs/555878900 complains about, seems to be used in the type annotation comments.
I've updated the pull request and removed the changes to the viewcode link. So there are no longer any custom nodes introduced by the patch. Only the option `html_add_permalinks_html` remains. Is there any chance for this patch to be merged?
Finally, I determined to support custom link text for permalinks. But it is not better to provide two ways to change the link text; `html_add_permalinks` and `html_add_permalinks_html`. Now I'm thinking about a new interface to control link text.

* `html_permalinks = True | False`: Enable or disable permalinks feature.
* `html_permalinks_icon = "¶"`: A text for the label of permalink. HTML tags are allowed.

Note: I think "add" prefix is a bit strange for the configuration name. So the new interface does not use it.

@shimizukawa Please let me know your idea if you have. 
```

## Patch

```diff
diff --git a/sphinx/builders/html/__init__.py b/sphinx/builders/html/__init__.py
--- a/sphinx/builders/html/__init__.py
+++ b/sphinx/builders/html/__init__.py
@@ -1205,6 +1205,16 @@ def validate_html_favicon(app: Sphinx, config: Config) -> None:
         config.html_favicon = None  # type: ignore
 
 
+def migrate_html_add_permalinks(app: Sphinx, config: Config) -> None:
+    """Migrate html_add_permalinks to html_permalinks*."""
+    if config.html_add_permalinks:
+        if (isinstance(config.html_add_permalinks, bool) and
+                config.html_add_permalinks is False):
+            config.html_permalinks = False  # type: ignore
+        else:
+            config.html_permalinks_icon = html.escape(config.html_add_permalinks)  # type: ignore  # NOQA
+
+
 # for compatibility
 import sphinxcontrib.serializinghtml  # NOQA
 
@@ -1235,7 +1245,9 @@ def setup(app: Sphinx) -> Dict[str, Any]:
     app.add_config_value('html_sidebars', {}, 'html')
     app.add_config_value('html_additional_pages', {}, 'html')
     app.add_config_value('html_domain_indices', True, 'html', [list])
-    app.add_config_value('html_add_permalinks', '¶', 'html')
+    app.add_config_value('html_add_permalinks', None, 'html')
+    app.add_config_value('html_permalinks', True, 'html')
+    app.add_config_value('html_permalinks_icon', '¶', 'html')
     app.add_config_value('html_use_index', True, 'html')
     app.add_config_value('html_split_index', False, 'html')
     app.add_config_value('html_copy_source', True, 'html')
@@ -1267,6 +1279,7 @@ def setup(app: Sphinx) -> Dict[str, Any]:
     # event handlers
     app.connect('config-inited', convert_html_css_files, priority=800)
     app.connect('config-inited', convert_html_js_files, priority=800)
+    app.connect('config-inited', migrate_html_add_permalinks, priority=800)
     app.connect('config-inited', validate_html_extra_path, priority=800)
     app.connect('config-inited', validate_html_static_path, priority=800)
     app.connect('config-inited', validate_html_logo, priority=800)
diff --git a/sphinx/writers/html.py b/sphinx/writers/html.py
--- a/sphinx/writers/html.py
+++ b/sphinx/writers/html.py
@@ -22,7 +22,7 @@
 
 from sphinx import addnodes
 from sphinx.builders import Builder
-from sphinx.deprecation import RemovedInSphinx40Warning
+from sphinx.deprecation import RemovedInSphinx40Warning, RemovedInSphinx50Warning
 from sphinx.locale import _, __, admonitionlabels
 from sphinx.util import logging
 from sphinx.util.docutils import SphinxTranslator
@@ -100,11 +100,6 @@ def __init__(self, *args: Any) -> None:
         self.docnames = [self.builder.current_docname]  # for singlehtml builder
         self.manpages_url = self.config.manpages_url
         self.protect_literal_text = 0
-        self.permalink_text = self.config.html_add_permalinks
-        # support backwards-compatible setting to a bool
-        if not isinstance(self.permalink_text, str):
-            self.permalink_text = '¶' if self.permalink_text else ''
-        self.permalink_text = self.encode(self.permalink_text)
         self.secnumber_suffix = self.config.html_secnumber_suffix
         self.param_separator = ''
         self.optional_param_level = 0
@@ -333,9 +328,10 @@ def append_fignumber(figtype: str, figure_id: str) -> None:
                 append_fignumber(figtype, node['ids'][0])
 
     def add_permalink_ref(self, node: Element, title: str) -> None:
-        if node['ids'] and self.permalink_text and self.builder.add_permalinks:
+        if node['ids'] and self.config.html_permalinks and self.builder.add_permalinks:
             format = '<a class="headerlink" href="#%s" title="%s">%s</a>'
-            self.body.append(format % (node['ids'][0], title, self.permalink_text))
+            self.body.append(format % (node['ids'][0], title,
+                                       self.config.html_permalinks_icon))
 
     def generate_targets_for_listing(self, node: Element) -> None:
         """Generate hyperlink targets for listings.
@@ -410,7 +406,7 @@ def visit_title(self, node: Element) -> None:
 
     def depart_title(self, node: Element) -> None:
         close_tag = self.context[-1]
-        if (self.permalink_text and self.builder.add_permalinks and
+        if (self.config.html_permalinks and self.builder.add_permalinks and
            node.parent.hasattr('ids') and node.parent['ids']):
             # add permalink anchor
             if close_tag.startswith('</h'):
@@ -420,7 +416,7 @@ def depart_title(self, node: Element) -> None:
                                  node.parent['ids'][0] +
                                  'title="%s">%s' % (
                                      _('Permalink to this headline'),
-                                     self.permalink_text))
+                                     self.config.html_permalinks_icon))
             elif isinstance(node.parent, nodes.table):
                 self.body.append('</span>')
                 self.add_permalink_ref(node.parent, _('Permalink to this table'))
@@ -838,3 +834,9 @@ def depart_math_block(self, node: Element, math_env: str = '') -> None:
 
     def unknown_visit(self, node: Node) -> None:
         raise NotImplementedError('Unknown node: ' + node.__class__.__name__)
+
+    @property
+    def permalink_text(self) -> str:
+        warnings.warn('HTMLTranslator.permalink_text is deprecated.',
+                      RemovedInSphinx50Warning, stacklevel=2)
+        return self.config.html_permalinks_icon
diff --git a/sphinx/writers/html5.py b/sphinx/writers/html5.py
--- a/sphinx/writers/html5.py
+++ b/sphinx/writers/html5.py
@@ -20,7 +20,7 @@
 
 from sphinx import addnodes
 from sphinx.builders import Builder
-from sphinx.deprecation import RemovedInSphinx40Warning
+from sphinx.deprecation import RemovedInSphinx40Warning, RemovedInSphinx50Warning
 from sphinx.locale import _, __, admonitionlabels
 from sphinx.util import logging
 from sphinx.util.docutils import SphinxTranslator
@@ -71,11 +71,6 @@ def __init__(self, *args: Any) -> None:
         self.docnames = [self.builder.current_docname]  # for singlehtml builder
         self.manpages_url = self.config.manpages_url
         self.protect_literal_text = 0
-        self.permalink_text = self.config.html_add_permalinks
-        # support backwards-compatible setting to a bool
-        if not isinstance(self.permalink_text, str):
-            self.permalink_text = '¶' if self.permalink_text else ''
-        self.permalink_text = self.encode(self.permalink_text)
         self.secnumber_suffix = self.config.html_secnumber_suffix
         self.param_separator = ''
         self.optional_param_level = 0
@@ -304,9 +299,10 @@ def append_fignumber(figtype: str, figure_id: str) -> None:
                 append_fignumber(figtype, node['ids'][0])
 
     def add_permalink_ref(self, node: Element, title: str) -> None:
-        if node['ids'] and self.permalink_text and self.builder.add_permalinks:
+        if node['ids'] and self.config.html_permalinks and self.builder.add_permalinks:
             format = '<a class="headerlink" href="#%s" title="%s">%s</a>'
-            self.body.append(format % (node['ids'][0], title, self.permalink_text))
+            self.body.append(format % (node['ids'][0], title,
+                                       self.config.html_permalinks_icon))
 
     # overwritten
     def visit_bullet_list(self, node: Element) -> None:
@@ -361,8 +357,8 @@ def visit_title(self, node: Element) -> None:
 
     def depart_title(self, node: Element) -> None:
         close_tag = self.context[-1]
-        if (self.permalink_text and self.builder.add_permalinks and
-           node.parent.hasattr('ids') and node.parent['ids']):
+        if (self.config.html_permalinks and self.builder.add_permalinks and
+                node.parent.hasattr('ids') and node.parent['ids']):
             # add permalink anchor
             if close_tag.startswith('</h'):
                 self.add_permalink_ref(node.parent, _('Permalink to this headline'))
@@ -371,7 +367,7 @@ def depart_title(self, node: Element) -> None:
                                  node.parent['ids'][0] +
                                  'title="%s">%s' % (
                                      _('Permalink to this headline'),
-                                     self.permalink_text))
+                                     self.config.html_permalinks_icon))
             elif isinstance(node.parent, nodes.table):
                 self.body.append('</span>')
                 self.add_permalink_ref(node.parent, _('Permalink to this table'))
@@ -786,3 +782,9 @@ def depart_math_block(self, node: Element, math_env: str = '') -> None:
 
     def unknown_visit(self, node: Node) -> None:
         raise NotImplementedError('Unknown node: ' + node.__class__.__name__)
+
+    @property
+    def permalink_text(self) -> str:
+        warnings.warn('HTMLTranslator.permalink_text is deprecated.',
+                      RemovedInSphinx50Warning, stacklevel=2)
+        return self.config.html_permalinks_icon

```

## Test Patch

```diff
diff --git a/tests/test_build_html.py b/tests/test_build_html.py
--- a/tests/test_build_html.py
+++ b/tests/test_build_html.py
@@ -1665,3 +1665,23 @@ def test_highlight_options_old(app):
                                     location=ANY, opts={})
         assert call_args[2] == call(ANY, 'java', force=False, linenos=False,
                                     location=ANY, opts={})
+
+
+@pytest.mark.sphinx('html', testroot='basic',
+                    confoverrides={'html_permalinks': False})
+def test_html_permalink_disable(app):
+    app.build()
+    content = (app.outdir / 'index.html').read_text()
+
+    assert '<h1>The basic Sphinx documentation for testing</h1>' in content
+
+
+@pytest.mark.sphinx('html', testroot='basic',
+                    confoverrides={'html_permalinks_icon': '<span>[PERMALINK]</span>'})
+def test_html_permalink_icon(app):
+    app.build()
+    content = (app.outdir / 'index.html').read_text()
+
+    assert ('<h1>The basic Sphinx documentation for testing<a class="headerlink" '
+            'href="#the-basic-sphinx-documentation-for-testing" '
+            'title="Permalink to this headline"><span>[PERMALINK]</span></a></h1>' in content)

```


## Code snippets

### 1 - sphinx/writers/html5.py:

Start line: 306, End line: 360

```python
class HTML5Translator(SphinxTranslator, BaseTranslator):

    def add_permalink_ref(self, node: Element, title: str) -> None:
        if node['ids'] and self.permalink_text and self.builder.add_permalinks:
            format = '<a class="headerlink" href="#%s" title="%s">%s</a>'
            self.body.append(format % (node['ids'][0], title, self.permalink_text))

    # overwritten
    def visit_bullet_list(self, node: Element) -> None:
        if len(node) == 1 and isinstance(node[0], addnodes.toctree):
            # avoid emitting empty <ul></ul>
            raise nodes.SkipNode
        super().visit_bullet_list(node)

    # overwritten
    def visit_definition(self, node: Element) -> None:
        # don't insert </dt> here.
        self.body.append(self.starttag(node, 'dd', ''))

    # overwritten
    def depart_definition(self, node: Element) -> None:
        self.body.append('</dd>\n')

    # overwritten
    def visit_classifier(self, node: Element) -> None:
        self.body.append(self.starttag(node, 'span', '', CLASS='classifier'))

    # overwritten
    def depart_classifier(self, node: Element) -> None:
        self.body.append('</span>')

        next_node = node.next_node(descend=False, siblings=True)  # type: Node
        if not isinstance(next_node, nodes.classifier):
            # close `<dt>` tag at the tail of classifiers
            self.body.append('</dt>')

    # overwritten
    def visit_term(self, node: Element) -> None:
        self.body.append(self.starttag(node, 'dt', ''))

    # overwritten
    def depart_term(self, node: Element) -> None:
        next_node = node.next_node(descend=False, siblings=True)  # type: Node
        if isinstance(next_node, nodes.classifier):
            # Leave the end tag to `self.depart_classifier()`, in case
            # there's a classifier.
            pass
        else:
            self.body.append('</dt>')

    # overwritten
    def visit_title(self, node: Element) -> None:
        super().visit_title(node)
        self.add_secnumber(node)
        self.add_fignumber(node.parent)
        if isinstance(node.parent, nodes.table):
            self.body.append('<span class="caption-text">')
```
### 2 - sphinx/directives/__init__.py:

Start line: 269, End line: 323

```python
from sphinx.directives.code import CodeBlock, Highlight, LiteralInclude  # noqa
from sphinx.directives.other import (Acks, Author, Centered, Class, HList, Include,  # noqa
                                     Only, SeeAlso, TabularColumns, TocTree, VersionChange)
from sphinx.directives.patches import Figure, Meta  # noqa
from sphinx.domains.index import IndexDirective  # noqa

deprecated_alias('sphinx.directives',
                 {
                     'Highlight': Highlight,
                     'CodeBlock': CodeBlock,
                     'LiteralInclude': LiteralInclude,
                     'TocTree': TocTree,
                     'Author': Author,
                     'Index': IndexDirective,
                     'VersionChange': VersionChange,
                     'SeeAlso': SeeAlso,
                     'TabularColumns': TabularColumns,
                     'Centered': Centered,
                     'Acks': Acks,
                     'HList': HList,
                     'Only': Only,
                     'Include': Include,
                     'Class': Class,
                     'Figure': Figure,
                     'Meta': Meta,
                 },
                 RemovedInSphinx40Warning,
                 {
                     'Highlight': 'sphinx.directives.code.Highlight',
                     'CodeBlock': 'sphinx.directives.code.CodeBlock',
                     'LiteralInclude': 'sphinx.directives.code.LiteralInclude',
                     'TocTree': 'sphinx.directives.other.TocTree',
                     'Author': 'sphinx.directives.other.Author',
                     'Index': 'sphinx.directives.other.IndexDirective',
                     'VersionChange': 'sphinx.directives.other.VersionChange',
                     'SeeAlso': 'sphinx.directives.other.SeeAlso',
                     'TabularColumns': 'sphinx.directives.other.TabularColumns',
                     'Centered': 'sphinx.directives.other.Centered',
                     'Acks': 'sphinx.directives.other.Acks',
                     'HList': 'sphinx.directives.other.HList',
                     'Only': 'sphinx.directives.other.Only',
                     'Include': 'sphinx.directives.other.Include',
                     'Class': 'sphinx.directives.other.Class',
                     'Figure': 'sphinx.directives.patches.Figure',
                     'Meta': 'sphinx.directives.patches.Meta',
                 })

deprecated_alias('sphinx.directives',
                 {
                     'DescDirective': ObjectDescription,
                 },
                 RemovedInSphinx50Warning,
                 {
                     'DescDirective': 'sphinx.directives.ObjectDescription',
                 })
```
### 3 - doc/conf.py:

Start line: 59, End line: 127

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

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'requests': ('https://requests.readthedocs.io/en/master', None),
}

# Sphinx document translation with sphinx gettext feature uses these settings:
locale_dirs = ['locale/']
gettext_compact = False


# -- Extension interface -------------------------------------------------------

from sphinx import addnodes  # noqa

event_sig_re = re.compile(r'([a-zA-Z-]+)\s*\((.*)\)')
```
### 4 - sphinx/directives/patches.py:

Start line: 53, End line: 67

```python
class Meta(html.Meta, SphinxDirective):
    def run(self) -> List[Node]:
        result = super().run()
        for node in result:
            if (isinstance(node, nodes.pending) and
               isinstance(node.details['nodes'][0], html.MetaBody.meta)):
                meta = node.details['nodes'][0]
                meta.source = self.env.doc2path(self.env.docname)
                meta.line = self.lineno
                meta.rawcontent = meta['content']  # type: ignore

                # docutils' meta nodes aren't picklable because the class is nested
                meta.__class__ = addnodes.meta  # type: ignore

        return result
```
### 5 - sphinx/directives/patches.py:

Start line: 9, End line: 24

```python
from typing import Any, Dict, List, Tuple, cast

from docutils import nodes
from docutils.nodes import Node, make_id, system_message
from docutils.parsers.rst import directives
from docutils.parsers.rst.directives import html, images, tables

from sphinx import addnodes
from sphinx.directives import optional_int
from sphinx.domains.math import MathDomain
from sphinx.util.docutils import SphinxDirective
from sphinx.util.nodes import set_source_info

if False:
    # For type annotation
    from sphinx.application import Sphinx
```
### 6 - sphinx/writers/html.py:

Start line: 82, End line: 169

```python
class HTMLTranslator(SphinxTranslator, BaseTranslator):
    """
    Our custom HTML translator.
    """

    builder = None  # type: StandaloneHTMLBuilder

    def __init__(self, *args: Any) -> None:
        if isinstance(args[0], nodes.document) and isinstance(args[1], Builder):
            document, builder = args
        else:
            warnings.warn('The order of arguments for HTMLTranslator has been changed. '
                          'Please give "document" as 1st and "builder" as 2nd.',
                          RemovedInSphinx40Warning, stacklevel=2)
            builder, document = args
        super().__init__(document, builder)

        self.highlighter = self.builder.highlighter
        self.docnames = [self.builder.current_docname]  # for singlehtml builder
        self.manpages_url = self.config.manpages_url
        self.protect_literal_text = 0
        self.permalink_text = self.config.html_add_permalinks
        # support backwards-compatible setting to a bool
        if not isinstance(self.permalink_text, str):
            self.permalink_text = '¶' if self.permalink_text else ''
        self.permalink_text = self.encode(self.permalink_text)
        self.secnumber_suffix = self.config.html_secnumber_suffix
        self.param_separator = ''
        self.optional_param_level = 0
        self._table_row_index = 0
        self._fieldlist_row_index = 0
        self.required_params_left = 0

    def visit_start_of_file(self, node: Element) -> None:
        # only occurs in the single-file builder
        self.docnames.append(node['docname'])
        self.body.append('<span id="document-%s"></span>' % node['docname'])

    def depart_start_of_file(self, node: Element) -> None:
        self.docnames.pop()

    def visit_desc(self, node: Element) -> None:
        self.body.append(self.starttag(node, 'dl', CLASS=node['objtype']))

    def depart_desc(self, node: Element) -> None:
        self.body.append('</dl>\n\n')

    def visit_desc_signature(self, node: Element) -> None:
        # the id is set automatically
        self.body.append(self.starttag(node, 'dt'))

    def depart_desc_signature(self, node: Element) -> None:
        if not node.get('is_multiline'):
            self.add_permalink_ref(node, _('Permalink to this definition'))
        self.body.append('</dt>\n')

    def visit_desc_signature_line(self, node: Element) -> None:
        pass

    def depart_desc_signature_line(self, node: Element) -> None:
        if node.get('add_permalink'):
            # the permalink info is on the parent desc_signature node
            self.add_permalink_ref(node.parent, _('Permalink to this definition'))
        self.body.append('<br />')

    def visit_desc_addname(self, node: Element) -> None:
        self.body.append(self.starttag(node, 'code', '', CLASS='descclassname'))

    def depart_desc_addname(self, node: Element) -> None:
        self.body.append('</code>')

    def visit_desc_type(self, node: Element) -> None:
        pass

    def depart_desc_type(self, node: Element) -> None:
        pass

    def visit_desc_returns(self, node: Element) -> None:
        self.body.append(' &#x2192; ')

    def depart_desc_returns(self, node: Element) -> None:
        pass

    def visit_desc_name(self, node: Element) -> None:
        self.body.append(self.starttag(node, 'code', '', CLASS='descname'))

    def depart_desc_name(self, node: Element) -> None:
        self.body.append('</code>')
```
### 7 - sphinx/builders/html/__init__.py:

Start line: 1212, End line: 1288

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
    app.add_config_value('html_add_permalinks', '¶', 'html')
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
    app.add_config_value('html_codeblock_linenos_style', 'table', 'html',
                         ENUM('table', 'inline'))
    app.add_config_value('html_math_renderer', None, 'env')
    app.add_config_value('html4_writer', False, 'html')

    # events
    app.add_event('html-collect-pages')
    app.add_event('html-page-context')

    # event handlers
    app.connect('config-inited', convert_html_css_files, priority=800)
    app.connect('config-inited', convert_html_js_files, priority=800)
    app.connect('config-inited', validate_html_extra_path, priority=800)
    app.connect('config-inited', validate_html_static_path, priority=800)
    app.connect('config-inited', validate_html_logo, priority=800)
    app.connect('config-inited', validate_html_favicon, priority=800)
    app.connect('builder-inited', validate_math_renderer)
    app.connect('html-page-context', setup_js_tag_helper)

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
### 8 - sphinx/writers/html5.py:

Start line: 407, End line: 429

```python
class HTML5Translator(SphinxTranslator, BaseTranslator):

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
### 9 - sphinx/builders/html/__init__.py:

Start line: 11, End line: 68

```python
import html
import os
import posixpath
import re
import sys
import warnings
from os import path
from typing import IO, Any, Dict, Iterable, Iterator, List, Set, Tuple
from urllib.parse import quote

from docutils import nodes
from docutils.core import publish_parts
from docutils.frontend import OptionParser
from docutils.io import DocTreeInput, StringOutput
from docutils.nodes import Node
from docutils.utils import relative_path

from sphinx import __display_version__, package_dir
from sphinx.application import Sphinx
from sphinx.builders import Builder
from sphinx.config import ENUM, Config
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
from sphinx.util import logging, md5, progress_message, status_iterator
from sphinx.util.docutils import is_html5_writer_available, new_document
from sphinx.util.fileutil import copy_asset
from sphinx.util.i18n import format_date
from sphinx.util.inventory import InventoryFile
from sphinx.util.matching import DOTFILES, Matcher, patmatch
from sphinx.util.osutil import copyfile, ensuredir, os_path, relative_uri
from sphinx.util.tags import Tags
from sphinx.writers.html import HTMLTranslator, HTMLWriter

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
### 10 - sphinx/writers/html.py:

Start line: 171, End line: 231

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

    def visit_desc_content(self, node: Element) -> None:
        self.body.append(self.starttag(node, 'dd', ''))

    def depart_desc_content(self, node: Element) -> None:
        self.body.append('</dd>')

    def visit_versionmodified(self, node: Element) -> None:
        self.body.append(self.starttag(node, 'div', CLASS=node['type']))

    def depart_versionmodified(self, node: Element) -> None:
        self.body.append('</div>\n')

    # overwritten
```
### 11 - sphinx/writers/html.py:

Start line: 456, End line: 478

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
### 12 - sphinx/writers/html5.py:

Start line: 53, End line: 140

```python
class HTML5Translator(SphinxTranslator, BaseTranslator):
    """
    Our custom HTML translator.
    """

    builder = None  # type: StandaloneHTMLBuilder

    def __init__(self, *args: Any) -> None:
        if isinstance(args[0], nodes.document) and isinstance(args[1], Builder):
            document, builder = args
        else:
            warnings.warn('The order of arguments for HTML5Translator has been changed. '
                          'Please give "document" as 1st and "builder" as 2nd.',
                          RemovedInSphinx40Warning, stacklevel=2)
            builder, document = args
        super().__init__(document, builder)

        self.highlighter = self.builder.highlighter
        self.docnames = [self.builder.current_docname]  # for singlehtml builder
        self.manpages_url = self.config.manpages_url
        self.protect_literal_text = 0
        self.permalink_text = self.config.html_add_permalinks
        # support backwards-compatible setting to a bool
        if not isinstance(self.permalink_text, str):
            self.permalink_text = '¶' if self.permalink_text else ''
        self.permalink_text = self.encode(self.permalink_text)
        self.secnumber_suffix = self.config.html_secnumber_suffix
        self.param_separator = ''
        self.optional_param_level = 0
        self._table_row_index = 0
        self._fieldlist_row_index = 0
        self.required_params_left = 0

    def visit_start_of_file(self, node: Element) -> None:
        # only occurs in the single-file builder
        self.docnames.append(node['docname'])
        self.body.append('<span id="document-%s"></span>' % node['docname'])

    def depart_start_of_file(self, node: Element) -> None:
        self.docnames.pop()

    def visit_desc(self, node: Element) -> None:
        self.body.append(self.starttag(node, 'dl', CLASS=node['objtype']))

    def depart_desc(self, node: Element) -> None:
        self.body.append('</dl>\n\n')

    def visit_desc_signature(self, node: Element) -> None:
        # the id is set automatically
        self.body.append(self.starttag(node, 'dt'))

    def depart_desc_signature(self, node: Element) -> None:
        if not node.get('is_multiline'):
            self.add_permalink_ref(node, _('Permalink to this definition'))
        self.body.append('</dt>\n')

    def visit_desc_signature_line(self, node: Element) -> None:
        pass

    def depart_desc_signature_line(self, node: Element) -> None:
        if node.get('add_permalink'):
            # the permalink info is on the parent desc_signature node
            self.add_permalink_ref(node.parent, _('Permalink to this definition'))
        self.body.append('<br />')

    def visit_desc_addname(self, node: Element) -> None:
        self.body.append(self.starttag(node, 'code', '', CLASS='sig-prename descclassname'))

    def depart_desc_addname(self, node: Element) -> None:
        self.body.append('</code>')

    def visit_desc_type(self, node: Element) -> None:
        pass

    def depart_desc_type(self, node: Element) -> None:
        pass

    def visit_desc_returns(self, node: Element) -> None:
        self.body.append(' &#x2192; ')

    def depart_desc_returns(self, node: Element) -> None:
        pass

    def visit_desc_name(self, node: Element) -> None:
        self.body.append(self.starttag(node, 'code', '', CLASS='sig-name descname'))

    def depart_desc_name(self, node: Element) -> None:
        self.body.append('</code>')
```
### 14 - sphinx/writers/html5.py:

Start line: 142, End line: 202

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

    def visit_desc_content(self, node: Element) -> None:
        self.body.append(self.starttag(node, 'dd', ''))

    def depart_desc_content(self, node: Element) -> None:
        self.body.append('</dd>')

    def visit_versionmodified(self, node: Element) -> None:
        self.body.append(self.starttag(node, 'div', CLASS=node['type']))

    def depart_versionmodified(self, node: Element) -> None:
        self.body.append('</div>\n')

    # overwritten
```
### 16 - sphinx/writers/html5.py:

Start line: 629, End line: 715

```python
class HTML5Translator(SphinxTranslator, BaseTranslator):

    def visit_note(self, node: Element) -> None:
        self.visit_admonition(node, 'note')

    def depart_note(self, node: Element) -> None:
        self.depart_admonition(node)

    def visit_warning(self, node: Element) -> None:
        self.visit_admonition(node, 'warning')

    def depart_warning(self, node: Element) -> None:
        self.depart_admonition(node)

    def visit_attention(self, node: Element) -> None:
        self.visit_admonition(node, 'attention')

    def depart_attention(self, node: Element) -> None:
        self.depart_admonition(node)

    def visit_caution(self, node: Element) -> None:
        self.visit_admonition(node, 'caution')

    def depart_caution(self, node: Element) -> None:
        self.depart_admonition(node)

    def visit_danger(self, node: Element) -> None:
        self.visit_admonition(node, 'danger')

    def depart_danger(self, node: Element) -> None:
        self.depart_admonition(node)

    def visit_error(self, node: Element) -> None:
        self.visit_admonition(node, 'error')

    def depart_error(self, node: Element) -> None:
        self.depart_admonition(node)

    def visit_hint(self, node: Element) -> None:
        self.visit_admonition(node, 'hint')

    def depart_hint(self, node: Element) -> None:
        self.depart_admonition(node)

    def visit_important(self, node: Element) -> None:
        self.visit_admonition(node, 'important')

    def depart_important(self, node: Element) -> None:
        self.depart_admonition(node)

    def visit_tip(self, node: Element) -> None:
        self.visit_admonition(node, 'tip')

    def depart_tip(self, node: Element) -> None:
        self.depart_admonition(node)

    def visit_literal_emphasis(self, node: Element) -> None:
        return self.visit_emphasis(node)

    def depart_literal_emphasis(self, node: Element) -> None:
        return self.depart_emphasis(node)

    def visit_literal_strong(self, node: Element) -> None:
        return self.visit_strong(node)

    def depart_literal_strong(self, node: Element) -> None:
        return self.depart_strong(node)

    def visit_abbreviation(self, node: Element) -> None:
        attrs = {}
        if node.hasattr('explanation'):
            attrs['title'] = node['explanation']
        self.body.append(self.starttag(node, 'abbr', '', **attrs))

    def depart_abbreviation(self, node: Element) -> None:
        self.body.append('</abbr>')

    def visit_manpage(self, node: Element) -> None:
        self.visit_literal_emphasis(node)
        if self.manpages_url:
            node['refuri'] = self.manpages_url.format(**node.attributes)
            self.visit_reference(node)

    def depart_manpage(self, node: Element) -> None:
        if self.manpages_url:
            self.depart_reference(node)
        self.depart_literal_emphasis(node)

    # overwritten to add even/odd classes
```
### 19 - sphinx/writers/html.py:

Start line: 693, End line: 792

```python
class HTMLTranslator(SphinxTranslator, BaseTranslator):

    def visit_note(self, node: Element) -> None:
        self.visit_admonition(node, 'note')

    def depart_note(self, node: Element) -> None:
        self.depart_admonition(node)

    def visit_warning(self, node: Element) -> None:
        self.visit_admonition(node, 'warning')

    def depart_warning(self, node: Element) -> None:
        self.depart_admonition(node)

    def visit_attention(self, node: Element) -> None:
        self.visit_admonition(node, 'attention')

    def depart_attention(self, node: Element) -> None:
        self.depart_admonition(node)

    def visit_caution(self, node: Element) -> None:
        self.visit_admonition(node, 'caution')

    def depart_caution(self, node: Element) -> None:
        self.depart_admonition(node)

    def visit_danger(self, node: Element) -> None:
        self.visit_admonition(node, 'danger')

    def depart_danger(self, node: Element) -> None:
        self.depart_admonition(node)

    def visit_error(self, node: Element) -> None:
        self.visit_admonition(node, 'error')

    def depart_error(self, node: Element) -> None:
        self.depart_admonition(node)

    def visit_hint(self, node: Element) -> None:
        self.visit_admonition(node, 'hint')

    def depart_hint(self, node: Element) -> None:
        self.depart_admonition(node)

    def visit_important(self, node: Element) -> None:
        self.visit_admonition(node, 'important')

    def depart_important(self, node: Element) -> None:
        self.depart_admonition(node)

    def visit_tip(self, node: Element) -> None:
        self.visit_admonition(node, 'tip')

    def depart_tip(self, node: Element) -> None:
        self.depart_admonition(node)

    def visit_literal_emphasis(self, node: Element) -> None:
        return self.visit_emphasis(node)

    def depart_literal_emphasis(self, node: Element) -> None:
        return self.depart_emphasis(node)

    def visit_literal_strong(self, node: Element) -> None:
        return self.visit_strong(node)

    def depart_literal_strong(self, node: Element) -> None:
        return self.depart_strong(node)

    def visit_abbreviation(self, node: Element) -> None:
        attrs = {}
        if node.hasattr('explanation'):
            attrs['title'] = node['explanation']
        self.body.append(self.starttag(node, 'abbr', '', **attrs))

    def depart_abbreviation(self, node: Element) -> None:
        self.body.append('</abbr>')

    def visit_manpage(self, node: Element) -> None:
        self.visit_literal_emphasis(node)
        if self.manpages_url:
            node['refuri'] = self.manpages_url.format(**node.attributes)
            self.visit_reference(node)

    def depart_manpage(self, node: Element) -> None:
        if self.manpages_url:
            self.depart_reference(node)
        self.depart_literal_emphasis(node)

    # overwritten to add even/odd classes

    def visit_table(self, node: Element) -> None:
        self._table_row_index = 0
        return super().visit_table(node)

    def visit_row(self, node: Element) -> None:
        self._table_row_index += 1
        if self._table_row_index % 2 == 0:
            node['classes'].append('row-even')
        else:
            node['classes'].append('row-odd')
        self.body.append(self.starttag(node, 'tr', ''))
        node.column = 0  # type: ignore
```
### 24 - sphinx/writers/html.py:

Start line: 354, End line: 409

```python
class HTMLTranslator(SphinxTranslator, BaseTranslator):

    # overwritten
    def visit_bullet_list(self, node: Element) -> None:
        if len(node) == 1 and isinstance(node[0], addnodes.toctree):
            # avoid emitting empty <ul></ul>
            raise nodes.SkipNode
        self.generate_targets_for_listing(node)
        super().visit_bullet_list(node)

    # overwritten
    def visit_enumerated_list(self, node: Element) -> None:
        self.generate_targets_for_listing(node)
        super().visit_enumerated_list(node)

    # overwritten
    def visit_definition(self, node: Element) -> None:
        # don't insert </dt> here.
        self.body.append(self.starttag(node, 'dd', ''))

    # overwritten
    def depart_definition(self, node: Element) -> None:
        self.body.append('</dd>\n')

    # overwritten
    def visit_classifier(self, node: Element) -> None:
        self.body.append(self.starttag(node, 'span', '', CLASS='classifier'))

    # overwritten
    def depart_classifier(self, node: Element) -> None:
        self.body.append('</span>')

        next_node = node.next_node(descend=False, siblings=True)  # type: Node
        if not isinstance(next_node, nodes.classifier):
            # close `<dt>` tag at the tail of classifiers
            self.body.append('</dt>')

    # overwritten
    def visit_term(self, node: Element) -> None:
        self.body.append(self.starttag(node, 'dt', ''))

    # overwritten
    def depart_term(self, node: Element) -> None:
        next_node = node.next_node(descend=False, siblings=True)  # type: Node
        if isinstance(next_node, nodes.classifier):
            # Leave the end tag to `self.depart_classifier()`, in case
            # there's a classifier.
            pass
        else:
            self.body.append('</dt>')

    # overwritten
    def visit_title(self, node: Element) -> None:
        super().visit_title(node)
        self.add_secnumber(node)
        self.add_fignumber(node.parent)
        if isinstance(node.parent, nodes.table):
            self.body.append('<span class="caption-text">')
```
### 25 - sphinx/writers/html.py:

Start line: 625, End line: 671

```python
class HTMLTranslator(SphinxTranslator, BaseTranslator):

    # overwritten
    def depart_image(self, node: Element) -> None:
        if node['uri'].lower().endswith(('svg', 'svgz')):
            pass
        else:
            super().depart_image(node)

    def visit_toctree(self, node: Element) -> None:
        # this only happens when formatting a toc from env.tocs -- in this
        # case we don't want to include the subtree
        raise nodes.SkipNode

    def visit_index(self, node: Element) -> None:
        raise nodes.SkipNode

    def visit_tabular_col_spec(self, node: Element) -> None:
        raise nodes.SkipNode

    def visit_glossary(self, node: Element) -> None:
        pass

    def depart_glossary(self, node: Element) -> None:
        pass

    def visit_acks(self, node: Element) -> None:
        pass

    def depart_acks(self, node: Element) -> None:
        pass

    def visit_hlist(self, node: Element) -> None:
        self.body.append('<table class="hlist"><tr>')

    def depart_hlist(self, node: Element) -> None:
        self.body.append('</tr></table>\n')

    def visit_hlistcol(self, node: Element) -> None:
        self.body.append('<td>')

    def depart_hlistcol(self, node: Element) -> None:
        self.body.append('</td>')

    def visit_option_group(self, node: Element) -> None:
        super().visit_option_group(node)
        self.context[-2] = self.context[-2].replace('&nbsp;', '&#160;')

    # overwritten
```
### 29 - sphinx/writers/html.py:

Start line: 794, End line: 841

```python
class HTMLTranslator(SphinxTranslator, BaseTranslator):

    def visit_entry(self, node: Element) -> None:
        super().visit_entry(node)
        if self.body[-1] == '&nbsp;':
            self.body[-1] = '&#160;'

    def visit_field_list(self, node: Element) -> None:
        self._fieldlist_row_index = 0
        return super().visit_field_list(node)

    def visit_field(self, node: Element) -> None:
        self._fieldlist_row_index += 1
        if self._fieldlist_row_index % 2 == 0:
            node['classes'].append('field-even')
        else:
            node['classes'].append('field-odd')
        self.body.append(self.starttag(node, 'tr', '', CLASS='field'))

    def visit_field_name(self, node: Element) -> None:
        context_count = len(self.context)
        super().visit_field_name(node)
        if context_count != len(self.context):
            self.context[-1] = self.context[-1].replace('&nbsp;', '&#160;')

    def visit_math(self, node: Element, math_env: str = '') -> None:
        name = self.builder.math_renderer_name
        visit, _ = self.builder.app.registry.html_inline_math_renderers[name]
        visit(self, node)

    def depart_math(self, node: Element, math_env: str = '') -> None:
        name = self.builder.math_renderer_name
        _, depart = self.builder.app.registry.html_inline_math_renderers[name]
        if depart:
            depart(self, node)

    def visit_math_block(self, node: Element, math_env: str = '') -> None:
        name = self.builder.math_renderer_name
        visit, _ = self.builder.app.registry.html_block_math_renderers[name]
        visit(self, node)

    def depart_math_block(self, node: Element, math_env: str = '') -> None:
        name = self.builder.math_renderer_name
        _, depart = self.builder.app.registry.html_block_math_renderers[name]
        if depart:
            depart(self, node)

    def unknown_visit(self, node: Node) -> None:
        raise NotImplementedError('Unknown node: ' + node.__class__.__name__)
```
### 36 - sphinx/writers/html.py:

Start line: 335, End line: 352

```python
class HTMLTranslator(SphinxTranslator, BaseTranslator):

    def add_permalink_ref(self, node: Element, title: str) -> None:
        if node['ids'] and self.permalink_text and self.builder.add_permalinks:
            format = '<a class="headerlink" href="#%s" title="%s">%s</a>'
            self.body.append(format % (node['ids'][0], title, self.permalink_text))

    def generate_targets_for_listing(self, node: Element) -> None:
        """Generate hyperlink targets for listings.

        Original visit_bullet_list(), visit_definition_list() and visit_enumerated_list()
        generates hyperlink targets inside listing tags (<ul>, <ol> and <dl>) if multiple
        IDs are assigned to listings.  That is invalid DOM structure.
        (This is a bug of docutils <= 0.12)

        This exports hyperlink targets before listings to make valid DOM structure.
        """
        for id in node['ids'][1:]:
            self.body.append('<span id="%s"></span>' % id)
            node['ids'].remove(id)
```
### 39 - sphinx/writers/html5.py:

Start line: 565, End line: 607

```python
class HTML5Translator(SphinxTranslator, BaseTranslator):

    # overwritten
    def depart_image(self, node: Element) -> None:
        if node['uri'].lower().endswith(('svg', 'svgz')):
            pass
        else:
            super().depart_image(node)

    def visit_toctree(self, node: Element) -> None:
        # this only happens when formatting a toc from env.tocs -- in this
        # case we don't want to include the subtree
        raise nodes.SkipNode

    def visit_index(self, node: Element) -> None:
        raise nodes.SkipNode

    def visit_tabular_col_spec(self, node: Element) -> None:
        raise nodes.SkipNode

    def visit_glossary(self, node: Element) -> None:
        pass

    def depart_glossary(self, node: Element) -> None:
        pass

    def visit_acks(self, node: Element) -> None:
        pass

    def depart_acks(self, node: Element) -> None:
        pass

    def visit_hlist(self, node: Element) -> None:
        self.body.append('<table class="hlist"><tr>')

    def depart_hlist(self, node: Element) -> None:
        self.body.append('</tr></table>\n')

    def visit_hlistcol(self, node: Element) -> None:
        self.body.append('<td>')

    def depart_hlistcol(self, node: Element) -> None:
        self.body.append('</td>')

    # overwritten
```
### 43 - sphinx/writers/html5.py:

Start line: 231, End line: 252

```python
class HTML5Translator(SphinxTranslator, BaseTranslator):

    def visit_number_reference(self, node: Element) -> None:
        self.visit_reference(node)

    def depart_number_reference(self, node: Element) -> None:
        self.depart_reference(node)

    # overwritten -- we don't want source comments to show up in the HTML
    def visit_comment(self, node: Element) -> None:  # type: ignore
        raise nodes.SkipNode

    # overwritten
    def visit_admonition(self, node: Element, name: str = '') -> None:
        self.body.append(self.starttag(
            node, 'div', CLASS=('admonition ' + name)))
        if name:
            node.insert(0, nodes.title(name, admonitionlabels[name]))

    def visit_seealso(self, node: Element) -> None:
        self.visit_admonition(node, 'seealso')

    def depart_seealso(self, node: Element) -> None:
        self.depart_admonition(node)
```
### 46 - sphinx/writers/html.py:

Start line: 260, End line: 282

```python
class HTMLTranslator(SphinxTranslator, BaseTranslator):

    def visit_number_reference(self, node: Element) -> None:
        self.visit_reference(node)

    def depart_number_reference(self, node: Element) -> None:
        self.depart_reference(node)

    # overwritten -- we don't want source comments to show up in the HTML
    def visit_comment(self, node: Element) -> None:  # type: ignore
        raise nodes.SkipNode

    # overwritten
    def visit_admonition(self, node: Element, name: str = '') -> None:
        self.body.append(self.starttag(
            node, 'div', CLASS=('admonition ' + name)))
        if name:
            node.insert(0, nodes.title(name, admonitionlabels[name]))
        self.set_first_last(node)

    def visit_seealso(self, node: Element) -> None:
        self.visit_admonition(node, 'seealso')

    def depart_seealso(self, node: Element) -> None:
        self.depart_admonition(node)
```
### 49 - sphinx/writers/html5.py:

Start line: 203, End line: 229

```python
class HTML5Translator(SphinxTranslator, BaseTranslator):
    def visit_reference(self, node: Element) -> None:
        atts = {'class': 'reference'}
        if node.get('internal') or 'refuri' not in node:
            atts['class'] += ' internal'
        else:
            atts['class'] += ' external'
        if 'refuri' in node:
            atts['href'] = node['refuri'] or '#'
            if self.settings.cloak_email_addresses and atts['href'].startswith('mailto:'):
                atts['href'] = self.cloak_mailto(atts['href'])
                self.in_mailto = True
        else:
            assert 'refid' in node, \
                   'References must have "refuri" or "refid" attribute.'
            atts['href'] = '#' + node['refid']
        if not isinstance(node.parent, nodes.TextElement):
            assert len(node) == 1 and isinstance(node[0], nodes.image)
            atts['class'] += ' image-reference'
        if 'reftitle' in node:
            atts['title'] = node['reftitle']
        if 'target' in node:
            atts['target'] = node['target']
        self.body.append(self.starttag(node, 'a', '', **atts))

        if node.get('secnumber'):
            self.body.append(('%s' + self.secnumber_suffix) %
                             '.'.join(map(str, node['secnumber'])))
```
### 53 - sphinx/writers/html.py:

Start line: 232, End line: 258

```python
class HTMLTranslator(SphinxTranslator, BaseTranslator):
    def visit_reference(self, node: Element) -> None:
        atts = {'class': 'reference'}
        if node.get('internal') or 'refuri' not in node:
            atts['class'] += ' internal'
        else:
            atts['class'] += ' external'
        if 'refuri' in node:
            atts['href'] = node['refuri'] or '#'
            if self.settings.cloak_email_addresses and atts['href'].startswith('mailto:'):
                atts['href'] = self.cloak_mailto(atts['href'])
                self.in_mailto = True
        else:
            assert 'refid' in node, \
                   'References must have "refuri" or "refid" attribute.'
            atts['href'] = '#' + node['refid']
        if not isinstance(node.parent, nodes.TextElement):
            assert len(node) == 1 and isinstance(node[0], nodes.image)
            atts['class'] += ' image-reference'
        if 'reftitle' in node:
            atts['title'] = node['reftitle']
        if 'target' in node:
            atts['target'] = node['target']
        self.body.append(self.starttag(node, 'a', '', **atts))

        if node.get('secnumber'):
            self.body.append(('%s' + self.secnumber_suffix) %
                             '.'.join(map(str, node['secnumber'])))
```
### 55 - sphinx/writers/html.py:

Start line: 480, End line: 505

```python
class HTMLTranslator(SphinxTranslator, BaseTranslator):

    def visit_doctest_block(self, node: Element) -> None:
        self.visit_literal_block(node)

    # overwritten to add the <div> (for XHTML compliance)
    def visit_block_quote(self, node: Element) -> None:
        self.body.append(self.starttag(node, 'blockquote') + '<div>')

    def depart_block_quote(self, node: Element) -> None:
        self.body.append('</div></blockquote>\n')

    # overwritten
    def visit_literal(self, node: Element) -> None:
        if 'kbd' in node['classes']:
            self.body.append(self.starttag(node, 'kbd', '',
                                           CLASS='docutils literal notranslate'))
        else:
            self.body.append(self.starttag(node, 'code', '',
                                           CLASS='docutils literal notranslate'))
            self.protect_literal_text += 1

    def depart_literal(self, node: Element) -> None:
        if 'kbd' in node['classes']:
            self.body.append('</kbd>')
        else:
            self.protect_literal_text -= 1
            self.body.append('</code>')
```
### 58 - sphinx/writers/html5.py:

Start line: 608, End line: 627

```python
class HTML5Translator(SphinxTranslator, BaseTranslator):
    def visit_Text(self, node: Text) -> None:
        text = node.astext()
        encoded = self.encode(text)
        if self.protect_literal_text:
            # moved here from base class's visit_literal to support
            # more formatting in literal nodes
            for token in self.words_and_spaces.findall(encoded):
                if token.strip():
                    # protect literal text from line wrapping
                    self.body.append('<span class="pre">%s</span>' % token)
                elif token in ' \n':
                    # allow breaks at whitespace
                    self.body.append(token)
                else:
                    # protect runs of multiple spaces; the last one can wrap
                    self.body.append('&#160;' * (len(token) - 1) + ' ')
        else:
            if self.in_mailto and self.settings.cloak_email_addresses:
                encoded = self.cloak_email(encoded)
            self.body.append(encoded)
```
### 60 - sphinx/writers/html5.py:

Start line: 431, End line: 456

```python
class HTML5Translator(SphinxTranslator, BaseTranslator):

    def visit_doctest_block(self, node: Element) -> None:
        self.visit_literal_block(node)

    # overwritten to add the <div> (for XHTML compliance)
    def visit_block_quote(self, node: Element) -> None:
        self.body.append(self.starttag(node, 'blockquote') + '<div>')

    def depart_block_quote(self, node: Element) -> None:
        self.body.append('</div></blockquote>\n')

    # overwritten
    def visit_literal(self, node: Element) -> None:
        if 'kbd' in node['classes']:
            self.body.append(self.starttag(node, 'kbd', '',
                                           CLASS='docutils literal notranslate'))
        else:
            self.body.append(self.starttag(node, 'code', '',
                                           CLASS='docutils literal notranslate'))
            self.protect_literal_text += 1

    def depart_literal(self, node: Element) -> None:
        if 'kbd' in node['classes']:
            self.body.append('</kbd>')
        else:
            self.protect_literal_text -= 1
            self.body.append('</code>')
```
### 63 - sphinx/writers/html5.py:

Start line: 362, End line: 383

```python
class HTML5Translator(SphinxTranslator, BaseTranslator):

    def depart_title(self, node: Element) -> None:
        close_tag = self.context[-1]
        if (self.permalink_text and self.builder.add_permalinks and
           node.parent.hasattr('ids') and node.parent['ids']):
            # add permalink anchor
            if close_tag.startswith('</h'):
                self.add_permalink_ref(node.parent, _('Permalink to this headline'))
            elif close_tag.startswith('</a></h'):
                self.body.append('</a><a class="headerlink" href="#%s" ' %
                                 node.parent['ids'][0] +
                                 'title="%s">%s' % (
                                     _('Permalink to this headline'),
                                     self.permalink_text))
            elif isinstance(node.parent, nodes.table):
                self.body.append('</span>')
                self.add_permalink_ref(node.parent, _('Permalink to this table'))
        elif isinstance(node.parent, nodes.table):
            self.body.append('</span>')

        super().depart_title(node)

    # overwritten
```
### 65 - sphinx/writers/html.py:

Start line: 55, End line: 79

```python
class HTMLWriter(Writer):

    # override embed-stylesheet default value to 0.
    settings_spec = copy.deepcopy(Writer.settings_spec)
    for _setting in settings_spec[2]:
        if '--embed-stylesheet' in _setting[1]:
            _setting[2]['default'] = 0

    def __init__(self, builder: "StandaloneHTMLBuilder") -> None:
        super().__init__()
        self.builder = builder

    def translate(self) -> None:
        # sadly, this is mostly copied from parent class
        visitor = self.builder.create_translator(self.document, self.builder)
        self.visitor = cast(HTMLTranslator, visitor)
        self.document.walkabout(visitor)
        self.output = self.visitor.astext()
        for attr in ('head_prefix', 'stylesheet', 'head', 'body_prefix',
                     'body_pre_docinfo', 'docinfo', 'body', 'fragment',
                     'body_suffix', 'meta', 'title', 'subtitle', 'header',
                     'footer', 'html_prolog', 'html_head', 'html_title',
                     'html_subtitle', 'html_body', ):
            setattr(self, attr, getattr(visitor, attr, None))
        self.clean_meta = ''.join(self.visitor.meta[2:])
```
### 66 - sphinx/writers/html.py:

Start line: 560, End line: 582

```python
class HTMLTranslator(SphinxTranslator, BaseTranslator):

    def visit_download_reference(self, node: Element) -> None:
        atts = {'class': 'reference download',
                'download': ''}

        if not self.builder.download_support:
            self.context.append('')
        elif 'refuri' in node:
            atts['class'] += ' external'
            atts['href'] = node['refuri']
            self.body.append(self.starttag(node, 'a', '', **atts))
            self.context.append('</a>')
        elif 'filename' in node:
            atts['class'] += ' internal'
            atts['href'] = posixpath.join(self.builder.dlpath, node['filename'])
            self.body.append(self.starttag(node, 'a', '', **atts))
            self.context.append('</a>')
        else:
            self.context.append('')

    def depart_download_reference(self, node: Element) -> None:
        self.body.append(self.context.pop())

    # overwritten
```
### 70 - sphinx/writers/html.py:

Start line: 411, End line: 432

```python
class HTMLTranslator(SphinxTranslator, BaseTranslator):

    def depart_title(self, node: Element) -> None:
        close_tag = self.context[-1]
        if (self.permalink_text and self.builder.add_permalinks and
           node.parent.hasattr('ids') and node.parent['ids']):
            # add permalink anchor
            if close_tag.startswith('</h'):
                self.add_permalink_ref(node.parent, _('Permalink to this headline'))
            elif close_tag.startswith('</a></h'):
                self.body.append('</a><a class="headerlink" href="#%s" ' %
                                 node.parent['ids'][0] +
                                 'title="%s">%s' % (
                                     _('Permalink to this headline'),
                                     self.permalink_text))
            elif isinstance(node.parent, nodes.table):
                self.body.append('</span>')
                self.add_permalink_ref(node.parent, _('Permalink to this table'))
        elif isinstance(node.parent, nodes.table):
            self.body.append('</span>')

        super().depart_title(node)

    # overwritten
```
### 71 - sphinx/writers/html5.py:

Start line: 500, End line: 522

```python
class HTML5Translator(SphinxTranslator, BaseTranslator):

    def visit_download_reference(self, node: Element) -> None:
        atts = {'class': 'reference download',
                'download': ''}

        if not self.builder.download_support:
            self.context.append('')
        elif 'refuri' in node:
            atts['class'] += ' external'
            atts['href'] = node['refuri']
            self.body.append(self.starttag(node, 'a', '', **atts))
            self.context.append('</a>')
        elif 'filename' in node:
            atts['class'] += ' internal'
            atts['href'] = posixpath.join(self.builder.dlpath, node['filename'])
            self.body.append(self.starttag(node, 'a', '', **atts))
            self.context.append('</a>')
        else:
            self.context.append('')

    def depart_download_reference(self, node: Element) -> None:
        self.body.append(self.context.pop())

    # overwritten
```
### 73 - sphinx/writers/html5.py:

Start line: 745, End line: 789

```python
class HTML5Translator(SphinxTranslator, BaseTranslator):

    def visit_row(self, node: Element) -> None:
        self._table_row_index += 1
        if self._table_row_index % 2 == 0:
            node['classes'].append('row-even')
        else:
            node['classes'].append('row-odd')
        self.body.append(self.starttag(node, 'tr', ''))
        node.column = 0  # type: ignore

    def visit_field_list(self, node: Element) -> None:
        self._fieldlist_row_index = 0
        return super().visit_field_list(node)

    def visit_field(self, node: Element) -> None:
        self._fieldlist_row_index += 1
        if self._fieldlist_row_index % 2 == 0:
            node['classes'].append('field-even')
        else:
            node['classes'].append('field-odd')

    def visit_math(self, node: Element, math_env: str = '') -> None:
        name = self.builder.math_renderer_name
        visit, _ = self.builder.app.registry.html_inline_math_renderers[name]
        visit(self, node)

    def depart_math(self, node: Element, math_env: str = '') -> None:
        name = self.builder.math_renderer_name
        _, depart = self.builder.app.registry.html_inline_math_renderers[name]
        if depart:
            depart(self, node)

    def visit_math_block(self, node: Element, math_env: str = '') -> None:
        name = self.builder.math_renderer_name
        visit, _ = self.builder.app.registry.html_block_math_renderers[name]
        visit(self, node)

    def depart_math_block(self, node: Element, math_env: str = '') -> None:
        name = self.builder.math_renderer_name
        _, depart = self.builder.app.registry.html_block_math_renderers[name]
        if depart:
            depart(self, node)

    def unknown_visit(self, node: Node) -> None:
        raise NotImplementedError('Unknown node: ' + node.__class__.__name__)
```
### 79 - sphinx/writers/html.py:

Start line: 672, End line: 691

```python
class HTMLTranslator(SphinxTranslator, BaseTranslator):
    def visit_Text(self, node: Text) -> None:
        text = node.astext()
        encoded = self.encode(text)
        if self.protect_literal_text:
            # moved here from base class's visit_literal to support
            # more formatting in literal nodes
            for token in self.words_and_spaces.findall(encoded):
                if token.strip():
                    # protect literal text from line wrapping
                    self.body.append('<span class="pre">%s</span>' % token)
                elif token in ' \n':
                    # allow breaks at whitespace
                    self.body.append(token)
                else:
                    # protect runs of multiple spaces; the last one can wrap
                    self.body.append('&#160;' * (len(token) - 1) + ' ')
        else:
            if self.in_mailto and self.settings.cloak_email_addresses:
                encoded = self.cloak_email(encoded)
            self.body.append(encoded)
```
### 81 - sphinx/writers/html.py:

Start line: 527, End line: 558

```python
class HTMLTranslator(SphinxTranslator, BaseTranslator):

    def depart_productionlist(self, node: Element) -> None:
        pass

    def visit_production(self, node: Element) -> None:
        pass

    def depart_production(self, node: Element) -> None:
        pass

    def visit_centered(self, node: Element) -> None:
        self.body.append(self.starttag(node, 'p', CLASS="centered") +
                         '<strong>')

    def depart_centered(self, node: Element) -> None:
        self.body.append('</strong></p>')

    # overwritten
    def should_be_compact_paragraph(self, node: Node) -> bool:
        """Determine if the <p> tags around paragraph can be omitted."""
        if isinstance(node.parent, addnodes.desc_content):
            # Never compact desc_content items.
            return False
        if isinstance(node.parent, addnodes.versionmodified):
            # Never compact versionmodified nodes.
            return False
        return super().should_be_compact_paragraph(node)

    def visit_compact_paragraph(self, node: Element) -> None:
        pass

    def depart_compact_paragraph(self, node: Element) -> None:
        pass
```
### 84 - sphinx/builders/html/__init__.py:

Start line: 1194, End line: 1212

```python
def validate_html_logo(app: Sphinx, config: Config) -> None:
    """Check html_logo setting."""
    if config.html_logo and not path.isfile(path.join(app.confdir, config.html_logo)):
        logger.warning(__('logo file %r does not exist'), config.html_logo)
        config.html_logo = None  # type: ignore


def validate_html_favicon(app: Sphinx, config: Config) -> None:
    """Check html_favicon setting."""
    if config.html_favicon and not path.isfile(path.join(app.confdir, config.html_favicon)):
        logger.warning(__('favicon file %r does not exist'), config.html_favicon)
        config.html_favicon = None  # type: ignore


# for compatibility
import sphinxcontrib.serializinghtml  # NOQA

import sphinx.builders.dirhtml  # NOQA
import sphinx.builders.singlehtml
```
### 97 - sphinx/writers/html5.py:

Start line: 478, End line: 498

```python
class HTML5Translator(SphinxTranslator, BaseTranslator):

    def depart_productionlist(self, node: Element) -> None:
        pass

    def visit_production(self, node: Element) -> None:
        pass

    def depart_production(self, node: Element) -> None:
        pass

    def visit_centered(self, node: Element) -> None:
        self.body.append(self.starttag(node, 'p', CLASS="centered") +
                         '<strong>')

    def depart_centered(self, node: Element) -> None:
        self.body.append('</strong></p>')

    def visit_compact_paragraph(self, node: Element) -> None:
        pass

    def depart_compact_paragraph(self, node: Element) -> None:
        pass
```
### 106 - sphinx/builders/html/__init__.py:

Start line: 915, End line: 968

```python
class StandaloneHTMLBuilder(Builder):

    def add_sidebars(self, pagename: str, ctx: Dict) -> None:
        def has_wildcard(pattern: str) -> bool:
            return any(char in pattern for char in '*?[')

        sidebars = None
        matched = None
        customsidebar = None

        # default sidebars settings for selected theme
        if self.theme.name == 'alabaster':
            # provide default settings for alabaster (for compatibility)
            # Note: this will be removed before Sphinx-2.0
            try:
                # get default sidebars settings from alabaster (if defined)
                theme_default_sidebars = self.theme.config.get('theme', 'sidebars')
                if theme_default_sidebars:
                    sidebars = [name.strip() for name in theme_default_sidebars.split(',')]
            except Exception:
                # fallback to better default settings
                sidebars = ['about.html', 'navigation.html', 'relations.html',
                            'searchbox.html', 'donate.html']
        else:
            theme_default_sidebars = self.theme.get_config('theme', 'sidebars', None)
            if theme_default_sidebars:
                sidebars = [name.strip() for name in theme_default_sidebars.split(',')]

        # user sidebar settings
        html_sidebars = self.get_builder_config('sidebars', 'html')
        for pattern, patsidebars in html_sidebars.items():
            if patmatch(pagename, pattern):
                if matched:
                    if has_wildcard(pattern):
                        # warn if both patterns contain wildcards
                        if has_wildcard(matched):
                            logger.warning(__('page %s matches two patterns in '
                                              'html_sidebars: %r and %r'),
                                           pagename, matched, pattern)
                        # else the already matched pattern is more specific
                        # than the present one, because it contains no wildcard
                        continue
                matched = pattern
                sidebars = patsidebars

        if sidebars is None:
            # keep defaults
            pass

        ctx['sidebars'] = sidebars
        ctx['customsidebar'] = customsidebar

    # --------- these are overwritten by the serialization builder

    def get_target_uri(self, docname: str, typ: str = None) -> str:
        return quote(docname) + self.link_suffix
```
### 116 - sphinx/writers/html5.py:

Start line: 280, End line: 304

```python
class HTML5Translator(SphinxTranslator, BaseTranslator):

    def add_fignumber(self, node: Element) -> None:
        def append_fignumber(figtype: str, figure_id: str) -> None:
            if self.builder.name == 'singlehtml':
                key = "%s/%s" % (self.docnames[-1], figtype)
            else:
                key = figtype

            if figure_id in self.builder.fignumbers.get(key, {}):
                self.body.append('<span class="caption-number">')
                prefix = self.config.numfig_format.get(figtype)
                if prefix is None:
                    msg = __('numfig_format is not defined for %s') % figtype
                    logger.warning(msg)
                else:
                    numbers = self.builder.fignumbers[key][figure_id]
                    self.body.append(prefix % '.'.join(map(str, numbers)) + ' ')
                    self.body.append('</span>')

        figtype = self.builder.env.domains['std'].get_enumerable_node_type(node)
        if figtype:
            if len(node['ids']) == 0:
                msg = __('Any IDs not assigned for %s node') % node.tagname
                logger.warning(msg, location=node)
            else:
                append_fignumber(figtype, node['ids'][0])
```
