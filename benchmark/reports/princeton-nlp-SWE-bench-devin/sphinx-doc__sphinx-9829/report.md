# sphinx-doc__sphinx-9829

| **sphinx-doc/sphinx** | `6c6cc8a6f50b18331cb818160d168d7bb9c03e55` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 528 |
| **Any found context length** | 528 |
| **Avg pos** | 2.0 |
| **Min pos** | 2 |
| **Max pos** | 2 |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sphinx/ext/mathjax.py b/sphinx/ext/mathjax.py
--- a/sphinx/ext/mathjax.py
+++ b/sphinx/ext/mathjax.py
@@ -81,7 +81,7 @@ def install_mathjax(app: Sphinx, pagename: str, templatename: str, context: Dict
     domain = cast(MathDomain, app.env.get_domain('math'))
     if app.registry.html_assets_policy == 'always' or domain.has_equations(pagename):
         # Enable mathjax only if equations exists
-        options = {'async': 'async'}
+        options = {'defer': 'defer'}
         if app.config.mathjax_options:
             options.update(app.config.mathjax_options)
         app.add_js_file(app.config.mathjax_path, **options)  # type: ignore

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sphinx/ext/mathjax.py | 84 | 84 | 2 | 1 | 528


## Problem Statement

```
Add a way to defer loading of MathJax
**Is your feature request related to a problem? Please describe.**

It is quite tricky to configure MathJax to work with Sphinx currently.

Sphinx loads MathJax asynchronously since https://github.com/sphinx-doc/sphinx/issues/3606 and https://github.com/sphinx-doc/sphinx/pull/5005.  While this was fine for MathJax 2, because of the special kind of ``<script>`` blocks mentioned in https://github.com/sphinx-doc/sphinx/issues/5616 , it doesn't work well with MathJax 3.

Indeed, in MathJax 3, MathJax expect a config `<script>` block to be present *before* MathJax is loaded. Sphinx 4 added `mathjax3_config` parameter:

\`\`\`
        if app.config.mathjax3_config:
            body = 'window.MathJax = %s' % json.dumps(app.config.mathjax3_config)
            app.add_js_file(None, body=body)
\`\`\`

This assumes that the `config` is a simple dictionary, which isn't sufficient: that configuration should be able to contain functions, for example.

The only possibility at the moment is to add a separate script file containing a MathJax configuration and to load it with ``app.add_js_file``.

**Describe the solution you'd like**

There are three possibilities:

- Allow arbitrary strings for mathjax3_config, and in that case don't JSON-serialize them.
- Change `async` to `defer` when loading MathJax.
- Make it possible for users to change `async` to `defer` themselves.  At the moment this isn't possible because the `async` flags is unconditionally added:

  \`\`\`
      if app.registry.html_assets_policy == 'always' or domain.has_equations(pagename):
        # Enable mathjax only if equations exists
        options = {'async': 'async'}
        if app.config.mathjax_options:
            options.update(app.config.mathjax_options)
  \`\`\`

The latter two are preferable because they would allow individual pages to use different MathJax config by using a `.. raw::` block to override the default MathJax configuration on a given page (the script in that ``raw`` block will run before MathJax loads thanks to the `defer` option).

CC @jfbu , the author of #5616.

Thanks!


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 sphinx/ext/mathjax.py** | 101 | 116| 198 | 198 | 1197 | 
| **-> 2 <-** | **1 sphinx/ext/mathjax.py** | 73 | 98| 330 | 528 | 1197 | 
| 3 | **1 sphinx/ext/mathjax.py** | 13 | 38| 222 | 750 | 1197 | 
| 4 | 2 sphinx/ext/imgmath.py | 345 | 365| 289 | 1039 | 4382 | 
| 5 | **2 sphinx/ext/mathjax.py** | 41 | 70| 356 | 1395 | 4382 | 
| 6 | 3 sphinx/domains/math.py | 127 | 156| 240 | 1635 | 5684 | 
| 7 | 4 sphinx/application.py | 939 | 988| 484 | 2119 | 17441 | 
| 8 | 5 sphinx/directives/patches.py | 224 | 260| 326 | 2445 | 19379 | 
| 9 | 6 sphinx/builders/latex/__init__.py | 465 | 525| 521 | 2966 | 25091 | 
| 10 | 7 sphinx/builders/html/__init__.py | 1305 | 1386| 1000 | 3966 | 37454 | 
| 11 | 7 sphinx/ext/imgmath.py | 11 | 82| 506 | 4472 | 37454 | 
| 12 | 7 sphinx/directives/patches.py | 194 | 222| 218 | 4690 | 37454 | 
| 13 | 7 sphinx/domains/math.py | 98 | 125| 300 | 4990 | 37454 | 
| 14 | 7 sphinx/builders/html/__init__.py | 1219 | 1241| 242 | 5232 | 37454 | 
| 15 | 8 sphinx/registry.py | 401 | 413| 157 | 5389 | 42094 | 
| 16 | 9 sphinx/ext/autodoc/__init__.py | 2798 | 2841| 528 | 5917 | 66006 | 
| 17 | 10 doc/conf.py | 1 | 82| 732 | 6649 | 67471 | 
| 18 | 11 sphinx/builders/latex/constants.py | 126 | 217| 1040 | 7689 | 69716 | 
| 19 | 11 sphinx/builders/html/__init__.py | 332 | 363| 278 | 7967 | 69716 | 
| 20 | 11 sphinx/ext/imgmath.py | 275 | 311| 328 | 8295 | 69716 | 
| 21 | 11 sphinx/domains/math.py | 39 | 96| 536 | 8831 | 69716 | 
| 22 | 12 sphinx/builders/latex/transforms.py | 532 | 548| 182 | 9013 | 74040 | 
| 23 | 12 sphinx/builders/latex/__init__.py | 528 | 561| 404 | 9417 | 74040 | 
| 24 | 12 sphinx/application.py | 1208 | 1222| 165 | 9582 | 74040 | 
| 25 | 12 sphinx/ext/imgmath.py | 314 | 342| 301 | 9883 | 74040 | 
| 26 | 12 sphinx/builders/html/__init__.py | 1164 | 1196| 281 | 10164 | 74040 | 
| 27 | 12 sphinx/builders/html/__init__.py | 314 | 330| 180 | 10344 | 74040 | 
| 28 | 12 sphinx/ext/imgmath.py | 213 | 272| 592 | 10936 | 74040 | 
| 29 | 12 sphinx/domains/math.py | 11 | 36| 188 | 11124 | 74040 | 
| 30 | 13 sphinx/jinja2glue.py | 145 | 193| 463 | 11587 | 75665 | 
| 31 | 14 sphinx/highlighting.py | 11 | 68| 613 | 12200 | 77204 | 
| 32 | 15 sphinx/util/docutils.py | 146 | 170| 194 | 12394 | 81405 | 
| 33 | 16 sphinx/util/jsdump.py | 107 | 201| 578 | 12972 | 82811 | 
| 34 | 16 sphinx/builders/latex/transforms.py | 613 | 630| 142 | 13114 | 82811 | 
| 35 | 16 sphinx/application.py | 1069 | 1090| 219 | 13333 | 82811 | 
| 36 | 16 sphinx/application.py | 990 | 1047| 596 | 13929 | 82811 | 
| 37 | 16 sphinx/highlighting.py | 71 | 180| 873 | 14802 | 82811 | 
| 38 | 17 sphinx/cmd/build.py | 33 | 98| 647 | 15449 | 85473 | 
| 39 | 17 sphinx/ext/imgmath.py | 85 | 106| 188 | 15637 | 85473 | 
| 40 | 18 setup.py | 81 | 173| 643 | 16280 | 87246 | 
| 41 | 18 sphinx/builders/latex/constants.py | 74 | 124| 537 | 16817 | 87246 | 
| 42 | 18 sphinx/registry.py | 415 | 466| 463 | 17280 | 87246 | 
| 43 | 18 sphinx/builders/latex/__init__.py | 11 | 42| 331 | 17611 | 87246 | 
| 44 | 18 sphinx/ext/autodoc/__init__.py | 13 | 114| 788 | 18399 | 87246 | 
| 45 | 18 sphinx/registry.py | 352 | 392| 426 | 18825 | 87246 | 
| 46 | 19 sphinx/builders/latex/theming.py | 11 | 48| 256 | 19081 | 88270 | 
| 47 | 20 sphinx/directives/code.py | 407 | 470| 642 | 19723 | 92121 | 
| 48 | 20 sphinx/ext/imgmath.py | 122 | 146| 268 | 19991 | 92121 | 
| 49 | 21 sphinx/domains/javascript.py | 11 | 33| 199 | 20190 | 96274 | 
| 50 | 21 sphinx/ext/autodoc/__init__.py | 1281 | 1400| 1004 | 21194 | 96274 | 
| 51 | 21 doc/conf.py | 141 | 162| 255 | 21449 | 96274 | 
| 52 | 22 sphinx/ext/graphviz.py | 405 | 421| 196 | 21645 | 100010 | 
| 53 | 22 sphinx/domains/javascript.py | 450 | 473| 200 | 21845 | 100010 | 
| 54 | 23 sphinx/extension.py | 44 | 82| 273 | 22118 | 100573 | 
| 55 | 23 setup.py | 1 | 78| 478 | 22596 | 100573 | 
| 56 | 24 sphinx/writers/latex.py | 284 | 436| 1688 | 24284 | 119964 | 
| 57 | 25 sphinx/ext/apidoc.py | 303 | 368| 752 | 25036 | 124198 | 
| 58 | 26 sphinx/ext/autodoc/typehints.py | 130 | 185| 460 | 25496 | 125652 | 
| 59 | 27 sphinx/ext/autodoc/directive.py | 9 | 47| 310 | 25806 | 127103 | 
| 60 | 27 sphinx/builders/latex/__init__.py | 44 | 99| 825 | 26631 | 127103 | 
| 61 | 27 sphinx/application.py | 1049 | 1067| 138 | 26769 | 127103 | 
| 62 | 27 sphinx/directives/patches.py | 9 | 44| 292 | 27061 | 127103 | 
| 63 | 27 sphinx/ext/autodoc/__init__.py | 2070 | 2275| 1863 | 28924 | 127103 | 
| 64 | 28 sphinx/ext/todo.py | 225 | 248| 214 | 29138 | 128944 | 
| 65 | 29 sphinx/directives/other.py | 369 | 393| 228 | 29366 | 132071 | 
| 66 | 30 sphinx/builders/manpage.py | 110 | 129| 166 | 29532 | 133034 | 
| 67 | 30 sphinx/application.py | 61 | 123| 494 | 30026 | 133034 | 
| 68 | 30 sphinx/ext/autodoc/__init__.py | 2278 | 2306| 248 | 30274 | 133034 | 
| 69 | 30 sphinx/jinja2glue.py | 11 | 44| 219 | 30493 | 133034 | 
| 70 | 30 sphinx/application.py | 1273 | 1289| 143 | 30636 | 133034 | 
| 71 | 30 sphinx/domains/javascript.py | 119 | 140| 283 | 30919 | 133034 | 
| 72 | 31 sphinx/domains/index.py | 32 | 46| 129 | 31048 | 133977 | 
| 73 | 32 sphinx/transforms/__init__.py | 408 | 429| 172 | 31220 | 137149 | 
| 74 | 33 sphinx/domains/python.py | 1460 | 1474| 110 | 31330 | 149767 | 
| 75 | 33 sphinx/builders/latex/constants.py | 11 | 72| 612 | 31942 | 149767 | 
| 76 | 34 sphinx/ext/viewcode.py | 344 | 363| 216 | 32158 | 152864 | 
| 77 | 35 sphinx/builders/epub3.py | 241 | 284| 607 | 32765 | 155451 | 
| 78 | 36 sphinx/cmd/quickstart.py | 11 | 119| 756 | 33521 | 161021 | 
| 79 | 37 sphinx/environment/__init__.py | 445 | 523| 687 | 34208 | 166542 | 
| 80 | 37 sphinx/ext/autodoc/__init__.py | 2338 | 2361| 210 | 34418 | 166542 | 
| 81 | 37 sphinx/writers/latex.py | 2065 | 2091| 251 | 34669 | 166542 | 
| 82 | 37 sphinx/domains/javascript.py | 334 | 393| 626 | 35295 | 166542 | 
| 83 | 37 sphinx/ext/autodoc/__init__.py | 2309 | 2336| 198 | 35493 | 166542 | 
| 84 | 37 sphinx/ext/autodoc/__init__.py | 601 | 642| 409 | 35902 | 166542 | 
| 85 | 37 sphinx/ext/autodoc/__init__.py | 1896 | 1934| 307 | 36209 | 166542 | 
| 86 | 38 sphinx/pycode/ast.py | 11 | 44| 214 | 36423 | 168629 | 
| 87 | 38 doc/conf.py | 83 | 138| 476 | 36899 | 168629 | 
| 88 | 39 utils/jssplitter_generator.py | 1 | 80| 529 | 37428 | 169603 | 
| 89 | 39 sphinx/util/docutils.py | 173 | 197| 192 | 37620 | 169603 | 


### Hint

```
I'm not good at loading JS. Could you let me know the impact of changing `async` to `defer`? Are there any incompatible change for users? If not, we can change the loading option for MathJax to `defer` in the next version.
I don't think it's an incompatible change.  For MDN:

> - If the async attribute is present, then the script will be executed asynchronously as soon as it downloads.
> - If the async attribute is absent but the defer attribute is present, then the script is executed when the page has finished parsing.

So changing `async` to `defer` just rules out certain behaviors (the script starts executing before the whole page is parsed), it shouldn't add any new ones.

I found an explanation for this topic:

>Note that here we use the defer attribute on both scripts so that they will execute in order, but still not block the rest of the page while the files are being downloaded to the browser. If the async attribute were used, there is no guarantee that the configuration would run first, and so you could get instances where MathJax doesn’t get properly configured, and they would seem to occur randomly.
>https://docs.mathjax.org/en/latest/web/configuration.html#using-a-local-file-for-configuration

I believe using defer option instead is a good alternative.
```

## Patch

```diff
diff --git a/sphinx/ext/mathjax.py b/sphinx/ext/mathjax.py
--- a/sphinx/ext/mathjax.py
+++ b/sphinx/ext/mathjax.py
@@ -81,7 +81,7 @@ def install_mathjax(app: Sphinx, pagename: str, templatename: str, context: Dict
     domain = cast(MathDomain, app.env.get_domain('math'))
     if app.registry.html_assets_policy == 'always' or domain.has_equations(pagename):
         # Enable mathjax only if equations exists
-        options = {'async': 'async'}
+        options = {'defer': 'defer'}
         if app.config.mathjax_options:
             options.update(app.config.mathjax_options)
         app.add_js_file(app.config.mathjax_path, **options)  # type: ignore

```

## Test Patch

```diff
diff --git a/tests/test_ext_math.py b/tests/test_ext_math.py
--- a/tests/test_ext_math.py
+++ b/tests/test_ext_math.py
@@ -71,7 +71,7 @@ def test_mathjax_options(app, status, warning):
     app.builder.build_all()
 
     content = (app.outdir / 'index.html').read_text()
-    assert ('<script async="async" integrity="sha384-0123456789" '
+    assert ('<script defer="defer" integrity="sha384-0123456789" '
             'src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">'
             '</script>' in content)
 

```


## Code snippets

### 1 - sphinx/ext/mathjax.py:

Start line: 101, End line: 116

```python
def setup(app: Sphinx) -> Dict[str, Any]:
    app.add_html_math_renderer('mathjax',
                               (html_visit_math, None),
                               (html_visit_displaymath, None))

    app.add_config_value('mathjax_path', MATHJAX_URL, 'html')
    app.add_config_value('mathjax_options', {}, 'html')
    app.add_config_value('mathjax_inline', [r'\(', r'\)'], 'html')
    app.add_config_value('mathjax_display', [r'\[', r'\]'], 'html')
    app.add_config_value('mathjax_config', None, 'html')
    app.add_config_value('mathjax2_config', lambda c: c.mathjax_config, 'html')
    app.add_config_value('mathjax3_config', None, 'html')
    app.connect('html-page-context', install_mathjax)

    return {'version': sphinx.__display_version__, 'parallel_read_safe': True}
```
### 2 - sphinx/ext/mathjax.py:

Start line: 73, End line: 98

```python
def install_mathjax(app: Sphinx, pagename: str, templatename: str, context: Dict,
                    event_arg: Any) -> None:
    if app.builder.format != 'html' or app.builder.math_renderer_name != 'mathjax':  # type: ignore  # NOQA
        return
    if not app.config.mathjax_path:
        raise ExtensionError('mathjax_path config value must be set for the '
                             'mathjax extension to work')

    domain = cast(MathDomain, app.env.get_domain('math'))
    if app.registry.html_assets_policy == 'always' or domain.has_equations(pagename):
        # Enable mathjax only if equations exists
        options = {'async': 'async'}
        if app.config.mathjax_options:
            options.update(app.config.mathjax_options)
        app.add_js_file(app.config.mathjax_path, **options)  # type: ignore

        if app.config.mathjax2_config:
            if app.config.mathjax_path == MATHJAX_URL:
                logger.warning(
                    'mathjax_config/mathjax2_config does not work '
                    'for the current MathJax version, use mathjax3_config instead')
            body = 'MathJax.Hub.Config(%s)' % json.dumps(app.config.mathjax2_config)
            app.add_js_file(None, type='text/x-mathjax-config', body=body)
        if app.config.mathjax3_config:
            body = 'window.MathJax = %s' % json.dumps(app.config.mathjax3_config)
            app.add_js_file(None, body=body)
```
### 3 - sphinx/ext/mathjax.py:

Start line: 13, End line: 38

```python
import json
from typing import Any, Dict, cast

from docutils import nodes

import sphinx
from sphinx.application import Sphinx
from sphinx.domains.math import MathDomain
from sphinx.errors import ExtensionError
from sphinx.locale import _
from sphinx.util.math import get_node_equation_number
from sphinx.writers.html import HTMLTranslator

# more information for mathjax secure url is here:
# https://docs.mathjax.org/en/latest/start.html#secure-access-to-the-cdn
MATHJAX_URL = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js'

logger = sphinx.util.logging.getLogger(__name__)


def html_visit_math(self: HTMLTranslator, node: nodes.math) -> None:
    self.body.append(self.starttag(node, 'span', '', CLASS='math notranslate nohighlight'))
    self.body.append(self.builder.config.mathjax_inline[0] +
                     self.encode(node.astext()) +
                     self.builder.config.mathjax_inline[1] + '</span>')
    raise nodes.SkipNode
```
### 4 - sphinx/ext/imgmath.py:

Start line: 345, End line: 365

```python
def setup(app: Sphinx) -> Dict[str, Any]:
    app.add_html_math_renderer('imgmath',
                               (html_visit_math, None),
                               (html_visit_displaymath, None))

    app.add_config_value('imgmath_image_format', 'png', 'html')
    app.add_config_value('imgmath_dvipng', 'dvipng', 'html')
    app.add_config_value('imgmath_dvisvgm', 'dvisvgm', 'html')
    app.add_config_value('imgmath_latex', 'latex', 'html')
    app.add_config_value('imgmath_use_preview', False, 'html')
    app.add_config_value('imgmath_dvipng_args',
                         ['-gamma', '1.5', '-D', '110', '-bg', 'Transparent'],
                         'html')
    app.add_config_value('imgmath_dvisvgm_args', ['--no-fonts'], 'html')
    app.add_config_value('imgmath_latex_args', [], 'html')
    app.add_config_value('imgmath_latex_preamble', '', 'html')
    app.add_config_value('imgmath_add_tooltips', True, 'html')
    app.add_config_value('imgmath_font_size', 12, 'html')
    app.connect('build-finished', cleanup_tempdir)
    return {'version': sphinx.__display_version__, 'parallel_read_safe': True}
```
### 5 - sphinx/ext/mathjax.py:

Start line: 41, End line: 70

```python
def html_visit_displaymath(self: HTMLTranslator, node: nodes.math_block) -> None:
    self.body.append(self.starttag(node, 'div', CLASS='math notranslate nohighlight'))
    if node['nowrap']:
        self.body.append(self.encode(node.astext()))
        self.body.append('</div>')
        raise nodes.SkipNode

    # necessary to e.g. set the id property correctly
    if node['number']:
        number = get_node_equation_number(self, node)
        self.body.append('<span class="eqno">(%s)' % number)
        self.add_permalink_ref(node, _('Permalink to this equation'))
        self.body.append('</span>')
    self.body.append(self.builder.config.mathjax_display[0])
    parts = [prt for prt in node.astext().split('\n\n') if prt.strip()]
    if len(parts) > 1:  # Add alignment if there are more than 1 equation
        self.body.append(r' \begin{align}\begin{aligned}')
    for i, part in enumerate(parts):
        part = self.encode(part)
        if r'\\' in part:
            self.body.append(r'\begin{split}' + part + r'\end{split}')
        else:
            self.body.append(part)
        if i < len(parts) - 1:  # append new line if not the last equation
            self.body.append(r'\\')
    if len(parts) > 1:  # Add alignment if there are more than 1 equation
        self.body.append(r'\end{aligned}\end{align} ')
    self.body.append(self.builder.config.mathjax_display[1])
    self.body.append('</div>\n')
    raise nodes.SkipNode
```
### 6 - sphinx/domains/math.py:

Start line: 127, End line: 156

```python
class MathDomain(Domain):

    def resolve_any_xref(self, env: BuildEnvironment, fromdocname: str, builder: "Builder",
                         target: str, node: pending_xref, contnode: Element
                         ) -> List[Tuple[str, Element]]:
        refnode = self.resolve_xref(env, fromdocname, builder, 'eq', target, node, contnode)
        if refnode is None:
            return []
        else:
            return [('eq', refnode)]

    def get_objects(self) -> List:
        return []

    def has_equations(self, docname: str = None) -> bool:
        if docname:
            return self.data['has_equations'].get(docname, False)
        else:
            return any(self.data['has_equations'].values())


def setup(app: "Sphinx") -> Dict[str, Any]:
    app.add_domain(MathDomain)
    app.add_role('eq', MathReferenceRole(warn_dangling=True))

    return {
        'version': 'builtin',
        'env_version': 2,
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
```
### 7 - sphinx/application.py:

Start line: 939, End line: 988

```python
class Sphinx:

    def add_js_file(self, filename: str, priority: int = 500, **kwargs: Any) -> None:
        """Register a JavaScript file to include in the HTML output.

        Add *filename* to the list of JavaScript files that the default HTML
        template will include in order of *priority* (ascending).  The filename
        must be relative to the HTML static path , or a full URI with scheme.
        If the priority of the JavaScript file is the same as others, the JavaScript
        files will be included in order of registration.  If the keyword
        argument ``body`` is given, its value will be added between the
        ``<script>`` tags. Extra keyword arguments are included as attributes of
        the ``<script>`` tag.

        Example::

            app.add_js_file('example.js')
            # => <script src="_static/example.js"></script>

            app.add_js_file('example.js', async="async")
            # => <script src="_static/example.js" async="async"></script>

            app.add_js_file(None, body="var myVariable = 'foo';")
            # => <script>var myVariable = 'foo';</script>

        .. list-table:: priority range for JavaScript files
           :widths: 20,80

           * - Priority
             - Main purpose in Sphinx
           * - 200
             - default priority for built-in JavaScript files
           * - 500
             - default priority for extensions
           * - 800
             - default priority for :confval:`html_js_files`

        A JavaScript file can be added to the specific HTML page when an extension
        calls this method on :event:`html-page-context` event.

        .. versionadded:: 0.5

        .. versionchanged:: 1.8
           Renamed from ``app.add_javascript()``.
           And it allows keyword arguments as attributes of script tag.

        .. versionchanged:: 3.5
           Take priority argument.  Allow to add a JavaScript file to the specific page.
        """
        self.registry.add_js_file(filename, priority=priority, **kwargs)
        if hasattr(self.builder, 'add_js_file'):
            self.builder.add_js_file(filename, priority=priority, **kwargs)  # type: ignore
```
### 8 - sphinx/directives/patches.py:

Start line: 224, End line: 260

```python
class MathDirective(SphinxDirective):

    def add_target(self, ret: List[Node]) -> None:
        node = cast(nodes.math_block, ret[0])

        # assign label automatically if math_number_all enabled
        if node['label'] == '' or (self.config.math_number_all and not node['label']):
            seq = self.env.new_serialno('sphinx.ext.math#equations')
            node['label'] = "%s:%d" % (self.env.docname, seq)

        # no targets and numbers are needed
        if not node['label']:
            return

        # register label to domain
        domain = cast(MathDomain, self.env.get_domain('math'))
        domain.note_equation(self.env.docname, node['label'], location=node)
        node['number'] = domain.get_equation_number_for(node['label'])

        # add target node
        node_id = make_id('equation-%s' % node['label'])
        target = nodes.target('', '', ids=[node_id])
        self.state.document.note_explicit_target(target)
        ret.insert(0, target)


def setup(app: "Sphinx") -> Dict[str, Any]:
    directives.register_directive('figure', Figure)
    directives.register_directive('meta', Meta)
    directives.register_directive('csv-table', CSVTable)
    directives.register_directive('code', Code)
    directives.register_directive('math', MathDirective)

    return {
        'version': 'builtin',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
```
### 9 - sphinx/builders/latex/__init__.py:

Start line: 465, End line: 525

```python
def validate_config_values(app: Sphinx, config: Config) -> None:
    for key in list(config.latex_elements):
        if key not in DEFAULT_SETTINGS:
            msg = __("Unknown configure key: latex_elements[%r], ignored.")
            logger.warning(msg % (key,))
            config.latex_elements.pop(key)


def validate_latex_theme_options(app: Sphinx, config: Config) -> None:
    for key in list(config.latex_theme_options):
        if key not in Theme.UPDATABLE_KEYS:
            msg = __("Unknown theme option: latex_theme_options[%r], ignored.")
            logger.warning(msg % (key,))
            config.latex_theme_options.pop(key)


def install_packages_for_ja(app: Sphinx) -> None:
    """Install packages for Japanese."""
    if app.config.language == 'ja' and app.config.latex_engine in ('platex', 'uplatex'):
        app.add_latex_package('pxjahyper', after_hyperref=True)


def default_latex_engine(config: Config) -> str:
    """ Better default latex_engine settings for specific languages. """
    if config.language == 'ja':
        return 'uplatex'
    elif (config.language or '').startswith('zh'):
        return 'xelatex'
    elif config.language == 'el':
        return 'xelatex'
    else:
        return 'pdflatex'


def default_latex_docclass(config: Config) -> Dict[str, str]:
    """ Better default latex_docclass settings for specific languages. """
    if config.language == 'ja':
        if config.latex_engine == 'uplatex':
            return {'manual': 'ujbook',
                    'howto': 'ujreport'}
        else:
            return {'manual': 'jsbook',
                    'howto': 'jreport'}
    else:
        return {}


def default_latex_use_xindy(config: Config) -> bool:
    """ Better default latex_use_xindy settings for specific engines. """
    return config.latex_engine in {'xelatex', 'lualatex'}


def default_latex_documents(config: Config) -> List[Tuple[str, str, str, str, str]]:
    """ Better default latex_documents settings. """
    project = texescape.escape(config.project, config.latex_engine)
    author = texescape.escape(config.author, config.latex_engine)
    return [(config.root_doc,
             make_filename_from_project(config.project) + '.tex',
             texescape.escape_abbr(project),
             texescape.escape_abbr(author),
             config.latex_theme)]
```
### 10 - sphinx/builders/html/__init__.py:

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
