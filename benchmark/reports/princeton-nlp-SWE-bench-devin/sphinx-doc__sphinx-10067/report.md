# sphinx-doc__sphinx-10067

| **sphinx-doc/sphinx** | `e1fa6c79be8b3928c21e312a0c0e65e1cfd9a7f7` |
| ---- | ---- |
| **No of patches** | 9 |
| **All found context length** | 8688 |
| **Any found context length** | 5042 |
| **Avg pos** | 30.11111111111111 |
| **Min pos** | 8 |
| **Max pos** | 62 |
| **Top file pos** | 4 |
| **Missing snippets** | 22 |
| **Missing patch files** | 1 |


## Expected patch

```diff
diff --git a/sphinx/application.py b/sphinx/application.py
--- a/sphinx/application.py
+++ b/sphinx/application.py
@@ -266,7 +266,7 @@ def _init_i18n(self) -> None:
         """Load translated strings from the configured localedirs if enabled in
         the configuration.
         """
-        if self.config.language is None:
+        if self.config.language == 'en':
             self.translator, has_translation = locale.init([], None)
         else:
             logger.info(bold(__('loading translations [%s]... ') % self.config.language),
@@ -285,8 +285,7 @@ def _init_i18n(self) -> None:
             locale_dirs += [path.join(package_dir, 'locale')]
 
             self.translator, has_translation = locale.init(locale_dirs, self.config.language)
-            if has_translation or self.config.language == 'en':
-                # "en" never needs to be translated
+            if has_translation:
                 logger.info(__('done'))
             else:
                 logger.info(__('not available for built-in messages'))
diff --git a/sphinx/builders/html/__init__.py b/sphinx/builders/html/__init__.py
--- a/sphinx/builders/html/__init__.py
+++ b/sphinx/builders/html/__init__.py
@@ -326,7 +326,7 @@ def init_js_files(self) -> None:
             attrs.setdefault('priority', 800)  # User's JSs are loaded after extensions'
             self.add_js_file(filename, **attrs)
 
-        if self.config.language and self._get_translations_js():
+        if self._get_translations_js():
             self.add_js_file('translations.js')
 
     def add_js_file(self, filename: str, **kwargs: Any) -> None:
@@ -431,8 +431,6 @@ def prepare_writing(self, docnames: Set[str]) -> None:
         if self.search:
             from sphinx.search import IndexBuilder
             lang = self.config.html_search_language or self.config.language
-            if not lang:
-                lang = 'en'
             self.indexer = IndexBuilder(self.env, lang,
                                         self.config.html_search_options,
                                         self.config.html_search_scorer)
@@ -767,10 +765,9 @@ def create_pygments_style_file(self) -> None:
 
     def copy_translation_js(self) -> None:
         """Copy a JavaScript file for translations."""
-        if self.config.language is not None:
-            jsfile = self._get_translations_js()
-            if jsfile:
-                copyfile(jsfile, path.join(self.outdir, '_static', 'translations.js'))
+        jsfile = self._get_translations_js()
+        if jsfile:
+            copyfile(jsfile, path.join(self.outdir, '_static', 'translations.js'))
 
     def copy_stemmer_js(self) -> None:
         """Copy a JavaScript file for stemmer."""
diff --git a/sphinx/builders/latex/__init__.py b/sphinx/builders/latex/__init__.py
--- a/sphinx/builders/latex/__init__.py
+++ b/sphinx/builders/latex/__init__.py
@@ -170,9 +170,8 @@ def init_context(self) -> None:
         self.context.update(ADDITIONAL_SETTINGS.get(self.config.latex_engine, {}))
 
         # Add special settings for (latex_engine, language_code)
-        if self.config.language:
-            key = (self.config.latex_engine, self.config.language[:2])
-            self.context.update(ADDITIONAL_SETTINGS.get(key, {}))
+        key = (self.config.latex_engine, self.config.language[:2])
+        self.context.update(ADDITIONAL_SETTINGS.get(key, {}))
 
         # Apply user settings to context
         self.context.update(self.config.latex_elements)
@@ -203,7 +202,7 @@ def update_context(self) -> None:
 
     def init_babel(self) -> None:
         self.babel = ExtBabel(self.config.language, not self.context['babel'])
-        if self.config.language and not self.babel.is_supported_language():
+        if not self.babel.is_supported_language():
             # emit warning if specified language is invalid
             # (only emitting, nothing changed to processing)
             logger.warning(__('no Babel option known for language %r'),
@@ -232,12 +231,11 @@ def init_multilingual(self) -> None:
             self.context['classoptions'] += ',' + self.babel.get_language()
             # this branch is not taken for xelatex/lualatex if default settings
             self.context['multilingual'] = self.context['babel']
-            if self.config.language:
-                self.context['shorthandoff'] = SHORTHANDOFF
+            self.context['shorthandoff'] = SHORTHANDOFF
 
-                # Times fonts don't work with Cyrillic languages
-                if self.babel.uses_cyrillic() and 'fontpkg' not in self.config.latex_elements:
-                    self.context['fontpkg'] = ''
+            # Times fonts don't work with Cyrillic languages
+            if self.babel.uses_cyrillic() and 'fontpkg' not in self.config.latex_elements:
+                self.context['fontpkg'] = ''
         elif self.context['polyglossia']:
             self.context['classoptions'] += ',' + self.babel.get_language()
             options = self.babel.get_mainlanguage_options()
@@ -380,14 +378,10 @@ def copy_support_files(self) -> None:
         # configure usage of xindy (impacts Makefile and latexmkrc)
         # FIXME: convert this rather to a confval with suitable default
         #        according to language ? but would require extra documentation
-        if self.config.language:
-            xindy_lang_option = \
-                XINDY_LANG_OPTIONS.get(self.config.language[:2],
-                                       '-L general -C utf8 ')
-            xindy_cyrillic = self.config.language[:2] in XINDY_CYRILLIC_SCRIPTS
-        else:
-            xindy_lang_option = '-L english -C utf8 '
-            xindy_cyrillic = False
+        xindy_lang_option = XINDY_LANG_OPTIONS.get(self.config.language[:2],
+                                                   '-L general -C utf8 ')
+        xindy_cyrillic = self.config.language[:2] in XINDY_CYRILLIC_SCRIPTS
+
         context = {
             'latex_engine':      self.config.latex_engine,
             'xindy_use':         self.config.latex_use_xindy,
@@ -474,7 +468,7 @@ def default_latex_engine(config: Config) -> str:
     """ Better default latex_engine settings for specific languages. """
     if config.language == 'ja':
         return 'uplatex'
-    elif (config.language or '').startswith('zh'):
+    elif config.language.startswith('zh'):
         return 'xelatex'
     elif config.language == 'el':
         return 'xelatex'
diff --git a/sphinx/builders/latex/util.py b/sphinx/builders/latex/util.py
--- a/sphinx/builders/latex/util.py
+++ b/sphinx/builders/latex/util.py
@@ -20,7 +20,7 @@ def __init__(self, language_code: str, use_polyglossia: bool = False) -> None:
         self.language_code = language_code
         self.use_polyglossia = use_polyglossia
         self.supported = True
-        super().__init__(language_code or '')
+        super().__init__(language_code)
 
     def uses_cyrillic(self) -> bool:
         return self.language in self.cyrillic_languages
diff --git a/sphinx/config.py b/sphinx/config.py
--- a/sphinx/config.py
+++ b/sphinx/config.py
@@ -100,7 +100,7 @@ class Config:
         # the real default is locale-dependent
         'today_fmt': (None, 'env', [str]),
 
-        'language': (None, 'env', [str]),
+        'language': ('en', 'env', [str]),
         'locale_dirs': (['locales'], 'env', []),
         'figure_language_filename': ('{root}.{language}{ext}', 'env', [str]),
         'gettext_allow_fuzzy_translations': (False, 'gettext', []),
diff --git a/sphinx/environment/__init__.py b/sphinx/environment/__init__.py
--- a/sphinx/environment/__init__.py
+++ b/sphinx/environment/__init__.py
@@ -261,7 +261,7 @@ def _update_settings(self, config: Config) -> None:
         """Update settings by new config."""
         self.settings['input_encoding'] = config.source_encoding
         self.settings['trim_footnote_reference_space'] = config.trim_footnote_reference_space
-        self.settings['language_code'] = config.language or 'en'
+        self.settings['language_code'] = config.language
 
         # Allow to disable by 3rd party extension (workaround)
         self.settings.setdefault('smart_quotes', True)
diff --git a/sphinx/environment/collectors/asset.py b/sphinx/environment/collectors/asset.py
--- a/sphinx/environment/collectors/asset.py
+++ b/sphinx/environment/collectors/asset.py
@@ -64,18 +64,16 @@ def process_doc(self, app: Sphinx, doctree: nodes.document) -> None:
                 rel_imgpath, full_imgpath = app.env.relfn2path(imguri, docname)
                 node['uri'] = rel_imgpath
 
-                if app.config.language:
-                    # Search language-specific figures at first
-                    i18n_imguri = get_image_filename_for_language(imguri, app.env)
-                    _, full_i18n_imgpath = app.env.relfn2path(i18n_imguri, docname)
-                    self.collect_candidates(app.env, full_i18n_imgpath, candidates, node)
+                # Search language-specific figures at first
+                i18n_imguri = get_image_filename_for_language(imguri, app.env)
+                _, full_i18n_imgpath = app.env.relfn2path(i18n_imguri, docname)
+                self.collect_candidates(app.env, full_i18n_imgpath, candidates, node)
 
                 self.collect_candidates(app.env, full_imgpath, candidates, node)
             else:
-                if app.config.language:
-                    # substitute imguri by figure_language_filename
-                    # (ex. foo.png -> foo.en.png)
-                    imguri = search_image_for_language(imguri, app.env)
+                # substitute imguri by figure_language_filename
+                # (ex. foo.png -> foo.en.png)
+                imguri = search_image_for_language(imguri, app.env)
 
                 # Update `node['uri']` to a relative path from srcdir
                 # from a relative path from current document.
diff --git a/sphinx/util/i18n.py b/sphinx/util/i18n.py
--- a/sphinx/util/i18n.py
+++ b/sphinx/util/i18n.py
@@ -10,14 +10,16 @@
 
 import os
 import re
+import warnings
 from datetime import datetime, timezone
 from os import path
-from typing import TYPE_CHECKING, Callable, Generator, List, NamedTuple, Optional, Tuple, Union
+from typing import TYPE_CHECKING, Callable, Generator, List, NamedTuple, Tuple, Union
 
 import babel.dates
 from babel.messages.mofile import write_mo
 from babel.messages.pofile import read_po
 
+from sphinx.deprecation import RemovedInSphinx70Warning
 from sphinx.errors import SphinxError
 from sphinx.locale import __
 from sphinx.util import logging
@@ -173,9 +175,11 @@ def docname_to_domain(docname: str, compaction: Union[bool, str]) -> str:
 date_format_re = re.compile('(%s)' % '|'.join(date_format_mappings))
 
 
-def babel_format_date(date: datetime, format: str, locale: Optional[str],
+def babel_format_date(date: datetime, format: str, locale: str,
                       formatter: Callable = babel.dates.format_date) -> str:
     if locale is None:
+        warnings.warn('The locale argument for babel_format_date() becomes required.',
+                      RemovedInSphinx70Warning)
         locale = 'en'
 
     # Check if we have the tzinfo attribute. If not we cannot do any time
@@ -194,7 +198,7 @@ def babel_format_date(date: datetime, format: str, locale: Optional[str],
         return format
 
 
-def format_date(format: str, date: datetime = None, language: Optional[str] = None) -> str:
+def format_date(format: str, date: datetime = None, language: str = None) -> str:
     if date is None:
         # If time is not specified, try to use $SOURCE_DATE_EPOCH variable
         # See https://wiki.debian.org/ReproducibleBuilds/TimestampsProposal
@@ -204,6 +208,11 @@ def format_date(format: str, date: datetime = None, language: Optional[str] = No
         else:
             date = datetime.now(timezone.utc).astimezone()
 
+    if language is None:
+        warnings.warn('The language argument for format_date() becomes required.',
+                      RemovedInSphinx70Warning)
+        language = 'en'
+
     result = []
     tokens = date_format_re.split(format)
     for token in tokens:
@@ -229,9 +238,6 @@ def format_date(format: str, date: datetime = None, language: Optional[str] = No
 
 
 def get_image_filename_for_language(filename: str, env: "BuildEnvironment") -> str:
-    if not env.config.language:
-        return filename
-
     filename_format = env.config.figure_language_filename
     d = dict()
     d['root'], d['ext'] = path.splitext(filename)
@@ -252,9 +258,6 @@ def get_image_filename_for_language(filename: str, env: "BuildEnvironment") -> s
 
 
 def search_image_for_language(filename: str, env: "BuildEnvironment") -> str:
-    if not env.config.language:
-        return filename
-
     translated = get_image_filename_for_language(filename, env)
     _, abspath = env.relfn2path(translated)
     if path.exists(abspath):
diff --git a/sphinx/writers/latex.py b/sphinx/writers/latex.py
--- a/sphinx/writers/latex.py
+++ b/sphinx/writers/latex.py
@@ -333,7 +333,7 @@ def __init__(self, document: nodes.document, builder: "LaTeXBuilder",
         if self.config.numfig and self.config.math_numfig:
             sphinxpkgoptions.append('mathnumfig')
 
-        if (self.config.language not in {None, 'en', 'ja'} and
+        if (self.config.language not in {'en', 'ja'} and
                 'fncychap' not in self.config.latex_elements):
             # use Sonny style if any language specified (except English)
             self.elements['fncychap'] = (r'\usepackage[Sonny]{fncychap}' + CR +
@@ -341,7 +341,7 @@ def __init__(self, document: nodes.document, builder: "LaTeXBuilder",
                                          r'\ChTitleVar{\Large\normalfont\sffamily}')
 
         self.babel = self.builder.babel
-        if self.config.language and not self.babel.is_supported_language():
+        if not self.babel.is_supported_language():
             # emit warning if specified language is invalid
             # (only emitting, nothing changed to processing)
             logger.warning(__('no Babel option known for language %r'),

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sphinx/application.py | 269 | 269 | - | 13 | -
| sphinx/application.py | 288 | 289 | - | 13 | -
| sphinx/builders/html/__init__.py | 329 | 329 | - | 4 | -
| sphinx/builders/html/__init__.py | 434 | 435 | - | 4 | -
| sphinx/builders/html/__init__.py | 770 | 773 | - | 4 | -
| sphinx/builders/latex/__init__.py | 173 | 175 | - | 5 | -
| sphinx/builders/latex/__init__.py | 206 | 206 | - | 5 | -
| sphinx/builders/latex/__init__.py | 235 | 240 | - | 5 | -
| sphinx/builders/latex/__init__.py | 383 | 390 | - | 5 | -
| sphinx/builders/latex/__init__.py | 477 | 477 | 8 | 5 | 5042
| sphinx/builders/latex/util.py | 23 | 23 | - | 15 | -
| sphinx/config.py | 103 | 103 | 17 | 12 | 8688
| sphinx/environment/__init__.py | 264 | 264 | - | 10 | -
| sphinx/environment/collectors/asset.py | 67 | 78 | - | - | -
| sphinx/util/i18n.py | 13 | 13 | - | 36 | -
| sphinx/util/i18n.py | 176 | 176 | - | 36 | -
| sphinx/util/i18n.py | 197 | 197 | - | 36 | -
| sphinx/util/i18n.py | 207 | 207 | - | 36 | -
| sphinx/util/i18n.py | 232 | 234 | 61 | 36 | 26050
| sphinx/util/i18n.py | 255 | 257 | 61 | 36 | 26050
| sphinx/writers/latex.py | 336 | 336 | 62 | 37 | 27553
| sphinx/writers/latex.py | 344 | 344 | 62 | 37 | 27553


## Problem Statement

```
To improve accessibility, set language in conf.py using sphinx-quickstart
**Is your feature request related to a problem? Please describe.**
By default, Sphinx documentation does not include the language, for example in `docs/conf.py`
`language = 'en'`

result in built web pages:
`<html lang="en">`

This leads to the following accessibility issue identified by [Lighthouse](https://developers.google.com/web/tools/lighthouse/):

`<html> element does not have a [lang] attribute `
> If a page doesn't specify a lang attribute, a screen reader assumes that the page is in the default language that the user chose when setting up the screen reader. If the page isn't actually in the default language, then the screen reader might not announce the page's text correctly. [Learn more](https://web.dev/html-has-lang/?utm_source=lighthouse&utm_medium=lr).`

Also, Sphinx sites thus do not by default take advantage of the [features offered by setting the language](https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-language).

This [accessibility issue is present in major sites including NumPy](https://googlechrome.github.io/lighthouse/viewer/?psiurl=https%3A%2F%2Fnumpy.org%2Fdoc%2Fstable%2F&strategy=mobile&category=performance&category=accessibility&category=best-practices&category=seo&category=pwa&utm_source=lh-chrome-ext).

**Describe the solution you'd like**
User already enters language when they run sphinx-quickstart:
\`\`\`
For a list of supported codes, see
https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-language.
> Project language [en]: 
\`\`\`

so it should automatically set that `language` value in the generated `conf.py` file.

It would also be nice if there was some prompt to set the `language` of existing Sphinx installations, upon an update of Sphinx version, or build of the documentation, for example.

**Describe alternatives you've considered**
Status quo, which retains accessibility issue.

**Additional context**
Related issue: #10056.



```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 sphinx/cmd/quickstart.py | 268 | 326| 739 | 739 | 5570 | 
| 2 | 1 sphinx/cmd/quickstart.py | 191 | 266| 675 | 1414 | 5570 | 
| 3 | 2 sphinx/locale/__init__.py | 241 | 269| 222 | 1636 | 7640 | 
| 4 | 2 sphinx/cmd/quickstart.py | 11 | 119| 756 | 2392 | 7640 | 
| 5 | 3 setup.py | 1 | 76| 478 | 2870 | 9414 | 
| 6 | **4 sphinx/builders/html/__init__.py** | 1305 | 1386| 1000 | 3870 | 21776 | 
| 7 | 4 setup.py | 174 | 251| 651 | 4521 | 21776 | 
| **-> 8 <-** | **5 sphinx/builders/latex/__init__.py** | 451 | 511| 521 | 5042 | 27354 | 
| 9 | 6 sphinx/util/docutils.py | 146 | 170| 194 | 5236 | 31685 | 
| 10 | 6 sphinx/cmd/quickstart.py | 544 | 611| 491 | 5727 | 31685 | 
| 11 | 7 doc/conf.py | 165 | 187| 269 | 5996 | 33406 | 
| 12 | 8 sphinx/ext/autodoc/__init__.py | 2846 | 2891| 557 | 6553 | 57777 | 
| 13 | 9 sphinx/builders/manpage.py | 110 | 129| 166 | 6719 | 58739 | 
| 14 | 9 doc/conf.py | 84 | 140| 502 | 7221 | 58739 | 
| 15 | **10 sphinx/environment/__init__.py** | 11 | 85| 535 | 7756 | 64256 | 
| 16 | 11 sphinx/directives/code.py | 33 | 56| 176 | 7932 | 68107 | 
| **-> 17 <-** | **12 sphinx/config.py** | 91 | 151| 756 | 8688 | 72582 | 
| 18 | **13 sphinx/application.py** | 1154 | 1168| 154 | 8842 | 84448 | 
| 19 | 14 sphinx/transforms/post_transforms/code.py | 49 | 85| 310 | 9152 | 85446 | 
| 20 | 14 sphinx/cmd/quickstart.py | 138 | 163| 234 | 9386 | 85446 | 
| 21 | **15 sphinx/builders/latex/util.py** | 31 | 57| 231 | 9617 | 85879 | 
| 22 | 15 sphinx/locale/__init__.py | 151 | 168| 132 | 9749 | 85879 | 
| 23 | 16 sphinx/highlighting.py | 11 | 68| 613 | 10362 | 87418 | 
| 24 | 17 sphinx/util/smartypants.py | 380 | 392| 137 | 10499 | 91561 | 
| 25 | 18 sphinx/builders/latex/constants.py | 74 | 124| 537 | 11036 | 93806 | 
| 26 | 18 sphinx/directives/code.py | 407 | 470| 642 | 11678 | 93806 | 
| 27 | 19 sphinx/domains/changeset.py | 49 | 107| 516 | 12194 | 95060 | 
| 28 | 19 sphinx/cmd/quickstart.py | 459 | 525| 752 | 12946 | 95060 | 
| 29 | **19 sphinx/environment/__init__.py** | 445 | 523| 687 | 13633 | 95060 | 
| 30 | **19 sphinx/application.py** | 1284 | 1300| 143 | 13776 | 95060 | 
| 31 | 19 sphinx/util/docutils.py | 126 | 143| 132 | 13908 | 95060 | 
| 32 | 19 sphinx/builders/latex/constants.py | 126 | 217| 1040 | 14948 | 95060 | 
| 33 | **19 sphinx/application.py** | 294 | 307| 132 | 15080 | 95060 | 
| 34 | **19 sphinx/config.py** | 464 | 483| 223 | 15303 | 95060 | 
| 35 | 20 sphinx/util/rst.py | 11 | 81| 556 | 15859 | 95928 | 
| 36 | 21 sphinx/setup_command.py | 93 | 120| 229 | 16088 | 97515 | 
| 37 | 22 sphinx/builders/changes.py | 125 | 168| 438 | 16526 | 99028 | 
| 38 | 22 sphinx/builders/manpage.py | 11 | 29| 154 | 16680 | 99028 | 
| 39 | **22 sphinx/config.py** | 486 | 500| 154 | 16834 | 99028 | 
| 40 | 23 sphinx/ext/intersphinx.py | 493 | 505| 145 | 16979 | 103693 | 
| 41 | 24 sphinx/builders/texinfo.py | 198 | 221| 242 | 17221 | 105719 | 
| 42 | **24 sphinx/builders/html/__init__.py** | 11 | 63| 443 | 17664 | 105719 | 
| 43 | 25 sphinx/ext/viewcode.py | 344 | 363| 216 | 17880 | 108813 | 
| 44 | 26 sphinx/writers/html5.py | 661 | 747| 686 | 18566 | 116087 | 
| 45 | 27 sphinx/domains/python.py | 1468 | 1482| 110 | 18676 | 128729 | 
| 46 | **27 sphinx/builders/latex/__init__.py** | 514 | 547| 404 | 19080 | 128729 | 
| 47 | 28 sphinx/builders/gettext.py | 289 | 305| 168 | 19248 | 131252 | 
| 48 | 28 sphinx/setup_command.py | 144 | 195| 440 | 19688 | 131252 | 
| 49 | **28 sphinx/application.py** | 13 | 57| 358 | 20046 | 131252 | 
| 50 | 29 sphinx/writers/manpage.py | 192 | 260| 531 | 20577 | 134740 | 
| 51 | 30 sphinx/search/__init__.py | 10 | 114| 766 | 21343 | 138806 | 
| 52 | 31 sphinx/__init__.py | 14 | 60| 476 | 21819 | 139356 | 
| 53 | 31 sphinx/setup_command.py | 14 | 91| 431 | 22250 | 139356 | 
| 54 | **31 sphinx/application.py** | 60 | 122| 494 | 22744 | 139356 | 
| 55 | 32 sphinx/ext/apidoc.py | 369 | 397| 374 | 23118 | 143590 | 
| 56 | 33 sphinx/transforms/__init__.py | 11 | 44| 231 | 23349 | 146752 | 
| 57 | **33 sphinx/builders/latex/__init__.py** | 42 | 97| 825 | 24174 | 146752 | 
| 58 | 34 sphinx/cmd/make_mode.py | 17 | 54| 532 | 24706 | 148454 | 
| 59 | 35 sphinx/writers/html.py | 721 | 819| 769 | 25475 | 156141 | 
| 60 | **35 sphinx/builders/latex/__init__.py** | 11 | 40| 315 | 25790 | 156141 | 
| **-> 61 <-** | **36 sphinx/util/i18n.py** | 231 | 264| 260 | 26050 | 158480 | 
| **-> 62 <-** | **37 sphinx/writers/latex.py** | 266 | 401| 1503 | 27553 | 177476 | 
| 63 | 38 sphinx/builders/linkcheck.py | 553 | 584| 325 | 27878 | 182283 | 
| 64 | 39 sphinx/writers/texinfo.py | 218 | 258| 433 | 28311 | 194613 | 


## Missing Patch Files

 * 1: sphinx/application.py
 * 2: sphinx/builders/html/__init__.py
 * 3: sphinx/builders/latex/__init__.py
 * 4: sphinx/builders/latex/util.py
 * 5: sphinx/config.py
 * 6: sphinx/environment/__init__.py
 * 7: sphinx/environment/collectors/asset.py
 * 8: sphinx/util/i18n.py
 * 9: sphinx/writers/latex.py

## Patch

```diff
diff --git a/sphinx/application.py b/sphinx/application.py
--- a/sphinx/application.py
+++ b/sphinx/application.py
@@ -266,7 +266,7 @@ def _init_i18n(self) -> None:
         """Load translated strings from the configured localedirs if enabled in
         the configuration.
         """
-        if self.config.language is None:
+        if self.config.language == 'en':
             self.translator, has_translation = locale.init([], None)
         else:
             logger.info(bold(__('loading translations [%s]... ') % self.config.language),
@@ -285,8 +285,7 @@ def _init_i18n(self) -> None:
             locale_dirs += [path.join(package_dir, 'locale')]
 
             self.translator, has_translation = locale.init(locale_dirs, self.config.language)
-            if has_translation or self.config.language == 'en':
-                # "en" never needs to be translated
+            if has_translation:
                 logger.info(__('done'))
             else:
                 logger.info(__('not available for built-in messages'))
diff --git a/sphinx/builders/html/__init__.py b/sphinx/builders/html/__init__.py
--- a/sphinx/builders/html/__init__.py
+++ b/sphinx/builders/html/__init__.py
@@ -326,7 +326,7 @@ def init_js_files(self) -> None:
             attrs.setdefault('priority', 800)  # User's JSs are loaded after extensions'
             self.add_js_file(filename, **attrs)
 
-        if self.config.language and self._get_translations_js():
+        if self._get_translations_js():
             self.add_js_file('translations.js')
 
     def add_js_file(self, filename: str, **kwargs: Any) -> None:
@@ -431,8 +431,6 @@ def prepare_writing(self, docnames: Set[str]) -> None:
         if self.search:
             from sphinx.search import IndexBuilder
             lang = self.config.html_search_language or self.config.language
-            if not lang:
-                lang = 'en'
             self.indexer = IndexBuilder(self.env, lang,
                                         self.config.html_search_options,
                                         self.config.html_search_scorer)
@@ -767,10 +765,9 @@ def create_pygments_style_file(self) -> None:
 
     def copy_translation_js(self) -> None:
         """Copy a JavaScript file for translations."""
-        if self.config.language is not None:
-            jsfile = self._get_translations_js()
-            if jsfile:
-                copyfile(jsfile, path.join(self.outdir, '_static', 'translations.js'))
+        jsfile = self._get_translations_js()
+        if jsfile:
+            copyfile(jsfile, path.join(self.outdir, '_static', 'translations.js'))
 
     def copy_stemmer_js(self) -> None:
         """Copy a JavaScript file for stemmer."""
diff --git a/sphinx/builders/latex/__init__.py b/sphinx/builders/latex/__init__.py
--- a/sphinx/builders/latex/__init__.py
+++ b/sphinx/builders/latex/__init__.py
@@ -170,9 +170,8 @@ def init_context(self) -> None:
         self.context.update(ADDITIONAL_SETTINGS.get(self.config.latex_engine, {}))
 
         # Add special settings for (latex_engine, language_code)
-        if self.config.language:
-            key = (self.config.latex_engine, self.config.language[:2])
-            self.context.update(ADDITIONAL_SETTINGS.get(key, {}))
+        key = (self.config.latex_engine, self.config.language[:2])
+        self.context.update(ADDITIONAL_SETTINGS.get(key, {}))
 
         # Apply user settings to context
         self.context.update(self.config.latex_elements)
@@ -203,7 +202,7 @@ def update_context(self) -> None:
 
     def init_babel(self) -> None:
         self.babel = ExtBabel(self.config.language, not self.context['babel'])
-        if self.config.language and not self.babel.is_supported_language():
+        if not self.babel.is_supported_language():
             # emit warning if specified language is invalid
             # (only emitting, nothing changed to processing)
             logger.warning(__('no Babel option known for language %r'),
@@ -232,12 +231,11 @@ def init_multilingual(self) -> None:
             self.context['classoptions'] += ',' + self.babel.get_language()
             # this branch is not taken for xelatex/lualatex if default settings
             self.context['multilingual'] = self.context['babel']
-            if self.config.language:
-                self.context['shorthandoff'] = SHORTHANDOFF
+            self.context['shorthandoff'] = SHORTHANDOFF
 
-                # Times fonts don't work with Cyrillic languages
-                if self.babel.uses_cyrillic() and 'fontpkg' not in self.config.latex_elements:
-                    self.context['fontpkg'] = ''
+            # Times fonts don't work with Cyrillic languages
+            if self.babel.uses_cyrillic() and 'fontpkg' not in self.config.latex_elements:
+                self.context['fontpkg'] = ''
         elif self.context['polyglossia']:
             self.context['classoptions'] += ',' + self.babel.get_language()
             options = self.babel.get_mainlanguage_options()
@@ -380,14 +378,10 @@ def copy_support_files(self) -> None:
         # configure usage of xindy (impacts Makefile and latexmkrc)
         # FIXME: convert this rather to a confval with suitable default
         #        according to language ? but would require extra documentation
-        if self.config.language:
-            xindy_lang_option = \
-                XINDY_LANG_OPTIONS.get(self.config.language[:2],
-                                       '-L general -C utf8 ')
-            xindy_cyrillic = self.config.language[:2] in XINDY_CYRILLIC_SCRIPTS
-        else:
-            xindy_lang_option = '-L english -C utf8 '
-            xindy_cyrillic = False
+        xindy_lang_option = XINDY_LANG_OPTIONS.get(self.config.language[:2],
+                                                   '-L general -C utf8 ')
+        xindy_cyrillic = self.config.language[:2] in XINDY_CYRILLIC_SCRIPTS
+
         context = {
             'latex_engine':      self.config.latex_engine,
             'xindy_use':         self.config.latex_use_xindy,
@@ -474,7 +468,7 @@ def default_latex_engine(config: Config) -> str:
     """ Better default latex_engine settings for specific languages. """
     if config.language == 'ja':
         return 'uplatex'
-    elif (config.language or '').startswith('zh'):
+    elif config.language.startswith('zh'):
         return 'xelatex'
     elif config.language == 'el':
         return 'xelatex'
diff --git a/sphinx/builders/latex/util.py b/sphinx/builders/latex/util.py
--- a/sphinx/builders/latex/util.py
+++ b/sphinx/builders/latex/util.py
@@ -20,7 +20,7 @@ def __init__(self, language_code: str, use_polyglossia: bool = False) -> None:
         self.language_code = language_code
         self.use_polyglossia = use_polyglossia
         self.supported = True
-        super().__init__(language_code or '')
+        super().__init__(language_code)
 
     def uses_cyrillic(self) -> bool:
         return self.language in self.cyrillic_languages
diff --git a/sphinx/config.py b/sphinx/config.py
--- a/sphinx/config.py
+++ b/sphinx/config.py
@@ -100,7 +100,7 @@ class Config:
         # the real default is locale-dependent
         'today_fmt': (None, 'env', [str]),
 
-        'language': (None, 'env', [str]),
+        'language': ('en', 'env', [str]),
         'locale_dirs': (['locales'], 'env', []),
         'figure_language_filename': ('{root}.{language}{ext}', 'env', [str]),
         'gettext_allow_fuzzy_translations': (False, 'gettext', []),
diff --git a/sphinx/environment/__init__.py b/sphinx/environment/__init__.py
--- a/sphinx/environment/__init__.py
+++ b/sphinx/environment/__init__.py
@@ -261,7 +261,7 @@ def _update_settings(self, config: Config) -> None:
         """Update settings by new config."""
         self.settings['input_encoding'] = config.source_encoding
         self.settings['trim_footnote_reference_space'] = config.trim_footnote_reference_space
-        self.settings['language_code'] = config.language or 'en'
+        self.settings['language_code'] = config.language
 
         # Allow to disable by 3rd party extension (workaround)
         self.settings.setdefault('smart_quotes', True)
diff --git a/sphinx/environment/collectors/asset.py b/sphinx/environment/collectors/asset.py
--- a/sphinx/environment/collectors/asset.py
+++ b/sphinx/environment/collectors/asset.py
@@ -64,18 +64,16 @@ def process_doc(self, app: Sphinx, doctree: nodes.document) -> None:
                 rel_imgpath, full_imgpath = app.env.relfn2path(imguri, docname)
                 node['uri'] = rel_imgpath
 
-                if app.config.language:
-                    # Search language-specific figures at first
-                    i18n_imguri = get_image_filename_for_language(imguri, app.env)
-                    _, full_i18n_imgpath = app.env.relfn2path(i18n_imguri, docname)
-                    self.collect_candidates(app.env, full_i18n_imgpath, candidates, node)
+                # Search language-specific figures at first
+                i18n_imguri = get_image_filename_for_language(imguri, app.env)
+                _, full_i18n_imgpath = app.env.relfn2path(i18n_imguri, docname)
+                self.collect_candidates(app.env, full_i18n_imgpath, candidates, node)
 
                 self.collect_candidates(app.env, full_imgpath, candidates, node)
             else:
-                if app.config.language:
-                    # substitute imguri by figure_language_filename
-                    # (ex. foo.png -> foo.en.png)
-                    imguri = search_image_for_language(imguri, app.env)
+                # substitute imguri by figure_language_filename
+                # (ex. foo.png -> foo.en.png)
+                imguri = search_image_for_language(imguri, app.env)
 
                 # Update `node['uri']` to a relative path from srcdir
                 # from a relative path from current document.
diff --git a/sphinx/util/i18n.py b/sphinx/util/i18n.py
--- a/sphinx/util/i18n.py
+++ b/sphinx/util/i18n.py
@@ -10,14 +10,16 @@
 
 import os
 import re
+import warnings
 from datetime import datetime, timezone
 from os import path
-from typing import TYPE_CHECKING, Callable, Generator, List, NamedTuple, Optional, Tuple, Union
+from typing import TYPE_CHECKING, Callable, Generator, List, NamedTuple, Tuple, Union
 
 import babel.dates
 from babel.messages.mofile import write_mo
 from babel.messages.pofile import read_po
 
+from sphinx.deprecation import RemovedInSphinx70Warning
 from sphinx.errors import SphinxError
 from sphinx.locale import __
 from sphinx.util import logging
@@ -173,9 +175,11 @@ def docname_to_domain(docname: str, compaction: Union[bool, str]) -> str:
 date_format_re = re.compile('(%s)' % '|'.join(date_format_mappings))
 
 
-def babel_format_date(date: datetime, format: str, locale: Optional[str],
+def babel_format_date(date: datetime, format: str, locale: str,
                       formatter: Callable = babel.dates.format_date) -> str:
     if locale is None:
+        warnings.warn('The locale argument for babel_format_date() becomes required.',
+                      RemovedInSphinx70Warning)
         locale = 'en'
 
     # Check if we have the tzinfo attribute. If not we cannot do any time
@@ -194,7 +198,7 @@ def babel_format_date(date: datetime, format: str, locale: Optional[str],
         return format
 
 
-def format_date(format: str, date: datetime = None, language: Optional[str] = None) -> str:
+def format_date(format: str, date: datetime = None, language: str = None) -> str:
     if date is None:
         # If time is not specified, try to use $SOURCE_DATE_EPOCH variable
         # See https://wiki.debian.org/ReproducibleBuilds/TimestampsProposal
@@ -204,6 +208,11 @@ def format_date(format: str, date: datetime = None, language: Optional[str] = No
         else:
             date = datetime.now(timezone.utc).astimezone()
 
+    if language is None:
+        warnings.warn('The language argument for format_date() becomes required.',
+                      RemovedInSphinx70Warning)
+        language = 'en'
+
     result = []
     tokens = date_format_re.split(format)
     for token in tokens:
@@ -229,9 +238,6 @@ def format_date(format: str, date: datetime = None, language: Optional[str] = No
 
 
 def get_image_filename_for_language(filename: str, env: "BuildEnvironment") -> str:
-    if not env.config.language:
-        return filename
-
     filename_format = env.config.figure_language_filename
     d = dict()
     d['root'], d['ext'] = path.splitext(filename)
@@ -252,9 +258,6 @@ def get_image_filename_for_language(filename: str, env: "BuildEnvironment") -> s
 
 
 def search_image_for_language(filename: str, env: "BuildEnvironment") -> str:
-    if not env.config.language:
-        return filename
-
     translated = get_image_filename_for_language(filename, env)
     _, abspath = env.relfn2path(translated)
     if path.exists(abspath):
diff --git a/sphinx/writers/latex.py b/sphinx/writers/latex.py
--- a/sphinx/writers/latex.py
+++ b/sphinx/writers/latex.py
@@ -333,7 +333,7 @@ def __init__(self, document: nodes.document, builder: "LaTeXBuilder",
         if self.config.numfig and self.config.math_numfig:
             sphinxpkgoptions.append('mathnumfig')
 
-        if (self.config.language not in {None, 'en', 'ja'} and
+        if (self.config.language not in {'en', 'ja'} and
                 'fncychap' not in self.config.latex_elements):
             # use Sonny style if any language specified (except English)
             self.elements['fncychap'] = (r'\usepackage[Sonny]{fncychap}' + CR +
@@ -341,7 +341,7 @@ def __init__(self, document: nodes.document, builder: "LaTeXBuilder",
                                          r'\ChTitleVar{\Large\normalfont\sffamily}')
 
         self.babel = self.builder.babel
-        if self.config.language and not self.babel.is_supported_language():
+        if not self.babel.is_supported_language():
             # emit warning if specified language is invalid
             # (only emitting, nothing changed to processing)
             logger.warning(__('no Babel option known for language %r'),

```

## Test Patch

```diff
diff --git a/tests/test_build_latex.py b/tests/test_build_latex.py
--- a/tests/test_build_latex.py
+++ b/tests/test_build_latex.py
@@ -528,7 +528,7 @@ def test_babel_with_no_language_settings(app, status, warning):
     assert '\\usepackage[Bjarne]{fncychap}' in result
     assert ('\\addto\\captionsenglish{\\renewcommand{\\contentsname}{Table of content}}\n'
             in result)
-    assert '\\shorthandoff' not in result
+    assert '\\shorthandoff{"}' in result
 
     # sphinxmessages.sty
     result = (app.outdir / 'sphinxmessages.sty').read_text()
diff --git a/tests/test_util_i18n.py b/tests/test_util_i18n.py
--- a/tests/test_util_i18n.py
+++ b/tests/test_util_i18n.py
@@ -98,15 +98,6 @@ def test_format_date():
 def test_get_filename_for_language(app):
     app.env.temp_data['docname'] = 'index'
 
-    # language is None
-    app.env.config.language = None
-    assert app.env.config.language is None
-    assert i18n.get_image_filename_for_language('foo.png', app.env) == 'foo.png'
-    assert i18n.get_image_filename_for_language('foo.bar.png', app.env) == 'foo.bar.png'
-    assert i18n.get_image_filename_for_language('subdir/foo.png', app.env) == 'subdir/foo.png'
-    assert i18n.get_image_filename_for_language('../foo.png', app.env) == '../foo.png'
-    assert i18n.get_image_filename_for_language('foo', app.env) == 'foo'
-
     # language is en
     app.env.config.language = 'en'
     assert i18n.get_image_filename_for_language('foo.png', app.env) == 'foo.en.png'
@@ -115,15 +106,6 @@ def test_get_filename_for_language(app):
     assert i18n.get_image_filename_for_language('../foo.png', app.env) == '../foo.en.png'
     assert i18n.get_image_filename_for_language('foo', app.env) == 'foo.en'
 
-    # modify figure_language_filename and language is None
-    app.env.config.language = None
-    app.env.config.figure_language_filename = 'images/{language}/{root}{ext}'
-    assert i18n.get_image_filename_for_language('foo.png', app.env) == 'foo.png'
-    assert i18n.get_image_filename_for_language('foo.bar.png', app.env) == 'foo.bar.png'
-    assert i18n.get_image_filename_for_language('subdir/foo.png', app.env) == 'subdir/foo.png'
-    assert i18n.get_image_filename_for_language('../foo.png', app.env) == '../foo.png'
-    assert i18n.get_image_filename_for_language('foo', app.env) == 'foo'
-
     # modify figure_language_filename and language is 'en'
     app.env.config.language = 'en'
     app.env.config.figure_language_filename = 'images/{language}/{root}{ext}'

```


## Code snippets

### 1 - sphinx/cmd/quickstart.py:

Start line: 268, End line: 326

```python
def ask_user(d: Dict) -> None:
    # ... other code

    if 'language' not in d:
        print()
        print(__('If the documents are to be written in a language other than English,\n'
                 'you can select a language here by its language code. Sphinx will then\n'
                 'translate text that it generates into that language.\n'
                 '\n'
                 'For a list of supported codes, see\n'
                 'https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-language.'))  # NOQA
        d['language'] = do_prompt(__('Project language'), 'en')
        if d['language'] == 'en':
            d['language'] = None

    if 'suffix' not in d:
        print()
        print(__('The file name suffix for source files. Commonly, this is either ".txt"\n'
                 'or ".rst". Only files with this suffix are considered documents.'))
        d['suffix'] = do_prompt(__('Source file suffix'), '.rst', suffix)

    if 'master' not in d:
        print()
        print(__('One document is special in that it is considered the top node of the\n'
                 '"contents tree", that is, it is the root of the hierarchical structure\n'
                 'of the documents. Normally, this is "index", but if your "index"\n'
                 'document is a custom template, you can also set this to another filename.'))
        d['master'] = do_prompt(__('Name of your master document (without suffix)'), 'index')

    while path.isfile(path.join(d['path'], d['master'] + d['suffix'])) or \
            path.isfile(path.join(d['path'], 'source', d['master'] + d['suffix'])):
        print()
        print(bold(__('Error: the master file %s has already been found in the '
                      'selected root path.') % (d['master'] + d['suffix'])))
        print(__('sphinx-quickstart will not overwrite the existing file.'))
        print()
        d['master'] = do_prompt(__('Please enter a new file name, or rename the '
                                   'existing file and press Enter'), d['master'])

    if 'extensions' not in d:
        print(__('Indicate which of the following Sphinx extensions should be enabled:'))
        d['extensions'] = []
        for name, description in EXTENSIONS.items():
            if do_prompt('%s: %s (y/n)' % (name, description), 'n', boolean):
                d['extensions'].append('sphinx.ext.%s' % name)

        # Handle conflicting options
        if {'sphinx.ext.imgmath', 'sphinx.ext.mathjax'}.issubset(d['extensions']):
            print(__('Note: imgmath and mathjax cannot be enabled at the same time. '
                     'imgmath has been deselected.'))
            d['extensions'].remove('sphinx.ext.imgmath')

    if 'makefile' not in d:
        print()
        print(__('A Makefile and a Windows command file can be generated for you so that you\n'
                 'only have to run e.g. `make html\' instead of invoking sphinx-build\n'
                 'directly.'))
        d['makefile'] = do_prompt(__('Create Makefile? (y/n)'), 'y', boolean)

    if 'batchfile' not in d:
        d['batchfile'] = do_prompt(__('Create Windows command file? (y/n)'), 'y', boolean)
    print()
```
### 2 - sphinx/cmd/quickstart.py:

Start line: 191, End line: 266

```python
def ask_user(d: Dict) -> None:

    print(bold(__('Welcome to the Sphinx %s quickstart utility.')) % __display_version__)
    print()
    print(__('Please enter values for the following settings (just press Enter to\n'
             'accept a default value, if one is given in brackets).'))

    if 'path' in d:
        print()
        print(bold(__('Selected root path: %s')) % d['path'])
    else:
        print()
        print(__('Enter the root path for documentation.'))
        d['path'] = do_prompt(__('Root path for the documentation'), '.', is_path)

    while path.isfile(path.join(d['path'], 'conf.py')) or \
            path.isfile(path.join(d['path'], 'source', 'conf.py')):
        print()
        print(bold(__('Error: an existing conf.py has been found in the '
                      'selected root path.')))
        print(__('sphinx-quickstart will not overwrite existing Sphinx projects.'))
        print()
        d['path'] = do_prompt(__('Please enter a new root path (or just Enter to exit)'),
                              '', is_path_or_empty)
        if not d['path']:
            sys.exit(1)

    if 'sep' not in d:
        print()
        print(__('You have two options for placing the build directory for Sphinx output.\n'
                 'Either, you use a directory "_build" within the root path, or you separate\n'
                 '"source" and "build" directories within the root path.'))
        d['sep'] = do_prompt(__('Separate source and build directories (y/n)'), 'n', boolean)

    if 'dot' not in d:
        print()
        print(__('Inside the root directory, two more directories will be created; "_templates"\n'      # NOQA
                 'for custom HTML templates and "_static" for custom stylesheets and other static\n'    # NOQA
                 'files. You can enter another prefix (such as ".") to replace the underscore.'))       # NOQA
        d['dot'] = do_prompt(__('Name prefix for templates and static dir'), '_', ok)

    if 'project' not in d:
        print()
        print(__('The project name will occur in several places in the built documentation.'))
        d['project'] = do_prompt(__('Project name'))
    if 'author' not in d:
        d['author'] = do_prompt(__('Author name(s)'))

    if 'version' not in d:
        print()
        print(__('Sphinx has the notion of a "version" and a "release" for the\n'
                 'software. Each version can have multiple releases. For example, for\n'
                 'Python the version is something like 2.5 or 3.0, while the release is\n'
                 'something like 2.5.1 or 3.0a1. If you don\'t need this dual structure,\n'
                 'just set both to the same value.'))
        d['version'] = do_prompt(__('Project version'), '', allow_empty)
    if 'release' not in d:
        d['release'] = do_prompt(__('Project release'), d['version'], allow_empty)
    # ... other code
```
### 3 - sphinx/locale/__init__.py:

Start line: 241, End line: 269

```python
# A shortcut for sphinx-core
#: Translation function for messages on documentation (menu, labels, themes and so on).
#: This function follows :confval:`language` setting.
_ = get_translation('sphinx')
#: Translation function for console messages
#: This function follows locale setting (`LC_ALL`, `LC_MESSAGES` and so on).
__ = get_translation('sphinx', 'console')


# labels
admonitionlabels = {
    'attention': _('Attention'),
    'caution':   _('Caution'),
    'danger':    _('Danger'),
    'error':     _('Error'),
    'hint':      _('Hint'),
    'important': _('Important'),
    'note':      _('Note'),
    'seealso':   _('See also'),
    'tip':       _('Tip'),
    'warning':   _('Warning'),
}

# Moved to sphinx.directives.other (will be overridden later)
versionlabels: Dict[str, str] = {}

# Moved to sphinx.domains.python (will be overridden later)
pairindextypes: Dict[str, str] = {}
```
### 4 - sphinx/cmd/quickstart.py:

Start line: 11, End line: 119

```python
import argparse
import locale
import os
import sys
import time
from collections import OrderedDict
from os import path
from typing import Any, Callable, Dict, List, Union

# try to import readline, unix specific enhancement
try:
    import readline
    if readline.__doc__ and 'libedit' in readline.__doc__:
        readline.parse_and_bind("bind ^I rl_complete")
        USE_LIBEDIT = True
    else:
        readline.parse_and_bind("tab: complete")
        USE_LIBEDIT = False
except ImportError:
    readline = None
    USE_LIBEDIT = False

from docutils.utils import column_width

import sphinx.locale
from sphinx import __display_version__, package_dir
from sphinx.locale import __
from sphinx.util.console import bold, color_terminal, colorize, nocolor, red  # type: ignore
from sphinx.util.osutil import ensuredir
from sphinx.util.template import SphinxRenderer

EXTENSIONS = OrderedDict([
    ('autodoc', __('automatically insert docstrings from modules')),
    ('doctest', __('automatically test code snippets in doctest blocks')),
    ('intersphinx', __('link between Sphinx documentation of different projects')),
    ('todo', __('write "todo" entries that can be shown or hidden on build')),
    ('coverage', __('checks for documentation coverage')),
    ('imgmath', __('include math, rendered as PNG or SVG images')),
    ('mathjax', __('include math, rendered in the browser by MathJax')),
    ('ifconfig', __('conditional inclusion of content based on config values')),
    ('viewcode', __('include links to the source code of documented Python objects')),
    ('githubpages', __('create .nojekyll file to publish the document on GitHub pages')),
])

DEFAULTS = {
    'path': '.',
    'sep': False,
    'dot': '_',
    'language': None,
    'suffix': '.rst',
    'master': 'index',
    'makefile': True,
    'batchfile': True,
}

PROMPT_PREFIX = '> '

if sys.platform == 'win32':
    # On Windows, show questions as bold because of color scheme of PowerShell (refs: #5294).
    COLOR_QUESTION = 'bold'
else:
    COLOR_QUESTION = 'purple'


# function to get input from terminal -- overridden by the test suite
def term_input(prompt: str) -> str:
    if sys.platform == 'win32':
        # Important: On windows, readline is not enabled by default.  In these
        #            environment, escape sequences have been broken.  To avoid the
        #            problem, quickstart uses ``print()`` to show prompt.
        print(prompt, end='')
        return input('')
    else:
        return input(prompt)


class ValidationError(Exception):
    """Raised for validation errors."""


def is_path(x: str) -> str:
    x = path.expanduser(x)
    if not path.isdir(x):
        raise ValidationError(__("Please enter a valid path name."))
    return x


def is_path_or_empty(x: str) -> str:
    if x == '':
        return x
    return is_path(x)


def allow_empty(x: str) -> str:
    return x


def nonempty(x: str) -> str:
    if not x:
        raise ValidationError(__("Please enter some text."))
    return x


def choice(*l: str) -> Callable[[str], str]:
    def val(x: str) -> str:
        if x not in l:
            raise ValidationError(__('Please enter one of %s.') % ', '.join(l))
        return x
    return val
```
### 5 - setup.py:

Start line: 1, End line: 76

```python
import os
import sys
from io import StringIO

from setuptools import find_packages, setup

import sphinx

with open('README.rst') as f:
    long_desc = f.read()

if sys.version_info < (3, 6):
    print('ERROR: Sphinx requires at least Python 3.6 to run.')
    sys.exit(1)

install_requires = [
    'sphinxcontrib-applehelp',
    'sphinxcontrib-devhelp',
    'sphinxcontrib-jsmath',
    'sphinxcontrib-htmlhelp>=2.0.0',
    'sphinxcontrib-serializinghtml>=1.1.5',
    'sphinxcontrib-qthelp',
    'Jinja2>=2.3',
    'Pygments>=2.0',
    'docutils>=0.14,<0.18',
    'snowballstemmer>=1.1',
    'babel>=1.3',
    'alabaster>=0.7,<0.8',
    'imagesize',
    'requests>=2.5.0',
    'packaging',
    "importlib-metadata>=4.4; python_version < '3.10'",
]

extras_require = {
    # Environment Marker works for wheel 0.24 or later
    ':sys_platform=="win32"': [
        'colorama>=0.3.5',
    ],
    'docs': [
        'sphinxcontrib-websupport',
    ],
    'lint': [
        'flake8>=3.5.0',
        'isort',
        'mypy>=0.931',
        'docutils-stubs',
        "types-typed-ast",
        "types-requests",
    ],
    'test': [
        'pytest',
        'pytest-cov',
        'html5lib',
        "typed_ast; python_version < '3.8'",
        'cython',
    ],
}

# Provide a "compile_catalog" command that also creates the translated
# JavaScript files if Babel is available.

cmdclass = {}


class Tee:
    def __init__(self, stream):
        self.stream = stream
        self.buffer = StringIO()

    def write(self, s):
        self.stream.write(s)
        self.buffer.write(s)

    def flush(self):
        self.stream.flush()
```
### 6 - sphinx/builders/html/__init__.py:

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
### 7 - setup.py:

Start line: 174, End line: 251

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
### 8 - sphinx/builders/latex/__init__.py:

Start line: 451, End line: 511

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
### 9 - sphinx/util/docutils.py:

Start line: 146, End line: 170

```python
@contextmanager
def using_user_docutils_conf(confdir: Optional[str]) -> Generator[None, None, None]:
    """Let docutils know the location of ``docutils.conf`` for Sphinx."""
    try:
        docutilsconfig = os.environ.get('DOCUTILSCONFIG', None)
        if confdir:
            os.environ['DOCUTILSCONFIG'] = path.join(path.abspath(confdir), 'docutils.conf')

        yield
    finally:
        if docutilsconfig is None:
            os.environ.pop('DOCUTILSCONFIG', None)
        else:
            os.environ['DOCUTILSCONFIG'] = docutilsconfig


@contextmanager
def patch_docutils(confdir: Optional[str] = None) -> Generator[None, None, None]:
    """Patch to docutils temporarily."""
    with patched_get_language(), using_user_docutils_conf(confdir):
        yield


class ElementLookupError(Exception):
    pass
```
### 10 - sphinx/cmd/quickstart.py:

Start line: 544, End line: 611

```python
def main(argv: List[str] = sys.argv[1:]) -> int:
    sphinx.locale.setlocale(locale.LC_ALL, '')
    sphinx.locale.init_console(os.path.join(package_dir, 'locale'), 'sphinx')

    if not color_terminal():
        nocolor()

    # parse options
    parser = get_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as err:
        return err.code

    d = vars(args)
    # delete None or False value
    d = {k: v for k, v in d.items() if v is not None}

    # handle use of CSV-style extension values
    d.setdefault('extensions', [])
    for ext in d['extensions'][:]:
        if ',' in ext:
            d['extensions'].remove(ext)
            d['extensions'].extend(ext.split(','))

    try:
        if 'quiet' in d:
            if not {'project', 'author'}.issubset(d):
                print(__('"quiet" is specified, but any of "project" or '
                         '"author" is not specified.'))
                return 1

        if {'quiet', 'project', 'author'}.issubset(d):
            # quiet mode with all required params satisfied, use default
            d.setdefault('version', '')
            d.setdefault('release', d['version'])
            d2 = DEFAULTS.copy()
            d2.update(d)
            d = d2

            if not valid_dir(d):
                print()
                print(bold(__('Error: specified path is not a directory, or sphinx'
                              ' files already exist.')))
                print(__('sphinx-quickstart only generate into a empty directory.'
                         ' Please specify a new root path.'))
                return 1
        else:
            ask_user(d)
    except (KeyboardInterrupt, EOFError):
        print()
        print('[Interrupted.]')
        return 130  # 128 + SIGINT

    for variable in d.get('variables', []):
        try:
            name, value = variable.split('=')
            d[name] = value
        except ValueError:
            print(__('Invalid template variable: %s') % variable)

    generate(d, overwrite=False, templatedir=args.templatedir)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
```
### 15 - sphinx/environment/__init__.py:

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
    'rfc_base_url': 'https://datatracker.ietf.org/doc/html/',
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
### 17 - sphinx/config.py:

Start line: 91, End line: 151

```python
class Config:

    config_values: Dict[str, Tuple] = {
        # general options
        'project': ('Python', 'env', []),
        'author': ('unknown', 'env', []),
        'project_copyright': ('', 'html', [str]),
        'copyright': (lambda c: c.project_copyright, 'html', [str]),
        'version': ('', 'env', []),
        'release': ('', 'env', []),
        'today': ('', 'env', []),
        # the real default is locale-dependent
        'today_fmt': (None, 'env', [str]),

        'language': (None, 'env', [str]),
        'locale_dirs': (['locales'], 'env', []),
        'figure_language_filename': ('{root}.{language}{ext}', 'env', [str]),
        'gettext_allow_fuzzy_translations': (False, 'gettext', []),

        'master_doc': ('index', 'env', []),
        'root_doc': (lambda config: config.master_doc, 'env', []),
        'source_suffix': ({'.rst': 'restructuredtext'}, 'env', Any),
        'source_encoding': ('utf-8-sig', 'env', []),
        'exclude_patterns': ([], 'env', []),
        'default_role': (None, 'env', [str]),
        'add_function_parentheses': (True, 'env', []),
        'add_module_names': (True, 'env', []),
        'trim_footnote_reference_space': (False, 'env', []),
        'show_authors': (False, 'env', []),
        'pygments_style': (None, 'html', [str]),
        'highlight_language': ('default', 'env', []),
        'highlight_options': ({}, 'env', []),
        'templates_path': ([], 'html', []),
        'template_bridge': (None, 'html', [str]),
        'keep_warnings': (False, 'env', []),
        'suppress_warnings': ([], 'env', []),
        'modindex_common_prefix': ([], 'html', []),
        'rst_epilog': (None, 'env', [str]),
        'rst_prolog': (None, 'env', [str]),
        'trim_doctest_flags': (True, 'env', []),
        'primary_domain': ('py', 'env', [NoneType]),
        'needs_sphinx': (None, None, [str]),
        'needs_extensions': ({}, None, []),
        'manpages_url': (None, 'env', []),
        'nitpicky': (False, None, []),
        'nitpick_ignore': ([], None, []),
        'nitpick_ignore_regex': ([], None, []),
        'numfig': (False, 'env', []),
        'numfig_secnum_depth': (1, 'env', []),
        'numfig_format': ({}, 'env', []),  # will be initialized in init_numfig_format()

        'math_number_all': (False, 'env', []),
        'math_eqref_format': (None, 'env', [str]),
        'math_numfig': (True, 'env', []),
        'tls_verify': (True, 'env', []),
        'tls_cacerts': (None, 'env', []),
        'user_agent': (None, 'env', [str]),
        'smartquotes': (True, 'env', []),
        'smartquotes_action': ('qDe', 'env', []),
        'smartquotes_excludes': ({'languages': ['ja'],
                                  'builders': ['man', 'text']},
                                 'env', []),
    }
```
### 18 - sphinx/application.py:

Start line: 1154, End line: 1168

```python
class Sphinx:

    def add_search_language(self, cls: Any) -> None:
        """Register a new language for the HTML search index.

        Add *cls*, which must be a subclass of
        :class:`sphinx.search.SearchLanguage`, as a support language for
        building the HTML full-text search index.  The class must have a *lang*
        attribute that indicates the language it should be used for.  See
        :confval:`html_search_language`.

        .. versionadded:: 1.1
        """
        logger.debug('[app] adding search language: %r', cls)
        from sphinx.search import SearchLanguage, languages
        assert issubclass(cls, SearchLanguage)
        languages[cls.lang] = cls
```
### 21 - sphinx/builders/latex/util.py:

Start line: 31, End line: 57

```python
class ExtBabel(Babel):

    def language_name(self, language_code: str) -> str:
        language = super().language_name(language_code)
        if language == 'ngerman' and self.use_polyglossia:
            # polyglossia calls new orthography (Neue Rechtschreibung) as
            # german (with new spelling option).
            return 'german'
        elif language:
            return language
        elif language_code.startswith('zh'):
            return 'english'  # fallback to english (behaves like supported)
        else:
            self.supported = False
            return 'english'  # fallback to english

    def get_mainlanguage_options(self) -> Optional[str]:
        """Return options for polyglossia's ``\\setmainlanguage``."""
        if self.use_polyglossia is False:
            return None
        elif self.language == 'german':
            language = super().language_name(self.language_code)
            if language == 'ngerman':
                return 'spelling=new'
            else:
                return 'spelling=old'
        else:
            return None
```
### 29 - sphinx/environment/__init__.py:

Start line: 445, End line: 523

```python
class BuildEnvironment:

    def check_dependents(self, app: "Sphinx", already: Set[str]) -> Generator[str, None, None]:
        to_rewrite: List[str] = []
        for docnames in self.events.emit('env-get-updated', self):
            to_rewrite.extend(docnames)
        for docname in set(to_rewrite):
            if docname not in already:
                yield docname

    # --------- SINGLE FILE READING --------------------------------------------

    def prepare_settings(self, docname: str) -> None:
        """Prepare to set up environment for reading."""
        self.temp_data['docname'] = docname
        # defaults to the global default, but can be re-set in a document
        self.temp_data['default_role'] = self.config.default_role
        self.temp_data['default_domain'] = \
            self.domains.get(self.config.primary_domain)

    # utilities to use while reading a document

    @property
    def docname(self) -> str:
        """Returns the docname of the document currently being parsed."""
        return self.temp_data['docname']

    def new_serialno(self, category: str = '') -> int:
        """Return a serial number, e.g. for index entry targets.

        The number is guaranteed to be unique in the current document.
        """
        key = category + 'serialno'
        cur = self.temp_data.get(key, 0)
        self.temp_data[key] = cur + 1
        return cur

    def note_dependency(self, filename: str) -> None:
        """Add *filename* as a dependency of the current document.

        This means that the document will be rebuilt if this file changes.

        *filename* should be absolute or relative to the source directory.
        """
        self.dependencies[self.docname].add(filename)

    def note_included(self, filename: str) -> None:
        """Add *filename* as a included from other document.

        This means the document is not orphaned.

        *filename* should be absolute or relative to the source directory.
        """
        self.included[self.docname].add(self.path2doc(filename))

    def note_reread(self) -> None:
        """Add the current document to the list of documents that will
        automatically be re-read at the next build.
        """
        self.reread_always.add(self.docname)

    def get_domain(self, domainname: str) -> Domain:
        """Return the domain instance with the specified name.

        Raises an ExtensionError if the domain is not registered.
        """
        try:
            return self.domains[domainname]
        except KeyError as exc:
            raise ExtensionError(__('Domain %r is not registered') % domainname) from exc

    # --------- RESOLVING REFERENCES AND TOCTREES ------------------------------

    def get_doctree(self, docname: str) -> nodes.document:
        """Read the doctree for a file from the pickle and return it."""
        filename = path.join(self.doctreedir, docname + '.doctree')
        with open(filename, 'rb') as f:
            doctree = pickle.load(f)
        doctree.settings.env = self
        doctree.reporter = LoggingReporter(self.doc2path(docname))
        return doctree
```
### 30 - sphinx/application.py:

Start line: 1284, End line: 1300

```python
class Sphinx:

    def set_html_assets_policy(self, policy):
        """Set the policy to include assets in HTML pages.

        - always: include the assets in all the pages
        - per_page: include the assets only in pages where they are used

        .. versionadded: 4.1
        """
        if policy not in ('always', 'per_page'):
            raise ValueError('policy %s is not supported' % policy)
        self.registry.html_assets_policy = policy

    @property
    def html_themes(self) -> Dict[str, str]:
        warnings.warn('app.html_themes is deprecated.',
                      RemovedInSphinx60Warning)
        return self.registry.html_themes
```
### 33 - sphinx/application.py:

Start line: 294, End line: 307

```python
class Sphinx:

    def _init_env(self, freshenv: bool) -> None:
        filename = path.join(self.doctreedir, ENV_PICKLE_FILENAME)
        if freshenv or not os.path.exists(filename):
            self.env = BuildEnvironment(self)
            self.env.find_files(self.config, self.builder)
        else:
            try:
                with progress_message(__('loading pickled environment')):
                    with open(filename, 'rb') as f:
                        self.env = pickle.load(f)
                        self.env.setup(self)
            except Exception as err:
                logger.info(__('failed: %s'), err)
                self._init_env(freshenv=True)
```
### 34 - sphinx/config.py:

Start line: 464, End line: 483

```python
def check_primary_domain(app: "Sphinx", config: Config) -> None:
    primary_domain = config.primary_domain
    if primary_domain and not app.registry.has_domain(primary_domain):
        logger.warning(__('primary_domain %r not found, ignored.'), primary_domain)
        config.primary_domain = None  # type: ignore


def check_root_doc(app: "Sphinx", env: "BuildEnvironment", added: Set[str],
                   changed: Set[str], removed: Set[str]) -> Set[str]:
    """Adjust root_doc to 'contents' to support an old project which does not have
    any root_doc setting.
    """
    if (app.config.root_doc == 'index' and
            'index' not in app.project.docnames and
            'contents' in app.project.docnames):
        logger.warning(__('Since v2.0, Sphinx uses "index" as root_doc by default. '
                          'Please add "root_doc = \'contents\'" to your conf.py.'))
        app.config.root_doc = "contents"  # type: ignore

    return changed
```
### 39 - sphinx/config.py:

Start line: 486, End line: 500

```python
def setup(app: "Sphinx") -> Dict[str, Any]:
    app.connect('config-inited', convert_source_suffix, priority=800)
    app.connect('config-inited', convert_highlight_options, priority=800)
    app.connect('config-inited', init_numfig_format, priority=800)
    app.connect('config-inited', correct_copyright_year, priority=800)
    app.connect('config-inited', check_confval_types, priority=800)
    app.connect('config-inited', check_primary_domain, priority=800)
    app.connect('env-get-outdated', check_root_doc)

    return {
        'version': 'builtin',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
```
### 42 - sphinx/builders/html/__init__.py:

Start line: 11, End line: 63

```python
import html
import os
import posixpath
import re
import sys
from datetime import datetime
from os import path
from typing import IO, Any, Dict, Iterable, Iterator, List, Set, Tuple, Type
from urllib.parse import quote

from docutils import nodes
from docutils.core import publish_parts
from docutils.frontend import OptionParser
from docutils.io import DocTreeInput, StringOutput
from docutils.nodes import Node
from docutils.utils import relative_path

from sphinx import __display_version__, package_dir
from sphinx import version_info as sphinx_version
from sphinx.application import Sphinx
from sphinx.builders import Builder
from sphinx.config import ENUM, Config
from sphinx.domains import Domain, Index, IndexEntry
from sphinx.environment.adapters.asset import ImageAdapter
from sphinx.environment.adapters.indexentries import IndexEntries
from sphinx.environment.adapters.toctree import TocTree
from sphinx.errors import ConfigError, ThemeError
from sphinx.highlighting import PygmentsBridge
from sphinx.locale import _, __
from sphinx.search import js_index
from sphinx.theming import HTMLThemeFactory
from sphinx.util import isurl, logging, md5, progress_message, status_iterator
from sphinx.util.docutils import is_html5_writer_available, new_document
from sphinx.util.fileutil import copy_asset
from sphinx.util.i18n import format_date
from sphinx.util.inventory import InventoryFile
from sphinx.util.matching import DOTFILES, Matcher, patmatch
from sphinx.util.osutil import copyfile, ensuredir, os_path, relative_uri
from sphinx.util.tags import Tags
from sphinx.writers.html import HTMLTranslator, HTMLWriter

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
### 46 - sphinx/builders/latex/__init__.py:

Start line: 514, End line: 547

```python
def setup(app: Sphinx) -> Dict[str, Any]:
    app.setup_extension('sphinx.builders.latex.transforms')

    app.add_builder(LaTeXBuilder)
    app.connect('config-inited', validate_config_values, priority=800)
    app.connect('config-inited', validate_latex_theme_options, priority=800)
    app.connect('builder-inited', install_packages_for_ja)

    app.add_config_value('latex_engine', default_latex_engine, None,
                         ENUM('pdflatex', 'xelatex', 'lualatex', 'platex', 'uplatex'))
    app.add_config_value('latex_documents', default_latex_documents, None)
    app.add_config_value('latex_logo', None, None, [str])
    app.add_config_value('latex_appendices', [], None)
    app.add_config_value('latex_use_latex_multicolumn', False, None)
    app.add_config_value('latex_use_xindy', default_latex_use_xindy, None, [bool])
    app.add_config_value('latex_toplevel_sectioning', None, None,
                         ENUM(None, 'part', 'chapter', 'section'))
    app.add_config_value('latex_domain_indices', True, None, [list])
    app.add_config_value('latex_show_urls', 'no', None)
    app.add_config_value('latex_show_pagerefs', False, None)
    app.add_config_value('latex_elements', {}, None)
    app.add_config_value('latex_additional_files', [], None)
    app.add_config_value('latex_theme', 'manual', None, [str])
    app.add_config_value('latex_theme_options', {}, None)
    app.add_config_value('latex_theme_path', [], None, [list])

    app.add_config_value('latex_docclass', default_latex_docclass, None)

    return {
        'version': 'builtin',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
```
### 49 - sphinx/application.py:

Start line: 13, End line: 57

```python
import os
import pickle
import sys
import warnings
from collections import deque
from io import StringIO
from os import path
from typing import IO, TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Type, Union

from docutils import nodes
from docutils.nodes import Element, TextElement
from docutils.parsers import Parser
from docutils.parsers.rst import Directive, roles
from docutils.transforms import Transform
from pygments.lexer import Lexer

import sphinx
from sphinx import locale, package_dir
from sphinx.config import Config
from sphinx.deprecation import RemovedInSphinx60Warning
from sphinx.domains import Domain, Index
from sphinx.environment import BuildEnvironment
from sphinx.environment.collectors import EnvironmentCollector
from sphinx.errors import ApplicationError, ConfigError, VersionRequirementError
from sphinx.events import EventManager
from sphinx.extension import Extension
from sphinx.highlighting import lexer_classes
from sphinx.locale import __
from sphinx.project import Project
from sphinx.registry import SphinxComponentRegistry
from sphinx.roles import XRefRole
from sphinx.theming import Theme
from sphinx.util import docutils, logging, progress_message
from sphinx.util.build_phase import BuildPhase
from sphinx.util.console import bold  # type: ignore
from sphinx.util.i18n import CatalogRepository
from sphinx.util.logging import prefixed_warnings
from sphinx.util.osutil import abspath, ensuredir, relpath
from sphinx.util.tags import Tags
from sphinx.util.typing import RoleFunction, TitleGetter

if TYPE_CHECKING:
    from docutils.nodes import Node  # NOQA

    from sphinx.builders import Builder
```
### 54 - sphinx/application.py:

Start line: 60, End line: 122

```python
builtin_extensions = (
    'sphinx.addnodes',
    'sphinx.builders.changes',
    'sphinx.builders.epub3',
    'sphinx.builders.dirhtml',
    'sphinx.builders.dummy',
    'sphinx.builders.gettext',
    'sphinx.builders.html',
    'sphinx.builders.latex',
    'sphinx.builders.linkcheck',
    'sphinx.builders.manpage',
    'sphinx.builders.singlehtml',
    'sphinx.builders.texinfo',
    'sphinx.builders.text',
    'sphinx.builders.xml',
    'sphinx.config',
    'sphinx.domains.c',
    'sphinx.domains.changeset',
    'sphinx.domains.citation',
    'sphinx.domains.cpp',
    'sphinx.domains.index',
    'sphinx.domains.javascript',
    'sphinx.domains.math',
    'sphinx.domains.python',
    'sphinx.domains.rst',
    'sphinx.domains.std',
    'sphinx.directives',
    'sphinx.directives.code',
    'sphinx.directives.other',
    'sphinx.directives.patches',
    'sphinx.extension',
    'sphinx.parsers',
    'sphinx.registry',
    'sphinx.roles',
    'sphinx.transforms',
    'sphinx.transforms.compact_bullet_list',
    'sphinx.transforms.i18n',
    'sphinx.transforms.references',
    'sphinx.transforms.post_transforms',
    'sphinx.transforms.post_transforms.code',
    'sphinx.transforms.post_transforms.images',
    'sphinx.util.compat',
    'sphinx.versioning',
    # collectors should be loaded by specific order
    'sphinx.environment.collectors.dependencies',
    'sphinx.environment.collectors.asset',
    'sphinx.environment.collectors.metadata',
    'sphinx.environment.collectors.title',
    'sphinx.environment.collectors.toctree',
    # 1st party extensions
    'sphinxcontrib.applehelp',
    'sphinxcontrib.devhelp',
    'sphinxcontrib.htmlhelp',
    'sphinxcontrib.serializinghtml',
    'sphinxcontrib.qthelp',
    # Strictly, alabaster theme is not a builtin extension,
    # but it is loaded automatically to use it as default theme.
    'alabaster',
)

ENV_PICKLE_FILENAME = 'environment.pickle'

logger = logging.getLogger(__name__)
```
### 57 - sphinx/builders/latex/__init__.py:

Start line: 42, End line: 97

```python
XINDY_LANG_OPTIONS = {
    # language codes from docutils.writers.latex2e.Babel
    # ! xindy language names may differ from those in use by LaTeX/babel
    # ! xindy does not support all Latin scripts as recognized by LaTeX/babel
    # ! not all xindy-supported languages appear in Babel.language_codes
    # cd /usr/local/texlive/2018/texmf-dist/xindy/modules/lang
    # find . -name '*utf8.xdy'
    # LATIN
    'sq': '-L albanian -C utf8 ',
    'hr': '-L croatian -C utf8 ',
    'cs': '-L czech -C utf8 ',
    'da': '-L danish -C utf8 ',
    'nl': '-L dutch-ij-as-ij -C utf8 ',
    'en': '-L english -C utf8 ',
    'eo': '-L esperanto -C utf8 ',
    'et': '-L estonian -C utf8 ',
    'fi': '-L finnish -C utf8 ',
    'fr': '-L french -C utf8 ',
    'de': '-L german-din5007 -C utf8 ',
    'is': '-L icelandic -C utf8 ',
    'it': '-L italian -C utf8 ',
    'la': '-L latin -C utf8 ',
    'lv': '-L latvian -C utf8 ',
    'lt': '-L lithuanian -C utf8 ',
    'dsb': '-L lower-sorbian -C utf8 ',
    'ds': '-L lower-sorbian -C utf8 ',   # trick, no conflict
    'nb': '-L norwegian -C utf8 ',
    'no': '-L norwegian -C utf8 ',       # and what about nynorsk?
    'pl': '-L polish -C utf8 ',
    'pt': '-L portuguese -C utf8 ',
    'ro': '-L romanian -C utf8 ',
    'sk': '-L slovak-small -C utf8 ',    # there is also slovak-large
    'sl': '-L slovenian -C utf8 ',
    'es': '-L spanish-modern -C utf8 ',  # there is also spanish-traditional
    'sv': '-L swedish -C utf8 ',
    'tr': '-L turkish -C utf8 ',
    'hsb': '-L upper-sorbian -C utf8 ',
    'hs': '-L upper-sorbian -C utf8 ',   # trick, no conflict
    'vi': '-L vietnamese -C utf8 ',
    # CYRILLIC
    # for usage with pdflatex, needs also cyrLICRutf8.xdy module
    'be': '-L belarusian -C utf8 ',
    'bg': '-L bulgarian -C utf8 ',
    'mk': '-L macedonian -C utf8 ',
    'mn': '-L mongolian-cyrillic -C utf8 ',
    'ru': '-L russian -C utf8 ',
    'sr': '-L serbian -C utf8 ',
    'sh-cyrl': '-L serbian -C utf8 ',
    'sh': '-L serbian -C utf8 ',         # trick, no conflict
    'uk': '-L ukrainian -C utf8 ',
    # GREEK
    # can work only with xelatex/lualatex, not supported by texindy+pdflatex
    'el': '-L greek -C utf8 ',
    # FIXME, not compatible with [:2] slice but does Sphinx support Greek ?
    'el-polyton': '-L greek-polytonic -C utf8 ',
}
```
### 60 - sphinx/builders/latex/__init__.py:

Start line: 11, End line: 40

```python
import os
from os import path
from typing import Any, Dict, Iterable, List, Tuple, Union

from docutils.frontend import OptionParser
from docutils.nodes import Node

import sphinx.builders.latex.nodes  # NOQA  # Workaround: import this before writer to avoid ImportError
from sphinx import addnodes, highlighting, package_dir
from sphinx.application import Sphinx
from sphinx.builders import Builder
from sphinx.builders.latex.constants import ADDITIONAL_SETTINGS, DEFAULT_SETTINGS, SHORTHANDOFF
from sphinx.builders.latex.theming import Theme, ThemeFactory
from sphinx.builders.latex.util import ExtBabel
from sphinx.config import ENUM, Config
from sphinx.environment.adapters.asset import ImageAdapter
from sphinx.errors import NoUri, SphinxError
from sphinx.locale import _, __
from sphinx.util import logging, progress_message, status_iterator, texescape
from sphinx.util.console import bold, darkgreen  # type: ignore
from sphinx.util.docutils import SphinxFileOutput, new_document
from sphinx.util.fileutil import copy_asset_file
from sphinx.util.i18n import format_date
from sphinx.util.nodes import inline_all_toctrees
from sphinx.util.osutil import SEP, make_filename_from_project
from sphinx.util.template import LaTeXRenderer
from sphinx.writers.latex import LaTeXTranslator, LaTeXWriter

# load docutils.nodes after loading sphinx.builders.latex.nodes
from docutils import nodes  # isort:skip
```
### 61 - sphinx/util/i18n.py:

Start line: 231, End line: 264

```python
def get_image_filename_for_language(filename: str, env: "BuildEnvironment") -> str:
    if not env.config.language:
        return filename

    filename_format = env.config.figure_language_filename
    d = dict()
    d['root'], d['ext'] = path.splitext(filename)
    dirname = path.dirname(d['root'])
    if dirname and not dirname.endswith(path.sep):
        dirname += path.sep
    docpath = path.dirname(env.docname)
    if docpath and not docpath.endswith(path.sep):
        docpath += path.sep
    d['path'] = dirname
    d['basename'] = path.basename(d['root'])
    d['docpath'] = docpath
    d['language'] = env.config.language
    try:
        return filename_format.format(**d)
    except KeyError as exc:
        raise SphinxError('Invalid figure_language_filename: %r' % exc) from exc


def search_image_for_language(filename: str, env: "BuildEnvironment") -> str:
    if not env.config.language:
        return filename

    translated = get_image_filename_for_language(filename, env)
    _, abspath = env.relfn2path(translated)
    if path.exists(abspath):
        return translated
    else:
        return filename
```
### 62 - sphinx/writers/latex.py:

Start line: 266, End line: 401

```python
class LaTeXTranslator(SphinxTranslator):
    builder: "LaTeXBuilder" = None

    secnumdepth = 2  # legacy sphinxhowto.cls uses this, whereas article.cls
    # default is originally 3. For book/report, 2 is already LaTeX default.
    ignore_missing_images = False

    def __init__(self, document: nodes.document, builder: "LaTeXBuilder",
                 theme: "Theme") -> None:
        super().__init__(document, builder)
        self.body: List[str] = []
        self.theme = theme

        # flags
        self.in_title = 0
        self.in_production_list = 0
        self.in_footnote = 0
        self.in_caption = 0
        self.in_term = 0
        self.needs_linetrimming = 0
        self.in_minipage = 0
        self.no_latex_floats = 0
        self.first_document = 1
        self.this_is_the_title = 1
        self.literal_whitespace = 0
        self.in_parsed_literal = 0
        self.compact_list = 0
        self.first_param = 0

        sphinxpkgoptions = []

        # sort out some elements
        self.elements = self.builder.context.copy()

        # initial section names
        self.sectionnames = LATEXSECTIONNAMES[:]
        if self.theme.toplevel_sectioning == 'section':
            self.sectionnames.remove('chapter')

        # determine top section level
        self.top_sectionlevel = 1
        if self.config.latex_toplevel_sectioning:
            try:
                self.top_sectionlevel = \
                    self.sectionnames.index(self.config.latex_toplevel_sectioning)
            except ValueError:
                logger.warning(__('unknown %r toplevel_sectioning for class %r') %
                               (self.config.latex_toplevel_sectioning, self.theme.docclass))

        if self.config.numfig:
            self.numfig_secnum_depth = self.config.numfig_secnum_depth
            if self.numfig_secnum_depth > 0:  # default is 1
                # numfig_secnum_depth as passed to sphinx.sty indices same names as in
                # LATEXSECTIONNAMES but with -1 for part, 0 for chapter, 1 for section...
                if len(self.sectionnames) < len(LATEXSECTIONNAMES) and \
                   self.top_sectionlevel > 0:
                    self.numfig_secnum_depth += self.top_sectionlevel
                else:
                    self.numfig_secnum_depth += self.top_sectionlevel - 1
                # this (minus one) will serve as minimum to LaTeX's secnumdepth
                self.numfig_secnum_depth = min(self.numfig_secnum_depth,
                                               len(LATEXSECTIONNAMES) - 1)
                # if passed key value is < 1 LaTeX will act as if 0; see sphinx.sty
                sphinxpkgoptions.append('numfigreset=%s' % self.numfig_secnum_depth)
            else:
                sphinxpkgoptions.append('nonumfigreset')

        if self.config.numfig and self.config.math_numfig:
            sphinxpkgoptions.append('mathnumfig')

        if (self.config.language not in {None, 'en', 'ja'} and
                'fncychap' not in self.config.latex_elements):
            # use Sonny style if any language specified (except English)
            self.elements['fncychap'] = (r'\usepackage[Sonny]{fncychap}' + CR +
                                         r'\ChNameVar{\Large\normalfont\sffamily}' + CR +
                                         r'\ChTitleVar{\Large\normalfont\sffamily}')

        self.babel = self.builder.babel
        if self.config.language and not self.babel.is_supported_language():
            # emit warning if specified language is invalid
            # (only emitting, nothing changed to processing)
            logger.warning(__('no Babel option known for language %r'),
                           self.config.language)

        minsecnumdepth = self.secnumdepth  # 2 from legacy sphinx manual/howto
        if self.document.get('tocdepth'):
            # reduce tocdepth if `part` or `chapter` is used for top_sectionlevel
            #   tocdepth = -1: show only parts
            #   tocdepth =  0: show parts and chapters
            #   tocdepth =  1: show parts, chapters and sections
            #   tocdepth =  2: show parts, chapters, sections and subsections
            #   ...
            tocdepth = self.document.get('tocdepth', 999) + self.top_sectionlevel - 2
            if len(self.sectionnames) < len(LATEXSECTIONNAMES) and \
               self.top_sectionlevel > 0:
                tocdepth += 1  # because top_sectionlevel is shifted by -1
            if tocdepth > len(LATEXSECTIONNAMES) - 2:  # default is 5 <-> subparagraph
                logger.warning(__('too large :maxdepth:, ignored.'))
                tocdepth = len(LATEXSECTIONNAMES) - 2

            self.elements['tocdepth'] = r'\setcounter{tocdepth}{%d}' % tocdepth
            minsecnumdepth = max(minsecnumdepth, tocdepth)

        if self.config.numfig and (self.config.numfig_secnum_depth > 0):
            minsecnumdepth = max(minsecnumdepth, self.numfig_secnum_depth - 1)

        if minsecnumdepth > self.secnumdepth:
            self.elements['secnumdepth'] = r'\setcounter{secnumdepth}{%d}' %\
                                           minsecnumdepth

        contentsname = document.get('contentsname')
        if contentsname:
            self.elements['contentsname'] = self.babel_renewcommand(r'\contentsname',
                                                                    contentsname)

        if self.elements['maxlistdepth']:
            sphinxpkgoptions.append('maxlistdepth=%s' % self.elements['maxlistdepth'])
        if sphinxpkgoptions:
            self.elements['sphinxpkgoptions'] = '[,%s]' % ','.join(sphinxpkgoptions)
        if self.elements['sphinxsetup']:
            self.elements['sphinxsetup'] = (r'\sphinxsetup{%s}' % self.elements['sphinxsetup'])
        if self.elements['extraclassoptions']:
            self.elements['classoptions'] += ',' + \
                                             self.elements['extraclassoptions']

        self.highlighter = highlighting.PygmentsBridge('latex', self.config.pygments_style,
                                                       latex_engine=self.config.latex_engine)
        self.context: List[Any] = []
        self.descstack: List[str] = []
        self.tables: List[Table] = []
        self.next_table_colspec: str = None
        self.bodystack: List[List[str]] = []
        self.footnote_restricted: Element = None
        self.pending_footnotes: List[nodes.footnote_reference] = []
        self.curfilestack: List[str] = []
        self.handled_abbrs: Set[str] = set()
```
