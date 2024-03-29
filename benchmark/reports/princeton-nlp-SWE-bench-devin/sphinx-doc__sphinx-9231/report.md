# sphinx-doc__sphinx-9231

| **sphinx-doc/sphinx** | `d6c19126c5ebd788619d491d4e70c949de9fd2ff` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 737 |
| **Any found context length** | 737 |
| **Avg pos** | 3.0 |
| **Min pos** | 3 |
| **Max pos** | 3 |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sphinx/builders/manpage.py b/sphinx/builders/manpage.py
--- a/sphinx/builders/manpage.py
+++ b/sphinx/builders/manpage.py
@@ -79,8 +79,9 @@ def write(self, *ignored: Any) -> None:
             docsettings.section = section
 
             if self.config.man_make_section_directory:
-                ensuredir(path.join(self.outdir, str(section)))
-                targetname = '%s/%s.%s' % (section, name, section)
+                dirname = 'man%s' % section
+                ensuredir(path.join(self.outdir, dirname))
+                targetname = '%s/%s.%s' % (dirname, name, section)
             else:
                 targetname = '%s.%s' % (name, section)
 

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sphinx/builders/manpage.py | 82 | 83 | 3 | 1 | 737


## Problem Statement

```
man_make_section_directory should not be enabled by default
Enabling `man_make_section_directory` by default in #8284 breaks projects relying on the previous behavior. This is a serious problem for Linux distributions that will end up with misplaced and unusable man pages. Please consider keeping it disabled by default; the benefit of being able to use MANPATH in the output directory does not justify this kind of breakage.

I also noticed that the current implementation generates paths like `<builddir>/1` instead of `<builddir>/man1`. Only the latter can be used with MANPATH which appears to be the main motivation behind #7996.

Examples of breakage I've seen so far (and we've only had sphinx 4.0.x in Arch Linux for three days):

[fish-shell](https://github.com/fish-shell/fish-shell) does not expect the section subdirectory and results in man pages for built-in shell commands being installed to `usr/share/fish/man/man1/1` instead of `usr/share/fish/man/man1` and also fails to filter out `fish.1`, `fish_indent.1` and `fish_key_reader.1` which are meant to be installed to `usr/share/man/man1`.

[llvm-project](https://github.com/llvm/llvm-project) copies the output directory to `usr/share/man/man1` resulting in paths like `usr/share/man/man1/1/foo.1` (note the additional `1` directory).

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 sphinx/builders/manpage.py** | 109 | 128| 166 | 166 | 954 | 
| 2 | **1 sphinx/builders/manpage.py** | 11 | 29| 154 | 320 | 954 | 
| **-> 3 <-** | **1 sphinx/builders/manpage.py** | 56 | 106| 417 | 737 | 954 | 
| 4 | 2 sphinx/cmd/make_mode.py | 17 | 54| 532 | 1269 | 2656 | 
| 5 | 3 sphinx/writers/manpage.py | 317 | 415| 757 | 2026 | 6143 | 
| 6 | 3 sphinx/writers/manpage.py | 192 | 256| 502 | 2528 | 6143 | 
| 7 | **3 sphinx/builders/manpage.py** | 32 | 54| 173 | 2701 | 6143 | 
| 8 | 4 doc/conf.py | 1 | 82| 731 | 3432 | 7607 | 
| 9 | 4 sphinx/cmd/make_mode.py | 96 | 140| 375 | 3807 | 7607 | 
| 10 | 4 sphinx/writers/manpage.py | 283 | 296| 128 | 3935 | 7607 | 
| 11 | 4 sphinx/writers/manpage.py | 258 | 281| 229 | 4164 | 7607 | 
| 12 | 4 sphinx/cmd/make_mode.py | 142 | 155| 149 | 4313 | 7607 | 
| 13 | 4 sphinx/cmd/make_mode.py | 57 | 84| 318 | 4631 | 7607 | 
| 14 | 4 sphinx/cmd/make_mode.py | 86 | 94| 114 | 4745 | 7607 | 
| 15 | 5 sphinx/__init__.py | 14 | 60| 475 | 5220 | 8156 | 
| 16 | 5 sphinx/writers/manpage.py | 416 | 464| 359 | 5579 | 8156 | 
| 17 | 5 doc/conf.py | 83 | 138| 476 | 6055 | 8156 | 
| 18 | 5 sphinx/writers/manpage.py | 11 | 40| 217 | 6272 | 8156 | 
| 19 | 6 sphinx/setup_command.py | 91 | 118| 229 | 6501 | 9701 | 
| 20 | 7 sphinx/cmd/build.py | 11 | 30| 132 | 6633 | 12362 | 
| 21 | 8 sphinx/builders/latex/__init__.py | 11 | 42| 331 | 6964 | 18074 | 
| 22 | 9 sphinx/builders/latex/constants.py | 74 | 124| 537 | 7501 | 20266 | 
| 23 | 10 sphinx/builders/texinfo.py | 198 | 220| 227 | 7728 | 22279 | 
| 24 | 10 sphinx/writers/manpage.py | 297 | 315| 201 | 7929 | 22279 | 
| 25 | 11 sphinx/util/osutil.py | 11 | 45| 214 | 8143 | 23913 | 
| 26 | 12 sphinx/cmd/quickstart.py | 453 | 519| 751 | 8894 | 29453 | 
| 27 | 13 sphinx/environment/__init__.py | 11 | 80| 492 | 9386 | 34901 | 
| 28 | 13 sphinx/cmd/build.py | 101 | 168| 762 | 10148 | 34901 | 
| 29 | 14 sphinx/directives/patches.py | 9 | 34| 192 | 10340 | 36745 | 
| 30 | 15 setup.py | 173 | 249| 638 | 10978 | 38472 | 
| 31 | 16 sphinx/builders/html/__init__.py | 1294 | 1375| 1000 | 11978 | 50760 | 
| 32 | 17 sphinx/util/__init__.py | 11 | 64| 446 | 12424 | 55513 | 
| 33 | 17 setup.py | 1 | 75| 445 | 12869 | 55513 | 
| 34 | 17 sphinx/setup_command.py | 142 | 190| 415 | 13284 | 55513 | 
| 35 | 17 sphinx/builders/html/__init__.py | 11 | 62| 432 | 13716 | 55513 | 
| 36 | 17 sphinx/cmd/quickstart.py | 538 | 605| 491 | 14207 | 55513 | 
| 37 | 18 sphinx/application.py | 13 | 58| 361 | 14568 | 67034 | 
| 38 | 18 sphinx/application.py | 333 | 382| 410 | 14978 | 67034 | 
| 39 | 19 sphinx/directives/__init__.py | 269 | 294| 167 | 15145 | 69286 | 
| 40 | 19 sphinx/writers/manpage.py | 71 | 190| 867 | 16012 | 69286 | 
| 41 | 20 sphinx/util/pycompat.py | 11 | 46| 328 | 16340 | 69779 | 
| 42 | 20 sphinx/builders/latex/constants.py | 126 | 214| 987 | 17327 | 69779 | 
| 43 | 20 sphinx/cmd/build.py | 169 | 199| 383 | 17710 | 69779 | 
| 44 | 20 sphinx/builders/latex/__init__.py | 465 | 525| 521 | 18231 | 69779 | 
| 45 | 21 sphinx/ext/apidoc.py | 303 | 368| 751 | 18982 | 74009 | 
| 46 | 21 sphinx/util/osutil.py | 71 | 171| 718 | 19700 | 74009 | 
| 47 | 22 sphinx/config.py | 462 | 481| 223 | 19923 | 78465 | 
| 48 | 22 sphinx/setup_command.py | 14 | 89| 415 | 20338 | 78465 | 
| 49 | 22 sphinx/setup_command.py | 120 | 140| 172 | 20510 | 78465 | 
| 50 | 23 sphinx/builders/changes.py | 125 | 168| 438 | 20948 | 79978 | 
| 51 | 24 sphinx/ext/autosummary/generate.py | 20 | 53| 257 | 21205 | 85277 | 
| 52 | 24 sphinx/ext/apidoc.py | 369 | 397| 374 | 21579 | 85277 | 
| 53 | 24 sphinx/application.py | 61 | 123| 494 | 22073 | 85277 | 
| 54 | 25 utils/bump_version.py | 149 | 180| 224 | 22297 | 86640 | 
| 55 | 26 sphinx/directives/code.py | 9 | 30| 148 | 22445 | 90491 | 
| 56 | 27 sphinx/ext/autosummary/__init__.py | 721 | 756| 325 | 22770 | 97002 | 
| 57 | 27 sphinx/directives/code.py | 407 | 470| 642 | 23412 | 97002 | 
| 58 | 28 sphinx/transforms/__init__.py | 11 | 44| 231 | 23643 | 100173 | 
| 59 | 29 sphinx/builders/epub3.py | 12 | 55| 288 | 23931 | 102760 | 
| 60 | 30 sphinx/writers/texinfo.py | 1296 | 1384| 699 | 24630 | 115074 | 
| 61 | 31 sphinx/util/smartypants.py | 380 | 392| 137 | 24767 | 119221 | 
| 62 | 31 sphinx/builders/texinfo.py | 128 | 168| 451 | 25218 | 119221 | 
| 63 | 32 sphinx/io.py | 10 | 39| 234 | 25452 | 120626 | 
| 64 | 33 sphinx/ext/autodoc/__init__.py | 2711 | 2754| 528 | 25980 | 143838 | 
| 65 | 33 sphinx/cmd/quickstart.py | 262 | 320| 739 | 26719 | 143838 | 
| 66 | 34 sphinx/builders/_epub_base.py | 11 | 118| 687 | 27406 | 150147 | 
| 67 | 34 sphinx/builders/texinfo.py | 11 | 36| 226 | 27632 | 150147 | 
| 68 | 34 sphinx/cmd/build.py | 202 | 299| 701 | 28333 | 150147 | 
| 69 | 35 sphinx/builders/singlehtml.py | 11 | 25| 112 | 28445 | 151943 | 
| 70 | 36 sphinx/registry.py | 11 | 51| 314 | 28759 | 156566 | 
| 71 | 37 sphinx/builders/dirhtml.py | 11 | 58| 330 | 29089 | 156946 | 
| 72 | 37 sphinx/cmd/quickstart.py | 11 | 119| 786 | 29875 | 156946 | 
| 73 | 37 sphinx/ext/autosummary/generate.py | 637 | 658| 162 | 30037 | 156946 | 
| 74 | 37 sphinx/builders/texinfo.py | 72 | 89| 189 | 30226 | 156946 | 
| 75 | 37 sphinx/writers/texinfo.py | 1490 | 1561| 519 | 30745 | 156946 | 
| 76 | 37 sphinx/builders/latex/__init__.py | 44 | 99| 825 | 31570 | 156946 | 
| 77 | 37 sphinx/config.py | 91 | 150| 740 | 32310 | 156946 | 
| 78 | 38 sphinx/builders/__init__.py | 11 | 48| 302 | 32612 | 162296 | 
| 79 | 38 sphinx/builders/latex/__init__.py | 452 | 462| 118 | 32730 | 162296 | 
| 80 | 38 sphinx/directives/patches.py | 213 | 249| 326 | 33056 | 162296 | 
| 81 | 38 sphinx/builders/__init__.py | 228 | 244| 172 | 33228 | 162296 | 
| 82 | 38 sphinx/environment/__init__.py | 284 | 310| 273 | 33501 | 162296 | 
| 83 | 39 sphinx/ext/githubpages.py | 11 | 37| 227 | 33728 | 162583 | 
| 84 | 40 sphinx/testing/util.py | 137 | 151| 188 | 33916 | 164315 | 
| 85 | 40 sphinx/builders/changes.py | 11 | 26| 124 | 34040 | 164315 | 
| 86 | 41 sphinx/builders/linkcheck.py | 134 | 229| 727 | 34767 | 169727 | 
| 87 | 41 sphinx/builders/latex/__init__.py | 328 | 369| 464 | 35231 | 169727 | 
| 88 | 42 sphinx/ext/autodoc/importer.py | 11 | 40| 211 | 35442 | 172189 | 
| 89 | 42 sphinx/writers/texinfo.py | 1210 | 1275| 457 | 35899 | 172189 | 
| 90 | 42 sphinx/transforms/__init__.py | 408 | 429| 172 | 36071 | 172189 | 
| 91 | 42 sphinx/cmd/make_mode.py | 158 | 168| 110 | 36181 | 172189 | 
| 92 | 42 sphinx/cmd/quickstart.py | 393 | 421| 385 | 36566 | 172189 | 
| 93 | 42 utils/bump_version.py | 67 | 102| 201 | 36767 | 172189 | 
| 94 | 42 sphinx/cmd/quickstart.py | 424 | 450| 158 | 36925 | 172189 | 
| 95 | 43 sphinx/directives/other.py | 369 | 393| 228 | 37153 | 175316 | 
| 96 | 44 sphinx/domains/std.py | 1113 | 1138| 209 | 37362 | 185613 | 
| 97 | 45 sphinx/builders/xml.py | 11 | 26| 116 | 37478 | 186452 | 
| 98 | 45 sphinx/builders/texinfo.py | 91 | 126| 358 | 37836 | 186452 | 
| 99 | 45 sphinx/builders/texinfo.py | 170 | 195| 254 | 38090 | 186452 | 
| 100 | 45 sphinx/cmd/quickstart.py | 185 | 260| 673 | 38763 | 186452 | 
| 101 | 46 utils/checks.py | 33 | 109| 545 | 39308 | 187358 | 
| 102 | 46 sphinx/builders/latex/__init__.py | 528 | 561| 404 | 39712 | 187358 | 
| 103 | 47 sphinx/ext/intersphinx.py | 26 | 51| 169 | 39881 | 191129 | 
| 104 | 48 sphinx/builders/latex/transforms.py | 102 | 127| 277 | 40158 | 195448 | 
| 105 | 48 sphinx/ext/intersphinx.py | 317 | 351| 424 | 40582 | 195448 | 


### Hint

```
Thank you for letting us know. I just reverted the change of default setting in #9232. It will be released as 4.0.2 soon. And I'll change the directory name in #9231. It will be released as 4.1.0.
```

## Patch

```diff
diff --git a/sphinx/builders/manpage.py b/sphinx/builders/manpage.py
--- a/sphinx/builders/manpage.py
+++ b/sphinx/builders/manpage.py
@@ -79,8 +79,9 @@ def write(self, *ignored: Any) -> None:
             docsettings.section = section
 
             if self.config.man_make_section_directory:
-                ensuredir(path.join(self.outdir, str(section)))
-                targetname = '%s/%s.%s' % (section, name, section)
+                dirname = 'man%s' % section
+                ensuredir(path.join(self.outdir, dirname))
+                targetname = '%s/%s.%s' % (dirname, name, section)
             else:
                 targetname = '%s.%s' % (name, section)
 

```

## Test Patch

```diff
diff --git a/tests/test_build_manpage.py b/tests/test_build_manpage.py
--- a/tests/test_build_manpage.py
+++ b/tests/test_build_manpage.py
@@ -34,7 +34,7 @@ def test_all(app, status, warning):
                     confoverrides={'man_make_section_directory': True})
 def test_man_make_section_directory(app, status, warning):
     app.build()
-    assert (app.outdir / '1' / 'python.1').exists()
+    assert (app.outdir / 'man1' / 'python.1').exists()
 
 
 @pytest.mark.sphinx('man', testroot='directive-code')

```


## Code snippets

### 1 - sphinx/builders/manpage.py:

Start line: 109, End line: 128

```python
def default_man_pages(config: Config) -> List[Tuple[str, str, str, List[str], int]]:
    """ Better default man_pages settings. """
    filename = make_filename_from_project(config.project)
    return [(config.root_doc, filename, '%s %s' % (config.project, config.release),
             [config.author], 1)]


def setup(app: Sphinx) -> Dict[str, Any]:
    app.add_builder(ManualPageBuilder)

    app.add_config_value('man_pages', default_man_pages, None)
    app.add_config_value('man_show_urls', False, None)
    app.add_config_value('man_make_section_directory', False, None)

    return {
        'version': 'builtin',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
```
### 2 - sphinx/builders/manpage.py:

Start line: 11, End line: 29

```python
from os import path
from typing import Any, Dict, List, Set, Tuple, Union

from docutils.frontend import OptionParser
from docutils.io import FileOutput

from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.builders import Builder
from sphinx.config import Config
from sphinx.errors import NoUri
from sphinx.locale import __
from sphinx.util import logging, progress_message
from sphinx.util.console import darkgreen  # type: ignore
from sphinx.util.nodes import inline_all_toctrees
from sphinx.util.osutil import ensuredir, make_filename_from_project
from sphinx.writers.manpage import ManualPageTranslator, ManualPageWriter

logger = logging.getLogger(__name__)
```
### 3 - sphinx/builders/manpage.py:

Start line: 56, End line: 106

```python
class ManualPageBuilder(Builder):

    @progress_message(__('writing'))
    def write(self, *ignored: Any) -> None:
        docwriter = ManualPageWriter(self)
        docsettings: Any = OptionParser(
            defaults=self.env.settings,
            components=(docwriter,),
            read_config_files=True).get_default_values()

        for info in self.config.man_pages:
            docname, name, description, authors, section = info
            if docname not in self.env.all_docs:
                logger.warning(__('"man_pages" config value references unknown '
                                  'document %s'), docname)
                continue
            if isinstance(authors, str):
                if authors:
                    authors = [authors]
                else:
                    authors = []

            docsettings.title = name
            docsettings.subtitle = description
            docsettings.authors = authors
            docsettings.section = section

            if self.config.man_make_section_directory:
                ensuredir(path.join(self.outdir, str(section)))
                targetname = '%s/%s.%s' % (section, name, section)
            else:
                targetname = '%s.%s' % (name, section)

            logger.info(darkgreen(targetname) + ' { ', nonl=True)
            destination = FileOutput(
                destination_path=path.join(self.outdir, targetname),
                encoding='utf-8')

            tree = self.env.get_doctree(docname)
            docnames: Set[str] = set()
            largetree = inline_all_toctrees(self, docnames, docname, tree,
                                            darkgreen, [docname])
            largetree.settings = docsettings
            logger.info('} ', nonl=True)
            self.env.resolve_references(largetree, docname, self)
            # remove pending_xref nodes
            for pendingnode in largetree.traverse(addnodes.pending_xref):
                pendingnode.replace_self(pendingnode.children)

            docwriter.write(largetree, destination)

    def finish(self) -> None:
        pass
```
### 4 - sphinx/cmd/make_mode.py:

Start line: 17, End line: 54

```python
import os
import subprocess
import sys
from os import path
from typing import List

import sphinx
from sphinx.cmd.build import build_main
from sphinx.util.console import blue, bold, color_terminal, nocolor  # type: ignore
from sphinx.util.osutil import cd, rmtree

BUILDERS = [
    ("",      "html",        "to make standalone HTML files"),
    ("",      "dirhtml",     "to make HTML files named index.html in directories"),
    ("",      "singlehtml",  "to make a single large HTML file"),
    ("",      "pickle",      "to make pickle files"),
    ("",      "json",        "to make JSON files"),
    ("",      "htmlhelp",    "to make HTML files and an HTML help project"),
    ("",      "qthelp",      "to make HTML files and a qthelp project"),
    ("",      "devhelp",     "to make HTML files and a Devhelp project"),
    ("",      "epub",        "to make an epub"),
    ("",      "latex",       "to make LaTeX files, you can set PAPER=a4 or PAPER=letter"),
    ("posix", "latexpdf",    "to make LaTeX and PDF files (default pdflatex)"),
    ("posix", "latexpdfja",  "to make LaTeX files and run them through platex/dvipdfmx"),
    ("",      "text",        "to make text files"),
    ("",      "man",         "to make manual pages"),
    ("",      "texinfo",     "to make Texinfo files"),
    ("posix", "info",        "to make Texinfo files and run them through makeinfo"),
    ("",      "gettext",     "to make PO message catalogs"),
    ("",      "changes",     "to make an overview of all changed/added/deprecated items"),
    ("",      "xml",         "to make Docutils-native XML files"),
    ("",      "pseudoxml",   "to make pseudoxml-XML files for display purposes"),
    ("",      "linkcheck",   "to check all external links for integrity"),
    ("",      "doctest",     "to run all doctests embedded in the documentation "
                             "(if enabled)"),
    ("",      "coverage",    "to run coverage check of the documentation (if enabled)"),
    ("",      "clean",       "to remove everything in the build directory"),
]
```
### 5 - sphinx/writers/manpage.py:

Start line: 317, End line: 415

```python
class ManualPageTranslator(SphinxTranslator, BaseTranslator):

    def visit_number_reference(self, node: Element) -> None:
        text = nodes.Text(node.get('title', '#'))
        self.visit_Text(text)
        raise nodes.SkipNode

    def visit_centered(self, node: Element) -> None:
        self.ensure_eol()
        self.body.append('.sp\n.ce\n')

    def depart_centered(self, node: Element) -> None:
        self.body.append('\n.ce 0\n')

    def visit_compact_paragraph(self, node: Element) -> None:
        pass

    def depart_compact_paragraph(self, node: Element) -> None:
        pass

    def visit_download_reference(self, node: Element) -> None:
        pass

    def depart_download_reference(self, node: Element) -> None:
        pass

    def visit_toctree(self, node: Element) -> None:
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
        bullet_list = cast(nodes.bullet_list, node[0])
        list_items = cast(Iterable[nodes.list_item], bullet_list)
        self.ensure_eol()
        bullet_list = cast(nodes.bullet_list, node[0])
        list_items = cast(Iterable[nodes.list_item], bullet_list)
        self.body.append(', '.join(n.astext() for n in list_items) + '.')
        self.body.append('\n')
        raise nodes.SkipNode

    def visit_hlist(self, node: Element) -> None:
        self.visit_bullet_list(node)

    def depart_hlist(self, node: Element) -> None:
        self.depart_bullet_list(node)

    def visit_hlistcol(self, node: Element) -> None:
        pass

    def depart_hlistcol(self, node: Element) -> None:
        pass

    def visit_literal_emphasis(self, node: Element) -> None:
        return self.visit_emphasis(node)

    def depart_literal_emphasis(self, node: Element) -> None:
        return self.depart_emphasis(node)

    def visit_literal_strong(self, node: Element) -> None:
        return self.visit_strong(node)

    def depart_literal_strong(self, node: Element) -> None:
        return self.depart_strong(node)

    def visit_abbreviation(self, node: Element) -> None:
        pass

    def depart_abbreviation(self, node: Element) -> None:
        pass

    def visit_manpage(self, node: Element) -> None:
        return self.visit_strong(node)

    def depart_manpage(self, node: Element) -> None:
        return self.depart_strong(node)

    # overwritten: handle section titles better than in 0.6 release
    def visit_caption(self, node: Element) -> None:
        if isinstance(node.parent, nodes.container) and node.parent.get('literal_block'):
            self.body.append('.sp\n')
        else:
            super().visit_caption(node)

    def depart_caption(self, node: Element) -> None:
        if isinstance(node.parent, nodes.container) and node.parent.get('literal_block'):
            self.body.append('\n')
        else:
            super().depart_caption(node)

    # overwritten: handle section titles better than in 0.6 release
```
### 6 - sphinx/writers/manpage.py:

Start line: 192, End line: 256

```python
class ManualPageTranslator(SphinxTranslator, BaseTranslator):

    def depart_desc_parameterlist(self, node: Element) -> None:
        self.body.append(')')

    def visit_desc_parameter(self, node: Element) -> None:
        if not self.first_param:
            self.body.append(', ')
        else:
            self.first_param = 0

    def depart_desc_parameter(self, node: Element) -> None:
        pass

    def visit_desc_optional(self, node: Element) -> None:
        self.body.append('[')

    def depart_desc_optional(self, node: Element) -> None:
        self.body.append(']')

    def visit_desc_annotation(self, node: Element) -> None:
        pass

    def depart_desc_annotation(self, node: Element) -> None:
        pass

    ##############################################

    def visit_versionmodified(self, node: Element) -> None:
        self.visit_paragraph(node)

    def depart_versionmodified(self, node: Element) -> None:
        self.depart_paragraph(node)

    # overwritten -- don't make whole of term bold if it includes strong node
    def visit_term(self, node: Element) -> None:
        if node.traverse(nodes.strong):
            self.body.append('\n')
        else:
            super().visit_term(node)

    # overwritten -- we don't want source comments to show up
    def visit_comment(self, node: Element) -> None:  # type: ignore
        raise nodes.SkipNode

    # overwritten -- added ensure_eol()
    def visit_footnote(self, node: Element) -> None:
        self.ensure_eol()
        super().visit_footnote(node)

    # overwritten -- handle footnotes rubric
    def visit_rubric(self, node: Element) -> None:
        self.ensure_eol()
        if len(node) == 1 and node.astext() in ('Footnotes', _('Footnotes')):
            self.body.append('.SH ' + self.deunicode(node.astext()).upper() + '\n')
            raise nodes.SkipNode
        else:
            self.body.append('.sp\n')

    def depart_rubric(self, node: Element) -> None:
        self.body.append('\n')

    def visit_seealso(self, node: Element) -> None:
        self.visit_admonition(node, 'seealso')

    def depart_seealso(self, node: Element) -> None:
        self.depart_admonition(node)
```
### 7 - sphinx/builders/manpage.py:

Start line: 32, End line: 54

```python
class ManualPageBuilder(Builder):
    """
    Builds groff output in manual page format.
    """
    name = 'man'
    format = 'man'
    epilog = __('The manual pages are in %(outdir)s.')

    default_translator_class = ManualPageTranslator
    supported_image_types: List[str] = []

    def init(self) -> None:
        if not self.config.man_pages:
            logger.warning(__('no "man_pages" config value found; no manual pages '
                              'will be written'))

    def get_outdated_docs(self) -> Union[str, List[str]]:
        return 'all manpages'  # for now

    def get_target_uri(self, docname: str, typ: str = None) -> str:
        if typ == 'token':
            return ''
        raise NoUri(docname, typ)
```
### 8 - doc/conf.py:

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
epub_publisher = 'http://sphinx-doc.org/'
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
### 9 - sphinx/cmd/make_mode.py:

Start line: 96, End line: 140

```python
class Make:

    def build_latexpdf(self) -> int:
        if self.run_generic_build('latex') > 0:
            return 1

        if sys.platform == 'win32':
            makecmd = os.environ.get('MAKE', 'make.bat')
        else:
            makecmd = self.makecmd
        try:
            with cd(self.builddir_join('latex')):
                return subprocess.call([makecmd, 'all-pdf'])
        except OSError:
            print('Error: Failed to run: %s' % makecmd)
            return 1

    def build_latexpdfja(self) -> int:
        if self.run_generic_build('latex') > 0:
            return 1

        if sys.platform == 'win32':
            makecmd = os.environ.get('MAKE', 'make.bat')
        else:
            makecmd = self.makecmd
        try:
            with cd(self.builddir_join('latex')):
                return subprocess.call([makecmd, 'all-pdf'])
        except OSError:
            print('Error: Failed to run: %s' % makecmd)
            return 1

    def build_info(self) -> int:
        if self.run_generic_build('texinfo') > 0:
            return 1
        try:
            with cd(self.builddir_join('texinfo')):
                return subprocess.call([self.makecmd, 'info'])
        except OSError:
            print('Error: Failed to run: %s' % self.makecmd)
            return 1

    def build_gettext(self) -> int:
        dtdir = self.builddir_join('gettext', '.doctrees')
        if self.run_generic_build('gettext', doctreedir=dtdir) > 0:
            return 1
        return 0
```
### 10 - sphinx/writers/manpage.py:

Start line: 283, End line: 296

```python
class ManualPageTranslator(SphinxTranslator, BaseTranslator):

    def visit_production(self, node: Element) -> None:
        pass

    def depart_production(self, node: Element) -> None:
        pass

    # overwritten -- don't emit a warning for images
    def visit_image(self, node: Element) -> None:
        if 'alt' in node.attributes:
            self.body.append(_('[image: %s]') % node['alt'] + '\n')
        self.body.append(_('[image]') + '\n')
        raise nodes.SkipNode

    # overwritten -- don't visit inner marked up nodes
```
