# sphinx-doc__sphinx-7615

| **sphinx-doc/sphinx** | `6ce265dc813f9ecb92bf1cdf8733fbada7f5c967` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | 9509 |
| **Avg pos** | 21.0 |
| **Min pos** | 21 |
| **Max pos** | 21 |
| **Top file pos** | 18 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sphinx/transforms/__init__.py b/sphinx/transforms/__init__.py
--- a/sphinx/transforms/__init__.py
+++ b/sphinx/transforms/__init__.py
@@ -23,6 +23,7 @@
 from sphinx.config import Config
 from sphinx.deprecation import RemovedInSphinx40Warning, deprecated_alias
 from sphinx.locale import _, __
+from sphinx.util import docutils
 from sphinx.util import logging
 from sphinx.util.docutils import new_document
 from sphinx.util.i18n import format_date
@@ -360,12 +361,18 @@ def is_available(self) -> bool:
     def get_tokens(self, txtnodes: List[Text]) -> Generator[Tuple[str, str], None, None]:
         # A generator that yields ``(texttype, nodetext)`` tuples for a list
         # of "Text" nodes (interface to ``smartquotes.educate_tokens()``).
-
-        texttype = {True: 'literal',  # "literal" text is not changed:
-                    False: 'plain'}
         for txtnode in txtnodes:
-            notsmartquotable = not is_smartquotable(txtnode)
-            yield (texttype[notsmartquotable], txtnode.astext())
+            if is_smartquotable(txtnode):
+                if docutils.__version_info__ >= (0, 16):
+                    # SmartQuotes uses backslash escapes instead of null-escapes
+                    text = re.sub(r'(?<=\x00)([-\\\'".`])', r'\\\1', str(txtnode))
+                else:
+                    text = txtnode.astext()
+
+                yield ('plain', text)
+            else:
+                # skip smart quotes
+                yield ('literal', txtnode.astext())
 
 
 class DoctreeReadEvent(SphinxTransform):

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sphinx/transforms/__init__.py | 26 | 26 | 21 | 18 | 9509
| sphinx/transforms/__init__.py | 363 | 368 | - | 18 | -


## Problem Statement

```
Sphinx, unlike Docutils, incorrectly renders consecutive backslashes
**Describe the bug**
Sphinx incorrectly renders four or more consecutive backslashes. In pure Docutils, they are renderer properly according with RST spec.

**To Reproduce**
The following snippet demonstrantes buggy rendering. 
\`\`\`
Two \\

Three \\\

Four \\\\

Five \\\\\

Six \\\\\\
\`\`\`

**Expected behavior**
Two backslashes should be rendered as `\`. Three still as `\`. Four and five as `\\`. Six as `\\\` and so on. This is how it works in Docutils.

**Screenshots**
![image](https://user-images.githubusercontent.com/383059/80948942-5cb29c00-8df3-11ea-8fe9-ca4bc390eef9.png)

**Environment info**
- OS: Linux
- Python version: 3.6
- Sphinx version: 3.0.2
- Sphinx extensions:  none

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 sphinx/util/texescape.py | 11 | 64| 623 | 623 | 1769 | 
| 2 | 2 sphinx/parsers.py | 71 | 119| 360 | 983 | 2637 | 
| 3 | 3 sphinx/util/smartypants.py | 376 | 388| 137 | 1120 | 6748 | 
| 4 | 4 utils/doclinter.py | 11 | 81| 467 | 1587 | 7267 | 
| 5 | 5 doc/conf.py | 59 | 126| 693 | 2280 | 8821 | 
| 6 | 6 sphinx/highlighting.py | 11 | 54| 379 | 2659 | 10129 | 
| 7 | 7 sphinx/util/jsdump.py | 12 | 32| 180 | 2839 | 11544 | 
| 8 | 7 doc/conf.py | 1 | 58| 502 | 3341 | 11544 | 
| 9 | 8 sphinx/util/__init__.py | 11 | 75| 505 | 3846 | 17331 | 
| 10 | 9 sphinx/cmd/quickstart.py | 11 | 119| 772 | 4618 | 22880 | 
| 11 | 9 sphinx/util/texescape.py | 143 | 170| 259 | 4877 | 22880 | 
| 12 | 10 sphinx/util/cfamily.py | 11 | 80| 764 | 5641 | 26380 | 
| 13 | 11 sphinx/util/docstrings.py | 48 | 84| 350 | 5991 | 27132 | 
| 14 | 12 sphinx/testing/util.py | 179 | 198| 193 | 6184 | 28898 | 
| 15 | 12 sphinx/util/texescape.py | 65 | 126| 712 | 6896 | 28898 | 
| 16 | 13 sphinx/util/nodes.py | 467 | 510| 689 | 7585 | 34302 | 
| 17 | 14 sphinx/ext/imgmath.py | 11 | 111| 681 | 8266 | 37688 | 
| 18 | 15 sphinx/__init__.py | 14 | 65| 495 | 8761 | 38257 | 
| 19 | 16 sphinx/ext/graphviz.py | 12 | 44| 236 | 8997 | 41850 | 
| 20 | 17 sphinx/ext/doctest.py | 12 | 51| 286 | 9283 | 46805 | 
| **-> 21 <-** | **18 sphinx/transforms/__init__.py** | 11 | 44| 226 | 9509 | 49952 | 
| 22 | 19 sphinx/writers/texinfo.py | 862 | 966| 788 | 10297 | 62126 | 
| 23 | 19 sphinx/writers/texinfo.py | 1215 | 1271| 418 | 10715 | 62126 | 
| 24 | 20 sphinx/util/console.py | 86 | 99| 155 | 10870 | 63126 | 
| 25 | 20 sphinx/writers/texinfo.py | 1292 | 1375| 673 | 11543 | 63126 | 
| 26 | 21 sphinx/writers/latex.py | 1082 | 1152| 538 | 12081 | 82808 | 
| 27 | 22 sphinx/ext/napoleon/docstring.py | 13 | 40| 327 | 12408 | 91616 | 
| 28 | 23 sphinx/util/docutils.py | 360 | 380| 173 | 12581 | 95736 | 
| 29 | 23 sphinx/writers/texinfo.py | 350 | 396| 422 | 13003 | 95736 | 
| 30 | 23 sphinx/ext/doctest.py | 240 | 267| 277 | 13280 | 95736 | 
| 31 | 24 sphinx/writers/html.py | 481 | 506| 238 | 13518 | 103102 | 
| 32 | 25 sphinx/writers/html5.py | 433 | 458| 239 | 13757 | 110005 | 
| 33 | 26 sphinx/io.py | 10 | 46| 275 | 14032 | 111698 | 
| 34 | 26 sphinx/ext/doctest.py | 309 | 324| 130 | 14162 | 111698 | 
| 35 | 26 sphinx/writers/texinfo.py | 752 | 860| 808 | 14970 | 111698 | 
| 36 | 26 sphinx/writers/html.py | 11 | 52| 305 | 15275 | 111698 | 
| 37 | 26 sphinx/highlighting.py | 57 | 166| 876 | 16151 | 111698 | 
| 38 | 27 sphinx/writers/text.py | 1045 | 1154| 793 | 16944 | 120652 | 
| 39 | 27 sphinx/writers/latex.py | 1454 | 1518| 863 | 17807 | 120652 | 
| 40 | 28 utils/checks.py | 33 | 109| 545 | 18352 | 121558 | 
| 41 | 29 sphinx/builders/html/__init__.py | 11 | 67| 457 | 18809 | 132780 | 
| 42 | 29 sphinx/writers/latex.py | 2124 | 2145| 256 | 19065 | 132780 | 
| 43 | 29 sphinx/writers/html5.py | 11 | 51| 301 | 19366 | 132780 | 
| 44 | 30 sphinx/setup_command.py | 112 | 134| 216 | 19582 | 134462 | 
| 45 | 31 setup.py | 1 | 75| 443 | 20025 | 136161 | 
| 46 | 31 sphinx/util/jsdump.py | 35 | 50| 165 | 20190 | 136161 | 
| 47 | 32 sphinx/writers/manpage.py | 313 | 411| 757 | 20947 | 139672 | 
| 48 | 33 sphinx/builders/latex/transforms.py | 91 | 116| 277 | 21224 | 143851 | 
| 49 | 33 sphinx/writers/latex.py | 1611 | 1670| 496 | 21720 | 143851 | 
| 50 | 34 sphinx/cmd/build.py | 11 | 30| 132 | 21852 | 146514 | 
| 51 | 34 sphinx/writers/texinfo.py | 1397 | 1448| 366 | 22218 | 146514 | 
| 52 | 34 sphinx/writers/texinfo.py | 1462 | 1529| 493 | 22711 | 146514 | 
| 53 | 34 sphinx/writers/latex.py | 579 | 642| 521 | 23232 | 146514 | 
| 54 | 34 sphinx/writers/text.py | 767 | 874| 830 | 24062 | 146514 | 
| 55 | 34 sphinx/writers/latex.py | 1924 | 2005| 605 | 24667 | 146514 | 
| 56 | 34 sphinx/writers/latex.py | 1586 | 1609| 290 | 24957 | 146514 | 
| 57 | 34 sphinx/writers/html.py | 171 | 231| 562 | 25519 | 146514 | 
| 58 | 34 sphinx/setup_command.py | 159 | 208| 415 | 25934 | 146514 | 
| 59 | 35 sphinx/domains/rst.py | 36 | 69| 331 | 26265 | 148984 | 
| 60 | 36 sphinx/ext/autosummary/__init__.py | 708 | 720| 101 | 26366 | 155449 | 
| 61 | 36 sphinx/writers/latex.py | 2067 | 2084| 170 | 26536 | 155449 | 
| 62 | 36 sphinx/writers/latex.py | 507 | 545| 384 | 26920 | 155449 | 
| 63 | 36 sphinx/writers/html.py | 795 | 842| 462 | 27382 | 155449 | 
| 64 | 36 sphinx/util/console.py | 102 | 143| 296 | 27678 | 155449 | 
| 65 | 37 sphinx/util/pycompat.py | 11 | 53| 350 | 28028 | 156240 | 
| 66 | 37 sphinx/writers/html.py | 354 | 409| 472 | 28500 | 156240 | 
| 67 | 37 setup.py | 172 | 247| 626 | 29126 | 156240 | 
| 68 | 37 sphinx/writers/latex.py | 1520 | 1584| 621 | 29747 | 156240 | 
| 69 | 37 sphinx/util/docutils.py | 147 | 171| 191 | 29938 | 156240 | 
| 70 | 37 sphinx/domains/rst.py | 260 | 286| 274 | 30212 | 156240 | 
| 71 | 38 sphinx/directives/__init__.py | 269 | 307| 273 | 30485 | 158735 | 
| 72 | 38 sphinx/writers/html.py | 694 | 793| 803 | 31288 | 158735 | 
| 73 | 38 sphinx/writers/html.py | 626 | 672| 362 | 31650 | 158735 | 
| 74 | 38 sphinx/domains/rst.py | 238 | 247| 126 | 31776 | 158735 | 
| 75 | 38 sphinx/testing/util.py | 10 | 49| 297 | 32073 | 158735 | 
| 76 | 38 sphinx/writers/latex.py | 796 | 858| 567 | 32640 | 158735 | 
| 77 | 38 sphinx/setup_command.py | 14 | 91| 419 | 33059 | 158735 | 
| 78 | 38 sphinx/ext/doctest.py | 397 | 473| 752 | 33811 | 158735 | 
| 79 | 38 sphinx/domains/rst.py | 11 | 33| 161 | 33972 | 158735 | 
| 80 | 39 sphinx/builders/latex/__init__.py | 11 | 42| 329 | 34301 | 164498 | 
| 81 | 39 sphinx/writers/latex.py | 692 | 794| 825 | 35126 | 164498 | 
| 82 | 39 sphinx/writers/texinfo.py | 968 | 1085| 846 | 35972 | 164498 | 
| 83 | **39 sphinx/transforms/__init__.py** | 274 | 315| 309 | 36281 | 164498 | 
| 84 | 39 sphinx/writers/html5.py | 143 | 203| 567 | 36848 | 164498 | 
| 85 | 39 sphinx/util/texescape.py | 129 | 140| 124 | 36972 | 164498 | 
| 86 | 40 sphinx/domains/python.py | 11 | 88| 598 | 37570 | 176135 | 
| 87 | 40 sphinx/util/__init__.py | 573 | 601| 265 | 37835 | 176135 | 
| 88 | 41 sphinx/util/rst.py | 11 | 78| 551 | 38386 | 176964 | 
| 89 | 41 sphinx/writers/html5.py | 567 | 609| 321 | 38707 | 176964 | 
| 90 | 41 sphinx/builders/html/__init__.py | 1184 | 1254| 858 | 39565 | 176964 | 
| 91 | 41 sphinx/ext/napoleon/docstring.py | 435 | 476| 345 | 39910 | 176964 | 
| 92 | 41 sphinx/writers/html5.py | 54 | 141| 796 | 40706 | 176964 | 
| 93 | 41 sphinx/ext/napoleon/docstring.py | 387 | 414| 258 | 40964 | 176964 | 
| 94 | 41 sphinx/domains/rst.py | 230 | 236| 113 | 41077 | 176964 | 
| 95 | 42 sphinx/pygments_styles.py | 38 | 96| 506 | 41583 | 177656 | 
| 96 | 43 sphinx/ext/autodoc/__init__.py | 986 | 1003| 217 | 41800 | 193380 | 
| 97 | 43 sphinx/writers/html5.py | 631 | 717| 686 | 42486 | 193380 | 
| 98 | 44 sphinx/transforms/i18n.py | 43 | 78| 323 | 42809 | 197961 | 
| 99 | 45 sphinx/transforms/post_transforms/code.py | 114 | 143| 208 | 43017 | 198989 | 
| 100 | 45 sphinx/writers/html.py | 82 | 169| 789 | 43806 | 198989 | 


## Patch

```diff
diff --git a/sphinx/transforms/__init__.py b/sphinx/transforms/__init__.py
--- a/sphinx/transforms/__init__.py
+++ b/sphinx/transforms/__init__.py
@@ -23,6 +23,7 @@
 from sphinx.config import Config
 from sphinx.deprecation import RemovedInSphinx40Warning, deprecated_alias
 from sphinx.locale import _, __
+from sphinx.util import docutils
 from sphinx.util import logging
 from sphinx.util.docutils import new_document
 from sphinx.util.i18n import format_date
@@ -360,12 +361,18 @@ def is_available(self) -> bool:
     def get_tokens(self, txtnodes: List[Text]) -> Generator[Tuple[str, str], None, None]:
         # A generator that yields ``(texttype, nodetext)`` tuples for a list
         # of "Text" nodes (interface to ``smartquotes.educate_tokens()``).
-
-        texttype = {True: 'literal',  # "literal" text is not changed:
-                    False: 'plain'}
         for txtnode in txtnodes:
-            notsmartquotable = not is_smartquotable(txtnode)
-            yield (texttype[notsmartquotable], txtnode.astext())
+            if is_smartquotable(txtnode):
+                if docutils.__version_info__ >= (0, 16):
+                    # SmartQuotes uses backslash escapes instead of null-escapes
+                    text = re.sub(r'(?<=\x00)([-\\\'".`])', r'\\\1', str(txtnode))
+                else:
+                    text = txtnode.astext()
+
+                yield ('plain', text)
+            else:
+                # skip smart quotes
+                yield ('literal', txtnode.astext())
 
 
 class DoctreeReadEvent(SphinxTransform):

```

## Test Patch

```diff
diff --git a/tests/test_markup.py b/tests/test_markup.py
--- a/tests/test_markup.py
+++ b/tests/test_markup.py
@@ -13,7 +13,6 @@
 import pytest
 from docutils import frontend, utils, nodes
 from docutils.parsers.rst import Parser as RstParser
-from docutils.transforms.universal import SmartQuotes
 
 from sphinx import addnodes
 from sphinx.builders.html.transforms import KeyboardTransform
@@ -21,6 +20,8 @@
 from sphinx.builders.latex.theming import ThemeFactory
 from sphinx.roles import XRefRole
 from sphinx.testing.util import Struct, assert_node
+from sphinx.transforms import SphinxSmartQuotes
+from sphinx.util import docutils
 from sphinx.util import texescape
 from sphinx.util.docutils import sphinx_domains
 from sphinx.writers.html import HTMLWriter, HTMLTranslator
@@ -67,7 +68,7 @@ def parse_(rst):
         document = new_document()
         parser = RstParser()
         parser.parse(rst, document)
-        SmartQuotes(document, startnode=None).apply()
+        SphinxSmartQuotes(document, startnode=None).apply()
         for msg in document.traverse(nodes.system_message):
             if msg['level'] == 1:
                 msg.replace_self([])
@@ -349,6 +350,21 @@ def test_inline(get_verifier, type, rst, html_expected, latex_expected):
     verifier(rst, html_expected, latex_expected)
 
 
+@pytest.mark.parametrize('type,rst,html_expected,latex_expected', [
+    (
+        'verify',
+        r'4 backslashes \\\\',
+        r'<p>4 backslashes \\</p>',
+        None,
+    ),
+])
+@pytest.mark.skipif(docutils.__version_info__ < (0, 16),
+                    reason='docutils-0.16 or above is required')
+def test_inline_docutils16(get_verifier, type, rst, html_expected, latex_expected):
+    verifier = get_verifier(type)
+    verifier(rst, html_expected, latex_expected)
+
+
 @pytest.mark.sphinx(confoverrides={'latex_engine': 'xelatex'})
 @pytest.mark.parametrize('type,rst,html_expected,latex_expected', [
     (

```


## Code snippets

### 1 - sphinx/util/texescape.py:

Start line: 11, End line: 64

```python
import re
from typing import Dict

from sphinx.deprecation import RemovedInSphinx40Warning, deprecated_alias


tex_replacements = [
    # map TeX special chars
    ('$', r'\$'),
    ('%', r'\%'),
    ('&', r'\&'),
    ('#', r'\#'),
    ('_', r'\_'),
    ('{', r'\{'),
    ('}', r'\}'),
    ('\\', r'\textbackslash{}'),
    ('~', r'\textasciitilde{}'),
    ('^', r'\textasciicircum{}'),
    # map chars to avoid mis-interpretation in LaTeX
    ('[', r'{[}'),
    (']', r'{]}'),
    # map special Unicode characters to TeX commands
    ('✓', r'\(\checkmark\)'),
    ('✔', r'\(\pmb{\checkmark}\)'),
    # used to separate -- in options
    ('﻿', r'{}'),
    # map some special Unicode characters to similar ASCII ones
    # (even for Unicode LaTeX as may not be supported by OpenType font)
    ('⎽', r'\_'),
    ('ℯ', r'e'),
    ('ⅈ', r'i'),
    # Greek alphabet not escaped: pdflatex handles it via textalpha and inputenc
    # OHM SIGN U+2126 is handled by LaTeX textcomp package
]

# A map to avoid TeX ligatures or character replacements in PDF output
# xelatex/lualatex/uplatex are handled differently (#5790, #6888)
ascii_tex_replacements = [
    # Note: the " renders curly in OT1 encoding but straight in T1, T2A, LY1...
    #       escaping it to \textquotedbl would break documents using OT1
    #       Sphinx does \shorthandoff{"} to avoid problems with some languages
    # There is no \text... LaTeX escape for the hyphen character -
    ('-', r'\sphinxhyphen{}'),  # -- and --- are TeX ligatures
    # ,, is a TeX ligature in T1 encoding, but escaping the comma adds
    # complications (whether by {}, or a macro) and is not done
    # the next two require textcomp package
    ("'", r'\textquotesingle{}'),  # else ' renders curly, and '' is a ligature
    ('`', r'\textasciigrave{}'),   # else \` and \`\` render curly
    ('<', r'\textless{}'),     # < is inv. exclam in OT1, << is a T1-ligature
    ('>', r'\textgreater{}'),  # > is inv. quest. mark in 0T1, >> a T1-ligature
]

# A map Unicode characters to LaTeX representation
# (for LaTeX engines which don't support unicode)
```
### 2 - sphinx/parsers.py:

Start line: 71, End line: 119

```python
class RSTParser(docutils.parsers.rst.Parser, Parser):
    """A reST parser for Sphinx."""

    def get_transforms(self) -> List["Type[Transform]"]:
        """Sphinx's reST parser replaces a transform class for smart-quotes by own's

        refs: sphinx.io.SphinxStandaloneReader
        """
        transforms = super().get_transforms()
        transforms.remove(SmartQuotes)
        return transforms

    def parse(self, inputstring: Union[str, StringList], document: nodes.document) -> None:
        """Parse text and generate a document tree."""
        self.setup_parse(inputstring, document)  # type: ignore
        self.statemachine = states.RSTStateMachine(
            state_classes=self.state_classes,
            initial_state=self.initial_state,
            debug=document.reporter.debug_flag)

        # preprocess inputstring
        if isinstance(inputstring, str):
            lines = docutils.statemachine.string2lines(
                inputstring, tab_width=document.settings.tab_width,
                convert_whitespace=True)

            inputlines = StringList(lines, document.current_source)
        else:
            inputlines = inputstring

        self.decorate(inputlines)
        self.statemachine.run(inputlines, document, inliner=self.inliner)
        self.finish_parse()

    def decorate(self, content: StringList) -> None:
        """Preprocess reST content before parsing."""
        prepend_prolog(content, self.config.rst_prolog)
        append_epilog(content, self.config.rst_epilog)


def setup(app: "Sphinx") -> Dict[str, Any]:
    app.add_source_parser(RSTParser)

    return {
        'version': 'builtin',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
```
### 3 - sphinx/util/smartypants.py:

Start line: 376, End line: 388

```python
if docutils_version < (0, 13, 2):
    # Monkey patch the old docutils versions to fix the issues mentioned
    # at https://sourceforge.net/p/docutils/bugs/313/
    # at https://sourceforge.net/p/docutils/bugs/317/
    # and more
    smartquotes.educateQuotes = educateQuotes
    smartquotes.educate_tokens = educate_tokens

    # Fix the issue with French quotes mentioned at
    # https://sourceforge.net/p/docutils/mailman/message/35760696/
    # Add/fix other languages as well
    smartquotes.smartchars.quotes = langquotes
```
### 4 - utils/doclinter.py:

Start line: 11, End line: 81

```python
import os
import re
import sys
from typing import List


MAX_LINE_LENGTH = 85
LONG_INTERPRETED_TEXT = re.compile(r'^\s*\W*(:(\w+:)+)?`.*`\W*$')
CODE_BLOCK_DIRECTIVE = re.compile(r'^(\s*)\.\. code-block::')
LEADING_SPACES = re.compile(r'^(\s*)')


def lint(path: str) -> int:
    with open(path) as f:
        document = f.readlines()

    errors = 0
    in_code_block = False
    code_block_depth = 0
    for i, line in enumerate(document):
        if line.endswith(' '):
            print('%s:%d: the line ends with whitespace.' %
                  (path, i + 1))
            errors += 1

        matched = CODE_BLOCK_DIRECTIVE.match(line)
        if matched:
            in_code_block = True
            code_block_depth = len(matched.group(1))
        elif in_code_block:
            if line.strip() == '':
                pass
            else:
                spaces = LEADING_SPACES.match(line).group(1)
                if len(spaces) < code_block_depth:
                    in_code_block = False
        elif LONG_INTERPRETED_TEXT.match(line):
            pass
        elif len(line) > MAX_LINE_LENGTH:
            if re.match(r'^\s*\.\. ', line):
                # ignore directives and hyperlink targets
                pass
            else:
                print('%s:%d: the line is too long (%d > %d).' %
                      (path, i + 1, len(line), MAX_LINE_LENGTH))
                errors += 1

    return errors


def main(args: List[str]) -> int:
    errors = 0
    for path in args:
        if os.path.isfile(path):
            errors += lint(path)
        elif os.path.isdir(path):
            for root, dirs, files in os.walk(path):
                for filename in files:
                    if filename.endswith('.rst'):
                        path = os.path.join(root, filename)
                        errors += lint(path)

    if errors:
        return 1
    else:
        return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
```
### 5 - doc/conf.py:

Start line: 59, End line: 126

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

# We're not using intersphinx right now, but if we did, this would be part of
# the mapping:
intersphinx_mapping = {'python': ('https://docs.python.org/3/', None)}

# Sphinx document translation with sphinx gettext feature uses these settings:
locale_dirs = ['locale/']
gettext_compact = False


# -- Extension interface -------------------------------------------------------

from sphinx import addnodes  # noqa

event_sig_re = re.compile(r'([a-zA-Z-]+)\s*\((.*)\)')
```
### 6 - sphinx/highlighting.py:

Start line: 11, End line: 54

```python
from functools import partial
from importlib import import_module
from typing import Any, Dict

from pygments import highlight
from pygments.filters import ErrorToken
from pygments.formatter import Formatter
from pygments.formatters import HtmlFormatter, LatexFormatter
from pygments.lexer import Lexer
from pygments.lexers import get_lexer_by_name, guess_lexer
from pygments.lexers import PythonLexer, Python3Lexer, PythonConsoleLexer, \
    CLexer, TextLexer, RstLexer
from pygments.style import Style
from pygments.styles import get_style_by_name
from pygments.util import ClassNotFound

from sphinx.locale import __
from sphinx.pygments_styles import SphinxStyle, NoneStyle
from sphinx.util import logging, texescape


logger = logging.getLogger(__name__)

lexers = {}  # type: Dict[str, Lexer]
lexer_classes = {
    'none': partial(TextLexer, stripnl=False),
    'python': partial(PythonLexer, stripnl=False),
    'python3': partial(Python3Lexer, stripnl=False),
    'pycon': partial(PythonConsoleLexer, stripnl=False),
    'pycon3': partial(PythonConsoleLexer, python3=True, stripnl=False),
    'rest': partial(RstLexer, stripnl=False),
    'c': partial(CLexer, stripnl=False),
}  # type: Dict[str, Lexer]


escape_hl_chars = {ord('\\'): '\\PYGZbs{}',
                   ord('{'): '\\PYGZob{}',
                   ord('}'): '\\PYGZcb{}'}

# used if Pygments is available
# use textcomp quote to get a true single quote
_LATEX_ADD_STYLES = r'''
\renewcommand\PYGZsq{\textquotesingle}
'''
```
### 7 - sphinx/util/jsdump.py:

Start line: 12, End line: 32

```python
import re
from typing import Any, Dict, IO, List, Match, Union

_str_re = re.compile(r'"(\\\\|\\"|[^"])*"')
_int_re = re.compile(r'\d+')
_name_re = re.compile(r'[a-zA-Z_]\w*')
_nameonly_re = re.compile(r'[a-zA-Z_][a-zA-Z0-9_]*$')

# escape \, ", control characters and everything outside ASCII
ESCAPE_ASCII = re.compile(r'([\\"]|[^\ -~])')
ESCAPE_DICT = {
    '\\': '\\\\',
    '"': '\\"',
    '\b': '\\b',
    '\f': '\\f',
    '\n': '\\n',
    '\r': '\\r',
    '\t': '\\t',
}

ESCAPED = re.compile(r'\\u.{4}|\\.')
```
### 8 - doc/conf.py:

Start line: 1, End line: 58

```python
# Sphinx documentation build configuration file

import re

import sphinx


extensions = ['sphinx.ext.autodoc', 'sphinx.ext.doctest', 'sphinx.ext.todo',
              'sphinx.ext.autosummary', 'sphinx.ext.extlinks',
              'sphinx.ext.intersphinx',
              'sphinx.ext.viewcode', 'sphinx.ext.inheritance_diagram']

master_doc = 'contents'
templates_path = ['_templates']
exclude_patterns = ['_build']

project = 'Sphinx'
copyright = '2007-2020, Georg Brandl and the Sphinx team'
version = sphinx.__display_version__
release = version
show_authors = True

html_theme = 'sphinx13'
html_theme_path = ['_themes']
modindex_common_prefix = ['sphinx.']
html_static_path = ['_static']
html_sidebars = {'index': ['indexsidebar.html', 'searchbox.html']}
html_additional_pages = {'index': 'index.html'}
html_use_opensearch = 'https://www.sphinx-doc.org/en/master'
html_baseurl = 'https://www.sphinx-doc.org/en/master/'

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
```
### 9 - sphinx/util/__init__.py:

Start line: 11, End line: 75

```python
import fnmatch
import functools
import os
import posixpath
import re
import sys
import tempfile
import traceback
import unicodedata
import warnings
from codecs import BOM_UTF8
from collections import deque
from datetime import datetime
from hashlib import md5
from importlib import import_module
from os import path
from time import mktime, strptime
from typing import Any, Callable, Dict, IO, Iterable, Iterator, List, Pattern, Set, Tuple
from urllib.parse import urlsplit, urlunsplit, quote_plus, parse_qsl, urlencode

from sphinx.deprecation import RemovedInSphinx40Warning
from sphinx.errors import (
    PycodeError, SphinxParallelError, ExtensionError, FiletypeNotFoundError
)
from sphinx.locale import __
from sphinx.util import logging
from sphinx.util.console import strip_colors, colorize, bold, term_width_line  # type: ignore
from sphinx.util.typing import PathMatcher
from sphinx.util import smartypants  # noqa

# import other utilities; partly for backwards compatibility, so don't
# prune unused ones indiscriminately
from sphinx.util.osutil import (  # noqa
    SEP, os_path, relative_uri, ensuredir, walk, mtimes_of_files, movefile,
    copyfile, copytimes, make_filename)
from sphinx.util.nodes import (   # noqa
    nested_parse_with_titles, split_explicit_title, explicit_title_re,
    caption_ref_re)
from sphinx.util.matching import patfilter  # noqa


if False:
    # For type annotation
    from typing import Type  # for python3.5.1
    from sphinx.application import Sphinx


logger = logging.getLogger(__name__)

# Generally useful regular expressions.
ws_re = re.compile(r'\s+')                      # type: Pattern
url_re = re.compile(r'(?P<schema>.+)://.*')     # type: Pattern


# High-level utility functions.

def docname_join(basedocname: str, docname: str) -> str:
    return posixpath.normpath(
        posixpath.join('/' + basedocname, '..', docname))[1:]


def path_stabilize(filepath: str) -> str:
    "normalize path separater and unicode string"
    newpath = filepath.replace(os.path.sep, SEP)
    return unicodedata.normalize('NFC', newpath)
```
### 10 - sphinx/cmd/quickstart.py:

Start line: 11, End line: 119

```python
import argparse
import locale
import os
import re
import sys
import time
import warnings
from collections import OrderedDict
from os import path
from typing import Any, Callable, Dict, List, Pattern, Union

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
    USE_LIBEDIT = False

from docutils.utils import column_width

import sphinx.locale
from sphinx import __display_version__, package_dir
from sphinx.deprecation import RemovedInSphinx40Warning
from sphinx.locale import __
from sphinx.util.console import (  # type: ignore
    colorize, bold, red, turquoise, nocolor, color_terminal
)
from sphinx.util.osutil import ensuredir
from sphinx.util.template import SphinxRenderer

TERM_ENCODING = getattr(sys.stdin, 'encoding', None)  # RemovedInSphinx40Warning

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
### 21 - sphinx/transforms/__init__.py:

Start line: 11, End line: 44

```python
import re
from typing import Any, Dict, Generator, List, Tuple

from docutils import nodes
from docutils.nodes import Element, Node, Text
from docutils.transforms import Transform, Transformer
from docutils.transforms.parts import ContentsFilter
from docutils.transforms.universal import SmartQuotes
from docutils.utils import normalize_language_tag
from docutils.utils.smartquotes import smartchars

from sphinx import addnodes
from sphinx.config import Config
from sphinx.deprecation import RemovedInSphinx40Warning, deprecated_alias
from sphinx.locale import _, __
from sphinx.util import logging
from sphinx.util.docutils import new_document
from sphinx.util.i18n import format_date
from sphinx.util.nodes import NodeMatcher, apply_source_workaround, is_smartquotable

if False:
    # For type annotation
    from sphinx.application import Sphinx
    from sphinx.domain.std import StandardDomain
    from sphinx.environment import BuildEnvironment


logger = logging.getLogger(__name__)

default_substitutions = {
    'version',
    'release',
    'today',
}
```
### 83 - sphinx/transforms/__init__.py:

Start line: 274, End line: 315

```python
class DoctestTransform(SphinxTransform):
    """Set "doctest" style to each doctest_block node"""
    default_priority = 500

    def apply(self, **kwargs: Any) -> None:
        for node in self.document.traverse(nodes.doctest_block):
            node['classes'].append('doctest')


class FigureAligner(SphinxTransform):
    """
    Align figures to center by default.
    """
    default_priority = 700

    def apply(self, **kwargs: Any) -> None:
        matcher = NodeMatcher(nodes.table, nodes.figure)
        for node in self.document.traverse(matcher):  # type: Element
            node.setdefault('align', 'default')


class FilterSystemMessages(SphinxTransform):
    """Filter system messages from a doctree."""
    default_priority = 999

    def apply(self, **kwargs: Any) -> None:
        filterlevel = 2 if self.config.keep_warnings else 5
        for node in self.document.traverse(nodes.system_message):
            if node['level'] < filterlevel:
                logger.debug('%s [filtered system message]', node.astext())
                node.parent.remove(node)


class SphinxContentsFilter(ContentsFilter):
    """
    Used with BuildEnvironment.add_toc_from() to discard cross-file links
    within table-of-contents link nodes.
    """
    visit_pending_xref = ContentsFilter.ignore_node_but_process_children

    def visit_image(self, node: nodes.image) -> None:
        raise nodes.SkipNode
```
