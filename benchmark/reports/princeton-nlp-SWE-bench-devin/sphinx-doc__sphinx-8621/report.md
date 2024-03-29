# sphinx-doc__sphinx-8621

| **sphinx-doc/sphinx** | `21698c14461d27933864d73e6fba568a154e83b3` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 324 |
| **Any found context length** | 324 |
| **Avg pos** | 1.0 |
| **Min pos** | 1 |
| **Max pos** | 1 |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sphinx/builders/html/transforms.py b/sphinx/builders/html/transforms.py
--- a/sphinx/builders/html/transforms.py
+++ b/sphinx/builders/html/transforms.py
@@ -37,7 +37,7 @@ class KeyboardTransform(SphinxPostTransform):
     """
     default_priority = 400
     builders = ('html',)
-    pattern = re.compile(r'(-|\+|\^|\s+)')
+    pattern = re.compile(r'(?<=.)(-|\+|\^|\s+)(?=.)')
 
     def run(self, **kwargs: Any) -> None:
         matcher = NodeMatcher(nodes.literal, classes=["kbd"])

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sphinx/builders/html/transforms.py | 40 | 40 | 1 | 1 | 324


## Problem Statement

```
kbd role produces incorrect HTML when compound-key separators (-, + or ^) are used as keystrokes
**Describe the bug**

The `:kbd:` role produces incorrect HTML when:

1) defining standalone keystrokes that use any of the compound-key separators (`-`, `+` and `^`)
2) defining compound keystrokes where one or more keystrokes use any of the compound-key separators (`-`, `+` and `^`)

**To Reproduce**

For the below three keyboard definitions:
\`\`\`
(1) :kbd:`-`
(2) :kbd:`+`
(3) :kbd:`Shift-+`
\`\`\`

The following three incorrect output is generated:

(1) `-` is treated as a separator with two "blank" keystrokes around it.

\`\`\`
<kbd class="kbd docutils literal notranslate"><kbd class="kbd docutils literal notranslate"></kbd>-<kbd class="kbd docutils literal notranslate"></kbd></kbd>
\`\`\`

(2) `+` is treated as a separator with two "blank" keystrokes around it.

\`\`\`
<kbd class="kbd docutils literal notranslate"><kbd class="kbd docutils literal notranslate"></kbd>+<kbd class="kbd docutils literal notranslate"></kbd></kbd>
\`\`\`

(3) `+` is treated as a separator within a compound-keystroke, with two "blank" keystrokes around it.

\`\`\`
<kbd class="kbd docutils literal notranslate"><kbd class="kbd docutils literal notranslate">Shift</kbd>-<kbd class="kbd docutils literal notranslate"></kbd>+<kbd class="kbd docutils literal notranslate"></kbd></kbd>
\`\`\`

**Expected behavior**

For single keystrokes that use `-`, `+` or`^`, just a single `kbd` element should be created.

For compound-keystrokes, the algorithm should differentiate between `-`, `+` and `^` characters appearing in separator vs keystroke positions (currently, it's very simplistic, it just treats all these characters as separators using a simple regexp).

**Screenshot**

![image](https://user-images.githubusercontent.com/698770/103331652-a2268680-4ab2-11eb-953a-2f50c8cb7a00.png)


**Environment info**
- OS: Windows
- Python version: 3.9.1
- Sphinx version: 3.4.0
- Sphinx extensions:  -
- Extra tools: -


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 sphinx/builders/html/transforms.py** | 11 | 70| 324 | 324 | 377 | 
| 2 | 2 sphinx/roles.py | 406 | 456| 492 | 816 | 6029 | 
| 3 | 2 sphinx/roles.py | 459 | 506| 400 | 1216 | 6029 | 
| 4 | 3 sphinx/util/texescape.py | 11 | 63| 623 | 1839 | 7798 | 
| 5 | 3 sphinx/roles.py | 509 | 525| 188 | 2027 | 7798 | 
| 6 | 3 sphinx/roles.py | 197 | 263| 701 | 2728 | 7798 | 
| 7 | 3 sphinx/roles.py | 11 | 46| 282 | 3010 | 7798 | 
| 8 | 4 sphinx/util/docutils.py | 360 | 380| 173 | 3183 | 11921 | 
| 9 | 5 sphinx/util/smartypants.py | 28 | 122| 1414 | 4597 | 16032 | 
| 10 | 6 sphinx/util/nodes.py | 469 | 512| 672 | 5269 | 21438 | 
| 11 | 7 sphinx/builders/latex/constants.py | 120 | 202| 925 | 6194 | 23518 | 
| 12 | 7 sphinx/util/docutils.py | 419 | 435| 186 | 6380 | 23518 | 
| 13 | 7 sphinx/roles.py | 91 | 105| 156 | 6536 | 23518 | 
| 14 | 7 sphinx/roles.py | 376 | 403| 225 | 6761 | 23518 | 
| 15 | 7 sphinx/roles.py | 528 | 540| 111 | 6872 | 23518 | 
| 16 | 8 sphinx/util/rst.py | 11 | 76| 541 | 7413 | 24371 | 
| 17 | 8 sphinx/roles.py | 335 | 373| 330 | 7743 | 24371 | 
| 18 | 9 sphinx/util/cfamily.py | 11 | 78| 763 | 8506 | 27873 | 
| 19 | 10 sphinx/ext/napoleon/docstring.py | 13 | 59| 515 | 9021 | 38540 | 
| 20 | 11 sphinx/util/__init__.py | 11 | 71| 506 | 9527 | 44842 | 
| 21 | 12 sphinx/domains/python.py | 1032 | 1052| 248 | 9775 | 56779 | 
| 22 | 13 sphinx/highlighting.py | 11 | 52| 373 | 10148 | 58081 | 
| 23 | 14 sphinx/writers/html.py | 697 | 796| 803 | 10951 | 65477 | 
| 24 | 15 sphinx/domains/cpp.py | 6499 | 7236| 6339 | 17290 | 128570 | 
| 25 | 16 sphinx/search/ja.py | 140 | 166| 755 | 18045 | 141210 | 
| 26 | 17 sphinx/domains/javascript.py | 299 | 316| 196 | 18241 | 145257 | 
| 27 | 17 sphinx/domains/python.py | 11 | 78| 526 | 18767 | 145257 | 
| 28 | 17 sphinx/util/texescape.py | 142 | 169| 259 | 19026 | 145257 | 
| 29 | 17 sphinx/highlighting.py | 55 | 164| 876 | 19902 | 145257 | 
| 30 | 17 sphinx/util/docutils.py | 342 | 358| 178 | 20080 | 145257 | 
| 31 | 17 sphinx/roles.py | 266 | 298| 337 | 20417 | 145257 | 
| 32 | 18 sphinx/writers/text.py | 1044 | 1156| 813 | 21230 | 154230 | 
| 33 | 19 sphinx/ext/extlinks.py | 26 | 71| 400 | 21630 | 154849 | 
| 34 | 19 sphinx/roles.py | 107 | 140| 338 | 21968 | 154849 | 
| 35 | 19 sphinx/roles.py | 543 | 572| 328 | 22296 | 154849 | 
| 36 | 19 sphinx/roles.py | 49 | 89| 373 | 22669 | 154849 | 
| 37 | 20 sphinx/domains/c.py | 3412 | 3836| 3706 | 26375 | 186044 | 
| 38 | 21 sphinx/writers/html5.py | 633 | 719| 686 | 27061 | 192967 | 
| 39 | 21 sphinx/writers/html.py | 354 | 409| 472 | 27533 | 192967 | 


## Patch

```diff
diff --git a/sphinx/builders/html/transforms.py b/sphinx/builders/html/transforms.py
--- a/sphinx/builders/html/transforms.py
+++ b/sphinx/builders/html/transforms.py
@@ -37,7 +37,7 @@ class KeyboardTransform(SphinxPostTransform):
     """
     default_priority = 400
     builders = ('html',)
-    pattern = re.compile(r'(-|\+|\^|\s+)')
+    pattern = re.compile(r'(?<=.)(-|\+|\^|\s+)(?=.)')
 
     def run(self, **kwargs: Any) -> None:
         matcher = NodeMatcher(nodes.literal, classes=["kbd"])

```

## Test Patch

```diff
diff --git a/tests/test_markup.py b/tests/test_markup.py
--- a/tests/test_markup.py
+++ b/tests/test_markup.py
@@ -251,6 +251,17 @@ def get(name):
          '</kbd></p>'),
         '\\sphinxkeyboard{\\sphinxupquote{Control+X}}',
     ),
+    (
+        # kbd role
+        'verify',
+        ':kbd:`Alt+^`',
+        ('<p><kbd class="kbd docutils literal notranslate">'
+         '<kbd class="kbd docutils literal notranslate">Alt</kbd>'
+         '+'
+         '<kbd class="kbd docutils literal notranslate">^</kbd>'
+         '</kbd></p>'),
+        '\\sphinxkeyboard{\\sphinxupquote{Alt+\\textasciicircum{}}}',
+    ),
     (
         # kbd role
         'verify',
@@ -266,6 +277,13 @@ def get(name):
          '</kbd></p>'),
         '\\sphinxkeyboard{\\sphinxupquote{M\\sphinxhyphen{}x  M\\sphinxhyphen{}s}}',
     ),
+    (
+        # kbd role
+        'verify',
+        ':kbd:`-`',
+        '<p><kbd class="kbd docutils literal notranslate">-</kbd></p>',
+        '\\sphinxkeyboard{\\sphinxupquote{\\sphinxhyphen{}}}',
+    ),
     (
         # non-interpolation of dashes in option role
         'verify_re',

```


## Code snippets

### 1 - sphinx/builders/html/transforms.py:

Start line: 11, End line: 70

```python
import re
from typing import Any, Dict

from docutils import nodes

from sphinx.application import Sphinx
from sphinx.transforms.post_transforms import SphinxPostTransform
from sphinx.util.nodes import NodeMatcher


class KeyboardTransform(SphinxPostTransform):
    """Transform :kbd: role to more detailed form.

    Before::

        <literal class="kbd">
            Control-x

    After::

        <literal class="kbd">
            <literal class="kbd">
                Control
            -
            <literal class="kbd">
                x
    """
    default_priority = 400
    builders = ('html',)
    pattern = re.compile(r'(-|\+|\^|\s+)')

    def run(self, **kwargs: Any) -> None:
        matcher = NodeMatcher(nodes.literal, classes=["kbd"])
        for node in self.document.traverse(matcher):  # type: nodes.literal
            parts = self.pattern.split(node[-1].astext())
            if len(parts) == 1:
                continue

            node.pop()
            while parts:
                key = parts.pop(0)
                node += nodes.literal('', key, classes=["kbd"])

                try:
                    # key separator (ex. -, +, ^)
                    sep = parts.pop(0)
                    node += nodes.Text(sep)
                except IndexError:
                    pass


def setup(app: Sphinx) -> Dict[str, Any]:
    app.add_post_transform(KeyboardTransform)

    return {
        'version': 'builtin',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
```
### 2 - sphinx/roles.py:

Start line: 406, End line: 456

```python
def emph_literal_role(typ: str, rawtext: str, text: str, lineno: int, inliner: Inliner,
                      options: Dict = {}, content: List[str] = []
                      ) -> Tuple[List[Node], List[system_message]]:
    warnings.warn('emph_literal_role() is deprecated. '
                  'Please use EmphasizedLiteral class instead.',
                  RemovedInSphinx40Warning, stacklevel=2)
    env = inliner.document.settings.env
    if not typ:
        assert env.temp_data['default_role']
        typ = env.temp_data['default_role'].lower()
    else:
        typ = typ.lower()

    retnode = nodes.literal(role=typ.lower(), classes=[typ])
    parts = list(parens_re.split(utils.unescape(text)))
    stack = ['']
    for part in parts:
        matched = parens_re.match(part)
        if matched:
            backslashes = len(part) - 1
            if backslashes % 2 == 1:    # escaped
                stack[-1] += "\\" * int((backslashes - 1) / 2) + part[-1]
            elif part[-1] == '{':       # rparen
                stack[-1] += "\\" * int(backslashes / 2)
                if len(stack) >= 2 and stack[-2] == "{":
                    # nested
                    stack[-1] += "{"
                else:
                    # start emphasis
                    stack.append('{')
                    stack.append('')
            else:                       # lparen
                stack[-1] += "\\" * int(backslashes / 2)
                if len(stack) == 3 and stack[1] == "{" and len(stack[2]) > 0:
                    # emphasized word found
                    if stack[0]:
                        retnode += nodes.Text(stack[0], stack[0])
                    retnode += nodes.emphasis(stack[2], stack[2])
                    stack = ['']
                else:
                    # emphasized word not found; the rparen is not a special symbol
                    stack.append('}')
                    stack = [''.join(stack)]
        else:
            stack[-1] += part
    if ''.join(stack):
        # remaining is treated as Text
        text = ''.join(stack)
        retnode += nodes.Text(text, text)

    return [retnode], []
```
### 3 - sphinx/roles.py:

Start line: 459, End line: 506

```python
class EmphasizedLiteral(SphinxRole):
    parens_re = re.compile(r'(\\\\|\\{|\\}|{|})')

    def run(self) -> Tuple[List[Node], List[system_message]]:
        children = self.parse(self.text)
        node = nodes.literal(self.rawtext, '', *children,
                             role=self.name.lower(), classes=[self.name])

        return [node], []

    def parse(self, text: str) -> List[Node]:
        result = []  # type: List[Node]

        stack = ['']
        for part in self.parens_re.split(text):
            if part == '\\\\':  # escaped backslash
                stack[-1] += '\\'
            elif part == '{':
                if len(stack) >= 2 and stack[-2] == "{":  # nested
                    stack[-1] += "{"
                else:
                    # start emphasis
                    stack.append('{')
                    stack.append('')
            elif part == '}':
                if len(stack) == 3 and stack[1] == "{" and len(stack[2]) > 0:
                    # emphasized word found
                    if stack[0]:
                        result.append(nodes.Text(stack[0], stack[0]))
                    result.append(nodes.emphasis(stack[2], stack[2]))
                    stack = ['']
                else:
                    # emphasized word not found; the rparen is not a special symbol
                    stack.append('}')
                    stack = [''.join(stack)]
            elif part == '\\{':  # escaped left-brace
                stack[-1] += '{'
            elif part == '\\}':  # escaped right-brace
                stack[-1] += '}'
            else:  # others (containing escaped braces)
                stack[-1] += part

        if ''.join(stack):
            # remaining is treated as Text
            text = ''.join(stack)
            result.append(nodes.Text(text, text))

        return result
```
### 4 - sphinx/util/texescape.py:

Start line: 11, End line: 63

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
### 5 - sphinx/roles.py:

Start line: 509, End line: 525

```python
_abbr_re = re.compile(r'\((.*)\)$', re.S)


def abbr_role(typ: str, rawtext: str, text: str, lineno: int, inliner: Inliner,
              options: Dict = {}, content: List[str] = []
              ) -> Tuple[List[Node], List[system_message]]:
    warnings.warn('abbr_role() is deprecated.  Please use Abbrevation class instead.',
                  RemovedInSphinx40Warning, stacklevel=2)
    text = utils.unescape(text)
    m = _abbr_re.search(text)
    if m is None:
        return [nodes.abbreviation(text, text, **options)], []
    abbr = text[:m.start()].strip()
    expl = m.group(1)
    options = options.copy()
    options['explanation'] = expl
    return [nodes.abbreviation(abbr, abbr, **options)], []
```
### 6 - sphinx/roles.py:

Start line: 197, End line: 263

```python
def indexmarkup_role(typ: str, rawtext: str, text: str, lineno: int, inliner: Inliner,
                     options: Dict = {}, content: List[str] = []
                     ) -> Tuple[List[Node], List[system_message]]:
    """Role for PEP/RFC references that generate an index entry."""
    warnings.warn('indexmarkup_role() is deprecated.  Please use PEP or RFC class instead.',
                  RemovedInSphinx40Warning, stacklevel=2)
    env = inliner.document.settings.env
    if not typ:
        assert env.temp_data['default_role']
        typ = env.temp_data['default_role'].lower()
    else:
        typ = typ.lower()

    has_explicit_title, title, target = split_explicit_title(text)
    title = utils.unescape(title)
    target = utils.unescape(target)
    targetid = 'index-%s' % env.new_serialno('index')
    indexnode = addnodes.index()
    targetnode = nodes.target('', '', ids=[targetid])
    inliner.document.note_explicit_target(targetnode)
    if typ == 'pep':
        indexnode['entries'] = [
            ('single', _('Python Enhancement Proposals; PEP %s') % target,
             targetid, '', None)]
        anchor = ''
        anchorindex = target.find('#')
        if anchorindex > 0:
            target, anchor = target[:anchorindex], target[anchorindex:]
        if not has_explicit_title:
            title = "PEP " + utils.unescape(title)
        try:
            pepnum = int(target)
        except ValueError:
            msg = inliner.reporter.error('invalid PEP number %s' % target,
                                         line=lineno)
            prb = inliner.problematic(rawtext, rawtext, msg)
            return [prb], [msg]
        ref = inliner.document.settings.pep_base_url + 'pep-%04d' % pepnum
        sn = nodes.strong(title, title)
        rn = nodes.reference('', '', internal=False, refuri=ref + anchor,
                             classes=[typ])
        rn += sn
        return [indexnode, targetnode, rn], []
    elif typ == 'rfc':
        indexnode['entries'] = [
            ('single', 'RFC; RFC %s' % target, targetid, '', None)]
        anchor = ''
        anchorindex = target.find('#')
        if anchorindex > 0:
            target, anchor = target[:anchorindex], target[anchorindex:]
        if not has_explicit_title:
            title = "RFC " + utils.unescape(title)
        try:
            rfcnum = int(target)
        except ValueError:
            msg = inliner.reporter.error('invalid RFC number %s' % target,
                                         line=lineno)
            prb = inliner.problematic(rawtext, rawtext, msg)
            return [prb], [msg]
        ref = inliner.document.settings.rfc_base_url + inliner.rfc_url % rfcnum
        sn = nodes.strong(title, title)
        rn = nodes.reference('', '', internal=False, refuri=ref + anchor,
                             classes=[typ])
        rn += sn
        return [indexnode, targetnode, rn], []
    else:
        raise ValueError('unknown role type: %s' % typ)
```
### 7 - sphinx/roles.py:

Start line: 11, End line: 46

```python
import re
import warnings
from typing import Any, Dict, List, Tuple

from docutils import nodes, utils
from docutils.nodes import Element, Node, TextElement, system_message
from docutils.parsers.rst.states import Inliner

from sphinx import addnodes
from sphinx.deprecation import RemovedInSphinx40Warning
from sphinx.locale import _
from sphinx.util import ws_re
from sphinx.util.docutils import ReferenceRole, SphinxRole
from sphinx.util.nodes import process_index_entry, set_role_source_info, split_explicit_title
from sphinx.util.typing import RoleFunction

if False:
    # For type annotation
    from typing import Type  # for python3.5.1

    from sphinx.application import Sphinx
    from sphinx.environment import BuildEnvironment


generic_docroles = {
    'command': addnodes.literal_strong,
    'dfn': nodes.emphasis,
    'kbd': nodes.literal,
    'mailheader': addnodes.literal_emphasis,
    'makevar': addnodes.literal_strong,
    'manpage': addnodes.manpage,
    'mimetype': addnodes.literal_emphasis,
    'newsgroup': addnodes.literal_emphasis,
    'program': addnodes.literal_strong,  # XXX should be an x-ref
    'regexp': nodes.literal,
}
```
### 8 - sphinx/util/docutils.py:

Start line: 360, End line: 380

```python
class SphinxRole:

    def __call__(self, name: str, rawtext: str, text: str, lineno: int,
                 inliner: Inliner, options: Dict = {}, content: List[str] = []
                 ) -> Tuple[List[Node], List[system_message]]:
        self.rawtext = rawtext
        self.text = unescape(text)
        self.lineno = lineno
        self.inliner = inliner
        self.options = options
        self.content = content

        # guess role type
        if name:
            self.name = name.lower()
        else:
            self.name = self.env.temp_data.get('default_role')
            if not self.name:
                self.name = self.env.config.default_role
            if not self.name:
                raise SphinxError('cannot determine default role!')

        return self.run()
```
### 9 - sphinx/util/smartypants.py:

Start line: 28, End line: 122

```python
import re
from typing import Generator, Iterable, Tuple

from docutils.utils import smartquotes

from sphinx.util.docutils import __version_info__ as docutils_version

langquotes = {'af':           '“”‘’',
              'af-x-altquot': '„”‚’',
              'bg':           '„“‚‘',  # Bulgarian, https://bg.wikipedia.org/wiki/Кавички
              'ca':           '«»“”',
              'ca-x-altquot': '“”‘’',
              'cs':           '„“‚‘',
              'cs-x-altquot': '»«›‹',
              'da':           '»«›‹',
              'da-x-altquot': '„“‚‘',
              # 'da-x-altquot2': '””’’',
              'de':           '„“‚‘',
              'de-x-altquot': '»«›‹',
              'de-ch':        '«»‹›',
              'el':           '«»“”',
              'en':           '“”‘’',
              'en-uk-x-altquot': '‘’“”',  # Attention: " → ‘ and ' → “ !
              'eo':           '“”‘’',
              'es':           '«»“”',
              'es-x-altquot': '“”‘’',
              'et':           '„“‚‘',  # no secondary quote listed in
              'et-x-altquot': '«»‹›',  # the sources above (wikipedia.org)
              'eu':           '«»‹›',
              'fi':           '””’’',
              'fi-x-altquot': '»»››',
              'fr':           ('« ', ' »', '“', '”'),  # full no-break space
              'fr-x-altquot': ('« ', ' »', '“', '”'),  # narrow no-break space
              'fr-ch':        '«»‹›',
              'fr-ch-x-altquot': ('« ',  ' »', '‹ ', ' ›'),  # narrow no-break space
              # http://typoguide.ch/
              'gl':           '«»“”',
              'he':           '”“»«',  # Hebrew is RTL, test position:
              'he-x-altquot': '„”‚’',  # low quotation marks are opening.
              # 'he-x-altquot': '“„‘‚',  # RTL: low quotation marks opening
              'hr':           '„”‘’',  # https://hrvatska-tipografija.com/polunavodnici/
              'hr-x-altquot': '»«›‹',
              'hsb':          '„“‚‘',
              'hsb-x-altquot': '»«›‹',
              'hu':           '„”«»',
              'is':           '„“‚‘',
              'it':           '«»“”',
              'it-ch':        '«»‹›',
              'it-x-altquot': '“”‘’',
              # 'it-x-altquot2': '“„‘‚',  # [7] in headlines
              'ja':           '「」『』',
              'lt':           '„“‚‘',
              'lv':           '„“‚‘',
              'mk':           '„“‚‘',  # Macedonian,
              # https://mk.wikipedia.org/wiki/Правопис_и_правоговор_на_македонскиот_јазик
              'nl':           '“”‘’',
              'nl-x-altquot': '„”‚’',
              # 'nl-x-altquot2': '””’’',
              'nb':           '«»’’',  # Norsk bokmål (canonical form 'no')
              'nn':           '«»’’',  # Nynorsk [10]
              'nn-x-altquot': '«»‘’',  # [8], [10]
              # 'nn-x-altquot2': '«»«»',  # [9], [10]
              # 'nn-x-altquot3': '„“‚‘',  # [10]
              'no':           '«»’’',  # Norsk bokmål [10]
              'no-x-altquot': '«»‘’',  # [8], [10]
              # 'no-x-altquot2': '«»«»',  # [9], [10]
              # 'no-x-altquot3': '„“‚‘',  # [10]
              'pl':           '„”«»',
              'pl-x-altquot': '«»‚’',
              # 'pl-x-altquot2': '„”‚’',
              # https://pl.wikipedia.org/wiki/Cudzys%C5%82%C3%B3w
              'pt':           '«»“”',
              'pt-br':        '“”‘’',
              'ro':           '„”«»',
              'ru':           '«»„“',
              'sh':           '„”‚’',  # Serbo-Croatian
              'sh-x-altquot': '»«›‹',
              'sk':           '„“‚‘',  # Slovak
              'sk-x-altquot': '»«›‹',
              'sl':           '„“‚‘',  # Slovenian
              'sl-x-altquot': '»«›‹',
              'sq':           '«»‹›',  # Albanian
              'sq-x-altquot': '“„‘‚',
              'sr':           '„”’’',
              'sr-x-altquot': '»«›‹',
              'sv':           '””’’',
              'sv-x-altquot': '»»››',
              'tr':           '“”‘’',
              'tr-x-altquot': '«»‹›',
              # 'tr-x-altquot2': '“„‘‚',  # [7] antiquated?
              'uk':           '«»„“',
              'uk-x-altquot': '„“‚‘',
              'zh-cn':        '“”‘’',
              'zh-tw':        '「」『』',
              }
```
### 10 - sphinx/util/nodes.py:

Start line: 469, End line: 512

```python
_non_id_chars = re.compile('[^a-zA-Z0-9._]+')
_non_id_at_ends = re.compile('^[-0-9._]+|-+$')
_non_id_translate = {
    0x00f8: 'o',       # o with stroke
    0x0111: 'd',       # d with stroke
    0x0127: 'h',       # h with stroke
    0x0131: 'i',       # dotless i
    0x0142: 'l',       # l with stroke
    0x0167: 't',       # t with stroke
    0x0180: 'b',       # b with stroke
    0x0183: 'b',       # b with topbar
    0x0188: 'c',       # c with hook
    0x018c: 'd',       # d with topbar
    0x0192: 'f',       # f with hook
    0x0199: 'k',       # k with hook
    0x019a: 'l',       # l with bar
    0x019e: 'n',       # n with long right leg
    0x01a5: 'p',       # p with hook
    0x01ab: 't',       # t with palatal hook
    0x01ad: 't',       # t with hook
    0x01b4: 'y',       # y with hook
    0x01b6: 'z',       # z with stroke
    0x01e5: 'g',       # g with stroke
    0x0225: 'z',       # z with hook
    0x0234: 'l',       # l with curl
    0x0235: 'n',       # n with curl
    0x0236: 't',       # t with curl
    0x0237: 'j',       # dotless j
    0x023c: 'c',       # c with stroke
    0x023f: 's',       # s with swash tail
    0x0240: 'z',       # z with swash tail
    0x0247: 'e',       # e with stroke
    0x0249: 'j',       # j with stroke
    0x024b: 'q',       # q with hook tail
    0x024d: 'r',       # r with stroke
    0x024f: 'y',       # y with stroke
}
_non_id_translate_digraphs = {
    0x00df: 'sz',      # ligature sz
    0x00e6: 'ae',      # ae
    0x0153: 'oe',      # ligature oe
    0x0238: 'db',      # db digraph
    0x0239: 'qp',      # qp digraph
}
```
