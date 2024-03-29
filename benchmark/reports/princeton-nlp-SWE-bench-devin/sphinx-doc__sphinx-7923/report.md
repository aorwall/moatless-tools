# sphinx-doc__sphinx-7923

| **sphinx-doc/sphinx** | `533b4ac7d6f2a1a20f08c3a595a2580a9742d944` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | - |
| **Avg pos** | - |
| **Min pos** | - |
| **Max pos** | - |
| **Top file pos** | - |
| **Missing snippets** | 1 |
| **Missing patch files** | 1 |


## Expected patch

```diff
diff --git a/sphinx/util/rst.py b/sphinx/util/rst.py
--- a/sphinx/util/rst.py
+++ b/sphinx/util/rst.py
@@ -103,6 +103,11 @@ def prepend_prolog(content: StringList, prolog: str) -> None:
 def append_epilog(content: StringList, epilog: str) -> None:
     """Append a string to content body as epilog."""
     if epilog:
-        content.append('', '<generated>', 0)
+        if 0 < len(content):
+            source, lineno = content.info(-1)
+        else:
+            source = '<generated>'
+            lineno = 0
+        content.append('', source, lineno + 1)
         for lineno, line in enumerate(epilog.splitlines()):
             content.append(line, '<rst_epilog>', lineno)

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sphinx/util/rst.py | 106 | 106 | - | - | -


## Problem Statement

```
Bad refs in pot files, when using rst_epilog
**To Reproduce**
conf.py
\`\`\`python
rst_epilog = """
.. |var1| replace:: VAR1
"""
\`\`\`
index.rst
\`\`\`
A
======

a
   b
\`\`\`

`make gettext` produces index.pot with bad string numbers and "\<generated\>" refs:
\`\`\`
#: ../../index.rst:2
msgid "A"
msgstr ""

#: ../../<generated>:1
msgid "a"
msgstr ""

#: ../../index.rst:5
msgid "b"
msgstr ""
\`\`\`



```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 doc/conf.py | 59 | 126| 693 | 693 | 1554 | 
| 2 | 2 sphinx/transforms/i18n.py | 222 | 377| 1673 | 2366 | 6135 | 
| 3 | 3 utils/doclinter.py | 11 | 81| 467 | 2833 | 6654 | 
| 4 | 3 doc/conf.py | 129 | 165| 359 | 3192 | 6654 | 
| 5 | 4 utils/checks.py | 33 | 109| 545 | 3737 | 7560 | 
| 6 | 5 sphinx/util/cfamily.py | 11 | 80| 766 | 4503 | 11071 | 
| 7 | 6 sphinx/transforms/post_transforms/__init__.py | 154 | 177| 279 | 4782 | 13071 | 
| 8 | 7 sphinx/builders/_epub_base.py | 11 | 102| 662 | 5444 | 19792 | 
| 9 | 8 sphinx/ext/autosummary/generate.py | 580 | 619| 360 | 5804 | 24975 | 
| 10 | 8 doc/conf.py | 1 | 58| 502 | 6306 | 24975 | 
| 11 | 9 sphinx/domains/rst.py | 11 | 33| 161 | 6467 | 27445 | 
| 12 | 10 sphinx/util/jsdump.py | 12 | 32| 180 | 6647 | 28860 | 
| 13 | 11 sphinx/testing/util.py | 10 | 50| 300 | 6947 | 30682 | 
| 14 | 12 sphinx/io.py | 10 | 46| 275 | 7222 | 32375 | 
| 15 | 13 sphinx/ext/napoleon/docstring.py | 13 | 40| 327 | 7549 | 41222 | 
| 16 | 13 sphinx/ext/autosummary/generate.py | 20 | 60| 283 | 7832 | 41222 | 
| 17 | 14 sphinx/builders/epub3.py | 12 | 53| 276 | 8108 | 43915 | 
| 18 | 15 sphinx/ext/autosummary/__init__.py | 55 | 108| 382 | 8490 | 50469 | 
| 19 | 16 sphinx/project.py | 11 | 27| 116 | 8606 | 51281 | 
| 20 | 16 sphinx/ext/autosummary/__init__.py | 731 | 766| 325 | 8931 | 51281 | 
| 21 | 17 sphinx/util/inspect.py | 11 | 47| 264 | 9195 | 57170 | 
| 22 | 18 sphinx/ext/apidoc.py | 347 | 412| 751 | 9946 | 61891 | 
| 23 | 19 sphinx/domains/python.py | 11 | 88| 598 | 10544 | 73571 | 
| 24 | 20 sphinx/builders/latex/constants.py | 70 | 119| 523 | 11067 | 75651 | 
| 25 | 21 sphinx/builders/gettext.py | 223 | 238| 161 | 11228 | 78402 | 
| 26 | 22 sphinx/roles.py | 92 | 106| 156 | 11384 | 84057 | 
| 27 | 23 sphinx/ext/graphviz.py | 12 | 43| 233 | 11617 | 87656 | 
| 28 | 24 setup.py | 1 | 75| 443 | 12060 | 89381 | 
| 29 | 25 sphinx/directives/patches.py | 71 | 107| 247 | 12307 | 90994 | 
| 30 | 25 sphinx/builders/_epub_base.py | 172 | 182| 133 | 12440 | 90994 | 
| 31 | 25 sphinx/domains/rst.py | 260 | 286| 274 | 12714 | 90994 | 
| 32 | 26 sphinx/util/smartypants.py | 376 | 388| 137 | 12851 | 95105 | 
| 33 | 27 sphinx/util/nodes.py | 468 | 511| 689 | 13540 | 100526 | 
| 34 | 28 sphinx/writers/manpage.py | 313 | 411| 757 | 14297 | 104037 | 
| 35 | 28 sphinx/builders/_epub_base.py | 256 | 287| 281 | 14578 | 104037 | 
| 36 | 29 sphinx/cmd/make_mode.py | 17 | 54| 515 | 15093 | 105722 | 
| 37 | 30 sphinx/directives/code.py | 418 | 482| 658 | 15751 | 109630 | 
| 38 | 31 sphinx/search/ja.py | 140 | 166| 755 | 16506 | 122270 | 
| 39 | 32 doc/development/tutorials/examples/todo.py | 30 | 53| 175 | 16681 | 123178 | 
| 40 | 32 sphinx/ext/autosummary/generate.py | 348 | 443| 783 | 17464 | 123178 | 
| 41 | 33 sphinx/util/texescape.py | 11 | 64| 623 | 18087 | 124947 | 
| 42 | 34 sphinx/writers/latex.py | 14 | 73| 453 | 18540 | 144651 | 
| 43 | 35 sphinx/ext/inheritance_diagram.py | 38 | 66| 214 | 18754 | 148506 | 
| 44 | 35 sphinx/roles.py | 198 | 264| 701 | 19455 | 148506 | 
| 45 | 36 sphinx/writers/texinfo.py | 1292 | 1375| 673 | 20128 | 160685 | 
| 46 | 36 sphinx/domains/python.py | 262 | 282| 219 | 20347 | 160685 | 
| 47 | 36 sphinx/builders/epub3.py | 194 | 230| 415 | 20762 | 160685 | 
| 48 | 37 sphinx/domains/std.py | 11 | 48| 323 | 21085 | 170899 | 
| 49 | 38 sphinx/util/pycompat.py | 11 | 54| 357 | 21442 | 171691 | 
| 50 | 39 sphinx/builders/singlehtml.py | 56 | 68| 141 | 21583 | 173550 | 
| 51 | 39 sphinx/writers/latex.py | 1588 | 1611| 290 | 21873 | 173550 | 
| 52 | 40 sphinx/util/i18n.py | 10 | 37| 166 | 22039 | 176412 | 
| 53 | 41 sphinx/ext/intersphinx.py | 261 | 339| 865 | 22904 | 180088 | 
| 54 | 42 sphinx/parsers.py | 11 | 28| 134 | 23038 | 180956 | 
| 55 | 42 sphinx/ext/autosummary/__init__.py | 716 | 728| 101 | 23139 | 180956 | 
| 56 | 43 sphinx/cmd/build.py | 11 | 30| 132 | 23271 | 183617 | 
| 57 | 43 sphinx/writers/latex.py | 1613 | 1672| 496 | 23767 | 183617 | 
| 58 | 43 sphinx/writers/texinfo.py | 862 | 966| 788 | 24555 | 183617 | 
| 59 | 43 sphinx/cmd/make_mode.py | 96 | 140| 375 | 24930 | 183617 | 
| 60 | 43 sphinx/domains/python.py | 1006 | 1026| 248 | 25178 | 183617 | 
| 61 | 44 sphinx/locale/__init__.py | 239 | 267| 232 | 25410 | 185681 | 
| 62 | 45 sphinx/config.py | 387 | 439| 474 | 25884 | 190083 | 
| 63 | 45 sphinx/util/jsdump.py | 35 | 50| 165 | 26049 | 190083 | 
| 64 | 45 setup.py | 78 | 169| 643 | 26692 | 190083 | 
| 65 | 45 sphinx/util/cfamily.py | 253 | 273| 181 | 26873 | 190083 | 
| 66 | 46 sphinx/cmd/quickstart.py | 11 | 119| 772 | 27645 | 195643 | 
| 67 | 46 sphinx/domains/python.py | 242 | 260| 214 | 27859 | 195643 | 
| 68 | 47 sphinx/__init__.py | 14 | 65| 494 | 28353 | 196211 | 
| 69 | 48 sphinx/transforms/__init__.py | 11 | 45| 234 | 28587 | 199417 | 
| 70 | 48 sphinx/ext/autosummary/generate.py | 622 | 643| 162 | 28749 | 199417 | 


## Missing Patch Files

 * 1: sphinx/util/rst.py

## Patch

```diff
diff --git a/sphinx/util/rst.py b/sphinx/util/rst.py
--- a/sphinx/util/rst.py
+++ b/sphinx/util/rst.py
@@ -103,6 +103,11 @@ def prepend_prolog(content: StringList, prolog: str) -> None:
 def append_epilog(content: StringList, epilog: str) -> None:
     """Append a string to content body as epilog."""
     if epilog:
-        content.append('', '<generated>', 0)
+        if 0 < len(content):
+            source, lineno = content.info(-1)
+        else:
+            source = '<generated>'
+            lineno = 0
+        content.append('', source, lineno + 1)
         for lineno, line in enumerate(epilog.splitlines()):
             content.append(line, '<rst_epilog>', lineno)

```

## Test Patch

```diff
diff --git a/tests/test_parser.py b/tests/test_parser.py
--- a/tests/test_parser.py
+++ b/tests/test_parser.py
@@ -50,7 +50,7 @@ def test_RSTParser_prolog_epilog(RSTStateMachine, app):
     (content, _), _ = RSTStateMachine().run.call_args
     assert list(content.xitems()) == [('dummy.rst', 0, 'hello Sphinx world'),
                                       ('dummy.rst', 1, 'Sphinx is a document generator'),
-                                      ('<generated>', 0, ''),
+                                      ('dummy.rst', 2, ''),
                                       ('<rst_epilog>', 0, 'this is rst_epilog'),
                                       ('<rst_epilog>', 1, 'good-bye reST!')]
 
diff --git a/tests/test_util_rst.py b/tests/test_util_rst.py
--- a/tests/test_util_rst.py
+++ b/tests/test_util_rst.py
@@ -32,7 +32,7 @@ def test_append_epilog(app):
 
     assert list(content.xitems()) == [('dummy.rst', 0, 'hello Sphinx world'),
                                       ('dummy.rst', 1, 'Sphinx is a document generator'),
-                                      ('<generated>', 0, ''),
+                                      ('dummy.rst', 2, ''),
                                       ('<rst_epilog>', 0, 'this is rst_epilog'),
                                       ('<rst_epilog>', 1, 'good-bye reST!')]
 

```


## Code snippets

### 1 - doc/conf.py:

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
### 2 - sphinx/transforms/i18n.py:

Start line: 222, End line: 377

```python
class Locale(SphinxTransform):

    def apply(self, **kwargs: Any) -> None:
        # ... other code
        for node, msg in extract_messages(self.document):
            if node.get('translated', False):  # to avoid double translation
                continue  # skip if the node is already translated by phase1

            msgstr = catalog.gettext(msg)
            # XXX add marker to untranslated parts
            if not msgstr or msgstr == msg:  # as-of-yet untranslated
                continue

            # update translatable nodes
            if isinstance(node, addnodes.translatable):
                node.apply_translated_message(msg, msgstr)
                continue

            # update meta nodes
            if isinstance(node, nodes.pending) and is_pending_meta(node):
                node.details['nodes'][0]['content'] = msgstr
                continue

            # Avoid "Literal block expected; none found." warnings.
            # If msgstr ends with '::' then it cause warning message at
            # parser.parse() processing.
            # literal-block-warning is only appear in avobe case.
            if msgstr.strip().endswith('::'):
                msgstr += '\n\n   dummy literal'
                # dummy literal node will discard by 'patch = patch[0]'

            # literalblock need literal block notation to avoid it become
            # paragraph.
            if isinstance(node, LITERAL_TYPE_NODES):
                msgstr = '::\n\n' + indent(msgstr, ' ' * 3)

            # Structural Subelements phase1
            # There is a possibility that only the title node is created.
            # see: http://docutils.sourceforge.net/docs/ref/doctree.html#structural-subelements
            if isinstance(node, nodes.title):
                # This generates: <section ...><title>msgstr</title></section>
                msgstr = msgstr + '\n' + '-' * len(msgstr) * 2

            patch = publish_msgstr(self.app, msgstr, source,
                                   node.line, self.config, settings)

            # Structural Subelements phase2
            if isinstance(node, nodes.title):
                # get <title> node that placed as a first child
                patch = patch.next_node()

            # ignore unexpected markups in translation message
            unexpected = (
                nodes.paragraph,    # expected form of translation
                nodes.title         # generated by above "Subelements phase2"
            )  # type: Tuple[Type[Element], ...]

            # following types are expected if
            # config.gettext_additional_targets is configured
            unexpected += LITERAL_TYPE_NODES
            unexpected += IMAGE_TYPE_NODES

            if not isinstance(patch, unexpected):
                continue  # skip

            # auto-numbered foot note reference should use original 'ids'.
            def list_replace_or_append(lst: List[N], old: N, new: N) -> None:
                if old in lst:
                    lst[lst.index(old)] = new
                else:
                    lst.append(new)

            is_autofootnote_ref = NodeMatcher(nodes.footnote_reference, auto=Any)
            old_foot_refs = node.traverse(is_autofootnote_ref)  # type: List[nodes.footnote_reference]  # NOQA
            new_foot_refs = patch.traverse(is_autofootnote_ref)  # type: List[nodes.footnote_reference]  # NOQA
            if len(old_foot_refs) != len(new_foot_refs):
                old_foot_ref_rawsources = [ref.rawsource for ref in old_foot_refs]
                new_foot_ref_rawsources = [ref.rawsource for ref in new_foot_refs]
                logger.warning(__('inconsistent footnote references in translated message.' +
                                  ' original: {0}, translated: {1}')
                               .format(old_foot_ref_rawsources, new_foot_ref_rawsources),
                               location=node)
            old_foot_namerefs = {}  # type: Dict[str, List[nodes.footnote_reference]]
            for r in old_foot_refs:
                old_foot_namerefs.setdefault(r.get('refname'), []).append(r)
            for newf in new_foot_refs:
                refname = newf.get('refname')
                refs = old_foot_namerefs.get(refname, [])
                if not refs:
                    continue

                oldf = refs.pop(0)
                newf['ids'] = oldf['ids']
                for id in newf['ids']:
                    self.document.ids[id] = newf

                if newf['auto'] == 1:
                    # autofootnote_refs
                    list_replace_or_append(self.document.autofootnote_refs, oldf, newf)
                else:
                    # symbol_footnote_refs
                    list_replace_or_append(self.document.symbol_footnote_refs, oldf, newf)

                if refname:
                    footnote_refs = self.document.footnote_refs.setdefault(refname, [])
                    list_replace_or_append(footnote_refs, oldf, newf)

                    refnames = self.document.refnames.setdefault(refname, [])
                    list_replace_or_append(refnames, oldf, newf)

            # reference should use new (translated) 'refname'.
            # * reference target ".. _Python: ..." is not translatable.
            # * use translated refname for section refname.
            # * inline reference "`Python <...>`_" has no 'refname'.
            is_refnamed_ref = NodeMatcher(nodes.reference, refname=Any)
            old_refs = node.traverse(is_refnamed_ref)  # type: List[nodes.reference]
            new_refs = patch.traverse(is_refnamed_ref)  # type: List[nodes.reference]
            if len(old_refs) != len(new_refs):
                old_ref_rawsources = [ref.rawsource for ref in old_refs]
                new_ref_rawsources = [ref.rawsource for ref in new_refs]
                logger.warning(__('inconsistent references in translated message.' +
                                  ' original: {0}, translated: {1}')
                               .format(old_ref_rawsources, new_ref_rawsources),
                               location=node)
            old_ref_names = [r['refname'] for r in old_refs]
            new_ref_names = [r['refname'] for r in new_refs]
            orphans = list(set(old_ref_names) - set(new_ref_names))
            for newr in new_refs:
                if not self.document.has_name(newr['refname']):
                    # Maybe refname is translated but target is not translated.
                    # Note: multiple translated refnames break link ordering.
                    if orphans:
                        newr['refname'] = orphans.pop(0)
                    else:
                        # orphan refnames is already empty!
                        # reference number is same in new_refs and old_refs.
                        pass

                self.document.note_refname(newr)

            # refnamed footnote should use original 'ids'.
            is_refnamed_footnote_ref = NodeMatcher(nodes.footnote_reference, refname=Any)
            old_foot_refs = node.traverse(is_refnamed_footnote_ref)
            new_foot_refs = patch.traverse(is_refnamed_footnote_ref)
            refname_ids_map = {}  # type: Dict[str, List[str]]
            if len(old_foot_refs) != len(new_foot_refs):
                old_foot_ref_rawsources = [ref.rawsource for ref in old_foot_refs]
                new_foot_ref_rawsources = [ref.rawsource for ref in new_foot_refs]
                logger.warning(__('inconsistent footnote references in translated message.' +
                                  ' original: {0}, translated: {1}')
                               .format(old_foot_ref_rawsources, new_foot_ref_rawsources),
                               location=node)
            for oldf in old_foot_refs:
                refname_ids_map.setdefault(oldf["refname"], []).append(oldf["ids"])
            for newf in new_foot_refs:
                refname = newf["refname"]
                if refname_ids_map.get(refname):
                    newf["ids"] = refname_ids_map[refname].pop(0)

            # citation should use original 'ids'.
  # ... other code
        # ... other code
```
### 3 - utils/doclinter.py:

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
### 4 - doc/conf.py:

Start line: 129, End line: 165

```python
def parse_event(env, sig, signode):
    m = event_sig_re.match(sig)
    if not m:
        signode += addnodes.desc_name(sig, sig)
        return sig
    name, args = m.groups()
    signode += addnodes.desc_name(name, name)
    plist = addnodes.desc_parameterlist()
    for arg in args.split(','):
        arg = arg.strip()
        plist += addnodes.desc_parameter(arg, arg)
    signode += plist
    return name


def setup(app):
    from sphinx.ext.autodoc import cut_lines
    from sphinx.util.docfields import GroupedField
    app.connect('autodoc-process-docstring', cut_lines(4, what=['module']))
    app.add_object_type('confval', 'confval',
                        objname='configuration value',
                        indextemplate='pair: %s; configuration value')
    app.add_object_type('setuptools-confval', 'setuptools-confval',
                        objname='setuptools configuration value',
                        indextemplate='pair: %s; setuptools configuration value')
    fdesc = GroupedField('parameter', label='Parameters',
                         names=['param'], can_collapse=True)
    app.add_object_type('event', 'event', 'pair: %s; event', parse_event,
                        doc_field_types=[fdesc])

    # workaround for RTD
    from sphinx.util import logging
    logger = logging.getLogger(__name__)
    app.info = lambda *args, **kwargs: logger.info(*args, **kwargs)
    app.warn = lambda *args, **kwargs: logger.warning(*args, **kwargs)
    app.debug = lambda *args, **kwargs: logger.debug(*args, **kwargs)
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
### 6 - sphinx/util/cfamily.py:

Start line: 11, End line: 80

```python
import re
import warnings
from copy import deepcopy
from typing import (
    Any, Callable, List, Match, Optional, Pattern, Tuple, Union
)

from docutils import nodes
from docutils.nodes import TextElement

from sphinx.config import Config
from sphinx.deprecation import RemovedInSphinx40Warning
from sphinx.util import logging

logger = logging.getLogger(__name__)

StringifyTransform = Callable[[Any], str]


_whitespace_re = re.compile(r'(?u)\s+')
anon_identifier_re = re.compile(r'(@[a-zA-Z0-9_])[a-zA-Z0-9_]*\b')
identifier_re = re.compile(r'''(?x)
    (   # This 'extends' _anon_identifier_re with the ordinary identifiers,
        # make sure they are in sync.
        (~?\b[a-zA-Z_])  # ordinary identifiers
    |   (@[a-zA-Z0-9_])  # our extension for names of anonymous entities
    )
    [a-zA-Z0-9_]*\b
''')
integer_literal_re = re.compile(r'[1-9][0-9]*')
octal_literal_re = re.compile(r'0[0-7]*')
hex_literal_re = re.compile(r'0[xX][0-9a-fA-F][0-9a-fA-F]*')
binary_literal_re = re.compile(r'0[bB][01][01]*')
integers_literal_suffix_re = re.compile(r'''(?x)
    # unsigned and/or (long) long, in any order, but at least one of them
    (
        ([uU]    ([lL]  |  (ll)  |  (LL))?)
        |
        (([lL]  |  (ll)  |  (LL))    [uU]?)
    )\b
    # the ending word boundary is important for distinguishing
    # between suffixes and UDLs in C++
''')
float_literal_re = re.compile(r'''(?x)
    [+-]?(
    # decimal
      ([0-9]+[eE][+-]?[0-9]+)
    | ([0-9]*\.[0-9]+([eE][+-]?[0-9]+)?)
    | ([0-9]+\.([eE][+-]?[0-9]+)?)
    # hex
    | (0[xX][0-9a-fA-F]+[pP][+-]?[0-9a-fA-F]+)
    | (0[xX][0-9a-fA-F]*\.[0-9a-fA-F]+([pP][+-]?[0-9a-fA-F]+)?)
    | (0[xX][0-9a-fA-F]+\.([pP][+-]?[0-9a-fA-F]+)?)
    )
''')
float_literal_suffix_re = re.compile(r'[fFlL]\b')
# the ending word boundary is important for distinguishing between suffixes and UDLs in C++
char_literal_re = re.compile(r'''(?x)
    ((?:u8)|u|U|L)?
    '(
      (?:[^\\'])
    | (\\(
        (?:['"?\\abfnrtv])
      | (?:[0-7]{1,3})
      | (?:x[0-9a-fA-F]{2})
      | (?:u[0-9a-fA-F]{4})
      | (?:U[0-9a-fA-F]{8})
      ))
    )'
''')
```
### 7 - sphinx/transforms/post_transforms/__init__.py:

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
### 8 - sphinx/builders/_epub_base.py:

Start line: 11, End line: 102

```python
import html
import os
import re
import warnings
from collections import namedtuple
from os import path
from typing import Any, Dict, List, Set, Tuple
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

from docutils import nodes
from docutils.nodes import Element, Node
from docutils.utils import smartquotes

from sphinx import addnodes
from sphinx.builders.html import BuildInfo, StandaloneHTMLBuilder
from sphinx.deprecation import RemovedInSphinx40Warning
from sphinx.locale import __
from sphinx.util import logging
from sphinx.util import status_iterator
from sphinx.util.fileutil import copy_asset_file
from sphinx.util.i18n import format_date
from sphinx.util.osutil import ensuredir, copyfile

try:
    from PIL import Image
except ImportError:
    Image = None


logger = logging.getLogger(__name__)


# (Fragment) templates from which the metainfo files content.opf and
# toc.ncx are created.
# This template section also defines strings that are embedded in the html
# output but that may be customized by (re-)setting module attributes,
# e.g. from conf.py.

COVERPAGE_NAME = 'epub-cover.xhtml'

TOCTREE_TEMPLATE = 'toctree-l%d'

LINK_TARGET_TEMPLATE = ' [%(uri)s]'

FOOTNOTE_LABEL_TEMPLATE = '#%d'

FOOTNOTES_RUBRIC_NAME = 'Footnotes'

CSS_LINK_TARGET_CLASS = 'link-target'

# XXX These strings should be localized according to epub_language
GUIDE_TITLES = {
    'toc': 'Table of Contents',
    'cover': 'Cover'
}

MEDIA_TYPES = {
    '.xhtml': 'application/xhtml+xml',
    '.css': 'text/css',
    '.png': 'image/png',
    '.gif': 'image/gif',
    '.svg': 'image/svg+xml',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.otf': 'application/x-font-otf',
    '.ttf': 'application/x-font-ttf',
    '.woff': 'application/font-woff',
}

VECTOR_GRAPHICS_EXTENSIONS = ('.svg',)

# Regular expression to match colons only in local fragment identifiers.
# If the URI contains a colon before the #,
# it is an external link that should not change.
REFURI_RE = re.compile("([^#:]*#)(.*)")


ManifestItem = namedtuple('ManifestItem', ['href', 'id', 'media_type'])
Spine = namedtuple('Spine', ['idref', 'linear'])
Guide = namedtuple('Guide', ['type', 'title', 'uri'])
NavPoint = namedtuple('NavPoint', ['navpoint', 'playorder', 'text', 'refuri', 'children'])


def sphinx_smarty_pants(t: str, language: str = 'en') -> str:
    t = t.replace('&quot;', '"')
    t = smartquotes.educateDashesOldSchool(t)
    t = smartquotes.educateQuotes(t, language)
    t = t.replace('"', '&quot;')
    return t


ssp = sphinx_smarty_pants
```
### 9 - sphinx/ext/autosummary/generate.py:

Start line: 580, End line: 619

```python
def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        usage='%(prog)s [OPTIONS] <SOURCE_FILE>...',
        epilog=__('For more information, visit <http://sphinx-doc.org/>.'),
        description=__("""
Generate ReStructuredText using autosummary directives.

sphinx-autogen is a frontend to sphinx.ext.autosummary.generate. It generates
the reStructuredText files from the autosummary directives contained in the
given input files.

The format of the autosummary directive is documented in the
``sphinx.ext.autosummary`` Python module and can be read using::

  pydoc sphinx.ext.autosummary
"""))

    parser.add_argument('--version', action='version', dest='show_version',
                        version='%%(prog)s %s' % __display_version__)

    parser.add_argument('source_file', nargs='+',
                        help=__('source files to generate rST files for'))

    parser.add_argument('-o', '--output-dir', action='store',
                        dest='output_dir',
                        help=__('directory to place all output in'))
    parser.add_argument('-s', '--suffix', action='store', dest='suffix',
                        default='rst',
                        help=__('default suffix for files (default: '
                                '%(default)s)'))
    parser.add_argument('-t', '--templates', action='store', dest='templates',
                        default=None,
                        help=__('custom template directory (default: '
                                '%(default)s)'))
    parser.add_argument('-i', '--imported-members', action='store_true',
                        dest='imported_members', default=False,
                        help=__('document imported members (default: '
                                '%(default)s)'))

    return parser
```
### 10 - doc/conf.py:

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
