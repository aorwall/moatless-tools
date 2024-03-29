# sphinx-doc__sphinx-7975

| **sphinx-doc/sphinx** | `4ec6cbe341fd84468c448e20082c778043bbea4b` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | 75485 |
| **Avg pos** | 44.0 |
| **Min pos** | 44 |
| **Max pos** | 44 |
| **Top file pos** | 2 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sphinx/environment/adapters/indexentries.py b/sphinx/environment/adapters/indexentries.py
--- a/sphinx/environment/adapters/indexentries.py
+++ b/sphinx/environment/adapters/indexentries.py
@@ -98,9 +98,8 @@ def keyfunc0(entry: Tuple[str, str]) -> Tuple[bool, str]:
             for subentry in indexentry[1].values():
                 subentry[0].sort(key=keyfunc0)  # type: ignore
 
-        # sort the index entries; put all symbols at the front, even those
-        # following the letters in ASCII, this is where the chr(127) comes from
-        def keyfunc(entry: Tuple[str, List]) -> Tuple[str, str]:
+        # sort the index entries
+        def keyfunc(entry: Tuple[str, List]) -> Tuple[Tuple[int, str], str]:
             key, (void, void, category_key) = entry
             if category_key:
                 # using specified category key to sort
@@ -108,11 +107,16 @@ def keyfunc(entry: Tuple[str, List]) -> Tuple[str, str]:
             lckey = unicodedata.normalize('NFD', key.lower())
             if lckey.startswith('\N{RIGHT-TO-LEFT MARK}'):
                 lckey = lckey[1:]
+
             if lckey[0:1].isalpha() or lckey.startswith('_'):
-                lckey = chr(127) + lckey
+                # put non-symbol characters at the folloing group (1)
+                sortkey = (1, lckey)
+            else:
+                # put symbols at the front of the index (0)
+                sortkey = (0, lckey)
             # ensure a determinstic order *within* letters by also sorting on
             # the entry itself
-            return (lckey, entry[0])
+            return (sortkey, entry[0])
         newlist = sorted(new.items(), key=keyfunc)
 
         if group_entries:

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sphinx/environment/adapters/indexentries.py | 101 | 103 | - | 2 | -
| sphinx/environment/adapters/indexentries.py | 111 | 114 | 44 | 2 | 75485


## Problem Statement

```
Two sections called Symbols in index
When using index entries with the following leading characters: _@_, _£_, and _←_ I get two sections called _Symbols_ in the HTML output, the first containing all _@_ entries before ”normal” words and the second containing _£_ and _←_ entries after the ”normal” words.  Both have the same anchor in HTML so the links at the top of the index page contain two _Symbols_ links, one before the letters and one after, but both lead to the first section.


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 sphinx/util/smartypants.py | 28 | 123| 1414 | 1414 | 4111 | 
| 2 | **2 sphinx/environment/adapters/indexentries.py** | 57 | 102| 583 | 1997 | 5771 | 
| 3 | 3 sphinx/util/nodes.py | 469 | 512| 689 | 2686 | 11197 | 
| 4 | 4 sphinx/writers/html.py | 697 | 796| 803 | 3489 | 18594 | 
| 5 | **4 sphinx/environment/adapters/indexentries.py** | 158 | 177| 264 | 3753 | 18594 | 
| 6 | 5 sphinx/domains/cpp.py | 3297 | 4086| 6565 | 10318 | 81160 | 
| 7 | 5 sphinx/domains/cpp.py | 4088 | 4851| 6558 | 16876 | 81160 | 
| 8 | 5 sphinx/domains/cpp.py | 11 | 697| 6100 | 22976 | 81160 | 
| 9 | 6 sphinx/domains/c.py | 807 | 1651| 6659 | 29635 | 111499 | 
| 10 | 7 sphinx/writers/text.py | 1045 | 1154| 793 | 30428 | 120455 | 
| 11 | 7 sphinx/domains/c.py | 11 | 804| 6434 | 36862 | 120455 | 
| 12 | 7 sphinx/domains/c.py | 2720 | 3500| 6534 | 43396 | 120455 | 
| 13 | 8 sphinx/util/cfamily.py | 11 | 80| 766 | 44162 | 123958 | 
| 14 | 9 sphinx/writers/texinfo.py | 1292 | 1375| 673 | 44835 | 136137 | 
| 15 | 9 sphinx/writers/texinfo.py | 968 | 1085| 846 | 45681 | 136137 | 
| 16 | 10 sphinx/writers/html5.py | 634 | 720| 686 | 46367 | 143071 | 
| 17 | 11 sphinx/search/zh.py | 36 | 220| 1482 | 47849 | 145103 | 
| 18 | 11 sphinx/writers/texinfo.py | 11 | 112| 595 | 48444 | 145103 | 
| 19 | 12 sphinx/writers/manpage.py | 313 | 411| 757 | 49201 | 148614 | 
| 20 | 12 sphinx/writers/texinfo.py | 862 | 966| 788 | 49989 | 148614 | 
| 21 | 12 sphinx/util/nodes.py | 370 | 405| 310 | 50299 | 148614 | 
| 22 | 12 sphinx/writers/html.py | 171 | 231| 562 | 50861 | 148614 | 
| 23 | 13 sphinx/addnodes.py | 281 | 380| 580 | 51441 | 151359 | 
| 24 | 13 sphinx/domains/c.py | 1653 | 2490| 6823 | 58264 | 151359 | 
| 25 | 13 sphinx/writers/texinfo.py | 1087 | 1186| 848 | 59112 | 151359 | 
| 26 | **13 sphinx/environment/adapters/indexentries.py** | 116 | 157| 417 | 59529 | 151359 | 
| 27 | 13 sphinx/writers/html.py | 354 | 409| 472 | 60001 | 151359 | 
| 28 | 13 sphinx/writers/html.py | 260 | 282| 214 | 60215 | 151359 | 
| 29 | 14 sphinx/search/ja.py | 400 | 423| 442 | 60657 | 163999 | 
| 30 | 14 sphinx/writers/html.py | 798 | 845| 462 | 61119 | 163999 | 
| 31 | 14 sphinx/writers/html.py | 232 | 258| 318 | 61437 | 163999 | 
| 32 | 14 sphinx/search/ja.py | 271 | 291| 631 | 62068 | 163999 | 
| 33 | 15 sphinx/writers/latex.py | 1456 | 1520| 863 | 62931 | 183906 | 
| 34 | 15 sphinx/domains/cpp.py | 699 | 1497| 6638 | 69569 | 183906 | 
| 35 | 15 sphinx/search/ja.py | 251 | 270| 716 | 70285 | 183906 | 
| 36 | 16 sphinx/builders/_epub_base.py | 11 | 102| 662 | 70947 | 190627 | 
| 37 | 16 sphinx/writers/html5.py | 232 | 253| 208 | 71155 | 190627 | 
| 38 | 17 sphinx/util/texescape.py | 11 | 64| 623 | 71778 | 192396 | 
| 39 | 18 doc/development/tutorials/examples/todo.py | 30 | 53| 175 | 71953 | 193304 | 
| 40 | 18 sphinx/writers/html5.py | 143 | 203| 567 | 72520 | 193304 | 
| 41 | 19 sphinx/search/en.py | 11 | 226| 1665 | 74185 | 195025 | 
| 42 | 19 sphinx/writers/text.py | 930 | 1043| 873 | 75058 | 195025 | 
| 43 | 20 sphinx/search/__init__.py | 160 | 189| 194 | 75252 | 199067 | 
| **-> 44 <-** | **20 sphinx/environment/adapters/indexentries.py** | 103 | 115| 233 | 75485 | 199067 | 


## Patch

```diff
diff --git a/sphinx/environment/adapters/indexentries.py b/sphinx/environment/adapters/indexentries.py
--- a/sphinx/environment/adapters/indexentries.py
+++ b/sphinx/environment/adapters/indexentries.py
@@ -98,9 +98,8 @@ def keyfunc0(entry: Tuple[str, str]) -> Tuple[bool, str]:
             for subentry in indexentry[1].values():
                 subentry[0].sort(key=keyfunc0)  # type: ignore
 
-        # sort the index entries; put all symbols at the front, even those
-        # following the letters in ASCII, this is where the chr(127) comes from
-        def keyfunc(entry: Tuple[str, List]) -> Tuple[str, str]:
+        # sort the index entries
+        def keyfunc(entry: Tuple[str, List]) -> Tuple[Tuple[int, str], str]:
             key, (void, void, category_key) = entry
             if category_key:
                 # using specified category key to sort
@@ -108,11 +107,16 @@ def keyfunc(entry: Tuple[str, List]) -> Tuple[str, str]:
             lckey = unicodedata.normalize('NFD', key.lower())
             if lckey.startswith('\N{RIGHT-TO-LEFT MARK}'):
                 lckey = lckey[1:]
+
             if lckey[0:1].isalpha() or lckey.startswith('_'):
-                lckey = chr(127) + lckey
+                # put non-symbol characters at the folloing group (1)
+                sortkey = (1, lckey)
+            else:
+                # put symbols at the front of the index (0)
+                sortkey = (0, lckey)
             # ensure a determinstic order *within* letters by also sorting on
             # the entry itself
-            return (lckey, entry[0])
+            return (sortkey, entry[0])
         newlist = sorted(new.items(), key=keyfunc)
 
         if group_entries:

```

## Test Patch

```diff
diff --git a/tests/test_environment_indexentries.py b/tests/test_environment_indexentries.py
--- a/tests/test_environment_indexentries.py
+++ b/tests/test_environment_indexentries.py
@@ -25,12 +25,14 @@ def test_create_single_index(app):
             ".. index:: ёлка\n"
             ".. index:: ‏תירבע‎\n"
             ".. index:: 9-symbol\n"
-            ".. index:: &-symbol\n")
+            ".. index:: &-symbol\n"
+            ".. index:: £100\n")
     restructuredtext.parse(app, text)
     index = IndexEntries(app.env).create_index(app.builder)
     assert len(index) == 6
     assert index[0] == ('Symbols', [('&-symbol', [[('', '#index-9')], [], None]),
-                                    ('9-symbol', [[('', '#index-8')], [], None])])
+                                    ('9-symbol', [[('', '#index-8')], [], None]),
+                                    ('£100', [[('', '#index-10')], [], None])])
     assert index[1] == ('D', [('docutils', [[('', '#index-0')], [], None])])
     assert index[2] == ('P', [('pip', [[], [('install', [('', '#index-2')]),
                                             ('upgrade', [('', '#index-3')])], None]),

```


## Code snippets

### 1 - sphinx/util/smartypants.py:

Start line: 28, End line: 123

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
### 2 - sphinx/environment/adapters/indexentries.py:

Start line: 57, End line: 102

```python
class IndexEntries:

    def create_index(self, builder: Builder, group_entries: bool = True,
                     _fixre: Pattern = re.compile(r'(.*) ([(][^()]*[)])')
                     ) -> List[Tuple[str, List[Tuple[str, Any]]]]:
        # ... other code

        domain = cast(IndexDomain, self.env.get_domain('index'))
        for fn, entries in domain.entries.items():
            # new entry types must be listed in directives/other.py!
            for type, value, tid, main, index_key in entries:
                try:
                    if type == 'single':
                        try:
                            entry, subentry = split_into(2, 'single', value)
                        except ValueError:
                            entry, = split_into(1, 'single', value)
                            subentry = ''
                        add_entry(entry, subentry, main, key=index_key)
                    elif type == 'pair':
                        first, second = split_into(2, 'pair', value)
                        add_entry(first, second, main, key=index_key)
                        add_entry(second, first, main, key=index_key)
                    elif type == 'triple':
                        first, second, third = split_into(3, 'triple', value)
                        add_entry(first, second + ' ' + third, main, key=index_key)
                        add_entry(second, third + ', ' + first, main, key=index_key)
                        add_entry(third, first + ' ' + second, main, key=index_key)
                    elif type == 'see':
                        first, second = split_into(2, 'see', value)
                        add_entry(first, _('see %s') % second, None,
                                  link=False, key=index_key)
                    elif type == 'seealso':
                        first, second = split_into(2, 'see', value)
                        add_entry(first, _('see also %s') % second, None,
                                  link=False, key=index_key)
                    else:
                        logger.warning(__('unknown index entry type %r'), type, location=fn)
                except ValueError as err:
                    logger.warning(str(err), location=fn)

        # sort the index entries for same keyword.
        def keyfunc0(entry: Tuple[str, str]) -> Tuple[bool, str]:
            main, uri = entry
            return (not main, uri)  # show main entries at first

        for indexentry in new.values():
            indexentry[0].sort(key=keyfunc0)
            for subentry in indexentry[1].values():
                subentry[0].sort(key=keyfunc0)  # type: ignore

        # sort the index entries; put all symbols at the front, even those
        # following the letters in ASCII, this is where the chr(127) comes from
        # ... other code
```
### 3 - sphinx/util/nodes.py:

Start line: 469, End line: 512

```python
_non_id_chars = re.compile('[^a-zA-Z0-9._]+')
_non_id_at_ends = re.compile('^[-0-9._]+|-+$')
_non_id_translate = {
    0x00f8: u'o',       # o with stroke
    0x0111: u'd',       # d with stroke
    0x0127: u'h',       # h with stroke
    0x0131: u'i',       # dotless i
    0x0142: u'l',       # l with stroke
    0x0167: u't',       # t with stroke
    0x0180: u'b',       # b with stroke
    0x0183: u'b',       # b with topbar
    0x0188: u'c',       # c with hook
    0x018c: u'd',       # d with topbar
    0x0192: u'f',       # f with hook
    0x0199: u'k',       # k with hook
    0x019a: u'l',       # l with bar
    0x019e: u'n',       # n with long right leg
    0x01a5: u'p',       # p with hook
    0x01ab: u't',       # t with palatal hook
    0x01ad: u't',       # t with hook
    0x01b4: u'y',       # y with hook
    0x01b6: u'z',       # z with stroke
    0x01e5: u'g',       # g with stroke
    0x0225: u'z',       # z with hook
    0x0234: u'l',       # l with curl
    0x0235: u'n',       # n with curl
    0x0236: u't',       # t with curl
    0x0237: u'j',       # dotless j
    0x023c: u'c',       # c with stroke
    0x023f: u's',       # s with swash tail
    0x0240: u'z',       # z with swash tail
    0x0247: u'e',       # e with stroke
    0x0249: u'j',       # j with stroke
    0x024b: u'q',       # q with hook tail
    0x024d: u'r',       # r with stroke
    0x024f: u'y',       # y with stroke
}
_non_id_translate_digraphs = {
    0x00df: u'sz',      # ligature sz
    0x00e6: u'ae',      # ae
    0x0153: u'oe',      # ligature oe
    0x0238: u'db',      # db digraph
    0x0239: u'qp',      # qp digraph
}
```
### 4 - sphinx/writers/html.py:

Start line: 697, End line: 796

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
### 5 - sphinx/environment/adapters/indexentries.py:

Start line: 158, End line: 177

```python
class IndexEntries:

    def create_index(self, builder: Builder, group_entries: bool = True,
                     _fixre: Pattern = re.compile(r'(.*) ([(][^()]*[)])')
                     ) -> List[Tuple[str, List[Tuple[str, Any]]]]:
        # ... other code
        def keyfunc3(item: Tuple[str, List]) -> str:
            # hack: mutating the subitems dicts to a list in the keyfunc
            k, v = item
            v[1] = sorted(((si, se) for (si, (se, void, void)) in v[1].items()),
                          key=keyfunc2)
            if v[2] is None:
                # now calculate the key
                if k.startswith('\N{RIGHT-TO-LEFT MARK}'):
                    k = k[1:]
                letter = unicodedata.normalize('NFD', k[0])[0].upper()
                if letter.isalpha() or letter == '_':
                    return letter
                else:
                    # get all other symbols under one heading
                    return _('Symbols')
            else:
                return v[2]
        return [(key_, list(group))
                for (key_, group) in groupby(newlist, keyfunc3)]
```
### 6 - sphinx/domains/cpp.py:

Start line: 3297, End line: 4086

```python
class ASTTemplateParamTemplateType(ASTTemplateParam):
    def __init__(self, nestedParams: "ASTTemplateParams",
                 data: ASTTemplateKeyParamPackIdDefault) -> None:
        assert nestedParams
        assert data
        self.nestedParams = nestedParams
        self.data = data

    @property
    def name(self) -> ASTNestedName:
        id = self.get_identifier()
        return ASTNestedName([ASTNestedNameElement(id, None)], [False], rooted=False)

    @property
    def isPack(self) -> bool:
        return self.data.parameterPack

    def get_identifier(self) -> ASTIdentifier:
        return self.data.get_identifier()

    def get_id(self, version: int, objectType: str = None, symbol: "Symbol" = None) -> str:
        assert version >= 2
        # this is not part of the normal name mangling in C++
        if symbol:
            # the anchor will be our parent
            return symbol.parent.declaration.get_id(version, prefixed=None)
        else:
            return self.nestedParams.get_id(version) + self.data.get_id(version)

    def _stringify(self, transform: StringifyTransform) -> str:
        return transform(self.nestedParams) + transform(self.data)

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        self.nestedParams.describe_signature(signode, 'noneIsName', env, symbol)
        signode += nodes.Text(' ')
        self.data.describe_signature(signode, mode, env, symbol)


class ASTTemplateParamNonType(ASTTemplateParam):
    def __init__(self,
                 param: Union[ASTTypeWithInit,
                              ASTTemplateParamConstrainedTypeWithInit]) -> None:
        assert param
        self.param = param

    @property
    def name(self) -> ASTNestedName:
        id = self.get_identifier()
        return ASTNestedName([ASTNestedNameElement(id, None)], [False], rooted=False)

    @property
    def isPack(self) -> bool:
        return self.param.isPack

    def get_identifier(self) -> ASTIdentifier:
        name = self.param.name
        if name:
            assert len(name.names) == 1
            assert name.names[0].identOrOp
            assert not name.names[0].templateArgs
            res = name.names[0].identOrOp
            assert isinstance(res, ASTIdentifier)
            return res
        else:
            return None

    def get_id(self, version: int, objectType: str = None, symbol: "Symbol" = None) -> str:
        assert version >= 2
        # this is not part of the normal name mangling in C++
        if symbol:
            # the anchor will be our parent
            return symbol.parent.declaration.get_id(version, prefixed=None)
        else:
            return '_' + self.param.get_id(version)

    def _stringify(self, transform: StringifyTransform) -> str:
        return transform(self.param)

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        self.param.describe_signature(signode, mode, env, symbol)


class ASTTemplateParams(ASTBase):
    def __init__(self, params: List[ASTTemplateParam]) -> None:
        assert params is not None
        self.params = params

    def get_id(self, version: int) -> str:
        assert version >= 2
        res = []
        res.append("I")
        for param in self.params:
            res.append(param.get_id(version))
        res.append("E")
        return ''.join(res)

    def _stringify(self, transform: StringifyTransform) -> str:
        res = []
        res.append("template<")
        res.append(", ".join(transform(a) for a in self.params))
        res.append("> ")
        return ''.join(res)

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        signode += nodes.Text("template<")
        first = True
        for param in self.params:
            if not first:
                signode += nodes.Text(", ")
            first = False
            param.describe_signature(signode, mode, env, symbol)
        signode += nodes.Text(">")

    def describe_signature_as_introducer(
            self, parentNode: desc_signature, mode: str, env: "BuildEnvironment",
            symbol: "Symbol", lineSpec: bool) -> None:
        def makeLine(parentNode: desc_signature) -> addnodes.desc_signature_line:
            signode = addnodes.desc_signature_line()
            parentNode += signode
            signode.sphinx_line_type = 'templateParams'
            return signode
        lineNode = makeLine(parentNode)
        lineNode += nodes.Text("template<")
        first = True
        for param in self.params:
            if not first:
                lineNode += nodes.Text(", ")
            first = False
            if lineSpec:
                lineNode = makeLine(parentNode)
            param.describe_signature(lineNode, mode, env, symbol)
        if lineSpec and not first:
            lineNode = makeLine(parentNode)
        lineNode += nodes.Text(">")


# Template introducers
################################################################################

class ASTTemplateIntroductionParameter(ASTBase):
    def __init__(self, identifier: ASTIdentifier, parameterPack: bool) -> None:
        self.identifier = identifier
        self.parameterPack = parameterPack

    @property
    def name(self) -> ASTNestedName:
        id = self.get_identifier()
        return ASTNestedName([ASTNestedNameElement(id, None)], [False], rooted=False)

    @property
    def isPack(self) -> bool:
        return self.parameterPack

    def get_identifier(self) -> ASTIdentifier:
        return self.identifier

    def get_id(self, version: int, objectType: str = None, symbol: "Symbol" = None) -> str:
        assert version >= 2
        # this is not part of the normal name mangling in C++
        if symbol:
            # the anchor will be our parent
            return symbol.parent.declaration.get_id(version, prefixed=None)
        else:
            if self.parameterPack:
                return 'Dp'
            else:
                return '0'  # we need to put something

    def get_id_as_arg(self, version: int) -> str:
        assert version >= 2
        # used for the implicit requires clause
        res = self.identifier.get_id(version)
        if self.parameterPack:
            return 'sp' + res
        else:
            return res

    def _stringify(self, transform: StringifyTransform) -> str:
        res = []
        if self.parameterPack:
            res.append('...')
        res.append(transform(self.identifier))
        return ''.join(res)

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        if self.parameterPack:
            signode += nodes.Text('...')
        self.identifier.describe_signature(signode, mode, env, '', '', symbol)


class ASTTemplateIntroduction(ASTBase):
    def __init__(self, concept: ASTNestedName,
                 params: List[ASTTemplateIntroductionParameter]) -> None:
        assert len(params) > 0
        self.concept = concept
        self.params = params

    def get_id(self, version: int) -> str:
        assert version >= 2
        # first do the same as a normal template parameter list
        res = []
        res.append("I")
        for param in self.params:
            res.append(param.get_id(version))
        res.append("E")
        # let's use X expr E, which is otherwise for constant template args
        res.append("X")
        res.append(self.concept.get_id(version))
        res.append("I")
        for param in self.params:
            res.append(param.get_id_as_arg(version))
        res.append("E")
        res.append("E")
        return ''.join(res)

    def _stringify(self, transform: StringifyTransform) -> str:
        res = []
        res.append(transform(self.concept))
        res.append('{')
        res.append(', '.join(transform(param) for param in self.params))
        res.append('} ')
        return ''.join(res)

    def describe_signature_as_introducer(
            self, parentNode: desc_signature, mode: str,
            env: "BuildEnvironment", symbol: "Symbol", lineSpec: bool) -> None:
        # Note: 'lineSpec' has no effect on template introductions.
        signode = addnodes.desc_signature_line()
        parentNode += signode
        signode.sphinx_line_type = 'templateIntroduction'
        self.concept.describe_signature(signode, 'markType', env, symbol)
        signode += nodes.Text('{')
        first = True
        for param in self.params:
            if not first:
                signode += nodes.Text(', ')
            first = False
            param.describe_signature(signode, mode, env, symbol)
        signode += nodes.Text('}')


################################################################################

class ASTTemplateDeclarationPrefix(ASTBase):
    def __init__(self,
                 templates: List[Union[ASTTemplateParams,
                                       ASTTemplateIntroduction]]) -> None:
        # templates is None means it's an explicit instantiation of a variable
        self.templates = templates

    def get_id(self, version: int) -> str:
        assert version >= 2
        # this is not part of a normal name mangling system
        res = []
        for t in self.templates:
            res.append(t.get_id(version))
        return ''.join(res)

    def _stringify(self, transform: StringifyTransform) -> str:
        res = []
        for t in self.templates:
            res.append(transform(t))
        return ''.join(res)

    def describe_signature(self, signode: desc_signature, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol", lineSpec: bool) -> None:
        verify_description_mode(mode)
        for t in self.templates:
            t.describe_signature_as_introducer(signode, 'lastIsName', env, symbol, lineSpec)


class ASTRequiresClause(ASTBase):
    def __init__(self, expr: ASTExpression) -> None:
        self.expr = expr

    def _stringify(self, transform: StringifyTransform) -> str:
        return 'requires ' + transform(self.expr)

    def describe_signature(self, signode: addnodes.desc_signature_line, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        signode += nodes.Text('requires ', 'requires ')
        self.expr.describe_signature(signode, mode, env, symbol)


################################################################################
################################################################################

class ASTDeclaration(ASTBase):
    def __init__(self, objectType: str, directiveType: str, visibility: str,
                 templatePrefix: ASTTemplateDeclarationPrefix,
                 requiresClause: ASTRequiresClause, declaration: Any,
                 trailingRequiresClause: ASTRequiresClause,
                 semicolon: bool = False) -> None:
        self.objectType = objectType
        self.directiveType = directiveType
        self.visibility = visibility
        self.templatePrefix = templatePrefix
        self.requiresClause = requiresClause
        self.declaration = declaration
        self.trailingRequiresClause = trailingRequiresClause
        self.semicolon = semicolon

        self.symbol = None  # type: Symbol
        # set by CPPObject._add_enumerator_to_parent
        self.enumeratorScopedSymbol = None  # type: Symbol

    def clone(self) -> "ASTDeclaration":
        templatePrefixClone = self.templatePrefix.clone() if self.templatePrefix else None
        requiresClasueClone = self.requiresClause.clone() if self.requiresClause else None
        trailingRequiresClasueClone = self.trailingRequiresClause.clone() \
            if self.trailingRequiresClause else None
        return ASTDeclaration(self.objectType, self.directiveType, self.visibility,
                              templatePrefixClone, requiresClasueClone,
                              self.declaration.clone(), trailingRequiresClasueClone,
                              self.semicolon)

    @property
    def name(self) -> ASTNestedName:
        return self.declaration.name

    @property
    def function_params(self) -> List[ASTFunctionParameter]:
        if self.objectType != 'function':
            return None
        return self.declaration.function_params

    def get_id(self, version: int, prefixed: bool = True) -> str:
        if version == 1:
            if self.templatePrefix:
                raise NoOldIdError()
            if self.objectType == 'enumerator' and self.enumeratorScopedSymbol:
                return self.enumeratorScopedSymbol.declaration.get_id(version)
            return self.declaration.get_id(version, self.objectType, self.symbol)
        # version >= 2
        if self.objectType == 'enumerator' and self.enumeratorScopedSymbol:
            return self.enumeratorScopedSymbol.declaration.get_id(version, prefixed)
        if prefixed:
            res = [_id_prefix[version]]
        else:
            res = []
        if self.templatePrefix:
            res.append(self.templatePrefix.get_id(version))
        if self.requiresClause or self.trailingRequiresClause:
            if version < 4:
                raise NoOldIdError()
            res.append('IQ')
            if self.requiresClause and self.trailingRequiresClause:
                res.append('aa')
            if self.requiresClause:
                res.append(self.requiresClause.expr.get_id(version))
            if self.trailingRequiresClause:
                res.append(self.trailingRequiresClause.expr.get_id(version))
            res.append('E')
        res.append(self.declaration.get_id(version, self.objectType, self.symbol))
        return ''.join(res)

    def get_newest_id(self) -> str:
        return self.get_id(_max_id, True)

    def _stringify(self, transform: StringifyTransform) -> str:
        res = []
        if self.visibility and self.visibility != "public":
            res.append(self.visibility)
            res.append(' ')
        if self.templatePrefix:
            res.append(transform(self.templatePrefix))
        if self.requiresClause:
            res.append(transform(self.requiresClause))
            res.append(' ')
        res.append(transform(self.declaration))
        if self.trailingRequiresClause:
            res.append(' ')
            res.append(transform(self.trailingRequiresClause))
        if self.semicolon:
            res.append(';')
        return ''.join(res)

    def describe_signature(self, signode: desc_signature, mode: str,
                           env: "BuildEnvironment", options: Dict) -> None:
        verify_description_mode(mode)
        assert self.symbol
        # The caller of the domain added a desc_signature node.
        # Always enable multiline:
        signode['is_multiline'] = True
        # Put each line in a desc_signature_line node.
        mainDeclNode = addnodes.desc_signature_line()
        mainDeclNode.sphinx_line_type = 'declarator'
        mainDeclNode['add_permalink'] = not self.symbol.isRedeclaration

        if self.templatePrefix:
            self.templatePrefix.describe_signature(signode, mode, env,
                                                   symbol=self.symbol,
                                                   lineSpec=options.get('tparam-line-spec'))
        if self.requiresClause:
            reqNode = addnodes.desc_signature_line()
            reqNode.sphinx_line_type = 'requiresClause'
            signode.append(reqNode)
            self.requiresClause.describe_signature(reqNode, 'markType', env, self.symbol)
        signode += mainDeclNode
        if self.visibility and self.visibility != "public":
            mainDeclNode += addnodes.desc_annotation(self.visibility + " ",
                                                     self.visibility + " ")
        if self.objectType == 'type':
            prefix = self.declaration.get_type_declaration_prefix()
            prefix += ' '
            mainDeclNode += addnodes.desc_annotation(prefix, prefix)
        elif self.objectType == 'concept':
            mainDeclNode += addnodes.desc_annotation('concept ', 'concept ')
        elif self.objectType == 'member':
            pass
        elif self.objectType == 'function':
            pass
        elif self.objectType == 'class':
            assert self.directiveType in ('class', 'struct')
            prefix = self.directiveType + ' '
            mainDeclNode += addnodes.desc_annotation(prefix, prefix)
        elif self.objectType == 'union':
            mainDeclNode += addnodes.desc_annotation('union ', 'union ')
        elif self.objectType == 'enum':
            if self.directiveType == 'enum':
                prefix = 'enum '
            elif self.directiveType == 'enum-class':
                prefix = 'enum class '
            elif self.directiveType == 'enum-struct':
                prefix = 'enum struct '
            else:
                assert False  # wrong directiveType used
            mainDeclNode += addnodes.desc_annotation(prefix, prefix)
        elif self.objectType == 'enumerator':
            mainDeclNode += addnodes.desc_annotation('enumerator ', 'enumerator ')
        else:
            assert False
        self.declaration.describe_signature(mainDeclNode, mode, env, self.symbol)
        lastDeclNode = mainDeclNode
        if self.trailingRequiresClause:
            trailingReqNode = addnodes.desc_signature_line()
            trailingReqNode.sphinx_line_type = 'trailingRequiresClause'
            signode.append(trailingReqNode)
            lastDeclNode = trailingReqNode
            self.trailingRequiresClause.describe_signature(
                trailingReqNode, 'markType', env, self.symbol)
        if self.semicolon:
            lastDeclNode += nodes.Text(';')


class ASTNamespace(ASTBase):
    def __init__(self, nestedName: ASTNestedName,
                 templatePrefix: ASTTemplateDeclarationPrefix) -> None:
        self.nestedName = nestedName
        self.templatePrefix = templatePrefix

    def _stringify(self, transform: StringifyTransform) -> str:
        res = []
        if self.templatePrefix:
            res.append(transform(self.templatePrefix))
        res.append(transform(self.nestedName))
        return ''.join(res)


class SymbolLookupResult:
    def __init__(self, symbols: Iterator["Symbol"], parentSymbol: "Symbol",
                 identOrOp: Union[ASTIdentifier, ASTOperator], templateParams: Any,
                 templateArgs: ASTTemplateArgs) -> None:
        self.symbols = symbols
        self.parentSymbol = parentSymbol
        self.identOrOp = identOrOp
        self.templateParams = templateParams
        self.templateArgs = templateArgs


class LookupKey:
    def __init__(self, data: List[Tuple[ASTNestedNameElement,
                                        Union[ASTTemplateParams,
                                              ASTTemplateIntroduction],
                                        str]]) -> None:
        self.data = data


class Symbol:
    debug_indent = 0
    debug_indent_string = "  "
    debug_lookup = False  # overridden by the corresponding config value
    debug_show_tree = False  # overridden by the corresponding config value

    def __copy__(self):
        assert False  # shouldn't happen

    def __deepcopy__(self, memo):
        if self.parent:
            assert False  # shouldn't happen
        else:
            # the domain base class makes a copy of the initial data, which is fine
            return Symbol(None, None, None, None, None, None)

    @staticmethod
    def debug_print(*args: Any) -> None:
        print(Symbol.debug_indent_string * Symbol.debug_indent, end="")
        print(*args)

    def _assert_invariants(self) -> None:
        if not self.parent:
            # parent == None means global scope, so declaration means a parent
            assert not self.identOrOp
            assert not self.templateParams
            assert not self.templateArgs
            assert not self.declaration
            assert not self.docname
        else:
            if self.declaration:
                assert self.docname

    def __setattr__(self, key: str, value: Any) -> None:
        if key == "children":
            assert False
        else:
            return super().__setattr__(key, value)

    def __init__(self, parent: "Symbol", identOrOp: Union[ASTIdentifier, ASTOperator],
                 templateParams: Union[ASTTemplateParams, ASTTemplateIntroduction],
                 templateArgs: Any, declaration: ASTDeclaration, docname: str) -> None:
        self.parent = parent
        # declarations in a single directive are linked together
        self.siblingAbove = None  # type: Symbol
        self.siblingBelow = None  # type: Symbol
        self.identOrOp = identOrOp
        self.templateParams = templateParams  # template<templateParams>
        self.templateArgs = templateArgs  # identifier<templateArgs>
        self.declaration = declaration
        self.docname = docname
        self.isRedeclaration = False
        self._assert_invariants()

        # Remember to modify Symbol.remove if modifications to the parent change.
        self._children = []  # type: List[Symbol]
        self._anonChildren = []  # type: List[Symbol]
        # note: _children includes _anonChildren
        if self.parent:
            self.parent._children.append(self)
        if self.declaration:
            self.declaration.symbol = self

        # Do symbol addition after self._children has been initialised.
        self._add_template_and_function_params()

    def _fill_empty(self, declaration: ASTDeclaration, docname: str) -> None:
        self._assert_invariants()
        assert not self.declaration
        assert not self.docname
        assert declaration
        assert docname
        self.declaration = declaration
        self.declaration.symbol = self
        self.docname = docname
        self._assert_invariants()
        # and symbol addition should be done as well
        self._add_template_and_function_params()

    def _add_template_and_function_params(self) -> None:
        if Symbol.debug_lookup:
            Symbol.debug_indent += 1
            Symbol.debug_print("_add_template_and_function_params:")
        # Note: we may be called from _fill_empty, so the symbols we want
        #       to add may actually already be present (as empty symbols).

        # add symbols for the template params
        if self.templateParams:
            for tp in self.templateParams.params:
                if not tp.get_identifier():
                    continue
                # only add a declaration if we our self are from a declaration
                if self.declaration:
                    decl = ASTDeclaration('templateParam', None, None, None, None, tp, None)
                else:
                    decl = None
                nne = ASTNestedNameElement(tp.get_identifier(), None)
                nn = ASTNestedName([nne], [False], rooted=False)
                self._add_symbols(nn, [], decl, self.docname)
        # add symbols for function parameters, if any
        if self.declaration is not None and self.declaration.function_params is not None:
            for fp in self.declaration.function_params:
                if fp.arg is None:
                    continue
                nn = fp.arg.name
                if nn is None:
                    continue
                # (comparing to the template params: we have checked that we are a declaration)
                decl = ASTDeclaration('functionParam', None, None, None, None, fp, None)
                assert not nn.rooted
                assert len(nn.names) == 1
                self._add_symbols(nn, [], decl, self.docname)
        if Symbol.debug_lookup:
            Symbol.debug_indent -= 1

    def remove(self) -> None:
        if self.parent is None:
            return
        assert self in self.parent._children
        self.parent._children.remove(self)
        self.parent = None

    def clear_doc(self, docname: str) -> None:
        newChildren = []  # type: List[Symbol]
        for sChild in self._children:
            sChild.clear_doc(docname)
            if sChild.declaration and sChild.docname == docname:
                sChild.declaration = None
                sChild.docname = None
                if sChild.siblingAbove is not None:
                    sChild.siblingAbove.siblingBelow = sChild.siblingBelow
                if sChild.siblingBelow is not None:
                    sChild.siblingBelow.siblingAbove = sChild.siblingAbove
                sChild.siblingAbove = None
                sChild.siblingBelow = None
            newChildren.append(sChild)
        self._children = newChildren

    def get_all_symbols(self) -> Iterator[Any]:
        yield self
        for sChild in self._children:
            for s in sChild.get_all_symbols():
                yield s

    @property
    def children_recurse_anon(self) -> Generator["Symbol", None, None]:
        for c in self._children:
            yield c
            if not c.identOrOp.is_anon():
                continue

            yield from c.children_recurse_anon

    def get_lookup_key(self) -> "LookupKey":
        # The pickle files for the environment and for each document are distinct.
        # The environment has all the symbols, but the documents has xrefs that
        # must know their scope. A lookup key is essentially a specification of
        # how to find a specific symbol.
        symbols = []
        s = self
        while s.parent:
            symbols.append(s)
            s = s.parent
        symbols.reverse()
        key = []
        for s in symbols:
            nne = ASTNestedNameElement(s.identOrOp, s.templateArgs)
            if s.declaration is not None:
                key.append((nne, s.templateParams, s.declaration.get_newest_id()))
            else:
                key.append((nne, s.templateParams, None))
        return LookupKey(key)

    def get_full_nested_name(self) -> ASTNestedName:
        symbols = []
        s = self
        while s.parent:
            symbols.append(s)
            s = s.parent
        symbols.reverse()
        names = []
        templates = []
        for s in symbols:
            names.append(ASTNestedNameElement(s.identOrOp, s.templateArgs))
            templates.append(False)
        return ASTNestedName(names, templates, rooted=False)

    def _find_first_named_symbol(self, identOrOp: Union[ASTIdentifier, ASTOperator],
                                 templateParams: Any, templateArgs: ASTTemplateArgs,
                                 templateShorthand: bool, matchSelf: bool,
                                 recurseInAnon: bool, correctPrimaryTemplateArgs: bool
                                 ) -> "Symbol":
        if Symbol.debug_lookup:
            Symbol.debug_print("_find_first_named_symbol ->")
        res = self._find_named_symbols(identOrOp, templateParams, templateArgs,
                                       templateShorthand, matchSelf, recurseInAnon,
                                       correctPrimaryTemplateArgs,
                                       searchInSiblings=False)
        try:
            return next(res)
        except StopIteration:
            return None

    def _find_named_symbols(self, identOrOp: Union[ASTIdentifier, ASTOperator],
                            templateParams: Any, templateArgs: ASTTemplateArgs,
                            templateShorthand: bool, matchSelf: bool,
                            recurseInAnon: bool, correctPrimaryTemplateArgs: bool,
                            searchInSiblings: bool) -> Iterator["Symbol"]:
        if Symbol.debug_lookup:
            Symbol.debug_indent += 1
            Symbol.debug_print("_find_named_symbols:")
            Symbol.debug_indent += 1
            Symbol.debug_print("self:")
            print(self.to_string(Symbol.debug_indent + 1), end="")
            Symbol.debug_print("identOrOp:                  ", identOrOp)
            Symbol.debug_print("templateParams:             ", templateParams)
            Symbol.debug_print("templateArgs:               ", templateArgs)
            Symbol.debug_print("templateShorthand:          ", templateShorthand)
            Symbol.debug_print("matchSelf:                  ", matchSelf)
            Symbol.debug_print("recurseInAnon:              ", recurseInAnon)
            Symbol.debug_print("correctPrimaryTemplateAargs:", correctPrimaryTemplateArgs)
            Symbol.debug_print("searchInSiblings:           ", searchInSiblings)

        def isSpecialization() -> bool:
            # the names of the template parameters must be given exactly as args
            # and params that are packs must in the args be the name expanded
            if len(templateParams.params) != len(templateArgs.args):
                return True
            # having no template params and no arguments is also a specialization
            if len(templateParams.params) == 0:
                return True
            for i in range(len(templateParams.params)):
                param = templateParams.params[i]
                arg = templateArgs.args[i]
                # TODO: doing this by string manipulation is probably not the most efficient
                paramName = str(param.name)
                argTxt = str(arg)
                isArgPackExpansion = argTxt.endswith('...')
                if param.isPack != isArgPackExpansion:
                    return True
                argName = argTxt[:-3] if isArgPackExpansion else argTxt
                if paramName != argName:
                    return True
            return False
        if correctPrimaryTemplateArgs:
            if templateParams is not None and templateArgs is not None:
                # If both are given, but it's not a specialization, then do lookup as if
                # there is no argument list.
                # For example: template<typename T> int A<T>::var;
                if not isSpecialization():
                    templateArgs = None

        def matches(s: "Symbol") -> bool:
            if s.identOrOp != identOrOp:
                return False
            if (s.templateParams is None) != (templateParams is None):
                if templateParams is not None:
                    # we query with params, they must match params
                    return False
                if not templateShorthand:
                    # we don't query with params, and we do care about them
                    return False
            if templateParams:
                # TODO: do better comparison
                if str(s.templateParams) != str(templateParams):
                    return False
            if (s.templateArgs is None) != (templateArgs is None):
                return False
            if s.templateArgs:
                # TODO: do better comparison
                if str(s.templateArgs) != str(templateArgs):
                    return False
            return True

        def candidates() -> Generator[Symbol, None, None]:
            s = self
            if Symbol.debug_lookup:
                Symbol.debug_print("searching in self:")
                print(s.to_string(Symbol.debug_indent + 1), end="")
            while True:
                if matchSelf:
                    yield s
                if recurseInAnon:
                    yield from s.children_recurse_anon
                else:
                    yield from s._children

                if s.siblingAbove is None:
                    break
                s = s.siblingAbove
                if Symbol.debug_lookup:
                    Symbol.debug_print("searching in sibling:")
                    print(s.to_string(Symbol.debug_indent + 1), end="")

        for s in candidates():
            if Symbol.debug_lookup:
                Symbol.debug_print("candidate:")
                print(s.to_string(Symbol.debug_indent + 1), end="")
            if matches(s):
                if Symbol.debug_lookup:
                    Symbol.debug_indent += 1
                    Symbol.debug_print("matches")
                    Symbol.debug_indent -= 3
                yield s
                if Symbol.debug_lookup:
                    Symbol.debug_indent += 2
        if Symbol.debug_lookup:
            Symbol.debug_indent -= 2
```
### 7 - sphinx/domains/cpp.py:

Start line: 4088, End line: 4851

```python
class Symbol:

    def _symbol_lookup(self, nestedName: ASTNestedName, templateDecls: List[Any],
                       onMissingQualifiedSymbol: Callable[["Symbol", Union[ASTIdentifier, ASTOperator], Any, ASTTemplateArgs], "Symbol"],  # NOQA
                       strictTemplateParamArgLists: bool, ancestorLookupType: str,
                       templateShorthand: bool, matchSelf: bool,
                       recurseInAnon: bool, correctPrimaryTemplateArgs: bool,
                       searchInSiblings: bool) -> SymbolLookupResult:
        # ancestorLookupType: if not None, specifies the target type of the lookup
        if Symbol.debug_lookup:
            Symbol.debug_indent += 1
            Symbol.debug_print("_symbol_lookup:")
            Symbol.debug_indent += 1
            Symbol.debug_print("self:")
            print(self.to_string(Symbol.debug_indent + 1), end="")
            Symbol.debug_print("nestedName:        ", nestedName)
            Symbol.debug_print("templateDecls:     ", templateDecls)
            Symbol.debug_print("strictTemplateParamArgLists:", strictTemplateParamArgLists)
            Symbol.debug_print("ancestorLookupType:", ancestorLookupType)
            Symbol.debug_print("templateShorthand: ", templateShorthand)
            Symbol.debug_print("matchSelf:         ", matchSelf)
            Symbol.debug_print("recurseInAnon:     ", recurseInAnon)
            Symbol.debug_print("correctPrimaryTemplateArgs: ", correctPrimaryTemplateArgs)
            Symbol.debug_print("searchInSiblings:  ", searchInSiblings)

        if strictTemplateParamArgLists:
            # Each template argument list must have a template parameter list.
            # But to declare a template there must be an additional template parameter list.
            assert (nestedName.num_templates() == len(templateDecls) or
                    nestedName.num_templates() + 1 == len(templateDecls))
        else:
            assert len(templateDecls) <= nestedName.num_templates() + 1

        names = nestedName.names

        # find the right starting point for lookup
        parentSymbol = self
        if nestedName.rooted:
            while parentSymbol.parent:
                parentSymbol = parentSymbol.parent
        if ancestorLookupType is not None:
            # walk up until we find the first identifier
            firstName = names[0]
            if not firstName.is_operator():
                while parentSymbol.parent:
                    if parentSymbol.find_identifier(firstName.identOrOp,
                                                    matchSelf=matchSelf,
                                                    recurseInAnon=recurseInAnon,
                                                    searchInSiblings=searchInSiblings):
                        # if we are in the scope of a constructor but wants to
                        # reference the class we need to walk one extra up
                        if (len(names) == 1 and ancestorLookupType == 'class' and matchSelf and
                                parentSymbol.parent and
                                parentSymbol.parent.identOrOp == firstName.identOrOp):
                            pass
                        else:
                            break
                    parentSymbol = parentSymbol.parent

        if Symbol.debug_lookup:
            Symbol.debug_print("starting point:")
            print(parentSymbol.to_string(Symbol.debug_indent + 1), end="")

        # and now the actual lookup
        iTemplateDecl = 0
        for name in names[:-1]:
            identOrOp = name.identOrOp
            templateArgs = name.templateArgs
            if strictTemplateParamArgLists:
                # there must be a parameter list
                if templateArgs:
                    assert iTemplateDecl < len(templateDecls)
                    templateParams = templateDecls[iTemplateDecl]
                    iTemplateDecl += 1
                else:
                    templateParams = None
            else:
                # take the next template parameter list if there is one
                # otherwise it's ok
                if templateArgs and iTemplateDecl < len(templateDecls):
                    templateParams = templateDecls[iTemplateDecl]
                    iTemplateDecl += 1
                else:
                    templateParams = None

            symbol = parentSymbol._find_first_named_symbol(
                identOrOp,
                templateParams, templateArgs,
                templateShorthand=templateShorthand,
                matchSelf=matchSelf,
                recurseInAnon=recurseInAnon,
                correctPrimaryTemplateArgs=correctPrimaryTemplateArgs)
            if symbol is None:
                symbol = onMissingQualifiedSymbol(parentSymbol, identOrOp,
                                                  templateParams, templateArgs)
                if symbol is None:
                    if Symbol.debug_lookup:
                        Symbol.debug_indent -= 2
                    return None
            # We have now matched part of a nested name, and need to match more
            # so even if we should matchSelf before, we definitely shouldn't
            # even more. (see also issue #2666)
            matchSelf = False
            parentSymbol = symbol

        if Symbol.debug_lookup:
            Symbol.debug_print("handle last name from:")
            print(parentSymbol.to_string(Symbol.debug_indent + 1), end="")

        # handle the last name
        name = names[-1]
        identOrOp = name.identOrOp
        templateArgs = name.templateArgs
        if iTemplateDecl < len(templateDecls):
            assert iTemplateDecl + 1 == len(templateDecls)
            templateParams = templateDecls[iTemplateDecl]
        else:
            assert iTemplateDecl == len(templateDecls)
            templateParams = None

        symbols = parentSymbol._find_named_symbols(
            identOrOp, templateParams, templateArgs,
            templateShorthand=templateShorthand, matchSelf=matchSelf,
            recurseInAnon=recurseInAnon, correctPrimaryTemplateArgs=False,
            searchInSiblings=searchInSiblings)
        if Symbol.debug_lookup:
            symbols = list(symbols)  # type: ignore
            Symbol.debug_indent -= 2
        return SymbolLookupResult(symbols, parentSymbol,
                                  identOrOp, templateParams, templateArgs)

    def _add_symbols(self, nestedName: ASTNestedName, templateDecls: List[Any],
                     declaration: ASTDeclaration, docname: str) -> "Symbol":
        # Used for adding a whole path of symbols, where the last may or may not
        # be an actual declaration.

        if Symbol.debug_lookup:
            Symbol.debug_indent += 1
            Symbol.debug_print("_add_symbols:")
            Symbol.debug_indent += 1
            Symbol.debug_print("tdecls:", templateDecls)
            Symbol.debug_print("nn:    ", nestedName)
            Symbol.debug_print("decl:  ", declaration)
            Symbol.debug_print("doc:   ", docname)

        def onMissingQualifiedSymbol(parentSymbol: "Symbol",
                                     identOrOp: Union[ASTIdentifier, ASTOperator],
                                     templateParams: Any, templateArgs: ASTTemplateArgs
                                     ) -> "Symbol":
            if Symbol.debug_lookup:
                Symbol.debug_indent += 1
                Symbol.debug_print("_add_symbols, onMissingQualifiedSymbol:")
                Symbol.debug_indent += 1
                Symbol.debug_print("templateParams:", templateParams)
                Symbol.debug_print("identOrOp:     ", identOrOp)
                Symbol.debug_print("templateARgs:  ", templateArgs)
                Symbol.debug_indent -= 2
            return Symbol(parent=parentSymbol, identOrOp=identOrOp,
                          templateParams=templateParams,
                          templateArgs=templateArgs, declaration=None,
                          docname=None)

        lookupResult = self._symbol_lookup(nestedName, templateDecls,
                                           onMissingQualifiedSymbol,
                                           strictTemplateParamArgLists=True,
                                           ancestorLookupType=None,
                                           templateShorthand=False,
                                           matchSelf=False,
                                           recurseInAnon=False,
                                           correctPrimaryTemplateArgs=True,
                                           searchInSiblings=False)
        assert lookupResult is not None  # we create symbols all the way, so that can't happen
        symbols = list(lookupResult.symbols)
        if len(symbols) == 0:
            if Symbol.debug_lookup:
                Symbol.debug_print("_add_symbols, result, no symbol:")
                Symbol.debug_indent += 1
                Symbol.debug_print("templateParams:", lookupResult.templateParams)
                Symbol.debug_print("identOrOp:     ", lookupResult.identOrOp)
                Symbol.debug_print("templateArgs:  ", lookupResult.templateArgs)
                Symbol.debug_print("declaration:   ", declaration)
                Symbol.debug_print("docname:       ", docname)
                Symbol.debug_indent -= 1
            symbol = Symbol(parent=lookupResult.parentSymbol,
                            identOrOp=lookupResult.identOrOp,
                            templateParams=lookupResult.templateParams,
                            templateArgs=lookupResult.templateArgs,
                            declaration=declaration,
                            docname=docname)
            if Symbol.debug_lookup:
                Symbol.debug_indent -= 2
            return symbol

        if Symbol.debug_lookup:
            Symbol.debug_print("_add_symbols, result, symbols:")
            Symbol.debug_indent += 1
            Symbol.debug_print("number symbols:", len(symbols))
            Symbol.debug_indent -= 1

        if not declaration:
            if Symbol.debug_lookup:
                Symbol.debug_print("no delcaration")
                Symbol.debug_indent -= 2
            # good, just a scope creation
            # TODO: what if we have more than one symbol?
            return symbols[0]

        noDecl = []
        withDecl = []
        dupDecl = []
        for s in symbols:
            if s.declaration is None:
                noDecl.append(s)
            elif s.isRedeclaration:
                dupDecl.append(s)
            else:
                withDecl.append(s)
        if Symbol.debug_lookup:
            Symbol.debug_print("#noDecl:  ", len(noDecl))
            Symbol.debug_print("#withDecl:", len(withDecl))
            Symbol.debug_print("#dupDecl: ", len(dupDecl))
        # With partial builds we may start with a large symbol tree stripped of declarations.
        # Essentially any combination of noDecl, withDecl, and dupDecls seems possible.
        # TODO: make partial builds fully work. What should happen when the primary symbol gets
        #  deleted, and other duplicates exist? The full document should probably be rebuild.

        # First check if one of those with a declaration matches.
        # If it's a function, we need to compare IDs,
        # otherwise there should be only one symbol with a declaration.
        def makeCandSymbol() -> "Symbol":
            if Symbol.debug_lookup:
                Symbol.debug_print("begin: creating candidate symbol")
            symbol = Symbol(parent=lookupResult.parentSymbol,
                            identOrOp=lookupResult.identOrOp,
                            templateParams=lookupResult.templateParams,
                            templateArgs=lookupResult.templateArgs,
                            declaration=declaration,
                            docname=docname)
            if Symbol.debug_lookup:
                Symbol.debug_print("end:   creating candidate symbol")
            return symbol
        if len(withDecl) == 0:
            candSymbol = None
        else:
            candSymbol = makeCandSymbol()

            def handleDuplicateDeclaration(symbol: "Symbol", candSymbol: "Symbol") -> None:
                if Symbol.debug_lookup:
                    Symbol.debug_indent += 1
                    Symbol.debug_print("redeclaration")
                    Symbol.debug_indent -= 1
                    Symbol.debug_indent -= 2
                # Redeclaration of the same symbol.
                # Let the new one be there, but raise an error to the client
                # so it can use the real symbol as subscope.
                # This will probably result in a duplicate id warning.
                candSymbol.isRedeclaration = True
                raise _DuplicateSymbolError(symbol, declaration)

            if declaration.objectType != "function":
                assert len(withDecl) <= 1
                handleDuplicateDeclaration(withDecl[0], candSymbol)
                # (not reachable)

            # a function, so compare IDs
            candId = declaration.get_newest_id()
            if Symbol.debug_lookup:
                Symbol.debug_print("candId:", candId)
            for symbol in withDecl:
                oldId = symbol.declaration.get_newest_id()
                if Symbol.debug_lookup:
                    Symbol.debug_print("oldId: ", oldId)
                if candId == oldId:
                    handleDuplicateDeclaration(symbol, candSymbol)
                    # (not reachable)
            # no candidate symbol found with matching ID
        # if there is an empty symbol, fill that one
        if len(noDecl) == 0:
            if Symbol.debug_lookup:
                Symbol.debug_print("no match, no empty, candSybmol is not None?:", candSymbol is not None)  # NOQA
                Symbol.debug_indent -= 2
            if candSymbol is not None:
                return candSymbol
            else:
                return makeCandSymbol()
        else:
            if Symbol.debug_lookup:
                Symbol.debug_print("no match, but fill an empty declaration, candSybmol is not None?:", candSymbol is not None)  # NOQA
                Symbol.debug_indent -= 2
            if candSymbol is not None:
                candSymbol.remove()
            # assert len(noDecl) == 1
            # TODO: enable assertion when we at some point find out how to do cleanup
            # for now, just take the first one, it should work fine ... right?
            symbol = noDecl[0]
            # If someone first opened the scope, and then later
            # declares it, e.g,
            # .. namespace:: Test
            # .. namespace:: nullptr
            # .. class:: Test
            symbol._fill_empty(declaration, docname)
            return symbol

    def merge_with(self, other: "Symbol", docnames: List[str],
                   env: "BuildEnvironment") -> None:
        if Symbol.debug_lookup:
            Symbol.debug_indent += 1
            Symbol.debug_print("merge_with:")
        assert other is not None

        def unconditionalAdd(self, otherChild):
            # TODO: hmm, should we prune by docnames?
            self._children.append(otherChild)
            otherChild.parent = self
            otherChild._assert_invariants()

        if Symbol.debug_lookup:
            Symbol.debug_indent += 1
        for otherChild in other._children:
            if Symbol.debug_lookup:
                Symbol.debug_print("otherChild:\n", otherChild.to_string(Symbol.debug_indent))
                Symbol.debug_indent += 1
            if otherChild.isRedeclaration:
                unconditionalAdd(self, otherChild)
                if Symbol.debug_lookup:
                    Symbol.debug_print("isRedeclaration")
                    Symbol.debug_indent -= 1
                continue
            candiateIter = self._find_named_symbols(
                identOrOp=otherChild.identOrOp,
                templateParams=otherChild.templateParams,
                templateArgs=otherChild.templateArgs,
                templateShorthand=False, matchSelf=False,
                recurseInAnon=False, correctPrimaryTemplateArgs=False,
                searchInSiblings=False)
            candidates = list(candiateIter)

            if Symbol.debug_lookup:
                Symbol.debug_print("raw candidate symbols:", len(candidates))
            symbols = [s for s in candidates if not s.isRedeclaration]
            if Symbol.debug_lookup:
                Symbol.debug_print("non-duplicate candidate symbols:", len(symbols))

            if len(symbols) == 0:
                unconditionalAdd(self, otherChild)
                if Symbol.debug_lookup:
                    Symbol.debug_indent -= 1
                continue

            ourChild = None
            if otherChild.declaration is None:
                if Symbol.debug_lookup:
                    Symbol.debug_print("no declaration in other child")
                ourChild = symbols[0]
            else:
                queryId = otherChild.declaration.get_newest_id()
                if Symbol.debug_lookup:
                    Symbol.debug_print("queryId:  ", queryId)
                for symbol in symbols:
                    if symbol.declaration is None:
                        if Symbol.debug_lookup:
                            Symbol.debug_print("empty candidate")
                        # if in the end we have non matching, but have an empty one,
                        # then just continue with that
                        ourChild = symbol
                        continue
                    candId = symbol.declaration.get_newest_id()
                    if Symbol.debug_lookup:
                        Symbol.debug_print("candidate:", candId)
                    if candId == queryId:
                        ourChild = symbol
                        break
            if Symbol.debug_lookup:
                Symbol.debug_indent -= 1
            if ourChild is None:
                unconditionalAdd(self, otherChild)
                continue
            if otherChild.declaration and otherChild.docname in docnames:
                if not ourChild.declaration:
                    ourChild._fill_empty(otherChild.declaration, otherChild.docname)
                elif ourChild.docname != otherChild.docname:
                    name = str(ourChild.declaration)
                    msg = __("Duplicate C++ declaration, also defined in '%s'.\n"
                             "Declaration is '%s'.")
                    msg = msg % (ourChild.docname, name)
                    logger.warning(msg, location=otherChild.docname)
                else:
                    # Both have declarations, and in the same docname.
                    # This can apparently happen, it should be safe to
                    # just ignore it, right?
                    # Hmm, only on duplicate declarations, right?
                    msg = "Internal C++ domain error during symbol merging.\n"
                    msg += "ourChild:\n" + ourChild.to_string(1)
                    msg += "\notherChild:\n" + otherChild.to_string(1)
                    logger.warning(msg, location=otherChild.docname)
            ourChild.merge_with(otherChild, docnames, env)
        if Symbol.debug_lookup:
            Symbol.debug_indent -= 2

    def add_name(self, nestedName: ASTNestedName,
                 templatePrefix: ASTTemplateDeclarationPrefix = None) -> "Symbol":
        if Symbol.debug_lookup:
            Symbol.debug_indent += 1
            Symbol.debug_print("add_name:")
        if templatePrefix:
            templateDecls = templatePrefix.templates
        else:
            templateDecls = []
        res = self._add_symbols(nestedName, templateDecls,
                                declaration=None, docname=None)
        if Symbol.debug_lookup:
            Symbol.debug_indent -= 1
        return res

    def add_declaration(self, declaration: ASTDeclaration, docname: str) -> "Symbol":
        if Symbol.debug_lookup:
            Symbol.debug_indent += 1
            Symbol.debug_print("add_declaration:")
        assert declaration
        assert docname
        nestedName = declaration.name
        if declaration.templatePrefix:
            templateDecls = declaration.templatePrefix.templates
        else:
            templateDecls = []
        res = self._add_symbols(nestedName, templateDecls, declaration, docname)
        if Symbol.debug_lookup:
            Symbol.debug_indent -= 1
        return res

    def find_identifier(self, identOrOp: Union[ASTIdentifier, ASTOperator],
                        matchSelf: bool, recurseInAnon: bool, searchInSiblings: bool
                        ) -> "Symbol":
        if Symbol.debug_lookup:
            Symbol.debug_indent += 1
            Symbol.debug_print("find_identifier:")
            Symbol.debug_indent += 1
            Symbol.debug_print("identOrOp:       ", identOrOp)
            Symbol.debug_print("matchSelf:       ", matchSelf)
            Symbol.debug_print("recurseInAnon:   ", recurseInAnon)
            Symbol.debug_print("searchInSiblings:", searchInSiblings)
            print(self.to_string(Symbol.debug_indent + 1), end="")
            Symbol.debug_indent -= 2
        current = self
        while current is not None:
            if Symbol.debug_lookup:
                Symbol.debug_indent += 2
                Symbol.debug_print("trying:")
                print(current.to_string(Symbol.debug_indent + 1), end="")
                Symbol.debug_indent -= 2
            if matchSelf and current.identOrOp == identOrOp:
                return current
            children = current.children_recurse_anon if recurseInAnon else current._children
            for s in children:
                if s.identOrOp == identOrOp:
                    return s
            if not searchInSiblings:
                break
            current = current.siblingAbove
        return None

    def direct_lookup(self, key: "LookupKey") -> "Symbol":
        if Symbol.debug_lookup:
            Symbol.debug_indent += 1
            Symbol.debug_print("direct_lookup:")
            Symbol.debug_indent += 1
        s = self
        for name, templateParams, id_ in key.data:
            if id_ is not None:
                res = None
                for cand in s._children:
                    if cand.declaration is None:
                        continue
                    if cand.declaration.get_newest_id() == id_:
                        res = cand
                        break
                s = res
            else:
                identOrOp = name.identOrOp
                templateArgs = name.templateArgs
                s = s._find_first_named_symbol(identOrOp,
                                               templateParams, templateArgs,
                                               templateShorthand=False,
                                               matchSelf=False,
                                               recurseInAnon=False,
                                               correctPrimaryTemplateArgs=False)
            if Symbol.debug_lookup:
                Symbol.debug_print("name:          ", name)
                Symbol.debug_print("templateParams:", templateParams)
                Symbol.debug_print("id:            ", id_)
                if s is not None:
                    print(s.to_string(Symbol.debug_indent + 1), end="")
                else:
                    Symbol.debug_print("not found")
            if s is None:
                if Symbol.debug_lookup:
                    Symbol.debug_indent -= 2
                return None
        if Symbol.debug_lookup:
            Symbol.debug_indent -= 2
        return s

    def find_name(self, nestedName: ASTNestedName, templateDecls: List[Any],
                  typ: str, templateShorthand: bool, matchSelf: bool,
                  recurseInAnon: bool, searchInSiblings: bool) -> Tuple[List["Symbol"], str]:
        # templateShorthand: missing template parameter lists for templates is ok
        # If the first component is None,
        # then the second component _may_ be a string explaining why.
        if Symbol.debug_lookup:
            Symbol.debug_indent += 1
            Symbol.debug_print("find_name:")
            Symbol.debug_indent += 1
            Symbol.debug_print("self:")
            print(self.to_string(Symbol.debug_indent + 1), end="")
            Symbol.debug_print("nestedName:       ", nestedName)
            Symbol.debug_print("templateDecls:    ", templateDecls)
            Symbol.debug_print("typ:              ", typ)
            Symbol.debug_print("templateShorthand:", templateShorthand)
            Symbol.debug_print("matchSelf:        ", matchSelf)
            Symbol.debug_print("recurseInAnon:    ", recurseInAnon)
            Symbol.debug_print("searchInSiblings: ", searchInSiblings)

        class QualifiedSymbolIsTemplateParam(Exception):
            pass

        def onMissingQualifiedSymbol(parentSymbol: "Symbol",
                                     identOrOp: Union[ASTIdentifier, ASTOperator],
                                     templateParams: Any,
                                     templateArgs: ASTTemplateArgs) -> "Symbol":
            # TODO: Maybe search without template args?
            #       Though, the correctPrimaryTemplateArgs does
            #       that for primary templates.
            #       Is there another case where it would be good?
            if parentSymbol.declaration is not None:
                if parentSymbol.declaration.objectType == 'templateParam':
                    raise QualifiedSymbolIsTemplateParam()
            return None

        try:
            lookupResult = self._symbol_lookup(nestedName, templateDecls,
                                               onMissingQualifiedSymbol,
                                               strictTemplateParamArgLists=False,
                                               ancestorLookupType=typ,
                                               templateShorthand=templateShorthand,
                                               matchSelf=matchSelf,
                                               recurseInAnon=recurseInAnon,
                                               correctPrimaryTemplateArgs=False,
                                               searchInSiblings=searchInSiblings)
        except QualifiedSymbolIsTemplateParam:
            return None, "templateParamInQualified"

        if lookupResult is None:
            # if it was a part of the qualification that could not be found
            if Symbol.debug_lookup:
                Symbol.debug_indent -= 2
            return None, None

        res = list(lookupResult.symbols)
        if len(res) != 0:
            if Symbol.debug_lookup:
                Symbol.debug_indent -= 2
            return res, None

        if lookupResult.parentSymbol.declaration is not None:
            if lookupResult.parentSymbol.declaration.objectType == 'templateParam':
                return None, "templateParamInQualified"

        # try without template params and args
        symbol = lookupResult.parentSymbol._find_first_named_symbol(
            lookupResult.identOrOp, None, None,
            templateShorthand=templateShorthand, matchSelf=matchSelf,
            recurseInAnon=recurseInAnon, correctPrimaryTemplateArgs=False)
        if Symbol.debug_lookup:
            Symbol.debug_indent -= 2
        if symbol is not None:
            return [symbol], None
        else:
            return None, None

    def find_declaration(self, declaration: ASTDeclaration, typ: str, templateShorthand: bool,
                         matchSelf: bool, recurseInAnon: bool) -> "Symbol":
        # templateShorthand: missing template parameter lists for templates is ok
        if Symbol.debug_lookup:
            Symbol.debug_indent += 1
            Symbol.debug_print("find_declaration:")
        nestedName = declaration.name
        if declaration.templatePrefix:
            templateDecls = declaration.templatePrefix.templates
        else:
            templateDecls = []

        def onMissingQualifiedSymbol(parentSymbol: "Symbol",
                                     identOrOp: Union[ASTIdentifier, ASTOperator],
                                     templateParams: Any,
                                     templateArgs: ASTTemplateArgs) -> "Symbol":
            return None

        lookupResult = self._symbol_lookup(nestedName, templateDecls,
                                           onMissingQualifiedSymbol,
                                           strictTemplateParamArgLists=False,
                                           ancestorLookupType=typ,
                                           templateShorthand=templateShorthand,
                                           matchSelf=matchSelf,
                                           recurseInAnon=recurseInAnon,
                                           correctPrimaryTemplateArgs=False,
                                           searchInSiblings=False)
        if Symbol.debug_lookup:
            Symbol.debug_indent -= 1
        if lookupResult is None:
            return None

        symbols = list(lookupResult.symbols)
        if len(symbols) == 0:
            return None

        querySymbol = Symbol(parent=lookupResult.parentSymbol,
                             identOrOp=lookupResult.identOrOp,
                             templateParams=lookupResult.templateParams,
                             templateArgs=lookupResult.templateArgs,
                             declaration=declaration,
                             docname='fakeDocnameForQuery')
        queryId = declaration.get_newest_id()
        for symbol in symbols:
            if symbol.declaration is None:
                continue
            candId = symbol.declaration.get_newest_id()
            if candId == queryId:
                querySymbol.remove()
                return symbol
        querySymbol.remove()
        return None

    def to_string(self, indent: int) -> str:
        res = [Symbol.debug_indent_string * indent]
        if not self.parent:
            res.append('::')
        else:
            if self.templateParams:
                res.append(str(self.templateParams))
                res.append('\n')
                res.append(Symbol.debug_indent_string * indent)
            if self.identOrOp:
                res.append(str(self.identOrOp))
            else:
                res.append(str(self.declaration))
            if self.templateArgs:
                res.append(str(self.templateArgs))
            if self.declaration:
                res.append(": ")
                if self.isRedeclaration:
                    res.append('!!duplicate!! ')
                res.append(str(self.declaration))
        if self.docname:
            res.append('\t(')
            res.append(self.docname)
            res.append(')')
        res.append('\n')
        return ''.join(res)

    def dump(self, indent: int) -> str:
        res = [self.to_string(indent)]
        for c in self._children:
            res.append(c.dump(indent + 1))
        return ''.join(res)


class DefinitionParser(BaseParser):
    # those without signedness and size modifiers
    # see https://en.cppreference.com/w/cpp/language/types
    _simple_fundemental_types = (
        'void', 'bool', 'char', 'wchar_t', 'char16_t', 'char32_t', 'int',
        'float', 'double', 'auto'
    )

    _prefix_keys = ('class', 'struct', 'enum', 'union', 'typename')

    @property
    def language(self) -> str:
        return 'C++'

    @property
    def id_attributes(self):
        return self.config.cpp_id_attributes

    @property
    def paren_attributes(self):
        return self.config.cpp_paren_attributes

    def _parse_string(self) -> str:
        if self.current_char != '"':
            return None
        startPos = self.pos
        self.pos += 1
        escape = False
        while True:
            if self.eof:
                self.fail("Unexpected end during inside string.")
            elif self.current_char == '"' and not escape:
                self.pos += 1
                break
            elif self.current_char == '\\':
                escape = True
            else:
                escape = False
            self.pos += 1
        return self.definition[startPos:self.pos]

    def _parse_literal(self) -> ASTLiteral:
        # -> integer-literal
        #  | character-literal
        #  | floating-literal
        #  | string-literal
        #  | boolean-literal -> "false" | "true"
        #  | pointer-literal -> "nullptr"
        #  | user-defined-literal

        def _udl(literal: ASTLiteral) -> ASTLiteral:
            if not self.match(udl_identifier_re):
                return literal
            # hmm, should we care if it's a keyword?
            # it looks like GCC does not disallow keywords
            ident = ASTIdentifier(self.matched_text)
            return ASTUserDefinedLiteral(literal, ident)

        self.skip_ws()
        if self.skip_word('nullptr'):
            return ASTPointerLiteral()
        if self.skip_word('true'):
            return ASTBooleanLiteral(True)
        if self.skip_word('false'):
            return ASTBooleanLiteral(False)
        pos = self.pos
        if self.match(float_literal_re):
            hasSuffix = self.match(float_literal_suffix_re)
            floatLit = ASTNumberLiteral(self.definition[pos:self.pos])
            if hasSuffix:
                return floatLit
            else:
                return _udl(floatLit)
        for regex in [binary_literal_re, hex_literal_re,
                      integer_literal_re, octal_literal_re]:
            if self.match(regex):
                hasSuffix = self.match(integers_literal_suffix_re)
                intLit = ASTNumberLiteral(self.definition[pos:self.pos])
                if hasSuffix:
                    return intLit
                else:
                    return _udl(intLit)

        string = self._parse_string()
        if string is not None:
            return _udl(ASTStringLiteral(string))

        # character-literal
        if self.match(char_literal_re):
            prefix = self.last_match.group(1)  # may be None when no prefix
            data = self.last_match.group(2)
            try:
                charLit = ASTCharLiteral(prefix, data)
            except UnicodeDecodeError as e:
                self.fail("Can not handle character literal. Internal error was: %s" % e)
            except UnsupportedMultiCharacterCharLiteral:
                self.fail("Can not handle character literal"
                          " resulting in multiple decoded characters.")
            return _udl(charLit)
        return None
    # ... other code
```
### 8 - sphinx/domains/cpp.py:

Start line: 11, End line: 697

```python
import re
from typing import (
    Any, Callable, Dict, Generator, Iterator, List, Tuple, Type, TypeVar, Union, Optional
)

from docutils import nodes
from docutils.nodes import Element, Node, TextElement, system_message
from docutils.parsers.rst import directives

from sphinx import addnodes
from sphinx.addnodes import desc_signature, pending_xref
from sphinx.application import Sphinx
from sphinx.builders import Builder
from sphinx.directives import ObjectDescription
from sphinx.domains import Domain, ObjType
from sphinx.environment import BuildEnvironment
from sphinx.errors import NoUri
from sphinx.locale import _, __
from sphinx.roles import SphinxRole, XRefRole
from sphinx.transforms import SphinxTransform
from sphinx.transforms.post_transforms import ReferencesResolver
from sphinx.util import logging
from sphinx.util.cfamily import (
    NoOldIdError, ASTBaseBase, ASTAttribute, ASTBaseParenExprList,
    verify_description_mode, StringifyTransform,
    BaseParser, DefinitionError, UnsupportedMultiCharacterCharLiteral,
    identifier_re, anon_identifier_re, integer_literal_re, octal_literal_re,
    hex_literal_re, binary_literal_re, integers_literal_suffix_re,
    float_literal_re, float_literal_suffix_re,
    char_literal_re
)
from sphinx.util.docfields import Field, GroupedField
from sphinx.util.docutils import SphinxDirective
from sphinx.util.nodes import make_refnode


logger = logging.getLogger(__name__)
T = TypeVar('T')

"""
    Important note on ids
    ----------------------------------------------------------------------------

    Multiple id generation schemes are used due to backwards compatibility.
    - v1: 1.2.3 <= version < 1.3
          The style used before the rewrite.
          It is not the actual old code, but a replication of the behaviour.
    - v2: 1.3 <= version < now
          Standardised mangling scheme from
          https://itanium-cxx-abi.github.io/cxx-abi/abi.html#mangling
          though not completely implemented.
    All versions are generated and attached to elements. The newest is used for
    the index. All of the versions should work as permalinks.


    Signature Nodes and Tagnames
    ----------------------------------------------------------------------------

    Each signature is in a desc_signature node, where all children are
    desc_signature_line nodes. Each of these lines will have the attribute
    'sphinx_line_type' set to one of the following (prioritized):
    - 'declarator', if the line contains the name of the declared object.
    - 'templateParams', if the line starts a template parameter list,
    - 'templateParams', if the line has template parameters
      Note: such lines might get a new tag in the future.
    - 'templateIntroduction, if the line is on the form 'conceptName{...}'
    No other desc_signature nodes should exist (so far).


    Grammar
    ----------------------------------------------------------------------------

    See https://www.nongnu.org/hcb/ for the grammar,
    and https://github.com/cplusplus/draft/blob/master/source/grammar.tex,
    and https://github.com/cplusplus/concepts-ts
    for the newest grammar.

    common grammar things:
        template-declaration ->
            "template" "<" template-parameter-list ">" declaration
        template-parameter-list ->
              template-parameter
            | template-parameter-list "," template-parameter
        template-parameter ->
              type-parameter
            | parameter-declaration # i.e., same as a function argument

        type-parameter ->
              "class"    "..."[opt] identifier[opt]
            | "class"               identifier[opt] "=" type-id
            | "typename" "..."[opt] identifier[opt]
            | "typename"            identifier[opt] "=" type-id
            | "template" "<" template-parameter-list ">"
                "class"  "..."[opt] identifier[opt]
            | "template" "<" template-parameter-list ">"
                "class"             identifier[opt] "=" id-expression
            # also, from C++17 we can have "typename" in template templates
        templateDeclPrefix ->
            "template" "<" template-parameter-list ">"

        simple-declaration ->
            attribute-specifier-seq[opt] decl-specifier-seq[opt]
                init-declarator-list[opt] ;
        # Make the semicolon optional.
        # For now: drop the attributes (TODO).
        # Use at most 1 init-declarator.
        -> decl-specifier-seq init-declarator
        -> decl-specifier-seq declarator initializer

        decl-specifier ->
              storage-class-specifier ->
                 (  "static" (only for member_object and function_object)
                  | "extern" (only for member_object and function_object)
                  | "register"
                 )
                 thread_local[opt] (only for member_object)
                                   (it can also appear before the others)

            | type-specifier -> trailing-type-specifier
            | function-specifier -> "inline" | "virtual" | "explicit" (only
              for function_object)
            | "friend" (only for function_object)
            | "constexpr" (only for member_object and function_object)
        trailing-type-specifier ->
              simple-type-specifier
            | elaborated-type-specifier
            | typename-specifier
            | cv-qualifier -> "const" | "volatile"
        stricter grammar for decl-specifier-seq (with everything, each object
        uses a subset):
            visibility storage-class-specifier function-specifier "friend"
            "constexpr" "volatile" "const" trailing-type-specifier
            # where trailing-type-specifier can no be cv-qualifier
        # Inside e.g., template paramters a strict subset is used
        # (see type-specifier-seq)
        trailing-type-specifier ->
              simple-type-specifier ->
                ::[opt] nested-name-specifier[opt] type-name
              | ::[opt] nested-name-specifier "template" simple-template-id
              | "char" | "bool" | ect.
              | decltype-specifier
            | elaborated-type-specifier ->
                class-key attribute-specifier-seq[opt] ::[opt]
                nested-name-specifier[opt] identifier
              | class-key ::[opt] nested-name-specifier[opt] template[opt]
                simple-template-id
              | "enum" ::[opt] nested-name-specifier[opt] identifier
            | typename-specifier ->
                "typename" ::[opt] nested-name-specifier identifier
              | "typename" ::[opt] nested-name-specifier template[opt]
                simple-template-id
        class-key -> "class" | "struct" | "union"
        type-name ->* identifier | simple-template-id
        # ignoring attributes and decltype, and then some left-factoring
        trailing-type-specifier ->
            rest-of-trailing
            ("class" | "struct" | "union" | "typename") rest-of-trailing
            build-in -> "char" | "bool" | ect.
            decltype-specifier
        rest-of-trailing -> (with some simplification)
            "::"[opt] list-of-elements-separated-by-::
        element ->
            "template"[opt] identifier ("<" template-argument-list ">")[opt]
        template-argument-list ->
              template-argument "..."[opt]
            | template-argument-list "," template-argument "..."[opt]
        template-argument ->
              constant-expression
            | type-specifier-seq abstract-declarator
            | id-expression


        declarator ->
              ptr-declarator
            | noptr-declarator parameters-and-qualifiers trailing-return-type
              (TODO: for now we don't support trailing-eturn-type)
        ptr-declarator ->
              noptr-declarator
            | ptr-operator ptr-declarator
        noptr-declarator ->
              declarator-id attribute-specifier-seq[opt] ->
                    "..."[opt] id-expression
                  | rest-of-trailing
            | noptr-declarator parameters-and-qualifiers
            | noptr-declarator "[" constant-expression[opt] "]"
              attribute-specifier-seq[opt]
            | "(" ptr-declarator ")"
        ptr-operator ->
              "*"  attribute-specifier-seq[opt] cv-qualifier-seq[opt]
            | "&   attribute-specifier-seq[opt]
            | "&&" attribute-specifier-seq[opt]
            | "::"[opt] nested-name-specifier "*" attribute-specifier-seq[opt]
                cv-qualifier-seq[opt]
        # function_object must use a parameters-and-qualifiers, the others may
        # use it (e.g., function poitners)
        parameters-and-qualifiers ->
            "(" parameter-clause ")" attribute-specifier-seq[opt]
            cv-qualifier-seq[opt] ref-qualifier[opt]
            exception-specification[opt]
        ref-qualifier -> "&" | "&&"
        exception-specification ->
            "noexcept" ("(" constant-expression ")")[opt]
            "throw" ("(" type-id-list ")")[opt]
        # TODO: we don't implement attributes
        # member functions can have initializers, but we fold them into here
        memberFunctionInit -> "=" "0"
        # (note: only "0" is allowed as the value, according to the standard,
        # right?)

        enum-head ->
            enum-key attribute-specifier-seq[opt] nested-name-specifier[opt]
                identifier enum-base[opt]
        enum-key -> "enum" | "enum struct" | "enum class"
        enum-base ->
            ":" type
        enumerator-definition ->
              identifier
            | identifier "=" constant-expression

    We additionally add the possibility for specifying the visibility as the
    first thing.

    concept_object:
        goal:
            just a declaration of the name (for now)

        grammar: only a single template parameter list, and the nested name
            may not have any template argument lists

            "template" "<" template-parameter-list ">"
            nested-name-specifier

    type_object:
        goal:
            either a single type (e.g., "MyClass:Something_T" or a typedef-like
            thing (e.g. "Something Something_T" or "int I_arr[]"
        grammar, single type: based on a type in a function parameter, but
        without a name:
               parameter-declaration
            -> attribute-specifier-seq[opt] decl-specifier-seq
               abstract-declarator[opt]
            # Drop the attributes
            -> decl-specifier-seq abstract-declarator[opt]
        grammar, typedef-like: no initilizer
            decl-specifier-seq declarator
        Can start with a templateDeclPrefix.

    member_object:
        goal: as a type_object which must have a declarator, and optionally
        with a initializer
        grammar:
            decl-specifier-seq declarator initializer
        Can start with a templateDeclPrefix.

    function_object:
        goal: a function declaration, TODO: what about templates? for now: skip
        grammar: no initializer
           decl-specifier-seq declarator
        Can start with a templateDeclPrefix.

    class_object:
        goal: a class declaration, but with specification of a base class
        grammar:
              nested-name "final"[opt] (":" base-specifier-list)[opt]
            base-specifier-list ->
              base-specifier "..."[opt]
            | base-specifier-list, base-specifier "..."[opt]
            base-specifier ->
              base-type-specifier
            | "virtual" access-spe"cifier[opt]    base-type-specifier
            | access-specifier[opt] "virtual"[opt] base-type-specifier
        Can start with a templateDeclPrefix.

    enum_object:
        goal: an unscoped enum or a scoped enum, optionally with the underlying
              type specified
        grammar:
            ("class" | "struct")[opt] visibility[opt] nested-name (":" type)[opt]
    enumerator_object:
        goal: an element in a scoped or unscoped enum. The name should be
              injected according to the scopedness.
        grammar:
            nested-name ("=" constant-expression)

    namespace_object:
        goal: a directive to put all following declarations in a specific scope
        grammar:
            nested-name
"""

udl_identifier_re = re.compile(r'''(?x)
    [a-zA-Z_][a-zA-Z0-9_]*\b   # note, no word boundary in the beginning
''')
_string_re = re.compile(r"[LuU8]?('([^'\\]*(?:\\.[^'\\]*)*)'"
                        r'|"([^"\\]*(?:\\.[^"\\]*)*)")', re.S)
_visibility_re = re.compile(r'\b(public|private|protected)\b')
_operator_re = re.compile(r'''(?x)
        \[\s*\]
    |   \(\s*\)
    |   \+\+ | --
    |   ->\*? | \,
    |   (<<|>>)=? | && | \|\|
    |   [!<>=/*%+|&^~-]=?
    |   (\b(and|and_eq|bitand|bitor|compl|not|not_eq|or|or_eq|xor|xor_eq)\b)
''')
_fold_operator_re = re.compile(r'''(?x)
        ->\*    |    \.\*    |    \,
    |   (<<|>>)=?    |    &&    |    \|\|
    |   !=
    |   [<>=/*%+|&^~-]=?
''')
# see https://en.cppreference.com/w/cpp/keyword
_keywords = [
    'alignas', 'alignof', 'and', 'and_eq', 'asm', 'auto', 'bitand', 'bitor',
    'bool', 'break', 'case', 'catch', 'char', 'char16_t', 'char32_t', 'class',
    'compl', 'concept', 'const', 'constexpr', 'const_cast', 'continue',
    'decltype', 'default', 'delete', 'do', 'double', 'dynamic_cast', 'else',
    'enum', 'explicit', 'export', 'extern', 'false', 'float', 'for', 'friend',
    'goto', 'if', 'inline', 'int', 'long', 'mutable', 'namespace', 'new',
    'noexcept', 'not', 'not_eq', 'nullptr', 'operator', 'or', 'or_eq',
    'private', 'protected', 'public', 'register', 'reinterpret_cast',
    'requires', 'return', 'short', 'signed', 'sizeof', 'static',
    'static_assert', 'static_cast', 'struct', 'switch', 'template', 'this',
    'thread_local', 'throw', 'true', 'try', 'typedef', 'typeid', 'typename',
    'union', 'unsigned', 'using', 'virtual', 'void', 'volatile', 'wchar_t',
    'while', 'xor', 'xor_eq'
]

_max_id = 4
_id_prefix = [None, '', '_CPPv2', '_CPPv3', '_CPPv4']
# Ids are used in lookup keys which are used across pickled files,
# so when _max_id changes, make sure to update the ENV_VERSION.

# ------------------------------------------------------------------------------
# Id v1 constants
# ------------------------------------------------------------------------------

_id_fundamental_v1 = {
    'char': 'c',
    'signed char': 'c',
    'unsigned char': 'C',
    'int': 'i',
    'signed int': 'i',
    'unsigned int': 'U',
    'long': 'l',
    'signed long': 'l',
    'unsigned long': 'L',
    'bool': 'b'
}
_id_shorthands_v1 = {
    'std::string': 'ss',
    'std::ostream': 'os',
    'std::istream': 'is',
    'std::iostream': 'ios',
    'std::vector': 'v',
    'std::map': 'm'
}
_id_operator_v1 = {
    'new': 'new-operator',
    'new[]': 'new-array-operator',
    'delete': 'delete-operator',
    'delete[]': 'delete-array-operator',
    # the arguments will make the difference between unary and binary
    # '+(unary)' : 'ps',
    # '-(unary)' : 'ng',
    # '&(unary)' : 'ad',
    # '*(unary)' : 'de',
    '~': 'inv-operator',
    '+': 'add-operator',
    '-': 'sub-operator',
    '*': 'mul-operator',
    '/': 'div-operator',
    '%': 'mod-operator',
    '&': 'and-operator',
    '|': 'or-operator',
    '^': 'xor-operator',
    '=': 'assign-operator',
    '+=': 'add-assign-operator',
    '-=': 'sub-assign-operator',
    '*=': 'mul-assign-operator',
    '/=': 'div-assign-operator',
    '%=': 'mod-assign-operator',
    '&=': 'and-assign-operator',
    '|=': 'or-assign-operator',
    '^=': 'xor-assign-operator',
    '<<': 'lshift-operator',
    '>>': 'rshift-operator',
    '<<=': 'lshift-assign-operator',
    '>>=': 'rshift-assign-operator',
    '==': 'eq-operator',
    '!=': 'neq-operator',
    '<': 'lt-operator',
    '>': 'gt-operator',
    '<=': 'lte-operator',
    '>=': 'gte-operator',
    '!': 'not-operator',
    '&&': 'sand-operator',
    '||': 'sor-operator',
    '++': 'inc-operator',
    '--': 'dec-operator',
    ',': 'comma-operator',
    '->*': 'pointer-by-pointer-operator',
    '->': 'pointer-operator',
    '()': 'call-operator',
    '[]': 'subscript-operator'
}

# ------------------------------------------------------------------------------
# Id v > 1 constants
# ------------------------------------------------------------------------------

_id_fundamental_v2 = {
    # not all of these are actually parsed as fundamental types, TODO: do that
    'void': 'v',
    'bool': 'b',
    'char': 'c',
    'signed char': 'a',
    'unsigned char': 'h',
    'wchar_t': 'w',
    'char32_t': 'Di',
    'char16_t': 'Ds',
    'short': 's',
    'short int': 's',
    'signed short': 's',
    'signed short int': 's',
    'unsigned short': 't',
    'unsigned short int': 't',
    'int': 'i',
    'signed': 'i',
    'signed int': 'i',
    'unsigned': 'j',
    'unsigned int': 'j',
    'long': 'l',
    'long int': 'l',
    'signed long': 'l',
    'signed long int': 'l',
    'unsigned long': 'm',
    'unsigned long int': 'm',
    'long long': 'x',
    'long long int': 'x',
    'signed long long': 'x',
    'signed long long int': 'x',
    'unsigned long long': 'y',
    'unsigned long long int': 'y',
    'float': 'f',
    'double': 'd',
    'long double': 'e',
    'auto': 'Da',
    'decltype(auto)': 'Dc',
    'std::nullptr_t': 'Dn'
}
_id_operator_v2 = {
    'new': 'nw',
    'new[]': 'na',
    'delete': 'dl',
    'delete[]': 'da',
    # the arguments will make the difference between unary and binary
    # in operator definitions
    # '+(unary)' : 'ps',
    # '-(unary)' : 'ng',
    # '&(unary)' : 'ad',
    # '*(unary)' : 'de',
    '~': 'co', 'compl': 'co',
    '+': 'pl',
    '-': 'mi',
    '*': 'ml',
    '/': 'dv',
    '%': 'rm',
    '&': 'an', 'bitand': 'an',
    '|': 'or', 'bitor': 'or',
    '^': 'eo', 'xor': 'eo',
    '=': 'aS',
    '+=': 'pL',
    '-=': 'mI',
    '*=': 'mL',
    '/=': 'dV',
    '%=': 'rM',
    '&=': 'aN', 'and_eq': 'aN',
    '|=': 'oR', 'or_eq': 'oR',
    '^=': 'eO', 'xor_eq': 'eO',
    '<<': 'ls',
    '>>': 'rs',
    '<<=': 'lS',
    '>>=': 'rS',
    '==': 'eq',
    '!=': 'ne', 'not_eq': 'ne',
    '<': 'lt',
    '>': 'gt',
    '<=': 'le',
    '>=': 'ge',
    '!': 'nt', 'not': 'nt',
    '&&': 'aa', 'and': 'aa',
    '||': 'oo', 'or': 'oo',
    '++': 'pp',
    '--': 'mm',
    ',': 'cm',
    '->*': 'pm',
    '->': 'pt',
    '()': 'cl',
    '[]': 'ix',
    '.*': 'ds'  # this one is not overloadable, but we need it for expressions
}
_id_operator_unary_v2 = {
    '++': 'pp_',
    '--': 'mm_',
    '*': 'de',
    '&': 'ad',
    '+': 'ps',
    '-': 'ng',
    '!': 'nt', 'not': 'nt',
    '~': 'co', 'compl': 'co'
}
_id_char_from_prefix = {
    None: 'c', 'u8': 'c',
    'u': 'Ds', 'U': 'Di', 'L': 'w'
}  # type: Dict[Any, str]
# these are ordered by preceedence
_expression_bin_ops = [
    ['||', 'or'],
    ['&&', 'and'],
    ['|', 'bitor'],
    ['^', 'xor'],
    ['&', 'bitand'],
    ['==', '!=', 'not_eq'],
    ['<=', '>=', '<', '>'],
    ['<<', '>>'],
    ['+', '-'],
    ['*', '/', '%'],
    ['.*', '->*']
]
_expression_unary_ops = ["++", "--", "*", "&", "+", "-", "!", "not", "~", "compl"]
_expression_assignment_ops = ["=", "*=", "/=", "%=", "+=", "-=",
                              ">>=", "<<=", "&=", "and_eq", "^=", "|=", "xor_eq", "or_eq"]
_id_explicit_cast = {
    'dynamic_cast': 'dc',
    'static_cast': 'sc',
    'const_cast': 'cc',
    'reinterpret_cast': 'rc'
}


class _DuplicateSymbolError(Exception):
    def __init__(self, symbol: "Symbol", declaration: "ASTDeclaration") -> None:
        assert symbol
        assert declaration
        self.symbol = symbol
        self.declaration = declaration

    def __str__(self) -> str:
        return "Internal C++ duplicate symbol error:\n%s" % self.symbol.dump(0)


class ASTBase(ASTBaseBase):
    pass


# Names
################################################################################

class ASTIdentifier(ASTBase):
    def __init__(self, identifier: str) -> None:
        assert identifier is not None
        assert len(identifier) != 0
        self.identifier = identifier

    def is_anon(self) -> bool:
        return self.identifier[0] == '@'

    def get_id(self, version: int) -> str:
        if self.is_anon() and version < 3:
            raise NoOldIdError()
        if version == 1:
            if self.identifier == 'size_t':
                return 's'
            else:
                return self.identifier
        if self.identifier == "std":
            return 'St'
        elif self.identifier[0] == "~":
            # a destructor, just use an arbitrary version of dtors
            return 'D0'
        else:
            if self.is_anon():
                return 'Ut%d_%s' % (len(self.identifier) - 1, self.identifier[1:])
            else:
                return str(len(self.identifier)) + self.identifier

    # and this is where we finally make a difference between __str__ and the display string

    def __str__(self) -> str:
        return self.identifier

    def get_display_string(self) -> str:
        return "[anonymous]" if self.is_anon() else self.identifier

    def describe_signature(self, signode: TextElement, mode: str, env: "BuildEnvironment",
                           prefix: str, templateArgs: str, symbol: "Symbol") -> None:
        verify_description_mode(mode)
        if mode == 'markType':
            targetText = prefix + self.identifier + templateArgs
            pnode = addnodes.pending_xref('', refdomain='cpp',
                                          reftype='identifier',
                                          reftarget=targetText, modname=None,
                                          classname=None)
            pnode['cpp:parent_key'] = symbol.get_lookup_key()
            if self.is_anon():
                pnode += nodes.strong(text="[anonymous]")
            else:
                pnode += nodes.Text(self.identifier)
            signode += pnode
        elif mode == 'lastIsName':
            if self.is_anon():
                signode += nodes.strong(text="[anonymous]")
            else:
                signode += addnodes.desc_name(self.identifier, self.identifier)
        elif mode == 'noneIsName':
            if self.is_anon():
                signode += nodes.strong(text="[anonymous]")
            else:
                signode += nodes.Text(self.identifier)
        elif mode == 'udl':
            # the target is 'operator""id' instead of just 'id'
            assert len(prefix) == 0
            assert len(templateArgs) == 0
            assert not self.is_anon()
            targetText = 'operator""' + self.identifier
            pnode = addnodes.pending_xref('', refdomain='cpp',
                                          reftype='identifier',
                                          reftarget=targetText, modname=None,
                                          classname=None)
            pnode['cpp:parent_key'] = symbol.get_lookup_key()
            pnode += nodes.Text(self.identifier)
            signode += pnode
        else:
            raise Exception('Unknown description mode: %s' % mode)


class ASTNestedNameElement(ASTBase):
    def __init__(self, identOrOp: Union[ASTIdentifier, "ASTOperator"],
                 templateArgs: "ASTTemplateArgs") -> None:
        self.identOrOp = identOrOp
        self.templateArgs = templateArgs

    def is_operator(self) -> bool:
        return False

    def get_id(self, version: int) -> str:
        res = self.identOrOp.get_id(version)
        if self.templateArgs:
            res += self.templateArgs.get_id(version)
        return res

    def _stringify(self, transform: StringifyTransform) -> str:
        res = transform(self.identOrOp)
        if self.templateArgs:
            res += transform(self.templateArgs)
        return res

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", prefix: str, symbol: "Symbol") -> None:
        tArgs = str(self.templateArgs) if self.templateArgs is not None else ''
        self.identOrOp.describe_signature(signode, mode, env, prefix, tArgs, symbol)
        if self.templateArgs is not None:
            self.templateArgs.describe_signature(signode, mode, env, symbol)


class ASTNestedName(ASTBase):
    def __init__(self, names: List[ASTNestedNameElement],
                 templates: List[bool], rooted: bool) -> None:
        assert len(names) > 0
        self.names = names
        self.templates = templates
        assert len(self.names) == len(self.templates)
        self.rooted = rooted

    @property
    def name(self) -> "ASTNestedName":
        return self

    def num_templates(self) -> int:
        count = 0
        for n in self.names:
            if n.is_operator():
                continue
            if n.templateArgs:
                count += 1
        return count
```
### 9 - sphinx/domains/c.py:

Start line: 807, End line: 1651

```python
# Declarator
################################################################################

class ASTArray(ASTBase):
    def __init__(self, static: bool, const: bool, volatile: bool, restrict: bool,
                 vla: bool, size: ASTExpression):
        self.static = static
        self.const = const
        self.volatile = volatile
        self.restrict = restrict
        self.vla = vla
        self.size = size
        if vla:
            assert size is None
        if size is not None:
            assert not vla

    def _stringify(self, transform: StringifyTransform) -> str:
        el = []
        if self.static:
            el.append('static')
        if self.restrict:
            el.append('restrict')
        if self.volatile:
            el.append('volatile')
        if self.const:
            el.append('const')
        if self.vla:
            return '[' + ' '.join(el) + '*]'
        elif self.size:
            el.append(transform(self.size))
        return '[' + ' '.join(el) + ']'

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        verify_description_mode(mode)
        signode.append(nodes.Text("["))
        addSpace = False

        def _add(signode: TextElement, text: str) -> bool:
            if addSpace:
                signode += nodes.Text(' ')
            signode += addnodes.desc_annotation(text, text)
            return True

        if self.static:
            addSpace = _add(signode, 'static')
        if self.restrict:
            addSpace = _add(signode, 'restrict')
        if self.volatile:
            addSpace = _add(signode, 'volatile')
        if self.const:
            addSpace = _add(signode, 'const')
        if self.vla:
            signode.append(nodes.Text('*'))
        elif self.size:
            if addSpace:
                signode += nodes.Text(' ')
            self.size.describe_signature(signode, mode, env, symbol)
        signode.append(nodes.Text("]"))


class ASTDeclarator(ASTBase):
    @property
    def name(self) -> ASTNestedName:
        raise NotImplementedError(repr(self))

    @property
    def function_params(self) -> List[ASTFunctionParameter]:
        raise NotImplementedError(repr(self))

    def require_space_after_declSpecs(self) -> bool:
        raise NotImplementedError(repr(self))


class ASTDeclaratorNameParam(ASTDeclarator):
    def __init__(self, declId: ASTNestedName,
                 arrayOps: List[ASTArray], param: ASTParameters) -> None:
        self.declId = declId
        self.arrayOps = arrayOps
        self.param = param

    @property
    def name(self) -> ASTNestedName:
        return self.declId

    @property
    def function_params(self) -> List[ASTFunctionParameter]:
        return self.param.function_params

    # ------------------------------------------------------------------------

    def require_space_after_declSpecs(self) -> bool:
        return self.declId is not None

    def _stringify(self, transform: StringifyTransform) -> str:
        res = []
        if self.declId:
            res.append(transform(self.declId))
        for op in self.arrayOps:
            res.append(transform(op))
        if self.param:
            res.append(transform(self.param))
        return ''.join(res)

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        verify_description_mode(mode)
        if self.declId:
            self.declId.describe_signature(signode, mode, env, symbol)
        for op in self.arrayOps:
            op.describe_signature(signode, mode, env, symbol)
        if self.param:
            self.param.describe_signature(signode, mode, env, symbol)


class ASTDeclaratorNameBitField(ASTDeclarator):
    def __init__(self, declId: ASTNestedName, size: ASTExpression):
        self.declId = declId
        self.size = size

    @property
    def name(self) -> ASTNestedName:
        return self.declId

    # ------------------------------------------------------------------------

    def require_space_after_declSpecs(self) -> bool:
        return self.declId is not None

    def _stringify(self, transform: StringifyTransform) -> str:
        res = []
        if self.declId:
            res.append(transform(self.declId))
        res.append(" : ")
        res.append(transform(self.size))
        return ''.join(res)

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        verify_description_mode(mode)
        if self.declId:
            self.declId.describe_signature(signode, mode, env, symbol)
        signode += nodes.Text(' : ', ' : ')
        self.size.describe_signature(signode, mode, env, symbol)


class ASTDeclaratorPtr(ASTDeclarator):
    def __init__(self, next: ASTDeclarator, restrict: bool, volatile: bool, const: bool,
                 attrs: Any) -> None:
        assert next
        self.next = next
        self.restrict = restrict
        self.volatile = volatile
        self.const = const
        self.attrs = attrs

    @property
    def name(self) -> ASTNestedName:
        return self.next.name

    @property
    def function_params(self) -> List[ASTFunctionParameter]:
        return self.next.function_params

    def require_space_after_declSpecs(self) -> bool:
        return self.const or self.volatile or self.restrict or \
            len(self.attrs) > 0 or \
            self.next.require_space_after_declSpecs()

    def _stringify(self, transform: StringifyTransform) -> str:
        res = ['*']
        for a in self.attrs:
            res.append(transform(a))
        if len(self.attrs) > 0 and (self.restrict or self.volatile or self.const):
            res.append(' ')
        if self.restrict:
            res.append('restrict')
        if self.volatile:
            if self.restrict:
                res.append(' ')
            res.append('volatile')
        if self.const:
            if self.restrict or self.volatile:
                res.append(' ')
            res.append('const')
        if self.const or self.volatile or self.restrict or len(self.attrs) > 0:
            if self.next.require_space_after_declSpecs():
                res.append(' ')
        res.append(transform(self.next))
        return ''.join(res)

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        verify_description_mode(mode)
        signode += nodes.Text("*")
        for a in self.attrs:
            a.describe_signature(signode)
        if len(self.attrs) > 0 and (self.restrict or self.volatile or self.const):
            signode += nodes.Text(' ')

        def _add_anno(signode: TextElement, text: str) -> None:
            signode += addnodes.desc_annotation(text, text)

        if self.restrict:
            _add_anno(signode, 'restrict')
        if self.volatile:
            if self.restrict:
                signode += nodes.Text(' ')
            _add_anno(signode, 'volatile')
        if self.const:
            if self.restrict or self.volatile:
                signode += nodes.Text(' ')
            _add_anno(signode, 'const')
        if self.const or self.volatile or self.restrict or len(self.attrs) > 0:
            if self.next.require_space_after_declSpecs():
                signode += nodes.Text(' ')
        self.next.describe_signature(signode, mode, env, symbol)


class ASTDeclaratorParen(ASTDeclarator):
    def __init__(self, inner: ASTDeclarator, next: ASTDeclarator) -> None:
        assert inner
        assert next
        self.inner = inner
        self.next = next
        # TODO: we assume the name and params are in inner

    @property
    def name(self) -> ASTNestedName:
        return self.inner.name

    @property
    def function_params(self) -> List[ASTFunctionParameter]:
        return self.inner.function_params

    def require_space_after_declSpecs(self) -> bool:
        return True

    def _stringify(self, transform: StringifyTransform) -> str:
        res = ['(']
        res.append(transform(self.inner))
        res.append(')')
        res.append(transform(self.next))
        return ''.join(res)

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        verify_description_mode(mode)
        signode += nodes.Text('(')
        self.inner.describe_signature(signode, mode, env, symbol)
        signode += nodes.Text(')')
        self.next.describe_signature(signode, "noneIsName", env, symbol)


# Initializer
################################################################################

class ASTParenExprList(ASTBaseParenExprList):
    def __init__(self, exprs: List[ASTExpression]) -> None:
        self.exprs = exprs

    def _stringify(self, transform: StringifyTransform) -> str:
        exprs = [transform(e) for e in self.exprs]
        return '(%s)' % ', '.join(exprs)

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        verify_description_mode(mode)
        signode.append(nodes.Text('('))
        first = True
        for e in self.exprs:
            if not first:
                signode.append(nodes.Text(', '))
            else:
                first = False
            e.describe_signature(signode, mode, env, symbol)
        signode.append(nodes.Text(')'))


class ASTBracedInitList(ASTBase):
    def __init__(self, exprs: List[ASTExpression], trailingComma: bool) -> None:
        self.exprs = exprs
        self.trailingComma = trailingComma

    def _stringify(self, transform: StringifyTransform) -> str:
        exprs = [transform(e) for e in self.exprs]
        trailingComma = ',' if self.trailingComma else ''
        return '{%s%s}' % (', '.join(exprs), trailingComma)

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        verify_description_mode(mode)
        signode.append(nodes.Text('{'))
        first = True
        for e in self.exprs:
            if not first:
                signode.append(nodes.Text(', '))
            else:
                first = False
            e.describe_signature(signode, mode, env, symbol)
        if self.trailingComma:
            signode.append(nodes.Text(','))
        signode.append(nodes.Text('}'))


class ASTInitializer(ASTBase):
    def __init__(self, value: Union[ASTBracedInitList, ASTExpression],
                 hasAssign: bool = True) -> None:
        self.value = value
        self.hasAssign = hasAssign

    def _stringify(self, transform: StringifyTransform) -> str:
        val = transform(self.value)
        if self.hasAssign:
            return ' = ' + val
        else:
            return val

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        verify_description_mode(mode)
        if self.hasAssign:
            signode.append(nodes.Text(' = '))
        self.value.describe_signature(signode, 'markType', env, symbol)


class ASTType(ASTBase):
    def __init__(self, declSpecs: ASTDeclSpecs, decl: ASTDeclarator) -> None:
        assert declSpecs
        assert decl
        self.declSpecs = declSpecs
        self.decl = decl

    @property
    def name(self) -> ASTNestedName:
        return self.decl.name

    @property
    def function_params(self) -> List[ASTFunctionParameter]:
        return self.decl.function_params

    def _stringify(self, transform: StringifyTransform) -> str:
        res = []
        declSpecs = transform(self.declSpecs)
        res.append(declSpecs)
        if self.decl.require_space_after_declSpecs() and len(declSpecs) > 0:
            res.append(' ')
        res.append(transform(self.decl))
        return ''.join(res)

    def get_type_declaration_prefix(self) -> str:
        if self.declSpecs.trailingTypeSpec:
            return 'typedef'
        else:
            return 'type'

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        verify_description_mode(mode)
        self.declSpecs.describe_signature(signode, 'markType', env, symbol)
        if (self.decl.require_space_after_declSpecs() and
                len(str(self.declSpecs)) > 0):
            signode += nodes.Text(' ')
        # for parameters that don't really declare new names we get 'markType',
        # this should not be propagated, but be 'noneIsName'.
        if mode == 'markType':
            mode = 'noneIsName'
        self.decl.describe_signature(signode, mode, env, symbol)


class ASTTypeWithInit(ASTBase):
    def __init__(self, type: ASTType, init: ASTInitializer) -> None:
        self.type = type
        self.init = init

    @property
    def name(self) -> ASTNestedName:
        return self.type.name

    def _stringify(self, transform: StringifyTransform) -> str:
        res = []
        res.append(transform(self.type))
        if self.init:
            res.append(transform(self.init))
        return ''.join(res)

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        verify_description_mode(mode)
        self.type.describe_signature(signode, mode, env, symbol)
        if self.init:
            self.init.describe_signature(signode, mode, env, symbol)


class ASTMacroParameter(ASTBase):
    def __init__(self, arg: ASTNestedName, ellipsis: bool = False) -> None:
        self.arg = arg
        self.ellipsis = ellipsis

    def _stringify(self, transform: StringifyTransform) -> str:
        if self.ellipsis:
            return '...'
        else:
            return transform(self.arg)

    def describe_signature(self, signode: Any, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        verify_description_mode(mode)
        if self.ellipsis:
            signode += nodes.Text('...')
        else:
            self.arg.describe_signature(signode, mode, env, symbol=symbol)


class ASTMacro(ASTBase):
    def __init__(self, ident: ASTNestedName, args: List[ASTMacroParameter]) -> None:
        self.ident = ident
        self.args = args

    @property
    def name(self) -> ASTNestedName:
        return self.ident

    def _stringify(self, transform: StringifyTransform) -> str:
        res = []
        res.append(transform(self.ident))
        if self.args is not None:
            res.append('(')
            first = True
            for arg in self.args:
                if not first:
                    res.append(', ')
                first = False
                res.append(transform(arg))
            res.append(')')
        return ''.join(res)

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        verify_description_mode(mode)
        self.ident.describe_signature(signode, mode, env, symbol)
        if self.args is None:
            return
        paramlist = addnodes.desc_parameterlist()
        for arg in self.args:
            param = addnodes.desc_parameter('', '', noemph=True)
            arg.describe_signature(param, 'param', env, symbol=symbol)
            paramlist += param
        signode += paramlist


class ASTStruct(ASTBase):
    def __init__(self, name: ASTNestedName) -> None:
        self.name = name

    def get_id(self, version: int, objectType: str, symbol: "Symbol") -> str:
        return symbol.get_full_nested_name().get_id(version)

    def _stringify(self, transform: StringifyTransform) -> str:
        return transform(self.name)

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        verify_description_mode(mode)
        self.name.describe_signature(signode, mode, env, symbol=symbol)


class ASTUnion(ASTBase):
    def __init__(self, name: ASTNestedName) -> None:
        self.name = name

    def get_id(self, version: int, objectType: str, symbol: "Symbol") -> str:
        return symbol.get_full_nested_name().get_id(version)

    def _stringify(self, transform: StringifyTransform) -> str:
        return transform(self.name)

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        verify_description_mode(mode)
        self.name.describe_signature(signode, mode, env, symbol=symbol)


class ASTEnum(ASTBase):
    def __init__(self, name: ASTNestedName) -> None:
        self.name = name

    def get_id(self, version: int, objectType: str, symbol: "Symbol") -> str:
        return symbol.get_full_nested_name().get_id(version)

    def _stringify(self, transform: StringifyTransform) -> str:
        return transform(self.name)

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        verify_description_mode(mode)
        self.name.describe_signature(signode, mode, env, symbol=symbol)


class ASTEnumerator(ASTBase):
    def __init__(self, name: ASTNestedName, init: ASTInitializer) -> None:
        self.name = name
        self.init = init

    def get_id(self, version: int, objectType: str, symbol: "Symbol") -> str:
        return symbol.get_full_nested_name().get_id(version)

    def _stringify(self, transform: StringifyTransform) -> str:
        res = []
        res.append(transform(self.name))
        if self.init:
            res.append(transform(self.init))
        return ''.join(res)

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        verify_description_mode(mode)
        self.name.describe_signature(signode, mode, env, symbol)
        if self.init:
            self.init.describe_signature(signode, 'markType', env, symbol)


class ASTDeclaration(ASTBaseBase):
    def __init__(self, objectType: str, directiveType: str, declaration: Any,
                 semicolon: bool = False) -> None:
        self.objectType = objectType
        self.directiveType = directiveType
        self.declaration = declaration
        self.semicolon = semicolon

        self.symbol = None  # type: Symbol
        # set by CObject._add_enumerator_to_parent
        self.enumeratorScopedSymbol = None  # type: Symbol

    def clone(self) -> "ASTDeclaration":
        return ASTDeclaration(self.objectType, self.directiveType,
                              self.declaration.clone(), self.semicolon)

    @property
    def name(self) -> ASTNestedName:
        return self.declaration.name

    @property
    def function_params(self) -> List[ASTFunctionParameter]:
        if self.objectType != 'function':
            return None
        return self.declaration.function_params

    def get_id(self, version: int, prefixed: bool = True) -> str:
        if self.objectType == 'enumerator' and self.enumeratorScopedSymbol:
            return self.enumeratorScopedSymbol.declaration.get_id(version, prefixed)
        id_ = self.symbol.get_full_nested_name().get_id(version)
        if prefixed:
            return _id_prefix[version] + id_
        else:
            return id_

    def get_newest_id(self) -> str:
        return self.get_id(_max_id, True)

    def _stringify(self, transform: StringifyTransform) -> str:
        res = transform(self.declaration)
        if self.semicolon:
            res += ';'
        return res

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", options: Dict) -> None:
        verify_description_mode(mode)
        assert self.symbol
        # The caller of the domain added a desc_signature node.
        # Always enable multiline:
        signode['is_multiline'] = True
        # Put each line in a desc_signature_line node.
        mainDeclNode = addnodes.desc_signature_line()
        mainDeclNode.sphinx_line_type = 'declarator'
        mainDeclNode['add_permalink'] = not self.symbol.isRedeclaration
        signode += mainDeclNode

        if self.objectType == 'member':
            pass
        elif self.objectType == 'function':
            pass
        elif self.objectType == 'macro':
            pass
        elif self.objectType == 'struct':
            mainDeclNode += addnodes.desc_annotation('struct ', 'struct ')
        elif self.objectType == 'union':
            mainDeclNode += addnodes.desc_annotation('union ', 'union ')
        elif self.objectType == 'enum':
            mainDeclNode += addnodes.desc_annotation('enum ', 'enum ')
        elif self.objectType == 'enumerator':
            mainDeclNode += addnodes.desc_annotation('enumerator ', 'enumerator ')
        elif self.objectType == 'type':
            prefix = self.declaration.get_type_declaration_prefix()
            prefix += ' '
            mainDeclNode += addnodes.desc_annotation(prefix, prefix)
        else:
            assert False
        self.declaration.describe_signature(mainDeclNode, mode, env, self.symbol)
        if self.semicolon:
            mainDeclNode += nodes.Text(';')


class SymbolLookupResult:
    def __init__(self, symbols: Iterator["Symbol"], parentSymbol: "Symbol",
                 ident: ASTIdentifier) -> None:
        self.symbols = symbols
        self.parentSymbol = parentSymbol
        self.ident = ident


class LookupKey:
    def __init__(self, data: List[Tuple[ASTIdentifier, str]]) -> None:
        self.data = data

    def __str__(self) -> str:
        return '[{}]'.format(', '.join("({}, {})".format(
            ident, id_) for ident, id_ in self.data))


class Symbol:
    debug_indent = 0
    debug_indent_string = "  "
    debug_lookup = False
    debug_show_tree = False

    def __copy__(self):
        assert False  # shouldn't happen

    def __deepcopy__(self, memo):
        if self.parent:
            assert False  # shouldn't happen
        else:
            # the domain base class makes a copy of the initial data, which is fine
            return Symbol(None, None, None, None)

    @staticmethod
    def debug_print(*args: Any) -> None:
        print(Symbol.debug_indent_string * Symbol.debug_indent, end="")
        print(*args)

    def _assert_invariants(self) -> None:
        if not self.parent:
            # parent == None means global scope, so declaration means a parent
            assert not self.declaration
            assert not self.docname
        else:
            if self.declaration:
                assert self.docname

    def __setattr__(self, key: str, value: Any) -> None:
        if key == "children":
            assert False
        else:
            return super().__setattr__(key, value)

    def __init__(self, parent: "Symbol", ident: ASTIdentifier,
                 declaration: ASTDeclaration, docname: str) -> None:
        self.parent = parent
        # declarations in a single directive are linked together
        self.siblingAbove = None  # type: Symbol
        self.siblingBelow = None  # type: Symbol
        self.ident = ident
        self.declaration = declaration
        self.docname = docname
        self.isRedeclaration = False
        self._assert_invariants()

        # Remember to modify Symbol.remove if modifications to the parent change.
        self._children = []  # type: List[Symbol]
        self._anonChildren = []  # type: List[Symbol]
        # note: _children includes _anonChildren
        if self.parent:
            self.parent._children.append(self)
        if self.declaration:
            self.declaration.symbol = self

        # Do symbol addition after self._children has been initialised.
        self._add_function_params()

    def _fill_empty(self, declaration: ASTDeclaration, docname: str) -> None:
        self._assert_invariants()
        assert not self.declaration
        assert not self.docname
        assert declaration
        assert docname
        self.declaration = declaration
        self.declaration.symbol = self
        self.docname = docname
        self._assert_invariants()
        # and symbol addition should be done as well
        self._add_function_params()

    def _add_function_params(self) -> None:
        if Symbol.debug_lookup:
            Symbol.debug_indent += 1
            Symbol.debug_print("_add_function_params:")
        # Note: we may be called from _fill_empty, so the symbols we want
        #       to add may actually already be present (as empty symbols).

        # add symbols for function parameters, if any
        if self.declaration is not None and self.declaration.function_params is not None:
            for p in self.declaration.function_params:
                if p.arg is None:
                    continue
                nn = p.arg.name
                if nn is None:
                    continue
                # (comparing to the template params: we have checked that we are a declaration)
                decl = ASTDeclaration('functionParam', None, p)
                assert not nn.rooted
                assert len(nn.names) == 1
                self._add_symbols(nn, decl, self.docname)
        if Symbol.debug_lookup:
            Symbol.debug_indent -= 1

    def remove(self) -> None:
        if self.parent is None:
            return
        assert self in self.parent._children
        self.parent._children.remove(self)
        self.parent = None

    def clear_doc(self, docname: str) -> None:
        for sChild in self._children:
            sChild.clear_doc(docname)
            if sChild.declaration and sChild.docname == docname:
                sChild.declaration = None
                sChild.docname = None
                if sChild.siblingAbove is not None:
                    sChild.siblingAbove.siblingBelow = sChild.siblingBelow
                if sChild.siblingBelow is not None:
                    sChild.siblingBelow.siblingAbove = sChild.siblingAbove
                sChild.siblingAbove = None
                sChild.siblingBelow = None

    def get_all_symbols(self) -> Iterator["Symbol"]:
        yield self
        for sChild in self._children:
            for s in sChild.get_all_symbols():
                yield s

    @property
    def children_recurse_anon(self) -> Iterator["Symbol"]:
        for c in self._children:
            yield c
            if not c.ident.is_anon():
                continue
            yield from c.children_recurse_anon

    def get_lookup_key(self) -> "LookupKey":
        # The pickle files for the environment and for each document are distinct.
        # The environment has all the symbols, but the documents has xrefs that
        # must know their scope. A lookup key is essentially a specification of
        # how to find a specific symbol.
        symbols = []
        s = self
        while s.parent:
            symbols.append(s)
            s = s.parent
        symbols.reverse()
        key = []
        for s in symbols:
            if s.declaration is not None:
                # TODO: do we need the ID?
                key.append((s.ident, s.declaration.get_newest_id()))
            else:
                key.append((s.ident, None))
        return LookupKey(key)

    def get_full_nested_name(self) -> ASTNestedName:
        symbols = []
        s = self
        while s.parent:
            symbols.append(s)
            s = s.parent
        symbols.reverse()
        names = []
        for s in symbols:
            names.append(s.ident)
        return ASTNestedName(names, rooted=False)

    def _find_first_named_symbol(self, ident: ASTIdentifier,
                                 matchSelf: bool, recurseInAnon: bool) -> "Symbol":
        # TODO: further simplification from C++ to C
        if Symbol.debug_lookup:
            Symbol.debug_print("_find_first_named_symbol ->")
        res = self._find_named_symbols(ident, matchSelf, recurseInAnon,
                                       searchInSiblings=False)
        try:
            return next(res)
        except StopIteration:
            return None

    def _find_named_symbols(self, ident: ASTIdentifier,
                            matchSelf: bool, recurseInAnon: bool,
                            searchInSiblings: bool) -> Iterator["Symbol"]:
        # TODO: further simplification from C++ to C
        if Symbol.debug_lookup:
            Symbol.debug_indent += 1
            Symbol.debug_print("_find_named_symbols:")
            Symbol.debug_indent += 1
            Symbol.debug_print("self:")
            print(self.to_string(Symbol.debug_indent + 1), end="")
            Symbol.debug_print("ident:            ", ident)
            Symbol.debug_print("matchSelf:        ", matchSelf)
            Symbol.debug_print("recurseInAnon:    ", recurseInAnon)
            Symbol.debug_print("searchInSiblings: ", searchInSiblings)

        def candidates() -> Generator["Symbol", None, None]:
            s = self
            if Symbol.debug_lookup:
                Symbol.debug_print("searching in self:")
                print(s.to_string(Symbol.debug_indent + 1), end="")
            while True:
                if matchSelf:
                    yield s
                if recurseInAnon:
                    yield from s.children_recurse_anon
                else:
                    yield from s._children

                if s.siblingAbove is None:
                    break
                s = s.siblingAbove
                if Symbol.debug_lookup:
                    Symbol.debug_print("searching in sibling:")
                    print(s.to_string(Symbol.debug_indent + 1), end="")

        for s in candidates():
            if Symbol.debug_lookup:
                Symbol.debug_print("candidate:")
                print(s.to_string(Symbol.debug_indent + 1), end="")
            if s.ident == ident:
                if Symbol.debug_lookup:
                    Symbol.debug_indent += 1
                    Symbol.debug_print("matches")
                    Symbol.debug_indent -= 3
                yield s
                if Symbol.debug_lookup:
                    Symbol.debug_indent += 2
        if Symbol.debug_lookup:
            Symbol.debug_indent -= 2
```
### 10 - sphinx/writers/text.py:

Start line: 1045, End line: 1154

```python
class TextTranslator(SphinxTranslator):

    def depart_emphasis(self, node: Element) -> None:
        self.add_text('*')

    def visit_literal_emphasis(self, node: Element) -> None:
        self.add_text('*')

    def depart_literal_emphasis(self, node: Element) -> None:
        self.add_text('*')

    def visit_strong(self, node: Element) -> None:
        self.add_text('**')

    def depart_strong(self, node: Element) -> None:
        self.add_text('**')

    def visit_literal_strong(self, node: Element) -> None:
        self.add_text('**')

    def depart_literal_strong(self, node: Element) -> None:
        self.add_text('**')

    def visit_abbreviation(self, node: Element) -> None:
        self.add_text('')

    def depart_abbreviation(self, node: Element) -> None:
        if node.hasattr('explanation'):
            self.add_text(' (%s)' % node['explanation'])

    def visit_manpage(self, node: Element) -> None:
        return self.visit_literal_emphasis(node)

    def depart_manpage(self, node: Element) -> None:
        return self.depart_literal_emphasis(node)

    def visit_title_reference(self, node: Element) -> None:
        self.add_text('*')

    def depart_title_reference(self, node: Element) -> None:
        self.add_text('*')

    def visit_literal(self, node: Element) -> None:
        self.add_text('"')

    def depart_literal(self, node: Element) -> None:
        self.add_text('"')

    def visit_subscript(self, node: Element) -> None:
        self.add_text('_')

    def depart_subscript(self, node: Element) -> None:
        pass

    def visit_superscript(self, node: Element) -> None:
        self.add_text('^')

    def depart_superscript(self, node: Element) -> None:
        pass

    def visit_footnote_reference(self, node: Element) -> None:
        self.add_text('[%s]' % node.astext())
        raise nodes.SkipNode

    def visit_citation_reference(self, node: Element) -> None:
        self.add_text('[%s]' % node.astext())
        raise nodes.SkipNode

    def visit_Text(self, node: Text) -> None:
        self.add_text(node.astext())

    def depart_Text(self, node: Text) -> None:
        pass

    def visit_generated(self, node: Element) -> None:
        pass

    def depart_generated(self, node: Element) -> None:
        pass

    def visit_inline(self, node: Element) -> None:
        if 'xref' in node['classes'] or 'term' in node['classes']:
            self.add_text('*')

    def depart_inline(self, node: Element) -> None:
        if 'xref' in node['classes'] or 'term' in node['classes']:
            self.add_text('*')

    def visit_container(self, node: Element) -> None:
        pass

    def depart_container(self, node: Element) -> None:
        pass

    def visit_problematic(self, node: Element) -> None:
        self.add_text('>>')

    def depart_problematic(self, node: Element) -> None:
        self.add_text('<<')

    def visit_system_message(self, node: Element) -> None:
        self.new_state(0)
        self.add_text('<SYSTEM MESSAGE: %s>' % node.astext())
        self.end_state()
        raise nodes.SkipNode

    def visit_comment(self, node: Element) -> None:
        raise nodes.SkipNode

    def visit_meta(self, node: Element) -> None:
        # only valid for HTML
        raise nodes.SkipNode
```
### 26 - sphinx/environment/adapters/indexentries.py:

Start line: 116, End line: 157

```python
class IndexEntries:

    def create_index(self, builder: Builder, group_entries: bool = True,
                     _fixre: Pattern = re.compile(r'(.*) ([(][^()]*[)])')
                     ) -> List[Tuple[str, List[Tuple[str, Any]]]]:
        # ... other code
        newlist = sorted(new.items(), key=keyfunc)

        if group_entries:
            # fixup entries: transform
            #   func() (in module foo)
            #   func() (in module bar)
            # into
            #   func()
            #     (in module foo)
            #     (in module bar)
            oldkey = ''
            oldsubitems = None  # type: Dict[str, List]
            i = 0
            while i < len(newlist):
                key, (targets, subitems, _key) = newlist[i]
                # cannot move if it has subitems; structure gets too complex
                if not subitems:
                    m = _fixre.match(key)
                    if m:
                        if oldkey == m.group(1):
                            # prefixes match: add entry as subitem of the
                            # previous entry
                            oldsubitems.setdefault(m.group(2), [[], {}, _key])[0].\
                                extend(targets)
                            del newlist[i]
                            continue
                        oldkey = m.group(1)
                    else:
                        oldkey = key
                oldsubitems = subitems
                i += 1

        # sort the sub-index entries
        def keyfunc2(entry: Tuple[str, List]) -> str:
            key = unicodedata.normalize('NFD', entry[0].lower())
            if key.startswith('\N{RIGHT-TO-LEFT MARK}'):
                key = key[1:]
            if key[0:1].isalpha() or key.startswith('_'):
                key = chr(127) + key
            return key

        # group the entries by letter
        # ... other code
```
### 44 - sphinx/environment/adapters/indexentries.py:

Start line: 103, End line: 115

```python
class IndexEntries:

    def create_index(self, builder: Builder, group_entries: bool = True,
                     _fixre: Pattern = re.compile(r'(.*) ([(][^()]*[)])')
                     ) -> List[Tuple[str, List[Tuple[str, Any]]]]:
        # ... other code
        def keyfunc(entry: Tuple[str, List]) -> Tuple[str, str]:
            key, (void, void, category_key) = entry
            if category_key:
                # using specified category key to sort
                key = category_key
            lckey = unicodedata.normalize('NFD', key.lower())
            if lckey.startswith('\N{RIGHT-TO-LEFT MARK}'):
                lckey = lckey[1:]
            if lckey[0:1].isalpha() or lckey.startswith('_'):
                lckey = chr(127) + lckey
            # ensure a determinstic order *within* letters by also sorting on
            # the entry itself
            return (lckey, entry[0])
        # ... other code
```
