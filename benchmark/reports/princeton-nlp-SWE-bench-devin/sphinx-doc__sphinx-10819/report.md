# sphinx-doc__sphinx-10819

| **sphinx-doc/sphinx** | `276f430b57957771f23355a6a1eb10a55899a677` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 4817 |
| **Any found context length** | 1052 |
| **Avg pos** | 36.0 |
| **Min pos** | 4 |
| **Max pos** | 14 |
| **Top file pos** | 1 |
| **Missing snippets** | 4 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sphinx/search/__init__.py b/sphinx/search/__init__.py
--- a/sphinx/search/__init__.py
+++ b/sphinx/search/__init__.py
@@ -14,6 +14,7 @@
 from sphinx import addnodes, package_dir
 from sphinx.deprecation import RemovedInSphinx70Warning
 from sphinx.environment import BuildEnvironment
+from sphinx.util import split_into
 
 
 class SearchLanguage:
@@ -242,6 +243,7 @@ def __init__(self, env: BuildEnvironment, lang: str, options: Dict, scoring: str
         # stemmed words in titles -> set(docname)
         self._title_mapping: Dict[str, Set[str]] = {}
         self._all_titles: Dict[str, List[Tuple[str, str]]] = {}  # docname -> all titles
+        self._index_entries: Dict[str, List[Tuple[str, str, str]]] = {}  # docname -> index entry
         self._stem_cache: Dict[str, str] = {}       # word -> stemmed word
         self._objtypes: Dict[Tuple[str, str], int] = {}     # objtype -> index
         # objtype index -> (domain, type, objname (localized))
@@ -380,10 +382,15 @@ def freeze(self) -> Dict[str, Any]:
             for title, titleid in titlelist:
                 alltitles.setdefault(title, []).append((fn2index[docname],  titleid))
 
+        index_entries: Dict[str, List[Tuple[int, str]]] = {}
+        for docname, entries in self._index_entries.items():
+            for entry, entry_id, main_entry in entries:
+                index_entries.setdefault(entry.lower(), []).append((fn2index[docname],  entry_id))
+
         return dict(docnames=docnames, filenames=filenames, titles=titles, terms=terms,
                     objects=objects, objtypes=objtypes, objnames=objnames,
                     titleterms=title_terms, envversion=self.env.version,
-                    alltitles=alltitles)
+                    alltitles=alltitles, indexentries=index_entries)
 
     def label(self) -> str:
         return "%s (code: %s)" % (self.lang.language_name, self.lang.lang)
@@ -441,6 +448,38 @@ def stem(word: str) -> str:
             if _filter(stemmed_word) and not already_indexed:
                 self._mapping.setdefault(stemmed_word, set()).add(docname)
 
+        # find explicit entries within index directives
+        _index_entries: Set[Tuple[str, str, str]] = set()
+        for node in doctree.findall(addnodes.index):
+            for entry_type, value, tid, main, *index_key in node['entries']:
+                tid = tid or ''
+                try:
+                    if entry_type == 'single':
+                        try:
+                            entry, subentry = split_into(2, 'single', value)
+                        except ValueError:
+                            entry, = split_into(1, 'single', value)
+                            subentry = ''
+                        _index_entries.add((entry, tid, main))
+                        if subentry:
+                            _index_entries.add((subentry, tid, main))
+                    elif entry_type == 'pair':
+                        first, second = split_into(2, 'pair', value)
+                        _index_entries.add((first, tid, main))
+                        _index_entries.add((second, tid, main))
+                    elif entry_type == 'triple':
+                        first, second, third = split_into(3, 'triple', value)
+                        _index_entries.add((first, tid, main))
+                        _index_entries.add((second, tid, main))
+                        _index_entries.add((third, tid, main))
+                    elif entry_type in {'see', 'seealso'}:
+                        first, second = split_into(2, 'see', value)
+                        _index_entries.add((first, tid, main))
+                except ValueError:
+                    pass
+
+        self._index_entries[docname] = sorted(_index_entries)
+
     def context_for_searchtool(self) -> Dict[str, Any]:
         if self.lang.js_splitter_code:
             js_splitter_code = self.lang.js_splitter_code

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sphinx/search/__init__.py | 17 | 17 | 13 | 1 | 4551
| sphinx/search/__init__.py | 245 | 245 | 4 | 1 | 1052
| sphinx/search/__init__.py | 383 | 383 | 14 | 1 | 4817
| sphinx/search/__init__.py | 444 | 444 | 5 | 1 | 1172


## Problem Statement

```
Use the index directive as a source for search
**Is your feature request related to a problem? Please describe.**

My problem is the search engine is not good at finding terms that are indexed, for example:
- https://docs.python.org/3/search.html?q=for should find https://docs.python.org/3/reference/compound_stmts.html#index-6
- https://docs.python.org/3/search.html?q=argument should find https://docs.python.org/3/glossary.html#term-argument
- https://docs.python.org/3/search.html?q=as should find https://docs.python.org/3/reference/compound_stmts.html#index-11 and a few others
- https://docs.python.org/3/search.html?q=function should find https://docs.python.org/3/glossary.html#term-function
- https://docs.python.org/3/search.html?q=pyobject should find https://docs.python.org/3/c-api/structures.html#c.PyObject
...

**Describe the solution you'd like**
I think using the global index as a source for the search engine is a good way to enhance this and allow people to manually boost a search result by using the bang of the index directive. (`.. index:: ! Python`).

I can try to implement it, but I'm still not sure this is a good idea.

Generated Index can point to anchors, I'm not sure the current searchindex can hold them in its current state.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 sphinx/search/__init__.py** | 149 | 175| 184 | 184 | 4389 | 
| 2 | 2 sphinx/domains/index.py | 55 | 84| 233 | 417 | 5285 | 
| 3 | **2 sphinx/search/__init__.py** | 355 | 365| 149 | 566 | 5285 | 
| **-> 4 <-** | **2 sphinx/search/__init__.py** | 227 | 271| 486 | 1052 | 5285 | 
| **-> 5 <-** | **2 sphinx/search/__init__.py** | 444 | 455| 120 | 1172 | 5285 | 
| 6 | 2 sphinx/domains/index.py | 87 | 121| 279 | 1451 | 5285 | 
| 7 | 3 sphinx/environment/adapters/indexentries.py | 47 | 91| 561 | 2012 | 6918 | 
| 8 | 4 sphinx/domains/python.py | 1118 | 1187| 567 | 2579 | 19749 | 
| 9 | 4 sphinx/domains/python.py | 576 | 595| 245 | 2824 | 19749 | 
| 10 | 5 doc/development/tutorials/examples/recipe.py | 35 | 69| 290 | 3114 | 20859 | 
| 11 | **5 sphinx/search/__init__.py** | 409 | 442| 341 | 3455 | 20859 | 
| 12 | 6 sphinx/util/nodes.py | 369 | 405| 313 | 3768 | 26463 | 
| **-> 13 <-** | **6 sphinx/search/__init__.py** | 1 | 107| 783 | 4551 | 26463 | 
| **-> 14 <-** | **6 sphinx/search/__init__.py** | 367 | 386| 266 | 4817 | 26463 | 
| 15 | 6 sphinx/environment/adapters/indexentries.py | 110 | 151| 417 | 5234 | 26463 | 
| 16 | 7 sphinx/directives/__init__.py | 103 | 132| 220 | 5454 | 29219 | 
| 17 | 7 doc/development/tutorials/examples/recipe.py | 72 | 97| 211 | 5665 | 29219 | 
| 18 | 8 sphinx/domains/__init__.py | 63 | 94| 331 | 5996 | 32625 | 
| 19 | 8 sphinx/environment/adapters/indexentries.py | 92 | 109| 271 | 6267 | 32625 | 
| 20 | 8 sphinx/environment/adapters/indexentries.py | 152 | 171| 264 | 6531 | 32625 | 
| 21 | **8 sphinx/search/__init__.py** | 201 | 224| 291 | 6822 | 32625 | 
| 22 | 9 sphinx/search/en.py | 1 | 19| 106 | 6928 | 34301 | 
| 23 | 10 sphinx/search/fr.py | 1 | 198| 1032 | 7960 | 35333 | 
| 24 | 10 sphinx/domains/index.py | 1 | 21| 137 | 8097 | 35333 | 
| 25 | 11 sphinx/domains/std.py | 911 | 919| 122 | 8219 | 45560 | 
| 26 | 12 sphinx/directives/other.py | 1 | 31| 240 | 8459 | 48701 | 
| 27 | 12 sphinx/search/en.py | 21 | 219| 1570 | 10029 | 48701 | 
| 28 | **12 sphinx/search/__init__.py** | 110 | 146| 350 | 10379 | 48701 | 
| 29 | 12 sphinx/domains/python.py | 1080 | 1100| 248 | 10627 | 48701 | 
| 30 | 13 sphinx/search/zh.py | 216 | 247| 240 | 10867 | 50683 | 
| 31 | 14 sphinx/directives/code.py | 417 | 480| 642 | 11509 | 54551 | 
| 32 | 14 sphinx/domains/std.py | 329 | 440| 890 | 12399 | 54551 | 
| 33 | 15 sphinx/ext/intersphinx.py | 290 | 319| 270 | 12669 | 60418 | 
| 34 | 16 sphinx/domains/rst.py | 159 | 183| 274 | 12943 | 62993 | 
| 35 | 17 doc/conf.py | 91 | 148| 507 | 13450 | 64757 | 
| 36 | 17 sphinx/domains/python.py | 344 | 378| 373 | 13823 | 64757 | 
| 37 | 17 sphinx/domains/std.py | 299 | 326| 264 | 14087 | 64757 | 
| 38 | 17 sphinx/environment/adapters/indexentries.py | 1 | 45| 400 | 14487 | 64757 | 
| 39 | 18 sphinx/ext/extlinks.py | 39 | 84| 411 | 14898 | 66040 | 
| 40 | 18 sphinx/ext/intersphinx.py | 1 | 49| 385 | 15283 | 66040 | 
| 41 | 19 sphinx/transforms/__init__.py | 209 | 223| 156 | 15439 | 69316 | 
| 42 | 19 sphinx/directives/other.py | 79 | 151| 699 | 16138 | 69316 | 
| 43 | 19 sphinx/domains/python.py | 1059 | 1077| 140 | 16278 | 69316 | 
| 44 | 20 sphinx/search/ja.py | 508 | 535| 200 | 16478 | 81905 | 
| 45 | 20 sphinx/ext/intersphinx.py | 322 | 355| 353 | 16831 | 81905 | 
| 46 | 20 sphinx/domains/python.py | 380 | 417| 332 | 17163 | 81905 | 
| 47 | 20 sphinx/directives/other.py | 351 | 365| 136 | 17299 | 81905 | 
| 48 | 21 sphinx/search/de.py | 1 | 302| 1412 | 18711 | 83317 | 
| 49 | 21 sphinx/directives/code.py | 1 | 22| 150 | 18861 | 83317 | 
| 50 | 21 sphinx/domains/python.py | 1399 | 1432| 299 | 19160 | 83317 | 
| 51 | 21 sphinx/search/ja.py | 132 | 158| 755 | 19915 | 83317 | 
| 52 | 21 sphinx/search/zh.py | 249 | 261| 127 | 20042 | 83317 | 
| 53 | 22 sphinx/ext/linkcode.py | 1 | 72| 479 | 20521 | 83796 | 
| 54 | 22 sphinx/domains/std.py | 545 | 620| 748 | 21269 | 83796 | 
| 55 | 23 sphinx/registry.py | 220 | 229| 124 | 21393 | 88687 | 
| 56 | **23 sphinx/search/__init__.py** | 308 | 353| 436 | 21829 | 88687 | 
| 57 | 24 sphinx/environment/__init__.py | 82 | 134| 678 | 22507 | 94909 | 
| 58 | 24 sphinx/domains/__init__.py | 96 | 148| 347 | 22854 | 94909 | 
| 59 | 25 sphinx/domains/javascript.py | 111 | 136| 278 | 23132 | 99179 | 
| 60 | 26 sphinx/application.py | 53 | 115| 494 | 23626 | 111160 | 
| 61 | 26 doc/conf.py | 1 | 90| 769 | 24395 | 111160 | 
| 62 | 26 sphinx/domains/std.py | 1013 | 1032| 296 | 24691 | 111160 | 
| 63 | 27 sphinx/search/es.py | 1 | 362| 136 | 24827 | 112990 | 
| 64 | 27 sphinx/directives/code.py | 382 | 415| 288 | 25115 | 112990 | 
| 65 | 28 sphinx/builders/linkcheck.py | 497 | 532| 277 | 25392 | 117799 | 
| 66 | 29 sphinx/search/ru.py | 1 | 250| 136 | 25528 | 119980 | 
| 67 | 30 sphinx/writers/manpage.py | 313 | 411| 757 | 26285 | 123423 | 
| 68 | 31 sphinx/ext/viewcode.py | 298 | 319| 229 | 26514 | 126328 | 
| 69 | 32 sphinx/domains/cpp.py | 7941 | 8160| 2052 | 28566 | 195601 | 
| 70 | 32 sphinx/domains/std.py | 89 | 111| 211 | 28777 | 195601 | 


### Hint

```
A non-python example, searching for `ciphertext` in [pyca/cryptography](https://github.com/pyca/cryptography), the glossary `ciphertext` term is pretty low on the page and hidden among mentions of it within autodoc pages.
There's also a bpo issue on that subject: https://bugs.python.org/issue42106

It provides more examples from the Python documentation

> For instance if you search "append": https://docs.python.org/3/search.html?q=append
>
> On my end, neither list nor MutableSequence appear anywhere on this page, even scrolling down.
>
> Searching for "list": https://docs.python.org/3/search.html?q=list
>
> The documentation for the builtin "list" object also doesn't appear on the page. [Data Structures](https://docs.python.org/3/tutorial/datastructures.html?highlight=list) and [built-in types](https://docs.python.org/3/library/stdtypes.html?highlight=list) appear below the fold and the former is genuinely useful but also very easy to miss (I had not actually noticed it before going back in order to get the various links and try to extensively describe the issue). Neither actually links to the `list` builtin type though.
>
> Above the fold we find various "list" methods and classes from the stdlib as well as the PDB `list` comment, none of which seems like the best match for the query.


This would also be useful for the core CPython documentation.

A
Yeah, as discussed, this would be extremely useful for the CPython docs, as brought up in python/cpython#60075 , python/cpython#89541 , python/cpython#86272,  python/cpython#86272 and probably others, so it would be fantastic if you could implement this. To be honest, it confuses me why the search index wouldn't include the...index...to begin with, heh.

I was going to offer to help, but I'm sure you'd do a far better job than I would. If you do need something within my (limited) skillset, like testing this or reviewing docs, etc, let me know. Thanks!
```

## Patch

```diff
diff --git a/sphinx/search/__init__.py b/sphinx/search/__init__.py
--- a/sphinx/search/__init__.py
+++ b/sphinx/search/__init__.py
@@ -14,6 +14,7 @@
 from sphinx import addnodes, package_dir
 from sphinx.deprecation import RemovedInSphinx70Warning
 from sphinx.environment import BuildEnvironment
+from sphinx.util import split_into
 
 
 class SearchLanguage:
@@ -242,6 +243,7 @@ def __init__(self, env: BuildEnvironment, lang: str, options: Dict, scoring: str
         # stemmed words in titles -> set(docname)
         self._title_mapping: Dict[str, Set[str]] = {}
         self._all_titles: Dict[str, List[Tuple[str, str]]] = {}  # docname -> all titles
+        self._index_entries: Dict[str, List[Tuple[str, str, str]]] = {}  # docname -> index entry
         self._stem_cache: Dict[str, str] = {}       # word -> stemmed word
         self._objtypes: Dict[Tuple[str, str], int] = {}     # objtype -> index
         # objtype index -> (domain, type, objname (localized))
@@ -380,10 +382,15 @@ def freeze(self) -> Dict[str, Any]:
             for title, titleid in titlelist:
                 alltitles.setdefault(title, []).append((fn2index[docname],  titleid))
 
+        index_entries: Dict[str, List[Tuple[int, str]]] = {}
+        for docname, entries in self._index_entries.items():
+            for entry, entry_id, main_entry in entries:
+                index_entries.setdefault(entry.lower(), []).append((fn2index[docname],  entry_id))
+
         return dict(docnames=docnames, filenames=filenames, titles=titles, terms=terms,
                     objects=objects, objtypes=objtypes, objnames=objnames,
                     titleterms=title_terms, envversion=self.env.version,
-                    alltitles=alltitles)
+                    alltitles=alltitles, indexentries=index_entries)
 
     def label(self) -> str:
         return "%s (code: %s)" % (self.lang.language_name, self.lang.lang)
@@ -441,6 +448,38 @@ def stem(word: str) -> str:
             if _filter(stemmed_word) and not already_indexed:
                 self._mapping.setdefault(stemmed_word, set()).add(docname)
 
+        # find explicit entries within index directives
+        _index_entries: Set[Tuple[str, str, str]] = set()
+        for node in doctree.findall(addnodes.index):
+            for entry_type, value, tid, main, *index_key in node['entries']:
+                tid = tid or ''
+                try:
+                    if entry_type == 'single':
+                        try:
+                            entry, subentry = split_into(2, 'single', value)
+                        except ValueError:
+                            entry, = split_into(1, 'single', value)
+                            subentry = ''
+                        _index_entries.add((entry, tid, main))
+                        if subentry:
+                            _index_entries.add((subentry, tid, main))
+                    elif entry_type == 'pair':
+                        first, second = split_into(2, 'pair', value)
+                        _index_entries.add((first, tid, main))
+                        _index_entries.add((second, tid, main))
+                    elif entry_type == 'triple':
+                        first, second, third = split_into(3, 'triple', value)
+                        _index_entries.add((first, tid, main))
+                        _index_entries.add((second, tid, main))
+                        _index_entries.add((third, tid, main))
+                    elif entry_type in {'see', 'seealso'}:
+                        first, second = split_into(2, 'see', value)
+                        _index_entries.add((first, tid, main))
+                except ValueError:
+                    pass
+
+        self._index_entries[docname] = sorted(_index_entries)
+
     def context_for_searchtool(self) -> Dict[str, Any]:
         if self.lang.js_splitter_code:
             js_splitter_code = self.lang.js_splitter_code

```

## Test Patch

```diff
diff --git a/tests/test_search.py b/tests/test_search.py
--- a/tests/test_search.py
+++ b/tests/test_search.py
@@ -178,7 +178,8 @@ def test_IndexBuilder():
                   'test': [0, 1, 2, 3]},
         'titles': ('title1_1', 'title1_2', 'title2_1', 'title2_2'),
         'titleterms': {'section_titl': [0, 1, 2, 3]},
-        'alltitles': {'section_title': [(0, 'section-title'), (1, 'section-title'), (2, 'section-title'), (3, 'section-title')]}
+        'alltitles': {'section_title': [(0, 'section-title'), (1, 'section-title'), (2, 'section-title'), (3, 'section-title')]},
+        'indexentries': {},
     }
     assert index._objtypes == {('dummy1', 'objtype1'): 0, ('dummy2', 'objtype1'): 1}
     assert index._objnames == {0: ('dummy1', 'objtype1', 'objtype1'),
@@ -236,7 +237,8 @@ def test_IndexBuilder():
                   'test': [0, 1]},
         'titles': ('title1_2', 'title2_2'),
         'titleterms': {'section_titl': [0, 1]},
-        'alltitles': {'section_title': [(0, 'section-title'), (1, 'section-title')]}
+        'alltitles': {'section_title': [(0, 'section-title'), (1, 'section-title')]},
+        'indexentries': {},
     }
     assert index._objtypes == {('dummy1', 'objtype1'): 0, ('dummy2', 'objtype1'): 1}
     assert index._objnames == {0: ('dummy1', 'objtype1', 'objtype1'),

```


## Code snippets

### 1 - sphinx/search/__init__.py:

Start line: 149, End line: 175

```python
class _JavaScriptIndex:
    """
    The search index as JavaScript file that calls a function
    on the documentation search object to register the index.
    """

    PREFIX = 'Search.setIndex('
    SUFFIX = ')'

    def dumps(self, data: Any) -> str:
        return self.PREFIX + json.dumps(data) + self.SUFFIX

    def loads(self, s: str) -> Any:
        data = s[len(self.PREFIX):-len(self.SUFFIX)]
        if not data or not s.startswith(self.PREFIX) or not \
           s.endswith(self.SUFFIX):
            raise ValueError('invalid data')
        return json.loads(data)

    def dump(self, data: Any, f: IO) -> None:
        f.write(self.dumps(data))

    def load(self, f: IO) -> Any:
        return self.loads(f.read())


js_index = _JavaScriptIndex()
```
### 2 - sphinx/domains/index.py:

Start line: 55, End line: 84

```python
class IndexDirective(SphinxDirective):
    """
    Directive to add entries to the index.
    """
    has_content = False
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = True
    option_spec: OptionSpec = {
        'name': directives.unchanged,
    }

    def run(self) -> List[Node]:
        arguments = self.arguments[0].split('\n')

        if 'name' in self.options:
            targetname = self.options['name']
            targetnode = nodes.target('', '', names=[targetname])
        else:
            targetid = 'index-%s' % self.env.new_serialno('index')
            targetnode = nodes.target('', '', ids=[targetid])

        self.state.document.note_explicit_target(targetnode)
        indexnode = addnodes.index()
        indexnode['entries'] = []
        indexnode['inline'] = False
        self.set_source_info(indexnode)
        for entry in arguments:
            indexnode['entries'].extend(process_index_entry(entry, targetnode['ids'][0]))
        return [indexnode, targetnode]
```
### 3 - sphinx/search/__init__.py:

Start line: 355, End line: 365

```python
class IndexBuilder:

    def get_terms(self, fn2index: Dict) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        rvs: Tuple[Dict[str, List[str]], Dict[str, List[str]]] = ({}, {})
        for rv, mapping in zip(rvs, (self._mapping, self._title_mapping)):
            for k, v in mapping.items():
                if len(v) == 1:
                    fn, = v
                    if fn in fn2index:
                        rv[k] = fn2index[fn]
                else:
                    rv[k] = sorted([fn2index[fn] for fn in v if fn in fn2index])
        return rvs
```
### 4 - sphinx/search/__init__.py:

Start line: 227, End line: 271

```python
class IndexBuilder:
    """
    Helper class that creates a search index based on the doctrees
    passed to the `feed` method.
    """
    formats = {
        'json':     json,
        'pickle':   pickle
    }

    def __init__(self, env: BuildEnvironment, lang: str, options: Dict, scoring: str) -> None:
        self.env = env
        self._titles: Dict[str, str] = {}           # docname -> title
        self._filenames: Dict[str, str] = {}        # docname -> filename
        self._mapping: Dict[str, Set[str]] = {}     # stemmed word -> set(docname)
        # stemmed words in titles -> set(docname)
        self._title_mapping: Dict[str, Set[str]] = {}
        self._all_titles: Dict[str, List[Tuple[str, str]]] = {}  # docname -> all titles
        self._stem_cache: Dict[str, str] = {}       # word -> stemmed word
        self._objtypes: Dict[Tuple[str, str], int] = {}     # objtype -> index
        # objtype index -> (domain, type, objname (localized))
        self._objnames: Dict[int, Tuple[str, str, str]] = {}
        # add language-specific SearchLanguage instance
        lang_class = languages.get(lang)

        # fallback; try again with language-code
        if lang_class is None and '_' in lang:
            lang_class = languages.get(lang.split('_')[0])

        if lang_class is None:
            self.lang: SearchLanguage = SearchEnglish(options)
        elif isinstance(lang_class, str):
            module, classname = lang_class.rsplit('.', 1)
            lang_class: Type[SearchLanguage] = getattr(import_module(module), classname)  # type: ignore[no-redef]
            self.lang = lang_class(options)  # type: ignore[operator]
        else:
            # it's directly a class (e.g. added by app.add_search_language)
            self.lang = lang_class(options)

        if scoring:
            with open(scoring, 'rb') as fp:
                self.js_scorer_code = fp.read().decode()
        else:
            self.js_scorer_code = ''
        self.js_splitter_code = ""
```
### 5 - sphinx/search/__init__.py:

Start line: 444, End line: 455

```python
class IndexBuilder:

    def context_for_searchtool(self) -> Dict[str, Any]:
        if self.lang.js_splitter_code:
            js_splitter_code = self.lang.js_splitter_code
        else:
            js_splitter_code = self.js_splitter_code

        return {
            'search_language_stemming_code': self.get_js_stemmer_code(),
            'search_language_stop_words': json.dumps(sorted(self.lang.stopwords)),
            'search_scorer_tool': self.js_scorer_code,
            'search_word_splitter_code': js_splitter_code,
        }
```
### 6 - sphinx/domains/index.py:

Start line: 87, End line: 121

```python
class IndexRole(ReferenceRole):
    def run(self) -> Tuple[List[Node], List[system_message]]:
        target_id = 'index-%s' % self.env.new_serialno('index')
        if self.has_explicit_title:
            # if an explicit target is given, process it as a full entry
            title = self.title
            entries = process_index_entry(self.target, target_id)
        else:
            # otherwise we just create a single entry
            if self.target.startswith('!'):
                title = self.title[1:]
                entries = [('single', self.target[1:], target_id, 'main', None)]
            else:
                title = self.title
                entries = [('single', self.target, target_id, '', None)]

        index = addnodes.index(entries=entries)
        target = nodes.target('', '', ids=[target_id])
        text = nodes.Text(title)
        self.set_source_info(index)
        return [index, target, text], []


def setup(app: "Sphinx") -> Dict[str, Any]:
    app.add_domain(IndexDomain)
    app.add_directive('index', IndexDirective)
    app.add_role('index', IndexRole())

    return {
        'version': 'builtin',
        'env_version': 1,
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
```
### 7 - sphinx/environment/adapters/indexentries.py:

Start line: 47, End line: 91

```python
class IndexEntries:

    def create_index(self, builder: Builder, group_entries: bool = True,
                     _fixre: Pattern = re.compile(r'(.*) ([(][^()]*[)])')
                     ) -> List[Tuple[str, List[Tuple[str, Any]]]]:
        # ... other code

        domain = cast(IndexDomain, self.env.get_domain('index'))
        for fn, entries in domain.entries.items():
            # new entry types must be listed in directives/other.py!
            for type, value, tid, main, index_key in entries:  # noqa: B007
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

        # sort the index entries
        # ... other code
```
### 8 - sphinx/domains/python.py:

Start line: 1118, End line: 1187

```python
class PythonModuleIndex(Index):
    """
    Index subclass to provide the Python module index.
    """

    name = 'modindex'
    localname = _('Python Module Index')
    shortname = _('modules')

    def generate(self, docnames: Iterable[str] = None
                 ) -> Tuple[List[Tuple[str, List[IndexEntry]]], bool]:
        content: Dict[str, List[IndexEntry]] = {}
        # list of prefixes to ignore
        ignores: List[str] = self.domain.env.config['modindex_common_prefix']
        ignores = sorted(ignores, key=len, reverse=True)
        # list of all modules, sorted by module name
        modules = sorted(self.domain.data['modules'].items(),
                         key=lambda x: x[0].lower())
        # sort out collapsible modules
        prev_modname = ''
        num_toplevels = 0
        for modname, (docname, node_id, synopsis, platforms, deprecated) in modules:
            if docnames and docname not in docnames:
                continue

            for ignore in ignores:
                if modname.startswith(ignore):
                    modname = modname[len(ignore):]
                    stripped = ignore
                    break
            else:
                stripped = ''

            # we stripped the whole module name?
            if not modname:
                modname, stripped = stripped, ''

            entries = content.setdefault(modname[0].lower(), [])

            package = modname.split('.')[0]
            if package != modname:
                # it's a submodule
                if prev_modname == package:
                    # first submodule - make parent a group head
                    if entries:
                        last = entries[-1]
                        entries[-1] = IndexEntry(last[0], 1, last[2], last[3],
                                                 last[4], last[5], last[6])
                elif not prev_modname.startswith(package):
                    # submodule without parent in list, add dummy entry
                    entries.append(IndexEntry(stripped + package, 1, '', '', '', '', ''))
                subtype = 2
            else:
                num_toplevels += 1
                subtype = 0

            qualifier = _('Deprecated') if deprecated else ''
            entries.append(IndexEntry(stripped + modname, subtype, docname,
                                      node_id, platforms, qualifier, synopsis))
            prev_modname = modname

        # apply heuristics when to collapse modindex at page load:
        # only collapse if number of toplevel modules is larger than
        # number of submodules
        collapse = len(modules) - num_toplevels < num_toplevels

        # sort by first letter
        sorted_content = sorted(content.items())

        return sorted_content, collapse
```
### 9 - sphinx/domains/python.py:

Start line: 576, End line: 595

```python
class PyObject(ObjectDescription[Tuple[str, str]]):

    def add_target_and_index(self, name_cls: Tuple[str, str], sig: str,
                             signode: desc_signature) -> None:
        modname = self.options.get('module', self.env.ref_context.get('py:module'))
        fullname = (modname + '.' if modname else '') + name_cls[0]
        node_id = make_id(self.env, self.state.document, '', fullname)
        signode['ids'].append(node_id)
        self.state.document.note_explicit_target(signode)

        domain = cast(PythonDomain, self.env.get_domain('py'))
        domain.note_object(fullname, self.objtype, node_id, location=signode)

        canonical_name = self.options.get('canonical')
        if canonical_name:
            domain.note_object(canonical_name, self.objtype, node_id, aliased=True,
                               location=signode)

        if 'noindexentry' not in self.options:
            indextext = self.get_index_text(modname, name_cls)
            if indextext:
                self.indexnode['entries'].append(('single', indextext, node_id, '', None))
```
### 10 - doc/development/tutorials/examples/recipe.py:

Start line: 35, End line: 69

```python
class IngredientIndex(Index):
    """A custom index that creates an ingredient matrix."""

    name = 'ingredient'
    localname = 'Ingredient Index'
    shortname = 'Ingredient'

    def generate(self, docnames=None):
        content = defaultdict(list)

        recipes = {name: (dispname, typ, docname, anchor)
                   for name, dispname, typ, docname, anchor, _
                   in self.domain.get_objects()}
        recipe_ingredients = self.domain.data['recipe_ingredients']
        ingredient_recipes = defaultdict(list)

        # flip from recipe_ingredients to ingredient_recipes
        for recipe_name, ingredients in recipe_ingredients.items():
            for ingredient in ingredients:
                ingredient_recipes[ingredient].append(recipe_name)

        # convert the mapping of ingredient to recipes to produce the expected
        # output, shown below, using the ingredient name as a key to group
        #
        # name, subtype, docname, anchor, extra, qualifier, description
        for ingredient, recipe_names in ingredient_recipes.items():
            for recipe_name in recipe_names:
                dispname, typ, docname, anchor = recipes[recipe_name]
                content[ingredient].append(
                    (dispname, 0, docname, anchor, docname, '', typ))

        # convert the dict to the sorted list of tuples expected
        content = sorted(content.items())

        return content, True
```
### 11 - sphinx/search/__init__.py:

Start line: 409, End line: 442

```python
class IndexBuilder:

    def feed(self, docname: str, filename: str, title: str, doctree: nodes.document) -> None:
        """Feed a doctree to the index."""
        self._titles[docname] = title
        self._filenames[docname] = filename

        visitor = WordCollector(doctree, self.lang)
        doctree.walk(visitor)

        # memoize self.lang.stem
        def stem(word: str) -> str:
            try:
                return self._stem_cache[word]
            except KeyError:
                self._stem_cache[word] = self.lang.stem(word).lower()
                return self._stem_cache[word]
        _filter = self.lang.word_filter

        self._all_titles[docname] = visitor.found_titles

        for word in visitor.found_title_words:
            stemmed_word = stem(word)
            if _filter(stemmed_word):
                self._title_mapping.setdefault(stemmed_word, set()).add(docname)
            elif _filter(word): # stemmer must not remove words from search index
                self._title_mapping.setdefault(word, set()).add(docname)

        for word in visitor.found_words:
            stemmed_word = stem(word)
            # again, stemmer must not remove words from search index
            if not _filter(stemmed_word) and _filter(word):
                stemmed_word = word
            already_indexed = docname in self._title_mapping.get(stemmed_word, set())
            if _filter(stemmed_word) and not already_indexed:
                self._mapping.setdefault(stemmed_word, set()).add(docname)
```
### 13 - sphinx/search/__init__.py:

Start line: 1, End line: 107

```python
"""Create a full-text search index for offline search."""
import html
import json
import pickle
import re
import warnings
from importlib import import_module
from os import path
from typing import IO, Any, Dict, Iterable, List, Optional, Set, Tuple, Type, Union

from docutils import nodes
from docutils.nodes import Element, Node

from sphinx import addnodes, package_dir
from sphinx.deprecation import RemovedInSphinx70Warning
from sphinx.environment import BuildEnvironment


class SearchLanguage:
    """
    This class is the base class for search natural language preprocessors.  If
    you want to add support for a new language, you should override the methods
    of this class.

    You should override `lang` class property too (e.g. 'en', 'fr' and so on).

    .. attribute:: stopwords

       This is a set of stop words of the target language.  Default `stopwords`
       is empty.  This word is used for building index and embedded in JS.

    .. attribute:: js_splitter_code

       Return splitter function of JavaScript version.  The function should be
       named as ``splitQuery``.  And it should take a string and return list of
       strings.

       .. versionadded:: 3.0

    .. attribute:: js_stemmer_code

       Return stemmer class of JavaScript version.  This class' name should be
       ``Stemmer`` and this class must have ``stemWord`` method.  This string is
       embedded as-is in searchtools.js.

       This class is used to preprocess search word which Sphinx HTML readers
       type, before searching index. Default implementation does nothing.
    """
    lang: Optional[str] = None
    language_name: Optional[str] = None
    stopwords: Set[str] = set()
    js_splitter_code: str = ""
    js_stemmer_rawcode: Optional[str] = None
    js_stemmer_code = """
/**
 * Dummy stemmer for languages without stemming rules.
 */
var Stemmer = function() {
  this.stemWord = function(w) {
    return w;
  }
}
"""

    _word_re = re.compile(r'(?u)\w+')

    def __init__(self, options: Dict) -> None:
        self.options = options
        self.init(options)

    def init(self, options: Dict) -> None:
        """
        Initialize the class with the options the user has given.
        """

    def split(self, input: str) -> List[str]:
        """
        This method splits a sentence into words.  Default splitter splits input
        at white spaces, which should be enough for most languages except CJK
        languages.
        """
        return self._word_re.findall(input)

    def stem(self, word: str) -> str:
        """
        This method implements stemming algorithm of the Python version.

        Default implementation does nothing.  You should implement this if the
        language has any stemming rules.

        This class is used to preprocess search words before registering them in
        the search index.  The stemming of the Python version and the JS version
        (given in the js_stemmer_code attribute) must be compatible.
        """
        return word

    def word_filter(self, word: str) -> bool:
        """
        Return true if the target word should be registered in the search index.
        This method is called after stemming.
        """
        return (
            len(word) == 0 or not (
                ((len(word) < 3) and (12353 < ord(word[0]) < 12436)) or
                (ord(word[0]) < 256 and (
                    word in self.stopwords
                ))))
```
### 14 - sphinx/search/__init__.py:

Start line: 367, End line: 386

```python
class IndexBuilder:

    def freeze(self) -> Dict[str, Any]:
        """Create a usable data structure for serializing."""
        docnames, titles = zip(*sorted(self._titles.items()))
        filenames = [self._filenames.get(docname) for docname in docnames]
        fn2index = {f: i for (i, f) in enumerate(docnames)}
        terms, title_terms = self.get_terms(fn2index)

        objects = self.get_objects(fn2index)  # populates _objtypes
        objtypes = {v: k[0] + ':' + k[1] for (k, v) in self._objtypes.items()}
        objnames = self._objnames

        alltitles: Dict[str, List[Tuple[int, str]]] = {}
        for docname, titlelist in self._all_titles.items():
            for title, titleid in titlelist:
                alltitles.setdefault(title, []).append((fn2index[docname],  titleid))

        return dict(docnames=docnames, filenames=filenames, titles=titles, terms=terms,
                    objects=objects, objtypes=objtypes, objnames=objnames,
                    titleterms=title_terms, envversion=self.env.version,
                    alltitles=alltitles)
```
### 21 - sphinx/search/__init__.py:

Start line: 201, End line: 224

```python
class WordCollector(nodes.NodeVisitor):

    def dispatch_visit(self, node: Node) -> None:
        if isinstance(node, nodes.comment):
            raise nodes.SkipNode
        elif isinstance(node, nodes.raw):
            if 'html' in node.get('format', '').split():
                # Some people might put content in raw HTML that should be searched,
                # so we just amateurishly strip HTML tags and index the remaining
                # content
                nodetext = re.sub(r'(?is)<style.*?</style>', '', node.astext())
                nodetext = re.sub(r'(?is)<script.*?</script>', '', nodetext)
                nodetext = re.sub(r'<[^<]+?>', '', nodetext)
                self.found_words.extend(self.lang.split(nodetext))
            raise nodes.SkipNode
        elif isinstance(node, nodes.Text):
            self.found_words.extend(self.lang.split(node.astext()))
        elif isinstance(node, nodes.title):
            title = node.astext()
            ids = node.parent['ids']
            self.found_titles.append((title, ids[0] if ids else None))
            self.found_title_words.extend(self.lang.split(title))
        elif isinstance(node, Element) and self.is_meta_keywords(node):
            keywords = node['content']
            keywords = [keyword.strip() for keyword in keywords.split(',')]
            self.found_words.extend(keywords)
```
### 28 - sphinx/search/__init__.py:

Start line: 110, End line: 146

```python
# SearchEnglish imported after SearchLanguage is defined due to circular import
from sphinx.search.en import SearchEnglish


def parse_stop_word(source: str) -> Set[str]:
    """
    Parse snowball style word list like this:

    * http://snowball.tartarus.org/algorithms/finnish/stop.txt
    """
    result: Set[str] = set()
    for line in source.splitlines():
        line = line.split('|')[0]  # remove comment
        result.update(line.split())
    return result


# maps language name to module.class or directly a class
languages: Dict[str, Union[str, Type[SearchLanguage]]] = {
    'da': 'sphinx.search.da.SearchDanish',
    'de': 'sphinx.search.de.SearchGerman',
    'en': SearchEnglish,
    'es': 'sphinx.search.es.SearchSpanish',
    'fi': 'sphinx.search.fi.SearchFinnish',
    'fr': 'sphinx.search.fr.SearchFrench',
    'hu': 'sphinx.search.hu.SearchHungarian',
    'it': 'sphinx.search.it.SearchItalian',
    'ja': 'sphinx.search.ja.SearchJapanese',
    'nl': 'sphinx.search.nl.SearchDutch',
    'no': 'sphinx.search.no.SearchNorwegian',
    'pt': 'sphinx.search.pt.SearchPortuguese',
    'ro': 'sphinx.search.ro.SearchRomanian',
    'ru': 'sphinx.search.ru.SearchRussian',
    'sv': 'sphinx.search.sv.SearchSwedish',
    'tr': 'sphinx.search.tr.SearchTurkish',
    'zh': 'sphinx.search.zh.SearchChinese',
}
```
### 56 - sphinx/search/__init__.py:

Start line: 308, End line: 353

```python
class IndexBuilder:

    def dump(self, stream: IO, format: Any) -> None:
        """Dump the frozen index to a stream."""
        if format == "jsdump":
            warnings.warn("format=jsdump is deprecated, use json instead",
                          RemovedInSphinx70Warning, stacklevel=2)
            format = self.formats["json"]
        elif isinstance(format, str):
            format = self.formats[format]
        format.dump(self.freeze(), stream)

    def get_objects(self, fn2index: Dict[str, int]
                    ) -> Dict[str, List[Tuple[int, int, int, str, str]]]:
        rv: Dict[str, List[Tuple[int, int, int, str, str]]] = {}
        otypes = self._objtypes
        onames = self._objnames
        for domainname, domain in sorted(self.env.domains.items()):
            for fullname, dispname, type, docname, anchor, prio in \
                    sorted(domain.get_objects()):
                if docname not in fn2index:
                    continue
                if prio < 0:
                    continue
                fullname = html.escape(fullname)
                dispname = html.escape(dispname)
                prefix, _, name = dispname.rpartition('.')
                plist = rv.setdefault(prefix, [])
                try:
                    typeindex = otypes[domainname, type]
                except KeyError:
                    typeindex = len(otypes)
                    otypes[domainname, type] = typeindex
                    otype = domain.object_types.get(type)
                    if otype:
                        # use str() to fire translation proxies
                        onames[typeindex] = (domainname, type,
                                             str(domain.get_type_name(otype)))
                    else:
                        onames[typeindex] = (domainname, type, type)
                if anchor == fullname:
                    shortanchor = ''
                elif anchor == type + '-' + fullname:
                    shortanchor = '-'
                else:
                    shortanchor = anchor
                plist.append((fn2index[docname], typeindex, prio, shortanchor, name))
        return rv
```
