# sphinx-doc__sphinx-7440

| **sphinx-doc/sphinx** | `9bb204dcabe6ba0fc422bf4a45ad0c79c680d90b` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 29960 |
| **Any found context length** | 419 |
| **Avg pos** | 81.0 |
| **Min pos** | 1 |
| **Max pos** | 80 |
| **Top file pos** | 1 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sphinx/domains/std.py b/sphinx/domains/std.py
--- a/sphinx/domains/std.py
+++ b/sphinx/domains/std.py
@@ -305,7 +305,7 @@ def make_glossary_term(env: "BuildEnvironment", textnodes: Iterable[Node], index
         term['ids'].append(node_id)
 
     std = cast(StandardDomain, env.get_domain('std'))
-    std.note_object('term', termtext.lower(), node_id, location=term)
+    std.note_object('term', termtext, node_id, location=term)
 
     # add an index entry too
     indexnode = addnodes.index()
@@ -565,7 +565,7 @@ class StandardDomain(Domain):
         # links to tokens in grammar productions
         'token':   TokenXRefRole(),
         # links to terms in glossary
-        'term':    XRefRole(lowercase=True, innernodeclass=nodes.inline,
+        'term':    XRefRole(innernodeclass=nodes.inline,
                             warn_dangling=True),
         # links to headings or arbitrary labels
         'ref':     XRefRole(lowercase=True, innernodeclass=nodes.inline,

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sphinx/domains/std.py | 308 | 308 | 1 | 1 | 419
| sphinx/domains/std.py | 568 | 568 | 80 | 1 | 29960


## Problem Statement

```
glossary duplicate term with a different case
**Describe the bug**
\`\`\`
Warning, treated as error:
doc/glossary.rst:243:duplicate term description of mysql, other instance in glossary
\`\`\`

**To Reproduce**
Steps to reproduce the behavior:
[.travis.yml#L168](https://github.com/phpmyadmin/phpmyadmin/blob/f7cc383674b7099190771b1db510c62bfbbf89a7/.travis.yml#L168)
\`\`\`
$ git clone --depth 1 https://github.com/phpmyadmin/phpmyadmin.git
$ cd doc
$ pip install 'Sphinx'
$ make html
\`\`\`

**Expected behavior**
MySQL != mysql term right ?

**Your project**
https://github.com/phpmyadmin/phpmyadmin/blame/master/doc/glossary.rst#L234


**Environment info**
- OS: Unix
- Python version: 3.6
- Sphinx version: 3.0.0

**Additional context**
Did occur some hours ago, maybe you just released the version

- https://travis-ci.org/github/williamdes/phpmyadmintest/jobs/671352365#L328



```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 sphinx/domains/std.py** | 276 | 316| 419 | 419 | 9952 | 
| 2 | **1 sphinx/domains/std.py** | 319 | 437| 942 | 1361 | 9952 | 
| 3 | 2 sphinx/transforms/post_transforms/__init__.py | 153 | 176| 279 | 1640 | 11937 | 
| 4 | 3 sphinx/errors.py | 38 | 67| 213 | 1853 | 12672 | 
| 5 | 4 sphinx/cmd/quickstart.py | 138 | 159| 168 | 2021 | 18216 | 
| 6 | 4 sphinx/errors.py | 70 | 126| 290 | 2311 | 18216 | 
| 7 | 5 sphinx/writers/texinfo.py | 968 | 1085| 846 | 3157 | 30390 | 
| 8 | 6 sphinx/ext/todo.py | 275 | 301| 261 | 3418 | 33086 | 
| 9 | **6 sphinx/domains/std.py** | 11 | 48| 323 | 3741 | 33086 | 
| 10 | 7 sphinx/writers/text.py | 767 | 874| 830 | 4571 | 42040 | 
| 11 | 8 sphinx/domains/python.py | 1303 | 1341| 304 | 4875 | 53489 | 
| 12 | 9 sphinx/ext/intersphinx.py | 261 | 339| 865 | 5740 | 57162 | 
| 13 | 10 sphinx/transforms/references.py | 11 | 68| 362 | 6102 | 57576 | 
| 14 | 11 sphinx/util/cfamily.py | 11 | 81| 732 | 6834 | 59586 | 
| 15 | 12 sphinx/domains/rst.py | 230 | 236| 113 | 6947 | 62056 | 
| 16 | 12 sphinx/ext/todo.py | 14 | 44| 207 | 7154 | 62056 | 
| 17 | 13 sphinx/roles.py | 92 | 106| 156 | 7310 | 67698 | 
| 18 | 13 sphinx/writers/texinfo.py | 862 | 966| 788 | 8098 | 67698 | 
| 19 | 14 sphinx/config.py | 458 | 477| 223 | 8321 | 72083 | 
| 20 | 14 sphinx/cmd/quickstart.py | 11 | 119| 772 | 9093 | 72083 | 
| 21 | 15 sphinx/util/nodes.py | 125 | 179| 683 | 9776 | 77480 | 
| 22 | 16 sphinx/environment/__init__.py | 11 | 82| 489 | 10265 | 83303 | 
| 23 | 17 sphinx/transforms/__init__.py | 251 | 271| 192 | 10457 | 86450 | 
| 24 | 18 sphinx/ext/viewcode.py | 121 | 139| 196 | 10653 | 88688 | 
| 25 | 19 sphinx/writers/manpage.py | 183 | 251| 538 | 11191 | 92199 | 
| 26 | 19 sphinx/domains/python.py | 11 | 68| 446 | 11637 | 92199 | 
| 27 | 20 sphinx/domains/__init__.py | 12 | 30| 140 | 11777 | 95710 | 
| 28 | 20 sphinx/domains/python.py | 221 | 239| 214 | 11991 | 95710 | 
| 29 | 21 doc/conf.py | 1 | 80| 767 | 12758 | 97246 | 
| 30 | 21 sphinx/config.py | 387 | 439| 474 | 13232 | 97246 | 
| 31 | 22 sphinx/util/__init__.py | 11 | 75| 505 | 13737 | 103023 | 
| 32 | 23 sphinx/writers/html.py | 354 | 409| 472 | 14209 | 110389 | 
| 33 | 24 sphinx/locale/__init__.py | 239 | 267| 232 | 14441 | 112442 | 
| 34 | 24 sphinx/util/cfamily.py | 115 | 130| 130 | 14571 | 112442 | 
| 35 | 25 sphinx/ext/autosummary/__init__.py | 55 | 106| 369 | 14940 | 118842 | 
| 36 | 26 sphinx/ext/graphviz.py | 12 | 44| 236 | 15176 | 122435 | 
| 37 | 27 sphinx/__init__.py | 14 | 64| 489 | 15665 | 122998 | 
| 38 | 27 sphinx/writers/html.py | 694 | 793| 803 | 16468 | 122998 | 
| 39 | 28 sphinx/domains/math.py | 11 | 40| 207 | 16675 | 124498 | 
| 40 | **28 sphinx/domains/std.py** | 877 | 885| 120 | 16795 | 124498 | 
| 41 | 28 sphinx/roles.py | 11 | 47| 284 | 17079 | 124498 | 
| 42 | 28 sphinx/domains/rst.py | 238 | 247| 126 | 17205 | 124498 | 
| 43 | 29 sphinx/deprecation.py | 11 | 31| 134 | 17339 | 125089 | 
| 44 | 30 sphinx/directives/__init__.py | 269 | 307| 273 | 17612 | 127584 | 
| 45 | 31 sphinx/directives/other.py | 9 | 40| 238 | 17850 | 130741 | 
| 46 | 31 sphinx/ext/todo.py | 80 | 104| 206 | 18056 | 130741 | 
| 47 | 31 sphinx/directives/other.py | 187 | 208| 141 | 18197 | 130741 | 
| 48 | 32 sphinx/ext/inheritance_diagram.py | 38 | 66| 212 | 18409 | 134594 | 
| 49 | 32 sphinx/writers/texinfo.py | 1450 | 1460| 122 | 18531 | 134594 | 
| 50 | 33 sphinx/directives/code.py | 9 | 31| 146 | 18677 | 138491 | 
| 51 | 34 sphinx/builders/dummy.py | 11 | 53| 223 | 18900 | 138767 | 
| 52 | 34 sphinx/writers/texinfo.py | 1462 | 1529| 493 | 19393 | 138767 | 
| 53 | 34 sphinx/writers/texinfo.py | 1292 | 1375| 673 | 20066 | 138767 | 
| 54 | 35 sphinx/search/__init__.py | 359 | 369| 151 | 20217 | 142775 | 
| 55 | 35 sphinx/writers/manpage.py | 313 | 411| 757 | 20974 | 142775 | 
| 56 | 36 sphinx/builders/devhelp.py | 13 | 39| 153 | 21127 | 143002 | 
| 57 | 36 sphinx/writers/text.py | 1045 | 1154| 793 | 21920 | 143002 | 
| 58 | 36 doc/conf.py | 81 | 140| 513 | 22433 | 143002 | 
| 59 | 37 utils/bump_version.py | 67 | 102| 201 | 22634 | 144365 | 
| 60 | 37 sphinx/domains/python.py | 966 | 986| 248 | 22882 | 144365 | 
| 61 | 37 sphinx/roles.py | 510 | 540| 294 | 23176 | 144365 | 
| 62 | 37 sphinx/util/nodes.py | 513 | 536| 202 | 23378 | 144365 | 
| 63 | 37 sphinx/ext/todo.py | 304 | 318| 143 | 23521 | 144365 | 
| 64 | 38 sphinx/ext/doctest.py | 240 | 267| 277 | 23798 | 149315 | 
| 65 | 39 sphinx/domains/citation.py | 11 | 31| 126 | 23924 | 150595 | 
| 66 | 39 sphinx/ext/autosummary/__init__.py | 718 | 753| 323 | 24247 | 150595 | 
| 67 | 40 sphinx/writers/latex.py | 1082 | 1152| 538 | 24785 | 170257 | 
| 68 | 40 sphinx/roles.py | 108 | 141| 338 | 25123 | 170257 | 
| 69 | 40 sphinx/transforms/__init__.py | 11 | 44| 226 | 25349 | 170257 | 
| 70 | 41 utils/checks.py | 33 | 109| 545 | 25894 | 171163 | 
| 71 | 41 sphinx/domains/python.py | 1142 | 1169| 326 | 26220 | 171163 | 
| 72 | 42 sphinx/util/logging.py | 11 | 56| 279 | 26499 | 174968 | 
| 73 | 42 sphinx/writers/latex.py | 14 | 73| 453 | 26952 | 174968 | 
| 74 | 43 sphinx/writers/html5.py | 631 | 717| 686 | 27638 | 181871 | 
| 75 | 43 sphinx/writers/latex.py | 1611 | 1670| 496 | 28134 | 181871 | 
| 76 | 43 sphinx/writers/texinfo.py | 1215 | 1271| 418 | 28552 | 181871 | 
| 77 | 44 sphinx/util/texescape.py | 143 | 170| 259 | 28811 | 183640 | 
| 78 | 45 sphinx/registry.py | 11 | 50| 307 | 29118 | 188134 | 
| 79 | 46 sphinx/project.py | 11 | 27| 116 | 29234 | 188946 | 
| **-> 80 <-** | **46 sphinx/domains/std.py** | 535 | 605| 726 | 29960 | 188946 | 
| 81 | 47 sphinx/ext/autodoc/typehints.py | 41 | 75| 299 | 30259 | 190042 | 
| 82 | 48 sphinx/cmd/build.py | 11 | 30| 132 | 30391 | 192705 | 
| 83 | 49 sphinx/transforms/i18n.py | 11 | 40| 219 | 30610 | 197286 | 


### Hint

```
Sorry for the inconvenience. Indeed, this must be a bug. I'll take a look this later.
```

## Patch

```diff
diff --git a/sphinx/domains/std.py b/sphinx/domains/std.py
--- a/sphinx/domains/std.py
+++ b/sphinx/domains/std.py
@@ -305,7 +305,7 @@ def make_glossary_term(env: "BuildEnvironment", textnodes: Iterable[Node], index
         term['ids'].append(node_id)
 
     std = cast(StandardDomain, env.get_domain('std'))
-    std.note_object('term', termtext.lower(), node_id, location=term)
+    std.note_object('term', termtext, node_id, location=term)
 
     # add an index entry too
     indexnode = addnodes.index()
@@ -565,7 +565,7 @@ class StandardDomain(Domain):
         # links to tokens in grammar productions
         'token':   TokenXRefRole(),
         # links to terms in glossary
-        'term':    XRefRole(lowercase=True, innernodeclass=nodes.inline,
+        'term':    XRefRole(innernodeclass=nodes.inline,
                             warn_dangling=True),
         # links to headings or arbitrary labels
         'ref':     XRefRole(lowercase=True, innernodeclass=nodes.inline,

```

## Test Patch

```diff
diff --git a/tests/test_domain_std.py b/tests/test_domain_std.py
--- a/tests/test_domain_std.py
+++ b/tests/test_domain_std.py
@@ -99,7 +99,7 @@ def test_glossary(app):
     text = (".. glossary::\n"
             "\n"
             "   term1\n"
-            "   term2\n"
+            "   TERM2\n"
             "       description\n"
             "\n"
             "   term3 : classifier\n"
@@ -114,7 +114,7 @@ def test_glossary(app):
     assert_node(doctree, (
         [glossary, definition_list, ([definition_list_item, ([term, ("term1",
                                                                      index)],
-                                                             [term, ("term2",
+                                                             [term, ("TERM2",
                                                                      index)],
                                                              definition)],
                                      [definition_list_item, ([term, ("term3",
@@ -127,7 +127,7 @@ def test_glossary(app):
     assert_node(doctree[0][0][0][0][1],
                 entries=[("single", "term1", "term-term1", "main", None)])
     assert_node(doctree[0][0][0][1][1],
-                entries=[("single", "term2", "term-term2", "main", None)])
+                entries=[("single", "TERM2", "term-TERM2", "main", None)])
     assert_node(doctree[0][0][0][2],
                 [definition, nodes.paragraph, "description"])
     assert_node(doctree[0][0][1][0][1],
@@ -143,7 +143,7 @@ def test_glossary(app):
     # index
     objects = list(app.env.get_domain("std").get_objects())
     assert ("term1", "term1", "term", "index", "term-term1", -1) in objects
-    assert ("term2", "term2", "term", "index", "term-term2", -1) in objects
+    assert ("TERM2", "TERM2", "term", "index", "term-TERM2", -1) in objects
     assert ("term3", "term3", "term", "index", "term-term3", -1) in objects
     assert ("term4", "term4", "term", "index", "term-term4", -1) in objects
 

```


## Code snippets

### 1 - sphinx/domains/std.py:

Start line: 276, End line: 316

```python
def make_glossary_term(env: "BuildEnvironment", textnodes: Iterable[Node], index_key: str,
                       source: str, lineno: int, node_id: str = None,
                       document: nodes.document = None) -> nodes.term:
    # get a text-only representation of the term and register it
    # as a cross-reference target
    term = nodes.term('', '', *textnodes)
    term.source = source
    term.line = lineno
    termtext = term.astext()

    if node_id:
        # node_id is given from outside (mainly i18n module), use it forcedly
        term['ids'].append(node_id)
    elif document:
        node_id = make_id(env, document, 'term', termtext)
        term['ids'].append(node_id)
        document.note_explicit_target(term)
    else:
        warnings.warn('make_glossary_term() expects document is passed as an argument.',
                      RemovedInSphinx40Warning)
        gloss_entries = env.temp_data.setdefault('gloss_entries', set())
        node_id = nodes.make_id('term-' + termtext)
        if node_id == 'term':
            # "term" is not good for node_id.  Generate it by sequence number instead.
            node_id = 'term-%d' % env.new_serialno('glossary')

        while node_id in gloss_entries:
            node_id = 'term-%d' % env.new_serialno('glossary')
        gloss_entries.add(node_id)
        term['ids'].append(node_id)

    std = cast(StandardDomain, env.get_domain('std'))
    std.note_object('term', termtext.lower(), node_id, location=term)

    # add an index entry too
    indexnode = addnodes.index()
    indexnode['entries'] = [('single', termtext, node_id, 'main', index_key)]
    indexnode.source, indexnode.line = term.source, term.line
    term.append(indexnode)

    return term
```
### 2 - sphinx/domains/std.py:

Start line: 319, End line: 437

```python
class Glossary(SphinxDirective):
    """
    Directive to create a glossary with cross-reference targets for :term:
    roles.
    """

    has_content = True
    required_arguments = 0
    optional_arguments = 0
    final_argument_whitespace = False
    option_spec = {
        'sorted': directives.flag,
    }

    def run(self) -> List[Node]:
        node = addnodes.glossary()
        node.document = self.state.document

        # This directive implements a custom format of the reST definition list
        # that allows multiple lines of terms before the definition.  This is
        # easy to parse since we know that the contents of the glossary *must
        # be* a definition list.

        # first, collect single entries
        entries = []  # type: List[Tuple[List[Tuple[str, str, int]], StringList]]
        in_definition = True
        in_comment = False
        was_empty = True
        messages = []  # type: List[Node]
        for line, (source, lineno) in zip(self.content, self.content.items):
            # empty line -> add to last definition
            if not line:
                if in_definition and entries:
                    entries[-1][1].append('', source, lineno)
                was_empty = True
                continue
            # unindented line -> a term
            if line and not line[0].isspace():
                # enable comments
                if line.startswith('.. '):
                    in_comment = True
                    continue
                else:
                    in_comment = False

                # first term of definition
                if in_definition:
                    if not was_empty:
                        messages.append(self.state.reporter.warning(
                            _('glossary term must be preceded by empty line'),
                            source=source, line=lineno))
                    entries.append(([(line, source, lineno)], StringList()))
                    in_definition = False
                # second term and following
                else:
                    if was_empty:
                        messages.append(self.state.reporter.warning(
                            _('glossary terms must not be separated by empty lines'),
                            source=source, line=lineno))
                    if entries:
                        entries[-1][0].append((line, source, lineno))
                    else:
                        messages.append(self.state.reporter.warning(
                            _('glossary seems to be misformatted, check indentation'),
                            source=source, line=lineno))
            elif in_comment:
                pass
            else:
                if not in_definition:
                    # first line of definition, determines indentation
                    in_definition = True
                    indent_len = len(line) - len(line.lstrip())
                if entries:
                    entries[-1][1].append(line[indent_len:], source, lineno)
                else:
                    messages.append(self.state.reporter.warning(
                        _('glossary seems to be misformatted, check indentation'),
                        source=source, line=lineno))
            was_empty = False

        # now, parse all the entries into a big definition list
        items = []
        for terms, definition in entries:
            termtexts = []          # type: List[str]
            termnodes = []          # type: List[Node]
            system_messages = []    # type: List[Node]
            for line, source, lineno in terms:
                parts = split_term_classifiers(line)
                # parse the term with inline markup
                # classifiers (parts[1:]) will not be shown on doctree
                textnodes, sysmsg = self.state.inline_text(parts[0], lineno)

                # use first classifier as a index key
                term = make_glossary_term(self.env, textnodes, parts[1], source, lineno,
                                          document=self.state.document)
                term.rawsource = line
                system_messages.extend(sysmsg)
                termtexts.append(term.astext())
                termnodes.append(term)

            termnodes.extend(system_messages)

            defnode = nodes.definition()
            if definition:
                self.state.nested_parse(definition, definition.items[0][1],
                                        defnode)
            termnodes.append(defnode)
            items.append((termtexts,
                          nodes.definition_list_item('', *termnodes)))

        if 'sorted' in self.options:
            items.sort(key=lambda x:
                       unicodedata.normalize('NFD', x[0][0].lower()))

        dlist = nodes.definition_list()
        dlist['classes'].append('glossary')
        dlist.extend(item[1] for item in items)
        node += dlist
        return messages + [node]
```
### 3 - sphinx/transforms/post_transforms/__init__.py:

Start line: 153, End line: 176

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
### 4 - sphinx/errors.py:

Start line: 38, End line: 67

```python
class SphinxWarning(SphinxError):
    """Warning, treated as error."""
    category = 'Warning, treated as error'


class ApplicationError(SphinxError):
    """Application initialization error."""
    category = 'Application error'


class ExtensionError(SphinxError):
    """Extension error."""
    category = 'Extension error'

    def __init__(self, message: str, orig_exc: Exception = None) -> None:
        super().__init__(message)
        self.message = message
        self.orig_exc = orig_exc

    def __repr__(self) -> str:
        if self.orig_exc:
            return '%s(%r, %r)' % (self.__class__.__name__,
                                   self.message, self.orig_exc)
        return '%s(%r)' % (self.__class__.__name__, self.message)

    def __str__(self) -> str:
        parent_str = super().__str__()
        if self.orig_exc:
            return '%s (exception: %s)' % (parent_str, self.orig_exc)
        return parent_str
```
### 5 - sphinx/cmd/quickstart.py:

Start line: 138, End line: 159

```python
def term_decode(text: Union[bytes, str]) -> str:
    warnings.warn('term_decode() is deprecated.',
                  RemovedInSphinx40Warning, stacklevel=2)

    if isinstance(text, str):
        return text

    # Use the known encoding, if possible
    if TERM_ENCODING:
        return text.decode(TERM_ENCODING)

    # If ascii is safe, use it with no warning
    if text.decode('ascii', 'replace').encode('ascii', 'replace') == text:
        return text.decode('ascii')

    print(turquoise(__('* Note: non-ASCII characters entered '
                       'and terminal encoding unknown -- assuming '
                       'UTF-8 or Latin-1.')))
    try:
        return text.decode()
    except UnicodeDecodeError:
        return text.decode('latin1')
```
### 6 - sphinx/errors.py:

Start line: 70, End line: 126

```python
class BuildEnvironmentError(SphinxError):
    """BuildEnvironment error."""
    category = 'BuildEnvironment error'


class ConfigError(SphinxError):
    """Configuration error."""
    category = 'Configuration error'


class DocumentError(SphinxError):
    """Document error."""
    category = 'Document error'


class ThemeError(SphinxError):
    """Theme error."""
    category = 'Theme error'


class VersionRequirementError(SphinxError):
    """Incompatible Sphinx version error."""
    category = 'Sphinx version error'


class SphinxParallelError(SphinxError):
    """Sphinx parallel build error."""

    category = 'Sphinx parallel build error'

    def __init__(self, message: str, traceback: Any) -> None:
        self.message = message
        self.traceback = traceback

    def __str__(self) -> str:
        return self.message


class PycodeError(Exception):
    """Pycode Python source code analyser error."""

    def __str__(self) -> str:
        res = self.args[0]
        if len(self.args) > 1:
            res += ' (exception was: %r)' % self.args[1]
        return res


class NoUri(Exception):
    """Raised by builder.get_relative_uri() if there is no URI available."""
    pass


class FiletypeNotFoundError(Exception):
    "Raised by get_filetype() if a filename matches no source suffix."
    pass
```
### 7 - sphinx/writers/texinfo.py:

Start line: 968, End line: 1085

```python
class TexinfoTranslator(SphinxTranslator):

    def visit_term(self, node: Element) -> None:
        for id in node.get('ids'):
            self.add_anchor(id, node)
        # anchors and indexes need to go in front
        for n in node[::]:
            if isinstance(n, (addnodes.index, nodes.target)):
                n.walkabout(self)
                node.remove(n)
        self.body.append('\n%s ' % self.at_item_x)
        self.at_item_x = '@itemx'

    def depart_term(self, node: Element) -> None:
        pass

    def visit_classifier(self, node: Element) -> None:
        self.body.append(' : ')

    def depart_classifier(self, node: Element) -> None:
        pass

    def visit_definition(self, node: Element) -> None:
        self.body.append('\n')

    def depart_definition(self, node: Element) -> None:
        pass

    # -- Tables

    def visit_table(self, node: Element) -> None:
        self.entry_sep = '@item'

    def depart_table(self, node: Element) -> None:
        self.body.append('\n@end multitable\n\n')

    def visit_tabular_col_spec(self, node: Element) -> None:
        pass

    def depart_tabular_col_spec(self, node: Element) -> None:
        pass

    def visit_colspec(self, node: Element) -> None:
        self.colwidths.append(node['colwidth'])
        if len(self.colwidths) != self.n_cols:
            return
        self.body.append('\n\n@multitable ')
        for i, n in enumerate(self.colwidths):
            self.body.append('{%s} ' % ('x' * (n + 2)))

    def depart_colspec(self, node: Element) -> None:
        pass

    def visit_tgroup(self, node: Element) -> None:
        self.colwidths = []
        self.n_cols = node['cols']

    def depart_tgroup(self, node: Element) -> None:
        pass

    def visit_thead(self, node: Element) -> None:
        self.entry_sep = '@headitem'

    def depart_thead(self, node: Element) -> None:
        pass

    def visit_tbody(self, node: Element) -> None:
        pass

    def depart_tbody(self, node: Element) -> None:
        pass

    def visit_row(self, node: Element) -> None:
        pass

    def depart_row(self, node: Element) -> None:
        self.entry_sep = '@item'

    def visit_entry(self, node: Element) -> None:
        self.body.append('\n%s\n' % self.entry_sep)
        self.entry_sep = '@tab'

    def depart_entry(self, node: Element) -> None:
        for i in range(node.get('morecols', 0)):
            self.body.append('\n@tab\n')

    # -- Field Lists

    def visit_field_list(self, node: Element) -> None:
        pass

    def depart_field_list(self, node: Element) -> None:
        pass

    def visit_field(self, node: Element) -> None:
        self.body.append('\n')

    def depart_field(self, node: Element) -> None:
        self.body.append('\n')

    def visit_field_name(self, node: Element) -> None:
        self.ensure_eol()
        self.body.append('@*')

    def depart_field_name(self, node: Element) -> None:
        self.body.append(': ')

    def visit_field_body(self, node: Element) -> None:
        pass

    def depart_field_body(self, node: Element) -> None:
        pass

    # -- Admonitions

    def visit_admonition(self, node: Element, name: str = '') -> None:
        if not name:
            title = cast(nodes.title, node[0])
            name = self.escape(title.astext())
        self.body.append('\n@cartouche\n@quotation %s ' % name)
```
### 8 - sphinx/ext/todo.py:

Start line: 275, End line: 301

```python
def purge_todos(app: Sphinx, env: BuildEnvironment, docname: str) -> None:
    warnings.warn('purge_todos() is deprecated.', RemovedInSphinx40Warning)
    if not hasattr(env, 'todo_all_todos'):
        return
    env.todo_all_todos = [todo for todo in env.todo_all_todos  # type: ignore
                          if todo['docname'] != docname]


def merge_info(app: Sphinx, env: BuildEnvironment, docnames: Iterable[str],
               other: BuildEnvironment) -> None:
    warnings.warn('merge_info() is deprecated.', RemovedInSphinx40Warning)
    if not hasattr(other, 'todo_all_todos'):
        return
    if not hasattr(env, 'todo_all_todos'):
        env.todo_all_todos = []  # type: ignore
    env.todo_all_todos.extend(other.todo_all_todos)  # type: ignore


def visit_todo_node(self: HTMLTranslator, node: todo_node) -> None:
    if self.config.todo_include_todos:
        self.visit_admonition(node)
    else:
        raise nodes.SkipNode


def depart_todo_node(self: HTMLTranslator, node: todo_node) -> None:
    self.depart_admonition(node)
```
### 9 - sphinx/domains/std.py:

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
option_desc_re = re.compile(r'((?:/|--|-|\+)?[^\s=]+)(=?\s*.*)')
# RE for grammar tokens
token_re = re.compile(r'`(\w+)`', re.U)
```
### 10 - sphinx/writers/text.py:

Start line: 767, End line: 874

```python
class TextTranslator(SphinxTranslator):

    def visit_table(self, node: Element) -> None:
        if self.table:
            raise NotImplementedError('Nested tables are not supported.')
        self.new_state(0)
        self.table = Table()

    def depart_table(self, node: Element) -> None:
        self.add_text(str(self.table))
        self.table = None
        self.end_state(wrap=False)

    def visit_acks(self, node: Element) -> None:
        bullet_list = cast(nodes.bullet_list, node[0])
        list_items = cast(Iterable[nodes.list_item], bullet_list)
        self.new_state(0)
        self.add_text(', '.join(n.astext() for n in list_items) + '.')
        self.end_state()
        raise nodes.SkipNode

    def visit_image(self, node: Element) -> None:
        if 'alt' in node.attributes:
            self.add_text(_('[image: %s]') % node['alt'])
        self.add_text(_('[image]'))
        raise nodes.SkipNode

    def visit_transition(self, node: Element) -> None:
        indent = sum(self.stateindent)
        self.new_state(0)
        self.add_text('=' * (MAXWIDTH - indent))
        self.end_state()
        raise nodes.SkipNode

    def visit_bullet_list(self, node: Element) -> None:
        self.list_counter.append(-1)

    def depart_bullet_list(self, node: Element) -> None:
        self.list_counter.pop()

    def visit_enumerated_list(self, node: Element) -> None:
        self.list_counter.append(node.get('start', 1) - 1)

    def depart_enumerated_list(self, node: Element) -> None:
        self.list_counter.pop()

    def visit_definition_list(self, node: Element) -> None:
        self.list_counter.append(-2)

    def depart_definition_list(self, node: Element) -> None:
        self.list_counter.pop()

    def visit_list_item(self, node: Element) -> None:
        if self.list_counter[-1] == -1:
            # bullet list
            self.new_state(2)
        elif self.list_counter[-1] == -2:
            # definition list
            pass
        else:
            # enumerated list
            self.list_counter[-1] += 1
            self.new_state(len(str(self.list_counter[-1])) + 2)

    def depart_list_item(self, node: Element) -> None:
        if self.list_counter[-1] == -1:
            self.end_state(first='* ')
        elif self.list_counter[-1] == -2:
            pass
        else:
            self.end_state(first='%s. ' % self.list_counter[-1])

    def visit_definition_list_item(self, node: Element) -> None:
        self._classifier_count_in_li = len(node.traverse(nodes.classifier))

    def depart_definition_list_item(self, node: Element) -> None:
        pass

    def visit_term(self, node: Element) -> None:
        self.new_state(0)

    def depart_term(self, node: Element) -> None:
        if not self._classifier_count_in_li:
            self.end_state(end=None)

    def visit_classifier(self, node: Element) -> None:
        self.add_text(' : ')

    def depart_classifier(self, node: Element) -> None:
        self._classifier_count_in_li -= 1
        if not self._classifier_count_in_li:
            self.end_state(end=None)

    def visit_definition(self, node: Element) -> None:
        self.new_state()

    def depart_definition(self, node: Element) -> None:
        self.end_state()

    def visit_field_list(self, node: Element) -> None:
        pass

    def depart_field_list(self, node: Element) -> None:
        pass

    def visit_field(self, node: Element) -> None:
        pass

    def depart_field(self, node: Element) -> None:
        pass
```
### 40 - sphinx/domains/std.py:

Start line: 877, End line: 885

```python
class StandardDomain(Domain):

    def _resolve_keyword_xref(self, env: "BuildEnvironment", fromdocname: str,
                              builder: "Builder", typ: str, target: str,
                              node: pending_xref, contnode: Element) -> Element:
        # keywords are oddballs: they are referenced by named labels
        docname, labelid, _ = self.labels.get(target, ('', '', ''))
        if not docname:
            return None
        return make_refnode(builder, fromdocname, docname,
                            labelid, contnode)
```
### 80 - sphinx/domains/std.py:

Start line: 535, End line: 605

```python
class StandardDomain(Domain):
    """
    Domain for all objects that don't fit into another domain or are added
    via the application interface.
    """

    name = 'std'
    label = 'Default'

    object_types = {
        'term': ObjType(_('glossary term'), 'term', searchprio=-1),
        'token': ObjType(_('grammar token'), 'token', searchprio=-1),
        'label': ObjType(_('reference label'), 'ref', 'keyword',
                         searchprio=-1),
        'envvar': ObjType(_('environment variable'), 'envvar'),
        'cmdoption': ObjType(_('program option'), 'option'),
        'doc': ObjType(_('document'), 'doc', searchprio=-1)
    }  # type: Dict[str, ObjType]

    directives = {
        'program': Program,
        'cmdoption': Cmdoption,  # old name for backwards compatibility
        'option': Cmdoption,
        'envvar': EnvVar,
        'glossary': Glossary,
        'productionlist': ProductionList,
    }  # type: Dict[str, Type[Directive]]
    roles = {
        'option':  OptionXRefRole(warn_dangling=True),
        'envvar':  EnvVarXRefRole(),
        # links to tokens in grammar productions
        'token':   TokenXRefRole(),
        # links to terms in glossary
        'term':    XRefRole(lowercase=True, innernodeclass=nodes.inline,
                            warn_dangling=True),
        # links to headings or arbitrary labels
        'ref':     XRefRole(lowercase=True, innernodeclass=nodes.inline,
                            warn_dangling=True),
        # links to labels of numbered figures, tables and code-blocks
        'numref':  XRefRole(lowercase=True,
                            warn_dangling=True),
        # links to labels, without a different title
        'keyword': XRefRole(warn_dangling=True),
        # links to documents
        'doc':     XRefRole(warn_dangling=True, innernodeclass=nodes.inline),
    }  # type: Dict[str, Union[RoleFunction, XRefRole]]

    initial_data = {
        'progoptions': {},      # (program, name) -> docname, labelid
        'objects': {},          # (type, name) -> docname, labelid
        'labels': {             # labelname -> docname, labelid, sectionname
            'genindex': ('genindex', '', _('Index')),
            'modindex': ('py-modindex', '', _('Module Index')),
            'search':   ('search', '', _('Search Page')),
        },
        'anonlabels': {         # labelname -> docname, labelid
            'genindex': ('genindex', ''),
            'modindex': ('py-modindex', ''),
            'search':   ('search', ''),
        },
    }

    dangling_warnings = {
        'term': 'term not in glossary: %(target)s',
        'ref':  'undefined label: %(target)s (if the link has no caption '
                'the label must precede a section header)',
        'numref':  'undefined label: %(target)s',
        'keyword': 'unknown keyword: %(target)s',
        'doc': 'unknown document: %(target)s',
        'option': 'unknown option: %(target)s',
    }
```
