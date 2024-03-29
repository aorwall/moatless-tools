# sphinx-doc__sphinx-11312

| **sphinx-doc/sphinx** | `5cf3dce36ec35c429724bf1312ece9faa0c8db39` |
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
diff --git a/sphinx/util/inspect.py b/sphinx/util/inspect.py
--- a/sphinx/util/inspect.py
+++ b/sphinx/util/inspect.py
@@ -350,38 +350,64 @@ def safe_getattr(obj: Any, name: str, *defargs: Any) -> Any:
         raise AttributeError(name) from exc
 
 
-def object_description(object: Any) -> str:
-    """A repr() implementation that returns text safe to use in reST context."""
-    if isinstance(object, dict):
+def object_description(obj: Any, *, _seen: frozenset = frozenset()) -> str:
+    """A repr() implementation that returns text safe to use in reST context.
+
+    Maintains a set of 'seen' object IDs to detect and avoid infinite recursion.
+    """
+    seen = _seen
+    if isinstance(obj, dict):
+        if id(obj) in seen:
+            return 'dict(...)'
+        seen |= {id(obj)}
         try:
-            sorted_keys = sorted(object)
-        except Exception:
-            pass  # Cannot sort dict keys, fall back to generic repr
-        else:
-            items = ("%s: %s" %
-                     (object_description(key), object_description(object[key]))
-                     for key in sorted_keys)
-            return "{%s}" % ", ".join(items)
-    elif isinstance(object, set):
+            sorted_keys = sorted(obj)
+        except TypeError:
+            # Cannot sort dict keys, fall back to using descriptions as a sort key
+            sorted_keys = sorted(obj, key=lambda k: object_description(k, _seen=seen))
+
+        items = ((object_description(key, _seen=seen),
+                  object_description(obj[key], _seen=seen)) for key in sorted_keys)
+        return '{%s}' % ', '.join(f'{key}: {value}' for (key, value) in items)
+    elif isinstance(obj, set):
+        if id(obj) in seen:
+            return 'set(...)'
+        seen |= {id(obj)}
         try:
-            sorted_values = sorted(object)
+            sorted_values = sorted(obj)
         except TypeError:
-            pass  # Cannot sort set values, fall back to generic repr
-        else:
-            return "{%s}" % ", ".join(object_description(x) for x in sorted_values)
-    elif isinstance(object, frozenset):
+            # Cannot sort set values, fall back to using descriptions as a sort key
+            sorted_values = sorted(obj, key=lambda x: object_description(x, _seen=seen))
+        return '{%s}' % ', '.join(object_description(x, _seen=seen) for x in sorted_values)
+    elif isinstance(obj, frozenset):
+        if id(obj) in seen:
+            return 'frozenset(...)'
+        seen |= {id(obj)}
         try:
-            sorted_values = sorted(object)
+            sorted_values = sorted(obj)
         except TypeError:
-            pass  # Cannot sort frozenset values, fall back to generic repr
-        else:
-            return "frozenset({%s})" % ", ".join(object_description(x)
-                                                 for x in sorted_values)
-    elif isinstance(object, enum.Enum):
-        return f"{object.__class__.__name__}.{object.name}"
+            # Cannot sort frozenset values, fall back to using descriptions as a sort key
+            sorted_values = sorted(obj, key=lambda x: object_description(x, _seen=seen))
+        return 'frozenset({%s})' % ', '.join(object_description(x, _seen=seen)
+                                             for x in sorted_values)
+    elif isinstance(obj, enum.Enum):
+        return f'{obj.__class__.__name__}.{obj.name}'
+    elif isinstance(obj, tuple):
+        if id(obj) in seen:
+            return 'tuple(...)'
+        seen |= frozenset([id(obj)])
+        return '(%s%s)' % (
+            ', '.join(object_description(x, _seen=seen) for x in obj),
+            ',' * (len(obj) == 1),
+        )
+    elif isinstance(obj, list):
+        if id(obj) in seen:
+            return 'list(...)'
+        seen |= {id(obj)}
+        return '[%s]' % ', '.join(object_description(x, _seen=seen) for x in obj)
 
     try:
-        s = repr(object)
+        s = repr(obj)
     except Exception as exc:
         raise ValueError from exc
     # Strip non-deterministic memory addresses such as

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sphinx/util/inspect.py | 353 | 384 | - | - | -


## Problem Statement

```
util.inspect.object_description: does not emit reliable ordering for a set nested within another collection
### Describe the bug

### Summary
Differences appear in some `sphinx` v5.3.0 generated `set` object descriptions for `alembic` v1.8.1, as demonstrated by [recent results visible on the Reproducible Builds diffoscope dashboard](https://tests.reproducible-builds.org/debian/rb-pkg/unstable/amd64/diffoscope-results/alembic.html).

Arguably it could make sense for code authors to intentionally write `set` elements in their code files in a way that does not correspond to their computed sort order -- as a means to communicate with human readers about abstract ideas that aren't relevant to computers at runtime, for example.

However, the current behaviour does result in non-reproducible documentation output.

### Details
In particular, the ordering of a class attribute with a value that contains a set-within-a-tuple seems unreliable across differing builds:

https://github.com/sqlalchemy/alembic/blob/a968c9d2832173ee7d5dde50c7573f7b99424c38/alembic/ddl/impl.py#L90

... is emitted variously as ...

\`\`\`
<span·class="pre">({'NUMERIC',</span>·<span·class="pre">'DECIMAL'},)</span>
\`\`\`

... or ...

\`\`\`
<span·class="pre">({'DECIMAL',</span>·<span·class="pre">'NUMERIC'},)</span>
\`\`\`

cc @lamby who has been [investigating a fix on the reproducible-builds mailing list](https://lists.reproducible-builds.org/pipermail/rb-general/2023-February/002862.html).

### How to Reproduce

It is not yet clear to me exactly what circumstances cause the ordering of elements to vary - and it's OK not to proceed until that's figured out (maybe not a blocker, but it would be nice to have confidence about the cause).

From searching around on previous issues while writing up this bugreport: I wonder if this could be an edge-case for / follow-up to #4834.

### Environment Information

Although these build log links are somewhat ephemeral, the system environment details for two builds that produce differing output are visible at:

- https://tests.reproducible-builds.org/debian/rbuild/unstable/amd64/alembic_1.8.1-2.rbuild.log.gz
- https://tests.reproducible-builds.org/debian/logs/unstable/amd64/alembic_1.8.1-2.build2.log.gz


### Sphinx extensions

\`\`\`python
https://github.com/sqlalchemy/alembic/blob/rel_1_8_1/docs/build/conf.py#L36-L42


sphinx.ext.autodoc
sphinx.ext.intersphinx
changelog
sphinx_paramlinks
sphinx_copybutton
\`\`\`


### Additional context

_No response_

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 doc/conf.py | 126 | 186| 755 | 755 | 2367 | 
| 2 | 2 sphinx/environment/__init__.py | 457 | 514| 534 | 1289 | 9069 | 
| 3 | 3 sphinx/cmd/build.py | 297 | 331| 266 | 1555 | 12007 | 
| 4 | 4 sphinx/builders/changes.py | 117 | 160| 430 | 1985 | 13440 | 
| 5 | 5 sphinx/errors.py | 71 | 128| 297 | 2282 | 14192 | 
| 6 | 6 sphinx/ext/autodoc/__init__.py | 823 | 843| 227 | 2509 | 37674 | 
| 7 | 7 sphinx/builders/html/__init__.py | 1348 | 1426| 997 | 3506 | 50331 | 
| 8 | 7 sphinx/builders/changes.py | 45 | 116| 777 | 4283 | 50331 | 
| 9 | 7 sphinx/builders/html/__init__.py | 166 | 192| 227 | 4510 | 50331 | 
| 10 | 7 doc/conf.py | 1 | 89| 746 | 5256 | 50331 | 
| 11 | 7 sphinx/environment/__init__.py | 516 | 587| 590 | 5846 | 50331 | 
| 12 | 8 sphinx/domains/c.py | 3394 | 3900| 4351 | 10197 | 82183 | 
| 13 | 8 sphinx/environment/__init__.py | 680 | 714| 283 | 10480 | 82183 | 
| 14 | 9 sphinx/transforms/__init__.py | 195 | 225| 229 | 10709 | 85391 | 
| 15 | 9 sphinx/environment/__init__.py | 136 | 260| 1229 | 11938 | 85391 | 
| 16 | 9 sphinx/ext/autodoc/__init__.py | 1 | 115| 808 | 12746 | 85391 | 
| 17 | 10 sphinx/domains/cpp.py | 7974 | 8194| 2071 | 14817 | 154546 | 
| 18 | 10 sphinx/ext/autodoc/__init__.py | 1078 | 1095| 182 | 14999 | 154546 | 
| 19 | 11 sphinx/versioning.py | 1 | 27| 140 | 15139 | 155955 | 
| 20 | 11 sphinx/environment/__init__.py | 80 | 133| 682 | 15821 | 155955 | 
| 21 | 11 sphinx/builders/changes.py | 1 | 20| 130 | 15951 | 155955 | 
| 22 | 12 sphinx/transforms/references.py | 1 | 48| 277 | 16228 | 156232 | 
| 23 | 12 sphinx/domains/cpp.py | 1 | 685| 6015 | 22243 | 156232 | 
| 24 | 13 sphinx/application.py | 1 | 102| 762 | 23005 | 168159 | 
| 25 | 13 sphinx/environment/__init__.py | 338 | 359| 205 | 23210 | 168159 | 
| 26 | 14 sphinx/builders/latex/constants.py | 68 | 118| 537 | 23747 | 170360 | 
| 27 | 14 sphinx/environment/__init__.py | 1 | 78| 524 | 24271 | 170360 | 
| 28 | 14 sphinx/ext/autodoc/__init__.py | 2561 | 2584| 221 | 24492 | 170360 | 
| 29 | 15 sphinx/util/cfamily.py | 1 | 44| 433 | 24925 | 174058 | 
| 30 | 15 doc/conf.py | 90 | 124| 359 | 25284 | 174058 | 
| 31 | 16 sphinx/config.py | 435 | 486| 478 | 25762 | 178833 | 
| 32 | 17 sphinx/builders/__init__.py | 434 | 474| 359 | 26121 | 184480 | 
| 33 | 17 sphinx/domains/c.py | 1 | 787| 6476 | 32597 | 184480 | 
| 34 | 18 sphinx/environment/collectors/metadata.py | 1 | 69| 528 | 33125 | 185008 | 
| 35 | 19 sphinx/ext/doctest.py | 397 | 473| 752 | 33877 | 190004 | 
| 36 | 19 sphinx/domains/cpp.py | 7263 | 7972| 6045 | 39922 | 190004 | 
| 37 | 19 sphinx/builders/html/__init__.py | 464 | 576| 1075 | 40997 | 190004 | 
| 38 | 19 sphinx/config.py | 90 | 157| 892 | 41889 | 190004 | 
| 39 | 20 sphinx/events.py | 1 | 47| 332 | 42221 | 190925 | 
| 40 | 20 sphinx/application.py | 343 | 390| 383 | 42604 | 190925 | 
| 41 | 20 sphinx/builders/html/__init__.py | 1 | 71| 518 | 43122 | 190925 | 
| 42 | 20 sphinx/builders/__init__.py | 571 | 579| 120 | 43242 | 190925 | 
| 43 | 20 sphinx/environment/__init__.py | 716 | 735| 172 | 43414 | 190925 | 
| 44 | 21 sphinx/ext/autodoc/typehints.py | 153 | 217| 521 | 43935 | 192593 | 
| 45 | 22 sphinx/directives/other.py | 1 | 33| 245 | 44180 | 195735 | 


## Missing Patch Files

 * 1: sphinx/util/inspect.py

## Patch

```diff
diff --git a/sphinx/util/inspect.py b/sphinx/util/inspect.py
--- a/sphinx/util/inspect.py
+++ b/sphinx/util/inspect.py
@@ -350,38 +350,64 @@ def safe_getattr(obj: Any, name: str, *defargs: Any) -> Any:
         raise AttributeError(name) from exc
 
 
-def object_description(object: Any) -> str:
-    """A repr() implementation that returns text safe to use in reST context."""
-    if isinstance(object, dict):
+def object_description(obj: Any, *, _seen: frozenset = frozenset()) -> str:
+    """A repr() implementation that returns text safe to use in reST context.
+
+    Maintains a set of 'seen' object IDs to detect and avoid infinite recursion.
+    """
+    seen = _seen
+    if isinstance(obj, dict):
+        if id(obj) in seen:
+            return 'dict(...)'
+        seen |= {id(obj)}
         try:
-            sorted_keys = sorted(object)
-        except Exception:
-            pass  # Cannot sort dict keys, fall back to generic repr
-        else:
-            items = ("%s: %s" %
-                     (object_description(key), object_description(object[key]))
-                     for key in sorted_keys)
-            return "{%s}" % ", ".join(items)
-    elif isinstance(object, set):
+            sorted_keys = sorted(obj)
+        except TypeError:
+            # Cannot sort dict keys, fall back to using descriptions as a sort key
+            sorted_keys = sorted(obj, key=lambda k: object_description(k, _seen=seen))
+
+        items = ((object_description(key, _seen=seen),
+                  object_description(obj[key], _seen=seen)) for key in sorted_keys)
+        return '{%s}' % ', '.join(f'{key}: {value}' for (key, value) in items)
+    elif isinstance(obj, set):
+        if id(obj) in seen:
+            return 'set(...)'
+        seen |= {id(obj)}
         try:
-            sorted_values = sorted(object)
+            sorted_values = sorted(obj)
         except TypeError:
-            pass  # Cannot sort set values, fall back to generic repr
-        else:
-            return "{%s}" % ", ".join(object_description(x) for x in sorted_values)
-    elif isinstance(object, frozenset):
+            # Cannot sort set values, fall back to using descriptions as a sort key
+            sorted_values = sorted(obj, key=lambda x: object_description(x, _seen=seen))
+        return '{%s}' % ', '.join(object_description(x, _seen=seen) for x in sorted_values)
+    elif isinstance(obj, frozenset):
+        if id(obj) in seen:
+            return 'frozenset(...)'
+        seen |= {id(obj)}
         try:
-            sorted_values = sorted(object)
+            sorted_values = sorted(obj)
         except TypeError:
-            pass  # Cannot sort frozenset values, fall back to generic repr
-        else:
-            return "frozenset({%s})" % ", ".join(object_description(x)
-                                                 for x in sorted_values)
-    elif isinstance(object, enum.Enum):
-        return f"{object.__class__.__name__}.{object.name}"
+            # Cannot sort frozenset values, fall back to using descriptions as a sort key
+            sorted_values = sorted(obj, key=lambda x: object_description(x, _seen=seen))
+        return 'frozenset({%s})' % ', '.join(object_description(x, _seen=seen)
+                                             for x in sorted_values)
+    elif isinstance(obj, enum.Enum):
+        return f'{obj.__class__.__name__}.{obj.name}'
+    elif isinstance(obj, tuple):
+        if id(obj) in seen:
+            return 'tuple(...)'
+        seen |= frozenset([id(obj)])
+        return '(%s%s)' % (
+            ', '.join(object_description(x, _seen=seen) for x in obj),
+            ',' * (len(obj) == 1),
+        )
+    elif isinstance(obj, list):
+        if id(obj) in seen:
+            return 'list(...)'
+        seen |= {id(obj)}
+        return '[%s]' % ', '.join(object_description(x, _seen=seen) for x in obj)
 
     try:
-        s = repr(object)
+        s = repr(obj)
     except Exception as exc:
         raise ValueError from exc
     # Strip non-deterministic memory addresses such as

```

## Test Patch

```diff
diff --git a/tests/test_util_inspect.py b/tests/test_util_inspect.py
--- a/tests/test_util_inspect.py
+++ b/tests/test_util_inspect.py
@@ -503,10 +503,32 @@ def test_set_sorting():
     assert description == "{'a', 'b', 'c', 'd', 'e', 'f', 'g'}"
 
 
+def test_set_sorting_enum():
+    class MyEnum(enum.Enum):
+        a = 1
+        b = 2
+        c = 3
+
+    set_ = set(MyEnum)
+    description = inspect.object_description(set_)
+    assert description == "{MyEnum.a, MyEnum.b, MyEnum.c}"
+
+
 def test_set_sorting_fallback():
     set_ = {None, 1}
     description = inspect.object_description(set_)
-    assert description in ("{1, None}", "{None, 1}")
+    assert description == "{1, None}"
+
+
+def test_deterministic_nested_collection_descriptions():
+    # sortable
+    assert inspect.object_description([{1, 2, 3, 10}]) == "[{1, 2, 3, 10}]"
+    assert inspect.object_description(({1, 2, 3, 10},)) == "({1, 2, 3, 10},)"
+    # non-sortable (elements of varying datatype)
+    assert inspect.object_description([{None, 1}]) == "[{1, None}]"
+    assert inspect.object_description(({None, 1},)) == "({1, None},)"
+    assert inspect.object_description([{None, 1, 'A'}]) == "[{'A', 1, None}]"
+    assert inspect.object_description(({None, 1, 'A'},)) == "({'A', 1, None},)"
 
 
 def test_frozenset_sorting():
@@ -518,7 +540,39 @@ def test_frozenset_sorting():
 def test_frozenset_sorting_fallback():
     frozenset_ = frozenset((None, 1))
     description = inspect.object_description(frozenset_)
-    assert description in ("frozenset({1, None})", "frozenset({None, 1})")
+    assert description == "frozenset({1, None})"
+
+
+def test_nested_tuple_sorting():
+    tuple_ = ({"c", "b", "a"},)  # nb. trailing comma
+    description = inspect.object_description(tuple_)
+    assert description == "({'a', 'b', 'c'},)"
+
+    tuple_ = ({"c", "b", "a"}, {"f", "e", "d"})
+    description = inspect.object_description(tuple_)
+    assert description == "({'a', 'b', 'c'}, {'d', 'e', 'f'})"
+
+
+def test_recursive_collection_description():
+    dict_a_, dict_b_ = {"a": 1}, {"b": 2}
+    dict_a_["link"], dict_b_["link"] = dict_b_, dict_a_
+    description_a, description_b = (
+        inspect.object_description(dict_a_),
+        inspect.object_description(dict_b_),
+    )
+    assert description_a == "{'a': 1, 'link': {'b': 2, 'link': dict(...)}}"
+    assert description_b == "{'b': 2, 'link': {'a': 1, 'link': dict(...)}}"
+
+    list_c_, list_d_ = [1, 2, 3, 4], [5, 6, 7, 8]
+    list_c_.append(list_d_)
+    list_d_.append(list_c_)
+    description_c, description_d = (
+        inspect.object_description(list_c_),
+        inspect.object_description(list_d_),
+    )
+
+    assert description_c == "[1, 2, 3, 4, [5, 6, 7, 8, list(...)]]"
+    assert description_d == "[5, 6, 7, 8, [1, 2, 3, 4, list(...)]]"
 
 
 def test_dict_customtype():

```


## Code snippets

### 1 - doc/conf.py:

Start line: 126, End line: 186

```python
nitpick_ignore = {
    ('cpp:class', 'template<typename TOuter> template<typename TInner> Wrapper::Outer<TOuter>::Inner'),  # NoQA: E501
    ('cpp:identifier', 'MyContainer'),
    ('js:func', 'SomeError'),
    ('js:func', 'number'),
    ('js:func', 'string'),
    ('py:attr', 'srcline'),
    ('py:class', 'Element'),  # sphinx.domains.Domain
    ('py:class', 'IndexEntry'),  # sphinx.domains.IndexEntry
    ('py:class', 'Node'),  # sphinx.domains.Domain
    ('py:class', 'NullTranslations'),  # gettext.NullTranslations
    ('py:class', 'RoleFunction'),  # sphinx.domains.Domain
    ('py:class', 'Theme'),  # sphinx.application.TemplateBridge
    ('py:class', 'TitleGetter'),  # sphinx.domains.Domain
    ('py:class', 'XRefRole'),  # sphinx.domains.Domain
    ('py:class', 'docutils.nodes.Element'),
    ('py:class', 'docutils.nodes.Node'),
    ('py:class', 'docutils.nodes.NodeVisitor'),
    ('py:class', 'docutils.nodes.TextElement'),
    ('py:class', 'docutils.nodes.document'),
    ('py:class', 'docutils.nodes.system_message'),
    ('py:class', 'docutils.parsers.Parser'),
    ('py:class', 'docutils.parsers.rst.states.Inliner'),
    ('py:class', 'docutils.transforms.Transform'),
    ('py:class', 'nodes.NodeVisitor'),
    ('py:class', 'nodes.document'),
    ('py:class', 'nodes.reference'),
    ('py:class', 'pygments.lexer.Lexer'),
    ('py:class', 'sphinx.directives.ObjDescT'),
    ('py:class', 'sphinx.domains.IndexEntry'),
    ('py:class', 'sphinx.ext.autodoc.Documenter'),
    ('py:class', 'sphinx.errors.NoUri'),
    ('py:class', 'sphinx.roles.XRefRole'),
    ('py:class', 'sphinx.search.SearchLanguage'),
    ('py:class', 'sphinx.theming.Theme'),
    ('py:class', 'sphinxcontrib.websupport.errors.DocumentNotFoundError'),
    ('py:class', 'sphinxcontrib.websupport.errors.UserNotAuthorizedError'),
    ('py:exc', 'docutils.nodes.SkipNode'),
    ('py:exc', 'sphinx.environment.NoUri'),
    ('py:func', 'setup'),
    ('py:func', 'sphinx.util.nodes.nested_parse_with_titles'),
    # Error in sphinxcontrib.websupport.core::WebSupport.add_comment
    ('py:meth', 'get_comments'),
    ('py:mod', 'autodoc'),
    ('py:mod', 'docutils.nodes'),
    ('py:mod', 'docutils.parsers.rst.directives'),
    ('py:mod', 'sphinx.ext'),
    ('py:obj', 'sphinx.util.relative_uri'),
    ('rst:role', 'c:any'),
    ('std:confval', 'autodoc_inherit_docstring'),
    ('std:confval', 'automodule_skip_lines'),
    ('std:confval', 'autossummary_imported_members'),
    ('std:confval', 'gettext_language_team'),
    ('std:confval', 'gettext_last_translator'),
    ('std:confval', 'globaltoc_collapse'),
    ('std:confval', 'globaltoc_includehidden'),
    ('std:confval', 'globaltoc_maxdepth'),
}


# -- Extension interface -------------------------------------------------------
```
### 2 - sphinx/environment/__init__.py:

Start line: 457, End line: 514

```python
class BuildEnvironment:

    def get_outdated_files(self, config_changed: bool) -> tuple[set[str], set[str], set[str]]:
        """Return (added, changed, removed) sets."""
        # clear all files no longer present
        removed = set(self.all_docs) - self.found_docs

        added: set[str] = set()
        changed: set[str] = set()

        if config_changed:
            # config values affect e.g. substitutions
            added = self.found_docs
        else:
            for docname in self.found_docs:
                if docname not in self.all_docs:
                    logger.debug('[build target] added %r', docname)
                    added.add(docname)
                    continue
                # if the doctree file is not there, rebuild
                filename = path.join(self.doctreedir, docname + '.doctree')
                if not path.isfile(filename):
                    logger.debug('[build target] changed %r', docname)
                    changed.add(docname)
                    continue
                # check the "reread always" list
                if docname in self.reread_always:
                    logger.debug('[build target] changed %r', docname)
                    changed.add(docname)
                    continue
                # check the mtime of the document
                mtime = self.all_docs[docname]
                newmtime = _last_modified_time(self.doc2path(docname))
                if newmtime > mtime:
                    # convert integer microseconds to floating-point seconds,
                    # and then to timezone-aware datetime objects.
                    mtime_dt = datetime.fromtimestamp(mtime / 1_000_000, tz=timezone.utc)
                    newmtime_dt = datetime.fromtimestamp(mtime / 1_000_000, tz=timezone.utc)
                    logger.debug('[build target] outdated %r: %s -> %s',
                                 docname, mtime_dt, newmtime_dt)
                    changed.add(docname)
                    continue
                # finally, check the mtime of dependencies
                for dep in self.dependencies[docname]:
                    try:
                        # this will do the right thing when dep is absolute too
                        deppath = path.join(self.srcdir, dep)
                        if not path.isfile(deppath):
                            changed.add(docname)
                            break
                        depmtime = _last_modified_time(deppath)
                        if depmtime > mtime:
                            changed.add(docname)
                            break
                    except OSError:
                        # give it another chance
                        changed.add(docname)
                        break

        return added, changed, removed
```
### 3 - sphinx/cmd/build.py:

Start line: 297, End line: 331

```python
def _bug_report_info() -> int:
    from platform import platform, python_implementation

    import docutils
    import jinja2
    import pygments

    print('Please paste all output below into the bug report template\n\n')
    print('```text')
    print(f'Platform:              {sys.platform}; ({platform()})')
    print(f'Python version:        {sys.version})')
    print(f'Python implementation: {python_implementation()}')
    print(f'Sphinx version:        {sphinx.__display_version__}')
    print(f'Docutils version:      {docutils.__version__}')
    print(f'Jinja2 version:        {jinja2.__version__}')
    print(f'Pygments version:      {pygments.__version__}')
    print('```')
    return 0


def main(argv: list[str] = sys.argv[1:]) -> int:
    locale.setlocale(locale.LC_ALL, '')
    sphinx.locale.init_console()

    if argv[:1] == ['--bug-report']:
        return _bug_report_info()
    if argv[:1] == ['-M']:
        return make_main(argv)
    else:
        return build_main(argv)


if __name__ == '__main__':
    raise SystemExit(main())
```
### 4 - sphinx/builders/changes.py:

Start line: 117, End line: 160

```python
class ChangesBuilder(Builder):

    def write(self, *ignored: Any) -> None:
        # ... other code
        for docname in self.env.all_docs:
            with open(self.env.doc2path(docname),
                      encoding=self.env.config.source_encoding) as f:
                try:
                    lines = f.readlines()
                except UnicodeDecodeError:
                    logger.warning(__('could not read %r for changelog creation'), docname)
                    continue
            targetfn = path.join(self.outdir, 'rst', os_path(docname)) + '.html'
            ensuredir(path.dirname(targetfn))
            with open(targetfn, 'w', encoding='utf-8') as f:
                text = ''.join(hl(i + 1, line) for (i, line) in enumerate(lines))
                ctx = {
                    'filename': self.env.doc2path(docname, False),
                    'text': text,
                }
                f.write(self.templates.render('changes/rstsource.html', ctx))
        themectx = {'theme_' + key: val for (key, val) in
                    self.theme.get_options({}).items()}
        copy_asset_file(path.join(package_dir, 'themes', 'default', 'static', 'default.css_t'),
                        self.outdir, context=themectx, renderer=self.templates)
        copy_asset_file(path.join(package_dir, 'themes', 'basic', 'static', 'basic.css'),
                        self.outdir)

    def hl(self, text: str, version: str) -> str:
        text = html.escape(text)
        for directive in ('versionchanged', 'versionadded', 'deprecated'):
            text = text.replace(f'.. {directive}:: {version}',
                                f'<b>.. {directive}:: {version}</b>')
        return text

    def finish(self) -> None:
        pass


def setup(app: Sphinx) -> dict[str, Any]:
    app.add_builder(ChangesBuilder)

    return {
        'version': 'builtin',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
```
### 5 - sphinx/errors.py:

Start line: 71, End line: 128

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
    """Raised by builder.get_relative_uri() or from missing-reference handlers
    if there is no URI available."""
    pass


class FiletypeNotFoundError(Exception):
    """Raised by get_filetype() if a filename matches no source suffix."""
    pass
```
### 6 - sphinx/ext/autodoc/__init__.py:

Start line: 823, End line: 843

```python
class Documenter:

    def sort_members(self, documenters: list[tuple[Documenter, bool]],
                     order: str) -> list[tuple[Documenter, bool]]:
        """Sort the given member list."""
        if order == 'groupwise':
            # sort by group; alphabetically within groups
            documenters.sort(key=lambda e: (e[0].member_order, e[0].name))
        elif order == 'bysource':
            # By default, member discovery order matches source order,
            # as dicts are insertion-ordered from Python 3.7.
            if self.analyzer:
                # sort by source order, by virtue of the module analyzer
                tagorder = self.analyzer.tagorder

                def keyfunc(entry: tuple[Documenter, bool]) -> int:
                    fullname = entry[0].name.split('::')[1]
                    return tagorder.get(fullname, len(tagorder))
                documenters.sort(key=keyfunc)
        else:  # alphabetical
            documenters.sort(key=lambda e: e[0].name)

        return documenters
```
### 7 - sphinx/builders/html/__init__.py:

Start line: 1348, End line: 1426

```python
def setup(app: Sphinx) -> dict[str, Any]:
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
    app.add_config_value('html_style', None, 'html', [list, str])
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
    app.add_config_value('html_show_search_summary', True, 'html')
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
    app.add_config_value('html_codeblock_linenos_style', 'inline', 'html',  # RemovedInSphinx70Warning  # noqa: E501
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
    app.connect('config-inited', error_on_html_4, priority=800)
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
### 8 - sphinx/builders/changes.py:

Start line: 45, End line: 116

```python
class ChangesBuilder(Builder):

    def write(self, *ignored: Any) -> None:
        version = self.config.version
        domain = cast(ChangeSetDomain, self.env.get_domain('changeset'))
        libchanges: dict[str, list[tuple[str, str, int]]] = {}
        apichanges: list[tuple[str, str, int]] = []
        otherchanges: dict[tuple[str, str], list[tuple[str, str, int]]] = {}

        changesets = domain.get_changesets_for(version)
        if not changesets:
            logger.info(bold(__('no changes in version %s.') % version))
            return
        logger.info(bold(__('writing summary file...')))
        for changeset in changesets:
            if isinstance(changeset.descname, tuple):
                descname = changeset.descname[0]
            else:
                descname = changeset.descname
            ttext = self.typemap[changeset.type]
            context = changeset.content.replace('\n', ' ')
            if descname and changeset.docname.startswith('c-api'):
                if context:
                    entry = f'<b>{descname}</b>: <i>{ttext}:</i> {context}'
                else:
                    entry = f'<b>{descname}</b>: <i>{ttext}</i>.'
                apichanges.append((entry, changeset.docname, changeset.lineno))
            elif descname or changeset.module:
                module = changeset.module or _('Builtins')
                if not descname:
                    descname = _('Module level')
                if context:
                    entry = f'<b>{descname}</b>: <i>{ttext}:</i> {context}'
                else:
                    entry = f'<b>{descname}</b>: <i>{ttext}</i>.'
                libchanges.setdefault(module, []).append((entry, changeset.docname,
                                                          changeset.lineno))
            else:
                if not context:
                    continue
                entry = f'<i>{ttext.capitalize()}:</i> {context}'
                title = self.env.titles[changeset.docname].astext()
                otherchanges.setdefault((changeset.docname, title), []).append(
                    (entry, changeset.docname, changeset.lineno))

        ctx = {
            'project': self.config.project,
            'version': version,
            'docstitle': self.config.html_title,
            'shorttitle': self.config.html_short_title,
            'libchanges': sorted(libchanges.items()),
            'apichanges': sorted(apichanges),
            'otherchanges': sorted(otherchanges.items()),
            'show_copyright': self.config.html_show_copyright,
            'show_sphinx': self.config.html_show_sphinx,
        }
        with open(path.join(self.outdir, 'index.html'), 'w', encoding='utf8') as f:
            f.write(self.templates.render('changes/frameset.html', ctx))
        with open(path.join(self.outdir, 'changes.html'), 'w', encoding='utf8') as f:
            f.write(self.templates.render('changes/versionchanges.html', ctx))

        hltext = ['.. versionadded:: %s' % version,
                  '.. versionchanged:: %s' % version,
                  '.. deprecated:: %s' % version]

        def hl(no: int, line: str) -> str:
            line = '<a name="L%s"> </a>' % no + html.escape(line)
            for x in hltext:
                if x in line:
                    line = '<span class="hl">%s</span>' % line
                    break
            return line

        logger.info(bold(__('copying source files...')))
        # ... other code
```
### 9 - sphinx/builders/html/__init__.py:

Start line: 166, End line: 192

```python
class BuildInfo:

    def __init__(
        self,
        config: Config | None = None,
        tags: Tags | None = None,
        config_categories: list[str] = [],
    ) -> None:
        self.config_hash = ''
        self.tags_hash = ''

        if config:
            values = {c.name: c.value for c in config.filter(config_categories)}
            self.config_hash = get_stable_hash(values)

        if tags:
            self.tags_hash = get_stable_hash(sorted(tags))

    def __eq__(self, other: BuildInfo) -> bool:  # type: ignore
        return (self.config_hash == other.config_hash and
                self.tags_hash == other.tags_hash)

    def dump(self, f: IO) -> None:
        f.write('# Sphinx build info version 1\n'
                '# This file hashes the configuration used when building these files.'
                ' When it is not found, a full rebuild will be done.\n'
                'config: %s\n'
                'tags: %s\n' %
                (self.config_hash, self.tags_hash))
```
### 10 - doc/conf.py:

Start line: 1, End line: 89

```python
# Sphinx documentation build configuration file

import os
import re
import time

import sphinx

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.doctest', 'sphinx.ext.todo',
              'sphinx.ext.autosummary', 'sphinx.ext.extlinks',
              'sphinx.ext.intersphinx',
              'sphinx.ext.viewcode', 'sphinx.ext.inheritance_diagram']

templates_path = ['_templates']
exclude_patterns = ['_build']

project = 'Sphinx'
copyright = f'2007-{time.strftime("%Y")}, the Sphinx developers'
version = sphinx.__display_version__
release = version
show_authors = True

html_theme = 'sphinx13'
html_theme_path = ['_themes']
html_css_files = [
    # 'basic.css',  # included through inheritance from the basic theme
    'sphinx13.css',
]
modindex_common_prefix = ['sphinx.']
html_static_path = ['_static']
html_title = 'Sphinx documentation'
html_additional_pages = {'contents': 'contents.html'}
html_use_opensearch = 'https://www.sphinx-doc.org/en/master'
html_baseurl = 'https://www.sphinx-doc.org/en/master/'
html_favicon = '_static/favicon.svg'

htmlhelp_basename = 'Sphinxdoc'

epub_theme = 'epub'
epub_basename = 'sphinx'
epub_author = 'the Sphinx developers'
epub_publisher = 'https://www.sphinx-doc.org/'
epub_uid = 'web-site'
epub_scheme = 'url'
epub_identifier = epub_publisher
epub_pre_files = [('index.xhtml', 'Welcome')]
epub_post_files = [('usage/installation.xhtml', 'Installing Sphinx'),
                   ('develop.xhtml', 'Sphinx development')]
epub_exclude_files = ['_static/opensearch.xml', '_static/doctools.js',
                      '_static/searchtools.js',
                      '_static/sphinx_highlight.js',
                      '_static/basic.css',
                      '_static/language_data.js',
                      'search.html', '_static/websupport.js']
epub_fix_images = False
epub_max_image_width = 0
epub_show_urls = 'inline'
epub_use_index = False
epub_description = 'Sphinx documentation generator system manual'

latex_documents = [('index', 'sphinx.tex', 'Sphinx Documentation',
                    'the Sphinx developers', 'manual', 1)]
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
\setlength{\tymin}{2cm}%     avoid too cramped table columns
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

linkcheck_timeout = 5

autodoc_member_order = 'groupwise'
autosummary_generate = False
todo_include_todos = True
```
