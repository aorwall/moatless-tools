# sphinx-doc__sphinx-7597

| **sphinx-doc/sphinx** | `c13ecd243709d1e210a030be5aa09b7714e35730` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 7343 |
| **Any found context length** | 216 |
| **Avg pos** | 55.0 |
| **Min pos** | 1 |
| **Max pos** | 22 |
| **Top file pos** | 1 |
| **Missing snippets** | 4 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sphinx/domains/python.py b/sphinx/domains/python.py
--- a/sphinx/domains/python.py
+++ b/sphinx/domains/python.py
@@ -77,17 +77,19 @@
                                          ('deprecated', bool)])
 
 
-def _parse_annotation(annotation: str) -> List[Node]:
-    """Parse type annotation."""
-    def make_xref(text: str) -> addnodes.pending_xref:
-        if text == 'None':
-            reftype = 'obj'
-        else:
-            reftype = 'class'
+def type_to_xref(text: str) -> addnodes.pending_xref:
+    """Convert a type string to a cross reference node."""
+    if text == 'None':
+        reftype = 'obj'
+    else:
+        reftype = 'class'
 
-        return pending_xref('', nodes.Text(text),
-                            refdomain='py', reftype=reftype, reftarget=text)
+    return pending_xref('', nodes.Text(text),
+                        refdomain='py', reftype=reftype, reftarget=text)
 
+
+def _parse_annotation(annotation: str) -> List[Node]:
+    """Parse type annotation."""
     def unparse(node: ast.AST) -> List[Node]:
         if isinstance(node, ast.Attribute):
             return [nodes.Text("%s.%s" % (unparse(node.value)[0], node.attr))]
@@ -133,10 +135,10 @@ def unparse(node: ast.AST) -> List[Node]:
         result = unparse(tree)
         for i, node in enumerate(result):
             if isinstance(node, nodes.Text):
-                result[i] = make_xref(str(node))
+                result[i] = type_to_xref(str(node))
         return result
     except SyntaxError:
-        return [make_xref(annotation)]
+        return [type_to_xref(annotation)]
 
 
 def _parse_arglist(arglist: str) -> addnodes.desc_parameterlist:
@@ -621,7 +623,7 @@ def handle_signature(self, sig: str, signode: desc_signature) -> Tuple[str, str]
 
         typ = self.options.get('type')
         if typ:
-            signode += addnodes.desc_annotation(typ, ': ' + typ)
+            signode += addnodes.desc_annotation(typ, '', nodes.Text(': '), type_to_xref(typ))
 
         value = self.options.get('value')
         if value:
@@ -866,7 +868,7 @@ def handle_signature(self, sig: str, signode: desc_signature) -> Tuple[str, str]
 
         typ = self.options.get('type')
         if typ:
-            signode += addnodes.desc_annotation(typ, ': ' + typ)
+            signode += addnodes.desc_annotation(typ, '', nodes.Text(': '), type_to_xref(typ))
 
         value = self.options.get('value')
         if value:

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sphinx/domains/python.py | 80 | 89 | 16 | 1 | 5456
| sphinx/domains/python.py | 136 | 139 | 16 | 1 | 5456
| sphinx/domains/python.py | 624 | 624 | 1 | 1 | 216
| sphinx/domains/python.py | 869 | 869 | 22 | 1 | 7343


## Problem Statement

```
py domain: Change a type annotation for variables to a hyperlink
**Is your feature request related to a problem? Please describe.**
py domain: Change a type annotation for variables to a hyperlink

**Describe the solution you'd like**

`type` option was added to python directives since 2.x. But it has been represented as mere text. It must be useful if it is converted to a hyperlink to the type definition.
\`\`\`
.. py:data:: foo
   :type: int
\`\`\`

**Describe alternatives you've considered**
No

**Additional context**
No


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 sphinx/domains/python.py** | 610 | 637| 216 | 216 | 11611 | 
| 2 | 2 sphinx/util/typing.py | 42 | 61| 160 | 376 | 13479 | 
| 3 | **2 sphinx/domains/python.py** | 978 | 996| 143 | 519 | 13479 | 
| 4 | 2 sphinx/util/typing.py | 64 | 105| 440 | 959 | 13479 | 
| 5 | **2 sphinx/domains/python.py** | 240 | 258| 214 | 1173 | 13479 | 
| 6 | **2 sphinx/domains/python.py** | 298 | 306| 126 | 1299 | 13479 | 
| 7 | 3 sphinx/domains/std.py | 621 | 643| 249 | 1548 | 23688 | 
| 8 | 3 sphinx/util/typing.py | 108 | 125| 184 | 1732 | 23688 | 
| 9 | 3 sphinx/util/typing.py | 127 | 198| 855 | 2587 | 23688 | 
| 10 | 4 sphinx/domains/rst.py | 141 | 172| 356 | 2943 | 26158 | 
| 11 | **4 sphinx/domains/python.py** | 999 | 1019| 248 | 3191 | 26158 | 
| 12 | 5 sphinx/ext/autodoc/typehints.py | 78 | 142| 536 | 3727 | 27254 | 
| 13 | **5 sphinx/domains/python.py** | 915 | 975| 521 | 4248 | 27254 | 
| 14 | 6 sphinx/directives/__init__.py | 52 | 74| 206 | 4454 | 29749 | 
| 15 | **6 sphinx/domains/python.py** | 11 | 77| 519 | 4973 | 29749 | 
| **-> 16 <-** | **6 sphinx/domains/python.py** | 80 | 139| 483 | 5456 | 29749 | 
| 17 | 7 sphinx/util/docfields.py | 212 | 228| 161 | 5617 | 33113 | 
| 18 | 7 sphinx/directives/__init__.py | 76 | 88| 128 | 5745 | 33113 | 
| 19 | **7 sphinx/domains/python.py** | 1110 | 1161| 516 | 6261 | 33113 | 
| 20 | 8 sphinx/ext/graphviz.py | 109 | 125| 128 | 6389 | 36706 | 
| 21 | 9 sphinx/ext/autodoc/__init__.py | 13 | 124| 810 | 7199 | 52530 | 
| **-> 22 <-** | **9 sphinx/domains/python.py** | 855 | 875| 144 | 7343 | 52530 | 
| 23 | 9 sphinx/directives/__init__.py | 269 | 307| 273 | 7616 | 52530 | 
| 24 | 10 sphinx/domains/cpp.py | 2382 | 2439| 494 | 8110 | 113890 | 
| 25 | 10 sphinx/domains/std.py | 243 | 260| 125 | 8235 | 113890 | 
| 26 | **10 sphinx/domains/python.py** | 697 | 751| 530 | 8765 | 113890 | 
| 27 | 10 sphinx/util/typing.py | 11 | 39| 200 | 8965 | 113890 | 
| 28 | **10 sphinx/domains/python.py** | 877 | 889| 132 | 9097 | 113890 | 
| 29 | 10 sphinx/ext/graphviz.py | 174 | 211| 314 | 9411 | 113890 | 
| 30 | 11 sphinx/ext/napoleon/__init__.py | 311 | 327| 158 | 9569 | 117653 | 
| 31 | 11 sphinx/domains/rst.py | 118 | 139| 186 | 9755 | 117653 | 
| 32 | 12 sphinx/domains/changeset.py | 48 | 98| 439 | 10194 | 118838 | 
| 33 | 13 sphinx/directives/other.py | 9 | 40| 238 | 10432 | 121995 | 
| 34 | 14 sphinx/directives/code.py | 34 | 57| 173 | 10605 | 125892 | 
| 35 | 15 sphinx/domains/javascript.py | 295 | 312| 196 | 10801 | 129907 | 
| 36 | 16 sphinx/ext/autodoc/type_comment.py | 117 | 139| 226 | 11027 | 131108 | 
| 37 | 16 sphinx/directives/other.py | 187 | 208| 141 | 11168 | 131108 | 
| 38 | 16 sphinx/domains/std.py | 747 | 773| 322 | 11490 | 131108 | 
| 39 | 16 sphinx/domains/std.py | 51 | 66| 140 | 11630 | 131108 | 
| 40 | 16 sphinx/domains/std.py | 214 | 240| 274 | 11904 | 131108 | 
| 41 | **16 sphinx/domains/python.py** | 309 | 353| 366 | 12270 | 131108 | 
| 42 | 17 sphinx/ext/autodoc/directive.py | 9 | 49| 298 | 12568 | 132369 | 
| 43 | 17 sphinx/domains/std.py | 661 | 676| 204 | 12772 | 132369 | 
| 44 | **17 sphinx/domains/python.py** | 754 | 791| 228 | 13000 | 132369 | 
| 45 | 17 sphinx/domains/std.py | 181 | 212| 257 | 13257 | 132369 | 
| 46 | 18 sphinx/ext/autosummary/__init__.py | 411 | 437| 217 | 13474 | 138834 | 
| 47 | 18 sphinx/directives/code.py | 60 | 84| 187 | 13661 | 138834 | 
| 48 | **18 sphinx/domains/python.py** | 1163 | 1173| 131 | 13792 | 138834 | 
| 49 | 18 sphinx/domains/std.py | 11 | 48| 323 | 14115 | 138834 | 
| 50 | **18 sphinx/domains/python.py** | 283 | 295| 140 | 14255 | 138834 | 
| 51 | **18 sphinx/domains/python.py** | 557 | 590| 293 | 14548 | 138834 | 
| 52 | 18 sphinx/domains/std.py | 127 | 178| 463 | 15011 | 138834 | 
| 53 | 18 sphinx/domains/std.py | 535 | 605| 723 | 15734 | 138834 | 
| 54 | 19 sphinx/highlighting.py | 57 | 166| 876 | 16610 | 140142 | 
| 55 | 20 sphinx/util/inspect.py | 620 | 693| 562 | 17172 | 145593 | 
| 56 | 21 sphinx/writers/html5.py | 631 | 717| 686 | 17858 | 152496 | 
| 57 | 21 sphinx/writers/html5.py | 143 | 203| 567 | 18425 | 152496 | 
| 58 | **21 sphinx/domains/python.py** | 815 | 852| 225 | 18650 | 152496 | 
| 59 | **21 sphinx/domains/python.py** | 1175 | 1203| 275 | 18925 | 152496 | 
| 60 | 22 doc/usage/extensions/example_google.py | 38 | 75| 245 | 19170 | 154481 | 
| 61 | 23 sphinx/pycode/ast.py | 11 | 44| 201 | 19371 | 155973 | 
| 62 | **23 sphinx/domains/python.py** | 544 | 554| 145 | 19516 | 155973 | 
| 63 | 24 doc/usage/extensions/example_numpy.py | 48 | 98| 276 | 19792 | 158081 | 
| 64 | 25 sphinx/locale/__init__.py | 239 | 267| 232 | 20024 | 160147 | 
| 65 | 26 sphinx/directives/patches.py | 9 | 25| 124 | 20148 | 161760 | 
| 66 | 26 sphinx/domains/cpp.py | 6364 | 7143| 6652 | 26800 | 161760 | 
| 67 | 27 sphinx/domains/c.py | 2693 | 3472| 6630 | 33430 | 190416 | 
| 69 | **28 sphinx/domains/python.py** | 1307 | 1319| 128 | 34351 | 199370 | 


## Patch

```diff
diff --git a/sphinx/domains/python.py b/sphinx/domains/python.py
--- a/sphinx/domains/python.py
+++ b/sphinx/domains/python.py
@@ -77,17 +77,19 @@
                                          ('deprecated', bool)])
 
 
-def _parse_annotation(annotation: str) -> List[Node]:
-    """Parse type annotation."""
-    def make_xref(text: str) -> addnodes.pending_xref:
-        if text == 'None':
-            reftype = 'obj'
-        else:
-            reftype = 'class'
+def type_to_xref(text: str) -> addnodes.pending_xref:
+    """Convert a type string to a cross reference node."""
+    if text == 'None':
+        reftype = 'obj'
+    else:
+        reftype = 'class'
 
-        return pending_xref('', nodes.Text(text),
-                            refdomain='py', reftype=reftype, reftarget=text)
+    return pending_xref('', nodes.Text(text),
+                        refdomain='py', reftype=reftype, reftarget=text)
 
+
+def _parse_annotation(annotation: str) -> List[Node]:
+    """Parse type annotation."""
     def unparse(node: ast.AST) -> List[Node]:
         if isinstance(node, ast.Attribute):
             return [nodes.Text("%s.%s" % (unparse(node.value)[0], node.attr))]
@@ -133,10 +135,10 @@ def unparse(node: ast.AST) -> List[Node]:
         result = unparse(tree)
         for i, node in enumerate(result):
             if isinstance(node, nodes.Text):
-                result[i] = make_xref(str(node))
+                result[i] = type_to_xref(str(node))
         return result
     except SyntaxError:
-        return [make_xref(annotation)]
+        return [type_to_xref(annotation)]
 
 
 def _parse_arglist(arglist: str) -> addnodes.desc_parameterlist:
@@ -621,7 +623,7 @@ def handle_signature(self, sig: str, signode: desc_signature) -> Tuple[str, str]
 
         typ = self.options.get('type')
         if typ:
-            signode += addnodes.desc_annotation(typ, ': ' + typ)
+            signode += addnodes.desc_annotation(typ, '', nodes.Text(': '), type_to_xref(typ))
 
         value = self.options.get('value')
         if value:
@@ -866,7 +868,7 @@ def handle_signature(self, sig: str, signode: desc_signature) -> Tuple[str, str]
 
         typ = self.options.get('type')
         if typ:
-            signode += addnodes.desc_annotation(typ, ': ' + typ)
+            signode += addnodes.desc_annotation(typ, '', nodes.Text(': '), type_to_xref(typ))
 
         value = self.options.get('value')
         if value:

```

## Test Patch

```diff
diff --git a/tests/test_domain_py.py b/tests/test_domain_py.py
--- a/tests/test_domain_py.py
+++ b/tests/test_domain_py.py
@@ -420,7 +420,8 @@ def test_pydata_signature(app):
     doctree = restructuredtext.parse(app, text)
     assert_node(doctree, (addnodes.index,
                           [desc, ([desc_signature, ([desc_name, "version"],
-                                                    [desc_annotation, ": int"],
+                                                    [desc_annotation, (": ",
+                                                                       [pending_xref, "int"])],
                                                     [desc_annotation, " = 1"])],
                                   desc_content)]))
     assert_node(doctree[1], addnodes.desc, desctype="data",
@@ -690,7 +691,8 @@ def test_pyattribute(app):
     assert_node(doctree[1][1][0], addnodes.index,
                 entries=[('single', 'attr (Class attribute)', 'Class.attr', '', None)])
     assert_node(doctree[1][1][1], ([desc_signature, ([desc_name, "attr"],
-                                                     [desc_annotation, ": str"],
+                                                     [desc_annotation, (": ",
+                                                                        [pending_xref, "str"])],
                                                      [desc_annotation, " = ''"])],
                                    [desc_content, ()]))
     assert 'Class.attr' in domain.objects

```


## Code snippets

### 1 - sphinx/domains/python.py:

Start line: 610, End line: 637

```python
class PyVariable(PyObject):
    """Description of a variable."""

    option_spec = PyObject.option_spec.copy()
    option_spec.update({
        'type': directives.unchanged,
        'value': directives.unchanged,
    })

    def handle_signature(self, sig: str, signode: desc_signature) -> Tuple[str, str]:
        fullname, prefix = super().handle_signature(sig, signode)

        typ = self.options.get('type')
        if typ:
            signode += addnodes.desc_annotation(typ, ': ' + typ)

        value = self.options.get('value')
        if value:
            signode += addnodes.desc_annotation(value, ' = ' + value)

        return fullname, prefix

    def get_index_text(self, modname: str, name_cls: Tuple[str, str]) -> str:
        name, cls = name_cls
        if modname:
            return _('%s (in module %s)') % (name, modname)
        else:
            return _('%s (built-in variable)') % name
```
### 2 - sphinx/util/typing.py:

Start line: 42, End line: 61

```python
def stringify(annotation: Any) -> str:
    """Stringify type annotation object."""
    if isinstance(annotation, str):
        return annotation
    elif isinstance(annotation, TypeVar):  # type: ignore
        return annotation.__name__
    elif not annotation:
        return repr(annotation)
    elif annotation is NoneType:  # type: ignore
        return 'None'
    elif (getattr(annotation, '__module__', None) == 'builtins' and
          hasattr(annotation, '__qualname__')):
        return annotation.__qualname__
    elif annotation is Ellipsis:
        return '...'

    if sys.version_info >= (3, 7):  # py37+
        return _stringify_py37(annotation)
    else:
        return _stringify_py36(annotation)
```
### 3 - sphinx/domains/python.py:

Start line: 978, End line: 996

```python
class PyCurrentModule(SphinxDirective):
    """
    This directive is just to tell Sphinx that we're documenting
    stuff in module foo, but links to module foo won't lead here.
    """

    has_content = False
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = False
    option_spec = {}  # type: Dict

    def run(self) -> List[Node]:
        modname = self.arguments[0].strip()
        if modname == 'None':
            self.env.ref_context.pop('py:module', None)
        else:
            self.env.ref_context['py:module'] = modname
        return []
```
### 4 - sphinx/util/typing.py:

Start line: 64, End line: 105

```python
def _stringify_py37(annotation: Any) -> str:
    """stringify() for py37+."""
    module = getattr(annotation, '__module__', None)
    if module == 'typing':
        if getattr(annotation, '_name', None):
            qualname = annotation._name
        elif getattr(annotation, '__qualname__', None):
            qualname = annotation.__qualname__
        elif getattr(annotation, '__forward_arg__', None):
            qualname = annotation.__forward_arg__
        else:
            qualname = stringify(annotation.__origin__)  # ex. Union
    elif hasattr(annotation, '__qualname__'):
        qualname = '%s.%s' % (module, annotation.__qualname__)
    elif hasattr(annotation, '__origin__'):
        # instantiated generic provided by a user
        qualname = stringify(annotation.__origin__)
    else:
        # we weren't able to extract the base type, appending arguments would
        # only make them appear twice
        return repr(annotation)

    if getattr(annotation, '__args__', None):
        if qualname == 'Union':
            if len(annotation.__args__) == 2 and annotation.__args__[1] is NoneType:  # type: ignore  # NOQA
                return 'Optional[%s]' % stringify(annotation.__args__[0])
            else:
                args = ', '.join(stringify(a) for a in annotation.__args__)
                return '%s[%s]' % (qualname, args)
        elif qualname == 'Callable':
            args = ', '.join(stringify(a) for a in annotation.__args__[:-1])
            returns = stringify(annotation.__args__[-1])
            return '%s[[%s], %s]' % (qualname, args, returns)
        elif str(annotation).startswith('typing.Annotated'):  # for py39+
            return stringify(annotation.__args__[0])
        elif getattr(annotation, '_special', False):
            return qualname
        else:
            args = ', '.join(stringify(a) for a in annotation.__args__)
            return '%s[%s]' % (qualname, args)

    return qualname
```
### 5 - sphinx/domains/python.py:

Start line: 240, End line: 258

```python
# This override allows our inline type specifiers to behave like :class: link
# when it comes to handling "." and "~" prefixes.
class PyXrefMixin:
    def make_xref(self, rolename: str, domain: str, target: str,
                  innernode: "Type[TextlikeNode]" = nodes.emphasis,
                  contnode: Node = None, env: BuildEnvironment = None) -> Node:
        result = super().make_xref(rolename, domain, target,  # type: ignore
                                   innernode, contnode, env)
        result['refspecific'] = True
        if target.startswith(('.', '~')):
            prefix, result['reftarget'] = target[0], target[1:]
            if prefix == '.':
                text = target[1:]
            elif prefix == '~':
                text = target.split('.')[-1]
            for node in result.traverse(nodes.Text):
                node.parent[node.parent.index(node)] = nodes.Text(text)
                break
        return result
```
### 6 - sphinx/domains/python.py:

Start line: 298, End line: 306

```python
class PyTypedField(PyXrefMixin, TypedField):
    def make_xref(self, rolename: str, domain: str, target: str,
                  innernode: "Type[TextlikeNode]" = nodes.emphasis,
                  contnode: Node = None, env: BuildEnvironment = None) -> Node:
        if rolename == 'class' and target == 'None':
            # None is not a type, so use obj role instead.
            rolename = 'obj'

        return super().make_xref(rolename, domain, target, innernode, contnode, env)
```
### 7 - sphinx/domains/std.py:

Start line: 621, End line: 643

```python
class StandardDomain(Domain):

    def note_hyperlink_target(self, name: str, docname: str, node_id: str,
                              title: str = '') -> None:
        """Add a hyperlink target for cross reference.

        .. warning::

           This is only for internal use.  Please don't use this from your extension.
           ``document.note_explicit_target()`` or ``note_implicit_target()`` are recommended to
           add a hyperlink target to the document.

           This only adds a hyperlink target to the StandardDomain.  And this does not add a
           node_id to node.  Therefore, it is very fragile to calling this without
           understanding hyperlink target framework in both docutils and Sphinx.

        .. versionadded:: 3.0
        """
        if name in self.anonlabels and self.anonlabels[name] != (docname, node_id):
            logger.warning(__('duplicate label %s, other instance in %s'),
                           name, self.env.doc2path(self.anonlabels[name][0]))

        self.anonlabels[name] = (docname, node_id)
        if title:
            self.labels[name] = (docname, node_id, title)
```
### 8 - sphinx/util/typing.py:

Start line: 108, End line: 125

```python
def _stringify_py36(annotation: Any) -> str:
    """stringify() for py35 and py36."""
    module = getattr(annotation, '__module__', None)
    if module == 'typing':
        if getattr(annotation, '_name', None):
            qualname = annotation._name
        elif getattr(annotation, '__qualname__', None):
            qualname = annotation.__qualname__
        elif getattr(annotation, '__forward_arg__', None):
            qualname = annotation.__forward_arg__
        elif getattr(annotation, '__origin__', None):
            qualname = stringify(annotation.__origin__)  # ex. Union
        else:
            qualname = repr(annotation).replace('typing.', '')
    elif hasattr(annotation, '__qualname__'):
        qualname = '%s.%s' % (module, annotation.__qualname__)
    else:
        qualname = repr(annotation)
    # ... other code
```
### 9 - sphinx/util/typing.py:

Start line: 127, End line: 198

```python
def _stringify_py36(annotation: Any) -> str:
    # ... other code

    if (isinstance(annotation, typing.TupleMeta) and  # type: ignore
            not hasattr(annotation, '__tuple_params__')):  # for Python 3.6
        params = annotation.__args__
        if params:
            param_str = ', '.join(stringify(p) for p in params)
            return '%s[%s]' % (qualname, param_str)
        else:
            return qualname
    elif isinstance(annotation, typing.GenericMeta):
        params = None
        if hasattr(annotation, '__args__'):
            # for Python 3.5.2+
            if annotation.__args__ is None or len(annotation.__args__) <= 2:  # type: ignore  # NOQA
                params = annotation.__args__  # type: ignore
            else:  # typing.Callable
                args = ', '.join(stringify(arg) for arg
                                 in annotation.__args__[:-1])  # type: ignore
                result = stringify(annotation.__args__[-1])  # type: ignore
                return '%s[[%s], %s]' % (qualname, args, result)
        elif hasattr(annotation, '__parameters__'):
            # for Python 3.5.0 and 3.5.1
            params = annotation.__parameters__  # type: ignore
        if params is not None:
            param_str = ', '.join(stringify(p) for p in params)
            return '%s[%s]' % (qualname, param_str)
    elif (hasattr(typing, 'UnionMeta') and
          isinstance(annotation, typing.UnionMeta) and  # type: ignore
          hasattr(annotation, '__union_params__')):  # for Python 3.5
        params = annotation.__union_params__
        if params is not None:
            if len(params) == 2 and params[1] is NoneType:  # type: ignore
                return 'Optional[%s]' % stringify(params[0])
            else:
                param_str = ', '.join(stringify(p) for p in params)
                return '%s[%s]' % (qualname, param_str)
    elif (hasattr(annotation, '__origin__') and
          annotation.__origin__ is typing.Union):  # for Python 3.5.2+
        params = annotation.__args__
        if params is not None:
            if len(params) == 2 and params[1] is NoneType:  # type: ignore
                return 'Optional[%s]' % stringify(params[0])
            else:
                param_str = ', '.join(stringify(p) for p in params)
                return 'Union[%s]' % param_str
    elif (isinstance(annotation, typing.CallableMeta) and  # type: ignore
          getattr(annotation, '__args__', None) is not None and
          hasattr(annotation, '__result__')):  # for Python 3.5
        # Skipped in the case of plain typing.Callable
        args = annotation.__args__
        if args is None:
            return qualname
        elif args is Ellipsis:
            args_str = '...'
        else:
            formatted_args = (stringify(a) for a in args)
            args_str = '[%s]' % ', '.join(formatted_args)
        return '%s[%s, %s]' % (qualname,
                               args_str,
                               stringify(annotation.__result__))
    elif (isinstance(annotation, typing.TupleMeta) and  # type: ignore
          hasattr(annotation, '__tuple_params__') and
          hasattr(annotation, '__tuple_use_ellipsis__')):  # for Python 3.5
        params = annotation.__tuple_params__
        if params is not None:
            param_strings = [stringify(p) for p in params]
            if annotation.__tuple_use_ellipsis__:
                param_strings.append('...')
            return '%s[%s]' % (qualname,
                               ', '.join(param_strings))

    return qualname
```
### 10 - sphinx/domains/rst.py:

Start line: 141, End line: 172

```python
class ReSTDirectiveOption(ReSTMarkup):

    def add_target_and_index(self, name: str, sig: str, signode: desc_signature) -> None:
        domain = cast(ReSTDomain, self.env.get_domain('rst'))

        directive_name = self.current_directive
        if directive_name:
            prefix = '-'.join([self.objtype, directive_name])
            objname = ':'.join([directive_name, name])
        else:
            prefix = self.objtype
            objname = name

        node_id = make_id(self.env, self.state.document, prefix, name)
        signode['ids'].append(node_id)

        # Assign old styled node_id not to break old hyperlinks (if possible)
        # Note: Will be removed in Sphinx-5.0 (RemovedInSphinx50Warning)
        old_node_id = self.make_old_id(name)
        if old_node_id not in self.state.document.ids and old_node_id not in signode['ids']:
            signode['ids'].append(old_node_id)

        self.state.document.note_explicit_target(signode)
        domain.note_object(self.objtype, objname, node_id, location=signode)

        if directive_name:
            key = name[0].upper()
            pair = [_('%s (directive)') % directive_name,
                    _(':%s: (directive option)') % name]
            self.indexnode['entries'].append(('pair', '; '.join(pair), node_id, '', key))
        else:
            key = name[0].upper()
            text = _(':%s: (directive option)') % name
            self.indexnode['entries'].append(('single', text, node_id, '', key))
```
### 11 - sphinx/domains/python.py:

Start line: 999, End line: 1019

```python
class PyXRefRole(XRefRole):
    def process_link(self, env: BuildEnvironment, refnode: Element,
                     has_explicit_title: bool, title: str, target: str) -> Tuple[str, str]:
        refnode['py:module'] = env.ref_context.get('py:module')
        refnode['py:class'] = env.ref_context.get('py:class')
        if not has_explicit_title:
            title = title.lstrip('.')    # only has a meaning for the target
            target = target.lstrip('~')  # only has a meaning for the title
            # if the first character is a tilde, don't display the module/class
            # parts of the contents
            if title[0:1] == '~':
                title = title[1:]
                dot = title.rfind('.')
                if dot != -1:
                    title = title[dot + 1:]
        # if the first character is a dot, search more specific namespaces first
        # else search builtins first
        if target[0:1] == '.':
            target = target[1:]
            refnode['refspecific'] = True
        return title, target
```
### 13 - sphinx/domains/python.py:

Start line: 915, End line: 975

```python
class PyModule(SphinxDirective):
    """
    Directive to mark description of a new module.
    """

    has_content = False
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = False
    option_spec = {
        'platform': lambda x: x,
        'synopsis': lambda x: x,
        'noindex': directives.flag,
        'deprecated': directives.flag,
    }

    def run(self) -> List[Node]:
        domain = cast(PythonDomain, self.env.get_domain('py'))

        modname = self.arguments[0].strip()
        noindex = 'noindex' in self.options
        self.env.ref_context['py:module'] = modname
        ret = []  # type: List[Node]
        if not noindex:
            # note module to the domain
            node_id = make_id(self.env, self.state.document, 'module', modname)
            target = nodes.target('', '', ids=[node_id], ismod=True)
            self.set_source_info(target)

            # Assign old styled node_id not to break old hyperlinks (if possible)
            # Note: Will removed in Sphinx-5.0  (RemovedInSphinx50Warning)
            old_node_id = self.make_old_id(modname)
            if node_id != old_node_id and old_node_id not in self.state.document.ids:
                target['ids'].append(old_node_id)

            self.state.document.note_explicit_target(target)

            domain.note_module(modname,
                               node_id,
                               self.options.get('synopsis', ''),
                               self.options.get('platform', ''),
                               'deprecated' in self.options)
            domain.note_object(modname, 'module', node_id, location=target)

            # the platform and synopsis aren't printed; in fact, they are only
            # used in the modindex currently
            ret.append(target)
            indextext = '%s; %s' % (pairindextypes['module'], modname)
            inode = addnodes.index(entries=[('pair', indextext, node_id, '', None)])
            ret.append(inode)
        return ret

    def make_old_id(self, name: str) -> str:
        """Generate old styled node_id.

        Old styled node_id is incompatible with docutils' node_id.
        It can contain dots and hyphens.

        .. note:: Old styled node_id was mainly used until Sphinx-3.0.
        """
        return 'module-%s' % name
```
### 15 - sphinx/domains/python.py:

Start line: 11, End line: 77

```python
import builtins
import inspect
import re
import typing
import warnings
from inspect import Parameter
from typing import Any, Dict, Iterable, Iterator, List, NamedTuple, Tuple
from typing import cast

from docutils import nodes
from docutils.nodes import Element, Node
from docutils.parsers.rst import directives

from sphinx import addnodes
from sphinx.addnodes import pending_xref, desc_signature
from sphinx.application import Sphinx
from sphinx.builders import Builder
from sphinx.deprecation import RemovedInSphinx40Warning, RemovedInSphinx50Warning
from sphinx.directives import ObjectDescription
from sphinx.domains import Domain, ObjType, Index, IndexEntry
from sphinx.environment import BuildEnvironment
from sphinx.locale import _, __
from sphinx.pycode.ast import ast, parse as ast_parse
from sphinx.roles import XRefRole
from sphinx.util import logging
from sphinx.util.docfields import Field, GroupedField, TypedField
from sphinx.util.docutils import SphinxDirective
from sphinx.util.inspect import signature_from_str
from sphinx.util.nodes import make_id, make_refnode
from sphinx.util.typing import TextlikeNode

if False:
    # For type annotation
    from typing import Type  # for python3.5.1


logger = logging.getLogger(__name__)


# REs for Python signatures
py_sig_re = re.compile(
    r'''^ ([\w.]*\.)?            # class name(s)
          (\w+)  \s*             # thing name
          (?: \(\s*(.*)\s*\)     # optional: arguments
           (?:\s* -> \s* (.*))?  #           return annotation
          )? $                   # and nothing more
          ''', re.VERBOSE)


pairindextypes = {
    'module':    _('module'),
    'keyword':   _('keyword'),
    'operator':  _('operator'),
    'object':    _('object'),
    'exception': _('exception'),
    'statement': _('statement'),
    'builtin':   _('built-in function'),
}

ObjectEntry = NamedTuple('ObjectEntry', [('docname', str),
                                         ('node_id', str),
                                         ('objtype', str)])
ModuleEntry = NamedTuple('ModuleEntry', [('docname', str),
                                         ('node_id', str),
                                         ('synopsis', str),
                                         ('platform', str),
                                         ('deprecated', bool)])
```
### 16 - sphinx/domains/python.py:

Start line: 80, End line: 139

```python
def _parse_annotation(annotation: str) -> List[Node]:
    """Parse type annotation."""
    def make_xref(text: str) -> addnodes.pending_xref:
        if text == 'None':
            reftype = 'obj'
        else:
            reftype = 'class'

        return pending_xref('', nodes.Text(text),
                            refdomain='py', reftype=reftype, reftarget=text)

    def unparse(node: ast.AST) -> List[Node]:
        if isinstance(node, ast.Attribute):
            return [nodes.Text("%s.%s" % (unparse(node.value)[0], node.attr))]
        elif isinstance(node, ast.Expr):
            return unparse(node.value)
        elif isinstance(node, ast.Index):
            return unparse(node.value)
        elif isinstance(node, ast.List):
            result = [addnodes.desc_sig_punctuation('', '[')]  # type: List[Node]
            for elem in node.elts:
                result.extend(unparse(elem))
                result.append(addnodes.desc_sig_punctuation('', ', '))
            result.pop()
            result.append(addnodes.desc_sig_punctuation('', ']'))
            return result
        elif isinstance(node, ast.Module):
            return sum((unparse(e) for e in node.body), [])
        elif isinstance(node, ast.Name):
            return [nodes.Text(node.id)]
        elif isinstance(node, ast.Subscript):
            result = unparse(node.value)
            result.append(addnodes.desc_sig_punctuation('', '['))
            result.extend(unparse(node.slice))
            result.append(addnodes.desc_sig_punctuation('', ']'))
            return result
        elif isinstance(node, ast.Tuple):
            if node.elts:
                result = []
                for elem in node.elts:
                    result.extend(unparse(elem))
                    result.append(addnodes.desc_sig_punctuation('', ', '))
                result.pop()
            else:
                result = [addnodes.desc_sig_punctuation('', '('),
                          addnodes.desc_sig_punctuation('', ')')]

            return result
        else:
            raise SyntaxError  # unsupported syntax

    try:
        tree = ast_parse(annotation)
        result = unparse(tree)
        for i, node in enumerate(result):
            if isinstance(node, nodes.Text):
                result[i] = make_xref(str(node))
        return result
    except SyntaxError:
        return [make_xref(annotation)]
```
### 19 - sphinx/domains/python.py:

Start line: 1110, End line: 1161

```python
class PythonDomain(Domain):
    """Python language domain."""
    name = 'py'
    label = 'Python'
    object_types = {
        'function':     ObjType(_('function'),      'func', 'obj'),
        'data':         ObjType(_('data'),          'data', 'obj'),
        'class':        ObjType(_('class'),         'class', 'exc', 'obj'),
        'exception':    ObjType(_('exception'),     'exc', 'class', 'obj'),
        'method':       ObjType(_('method'),        'meth', 'obj'),
        'classmethod':  ObjType(_('class method'),  'meth', 'obj'),
        'staticmethod': ObjType(_('static method'), 'meth', 'obj'),
        'attribute':    ObjType(_('attribute'),     'attr', 'obj'),
        'module':       ObjType(_('module'),        'mod', 'obj'),
    }  # type: Dict[str, ObjType]

    directives = {
        'function':        PyFunction,
        'data':            PyVariable,
        'class':           PyClasslike,
        'exception':       PyClasslike,
        'method':          PyMethod,
        'classmethod':     PyClassMethod,
        'staticmethod':    PyStaticMethod,
        'attribute':       PyAttribute,
        'module':          PyModule,
        'currentmodule':   PyCurrentModule,
        'decorator':       PyDecoratorFunction,
        'decoratormethod': PyDecoratorMethod,
    }
    roles = {
        'data':  PyXRefRole(),
        'exc':   PyXRefRole(),
        'func':  PyXRefRole(fix_parens=True),
        'class': PyXRefRole(),
        'const': PyXRefRole(),
        'attr':  PyXRefRole(),
        'meth':  PyXRefRole(fix_parens=True),
        'mod':   PyXRefRole(),
        'obj':   PyXRefRole(),
    }
    initial_data = {
        'objects': {},  # fullname -> docname, objtype
        'modules': {},  # modname -> docname, synopsis, platform, deprecated
    }  # type: Dict[str, Dict[str, Tuple[Any]]]
    indices = [
        PythonModuleIndex,
    ]

    @property
    def objects(self) -> Dict[str, ObjectEntry]:
        return self.data.setdefault('objects', {})  # fullname -> ObjectEntry
```
### 22 - sphinx/domains/python.py:

Start line: 855, End line: 875

```python
class PyAttribute(PyObject):
    """Description of an attribute."""

    option_spec = PyObject.option_spec.copy()
    option_spec.update({
        'type': directives.unchanged,
        'value': directives.unchanged,
    })

    def handle_signature(self, sig: str, signode: desc_signature) -> Tuple[str, str]:
        fullname, prefix = super().handle_signature(sig, signode)

        typ = self.options.get('type')
        if typ:
            signode += addnodes.desc_annotation(typ, ': ' + typ)

        value = self.options.get('value')
        if value:
            signode += addnodes.desc_annotation(value, ' = ' + value)

        return fullname, prefix
```
### 26 - sphinx/domains/python.py:

Start line: 697, End line: 751

```python
class PyClassmember(PyObject):

    def get_index_text(self, modname: str, name_cls: Tuple[str, str]) -> str:
        name, cls = name_cls
        add_modules = self.env.config.add_module_names
        if self.objtype == 'method':
            try:
                clsname, methname = name.rsplit('.', 1)
            except ValueError:
                if modname:
                    return _('%s() (in module %s)') % (name, modname)
                else:
                    return '%s()' % name
            if modname and add_modules:
                return _('%s() (%s.%s method)') % (methname, modname, clsname)
            else:
                return _('%s() (%s method)') % (methname, clsname)
        elif self.objtype == 'staticmethod':
            try:
                clsname, methname = name.rsplit('.', 1)
            except ValueError:
                if modname:
                    return _('%s() (in module %s)') % (name, modname)
                else:
                    return '%s()' % name
            if modname and add_modules:
                return _('%s() (%s.%s static method)') % (methname, modname,
                                                          clsname)
            else:
                return _('%s() (%s static method)') % (methname, clsname)
        elif self.objtype == 'classmethod':
            try:
                clsname, methname = name.rsplit('.', 1)
            except ValueError:
                if modname:
                    return _('%s() (in module %s)') % (name, modname)
                else:
                    return '%s()' % name
            if modname:
                return _('%s() (%s.%s class method)') % (methname, modname,
                                                         clsname)
            else:
                return _('%s() (%s class method)') % (methname, clsname)
        elif self.objtype == 'attribute':
            try:
                clsname, attrname = name.rsplit('.', 1)
            except ValueError:
                if modname:
                    return _('%s (in module %s)') % (name, modname)
                else:
                    return name
            if modname and add_modules:
                return _('%s (%s.%s attribute)') % (attrname, modname, clsname)
            else:
                return _('%s (%s attribute)') % (attrname, clsname)
        else:
            return ''
```
### 28 - sphinx/domains/python.py:

Start line: 877, End line: 889

```python
class PyAttribute(PyObject):

    def get_index_text(self, modname: str, name_cls: Tuple[str, str]) -> str:
        name, cls = name_cls
        try:
            clsname, attrname = name.rsplit('.', 1)
            if modname and self.env.config.add_module_names:
                clsname = '.'.join([modname, clsname])
        except ValueError:
            if modname:
                return _('%s (in module %s)') % (name, modname)
            else:
                return name

        return _('%s (%s attribute)') % (attrname, clsname)
```
### 41 - sphinx/domains/python.py:

Start line: 309, End line: 353

```python
class PyObject(ObjectDescription):
    """
    Description of a general Python object.

    :cvar allow_nesting: Class is an object that allows for nested namespaces
    :vartype allow_nesting: bool
    """
    option_spec = {
        'noindex': directives.flag,
        'module': directives.unchanged,
        'annotation': directives.unchanged,
    }

    doc_field_types = [
        PyTypedField('parameter', label=_('Parameters'),
                     names=('param', 'parameter', 'arg', 'argument',
                            'keyword', 'kwarg', 'kwparam'),
                     typerolename='class', typenames=('paramtype', 'type'),
                     can_collapse=True),
        PyTypedField('variable', label=_('Variables'), rolename='obj',
                     names=('var', 'ivar', 'cvar'),
                     typerolename='class', typenames=('vartype',),
                     can_collapse=True),
        PyGroupedField('exceptions', label=_('Raises'), rolename='exc',
                       names=('raises', 'raise', 'exception', 'except'),
                       can_collapse=True),
        Field('returnvalue', label=_('Returns'), has_arg=False,
              names=('returns', 'return')),
        PyField('returntype', label=_('Return type'), has_arg=False,
                names=('rtype',), bodyrolename='class'),
    ]

    allow_nesting = False

    def get_signature_prefix(self, sig: str) -> str:
        """May return a prefix to put before the object name in the
        signature.
        """
        return ''

    def needs_arglist(self) -> bool:
        """May return true if an empty argument list is to be generated even if
        the document contains none.
        """
        return False
```
### 44 - sphinx/domains/python.py:

Start line: 754, End line: 791

```python
class PyMethod(PyObject):
    """Description of a method."""

    option_spec = PyObject.option_spec.copy()
    option_spec.update({
        'abstractmethod': directives.flag,
        'async': directives.flag,
        'classmethod': directives.flag,
        'final': directives.flag,
        'property': directives.flag,
        'staticmethod': directives.flag,
    })

    def needs_arglist(self) -> bool:
        if 'property' in self.options:
            return False
        else:
            return True

    def get_signature_prefix(self, sig: str) -> str:
        prefix = []
        if 'final' in self.options:
            prefix.append('final')
        if 'abstractmethod' in self.options:
            prefix.append('abstract')
        if 'async' in self.options:
            prefix.append('async')
        if 'classmethod' in self.options:
            prefix.append('classmethod')
        if 'property' in self.options:
            prefix.append('property')
        if 'staticmethod' in self.options:
            prefix.append('static')

        if prefix:
            return ' '.join(prefix) + ' '
        else:
            return ''
```
### 48 - sphinx/domains/python.py:

Start line: 1163, End line: 1173

```python
class PythonDomain(Domain):

    def note_object(self, name: str, objtype: str, node_id: str, location: Any = None) -> None:
        """Note a python object for cross reference.

        .. versionadded:: 2.1
        """
        if name in self.objects:
            other = self.objects[name]
            logger.warning(__('duplicate object description of %s, '
                              'other instance in %s, use :noindex: for one of them'),
                           name, other.docname, location=location)
        self.objects[name] = ObjectEntry(self.env.docname, node_id, objtype)
```
### 50 - sphinx/domains/python.py:

Start line: 283, End line: 295

```python
class PyField(PyXrefMixin, Field):
    def make_xref(self, rolename: str, domain: str, target: str,
                  innernode: "Type[TextlikeNode]" = nodes.emphasis,
                  contnode: Node = None, env: BuildEnvironment = None) -> Node:
        if rolename == 'class' and target == 'None':
            # None is not a type, so use obj role instead.
            rolename = 'obj'

        return super().make_xref(rolename, domain, target, innernode, contnode, env)


class PyGroupedField(PyXrefMixin, GroupedField):
    pass
```
### 51 - sphinx/domains/python.py:

Start line: 557, End line: 590

```python
class PyFunction(PyObject):
    """Description of a function."""

    option_spec = PyObject.option_spec.copy()
    option_spec.update({
        'async': directives.flag,
    })

    def get_signature_prefix(self, sig: str) -> str:
        if 'async' in self.options:
            return 'async '
        else:
            return ''

    def needs_arglist(self) -> bool:
        return True

    def add_target_and_index(self, name_cls: Tuple[str, str], sig: str,
                             signode: desc_signature) -> None:
        super().add_target_and_index(name_cls, sig, signode)
        modname = self.options.get('module', self.env.ref_context.get('py:module'))
        node_id = signode['ids'][0]

        name, cls = name_cls
        if modname:
            text = _('%s() (in module %s)') % (name, modname)
            self.indexnode['entries'].append(('single', text, node_id, '', None))
        else:
            text = '%s; %s()' % (pairindextypes['builtin'], name)
            self.indexnode['entries'].append(('pair', text, node_id, '', None))

    def get_index_text(self, modname: str, name_cls: Tuple[str, str]) -> str:
        # add index in own add_target_and_index() instead.
        return None
```
### 58 - sphinx/domains/python.py:

Start line: 815, End line: 852

```python
class PyClassMethod(PyMethod):
    """Description of a classmethod."""

    option_spec = PyObject.option_spec.copy()

    def run(self) -> List[Node]:
        self.name = 'py:method'
        self.options['classmethod'] = True

        return super().run()


class PyStaticMethod(PyMethod):
    """Description of a staticmethod."""

    option_spec = PyObject.option_spec.copy()

    def run(self) -> List[Node]:
        self.name = 'py:method'
        self.options['staticmethod'] = True

        return super().run()


class PyDecoratorMethod(PyMethod):
    """Description of a decoratormethod."""

    def run(self) -> List[Node]:
        self.name = 'py:method'
        return super().run()

    def handle_signature(self, sig: str, signode: desc_signature) -> Tuple[str, str]:
        ret = super().handle_signature(sig, signode)
        signode.insert(0, addnodes.desc_addname('@', '@'))
        return ret

    def needs_arglist(self) -> bool:
        return False
```
### 59 - sphinx/domains/python.py:

Start line: 1175, End line: 1203

```python
class PythonDomain(Domain):

    @property
    def modules(self) -> Dict[str, ModuleEntry]:
        return self.data.setdefault('modules', {})  # modname -> ModuleEntry

    def note_module(self, name: str, node_id: str, synopsis: str,
                    platform: str, deprecated: bool) -> None:
        """Note a python module for cross reference.

        .. versionadded:: 2.1
        """
        self.modules[name] = ModuleEntry(self.env.docname, node_id,
                                         synopsis, platform, deprecated)

    def clear_doc(self, docname: str) -> None:
        for fullname, obj in list(self.objects.items()):
            if obj.docname == docname:
                del self.objects[fullname]
        for modname, mod in list(self.modules.items()):
            if mod.docname == docname:
                del self.modules[modname]

    def merge_domaindata(self, docnames: List[str], otherdata: Dict) -> None:
        # XXX check duplicates?
        for fullname, obj in otherdata['objects'].items():
            if obj.docname in docnames:
                self.objects[fullname] = obj
        for modname, mod in otherdata['modules'].items():
            if mod.docname in docnames:
                self.modules[modname] = mod
```
### 62 - sphinx/domains/python.py:

Start line: 544, End line: 554

```python
class PyModulelevel(PyObject):

    def get_index_text(self, modname: str, name_cls: Tuple[str, str]) -> str:
        if self.objtype == 'function':
            if not modname:
                return _('%s() (built-in function)') % name_cls[0]
            return _('%s() (in module %s)') % (name_cls[0], modname)
        elif self.objtype == 'data':
            if not modname:
                return _('%s (built-in variable)') % name_cls[0]
            return _('%s (in module %s)') % (name_cls[0], modname)
        else:
            return ''
```
### 69 - sphinx/domains/python.py:

Start line: 1307, End line: 1319

```python
class PythonDomain(Domain):

    def _make_module_refnode(self, builder: Builder, fromdocname: str, name: str,
                             contnode: Node) -> Element:
        # get additional info for modules
        module = self.modules[name]
        title = name
        if module.synopsis:
            title += ': ' + module.synopsis
        if module.deprecated:
            title += _(' (deprecated)')
        if module.platform:
            title += ' (' + module.platform + ')'
        return make_refnode(builder, fromdocname, module.docname, module.node_id,
                            contnode, title)
```
