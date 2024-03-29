# sphinx-doc__sphinx-9931

| **sphinx-doc/sphinx** | `17dfa811078205bd415700361e97e945112b89eb` |
| ---- | ---- |
| **No of patches** | 4 |
| **All found context length** | 9789 |
| **Any found context length** | 3180 |
| **Avg pos** | 31.5 |
| **Min pos** | 1 |
| **Max pos** | 37 |
| **Top file pos** | 1 |
| **Missing snippets** | 22 |
| **Missing patch files** | 1 |


## Expected patch

```diff
diff --git a/sphinx/domains/python.py b/sphinx/domains/python.py
--- a/sphinx/domains/python.py
+++ b/sphinx/domains/python.py
@@ -80,7 +80,8 @@ class ModuleEntry(NamedTuple):
     deprecated: bool
 
 
-def type_to_xref(target: str, env: BuildEnvironment = None) -> addnodes.pending_xref:
+def type_to_xref(target: str, env: BuildEnvironment = None, suppress_prefix: bool = False
+                 ) -> addnodes.pending_xref:
     """Convert a type string to a cross reference node."""
     if target == 'None':
         reftype = 'obj'
@@ -101,6 +102,8 @@ def type_to_xref(target: str, env: BuildEnvironment = None) -> addnodes.pending_
     elif target.startswith('~'):
         target = target[1:]
         text = target.split('.')[-1]
+    elif suppress_prefix:
+        text = target.split('.')[-1]
     else:
         text = target
 
@@ -150,6 +153,8 @@ def unparse(node: ast.AST) -> List[Node]:
             return unparse(node.value)
         elif isinstance(node, ast.Index):
             return unparse(node.value)
+        elif isinstance(node, ast.Invert):
+            return [addnodes.desc_sig_punctuation('', '~')]
         elif isinstance(node, ast.List):
             result = [addnodes.desc_sig_punctuation('', '[')]
             if node.elts:
@@ -180,6 +185,8 @@ def unparse(node: ast.AST) -> List[Node]:
                     if isinstance(subnode, nodes.Text):
                         result[i] = nodes.literal('', '', subnode)
             return result
+        elif isinstance(node, ast.UnaryOp):
+            return unparse(node.op) + unparse(node.operand)
         elif isinstance(node, ast.Tuple):
             if node.elts:
                 result = []
@@ -209,12 +216,19 @@ def unparse(node: ast.AST) -> List[Node]:
 
     try:
         tree = ast_parse(annotation)
-        result = unparse(tree)
-        for i, node in enumerate(result):
+        result: List[Node] = []
+        for node in unparse(tree):
             if isinstance(node, nodes.literal):
-                result[i] = node[0]
+                result.append(node[0])
             elif isinstance(node, nodes.Text) and node.strip():
-                result[i] = type_to_xref(str(node), env)
+                if (result and isinstance(result[-1], addnodes.desc_sig_punctuation) and
+                        result[-1].astext() == '~'):
+                    result.pop()
+                    result.append(type_to_xref(str(node), env, suppress_prefix=True))
+                else:
+                    result.append(type_to_xref(str(node), env))
+            else:
+                result.append(node)
         return result
     except SyntaxError:
         return [type_to_xref(annotation, env)]
diff --git a/sphinx/ext/autodoc/__init__.py b/sphinx/ext/autodoc/__init__.py
--- a/sphinx/ext/autodoc/__init__.py
+++ b/sphinx/ext/autodoc/__init__.py
@@ -1295,6 +1295,8 @@ def can_document_member(cls, member: Any, membername: str, isattr: bool, parent:
     def format_args(self, **kwargs: Any) -> str:
         if self.config.autodoc_typehints in ('none', 'description'):
             kwargs.setdefault('show_annotation', False)
+        if self.config.autodoc_unqualified_typehints:
+            kwargs.setdefault('unqualified_typehints', True)
 
         try:
             self.env.app.emit('autodoc-before-process-signature', self.object, False)
@@ -1323,6 +1325,9 @@ def add_directive_header(self, sig: str) -> None:
             self.add_line('   :async:', sourcename)
 
     def format_signature(self, **kwargs: Any) -> str:
+        if self.config.autodoc_unqualified_typehints:
+            kwargs.setdefault('unqualified_typehints', True)
+
         sigs = []
         if (self.analyzer and
                 '.'.join(self.objpath) in self.analyzer.overloads and
@@ -1561,6 +1566,8 @@ def get_user_defined_function_or_method(obj: Any, attr: str) -> Any:
     def format_args(self, **kwargs: Any) -> str:
         if self.config.autodoc_typehints in ('none', 'description'):
             kwargs.setdefault('show_annotation', False)
+        if self.config.autodoc_unqualified_typehints:
+            kwargs.setdefault('unqualified_typehints', True)
 
         try:
             self._signature_class, self._signature_method_name, sig = self._get_signature()
@@ -1582,6 +1589,9 @@ def format_signature(self, **kwargs: Any) -> str:
             # do not show signatures
             return ''
 
+        if self.config.autodoc_unqualified_typehints:
+            kwargs.setdefault('unqualified_typehints', True)
+
         sig = super().format_signature()
         sigs = []
 
@@ -2110,6 +2120,8 @@ def import_object(self, raiseerror: bool = False) -> bool:
     def format_args(self, **kwargs: Any) -> str:
         if self.config.autodoc_typehints in ('none', 'description'):
             kwargs.setdefault('show_annotation', False)
+        if self.config.autodoc_unqualified_typehints:
+            kwargs.setdefault('unqualified_typehints', True)
 
         try:
             if self.object == object.__init__ and self.parent != object:
@@ -2160,6 +2172,9 @@ def document_members(self, all_members: bool = False) -> None:
         pass
 
     def format_signature(self, **kwargs: Any) -> str:
+        if self.config.autodoc_unqualified_typehints:
+            kwargs.setdefault('unqualified_typehints', True)
+
         sigs = []
         if (self.analyzer and
                 '.'.join(self.objpath) in self.analyzer.overloads and
@@ -2833,6 +2848,7 @@ def setup(app: Sphinx) -> Dict[str, Any]:
     app.add_config_value('autodoc_typehints_description_target', 'all', True,
                          ENUM('all', 'documented'))
     app.add_config_value('autodoc_type_aliases', {}, True)
+    app.add_config_value('autodoc_unqualified_typehints', False, 'env')
     app.add_config_value('autodoc_warningiserror', True, True)
     app.add_config_value('autodoc_inherit_docstrings', True, True)
     app.add_event('autodoc-before-process-signature')
diff --git a/sphinx/util/inspect.py b/sphinx/util/inspect.py
--- a/sphinx/util/inspect.py
+++ b/sphinx/util/inspect.py
@@ -744,10 +744,13 @@ def evaluate(annotation: Any, globalns: Dict, localns: Dict) -> Any:
 
 
 def stringify_signature(sig: inspect.Signature, show_annotation: bool = True,
-                        show_return_annotation: bool = True) -> str:
+                        show_return_annotation: bool = True,
+                        unqualified_typehints: bool = False) -> str:
     """Stringify a Signature object.
 
     :param show_annotation: Show annotation in result
+    :param unqualified_typehints: Show annotations as unqualified
+                                  (ex. io.StringIO -> StringIO)
     """
     args = []
     last_kind = None
@@ -771,7 +774,7 @@ def stringify_signature(sig: inspect.Signature, show_annotation: bool = True,
 
         if show_annotation and param.annotation is not param.empty:
             arg.write(': ')
-            arg.write(stringify_annotation(param.annotation))
+            arg.write(stringify_annotation(param.annotation, unqualified_typehints))
         if param.default is not param.empty:
             if show_annotation and param.annotation is not param.empty:
                 arg.write(' = ')
@@ -791,7 +794,7 @@ def stringify_signature(sig: inspect.Signature, show_annotation: bool = True,
             show_return_annotation is False):
         return '(%s)' % ', '.join(args)
     else:
-        annotation = stringify_annotation(sig.return_annotation)
+        annotation = stringify_annotation(sig.return_annotation, unqualified_typehints)
         return '(%s) -> %s' % (', '.join(args), annotation)
 
 
diff --git a/sphinx/util/typing.py b/sphinx/util/typing.py
--- a/sphinx/util/typing.py
+++ b/sphinx/util/typing.py
@@ -299,10 +299,19 @@ def _restify_py36(cls: Optional[Type]) -> str:
             return ':py:obj:`%s.%s`' % (cls.__module__, qualname)
 
 
-def stringify(annotation: Any) -> str:
-    """Stringify type annotation object."""
+def stringify(annotation: Any, smartref: bool = False) -> str:
+    """Stringify type annotation object.
+
+    :param smartref: If true, add "~" prefix to the result to remove the leading
+                     module and class names from the reference text
+    """
     from sphinx.util import inspect  # lazy loading
 
+    if smartref:
+        prefix = '~'
+    else:
+        prefix = ''
+
     if isinstance(annotation, str):
         if annotation.startswith("'") and annotation.endswith("'"):
             # might be a double Forward-ref'ed type.  Go unquoting.
@@ -313,11 +322,11 @@ def stringify(annotation: Any) -> str:
         if annotation.__module__ == 'typing':
             return annotation.__name__
         else:
-            return '.'.join([annotation.__module__, annotation.__name__])
+            return prefix + '.'.join([annotation.__module__, annotation.__name__])
     elif inspect.isNewType(annotation):
         if sys.version_info > (3, 10):
             # newtypes have correct module info since Python 3.10+
-            return '%s.%s' % (annotation.__module__, annotation.__name__)
+            return prefix + '%s.%s' % (annotation.__module__, annotation.__name__)
         else:
             return annotation.__name__
     elif not annotation:
@@ -325,7 +334,7 @@ def stringify(annotation: Any) -> str:
     elif annotation is NoneType:
         return 'None'
     elif annotation in INVALID_BUILTIN_CLASSES:
-        return INVALID_BUILTIN_CLASSES[annotation]
+        return prefix + INVALID_BUILTIN_CLASSES[annotation]
     elif str(annotation).startswith('typing.Annotated'):  # for py310+
         pass
     elif (getattr(annotation, '__module__', None) == 'builtins' and
@@ -338,28 +347,36 @@ def stringify(annotation: Any) -> str:
         return '...'
 
     if sys.version_info >= (3, 7):  # py37+
-        return _stringify_py37(annotation)
+        return _stringify_py37(annotation, smartref)
     else:
-        return _stringify_py36(annotation)
+        return _stringify_py36(annotation, smartref)
 
 
-def _stringify_py37(annotation: Any) -> str:
+def _stringify_py37(annotation: Any, smartref: bool = False) -> str:
     """stringify() for py37+."""
     module = getattr(annotation, '__module__', None)
-    if module == 'typing':
+    modprefix = ''
+    if module == 'typing' and getattr(annotation, '__forward_arg__', None):
+        qualname = annotation.__forward_arg__
+    elif module == 'typing':
         if getattr(annotation, '_name', None):
             qualname = annotation._name
         elif getattr(annotation, '__qualname__', None):
             qualname = annotation.__qualname__
-        elif getattr(annotation, '__forward_arg__', None):
-            qualname = annotation.__forward_arg__
         else:
             qualname = stringify(annotation.__origin__)  # ex. Union
+
+        if smartref:
+            modprefix = '~%s.' % module
     elif hasattr(annotation, '__qualname__'):
-        qualname = '%s.%s' % (module, annotation.__qualname__)
+        if smartref:
+            modprefix = '~%s.' % module
+        else:
+            modprefix = '%s.' % module
+        qualname = annotation.__qualname__
     elif hasattr(annotation, '__origin__'):
         # instantiated generic provided by a user
-        qualname = stringify(annotation.__origin__)
+        qualname = stringify(annotation.__origin__, smartref)
     elif UnionType and isinstance(annotation, UnionType):  # types.Union (for py3.10+)
         qualname = 'types.Union'
     else:
@@ -374,54 +391,63 @@ def _stringify_py37(annotation: Any) -> str:
         elif qualname in ('Optional', 'Union'):
             if len(annotation.__args__) > 1 and annotation.__args__[-1] is NoneType:
                 if len(annotation.__args__) > 2:
-                    args = ', '.join(stringify(a) for a in annotation.__args__[:-1])
-                    return 'Optional[Union[%s]]' % args
+                    args = ', '.join(stringify(a, smartref) for a in annotation.__args__[:-1])
+                    return '%sOptional[%sUnion[%s]]' % (modprefix, modprefix, args)
                 else:
-                    return 'Optional[%s]' % stringify(annotation.__args__[0])
+                    return '%sOptional[%s]' % (modprefix,
+                                               stringify(annotation.__args__[0], smartref))
             else:
-                args = ', '.join(stringify(a) for a in annotation.__args__)
-                return 'Union[%s]' % args
+                args = ', '.join(stringify(a, smartref) for a in annotation.__args__)
+                return '%sUnion[%s]' % (modprefix, args)
         elif qualname == 'types.Union':
             if len(annotation.__args__) > 1 and None in annotation.__args__:
                 args = ' | '.join(stringify(a) for a in annotation.__args__ if a)
-                return 'Optional[%s]' % args
+                return '%sOptional[%s]' % (modprefix, args)
             else:
                 return ' | '.join(stringify(a) for a in annotation.__args__)
         elif qualname == 'Callable':
-            args = ', '.join(stringify(a) for a in annotation.__args__[:-1])
-            returns = stringify(annotation.__args__[-1])
-            return '%s[[%s], %s]' % (qualname, args, returns)
+            args = ', '.join(stringify(a, smartref) for a in annotation.__args__[:-1])
+            returns = stringify(annotation.__args__[-1], smartref)
+            return '%s%s[[%s], %s]' % (modprefix, qualname, args, returns)
         elif qualname == 'Literal':
             args = ', '.join(repr(a) for a in annotation.__args__)
-            return '%s[%s]' % (qualname, args)
+            return '%s%s[%s]' % (modprefix, qualname, args)
         elif str(annotation).startswith('typing.Annotated'):  # for py39+
-            return stringify(annotation.__args__[0])
+            return stringify(annotation.__args__[0], smartref)
         elif all(is_system_TypeVar(a) for a in annotation.__args__):
             # Suppress arguments if all system defined TypeVars (ex. Dict[KT, VT])
-            return qualname
+            return modprefix + qualname
         else:
-            args = ', '.join(stringify(a) for a in annotation.__args__)
-            return '%s[%s]' % (qualname, args)
+            args = ', '.join(stringify(a, smartref) for a in annotation.__args__)
+            return '%s%s[%s]' % (modprefix, qualname, args)
 
-    return qualname
+    return modprefix + qualname
 
 
-def _stringify_py36(annotation: Any) -> str:
+def _stringify_py36(annotation: Any, smartref: bool = False) -> str:
     """stringify() for py36."""
     module = getattr(annotation, '__module__', None)
-    if module == 'typing':
+    modprefix = ''
+    if module == 'typing' and getattr(annotation, '__forward_arg__', None):
+        qualname = annotation.__forward_arg__
+    elif module == 'typing':
         if getattr(annotation, '_name', None):
             qualname = annotation._name
         elif getattr(annotation, '__qualname__', None):
             qualname = annotation.__qualname__
-        elif getattr(annotation, '__forward_arg__', None):
-            qualname = annotation.__forward_arg__
         elif getattr(annotation, '__origin__', None):
             qualname = stringify(annotation.__origin__)  # ex. Union
         else:
             qualname = repr(annotation).replace('typing.', '')
+
+        if smartref:
+            modprefix = '~%s.' % module
     elif hasattr(annotation, '__qualname__'):
-        qualname = '%s.%s' % (module, annotation.__qualname__)
+        if smartref:
+            modprefix = '~%s.' % module
+        else:
+            modprefix = '%s.' % module
+        qualname = annotation.__qualname__
     else:
         qualname = repr(annotation)
 
@@ -429,10 +455,10 @@ def _stringify_py36(annotation: Any) -> str:
             not hasattr(annotation, '__tuple_params__')):  # for Python 3.6
         params = annotation.__args__
         if params:
-            param_str = ', '.join(stringify(p) for p in params)
-            return '%s[%s]' % (qualname, param_str)
+            param_str = ', '.join(stringify(p, smartref) for p in params)
+            return '%s%s[%s]' % (modprefix, qualname, param_str)
         else:
-            return qualname
+            return modprefix + qualname
     elif isinstance(annotation, typing.GenericMeta):
         params = None
         if annotation.__args__ is None or len(annotation.__args__) <= 2:  # type: ignore  # NOQA
@@ -440,28 +466,28 @@ def _stringify_py36(annotation: Any) -> str:
         elif annotation.__origin__ == Generator:  # type: ignore
             params = annotation.__args__  # type: ignore
         else:  # typing.Callable
-            args = ', '.join(stringify(arg) for arg
+            args = ', '.join(stringify(arg, smartref) for arg
                              in annotation.__args__[:-1])  # type: ignore
             result = stringify(annotation.__args__[-1])  # type: ignore
-            return '%s[[%s], %s]' % (qualname, args, result)
+            return '%s%s[[%s], %s]' % (modprefix, qualname, args, result)
         if params is not None:
-            param_str = ', '.join(stringify(p) for p in params)
-            return '%s[%s]' % (qualname, param_str)
+            param_str = ', '.join(stringify(p, smartref) for p in params)
+            return '%s%s[%s]' % (modprefix, qualname, param_str)
     elif (hasattr(annotation, '__origin__') and
           annotation.__origin__ is typing.Union):
         params = annotation.__args__
         if params is not None:
             if len(params) > 1 and params[-1] is NoneType:
                 if len(params) > 2:
-                    param_str = ", ".join(stringify(p) for p in params[:-1])
-                    return 'Optional[Union[%s]]' % param_str
+                    param_str = ", ".join(stringify(p, smartref) for p in params[:-1])
+                    return '%sOptional[%sUnion[%s]]' % (modprefix, modprefix, param_str)
                 else:
-                    return 'Optional[%s]' % stringify(params[0])
+                    return '%sOptional[%s]' % (modprefix, stringify(params[0]))
             else:
-                param_str = ', '.join(stringify(p) for p in params)
-                return 'Union[%s]' % param_str
+                param_str = ', '.join(stringify(p, smartref) for p in params)
+                return '%sUnion[%s]' % (modprefix, param_str)
 
-    return qualname
+    return modprefix + qualname
 
 
 deprecated_alias('sphinx.util.typing',

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sphinx/domains/python.py | 83 | 83 | - | 8 | -
| sphinx/domains/python.py | 104 | 104 | - | 8 | -
| sphinx/domains/python.py | 153 | 153 | - | 8 | -
| sphinx/domains/python.py | 183 | 183 | - | 8 | -
| sphinx/domains/python.py | 212 | 217 | - | 8 | -
| sphinx/ext/autodoc/__init__.py | 1298 | 1298 | 8 | 1 | 6934
| sphinx/ext/autodoc/__init__.py | 1326 | 1326 | 8 | 1 | 6934
| sphinx/ext/autodoc/__init__.py | 1564 | 1564 | 37 | 1 | 17877
| sphinx/ext/autodoc/__init__.py | 1585 | 1585 | 37 | 1 | 17877
| sphinx/ext/autodoc/__init__.py | 2113 | 2113 | 14 | 1 | 9789
| sphinx/ext/autodoc/__init__.py | 2163 | 2163 | 14 | 1 | 9789
| sphinx/ext/autodoc/__init__.py | 2836 | 2836 | 6 | 1 | 5198
| sphinx/util/inspect.py | 747 | 747 | - | 4 | -
| sphinx/util/inspect.py | 774 | 774 | - | 4 | -
| sphinx/util/inspect.py | 794 | 794 | - | 4 | -
| sphinx/util/typing.py | 302 | 303 | - | - | -
| sphinx/util/typing.py | 316 | 320 | - | - | -
| sphinx/util/typing.py | 328 | 328 | - | - | -
| sphinx/util/typing.py | 341 | 362 | - | - | -
| sphinx/util/typing.py | 377 | 424 | - | - | -
| sphinx/util/typing.py | 432 | 435 | - | - | -
| sphinx/util/typing.py | 443 | 464 | - | - | -


## Problem Statement

```
autodoc add_module_names equivalent for arguments
The `add_module_names = False` configuration seems to only affect the class/function/attribute header names.
The type hints are still always rendered as fully qualified names.

`mypackage/mymodule.py`:
\`\`\`python
class MyClass:
    """Whatever 1."""
    pass


def foo(arg: MyClass):
    """Whatever 2."""
    pass
\`\`\`

`conf.py`:
\`\`\`python
# ...
add_module_names = False
# ...
\`\`\`

`index.rst`:
\`\`\`rst
mypackage.mymodule module
=========================

.. automodule:: mypackage.mymodule
   :members:
   :undoc-members:
   :show-inheritance:
\`\`\`

Expected documentation:
\`\`\`
foo(arg: MyClass)
    Whatever 2.
\`\`\`

Actual documentation:
\`\`\`
foo(arg: mypackage.mymodule.MyClass)
    Whatever 2.
\`\`\`

## Describe the solution you'd like

I would be OK with any of the following:
\`\`\`python
add_module_names = False # now affects type annotations too
# or
add_type_module_names = False # new sphinx config option (name up for debate)
# or
autodoc_add_module_names = False # new autodoc config option (name up for debate)
\`\`\`

## Describe alternatives you've considered

There's a [StackOverflow post](https://stackoverflow.com/questions/51394955/sphinx-remove-module-prefix-for-args-in-automodule) which suggests using the `autodoc_docstring_signature` option to manually specify the function signature. This is not really a viable solution in my opinion.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 sphinx/ext/autodoc/__init__.py** | 1434 | 1784| 3180 | 3180 | 24008 | 
| 2 | **1 sphinx/ext/autodoc/__init__.py** | 448 | 496| 359 | 3539 | 24008 | 
| 3 | **1 sphinx/ext/autodoc/__init__.py** | 13 | 114| 788 | 4327 | 24008 | 
| 4 | 2 sphinx/ext/autodoc/mock.py | 73 | 98| 217 | 4544 | 25393 | 
| 5 | **2 sphinx/ext/autodoc/__init__.py** | 1045 | 1056| 126 | 4670 | 25393 | 
| **-> 6 <-** | **2 sphinx/ext/autodoc/__init__.py** | 2808 | 2851| 528 | 5198 | 25393 | 
| 7 | 3 doc/conf.py | 1 | 82| 732 | 5930 | 26858 | 
| **-> 8 <-** | **3 sphinx/ext/autodoc/__init__.py** | 1281 | 1400| 1004 | 6934 | 26858 | 
| 9 | **3 sphinx/ext/autodoc/__init__.py** | 1994 | 2019| 268 | 7202 | 26858 | 
| 10 | **3 sphinx/ext/autodoc/__init__.py** | 377 | 386| 121 | 7323 | 26858 | 
| 11 | **3 sphinx/ext/autodoc/__init__.py** | 2587 | 2610| 229 | 7552 | 26858 | 
| 12 | **3 sphinx/ext/autodoc/__init__.py** | 1031 | 1043| 124 | 7676 | 26858 | 
| 13 | **4 sphinx/util/inspect.py** | 559 | 589| 250 | 7926 | 33866 | 
| **-> 14 <-** | **4 sphinx/ext/autodoc/__init__.py** | 2079 | 2284| 1863 | 9789 | 33866 | 
| 15 | **4 sphinx/ext/autodoc/__init__.py** | 1085 | 1109| 209 | 9998 | 33866 | 
| 16 | **4 sphinx/ext/autodoc/__init__.py** | 2021 | 2058| 320 | 10318 | 33866 | 
| 17 | 5 sphinx/ext/autodoc/importer.py | 11 | 40| 211 | 10529 | 36328 | 
| 18 | **5 sphinx/ext/autodoc/__init__.py** | 2612 | 2636| 295 | 10824 | 36328 | 
| 19 | **5 sphinx/ext/autodoc/__init__.py** | 117 | 158| 284 | 11108 | 36328 | 
| 20 | **5 sphinx/ext/autodoc/__init__.py** | 1244 | 1260| 184 | 11292 | 36328 | 
| 21 | 5 sphinx/ext/autodoc/importer.py | 77 | 147| 649 | 11941 | 36328 | 
| 22 | 6 sphinx/ext/autodoc/directive.py | 9 | 47| 310 | 12251 | 37779 | 
| 23 | **6 sphinx/ext/autodoc/__init__.py** | 1131 | 1149| 179 | 12430 | 37779 | 
| 24 | **6 sphinx/ext/autodoc/__init__.py** | 388 | 423| 317 | 12747 | 37779 | 
| 25 | **6 sphinx/ext/autodoc/__init__.py** | 299 | 375| 751 | 13498 | 37779 | 
| 26 | 7 sphinx/ext/autodoc/typehints.py | 130 | 185| 460 | 13958 | 39233 | 
| 27 | **7 sphinx/ext/autodoc/__init__.py** | 987 | 1029| 379 | 14337 | 39233 | 
| 28 | **7 sphinx/ext/autodoc/__init__.py** | 1825 | 1840| 129 | 14466 | 39233 | 
| 29 | **7 sphinx/ext/autodoc/__init__.py** | 713 | 821| 890 | 15356 | 39233 | 
| 30 | **7 sphinx/ext/autodoc/__init__.py** | 587 | 599| 139 | 15495 | 39233 | 
| 31 | **8 sphinx/domains/python.py** | 1054 | 1072| 140 | 15635 | 52018 | 
| 32 | 9 sphinx/ext/autosummary/generate.py | 241 | 276| 343 | 15978 | 57554 | 
| 33 | **9 sphinx/ext/autodoc/__init__.py** | 1111 | 1128| 182 | 16160 | 57554 | 
| 34 | **9 sphinx/ext/autodoc/__init__.py** | 2655 | 2680| 316 | 16476 | 57554 | 
| 35 | 10 sphinx/ext/apidoc.py | 303 | 368| 752 | 17228 | 61788 | 
| 36 | 10 sphinx/ext/autodoc/mock.py | 11 | 70| 461 | 17689 | 61788 | 
| **-> 37 <-** | **10 sphinx/ext/autodoc/__init__.py** | 1403 | 1784| 188 | 17877 | 61788 | 
| 38 | **10 sphinx/ext/autodoc/__init__.py** | 2778 | 2808| 363 | 18240 | 61788 | 
| 39 | **10 sphinx/ext/autodoc/__init__.py** | 2565 | 2585| 231 | 18471 | 61788 | 
| 40 | 10 sphinx/ext/autosummary/generate.py | 492 | 513| 217 | 18688 | 61788 | 
| 41 | **10 sphinx/ext/autodoc/__init__.py** | 2638 | 2653| 182 | 18870 | 61788 | 
| 42 | 10 sphinx/ext/autosummary/generate.py | 173 | 192| 176 | 19046 | 61788 | 
| 43 | **10 sphinx/util/inspect.py** | 592 | 623| 249 | 19295 | 61788 | 
| 44 | 10 sphinx/ext/autosummary/generate.py | 20 | 53| 262 | 19557 | 61788 | 
| 45 | **10 sphinx/ext/autodoc/__init__.py** | 533 | 553| 235 | 19792 | 61788 | 
| 46 | **10 sphinx/ext/autodoc/__init__.py** | 1904 | 1942| 307 | 20099 | 61788 | 
| 47 | 10 sphinx/ext/autosummary/generate.py | 303 | 316| 202 | 20301 | 61788 | 
| 48 | 11 sphinx/ext/autodoc/deprecated.py | 114 | 127| 114 | 20415 | 62748 | 
| 49 | **11 sphinx/ext/autodoc/__init__.py** | 2287 | 2315| 248 | 20663 | 62748 | 
| 50 | **11 sphinx/ext/autodoc/__init__.py** | 1263 | 1278| 166 | 20829 | 62748 | 
| 51 | 12 sphinx/application.py | 1106 | 1127| 263 | 21092 | 74516 | 
| 52 | 12 sphinx/application.py | 1129 | 1142| 173 | 21265 | 74516 | 
| 53 | **12 sphinx/domains/python.py** | 11 | 80| 525 | 21790 | 74516 | 
| 54 | **12 sphinx/ext/autodoc/__init__.py** | 2347 | 2370| 210 | 22000 | 74516 | 
| 55 | **12 sphinx/ext/autodoc/__init__.py** | 1843 | 1859| 137 | 22137 | 74516 | 
| 56 | **12 sphinx/ext/autodoc/__init__.py** | 1185 | 1242| 434 | 22571 | 74516 | 
| 57 | **12 sphinx/ext/autodoc/__init__.py** | 1152 | 1182| 287 | 22858 | 74516 | 
| 58 | **12 sphinx/ext/autodoc/__init__.py** | 893 | 984| 857 | 23715 | 74516 | 
| 59 | **12 sphinx/ext/autodoc/__init__.py** | 1886 | 1901| 160 | 23875 | 74516 | 
| 60 | **12 sphinx/ext/autodoc/__init__.py** | 1945 | 1992| 365 | 24240 | 74516 | 
| 61 | 13 sphinx/domains/c.py | 11 | 753| 6364 | 30604 | 106959 | 
| 62 | 13 sphinx/ext/autosummary/generate.py | 333 | 381| 516 | 31120 | 106959 | 
| 63 | 13 sphinx/ext/autodoc/importer.py | 150 | 176| 183 | 31303 | 106959 | 
| 64 | 13 sphinx/ext/autodoc/importer.py | 62 | 74| 123 | 31426 | 106959 | 
| 65 | 14 sphinx/domains/std.py | 259 | 276| 122 | 31548 | 117256 | 
| 66 | **14 sphinx/ext/autodoc/__init__.py** | 1862 | 1884| 197 | 31745 | 117256 | 
| 67 | 14 sphinx/ext/autosummary/generate.py | 318 | 331| 198 | 31943 | 117256 | 
| 68 | 14 doc/conf.py | 141 | 162| 255 | 32198 | 117256 | 
| 69 | **14 sphinx/domains/python.py** | 991 | 1051| 521 | 32719 | 117256 | 
| 70 | 14 sphinx/ext/autodoc/deprecated.py | 11 | 29| 155 | 32874 | 117256 | 
| 71 | 14 sphinx/ext/autodoc/importer.py | 43 | 59| 125 | 32999 | 117256 | 
| 72 | 15 sphinx/deprecation.py | 11 | 32| 140 | 33139 | 117942 | 
| 73 | **15 sphinx/domains/python.py** | 968 | 988| 178 | 33317 | 117942 | 
| 74 | 15 sphinx/ext/autodoc/deprecated.py | 32 | 47| 140 | 33457 | 117942 | 
| 75 | **15 sphinx/ext/autodoc/__init__.py** | 823 | 866| 451 | 33908 | 117942 | 
| 76 | 15 sphinx/domains/c.py | 3606 | 3919| 2940 | 36848 | 117942 | 
| 77 | 15 sphinx/ext/autosummary/generate.py | 194 | 238| 310 | 37158 | 117942 | 
| 78 | **15 sphinx/ext/autodoc/__init__.py** | 601 | 642| 409 | 37567 | 117942 | 
| 79 | **15 sphinx/ext/autodoc/__init__.py** | 1787 | 1822| 245 | 37812 | 117942 | 
| 80 | **15 sphinx/ext/autodoc/__init__.py** | 2508 | 2542| 291 | 38103 | 117942 | 
| 81 | 16 doc/development/tutorials/examples/autodoc_intenum.py | 1 | 25| 183 | 38286 | 118316 | 
| 82 | 17 sphinx/domains/javascript.py | 248 | 273| 161 | 38447 | 122469 | 
| 83 | **17 sphinx/ext/autodoc/__init__.py** | 425 | 446| 199 | 38646 | 122469 | 
| 84 | 17 sphinx/deprecation.py | 35 | 60| 236 | 38882 | 122469 | 
| 85 | **17 sphinx/ext/autodoc/__init__.py** | 1058 | 1083| 207 | 39089 | 122469 | 
| 86 | 18 sphinx/directives/other.py | 9 | 39| 240 | 39329 | 125602 | 
| 87 | 19 sphinx/directives/__init__.py | 50 | 72| 187 | 39516 | 127854 | 
| 88 | **19 sphinx/ext/autodoc/__init__.py** | 2683 | 2757| 624 | 40140 | 127854 | 
| 89 | 19 sphinx/ext/autodoc/directive.py | 125 | 175| 453 | 40593 | 127854 | 
| 90 | 20 sphinx/domains/cpp.py | 7328 | 7915| 5169 | 45762 | 194757 | 
| 91 | 20 sphinx/ext/autosummary/generate.py | 85 | 97| 163 | 45925 | 194757 | 
| 92 | 20 sphinx/ext/apidoc.py | 17 | 73| 402 | 46327 | 194757 | 
| 93 | 20 sphinx/ext/autosummary/generate.py | 278 | 301| 284 | 46611 | 194757 | 


## Missing Patch Files

 * 1: sphinx/domains/python.py
 * 2: sphinx/ext/autodoc/__init__.py
 * 3: sphinx/util/inspect.py
 * 4: sphinx/util/typing.py

### Hint

```
+1 for adding a new confval only for autodoc. It would be nice if we can give better name to it. I feel "add_module_names" is a bit ambiguous and difficult to understand its behavior from the name.
To be clear, the [`add_module_names` confval](https://www.sphinx-doc.org/en/master/usage/configuration.html?highlight=add_module_names#confval-add_module_names) already exists. It's just that it doesn't affect parameter types for some reason.
Sorry for confuse. I know that. I thought it's better to separate the confval for autodoc and python-domain.
Just FYI: Sphinx-4.0 provides a new configuration `python_use_unqualified_type_names`. It suppresses the module name if hyperlinks can be resolved.
So does this mean that the proposed `add_module_names` configuration is completely unnecessary?
My understanding is that `python_use_unqualified_type_names` only works with resolved refs and only for the `python` domain. Although, I would personally consider this issue fixed if `python_use_unqualified_type_names` would also work for unresolved refs.
Oh that's a fair point! Yep.
Also, not sure if this deserves a separate issue/feature request, but one more place where sphinx currently produces fully qualified names is in the package/module headings generated with `apidoc`. So for example, if you have
\`\`\`
foo/__init__.py
foo/bar/__init__.py
foo/bar/baz.py
\`\`\`
you will get the documentation for
\`\`\`
foo package
 - foo.bar package
    - foo.bar.baz module
\`\`\`
instead of
\`\`\`
foo package
 - bar package
    - baz module
\`\`\`

The FQN version is kind of redundant, since the parent packages of `bar` and `baz` are already obvious from the ToC. And with longer package/module names, this can lead to really ugly wrapping in the sidebar ToC.
```

## Patch

```diff
diff --git a/sphinx/domains/python.py b/sphinx/domains/python.py
--- a/sphinx/domains/python.py
+++ b/sphinx/domains/python.py
@@ -80,7 +80,8 @@ class ModuleEntry(NamedTuple):
     deprecated: bool
 
 
-def type_to_xref(target: str, env: BuildEnvironment = None) -> addnodes.pending_xref:
+def type_to_xref(target: str, env: BuildEnvironment = None, suppress_prefix: bool = False
+                 ) -> addnodes.pending_xref:
     """Convert a type string to a cross reference node."""
     if target == 'None':
         reftype = 'obj'
@@ -101,6 +102,8 @@ def type_to_xref(target: str, env: BuildEnvironment = None) -> addnodes.pending_
     elif target.startswith('~'):
         target = target[1:]
         text = target.split('.')[-1]
+    elif suppress_prefix:
+        text = target.split('.')[-1]
     else:
         text = target
 
@@ -150,6 +153,8 @@ def unparse(node: ast.AST) -> List[Node]:
             return unparse(node.value)
         elif isinstance(node, ast.Index):
             return unparse(node.value)
+        elif isinstance(node, ast.Invert):
+            return [addnodes.desc_sig_punctuation('', '~')]
         elif isinstance(node, ast.List):
             result = [addnodes.desc_sig_punctuation('', '[')]
             if node.elts:
@@ -180,6 +185,8 @@ def unparse(node: ast.AST) -> List[Node]:
                     if isinstance(subnode, nodes.Text):
                         result[i] = nodes.literal('', '', subnode)
             return result
+        elif isinstance(node, ast.UnaryOp):
+            return unparse(node.op) + unparse(node.operand)
         elif isinstance(node, ast.Tuple):
             if node.elts:
                 result = []
@@ -209,12 +216,19 @@ def unparse(node: ast.AST) -> List[Node]:
 
     try:
         tree = ast_parse(annotation)
-        result = unparse(tree)
-        for i, node in enumerate(result):
+        result: List[Node] = []
+        for node in unparse(tree):
             if isinstance(node, nodes.literal):
-                result[i] = node[0]
+                result.append(node[0])
             elif isinstance(node, nodes.Text) and node.strip():
-                result[i] = type_to_xref(str(node), env)
+                if (result and isinstance(result[-1], addnodes.desc_sig_punctuation) and
+                        result[-1].astext() == '~'):
+                    result.pop()
+                    result.append(type_to_xref(str(node), env, suppress_prefix=True))
+                else:
+                    result.append(type_to_xref(str(node), env))
+            else:
+                result.append(node)
         return result
     except SyntaxError:
         return [type_to_xref(annotation, env)]
diff --git a/sphinx/ext/autodoc/__init__.py b/sphinx/ext/autodoc/__init__.py
--- a/sphinx/ext/autodoc/__init__.py
+++ b/sphinx/ext/autodoc/__init__.py
@@ -1295,6 +1295,8 @@ def can_document_member(cls, member: Any, membername: str, isattr: bool, parent:
     def format_args(self, **kwargs: Any) -> str:
         if self.config.autodoc_typehints in ('none', 'description'):
             kwargs.setdefault('show_annotation', False)
+        if self.config.autodoc_unqualified_typehints:
+            kwargs.setdefault('unqualified_typehints', True)
 
         try:
             self.env.app.emit('autodoc-before-process-signature', self.object, False)
@@ -1323,6 +1325,9 @@ def add_directive_header(self, sig: str) -> None:
             self.add_line('   :async:', sourcename)
 
     def format_signature(self, **kwargs: Any) -> str:
+        if self.config.autodoc_unqualified_typehints:
+            kwargs.setdefault('unqualified_typehints', True)
+
         sigs = []
         if (self.analyzer and
                 '.'.join(self.objpath) in self.analyzer.overloads and
@@ -1561,6 +1566,8 @@ def get_user_defined_function_or_method(obj: Any, attr: str) -> Any:
     def format_args(self, **kwargs: Any) -> str:
         if self.config.autodoc_typehints in ('none', 'description'):
             kwargs.setdefault('show_annotation', False)
+        if self.config.autodoc_unqualified_typehints:
+            kwargs.setdefault('unqualified_typehints', True)
 
         try:
             self._signature_class, self._signature_method_name, sig = self._get_signature()
@@ -1582,6 +1589,9 @@ def format_signature(self, **kwargs: Any) -> str:
             # do not show signatures
             return ''
 
+        if self.config.autodoc_unqualified_typehints:
+            kwargs.setdefault('unqualified_typehints', True)
+
         sig = super().format_signature()
         sigs = []
 
@@ -2110,6 +2120,8 @@ def import_object(self, raiseerror: bool = False) -> bool:
     def format_args(self, **kwargs: Any) -> str:
         if self.config.autodoc_typehints in ('none', 'description'):
             kwargs.setdefault('show_annotation', False)
+        if self.config.autodoc_unqualified_typehints:
+            kwargs.setdefault('unqualified_typehints', True)
 
         try:
             if self.object == object.__init__ and self.parent != object:
@@ -2160,6 +2172,9 @@ def document_members(self, all_members: bool = False) -> None:
         pass
 
     def format_signature(self, **kwargs: Any) -> str:
+        if self.config.autodoc_unqualified_typehints:
+            kwargs.setdefault('unqualified_typehints', True)
+
         sigs = []
         if (self.analyzer and
                 '.'.join(self.objpath) in self.analyzer.overloads and
@@ -2833,6 +2848,7 @@ def setup(app: Sphinx) -> Dict[str, Any]:
     app.add_config_value('autodoc_typehints_description_target', 'all', True,
                          ENUM('all', 'documented'))
     app.add_config_value('autodoc_type_aliases', {}, True)
+    app.add_config_value('autodoc_unqualified_typehints', False, 'env')
     app.add_config_value('autodoc_warningiserror', True, True)
     app.add_config_value('autodoc_inherit_docstrings', True, True)
     app.add_event('autodoc-before-process-signature')
diff --git a/sphinx/util/inspect.py b/sphinx/util/inspect.py
--- a/sphinx/util/inspect.py
+++ b/sphinx/util/inspect.py
@@ -744,10 +744,13 @@ def evaluate(annotation: Any, globalns: Dict, localns: Dict) -> Any:
 
 
 def stringify_signature(sig: inspect.Signature, show_annotation: bool = True,
-                        show_return_annotation: bool = True) -> str:
+                        show_return_annotation: bool = True,
+                        unqualified_typehints: bool = False) -> str:
     """Stringify a Signature object.
 
     :param show_annotation: Show annotation in result
+    :param unqualified_typehints: Show annotations as unqualified
+                                  (ex. io.StringIO -> StringIO)
     """
     args = []
     last_kind = None
@@ -771,7 +774,7 @@ def stringify_signature(sig: inspect.Signature, show_annotation: bool = True,
 
         if show_annotation and param.annotation is not param.empty:
             arg.write(': ')
-            arg.write(stringify_annotation(param.annotation))
+            arg.write(stringify_annotation(param.annotation, unqualified_typehints))
         if param.default is not param.empty:
             if show_annotation and param.annotation is not param.empty:
                 arg.write(' = ')
@@ -791,7 +794,7 @@ def stringify_signature(sig: inspect.Signature, show_annotation: bool = True,
             show_return_annotation is False):
         return '(%s)' % ', '.join(args)
     else:
-        annotation = stringify_annotation(sig.return_annotation)
+        annotation = stringify_annotation(sig.return_annotation, unqualified_typehints)
         return '(%s) -> %s' % (', '.join(args), annotation)
 
 
diff --git a/sphinx/util/typing.py b/sphinx/util/typing.py
--- a/sphinx/util/typing.py
+++ b/sphinx/util/typing.py
@@ -299,10 +299,19 @@ def _restify_py36(cls: Optional[Type]) -> str:
             return ':py:obj:`%s.%s`' % (cls.__module__, qualname)
 
 
-def stringify(annotation: Any) -> str:
-    """Stringify type annotation object."""
+def stringify(annotation: Any, smartref: bool = False) -> str:
+    """Stringify type annotation object.
+
+    :param smartref: If true, add "~" prefix to the result to remove the leading
+                     module and class names from the reference text
+    """
     from sphinx.util import inspect  # lazy loading
 
+    if smartref:
+        prefix = '~'
+    else:
+        prefix = ''
+
     if isinstance(annotation, str):
         if annotation.startswith("'") and annotation.endswith("'"):
             # might be a double Forward-ref'ed type.  Go unquoting.
@@ -313,11 +322,11 @@ def stringify(annotation: Any) -> str:
         if annotation.__module__ == 'typing':
             return annotation.__name__
         else:
-            return '.'.join([annotation.__module__, annotation.__name__])
+            return prefix + '.'.join([annotation.__module__, annotation.__name__])
     elif inspect.isNewType(annotation):
         if sys.version_info > (3, 10):
             # newtypes have correct module info since Python 3.10+
-            return '%s.%s' % (annotation.__module__, annotation.__name__)
+            return prefix + '%s.%s' % (annotation.__module__, annotation.__name__)
         else:
             return annotation.__name__
     elif not annotation:
@@ -325,7 +334,7 @@ def stringify(annotation: Any) -> str:
     elif annotation is NoneType:
         return 'None'
     elif annotation in INVALID_BUILTIN_CLASSES:
-        return INVALID_BUILTIN_CLASSES[annotation]
+        return prefix + INVALID_BUILTIN_CLASSES[annotation]
     elif str(annotation).startswith('typing.Annotated'):  # for py310+
         pass
     elif (getattr(annotation, '__module__', None) == 'builtins' and
@@ -338,28 +347,36 @@ def stringify(annotation: Any) -> str:
         return '...'
 
     if sys.version_info >= (3, 7):  # py37+
-        return _stringify_py37(annotation)
+        return _stringify_py37(annotation, smartref)
     else:
-        return _stringify_py36(annotation)
+        return _stringify_py36(annotation, smartref)
 
 
-def _stringify_py37(annotation: Any) -> str:
+def _stringify_py37(annotation: Any, smartref: bool = False) -> str:
     """stringify() for py37+."""
     module = getattr(annotation, '__module__', None)
-    if module == 'typing':
+    modprefix = ''
+    if module == 'typing' and getattr(annotation, '__forward_arg__', None):
+        qualname = annotation.__forward_arg__
+    elif module == 'typing':
         if getattr(annotation, '_name', None):
             qualname = annotation._name
         elif getattr(annotation, '__qualname__', None):
             qualname = annotation.__qualname__
-        elif getattr(annotation, '__forward_arg__', None):
-            qualname = annotation.__forward_arg__
         else:
             qualname = stringify(annotation.__origin__)  # ex. Union
+
+        if smartref:
+            modprefix = '~%s.' % module
     elif hasattr(annotation, '__qualname__'):
-        qualname = '%s.%s' % (module, annotation.__qualname__)
+        if smartref:
+            modprefix = '~%s.' % module
+        else:
+            modprefix = '%s.' % module
+        qualname = annotation.__qualname__
     elif hasattr(annotation, '__origin__'):
         # instantiated generic provided by a user
-        qualname = stringify(annotation.__origin__)
+        qualname = stringify(annotation.__origin__, smartref)
     elif UnionType and isinstance(annotation, UnionType):  # types.Union (for py3.10+)
         qualname = 'types.Union'
     else:
@@ -374,54 +391,63 @@ def _stringify_py37(annotation: Any) -> str:
         elif qualname in ('Optional', 'Union'):
             if len(annotation.__args__) > 1 and annotation.__args__[-1] is NoneType:
                 if len(annotation.__args__) > 2:
-                    args = ', '.join(stringify(a) for a in annotation.__args__[:-1])
-                    return 'Optional[Union[%s]]' % args
+                    args = ', '.join(stringify(a, smartref) for a in annotation.__args__[:-1])
+                    return '%sOptional[%sUnion[%s]]' % (modprefix, modprefix, args)
                 else:
-                    return 'Optional[%s]' % stringify(annotation.__args__[0])
+                    return '%sOptional[%s]' % (modprefix,
+                                               stringify(annotation.__args__[0], smartref))
             else:
-                args = ', '.join(stringify(a) for a in annotation.__args__)
-                return 'Union[%s]' % args
+                args = ', '.join(stringify(a, smartref) for a in annotation.__args__)
+                return '%sUnion[%s]' % (modprefix, args)
         elif qualname == 'types.Union':
             if len(annotation.__args__) > 1 and None in annotation.__args__:
                 args = ' | '.join(stringify(a) for a in annotation.__args__ if a)
-                return 'Optional[%s]' % args
+                return '%sOptional[%s]' % (modprefix, args)
             else:
                 return ' | '.join(stringify(a) for a in annotation.__args__)
         elif qualname == 'Callable':
-            args = ', '.join(stringify(a) for a in annotation.__args__[:-1])
-            returns = stringify(annotation.__args__[-1])
-            return '%s[[%s], %s]' % (qualname, args, returns)
+            args = ', '.join(stringify(a, smartref) for a in annotation.__args__[:-1])
+            returns = stringify(annotation.__args__[-1], smartref)
+            return '%s%s[[%s], %s]' % (modprefix, qualname, args, returns)
         elif qualname == 'Literal':
             args = ', '.join(repr(a) for a in annotation.__args__)
-            return '%s[%s]' % (qualname, args)
+            return '%s%s[%s]' % (modprefix, qualname, args)
         elif str(annotation).startswith('typing.Annotated'):  # for py39+
-            return stringify(annotation.__args__[0])
+            return stringify(annotation.__args__[0], smartref)
         elif all(is_system_TypeVar(a) for a in annotation.__args__):
             # Suppress arguments if all system defined TypeVars (ex. Dict[KT, VT])
-            return qualname
+            return modprefix + qualname
         else:
-            args = ', '.join(stringify(a) for a in annotation.__args__)
-            return '%s[%s]' % (qualname, args)
+            args = ', '.join(stringify(a, smartref) for a in annotation.__args__)
+            return '%s%s[%s]' % (modprefix, qualname, args)
 
-    return qualname
+    return modprefix + qualname
 
 
-def _stringify_py36(annotation: Any) -> str:
+def _stringify_py36(annotation: Any, smartref: bool = False) -> str:
     """stringify() for py36."""
     module = getattr(annotation, '__module__', None)
-    if module == 'typing':
+    modprefix = ''
+    if module == 'typing' and getattr(annotation, '__forward_arg__', None):
+        qualname = annotation.__forward_arg__
+    elif module == 'typing':
         if getattr(annotation, '_name', None):
             qualname = annotation._name
         elif getattr(annotation, '__qualname__', None):
             qualname = annotation.__qualname__
-        elif getattr(annotation, '__forward_arg__', None):
-            qualname = annotation.__forward_arg__
         elif getattr(annotation, '__origin__', None):
             qualname = stringify(annotation.__origin__)  # ex. Union
         else:
             qualname = repr(annotation).replace('typing.', '')
+
+        if smartref:
+            modprefix = '~%s.' % module
     elif hasattr(annotation, '__qualname__'):
-        qualname = '%s.%s' % (module, annotation.__qualname__)
+        if smartref:
+            modprefix = '~%s.' % module
+        else:
+            modprefix = '%s.' % module
+        qualname = annotation.__qualname__
     else:
         qualname = repr(annotation)
 
@@ -429,10 +455,10 @@ def _stringify_py36(annotation: Any) -> str:
             not hasattr(annotation, '__tuple_params__')):  # for Python 3.6
         params = annotation.__args__
         if params:
-            param_str = ', '.join(stringify(p) for p in params)
-            return '%s[%s]' % (qualname, param_str)
+            param_str = ', '.join(stringify(p, smartref) for p in params)
+            return '%s%s[%s]' % (modprefix, qualname, param_str)
         else:
-            return qualname
+            return modprefix + qualname
     elif isinstance(annotation, typing.GenericMeta):
         params = None
         if annotation.__args__ is None or len(annotation.__args__) <= 2:  # type: ignore  # NOQA
@@ -440,28 +466,28 @@ def _stringify_py36(annotation: Any) -> str:
         elif annotation.__origin__ == Generator:  # type: ignore
             params = annotation.__args__  # type: ignore
         else:  # typing.Callable
-            args = ', '.join(stringify(arg) for arg
+            args = ', '.join(stringify(arg, smartref) for arg
                              in annotation.__args__[:-1])  # type: ignore
             result = stringify(annotation.__args__[-1])  # type: ignore
-            return '%s[[%s], %s]' % (qualname, args, result)
+            return '%s%s[[%s], %s]' % (modprefix, qualname, args, result)
         if params is not None:
-            param_str = ', '.join(stringify(p) for p in params)
-            return '%s[%s]' % (qualname, param_str)
+            param_str = ', '.join(stringify(p, smartref) for p in params)
+            return '%s%s[%s]' % (modprefix, qualname, param_str)
     elif (hasattr(annotation, '__origin__') and
           annotation.__origin__ is typing.Union):
         params = annotation.__args__
         if params is not None:
             if len(params) > 1 and params[-1] is NoneType:
                 if len(params) > 2:
-                    param_str = ", ".join(stringify(p) for p in params[:-1])
-                    return 'Optional[Union[%s]]' % param_str
+                    param_str = ", ".join(stringify(p, smartref) for p in params[:-1])
+                    return '%sOptional[%sUnion[%s]]' % (modprefix, modprefix, param_str)
                 else:
-                    return 'Optional[%s]' % stringify(params[0])
+                    return '%sOptional[%s]' % (modprefix, stringify(params[0]))
             else:
-                param_str = ', '.join(stringify(p) for p in params)
-                return 'Union[%s]' % param_str
+                param_str = ', '.join(stringify(p, smartref) for p in params)
+                return '%sUnion[%s]' % (modprefix, param_str)
 
-    return qualname
+    return modprefix + qualname
 
 
 deprecated_alias('sphinx.util.typing',

```

## Test Patch

```diff
diff --git a/tests/test_domain_py.py b/tests/test_domain_py.py
--- a/tests/test_domain_py.py
+++ b/tests/test_domain_py.py
@@ -350,6 +350,18 @@ def test_parse_annotation(app):
     assert_node(doctree[0], pending_xref, refdomain="py", reftype="obj", reftarget="None")
 
 
+def test_parse_annotation_suppress(app):
+    doctree = _parse_annotation("~typing.Dict[str, str]", app.env)
+    assert_node(doctree, ([pending_xref, "Dict"],
+                          [desc_sig_punctuation, "["],
+                          [pending_xref, "str"],
+                          [desc_sig_punctuation, ","],
+                          desc_sig_space,
+                          [pending_xref, "str"],
+                          [desc_sig_punctuation, "]"]))
+    assert_node(doctree[0], pending_xref, refdomain="py", reftype="class", reftarget="typing.Dict")
+
+
 @pytest.mark.skipif(sys.version_info < (3, 8), reason='python 3.8+ is required.')
 def test_parse_annotation_Literal(app):
     doctree = _parse_annotation("Literal[True, False]", app.env)
diff --git a/tests/test_ext_autodoc_configs.py b/tests/test_ext_autodoc_configs.py
--- a/tests/test_ext_autodoc_configs.py
+++ b/tests/test_ext_autodoc_configs.py
@@ -1142,6 +1142,99 @@ def test_autodoc_typehints_description_and_type_aliases(app):
             '      myint\n' == context)
 
 
+@pytest.mark.sphinx('html', testroot='ext-autodoc',
+                    confoverrides={'autodoc_unqualified_typehints': True})
+def test_autodoc_unqualified_typehints(app):
+    if sys.version_info < (3, 7):
+        Any = 'Any'
+    else:
+        Any = '~typing.Any'
+
+    options = {"members": None,
+               "undoc-members": None}
+    actual = do_autodoc(app, 'module', 'target.typehints', options)
+    assert list(actual) == [
+        '',
+        '.. py:module:: target.typehints',
+        '',
+        '',
+        '.. py:data:: CONST1',
+        '   :module: target.typehints',
+        '   :type: int',
+        '',
+        '',
+        '.. py:class:: Math(s: str, o: ~typing.Optional[%s] = None)' % Any,
+        '   :module: target.typehints',
+        '',
+        '',
+        '   .. py:attribute:: Math.CONST1',
+        '      :module: target.typehints',
+        '      :type: int',
+        '',
+        '',
+        '   .. py:attribute:: Math.CONST2',
+        '      :module: target.typehints',
+        '      :type: int',
+        '      :value: 1',
+        '',
+        '',
+        '   .. py:method:: Math.decr(a: int, b: int = 1) -> int',
+        '      :module: target.typehints',
+        '',
+        '',
+        '   .. py:method:: Math.horse(a: str, b: int) -> None',
+        '      :module: target.typehints',
+        '',
+        '',
+        '   .. py:method:: Math.incr(a: int, b: int = 1) -> int',
+        '      :module: target.typehints',
+        '',
+        '',
+        '   .. py:method:: Math.nothing() -> None',
+        '      :module: target.typehints',
+        '',
+        '',
+        '   .. py:property:: Math.prop',
+        '      :module: target.typehints',
+        '      :type: int',
+        '',
+        '',
+        '.. py:class:: NewAnnotation(i: int)',
+        '   :module: target.typehints',
+        '',
+        '',
+        '.. py:class:: NewComment(i: int)',
+        '   :module: target.typehints',
+        '',
+        '',
+        '.. py:class:: SignatureFromMetaclass(a: int)',
+        '   :module: target.typehints',
+        '',
+        '',
+        '.. py:function:: complex_func(arg1: str, arg2: List[int], arg3: Tuple[int, '
+        'Union[str, Unknown]] = None, *args: str, **kwargs: str) -> None',
+        '   :module: target.typehints',
+        '',
+        '',
+        '.. py:function:: decr(a: int, b: int = 1) -> int',
+        '   :module: target.typehints',
+        '',
+        '',
+        '.. py:function:: incr(a: int, b: int = 1) -> int',
+        '   :module: target.typehints',
+        '',
+        '',
+        '.. py:function:: missing_attr(c, a: str, b: Optional[str] = None) -> str',
+        '   :module: target.typehints',
+        '',
+        '',
+        '.. py:function:: tuple_args(x: ~typing.Tuple[int, ~typing.Union[int, str]]) '
+        '-> ~typing.Tuple[int, int]',
+        '   :module: target.typehints',
+        '',
+    ]
+
+
 @pytest.mark.sphinx('html', testroot='ext-autodoc')
 def test_autodoc_default_options(app):
     # no settings
diff --git a/tests/test_util_inspect.py b/tests/test_util_inspect.py
--- a/tests/test_util_inspect.py
+++ b/tests/test_util_inspect.py
@@ -259,6 +259,10 @@ def test_signature_annotations():
     sig = inspect.signature(f7)
     assert stringify_signature(sig, show_return_annotation=False) == '(x: Optional[int] = None, y: dict = {})'
 
+    # unqualified_typehints is True
+    sig = inspect.signature(f7)
+    assert stringify_signature(sig, unqualified_typehints=True) == '(x: ~typing.Optional[int] = None, y: dict = {}) -> None'
+
 
 @pytest.mark.skipif(sys.version_info < (3, 8), reason='python 3.8+ is required.')
 @pytest.mark.sphinx(testroot='ext-autodoc')
diff --git a/tests/test_util_typing.py b/tests/test_util_typing.py
--- a/tests/test_util_typing.py
+++ b/tests/test_util_typing.py
@@ -178,78 +178,156 @@ def test_restify_mock():
 
 
 def test_stringify():
-    assert stringify(int) == "int"
-    assert stringify(str) == "str"
-    assert stringify(None) == "None"
-    assert stringify(Integral) == "numbers.Integral"
-    assert stringify(Struct) == "struct.Struct"
-    assert stringify(TracebackType) == "types.TracebackType"
-    assert stringify(Any) == "Any"
+    assert stringify(int, False) == "int"
+    assert stringify(int, True) == "int"
+
+    assert stringify(str, False) == "str"
+    assert stringify(str, True) == "str"
+
+    assert stringify(None, False) == "None"
+    assert stringify(None, True) == "None"
+
+    assert stringify(Integral, False) == "numbers.Integral"
+    assert stringify(Integral, True) == "~numbers.Integral"
+
+    assert stringify(Struct, False) == "struct.Struct"
+    assert stringify(Struct, True) == "~struct.Struct"
+
+    assert stringify(TracebackType, False) == "types.TracebackType"
+    assert stringify(TracebackType, True) == "~types.TracebackType"
+
+    assert stringify(Any, False) == "Any"
+    assert stringify(Any, True) == "~typing.Any"
 
 
 def test_stringify_type_hints_containers():
-    assert stringify(List) == "List"
-    assert stringify(Dict) == "Dict"
-    assert stringify(List[int]) == "List[int]"
-    assert stringify(List[str]) == "List[str]"
-    assert stringify(Dict[str, float]) == "Dict[str, float]"
-    assert stringify(Tuple[str, str, str]) == "Tuple[str, str, str]"
-    assert stringify(Tuple[str, ...]) == "Tuple[str, ...]"
-    assert stringify(Tuple[()]) == "Tuple[()]"
-    assert stringify(List[Dict[str, Tuple]]) == "List[Dict[str, Tuple]]"
-    assert stringify(MyList[Tuple[int, int]]) == "tests.test_util_typing.MyList[Tuple[int, int]]"
-    assert stringify(Generator[None, None, None]) == "Generator[None, None, None]"
+    assert stringify(List, False) == "List"
+    assert stringify(List, True) == "~typing.List"
+
+    assert stringify(Dict, False) == "Dict"
+    assert stringify(Dict, True) == "~typing.Dict"
+
+    assert stringify(List[int], False) == "List[int]"
+    assert stringify(List[int], True) == "~typing.List[int]"
+
+    assert stringify(List[str], False) == "List[str]"
+    assert stringify(List[str], True) == "~typing.List[str]"
+
+    assert stringify(Dict[str, float], False) == "Dict[str, float]"
+    assert stringify(Dict[str, float], True) == "~typing.Dict[str, float]"
+
+    assert stringify(Tuple[str, str, str], False) == "Tuple[str, str, str]"
+    assert stringify(Tuple[str, str, str], True) == "~typing.Tuple[str, str, str]"
+
+    assert stringify(Tuple[str, ...], False) == "Tuple[str, ...]"
+    assert stringify(Tuple[str, ...], True) == "~typing.Tuple[str, ...]"
+
+    assert stringify(Tuple[()], False) == "Tuple[()]"
+    assert stringify(Tuple[()], True) == "~typing.Tuple[()]"
+
+    assert stringify(List[Dict[str, Tuple]], False) == "List[Dict[str, Tuple]]"
+    assert stringify(List[Dict[str, Tuple]], True) == "~typing.List[~typing.Dict[str, ~typing.Tuple]]"
+
+    assert stringify(MyList[Tuple[int, int]], False) == "tests.test_util_typing.MyList[Tuple[int, int]]"
+    assert stringify(MyList[Tuple[int, int]], True) == "~tests.test_util_typing.MyList[~typing.Tuple[int, int]]"
+
+    assert stringify(Generator[None, None, None], False) == "Generator[None, None, None]"
+    assert stringify(Generator[None, None, None], True) == "~typing.Generator[None, None, None]"
 
 
 @pytest.mark.skipif(sys.version_info < (3, 9), reason='python 3.9+ is required.')
 def test_stringify_type_hints_pep_585():
-    assert stringify(list[int]) == "list[int]"
-    assert stringify(list[str]) == "list[str]"
-    assert stringify(dict[str, float]) == "dict[str, float]"
-    assert stringify(tuple[str, str, str]) == "tuple[str, str, str]"
-    assert stringify(tuple[str, ...]) == "tuple[str, ...]"
-    assert stringify(tuple[()]) == "tuple[()]"
-    assert stringify(list[dict[str, tuple]]) == "list[dict[str, tuple]]"
-    assert stringify(type[int]) == "type[int]"
+    assert stringify(list[int], False) == "list[int]"
+    assert stringify(list[int], True) == "list[int]"
+
+    assert stringify(list[str], False) == "list[str]"
+    assert stringify(list[str], True) == "list[str]"
+
+    assert stringify(dict[str, float], False) == "dict[str, float]"
+    assert stringify(dict[str, float], True) == "dict[str, float]"
+
+    assert stringify(tuple[str, str, str], False) == "tuple[str, str, str]"
+    assert stringify(tuple[str, str, str], True) == "tuple[str, str, str]"
+
+    assert stringify(tuple[str, ...], False) == "tuple[str, ...]"
+    assert stringify(tuple[str, ...], True) == "tuple[str, ...]"
+
+    assert stringify(tuple[()], False) == "tuple[()]"
+    assert stringify(tuple[()], True) == "tuple[()]"
+
+    assert stringify(list[dict[str, tuple]], False) == "list[dict[str, tuple]]"
+    assert stringify(list[dict[str, tuple]], True) == "list[dict[str, tuple]]"
+
+    assert stringify(type[int], False) == "type[int]"
+    assert stringify(type[int], True) == "type[int]"
 
 
 @pytest.mark.skipif(sys.version_info < (3, 9), reason='python 3.9+ is required.')
 def test_stringify_Annotated():
     from typing import Annotated  # type: ignore
-    assert stringify(Annotated[str, "foo", "bar"]) == "str"  # NOQA
+    assert stringify(Annotated[str, "foo", "bar"], False) == "str"  # NOQA
+    assert stringify(Annotated[str, "foo", "bar"], True) == "str"  # NOQA
 
 
 def test_stringify_type_hints_string():
-    assert stringify("int") == "int"
-    assert stringify("str") == "str"
-    assert stringify(List["int"]) == "List[int]"
-    assert stringify("Tuple[str]") == "Tuple[str]"
-    assert stringify("unknown") == "unknown"
+    assert stringify("int", False) == "int"
+    assert stringify("int", True) == "int"
+
+    assert stringify("str", False) == "str"
+    assert stringify("str", True) == "str"
+
+    assert stringify(List["int"], False) == "List[int]"
+    assert stringify(List["int"], True) == "~typing.List[int]"
+
+    assert stringify("Tuple[str]", False) == "Tuple[str]"
+    assert stringify("Tuple[str]", True) == "Tuple[str]"
+
+    assert stringify("unknown", False) == "unknown"
+    assert stringify("unknown", True) == "unknown"
 
 
 def test_stringify_type_hints_Callable():
-    assert stringify(Callable) == "Callable"
+    assert stringify(Callable, False) == "Callable"
+    assert stringify(Callable, True) == "~typing.Callable"
 
     if sys.version_info >= (3, 7):
-        assert stringify(Callable[[str], int]) == "Callable[[str], int]"
-        assert stringify(Callable[..., int]) == "Callable[[...], int]"
+        assert stringify(Callable[[str], int], False) == "Callable[[str], int]"
+        assert stringify(Callable[[str], int], True) == "~typing.Callable[[str], int]"
+
+        assert stringify(Callable[..., int], False) == "Callable[[...], int]"
+        assert stringify(Callable[..., int], True) == "~typing.Callable[[...], int]"
     else:
-        assert stringify(Callable[[str], int]) == "Callable[str, int]"
-        assert stringify(Callable[..., int]) == "Callable[..., int]"
+        assert stringify(Callable[[str], int], False) == "Callable[str, int]"
+        assert stringify(Callable[[str], int], True) == "~typing.Callable[str, int]"
+
+        assert stringify(Callable[..., int], False) == "Callable[..., int]"
+        assert stringify(Callable[..., int], True) == "~typing.Callable[..., int]"
 
 
 def test_stringify_type_hints_Union():
-    assert stringify(Optional[int]) == "Optional[int]"
-    assert stringify(Union[str, None]) == "Optional[str]"
-    assert stringify(Union[int, str]) == "Union[int, str]"
+    assert stringify(Optional[int], False) == "Optional[int]"
+    assert stringify(Optional[int], True) == "~typing.Optional[int]"
+
+    assert stringify(Union[str, None], False) == "Optional[str]"
+    assert stringify(Union[str, None], True) == "~typing.Optional[str]"
+
+    assert stringify(Union[int, str], False) == "Union[int, str]"
+    assert stringify(Union[int, str], True) == "~typing.Union[int, str]"
 
     if sys.version_info >= (3, 7):
-        assert stringify(Union[int, Integral]) == "Union[int, numbers.Integral]"
-        assert (stringify(Union[MyClass1, MyClass2]) ==
+        assert stringify(Union[int, Integral], False) == "Union[int, numbers.Integral]"
+        assert stringify(Union[int, Integral], True) == "~typing.Union[int, ~numbers.Integral]"
+
+        assert (stringify(Union[MyClass1, MyClass2], False) ==
                 "Union[tests.test_util_typing.MyClass1, tests.test_util_typing.<MyClass2>]")
+        assert (stringify(Union[MyClass1, MyClass2], True) ==
+                "~typing.Union[~tests.test_util_typing.MyClass1, ~tests.test_util_typing.<MyClass2>]")
     else:
-        assert stringify(Union[int, Integral]) == "numbers.Integral"
-        assert stringify(Union[MyClass1, MyClass2]) == "tests.test_util_typing.MyClass1"
+        assert stringify(Union[int, Integral], False) == "numbers.Integral"
+        assert stringify(Union[int, Integral], True) == "~numbers.Integral"
+
+        assert stringify(Union[MyClass1, MyClass2], False) == "tests.test_util_typing.MyClass1"
+        assert stringify(Union[MyClass1, MyClass2], True) == "~tests.test_util_typing.MyClass1"
 
 
 def test_stringify_type_hints_typevars():
@@ -258,52 +336,83 @@ def test_stringify_type_hints_typevars():
     T_contra = TypeVar('T_contra', contravariant=True)
 
     if sys.version_info < (3, 7):
-        assert stringify(T) == "T"
-        assert stringify(T_co) == "T_co"
-        assert stringify(T_contra) == "T_contra"
-        assert stringify(List[T]) == "List[T]"
+        assert stringify(T, False) == "T"
+        assert stringify(T, True) == "T"
+
+        assert stringify(T_co, False) == "T_co"
+        assert stringify(T_co, True) == "T_co"
+
+        assert stringify(T_contra, False) == "T_contra"
+        assert stringify(T_contra, True) == "T_contra"
+
+        assert stringify(List[T], False) == "List[T]"
+        assert stringify(List[T], True) == "~typing.List[T]"
     else:
-        assert stringify(T) == "tests.test_util_typing.T"
-        assert stringify(T_co) == "tests.test_util_typing.T_co"
-        assert stringify(T_contra) == "tests.test_util_typing.T_contra"
-        assert stringify(List[T]) == "List[tests.test_util_typing.T]"
+        assert stringify(T, False) == "tests.test_util_typing.T"
+        assert stringify(T, True) == "~tests.test_util_typing.T"
+
+        assert stringify(T_co, False) == "tests.test_util_typing.T_co"
+        assert stringify(T_co, True) == "~tests.test_util_typing.T_co"
+
+        assert stringify(T_contra, False) == "tests.test_util_typing.T_contra"
+        assert stringify(T_contra, True) == "~tests.test_util_typing.T_contra"
+
+        assert stringify(List[T], False) == "List[tests.test_util_typing.T]"
+        assert stringify(List[T], True) == "~typing.List[~tests.test_util_typing.T]"
 
     if sys.version_info >= (3, 10):
-        assert stringify(MyInt) == "tests.test_util_typing.MyInt"
+        assert stringify(MyInt, False) == "tests.test_util_typing.MyInt"
+        assert stringify(MyInt, True) == "~tests.test_util_typing.MyInt"
     else:
-        assert stringify(MyInt) == "MyInt"
+        assert stringify(MyInt, False) == "MyInt"
+        assert stringify(MyInt, True) == "MyInt"
 
 
 def test_stringify_type_hints_custom_class():
-    assert stringify(MyClass1) == "tests.test_util_typing.MyClass1"
-    assert stringify(MyClass2) == "tests.test_util_typing.<MyClass2>"
+    assert stringify(MyClass1, False) == "tests.test_util_typing.MyClass1"
+    assert stringify(MyClass1, True) == "~tests.test_util_typing.MyClass1"
+
+    assert stringify(MyClass2, False) == "tests.test_util_typing.<MyClass2>"
+    assert stringify(MyClass2, True) == "~tests.test_util_typing.<MyClass2>"
 
 
 def test_stringify_type_hints_alias():
     MyStr = str
     MyTuple = Tuple[str, str]
-    assert stringify(MyStr) == "str"
-    assert stringify(MyTuple) == "Tuple[str, str]"  # type: ignore
+
+    assert stringify(MyStr, False) == "str"
+    assert stringify(MyStr, True) == "str"
+
+    assert stringify(MyTuple, False) == "Tuple[str, str]"  # type: ignore
+    assert stringify(MyTuple, True) == "~typing.Tuple[str, str]"  # type: ignore
 
 
 @pytest.mark.skipif(sys.version_info < (3, 8), reason='python 3.8+ is required.')
 def test_stringify_type_Literal():
     from typing import Literal  # type: ignore
-    assert stringify(Literal[1, "2", "\r"]) == "Literal[1, '2', '\\r']"
+    assert stringify(Literal[1, "2", "\r"], False) == "Literal[1, '2', '\\r']"
+    assert stringify(Literal[1, "2", "\r"], True) == "~typing.Literal[1, '2', '\\r']"
 
 
 @pytest.mark.skipif(sys.version_info < (3, 10), reason='python 3.10+ is required.')
 def test_stringify_type_union_operator():
-    assert stringify(int | None) == "int | None"  # type: ignore
-    assert stringify(int | str) == "int | str"  # type: ignore
-    assert stringify(int | str | None) == "int | str | None"  # type: ignore
+    assert stringify(int | None, False) == "int | None"  # type: ignore
+    assert stringify(int | None, True) == "int | None"  # type: ignore
+
+    assert stringify(int | str, False) == "int | str"  # type: ignore
+    assert stringify(int | str, True) == "int | str"  # type: ignore
+
+    assert stringify(int | str | None, False) == "int | str | None"  # type: ignore
+    assert stringify(int | str | None, True) == "int | str | None"  # type: ignore
 
 
 def test_stringify_broken_type_hints():
-    assert stringify(BrokenType) == 'tests.test_util_typing.BrokenType'
+    assert stringify(BrokenType, False) == 'tests.test_util_typing.BrokenType'
+    assert stringify(BrokenType, True) == '~tests.test_util_typing.BrokenType'
 
 
 def test_stringify_mock():
     with mock(['unknown']):
         import unknown
-        assert stringify(unknown.secret.Class) == 'unknown.secret.Class'
+        assert stringify(unknown.secret.Class, False) == 'unknown.secret.Class'
+        assert stringify(unknown.secret.Class, True) == 'unknown.secret.Class'

```


## Code snippets

### 1 - sphinx/ext/autodoc/__init__.py:

Start line: 1434, End line: 1784

```python
class ClassDocumenter(DocstringSignatureMixin, ModuleLevelDocumenter):  # type: ignore
    """
    Specialized Documenter subclass for classes.
    """
    objtype = 'class'
    member_order = 20
    option_spec: OptionSpec = {
        'members': members_option, 'undoc-members': bool_option,
        'noindex': bool_option, 'inherited-members': inherited_members_option,
        'show-inheritance': bool_option, 'member-order': member_order_option,
        'exclude-members': exclude_members_option,
        'private-members': members_option, 'special-members': members_option,
        'class-doc-from': class_doc_from_option,
    }

    _signature_class: Any = None
    _signature_method_name: str = None

    def __init__(self, *args: Any) -> None:
        super().__init__(*args)

        if self.config.autodoc_class_signature == 'separated':
            self.options = self.options.copy()

            # show __init__() method
            if self.options.special_members is None:
                self.options['special-members'] = ['__new__', '__init__']
            else:
                self.options.special_members.append('__new__')
                self.options.special_members.append('__init__')

        merge_members_option(self.options)

    @classmethod
    def can_document_member(cls, member: Any, membername: str, isattr: bool, parent: Any
                            ) -> bool:
        return isinstance(member, type)

    def import_object(self, raiseerror: bool = False) -> bool:
        ret = super().import_object(raiseerror)
        # if the class is documented under another name, document it
        # as data/attribute
        if ret:
            if hasattr(self.object, '__name__'):
                self.doc_as_attr = (self.objpath[-1] != self.object.__name__)
            else:
                self.doc_as_attr = True
        return ret

    def _get_signature(self) -> Tuple[Optional[Any], Optional[str], Optional[Signature]]:
        def get_user_defined_function_or_method(obj: Any, attr: str) -> Any:
            """ Get the `attr` function or method from `obj`, if it is user-defined. """
            if inspect.is_builtin_class_method(obj, attr):
                return None
            attr = self.get_attr(obj, attr, None)
            if not (inspect.ismethod(attr) or inspect.isfunction(attr)):
                return None
            return attr

        # This sequence is copied from inspect._signature_from_callable.
        # ValueError means that no signature could be found, so we keep going.

        # First, we check the obj has a __signature__ attribute
        if (hasattr(self.object, '__signature__') and
                isinstance(self.object.__signature__, Signature)):
            return None, None, self.object.__signature__

        # Next, let's see if it has an overloaded __call__ defined
        # in its metaclass
        call = get_user_defined_function_or_method(type(self.object), '__call__')

        if call is not None:
            if "{0.__module__}.{0.__qualname__}".format(call) in _METACLASS_CALL_BLACKLIST:
                call = None

        if call is not None:
            self.env.app.emit('autodoc-before-process-signature', call, True)
            try:
                sig = inspect.signature(call, bound_method=True,
                                        type_aliases=self.config.autodoc_type_aliases)
                return type(self.object), '__call__', sig
            except ValueError:
                pass

        # Now we check if the 'obj' class has a '__new__' method
        new = get_user_defined_function_or_method(self.object, '__new__')

        if new is not None:
            if "{0.__module__}.{0.__qualname__}".format(new) in _CLASS_NEW_BLACKLIST:
                new = None

        if new is not None:
            self.env.app.emit('autodoc-before-process-signature', new, True)
            try:
                sig = inspect.signature(new, bound_method=True,
                                        type_aliases=self.config.autodoc_type_aliases)
                return self.object, '__new__', sig
            except ValueError:
                pass

        # Finally, we should have at least __init__ implemented
        init = get_user_defined_function_or_method(self.object, '__init__')
        if init is not None:
            self.env.app.emit('autodoc-before-process-signature', init, True)
            try:
                sig = inspect.signature(init, bound_method=True,
                                        type_aliases=self.config.autodoc_type_aliases)
                return self.object, '__init__', sig
            except ValueError:
                pass

        # None of the attributes are user-defined, so fall back to let inspect
        # handle it.
        # We don't know the exact method that inspect.signature will read
        # the signature from, so just pass the object itself to our hook.
        self.env.app.emit('autodoc-before-process-signature', self.object, False)
        try:
            sig = inspect.signature(self.object, bound_method=False,
                                    type_aliases=self.config.autodoc_type_aliases)
            return None, None, sig
        except ValueError:
            pass

        # Still no signature: happens e.g. for old-style classes
        # with __init__ in C and no `__text_signature__`.
        return None, None, None

    def format_args(self, **kwargs: Any) -> str:
        if self.config.autodoc_typehints in ('none', 'description'):
            kwargs.setdefault('show_annotation', False)

        try:
            self._signature_class, self._signature_method_name, sig = self._get_signature()
        except TypeError as exc:
            # __signature__ attribute contained junk
            logger.warning(__("Failed to get a constructor signature for %s: %s"),
                           self.fullname, exc)
            return None

        if sig is None:
            return None

        return stringify_signature(sig, show_return_annotation=False, **kwargs)

    def format_signature(self, **kwargs: Any) -> str:
        if self.doc_as_attr:
            return ''
        if self.config.autodoc_class_signature == 'separated':
            # do not show signatures
            return ''

        sig = super().format_signature()
        sigs = []

        overloads = self.get_overloaded_signatures()
        if overloads and self.config.autodoc_typehints != 'none':
            # Use signatures for overloaded methods instead of the implementation method.
            method = safe_getattr(self._signature_class, self._signature_method_name, None)
            __globals__ = safe_getattr(method, '__globals__', {})
            for overload in overloads:
                overload = evaluate_signature(overload, __globals__,
                                              self.config.autodoc_type_aliases)

                parameters = list(overload.parameters.values())
                overload = overload.replace(parameters=parameters[1:],
                                            return_annotation=Parameter.empty)
                sig = stringify_signature(overload, **kwargs)
                sigs.append(sig)
        else:
            sigs.append(sig)

        return "\n".join(sigs)

    def get_overloaded_signatures(self) -> List[Signature]:
        if self._signature_class and self._signature_method_name:
            for cls in self._signature_class.__mro__:
                try:
                    analyzer = ModuleAnalyzer.for_module(cls.__module__)
                    analyzer.analyze()
                    qualname = '.'.join([cls.__qualname__, self._signature_method_name])
                    if qualname in analyzer.overloads:
                        return analyzer.overloads.get(qualname)
                    elif qualname in analyzer.tagorder:
                        # the constructor is defined in the class, but not overridden.
                        return []
                except PycodeError:
                    pass

        return []

    def get_canonical_fullname(self) -> Optional[str]:
        __modname__ = safe_getattr(self.object, '__module__', self.modname)
        __qualname__ = safe_getattr(self.object, '__qualname__', None)
        if __qualname__ is None:
            __qualname__ = safe_getattr(self.object, '__name__', None)
        if __qualname__ and '<locals>' in __qualname__:
            # No valid qualname found if the object is defined as locals
            __qualname__ = None

        if __modname__ and __qualname__:
            return '.'.join([__modname__, __qualname__])
        else:
            return None

    def add_directive_header(self, sig: str) -> None:
        sourcename = self.get_sourcename()

        if self.doc_as_attr:
            self.directivetype = 'attribute'
        super().add_directive_header(sig)

        if self.analyzer and '.'.join(self.objpath) in self.analyzer.finals:
            self.add_line('   :final:', sourcename)

        canonical_fullname = self.get_canonical_fullname()
        if not self.doc_as_attr and canonical_fullname and self.fullname != canonical_fullname:
            self.add_line('   :canonical: %s' % canonical_fullname, sourcename)

        # add inheritance info, if wanted
        if not self.doc_as_attr and self.options.show_inheritance:
            if inspect.getorigbases(self.object):
                # A subclass of generic types
                # refs: PEP-560 <https://www.python.org/dev/peps/pep-0560/>
                bases = list(self.object.__orig_bases__)
            elif hasattr(self.object, '__bases__') and len(self.object.__bases__):
                # A normal class
                bases = list(self.object.__bases__)
            else:
                bases = []

            self.env.events.emit('autodoc-process-bases',
                                 self.fullname, self.object, self.options, bases)

            base_classes = [restify(cls) for cls in bases]
            sourcename = self.get_sourcename()
            self.add_line('', sourcename)
            self.add_line('   ' + _('Bases: %s') % ', '.join(base_classes), sourcename)

    def get_object_members(self, want_all: bool) -> Tuple[bool, ObjectMembers]:
        members = get_class_members(self.object, self.objpath, self.get_attr)
        if not want_all:
            if not self.options.members:
                return False, []  # type: ignore
            # specific members given
            selected = []
            for name in self.options.members:  # type: str
                if name in members:
                    selected.append(members[name])
                else:
                    logger.warning(__('missing attribute %s in object %s') %
                                   (name, self.fullname), type='autodoc')
            return False, selected
        elif self.options.inherited_members:
            return False, list(members.values())
        else:
            return False, [m for m in members.values() if m.class_ == self.object]

    def get_doc(self, ignore: int = None) -> Optional[List[List[str]]]:
        if self.doc_as_attr:
            # Don't show the docstring of the class when it is an alias.
            comment = self.get_variable_comment()
            if comment:
                return []
            else:
                return None

        lines = getattr(self, '_new_docstrings', None)
        if lines is not None:
            return lines

        classdoc_from = self.options.get('class-doc-from', self.config.autoclass_content)

        docstrings = []
        attrdocstring = getdoc(self.object, self.get_attr)
        if attrdocstring:
            docstrings.append(attrdocstring)

        # for classes, what the "docstring" is can be controlled via a
        # config value; the default is only the class docstring
        if classdoc_from in ('both', 'init'):
            __init__ = self.get_attr(self.object, '__init__', None)
            initdocstring = getdoc(__init__, self.get_attr,
                                   self.config.autodoc_inherit_docstrings,
                                   self.object, '__init__')
            # for new-style classes, no __init__ means default __init__
            if (initdocstring is not None and
                (initdocstring == object.__init__.__doc__ or  # for pypy
                 initdocstring.strip() == object.__init__.__doc__)):  # for !pypy
                initdocstring = None
            if not initdocstring:
                # try __new__
                __new__ = self.get_attr(self.object, '__new__', None)
                initdocstring = getdoc(__new__, self.get_attr,
                                       self.config.autodoc_inherit_docstrings,
                                       self.object, '__new__')
                # for new-style classes, no __new__ means default __new__
                if (initdocstring is not None and
                    (initdocstring == object.__new__.__doc__ or  # for pypy
                     initdocstring.strip() == object.__new__.__doc__)):  # for !pypy
                    initdocstring = None
            if initdocstring:
                if classdoc_from == 'init':
                    docstrings = [initdocstring]
                else:
                    docstrings.append(initdocstring)

        tab_width = self.directive.state.document.settings.tab_width
        return [prepare_docstring(docstring, ignore, tab_width) for docstring in docstrings]

    def get_variable_comment(self) -> Optional[List[str]]:
        try:
            key = ('', '.'.join(self.objpath))
            if self.doc_as_attr:
                analyzer = ModuleAnalyzer.for_module(self.modname)
            else:
                analyzer = ModuleAnalyzer.for_module(self.get_real_modname())
            analyzer.analyze()
            return list(analyzer.attr_docs.get(key, []))
        except PycodeError:
            return None

    def add_content(self, more_content: Optional[StringList], no_docstring: bool = False
                    ) -> None:
        if self.doc_as_attr and self.modname != self.get_real_modname():
            # override analyzer to obtain doccomment around its definition.
            self.analyzer = ModuleAnalyzer.for_module(self.modname)
            self.analyzer.analyze()

        if self.doc_as_attr and not self.get_variable_comment():
            try:
                more_content = StringList([_('alias of %s') % restify(self.object)], source='')
            except AttributeError:
                pass  # Invalid class object is passed.

        super().add_content(more_content)

    def document_members(self, all_members: bool = False) -> None:
        if self.doc_as_attr:
            return
        super().document_members(all_members)

    def generate(self, more_content: Optional[StringList] = None, real_modname: str = None,
                 check_module: bool = False, all_members: bool = False) -> None:
        # Do not pass real_modname and use the name from the __module__
        # attribute of the class.
        # If a class gets imported into the module real_modname
        # the analyzer won't find the source of the class, if
        # it looks in real_modname.
        return super().generate(more_content=more_content,
                                check_module=check_module,
                                all_members=all_members)
```
### 2 - sphinx/ext/autodoc/__init__.py:

Start line: 448, End line: 496

```python
class Documenter:

    def get_real_modname(self) -> str:
        """Get the real module name of an object to document.

        It can differ from the name of the module through which the object was
        imported.
        """
        return self.get_attr(self.object, '__module__', None) or self.modname

    def check_module(self) -> bool:
        """Check if *self.object* is really defined in the module given by
        *self.modname*.
        """
        if self.options.imported_members:
            return True

        subject = inspect.unpartial(self.object)
        modname = self.get_attr(subject, '__module__', None)
        if modname and modname != self.modname:
            return False
        return True

    def format_args(self, **kwargs: Any) -> str:
        """Format the argument signature of *self.object*.

        Should return None if the object does not have a signature.
        """
        return None

    def format_name(self) -> str:
        """Format the name of *self.object*.

        This normally should be something that can be parsed by the generated
        directive, but doesn't need to be (Sphinx will display it unparsed
        then).
        """
        # normally the name doesn't contain the module (except for module
        # directives of course)
        return '.'.join(self.objpath) or self.modname

    def _call_format_args(self, **kwargs: Any) -> str:
        if kwargs:
            try:
                return self.format_args(**kwargs)
            except TypeError:
                # avoid chaining exceptions, by putting nothing here
                pass

        # retry without arguments for old documenters
        return self.format_args()
```
### 3 - sphinx/ext/autodoc/__init__.py:

Start line: 13, End line: 114

```python
import re
import warnings
from inspect import Parameter, Signature
from types import ModuleType
from typing import (TYPE_CHECKING, Any, Callable, Dict, Iterator, List, Optional, Sequence,
                    Set, Tuple, Type, TypeVar, Union)

from docutils.statemachine import StringList

import sphinx
from sphinx.application import Sphinx
from sphinx.config import ENUM, Config
from sphinx.deprecation import RemovedInSphinx50Warning, RemovedInSphinx60Warning
from sphinx.environment import BuildEnvironment
from sphinx.ext.autodoc.importer import (get_class_members, get_object_members, import_module,
                                         import_object)
from sphinx.ext.autodoc.mock import ismock, mock, undecorate
from sphinx.locale import _, __
from sphinx.pycode import ModuleAnalyzer, PycodeError
from sphinx.util import inspect, logging
from sphinx.util.docstrings import prepare_docstring, separate_metadata
from sphinx.util.inspect import (evaluate_signature, getdoc, object_description, safe_getattr,
                                 stringify_signature)
from sphinx.util.typing import OptionSpec, get_type_hints, restify
from sphinx.util.typing import stringify as stringify_typehint

if TYPE_CHECKING:
    from sphinx.ext.autodoc.directive import DocumenterBridge


logger = logging.getLogger(__name__)


# This type isn't exposed directly in any modules, but can be found
# here in most Python versions
MethodDescriptorType = type(type.__subclasses__)


#: extended signature RE: with explicit module name separated by ::
py_ext_sig_re = re.compile(
    r'''^ ([\w.]+::)?            # explicit module name
          ([\w.]+\.)?            # module and/or class name(s)
          (\w+)  \s*             # thing name
          (?: \((.*)\)           # optional: arguments
           (?:\s* -> \s* (.*))?  #           return annotation
          )? $                   # and nothing more
          ''', re.VERBOSE)
special_member_re = re.compile(r'^__\S+__$')


def identity(x: Any) -> Any:
    return x


class _All:
    """A special value for :*-members: that matches to any member."""

    def __contains__(self, item: Any) -> bool:
        return True

    def append(self, item: Any) -> None:
        pass  # nothing


class _Empty:
    """A special value for :exclude-members: that never matches to any member."""

    def __contains__(self, item: Any) -> bool:
        return False


ALL = _All()
EMPTY = _Empty()
UNINITIALIZED_ATTR = object()
INSTANCEATTR = object()
SLOTSATTR = object()


def members_option(arg: Any) -> Union[object, List[str]]:
    """Used to convert the :members: option to auto directives."""
    if arg in (None, True):
        return ALL
    elif arg is False:
        return None
    else:
        return [x.strip() for x in arg.split(',') if x.strip()]


def members_set_option(arg: Any) -> Union[object, Set[str]]:
    """Used to convert the :members: option to auto directives."""
    warnings.warn("members_set_option() is deprecated.",
                  RemovedInSphinx50Warning, stacklevel=2)
    if arg is None:
        return ALL
    return {x.strip() for x in arg.split(',') if x.strip()}


def exclude_members_option(arg: Any) -> Union[object, Set[str]]:
    """Used to convert the :exclude-members: option."""
    if arg in (None, True):
        return EMPTY
    return {x.strip() for x in arg.split(',') if x.strip()}
```
### 4 - sphinx/ext/autodoc/mock.py:

Start line: 73, End line: 98

```python
def _make_subclass(name: str, module: str, superclass: Any = _MockObject,
                   attributes: Any = None, decorator_args: Tuple = ()) -> Any:
    attrs = {'__module__': module,
             '__display_name__': module + '.' + name,
             '__name__': name,
             '__sphinx_decorator_args__': decorator_args}
    attrs.update(attributes or {})

    return type(name, (superclass,), attrs)


class _MockModule(ModuleType):
    """Used by autodoc_mock_imports."""
    __file__ = os.devnull
    __sphinx_mock__ = True

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.__all__: List[str] = []
        self.__path__: List[str] = []

    def __getattr__(self, name: str) -> _MockObject:
        return _make_subclass(name, self.__name__)()

    def __repr__(self) -> str:
        return self.__name__
```
### 5 - sphinx/ext/autodoc/__init__.py:

Start line: 1045, End line: 1056

```python
class ModuleDocumenter(Documenter):

    def add_directive_header(self, sig: str) -> None:
        Documenter.add_directive_header(self, sig)

        sourcename = self.get_sourcename()

        # add some module-specific options
        if self.options.synopsis:
            self.add_line('   :synopsis: ' + self.options.synopsis, sourcename)
        if self.options.platform:
            self.add_line('   :platform: ' + self.options.platform, sourcename)
        if self.options.deprecated:
            self.add_line('   :deprecated:', sourcename)
```
### 6 - sphinx/ext/autodoc/__init__.py:

Start line: 2808, End line: 2851

```python
# NOQA


def setup(app: Sphinx) -> Dict[str, Any]:
    app.add_autodocumenter(ModuleDocumenter)
    app.add_autodocumenter(ClassDocumenter)
    app.add_autodocumenter(ExceptionDocumenter)
    app.add_autodocumenter(DataDocumenter)
    app.add_autodocumenter(NewTypeDataDocumenter)
    app.add_autodocumenter(FunctionDocumenter)
    app.add_autodocumenter(DecoratorDocumenter)
    app.add_autodocumenter(MethodDocumenter)
    app.add_autodocumenter(AttributeDocumenter)
    app.add_autodocumenter(PropertyDocumenter)
    app.add_autodocumenter(NewTypeAttributeDocumenter)

    app.add_config_value('autoclass_content', 'class', True, ENUM('both', 'class', 'init'))
    app.add_config_value('autodoc_member_order', 'alphabetical', True,
                         ENUM('alphabetic', 'alphabetical', 'bysource', 'groupwise'))
    app.add_config_value('autodoc_class_signature', 'mixed', True, ENUM('mixed', 'separated'))
    app.add_config_value('autodoc_default_options', {}, True)
    app.add_config_value('autodoc_docstring_signature', True, True)
    app.add_config_value('autodoc_mock_imports', [], True)
    app.add_config_value('autodoc_typehints', "signature", True,
                         ENUM("signature", "description", "none", "both"))
    app.add_config_value('autodoc_typehints_description_target', 'all', True,
                         ENUM('all', 'documented'))
    app.add_config_value('autodoc_type_aliases', {}, True)
    app.add_config_value('autodoc_warningiserror', True, True)
    app.add_config_value('autodoc_inherit_docstrings', True, True)
    app.add_event('autodoc-before-process-signature')
    app.add_event('autodoc-process-docstring')
    app.add_event('autodoc-process-signature')
    app.add_event('autodoc-skip-member')
    app.add_event('autodoc-process-bases')

    app.connect('config-inited', migrate_autodoc_member_order, priority=800)

    app.setup_extension('sphinx.ext.autodoc.preserve_defaults')
    app.setup_extension('sphinx.ext.autodoc.type_comment')
    app.setup_extension('sphinx.ext.autodoc.typehints')

    return {'version': sphinx.__display_version__, 'parallel_read_safe': True}
```
### 7 - doc/conf.py:

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
epub_publisher = 'https://www.sphinx-doc.org/'
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
### 8 - sphinx/ext/autodoc/__init__.py:

Start line: 1281, End line: 1400

```python
class FunctionDocumenter(DocstringSignatureMixin, ModuleLevelDocumenter):  # type: ignore
    """
    Specialized Documenter subclass for functions.
    """
    objtype = 'function'
    member_order = 30

    @classmethod
    def can_document_member(cls, member: Any, membername: str, isattr: bool, parent: Any
                            ) -> bool:
        # supports functions, builtins and bound methods exported at the module level
        return (inspect.isfunction(member) or inspect.isbuiltin(member) or
                (inspect.isroutine(member) and isinstance(parent, ModuleDocumenter)))

    def format_args(self, **kwargs: Any) -> str:
        if self.config.autodoc_typehints in ('none', 'description'):
            kwargs.setdefault('show_annotation', False)

        try:
            self.env.app.emit('autodoc-before-process-signature', self.object, False)
            sig = inspect.signature(self.object, type_aliases=self.config.autodoc_type_aliases)
            args = stringify_signature(sig, **kwargs)
        except TypeError as exc:
            logger.warning(__("Failed to get a function signature for %s: %s"),
                           self.fullname, exc)
            return None
        except ValueError:
            args = ''

        if self.config.strip_signature_backslash:
            # escape backslashes for reST
            args = args.replace('\\', '\\\\')
        return args

    def document_members(self, all_members: bool = False) -> None:
        pass

    def add_directive_header(self, sig: str) -> None:
        sourcename = self.get_sourcename()
        super().add_directive_header(sig)

        if inspect.iscoroutinefunction(self.object) or inspect.isasyncgenfunction(self.object):
            self.add_line('   :async:', sourcename)

    def format_signature(self, **kwargs: Any) -> str:
        sigs = []
        if (self.analyzer and
                '.'.join(self.objpath) in self.analyzer.overloads and
                self.config.autodoc_typehints != 'none'):
            # Use signatures for overloaded functions instead of the implementation function.
            overloaded = True
        else:
            overloaded = False
            sig = super().format_signature(**kwargs)
            sigs.append(sig)

        if inspect.is_singledispatch_function(self.object):
            # append signature of singledispatch'ed functions
            for typ, func in self.object.registry.items():
                if typ is object:
                    pass  # default implementation. skipped.
                else:
                    dispatchfunc = self.annotate_to_first_argument(func, typ)
                    if dispatchfunc:
                        documenter = FunctionDocumenter(self.directive, '')
                        documenter.object = dispatchfunc
                        documenter.objpath = [None]
                        sigs.append(documenter.format_signature())
        if overloaded:
            actual = inspect.signature(self.object,
                                       type_aliases=self.config.autodoc_type_aliases)
            __globals__ = safe_getattr(self.object, '__globals__', {})
            for overload in self.analyzer.overloads.get('.'.join(self.objpath)):
                overload = self.merge_default_value(actual, overload)
                overload = evaluate_signature(overload, __globals__,
                                              self.config.autodoc_type_aliases)

                sig = stringify_signature(overload, **kwargs)
                sigs.append(sig)

        return "\n".join(sigs)

    def merge_default_value(self, actual: Signature, overload: Signature) -> Signature:
        """Merge default values of actual implementation to the overload variants."""
        parameters = list(overload.parameters.values())
        for i, param in enumerate(parameters):
            actual_param = actual.parameters.get(param.name)
            if actual_param and param.default == '...':
                parameters[i] = param.replace(default=actual_param.default)

        return overload.replace(parameters=parameters)

    def annotate_to_first_argument(self, func: Callable, typ: Type) -> Optional[Callable]:
        """Annotate type hint to the first argument of function if needed."""
        try:
            sig = inspect.signature(func, type_aliases=self.config.autodoc_type_aliases)
        except TypeError as exc:
            logger.warning(__("Failed to get a function signature for %s: %s"),
                           self.fullname, exc)
            return None
        except ValueError:
            return None

        if len(sig.parameters) == 0:
            return None

        def dummy():
            pass

        params = list(sig.parameters.values())
        if params[0].annotation is Parameter.empty:
            params[0] = params[0].replace(annotation=typ)
            try:
                dummy.__signature__ = sig.replace(parameters=params)  # type: ignore
                return dummy
            except (AttributeError, TypeError):
                # failed to update signature (ex. built-in or extension types)
                return None
        else:
            return None
```
### 9 - sphinx/ext/autodoc/__init__.py:

Start line: 1994, End line: 2019

```python
class DataDocumenter(GenericAliasMixin, NewTypeMixin, TypeVarMixin,
                     UninitializedGlobalVariableMixin, ModuleLevelDocumenter):

    def add_directive_header(self, sig: str) -> None:
        super().add_directive_header(sig)
        sourcename = self.get_sourcename()
        if self.options.annotation is SUPPRESS or self.should_suppress_directive_header():
            pass
        elif self.options.annotation:
            self.add_line('   :annotation: %s' % self.options.annotation,
                          sourcename)
        else:
            if self.config.autodoc_typehints != 'none':
                # obtain annotation for this data
                annotations = get_type_hints(self.parent, None,
                                             self.config.autodoc_type_aliases)
                if self.objpath[-1] in annotations:
                    objrepr = stringify_typehint(annotations.get(self.objpath[-1]))
                    self.add_line('   :type: ' + objrepr, sourcename)

            try:
                if (self.options.no_value or self.should_suppress_value_header() or
                        ismock(self.object)):
                    pass
                else:
                    objrepr = object_description(self.object)
                    self.add_line('   :value: ' + objrepr, sourcename)
            except ValueError:
                pass
```
### 10 - sphinx/ext/autodoc/__init__.py:

Start line: 377, End line: 386

```python
class Documenter:

    def resolve_name(self, modname: str, parents: Any, path: str, base: Any
                     ) -> Tuple[str, List[str]]:
        """Resolve the module and name of the object to document given by the
        arguments and the current module/class.

        Must return a pair of the module name and a chain of attributes; for
        example, it would return ``('zipfile', ['ZipFile', 'open'])`` for the
        ``zipfile.ZipFile.open`` method.
        """
        raise NotImplementedError('must be implemented in subclasses')
```
### 11 - sphinx/ext/autodoc/__init__.py:

Start line: 2587, End line: 2610

```python
class AttributeDocumenter(GenericAliasMixin, NewTypeMixin, SlotsMixin,  # type: ignore
                          TypeVarMixin, RuntimeInstanceAttributeMixin,
                          UninitializedInstanceAttributeMixin, NonDataDescriptorMixin,
                          DocstringStripSignatureMixin, ClassLevelDocumenter):

    def import_object(self, raiseerror: bool = False) -> bool:
        ret = super().import_object(raiseerror)
        if inspect.isenumattribute(self.object):
            self.object = self.object.value
        if self.parent:
            self.update_annotations(self.parent)

        return ret

    def get_real_modname(self) -> str:
        real_modname = self.get_attr(self.parent or self.object, '__module__', None)
        return real_modname or self.modname

    def should_suppress_value_header(self) -> bool:
        if super().should_suppress_value_header():
            return True
        else:
            doc = self.get_doc()
            if doc:
                docstring, metadata = separate_metadata('\n'.join(sum(doc, [])))
                if 'hide-value' in metadata:
                    return True

        return False
```
### 12 - sphinx/ext/autodoc/__init__.py:

Start line: 1031, End line: 1043

```python
class ModuleDocumenter(Documenter):

    def import_object(self, raiseerror: bool = False) -> bool:
        ret = super().import_object(raiseerror)

        try:
            if not self.options.ignore_module_all:
                self.__all__ = inspect.getall(self.object)
        except ValueError as exc:
            # invalid __all__ found.
            logger.warning(__('__all__ should be a list of strings, not %r '
                              '(in module %s) -- ignoring __all__') %
                           (exc.args[0], self.fullname), type='autodoc')

        return ret
```
### 13 - sphinx/util/inspect.py:

Start line: 559, End line: 589

```python
class TypeAliasModule:
    """Pseudo module class for autodoc_type_aliases."""

    def __init__(self, modname: str, mapping: Dict[str, str]) -> None:
        self.__modname = modname
        self.__mapping = mapping

        self.__module: Optional[ModuleType] = None

    def __getattr__(self, name: str) -> Any:
        fullname = '.'.join(filter(None, [self.__modname, name]))
        if fullname in self.__mapping:
            # exactly matched
            return TypeAliasForwardRef(self.__mapping[fullname])
        else:
            prefix = fullname + '.'
            nested = {k: v for k, v in self.__mapping.items() if k.startswith(prefix)}
            if nested:
                # sub modules or classes found
                return TypeAliasModule(fullname, nested)
            else:
                # no sub modules or classes found.
                try:
                    # return the real submodule if exists
                    return import_module(fullname)
                except ImportError:
                    # return the real class
                    if self.__module is None:
                        self.__module = import_module(self.__modname)

                    return getattr(self.__module, name)
```
### 14 - sphinx/ext/autodoc/__init__.py:

Start line: 2079, End line: 2284

```python
class MethodDocumenter(DocstringSignatureMixin, ClassLevelDocumenter):  # type: ignore
    """
    Specialized Documenter subclass for methods (normal, static and class).
    """
    objtype = 'method'
    directivetype = 'method'
    member_order = 50
    priority = 1  # must be more than FunctionDocumenter

    @classmethod
    def can_document_member(cls, member: Any, membername: str, isattr: bool, parent: Any
                            ) -> bool:
        return inspect.isroutine(member) and not isinstance(parent, ModuleDocumenter)

    def import_object(self, raiseerror: bool = False) -> bool:
        ret = super().import_object(raiseerror)
        if not ret:
            return ret

        # to distinguish classmethod/staticmethod
        obj = self.parent.__dict__.get(self.object_name)
        if obj is None:
            obj = self.object

        if (inspect.isclassmethod(obj) or
                inspect.isstaticmethod(obj, cls=self.parent, name=self.object_name)):
            # document class and static members before ordinary ones
            self.member_order = self.member_order - 1

        return ret

    def format_args(self, **kwargs: Any) -> str:
        if self.config.autodoc_typehints in ('none', 'description'):
            kwargs.setdefault('show_annotation', False)

        try:
            if self.object == object.__init__ and self.parent != object:
                # Classes not having own __init__() method are shown as no arguments.
                #
                # Note: The signature of object.__init__() is (self, /, *args, **kwargs).
                #       But it makes users confused.
                args = '()'
            else:
                if inspect.isstaticmethod(self.object, cls=self.parent, name=self.object_name):
                    self.env.app.emit('autodoc-before-process-signature', self.object, False)
                    sig = inspect.signature(self.object, bound_method=False,
                                            type_aliases=self.config.autodoc_type_aliases)
                else:
                    self.env.app.emit('autodoc-before-process-signature', self.object, True)
                    sig = inspect.signature(self.object, bound_method=True,
                                            type_aliases=self.config.autodoc_type_aliases)
                args = stringify_signature(sig, **kwargs)
        except TypeError as exc:
            logger.warning(__("Failed to get a method signature for %s: %s"),
                           self.fullname, exc)
            return None
        except ValueError:
            args = ''

        if self.config.strip_signature_backslash:
            # escape backslashes for reST
            args = args.replace('\\', '\\\\')
        return args

    def add_directive_header(self, sig: str) -> None:
        super().add_directive_header(sig)

        sourcename = self.get_sourcename()
        obj = self.parent.__dict__.get(self.object_name, self.object)
        if inspect.isabstractmethod(obj):
            self.add_line('   :abstractmethod:', sourcename)
        if inspect.iscoroutinefunction(obj) or inspect.isasyncgenfunction(obj):
            self.add_line('   :async:', sourcename)
        if inspect.isclassmethod(obj):
            self.add_line('   :classmethod:', sourcename)
        if inspect.isstaticmethod(obj, cls=self.parent, name=self.object_name):
            self.add_line('   :staticmethod:', sourcename)
        if self.analyzer and '.'.join(self.objpath) in self.analyzer.finals:
            self.add_line('   :final:', sourcename)

    def document_members(self, all_members: bool = False) -> None:
        pass

    def format_signature(self, **kwargs: Any) -> str:
        sigs = []
        if (self.analyzer and
                '.'.join(self.objpath) in self.analyzer.overloads and
                self.config.autodoc_typehints != 'none'):
            # Use signatures for overloaded methods instead of the implementation method.
            overloaded = True
        else:
            overloaded = False
            sig = super().format_signature(**kwargs)
            sigs.append(sig)

        meth = self.parent.__dict__.get(self.objpath[-1])
        if inspect.is_singledispatch_method(meth):
            # append signature of singledispatch'ed functions
            for typ, func in meth.dispatcher.registry.items():
                if typ is object:
                    pass  # default implementation. skipped.
                else:
                    dispatchmeth = self.annotate_to_first_argument(func, typ)
                    if dispatchmeth:
                        documenter = MethodDocumenter(self.directive, '')
                        documenter.parent = self.parent
                        documenter.object = dispatchmeth
                        documenter.objpath = [None]
                        sigs.append(documenter.format_signature())
        if overloaded:
            if inspect.isstaticmethod(self.object, cls=self.parent, name=self.object_name):
                actual = inspect.signature(self.object, bound_method=False,
                                           type_aliases=self.config.autodoc_type_aliases)
            else:
                actual = inspect.signature(self.object, bound_method=True,
                                           type_aliases=self.config.autodoc_type_aliases)

            __globals__ = safe_getattr(self.object, '__globals__', {})
            for overload in self.analyzer.overloads.get('.'.join(self.objpath)):
                overload = self.merge_default_value(actual, overload)
                overload = evaluate_signature(overload, __globals__,
                                              self.config.autodoc_type_aliases)

                if not inspect.isstaticmethod(self.object, cls=self.parent,
                                              name=self.object_name):
                    parameters = list(overload.parameters.values())
                    overload = overload.replace(parameters=parameters[1:])
                sig = stringify_signature(overload, **kwargs)
                sigs.append(sig)

        return "\n".join(sigs)

    def merge_default_value(self, actual: Signature, overload: Signature) -> Signature:
        """Merge default values of actual implementation to the overload variants."""
        parameters = list(overload.parameters.values())
        for i, param in enumerate(parameters):
            actual_param = actual.parameters.get(param.name)
            if actual_param and param.default == '...':
                parameters[i] = param.replace(default=actual_param.default)

        return overload.replace(parameters=parameters)

    def annotate_to_first_argument(self, func: Callable, typ: Type) -> Optional[Callable]:
        """Annotate type hint to the first argument of function if needed."""
        try:
            sig = inspect.signature(func, type_aliases=self.config.autodoc_type_aliases)
        except TypeError as exc:
            logger.warning(__("Failed to get a method signature for %s: %s"),
                           self.fullname, exc)
            return None
        except ValueError:
            return None

        if len(sig.parameters) == 1:
            return None

        def dummy():
            pass

        params = list(sig.parameters.values())
        if params[1].annotation is Parameter.empty:
            params[1] = params[1].replace(annotation=typ)
            try:
                dummy.__signature__ = sig.replace(parameters=params)  # type: ignore
                return dummy
            except (AttributeError, TypeError):
                # failed to update signature (ex. built-in or extension types)
                return None
        else:
            return None

    def get_doc(self, ignore: int = None) -> Optional[List[List[str]]]:
        if self._new_docstrings is not None:
            # docstring already returned previously, then modified by
            # `DocstringSignatureMixin`.  Just return the previously-computed
            # result, so that we don't lose the processing done by
            # `DocstringSignatureMixin`.
            return self._new_docstrings
        if self.objpath[-1] == '__init__':
            docstring = getdoc(self.object, self.get_attr,
                               self.config.autodoc_inherit_docstrings,
                               self.parent, self.object_name)
            if (docstring is not None and
                (docstring == object.__init__.__doc__ or  # for pypy
                 docstring.strip() == object.__init__.__doc__)):  # for !pypy
                docstring = None
            if docstring:
                tab_width = self.directive.state.document.settings.tab_width
                return [prepare_docstring(docstring, tabsize=tab_width)]
            else:
                return []
        elif self.objpath[-1] == '__new__':
            docstring = getdoc(self.object, self.get_attr,
                               self.config.autodoc_inherit_docstrings,
                               self.parent, self.object_name)
            if (docstring is not None and
                (docstring == object.__new__.__doc__ or  # for pypy
                 docstring.strip() == object.__new__.__doc__)):  # for !pypy
                docstring = None
            if docstring:
                tab_width = self.directive.state.document.settings.tab_width
                return [prepare_docstring(docstring, tabsize=tab_width)]
            else:
                return []
        else:
            return super().get_doc()
```
### 15 - sphinx/ext/autodoc/__init__.py:

Start line: 1085, End line: 1109

```python
class ModuleDocumenter(Documenter):

    def get_object_members(self, want_all: bool) -> Tuple[bool, ObjectMembers]:
        members = self.get_module_members()
        if want_all:
            if self.__all__ is None:
                # for implicit module members, check __module__ to avoid
                # documenting imported objects
                return True, list(members.values())
            else:
                for member in members.values():
                    if member.__name__ not in self.__all__:
                        member.skipped = True

                return False, list(members.values())
        else:
            memberlist = self.options.members or []
            ret = []
            for name in memberlist:
                if name in members:
                    ret.append(members[name])
                else:
                    logger.warning(__('missing attribute mentioned in :members: option: '
                                      'module %s, attribute %s') %
                                   (safe_getattr(self.object, '__name__', '???'), name),
                                   type='autodoc')
            return False, ret
```
### 16 - sphinx/ext/autodoc/__init__.py:

Start line: 2021, End line: 2058

```python
class DataDocumenter(GenericAliasMixin, NewTypeMixin, TypeVarMixin,
                     UninitializedGlobalVariableMixin, ModuleLevelDocumenter):

    def document_members(self, all_members: bool = False) -> None:
        pass

    def get_real_modname(self) -> str:
        real_modname = self.get_attr(self.parent or self.object, '__module__', None)
        return real_modname or self.modname

    def get_module_comment(self, attrname: str) -> Optional[List[str]]:
        try:
            analyzer = ModuleAnalyzer.for_module(self.modname)
            analyzer.analyze()
            key = ('', attrname)
            if key in analyzer.attr_docs:
                return list(analyzer.attr_docs[key])
        except PycodeError:
            pass

        return None

    def get_doc(self, ignore: int = None) -> Optional[List[List[str]]]:
        # Check the variable has a docstring-comment
        comment = self.get_module_comment(self.objpath[-1])
        if comment:
            return [comment]
        else:
            return super().get_doc(ignore)

    def add_content(self, more_content: Optional[StringList], no_docstring: bool = False
                    ) -> None:
        # Disable analyzing variable comment on Documenter.add_content() to control it on
        # DataDocumenter.add_content()
        self.analyzer = None

        if not more_content:
            more_content = StringList()

        self.update_content(more_content)
        super().add_content(more_content, no_docstring=no_docstring)
```
### 18 - sphinx/ext/autodoc/__init__.py:

Start line: 2612, End line: 2636

```python
class AttributeDocumenter(GenericAliasMixin, NewTypeMixin, SlotsMixin,  # type: ignore
                          TypeVarMixin, RuntimeInstanceAttributeMixin,
                          UninitializedInstanceAttributeMixin, NonDataDescriptorMixin,
                          DocstringStripSignatureMixin, ClassLevelDocumenter):

    def add_directive_header(self, sig: str) -> None:
        super().add_directive_header(sig)
        sourcename = self.get_sourcename()
        if self.options.annotation is SUPPRESS or self.should_suppress_directive_header():
            pass
        elif self.options.annotation:
            self.add_line('   :annotation: %s' % self.options.annotation, sourcename)
        else:
            if self.config.autodoc_typehints != 'none':
                # obtain type annotation for this attribute
                annotations = get_type_hints(self.parent, None,
                                             self.config.autodoc_type_aliases)
                if self.objpath[-1] in annotations:
                    objrepr = stringify_typehint(annotations.get(self.objpath[-1]))
                    self.add_line('   :type: ' + objrepr, sourcename)

            try:
                if (self.options.no_value or self.should_suppress_value_header() or
                        ismock(self.object)):
                    pass
                else:
                    objrepr = object_description(self.object)
                    self.add_line('   :value: ' + objrepr, sourcename)
            except ValueError:
                pass
```
### 19 - sphinx/ext/autodoc/__init__.py:

Start line: 117, End line: 158

```python
def inherited_members_option(arg: Any) -> Union[object, Set[str]]:
    """Used to convert the :members: option to auto directives."""
    if arg in (None, True):
        return 'object'
    else:
        return arg


def member_order_option(arg: Any) -> Optional[str]:
    """Used to convert the :members: option to auto directives."""
    if arg in (None, True):
        return None
    elif arg in ('alphabetical', 'bysource', 'groupwise'):
        return arg
    else:
        raise ValueError(__('invalid value for member-order option: %s') % arg)


def class_doc_from_option(arg: Any) -> Optional[str]:
    """Used to convert the :class-doc-from: option to autoclass directives."""
    if arg in ('both', 'class', 'init'):
        return arg
    else:
        raise ValueError(__('invalid value for class-doc-from option: %s') % arg)


SUPPRESS = object()


def annotation_option(arg: Any) -> Any:
    if arg in (None, True):
        # suppress showing the representation of the object
        return SUPPRESS
    else:
        return arg


def bool_option(arg: Any) -> bool:
    """Used to convert flag options to auto directives.  (Instead of
    directives.flag(), which returns None).
    """
    return True
```
### 20 - sphinx/ext/autodoc/__init__.py:

Start line: 1244, End line: 1260

```python
class DocstringSignatureMixin:

    def get_doc(self, ignore: int = None) -> List[List[str]]:
        if self._new_docstrings is not None:
            return self._new_docstrings
        return super().get_doc(ignore)  # type: ignore

    def format_signature(self, **kwargs: Any) -> str:
        if self.args is None and self.config.autodoc_docstring_signature:  # type: ignore
            # only act if a signature is not explicitly given already, and if
            # the feature is enabled
            result = self._find_signature()
            if result is not None:
                self.args, self.retann = result
        sig = super().format_signature(**kwargs)  # type: ignore
        if self._signatures:
            return "\n".join([sig] + self._signatures)
        else:
            return sig
```
### 23 - sphinx/ext/autodoc/__init__.py:

Start line: 1131, End line: 1149

```python
class ModuleLevelDocumenter(Documenter):
    """
    Specialized Documenter subclass for objects on module level (functions,
    classes, data/constants).
    """
    def resolve_name(self, modname: str, parents: Any, path: str, base: Any
                     ) -> Tuple[str, List[str]]:
        if modname is None:
            if path:
                modname = path.rstrip('.')
            else:
                # if documenting a toplevel object without explicit module,
                # it can be contained in another auto directive ...
                modname = self.env.temp_data.get('autodoc:module')
                # ... or in the scope of a module directive
                if not modname:
                    modname = self.env.ref_context.get('py:module')
                # ... else, it stays None, which means invalid
        return modname, parents + [base]
```
### 24 - sphinx/ext/autodoc/__init__.py:

Start line: 388, End line: 423

```python
class Documenter:

    def parse_name(self) -> bool:
        """Determine what module to import and what attribute to document.

        Returns True and sets *self.modname*, *self.objpath*, *self.fullname*,
        *self.args* and *self.retann* if parsing and resolving was successful.
        """
        # first, parse the definition -- auto directives for classes and
        # functions can contain a signature which is then used instead of
        # an autogenerated one
        try:
            matched = py_ext_sig_re.match(self.name)
            explicit_modname, path, base, args, retann = matched.groups()
        except AttributeError:
            logger.warning(__('invalid signature for auto%s (%r)') % (self.objtype, self.name),
                           type='autodoc')
            return False

        # support explicit module and class name separation via ::
        if explicit_modname is not None:
            modname = explicit_modname[:-2]
            parents = path.rstrip('.').split('.') if path else []
        else:
            modname = None
            parents = []

        with mock(self.config.autodoc_mock_imports):
            self.modname, self.objpath = self.resolve_name(modname, parents, path, base)

        if not self.modname:
            return False

        self.args = args
        self.retann = retann
        self.fullname = ((self.modname or '') +
                         ('.' + '.'.join(self.objpath) if self.objpath else ''))
        return True
```
### 25 - sphinx/ext/autodoc/__init__.py:

Start line: 299, End line: 375

```python
class Documenter:
    """
    A Documenter knows how to autodocument a single object type.  When
    registered with the AutoDirective, it will be used to document objects
    of that type when needed by autodoc.

    Its *objtype* attribute selects what auto directive it is assigned to
    (the directive name is 'auto' + objtype), and what directive it generates
    by default, though that can be overridden by an attribute called
    *directivetype*.

    A Documenter has an *option_spec* that works like a docutils directive's;
    in fact, it will be used to parse an auto directive's options that matches
    the Documenter.
    """
    #: name by which the directive is called (auto...) and the default
    #: generated directive name
    objtype = 'object'
    #: indentation by which to indent the directive content
    content_indent = '   '
    #: priority if multiple documenters return True from can_document_member
    priority = 0
    #: order if autodoc_member_order is set to 'groupwise'
    member_order = 0
    #: true if the generated content may contain titles
    titles_allowed = False

    option_spec: OptionSpec = {
        'noindex': bool_option
    }

    def get_attr(self, obj: Any, name: str, *defargs: Any) -> Any:
        """getattr() override for types such as Zope interfaces."""
        return autodoc_attrgetter(self.env.app, obj, name, *defargs)

    @classmethod
    def can_document_member(cls, member: Any, membername: str, isattr: bool, parent: Any
                            ) -> bool:
        """Called to see if a member can be documented by this Documenter."""
        raise NotImplementedError('must be implemented in subclasses')

    def __init__(self, directive: "DocumenterBridge", name: str, indent: str = '') -> None:
        self.directive = directive
        self.config: Config = directive.env.config
        self.env: BuildEnvironment = directive.env
        self.options = directive.genopt
        self.name = name
        self.indent = indent
        # the module and object path within the module, and the fully
        # qualified name (all set after resolve_name succeeds)
        self.modname: str = None
        self.module: ModuleType = None
        self.objpath: List[str] = None
        self.fullname: str = None
        # extra signature items (arguments and return annotation,
        # also set after resolve_name succeeds)
        self.args: str = None
        self.retann: str = None
        # the object to document (set after import_object succeeds)
        self.object: Any = None
        self.object_name: str = None
        # the parent/owner of the object to document
        self.parent: Any = None
        # the module analyzer to get at attribute docs, or None
        self.analyzer: ModuleAnalyzer = None

    @property
    def documenters(self) -> Dict[str, Type["Documenter"]]:
        """Returns registered Documenter classes"""
        return self.env.app.registry.documenters

    def add_line(self, line: str, source: str, *lineno: int) -> None:
        """Append one line of generated reST to the output."""
        if line.strip():  # not a blank line
            self.directive.result.append(self.indent + line, source, *lineno)
        else:
            self.directive.result.append('', source, *lineno)
```
### 27 - sphinx/ext/autodoc/__init__.py:

Start line: 987, End line: 1029

```python
class ModuleDocumenter(Documenter):
    """
    Specialized Documenter subclass for modules.
    """
    objtype = 'module'
    content_indent = ''
    titles_allowed = True

    option_spec: OptionSpec = {
        'members': members_option, 'undoc-members': bool_option,
        'noindex': bool_option, 'inherited-members': inherited_members_option,
        'show-inheritance': bool_option, 'synopsis': identity,
        'platform': identity, 'deprecated': bool_option,
        'member-order': member_order_option, 'exclude-members': exclude_members_option,
        'private-members': members_option, 'special-members': members_option,
        'imported-members': bool_option, 'ignore-module-all': bool_option
    }

    def __init__(self, *args: Any) -> None:
        super().__init__(*args)
        merge_members_option(self.options)
        self.__all__: Optional[Sequence[str]] = None

    @classmethod
    def can_document_member(cls, member: Any, membername: str, isattr: bool, parent: Any
                            ) -> bool:
        # don't document submodules automatically
        return False

    def resolve_name(self, modname: str, parents: Any, path: str, base: Any
                     ) -> Tuple[str, List[str]]:
        if modname is not None:
            logger.warning(__('"::" in automodule name doesn\'t make sense'),
                           type='autodoc')
        return (path or '') + base, []

    def parse_name(self) -> bool:
        ret = super().parse_name()
        if self.args or self.retann:
            logger.warning(__('signature arguments or return annotation '
                              'given for automodule %s') % self.fullname,
                           type='autodoc')
        return ret
```
### 28 - sphinx/ext/autodoc/__init__.py:

Start line: 1825, End line: 1840

```python
class GenericAliasMixin(DataDocumenterMixinBase):
    """
    Mixin for DataDocumenter and AttributeDocumenter to provide the feature for
    supporting GenericAliases.
    """

    def should_suppress_directive_header(self) -> bool:
        return (inspect.isgenericalias(self.object) or
                super().should_suppress_directive_header())

    def update_content(self, more_content: StringList) -> None:
        if inspect.isgenericalias(self.object):
            more_content.append(_('alias of %s') % restify(self.object), '')
            more_content.append('', '')

        super().update_content(more_content)
```
### 29 - sphinx/ext/autodoc/__init__.py:

Start line: 713, End line: 821

```python
class Documenter:

    def filter_members(self, members: ObjectMembers, want_all: bool
                       ) -> List[Tuple[str, Any, bool]]:
        # ... other code
        for obj in members:
            membername, member = obj
            # if isattr is True, the member is documented as an attribute
            if member is INSTANCEATTR:
                isattr = True
            elif (namespace, membername) in attr_docs:
                isattr = True
            else:
                isattr = False

            doc = getdoc(member, self.get_attr, self.config.autodoc_inherit_docstrings,
                         self.object, membername)
            if not isinstance(doc, str):
                # Ignore non-string __doc__
                doc = None

            # if the member __doc__ is the same as self's __doc__, it's just
            # inherited and therefore not the member's doc
            cls = self.get_attr(member, '__class__', None)
            if cls:
                cls_doc = self.get_attr(cls, '__doc__', None)
                if cls_doc == doc:
                    doc = None

            if isinstance(obj, ObjectMember) and obj.docstring:
                # hack for ClassDocumenter to inject docstring via ObjectMember
                doc = obj.docstring

            doc, metadata = separate_metadata(doc)
            has_doc = bool(doc)

            if 'private' in metadata:
                # consider a member private if docstring has "private" metadata
                isprivate = True
            elif 'public' in metadata:
                # consider a member public if docstring has "public" metadata
                isprivate = False
            else:
                isprivate = membername.startswith('_')

            keep = False
            if ismock(member) and (namespace, membername) not in attr_docs:
                # mocked module or object
                pass
            elif self.options.exclude_members and membername in self.options.exclude_members:
                # remove members given by exclude-members
                keep = False
            elif want_all and special_member_re.match(membername):
                # special __methods__
                if self.options.special_members and membername in self.options.special_members:
                    if membername == '__doc__':
                        keep = False
                    elif is_filtered_inherited_member(membername, obj):
                        keep = False
                    else:
                        keep = has_doc or self.options.undoc_members
                else:
                    keep = False
            elif (namespace, membername) in attr_docs:
                if want_all and isprivate:
                    if self.options.private_members is None:
                        keep = False
                    else:
                        keep = membername in self.options.private_members
                else:
                    # keep documented attributes
                    keep = True
            elif want_all and isprivate:
                if has_doc or self.options.undoc_members:
                    if self.options.private_members is None:
                        keep = False
                    elif is_filtered_inherited_member(membername, obj):
                        keep = False
                    else:
                        keep = membername in self.options.private_members
                else:
                    keep = False
            else:
                if (self.options.members is ALL and
                        is_filtered_inherited_member(membername, obj)):
                    keep = False
                else:
                    # ignore undocumented members if :undoc-members: is not given
                    keep = has_doc or self.options.undoc_members

            if isinstance(obj, ObjectMember) and obj.skipped:
                # forcedly skipped member (ex. a module attribute not defined in __all__)
                keep = False

            # give the user a chance to decide whether this member
            # should be skipped
            if self.env.app:
                # let extensions preprocess docstrings
                try:
                    skip_user = self.env.app.emit_firstresult(
                        'autodoc-skip-member', self.objtype, membername, member,
                        not keep, self.options)
                    if skip_user is not None:
                        keep = not skip_user
                except Exception as exc:
                    logger.warning(__('autodoc: failed to determine %r to be documented, '
                                      'the following exception was raised:\n%s'),
                                   member, exc, type='autodoc')
                    keep = False

            if keep:
                ret.append((membername, member, isattr))

        return ret
```
### 30 - sphinx/ext/autodoc/__init__.py:

Start line: 587, End line: 599

```python
class Documenter:

    def get_sourcename(self) -> str:
        if (inspect.safe_getattr(self.object, '__module__', None) and
                inspect.safe_getattr(self.object, '__qualname__', None)):
            # Get the correct location of docstring from self.object
            # to support inherited methods
            fullname = '%s.%s' % (self.object.__module__, self.object.__qualname__)
        else:
            fullname = self.fullname

        if self.analyzer:
            return '%s:docstring of %s' % (self.analyzer.srcname, fullname)
        else:
            return 'docstring of %s' % fullname
```
### 31 - sphinx/domains/python.py:

Start line: 1054, End line: 1072

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
    option_spec: OptionSpec = {}

    def run(self) -> List[Node]:
        modname = self.arguments[0].strip()
        if modname == 'None':
            self.env.ref_context.pop('py:module', None)
        else:
            self.env.ref_context['py:module'] = modname
        return []
```
### 33 - sphinx/ext/autodoc/__init__.py:

Start line: 1111, End line: 1128

```python
class ModuleDocumenter(Documenter):

    def sort_members(self, documenters: List[Tuple["Documenter", bool]],
                     order: str) -> List[Tuple["Documenter", bool]]:
        if order == 'bysource' and self.__all__:
            # Sort alphabetically first (for members not listed on the __all__)
            documenters.sort(key=lambda e: e[0].name)

            # Sort by __all__
            def keyfunc(entry: Tuple[Documenter, bool]) -> int:
                name = entry[0].name.split('::')[1]
                if self.__all__ and name in self.__all__:
                    return self.__all__.index(name)
                else:
                    return len(self.__all__)
            documenters.sort(key=keyfunc)

            return documenters
        else:
            return super().sort_members(documenters, order)
```
### 34 - sphinx/ext/autodoc/__init__.py:

Start line: 2655, End line: 2680

```python
class AttributeDocumenter(GenericAliasMixin, NewTypeMixin, SlotsMixin,  # type: ignore
                          TypeVarMixin, RuntimeInstanceAttributeMixin,
                          UninitializedInstanceAttributeMixin, NonDataDescriptorMixin,
                          DocstringStripSignatureMixin, ClassLevelDocumenter):

    def get_doc(self, ignore: int = None) -> Optional[List[List[str]]]:
        # Check the attribute has a docstring-comment
        comment = self.get_attribute_comment(self.parent, self.objpath[-1])
        if comment:
            return [comment]

        try:
            # Disable `autodoc_inherit_docstring` temporarily to avoid to obtain
            # a docstring from the value which descriptor returns unexpectedly.
            # ref: https://github.com/sphinx-doc/sphinx/issues/7805
            orig = self.config.autodoc_inherit_docstrings
            self.config.autodoc_inherit_docstrings = False  # type: ignore
            return super().get_doc(ignore)
        finally:
            self.config.autodoc_inherit_docstrings = orig  # type: ignore

    def add_content(self, more_content: Optional[StringList], no_docstring: bool = False
                    ) -> None:
        # Disable analyzing attribute comment on Documenter.add_content() to control it on
        # AttributeDocumenter.add_content()
        self.analyzer = None

        if more_content is None:
            more_content = StringList()
        self.update_content(more_content)
        super().add_content(more_content, no_docstring)
```
### 37 - sphinx/ext/autodoc/__init__.py:

Start line: 1403, End line: 1784

```python
class DecoratorDocumenter(FunctionDocumenter):
    """
    Specialized Documenter subclass for decorator functions.
    """
    objtype = 'decorator'

    # must be lower than FunctionDocumenter
    priority = -1

    def format_args(self, **kwargs: Any) -> Any:
        args = super().format_args(**kwargs)
        if ',' in args:
            return args
        else:
            return None


# Types which have confusing metaclass signatures it would be best not to show.
# These are listed by name, rather than storing the objects themselves, to avoid
# needing to import the modules.
_METACLASS_CALL_BLACKLIST = [
    'enum.EnumMeta.__call__',
]


# Types whose __new__ signature is a pass-through.
_CLASS_NEW_BLACKLIST = [
    'typing.Generic.__new__',
]


class ClassDocumenter(DocstringSignatureMixin, ModuleLevelDocumenter):
```
### 38 - sphinx/ext/autodoc/__init__.py:

Start line: 2778, End line: 2808

```python
def get_documenters(app: Sphinx) -> Dict[str, Type[Documenter]]:
    """Returns registered Documenter classes"""
    warnings.warn("get_documenters() is deprecated.", RemovedInSphinx50Warning, stacklevel=2)
    return app.registry.documenters


def autodoc_attrgetter(app: Sphinx, obj: Any, name: str, *defargs: Any) -> Any:
    """Alternative getattr() for types"""
    for typ, func in app.registry.autodoc_attrgettrs.items():
        if isinstance(obj, typ):
            return func(obj, name, *defargs)

    return safe_getattr(obj, name, *defargs)


def migrate_autodoc_member_order(app: Sphinx, config: Config) -> None:
    if config.autodoc_member_order == 'alphabetic':
        # RemovedInSphinx50Warning
        logger.warning(__('autodoc_member_order now accepts "alphabetical" '
                          'instead of "alphabetic". Please update your setting.'))
        config.autodoc_member_order = 'alphabetical'  # type: ignore


# for compatibility
from sphinx.ext.autodoc.deprecated import DataDeclarationDocumenter  # NOQA
from sphinx.ext.autodoc.deprecated import GenericAliasDocumenter  # NOQA
from sphinx.ext.autodoc.deprecated import InstanceAttributeDocumenter  # NOQA
from sphinx.ext.autodoc.deprecated import SingledispatchFunctionDocumenter  # NOQA
from sphinx.ext.autodoc.deprecated import SingledispatchMethodDocumenter  # NOQA
from sphinx.ext.autodoc.deprecated import SlotsAttributeDocumenter  # NOQA
from sphinx.ext.autodoc.deprecated import TypeVarDocumenter
```
### 39 - sphinx/ext/autodoc/__init__.py:

Start line: 2565, End line: 2585

```python
class AttributeDocumenter(GenericAliasMixin, NewTypeMixin, SlotsMixin,  # type: ignore
                          TypeVarMixin, RuntimeInstanceAttributeMixin,
                          UninitializedInstanceAttributeMixin, NonDataDescriptorMixin,
                          DocstringStripSignatureMixin, ClassLevelDocumenter):

    def update_annotations(self, parent: Any) -> None:
        """Update __annotations__ to support type_comment and so on."""
        try:
            annotations = dict(inspect.getannotations(parent))
            parent.__annotations__ = annotations

            for cls in inspect.getmro(parent):
                try:
                    module = safe_getattr(cls, '__module__')
                    qualname = safe_getattr(cls, '__qualname__')

                    analyzer = ModuleAnalyzer.for_module(module)
                    analyzer.analyze()
                    for (classname, attrname), annotation in analyzer.annotations.items():
                        if classname == qualname and attrname not in annotations:
                            annotations[attrname] = annotation
                except (AttributeError, PycodeError):
                    pass
        except (AttributeError, TypeError):
            # Failed to set __annotations__ (built-in, extensions, etc.)
            pass
```
### 41 - sphinx/ext/autodoc/__init__.py:

Start line: 2638, End line: 2653

```python
class AttributeDocumenter(GenericAliasMixin, NewTypeMixin, SlotsMixin,  # type: ignore
                          TypeVarMixin, RuntimeInstanceAttributeMixin,
                          UninitializedInstanceAttributeMixin, NonDataDescriptorMixin,
                          DocstringStripSignatureMixin, ClassLevelDocumenter):

    def get_attribute_comment(self, parent: Any, attrname: str) -> Optional[List[str]]:
        for cls in inspect.getmro(parent):
            try:
                module = safe_getattr(cls, '__module__')
                qualname = safe_getattr(cls, '__qualname__')

                analyzer = ModuleAnalyzer.for_module(module)
                analyzer.analyze()
                if qualname and self.objpath:
                    key = (qualname, attrname)
                    if key in analyzer.attr_docs:
                        return list(analyzer.attr_docs[key])
            except (AttributeError, PycodeError):
                pass

        return None
```
### 43 - sphinx/util/inspect.py:

Start line: 592, End line: 623

```python
class TypeAliasNamespace(Dict[str, Any]):
    """Pseudo namespace class for autodoc_type_aliases.

    This enables to look up nested modules and classes like `mod1.mod2.Class`.
    """

    def __init__(self, mapping: Dict[str, str]) -> None:
        self.__mapping = mapping

    def __getitem__(self, key: str) -> Any:
        if key in self.__mapping:
            # exactly matched
            return TypeAliasForwardRef(self.__mapping[key])
        else:
            prefix = key + '.'
            nested = {k: v for k, v in self.__mapping.items() if k.startswith(prefix)}
            if nested:
                # sub modules or classes found
                return TypeAliasModule(key, nested)
            else:
                raise KeyError


def _should_unwrap(subject: Callable) -> bool:
    """Check the function should be unwrapped on getting signature."""
    __globals__ = getglobals(subject)
    if (__globals__.get('__name__') == 'contextlib' and
            __globals__.get('__file__') == contextlib.__file__):
        # contextmanger should be unwrapped
        return True

    return False
```
### 45 - sphinx/ext/autodoc/__init__.py:

Start line: 533, End line: 553

```python
class Documenter:

    def add_directive_header(self, sig: str) -> None:
        """Add the directive header and options to the generated content."""
        domain = getattr(self, 'domain', 'py')
        directive = getattr(self, 'directivetype', self.objtype)
        name = self.format_name()
        sourcename = self.get_sourcename()

        # one signature per line, indented by column
        prefix = '.. %s:%s:: ' % (domain, directive)
        for i, sig_line in enumerate(sig.split("\n")):
            self.add_line('%s%s%s' % (prefix, name, sig_line),
                          sourcename)
            if i == 0:
                prefix = " " * len(prefix)

        if self.options.noindex:
            self.add_line('   :noindex:', sourcename)
        if self.objpath:
            # Be explicit about the module, this is necessary since .. class::
            # etc. don't support a prepended module name
            self.add_line('   :module: %s' % self.modname, sourcename)
```
### 46 - sphinx/ext/autodoc/__init__.py:

Start line: 1904, End line: 1942

```python
class UninitializedGlobalVariableMixin(DataDocumenterMixinBase):
    """
    Mixin for DataDocumenter to provide the feature for supporting uninitialized
    (type annotation only) global variables.
    """

    def import_object(self, raiseerror: bool = False) -> bool:
        try:
            return super().import_object(raiseerror=True)  # type: ignore
        except ImportError as exc:
            # annotation only instance variable (PEP-526)
            try:
                with mock(self.config.autodoc_mock_imports):
                    parent = import_module(self.modname, self.config.autodoc_warningiserror)
                    annotations = get_type_hints(parent, None,
                                                 self.config.autodoc_type_aliases)
                    if self.objpath[-1] in annotations:
                        self.object = UNINITIALIZED_ATTR
                        self.parent = parent
                        return True
            except ImportError:
                pass

            if raiseerror:
                raise
            else:
                logger.warning(exc.args[0], type='autodoc', subtype='import_object')
                self.env.note_reread()
                return False

    def should_suppress_value_header(self) -> bool:
        return (self.object is UNINITIALIZED_ATTR or
                super().should_suppress_value_header())

    def get_doc(self, ignore: int = None) -> Optional[List[List[str]]]:
        if self.object is UNINITIALIZED_ATTR:
            return []
        else:
            return super().get_doc(ignore)  # type: ignore
```
### 49 - sphinx/ext/autodoc/__init__.py:

Start line: 2287, End line: 2315

```python
class NonDataDescriptorMixin(DataDocumenterMixinBase):
    """
    Mixin for AttributeDocumenter to provide the feature for supporting non
    data-descriptors.

    .. note:: This mix-in must be inherited after other mix-ins.  Otherwise, docstring
              and :value: header will be suppressed unexpectedly.
    """

    def import_object(self, raiseerror: bool = False) -> bool:
        ret = super().import_object(raiseerror)  # type: ignore
        if ret and not inspect.isattributedescriptor(self.object):
            self.non_data_descriptor = True
        else:
            self.non_data_descriptor = False

        return ret

    def should_suppress_value_header(self) -> bool:
        return (not getattr(self, 'non_data_descriptor', False) or
                super().should_suppress_directive_header())

    def get_doc(self, ignore: int = None) -> Optional[List[List[str]]]:
        if getattr(self, 'non_data_descriptor', False):
            # the docstring of non datadescriptor is very probably the wrong thing
            # to display
            return None
        else:
            return super().get_doc(ignore)  # type: ignore
```
### 50 - sphinx/ext/autodoc/__init__.py:

Start line: 1263, End line: 1278

```python
class DocstringStripSignatureMixin(DocstringSignatureMixin):
    """
    Mixin for AttributeDocumenter to provide the
    feature of stripping any function signature from the docstring.
    """
    def format_signature(self, **kwargs: Any) -> str:
        if self.args is None and self.config.autodoc_docstring_signature:  # type: ignore
            # only act if a signature is not explicitly given already, and if
            # the feature is enabled
            result = self._find_signature()
            if result is not None:
                # Discarding _args is a only difference with
                # DocstringSignatureMixin.format_signature.
                # Documenter.format_signature use self.args value to format.
                _args, self.retann = result
        return super().format_signature(**kwargs)
```
### 53 - sphinx/domains/python.py:

Start line: 11, End line: 80

```python
import builtins
import inspect
import re
import sys
import typing
import warnings
from inspect import Parameter
from typing import Any, Dict, Iterable, Iterator, List, NamedTuple, Optional, Tuple, Type, cast

from docutils import nodes
from docutils.nodes import Element, Node
from docutils.parsers.rst import directives
from docutils.parsers.rst.states import Inliner

from sphinx import addnodes
from sphinx.addnodes import desc_signature, pending_xref, pending_xref_condition
from sphinx.application import Sphinx
from sphinx.builders import Builder
from sphinx.deprecation import RemovedInSphinx50Warning, RemovedInSphinx60Warning
from sphinx.directives import ObjectDescription
from sphinx.domains import Domain, Index, IndexEntry, ObjType
from sphinx.environment import BuildEnvironment
from sphinx.locale import _, __
from sphinx.pycode.ast import ast
from sphinx.pycode.ast import parse as ast_parse
from sphinx.roles import XRefRole
from sphinx.util import logging
from sphinx.util.docfields import Field, GroupedField, TypedField
from sphinx.util.docutils import SphinxDirective
from sphinx.util.inspect import signature_from_str
from sphinx.util.nodes import find_pending_xref_condition, make_id, make_refnode
from sphinx.util.typing import OptionSpec, TextlikeNode

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


class ObjectEntry(NamedTuple):
    docname: str
    node_id: str
    objtype: str
    aliased: bool


class ModuleEntry(NamedTuple):
    docname: str
    node_id: str
    synopsis: str
    platform: str
    deprecated: bool
```
### 54 - sphinx/ext/autodoc/__init__.py:

Start line: 2347, End line: 2370

```python
class SlotsMixin(DataDocumenterMixinBase):

    def get_doc(self, ignore: int = None) -> Optional[List[List[str]]]:
        if self.object is SLOTSATTR:
            try:
                __slots__ = inspect.getslots(self.parent)
                if __slots__ and __slots__.get(self.objpath[-1]):
                    docstring = prepare_docstring(__slots__[self.objpath[-1]])
                    return [docstring]
                else:
                    return []
            except ValueError as exc:
                logger.warning(__('Invalid __slots__ found on %s. Ignored.'),
                               (self.parent.__qualname__, exc), type='autodoc')
                return []
        else:
            return super().get_doc(ignore)  # type: ignore

    @property
    def _datadescriptor(self) -> bool:
        warnings.warn('AttributeDocumenter._datadescriptor() is deprecated.',
                      RemovedInSphinx60Warning)
        if self.object is SLOTSATTR:
            return True
        else:
            return False
```
### 55 - sphinx/ext/autodoc/__init__.py:

Start line: 1843, End line: 1859

```python
class NewTypeMixin(DataDocumenterMixinBase):
    """
    Mixin for DataDocumenter and AttributeDocumenter to provide the feature for
    supporting NewTypes.
    """

    def should_suppress_directive_header(self) -> bool:
        return (inspect.isNewType(self.object) or
                super().should_suppress_directive_header())

    def update_content(self, more_content: StringList) -> None:
        if inspect.isNewType(self.object):
            supertype = restify(self.object.__supertype__)
            more_content.append(_('alias of %s') % supertype, '')
            more_content.append('', '')

        super().update_content(more_content)
```
### 56 - sphinx/ext/autodoc/__init__.py:

Start line: 1185, End line: 1242

```python
class DocstringSignatureMixin:
    """
    Mixin for FunctionDocumenter and MethodDocumenter to provide the
    feature of reading the signature from the docstring.
    """
    _new_docstrings: List[List[str]] = None
    _signatures: List[str] = None

    def _find_signature(self) -> Tuple[str, str]:
        # candidates of the object name
        valid_names = [self.objpath[-1]]  # type: ignore
        if isinstance(self, ClassDocumenter):
            valid_names.append('__init__')
            if hasattr(self.object, '__mro__'):
                valid_names.extend(cls.__name__ for cls in self.object.__mro__)

        docstrings = self.get_doc()
        if docstrings is None:
            return None, None
        self._new_docstrings = docstrings[:]
        self._signatures = []
        result = None
        for i, doclines in enumerate(docstrings):
            for j, line in enumerate(doclines):
                if not line:
                    # no lines in docstring, no match
                    break

                if line.endswith('\\'):
                    line = line.rstrip('\\').rstrip()

                # match first line of docstring against signature RE
                match = py_ext_sig_re.match(line)
                if not match:
                    break
                exmod, path, base, args, retann = match.groups()

                # the base name must match ours
                if base not in valid_names:
                    break

                # re-prepare docstring to ignore more leading indentation
                tab_width = self.directive.state.document.settings.tab_width  # type: ignore
                self._new_docstrings[i] = prepare_docstring('\n'.join(doclines[j + 1:]),
                                                            tabsize=tab_width)

                if result is None:
                    # first signature
                    result = args, retann
                else:
                    # subsequent signatures
                    self._signatures.append("(%s) -> %s" % (args, retann))

            if result:
                # finish the loop when signature found
                break

        return result
```
### 57 - sphinx/ext/autodoc/__init__.py:

Start line: 1152, End line: 1182

```python
class ClassLevelDocumenter(Documenter):
    """
    Specialized Documenter subclass for objects on class level (methods,
    attributes).
    """
    def resolve_name(self, modname: str, parents: Any, path: str, base: Any
                     ) -> Tuple[str, List[str]]:
        if modname is None:
            if path:
                mod_cls = path.rstrip('.')
            else:
                mod_cls = None
                # if documenting a class-level object without path,
                # there must be a current class, either from a parent
                # auto directive ...
                mod_cls = self.env.temp_data.get('autodoc:class')
                # ... or from a class directive
                if mod_cls is None:
                    mod_cls = self.env.ref_context.get('py:class')
                # ... if still None, there's no way to know
                if mod_cls is None:
                    return None, []
            modname, sep, cls = mod_cls.rpartition('.')
            parents = [cls]
            # if the module name is still missing, get it like above
            if not modname:
                modname = self.env.temp_data.get('autodoc:module')
            if not modname:
                modname = self.env.ref_context.get('py:module')
            # ... else, it stays None, which means invalid
        return modname, parents + [base]
```
### 58 - sphinx/ext/autodoc/__init__.py:

Start line: 893, End line: 984

```python
class Documenter:

    def generate(self, more_content: Optional[StringList] = None, real_modname: str = None,
                 check_module: bool = False, all_members: bool = False) -> None:
        """Generate reST for the object given by *self.name*, and possibly for
        its members.

        If *more_content* is given, include that content. If *real_modname* is
        given, use that module name to find attribute docs. If *check_module* is
        True, only generate if the object is defined in the module name it is
        imported from. If *all_members* is True, document all members.
        """
        if not self.parse_name():
            # need a module to import
            logger.warning(
                __('don\'t know which module to import for autodocumenting '
                   '%r (try placing a "module" or "currentmodule" directive '
                   'in the document, or giving an explicit module name)') %
                self.name, type='autodoc')
            return

        # now, import the module and get object to document
        if not self.import_object():
            return

        # If there is no real module defined, figure out which to use.
        # The real module is used in the module analyzer to look up the module
        # where the attribute documentation would actually be found in.
        # This is used for situations where you have a module that collects the
        # functions and classes of internal submodules.
        guess_modname = self.get_real_modname()
        self.real_modname: str = real_modname or guess_modname

        # try to also get a source code analyzer for attribute docs
        try:
            self.analyzer = ModuleAnalyzer.for_module(self.real_modname)
            # parse right now, to get PycodeErrors on parsing (results will
            # be cached anyway)
            self.analyzer.find_attr_docs()
        except PycodeError as exc:
            logger.debug('[autodoc] module analyzer failed: %s', exc)
            # no source file -- e.g. for builtin and C modules
            self.analyzer = None
            # at least add the module.__file__ as a dependency
            if hasattr(self.module, '__file__') and self.module.__file__:
                self.directive.record_dependencies.add(self.module.__file__)
        else:
            self.directive.record_dependencies.add(self.analyzer.srcname)

        if self.real_modname != guess_modname:
            # Add module to dependency list if target object is defined in other module.
            try:
                analyzer = ModuleAnalyzer.for_module(guess_modname)
                self.directive.record_dependencies.add(analyzer.srcname)
            except PycodeError:
                pass

        docstrings: List[str] = sum(self.get_doc() or [], [])
        if ismock(self.object) and not docstrings:
            logger.warning(__('A mocked object is detected: %r'),
                           self.name, type='autodoc')

        # check __module__ of object (for members not given explicitly)
        if check_module:
            if not self.check_module():
                return

        sourcename = self.get_sourcename()

        # make sure that the result starts with an empty line.  This is
        # necessary for some situations where another directive preprocesses
        # reST and no starting newline is present
        self.add_line('', sourcename)

        # format the object's signature, if any
        try:
            sig = self.format_signature()
        except Exception as exc:
            logger.warning(__('error while formatting signature for %s: %s'),
                           self.fullname, exc, type='autodoc')
            return

        # generate the directive header and options, if applicable
        self.add_directive_header(sig)
        self.add_line('', sourcename)

        # e.g. the module directive doesn't have content
        self.indent += self.content_indent

        # add all content (from docstrings, attribute docs etc.)
        self.add_content(more_content)

        # document members, if possible
        self.document_members(all_members)
```
### 59 - sphinx/ext/autodoc/__init__.py:

Start line: 1886, End line: 1901

```python
class TypeVarMixin(DataDocumenterMixinBase):

    def update_content(self, more_content: StringList) -> None:
        if isinstance(self.object, TypeVar):
            attrs = [repr(self.object.__name__)]
            for constraint in self.object.__constraints__:
                attrs.append(stringify_typehint(constraint))
            if self.object.__bound__:
                attrs.append(r"bound=\ " + restify(self.object.__bound__))
            if self.object.__covariant__:
                attrs.append("covariant=True")
            if self.object.__contravariant__:
                attrs.append("contravariant=True")

            more_content.append(_('alias of TypeVar(%s)') % ", ".join(attrs), '')
            more_content.append('', '')

        super().update_content(more_content)
```
### 60 - sphinx/ext/autodoc/__init__.py:

Start line: 1945, End line: 1992

```python
class DataDocumenter(GenericAliasMixin, NewTypeMixin, TypeVarMixin,
                     UninitializedGlobalVariableMixin, ModuleLevelDocumenter):
    """
    Specialized Documenter subclass for data items.
    """
    objtype = 'data'
    member_order = 40
    priority = -10
    option_spec: OptionSpec = dict(ModuleLevelDocumenter.option_spec)
    option_spec["annotation"] = annotation_option
    option_spec["no-value"] = bool_option

    @classmethod
    def can_document_member(cls, member: Any, membername: str, isattr: bool, parent: Any
                            ) -> bool:
        return isinstance(parent, ModuleDocumenter) and isattr

    def update_annotations(self, parent: Any) -> None:
        """Update __annotations__ to support type_comment and so on."""
        annotations = dict(inspect.getannotations(parent))
        parent.__annotations__ = annotations

        try:
            analyzer = ModuleAnalyzer.for_module(self.modname)
            analyzer.analyze()
            for (classname, attrname), annotation in analyzer.annotations.items():
                if classname == '' and attrname not in annotations:
                    annotations[attrname] = annotation
        except PycodeError:
            pass

    def import_object(self, raiseerror: bool = False) -> bool:
        ret = super().import_object(raiseerror)
        if self.parent:
            self.update_annotations(self.parent)

        return ret

    def should_suppress_value_header(self) -> bool:
        if super().should_suppress_value_header():
            return True
        else:
            doc = self.get_doc()
            docstring, metadata = separate_metadata('\n'.join(sum(doc, [])))
            if 'hide-value' in metadata:
                return True

        return False
```
### 66 - sphinx/ext/autodoc/__init__.py:

Start line: 1862, End line: 1884

```python
class TypeVarMixin(DataDocumenterMixinBase):
    """
    Mixin for DataDocumenter and AttributeDocumenter to provide the feature for
    supporting TypeVars.
    """

    def should_suppress_directive_header(self) -> bool:
        return (isinstance(self.object, TypeVar) or
                super().should_suppress_directive_header())

    def get_doc(self, ignore: int = None) -> Optional[List[List[str]]]:
        if ignore is not None:
            warnings.warn("The 'ignore' argument to autodoc.%s.get_doc() is deprecated."
                          % self.__class__.__name__,
                          RemovedInSphinx50Warning, stacklevel=2)

        if isinstance(self.object, TypeVar):
            if self.object.__doc__ != TypeVar.__doc__:
                return super().get_doc()  # type: ignore
            else:
                return []
        else:
            return super().get_doc()  # type: ignore
```
### 69 - sphinx/domains/python.py:

Start line: 991, End line: 1051

```python
class PyModule(SphinxDirective):
    """
    Directive to mark description of a new module.
    """

    has_content = False
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = False
    option_spec: OptionSpec = {
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
        ret: List[Node] = []
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
### 73 - sphinx/domains/python.py:

Start line: 968, End line: 988

```python
class PyDecoratorMixin:
    """
    Mixin for decorator directives.
    """
    def handle_signature(self, sig: str, signode: desc_signature) -> Tuple[str, str]:
        for cls in self.__class__.__mro__:
            if cls.__name__ != 'DirectiveAdapter':
                warnings.warn('PyDecoratorMixin is deprecated. '
                              'Please check the implementation of %s' % cls,
                              RemovedInSphinx50Warning, stacklevel=2)
                break
        else:
            warnings.warn('PyDecoratorMixin is deprecated',
                          RemovedInSphinx50Warning, stacklevel=2)

        ret = super().handle_signature(sig, signode)  # type: ignore
        signode.insert(0, addnodes.desc_addname('@', '@'))
        return ret

    def needs_arglist(self) -> bool:
        return False
```
### 75 - sphinx/ext/autodoc/__init__.py:

Start line: 823, End line: 866

```python
class Documenter:

    def document_members(self, all_members: bool = False) -> None:
        """Generate reST for member documentation.

        If *all_members* is True, document all members, else those given by
        *self.options.members*.
        """
        # set current namespace for finding members
        self.env.temp_data['autodoc:module'] = self.modname
        if self.objpath:
            self.env.temp_data['autodoc:class'] = self.objpath[0]

        want_all = (all_members or
                    self.options.inherited_members or
                    self.options.members is ALL)
        # find out which members are documentable
        members_check_module, members = self.get_object_members(want_all)

        # document non-skipped members
        memberdocumenters: List[Tuple[Documenter, bool]] = []
        for (mname, member, isattr) in self.filter_members(members, want_all):
            classes = [cls for cls in self.documenters.values()
                       if cls.can_document_member(member, mname, isattr, self)]
            if not classes:
                # don't know how to document this member
                continue
            # prefer the documenter with the highest priority
            classes.sort(key=lambda cls: cls.priority)
            # give explicitly separated module name, so that members
            # of inner classes can be documented
            full_mname = self.modname + '::' + '.'.join(self.objpath + [mname])
            documenter = classes[-1](self.directive, full_mname, self.indent)
            memberdocumenters.append((documenter, isattr))

        member_order = self.options.member_order or self.config.autodoc_member_order
        memberdocumenters = self.sort_members(memberdocumenters, member_order)

        for documenter, isattr in memberdocumenters:
            documenter.generate(
                all_members=True, real_modname=self.real_modname,
                check_module=members_check_module and not isattr)

        # reset current objects
        self.env.temp_data['autodoc:module'] = None
        self.env.temp_data['autodoc:class'] = None
```
### 78 - sphinx/ext/autodoc/__init__.py:

Start line: 601, End line: 642

```python
class Documenter:

    def add_content(self, more_content: Optional[StringList], no_docstring: bool = False
                    ) -> None:
        """Add content from docstrings, attribute documentation and user."""
        if no_docstring:
            warnings.warn("The 'no_docstring' argument to %s.add_content() is deprecated."
                          % self.__class__.__name__,
                          RemovedInSphinx50Warning, stacklevel=2)

        # set sourcename and add content from attribute documentation
        sourcename = self.get_sourcename()
        if self.analyzer:
            attr_docs = self.analyzer.find_attr_docs()
            if self.objpath:
                key = ('.'.join(self.objpath[:-1]), self.objpath[-1])
                if key in attr_docs:
                    no_docstring = True
                    # make a copy of docstring for attributes to avoid cache
                    # the change of autodoc-process-docstring event.
                    docstrings = [list(attr_docs[key])]

                    for i, line in enumerate(self.process_doc(docstrings)):
                        self.add_line(line, sourcename, i)

        # add content from docstrings
        if not no_docstring:
            docstrings = self.get_doc()
            if docstrings is None:
                # Do not call autodoc-process-docstring on get_doc() returns None.
                pass
            else:
                if not docstrings:
                    # append at least a dummy docstring, so that the event
                    # autodoc-process-docstring is fired and can add some
                    # content if desired
                    docstrings.append([])
                for i, line in enumerate(self.process_doc(docstrings)):
                    self.add_line(line, sourcename, i)

        # add additional content (e.g. from document), if present
        if more_content:
            for line, src in zip(more_content.data, more_content.items):
                self.add_line(line, src[0], src[1])
```
### 79 - sphinx/ext/autodoc/__init__.py:

Start line: 1787, End line: 1822

```python
class ExceptionDocumenter(ClassDocumenter):
    """
    Specialized ClassDocumenter subclass for exceptions.
    """
    objtype = 'exception'
    member_order = 10

    # needs a higher priority than ClassDocumenter
    priority = 10

    @classmethod
    def can_document_member(cls, member: Any, membername: str, isattr: bool, parent: Any
                            ) -> bool:
        return isinstance(member, type) and issubclass(member, BaseException)


class DataDocumenterMixinBase:
    # define types of instance variables
    config: Config = None
    env: BuildEnvironment = None
    modname: str = None
    parent: Any = None
    object: Any = None
    objpath: List[str] = None

    def should_suppress_directive_header(self) -> bool:
        """Check directive header should be suppressed."""
        return False

    def should_suppress_value_header(self) -> bool:
        """Check :value: header should be suppressed."""
        return False

    def update_content(self, more_content: StringList) -> None:
        """Update docstring for the NewType object."""
        pass
```
### 80 - sphinx/ext/autodoc/__init__.py:

Start line: 2508, End line: 2542

```python
class AttributeDocumenter(GenericAliasMixin, NewTypeMixin, SlotsMixin,  # type: ignore
                          TypeVarMixin, RuntimeInstanceAttributeMixin,
                          UninitializedInstanceAttributeMixin, NonDataDescriptorMixin,
                          DocstringStripSignatureMixin, ClassLevelDocumenter):
    """
    Specialized Documenter subclass for attributes.
    """
    objtype = 'attribute'
    member_order = 60
    option_spec: OptionSpec = dict(ModuleLevelDocumenter.option_spec)
    option_spec["annotation"] = annotation_option
    option_spec["no-value"] = bool_option

    # must be higher than the MethodDocumenter, else it will recognize
    # some non-data descriptors as methods
    priority = 10

    @staticmethod
    def is_function_or_method(obj: Any) -> bool:
        return inspect.isfunction(obj) or inspect.isbuiltin(obj) or inspect.ismethod(obj)

    @classmethod
    def can_document_member(cls, member: Any, membername: str, isattr: bool, parent: Any
                            ) -> bool:
        if isinstance(parent, ModuleDocumenter):
            return False
        elif inspect.isattributedescriptor(member):
            return True
        elif not inspect.isroutine(member) and not isinstance(member, type):
            return True
        else:
            return False

    def document_members(self, all_members: bool = False) -> None:
        pass
```
### 83 - sphinx/ext/autodoc/__init__.py:

Start line: 425, End line: 446

```python
class Documenter:

    def import_object(self, raiseerror: bool = False) -> bool:
        """Import the object given by *self.modname* and *self.objpath* and set
        it as *self.object*.

        Returns True if successful, False if an error occurred.
        """
        with mock(self.config.autodoc_mock_imports):
            try:
                ret = import_object(self.modname, self.objpath, self.objtype,
                                    attrgetter=self.get_attr,
                                    warningiserror=self.config.autodoc_warningiserror)
                self.module, self.parent, self.object_name, self.object = ret
                if ismock(self.object):
                    self.object = undecorate(self.object)
                return True
            except ImportError as exc:
                if raiseerror:
                    raise
                else:
                    logger.warning(exc.args[0], type='autodoc', subtype='import_object')
                    self.env.note_reread()
                    return False
```
### 85 - sphinx/ext/autodoc/__init__.py:

Start line: 1058, End line: 1083

```python
class ModuleDocumenter(Documenter):

    def get_module_members(self) -> Dict[str, ObjectMember]:
        """Get members of target module."""
        if self.analyzer:
            attr_docs = self.analyzer.attr_docs
        else:
            attr_docs = {}

        members: Dict[str, ObjectMember] = {}
        for name in dir(self.object):
            try:
                value = safe_getattr(self.object, name, None)
                if ismock(value):
                    value = undecorate(value)
                docstring = attr_docs.get(('', name), [])
                members[name] = ObjectMember(name, value, docstring="\n".join(docstring))
            except AttributeError:
                continue

        # annotation only member (ex. attr: int)
        for name in inspect.getannotations(self.object):
            if name not in members:
                docstring = attr_docs.get(('', name), [])
                members[name] = ObjectMember(name, INSTANCEATTR,
                                             docstring="\n".join(docstring))

        return members
```
### 88 - sphinx/ext/autodoc/__init__.py:

Start line: 2683, End line: 2757

```python
class PropertyDocumenter(DocstringStripSignatureMixin, ClassLevelDocumenter):  # type: ignore
    """
    Specialized Documenter subclass for properties.
    """
    objtype = 'property'
    member_order = 60

    # before AttributeDocumenter
    priority = AttributeDocumenter.priority + 1

    @classmethod
    def can_document_member(cls, member: Any, membername: str, isattr: bool, parent: Any
                            ) -> bool:
        if isinstance(parent, ClassDocumenter):
            if inspect.isproperty(member):
                return True
            else:
                __dict__ = safe_getattr(parent.object, '__dict__', {})
                obj = __dict__.get(membername)
                return isinstance(obj, classmethod) and inspect.isproperty(obj.__func__)
        else:
            return False

    def import_object(self, raiseerror: bool = False) -> bool:
        """Check the exisitence of uninitialized instance attribute when failed to import
        the attribute."""
        ret = super().import_object(raiseerror)
        if ret and not inspect.isproperty(self.object):
            __dict__ = safe_getattr(self.parent, '__dict__', {})
            obj = __dict__.get(self.objpath[-1])
            if isinstance(obj, classmethod) and inspect.isproperty(obj.__func__):
                self.object = obj.__func__
                self.isclassmethod = True
                return True
            else:
                return False

        self.isclassmethod = False
        return ret

    def document_members(self, all_members: bool = False) -> None:
        pass

    def get_real_modname(self) -> str:
        real_modname = self.get_attr(self.parent or self.object, '__module__', None)
        return real_modname or self.modname

    def add_directive_header(self, sig: str) -> None:
        super().add_directive_header(sig)
        sourcename = self.get_sourcename()
        if inspect.isabstractmethod(self.object):
            self.add_line('   :abstractmethod:', sourcename)
        if self.isclassmethod:
            self.add_line('   :classmethod:', sourcename)

        if safe_getattr(self.object, 'fget', None):  # property
            func = self.object.fget
        elif safe_getattr(self.object, 'func', None):  # cached_property
            func = self.object.func
        else:
            func = None

        if func and self.config.autodoc_typehints != 'none':
            try:
                signature = inspect.signature(func,
                                              type_aliases=self.config.autodoc_type_aliases)
                if signature.return_annotation is not Parameter.empty:
                    objrepr = stringify_typehint(signature.return_annotation)
                    self.add_line('   :type: ' + objrepr, sourcename)
            except TypeError as exc:
                logger.warning(__("Failed to get a function signature for %s: %s"),
                               self.fullname, exc)
                return None
            except ValueError:
                return None
```
