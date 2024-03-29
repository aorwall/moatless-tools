# sphinx-doc__sphinx-8007

| **sphinx-doc/sphinx** | `9175da437ed84e43b00ed4c28ccfe629260d3c13` |
| ---- | ---- |
| **No of patches** | 3 |
| **All found context length** | 16672 |
| **Any found context length** | 2191 |
| **Avg pos** | 97.0 |
| **Min pos** | 3 |
| **Max pos** | 72 |
| **Top file pos** | 1 |
| **Missing snippets** | 15 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sphinx/ext/autodoc/__init__.py b/sphinx/ext/autodoc/__init__.py
--- a/sphinx/ext/autodoc/__init__.py
+++ b/sphinx/ext/autodoc/__init__.py
@@ -1213,7 +1213,8 @@ def format_args(self, **kwargs: Any) -> str:
 
         try:
             self.env.app.emit('autodoc-before-process-signature', self.object, False)
-            sig = inspect.signature(self.object, follow_wrapped=True)
+            sig = inspect.signature(self.object, follow_wrapped=True,
+                                    type_aliases=self.env.config.autodoc_type_aliases)
             args = stringify_signature(sig, **kwargs)
         except TypeError as exc:
             logger.warning(__("Failed to get a function signature for %s: %s"),
@@ -1262,7 +1263,9 @@ def format_signature(self, **kwargs: Any) -> str:
         if overloaded:
             __globals__ = safe_getattr(self.object, '__globals__', {})
             for overload in self.analyzer.overloads.get('.'.join(self.objpath)):
-                overload = evaluate_signature(overload, __globals__)
+                overload = evaluate_signature(overload, __globals__,
+                                              self.env.config.autodoc_type_aliases)
+
                 sig = stringify_signature(overload, **kwargs)
                 sigs.append(sig)
 
@@ -1271,7 +1274,7 @@ def format_signature(self, **kwargs: Any) -> str:
     def annotate_to_first_argument(self, func: Callable, typ: Type) -> None:
         """Annotate type hint to the first argument of function if needed."""
         try:
-            sig = inspect.signature(func)
+            sig = inspect.signature(func, type_aliases=self.env.config.autodoc_type_aliases)
         except TypeError as exc:
             logger.warning(__("Failed to get a function signature for %s: %s"),
                            self.fullname, exc)
@@ -1392,7 +1395,8 @@ def get_user_defined_function_or_method(obj: Any, attr: str) -> Any:
         if call is not None:
             self.env.app.emit('autodoc-before-process-signature', call, True)
             try:
-                sig = inspect.signature(call, bound_method=True)
+                sig = inspect.signature(call, bound_method=True,
+                                        type_aliases=self.env.config.autodoc_type_aliases)
                 return type(self.object), '__call__', sig
             except ValueError:
                 pass
@@ -1407,7 +1411,8 @@ def get_user_defined_function_or_method(obj: Any, attr: str) -> Any:
         if new is not None:
             self.env.app.emit('autodoc-before-process-signature', new, True)
             try:
-                sig = inspect.signature(new, bound_method=True)
+                sig = inspect.signature(new, bound_method=True,
+                                        type_aliases=self.env.config.autodoc_type_aliases)
                 return self.object, '__new__', sig
             except ValueError:
                 pass
@@ -1417,7 +1422,8 @@ def get_user_defined_function_or_method(obj: Any, attr: str) -> Any:
         if init is not None:
             self.env.app.emit('autodoc-before-process-signature', init, True)
             try:
-                sig = inspect.signature(init, bound_method=True)
+                sig = inspect.signature(init, bound_method=True,
+                                        type_aliases=self.env.config.autodoc_type_aliases)
                 return self.object, '__init__', sig
             except ValueError:
                 pass
@@ -1428,7 +1434,8 @@ def get_user_defined_function_or_method(obj: Any, attr: str) -> Any:
         # the signature from, so just pass the object itself to our hook.
         self.env.app.emit('autodoc-before-process-signature', self.object, False)
         try:
-            sig = inspect.signature(self.object, bound_method=False)
+            sig = inspect.signature(self.object, bound_method=False,
+                                    type_aliases=self.env.config.autodoc_type_aliases)
             return None, None, sig
         except ValueError:
             pass
@@ -1475,7 +1482,8 @@ def format_signature(self, **kwargs: Any) -> str:
             method = safe_getattr(self._signature_class, self._signature_method_name, None)
             __globals__ = safe_getattr(method, '__globals__', {})
             for overload in self.analyzer.overloads.get(qualname):
-                overload = evaluate_signature(overload, __globals__)
+                overload = evaluate_signature(overload, __globals__,
+                                              self.env.config.autodoc_type_aliases)
 
                 parameters = list(overload.parameters.values())
                 overload = overload.replace(parameters=parameters[1:],
@@ -1820,11 +1828,13 @@ def format_args(self, **kwargs: Any) -> str:
             else:
                 if inspect.isstaticmethod(self.object, cls=self.parent, name=self.object_name):
                     self.env.app.emit('autodoc-before-process-signature', self.object, False)
-                    sig = inspect.signature(self.object, bound_method=False)
+                    sig = inspect.signature(self.object, bound_method=False,
+                                            type_aliases=self.env.config.autodoc_type_aliases)
                 else:
                     self.env.app.emit('autodoc-before-process-signature', self.object, True)
                     sig = inspect.signature(self.object, bound_method=True,
-                                            follow_wrapped=True)
+                                            follow_wrapped=True,
+                                            type_aliases=self.env.config.autodoc_type_aliases)
                 args = stringify_signature(sig, **kwargs)
         except TypeError as exc:
             logger.warning(__("Failed to get a method signature for %s: %s"),
@@ -1884,7 +1894,9 @@ def format_signature(self, **kwargs: Any) -> str:
         if overloaded:
             __globals__ = safe_getattr(self.object, '__globals__', {})
             for overload in self.analyzer.overloads.get('.'.join(self.objpath)):
-                overload = evaluate_signature(overload, __globals__)
+                overload = evaluate_signature(overload, __globals__,
+                                              self.env.config.autodoc_type_aliases)
+
                 if not inspect.isstaticmethod(self.object, cls=self.parent,
                                               name=self.object_name):
                     parameters = list(overload.parameters.values())
@@ -1897,7 +1909,7 @@ def format_signature(self, **kwargs: Any) -> str:
     def annotate_to_first_argument(self, func: Callable, typ: Type) -> None:
         """Annotate type hint to the first argument of function if needed."""
         try:
-            sig = inspect.signature(func)
+            sig = inspect.signature(func, type_aliases=self.env.config.autodoc_type_aliases)
         except TypeError as exc:
             logger.warning(__("Failed to get a method signature for %s: %s"),
                            self.fullname, exc)
@@ -2237,6 +2249,7 @@ def setup(app: Sphinx) -> Dict[str, Any]:
     app.add_config_value('autodoc_mock_imports', [], True)
     app.add_config_value('autodoc_typehints', "signature", True,
                          ENUM("signature", "description", "none"))
+    app.add_config_value('autodoc_type_aliases', {}, True)
     app.add_config_value('autodoc_warningiserror', True, True)
     app.add_config_value('autodoc_inherit_docstrings', True, True)
     app.add_event('autodoc-before-process-signature')
diff --git a/sphinx/util/inspect.py b/sphinx/util/inspect.py
--- a/sphinx/util/inspect.py
+++ b/sphinx/util/inspect.py
@@ -439,8 +439,8 @@ def _should_unwrap(subject: Callable) -> bool:
     return False
 
 
-def signature(subject: Callable, bound_method: bool = False, follow_wrapped: bool = False
-              ) -> inspect.Signature:
+def signature(subject: Callable, bound_method: bool = False, follow_wrapped: bool = False,
+              type_aliases: Dict = {}) -> inspect.Signature:
     """Return a Signature object for the given *subject*.
 
     :param bound_method: Specify *subject* is a bound method or not
@@ -470,7 +470,7 @@ def signature(subject: Callable, bound_method: bool = False, follow_wrapped: boo
 
     try:
         # Update unresolved annotations using ``get_type_hints()``.
-        annotations = typing.get_type_hints(subject)
+        annotations = typing.get_type_hints(subject, None, type_aliases)
         for i, param in enumerate(parameters):
             if isinstance(param.annotation, str) and param.name in annotations:
                 parameters[i] = param.replace(annotation=annotations[param.name])
diff --git a/sphinx/util/typing.py b/sphinx/util/typing.py
--- a/sphinx/util/typing.py
+++ b/sphinx/util/typing.py
@@ -63,7 +63,11 @@ def is_system_TypeVar(typ: Any) -> bool:
 def stringify(annotation: Any) -> str:
     """Stringify type annotation object."""
     if isinstance(annotation, str):
-        return annotation
+        if annotation.startswith("'") and annotation.endswith("'"):
+            # might be a double Forward-ref'ed type.  Go unquoting.
+            return annotation[1:-2]
+        else:
+            return annotation
     elif isinstance(annotation, TypeVar):  # type: ignore
         return annotation.__name__
     elif not annotation:

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sphinx/ext/autodoc/__init__.py | 1216 | 1216 | 6 | 1 | 3415
| sphinx/ext/autodoc/__init__.py | 1265 | 1265 | 6 | 1 | 3415
| sphinx/ext/autodoc/__init__.py | 1274 | 1274 | 6 | 1 | 3415
| sphinx/ext/autodoc/__init__.py | 1395 | 1395 | 23 | 1 | 24911
| sphinx/ext/autodoc/__init__.py | 1410 | 1410 | 23 | 1 | 24911
| sphinx/ext/autodoc/__init__.py | 1420 | 1420 | 23 | 1 | 24911
| sphinx/ext/autodoc/__init__.py | 1431 | 1431 | 23 | 1 | 24911
| sphinx/ext/autodoc/__init__.py | 1478 | 1478 | 23 | 1 | 24911
| sphinx/ext/autodoc/__init__.py | 1823 | 1827 | 3 | 1 | 2191
| sphinx/ext/autodoc/__init__.py | 1887 | 1887 | 3 | 1 | 2191
| sphinx/ext/autodoc/__init__.py | 1900 | 1900 | 3 | 1 | 2191
| sphinx/ext/autodoc/__init__.py | 2240 | 2240 | 17 | 1 | 16672
| sphinx/util/inspect.py | 442 | 443 | - | 19 | -
| sphinx/util/inspect.py | 473 | 473 | - | 19 | -
| sphinx/util/typing.py | 66 | 66 | 72 | 6 | 102737


## Problem Statement

```
Option for not unfolding aliases
Would it be possible to add an option for autodoc not to unfold user-defined type aliases? 
For example, if I introduce a type synonym Position = int and then define a method with argument pos: Position then I would like to see this typing in the documentation and not pos: int. For me, unfolding the alias is loosing information on how the program is built, something a documentation should not do, unless required by the author.


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 sphinx/ext/autodoc/__init__.py** | 1711 | 1732| 186 | 186 | 19210 | 
| 2 | **1 sphinx/ext/autodoc/__init__.py** | 13 | 116| 775 | 961 | 19210 | 
| **-> 3 <-** | **1 sphinx/ext/autodoc/__init__.py** | 1777 | 1917| 1230 | 2191 | 19210 | 
| 4 | 2 sphinx/ext/autodoc/directive.py | 9 | 49| 298 | 2489 | 20471 | 
| 5 | **2 sphinx/ext/autodoc/__init__.py** | 1764 | 1774| 128 | 2617 | 20471 | 
| **-> 6 <-** | **2 sphinx/ext/autodoc/__init__.py** | 1196 | 1292| 798 | 3415 | 20471 | 
| 7 | **2 sphinx/ext/autodoc/__init__.py** | 119 | 152| 207 | 3622 | 20471 | 
| 8 | 3 sphinx/domains/c.py | 2747 | 3561| 6840 | 10462 | 51330 | 
| 9 | 4 sphinx/ext/autodoc/typehints.py | 83 | 139| 460 | 10922 | 52375 | 
| 10 | **4 sphinx/ext/autodoc/__init__.py** | 1673 | 1708| 279 | 11201 | 52375 | 
| 11 | **4 sphinx/ext/autodoc/__init__.py** | 1735 | 1762| 237 | 11438 | 52375 | 
| **-> 12 <-** | **4 sphinx/ext/autodoc/__init__.py** | 1334 | 1589| 2423 | 13861 | 52375 | 
| 13 | **4 sphinx/ext/autodoc/__init__.py** | 1920 | 2061| 1180 | 15041 | 52375 | 
| 14 | **4 sphinx/ext/autodoc/__init__.py** | 648 | 744| 795 | 15836 | 52375 | 
| 15 | **4 sphinx/ext/autodoc/__init__.py** | 986 | 1007| 192 | 16028 | 52375 | 
| 16 | 5 sphinx/ext/autodoc/importer.py | 240 | 260| 181 | 16209 | 54393 | 
| **-> 17 <-** | **5 sphinx/ext/autodoc/__init__.py** | 2216 | 2253| 463 | 16672 | 54393 | 
| 18 | 5 sphinx/ext/autodoc/directive.py | 78 | 89| 125 | 16797 | 54393 | 
| 19 | **5 sphinx/ext/autodoc/__init__.py** | 261 | 334| 776 | 17573 | 54393 | 
| 20 | **6 sphinx/util/typing.py** | 11 | 60| 376 | 17949 | 56552 | 
| 21 | 7 sphinx/domains/cpp.py | 6484 | 7217| 6279 | 24228 | 119376 | 
| 22 | 8 setup.py | 1 | 75| 443 | 24671 | 121101 | 
| **-> 23 <-** | **8 sphinx/ext/autodoc/__init__.py** | 1295 | 1589| 240 | 24911 | 121101 | 
| 24 | **8 sphinx/ext/autodoc/__init__.py** | 2095 | 2136| 321 | 25232 | 121101 | 
| 25 | 9 sphinx/ext/autodoc/type_comment.py | 11 | 37| 247 | 25479 | 122333 | 
| 26 | 10 sphinx/ext/autosummary/generate.py | 90 | 120| 289 | 25768 | 127616 | 
| 27 | 10 sphinx/ext/autodoc/importer.py | 11 | 36| 172 | 25940 | 127616 | 
| 28 | 10 sphinx/domains/c.py | 11 | 811| 6488 | 32428 | 127616 | 
| 29 | 11 sphinx/util/docfields.py | 149 | 175| 259 | 32687 | 130980 | 
| 30 | **11 sphinx/ext/autodoc/__init__.py** | 2193 | 2213| 226 | 32913 | 130980 | 
| 31 | 12 sphinx/ext/autodoc/mock.py | 11 | 68| 451 | 33364 | 132087 | 
| 32 | 12 sphinx/ext/autodoc/directive.py | 109 | 159| 453 | 33817 | 132087 | 
| 33 | 12 sphinx/ext/autodoc/type_comment.py | 117 | 142| 257 | 34074 | 132087 | 
| 34 | 12 sphinx/ext/autodoc/mock.py | 71 | 93| 200 | 34274 | 132087 | 
| 35 | **12 sphinx/ext/autodoc/__init__.py** | 1009 | 1026| 177 | 34451 | 132087 | 
| 36 | **12 sphinx/ext/autodoc/__init__.py** | 973 | 984| 126 | 34577 | 132087 | 
| 37 | **12 sphinx/ext/autodoc/__init__.py** | 2157 | 2176| 172 | 34749 | 132087 | 
| 38 | 12 sphinx/domains/cpp.py | 2457 | 3302| 6722 | 41471 | 132087 | 
| 39 | 13 sphinx/deprecation.py | 11 | 36| 164 | 41635 | 132799 | 
| 40 | 13 sphinx/ext/autodoc/typehints.py | 11 | 38| 200 | 41835 | 132799 | 
| 41 | 13 sphinx/domains/cpp.py | 1535 | 2395| 6582 | 48417 | 132799 | 
| 42 | 13 sphinx/domains/c.py | 1672 | 2509| 6821 | 55238 | 132799 | 
| 43 | **13 sphinx/util/typing.py** | 153 | 230| 915 | 56153 | 132799 | 
| 44 | **13 sphinx/ext/autodoc/__init__.py** | 1050 | 1080| 287 | 56440 | 132799 | 
| 45 | 13 sphinx/domains/c.py | 814 | 1670| 6742 | 63182 | 132799 | 
| 46 | 13 sphinx/ext/autodoc/typehints.py | 41 | 80| 324 | 63506 | 132799 | 
| 47 | 14 sphinx/transforms/references.py | 11 | 68| 362 | 63868 | 133213 | 
| 48 | 14 sphinx/ext/autosummary/generate.py | 20 | 60| 289 | 64157 | 133213 | 
| 49 | 14 sphinx/domains/cpp.py | 5639 | 6482| 6754 | 70911 | 133213 | 
| 50 | 15 sphinx/directives/other.py | 187 | 208| 141 | 71052 | 136387 | 
| 51 | 15 sphinx/ext/autodoc/directive.py | 52 | 75| 201 | 71253 | 136387 | 
| 52 | 16 sphinx/directives/__init__.py | 76 | 88| 128 | 71381 | 139136 | 
| 53 | 17 sphinx/domains/python.py | 11 | 78| 522 | 71903 | 151037 | 
| 54 | 18 sphinx/transforms/post_transforms/__init__.py | 154 | 177| 279 | 72182 | 153037 | 
| 55 | 18 sphinx/directives/__init__.py | 52 | 74| 206 | 72388 | 153037 | 
| 56 | 18 sphinx/directives/__init__.py | 269 | 329| 527 | 72915 | 153037 | 
| 57 | **19 sphinx/util/inspect.py** | 738 | 811| 562 | 73477 | 159436 | 
| 58 | 19 sphinx/domains/cpp.py | 699 | 1497| 6638 | 80115 | 159436 | 
| 59 | 19 sphinx/directives/other.py | 9 | 40| 238 | 80353 | 159436 | 
| 60 | 19 sphinx/domains/cpp.py | 4861 | 5637| 6672 | 87025 | 159436 | 
| 61 | 20 sphinx/domains/std.py | 256 | 273| 125 | 87150 | 169777 | 
| 62 | **20 sphinx/ext/autodoc/__init__.py** | 1029 | 1047| 179 | 87329 | 169777 | 
| 63 | 20 sphinx/domains/cpp.py | 11 | 697| 6099 | 93428 | 169777 | 
| 64 | **20 sphinx/ext/autodoc/__init__.py** | 746 | 789| 454 | 93882 | 169777 | 
| 65 | 20 sphinx/domains/python.py | 1369 | 1407| 304 | 94186 | 169777 | 
| 66 | 21 sphinx/ext/doctest.py | 161 | 201| 281 | 94467 | 174865 | 
| 67 | 21 sphinx/domains/cpp.py | 4096 | 4859| 6556 | 101023 | 174865 | 
| 68 | 22 sphinx/util/cfamily.py | 11 | 80| 766 | 101789 | 178370 | 
| 69 | **22 sphinx/ext/autodoc/__init__.py** | 155 | 181| 236 | 102025 | 178370 | 
| 70 | 22 sphinx/domains/std.py | 11 | 48| 323 | 102348 | 178370 | 
| 71 | **22 sphinx/ext/autodoc/__init__.py** | 490 | 510| 235 | 102583 | 178370 | 
| **-> 72 <-** | **22 sphinx/util/typing.py** | 63 | 82| 154 | 102737 | 178370 | 
| 73 | 23 sphinx/application.py | 1033 | 1046| 174 | 102911 | 189139 | 
| 74 | 23 sphinx/ext/autodoc/importer.py | 39 | 55| 125 | 103036 | 189139 | 
| 75 | **23 sphinx/util/inspect.py** | 11 | 48| 276 | 103312 | 189139 | 


### Hint

```
How did you define the type? If you're using [type aliases](https://mypy.readthedocs.io/en/latest/kinds_of_types.html#type-aliases), it does not define a new type as mypy docs says:

>A type alias does not create a new type. It’s just a shorthand notation for another type – it’s equivalent to the target type except for generic aliases.

I think it is hard to control from Sphinx side.
> How did you define the type? If you're using type aliases 
> <https://mypy.readthedocs.io/en/latest/kinds_of_types.html#type-aliases>, 
> it does not define a new type as mypy docs says:
>
>     A type alias does not create a new type. It’s just a shorthand
>     notation for another type – it’s equivalent to the target type
>     except for generic aliases.
>
> I think it is hard to control from Sphinx side.
>
Thank you for your quick and kind answer. I can see now why Sphinx 
behaves like that.

Although, it is really disappointing. I believe that you agree with me. 
Giving types to parameters serves documenting the code. Type aliases 
make this part of documentation clearer. Until they get automatically 
processed in order for the documentation to look nice...




It would be very nice if realized. Do you have good idea to do that? As my understanding, new static source code analyzer is needed to do that. Please let me know good libraries for that.
Sorry for responding so late. 

I think that the problem comes from relying on the full import mechanism of Python instead of on just its parsing phase. 
There are libraries, such as ast or typed_ast, that build an abstract syntax tree of a script. This tree can further be processed to typeset the script it nicely. Such approach is used, for example, by the 'build' formatter. 
I did an experiment. Its data and results are quoted below. I defined type aliases in two files, a.py and b.py. The a.py script was then subject to parsing by a ast_test.py script. Apparently, the parsing did not unfold the type aliases, they remained just plain identifiers.

What do you think?
Ryszard

##### a.py: #####

from typing import Tuple
from b import T

U = Tuple[int, int]

def f(a: U) -> T:
    """A doc string."""
    return a


##### b.py: #####

from typing import Tuple

T = Tuple[int, int]


##### ast_test.py: #####

from typed_ast import ast3
import astpretty

def main():
    source = open("a.py", "r") 
    tree = ast3.parse(source.read())
    for i in range(len(tree.body)):
        astpretty.pprint(tree.body[i])

if __name__ == "__main__":
    main()


##### Result of 'python3 ast_test.py': #####

ImportFrom(
    lineno=1,
    col_offset=0,
    module='typing',
    names=[alias(name='Tuple', asname=None)],
    level=0,
)
ImportFrom(
    lineno=2,
    col_offset=0,
    module='b',
    names=[alias(name='T', asname=None)],
    level=0,
)
Assign(
    lineno=4,
    col_offset=0,
    targets=[Name(lineno=4, col_offset=0, id='U', ctx=Store())],
    value=Subscript(
        lineno=4,
        col_offset=4,
        value=Name(lineno=4, col_offset=4, id='Tuple', ctx=Load()),
        slice=Index(
            value=Tuple(
                lineno=4,
                col_offset=10,
                elts=[
                    Name(lineno=4, col_offset=10, id='int', ctx=Load()),
                    Name(lineno=4, col_offset=15, id='int', ctx=Load()),
                ],
                ctx=Load(),
            ),
        ),
        ctx=Load(),
    ),
    type_comment=None,
)
FunctionDef(
    lineno=6,
    col_offset=0,
    name='f',
    args=arguments(
        args=[
            arg(
                lineno=6,
                col_offset=6,
                arg='a',
                annotation=Name(lineno=6, col_offset=9, id='U', ctx=Load()),
                type_comment=None,
            ),
        ],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[],
    ),
    body=[
        Expr(
            lineno=7,
            col_offset=4,
            value=Str(lineno=7, col_offset=4, s='A doc string.', kind=''),
        ),
        Return(
            lineno=8,
            col_offset=4,
            value=Name(lineno=8, col_offset=11, id='a', ctx=Load()),
        ),
    ],
    decorator_list=[],
    returns=Name(lineno=6, col_offset=15, id='T', ctx=Load()),
    type_comment=None,
)


This is doable with zero alterations to the Sphinx codebase. All you need is the `annotations` future-import (introduced in [PEP 563](https://www.python.org/dev/peps/pep-0563/) and available in Python >=3.7) and a monkey-patched `typing.get_type_hints`:

\`\`\`python
import typing
typing.get_type_hints = lambda obj, *unused: obj
\`\`\`

Sphinx uses `typing.get_type_hints` to [resolve](https://github.com/sphinx-doc/sphinx/blob/7d3ad79392c4fc1ee9ad162b7cc107f391549b58/sphinx/util/inspect.py#L357) string annotations, so the future-import alone is not sufficient. Making this behavior configurable will allow to avoid monkey-patching.
When trying:

\`\`\`
from __future__ import annotations
import typing
typing.get_type_hints = lambda obj, *unused: obj
\`\`\`

sphinx_autodocs_typehints fails with 
`AttributeError: '_SpecialForm' object has no attribute 'items'`

That may not be directly related to Sphinx 
```

## Patch

```diff
diff --git a/sphinx/ext/autodoc/__init__.py b/sphinx/ext/autodoc/__init__.py
--- a/sphinx/ext/autodoc/__init__.py
+++ b/sphinx/ext/autodoc/__init__.py
@@ -1213,7 +1213,8 @@ def format_args(self, **kwargs: Any) -> str:
 
         try:
             self.env.app.emit('autodoc-before-process-signature', self.object, False)
-            sig = inspect.signature(self.object, follow_wrapped=True)
+            sig = inspect.signature(self.object, follow_wrapped=True,
+                                    type_aliases=self.env.config.autodoc_type_aliases)
             args = stringify_signature(sig, **kwargs)
         except TypeError as exc:
             logger.warning(__("Failed to get a function signature for %s: %s"),
@@ -1262,7 +1263,9 @@ def format_signature(self, **kwargs: Any) -> str:
         if overloaded:
             __globals__ = safe_getattr(self.object, '__globals__', {})
             for overload in self.analyzer.overloads.get('.'.join(self.objpath)):
-                overload = evaluate_signature(overload, __globals__)
+                overload = evaluate_signature(overload, __globals__,
+                                              self.env.config.autodoc_type_aliases)
+
                 sig = stringify_signature(overload, **kwargs)
                 sigs.append(sig)
 
@@ -1271,7 +1274,7 @@ def format_signature(self, **kwargs: Any) -> str:
     def annotate_to_first_argument(self, func: Callable, typ: Type) -> None:
         """Annotate type hint to the first argument of function if needed."""
         try:
-            sig = inspect.signature(func)
+            sig = inspect.signature(func, type_aliases=self.env.config.autodoc_type_aliases)
         except TypeError as exc:
             logger.warning(__("Failed to get a function signature for %s: %s"),
                            self.fullname, exc)
@@ -1392,7 +1395,8 @@ def get_user_defined_function_or_method(obj: Any, attr: str) -> Any:
         if call is not None:
             self.env.app.emit('autodoc-before-process-signature', call, True)
             try:
-                sig = inspect.signature(call, bound_method=True)
+                sig = inspect.signature(call, bound_method=True,
+                                        type_aliases=self.env.config.autodoc_type_aliases)
                 return type(self.object), '__call__', sig
             except ValueError:
                 pass
@@ -1407,7 +1411,8 @@ def get_user_defined_function_or_method(obj: Any, attr: str) -> Any:
         if new is not None:
             self.env.app.emit('autodoc-before-process-signature', new, True)
             try:
-                sig = inspect.signature(new, bound_method=True)
+                sig = inspect.signature(new, bound_method=True,
+                                        type_aliases=self.env.config.autodoc_type_aliases)
                 return self.object, '__new__', sig
             except ValueError:
                 pass
@@ -1417,7 +1422,8 @@ def get_user_defined_function_or_method(obj: Any, attr: str) -> Any:
         if init is not None:
             self.env.app.emit('autodoc-before-process-signature', init, True)
             try:
-                sig = inspect.signature(init, bound_method=True)
+                sig = inspect.signature(init, bound_method=True,
+                                        type_aliases=self.env.config.autodoc_type_aliases)
                 return self.object, '__init__', sig
             except ValueError:
                 pass
@@ -1428,7 +1434,8 @@ def get_user_defined_function_or_method(obj: Any, attr: str) -> Any:
         # the signature from, so just pass the object itself to our hook.
         self.env.app.emit('autodoc-before-process-signature', self.object, False)
         try:
-            sig = inspect.signature(self.object, bound_method=False)
+            sig = inspect.signature(self.object, bound_method=False,
+                                    type_aliases=self.env.config.autodoc_type_aliases)
             return None, None, sig
         except ValueError:
             pass
@@ -1475,7 +1482,8 @@ def format_signature(self, **kwargs: Any) -> str:
             method = safe_getattr(self._signature_class, self._signature_method_name, None)
             __globals__ = safe_getattr(method, '__globals__', {})
             for overload in self.analyzer.overloads.get(qualname):
-                overload = evaluate_signature(overload, __globals__)
+                overload = evaluate_signature(overload, __globals__,
+                                              self.env.config.autodoc_type_aliases)
 
                 parameters = list(overload.parameters.values())
                 overload = overload.replace(parameters=parameters[1:],
@@ -1820,11 +1828,13 @@ def format_args(self, **kwargs: Any) -> str:
             else:
                 if inspect.isstaticmethod(self.object, cls=self.parent, name=self.object_name):
                     self.env.app.emit('autodoc-before-process-signature', self.object, False)
-                    sig = inspect.signature(self.object, bound_method=False)
+                    sig = inspect.signature(self.object, bound_method=False,
+                                            type_aliases=self.env.config.autodoc_type_aliases)
                 else:
                     self.env.app.emit('autodoc-before-process-signature', self.object, True)
                     sig = inspect.signature(self.object, bound_method=True,
-                                            follow_wrapped=True)
+                                            follow_wrapped=True,
+                                            type_aliases=self.env.config.autodoc_type_aliases)
                 args = stringify_signature(sig, **kwargs)
         except TypeError as exc:
             logger.warning(__("Failed to get a method signature for %s: %s"),
@@ -1884,7 +1894,9 @@ def format_signature(self, **kwargs: Any) -> str:
         if overloaded:
             __globals__ = safe_getattr(self.object, '__globals__', {})
             for overload in self.analyzer.overloads.get('.'.join(self.objpath)):
-                overload = evaluate_signature(overload, __globals__)
+                overload = evaluate_signature(overload, __globals__,
+                                              self.env.config.autodoc_type_aliases)
+
                 if not inspect.isstaticmethod(self.object, cls=self.parent,
                                               name=self.object_name):
                     parameters = list(overload.parameters.values())
@@ -1897,7 +1909,7 @@ def format_signature(self, **kwargs: Any) -> str:
     def annotate_to_first_argument(self, func: Callable, typ: Type) -> None:
         """Annotate type hint to the first argument of function if needed."""
         try:
-            sig = inspect.signature(func)
+            sig = inspect.signature(func, type_aliases=self.env.config.autodoc_type_aliases)
         except TypeError as exc:
             logger.warning(__("Failed to get a method signature for %s: %s"),
                            self.fullname, exc)
@@ -2237,6 +2249,7 @@ def setup(app: Sphinx) -> Dict[str, Any]:
     app.add_config_value('autodoc_mock_imports', [], True)
     app.add_config_value('autodoc_typehints', "signature", True,
                          ENUM("signature", "description", "none"))
+    app.add_config_value('autodoc_type_aliases', {}, True)
     app.add_config_value('autodoc_warningiserror', True, True)
     app.add_config_value('autodoc_inherit_docstrings', True, True)
     app.add_event('autodoc-before-process-signature')
diff --git a/sphinx/util/inspect.py b/sphinx/util/inspect.py
--- a/sphinx/util/inspect.py
+++ b/sphinx/util/inspect.py
@@ -439,8 +439,8 @@ def _should_unwrap(subject: Callable) -> bool:
     return False
 
 
-def signature(subject: Callable, bound_method: bool = False, follow_wrapped: bool = False
-              ) -> inspect.Signature:
+def signature(subject: Callable, bound_method: bool = False, follow_wrapped: bool = False,
+              type_aliases: Dict = {}) -> inspect.Signature:
     """Return a Signature object for the given *subject*.
 
     :param bound_method: Specify *subject* is a bound method or not
@@ -470,7 +470,7 @@ def signature(subject: Callable, bound_method: bool = False, follow_wrapped: boo
 
     try:
         # Update unresolved annotations using ``get_type_hints()``.
-        annotations = typing.get_type_hints(subject)
+        annotations = typing.get_type_hints(subject, None, type_aliases)
         for i, param in enumerate(parameters):
             if isinstance(param.annotation, str) and param.name in annotations:
                 parameters[i] = param.replace(annotation=annotations[param.name])
diff --git a/sphinx/util/typing.py b/sphinx/util/typing.py
--- a/sphinx/util/typing.py
+++ b/sphinx/util/typing.py
@@ -63,7 +63,11 @@ def is_system_TypeVar(typ: Any) -> bool:
 def stringify(annotation: Any) -> str:
     """Stringify type annotation object."""
     if isinstance(annotation, str):
-        return annotation
+        if annotation.startswith("'") and annotation.endswith("'"):
+            # might be a double Forward-ref'ed type.  Go unquoting.
+            return annotation[1:-2]
+        else:
+            return annotation
     elif isinstance(annotation, TypeVar):  # type: ignore
         return annotation.__name__
     elif not annotation:

```

## Test Patch

```diff
diff --git a/tests/roots/test-ext-autodoc/target/annotations.py b/tests/roots/test-ext-autodoc/target/annotations.py
new file mode 100644
--- /dev/null
+++ b/tests/roots/test-ext-autodoc/target/annotations.py
@@ -0,0 +1,25 @@
+from __future__ import annotations
+from typing import overload
+
+
+myint = int
+
+
+def sum(x: myint, y: myint) -> myint:
+    """docstring"""
+    return x + y
+
+
+@overload
+def mult(x: myint, y: myint) -> myint:
+    ...
+
+
+@overload
+def mult(x: float, y: float) -> float:
+    ...
+
+
+def mult(x, y):
+    """docstring"""
+    return x, y
diff --git a/tests/test_ext_autodoc_configs.py b/tests/test_ext_autodoc_configs.py
--- a/tests/test_ext_autodoc_configs.py
+++ b/tests/test_ext_autodoc_configs.py
@@ -642,6 +642,54 @@ def test_autodoc_typehints_description_for_invalid_node(app):
     restructuredtext.parse(app, text)  # raises no error
 
 
+@pytest.mark.skipif(sys.version_info < (3, 7), reason='python 3.7+ is required.')
+@pytest.mark.sphinx('text', testroot='ext-autodoc')
+def test_autodoc_type_aliases(app):
+    # default
+    options = {"members": None}
+    actual = do_autodoc(app, 'module', 'target.annotations', options)
+    assert list(actual) == [
+        '',
+        '.. py:module:: target.annotations',
+        '',
+        '',
+        '.. py:function:: mult(x: int, y: int) -> int',
+        '                 mult(x: float, y: float) -> float',
+        '   :module: target.annotations',
+        '',
+        '   docstring',
+        '',
+        '',
+        '.. py:function:: sum(x: int, y: int) -> int',
+        '   :module: target.annotations',
+        '',
+        '   docstring',
+        '',
+    ]
+
+    # define aliases
+    app.config.autodoc_type_aliases = {'myint': 'myint'}
+    actual = do_autodoc(app, 'module', 'target.annotations', options)
+    assert list(actual) == [
+        '',
+        '.. py:module:: target.annotations',
+        '',
+        '',
+        '.. py:function:: mult(x: myint, y: myint) -> myint',
+        '                 mult(x: float, y: float) -> float',
+        '   :module: target.annotations',
+        '',
+        '   docstring',
+        '',
+        '',
+        '.. py:function:: sum(x: myint, y: myint) -> myint',
+        '   :module: target.annotations',
+        '',
+        '   docstring',
+        '',
+    ]
+
+
 @pytest.mark.sphinx('html', testroot='ext-autodoc')
 def test_autodoc_default_options(app):
     # no settings

```


## Code snippets

### 1 - sphinx/ext/autodoc/__init__.py:

Start line: 1711, End line: 1732

```python
class GenericAliasDocumenter(DataDocumenter):
    """
    Specialized Documenter subclass for GenericAliases.
    """

    objtype = 'genericalias'
    directivetype = 'data'
    priority = DataDocumenter.priority + 1

    @classmethod
    def can_document_member(cls, member: Any, membername: str, isattr: bool, parent: Any
                            ) -> bool:
        return inspect.isgenericalias(member)

    def add_directive_header(self, sig: str) -> None:
        self.options.annotation = SUPPRESS  # type: ignore
        super().add_directive_header(sig)

    def add_content(self, more_content: Any, no_docstring: bool = False) -> None:
        name = stringify_typehint(self.object)
        content = StringList([_('alias of %s') % name], source='')
        super().add_content(content)
```
### 2 - sphinx/ext/autodoc/__init__.py:

Start line: 13, End line: 116

```python
import importlib
import re
import warnings
from inspect import Parameter, Signature
from types import ModuleType
from typing import (
    Any, Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Type, TypeVar, Union
)
from typing import get_type_hints

from docutils.statemachine import StringList

import sphinx
from sphinx.application import Sphinx
from sphinx.config import Config, ENUM
from sphinx.deprecation import RemovedInSphinx40Warning, RemovedInSphinx50Warning
from sphinx.environment import BuildEnvironment
from sphinx.ext.autodoc.importer import import_object, get_module_members, get_object_members
from sphinx.ext.autodoc.mock import mock
from sphinx.locale import _, __
from sphinx.pycode import ModuleAnalyzer, PycodeError
from sphinx.util import inspect
from sphinx.util import logging
from sphinx.util.docstrings import extract_metadata, prepare_docstring
from sphinx.util.inspect import (
    evaluate_signature, getdoc, object_description, safe_getattr, stringify_signature
)
from sphinx.util.typing import stringify as stringify_typehint

if False:
    # For type annotation
    from typing import Type  # NOQA # for python3.5.1
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
    if arg is None or arg is True:
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
    if arg is None:
        return EMPTY
    return {x.strip() for x in arg.split(',') if x.strip()}
```
### 3 - sphinx/ext/autodoc/__init__.py:

Start line: 1777, End line: 1917

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
        return inspect.isroutine(member) and \
            not isinstance(parent, ModuleDocumenter)

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
        if self.env.config.autodoc_typehints in ('none', 'description'):
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
                    sig = inspect.signature(self.object, bound_method=False)
                else:
                    self.env.app.emit('autodoc-before-process-signature', self.object, True)
                    sig = inspect.signature(self.object, bound_method=True,
                                            follow_wrapped=True)
                args = stringify_signature(sig, **kwargs)
        except TypeError as exc:
            logger.warning(__("Failed to get a method signature for %s: %s"),
                           self.fullname, exc)
            return None
        except ValueError:
            args = ''

        if self.env.config.strip_signature_backslash:
            # escape backslashes for reST
            args = args.replace('\\', '\\\\')
        return args

    def add_directive_header(self, sig: str) -> None:
        super().add_directive_header(sig)

        sourcename = self.get_sourcename()
        obj = self.parent.__dict__.get(self.object_name, self.object)
        if inspect.isabstractmethod(obj):
            self.add_line('   :abstractmethod:', sourcename)
        if inspect.iscoroutinefunction(obj):
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
        if self.analyzer and '.'.join(self.objpath) in self.analyzer.overloads:
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
                    self.annotate_to_first_argument(func, typ)

                    documenter = MethodDocumenter(self.directive, '')
                    documenter.parent = self.parent
                    documenter.object = func
                    documenter.objpath = [None]
                    sigs.append(documenter.format_signature())
        if overloaded:
            __globals__ = safe_getattr(self.object, '__globals__', {})
            for overload in self.analyzer.overloads.get('.'.join(self.objpath)):
                overload = evaluate_signature(overload, __globals__)
                if not inspect.isstaticmethod(self.object, cls=self.parent,
                                              name=self.object_name):
                    parameters = list(overload.parameters.values())
                    overload = overload.replace(parameters=parameters[1:])
                sig = stringify_signature(overload, **kwargs)
                sigs.append(sig)

        return "\n".join(sigs)

    def annotate_to_first_argument(self, func: Callable, typ: Type) -> None:
        """Annotate type hint to the first argument of function if needed."""
        try:
            sig = inspect.signature(func)
        except TypeError as exc:
            logger.warning(__("Failed to get a method signature for %s: %s"),
                           self.fullname, exc)
            return
        except ValueError:
            return
        if len(sig.parameters) == 1:
            return

        params = list(sig.parameters.values())
        if params[1].annotation is Parameter.empty:
            params[1] = params[1].replace(annotation=typ)
            try:
                func.__signature__ = sig.replace(parameters=params)  # type: ignore
            except TypeError:
                # failed to update signature (ex. built-in or extension types)
                return
```
### 4 - sphinx/ext/autodoc/directive.py:

Start line: 9, End line: 49

```python
import warnings
from typing import Any, Callable, Dict, List, Set

from docutils import nodes
from docutils.nodes import Element, Node
from docutils.parsers.rst.states import RSTState, Struct
from docutils.statemachine import StringList
from docutils.utils import Reporter, assemble_option_dict

from sphinx.config import Config
from sphinx.deprecation import RemovedInSphinx40Warning
from sphinx.environment import BuildEnvironment
from sphinx.ext.autodoc import Documenter, Options
from sphinx.util import logging
from sphinx.util.docutils import SphinxDirective, switch_source_input
from sphinx.util.nodes import nested_parse_with_titles

if False:
    # For type annotation
    from typing import Type  # for python3.5.1


logger = logging.getLogger(__name__)


# common option names for autodoc directives
AUTODOC_DEFAULT_OPTIONS = ['members', 'undoc-members', 'inherited-members',
                           'show-inheritance', 'private-members', 'special-members',
                           'ignore-module-all', 'exclude-members', 'member-order',
                           'imported-members']


class DummyOptionSpec(dict):
    """An option_spec allows any options."""

    def __bool__(self) -> bool:
        """Behaves like some options are defined."""
        return True

    def __getitem__(self, key: str) -> Callable[[str], str]:
        return lambda x: x
```
### 5 - sphinx/ext/autodoc/__init__.py:

Start line: 1764, End line: 1774

```python
class TypeVarDocumenter(DataDocumenter):

    def add_content(self, more_content: Any, no_docstring: bool = False) -> None:
        attrs = [repr(self.object.__name__)]
        for constraint in self.object.__constraints__:
            attrs.append(stringify_typehint(constraint))
        if self.object.__covariant__:
            attrs.append("covariant=True")
        if self.object.__contravariant__:
            attrs.append("contravariant=True")

        content = StringList([_('alias of TypeVar(%s)') % ", ".join(attrs)], source='')
        super().add_content(content)
```
### 6 - sphinx/ext/autodoc/__init__.py:

Start line: 1196, End line: 1292

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
        if self.env.config.autodoc_typehints in ('none', 'description'):
            kwargs.setdefault('show_annotation', False)

        try:
            self.env.app.emit('autodoc-before-process-signature', self.object, False)
            sig = inspect.signature(self.object, follow_wrapped=True)
            args = stringify_signature(sig, **kwargs)
        except TypeError as exc:
            logger.warning(__("Failed to get a function signature for %s: %s"),
                           self.fullname, exc)
            return None
        except ValueError:
            args = ''

        if self.env.config.strip_signature_backslash:
            # escape backslashes for reST
            args = args.replace('\\', '\\\\')
        return args

    def document_members(self, all_members: bool = False) -> None:
        pass

    def add_directive_header(self, sig: str) -> None:
        sourcename = self.get_sourcename()
        super().add_directive_header(sig)

        if inspect.iscoroutinefunction(self.object):
            self.add_line('   :async:', sourcename)

    def format_signature(self, **kwargs: Any) -> str:
        sigs = []
        if self.analyzer and '.'.join(self.objpath) in self.analyzer.overloads:
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
                    self.annotate_to_first_argument(func, typ)

                    documenter = FunctionDocumenter(self.directive, '')
                    documenter.object = func
                    documenter.objpath = [None]
                    sigs.append(documenter.format_signature())
        if overloaded:
            __globals__ = safe_getattr(self.object, '__globals__', {})
            for overload in self.analyzer.overloads.get('.'.join(self.objpath)):
                overload = evaluate_signature(overload, __globals__)
                sig = stringify_signature(overload, **kwargs)
                sigs.append(sig)

        return "\n".join(sigs)

    def annotate_to_first_argument(self, func: Callable, typ: Type) -> None:
        """Annotate type hint to the first argument of function if needed."""
        try:
            sig = inspect.signature(func)
        except TypeError as exc:
            logger.warning(__("Failed to get a function signature for %s: %s"),
                           self.fullname, exc)
            return
        except ValueError:
            return

        if len(sig.parameters) == 0:
            return

        params = list(sig.parameters.values())
        if params[0].annotation is Parameter.empty:
            params[0] = params[0].replace(annotation=typ)
            try:
                func.__signature__ = sig.replace(parameters=params)  # type: ignore
            except TypeError:
                # failed to update signature (ex. built-in or extension types)
                return
```
### 7 - sphinx/ext/autodoc/__init__.py:

Start line: 119, End line: 152

```python
def inherited_members_option(arg: Any) -> Union[object, Set[str]]:
    """Used to convert the :members: option to auto directives."""
    if arg is None:
        return 'object'
    else:
        return arg


def member_order_option(arg: Any) -> Optional[str]:
    """Used to convert the :members: option to auto directives."""
    if arg is None:
        return None
    elif arg in ('alphabetical', 'bysource', 'groupwise'):
        return arg
    else:
        raise ValueError(__('invalid value for member-order option: %s') % arg)


SUPPRESS = object()


def annotation_option(arg: Any) -> Any:
    if arg is None:
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
### 8 - sphinx/domains/c.py:

Start line: 2747, End line: 3561

```python
class DefinitionParser(BaseParser):
    # those without signedness and size modifiers
    _simple_fundamental_types =
    # ... other code

    def _parse_declarator(self, named: Union[bool, str], paramMode: str,
                          typed: bool = True) -> ASTDeclarator:
        # 'typed' here means 'parse return type stuff'
        if paramMode not in ('type', 'function'):
            raise Exception(
                "Internal error, unknown paramMode '%s'." % paramMode)
        prevErrors = []
        self.skip_ws()
        if typed and self.skip_string('*'):
            self.skip_ws()
            restrict = False
            volatile = False
            const = False
            attrs = []
            while 1:
                if not restrict:
                    restrict = self.skip_word_and_ws('restrict')
                    if restrict:
                        continue
                if not volatile:
                    volatile = self.skip_word_and_ws('volatile')
                    if volatile:
                        continue
                if not const:
                    const = self.skip_word_and_ws('const')
                    if const:
                        continue
                attr = self._parse_attribute()
                if attr is not None:
                    attrs.append(attr)
                    continue
                break
            next = self._parse_declarator(named, paramMode, typed)
            return ASTDeclaratorPtr(next=next,
                                    restrict=restrict, volatile=volatile, const=const,
                                    attrs=attrs)
        if typed and self.current_char == '(':  # note: peeking, not skipping
            # maybe this is the beginning of params, try that first,
            # otherwise assume it's noptr->declarator > ( ptr-declarator )
            pos = self.pos
            try:
                # assume this is params
                res = self._parse_declarator_name_suffix(named, paramMode,
                                                         typed)
                return res
            except DefinitionError as exParamQual:
                msg = "If declarator-id with parameters"
                if paramMode == 'function':
                    msg += " (e.g., 'void f(int arg)')"
                prevErrors.append((exParamQual, msg))
                self.pos = pos
                try:
                    assert self.current_char == '('
                    self.skip_string('(')
                    # TODO: hmm, if there is a name, it must be in inner, right?
                    # TODO: hmm, if there must be parameters, they must b
                    # inside, right?
                    inner = self._parse_declarator(named, paramMode, typed)
                    if not self.skip_string(')'):
                        self.fail("Expected ')' in \"( ptr-declarator )\"")
                    next = self._parse_declarator(named=False,
                                                  paramMode="type",
                                                  typed=typed)
                    return ASTDeclaratorParen(inner=inner, next=next)
                except DefinitionError as exNoPtrParen:
                    self.pos = pos
                    msg = "If parenthesis in noptr-declarator"
                    if paramMode == 'function':
                        msg += " (e.g., 'void (*f(int arg))(double)')"
                    prevErrors.append((exNoPtrParen, msg))
                    header = "Error in declarator"
                    raise self._make_multi_error(prevErrors, header) from exNoPtrParen
        pos = self.pos
        try:
            return self._parse_declarator_name_suffix(named, paramMode, typed)
        except DefinitionError as e:
            self.pos = pos
            prevErrors.append((e, "If declarator-id"))
            header = "Error in declarator or parameters"
            raise self._make_multi_error(prevErrors, header) from e

    def _parse_initializer(self, outer: str = None, allowFallback: bool = True
                           ) -> ASTInitializer:
        self.skip_ws()
        if outer == 'member' and False:  # TODO
            bracedInit = self._parse_braced_init_list()
            if bracedInit is not None:
                return ASTInitializer(bracedInit, hasAssign=False)

        if not self.skip_string('='):
            return None

        bracedInit = self._parse_braced_init_list()
        if bracedInit is not None:
            return ASTInitializer(bracedInit)

        if outer == 'member':
            fallbackEnd = []  # type: List[str]
        elif outer is None:  # function parameter
            fallbackEnd = [',', ')']
        else:
            self.fail("Internal error, initializer for outer '%s' not "
                      "implemented." % outer)

        def parser():
            return self._parse_assignment_expression()

        value = self._parse_expression_fallback(fallbackEnd, parser, allow=allowFallback)
        return ASTInitializer(value)

    def _parse_type(self, named: Union[bool, str], outer: str = None) -> ASTType:
        """
        named=False|'maybe'|True: 'maybe' is e.g., for function objects which
        doesn't need to name the arguments
        """
        if outer:  # always named
            if outer not in ('type', 'member', 'function'):
                raise Exception('Internal error, unknown outer "%s".' % outer)
            assert named

        if outer == 'type':
            # We allow type objects to just be a name.
            prevErrors = []
            startPos = self.pos
            # first try without the type
            try:
                declSpecs = self._parse_decl_specs(outer=outer, typed=False)
                decl = self._parse_declarator(named=True, paramMode=outer,
                                              typed=False)
                self.assert_end(allowSemicolon=True)
            except DefinitionError as exUntyped:
                desc = "If just a name"
                prevErrors.append((exUntyped, desc))
                self.pos = startPos
                try:
                    declSpecs = self._parse_decl_specs(outer=outer)
                    decl = self._parse_declarator(named=True, paramMode=outer)
                except DefinitionError as exTyped:
                    self.pos = startPos
                    desc = "If typedef-like declaration"
                    prevErrors.append((exTyped, desc))
                    # Retain the else branch for easier debugging.
                    # TODO: it would be nice to save the previous stacktrace
                    #       and output it here.
                    if True:
                        header = "Type must be either just a name or a "
                        header += "typedef-like declaration."
                        raise self._make_multi_error(prevErrors, header) from exTyped
                    else:
                        # For testing purposes.
                        # do it again to get the proper traceback (how do you
                        # reliably save a traceback when an exception is
                        # constructed?)
                        self.pos = startPos
                        typed = True
                        declSpecs = self._parse_decl_specs(outer=outer, typed=typed)
                        decl = self._parse_declarator(named=True, paramMode=outer,
                                                      typed=typed)
        elif outer == 'function':
            declSpecs = self._parse_decl_specs(outer=outer)
            decl = self._parse_declarator(named=True, paramMode=outer)
        else:
            paramMode = 'type'
            if outer == 'member':  # i.e., member
                named = True
            declSpecs = self._parse_decl_specs(outer=outer)
            decl = self._parse_declarator(named=named, paramMode=paramMode)
        return ASTType(declSpecs, decl)

    def _parse_type_with_init(self, named: Union[bool, str], outer: str) -> ASTTypeWithInit:
        if outer:
            assert outer in ('type', 'member', 'function')
        type = self._parse_type(outer=outer, named=named)
        init = self._parse_initializer(outer=outer)
        return ASTTypeWithInit(type, init)

    def _parse_macro(self) -> ASTMacro:
        self.skip_ws()
        ident = self._parse_nested_name()
        if ident is None:
            self.fail("Expected identifier in macro definition.")
        self.skip_ws()
        if not self.skip_string_and_ws('('):
            return ASTMacro(ident, None)
        if self.skip_string(')'):
            return ASTMacro(ident, [])
        args = []
        while 1:
            self.skip_ws()
            if self.skip_string('...'):
                args.append(ASTMacroParameter(None, True))
                self.skip_ws()
                if not self.skip_string(')'):
                    self.fail('Expected ")" after "..." in macro parameters.')
                break
            if not self.match(identifier_re):
                self.fail("Expected identifier in macro parameters.")
            nn = ASTNestedName([ASTIdentifier(self.matched_text)], rooted=False)
            # Allow named variadic args:
            # https://gcc.gnu.org/onlinedocs/cpp/Variadic-Macros.html
            self.skip_ws()
            if self.skip_string_and_ws('...'):
                args.append(ASTMacroParameter(nn, False, True))
                self.skip_ws()
                if not self.skip_string(')'):
                    self.fail('Expected ")" after "..." in macro parameters.')
                break
            args.append(ASTMacroParameter(nn))
            if self.skip_string_and_ws(','):
                continue
            elif self.skip_string_and_ws(')'):
                break
            else:
                self.fail("Expected identifier, ')', or ',' in macro parameter list.")
        return ASTMacro(ident, args)

    def _parse_struct(self) -> ASTStruct:
        name = self._parse_nested_name()
        return ASTStruct(name)

    def _parse_union(self) -> ASTUnion:
        name = self._parse_nested_name()
        return ASTUnion(name)

    def _parse_enum(self) -> ASTEnum:
        name = self._parse_nested_name()
        return ASTEnum(name)

    def _parse_enumerator(self) -> ASTEnumerator:
        name = self._parse_nested_name()
        self.skip_ws()
        init = None
        if self.skip_string('='):
            self.skip_ws()

            def parser() -> ASTExpression:
                return self._parse_constant_expression()

            initVal = self._parse_expression_fallback([], parser)
            init = ASTInitializer(initVal)
        return ASTEnumerator(name, init)

    def parse_pre_v3_type_definition(self) -> ASTDeclaration:
        self.skip_ws()
        declaration = None  # type: Any
        if self.skip_word('struct'):
            typ = 'struct'
            declaration = self._parse_struct()
        elif self.skip_word('union'):
            typ = 'union'
            declaration = self._parse_union()
        elif self.skip_word('enum'):
            typ = 'enum'
            declaration = self._parse_enum()
        else:
            self.fail("Could not parse pre-v3 type directive."
                      " Must start with 'struct', 'union', or 'enum'.")
        return ASTDeclaration(typ, typ, declaration, False)

    def parse_declaration(self, objectType: str, directiveType: str) -> ASTDeclaration:
        if objectType not in ('function', 'member',
                              'macro', 'struct', 'union', 'enum', 'enumerator', 'type'):
            raise Exception('Internal error, unknown objectType "%s".' % objectType)
        if directiveType not in ('function', 'member', 'var',
                                 'macro', 'struct', 'union', 'enum', 'enumerator', 'type'):
            raise Exception('Internal error, unknown directiveType "%s".' % directiveType)

        declaration = None  # type: Any
        if objectType == 'member':
            declaration = self._parse_type_with_init(named=True, outer='member')
        elif objectType == 'function':
            declaration = self._parse_type(named=True, outer='function')
        elif objectType == 'macro':
            declaration = self._parse_macro()
        elif objectType == 'struct':
            declaration = self._parse_struct()
        elif objectType == 'union':
            declaration = self._parse_union()
        elif objectType == 'enum':
            declaration = self._parse_enum()
        elif objectType == 'enumerator':
            declaration = self._parse_enumerator()
        elif objectType == 'type':
            declaration = self._parse_type(named=True, outer='type')
        else:
            assert False
        if objectType != 'macro':
            self.skip_ws()
            semicolon = self.skip_string(';')
        else:
            semicolon = False
        return ASTDeclaration(objectType, directiveType, declaration, semicolon)

    def parse_namespace_object(self) -> ASTNestedName:
        return self._parse_nested_name()

    def parse_xref_object(self) -> ASTNestedName:
        name = self._parse_nested_name()
        # if there are '()' left, just skip them
        self.skip_ws()
        self.skip_string('()')
        self.assert_end()
        return name

    def parse_expression(self) -> Union[ASTExpression, ASTType]:
        pos = self.pos
        res = None  # type: Union[ASTExpression, ASTType]
        try:
            res = self._parse_expression()
            self.skip_ws()
            self.assert_end()
        except DefinitionError as exExpr:
            self.pos = pos
            try:
                res = self._parse_type(False)
                self.skip_ws()
                self.assert_end()
            except DefinitionError as exType:
                header = "Error when parsing (type) expression."
                errs = []
                errs.append((exExpr, "If expression"))
                errs.append((exType, "If type"))
                raise self._make_multi_error(errs, header) from exType
        return res


def _make_phony_error_name() -> ASTNestedName:
    return ASTNestedName([ASTIdentifier("PhonyNameDueToError")], rooted=False)


class CObject(ObjectDescription):
    """
    Description of a C language object.
    """

    doc_field_types = [
        TypedField('parameter', label=_('Parameters'),
                   names=('param', 'parameter', 'arg', 'argument'),
                   typerolename='type', typenames=('type',)),
        Field('returnvalue', label=_('Returns'), has_arg=False,
              names=('returns', 'return')),
        Field('returntype', label=_('Return type'), has_arg=False,
              names=('rtype',)),
    ]

    option_spec = {
        'noindexentry': directives.flag,
    }

    def _add_enumerator_to_parent(self, ast: ASTDeclaration) -> None:
        assert ast.objectType == 'enumerator'
        # find the parent, if it exists && is an enum
        #                  then add the name to the parent scope
        symbol = ast.symbol
        assert symbol
        assert symbol.ident is not None
        parentSymbol = symbol.parent
        assert parentSymbol
        if parentSymbol.parent is None:
            # TODO: we could warn, but it is somewhat equivalent to
            # enumeratorss, without the enum
            return  # no parent
        parentDecl = parentSymbol.declaration
        if parentDecl is None:
            # the parent is not explicitly declared
            # TODO: we could warn, but?
            return
        if parentDecl.objectType != 'enum':
            # TODO: maybe issue a warning, enumerators in non-enums is weird,
            # but it is somewhat equivalent to enumeratorss, without the enum
            return
        if parentDecl.directiveType != 'enum':
            return

        targetSymbol = parentSymbol.parent
        s = targetSymbol.find_identifier(symbol.ident, matchSelf=False, recurseInAnon=True,
                                         searchInSiblings=False)
        if s is not None:
            # something is already declared with that name
            return
        declClone = symbol.declaration.clone()
        declClone.enumeratorScopedSymbol = symbol
        Symbol(parent=targetSymbol, ident=symbol.ident,
               declaration=declClone,
               docname=self.env.docname)

    def add_target_and_index(self, ast: ASTDeclaration, sig: str,
                             signode: TextElement) -> None:
        ids = []
        for i in range(1, _max_id + 1):
            try:
                id = ast.get_id(version=i)
                ids.append(id)
            except NoOldIdError:
                assert i < _max_id
        # let's keep the newest first
        ids = list(reversed(ids))
        newestId = ids[0]
        assert newestId  # shouldn't be None

        name = ast.symbol.get_full_nested_name().get_display_string().lstrip('.')
        if newestId not in self.state.document.ids:
            # always add the newest id
            assert newestId
            signode['ids'].append(newestId)
            # only add compatibility ids when there are no conflicts
            for id in ids[1:]:
                if not id:  # is None when the element didn't exist in that version
                    continue
                if id not in self.state.document.ids:
                    signode['ids'].append(id)

            self.state.document.note_explicit_target(signode)

            domain = cast(CDomain, self.env.get_domain('c'))
            if name not in domain.objects:
                domain.objects[name] = (domain.env.docname, newestId, self.objtype)

        if 'noindexentry' not in self.options:
            indexText = self.get_index_text(name)
            self.indexnode['entries'].append(('single', indexText, newestId, '', None))

    @property
    def object_type(self) -> str:
        raise NotImplementedError()

    @property
    def display_object_type(self) -> str:
        return self.object_type

    def get_index_text(self, name: str) -> str:
        return _('%s (C %s)') % (name, self.display_object_type)

    def parse_definition(self, parser: DefinitionParser) -> ASTDeclaration:
        return parser.parse_declaration(self.object_type, self.objtype)

    def parse_pre_v3_type_definition(self, parser: DefinitionParser) -> ASTDeclaration:
        return parser.parse_pre_v3_type_definition()

    def describe_signature(self, signode: TextElement, ast: Any, options: Dict) -> None:
        ast.describe_signature(signode, 'lastIsName', self.env, options)

    def run(self) -> List[Node]:
        env = self.state.document.settings.env  # from ObjectDescription.run
        if 'c:parent_symbol' not in env.temp_data:
            root = env.domaindata['c']['root_symbol']
            env.temp_data['c:parent_symbol'] = root
            env.ref_context['c:parent_key'] = root.get_lookup_key()

        # When multiple declarations are made in the same directive
        # they need to know about each other to provide symbol lookup for function parameters.
        # We use last_symbol to store the latest added declaration in a directive.
        env.temp_data['c:last_symbol'] = None
        return super().run()

    def handle_signature(self, sig: str, signode: TextElement) -> ASTDeclaration:
        parentSymbol = self.env.temp_data['c:parent_symbol']  # type: Symbol

        parser = DefinitionParser(sig, location=signode, config=self.env.config)
        try:
            try:
                ast = self.parse_definition(parser)
                parser.assert_end()
            except DefinitionError as eOrig:
                if not self.env.config['c_allow_pre_v3']:
                    raise
                if self.objtype != 'type':
                    raise
                try:
                    ast = self.parse_pre_v3_type_definition(parser)
                    parser.assert_end()
                except DefinitionError:
                    raise eOrig
                self.object_type = ast.objectType  # type: ignore
                if self.env.config['c_warn_on_allowed_pre_v3']:
                    msg = "{}: Pre-v3 C type directive '.. c:type:: {}' converted to " \
                          "'.. c:{}:: {}'." \
                          "\nThe original parsing error was:\n{}"
                    msg = msg.format(RemovedInSphinx50Warning.__name__,
                                     sig, ast.objectType, ast, eOrig)
                    logger.warning(msg, location=signode)
        except DefinitionError as e:
            logger.warning(e, location=signode)
            # It is easier to assume some phony name than handling the error in
            # the possibly inner declarations.
            name = _make_phony_error_name()
            symbol = parentSymbol.add_name(name)
            self.env.temp_data['c:last_symbol'] = symbol
            raise ValueError from e

        try:
            symbol = parentSymbol.add_declaration(ast, docname=self.env.docname)
            # append the new declaration to the sibling list
            assert symbol.siblingAbove is None
            assert symbol.siblingBelow is None
            symbol.siblingAbove = self.env.temp_data['c:last_symbol']
            if symbol.siblingAbove is not None:
                assert symbol.siblingAbove.siblingBelow is None
                symbol.siblingAbove.siblingBelow = symbol
            self.env.temp_data['c:last_symbol'] = symbol
        except _DuplicateSymbolError as e:
            # Assume we are actually in the old symbol,
            # instead of the newly created duplicate.
            self.env.temp_data['c:last_symbol'] = e.symbol
            msg = __("Duplicate C declaration, also defined in '%s'.\n"
                     "Declaration is '%s'.")
            msg = msg % (e.symbol.docname, sig)
            logger.warning(msg, location=signode)

        if ast.objectType == 'enumerator':
            self._add_enumerator_to_parent(ast)

        # note: handle_signature may be called multiple time per directive,
        # if it has multiple signatures, so don't mess with the original options.
        options = dict(self.options)
        self.describe_signature(signode, ast, options)
        return ast

    def before_content(self) -> None:
        lastSymbol = self.env.temp_data['c:last_symbol']  # type: Symbol
        assert lastSymbol
        self.oldParentSymbol = self.env.temp_data['c:parent_symbol']
        self.oldParentKey = self.env.ref_context['c:parent_key']  # type: LookupKey
        self.env.temp_data['c:parent_symbol'] = lastSymbol
        self.env.ref_context['c:parent_key'] = lastSymbol.get_lookup_key()

    def after_content(self) -> None:
        self.env.temp_data['c:parent_symbol'] = self.oldParentSymbol
        self.env.ref_context['c:parent_key'] = self.oldParentKey

    def make_old_id(self, name: str) -> str:
        """Generate old styled node_id for C objects.

        .. note:: Old Styled node_id was used until Sphinx-3.0.
                  This will be removed in Sphinx-5.0.
        """
        return 'c.' + name


class CMemberObject(CObject):
    object_type = 'member'

    @property
    def display_object_type(self) -> str:
        # the distinction between var and member is only cosmetic
        assert self.objtype in ('member', 'var')
        return self.objtype


class CFunctionObject(CObject):
    object_type = 'function'


class CMacroObject(CObject):
    object_type = 'macro'


class CStructObject(CObject):
    object_type = 'struct'


class CUnionObject(CObject):
    object_type = 'union'


class CEnumObject(CObject):
    object_type = 'enum'


class CEnumeratorObject(CObject):
    object_type = 'enumerator'


class CTypeObject(CObject):
    object_type = 'type'


class CNamespaceObject(SphinxDirective):
    """
    This directive is just to tell Sphinx that we're documenting stuff in
    namespace foo.
    """

    has_content = False
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = True
    option_spec = {}  # type: Dict

    def run(self) -> List[Node]:
        rootSymbol = self.env.domaindata['c']['root_symbol']
        if self.arguments[0].strip() in ('NULL', '0', 'nullptr'):
            symbol = rootSymbol
            stack = []  # type: List[Symbol]
        else:
            parser = DefinitionParser(self.arguments[0],
                                      location=self.get_source_info(),
                                      config=self.env.config)
            try:
                name = parser.parse_namespace_object()
                parser.assert_end()
            except DefinitionError as e:
                logger.warning(e, location=self.get_source_info())
                name = _make_phony_error_name()
            symbol = rootSymbol.add_name(name)
            stack = [symbol]
        self.env.temp_data['c:parent_symbol'] = symbol
        self.env.temp_data['c:namespace_stack'] = stack
        self.env.ref_context['c:parent_key'] = symbol.get_lookup_key()
        return []


class CNamespacePushObject(SphinxDirective):
    has_content = False
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = True
    option_spec = {}  # type: Dict

    def run(self) -> List[Node]:
        if self.arguments[0].strip() in ('NULL', '0', 'nullptr'):
            return []
        parser = DefinitionParser(self.arguments[0],
                                  location=self.get_source_info(),
                                  config=self.env.config)
        try:
            name = parser.parse_namespace_object()
            parser.assert_end()
        except DefinitionError as e:
            logger.warning(e, location=self.get_source_info())
            name = _make_phony_error_name()
        oldParent = self.env.temp_data.get('c:parent_symbol', None)
        if not oldParent:
            oldParent = self.env.domaindata['c']['root_symbol']
        symbol = oldParent.add_name(name)
        stack = self.env.temp_data.get('c:namespace_stack', [])
        stack.append(symbol)
        self.env.temp_data['c:parent_symbol'] = symbol
        self.env.temp_data['c:namespace_stack'] = stack
        self.env.ref_context['c:parent_key'] = symbol.get_lookup_key()
        return []


class CNamespacePopObject(SphinxDirective):
    has_content = False
    required_arguments = 0
    optional_arguments = 0
    final_argument_whitespace = True
    option_spec = {}  # type: Dict

    def run(self) -> List[Node]:
        stack = self.env.temp_data.get('c:namespace_stack', None)
        if not stack or len(stack) == 0:
            logger.warning("C namespace pop on empty stack. Defaulting to gobal scope.",
                           location=self.get_source_info())
            stack = []
        else:
            stack.pop()
        if len(stack) > 0:
            symbol = stack[-1]
        else:
            symbol = self.env.domaindata['c']['root_symbol']
        self.env.temp_data['c:parent_symbol'] = symbol
        self.env.temp_data['c:namespace_stack'] = stack
        self.env.ref_context['cp:parent_key'] = symbol.get_lookup_key()
        return []


class AliasNode(nodes.Element):
    def __init__(self, sig: str, maxdepth: int, document: Any, env: "BuildEnvironment" = None,
                 parentKey: LookupKey = None) -> None:
        super().__init__()
        self.sig = sig
        self.maxdepth = maxdepth
        assert maxdepth >= 0
        self.document = document
        if env is not None:
            if 'c:parent_symbol' not in env.temp_data:
                root = env.domaindata['c']['root_symbol']
                env.temp_data['c:parent_symbol'] = root
            self.parentKey = env.temp_data['c:parent_symbol'].get_lookup_key()
        else:
            assert parentKey is not None
            self.parentKey = parentKey

    def copy(self: T) -> T:
        return self.__class__(self.sig, env=None, parentKey=self.parentKey)  # type: ignore


class AliasTransform(SphinxTransform):
    default_priority = ReferencesResolver.default_priority - 1

    def _render_symbol(self, s: Symbol, maxdepth: int, document: Any) -> List[Node]:
        nodes = []  # type: List[Node]
        options = dict()  # type: ignore
        signode = addnodes.desc_signature('', '')
        nodes.append(signode)
        s.declaration.describe_signature(signode, 'markName', self.env, options)
        if maxdepth == 0:
            recurse = True
        elif maxdepth == 1:
            recurse = False
        else:
            maxdepth -= 1
            recurse = True
        if recurse:
            content = addnodes.desc_content()
            desc = addnodes.desc()
            content.append(desc)
            desc.document = document
            desc['domain'] = 'c'
            # 'desctype' is a backwards compatible attribute
            desc['objtype'] = desc['desctype'] = 'alias'
            desc['noindex'] = True

            for sChild in s.children:
                childNodes = self._render_symbol(sChild, maxdepth, document)
                desc.extend(childNodes)

            if len(desc.children) != 0:
                nodes.append(content)
        return nodes

    def apply(self, **kwargs: Any) -> None:
        for node in self.document.traverse(AliasNode):
            sig = node.sig
            parentKey = node.parentKey
            try:
                parser = DefinitionParser(sig, location=node,
                                          config=self.env.config)
                name = parser.parse_xref_object()
            except DefinitionError as e:
                logger.warning(e, location=node)
                name = None

            if name is None:
                # could not be parsed, so stop here
                signode = addnodes.desc_signature(sig, '')
                signode.clear()
                signode += addnodes.desc_name(sig, sig)
                node.replace_self(signode)
                continue

            rootSymbol = self.env.domains['c'].data['root_symbol']  # type: Symbol
            parentSymbol = rootSymbol.direct_lookup(parentKey)  # type: Symbol
            if not parentSymbol:
                print("Target: ", sig)
                print("ParentKey: ", parentKey)
                print(rootSymbol.dump(1))
            assert parentSymbol  # should be there

            s = parentSymbol.find_declaration(
                name, 'any',
                matchSelf=True, recurseInAnon=True)
            if s is None:
                signode = addnodes.desc_signature(sig, '')
                node.append(signode)
                signode.clear()
                signode += addnodes.desc_name(sig, sig)

                logger.warning("Could not find C declaration for alias '%s'." % name,
                               location=node)
                node.replace_self(signode)
                continue

            nodes = self._render_symbol(s, maxdepth=node.maxdepth, document=node.document)
            node.replace_self(nodes)


class CAliasObject(ObjectDescription):
    option_spec = {
        'maxdepth': directives.nonnegative_int
    }  # type: Dict

    def run(self) -> List[Node]:
        if ':' in self.name:
            self.domain, self.objtype = self.name.split(':', 1)
        else:
            self.domain, self.objtype = '', self.name

        node = addnodes.desc()
        node.document = self.state.document
        node['domain'] = self.domain
        # 'desctype' is a backwards compatible attribute
        node['objtype'] = node['desctype'] = self.objtype
        node['noindex'] = True

        self.names = []  # type: List[str]
        maxdepth = self.options.get('maxdepth', 1)
        signatures = self.get_signatures()
        for i, sig in enumerate(signatures):
            node.append(AliasNode(sig, maxdepth, self.state.document, env=self.env))
        return [node]


class CXRefRole(XRefRole):
    def process_link(self, env: BuildEnvironment, refnode: Element,
                     has_explicit_title: bool, title: str, target: str) -> Tuple[str, str]:
        refnode.attributes.update(env.ref_context)

        if not has_explicit_title:
            # major hax: replace anon names via simple string manipulation.
            # Can this actually fail?
            title = anon_identifier_re.sub("[anonymous]", str(title))

        if not has_explicit_title:
            target = target.lstrip('~')  # only has a meaning for the title
            # if the first character is a tilde, don't display the module/class
            # parts of the contents
            if title[0:1] == '~':
                title = title[1:]
                dot = title.rfind('.')
                if dot != -1:
                    title = title[dot + 1:]
        return title, target
```
### 9 - sphinx/ext/autodoc/typehints.py:

Start line: 83, End line: 139

```python
def modify_field_list(node: nodes.field_list, annotations: Dict[str, str]) -> None:
    arguments = {}  # type: Dict[str, Dict[str, bool]]
    fields = cast(Iterable[nodes.field], node)
    for field in fields:
        field_name = field[0].astext()
        parts = re.split(' +', field_name)
        if parts[0] == 'param':
            if len(parts) == 2:
                # :param xxx:
                arg = arguments.setdefault(parts[1], {})
                arg['param'] = True
            elif len(parts) > 2:
                # :param xxx yyy:
                name = ' '.join(parts[2:])
                arg = arguments.setdefault(name, {})
                arg['param'] = True
                arg['type'] = True
        elif parts[0] == 'type':
            name = ' '.join(parts[1:])
            arg = arguments.setdefault(name, {})
            arg['type'] = True
        elif parts[0] == 'rtype':
            arguments['return'] = {'type': True}

    for name, annotation in annotations.items():
        if name == 'return':
            continue

        arg = arguments.get(name, {})
        if not arg.get('type'):
            field = nodes.field()
            field += nodes.field_name('', 'type ' + name)
            field += nodes.field_body('', nodes.paragraph('', annotation))
            node += field
        if not arg.get('param'):
            field = nodes.field()
            field += nodes.field_name('', 'param ' + name)
            field += nodes.field_body('', nodes.paragraph('', ''))
            node += field

    if 'return' in annotations and 'return' not in arguments:
        field = nodes.field()
        field += nodes.field_name('', 'rtype')
        field += nodes.field_body('', nodes.paragraph('', annotation))
        node += field


def setup(app: Sphinx) -> Dict[str, Any]:
    app.connect('autodoc-process-signature', record_typehints)
    app.connect('object-description-transform', merge_typehints)

    return {
        'version': 'builtin',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
```
### 10 - sphinx/ext/autodoc/__init__.py:

Start line: 1673, End line: 1708

```python
class DataDeclarationDocumenter(DataDocumenter):
    """
    Specialized Documenter subclass for data that cannot be imported
    because they are declared without initial value (refs: PEP-526).
    """
    objtype = 'datadecl'
    directivetype = 'data'
    member_order = 60

    # must be higher than AttributeDocumenter
    priority = 11

    @classmethod
    def can_document_member(cls, member: Any, membername: str, isattr: bool, parent: Any
                            ) -> bool:
        """This documents only INSTANCEATTR members."""
        return (isinstance(parent, ModuleDocumenter) and
                isattr and
                member is INSTANCEATTR)

    def import_object(self, raiseerror: bool = False) -> bool:
        """Never import anything."""
        # disguise as a data
        self.objtype = 'data'
        self.object = UNINITIALIZED_ATTR
        try:
            # import module to obtain type annotation
            self.parent = importlib.import_module(self.modname)
        except ImportError:
            pass

        return True

    def add_content(self, more_content: Any, no_docstring: bool = False) -> None:
        """Never try to get a docstring from the object."""
        super().add_content(more_content, no_docstring=True)
```
### 11 - sphinx/ext/autodoc/__init__.py:

Start line: 1735, End line: 1762

```python
class TypeVarDocumenter(DataDocumenter):
    """
    Specialized Documenter subclass for TypeVars.
    """

    objtype = 'typevar'
    directivetype = 'data'
    priority = DataDocumenter.priority + 1

    @classmethod
    def can_document_member(cls, member: Any, membername: str, isattr: bool, parent: Any
                            ) -> bool:
        return isinstance(member, TypeVar) and isattr  # type: ignore

    def add_directive_header(self, sig: str) -> None:
        self.options.annotation = SUPPRESS  # type: ignore
        super().add_directive_header(sig)

    def get_doc(self, encoding: str = None, ignore: int = None) -> List[List[str]]:
        if ignore is not None:
            warnings.warn("The 'ignore' argument to autodoc.%s.get_doc() is deprecated."
                          % self.__class__.__name__,
                          RemovedInSphinx50Warning, stacklevel=2)

        if self.object.__doc__ != TypeVar.__doc__:
            return super().get_doc()
        else:
            return []
```
### 12 - sphinx/ext/autodoc/__init__.py:

Start line: 1334, End line: 1589

```python
class ClassDocumenter(DocstringSignatureMixin, ModuleLevelDocumenter):  # type: ignore
    """
    Specialized Documenter subclass for classes.
    """
    objtype = 'class'
    member_order = 20
    option_spec = {
        'members': members_option, 'undoc-members': bool_option,
        'noindex': bool_option, 'inherited-members': inherited_members_option,
        'show-inheritance': bool_option, 'member-order': member_order_option,
        'exclude-members': exclude_members_option,
        'private-members': members_option, 'special-members': members_option,
    }  # type: Dict[str, Callable]

    _signature_class = None  # type: Any
    _signature_method_name = None  # type: str

    def __init__(self, *args: Any) -> None:
        super().__init__(*args)
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

        # First, let's see if it has an overloaded __call__ defined
        # in its metaclass
        call = get_user_defined_function_or_method(type(self.object), '__call__')

        if call is not None:
            if "{0.__module__}.{0.__qualname__}".format(call) in _METACLASS_CALL_BLACKLIST:
                call = None

        if call is not None:
            self.env.app.emit('autodoc-before-process-signature', call, True)
            try:
                sig = inspect.signature(call, bound_method=True)
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
                sig = inspect.signature(new, bound_method=True)
                return self.object, '__new__', sig
            except ValueError:
                pass

        # Finally, we should have at least __init__ implemented
        init = get_user_defined_function_or_method(self.object, '__init__')
        if init is not None:
            self.env.app.emit('autodoc-before-process-signature', init, True)
            try:
                sig = inspect.signature(init, bound_method=True)
                return self.object, '__init__', sig
            except ValueError:
                pass

        # None of the attributes are user-defined, so fall back to let inspect
        # handle it.
        # We don't know the exact method that inspect.signature will read
        # the signature from, so just pass the object itself to our hook.
        self.env.app.emit('autodoc-before-process-signature', self.object, False)
        try:
            sig = inspect.signature(self.object, bound_method=False)
            return None, None, sig
        except ValueError:
            pass

        # Still no signature: happens e.g. for old-style classes
        # with __init__ in C and no `__text_signature__`.
        return None, None, None

    def format_args(self, **kwargs: Any) -> str:
        if self.env.config.autodoc_typehints in ('none', 'description'):
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

        sig = super().format_signature()

        overloaded = False
        qualname = None
        # TODO: recreate analyzer for the module of class (To be clear, owner of the method)
        if self._signature_class and self._signature_method_name and self.analyzer:
            qualname = '.'.join([self._signature_class.__qualname__,
                                 self._signature_method_name])
            if qualname in self.analyzer.overloads:
                overloaded = True

        sigs = []
        if overloaded:
            # Use signatures for overloaded methods instead of the implementation method.
            method = safe_getattr(self._signature_class, self._signature_method_name, None)
            __globals__ = safe_getattr(method, '__globals__', {})
            for overload in self.analyzer.overloads.get(qualname):
                overload = evaluate_signature(overload, __globals__)

                parameters = list(overload.parameters.values())
                overload = overload.replace(parameters=parameters[1:],
                                            return_annotation=Parameter.empty)
                sig = stringify_signature(overload, **kwargs)
                sigs.append(sig)
        else:
            sigs.append(sig)

        return "\n".join(sigs)

    def add_directive_header(self, sig: str) -> None:
        sourcename = self.get_sourcename()

        if self.doc_as_attr:
            self.directivetype = 'attribute'
        super().add_directive_header(sig)

        if self.analyzer and '.'.join(self.objpath) in self.analyzer.finals:
            self.add_line('   :final:', sourcename)

        # add inheritance info, if wanted
        if not self.doc_as_attr and self.options.show_inheritance:
            sourcename = self.get_sourcename()
            self.add_line('', sourcename)
            if hasattr(self.object, '__bases__') and len(self.object.__bases__):
                bases = [':class:`%s`' % b.__name__
                         if b.__module__ in ('__builtin__', 'builtins')
                         else ':class:`%s.%s`' % (b.__module__, b.__qualname__)
                         for b in self.object.__bases__]
                self.add_line('   ' + _('Bases: %s') % ', '.join(bases),
                              sourcename)

    def get_doc(self, encoding: str = None, ignore: int = None) -> List[List[str]]:
        if encoding is not None:
            warnings.warn("The 'encoding' argument to autodoc.%s.get_doc() is deprecated."
                          % self.__class__.__name__,
                          RemovedInSphinx40Warning, stacklevel=2)
        lines = getattr(self, '_new_docstrings', None)
        if lines is not None:
            return lines

        content = self.env.config.autoclass_content

        docstrings = []
        attrdocstring = self.get_attr(self.object, '__doc__', None)
        if attrdocstring:
            docstrings.append(attrdocstring)

        # for classes, what the "docstring" is can be controlled via a
        # config value; the default is only the class docstring
        if content in ('both', 'init'):
            __init__ = self.get_attr(self.object, '__init__', None)
            initdocstring = getdoc(__init__, self.get_attr,
                                   self.env.config.autodoc_inherit_docstrings,
                                   self.parent, self.object_name)
            # for new-style classes, no __init__ means default __init__
            if (initdocstring is not None and
                (initdocstring == object.__init__.__doc__ or  # for pypy
                 initdocstring.strip() == object.__init__.__doc__)):  # for !pypy
                initdocstring = None
            if not initdocstring:
                # try __new__
                __new__ = self.get_attr(self.object, '__new__', None)
                initdocstring = getdoc(__new__, self.get_attr,
                                       self.env.config.autodoc_inherit_docstrings,
                                       self.parent, self.object_name)
                # for new-style classes, no __new__ means default __new__
                if (initdocstring is not None and
                    (initdocstring == object.__new__.__doc__ or  # for pypy
                     initdocstring.strip() == object.__new__.__doc__)):  # for !pypy
                    initdocstring = None
            if initdocstring:
                if content == 'init':
                    docstrings = [initdocstring]
                else:
                    docstrings.append(initdocstring)

        tab_width = self.directive.state.document.settings.tab_width
        return [prepare_docstring(docstring, ignore, tab_width) for docstring in docstrings]

    def add_content(self, more_content: Any, no_docstring: bool = False) -> None:
        if self.doc_as_attr:
            classname = safe_getattr(self.object, '__qualname__', None)
            if not classname:
                classname = safe_getattr(self.object, '__name__', None)
            if classname:
                module = safe_getattr(self.object, '__module__', None)
                parentmodule = safe_getattr(self.parent, '__module__', None)
                if module and module != parentmodule:
                    classname = str(module) + '.' + str(classname)
                content = StringList([_('alias of :class:`%s`') % classname], source='')
                super().add_content(content, no_docstring=True)
        else:
            super().add_content(more_content)

    def document_members(self, all_members: bool = False) -> None:
        if self.doc_as_attr:
            return
        super().document_members(all_members)

    def generate(self, more_content: Any = None, real_modname: str = None,
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
### 13 - sphinx/ext/autodoc/__init__.py:

Start line: 1920, End line: 2061

```python
class SingledispatchMethodDocumenter(MethodDocumenter):
    """
    Used to be a specialized Documenter subclass for singledispatch'ed methods.

    Retained for backwards compatibility, now does the same as the MethodDocumenter
    """


class AttributeDocumenter(DocstringStripSignatureMixin, ClassLevelDocumenter):  # type: ignore
    """
    Specialized Documenter subclass for attributes.
    """
    objtype = 'attribute'
    member_order = 60
    option_spec = dict(ModuleLevelDocumenter.option_spec)
    option_spec["annotation"] = annotation_option

    # must be higher than the MethodDocumenter, else it will recognize
    # some non-data descriptors as methods
    priority = 10

    @staticmethod
    def is_function_or_method(obj: Any) -> bool:
        return inspect.isfunction(obj) or inspect.isbuiltin(obj) or inspect.ismethod(obj)

    @classmethod
    def can_document_member(cls, member: Any, membername: str, isattr: bool, parent: Any
                            ) -> bool:
        if inspect.isattributedescriptor(member):
            return True
        elif (not isinstance(parent, ModuleDocumenter) and
              not inspect.isroutine(member) and
              not isinstance(member, type)):
            return True
        else:
            return False

    def document_members(self, all_members: bool = False) -> None:
        pass

    def isinstanceattribute(self) -> bool:
        """Check the subject is an instance attribute."""
        try:
            analyzer = ModuleAnalyzer.for_module(self.modname)
            attr_docs = analyzer.find_attr_docs()
            if self.objpath:
                key = ('.'.join(self.objpath[:-1]), self.objpath[-1])
                if key in attr_docs:
                    return True

            return False
        except PycodeError:
            return False

    def import_object(self, raiseerror: bool = False) -> bool:
        try:
            ret = super().import_object(raiseerror=True)
            if inspect.isenumattribute(self.object):
                self.object = self.object.value
            if inspect.isattributedescriptor(self.object):
                self._datadescriptor = True
            else:
                # if it's not a data descriptor
                self._datadescriptor = False
        except ImportError as exc:
            if self.isinstanceattribute():
                self.object = INSTANCEATTR
                self._datadescriptor = False
                ret = True
            elif raiseerror:
                raise
            else:
                logger.warning(exc.args[0], type='autodoc', subtype='import_object')
                self.env.note_reread()
                ret = False

        return ret

    def get_real_modname(self) -> str:
        return self.get_attr(self.parent or self.object, '__module__', None) \
            or self.modname

    def add_directive_header(self, sig: str) -> None:
        super().add_directive_header(sig)
        sourcename = self.get_sourcename()
        if not self.options.annotation:
            # obtain type annotation for this attribute
            try:
                annotations = get_type_hints(self.parent)
            except NameError:
                # Failed to evaluate ForwardRef (maybe TYPE_CHECKING)
                annotations = safe_getattr(self.parent, '__annotations__', {})
            except TypeError:
                annotations = {}
            except KeyError:
                # a broken class found (refs: https://github.com/sphinx-doc/sphinx/issues/8084)
                annotations = {}
            except AttributeError:
                # AttributeError is raised on 3.5.2 (fixed by 3.5.3)
                annotations = {}

            if self.objpath[-1] in annotations:
                objrepr = stringify_typehint(annotations.get(self.objpath[-1]))
                self.add_line('   :type: ' + objrepr, sourcename)
            else:
                key = ('.'.join(self.objpath[:-1]), self.objpath[-1])
                if self.analyzer and key in self.analyzer.annotations:
                    self.add_line('   :type: ' + self.analyzer.annotations[key],
                                  sourcename)

            # data descriptors do not have useful values
            if not self._datadescriptor:
                try:
                    if self.object is INSTANCEATTR:
                        pass
                    else:
                        objrepr = object_description(self.object)
                        self.add_line('   :value: ' + objrepr, sourcename)
                except ValueError:
                    pass
        elif self.options.annotation is SUPPRESS:
            pass
        else:
            self.add_line('   :annotation: %s' % self.options.annotation, sourcename)

    def get_doc(self, encoding: str = None, ignore: int = None) -> List[List[str]]:
        try:
            # Disable `autodoc_inherit_docstring` temporarily to avoid to obtain
            # a docstring from the value which descriptor returns unexpectedly.
            # ref: https://github.com/sphinx-doc/sphinx/issues/7805
            orig = self.env.config.autodoc_inherit_docstrings
            self.env.config.autodoc_inherit_docstrings = False  # type: ignore
            return super().get_doc(encoding, ignore)
        finally:
            self.env.config.autodoc_inherit_docstrings = orig  # type: ignore

    def add_content(self, more_content: Any, no_docstring: bool = False) -> None:
        if not self._datadescriptor:
            # if it's not a data descriptor, its docstring is very probably the
            # wrong thing to display
            no_docstring = True
        super().add_content(more_content, no_docstring)
```
### 14 - sphinx/ext/autodoc/__init__.py:

Start line: 648, End line: 744

```python
class Documenter:

    def filter_members(self, members: List[Tuple[str, Any]], want_all: bool
                       ) -> List[Tuple[str, Any, bool]]:
        # ... other code
        for (membername, member) in members:
            # if isattr is True, the member is documented as an attribute
            if member is INSTANCEATTR:
                isattr = True
            else:
                isattr = False

            doc = getdoc(member, self.get_attr, self.env.config.autodoc_inherit_docstrings,
                         self.parent, self.object_name)
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
            has_doc = bool(doc)

            metadata = extract_metadata(doc)
            if 'private' in metadata:
                # consider a member private if docstring has "private" metadata
                isprivate = True
            elif 'public' in metadata:
                # consider a member public if docstring has "public" metadata
                isprivate = False
            else:
                isprivate = membername.startswith('_')

            keep = False
            if safe_getattr(member, '__sphinx_mock__', False):
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
                    elif is_filtered_inherited_member(membername):
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
                isattr = True
            elif want_all and isprivate:
                if has_doc or self.options.undoc_members:
                    if self.options.private_members is None:
                        keep = False
                    elif is_filtered_inherited_member(membername):
                        keep = False
                    else:
                        keep = membername in self.options.private_members
                else:
                    keep = False
            else:
                if self.options.members is ALL and is_filtered_inherited_member(membername):
                    keep = False
                else:
                    # ignore undocumented members if :undoc-members: is not given
                    keep = has_doc or self.options.undoc_members

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
### 15 - sphinx/ext/autodoc/__init__.py:

Start line: 986, End line: 1007

```python
class ModuleDocumenter(Documenter):

    def get_object_members(self, want_all: bool) -> Tuple[bool, List[Tuple[str, Any]]]:
        if want_all:
            if self.__all__:
                memberlist = self.__all__
            else:
                # for implicit module members, check __module__ to avoid
                # documenting imported objects
                return True, get_module_members(self.object)
        else:
            memberlist = self.options.members or []
        ret = []
        for mname in memberlist:
            try:
                ret.append((mname, safe_getattr(self.object, mname)))
            except AttributeError:
                logger.warning(
                    __('missing attribute mentioned in :members: or __all__: '
                       'module %s, attribute %s') %
                    (safe_getattr(self.object, '__name__', '???'), mname),
                    type='autodoc'
                )
        return False, ret
```
### 17 - sphinx/ext/autodoc/__init__.py:

Start line: 2216, End line: 2253

```python
def setup(app: Sphinx) -> Dict[str, Any]:
    app.add_autodocumenter(ModuleDocumenter)
    app.add_autodocumenter(ClassDocumenter)
    app.add_autodocumenter(ExceptionDocumenter)
    app.add_autodocumenter(DataDocumenter)
    app.add_autodocumenter(DataDeclarationDocumenter)
    app.add_autodocumenter(GenericAliasDocumenter)
    app.add_autodocumenter(TypeVarDocumenter)
    app.add_autodocumenter(FunctionDocumenter)
    app.add_autodocumenter(DecoratorDocumenter)
    app.add_autodocumenter(MethodDocumenter)
    app.add_autodocumenter(AttributeDocumenter)
    app.add_autodocumenter(PropertyDocumenter)
    app.add_autodocumenter(InstanceAttributeDocumenter)
    app.add_autodocumenter(SlotsAttributeDocumenter)

    app.add_config_value('autoclass_content', 'class', True, ENUM('both', 'class', 'init'))
    app.add_config_value('autodoc_member_order', 'alphabetical', True,
                         ENUM('alphabetic', 'alphabetical', 'bysource', 'groupwise'))
    app.add_config_value('autodoc_default_options', {}, True)
    app.add_config_value('autodoc_docstring_signature', True, True)
    app.add_config_value('autodoc_mock_imports', [], True)
    app.add_config_value('autodoc_typehints', "signature", True,
                         ENUM("signature", "description", "none"))
    app.add_config_value('autodoc_warningiserror', True, True)
    app.add_config_value('autodoc_inherit_docstrings', True, True)
    app.add_event('autodoc-before-process-signature')
    app.add_event('autodoc-process-docstring')
    app.add_event('autodoc-process-signature')
    app.add_event('autodoc-skip-member')

    app.connect('config-inited', migrate_autodoc_member_order, priority=800)

    app.setup_extension('sphinx.ext.autodoc.type_comment')
    app.setup_extension('sphinx.ext.autodoc.typehints')

    return {'version': sphinx.__display_version__, 'parallel_read_safe': True}
```
### 19 - sphinx/ext/autodoc/__init__.py:

Start line: 261, End line: 334

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
    the documenter.
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

    option_spec = {'noindex': bool_option}  # type: Dict[str, Callable]

    def get_attr(self, obj: Any, name: str, *defargs: Any) -> Any:
        """getattr() override for types such as Zope interfaces."""
        return autodoc_attrgetter(self.env.app, obj, name, *defargs)

    @classmethod
    def can_document_member(cls, member: Any, membername: str, isattr: bool, parent: Any
                            ) -> bool:
        """Called to see if a member can be documented by this documenter."""
        raise NotImplementedError('must be implemented in subclasses')

    def __init__(self, directive: "DocumenterBridge", name: str, indent: str = '') -> None:
        self.directive = directive
        self.env = directive.env    # type: BuildEnvironment
        self.options = directive.genopt
        self.name = name
        self.indent = indent
        # the module and object path within the module, and the fully
        # qualified name (all set after resolve_name succeeds)
        self.modname = None         # type: str
        self.module = None          # type: ModuleType
        self.objpath = None         # type: List[str]
        self.fullname = None        # type: str
        # extra signature items (arguments and return annotation,
        # also set after resolve_name succeeds)
        self.args = None            # type: str
        self.retann = None          # type: str
        # the object to document (set after import_object succeeds)
        self.object = None          # type: Any
        self.object_name = None     # type: str
        # the parent/owner of the object to document
        self.parent = None          # type: Any
        # the module analyzer to get at attribute docs, or None
        self.analyzer = None        # type: ModuleAnalyzer

    @property
    def documenters(self) -> Dict[str, "Type[Documenter]"]:
        """Returns registered Documenter classes"""
        return self.env.app.registry.documenters

    def add_line(self, line: str, source: str, *lineno: int) -> None:
        """Append one line of generated reST to the output."""
        if line.strip():  # not a blank line
            self.directive.result.append(self.indent + line, source, *lineno)
        else:
            self.directive.result.append('', source, *lineno)
```
### 20 - sphinx/util/typing.py:

Start line: 11, End line: 60

```python
import sys
import typing
from typing import Any, Callable, Dict, Generator, List, Tuple, TypeVar, Union

from docutils import nodes
from docutils.parsers.rst.states import Inliner


if sys.version_info > (3, 7):
    from typing import ForwardRef
else:
    from typing import _ForwardRef  # type: ignore

    class ForwardRef:
        """A pseudo ForwardRef class for py35 and py36."""
        def __init__(self, arg: Any, is_argument: bool = True) -> None:
            self.arg = arg

        def _evaluate(self, globalns: Dict, localns: Dict) -> Any:
            ref = _ForwardRef(self.arg)
            return ref._eval_type(globalns, localns)


# An entry of Directive.option_spec
DirectiveOption = Callable[[str], Any]

# Text like nodes which are initialized with text and rawsource
TextlikeNode = Union[nodes.Text, nodes.TextElement]

# type of None
NoneType = type(None)

# path matcher
PathMatcher = Callable[[str], bool]

# common role functions
RoleFunction = Callable[[str, str, str, int, Inliner, Dict[str, Any], List[str]],
                        Tuple[List[nodes.Node], List[nodes.system_message]]]

# title getter functions for enumerable nodes (see sphinx.domains.std)
TitleGetter = Callable[[nodes.Node], str]

# inventory data on memory
Inventory = Dict[str, Dict[str, Tuple[str, str, str, str]]]


def is_system_TypeVar(typ: Any) -> bool:
    """Check *typ* is system defined TypeVar."""
    modname = getattr(typ, '__module__', '')
    return modname == 'typing' and isinstance(typ, TypeVar)  # type: ignore
```
### 23 - sphinx/ext/autodoc/__init__.py:

Start line: 1295, End line: 1589

```python
class SingledispatchFunctionDocumenter(FunctionDocumenter):
    """
    Used to be a specialized Documenter subclass for singledispatch'ed functions.

    Retained for backwards compatibility, now does the same as the FunctionDocumenter
    """


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


# Types whose __new__ signature is a pass-thru.
_CLASS_NEW_BLACKLIST = [
    'typing.Generic.__new__',
]


class ClassDocumenter(DocstringSignatureMixin, ModuleLevelDocumenter):
```
### 24 - sphinx/ext/autodoc/__init__.py:

Start line: 2095, End line: 2136

```python
class InstanceAttributeDocumenter(AttributeDocumenter):
    """
    Specialized Documenter subclass for attributes that cannot be imported
    because they are instance attributes (e.g. assigned in __init__).
    """
    objtype = 'instanceattribute'
    directivetype = 'attribute'
    member_order = 60

    # must be higher than AttributeDocumenter
    priority = 11

    @classmethod
    def can_document_member(cls, member: Any, membername: str, isattr: bool, parent: Any
                            ) -> bool:
        """This documents only INSTANCEATTR members."""
        return (not isinstance(parent, ModuleDocumenter) and
                isattr and
                member is INSTANCEATTR)

    def import_parent(self) -> Any:
        try:
            parent = importlib.import_module(self.modname)
            for name in self.objpath[:-1]:
                parent = self.get_attr(parent, name)

            return parent
        except (ImportError, AttributeError):
            return None

    def import_object(self, raiseerror: bool = False) -> bool:
        """Never import anything."""
        # disguise as an attribute
        self.objtype = 'attribute'
        self.object = INSTANCEATTR
        self.parent = self.import_parent()
        self._datadescriptor = False
        return True

    def add_content(self, more_content: Any, no_docstring: bool = False) -> None:
        """Never try to get a docstring from the object."""
        super().add_content(more_content, no_docstring=True)
```
### 30 - sphinx/ext/autodoc/__init__.py:

Start line: 2193, End line: 2213

```python
def get_documenters(app: Sphinx) -> Dict[str, "Type[Documenter]"]:
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
```
### 35 - sphinx/ext/autodoc/__init__.py:

Start line: 1009, End line: 1026

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
                if name in self.__all__:
                    return self.__all__.index(name)
                else:
                    return len(self.__all__)
            documenters.sort(key=keyfunc)

            return documenters
        else:
            return super().sort_members(documenters, order)
```
### 36 - sphinx/ext/autodoc/__init__.py:

Start line: 973, End line: 984

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
### 37 - sphinx/ext/autodoc/__init__.py:

Start line: 2157, End line: 2176

```python
class SlotsAttributeDocumenter(AttributeDocumenter):

    def import_object(self, raiseerror: bool = False) -> bool:
        """Never import anything."""
        # disguise as an attribute
        self.objtype = 'attribute'
        self._datadescriptor = True

        with mock(self.env.config.autodoc_mock_imports):
            try:
                ret = import_object(self.modname, self.objpath[:-1], 'class',
                                    attrgetter=self.get_attr,
                                    warningiserror=self.env.config.autodoc_warningiserror)
                self.module, _, _, self.parent = ret
                return True
            except ImportError as exc:
                if raiseerror:
                    raise
                else:
                    logger.warning(exc.args[0], type='autodoc', subtype='import_object')
                    self.env.note_reread()
                    return False
```
### 43 - sphinx/util/typing.py:

Start line: 153, End line: 230

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
            elif annotation.__origin__ == Generator:  # type: ignore
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
            if len(params) == 2 and params[1] is NoneType:
                return 'Optional[%s]' % stringify(params[0])
            else:
                param_str = ', '.join(stringify(p) for p in params)
                return '%s[%s]' % (qualname, param_str)
    elif (hasattr(annotation, '__origin__') and
          annotation.__origin__ is typing.Union):  # for Python 3.5.2+
        params = annotation.__args__
        if params is not None:
            if len(params) > 1 and params[-1] is NoneType:
                if len(params) > 2:
                    param_str = ", ".join(stringify(p) for p in params[:-1])
                    return 'Optional[Union[%s]]' % param_str
                else:
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
### 44 - sphinx/ext/autodoc/__init__.py:

Start line: 1050, End line: 1080

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
### 57 - sphinx/util/inspect.py:

Start line: 738, End line: 811

```python
class Signature:

    def format_args(self, show_annotation: bool = True) -> str:
        def get_annotation(param: Parameter) -> Any:
            if isinstance(param.annotation, str) and param.name in self.annotations:
                return self.annotations[param.name]
            else:
                return param.annotation

        args = []
        last_kind = None
        for i, param in enumerate(self.parameters.values()):
            # skip first argument if subject is bound method
            if self.skip_first_argument and i == 0:
                continue

            arg = StringIO()

            # insert '*' between POSITIONAL args and KEYWORD_ONLY args::
            #     func(a, b, *, c, d):
            if param.kind == param.KEYWORD_ONLY and last_kind in (param.POSITIONAL_OR_KEYWORD,
                                                                  param.POSITIONAL_ONLY,
                                                                  None):
                args.append('*')

            if param.kind in (param.POSITIONAL_ONLY,
                              param.POSITIONAL_OR_KEYWORD,
                              param.KEYWORD_ONLY):
                arg.write(param.name)
                if show_annotation and param.annotation is not param.empty:
                    arg.write(': ')
                    arg.write(stringify_annotation(get_annotation(param)))
                if param.default is not param.empty:
                    if param.annotation is param.empty or show_annotation is False:
                        arg.write('=')
                        arg.write(object_description(param.default))
                    else:
                        arg.write(' = ')
                        arg.write(object_description(param.default))
            elif param.kind == param.VAR_POSITIONAL:
                arg.write('*')
                arg.write(param.name)
                if show_annotation and param.annotation is not param.empty:
                    arg.write(': ')
                    arg.write(stringify_annotation(get_annotation(param)))
            elif param.kind == param.VAR_KEYWORD:
                arg.write('**')
                arg.write(param.name)
                if show_annotation and param.annotation is not param.empty:
                    arg.write(': ')
                    arg.write(stringify_annotation(get_annotation(param)))

            args.append(arg.getvalue())
            last_kind = param.kind

        if self.return_annotation is Parameter.empty or show_annotation is False:
            return '(%s)' % ', '.join(args)
        else:
            if 'return' in self.annotations:
                annotation = stringify_annotation(self.annotations['return'])
            else:
                annotation = stringify_annotation(self.return_annotation)

            return '(%s) -> %s' % (', '.join(args), annotation)

    def format_annotation(self, annotation: Any) -> str:
        """Return formatted representation of a type annotation."""
        return stringify_annotation(annotation)

    def format_annotation_new(self, annotation: Any) -> str:
        """format_annotation() for py37+"""
        return stringify_annotation(annotation)

    def format_annotation_old(self, annotation: Any) -> str:
        """format_annotation() for py36 or below"""
        return stringify_annotation(annotation)
```
### 62 - sphinx/ext/autodoc/__init__.py:

Start line: 1029, End line: 1047

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
### 64 - sphinx/ext/autodoc/__init__.py:

Start line: 746, End line: 789

```python
class Documenter:

    def document_members(self, all_members: bool = False) -> None:
        """Generate reST for member documentation.

        If *all_members* is True, do all members, else those given by
        *self.options.members*.
        """
        # set current namespace for finding members
        self.env.temp_data['autodoc:module'] = self.modname
        if self.objpath:
            self.env.temp_data['autodoc:class'] = self.objpath[0]

        want_all = all_members or self.options.inherited_members or \
            self.options.members is ALL
        # find out which members are documentable
        members_check_module, members = self.get_object_members(want_all)

        # document non-skipped members
        memberdocumenters = []  # type: List[Tuple[Documenter, bool]]
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
            full_mname = self.modname + '::' + \
                '.'.join(self.objpath + [mname])
            documenter = classes[-1](self.directive, full_mname, self.indent)
            memberdocumenters.append((documenter, isattr))

        member_order = self.options.member_order or self.env.config.autodoc_member_order
        memberdocumenters = self.sort_members(memberdocumenters, member_order)

        for documenter, isattr in memberdocumenters:
            documenter.generate(
                all_members=True, real_modname=self.real_modname,
                check_module=members_check_module and not isattr)

        # reset current objects
        self.env.temp_data['autodoc:module'] = None
        self.env.temp_data['autodoc:class'] = None
```
### 69 - sphinx/ext/autodoc/__init__.py:

Start line: 155, End line: 181

```python
def merge_special_members_option(options: Dict) -> None:
    """Merge :special-members: option to :members: option."""
    warnings.warn("merge_special_members_option() is deprecated.",
                  RemovedInSphinx50Warning, stacklevel=2)
    if 'special-members' in options and options['special-members'] is not ALL:
        if options.get('members') is ALL:
            pass
        elif options.get('members'):
            for member in options['special-members']:
                if member not in options['members']:
                    options['members'].append(member)
        else:
            options['members'] = options['special-members']


def merge_members_option(options: Dict) -> None:
    """Merge :*-members: option to the :members: option."""
    if options.get('members') is ALL:
        # merging is not needed when members: ALL
        return

    members = options.setdefault('members', [])
    for key in {'private-members', 'special-members'}:
        if key in options and options[key] not in (ALL, None):
            for member in options[key]:
                if member not in members:
                    members.append(member)
```
### 71 - sphinx/ext/autodoc/__init__.py:

Start line: 490, End line: 510

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
### 72 - sphinx/util/typing.py:

Start line: 63, End line: 82

```python
def stringify(annotation: Any) -> str:
    """Stringify type annotation object."""
    if isinstance(annotation, str):
        return annotation
    elif isinstance(annotation, TypeVar):  # type: ignore
        return annotation.__name__
    elif not annotation:
        return repr(annotation)
    elif annotation is NoneType:
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
### 75 - sphinx/util/inspect.py:

Start line: 11, End line: 48

```python
import builtins
import contextlib
import enum
import inspect
import re
import sys
import types
import typing
import warnings
from functools import partial, partialmethod
from inspect import (  # NOQA
    Parameter, isclass, ismethod, ismethoddescriptor, ismodule
)
from io import StringIO
from typing import Any, Callable, Dict, Mapping, List, Optional, Tuple
from typing import cast

from sphinx.deprecation import RemovedInSphinx40Warning, RemovedInSphinx50Warning
from sphinx.pycode.ast import ast  # for py35-37
from sphinx.pycode.ast import unparse as ast_unparse
from sphinx.util import logging
from sphinx.util.typing import ForwardRef
from sphinx.util.typing import stringify as stringify_annotation

if sys.version_info > (3, 7):
    from types import (
        ClassMethodDescriptorType,
        MethodDescriptorType,
        WrapperDescriptorType
    )
else:
    ClassMethodDescriptorType = type(object.__init__)
    MethodDescriptorType = type(str.join)
    WrapperDescriptorType = type(dict.__dict__['fromkeys'])

logger = logging.getLogger(__name__)

memory_address_re = re.compile(r' at 0x[0-9a-f]{8,16}(?=>)', re.IGNORECASE)
```
