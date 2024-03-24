# pylint-dev__astroid-1268

| **pylint-dev/astroid** | `ce5cbce5ba11cdc2f8139ade66feea1e181a7944` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | 640 |
| **Avg pos** | 171.0 |
| **Min pos** | 1 |
| **Max pos** | 33 |
| **Top file pos** | 1 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/astroid/nodes/as_string.py b/astroid/nodes/as_string.py
--- a/astroid/nodes/as_string.py
+++ b/astroid/nodes/as_string.py
@@ -36,6 +36,7 @@
         MatchSingleton,
         MatchStar,
         MatchValue,
+        Unknown,
     )
 
 # pylint: disable=unused-argument
@@ -643,6 +644,9 @@ def visit_property(self, node):
     def visit_evaluatedobject(self, node):
         return node.original.accept(self)
 
+    def visit_unknown(self, node: "Unknown") -> str:
+        return str(node)
+
 
 def _import_string(names):
     """return a list of (name, asname) formatted as a string"""

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| astroid/nodes/as_string.py | 36 | - | - | 1 | -
| astroid/nodes/as_string.py | 643 | - | 33 | 1 | 14165


## Problem Statement

```
'AsStringVisitor' object has no attribute 'visit_unknown'
\`\`\`python
>>> import astroid
>>> astroid.nodes.Unknown().as_string()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/Users/tusharsadhwani/code/marvin-python/venv/lib/python3.9/site-packages/astroid/nodes/node_ng.py", line 609, in as_string
    return AsStringVisitor()(self)
  File "/Users/tusharsadhwani/code/marvin-python/venv/lib/python3.9/site-packages/astroid/nodes/as_string.py", line 56, in __call__
    return node.accept(self).replace(DOC_NEWLINE, "\n")
  File "/Users/tusharsadhwani/code/marvin-python/venv/lib/python3.9/site-packages/astroid/nodes/node_ng.py", line 220, in accept
    func = getattr(visitor, "visit_" + self.__class__.__name__.lower())
AttributeError: 'AsStringVisitor' object has no attribute 'visit_unknown'
>>> 
\`\`\`
### `python -c "from astroid import __pkginfo__; print(__pkginfo__.version)"` output

2.8.6-dev0

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- |
| **-> 1 <-** | **1 astroid/nodes/as_string.py** | 98 | 173| 640 | 640 | 
| **-> 2 <-** | **1 astroid/nodes/as_string.py** | 488 | 573| 809 | 1449 | 
| **-> 3 <-** | **1 astroid/nodes/as_string.py** | 344 | 432| 808 | 2257 | 
| **-> 4 <-** | **1 astroid/nodes/as_string.py** | 188 | 282| 791 | 3048 | 
| **-> 5 <-** | **1 astroid/nodes/as_string.py** | 434 | 474| 356 | 3404 | 
| **-> 6 <-** | **1 astroid/nodes/as_string.py** | 314 | 323| 123 | 3527 | 
| **-> 7 <-** | **1 astroid/nodes/as_string.py** | 290 | 288| 281 | 3808 | 
| **-> 8 <-** | **1 astroid/nodes/as_string.py** | 609 | 607| 181 | 3989 | 
| **-> 9 <-** | **1 astroid/nodes/as_string.py** | 45 | 42| 373 | 4362 | 
| **-> 10 <-** | **1 astroid/nodes/as_string.py** | 622 | 659| 261 | 4623 | 
| 11 | 2 astroid/nodes/node_ng.py | 0 | 50| 325 | 4948 | 
| 12 | 2 astroid/nodes/node_ng.py | 217 | 275| 418 | 5366 | 
| **-> 13 <-** | **2 astroid/nodes/as_string.py** | 593 | 603| 146 | 5512 | 
| **-> 14 <-** | **2 astroid/nodes/as_string.py** | 581 | 579| 184 | 5696 | 
| **-> 15 <-** | **2 astroid/nodes/as_string.py** | 476 | 486| 137 | 5833 | 
| 16 | 3 astroid/nodes/node_classes.py | 4318 | 4332| 108 | 5941 | 
| 17 | 4 astroid/rebuilder.py | 158 | 379| 1591 | 7532 | 
| 18 | 4 astroid/rebuilder.py | 506 | 729| 1621 | 9153 | 
| 19 | 5 astroid/exceptions.py | 17 | 51| 197 | 9350 | 
| **-> 20 <-** | **5 astroid/nodes/as_string.py** | 175 | 186| 167 | 9517 | 
| **-> 21 <-** | **5 astroid/nodes/as_string.py** | 325 | 342| 149 | 9666 | 
| 22 | 5 astroid/rebuilder.py | 1490 | 1580| 781 | 10447 | 
| 23 | 6 astroid/builder.py | 222 | 263| 279 | 10726 | 
| 24 | 7 astroid/manager.py | 31 | 60| 181 | 10907 | 
| 25 | 8 astroid/interpreter/objectmodel.py | 130 | 201| 444 | 11351 | 
| 26 | 8 astroid/rebuilder.py | 1363 | 1394| 312 | 11663 | 
| 27 | 9 astroid/__pkginfo__.py | 26 | 28| 18 | 11681 | 
| 28 | 9 astroid/rebuilder.py | 1411 | 1470| 661 | 12342 | 
| 29 | 10 astroid/__init__.py | 42 | 170| 701 | 13043 | 
| 30 | 11 astroid/node_classes.py | 0 | 93| 409 | 13452 | 
| 31 | 12 astroid/transforms.py | 56 | 96| 318 | 13770 | 
| 32 | 13 astroid/_ast.py | 22 | 19| 244 | 14014 | 
| **-> 33 <-** | **13 astroid/nodes/as_string.py** | 76 | 96| 151 | 14165 | 
| 34 | 14 astroid/nodes/__init__.py | 216 | 309| 443 | 14608 | 
| 35 | 15 astroid/brain/brain_gi.py | 25 | 68| 209 | 14817 | 
| 36 | 15 astroid/rebuilder.py | 32 | 82| 312 | 15129 | 
| 37 | 15 astroid/rebuilder.py | 1251 | 1281| 258 | 15387 | 
| 38 | 16 astroid/nodes/scoped_nodes.py | 42 | 96| 330 | 15717 | 
| 39 | 16 astroid/nodes/__init__.py | 26 | 214| 731 | 16448 | 
| 40 | 17 astroid/raw_building.py | 26 | 59| 219 | 16667 | 
| 41 | 17 astroid/transforms.py | 18 | 15| 283 | 16950 | 
| 42 | 17 astroid/brain/brain_gi.py | 222 | 262| 245 | 17195 | 
| 43 | 18 astroid/interpreter/_import/util.py | 4 | 16| 55 | 17250 | 
| 44 | 18 astroid/interpreter/objectmodel.py | 611 | 706| 629 | 17879 | 
| 45 | 18 astroid/rebuilder.py | 1396 | 1409| 130 | 18009 | 
| 46 | 19 astroid/brain/brain_argparse.py | 5 | 2| 235 | 18244 | 
| 47 | 19 astroid/exceptions.py | 78 | 91| 113 | 18357 | 
| 48 | 19 astroid/raw_building.py | 389 | 430| 360 | 18717 | 
| 49 | 19 astroid/rebuilder.py | 890 | 888| 296 | 19013 | 
| 50 | 19 astroid/exceptions.py | 274 | 301| 237 | 19250 | 
| 51 | 19 astroid/interpreter/objectmodel.py | 377 | 402| 247 | 19497 | 
| 52 | 19 astroid/rebuilder.py | 381 | 489| 786 | 20283 | 
| 53 | 19 astroid/interpreter/objectmodel.py | 54 | 51| 237 | 20520 | 
| 54 | 19 astroid/rebuilder.py | 1712 | 1726| 185 | 20705 | 
| 55 | 19 astroid/rebuilder.py | 1640 | 1689| 492 | 21197 | 
| 56 | 19 astroid/manager.py | 128 | 154| 251 | 21448 | 
| 57 | 20 astroid/brain/brain_http.py | 17 | 142| 1512 | 22960 | 
| 58 | 20 astroid/nodes/node_ng.py | 204 | 215| 112 | 23072 | 
| 59 | 20 astroid/nodes/node_classes.py | 38 | 71| 196 | 23268 | 
| 60 | 20 astroid/exceptions.py | 113 | 142| 172 | 23440 | 
| 61 | 20 astroid/rebuilder.py | 1167 | 1198| 340 | 23780 | 
| 62 | 21 astroid/modutils.py | 121 | 184| 488 | 24268 | 
| 63 | 21 astroid/rebuilder.py | 947 | 961| 188 | 24456 | 
| 64 | 21 astroid/rebuilder.py | 1612 | 1620| 119 | 24575 | 
| 65 | 22 astroid/bases.py | 30 | 81| 330 | 24905 | 
| 66 | 22 astroid/rebuilder.py | 1283 | 1325| 379 | 25284 | 
| 67 | 22 astroid/nodes/node_ng.py | 538 | 608| 421 | 25705 | 
| 68 | 23 astroid/brain/brain_typing.py | 14 | 114| 577 | 26282 | 
| 69 | 23 astroid/rebuilder.py | 139 | 137| 211 | 26493 | 
| 70 | 23 astroid/builder.py | 23 | 65| 299 | 26792 | 
| 71 | 23 astroid/rebuilder.py | 731 | 787| 439 | 27231 | 
| 72 | 23 astroid/brain/brain_http.py | 145 | 215| 745 | 27976 | 
| 73 | 23 astroid/rebuilder.py | 1343 | 1361| 239 | 28215 | 
| 74 | 23 astroid/nodes/node_ng.py | 692 | 741| 451 | 28666 | 
| 75 | 24 astroid/scoped_nodes.py | 0 | 29| 176 | 28842 | 
| 76 | 24 astroid/exceptions.py | 94 | 110| 116 | 28958 | 
| 77 | 24 astroid/rebuilder.py | 1327 | 1341| 150 | 29108 | 
| 78 | 24 astroid/rebuilder.py | 1728 | 1827| 887 | 29995 | 
| 79 | 24 astroid/raw_building.py | 268 | 290| 168 | 30163 | 
| 80 | 24 astroid/modutils.py | 41 | 120| 773 | 30936 | 
| 81 | 24 astroid/interpreter/objectmodel.py | 204 | 225| 159 | 31095 | 
| 82 | 25 astroid/brain/brain_fstrings.py | 34 | 52| 194 | 31289 | 
| 83 | 26 astroid/brain/brain_builtin_inference.py | 23 | 118| 652 | 31941 | 
| 84 | 26 astroid/exceptions.py | 217 | 271| 364 | 32305 | 
| 85 | 26 astroid/_ast.py | 84 | 81| 147 | 32452 | 
| 86 | 26 astroid/rebuilder.py | 1066 | 1090| 220 | 32672 | 
| 87 | 26 astroid/rebuilder.py | 1124 | 1144| 205 | 32877 | 
| 88 | 27 astroid/helpers.py | 53 | 71| 161 | 33038 | 
| 89 | 27 astroid/rebuilder.py | 997 | 1013| 115 | 33153 | 
| 90 | 27 astroid/rebuilder.py | 491 | 502| 155 | 33308 | 
| 91 | 27 astroid/builder.py | 270 | 267| 191 | 33499 | 
| 92 | 27 astroid/manager.py | 303 | 345| 347 | 33846 | 
| 93 | 27 astroid/builder.py | 147 | 145| 233 | 34079 | 
| 94 | 27 astroid/nodes/node_classes.py | 2649 | 2704| 438 | 34517 | 
| 95 | 27 astroid/brain/brain_gi.py | 169 | 219| 440 | 34957 | 
| 96 | 28 astroid/inference.py | 254 | 274| 138 | 35095 | 
| 97 | 28 astroid/interpreter/objectmodel.py | 405 | 436| 212 | 35307 | 
| 98 | 29 astroid/brain/brain_pytest.py | 18 | 15| 455 | 35762 | 
| 99 | 30 astroid/protocols.py | 275 | 286| 150 | 35912 | 
| 100 | 30 astroid/interpreter/objectmodel.py | 531 | 528| 215 | 36127 | 
| 101 | 30 astroid/nodes/node_classes.py | 4802 | 4844| 350 | 36477 | 
| 102 | 31 astroid/brain/brain_numpy_utils.py | 10 | 42| 223 | 36700 | 
| 103 | 31 astroid/nodes/scoped_nodes.py | 712 | 738| 219 | 36919 | 
| 104 | 31 astroid/rebuilder.py | 1233 | 1246| 161 | 37080 | 
| 105 | 31 astroid/nodes/node_ng.py | 53 | 108| 472 | 37552 | 
| 106 | 32 astroid/mixins.py | 104 | 118| 124 | 37676 | 
| 107 | 33 astroid/brain/brain_namedtuple_enum.py | 29 | 65| 213 | 37889 | 
| 108 | 34 astroid/const.py | 0 | 20| 131 | 38020 | 
| 109 | 34 astroid/rebuilder.py | 1152 | 1150| 211 | 38231 | 
| 110 | 34 astroid/rebuilder.py | 789 | 800| 153 | 38384 | 
| 111 | 34 astroid/interpreter/objectmodel.py | 267 | 288| 179 | 38563 | 
| 112 | 34 astroid/protocols.py | 712 | 795| 610 | 39173 | 
| 113 | 34 astroid/rebuilder.py | 1472 | 1488| 217 | 39390 | 
| 114 | 34 astroid/manager.py | 270 | 286| 156 | 39546 | 
| 115 | 34 astroid/rebuilder.py | 963 | 972| 115 | 39661 | 
| 116 | 34 astroid/nodes/scoped_nodes.py | 604 | 602| 284 | 39945 | 
| 117 | 34 astroid/rebuilder.py | 974 | 995| 190 | 40135 | 
| 118 | 34 astroid/nodes/node_ng.py | 648 | 660| 189 | 40324 | 
| 119 | 34 astroid/inference.py | 334 | 357| 168 | 40492 | 
| 120 | 35 astroid/objects.py | 80 | 133| 405 | 40897 | 
| 121 | 35 astroid/rebuilder.py | 1096 | 1094| 176 | 41073 | 
| 122 | 35 astroid/protocols.py | 30 | 131| 754 | 41827 | 
| 123 | 36 astroid/interpreter/_import/spec.py | 150 | 171| 204 | 42031 | 
| 124 | 37 astroid/brain/brain_random.py | 22 | 19| 247 | 42278 | 
| 125 | 38 astroid/brain/brain_re.py | 9 | 6| 254 | 42532 | 
| 126 | 39 astroid/brain/brain_functools.py | 7 | 19| 116 | 42648 | 
| 127 | 39 astroid/interpreter/objectmodel.py | 98 | 96| 155 | 42803 | 
| 128 | 40 astroid/brain/brain_ssl.py | 17 | 14| 661 | 43464 | 
| 129 | 40 astroid/rebuilder.py | 813 | 879| 663 | 44127 | 
| 130 | 40 astroid/rebuilder.py | 1015 | 1026| 114 | 44241 | 
| 131 | 40 astroid/manager.py | 156 | 220| 499 | 44740 | 
| 132 | 40 astroid/inference.py | 786 | 817| 243 | 44983 | 
| 133 | 40 astroid/exceptions.py | 54 | 75| 191 | 45174 | 
| 134 | 40 astroid/brain/brain_builtin_inference.py | 902 | 930| 259 | 45433 | 
| 135 | 40 astroid/nodes/node_classes.py | 4673 | 4727| 390 | 45823 | 
| 136 | 40 astroid/exceptions.py | 199 | 214| 136 | 45959 | 
| 137 | 40 astroid/nodes/scoped_nodes.py | 2259 | 2305| 406 | 46365 | 
| 138 | 40 astroid/nodes/node_classes.py | 1044 | 1084| 244 | 46609 | 
| 139 | 40 astroid/raw_building.py | 293 | 303| 290 | 46899 | 
| 140 | 41 astroid/interpreter/dunder_lookup.py | 35 | 32| 333 | 47232 | 
| 141 | 41 astroid/rebuilder.py | 1212 | 1231| 193 | 47425 | 
| 142 | 41 astroid/rebuilder.py | 1622 | 1638| 183 | 47608 | 
| 143 | 41 astroid/brain/brain_typing.py | 409 | 437| 217 | 47825 | 
| 144 | 41 astroid/nodes/node_classes.py | 3155 | 3204| 358 | 48183 | 
| 145 | 41 astroid/interpreter/_import/spec.py | 174 | 192| 150 | 48333 | 
| 146 | 42 astroid/brain/brain_attrs.py | 8 | 36| 204 | 48537 | 
| 147 | 43 astroid/brain/brain_dataclasses.py | 427 | 443| 104 | 48641 | 
| 148 | 43 astroid/objects.py | 282 | 302| 185 | 48826 | 
| 149 | 43 astroid/interpreter/objectmodel.py | 71 | 73| 162 | 48988 | 
| 150 | 43 astroid/rebuilder.py | 1041 | 1054| 155 | 49143 | 
| 151 | 43 astroid/protocols.py | 324 | 380| 478 | 49621 | 
| 152 | 43 astroid/inference.py | 277 | 303| 234 | 49855 | 


### Hint

```
Thank you for opening the issue.
I don't believe `Unknown().as_string()` is ever called regularly. AFAIK it's only used during inference. What should the string representation of an `Unknown` node be? So not sure this needs to be addressed.
Probably just `'Unknown'`.
It's mostly only a problem when we do something like this:

\`\`\`python
inferred = infer(node)
if inferred is not Uninferable:
    if inferred.as_string().contains(some_value):
        ...
\`\`\`
So for the most part, as long as it doesn't crash we're good.
```

## Patch

```diff
diff --git a/astroid/nodes/as_string.py b/astroid/nodes/as_string.py
--- a/astroid/nodes/as_string.py
+++ b/astroid/nodes/as_string.py
@@ -36,6 +36,7 @@
         MatchSingleton,
         MatchStar,
         MatchValue,
+        Unknown,
     )
 
 # pylint: disable=unused-argument
@@ -643,6 +644,9 @@ def visit_property(self, node):
     def visit_evaluatedobject(self, node):
         return node.original.accept(self)
 
+    def visit_unknown(self, node: "Unknown") -> str:
+        return str(node)
+
 
 def _import_string(names):
     """return a list of (name, asname) formatted as a string"""

```

## Test Patch

```diff
diff --git a/tests/unittest_nodes.py b/tests/unittest_nodes.py
--- a/tests/unittest_nodes.py
+++ b/tests/unittest_nodes.py
@@ -306,6 +306,11 @@ def test_f_strings(self):
         ast = abuilder.string_build(code)
         self.assertEqual(ast.as_string().strip(), code.strip())
 
+    @staticmethod
+    def test_as_string_unknown() -> None:
+        assert nodes.Unknown().as_string() == "Unknown.Unknown()"
+        assert nodes.Unknown(lineno=1, col_offset=0).as_string() == "Unknown.Unknown()"
+
 
 class _NodeTest(unittest.TestCase):
     """test transformation of If Node"""

```


## Code snippets

### 1 - astroid/nodes/as_string.py:

Start line: 98, End line: 173

```python
class AsStringVisitor:

    # visit_<node> methods ###########################################

    def visit_await(self, node):
        return f"await {node.value.accept(self)}"

    def visit_asyncwith(self, node):
        return f"async {self.visit_with(node)}"

    def visit_asyncfor(self, node):
        return f"async {self.visit_for(node)}"

    def visit_arguments(self, node):
        """return an astroid.Function node as string"""
        return node.format_args()

    def visit_assignattr(self, node):
        """return an astroid.AssAttr node as string"""
        return self.visit_attribute(node)

    def visit_assert(self, node):
        """return an astroid.Assert node as string"""
        if node.fail:
            return f"assert {node.test.accept(self)}, {node.fail.accept(self)}"
        return f"assert {node.test.accept(self)}"

    def visit_assignname(self, node):
        """return an astroid.AssName node as string"""
        return node.name

    def visit_assign(self, node):
        """return an astroid.Assign node as string"""
        lhs = " = ".join(n.accept(self) for n in node.targets)
        return f"{lhs} = {node.value.accept(self)}"

    def visit_augassign(self, node):
        """return an astroid.AugAssign node as string"""
        return f"{node.target.accept(self)} {node.op} {node.value.accept(self)}"

    def visit_annassign(self, node):
        """Return an astroid.AugAssign node as string"""

        target = node.target.accept(self)
        annotation = node.annotation.accept(self)
        if node.value is None:
            return f"{target}: {annotation}"
        return f"{target}: {annotation} = {node.value.accept(self)}"

    def visit_binop(self, node):
        """return an astroid.BinOp node as string"""
        left = self._precedence_parens(node, node.left)
        right = self._precedence_parens(node, node.right, is_left=False)
        if node.op == "**":
            return f"{left}{node.op}{right}"

        return f"{left} {node.op} {right}"

    def visit_boolop(self, node):
        """return an astroid.BoolOp node as string"""
        values = [f"{self._precedence_parens(node, n)}" for n in node.values]
        return (f" {node.op} ").join(values)

    def visit_break(self, node):
        """return an astroid.Break node as string"""
        return "break"

    def visit_call(self, node):
        """return an astroid.Call node as string"""
        expr_str = self._precedence_parens(node, node.func)
        args = [arg.accept(self) for arg in node.args]
        if node.keywords:
            keywords = [kwarg.accept(self) for kwarg in node.keywords]
        else:
            keywords = []

        args.extend(keywords)
        return f"{expr_str}({', '.join(args)})"
```
### 2 - astroid/nodes/as_string.py:

Start line: 488, End line: 573

```python
class AsStringVisitor:

    def visit_tryexcept(self, node):
        """return an astroid.TryExcept node as string"""
        trys = [f"try:\n{self._stmt_list(node.body)}"]
        for handler in node.handlers:
            trys.append(handler.accept(self))
        if node.orelse:
            trys.append(f"else:\n{self._stmt_list(node.orelse)}")
        return "\n".join(trys)

    def visit_tryfinally(self, node):
        """return an astroid.TryFinally node as string"""
        return "try:\n{}\nfinally:\n{}".format(
            self._stmt_list(node.body), self._stmt_list(node.finalbody)
        )

    def visit_tuple(self, node):
        """return an astroid.Tuple node as string"""
        if len(node.elts) == 1:
            return f"({node.elts[0].accept(self)}, )"
        return f"({', '.join(child.accept(self) for child in node.elts)})"

    def visit_unaryop(self, node):
        """return an astroid.UnaryOp node as string"""
        if node.op == "not":
            operator = "not "
        else:
            operator = node.op
        return f"{operator}{self._precedence_parens(node, node.operand)}"

    def visit_while(self, node):
        """return an astroid.While node as string"""
        whiles = f"while {node.test.accept(self)}:\n{self._stmt_list(node.body)}"
        if node.orelse:
            whiles = f"{whiles}\nelse:\n{self._stmt_list(node.orelse)}"
        return whiles

    def visit_with(self, node):  # 'with' without 'as' is possible
        """return an astroid.With node as string"""
        items = ", ".join(
            f"{expr.accept(self)}" + (v and f" as {v.accept(self)}" or "")
            for expr, v in node.items
        )
        return f"with {items}:\n{self._stmt_list(node.body)}"

    def visit_yield(self, node):
        """yield an ast.Yield node as string"""
        yi_val = (" " + node.value.accept(self)) if node.value else ""
        expr = "yield" + yi_val
        if node.parent.is_statement:
            return expr

        return f"({expr})"

    def visit_yieldfrom(self, node):
        """Return an astroid.YieldFrom node as string."""
        yi_val = (" " + node.value.accept(self)) if node.value else ""
        expr = "yield from" + yi_val
        if node.parent.is_statement:
            return expr

        return f"({expr})"

    def visit_starred(self, node):
        """return Starred node as string"""
        return "*" + node.value.accept(self)

    def visit_match(self, node: "Match") -> str:
        """Return an astroid.Match node as string."""
        return f"match {node.subject.accept(self)}:\n{self._stmt_list(node.cases)}"

    def visit_matchcase(self, node: "MatchCase") -> str:
        """Return an astroid.MatchCase node as string."""
        guard_str = f" if {node.guard.accept(self)}" if node.guard else ""
        return (
            f"case {node.pattern.accept(self)}{guard_str}:\n"
            f"{self._stmt_list(node.body)}"
        )

    def visit_matchvalue(self, node: "MatchValue") -> str:
        """Return an astroid.MatchValue node as string."""
        return node.value.accept(self)

    @staticmethod
    def visit_matchsingleton(node: "MatchSingleton") -> str:
        """Return an astroid.MatchSingleton node as string."""
        return str(node.value)
```
### 3 - astroid/nodes/as_string.py:

Start line: 344, End line: 432

```python
class AsStringVisitor:

    def visit_functiondef(self, node):
        """return an astroid.FunctionDef node as string"""
        return self.handle_functiondef(node, "def")

    def visit_asyncfunctiondef(self, node):
        """return an astroid.AsyncFunction node as string"""
        return self.handle_functiondef(node, "async def")

    def visit_generatorexp(self, node):
        """return an astroid.GeneratorExp node as string"""
        return "({} {})".format(
            node.elt.accept(self), " ".join(n.accept(self) for n in node.generators)
        )

    def visit_attribute(self, node):
        """return an astroid.Getattr node as string"""
        left = self._precedence_parens(node, node.expr)
        if left.isdigit():
            left = f"({left})"
        return f"{left}.{node.attrname}"

    def visit_global(self, node):
        """return an astroid.Global node as string"""
        return f"global {', '.join(node.names)}"

    def visit_if(self, node):
        """return an astroid.If node as string"""
        ifs = [f"if {node.test.accept(self)}:\n{self._stmt_list(node.body)}"]
        if node.has_elif_block():
            ifs.append(f"el{self._stmt_list(node.orelse, indent=False)}")
        elif node.orelse:
            ifs.append(f"else:\n{self._stmt_list(node.orelse)}")
        return "\n".join(ifs)

    def visit_ifexp(self, node):
        """return an astroid.IfExp node as string"""
        return "{} if {} else {}".format(
            self._precedence_parens(node, node.body, is_left=True),
            self._precedence_parens(node, node.test, is_left=True),
            self._precedence_parens(node, node.orelse, is_left=False),
        )

    def visit_import(self, node):
        """return an astroid.Import node as string"""
        return f"import {_import_string(node.names)}"

    def visit_keyword(self, node):
        """return an astroid.Keyword node as string"""
        if node.arg is None:
            return f"**{node.value.accept(self)}"
        return f"{node.arg}={node.value.accept(self)}"

    def visit_lambda(self, node):
        """return an astroid.Lambda node as string"""
        args = node.args.accept(self)
        body = node.body.accept(self)
        if args:
            return f"lambda {args}: {body}"

        return f"lambda: {body}"

    def visit_list(self, node):
        """return an astroid.List node as string"""
        return f"[{', '.join(child.accept(self) for child in node.elts)}]"

    def visit_listcomp(self, node):
        """return an astroid.ListComp node as string"""
        return "[{} {}]".format(
            node.elt.accept(self), " ".join(n.accept(self) for n in node.generators)
        )

    def visit_module(self, node):
        """return an astroid.Module node as string"""
        docs = f'"""{node.doc}"""\n\n' if node.doc else ""
        return docs + "\n".join(n.accept(self) for n in node.body) + "\n\n"

    def visit_name(self, node):
        """return an astroid.Name node as string"""
        return node.name

    def visit_namedexpr(self, node):
        """Return an assignment expression node as string"""
        target = node.target.accept(self)
        value = node.value.accept(self)
        return f"{target} := {value}"

    def visit_nonlocal(self, node):
        """return an astroid.Nonlocal node as string"""
        return f"nonlocal {', '.join(node.names)}"
```
### 4 - astroid/nodes/as_string.py:

Start line: 188, End line: 282

```python
class AsStringVisitor:

    def visit_compare(self, node):
        """return an astroid.Compare node as string"""
        rhs_str = " ".join(
            f"{op} {self._precedence_parens(node, expr, is_left=False)}"
            for op, expr in node.ops
        )
        return f"{self._precedence_parens(node, node.left)} {rhs_str}"

    def visit_comprehension(self, node):
        """return an astroid.Comprehension node as string"""
        ifs = "".join(f" if {n.accept(self)}" for n in node.ifs)
        generated = f"for {node.target.accept(self)} in {node.iter.accept(self)}{ifs}"
        return f"{'async ' if node.is_async else ''}{generated}"

    def visit_const(self, node):
        """return an astroid.Const node as string"""
        if node.value is Ellipsis:
            return "..."
        return repr(node.value)

    def visit_continue(self, node):
        """return an astroid.Continue node as string"""
        return "continue"

    def visit_delete(self, node):  # XXX check if correct
        """return an astroid.Delete node as string"""
        return f"del {', '.join(child.accept(self) for child in node.targets)}"

    def visit_delattr(self, node):
        """return an astroid.DelAttr node as string"""
        return self.visit_attribute(node)

    def visit_delname(self, node):
        """return an astroid.DelName node as string"""
        return node.name

    def visit_decorators(self, node):
        """return an astroid.Decorators node as string"""
        return "@%s\n" % "\n@".join(item.accept(self) for item in node.nodes)

    def visit_dict(self, node):
        """return an astroid.Dict node as string"""
        return "{%s}" % ", ".join(self._visit_dict(node))

    def _visit_dict(self, node):
        for key, value in node.items:
            key = key.accept(self)
            value = value.accept(self)
            if key == "**":
                # It can only be a DictUnpack node.
                yield key + value
            else:
                yield f"{key}: {value}"

    def visit_dictunpack(self, node):
        return "**"

    def visit_dictcomp(self, node):
        """return an astroid.DictComp node as string"""
        return "{{{}: {} {}}}".format(
            node.key.accept(self),
            node.value.accept(self),
            " ".join(n.accept(self) for n in node.generators),
        )

    def visit_expr(self, node):
        """return an astroid.Discard node as string"""
        return node.value.accept(self)

    def visit_emptynode(self, node):
        """dummy method for visiting an Empty node"""
        return ""

    def visit_excepthandler(self, node):
        if node.type:
            if node.name:
                excs = f"except {node.type.accept(self)} as {node.name.accept(self)}"
            else:
                excs = f"except {node.type.accept(self)}"
        else:
            excs = "except"
        return f"{excs}:\n{self._stmt_list(node.body)}"

    def visit_empty(self, node):
        """return an Empty node as string"""
        return ""

    def visit_for(self, node):
        """return an astroid.For node as string"""
        fors = "for {} in {}:\n{}".format(
            node.target.accept(self), node.iter.accept(self), self._stmt_list(node.body)
        )
        if node.orelse:
            fors = f"{fors}\nelse:\n{self._stmt_list(node.orelse)}"
        return fors
```
### 5 - astroid/nodes/as_string.py:

Start line: 434, End line: 474

```python
class AsStringVisitor:

    def visit_pass(self, node):
        """return an astroid.Pass node as string"""
        return "pass"

    def visit_raise(self, node):
        """return an astroid.Raise node as string"""
        if node.exc:
            if node.cause:
                return f"raise {node.exc.accept(self)} from {node.cause.accept(self)}"
            return f"raise {node.exc.accept(self)}"
        return "raise"

    def visit_return(self, node):
        """return an astroid.Return node as string"""
        if node.is_tuple_return() and len(node.value.elts) > 1:
            elts = [child.accept(self) for child in node.value.elts]
            return f"return {', '.join(elts)}"

        if node.value:
            return f"return {node.value.accept(self)}"

        return "return"

    def visit_set(self, node):
        """return an astroid.Set node as string"""
        return "{%s}" % ", ".join(child.accept(self) for child in node.elts)

    def visit_setcomp(self, node):
        """return an astroid.SetComp node as string"""
        return "{{{} {}}}".format(
            node.elt.accept(self), " ".join(n.accept(self) for n in node.generators)
        )

    def visit_slice(self, node):
        """return an astroid.Slice node as string"""
        lower = node.lower.accept(self) if node.lower else ""
        upper = node.upper.accept(self) if node.upper else ""
        step = node.step.accept(self) if node.step else ""
        if step:
            return f"{lower}:{upper}:{step}"
        return f"{lower}:{upper}"
```
### 6 - astroid/nodes/as_string.py:

Start line: 314, End line: 323

```python
class AsStringVisitor:

    def visit_formattedvalue(self, node):
        result = node.value.accept(self)
        if node.conversion and node.conversion >= 0:
            # e.g. if node.conversion == 114: result += "!r"
            result += "!" + chr(node.conversion)
        if node.format_spec:
            # The format spec is itself a JoinedString, i.e. an f-string
            # We strip the f and quotes of the ends
            result += ":" + node.format_spec.accept(self)[2:-1]
        return "{%s}" % result
```
### 7 - astroid/nodes/as_string.py:

Start line: 290, End line: 288

```python
class AsStringVisitor:

    def visit_importfrom(self, node):
        """return an astroid.ImportFrom node as string"""
        return "from {} import {}".format(
            "." * (node.level or 0) + node.modname, _import_string(node.names)
        )

    def visit_joinedstr(self, node):
        string = "".join(
            # Use repr on the string literal parts
            # to get proper escapes, e.g. \n, \\, \"
            # But strip the quotes off the ends
            # (they will always be one character: ' or ")
            repr(value.value)[1:-1]
            # Literal braces must be doubled to escape them
            .replace("{", "{{").replace("}", "}}")
            # Each value in values is either a string literal (Const)
            # or a FormattedValue
            if type(value).__name__ == "Const" else value.accept(self)
            for value in node.values
        )

        # Try to find surrounding quotes that don't appear at all in the string.
        # Because the formatted values inside {} can't contain backslash (\)
        # using a triple quote is sometimes necessary
        for quote in ("'", '"', '"""', "'''"):
            if quote not in string:
                break

        return "f" + quote + string + quote
```
### 8 - astroid/nodes/as_string.py:

Start line: 609, End line: 607

```python
class AsStringVisitor:

    def visit_matchstar(self, node: "MatchStar") -> str:
        """Return an astroid.MatchStar node as string."""
        return f"*{node.name.accept(self) if node.name else '_'}"

    def visit_matchas(self, node: "MatchAs") -> str:
        """Return an astroid.MatchAs node as string."""
        # pylint: disable=import-outside-toplevel
        # Prevent circular dependency
        from astroid.nodes.node_classes import MatchClass, MatchMapping, MatchSequence

        if isinstance(node.parent, (MatchSequence, MatchMapping, MatchClass)):
            return node.name.accept(self) if node.name else "_"
        return (
            f"{node.pattern.accept(self) if node.pattern else '_'}"
            f"{f' as {node.name.accept(self)}' if node.name else ''}"
        )
```
### 9 - astroid/nodes/as_string.py:

Start line: 45, End line: 42

```python
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from astroid.nodes.node_classes import (
        Match,
        MatchAs,
        MatchCase,
        MatchClass,
        MatchMapping,
        MatchOr,
        MatchSequence,
        MatchSingleton,
        MatchStar,
        MatchValue,
    )

# pylint: disable=unused-argument

DOC_NEWLINE = "\0"


# Visitor pattern require argument all the time and is not better with staticmethod
# noinspection PyUnusedLocal,PyMethodMayBeStatic
class AsStringVisitor:
    """Visitor to render an Astroid node as a valid python code string"""

    def __init__(self, indent="    "):
        self.indent = indent

    def __call__(self, node):
        """Makes this visitor behave as a simple function"""
        return node.accept(self).replace(DOC_NEWLINE, "\n")

    def _docs_dedent(self, doc):
        """Stop newlines in docs being indented by self._stmt_list"""
        return '\n{}"""{}"""'.format(self.indent, doc.replace("\n", DOC_NEWLINE))

    def _stmt_list(self, stmts, indent=True):
        """return a list of nodes to string"""
        stmts = "\n".join(nstr for nstr in [n.accept(self) for n in stmts] if nstr)
        if indent:
            return self.indent + stmts.replace("\n", "\n" + self.indent)

        return stmts

    def _precedence_parens(self, node, child, is_left=True):
        """Wrap child in parens only if required to keep same semantics"""
        if self._should_wrap(node, child, is_left):
            return f"({child.accept(self)})"

        return child.accept(self)
```
### 10 - astroid/nodes/as_string.py:

Start line: 622, End line: 659

```python
class AsStringVisitor:

    def visit_matchor(self, node: "MatchOr") -> str:
        """Return an astroid.MatchOr node as string."""
        if node.patterns is None:
            raise Exception(f"{node} does not have pattern nodes")
        return " | ".join(p.accept(self) for p in node.patterns)

    # These aren't for real AST nodes, but for inference objects.

    def visit_frozenset(self, node):
        return node.parent.accept(self)

    def visit_super(self, node):
        return node.parent.accept(self)

    def visit_uninferable(self, node):
        return str(node)

    def visit_property(self, node):
        return node.function.accept(self)

    def visit_evaluatedobject(self, node):
        return node.original.accept(self)


def _import_string(names):
    """return a list of (name, asname) formatted as a string"""
    _names = []
    for name, asname in names:
        if asname is not None:
            _names.append(f"{name} as {asname}")
        else:
            _names.append(name)
    return ", ".join(_names)


# This sets the default indent to 4 spaces.
to_code = AsStringVisitor("    ")
```
### 13 - astroid/nodes/as_string.py:

Start line: 593, End line: 603

```python
class AsStringVisitor:

    def visit_matchclass(self, node: "MatchClass") -> str:
        """Return an astroid.MatchClass node as string."""
        if node.cls is None:
            raise Exception(f"{node} does not have a 'cls' node")
        class_strings: List[str] = []
        if node.patterns:
            class_strings.extend(p.accept(self) for p in node.patterns)
        if node.kwd_attrs and node.kwd_patterns:
            for attr, pattern in zip(node.kwd_attrs, node.kwd_patterns):
                class_strings.append(f"{attr}={pattern.accept(self)}")
        return f"{node.cls.accept(self)}({', '.join(class_strings)})"
```
### 14 - astroid/nodes/as_string.py:

Start line: 581, End line: 579

```python
class AsStringVisitor:

    def visit_matchsequence(self, node: "MatchSequence") -> str:
        """Return an astroid.MatchSequence node as string."""
        if node.patterns is None:
            return "[]"
        return f"[{', '.join(p.accept(self) for p in node.patterns)}]"

    def visit_matchmapping(self, node: "MatchMapping") -> str:
        """Return an astroid.MatchMapping node as string."""
        mapping_strings: List[str] = []
        if node.keys and node.patterns:
            mapping_strings.extend(
                f"{key.accept(self)}: {p.accept(self)}"
                for key, p in zip(node.keys, node.patterns)
            )
        if node.rest:
            mapping_strings.append(f"**{node.rest.accept(self)}")
        return f"{'{'}{', '.join(mapping_strings)}{'}'}"
```
### 15 - astroid/nodes/as_string.py:

Start line: 476, End line: 486

```python
class AsStringVisitor:

    def visit_subscript(self, node):
        """return an astroid.Subscript node as string"""
        idx = node.slice
        if idx.__class__.__name__.lower() == "index":
            idx = idx.value
        idxstr = idx.accept(self)
        if idx.__class__.__name__.lower() == "tuple" and idx.elts:
            # Remove parenthesis in tuple and extended slice.
            # a[(::1, 1:)] is not valid syntax.
            idxstr = idxstr[1:-1]
        return f"{self._precedence_parens(node, node.value)}[{idxstr}]"
```
### 20 - astroid/nodes/as_string.py:

Start line: 175, End line: 186

```python
class AsStringVisitor:

    def visit_classdef(self, node):
        """return an astroid.ClassDef node as string"""
        decorate = node.decorators.accept(self) if node.decorators else ""
        args = [n.accept(self) for n in node.bases]
        if node._metaclass and not node.has_metaclass_hack():
            args.append("metaclass=" + node._metaclass.accept(self))
        args += [n.accept(self) for n in node.keywords]
        args = f"({', '.join(args)})" if args else ""
        docs = self._docs_dedent(node.doc) if node.doc else ""
        return "\n\n{}class {}{}:{}\n{}\n".format(
            decorate, node.name, args, docs, self._stmt_list(node.body)
        )
```
### 21 - astroid/nodes/as_string.py:

Start line: 325, End line: 342

```python
class AsStringVisitor:

    def handle_functiondef(self, node, keyword):
        """return a (possibly async) function definition node as string"""
        decorate = node.decorators.accept(self) if node.decorators else ""
        docs = self._docs_dedent(node.doc) if node.doc else ""
        trailer = ":"
        if node.returns:
            return_annotation = " -> " + node.returns.as_string()
            trailer = return_annotation + ":"
        def_format = "\n%s%s %s(%s)%s%s\n%s"
        return def_format % (
            decorate,
            keyword,
            node.name,
            node.args.accept(self),
            trailer,
            docs,
            self._stmt_list(node.body),
        )
```
### 33 - astroid/nodes/as_string.py:

Start line: 76, End line: 96

```python
class AsStringVisitor:

    def _should_wrap(self, node, child, is_left):
        """Wrap child if:
        - it has lower precedence
        - same precedence with position opposite to associativity direction
        """
        node_precedence = node.op_precedence()
        child_precedence = child.op_precedence()

        if node_precedence > child_precedence:
            # 3 * (4 + 5)
            return True

        if (
            node_precedence == child_precedence
            and is_left != node.op_left_associative()
        ):
            # 3 - (4 - 5)
            # (2**3)**4
            return True

        return False
```
