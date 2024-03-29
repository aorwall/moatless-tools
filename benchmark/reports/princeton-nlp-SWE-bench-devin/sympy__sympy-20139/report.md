# sympy__sympy-20139

| **sympy/sympy** | `3449cecacb1938d47ce2eb628a812e4ecf6702f1` |
| ---- | ---- |
| **No of patches** | 3 |
| **All found context length** | 14411 |
| **Any found context length** | 362 |
| **Avg pos** | 18.0 |
| **Min pos** | 1 |
| **Max pos** | 27 |
| **Top file pos** | 1 |
| **Missing snippets** | 6 |
| **Missing patch files** | 1 |


## Expected patch

```diff
diff --git a/sympy/core/symbol.py b/sympy/core/symbol.py
--- a/sympy/core/symbol.py
+++ b/sympy/core/symbol.py
@@ -42,6 +42,7 @@ def __getnewargs__(self):
     def _hashable_content(self):
         return (self.name,)
 
+
 def _filter_assumptions(kwargs):
     """Split the given dict into assumptions and non-assumptions.
     Keys are taken as assumptions if they correspond to an
diff --git a/sympy/matrices/expressions/matexpr.py b/sympy/matrices/expressions/matexpr.py
--- a/sympy/matrices/expressions/matexpr.py
+++ b/sympy/matrices/expressions/matexpr.py
@@ -6,6 +6,7 @@
 from sympy.core import S, Symbol, Integer, Basic, Expr, Mul, Add
 from sympy.core.decorators import call_highest_priority
 from sympy.core.compatibility import SYMPY_INTS, default_sort_key
+from sympy.core.symbol import Str
 from sympy.core.sympify import SympifyError, _sympify
 from sympy.functions import conjugate, adjoint
 from sympy.functions.special.tensor_functions import KroneckerDelta
@@ -772,7 +773,7 @@ def __new__(cls, name, n, m):
         cls._check_dim(n)
 
         if isinstance(name, str):
-            name = Symbol(name)
+            name = Str(name)
         obj = Basic.__new__(cls, name, n, m)
         return obj
 
diff --git a/sympy/printing/dot.py b/sympy/printing/dot.py
--- a/sympy/printing/dot.py
+++ b/sympy/printing/dot.py
@@ -35,6 +35,7 @@ def purestr(x, with_args=False):
 
     >>> from sympy import Float, Symbol, MatrixSymbol
     >>> from sympy import Integer # noqa: F401
+    >>> from sympy.core.symbol import Str # noqa: F401
     >>> from sympy.printing.dot import purestr
 
     Applying ``purestr`` for basic symbolic object:
@@ -51,7 +52,7 @@ def purestr(x, with_args=False):
     For matrix symbol:
     >>> code = purestr(MatrixSymbol('x', 2, 2))
     >>> code
-    "MatrixSymbol(Symbol('x'), Integer(2), Integer(2))"
+    "MatrixSymbol(Str('x'), Integer(2), Integer(2))"
     >>> eval(code) == MatrixSymbol('x', 2, 2)
     True
 
@@ -59,8 +60,8 @@ def purestr(x, with_args=False):
     >>> purestr(Float(2), with_args=True)
     ("Float('2.0', precision=53)", ())
     >>> purestr(MatrixSymbol('x', 2, 2), with_args=True)
-    ("MatrixSymbol(Symbol('x'), Integer(2), Integer(2))",
-     ("Symbol('x')", 'Integer(2)', 'Integer(2)'))
+    ("MatrixSymbol(Str('x'), Integer(2), Integer(2))",
+     ("Str('x')", 'Integer(2)', 'Integer(2)'))
     """
     sargs = ()
     if not isinstance(x, Basic):

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sympy/core/symbol.py | 45 | 45 | 26 | 8 | 14411
| sympy/matrices/expressions/matexpr.py | 9 | 9 | 27 | 1 | 14653
| sympy/matrices/expressions/matexpr.py | 775 | 775 | 1 | 1 | 362
| sympy/printing/dot.py | 38 | 38 | - | - | -
| sympy/printing/dot.py | 54 | 54 | - | - | -
| sympy/printing/dot.py | 62 | 63 | - | - | -


## Problem Statement

```
Use Str instead of Symbol for name of MatrixSymbol
<!-- Your title above should be a short description of what
was changed. Do not include the issue number in the title. -->

#### References to other Issues or PRs
<!-- If this pull request fixes an issue, write "Fixes #NNNN" in that exact
format, e.g. "Fixes #1234" (see
https://tinyurl.com/auto-closing for more information). Also, please
write a comment on that issue linking back to this pull request once it is
open. -->


#### Brief description of what is fixed or changed


#### Other comments


#### Release Notes

<!-- Write the release notes for this release below. See
https://github.com/sympy/sympy/wiki/Writing-Release-Notes for more information
on how to write release notes. The bot will check your release notes
automatically to see if they are formatted correctly. -->

<!-- BEGIN RELEASE NOTES -->
- matrices
  - `MatrixSymbol` will store Str in its first argument.
<!-- END RELEASE NOTES -->

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 sympy/matrices/expressions/matexpr.py** | 747 | 799| 362 | 362 | 7519 | 
| 2 | 2 sympy/matrices/matrices.py | 801 | 815| 197 | 559 | 26374 | 
| 3 | 2 sympy/matrices/matrices.py | 2158 | 2201| 575 | 1134 | 26374 | 
| 4 | 3 sympy/printing/pretty/pretty_symbology.py | 107 | 183| 759 | 1893 | 32326 | 
| 5 | 4 sympy/printing/str.py | 196 | 230| 382 | 2275 | 39957 | 
| 6 | 5 sympy/matrices/sparse.py | 674 | 700| 291 | 2566 | 49154 | 
| 7 | **5 sympy/matrices/expressions/matexpr.py** | 801 | 817| 177 | 2743 | 49154 | 
| 8 | 5 sympy/matrices/matrices.py | 1 | 61| 660 | 3403 | 49154 | 
| 9 | 5 sympy/printing/str.py | 232 | 247| 161 | 3564 | 49154 | 
| 10 | 5 sympy/matrices/matrices.py | 1994 | 2070| 697 | 4261 | 49154 | 
| 11 | **5 sympy/matrices/expressions/matexpr.py** | 34 | 138| 788 | 5049 | 49154 | 
| 12 | 5 sympy/printing/str.py | 734 | 813| 650 | 5699 | 49154 | 
| 13 | 6 sympy/abc.py | 74 | 114| 441 | 6140 | 50323 | 
| 14 | 7 sympy/matrices/__init__.py | 1 | 68| 648 | 6788 | 50971 | 
| 15 | **8 sympy/core/symbol.py** | 285 | 342| 446 | 7234 | 57788 | 
| 16 | 9 sympy/printing/pretty/pretty.py | 787 | 806| 207 | 7441 | 81938 | 
| 17 | 10 sympy/matrices/expressions/blockmatrix.py | 1 | 21| 250 | 7691 | 87825 | 
| 18 | 10 sympy/matrices/matrices.py | 2072 | 2156| 791 | 8482 | 87825 | 
| 19 | 11 sympy/physics/mechanics/system.py | 9 | 207| 1886 | 10368 | 91806 | 
| 20 | 12 sympy/parsing/autolev/_listener_autolev_antlr.py | 996 | 1089| 1226 | 11594 | 114927 | 
| 21 | 12 sympy/matrices/matrices.py | 652 | 690| 285 | 11879 | 114927 | 
| 22 | 12 sympy/matrices/sparse.py | 1 | 22| 166 | 12045 | 114927 | 
| 23 | 13 sympy/__init__.py | 1 | 70| 687 | 12732 | 124344 | 
| 24 | 14 sympy/matrices/common.py | 1 | 53| 334 | 13066 | 147202 | 
| 25 | **14 sympy/core/symbol.py** | 534 | 649| 1075 | 14141 | 147202 | 
| **-> 26 <-** | **14 sympy/core/symbol.py** | 18 | 54| 270 | 14411 | 147202 | 
| **-> 27 <-** | **14 sympy/matrices/expressions/matexpr.py** | 1 | 31| 242 | 14653 | 147202 | 
| 28 | 14 sympy/abc.py | 1 | 73| 727 | 15380 | 147202 | 
| 29 | 15 sympy/matrices/expressions/matadd.py | 1 | 14| 138 | 15518 | 148329 | 
| 30 | 16 sympy/printing/rust.py | 435 | 461| 243 | 15761 | 153727 | 
| 31 | 16 sympy/matrices/matrices.py | 1897 | 1942| 462 | 16223 | 153727 | 
| 32 | **16 sympy/matrices/expressions/matexpr.py** | 685 | 715| 246 | 16469 | 153727 | 
| 33 | 16 sympy/printing/pretty/pretty.py | 809 | 834| 262 | 16731 | 153727 | 
| 34 | 16 sympy/matrices/common.py | 1210 | 1292| 715 | 17446 | 153727 | 
| 35 | 17 sympy/matrices/expressions/__init__.py | 1 | 61| 435 | 17881 | 154162 | 
| 36 | 17 sympy/__init__.py | 180 | 224| 668 | 18549 | 154162 | 
| 37 | 17 sympy/printing/pretty/pretty_symbology.py | 552 | 585| 301 | 18850 | 154162 | 
| 38 | 17 sympy/matrices/matrices.py | 1364 | 1388| 296 | 19146 | 154162 | 
| 39 | 17 sympy/matrices/common.py | 153 | 218| 496 | 19642 | 154162 | 
| 40 | 18 examples/intermediate/vandermonde.py | 1 | 38| 225 | 19867 | 155527 | 
| 41 | 18 sympy/printing/pretty/pretty_symbology.py | 184 | 215| 314 | 20181 | 155527 | 
| 42 | 19 sympy/matrices/immutable.py | 1 | 26| 183 | 20364 | 157013 | 
| 43 | 19 sympy/printing/pretty/pretty.py | 761 | 784| 261 | 20625 | 157013 | 
| 44 | 19 sympy/printing/str.py | 396 | 455| 570 | 21195 | 157013 | 
| 45 | 20 sympy/printing/mathml.py | 1003 | 1036| 286 | 21481 | 173836 | 
| 46 | **20 sympy/core/symbol.py** | 261 | 283| 287 | 21768 | 173836 | 
| 47 | 21 sympy/core/backend.py | 1 | 34| 560 | 22328 | 174396 | 
| 48 | 22 sympy/stats/matrix_distributions.py | 1 | 8| 105 | 22433 | 178099 | 
| 49 | 23 bin/authors_update.py | 71 | 141| 773 | 23206 | 179632 | 
| 50 | 24 sympy/stats/stochastic_process_types.py | 103 | 133| 200 | 23406 | 194257 | 
| 51 | 24 sympy/matrices/matrices.py | 362 | 429| 683 | 24089 | 194257 | 
| 52 | 24 sympy/parsing/autolev/_listener_autolev_antlr.py | 721 | 862| 1654 | 25743 | 194257 | 
| 53 | 24 sympy/printing/pretty/pretty_symbology.py | 257 | 297| 603 | 26346 | 194257 | 
| 54 | 24 sympy/physics/mechanics/system.py | 209 | 309| 838 | 27184 | 194257 | 
| 55 | 24 sympy/matrices/common.py | 1088 | 1123| 341 | 27525 | 194257 | 
| 56 | 25 sympy/matrices/expressions/matmul.py | 1 | 15| 140 | 27665 | 197677 | 
| 57 | 25 sympy/physics/mechanics/system.py | 1 | 445| 54 | 27719 | 197677 | 


## Missing Patch Files

 * 1: sympy/core/symbol.py
 * 2: sympy/matrices/expressions/matexpr.py
 * 3: sympy/printing/dot.py

### Hint

```
:white_check_mark:

Hi, I am the [SymPy bot](https://github.com/sympy/sympy-bot) (v160). I'm here to help you write a release notes entry. Please read the [guide on how to write release notes](https://github.com/sympy/sympy/wiki/Writing-Release-Notes).



Your release notes are in good order.

Here is what the release notes will look like:
* matrices
  - `MatrixSymbol` will store Str in its first argument. ([#19715](https://github.com/sympy/sympy/pull/19715) by [@sylee957](https://github.com/sylee957))

This will be added to https://github.com/sympy/sympy/wiki/Release-Notes-for-1.7.

Note: This comment will be updated with the latest check if you edit the pull request. You need to reload the page to see it. <details><summary>Click here to see the pull request description that was parsed.</summary>

    <!-- Your title above should be a short description of what
    was changed. Do not include the issue number in the title. -->

    #### References to other Issues or PRs
    <!-- If this pull request fixes an issue, write "Fixes #NNNN" in that exact
    format, e.g. "Fixes #1234" (see
    https://tinyurl.com/auto-closing for more information). Also, please
    write a comment on that issue linking back to this pull request once it is
    open. -->


    #### Brief description of what is fixed or changed


    #### Other comments


    #### Release Notes

    <!-- Write the release notes for this release below. See
    https://github.com/sympy/sympy/wiki/Writing-Release-Notes for more information
    on how to write release notes. The bot will check your release notes
    automatically to see if they are formatted correctly. -->

    <!-- BEGIN RELEASE NOTES -->
    - matrices
      - `MatrixSymbol` will store Str in its first argument.
    <!-- END RELEASE NOTES -->

</details><p>

I missed the introduction of `Str`. I don't see anything in the release notes about it. 
@sylee957 Any news on this?
This needs progress in #19841 to resolve the failing tests
```

## Patch

```diff
diff --git a/sympy/core/symbol.py b/sympy/core/symbol.py
--- a/sympy/core/symbol.py
+++ b/sympy/core/symbol.py
@@ -42,6 +42,7 @@ def __getnewargs__(self):
     def _hashable_content(self):
         return (self.name,)
 
+
 def _filter_assumptions(kwargs):
     """Split the given dict into assumptions and non-assumptions.
     Keys are taken as assumptions if they correspond to an
diff --git a/sympy/matrices/expressions/matexpr.py b/sympy/matrices/expressions/matexpr.py
--- a/sympy/matrices/expressions/matexpr.py
+++ b/sympy/matrices/expressions/matexpr.py
@@ -6,6 +6,7 @@
 from sympy.core import S, Symbol, Integer, Basic, Expr, Mul, Add
 from sympy.core.decorators import call_highest_priority
 from sympy.core.compatibility import SYMPY_INTS, default_sort_key
+from sympy.core.symbol import Str
 from sympy.core.sympify import SympifyError, _sympify
 from sympy.functions import conjugate, adjoint
 from sympy.functions.special.tensor_functions import KroneckerDelta
@@ -772,7 +773,7 @@ def __new__(cls, name, n, m):
         cls._check_dim(n)
 
         if isinstance(name, str):
-            name = Symbol(name)
+            name = Str(name)
         obj = Basic.__new__(cls, name, n, m)
         return obj
 
diff --git a/sympy/printing/dot.py b/sympy/printing/dot.py
--- a/sympy/printing/dot.py
+++ b/sympy/printing/dot.py
@@ -35,6 +35,7 @@ def purestr(x, with_args=False):
 
     >>> from sympy import Float, Symbol, MatrixSymbol
     >>> from sympy import Integer # noqa: F401
+    >>> from sympy.core.symbol import Str # noqa: F401
     >>> from sympy.printing.dot import purestr
 
     Applying ``purestr`` for basic symbolic object:
@@ -51,7 +52,7 @@ def purestr(x, with_args=False):
     For matrix symbol:
     >>> code = purestr(MatrixSymbol('x', 2, 2))
     >>> code
-    "MatrixSymbol(Symbol('x'), Integer(2), Integer(2))"
+    "MatrixSymbol(Str('x'), Integer(2), Integer(2))"
     >>> eval(code) == MatrixSymbol('x', 2, 2)
     True
 
@@ -59,8 +60,8 @@ def purestr(x, with_args=False):
     >>> purestr(Float(2), with_args=True)
     ("Float('2.0', precision=53)", ())
     >>> purestr(MatrixSymbol('x', 2, 2), with_args=True)
-    ("MatrixSymbol(Symbol('x'), Integer(2), Integer(2))",
-     ("Symbol('x')", 'Integer(2)', 'Integer(2)'))
+    ("MatrixSymbol(Str('x'), Integer(2), Integer(2))",
+     ("Str('x')", 'Integer(2)', 'Integer(2)'))
     """
     sargs = ()
     if not isinstance(x, Basic):

```

## Test Patch

```diff
diff --git a/sympy/printing/tests/test_dot.py b/sympy/printing/tests/test_dot.py
--- a/sympy/printing/tests/test_dot.py
+++ b/sympy/printing/tests/test_dot.py
@@ -101,8 +101,8 @@ def test_Matrix_and_non_basics():
 # Nodes #
 #########
 
-"MatrixSymbol(Symbol('X'), Symbol('n'), Symbol('n'))_()" ["color"="black", "label"="MatrixSymbol", "shape"="ellipse"];
-"Symbol('X')_(0,)" ["color"="black", "label"="X", "shape"="ellipse"];
+"MatrixSymbol(Str('X'), Symbol('n'), Symbol('n'))_()" ["color"="black", "label"="MatrixSymbol", "shape"="ellipse"];
+"Str('X')_(0,)" ["color"="blue", "label"="X", "shape"="ellipse"];
 "Symbol('n')_(1,)" ["color"="black", "label"="n", "shape"="ellipse"];
 "Symbol('n')_(2,)" ["color"="black", "label"="n", "shape"="ellipse"];
 
@@ -110,9 +110,9 @@ def test_Matrix_and_non_basics():
 # Edges #
 #########
 
-"MatrixSymbol(Symbol('X'), Symbol('n'), Symbol('n'))_()" -> "Symbol('X')_(0,)";
-"MatrixSymbol(Symbol('X'), Symbol('n'), Symbol('n'))_()" -> "Symbol('n')_(1,)";
-"MatrixSymbol(Symbol('X'), Symbol('n'), Symbol('n'))_()" -> "Symbol('n')_(2,)";
+"MatrixSymbol(Str('X'), Symbol('n'), Symbol('n'))_()" -> "Str('X')_(0,)";
+"MatrixSymbol(Str('X'), Symbol('n'), Symbol('n'))_()" -> "Symbol('n')_(1,)";
+"MatrixSymbol(Str('X'), Symbol('n'), Symbol('n'))_()" -> "Symbol('n')_(2,)";
 }"""
 
 
diff --git a/sympy/printing/tests/test_repr.py b/sympy/printing/tests/test_repr.py
--- a/sympy/printing/tests/test_repr.py
+++ b/sympy/printing/tests/test_repr.py
@@ -6,6 +6,7 @@
     sqrt, root, AlgebraicNumber, Symbol, Dummy, Wild, MatrixSymbol)
 from sympy.combinatorics import Cycle, Permutation
 from sympy.core.compatibility import exec_
+from sympy.core.symbol import Str
 from sympy.geometry import Point, Ellipse
 from sympy.printing import srepr
 from sympy.polys import ring, field, ZZ, QQ, lex, grlex, Poly
@@ -16,7 +17,7 @@
 
 # eval(srepr(expr)) == expr has to succeed in the right environment. The right
 # environment is the scope of "from sympy import *" for most cases.
-ENV = {}  # type: Dict[str, Any]
+ENV = {"Str": Str}  # type: Dict[str, Any]
 exec_("from sympy import *", ENV)
 
 
@@ -295,9 +296,9 @@ def test_matrix_expressions():
     n = symbols('n', integer=True)
     A = MatrixSymbol("A", n, n)
     B = MatrixSymbol("B", n, n)
-    sT(A, "MatrixSymbol(Symbol('A'), Symbol('n', integer=True), Symbol('n', integer=True))")
-    sT(A*B, "MatMul(MatrixSymbol(Symbol('A'), Symbol('n', integer=True), Symbol('n', integer=True)), MatrixSymbol(Symbol('B'), Symbol('n', integer=True), Symbol('n', integer=True)))")
-    sT(A + B, "MatAdd(MatrixSymbol(Symbol('A'), Symbol('n', integer=True), Symbol('n', integer=True)), MatrixSymbol(Symbol('B'), Symbol('n', integer=True), Symbol('n', integer=True)))")
+    sT(A, "MatrixSymbol(Str('A'), Symbol('n', integer=True), Symbol('n', integer=True))")
+    sT(A*B, "MatMul(MatrixSymbol(Str('A'), Symbol('n', integer=True), Symbol('n', integer=True)), MatrixSymbol(Str('B'), Symbol('n', integer=True), Symbol('n', integer=True)))")
+    sT(A + B, "MatAdd(MatrixSymbol(Str('A'), Symbol('n', integer=True), Symbol('n', integer=True)), MatrixSymbol(Str('B'), Symbol('n', integer=True), Symbol('n', integer=True)))")
 
 
 def test_Cycle():
diff --git a/sympy/printing/tests/test_tree.py b/sympy/printing/tests/test_tree.py
--- a/sympy/printing/tests/test_tree.py
+++ b/sympy/printing/tests/test_tree.py
@@ -184,11 +184,11 @@ def test_print_tree_MatAdd_noassumptions():
     test_str = \
 """MatAdd: A + B
 +-MatrixSymbol: A
-| +-Symbol: A
+| +-Str: A
 | +-Integer: 3
 | +-Integer: 3
 +-MatrixSymbol: B
-  +-Symbol: B
+  +-Str: B
   +-Integer: 3
   +-Integer: 3
 """
diff --git a/sympy/simplify/tests/test_powsimp.py b/sympy/simplify/tests/test_powsimp.py
--- a/sympy/simplify/tests/test_powsimp.py
+++ b/sympy/simplify/tests/test_powsimp.py
@@ -2,6 +2,7 @@
     symbols, powsimp, MatrixSymbol, sqrt, pi, Mul, gamma, Function,
     S, I, exp, simplify, sin, E, log, hyper, Symbol, Dummy, powdenest, root,
     Rational, oo, signsimp)
+from sympy.core.symbol import Str
 
 from sympy.abc import x, y, z, a, b
 
@@ -227,7 +228,7 @@ def test_issue_9324_powsimp_on_matrix_symbol():
     M = MatrixSymbol('M', 10, 10)
     expr = powsimp(M, deep=True)
     assert expr == M
-    assert expr.args[0] == Symbol('M')
+    assert expr.args[0] == Str('M')
 
 
 def test_issue_6367():
diff --git a/sympy/unify/tests/test_sympy.py b/sympy/unify/tests/test_sympy.py
--- a/sympy/unify/tests/test_sympy.py
+++ b/sympy/unify/tests/test_sympy.py
@@ -1,4 +1,5 @@
 from sympy import Add, Basic, symbols, Symbol, And
+from sympy.core.symbol import Str
 from sympy.unify.core import Compound, Variable
 from sympy.unify.usympy import (deconstruct, construct, unify, is_associative,
         is_commutative)
@@ -100,8 +101,8 @@ def test_matrix():
     X = MatrixSymbol('X', n, n)
     Y = MatrixSymbol('Y', 2, 2)
     Z = MatrixSymbol('Z', 2, 3)
-    assert list(unify(X, Y, {}, variables=[n, Symbol('X')])) == [{Symbol('X'): Symbol('Y'), n: 2}]
-    assert list(unify(X, Z, {}, variables=[n, Symbol('X')])) == []
+    assert list(unify(X, Y, {}, variables=[n, Str('X')])) == [{Str('X'): Str('Y'), n: 2}]
+    assert list(unify(X, Z, {}, variables=[n, Str('X')])) == []
 
 def test_non_frankenAdds():
     # the is_commutative property used to fail because of Basic.__new__

```


## Code snippets

### 1 - sympy/matrices/expressions/matexpr.py:

Start line: 747, End line: 799

```python
class MatrixSymbol(MatrixExpr):
    """Symbolic representation of a Matrix object

    Creates a SymPy Symbol to represent a Matrix. This matrix has a shape and
    can be included in Matrix Expressions

    Examples
    ========

    >>> from sympy import MatrixSymbol, Identity
    >>> A = MatrixSymbol('A', 3, 4) # A 3 by 4 Matrix
    >>> B = MatrixSymbol('B', 4, 3) # A 4 by 3 Matrix
    >>> A.shape
    (3, 4)
    >>> 2*A*B + Identity(3)
    I + 2*A*B
    """
    is_commutative = False
    is_symbol = True
    _diff_wrt = True

    def __new__(cls, name, n, m):
        n, m = _sympify(n), _sympify(m)

        cls._check_dim(m)
        cls._check_dim(n)

        if isinstance(name, str):
            name = Symbol(name)
        obj = Basic.__new__(cls, name, n, m)
        return obj

    @property
    def shape(self):
        return self.args[1], self.args[2]

    @property
    def name(self):
        return self.args[0].name

    def _entry(self, i, j, **kwargs):
        return MatrixElement(self, i, j)

    @property
    def free_symbols(self):
        return {self}

    def _eval_simplify(self, **kwargs):
        return self

    def _eval_derivative(self, x):
        # x is a scalar:
        return ZeroMatrix(self.shape[0], self.shape[1])
```
### 2 - sympy/matrices/matrices.py:

Start line: 801, End line: 815

```python
class MatrixBase(MatrixDeprecated,
                 MatrixCalculus,
                 MatrixEigen,
                 MatrixCommon,
                 Printable):

    def __str__(self):
        if self.rows == 0 or self.cols == 0:
            return 'Matrix(%s, %s, [])' % (self.rows, self.cols)
        return "Matrix(%s)" % str(self.tolist())

    def _format_str(self, printer=None):
        if not printer:
            from sympy.printing.str import StrPrinter
            printer = StrPrinter()
        # Handle zero dimensions:
        if self.rows == 0 or self.cols == 0:
            return 'Matrix(%s, %s, [])' % (self.rows, self.cols)
        if self.rows == 1:
            return "Matrix([%s])" % self.table(printer, rowsep=',\n')
        return "Matrix([\n%s])" % self.table(printer, rowsep=',\n')
```
### 3 - sympy/matrices/matrices.py:

Start line: 2158, End line: 2201

```python
class MatrixBase(MatrixDeprecated,
                 MatrixCalculus,
                 MatrixEigen,
                 MatrixCommon,
                 Printable):

    def inv(self, method=None, iszerofunc=_iszero, try_block_diag=False):
        return _inv(self, method=method, iszerofunc=iszerofunc,
                try_block_diag=try_block_diag)

    def connected_components(self):
        return _connected_components(self)

    def connected_components_decomposition(self):
        return _connected_components_decomposition(self)

    rank_decomposition.__doc__     = _rank_decomposition.__doc__
    cholesky.__doc__               = _cholesky.__doc__
    LDLdecomposition.__doc__       = _LDLdecomposition.__doc__
    LUdecomposition.__doc__        = _LUdecomposition.__doc__
    LUdecomposition_Simple.__doc__ = _LUdecomposition_Simple.__doc__
    LUdecompositionFF.__doc__      = _LUdecompositionFF.__doc__
    QRdecomposition.__doc__        = _QRdecomposition.__doc__

    diagonal_solve.__doc__         = _diagonal_solve.__doc__
    lower_triangular_solve.__doc__ = _lower_triangular_solve.__doc__
    upper_triangular_solve.__doc__ = _upper_triangular_solve.__doc__
    cholesky_solve.__doc__         = _cholesky_solve.__doc__
    LDLsolve.__doc__               = _LDLsolve.__doc__
    LUsolve.__doc__                = _LUsolve.__doc__
    QRsolve.__doc__                = _QRsolve.__doc__
    gauss_jordan_solve.__doc__     = _gauss_jordan_solve.__doc__
    pinv_solve.__doc__             = _pinv_solve.__doc__
    solve.__doc__                  = _solve.__doc__
    solve_least_squares.__doc__    = _solve_least_squares.__doc__

    pinv.__doc__                   = _pinv.__doc__
    inv_mod.__doc__                = _inv_mod.__doc__
    inverse_ADJ.__doc__            = _inv_ADJ.__doc__
    inverse_GE.__doc__             = _inv_GE.__doc__
    inverse_LU.__doc__             = _inv_LU.__doc__
    inverse_CH.__doc__             = _inv_CH.__doc__
    inverse_LDL.__doc__            = _inv_LDL.__doc__
    inverse_QR.__doc__             = _inv_QR.__doc__
    inverse_BLOCK.__doc__          = _inv_block.__doc__
    inv.__doc__                    = _inv.__doc__

    connected_components.__doc__   = _connected_components.__doc__
    connected_components_decomposition.__doc__ = \
        _connected_components_decomposition.__doc__
```
### 4 - sympy/printing/pretty/pretty_symbology.py:

Start line: 107, End line: 183

```python
def xstr(*args):
    SymPyDeprecationWarning(
        feature="``xstr`` function",
        useinstead="``str``",
        deprecated_since_version="1.7").warn()
    return str(*args)

# GREEK
g = lambda l: U('GREEK SMALL LETTER %s' % l.upper())
G = lambda l: U('GREEK CAPITAL LETTER %s' % l.upper())

greek_letters = list(greeks) # make a copy
# deal with Unicode's funny spelling of lambda
greek_letters[greek_letters.index('lambda')] = 'lamda'

# {}  greek letter -> (g,G)
greek_unicode = dict((L, g(L)) for L in greek_letters)
greek_unicode.update((L[0].upper() + L[1:], G(L)) for L in greek_letters)

# aliases
greek_unicode['lambda'] = greek_unicode['lamda']
greek_unicode['Lambda'] = greek_unicode['Lamda']
greek_unicode['varsigma'] = '\N{GREEK SMALL LETTER FINAL SIGMA}'

# BOLD
b = lambda l: U('MATHEMATICAL BOLD SMALL %s' % l.upper())
B = lambda l: U('MATHEMATICAL BOLD CAPITAL %s' % l.upper())

bold_unicode = dict((l, b(l)) for l in ascii_lowercase)
bold_unicode.update((L, B(L)) for L in ascii_uppercase)

# GREEK BOLD
gb = lambda l: U('MATHEMATICAL BOLD SMALL %s' % l.upper())
GB = lambda l: U('MATHEMATICAL BOLD CAPITAL  %s' % l.upper())

greek_bold_letters = list(greeks) # make a copy, not strictly required here
# deal with Unicode's funny spelling of lambda
greek_bold_letters[greek_bold_letters.index('lambda')] = 'lamda'

# {}  greek letter -> (g,G)
greek_bold_unicode = dict((L, g(L)) for L in greek_bold_letters)
greek_bold_unicode.update((L[0].upper() + L[1:], G(L)) for L in greek_bold_letters)
greek_bold_unicode['lambda'] = greek_unicode['lamda']
greek_bold_unicode['Lambda'] = greek_unicode['Lamda']
greek_bold_unicode['varsigma'] = '\N{MATHEMATICAL BOLD SMALL FINAL SIGMA}'

digit_2txt = {
    '0':    'ZERO',
    '1':    'ONE',
    '2':    'TWO',
    '3':    'THREE',
    '4':    'FOUR',
    '5':    'FIVE',
    '6':    'SIX',
    '7':    'SEVEN',
    '8':    'EIGHT',
    '9':    'NINE',
}

symb_2txt = {
    '+':    'PLUS SIGN',
    '-':    'MINUS',
    '=':    'EQUALS SIGN',
    '(':    'LEFT PARENTHESIS',
    ')':    'RIGHT PARENTHESIS',
    '[':    'LEFT SQUARE BRACKET',
    ']':    'RIGHT SQUARE BRACKET',
    '{':    'LEFT CURLY BRACKET',
    '}':    'RIGHT CURLY BRACKET',

    # non-std
    '{}':   'CURLY BRACKET',
    'sum':  'SUMMATION',
    'int':  'INTEGRAL',
}

# SUBSCRIPT & SUPERSCRIPT
```
### 5 - sympy/printing/str.py:

Start line: 196, End line: 230

```python
class StrPrinter(Printer):

    def _print_AccumulationBounds(self, i):
        return "AccumBounds(%s, %s)" % (self._print(i.min),
                                        self._print(i.max))

    def _print_Inverse(self, I):
        return "%s**(-1)" % self.parenthesize(I.arg, PRECEDENCE["Pow"])

    def _print_Lambda(self, obj):
        expr = obj.expr
        sig = obj.signature
        if len(sig) == 1 and sig[0].is_symbol:
            sig = sig[0]
        return "Lambda(%s, %s)" % (self._print(sig), self._print(expr))

    def _print_LatticeOp(self, expr):
        args = sorted(expr.args, key=default_sort_key)
        return expr.func.__name__ + "(%s)" % ", ".join(self._print(arg) for arg in args)

    def _print_Limit(self, expr):
        e, z, z0, dir = expr.args
        if str(dir) == "+":
            return "Limit(%s, %s, %s)" % tuple(map(self._print, (e, z, z0)))
        else:
            return "Limit(%s, %s, %s, dir='%s')" % tuple(map(self._print,
                                                            (e, z, z0, dir)))

    def _print_list(self, expr):
        return "[%s]" % self.stringify(expr, ", ")

    def _print_MatrixBase(self, expr):
        return expr._format_str(self)

    def _print_MatrixElement(self, expr):
        return self.parenthesize(expr.parent, PRECEDENCE["Atom"], strict=True) \
            + '[%s, %s]' % (self._print(expr.i), self._print(expr.j))
```
### 6 - sympy/matrices/sparse.py:

Start line: 674, End line: 700

```python
class SparseMatrix(MatrixBase):

    RL = property(row_list, None, None, "Alternate faster representation")
    CL = property(col_list, None, None, "Alternate faster representation")

    def liupc(self):
        return _liupc(self)

    def row_structure_symbolic_cholesky(self):
        return _row_structure_symbolic_cholesky(self)

    def cholesky(self, hermitian=True):
        return _cholesky_sparse(self, hermitian=hermitian)

    def LDLdecomposition(self, hermitian=True):
        return _LDLdecomposition_sparse(self, hermitian=hermitian)

    def lower_triangular_solve(self, rhs):
        return _lower_triangular_solve_sparse(self, rhs)

    def upper_triangular_solve(self, rhs):
        return _upper_triangular_solve_sparse(self, rhs)

    liupc.__doc__                           = _liupc.__doc__
    row_structure_symbolic_cholesky.__doc__ = _row_structure_symbolic_cholesky.__doc__
    cholesky.__doc__                        = _cholesky_sparse.__doc__
    LDLdecomposition.__doc__                = _LDLdecomposition_sparse.__doc__
    lower_triangular_solve.__doc__          = lower_triangular_solve.__doc__
    upper_triangular_solve.__doc__          = upper_triangular_solve.__doc__
```
### 7 - sympy/matrices/expressions/matexpr.py:

Start line: 801, End line: 817

```python
class MatrixSymbol(MatrixExpr):

    def _eval_derivative_matrix_lines(self, x):
        if self != x:
            first = ZeroMatrix(x.shape[0], self.shape[0]) if self.shape[0] != 1 else S.Zero
            second = ZeroMatrix(x.shape[1], self.shape[1]) if self.shape[1] != 1 else S.Zero
            return [_LeftRightArgs(
                [first, second],
            )]
        else:
            first = Identity(self.shape[0]) if self.shape[0] != 1 else S.One
            second = Identity(self.shape[1]) if self.shape[1] != 1 else S.One
            return [_LeftRightArgs(
                [first, second],
            )]


def matrix_symbols(expr):
    return [sym for sym in expr.free_symbols if sym.is_Matrix]
```
### 8 - sympy/matrices/matrices.py:

Start line: 1, End line: 61

```python
import mpmath as mp

from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.compatibility import (
    Callable, NotIterable, as_int, is_sequence)
from sympy.core.decorators import deprecated
from sympy.core.expr import Expr
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol, uniquely_named_symbol
from sympy.core.sympify import sympify
from sympy.core.sympify import _sympify
from sympy.functions import exp, factorial, log
from sympy.functions.elementary.miscellaneous import Max, Min, sqrt
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.polys import cancel
from sympy.printing import sstr
from sympy.printing.defaults import Printable
from sympy.simplify import simplify as _simplify
from sympy.utilities.exceptions import SymPyDeprecationWarning
from sympy.utilities.iterables import flatten
from sympy.utilities.misc import filldedent

from .common import (
    MatrixCommon, MatrixError, NonSquareMatrixError, NonInvertibleMatrixError,
    ShapeError)

from .utilities import _iszero, _is_zero_after_expand_mul

from .determinant import (
    _find_reasonable_pivot, _find_reasonable_pivot_naive,
    _adjugate, _charpoly, _cofactor, _cofactor_matrix,
    _det, _det_bareiss, _det_berkowitz, _det_LU, _minor, _minor_submatrix)

from .reductions import _is_echelon, _echelon_form, _rank, _rref
from .subspaces import _columnspace, _nullspace, _rowspace, _orthogonalize

from .eigen import (
    _eigenvals, _eigenvects,
    _bidiagonalize, _bidiagonal_decomposition,
    _is_diagonalizable, _diagonalize,
    _is_positive_definite, _is_positive_semidefinite,
    _is_negative_definite, _is_negative_semidefinite, _is_indefinite,
    _jordan_form, _left_eigenvects, _singular_values)

from .decompositions import (
    _rank_decomposition, _cholesky, _LDLdecomposition,
    _LUdecomposition, _LUdecomposition_Simple, _LUdecompositionFF,
    _QRdecomposition)

from .graph import _connected_components, _connected_components_decomposition

from .solvers import (
    _diagonal_solve, _lower_triangular_solve, _upper_triangular_solve,
    _cholesky_solve, _LDLsolve, _LUsolve, _QRsolve, _gauss_jordan_solve,
    _pinv_solve, _solve, _solve_least_squares)

from .inverse import (
    _pinv, _inv_mod, _inv_ADJ, _inv_GE, _inv_LU, _inv_CH, _inv_LDL, _inv_QR,
    _inv, _inv_block)
```
### 9 - sympy/printing/str.py:

Start line: 232, End line: 247

```python
class StrPrinter(Printer):

    def _print_MatrixSlice(self, expr):
        def strslice(x, dim):
            x = list(x)
            if x[2] == 1:
                del x[2]
            if x[0] == 0:
                x[0] = ''
            if x[1] == dim:
                x[1] = ''
            return ':'.join(map(lambda arg: self._print(arg), x))
        return (self.parenthesize(expr.parent, PRECEDENCE["Atom"], strict=True) + '[' +
                strslice(expr.rowslice, expr.parent.rows) + ', ' +
                strslice(expr.colslice, expr.parent.cols) + ']')

    def _print_DeferredVector(self, expr):
        return expr.name
```
### 10 - sympy/matrices/matrices.py:

Start line: 1994, End line: 2070

```python
class MatrixBase(MatrixDeprecated,
                 MatrixCalculus,
                 MatrixEigen,
                 MatrixCommon,
                 Printable):

    def table(self, printer, rowstart='[', rowend=']', rowsep='\n',
              colsep=', ', align='right'):
        r"""
        String form of Matrix as a table.

        ``printer`` is the printer to use for on the elements (generally
        something like StrPrinter())

        ``rowstart`` is the string used to start each row (by default '[').

        ``rowend`` is the string used to end each row (by default ']').

        ``rowsep`` is the string used to separate rows (by default a newline).

        ``colsep`` is the string used to separate columns (by default ', ').

        ``align`` defines how the elements are aligned. Must be one of 'left',
        'right', or 'center'.  You can also use '<', '>', and '^' to mean the
        same thing, respectively.

        This is used by the string printer for Matrix.

        Examples
        ========

        >>> from sympy import Matrix
        >>> from sympy.printing.str import StrPrinter
        >>> M = Matrix([[1, 2], [-33, 4]])
        >>> printer = StrPrinter()
        >>> M.table(printer)
        '[  1, 2]\n[-33, 4]'
        >>> print(M.table(printer))
        [  1, 2]
        [-33, 4]
        >>> print(M.table(printer, rowsep=',\n'))
        [  1, 2],
        [-33, 4]
        >>> print('[%s]' % M.table(printer, rowsep=',\n'))
        [[  1, 2],
        [-33, 4]]
        >>> print(M.table(printer, colsep=' '))
        [  1 2]
        [-33 4]
        >>> print(M.table(printer, align='center'))
        [ 1 , 2]
        [-33, 4]
        >>> print(M.table(printer, rowstart='{', rowend='}'))
        {  1, 2}
        {-33, 4}
        """
        # Handle zero dimensions:
        if self.rows == 0 or self.cols == 0:
            return '[]'
        # Build table of string representations of the elements
        res = []
        # Track per-column max lengths for pretty alignment
        maxlen = [0] * self.cols
        for i in range(self.rows):
            res.append([])
            for j in range(self.cols):
                s = printer._print(self[i, j])
                res[-1].append(s)
                maxlen[j] = max(len(s), maxlen[j])
        # Patch strings together
        align = {
            'left': 'ljust',
            'right': 'rjust',
            'center': 'center',
            '<': 'ljust',
            '>': 'rjust',
            '^': 'center',
        }[align]
        for i, row in enumerate(res):
            for j, elem in enumerate(row):
                row[j] = getattr(elem, align)(maxlen[j])
            res[i] = rowstart + colsep.join(row) + rowend
        return rowsep.join(res)
```
### 11 - sympy/matrices/expressions/matexpr.py:

Start line: 34, End line: 138

```python
class MatrixExpr(Expr):
    """Superclass for Matrix Expressions

    MatrixExprs represent abstract matrices, linear transformations represented
    within a particular basis.

    Examples
    ========

    >>> from sympy import MatrixSymbol
    >>> A = MatrixSymbol('A', 3, 3)
    >>> y = MatrixSymbol('y', 3, 1)
    >>> x = (A.T*A).I * A * y

    See Also
    ========

    MatrixSymbol, MatAdd, MatMul, Transpose, Inverse
    """

    # Should not be considered iterable by the
    # sympy.core.compatibility.iterable function. Subclass that actually are
    # iterable (i.e., explicit matrices) should set this to True.
    _iterable = False

    _op_priority = 11.0

    is_Matrix = True  # type: bool
    is_MatrixExpr = True  # type: bool
    is_Identity = None  # type: FuzzyBool
    is_Inverse = False
    is_Transpose = False
    is_ZeroMatrix = False
    is_MatAdd = False
    is_MatMul = False

    is_commutative = False
    is_number = False
    is_symbol = False
    is_scalar = False

    def __new__(cls, *args, **kwargs):
        args = map(_sympify, args)
        return Basic.__new__(cls, *args, **kwargs)

    # The following is adapted from the core Expr object

    @property
    def _add_handler(self):
        return MatAdd

    @property
    def _mul_handler(self):
        return MatMul

    def __neg__(self):
        return MatMul(S.NegativeOne, self).doit()

    def __abs__(self):
        raise NotImplementedError

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__radd__')
    def __add__(self, other):
        return MatAdd(self, other, check=True).doit()

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__add__')
    def __radd__(self, other):
        return MatAdd(other, self, check=True).doit()

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__rsub__')
    def __sub__(self, other):
        return MatAdd(self, -other, check=True).doit()

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__sub__')
    def __rsub__(self, other):
        return MatAdd(other, -self, check=True).doit()

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__rmul__')
    def __mul__(self, other):
        return MatMul(self, other).doit()

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__rmul__')
    def __matmul__(self, other):
        return MatMul(self, other).doit()

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__mul__')
    def __rmul__(self, other):
        return MatMul(other, self).doit()

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__mul__')
    def __rmatmul__(self, other):
        return MatMul(other, self).doit()

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__rpow__')
    def __pow__(self, other):
        return MatPow(self, other).doit()
```
### 15 - sympy/core/symbol.py:

Start line: 285, End line: 342

```python
class Symbol(AtomicExpr, Boolean):

    __xnew__ = staticmethod(
        __new_stage2__)            # never cached (e.g. dummy)
    __xnew_cached_ = staticmethod(
        cacheit(__new_stage2__))   # symbols are always cached

    def __getnewargs__(self):
        return (self.name,)

    def __getstate__(self):
        return {'_assumptions': self._assumptions}

    def _hashable_content(self):
        # Note: user-specified assumptions not hashed, just derived ones
        return (self.name,) + tuple(sorted(self.assumptions0.items()))

    def _eval_subs(self, old, new):
        from sympy.core.power import Pow
        if old.is_Pow:
            return Pow(self, S.One, evaluate=False)._eval_subs(old, new)

    @property
    def assumptions0(self):
        return {key: value for key, value
                in self._assumptions.items() if value is not None}

    @cacheit
    def sort_key(self, order=None):
        return self.class_key(), (1, (self.name,)), S.One.sort_key(), S.One

    def as_dummy(self):
        # only put commutativity in explicitly if it is False
        return Dummy(self.name) if self.is_commutative is not False \
            else Dummy(self.name, commutative=self.is_commutative)

    def as_real_imag(self, deep=True, **hints):
        from sympy import im, re
        if hints.get('ignore') == self:
            return None
        else:
            return (re(self), im(self))

    def _sage_(self):
        import sage.all as sage
        return sage.var(self.name)

    def is_constant(self, *wrt, **flags):
        if not wrt:
            return False
        return not self in wrt

    @property
    def free_symbols(self):
        return {self}

    binary_symbols = free_symbols  # in this case, not always

    def as_set(self):
        return S.UniversalSet
```
### 25 - sympy/core/symbol.py:

Start line: 534, End line: 649

```python
def symbols(names, *, cls=Symbol, **args):
    r"""
    Transform strings into instances of :class:`Symbol` class.

    :func:`symbols` function returns a sequence of symbols with names taken
    from ``names`` argument, which can be a comma or whitespace delimited
    string, or a sequence of strings::

        >>> from sympy import symbols, Function

        >>> x, y, z = symbols('x,y,z')
        >>> a, b, c = symbols('a b c')

    The type of output is dependent on the properties of input arguments::

        >>> symbols('x')
        x
        >>> symbols('x,')
        (x,)
        >>> symbols('x,y')
        (x, y)
        >>> symbols(('a', 'b', 'c'))
        (a, b, c)
        >>> symbols(['a', 'b', 'c'])
        [a, b, c]
        >>> symbols({'a', 'b', 'c'})
        {a, b, c}

    If an iterable container is needed for a single symbol, set the ``seq``
    argument to ``True`` or terminate the symbol name with a comma::

        >>> symbols('x', seq=True)
        (x,)

    To reduce typing, range syntax is supported to create indexed symbols.
    Ranges are indicated by a colon and the type of range is determined by
    the character to the right of the colon. If the character is a digit
    then all contiguous digits to the left are taken as the nonnegative
    starting value (or 0 if there is no digit left of the colon) and all
    contiguous digits to the right are taken as 1 greater than the ending
    value::

        >>> symbols('x:10')
        (x0, x1, x2, x3, x4, x5, x6, x7, x8, x9)

        >>> symbols('x5:10')
        (x5, x6, x7, x8, x9)
        >>> symbols('x5(:2)')
        (x50, x51)

        >>> symbols('x5:10,y:5')
        (x5, x6, x7, x8, x9, y0, y1, y2, y3, y4)

        >>> symbols(('x5:10', 'y:5'))
        ((x5, x6, x7, x8, x9), (y0, y1, y2, y3, y4))

    If the character to the right of the colon is a letter, then the single
    letter to the left (or 'a' if there is none) is taken as the start
    and all characters in the lexicographic range *through* the letter to
    the right are used as the range::

        >>> symbols('x:z')
        (x, y, z)
        >>> symbols('x:c')  # null range
        ()
        >>> symbols('x(:c)')
        (xa, xb, xc)

        >>> symbols(':c')
        (a, b, c)

        >>> symbols('a:d, x:z')
        (a, b, c, d, x, y, z)

        >>> symbols(('a:d', 'x:z'))
        ((a, b, c, d), (x, y, z))

    Multiple ranges are supported; contiguous numerical ranges should be
    separated by parentheses to disambiguate the ending number of one
    range from the starting number of the next::

        >>> symbols('x:2(1:3)')
        (x01, x02, x11, x12)
        >>> symbols(':3:2')  # parsing is from left to right
        (00, 01, 10, 11, 20, 21)

    Only one pair of parentheses surrounding ranges are removed, so to
    include parentheses around ranges, double them. And to include spaces,
    commas, or colons, escape them with a backslash::

        >>> symbols('x((a:b))')
        (x(a), x(b))
        >>> symbols(r'x(:1\,:2)')  # or r'x((:1)\,(:2))'
        (x(0,0), x(0,1))

    All newly created symbols have assumptions set according to ``args``::

        >>> a = symbols('a', integer=True)
        >>> a.is_integer
        True

        >>> x, y, z = symbols('x,y,z', real=True)
        >>> x.is_real and y.is_real and z.is_real
        True

    Despite its name, :func:`symbols` can create symbol-like objects like
    instances of Function or Wild classes. To achieve this, set ``cls``
    keyword argument to the desired type::

        >>> symbols('f,g,h', cls=Function)
        (f, g, h)

        >>> type(_[0])
        <class 'sympy.core.function.UndefinedFunction'>

    """
    # ... other code
```
### 26 - sympy/core/symbol.py:

Start line: 18, End line: 54

```python
class Str(Atom):
    """
    Represents string in SymPy.

    Explanation
    ===========

    Previously, ``Symbol`` was used where string is needed in ``args`` of SymPy
    objects, e.g. denoting the name of the instance. However, since ``Symbol``
    represents mathematical scalar, this class should be used instead.

    """
    __slots__ = ('name',)

    def __new__(cls, name, **kwargs):
        if not isinstance(name, str):
            raise TypeError("name should be a string, not %s" % repr(type(name)))
        obj = Expr.__new__(cls, **kwargs)
        obj.name = name
        return obj

    def __getnewargs__(self):
        return (self.name,)

    def _hashable_content(self):
        return (self.name,)

def _filter_assumptions(kwargs):
    """Split the given dict into assumptions and non-assumptions.
    Keys are taken as assumptions if they correspond to an
    entry in ``_assume_defined``.
    """
    assumptions, nonassumptions = map(dict, sift(kwargs.items(),
        lambda i: i[0] in _assume_defined,
        binary=True))
    Symbol._sanitize(assumptions)
    return assumptions, nonassumptions
```
### 27 - sympy/matrices/expressions/matexpr.py:

Start line: 1, End line: 31

```python
from sympy.core.logic import FuzzyBool

from functools import wraps, reduce
import collections

from sympy.core import S, Symbol, Integer, Basic, Expr, Mul, Add
from sympy.core.decorators import call_highest_priority
from sympy.core.compatibility import SYMPY_INTS, default_sort_key
from sympy.core.sympify import SympifyError, _sympify
from sympy.functions import conjugate, adjoint
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.matrices.common import NonSquareMatrixError
from sympy.simplify import simplify
from sympy.utilities.misc import filldedent
from sympy.multipledispatch import dispatch


def _sympifyit(arg, retval=None):
    # This version of _sympifyit sympifies MutableMatrix objects
    def deco(func):
        @wraps(func)
        def __sympifyit_wrapper(a, b):
            try:
                b = _sympify(b)
                return func(a, b)
            except SympifyError:
                return retval

        return __sympifyit_wrapper

    return deco
```
### 32 - sympy/matrices/expressions/matexpr.py:

Start line: 685, End line: 715

```python
class MatrixElement(Expr):
    parent = property(lambda self: self.args[0])
    i = property(lambda self: self.args[1])
    j = property(lambda self: self.args[2])
    _diff_wrt = True
    is_symbol = True
    is_commutative = True

    def __new__(cls, name, n, m):
        n, m = map(_sympify, (n, m))
        from sympy import MatrixBase
        if isinstance(name, (MatrixBase,)):
            if n.is_Integer and m.is_Integer:
                return name[n, m]
        if isinstance(name, str):
            name = Symbol(name)
        name = _sympify(name)
        obj = Expr.__new__(cls, name, n, m)
        return obj

    def doit(self, **kwargs):
        deep = kwargs.get('deep', True)
        if deep:
            args = [arg.doit(**kwargs) for arg in self.args]
        else:
            args = self.args
        return args[0][args[1], args[2]]

    @property
    def indices(self):
        return self.args[1:]
```
### 46 - sympy/core/symbol.py:

Start line: 261, End line: 283

```python
class Symbol(AtomicExpr, Boolean):

    def __new_stage2__(cls, name, **assumptions):
        if not isinstance(name, str):
            raise TypeError("name should be a string, not %s" % repr(type(name)))

        obj = Expr.__new__(cls)
        obj.name = name

        # TODO: Issue #8873: Forcing the commutative assumption here means
        # later code such as ``srepr()`` cannot tell whether the user
        # specified ``commutative=True`` or omitted it.  To workaround this,
        # we keep a copy of the assumptions dict, then create the StdFactKB,
        # and finally overwrite its ``._generator`` with the dict copy.  This
        # is a bit of a hack because we assume StdFactKB merely copies the
        # given dict as ``._generator``, but future modification might, e.g.,
        # compute a minimal equivalent assumption set.
        tmp_asm_copy = assumptions.copy()

        # be strict about commutativity
        is_commutative = fuzzy_bool(assumptions.get('commutative', True))
        assumptions['commutative'] = is_commutative
        obj._assumptions = StdFactKB(assumptions)
        obj._assumptions._generator = tmp_asm_copy  # Issue #8873
        return obj
```
