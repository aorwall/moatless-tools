# sympy__sympy-14817

| **sympy/sympy** | `0dbdc0ea83d339936da175f8c3a97d0d6bafb9f8` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 576 |
| **Any found context length** | 576 |
| **Avg pos** | 2.0 |
| **Min pos** | 2 |
| **Max pos** | 2 |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sympy/printing/pretty/pretty.py b/sympy/printing/pretty/pretty.py
--- a/sympy/printing/pretty/pretty.py
+++ b/sympy/printing/pretty/pretty.py
@@ -825,7 +825,8 @@ def _print_MatAdd(self, expr):
             if s is None:
                 s = pform     # First element
             else:
-                if S(item.args[0]).is_negative:
+                coeff = item.as_coeff_mmul()[0]
+                if _coeff_isneg(S(coeff)):
                     s = prettyForm(*stringPict.next(s, ' '))
                     pform = self._print(item)
                 else:

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sympy/printing/pretty/pretty.py | 828 | 828 | 2 | 1 | 576


## Problem Statement

```
Error pretty printing MatAdd
\`\`\`py
>>> pprint(MatrixSymbol('x', n, n) + MatrixSymbol('y*', n, n))
Traceback (most recent call last):
  File "./sympy/core/sympify.py", line 368, in sympify
    expr = parse_expr(a, local_dict=locals, transformations=transformations, evaluate=evaluate)
  File "./sympy/parsing/sympy_parser.py", line 950, in parse_expr
    return eval_expr(code, local_dict, global_dict)
  File "./sympy/parsing/sympy_parser.py", line 863, in eval_expr
    code, global_dict, local_dict)  # take local objects in preference
  File "<string>", line 1
    Symbol ('y' )*
                 ^
SyntaxError: unexpected EOF while parsing

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "./sympy/printing/pretty/pretty.py", line 2371, in pretty_print
    use_unicode_sqrt_char=use_unicode_sqrt_char))
  File "./sympy/printing/pretty/pretty.py", line 2331, in pretty
    return pp.doprint(expr)
  File "./sympy/printing/pretty/pretty.py", line 62, in doprint
    return self._print(expr).render(**self._settings)
  File "./sympy/printing/printer.py", line 274, in _print
    return getattr(self, printmethod)(expr, *args, **kwargs)
  File "./sympy/printing/pretty/pretty.py", line 828, in _print_MatAdd
    if S(item.args[0]).is_negative:
  File "./sympy/core/sympify.py", line 370, in sympify
    raise SympifyError('could not parse %r' % a, exc)
sympy.core.sympify.SympifyError: Sympify of expression 'could not parse 'y*'' failed, because of exception being raised:
SyntaxError: unexpected EOF while parsing (<string>, line 1)
\`\`\`

The code shouldn't be using sympify to handle string arguments from MatrixSymbol.

I don't even understand what the code is doing. Why does it omit the `+` when the first argument is negative? This seems to assume that the arguments of MatAdd have a certain form, and that they will always print a certain way if they are negative. 

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 sympy/printing/pretty/pretty.py** | 837 | 887| 449 | 449 | 20623 | 
| **-> 2 <-** | **1 sympy/printing/pretty/pretty.py** | 821 | 835| 127 | 576 | 20623 | 
| 3 | 2 sympy/matrices/expressions/matexpr.py | 142 | 202| 443 | 1019 | 26775 | 
| 4 | **2 sympy/printing/pretty/pretty.py** | 750 | 769| 207 | 1226 | 26775 | 
| 5 | 2 sympy/matrices/expressions/matexpr.py | 1 | 30| 229 | 1455 | 26775 | 
| 6 | **2 sympy/printing/pretty/pretty.py** | 796 | 819| 216 | 1671 | 26775 | 
| 7 | 3 sympy/matrices/expressions/matadd.py | 1 | 14| 131 | 1802 | 27693 | 
| 8 | **3 sympy/printing/pretty/pretty.py** | 722 | 747| 284 | 2086 | 27693 | 
| 9 | 4 sympy/matrices/expressions/matmul.py | 78 | 133| 467 | 2553 | 30069 | 
| 10 | 5 sympy/printing/latex.py | 1450 | 1482| 263 | 2816 | 53023 | 
| 11 | 5 sympy/matrices/expressions/matadd.py | 16 | 62| 363 | 3179 | 53023 | 
| 12 | 6 sympy/matrices/common.py | 2038 | 2071| 299 | 3478 | 70355 | 
| 13 | 6 sympy/matrices/expressions/matexpr.py | 33 | 140| 794 | 4272 | 70355 | 
| 14 | 6 sympy/matrices/expressions/matadd.py | 65 | 80| 125 | 4397 | 70355 | 
| 15 | 7 sympy/matrices/expressions/blockmatrix.py | 1 | 19| 206 | 4603 | 74033 | 
| 16 | **7 sympy/printing/pretty/pretty.py** | 989 | 1021| 354 | 4957 | 74033 | 
| 17 | 8 sympy/printing/fcode.py | 241 | 282| 352 | 5309 | 79784 | 
| 18 | 9 sympy/matrices/expressions/matpow.py | 51 | 79| 242 | 5551 | 80432 | 
| 19 | 10 sympy/matrices/immutable.py | 1 | 14| 113 | 5664 | 81801 | 
| 20 | **10 sympy/printing/pretty/pretty.py** | 1456 | 1482| 249 | 5913 | 81801 | 
| 21 | 11 sympy/core/sympify.py | 78 | 259| 1755 | 7668 | 85816 | 
| 22 | 11 sympy/matrices/expressions/matmul.py | 1 | 12| 115 | 7783 | 85816 | 
| 23 | 11 sympy/matrices/expressions/matmul.py | 261 | 296| 229 | 8012 | 85816 | 
| 24 | 11 sympy/matrices/expressions/matpow.py | 1 | 49| 411 | 8423 | 85816 | 
| 25 | 11 sympy/printing/latex.py | 1484 | 1496| 159 | 8582 | 85816 | 
| 26 | 11 sympy/matrices/expressions/matexpr.py | 643 | 699| 422 | 9004 | 85816 | 
| 27 | **11 sympy/printing/pretty/pretty.py** | 772 | 794| 224 | 9228 | 85816 | 
| 28 | 11 sympy/printing/latex.py | 1402 | 1427| 271 | 9499 | 85816 | 
| 29 | 12 sympy/__init__.py | 1 | 95| 697 | 10196 | 86513 | 
| 30 | 12 sympy/matrices/expressions/matmul.py | 48 | 76| 316 | 10512 | 86513 | 
| 31 | 13 sympy/printing/str.py | 323 | 376| 416 | 10928 | 93660 | 
| 32 | **13 sympy/printing/pretty/pretty.py** | 1484 | 1525| 325 | 11253 | 93660 | 
| 33 | **13 sympy/printing/pretty/pretty.py** | 569 | 623| 471 | 11724 | 93660 | 
| 34 | 13 sympy/core/sympify.py | 260 | 372| 840 | 12564 | 93660 | 
| 35 | 14 sympy/printing/glsl.py | 312 | 490| 1932 | 14496 | 98458 | 
| 36 | 14 sympy/matrices/common.py | 2073 | 2108| 310 | 14806 | 98458 | 
| 37 | **14 sympy/printing/pretty/pretty.py** | 1 | 34| 287 | 15093 | 98458 | 
| 38 | 15 sympy/printing/repr.py | 99 | 143| 419 | 15512 | 100558 | 
| 39 | 16 sympy/diffgeom/diffgeom.py | 1 | 20| 174 | 15686 | 115067 | 
| 40 | 17 sympy/matrices/sparse.py | 1 | 17| 117 | 15803 | 125510 | 
| 41 | 17 sympy/matrices/expressions/matexpr.py | 446 | 470| 237 | 16040 | 125510 | 
| 42 | 17 sympy/matrices/expressions/matexpr.py | 204 | 248| 417 | 16457 | 125510 | 
| 43 | 18 sympy/core/expr.py | 92 | 150| 483 | 16940 | 154498 | 
| 44 | 19 sympy/printing/pycode.py | 180 | 211| 288 | 17228 | 158869 | 
| 45 | 19 sympy/matrices/expressions/matexpr.py | 590 | 614| 215 | 17443 | 158869 | 
| 46 | 19 sympy/matrices/common.py | 1919 | 1948| 299 | 17742 | 158869 | 
| 47 | 19 sympy/printing/pycode.py | 429 | 449| 195 | 17937 | 158869 | 
| 48 | 20 sympy/core/add.py | 667 | 719| 383 | 18320 | 167460 | 
| 49 | **20 sympy/printing/pretty/pretty.py** | 524 | 567| 480 | 18800 | 167460 | 
| 50 | 21 sympy/printing/theanocode.py | 98 | 154| 523 | 19323 | 169543 | 
| 51 | 21 sympy/printing/repr.py | 85 | 97| 139 | 19462 | 169543 | 
| 52 | 22 sympy/polys/polymatrix.py | 77 | 87| 143 | 19605 | 170486 | 
| 53 | 23 sympy/printing/julia.py | 213 | 255| 290 | 19895 | 176237 | 
| 54 | 24 sympy/printing/ccode.py | 627 | 763| 1518 | 21413 | 183318 | 
| 55 | 25 sympy/printing/rust.py | 429 | 466| 349 | 21762 | 188755 | 
| 56 | 25 sympy/matrices/expressions/matexpr.py | 472 | 587| 1155 | 22917 | 188755 | 
| 57 | **25 sympy/printing/pretty/pretty.py** | 1103 | 1149| 403 | 23320 | 188755 | 
| 58 | 25 sympy/printing/ccode.py | 355 | 390| 388 | 23708 | 188755 | 
| 59 | 25 sympy/core/add.py | 637 | 650| 139 | 23847 | 188755 | 
| 60 | **25 sympy/printing/pretty/pretty.py** | 1151 | 1233| 773 | 24620 | 188755 | 
| 61 | 25 sympy/printing/latex.py | 1505 | 1535| 303 | 24923 | 188755 | 
| 62 | **25 sympy/printing/pretty/pretty.py** | 131 | 188| 556 | 25479 | 188755 | 
| 63 | 26 sympy/matrices/expressions/__init__.py | 1 | 20| 187 | 25666 | 188943 | 
| 64 | 26 sympy/printing/latex.py | 1428 | 1448| 208 | 25874 | 188943 | 
| 65 | 26 sympy/core/add.py | 860 | 876| 126 | 26000 | 188943 | 
| 66 | 26 sympy/core/add.py | 583 | 635| 383 | 26383 | 188943 | 
| 67 | 27 sympy/printing/rcode.py | 230 | 264| 372 | 26755 | 192668 | 
| 68 | 27 sympy/core/add.py | 652 | 665| 139 | 26894 | 192668 | 
| 69 | 27 sympy/printing/str.py | 250 | 265| 142 | 27036 | 192668 | 
| 70 | 27 sympy/printing/julia.py | 341 | 349| 139 | 27175 | 192668 | 
| 71 | 27 sympy/matrices/expressions/matexpr.py | 287 | 320| 355 | 27530 | 192668 | 
| 72 | 27 sympy/printing/glsl.py | 134 | 167| 339 | 27869 | 192668 | 
| 73 | 28 sympy/interactive/printing.py | 149 | 230| 698 | 28567 | 196450 | 
| 74 | 28 sympy/core/sympify.py | 375 | 401| 180 | 28747 | 196450 | 
| 75 | 28 sympy/core/add.py | 828 | 858| 296 | 29043 | 196450 | 
| 76 | 28 sympy/printing/theanocode.py | 1 | 64| 563 | 29606 | 196450 | 


### Hint

```
Looks like it comes from fbbbd392e6c which is from https://github.com/sympy/sympy/pull/14248. CC @jashan498 
`_print_MatAdd` should use the same methods as `_print_Add` to determine whether or not to include a plus or minus sign. If it's possible, it could even reuse the same code. 
```

## Patch

```diff
diff --git a/sympy/printing/pretty/pretty.py b/sympy/printing/pretty/pretty.py
--- a/sympy/printing/pretty/pretty.py
+++ b/sympy/printing/pretty/pretty.py
@@ -825,7 +825,8 @@ def _print_MatAdd(self, expr):
             if s is None:
                 s = pform     # First element
             else:
-                if S(item.args[0]).is_negative:
+                coeff = item.as_coeff_mmul()[0]
+                if _coeff_isneg(S(coeff)):
                     s = prettyForm(*stringPict.next(s, ' '))
                     pform = self._print(item)
                 else:

```

## Test Patch

```diff
diff --git a/sympy/printing/pretty/tests/test_pretty.py b/sympy/printing/pretty/tests/test_pretty.py
--- a/sympy/printing/pretty/tests/test_pretty.py
+++ b/sympy/printing/pretty/tests/test_pretty.py
@@ -6094,11 +6094,16 @@ def test_MatrixSymbol_printing():
     A = MatrixSymbol("A", 3, 3)
     B = MatrixSymbol("B", 3, 3)
     C = MatrixSymbol("C", 3, 3)
-
     assert pretty(-A*B*C) == "-A*B*C"
     assert pretty(A - B) == "-B + A"
     assert pretty(A*B*C - A*B - B*C) == "-A*B -B*C + A*B*C"
 
+    # issue #14814
+    x = MatrixSymbol('x', n, n)
+    y = MatrixSymbol('y*', n, n)
+    assert pretty(x + y) == "x + y*"
+    assert pretty(-a*x + -2*y*y) == "-a*x -2*y**y*"
+
 
 def test_degree_printing():
     expr1 = 90*degree

```


## Code snippets

### 1 - sympy/printing/pretty/pretty.py:

Start line: 837, End line: 887

```python
class PrettyPrinter(Printer):

    def _print_MatMul(self, expr):
        args = list(expr.args)
        from sympy import Add, MatAdd, HadamardProduct
        for i, a in enumerate(args):
            if (isinstance(a, (Add, MatAdd, HadamardProduct))
                    and len(expr.args) > 1):
                args[i] = prettyForm(*self._print(a).parens())
            else:
                args[i] = self._print(a)

        return prettyForm.__mul__(*args)

    def _print_DotProduct(self, expr):
        args = list(expr.args)

        for i, a in enumerate(args):
            args[i] = self._print(a)
        return prettyForm.__mul__(*args)

    def _print_MatPow(self, expr):
        pform = self._print(expr.base)
        from sympy.matrices import MatrixSymbol
        if not isinstance(expr.base, MatrixSymbol):
            pform = prettyForm(*pform.parens())
        pform = pform**(self._print(expr.exp))
        return pform

    def _print_HadamardProduct(self, expr):
        from sympy import MatAdd, MatMul
        if self._use_unicode:
            delim = pretty_atom('Ring')
        else:
            delim = '.*'
        return self._print_seq(expr.args, None, None, delim,
                parenthesize=lambda x: isinstance(x, (MatAdd, MatMul)))

    def _print_KroneckerProduct(self, expr):
        from sympy import MatAdd, MatMul
        if self._use_unicode:
            delim = u' \N{N-ARY CIRCLED TIMES OPERATOR} '
        else:
            delim = ' x '
        return self._print_seq(expr.args, None, None, delim,
                parenthesize=lambda x: isinstance(x, (MatAdd, MatMul)))

    _print_MatrixSymbol = _print_Symbol

    def _print_FunctionMatrix(self, X):
        D = self._print(X.lamda.expr)
        D = prettyForm(*D.parens('[', ']'))
        return D
```
### 2 - sympy/printing/pretty/pretty.py:

Start line: 821, End line: 835

```python
class PrettyPrinter(Printer):

    def _print_MatAdd(self, expr):
        s = None
        for item in expr.args:
            pform = self._print(item)
            if s is None:
                s = pform     # First element
            else:
                if S(item.args[0]).is_negative:
                    s = prettyForm(*stringPict.next(s, ' '))
                    pform = self._print(item)
                else:
                    s = prettyForm(*stringPict.next(s, ' + '))
                s = prettyForm(*stringPict.next(s, pform))

        return s
```
### 3 - sympy/matrices/expressions/matexpr.py:

Start line: 142, End line: 202

```python
class MatrixExpr(Expr):

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__pow__')
    def __rpow__(self, other):
        raise NotImplementedError("Matrix Power not defined")

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__rdiv__')
    def __div__(self, other):
        return self * other**S.NegativeOne

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__div__')
    def __rdiv__(self, other):
        raise NotImplementedError()
        #return MatMul(other, Pow(self, S.NegativeOne))

    __truediv__ = __div__
    __rtruediv__ = __rdiv__

    @property
    def rows(self):
        return self.shape[0]

    @property
    def cols(self):
        return self.shape[1]

    @property
    def is_square(self):
        return self.rows == self.cols

    def _eval_conjugate(self):
        from sympy.matrices.expressions.adjoint import Adjoint
        from sympy.matrices.expressions.transpose import Transpose
        return Adjoint(Transpose(self))

    def as_real_imag(self):
        from sympy import I
        real = (S(1)/2) * (self + self._eval_conjugate())
        im = (self - self._eval_conjugate())/(2*I)
        return (real, im)

    def _eval_inverse(self):
        from sympy.matrices.expressions.inverse import Inverse
        return Inverse(self)

    def _eval_transpose(self):
        return Transpose(self)

    def _eval_power(self, exp):
        return MatPow(self, exp)

    def _eval_simplify(self, **kwargs):
        if self.is_Atom:
            return self
        else:
            return self.__class__(*[simplify(x, **kwargs) for x in self.args])

    def _eval_adjoint(self):
        from sympy.matrices.expressions.adjoint import Adjoint
        return Adjoint(self)
```
### 4 - sympy/printing/pretty/pretty.py:

Start line: 750, End line: 769

```python
class PrettyPrinter(Printer):


    def _print_MatrixElement(self, expr):
        from sympy.matrices import MatrixSymbol
        from sympy import Symbol
        if (isinstance(expr.parent, MatrixSymbol)
                and expr.i.is_number and expr.j.is_number):
            return self._print(
                    Symbol(expr.parent.name + '_%d%d' % (expr.i, expr.j)))
        else:
            prettyFunc = self._print(expr.parent)
            prettyFunc = prettyForm(*prettyFunc.parens())
            prettyIndices = self._print_seq((expr.i, expr.j), delimiter=', '
                    ).parens(left='[', right=']')[0]
            pform = prettyForm(binding=prettyForm.FUNC,
                    *stringPict.next(prettyFunc, prettyIndices))

            # store pform parts so it can be reassembled e.g. when powered
            pform.prettyFunc = prettyFunc
            pform.prettyArgs = prettyIndices

            return pform
```
### 5 - sympy/matrices/expressions/matexpr.py:

Start line: 1, End line: 30

```python
from __future__ import print_function, division

from functools import wraps, reduce
import collections

from sympy.core import S, Symbol, Tuple, Integer, Basic, Expr, Eq
from sympy.core.decorators import call_highest_priority
from sympy.core.compatibility import range, SYMPY_INTS, default_sort_key
from sympy.core.sympify import SympifyError, sympify
from sympy.functions import conjugate, adjoint
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.matrices import ShapeError
from sympy.simplify import simplify
from sympy.utilities.misc import filldedent


def _sympifyit(arg, retval=None):
    # This version of _sympifyit sympifies MutableMatrix objects
    def deco(func):
        @wraps(func)
        def __sympifyit_wrapper(a, b):
            try:
                b = sympify(b, strict=True)
                return func(a, b)
            except SympifyError:
                return retval

        return __sympifyit_wrapper

    return deco
```
### 6 - sympy/printing/pretty/pretty.py:

Start line: 796, End line: 819

```python
class PrettyPrinter(Printer):

    def _print_Transpose(self, expr):
        pform = self._print(expr.arg)
        from sympy.matrices import MatrixSymbol
        if not isinstance(expr.arg, MatrixSymbol):
            pform = prettyForm(*pform.parens())
        pform = pform**(prettyForm('T'))
        return pform

    def _print_Adjoint(self, expr):
        pform = self._print(expr.arg)
        if self._use_unicode:
            dag = prettyForm(u'\N{DAGGER}')
        else:
            dag = prettyForm('+')
        from sympy.matrices import MatrixSymbol
        if not isinstance(expr.arg, MatrixSymbol):
            pform = prettyForm(*pform.parens())
        pform = pform**dag
        return pform

    def _print_BlockMatrix(self, B):
        if B.blocks.shape == (1, 1):
            return self._print(B.blocks[0, 0])
        return self._print(B.blocks)
```
### 7 - sympy/matrices/expressions/matadd.py:

Start line: 1, End line: 14

```python
from __future__ import print_function, division

from sympy.core.compatibility import reduce
from operator import add

from sympy.core import Add, Basic, sympify
from sympy.functions import adjoint
from sympy.matrices.matrices import MatrixBase
from sympy.matrices.expressions.transpose import transpose
from sympy.strategies import (rm_id, unpack, flatten, sort, condition,
        exhaust, do_one, glom)
from sympy.matrices.expressions.matexpr import MatrixExpr, ShapeError, ZeroMatrix
from sympy.utilities import default_sort_key, sift
from sympy.core.operations import AssocOp
```
### 8 - sympy/printing/pretty/pretty.py:

Start line: 722, End line: 747

```python
class PrettyPrinter(Printer):

    def _print_MatrixBase(self, e):
        D = self._print_matrix_contents(e)
        D.baseline = D.height()//2
        D = prettyForm(*D.parens('[', ']'))
        return D
    _print_ImmutableMatrix = _print_MatrixBase
    _print_Matrix = _print_MatrixBase

    def _print_TensorProduct(self, expr):
        # This should somehow share the code with _print_WedgeProduct:
        circled_times = "\u2297"
        return self._print_seq(expr.args, None, None, circled_times,
            parenthesize=lambda x: precedence_traditional(x) <= PRECEDENCE["Mul"])

    def _print_WedgeProduct(self, expr):
        # This should somehow share the code with _print_TensorProduct:
        wedge_symbol = u"\u2227"
        return self._print_seq(expr.args, None, None, wedge_symbol,
            parenthesize=lambda x: precedence_traditional(x) <= PRECEDENCE["Mul"])

    def _print_Trace(self, e):
        D = self._print(e.arg)
        D = prettyForm(*D.parens('(',')'))
        D.baseline = D.height()//2
        D = prettyForm(*D.left('\n'*(0) + 'tr'))
        return D
```
### 9 - sympy/matrices/expressions/matmul.py:

Start line: 78, End line: 133

```python
class MatMul(MatrixExpr):

    def as_coeff_matrices(self):
        scalars = [x for x in self.args if not x.is_Matrix]
        matrices = [x for x in self.args if x.is_Matrix]
        coeff = Mul(*scalars)

        return coeff, matrices

    def as_coeff_mmul(self):
        coeff, matrices = self.as_coeff_matrices()
        return coeff, MatMul(*matrices)

    def _eval_transpose(self):
        return MatMul(*[transpose(arg) for arg in self.args[::-1]]).doit()

    def _eval_adjoint(self):
        return MatMul(*[adjoint(arg) for arg in self.args[::-1]]).doit()

    def _eval_trace(self):
        factor, mmul = self.as_coeff_mmul()
        if factor != 1:
            from .trace import trace
            return factor * trace(mmul.doit())
        else:
            raise NotImplementedError("Can't simplify any further")

    def _eval_determinant(self):
        from sympy.matrices.expressions.determinant import Determinant
        factor, matrices = self.as_coeff_matrices()
        square_matrices = only_squares(*matrices)
        return factor**self.rows * Mul(*list(map(Determinant, square_matrices)))

    def _eval_inverse(self):
        try:
            return MatMul(*[
                arg.inverse() if isinstance(arg, MatrixExpr) else arg**-1
                    for arg in self.args[::-1]]).doit()
        except ShapeError:
            from sympy.matrices.expressions.inverse import Inverse
            return Inverse(self)

    def doit(self, **kwargs):
        deep = kwargs.get('deep', True)
        if deep:
            args = [arg.doit(**kwargs) for arg in self.args]
        else:
            args = self.args
        return canonicalize(MatMul(*args))

    # Needed for partial compatibility with Mul
    def args_cnc(self, **kwargs):
        coeff, matrices = self.as_coeff_matrices()
        # I don't know how coeff could have noncommutative factors, but this
        # handles it.
        coeff_c, coeff_nc = coeff.args_cnc(**kwargs)

        return coeff_c, coeff_nc + matrices
```
### 10 - sympy/printing/latex.py:

Start line: 1450, End line: 1482

```python
class LatexPrinter(Printer):

    def _print_BlockMatrix(self, expr):
        return self._print(expr.blocks)

    def _print_Transpose(self, expr):
        mat = expr.arg
        from sympy.matrices import MatrixSymbol
        if not isinstance(mat, MatrixSymbol):
            return r"\left(%s\right)^T" % self._print(mat)
        else:
            return "%s^T" % self._print(mat)

    def _print_Adjoint(self, expr):
        mat = expr.arg
        from sympy.matrices import MatrixSymbol
        if not isinstance(mat, MatrixSymbol):
            return r"\left(%s\right)^\dagger" % self._print(mat)
        else:
            return r"%s^\dagger" % self._print(mat)

    def _print_MatAdd(self, expr):
        terms = [self._print(t) for t in expr.args]
        l = []
        for t in terms:
            if t.startswith('-'):
                sign = "-"
                t = t[1:]
            else:
                sign = "+"
            l.extend([sign, t])
        sign = l.pop(0)
        if sign == '+':
            sign = ""
        return sign + ' '.join(l)
```
### 16 - sympy/printing/pretty/pretty.py:

Start line: 989, End line: 1021

```python
class PrettyPrinter(Printer):

    def _print_NDimArray(self, expr):
        from sympy import ImmutableMatrix

        if expr.rank() == 0:
            return self._print(expr[()])

        level_str = [[]] + [[] for i in range(expr.rank())]
        shape_ranges = [list(range(i)) for i in expr.shape]
        for outer_i in itertools.product(*shape_ranges):
            level_str[-1].append(expr[outer_i])
            even = True
            for back_outer_i in range(expr.rank()-1, -1, -1):
                if len(level_str[back_outer_i+1]) < expr.shape[back_outer_i]:
                    break
                if even:
                    level_str[back_outer_i].append(level_str[back_outer_i+1])
                else:
                    level_str[back_outer_i].append(ImmutableMatrix(level_str[back_outer_i+1]))
                    if len(level_str[back_outer_i + 1]) == 1:
                        level_str[back_outer_i][-1] = ImmutableMatrix([[level_str[back_outer_i][-1]]])
                even = not even
                level_str[back_outer_i+1] = []

        out_expr = level_str[0][0]
        if expr.rank() % 2 == 1:
            out_expr = ImmutableMatrix([out_expr])

        return self._print(out_expr)

    _print_ImmutableDenseNDimArray = _print_NDimArray
    _print_ImmutableSparseNDimArray = _print_NDimArray
    _print_MutableDenseNDimArray = _print_NDimArray
    _print_MutableSparseNDimArray = _print_NDimArray
```
### 20 - sympy/printing/pretty/pretty.py:

Start line: 1456, End line: 1482

```python
class PrettyPrinter(Printer):

    def _print_Add(self, expr, order=None):
        if self.order == 'none':
            terms = list(expr.args)
        else:
            terms = self._as_ordered_terms(expr, order=order)
        pforms, indices = [], []

        def pretty_negative(pform, index):
            """Prepend a minus sign to a pretty form. """
            #TODO: Move this code to prettyForm
            if index == 0:
                if pform.height() > 1:
                    pform_neg = '- '
                else:
                    pform_neg = '-'
            else:
                pform_neg = ' - '

            if (pform.binding > prettyForm.NEG
                or pform.binding == prettyForm.ADD):
                p = stringPict(*pform.parens())
            else:
                p = pform
            p = stringPict.next(pform_neg, p)
            # Lower the binding to NEG, even if it was higher. Otherwise, it
            # will print as a + ( - (b)), instead of a - (b).
            return prettyForm(binding=prettyForm.NEG, *p)
        # ... other code
```
### 27 - sympy/printing/pretty/pretty.py:

Start line: 772, End line: 794

```python
class PrettyPrinter(Printer):


    def _print_MatrixSlice(self, m):
        # XXX works only for applied functions
        prettyFunc = self._print(m.parent)
        def ppslice(x):
            x = list(x)
            if x[2] == 1:
                del x[2]
            if x[1] == x[0] + 1:
                del x[1]
            if x[0] == 0:
                x[0] = ''
            return prettyForm(*self._print_seq(x, delimiter=':'))
        prettyArgs = self._print_seq((ppslice(m.rowslice),
            ppslice(m.colslice)), delimiter=', ').parens(left='[', right=']')[0]

        pform = prettyForm(
            binding=prettyForm.FUNC, *stringPict.next(prettyFunc, prettyArgs))

        # store pform parts so it can be reassembled e.g. when powered
        pform.prettyFunc = prettyFunc
        pform.prettyArgs = prettyArgs

        return pform
```
### 32 - sympy/printing/pretty/pretty.py:

Start line: 1484, End line: 1525

```python
class PrettyPrinter(Printer):

    def _print_Add(self, expr, order=None):
        # ... other code

        for i, term in enumerate(terms):
            if term.is_Mul and _coeff_isneg(term):
                coeff, other = term.as_coeff_mul(rational=False)
                pform = self._print(Mul(-coeff, *other, evaluate=False))
                pforms.append(pretty_negative(pform, i))
            elif term.is_Rational and term.q > 1:
                pforms.append(None)
                indices.append(i)
            elif term.is_Number and term < 0:
                pform = self._print(-term)
                pforms.append(pretty_negative(pform, i))
            elif term.is_Relational:
                pforms.append(prettyForm(*self._print(term).parens()))
            else:
                pforms.append(self._print(term))

        if indices:
            large = True

            for pform in pforms:
                if pform is not None and pform.height() > 1:
                    break
            else:
                large = False

            for i in indices:
                term, negative = terms[i], False

                if term < 0:
                    term, negative = -term, True

                if large:
                    pform = prettyForm(str(term.p))/prettyForm(str(term.q))
                else:
                    pform = self._print(term)

                if negative:
                    pform = pretty_negative(pform, i)

                pforms[i] = pform

        return prettyForm.__add__(*pforms)
```
### 33 - sympy/printing/pretty/pretty.py:

Start line: 569, End line: 623

```python
class PrettyPrinter(Printer):

    def _print_Sum(self, expr):
        # ... other code

        f = expr.function

        prettyF = self._print(f)

        if f.is_Add:  # add parens
            prettyF = prettyForm(*prettyF.parens())

        H = prettyF.height() + 2

        # \sum \sum \sum ...
        first = True
        max_upper = 0
        sign_height = 0

        for lim in expr.limits:
            if len(lim) == 3:
                prettyUpper = self._print(lim[2])
                prettyLower = self._print(Equality(lim[0], lim[1]))
            elif len(lim) == 2:
                prettyUpper = self._print("")
                prettyLower = self._print(Equality(lim[0], lim[1]))
            elif len(lim) == 1:
                prettyUpper = self._print("")
                prettyLower = self._print(lim[0])

            max_upper = max(max_upper, prettyUpper.height())

            # Create sum sign based on the height of the argument
            d, h, slines, adjustment = asum(
                H, prettyLower.width(), prettyUpper.width(), ascii_mode)
            prettySign = stringPict('')
            prettySign = prettyForm(*prettySign.stack(*slines))

            if first:
                sign_height = prettySign.height()

            prettySign = prettyForm(*prettySign.above(prettyUpper))
            prettySign = prettyForm(*prettySign.below(prettyLower))

            if first:
                # change F baseline so it centers on the sign
                prettyF.baseline -= d - (prettyF.height()//2 -
                                         prettyF.baseline) - adjustment
                first = False

            # put padding to the right
            pad = stringPict('')
            pad = prettyForm(*pad.stack(*[' ']*h))
            prettySign = prettyForm(*prettySign.right(pad))
            # put the present prettyF to the right
            prettyF = prettyForm(*prettySign.right(prettyF))

        prettyF.baseline = max_upper + sign_height//2
        prettyF.binding = prettyForm.MUL
        return prettyF
```
### 37 - sympy/printing/pretty/pretty.py:

Start line: 1, End line: 34

```python
from __future__ import print_function, division

import itertools

from sympy.core import S
from sympy.core.containers import Tuple
from sympy.core.function import _coeff_isneg
from sympy.core.mod import Mod
from sympy.core.mul import Mul
from sympy.core.numbers import Rational
from sympy.core.power import Pow
from sympy.core.relational import Equality
from sympy.core.symbol import Symbol
from sympy.printing.precedence import PRECEDENCE, precedence, precedence_traditional
from sympy.utilities import group
from sympy.utilities.iterables import has_variety
from sympy.core.sympify import SympifyError
from sympy.core.compatibility import range
from sympy.core.add import Add

from sympy.printing.printer import Printer
from sympy.printing.str import sstr
from sympy.printing.conventions import requires_partial

from .stringpict import prettyForm, stringPict
from .pretty_symbology import xstr, hobj, vobj, xobj, xsym, pretty_symbol, \
    pretty_atom, pretty_use_unicode, pretty_try_use_unicode, greek_unicode, U, \
    annotated

from sympy.utilities import default_sort_key

# rename for usage from outside
pprint_use_unicode = pretty_use_unicode
pprint_try_use_unicode = pretty_try_use_unicode
```
### 49 - sympy/printing/pretty/pretty.py:

Start line: 524, End line: 567

```python
class PrettyPrinter(Printer):

    def _print_Sum(self, expr):
        ascii_mode = not self._use_unicode

        def asum(hrequired, lower, upper, use_ascii):
            def adjust(s, wid=None, how='<^>'):
                if not wid or len(s) > wid:
                    return s
                need = wid - len(s)
                if how == '<^>' or how == "<" or how not in list('<^>'):
                    return s + ' '*need
                half = need//2
                lead = ' '*half
                if how == ">":
                    return " "*need + s
                return lead + s + ' '*(need - len(lead))

            h = max(hrequired, 2)
            d = h//2
            w = d + 1
            more = hrequired % 2

            lines = []
            if use_ascii:
                lines.append("_"*(w) + ' ')
                lines.append(r"\%s`" % (' '*(w - 1)))
                for i in range(1, d):
                    lines.append('%s\\%s' % (' '*i, ' '*(w - i)))
                if more:
                    lines.append('%s)%s' % (' '*(d), ' '*(w - d)))
                for i in reversed(range(1, d)):
                    lines.append('%s/%s' % (' '*i, ' '*(w - i)))
                lines.append("/" + "_"*(w - 1) + ',')
                return d, h + more, lines, 0
            else:
                w = w + more
                d = d + more
                vsum = vobj('sum', 4)
                lines.append("_"*(w))
                for i in range(0, d):
                    lines.append('%s%s%s' % (' '*i, vsum[2], ' '*(w - i - 1)))
                for i in reversed(range(0, d)):
                    lines.append('%s%s%s' % (' '*i, vsum[4], ' '*(w - i - 1)))
                lines.append(vsum[8]*(w))
                return d, h + 2*more, lines, more
        # ... other code
```
### 57 - sympy/printing/pretty/pretty.py:

Start line: 1103, End line: 1149

```python
class PrettyPrinter(Printer):

    def _print_hyper(self, e):
        # FIXME refactor Matrix, Piecewise, and this into a tabular environment
        ap = [self._print(a) for a in e.ap]
        bq = [self._print(b) for b in e.bq]

        P = self._print(e.argument)
        P.baseline = P.height()//2

        # Drawing result - first create the ap, bq vectors
        D = None
        for v in [ap, bq]:
            D_row = self._hprint_vec(v)
            if D is None:
                D = D_row       # first row in a picture
            else:
                D = prettyForm(*D.below(' '))
                D = prettyForm(*D.below(D_row))

        # make sure that the argument `z' is centred vertically
        D.baseline = D.height()//2

        # insert horizontal separator
        P = prettyForm(*P.left(' '))
        D = prettyForm(*D.right(' '))

        # insert separating `|`
        D = self._hprint_vseparator(D, P)

        # add parens
        D = prettyForm(*D.parens('(', ')'))

        # create the F symbol
        above = D.height()//2 - 1
        below = D.height() - above - 1

        sz, t, b, add, img = annotated('F')
        F = prettyForm('\n' * (above - t) + img + '\n' * (below - b),
                       baseline=above + sz)
        add = (sz + 1)//2

        F = prettyForm(*F.left(self._print(len(e.ap))))
        F = prettyForm(*F.right(self._print(len(e.bq))))
        F.baseline = above + add

        D = prettyForm(*F.right(' ', D))

        return D
```
### 60 - sympy/printing/pretty/pretty.py:

Start line: 1151, End line: 1233

```python
class PrettyPrinter(Printer):

    def _print_meijerg(self, e):
        # FIXME refactor Matrix, Piecewise, and this into a tabular environment

        v = {}
        v[(0, 0)] = [self._print(a) for a in e.an]
        v[(0, 1)] = [self._print(a) for a in e.aother]
        v[(1, 0)] = [self._print(b) for b in e.bm]
        v[(1, 1)] = [self._print(b) for b in e.bother]

        P = self._print(e.argument)
        P.baseline = P.height()//2

        vp = {}
        for idx in v:
            vp[idx] = self._hprint_vec(v[idx])

        for i in range(2):
            maxw = max(vp[(0, i)].width(), vp[(1, i)].width())
            for j in range(2):
                s = vp[(j, i)]
                left = (maxw - s.width()) // 2
                right = maxw - left - s.width()
                s = prettyForm(*s.left(' ' * left))
                s = prettyForm(*s.right(' ' * right))
                vp[(j, i)] = s

        D1 = prettyForm(*vp[(0, 0)].right('  ', vp[(0, 1)]))
        D1 = prettyForm(*D1.below(' '))
        D2 = prettyForm(*vp[(1, 0)].right('  ', vp[(1, 1)]))
        D = prettyForm(*D1.below(D2))

        # make sure that the argument `z' is centred vertically
        D.baseline = D.height()//2

        # insert horizontal separator
        P = prettyForm(*P.left(' '))
        D = prettyForm(*D.right(' '))

        # insert separating `|`
        D = self._hprint_vseparator(D, P)

        # add parens
        D = prettyForm(*D.parens('(', ')'))

        # create the G symbol
        above = D.height()//2 - 1
        below = D.height() - above - 1

        sz, t, b, add, img = annotated('G')
        F = prettyForm('\n' * (above - t) + img + '\n' * (below - b),
                       baseline=above + sz)

        pp = self._print(len(e.ap))
        pq = self._print(len(e.bq))
        pm = self._print(len(e.bm))
        pn = self._print(len(e.an))

        def adjust(p1, p2):
            diff = p1.width() - p2.width()
            if diff == 0:
                return p1, p2
            elif diff > 0:
                return p1, prettyForm(*p2.left(' '*diff))
            else:
                return prettyForm(*p1.left(' '*-diff)), p2
        pp, pm = adjust(pp, pm)
        pq, pn = adjust(pq, pn)
        pu = prettyForm(*pm.right(', ', pn))
        pl = prettyForm(*pp.right(', ', pq))

        ht = F.baseline - above - 2
        if ht > 0:
            pu = prettyForm(*pu.below('\n'*ht))
        p = prettyForm(*pu.below(pl))

        F.baseline = above
        F = prettyForm(*F.right(p))

        F.baseline = above + add

        D = prettyForm(*F.right(' ', D))

        return D
```
### 62 - sympy/printing/pretty/pretty.py:

Start line: 131, End line: 188

```python
class PrettyPrinter(Printer):

    def _print_Gradient(self, e):
        func = e._expr
        pform = self._print(func)
        pform = prettyForm(*pform.left('('))
        pform = prettyForm(*pform.right(')'))
        pform = prettyForm(*pform.left(self._print(U('DOT OPERATOR'))))
        pform = prettyForm(*pform.left(self._print(U('NABLA'))))
        return pform

    def _print_Atom(self, e):
        try:
            # print atoms like Exp1 or Pi
            return prettyForm(pretty_atom(e.__class__.__name__))
        except KeyError:
            return self.emptyPrinter(e)

    # Infinity inherits from Number, so we have to override _print_XXX order
    _print_Infinity = _print_Atom
    _print_NegativeInfinity = _print_Atom
    _print_EmptySet = _print_Atom
    _print_Naturals = _print_Atom
    _print_Naturals0 = _print_Atom
    _print_Integers = _print_Atom
    _print_Complexes = _print_Atom

    def _print_Reals(self, e):
        if self._use_unicode:
            return self._print_Atom(e)
        else:
            inf_list = ['-oo', 'oo']
            return self._print_seq(inf_list, '(', ')')

    def _print_subfactorial(self, e):
        x = e.args[0]
        pform = self._print(x)
        # Add parentheses if needed
        if not ((x.is_Integer and x.is_nonnegative) or x.is_Symbol):
            pform = prettyForm(*pform.parens())
        pform = prettyForm(*pform.left('!'))
        return pform

    def _print_factorial(self, e):
        x = e.args[0]
        pform = self._print(x)
        # Add parentheses if needed
        if not ((x.is_Integer and x.is_nonnegative) or x.is_Symbol):
            pform = prettyForm(*pform.parens())
        pform = prettyForm(*pform.right('!'))
        return pform

    def _print_factorial2(self, e):
        x = e.args[0]
        pform = self._print(x)
        # Add parentheses if needed
        if not ((x.is_Integer and x.is_nonnegative) or x.is_Symbol):
            pform = prettyForm(*pform.parens())
        pform = prettyForm(*pform.right('!!'))
        return pform
```
