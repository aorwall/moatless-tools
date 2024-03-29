# sympy__sympy-19110

| **sympy/sympy** | `542a1758e517c3b5e95e480dcd49b9b24a01f191` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | - |
| **Avg pos** | - |
| **Min pos** | - |
| **Max pos** | - |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sympy/matrices/expressions/matexpr.py b/sympy/matrices/expressions/matexpr.py
--- a/sympy/matrices/expressions/matexpr.py
+++ b/sympy/matrices/expressions/matexpr.py
@@ -996,12 +996,6 @@ def conjugate(self):
     def _entry(self, i, j, **kwargs):
         return S.Zero
 
-    def __nonzero__(self):
-        return False
-
-    __bool__ = __nonzero__
-
-
 class GenericZeroMatrix(ZeroMatrix):
     """
     A zero matrix without a specified shape

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sympy/matrices/expressions/matexpr.py | 999 | 1004 | - | 1 | -


## Problem Statement

```
ZeroMatrix should not be falsey
We have:
\`\`\`julia
In [10]: Z = ZeroMatrix(2, 3)                                                                                                     

In [11]: Ze = Z.as_explicit()                                                                                                     

In [12]: Z                                                                                                                        
Out[12]: 𝟘

In [13]: Ze                                                                                                                       
Out[13]: 
⎡0  0  0⎤
⎢       ⎥
⎣0  0  0⎦

In [14]: bool(Z)                                                                                                                  
Out[14]: False

In [15]: bool(Ze)                                                                                                                 
Out[15]: True
\`\`\`
I don't see any sense in having a ZeroMatrix instance evaluate to False. This happens because of the `__nonzero__` method defined for `ZeroMatrix`:
https://github.com/sympy/sympy/blob/542a1758e517c3b5e95e480dcd49b9b24a01f191/sympy/matrices/expressions/matexpr.py#L999-L1002
The `__nonzero__` method is not needed now that Python 2 is not supported. The `__bool__` method is not needed because a `ZeroMatrix` should not evaluate to False in a boolean context.

The linked lines of code should simply be removed.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 sympy/matrices/expressions/matexpr.py** | 943 | 1002| 395 | 395 | 9351 | 
| 2 | **1 sympy/matrices/expressions/matexpr.py** | 1005 | 1037| 229 | 624 | 9351 | 
| 3 | 2 sympy/matrices/common.py | 1651 | 1684| 268 | 892 | 30296 | 
| 4 | 3 sympy/matrices/reductions.py | 136 | 149| 154 | 1046 | 32988 | 
| 5 | 4 sympy/matrices/dense.py | 1 | 26| 215 | 1261 | 42944 | 
| 6 | 5 sympy/matrices/matrices.py | 670 | 708| 285 | 1546 | 62522 | 
| 7 | **5 sympy/matrices/expressions/matexpr.py** | 1040 | 1109| 462 | 2008 | 62522 | 
| 8 | 5 sympy/matrices/matrices.py | 1982 | 2014| 233 | 2241 | 62522 | 
| 9 | 6 sympy/matrices/expressions/blockmatrix.py | 262 | 283| 178 | 2419 | 67911 | 
| 10 | 6 sympy/matrices/common.py | 1089 | 1108| 123 | 2542 | 67911 | 
| 11 | 6 sympy/matrices/expressions/blockmatrix.py | 1 | 22| 251 | 2793 | 67911 | 
| 12 | 6 sympy/matrices/common.py | 2659 | 2715| 463 | 3256 | 67911 | 
| 13 | 6 sympy/matrices/matrices.py | 2251 | 2294| 572 | 3828 | 67911 | 
| 14 | 7 sympy/assumptions/ask.py | 831 | 861| 247 | 4075 | 78662 | 
| 15 | 8 sympy/logic/boolalg.py | 343 | 409| 365 | 4440 | 103533 | 
| 16 | 8 sympy/matrices/matrices.py | 2165 | 2249| 788 | 5228 | 103533 | 
| 17 | 9 sympy/matrices/sparse.py | 28 | 112| 705 | 5933 | 112827 | 
| 18 | 9 sympy/matrices/matrices.py | 1 | 63| 664 | 6597 | 112827 | 
| 19 | 9 sympy/matrices/common.py | 2558 | 2612| 454 | 7051 | 112827 | 
| 20 | 9 sympy/logic/boolalg.py | 224 | 340| 947 | 7998 | 112827 | 
| 21 | 10 sympy/matrices/__init__.py | 1 | 64| 625 | 8623 | 113452 | 
| 22 | **10 sympy/matrices/expressions/matexpr.py** | 376 | 399| 150 | 8773 | 113452 | 
| 23 | 10 sympy/matrices/dense.py | 40 | 58| 146 | 8919 | 113452 | 
| 24 | 11 sympy/assumptions/handlers/matrices.py | 327 | 365| 261 | 9180 | 117914 | 
| 25 | 12 sympy/integrals/meijerint.py | 1012 | 1053| 628 | 9808 | 142212 | 
| 26 | 12 sympy/matrices/matrices.py | 1761 | 1788| 242 | 10050 | 142212 | 
| 27 | 12 sympy/matrices/dense.py | 170 | 198| 296 | 10346 | 142212 | 
| 28 | **12 sympy/matrices/expressions/matexpr.py** | 832 | 844| 157 | 10503 | 142212 | 
| 29 | 12 sympy/matrices/common.py | 1393 | 1426| 263 | 10766 | 142212 | 
| 30 | 12 sympy/assumptions/handlers/matrices.py | 49 | 100| 371 | 11137 | 142212 | 
| 31 | 12 sympy/matrices/sparse.py | 352 | 373| 228 | 11365 | 142212 | 
| 32 | **12 sympy/matrices/expressions/matexpr.py** | 446 | 470| 240 | 11605 | 142212 | 
| 33 | 12 sympy/matrices/sparse.py | 540 | 580| 317 | 11922 | 142212 | 
| 34 | 12 sympy/matrices/expressions/blockmatrix.py | 196 | 215| 201 | 12123 | 142212 | 
| 35 | 13 sympy/matrices/eigen.py | 731 | 812| 546 | 12669 | 151065 | 
| 36 | 14 sympy/codegen/rewriting.py | 232 | 255| 197 | 12866 | 153125 | 
| 37 | 15 sympy/matrices/expressions/__init__.py | 1 | 57| 408 | 13274 | 153533 | 
| 38 | **15 sympy/matrices/expressions/matexpr.py** | 128 | 231| 792 | 14066 | 153533 | 
| 39 | 16 sympy/matrices/expressions/matmul.py | 387 | 407| 184 | 14250 | 156902 | 
| 40 | 16 sympy/matrices/matrices.py | 1408 | 1432| 293 | 14543 | 156902 | 
| 41 | 16 sympy/matrices/matrices.py | 1935 | 1980| 459 | 15002 | 156902 | 
| 42 | **16 sympy/matrices/expressions/matexpr.py** | 472 | 603| 1256 | 16258 | 156902 | 
| 43 | 16 sympy/matrices/expressions/blockmatrix.py | 133 | 194| 469 | 16727 | 156902 | 
| 44 | 16 sympy/matrices/expressions/blockmatrix.py | 240 | 260| 210 | 16937 | 156902 | 
| 45 | 16 sympy/matrices/common.py | 153 | 205| 404 | 17341 | 156902 | 
| 46 | 17 sympy/polys/multivariate_resultants.py | 214 | 239| 224 | 17565 | 160604 | 
| 47 | 17 sympy/matrices/common.py | 1245 | 1314| 631 | 18196 | 160604 | 
| 48 | 18 sympy/matrices/expressions/inverse.py | 84 | 108| 175 | 18371 | 161325 | 
| 49 | 18 sympy/matrices/expressions/matmul.py | 146 | 185| 354 | 18725 | 161325 | 
| 50 | 19 sympy/matrices/determinant.py | 629 | 671| 432 | 19157 | 168560 | 
| 51 | 19 sympy/matrices/matrices.py | 759 | 781| 172 | 19329 | 168560 | 
| 52 | 19 sympy/matrices/expressions/matmul.py | 1 | 16| 133 | 19462 | 168560 | 
| 53 | 19 sympy/assumptions/ask.py | 465 | 503| 285 | 19747 | 168560 | 
| 54 | 19 sympy/matrices/matrices.py | 97 | 146| 577 | 20324 | 168560 | 
| 55 | **19 sympy/matrices/expressions/matexpr.py** | 36 | 126| 714 | 21038 | 168560 | 
| 56 | 20 sympy/matrices/expressions/matpow.py | 60 | 102| 386 | 21424 | 169816 | 
| 57 | 20 sympy/matrices/sparse.py | 220 | 274| 465 | 21889 | 169816 | 
| 58 | 21 sympy/matrices/inverse.py | 189 | 223| 256 | 22145 | 173148 | 
| 59 | 21 sympy/matrices/matrices.py | 851 | 868| 207 | 22352 | 173148 | 
| 60 | 21 sympy/matrices/matrices.py | 817 | 849| 343 | 22695 | 173148 | 
| 61 | 21 sympy/matrices/common.py | 1 | 53| 325 | 23020 | 173148 | 
| 62 | 22 sympy/matrices/expressions/funcmatrix.py | 82 | 121| 300 | 23320 | 174122 | 
| 63 | 22 sympy/matrices/dense.py | 262 | 291| 262 | 23582 | 174122 | 
| 64 | 22 sympy/assumptions/ask.py | 734 | 762| 229 | 23811 | 174122 | 
| 65 | **22 sympy/matrices/expressions/matexpr.py** | 760 | 830| 502 | 24313 | 174122 | 
| 66 | 23 sympy/matrices/immutable.py | 26 | 136| 927 | 25240 | 175704 | 
| 67 | 23 sympy/matrices/expressions/matmul.py | 410 | 445| 229 | 25469 | 175704 | 
| 68 | 23 sympy/assumptions/ask.py | 1055 | 1078| 140 | 25609 | 175704 | 
| 69 | 23 sympy/matrices/matrices.py | 364 | 435| 732 | 26341 | 175704 | 
| 70 | 23 sympy/matrices/expressions/blockmatrix.py | 80 | 131| 525 | 26866 | 175704 | 
| 71 | 24 sympy/assumptions/refine.py | 337 | 358| 186 | 27052 | 178630 | 
| 72 | 25 sympy/matrices/utilities.py | 1 | 49| 468 | 27520 | 179442 | 
| 73 | 25 sympy/matrices/determinant.py | 1 | 15| 132 | 27652 | 179442 | 
| 74 | 25 sympy/matrices/sparse.py | 653 | 679| 291 | 27943 | 179442 | 
| 75 | 25 sympy/matrices/determinant.py | 98 | 119| 260 | 28203 | 179442 | 
| 76 | 26 sympy/assumptions/handlers/order.py | 138 | 185| 294 | 28497 | 181781 | 
| 77 | 27 sympy/core/symbol.py | 176 | 194| 176 | 28673 | 188247 | 
| 78 | **27 sympy/matrices/expressions/matexpr.py** | 908 | 940| 219 | 28892 | 188247 | 
| 79 | 27 sympy/matrices/common.py | 2304 | 2317| 144 | 29036 | 188247 | 
| 80 | 27 sympy/matrices/expressions/matmul.py | 70 | 108| 366 | 29402 | 188247 | 
| 81 | **27 sympy/matrices/expressions/matexpr.py** | 698 | 728| 246 | 29648 | 188247 | 


## Patch

```diff
diff --git a/sympy/matrices/expressions/matexpr.py b/sympy/matrices/expressions/matexpr.py
--- a/sympy/matrices/expressions/matexpr.py
+++ b/sympy/matrices/expressions/matexpr.py
@@ -996,12 +996,6 @@ def conjugate(self):
     def _entry(self, i, j, **kwargs):
         return S.Zero
 
-    def __nonzero__(self):
-        return False
-
-    __bool__ = __nonzero__
-
-
 class GenericZeroMatrix(ZeroMatrix):
     """
     A zero matrix without a specified shape

```

## Test Patch

```diff
diff --git a/sympy/matrices/expressions/tests/test_matexpr.py b/sympy/matrices/expressions/tests/test_matexpr.py
--- a/sympy/matrices/expressions/tests/test_matexpr.py
+++ b/sympy/matrices/expressions/tests/test_matexpr.py
@@ -127,7 +127,7 @@ def test_ZeroMatrix():
     assert Z*A.T == ZeroMatrix(n, n)
     assert A - A == ZeroMatrix(*A.shape)
 
-    assert not Z
+    assert Z
 
     assert transpose(Z) == ZeroMatrix(m, n)
     assert Z.conjugate() == Z

```


## Code snippets

### 1 - sympy/matrices/expressions/matexpr.py:

Start line: 943, End line: 1002

```python
class ZeroMatrix(MatrixExpr):
    """The Matrix Zero 0 - additive identity

    Examples
    ========

    >>> from sympy import MatrixSymbol, ZeroMatrix
    >>> A = MatrixSymbol('A', 3, 5)
    >>> Z = ZeroMatrix(3, 5)
    >>> A + Z
    A
    >>> Z*A.T
    0
    """
    is_ZeroMatrix = True

    def __new__(cls, m, n):
        m, n = _sympify(m), _sympify(n)
        cls._check_dim(m)
        cls._check_dim(n)

        return super(ZeroMatrix, cls).__new__(cls, m, n)

    @property
    def shape(self):
        return (self.args[0], self.args[1])

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__rpow__')
    def __pow__(self, other):
        if other != 1 and not self.is_square:
            raise NonSquareMatrixError("Power of non-square matrix %s" % self)
        if other == 0:
            return Identity(self.rows)
        if other < 1:
            raise NonInvertibleMatrixError("Matrix det == 0; not invertible")
        return self

    def _eval_transpose(self):
        return ZeroMatrix(self.cols, self.rows)

    def _eval_trace(self):
        return S.Zero

    def _eval_determinant(self):
        return S.Zero

    def _eval_inverse(self):
        raise NonInvertibleMatrixError("Matrix det == 0; not invertible.")

    def conjugate(self):
        return self

    def _entry(self, i, j, **kwargs):
        return S.Zero

    def __nonzero__(self):
        return False

    __bool__ = __nonzero__
```
### 2 - sympy/matrices/expressions/matexpr.py:

Start line: 1005, End line: 1037

```python
class GenericZeroMatrix(ZeroMatrix):
    """
    A zero matrix without a specified shape

    This exists primarily so MatAdd() with no arguments can return something
    meaningful.
    """
    def __new__(cls):
        # super(ZeroMatrix, cls) instead of super(GenericZeroMatrix, cls)
        # because ZeroMatrix.__new__ doesn't have the same signature
        return super(ZeroMatrix, cls).__new__(cls)

    @property
    def rows(self):
        raise TypeError("GenericZeroMatrix does not have a specified shape")

    @property
    def cols(self):
        raise TypeError("GenericZeroMatrix does not have a specified shape")

    @property
    def shape(self):
        raise TypeError("GenericZeroMatrix does not have a specified shape")

    # Avoid Matrix.__eq__ which might call .shape
    def __eq__(self, other):
        return isinstance(other, GenericZeroMatrix)

    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        return super(GenericZeroMatrix, self).__hash__()
```
### 3 - sympy/matrices/common.py:

Start line: 1651, End line: 1684

```python
class MatrixProperties(MatrixRequired):

    @property
    def is_zero_matrix(self):
        """Checks if a matrix is a zero matrix.

        A matrix is zero if every element is zero.  A matrix need not be square
        to be considered zero.  The empty matrix is zero by the principle of
        vacuous truth.  For a matrix that may or may not be zero (e.g.
        contains a symbol), this will be None

        Examples
        ========

        >>> from sympy import Matrix, zeros
        >>> from sympy.abc import x
        >>> a = Matrix([[0, 0], [0, 0]])
        >>> b = zeros(3, 4)
        >>> c = Matrix([[0, 1], [0, 0]])
        >>> d = Matrix([])
        >>> e = Matrix([[x, 0], [0, 0]])
        >>> a.is_zero_matrix
        True
        >>> b.is_zero_matrix
        True
        >>> c.is_zero_matrix
        False
        >>> d.is_zero_matrix
        True
        >>> e.is_zero_matrix
        """
        return self._eval_is_zero_matrix()

    def values(self):
        """Return non-zero values of self."""
        return self._eval_values()
```
### 4 - sympy/matrices/reductions.py:

Start line: 136, End line: 149

```python
def _is_echelon(M, iszerofunc=_iszero):
    """Returns `True` if the matrix is in echelon form. That is, all rows of
    zeros are at the bottom, and below each leading non-zero in a row are
    exclusively zeros."""

    if M.rows <= 0 or M.cols <= 0:
        return True

    zeros_below = all(iszerofunc(t) for t in M[1:, 0])

    if iszerofunc(M[0, 0]):
        return zeros_below and _is_echelon(M[:, 1:], iszerofunc)

    return zeros_below and _is_echelon(M[1:, 1:], iszerofunc)
```
### 5 - sympy/matrices/dense.py:

Start line: 1, End line: 26

```python
from __future__ import division, print_function

import random

from sympy.core import SympifyError, Add
from sympy.core.basic import Basic
from sympy.core.compatibility import is_sequence, reduce
from sympy.core.expr import Expr
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.matrices.common import \
    a2idx, classof, ShapeError
from sympy.matrices.matrices import MatrixBase
from sympy.simplify.simplify import simplify as _simplify
from sympy.utilities.decorator import doctest_depends_on
from sympy.utilities.misc import filldedent

from .decompositions import _cholesky, _LDLdecomposition
from .solvers import _lower_triangular_solve, _upper_triangular_solve


def _iszero(x):
    """Returns True if x is zero."""
    return x.is_zero
```
### 6 - sympy/matrices/matrices.py:

Start line: 670, End line: 708

```python
class MatrixDeprecated(MatrixCommon):

    def berkowitz(self):
        from sympy.matrices import zeros
        berk = ((1,),)
        if not self:
            return berk

        if not self.is_square:
            raise NonSquareMatrixError()

        A, N = self, self.rows
        transforms = [0] * (N - 1)

        for n in range(N, 1, -1):
            T, k = zeros(n + 1, n), n - 1

            R, C = -A[k, :k], A[:k, k]
            A, a = A[:k, :k], -A[k, k]

            items = [C]

            for i in range(0, n - 2):
                items.append(A * items[i])

            for i, B in enumerate(items):
                items[i] = (R * B)[0, 0]

            items = [self.one, a] + items

            for i in range(n):
                T[i:, i] = items[:n - i + 1]

            transforms[k - 1] = T

        polys = [self._new([self.one, -A[0, 0]])]

        for i, T in enumerate(transforms):
            polys.append(T * polys[i])

        return berk + tuple(map(tuple, polys))
```
### 7 - sympy/matrices/expressions/matexpr.py:

Start line: 1040, End line: 1109

```python
class OneMatrix(MatrixExpr):
    """
    Matrix whose all entries are ones.
    """
    def __new__(cls, m, n, evaluate=False):
        m, n = _sympify(m), _sympify(n)
        cls._check_dim(m)
        cls._check_dim(n)

        if evaluate:
            condition = Eq(m, 1) & Eq(n, 1)
            if condition == True:
                return Identity(1)

        obj = super(OneMatrix, cls).__new__(cls, m, n)
        return obj

    @property
    def shape(self):
        return self._args

    def as_explicit(self):
        from sympy import ImmutableDenseMatrix
        return ImmutableDenseMatrix.ones(*self.shape)

    def doit(self, **hints):
        args = self.args
        if hints.get('deep', True):
            args = [a.doit(**hints) for a in args]
        return self.func(*args, evaluate=True)

    def _eval_transpose(self):
        return OneMatrix(self.cols, self.rows)

    def _eval_trace(self):
        return S.One*self.rows

    def _is_1x1(self):
        """Returns true if the matrix is known to be 1x1"""
        shape = self.shape
        return Eq(shape[0], 1) & Eq(shape[1], 1)

    def _eval_determinant(self):
        condition = self._is_1x1()
        if condition == True:
            return S.One
        elif condition == False:
            return S.Zero
        else:
            from sympy import Determinant
            return Determinant(self)

    def _eval_inverse(self):
        condition = self._is_1x1()
        if condition == True:
            return Identity(1)
        elif condition == False:
            raise NonInvertibleMatrixError("Matrix det == 0; not invertible.")
        else:
            return Inverse(self)

    def conjugate(self):
        return self

    def _entry(self, i, j, **kwargs):
        return S.One


def matrix_symbols(expr):
    return [sym for sym in expr.free_symbols if sym.is_Matrix]
```
### 8 - sympy/matrices/matrices.py:

Start line: 1982, End line: 2014

```python
class MatrixBase(MatrixDeprecated,
                 MatrixCalculus,
                 MatrixEigen,
                 MatrixCommon):

    def print_nonzero(self, symb="X"):
        """Shows location of non-zero entries for fast shape lookup.

        Examples
        ========

        >>> from sympy.matrices import Matrix, eye
        >>> m = Matrix(2, 3, lambda i, j: i*3+j)
        >>> m
        Matrix([
        [0, 1, 2],
        [3, 4, 5]])
        >>> m.print_nonzero()
        [ XX]
        [XXX]
        >>> m = eye(4)
        >>> m.print_nonzero("x")
        [x   ]
        [ x  ]
        [  x ]
        [   x]

        """
        s = []
        for i in range(self.rows):
            line = []
            for j in range(self.cols):
                if self[i, j] == 0:
                    line.append(" ")
                else:
                    line.append(str(symb))
            s.append("[%s]" % ''.join(line))
        print('\n'.join(s))
```
### 9 - sympy/matrices/expressions/blockmatrix.py:

Start line: 262, End line: 283

```python
class BlockMatrix(MatrixExpr):

    @property
    def is_Identity(self):
        if self.blockshape[0] != self.blockshape[1]:
            return False
        for i in range(self.blockshape[0]):
            for j in range(self.blockshape[1]):
                if i==j and not self.blocks[i, j].is_Identity:
                    return False
                if i!=j and not self.blocks[i, j].is_ZeroMatrix:
                    return False
        return True

    @property
    def is_structurally_symmetric(self):
        return self.rowblocksizes == self.colblocksizes

    def equals(self, other):
        if self == other:
            return True
        if (isinstance(other, BlockMatrix) and self.blocks == other.blocks):
            return True
        return super(BlockMatrix, self).equals(other)
```
### 10 - sympy/matrices/common.py:

Start line: 1089, End line: 1108

```python
class MatrixSpecial(MatrixRequired):

    @classmethod
    def zeros(kls, rows, cols=None, **kwargs):
        """Returns a matrix of zeros.

        Args
        ====

        rows : rows of the matrix
        cols : cols of the matrix (if None, cols=rows)

        kwargs
        ======
        cls : class of the returned matrix
        """
        if cols is None:
            cols = rows
        klass = kwargs.get('cls', kls)
        rows, cols = as_int(rows), as_int(cols)

        return klass._eval_zeros(rows, cols)
```
### 22 - sympy/matrices/expressions/matexpr.py:

Start line: 376, End line: 399

```python
class MatrixExpr(Expr):

    def __array__(self):
        from numpy import empty
        a = empty(self.shape, dtype=object)
        for i in range(self.rows):
            for j in range(self.cols):
                a[i, j] = self[i, j]
        return a

    def equals(self, other):
        """
        Test elementwise equality between matrices, potentially of different
        types

        >>> from sympy import Identity, eye
        >>> Identity(3).equals(eye(3))
        True
        """
        return self.as_explicit().equals(other)

    def canonicalize(self):
        return self

    def as_coeff_mmul(self):
        return 1, MatMul(self)
```
### 28 - sympy/matrices/expressions/matexpr.py:

Start line: 832, End line: 844

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
```
### 32 - sympy/matrices/expressions/matexpr.py:

Start line: 446, End line: 470

```python
class MatrixExpr(Expr):

    @staticmethod
    def from_index_summation(expr, first_index=None, last_index=None, dimensions=None):
        # ... other code

        def remove_matelement(expr, i1, i2):

            def repl_match(pos):
                def func(x):
                    if not isinstance(x, MatrixElement):
                        return False
                    if x.args[pos] != i1:
                        return False
                    if x.args[3-pos] == 0:
                        if x.args[0].shape[2-pos] == 1:
                            return True
                        else:
                            return False
                    return True
                return func

            expr = expr.replace(repl_match(1),
                lambda x: x.args[0])
            expr = expr.replace(repl_match(2),
                lambda x: transpose(x.args[0]))

            # Make sure that all Mul are transformed to MatMul and that they
            # are flattened:
            rule = bottom_up(lambda x: reduce(lambda a, b: a*b, x.args) if isinstance(x, (Mul, MatMul)) else x)
            return rule(expr)
        # ... other code
```
### 38 - sympy/matrices/expressions/matexpr.py:

Start line: 128, End line: 231

```python
class MatrixExpr(Expr):

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__rpow__')
    def __pow__(self, other):
        if not self.is_square:
            raise NonSquareMatrixError("Power of non-square matrix %s" % self)
        elif self.is_Identity:
            return self
        elif other == S.Zero:
            return Identity(self.rows)
        elif other == S.One:
            return self
        return MatPow(self, other).doit(deep=False)

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

    __truediv__ = __div__  # type: Callable[[MatrixExpr, Any], Any]
    __rtruediv__ = __rdiv__  # type: Callable[[MatrixExpr, Any], Any]

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

    def as_real_imag(self, deep=True, **hints):
        from sympy import I
        real = S.Half * (self + self._eval_conjugate())
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
            return self.func(*[simplify(x, **kwargs) for x in self.args])

    def _eval_adjoint(self):
        from sympy.matrices.expressions.adjoint import Adjoint
        return Adjoint(self)

    def _eval_derivative_array(self, x):
        if isinstance(x, MatrixExpr):
            return _matrix_derivative(self, x)
        else:
            return self._eval_derivative(x)

    def _eval_derivative_n_times(self, x, n):
        return Basic._eval_derivative_n_times(self, x, n)

    def _visit_eval_derivative_scalar(self, x):
        # `x` is a scalar:
        if x.has(self):
            return _matrix_derivative(x, self)
        else:
            return ZeroMatrix(*self.shape)

    def _visit_eval_derivative_array(self, x):
        if x.has(self):
            return _matrix_derivative(x, self)
        else:
            from sympy import Derivative
            return Derivative(x, self)

    def _accept_eval_derivative(self, s):
        from sympy import MatrixBase, NDimArray
        if isinstance(s, (MatrixBase, NDimArray, MatrixExpr)):
            return s._visit_eval_derivative_array(self)
        else:
            return s._visit_eval_derivative_scalar(self)
```
### 42 - sympy/matrices/expressions/matexpr.py:

Start line: 472, End line: 603

```python
class MatrixExpr(Expr):

    @staticmethod
    def from_index_summation(expr, first_index=None, last_index=None, dimensions=None):
        # ... other code

        def recurse_expr(expr, index_ranges={}):
            if expr.is_Mul:
                nonmatargs = []
                pos_arg = []
                pos_ind = []
                dlinks = {}
                link_ind = []
                counter = 0
                args_ind = []
                for arg in expr.args:
                    retvals = recurse_expr(arg, index_ranges)
                    assert isinstance(retvals, list)
                    if isinstance(retvals, list):
                        for i in retvals:
                            args_ind.append(i)
                    else:
                        args_ind.append(retvals)
                for arg_symbol, arg_indices in args_ind:
                    if arg_indices is None:
                        nonmatargs.append(arg_symbol)
                        continue
                    if isinstance(arg_symbol, MatrixElement):
                        arg_symbol = arg_symbol.args[0]
                    pos_arg.append(arg_symbol)
                    pos_ind.append(arg_indices)
                    link_ind.append([None]*len(arg_indices))
                    for i, ind in enumerate(arg_indices):
                        if ind in dlinks:
                            other_i = dlinks[ind]
                            link_ind[counter][i] = other_i
                            link_ind[other_i[0]][other_i[1]] = (counter, i)
                        dlinks[ind] = (counter, i)
                    counter += 1
                counter2 = 0
                lines = {}
                while counter2 < len(link_ind):
                    for i, e in enumerate(link_ind):
                        if None in e:
                            line_start_index = (i, e.index(None))
                            break
                    cur_ind_pos = line_start_index
                    cur_line = []
                    index1 = pos_ind[cur_ind_pos[0]][cur_ind_pos[1]]
                    while True:
                        d, r = cur_ind_pos
                        if pos_arg[d] != 1:
                            if r % 2 == 1:
                                cur_line.append(transpose(pos_arg[d]))
                            else:
                                cur_line.append(pos_arg[d])
                        next_ind_pos = link_ind[d][1-r]
                        counter2 += 1
                        # Mark as visited, there will be no `None` anymore:
                        link_ind[d] = (-1, -1)
                        if next_ind_pos is None:
                            index2 = pos_ind[d][1-r]
                            lines[(index1, index2)] = cur_line
                            break
                        cur_ind_pos = next_ind_pos
                lines = {k: MatMul.fromiter(v) if len(v) != 1 else v[0] for k, v in lines.items()}
                return [(Mul.fromiter(nonmatargs), None)] + [
                    (MatrixElement(a, i, j), (i, j)) for (i, j), a in lines.items()
                ]
            elif expr.is_Add:
                res = [recurse_expr(i) for i in expr.args]
                d = collections.defaultdict(list)
                for res_addend in res:
                    scalar = 1
                    for elem, indices in res_addend:
                        if indices is None:
                            scalar = elem
                            continue
                        indices = tuple(sorted(indices, key=default_sort_key))
                        d[indices].append(scalar*remove_matelement(elem, *indices))
                        scalar = 1
                return [(MatrixElement(Add.fromiter(v), *k), k) for k, v in d.items()]
            elif isinstance(expr, KroneckerDelta):
                i1, i2 = expr.args
                if dimensions is not None:
                    identity = Identity(dimensions[0])
                else:
                    identity = S.One
                return [(MatrixElement(identity, i1, i2), (i1, i2))]
            elif isinstance(expr, MatrixElement):
                matrix_symbol, i1, i2 = expr.args
                if i1 in index_ranges:
                    r1, r2 = index_ranges[i1]
                    if r1 != 0 or matrix_symbol.shape[0] != r2+1:
                        raise ValueError("index range mismatch: {0} vs. (0, {1})".format(
                            (r1, r2), matrix_symbol.shape[0]))
                if i2 in index_ranges:
                    r1, r2 = index_ranges[i2]
                    if r1 != 0 or matrix_symbol.shape[1] != r2+1:
                        raise ValueError("index range mismatch: {0} vs. (0, {1})".format(
                            (r1, r2), matrix_symbol.shape[1]))
                if (i1 == i2) and (i1 in index_ranges):
                    return [(trace(matrix_symbol), None)]
                return [(MatrixElement(matrix_symbol, i1, i2), (i1, i2))]
            elif isinstance(expr, Sum):
                return recurse_expr(
                    expr.args[0],
                    index_ranges={i[0]: i[1:] for i in expr.args[1:]}
                )
            else:
                return [(expr, None)]

        retvals = recurse_expr(expr)
        factors, indices = zip(*retvals)
        retexpr = Mul.fromiter(factors)
        if len(indices) == 0 or list(set(indices)) == [None]:
            return retexpr
        if first_index is None:
            for i in indices:
                if i is not None:
                    ind0 = i
                    break
            return remove_matelement(retexpr, *ind0)
        else:
            return remove_matelement(retexpr, first_index, last_index)

    def applyfunc(self, func):
        from .applyfunc import ElementwiseApplyFunction
        return ElementwiseApplyFunction(func, self)

    def _eval_Eq(self, other):
        if not isinstance(other, MatrixExpr):
            return False
        if self.shape != other.shape:
            return False
        if (self - other).is_ZeroMatrix:
            return True
        return Eq(self, other, evaluate=False)
```
### 55 - sympy/matrices/expressions/matexpr.py:

Start line: 36, End line: 126

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
```
### 65 - sympy/matrices/expressions/matexpr.py:

Start line: 760, End line: 830

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

    def _hashable_content(self):
        return (self.name, self.shape)

    @property
    def shape(self):
        return self.args[1:3]

    @property
    def name(self):
        return self.args[0].name

    def _eval_subs(self, old, new):
        # only do substitutions in shape
        shape = Tuple(*self.shape)._subs(old, new)
        return MatrixSymbol(self.args[0], *shape)

    def __call__(self, *args):
        raise TypeError("%s object is not callable" % self.__class__)

    def _entry(self, i, j, **kwargs):
        return MatrixElement(self, i, j)

    @property
    def free_symbols(self):
        return set((self,))

    def doit(self, **hints):
        if hints.get('deep', True):
            return type(self)(self.args[0], self.args[1].doit(**hints),
                    self.args[2].doit(**hints))
        else:
            return self

    def _eval_simplify(self, **kwargs):
        return self

    def _eval_derivative(self, x):
        # x is a scalar:
        return ZeroMatrix(self.shape[0], self.shape[1])
```
### 78 - sympy/matrices/expressions/matexpr.py:

Start line: 908, End line: 940

```python
class GenericIdentity(Identity):
    """
    An identity matrix without a specified shape

    This exists primarily so MatMul() with no arguments can return something
    meaningful.
    """
    def __new__(cls):
        # super(Identity, cls) instead of super(GenericIdentity, cls) because
        # Identity.__new__ doesn't have the same signature
        return super(Identity, cls).__new__(cls)

    @property
    def rows(self):
        raise TypeError("GenericIdentity does not have a specified shape")

    @property
    def cols(self):
        raise TypeError("GenericIdentity does not have a specified shape")

    @property
    def shape(self):
        raise TypeError("GenericIdentity does not have a specified shape")

    # Avoid Matrix.__eq__ which might call .shape
    def __eq__(self, other):
        return isinstance(other, GenericIdentity)

    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        return super(GenericIdentity, self).__hash__()
```
### 81 - sympy/matrices/expressions/matexpr.py:

Start line: 698, End line: 728

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
