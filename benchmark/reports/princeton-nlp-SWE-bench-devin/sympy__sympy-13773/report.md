# sympy__sympy-13773

| **sympy/sympy** | `7121bdf1facdd90d05b6994b4c2e5b2865a4638a` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 19053 |
| **Any found context length** | 602 |
| **Avg pos** | 24.0 |
| **Min pos** | 2 |
| **Max pos** | 22 |
| **Top file pos** | 1 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sympy/matrices/common.py b/sympy/matrices/common.py
--- a/sympy/matrices/common.py
+++ b/sympy/matrices/common.py
@@ -1973,6 +1973,10 @@ def __div__(self, other):
 
     @call_highest_priority('__rmatmul__')
     def __matmul__(self, other):
+        other = _matrixify(other)
+        if not getattr(other, 'is_Matrix', False) and not getattr(other, 'is_MatrixLike', False):
+            return NotImplemented
+
         return self.__mul__(other)
 
     @call_highest_priority('__rmul__')
@@ -2066,6 +2070,10 @@ def __radd__(self, other):
 
     @call_highest_priority('__matmul__')
     def __rmatmul__(self, other):
+        other = _matrixify(other)
+        if not getattr(other, 'is_Matrix', False) and not getattr(other, 'is_MatrixLike', False):
+            return NotImplemented
+
         return self.__rmul__(other)
 
     @call_highest_priority('__mul__')

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sympy/matrices/common.py | 1976 | 1976 | 22 | 1 | 19053
| sympy/matrices/common.py | 2069 | 2069 | 2 | 1 | 602


## Problem Statement

```
@ (__matmul__) should fail if one argument is not a matrix
\`\`\`
>>> A = Matrix([[1, 2], [3, 4]])
>>> B = Matrix([[2, 3], [1, 2]])
>>> A@B
Matrix([
[ 4,  7],
[10, 17]])
>>> 2@B
Matrix([
[4, 6],
[2, 4]])
\`\`\`

Right now `@` (`__matmul__`) just copies `__mul__`, but it should actually only work if the multiplication is actually a matrix multiplication. 

This is also how NumPy works

\`\`\`
>>> import numpy as np
>>> a = np.array([[1, 2], [3, 4]])
>>> 2*a
array([[2, 4],
       [6, 8]])
>>> 2@a
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: Scalar operands are not allowed, use '*' instead
\`\`\`

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 sympy/matrices/common.py** | 2098 | 2141| 328 | 328 | 17221 | 
| **-> 2 <-** | **1 sympy/matrices/common.py** | 2065 | 2096| 274 | 602 | 17221 | 
| 3 | **1 sympy/matrices/common.py** | 1978 | 2028| 423 | 1025 | 17221 | 
| 4 | 2 sympy/matrices/matrices.py | 1596 | 2419| 6191 | 7216 | 52486 | 
| 5 | **2 sympy/matrices/common.py** | 2030 | 2063| 299 | 7515 | 52486 | 
| 6 | **2 sympy/matrices/common.py** | 1889 | 1917| 288 | 7803 | 52486 | 
| 7 | 3 sympy/matrices/expressions/matmul.py | 78 | 133| 467 | 8270 | 54862 | 
| 8 | 4 sympy/matrices/dense.py | 1303 | 1343| 293 | 8563 | 66360 | 
| 9 | 5 sympy/matrices/sparse.py | 949 | 1009| 468 | 9031 | 76803 | 
| 10 | 5 sympy/matrices/expressions/matmul.py | 15 | 46| 224 | 9255 | 76803 | 
| 11 | **5 sympy/matrices/common.py** | 1919 | 1944| 259 | 9514 | 76803 | 
| 12 | 5 sympy/matrices/sparse.py | 1161 | 1217| 507 | 10021 | 76803 | 
| 13 | 5 sympy/matrices/matrices.py | 3255 | 4046| 6362 | 16383 | 76803 | 
| 14 | 6 sympy/matrices/expressions/matadd.py | 65 | 80| 125 | 16508 | 77718 | 
| 15 | 6 sympy/matrices/expressions/matmul.py | 48 | 76| 316 | 16824 | 77718 | 
| 16 | 6 sympy/matrices/expressions/matmul.py | 158 | 198| 380 | 17204 | 77718 | 
| 17 | 7 sympy/matrices/expressions/matexpr.py | 365 | 388| 150 | 17354 | 83786 | 
| 18 | 7 sympy/matrices/expressions/matmul.py | 136 | 156| 169 | 17523 | 83786 | 
| 19 | 8 sympy/matrices/expressions/dotproduct.py | 1 | 27| 251 | 17774 | 84329 | 
| 20 | 8 sympy/matrices/expressions/matexpr.py | 142 | 202| 443 | 18217 | 84329 | 
| 21 | 8 sympy/matrices/dense.py | 163 | 206| 531 | 18748 | 84329 | 
| **-> 22 <-** | **8 sympy/matrices/common.py** | 1946 | 1976| 305 | 19053 | 84329 | 
| 23 | **8 sympy/matrices/common.py** | 1479 | 1524| 326 | 19379 | 84329 | 
| 24 | 8 sympy/matrices/expressions/matexpr.py | 33 | 140| 794 | 20173 | 84329 | 
| 25 | 8 sympy/matrices/sparse.py | 345 | 368| 245 | 20418 | 84329 | 
| 26 | 9 sympy/matrices/expressions/hadamard.py | 1 | 31| 245 | 20663 | 84970 | 
| 27 | **9 sympy/matrices/common.py** | 1 | 40| 238 | 20901 | 84970 | 
| 28 | 9 sympy/matrices/expressions/dotproduct.py | 29 | 44| 173 | 21074 | 84970 | 
| 29 | 10 sympy/physics/optics/gaussopt.py | 139 | 197| 318 | 21392 | 90665 | 
| 30 | **10 sympy/matrices/common.py** | 2265 | 2289| 198 | 21590 | 90665 | 
| 31 | **10 sympy/matrices/common.py** | 2292 | 2324| 234 | 21824 | 90665 | 
| 32 | 11 sympy/matrices/expressions/blockmatrix.py | 345 | 362| 158 | 21982 | 94343 | 
| 33 | 11 sympy/matrices/expressions/matmul.py | 261 | 296| 229 | 22211 | 94343 | 
| 34 | 11 sympy/matrices/sparse.py | 93 | 148| 465 | 22676 | 94343 | 
| 35 | 11 sympy/matrices/expressions/matmul.py | 1 | 12| 115 | 22791 | 94343 | 
| 36 | 11 sympy/matrices/dense.py | 271 | 282| 130 | 22921 | 94343 | 
| 37 | 11 sympy/matrices/expressions/hadamard.py | 34 | 84| 396 | 23317 | 94343 | 
| 38 | 11 sympy/matrices/expressions/matexpr.py | 390 | 433| 415 | 23732 | 94343 | 
| 39 | 11 sympy/matrices/expressions/matadd.py | 83 | 117| 297 | 24029 | 94343 | 
| 40 | 11 sympy/matrices/expressions/blockmatrix.py | 21 | 126| 821 | 24850 | 94343 | 
| 41 | **11 sympy/matrices/common.py** | 1859 | 1886| 221 | 25071 | 94343 | 
| 42 | 12 sympy/matrices/expressions/matpow.py | 1 | 49| 411 | 25482 | 94991 | 
| 43 | 12 sympy/matrices/expressions/matexpr.py | 632 | 688| 422 | 25904 | 94991 | 
| 44 | 12 sympy/matrices/expressions/dotproduct.py | 46 | 59| 131 | 26035 | 94991 | 
| 45 | 13 sympy/matrices/densearith.py | 149 | 184| 298 | 26333 | 96749 | 
| 46 | 14 sympy/matrices/expressions/funcmatrix.py | 1 | 52| 387 | 26720 | 97136 | 
| 47 | **14 sympy/matrices/common.py** | 43 | 70| 245 | 26965 | 97136 | 
| 48 | 15 sympy/tensor/array/__init__.py | 1 | 209| 2238 | 29203 | 99375 | 
| 49 | 15 sympy/matrices/sparse.py | 370 | 390| 197 | 29400 | 99375 | 
| 50 | **15 sympy/matrices/common.py** | 1526 | 1551| 171 | 29571 | 99375 | 
| 51 | 15 sympy/matrices/expressions/matexpr.py | 579 | 603| 215 | 29786 | 99375 | 
| 52 | 15 sympy/matrices/expressions/matpow.py | 51 | 79| 242 | 30028 | 99375 | 
| 53 | 16 sympy/tensor/array/arrayop.py | 1 | 64| 658 | 30686 | 101998 | 
| 54 | 17 sympy/matrices/__init__.py | 1 | 28| 253 | 30939 | 102251 | 
| 55 | 18 sympy/physics/quantum/tensorproduct.py | 49 | 117| 560 | 31499 | 105515 | 
| 56 | 19 sympy/polys/monomials.py | 101 | 115| 120 | 31619 | 109880 | 
| 57 | 19 sympy/matrices/expressions/matmul.py | 233 | 258| 212 | 31831 | 109880 | 
| 58 | 19 sympy/matrices/expressions/matexpr.py | 276 | 309| 355 | 32186 | 109880 | 
| 59 | 20 sympy/polys/polymatrix.py | 77 | 87| 143 | 32329 | 110823 | 
| 60 | 21 sympy/matrices/expressions/transpose.py | 1 | 71| 448 | 32777 | 111414 | 
| 61 | **21 sympy/matrices/common.py** | 2189 | 2240| 443 | 33220 | 111414 | 
| 62 | 21 sympy/matrices/expressions/matexpr.py | 461 | 576| 1155 | 34375 | 111414 | 
| 63 | **21 sympy/matrices/common.py** | 1824 | 1857| 198 | 34573 | 111414 | 
| 64 | **21 sympy/matrices/common.py** | 1582 | 1606| 221 | 34794 | 111414 | 
| 65 | 22 sympy/combinatorics/permutations.py | 1 | 45| 341 | 35135 | 133932 | 
| 66 | 23 sympy/core/mul.py | 91 | 174| 913 | 36048 | 148221 | 
| 67 | 23 sympy/matrices/dense.py | 395 | 429| 300 | 36348 | 148221 | 
| 68 | **23 sympy/matrices/common.py** | 1789 | 1822| 235 | 36583 | 148221 | 
| 69 | 23 sympy/matrices/matrices.py | 726 | 792| 700 | 37283 | 148221 | 
| 70 | 23 sympy/matrices/sparse.py | 1070 | 1094| 292 | 37575 | 148221 | 
| 71 | 23 sympy/polys/polymatrix.py | 55 | 75| 236 | 37811 | 148221 | 
| 72 | 23 sympy/matrices/densearith.py | 187 | 209| 196 | 38007 | 148221 | 
| 73 | 23 sympy/matrices/expressions/blockmatrix.py | 321 | 342| 217 | 38224 | 148221 | 
| 74 | **23 sympy/matrices/common.py** | 220 | 273| 308 | 38532 | 148221 | 
| 75 | **23 sympy/matrices/common.py** | 363 | 381| 142 | 38674 | 148221 | 
| 76 | 23 sympy/core/mul.py | 235 | 364| 964 | 39638 | 148221 | 
| 77 | 23 sympy/matrices/expressions/matexpr.py | 435 | 459| 237 | 39875 | 148221 | 
| 78 | 24 sympy/assumptions/handlers/matrices.py | 430 | 443| 133 | 40008 | 151483 | 
| 79 | 25 sympy/matrices/expressions/inverse.py | 1 | 64| 411 | 40419 | 152069 | 
| 80 | **25 sympy/matrices/common.py** | 1713 | 1731| 132 | 40551 | 152069 | 
| 81 | 25 sympy/matrices/densearith.py | 229 | 254| 181 | 40732 | 152069 | 
| 82 | **25 sympy/matrices/common.py** | 138 | 180| 341 | 41073 | 152069 | 
| 83 | **25 sympy/matrices/common.py** | 729 | 759| 337 | 41410 | 152069 | 
| 84 | 25 sympy/matrices/sparse.py | 42 | 91| 457 | 41867 | 152069 | 
| 85 | **25 sympy/matrices/common.py** | 112 | 136| 175 | 42042 | 152069 | 
| 86 | 25 sympy/matrices/matrices.py | 2421 | 2497| 670 | 42712 | 152069 | 
| 87 | 26 sympy/physics/secondquant.py | 1592 | 1634| 349 | 43061 | 174628 | 
| 88 | **26 sympy/matrices/common.py** | 457 | 485| 218 | 43279 | 174628 | 
| 89 | **26 sympy/matrices/common.py** | 73 | 110| 308 | 43587 | 174628 | 
| 90 | 26 sympy/matrices/sparse.py | 1 | 17| 117 | 43704 | 174628 | 
| 91 | **26 sympy/matrices/common.py** | 2243 | 2262| 170 | 43874 | 174628 | 
| 92 | 26 sympy/matrices/sparse.py | 254 | 269| 159 | 44033 | 174628 | 
| 93 | 26 sympy/matrices/expressions/blockmatrix.py | 160 | 195| 282 | 44315 | 174628 | 
| 94 | 26 sympy/matrices/dense.py | 350 | 392| 313 | 44628 | 174628 | 
| 95 | 26 sympy/matrices/expressions/blockmatrix.py | 1 | 19| 206 | 44834 | 174628 | 
| 96 | 26 sympy/matrices/sparse.py | 556 | 586| 259 | 45093 | 174628 | 
| 97 | 26 sympy/matrices/dense.py | 507 | 527| 182 | 45275 | 174628 | 
| 98 | 26 sympy/matrices/dense.py | 680 | 704| 215 | 45490 | 174628 | 
| 99 | 26 sympy/matrices/expressions/matmul.py | 200 | 211| 116 | 45606 | 174628 | 
| 100 | 27 sympy/codegen/ast.py | 114 | 137| 309 | 45915 | 182613 | 
| 101 | 27 sympy/matrices/dense.py | 431 | 474| 377 | 46292 | 182613 | 
| 102 | 27 sympy/physics/optics/gaussopt.py | 125 | 137| 134 | 46426 | 182613 | 
| 103 | 28 sympy/matrices/immutable.py | 46 | 62| 185 | 46611 | 183982 | 
| 104 | 28 sympy/matrices/immutable.py | 64 | 93| 267 | 46878 | 183982 | 
| 105 | 28 sympy/matrices/expressions/matadd.py | 16 | 62| 360 | 47238 | 183982 | 
| 106 | 29 sympy/polys/agca/homomorphisms.py | 418 | 508| 846 | 48084 | 189738 | 
| 107 | 29 sympy/matrices/expressions/matadd.py | 1 | 14| 131 | 48215 | 189738 | 
| 108 | 29 sympy/matrices/sparse.py | 271 | 289| 198 | 48413 | 189738 | 
| 109 | 29 sympy/matrices/immutable.py | 95 | 123| 272 | 48685 | 189738 | 
| 110 | 29 sympy/core/mul.py | 176 | 234| 536 | 49221 | 189738 | 
| 111 | 29 sympy/matrices/expressions/matexpr.py | 690 | 738| 261 | 49482 | 189738 | 
| 112 | 29 sympy/matrices/expressions/matexpr.py | 241 | 274| 249 | 49731 | 189738 | 
| 113 | 30 sympy/physics/quantum/matrixutils.py | 1 | 62| 440 | 50171 | 192331 | 
| 114 | **30 sympy/matrices/common.py** | 1553 | 1580| 181 | 50352 | 192331 | 
| 115 | **30 sympy/matrices/common.py** | 1753 | 1771| 168 | 50520 | 192331 | 
| 116 | **30 sympy/matrices/common.py** | 584 | 604| 144 | 50664 | 192331 | 
| 117 | 30 sympy/matrices/matrices.py | 1539 | 1594| 350 | 51014 | 192331 | 
| 118 | 30 sympy/matrices/densearith.py | 1 | 43| 286 | 51300 | 192331 | 
| 119 | 31 sympy/tensor/array/ndim_array.py | 301 | 353| 438 | 51738 | 195629 | 
| 120 | 31 sympy/matrices/immutable.py | 1 | 14| 113 | 51851 | 195629 | 
| 121 | 31 sympy/matrices/sparse.py | 291 | 320| 341 | 52192 | 195629 | 
| 122 | 31 sympy/matrices/sparse.py | 1219 | 1248| 241 | 52433 | 195629 | 
| 123 | 32 sympy/tensor/array/dense_ndim_array.py | 70 | 100| 239 | 52672 | 197197 | 
| 124 | 32 sympy/matrices/dense.py | 28 | 37| 122 | 52794 | 197197 | 
| 125 | 33 sympy/matrices/expressions/adjoint.py | 1 | 65| 415 | 53209 | 197612 | 
| 126 | 33 sympy/matrices/expressions/blockmatrix.py | 413 | 435| 217 | 53426 | 197612 | 
| 127 | 33 sympy/matrices/sparse.py | 1280 | 1304| 201 | 53627 | 197612 | 
| 128 | 33 sympy/matrices/sparse.py | 1011 | 1033| 225 | 53852 | 197612 | 
| 129 | 33 sympy/matrices/expressions/blockmatrix.py | 380 | 389| 127 | 53979 | 197612 | 
| 130 | 33 sympy/physics/optics/gaussopt.py | 51 | 109| 323 | 54302 | 197612 | 
| 131 | 33 sympy/matrices/dense.py | 747 | 777| 243 | 54545 | 197612 | 


### Hint

```
Note to anyone fixing this: `@`/`__matmul__` only works in Python 3.5+. 
I would like to work on this issue.
```

## Patch

```diff
diff --git a/sympy/matrices/common.py b/sympy/matrices/common.py
--- a/sympy/matrices/common.py
+++ b/sympy/matrices/common.py
@@ -1973,6 +1973,10 @@ def __div__(self, other):
 
     @call_highest_priority('__rmatmul__')
     def __matmul__(self, other):
+        other = _matrixify(other)
+        if not getattr(other, 'is_Matrix', False) and not getattr(other, 'is_MatrixLike', False):
+            return NotImplemented
+
         return self.__mul__(other)
 
     @call_highest_priority('__rmul__')
@@ -2066,6 +2070,10 @@ def __radd__(self, other):
 
     @call_highest_priority('__matmul__')
     def __rmatmul__(self, other):
+        other = _matrixify(other)
+        if not getattr(other, 'is_Matrix', False) and not getattr(other, 'is_MatrixLike', False):
+            return NotImplemented
+
         return self.__rmul__(other)
 
     @call_highest_priority('__mul__')

```

## Test Patch

```diff
diff --git a/sympy/matrices/tests/test_commonmatrix.py b/sympy/matrices/tests/test_commonmatrix.py
--- a/sympy/matrices/tests/test_commonmatrix.py
+++ b/sympy/matrices/tests/test_commonmatrix.py
@@ -674,6 +674,30 @@ def test_multiplication():
         assert c[1, 0] == 3*5
         assert c[1, 1] == 0
 
+def test_matmul():
+    a = Matrix([[1, 2], [3, 4]])
+
+    assert a.__matmul__(2) == NotImplemented
+
+    assert a.__rmatmul__(2) == NotImplemented
+
+    #This is done this way because @ is only supported in Python 3.5+
+    #To check 2@a case
+    try:
+        eval('2 @ a')
+    except SyntaxError:
+        pass
+    except TypeError:  #TypeError is raised in case of NotImplemented is returned
+        pass
+
+    #Check a@2 case
+    try:
+        eval('a @ 2')
+    except SyntaxError:
+        pass
+    except TypeError:  #TypeError is raised in case of NotImplemented is returned
+        pass
+
 def test_power():
     raises(NonSquareMatrixError, lambda: Matrix((1, 2))**2)
 

```


## Code snippets

### 1 - sympy/matrices/common.py:

Start line: 2098, End line: 2141

```python
class MatrixArithmetic(MatrixRequired):

    @call_highest_priority('__sub__')
    def __rsub__(self, a):
        return (-self) + a

    @call_highest_priority('__rsub__')
    def __sub__(self, a):
        return self + (-a)

    @call_highest_priority('__rtruediv__')
    def __truediv__(self, other):
        return self.__div__(other)

    def multiply_elementwise(self, other):
        """Return the Hadamard product (elementwise product) of A and B

        Examples
        ========

        >>> from sympy.matrices import Matrix
        >>> A = Matrix([[0, 1, 2], [3, 4, 5]])
        >>> B = Matrix([[1, 10, 100], [100, 10, 1]])
        >>> A.multiply_elementwise(B)
        Matrix([
        [  0, 10, 200],
        [300, 40,   5]])

        See Also
        ========

        cross
        dot
        multiply
        """
        if self.shape != other.shape:
            raise ShapeError("Matrix shapes must agree {} != {}".format(self.shape, other.shape))

        return self._eval_matrix_mul_elementwise(other)


class MatrixCommon(MatrixArithmetic, MatrixOperations, MatrixProperties,
                  MatrixSpecial, MatrixShaping):
    """All common matrix operations including basic arithmetic, shaping,
    and special matrices like `zeros`, and `eye`."""
    _diff_wrt = True
```
### 2 - sympy/matrices/common.py:

Start line: 2065, End line: 2096

```python
class MatrixArithmetic(MatrixRequired):

    @call_highest_priority('__add__')
    def __radd__(self, other):
        return self + other

    @call_highest_priority('__matmul__')
    def __rmatmul__(self, other):
        return self.__rmul__(other)

    @call_highest_priority('__mul__')
    def __rmul__(self, other):
        other = _matrixify(other)
        # matrix-like objects can have shapes.  This is
        # our first sanity check.
        if hasattr(other, 'shape') and len(other.shape) == 2:
            if self.shape[0] != other.shape[1]:
                raise ShapeError("Matrix size mismatch.")

        # honest sympy matrices defer to their class's routine
        if getattr(other, 'is_Matrix', False):
            return other._new(other.as_mutable() * self)
        # Matrix-like objects can be passed to CommonMatrix routines directly.
        if getattr(other, 'is_MatrixLike', False):
            return MatrixArithmetic._eval_matrix_rmul(self, other)

        # if 'other' is not iterable then scalar multiplication.
        if not isinstance(other, collections.Iterable):
            try:
                return self._eval_scalar_rmul(other)
            except TypeError:
                pass

        return NotImplemented
```
### 3 - sympy/matrices/common.py:

Start line: 1978, End line: 2028

```python
class MatrixArithmetic(MatrixRequired):

    @call_highest_priority('__rmul__')
    def __mul__(self, other):
        """Return self*other where other is either a scalar or a matrix
        of compatible dimensions.

        Examples
        ========

        >>> from sympy.matrices import Matrix
        >>> A = Matrix([[1, 2, 3], [4, 5, 6]])
        >>> 2*A == A*2 == Matrix([[2, 4, 6], [8, 10, 12]])
        True
        >>> B = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> A*B
        Matrix([
        [30, 36, 42],
        [66, 81, 96]])
        >>> B*A
        Traceback (most recent call last):
        ...
        ShapeError: Matrices size mismatch.
        >>>

        See Also
        ========

        matrix_multiply_elementwise
        """
        other = _matrixify(other)
        # matrix-like objects can have shapes.  This is
        # our first sanity check.
        if hasattr(other, 'shape') and len(other.shape) == 2:
            if self.shape[1] != other.shape[0]:
                raise ShapeError("Matrix size mismatch: %s * %s." % (
                    self.shape, other.shape))

        # honest sympy matrices defer to their class's routine
        if getattr(other, 'is_Matrix', False):
            return self._eval_matrix_mul(other)
        # Matrix-like objects can be passed to CommonMatrix routines directly.
        if getattr(other, 'is_MatrixLike', False):
            return MatrixArithmetic._eval_matrix_mul(self, other)

        # if 'other' is not iterable then scalar multiplication.
        if not isinstance(other, collections.Iterable):
            try:
                return self._eval_scalar_mul(other)
            except TypeError:
                pass

        return NotImplemented
```
### 4 - sympy/matrices/matrices.py:

Start line: 1596, End line: 2419

```python
class MatrixCalculus(MatrixCommon):

    def jacobian(self, X):
        """Calculates the Jacobian matrix (derivative of a vector-valued function).

        Parameters
        ==========

        self : vector of expressions representing functions f_i(x_1, ..., x_n).
        X : set of x_i's in order, it can be a list or a Matrix

        Both self and X can be a row or a column matrix in any order
        (i.e., jacobian() should always work).

        Examples
        ========

        >>> from sympy import sin, cos, Matrix
        >>> from sympy.abc import rho, phi
        >>> X = Matrix([rho*cos(phi), rho*sin(phi), rho**2])
        >>> Y = Matrix([rho, phi])
        >>> X.jacobian(Y)
        Matrix([
        [cos(phi), -rho*sin(phi)],
        [sin(phi),  rho*cos(phi)],
        [   2*rho,             0]])
        >>> X = Matrix([rho*cos(phi), rho*sin(phi)])
        >>> X.jacobian(Y)
        Matrix([
        [cos(phi), -rho*sin(phi)],
        [sin(phi),  rho*cos(phi)]])

        See Also
        ========

        hessian
        wronskian
        """
        if not isinstance(X, MatrixBase):
            X = self._new(X)
        # Both X and self can be a row or a column matrix, so we need to make
        # sure all valid combinations work, but everything else fails:
        if self.shape[0] == 1:
            m = self.shape[1]
        elif self.shape[1] == 1:
            m = self.shape[0]
        else:
            raise TypeError("self must be a row or a column matrix")
        if X.shape[0] == 1:
            n = X.shape[1]
        elif X.shape[1] == 1:
            n = X.shape[0]
        else:
            raise TypeError("X must be a row or a column matrix")

        # m is the number of functions and n is the number of variables
        # computing the Jacobian is now easy:
        return self._new(m, n, lambda j, i: self[j].diff(X[i]))

    def limit(self, *args):
        """Calculate the limit of each element in the matrix.
        ``args`` will be passed to the ``limit`` function.

        Examples
        ========

        >>> from sympy.matrices import Matrix
        >>> from sympy.abc import x, y
        >>> M = Matrix([[x, y], [1, 0]])
        >>> M.limit(x, 2)
        Matrix([
        [2, y],
        [1, 0]])

        See Also
        ========

        integrate
        diff
        """
        return self.applyfunc(lambda x: x.limit(*args))


# https://github.com/sympy/sympy/pull/12854
class MatrixDeprecated(MatrixCommon):
    """A class to house deprecated matrix methods."""

    def berkowitz_charpoly(self, x=Dummy('lambda'), simplify=_simplify):
        return self.charpoly(x=x)

    def berkowitz_det(self):
        """Computes determinant using Berkowitz method.

        See Also
        ========

        det
        berkowitz
        """
        return self.det(method='berkowitz')

    def berkowitz_eigenvals(self, **flags):
        """Computes eigenvalues of a Matrix using Berkowitz method.

        See Also
        ========

        berkowitz
        """
        return self.eigenvals(**flags)

    def berkowitz_minors(self):
        """Computes principal minors using Berkowitz method.

        See Also
        ========

        berkowitz
        """
        sign, minors = S.One, []

        for poly in self.berkowitz():
            minors.append(sign * poly[-1])
            sign = -sign

        return tuple(minors)

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

            items = [S.One, a] + items

            for i in range(n):
                T[i:, i] = items[:n - i + 1]

            transforms[k - 1] = T

        polys = [self._new([S.One, -A[0, 0]])]

        for i, T in enumerate(transforms):
            polys.append(T * polys[i])

        return berk + tuple(map(tuple, polys))

    def cofactorMatrix(self, method="berkowitz"):
        return self.cofactor_matrix(method=method)

    def det_bareis(self):
        return self.det(method='bareiss')

    def det_bareiss(self):
        """Compute matrix determinant using Bareiss' fraction-free
        algorithm which is an extension of the well known Gaussian
        elimination method. This approach is best suited for dense
        symbolic matrices and will result in a determinant with
        minimal number of fractions. It means that less term
        rewriting is needed on resulting formulae.

        TODO: Implement algorithm for sparse matrices (SFF),
        http://www.eecis.udel.edu/~saunders/papers/sffge/it5.ps.

        See Also
        ========

        det
        berkowitz_det
        """
        return self.det(method='bareiss')

    def det_LU_decomposition(self):
        """Compute matrix determinant using LU decomposition


        Note that this method fails if the LU decomposition itself
        fails. In particular, if the matrix has no inverse this method
        will fail.

        TODO: Implement algorithm for sparse matrices (SFF),
        http://www.eecis.udel.edu/~saunders/papers/sffge/it5.ps.

        See Also
        ========


        det
        det_bareiss
        berkowitz_det
        """
        return self.det(method='lu')

    def jordan_cell(self, eigenval, n):
        return self.jordan_block(size=n, eigenvalue=eigenval)

    def jordan_cells(self, calc_transformation=True):
        P, J = self.jordan_form()
        return P, J.get_diag_blocks()

    def minorEntry(self, i, j, method="berkowitz"):
        return self.minor(i, j, method=method)

    def minorMatrix(self, i, j):
        return self.minor_submatrix(i, j)

    def permuteBkwd(self, perm):
        """Permute the rows of the matrix with the given permutation in reverse."""
        return self.permute_rows(perm, direction='backward')

    def permuteFwd(self, perm):
        """Permute the rows of the matrix with the given permutation."""
        return self.permute_rows(perm, direction='forward')


class MatrixBase(MatrixDeprecated,
                 MatrixCalculus,
                 MatrixEigen,
                 MatrixCommon):
    """Base class for matrix objects."""
    # Added just for numpy compatibility
    __array_priority__ = 11

    is_Matrix = True
    _class_priority = 3
    _sympify = staticmethod(sympify)

    __hash__ = None  # Mutable

    def __array__(self):
        from .dense import matrix2numpy
        return matrix2numpy(self)

    def __getattr__(self, attr):
        if attr in ('diff', 'integrate', 'limit'):
            def doit(*args):
                item_doit = lambda item: getattr(item, attr)(*args)
                return self.applyfunc(item_doit)

            return doit
        else:
            raise AttributeError(
                "%s has no attribute %s." % (self.__class__.__name__, attr))

    def __len__(self):
        """Return the number of elements of self.

        Implemented mainly so bool(Matrix()) == False.
        """
        return self.rows * self.cols

    def __mathml__(self):
        mml = ""
        for i in range(self.rows):
            mml += "<matrixrow>"
            for j in range(self.cols):
                mml += self[i, j].__mathml__()
            mml += "</matrixrow>"
        return "<matrix>" + mml + "</matrix>"

    # needed for python 2 compatibility
    def __ne__(self, other):
        return not self == other

    def _matrix_pow_by_jordan_blocks(self, num):
        from sympy.matrices import diag, MutableMatrix
        from sympy import binomial

        def jordan_cell_power(jc, n):
            N = jc.shape[0]
            l = jc[0, 0]
            if l == 0 and (n < N - 1) != False:
                raise ValueError("Matrix det == 0; not invertible")
            elif l == 0 and N > 1 and n % 1 != 0:
                raise ValueError("Non-integer power cannot be evaluated")
            for i in range(N):
                for j in range(N-i):
                    bn = binomial(n, i)
                    if isinstance(bn, binomial):
                        bn = bn._eval_expand_func()
                    jc[j, i+j] = l**(n-i)*bn

        P, J = self.jordan_form()
        jordan_cells = J.get_diag_blocks()
        # Make sure jordan_cells matrices are mutable:
        jordan_cells = [MutableMatrix(j) for j in jordan_cells]
        for j in jordan_cells:
            jordan_cell_power(j, num)
        return self._new(P*diag(*jordan_cells)*P.inv())

    def __repr__(self):
        return sstr(self)

    def __str__(self):
        if self.rows == 0 or self.cols == 0:
            return 'Matrix(%s, %s, [])' % (self.rows, self.cols)
        return "Matrix(%s)" % str(self.tolist())

    def _diagonalize_clear_subproducts(self):
        del self._is_symbolic
        del self._is_symmetric
        del self._eigenvects

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

    @classmethod
    def _handle_creation_inputs(cls, *args, **kwargs):
        """Return the number of rows, cols and flat matrix elements.

        Examples
        ========

        >>> from sympy import Matrix, I

        Matrix can be constructed as follows:

        * from a nested list of iterables

        >>> Matrix( ((1, 2+I), (3, 4)) )
        Matrix([
        [1, 2 + I],
        [3,     4]])

        * from un-nested iterable (interpreted as a column)

        >>> Matrix( [1, 2] )
        Matrix([
        [1],
        [2]])

        * from un-nested iterable with dimensions

        >>> Matrix(1, 2, [1, 2] )
        Matrix([[1, 2]])

        * from no arguments (a 0 x 0 matrix)

        >>> Matrix()
        Matrix(0, 0, [])

        * from a rule

        >>> Matrix(2, 2, lambda i, j: i/(j + 1) )
        Matrix([
        [0,   0],
        [1, 1/2]])

        """
        from sympy.matrices.sparse import SparseMatrix

        flat_list = None

        if len(args) == 1:
            # Matrix(SparseMatrix(...))
            if isinstance(args[0], SparseMatrix):
                return args[0].rows, args[0].cols, flatten(args[0].tolist())

            # Matrix(Matrix(...))
            elif isinstance(args[0], MatrixBase):
                return args[0].rows, args[0].cols, args[0]._mat

            # Matrix(MatrixSymbol('X', 2, 2))
            elif isinstance(args[0], Basic) and args[0].is_Matrix:
                return args[0].rows, args[0].cols, args[0].as_explicit()._mat

            # Matrix(numpy.ones((2, 2)))
            elif hasattr(args[0], "__array__"):
                # NumPy array or matrix or some other object that implements
                # __array__. So let's first use this method to get a
                # numpy.array() and then make a python list out of it.
                arr = args[0].__array__()
                if len(arr.shape) == 2:
                    rows, cols = arr.shape[0], arr.shape[1]
                    flat_list = [cls._sympify(i) for i in arr.ravel()]
                    return rows, cols, flat_list
                elif len(arr.shape) == 1:
                    rows, cols = arr.shape[0], 1
                    flat_list = [S.Zero] * rows
                    for i in range(len(arr)):
                        flat_list[i] = cls._sympify(arr[i])
                    return rows, cols, flat_list
                else:
                    raise NotImplementedError(
                        "SymPy supports just 1D and 2D matrices")

            # Matrix([1, 2, 3]) or Matrix([[1, 2], [3, 4]])
            elif is_sequence(args[0]) \
                    and not isinstance(args[0], DeferredVector):
                in_mat = []
                ncol = set()
                for row in args[0]:
                    if isinstance(row, MatrixBase):
                        in_mat.extend(row.tolist())
                        if row.cols or row.rows:  # only pay attention if it's not 0x0
                            ncol.add(row.cols)
                    else:
                        in_mat.append(row)
                        try:
                            ncol.add(len(row))
                        except TypeError:
                            ncol.add(1)
                if len(ncol) > 1:
                    raise ValueError("Got rows of variable lengths: %s" %
                                     sorted(list(ncol)))
                cols = ncol.pop() if ncol else 0
                rows = len(in_mat) if cols else 0
                if rows:
                    if not is_sequence(in_mat[0]):
                        cols = 1
                        flat_list = [cls._sympify(i) for i in in_mat]
                        return rows, cols, flat_list
                flat_list = []
                for j in range(rows):
                    for i in range(cols):
                        flat_list.append(cls._sympify(in_mat[j][i]))

        elif len(args) == 3:
            rows = as_int(args[0])
            cols = as_int(args[1])

            if rows < 0 or cols < 0:
                raise ValueError("Cannot create a {} x {} matrix. "
                                 "Both dimensions must be positive".format(rows, cols))

            # Matrix(2, 2, lambda i, j: i+j)
            if len(args) == 3 and isinstance(args[2], collections.Callable):
                op = args[2]
                flat_list = []
                for i in range(rows):
                    flat_list.extend(
                        [cls._sympify(op(cls._sympify(i), cls._sympify(j)))
                         for j in range(cols)])

            # Matrix(2, 2, [1, 2, 3, 4])
            elif len(args) == 3 and is_sequence(args[2]):
                flat_list = args[2]
                if len(flat_list) != rows * cols:
                    raise ValueError(
                        'List length should be equal to rows*columns')
                flat_list = [cls._sympify(i) for i in flat_list]


        # Matrix()
        elif len(args) == 0:
            # Empty Matrix
            rows = cols = 0
            flat_list = []

        if flat_list is None:
            raise TypeError("Data type not understood")

        return rows, cols, flat_list

    def _setitem(self, key, value):
        """Helper to set value at location given by key.

        Examples
        ========

        >>> from sympy import Matrix, I, zeros, ones
        >>> m = Matrix(((1, 2+I), (3, 4)))
        >>> m
        Matrix([
        [1, 2 + I],
        [3,     4]])
        >>> m[1, 0] = 9
        >>> m
        Matrix([
        [1, 2 + I],
        [9,     4]])
        >>> m[1, 0] = [[0, 1]]

        To replace row r you assign to position r*m where m
        is the number of columns:

        >>> M = zeros(4)
        >>> m = M.cols
        >>> M[3*m] = ones(1, m)*2; M
        Matrix([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [2, 2, 2, 2]])

        And to replace column c you can assign to position c:

        >>> M[2] = ones(m, 1)*4; M
        Matrix([
        [0, 0, 4, 0],
        [0, 0, 4, 0],
        [0, 0, 4, 0],
        [2, 2, 4, 2]])
        """
        from .dense import Matrix

        is_slice = isinstance(key, slice)
        i, j = key = self.key2ij(key)
        is_mat = isinstance(value, MatrixBase)
        if type(i) is slice or type(j) is slice:
            if is_mat:
                self.copyin_matrix(key, value)
                return
            if not isinstance(value, Expr) and is_sequence(value):
                self.copyin_list(key, value)
                return
            raise ValueError('unexpected value: %s' % value)
        else:
            if (not is_mat and
                    not isinstance(value, Basic) and is_sequence(value)):
                value = Matrix(value)
                is_mat = True
            if is_mat:
                if is_slice:
                    key = (slice(*divmod(i, self.cols)),
                           slice(*divmod(j, self.cols)))
                else:
                    key = (slice(i, i + value.rows),
                           slice(j, j + value.cols))
                self.copyin_matrix(key, value)
            else:
                return i, j, self._sympify(value)
            return

    def add(self, b):
        """Return self + b """
        return self + b

    def cholesky_solve(self, rhs):
        """Solves Ax = B using Cholesky decomposition,
        for a general square non-singular matrix.
        For a non-square matrix with rows > cols,
        the least squares solution is returned.

        See Also
        ========

        lower_triangular_solve
        upper_triangular_solve
        gauss_jordan_solve
        diagonal_solve
        LDLsolve
        LUsolve
        QRsolve
        pinv_solve
        """
        if self.is_symmetric():
            L = self._cholesky()
        elif self.rows >= self.cols:
            L = (self.T * self)._cholesky()
            rhs = self.T * rhs
        else:
            raise NotImplementedError('Under-determined System. '
                                      'Try M.gauss_jordan_solve(rhs)')
        Y = L._lower_triangular_solve(rhs)
        return (L.T)._upper_triangular_solve(Y)

    def cholesky(self):
        """Returns the Cholesky decomposition L of a matrix A
        such that L * L.T = A

        A must be a square, symmetric, positive-definite
        and non-singular matrix.

        Examples
        ========

        >>> from sympy.matrices import Matrix
        >>> A = Matrix(((25, 15, -5), (15, 18, 0), (-5, 0, 11)))
        >>> A.cholesky()
        Matrix([
        [ 5, 0, 0],
        [ 3, 3, 0],
        [-1, 1, 3]])
        >>> A.cholesky() * A.cholesky().T
        Matrix([
        [25, 15, -5],
        [15, 18,  0],
        [-5,  0, 11]])

        See Also
        ========

        LDLdecomposition
        LUdecomposition
        QRdecomposition
        """

        if not self.is_square:
            raise NonSquareMatrixError("Matrix must be square.")
        if not self.is_symmetric():
            raise ValueError("Matrix must be symmetric.")
        return self._cholesky()

    def condition_number(self):
        """Returns the condition number of a matrix.

        This is the maximum singular value divided by the minimum singular value

        Examples
        ========

        >>> from sympy import Matrix, S
        >>> A = Matrix([[1, 0, 0], [0, 10, 0], [0, 0, S.One/10]])
        >>> A.condition_number()
        100

        See Also
        ========

        singular_values
        """
        if not self:
            return S.Zero
        singularvalues = self.singular_values()
        return Max(*singularvalues) / Min(*singularvalues)

    def copy(self):
        """
        Returns the copy of a matrix.

        Examples
        ========

        >>> from sympy import Matrix
        >>> A = Matrix(2, 2, [1, 2, 3, 4])
        >>> A.copy()
        Matrix([
        [1, 2],
        [3, 4]])

        """
        return self._new(self.rows, self.cols, self._mat)

    def cross(self, b):
        r"""
        Return the cross product of ``self`` and ``b`` relaxing the condition
        of compatible dimensions: if each has 3 elements, a matrix of the
        same type and shape as ``self`` will be returned. If ``b`` has the same
        shape as ``self`` then common identities for the cross product (like
        `a \times b = - b \times a`) will hold.

        Parameters
        ==========
            b : 3x1 or 1x3 Matrix

        See Also
        ========

        dot
        multiply
        multiply_elementwise
        """
        if not is_sequence(b):
            raise TypeError(
                "`b` must be an ordered iterable or Matrix, not %s." %
                type(b))
        if not (self.rows * self.cols == b.rows * b.cols == 3):
            raise ShapeError("Dimensions incorrect for cross product: %s x %s" %
                             ((self.rows, self.cols), (b.rows, b.cols)))
        else:
            return self._new(self.rows, self.cols, (
                (self[1] * b[2] - self[2] * b[1]),
                (self[2] * b[0] - self[0] * b[2]),
                (self[0] * b[1] - self[1] * b[0])))

    @property
    def D(self):
        """Return Dirac conjugate (if self.rows == 4).

        Examples
        ========

        >>> from sympy import Matrix, I, eye
        >>> m = Matrix((0, 1 + I, 2, 3))
        >>> m.D
        Matrix([[0, 1 - I, -2, -3]])
        >>> m = (eye(4) + I*eye(4))
        >>> m[0, 3] = 2
        >>> m.D
        Matrix([
        [1 - I,     0,      0,      0],
        [    0, 1 - I,      0,      0],
        [    0,     0, -1 + I,      0],
        [    2,     0,      0, -1 + I]])

        If the matrix does not have 4 rows an AttributeError will be raised
        because this property is only defined for matrices with 4 rows.

        >>> Matrix(eye(2)).D
        Traceback (most recent call last):
        ...
        AttributeError: Matrix has no attribute D.

        See Also
        ========

        conjugate: By-element conjugation
        H: Hermite conjugation
        """
        from sympy.physics.matrices import mgamma
        if self.rows != 4:
            # In Python 3.2, properties can only return an AttributeError
            # so we can't raise a ShapeError -- see commit which added the
            # first line of this inline comment. Also, there is no need
            # for a message since MatrixBase will raise the AttributeError
            raise AttributeError
        return self.H * mgamma(0)

    def diagonal_solve(self, rhs):
        """Solves Ax = B efficiently, where A is a diagonal Matrix,
        with non-zero diagonal entries.

        Examples
        ========

        >>> from sympy.matrices import Matrix, eye
        >>> A = eye(2)*2
        >>> B = Matrix([[1, 2], [3, 4]])
        >>> A.diagonal_solve(B) == B/2
        True

        See Also
        ========

        lower_triangular_solve
        upper_triangular_solve
        gauss_jordan_solve
        cholesky_solve
        LDLsolve
        LUsolve
        QRsolve
        pinv_solve
        """
        if not self.is_diagonal:
            raise TypeError("Matrix should be diagonal")
        if rhs.rows != self.rows:
            raise TypeError("Size mis-match")
        return self._diagonal_solve(rhs)

    def dot(self, b):
        """Return the dot product of Matrix self and b relaxing the condition
        of compatible dimensions: if either the number of rows or columns are
        the same as the length of b then the dot product is returned. If self
        is a row or column vector, a scalar is returned. Otherwise, a list
        of results is returned (and in that case the number of columns in self
        must match the length of b).

        Examples
        ========

        >>> from sympy import Matrix
        >>> M = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> v = [1, 1, 1]
        >>> M.row(0).dot(v)
        6
        >>> M.col(0).dot(v)
        12
        >>> M.dot(v)
        [6, 15, 24]

        See Also
        ========

        cross
        multiply
        multiply_elementwise
        """
        from .dense import Matrix

        if not isinstance(b, MatrixBase):
            if is_sequence(b):
                if len(b) != self.cols and len(b) != self.rows:
                    raise ShapeError(
                        "Dimensions incorrect for dot product: %s, %s" % (
                            self.shape, len(b)))
                return self.dot(Matrix(b))
            else:
                raise TypeError(
                    "`b` must be an ordered iterable or Matrix, not %s." %
                    type(b))

        mat = self
        if mat.cols == b.rows:
            if b.cols != 1:
                mat = mat.T
                b = b.T
            prod = flatten((mat * b).tolist())
            if len(prod) == 1:
                return prod[0]
            return prod
        if mat.cols == b.cols:
            return mat.dot(b.T)
        elif mat.rows == b.rows:
            return mat.T.dot(b)
        else:
            raise ShapeError("Dimensions incorrect for dot product: %s, %s" % (
                self.shape, b.shape))
```
### 5 - sympy/matrices/common.py:

Start line: 2030, End line: 2063

```python
class MatrixArithmetic(MatrixRequired):

    def __neg__(self):
        return self._eval_scalar_mul(-1)

    @call_highest_priority('__rpow__')
    def __pow__(self, num):
        if not self.rows == self.cols:
            raise NonSquareMatrixError()
        try:
            a = self
            num = sympify(num)
            if num.is_Number and num % 1 == 0:
                if a.rows == 1:
                    return a._new([[a[0]**num]])
                if num == 0:
                    return self._new(self.rows, self.cols, lambda i, j: int(i == j))
                if num < 0:
                    num = -num
                    a = a.inv()
                # When certain conditions are met,
                # Jordan block algorithm is faster than
                # computation by recursion.
                elif a.rows == 2 and num > 100000:
                    try:
                        return a._matrix_pow_by_jordan_blocks(num)
                    except (AttributeError, MatrixError):
                        pass
                return a._eval_pow_by_recursion(num)
            elif isinstance(num, (Expr, float)):
                return a._matrix_pow_by_jordan_blocks(num)
            else:
                raise TypeError(
                    "Only SymPy expressions or integers are supported as exponent for matrices")
        except AttributeError:
            raise TypeError("Don't know how to raise {} to {}".format(self.__class__, num))
```
### 6 - sympy/matrices/common.py:

Start line: 1889, End line: 1917

```python
class MatrixArithmetic(MatrixRequired):
    """Provides basic matrix arithmetic operations.
    Should not be instantiated directly."""

    _op_priority = 10.01

    def _eval_Abs(self):
        return self._new(self.rows, self.cols, lambda i, j: Abs(self[i, j]))

    def _eval_add(self, other):
        return self._new(self.rows, self.cols,
                         lambda i, j: self[i, j] + other[i, j])

    def _eval_matrix_mul(self, other):
        def entry(i, j):
            try:
                return sum(self[i,k]*other[k,j] for k in range(self.cols))
            except TypeError:
                # Block matrices don't work with `sum` or `Add` (ISSUE #11599)
                # They don't work with `sum` because `sum` tries to add `0`
                # initially, and for a matrix, that is a mix of a scalar and
                # a matrix, which raises a TypeError. Fall back to a
                # block-matrix-safe way to multiply if the `sum` fails.
                ret = self[i, 0]*other[0, j]
                for k in range(1, self.cols):
                    ret += self[i, k]*other[k, j]
                return ret

        return self._new(self.rows, other.cols, entry)
```
### 7 - sympy/matrices/expressions/matmul.py:

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
### 8 - sympy/matrices/dense.py:

Start line: 1303, End line: 1343

```python
def matrix_multiply_elementwise(A, B):
    """Return the Hadamard product (elementwise product) of A and B

    >>> from sympy.matrices import matrix_multiply_elementwise
    >>> from sympy.matrices import Matrix
    >>> A = Matrix([[0, 1, 2], [3, 4, 5]])
    >>> B = Matrix([[1, 10, 100], [100, 10, 1]])
    >>> matrix_multiply_elementwise(A, B)
    Matrix([
    [  0, 10, 200],
    [300, 40,   5]])

    See Also
    ========

    __mul__
    """
    if A.shape != B.shape:
        raise ShapeError()
    shape = A.shape
    return classof(A, B)._new(shape[0], shape[1],
                              lambda i, j: A[i, j]*B[i, j])


def ones(*args, **kwargs):
    """Returns a matrix of ones with ``rows`` rows and ``cols`` columns;
    if ``cols`` is omitted a square matrix will be returned.

    See Also
    ========

    zeros
    eye
    diag
    """

    if 'c' in kwargs:
        kwargs['cols'] = kwargs.pop('c')
    from .dense import Matrix

    return Matrix.ones(*args, **kwargs)
```
### 9 - sympy/matrices/sparse.py:

Start line: 949, End line: 1009

```python
class MutableSparseMatrix(SparseMatrix, MatrixBase):

    def col_join(self, other):
        """Returns B augmented beneath A (row-wise joining)::

            [A]
            [B]

        Examples
        ========

        >>> from sympy import SparseMatrix, Matrix, ones
        >>> A = SparseMatrix(ones(3))
        >>> A
        Matrix([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]])
        >>> B = SparseMatrix.eye(3)
        >>> B
        Matrix([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]])
        >>> C = A.col_join(B); C
        Matrix([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]])
        >>> C == A.col_join(Matrix(B))
        True

        Joining along columns is the same as appending rows at the end
        of the matrix:

        >>> C == A.row_insert(A.rows, Matrix(B))
        True
        """
        # A null matrix can always be stacked (see  #10770)
        if self.rows == 0 and self.cols != other.cols:
            return self._new(0, other.cols, []).col_join(other)

        A, B = self, other
        if not A.cols == B.cols:
            raise ShapeError()
        A = A.copy()
        if not isinstance(B, SparseMatrix):
            k = 0
            b = B._mat
            for i in range(B.rows):
                for j in range(B.cols):
                    v = b[k]
                    if v:
                        A._smat[(i + A.rows, j)] = v
                    k += 1
        else:
            for (i, j), v in B._smat.items():
                A._smat[i + A.rows, j] = v
        A.rows += B.rows
        return A
```
### 10 - sympy/matrices/expressions/matmul.py:

Start line: 15, End line: 46

```python
class MatMul(MatrixExpr):
    """
    A product of matrix expressions

    Examples
    ========

    >>> from sympy import MatMul, MatrixSymbol
    >>> A = MatrixSymbol('A', 5, 4)
    >>> B = MatrixSymbol('B', 4, 3)
    >>> C = MatrixSymbol('C', 3, 6)
    >>> MatMul(A, B, C)
    A*B*C
    """
    is_MatMul = True

    def __new__(cls, *args, **kwargs):
        check = kwargs.get('check', True)

        args = list(map(sympify, args))
        obj = Basic.__new__(cls, *args)
        factor, matrices = obj.as_coeff_matrices()
        if check:
            validate(*matrices)
        if not matrices:
            return factor
        return obj

    @property
    def shape(self):
        matrices = [arg for arg in self.args if arg.is_Matrix]
        return (matrices[0].rows, matrices[-1].cols)
```
### 11 - sympy/matrices/common.py:

Start line: 1919, End line: 1944

```python
class MatrixArithmetic(MatrixRequired):

    def _eval_matrix_mul_elementwise(self, other):
        return self._new(self.rows, self.cols, lambda i, j: self[i,j]*other[i,j])

    def _eval_matrix_rmul(self, other):
        def entry(i, j):
            return sum(other[i,k]*self[k,j] for k in range(other.cols))
        return self._new(other.rows, self.cols, entry)

    def _eval_pow_by_recursion(self, num):
        if num == 1:
            return self
        if num % 2 == 1:
            return self * self._eval_pow_by_recursion(num - 1)
        ret = self._eval_pow_by_recursion(num // 2)
        return ret * ret

    def _eval_scalar_mul(self, other):
        return self._new(self.rows, self.cols, lambda i, j: self[i,j]*other)

    def _eval_scalar_rmul(self, other):
        return self._new(self.rows, self.cols, lambda i, j: other*self[i,j])

    # python arithmetic functions
    def __abs__(self):
        """Returns a new matrix with entry-wise absolute values."""
        return self._eval_Abs()
```
### 22 - sympy/matrices/common.py:

Start line: 1946, End line: 1976

```python
class MatrixArithmetic(MatrixRequired):

    @call_highest_priority('__radd__')
    def __add__(self, other):
        """Return self + other, raising ShapeError if shapes don't match."""
        other = _matrixify(other)
        # matrix-like objects can have shapes.  This is
        # our first sanity check.
        if hasattr(other, 'shape'):
            if self.shape != other.shape:
                raise ShapeError("Matrix size mismatch: %s + %s" % (
                    self.shape, other.shape))

        # honest sympy matrices defer to their class's routine
        if getattr(other, 'is_Matrix', False):
            # call the highest-priority class's _eval_add
            a, b = self, other
            if a.__class__ != classof(a, b):
                b, a = a, b
            return a._eval_add(b)
        # Matrix-like objects can be passed to CommonMatrix routines directly.
        if getattr(other, 'is_MatrixLike', False):
            return MatrixArithmetic._eval_add(self, other)

        raise TypeError('cannot add %s and %s' % (type(self), type(other)))

    @call_highest_priority('__rdiv__')
    def __div__(self, other):
        return self * (S.One / other)

    @call_highest_priority('__rmatmul__')
    def __matmul__(self, other):
        return self.__mul__(other)
```
### 23 - sympy/matrices/common.py:

Start line: 1479, End line: 1524

```python
class MatrixOperations(MatrixRequired):
    """Provides basic matrix shape and elementwise
    operations.  Should not be instantiated directly."""

    def _eval_adjoint(self):
        return self.transpose().conjugate()

    def _eval_applyfunc(self, f):
        out = self._new(self.rows, self.cols, [f(x) for x in self])
        return out

    def _eval_as_real_imag(self):
        from sympy.functions.elementary.complexes import re, im

        return (self.applyfunc(re), self.applyfunc(im))

    def _eval_conjugate(self):
        return self.applyfunc(lambda x: x.conjugate())

    def _eval_permute_cols(self, perm):
        # apply the permutation to a list
        mapping = list(perm)

        def entry(i, j):
            return self[i, mapping[j]]

        return self._new(self.rows, self.cols, entry)

    def _eval_permute_rows(self, perm):
        # apply the permutation to a list
        mapping = list(perm)

        def entry(i, j):
            return self[mapping[i], j]

        return self._new(self.rows, self.cols, entry)

    def _eval_trace(self):
        return sum(self[i, i] for i in range(self.rows))

    def _eval_transpose(self):
        return self._new(self.cols, self.rows, lambda i, j: self[j, i])

    def adjoint(self):
        """Conjugate transpose or Hermitian conjugation."""
        return self._eval_adjoint()
```
### 27 - sympy/matrices/common.py:

Start line: 1, End line: 40

```python
"""
Basic methods common to all matrices to be used
when creating more advanced matrices (e.g., matrices over rings,
etc.).
"""

from __future__ import print_function, division

import collections
from sympy.core.add import Add
from sympy.core.basic import Basic, Atom
from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.core.function import count_ops
from sympy.core.singleton import S
from sympy.core.sympify import sympify
from sympy.core.compatibility import is_sequence, default_sort_key, range, \
    NotIterable

from sympy.simplify import simplify as _simplify, signsimp, nsimplify
from sympy.utilities.iterables import flatten
from sympy.functions import Abs
from sympy.core.compatibility import reduce, as_int, string_types
from sympy.assumptions.refine import refine
from sympy.core.decorators import call_highest_priority

from types import FunctionType


class MatrixError(Exception):
    pass


class ShapeError(ValueError, MatrixError):
    """Wrong matrix shape"""
    pass


class NonSquareMatrixError(ShapeError):
    pass
```
### 30 - sympy/matrices/common.py:

Start line: 2265, End line: 2289

```python
def _matrixify(mat):
    """If `mat` is a Matrix or is matrix-like,
    return a Matrix or MatrixWrapper object.  Otherwise
    `mat` is passed through without modification."""
    if getattr(mat, 'is_Matrix', False):
        return mat
    if hasattr(mat, 'shape'):
        if len(mat.shape) == 2:
            return _MatrixWrapper(mat)
    return mat


def a2idx(j, n=None):
    """Return integer after making positive and validating against n."""
    if type(j) is not int:
        try:
            j = j.__index__()
        except AttributeError:
            raise IndexError("Invalid index a[%r]" % (j,))
    if n is not None:
        if j < 0:
            j += n
        if not (j >= 0 and j < n):
            raise IndexError("Index out of range: a[%s]" % (j,))
    return int(j)
```
### 31 - sympy/matrices/common.py:

Start line: 2292, End line: 2324

```python
def classof(A, B):
    """
    Get the type of the result when combining matrices of different types.

    Currently the strategy is that immutability is contagious.

    Examples
    ========

    >>> from sympy import Matrix, ImmutableMatrix
    >>> from sympy.matrices.matrices import classof
    >>> M = Matrix([[1, 2], [3, 4]]) # a Mutable Matrix
    >>> IM = ImmutableMatrix([[1, 2], [3, 4]])
    >>> classof(M, IM)
    <class 'sympy.matrices.immutable.ImmutableDenseMatrix'>
    """
    try:
        if A._class_priority > B._class_priority:
            return A.__class__
        else:
            return B.__class__
    except Exception:
        pass
    try:
        import numpy
        if isinstance(A, numpy.ndarray):
            return B.__class__
        if isinstance(B, numpy.ndarray):
            return A.__class__
    except Exception:
        pass
    raise TypeError("Incompatible classes %s, %s" % (A.__class__, B.__class__))
```
### 41 - sympy/matrices/common.py:

Start line: 1859, End line: 1886

```python
class MatrixOperations(MatrixRequired):

    T = property(transpose, None, None, "Matrix transposition.")

    C = property(conjugate, None, None, "By-element conjugation.")

    n = evalf

    def xreplace(self, rule):  # should mirror core.basic.xreplace
        """Return a new matrix with xreplace applied to each entry.

        Examples
        ========

        >>> from sympy.abc import x, y
        >>> from sympy.matrices import SparseMatrix, Matrix
        >>> SparseMatrix(1, 1, [x])
        Matrix([[x]])
        >>> _.xreplace({x: y})
        Matrix([[y]])
        >>> Matrix(_).xreplace({y: x})
        Matrix([[x]])
        """
        return self.applyfunc(lambda x: x.xreplace(rule))

    _eval_simplify = simplify

    def _eval_trigsimp(self, **opts):
        from sympy.simplify import trigsimp
        return self.applyfunc(lambda x: trigsimp(x, **opts))
```
### 47 - sympy/matrices/common.py:

Start line: 43, End line: 70

```python
class MatrixRequired(object):
    """All subclasses of matrix objects must implement the
    required matrix properties listed here."""
    rows = None
    cols = None
    shape = None
    _simplify = None

    @classmethod
    def _new(cls, *args, **kwargs):
        """`_new` must, at minimum, be callable as
        `_new(rows, cols, mat) where mat is a flat list of the
        elements of the matrix."""
        raise NotImplementedError("Subclasses must implement this.")

    def __eq__(self, other):
        raise NotImplementedError("Subclasses must implement this.")

    def __getitem__(self, key):
        """Implementations of __getitem__ should accept ints, in which
        case the matrix is indexed as a flat list, tuples (i,j) in which
        case the (i,j) entry is returned, slices, or mixed tuples (a,b)
        where a and b are any combintion of slices and integers."""
        raise NotImplementedError("Subclasses must implement this.")

    def __len__(self):
        """The total number of entries in the matrix."""
        raise NotImplementedError("Subclasses must implement this.")
```
### 50 - sympy/matrices/common.py:

Start line: 1526, End line: 1551

```python
class MatrixOperations(MatrixRequired):

    def applyfunc(self, f):
        """Apply a function to each element of the matrix.

        Examples
        ========

        >>> from sympy import Matrix
        >>> m = Matrix(2, 2, lambda i, j: i*2+j)
        >>> m
        Matrix([
        [0, 1],
        [2, 3]])
        >>> m.applyfunc(lambda i: 2*i)
        Matrix([
        [0, 2],
        [4, 6]])

        """
        if not callable(f):
            raise TypeError("`f` must be callable.")

        return self._eval_applyfunc(f)

    def as_real_imag(self):
        """Returns a tuple containing the (real, imaginary) part of matrix."""
        return self._eval_as_real_imag()
```
### 61 - sympy/matrices/common.py:

Start line: 2189, End line: 2240

```python
class _MinimalMatrix(object):

    def __getitem__(self, key):
        def _normalize_slices(row_slice, col_slice):
            """Ensure that row_slice and col_slice don't have
            `None` in their arguments.  Any integers are converted
            to slices of length 1"""
            if not isinstance(row_slice, slice):
                row_slice = slice(row_slice, row_slice + 1, None)
            row_slice = slice(*row_slice.indices(self.rows))

            if not isinstance(col_slice, slice):
                col_slice = slice(col_slice, col_slice + 1, None)
            col_slice = slice(*col_slice.indices(self.cols))

            return (row_slice, col_slice)

        def _coord_to_index(i, j):
            """Return the index in _mat corresponding
            to the (i,j) position in the matrix. """
            return i * self.cols + j

        if isinstance(key, tuple):
            i, j = key
            if isinstance(i, slice) or isinstance(j, slice):
                # if the coordinates are not slices, make them so
                # and expand the slices so they don't contain `None`
                i, j = _normalize_slices(i, j)

                rowsList, colsList = list(range(self.rows))[i], \
                                     list(range(self.cols))[j]
                indices = (i * self.cols + j for i in rowsList for j in
                           colsList)
                return self._new(len(rowsList), len(colsList),
                                 list(self.mat[i] for i in indices))

            # if the key is a tuple of ints, change
            # it to an array index
            key = _coord_to_index(i, j)
        return self.mat[key]

    def __eq__(self, other):
        return self.shape == other.shape and list(self) == list(other)

    def __len__(self):
        return self.rows*self.cols

    def __repr__(self):
        return "_MinimalMatrix({}, {}, {})".format(self.rows, self.cols,
                                                   self.mat)

    @property
    def shape(self):
        return (self.rows, self.cols)
```
### 63 - sympy/matrices/common.py:

Start line: 1824, End line: 1857

```python
class MatrixOperations(MatrixRequired):

    def transpose(self):
        """
        Returns the transpose of the matrix.

        Examples
        ========

        >>> from sympy import Matrix
        >>> A = Matrix(2, 2, [1, 2, 3, 4])
        >>> A.transpose()
        Matrix([
        [1, 3],
        [2, 4]])

        >>> from sympy import Matrix, I
        >>> m=Matrix(((1, 2+I), (3, 4)))
        >>> m
        Matrix([
        [1, 2 + I],
        [3,     4]])
        >>> m.transpose()
        Matrix([
        [    1, 3],
        [2 + I, 4]])
        >>> m.T == m.transpose()
        True

        See Also
        ========

        conjugate: By-element conjugation

        """
        return self._eval_transpose()
```
### 64 - sympy/matrices/common.py:

Start line: 1582, End line: 1606

```python
class MatrixOperations(MatrixRequired):

    def doit(self, **kwargs):
        return self.applyfunc(lambda x: x.doit())

    def evalf(self, prec=None, **options):
        """Apply evalf() to each element of self."""
        return self.applyfunc(lambda i: i.evalf(prec, **options))

    def expand(self, deep=True, modulus=None, power_base=True, power_exp=True,
               mul=True, log=True, multinomial=True, basic=True, **hints):
        """Apply core.function.expand to each entry of the matrix.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy.matrices import Matrix
        >>> Matrix(1, 1, [x*(x+1)])
        Matrix([[x*(x + 1)]])
        >>> _.expand()
        Matrix([[x**2 + x]])

        """
        return self.applyfunc(lambda x: x.expand(
            deep, modulus, power_base, power_exp, mul, log, multinomial, basic,
            **hints))
```
### 68 - sympy/matrices/common.py:

Start line: 1789, End line: 1822

```python
class MatrixOperations(MatrixRequired):

    def subs(self, *args, **kwargs):  # should mirror core.basic.subs
        """Return a new matrix with subs applied to each entry.

        Examples
        ========

        >>> from sympy.abc import x, y
        >>> from sympy.matrices import SparseMatrix, Matrix
        >>> SparseMatrix(1, 1, [x])
        Matrix([[x]])
        >>> _.subs(x, y)
        Matrix([[y]])
        >>> Matrix(_).subs(y, x)
        Matrix([[x]])
        """
        return self.applyfunc(lambda x: x.subs(*args, **kwargs))

    def trace(self):
        """
        Returns the trace of a square matrix i.e. the sum of the
        diagonal elements.

        Examples
        ========

        >>> from sympy import Matrix
        >>> A = Matrix(2, 2, [1, 2, 3, 4])
        >>> A.trace()
        5

        """
        if not self.rows == self.cols:
            raise NonSquareMatrixError()
        return self._eval_trace()
```
### 74 - sympy/matrices/common.py:

Start line: 220, End line: 273

```python
class MatrixShaping(MatrixRequired):

    def col_join(self, other):
        """Concatenates two matrices along self's last and other's first row.

        Examples
        ========

        >>> from sympy import zeros, ones
        >>> M = zeros(3)
        >>> V = ones(1, 3)
        >>> M.col_join(V)
        Matrix([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [1, 1, 1]])

        See Also
        ========

        col
        row_join
        """
        # A null matrix can always be stacked (see  #10770)
        if self.rows == 0 and self.cols != other.cols:
            return self._new(0, other.cols, []).col_join(other)

        if self.cols != other.cols:
            raise ShapeError(
                "`self` and `other` must have the same number of columns.")
        return self._eval_col_join(other)

    def col(self, j):
        """Elementary column selector.

        Examples
        ========

        >>> from sympy import eye
        >>> eye(2).col(0)
        Matrix([
        [1],
        [0]])

        See Also
        ========

        row
        col_op
        col_swap
        col_del
        col_join
        col_insert
        """
        return self[:, j]
```
### 75 - sympy/matrices/common.py:

Start line: 363, End line: 381

```python
class MatrixShaping(MatrixRequired):

    @classmethod
    def hstack(cls, *args):
        """Return a matrix formed by joining args horizontally (i.e.
        by repeated application of row_join).

        Examples
        ========

        >>> from sympy.matrices import Matrix, eye
        >>> Matrix.hstack(eye(2), 2*eye(2))
        Matrix([
        [1, 0, 2, 0],
        [0, 1, 0, 2]])
        """
        if len(args) == 0:
            return cls._new()

        kls = type(args[0])
        return reduce(kls.row_join, args)
```
### 80 - sympy/matrices/common.py:

Start line: 1713, End line: 1731

```python
class MatrixOperations(MatrixRequired):

    def permute_cols(self, swaps, direction='forward'):
        """Alias for `self.permute(swaps, orientation='cols', direction=direction)`

        See Also
        ========

        permute
        """
        return self.permute(swaps, orientation='cols', direction=direction)

    def permute_rows(self, swaps, direction='forward'):
        """Alias for `self.permute(swaps, orientation='rows', direction=direction)`

        See Also
        ========

        permute
        """
        return self.permute(swaps, orientation='rows', direction=direction)
```
### 82 - sympy/matrices/common.py:

Start line: 138, End line: 180

```python
class MatrixShaping(MatrixRequired):

    def _eval_row_del(self, row):
        def entry(i, j):
            return self[i, j] if i < row else self[i + 1, j]
        return self._new(self.rows - 1, self.cols, entry)

    def _eval_row_insert(self, pos, other):
        entries = list(self)
        insert_pos = pos * self.cols
        entries[insert_pos:insert_pos] = list(other)
        return self._new(self.rows + other.rows, self.cols, entries)

    def _eval_row_join(self, other):
        cols = self.cols

        def entry(i, j):
            if j < cols:
                return self[i, j]
            return other[i, j - cols]

        return classof(self, other)._new(self.rows, self.cols + other.cols,
                                         lambda i, j: entry(i, j))

    def _eval_tolist(self):
        return [list(self[i,:]) for i in range(self.rows)]

    def _eval_vec(self):
        rows = self.rows

        def entry(n, _):
            # we want to read off the columns first
            j = n // rows
            i = n - j * rows
            return self[i, j]

        return self._new(len(self), 1, entry)

    def col_del(self, col):
        """Delete the specified column."""
        if col < 0:
            col += self.cols
        if not 0 <= col < self.cols:
            raise ValueError("Column {} out of range.".format(col))
        return self._eval_col_del(col)
```
### 83 - sympy/matrices/common.py:

Start line: 729, End line: 759

```python
class MatrixSpecial(MatrixRequired):

    @classmethod
    def diag(kls, *args, **kwargs):
        # ... other code

        def size(m):
            """Compute the size of the diagonal block"""
            if hasattr(m, 'rows'):
                return m.rows, m.cols
            return 1, 1
        diag_rows = sum(size(m)[0] for m in args)
        diag_cols =  sum(size(m)[1] for m in args)
        rows = kwargs.get('rows', diag_rows)
        cols = kwargs.get('cols', diag_cols)
        if rows < diag_rows or cols < diag_cols:
            raise ValueError("A {} x {} diagnal matrix cannot accommodate a"
                             "diagonal of size at least {} x {}.".format(rows, cols,
                                                                         diag_rows, diag_cols))

        # fill a default dict with the diagonal entries
        diag_entries = collections.defaultdict(lambda: S.Zero)
        row_pos, col_pos = 0, 0
        for m in args:
            if hasattr(m, 'rows'):
                # in this case, we're a matrix
                for i in range(m.rows):
                    for j in range(m.cols):
                        diag_entries[(i + row_pos, j + col_pos)] = m[i, j]
                row_pos += m.rows
                col_pos += m.cols
            else:
                # in this case, we're a single value
                diag_entries[(row_pos, col_pos)] = m
                row_pos += 1
                col_pos += 1
        return klass._eval_diag(rows, cols, diag_entries)
```
### 85 - sympy/matrices/common.py:

Start line: 112, End line: 136

```python
class MatrixShaping(MatrixRequired):

    def _eval_get_diag_blocks(self):
        sub_blocks = []

        def recurse_sub_blocks(M):
            i = 1
            while i <= M.shape[0]:
                if i == 1:
                    to_the_right = M[0, i:]
                    to_the_bottom = M[i:, 0]
                else:
                    to_the_right = M[:i, i:]
                    to_the_bottom = M[i:, :i]
                if any(to_the_right) or any(to_the_bottom):
                    i += 1
                    continue
                else:
                    sub_blocks.append(M[:i, :i])
                    if M.shape == M[:i, :i].shape:
                        return
                    else:
                        recurse_sub_blocks(M[i:, i:])
                        return

        recurse_sub_blocks(self)
        return sub_blocks
```
### 88 - sympy/matrices/common.py:

Start line: 457, End line: 485

```python
class MatrixShaping(MatrixRequired):

    def row_join(self, other):
        """Concatenates two matrices along self's last and rhs's first column

        Examples
        ========

        >>> from sympy import zeros, ones
        >>> M = zeros(3)
        >>> V = ones(3, 1)
        >>> M.row_join(V)
        Matrix([
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 1]])

        See Also
        ========

        row
        col_join
        """
        # A null matrix can always be stacked (see  #10770)
        if self.cols == 0 and self.rows != other.rows:
            return self._new(other.rows, 0, []).row_join(other)

        if self.rows != other.rows:
            raise ShapeError(
                "`self` and `rhs` must have the same number of rows.")
        return self._eval_row_join(other)
```
### 89 - sympy/matrices/common.py:

Start line: 73, End line: 110

```python
class MatrixShaping(MatrixRequired):
    """Provides basic matrix shaping and extracting of submatrices"""

    def _eval_col_del(self, col):
        def entry(i, j):
            return self[i, j] if j < col else self[i, j + 1]
        return self._new(self.rows, self.cols - 1, entry)

    def _eval_col_insert(self, pos, other):
        cols = self.cols

        def entry(i, j):
            if j < pos:
                return self[i, j]
            elif pos <= j < pos + other.cols:
                return other[i, j - pos]
            return self[i, j - other.cols]

        return self._new(self.rows, self.cols + other.cols,
                         lambda i, j: entry(i, j))

    def _eval_col_join(self, other):
        rows = self.rows

        def entry(i, j):
            if i < rows:
                return self[i, j]
            return other[i - rows, j]

        return classof(self, other)._new(self.rows + other.rows, self.cols,
                                         lambda i, j: entry(i, j))

    def _eval_extract(self, rowsList, colsList):
        mat = list(self)
        cols = self.cols
        indices = (i * cols + j for i in rowsList for j in colsList)
        return self._new(len(rowsList), len(colsList),
                         list(mat[i] for i in indices))
```
### 91 - sympy/matrices/common.py:

Start line: 2243, End line: 2262

```python
class _MatrixWrapper(object):
    """Wrapper class providing the minimum functionality
    for a matrix-like object: .rows, .cols, .shape, indexability,
    and iterability.  CommonMatrix math operations should work
    on matrix-like objects.  For example, wrapping a numpy
    matrix in a MatrixWrapper allows it to be passed to CommonMatrix.
    """
    is_MatrixLike = True

    def __init__(self, mat, shape=None):
        self.mat = mat
        self.rows, self.cols = mat.shape if shape is None else shape

    def __getattr__(self, attr):
        """Most attribute access is passed straight through
        to the stored matrix"""
        return getattr(self.mat, attr)

    def __getitem__(self, key):
        return self.mat.__getitem__(key)
```
### 114 - sympy/matrices/common.py:

Start line: 1553, End line: 1580

```python
class MatrixOperations(MatrixRequired):

    def conjugate(self):
        """Return the by-element conjugation.

        Examples
        ========

        >>> from sympy.matrices import SparseMatrix
        >>> from sympy import I
        >>> a = SparseMatrix(((1, 2 + I), (3, 4), (I, -I)))
        >>> a
        Matrix([
        [1, 2 + I],
        [3,     4],
        [I,    -I]])
        >>> a.C
        Matrix([
        [ 1, 2 - I],
        [ 3,     4],
        [-I,     I]])

        See Also
        ========

        transpose: Matrix transposition
        H: Hermite conjugation
        D: Dirac conjugation
        """
        return self._eval_conjugate()
```
### 115 - sympy/matrices/common.py:

Start line: 1753, End line: 1771

```python
class MatrixOperations(MatrixRequired):

    def replace(self, F, G, map=False):
        """Replaces Function F in Matrix entries with Function G.

        Examples
        ========

        >>> from sympy import symbols, Function, Matrix
        >>> F, G = symbols('F, G', cls=Function)
        >>> M = Matrix(2, 2, lambda i, j: F(i+j)) ; M
        Matrix([
        [F(0), F(1)],
        [F(1), F(2)]])
        >>> N = M.replace(F,G)
        >>> N
        Matrix([
        [G(0), G(1)],
        [G(1), G(2)]])
        """
        return self.applyfunc(lambda x: x.replace(F, G, map))
```
### 116 - sympy/matrices/common.py:

Start line: 584, End line: 604

```python
class MatrixShaping(MatrixRequired):

    @classmethod
    def vstack(cls, *args):
        """Return a matrix formed by joining args vertically (i.e.
        by repeated application of col_join).

        Examples
        ========

        >>> from sympy.matrices import Matrix, eye
        >>> Matrix.vstack(eye(2), 2*eye(2))
        Matrix([
        [1, 0],
        [0, 1],
        [2, 0],
        [0, 2]])
        """
        if len(args) == 0:
            return cls._new()

        kls = type(args[0])
        return reduce(kls.col_join, args)
```
