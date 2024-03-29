# sympy__sympy-13031

| **sympy/sympy** | `2dfa7457f20ee187fbb09b5b6a1631da4458388c` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | 47276 |
| **Avg pos** | 112.0 |
| **Min pos** | 112 |
| **Max pos** | 112 |
| **Top file pos** | 4 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sympy/matrices/sparse.py b/sympy/matrices/sparse.py
--- a/sympy/matrices/sparse.py
+++ b/sympy/matrices/sparse.py
@@ -985,8 +985,10 @@ def col_join(self, other):
         >>> C == A.row_insert(A.rows, Matrix(B))
         True
         """
-        if not self:
-            return type(self)(other)
+        # A null matrix can always be stacked (see  #10770)
+        if self.rows == 0 and self.cols != other.cols:
+            return self._new(0, other.cols, []).col_join(other)
+
         A, B = self, other
         if not A.cols == B.cols:
             raise ShapeError()
@@ -1191,8 +1193,10 @@ def row_join(self, other):
         >>> C == A.col_insert(A.cols, B)
         True
         """
-        if not self:
-            return type(self)(other)
+        # A null matrix can always be stacked (see  #10770)
+        if self.cols == 0 and self.rows != other.rows:
+            return self._new(other.rows, 0, []).row_join(other)
+
         A, B = self, other
         if not A.rows == B.rows:
             raise ShapeError()

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sympy/matrices/sparse.py | 988 | 989 | - | 4 | -
| sympy/matrices/sparse.py | 1194 | 1195 | 112 | 4 | 47276


## Problem Statement

```
Behavior of Matrix hstack and vstack changed in sympy 1.1
In sympy 1.0:
\`\`\`
import sympy as sy
M1 = sy.Matrix.zeros(0, 0)
M2 = sy.Matrix.zeros(0, 1)
M3 = sy.Matrix.zeros(0, 2)
M4 = sy.Matrix.zeros(0, 3)
sy.Matrix.hstack(M1, M2, M3, M4).shape
\`\`\`
returns 
`(0, 6)`

Now, same in sympy 1.1:
\`\`\`
import sympy as sy
M1 = sy.Matrix.zeros(0, 0)
M2 = sy.Matrix.zeros(0, 1)
M3 = sy.Matrix.zeros(0, 2)
M4 = sy.Matrix.zeros(0, 3)
sy.Matrix.hstack(M1, M2, M3, M4).shape
\`\`\`
returns
`(0, 3)
`
whereas:
\`\`\`
import sympy as sy
M1 = sy.Matrix.zeros(1, 0)
M2 = sy.Matrix.zeros(1, 1)
M3 = sy.Matrix.zeros(1, 2)
M4 = sy.Matrix.zeros(1, 3)
sy.Matrix.hstack(M1, M2, M3, M4).shape
\`\`\`
returns
`(1, 6)
`

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 sympy/matrices/common.py | 558 | 582| 125 | 125 | 17167 | 
| 2 | 2 sympy/matrices/dense.py | 431 | 474| 377 | 502 | 28633 | 
| 3 | 2 sympy/matrices/common.py | 363 | 381| 142 | 644 | 28633 | 
| 4 | 2 sympy/matrices/common.py | 584 | 604| 144 | 788 | 28633 | 
| 5 | 3 sympy/matrices/expressions/blockmatrix.py | 1 | 19| 206 | 994 | 32311 | 
| 6 | 3 sympy/matrices/expressions/blockmatrix.py | 160 | 195| 282 | 1276 | 32311 | 
| 7 | 3 sympy/matrices/common.py | 112 | 136| 175 | 1451 | 32311 | 
| 8 | **4 sympy/matrices/sparse.py** | 42 | 91| 457 | 1908 | 42686 | 
| 9 | 5 sympy/matrices/expressions/matexpr.py | 140 | 235| 683 | 2591 | 46512 | 
| 10 | 6 sympy/polys/polymatrix.py | 77 | 87| 143 | 2734 | 47455 | 
| 11 | **6 sympy/matrices/sparse.py** | 93 | 148| 465 | 3199 | 47455 | 
| 12 | 6 sympy/matrices/expressions/blockmatrix.py | 21 | 126| 821 | 4020 | 47455 | 
| 13 | 6 sympy/matrices/common.py | 138 | 180| 341 | 4361 | 47455 | 
| 14 | 7 examples/intermediate/vandermonde.py | 114 | 172| 535 | 4896 | 48846 | 
| 15 | 7 sympy/matrices/common.py | 2059 | 2088| 268 | 5164 | 48846 | 
| 16 | 7 sympy/polys/polymatrix.py | 55 | 75| 236 | 5400 | 48846 | 
| 17 | 7 sympy/matrices/common.py | 729 | 759| 338 | 5738 | 48846 | 
| 18 | 7 sympy/matrices/common.py | 1915 | 1940| 259 | 5997 | 48846 | 
| 19 | 8 sympy/matrices/immutable.py | 95 | 123| 272 | 6269 | 50215 | 
| 20 | 9 sympy/assumptions/handlers/matrices.py | 50 | 86| 268 | 6537 | 53477 | 
| 21 | **9 sympy/matrices/sparse.py** | 291 | 320| 341 | 6878 | 53477 | 
| 22 | 9 examples/intermediate/vandermonde.py | 1 | 39| 236 | 7114 | 53477 | 
| 23 | 10 sympy/printing/pretty/pretty.py | 708 | 730| 224 | 7338 | 72161 | 
| 24 | 10 sympy/matrices/dense.py | 163 | 206| 531 | 7869 | 72161 | 
| 25 | 10 sympy/printing/pretty/pretty.py | 732 | 801| 597 | 8466 | 72161 | 
| 26 | **10 sympy/matrices/sparse.py** | 271 | 289| 198 | 8664 | 72161 | 
| 27 | **10 sympy/matrices/sparse.py** | 1068 | 1092| 292 | 8956 | 72161 | 
| 28 | 11 sympy/tensor/array/__init__.py | 1 | 209| 2238 | 11194 | 74400 | 
| 29 | 12 sympy/tensor/indexed.py | 370 | 410| 328 | 11522 | 80017 | 
| 30 | 12 sympy/matrices/common.py | 1 | 40| 238 | 11760 | 80017 | 
| 31 | 13 sympy/physics/quantum/matrixcache.py | 83 | 103| 434 | 12194 | 81008 | 
| 32 | 13 sympy/matrices/expressions/matexpr.py | 391 | 446| 411 | 12605 | 81008 | 
| 33 | 13 sympy/matrices/common.py | 73 | 110| 310 | 12915 | 81008 | 
| 34 | **13 sympy/matrices/sparse.py** | 370 | 390| 197 | 13112 | 81008 | 
| 35 | 13 sympy/matrices/common.py | 2181 | 2232| 443 | 13555 | 81008 | 
| 36 | 13 sympy/matrices/common.py | 383 | 415| 271 | 13826 | 81008 | 
| 37 | 13 sympy/matrices/common.py | 2024 | 2057| 299 | 14125 | 81008 | 
| 38 | 13 sympy/matrices/expressions/matexpr.py | 32 | 138| 782 | 14907 | 81008 | 
| 39 | **13 sympy/matrices/sparse.py** | 322 | 343| 228 | 15135 | 81008 | 
| 40 | 14 sympy/matrices/expressions/slice.py | 71 | 85| 163 | 15298 | 81998 | 
| 41 | **14 sympy/matrices/sparse.py** | 1 | 17| 117 | 15415 | 81998 | 
| 42 | 15 sympy/polys/agca/homomorphisms.py | 418 | 508| 845 | 16260 | 87753 | 
| 43 | 15 sympy/matrices/common.py | 1479 | 1524| 326 | 16586 | 87753 | 
| 44 | 15 sympy/matrices/dense.py | 271 | 282| 130 | 16716 | 87753 | 
| 45 | 15 sympy/printing/pretty/pretty.py | 686 | 705| 207 | 16923 | 87753 | 
| 46 | 16 sympy/matrices/expressions/matmul.py | 75 | 130| 467 | 17390 | 90085 | 
| 47 | 16 sympy/physics/quantum/matrixcache.py | 66 | 80| 165 | 17555 | 90085 | 
| 48 | 17 sympy/matrices/__init__.py | 1 | 30| 260 | 17815 | 90345 | 
| 49 | 18 sympy/physics/quantum/matrixutils.py | 222 | 297| 556 | 18371 | 92938 | 
| 50 | 18 sympy/matrices/expressions/matexpr.py | 237 | 270| 351 | 18722 | 92938 | 
| 51 | 18 sympy/assumptions/handlers/matrices.py | 445 | 457| 122 | 18844 | 92938 | 
| 52 | 18 sympy/matrices/dense.py | 706 | 745| 273 | 19117 | 92938 | 
| 53 | 18 sympy/matrices/expressions/blockmatrix.py | 197 | 262| 522 | 19639 | 92938 | 
| 54 | 18 sympy/matrices/expressions/matexpr.py | 352 | 388| 316 | 19955 | 92938 | 
| 55 | 19 examples/advanced/qft.py | 85 | 138| 607 | 20562 | 94253 | 
| 56 | 19 sympy/polys/polymatrix.py | 1 | 53| 586 | 21148 | 94253 | 
| 57 | 20 sympy/holonomic/linearsolver.py | 35 | 47| 130 | 21278 | 94990 | 
| 58 | 20 sympy/matrices/immutable.py | 46 | 62| 185 | 21463 | 94990 | 
| 59 | **20 sympy/matrices/sparse.py** | 1033 | 1066| 281 | 21744 | 94990 | 
| 60 | 20 sympy/matrices/expressions/matmul.py | 1 | 12| 115 | 21859 | 94990 | 
| 61 | 20 sympy/matrices/expressions/slice.py | 88 | 115| 227 | 22086 | 94990 | 
| 62 | 20 sympy/matrices/common.py | 900 | 973| 621 | 22707 | 94990 | 
| 63 | **20 sympy/matrices/sparse.py** | 1246 | 1274| 238 | 22945 | 94990 | 
| 64 | 21 sympy/matrices/expressions/matpow.py | 1 | 48| 408 | 23353 | 95635 | 
| 65 | 21 sympy/assumptions/handlers/matrices.py | 472 | 486| 130 | 23483 | 95635 | 
| 66 | 21 sympy/physics/quantum/matrixutils.py | 1 | 62| 440 | 23923 | 95635 | 
| 67 | 21 sympy/matrices/dense.py | 529 | 554| 171 | 24094 | 95635 | 
| 68 | 21 sympy/matrices/expressions/matmul.py | 48 | 73| 272 | 24366 | 95635 | 
| 69 | 22 sympy/polys/fglmtools.py | 72 | 86| 111 | 24477 | 96959 | 
| 70 | 23 sympy/matrices/expressions/matadd.py | 1 | 14| 131 | 24608 | 97874 | 
| 71 | 23 sympy/physics/quantum/matrixutils.py | 78 | 110| 274 | 24882 | 97874 | 
| 72 | 23 sympy/matrices/common.py | 527 | 556| 216 | 25098 | 97874 | 
| 73 | 23 sympy/matrices/expressions/blockmatrix.py | 413 | 435| 217 | 25315 | 97874 | 
| 74 | 23 sympy/matrices/expressions/slice.py | 57 | 69| 174 | 25489 | 97874 | 
| 75 | **23 sympy/matrices/sparse.py** | 412 | 428| 158 | 25647 | 97874 | 
| 76 | 24 sympy/printing/str.py | 243 | 258| 142 | 25789 | 104351 | 
| 77 | 25 sympy/matrices/matrices.py | 785 | 1526| 6358 | 32147 | 140489 | 
| 78 | 25 sympy/assumptions/handlers/matrices.py | 459 | 469| 112 | 32259 | 140489 | 
| 79 | 26 sympy/vector/functions.py | 347 | 383| 245 | 32504 | 143942 | 
| 80 | 26 sympy/matrices/common.py | 220 | 273| 308 | 32812 | 143942 | 
| 81 | 26 sympy/matrices/common.py | 1885 | 1913| 288 | 33100 | 143942 | 
| 82 | 26 sympy/matrices/common.py | 457 | 485| 218 | 33318 | 143942 | 
| 83 | 26 sympy/matrices/expressions/matexpr.py | 301 | 324| 140 | 33458 | 143942 | 
| 84 | 26 sympy/matrices/common.py | 1713 | 1731| 132 | 33590 | 143942 | 
| 85 | 27 sympy/matrices/expressions/dotproduct.py | 29 | 44| 173 | 33763 | 144485 | 
| 86 | 27 sympy/tensor/indexed.py | 1 | 108| 931 | 34694 | 144485 | 
| 87 | 27 sympy/assumptions/handlers/matrices.py | 265 | 293| 185 | 34879 | 144485 | 
| 88 | 28 sympy/vector/dyadic.py | 130 | 173| 343 | 35222 | 146557 | 
| 89 | 28 sympy/matrices/common.py | 182 | 218| 240 | 35462 | 146557 | 
| 90 | 28 sympy/matrices/immutable.py | 1 | 14| 113 | 35575 | 146557 | 
| 91 | 28 sympy/matrices/common.py | 417 | 455| 254 | 35829 | 146557 | 
| 92 | 28 sympy/matrices/common.py | 2090 | 2133| 323 | 36152 | 146557 | 
| 93 | 28 sympy/printing/pretty/pretty.py | 880 | 912| 356 | 36508 | 146557 | 
| 94 | 28 sympy/matrices/expressions/slice.py | 1 | 30| 234 | 36742 | 146557 | 
| 95 | 28 sympy/matrices/matrices.py | 3234 | 4019| 6294 | 43036 | 146557 | 
| 96 | 29 sympy/diffgeom/diffgeom.py | 1 | 20| 171 | 43207 | 161135 | 
| 97 | 29 sympy/matrices/common.py | 487 | 525| 193 | 43400 | 161135 | 
| 98 | **29 sympy/matrices/sparse.py** | 1009 | 1031| 225 | 43625 | 161135 | 
| 99 | 30 sympy/polys/subresultants_qq_zz.py | 1900 | 1927| 311 | 43936 | 184758 | 
| 100 | 30 sympy/holonomic/linearsolver.py | 49 | 95| 392 | 44328 | 184758 | 
| 101 | 30 sympy/matrices/expressions/matadd.py | 64 | 79| 125 | 44453 | 184758 | 
| 102 | 30 sympy/matrices/common.py | 43 | 70| 246 | 44699 | 184758 | 
| 103 | 30 sympy/matrices/expressions/matexpr.py | 448 | 496| 258 | 44957 | 184758 | 
| 104 | 31 sympy/physics/matrices.py | 1 | 44| 243 | 45200 | 186240 | 
| 105 | 31 sympy/matrices/dense.py | 779 | 815| 214 | 45414 | 186240 | 
| 106 | 32 sympy/matrices/densearith.py | 187 | 209| 196 | 45610 | 187998 | 
| 107 | 32 sympy/matrices/expressions/blockmatrix.py | 301 | 319| 123 | 45733 | 187998 | 
| 108 | 33 sympy/physics/mechanics/linearize.py | 172 | 220| 565 | 46298 | 192235 | 
| 109 | 33 sympy/printing/pretty/pretty.py | 670 | 683| 141 | 46439 | 192235 | 
| 110 | 33 sympy/matrices/expressions/matexpr.py | 1 | 29| 214 | 46653 | 192235 | 
| 111 | 33 sympy/matrices/expressions/matexpr.py | 326 | 349| 150 | 46803 | 192235 | 
| **-> 112 <-** | **33 sympy/matrices/sparse.py** | 1159 | 1213| 473 | 47276 | 192235 | 
| 113 | 33 sympy/matrices/dense.py | 507 | 527| 182 | 47458 | 192235 | 
| 114 | 34 sympy/__init__.py | 1 | 93| 668 | 48126 | 192903 | 
| 115 | 34 sympy/matrices/matrices.py | 1583 | 2401| 6167 | 54293 | 192903 | 


### Hint

```
CC @siefkenj 
I update my comment in case someone already read it. We still have an issue with matrices shape in [pyphs](https://github.com/pyphs/pyphs/issues/49#issuecomment-316618994), but hstack and vstack seem ok in sympy 1.1.1rc1:

\`\`\`
>>> import sympy as sy
>>> sy.__version__
'1.1.1rc1'
>>> '1.1.1rc1'
'1.1.1rc1'
>>> matrices = [sy.Matrix.zeros(0, n) for n in range(4)]
>>> sy.Matrix.hstack(*matrices).shape
(0, 6)
>>> matrices = [sy.Matrix.zeros(1, n) for n in range(4)]
>>> sy.Matrix.hstack(*matrices).shape
(1, 6)
>>> matrices = [sy.Matrix.zeros(n, 0) for n in range(4)]
>>> sy.Matrix.vstack(*matrices).shape
(6, 0)
>>> matrices = [sy.Matrix.zeros(1, n) for n in range(4)]
>>> sy.Matrix.hstack(*matrices).shape
(1, 6)
>>> 
\`\`\`
The problem is solved with Matrix but not SparseMatrix:
\`\`\`
>>> import sympy as sy
>>> sy.__version__
'1.1.1rc1'
>>> matrices = [Matrix.zeros(0, n) for n in range(4)]
>>> Matrix.hstack(*matrices)
Matrix(0, 6, [])
>>> sparse_matrices = [SparseMatrix.zeros(0, n) for n in range(4)]
>>> SparseMatrix.hstack(*sparse_matrices)
Matrix(0, 3, [])
>>> 
\`\`\`
Bisected to 27e9ee425819fa09a4cbb8179fb38939cc693249. Should we revert that commit? CC @aravindkanna
Any thoughts? This is the last fix to potentially go in the 1.1.1 release, but I want to cut a release candidate today or tomorrow, so speak now, or hold your peace (until the next major release).
I am away at a conference. The change should be almost identical to the fix for dense matrices, if someone can manage to get a patch in. I *might* be able to do it tomorrow.
Okay.  I've looked this over and its convoluted...

`SparseMatrix` should impliment `_eval_col_join`.  `col_join` should not be implemented.  It is, and that is what `hstack` is calling, which is why my previous patch didn't fix `SparseMatrix`s as well.  However, the patch that @asmeurer referenced ensures that `SparseMatrix.row_join(DenseMatrix)` returns a `SparseMatrix` whereas `CommonMatrix.row_join(SparseMatrix, DenseMatrix)` returns a `classof(SparseMatrix, DenseMatrix)` which happens to be a `DenseMatrix`.  I don't think that these should behave differently.  This API needs to be better thought out.
So is there a simple fix that can be made for the release or should this be postponed?
```

## Patch

```diff
diff --git a/sympy/matrices/sparse.py b/sympy/matrices/sparse.py
--- a/sympy/matrices/sparse.py
+++ b/sympy/matrices/sparse.py
@@ -985,8 +985,10 @@ def col_join(self, other):
         >>> C == A.row_insert(A.rows, Matrix(B))
         True
         """
-        if not self:
-            return type(self)(other)
+        # A null matrix can always be stacked (see  #10770)
+        if self.rows == 0 and self.cols != other.cols:
+            return self._new(0, other.cols, []).col_join(other)
+
         A, B = self, other
         if not A.cols == B.cols:
             raise ShapeError()
@@ -1191,8 +1193,10 @@ def row_join(self, other):
         >>> C == A.col_insert(A.cols, B)
         True
         """
-        if not self:
-            return type(self)(other)
+        # A null matrix can always be stacked (see  #10770)
+        if self.cols == 0 and self.rows != other.rows:
+            return self._new(other.rows, 0, []).row_join(other)
+
         A, B = self, other
         if not A.rows == B.rows:
             raise ShapeError()

```

## Test Patch

```diff
diff --git a/sympy/matrices/tests/test_sparse.py b/sympy/matrices/tests/test_sparse.py
--- a/sympy/matrices/tests/test_sparse.py
+++ b/sympy/matrices/tests/test_sparse.py
@@ -26,6 +26,12 @@ def sparse_zeros(n):
     assert type(a.row_join(b)) == type(a)
     assert type(a.col_join(b)) == type(a)
 
+    # make sure 0 x n matrices get stacked correctly
+    sparse_matrices = [SparseMatrix.zeros(0, n) for n in range(4)]
+    assert SparseMatrix.hstack(*sparse_matrices) == Matrix(0, 6, [])
+    sparse_matrices = [SparseMatrix.zeros(n, 0) for n in range(4)]
+    assert SparseMatrix.vstack(*sparse_matrices) == Matrix(6, 0, [])
+
     # test element assignment
     a = SparseMatrix((
         (1, 0),

```


## Code snippets

### 1 - sympy/matrices/common.py:

Start line: 558, End line: 582

```python
class MatrixShaping(MatrixRequired):

    def vec(self):
        """Return the Matrix converted into a one column matrix by stacking columns

        Examples
        ========

        >>> from sympy import Matrix
        >>> m=Matrix([[1, 3], [2, 4]])
        >>> m
        Matrix([
        [1, 3],
        [2, 4]])
        >>> m.vec()
        Matrix([
        [1],
        [2],
        [3],
        [4]])

        See Also
        ========

        vech
        """
        return self._eval_vec()
```
### 2 - sympy/matrices/dense.py:

Start line: 431, End line: 474

```python
class MutableDenseMatrix(DenseMatrix, MatrixBase):

    def __setitem__(self, key, value):
        """

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
        rv = self._setitem(key, value)
        if rv is not None:
            i, j, value = rv
            self._mat[i*self.cols + j] = value
```
### 3 - sympy/matrices/common.py:

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
### 4 - sympy/matrices/common.py:

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
### 5 - sympy/matrices/expressions/blockmatrix.py:

Start line: 1, End line: 19

```python
from __future__ import print_function, division

from sympy import ask, Q
from sympy.core import Basic, Add, sympify
from sympy.core.compatibility import range
from sympy.strategies import typed, exhaust, condition, do_one, unpack
from sympy.strategies.traverse import bottom_up
from sympy.utilities import sift

from sympy.matrices.expressions.matexpr import MatrixExpr, ZeroMatrix, Identity
from sympy.matrices.expressions.matmul import MatMul
from sympy.matrices.expressions.matadd import MatAdd
from sympy.matrices.expressions.transpose import Transpose, transpose
from sympy.matrices.expressions.trace import Trace
from sympy.matrices.expressions.determinant import det, Determinant
from sympy.matrices.expressions.slice import MatrixSlice
from sympy.matrices.expressions.inverse import Inverse
from sympy.matrices import Matrix, ShapeError
from sympy.functions.elementary.complexes import re, im
```
### 6 - sympy/matrices/expressions/blockmatrix.py:

Start line: 160, End line: 195

```python
class BlockMatrix(MatrixExpr):

    def _entry(self, i, j):
        # Find row entry
        for row_block, numrows in enumerate(self.rowblocksizes):
            if (i < numrows) != False:
                break
            else:
                i -= numrows
        for col_block, numcols in enumerate(self.colblocksizes):
            if (j < numcols) != False:
                break
            else:
                j -= numcols
        return self.blocks[row_block, col_block][i, j]

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
### 7 - sympy/matrices/common.py:

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
### 8 - sympy/matrices/sparse.py:

Start line: 42, End line: 91

```python
class SparseMatrix(MatrixBase):

    def __new__(cls, *args, **kwargs):
        self = object.__new__(cls)
        if len(args) == 1 and isinstance(args[0], SparseMatrix):
            self.rows = args[0].rows
            self.cols = args[0].cols
            self._smat = dict(args[0]._smat)
            return self

        self._smat = {}

        if len(args) == 3:
            self.rows = as_int(args[0])
            self.cols = as_int(args[1])

            if isinstance(args[2], collections.Callable):
                op = args[2]
                for i in range(self.rows):
                    for j in range(self.cols):
                        value = self._sympify(
                            op(self._sympify(i), self._sympify(j)))
                        if value:
                            self._smat[(i, j)] = value
            elif isinstance(args[2], (dict, Dict)):
                # manual copy, copy.deepcopy() doesn't work
                for key in args[2].keys():
                    v = args[2][key]
                    if v:
                        self._smat[key] = self._sympify(v)
            elif is_sequence(args[2]):
                if len(args[2]) != self.rows*self.cols:
                    raise ValueError(
                        'List length (%s) != rows*columns (%s)' %
                        (len(args[2]), self.rows*self.cols))
                flat_list = args[2]
                for i in range(self.rows):
                    for j in range(self.cols):
                        value = self._sympify(flat_list[i*self.cols + j])
                        if value:
                            self._smat[(i, j)] = value
        else:
            # handle full matrix forms with _handle_creation_inputs
            r, c, _list = Matrix._handle_creation_inputs(*args)
            self.rows = r
            self.cols = c
            for i in range(self.rows):
                for j in range(self.cols):
                    value = _list[self.cols*i + j]
                    if value:
                        self._smat[(i, j)] = value
        return self
```
### 9 - sympy/matrices/expressions/matexpr.py:

Start line: 140, End line: 235

```python
class MatrixExpr(Basic):

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

    def _entry(self, i, j):
        raise NotImplementedError(
            "Indexing not implemented for %s" % self.__class__.__name__)

    def adjoint(self):
        return adjoint(self)

    def as_coeff_Mul(self, rational=False):
        """Efficiently extract the coefficient of a product. """
        return S.One, self

    def conjugate(self):
        return conjugate(self)

    def transpose(self):
        from sympy.matrices.expressions.transpose import transpose
        return transpose(self)

    T = property(transpose, None, None, 'Matrix transposition.')

    def inverse(self):
        return self._eval_inverse()

    @property
    def I(self):
        return self.inverse()

    def valid_index(self, i, j):
        def is_valid(idx):
            return isinstance(idx, (int, Integer, Symbol, Expr))
        return (is_valid(i) and is_valid(j) and
                (self.rows is None or
                (0 <= i) != False and (i < self.rows) != False) and
                (0 <= j) != False and (j < self.cols) != False)
```
### 10 - sympy/polys/polymatrix.py:

Start line: 77, End line: 87

```python
class MutablePolyDenseMatrix(MutableDenseMatrix):

    def _eval_scalar_mul(self, other):
        mat = [Poly(a.as_expr()*other, *a.gens) if isinstance(a, Poly) else a*other for a in self._mat]
        return self.__class__(self.rows, self.cols, mat, copy=False)

    def _eval_scalar_rmul(self, other):
        mat = [Poly(other*a.as_expr(), *a.gens) if isinstance(a, Poly) else other*a for a in self._mat]
        return self.__class__(self.rows, self.cols, mat, copy=False)


MutablePolyMatrix = PolyMatrix = MutablePolyDenseMatrix
```
### 11 - sympy/matrices/sparse.py:

Start line: 93, End line: 148

```python
class SparseMatrix(MatrixBase):

    def __eq__(self, other):
        try:
            if self.shape != other.shape:
                return False
            if isinstance(other, SparseMatrix):
                return self._smat == other._smat
            elif isinstance(other, MatrixBase):
                return self._smat == MutableSparseMatrix(other)._smat
        except AttributeError:
            return False

    def __getitem__(self, key):

        if isinstance(key, tuple):
            i, j = key
            try:
                i, j = self.key2ij(key)
                return self._smat.get((i, j), S.Zero)
            except (TypeError, IndexError):
                if isinstance(i, slice):
                    # XXX remove list() when PY2 support is dropped
                    i = list(range(self.rows))[i]
                elif is_sequence(i):
                    pass
                elif isinstance(i, Expr) and not i.is_number:
                    from sympy.matrices.expressions.matexpr import MatrixElement
                    return MatrixElement(self, i, j)
                else:
                    if i >= self.rows:
                        raise IndexError('Row index out of bounds')
                    i = [i]
                if isinstance(j, slice):
                    # XXX remove list() when PY2 support is dropped
                    j = list(range(self.cols))[j]
                elif is_sequence(j):
                    pass
                elif isinstance(j, Expr) and not j.is_number:
                    from sympy.matrices.expressions.matexpr import MatrixElement
                    return MatrixElement(self, i, j)
                else:
                    if j >= self.cols:
                        raise IndexError('Col index out of bounds')
                    j = [j]
                return self.extract(i, j)

        # check for single arg, like M[:] or M[3]
        if isinstance(key, slice):
            lo, hi = key.indices(len(self))[:2]
            L = []
            for i in range(lo, hi):
                m, n = divmod(i, self.cols)
                L.append(self._smat.get((m, n), S.Zero))
            return L

        i, j = divmod(a2idx(key, len(self)), self.cols)
        return self._smat.get((i, j), S.Zero)
```
### 21 - sympy/matrices/sparse.py:

Start line: 291, End line: 320

```python
class SparseMatrix(MatrixBase):

    def _eval_extract(self, rowsList, colsList):
        urow = list(uniq(rowsList))
        ucol = list(uniq(colsList))
        smat = {}
        if len(urow)*len(ucol) < len(self._smat):
            # there are fewer elements requested than there are elements in the matrix
            for i, r in enumerate(urow):
                for j, c in enumerate(ucol):
                    smat[i, j] = self._smat.get((r, c), 0)
        else:
            # most of the request will be zeros so check all of self's entries,
            # keeping only the ones that are desired
            for rk, ck in self._smat:
                if rk in urow and ck in ucol:
                    smat[(urow.index(rk), ucol.index(ck))] = self._smat[(rk, ck)]

        rv = self._new(len(urow), len(ucol), smat)
        # rv is nominally correct but there might be rows/cols
        # which require duplication
        if len(rowsList) != len(urow):
            for i, r in enumerate(rowsList):
                i_previous = rowsList.index(r)
                if i_previous != i:
                    rv = rv.row_insert(i, rv.row(i_previous))
        if len(colsList) != len(ucol):
            for i, c in enumerate(colsList):
                i_previous = colsList.index(c)
                if i_previous != i:
                    rv = rv.col_insert(i, rv.col(i_previous))
        return rv
```
### 26 - sympy/matrices/sparse.py:

Start line: 271, End line: 289

```python
class SparseMatrix(MatrixBase):

    def _eval_col_insert(self, icol, other):
        if not isinstance(other, SparseMatrix):
            other = SparseMatrix(other)
        new_smat = {}
        # make room for the new rows
        for key, val in self._smat.items():
            row, col = key
            if col >= icol:
                col += other.cols
            new_smat[(row, col)] = val
        # add other's keys
        for key, val in other._smat.items():
            row, col = key
            new_smat[(row, col + icol)] = val
        return self._new(self.rows, self.cols + other.cols, new_smat)

    def _eval_conjugate(self):
        smat = {key: val.conjugate() for key,val in self._smat.items()}
        return self._new(self.rows, self.cols, smat)
```
### 27 - sympy/matrices/sparse.py:

Start line: 1068, End line: 1092

```python
class MutableSparseMatrix(SparseMatrix, MatrixBase):

    def copyin_matrix(self, key, value):
        # include this here because it's not part of BaseMatrix
        rlo, rhi, clo, chi = self.key2bounds(key)
        shape = value.shape
        dr, dc = rhi - rlo, chi - clo
        if shape != (dr, dc):
            raise ShapeError(
                "The Matrix `value` doesn't have the same dimensions "
                "as the in sub-Matrix given by `key`.")
        if not isinstance(value, SparseMatrix):
            for i in range(value.rows):
                for j in range(value.cols):
                    self[i + rlo, j + clo] = value[i, j]
        else:
            if (rhi - rlo)*(chi - clo) < len(self):
                for i in range(rlo, rhi):
                    for j in range(clo, chi):
                        self._smat.pop((i, j), None)
            else:
                for i, j, v in self.row_list():
                    if rlo <= i < rhi and clo <= j < chi:
                        self._smat.pop((i, j), None)
            for k, v in value._smat.items():
                i, j = k
                self[i + rlo, j + clo] = value[i, j]
```
### 34 - sympy/matrices/sparse.py:

Start line: 370, End line: 390

```python
class SparseMatrix(MatrixBase):

    def _eval_row_insert(self, irow, other):
        if not isinstance(other, SparseMatrix):
            other = SparseMatrix(other)
        new_smat = {}
        # make room for the new rows
        for key, val in self._smat.items():
            row, col = key
            if row >= irow:
                row += other.rows
            new_smat[(row, col)] = val
        # add other's keys
        for key, val in other._smat.items():
            row, col = key
            new_smat[(row + irow, col)] = val
        return self._new(self.rows + other.rows, self.cols, new_smat)

    def _eval_scalar_mul(self, other):
        return self.applyfunc(lambda x: x*other)

    def _eval_scalar_rmul(self, other):
        return self.applyfunc(lambda x: other*x)
```
### 39 - sympy/matrices/sparse.py:

Start line: 322, End line: 343

```python
class SparseMatrix(MatrixBase):

    @classmethod
    def _eval_eye(cls, rows, cols):
        entries = {(i,i): S.One for i in range(min(rows, cols))}
        return cls._new(rows, cols, entries)

    def _eval_has(self, *patterns):
        # if the matrix has any zeros, see if S.Zero
        # has the pattern.  If _smat is full length,
        # the matrix has no zeros.
        zhas = S.Zero.has(*patterns)
        if len(self._smat) == self.rows*self.cols:
            zhas = False
        return any(self[key].has(*patterns) for key in self._smat) or zhas

    def _eval_is_Identity(self):
        if not all(self[i, i] == 1 for i in range(self.rows)):
            return False
        return len(self._smat) == self.rows

    def _eval_is_symmetric(self, simpfunc):
        diff = (self - self.T).applyfunc(simpfunc)
        return len(diff.values()) == 0
```
### 41 - sympy/matrices/sparse.py:

Start line: 1, End line: 17

```python
from __future__ import print_function, division

import copy
from collections import defaultdict

from sympy.core.containers import Dict
from sympy.core.expr import Expr
from sympy.core.compatibility import is_sequence, as_int, range
from sympy.core.logic import fuzzy_and
from sympy.core.singleton import S
from sympy.functions import Abs
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.utilities.iterables import uniq

from .matrices import MatrixBase, ShapeError, a2idx
from .dense import Matrix
import collections
```
### 59 - sympy/matrices/sparse.py:

Start line: 1033, End line: 1066

```python
class MutableSparseMatrix(SparseMatrix, MatrixBase):

    def col_swap(self, i, j):
        """Swap, in place, columns i and j.

        Examples
        ========

        >>> from sympy.matrices import SparseMatrix
        >>> S = SparseMatrix.eye(3); S[2, 1] = 2
        >>> S.col_swap(1, 0); S
        Matrix([
        [0, 1, 0],
        [1, 0, 0],
        [2, 0, 1]])
        """
        if i > j:
            i, j = j, i
        rows = self.col_list()
        temp = []
        for ii, jj, v in rows:
            if jj == i:
                self._smat.pop((ii, jj))
                temp.append((ii, v))
            elif jj == j:
                self._smat.pop((ii, jj))
                self._smat[ii, i] = v
            elif jj > j:
                break
        for k, v in temp:
            self._smat[k, j] = v

    def copyin_list(self, key, value):
        if not is_sequence(value):
            raise TypeError("`value` must be of type list or tuple.")
        self.copyin_matrix(key, Matrix(value))
```
### 63 - sympy/matrices/sparse.py:

Start line: 1246, End line: 1274

```python
class MutableSparseMatrix(SparseMatrix, MatrixBase):

    def row_swap(self, i, j):
        """Swap, in place, columns i and j.

        Examples
        ========

        >>> from sympy.matrices import SparseMatrix
        >>> S = SparseMatrix.eye(3); S[2, 1] = 2
        >>> S.row_swap(1, 0); S
        Matrix([
        [0, 1, 0],
        [1, 0, 0],
        [0, 2, 1]])
        """
        if i > j:
            i, j = j, i
        rows = self.row_list()
        temp = []
        for ii, jj, v in rows:
            if ii == i:
                self._smat.pop((ii, jj))
                temp.append((jj, v))
            elif ii == j:
                self._smat.pop((ii, jj))
                self._smat[i, jj] = v
            elif ii > j:
                break
        for k, v in temp:
            self._smat[j, k] = v
```
### 75 - sympy/matrices/sparse.py:

Start line: 412, End line: 428

```python
class SparseMatrix(MatrixBase):

    def _eval_values(self):
        return [v for k,v in self._smat.items() if not v.is_zero]

    @classmethod
    def _eval_zeros(cls, rows, cols):
        return cls._new(rows, cols, {})

    def _LDL_solve(self, rhs):
        # for speed reasons, this is not uncommented, but if you are
        # having difficulties, try uncommenting to make sure that the
        # input matrix is symmetric

        #assert self.is_symmetric()
        L, D = self._LDL_sparse()
        Z = L._lower_triangular_solve(rhs)
        Y = D._diagonal_solve(Z)
        return L.T._upper_triangular_solve(Y)
```
### 98 - sympy/matrices/sparse.py:

Start line: 1009, End line: 1031

```python
class MutableSparseMatrix(SparseMatrix, MatrixBase):

    def col_op(self, j, f):
        """In-place operation on col j using two-arg functor whose args are
        interpreted as (self[i, j], i) for i in range(self.rows).

        Examples
        ========

        >>> from sympy.matrices import SparseMatrix
        >>> M = SparseMatrix.eye(3)*2
        >>> M[1, 0] = -1
        >>> M.col_op(1, lambda v, i: v + 2*M[i, 0]); M
        Matrix([
        [ 2, 4, 0],
        [-1, 0, 0],
        [ 0, 0, 2]])
        """
        for i in range(self.rows):
            v = self._smat.get((i, j), S.Zero)
            fv = f(v, i)
            if fv:
                self._smat[(i, j)] = fv
            elif v:
                self._smat.pop((i, j))
```
### 112 - sympy/matrices/sparse.py:

Start line: 1159, End line: 1213

```python
class MutableSparseMatrix(SparseMatrix, MatrixBase):

    def row_join(self, other):
        """Returns B appended after A (column-wise augmenting)::

            [A B]

        Examples
        ========

        >>> from sympy import SparseMatrix, Matrix
        >>> A = SparseMatrix(((1, 0, 1), (0, 1, 0), (1, 1, 0)))
        >>> A
        Matrix([
        [1, 0, 1],
        [0, 1, 0],
        [1, 1, 0]])
        >>> B = SparseMatrix(((1, 0, 0), (0, 1, 0), (0, 0, 1)))
        >>> B
        Matrix([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]])
        >>> C = A.row_join(B); C
        Matrix([
        [1, 0, 1, 1, 0, 0],
        [0, 1, 0, 0, 1, 0],
        [1, 1, 0, 0, 0, 1]])
        >>> C == A.row_join(Matrix(B))
        True

        Joining at row ends is the same as appending columns at the end
        of the matrix:

        >>> C == A.col_insert(A.cols, B)
        True
        """
        if not self:
            return type(self)(other)
        A, B = self, other
        if not A.rows == B.rows:
            raise ShapeError()
        A = A.copy()
        if not isinstance(B, SparseMatrix):
            k = 0
            b = B._mat
            for i in range(B.rows):
                for j in range(B.cols):
                    v = b[k]
                    if v:
                        A._smat[(i, j + A.cols)] = v
                    k += 1
        else:
            for (i, j), v in B._smat.items():
                A._smat[(i, j + A.cols)] = v
        A.cols += B.cols
        return A
```
