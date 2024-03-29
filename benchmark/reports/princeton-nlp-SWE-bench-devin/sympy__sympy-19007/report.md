# sympy__sympy-19007

| **sympy/sympy** | `f9e030b57623bebdc2efa7f297c1b5ede08fcebf` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 3524 |
| **Any found context length** | 787 |
| **Avg pos** | 12.0 |
| **Min pos** | 2 |
| **Max pos** | 10 |
| **Top file pos** | 1 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sympy/matrices/expressions/blockmatrix.py b/sympy/matrices/expressions/blockmatrix.py
--- a/sympy/matrices/expressions/blockmatrix.py
+++ b/sympy/matrices/expressions/blockmatrix.py
@@ -7,7 +7,7 @@
 from sympy.utilities import sift
 from sympy.utilities.misc import filldedent
 
-from sympy.matrices.expressions.matexpr import MatrixExpr, ZeroMatrix, Identity
+from sympy.matrices.expressions.matexpr import MatrixExpr, ZeroMatrix, Identity, MatrixElement
 from sympy.matrices.expressions.matmul import MatMul
 from sympy.matrices.expressions.matadd import MatAdd
 from sympy.matrices.expressions.matpow import MatPow
@@ -234,16 +234,24 @@ def transpose(self):
 
     def _entry(self, i, j, **kwargs):
         # Find row entry
+        orig_i, orig_j = i, j
         for row_block, numrows in enumerate(self.rowblocksizes):
-            if (i < numrows) != False:
+            cmp = i < numrows
+            if cmp == True:
                 break
-            else:
+            elif cmp == False:
                 i -= numrows
+            elif row_block < self.blockshape[0] - 1:
+                # Can't tell which block and it's not the last one, return unevaluated
+                return MatrixElement(self, orig_i, orig_j)
         for col_block, numcols in enumerate(self.colblocksizes):
-            if (j < numcols) != False:
+            cmp = j < numcols
+            if cmp == True:
                 break
-            else:
+            elif cmp == False:
                 j -= numcols
+            elif col_block < self.blockshape[1] - 1:
+                return MatrixElement(self, orig_i, orig_j)
         return self.blocks[row_block, col_block][i, j]
 
     @property

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sympy/matrices/expressions/blockmatrix.py | 10 | 10 | 10 | 1 | 3524
| sympy/matrices/expressions/blockmatrix.py | 237 | 244 | 2 | 1 | 787


## Problem Statement

```
Wrong matrix element fetched from BlockMatrix
Given this code:
\`\`\`
from sympy import *
n, i = symbols('n, i', integer=True)
A = MatrixSymbol('A', 1, 1)
B = MatrixSymbol('B', n, 1)
C = BlockMatrix([[A], [B]])
print('C is')
pprint(C)
print('C[i, 0] is')
pprint(C[i, 0])
\`\`\`
I get this output:
\`\`\`
C is
⎡A⎤
⎢ ⎥
⎣B⎦
C[i, 0] is
(A)[i, 0]
\`\`\`
`(A)[i, 0]` is the wrong here. `C[i, 0]` should not be simplified as that element may come from either `A` or `B`.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 sympy/matrices/expressions/blockmatrix.py** | 22 | 77| 502 | 502 | 4671 | 
| **-> 2 <-** | **1 sympy/matrices/expressions/blockmatrix.py** | 235 | 270| 285 | 787 | 4671 | 
| 3 | **1 sympy/matrices/expressions/blockmatrix.py** | 130 | 210| 641 | 1428 | 4671 | 
| 4 | 2 sympy/matrices/expressions/matexpr.py | 696 | 726| 246 | 1674 | 13770 | 
| 5 | 3 sympy/matrices/matrices.py | 667 | 705| 285 | 1959 | 33217 | 
| 6 | **3 sympy/matrices/expressions/blockmatrix.py** | 498 | 507| 127 | 2086 | 33217 | 
| 7 | 4 sympy/assumptions/handlers/matrices.py | 537 | 550| 133 | 2219 | 37477 | 
| 8 | **4 sympy/matrices/expressions/blockmatrix.py** | 273 | 344| 567 | 2786 | 37477 | 
| 9 | **4 sympy/matrices/expressions/blockmatrix.py** | 78 | 128| 521 | 3307 | 37477 | 
| **-> 10 <-** | **4 sympy/matrices/expressions/blockmatrix.py** | 1 | 20| 217 | 3524 | 37477 | 
| 11 | **4 sympy/matrices/expressions/blockmatrix.py** | 453 | 476| 206 | 3730 | 37477 | 
| 12 | 5 sympy/matrices/expressions/permutation.py | 101 | 163| 423 | 4153 | 39440 | 
| 13 | **5 sympy/matrices/expressions/blockmatrix.py** | 347 | 397| 354 | 4507 | 39440 | 
| 14 | 5 sympy/matrices/expressions/matexpr.py | 758 | 828| 502 | 5009 | 39440 | 
| 15 | 6 sympy/codegen/array_utils.py | 1247 | 1324| 709 | 5718 | 52865 | 
| 16 | **6 sympy/matrices/expressions/blockmatrix.py** | 531 | 553| 217 | 5935 | 52865 | 
| 17 | 7 sympy/matrices/sparse.py | 220 | 274| 465 | 6400 | 62142 | 
| 18 | 7 sympy/matrices/matrices.py | 814 | 846| 343 | 6743 | 62142 | 
| 19 | 7 sympy/matrices/matrices.py | 1400 | 1424| 293 | 7036 | 62142 | 
| 20 | 7 sympy/matrices/expressions/matexpr.py | 728 | 755| 265 | 7301 | 62142 | 
| 21 | 8 sympy/matrices/common.py | 1007 | 1043| 347 | 7648 | 82935 | 
| 22 | 8 sympy/matrices/common.py | 2220 | 2253| 272 | 7920 | 82935 | 
| 23 | **8 sympy/matrices/expressions/blockmatrix.py** | 212 | 233| 167 | 8087 | 82935 | 
| 24 | 9 sympy/matrices/dense.py | 1146 | 1181| 249 | 8336 | 92891 | 
| 25 | 10 sympy/physics/optics/gaussopt.py | 138 | 196| 322 | 8658 | 98580 | 
| 26 | 10 sympy/matrices/matrices.py | 2279 | 2294| 130 | 8788 | 98580 | 
| 27 | **10 sympy/matrices/expressions/blockmatrix.py** | 419 | 431| 127 | 8915 | 98580 | 
| 28 | 10 sympy/matrices/common.py | 816 | 866| 515 | 9430 | 98580 | 
| 29 | 11 examples/intermediate/vandermonde.py | 1 | 38| 225 | 9655 | 99945 | 
| 30 | **11 sympy/matrices/expressions/blockmatrix.py** | 478 | 496| 142 | 9797 | 99945 | 
| 31 | **11 sympy/matrices/expressions/blockmatrix.py** | 399 | 417| 123 | 9920 | 99945 | 
| 32 | **11 sympy/matrices/expressions/blockmatrix.py** | 509 | 527| 187 | 10107 | 99945 | 
| 33 | 11 sympy/matrices/matrices.py | 734 | 753| 191 | 10298 | 99945 | 
| 34 | 11 sympy/matrices/matrices.py | 1505 | 1592| 697 | 10995 | 99945 | 
| 35 | 11 sympy/matrices/matrices.py | 963 | 1129| 1548 | 12543 | 99945 | 
| 36 | 11 sympy/matrices/matrices.py | 2243 | 2276| 509 | 13052 | 99945 | 
| 37 | 12 sympy/solvers/solveset.py | 2247 | 2340| 750 | 13802 | 130714 | 
| 38 | 12 sympy/matrices/expressions/matexpr.py | 830 | 842| 157 | 13959 | 130714 | 
| 39 | 12 sympy/matrices/sparse.py | 650 | 676| 291 | 14250 | 130714 | 
| 40 | 13 sympy/solvers/solvers.py | 2418 | 2463| 323 | 14573 | 163389 | 
| 41 | 13 sympy/matrices/sparse.py | 321 | 350| 340 | 14913 | 163389 | 
| 42 | 14 sympy/physics/mechanics/linearize.py | 173 | 221| 565 | 15478 | 167633 | 
| 43 | 14 sympy/matrices/expressions/matexpr.py | 399 | 442| 418 | 15896 | 167633 | 
| 44 | 15 sympy/matrices/determinant.py | 1 | 15| 132 | 16028 | 174868 | 
| 45 | 15 sympy/matrices/matrices.py | 792 | 812| 177 | 16205 | 174868 | 
| 46 | 15 sympy/matrices/matrices.py | 1 | 60| 643 | 16848 | 174868 | 
| 47 | 15 sympy/matrices/expressions/matexpr.py | 285 | 318| 361 | 17209 | 174868 | 
| 48 | 15 sympy/matrices/common.py | 127 | 151| 175 | 17384 | 174868 | 
| 49 | 15 sympy/matrices/expressions/matexpr.py | 845 | 903| 304 | 17688 | 174868 | 
| 50 | 16 sympy/printing/pretty/pretty.py | 789 | 808| 207 | 17895 | 198300 | 


### Hint

```
I was aware of the problem that the coordinates were loosely handled even if the matrix had symbolic dimensions
I also think that `C[3, 0]` should be undefined because there is no guarantee that n is sufficiently large to contain elements.
`C[3, 0]` should just stay unevaluated, since it might be valid (I assume that's what you mean by 'undefined'). It should be possible to handle some cases properly, for example `C[n, 0]` should return `B[n - 1, 0]`.

If I get some time I might have a go at it, seems to be a nice first PR.

**EDIT:** Sorry that's not even true. If `n` is zero, then `C[n, 0]` is not `B[n - 1, 0]`.
```

## Patch

```diff
diff --git a/sympy/matrices/expressions/blockmatrix.py b/sympy/matrices/expressions/blockmatrix.py
--- a/sympy/matrices/expressions/blockmatrix.py
+++ b/sympy/matrices/expressions/blockmatrix.py
@@ -7,7 +7,7 @@
 from sympy.utilities import sift
 from sympy.utilities.misc import filldedent
 
-from sympy.matrices.expressions.matexpr import MatrixExpr, ZeroMatrix, Identity
+from sympy.matrices.expressions.matexpr import MatrixExpr, ZeroMatrix, Identity, MatrixElement
 from sympy.matrices.expressions.matmul import MatMul
 from sympy.matrices.expressions.matadd import MatAdd
 from sympy.matrices.expressions.matpow import MatPow
@@ -234,16 +234,24 @@ def transpose(self):
 
     def _entry(self, i, j, **kwargs):
         # Find row entry
+        orig_i, orig_j = i, j
         for row_block, numrows in enumerate(self.rowblocksizes):
-            if (i < numrows) != False:
+            cmp = i < numrows
+            if cmp == True:
                 break
-            else:
+            elif cmp == False:
                 i -= numrows
+            elif row_block < self.blockshape[0] - 1:
+                # Can't tell which block and it's not the last one, return unevaluated
+                return MatrixElement(self, orig_i, orig_j)
         for col_block, numcols in enumerate(self.colblocksizes):
-            if (j < numcols) != False:
+            cmp = j < numcols
+            if cmp == True:
                 break
-            else:
+            elif cmp == False:
                 j -= numcols
+            elif col_block < self.blockshape[1] - 1:
+                return MatrixElement(self, orig_i, orig_j)
         return self.blocks[row_block, col_block][i, j]
 
     @property

```

## Test Patch

```diff
diff --git a/sympy/matrices/expressions/tests/test_blockmatrix.py b/sympy/matrices/expressions/tests/test_blockmatrix.py
--- a/sympy/matrices/expressions/tests/test_blockmatrix.py
+++ b/sympy/matrices/expressions/tests/test_blockmatrix.py
@@ -192,7 +192,6 @@ def test_BlockDiagMatrix():
 def test_blockcut():
     A = MatrixSymbol('A', n, m)
     B = blockcut(A, (n/2, n/2), (m/2, m/2))
-    assert A[i, j] == B[i, j]
     assert B == BlockMatrix([[A[:n/2, :m/2], A[:n/2, m/2:]],
                              [A[n/2:, :m/2], A[n/2:, m/2:]]])
 
diff --git a/sympy/matrices/expressions/tests/test_indexing.py b/sympy/matrices/expressions/tests/test_indexing.py
--- a/sympy/matrices/expressions/tests/test_indexing.py
+++ b/sympy/matrices/expressions/tests/test_indexing.py
@@ -1,7 +1,7 @@
 from sympy import (symbols, MatrixSymbol, MatPow, BlockMatrix, KroneckerDelta,
         Identity, ZeroMatrix, ImmutableMatrix, eye, Sum, Dummy, trace,
         Symbol)
-from sympy.testing.pytest import raises
+from sympy.testing.pytest import raises, XFAIL
 from sympy.matrices.expressions.matexpr import MatrixElement, MatrixExpr
 
 k, l, m, n = symbols('k l m n', integer=True)
@@ -83,6 +83,72 @@ def test_block_index():
     assert BI.as_explicit().equals(eye(6))
 
 
+def test_block_index_symbolic():
+    # Note that these matrices may be zero-sized and indices may be negative, which causes
+    # all naive simplifications given in the comments to be invalid
+    A1 = MatrixSymbol('A1', n, k)
+    A2 = MatrixSymbol('A2', n, l)
+    A3 = MatrixSymbol('A3', m, k)
+    A4 = MatrixSymbol('A4', m, l)
+    A = BlockMatrix([[A1, A2], [A3, A4]])
+    assert A[0, 0] == MatrixElement(A, 0, 0)  # Cannot be A1[0, 0]
+    assert A[n - 1, k - 1] == A1[n - 1, k - 1]
+    assert A[n, k] == A4[0, 0]
+    assert A[n + m - 1, 0] == MatrixElement(A, n + m - 1, 0)  # Cannot be A3[m - 1, 0]
+    assert A[0, k + l - 1] == MatrixElement(A, 0, k + l - 1)  # Cannot be A2[0, l - 1]
+    assert A[n + m - 1, k + l - 1] == MatrixElement(A, n + m - 1, k + l - 1)  # Cannot be A4[m - 1, l - 1]
+    assert A[i, j] == MatrixElement(A, i, j)
+    assert A[n + i, k + j] == MatrixElement(A, n + i, k + j)  # Cannot be A4[i, j]
+    assert A[n - i - 1, k - j - 1] == MatrixElement(A, n - i - 1, k - j - 1)  # Cannot be A1[n - i - 1, k - j - 1]
+
+
+def test_block_index_symbolic_nonzero():
+    # All invalid simplifications from test_block_index_symbolic() that become valid if all
+    # matrices have nonzero size and all indices are nonnegative
+    k, l, m, n = symbols('k l m n', integer=True, positive=True)
+    i, j = symbols('i j', integer=True, nonnegative=True)
+    A1 = MatrixSymbol('A1', n, k)
+    A2 = MatrixSymbol('A2', n, l)
+    A3 = MatrixSymbol('A3', m, k)
+    A4 = MatrixSymbol('A4', m, l)
+    A = BlockMatrix([[A1, A2], [A3, A4]])
+    assert A[0, 0] == A1[0, 0]
+    assert A[n + m - 1, 0] == A3[m - 1, 0]
+    assert A[0, k + l - 1] == A2[0, l - 1]
+    assert A[n + m - 1, k + l - 1] == A4[m - 1, l - 1]
+    assert A[i, j] == MatrixElement(A, i, j)
+    assert A[n + i, k + j] == A4[i, j]
+    assert A[n - i - 1, k - j - 1] == A1[n - i - 1, k - j - 1]
+    assert A[2 * n, 2 * k] == A4[n, k]
+
+
+def test_block_index_large():
+    n, m, k = symbols('n m k', integer=True, positive=True)
+    i = symbols('i', integer=True, nonnegative=True)
+    A1 = MatrixSymbol('A1', n, n)
+    A2 = MatrixSymbol('A2', n, m)
+    A3 = MatrixSymbol('A3', n, k)
+    A4 = MatrixSymbol('A4', m, n)
+    A5 = MatrixSymbol('A5', m, m)
+    A6 = MatrixSymbol('A6', m, k)
+    A7 = MatrixSymbol('A7', k, n)
+    A8 = MatrixSymbol('A8', k, m)
+    A9 = MatrixSymbol('A9', k, k)
+    A = BlockMatrix([[A1, A2, A3], [A4, A5, A6], [A7, A8, A9]])
+    assert A[n + i, n + i] == MatrixElement(A, n + i, n + i)
+
+
+@XFAIL
+def test_block_index_symbolic_fail():
+    # To make this work, symbolic matrix dimensions would need to be somehow assumed nonnegative
+    # even if the symbols aren't specified as such.  Then 2 * n < n would correctly evaluate to
+    # False in BlockMatrix._entry()
+    A1 = MatrixSymbol('A1', n, 1)
+    A2 = MatrixSymbol('A2', m, 1)
+    A = BlockMatrix([[A1], [A2]])
+    assert A[2 * n, 0] == A2[n, 0]
+
+
 def test_slicing():
     A.as_explicit()[0, :]  # does not raise an error
 

```


## Code snippets

### 1 - sympy/matrices/expressions/blockmatrix.py:

Start line: 22, End line: 77

```python
class BlockMatrix(MatrixExpr):
    """A BlockMatrix is a Matrix comprised of other matrices.

    The submatrices are stored in a SymPy Matrix object but accessed as part of
    a Matrix Expression

    >>> from sympy import (MatrixSymbol, BlockMatrix, symbols,
    ...     Identity, ZeroMatrix, block_collapse)
    >>> n,m,l = symbols('n m l')
    >>> X = MatrixSymbol('X', n, n)
    >>> Y = MatrixSymbol('Y', m ,m)
    >>> Z = MatrixSymbol('Z', n, m)
    >>> B = BlockMatrix([[X, Z], [ZeroMatrix(m,n), Y]])
    >>> print(B)
    Matrix([
    [X, Z],
    [0, Y]])

    >>> C = BlockMatrix([[Identity(n), Z]])
    >>> print(C)
    Matrix([[I, Z]])

    >>> print(block_collapse(C*B))
    Matrix([[X, Z + Z*Y]])

    Some matrices might be comprised of rows of blocks with
    the matrices in each row having the same height and the
    rows all having the same total number of columns but
    not having the same number of columns for each matrix
    in each row. In this case, the matrix is not a block
    matrix and should be instantiated by Matrix.

    >>> from sympy import ones, Matrix
    >>> dat = [
    ... [ones(3,2), ones(3,3)*2],
    ... [ones(2,3)*3, ones(2,2)*4]]
    ...
    >>> BlockMatrix(dat)
    Traceback (most recent call last):
    ...
    ValueError:
    Although this matrix is comprised of blocks, the blocks do not fill
    the matrix in a size-symmetric fashion. To create a full matrix from
    these arguments, pass them directly to Matrix.
    >>> Matrix(dat)
    Matrix([
    [1, 1, 2, 2, 2],
    [1, 1, 2, 2, 2],
    [1, 1, 2, 2, 2],
    [3, 3, 3, 4, 4],
    [3, 3, 3, 4, 4]])

    See Also
    ========
    sympy.matrices.matrices.MatrixBase.irregular
    """
```
### 2 - sympy/matrices/expressions/blockmatrix.py:

Start line: 235, End line: 270

```python
class BlockMatrix(MatrixExpr):

    def _entry(self, i, j, **kwargs):
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
### 3 - sympy/matrices/expressions/blockmatrix.py:

Start line: 130, End line: 210

```python
class BlockMatrix(MatrixExpr):

    @property
    def shape(self):
        numrows = numcols = 0
        M = self.blocks
        for i in range(M.shape[0]):
            numrows += M[i, 0].shape[0]
        for i in range(M.shape[1]):
            numcols += M[0, i].shape[1]
        return (numrows, numcols)

    @property
    def blockshape(self):
        return self.blocks.shape

    @property
    def blocks(self):
        return self.args[0]

    @property
    def rowblocksizes(self):
        return [self.blocks[i, 0].rows for i in range(self.blockshape[0])]

    @property
    def colblocksizes(self):
        return [self.blocks[0, i].cols for i in range(self.blockshape[1])]

    def structurally_equal(self, other):
        return (isinstance(other, BlockMatrix)
            and self.shape == other.shape
            and self.blockshape == other.blockshape
            and self.rowblocksizes == other.rowblocksizes
            and self.colblocksizes == other.colblocksizes)

    def _blockmul(self, other):
        if (isinstance(other, BlockMatrix) and
                self.colblocksizes == other.rowblocksizes):
            return BlockMatrix(self.blocks*other.blocks)

        return self * other

    def _blockadd(self, other):
        if (isinstance(other, BlockMatrix)
                and self.structurally_equal(other)):
            return BlockMatrix(self.blocks + other.blocks)

        return self + other

    def _eval_transpose(self):
        # Flip all the individual matrices
        matrices = [transpose(matrix) for matrix in self.blocks]
        # Make a copy
        M = Matrix(self.blockshape[0], self.blockshape[1], matrices)
        # Transpose the block structure
        M = M.transpose()
        return BlockMatrix(M)

    def _eval_trace(self):
        if self.rowblocksizes == self.colblocksizes:
            return Add(*[Trace(self.blocks[i, i])
                        for i in range(self.blockshape[0])])
        raise NotImplementedError(
            "Can't perform trace of irregular blockshape")

    def _eval_determinant(self):
        if self.blockshape == (2, 2):
            [[A, B],
             [C, D]] = self.blocks.tolist()
            if ask(Q.invertible(A)):
                return det(A)*det(D - C*A.I*B)
            elif ask(Q.invertible(D)):
                return det(D)*det(A - B*D.I*C)
        return Determinant(self)

    def as_real_imag(self):
        real_matrices = [re(matrix) for matrix in self.blocks]
        real_matrices = Matrix(self.blockshape[0], self.blockshape[1], real_matrices)

        im_matrices = [im(matrix) for matrix in self.blocks]
        im_matrices = Matrix(self.blockshape[0], self.blockshape[1], im_matrices)

        return (real_matrices, im_matrices)
```
### 4 - sympy/matrices/expressions/matexpr.py:

Start line: 696, End line: 726

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
### 5 - sympy/matrices/matrices.py:

Start line: 667, End line: 705

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
### 6 - sympy/matrices/expressions/blockmatrix.py:

Start line: 498, End line: 507

```python
def blockinverse_2x2(expr):
    if isinstance(expr.arg, BlockMatrix) and expr.arg.blockshape == (2, 2):
        # Cite: The Matrix Cookbook Section 9.1.3
        [[A, B],
         [C, D]] = expr.arg.blocks.tolist()

        return BlockMatrix([[ (A - B*D.I*C).I,  (-A).I*B*(D - C*A.I*B).I],
                            [-(D - C*A.I*B).I*C*A.I,     (D - C*A.I*B).I]])
    else:
        return expr
```
### 7 - sympy/assumptions/handlers/matrices.py:

Start line: 537, End line: 550

```python
def BM_elements(predicate, expr, assumptions):
    """ Block Matrix elements """
    return all(ask(predicate(b), assumptions) for b in expr.blocks)

def MS_elements(predicate, expr, assumptions):
    """ Matrix Slice elements """
    return ask(predicate(expr.parent), assumptions)

def MatMul_elements(matrix_predicate, scalar_predicate, expr, assumptions):
    d = sift(expr.args, lambda x: isinstance(x, MatrixExpr))
    factors, matrices = d[False], d[True]
    return fuzzy_and([
        test_closed_group(Basic(*factors), assumptions, scalar_predicate),
        test_closed_group(Basic(*matrices), assumptions, matrix_predicate)])
```
### 8 - sympy/matrices/expressions/blockmatrix.py:

Start line: 273, End line: 344

```python
class BlockDiagMatrix(BlockMatrix):
    """
    A BlockDiagMatrix is a BlockMatrix with matrices only along the diagonal

    >>> from sympy import MatrixSymbol, BlockDiagMatrix, symbols, Identity
    >>> n, m, l = symbols('n m l')
    >>> X = MatrixSymbol('X', n, n)
    >>> Y = MatrixSymbol('Y', m ,m)
    >>> BlockDiagMatrix(X, Y)
    Matrix([
    [X, 0],
    [0, Y]])

    See Also
    ========
    sympy.matrices.dense.diag
    """
    def __new__(cls, *mats):
        return Basic.__new__(BlockDiagMatrix, *mats)

    @property
    def diag(self):
        return self.args

    @property
    def blocks(self):
        from sympy.matrices.immutable import ImmutableDenseMatrix
        mats = self.args
        data = [[mats[i] if i == j else ZeroMatrix(mats[i].rows, mats[j].cols)
                        for j in range(len(mats))]
                        for i in range(len(mats))]
        return ImmutableDenseMatrix(data, evaluate=False)

    @property
    def shape(self):
        return (sum(block.rows for block in self.args),
                sum(block.cols for block in self.args))

    @property
    def blockshape(self):
        n = len(self.args)
        return (n, n)

    @property
    def rowblocksizes(self):
        return [block.rows for block in self.args]

    @property
    def colblocksizes(self):
        return [block.cols for block in self.args]

    def _eval_inverse(self, expand='ignored'):
        return BlockDiagMatrix(*[mat.inverse() for mat in self.args])

    def _eval_transpose(self):
        return BlockDiagMatrix(*[mat.transpose() for mat in self.args])

    def _blockmul(self, other):
        if (isinstance(other, BlockDiagMatrix) and
                self.colblocksizes == other.rowblocksizes):
            return BlockDiagMatrix(*[a*b for a, b in zip(self.args, other.args)])
        else:
            return BlockMatrix._blockmul(self, other)

    def _blockadd(self, other):
        if (isinstance(other, BlockDiagMatrix) and
                self.blockshape == other.blockshape and
                self.rowblocksizes == other.rowblocksizes and
                self.colblocksizes == other.colblocksizes):
            return BlockDiagMatrix(*[a + b for a, b in zip(self.args, other.args)])
        else:
            return BlockMatrix._blockadd(self, other)
```
### 9 - sympy/matrices/expressions/blockmatrix.py:

Start line: 78, End line: 128

```python
class BlockMatrix(MatrixExpr):
    def __new__(cls, *args, **kwargs):
        from sympy.matrices.immutable import ImmutableDenseMatrix
        from sympy.utilities.iterables import is_sequence
        isMat = lambda i: getattr(i, 'is_Matrix', False)
        if len(args) != 1 or \
                not is_sequence(args[0]) or \
                len(set([isMat(r) for r in args[0]])) != 1:
            raise ValueError(filldedent('''
                expecting a sequence of 1 or more rows
                containing Matrices.'''))
        rows = args[0] if args else []
        if not isMat(rows):
            if rows and isMat(rows[0]):
                rows = [rows]  # rows is not list of lists or []
            # regularity check
            # same number of matrices in each row
            blocky = ok = len(set([len(r) for r in rows])) == 1
            if ok:
                # same number of rows for each matrix in a row
                for r in rows:
                    ok = len(set([i.rows for i in r])) == 1
                    if not ok:
                        break
                blocky = ok
                # same number of cols for each matrix in each col
                for c in range(len(rows[0])):
                    ok = len(set([rows[i][c].cols
                        for i in range(len(rows))])) == 1
                    if not ok:
                        break
            if not ok:
                # same total cols in each row
                ok = len(set([
                    sum([i.cols for i in r]) for r in rows])) == 1
                if blocky and ok:
                    raise ValueError(filldedent('''
                        Although this matrix is comprised of blocks,
                        the blocks do not fill the matrix in a
                        size-symmetric fashion. To create a full matrix
                        from these arguments, pass them directly to
                        Matrix.'''))
                raise ValueError(filldedent('''
                    When there are not the same number of rows in each
                    row's matrices or there are not the same number of
                    total columns in each row, the matrix is not a
                    block matrix. If this matrix is known to consist of
                    blocks fully filling a 2-D space then see
                    Matrix.irregular.'''))
        mat = ImmutableDenseMatrix(rows, evaluate=False)
        obj = Basic.__new__(cls, mat)
        return obj
```
### 10 - sympy/matrices/expressions/blockmatrix.py:

Start line: 1, End line: 20

```python
from __future__ import print_function, division

from sympy import ask, Q
from sympy.core import Basic, Add
from sympy.strategies import typed, exhaust, condition, do_one, unpack
from sympy.strategies.traverse import bottom_up
from sympy.utilities import sift
from sympy.utilities.misc import filldedent

from sympy.matrices.expressions.matexpr import MatrixExpr, ZeroMatrix, Identity
from sympy.matrices.expressions.matmul import MatMul
from sympy.matrices.expressions.matadd import MatAdd
from sympy.matrices.expressions.matpow import MatPow
from sympy.matrices.expressions.transpose import Transpose, transpose
from sympy.matrices.expressions.trace import Trace
from sympy.matrices.expressions.determinant import det, Determinant
from sympy.matrices.expressions.slice import MatrixSlice
from sympy.matrices.expressions.inverse import Inverse
from sympy.matrices import Matrix, ShapeError
from sympy.functions.elementary.complexes import re, im
```
### 11 - sympy/matrices/expressions/blockmatrix.py:

Start line: 453, End line: 476

```python
def bc_matmul(expr):
    if isinstance(expr, MatPow):
        if expr.args[1].is_Integer:
            factor, matrices = (1, [expr.args[0]]*expr.args[1])
        else:
            return expr
    else:
        factor, matrices = expr.as_coeff_matrices()

    i = 0
    while (i+1 < len(matrices)):
        A, B = matrices[i:i+2]
        if isinstance(A, BlockMatrix) and isinstance(B, BlockMatrix):
            matrices[i] = A._blockmul(B)
            matrices.pop(i+1)
        elif isinstance(A, BlockMatrix):
            matrices[i] = A._blockmul(BlockMatrix([[B]]))
            matrices.pop(i+1)
        elif isinstance(B, BlockMatrix):
            matrices[i] = BlockMatrix([[A]])._blockmul(B)
            matrices.pop(i+1)
        else:
            i+=1
    return MatMul(factor, *matrices).doit()
```
### 13 - sympy/matrices/expressions/blockmatrix.py:

Start line: 347, End line: 397

```python
def block_collapse(expr):
    """Evaluates a block matrix expression

    >>> from sympy import MatrixSymbol, BlockMatrix, symbols, \
                          Identity, Matrix, ZeroMatrix, block_collapse
    >>> n,m,l = symbols('n m l')
    >>> X = MatrixSymbol('X', n, n)
    >>> Y = MatrixSymbol('Y', m ,m)
    >>> Z = MatrixSymbol('Z', n, m)
    >>> B = BlockMatrix([[X, Z], [ZeroMatrix(m, n), Y]])
    >>> print(B)
    Matrix([
    [X, Z],
    [0, Y]])

    >>> C = BlockMatrix([[Identity(n), Z]])
    >>> print(C)
    Matrix([[I, Z]])

    >>> print(block_collapse(C*B))
    Matrix([[X, Z + Z*Y]])
    """
    from sympy.strategies.util import expr_fns

    hasbm = lambda expr: isinstance(expr, MatrixExpr) and expr.has(BlockMatrix)

    conditioned_rl = condition(
        hasbm,
        typed(
            {MatAdd: do_one(bc_matadd, bc_block_plus_ident),
             MatMul: do_one(bc_matmul, bc_dist),
             MatPow: bc_matmul,
             Transpose: bc_transpose,
             Inverse: bc_inverse,
             BlockMatrix: do_one(bc_unpack, deblock)}
        )
    )

    rule = exhaust(
        bottom_up(
            exhaust(conditioned_rl),
            fns=expr_fns
        )
    )

    result = rule(expr)
    doit = getattr(result, 'doit', None)
    if doit is not None:
        return doit()
    else:
        return result
```
### 16 - sympy/matrices/expressions/blockmatrix.py:

Start line: 531, End line: 553

```python
def reblock_2x2(B):
    """ Reblock a BlockMatrix so that it has 2x2 blocks of block matrices """
    if not isinstance(B, BlockMatrix) or not all(d > 2 for d in B.blocks.shape):
        return B

    BM = BlockMatrix  # for brevity's sake
    return BM([[   B.blocks[0,  0],  BM(B.blocks[0,  1:])],
               [BM(B.blocks[1:, 0]), BM(B.blocks[1:, 1:])]])


def bounds(sizes):
    """ Convert sequence of numbers into pairs of low-high pairs

    >>> from sympy.matrices.expressions.blockmatrix import bounds
    >>> bounds((1, 10, 50))
    [(0, 1), (1, 11), (11, 61)]
    """
    low = 0
    rv = []
    for size in sizes:
        rv.append((low, low + size))
        low += size
    return rv
```
### 23 - sympy/matrices/expressions/blockmatrix.py:

Start line: 212, End line: 233

```python
class BlockMatrix(MatrixExpr):

    def transpose(self):
        """Return transpose of matrix.

        Examples
        ========

        >>> from sympy import MatrixSymbol, BlockMatrix, ZeroMatrix
        >>> from sympy.abc import l, m, n
        >>> X = MatrixSymbol('X', n, n)
        >>> Y = MatrixSymbol('Y', m ,m)
        >>> Z = MatrixSymbol('Z', n, m)
        >>> B = BlockMatrix([[X, Z], [ZeroMatrix(m,n), Y]])
        >>> B.transpose()
        Matrix([
        [X.T,  0],
        [Z.T, Y.T]])
        >>> _.transpose()
        Matrix([
        [X, Z],
        [0, Y]])
        """
        return self._eval_transpose()
```
### 27 - sympy/matrices/expressions/blockmatrix.py:

Start line: 419, End line: 431

```python
def bc_block_plus_ident(expr):
    idents = [arg for arg in expr.args if arg.is_Identity]
    if not idents:
        return expr

    blocks = [arg for arg in expr.args if isinstance(arg, BlockMatrix)]
    if (blocks and all(b.structurally_equal(blocks[0]) for b in blocks)
               and blocks[0].is_structurally_symmetric):
        block_id = BlockDiagMatrix(*[Identity(k)
                                        for k in blocks[0].rowblocksizes])
        return MatAdd(block_id * len(idents), *blocks).doit()

    return expr
```
### 30 - sympy/matrices/expressions/blockmatrix.py:

Start line: 478, End line: 496

```python
def bc_transpose(expr):
    collapse = block_collapse(expr.arg)
    return collapse._eval_transpose()


def bc_inverse(expr):
    if isinstance(expr.arg, BlockDiagMatrix):
        return expr._eval_inverse()

    expr2 = blockinverse_1x1(expr)
    if expr != expr2:
        return expr2
    return blockinverse_2x2(Inverse(reblock_2x2(expr.arg)))

def blockinverse_1x1(expr):
    if isinstance(expr.arg, BlockMatrix) and expr.arg.blockshape == (1, 1):
        mat = Matrix([[expr.arg.blocks[0].inverse()]])
        return BlockMatrix(mat)
    return expr
```
### 31 - sympy/matrices/expressions/blockmatrix.py:

Start line: 399, End line: 417

```python
def bc_unpack(expr):
    if expr.blockshape == (1, 1):
        return expr.blocks[0, 0]
    return expr

def bc_matadd(expr):
    args = sift(expr.args, lambda M: isinstance(M, BlockMatrix))
    blocks = args[True]
    if not blocks:
        return expr

    nonblocks = args[False]
    block = blocks[0]
    for b in blocks[1:]:
        block = block._blockadd(b)
    if nonblocks:
        return MatAdd(*nonblocks) + block
    else:
        return block
```
### 32 - sympy/matrices/expressions/blockmatrix.py:

Start line: 509, End line: 527

```python
def deblock(B):
    """ Flatten a BlockMatrix of BlockMatrices """
    if not isinstance(B, BlockMatrix) or not B.blocks.has(BlockMatrix):
        return B
    wrap = lambda x: x if isinstance(x, BlockMatrix) else BlockMatrix([[x]])
    bb = B.blocks.applyfunc(wrap)  # everything is a block

    from sympy import Matrix
    try:
        MM = Matrix(0, sum(bb[0, i].blocks.shape[1] for i in range(bb.shape[1])), [])
        for row in range(0, bb.shape[0]):
            M = Matrix(bb[row, 0].blocks)
            for col in range(1, bb.shape[1]):
                M = M.row_join(bb[row, col].blocks)
            MM = MM.col_join(M)

        return BlockMatrix(MM)
    except ShapeError:
        return B
```
