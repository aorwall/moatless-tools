# sympy__sympy-21527

| **sympy/sympy** | `31d469a5335c81ec4a437e36a861945a6b43d916` |
| ---- | ---- |
| **No of patches** | 5 |
| **All found context length** | 3687 |
| **Any found context length** | 3687 |
| **Avg pos** | 1.0 |
| **Min pos** | 5 |
| **Max pos** | 5 |
| **Top file pos** | 2 |
| **Missing snippets** | 7 |
| **Missing patch files** | 4 |


## Expected patch

```diff
diff --git a/sympy/polys/constructor.py b/sympy/polys/constructor.py
--- a/sympy/polys/constructor.py
+++ b/sympy/polys/constructor.py
@@ -48,7 +48,7 @@ def _construct_simple(coeffs, opt):
                         float_numbers.append(x)
                     if y.is_Float:
                         float_numbers.append(y)
-            if is_algebraic(coeff):
+            elif is_algebraic(coeff):
                 if floats:
                     # there are both algebraics and reals -> EX
                     return False
diff --git a/sympy/polys/matrices/ddm.py b/sympy/polys/matrices/ddm.py
--- a/sympy/polys/matrices/ddm.py
+++ b/sympy/polys/matrices/ddm.py
@@ -284,7 +284,9 @@ def applyfunc(self, func, domain):
     def rref(a):
         """Reduced-row echelon form of a and list of pivots"""
         b = a.copy()
-        pivots = ddm_irref(b)
+        K = a.domain
+        partial_pivot = K.is_RealField or K.is_ComplexField
+        pivots = ddm_irref(b, _partial_pivot=partial_pivot)
         return b, pivots
 
     def nullspace(a):
diff --git a/sympy/polys/matrices/dense.py b/sympy/polys/matrices/dense.py
--- a/sympy/polys/matrices/dense.py
+++ b/sympy/polys/matrices/dense.py
@@ -85,7 +85,7 @@ def ddm_imatmul(a, b, c):
             ai[j] = sum(map(mul, bi, cTj), ai[j])
 
 
-def ddm_irref(a):
+def ddm_irref(a, _partial_pivot=False):
     """a  <--  rref(a)"""
     # a is (m x n)
     m = len(a)
@@ -97,6 +97,15 @@ def ddm_irref(a):
     pivots = []
 
     for j in range(n):
+        # Proper pivoting should be used for all domains for performance
+        # reasons but it is only strictly needed for RR and CC (and possibly
+        # other domains like RR(x)). This path is used by DDM.rref() if the
+        # domain is RR or CC. It uses partial (row) pivoting based on the
+        # absolute value of the pivot candidates.
+        if _partial_pivot:
+            ip = max(range(i, m), key=lambda ip: abs(a[ip][j]))
+            a[i], a[ip] = a[ip], a[i]
+
         # pivot
         aij = a[i][j]
 
diff --git a/sympy/polys/matrices/linsolve.py b/sympy/polys/matrices/linsolve.py
--- a/sympy/polys/matrices/linsolve.py
+++ b/sympy/polys/matrices/linsolve.py
@@ -73,6 +73,12 @@ def _linsolve(eqs, syms):
     Aaug = sympy_dict_to_dm(eqsdict, rhs, syms)
     K = Aaug.domain
 
+    # sdm_irref has issues with float matrices. This uses the ddm_rref()
+    # function. When sdm_rref() can handle float matrices reasonably this
+    # should be removed...
+    if K.is_RealField or K.is_ComplexField:
+        Aaug = Aaug.to_ddm().rref()[0].to_sdm()
+
     # Compute reduced-row echelon form (RREF)
     Arref, pivots, nzcols = sdm_irref(Aaug)
 
diff --git a/sympy/polys/matrices/sdm.py b/sympy/polys/matrices/sdm.py
--- a/sympy/polys/matrices/sdm.py
+++ b/sympy/polys/matrices/sdm.py
@@ -904,6 +904,8 @@ def sdm_irref(A):
             Ajnz = set(Aj)
             for k in Ajnz - Ainz:
                 Ai[k] = - Aij * Aj[k]
+            Ai.pop(j)
+            Ainz.remove(j)
             for k in Ajnz & Ainz:
                 Aik = Ai[k] - Aij * Aj[k]
                 if Aik:
@@ -938,6 +940,8 @@ def sdm_irref(A):
             for l in Ainz - Aknz:
                 Ak[l] = - Akj * Ai[l]
                 nonzero_columns[l].add(k)
+            Ak.pop(j)
+            Aknz.remove(j)
             for l in Ainz & Aknz:
                 Akl = Ak[l] - Akj * Ai[l]
                 if Akl:

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sympy/polys/constructor.py | 51 | 51 | - | - | -
| sympy/polys/matrices/ddm.py | 287 | 287 | - | - | -
| sympy/polys/matrices/dense.py | 88 | 88 | - | - | -
| sympy/polys/matrices/dense.py | 100 | 100 | - | - | -
| sympy/polys/matrices/linsolve.py | 76 | 76 | 5 | 2 | 3687
| sympy/polys/matrices/sdm.py | 907 | 907 | - | - | -
| sympy/polys/matrices/sdm.py | 941 | 941 | - | - | -


## Problem Statement

```
linsolve fails simple system of two equations
\`\`\`
import sympy
x,y = sympy.symbols('x, y')

sympy.linsolve([sympy.Eq(y, x), sympy.Eq(y, 0.0215 * x)], (x, y))
>> FiniteSet((0, 0))

sympy.linsolve([sympy.Eq(y, x), sympy.Eq(y, 0.0216 * x)], (x, y))
>> FiniteSet((-4.07992766242527e+17*y, 1.0*y))

sympy.linsolve([sympy.Eq(y, x), sympy.Eq(y, 0.0217 * x)], (x, y))
>> FiniteSet((0, 0))
\`\`\`

Any thoughts on why these don't all return the same solution? Thanks!

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 sympy/solvers/solveset.py | 2740 | 2850| 912 | 912 | 33460 | 
| 2 | 1 sympy/solvers/solveset.py | 2587 | 2739| 1513 | 2425 | 33460 | 
| 3 | 1 sympy/solvers/solveset.py | 3628 | 3685| 494 | 2919 | 33460 | 
| 4 | 1 sympy/solvers/solveset.py | 2565 | 2837| 188 | 3107 | 33460 | 
| **-> 5 <-** | **2 sympy/polys/matrices/linsolve.py** | 46 | 108| 580 | 3687 | 35221 | 
| 6 | 2 sympy/solvers/solveset.py | 1 | 66| 692 | 4379 | 35221 | 
| 7 | 2 sympy/solvers/solveset.py | 3473 | 3627| 1925 | 6304 | 35221 | 
| 8 | 3 sympy/solvers/recurr.py | 745 | 834| 660 | 6964 | 42100 | 
| 9 | 4 sympy/solvers/solvers.py | 2755 | 2865| 1166 | 8130 | 73791 | 
| 10 | **4 sympy/polys/matrices/linsolve.py** | 1 | 43| 383 | 8513 | 73791 | 
| 11 | 5 sympy/polys/solvers.py | 272 | 292| 161 | 8674 | 77241 | 
| 12 | 5 sympy/solvers/solvers.py | 380 | 832| 4494 | 13168 | 77241 | 
| 13 | 6 sympy/solvers/__init__.py | 1 | 72| 626 | 13794 | 77867 | 
| 14 | 7 sympy/solvers/pde.py | 744 | 802| 682 | 14476 | 87270 | 
| 15 | 7 sympy/solvers/solvers.py | 1890 | 1981| 822 | 15298 | 87270 | 
| 16 | 7 sympy/polys/solvers.py | 178 | 270| 784 | 16082 | 87270 | 
| 17 | 7 sympy/solvers/solveset.py | 1525 | 1562| 333 | 16415 | 87270 | 
| 18 | 7 sympy/solvers/solveset.py | 2299 | 2388| 593 | 17008 | 87270 | 


## Missing Patch Files

 * 1: sympy/polys/constructor.py
 * 2: sympy/polys/matrices/ddm.py
 * 3: sympy/polys/matrices/dense.py
 * 4: sympy/polys/matrices/linsolve.py
 * 5: sympy/polys/matrices/sdm.py

### Hint

```
It seems that in rref the pivot is not fully cancelled due to a rounding error so e.g. we have something like:
\`\`\`python
In [1]: M = Matrix([[1.0, 1.0], [3.1, 1.0]])

In [2]: M
Out[2]: 
⎡1.0  1.0⎤
⎢        ⎥
⎣3.1  1.0⎦
\`\`\`
Then one step of row reduction gives:
\`\`\`python
In [3]: M = Matrix([[1.0, 1.0], [1e-16, -2.1]])

In [4]: M
Out[4]: 
⎡  1.0    1.0 ⎤
⎢             ⎥
⎣1.0e-16  -2.1⎦
\`\`\`
With exact arithmetic the 1e-16 would have been 0 but the rounding error makes it not zero and then it throws off subsequent steps. I think that the solution is to make it exactly zero:
\`\`\`diff
diff --git a/sympy/polys/matrices/sdm.py b/sympy/polys/matrices/sdm.py
index cfa624185a..647eb6af3d 100644
--- a/sympy/polys/matrices/sdm.py
+++ b/sympy/polys/matrices/sdm.py
@@ -904,6 +904,8 @@ def sdm_irref(A):
             Ajnz = set(Aj)
             for k in Ajnz - Ainz:
                 Ai[k] = - Aij * Aj[k]
+            Ai.pop(j)
+            Ainz.remove(j)
             for k in Ajnz & Ainz:
                 Aik = Ai[k] - Aij * Aj[k]
                 if Aik:
\`\`\`
That gives:
\`\`\`python
In [1]: import sympy

In [2]: sympy.linsolve([sympy.Eq(y, x), sympy.Eq(y, 0.0215 * x)], (x, y))
Out[2]: {(0, 0)}

In [3]: sympy.linsolve([sympy.Eq(y, x), sympy.Eq(y, 0.0216 * x)], (x, y))
Out[3]: {(0, 0)}

In [4]: sympy.linsolve([sympy.Eq(y, x), sympy.Eq(y, 0.0217 * x)], (x, y))
Out[4]: {(0, 0)}
\`\`\`
@tyler-herzer-volumetric can you try out that diff?
Haven't been able to replicate the issue after making that change. Thanks a lot @oscarbenjamin!
This example still fails with the diff:
\`\`\`python
In [1]: linsolve([0.4*x + 0.3*y + 0.2, 0.4*x + 0.3*y + 0.3], [x, y])
Out[1]: {(1.35107988821115e+15, -1.8014398509482e+15)}
\`\`\`
In this case although the pivot is set to zero actually the matrix is singular but row reduction leads to something like:
\`\`\`python
In [3]: Matrix([[1, 1], [0, 1e-17]])
Out[3]: 
⎡1     1   ⎤
⎢          ⎥
⎣0  1.0e-17⎦
\`\`\`
That's a trickier case. It seems that numpy can pick up on it e.g.:
\`\`\`python
In [52]: M = np.array([[0.4, 0.3], [0.4, 0.3]])

In [53]: b = np.array([0.2, 0.3])

In [54]: np.linalg.solve(M, b)
---------------------------------------------------------------------------
LinAlgError: Singular matrix 
\`\`\`
A slight modification or rounding error leads to the same large result though:
\`\`\`python
In [55]: M = np.array([[0.4, 0.3], [0.4, 0.3-3e-17]])

In [56]: np.linalg.solve(M, b)
Out[56]: array([ 1.35107989e+15, -1.80143985e+15])
\`\`\`
I'm not sure it's possible to arrange the floating point calculation so that cases like this are picked up as being singular without introducing some kind of heuristic threshold for the determinant. This fails in numpy with even fairly simple examples:
\`\`\`python
In [83]: b
Out[83]: array([0.2, 0.3, 0.5])

In [84]: M
Out[84]: 
array([[0.1, 0.2, 0.3],
       [0.4, 0.5, 0.6],
       [0.7, 0.8, 0.9]])

In [85]: np.linalg.solve(M, b)
Out[85]: array([-4.50359963e+14,  9.00719925e+14, -4.50359963e+14])

In [86]: np.linalg.det(M)
Out[86]: 6.661338147750926e-18
\`\`\`
Maybe the not full-rank case for float matrices isn't so important since it can't be done reliably with floats. I guess that the diff shown above is good enough then since it fixes the calculation in the full rank case.
There is another problematic case. This should have a unique solution but a parametric solution is returned instead:
\`\`\`python
In [21]: eqs = [0.8*x + 0.8*z + 0.2, 0.9*x + 0.7*y + 0.2*z + 0.9, 0.7*x + 0.2*y + 0.2*z + 0.5]

In [22]: linsolve(eqs, [x, y, z])
Out[22]: {(-0.32258064516129⋅z - 0.548387096774194, 1.22033022161007e+16⋅z - 5.37526407137769e+15, 1.0⋅z)}
\`\`\`
That seems to be another bug in the sparse rref routine somehow:
\`\`\`python
In [34]: M = Matrix([
    ...: [0.8,   0, 0.8, -0.2],
    ...: [0.9, 0.7, 0.2, -0.9],
    ...: [0.7, 0.2, 0.2, -0.5]])

In [35]: from sympy.polys.matrices import DomainMatrix

In [36]: dM = DomainMatrix.from_Matrix(M)

In [37]: M.rref()
Out[37]: 
⎛⎡1  0  0  -0.690476190476191⎤           ⎞
⎜⎢                           ⎥           ⎟
⎜⎢0  1  0  -0.523809523809524⎥, (0, 1, 2)⎟
⎜⎢                           ⎥           ⎟
⎝⎣0  0  1  0.440476190476191 ⎦           ⎠

In [38]: dM.rref()[0].to_Matrix()
Out[38]: 
⎡1.0  5.55111512312578e-17    0.32258064516129      -0.548387096774194  ⎤
⎢                                                                       ⎥
⎢0.0          1.0           -1.22033022161007e+16  -5.37526407137769e+15⎥
⎢                                                                       ⎥
⎣0.0          0.0                    0.0                    0.0         ⎦

In [39]: dM.to_dense().rref()[0].to_Matrix()
Out[39]: 
⎡1.0  0.0  0.0  -0.69047619047619 ⎤
⎢                                 ⎥
⎢0.0  1.0  0.0  -0.523809523809524⎥
⎢                                 ⎥
⎣0.0  0.0  1.0   0.44047619047619 ⎦
\`\`\`
The last one was a similar problem to do with cancelling above the pivot:
\`\`\`diff
diff --git a/sympy/polys/matrices/sdm.py b/sympy/polys/matrices/sdm.py
index cfa624185a..7c4ad43660 100644
--- a/sympy/polys/matrices/sdm.py
+++ b/sympy/polys/matrices/sdm.py
@@ -904,6 +904,8 @@ def sdm_irref(A):
             Ajnz = set(Aj)
             for k in Ajnz - Ainz:
                 Ai[k] = - Aij * Aj[k]
+            Ai.pop(j)
+            Ainz.remove(j)
             for k in Ajnz & Ainz:
                 Aik = Ai[k] - Aij * Aj[k]
                 if Aik:
@@ -938,6 +940,8 @@ def sdm_irref(A):
             for l in Ainz - Aknz:
                 Ak[l] = - Akj * Ai[l]
                 nonzero_columns[l].add(k)
+            Ak.pop(j)
+            Aknz.remove(j)
             for l in Ainz & Aknz:
                 Akl = Ak[l] - Akj * Ai[l]
                 if Akl:
diff --git a/sympy/polys/matrices/tests/test_linsolve.py b/sympy/polys/matrices/tests/test_linsolve.py
index eda4cdbdf3..6b79842fa7 100644
--- a/sympy/polys/matrices/tests/test_linsolve.py
+++ b/sympy/polys/matrices/tests/test_linsolve.py
@@ -7,7 +7,7 @@
 from sympy.testing.pytest import raises
 
 from sympy import S, Eq, I
-from sympy.abc import x, y
+from sympy.abc import x, y, z
 
 from sympy.polys.matrices.linsolve import _linsolve
 from sympy.polys.solvers import PolyNonlinearError
@@ -23,6 +23,14 @@ def test__linsolve():
     raises(PolyNonlinearError, lambda: _linsolve([x*(1 + x)], [x]))
 
 
+def test__linsolve_float():
+    assert _linsolve([Eq(y, x), Eq(y, 0.0216 * x)], (x, y)) == {x:0, y:0}
+
+    eqs = [0.8*x + 0.8*z + 0.2, 0.9*x + 0.7*y + 0.2*z + 0.9, 0.7*x + 0.2*y + 0.2*z + 0.5]
+    sol = {x:-0.69047619047619047, y:-0.52380952380952395, z:0.44047619047619047}
+    assert _linsolve(eqs, [x,y,z]) == sol
+
+
 def test__linsolve_deprecated():
     assert _linsolve([Eq(x**2, x**2+y)], [x, y]) == {x:x, y:S.Zero}
     assert _linsolve([(x+y)**2-x**2], [x]) == {x:-y/2}
\`\`\`
Another problem has emerged though:
\`\`\`python
In [1]: eqs = [0.9*x + 0.3*y + 0.4*z + 0.6, 0.6*x + 0.9*y + 0.1*z + 0.7, 0.4*x + 0.6*y + 0.9*z + 0.5]

In [2]: linsolve(eqs, [x, y, z])
Out[2]: {(-0.5, -0.4375, -0.0400000000000001)}

In [3]: solve(eqs, [x, y, z])
Out[3]: {x: -0.502857142857143, y: -0.438095238095238, z: -0.04}
\`\`\`
> Another problem has emerged though:
> 
> \`\`\`python
> In [1]: eqs = [0.9*x + 0.3*y + 0.4*z + 0.6, 0.6*x + 0.9*y + 0.1*z + 0.7, 0.4*x + 0.6*y + 0.9*z + 0.5]
> \`\`\`

In this case the problem is that something like `1e-17` is chosen (incorrectly) as a pivot. It doesn't look easy to resolve this because the `sdm_rref` routine doesn't have an easy way of incorporating pivoting and is really designed for exact domains.

Probably float matrices should be handled by mpmath or at least a separate routine.
```

## Patch

```diff
diff --git a/sympy/polys/constructor.py b/sympy/polys/constructor.py
--- a/sympy/polys/constructor.py
+++ b/sympy/polys/constructor.py
@@ -48,7 +48,7 @@ def _construct_simple(coeffs, opt):
                         float_numbers.append(x)
                     if y.is_Float:
                         float_numbers.append(y)
-            if is_algebraic(coeff):
+            elif is_algebraic(coeff):
                 if floats:
                     # there are both algebraics and reals -> EX
                     return False
diff --git a/sympy/polys/matrices/ddm.py b/sympy/polys/matrices/ddm.py
--- a/sympy/polys/matrices/ddm.py
+++ b/sympy/polys/matrices/ddm.py
@@ -284,7 +284,9 @@ def applyfunc(self, func, domain):
     def rref(a):
         """Reduced-row echelon form of a and list of pivots"""
         b = a.copy()
-        pivots = ddm_irref(b)
+        K = a.domain
+        partial_pivot = K.is_RealField or K.is_ComplexField
+        pivots = ddm_irref(b, _partial_pivot=partial_pivot)
         return b, pivots
 
     def nullspace(a):
diff --git a/sympy/polys/matrices/dense.py b/sympy/polys/matrices/dense.py
--- a/sympy/polys/matrices/dense.py
+++ b/sympy/polys/matrices/dense.py
@@ -85,7 +85,7 @@ def ddm_imatmul(a, b, c):
             ai[j] = sum(map(mul, bi, cTj), ai[j])
 
 
-def ddm_irref(a):
+def ddm_irref(a, _partial_pivot=False):
     """a  <--  rref(a)"""
     # a is (m x n)
     m = len(a)
@@ -97,6 +97,15 @@ def ddm_irref(a):
     pivots = []
 
     for j in range(n):
+        # Proper pivoting should be used for all domains for performance
+        # reasons but it is only strictly needed for RR and CC (and possibly
+        # other domains like RR(x)). This path is used by DDM.rref() if the
+        # domain is RR or CC. It uses partial (row) pivoting based on the
+        # absolute value of the pivot candidates.
+        if _partial_pivot:
+            ip = max(range(i, m), key=lambda ip: abs(a[ip][j]))
+            a[i], a[ip] = a[ip], a[i]
+
         # pivot
         aij = a[i][j]
 
diff --git a/sympy/polys/matrices/linsolve.py b/sympy/polys/matrices/linsolve.py
--- a/sympy/polys/matrices/linsolve.py
+++ b/sympy/polys/matrices/linsolve.py
@@ -73,6 +73,12 @@ def _linsolve(eqs, syms):
     Aaug = sympy_dict_to_dm(eqsdict, rhs, syms)
     K = Aaug.domain
 
+    # sdm_irref has issues with float matrices. This uses the ddm_rref()
+    # function. When sdm_rref() can handle float matrices reasonably this
+    # should be removed...
+    if K.is_RealField or K.is_ComplexField:
+        Aaug = Aaug.to_ddm().rref()[0].to_sdm()
+
     # Compute reduced-row echelon form (RREF)
     Arref, pivots, nzcols = sdm_irref(Aaug)
 
diff --git a/sympy/polys/matrices/sdm.py b/sympy/polys/matrices/sdm.py
--- a/sympy/polys/matrices/sdm.py
+++ b/sympy/polys/matrices/sdm.py
@@ -904,6 +904,8 @@ def sdm_irref(A):
             Ajnz = set(Aj)
             for k in Ajnz - Ainz:
                 Ai[k] = - Aij * Aj[k]
+            Ai.pop(j)
+            Ainz.remove(j)
             for k in Ajnz & Ainz:
                 Aik = Ai[k] - Aij * Aj[k]
                 if Aik:
@@ -938,6 +940,8 @@ def sdm_irref(A):
             for l in Ainz - Aknz:
                 Ak[l] = - Akj * Ai[l]
                 nonzero_columns[l].add(k)
+            Ak.pop(j)
+            Aknz.remove(j)
             for l in Ainz & Aknz:
                 Akl = Ak[l] - Akj * Ai[l]
                 if Akl:

```

## Test Patch

```diff
diff --git a/sympy/polys/matrices/tests/test_linsolve.py b/sympy/polys/matrices/tests/test_linsolve.py
--- a/sympy/polys/matrices/tests/test_linsolve.py
+++ b/sympy/polys/matrices/tests/test_linsolve.py
@@ -7,7 +7,7 @@
 from sympy.testing.pytest import raises
 
 from sympy import S, Eq, I
-from sympy.abc import x, y
+from sympy.abc import x, y, z
 
 from sympy.polys.matrices.linsolve import _linsolve
 from sympy.polys.solvers import PolyNonlinearError
@@ -23,6 +23,83 @@ def test__linsolve():
     raises(PolyNonlinearError, lambda: _linsolve([x*(1 + x)], [x]))
 
 
+def test__linsolve_float():
+
+    # This should give the exact answer:
+    eqs = [
+        y - x,
+        y - 0.0216 * x
+    ]
+    sol = {x:0.0, y:0.0}
+    assert _linsolve(eqs, (x, y)) == sol
+
+    # Other cases should be close to eps
+
+    def all_close(sol1, sol2, eps=1e-15):
+        close = lambda a, b: abs(a - b) < eps
+        assert sol1.keys() == sol2.keys()
+        return all(close(sol1[s], sol2[s]) for s in sol1)
+
+    eqs = [
+        0.8*x +         0.8*z + 0.2,
+        0.9*x + 0.7*y + 0.2*z + 0.9,
+        0.7*x + 0.2*y + 0.2*z + 0.5
+    ]
+    sol_exact = {x:-29/42, y:-11/21, z:37/84}
+    sol_linsolve = _linsolve(eqs, [x,y,z])
+    assert all_close(sol_exact, sol_linsolve)
+
+    eqs = [
+        0.9*x + 0.3*y + 0.4*z + 0.6,
+        0.6*x + 0.9*y + 0.1*z + 0.7,
+        0.4*x + 0.6*y + 0.9*z + 0.5
+    ]
+    sol_exact = {x:-88/175, y:-46/105, z:-1/25}
+    sol_linsolve = _linsolve(eqs, [x,y,z])
+    assert all_close(sol_exact, sol_linsolve)
+
+    eqs = [
+        0.4*x + 0.3*y + 0.6*z + 0.7,
+        0.4*x + 0.3*y + 0.9*z + 0.9,
+        0.7*x + 0.9*y,
+    ]
+    sol_exact = {x:-9/5, y:7/5, z:-2/3}
+    sol_linsolve = _linsolve(eqs, [x,y,z])
+    assert all_close(sol_exact, sol_linsolve)
+
+    eqs = [
+        x*(0.7 + 0.6*I) + y*(0.4 + 0.7*I) + z*(0.9 + 0.1*I) + 0.5,
+        0.2*I*x + 0.2*I*y + z*(0.9 + 0.2*I) + 0.1,
+        x*(0.9 + 0.7*I) + y*(0.9 + 0.7*I) + z*(0.9 + 0.4*I) + 0.4,
+    ]
+    sol_exact = {
+        x:-6157/7995 - 411/5330*I,
+        y:8519/15990 + 1784/7995*I,
+        z:-34/533 + 107/1599*I,
+    }
+    sol_linsolve = _linsolve(eqs, [x,y,z])
+    assert all_close(sol_exact, sol_linsolve)
+
+    # XXX: This system for x and y over RR(z) is problematic.
+    #
+    # eqs = [
+    #     x*(0.2*z + 0.9) + y*(0.5*z + 0.8) + 0.6,
+    #     0.1*x*z + y*(0.1*z + 0.6) + 0.9,
+    # ]
+    #
+    # linsolve(eqs, [x, y])
+    # The solution for x comes out as
+    #
+    #       -3.9e-5*z**2 - 3.6e-5*z - 8.67361737988404e-20
+    #  x =  ----------------------------------------------
+    #           3.0e-6*z**3 - 1.3e-5*z**2 - 5.4e-5*z
+    #
+    # The 8e-20 in the numerator should be zero which would allow z to cancel
+    # from top and bottom. It should be possible to avoid this somehow because
+    # the inverse of the matrix only has a quadratic factor (the determinant)
+    # in the denominator.
+
+
 def test__linsolve_deprecated():
     assert _linsolve([Eq(x**2, x**2+y)], [x, y]) == {x:x, y:S.Zero}
     assert _linsolve([(x+y)**2-x**2], [x]) == {x:-y/2}
diff --git a/sympy/polys/tests/test_constructor.py b/sympy/polys/tests/test_constructor.py
--- a/sympy/polys/tests/test_constructor.py
+++ b/sympy/polys/tests/test_constructor.py
@@ -27,6 +27,9 @@ def test_construct_domain():
     assert isinstance(result[0], ComplexField)
     assert result[1] == [CC(3.14), CC(1.0j), CC(0.5)]
 
+    assert construct_domain([1.0+I]) == (CC, [CC(1.0, 1.0)])
+    assert construct_domain([2.0+3.0*I]) == (CC, [CC(2.0, 3.0)])
+
     assert construct_domain([1, I]) == (ZZ_I, [ZZ_I(1, 0), ZZ_I(0, 1)])
     assert construct_domain([1, I/2]) == (QQ_I, [QQ_I(1, 0), QQ_I(0, S.Half)])
 
diff --git a/sympy/polys/tests/test_polytools.py b/sympy/polys/tests/test_polytools.py
--- a/sympy/polys/tests/test_polytools.py
+++ b/sympy/polys/tests/test_polytools.py
@@ -51,6 +51,7 @@
 from sympy.polys.fields import field
 from sympy.polys.domains import FF, ZZ, QQ, ZZ_I, QQ_I, RR, EX
 from sympy.polys.domains.realfield import RealField
+from sympy.polys.domains.complexfield import ComplexField
 from sympy.polys.orderings import lex, grlex, grevlex
 
 from sympy import (
@@ -387,6 +388,7 @@ def test_Poly__new__():
              modulus=65537, symmetric=False)
 
     assert isinstance(Poly(x**2 + x + 1.0).get_domain(), RealField)
+    assert isinstance(Poly(x**2 + x + I + 1.0).get_domain(), ComplexField)
 
 
 def test_Poly__args():

```


## Code snippets

### 1 - sympy/solvers/solveset.py:

Start line: 2740, End line: 2850

```python
def linsolve(system, *symbols):
    if not system:
        return S.EmptySet

    # If second argument is an iterable
    if symbols and hasattr(symbols[0], '__iter__'):
        symbols = symbols[0]
    sym_gen = isinstance(symbols, GeneratorType)

    b = None  # if we don't get b the input was bad
    syms_needed_msg = None

    # unpack system

    if hasattr(system, '__iter__'):

        # 1). (A, b)
        if len(system) == 2 and isinstance(system[0], MatrixBase):
            A, b = system

        # 2). (eq1, eq2, ...)
        if not isinstance(system[0], MatrixBase):
            if sym_gen or not symbols:
                raise ValueError(filldedent('''
                    When passing a system of equations, the explicit
                    symbols for which a solution is being sought must
                    be given as a sequence, too.
                '''))

            #
            # Pass to the sparse solver implemented in polys. It is important
            # that we do not attempt to convert the equations to a matrix
            # because that would be very inefficient for large sparse systems
            # of equations.
            #
            eqs = system
            eqs = [sympify(eq) for eq in eqs]
            try:
                sol = _linsolve(eqs, symbols)
            except PolyNonlinearError as exc:
                # e.g. cos(x) contains an element of the set of generators
                raise NonlinearError(str(exc))

            if sol is None:
                return S.EmptySet

            sol = FiniteSet(Tuple(*(sol.get(sym, sym) for sym in symbols)))
            return sol

    elif isinstance(system, MatrixBase) and not (
            symbols and not isinstance(symbols, GeneratorType) and
            isinstance(symbols[0], MatrixBase)):
        # 3). A augmented with b
        A, b = system[:, :-1], system[:, -1:]

    if b is None:
        raise ValueError("Invalid arguments")

    syms_needed_msg  = syms_needed_msg or 'columns of A'

    if sym_gen:
        symbols = [next(symbols) for i in range(A.cols)]
        if any(set(symbols) & (A.free_symbols | b.free_symbols)):
            raise ValueError(filldedent('''
                At least one of the symbols provided
                already appears in the system to be solved.
                One way to avoid this is to use Dummy symbols in
                the generator, e.g. numbered_symbols('%s', cls=Dummy)
            ''' % symbols[0].name.rstrip('1234567890')))

    if not symbols:
        symbols = [Dummy() for _ in range(A.cols)]
        name = _uniquely_named_symbol('tau', (A, b),
            compare=lambda i: str(i).rstrip('1234567890')).name
        gen  = numbered_symbols(name)
    else:
        gen = None

    # This is just a wrapper for solve_lin_sys
    eqs = []
    rows = A.tolist()
    for rowi, bi in zip(rows, b):
        terms = [elem * sym for elem, sym in zip(rowi, symbols) if elem]
        terms.append(-bi)
        eqs.append(Add(*terms))

    eqs, ring = sympy_eqs_to_ring(eqs, symbols)
    sol = solve_lin_sys(eqs, ring, _raw=False)
    if sol is None:
        return S.EmptySet
    #sol = {sym:val for sym, val in sol.items() if sym != val}
    sol = FiniteSet(Tuple(*(sol.get(sym, sym) for sym in symbols)))

    if gen is not None:
        solsym = sol.free_symbols
        rep = {sym: next(gen) for sym in symbols if sym in solsym}
        sol = sol.subs(rep)

    return sol


##############################################################################
# ------------------------------nonlinsolve ---------------------------------#
##############################################################################


def _return_conditionset(eqs, symbols):
    # return conditionset
    eqs = (Eq(lhs, 0) for lhs in eqs)
    condition_set = ConditionSet(
        Tuple(*symbols), And(*eqs), S.Complexes**len(symbols))
    return condition_set
```
### 2 - sympy/solvers/solveset.py:

Start line: 2587, End line: 2739

```python
def linsolve(system, *symbols):
    r"""
    Solve system of N linear equations with M variables; both
    underdetermined and overdetermined systems are supported.
    The possible number of solutions is zero, one or infinite.
    Zero solutions throws a ValueError, whereas infinite
    solutions are represented parametrically in terms of the given
    symbols. For unique solution a FiniteSet of ordered tuples
    is returned.

    All Standard input formats are supported:
    For the given set of Equations, the respective input types
    are given below:

    .. math:: 3x + 2y -   z = 1
    .. math:: 2x - 2y + 4z = -2
    .. math:: 2x -   y + 2z = 0

    * Augmented Matrix Form, `system` given below:

    ::

              [3   2  -1  1]
     system = [2  -2   4 -2]
              [2  -1   2  0]

    * List Of Equations Form

    `system  =  [3x + 2y - z - 1, 2x - 2y + 4z + 2, 2x - y + 2z]`

    * Input A & b Matrix Form (from Ax = b) are given as below:

    ::

         [3   2  -1 ]         [  1 ]
     A = [2  -2   4 ]    b =  [ -2 ]
         [2  -1   2 ]         [  0 ]

    `system = (A, b)`

    Symbols can always be passed but are actually only needed
    when 1) a system of equations is being passed and 2) the
    system is passed as an underdetermined matrix and one wants
    to control the name of the free variables in the result.
    An error is raised if no symbols are used for case 1, but if
    no symbols are provided for case 2, internally generated symbols
    will be provided. When providing symbols for case 2, there should
    be at least as many symbols are there are columns in matrix A.

    The algorithm used here is Gauss-Jordan elimination, which
    results, after elimination, in a row echelon form matrix.

    Returns
    =======

    A FiniteSet containing an ordered tuple of values for the
    unknowns for which the `system` has a solution. (Wrapping
    the tuple in FiniteSet is used to maintain a consistent
    output format throughout solveset.)

    Returns EmptySet, if the linear system is inconsistent.

    Raises
    ======

    ValueError
        The input is not valid.
        The symbols are not given.

    Examples
    ========

    >>> from sympy import Matrix, linsolve, symbols
    >>> x, y, z = symbols("x, y, z")
    >>> A = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
    >>> b = Matrix([3, 6, 9])
    >>> A
    Matrix([
    [1, 2,  3],
    [4, 5,  6],
    [7, 8, 10]])
    >>> b
    Matrix([
    [3],
    [6],
    [9]])
    >>> linsolve((A, b), [x, y, z])
    FiniteSet((-1, 2, 0))

    * Parametric Solution: In case the system is underdetermined, the
      function will return a parametric solution in terms of the given
      symbols. Those that are free will be returned unchanged. e.g. in
      the system below, `z` is returned as the solution for variable z;
      it can take on any value.

    >>> A = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> b = Matrix([3, 6, 9])
    >>> linsolve((A, b), x, y, z)
    FiniteSet((z - 1, 2 - 2*z, z))

    If no symbols are given, internally generated symbols will be used.
    The `tau0` in the 3rd position indicates (as before) that the 3rd
    variable -- whatever it's named -- can take on any value:

    >>> linsolve((A, b))
    FiniteSet((tau0 - 1, 2 - 2*tau0, tau0))

    * List of Equations as input

    >>> Eqns = [3*x + 2*y - z - 1, 2*x - 2*y + 4*z + 2, - x + y/2 - z]
    >>> linsolve(Eqns, x, y, z)
    FiniteSet((1, -2, -2))

    * Augmented Matrix as input

    >>> aug = Matrix([[2, 1, 3, 1], [2, 6, 8, 3], [6, 8, 18, 5]])
    >>> aug
    Matrix([
    [2, 1,  3, 1],
    [2, 6,  8, 3],
    [6, 8, 18, 5]])
    >>> linsolve(aug, x, y, z)
    FiniteSet((3/10, 2/5, 0))

    * Solve for symbolic coefficients

    >>> a, b, c, d, e, f = symbols('a, b, c, d, e, f')
    >>> eqns = [a*x + b*y - c, d*x + e*y - f]
    >>> linsolve(eqns, x, y)
    FiniteSet(((-b*f + c*e)/(a*e - b*d), (a*f - c*d)/(a*e - b*d)))

    * A degenerate system returns solution as set of given
      symbols.

    >>> system = Matrix(([0, 0, 0], [0, 0, 0], [0, 0, 0]))
    >>> linsolve(system, x, y)
    FiniteSet((x, y))

    * For an empty system linsolve returns empty set

    >>> linsolve([], x)
    EmptySet

    * An error is raised if, after expansion, any nonlinearity
      is detected:

    >>> linsolve([x*(1/x - 1), (y - 1)**2 - y**2 + 1], x, y)
    FiniteSet((1, 1))
    >>> linsolve([x**2 - 1], x)
    Traceback (most recent call last):
    ...
    NonlinearError:
    nonlinear term encountered: x**2
    """
    # ... other code
```
### 3 - sympy/solvers/solveset.py:

Start line: 3628, End line: 3685

```python
def nonlinsolve(system, *symbols):
    from sympy.polys.polytools import is_zero_dimensional

    if not system:
        return S.EmptySet

    if not symbols:
        msg = ('Symbols must be given, for which solution of the '
               'system is to be found.')
        raise ValueError(filldedent(msg))

    if hasattr(symbols[0], '__iter__'):
        symbols = symbols[0]

    if not is_sequence(symbols) or not symbols:
        msg = ('Symbols must be given, for which solution of the '
               'system is to be found.')
        raise IndexError(filldedent(msg))

    system, symbols, swap = recast_to_symbols(system, symbols)
    if swap:
        soln = nonlinsolve(system, symbols)
        return FiniteSet(*[tuple(i.xreplace(swap) for i in s) for s in soln])

    if len(system) == 1 and len(symbols) == 1:
        return _solveset_work(system, symbols)

    # main code of def nonlinsolve() starts from here
    polys, polys_expr, nonpolys, denominators = _separate_poly_nonpoly(
        system, symbols)

    if len(symbols) == len(polys):
        # If all the equations in the system are poly
        if is_zero_dimensional(polys, symbols):
            # finite number of soln (Zero dimensional system)
            try:
                return _handle_zero_dimensional(polys, symbols, system)
            except NotImplementedError:
                # Right now it doesn't fail for any polynomial system of
                # equation. If `solve_poly_system` fails then `substitution`
                # method will handle it.
                result = substitution(
                    polys_expr, symbols, exclude=denominators)
                return result

        # positive dimensional system
        res = _handle_positive_dimensional(polys, symbols, denominators)
        if res is EmptySet and any(not p.domain.is_Exact for p in polys):
            raise NotImplementedError("Equation not in exact domain. Try converting to rational")
        else:
            return res

    else:
        # If all the equations are not polynomial.
        # Use `substitution` method for the system
        result = substitution(
            polys_expr + nonpolys, symbols, exclude=denominators)
        return result
```
### 4 - sympy/solvers/solveset.py:

Start line: 2565, End line: 2837

```python
def linear_eq_to_matrix(equations, *symbols):
    # ... other code
    if isinstance(equations, MatrixBase):
        equations = list(equations)
    elif isinstance(equations, (Expr, Eq)):
        equations = [equations]
    elif not is_sequence(equations):
        raise ValueError(filldedent('''
            Equation(s) must be given as a sequence, Expr,
            Eq or Matrix.
            '''))

    A, b = [], []
    for i, f in enumerate(equations):
        if isinstance(f, Equality):
            f = f.rewrite(Add, evaluate=False)
        coeff_list = linear_coeffs(f, *symbols)
        b.append(-coeff_list.pop())
        A.append(coeff_list)
    A, b = map(Matrix, (A, b))
    return A, b


def linsolve(system, *symbols):
    # ... other code
```
### 5 - sympy/polys/matrices/linsolve.py:

Start line: 46, End line: 108

```python
def _linsolve(eqs, syms):
    """Solve a linear system of equations.

    Examples
    ========

    Solve a linear system with a unique solution:

    >>> from sympy import symbols, Eq
    >>> from sympy.polys.matrices.linsolve import _linsolve
    >>> x, y = symbols('x, y')
    >>> eqs = [Eq(x + y, 1), Eq(x - y, 2)]
    >>> _linsolve(eqs, [x, y])
    {x: 3/2, y: -1/2}

    In the case of underdetermined systems the solution will be expressed in
    terms of the unknown symbols that are unconstrained:

    >>> _linsolve([Eq(x + y, 0)], [x, y])
    {x: -y, y: y}

    """
    # Number of unknowns (columns in the non-augmented matrix)
    nsyms = len(syms)

    # Convert to sparse augmented matrix (len(eqs) x (nsyms+1))
    eqsdict, rhs = _linear_eq_to_dict(eqs, syms)
    Aaug = sympy_dict_to_dm(eqsdict, rhs, syms)
    K = Aaug.domain

    # Compute reduced-row echelon form (RREF)
    Arref, pivots, nzcols = sdm_irref(Aaug)

    # No solution:
    if pivots and pivots[-1] == nsyms:
        return None

    # Particular solution for non-homogeneous system:
    P = sdm_particular_from_rref(Arref, nsyms+1, pivots)

    # Nullspace - general solution to homogeneous system
    # Note: using nsyms not nsyms+1 to ignore last column
    V, nonpivots = sdm_nullspace_from_rref(Arref, K.one, nsyms, pivots, nzcols)

    # Collect together terms from particular and nullspace:
    sol = defaultdict(list)
    for i, v in P.items():
        sol[syms[i]].append(K.to_sympy(v))
    for npi, Vi in zip(nonpivots, V):
        sym = syms[npi]
        for i, v in Vi.items():
            sol[syms[i]].append(sym * K.to_sympy(v))

    # Use a single call to Add for each term:
    sol = {s: Add(*terms) for s, terms in sol.items()}

    # Fill in the zeros:
    zero = S.Zero
    for s in set(syms) - set(sol):
        sol[s] = zero

    # All done!
    return sol
```
### 6 - sympy/solvers/solveset.py:

Start line: 1, End line: 66

```python
"""
This module contains functions to:

    - solve a single equation for a single variable, in any domain either real or complex.

    - solve a single transcendental equation for a single variable in any domain either real or complex.
      (currently supports solving in real domain only)

    - solve a system of linear equations with N variables and M equations.

    - solve a system of Non Linear Equations with N variables and M equations
"""
from sympy.core.sympify import sympify
from sympy.core import (S, Pow, Dummy, pi, Expr, Wild, Mul, Equality,
                        Add)
from sympy.core.containers import Tuple
from sympy.core.numbers import I, Number, Rational, oo
from sympy.core.function import (Lambda, expand_complex, AppliedUndef,
                                expand_log)
from sympy.core.mod import Mod
from sympy.core.numbers import igcd
from sympy.core.relational import Eq, Ne, Relational
from sympy.core.symbol import Symbol, _uniquely_named_symbol
from sympy.core.sympify import _sympify
from sympy.simplify.simplify import simplify, fraction, trigsimp
from sympy.simplify import powdenest, logcombine
from sympy.functions import (log, Abs, tan, cot, sin, cos, sec, csc, exp,
                             acos, asin, acsc, asec, arg,
                             piecewise_fold, Piecewise)
from sympy.functions.elementary.trigonometric import (TrigonometricFunction,
                                                      HyperbolicFunction)
from sympy.functions.elementary.miscellaneous import real_root
from sympy.logic.boolalg import And
from sympy.sets import (FiniteSet, EmptySet, imageset, Interval, Intersection,
                        Union, ConditionSet, ImageSet, Complement, Contains)
from sympy.sets.sets import Set, ProductSet
from sympy.matrices import Matrix, MatrixBase
from sympy.ntheory import totient
from sympy.ntheory.factor_ import divisors
from sympy.ntheory.residue_ntheory import discrete_log, nthroot_mod
from sympy.polys import (roots, Poly, degree, together, PolynomialError,
                         RootOf, factor, lcm, gcd)
from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.polytools import invert
from sympy.polys.solvers import (sympy_eqs_to_ring, solve_lin_sys,
    PolyNonlinearError)
from sympy.polys.matrices.linsolve import _linsolve
from sympy.solvers.solvers import (checksol, denoms, unrad,
    _simple_dens, recast_to_symbols)
from sympy.solvers.polysys import solve_poly_system
from sympy.solvers.inequalities import solve_univariate_inequality
from sympy.utilities import filldedent
from sympy.utilities.iterables import numbered_symbols, has_dups
from sympy.calculus.util import periodicity, continuous_domain
from sympy.core.compatibility import ordered, default_sort_key, is_sequence

from types import GeneratorType
from collections import defaultdict


class NonlinearError(ValueError):
    """Raised when unexpectedly encountering nonlinear equations"""
    pass


_rc = Dummy("R", real=True), Dummy("C", complex=True)
```
### 7 - sympy/solvers/solveset.py:

Start line: 3473, End line: 3627

```python
def nonlinsolve(system, *symbols):
    r"""
    Solve system of N nonlinear equations with M variables, which means both
    under and overdetermined systems are supported. Positive dimensional
    system is also supported (A system with infinitely many solutions is said
    to be positive-dimensional). In Positive dimensional system solution will
    be dependent on at least one symbol. Returns both real solution
    and complex solution(If system have). The possible number of solutions
    is zero, one or infinite.

    Parameters
    ==========

    system : list of equations
        The target system of equations
    symbols : list of Symbols
        symbols should be given as a sequence eg. list

    Returns
    =======

    A FiniteSet of ordered tuple of values of `symbols` for which the `system`
    has solution. Order of values in the tuple is same as symbols present in
    the parameter `symbols`.

    Please note that general FiniteSet is unordered, the solution returned
    here is not simply a FiniteSet of solutions, rather it is a FiniteSet of
    ordered tuple, i.e. the first & only argument to FiniteSet is a tuple of
    solutions, which is ordered, & hence the returned solution is ordered.

    Also note that solution could also have been returned as an ordered tuple,
    FiniteSet is just a wrapper `{}` around the tuple. It has no other
    significance except for the fact it is just used to maintain a consistent
    output format throughout the solveset.

    For the given set of Equations, the respective input types
    are given below:

    .. math:: x*y - 1 = 0
    .. math:: 4*x**2 + y**2 - 5 = 0

    `system  = [x*y - 1, 4*x**2 + y**2 - 5]`
    `symbols = [x, y]`

    Raises
    ======

    ValueError
        The input is not valid.
        The symbols are not given.
    AttributeError
        The input symbols are not `Symbol` type.

    Examples
    ========

    >>> from sympy.core.symbol import symbols
    >>> from sympy.solvers.solveset import nonlinsolve
    >>> x, y, z = symbols('x, y, z', real=True)
    >>> nonlinsolve([x*y - 1, 4*x**2 + y**2 - 5], [x, y])
    FiniteSet((-1, -1), (-1/2, -2), (1/2, 2), (1, 1))

    1. Positive dimensional system and complements:

    >>> from sympy import pprint
    >>> from sympy.polys.polytools import is_zero_dimensional
    >>> a, b, c, d = symbols('a, b, c, d', extended_real=True)
    >>> eq1 =  a + b + c + d
    >>> eq2 = a*b + b*c + c*d + d*a
    >>> eq3 = a*b*c + b*c*d + c*d*a + d*a*b
    >>> eq4 = a*b*c*d - 1
    >>> system = [eq1, eq2, eq3, eq4]
    >>> is_zero_dimensional(system)
    False
    >>> pprint(nonlinsolve(system, [a, b, c, d]), use_unicode=False)
      -1       1               1      -1
    {(---, -d, -, {d} \ {0}), (-, -d, ---, {d} \ {0})}
       d       d               d       d
    >>> nonlinsolve([(x+y)**2 - 4, x + y - 2], [x, y])
    FiniteSet((2 - y, y))

    2. If some of the equations are non-polynomial then `nonlinsolve`
    will call the `substitution` function and return real and complex solutions,
    if present.

    >>> from sympy import exp, sin
    >>> nonlinsolve([exp(x) - sin(y), y**2 - 4], [x, y])
    FiniteSet((ImageSet(Lambda(_n, 2*_n*I*pi + log(sin(2))), Integers), 2),
            (ImageSet(Lambda(_n, I*(2*_n*pi + pi) + log(sin(2))), Integers), -2))


    3. If system is non-linear polynomial and zero-dimensional then it
    returns both solution (real and complex solutions, if present) using
    `solve_poly_system`:

    >>> from sympy import sqrt
    >>> nonlinsolve([x**2 - 2*y**2 -2, x*y - 2], [x, y])
    FiniteSet((-2, -1), (2, 1), (-sqrt(2)*I, sqrt(2)*I), (sqrt(2)*I, -sqrt(2)*I))

    4. `nonlinsolve` can solve some linear (zero or positive dimensional)
    system (because it uses the `groebner` function to get the
    groebner basis and then uses the `substitution` function basis as the
    new `system`). But it is not recommended to solve linear system using
    `nonlinsolve`, because `linsolve` is better for general linear systems.

    >>> nonlinsolve([x + 2*y -z - 3, x - y - 4*z + 9 , y + z - 4], [x, y, z])
    FiniteSet((3*z - 5, 4 - z, z))

    5. System having polynomial equations and only real solution is
    solved using `solve_poly_system`:

    >>> e1 = sqrt(x**2 + y**2) - 10
    >>> e2 = sqrt(y**2 + (-x + 10)**2) - 3
    >>> nonlinsolve((e1, e2), (x, y))
    FiniteSet((191/20, -3*sqrt(391)/20), (191/20, 3*sqrt(391)/20))
    >>> nonlinsolve([x**2 + 2/y - 2, x + y - 3], [x, y])
    FiniteSet((1, 2), (1 - sqrt(5), 2 + sqrt(5)), (1 + sqrt(5), 2 - sqrt(5)))
    >>> nonlinsolve([x**2 + 2/y - 2, x + y - 3], [y, x])
    FiniteSet((2, 1), (2 - sqrt(5), 1 + sqrt(5)), (2 + sqrt(5), 1 - sqrt(5)))

    6. It is better to use symbols instead of Trigonometric Function or
    Function (e.g. replace `sin(x)` with symbol, replace `f(x)` with symbol
    and so on. Get soln from `nonlinsolve` and then using `solveset` get
    the value of `x`)

    How nonlinsolve is better than old solver `_solve_system` :
    ===========================================================

    1. A positive dimensional system solver : nonlinsolve can return
    solution for positive dimensional system. It finds the
    Groebner Basis of the positive dimensional system(calling it as
    basis) then we can start solving equation(having least number of
    variable first in the basis) using solveset and substituting that
    solved solutions into other equation(of basis) to get solution in
    terms of minimum variables. Here the important thing is how we
    are substituting the known values and in which equations.

    2. Real and Complex both solutions : nonlinsolve returns both real
    and complex solution. If all the equations in the system are polynomial
    then using `solve_poly_system` both real and complex solution is returned.
    If all the equations in the system are not polynomial equation then goes to
    `substitution` method with this polynomial and non polynomial equation(s),
    to solve for unsolved variables. Here to solve for particular variable
    solveset_real and solveset_complex is used. For both real and complex
    solution function `_solve_using_know_values` is used inside `substitution`
    function.(`substitution` function will be called when there is any non
    polynomial equation(s) is present). When solution is valid then add its
    general solution in the final result.

    3. Complement and Intersection will be added if any : nonlinsolve maintains
    dict for complements and Intersections. If solveset find complements or/and
    Intersection with any Interval or set during the execution of
    `substitution` function ,then complement or/and Intersection for that
    variable is added before returning final solution.

    """
    # ... other code
```
### 8 - sympy/solvers/recurr.py:

Start line: 745, End line: 834

```python
def rsolve(f, y, init=None):
    # ... other code
    for k in h_part:
        h_part[k] = Add(*h_part[k])
    h_part.default_factory = lambda: 0
    i_part = Add(*i_part)

    for k, coeff in h_part.items():
        h_part[k] = simplify(coeff)

    common = S.One

    if not i_part.is_zero and not i_part.is_hypergeometric(n) and \
       not (i_part.is_Add and all(map(lambda x: x.is_hypergeometric(n), i_part.expand().args))):
        raise ValueError("The independent term should be a sum of hypergeometric functions, got '%s'" % i_part)

    for coeff in h_part.values():
        if coeff.is_rational_function(n):
            if not coeff.is_polynomial(n):
                common = lcm(common, coeff.as_numer_denom()[1], n)
        else:
            raise ValueError(
                "Polynomial or rational function expected, got '%s'" % coeff)

    i_numer, i_denom = i_part.as_numer_denom()

    if i_denom.is_polynomial(n):
        common = lcm(common, i_denom, n)

    if common is not S.One:
        for k, coeff in h_part.items():
            numer, denom = coeff.as_numer_denom()
            h_part[k] = numer*quo(common, denom, n)

        i_part = i_numer*quo(common, i_denom, n)

    K_min = min(h_part.keys())

    if K_min < 0:
        K = abs(K_min)

        H_part = defaultdict(lambda: S.Zero)
        i_part = i_part.subs(n, n + K).expand()
        common = common.subs(n, n + K).expand()

        for k, coeff in h_part.items():
            H_part[k + K] = coeff.subs(n, n + K).expand()
    else:
        H_part = h_part

    K_max = max(H_part.keys())
    coeffs = [H_part[i] for i in range(K_max + 1)]

    result = rsolve_hyper(coeffs, -i_part, n, symbols=True)

    if result is None:
        return None

    solution, symbols = result

    if init == {} or init == []:
        init = None

    if symbols and init is not None:
        if isinstance(init, list):
            init = {i: init[i] for i in range(len(init))}

        equations = []

        for k, v in init.items():
            try:
                i = int(k)
            except TypeError:
                if k.is_Function and k.func == y.func:
                    i = int(k.args[0])
                else:
                    raise ValueError("Integer or term expected, got '%s'" % k)

            eq = solution.subs(n, i) - v
            if eq.has(S.NaN):
                eq = solution.limit(n, i) - v
            equations.append(eq)

        result = solve(equations, *symbols)

        if not result:
            return None
        else:
            solution = solution.subs(result)

    return solution
```
### 9 - sympy/solvers/solvers.py:

Start line: 2755, End line: 2865

```python
# TODO: option for calculating J numerically

@conserve_mpmath_dps
def nsolve(*args, dict=False, **kwargs):
    r"""
    Solve a nonlinear equation system numerically: ``nsolve(f, [args,] x0,
    modules=['mpmath'], **kwargs)``.

    Explanation
    ===========

    ``f`` is a vector function of symbolic expressions representing the system.
    *args* are the variables. If there is only one variable, this argument can
    be omitted. ``x0`` is a starting vector close to a solution.

    Use the modules keyword to specify which modules should be used to
    evaluate the function and the Jacobian matrix. Make sure to use a module
    that supports matrices. For more information on the syntax, please see the
    docstring of ``lambdify``.

    If the keyword arguments contain ``dict=True`` (default is False) ``nsolve``
    will return a list (perhaps empty) of solution mappings. This might be
    especially useful if you want to use ``nsolve`` as a fallback to solve since
    using the dict argument for both methods produces return values of
    consistent type structure. Please note: to keep this consistent with
    ``solve``, the solution will be returned in a list even though ``nsolve``
    (currently at least) only finds one solution at a time.

    Overdetermined systems are supported.

    Examples
    ========

    >>> from sympy import Symbol, nsolve
    >>> import mpmath
    >>> mpmath.mp.dps = 15
    >>> x1 = Symbol('x1')
    >>> x2 = Symbol('x2')
    >>> f1 = 3 * x1**2 - 2 * x2**2 - 1
    >>> f2 = x1**2 - 2 * x1 + x2**2 + 2 * x2 - 8
    >>> print(nsolve((f1, f2), (x1, x2), (-1, 1)))
    Matrix([[-1.19287309935246], [1.27844411169911]])

    For one-dimensional functions the syntax is simplified:

    >>> from sympy import sin, nsolve
    >>> from sympy.abc import x
    >>> nsolve(sin(x), x, 2)
    3.14159265358979
    >>> nsolve(sin(x), 2)
    3.14159265358979

    To solve with higher precision than the default, use the prec argument:

    >>> from sympy import cos
    >>> nsolve(cos(x) - x, 1)
    0.739085133215161
    >>> nsolve(cos(x) - x, 1, prec=50)
    0.73908513321516064165531208767387340401341175890076
    >>> cos(_)
    0.73908513321516064165531208767387340401341175890076

    To solve for complex roots of real functions, a nonreal initial point
    must be specified:

    >>> from sympy import I
    >>> nsolve(x**2 + 2, I)
    1.4142135623731*I

    ``mpmath.findroot`` is used and you can find their more extensive
    documentation, especially concerning keyword parameters and
    available solvers. Note, however, that functions which are very
    steep near the root, the verification of the solution may fail. In
    this case you should use the flag ``verify=False`` and
    independently verify the solution.

    >>> from sympy import cos, cosh
    >>> f = cos(x)*cosh(x) - 1
    >>> nsolve(f, 3.14*100)
    Traceback (most recent call last):
    ...
    ValueError: Could not find root within given tolerance. (1.39267e+230 > 2.1684e-19)
    >>> ans = nsolve(f, 3.14*100, verify=False); ans
    312.588469032184
    >>> f.subs(x, ans).n(2)
    2.1e+121
    >>> (f/f.diff(x)).subs(x, ans).n(2)
    7.4e-15

    One might safely skip the verification if bounds of the root are known
    and a bisection method is used:

    >>> bounds = lambda i: (3.14*i, 3.14*(i + 1))
    >>> nsolve(f, bounds(100), solver='bisect', verify=False)
    315.730061685774

    Alternatively, a function may be better behaved when the
    denominator is ignored. Since this is not always the case, however,
    the decision of what function to use is left to the discretion of
    the user.

    >>> eq = x**2/(1 - x)/(1 - 2*x)**2 - 100
    >>> nsolve(eq, 0.46)
    Traceback (most recent call last):
    ...
    ValueError: Could not find root within given tolerance. (10000 > 2.1684e-19)
    Try another starting point or tweak arguments.
    >>> nsolve(eq.as_numer_denom()[0], 0.46)
    0.46792545969349058

    """
    # ... other code
```
### 10 - sympy/polys/matrices/linsolve.py:

Start line: 1, End line: 43

```python
#
# sympy.polys.matrices.linsolve module
#
# This module defines the _linsolve function which is the internal workhorse
# used by linsolve. This computes the solution of a system of linear equations
# using the SDM sparse matrix implementation in sympy.polys.matrices.sdm. This
# is a replacement for solve_lin_sys in sympy.polys.solvers which is
# inefficient for large sparse systems due to the use of a PolyRing with many
# generators:
#
#     https://github.com/sympy/sympy/issues/20857
#
# The implementation of _linsolve here handles:
#
# - Extracting the coefficients from the Expr/Eq input equations.
# - Constructing a domain and converting the coefficients to
#   that domain.
# - Using the SDM.rref, SDM.nullspace etc methods to generate the full
#   solution working with arithmetic only in the domain of the coefficients.
#
# The routines here are particularly designed to be efficient for large sparse
# systems of linear equations although as well as dense systems. It is
# possible that for some small dense systems solve_lin_sys which uses the
# dense matrix implementation DDM will be more efficient. With smaller systems
# though the bulk of the time is spent just preprocessing the inputs and the
# relative time spent in rref is too small to be noticeable.
#

from collections import defaultdict

from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.singleton import S

from sympy.polys.constructor import construct_domain
from sympy.polys.solvers import PolyNonlinearError

from .sdm import (
    SDM,
    sdm_irref,
    sdm_particular_from_rref,
    sdm_nullspace_from_rref
)
```
