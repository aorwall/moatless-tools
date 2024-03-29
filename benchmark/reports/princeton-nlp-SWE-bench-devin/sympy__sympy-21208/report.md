# sympy__sympy-21208

| **sympy/sympy** | `f9badb21b01f4f52ce4d545d071086ee650cd282` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | 8368 |
| **Avg pos** | 23.0 |
| **Min pos** | 23 |
| **Max pos** | 23 |
| **Top file pos** | 3 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sympy/matrices/expressions/matexpr.py b/sympy/matrices/expressions/matexpr.py
--- a/sympy/matrices/expressions/matexpr.py
+++ b/sympy/matrices/expressions/matexpr.py
@@ -653,7 +653,7 @@ def _matrix_derivative(expr, x):
 
     from sympy.tensor.array.expressions.conv_array_to_matrix import convert_array_to_matrix
 
-    parts = [[convert_array_to_matrix(j).doit() for j in i] for i in parts]
+    parts = [[convert_array_to_matrix(j) for j in i] for i in parts]
 
     def _get_shape(elem):
         if isinstance(elem, MatrixExpr):
@@ -897,7 +897,7 @@ def build(self):
         data = [self._build(i) for i in self._lines]
         if self.higher != 1:
             data += [self._build(self.higher)]
-        data = [i.doit() for i in data]
+        data = [i for i in data]
         return data
 
     def matrix_form(self):

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sympy/matrices/expressions/matexpr.py | 656 | 656 | 23 | 3 | 8368
| sympy/matrices/expressions/matexpr.py | 900 | 900 | - | 3 | -


## Problem Statement

```
Results diverge when use `diff` on a matrix or its elemetns
create a one-element matrix A as below:
\`\`\`python
>>> from sympy import *
>>> t = symbols('t')
>>> x = Function('x')(t)
>>> dx = x.diff(t)
>>> A = Matrix([cos(x) + cos(x) * dx])
\`\`\`
when use `diff` on matrix A:
\`\`\`python
>>> (A.diff(x))[0,0]
-sin(x(t))
\`\`\`
when use `diff` on the single element of A: 
\`\`\`python
>>> A[0,0].diff(x)
-sin(x(t))*Derivative(x(t), t) - sin(x(t))
\`\`\`
but if use `applyfunc` method on A, the result is the same as above:
\`\`\`python
>>> A.applyfunc(lambda ij: ij.diff(x))[0,0]
-sin(x(t))*Derivative(x(t), t) - sin(x(t))
\`\`\`
is this a bug or supposed behavior of matrix calculus?

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 sympy/matrices/expressions/applyfunc.py | 104 | 124| 164 | 164 | 1351 | 
| 2 | 1 sympy/matrices/expressions/applyfunc.py | 126 | 192| 519 | 683 | 1351 | 
| 3 | 2 sympy/matrices/matrices.py | 438 | 469| 208 | 891 | 20579 | 
| 4 | **3 sympy/matrices/expressions/matexpr.py** | 727 | 754| 265 | 1156 | 28104 | 
| 5 | 4 sympy/solvers/ode/systems.py | 1087 | 1135| 354 | 1510 | 47030 | 
| 6 | **4 sympy/matrices/expressions/matexpr.py** | 669 | 692| 191 | 1701 | 47030 | 
| 7 | 5 sympy/tensor/array/expressions/arrayexpr_derivatives.py | 113 | 125| 128 | 1829 | 48558 | 
| 8 | 5 sympy/matrices/expressions/applyfunc.py | 73 | 102| 183 | 2012 | 48558 | 
| 9 | 5 sympy/matrices/expressions/applyfunc.py | 1 | 45| 330 | 2342 | 48558 | 
| 10 | 5 sympy/matrices/expressions/applyfunc.py | 47 | 71| 184 | 2526 | 48558 | 
| 11 | 5 sympy/tensor/array/expressions/arrayexpr_derivatives.py | 97 | 110| 117 | 2643 | 48558 | 
| 12 | 6 sympy/core/function.py | 1268 | 1368| 832 | 3475 | 76783 | 
| 13 | 7 sympy/calculus/finite_diff.py | 367 | 421| 492 | 3967 | 82182 | 
| 14 | 7 sympy/core/function.py | 1939 | 1947| 142 | 4109 | 82182 | 
| 15 | 7 sympy/core/function.py | 2438 | 2837| 571 | 4680 | 82182 | 
| 16 | 8 sympy/tensor/array/arrayop.py | 267 | 326| 570 | 5250 | 86679 | 
| 17 | 8 sympy/matrices/matrices.py | 501 | 556| 488 | 5738 | 86679 | 
| 18 | 8 sympy/solvers/ode/systems.py | 498 | 570| 548 | 6286 | 86679 | 
| 19 | 8 sympy/matrices/matrices.py | 1503 | 1596| 752 | 7038 | 86679 | 
| 20 | 9 sympy/matrices/common.py | 2005 | 2031| 187 | 7225 | 111344 | 
| 21 | 10 sympy/matrices/expressions/matmul.py | 186 | 209| 195 | 7420 | 114940 | 
| 22 | 11 sympy/functions/special/hyper.py | 527 | 611| 744 | 8164 | 125706 | 
| **-> 23 <-** | **11 sympy/matrices/expressions/matexpr.py** | 642 | 667| 204 | 8368 | 125706 | 
| 24 | **11 sympy/matrices/expressions/matexpr.py** | 695 | 725| 246 | 8614 | 125706 | 
| 25 | 12 sympy/diffgeom/diffgeom.py | 1429 | 1452| 218 | 8832 | 142303 | 
| 26 | 13 sympy/geometry/util.py | 538 | 604| 527 | 9359 | 147716 | 
| 27 | 13 sympy/diffgeom/diffgeom.py | 1 | 27| 209 | 9568 | 147716 | 
| 28 | 13 sympy/tensor/array/expressions/arrayexpr_derivatives.py | 79 | 94| 200 | 9768 | 147716 | 
| 29 | **13 sympy/matrices/expressions/matexpr.py** | 468 | 590| 1188 | 10956 | 147716 | 
| 30 | 13 sympy/solvers/ode/systems.py | 650 | 1007| 432 | 11388 | 147716 | 
| 31 | 13 sympy/core/function.py | 1053 | 1230| 1482 | 12870 | 147716 | 
| 32 | 14 sympy/tensor/array/__init__.py | 1 | 228| 2451 | 15321 | 150363 | 
| 33 | 15 sympy/holonomic/holonomic.py | 842 | 932| 744 | 16065 | 175018 | 
| 34 | 15 sympy/diffgeom/diffgeom.py | 973 | 1013| 446 | 16511 | 175018 | 
| 35 | 15 sympy/matrices/common.py | 2698 | 2711| 128 | 16639 | 175018 | 
| 36 | 15 sympy/core/function.py | 1232 | 1266| 267 | 16906 | 175018 | 
| 37 | 15 sympy/diffgeom/diffgeom.py | 1075 | 1123| 540 | 17446 | 175018 | 
| 38 | 15 sympy/diffgeom/diffgeom.py | 1467 | 1479| 120 | 17566 | 175018 | 
| 39 | 15 sympy/tensor/array/expressions/arrayexpr_derivatives.py | 1 | 20| 192 | 17758 | 175018 | 
| 40 | 15 sympy/tensor/array/expressions/arrayexpr_derivatives.py | 128 | 173| 392 | 18150 | 175018 | 
| 41 | 15 sympy/calculus/finite_diff.py | 424 | 473| 660 | 18810 | 175018 | 
| 42 | 15 sympy/diffgeom/diffgeom.py | 1016 | 1073| 395 | 19205 | 175018 | 
| 43 | 15 sympy/solvers/ode/systems.py | 573 | 629| 417 | 19622 | 175018 | 
| 44 | 15 sympy/core/function.py | 1837 | 1936| 1048 | 20670 | 175018 | 
| 45 | **15 sympy/matrices/expressions/matexpr.py** | 811 | 827| 177 | 20847 | 175018 | 
| 46 | 15 sympy/diffgeom/diffgeom.py | 1454 | 1465| 147 | 20994 | 175018 | 
| 47 | 16 sympy/functions/elementary/complexes.py | 847 | 903| 312 | 21306 | 185472 | 
| 48 | 16 sympy/calculus/finite_diff.py | 474 | 487| 180 | 21486 | 185472 | 
| 49 | **16 sympy/matrices/expressions/matexpr.py** | 150 | 268| 832 | 22318 | 185472 | 
| 50 | 17 sympy/series/limitseq.py | 18 | 63| 314 | 22632 | 187465 | 
| 51 | 18 sympy/tensor/array/expressions/array_expressions.py | 722 | 754| 193 | 22825 | 199264 | 


### Hint

```
`.diff()` is running this internally:

\`\`\`ipython
In [1]: import sympy as sm

In [2]: t = sm.symbols('t')

In [3]: x = sm.Function('x')(t)

In [4]: dx = x.diff(t)

In [19]: from sympy.tensor.array.array_derivatives import ArrayDerivative

In [26]: ArrayDerivative(sm.Matrix([sm.cos(x) + sm.cos(x)*dx]), x, evaluate=True)
Out[26]: Matrix([[-sin(x(t))]])

In [27]: ArrayDerivative(sm.cos(x) + sm.cos(x)*dx, x, evaluate=True)
Out[27]: -sin(x(t))*Derivative(x(t), t) - sin(x(t))
\`\`\`

I tried this in SymPy 1.0 at it works as expected. The code for `Matrix.diff()` is:

\`\`\`
Signature: A.diff(*args)
Source:   
    def diff(self, *args):
        """Calculate the derivative of each element in the matrix.

        Examples
        ========

        >>> from sympy.matrices import Matrix
        >>> from sympy.abc import x, y
        >>> M = Matrix([[x, y], [1, 0]])
        >>> M.diff(x)
        Matrix([
        [1, 0],
        [0, 0]])

        See Also
        ========

        integrate
        limit
        """
        return self._new(self.rows, self.cols,
                lambda i, j: self[i, j].diff(*args))
File:      ~/src/sympy/sympy/matrices/matrices.py
Type:      method

\`\`\`

So I think that ArrayDerivative change has introduced this bug.
@Upabjojr This seems like a bug tied to the introduction of tensor code into the diff() of Matrix. I'm guessing that is something you added.
I'll assign this issue to myself for now. Will look into it when I have some time.
A quick fix it to switch back to the simple .diff of each element. I couldn't make heads or tails of the ArrayDerivative code with a quick look.
Switching it back causes A.diff(A) to fail:

\`\`\`
________________________________________ sympy/matrices/tests/test_matrices.py:test_diff_by_matrix _________________________________________
Traceback (most recent call last):
  File "/home/moorepants/src/sympy/sympy/matrices/tests/test_matrices.py", line 1919, in test_diff_by_matrix
    assert A.diff(A) == Array([[[[1, 0], [0, 0]], [[0, 1], [0, 0]]], [[[0, 0], [1, 0]], [[0, 0], [0, 1]]]])
AssertionError

\`\`\`
The problem lies in the expressions submodule of the matrices. Matrix expressions are handled differently from actual expressions.

https://github.com/sympy/sympy/blob/f9badb21b01f4f52ce4d545d071086ee650cd282/sympy/core/function.py#L1456-L1462

Every derivative requested expression passes through this point. The old variable is replaced with a new dummy variable (`xi`). Due to this some of the properties of the old symbol are lost. `is_Functon` being one of them. In the `old_v` (`x(t)`) this property is `True` but for `v` it is False. That is **not** an issue here as the old variable is substituted back after the computation of derivative. All the expressions pass through this point. The problem starts when we apply `doit()` on such dummy value expressions. Instead of treating the dummy variable as a `Function`, it is treated as regular `sympy.core.symbol.Symbol`. 

The best way to avoid this is to not use `doit()` on such expressions until the derivative is calculated and the old variable is substituted back. It is therefore nowhere used in the core expression module. However, in the matexpr module it is used twice and this is where the results deviate. 

\`\`\`python
In [1]: from sympy import *
        from sympy import __version__ as ver
        t = symbols('t')
        x = Function('x')(t)
        ver
Out[1]: '1.8.dev'

In [2]: xi = Dummy("xi")
        new = x.xreplace({x: xi})
        print(Derivative(x, t).doit())
        print(Derivative(new, t).doit())
Out[2]: Derivative(x(t), t)
        0
\`\`\`

https://github.com/sympy/sympy/blob/f9badb21b01f4f52ce4d545d071086ee650cd282/sympy/matrices/expressions/matexpr.py#L656

and 

https://github.com/sympy/sympy/blob/f9badb21b01f4f52ce4d545d071086ee650cd282/sympy/matrices/expressions/matexpr.py#L900

These two `doit()` reduce the `-sin(_xi)*Derivative(_xi, t)` part of the derivative computed to `0` (treating `_xi` as a symbol).

I removed the `.doit()` part from both the lines and it worked. My diff-

\`\`\`diff
$ git diff
diff --git a/sympy/matrices/expressions/matexpr.py b/sympy/matrices/expressions/matexpr.py
index 6dc87b5754..e11946015c 100644
--- a/sympy/matrices/expressions/matexpr.py
+++ b/sympy/matrices/expressions/matexpr.py
@@ -653,7 +653,7 @@ def _matrix_derivative(expr, x):

     from sympy.tensor.array.expressions.conv_array_to_matrix import convert_array_to_matrix

-    parts = [[convert_array_to_matrix(j).doit() for j in i] for i in parts]
+    parts = [[convert_array_to_matrix(j) for j in i] for i in parts]

     def _get_shape(elem):
         if isinstance(elem, MatrixExpr):
@@ -897,7 +897,7 @@ def build(self):
         data = [self._build(i) for i in self._lines]
         if self.higher != 1:
             data += [self._build(self.higher)]
-        data = [i.doit() for i in data]
+        data = [i for i in data]
         return data

     def matrix_form(self):
\`\`\`

I also ran all the tests and all the tests passed. Just to check that I was not anything wrong or ignoring some important purpose of that `.doit()` part. I tried with some of my own examples using both `expr.diff(x)` and `matrix.diff(x)` and the results were the same for both every single time. If `doit()` has some important purpose then please let me know. Also if everything is fine then should I open a PR?
> Also if everything is fine then should I open a PR?

Of course! It's easier to discuss edits in PRs because you can put comments on the code there.
```

## Patch

```diff
diff --git a/sympy/matrices/expressions/matexpr.py b/sympy/matrices/expressions/matexpr.py
--- a/sympy/matrices/expressions/matexpr.py
+++ b/sympy/matrices/expressions/matexpr.py
@@ -653,7 +653,7 @@ def _matrix_derivative(expr, x):
 
     from sympy.tensor.array.expressions.conv_array_to_matrix import convert_array_to_matrix
 
-    parts = [[convert_array_to_matrix(j).doit() for j in i] for i in parts]
+    parts = [[convert_array_to_matrix(j) for j in i] for i in parts]
 
     def _get_shape(elem):
         if isinstance(elem, MatrixExpr):
@@ -897,7 +897,7 @@ def build(self):
         data = [self._build(i) for i in self._lines]
         if self.higher != 1:
             data += [self._build(self.higher)]
-        data = [i.doit() for i in data]
+        data = [i for i in data]
         return data
 
     def matrix_form(self):

```

## Test Patch

```diff
diff --git a/sympy/matrices/expressions/tests/test_matexpr.py b/sympy/matrices/expressions/tests/test_matexpr.py
--- a/sympy/matrices/expressions/tests/test_matexpr.py
+++ b/sympy/matrices/expressions/tests/test_matexpr.py
@@ -1,8 +1,9 @@
 from sympy import (KroneckerDelta, diff, Sum, Dummy, factor,
                    expand, zeros, gcd_terms, Eq, Symbol)
 
-from sympy.core import S, symbols, Add, Mul, SympifyError, Rational
-from sympy.functions import sin, cos, sqrt, cbrt, exp
+from sympy.core import (S, symbols, Add, Mul, SympifyError, Rational,
+                    Function)
+from sympy.functions import sin, cos, tan, sqrt, cbrt, exp
 from sympy.simplify import simplify
 from sympy.matrices import (ImmutableMatrix, Inverse, MatAdd, MatMul,
         MatPow, Matrix, MatrixExpr, MatrixSymbol, ShapeError,
@@ -340,6 +341,18 @@ def test_issue_7842():
     assert Eq(A, B) == True
 
 
+def test_issue_21195():
+    t = symbols('t')
+    x = Function('x')(t)
+    dx = x.diff(t)
+    exp1 = cos(x) + cos(x)*dx
+    exp2 = sin(x) + tan(x)*(dx.diff(t))
+    exp3 = sin(x)*sin(t)*(dx.diff(t)).diff(t)
+    A = Matrix([[exp1], [exp2], [exp3]])
+    B = Matrix([[exp1.diff(x)], [exp2.diff(x)], [exp3.diff(x)]])
+    assert A.diff(x) == B
+
+
 def test_MatMul_postprocessor():
     z = zeros(2)
     z1 = ZeroMatrix(2, 2)

```


## Code snippets

### 1 - sympy/matrices/expressions/applyfunc.py:

Start line: 104, End line: 124

```python
class ElementwiseApplyFunction(MatrixExpr):

    def _entry(self, i, j, **kwargs):
        return self.function(self.expr._entry(i, j, **kwargs))

    def _get_function_fdiff(self):
        d = Dummy("d")
        function = self.function(d)
        fdiff = function.diff(d)
        if isinstance(fdiff, Function):
            fdiff = type(fdiff)
        else:
            fdiff = Lambda(d, fdiff)
        return fdiff

    def _eval_derivative(self, x):
        from sympy import hadamard_product
        dexpr = self.expr.diff(x)
        fdiff = self._get_function_fdiff()
        return hadamard_product(
            dexpr,
            ElementwiseApplyFunction(fdiff, self.expr)
        )
```
### 2 - sympy/matrices/expressions/applyfunc.py:

Start line: 126, End line: 192

```python
class ElementwiseApplyFunction(MatrixExpr):

    def _eval_derivative_matrix_lines(self, x):
        from sympy import Identity
        from sympy.tensor.array.expressions.array_expressions import ArrayContraction
        from sympy.tensor.array.expressions.array_expressions import ArrayDiagonal
        from sympy.tensor.array.expressions.array_expressions import ArrayTensorProduct
        from sympy.core.expr import ExprBuilder

        fdiff = self._get_function_fdiff()
        lr = self.expr._eval_derivative_matrix_lines(x)
        ewdiff = ElementwiseApplyFunction(fdiff, self.expr)
        if 1 in x.shape:
            # Vector:
            iscolumn = self.shape[1] == 1
            for i in lr:
                if iscolumn:
                    ptr1 = i.first_pointer
                    ptr2 = Identity(self.shape[1])
                else:
                    ptr1 = Identity(self.shape[0])
                    ptr2 = i.second_pointer

                subexpr = ExprBuilder(
                    ArrayDiagonal,
                    [
                        ExprBuilder(
                            ArrayTensorProduct,
                            [
                                ewdiff,
                                ptr1,
                                ptr2,
                            ]
                        ),
                        (0, 2) if iscolumn else (1, 4)
                    ],
                    validator=ArrayDiagonal._validate
                )
                i._lines = [subexpr]
                i._first_pointer_parent = subexpr.args[0].args
                i._first_pointer_index = 1
                i._second_pointer_parent = subexpr.args[0].args
                i._second_pointer_index = 2
        else:
            # Matrix case:
            for i in lr:
                ptr1 = i.first_pointer
                ptr2 = i.second_pointer
                newptr1 = Identity(ptr1.shape[1])
                newptr2 = Identity(ptr2.shape[1])
                subexpr = ExprBuilder(
                    ArrayContraction,
                    [
                        ExprBuilder(
                            ArrayTensorProduct,
                            [ptr1, newptr1, ewdiff, ptr2, newptr2]
                        ),
                        (1, 2, 4),
                        (5, 7, 8),
                    ],
                    validator=ArrayContraction._validate
                )
                i._first_pointer_parent = subexpr.args[0].args
                i._first_pointer_index = 1
                i._second_pointer_parent = subexpr.args[0].args
                i._second_pointer_index = 4
                i._lines = [subexpr]
        return lr
```
### 3 - sympy/matrices/matrices.py:

Start line: 438, End line: 469

```python
class MatrixCalculus(MatrixCommon):
    """Provides calculus-related matrix operations."""

    def diff(self, *args, **kwargs):
        """Calculate the derivative of each element in the matrix.
        ``args`` will be passed to the ``integrate`` function.

        Examples
        ========

        >>> from sympy.matrices import Matrix
        >>> from sympy.abc import x, y
        >>> M = Matrix([[x, y], [1, 0]])
        >>> M.diff(x)
        Matrix([
        [1, 0],
        [0, 0]])

        See Also
        ========

        integrate
        limit
        """
        # XXX this should be handled here rather than in Derivative
        from sympy.tensor.array.array_derivatives import ArrayDerivative
        kwargs.setdefault('evaluate', True)
        deriv = ArrayDerivative(self, *args, evaluate=True)
        if not isinstance(self, Basic):
            return deriv.as_mutable()
        else:
            return deriv
```
### 4 - sympy/matrices/expressions/matexpr.py:

Start line: 727, End line: 754

```python
class MatrixElement(Expr):

    def _eval_derivative(self, v):
        from sympy import Sum, symbols, Dummy

        if not isinstance(v, MatrixElement):
            from sympy import MatrixBase
            if isinstance(self.parent, MatrixBase):
                return self.parent.diff(v)[self.i, self.j]
            return S.Zero

        M = self.args[0]

        m, n = self.parent.shape

        if M == v.args[0]:
            return KroneckerDelta(self.args[1], v.args[1], (0, m-1)) * \
                   KroneckerDelta(self.args[2], v.args[2], (0, n-1))

        if isinstance(M, Inverse):
            i, j = self.args[1:]
            i1, i2 = symbols("z1, z2", cls=Dummy)
            Y = M.args[0]
            r1, r2 = Y.shape
            return -Sum(M[i, i1]*Y[i1, i2].diff(v)*M[i2, j], (i1, 0, r1-1), (i2, 0, r2-1))

        if self.has(v.args[0]):
            return None

        return S.Zero
```
### 5 - sympy/solvers/ode/systems.py:

Start line: 1087, End line: 1135

```python
def _is_commutative_anti_derivative(A, t):
    r"""
    Helper function for determining if the Matrix passed is commutative with its antiderivative

    Explanation
    ===========

    This function checks if the Matrix $A$ passed is commutative with its antiderivative with respect
    to the independent variable $t$.

    .. math::
        B(t) = \int A(t) dt

    The function outputs two values, first one being the antiderivative $B(t)$, second one being a
    boolean value, if True, then the matrix $A(t)$ passed is commutative with $B(t)$, else the matrix
    passed isn't commutative with $B(t)$.

    Parameters
    ==========

    A : Matrix
        The matrix which has to be checked
    t : Symbol
        Independent variable

    Examples
    ========

    >>> from sympy import symbols, Matrix
    >>> from sympy.solvers.ode.systems import _is_commutative_anti_derivative
    >>> t = symbols("t")
    >>> A = Matrix([[1, t], [-t, 1]])

    >>> B, is_commuting = _is_commutative_anti_derivative(A, t)
    >>> is_commuting
    True

    Returns
    =======

    Matrix, Boolean

    """
    B = integrate(A, t)
    is_commuting = (B*A - A*B).applyfunc(expand).applyfunc(factor_terms).is_zero_matrix

    is_commuting = False if is_commuting is None else is_commuting

    return B, is_commuting
```
### 6 - sympy/matrices/expressions/matexpr.py:

Start line: 669, End line: 692

```python
def _matrix_derivative(expr, x):
    # ... other code

    def contract_one_dims(parts):
        if len(parts) == 1:
            return parts[0]
        else:
            p1, p2 = parts[:2]
            if p2.is_Matrix:
                p2 = p2.T
            if p1 == Identity(1):
                pbase = p2
            elif p2 == Identity(1):
                pbase = p1
            else:
                pbase = p1*p2
            if len(parts) == 2:
                return pbase
            else:  # len(parts) > 2
                if pbase.is_Matrix:
                    raise ValueError("")
                return pbase*Mul.fromiter(parts[2:])

    if rank <= 2:
        return Add.fromiter([contract_one_dims(i) for i in parts])

    return ArrayDerivative(expr, x)
```
### 7 - sympy/tensor/array/expressions/arrayexpr_derivatives.py:

Start line: 113, End line: 125

```python
@array_derive.register(ArrayElementwiseApplyFunc)
def _(expr: ArrayElementwiseApplyFunc, x: Expr):
    fdiff = expr._get_function_fdiff()
    subexpr = expr.expr
    dsubexpr = array_derive(subexpr, x)
    tp = ArrayTensorProduct(
        dsubexpr,
        ArrayElementwiseApplyFunc(fdiff, subexpr)
    )
    b = get_rank(x)
    c = get_rank(expr)
    diag_indices = [(b + i, b + c + i) for i in range(c)]
    return ArrayDiagonal(tp, *diag_indices)
```
### 8 - sympy/matrices/expressions/applyfunc.py:

Start line: 73, End line: 102

```python
class ElementwiseApplyFunction(MatrixExpr):

    @property
    def function(self):
        return self.args[0]

    @property
    def expr(self):
        return self.args[1]

    @property
    def shape(self):
        return self.expr.shape

    def doit(self, **kwargs):
        deep = kwargs.get("deep", True)
        expr = self.expr
        if deep:
            expr = expr.doit(**kwargs)
        function = self.function
        if isinstance(function, Lambda) and function.is_identity:
            # This is a Lambda containing the identity function.
            return expr
        if isinstance(expr, MatrixBase):
            return expr.applyfunc(self.function)
        elif isinstance(expr, ElementwiseApplyFunction):
            return ElementwiseApplyFunction(
                lambda x: self.function(expr.function(x)),
                expr.expr
            ).doit()
        else:
            return self
```
### 9 - sympy/matrices/expressions/applyfunc.py:

Start line: 1, End line: 45

```python
from sympy.matrices.expressions import MatrixExpr
from sympy import MatrixBase, Dummy, Lambda, Function, FunctionClass
from sympy.core.sympify import sympify, _sympify


class ElementwiseApplyFunction(MatrixExpr):
    r"""
    Apply function to a matrix elementwise without evaluating.

    Examples
    ========

    It can be created by calling ``.applyfunc(<function>)`` on a matrix
    expression:

    >>> from sympy.matrices.expressions import MatrixSymbol
    >>> from sympy.matrices.expressions.applyfunc import ElementwiseApplyFunction
    >>> from sympy import exp
    >>> X = MatrixSymbol("X", 3, 3)
    >>> X.applyfunc(exp)
    Lambda(_d, exp(_d)).(X)

    Otherwise using the class constructor:

    >>> from sympy import eye
    >>> expr = ElementwiseApplyFunction(exp, eye(3))
    >>> expr
    Lambda(_d, exp(_d)).(Matrix([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]]))
    >>> expr.doit()
    Matrix([
    [E, 1, 1],
    [1, E, 1],
    [1, 1, E]])

    Notice the difference with the real mathematical functions:

    >>> exp(eye(3))
    Matrix([
    [E, 0, 0],
    [0, E, 0],
    [0, 0, E]])
    """
```
### 10 - sympy/matrices/expressions/applyfunc.py:

Start line: 47, End line: 71

```python
class ElementwiseApplyFunction(MatrixExpr):

    def __new__(cls, function, expr):
        expr = _sympify(expr)
        if not expr.is_Matrix:
            raise ValueError("{} must be a matrix instance.".format(expr))

        if not isinstance(function, (FunctionClass, Lambda)):
            d = Dummy('d')
            function = Lambda(d, function(d))

        function = sympify(function)
        if not isinstance(function, (FunctionClass, Lambda)):
            raise ValueError(
                "{} should be compatible with SymPy function classes."
                .format(function))

        if 1 not in function.nargs:
            raise ValueError(
                '{} should be able to accept 1 arguments.'.format(function))

        if not isinstance(function, Lambda):
            d = Dummy('d')
            function = Lambda(d, function(d))

        obj = MatrixExpr.__new__(cls, function, expr)
        return obj
```
### 23 - sympy/matrices/expressions/matexpr.py:

Start line: 642, End line: 667

```python
Basic._constructor_postprocessor_mapping[MatrixExpr] = {
    "Mul": [get_postprocessor(Mul)],
    "Add": [get_postprocessor(Add)],
}


def _matrix_derivative(expr, x):
    from sympy.tensor.array.array_derivatives import ArrayDerivative
    lines = expr._eval_derivative_matrix_lines(x)

    parts = [i.build() for i in lines]

    from sympy.tensor.array.expressions.conv_array_to_matrix import convert_array_to_matrix

    parts = [[convert_array_to_matrix(j).doit() for j in i] for i in parts]

    def _get_shape(elem):
        if isinstance(elem, MatrixExpr):
            return elem.shape
        return 1, 1

    def get_rank(parts):
        return sum([j not in (1, None) for i in parts for j in _get_shape(i)])

    ranks = [get_rank(i) for i in parts]
    rank = ranks[0]
    # ... other code
```
### 24 - sympy/matrices/expressions/matexpr.py:

Start line: 695, End line: 725

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
### 29 - sympy/matrices/expressions/matexpr.py:

Start line: 468, End line: 590

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
                        raise ValueError("index range mismatch: {} vs. (0, {})".format(
                            (r1, r2), matrix_symbol.shape[0]))
                if i2 in index_ranges:
                    r1, r2 = index_ranges[i2]
                    if r1 != 0 or matrix_symbol.shape[1] != r2+1:
                        raise ValueError("index range mismatch: {} vs. (0, {})".format(
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
```
### 45 - sympy/matrices/expressions/matexpr.py:

Start line: 811, End line: 827

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
### 49 - sympy/matrices/expressions/matexpr.py:

Start line: 150, End line: 268

```python
class MatrixExpr(Expr):

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__pow__')
    def __rpow__(self, other):
        raise NotImplementedError("Matrix Power not defined")

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__rtruediv__')
    def __truediv__(self, other):
        return self * other**S.NegativeOne

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__truediv__')
    def __rtruediv__(self, other):
        raise NotImplementedError()
        #return MatMul(other, Pow(self, S.NegativeOne))

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
        """
        Override this in sub-classes to implement simplification of powers.  The cases where the exponent
        is -1, 0, 1 are already covered in MatPow.doit(), so implementations can exclude these cases.
        """
        return MatPow(self, exp)

    def _eval_simplify(self, **kwargs):
        if self.is_Atom:
            return self
        else:
            return self.func(*[simplify(x, **kwargs) for x in self.args])

    def _eval_adjoint(self):
        from sympy.matrices.expressions.adjoint import Adjoint
        return Adjoint(self)

    def _eval_derivative_n_times(self, x, n):
        return Basic._eval_derivative_n_times(self, x, n)

    def _eval_derivative(self, x):
        # `x` is a scalar:
        if self.has(x):
            # See if there are other methods using it:
            return super()._eval_derivative(x)
        else:
            return ZeroMatrix(*self.shape)

    @classmethod
    def _check_dim(cls, dim):
        """Helper function to check invalid matrix dimensions"""
        from sympy.core.assumptions import check_assumptions
        ok = check_assumptions(dim, integer=True, nonnegative=True)
        if ok is False:
            raise ValueError(
                "The dimension specification {} should be "
                "a nonnegative integer.".format(dim))


    def _entry(self, i, j, **kwargs):
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

    @property
    def T(self):
        '''Matrix transposition'''
        return self.transpose()

    def inverse(self):
        if not self.is_square:
            raise NonSquareMatrixError('Inverse of non-square matrix')
        return self._eval_inverse()

    def inv(self):
        return self.inverse()

    @property
    def I(self):
        return self.inverse()
```
