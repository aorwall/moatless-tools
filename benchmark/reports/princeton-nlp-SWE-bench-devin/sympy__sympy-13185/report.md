# sympy__sympy-13185

| **sympy/sympy** | `5f35c254434bfda69ecf2b6879590ec6b3478136` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | - |
| **Avg pos** | - |
| **Min pos** | - |
| **Max pos** | - |
| **Top file pos** | 1 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sympy/simplify/cse_main.py b/sympy/simplify/cse_main.py
--- a/sympy/simplify/cse_main.py
+++ b/sympy/simplify/cse_main.py
@@ -485,6 +485,7 @@ def tree_cse(exprs, symbols, opt_subs=None, order='canonical', ignore=()):
         Substitutions containing any Symbol from ``ignore`` will be ignored.
     """
     from sympy.matrices.expressions import MatrixExpr, MatrixSymbol, MatMul, MatAdd
+    from sympy.matrices.expressions.matexpr import MatrixElement
 
     if opt_subs is None:
         opt_subs = dict()
@@ -500,7 +501,7 @@ def _find_repeated(expr):
         if not isinstance(expr, (Basic, Unevaluated)):
             return
 
-        if isinstance(expr, Basic) and (expr.is_Atom or expr.is_Order):
+        if isinstance(expr, Basic) and (expr.is_Atom or expr.is_Order or isinstance(expr, MatrixSymbol) or isinstance(expr, MatrixElement)):
             if expr.is_Symbol:
                 excluded_symbols.add(expr)
             return

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sympy/simplify/cse_main.py | 488 | 488 | - | 1 | -
| sympy/simplify/cse_main.py | 503 | 503 | - | 1 | -


## Problem Statement

```
cse() has strange behaviour for MatrixSymbol indexing
Example: 
\`\`\`python
import sympy as sp
from pprint import pprint


def sub_in_matrixsymbols(exp, matrices):
    for matrix in matrices:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                name = "%s_%d_%d" % (matrix.name, i, j)
                sym = sp.symbols(name)
                exp = exp.subs(sym, matrix[i, j])
    return exp


def t44(name):
    return sp.Matrix(4, 4, lambda i, j: sp.symbols('%s_%d_%d' % (name, i, j)))


# Construct matrices of symbols that work with our
# expressions. (MatrixSymbols does not.)
a = t44("a")
b = t44("b")

# Set up expression. This is a just a simple example.
e = a * b

# Put in matrixsymbols. (Gives array-input in codegen.)
e2 = sub_in_matrixsymbols(e, [sp.MatrixSymbol("a", 4, 4), sp.MatrixSymbol("b", 4, 4)])
cse_subs, cse_reduced = sp.cse(e2)
pprint((cse_subs, cse_reduced))

# Codegen, etc..
print "\nccode:"
for sym, expr in cse_subs:
    constants, not_c, c_expr = sympy.printing.ccode(
        expr,
        human=False,
        assign_to=sympy.printing.ccode(sym),
    )
    assert not constants, constants
    assert not not_c, not_c
    print "%s\n" % c_expr

\`\`\`

This gives the following output:

\`\`\`
([(x0, a),
  (x1, x0[0, 0]),
  (x2, b),
  (x3, x2[0, 0]),
  (x4, x0[0, 1]),
  (x5, x2[1, 0]),
  (x6, x0[0, 2]),
  (x7, x2[2, 0]),
  (x8, x0[0, 3]),
  (x9, x2[3, 0]),
  (x10, x2[0, 1]),
  (x11, x2[1, 1]),
  (x12, x2[2, 1]),
  (x13, x2[3, 1]),
  (x14, x2[0, 2]),
  (x15, x2[1, 2]),
  (x16, x2[2, 2]),
  (x17, x2[3, 2]),
  (x18, x2[0, 3]),
  (x19, x2[1, 3]),
  (x20, x2[2, 3]),
  (x21, x2[3, 3]),
  (x22, x0[1, 0]),
  (x23, x0[1, 1]),
  (x24, x0[1, 2]),
  (x25, x0[1, 3]),
  (x26, x0[2, 0]),
  (x27, x0[2, 1]),
  (x28, x0[2, 2]),
  (x29, x0[2, 3]),
  (x30, x0[3, 0]),
  (x31, x0[3, 1]),
  (x32, x0[3, 2]),
  (x33, x0[3, 3])],
 [Matrix([
[    x1*x3 + x4*x5 + x6*x7 + x8*x9,     x1*x10 + x11*x4 + x12*x6 + x13*x8,     x1*x14 + x15*x4 + x16*x6 + x17*x8,     x1*x18 + x19*x4 + x20*x6 + x21*x8],
[x22*x3 + x23*x5 + x24*x7 + x25*x9, x10*x22 + x11*x23 + x12*x24 + x13*x25, x14*x22 + x15*x23 + x16*x24 + x17*x25, x18*x22 + x19*x23 + x20*x24 + x21*x25],
[x26*x3 + x27*x5 + x28*x7 + x29*x9, x10*x26 + x11*x27 + x12*x28 + x13*x29, x14*x26 + x15*x27 + x16*x28 + x17*x29, x18*x26 + x19*x27 + x20*x28 + x21*x29],
[x3*x30 + x31*x5 + x32*x7 + x33*x9, x10*x30 + x11*x31 + x12*x32 + x13*x33, x14*x30 + x15*x31 + x16*x32 + x17*x33, x18*x30 + x19*x31 + x20*x32 + x21*x33]])])

ccode:
x0[0] = a[0];
x0[1] = a[1];
x0[2] = a[2];
x0[3] = a[3];
x0[4] = a[4];
x0[5] = a[5];
x0[6] = a[6];
x0[7] = a[7];
x0[8] = a[8];
x0[9] = a[9];
x0[10] = a[10];
x0[11] = a[11];
x0[12] = a[12];
x0[13] = a[13];
x0[14] = a[14];
x0[15] = a[15];
x1 = x0[0];
x2[0] = b[0];
x2[1] = b[1];
x2[2] = b[2];
x2[3] = b[3];
x2[4] = b[4];
x2[5] = b[5];
x2[6] = b[6];
x2[7] = b[7];
x2[8] = b[8];
x2[9] = b[9];
x2[10] = b[10];
x2[11] = b[11];
x2[12] = b[12];
x2[13] = b[13];
x2[14] = b[14];
x2[15] = b[15];
x3 = x2[0];
x4 = x0[1];
x5 = x2[4];
x6 = x0[2];
x7 = x2[8];
x8 = x0[3];
x9 = x2[12];
x10 = x2[1];
x11 = x2[5];
x12 = x2[9];
x13 = x2[13];
x14 = x2[2];
x15 = x2[6];
x16 = x2[10];
x17 = x2[14];
x18 = x2[3];
x19 = x2[7];
x20 = x2[11];
x21 = x2[15];
x22 = x0[4];
x23 = x0[5];
x24 = x0[6];
x25 = x0[7];
x26 = x0[8];
x27 = x0[9];
x28 = x0[10];
x29 = x0[11];
x30 = x0[12];
x31 = x0[13];
x32 = x0[14];
x33 = x0[15];
\`\`\`

`x0` and `x2` are just copies of the matrices `a` and `b`, respectively.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 sympy/simplify/cse_main.py** | 683 | 750| 569 | 569 | 5888 | 
| 2 | 2 sympy/matrices/expressions/matexpr.py | 632 | 688| 422 | 991 | 11956 | 
| 3 | 3 sympy/printing/ccode.py | 612 | 748| 1518 | 2509 | 18918 | 
| 4 | 4 examples/advanced/qft.py | 85 | 138| 607 | 3116 | 20233 | 
| 5 | 4 sympy/matrices/expressions/matexpr.py | 142 | 202| 443 | 3559 | 20233 | 
| 6 | 4 sympy/matrices/expressions/matexpr.py | 390 | 433| 415 | 3974 | 20233 | 
| 7 | 5 sympy/solvers/solvers.py | 3316 | 3414| 1432 | 5406 | 51123 | 
| 8 | 6 sympy/matrices/expressions/blockmatrix.py | 1 | 19| 206 | 5612 | 54801 | 
| 9 | 7 examples/intermediate/vandermonde.py | 114 | 172| 535 | 6147 | 56192 | 
| 10 | 7 sympy/solvers/solvers.py | 420 | 870| 4430 | 10577 | 56192 | 
| 11 | 8 sympy/integrals/rubi/utility_function.py | 1781 | 2270| 5930 | 16507 | 138211 | 
| 12 | 8 sympy/matrices/expressions/matexpr.py | 461 | 576| 1155 | 17662 | 138211 | 
| 13 | 8 sympy/matrices/expressions/matexpr.py | 33 | 140| 794 | 18456 | 138211 | 
| 14 | 9 sympy/solvers/solveset.py | 1032 | 1116| 682 | 19138 | 158341 | 
| 15 | 10 sympy/matrices/expressions/matmul.py | 78 | 133| 467 | 19605 | 160717 | 
| 16 | 11 sympy/matrices/sparse.py | 1 | 17| 117 | 19722 | 171160 | 
| 17 | 11 sympy/solvers/solveset.py | 1118 | 1138| 171 | 19893 | 171160 | 
| 18 | 11 sympy/matrices/expressions/matexpr.py | 435 | 459| 237 | 20130 | 171160 | 
| 19 | **11 sympy/simplify/cse_main.py** | 542 | 606| 427 | 20557 | 171160 | 
| 20 | 11 sympy/matrices/expressions/matmul.py | 48 | 76| 316 | 20873 | 171160 | 
| 21 | 11 examples/intermediate/vandermonde.py | 1 | 39| 236 | 21109 | 171160 | 


### Hint

```
Can you create a very simple example using MatrixSymbol and the expected output that you'd like to see?
I think one would expect the output to be similar to the following (except for the expression returned by CSE being a matrix where the individual elements are terms as defined by matrix multiplication, that is, unchanged by `cse()`).

\`\`\`py
import sympy as sp
from pprint import pprint
import sympy.printing.ccode


def print_ccode(assign_to, expr):
    constants, not_c, c_expr = sympy.printing.ccode(
        expr,
        human=False,
        assign_to=assign_to,
    )
    assert not constants, constants
    assert not not_c, not_c
    print "%s" % c_expr


a = sp.MatrixSymbol("a", 4, 4)
b = sp.MatrixSymbol("b", 4, 4)

# Set up expression. This is a just a simple example.
e = a * b
print "\nexpr:"
print e

cse_subs, cse_reduced = sp.cse(e)
print "\ncse(expr):"
pprint((cse_subs, cse_reduced))

# Codegen.
print "\nccode:"
for sym, expr in cse_subs:
    print_ccode(sympy.printing.ccode(sym), expr)
assert len(cse_reduced) == 1
print_ccode(sympy.printing.ccode(sp.symbols("result")), cse_reduced[0])
\`\`\`

Gives the output:

\`\`\`
expr:
a*b

cse(expr):
([], [a*b])

ccode:
result[0] = a[0]*b[0] + a[1]*b[4] + a[2]*b[8] + a[3]*b[12];
result[1] = a[0]*b[1] + a[1]*b[5] + a[2]*b[9] + a[3]*b[13];
result[2] = a[0]*b[2] + a[1]*b[6] + a[2]*b[10] + a[3]*b[14];
result[3] = a[0]*b[3] + a[1]*b[7] + a[2]*b[11] + a[3]*b[15];
result[4] = a[4]*b[0] + a[5]*b[4] + a[6]*b[8] + a[7]*b[12];
result[5] = a[4]*b[1] + a[5]*b[5] + a[6]*b[9] + a[7]*b[13];
result[6] = a[4]*b[2] + a[5]*b[6] + a[6]*b[10] + a[7]*b[14];
result[7] = a[4]*b[3] + a[5]*b[7] + a[6]*b[11] + a[7]*b[15];
result[8] = a[8]*b[0] + a[9]*b[4] + a[10]*b[8] + a[11]*b[12];
result[9] = a[8]*b[1] + a[9]*b[5] + a[10]*b[9] + a[11]*b[13];
result[10] = a[8]*b[2] + a[9]*b[6] + a[10]*b[10] + a[11]*b[14];
result[11] = a[8]*b[3] + a[9]*b[7] + a[10]*b[11] + a[11]*b[15];
result[12] = a[12]*b[0] + a[13]*b[4] + a[14]*b[8] + a[15]*b[12];
result[13] = a[12]*b[1] + a[13]*b[5] + a[14]*b[9] + a[15]*b[13];
result[14] = a[12]*b[2] + a[13]*b[6] + a[14]*b[10] + a[15]*b[14];
result[15] = a[12]*b[3] + a[13]*b[7] + a[14]*b[11] + a[15]*b[15];
\`\`\`
Thanks. Note that it doesn't look like cse is well tested (i.e. designed) for MatrixSymbols based on the unit tests: https://github.com/sympy/sympy/blob/master/sympy/simplify/tests/test_cse.py#L315. Those tests don't really prove that it works as desired. So this definitely needs to be fixed.
The first part works as expected:

\`\`\`
In [1]: import sympy as sm

In [2]: M = sm.MatrixSymbol('M', 3, 3)

In [3]: B = sm.MatrixSymbol('B', 3, 3)

In [4]: M * B
Out[4]: M*B

In [5]: sm.cse(M * B)
Out[5]: ([], [M*B])
\`\`\`
For the ccode of an expression of MatrixSymbols, I would not expect it to print the results as you have them. MatrixSymbols should map to a matrix algebra library like BLAS and LINPACK. But Matrix, on the other hand, should do what you expect. Note how this works:

\`\`\`
In [8]: M = sm.Matrix(3, 3, lambda i, j: sm.Symbol('M_{}{}'.format(i, j)))

In [9]: M
Out[9]: 
Matrix([
[M_00, M_01, M_02],
[M_10, M_11, M_12],
[M_20, M_21, M_22]])

In [10]: B = sm.Matrix(3, 3, lambda i, j: sm.Symbol('B_{}{}'.format(i, j)))

In [11]: B
Out[11]: 
Matrix([
[B_00, B_01, B_02],
[B_10, B_11, B_12],
[B_20, B_21, B_22]])

In [12]: M * B
Out[12]: 
Matrix([
[B_00*M_00 + B_10*M_01 + B_20*M_02, B_01*M_00 + B_11*M_01 + B_21*M_02, B_02*M_00 + B_12*M_01 + B_22*M_02],
[B_00*M_10 + B_10*M_11 + B_20*M_12, B_01*M_10 + B_11*M_11 + B_21*M_12, B_02*M_10 + B_12*M_11 + B_22*M_12],
[B_00*M_20 + B_10*M_21 + B_20*M_22, B_01*M_20 + B_11*M_21 + B_21*M_22, B_02*M_20 + B_12*M_21 + B_22*M_22]])

In [13]: sm.cse(M * B)
Out[13]: 
([], [Matrix([
  [B_00*M_00 + B_10*M_01 + B_20*M_02, B_01*M_00 + B_11*M_01 + B_21*M_02, B_02*M_00 + B_12*M_01 + B_22*M_02],
  [B_00*M_10 + B_10*M_11 + B_20*M_12, B_01*M_10 + B_11*M_11 + B_21*M_12, B_02*M_10 + B_12*M_11 + B_22*M_12],
  [B_00*M_20 + B_10*M_21 + B_20*M_22, B_01*M_20 + B_11*M_21 + B_21*M_22, B_02*M_20 + B_12*M_21 + B_22*M_22]])])

In [17]: print(sm.ccode(M * B, assign_to=sm.MatrixSymbol('E', 3, 3)))
E[0] = B_00*M_00 + B_10*M_01 + B_20*M_02;
E[1] = B_01*M_00 + B_11*M_01 + B_21*M_02;
E[2] = B_02*M_00 + B_12*M_01 + B_22*M_02;
E[3] = B_00*M_10 + B_10*M_11 + B_20*M_12;
E[4] = B_01*M_10 + B_11*M_11 + B_21*M_12;
E[5] = B_02*M_10 + B_12*M_11 + B_22*M_12;
E[6] = B_00*M_20 + B_10*M_21 + B_20*M_22;
E[7] = B_01*M_20 + B_11*M_21 + B_21*M_22;
E[8] = B_02*M_20 + B_12*M_21 + B_22*M_22;
\`\`\`
But in order to get a single input argument from codegen it cannot be different symbols, and if you replace each symbol with a `MatrixSymbol[i, j]` then `cse()` starts doing the above non-optiimizations for some reason.
As far as I know, `codegen` does not work with Matrix or MatrixSymbol's in any meaningful way. There are related issues:

#11456
#4367
#10522

In general, there needs to be work done in the code generators to properly support matrices.

As a work around, I suggest using `ccode` and a custom template to get the result you want.
```

## Patch

```diff
diff --git a/sympy/simplify/cse_main.py b/sympy/simplify/cse_main.py
--- a/sympy/simplify/cse_main.py
+++ b/sympy/simplify/cse_main.py
@@ -485,6 +485,7 @@ def tree_cse(exprs, symbols, opt_subs=None, order='canonical', ignore=()):
         Substitutions containing any Symbol from ``ignore`` will be ignored.
     """
     from sympy.matrices.expressions import MatrixExpr, MatrixSymbol, MatMul, MatAdd
+    from sympy.matrices.expressions.matexpr import MatrixElement
 
     if opt_subs is None:
         opt_subs = dict()
@@ -500,7 +501,7 @@ def _find_repeated(expr):
         if not isinstance(expr, (Basic, Unevaluated)):
             return
 
-        if isinstance(expr, Basic) and (expr.is_Atom or expr.is_Order):
+        if isinstance(expr, Basic) and (expr.is_Atom or expr.is_Order or isinstance(expr, MatrixSymbol) or isinstance(expr, MatrixElement)):
             if expr.is_Symbol:
                 excluded_symbols.add(expr)
             return

```

## Test Patch

```diff
diff --git a/sympy/simplify/tests/test_cse.py b/sympy/simplify/tests/test_cse.py
--- a/sympy/simplify/tests/test_cse.py
+++ b/sympy/simplify/tests/test_cse.py
@@ -323,6 +323,10 @@ def test_cse_MatrixSymbol():
     B = MatrixSymbol("B", n, n)
     assert cse(B) == ([], [B])
 
+    assert cse(A[0] * A[0]) == ([], [A[0]*A[0]])
+
+    assert cse(A[0,0]*A[0,1] + A[0,0]*A[0,1]*A[0,2]) == ([(x0, A[0, 0]*A[0, 1])], [x0*A[0, 2] + x0])
+
 def test_cse_MatrixExpr():
     from sympy import MatrixSymbol
     A = MatrixSymbol('A', 3, 3)

```


## Code snippets

### 1 - sympy/simplify/cse_main.py:

Start line: 683, End line: 750

```python
def cse(exprs, symbols=None, optimizations=None, postprocess=None,
        order='canonical', ignore=()):
    from sympy.matrices import (MatrixBase, Matrix, ImmutableMatrix,
                                SparseMatrix, ImmutableSparseMatrix)

    # Handle the case if just one expression was passed.
    if isinstance(exprs, (Basic, MatrixBase)):
        exprs = [exprs]

    copy = exprs
    temp = []
    for e in exprs:
        if isinstance(e, (Matrix, ImmutableMatrix)):
            temp.append(Tuple(*e._mat))
        elif isinstance(e, (SparseMatrix, ImmutableSparseMatrix)):
            temp.append(Tuple(*e._smat.items()))
        else:
            temp.append(e)
    exprs = temp
    del temp

    if optimizations is None:
        optimizations = list()
    elif optimizations == 'basic':
        optimizations = basic_optimizations

    # Preprocess the expressions to give us better optimization opportunities.
    reduced_exprs = [preprocess_for_cse(e, optimizations) for e in exprs]

    if symbols is None:
        symbols = numbered_symbols(cls=Symbol)
    else:
        # In case we get passed an iterable with an __iter__ method instead of
        # an actual iterator.
        symbols = iter(symbols)

    # Find other optimization opportunities.
    opt_subs = opt_cse(reduced_exprs, order)

    # Main CSE algorithm.
    replacements, reduced_exprs = tree_cse(reduced_exprs, symbols, opt_subs,
                                           order, ignore)

    # Postprocess the expressions to return the expressions to canonical form.
    exprs = copy
    for i, (sym, subtree) in enumerate(replacements):
        subtree = postprocess_for_cse(subtree, optimizations)
        replacements[i] = (sym, subtree)
    reduced_exprs = [postprocess_for_cse(e, optimizations)
                     for e in reduced_exprs]

    # Get the matrices back
    for i, e in enumerate(exprs):
        if isinstance(e, (Matrix, ImmutableMatrix)):
            reduced_exprs[i] = Matrix(e.rows, e.cols, reduced_exprs[i])
            if isinstance(e, ImmutableMatrix):
                reduced_exprs[i] = reduced_exprs[i].as_immutable()
        elif isinstance(e, (SparseMatrix, ImmutableSparseMatrix)):
            m = SparseMatrix(e.rows, e.cols, {})
            for k, v in reduced_exprs[i]:
                m[k] = v
            if isinstance(e, ImmutableSparseMatrix):
                m = m.as_immutable()
            reduced_exprs[i] = m

    if postprocess is None:
        return replacements, reduced_exprs

    return postprocess(replacements, reduced_exprs)
```
### 2 - sympy/matrices/expressions/matexpr.py:

Start line: 632, End line: 688

```python
class MatrixSymbol(MatrixExpr):
    """Symbolic representation of a Matrix object

    Creates a SymPy Symbol to represent a Matrix. This matrix has a shape and
    can be included in Matrix Expressions

    >>> from sympy import MatrixSymbol, Identity
    >>> A = MatrixSymbol('A', 3, 4) # A 3 by 4 Matrix
    >>> B = MatrixSymbol('B', 4, 3) # A 4 by 3 Matrix
    >>> A.shape
    (3, 4)
    >>> 2*A*B + Identity(3)
    I + 2*A*B
    """
    is_commutative = False
    _diff_wrt = True

    def __new__(cls, name, n, m):
        n, m = sympify(n), sympify(m)
        obj = Basic.__new__(cls, name, n, m)
        return obj

    def _hashable_content(self):
        return(self.name, self.shape)

    @property
    def shape(self):
        return self.args[1:3]

    @property
    def name(self):
        return self.args[0]

    def _eval_subs(self, old, new):
        # only do substitutions in shape
        shape = Tuple(*self.shape)._subs(old, new)
        return MatrixSymbol(self.name, *shape)

    def __call__(self, *args):
        raise TypeError( "%s object is not callable" % self.__class__ )

    def _entry(self, i, j, **kwargs):
        return MatrixElement(self, i, j)

    @property
    def free_symbols(self):
        return set((self,))

    def doit(self, **hints):
        if hints.get('deep', True):
            return type(self)(self.name, self.args[1].doit(**hints),
                    self.args[2].doit(**hints))
        else:
            return self

    def _eval_simplify(self, **kwargs):
        return self
```
### 3 - sympy/printing/ccode.py:

Start line: 612, End line: 748

```python
def ccode(expr, assign_to=None, standard='c99', **settings):
    """Converts an expr to a string of c code

    Parameters
    ==========

    expr : Expr
        A sympy expression to be converted.
    assign_to : optional
        When given, the argument is used as the name of the variable to which
        the expression is assigned. Can be a string, ``Symbol``,
        ``MatrixSymbol``, or ``Indexed`` type. This is helpful in case of
        line-wrapping, or for expressions that generate multi-line statements.
    standard : str, optional
        String specifying the standard. If your compiler supports a more modern
        standard you may set this to 'c99' to allow the printer to use more math
        functions. [default='c89'].
    precision : integer, optional
        The precision for numbers such as pi [default=17].
    user_functions : dict, optional
        A dictionary where the keys are string representations of either
        ``FunctionClass`` or ``UndefinedFunction`` instances and the values
        are their desired C string representations. Alternatively, the
        dictionary value can be a list of tuples i.e. [(argument_test,
        cfunction_string)] or [(argument_test, cfunction_formater)]. See below
        for examples.
    dereference : iterable, optional
        An iterable of symbols that should be dereferenced in the printed code
        expression. These would be values passed by address to the function.
        For example, if ``dereference=[a]``, the resulting code would print
        ``(*a)`` instead of ``a``.
    human : bool, optional
        If True, the result is a single string that may contain some constant
        declarations for the number symbols. If False, the same information is
        returned in a tuple of (symbols_to_declare, not_supported_functions,
        code_text). [default=True].
    contract: bool, optional
        If True, ``Indexed`` instances are assumed to obey tensor contraction
        rules and the corresponding nested loops over indices are generated.
        Setting contract=False will not generate loops, instead the user is
        responsible to provide values for the indices in the code.
        [default=True].

    Examples
    ========

    >>> from sympy import ccode, symbols, Rational, sin, ceiling, Abs, Function
    >>> x, tau = symbols("x, tau")
    >>> expr = (2*tau)**Rational(7, 2)
    >>> ccode(expr)
    '8*M_SQRT2*pow(tau, 7.0/2.0)'
    >>> ccode(expr, math_macros={})
    '8*sqrt(2)*pow(tau, 7.0/2.0)'
    >>> ccode(sin(x), assign_to="s")
    's = sin(x);'
    >>> from sympy.codegen.ast import real, float80
    >>> ccode(expr, type_aliases={real: float80})
    '8*M_SQRT2l*powl(tau, 7.0L/2.0L)'

    Simple custom printing can be defined for certain types by passing a
    dictionary of {"type" : "function"} to the ``user_functions`` kwarg.
    Alternatively, the dictionary value can be a list of tuples i.e.
    [(argument_test, cfunction_string)].

    >>> custom_functions = {
    ...   "ceiling": "CEIL",
    ...   "Abs": [(lambda x: not x.is_integer, "fabs"),
    ...           (lambda x: x.is_integer, "ABS")],
    ...   "func": "f"
    ... }
    >>> func = Function('func')
    >>> ccode(func(Abs(x) + ceiling(x)), standard='C89', user_functions=custom_functions)
    'f(fabs(x) + CEIL(x))'

    or if the C-function takes a subset of the original arguments:

    >>> ccode(2**x + 3**x, standard='C99', user_functions={'Pow': [
    ...   (lambda b, e: b == 2, lambda b, e: 'exp2(%s)' % e),
    ...   (lambda b, e: b != 2, 'pow')]})
    'exp2(x) + pow(3, x)'

    ``Piecewise`` expressions are converted into conditionals. If an
    ``assign_to`` variable is provided an if statement is created, otherwise
    the ternary operator is used. Note that if the ``Piecewise`` lacks a
    default term, represented by ``(expr, True)`` then an error will be thrown.
    This is to prevent generating an expression that may not evaluate to
    anything.

    >>> from sympy import Piecewise
    >>> expr = Piecewise((x + 1, x > 0), (x, True))
    >>> print(ccode(expr, tau, standard='C89'))
    if (x > 0) {
    tau = x + 1;
    }
    else {
    tau = x;
    }

    Support for loops is provided through ``Indexed`` types. With
    ``contract=True`` these expressions will be turned into loops, whereas
    ``contract=False`` will just print the assignment expression that should be
    looped over:

    >>> from sympy import Eq, IndexedBase, Idx
    >>> len_y = 5
    >>> y = IndexedBase('y', shape=(len_y,))
    >>> t = IndexedBase('t', shape=(len_y,))
    >>> Dy = IndexedBase('Dy', shape=(len_y-1,))
    >>> i = Idx('i', len_y-1)
    >>> e=Eq(Dy[i], (y[i+1]-y[i])/(t[i+1]-t[i]))
    >>> ccode(e.rhs, assign_to=e.lhs, contract=False, standard='C89')
    'Dy[i] = (y[i + 1] - y[i])/(t[i + 1] - t[i]);'

    Matrices are also supported, but a ``MatrixSymbol`` of the same dimensions
    must be provided to ``assign_to``. Note that any expression that can be
    generated normally can also exist inside a Matrix:

    >>> from sympy import Matrix, MatrixSymbol
    >>> mat = Matrix([x**2, Piecewise((x + 1, x > 0), (x, True)), sin(x)])
    >>> A = MatrixSymbol('A', 3, 1)
    >>> print(ccode(mat, A, standard='C89'))
    A[0] = pow(x, 2);
    if (x > 0) {
       A[1] = x + 1;
    }
    else {
       A[1] = x;
    }
    A[2] = sin(x);
    """
    return c_code_printers[standard.lower()](settings).doprint(expr, assign_to)


def print_ccode(expr, **settings):
    """Prints C representation of the given expression."""
    print(ccode(expr, **settings))
```
### 4 - examples/advanced/qft.py:

Start line: 85, End line: 138

```python
def main():
    a = Symbol("a", real=True)
    b = Symbol("b", real=True)
    c = Symbol("c", real=True)

    p = (a, b, c)

    assert u(p, 1).D*u(p, 2) == Matrix(1, 1, [0])
    assert u(p, 2).D*u(p, 1) == Matrix(1, 1, [0])

    p1, p2, p3 = [Symbol(x, real=True) for x in ["p1", "p2", "p3"]]
    pp1, pp2, pp3 = [Symbol(x, real=True) for x in ["pp1", "pp2", "pp3"]]
    k1, k2, k3 = [Symbol(x, real=True) for x in ["k1", "k2", "k3"]]
    kp1, kp2, kp3 = [Symbol(x, real=True) for x in ["kp1", "kp2", "kp3"]]

    p = (p1, p2, p3)
    pp = (pp1, pp2, pp3)

    k = (k1, k2, k3)
    kp = (kp1, kp2, kp3)

    mu = Symbol("mu")

    e = (pslash(p) + m*ones(4))*(pslash(k) - m*ones(4))
    f = pslash(p) + m*ones(4)
    g = pslash(p) - m*ones(4)

    xprint('Tr(f*g)', Tr(f*g))

    M0 = [(v(pp, 1).D*mgamma(mu)*u(p, 1))*(u(k, 1).D*mgamma(mu, True) *
                                                 v(kp, 1)) for mu in range(4)]
    M = M0[0] + M0[1] + M0[2] + M0[3]
    M = M[0]
    if not isinstance(M, Basic):
        raise TypeError("Invalid type of variable")

    d = Symbol("d", real=True)  # d=E+m

    xprint('M', M)
    print("-"*40)
    M = ((M.subs(E, d - m)).expand()*d**2).expand()
    xprint('M2', 1 / (E + m)**2*M)
    print("-"*40)
    x, y = M.as_real_imag()
    xprint('Re(M)', x)
    xprint('Im(M)', y)
    e = x**2 + y**2
    xprint('abs(M)**2', e)
    print("-"*40)
    xprint('Expand(abs(M)**2)', e.expand())

if __name__ == "__main__":
    main()
```
### 5 - sympy/matrices/expressions/matexpr.py:

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
### 6 - sympy/matrices/expressions/matexpr.py:

Start line: 390, End line: 433

```python
class MatrixExpr(Expr):

    @staticmethod
    def from_index_summation(expr, first_index=None, last_index=None):
        r"""
        Parse expression of matrices with explicitly summed indices into a
        matrix expression without indices, if possible.

        This transformation expressed in mathematical notation:

        `\sum_{j=0}^{N-1} A_{i,j} B_{j,k} \Longrightarrow \mathbf{A}\cdot \mathbf{B}`

        Optional parameter ``first_index``: specify which free index to use as
        the index starting the expression.

        Examples
        ========

        >>> from sympy import MatrixSymbol, MatrixExpr, Sum, Symbol
        >>> from sympy.abc import i, j, k, l, N
        >>> A = MatrixSymbol("A", N, N)
        >>> B = MatrixSymbol("B", N, N)
        >>> expr = Sum(A[i, j]*B[j, k], (j, 0, N-1))
        >>> MatrixExpr.from_index_summation(expr)
        A*B

        Transposition is detected:

        >>> expr = Sum(A[j, i]*B[j, k], (j, 0, N-1))
        >>> MatrixExpr.from_index_summation(expr)
        A.T*B

        Detect the trace:

        >>> expr = Sum(A[i, i], (i, 0, N-1))
        >>> MatrixExpr.from_index_summation(expr)
        Trace(A)

        More complicated expressions:

        >>> expr = Sum(A[i, j]*B[k, j]*A[l, k], (j, 0, N-1), (k, 0, N-1))
        >>> MatrixExpr.from_index_summation(expr)
        A*B.T*A.T
        """
        from sympy import Sum, Mul, Add, MatMul, transpose, trace
        from sympy.strategies.traverse import bottom_up
        # ... other code
```
### 7 - sympy/solvers/solvers.py:

Start line: 3316, End line: 3414

```python
def unrad(eq, *syms, **flags):

    if len(rterms) == 1 and not (rterms[0].is_Add and lcm > 2):
    else:
        # ... other code

        if len(rterms) == 2:
            if not others:
                eq = rterms[0]**lcm - (-rterms[1])**lcm
                ok = True
            elif not log(lcm, 2).is_Integer:
                # the lcm-is-power-of-two case is handled below
                r0, r1 = rterms
                if flags.get('_reverse', False):
                    r1, r0 = r0, r1
                i0 = _rads0, _bases0, lcm0 = _rads_bases_lcm(r0.as_poly())
                i1 = _rads1, _bases1, lcm1 = _rads_bases_lcm(r1.as_poly())
                for reverse in range(2):
                    if reverse:
                        i0, i1 = i1, i0
                        r0, r1 = r1, r0
                    _rads1, _, lcm1 = i1
                    _rads1 = Mul(*_rads1)
                    t1 = _rads1**lcm1
                    c = covsym**lcm1 - t1
                    for x in syms:
                        try:
                            sol = _solve(c, x, **uflags)
                            if not sol:
                                raise NotImplementedError
                            neweq = r0.subs(x, sol[0]) + covsym*r1/_rads1 + \
                                others
                            tmp = unrad(neweq, covsym)
                            if tmp:
                                eq, newcov = tmp
                                if newcov:
                                    newp, newc = newcov
                                    _cov(newp, c.subs(covsym,
                                        _solve(newc, covsym, **uflags)[0]))
                                else:
                                    _cov(covsym, c)
                            else:
                                eq = neweq
                                _cov(covsym, c)
                            ok = True
                            break
                        except NotImplementedError:
                            if reverse:
                                raise NotImplementedError(
                                    'no successful change of variable found')
                            else:
                                pass
                    if ok:
                        break
        elif len(rterms) == 3:
            # two cube roots and another with order less than 5
            # (so an analytical solution can be found) or a base
            # that matches one of the cube root bases
            info = [_rads_bases_lcm(i.as_poly()) for i in rterms]
            RAD = 0
            BASES = 1
            LCM = 2
            if info[0][LCM] != 3:
                info.append(info.pop(0))
                rterms.append(rterms.pop(0))
            elif info[1][LCM] != 3:
                info.append(info.pop(1))
                rterms.append(rterms.pop(1))
            if info[0][LCM] == info[1][LCM] == 3:
                if info[1][BASES] != info[2][BASES]:
                    info[0], info[1] = info[1], info[0]
                    rterms[0], rterms[1] = rterms[1], rterms[0]
                if info[1][BASES] == info[2][BASES]:
                    eq = rterms[0]**3 + (rterms[1] + rterms[2] + others)**3
                    ok = True
                elif info[2][LCM] < 5:
                    # a*root(A, 3) + b*root(B, 3) + others = c
                    a, b, c, d, A, B = [Dummy(i) for i in 'abcdAB']
                    # zz represents the unraded expression into which the
                    # specifics for this case are substituted
                    zz = (c - d)*(A**3*a**9 + 3*A**2*B*a**6*b**3 -
                        3*A**2*a**6*c**3 + 9*A**2*a**6*c**2*d - 9*A**2*a**6*c*d**2 +
                        3*A**2*a**6*d**3 + 3*A*B**2*a**3*b**6 + 21*A*B*a**3*b**3*c**3 -
                        63*A*B*a**3*b**3*c**2*d + 63*A*B*a**3*b**3*c*d**2 -
                        21*A*B*a**3*b**3*d**3 + 3*A*a**3*c**6 - 18*A*a**3*c**5*d +
                        45*A*a**3*c**4*d**2 - 60*A*a**3*c**3*d**3 + 45*A*a**3*c**2*d**4 -
                        18*A*a**3*c*d**5 + 3*A*a**3*d**6 + B**3*b**9 - 3*B**2*b**6*c**3 +
                        9*B**2*b**6*c**2*d - 9*B**2*b**6*c*d**2 + 3*B**2*b**6*d**3 +
                        3*B*b**3*c**6 - 18*B*b**3*c**5*d + 45*B*b**3*c**4*d**2 -
                        60*B*b**3*c**3*d**3 + 45*B*b**3*c**2*d**4 - 18*B*b**3*c*d**5 +
                        3*B*b**3*d**6 - c**9 + 9*c**8*d - 36*c**7*d**2 + 84*c**6*d**3 -
                        126*c**5*d**4 + 126*c**4*d**5 - 84*c**3*d**6 + 36*c**2*d**7 -
                        9*c*d**8 + d**9)
                    def _t(i):
                        b = Mul(*info[i][RAD])
                        return cancel(rterms[i]/b), Mul(*info[i][BASES])
                    aa, AA = _t(0)
                    bb, BB = _t(1)
                    cc = -rterms[2]
                    dd = others
                    eq = zz.xreplace(dict(zip(
                        (a, A, b, B, c, d),
                        (aa, AA, bb, BB, cc, dd))))
                    ok = True
        # handle power-of-2 cases
        # ... other code
    # ... other code
```
### 8 - sympy/matrices/expressions/blockmatrix.py:

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
### 9 - examples/intermediate/vandermonde.py:

Start line: 114, End line: 172

```python
def main():
    order = 2
    V, tmp_syms, _ = vandermonde(order)
    print("Vandermonde matrix of order 2 in 1 dimension")
    pprint(V)

    print('-'*79)
    print("Computing the determinant and comparing to \sum_{0<i<j<=3}(a_j - a_i)")

    det_sum = 1
    for j in range(order + 1):
        for i in range(j):
            det_sum *= (tmp_syms[j][0] - tmp_syms[i][0])

    print("""
    det(V) = %(det)s
    \sum   = %(sum)s
           = %(sum_expand)s
    """ % {"det": V.det(),
            "sum": det_sum,
            "sum_expand": det_sum.expand(),
          })

    print('-'*79)
    print("Polynomial fitting with a Vandermonde Matrix:")
    x, y, z = symbols('x,y,z')

    points = [(0, 3), (1, 2), (2, 3)]
    print("""
    Quadratic function, represented by 3 points:
       points = %(pts)s
       f = %(f)s
    """ % {"pts": points,
            "f": gen_poly(points, 2, [x]),
          })

    points = [(0, 1, 1), (1, 0, 0), (1, 1, 0), (Rational(1, 2), 0, 0),
              (0, Rational(1, 2), 0), (Rational(1, 2), Rational(1, 2), 0)]
    print("""
    2D Quadratic function, represented by 6 points:
       points = %(pts)s
       f = %(f)s
    """ % {"pts": points,
            "f": gen_poly(points, 2, [x, y]),
          })

    points = [(0, 1, 1, 1), (1, 1, 0, 0), (1, 0, 1, 0), (1, 1, 1, 1)]
    print("""
    3D linear function, represented by 4 points:
       points = %(pts)s
       f = %(f)s
    """ % {"pts": points,
            "f": gen_poly(points, 1, [x, y, z]),
          })


if __name__ == "__main__":
    main()
```
### 10 - sympy/solvers/solvers.py:

Start line: 420, End line: 870

```python
def solve(f, *symbols, **flags):
    r"""
    Algebraically solves equations and systems of equations.

    Currently supported are:
        - polynomial,
        - transcendental
        - piecewise combinations of the above
        - systems of linear and polynomial equations
        - systems containing relational expressions.

    Input is formed as:

    * f
        - a single Expr or Poly that must be zero,
        - an Equality
        - a Relational expression or boolean
        - iterable of one or more of the above

    * symbols (object(s) to solve for) specified as
        - none given (other non-numeric objects will be used)
        - single symbol
        - denested list of symbols
          e.g. solve(f, x, y)
        - ordered iterable of symbols
          e.g. solve(f, [x, y])

    * flags
        'dict'=True (default is False)
            return list (perhaps empty) of solution mappings
        'set'=True (default is False)
            return list of symbols and set of tuple(s) of solution(s)
        'exclude=[] (default)'
            don't try to solve for any of the free symbols in exclude;
            if expressions are given, the free symbols in them will
            be extracted automatically.
        'check=True (default)'
            If False, don't do any testing of solutions. This can be
            useful if one wants to include solutions that make any
            denominator zero.
        'numerical=True (default)'
            do a fast numerical check if ``f`` has only one symbol.
        'minimal=True (default is False)'
            a very fast, minimal testing.
        'warn=True (default is False)'
            show a warning if checksol() could not conclude.
        'simplify=True (default)'
            simplify all but polynomials of order 3 or greater before
            returning them and (if check is not False) use the
            general simplify function on the solutions and the
            expression obtained when they are substituted into the
            function which should be zero
        'force=True (default is False)'
            make positive all symbols without assumptions regarding sign.
        'rational=True (default)'
            recast Floats as Rational; if this option is not used, the
            system containing floats may fail to solve because of issues
            with polys. If rational=None, Floats will be recast as
            rationals but the answer will be recast as Floats. If the
            flag is False then nothing will be done to the Floats.
        'manual=True (default is False)'
            do not use the polys/matrix method to solve a system of
            equations, solve them one at a time as you might "manually"
        'implicit=True (default is False)'
            allows solve to return a solution for a pattern in terms of
            other functions that contain that pattern; this is only
            needed if the pattern is inside of some invertible function
            like cos, exp, ....
        'particular=True (default is False)'
            instructs solve to try to find a particular solution to a linear
            system with as many zeros as possible; this is very expensive
        'quick=True (default is False)'
            when using particular=True, use a fast heuristic instead to find a
            solution with many zeros (instead of using the very slow method
            guaranteed to find the largest number of zeros possible)
        'cubics=True (default)'
            return explicit solutions when cubic expressions are encountered
        'quartics=True (default)'
            return explicit solutions when quartic expressions are encountered
        'quintics=True (default)'
            return explicit solutions (if possible) when quintic expressions
            are encountered

    Examples
    ========

    The output varies according to the input and can be seen by example::

        >>> from sympy import solve, Poly, Eq, Function, exp
        >>> from sympy.abc import x, y, z, a, b
        >>> f = Function('f')

    * boolean or univariate Relational

        >>> solve(x < 3)
        (-oo < x) & (x < 3)


    * to always get a list of solution mappings, use flag dict=True

        >>> solve(x - 3, dict=True)
        [{x: 3}]
        >>> sol = solve([x - 3, y - 1], dict=True)
        >>> sol
        [{x: 3, y: 1}]
        >>> sol[0][x]
        3
        >>> sol[0][y]
        1


    * to get a list of symbols and set of solution(s) use flag set=True

        >>> solve([x**2 - 3, y - 1], set=True)
        ([x, y], {(-sqrt(3), 1), (sqrt(3), 1)})


    * single expression and single symbol that is in the expression

        >>> solve(x - y, x)
        [y]
        >>> solve(x - 3, x)
        [3]
        >>> solve(Eq(x, 3), x)
        [3]
        >>> solve(Poly(x - 3), x)
        [3]
        >>> solve(x**2 - y**2, x, set=True)
        ([x], {(-y,), (y,)})
        >>> solve(x**4 - 1, x, set=True)
        ([x], {(-1,), (1,), (-I,), (I,)})

    * single expression with no symbol that is in the expression

        >>> solve(3, x)
        []
        >>> solve(x - 3, y)
        []

    * single expression with no symbol given

          In this case, all free symbols will be selected as potential
          symbols to solve for. If the equation is univariate then a list
          of solutions is returned; otherwise -- as is the case when symbols are
          given as an iterable of length > 1 -- a list of mappings will be returned.

            >>> solve(x - 3)
            [3]
            >>> solve(x**2 - y**2)
            [{x: -y}, {x: y}]
            >>> solve(z**2*x**2 - z**2*y**2)
            [{x: -y}, {x: y}, {z: 0}]
            >>> solve(z**2*x - z**2*y**2)
            [{x: y**2}, {z: 0}]

    * when an object other than a Symbol is given as a symbol, it is
      isolated algebraically and an implicit solution may be obtained.
      This is mostly provided as a convenience to save one from replacing
      the object with a Symbol and solving for that Symbol. It will only
      work if the specified object can be replaced with a Symbol using the
      subs method.

          >>> solve(f(x) - x, f(x))
          [x]
          >>> solve(f(x).diff(x) - f(x) - x, f(x).diff(x))
          [x + f(x)]
          >>> solve(f(x).diff(x) - f(x) - x, f(x))
          [-x + Derivative(f(x), x)]
          >>> solve(x + exp(x)**2, exp(x), set=True)
          ([exp(x)], {(-sqrt(-x),), (sqrt(-x),)})

          >>> from sympy import Indexed, IndexedBase, Tuple, sqrt
          >>> A = IndexedBase('A')
          >>> eqs = Tuple(A[1] + A[2] - 3, A[1] - A[2] + 1)
          >>> solve(eqs, eqs.atoms(Indexed))
          {A[1]: 1, A[2]: 2}

        * To solve for a *symbol* implicitly, use 'implicit=True':

            >>> solve(x + exp(x), x)
            [-LambertW(1)]
            >>> solve(x + exp(x), x, implicit=True)
            [-exp(x)]

        * It is possible to solve for anything that can be targeted with
          subs:

            >>> solve(x + 2 + sqrt(3), x + 2)
            [-sqrt(3)]
            >>> solve((x + 2 + sqrt(3), x + 4 + y), y, x + 2)
            {y: -2 + sqrt(3), x + 2: -sqrt(3)}

        * Nothing heroic is done in this implicit solving so you may end up
          with a symbol still in the solution:

            >>> eqs = (x*y + 3*y + sqrt(3), x + 4 + y)
            >>> solve(eqs, y, x + 2)
            {y: -sqrt(3)/(x + 3), x + 2: (-2*x - 6 + sqrt(3))/(x + 3)}
            >>> solve(eqs, y*x, x)
            {x: -y - 4, x*y: -3*y - sqrt(3)}

        * if you attempt to solve for a number remember that the number
          you have obtained does not necessarily mean that the value is
          equivalent to the expression obtained:

            >>> solve(sqrt(2) - 1, 1)
            [sqrt(2)]
            >>> solve(x - y + 1, 1)  # /!\ -1 is targeted, too
            [x/(y - 1)]
            >>> [_.subs(z, -1) for _ in solve((x - y + 1).subs(-1, z), 1)]
            [-x + y]

        * To solve for a function within a derivative, use dsolve.

    * single expression and more than 1 symbol

        * when there is a linear solution

            >>> solve(x - y**2, x, y)
            [{x: y**2}]
            >>> solve(x**2 - y, x, y)
            [{y: x**2}]

        * when undetermined coefficients are identified

            * that are linear

                >>> solve((a + b)*x - b + 2, a, b)
                {a: -2, b: 2}

            * that are nonlinear

                >>> solve((a + b)*x - b**2 + 2, a, b, set=True)
                ([a, b], {(-sqrt(2), sqrt(2)), (sqrt(2), -sqrt(2))})

        * if there is no linear solution then the first successful
          attempt for a nonlinear solution will be returned

            >>> solve(x**2 - y**2, x, y)
            [{x: -y}, {x: y}]
            >>> solve(x**2 - y**2/exp(x), x, y)
            [{x: 2*LambertW(y/2)}]
            >>> solve(x**2 - y**2/exp(x), y, x)
            [{y: -x*sqrt(exp(x))}, {y: x*sqrt(exp(x))}]

    * iterable of one or more of the above

        * involving relationals or bools

            >>> solve([x < 3, x - 2])
            Eq(x, 2)
            >>> solve([x > 3, x - 2])
            False

        * when the system is linear

            * with a solution

                >>> solve([x - 3], x)
                {x: 3}
                >>> solve((x + 5*y - 2, -3*x + 6*y - 15), x, y)
                {x: -3, y: 1}
                >>> solve((x + 5*y - 2, -3*x + 6*y - 15), x, y, z)
                {x: -3, y: 1}
                >>> solve((x + 5*y - 2, -3*x + 6*y - z), z, x, y)
                {x: -5*y + 2, z: 21*y - 6}

            * without a solution

                >>> solve([x + 3, x - 3])
                []

        * when the system is not linear

            >>> solve([x**2 + y -2, y**2 - 4], x, y, set=True)
            ([x, y], {(-2, -2), (0, 2), (2, -2)})

        * if no symbols are given, all free symbols will be selected and a list
          of mappings returned

            >>> solve([x - 2, x**2 + y])
            [{x: 2, y: -4}]
            >>> solve([x - 2, x**2 + f(x)], {f(x), x})
            [{x: 2, f(x): -4}]

        * if any equation doesn't depend on the symbol(s) given it will be
          eliminated from the equation set and an answer may be given
          implicitly in terms of variables that were not of interest

            >>> solve([x - y, y - 3], x)
            {x: y}

    Notes
    =====

    solve() with check=True (default) will run through the symbol tags to
    elimate unwanted solutions.  If no assumptions are included all possible
    solutions will be returned.

        >>> from sympy import Symbol, solve
        >>> x = Symbol("x")
        >>> solve(x**2 - 1)
        [-1, 1]

    By using the positive tag only one solution will be returned:

        >>> pos = Symbol("pos", positive=True)
        >>> solve(pos**2 - 1)
        [1]


    Assumptions aren't checked when `solve()` input involves
    relationals or bools.

    When the solutions are checked, those that make any denominator zero
    are automatically excluded. If you do not want to exclude such solutions
    then use the check=False option:

        >>> from sympy import sin, limit
        >>> solve(sin(x)/x)  # 0 is excluded
        [pi]

    If check=False then a solution to the numerator being zero is found: x = 0.
    In this case, this is a spurious solution since sin(x)/x has the well known
    limit (without dicontinuity) of 1 at x = 0:

        >>> solve(sin(x)/x, check=False)
        [0, pi]

    In the following case, however, the limit exists and is equal to the the
    value of x = 0 that is excluded when check=True:

        >>> eq = x**2*(1/x - z**2/x)
        >>> solve(eq, x)
        []
        >>> solve(eq, x, check=False)
        [0]
        >>> limit(eq, x, 0, '-')
        0
        >>> limit(eq, x, 0, '+')
        0

    Disabling high-order, explicit solutions
    ----------------------------------------

    When solving polynomial expressions, one might not want explicit solutions
    (which can be quite long). If the expression is univariate, CRootOf
    instances will be returned instead:

        >>> solve(x**3 - x + 1)
        [-1/((-1/2 - sqrt(3)*I/2)*(3*sqrt(69)/2 + 27/2)**(1/3)) - (-1/2 -
        sqrt(3)*I/2)*(3*sqrt(69)/2 + 27/2)**(1/3)/3, -(-1/2 +
        sqrt(3)*I/2)*(3*sqrt(69)/2 + 27/2)**(1/3)/3 - 1/((-1/2 +
        sqrt(3)*I/2)*(3*sqrt(69)/2 + 27/2)**(1/3)), -(3*sqrt(69)/2 +
        27/2)**(1/3)/3 - 1/(3*sqrt(69)/2 + 27/2)**(1/3)]
        >>> solve(x**3 - x + 1, cubics=False)
        [CRootOf(x**3 - x + 1, 0),
         CRootOf(x**3 - x + 1, 1),
         CRootOf(x**3 - x + 1, 2)]

        If the expression is multivariate, no solution might be returned:

        >>> solve(x**3 - x + a, x, cubics=False)
        []

    Sometimes solutions will be obtained even when a flag is False because the
    expression could be factored. In the following example, the equation can
    be factored as the product of a linear and a quadratic factor so explicit
    solutions (which did not require solving a cubic expression) are obtained:

        >>> eq = x**3 + 3*x**2 + x - 1
        >>> solve(eq, cubics=False)
        [-1, -1 + sqrt(2), -sqrt(2) - 1]

    Solving equations involving radicals
    ------------------------------------

    Because of SymPy's use of the principle root (issue #8789), some solutions
    to radical equations will be missed unless check=False:

        >>> from sympy import root
        >>> eq = root(x**3 - 3*x**2, 3) + 1 - x
        >>> solve(eq)
        []
        >>> solve(eq, check=False)
        [1/3]

    In the above example there is only a single solution to the equation. Other
    expressions will yield spurious roots which must be checked manually;
    roots which give a negative argument to odd-powered radicals will also need
    special checking:

        >>> from sympy import real_root, S
        >>> eq = root(x, 3) - root(x, 5) + S(1)/7
        >>> solve(eq)  # this gives 2 solutions but misses a 3rd
        [CRootOf(7*_p**5 - 7*_p**3 + 1, 1)**15,
        CRootOf(7*_p**5 - 7*_p**3 + 1, 2)**15]
        >>> sol = solve(eq, check=False)
        >>> [abs(eq.subs(x,i).n(2)) for i in sol]
        [0.48, 0.e-110, 0.e-110, 0.052, 0.052]

        The first solution is negative so real_root must be used to see that
        it satisfies the expression:

        >>> abs(real_root(eq.subs(x, sol[0])).n(2))
        0.e-110

    If the roots of the equation are not real then more care will be necessary
    to find the roots, especially for higher order equations. Consider the
    following expression:

        >>> expr = root(x, 3) - root(x, 5)

    We will construct a known value for this expression at x = 3 by selecting
    the 1-th root for each radical:

        >>> expr1 = root(x, 3, 1) - root(x, 5, 1)
        >>> v = expr1.subs(x, -3)

    The solve function is unable to find any exact roots to this equation:

        >>> eq = Eq(expr, v); eq1 = Eq(expr1, v)
        >>> solve(eq, check=False), solve(eq1, check=False)
        ([], [])

    The function unrad, however, can be used to get a form of the equation for
    which numerical roots can be found:

        >>> from sympy.solvers.solvers import unrad
        >>> from sympy import nroots
        >>> e, (p, cov) = unrad(eq)
        >>> pvals = nroots(e)
        >>> inversion = solve(cov, x)[0]
        >>> xvals = [inversion.subs(p, i) for i in pvals]

    Although eq or eq1 could have been used to find xvals, the solution can
    only be verified with expr1:

        >>> z = expr - v
        >>> [xi.n(chop=1e-9) for xi in xvals if abs(z.subs(x, xi).n()) < 1e-9]
        []
        >>> z1 = expr1 - v
        >>> [xi.n(chop=1e-9) for xi in xvals if abs(z1.subs(x, xi).n()) < 1e-9]
        [-3.0]

    See Also
    ========

        - rsolve() for solving recurrence relationships
        - dsolve() for solving differential equations

    """
    # ... other code
```
### 19 - sympy/simplify/cse_main.py:

Start line: 542, End line: 606

```python
def tree_cse(exprs, symbols, opt_subs=None, order='canonical', ignore=()):
    # ... other code

    def _rebuild(expr):
        if not isinstance(expr, (Basic, Unevaluated)):
            return expr

        if not expr.args:
            return expr

        if iterable(expr):
            new_args = [_rebuild(arg) for arg in expr]
            return expr.func(*new_args)

        if expr in subs:
            return subs[expr]

        orig_expr = expr
        if expr in opt_subs:
            expr = opt_subs[expr]

        # If enabled, parse Muls and Adds arguments by order to ensure
        # replacement order independent from hashes
        if order != 'none':
            if isinstance(expr, (Mul, MatMul)):
                c, nc = expr.args_cnc()
                if c == [1]:
                    args = nc
                else:
                    args = list(ordered(c)) + nc
            elif isinstance(expr, (Add, MatAdd)):
                args = list(ordered(expr.args))
            else:
                args = expr.args
        else:
            args = expr.args

        new_args = list(map(_rebuild, args))
        if isinstance(expr, Unevaluated) or new_args != args:
            new_expr = expr.func(*new_args)
        else:
            new_expr = expr

        if orig_expr in to_eliminate:
            try:
                sym = next(symbols)
            except StopIteration:
                raise ValueError("Symbols iterator ran out of symbols.")

            if isinstance(orig_expr, MatrixExpr):
                sym = MatrixSymbol(sym.name, orig_expr.rows,
                    orig_expr.cols)

            subs[orig_expr] = sym
            replacements.append((sym, new_expr))
            return sym

        else:
            return new_expr

    reduced_exprs = []
    for e in exprs:
        if isinstance(e, Basic):
            reduced_e = _rebuild(e)
        else:
            reduced_e = e
        reduced_exprs.append(reduced_e)
    return replacements, reduced_exprs
```
