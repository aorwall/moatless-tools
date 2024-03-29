# sympy__sympy-18030

| **sympy/sympy** | `7501960ea18912f9055a32be50bda30805fc0c95` |
| ---- | ---- |
| **No of patches** | 2 |
| **All found context length** | - |
| **Any found context length** | 432 |
| **Avg pos** | 2.0 |
| **Min pos** | 1 |
| **Max pos** | 2 |
| **Top file pos** | 1 |
| **Missing snippets** | 6 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sympy/polys/polyfuncs.py b/sympy/polys/polyfuncs.py
--- a/sympy/polys/polyfuncs.py
+++ b/sympy/polys/polyfuncs.py
@@ -2,7 +2,7 @@
 
 from __future__ import print_function, division
 
-from sympy.core import S, Basic, Add, Mul, symbols
+from sympy.core import S, Basic, Add, Mul, symbols, Dummy
 from sympy.core.compatibility import range
 from sympy.functions.combinatorial.factorials import factorial
 from sympy.polys.polyerrors import (
@@ -205,13 +205,14 @@ def horner(f, *gens, **args):
 @public
 def interpolate(data, x):
     """
-    Construct an interpolating polynomial for the data points.
+    Construct an interpolating polynomial for the data points
+    evaluated at point x (which can be symbolic or numeric).
 
     Examples
     ========
 
     >>> from sympy.polys.polyfuncs import interpolate
-    >>> from sympy.abc import x
+    >>> from sympy.abc import a, b, x
 
     A list is interpreted as though it were paired with a range starting
     from 1:
@@ -232,30 +233,39 @@ def interpolate(data, x):
     >>> interpolate({-1: 2, 1: 2, 2: 5}, x)
     x**2 + 1
 
+    If the interpolation is going to be used only once then the
+    value of interest can be passed instead of passing a symbol:
+
+    >>> interpolate([1, 4, 9], 5)
+    25
+
+    Symbolic coordinates are also supported:
+
+    >>> [(i,interpolate((a, b), i)) for i in range(1, 4)]
+    [(1, a), (2, b), (3, -a + 2*b)]
     """
     n = len(data)
-    poly = None
 
     if isinstance(data, dict):
+        if x in data:
+            return S(data[x])
         X, Y = list(zip(*data.items()))
-        poly = interpolating_poly(n, x, X, Y)
     else:
         if isinstance(data[0], tuple):
             X, Y = list(zip(*data))
-            poly = interpolating_poly(n, x, X, Y)
+            if x in X:
+                return S(Y[X.index(x)])
         else:
+            if x in range(1, n + 1):
+                return S(data[x - 1])
             Y = list(data)
+            X = list(range(1, n + 1))
 
-            numert = Mul(*[(x - i) for i in range(1, n + 1)])
-            denom = -factorial(n - 1) if n%2 == 0 else factorial(n - 1)
-            coeffs = []
-            for i in range(1, n + 1):
-                coeffs.append(numert/(x - i)/denom)
-                denom = denom/(i - n)*i
-
-            poly = Add(*[coeff*y for coeff, y in zip(coeffs, Y)])
-
-    return poly.expand()
+    try:
+        return interpolating_poly(n, x, X, Y).expand()
+    except ValueError:
+        d = Dummy()
+        return interpolating_poly(n, d, X, Y).expand().subs(d, x)
 
 
 @public
diff --git a/sympy/polys/specialpolys.py b/sympy/polys/specialpolys.py
--- a/sympy/polys/specialpolys.py
+++ b/sympy/polys/specialpolys.py
@@ -4,6 +4,7 @@
 
 from sympy.core import Add, Mul, Symbol, sympify, Dummy, symbols
 from sympy.core.compatibility import range, string_types
+from sympy.core.containers import Tuple
 from sympy.core.singleton import S
 from sympy.functions.elementary.miscellaneous import sqrt
 from sympy.ntheory import nextprime
@@ -19,7 +20,7 @@
 from sympy.polys.polyclasses import DMP
 from sympy.polys.polytools import Poly, PurePoly
 from sympy.polys.polyutils import _analyze_gens
-from sympy.utilities import subsets, public
+from sympy.utilities import subsets, public, filldedent
 
 
 @public
@@ -142,15 +143,29 @@ def random_poly(x, n, inf, sup, domain=ZZ, polys=False):
 
 @public
 def interpolating_poly(n, x, X='x', Y='y'):
-    """Construct Lagrange interpolating polynomial for ``n`` data points. """
+    """Construct Lagrange interpolating polynomial for ``n``
+    data points. If a sequence of values are given for ``X`` and ``Y``
+    then the first ``n`` values will be used.
+    """
+    ok = getattr(x, 'free_symbols', None)
+
     if isinstance(X, string_types):
         X = symbols("%s:%s" % (X, n))
+    elif ok and ok & Tuple(*X).free_symbols:
+        ok = False
 
     if isinstance(Y, string_types):
         Y = symbols("%s:%s" % (Y, n))
+    elif ok and ok & Tuple(*Y).free_symbols:
+        ok = False
+
+    if not ok:
+        raise ValueError(filldedent('''
+            Expecting symbol for x that does not appear in X or Y.
+            Use `interpolate(list(zip(X, Y)), x)` instead.'''))
 
     coeffs = []
-    numert = Mul(*[(x - u) for u in X])
+    numert = Mul(*[x - X[i] for i in range(n)])
 
     for i in range(n):
         numer = numert/(x - X[i])

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sympy/polys/polyfuncs.py | 5 | 5 | - | 1 | -
| sympy/polys/polyfuncs.py | 208 | 214 | 1 | 1 | 432
| sympy/polys/polyfuncs.py | 235 | 256 | 1 | 1 | 432
| sympy/polys/specialpolys.py | 7 | 7 | - | 2 | -
| sympy/polys/specialpolys.py | 22 | 22 | - | 2 | -
| sympy/polys/specialpolys.py | 145 | 153 | 2 | 2 | 601


## Problem Statement

```
interpolate could provide value instead of nan
\`\`\`python
>>> y = (18,25,43,70,115)
>>> interpolate(y,5)
nan
\`\`\`
Since the default x value for interpolation is `range(1, len(y)+1)` the interpolation at 5 could just return 115 instead of nan.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 sympy/polys/polyfuncs.py** | 205 | 258| 432 | 432 | 2996 | 
| **-> 2 <-** | **2 sympy/polys/specialpolys.py** | 143 | 160| 169 | 601 | 7354 | 
| 3 | 3 sympy/functions/special/bsplines.py | 285 | 299| 172 | 773 | 10195 | 
| 4 | **3 sympy/polys/polyfuncs.py** | 261 | 318| 597 | 1370 | 10195 | 
| 5 | 4 sympy/plotting/pygletplot/plot_interval.py | 167 | 183| 143 | 1513 | 11522 | 
| 6 | 5 sympy/plotting/intervalmath/lib_interval.py | 310 | 328| 142 | 1655 | 15173 | 
| 7 | 6 sympy/core/expr.py | 3682 | 3757| 907 | 2562 | 48300 | 
| 8 | 6 sympy/plotting/intervalmath/lib_interval.py | 331 | 349| 142 | 2704 | 48300 | 
| 9 | 6 sympy/functions/special/bsplines.py | 216 | 284| 776 | 3480 | 48300 | 
| 10 | 7 sympy/plotting/intervalmath/interval_arithmetic.py | 372 | 417| 360 | 3840 | 51493 | 
| 11 | 8 sympy/polys/euclidtools.py | 1253 | 1267| 139 | 3979 | 66344 | 
| 12 | 9 sympy/calculus/finite_diff.py | 201 | 286| 770 | 4749 | 71596 | 
| 13 | 10 sympy/plotting/textplot.py | 1 | 46| 338 | 5087 | 72832 | 
| 14 | 11 sympy/polys/heuristicgcd.py | 122 | 152| 210 | 5297 | 73923 | 
| 15 | 12 sympy/plotting/plot_implicit.py | 154 | 169| 190 | 5487 | 77388 | 
| 16 | 12 sympy/plotting/plot_implicit.py | 95 | 122| 353 | 5840 | 77388 | 
| 17 | 13 sympy/utilities/iterables.py | 1216 | 1232| 113 | 5953 | 99262 | 
| 18 | 14 sympy/plotting/pygletplot/util.py | 105 | 141| 294 | 6247 | 100754 | 
| 19 | 15 sympy/core/power.py | 42 | 112| 541 | 6788 | 116216 | 
| 20 | 15 sympy/utilities/iterables.py | 1197 | 1213| 109 | 6897 | 116216 | 
| 21 | 15 sympy/plotting/intervalmath/interval_arithmetic.py | 281 | 320| 341 | 7238 | 116216 | 
| 22 | 15 sympy/core/power.py | 170 | 252| 1060 | 8298 | 116216 | 
| 23 | 16 sympy/sets/handlers/mul.py | 1 | 43| 299 | 8597 | 116722 | 
| 24 | 17 sympy/integrals/meijerint.py | 1641 | 1691| 616 | 9213 | 141009 | 
| 25 | 17 sympy/plotting/plot_implicit.py | 74 | 93| 181 | 9394 | 141009 | 
| 26 | 17 sympy/plotting/intervalmath/interval_arithmetic.py | 244 | 279| 292 | 9686 | 141009 | 
| 27 | 17 sympy/plotting/intervalmath/lib_interval.py | 35 | 54| 163 | 9849 | 141009 | 
| 28 | 18 sympy/holonomic/holonomic.py | 1099 | 1129| 264 | 10113 | 165683 | 
| 29 | 18 sympy/holonomic/holonomic.py | 2589 | 2632| 356 | 10469 | 165683 | 
| 30 | 18 sympy/plotting/intervalmath/lib_interval.py | 390 | 411| 204 | 10673 | 165683 | 
| 31 | 18 sympy/plotting/intervalmath/interval_arithmetic.py | 343 | 369| 210 | 10883 | 165683 | 
| 32 | 19 sympy/polys/numberfields.py | 1071 | 1084| 147 | 11030 | 174849 | 
| 33 | 20 sympy/solvers/recurr.py | 726 | 830| 715 | 11745 | 181559 | 
| 34 | 20 sympy/plotting/pygletplot/plot_interval.py | 113 | 165| 355 | 12100 | 181559 | 
| 35 | 20 sympy/plotting/intervalmath/interval_arithmetic.py | 1 | 39| 332 | 12432 | 181559 | 
| 36 | 20 sympy/plotting/intervalmath/lib_interval.py | 57 | 89| 265 | 12697 | 181559 | 
| 37 | 20 sympy/plotting/pygletplot/plot_interval.py | 1 | 111| 838 | 13535 | 181559 | 
| 38 | 20 sympy/plotting/plot_implicit.py | 123 | 152| 309 | 13844 | 181559 | 


### Hint

```
The simplest fix is to have the function check to see if x is not a symbol:
\`\`\`python
>>> interpolate((1,2,3),1)
nan
>>> interpolate((1,2,3),x).subs(x,1)
1
\`\`\`
So in the function a check at the top would be like
\`\`\`python
if not isinstance(x, Symbol):
    d = Dummy()
    return interpolate(data, d).subs(d, x)
\`\`\`
Or the docstring could be annotated to say that `x` should be a Symbol.
There now exist two functions, `interpolate` and `interpolating_poly`, that construct an interpolating polynomial, only their parameters are slightly different. It is reasonable that `interpolating_poly` would return a polynomial (expression) in the given symbol. However, `interpolate` could return the *value* of the polynomial at the given *point*. So I am +1 for this change.
```

## Patch

```diff
diff --git a/sympy/polys/polyfuncs.py b/sympy/polys/polyfuncs.py
--- a/sympy/polys/polyfuncs.py
+++ b/sympy/polys/polyfuncs.py
@@ -2,7 +2,7 @@
 
 from __future__ import print_function, division
 
-from sympy.core import S, Basic, Add, Mul, symbols
+from sympy.core import S, Basic, Add, Mul, symbols, Dummy
 from sympy.core.compatibility import range
 from sympy.functions.combinatorial.factorials import factorial
 from sympy.polys.polyerrors import (
@@ -205,13 +205,14 @@ def horner(f, *gens, **args):
 @public
 def interpolate(data, x):
     """
-    Construct an interpolating polynomial for the data points.
+    Construct an interpolating polynomial for the data points
+    evaluated at point x (which can be symbolic or numeric).
 
     Examples
     ========
 
     >>> from sympy.polys.polyfuncs import interpolate
-    >>> from sympy.abc import x
+    >>> from sympy.abc import a, b, x
 
     A list is interpreted as though it were paired with a range starting
     from 1:
@@ -232,30 +233,39 @@ def interpolate(data, x):
     >>> interpolate({-1: 2, 1: 2, 2: 5}, x)
     x**2 + 1
 
+    If the interpolation is going to be used only once then the
+    value of interest can be passed instead of passing a symbol:
+
+    >>> interpolate([1, 4, 9], 5)
+    25
+
+    Symbolic coordinates are also supported:
+
+    >>> [(i,interpolate((a, b), i)) for i in range(1, 4)]
+    [(1, a), (2, b), (3, -a + 2*b)]
     """
     n = len(data)
-    poly = None
 
     if isinstance(data, dict):
+        if x in data:
+            return S(data[x])
         X, Y = list(zip(*data.items()))
-        poly = interpolating_poly(n, x, X, Y)
     else:
         if isinstance(data[0], tuple):
             X, Y = list(zip(*data))
-            poly = interpolating_poly(n, x, X, Y)
+            if x in X:
+                return S(Y[X.index(x)])
         else:
+            if x in range(1, n + 1):
+                return S(data[x - 1])
             Y = list(data)
+            X = list(range(1, n + 1))
 
-            numert = Mul(*[(x - i) for i in range(1, n + 1)])
-            denom = -factorial(n - 1) if n%2 == 0 else factorial(n - 1)
-            coeffs = []
-            for i in range(1, n + 1):
-                coeffs.append(numert/(x - i)/denom)
-                denom = denom/(i - n)*i
-
-            poly = Add(*[coeff*y for coeff, y in zip(coeffs, Y)])
-
-    return poly.expand()
+    try:
+        return interpolating_poly(n, x, X, Y).expand()
+    except ValueError:
+        d = Dummy()
+        return interpolating_poly(n, d, X, Y).expand().subs(d, x)
 
 
 @public
diff --git a/sympy/polys/specialpolys.py b/sympy/polys/specialpolys.py
--- a/sympy/polys/specialpolys.py
+++ b/sympy/polys/specialpolys.py
@@ -4,6 +4,7 @@
 
 from sympy.core import Add, Mul, Symbol, sympify, Dummy, symbols
 from sympy.core.compatibility import range, string_types
+from sympy.core.containers import Tuple
 from sympy.core.singleton import S
 from sympy.functions.elementary.miscellaneous import sqrt
 from sympy.ntheory import nextprime
@@ -19,7 +20,7 @@
 from sympy.polys.polyclasses import DMP
 from sympy.polys.polytools import Poly, PurePoly
 from sympy.polys.polyutils import _analyze_gens
-from sympy.utilities import subsets, public
+from sympy.utilities import subsets, public, filldedent
 
 
 @public
@@ -142,15 +143,29 @@ def random_poly(x, n, inf, sup, domain=ZZ, polys=False):
 
 @public
 def interpolating_poly(n, x, X='x', Y='y'):
-    """Construct Lagrange interpolating polynomial for ``n`` data points. """
+    """Construct Lagrange interpolating polynomial for ``n``
+    data points. If a sequence of values are given for ``X`` and ``Y``
+    then the first ``n`` values will be used.
+    """
+    ok = getattr(x, 'free_symbols', None)
+
     if isinstance(X, string_types):
         X = symbols("%s:%s" % (X, n))
+    elif ok and ok & Tuple(*X).free_symbols:
+        ok = False
 
     if isinstance(Y, string_types):
         Y = symbols("%s:%s" % (Y, n))
+    elif ok and ok & Tuple(*Y).free_symbols:
+        ok = False
+
+    if not ok:
+        raise ValueError(filldedent('''
+            Expecting symbol for x that does not appear in X or Y.
+            Use `interpolate(list(zip(X, Y)), x)` instead.'''))
 
     coeffs = []
-    numert = Mul(*[(x - u) for u in X])
+    numert = Mul(*[x - X[i] for i in range(n)])
 
     for i in range(n):
         numer = numert/(x - X[i])

```

## Test Patch

```diff
diff --git a/sympy/polys/tests/test_polyfuncs.py b/sympy/polys/tests/test_polyfuncs.py
--- a/sympy/polys/tests/test_polyfuncs.py
+++ b/sympy/polys/tests/test_polyfuncs.py
@@ -84,6 +84,11 @@ def test_interpolate():
         -S(13)*x**3/24 + S(12)*x**2 - S(2003)*x/24 + 187
     assert interpolate([(1, 3), (0, 6), (2, 5), (5, 7), (-2, 4)], x) == \
         S(-61)*x**4/280 + S(247)*x**3/210 + S(139)*x**2/280 - S(1871)*x/420 + 6
+    assert interpolate((9, 4, 9), 3) == 9
+    assert interpolate((1, 9, 16), 1) is S.One
+    assert interpolate(((x, 1), (2, 3)), x) is S.One
+    assert interpolate(dict([(x, 1), (2, 3)]), x) is S.One
+    assert interpolate(((2, x), (1, 3)), x) == x**2 - 4*x + 6
 
 
 def test_rational_interpolate():
diff --git a/sympy/polys/tests/test_specialpolys.py b/sympy/polys/tests/test_specialpolys.py
--- a/sympy/polys/tests/test_specialpolys.py
+++ b/sympy/polys/tests/test_specialpolys.py
@@ -1,6 +1,6 @@
 """Tests for functions for generating interesting polynomials. """
 
-from sympy import Poly, ZZ, symbols, sqrt, prime, Add
+from sympy import Poly, ZZ, symbols, sqrt, prime, Add, S
 from sympy.utilities.iterables import permute_signs
 from sympy.utilities.pytest import raises
 
@@ -96,6 +96,20 @@ def test_interpolating_poly():
         y2*(x - x0)*(x - x1)*(x - x3)/((x2 - x0)*(x2 - x1)*(x2 - x3)) + \
         y3*(x - x0)*(x - x1)*(x - x2)/((x3 - x0)*(x3 - x1)*(x3 - x2))
 
+    raises(ValueError, lambda:
+        interpolating_poly(2, x, (x, 2), (1, 3)))
+    raises(ValueError, lambda:
+        interpolating_poly(2, x, (x + y, 2), (1, 3)))
+    raises(ValueError, lambda:
+        interpolating_poly(2, x + y, (x, 2), (1, 3)))
+    raises(ValueError, lambda:
+        interpolating_poly(2, 3, (4, 5), (6, 7)))
+    raises(ValueError, lambda:
+        interpolating_poly(2, 3, (4, 5), (6, 7, 8)))
+    assert interpolating_poly(0, x, (1, 2), (3, 4)) == 0
+    assert interpolating_poly(1, x, (1, 2), (3, 4)) == 3
+    assert interpolating_poly(2, x, (1, 2), (3, 4)) == x + 2
+
 
 def test_fateman_poly_F_1():
     f, g, h = fateman_poly_F_1(1)

```


## Code snippets

### 1 - sympy/polys/polyfuncs.py:

Start line: 205, End line: 258

```python
@public
def interpolate(data, x):
    """
    Construct an interpolating polynomial for the data points.

    Examples
    ========

    >>> from sympy.polys.polyfuncs import interpolate
    >>> from sympy.abc import x

    A list is interpreted as though it were paired with a range starting
    from 1:

    >>> interpolate([1, 4, 9, 16], x)
    x**2

    This can be made explicit by giving a list of coordinates:

    >>> interpolate([(1, 1), (2, 4), (3, 9)], x)
    x**2

    The (x, y) coordinates can also be given as keys and values of a
    dictionary (and the points need not be equispaced):

    >>> interpolate([(-1, 2), (1, 2), (2, 5)], x)
    x**2 + 1
    >>> interpolate({-1: 2, 1: 2, 2: 5}, x)
    x**2 + 1

    """
    n = len(data)
    poly = None

    if isinstance(data, dict):
        X, Y = list(zip(*data.items()))
        poly = interpolating_poly(n, x, X, Y)
    else:
        if isinstance(data[0], tuple):
            X, Y = list(zip(*data))
            poly = interpolating_poly(n, x, X, Y)
        else:
            Y = list(data)

            numert = Mul(*[(x - i) for i in range(1, n + 1)])
            denom = -factorial(n - 1) if n%2 == 0 else factorial(n - 1)
            coeffs = []
            for i in range(1, n + 1):
                coeffs.append(numert/(x - i)/denom)
                denom = denom/(i - n)*i

            poly = Add(*[coeff*y for coeff, y in zip(coeffs, Y)])

    return poly.expand()
```
### 2 - sympy/polys/specialpolys.py:

Start line: 143, End line: 160

```python
@public
def interpolating_poly(n, x, X='x', Y='y'):
    """Construct Lagrange interpolating polynomial for ``n`` data points. """
    if isinstance(X, string_types):
        X = symbols("%s:%s" % (X, n))

    if isinstance(Y, string_types):
        Y = symbols("%s:%s" % (Y, n))

    coeffs = []
    numert = Mul(*[(x - u) for u in X])

    for i in range(n):
        numer = numert/(x - X[i])
        denom = Mul(*[(X[i] - X[j]) for j in range(n) if i != j])
        coeffs.append(numer/denom)

    return Add(*[coeff*y for coeff, y in zip(coeffs, Y)])
```
### 3 - sympy/functions/special/bsplines.py:

Start line: 285, End line: 299

```python
def interpolating_spline(d, x, X, Y):
    # ... other code
    ival = [e.atoms(Number) for e in intervals]
    ival = [list(sorted(e))[0] for e in ival]
    com = zip(ival, intervals)
    com = sorted(com, key=lambda x: x[0])
    intervals = [y for x, y in com]

    basis_dicts = [dict((c, e) for (e, c) in b.args) for b in basis]
    spline = []
    for i in intervals:
        piece = sum(
            [c * d.get(i, S.Zero) for (c, d) in zip(coeff, basis_dicts)], S.Zero
        )
        spline.append((piece, i))
    return Piecewise(*spline)
```
### 4 - sympy/polys/polyfuncs.py:

Start line: 261, End line: 318

```python
@public
def rational_interpolate(data, degnum, X=symbols('x')):
    """
    Returns a rational interpolation, where the data points are element of
    any integral domain.

    The first argument  contains the data (as a list of coordinates). The
    ``degnum`` argument is the degree in the numerator of the rational
    function. Setting it too high will decrease the maximal degree in the
    denominator for the same amount of data.

    Examples
    ========

    >>> from sympy.polys.polyfuncs import rational_interpolate

    >>> data = [(1, -210), (2, -35), (3, 105), (4, 231), (5, 350), (6, 465)]
    >>> rational_interpolate(data, 2)
    (105*x**2 - 525)/(x + 1)

    Values do not need to be integers:

    >>> from sympy import sympify
    >>> x = [1, 2, 3, 4, 5, 6]
    >>> y = sympify("[-1, 0, 2, 22/5, 7, 68/7]")
    >>> rational_interpolate(zip(x, y), 2)
    (3*x**2 - 7*x + 2)/(x + 1)

    The symbol for the variable can be changed if needed:
    >>> from sympy import symbols
    >>> z = symbols('z')
    >>> rational_interpolate(data, 2, X=z)
    (105*z**2 - 525)/(z + 1)

    References
    ==========

    .. [1] Algorithm is adapted from:
           http://axiom-wiki.newsynthesis.org/RationalInterpolation

    """
    from sympy.matrices.dense import ones

    xdata, ydata = list(zip(*data))

    k = len(xdata) - degnum - 1
    if k < 0:
        raise OptionError("Too few values for the required degree.")
    c = ones(degnum + k + 1, degnum + k + 2)
    for j in range(max(degnum, k)):
        for i in range(degnum + k + 1):
            c[i, j + 1] = c[i, j]*xdata[i]
    for j in range(k + 1):
        for i in range(degnum + k + 1):
            c[i, degnum + k + 1 - j] = -c[i, k - j]*ydata[i]
    r = c.nullspace()[0]
    return (sum(r[i] * X**i for i in range(degnum + 1))
            / sum(r[i + degnum + 1] * X**i for i in range(k + 1)))
```
### 5 - sympy/plotting/pygletplot/plot_interval.py:

Start line: 167, End line: 183

```python
class PlotInterval(object):

    @require_all_args
    def vrange2(self):
        """
        Yields v_steps pairs of sympy numbers ranging from
        (v_min, v_min + step) to (v_max - step, v_max).
        """
        d = (self.v_max - self.v_min) / self.v_steps
        a = self.v_min + (d * Integer(0))
        for i in range(self.v_steps):
            b = self.v_min + (d * Integer(i + 1))
            yield a, b
            a = b

    def frange(self):
        for i in self.vrange():
            yield float(i.evalf())
```
### 6 - sympy/plotting/intervalmath/lib_interval.py:

Start line: 310, End line: 328

```python
def ceil(x):
    """Evaluates the ceiling of an interval"""
    np = import_module('numpy')
    if isinstance(x, (int, float)):
        return interval(np.ceil(x))
    elif isinstance(x, interval):
        if x.is_valid is False:
            return interval(-np.inf, np.inf, is_valid=False)
        else:
            start = np.ceil(x.start)
            end = np.ceil(x.end)
            #Continuous over the interval
            if start == end:
                return interval(start, end, is_valid=x.is_valid)
            else:
                #Not continuous over the interval
                return interval(start, end, is_valid=None)
    else:
        return NotImplementedError
```
### 7 - sympy/core/expr.py:

Start line: 3682, End line: 3757

```python
class Expr(Basic, EvalfMixin):

    def round(self, n=None):
        # mpmath shows us that the first 18 digits are
        #     >>> Float(.575).n(18)
        #     0.574999999999999956
        # The default precision is 15 digits and if we ask
        # for 15 we get
        #     >>> Float(.575).n(15)
        #     0.575000000000000
        # mpmath handles rounding at the 15th digit. But we
        # need to be careful since the user might be asking
        # for rounding at the last digit and our semantics
        # are to round toward the even final digit when there
        # is a tie. So the extra digit will be used to make
        # that decision. In this case, the value is the same
        # to 15 digits:
        #     >>> Float(.575).n(16)
        #     0.5750000000000000
        # Now converting this to the 15 known digits gives
        #     575000000000000.0
        # which rounds to integer
        #    5750000000000000
        # And now we can round to the desired digt, e.g. at
        # the second from the left and we get
        #    5800000000000000
        # and rescaling that gives
        #    0.58
        # as the final result.
        # If the value is made slightly less than 0.575 we might
        # still obtain the same value:
        #    >>> Float(.575-1e-16).n(16)*10**15
        #    574999999999999.8
        # What 15 digits best represents the known digits (which are
        # to the left of the decimal? 5750000000000000, the same as
        # before. The only way we will round down (in this case) is
        # if we declared that we had more than 15 digits of precision.
        # For example, if we use 16 digits of precision, the integer
        # we deal with is
        #    >>> Float(.575-1e-16).n(17)*10**16
        #    5749999999999998.4
        # and this now rounds to 5749999999999998 and (if we round to
        # the 2nd digit from the left) we get 5700000000000000.
        #
        xf = x.n(dps + extra)*Pow(10, shift)
        xi = Integer(xf)
        # use the last digit to select the value of xi
        # nearest to x before rounding at the desired digit
        sign = 1 if x > 0 else -1
        dif2 = sign*(xf - xi).n(extra)
        if dif2 < 0:
            raise NotImplementedError(
                'not expecting int(x) to round away from 0')
        if dif2 > .5:
            xi += sign  # round away from 0
        elif dif2 == .5:
            xi += sign if xi%2 else -sign  # round toward even
        # shift p to the new position
        ip = p - shift
        # let Python handle the int rounding then rescale
        xr = xi.round(ip) # when Py2 is drop make this round(xi.p, ip)
        # restore scale
        rv = Rational(xr, Pow(10, shift))
        # return Float or Integer
        if rv.is_Integer:
            if n is None:  # the single-arg case
                return rv
            # use str or else it won't be a float
            return Float(str(rv), dps)  # keep same precision
        else:
            if not allow and rv > self:
                allow += 1
            return Float(rv, allow)

    __round__ = round

    def _eval_derivative_matrix_lines(self, x):
        from sympy.matrices.expressions.matexpr import _LeftRightArgs
        return [_LeftRightArgs([S.One, S.One], higher=self._eval_derivative(x))]
```
### 8 - sympy/plotting/intervalmath/lib_interval.py:

Start line: 331, End line: 349

```python
def floor(x):
    """Evaluates the floor of an interval"""
    np = import_module('numpy')
    if isinstance(x, (int, float)):
        return interval(np.floor(x))
    elif isinstance(x, interval):
        if x.is_valid is False:
            return interval(-np.inf, np.inf, is_valid=False)
        else:
            start = np.floor(x.start)
            end = np.floor(x.end)
            #continuous over the argument
            if start == end:
                return interval(start, end, is_valid=x.is_valid)
            else:
                #not continuous over the interval
                return interval(start, end, is_valid=None)
    else:
        return NotImplementedError
```
### 9 - sympy/functions/special/bsplines.py:

Start line: 216, End line: 284

```python
def interpolating_spline(d, x, X, Y):
    """
    Return spline of degree *d*, passing through the given *X*
    and *Y* values.

    Explanation
    ===========

    This function returns a piecewise function such that each part is
    a polynomial of degree not greater than *d*. The value of *d*
    must be 1 or greater and the values of *X* must be strictly
    increasing.

    Examples
    ========

    >>> from sympy import interpolating_spline
    >>> from sympy.abc import x
    >>> interpolating_spline(1, x, [1, 2, 4, 7], [3, 6, 5, 7])
    Piecewise((3*x, (x >= 1) & (x <= 2)),
            (7 - x/2, (x >= 2) & (x <= 4)),
            (2*x/3 + 7/3, (x >= 4) & (x <= 7)))
    >>> interpolating_spline(3, x, [-2, 0, 1, 3, 4], [4, 2, 1, 1, 3])
    Piecewise((7*x**3/117 + 7*x**2/117 - 131*x/117 + 2, (x >= -2) & (x <= 1)),
            (10*x**3/117 - 2*x**2/117 - 122*x/117 + 77/39, (x >= 1) & (x <= 4)))

    See Also
    ========

    bspline_basis_set, interpolating_poly

    """
    from sympy import symbols, Number, Dummy, Rational
    from sympy.solvers.solveset import linsolve
    from sympy.matrices.dense import Matrix

    # Input sanitization
    d = sympify(d)
    if not (d.is_Integer and d.is_positive):
        raise ValueError("Spline degree must be a positive integer, not %s." % d)
    if len(X) != len(Y):
        raise ValueError("Number of X and Y coordinates must be the same.")
    if len(X) < d + 1:
        raise ValueError("Degree must be less than the number of control points.")
    if not all(a < b for a, b in zip(X, X[1:])):
        raise ValueError("The x-coordinates must be strictly increasing.")

    # Evaluating knots value
    if d.is_odd:
        j = (d + 1) // 2
        interior_knots = X[j:-j]
    else:
        j = d // 2
        interior_knots = [
            Rational(a + b, 2) for a, b in zip(X[j : -j - 1], X[j + 1 : -j])
        ]

    knots = [X[0]] * (d + 1) + list(interior_knots) + [X[-1]] * (d + 1)

    basis = bspline_basis_set(d, knots, x)

    A = [[b.subs(x, v) for b in basis] for v in X]

    coeff = linsolve((Matrix(A), Matrix(Y)), symbols("c0:{}".format(len(X)), cls=Dummy))
    coeff = list(coeff)[0]
    intervals = set([c for b in basis for (e, c) in b.args if c != True])

    # Sorting the intervals
    #  ival contains the end-points of each interval
    # ... other code
```
### 10 - sympy/plotting/intervalmath/interval_arithmetic.py:

Start line: 372, End line: 417

```python
def _pow_float(inter, power):
    """Evaluates an interval raised to a floating point."""
    power_rational = nsimplify(power)
    num, denom = power_rational.as_numer_denom()
    if num % 2 == 0:
        start = abs(inter.start)**power
        end = abs(inter.end)**power
        if start < 0:
            ret = interval(0, max(start, end))
        else:
            ret = interval(start, end)
        return ret
    elif denom % 2 == 0:
        if inter.end < 0:
            return interval(-float('inf'), float('inf'), is_valid=False)
        elif inter.start < 0:
            return interval(0, inter.end**power, is_valid=None)
        else:
            return interval(inter.start**power, inter.end**power)
    else:
        if inter.start < 0:
            start = -abs(inter.start)**power
        else:
            start = inter.start**power

        if inter.end < 0:
            end = -abs(inter.end)**power
        else:
            end = inter.end**power

        return interval(start, end, is_valid=inter.is_valid)


def _pow_int(inter, power):
    """Evaluates an interval raised to an integer power"""
    power = int(power)
    if power & 1:
        return interval(inter.start**power, inter.end**power)
    else:
        if inter.start < 0 and inter.end > 0:
            start = 0
            end = max(inter.start**power, inter.end**power)
            return interval(start, end)
        else:
            return interval(inter.start**power, inter.end**power)
```
