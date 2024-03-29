# sympy__sympy-17696

| **sympy/sympy** | `fed3bb83dec834bd75fd8bcd68fc0c31387f394a` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 167 |
| **Any found context length** | 167 |
| **Avg pos** | 2.0 |
| **Min pos** | 1 |
| **Max pos** | 1 |
| **Top file pos** | 1 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sympy/assumptions/refine.py b/sympy/assumptions/refine.py
--- a/sympy/assumptions/refine.py
+++ b/sympy/assumptions/refine.py
@@ -291,6 +291,47 @@ def _refine_reim(expr, assumptions):
     return None
 
 
+def refine_sign(expr, assumptions):
+    """
+    Handler for sign
+
+    Examples
+    ========
+
+    >>> from sympy.assumptions.refine import refine_sign
+    >>> from sympy import Symbol, Q, sign, im
+    >>> x = Symbol('x', real = True)
+    >>> expr = sign(x)
+    >>> refine_sign(expr, Q.positive(x) & Q.nonzero(x))
+    1
+    >>> refine_sign(expr, Q.negative(x) & Q.nonzero(x))
+    -1
+    >>> refine_sign(expr, Q.zero(x))
+    0
+    >>> y = Symbol('y', imaginary = True)
+    >>> expr = sign(y)
+    >>> refine_sign(expr, Q.positive(im(y)))
+    I
+    >>> refine_sign(expr, Q.negative(im(y)))
+    -I
+    """
+    arg = expr.args[0]
+    if ask(Q.zero(arg), assumptions):
+        return S.Zero
+    if ask(Q.real(arg)):
+        if ask(Q.positive(arg), assumptions):
+            return S.One
+        if ask(Q.negative(arg), assumptions):
+            return S.NegativeOne
+    if ask(Q.imaginary(arg)):
+        arg_re, arg_im = arg.as_real_imag()
+        if ask(Q.positive(arg_im), assumptions):
+            return S.ImaginaryUnit
+        if ask(Q.negative(arg_im), assumptions):
+            return -S.ImaginaryUnit
+    return expr
+
+
 handlers_dict = {
     'Abs': refine_abs,
     'Pow': refine_Pow,
@@ -302,5 +343,6 @@ def _refine_reim(expr, assumptions):
     'StrictGreaterThan': refine_Relational,
     'StrictLessThan': refine_Relational,
     're': refine_re,
-    'im': refine_im
+    'im': refine_im,
+    'sign': refine_sign
 }

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sympy/assumptions/refine.py | 294 | 294 | 1 | 1 | 167
| sympy/assumptions/refine.py | 305 | 305 | 1 | 1 | 167


## Problem Statement

```
Refine with sign
Consider the following code:
\`\`\`
from sympy import *
x = Symbol('x', real = True)

expr = sign(x)
expr2 = refine(expr, Q.positive(x))
expr3 = refine(expr, Q.positive(x) & Q.nonzero(x))
expr4 = refine(expr, Q.positive(x + 1))
\`\`\`
All the returned expression are `sign(x)`. However, at least for `expr3` and `expr4`, the results should be `1`. This probably is due to the lack of capabilities for `refine`. A PR similar to #17019 should fix this behaviour. 

Related issues: #8326 and #17052.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 sympy/assumptions/refine.py** | 283 | 307| 167 | 167 | 2397 | 
| 2 | **1 sympy/assumptions/refine.py** | 230 | 260| 216 | 383 | 2397 | 
| 3 | **1 sympy/assumptions/refine.py** | 263 | 280| 138 | 521 | 2397 | 
| 4 | **1 sympy/assumptions/refine.py** | 1 | 44| 319 | 840 | 2397 | 
| 5 | 2 sympy/simplify/simplify.py | 331 | 390| 504 | 1344 | 19745 | 
| 6 | **2 sympy/assumptions/refine.py** | 47 | 83| 270 | 1614 | 19745 | 
| 7 | 3 sympy/core/expr.py | 278 | 313| 500 | 2114 | 53598 | 
| 8 | 4 sympy/core/exprtools.py | 110 | 173| 567 | 2681 | 66040 | 
| 9 | 5 sympy/series/gruntz.py | 354 | 408| 399 | 3080 | 72405 | 
| 10 | 5 sympy/core/exprtools.py | 174 | 211| 363 | 3443 | 72405 | 
| 11 | **5 sympy/assumptions/refine.py** | 86 | 181| 820 | 4263 | 72405 | 
| 12 | 5 sympy/core/expr.py | 876 | 929| 418 | 4681 | 72405 | 
| 13 | 6 sympy/matrices/expressions/determinant.py | 64 | 85| 143 | 4824 | 72893 | 
| 14 | 7 sympy/matrices/expressions/matmul.py | 371 | 406| 229 | 5053 | 76098 | 
| 15 | 7 sympy/core/expr.py | 2436 | 2480| 433 | 5486 | 76098 | 
| 16 | 8 sympy/functions/elementary/complexes.py | 358 | 397| 316 | 5802 | 85470 | 
| 17 | 9 sympy/integrals/transforms.py | 873 | 914| 359 | 6161 | 102232 | 
| 18 | 9 sympy/simplify/simplify.py | 1469 | 1502| 347 | 6508 | 102232 | 
| 19 | 10 sympy/codegen/rewriting.py | 234 | 257| 197 | 6705 | 104321 | 
| 20 | 10 sympy/functions/elementary/complexes.py | 294 | 339| 387 | 7092 | 104321 | 
| 21 | 10 sympy/core/expr.py | 646 | 741| 806 | 7898 | 104321 | 
| 22 | 10 sympy/core/expr.py | 931 | 964| 304 | 8202 | 104321 | 
| 23 | 11 sympy/core/mul.py | 1395 | 1428| 250 | 8452 | 120143 | 
| 24 | 12 sympy/polys/subresultants_qq_zz.py | 1704 | 1765| 548 | 9000 | 146437 | 
| 25 | 12 sympy/functions/elementary/complexes.py | 242 | 292| 302 | 9302 | 146437 | 
| 26 | 12 sympy/core/expr.py | 817 | 874| 547 | 9849 | 146437 | 
| 27 | 12 sympy/integrals/transforms.py | 938 | 963| 231 | 10080 | 146437 | 
| 28 | 12 sympy/core/exprtools.py | 1208 | 1258| 486 | 10566 | 146437 | 
| 29 | 13 sympy/matrices/expressions/inverse.py | 79 | 103| 175 | 10741 | 147136 | 
| 30 | **13 sympy/assumptions/refine.py** | 184 | 227| 465 | 11206 | 147136 | 
| 31 | 13 sympy/simplify/simplify.py | 1295 | 1322| 247 | 11453 | 147136 | 
| 32 | 13 sympy/core/expr.py | 2857 | 2912| 582 | 12035 | 147136 | 
| 33 | 14 sympy/solvers/solvers.py | 456 | 909| 4473 | 16508 | 179669 | 
| 34 | 14 sympy/functions/elementary/complexes.py | 341 | 356| 169 | 16677 | 179669 | 
| 35 | 14 sympy/core/expr.py | 332 | 353| 226 | 16903 | 179669 | 
| 36 | 15 sympy/simplify/trigsimp.py | 539 | 590| 492 | 17395 | 191897 | 
| 37 | 15 sympy/core/expr.py | 144 | 206| 536 | 17931 | 191897 | 
| 38 | 15 sympy/simplify/simplify.py | 1279 | 1293| 181 | 18112 | 191897 | 
| 39 | 15 sympy/core/expr.py | 378 | 400| 222 | 18334 | 191897 | 
| 40 | 15 sympy/core/expr.py | 3754 | 3830| 925 | 19259 | 191897 | 


### Hint

```
I would like to work on this issue.
Can someone guide me on exactly what has to be done?
@kmm555 you can write a function in `refine.py` similar to `refine_abs()`, which returns `0` if the argument is equal to `0`, `1` if positive and so on (see the possible output of `sign` in `complexes.py`
```

## Patch

```diff
diff --git a/sympy/assumptions/refine.py b/sympy/assumptions/refine.py
--- a/sympy/assumptions/refine.py
+++ b/sympy/assumptions/refine.py
@@ -291,6 +291,47 @@ def _refine_reim(expr, assumptions):
     return None
 
 
+def refine_sign(expr, assumptions):
+    """
+    Handler for sign
+
+    Examples
+    ========
+
+    >>> from sympy.assumptions.refine import refine_sign
+    >>> from sympy import Symbol, Q, sign, im
+    >>> x = Symbol('x', real = True)
+    >>> expr = sign(x)
+    >>> refine_sign(expr, Q.positive(x) & Q.nonzero(x))
+    1
+    >>> refine_sign(expr, Q.negative(x) & Q.nonzero(x))
+    -1
+    >>> refine_sign(expr, Q.zero(x))
+    0
+    >>> y = Symbol('y', imaginary = True)
+    >>> expr = sign(y)
+    >>> refine_sign(expr, Q.positive(im(y)))
+    I
+    >>> refine_sign(expr, Q.negative(im(y)))
+    -I
+    """
+    arg = expr.args[0]
+    if ask(Q.zero(arg), assumptions):
+        return S.Zero
+    if ask(Q.real(arg)):
+        if ask(Q.positive(arg), assumptions):
+            return S.One
+        if ask(Q.negative(arg), assumptions):
+            return S.NegativeOne
+    if ask(Q.imaginary(arg)):
+        arg_re, arg_im = arg.as_real_imag()
+        if ask(Q.positive(arg_im), assumptions):
+            return S.ImaginaryUnit
+        if ask(Q.negative(arg_im), assumptions):
+            return -S.ImaginaryUnit
+    return expr
+
+
 handlers_dict = {
     'Abs': refine_abs,
     'Pow': refine_Pow,
@@ -302,5 +343,6 @@ def _refine_reim(expr, assumptions):
     'StrictGreaterThan': refine_Relational,
     'StrictLessThan': refine_Relational,
     're': refine_re,
-    'im': refine_im
+    'im': refine_im,
+    'sign': refine_sign
 }

```

## Test Patch

```diff
diff --git a/sympy/assumptions/tests/test_refine.py b/sympy/assumptions/tests/test_refine.py
--- a/sympy/assumptions/tests/test_refine.py
+++ b/sympy/assumptions/tests/test_refine.py
@@ -1,8 +1,10 @@
 from sympy import (Abs, exp, Expr, I, pi, Q, Rational, refine, S, sqrt,
-                   atan, atan2, nan, Symbol, re, im)
+                   atan, atan2, nan, Symbol, re, im, sign)
 from sympy.abc import w, x, y, z
 from sympy.core.relational import Eq, Ne
 from sympy.functions.elementary.piecewise import Piecewise
+from sympy.utilities.pytest import slow
+from sympy.core import S
 
 
 def test_Abs():
@@ -170,6 +172,23 @@ def test_complex():
         & Q.real(z)) == w*z + x*y
 
 
+def test_sign():
+    x = Symbol('x', real = True)
+    assert refine(sign(x), Q.positive(x)) == 1
+    assert refine(sign(x), Q.negative(x)) == -1
+    assert refine(sign(x), Q.zero(x)) == 0
+    assert refine(sign(x), True) == sign(x)
+    assert refine(sign(Abs(x)), Q.nonzero(x)) == 1
+
+    x = Symbol('x', imaginary=True)
+    assert refine(sign(x), Q.positive(im(x))) == S.ImaginaryUnit
+    assert refine(sign(x), Q.negative(im(x))) == -S.ImaginaryUnit
+    assert refine(sign(x), True) == sign(x)
+
+    x = Symbol('x', complex=True)
+    assert refine(sign(x), Q.zero(x)) == 0
+
+
 def test_func_args():
     class MyClass(Expr):
         # A class with nontrivial .func

```


## Code snippets

### 1 - sympy/assumptions/refine.py:

Start line: 283, End line: 307

```python
def _refine_reim(expr, assumptions):
    # Helper function for refine_re & refine_im
    expanded = expr.expand(complex = True)
    if expanded != expr:
        refined = refine(expanded, assumptions)
        if refined != expanded:
            return refined
    # Best to leave the expression as is
    return None


handlers_dict = {
    'Abs': refine_abs,
    'Pow': refine_Pow,
    'atan2': refine_atan2,
    'Equality': refine_Relational,
    'Unequality': refine_Relational,
    'GreaterThan': refine_Relational,
    'LessThan': refine_Relational,
    'StrictGreaterThan': refine_Relational,
    'StrictLessThan': refine_Relational,
    're': refine_re,
    'im': refine_im
}
```
### 2 - sympy/assumptions/refine.py:

Start line: 230, End line: 260

```python
def refine_Relational(expr, assumptions):
    """
    Handler for Relational

    >>> from sympy.assumptions.refine import refine_Relational
    >>> from sympy.assumptions.ask import Q
    >>> from sympy.abc import x
    >>> refine_Relational(x<0, ~Q.is_true(x<0))
    False
    """
    return ask(Q.is_true(expr), assumptions)


def refine_re(expr, assumptions):
    """
    Handler for real part.

    >>> from sympy.assumptions.refine import refine_re
    >>> from sympy import Q, re
    >>> from sympy.abc import x
    >>> refine_re(re(x), Q.real(x))
    x
    >>> refine_re(re(x), Q.imaginary(x))
    0
    """
    arg = expr.args[0]
    if ask(Q.real(arg), assumptions):
        return arg
    if ask(Q.imaginary(arg), assumptions):
        return S.Zero
    return _refine_reim(expr, assumptions)
```
### 3 - sympy/assumptions/refine.py:

Start line: 263, End line: 280

```python
def refine_im(expr, assumptions):
    """
    Handler for imaginary part.

    >>> from sympy.assumptions.refine import refine_im
    >>> from sympy import Q, im
    >>> from sympy.abc import x
    >>> refine_im(im(x), Q.real(x))
    0
    >>> refine_im(im(x), Q.imaginary(x))
    -I*x
    """
    arg = expr.args[0]
    if ask(Q.real(arg), assumptions):
        return S.Zero
    if ask(Q.imaginary(arg), assumptions):
        return - S.ImaginaryUnit * arg
    return _refine_reim(expr, assumptions)
```
### 4 - sympy/assumptions/refine.py:

Start line: 1, End line: 44

```python
from __future__ import print_function, division

from sympy.core import S, Add, Expr, Basic, Mul
from sympy.assumptions import Q, ask

def refine(expr, assumptions=True):
    """
    Simplify an expression using assumptions.

    Gives the form of expr that would be obtained if symbols
    in it were replaced by explicit numerical expressions satisfying
    the assumptions.

    Examples
    ========

        >>> from sympy import refine, sqrt, Q
        >>> from sympy.abc import x
        >>> refine(sqrt(x**2), Q.real(x))
        Abs(x)
        >>> refine(sqrt(x**2), Q.positive(x))
        x

    """
    if not isinstance(expr, Basic):
        return expr
    if not expr.is_Atom:
        args = [refine(arg, assumptions) for arg in expr.args]
        # TODO: this will probably not work with Integral or Polynomial
        expr = expr.func(*args)
    if hasattr(expr, '_eval_refine'):
        ref_expr = expr._eval_refine(assumptions)
        if ref_expr is not None:
            return ref_expr
    name = expr.__class__.__name__
    handler = handlers_dict.get(name, None)
    if handler is None:
        return expr
    new_expr = handler(expr, assumptions)
    if (new_expr is None) or (expr == new_expr):
        return expr
    if not isinstance(new_expr, Expr):
        return new_expr
    return refine(new_expr, assumptions)
```
### 5 - sympy/simplify/simplify.py:

Start line: 331, End line: 390

```python
def signsimp(expr, evaluate=None):
    """Make all Add sub-expressions canonical wrt sign.

    If an Add subexpression, ``a``, can have a sign extracted,
    as determined by could_extract_minus_sign, it is replaced
    with Mul(-1, a, evaluate=False). This allows signs to be
    extracted from powers and products.

    Examples
    ========

    >>> from sympy import signsimp, exp, symbols
    >>> from sympy.abc import x, y
    >>> i = symbols('i', odd=True)
    >>> n = -1 + 1/x
    >>> n/x/(-n)**2 - 1/n/x
    (-1 + 1/x)/(x*(1 - 1/x)**2) - 1/(x*(-1 + 1/x))
    >>> signsimp(_)
    0
    >>> x*n + x*-n
    x*(-1 + 1/x) + x*(1 - 1/x)
    >>> signsimp(_)
    0

    Since powers automatically handle leading signs

    >>> (-2)**i
    -2**i

    signsimp can be used to put the base of a power with an integer
    exponent into canonical form:

    >>> n**i
    (-1 + 1/x)**i

    By default, signsimp doesn't leave behind any hollow simplification:
    if making an Add canonical wrt sign didn't change the expression, the
    original Add is restored. If this is not desired then the keyword
    ``evaluate`` can be set to False:

    >>> e = exp(y - x)
    >>> signsimp(e) == e
    True
    >>> signsimp(e, evaluate=False)
    exp(-(x - y))

    """
    if evaluate is None:
        evaluate = global_evaluate[0]
    expr = sympify(expr)
    if not isinstance(expr, Expr) or expr.is_Atom:
        return expr
    e = sub_post(sub_pre(expr))
    if not isinstance(e, Expr) or e.is_Atom:
        return e
    if e.is_Add:
        return e.func(*[signsimp(a, evaluate) for a in e.args])
    if evaluate:
        e = e.xreplace({m: -(-m) for m in e.atoms(Mul) if -(-m) != m})
    return e
```
### 6 - sympy/assumptions/refine.py:

Start line: 47, End line: 83

```python
def refine_abs(expr, assumptions):
    """
    Handler for the absolute value.

    Examples
    ========

    >>> from sympy import Symbol, Q, refine, Abs
    >>> from sympy.assumptions.refine import refine_abs
    >>> from sympy.abc import x
    >>> refine_abs(Abs(x), Q.real(x))
    >>> refine_abs(Abs(x), Q.positive(x))
    x
    >>> refine_abs(Abs(x), Q.negative(x))
    -x

    """
    from sympy.core.logic import fuzzy_not
    from sympy import Abs
    arg = expr.args[0]
    if ask(Q.real(arg), assumptions) and \
            fuzzy_not(ask(Q.negative(arg), assumptions)):
        # if it's nonnegative
        return arg
    if ask(Q.negative(arg), assumptions):
        return -arg
    # arg is Mul
    if isinstance(arg, Mul):
        r = [refine(abs(a), assumptions) for a in arg.args]
        non_abs = []
        in_abs = []
        for i in r:
            if isinstance(i, Abs):
                in_abs.append(i.args[0])
            else:
                non_abs.append(i)
        return Mul(*non_abs) * Abs(Mul(*in_abs))
```
### 7 - sympy/core/expr.py:

Start line: 278, End line: 313

```python
class Expr(Basic, EvalfMixin):

    def __int__(self):
        # Although we only need to round to the units position, we'll
        # get one more digit so the extra testing below can be avoided
        # unless the rounded value rounded to an integer, e.g. if an
        # expression were equal to 1.9 and we rounded to the unit position
        # we would get a 2 and would not know if this rounded up or not
        # without doing a test (as done below). But if we keep an extra
        # digit we know that 1.9 is not the same as 1 and there is no
        # need for further testing: our int value is correct. If the value
        # were 1.99, however, this would round to 2.0 and our int value is
        # off by one. So...if our round value is the same as the int value
        # (regardless of how much extra work we do to calculate extra decimal
        # places) we need to test whether we are off by one.
        from sympy import Dummy
        if not self.is_number:
            raise TypeError("can't convert symbols to int")
        r = self.round(2)
        if not r.is_Number:
            raise TypeError("can't convert complex to int")
        if r in (S.NaN, S.Infinity, S.NegativeInfinity):
            raise TypeError("can't convert %s to int" % r)
        i = int(r)
        if not i:
            return 0
        # off-by-one check
        if i == r and not (self - i).equals(0):
            isign = 1 if i > 0 else -1
            x = Dummy()
            # in the following (self - i).evalf(2) will not always work while
            # (self - r).evalf(2) and the use of subs does; if the test that
            # was added when this comment was added passes, it might be safe
            # to simply use sign to compute this rather than doing this by hand:
            diff_sign = 1 if (self - x).evalf(2, subs={x: i}) > 0 else -1
            if diff_sign != isign:
                i -= isign
        return i
```
### 8 - sympy/core/exprtools.py:

Start line: 110, End line: 173

```python
def _monotonic_sign(self):
    # ... other code
    if len(free) == 1:
        if self.is_polynomial():
            from sympy.polys.polytools import real_roots
            from sympy.polys.polyroots import roots
            from sympy.polys.polyerrors import PolynomialError
            x = free.pop()
            x0 = _monotonic_sign(x)
            if x0 == _eps or x0 == -_eps:
                x0 = S.Zero
            if x0 is not None:
                d = self.diff(x)
                if d.is_number:
                    currentroots = []
                else:
                    try:
                        currentroots = real_roots(d)
                    except (PolynomialError, NotImplementedError):
                        currentroots = [r for r in roots(d, x) if r.is_extended_real]
                y = self.subs(x, x0)
                if x.is_nonnegative and all(r <= x0 for r in currentroots):
                    if y.is_nonnegative and d.is_positive:
                        if y:
                            return y if y.is_positive else Dummy('pos', positive=True)
                        else:
                            return Dummy('nneg', nonnegative=True)
                    if y.is_nonpositive and d.is_negative:
                        if y:
                            return y if y.is_negative else Dummy('neg', negative=True)
                        else:
                            return Dummy('npos', nonpositive=True)
                elif x.is_nonpositive and all(r >= x0 for r in currentroots):
                    if y.is_nonnegative and d.is_negative:
                        if y:
                            return Dummy('pos', positive=True)
                        else:
                            return Dummy('nneg', nonnegative=True)
                    if y.is_nonpositive and d.is_positive:
                        if y:
                            return Dummy('neg', negative=True)
                        else:
                            return Dummy('npos', nonpositive=True)
        else:
            n, d = self.as_numer_denom()
            den = None
            if n.is_number:
                den = _monotonic_sign(d)
            elif not d.is_number:
                if _monotonic_sign(n) is not None:
                    den = _monotonic_sign(d)
            if den is not None and (den.is_positive or den.is_negative):
                v = n*den
                if v.is_positive:
                    return Dummy('pos', positive=True)
                elif v.is_nonnegative:
                    return Dummy('nneg', nonnegative=True)
                elif v.is_negative:
                    return Dummy('neg', negative=True)
                elif v.is_nonpositive:
                    return Dummy('npos', nonpositive=True)
        return None

    # multivariate
    c, a = self.as_coeff_Add()
    v = None
    # ... other code
```
### 9 - sympy/series/gruntz.py:

Start line: 354, End line: 408

```python
@debug
@cacheit
@timeit
def sign(e, x):
    """
    Returns a sign of an expression e(x) for x->oo.

    ::

        e >  0 for x sufficiently large ...  1
        e == 0 for x sufficiently large ...  0
        e <  0 for x sufficiently large ... -1

    The result of this function is currently undefined if e changes sign
    arbitrarily often for arbitrarily large x (e.g. sin(x)).

    Note that this returns zero only if e is *constantly* zero
    for x sufficiently large. [If e is constant, of course, this is just
    the same thing as the sign of e.]
    """
    from sympy import sign as _sign
    if not isinstance(e, Basic):
        raise TypeError("e should be an instance of Basic")

    if e.is_positive:
        return 1
    elif e.is_negative:
        return -1
    elif e.is_zero:
        return 0

    elif not e.has(x):
        return _sign(e)
    elif e == x:
        return 1
    elif e.is_Mul:
        a, b = e.as_two_terms()
        sa = sign(a, x)
        if not sa:
            return 0
        return sa * sign(b, x)
    elif isinstance(e, exp):
        return 1
    elif e.is_Pow:
        s = sign(e.base, x)
        if s == 1:
            return 1
        if e.exp.is_Integer:
            return s**e.exp
    elif isinstance(e, log):
        return sign(e.args[0] - 1, x)

    # if all else fails, do it the hard way
    c0, e0 = mrv_leadterm(e, x)
    return sign(c0, x)
```
### 10 - sympy/core/exprtools.py:

Start line: 174, End line: 211

```python
def _monotonic_sign(self):
    # ... other code
    if not a.is_polynomial():
        # F/A or A/F where A is a number and F is a signed, rational monomial
        n, d = a.as_numer_denom()
        if not (n.is_number or d.is_number):
            return
        if (
                a.is_Mul or a.is_Pow) and \
                a.is_rational and \
                all(p.exp.is_Integer for p in a.atoms(Pow) if p.is_Pow) and \
                (a.is_positive or a.is_negative):
            v = S.One
            for ai in Mul.make_args(a):
                if ai.is_number:
                    v *= ai
                    continue
                reps = {}
                for x in ai.free_symbols:
                    reps[x] = _monotonic_sign(x)
                    if reps[x] is None:
                        return
                v *= ai.subs(reps)
    elif c:
        # signed linear expression
        if not any(p for p in a.atoms(Pow) if not p.is_number) and (a.is_nonpositive or a.is_nonnegative):
            free = list(a.free_symbols)
            p = {}
            for i in free:
                v = _monotonic_sign(i)
                if v is None:
                    return
                p[i] = v or (_eps if i.is_nonnegative else -_eps)
            v = a.xreplace(p)
    if v is not None:
        rv = v + c
        if v.is_nonnegative and rv.is_positive:
            return rv.subs(_eps, 0)
        if v.is_nonpositive and rv.is_negative:
            return rv.subs(_eps, 0)
```
### 11 - sympy/assumptions/refine.py:

Start line: 86, End line: 181

```python
def refine_Pow(expr, assumptions):
    """
    Handler for instances of Pow.

    >>> from sympy import Symbol, Q
    >>> from sympy.assumptions.refine import refine_Pow
    >>> from sympy.abc import x,y,z
    >>> refine_Pow((-1)**x, Q.real(x))
    >>> refine_Pow((-1)**x, Q.even(x))
    1
    >>> refine_Pow((-1)**x, Q.odd(x))
    -1

    For powers of -1, even parts of the exponent can be simplified:

    >>> refine_Pow((-1)**(x+y), Q.even(x))
    (-1)**y
    >>> refine_Pow((-1)**(x+y+z), Q.odd(x) & Q.odd(z))
    (-1)**y
    >>> refine_Pow((-1)**(x+y+2), Q.odd(x))
    (-1)**(y + 1)
    >>> refine_Pow((-1)**(x+3), True)
    (-1)**(x + 1)

    """
    from sympy.core import Pow, Rational
    from sympy.functions.elementary.complexes import Abs
    from sympy.functions import sign
    if isinstance(expr.base, Abs):
        if ask(Q.real(expr.base.args[0]), assumptions) and \
                ask(Q.even(expr.exp), assumptions):
            return expr.base.args[0] ** expr.exp
    if ask(Q.real(expr.base), assumptions):
        if expr.base.is_number:
            if ask(Q.even(expr.exp), assumptions):
                return abs(expr.base) ** expr.exp
            if ask(Q.odd(expr.exp), assumptions):
                return sign(expr.base) * abs(expr.base) ** expr.exp
        if isinstance(expr.exp, Rational):
            if type(expr.base) is Pow:
                return abs(expr.base.base) ** (expr.base.exp * expr.exp)

        if expr.base is S.NegativeOne:
            if expr.exp.is_Add:

                old = expr

                # For powers of (-1) we can remove
                #  - even terms
                #  - pairs of odd terms
                #  - a single odd term + 1
                #  - A numerical constant N can be replaced with mod(N,2)

                coeff, terms = expr.exp.as_coeff_add()
                terms = set(terms)
                even_terms = set([])
                odd_terms = set([])
                initial_number_of_terms = len(terms)

                for t in terms:
                    if ask(Q.even(t), assumptions):
                        even_terms.add(t)
                    elif ask(Q.odd(t), assumptions):
                        odd_terms.add(t)

                terms -= even_terms
                if len(odd_terms) % 2:
                    terms -= odd_terms
                    new_coeff = (coeff + S.One) % 2
                else:
                    terms -= odd_terms
                    new_coeff = coeff % 2

                if new_coeff != coeff or len(terms) < initial_number_of_terms:
                    terms.add(new_coeff)
                    expr = expr.base**(Add(*terms))

                # Handle (-1)**((-1)**n/2 + m/2)
                e2 = 2*expr.exp
                if ask(Q.even(e2), assumptions):
                    if e2.could_extract_minus_sign():
                        e2 *= expr.base
                if e2.is_Add:
                    i, p = e2.as_two_terms()
                    if p.is_Pow and p.base is S.NegativeOne:
                        if ask(Q.integer(p.exp), assumptions):
                            i = (i + 1)/2
                            if ask(Q.even(i), assumptions):
                                return expr.base**p.exp
                            elif ask(Q.odd(i), assumptions):
                                return expr.base**(p.exp + 1)
                            else:
                                return expr.base**(p.exp + i)

                if old != expr:
                    return expr
```
### 30 - sympy/assumptions/refine.py:

Start line: 184, End line: 227

```python
def refine_atan2(expr, assumptions):
    """
    Handler for the atan2 function

    Examples
    ========

    >>> from sympy import Symbol, Q, refine, atan2
    >>> from sympy.assumptions.refine import refine_atan2
    >>> from sympy.abc import x, y
    >>> refine_atan2(atan2(y,x), Q.real(y) & Q.positive(x))
    atan(y/x)
    >>> refine_atan2(atan2(y,x), Q.negative(y) & Q.negative(x))
    atan(y/x) - pi
    >>> refine_atan2(atan2(y,x), Q.positive(y) & Q.negative(x))
    atan(y/x) + pi
    >>> refine_atan2(atan2(y,x), Q.zero(y) & Q.negative(x))
    pi
    >>> refine_atan2(atan2(y,x), Q.positive(y) & Q.zero(x))
    pi/2
    >>> refine_atan2(atan2(y,x), Q.negative(y) & Q.zero(x))
    -pi/2
    >>> refine_atan2(atan2(y,x), Q.zero(y) & Q.zero(x))
    nan
    """
    from sympy.functions.elementary.trigonometric import atan
    from sympy.core import S
    y, x = expr.args
    if ask(Q.real(y) & Q.positive(x), assumptions):
        return atan(y / x)
    elif ask(Q.negative(y) & Q.negative(x), assumptions):
        return atan(y / x) - S.Pi
    elif ask(Q.positive(y) & Q.negative(x), assumptions):
        return atan(y / x) + S.Pi
    elif ask(Q.zero(y) & Q.negative(x), assumptions):
        return S.Pi
    elif ask(Q.positive(y) & Q.zero(x), assumptions):
        return S.Pi/2
    elif ask(Q.negative(y) & Q.zero(x), assumptions):
        return -S.Pi/2
    elif ask(Q.zero(y) & Q.zero(x), assumptions):
        return S.NaN
    else:
        return expr
```
