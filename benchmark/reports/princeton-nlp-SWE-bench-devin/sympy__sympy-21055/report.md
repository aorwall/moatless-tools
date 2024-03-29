# sympy__sympy-21055

| **sympy/sympy** | `748ce73479ee2cd5c861431091001cc18943c735` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 1659 |
| **Any found context length** | 832 |
| **Avg pos** | 9.0 |
| **Min pos** | 3 |
| **Max pos** | 6 |
| **Top file pos** | 1 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sympy/assumptions/refine.py b/sympy/assumptions/refine.py
--- a/sympy/assumptions/refine.py
+++ b/sympy/assumptions/refine.py
@@ -297,6 +297,28 @@ def refine_im(expr, assumptions):
         return - S.ImaginaryUnit * arg
     return _refine_reim(expr, assumptions)
 
+def refine_arg(expr, assumptions):
+    """
+    Handler for complex argument
+
+    Explanation
+    ===========
+
+    >>> from sympy.assumptions.refine import refine_arg
+    >>> from sympy import Q, arg
+    >>> from sympy.abc import x
+    >>> refine_arg(arg(x), Q.positive(x))
+    0
+    >>> refine_arg(arg(x), Q.negative(x))
+    pi
+    """
+    rg = expr.args[0]
+    if ask(Q.positive(rg), assumptions):
+        return S.Zero
+    if ask(Q.negative(rg), assumptions):
+        return S.Pi
+    return None
+
 
 def _refine_reim(expr, assumptions):
     # Helper function for refine_re & refine_im
@@ -379,6 +401,7 @@ def refine_matrixelement(expr, assumptions):
     'atan2': refine_atan2,
     're': refine_re,
     'im': refine_im,
+    'arg': refine_arg,
     'sign': refine_sign,
     'MatrixElement': refine_matrixelement
 }  # type: Dict[str, Callable[[Expr, Boolean], Expr]]

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sympy/assumptions/refine.py | 300 | 300 | 3 | 1 | 832
| sympy/assumptions/refine.py | 382 | 382 | 6 | 1 | 1659


## Problem Statement

```
`refine()` does not understand how to simplify complex arguments
Just learned about the refine-function, which would come in handy frequently for me.  But
`refine()` does not recognize that argument functions simplify for real numbers.

\`\`\`
>>> from sympy import *                                                     
>>> var('a,x')                                                              
>>> J = Integral(sin(x)*exp(-a*x),(x,0,oo))                                     
>>> J.doit()
	Piecewise((1/(a**2 + 1), 2*Abs(arg(a)) < pi), (Integral(exp(-a*x)*sin(x), (x, 0, oo)), True))
>>> refine(J.doit(),Q.positive(a))                                                 
        Piecewise((1/(a**2 + 1), 2*Abs(arg(a)) < pi), (Integral(exp(-a*x)*sin(x), (x, 0, oo)), True))
>>> refine(abs(a),Q.positive(a))                                            
	a
>>> refine(arg(a),Q.positive(a))                                            
	arg(a)
\`\`\`
I cann't find any open issues identifying this.  Easy to fix, though.



```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 sympy/assumptions/refine.py** | 1 | 66| 482 | 482 | 2949 | 
| 2 | **1 sympy/assumptions/refine.py** | 255 | 275| 136 | 618 | 2949 | 
| **-> 3 <-** | **1 sympy/assumptions/refine.py** | 278 | 309| 214 | 832 | 2949 | 
| 4 | **1 sympy/assumptions/refine.py** | 312 | 350| 300 | 1132 | 2949 | 
| 5 | **1 sympy/assumptions/refine.py** | 69 | 105| 266 | 1398 | 2949 | 
| **-> 6 <-** | **1 sympy/assumptions/refine.py** | 353 | 385| 261 | 1659 | 2949 | 
| 7 | 2 sympy/integrals/transforms.py | 979 | 1004| 231 | 1890 | 19959 | 
| 8 | 3 sympy/core/basic.py | 1672 | 1701| 237 | 2127 | 35852 | 
| 9 | 4 sympy/simplify/simplify.py | 1264 | 1331| 695 | 2822 | 54780 | 
| 10 | **4 sympy/assumptions/refine.py** | 209 | 252| 461 | 3283 | 54780 | 
| 11 | 5 sympy/core/expr.py | 3635 | 3707| 666 | 3949 | 88567 | 
| 12 | 5 sympy/simplify/simplify.py | 1349 | 1376| 247 | 4196 | 88567 | 
| 13 | 5 sympy/simplify/simplify.py | 1533 | 1559| 293 | 4489 | 88567 | 
| 14 | 5 sympy/simplify/simplify.py | 1333 | 1347| 181 | 4670 | 88567 | 
| 15 | 6 sympy/matrices/expressions/matmul.py | 431 | 466| 229 | 4899 | 92163 | 
| 16 | 7 sympy/matrices/common.py | 2271 | 2289| 156 | 5055 | 116739 | 
| 17 | **7 sympy/assumptions/refine.py** | 108 | 206| 825 | 5880 | 116739 | 
| 18 | 8 sympy/functions/elementary/piecewise.py | 1162 | 1176| 142 | 6022 | 127915 | 
| 19 | 8 sympy/simplify/simplify.py | 658 | 757| 843 | 6865 | 127915 | 
| 20 | 8 sympy/integrals/transforms.py | 911 | 955| 366 | 7231 | 127915 | 
| 21 | 9 sympy/integrals/meijerint.py | 600 | 645| 507 | 7738 | 152318 | 
| 22 | 9 sympy/simplify/simplify.py | 1560 | 1575| 153 | 7891 | 152318 | 


## Patch

```diff
diff --git a/sympy/assumptions/refine.py b/sympy/assumptions/refine.py
--- a/sympy/assumptions/refine.py
+++ b/sympy/assumptions/refine.py
@@ -297,6 +297,28 @@ def refine_im(expr, assumptions):
         return - S.ImaginaryUnit * arg
     return _refine_reim(expr, assumptions)
 
+def refine_arg(expr, assumptions):
+    """
+    Handler for complex argument
+
+    Explanation
+    ===========
+
+    >>> from sympy.assumptions.refine import refine_arg
+    >>> from sympy import Q, arg
+    >>> from sympy.abc import x
+    >>> refine_arg(arg(x), Q.positive(x))
+    0
+    >>> refine_arg(arg(x), Q.negative(x))
+    pi
+    """
+    rg = expr.args[0]
+    if ask(Q.positive(rg), assumptions):
+        return S.Zero
+    if ask(Q.negative(rg), assumptions):
+        return S.Pi
+    return None
+
 
 def _refine_reim(expr, assumptions):
     # Helper function for refine_re & refine_im
@@ -379,6 +401,7 @@ def refine_matrixelement(expr, assumptions):
     'atan2': refine_atan2,
     're': refine_re,
     'im': refine_im,
+    'arg': refine_arg,
     'sign': refine_sign,
     'MatrixElement': refine_matrixelement
 }  # type: Dict[str, Callable[[Expr, Boolean], Expr]]

```

## Test Patch

```diff
diff --git a/sympy/assumptions/tests/test_refine.py b/sympy/assumptions/tests/test_refine.py
--- a/sympy/assumptions/tests/test_refine.py
+++ b/sympy/assumptions/tests/test_refine.py
@@ -1,5 +1,5 @@
 from sympy import (Abs, exp, Expr, I, pi, Q, Rational, refine, S, sqrt,
-                   atan, atan2, nan, Symbol, re, im, sign)
+                   atan, atan2, nan, Symbol, re, im, sign, arg)
 from sympy.abc import w, x, y, z
 from sympy.core.relational import Eq, Ne
 from sympy.functions.elementary.piecewise import Piecewise
@@ -160,6 +160,10 @@ def test_sign():
     x = Symbol('x', complex=True)
     assert refine(sign(x), Q.zero(x)) == 0
 
+def test_arg():
+    x = Symbol('x', complex = True)
+    assert refine(arg(x), Q.positive(x)) == 0
+    assert refine(arg(x), Q.negative(x)) == pi
 
 def test_func_args():
     class MyClass(Expr):

```


## Code snippets

### 1 - sympy/assumptions/refine.py:

Start line: 1, End line: 66

```python
from typing import Dict, Callable

from sympy.core import S, Add, Expr, Basic, Mul
from sympy.logic.boolalg import Boolean

from sympy.assumptions import ask, Q  # type: ignore


def refine(expr, assumptions=True):
    """
    Simplify an expression using assumptions.

    Explanation
    ===========

    Unlike :func:`~.simplify()` which performs structural simplification
    without any assumption, this function transforms the expression into
    the form which is only valid under certain assumptions. Note that
    ``simplify()`` is generally not done in refining process.

    Refining boolean expression involves reducing it to ``True`` or
    ``False``. Unlike :func:~.`ask()`, the expression will not be reduced
    if the truth value cannot be determined.

    Examples
    ========

    >>> from sympy import refine, sqrt, Q
    >>> from sympy.abc import x
    >>> refine(sqrt(x**2), Q.real(x))
    Abs(x)
    >>> refine(sqrt(x**2), Q.positive(x))
    x

    >>> refine(Q.real(x), Q.positive(x))
    True
    >>> refine(Q.positive(x), Q.real(x))
    Q.positive(x)

    See Also
    ========

    sympy.simplify.simplify.simplify : Structural simplification without assumptions.
    sympy.assumptions.ask.ask : Query for boolean expressions using assumptions.
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
### 2 - sympy/assumptions/refine.py:

Start line: 255, End line: 275

```python
def refine_re(expr, assumptions):
    """
    Handler for real part.

    Examples
    ========

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

Start line: 278, End line: 309

```python
def refine_im(expr, assumptions):
    """
    Handler for imaginary part.

    Explanation
    ===========

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


def _refine_reim(expr, assumptions):
    # Helper function for refine_re & refine_im
    expanded = expr.expand(complex = True)
    if expanded != expr:
        refined = refine(expanded, assumptions)
        if refined != expanded:
            return refined
    # Best to leave the expression as is
    return None
```
### 4 - sympy/assumptions/refine.py:

Start line: 312, End line: 350

```python
def refine_sign(expr, assumptions):
    """
    Handler for sign.

    Examples
    ========

    >>> from sympy.assumptions.refine import refine_sign
    >>> from sympy import Symbol, Q, sign, im
    >>> x = Symbol('x', real = True)
    >>> expr = sign(x)
    >>> refine_sign(expr, Q.positive(x) & Q.nonzero(x))
    1
    >>> refine_sign(expr, Q.negative(x) & Q.nonzero(x))
    -1
    >>> refine_sign(expr, Q.zero(x))
    0
    >>> y = Symbol('y', imaginary = True)
    >>> expr = sign(y)
    >>> refine_sign(expr, Q.positive(im(y)))
    I
    >>> refine_sign(expr, Q.negative(im(y)))
    -I
    """
    arg = expr.args[0]
    if ask(Q.zero(arg), assumptions):
        return S.Zero
    if ask(Q.real(arg)):
        if ask(Q.positive(arg), assumptions):
            return S.One
        if ask(Q.negative(arg), assumptions):
            return S.NegativeOne
    if ask(Q.imaginary(arg)):
        arg_re, arg_im = arg.as_real_imag()
        if ask(Q.positive(arg_im), assumptions):
            return S.ImaginaryUnit
        if ask(Q.negative(arg_im), assumptions):
            return -S.ImaginaryUnit
    return expr
```
### 5 - sympy/assumptions/refine.py:

Start line: 69, End line: 105

```python
def refine_abs(expr, assumptions):
    """
    Handler for the absolute value.

    Examples
    ========

    >>> from sympy import Q, Abs
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
### 6 - sympy/assumptions/refine.py:

Start line: 353, End line: 385

```python
def refine_matrixelement(expr, assumptions):
    """
    Handler for symmetric part.

    Examples
    ========

    >>> from sympy.assumptions.refine import refine_matrixelement
    >>> from sympy import Q
    >>> from sympy.matrices.expressions.matexpr import MatrixSymbol
    >>> X = MatrixSymbol('X', 3, 3)
    >>> refine_matrixelement(X[0, 1], Q.symmetric(X))
    X[0, 1]
    >>> refine_matrixelement(X[1, 0], Q.symmetric(X))
    X[0, 1]
    """
    from sympy.matrices.expressions.matexpr import MatrixElement
    matrix, i, j = expr.args
    if ask(Q.symmetric(matrix), assumptions):
        if (i - j).could_extract_minus_sign():
            return expr
        return MatrixElement(matrix, j, i)

handlers_dict = {
    'Abs': refine_abs,
    'Pow': refine_Pow,
    'atan2': refine_atan2,
    're': refine_re,
    'im': refine_im,
    'sign': refine_sign,
    'MatrixElement': refine_matrixelement
}  # type: Dict[str, Callable[[Expr, Boolean], Expr]]
```
### 7 - sympy/integrals/transforms.py:

Start line: 979, End line: 1004

```python
def _simplifyconds(expr, s, a):
    # ... other code

    def replie(x, y):
        """ simplify x < y """
        if not (x.is_positive or isinstance(x, Abs)) \
                or not (y.is_positive or isinstance(y, Abs)):
            return (x < y)
        r = bigger(x, y)
        if r is not None:
            return not r
        return (x < y)

    def replue(x, y):
        b = bigger(x, y)
        if b == True or b == False:
            return True
        return Unequality(x, y)

    def repl(ex, *args):
        if ex == True or ex == False:
            return bool(ex)
        return ex.replace(*args)
    from sympy.simplify.radsimp import collect_abs
    expr = collect_abs(expr)
    expr = repl(expr, StrictLessThan, replie)
    expr = repl(expr, StrictGreaterThan, lambda x, y: replie(y, x))
    expr = repl(expr, Unequality, replue)
    return S(expr)
```
### 8 - sympy/core/basic.py:

Start line: 1672, End line: 1701

```python
class Basic(Printable, metaclass=ManagedProperties):

    def simplify(self, **kwargs):
        """See the simplify function in sympy.simplify"""
        from sympy.simplify import simplify
        return simplify(self, **kwargs)

    def refine(self, assumption=True):
        """See the refine function in sympy.assumptions"""
        from sympy.assumptions import refine
        return refine(self, assumption)

    def _eval_rewrite(self, pattern, rule, **hints):
        if self.is_Atom:
            if hasattr(self, rule):
                return getattr(self, rule)()
            return self

        if hints.get('deep', True):
            args = [a._eval_rewrite(pattern, rule, **hints)
                        if isinstance(a, Basic) else a
                        for a in self.args]
        else:
            args = self.args

        if pattern is None or isinstance(self, pattern):
            if hasattr(self, rule):
                rewritten = getattr(self, rule)(*args, **hints)
                if rewritten is not None:
                    return rewritten

        return self.func(*args) if hints.get('evaluate', True) else self
```
### 9 - sympy/simplify/simplify.py:

Start line: 1264, End line: 1331

```python
def besselsimp(expr):
    """
    Simplify bessel-type functions.

    Explanation
    ===========

    This routine tries to simplify bessel-type functions. Currently it only
    works on the Bessel J and I functions, however. It works by looking at all
    such functions in turn, and eliminating factors of "I" and "-1" (actually
    their polar equivalents) in front of the argument. Then, functions of
    half-integer order are rewritten using strigonometric functions and
    functions of integer order (> 1) are rewritten using functions
    of low order.  Finally, if the expression was changed, compute
    factorization of the result with factor().

    >>> from sympy import besselj, besseli, besselsimp, polar_lift, I, S
    >>> from sympy.abc import z, nu
    >>> besselsimp(besselj(nu, z*polar_lift(-1)))
    exp(I*pi*nu)*besselj(nu, z)
    >>> besselsimp(besseli(nu, z*polar_lift(-I)))
    exp(-I*pi*nu/2)*besselj(nu, z)
    >>> besselsimp(besseli(S(-1)/2, z))
    sqrt(2)*cosh(z)/(sqrt(pi)*sqrt(z))
    >>> besselsimp(z*besseli(0, z) + z*(besseli(2, z))/2 + besseli(1, z))
    3*z*besseli(0, z)/2
    """
    # TODO
    # - better algorithm?
    # - simplify (cos(pi*b)*besselj(b,z) - besselj(-b,z))/sin(pi*b) ...
    # - use contiguity relations?

    def replacer(fro, to, factors):
        factors = set(factors)

        def repl(nu, z):
            if factors.intersection(Mul.make_args(z)):
                return to(nu, z)
            return fro(nu, z)
        return repl

    def torewrite(fro, to):
        def tofunc(nu, z):
            return fro(nu, z).rewrite(to)
        return tofunc

    def tominus(fro):
        def tofunc(nu, z):
            return exp(I*pi*nu)*fro(nu, exp_polar(-I*pi)*z)
        return tofunc

    orig_expr = expr

    ifactors = [I, exp_polar(I*pi/2), exp_polar(-I*pi/2)]
    expr = expr.replace(
        besselj, replacer(besselj,
        torewrite(besselj, besseli), ifactors))
    expr = expr.replace(
        besseli, replacer(besseli,
        torewrite(besseli, besselj), ifactors))

    minusfactors = [-1, exp_polar(I*pi)]
    expr = expr.replace(
        besselj, replacer(besselj, tominus(besselj), minusfactors))
    expr = expr.replace(
        besseli, replacer(besseli, tominus(besseli), minusfactors))

    z0 = Dummy('z')
    # ... other code
```
### 10 - sympy/assumptions/refine.py:

Start line: 209, End line: 252

```python
def refine_atan2(expr, assumptions):
    """
    Handler for the atan2 function.

    Examples
    ========

    >>> from sympy import Q, atan2
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
### 17 - sympy/assumptions/refine.py:

Start line: 108, End line: 206

```python
def refine_Pow(expr, assumptions):
    """
    Handler for instances of Pow.

    Examples
    ========

    >>> from sympy import Q
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
                even_terms = set()
                odd_terms = set()
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
