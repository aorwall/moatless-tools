# sympy__sympy-17251

| **sympy/sympy** | `8ca4a683d58ac1f61cfd2e4dacf7f58b9c0fefab` |
| ---- | ---- |
| **No of patches** | 2 |
| **All found context length** | 4137 |
| **Any found context length** | 4137 |
| **Avg pos** | 11.0 |
| **Min pos** | 11 |
| **Max pos** | 11 |
| **Top file pos** | 2 |
| **Missing snippets** | 3 |
| **Missing patch files** | 1 |


## Expected patch

```diff
diff --git a/sympy/functions/elementary/exponential.py b/sympy/functions/elementary/exponential.py
--- a/sympy/functions/elementary/exponential.py
+++ b/sympy/functions/elementary/exponential.py
@@ -250,18 +250,23 @@ def eval(cls, arg):
         elif isinstance(arg, SetExpr):
             return arg._eval_func(cls)
         elif arg.is_Mul:
-            if arg.is_number or arg.is_Symbol:
-                coeff = arg.coeff(S.Pi*S.ImaginaryUnit)
-                if coeff:
-                    if ask(Q.integer(2*coeff)):
-                        if ask(Q.even(coeff)):
-                            return S.One
-                        elif ask(Q.odd(coeff)):
-                            return S.NegativeOne
-                        elif ask(Q.even(coeff + S.Half)):
-                            return -S.ImaginaryUnit
-                        elif ask(Q.odd(coeff + S.Half)):
-                            return S.ImaginaryUnit
+            coeff = arg.as_coefficient(S.Pi*S.ImaginaryUnit)
+            if coeff:
+                if (2*coeff).is_integer:
+                    if coeff.is_even:
+                        return S.One
+                    elif coeff.is_odd:
+                        return S.NegativeOne
+                    elif (coeff + S.Half).is_even:
+                        return -S.ImaginaryUnit
+                    elif (coeff + S.Half).is_odd:
+                        return S.ImaginaryUnit
+                elif coeff.is_Rational:
+                    ncoeff = coeff % 2 # restrict to [0, 2pi)
+                    if ncoeff > 1: # restrict to (-pi, pi]
+                        ncoeff -= 2
+                    if ncoeff != coeff:
+                        return cls(ncoeff*S.Pi*S.ImaginaryUnit)
 
             # Warning: code in risch.py will be very sensitive to changes
             # in this (see DifferentialExtension).
@@ -292,16 +297,21 @@ def eval(cls, arg):
         elif arg.is_Add:
             out = []
             add = []
+            argchanged = False
             for a in arg.args:
                 if a is S.One:
                     add.append(a)
                     continue
                 newa = cls(a)
                 if isinstance(newa, cls):
-                    add.append(a)
+                    if newa.args[0] != a:
+                        add.append(newa.args[0])
+                        argchanged = True
+                    else:
+                        add.append(a)
                 else:
                     out.append(newa)
-            if out:
+            if out or argchanged:
                 return Mul(*out)*cls(Add(*add), evaluate=False)
 
         elif isinstance(arg, MatrixBase):
diff --git a/sympy/physics/matrices.py b/sympy/physics/matrices.py
--- a/sympy/physics/matrices.py
+++ b/sympy/physics/matrices.py
@@ -171,8 +171,8 @@ def mdft(n):
     >>> mdft(3)
     Matrix([
     [sqrt(3)/3,                sqrt(3)/3,                sqrt(3)/3],
-    [sqrt(3)/3, sqrt(3)*exp(-2*I*pi/3)/3, sqrt(3)*exp(-4*I*pi/3)/3],
-    [sqrt(3)/3, sqrt(3)*exp(-4*I*pi/3)/3, sqrt(3)*exp(-8*I*pi/3)/3]])
+    [sqrt(3)/3, sqrt(3)*exp(-2*I*pi/3)/3,  sqrt(3)*exp(2*I*pi/3)/3],
+    [sqrt(3)/3,  sqrt(3)*exp(2*I*pi/3)/3, sqrt(3)*exp(-2*I*pi/3)/3]])
     """
     mat = [[None for x in range(n)] for y in range(n)]
     base = exp(-2*pi*I/n)

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sympy/functions/elementary/exponential.py | 253 | 264 | 11 | 2 | 4137
| sympy/functions/elementary/exponential.py | 295 | 298 | 11 | 2 | 4137
| sympy/physics/matrices.py | 174 | 175 | - | - | -


## Problem Statement

```
exp doesn't simplify based on its periodicity
In current master, `exp` doesn't use its periodicity to automatically reduce its argument, not even for purely imaginary arguments:
\`\`\`
>>> exp(9*I*pi/4)
 9⋅ⅈ⋅π
 ─────
   4
ℯ
>>> simplify(exp(9*I*pi/4))
 9⋅ⅈ⋅π
 ─────
   4
ℯ
>>> a = exp(9*I*pi/4) - exp(I*pi/4); a
   ⅈ⋅π    9⋅ⅈ⋅π
   ───    ─────
    4       4
- ℯ    + ℯ
>>> simplify(a)
            9⋅ⅈ⋅π
            ─────
  4 ____      4
- ╲╱ -1  + ℯ
>>> expand_complex(a)
0
\`\`\`

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 sympy/functions/elementary/complexes.py | 871 | 896| 228 | 228 | 9241 | 
| 2 | **2 sympy/functions/elementary/exponential.py** | 118 | 155| 317 | 545 | 16317 | 
| 3 | **2 sympy/functions/elementary/exponential.py** | 157 | 184| 243 | 788 | 16317 | 
| 4 | 3 sympy/simplify/simplify.py | 1104 | 1168| 688 | 1476 | 32540 | 
| 5 | 4 sympy/simplify/trigsimp.py | 591 | 603| 134 | 1610 | 44768 | 
| 6 | 4 sympy/simplify/trigsimp.py | 511 | 537| 217 | 1827 | 44768 | 
| 7 | 4 sympy/simplify/simplify.py | 1336 | 1369| 347 | 2174 | 44768 | 
| 8 | 4 sympy/simplify/simplify.py | 1170 | 1189| 194 | 2368 | 44768 | 
| 9 | 5 sympy/core/expr.py | 2453 | 2524| 689 | 3057 | 77110 | 
| 10 | 5 sympy/simplify/trigsimp.py | 539 | 590| 492 | 3549 | 77110 | 
| **-> 11 <-** | **5 sympy/functions/elementary/exponential.py** | 226 | 308| 588 | 4137 | 77110 | 
| 12 | 5 sympy/functions/elementary/complexes.py | 920 | 956| 390 | 4527 | 77110 | 
| 13 | 5 sympy/simplify/simplify.py | 1370 | 1385| 153 | 4680 | 77110 | 
| 14 | **5 sympy/functions/elementary/exponential.py** | 382 | 405| 223 | 4903 | 77110 | 
| 15 | 6 sympy/simplify/hyperexpand.py | 69 | 82| 240 | 5143 | 101877 | 
| 16 | **6 sympy/functions/elementary/exponential.py** | 334 | 365| 262 | 5405 | 101877 | 
| 17 | 6 sympy/functions/elementary/complexes.py | 1131 | 1160| 324 | 5729 | 101877 | 
| 18 | 6 sympy/functions/elementary/complexes.py | 992 | 1050| 528 | 6257 | 101877 | 
| 19 | 6 sympy/simplify/simplify.py | 380 | 516| 1420 | 7677 | 101877 | 
| 20 | 7 sympy/core/numbers.py | 1349 | 1379| 333 | 8010 | 131974 | 
| 21 | 8 sympy/simplify/radsimp.py | 792 | 893| 995 | 9005 | 141613 | 
| 22 | 8 sympy/simplify/simplify.py | 618 | 658| 363 | 9368 | 141613 | 
| 23 | 9 sympy/simplify/powsimp.py | 485 | 572| 857 | 10225 | 148266 | 
| 24 | 9 sympy/simplify/hyperexpand.py | 234 | 265| 507 | 10732 | 148266 | 
| 25 | 9 sympy/functions/elementary/complexes.py | 792 | 827| 219 | 10951 | 148266 | 
| 26 | 9 sympy/simplify/hyperexpand.py | 198 | 233| 686 | 11637 | 148266 | 
| 27 | 9 sympy/simplify/simplify.py | 1252 | 1334| 780 | 12417 | 148266 | 
| 28 | 10 sympy/functions/special/error_functions.py | 1221 | 1232| 172 | 12589 | 167770 | 
| 29 | **10 sympy/functions/elementary/exponential.py** | 407 | 425| 223 | 12812 | 167770 | 
| 30 | 11 sympy/core/function.py | 2780 | 2800| 166 | 12978 | 194539 | 
| 31 | **11 sympy/functions/elementary/exponential.py** | 427 | 474| 493 | 13471 | 194539 | 
| 32 | 11 sympy/simplify/radsimp.py | 895 | 917| 278 | 13749 | 194539 | 
| 33 | **11 sympy/functions/elementary/exponential.py** | 367 | 380| 182 | 13931 | 194539 | 
| 34 | 11 sympy/simplify/simplify.py | 1 | 32| 396 | 14327 | 194539 | 
| 35 | 11 sympy/simplify/trigsimp.py | 985 | 1065| 724 | 15051 | 194539 | 
| 36 | 11 sympy/simplify/radsimp.py | 672 | 744| 753 | 15804 | 194539 | 
| 37 | 11 sympy/simplify/simplify.py | 517 | 617| 836 | 16640 | 194539 | 
| 38 | 11 sympy/functions/elementary/complexes.py | 898 | 918| 174 | 16814 | 194539 | 


## Missing Patch Files

 * 1: sympy/functions/elementary/exponential.py
 * 2: sympy/physics/matrices.py

## Patch

```diff
diff --git a/sympy/functions/elementary/exponential.py b/sympy/functions/elementary/exponential.py
--- a/sympy/functions/elementary/exponential.py
+++ b/sympy/functions/elementary/exponential.py
@@ -250,18 +250,23 @@ def eval(cls, arg):
         elif isinstance(arg, SetExpr):
             return arg._eval_func(cls)
         elif arg.is_Mul:
-            if arg.is_number or arg.is_Symbol:
-                coeff = arg.coeff(S.Pi*S.ImaginaryUnit)
-                if coeff:
-                    if ask(Q.integer(2*coeff)):
-                        if ask(Q.even(coeff)):
-                            return S.One
-                        elif ask(Q.odd(coeff)):
-                            return S.NegativeOne
-                        elif ask(Q.even(coeff + S.Half)):
-                            return -S.ImaginaryUnit
-                        elif ask(Q.odd(coeff + S.Half)):
-                            return S.ImaginaryUnit
+            coeff = arg.as_coefficient(S.Pi*S.ImaginaryUnit)
+            if coeff:
+                if (2*coeff).is_integer:
+                    if coeff.is_even:
+                        return S.One
+                    elif coeff.is_odd:
+                        return S.NegativeOne
+                    elif (coeff + S.Half).is_even:
+                        return -S.ImaginaryUnit
+                    elif (coeff + S.Half).is_odd:
+                        return S.ImaginaryUnit
+                elif coeff.is_Rational:
+                    ncoeff = coeff % 2 # restrict to [0, 2pi)
+                    if ncoeff > 1: # restrict to (-pi, pi]
+                        ncoeff -= 2
+                    if ncoeff != coeff:
+                        return cls(ncoeff*S.Pi*S.ImaginaryUnit)
 
             # Warning: code in risch.py will be very sensitive to changes
             # in this (see DifferentialExtension).
@@ -292,16 +297,21 @@ def eval(cls, arg):
         elif arg.is_Add:
             out = []
             add = []
+            argchanged = False
             for a in arg.args:
                 if a is S.One:
                     add.append(a)
                     continue
                 newa = cls(a)
                 if isinstance(newa, cls):
-                    add.append(a)
+                    if newa.args[0] != a:
+                        add.append(newa.args[0])
+                        argchanged = True
+                    else:
+                        add.append(a)
                 else:
                     out.append(newa)
-            if out:
+            if out or argchanged:
                 return Mul(*out)*cls(Add(*add), evaluate=False)
 
         elif isinstance(arg, MatrixBase):
diff --git a/sympy/physics/matrices.py b/sympy/physics/matrices.py
--- a/sympy/physics/matrices.py
+++ b/sympy/physics/matrices.py
@@ -171,8 +171,8 @@ def mdft(n):
     >>> mdft(3)
     Matrix([
     [sqrt(3)/3,                sqrt(3)/3,                sqrt(3)/3],
-    [sqrt(3)/3, sqrt(3)*exp(-2*I*pi/3)/3, sqrt(3)*exp(-4*I*pi/3)/3],
-    [sqrt(3)/3, sqrt(3)*exp(-4*I*pi/3)/3, sqrt(3)*exp(-8*I*pi/3)/3]])
+    [sqrt(3)/3, sqrt(3)*exp(-2*I*pi/3)/3,  sqrt(3)*exp(2*I*pi/3)/3],
+    [sqrt(3)/3,  sqrt(3)*exp(2*I*pi/3)/3, sqrt(3)*exp(-2*I*pi/3)/3]])
     """
     mat = [[None for x in range(n)] for y in range(n)]
     base = exp(-2*pi*I/n)

```

## Test Patch

```diff
diff --git a/sympy/functions/elementary/tests/test_exponential.py b/sympy/functions/elementary/tests/test_exponential.py
--- a/sympy/functions/elementary/tests/test_exponential.py
+++ b/sympy/functions/elementary/tests/test_exponential.py
@@ -48,6 +48,25 @@ def test_exp_values():
     assert exp(oo, evaluate=False).is_finite is False
 
 
+def test_exp_period():
+    assert exp(9*I*pi/4) == exp(I*pi/4)
+    assert exp(46*I*pi/18) == exp(5*I*pi/9)
+    assert exp(25*I*pi/7) == exp(-3*I*pi/7)
+    assert exp(-19*I*pi/3) == exp(-I*pi/3)
+    assert exp(37*I*pi/8) - exp(-11*I*pi/8) == 0
+    assert exp(-5*I*pi/3) / exp(11*I*pi/5) * exp(148*I*pi/15) == 1
+
+    assert exp(2 - 17*I*pi/5) == exp(2 + 3*I*pi/5)
+    assert exp(log(3) + 29*I*pi/9) == 3 * exp(-7*I*pi/9)
+
+    n = Symbol('n', integer=True)
+    e = Symbol('e', even=True)
+    assert exp(e*I*pi) == 1
+    assert exp((e + 1)*I*pi) == -1
+    assert exp((1 + 4*n)*I*pi/2) == I
+    assert exp((-1 + 4*n)*I*pi/2) == -I
+
+
 def test_exp_log():
     x = Symbol("x", real=True)
     assert log(exp(x)) == x

```


## Code snippets

### 1 - sympy/functions/elementary/complexes.py:

Start line: 871, End line: 896

```python
class periodic_argument(Function):
    """
    Represent the argument on a quotient of the Riemann surface of the
    logarithm. That is, given a period P, always return a value in
    (-P/2, P/2], by using exp(P*I) == 1.

    >>> from sympy import exp, exp_polar, periodic_argument, unbranched_argument
    >>> from sympy import I, pi
    >>> unbranched_argument(exp(5*I*pi))
    pi
    >>> unbranched_argument(exp_polar(5*I*pi))
    5*pi
    >>> periodic_argument(exp_polar(5*I*pi), 2*pi)
    pi
    >>> periodic_argument(exp_polar(5*I*pi), 3*pi)
    -pi
    >>> periodic_argument(exp_polar(5*I*pi), pi)
    0

    See Also
    ========

    sympy.functions.elementary.exponential.exp_polar
    polar_lift : Lift argument to the Riemann surface of the logarithm
    principal_branch
    """
```
### 2 - sympy/functions/elementary/exponential.py:

Start line: 118, End line: 155

```python
class exp_polar(ExpBase):
    r"""
    Represent a 'polar number' (see g-function Sphinx documentation).

    ``exp_polar`` represents the function
    `Exp: \mathbb{C} \rightarrow \mathcal{S}`, sending the complex number
    `z = a + bi` to the polar number `r = exp(a), \theta = b`. It is one of
    the main functions to construct polar numbers.

    >>> from sympy import exp_polar, pi, I, exp

    The main difference is that polar numbers don't "wrap around" at `2 \pi`:

    >>> exp(2*pi*I)
    1
    >>> exp_polar(2*pi*I)
    exp_polar(2*I*pi)

    apart from that they behave mostly like classical complex numbers:

    >>> exp_polar(2)*exp_polar(3)
    exp_polar(5)

    See Also
    ========

    sympy.simplify.simplify.powsimp
    sympy.functions.elementary.complexes.polar_lift
    sympy.functions.elementary.complexes.periodic_argument
    sympy.functions.elementary.complexes.principal_branch
    """

    is_polar = True
    is_comparable = False  # cannot be evalf'd

    def _eval_Abs(self):   # Abs is never a polar number
        from sympy.functions.elementary.complexes import re
        return exp(re(self.args[0]))
```
### 3 - sympy/functions/elementary/exponential.py:

Start line: 157, End line: 184

```python
class exp_polar(ExpBase):

    def _eval_evalf(self, prec):
        """ Careful! any evalf of polar numbers is flaky """
        from sympy import im, pi, re
        i = im(self.args[0])
        try:
            bad = (i <= -pi or i > pi)
        except TypeError:
            bad = True
        if bad:
            return self  # cannot evalf for this argument
        res = exp(self.args[0])._eval_evalf(prec)
        if i > 0 and im(res) < 0:
            # i ~ pi, but exp(I*i) evaluated to argument slightly bigger than pi
            return re(res)
        return res

    def _eval_power(self, other):
        return self.func(self.args[0]*other)

    def _eval_is_extended_real(self):
        if self.args[0].is_extended_real:
            return True

    def as_base_exp(self):
        # XXX exp_polar(0) is special!
        if self.args[0] == 0:
            return self, S(1)
        return ExpBase.as_base_exp(self)
```
### 4 - sympy/simplify/simplify.py:

Start line: 1104, End line: 1168

```python
def besselsimp(expr):
    """
    Simplify bessel-type functions.

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
### 5 - sympy/simplify/trigsimp.py:

Start line: 591, End line: 603

```python
def exptrigsimp(expr):
    # ... other code
    newexpr = bottom_up(newexpr, f)

    # sin/cos and sinh/cosh ratios to tan and tanh, respectively
    if newexpr.has(HyperbolicFunction):
        e, f = hyper_as_trig(newexpr)
        newexpr = f(TR2i(e))
    if newexpr.has(TrigonometricFunction):
        newexpr = TR2i(newexpr)

    # can we ever generate an I where there was none previously?
    if not (newexpr.has(I) and not expr.has(I)):
        expr = newexpr
    return expr
```
### 6 - sympy/simplify/trigsimp.py:

Start line: 511, End line: 537

```python
def exptrigsimp(expr):
    """
    Simplifies exponential / trigonometric / hyperbolic functions.

    Examples
    ========

    >>> from sympy import exptrigsimp, exp, cosh, sinh
    >>> from sympy.abc import z

    >>> exptrigsimp(exp(z) + exp(-z))
    2*cosh(z)
    >>> exptrigsimp(cosh(z) - sinh(z))
    exp(-z)
    """
    from sympy.simplify.fu import hyper_as_trig, TR2i
    from sympy.simplify.simplify import bottom_up

    def exp_trig(e):
        # select the better of e, and e rewritten in terms of exp or trig
        # functions
        choices = [e]
        if e.has(*_trigs):
            choices.append(e.rewrite(exp))
        choices.append(e.rewrite(cos))
        return min(*choices, key=count_ops)
    newexpr = bottom_up(expr, exp_trig)
    # ... other code
```
### 7 - sympy/simplify/simplify.py:

Start line: 1336, End line: 1369

```python
def nsimplify(expr, constants=(), tolerance=None, full=False, rational=None,
    rational_conversion='base10'):
    # ... other code

    exprval = expr.evalf(prec, chop=True)
    re, im = exprval.as_real_imag()

    # safety check to make sure that this evaluated to a number
    if not (re.is_Number and im.is_Number):
        return expr

    def nsimplify_real(x):
        orig = mpmath.mp.dps
        xv = x._to_mpmath(bprec)
        try:
            # We'll be happy with low precision if a simple fraction
            if not (tolerance or full):
                mpmath.mp.dps = 15
                rat = mpmath.pslq([xv, 1])
                if rat is not None:
                    return Rational(-int(rat[1]), int(rat[0]))
            mpmath.mp.dps = prec
            newexpr = mpmath.identify(xv, constants=constants_dict,
                tol=tolerance, full=full)
            if not newexpr:
                raise ValueError
            if full:
                newexpr = newexpr[0]
            expr = sympify(newexpr)
            if x and not expr:  # don't let x become 0
                raise ValueError
            if expr.is_finite is False and not xv in [mpmath.inf, mpmath.ninf]:
                raise ValueError
            return expr
        finally:
            # even though there are returns above, this is executed
            # before leaving
            mpmath.mp.dps = orig
    # ... other code
```
### 8 - sympy/simplify/simplify.py:

Start line: 1170, End line: 1189

```python
def besselsimp(expr):
    # ... other code

    def expander(fro):
        def repl(nu, z):
            if (nu % 1) == S(1)/2:
                return simplify(trigsimp(unpolarify(
                        fro(nu, z0).rewrite(besselj).rewrite(jn).expand(
                            func=True)).subs(z0, z)))
            elif nu.is_Integer and nu > 1:
                return fro(nu, z).expand(func=True)
            return fro(nu, z)
        return repl

    expr = expr.replace(besselj, expander(besselj))
    expr = expr.replace(bessely, expander(bessely))
    expr = expr.replace(besseli, expander(besseli))
    expr = expr.replace(besselk, expander(besselk))

    if expr != orig_expr:
        expr = expr.factor()

    return expr
```
### 9 - sympy/core/expr.py:

Start line: 2453, End line: 2524

```python
class Expr(Basic, EvalfMixin):

    def extract_branch_factor(self, allow_half=False):
        """
        Try to write self as ``exp_polar(2*pi*I*n)*z`` in a nice way.
        Return (z, n).

        >>> from sympy import exp_polar, I, pi
        >>> from sympy.abc import x, y
        >>> exp_polar(I*pi).extract_branch_factor()
        (exp_polar(I*pi), 0)
        >>> exp_polar(2*I*pi).extract_branch_factor()
        (1, 1)
        >>> exp_polar(-pi*I).extract_branch_factor()
        (exp_polar(I*pi), -1)
        >>> exp_polar(3*pi*I + x).extract_branch_factor()
        (exp_polar(x + I*pi), 1)
        >>> (y*exp_polar(-5*pi*I)*exp_polar(3*pi*I + 2*pi*x)).extract_branch_factor()
        (y*exp_polar(2*pi*x), -1)
        >>> exp_polar(-I*pi/2).extract_branch_factor()
        (exp_polar(-I*pi/2), 0)

        If allow_half is True, also extract exp_polar(I*pi):

        >>> exp_polar(I*pi).extract_branch_factor(allow_half=True)
        (1, 1/2)
        >>> exp_polar(2*I*pi).extract_branch_factor(allow_half=True)
        (1, 1)
        >>> exp_polar(3*I*pi).extract_branch_factor(allow_half=True)
        (1, 3/2)
        >>> exp_polar(-I*pi).extract_branch_factor(allow_half=True)
        (1, -1/2)
        """
        from sympy import exp_polar, pi, I, ceiling, Add
        n = S(0)
        res = S(1)
        args = Mul.make_args(self)
        exps = []
        for arg in args:
            if isinstance(arg, exp_polar):
                exps += [arg.exp]
            else:
                res *= arg
        piimult = S(0)
        extras = []
        while exps:
            exp = exps.pop()
            if exp.is_Add:
                exps += exp.args
                continue
            if exp.is_Mul:
                coeff = exp.as_coefficient(pi*I)
                if coeff is not None:
                    piimult += coeff
                    continue
            extras += [exp]
        if piimult.is_number:
            coeff = piimult
            tail = ()
        else:
            coeff, tail = piimult.as_coeff_add(*piimult.free_symbols)
        # round down to nearest multiple of 2
        branchfact = ceiling(coeff/2 - S(1)/2)*2
        n += branchfact/2
        c = coeff - branchfact
        if allow_half:
            nc = c.extract_additively(1)
            if nc is not None:
                n += S(1)/2
                c = nc
        newexp = pi*I*Add(*((c, ) + tail)) + Add(*extras)
        if newexp != 0:
            res *= exp_polar(newexp)
        return res, n
```
### 10 - sympy/simplify/trigsimp.py:

Start line: 539, End line: 590

```python
def exptrigsimp(expr):
    # ... other code

    def f(rv):
        if not rv.is_Mul:
            return rv
        commutative_part, noncommutative_part = rv.args_cnc()
        # Since as_powers_dict loses order information,
        # if there is more than one noncommutative factor,
        # it should only be used to simplify the commutative part.
        if (len(noncommutative_part) > 1):
            return f(Mul(*commutative_part))*Mul(*noncommutative_part)
        rvd = rv.as_powers_dict()
        newd = rvd.copy()

        def signlog(expr, sign=1):
            if expr is S.Exp1:
                return sign, 1
            elif isinstance(expr, exp):
                return sign, expr.args[0]
            elif sign == 1:
                return signlog(-expr, sign=-1)
            else:
                return None, None

        ee = rvd[S.Exp1]
        for k in rvd:
            if k.is_Add and len(k.args) == 2:
                # k == c*(1 + sign*E**x)
                c = k.args[0]
                sign, x = signlog(k.args[1]/c)
                if not x:
                    continue
                m = rvd[k]
                newd[k] -= m
                if ee == -x*m/2:
                    # sinh and cosh
                    newd[S.Exp1] -= ee
                    ee = 0
                    if sign == 1:
                        newd[2*c*cosh(x/2)] += m
                    else:
                        newd[-2*c*sinh(x/2)] += m
                elif newd[1 - sign*S.Exp1**x] == -m:
                    # tanh
                    del newd[1 - sign*S.Exp1**x]
                    if sign == 1:
                        newd[-c/tanh(x/2)] += m
                    else:
                        newd[-c*tanh(x/2)] += m
                else:
                    newd[1 + sign*S.Exp1**x] += m
                    newd[c] += m

        return Mul(*[k**newd[k] for k in newd])
    # ... other code
```
### 11 - sympy/functions/elementary/exponential.py:

Start line: 226, End line: 308

```python
class exp(ExpBase):

    @classmethod
    def eval(cls, arg):
        from sympy.assumptions import ask, Q
        from sympy.calculus import AccumBounds
        from sympy.sets.setexpr import SetExpr
        from sympy.matrices.matrices import MatrixBase
        from sympy import logcombine
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Zero:
                return S.One
            elif arg is S.One:
                return S.Exp1
            elif arg is S.Infinity:
                return S.Infinity
            elif arg is S.NegativeInfinity:
                return S.Zero
        elif arg is S.ComplexInfinity:
            return S.NaN
        elif isinstance(arg, log):
            return arg.args[0]
        elif isinstance(arg, AccumBounds):
            return AccumBounds(exp(arg.min), exp(arg.max))
        elif isinstance(arg, SetExpr):
            return arg._eval_func(cls)
        elif arg.is_Mul:
            if arg.is_number or arg.is_Symbol:
                coeff = arg.coeff(S.Pi*S.ImaginaryUnit)
                if coeff:
                    if ask(Q.integer(2*coeff)):
                        if ask(Q.even(coeff)):
                            return S.One
                        elif ask(Q.odd(coeff)):
                            return S.NegativeOne
                        elif ask(Q.even(coeff + S.Half)):
                            return -S.ImaginaryUnit
                        elif ask(Q.odd(coeff + S.Half)):
                            return S.ImaginaryUnit

            # Warning: code in risch.py will be very sensitive to changes
            # in this (see DifferentialExtension).

            # look for a single log factor

            coeff, terms = arg.as_coeff_Mul()

            # but it can't be multiplied by oo
            if coeff in [S.NegativeInfinity, S.Infinity]:
                return None

            coeffs, log_term = [coeff], None
            for term in Mul.make_args(terms):
                term_ = logcombine(term)
                if isinstance(term_, log):
                    if log_term is None:
                        log_term = term_.args[0]
                    else:
                        return None
                elif term.is_comparable:
                    coeffs.append(term)
                else:
                    return None

            return log_term**Mul(*coeffs) if log_term else None

        elif arg.is_Add:
            out = []
            add = []
            for a in arg.args:
                if a is S.One:
                    add.append(a)
                    continue
                newa = cls(a)
                if isinstance(newa, cls):
                    add.append(a)
                else:
                    out.append(newa)
            if out:
                return Mul(*out)*cls(Add(*add), evaluate=False)

        elif isinstance(arg, MatrixBase):
            return arg.exp()
```
### 14 - sympy/functions/elementary/exponential.py:

Start line: 382, End line: 405

```python
class exp(ExpBase):

    def _eval_is_extended_real(self):
        if self.args[0].is_extended_real:
            return True
        elif self.args[0].is_imaginary:
            arg2 = -S(2) * S.ImaginaryUnit * self.args[0] / S.Pi
            return arg2.is_even

    def _eval_is_algebraic(self):
        s = self.func(*self.args)
        if s.func == self.func:
            if fuzzy_not(self.exp.is_zero):
                if self.exp.is_algebraic:
                    return False
                elif (self.exp/S.Pi).is_rational:
                    return False
        else:
            return s.is_algebraic

    def _eval_is_extended_positive(self):
        if self.args[0].is_extended_real:
            return not self.args[0] is S.NegativeInfinity
        elif self.args[0].is_imaginary:
            arg2 = -S.ImaginaryUnit * self.args[0] / S.Pi
            return arg2.is_even
```
### 16 - sympy/functions/elementary/exponential.py:

Start line: 334, End line: 365

```python
class exp(ExpBase):

    def as_real_imag(self, deep=True, **hints):
        """
        Returns this function as a 2-tuple representing a complex number.

        Examples
        ========

        >>> from sympy import I
        >>> from sympy.abc import x
        >>> from sympy.functions import exp
        >>> exp(x).as_real_imag()
        (exp(re(x))*cos(im(x)), exp(re(x))*sin(im(x)))
        >>> exp(1).as_real_imag()
        (E, 0)
        >>> exp(I).as_real_imag()
        (cos(1), sin(1))
        >>> exp(1+I).as_real_imag()
        (E*cos(1), E*sin(1))

        See Also
        ========

        sympy.functions.elementary.complexes.re
        sympy.functions.elementary.complexes.im
        """
        import sympy
        re, im = self.args[0].as_real_imag()
        if deep:
            re = re.expand(deep, **hints)
            im = im.expand(deep, **hints)
        cos, sin = sympy.cos(im), sympy.sin(im)
        return (exp(re)*cos, exp(re)*sin)
```
### 29 - sympy/functions/elementary/exponential.py:

Start line: 407, End line: 425

```python
class exp(ExpBase):

    def _eval_nseries(self, x, n, logx):
        # NOTE Please see the comment at the beginning of this file, labelled
        #      IMPORTANT.
        from sympy import limit, oo, Order, powsimp
        arg = self.args[0]
        arg_series = arg._eval_nseries(x, n=n, logx=logx)
        if arg_series.is_Order:
            return 1 + arg_series
        arg0 = limit(arg_series.removeO(), x, 0)
        if arg0 in [-oo, oo]:
            return self
        t = Dummy("t")
        exp_series = exp(t)._taylor(t, n)
        o = exp_series.getO()
        exp_series = exp_series.removeO()
        r = exp(arg0)*exp_series.subs(t, arg_series - arg0)
        r += Order(o.expr.subs(t, (arg_series - arg0)), x)
        r = r.expand()
        return powsimp(r, deep=True, combine='exp')
```
### 31 - sympy/functions/elementary/exponential.py:

Start line: 427, End line: 474

```python
class exp(ExpBase):

    def _taylor(self, x, n):
        from sympy import Order
        l = []
        g = None
        for i in range(n):
            g = self.taylor_term(i, self.args[0], g)
            g = g.nseries(x, n=n)
            l.append(g)
        return Add(*l) + Order(x**n, x)

    def _eval_as_leading_term(self, x):
        from sympy import Order
        arg = self.args[0]
        if arg.is_Add:
            return Mul(*[exp(f).as_leading_term(x) for f in arg.args])
        arg = self.args[0].as_leading_term(x)
        if Order(1, x).contains(arg):
            return S.One
        return exp(arg)

    def _eval_rewrite_as_sin(self, arg, **kwargs):
        from sympy import sin
        I = S.ImaginaryUnit
        return sin(I*arg + S.Pi/2) - I*sin(I*arg)

    def _eval_rewrite_as_cos(self, arg, **kwargs):
        from sympy import cos
        I = S.ImaginaryUnit
        return cos(I*arg) + I*cos(I*arg + S.Pi/2)

    def _eval_rewrite_as_tanh(self, arg, **kwargs):
        from sympy import tanh
        return (1 + tanh(arg/2))/(1 - tanh(arg/2))

    def _eval_rewrite_as_sqrt(self, arg, **kwargs):
        from sympy.functions.elementary.trigonometric import sin, cos
        if arg.is_Mul:
            coeff = arg.coeff(S.Pi*S.ImaginaryUnit)
            if coeff and coeff.is_number:
                cosine, sine = cos(S.Pi*coeff), sin(S.Pi*coeff)
                if not isinstance(cosine, cos) and not isinstance (sine, sin):
                    return cosine + S.ImaginaryUnit*sine

    def _eval_rewrite_as_Pow(self, arg, **kwargs):
        if arg.is_Mul:
            logs = [a for a in arg.args if isinstance(a, log) and len(a.args) == 1]
            if logs:
                return Pow(logs[0].args[0], arg.coeff(logs[0]))
```
### 33 - sympy/functions/elementary/exponential.py:

Start line: 367, End line: 380

```python
class exp(ExpBase):

    def _eval_subs(self, old, new):
        # keep processing of power-like args centralized in Pow
        if old.is_Pow:  # handle (exp(3*log(x))).subs(x**2, z) -> z**(3/2)
            old = exp(old.exp*log(old.base))
        elif old is S.Exp1 and new.is_Function:
            old = exp
        if isinstance(old, exp) or old is S.Exp1:
            f = lambda a: Pow(*a.as_base_exp(), evaluate=False) if (
                a.is_Pow or isinstance(a, exp)) else a
            return Pow._eval_subs(f(self), f(old), new)

        if old is exp and not new.is_Function:
            return new**self.exp._subs(old, new)
        return Function._eval_subs(self, old, new)
```
