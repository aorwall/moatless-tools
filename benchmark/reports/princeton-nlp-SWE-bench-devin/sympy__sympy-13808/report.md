# sympy__sympy-13808

| **sympy/sympy** | `4af4487cfd254af747670a2324b6f24ae0a55a66` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | - |
| **Avg pos** | - |
| **Min pos** | - |
| **Max pos** | - |
| **Top file pos** | 7 |
| **Missing snippets** | 3 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sympy/integrals/integrals.py b/sympy/integrals/integrals.py
--- a/sympy/integrals/integrals.py
+++ b/sympy/integrals/integrals.py
@@ -8,7 +8,7 @@
 from sympy.core.expr import Expr
 from sympy.core.function import diff
 from sympy.core.mul import Mul
-from sympy.core.numbers import oo
+from sympy.core.numbers import oo, pi
 from sympy.core.relational import Eq, Ne
 from sympy.core.singleton import S
 from sympy.core.symbol import (Dummy, Symbol, Wild)
@@ -19,9 +19,10 @@
 from sympy.matrices import MatrixBase
 from sympy.utilities.misc import filldedent
 from sympy.polys import Poly, PolynomialError
-from sympy.functions import Piecewise, sqrt, sign, piecewise_fold
-from sympy.functions.elementary.complexes import Abs, sign
+from sympy.functions import Piecewise, sqrt, sign, piecewise_fold, tan, cot, atan
 from sympy.functions.elementary.exponential import log
+from sympy.functions.elementary.integers import floor
+from sympy.functions.elementary.complexes import Abs, sign
 from sympy.functions.elementary.miscellaneous import Min, Max
 from sympy.series import limit
 from sympy.series.order import Order
@@ -532,6 +533,30 @@ def try_meijerg(function, xab):
                             function = ret
                             continue
 
+            if not isinstance(antideriv, Integral) and antideriv is not None:
+                sym = xab[0]
+                for atan_term in antideriv.atoms(atan):
+                    atan_arg = atan_term.args[0]
+                    # Checking `atan_arg` to be linear combination of `tan` or `cot`
+                    for tan_part in atan_arg.atoms(tan):
+                        x1 = Dummy('x1')
+                        tan_exp1 = atan_arg.subs(tan_part, x1)
+                        # The coefficient of `tan` should be constant
+                        coeff = tan_exp1.diff(x1)
+                        if x1 not in coeff.free_symbols:
+                            a = tan_part.args[0]
+                            antideriv = antideriv.subs(atan_term, Add(atan_term,
+                                sign(coeff)*pi*floor((a-pi/2)/pi)))
+                    for cot_part in atan_arg.atoms(cot):
+                        x1 = Dummy('x1')
+                        cot_exp1 = atan_arg.subs(cot_part, x1)
+                        # The coefficient of `cot` should be constant
+                        coeff = cot_exp1.diff(x1)
+                        if x1 not in coeff.free_symbols:
+                            a = cot_part.args[0]
+                            antideriv = antideriv.subs(atan_term, Add(atan_term,
+                                sign(coeff)*pi*floor((a)/pi)))
+
             if antideriv is None:
                 undone_limits.append(xab)
                 function = self.func(*([function] + [xab])).factor()

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sympy/integrals/integrals.py | 11 | 11 | - | 7 | -
| sympy/integrals/integrals.py | 22 | 23 | - | 7 | -
| sympy/integrals/integrals.py | 535 | 535 | - | 7 | -


## Problem Statement

```
integrate(1/(2-cos(theta)),(theta,0,pi))
Sympy produces NaN.

Actually for integrate(1/(a-cos(theta)),(theta,0,pi)) for a > 1 should be pi/sqrt((a-1)*(a+1)). So, the right answer should be pi/sqrt(3).

However sympy seems to use the subtitution like t = tan(x/2) which is infinite when x = pi. When I try integrate(1/(2-cos(theta)),theta) , I get "sqrt(3)_I_(-log(tan(x/2) - sqrt(3)_I/3) + log(tan(x/2) + sqrt(3)_I/3))/3". Simplify() or trigsimp() doesn't work. And I don't understand why imaginary number appears.

http://www.sympygamma.com/input/?i=integrate%281%2F%282-cos%28x%29%29%2Cx%29
http://www.wolframalpha.com/input/?i=integrate+1%2F%282-cos%28x%29%29+for+x+from+0+to+pi+


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 sympy/integrals/manualintegrate.py | 659 | 684| 445 | 445 | 12060 | 
| 2 | 2 sympy/functions/special/error_functions.py | 1698 | 1798| 770 | 1215 | 31715 | 
| 3 | 2 sympy/integrals/manualintegrate.py | 759 | 818| 702 | 1917 | 31715 | 
| 4 | 2 sympy/integrals/manualintegrate.py | 1149 | 1184| 359 | 2276 | 31715 | 
| 5 | 3 sympy/integrals/trigonometry.py | 139 | 246| 1173 | 3449 | 34917 | 
| 6 | 3 sympy/integrals/manualintegrate.py | 1112 | 1147| 366 | 3815 | 34917 | 
| 7 | 3 sympy/integrals/trigonometry.py | 294 | 334| 381 | 4196 | 34917 | 
| 8 | 3 sympy/integrals/trigonometry.py | 1 | 30| 261 | 4457 | 34917 | 
| 9 | 3 sympy/integrals/manualintegrate.py | 686 | 701| 147 | 4604 | 34917 | 
| 10 | 3 sympy/functions/special/error_functions.py | 1890 | 1986| 716 | 5320 | 34917 | 
| 11 | 3 sympy/integrals/manualintegrate.py | 724 | 743| 206 | 5526 | 34917 | 
| 12 | 3 sympy/integrals/manualintegrate.py | 703 | 722| 175 | 5701 | 34917 | 
| 13 | 3 sympy/integrals/manualintegrate.py | 745 | 757| 156 | 5857 | 34917 | 
| 14 | 3 sympy/functions/special/error_functions.py | 1583 | 1595| 199 | 6056 | 34917 | 
| 15 | 3 sympy/integrals/trigonometry.py | 249 | 291| 402 | 6458 | 34917 | 
| 16 | 3 sympy/integrals/manualintegrate.py | 590 | 658| 731 | 7189 | 34917 | 
| 17 | 3 sympy/integrals/manualintegrate.py | 1097 | 1110| 122 | 7311 | 34917 | 
| 18 | 4 sympy/integrals/__init__.py | 1 | 27| 217 | 7528 | 35135 | 
| 19 | 4 sympy/integrals/manualintegrate.py | 515 | 536| 188 | 7716 | 35135 | 
| 20 | 4 sympy/integrals/manualintegrate.py | 474 | 513| 389 | 8105 | 35135 | 
| 21 | 4 sympy/integrals/manualintegrate.py | 581 | 588| 103 | 8208 | 35135 | 
| 22 | 5 sympy/integrals/quadrature.py | 1 | 11| 118 | 8326 | 40893 | 
| 23 | 6 sympy/functions/special/elliptic_integrals.py | 314 | 352| 384 | 8710 | 44724 | 
| 24 | 6 sympy/integrals/manualintegrate.py | 256 | 278| 212 | 8922 | 44724 | 
| 25 | 6 sympy/functions/special/error_functions.py | 1598 | 1696| 668 | 9590 | 44724 | 
| 26 | **7 sympy/integrals/integrals.py** | 107 | 140| 347 | 9937 | 57294 | 
| 27 | 7 sympy/integrals/manualintegrate.py | 538 | 552| 233 | 10170 | 57294 | 
| 28 | 7 sympy/functions/special/elliptic_integrals.py | 354 | 362| 117 | 10287 | 57294 | 
| 29 | 7 sympy/integrals/trigonometry.py | 33 | 125| 841 | 11128 | 57294 | 
| 30 | 8 sympy/integrals/singularityfunctions.py | 1 | 65| 648 | 11776 | 57943 | 
| 31 | 8 sympy/functions/special/elliptic_integrals.py | 245 | 270| 302 | 12078 | 57943 | 
| 32 | 8 sympy/integrals/manualintegrate.py | 307 | 324| 292 | 12370 | 57943 | 
| 33 | 8 sympy/functions/special/error_functions.py | 1570 | 1581| 120 | 12490 | 57943 | 
| 34 | 9 sympy/functions/elementary/trigonometric.py | 750 | 786| 767 | 13257 | 81285 | 
| 35 | 9 sympy/functions/special/elliptic_integrals.py | 1 | 12| 115 | 13372 | 81285 | 
| 36 | 10 sympy/integrals/deltafunctions.py | 139 | 199| 670 | 14042 | 83113 | 
| 37 | 11 sympy/integrals/heurisch.py | 1 | 33| 325 | 14367 | 89190 | 
| 38 | 11 sympy/integrals/manualintegrate.py | 117 | 172| 415 | 14782 | 89190 | 
| 39 | 11 sympy/functions/special/error_functions.py | 2311 | 2333| 345 | 15127 | 89190 | 
| 40 | 12 sympy/integrals/rubi/rules/integrand_simplification.py | 1 | 19| 509 | 15636 | 98745 | 
| 41 | 13 sympy/integrals/meijerint.py | 179 | 228| 808 | 16444 | 122699 | 
| 42 | 13 sympy/integrals/manualintegrate.py | 346 | 410| 606 | 17050 | 122699 | 
| 43 | **13 sympy/integrals/integrals.py** | 1022 | 1049| 255 | 17305 | 122699 | 
| 44 | 13 sympy/integrals/manualintegrate.py | 554 | 579| 269 | 17574 | 122699 | 
| 45 | 13 sympy/functions/special/error_functions.py | 2031 | 2054| 206 | 17780 | 122699 | 
| 46 | 13 sympy/integrals/trigonometry.py | 126 | 137| 185 | 17965 | 122699 | 
| 47 | 13 sympy/functions/special/elliptic_integrals.py | 364 | 386| 332 | 18297 | 122699 | 


## Patch

```diff
diff --git a/sympy/integrals/integrals.py b/sympy/integrals/integrals.py
--- a/sympy/integrals/integrals.py
+++ b/sympy/integrals/integrals.py
@@ -8,7 +8,7 @@
 from sympy.core.expr import Expr
 from sympy.core.function import diff
 from sympy.core.mul import Mul
-from sympy.core.numbers import oo
+from sympy.core.numbers import oo, pi
 from sympy.core.relational import Eq, Ne
 from sympy.core.singleton import S
 from sympy.core.symbol import (Dummy, Symbol, Wild)
@@ -19,9 +19,10 @@
 from sympy.matrices import MatrixBase
 from sympy.utilities.misc import filldedent
 from sympy.polys import Poly, PolynomialError
-from sympy.functions import Piecewise, sqrt, sign, piecewise_fold
-from sympy.functions.elementary.complexes import Abs, sign
+from sympy.functions import Piecewise, sqrt, sign, piecewise_fold, tan, cot, atan
 from sympy.functions.elementary.exponential import log
+from sympy.functions.elementary.integers import floor
+from sympy.functions.elementary.complexes import Abs, sign
 from sympy.functions.elementary.miscellaneous import Min, Max
 from sympy.series import limit
 from sympy.series.order import Order
@@ -532,6 +533,30 @@ def try_meijerg(function, xab):
                             function = ret
                             continue
 
+            if not isinstance(antideriv, Integral) and antideriv is not None:
+                sym = xab[0]
+                for atan_term in antideriv.atoms(atan):
+                    atan_arg = atan_term.args[0]
+                    # Checking `atan_arg` to be linear combination of `tan` or `cot`
+                    for tan_part in atan_arg.atoms(tan):
+                        x1 = Dummy('x1')
+                        tan_exp1 = atan_arg.subs(tan_part, x1)
+                        # The coefficient of `tan` should be constant
+                        coeff = tan_exp1.diff(x1)
+                        if x1 not in coeff.free_symbols:
+                            a = tan_part.args[0]
+                            antideriv = antideriv.subs(atan_term, Add(atan_term,
+                                sign(coeff)*pi*floor((a-pi/2)/pi)))
+                    for cot_part in atan_arg.atoms(cot):
+                        x1 = Dummy('x1')
+                        cot_exp1 = atan_arg.subs(cot_part, x1)
+                        # The coefficient of `cot` should be constant
+                        coeff = cot_exp1.diff(x1)
+                        if x1 not in coeff.free_symbols:
+                            a = cot_part.args[0]
+                            antideriv = antideriv.subs(atan_term, Add(atan_term,
+                                sign(coeff)*pi*floor((a)/pi)))
+
             if antideriv is None:
                 undone_limits.append(xab)
                 function = self.func(*([function] + [xab])).factor()

```

## Test Patch

```diff
diff --git a/sympy/integrals/tests/test_integrals.py b/sympy/integrals/tests/test_integrals.py
--- a/sympy/integrals/tests/test_integrals.py
+++ b/sympy/integrals/tests/test_integrals.py
@@ -8,6 +8,7 @@
     symbols, sympify, tan, trigsimp, Tuple, Si, Ci
 )
 from sympy.functions.elementary.complexes import periodic_argument
+from sympy.functions.elementary.integers import floor
 from sympy.integrals.risch import NonElementaryIntegral
 from sympy.physics import units
 from sympy.core.compatibility import range
@@ -316,6 +317,22 @@ def test_issue_7450():
     assert re(ans) == S.Half and im(ans) == -S.Half
 
 
+def test_issue_8623():
+    assert integrate((1 + cos(2*x)) / (3 - 2*cos(2*x)), (x, 0, pi)) == -pi/2 + sqrt(5)*pi/2
+    assert integrate((1 + cos(2*x))/(3 - 2*cos(2*x))) == -x/2 + sqrt(5)*(atan(sqrt(5)*tan(x)) + \
+        pi*floor((x - pi/2)/pi))/2
+
+
+def test_issue_9569():
+    assert integrate(1 / (2 - cos(x)), (x, 0, pi)) == pi/sqrt(3)
+    assert integrate(1/(2 - cos(x))) == 2*sqrt(3)*(atan(sqrt(3)*tan(x/2)) + pi*floor((x/2 - pi/2)/pi))/3
+
+
+def test_issue_13749():
+    assert integrate(1 / (2 + cos(x)), (x, 0, pi)) == pi/sqrt(3)
+    assert integrate(1/(2 + cos(x))) == 2*sqrt(3)*(atan(sqrt(3)*tan(x/2)/3) + pi*floor((x/2 - pi/2)/pi))/3
+
+
 def test_matrices():
     M = Matrix(2, 2, lambda i, j: (i + j + 1)*sin((i + j + 1)*x))
 
@@ -1161,7 +1178,7 @@ def test_issue_4803():
 
 
 def test_issue_4234():
-    assert integrate(1/sqrt(1 + tan(x)**2)) == tan(x) / sqrt(1 + tan(x)**2)
+    assert integrate(1/sqrt(1 + tan(x)**2)) == tan(x)/sqrt(1 + tan(x)**2)
 
 
 def test_issue_4492():

```


## Code snippets

### 1 - sympy/integrals/manualintegrate.py:

Start line: 659, End line: 684

```python
tansec_seceven = trig_rewriter(
    lambda a, b, m, n, i, symbol: ( (1 + sympy.tan(b*symbol)**2) ** (n/2 - 1) *
                                    sympy.sec(b*symbol)**2 *
                                    sympy.tan(a*symbol) ** m ))

tansec_tanodd_condition = uncurry(lambda a, b, m, n, i, s: m.is_odd)
tansec_tanodd = trig_rewriter(
    lambda a, b, m, n, i, symbol: ( (sympy.sec(a*symbol)**2 - 1) ** ((m - 1) / 2) *
                                     sympy.tan(a*symbol) *
                                     sympy.sec(b*symbol) ** n ))

tan_tansquared_condition = uncurry(lambda a, b, m, n, i, s: m == 2 and n == 0)
tan_tansquared = trig_rewriter(
    lambda a, b, m, n, i, symbol: ( sympy.sec(a*symbol)**2 - 1))

cotcsc_csceven_condition = uncurry(lambda a, b, m, n, i, s: n.is_even and n >= 4)
cotcsc_csceven = trig_rewriter(
    lambda a, b, m, n, i, symbol: ( (1 + sympy.cot(b*symbol)**2) ** (n/2 - 1) *
                                    sympy.csc(b*symbol)**2 *
                                    sympy.cot(a*symbol) ** m ))

cotcsc_cotodd_condition = uncurry(lambda a, b, m, n, i, s: m.is_odd)
cotcsc_cotodd = trig_rewriter(
    lambda a, b, m, n, i, symbol: ( (sympy.csc(a*symbol)**2 - 1) ** ((m - 1) / 2) *
                                    sympy.cot(a*symbol) *
                                    sympy.csc(b*symbol) ** n ))
```
### 2 - sympy/functions/special/error_functions.py:

Start line: 1698, End line: 1798

```python
class Ci(TrigonometricIntegral):
    r"""
    Cosine integral.

    This function is defined for positive `x` by

    .. math:: \operatorname{Ci}(x) = \gamma + \log{x}
                         + \int_0^x \frac{\cos{t} - 1}{t} \mathrm{d}t
           = -\int_x^\infty \frac{\cos{t}}{t} \mathrm{d}t,

    where `\gamma` is the Euler-Mascheroni constant.

    We have

    .. math:: \operatorname{Ci}(z) =
        -\frac{\operatorname{E}_1\left(e^{i\pi/2} z\right)
               + \operatorname{E}_1\left(e^{-i \pi/2} z\right)}{2}

    which holds for all polar `z` and thus provides an analytic
    continuation to the Riemann surface of the logarithm.

    The formula also holds as stated
    for `z \in \mathbb{C}` with `\Re(z) > 0`.
    By lifting to the principal branch we obtain an analytic function on the
    cut complex plane.

    Examples
    ========

    >>> from sympy import Ci
    >>> from sympy.abc import z

    The cosine integral is a primitive of `\cos(z)/z`:

    >>> Ci(z).diff(z)
    cos(z)/z

    It has a logarithmic branch point at the origin:

    >>> from sympy import exp_polar, I, pi
    >>> Ci(z*exp_polar(2*I*pi))
    Ci(z) + 2*I*pi

    The cosine integral behaves somewhat like ordinary `\cos` under multiplication by `i`:

    >>> from sympy import polar_lift
    >>> Ci(polar_lift(I)*z)
    Chi(z) + I*pi/2
    >>> Ci(polar_lift(-1)*z)
    Ci(z) + I*pi

    It can also be expressed in terms of exponential integrals:

    >>> from sympy import expint
    >>> Ci(z).rewrite(expint)
    -expint(1, z*exp_polar(-I*pi/2))/2 - expint(1, z*exp_polar(I*pi/2))/2

    See Also
    ========

    Si: Sine integral.
    Shi: Hyperbolic sine integral.
    Chi: Hyperbolic cosine integral.
    Ei: Exponential integral.
    expint: Generalised exponential integral.
    E1: Special case of the generalised exponential integral.
    li: Logarithmic integral.
    Li: Offset logarithmic integral.

    References
    ==========

    .. [1] http://en.wikipedia.org/wiki/Trigonometric_integral

    """

    _trigfunc = cos
    _atzero = S.ComplexInfinity

    @classmethod
    def _atinf(cls):
        return S.Zero

    @classmethod
    def _atneginf(cls):
        return I*pi

    @classmethod
    def _minusfactor(cls, z):
        return Ci(z) + I*pi

    @classmethod
    def _Ifactor(cls, z, sign):
        return Chi(z) + I*pi/2*sign

    def _eval_rewrite_as_expint(self, z):
        return -(E1(polar_lift(I)*z) + E1(polar_lift(-I)*z))/2

    def _sage_(self):
        import sage.all as sage
        return sage.cos_integral(self.args[0]._sage_())
```
### 3 - sympy/integrals/manualintegrate.py:

Start line: 759, End line: 818

```python
def trig_substitution_rule(integral):
    integrand, symbol = integral
    A = sympy.Wild('a', exclude=[0, symbol])
    B = sympy.Wild('b', exclude=[0, symbol])
    theta = sympy.Dummy("theta")
    target_pattern = A + B*symbol**2

    matches = integrand.find(target_pattern)
    for expr in matches:
        match = expr.match(target_pattern)
        a = match.get(A, ZERO)
        b = match.get(B, ZERO)

        a_positive = ((a.is_number and a > 0) or a.is_positive)
        b_positive = ((b.is_number and b > 0) or b.is_positive)
        a_negative = ((a.is_number and a < 0) or a.is_negative)
        b_negative = ((b.is_number and b < 0) or b.is_negative)
        x_func = None
        if a_positive and b_positive:
            # a**2 + b*x**2. Assume sec(theta) > 0, -pi/2 < theta < pi/2
            x_func = (sympy.sqrt(a)/sympy.sqrt(b)) * sympy.tan(theta)
            # Do not restrict the domain: tan(theta) takes on any real
            # value on the interval -pi/2 < theta < pi/2 so x takes on
            # any value
            restriction = True
        elif a_positive and b_negative:
            # a**2 - b*x**2. Assume cos(theta) > 0, -pi/2 < theta < pi/2
            constant = sympy.sqrt(a)/sympy.sqrt(-b)
            x_func = constant * sympy.sin(theta)
            restriction = sympy.And(symbol > -constant, symbol < constant)
        elif a_negative and b_positive:
            # b*x**2 - a**2. Assume sin(theta) > 0, 0 < theta < pi
            constant = sympy.sqrt(-a)/sympy.sqrt(b)
            x_func = constant * sympy.sec(theta)
            restriction = sympy.And(symbol > -constant, symbol < constant)
        if x_func:
            # Manually simplify sqrt(trig(theta)**2) to trig(theta)
            # Valid due to assumed domain restriction
            substitutions = {}
            for f in [sympy.sin, sympy.cos, sympy.tan,
                      sympy.sec, sympy.csc, sympy.cot]:
                substitutions[sympy.sqrt(f(theta)**2)] = f(theta)
                substitutions[sympy.sqrt(f(theta)**(-2))] = 1/f(theta)

            replaced = integrand.subs(symbol, x_func).trigsimp()
            replaced = replaced.subs(substitutions)
            if not replaced.has(symbol):
                replaced *= manual_diff(x_func, theta)
                replaced = replaced.trigsimp()
                secants = replaced.find(1/sympy.cos(theta))
                if secants:
                    replaced = replaced.xreplace({
                        1/sympy.cos(theta): sympy.sec(theta)
                    })

                substep = integral_steps(replaced, theta)
                if not contains_dont_know(substep):
                    return TrigSubstitutionRule(
                        theta, x_func, replaced, substep, restriction,
                        integrand, symbol)
```
### 4 - sympy/integrals/manualintegrate.py:

Start line: 1149, End line: 1184

```python
@evaluates(TrigSubstitutionRule)
def eval_trigsubstitution(theta, func, rewritten, substep, restriction, integrand, symbol):
    func = func.subs(sympy.sec(theta), 1/sympy.cos(theta))

    trig_function = list(func.find(TrigonometricFunction))
    assert len(trig_function) == 1
    trig_function = trig_function[0]
    relation = sympy.solve(symbol - func, trig_function)
    assert len(relation) == 1
    numer, denom = sympy.fraction(relation[0])

    if isinstance(trig_function, sympy.sin):
        opposite = numer
        hypotenuse = denom
        adjacent = sympy.sqrt(denom**2 - numer**2)
        inverse = sympy.asin(relation[0])
    elif isinstance(trig_function, sympy.cos):
        adjacent = numer
        hypotenuse = denom
        opposite = sympy.sqrt(denom**2 - numer**2)
        inverse = sympy.acos(relation[0])
    elif isinstance(trig_function, sympy.tan):
        opposite = numer
        adjacent = denom
        hypotenuse = sympy.sqrt(denom**2 + numer**2)
        inverse = sympy.atan(relation[0])

    substitution = [
        (sympy.sin(theta), opposite/hypotenuse),
        (sympy.cos(theta), adjacent/hypotenuse),
        (sympy.tan(theta), opposite/adjacent),
        (theta, inverse)
    ]
    return sympy.Piecewise(
        (_manualintegrate(substep).subs(substitution).trigsimp(), restriction)
    )
```
### 5 - sympy/integrals/trigonometry.py:

Start line: 139, End line: 246

```python
def trigintegrate(f, x, conds='piecewise'):
    # ... other code

    if n_:
        #  2k         2 k             i             2i
        # C   = (1 - S )  = sum(i, (-) * B(k, i) * S  )
        if m > 0:
            for i in range(0, m//2 + 1):
                res += ((-1)**i * binomial(m//2, i) *
                        _sin_pow_integrate(n + 2*i, x))

        elif m == 0:
            res = _sin_pow_integrate(n, x)
        else:

            # m < 0 , |n| > |m|
            #  /
            # |
            # |    m       n
            # | cos (x) sin (x) dx =
            # |
            # |
            #/
            #                                      /
            #                                     |
            #   -1        m+1     n-1     n - 1   |     m+2     n-2
            # ________ cos (x) sin (x) + _______  |  cos (x) sin (x) dx
            #                                     |
            #   m + 1                     m + 1   |
            #                                    /

            res = (Rational(-1, m + 1) * cos(x)**(m + 1) * sin(x)**(n - 1) +
                   Rational(n - 1, m + 1) *
                   trigintegrate(cos(x)**(m + 2)*sin(x)**(n - 2), x))

    elif m_:
        #  2k         2 k            i             2i
        # S   = (1 - C ) = sum(i, (-) * B(k, i) * C  )
        if n > 0:

            #      /                            /
            #     |                            |
            #     |    m       n               |    -m         n
            #     | cos (x)*sin (x) dx  or     | cos (x) * sin (x) dx
            #     |                            |
            #    /                            /
            #
            #    |m| > |n| ; m, n >0 ; m, n belong to Z - {0}
            #       n                                         2
            #    sin (x) term is expanded here in terms of cos (x),
            #    and then integrated.
            #

            for i in range(0, n//2 + 1):
                res += ((-1)**i * binomial(n//2, i) *
                        _cos_pow_integrate(m + 2*i, x))

        elif n == 0:

            #   /
            #  |
            #  |  1
            #  | _ _ _
            #  |    m
            #  | cos (x)
            # /
            #

            res = _cos_pow_integrate(m, x)
        else:

            # n < 0 , |m| > |n|
            #  /
            # |
            # |    m       n
            # | cos (x) sin (x) dx =
            # |
            # |
            #/
            #                                      /
            #                                     |
            #    1        m-1     n+1     m - 1   |     m-2     n+2
            #  _______ cos (x) sin (x) + _______  |  cos (x) sin (x) dx
            #                                     |
            #   n + 1                     n + 1   |
            #                                    /

            res = (Rational(1, n + 1) * cos(x)**(m - 1)*sin(x)**(n + 1) +
                   Rational(m - 1, n + 1) *
                   trigintegrate(cos(x)**(m - 2)*sin(x)**(n + 2), x))

    else:
        if m == n:
            ##Substitute sin(2x)/2 for sin(x)cos(x) and then Integrate.
            res = integrate((Rational(1, 2)*sin(2*x))**m, x)
        elif (m == -n):
            if n < 0:
                # Same as the scheme described above.
                # the function argument to integrate in the end will
                # be 1 , this cannot be integrated by trigintegrate.
                # Hence use sympy.integrals.integrate.
                res = (Rational(1, n + 1) * cos(x)**(m - 1) * sin(x)**(n + 1) +
                       Rational(m - 1, n + 1) *
                       integrate(cos(x)**(m - 2) * sin(x)**(n + 2), x))
            else:
                res = (Rational(-1, m + 1) * cos(x)**(m + 1) * sin(x)**(n - 1) +
                       Rational(n - 1, m + 1) *
                       integrate(cos(x)**(m + 2)*sin(x)**(n - 2), x))
    if conds == 'piecewise':
        return Piecewise((res.subs(x, a*x) / a, Ne(a, 0)), (zz, True))
    return res.subs(x, a*x) / a
```
### 6 - sympy/integrals/manualintegrate.py:

Start line: 1112, End line: 1147

```python
@evaluates(ArctanRule)
def eval_arctan(a, b, c, integrand, symbol):
    return a / b * 1 / sympy.sqrt(c / b) * sympy.atan(symbol / sympy.sqrt(c / b))

@evaluates(ArccothRule)
def eval_arccoth(a, b, c, integrand, symbol):
    return - a / b * 1 / sympy.sqrt(-c / b) * sympy.acoth(symbol / sympy.sqrt(-c / b))

@evaluates(ArctanhRule)
def eval_arctanh(a, b, c, integrand, symbol):
    return - a / b * 1 / sympy.sqrt(-c / b) * sympy.atanh(symbol / sympy.sqrt(-c / b))

@evaluates(ReciprocalRule)
def eval_reciprocal(func, integrand, symbol):
    return sympy.ln(func)

@evaluates(ArcsinRule)
def eval_arcsin(integrand, symbol):
    return sympy.asin(symbol)

@evaluates(InverseHyperbolicRule)
def eval_inversehyperbolic(func, integrand, symbol):
    return func(symbol)

@evaluates(AlternativeRule)
def eval_alternative(alternatives, integrand, symbol):
    return _manualintegrate(alternatives[0])

@evaluates(RewriteRule)
def eval_rewrite(rewritten, substep, integrand, symbol):
    return _manualintegrate(substep)

@evaluates(PiecewiseRule)
def eval_piecewise(substeps, integrand, symbol):
    return sympy.Piecewise(*[(_manualintegrate(substep), cond)
                             for substep, cond in substeps])
```
### 7 - sympy/integrals/trigonometry.py:

Start line: 294, End line: 334

```python
def _cos_pow_integrate(n, x):
    if n > 0:
        if n == 1:
            #Recursion break.
            return sin(x)

        # n > 0
        #  /                                                 /
        # |                                                 |
        # |    n            1               n-1     n - 1   |     n-2
        # | sin (x) dx =  ______ sin (x) cos (x) + _______  |  cos (x) dx
        # |                                                 |
        # |                 n                         n     |
        #/                                                 /
        #

        return (Rational(1, n) * sin(x) * cos(x)**(n - 1) +
                Rational(n - 1, n) * _cos_pow_integrate(n - 2, x))

    if n < 0:
        if n == -1:
            ##Recursion break
            return trigintegrate(1/cos(x), x)

        # n < 0
        #  /                                                 /
        # |                                                 |
        # |    n            -1              n+1     n + 2   |     n+2
        # | cos (x) dx = _______ sin (x) cos (x) + _______  |  cos (x) dx
        # |                                                 |
        # |               n + 1                     n + 1   |
        #/                                                 /
        #

        return (Rational(-1, n + 1) * sin(x) * cos(x)**(n + 1) +
                Rational(n + 2, n + 1) * _cos_pow_integrate(n + 2, x))
    else:
        # n == 0
        #Recursion Break.
        return x
```
### 8 - sympy/integrals/trigonometry.py:

Start line: 1, End line: 30

```python
# -*- coding: utf-8 -*-

from __future__ import print_function, division

from sympy.core.compatibility import range
from sympy.core import cacheit, Dummy, Ne, Integer, Rational, S, Wild
from sympy.functions import binomial, sin, cos, Piecewise

# TODO sin(a*x)*cos(b*x) -> sin((a+b)x) + sin((a-b)x) ?

# creating, each time, Wild's and sin/cos/Mul is expensive. Also, our match &
# subs are very slow when not cached, and if we create Wild each time, we
# effectively block caching.
#
# so we cache the pattern

# need to use a function instead of lamda since hash of lambda changes on
# each call to _pat_sincos
def _integer_instance(n):
    return isinstance(n , Integer)

@cacheit
def _pat_sincos(x):
    a = Wild('a', exclude=[x])
    n, m = [Wild(s, exclude=[x], properties=[_integer_instance])
                for s in 'nm']
    pat = sin(a*x)**n * cos(a*x)**m
    return pat, a, n, m

_u = Dummy('u')
```
### 9 - sympy/integrals/manualintegrate.py:

Start line: 686, End line: 701

```python
def trig_sincos_rule(integral):
    integrand, symbol = integral

    if any(integrand.has(f) for f in (sympy.sin, sympy.cos)):
        pattern, a, b, m, n = sincos_pattern(symbol)
        match = integrand.match(pattern)
        if not match:
            return

        return multiplexer({
            sincos_botheven_condition: sincos_botheven,
            sincos_sinodd_condition: sincos_sinodd,
            sincos_cosodd_condition: sincos_cosodd
        })(tuple(
            [match.get(i, ZERO) for i in (a, b, m, n)] +
            [integrand, symbol]))
```
### 10 - sympy/functions/special/error_functions.py:

Start line: 1890, End line: 1986

```python
class Chi(TrigonometricIntegral):
    r"""
    Cosh integral.

    This function is defined for positive :math:`x` by

    .. math:: \operatorname{Chi}(x) = \gamma + \log{x}
                         + \int_0^x \frac{\cosh{t} - 1}{t} \mathrm{d}t,

    where :math:`\gamma` is the Euler-Mascheroni constant.

    We have

    .. math:: \operatorname{Chi}(z) = \operatorname{Ci}\left(e^{i \pi/2}z\right)
                         - i\frac{\pi}{2},

    which holds for all polar :math:`z` and thus provides an analytic
    continuation to the Riemann surface of the logarithm.
    By lifting to the principal branch we obtain an analytic function on the
    cut complex plane.

    Examples
    ========

    >>> from sympy import Chi
    >>> from sympy.abc import z

    The `\cosh` integral is a primitive of `\cosh(z)/z`:

    >>> Chi(z).diff(z)
    cosh(z)/z

    It has a logarithmic branch point at the origin:

    >>> from sympy import exp_polar, I, pi
    >>> Chi(z*exp_polar(2*I*pi))
    Chi(z) + 2*I*pi

    The `\cosh` integral behaves somewhat like ordinary `\cosh` under multiplication by `i`:

    >>> from sympy import polar_lift
    >>> Chi(polar_lift(I)*z)
    Ci(z) + I*pi/2
    >>> Chi(polar_lift(-1)*z)
    Chi(z) + I*pi

    It can also be expressed in terms of exponential integrals:

    >>> from sympy import expint
    >>> Chi(z).rewrite(expint)
    -expint(1, z)/2 - expint(1, z*exp_polar(I*pi))/2 - I*pi/2

    See Also
    ========

    Si: Sine integral.
    Ci: Cosine integral.
    Shi: Hyperbolic sine integral.
    Ei: Exponential integral.
    expint: Generalised exponential integral.
    E1: Special case of the generalised exponential integral.
    li: Logarithmic integral.
    Li: Offset logarithmic integral.

    References
    ==========

    .. [1] http://en.wikipedia.org/wiki/Trigonometric_integral

    """

    _trigfunc = cosh
    _atzero = S.ComplexInfinity

    @classmethod
    def _atinf(cls):
        return S.Infinity

    @classmethod
    def _atneginf(cls):
        return S.Infinity

    @classmethod
    def _minusfactor(cls, z):
        return Chi(z) + I*pi

    @classmethod
    def _Ifactor(cls, z, sign):
        return Ci(z) + I*pi/2*sign

    def _eval_rewrite_as_expint(self, z):
        from sympy import exp_polar
        return -I*pi/2 - (E1(z) + E1(exp_polar(I*pi)*z))/2

    def _sage_(self):
        import sage.all as sage
        return sage.cosh_integral(self.args[0]._sage_())
```
### 26 - sympy/integrals/integrals.py:

Start line: 107, End line: 140

```python
class Integral(AddWithLimits):

    def _eval_is_zero(self):
        # This is a very naive and quick test, not intended to do the integral to
        # answer whether it is zero or not, e.g. Integral(sin(x), (x, 0, 2*pi))
        # is zero but this routine should return None for that case. But, like
        # Mul, there are trivial situations for which the integral will be
        # zero so we check for those.
        if self.function.is_zero:
            return True
        got_none = False
        for l in self.limits:
            if len(l) == 3:
                z = (l[1] == l[2]) or (l[1] - l[2]).is_zero
                if z:
                    return True
                elif z is None:
                    got_none = True
        free = self.function.free_symbols
        for xab in self.limits:
            if len(xab) == 1:
                free.add(xab[0])
                continue
            if len(xab) == 2 and xab[0] not in free:
                if xab[1].is_zero:
                    return True
                elif xab[1].is_zero is None:
                    got_none = True
            # take integration symbol out of free since it will be replaced
            # with the free symbols in the limits
            free.discard(xab[0])
            # add in the new symbols
            for i in xab[1:]:
                free.update(i.free_symbols)
        if self.function.is_zero is False and got_none is False:
            return False
```
### 43 - sympy/integrals/integrals.py:

Start line: 1022, End line: 1049

```python
class Integral(AddWithLimits):

    def _eval_lseries(self, x, logx):
        expr = self.as_dummy()
        symb = x
        for l in expr.limits:
            if x in l[1:]:
                symb = l[0]
                break
        for term in expr.function.lseries(symb, logx):
            yield integrate(term, *expr.limits)

    def _eval_nseries(self, x, n, logx):
        expr = self.as_dummy()
        symb = x
        for l in expr.limits:
            if x in l[1:]:
                symb = l[0]
                break
        terms, order = expr.function.nseries(
            x=symb, n=n, logx=logx).as_coeff_add(Order)
        order = [o.subs(symb, x) for o in order]
        return integrate(terms, *expr.limits) + Add(*order)*x

    def _eval_as_leading_term(self, x):
        series_gen = self.args[0].lseries(x)
        for leading_term in series_gen:
            if leading_term != 0:
                break
        return integrate(leading_term, *self.args[1:])
```
