# sympy__sympy-16334

| **sympy/sympy** | `356a73cd676e0c3f1a1c3057a6895db0d82a1be7` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | - |
| **Avg pos** | - |
| **Min pos** | - |
| **Max pos** | - |
| **Top file pos** | 7 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sympy/core/power.py b/sympy/core/power.py
--- a/sympy/core/power.py
+++ b/sympy/core/power.py
@@ -437,6 +437,9 @@ def _eval_is_positive(self):
                 return True
             if self.exp.is_odd:
                 return False
+        elif self.base.is_zero:
+            if self.exp.is_real:
+                return self.exp.is_zero
         elif self.base.is_nonpositive:
             if self.exp.is_odd:
                 return False
@@ -459,6 +462,9 @@ def _eval_is_negative(self):
         elif self.base.is_positive:
             if self.exp.is_real:
                 return False
+        elif self.base.is_zero:
+            if self.exp.is_real:
+                return False
         elif self.base.is_nonnegative:
             if self.exp.is_nonnegative:
                 return False

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sympy/core/power.py | 440 | 440 | - | 7 | -
| sympy/core/power.py | 462 | 462 | - | 7 | -


## Problem Statement

```
S(0)**real(!=0) should be (0 or zoo) and hence non-positive. 
Consider the following code from master:
\`\`\`py
>>> from sympy import symbols, ask, Q
>>> from sympy.abc import x,y,z
>>> p = symbols('p', real=True, zero=False)
>>> q = symbols('q', zero=True)
>>> (q**p).is_positive
>>>
\`\`\`
Since `0**a`(where a is real and non-zero) should always be (0 or `zoo`). Therefore, the result should have been `False`.


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 sympy/assumptions/ask.py | 497 | 535| 294 | 294 | 10797 | 
| 2 | 1 sympy/assumptions/ask.py | 71 | 126| 490 | 784 | 10797 | 
| 3 | 1 sympy/assumptions/ask.py | 395 | 431| 331 | 1115 | 10797 | 
| 4 | 1 sympy/assumptions/ask.py | 537 | 568| 265 | 1380 | 10797 | 
| 5 | 2 sympy/assumptions/handlers/sets.py | 234 | 287| 560 | 1940 | 15521 | 
| 6 | 2 sympy/assumptions/ask.py | 862 | 892| 248 | 2188 | 15521 | 
| 7 | 2 sympy/assumptions/ask.py | 570 | 601| 275 | 2463 | 15521 | 
| 8 | 2 sympy/assumptions/ask.py | 471 | 495| 153 | 2616 | 15521 | 
| 9 | 2 sympy/assumptions/ask.py | 1046 | 1063| 113 | 2729 | 15521 | 
| 10 | 2 sympy/assumptions/ask.py | 433 | 469| 343 | 3072 | 15521 | 
| 11 | 3 sympy/functions/elementary/complexes.py | 93 | 118| 243 | 3315 | 24582 | 
| 12 | 4 sympy/assumptions/handlers/order.py | 217 | 245| 217 | 3532 | 26927 | 
| 13 | 4 sympy/assumptions/handlers/order.py | 247 | 366| 799 | 4331 | 26927 | 
| 14 | 4 sympy/assumptions/ask.py | 128 | 150| 143 | 4474 | 26927 | 
| 15 | 5 sympy/physics/secondquant.py | 343 | 390| 228 | 4702 | 49486 | 
| 16 | 6 sympy/core/mul.py | 1275 | 1288| 156 | 4858 | 64255 | 
| 17 | **7 sympy/core/power.py** | 157 | 239| 1060 | 5918 | 78846 | 
| 18 | 7 sympy/assumptions/ask.py | 152 | 178| 165 | 6083 | 78846 | 
| 19 | 8 sympy/polys/subresultants_qq_zz.py | 1704 | 1765| 548 | 6631 | 105140 | 
| 20 | 8 sympy/assumptions/handlers/sets.py | 289 | 330| 289 | 6920 | 105140 | 
| 21 | 8 sympy/assumptions/handlers/order.py | 97 | 136| 263 | 7183 | 105140 | 
| 22 | 9 sympy/core/expr.py | 2574 | 2634| 448 | 7631 | 134818 | 
| 23 | 9 sympy/assumptions/handlers/order.py | 139 | 184| 290 | 7921 | 134818 | 
| 24 | 9 sympy/assumptions/handlers/sets.py | 486 | 534| 419 | 8340 | 134818 | 
| 25 | 9 sympy/assumptions/ask.py | 208 | 233| 188 | 8528 | 134818 | 
| 26 | 10 sympy/core/evalf.py | 232 | 255| 215 | 8743 | 148499 | 
| 27 | 11 sympy/matrices/dense.py | 1 | 24| 200 | 8943 | 160367 | 
| 28 | 11 sympy/physics/secondquant.py | 1834 | 1856| 198 | 9141 | 160367 | 
| 29 | 12 sympy/solvers/solvers.py | 454 | 907| 4459 | 13600 | 192640 | 


## Patch

```diff
diff --git a/sympy/core/power.py b/sympy/core/power.py
--- a/sympy/core/power.py
+++ b/sympy/core/power.py
@@ -437,6 +437,9 @@ def _eval_is_positive(self):
                 return True
             if self.exp.is_odd:
                 return False
+        elif self.base.is_zero:
+            if self.exp.is_real:
+                return self.exp.is_zero
         elif self.base.is_nonpositive:
             if self.exp.is_odd:
                 return False
@@ -459,6 +462,9 @@ def _eval_is_negative(self):
         elif self.base.is_positive:
             if self.exp.is_real:
                 return False
+        elif self.base.is_zero:
+            if self.exp.is_real:
+                return False
         elif self.base.is_nonnegative:
             if self.exp.is_nonnegative:
                 return False

```

## Test Patch

```diff
diff --git a/sympy/core/tests/test_assumptions.py b/sympy/core/tests/test_assumptions.py
--- a/sympy/core/tests/test_assumptions.py
+++ b/sympy/core/tests/test_assumptions.py
@@ -786,6 +786,32 @@ def test_Mul_is_prime_composite():
     assert ( (x+1)*(y+1) ).is_prime is None
     assert ( (x+1)*(y+1) ).is_composite is None
 
+
+def test_Pow_is_pos_neg():
+    z = Symbol('z', real=True)
+    w = Symbol('w', nonpositive=True)
+
+    assert (S(-1)**S(2)).is_positive is True
+    assert (S(1)**z).is_positive is True
+    assert (S(-1)**S(3)).is_positive is False
+    assert (S(0)**S(0)).is_positive is True  # 0**0 is 1
+    assert (w**S(3)).is_positive is False
+    assert (w**S(2)).is_positive is None
+    assert (I**2).is_positive is False
+    assert (I**4).is_positive is True
+
+    # tests emerging from #16332 issue
+    p = Symbol('p', zero=True)
+    q = Symbol('q', zero=False, real=True)
+    j = Symbol('j', zero=False, even=True)
+    x = Symbol('x', zero=True)
+    y = Symbol('y', zero=True)
+    assert (p**q).is_positive is False
+    assert (p**q).is_negative is False
+    assert (p**j).is_positive is False
+    assert (x**y).is_positive is True   # 0**0
+    assert (x**y).is_negative is False
+
 def test_Pow_is_prime_composite():
     from sympy import Pow
     x = Symbol('x', positive=True, integer=True)

```


## Code snippets

### 1 - sympy/assumptions/ask.py:

Start line: 497, End line: 535

```python
class AssumptionKeys(object):

    @predicate_memo
    def nonzero(self):
        """
        Nonzero real number predicate.

        ``ask(Q.nonzero(x))`` is true iff ``x`` is real and ``x`` is not zero.  Note in
        particular that ``Q.nonzero(x)`` is false if ``x`` is not real.  Use
        ``~Q.zero(x)`` if you want the negation of being zero without any real
        assumptions.

        A few important facts about nonzero numbers:

        - ``Q.nonzero`` is logically equivalent to ``Q.positive | Q.negative``.

        - See the documentation of ``Q.real`` for more information about
          related facts.

        Examples
        ========

        >>> from sympy import Q, ask, symbols, I, oo
        >>> x = symbols('x')
        >>> print(ask(Q.nonzero(x), ~Q.zero(x)))
        None
        >>> ask(Q.nonzero(x), Q.positive(x))
        True
        >>> ask(Q.nonzero(x), Q.zero(x))
        False
        >>> ask(Q.nonzero(0))
        False
        >>> ask(Q.nonzero(I))
        False
        >>> ask(~Q.zero(I))
        True
        >>> ask(Q.nonzero(oo))  #doctest: +SKIP
        False

        """
        return Predicate('nonzero')
```
### 2 - sympy/assumptions/ask.py:

Start line: 71, End line: 126

```python
class AssumptionKeys(object):

    @predicate_memo
    def real(self):
        r"""
        Real number predicate.

        ``Q.real(x)`` is true iff ``x`` is a real number, i.e., it is in the
        interval `(-\infty, \infty)`.  Note that, in particular the infinities
        are not real. Use ``Q.extended_real`` if you want to consider those as
        well.

        A few important facts about reals:

        - Every real number is positive, negative, or zero.  Furthermore,
          because these sets are pairwise disjoint, each real number is exactly
          one of those three.

        - Every real number is also complex.

        - Every real number is finite.

        - Every real number is either rational or irrational.

        - Every real number is either algebraic or transcendental.

        - The facts ``Q.negative``, ``Q.zero``, ``Q.positive``,
          ``Q.nonnegative``, ``Q.nonpositive``, ``Q.nonzero``, ``Q.integer``,
          ``Q.rational``, and ``Q.irrational`` all imply ``Q.real``, as do all
          facts that imply those facts.

        - The facts ``Q.algebraic``, and ``Q.transcendental`` do not imply
          ``Q.real``; they imply ``Q.complex``. An algebraic or transcendental
          number may or may not be real.

        - The "non" facts (i.e., ``Q.nonnegative``, ``Q.nonzero``,
          ``Q.nonpositive`` and ``Q.noninteger``) are not equivalent to not the
          fact, but rather, not the fact *and* ``Q.real``.  For example,
          ``Q.nonnegative`` means ``~Q.negative & Q.real``. So for example,
          ``I`` is not nonnegative, nonzero, or nonpositive.

        Examples
        ========

        >>> from sympy import Q, ask, symbols
        >>> x = symbols('x')
        >>> ask(Q.real(x), Q.positive(x))
        True
        >>> ask(Q.real(0))
        True

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Real_number

        """
        return Predicate('real')
```
### 3 - sympy/assumptions/ask.py:

Start line: 395, End line: 431

```python
class AssumptionKeys(object):

    @predicate_memo
    def positive(self):
        r"""
        Positive real number predicate.

        ``Q.positive(x)`` is true iff ``x`` is real and `x > 0`, that is if ``x``
        is in the interval `(0, \infty)`.  In particular, infinity is not
        positive.

        A few important facts about positive numbers:

        - Note that ``Q.nonpositive`` and ``~Q.positive`` are *not* the same
          thing. ``~Q.positive(x)`` simply means that ``x`` is not positive,
          whereas ``Q.nonpositive(x)`` means that ``x`` is real and not
          positive, i.e., ``Q.nonpositive(x)`` is logically equivalent to
          `Q.negative(x) | Q.zero(x)``.  So for example, ``~Q.positive(I)`` is
          true, whereas ``Q.nonpositive(I)`` is false.

        - See the documentation of ``Q.real`` for more information about
          related facts.

        Examples
        ========

        >>> from sympy import Q, ask, symbols, I
        >>> x = symbols('x')
        >>> ask(Q.positive(x), Q.real(x) & ~Q.negative(x) & ~Q.zero(x))
        True
        >>> ask(Q.positive(1))
        True
        >>> ask(Q.nonpositive(I))
        False
        >>> ask(~Q.positive(I))
        True

        """
        return Predicate('positive')
```
### 4 - sympy/assumptions/ask.py:

Start line: 537, End line: 568

```python
class AssumptionKeys(object):

    @predicate_memo
    def nonpositive(self):
        """
        Nonpositive real number predicate.

        ``ask(Q.nonpositive(x))`` is true iff ``x`` belongs to the set of
        negative numbers including zero.

        - Note that ``Q.nonpositive`` and ``~Q.positive`` are *not* the same
          thing. ``~Q.positive(x)`` simply means that ``x`` is not positive,
          whereas ``Q.nonpositive(x)`` means that ``x`` is real and not
          positive, i.e., ``Q.nonpositive(x)`` is logically equivalent to
          `Q.negative(x) | Q.zero(x)``.  So for example, ``~Q.positive(I)`` is
          true, whereas ``Q.nonpositive(I)`` is false.

        Examples
        ========

        >>> from sympy import Q, ask, I
        >>> ask(Q.nonpositive(-1))
        True
        >>> ask(Q.nonpositive(0))
        True
        >>> ask(Q.nonpositive(1))
        False
        >>> ask(Q.nonpositive(I))
        False
        >>> ask(Q.nonpositive(-I))
        False

        """
        return Predicate('nonpositive')
```
### 5 - sympy/assumptions/handlers/sets.py:

Start line: 234, End line: 287

```python
class AskRealHandler(CommonHandler):

    @staticmethod
    def Pow(expr, assumptions):
        """
        Real**Integer              -> Real
        Positive**Real             -> Real
        Real**(Integer/Even)       -> Real if base is nonnegative
        Real**(Integer/Odd)        -> Real
        Imaginary**(Integer/Even)  -> Real
        Imaginary**(Integer/Odd)   -> not Real
        Imaginary**Real            -> ? since Real could be 0 (giving real) or 1 (giving imaginary)
        b**Imaginary               -> Real if log(b) is imaginary and b != 0 and exponent != integer multiple of I*pi/log(b)
        Real**Real                 -> ? e.g. sqrt(-1) is imaginary and sqrt(2) is not
        """
        if expr.is_number:
            return AskRealHandler._number(expr, assumptions)

        if expr.base.func == exp:
            if ask(Q.imaginary(expr.base.args[0]), assumptions):
                if ask(Q.imaginary(expr.exp), assumptions):
                    return True
            # If the i = (exp's arg)/(I*pi) is an integer or half-integer
            # multiple of I*pi then 2*i will be an integer. In addition,
            # exp(i*I*pi) = (-1)**i so the overall realness of the expr
            # can be determined by replacing exp(i*I*pi) with (-1)**i.
            i = expr.base.args[0]/I/pi
            if ask(Q.integer(2*i), assumptions):
                return ask(Q.real(((-1)**i)**expr.exp), assumptions)
            return

        if ask(Q.imaginary(expr.base), assumptions):
            if ask(Q.integer(expr.exp), assumptions):
                odd = ask(Q.odd(expr.exp), assumptions)
                if odd is not None:
                    return not odd
                return

        if ask(Q.imaginary(expr.exp), assumptions):
            imlog = ask(Q.imaginary(log(expr.base)), assumptions)
            if imlog is not None:
                # I**i -> real, log(I) is imag;
                # (2*I)**i -> complex, log(2*I) is not imag
                return imlog

        if ask(Q.real(expr.base), assumptions):
            if ask(Q.real(expr.exp), assumptions):
                if expr.exp.is_Rational and \
                        ask(Q.even(expr.exp.q), assumptions):
                    return ask(Q.positive(expr.base), assumptions)
                elif ask(Q.integer(expr.exp), assumptions):
                    return True
                elif ask(Q.positive(expr.base), assumptions):
                    return True
                elif ask(Q.negative(expr.base), assumptions):
                    return False
```
### 6 - sympy/assumptions/ask.py:

Start line: 862, End line: 892

```python
class AssumptionKeys(object):

    @predicate_memo
    def positive_definite(self):
        r"""
        Positive definite matrix predicate.

        If ``M`` is a :math:``n \times n`` symmetric real matrix, it is said
        to be positive definite if :math:`Z^TMZ` is positive for
        every non-zero column vector ``Z`` of ``n`` real numbers.

        Examples
        ========

        >>> from sympy import Q, ask, MatrixSymbol, Identity
        >>> X = MatrixSymbol('X', 2, 2)
        >>> Y = MatrixSymbol('Y', 2, 3)
        >>> Z = MatrixSymbol('Z', 2, 2)
        >>> ask(Q.positive_definite(Y))
        False
        >>> ask(Q.positive_definite(Identity(3)))
        True
        >>> ask(Q.positive_definite(X + Z), Q.positive_definite(X) &
        ...     Q.positive_definite(Z))
        True

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Positive-definite_matrix

        """
        return Predicate('positive_definite')
```
### 7 - sympy/assumptions/ask.py:

Start line: 570, End line: 601

```python
class AssumptionKeys(object):

    @predicate_memo
    def nonnegative(self):
        """
        Nonnegative real number predicate.

        ``ask(Q.nonnegative(x))`` is true iff ``x`` belongs to the set of
        positive numbers including zero.

        - Note that ``Q.nonnegative`` and ``~Q.negative`` are *not* the same
          thing. ``~Q.negative(x)`` simply means that ``x`` is not negative,
          whereas ``Q.nonnegative(x)`` means that ``x`` is real and not
          negative, i.e., ``Q.nonnegative(x)`` is logically equivalent to
          ``Q.zero(x) | Q.positive(x)``.  So for example, ``~Q.negative(I)`` is
          true, whereas ``Q.nonnegative(I)`` is false.

        Examples
        ========

        >>> from sympy import Q, ask, I
        >>> ask(Q.nonnegative(1))
        True
        >>> ask(Q.nonnegative(0))
        True
        >>> ask(Q.nonnegative(-1))
        False
        >>> ask(Q.nonnegative(I))
        False
        >>> ask(Q.nonnegative(-I))
        False

        """
        return Predicate('nonnegative')
```
### 8 - sympy/assumptions/ask.py:

Start line: 471, End line: 495

```python
class AssumptionKeys(object):

    @predicate_memo
    def zero(self):
        """
        Zero number predicate.

        ``ask(Q.zero(x))`` is true iff the value of ``x`` is zero.

        Examples
        ========

        >>> from sympy import ask, Q, oo, symbols
        >>> x, y = symbols('x, y')
        >>> ask(Q.zero(0))
        True
        >>> ask(Q.zero(1/oo))
        True
        >>> ask(Q.zero(0*oo))
        False
        >>> ask(Q.zero(1))
        False
        >>> ask(Q.zero(x*y), Q.zero(x) | Q.zero(y))
        True

        """
        return Predicate('zero')
```
### 9 - sympy/assumptions/ask.py:

Start line: 1046, End line: 1063

```python
class AssumptionKeys(object):

    @predicate_memo
    def real_elements(self):
        """
        Real elements matrix predicate.

        ``Q.real_elements(x)`` is true iff all the elements of ``x``
        are real numbers.

        Examples
        ========

        >>> from sympy import Q, ask, MatrixSymbol
        >>> X = MatrixSymbol('X', 4, 4)
        >>> ask(Q.real(X[1, 2]), Q.real_elements(X))
        True

        """
        return Predicate('real_elements')
```
### 10 - sympy/assumptions/ask.py:

Start line: 433, End line: 469

```python
class AssumptionKeys(object):

    @predicate_memo
    def negative(self):
        r"""
        Negative number predicate.

        ``Q.negative(x)`` is true iff ``x`` is a real number and :math:`x < 0`, that is,
        it is in the interval :math:`(-\infty, 0)`.  Note in particular that negative
        infinity is not negative.

        A few important facts about negative numbers:

        - Note that ``Q.nonnegative`` and ``~Q.negative`` are *not* the same
          thing. ``~Q.negative(x)`` simply means that ``x`` is not negative,
          whereas ``Q.nonnegative(x)`` means that ``x`` is real and not
          negative, i.e., ``Q.nonnegative(x)`` is logically equivalent to
          ``Q.zero(x) | Q.positive(x)``.  So for example, ``~Q.negative(I)`` is
          true, whereas ``Q.nonnegative(I)`` is false.

        - See the documentation of ``Q.real`` for more information about
          related facts.

        Examples
        ========

        >>> from sympy import Q, ask, symbols, I
        >>> x = symbols('x')
        >>> ask(Q.negative(x), Q.real(x) & ~Q.positive(x) & ~Q.zero(x))
        True
        >>> ask(Q.negative(-1))
        True
        >>> ask(Q.nonnegative(I))
        False
        >>> ask(~Q.negative(I))
        True

        """
        return Predicate('negative')
```
### 17 - sympy/core/power.py:

Start line: 157, End line: 239

```python
class Pow(Expr):
    """
    Defines the expression x**y as "x raised to a power y"

    Singleton definitions involving (0, 1, -1, oo, -oo, I, -I):

    +--------------+---------+-----------------------------------------------+
    | expr         | value   | reason                                        |
    +==============+=========+===============================================+
    | z**0         | 1       | Although arguments over 0**0 exist, see [2].  |
    +--------------+---------+-----------------------------------------------+
    | z**1         | z       |                                               |
    +--------------+---------+-----------------------------------------------+
    | (-oo)**(-1)  | 0       |                                               |
    +--------------+---------+-----------------------------------------------+
    | (-1)**-1     | -1      |                                               |
    +--------------+---------+-----------------------------------------------+
    | S.Zero**-1   | zoo     | This is not strictly true, as 0**-1 may be    |
    |              |         | undefined, but is convenient in some contexts |
    |              |         | where the base is assumed to be positive.     |
    +--------------+---------+-----------------------------------------------+
    | 1**-1        | 1       |                                               |
    +--------------+---------+-----------------------------------------------+
    | oo**-1       | 0       |                                               |
    +--------------+---------+-----------------------------------------------+
    | 0**oo        | 0       | Because for all complex numbers z near        |
    |              |         | 0, z**oo -> 0.                                |
    +--------------+---------+-----------------------------------------------+
    | 0**-oo       | zoo     | This is not strictly true, as 0**oo may be    |
    |              |         | oscillating between positive and negative     |
    |              |         | values or rotating in the complex plane.      |
    |              |         | It is convenient, however, when the base      |
    |              |         | is positive.                                  |
    +--------------+---------+-----------------------------------------------+
    | 1**oo        | nan     | Because there are various cases where         |
    | 1**-oo       |         | lim(x(t),t)=1, lim(y(t),t)=oo (or -oo),       |
    |              |         | but lim( x(t)**y(t), t) != 1.  See [3].       |
    +--------------+---------+-----------------------------------------------+
    | b**zoo       | nan     | Because b**z has no limit as z -> zoo         |
    +--------------+---------+-----------------------------------------------+
    | (-1)**oo     | nan     | Because of oscillations in the limit.         |
    | (-1)**(-oo)  |         |                                               |
    +--------------+---------+-----------------------------------------------+
    | oo**oo       | oo      |                                               |
    +--------------+---------+-----------------------------------------------+
    | oo**-oo      | 0       |                                               |
    +--------------+---------+-----------------------------------------------+
    | (-oo)**oo    | nan     |                                               |
    | (-oo)**-oo   |         |                                               |
    +--------------+---------+-----------------------------------------------+
    | oo**I        | nan     | oo**e could probably be best thought of as    |
    | (-oo)**I     |         | the limit of x**e for real x as x tends to    |
    |              |         | oo. If e is I, then the limit does not exist  |
    |              |         | and nan is used to indicate that.             |
    +--------------+---------+-----------------------------------------------+
    | oo**(1+I)    | zoo     | If the real part of e is positive, then the   |
    | (-oo)**(1+I) |         | limit of abs(x**e) is oo. So the limit value  |
    |              |         | is zoo.                                       |
    +--------------+---------+-----------------------------------------------+
    | oo**(-1+I)   | 0       | If the real part of e is negative, then the   |
    | -oo**(-1+I)  |         | limit is 0.                                   |
    +--------------+---------+-----------------------------------------------+

    Because symbolic computations are more flexible that floating point
    calculations and we prefer to never return an incorrect answer,
    we choose not to conform to all IEEE 754 conventions.  This helps
    us avoid extra test-case code in the calculation of limits.

    See Also
    ========

    sympy.core.numbers.Infinity
    sympy.core.numbers.NegativeInfinity
    sympy.core.numbers.NaN

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Exponentiation
    .. [2] https://en.wikipedia.org/wiki/Exponentiation#Zero_to_the_power_of_zero
    .. [3] https://en.wikipedia.org/wiki/Indeterminate_forms

    """
```
