# sympy__sympy-15567

| **sympy/sympy** | `39fe1d243440277a01d15fabc58dd36fc8c12f65` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 221 |
| **Any found context length** | 221 |
| **Avg pos** | 1.0 |
| **Min pos** | 1 |
| **Max pos** | 1 |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sympy/core/numbers.py b/sympy/core/numbers.py
--- a/sympy/core/numbers.py
+++ b/sympy/core/numbers.py
@@ -575,7 +575,7 @@ def __divmod__(self, other):
             return Tuple(*divmod(self.p, other.p))
         else:
             rat = self/other
-        w = sign(rat)*int(abs(rat))  # = rat.floor()
+        w = int(rat) if rat > 0 else int(rat) - 1
         r = self - other*w
         return Tuple(w, r)
 

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sympy/core/numbers.py | 578 | 578 | 1 | 1 | 221


## Problem Statement

```
SymPy's Number.__divmod__ doesn't agree with the builtin divmod
\`\`\`py
>>> divmod(4, -2.1)
(-2.0, -0.20000000000000018)
>>> divmod(S(4), S(-2.1))
(-1, 1.9)
\`\`\`

Both are mathematically correct according to the invariant in the `divmod` docstring, `div*y + mod == x`, but we should be consistent with Python. In general in Python, the sign of mod should be the same as the sign of the second argument.

\`\`\`py
>>> -1*-2.1 + 1.9
4.0
>>> -2.0*-2.1 + -0.2
4.0
\`\`\`

Our `Mod` is already correct, so it's just `Number.__divmod__` that needs to be corrected

\`\`\`py
>>> Mod(4, -2.1)
-0.200000000000000
\`\`\`

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 sympy/core/numbers.py** | 557 | 580| 221 | 221 | 29648 | 
| 2 | **1 sympy/core/numbers.py** | 2060 | 2072| 126 | 347 | 29648 | 
| 3 | **1 sympy/core/numbers.py** | 582 | 660| 604 | 951 | 29648 | 
| 4 | **1 sympy/core/numbers.py** | 1187 | 1209| 283 | 1234 | 29648 | 
| 5 | **1 sympy/core/numbers.py** | 437 | 499| 542 | 1776 | 29648 | 
| 6 | **1 sympy/core/numbers.py** | 1660 | 1677| 188 | 1964 | 29648 | 
| 7 | 2 sympy/core/mod.py | 26 | 89| 434 | 2398 | 31266 | 
| 8 | 2 sympy/core/mod.py | 1 | 24| 132 | 2530 | 31266 | 
| 9 | **2 sympy/core/numbers.py** | 1631 | 1645| 156 | 2686 | 31266 | 
| 10 | **2 sympy/core/numbers.py** | 662 | 683| 157 | 2843 | 31266 | 
| 11 | **2 sympy/core/numbers.py** | 1211 | 1218| 110 | 2953 | 31266 | 
| 12 | **2 sympy/core/numbers.py** | 1646 | 1658| 144 | 3097 | 31266 | 
| 13 | **2 sympy/core/numbers.py** | 3882 | 3938| 358 | 3455 | 31266 | 
| 14 | 2 sympy/core/mod.py | 91 | 183| 845 | 4300 | 31266 | 
| 15 | 2 sympy/core/mod.py | 185 | 212| 260 | 4560 | 31266 | 
| 16 | **2 sympy/core/numbers.py** | 1097 | 1185| 752 | 5312 | 31266 | 
| 17 | **2 sympy/core/numbers.py** | 3083 | 3154| 498 | 5810 | 31266 | 
| 18 | **2 sympy/core/numbers.py** | 110 | 134| 184 | 5994 | 31266 | 
| 19 | **2 sympy/core/numbers.py** | 1848 | 1871| 183 | 6177 | 31266 | 
| 20 | **2 sympy/core/numbers.py** | 536 | 555| 179 | 6356 | 31266 | 
| 21 | **2 sympy/core/numbers.py** | 2636 | 2656| 194 | 6550 | 31266 | 
| 22 | **2 sympy/core/numbers.py** | 3016 | 3045| 191 | 6741 | 31266 | 
| 23 | 3 sympy/sets/handlers/mul.py | 50 | 81| 207 | 6948 | 31850 | 
| 24 | **3 sympy/core/numbers.py** | 1617 | 1629| 141 | 7089 | 31850 | 
| 25 | **3 sympy/core/numbers.py** | 1291 | 1309| 174 | 7263 | 31850 | 
| 26 | **3 sympy/core/numbers.py** | 1717 | 1738| 163 | 7426 | 31850 | 
| 27 | **3 sympy/core/numbers.py** | 1797 | 1813| 172 | 7598 | 31850 | 
| 28 | **3 sympy/core/numbers.py** | 1679 | 1715| 402 | 8000 | 31850 | 
| 29 | **3 sympy/core/numbers.py** | 1740 | 1759| 166 | 8166 | 31850 | 
| 30 | **3 sympy/core/numbers.py** | 1 | 38| 323 | 8489 | 31850 | 
| 31 | **3 sympy/core/numbers.py** | 1779 | 1795| 172 | 8661 | 31850 | 
| 32 | **3 sympy/core/numbers.py** | 1761 | 1777| 172 | 8833 | 31850 | 
| 33 | 4 sympy/polys/domains/modularinteger.py | 1 | 177| 1032 | 9865 | 33094 | 
| 34 | **4 sympy/core/numbers.py** | 1252 | 1289| 339 | 10204 | 33094 | 
| 35 | **4 sympy/core/numbers.py** | 1591 | 1604| 131 | 10335 | 33094 | 
| 36 | **4 sympy/core/numbers.py** | 1815 | 1831| 172 | 10507 | 33094 | 
| 37 | **4 sympy/core/numbers.py** | 1311 | 1326| 158 | 10665 | 33094 | 
| 38 | **4 sympy/core/numbers.py** | 1328 | 1343| 158 | 10823 | 33094 | 
| 39 | **4 sympy/core/numbers.py** | 2074 | 2162| 750 | 11573 | 33094 | 
| 40 | **4 sympy/core/numbers.py** | 2029 | 2058| 172 | 11745 | 33094 | 
| 41 | **4 sympy/core/numbers.py** | 2793 | 2821| 183 | 11928 | 33094 | 
| 42 | **4 sympy/core/numbers.py** | 1605 | 1616| 125 | 12053 | 33094 | 
| 43 | **4 sympy/core/numbers.py** | 1345 | 1380| 277 | 12330 | 33094 | 
| 44 | **4 sympy/core/numbers.py** | 502 | 534| 299 | 12629 | 33094 | 
| 45 | **4 sympy/core/numbers.py** | 685 | 779| 765 | 13394 | 33094 | 
| 46 | **4 sympy/core/numbers.py** | 2284 | 2334| 470 | 13864 | 33094 | 
| 47 | **4 sympy/core/numbers.py** | 1475 | 1548| 489 | 14353 | 33094 | 
| 48 | **4 sympy/core/numbers.py** | 2164 | 2209| 347 | 14700 | 33094 | 
| 49 | **4 sympy/core/numbers.py** | 2862 | 2933| 494 | 15194 | 33094 | 
| 50 | **4 sympy/core/numbers.py** | 1578 | 1590| 132 | 15326 | 33094 | 
| 51 | 5 sympy/physics/vector/dyadic.py | 101 | 121| 147 | 15473 | 37745 | 
| 52 | 6 sympy/polys/rings.py | 1292 | 1319| 217 | 15690 | 55300 | 
| 53 | **6 sympy/core/numbers.py** | 3486 | 3554| 429 | 16119 | 55300 | 
| 54 | **6 sympy/core/numbers.py** | 3557 | 3618| 433 | 16552 | 55300 | 
| 55 | 7 sympy/solvers/diophantine.py | 1 | 93| 695 | 17247 | 86542 | 
| 56 | **7 sympy/core/numbers.py** | 1032 | 1081| 522 | 17769 | 86542 | 
| 57 | **7 sympy/core/numbers.py** | 2521 | 2548| 227 | 17996 | 86542 | 
| 58 | **7 sympy/core/numbers.py** | 1083 | 1095| 130 | 18126 | 86542 | 
| 59 | **7 sympy/core/numbers.py** | 937 | 1030| 807 | 18933 | 86542 | 
| 60 | 8 sympy/physics/quantum/sho1d.py | 380 | 395| 139 | 19072 | 92012 | 
| 61 | 9 sympy/ntheory/__init__.py | 1 | 22| 255 | 19327 | 92268 | 
| 62 | 10 sympy/polys/domains/pythonrational.py | 208 | 288| 500 | 19827 | 94311 | 
| 63 | 11 sympy/core/expr.py | 3422 | 3450| 242 | 20069 | 123427 | 
| 64 | 11 sympy/polys/domains/modularinteger.py | 179 | 210| 211 | 20280 | 123427 | 
| 65 | 12 sympy/polys/modulargcd.py | 1709 | 1770| 410 | 20690 | 141922 | 
| 66 | 13 sympy/matrices/common.py | 2041 | 2074| 299 | 20989 | 159281 | 
| 67 | 13 sympy/core/expr.py | 163 | 178| 131 | 21120 | 159281 | 
| 68 | **13 sympy/core/numbers.py** | 2598 | 2634| 166 | 21286 | 159281 | 
| 69 | **13 sympy/core/numbers.py** | 2211 | 2283| 687 | 21973 | 159281 | 
| 70 | 14 sympy/diffgeom/diffgeom.py | 1 | 20| 174 | 22147 | 173724 | 
| 71 | **14 sympy/core/numbers.py** | 206 | 224| 146 | 22293 | 173724 | 
| 72 | 15 sympy/ntheory/partitions_.py | 1 | 11| 142 | 22435 | 175836 | 
| 73 | 15 sympy/core/expr.py | 180 | 218| 341 | 22776 | 175836 | 


## Patch

```diff
diff --git a/sympy/core/numbers.py b/sympy/core/numbers.py
--- a/sympy/core/numbers.py
+++ b/sympy/core/numbers.py
@@ -575,7 +575,7 @@ def __divmod__(self, other):
             return Tuple(*divmod(self.p, other.p))
         else:
             rat = self/other
-        w = sign(rat)*int(abs(rat))  # = rat.floor()
+        w = int(rat) if rat > 0 else int(rat) - 1
         r = self - other*w
         return Tuple(w, r)
 

```

## Test Patch

```diff
diff --git a/sympy/core/tests/test_numbers.py b/sympy/core/tests/test_numbers.py
--- a/sympy/core/tests/test_numbers.py
+++ b/sympy/core/tests/test_numbers.py
@@ -191,6 +191,9 @@ def test_divmod():
     assert divmod(S(-3), S(2)) == (-2, 1)
     assert divmod(S(-3), 2) == (-2, 1)
 
+    assert divmod(S(4), S(-3.1)) == Tuple(-2, -2.2)
+    assert divmod(S(4), S(-2.1)) == divmod(4, -2.1)
+    assert divmod(S(-8), S(-2.5) ) == Tuple(3 , -0.5)
 
 def test_igcd():
     assert igcd(0, 0) == 0

```


## Code snippets

### 1 - sympy/core/numbers.py:

Start line: 557, End line: 580

```python
class Number(AtomicExpr):

    def invert(self, other, *gens, **args):
        from sympy.polys.polytools import invert
        if getattr(other, 'is_number', True):
            return mod_inverse(self, other)
        return invert(self, other, *gens, **args)

    def __divmod__(self, other):
        from .containers import Tuple
        from sympy.functions.elementary.complexes import sign

        try:
            other = Number(other)
        except TypeError:
            msg = "unsupported operand type(s) for divmod(): '%s' and '%s'"
            raise TypeError(msg % (type(self).__name__, type(other).__name__))
        if not other:
            raise ZeroDivisionError('modulo by zero')
        if self.is_Integer and other.is_Integer:
            return Tuple(*divmod(self.p, other.p))
        else:
            rat = self/other
        w = sign(rat)*int(abs(rat))  # = rat.floor()
        r = self - other*w
        return Tuple(w, r)
```
### 2 - sympy/core/numbers.py:

Start line: 2060, End line: 2072

```python
class Integer(Rational):

    def __rdivmod__(self, other):
        from .containers import Tuple
        if isinstance(other, integer_types) and global_evaluate[0]:
            return Tuple(*(divmod(other, self.p)))
        else:
            try:
                other = Number(other)
            except TypeError:
                msg = "unsupported operand type(s) for divmod(): '%s' and '%s'"
                oname = type(other).__name__
                sname = type(self).__name__
                raise TypeError(msg % (oname, sname))
            return Number.__divmod__(other, self)
```
### 3 - sympy/core/numbers.py:

Start line: 582, End line: 660

```python
class Number(AtomicExpr):

    def __rdivmod__(self, other):
        try:
            other = Number(other)
        except TypeError:
            msg = "unsupported operand type(s) for divmod(): '%s' and '%s'"
            raise TypeError(msg % (type(other).__name__, type(self).__name__))
        return divmod(other, self)

    def __round__(self, *args):
        return round(float(self), *args)

    def _as_mpf_val(self, prec):
        """Evaluation of mpf tuple accurate to at least prec bits."""
        raise NotImplementedError('%s needs ._as_mpf_val() method' %
            (self.__class__.__name__))

    def _eval_evalf(self, prec):
        return Float._new(self._as_mpf_val(prec), prec)

    def _as_mpf_op(self, prec):
        prec = max(prec, self._prec)
        return self._as_mpf_val(prec), prec

    def __float__(self):
        return mlib.to_float(self._as_mpf_val(53))

    def floor(self):
        raise NotImplementedError('%s needs .floor() method' %
            (self.__class__.__name__))

    def ceiling(self):
        raise NotImplementedError('%s needs .ceiling() method' %
            (self.__class__.__name__))

    def _eval_conjugate(self):
        return self

    def _eval_order(self, *symbols):
        from sympy import Order
        # Order(5, x, y) -> Order(1,x,y)
        return Order(S.One, *symbols)

    def _eval_subs(self, old, new):
        if old == -self:
            return -new
        return self  # there is no other possibility

    def _eval_is_finite(self):
        return True

    @classmethod
    def class_key(cls):
        return 1, 0, 'Number'

    @cacheit
    def sort_key(self, order=None):
        return self.class_key(), (0, ()), (), self

    @_sympifyit('other', NotImplemented)
    def __add__(self, other):
        if isinstance(other, Number) and global_evaluate[0]:
            if other is S.NaN:
                return S.NaN
            elif other is S.Infinity:
                return S.Infinity
            elif other is S.NegativeInfinity:
                return S.NegativeInfinity
        return AtomicExpr.__add__(self, other)

    @_sympifyit('other', NotImplemented)
    def __sub__(self, other):
        if isinstance(other, Number) and global_evaluate[0]:
            if other is S.NaN:
                return S.NaN
            elif other is S.Infinity:
                return S.NegativeInfinity
            elif other is S.NegativeInfinity:
                return S.Infinity
        return AtomicExpr.__sub__(self, other)
```
### 4 - sympy/core/numbers.py:

Start line: 1187, End line: 1209

```python
class Float(Number):

    @_sympifyit('other', NotImplemented)
    def __div__(self, other):
        if isinstance(other, Number) and other != 0 and global_evaluate[0]:
            rhs, prec = other._as_mpf_op(self._prec)
            return Float._new(mlib.mpf_div(self._mpf_, rhs, prec, rnd), prec)
        return Number.__div__(self, other)

    __truediv__ = __div__

    @_sympifyit('other', NotImplemented)
    def __mod__(self, other):
        if isinstance(other, Rational) and other.q != 1 and global_evaluate[0]:
            # calculate mod with Rationals, *then* round the result
            return Float(Rational.__mod__(Rational(self), other),
                         precision=self._prec)
        if isinstance(other, Float) and global_evaluate[0]:
            r = self/other
            if r == int(r):
                return Float(0, precision=max(self._prec, other._prec))
        if isinstance(other, Number) and global_evaluate[0]:
            rhs, prec = other._as_mpf_op(self._prec)
            return Float._new(mlib.mpf_mod(self._mpf_, rhs, prec, rnd), prec)
        return Number.__mod__(self, other)
```
### 5 - sympy/core/numbers.py:

Start line: 437, End line: 499

```python
def mod_inverse(a, m):
    """
    Return the number c such that, (a * c) = 1 (mod m)
    where c has the same sign as m. If no such value exists,
    a ValueError is raised.

    Examples
    ========

    >>> from sympy import S
    >>> from sympy.core.numbers import mod_inverse

    Suppose we wish to find multiplicative inverse x of
    3 modulo 11. This is the same as finding x such
    that 3 * x = 1 (mod 11). One value of x that satisfies
    this congruence is 4. Because 3 * 4 = 12 and 12 = 1 (mod 11).
    This is the value return by mod_inverse:

    >>> mod_inverse(3, 11)
    4
    >>> mod_inverse(-3, 11)
    7

    When there is a common factor between the numerators of
    ``a`` and ``m`` the inverse does not exist:

    >>> mod_inverse(2, 4)
    Traceback (most recent call last):
    ...
    ValueError: inverse of 2 mod 4 does not exist

    >>> mod_inverse(S(2)/7, S(5)/2)
    7/2

    References
    ==========
    - https://en.wikipedia.org/wiki/Modular_multiplicative_inverse
    - https://en.wikipedia.org/wiki/Extended_Euclidean_algorithm
    """
    c = None
    try:
        a, m = as_int(a), as_int(m)
        if m != 1 and m != -1:
            x, y, g = igcdex(a, m)
            if g == 1:
                c = x % m
    except ValueError:
        a, m = sympify(a), sympify(m)
        if not (a.is_number and m.is_number):
            raise TypeError(filldedent('''
                Expected numbers for arguments; symbolic `mod_inverse`
                is not implemented
                but symbolic expressions can be handled with the
                similar function,
                sympy.polys.polytools.invert'''))
        big = (m > 1)
        if not (big is S.true or big is S.false):
            raise ValueError('m > 1 did not evaluate; try to simplify %s' % m)
        elif big:
            c = 1/a
    if c is None:
        raise ValueError('inverse of %s (mod %s) does not exist' % (a, m))
    return c
```
### 6 - sympy/core/numbers.py:

Start line: 1660, End line: 1677

```python
class Rational(Number):

    @_sympifyit('other', NotImplemented)
    def __mod__(self, other):
        if global_evaluate[0]:
            if isinstance(other, Rational):
                n = (self.p*other.q) // (other.p*self.q)
                return Rational(self.p*other.q - n*other.p*self.q, self.q*other.q)
            if isinstance(other, Float):
                # calculate mod with Rationals, *then* round the answer
                return Float(self.__mod__(Rational(other)),
                             precision=other._prec)
            return Number.__mod__(self, other)
        return Number.__mod__(self, other)

    @_sympifyit('other', NotImplemented)
    def __rmod__(self, other):
        if isinstance(other, Rational):
            return Rational.__mod__(other, self)
        return Number.__rmod__(self, other)
```
### 7 - sympy/core/mod.py:

Start line: 26, End line: 89

```python
class Mod(Function):

    @classmethod
    def eval(cls, p, q):
        from sympy.core.add import Add
        from sympy.core.mul import Mul
        from sympy.core.singleton import S
        from sympy.core.exprtools import gcd_terms
        from sympy.polys.polytools import gcd

        def doit(p, q):
            """Try to return p % q if both are numbers or +/-p is known
            to be less than or equal q.
            """

            if q == S.Zero:
                raise ZeroDivisionError("Modulo by zero")
            if p.is_infinite or q.is_infinite or p is nan or q is nan:
                return nan
            if p == S.Zero or p == q or p == -q or (p.is_integer and q == 1):
                return S.Zero

            if q.is_Number:
                if p.is_Number:
                    return (p % q)
                if q == 2:
                    if p.is_even:
                        return S.Zero
                    elif p.is_odd:
                        return S.One

            if hasattr(p, '_eval_Mod'):
                rv = getattr(p, '_eval_Mod')(q)
                if rv is not None:
                    return rv

            # by ratio
            r = p/q
            try:
                d = int(r)
            except TypeError:
                pass
            else:
                if type(d) is int:
                    rv = p - d*q
                    if (rv*q < 0) == True:
                        rv += q
                    return rv

            # by difference
            # -2|q| < p < 2|q|
            d = abs(p)
            for _ in range(2):
                d -= abs(q)
                if d.is_negative:
                    if q.is_positive:
                        if p.is_positive:
                            return d + q
                        elif p.is_negative:
                            return -d
                    elif q.is_negative:
                        if p.is_positive:
                            return d
                        elif p.is_negative:
                            return -d + q
                    break
        # ... other code
```
### 8 - sympy/core/mod.py:

Start line: 1, End line: 24

```python
from __future__ import print_function, division

from sympy.core.numbers import nan
from .function import Function


class Mod(Function):
    """Represents a modulo operation on symbolic expressions.

    Receives two arguments, dividend p and divisor q.

    The convention used is the same as Python's: the remainder always has the
    same sign as the divisor.

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> x**2 % y
    Mod(x**2, y)
    >>> _.subs({x: 5, y: 6})
    1

    """
```
### 9 - sympy/core/numbers.py:

Start line: 1631, End line: 1645

```python
class Rational(Number):

    @_sympifyit('other', NotImplemented)
    def __div__(self, other):
        if global_evaluate[0]:
            if isinstance(other, Integer):
                if self.p and other.p == S.Zero:
                    return S.ComplexInfinity
                else:
                    return Rational(self.p, self.q*other.p, igcd(self.p, other.p))
            elif isinstance(other, Rational):
                return Rational(self.p*other.q, self.q*other.p, igcd(self.p, other.p)*igcd(self.q, other.q))
            elif isinstance(other, Float):
                return self*(1/other)
            else:
                return Number.__div__(self, other)
        return Number.__div__(self, other)
```
### 10 - sympy/core/numbers.py:

Start line: 662, End line: 683

```python
class Number(AtomicExpr):

    @_sympifyit('other', NotImplemented)
    def __mul__(self, other):
        if isinstance(other, Number) and global_evaluate[0]:
            if other is S.NaN:
                return S.NaN
            elif other is S.Infinity:
                if self.is_zero:
                    return S.NaN
                elif self.is_positive:
                    return S.Infinity
                else:
                    return S.NegativeInfinity
            elif other is S.NegativeInfinity:
                if self.is_zero:
                    return S.NaN
                elif self.is_positive:
                    return S.NegativeInfinity
                else:
                    return S.Infinity
        elif isinstance(other, Tuple):
            return NotImplemented
        return AtomicExpr.__mul__(self, other)
```
### 11 - sympy/core/numbers.py:

Start line: 1211, End line: 1218

```python
class Float(Number):

    @_sympifyit('other', NotImplemented)
    def __rmod__(self, other):
        if isinstance(other, Float) and global_evaluate[0]:
            return other.__mod__(self)
        if isinstance(other, Number) and global_evaluate[0]:
            rhs, prec = other._as_mpf_op(self._prec)
            return Float._new(mlib.mpf_mod(rhs, self._mpf_, prec, rnd), prec)
        return Number.__rmod__(self, other)
```
### 12 - sympy/core/numbers.py:

Start line: 1646, End line: 1658

```python
class Rational(Number):
    @_sympifyit('other', NotImplemented)
    def __rdiv__(self, other):
        if global_evaluate[0]:
            if isinstance(other, Integer):
                return Rational(other.p*self.q, self.p, igcd(self.p, other.p))
            elif isinstance(other, Rational):
                return Rational(other.p*self.q, other.q*self.p, igcd(self.p, other.p)*igcd(self.q, other.q))
            elif isinstance(other, Float):
                return other*(1/self)
            else:
                return Number.__rdiv__(self, other)
        return Number.__rdiv__(self, other)
    __truediv__ = __div__
```
### 13 - sympy/core/numbers.py:

Start line: 3882, End line: 3938

```python
I = S.ImaginaryUnit


def sympify_fractions(f):
    return Rational(f.numerator, f.denominator, 1)

converter[fractions.Fraction] = sympify_fractions

try:
    if HAS_GMPY == 2:
        import gmpy2 as gmpy
    elif HAS_GMPY == 1:
        import gmpy
    else:
        raise ImportError

    def sympify_mpz(x):
        return Integer(long(x))

    def sympify_mpq(x):
        return Rational(long(x.numerator), long(x.denominator))

    converter[type(gmpy.mpz(1))] = sympify_mpz
    converter[type(gmpy.mpq(1, 2))] = sympify_mpq
except ImportError:
    pass


def sympify_mpmath(x):
    return Expr._from_mpmath(x, x.context.prec)

converter[mpnumeric] = sympify_mpmath


def sympify_mpq(x):
    p, q = x._mpq_
    return Rational(p, q, 1)

converter[type(mpmath.rational.mpq(1, 2))] = sympify_mpq


def sympify_complex(a):
    real, imag = list(map(sympify, (a.real, a.imag)))
    return real + S.ImaginaryUnit*imag

converter[complex] = sympify_complex

_intcache[0] = S.Zero
_intcache[1] = S.One
_intcache[-1] = S.NegativeOne

from .power import Pow, integer_nthroot
from .mul import Mul
Mul.identity = One()
from .add import Add
Add.identity = Zero()
```
### 16 - sympy/core/numbers.py:

Start line: 1097, End line: 1185

```python
class Float(Number):

    # mpz can't be pickled
    def __getnewargs__(self):
        return (mlib.to_pickable(self._mpf_),)

    def __getstate__(self):
        return {'_prec': self._prec}

    def _hashable_content(self):
        return (self._mpf_, self._prec)

    def floor(self):
        return Integer(int(mlib.to_int(
            mlib.mpf_floor(self._mpf_, self._prec))))

    def ceiling(self):
        return Integer(int(mlib.to_int(
            mlib.mpf_ceil(self._mpf_, self._prec))))

    @property
    def num(self):
        return mpmath.mpf(self._mpf_)

    def _as_mpf_val(self, prec):
        rv = mpf_norm(self._mpf_, prec)
        if rv != self._mpf_ and self._prec == prec:
            debug(self._mpf_, rv)
        return rv

    def _as_mpf_op(self, prec):
        return self._mpf_, max(prec, self._prec)

    def _eval_is_finite(self):
        if self._mpf_ in (_mpf_inf, _mpf_ninf):
            return False
        return True

    def _eval_is_infinite(self):
        if self._mpf_ in (_mpf_inf, _mpf_ninf):
            return True
        return False

    def _eval_is_integer(self):
        return self._mpf_ == _mpf_zero

    def _eval_is_negative(self):
        if self._mpf_ == _mpf_ninf:
            return True
        if self._mpf_ == _mpf_inf:
            return False
        return self.num < 0

    def _eval_is_positive(self):
        if self._mpf_ == _mpf_inf:
            return True
        if self._mpf_ == _mpf_ninf:
            return False
        return self.num > 0

    def _eval_is_zero(self):
        return self._mpf_ == _mpf_zero

    def __nonzero__(self):
        return self._mpf_ != _mpf_zero

    __bool__ = __nonzero__

    def __neg__(self):
        return Float._new(mlib.mpf_neg(self._mpf_), self._prec)

    @_sympifyit('other', NotImplemented)
    def __add__(self, other):
        if isinstance(other, Number) and global_evaluate[0]:
            rhs, prec = other._as_mpf_op(self._prec)
            return Float._new(mlib.mpf_add(self._mpf_, rhs, prec, rnd), prec)
        return Number.__add__(self, other)

    @_sympifyit('other', NotImplemented)
    def __sub__(self, other):
        if isinstance(other, Number) and global_evaluate[0]:
            rhs, prec = other._as_mpf_op(self._prec)
            return Float._new(mlib.mpf_sub(self._mpf_, rhs, prec, rnd), prec)
        return Number.__sub__(self, other)

    @_sympifyit('other', NotImplemented)
    def __mul__(self, other):
        if isinstance(other, Number) and global_evaluate[0]:
            rhs, prec = other._as_mpf_op(self._prec)
            return Float._new(mlib.mpf_mul(self._mpf_, rhs, prec, rnd), prec)
        return Number.__mul__(self, other)
```
### 17 - sympy/core/numbers.py:

Start line: 3083, End line: 3154

```python
class NegativeInfinity(with_metaclass(Singleton, Number)):

    def _as_mpf_val(self, prec):
        return mlib.fninf

    def _sage_(self):
        import sage.all as sage
        return -(sage.oo)

    def __hash__(self):
        return super(NegativeInfinity, self).__hash__()

    def __eq__(self, other):
        return other is S.NegativeInfinity

    def __ne__(self, other):
        return other is not S.NegativeInfinity

    def __lt__(self, other):
        try:
            other = _sympify(other)
        except SympifyError:
            raise TypeError("Invalid comparison %s < %s" % (self, other))
        if other.is_real:
            if other.is_finite or other is S.Infinity:
                return S.true
            elif other.is_nonnegative:
                return S.true
            elif other.is_infinite and other.is_negative:
                return S.false
        return Expr.__lt__(self, other)

    def __le__(self, other):
        try:
            other = _sympify(other)
        except SympifyError:
            raise TypeError("Invalid comparison %s <= %s" % (self, other))
        if other.is_real:
            return S.true
        return Expr.__le__(self, other)

    def __gt__(self, other):
        try:
            other = _sympify(other)
        except SympifyError:
            raise TypeError("Invalid comparison %s > %s" % (self, other))
        if other.is_real:
            return S.false
        return Expr.__gt__(self, other)

    def __ge__(self, other):
        try:
            other = _sympify(other)
        except SympifyError:
            raise TypeError("Invalid comparison %s >= %s" % (self, other))
        if other.is_real:
            if other.is_finite or other is S.Infinity:
                return S.false
            elif other.is_nonnegative:
                return S.false
            elif other.is_infinite and other.is_negative:
                return S.true
        return Expr.__ge__(self, other)

    def __mod__(self, other):
        return S.NaN

    __rmod__ = __mod__

    def floor(self):
        return self

    def ceiling(self):
        return self
```
### 18 - sympy/core/numbers.py:

Start line: 110, End line: 134

```python
# TODO: we should use the warnings module
_errdict = {"divide": False}


def seterr(divide=False):
    """
    Should sympy raise an exception on 0/0 or return a nan?

    divide == True .... raise an exception
    divide == False ... return nan
    """
    if _errdict["divide"] != divide:
        clear_cache()
        _errdict["divide"] = divide


def _as_integer_ratio(p):
    neg_pow, man, expt, bc = getattr(p, '_mpf_', mpmath.mpf(p)._mpf_)
    p = [1, -1][neg_pow % 2]*man
    if expt < 0:
        q = 2**-expt
    else:
        q = 1
        p *= 2**expt
    return int(p), int(q)
```
### 19 - sympy/core/numbers.py:

Start line: 1848, End line: 1871

```python
class Rational(Number):

    @_sympifyit('other', NotImplemented)
    def gcd(self, other):
        if isinstance(other, Rational):
            if other is S.Zero:
                return other
            return Rational(
                Integer(igcd(self.p, other.p)),
                Integer(ilcm(self.q, other.q)))
        return Number.gcd(self, other)

    @_sympifyit('other', NotImplemented)
    def lcm(self, other):
        if isinstance(other, Rational):
            return Rational(
                self.p // igcd(self.p, other.p) * other.p,
                igcd(self.q, other.q))
        return Number.lcm(self, other)

    def as_numer_denom(self):
        return Integer(self.p), Integer(self.q)

    def _sage_(self):
        import sage.all as sage
        return sage.Integer(self.p)/sage.Integer(self.q)
```
### 20 - sympy/core/numbers.py:

Start line: 536, End line: 555

```python
class Number(AtomicExpr):

    def __new__(cls, *obj):
        if len(obj) == 1:
            obj = obj[0]

        if isinstance(obj, Number):
            return obj
        if isinstance(obj, SYMPY_INTS):
            return Integer(obj)
        if isinstance(obj, tuple) and len(obj) == 2:
            return Rational(*obj)
        if isinstance(obj, (float, mpmath.mpf, decimal.Decimal)):
            return Float(obj)
        if isinstance(obj, string_types):
            val = sympify(obj)
            if isinstance(val, Number):
                return val
            else:
                raise ValueError('String "%s" does not denote a Number' % obj)
        msg = "expected str|int|long|float|Decimal|Number object but got %r"
        raise TypeError(msg % type(obj).__name__)
```
### 21 - sympy/core/numbers.py:

Start line: 2636, End line: 2656

```python
class NegativeOne(with_metaclass(Singleton, IntegerConstant)):

    def _eval_power(self, expt):
        if expt.is_odd:
            return S.NegativeOne
        if expt.is_even:
            return S.One
        if isinstance(expt, Number):
            if isinstance(expt, Float):
                return Float(-1.0)**expt
            if expt is S.NaN:
                return S.NaN
            if expt is S.Infinity or expt is S.NegativeInfinity:
                return S.NaN
            if expt is S.Half:
                return S.ImaginaryUnit
            if isinstance(expt, Rational):
                if expt.q == 2:
                    return S.ImaginaryUnit**Integer(expt.p)
                i, r = divmod(expt.p, expt.q)
                if i:
                    return self**i*self**Rational(r, expt.q)
        return
```
### 22 - sympy/core/numbers.py:

Start line: 3016, End line: 3045

```python
class NegativeInfinity(with_metaclass(Singleton, Number)):

    @_sympifyit('other', NotImplemented)
    def __div__(self, other):
        if isinstance(other, Number):
            if other is S.Infinity or \
                other is S.NegativeInfinity or \
                    other is S.NaN:
                return S.NaN
            elif other.is_Float:
                if other == Float('-inf') or \
                    other == Float('inf') or \
                        other is S.NaN:
                    return S.NaN
                elif other.is_nonnegative:
                    return Float('-inf')
                else:
                    return Float('inf')
            else:
                if other >= 0:
                    return S.NegativeInfinity
                else:
                    return S.Infinity
        return NotImplemented

    __truediv__ = __div__

    def __abs__(self):
        return S.Infinity

    def __neg__(self):
        return S.Infinity
```
### 24 - sympy/core/numbers.py:

Start line: 1617, End line: 1629

```python
class Rational(Number):
    @_sympifyit('other', NotImplemented)
    def __mul__(self, other):
        if global_evaluate[0]:
            if isinstance(other, Integer):
                return Rational(self.p*other.p, self.q, igcd(other.p, self.q))
            elif isinstance(other, Rational):
                return Rational(self.p*other.p, self.q*other.q, igcd(self.p, other.q)*igcd(self.q, other.p))
            elif isinstance(other, Float):
                return other*self
            else:
                return Number.__mul__(self, other)
        return Number.__mul__(self, other)
    __rmul__ = __mul__
```
### 25 - sympy/core/numbers.py:

Start line: 1291, End line: 1309

```python
class Float(Number):

    def __ne__(self, other):
        return not self == other

    def __gt__(self, other):
        try:
            other = _sympify(other)
        except SympifyError:
            raise TypeError("Invalid comparison %s > %s" % (self, other))
        if other.is_NumberSymbol:
            return other.__lt__(self)
        if other.is_Rational and not other.is_Integer:
            self *= other.q
            other = _sympify(other.p)
        elif other.is_comparable:
            other = other.evalf()
        if other.is_Number and other is not S.NaN:
            return _sympify(bool(
                mlib.mpf_gt(self._mpf_, other._as_mpf_val(self._prec))))
        return Expr.__gt__(self, other)
```
### 26 - sympy/core/numbers.py:

Start line: 1717, End line: 1738

```python
class Rational(Number):

    def _as_mpf_val(self, prec):
        return mlib.from_rational(self.p, self.q, prec, rnd)

    def _mpmath_(self, prec, rnd):
        return mpmath.make_mpf(mlib.from_rational(self.p, self.q, prec, rnd))

    def __abs__(self):
        return Rational(abs(self.p), self.q)

    def __int__(self):
        p, q = self.p, self.q
        if p < 0:
            return -int(-p//q)
        return int(p//q)

    __long__ = __int__

    def floor(self):
        return Integer(self.p // self.q)

    def ceiling(self):
        return -Integer(-self.p // self.q)
```
### 27 - sympy/core/numbers.py:

Start line: 1797, End line: 1813

```python
class Rational(Number):

    def __lt__(self, other):
        try:
            other = _sympify(other)
        except SympifyError:
            raise TypeError("Invalid comparison %s < %s" % (self, other))
        if other.is_NumberSymbol:
            return other.__gt__(self)
        expr = self
        if other.is_Number:
            if other.is_Rational:
                return _sympify(bool(self.p*other.q < self.q*other.p))
            if other.is_Float:
                return _sympify(bool(mlib.mpf_lt(
                    self._as_mpf_val(other._prec), other._mpf_)))
        elif other.is_number and other.is_real:
            expr, other = Integer(self.p), self.q*other
        return Expr.__lt__(expr, other)
```
### 28 - sympy/core/numbers.py:

Start line: 1679, End line: 1715

```python
class Rational(Number):

    def _eval_power(self, expt):
        if isinstance(expt, Number):
            if isinstance(expt, Float):
                return self._eval_evalf(expt._prec)**expt
            if expt.is_negative:
                # (3/4)**-2 -> (4/3)**2
                ne = -expt
                if (ne is S.One):
                    return Rational(self.q, self.p)
                if self.is_negative:
                    return S.NegativeOne**expt*Rational(self.q, -self.p)**ne
                else:
                    return Rational(self.q, self.p)**ne
            if expt is S.Infinity:  # -oo already caught by test for negative
                if self.p > self.q:
                    # (3/2)**oo -> oo
                    return S.Infinity
                if self.p < -self.q:
                    # (-3/2)**oo -> oo + I*oo
                    return S.Infinity + S.Infinity*S.ImaginaryUnit
                return S.Zero
            if isinstance(expt, Integer):
                # (4/3)**2 -> 4**2 / 3**2
                return Rational(self.p**expt.p, self.q**expt.p, 1)
            if isinstance(expt, Rational):
                if self.p != 1:
                    # (4/3)**(5/6) -> 4**(5/6)*3**(-5/6)
                    return Integer(self.p)**expt*Integer(self.q)**(-expt)
                # as the above caught negative self.p, now self is positive
                return Integer(self.q)**Rational(
                expt.p*(expt.q - 1), expt.q) / \
                    Integer(self.q)**Integer(expt.p)

        if self.is_negative and expt.is_even:
            return (-self)**expt

        return
```
### 29 - sympy/core/numbers.py:

Start line: 1740, End line: 1759

```python
class Rational(Number):

    def __eq__(self, other):
        try:
            other = _sympify(other)
        except SympifyError:
            return NotImplemented
        if other.is_NumberSymbol:
            if other.is_irrational:
                return False
            return other.__eq__(self)
        if other.is_Number:
            if other.is_Rational:
                # a Rational is always in reduced form so will never be 2/4
                # so we can just check equivalence of args
                return self.p == other.p and self.q == other.q
            if other.is_Float:
                return mlib.mpf_eq(self._as_mpf_val(other._prec), other._mpf_)
        return False

    def __ne__(self, other):
        return not self == other
```
### 30 - sympy/core/numbers.py:

Start line: 1, End line: 38

```python
from __future__ import print_function, division

import decimal
import fractions
import math
import warnings
import re as regex
from collections import defaultdict

from .containers import Tuple
from .sympify import converter, sympify, _sympify, SympifyError, _convert_numpy_types
from .singleton import S, Singleton
from .expr import Expr, AtomicExpr
from .decorators import _sympifyit
from .cache import cacheit, clear_cache
from .logic import fuzzy_not
from sympy.core.compatibility import (
    as_int, integer_types, long, string_types, with_metaclass, HAS_GMPY,
    SYMPY_INTS, int_info)
from sympy.core.cache import lru_cache

import mpmath
import mpmath.libmp as mlib
from mpmath.libmp.backend import MPZ
from mpmath.libmp import mpf_pow, mpf_pi, mpf_e, phi_fixed
from mpmath.ctx_mp import mpnumeric
from mpmath.libmp.libmpf import (
    finf as _mpf_inf, fninf as _mpf_ninf,
    fnan as _mpf_nan, fzero as _mpf_zero, _normalize as mpf_normalize,
    prec_to_dps)
from sympy.utilities.misc import debug, filldedent
from .evaluate import global_evaluate

from sympy.utilities.exceptions import SymPyDeprecationWarning

rnd = mlib.round_nearest

_LOG2 = math.log(2)
```
### 31 - sympy/core/numbers.py:

Start line: 1779, End line: 1795

```python
class Rational(Number):

    def __ge__(self, other):
        try:
            other = _sympify(other)
        except SympifyError:
            raise TypeError("Invalid comparison %s >= %s" % (self, other))
        if other.is_NumberSymbol:
            return other.__le__(self)
        expr = self
        if other.is_Number:
            if other.is_Rational:
                 return _sympify(bool(self.p*other.q >= self.q*other.p))
            if other.is_Float:
                return _sympify(bool(mlib.mpf_ge(
                    self._as_mpf_val(other._prec), other._mpf_)))
        elif other.is_number and other.is_real:
            expr, other = Integer(self.p), self.q*other
        return Expr.__ge__(expr, other)
```
### 32 - sympy/core/numbers.py:

Start line: 1761, End line: 1777

```python
class Rational(Number):

    def __gt__(self, other):
        try:
            other = _sympify(other)
        except SympifyError:
            raise TypeError("Invalid comparison %s > %s" % (self, other))
        if other.is_NumberSymbol:
            return other.__lt__(self)
        expr = self
        if other.is_Number:
            if other.is_Rational:
                return _sympify(bool(self.p*other.q > self.q*other.p))
            if other.is_Float:
                return _sympify(bool(mlib.mpf_gt(
                    self._as_mpf_val(other._prec), other._mpf_)))
        elif other.is_number and other.is_real:
            expr, other = Integer(self.p), self.q*other
        return Expr.__gt__(expr, other)
```
### 34 - sympy/core/numbers.py:

Start line: 1252, End line: 1289

```python
class Float(Number):

    def __abs__(self):
        return Float._new(mlib.mpf_abs(self._mpf_), self._prec)

    def __int__(self):
        if self._mpf_ == _mpf_zero:
            return 0
        return int(mlib.to_int(self._mpf_))  # uses round_fast = round_down

    __long__ = __int__

    def __eq__(self, other):
        if isinstance(other, float):
            # coerce to Float at same precision
            o = Float(other)
            try:
                ompf = o._as_mpf_val(self._prec)
            except ValueError:
                return False
            return bool(mlib.mpf_eq(self._mpf_, ompf))
        try:
            other = _sympify(other)
        except SympifyError:
            return NotImplemented
        if other.is_NumberSymbol:
            if other.is_irrational:
                return False
            return other.__eq__(self)
        if other.is_Float:
            return bool(mlib.mpf_eq(self._mpf_, other._mpf_))
        if other.is_Number:
            # numbers should compare at the same precision;
            # all _as_mpf_val routines should be sure to abide
            # by the request to change the prec if necessary; if
            # they don't, the equality test will fail since it compares
            # the mpf tuples
            ompf = other._as_mpf_val(self._prec)
            return bool(mlib.mpf_eq(self._mpf_, ompf))
        return False    # Float != non-Number
```
### 35 - sympy/core/numbers.py:

Start line: 1591, End line: 1604

```python
class Rational(Number):
    __radd__ = __add__

    @_sympifyit('other', NotImplemented)
    def __sub__(self, other):
        if global_evaluate[0]:
            if isinstance(other, Integer):
                return Rational(self.p - self.q*other.p, self.q, 1)
            elif isinstance(other, Rational):
                return Rational(self.p*other.q - self.q*other.p, self.q*other.q)
            elif isinstance(other, Float):
                return -other + self
            else:
                return Number.__sub__(self, other)
        return Number.__sub__(self, other)
```
### 36 - sympy/core/numbers.py:

Start line: 1815, End line: 1831

```python
class Rational(Number):

    def __le__(self, other):
        try:
            other = _sympify(other)
        except SympifyError:
            raise TypeError("Invalid comparison %s <= %s" % (self, other))
        expr = self
        if other.is_NumberSymbol:
            return other.__ge__(self)
        elif other.is_Number:
            if other.is_Rational:
                return _sympify(bool(self.p*other.q <= self.q*other.p))
            if other.is_Float:
                return _sympify(bool(mlib.mpf_le(
                    self._as_mpf_val(other._prec), other._mpf_)))
        elif other.is_number and other.is_real:
            expr, other = Integer(self.p), self.q*other
        return Expr.__le__(expr, other)
```
### 37 - sympy/core/numbers.py:

Start line: 1311, End line: 1326

```python
class Float(Number):

    def __ge__(self, other):
        try:
            other = _sympify(other)
        except SympifyError:
            raise TypeError("Invalid comparison %s >= %s" % (self, other))
        if other.is_NumberSymbol:
            return other.__le__(self)
        if other.is_Rational and not other.is_Integer:
            self *= other.q
            other = _sympify(other.p)
        elif other.is_comparable:
            other = other.evalf()
        if other.is_Number and other is not S.NaN:
            return _sympify(bool(
                mlib.mpf_ge(self._mpf_, other._as_mpf_val(self._prec))))
        return Expr.__ge__(self, other)
```
### 38 - sympy/core/numbers.py:

Start line: 1328, End line: 1343

```python
class Float(Number):

    def __lt__(self, other):
        try:
            other = _sympify(other)
        except SympifyError:
            raise TypeError("Invalid comparison %s < %s" % (self, other))
        if other.is_NumberSymbol:
            return other.__gt__(self)
        if other.is_Rational and not other.is_Integer:
            self *= other.q
            other = _sympify(other.p)
        elif other.is_comparable:
            other = other.evalf()
        if other.is_Number and other is not S.NaN:
            return _sympify(bool(
                mlib.mpf_lt(self._mpf_, other._as_mpf_val(self._prec))))
        return Expr.__lt__(self, other)
```
### 39 - sympy/core/numbers.py:

Start line: 2074, End line: 2162

```python
class Integer(Rational):

    # TODO make it decorator + bytecodehacks?
    def __add__(self, other):
        if global_evaluate[0]:
            if isinstance(other, integer_types):
                return Integer(self.p + other)
            elif isinstance(other, Integer):
                return Integer(self.p + other.p)
            elif isinstance(other, Rational):
                return Rational(self.p*other.q + other.p, other.q, 1)
            return Rational.__add__(self, other)
        else:
            return Add(self, other)

    def __radd__(self, other):
        if global_evaluate[0]:
            if isinstance(other, integer_types):
                return Integer(other + self.p)
            elif isinstance(other, Rational):
                return Rational(other.p + self.p*other.q, other.q, 1)
            return Rational.__radd__(self, other)
        return Rational.__radd__(self, other)

    def __sub__(self, other):
        if global_evaluate[0]:
            if isinstance(other, integer_types):
                return Integer(self.p - other)
            elif isinstance(other, Integer):
                return Integer(self.p - other.p)
            elif isinstance(other, Rational):
                return Rational(self.p*other.q - other.p, other.q, 1)
            return Rational.__sub__(self, other)
        return Rational.__sub__(self, other)

    def __rsub__(self, other):
        if global_evaluate[0]:
            if isinstance(other, integer_types):
                return Integer(other - self.p)
            elif isinstance(other, Rational):
                return Rational(other.p - self.p*other.q, other.q, 1)
            return Rational.__rsub__(self, other)
        return Rational.__rsub__(self, other)

    def __mul__(self, other):
        if global_evaluate[0]:
            if isinstance(other, integer_types):
                return Integer(self.p*other)
            elif isinstance(other, Integer):
                return Integer(self.p*other.p)
            elif isinstance(other, Rational):
                return Rational(self.p*other.p, other.q, igcd(self.p, other.q))
            return Rational.__mul__(self, other)
        return Rational.__mul__(self, other)

    def __rmul__(self, other):
        if global_evaluate[0]:
            if isinstance(other, integer_types):
                return Integer(other*self.p)
            elif isinstance(other, Rational):
                return Rational(other.p*self.p, other.q, igcd(self.p, other.q))
            return Rational.__rmul__(self, other)
        return Rational.__rmul__(self, other)

    def __mod__(self, other):
        if global_evaluate[0]:
            if isinstance(other, integer_types):
                return Integer(self.p % other)
            elif isinstance(other, Integer):
                return Integer(self.p % other.p)
            return Rational.__mod__(self, other)
        return Rational.__mod__(self, other)

    def __rmod__(self, other):
        if global_evaluate[0]:
            if isinstance(other, integer_types):
                return Integer(other % self.p)
            elif isinstance(other, Integer):
                return Integer(other.p % self.p)
            return Rational.__rmod__(self, other)
        return Rational.__rmod__(self, other)

    def __eq__(self, other):
        if isinstance(other, integer_types):
            return (self.p == other)
        elif isinstance(other, Integer):
            return (self.p == other.p)
        return Rational.__eq__(self, other)

    def __ne__(self, other):
        return not self == other
```
### 40 - sympy/core/numbers.py:

Start line: 2029, End line: 2058

```python
class Integer(Rational):

    def __getnewargs__(self):
        return (self.p,)

    # Arithmetic operations are here for efficiency
    def __int__(self):
        return self.p

    __long__ = __int__

    def floor(self):
        return Integer(self.p)

    def ceiling(self):
        return Integer(self.p)

    def __neg__(self):
        return Integer(-self.p)

    def __abs__(self):
        if self.p >= 0:
            return self
        else:
            return Integer(-self.p)

    def __divmod__(self, other):
        from .containers import Tuple
        if isinstance(other, Integer) and global_evaluate[0]:
            return Tuple(*(divmod(self.p, other.p)))
        else:
            return Number.__divmod__(self, other)
```
### 41 - sympy/core/numbers.py:

Start line: 2793, End line: 2821

```python
class Infinity(with_metaclass(Singleton, Number)):

    @_sympifyit('other', NotImplemented)
    def __div__(self, other):
        if isinstance(other, Number):
            if other is S.Infinity or \
                other is S.NegativeInfinity or \
                    other is S.NaN:
                return S.NaN
            elif other.is_Float:
                if other == Float('-inf') or \
                        other == Float('inf'):
                    return S.NaN
                elif other.is_nonnegative:
                    return Float('inf')
                else:
                    return Float('-inf')
            else:
                if other >= 0:
                    return S.Infinity
                else:
                    return S.NegativeInfinity
        return NotImplemented

    __truediv__ = __div__

    def __abs__(self):
        return S.Infinity

    def __neg__(self):
        return S.NegativeInfinity
```
### 42 - sympy/core/numbers.py:

Start line: 1605, End line: 1616

```python
class Rational(Number):
    @_sympifyit('other', NotImplemented)
    def __rsub__(self, other):
        if global_evaluate[0]:
            if isinstance(other, Integer):
                return Rational(self.q*other.p - self.p, self.q, 1)
            elif isinstance(other, Rational):
                return Rational(self.q*other.p - self.p*other.q, self.q*other.q)
            elif isinstance(other, Float):
                return -self + other
            else:
                return Number.__rsub__(self, other)
        return Number.__rsub__(self, other)
```
### 43 - sympy/core/numbers.py:

Start line: 1345, End line: 1380

```python
class Float(Number):

    def __le__(self, other):
        try:
            other = _sympify(other)
        except SympifyError:
            raise TypeError("Invalid comparison %s <= %s" % (self, other))
        if other.is_NumberSymbol:
            return other.__ge__(self)
        if other.is_Rational and not other.is_Integer:
            self *= other.q
            other = _sympify(other.p)
        elif other.is_comparable:
            other = other.evalf()
        if other.is_Number and other is not S.NaN:
            return _sympify(bool(
                mlib.mpf_le(self._mpf_, other._as_mpf_val(self._prec))))
        return Expr.__le__(self, other)

    def __hash__(self):
        return super(Float, self).__hash__()

    def epsilon_eq(self, other, epsilon="1e-15"):
        return abs(self - other) < Float(epsilon)

    def _sage_(self):
        import sage.all as sage
        return sage.RealNumber(str(self))

    def __format__(self, format_spec):
        return format(decimal.Decimal(str(self)), format_spec)


# Add sympify converters
converter[float] = converter[decimal.Decimal] = Float

# this is here to work nicely in Sage
RealNumber = Float
```
### 44 - sympy/core/numbers.py:

Start line: 502, End line: 534

```python
class Number(AtomicExpr):
    """Represents atomic numbers in SymPy.

    Floating point numbers are represented by the Float class.
    Rational numbers (of any size) are represented by the Rational class.
    Integer numbers (of any size) are represented by the Integer class.
    Float and Rational are subclasses of Number; Integer is a subclass
    of Rational.

    For example, ``2/3`` is represented as ``Rational(2, 3)`` which is
    a different object from the floating point number obtained with
    Python division ``2/3``. Even for numbers that are exactly
    represented in binary, there is a difference between how two forms,
    such as ``Rational(1, 2)`` and ``Float(0.5)``, are used in SymPy.
    The rational form is to be preferred in symbolic computations.

    Other kinds of numbers, such as algebraic numbers ``sqrt(2)`` or
    complex numbers ``3 + 4*I``, are not instances of Number class as
    they are not atomic.

    See Also
    ========

    Float, Integer, Rational
    """
    is_commutative = True
    is_number = True
    is_Number = True

    __slots__ = []

    # Used to make max(x._prec, y._prec) return x._prec when only x is a float
    _prec = -1
```
### 45 - sympy/core/numbers.py:

Start line: 685, End line: 779

```python
class Number(AtomicExpr):

    @_sympifyit('other', NotImplemented)
    def __div__(self, other):
        if isinstance(other, Number) and global_evaluate[0]:
            if other is S.NaN:
                return S.NaN
            elif other is S.Infinity or other is S.NegativeInfinity:
                return S.Zero
        return AtomicExpr.__div__(self, other)

    __truediv__ = __div__

    def __eq__(self, other):
        raise NotImplementedError('%s needs .__eq__() method' %
            (self.__class__.__name__))

    def __ne__(self, other):
        raise NotImplementedError('%s needs .__ne__() method' %
            (self.__class__.__name__))

    def __lt__(self, other):
        try:
            other = _sympify(other)
        except SympifyError:
            raise TypeError("Invalid comparison %s < %s" % (self, other))
        raise NotImplementedError('%s needs .__lt__() method' %
            (self.__class__.__name__))

    def __le__(self, other):
        try:
            other = _sympify(other)
        except SympifyError:
            raise TypeError("Invalid comparison %s <= %s" % (self, other))
        raise NotImplementedError('%s needs .__le__() method' %
            (self.__class__.__name__))

    def __gt__(self, other):
        try:
            other = _sympify(other)
        except SympifyError:
            raise TypeError("Invalid comparison %s > %s" % (self, other))
        return _sympify(other).__lt__(self)

    def __ge__(self, other):
        try:
            other = _sympify(other)
        except SympifyError:
            raise TypeError("Invalid comparison %s >= %s" % (self, other))
        return _sympify(other).__le__(self)

    def __hash__(self):
        return super(Number, self).__hash__()

    def is_constant(self, *wrt, **flags):
        return True

    def as_coeff_mul(self, *deps, **kwargs):
        # a -> c*t
        if self.is_Rational or not kwargs.pop('rational', True):
            return self, tuple()
        elif self.is_negative:
            return S.NegativeOne, (-self,)
        return S.One, (self,)

    def as_coeff_add(self, *deps):
        # a -> c + t
        if self.is_Rational:
            return self, tuple()
        return S.Zero, (self,)

    def as_coeff_Mul(self, rational=False):
        """Efficiently extract the coefficient of a product. """
        if rational and not self.is_Rational:
            return S.One, self
        return (self, S.One) if self else (S.One, self)

    def as_coeff_Add(self, rational=False):
        """Efficiently extract the coefficient of a summation. """
        if not rational:
            return self, S.Zero
        return S.Zero, self

    def gcd(self, other):
        """Compute GCD of `self` and `other`. """
        from sympy.polys import gcd
        return gcd(self, other)

    def lcm(self, other):
        """Compute LCM of `self` and `other`. """
        from sympy.polys import lcm
        return lcm(self, other)

    def cofactors(self, other):
        """Compute GCD and cofactors of `self` and `other`. """
        from sympy.polys import cofactors
        return cofactors(self, other)
```
### 46 - sympy/core/numbers.py:

Start line: 2284, End line: 2334

```python
class Integer(Rational):

    def _eval_power(self, expt):
        # ... other code
        for prime, exponent in dict.items():
            exponent *= expt.p
            # remove multiples of expt.q: (2**12)**(1/10) -> 2*(2**2)**(1/10)
            div_e, div_m = divmod(exponent, expt.q)
            if div_e > 0:
                out_int *= prime**div_e
            if div_m > 0:
                # see if the reduced exponent shares a gcd with e.q
                # (2**2)**(1/10) -> 2**(1/5)
                g = igcd(div_m, expt.q)
                if g != 1:
                    out_rad *= Pow(prime, Rational(div_m//g, expt.q//g))
                else:
                    sqr_dict[prime] = div_m
        # identify gcd of remaining powers
        for p, ex in sqr_dict.items():
            if sqr_gcd == 0:
                sqr_gcd = ex
            else:
                sqr_gcd = igcd(sqr_gcd, ex)
                if sqr_gcd == 1:
                    break
        for k, v in sqr_dict.items():
            sqr_int *= k**(v//sqr_gcd)
        if sqr_int == b_pos and out_int == 1 and out_rad == 1:
            result = None
        else:
            result = out_int*out_rad*Pow(sqr_int, Rational(sqr_gcd, expt.q))
            if self.is_negative:
                result *= Pow(S.NegativeOne, expt)
        return result

    def _eval_is_prime(self):
        from sympy.ntheory import isprime

        return isprime(self)

    def _eval_is_composite(self):
        if self > 1:
            return fuzzy_not(self.is_prime)
        else:
            return False

    def as_numer_denom(self):
        return self, S.One

    def __floordiv__(self, other):
        return Integer(self.p // Integer(other).p)

    def __rfloordiv__(self, other):
        return Integer(Integer(other).p // self.p)
```
### 47 - sympy/core/numbers.py:

Start line: 1475, End line: 1548

```python
class Rational(Number):

    @cacheit
    def __new__(cls, p, q=None, gcd=None):
        if q is None:
            if isinstance(p, Rational):
                return p

            if isinstance(p, SYMPY_INTS):
                pass
            else:
                if isinstance(p, (float, Float)):
                    return Rational(*_as_integer_ratio(p))

                if not isinstance(p, string_types):
                    try:
                        p = sympify(p)
                    except (SympifyError, SyntaxError):
                        pass  # error will raise below
                else:
                    if p.count('/') > 1:
                        raise TypeError('invalid input: %s' % p)
                    p = p.replace(' ', '')
                    pq = p.rsplit('/', 1)
                    if len(pq) == 2:
                        p, q = pq
                        fp = fractions.Fraction(p)
                        fq = fractions.Fraction(q)
                        p = fp/fq
                    try:
                        p = fractions.Fraction(p)
                    except ValueError:
                        pass  # error will raise below
                    else:
                        return Rational(p.numerator, p.denominator, 1)

                if not isinstance(p, Rational):
                    raise TypeError('invalid input: %s' % p)

            q = 1
            gcd = 1
        else:
            p = Rational(p)
            q = Rational(q)

        if isinstance(q, Rational):
            p *= q.q
            q = q.p
        if isinstance(p, Rational):
            q *= p.q
            p = p.p

        # p and q are now integers
        if q == 0:
            if p == 0:
                if _errdict["divide"]:
                    raise ValueError("Indeterminate 0/0")
                else:
                    return S.NaN
            return S.ComplexInfinity
        if q < 0:
            q = -q
            p = -p
        if not gcd:
            gcd = igcd(abs(p), q)
        if gcd > 1:
            p //= gcd
            q //= gcd
        if q == 1:
            return Integer(p)
        if p == 1 and q == 2:
            return S.Half
        obj = Expr.__new__(cls)
        obj.p = p
        obj.q = q
        return obj
```
### 48 - sympy/core/numbers.py:

Start line: 2164, End line: 2209

```python
class Integer(Rational):

    def __gt__(self, other):
        try:
            other = _sympify(other)
        except SympifyError:
            raise TypeError("Invalid comparison %s > %s" % (self, other))
        if other.is_Integer:
            return _sympify(self.p > other.p)
        return Rational.__gt__(self, other)

    def __lt__(self, other):
        try:
            other = _sympify(other)
        except SympifyError:
            raise TypeError("Invalid comparison %s < %s" % (self, other))
        if other.is_Integer:
            return _sympify(self.p < other.p)
        return Rational.__lt__(self, other)

    def __ge__(self, other):
        try:
            other = _sympify(other)
        except SympifyError:
            raise TypeError("Invalid comparison %s >= %s" % (self, other))
        if other.is_Integer:
            return _sympify(self.p >= other.p)
        return Rational.__ge__(self, other)

    def __le__(self, other):
        try:
            other = _sympify(other)
        except SympifyError:
            raise TypeError("Invalid comparison %s <= %s" % (self, other))
        if other.is_Integer:
            return _sympify(self.p <= other.p)
        return Rational.__le__(self, other)

    def __hash__(self):
        return hash(self.p)

    def __index__(self):
        return self.p

    ########################################

    def _eval_is_odd(self):
        return bool(self.p % 2)
```
### 49 - sympy/core/numbers.py:

Start line: 2862, End line: 2933

```python
class Infinity(with_metaclass(Singleton, Number)):

    def _as_mpf_val(self, prec):
        return mlib.finf

    def _sage_(self):
        import sage.all as sage
        return sage.oo

    def __hash__(self):
        return super(Infinity, self).__hash__()

    def __eq__(self, other):
        return other is S.Infinity

    def __ne__(self, other):
        return other is not S.Infinity

    def __lt__(self, other):
        try:
            other = _sympify(other)
        except SympifyError:
            raise TypeError("Invalid comparison %s < %s" % (self, other))
        if other.is_real:
            return S.false
        return Expr.__lt__(self, other)

    def __le__(self, other):
        try:
            other = _sympify(other)
        except SympifyError:
            raise TypeError("Invalid comparison %s <= %s" % (self, other))
        if other.is_real:
            if other.is_finite or other is S.NegativeInfinity:
                return S.false
            elif other.is_nonpositive:
                return S.false
            elif other.is_infinite and other.is_positive:
                return S.true
        return Expr.__le__(self, other)

    def __gt__(self, other):
        try:
            other = _sympify(other)
        except SympifyError:
            raise TypeError("Invalid comparison %s > %s" % (self, other))
        if other.is_real:
            if other.is_finite or other is S.NegativeInfinity:
                return S.true
            elif other.is_nonpositive:
                return S.true
            elif other.is_infinite and other.is_positive:
                return S.false
        return Expr.__gt__(self, other)

    def __ge__(self, other):
        try:
            other = _sympify(other)
        except SympifyError:
            raise TypeError("Invalid comparison %s >= %s" % (self, other))
        if other.is_real:
            return S.true
        return Expr.__ge__(self, other)

    def __mod__(self, other):
        return S.NaN

    __rmod__ = __mod__

    def floor(self):
        return self

    def ceiling(self):
        return self
```
### 50 - sympy/core/numbers.py:

Start line: 1578, End line: 1590

```python
class Rational(Number):

    @_sympifyit('other', NotImplemented)
    def __add__(self, other):
        if global_evaluate[0]:
            if isinstance(other, Integer):
                return Rational(self.p + self.q*other.p, self.q, 1)
            elif isinstance(other, Rational):
                #TODO: this can probably be optimized more
                return Rational(self.p*other.q + self.q*other.p, self.q*other.q)
            elif isinstance(other, Float):
                return other + self
            else:
                return Number.__add__(self, other)
        return Number.__add__(self, other)
```
### 53 - sympy/core/numbers.py:

Start line: 3486, End line: 3554

```python
E = S.Exp1


class Pi(with_metaclass(Singleton, NumberSymbol)):
    r"""The `\pi` constant.

    The transcendental number `\pi = 3.141592654\ldots` represents the ratio
    of a circle's circumference to its diameter, the area of the unit circle,
    the half-period of trigonometric functions, and many other things
    in mathematics.

    Pi is a singleton, and can be accessed by ``S.Pi``, or can
    be imported as ``pi``.

    Examples
    ========

    >>> from sympy import S, pi, oo, sin, exp, integrate, Symbol
    >>> S.Pi
    pi
    >>> pi > 3
    True
    >>> pi.is_irrational
    True
    >>> x = Symbol('x')
    >>> sin(x + 2*pi)
    sin(x)
    >>> integrate(exp(-x**2), (x, -oo, oo))
    sqrt(pi)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Pi
    """

    is_real = True
    is_positive = True
    is_negative = False
    is_irrational = True
    is_number = True
    is_algebraic = False
    is_transcendental = True

    __slots__ = []

    def _latex(self, printer):
        return r"\pi"

    @staticmethod
    def __abs__():
        return S.Pi

    def __int__(self):
        return 3

    def _as_mpf_val(self, prec):
        return mpf_pi(prec)

    def approximation_interval(self, number_cls):
        if issubclass(number_cls, Integer):
            return (Integer(3), Integer(4))
        elif issubclass(number_cls, Rational):
            return (Rational(223, 71), Rational(22, 7))

    def _sage_(self):
        import sage.all as sage
        return sage.pi
pi = S.Pi
```
### 54 - sympy/core/numbers.py:

Start line: 3557, End line: 3618

```python
class GoldenRatio(with_metaclass(Singleton, NumberSymbol)):
    r"""The golden ratio, `\phi`.

    `\phi = \frac{1 + \sqrt{5}}{2}` is algebraic number.  Two quantities
    are in the golden ratio if their ratio is the same as the ratio of
    their sum to the larger of the two quantities, i.e. their maximum.

    GoldenRatio is a singleton, and can be accessed by ``S.GoldenRatio``.

    Examples
    ========

    >>> from sympy import S
    >>> S.GoldenRatio > 1
    True
    >>> S.GoldenRatio.expand(func=True)
    1/2 + sqrt(5)/2
    >>> S.GoldenRatio.is_irrational
    True

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Golden_ratio
    """

    is_real = True
    is_positive = True
    is_negative = False
    is_irrational = True
    is_number = True
    is_algebraic = True
    is_transcendental = False

    __slots__ = []

    def _latex(self, printer):
        return r"\phi"

    def __int__(self):
        return 1

    def _as_mpf_val(self, prec):
         # XXX track down why this has to be increased
        rv = mlib.from_man_exp(phi_fixed(prec + 10), -prec - 10)
        return mpf_norm(rv, prec)

    def _eval_expand_func(self, **hints):
        from sympy import sqrt
        return S.Half + S.Half*sqrt(5)

    def approximation_interval(self, number_cls):
        if issubclass(number_cls, Integer):
            return (S.One, Rational(2))
        elif issubclass(number_cls, Rational):
            pass

    def _sage_(self):
        import sage.all as sage
        return sage.golden_ratio

    _eval_rewrite_as_sqrt = _eval_expand_func
```
### 56 - sympy/core/numbers.py:

Start line: 1032, End line: 1081

```python
class Float(Number):

    def __new__(cls, num, dps=None, prec=None, precision=None):
        # ... other code

        if isinstance(num, float):
            _mpf_ = mlib.from_float(num, precision, rnd)
        elif isinstance(num, string_types):
            _mpf_ = mlib.from_str(num, precision, rnd)
        elif isinstance(num, decimal.Decimal):
            if num.is_finite():
                _mpf_ = mlib.from_str(str(num), precision, rnd)
            elif num.is_nan():
                _mpf_ = _mpf_nan
            elif num.is_infinite():
                if num > 0:
                    _mpf_ = _mpf_inf
                else:
                    _mpf_ = _mpf_ninf
            else:
                raise ValueError("unexpected decimal value %s" % str(num))
        elif isinstance(num, tuple) and len(num) in (3, 4):
            if type(num[1]) is str:
                # it's a hexadecimal (coming from a pickled object)
                # assume that it is in standard form
                num = list(num)
                # If we're loading an object pickled in Python 2 into
                # Python 3, we may need to strip a tailing 'L' because
                # of a shim for int on Python 3, see issue #13470.
                if num[1].endswith('L'):
                    num[1] = num[1][:-1]
                num[1] = MPZ(num[1], 16)
                _mpf_ = tuple(num)
            else:
                if len(num) == 4:
                    # handle normalization hack
                    return Float._new(num, precision)
                else:
                    return (S.NegativeOne**num[0]*num[1]*S(2)**num[2]).evalf(precision)
        else:
            try:
                _mpf_ = num._as_mpf_val(precision)
            except (NotImplementedError, AttributeError):
                _mpf_ = mpmath.mpf(num, prec=precision)._mpf_

        # special cases
        if _mpf_ == _mpf_zero:
            pass  # we want a Float
        elif _mpf_ == _mpf_nan:
            return S.NaN

        obj = Expr.__new__(cls)
        obj._mpf_ = _mpf_
        obj._prec = precision
        return obj
```
### 57 - sympy/core/numbers.py:

Start line: 2521, End line: 2548

```python
class Zero(with_metaclass(Singleton, IntegerConstant)):

    def _eval_power(self, expt):
        if expt.is_positive:
            return self
        if expt.is_negative:
            return S.ComplexInfinity
        if expt.is_real is False:
            return S.NaN
        # infinities are already handled with pos and neg
        # tests above; now throw away leading numbers on Mul
        # exponent
        coeff, terms = expt.as_coeff_Mul()
        if coeff.is_negative:
            return S.ComplexInfinity**terms
        if coeff is not S.One:  # there is a Number to discard
            return self**terms

    def _eval_order(self, *symbols):
        # Order(0,x) -> 0
        return self

    def __nonzero__(self):
        return False

    __bool__ = __nonzero__

    def as_coeff_Mul(self, rational=False):  # XXX this routine should be deleted
        """Efficiently extract the coefficient of a summation. """
        return S.One, self
```
### 58 - sympy/core/numbers.py:

Start line: 1083, End line: 1095

```python
class Float(Number):

    @classmethod
    def _new(cls, _mpf_, _prec):
        # special cases
        if _mpf_ == _mpf_zero:
            return S.Zero  # XXX this is different from Float which gives 0.0
        elif _mpf_ == _mpf_nan:
            return S.NaN

        obj = Expr.__new__(cls)
        obj._mpf_ = mpf_norm(_mpf_, _prec)
        # XXX: Should this be obj._prec = obj._mpf_[3]?
        obj._prec = _prec
        return obj
```
### 59 - sympy/core/numbers.py:

Start line: 937, End line: 1030

```python
class Float(Number):
    __slots__ = ['_mpf_', '_prec']

    # A Float represents many real numbers,
    # both rational and irrational.
    is_rational = None
    is_irrational = None
    is_number = True

    is_real = True

    is_Float = True

    def __new__(cls, num, dps=None, prec=None, precision=None):
        if prec is not None:
            SymPyDeprecationWarning(
                            feature="Using 'prec=XX' to denote decimal precision",
                            useinstead="'dps=XX' for decimal precision and 'precision=XX' "\
                                              "for binary precision",
                            issue=12820,
                            deprecated_since_version="1.1").warn()
            dps = prec
        del prec  # avoid using this deprecated kwarg

        if dps is not None and precision is not None:
            raise ValueError('Both decimal and binary precision supplied. '
                             'Supply only one. ')

        if isinstance(num, string_types):
            num = num.replace(' ', '')
            if num.startswith('.') and len(num) > 1:
                num = '0' + num
            elif num.startswith('-.') and len(num) > 2:
                num = '-0.' + num[2:]
        elif isinstance(num, float) and num == 0:
            num = '0'
        elif isinstance(num, (SYMPY_INTS, Integer)):
            num = str(num)  # faster than mlib.from_int
        elif num is S.Infinity:
            num = '+inf'
        elif num is S.NegativeInfinity:
            num = '-inf'
        elif type(num).__module__ == 'numpy': # support for numpy datatypes
            num = _convert_numpy_types(num)
        elif isinstance(num, mpmath.mpf):
            if precision is None:
                if dps is None:
                    precision = num.context.prec
            num = num._mpf_

        if dps is None and precision is None:
            dps = 15
            if isinstance(num, Float):
                return num
            if isinstance(num, string_types) and _literal_float(num):
                try:
                    Num = decimal.Decimal(num)
                except decimal.InvalidOperation:
                    pass
                else:
                    isint = '.' not in num
                    num, dps = _decimal_to_Rational_prec(Num)
                    if num.is_Integer and isint:
                        dps = max(dps, len(str(num).lstrip('-')))
                    dps = max(15, dps)
                    precision = mlib.libmpf.dps_to_prec(dps)
        elif precision == '' and dps is None or precision is None and dps == '':
            if not isinstance(num, string_types):
                raise ValueError('The null string can only be used when '
                'the number to Float is passed as a string or an integer.')
            ok = None
            if _literal_float(num):
                try:
                    Num = decimal.Decimal(num)
                except decimal.InvalidOperation:
                    pass
                else:
                    isint = '.' not in num
                    num, dps = _decimal_to_Rational_prec(Num)
                    if num.is_Integer and isint:
                        dps = max(dps, len(str(num).lstrip('-')))
                        precision = mlib.libmpf.dps_to_prec(dps)
                    ok = True
            if ok is None:
                raise ValueError('string-float not recognized: %s' % num)

        # decimal precision(dps) is set and maybe binary precision(precision)
        # as well.From here on binary precision is used to compute the Float.
        # Hence, if supplied use binary precision else translate from decimal
        # precision.

        if precision is None or precision == '':
            precision = mlib.libmpf.dps_to_prec(dps)

        precision = int(precision)
        # ... other code
```
### 68 - sympy/core/numbers.py:

Start line: 2598, End line: 2634

```python
class NegativeOne(with_metaclass(Singleton, IntegerConstant)):
    """The number negative one.

    NegativeOne is a singleton, and can be accessed by ``S.NegativeOne``.

    Examples
    ========

    >>> from sympy import S, Integer
    >>> Integer(-1) is S.NegativeOne
    True

    See Also
    ========

    One

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/%E2%88%921_%28number%29

    """
    is_number = True

    p = -1
    q = 1

    __slots__ = []

    @staticmethod
    def __abs__():
        return S.One

    @staticmethod
    def __neg__():
        return S.One
```
### 69 - sympy/core/numbers.py:

Start line: 2211, End line: 2283

```python
class Integer(Rational):

    def _eval_power(self, expt):
        """
        Tries to do some simplifications on self**expt

        Returns None if no further simplifications can be done

        When exponent is a fraction (so we have for example a square root),
        we try to find a simpler representation by factoring the argument
        up to factors of 2**15, e.g.

          - sqrt(4) becomes 2
          - sqrt(-4) becomes 2*I
          - (2**(3+7)*3**(6+7))**Rational(1,7) becomes 6*18**(3/7)

        Further simplification would require a special call to factorint on
        the argument which is not done here for sake of speed.

        """
        from sympy import perfect_power

        if expt is S.Infinity:
            if self.p > S.One:
                return S.Infinity
            # cases -1, 0, 1 are done in their respective classes
            return S.Infinity + S.ImaginaryUnit*S.Infinity
        if expt is S.NegativeInfinity:
            return Rational(1, self)**S.Infinity
        if not isinstance(expt, Number):
            # simplify when expt is even
            # (-2)**k --> 2**k
            if self.is_negative and expt.is_even:
                return (-self)**expt
        if isinstance(expt, Float):
            # Rational knows how to exponentiate by a Float
            return super(Integer, self)._eval_power(expt)
        if not isinstance(expt, Rational):
            return
        if expt is S.Half and self.is_negative:
            # we extract I for this special case since everyone is doing so
            return S.ImaginaryUnit*Pow(-self, expt)
        if expt.is_negative:
            # invert base and change sign on exponent
            ne = -expt
            if self.is_negative:
                    return S.NegativeOne**expt*Rational(1, -self)**ne
            else:
                return Rational(1, self.p)**ne
        # see if base is a perfect root, sqrt(4) --> 2
        x, xexact = integer_nthroot(abs(self.p), expt.q)
        if xexact:
            # if it's a perfect root we've finished
            result = Integer(x**abs(expt.p))
            if self.is_negative:
                result *= S.NegativeOne**expt
            return result

        # The following is an algorithm where we collect perfect roots
        # from the factors of base.

        # if it's not an nth root, it still might be a perfect power
        b_pos = int(abs(self.p))
        p = perfect_power(b_pos)
        if p is not False:
            dict = {p[0]: p[1]}
        else:
            dict = Integer(b_pos).factors(limit=2**15)

        # now process the dict of factors
        out_int = 1  # integer part
        out_rad = 1  # extracted radicals
        sqr_int = 1
        sqr_gcd = 0
        sqr_dict = {}
        # ... other code
```
### 71 - sympy/core/numbers.py:

Start line: 206, End line: 224

```python
try:
    from math import gcd as igcd2
except ImportError:
    def igcd2(a, b):
        """Compute gcd of two Python integers a and b."""
        if (a.bit_length() > BIGBITS and
            b.bit_length() > BIGBITS):
            return igcd_lehmer(a, b)

        a, b = abs(a), abs(b)
        while b:
            a, b = b, a % b
        return a


# Use Lehmer's algorithm only for very large numbers.
# The limit could be different on Python 2.7 and 3.x.
# If so, then this could be defined in compatibility.py.
BIGBITS = 5000
```
