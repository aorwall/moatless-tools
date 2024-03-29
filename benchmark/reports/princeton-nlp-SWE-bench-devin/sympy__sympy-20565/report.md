# sympy__sympy-20565

| **sympy/sympy** | `7813fc7f409838fe4c317321fd11c285a98b4ceb` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 332 |
| **Any found context length** | 332 |
| **Avg pos** | 1.0 |
| **Min pos** | 1 |
| **Max pos** | 1 |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sympy/sets/fancysets.py b/sympy/sets/fancysets.py
--- a/sympy/sets/fancysets.py
+++ b/sympy/sets/fancysets.py
@@ -42,8 +42,6 @@ class Rationals(Set, metaclass=Singleton):
     def _contains(self, other):
         if not isinstance(other, Expr):
             return False
-        if other.is_Number:
-            return other.is_Rational
         return other.is_rational
 
     def __iter__(self):

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sympy/sets/fancysets.py | 45 | 46 | 1 | 1 | 332


## Problem Statement

```
Rationals does not contain floats
The `Rationals` set should contain all floating point numbers.

\`\`\`python
import sympy

sympy.Rationals.contains(0.5)
\`\`\`

returns `False` but should return `True`

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 sympy/sets/fancysets.py** | 20 | 66| 332 | 332 | 10831 | 
| 2 | 2 sympy/core/numbers.py | 1873 | 1917| 400 | 732 | 40569 | 
| 3 | 2 sympy/core/numbers.py | 1919 | 1941| 184 | 916 | 40569 | 
| 4 | 2 sympy/core/numbers.py | 1386 | 1425| 346 | 1262 | 40569 | 
| 5 | 2 sympy/core/numbers.py | 1943 | 1976| 241 | 1503 | 40569 | 
| 6 | 2 sympy/core/numbers.py | 1330 | 1352| 282 | 1785 | 40569 | 
| 7 | 2 sympy/core/numbers.py | 1510 | 1600| 592 | 2377 | 40569 | 
| 8 | 2 sympy/core/numbers.py | 1761 | 1775| 160 | 2537 | 40569 | 
| 9 | 2 sympy/core/numbers.py | 1776 | 1787| 138 | 2675 | 40569 | 
| 10 | 2 sympy/core/numbers.py | 1846 | 1871| 180 | 2855 | 40569 | 
| 11 | 2 sympy/core/numbers.py | 1789 | 1806| 186 | 3041 | 40569 | 
| 12 | 2 sympy/core/numbers.py | 1693 | 1720| 214 | 3255 | 40569 | 
| 13 | 2 sympy/core/numbers.py | 1747 | 1759| 139 | 3394 | 40569 | 
| 14 | 2 sympy/core/numbers.py | 1721 | 1734| 129 | 3523 | 40569 | 
| 15 | 2 sympy/core/numbers.py | 205 | 223| 186 | 3709 | 40569 | 
| 16 | 2 sympy/core/numbers.py | 1457 | 1507| 360 | 4069 | 40569 | 
| 17 | 2 sympy/core/numbers.py | 1735 | 1746| 123 | 4192 | 40569 | 
| 18 | 2 sympy/core/numbers.py | 1990 | 2019| 203 | 4395 | 40569 | 
| 19 | 2 sympy/core/numbers.py | 1427 | 1455| 304 | 4699 | 40569 | 
| 20 | 2 sympy/core/numbers.py | 1808 | 1844| 404 | 5103 | 40569 | 
| 21 | 2 sympy/core/numbers.py | 1316 | 1328| 173 | 5276 | 40569 | 
| 22 | 2 sympy/core/numbers.py | 1602 | 1675| 488 | 5764 | 40569 | 
| 23 | 3 sympy/polys/domains/pythonrational.py | 1 | 100| 621 | 6385 | 42556 | 
| 24 | 3 sympy/core/numbers.py | 1219 | 1314| 767 | 7152 | 42556 | 
| 25 | 3 sympy/core/numbers.py | 2266 | 2302| 231 | 7383 | 42556 | 
| 26 | 4 sympy/core/expr.py | 2609 | 2677| 522 | 7905 | 76248 | 
| 27 | 4 sympy/polys/domains/pythonrational.py | 205 | 279| 468 | 8373 | 76248 | 
| 28 | 5 sympy/assumptions/ask.py | 289 | 317| 139 | 8512 | 87294 | 
| 29 | **5 sympy/sets/fancysets.py** | 231 | 267| 218 | 8730 | 87294 | 
| 30 | 5 sympy/core/numbers.py | 1029 | 1103| 651 | 9381 | 87294 | 
| 31 | 6 sympy/assumptions/handlers/sets.py | 97 | 162| 421 | 9802 | 92268 | 
| 32 | 6 sympy/core/numbers.py | 567 | 602| 306 | 10108 | 92268 | 
| 33 | 7 sympy/core/basic.py | 635 | 680| 374 | 10482 | 108103 | 
| 34 | 7 sympy/assumptions/ask.py | 319 | 349| 171 | 10653 | 108103 | 
| 35 | 7 sympy/core/numbers.py | 2564 | 2623| 274 | 10927 | 108103 | 
| 36 | 7 sympy/core/numbers.py | 1354 | 1384| 333 | 11260 | 108103 | 
| 37 | 7 sympy/core/numbers.py | 866 | 1028| 1446 | 12706 | 108103 | 
| 38 | 8 sympy/core/function.py | 3266 | 3355| 789 | 13495 | 136165 | 
| 39 | 9 sympy/parsing/sympy_parser.py | 784 | 800| 131 | 13626 | 144417 | 
| 40 | 9 sympy/polys/domains/pythonrational.py | 178 | 203| 263 | 13889 | 144417 | 
| 41 | 9 sympy/polys/domains/pythonrational.py | 102 | 121| 222 | 14111 | 144417 | 
| 42 | 9 sympy/core/numbers.py | 2021 | 2047| 187 | 14298 | 144417 | 
| 43 | 10 sympy/simplify/sqrtdenest.py | 45 | 70| 218 | 14516 | 151018 | 
| 44 | 10 sympy/core/numbers.py | 1105 | 1149| 423 | 14939 | 151018 | 
| 45 | 10 sympy/core/numbers.py | 178 | 202| 184 | 15123 | 151018 | 
| 46 | 10 sympy/assumptions/ask.py | 67 | 125| 496 | 15619 | 151018 | 
| 47 | 10 sympy/polys/domains/pythonrational.py | 153 | 176| 232 | 15851 | 151018 | 
| 48 | 11 sympy/concrete/guess.py | 110 | 169| 553 | 16404 | 155987 | 
| 49 | 11 sympy/core/numbers.py | 1151 | 1200| 522 | 16926 | 155987 | 
| 50 | 11 sympy/core/numbers.py | 1677 | 1691| 128 | 17054 | 155987 | 
| 51 | 11 sympy/core/numbers.py | 2167 | 2264| 786 | 17840 | 155987 | 
| 52 | 11 sympy/core/numbers.py | 2767 | 2796| 146 | 17986 | 155987 | 


### Hint

```
Under the assumptions system, Float.is_rational intentionally gives None. I think the sets should follow the same strict rules. The issue is that while it is true that floating point numbers are represented by a rational number, they are not rational numbers in the sense that they do not follow the behavior of rational numbers. 

IMO this should be "wont fix". 
If `Float.is_rational` gives `None` then I would expect `Rational.contains` to give something like an unevaluated `Contains` rather than `False`.

The way I would see this is that a Float represents a number that is only known approximately. The precise internal representation is always rational but we think of it as representing some true number that lies in an interval around that rational number. On that basis we don't know if it is really rational or not because irrational numbers are always arbitrarily close to any rational number. On that reasoning it makes sense that `is_rational` gives `None` because neither `True` or `False` are known to be correct. Following the same reasoning `Rational.contains` should also be indeterminate rather than `False`.
I naïvely thought that this was simply a bug. Thanks @asmeurer and @oscarbenjamin for your insights. There are three cases then for the result of the `Rational.contains`.

1. False (the current result)

I would still argue that this is wrong. False means that a float is not rational. This would mean that it is either irrational or complex. I think we could rule out a float being complex so then it has to be irrational. I believe that this is clearly not true.

2. Indeterminate

The arguments given above for this are pretty strong and I think this is better than option 1. Indeterminate would mean that it is unknown whether it is rational or irrational. Given the argument from @oscarbenjamin that a float represents an approximation of an underlying number it is clear that we don't know if it is irrational or rational.

3. True

If we instead see the float as the actual exact number instead of it being an approximation of another underlying number it would make most sense to say that all floats are rationals. It is indeed impossible to represent any irrational number as a float. In my example the 0.5 meant, at least to me, the exact number 0.5, i.e. one half, which is clearly rational. It also doesn't feel useful to keep it as unresolved or indeterminate because no amount of additional information could ever resolve this. Like in this example:

\`\`\`python
import sympy

x = sympy.Symbol('x')
expr = sympy.Rationals.contains(x)
# expr = Contains(x, Rationals)
expr.subs({x: 1})
sympy.Rationals.contains(x)
# True
\`\`\`

I  would argue that option 3 is the best option and that option 2 is better than option 1.
> In my example the 0.5 meant, at least to me, the exact number 0.5, i.e. one half

This is a fundamental conceptual problem in sympy. Most users who use floats do not intend for them to have the effect that they have.
I found an argument for the current False result:

If we see a float as an approximation of an underlying real number it could make sense to say that a float is (at least most likely) irrational since most real numbers are irrational. So it turns out that cases could be made for all three options.

I agree with the last comment from @oscarbenjamin. A float does not behave as you think.

I guess we could close this issue if not someone else thinks that `True` or `Indeterminate` would be better options. I am not so certain any longer.
I think that at least the results for `is_rational` and `contains` are inconsistent so there is an issue here in any case.
I can work on making `contains` and `is_rational` both indeterminate. I think that could be a good first step for conformity and then the community can continue the discussion on the underlying assumption.
I think that the fix is just:
\`\`\`diff
diff --git a/sympy/sets/fancysets.py b/sympy/sets/fancysets.py
index 844c9ee9c1..295e2e7e7c 100644
--- a/sympy/sets/fancysets.py
+++ b/sympy/sets/fancysets.py
@@ -42,8 +42,6 @@ class Rationals(Set, metaclass=Singleton):
     def _contains(self, other):
         if not isinstance(other, Expr):
             return False
-        if other.is_Number:
-            return other.is_Rational
         return other.is_rational
 
     def __iter__(self):
\`\`\`
There is at least one test in sets that would need to be updated though.
When comparing sympy thinks that 0.5 and 1/2 are equal.

\`\`\`python
> sympy.Eq(sympy.Rational(1, 2), 0.5)
True
\`\`\`

To be consistent this should also be indeterminate. Or by letting floats be rational.
There's discussion about this at #20033, but to be sure, `==` in SymPy means strict structural equality, not mathematical equality. So it shouldn't necessarily match the semantics here. Eq is more mathematical so there is perhaps a stronger argument to make it unevaluated. 
```

## Patch

```diff
diff --git a/sympy/sets/fancysets.py b/sympy/sets/fancysets.py
--- a/sympy/sets/fancysets.py
+++ b/sympy/sets/fancysets.py
@@ -42,8 +42,6 @@ class Rationals(Set, metaclass=Singleton):
     def _contains(self, other):
         if not isinstance(other, Expr):
             return False
-        if other.is_Number:
-            return other.is_Rational
         return other.is_rational
 
     def __iter__(self):

```

## Test Patch

```diff
diff --git a/sympy/sets/tests/test_fancysets.py b/sympy/sets/tests/test_fancysets.py
--- a/sympy/sets/tests/test_fancysets.py
+++ b/sympy/sets/tests/test_fancysets.py
@@ -1046,7 +1046,7 @@ def test_Rationals():
         Rational(1, 3), 3, Rational(-1, 3), -3, Rational(2, 3)]
     assert Basic() not in S.Rationals
     assert S.Half in S.Rationals
-    assert 1.0 not in S.Rationals
+    assert S.Rationals.contains(0.5) == Contains(0.5, S.Rationals, evaluate=False)
     assert 2 in S.Rationals
     r = symbols('r', rational=True)
     assert r in S.Rationals

```


## Code snippets

### 1 - sympy/sets/fancysets.py:

Start line: 20, End line: 66

```python
class Rationals(Set, metaclass=Singleton):
    """
    Represents the rational numbers. This set is also available as
    the Singleton, S.Rationals.

    Examples
    ========

    >>> from sympy import S
    >>> S.Half in S.Rationals
    True
    >>> iterable = iter(S.Rationals)
    >>> [next(iterable) for i in range(12)]
    [0, 1, -1, 1/2, 2, -1/2, -2, 1/3, 3, -1/3, -3, 2/3]
    """

    is_iterable = True
    _inf = S.NegativeInfinity
    _sup = S.Infinity
    is_empty = False
    is_finite_set = False

    def _contains(self, other):
        if not isinstance(other, Expr):
            return False
        if other.is_Number:
            return other.is_Rational
        return other.is_rational

    def __iter__(self):
        from sympy.core.numbers import igcd, Rational
        yield S.Zero
        yield S.One
        yield S.NegativeOne
        d = 2
        while True:
            for n in range(d):
                if igcd(n, d) == 1:
                    yield Rational(n, d)
                    yield Rational(d, n)
                    yield Rational(-n, d)
                    yield Rational(-d, n)
            d += 1

    @property
    def _boundary(self):
        return S.Reals
```
### 2 - sympy/core/numbers.py:

Start line: 1873, End line: 1917

```python
class Rational(Number):

    def __eq__(self, other):
        from sympy.core.power import integer_log
        try:
            other = _sympify(other)
        except SympifyError:
            return NotImplemented
        if not isinstance(other, Number):
            # S(0) == S.false is False
            # S(0) == False is True
            return False
        if not self:
            return not other
        if other.is_NumberSymbol:
            if other.is_irrational:
                return False
            return other.__eq__(self)
        if other.is_Rational:
            # a Rational is always in reduced form so will never be 2/4
            # so we can just check equivalence of args
            return self.p == other.p and self.q == other.q
        if other.is_Float:
            # all Floats have a denominator that is a power of 2
            # so if self doesn't, it can't be equal to other
            if self.q & (self.q - 1):
                return False
            s, m, t = other._mpf_[:3]
            if s:
                m = -m
            if not t:
                # other is an odd integer
                if not self.is_Integer or self.is_even:
                    return False
                return m == self.p
            if t > 0:
                # other is an even integer
                if not self.is_Integer:
                    return False
                # does m*2**t == self.p
                return self.p and not self.p % m and \
                    integer_log(self.p//m, 2) == (t, True)
            # does non-integer s*m/2**-t = p/q?
            if self.is_Integer:
                return False
            return m == self.p and integer_log(self.q, 2) == (-t, True)
        return False
```
### 3 - sympy/core/numbers.py:

Start line: 1919, End line: 1941

```python
class Rational(Number):

    def __ne__(self, other):
        return not self == other

    def _Rrel(self, other, attr):
        # if you want self < other, pass self, other, __gt__
        try:
            other = _sympify(other)
        except SympifyError:
            return NotImplemented
        if other.is_Number:
            op = None
            s, o = self, other
            if other.is_NumberSymbol:
                op = getattr(o, attr)
            elif other.is_Float:
                op = getattr(o, attr)
            elif other.is_Rational:
                s, o = Integer(s.p*o.q), Integer(s.q*o.p)
                op = getattr(o, attr)
            if op:
                return op(s)
            if o.is_number and o.is_extended_real:
                return Integer(s.p), s.q*o
```
### 4 - sympy/core/numbers.py:

Start line: 1386, End line: 1425

```python
class Float(Number):

    def __abs__(self):
        return Float._new(mlib.mpf_abs(self._mpf_), self._prec)

    def __int__(self):
        if self._mpf_ == fzero:
            return 0
        return int(mlib.to_int(self._mpf_))  # uses round_fast = round_down

    def __eq__(self, other):
        from sympy.logic.boolalg import Boolean
        try:
            other = _sympify(other)
        except SympifyError:
            return NotImplemented
        if not self:
            return not other
        if isinstance(other, Boolean):
            return False
        if other.is_NumberSymbol:
            if other.is_irrational:
                return False
            return other.__eq__(self)
        if other.is_Float:
            # comparison is exact
            # so Float(.1, 3) != Float(.1, 33)
            return self._mpf_ == other._mpf_
        if other.is_Rational:
            return other.__eq__(self)
        if other.is_Number:
            # numbers should compare at the same precision;
            # all _as_mpf_val routines should be sure to abide
            # by the request to change the prec if necessary; if
            # they don't, the equality test will fail since it compares
            # the mpf tuples
            ompf = other._as_mpf_val(self._prec)
            return bool(mlib.mpf_eq(self._mpf_, ompf))
        return False    # Float != non-Number

    def __ne__(self, other):
        return not self == other
```
### 5 - sympy/core/numbers.py:

Start line: 1943, End line: 1976

```python
class Rational(Number):

    def __gt__(self, other):
        rv = self._Rrel(other, '__lt__')
        if rv is None:
            rv = self, other
        elif not type(rv) is tuple:
            return rv
        return Expr.__gt__(*rv)

    def __ge__(self, other):
        rv = self._Rrel(other, '__le__')
        if rv is None:
            rv = self, other
        elif not type(rv) is tuple:
            return rv
        return Expr.__ge__(*rv)

    def __lt__(self, other):
        rv = self._Rrel(other, '__gt__')
        if rv is None:
            rv = self, other
        elif not type(rv) is tuple:
            return rv
        return Expr.__lt__(*rv)

    def __le__(self, other):
        rv = self._Rrel(other, '__ge__')
        if rv is None:
            rv = self, other
        elif not type(rv) is tuple:
            return rv
        return Expr.__le__(*rv)

    def __hash__(self):
        return super().__hash__()
```
### 6 - sympy/core/numbers.py:

Start line: 1330, End line: 1352

```python
class Float(Number):

    @_sympifyit('other', NotImplemented)
    def __mod__(self, other):
        if isinstance(other, Rational) and other.q != 1 and global_parameters.evaluate:
            # calculate mod with Rationals, *then* round the result
            return Float(Rational.__mod__(Rational(self), other),
                         precision=self._prec)
        if isinstance(other, Float) and global_parameters.evaluate:
            r = self/other
            if r == int(r):
                return Float(0, precision=max(self._prec, other._prec))
        if isinstance(other, Number) and global_parameters.evaluate:
            rhs, prec = other._as_mpf_op(self._prec)
            return Float._new(mlib.mpf_mod(self._mpf_, rhs, prec, rnd), prec)
        return Number.__mod__(self, other)

    @_sympifyit('other', NotImplemented)
    def __rmod__(self, other):
        if isinstance(other, Float) and global_parameters.evaluate:
            return other.__mod__(self)
        if isinstance(other, Number) and global_parameters.evaluate:
            rhs, prec = other._as_mpf_op(self._prec)
            return Float._new(mlib.mpf_mod(rhs, self._mpf_, prec, rnd), prec)
        return Number.__rmod__(self, other)
```
### 7 - sympy/core/numbers.py:

Start line: 1510, End line: 1600

```python
class Rational(Number):
    """Represents rational numbers (p/q) of any size.

    Examples
    ========

    >>> from sympy import Rational, nsimplify, S, pi
    >>> Rational(1, 2)
    1/2

    Rational is unprejudiced in accepting input. If a float is passed, the
    underlying value of the binary representation will be returned:

    >>> Rational(.5)
    1/2
    >>> Rational(.2)
    3602879701896397/18014398509481984

    If the simpler representation of the float is desired then consider
    limiting the denominator to the desired value or convert the float to
    a string (which is roughly equivalent to limiting the denominator to
    10**12):

    >>> Rational(str(.2))
    1/5
    >>> Rational(.2).limit_denominator(10**12)
    1/5

    An arbitrarily precise Rational is obtained when a string literal is
    passed:

    >>> Rational("1.23")
    123/100
    >>> Rational('1e-2')
    1/100
    >>> Rational(".1")
    1/10
    >>> Rational('1e-2/3.2')
    1/320

    The conversion of other types of strings can be handled by
    the sympify() function, and conversion of floats to expressions
    or simple fractions can be handled with nsimplify:

    >>> S('.[3]')  # repeating digits in brackets
    1/3
    >>> S('3**2/10')  # general expressions
    9/10
    >>> nsimplify(.3)  # numbers that have a simple form
    3/10

    But if the input does not reduce to a literal Rational, an error will
    be raised:

    >>> Rational(pi)
    Traceback (most recent call last):
    ...
    TypeError: invalid input: pi


    Low-level
    ---------

    Access numerator and denominator as .p and .q:

    >>> r = Rational(3, 4)
    >>> r
    3/4
    >>> r.p
    3
    >>> r.q
    4

    Note that p and q return integers (not SymPy Integers) so some care
    is needed when using them in expressions:

    >>> r.p/r.q
    0.75

    See Also
    ========
    sympy.core.sympify.sympify, sympy.simplify.simplify.nsimplify
    """
    is_real = True
    is_integer = False
    is_rational = True
    is_number = True

    __slots__ = ('p', 'q')

    is_Rational = True
```
### 8 - sympy/core/numbers.py:

Start line: 1761, End line: 1775

```python
class Rational(Number):

    @_sympifyit('other', NotImplemented)
    def __truediv__(self, other):
        if global_parameters.evaluate:
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
                return Number.__truediv__(self, other)
        return Number.__truediv__(self, other)
```
### 9 - sympy/core/numbers.py:

Start line: 1776, End line: 1787

```python
class Rational(Number):
    @_sympifyit('other', NotImplemented)
    def __rtruediv__(self, other):
        if global_parameters.evaluate:
            if isinstance(other, Integer):
                return Rational(other.p*self.q, self.p, igcd(self.p, other.p))
            elif isinstance(other, Rational):
                return Rational(other.p*self.q, other.q*self.p, igcd(self.p, other.p)*igcd(self.q, other.q))
            elif isinstance(other, Float):
                return other*(1/self)
            else:
                return Number.__rtruediv__(self, other)
        return Number.__rtruediv__(self, other)
```
### 10 - sympy/core/numbers.py:

Start line: 1846, End line: 1871

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

    def floor(self):
        return Integer(self.p // self.q)

    def ceiling(self):
        return -Integer(-self.p // self.q)

    def __floor__(self):
        return self.floor()

    def __ceil__(self):
        return self.ceiling()
```
### 29 - sympy/sets/fancysets.py:

Start line: 231, End line: 267

```python
class Reals(Interval, metaclass=Singleton):
    """
    Represents all real numbers
    from negative infinity to positive infinity,
    including all integer, rational and irrational numbers.
    This set is also available as the Singleton, S.Reals.


    Examples
    ========

    >>> from sympy import S, Rational, pi, I
    >>> 5 in S.Reals
    True
    >>> Rational(-1, 2) in S.Reals
    True
    >>> pi in S.Reals
    True
    >>> 3*I in S.Reals
    False
    >>> S.Reals.contains(pi)
    True


    See Also
    ========

    ComplexRegion
    """
    def __new__(cls):
        return Interval.__new__(cls, S.NegativeInfinity, S.Infinity)

    def __eq__(self, other):
        return other == Interval(S.NegativeInfinity, S.Infinity)

    def __hash__(self):
        return hash(Interval(S.NegativeInfinity, S.Infinity))
```
