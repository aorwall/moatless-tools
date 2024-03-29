# sympy__sympy-24370

| **sympy/sympy** | `36a36f87dd3ac94593d8de186efd3532c77f5191` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 3266 |
| **Any found context length** | 3266 |
| **Avg pos** | 7.0 |
| **Min pos** | 7 |
| **Max pos** | 7 |
| **Top file pos** | 2 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sympy/core/numbers.py b/sympy/core/numbers.py
--- a/sympy/core/numbers.py
+++ b/sympy/core/numbers.py
@@ -2423,7 +2423,7 @@ def __floordiv__(self, other):
             return NotImplemented
         if isinstance(other, Integer):
             return Integer(self.p // other)
-        return Integer(divmod(self, other)[0])
+        return divmod(self, other)[0]
 
     def __rfloordiv__(self, other):
         return Integer(Integer(other).p // self.p)

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sympy/core/numbers.py | 2426 | 2426 | 7 | 2 | 3266


## Problem Statement

```
Floor division with sympy.Integer gives: Argument of Integer should be of numeric type, got floor(1024/s0)
\`\`\`
import sympy

s0 = sympy.Symbol('s0')
sympy.Integer(1024)//s0
\`\`\`

gives

\`\`\`
Traceback (most recent call last):
  File "/Users/ezyang/Dev/sympy/sympy/core/numbers.py", line 2098, in __new__
    ival = int(i)
  File "/Users/ezyang/Dev/sympy/sympy/core/expr.py", line 320, in __int__
    raise TypeError("Cannot convert symbols to int")
TypeError: Cannot convert symbols to int

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "repro.py", line 4, in <module>
    sympy.Integer(1024)//s0
  File "/Users/ezyang/Dev/sympy/sympy/core/decorators.py", line 65, in __sympifyit_wrapper
    return func(a, b)
  File "/Users/ezyang/Dev/sympy/sympy/core/numbers.py", line 2426, in __floordiv__
    return Integer(divmod(self, other)[0])
  File "/Users/ezyang/Dev/sympy/sympy/core/cache.py", line 72, in wrapper
    retval = cfunc(*args, **kwargs)
  File "/Users/ezyang/Dev/sympy/sympy/core/numbers.py", line 2100, in __new__
    raise TypeError(
TypeError: Argument of Integer should be of numeric type, got floor(1024/s0).
\`\`\`

oddly enough, it works if the lhs is a plain Python int.
Floor division with sympy.Integer gives: Argument of Integer should be of numeric type, got floor(1024/s0)
\`\`\`
import sympy

s0 = sympy.Symbol('s0')
sympy.Integer(1024)//s0
\`\`\`

gives

\`\`\`
Traceback (most recent call last):
  File "/Users/ezyang/Dev/sympy/sympy/core/numbers.py", line 2098, in __new__
    ival = int(i)
  File "/Users/ezyang/Dev/sympy/sympy/core/expr.py", line 320, in __int__
    raise TypeError("Cannot convert symbols to int")
TypeError: Cannot convert symbols to int

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "repro.py", line 4, in <module>
    sympy.Integer(1024)//s0
  File "/Users/ezyang/Dev/sympy/sympy/core/decorators.py", line 65, in __sympifyit_wrapper
    return func(a, b)
  File "/Users/ezyang/Dev/sympy/sympy/core/numbers.py", line 2426, in __floordiv__
    return Integer(divmod(self, other)[0])
  File "/Users/ezyang/Dev/sympy/sympy/core/cache.py", line 72, in wrapper
    retval = cfunc(*args, **kwargs)
  File "/Users/ezyang/Dev/sympy/sympy/core/numbers.py", line 2100, in __new__
    raise TypeError(
TypeError: Argument of Integer should be of numeric type, got floor(1024/s0).
\`\`\`

oddly enough, it works if the lhs is a plain Python int.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 sympy/functions/elementary/integers.py | 178 | 244| 533 | 533 | 4825 | 
| 2 | 1 sympy/functions/elementary/integers.py | 143 | 157| 186 | 719 | 4825 | 
| 3 | 1 sympy/functions/elementary/integers.py | 159 | 176| 227 | 946 | 4825 | 
| 4 | 1 sympy/functions/elementary/integers.py | 95 | 141| 305 | 1251 | 4825 | 
| 5 | 1 sympy/functions/elementary/integers.py | 336 | 402| 532 | 1783 | 4825 | 
| 6 | **2 sympy/core/numbers.py** | 2114 | 2221| 811 | 2594 | 39388 | 
| **-> 7 <-** | **2 sympy/core/numbers.py** | 2406 | 2499| 672 | 3266 | 39388 | 
| 8 | 2 sympy/functions/elementary/integers.py | 495 | 574| 574 | 3840 | 39388 | 
| 9 | **2 sympy/core/numbers.py** | 2223 | 2296| 492 | 4332 | 39388 | 
| 10 | **2 sympy/core/numbers.py** | 1301 | 1330| 371 | 4703 | 39388 | 
| 11 | **2 sympy/core/numbers.py** | 4438 | 4497| 445 | 5148 | 39388 | 
| 12 | **2 sympy/core/numbers.py** | 2087 | 2112| 246 | 5394 | 39388 | 
| 13 | **2 sympy/core/numbers.py** | 659 | 736| 563 | 5957 | 39388 | 
| 14 | 2 sympy/functions/elementary/integers.py | 247 | 299| 350 | 6307 | 39388 | 
| 15 | **2 sympy/core/numbers.py** | 1198 | 1299| 834 | 7141 | 39388 | 
| 16 | 3 sympy/core/expr.py | 247 | 303| 522 | 7663 | 74088 | 
| 17 | 3 sympy/functions/elementary/integers.py | 301 | 315| 186 | 7849 | 74088 | 
| 18 | **3 sympy/core/numbers.py** | 2374 | 2404| 357 | 8206 | 74088 | 
| 19 | 3 sympy/functions/elementary/integers.py | 405 | 458| 327 | 8533 | 74088 | 
| 20 | **3 sympy/core/numbers.py** | 1751 | 1765| 160 | 8693 | 74088 | 
| 21 | **3 sympy/core/numbers.py** | 2298 | 2373| 704 | 9397 | 74088 | 
| 22 | 3 sympy/functions/elementary/integers.py | 317 | 334| 227 | 9624 | 74088 | 
| 23 | 3 sympy/functions/elementary/integers.py | 594 | 626| 309 | 9933 | 74088 | 
| 24 | 4 sympy/core/benchmarks/bench_numbers.py | 1 | 92| 359 | 10292 | 74448 | 
| 25 | **4 sympy/core/numbers.py** | 625 | 657| 280 | 10572 | 74448 | 
| 26 | **4 sympy/core/numbers.py** | 1434 | 1480| 344 | 10916 | 74448 | 
| 27 | **4 sympy/core/numbers.py** | 178 | 202| 184 | 11100 | 74448 | 
| 28 | **4 sympy/core/numbers.py** | 3417 | 3448| 192 | 11292 | 74448 | 
| 29 | 5 sympy/core/evalf.py | 363 | 405| 408 | 11700 | 90828 | 
| 30 | **5 sympy/core/numbers.py** | 1766 | 1777| 138 | 11838 | 90828 | 
| 31 | **5 sympy/core/numbers.py** | 595 | 623| 252 | 12090 | 90828 | 
| 32 | **5 sympy/core/numbers.py** | 1402 | 1432| 307 | 12397 | 90828 | 
| 33 | **5 sympy/core/numbers.py** | 738 | 759| 155 | 12552 | 90828 | 
| 34 | 5 sympy/functions/elementary/integers.py | 459 | 493| 247 | 12799 | 90828 | 
| 35 | **5 sympy/core/numbers.py** | 1798 | 1841| 502 | 13301 | 90828 | 
| 36 | **5 sympy/core/numbers.py** | 1843 | 1868| 180 | 13481 | 90828 | 
| 37 | 5 sympy/functions/elementary/integers.py | 17 | 92| 536 | 14017 | 90828 | 
| 38 | **5 sympy/core/numbers.py** | 1779 | 1796| 186 | 14203 | 90828 | 
| 39 | **5 sympy/core/numbers.py** | 1941 | 1974| 237 | 14440 | 90828 | 
| 40 | **5 sympy/core/numbers.py** | 1588 | 1665| 513 | 14953 | 90828 | 
| 41 | **5 sympy/core/numbers.py** | 1988 | 2015| 180 | 15133 | 90828 | 
| 42 | **5 sympy/core/numbers.py** | 1917 | 1939| 184 | 15317 | 90828 | 
| 43 | 6 sympy/sets/handlers/mul.py | 51 | 80| 192 | 15509 | 91353 | 
| 44 | **6 sympy/core/numbers.py** | 1364 | 1400| 330 | 15839 | 91353 | 
| 45 | **6 sympy/core/numbers.py** | 1737 | 1749| 139 | 15978 | 91353 | 
| 46 | 6 sympy/functions/elementary/integers.py | 1 | 15| 142 | 16120 | 91353 | 
| 47 | **6 sympy/core/numbers.py** | 3087 | 3111| 205 | 16325 | 91353 | 
| 48 | 6 sympy/functions/elementary/integers.py | 576 | 592| 189 | 16514 | 91353 | 
| 49 | **6 sympy/core/numbers.py** | 1 | 37| 338 | 16852 | 91353 | 
| 50 | **6 sympy/core/numbers.py** | 1870 | 1915| 398 | 17250 | 91353 | 
| 51 | **6 sympy/core/numbers.py** | 3583 | 3615| 216 | 17466 | 91353 | 
| 52 | 7 sympy/core/mod.py | 222 | 238| 128 | 17594 | 93131 | 
| 53 | **7 sympy/core/numbers.py** | 1725 | 1736| 123 | 17717 | 93131 | 
| 54 | 8 sympy/interactive/session.py | 85 | 134| 401 | 18118 | 96791 | 
| 55 | **8 sympy/core/numbers.py** | 3941 | 4027| 648 | 18766 | 96791 | 
| 56 | 8 sympy/core/expr.py | 3889 | 3948| 698 | 19464 | 96791 | 
| 57 | **8 sympy/core/numbers.py** | 1711 | 1724| 129 | 19593 | 96791 | 
| 58 | **8 sympy/core/numbers.py** | 1683 | 1710| 214 | 19807 | 96791 | 
| 59 | **8 sympy/core/numbers.py** | 3206 | 3226| 190 | 19997 | 96791 | 
| 60 | 9 sympy/plotting/intervalmath/lib_interval.py | 329 | 347| 142 | 20139 | 100427 | 
| 61 | **9 sympy/core/numbers.py** | 1332 | 1362| 335 | 20474 | 100427 | 
| 62 | 10 sympy/solvers/diophantine/diophantine.py | 1 | 35| 385 | 20859 | 137095 | 
| 63 | 10 sympy/solvers/diophantine/diophantine.py | 1206 | 1271| 416 | 21275 | 137095 | 
| 64 | 11 sympy/polys/agca/extensions.py | 85 | 100| 176 | 21451 | 139446 | 
| 65 | 11 sympy/core/expr.py | 305 | 340| 504 | 21955 | 139446 | 
| 66 | 12 sympy/plotting/intervalmath/interval_arithmetic.py | 280 | 319| 342 | 22297 | 142613 | 
| 67 | 13 sympy/functions/special/error_functions.py | 2719 | 2742| 270 | 22567 | 164991 | 
| 68 | **13 sympy/core/numbers.py** | 1181 | 1196| 165 | 22732 | 164991 | 
| 69 | **13 sympy/core/numbers.py** | 1123 | 1179| 623 | 23355 | 164991 | 
| 70 | 14 sympy/polys/polyerrors.py | 30 | 55| 269 | 23624 | 166135 | 
| 71 | 15 sympy/functions/elementary/piecewise.py | 1 | 16| 179 | 23803 | 179497 | 
| 72 | **15 sympy/core/numbers.py** | 4029 | 4097| 421 | 24224 | 179497 | 
| 73 | 16 sympy/core/sympify.py | 103 | 352| 2286 | 26510 | 184665 | 


### Hint

```
The fix seems to be
\`\`\`diff
diff --git a/sympy/core/numbers.py b/sympy/core/numbers.py
index 3b1aec2..52f7ea4 100644
--- a/sympy/core/numbers.py
+++ b/sympy/core/numbers.py
@@ -2423,7 +2423,7 @@ def __floordiv__(self, other):
             return NotImplemented
         if isinstance(other, Integer):
             return Integer(self.p // other)
-        return Integer(divmod(self, other)[0])
+        return divmod(self, other)[0]
 
     def __rfloordiv__(self, other):
         return Integer(Integer(other).p // self.p)
\`\`\`
The fix seems to be
\`\`\`diff
diff --git a/sympy/core/numbers.py b/sympy/core/numbers.py
index 3b1aec2..52f7ea4 100644
--- a/sympy/core/numbers.py
+++ b/sympy/core/numbers.py
@@ -2423,7 +2423,7 @@ def __floordiv__(self, other):
             return NotImplemented
         if isinstance(other, Integer):
             return Integer(self.p // other)
-        return Integer(divmod(self, other)[0])
+        return divmod(self, other)[0]
 
     def __rfloordiv__(self, other):
         return Integer(Integer(other).p // self.p)
\`\`\`
```

## Patch

```diff
diff --git a/sympy/core/numbers.py b/sympy/core/numbers.py
--- a/sympy/core/numbers.py
+++ b/sympy/core/numbers.py
@@ -2423,7 +2423,7 @@ def __floordiv__(self, other):
             return NotImplemented
         if isinstance(other, Integer):
             return Integer(self.p // other)
-        return Integer(divmod(self, other)[0])
+        return divmod(self, other)[0]
 
     def __rfloordiv__(self, other):
         return Integer(Integer(other).p // self.p)

```

## Test Patch

```diff
diff --git a/sympy/core/tests/test_numbers.py b/sympy/core/tests/test_numbers.py
--- a/sympy/core/tests/test_numbers.py
+++ b/sympy/core/tests/test_numbers.py
@@ -16,6 +16,7 @@
 from sympy.core.symbol import Dummy, Symbol
 from sympy.core.sympify import sympify
 from sympy.functions.combinatorial.factorials import factorial
+from sympy.functions.elementary.integers import floor
 from sympy.functions.combinatorial.numbers import fibonacci
 from sympy.functions.elementary.exponential import exp, log
 from sympy.functions.elementary.miscellaneous import sqrt, cbrt
@@ -121,6 +122,7 @@ def test_mod():
 
 
 def test_divmod():
+    x = Symbol("x")
     assert divmod(S(12), S(8)) == Tuple(1, 4)
     assert divmod(-S(12), S(8)) == Tuple(-2, 4)
     assert divmod(S.Zero, S.One) == Tuple(0, 0)
@@ -128,6 +130,7 @@ def test_divmod():
     raises(ZeroDivisionError, lambda: divmod(S.One, S.Zero))
     assert divmod(S(12), 8) == Tuple(1, 4)
     assert divmod(12, S(8)) == Tuple(1, 4)
+    assert S(1024)//x == 1024//x == floor(1024/x)
 
     assert divmod(S("2"), S("3/2")) == Tuple(S("1"), S("1/2"))
     assert divmod(S("3/2"), S("2")) == Tuple(S("0"), S("3/2"))

```


## Code snippets

### 1 - sympy/functions/elementary/integers.py:

Start line: 178, End line: 244

```python
class floor(RoundFunction):

    def _eval_is_negative(self):
        return self.args[0].is_negative

    def _eval_is_nonnegative(self):
        return self.args[0].is_nonnegative

    def _eval_rewrite_as_ceiling(self, arg, **kwargs):
        return -ceiling(-arg)

    def _eval_rewrite_as_frac(self, arg, **kwargs):
        return arg - frac(arg)

    def __le__(self, other):
        other = S(other)
        if self.args[0].is_real:
            if other.is_integer:
                return self.args[0] < other + 1
            if other.is_number and other.is_real:
                return self.args[0] < ceiling(other)
        if self.args[0] == other and other.is_real:
            return S.true
        if other is S.Infinity and self.is_finite:
            return S.true

        return Le(self, other, evaluate=False)

    def __ge__(self, other):
        other = S(other)
        if self.args[0].is_real:
            if other.is_integer:
                return self.args[0] >= other
            if other.is_number and other.is_real:
                return self.args[0] >= ceiling(other)
        if self.args[0] == other and other.is_real:
            return S.false
        if other is S.NegativeInfinity and self.is_finite:
            return S.true

        return Ge(self, other, evaluate=False)

    def __gt__(self, other):
        other = S(other)
        if self.args[0].is_real:
            if other.is_integer:
                return self.args[0] >= other + 1
            if other.is_number and other.is_real:
                return self.args[0] >= ceiling(other)
        if self.args[0] == other and other.is_real:
            return S.false
        if other is S.NegativeInfinity and self.is_finite:
            return S.true

        return Gt(self, other, evaluate=False)

    def __lt__(self, other):
        other = S(other)
        if self.args[0].is_real:
            if other.is_integer:
                return self.args[0] < other
            if other.is_number and other.is_real:
                return self.args[0] < ceiling(other)
        if self.args[0] == other and other.is_real:
            return S.false
        if other is S.Infinity and self.is_finite:
            return S.true

        return Lt(self, other, evaluate=False)
```
### 2 - sympy/functions/elementary/integers.py:

Start line: 143, End line: 157

```python
class floor(RoundFunction):

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        from sympy.calculus.accumulationbounds import AccumBounds
        arg = self.args[0]
        arg0 = arg.subs(x, 0)
        r = self.subs(x, 0)
        if arg0 is S.NaN or isinstance(arg0, AccumBounds):
            arg0 = arg.limit(x, 0, dir='-' if re(cdir).is_negative else '+')
            r = floor(arg0)
        if arg0.is_finite:
            if arg0 == r:
                ndir = arg.dir(x, cdir=cdir)
                return r - 1 if ndir.is_negative else r
            else:
                return r
        return arg.as_leading_term(x, logx=logx, cdir=cdir)
```
### 3 - sympy/functions/elementary/integers.py:

Start line: 159, End line: 176

```python
class floor(RoundFunction):

    def _eval_nseries(self, x, n, logx, cdir=0):
        arg = self.args[0]
        arg0 = arg.subs(x, 0)
        r = self.subs(x, 0)
        if arg0 is S.NaN:
            arg0 = arg.limit(x, 0, dir='-' if re(cdir).is_negative else '+')
            r = floor(arg0)
        if arg0.is_infinite:
            from sympy.calculus.accumulationbounds import AccumBounds
            from sympy.series.order import Order
            s = arg._eval_nseries(x, n, logx, cdir)
            o = Order(1, (x, 0)) if n <= 0 else AccumBounds(-1, 0)
            return s + o
        if arg0 == r:
            ndir = arg.dir(x, cdir=cdir if cdir != 0 else 1)
            return r - 1 if ndir.is_negative else r
        else:
            return r
```
### 4 - sympy/functions/elementary/integers.py:

Start line: 95, End line: 141

```python
class floor(RoundFunction):
    """
    Floor is a univariate function which returns the largest integer
    value not greater than its argument. This implementation
    generalizes floor to complex numbers by taking the floor of the
    real and imaginary parts separately.

    Examples
    ========

    >>> from sympy import floor, E, I, S, Float, Rational
    >>> floor(17)
    17
    >>> floor(Rational(23, 10))
    2
    >>> floor(2*E)
    5
    >>> floor(-Float(0.567))
    -1
    >>> floor(-I/2)
    -I
    >>> floor(S(5)/2 + 5*I/2)
    2 + 2*I

    See Also
    ========

    sympy.functions.elementary.integers.ceiling

    References
    ==========

    .. [1] "Concrete mathematics" by Graham, pp. 87
    .. [2] http://mathworld.wolfram.com/FloorFunction.html

    """
    _dir = -1

    @classmethod
    def _eval_number(cls, arg):
        if arg.is_Number:
            return arg.floor()
        elif any(isinstance(i, j)
                for i in (arg, -arg) for j in (floor, ceiling)):
            return arg
        if arg.is_NumberSymbol:
            return arg.approximation_interval(Integer)[0]
```
### 5 - sympy/functions/elementary/integers.py:

Start line: 336, End line: 402

```python
class ceiling(RoundFunction):

    def _eval_rewrite_as_floor(self, arg, **kwargs):
        return -floor(-arg)

    def _eval_rewrite_as_frac(self, arg, **kwargs):
        return arg + frac(-arg)

    def _eval_is_positive(self):
        return self.args[0].is_positive

    def _eval_is_nonpositive(self):
        return self.args[0].is_nonpositive

    def __lt__(self, other):
        other = S(other)
        if self.args[0].is_real:
            if other.is_integer:
                return self.args[0] <= other - 1
            if other.is_number and other.is_real:
                return self.args[0] <= floor(other)
        if self.args[0] == other and other.is_real:
            return S.false
        if other is S.Infinity and self.is_finite:
            return S.true

        return Lt(self, other, evaluate=False)

    def __gt__(self, other):
        other = S(other)
        if self.args[0].is_real:
            if other.is_integer:
                return self.args[0] > other
            if other.is_number and other.is_real:
                return self.args[0] > floor(other)
        if self.args[0] == other and other.is_real:
            return S.false
        if other is S.NegativeInfinity and self.is_finite:
            return S.true

        return Gt(self, other, evaluate=False)

    def __ge__(self, other):
        other = S(other)
        if self.args[0].is_real:
            if other.is_integer:
                return self.args[0] > other - 1
            if other.is_number and other.is_real:
                return self.args[0] > floor(other)
        if self.args[0] == other and other.is_real:
            return S.true
        if other is S.NegativeInfinity and self.is_finite:
            return S.true

        return Ge(self, other, evaluate=False)

    def __le__(self, other):
        other = S(other)
        if self.args[0].is_real:
            if other.is_integer:
                return self.args[0] <= other
            if other.is_number and other.is_real:
                return self.args[0] <= floor(other)
        if self.args[0] == other and other.is_real:
            return S.false
        if other is S.Infinity and self.is_finite:
            return S.true

        return Le(self, other, evaluate=False)
```
### 6 - sympy/core/numbers.py:

Start line: 2114, End line: 2221

```python
class Integer(Rational):

    def __getnewargs__(self):
        return (self.p,)

    # Arithmetic operations are here for efficiency
    def __int__(self):
        return self.p

    def floor(self):
        return Integer(self.p)

    def ceiling(self):
        return Integer(self.p)

    def __floor__(self):
        return self.floor()

    def __ceil__(self):
        return self.ceiling()

    def __neg__(self):
        return Integer(-self.p)

    def __abs__(self):
        if self.p >= 0:
            return self
        else:
            return Integer(-self.p)

    def __divmod__(self, other):
        if isinstance(other, Integer) and global_parameters.evaluate:
            return Tuple(*(divmod(self.p, other.p)))
        else:
            return Number.__divmod__(self, other)

    def __rdivmod__(self, other):
        if isinstance(other, int) and global_parameters.evaluate:
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

    # TODO make it decorator + bytecodehacks?
    def __add__(self, other):
        if global_parameters.evaluate:
            if isinstance(other, int):
                return Integer(self.p + other)
            elif isinstance(other, Integer):
                return Integer(self.p + other.p)
            elif isinstance(other, Rational):
                return Rational(self.p*other.q + other.p, other.q, 1)
            return Rational.__add__(self, other)
        else:
            return Add(self, other)

    def __radd__(self, other):
        if global_parameters.evaluate:
            if isinstance(other, int):
                return Integer(other + self.p)
            elif isinstance(other, Rational):
                return Rational(other.p + self.p*other.q, other.q, 1)
            return Rational.__radd__(self, other)
        return Rational.__radd__(self, other)

    def __sub__(self, other):
        if global_parameters.evaluate:
            if isinstance(other, int):
                return Integer(self.p - other)
            elif isinstance(other, Integer):
                return Integer(self.p - other.p)
            elif isinstance(other, Rational):
                return Rational(self.p*other.q - other.p, other.q, 1)
            return Rational.__sub__(self, other)
        return Rational.__sub__(self, other)

    def __rsub__(self, other):
        if global_parameters.evaluate:
            if isinstance(other, int):
                return Integer(other - self.p)
            elif isinstance(other, Rational):
                return Rational(other.p - self.p*other.q, other.q, 1)
            return Rational.__rsub__(self, other)
        return Rational.__rsub__(self, other)

    def __mul__(self, other):
        if global_parameters.evaluate:
            if isinstance(other, int):
                return Integer(self.p*other)
            elif isinstance(other, Integer):
                return Integer(self.p*other.p)
            elif isinstance(other, Rational):
                return Rational(self.p*other.p, other.q, igcd(self.p, other.q))
            return Rational.__mul__(self, other)
        return Rational.__mul__(self, other)

    def __rmul__(self, other):
        if global_parameters.evaluate:
            if isinstance(other, int):
                return Integer(other*self.p)
            elif isinstance(other, Rational):
                return Rational(other.p*self.p, other.q, igcd(self.p, other.q))
            return Rational.__rmul__(self, other)
        return Rational.__rmul__(self, other)
```
### 7 - sympy/core/numbers.py:

Start line: 2406, End line: 2499

```python
class Integer(Rational):

    def _eval_is_prime(self):
        from sympy.ntheory.primetest import isprime

        return isprime(self)

    def _eval_is_composite(self):
        if self > 1:
            return fuzzy_not(self.is_prime)
        else:
            return False

    def as_numer_denom(self):
        return self, S.One

    @_sympifyit('other', NotImplemented)
    def __floordiv__(self, other):
        if not isinstance(other, Expr):
            return NotImplemented
        if isinstance(other, Integer):
            return Integer(self.p // other)
        return Integer(divmod(self, other)[0])

    def __rfloordiv__(self, other):
        return Integer(Integer(other).p // self.p)

    # These bitwise operations (__lshift__, __rlshift__, ..., __invert__) are defined
    # for Integer only and not for general SymPy expressions. This is to achieve
    # compatibility with the numbers.Integral ABC which only defines these operations
    # among instances of numbers.Integral. Therefore, these methods check explicitly for
    # integer types rather than using sympify because they should not accept arbitrary
    # symbolic expressions and there is no symbolic analogue of numbers.Integral's
    # bitwise operations.
    def __lshift__(self, other):
        if isinstance(other, (int, Integer, numbers.Integral)):
            return Integer(self.p << int(other))
        else:
            return NotImplemented

    def __rlshift__(self, other):
        if isinstance(other, (int, numbers.Integral)):
            return Integer(int(other) << self.p)
        else:
            return NotImplemented

    def __rshift__(self, other):
        if isinstance(other, (int, Integer, numbers.Integral)):
            return Integer(self.p >> int(other))
        else:
            return NotImplemented

    def __rrshift__(self, other):
        if isinstance(other, (int, numbers.Integral)):
            return Integer(int(other) >> self.p)
        else:
            return NotImplemented

    def __and__(self, other):
        if isinstance(other, (int, Integer, numbers.Integral)):
            return Integer(self.p & int(other))
        else:
            return NotImplemented

    def __rand__(self, other):
        if isinstance(other, (int, numbers.Integral)):
            return Integer(int(other) & self.p)
        else:
            return NotImplemented

    def __xor__(self, other):
        if isinstance(other, (int, Integer, numbers.Integral)):
            return Integer(self.p ^ int(other))
        else:
            return NotImplemented

    def __rxor__(self, other):
        if isinstance(other, (int, numbers.Integral)):
            return Integer(int(other) ^ self.p)
        else:
            return NotImplemented

    def __or__(self, other):
        if isinstance(other, (int, Integer, numbers.Integral)):
            return Integer(self.p | int(other))
        else:
            return NotImplemented

    def __ror__(self, other):
        if isinstance(other, (int, numbers.Integral)):
            return Integer(int(other) | self.p)
        else:
            return NotImplemented

    def __invert__(self):
        return Integer(~self.p)
```
### 8 - sympy/functions/elementary/integers.py:

Start line: 495, End line: 574

```python
class frac(Function):

    def _eval_rewrite_as_floor(self, arg, **kwargs):
        return arg - floor(arg)

    def _eval_rewrite_as_ceiling(self, arg, **kwargs):
        return arg + ceiling(-arg)

    def _eval_is_finite(self):
        return True

    def _eval_is_real(self):
        return self.args[0].is_extended_real

    def _eval_is_imaginary(self):
        return self.args[0].is_imaginary

    def _eval_is_integer(self):
        return self.args[0].is_integer

    def _eval_is_zero(self):
        return fuzzy_or([self.args[0].is_zero, self.args[0].is_integer])

    def _eval_is_negative(self):
        return False

    def __ge__(self, other):
        if self.is_extended_real:
            other = _sympify(other)
            # Check if other <= 0
            if other.is_extended_nonpositive:
                return S.true
            # Check if other >= 1
            res = self._value_one_or_more(other)
            if res is not None:
                return not(res)
        return Ge(self, other, evaluate=False)

    def __gt__(self, other):
        if self.is_extended_real:
            other = _sympify(other)
            # Check if other < 0
            res = self._value_one_or_more(other)
            if res is not None:
                return not(res)
            # Check if other >= 1
            if other.is_extended_negative:
                return S.true
        return Gt(self, other, evaluate=False)

    def __le__(self, other):
        if self.is_extended_real:
            other = _sympify(other)
            # Check if other < 0
            if other.is_extended_negative:
                return S.false
            # Check if other >= 1
            res = self._value_one_or_more(other)
            if res is not None:
                return res
        return Le(self, other, evaluate=False)

    def __lt__(self, other):
        if self.is_extended_real:
            other = _sympify(other)
            # Check if other <= 0
            if other.is_extended_nonpositive:
                return S.false
            # Check if other >= 1
            res = self._value_one_or_more(other)
            if res is not None:
                return res
        return Lt(self, other, evaluate=False)

    def _value_one_or_more(self, other):
        if other.is_extended_real:
            if other.is_number:
                res = other >= 1
                if res and not isinstance(res, Relational):
                    return S.true
            if other.is_integer and other.is_positive:
                return S.true
```
### 9 - sympy/core/numbers.py:

Start line: 2223, End line: 2296

```python
class Integer(Rational):

    def __mod__(self, other):
        if global_parameters.evaluate:
            if isinstance(other, int):
                return Integer(self.p % other)
            elif isinstance(other, Integer):
                return Integer(self.p % other.p)
            return Rational.__mod__(self, other)
        return Rational.__mod__(self, other)

    def __rmod__(self, other):
        if global_parameters.evaluate:
            if isinstance(other, int):
                return Integer(other % self.p)
            elif isinstance(other, Integer):
                return Integer(other.p % self.p)
            return Rational.__rmod__(self, other)
        return Rational.__rmod__(self, other)

    def __eq__(self, other):
        if isinstance(other, int):
            return (self.p == other)
        elif isinstance(other, Integer):
            return (self.p == other.p)
        return Rational.__eq__(self, other)

    def __ne__(self, other):
        return not self == other

    def __gt__(self, other):
        try:
            other = _sympify(other)
        except SympifyError:
            return NotImplemented
        if other.is_Integer:
            return _sympify(self.p > other.p)
        return Rational.__gt__(self, other)

    def __lt__(self, other):
        try:
            other = _sympify(other)
        except SympifyError:
            return NotImplemented
        if other.is_Integer:
            return _sympify(self.p < other.p)
        return Rational.__lt__(self, other)

    def __ge__(self, other):
        try:
            other = _sympify(other)
        except SympifyError:
            return NotImplemented
        if other.is_Integer:
            return _sympify(self.p >= other.p)
        return Rational.__ge__(self, other)

    def __le__(self, other):
        try:
            other = _sympify(other)
        except SympifyError:
            return NotImplemented
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
### 10 - sympy/core/numbers.py:

Start line: 1301, End line: 1330

```python
class Float(Number):

    @_sympifyit('other', NotImplemented)
    def __truediv__(self, other):
        if isinstance(other, Number) and other != 0 and global_parameters.evaluate:
            rhs, prec = other._as_mpf_op(self._prec)
            return Float._new(mlib.mpf_div(self._mpf_, rhs, prec, rnd), prec)
        return Number.__truediv__(self, other)

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
### 11 - sympy/core/numbers.py:

Start line: 4438, End line: 4497

```python
I = S.ImaginaryUnit

@dispatch(Tuple, Number) # type:ignore
def _eval_is_eq(self, other): # noqa: F811
    return False

def sympify_fractions(f):
    return Rational(f.numerator, f.denominator, 1)

_sympy_converter[fractions.Fraction] = sympify_fractions

if HAS_GMPY:
    def sympify_mpz(x):
        return Integer(int(x))

    # XXX: The sympify_mpq function here was never used because it is
    # overridden by the other sympify_mpq function below. Maybe it should just
    # be removed or maybe it should be used for something...
    def sympify_mpq(x):
        return Rational(int(x.numerator), int(x.denominator))

    _sympy_converter[type(gmpy.mpz(1))] = sympify_mpz
    _sympy_converter[type(gmpy.mpq(1, 2))] = sympify_mpq


def sympify_mpmath_mpq(x):
    p, q = x._mpq_
    return Rational(p, q, 1)

_sympy_converter[type(mpmath.rational.mpq(1, 2))] = sympify_mpmath_mpq


def sympify_mpmath(x):
    return Expr._from_mpmath(x, x.context.prec)

_sympy_converter[mpnumeric] = sympify_mpmath


def sympify_complex(a):
    real, imag = list(map(sympify, (a.real, a.imag)))
    return real + S.ImaginaryUnit*imag

_sympy_converter[complex] = sympify_complex

from .power import Pow, integer_nthroot
from .mul import Mul
Mul.identity = One()
from .add import Add
Add.identity = Zero()

def _register_classes():
    numbers.Number.register(Number)
    numbers.Real.register(Float)
    numbers.Rational.register(Rational)
    numbers.Integral.register(Integer)

_register_classes()

_illegal = (S.NaN, S.Infinity, S.NegativeInfinity, S.ComplexInfinity)
```
### 12 - sympy/core/numbers.py:

Start line: 2087, End line: 2112

```python
class Integer(Rational):

    @cacheit
    def __new__(cls, i):
        if isinstance(i, str):
            i = i.replace(' ', '')
        # whereas we cannot, in general, make a Rational from an
        # arbitrary expression, we can make an Integer unambiguously
        # (except when a non-integer expression happens to round to
        # an integer). So we proceed by taking int() of the input and
        # let the int routines determine whether the expression can
        # be made into an int or whether an error should be raised.
        try:
            ival = int(i)
        except TypeError:
            raise TypeError(
                "Argument of Integer should be of numeric type, got %s." % i)
        # We only work with well-behaved integer types. This converts, for
        # example, numpy.int32 instances.
        if ival == 1:
            return S.One
        if ival == -1:
            return S.NegativeOne
        if ival == 0:
            return S.Zero
        obj = Expr.__new__(cls)
        obj.p = ival
        return obj
```
### 13 - sympy/core/numbers.py:

Start line: 659, End line: 736

```python
class Number(AtomicExpr):

    def __rdivmod__(self, other):
        try:
            other = Number(other)
        except TypeError:
            return NotImplemented
        return divmod(other, self)

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

    def __floor__(self):
        return self.floor()

    def __ceil__(self):
        return self.ceiling()

    def _eval_conjugate(self):
        return self

    def _eval_order(self, *symbols):
        from sympy.series.order import Order
        # Order(5, x, y) -> Order(1,x,y)
        return Order(S.One, *symbols)

    def _eval_subs(self, old, new):
        if old == -self:
            return -new
        return self  # there is no other possibility

    @classmethod
    def class_key(cls):
        return 1, 0, 'Number'

    @cacheit
    def sort_key(self, order=None):
        return self.class_key(), (0, ()), (), self

    @_sympifyit('other', NotImplemented)
    def __add__(self, other):
        if isinstance(other, Number) and global_parameters.evaluate:
            if other is S.NaN:
                return S.NaN
            elif other is S.Infinity:
                return S.Infinity
            elif other is S.NegativeInfinity:
                return S.NegativeInfinity
        return AtomicExpr.__add__(self, other)

    @_sympifyit('other', NotImplemented)
    def __sub__(self, other):
        if isinstance(other, Number) and global_parameters.evaluate:
            if other is S.NaN:
                return S.NaN
            elif other is S.Infinity:
                return S.NegativeInfinity
            elif other is S.NegativeInfinity:
                return S.Infinity
        return AtomicExpr.__sub__(self, other)
```
### 15 - sympy/core/numbers.py:

Start line: 1198, End line: 1299

```python
class Float(Number):

    # mpz can't be pickled
    def __getnewargs_ex__(self):
        return ((mlib.to_pickable(self._mpf_),), {'precision': self._prec})

    def _hashable_content(self):
        return (self._mpf_, self._prec)

    def floor(self):
        return Integer(int(mlib.to_int(
            mlib.mpf_floor(self._mpf_, self._prec))))

    def ceiling(self):
        return Integer(int(mlib.to_int(
            mlib.mpf_ceil(self._mpf_, self._prec))))

    def __floor__(self):
        return self.floor()

    def __ceil__(self):
        return self.ceiling()

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
        return self._mpf_ == fzero

    def _eval_is_negative(self):
        if self._mpf_ in (_mpf_ninf, _mpf_inf):
            return False
        return self.num < 0

    def _eval_is_positive(self):
        if self._mpf_ in (_mpf_ninf, _mpf_inf):
            return False
        return self.num > 0

    def _eval_is_extended_negative(self):
        if self._mpf_ == _mpf_ninf:
            return True
        if self._mpf_ == _mpf_inf:
            return False
        return self.num < 0

    def _eval_is_extended_positive(self):
        if self._mpf_ == _mpf_inf:
            return True
        if self._mpf_ == _mpf_ninf:
            return False
        return self.num > 0

    def _eval_is_zero(self):
        return self._mpf_ == fzero

    def __bool__(self):
        return self._mpf_ != fzero

    def __neg__(self):
        if not self:
            return self
        return Float._new(mlib.mpf_neg(self._mpf_), self._prec)

    @_sympifyit('other', NotImplemented)
    def __add__(self, other):
        if isinstance(other, Number) and global_parameters.evaluate:
            rhs, prec = other._as_mpf_op(self._prec)
            return Float._new(mlib.mpf_add(self._mpf_, rhs, prec, rnd), prec)
        return Number.__add__(self, other)

    @_sympifyit('other', NotImplemented)
    def __sub__(self, other):
        if isinstance(other, Number) and global_parameters.evaluate:
            rhs, prec = other._as_mpf_op(self._prec)
            return Float._new(mlib.mpf_sub(self._mpf_, rhs, prec, rnd), prec)
        return Number.__sub__(self, other)

    @_sympifyit('other', NotImplemented)
    def __mul__(self, other):
        if isinstance(other, Number) and global_parameters.evaluate:
            rhs, prec = other._as_mpf_op(self._prec)
            return Float._new(mlib.mpf_mul(self._mpf_, rhs, prec, rnd), prec)
        return Number.__mul__(self, other)
```
### 18 - sympy/core/numbers.py:

Start line: 2374, End line: 2404

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
                    out_rad *= Pow(prime, Rational(div_m//g, expt.q//g, 1))
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
```
### 20 - sympy/core/numbers.py:

Start line: 1751, End line: 1765

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
### 21 - sympy/core/numbers.py:

Start line: 2298, End line: 2373

```python
class Integer(Rational):

    def _eval_power(self, expt):
        """
        Tries to do some simplifications on self**expt

        Returns None if no further simplifications can be done.

        Explanation
        ===========

        When exponent is a fraction (so we have for example a square root),
        we try to find a simpler representation by factoring the argument
        up to factors of 2**15, e.g.

          - sqrt(4) becomes 2
          - sqrt(-4) becomes 2*I
          - (2**(3+7)*3**(6+7))**Rational(1,7) becomes 6*18**(3/7)

        Further simplification would require a special call to factorint on
        the argument which is not done here for sake of speed.

        """
        from sympy.ntheory.factor_ import perfect_power

        if expt is S.Infinity:
            if self.p > S.One:
                return S.Infinity
            # cases -1, 0, 1 are done in their respective classes
            return S.Infinity + S.ImaginaryUnit*S.Infinity
        if expt is S.NegativeInfinity:
            return Rational(1, self, 1)**S.Infinity
        if not isinstance(expt, Number):
            # simplify when expt is even
            # (-2)**k --> 2**k
            if self.is_negative and expt.is_even:
                return (-self)**expt
        if isinstance(expt, Float):
            # Rational knows how to exponentiate by a Float
            return super()._eval_power(expt)
        if not isinstance(expt, Rational):
            return
        if expt is S.Half and self.is_negative:
            # we extract I for this special case since everyone is doing so
            return S.ImaginaryUnit*Pow(-self, expt)
        if expt.is_negative:
            # invert base and change sign on exponent
            ne = -expt
            if self.is_negative:
                return S.NegativeOne**expt*Rational(1, -self, 1)**ne
            else:
                return Rational(1, self.p, 1)**ne
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
### 25 - sympy/core/numbers.py:

Start line: 625, End line: 657

```python
class Number(AtomicExpr):

    def could_extract_minus_sign(self):
        return bool(self.is_extended_negative)

    def invert(self, other, *gens, **args):
        from sympy.polys.polytools import invert
        if getattr(other, 'is_number', True):
            return mod_inverse(self, other)
        return invert(self, other, *gens, **args)

    def __divmod__(self, other):
        from sympy.functions.elementary.complexes import sign

        try:
            other = Number(other)
            if self.is_infinite or S.NaN in (self, other):
                return (S.NaN, S.NaN)
        except TypeError:
            return NotImplemented
        if not other:
            raise ZeroDivisionError('modulo by zero')
        if self.is_Integer and other.is_Integer:
            return Tuple(*divmod(self.p, other.p))
        elif isinstance(other, Float):
            rat = self/Rational(other)
        else:
            rat = self/other
        if other.is_finite:
            w = int(rat) if rat >= 0 else int(rat) - 1
            r = self - other*w
        else:
            w = 0 if not self or (sign(self) == sign(other)) else -1
            r = other if w else self
        return Tuple(w, r)
```
### 26 - sympy/core/numbers.py:

Start line: 1434, End line: 1480

```python
class Float(Number):

    def __gt__(self, other):
        if isinstance(other, NumberSymbol):
            return other.__lt__(self)
        rv = self._Frel(other, mlib.mpf_gt)
        if rv is None:
            return Expr.__gt__(self, other)
        return rv

    def __ge__(self, other):
        if isinstance(other, NumberSymbol):
            return other.__le__(self)
        rv = self._Frel(other, mlib.mpf_ge)
        if rv is None:
            return Expr.__ge__(self, other)
        return rv

    def __lt__(self, other):
        if isinstance(other, NumberSymbol):
            return other.__gt__(self)
        rv = self._Frel(other, mlib.mpf_lt)
        if rv is None:
            return Expr.__lt__(self, other)
        return rv

    def __le__(self, other):
        if isinstance(other, NumberSymbol):
            return other.__ge__(self)
        rv = self._Frel(other, mlib.mpf_le)
        if rv is None:
            return Expr.__le__(self, other)
        return rv

    def __hash__(self):
        return super().__hash__()

    def epsilon_eq(self, other, epsilon="1e-15"):
        return abs(self - other) < Float(epsilon)

    def __format__(self, format_spec):
        return format(decimal.Decimal(str(self)), format_spec)


# Add sympify converters
_sympy_converter[float] = _sympy_converter[decimal.Decimal] = Float

# this is here to work nicely in Sage
RealNumber = Float
```
### 27 - sympy/core/numbers.py:

Start line: 178, End line: 202

```python
# TODO: we should use the warnings module
_errdict = {"divide": False}


def seterr(divide=False):
    """
    Should SymPy raise an exception on 0/0 or return a nan?

    divide == True .... raise an exception
    divide == False ... return nan
    """
    if _errdict["divide"] != divide:
        clear_cache()
        _errdict["divide"] = divide


def _as_integer_ratio(p):
    neg_pow, man, expt, _ = getattr(p, '_mpf_', mpmath.mpf(p)._mpf_)
    p = [1, -1][neg_pow % 2]*man
    if expt < 0:
        q = 2**-expt
    else:
        q = 1
        p *= 2**expt
    return int(p), int(q)
```
### 28 - sympy/core/numbers.py:

Start line: 3417, End line: 3448

```python
class Infinity(Number, metaclass=Singleton):

    def _as_mpf_val(self, prec):
        return mlib.finf

    def __hash__(self):
        return super().__hash__()

    def __eq__(self, other):
        return other is S.Infinity or other == float('inf')

    def __ne__(self, other):
        return other is not S.Infinity and other != float('inf')

    __gt__ = Expr.__gt__
    __ge__ = Expr.__ge__
    __lt__ = Expr.__lt__
    __le__ = Expr.__le__

    @_sympifyit('other', NotImplemented)
    def __mod__(self, other):
        if not isinstance(other, Expr):
            return NotImplemented
        return S.NaN

    __rmod__ = __mod__

    def floor(self):
        return self

    def ceiling(self):
        return self

oo = S.Infinity
```
### 30 - sympy/core/numbers.py:

Start line: 1766, End line: 1777

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
### 31 - sympy/core/numbers.py:

Start line: 595, End line: 623

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
        if isinstance(obj, str):
            _obj = obj.lower()  # float('INF') == float('inf')
            if _obj == 'nan':
                return S.NaN
            elif _obj == 'inf':
                return S.Infinity
            elif _obj == '+inf':
                return S.Infinity
            elif _obj == '-inf':
                return S.NegativeInfinity
            val = sympify(obj)
            if isinstance(val, Number):
                return val
            else:
                raise ValueError('String "%s" does not denote a Number' % obj)
        msg = "expected str|int|long|float|Decimal|Number object but got %r"
        raise TypeError(msg % type(obj).__name__)
```
### 32 - sympy/core/numbers.py:

Start line: 1402, End line: 1432

```python
class Float(Number):

    def __ne__(self, other):
        return not self == other

    def _Frel(self, other, op):
        try:
            other = _sympify(other)
        except SympifyError:
            return NotImplemented
        if other.is_Rational:
            # test self*other.q <?> other.p without losing precision
            '''
            >>> f = Float(.1,2)
            >>> i = 1234567890
            >>> (f*i)._mpf_
            (0, 471, 18, 9)
            >>> mlib.mpf_mul(f._mpf_, mlib.from_int(i))
            (0, 505555550955, -12, 39)
            '''
            smpf = mlib.mpf_mul(self._mpf_, mlib.from_int(other.q))
            ompf = mlib.from_int(other.p)
            return _sympify(bool(op(smpf, ompf)))
        elif other.is_Float:
            return _sympify(bool(
                        op(self._mpf_, other._mpf_)))
        elif other.is_comparable and other not in (
                S.Infinity, S.NegativeInfinity):
            other = other.evalf(prec_to_dps(self._prec))
            if other._prec > 1:
                if other.is_Number:
                    return _sympify(bool(
                        op(self._mpf_, other._as_mpf_val(self._prec))))
```
### 33 - sympy/core/numbers.py:

Start line: 738, End line: 759

```python
class Number(AtomicExpr):

    @_sympifyit('other', NotImplemented)
    def __mul__(self, other):
        if isinstance(other, Number) and global_parameters.evaluate:
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
### 35 - sympy/core/numbers.py:

Start line: 1798, End line: 1841

```python
class Rational(Number):

    def _eval_power(self, expt):
        if isinstance(expt, Number):
            if isinstance(expt, Float):
                return self._eval_evalf(expt._prec)**expt
            if expt.is_extended_negative:
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
                intpart = expt.p // expt.q
                if intpart:
                    intpart += 1
                    remfracpart = intpart*expt.q - expt.p
                    ratfracpart = Rational(remfracpart, expt.q)
                    if self.p != 1:
                        return Integer(self.p)**expt*Integer(self.q)**ratfracpart*Rational(1, self.q**intpart, 1)
                    return Integer(self.q)**ratfracpart*Rational(1, self.q**intpart, 1)
                else:
                    remfracpart = expt.q - expt.p
                    ratfracpart = Rational(remfracpart, expt.q)
                    if self.p != 1:
                        return Integer(self.p)**expt*Integer(self.q)**ratfracpart*Rational(1, self.q, 1)
                    return Integer(self.q)**ratfracpart*Rational(1, self.q, 1)

        if self.is_extended_negative and expt.is_even:
            return (-self)**expt

        return
```
### 36 - sympy/core/numbers.py:

Start line: 1843, End line: 1868

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
### 38 - sympy/core/numbers.py:

Start line: 1779, End line: 1796

```python
class Rational(Number):

    @_sympifyit('other', NotImplemented)
    def __mod__(self, other):
        if global_parameters.evaluate:
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
### 39 - sympy/core/numbers.py:

Start line: 1941, End line: 1974

```python
class Rational(Number):

    def __gt__(self, other):
        rv = self._Rrel(other, '__lt__')
        if rv is None:
            rv = self, other
        elif not isinstance(rv, tuple):
            return rv
        return Expr.__gt__(*rv)

    def __ge__(self, other):
        rv = self._Rrel(other, '__le__')
        if rv is None:
            rv = self, other
        elif not isinstance(rv, tuple):
            return rv
        return Expr.__ge__(*rv)

    def __lt__(self, other):
        rv = self._Rrel(other, '__gt__')
        if rv is None:
            rv = self, other
        elif not isinstance(rv, tuple):
            return rv
        return Expr.__lt__(*rv)

    def __le__(self, other):
        rv = self._Rrel(other, '__ge__')
        if rv is None:
            rv = self, other
        elif not isinstance(rv, tuple):
            return rv
        return Expr.__le__(*rv)

    def __hash__(self):
        return super().__hash__()
```
### 40 - sympy/core/numbers.py:

Start line: 1588, End line: 1665

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

                if not isinstance(p, str):
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

        if not isinstance(p, SYMPY_INTS):
            p = Rational(p)
            q *= p.q
            p = p.p
        else:
            p = int(p)

        if not isinstance(q, SYMPY_INTS):
            q = Rational(q)
            p *= q.q
            q = q.p
        else:
            q = int(q)

        # p and q are now ints
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
### 41 - sympy/core/numbers.py:

Start line: 1988, End line: 2015

```python
class Rational(Number):

    @property
    def numerator(self):
        return self.p

    @property
    def denominator(self):
        return self.q

    @_sympifyit('other', NotImplemented)
    def gcd(self, other):
        if isinstance(other, Rational):
            if other == S.Zero:
                return other
            return Rational(
                igcd(self.p, other.p),
                ilcm(self.q, other.q))
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
```
### 42 - sympy/core/numbers.py:

Start line: 1917, End line: 1939

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
### 44 - sympy/core/numbers.py:

Start line: 1364, End line: 1400

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
        if not self:
            return not other
        return False    # Float != non-Number
```
### 45 - sympy/core/numbers.py:

Start line: 1737, End line: 1749

```python
class Rational(Number):
    @_sympifyit('other', NotImplemented)
    def __mul__(self, other):
        if global_parameters.evaluate:
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
### 47 - sympy/core/numbers.py:

Start line: 3087, End line: 3111

```python
class Zero(IntegerConstant, metaclass=Singleton):

    def _eval_power(self, expt):
        if expt.is_extended_positive:
            return self
        if expt.is_extended_negative:
            return S.ComplexInfinity
        if expt.is_extended_real is False:
            return S.NaN
        if expt.is_zero:
            return S.One

        # infinities are already handled with pos and neg
        # tests above; now throw away leading numbers on Mul
        # exponent since 0**-x = zoo**x even when x == 0
        coeff, terms = expt.as_coeff_Mul()
        if coeff.is_negative:
            return S.ComplexInfinity**terms
        if coeff is not S.One:  # there is a Number to discard
            return self**terms

    def _eval_order(self, *symbols):
        # Order(0,x) -> 0
        return self

    def __bool__(self):
        return False
```
### 49 - sympy/core/numbers.py:

Start line: 1, End line: 37

```python
from __future__ import annotations

import numbers
import decimal
import fractions
import math
import re as regex
import sys
from functools import lru_cache

from .containers import Tuple
from .sympify import (SympifyError, _sympy_converter, sympify, _convert_numpy_types,
              _sympify, _is_numpy_instance)
from .singleton import S, Singleton
from .basic import Basic
from .expr import Expr, AtomicExpr
from .evalf import pure_complex
from .cache import cacheit, clear_cache
from .decorators import _sympifyit
from .logic import fuzzy_not
from .kind import NumberKind
from sympy.external.gmpy import SYMPY_INTS, HAS_GMPY, gmpy
from sympy.multipledispatch import dispatch
import mpmath
import mpmath.libmp as mlib
from mpmath.libmp import bitcount, round_nearest as rnd
from mpmath.libmp.backend import MPZ
from mpmath.libmp import mpf_pow, mpf_pi, mpf_e, phi_fixed
from mpmath.ctx_mp import mpnumeric
from mpmath.libmp.libmpf import (
    finf as _mpf_inf, fninf as _mpf_ninf,
    fnan as _mpf_nan, fzero, _normalize as mpf_normalize,
    prec_to_dps, dps_to_prec)
from sympy.utilities.misc import as_int, debug, filldedent
from .parameters import global_parameters

_LOG2 = math.log(2)
```
### 50 - sympy/core/numbers.py:

Start line: 1870, End line: 1915

```python
class Rational(Number):

    def __eq__(self, other):
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

            from .power import integer_log
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
### 51 - sympy/core/numbers.py:

Start line: 3583, End line: 3615

```python
class NegativeInfinity(Number, metaclass=Singleton):

    def _as_mpf_val(self, prec):
        return mlib.fninf

    def __hash__(self):
        return super().__hash__()

    def __eq__(self, other):
        return other is S.NegativeInfinity or other == float('-inf')

    def __ne__(self, other):
        return other is not S.NegativeInfinity and other != float('-inf')

    __gt__ = Expr.__gt__
    __ge__ = Expr.__ge__
    __lt__ = Expr.__lt__
    __le__ = Expr.__le__

    @_sympifyit('other', NotImplemented)
    def __mod__(self, other):
        if not isinstance(other, Expr):
            return NotImplemented
        return S.NaN

    __rmod__ = __mod__

    def floor(self):
        return self

    def ceiling(self):
        return self

    def as_powers_dict(self):
        return {S.NegativeOne: 1, S.Infinity: 1}
```
### 53 - sympy/core/numbers.py:

Start line: 1725, End line: 1736

```python
class Rational(Number):
    @_sympifyit('other', NotImplemented)
    def __rsub__(self, other):
        if global_parameters.evaluate:
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
### 55 - sympy/core/numbers.py:

Start line: 3941, End line: 4027

```python
class Exp1(NumberSymbol, metaclass=Singleton):

    def _eval_power_exp_is_pow(self, arg):
        if arg.is_Number:
            if arg is oo:
                return oo
            elif arg == -oo:
                return S.Zero
        from sympy.functions.elementary.exponential import log
        if isinstance(arg, log):
            return arg.args[0]

        # don't autoexpand Pow or Mul (see the issue 3351):
        elif not arg.is_Add:
            Ioo = I*oo
            if arg in [Ioo, -Ioo]:
                return nan

            coeff = arg.coeff(pi*I)
            if coeff:
                if (2*coeff).is_integer:
                    if coeff.is_even:
                        return S.One
                    elif coeff.is_odd:
                        return S.NegativeOne
                    elif (coeff + S.Half).is_even:
                        return -I
                    elif (coeff + S.Half).is_odd:
                        return I
                elif coeff.is_Rational:
                    ncoeff = coeff % 2 # restrict to [0, 2pi)
                    if ncoeff > 1: # restrict to (-pi, pi]
                        ncoeff -= 2
                    if ncoeff != coeff:
                        return S.Exp1**(ncoeff*S.Pi*S.ImaginaryUnit)

            # Warning: code in risch.py will be very sensitive to changes
            # in this (see DifferentialExtension).

            # look for a single log factor

            coeff, terms = arg.as_coeff_Mul()

            # but it can't be multiplied by oo
            if coeff in (oo, -oo):
                return

            coeffs, log_term = [coeff], None
            for term in Mul.make_args(terms):
                if isinstance(term, log):
                    if log_term is None:
                        log_term = term.args[0]
                    else:
                        return
                elif term.is_comparable:
                    coeffs.append(term)
                else:
                    return

            return log_term**Mul(*coeffs) if log_term else None
        elif arg.is_Add:
            out = []
            add = []
            argchanged = False
            for a in arg.args:
                if a is S.One:
                    add.append(a)
                    continue
                newa = self**a
                if isinstance(newa, Pow) and newa.base is self:
                    if newa.exp != a:
                        add.append(newa.exp)
                        argchanged = True
                    else:
                        add.append(a)
                else:
                    out.append(newa)
            if out or argchanged:
                return Mul(*out)*Pow(self, Add(*add), evaluate=False)
        elif arg.is_Matrix:
            return arg.exp()

    def _eval_rewrite_as_sin(self, **kwargs):
        from sympy.functions.elementary.trigonometric import sin
        return sin(I + S.Pi/2) - I*sin(I)

    def _eval_rewrite_as_cos(self, **kwargs):
        from sympy.functions.elementary.trigonometric import cos
        return cos(I) + I*cos(I + S.Pi/2)
```
### 57 - sympy/core/numbers.py:

Start line: 1711, End line: 1724

```python
class Rational(Number):
    __radd__ = __add__

    @_sympifyit('other', NotImplemented)
    def __sub__(self, other):
        if global_parameters.evaluate:
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
### 58 - sympy/core/numbers.py:

Start line: 1683, End line: 1710

```python
class Rational(Number):

    def __getnewargs__(self):
        return (self.p, self.q)

    def _hashable_content(self):
        return (self.p, self.q)

    def _eval_is_positive(self):
        return self.p > 0

    def _eval_is_zero(self):
        return self.p == 0

    def __neg__(self):
        return Rational(-self.p, self.q)

    @_sympifyit('other', NotImplemented)
    def __add__(self, other):
        if global_parameters.evaluate:
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
### 59 - sympy/core/numbers.py:

Start line: 3206, End line: 3226

```python
class NegativeOne(IntegerConstant, metaclass=Singleton):

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
            if expt in (S.Infinity, S.NegativeInfinity):
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
### 61 - sympy/core/numbers.py:

Start line: 1332, End line: 1362

```python
class Float(Number):

    def _eval_power(self, expt):
        """
        expt is symbolic object but not equal to 0, 1

        (-p)**r -> exp(r*log(-p)) -> exp(r*(log(p) + I*Pi)) ->
                  -> p**r*(sin(Pi*r) + cos(Pi*r)*I)
        """
        if self == 0:
            if expt.is_extended_positive:
                return self
            if expt.is_extended_negative:
                return S.ComplexInfinity
        if isinstance(expt, Number):
            if isinstance(expt, Integer):
                prec = self._prec
                return Float._new(
                    mlib.mpf_pow_int(self._mpf_, expt.p, prec, rnd), prec)
            elif isinstance(expt, Rational) and \
                    expt.p == 1 and expt.q % 2 and self.is_negative:
                return Pow(S.NegativeOne, expt, evaluate=False)*(
                    -self)._eval_power(expt)
            expt, prec = expt._as_mpf_op(self._prec)
            mpfself = self._mpf_
            try:
                y = mpf_pow(mpfself, expt, prec, rnd)
                return Float._new(y, prec)
            except mlib.ComplexResult:
                re, im = mlib.mpc_pow(
                    (mpfself, fzero), (expt, fzero), prec, rnd)
                return Float._new(re, prec) + \
                    Float._new(im, prec)*S.ImaginaryUnit
```
### 68 - sympy/core/numbers.py:

Start line: 1181, End line: 1196

```python
class Float(Number):

    @classmethod
    def _new(cls, _mpf_, _prec, zero=True):
        # special cases
        if zero and _mpf_ == fzero:
            return S.Zero  # Float(0) -> 0.0; Float._new((0,0,0,0)) -> 0
        elif _mpf_ == _mpf_nan:
            return S.NaN
        elif _mpf_ == _mpf_inf:
            return S.Infinity
        elif _mpf_ == _mpf_ninf:
            return S.NegativeInfinity

        obj = Expr.__new__(cls)
        obj._mpf_ = mpf_norm(_mpf_, _prec)
        obj._prec = _prec
        return obj
```
### 69 - sympy/core/numbers.py:

Start line: 1123, End line: 1179

```python
class Float(Number):

    def __new__(cls, num, dps=None, precision=None):
        # ... other code

        if isinstance(num, float):
            _mpf_ = mlib.from_float(num, precision, rnd)
        elif isinstance(num, str):
            _mpf_ = mlib.from_str(num, precision, rnd)
        elif isinstance(num, decimal.Decimal):
            if num.is_finite():
                _mpf_ = mlib.from_str(str(num), precision, rnd)
            elif num.is_nan():
                return S.NaN
            elif num.is_infinite():
                if num > 0:
                    return S.Infinity
                return S.NegativeInfinity
            else:
                raise ValueError("unexpected decimal value %s" % str(num))
        elif isinstance(num, tuple) and len(num) in (3, 4):
            if isinstance(num[1], str):
                # it's a hexadecimal (coming from a pickled object)
                num = list(num)
                # If we're loading an object pickled in Python 2 into
                # Python 3, we may need to strip a tailing 'L' because
                # of a shim for int on Python 3, see issue #13470.
                if num[1].endswith('L'):
                    num[1] = num[1][:-1]
                # Strip leading '0x' - gmpy2 only documents such inputs
                # with base prefix as valid when the 2nd argument (base) is 0.
                # When mpmath uses Sage as the backend, however, it
                # ends up including '0x' when preparing the picklable tuple.
                # See issue #19690.
                if num[1].startswith('0x'):
                    num[1] = num[1][2:]
                # Now we can assume that it is in standard form
                num[1] = MPZ(num[1], 16)
                _mpf_ = tuple(num)
            else:
                if len(num) == 4:
                    # handle normalization hack
                    return Float._new(num, precision)
                else:
                    if not all((
                            num[0] in (0, 1),
                            num[1] >= 0,
                            all(type(i) in (int, int) for i in num)
                            )):
                        raise ValueError('malformed mpf: %s' % (num,))
                    # don't compute number or else it may
                    # over/underflow
                    return Float._new(
                        (num[0], num[1], num[2], bitcount(num[1])),
                        precision)
        else:
            try:
                _mpf_ = num._as_mpf_val(precision)
            except (NotImplementedError, AttributeError):
                _mpf_ = mpmath.mpf(num, prec=precision)._mpf_

        return cls._new(_mpf_, precision, zero=False)
```
### 72 - sympy/core/numbers.py:

Start line: 4029, End line: 4097

```python
E = S.Exp1


class Pi(NumberSymbol, metaclass=Singleton):
    r"""The `\pi` constant.

    Explanation
    ===========

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

    __slots__ = ()

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
            return (Rational(223, 71, 1), Rational(22, 7, 1))

pi = S.Pi
```
