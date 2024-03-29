# sympy__sympy-17150

| **sympy/sympy** | `dcc4430810a88d239d75f16c5c3403cd6926d666` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 18823 |
| **Any found context length** | 595 |
| **Avg pos** | 58.0 |
| **Min pos** | 1 |
| **Max pos** | 57 |
| **Top file pos** | 1 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sympy/functions/elementary/exponential.py b/sympy/functions/elementary/exponential.py
--- a/sympy/functions/elementary/exponential.py
+++ b/sympy/functions/elementary/exponential.py
@@ -481,6 +481,15 @@ class log(Function):
     a logarithm of a different base ``b``, use ``log(x, b)``,
     which is essentially short-hand for ``log(x)/log(b)``.
 
+    Examples
+    ========
+
+    >>> from sympy import log, S
+    >>> log(8, 2)
+    3
+    >>> log(S(8)/3, 2)
+    -log(3)/log(2) + 3
+
     See Also
     ========
 
@@ -522,11 +531,7 @@ def eval(cls, arg, base=None):
                 # or else expand_log in Mul would have to handle this
                 n = multiplicity(base, arg)
                 if n:
-                    den = base**n
-                    if den.is_Integer:
-                        return n + log(arg // den) / log(base)
-                    else:
-                        return n + log(arg / den) / log(base)
+                    return n + log(arg / base**n) / log(base)
                 else:
                     return log(arg)/log(base)
             except ValueError:

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sympy/functions/elementary/exponential.py | 484 | 484 | 57 | 1 | 18823
| sympy/functions/elementary/exponential.py | 525 | 529 | 1 | 1 | 595


## Problem Statement

```
Incorrect extraction of base powers in log class
Evaluating `log(Rational(408,499),2)` produces `zoo`, but it should produce `log(Rational(51,499))/log(2) + 3`.

The issue seems to originate around line `531` in `sympy/functions/elementary/exponential.py` during extraction of base powers, where `arg // den` is evaluated to `0` but should evaluate to `Rational(51,499)`:

                    if den.is_Integer:
                        return n + log(arg // den) / log(base)
                    else:
                        return n + log(arg / den) / log(base)

I would suggest to fix the issue by removing the `if` conditional and keeping the else branch (seems like a case of premature optimization). Alternatively, this also seems to fix the issue:

                    if arg.is_Integer and den.is_Integer:
                        return n + log(arg // den) / log(base)
                    else:
                        return n + log(arg / den) / log(base)

That said, seeing that this code was not changed recently, the issue may run deeper.


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 sympy/functions/elementary/exponential.py** | 505 | 586| 595 | 595 | 7069 | 
| 2 | **1 sympy/functions/elementary/exponential.py** | 612 | 657| 411 | 1006 | 7069 | 
| 3 | **1 sympy/functions/elementary/exponential.py** | 704 | 741| 307 | 1313 | 7069 | 
| 4 | **1 sympy/functions/elementary/exponential.py** | 659 | 669| 136 | 1449 | 7069 | 
| 5 | 2 sympy/core/numbers.py | 1801 | 1837| 404 | 1853 | 37159 | 
| 6 | **2 sympy/functions/elementary/exponential.py** | 743 | 779| 384 | 2237 | 37159 | 
| 7 | 2 sympy/core/numbers.py | 2301 | 2373| 691 | 2928 | 37159 | 
| 8 | **2 sympy/functions/elementary/exponential.py** | 367 | 380| 182 | 3110 | 37159 | 
| 9 | **2 sympy/functions/elementary/exponential.py** | 407 | 425| 223 | 3333 | 37159 | 
| 10 | **2 sympy/functions/elementary/exponential.py** | 588 | 610| 198 | 3531 | 37159 | 
| 11 | **2 sympy/functions/elementary/exponential.py** | 226 | 308| 588 | 4119 | 37159 | 
| 12 | 3 sympy/simplify/powsimp.py | 485 | 572| 857 | 4976 | 43812 | 
| 13 | 4 sympy/core/exprtools.py | 263 | 284| 150 | 5126 | 56256 | 
| 14 | 5 sympy/core/power.py | 1210 | 1232| 244 | 5370 | 71274 | 
| 15 | 5 sympy/core/power.py | 741 | 795| 645 | 6015 | 71274 | 
| 16 | 5 sympy/core/power.py | 541 | 594| 448 | 6463 | 71274 | 
| 17 | 5 sympy/core/numbers.py | 2374 | 2404| 354 | 6817 | 71274 | 
| 18 | 5 sympy/core/numbers.py | 1868 | 1912| 400 | 7217 | 71274 | 
| 19 | **5 sympy/functions/elementary/exponential.py** | 157 | 184| 243 | 7460 | 71274 | 
| 20 | 5 sympy/core/numbers.py | 1782 | 1799| 188 | 7648 | 71274 | 
| 21 | 6 sympy/integrals/risch.py | 340 | 391| 624 | 8272 | 89079 | 
| 22 | 6 sympy/core/power.py | 1275 | 1287| 121 | 8393 | 89079 | 
| 23 | 6 sympy/core/numbers.py | 1914 | 1936| 198 | 8591 | 89079 | 
| 24 | **6 sympy/functions/elementary/exponential.py** | 63 | 115| 350 | 8941 | 89079 | 
| 25 | 6 sympy/core/power.py | 1357 | 1364| 142 | 9083 | 89079 | 
| 26 | 6 sympy/core/numbers.py | 1938 | 1971| 245 | 9328 | 89079 | 
| 27 | 7 sympy/codegen/rewriting.py | 130 | 166| 398 | 9726 | 91168 | 
| 28 | 7 sympy/core/numbers.py | 1727 | 1738| 125 | 9851 | 91168 | 
| 29 | 7 sympy/simplify/powsimp.py | 668 | 692| 251 | 10102 | 91168 | 
| 30 | **7 sympy/functions/elementary/exponential.py** | 427 | 474| 493 | 10595 | 91168 | 
| 31 | 7 sympy/core/numbers.py | 1700 | 1712| 132 | 10727 | 91168 | 
| 32 | 7 sympy/simplify/powsimp.py | 127 | 225| 1015 | 11742 | 91168 | 
| 33 | 7 sympy/core/exprtools.py | 214 | 260| 321 | 12063 | 91168 | 
| 34 | **7 sympy/functions/elementary/exponential.py** | 382 | 405| 223 | 12286 | 91168 | 
| 35 | **7 sympy/functions/elementary/exponential.py** | 1 | 15| 136 | 12422 | 91168 | 
| 36 | 7 sympy/core/numbers.py | 1985 | 2014| 203 | 12625 | 91168 | 
| 37 | 7 sympy/core/power.py | 331 | 412| 803 | 13428 | 91168 | 
| 38 | 7 sympy/core/power.py | 1255 | 1273| 133 | 13561 | 91168 | 
| 39 | 7 sympy/core/numbers.py | 1753 | 1767| 156 | 13717 | 91168 | 
| 40 | 7 sympy/core/numbers.py | 1768 | 1780| 144 | 13861 | 91168 | 
| 41 | 7 sympy/core/numbers.py | 1713 | 1726| 131 | 13992 | 91168 | 
| 42 | 7 sympy/core/power.py | 596 | 636| 293 | 14285 | 91168 | 
| 43 | 7 sympy/core/power.py | 638 | 680| 339 | 14624 | 91168 | 
| 44 | 7 sympy/core/numbers.py | 1739 | 1751| 141 | 14765 | 91168 | 
| 45 | 8 sympy/functions/special/hyper.py | 980 | 1002| 261 | 15026 | 101502 | 
| 46 | 8 sympy/integrals/risch.py | 393 | 425| 370 | 15396 | 101502 | 
| 47 | 8 sympy/core/power.py | 437 | 480| 305 | 15701 | 101502 | 
| 48 | 9 sympy/polys/domains/pythonrational.py | 207 | 287| 500 | 16201 | 103545 | 
| 49 | 9 sympy/core/power.py | 1289 | 1322| 350 | 16551 | 103545 | 
| 50 | 9 sympy/core/power.py | 414 | 435| 248 | 16799 | 103545 | 
| 51 | 9 sympy/core/numbers.py | 1839 | 1866| 188 | 16987 | 103545 | 
| 52 | 9 sympy/core/numbers.py | 2164 | 2252| 750 | 17737 | 103545 | 
| 53 | **9 sympy/functions/elementary/exponential.py** | 310 | 332| 140 | 17877 | 103545 | 
| 54 | 10 sympy/functions/special/zeta_functions.py | 307 | 327| 194 | 18071 | 109079 | 
| 55 | **10 sympy/functions/elementary/exponential.py** | 118 | 155| 317 | 18388 | 109079 | 
| 56 | **10 sympy/functions/elementary/exponential.py** | 671 | 702| 259 | 18647 | 109079 | 
| **-> 57 <-** | **10 sympy/functions/elementary/exponential.py** | 477 | 503| 176 | 18823 | 109079 | 
| 58 | 11 sympy/codegen/cfunctions.py | 215 | 266| 314 | 19137 | 112171 | 
| 59 | 11 sympy/core/power.py | 1 | 19| 142 | 19279 | 112171 | 
| 60 | 11 sympy/core/power.py | 169 | 251| 1060 | 20339 | 112171 | 
| 61 | 11 sympy/core/numbers.py | 1349 | 1379| 333 | 20672 | 112171 | 
| 62 | 11 sympy/core/numbers.py | 2254 | 2299| 347 | 21019 | 112171 | 
| 63 | 11 sympy/simplify/powsimp.py | 253 | 270| 223 | 21242 | 112171 | 
| 64 | 11 sympy/core/power.py | 1183 | 1208| 239 | 21481 | 112171 | 
| 65 | 11 sympy/core/numbers.py | 1597 | 1670| 489 | 21970 | 112171 | 
| 66 | 12 sympy/benchmarks/bench_discrete_log.py | 1 | 51| 578 | 22548 | 113035 | 
| 67 | 12 sympy/codegen/rewriting.py | 169 | 199| 248 | 22796 | 113035 | 
| 68 | 13 sympy/functions/special/error_functions.py | 2371 | 2387| 171 | 22967 | 132539 | 
| 69 | 13 sympy/functions/special/error_functions.py | 1184 | 1205| 265 | 23232 | 132539 | 
| 70 | 14 sympy/simplify/simplify.py | 869 | 913| 415 | 23647 | 148675 | 
| 71 | 14 sympy/core/numbers.py | 2613 | 2640| 228 | 23875 | 148675 | 
| 72 | 14 sympy/core/numbers.py | 1 | 39| 340 | 24215 | 148675 | 
| 73 | 14 sympy/core/power.py | 1110 | 1181| 745 | 24960 | 148675 | 
| 74 | 14 sympy/simplify/powsimp.py | 17 | 101| 843 | 25803 | 148675 | 
| 75 | 14 sympy/core/numbers.py | 3915 | 3975| 364 | 26167 | 148675 | 
| 76 | 14 sympy/core/power.py | 1234 | 1253| 167 | 26334 | 148675 | 
| 77 | 15 sympy/solvers/solvers.py | 3325 | 3371| 429 | 26763 | 180942 | 
| 78 | 15 sympy/core/power.py | 1677 | 1711| 258 | 27021 | 180942 | 


## Patch

```diff
diff --git a/sympy/functions/elementary/exponential.py b/sympy/functions/elementary/exponential.py
--- a/sympy/functions/elementary/exponential.py
+++ b/sympy/functions/elementary/exponential.py
@@ -481,6 +481,15 @@ class log(Function):
     a logarithm of a different base ``b``, use ``log(x, b)``,
     which is essentially short-hand for ``log(x)/log(b)``.
 
+    Examples
+    ========
+
+    >>> from sympy import log, S
+    >>> log(8, 2)
+    3
+    >>> log(S(8)/3, 2)
+    -log(3)/log(2) + 3
+
     See Also
     ========
 
@@ -522,11 +531,7 @@ def eval(cls, arg, base=None):
                 # or else expand_log in Mul would have to handle this
                 n = multiplicity(base, arg)
                 if n:
-                    den = base**n
-                    if den.is_Integer:
-                        return n + log(arg // den) / log(base)
-                    else:
-                        return n + log(arg / den) / log(base)
+                    return n + log(arg / base**n) / log(base)
                 else:
                     return log(arg)/log(base)
             except ValueError:

```

## Test Patch

```diff
diff --git a/sympy/functions/elementary/tests/test_exponential.py b/sympy/functions/elementary/tests/test_exponential.py
--- a/sympy/functions/elementary/tests/test_exponential.py
+++ b/sympy/functions/elementary/tests/test_exponential.py
@@ -212,6 +212,8 @@ def test_log_base():
     assert log(Rational(2, 3), Rational(1, 3)) == -log(2)/log(3) + 1
     assert log(Rational(2, 3), Rational(2, 5)) == \
         log(S(2)/3)/log(S(2)/5)
+    # issue 17148
+    assert log(S(8)/3, 2) == -log(3)/log(2) + 3
 
 
 def test_log_symbolic():

```


## Code snippets

### 1 - sympy/functions/elementary/exponential.py:

Start line: 505, End line: 586

```python
class log(Function):

    @classmethod
    def eval(cls, arg, base=None):
        from sympy import unpolarify
        from sympy.calculus import AccumBounds
        from sympy.sets.setexpr import SetExpr

        arg = sympify(arg)

        if base is not None:
            base = sympify(base)
            if base == 1:
                if arg == 1:
                    return S.NaN
                else:
                    return S.ComplexInfinity
            try:
                # handle extraction of powers of the base now
                # or else expand_log in Mul would have to handle this
                n = multiplicity(base, arg)
                if n:
                    den = base**n
                    if den.is_Integer:
                        return n + log(arg // den) / log(base)
                    else:
                        return n + log(arg / den) / log(base)
                else:
                    return log(arg)/log(base)
            except ValueError:
                pass
            if base is not S.Exp1:
                return cls(arg)/cls(base)
            else:
                return cls(arg)

        if arg.is_Number:
            if arg is S.Zero:
                return S.ComplexInfinity
            elif arg is S.One:
                return S.Zero
            elif arg is S.Infinity:
                return S.Infinity
            elif arg is S.NegativeInfinity:
                return S.Infinity
            elif arg is S.NaN:
                return S.NaN
            elif arg.is_Rational and arg.p == 1:
                return -cls(arg.q)

        if isinstance(arg, exp) and arg.args[0].is_extended_real:
            return arg.args[0]
        elif isinstance(arg, exp_polar):
            return unpolarify(arg.exp)
        elif isinstance(arg, AccumBounds):
            if arg.min.is_positive:
                return AccumBounds(log(arg.min), log(arg.max))
            else:
                return
        elif isinstance(arg, SetExpr):
            return arg._eval_func(cls)

        if arg.is_number:
            if arg.is_negative:
                return S.Pi * S.ImaginaryUnit + cls(-arg)
            elif arg is S.ComplexInfinity:
                return S.ComplexInfinity
            elif arg is S.Exp1:
                return S.One

        # don't autoexpand Pow or Mul (see the issue 3351):
        if not arg.is_Add:
            coeff = arg.as_coefficient(S.ImaginaryUnit)

            if coeff is not None:
                if coeff is S.Infinity:
                    return S.Infinity
                elif coeff is S.NegativeInfinity:
                    return S.Infinity
                elif coeff.is_Rational:
                    if coeff.is_nonnegative:
                        return S.Pi * S.ImaginaryUnit * S.Half + cls(coeff)
                    else:
                        return -S.Pi * S.ImaginaryUnit * S.Half + cls(-coeff)
```
### 2 - sympy/functions/elementary/exponential.py:

Start line: 612, End line: 657

```python
class log(Function):

    def _eval_expand_log(self, deep=True, **hints):
        from sympy import unpolarify, expand_log
        from sympy.concrete import Sum, Product
        force = hints.get('force', False)
        if (len(self.args) == 2):
            return expand_log(self.func(*self.args), deep=deep, force=force)
        arg = self.args[0]
        if arg.is_Integer:
            # remove perfect powers
            p = perfect_power(int(arg))
            if p is not False:
                return p[1]*self.func(p[0])
        elif arg.is_Rational:
            return log(arg.p) - log(arg.q)
        elif arg.is_Mul:
            expr = []
            nonpos = []
            for x in arg.args:
                if force or x.is_positive or x.is_polar:
                    a = self.func(x)
                    if isinstance(a, log):
                        expr.append(self.func(x)._eval_expand_log(**hints))
                    else:
                        expr.append(a)
                elif x.is_negative:
                    a = self.func(-x)
                    expr.append(a)
                    nonpos.append(S.NegativeOne)
                else:
                    nonpos.append(x)
            return Add(*expr) + log(Mul(*nonpos))
        elif arg.is_Pow or isinstance(arg, exp):
            if force or (arg.exp.is_extended_real and (arg.base.is_positive or ((arg.exp+1)
                .is_positive and (arg.exp-1).is_nonpositive))) or arg.base.is_polar:
                b = arg.base
                e = arg.exp
                a = self.func(b)
                if isinstance(a, log):
                    return unpolarify(e) * a._eval_expand_log(**hints)
                else:
                    return unpolarify(e) * a
        elif isinstance(arg, Product):
            if arg.function.is_positive:
                return Sum(log(arg.function), *arg.limits)

        return self.func(arg)
```
### 3 - sympy/functions/elementary/exponential.py:

Start line: 704, End line: 741

```python
class log(Function):

    def _eval_is_rational(self):
        s = self.func(*self.args)
        if s.func == self.func:
            if (self.args[0] - 1).is_zero:
                return True
            if s.args[0].is_rational and fuzzy_not((self.args[0] - 1).is_zero):
                return False
        else:
            return s.is_rational

    def _eval_is_algebraic(self):
        s = self.func(*self.args)
        if s.func == self.func:
            if (self.args[0] - 1).is_zero:
                return True
            elif fuzzy_not((self.args[0] - 1).is_zero):
                if self.args[0].is_algebraic:
                    return False
        else:
            return s.is_algebraic

    def _eval_is_extended_real(self):
        return self.args[0].is_extended_positive

    def _eval_is_finite(self):
        arg = self.args[0]
        if arg.is_zero:
            return False
        return arg.is_finite

    def _eval_is_extended_positive(self):
        return (self.args[0] - 1).is_extended_positive

    def _eval_is_zero(self):
        return (self.args[0] - 1).is_zero

    def _eval_is_extended_nonnegative(self):
        return (self.args[0] - 1).is_extended_nonnegative
```
### 4 - sympy/functions/elementary/exponential.py:

Start line: 659, End line: 669

```python
class log(Function):

    def _eval_simplify(self, ratio, measure, rational, inverse):
        from sympy.simplify.simplify import expand_log, simplify, inversecombine
        if (len(self.args) == 2):
            return simplify(self.func(*self.args), ratio=ratio, measure=measure,
                            rational=rational, inverse=inverse)
        expr = self.func(simplify(self.args[0], ratio=ratio, measure=measure,
                         rational=rational, inverse=inverse))
        if inverse:
            expr = inversecombine(expr)
        expr = expand_log(expr, deep=True)
        return min([expr, self], key=measure)
```
### 5 - sympy/core/numbers.py:

Start line: 1801, End line: 1837

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
                if self.p != 1:
                    # (4/3)**(5/6) -> 4**(5/6)*3**(-5/6)
                    return Integer(self.p)**expt*Integer(self.q)**(-expt)
                # as the above caught negative self.p, now self is positive
                return Integer(self.q)**Rational(
                expt.p*(expt.q - 1), expt.q) / \
                    Integer(self.q)**Integer(expt.p)

        if self.is_extended_negative and expt.is_even:
            return (-self)**expt

        return
```
### 6 - sympy/functions/elementary/exponential.py:

Start line: 743, End line: 779

```python
class log(Function):

    def _eval_nseries(self, x, n, logx):
        # NOTE Please see the comment at the beginning of this file, labelled
        #      IMPORTANT.
        from sympy import cancel, Order
        if not logx:
            logx = log(x)
        if self.args[0] == x:
            return logx
        arg = self.args[0]
        k, l = Wild("k"), Wild("l")
        r = arg.match(k*x**l)
        if r is not None:
            k, l = r[k], r[l]
            if l != 0 and not l.has(x) and not k.has(x):
                r = log(k) + l*logx  # XXX true regardless of assumptions?
                return r

        # TODO new and probably slow
        s = self.args[0].nseries(x, n=n, logx=logx)
        while s.is_Order:
            n += 1
            s = self.args[0].nseries(x, n=n, logx=logx)
        a, b = s.leadterm(x)
        p = cancel(s/(a*x**b) - 1)
        g = None
        l = []
        for i in range(n + 2):
            g = log.taylor_term(i, p, g)
            g = g.nseries(x, n=n, logx=logx)
            l.append(g)
        return log(a) + b*logx + Add(*l) + Order(p**n, x)

    def _eval_as_leading_term(self, x):
        arg = self.args[0].as_leading_term(x)
        if arg is S.One:
            return (self.args[0] - 1).as_leading_term(x)
        return self.func(arg)
```
### 7 - sympy/core/numbers.py:

Start line: 2301, End line: 2373

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
        from sympy.ntheory.factor_ import perfect_power

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
### 8 - sympy/functions/elementary/exponential.py:

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
### 9 - sympy/functions/elementary/exponential.py:

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
### 10 - sympy/functions/elementary/exponential.py:

Start line: 588, End line: 610

```python
class log(Function):

    def as_base_exp(self):
        """
        Returns this function in the form (base, exponent).
        """
        return self, S.One

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):  # of log(1+x)
        r"""
        Returns the next term in the Taylor series expansion of `\log(1+x)`.
        """
        from sympy import powsimp
        if n < 0:
            return S.Zero
        x = sympify(x)
        if n == 0:
            return x
        if previous_terms:
            p = previous_terms[-1]
            if p is not None:
                return powsimp((-n) * p * x / (n + 1), deep=True, combine='exp')
        return (1 - 2*(n % 2)) * x**(n + 1)/(n + 1)
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
### 19 - sympy/functions/elementary/exponential.py:

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
### 24 - sympy/functions/elementary/exponential.py:

Start line: 63, End line: 115

```python
class ExpBase(Function):

    @property
    def exp(self):
        """
        Returns the exponent of the function.
        """
        return self.args[0]

    def as_base_exp(self):
        """
        Returns the 2-tuple (base, exponent).
        """
        return self.func(1), Mul(*self.args)

    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate())

    def _eval_is_finite(self):
        arg = self.args[0]
        if arg.is_infinite:
            if arg.is_negative:
                return True
            if arg.is_positive:
                return False
        if arg.is_finite:
            return True

    def _eval_is_rational(self):
        s = self.func(*self.args)
        if s.func == self.func:
            if s.exp is S.Zero:
                return True
            elif s.exp.is_rational and fuzzy_not(s.exp.is_zero):
                return False
        else:
            return s.is_rational

    def _eval_is_zero(self):
        return (self.args[0] is S.NegativeInfinity)

    def _eval_power(self, other):
        """exp(arg)**e -> exp(arg*e) if assumptions allow it.
        """
        b, e = self.as_base_exp()
        return Pow._eval_power(Pow(b, e, evaluate=False), other)

    def _eval_expand_power_exp(self, **hints):
        arg = self.args[0]
        if arg.is_Add and arg.is_commutative:
            expr = 1
            for x in arg.args:
                expr *= self.func(x)
            return expr
        return self.func(arg)
```
### 30 - sympy/functions/elementary/exponential.py:

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
### 34 - sympy/functions/elementary/exponential.py:

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
### 35 - sympy/functions/elementary/exponential.py:

Start line: 1, End line: 15

```python
from __future__ import print_function, division

from sympy.core import sympify
from sympy.core.add import Add
from sympy.core.cache import cacheit
from sympy.core.compatibility import range
from sympy.core.function import Function, ArgumentIndexError, _coeff_isneg
from sympy.core.logic import fuzzy_not
from sympy.core.mul import Mul
from sympy.core.numbers import Integer
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Wild, Dummy
from sympy.functions.combinatorial.factorials import factorial
from sympy.ntheory import multiplicity, perfect_power
```
### 53 - sympy/functions/elementary/exponential.py:

Start line: 310, End line: 332

```python
class exp(ExpBase):

    @property
    def base(self):
        """
        Returns the base of the exponential function.
        """
        return S.Exp1

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        """
        Calculates the next term in the Taylor series expansion.
        """
        if n < 0:
            return S.Zero
        if n == 0:
            return S.One
        x = sympify(x)
        if previous_terms:
            p = previous_terms[-1]
            if p is not None:
                return p * x / n
        return x**n/factorial(n)
```
### 55 - sympy/functions/elementary/exponential.py:

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
### 56 - sympy/functions/elementary/exponential.py:

Start line: 671, End line: 702

```python
class log(Function):

    def as_real_imag(self, deep=True, **hints):
        """
        Returns this function as a complex coordinate.

        Examples
        ========

        >>> from sympy import I
        >>> from sympy.abc import x
        >>> from sympy.functions import log
        >>> log(x).as_real_imag()
        (log(Abs(x)), arg(x))
        >>> log(I).as_real_imag()
        (0, pi/2)
        >>> log(1 + I).as_real_imag()
        (log(sqrt(2)), pi/4)
        >>> log(I*x).as_real_imag()
        (log(Abs(x)), arg(I*x))

        """
        from sympy import Abs, arg
        if deep:
            abs = Abs(self.args[0].expand(deep, **hints))
            arg = arg(self.args[0].expand(deep, **hints))
        else:
            abs = Abs(self.args[0])
            arg = arg(self.args[0])
        if hints.get('log', False):  # Expand the log
            hints['complex'] = False
            return (log(abs).expand(deep, **hints), arg)
        else:
            return (log(abs), arg)
```
### 57 - sympy/functions/elementary/exponential.py:

Start line: 477, End line: 503

```python
class log(Function):
    r"""
    The natural logarithm function `\ln(x)` or `\log(x)`.
    Logarithms are taken with the natural base, `e`. To get
    a logarithm of a different base ``b``, use ``log(x, b)``,
    which is essentially short-hand for ``log(x)/log(b)``.

    See Also
    ========

    exp
    """

    def fdiff(self, argindex=1):
        """
        Returns the first derivative of the function.
        """
        if argindex == 1:
            return 1/self.args[0]
        else:
            raise ArgumentIndexError(self, argindex)

    def inverse(self, argindex=1):
        r"""
        Returns `e^x`, the inverse function of `\log(x)`.
        """
        return exp
```
