# sympy__sympy-21379

| **sympy/sympy** | `624217179aaf8d094e6ff75b7493ad1ee47599b0` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | - |
| **Avg pos** | - |
| **Min pos** | - |
| **Max pos** | - |
| **Top file pos** | - |
| **Missing snippets** | 2 |
| **Missing patch files** | 1 |


## Expected patch

```diff
diff --git a/sympy/core/mod.py b/sympy/core/mod.py
--- a/sympy/core/mod.py
+++ b/sympy/core/mod.py
@@ -40,6 +40,7 @@ def eval(cls, p, q):
         from sympy.core.mul import Mul
         from sympy.core.singleton import S
         from sympy.core.exprtools import gcd_terms
+        from sympy.polys.polyerrors import PolynomialError
         from sympy.polys.polytools import gcd
 
         def doit(p, q):
@@ -166,10 +167,13 @@ def doit(p, q):
         # XXX other possibilities?
 
         # extract gcd; any further simplification should be done by the user
-        G = gcd(p, q)
-        if G != 1:
-            p, q = [
-                gcd_terms(i/G, clear=False, fraction=False) for i in (p, q)]
+        try:
+            G = gcd(p, q)
+            if G != 1:
+                p, q = [gcd_terms(i/G, clear=False, fraction=False)
+                        for i in (p, q)]
+        except PolynomialError:  # issue 21373
+            G = S.One
         pwas, qwas = p, q
 
         # simplify terms

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sympy/core/mod.py | 43 | 43 | - | - | -
| sympy/core/mod.py | 169 | 172 | - | - | -


## Problem Statement

```
Unexpected `PolynomialError` when using simple `subs()` for particular expressions
I am seeing weird behavior with `subs` for particular expressions with hyperbolic sinusoids with piecewise arguments. When applying `subs`, I obtain an unexpected `PolynomialError`. For context, I was umbrella-applying a casting from int to float of all int atoms for a bunch of random expressions before using a tensorflow lambdify to avoid potential tensorflow type errors. You can pretend the expression below has a `+ 1` at the end, but below is the MWE that I could produce.

See the expression below, and the conditions in which the exception arises.

Sympy version: 1.8.dev

\`\`\`python
from sympy import *
from sympy.core.cache import clear_cache

x, y, z = symbols('x y z')

clear_cache()
expr = exp(sinh(Piecewise((x, y > x), (y, True)) / z))
# This works fine
expr.subs({1: 1.0})

clear_cache()
x, y, z = symbols('x y z', real=True)
expr = exp(sinh(Piecewise((x, y > x), (y, True)) / z))
# This fails with "PolynomialError: Piecewise generators do not make sense"
expr.subs({1: 1.0})  # error
# Now run it again (isympy...) w/o clearing cache and everything works as expected without error
expr.subs({1: 1.0})
\`\`\`

I am not really sure where the issue is, but I think it has something to do with the order of assumptions in this specific type of expression. Here is what I found-

- The error only (AFAIK) happens with `cosh` or `tanh` in place of `sinh`, otherwise it succeeds
- The error goes away if removing the division by `z`
- The error goes away if removing `exp` (but stays for most unary functions, `sin`, `log`, etc.)
- The error only happens with real symbols for `x` and `y` (`z` does not have to be real)

Not too sure how to debug this one.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 sympy/core/expr.py | 3153 | 3211| 595 | 595 | 33788 | 
| 2 | 2 sympy/core/power.py | 851 | 908| 695 | 1290 | 50892 | 
| 3 | 3 sympy/solvers/solvers.py | 2550 | 2677| 1323 | 2613 | 82583 | 
| 4 | 3 sympy/solvers/solvers.py | 380 | 832| 4494 | 7107 | 82583 | 
| 5 | 4 sympy/physics/mechanics/functions.py | 658 | 678| 248 | 7355 | 88843 | 
| 6 | 4 sympy/physics/mechanics/functions.py | 644 | 656| 141 | 7496 | 88843 | 
| 7 | 5 sympy/solvers/bivariate.py | 281 | 362| 830 | 8326 | 94099 | 
| 8 | 6 sympy/concrete/summations.py | 1 | 29| 310 | 8636 | 108311 | 
| 9 | 7 sympy/integrals/rubi/utility_function.py | 6501 | 7009| 6205 | 14841 | 193258 | 
| 10 | 8 sympy/codegen/rewriting.py | 137 | 175| 397 | 15238 | 195733 | 
| 11 | 8 sympy/codegen/rewriting.py | 263 | 293| 298 | 15536 | 195733 | 
| 12 | 8 sympy/solvers/solvers.py | 3366 | 3483| 1593 | 17129 | 195733 | 


## Missing Patch Files

 * 1: sympy/core/mod.py

### Hint

```
Some functions call `Mod` when evaluated. That does not work well with arguments involving `Piecewise` expressions. In particular, calling `gcd` will lead to `PolynomialError`. That error should be caught by something like this:
\`\`\`
--- a/sympy/core/mod.py
+++ b/sympy/core/mod.py
@@ -40,6 +40,7 @@ def eval(cls, p, q):
         from sympy.core.mul import Mul
         from sympy.core.singleton import S
         from sympy.core.exprtools import gcd_terms
+        from sympy.polys.polyerrors import PolynomialError
         from sympy.polys.polytools import gcd
 
         def doit(p, q):
@@ -166,10 +167,13 @@ def doit(p, q):
         # XXX other possibilities?
 
         # extract gcd; any further simplification should be done by the user
-        G = gcd(p, q)
-        if G != 1:
-            p, q = [
-                gcd_terms(i/G, clear=False, fraction=False) for i in (p, q)]
+        try:
+            G = gcd(p, q)
+            if G != 1:
+                p, q = [gcd_terms(i/G, clear=False, fraction=False)
+                        for i in (p, q)]
+        except PolynomialError:
+            G = S.One
         pwas, qwas = p, q
 
         # simplify terms
\`\`\`
I can't seem to reproduce the OP problem. One suggestion for debugging is to disable the cache e.g. `SYMPY_USE_CACHE=no` but if that makes the problem go away then I guess it's to do with caching somehow and I'm not sure how to debug...

I can see what @jksuom is referring to:
\`\`\`python
In [2]: (Piecewise((x, y > x), (y, True)) / z) % 1
---------------------------------------------------------------------------
PolynomialError
\`\`\`
That should be fixed.

As an aside you might prefer to use `nfloat` rather than `expr.subs({1:1.0})`:
https://docs.sympy.org/latest/modules/core.html#sympy.core.function.nfloat
@oscarbenjamin My apologies - I missed a line in the post recreating the expression with real x/y/z. Here is the minimum code to reproduce (may require running w/o cache):
\`\`\`python
from sympy import *

x, y, z = symbols('x y z', real=True)
expr = exp(sinh(Piecewise((x, y > x), (y, True)) / z))
expr.subs({1: 1.0})
\`\`\`

Your code minimally identifies the real problem, however. Thanks for pointing out `nfloat`, but this also induces the exact same error.


@jksuom I can confirm that your patch fixes the issue on my end! I can put in a PR, and add the minimal test given by @oscarbenjamin, if you would like
Okay I can reproduce it now.

The PR would be good thanks.

I think that we also need to figure out what the caching issue is though. The error should be deterministic.

I was suggesting `nfloat` not to fix this issue but because it's possibly a better way of doing what you suggested. I expect that tensorflow is more efficient with integer exponents than float exponents.
This is the full traceback:
\`\`\`python
Traceback (most recent call last):
  File "/Users/enojb/current/sympy/sympy/sympy/core/assumptions.py", line 454, in getit
    return self._assumptions[fact]
KeyError: 'zero'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "y.py", line 5, in <module>
    expr.subs({1: 1.0})
  File "/Users/enojb/current/sympy/sympy/sympy/core/basic.py", line 949, in subs
    rv = rv._subs(old, new, **kwargs)
  File "/Users/enojb/current/sympy/sympy/sympy/core/cache.py", line 72, in wrapper
    retval = cfunc(*args, **kwargs)
  File "/Users/enojb/current/sympy/sympy/sympy/core/basic.py", line 1063, in _subs
    rv = fallback(self, old, new)
  File "/Users/enojb/current/sympy/sympy/sympy/core/basic.py", line 1040, in fallback
    rv = self.func(*args)
  File "/Users/enojb/current/sympy/sympy/sympy/core/cache.py", line 72, in wrapper
    retval = cfunc(*args, **kwargs)
  File "/Users/enojb/current/sympy/sympy/sympy/core/function.py", line 473, in __new__
    result = super().__new__(cls, *args, **options)
  File "/Users/enojb/current/sympy/sympy/sympy/core/cache.py", line 72, in wrapper
    retval = cfunc(*args, **kwargs)
  File "/Users/enojb/current/sympy/sympy/sympy/core/function.py", line 285, in __new__
    evaluated = cls.eval(*args)
  File "/Users/enojb/current/sympy/sympy/sympy/functions/elementary/exponential.py", line 369, in eval
    if arg.is_zero:
  File "/Users/enojb/current/sympy/sympy/sympy/core/assumptions.py", line 458, in getit
    return _ask(fact, self)
  File "/Users/enojb/current/sympy/sympy/sympy/core/assumptions.py", line 513, in _ask
    _ask(pk, obj)
  File "/Users/enojb/current/sympy/sympy/sympy/core/assumptions.py", line 513, in _ask
    _ask(pk, obj)
  File "/Users/enojb/current/sympy/sympy/sympy/core/assumptions.py", line 513, in _ask
    _ask(pk, obj)
  [Previous line repeated 2 more times]
  File "/Users/enojb/current/sympy/sympy/sympy/core/assumptions.py", line 501, in _ask
    a = evaluate(obj)
  File "/Users/enojb/current/sympy/sympy/sympy/functions/elementary/hyperbolic.py", line 251, in _eval_is_real
    return (im%pi).is_zero
  File "/Users/enojb/current/sympy/sympy/sympy/core/decorators.py", line 266, in _func
    return func(self, other)
  File "/Users/enojb/current/sympy/sympy/sympy/core/decorators.py", line 136, in binary_op_wrapper
    return func(self, other)
  File "/Users/enojb/current/sympy/sympy/sympy/core/expr.py", line 280, in __mod__
    return Mod(self, other)
  File "/Users/enojb/current/sympy/sympy/sympy/core/cache.py", line 72, in wrapper
    retval = cfunc(*args, **kwargs)
  File "/Users/enojb/current/sympy/sympy/sympy/core/function.py", line 473, in __new__
    result = super().__new__(cls, *args, **options)
  File "/Users/enojb/current/sympy/sympy/sympy/core/cache.py", line 72, in wrapper
    retval = cfunc(*args, **kwargs)
  File "/Users/enojb/current/sympy/sympy/sympy/core/function.py", line 285, in __new__
    evaluated = cls.eval(*args)
  File "/Users/enojb/current/sympy/sympy/sympy/core/mod.py", line 169, in eval
    G = gcd(p, q)
  File "/Users/enojb/current/sympy/sympy/sympy/polys/polytools.py", line 5306, in gcd
    (F, G), opt = parallel_poly_from_expr((f, g), *gens, **args)
  File "/Users/enojb/current/sympy/sympy/sympy/polys/polytools.py", line 4340, in parallel_poly_from_expr
    return _parallel_poly_from_expr(exprs, opt)
  File "/Users/enojb/current/sympy/sympy/sympy/polys/polytools.py", line 4399, in _parallel_poly_from_expr
    raise PolynomialError("Piecewise generators do not make sense")
sympy.polys.polyerrors.PolynomialError: Piecewise generators do not make sense
\`\`\`
The issue arises during a query in the old assumptions. The exponential function checks if its argument is zero here:
https://github.com/sympy/sympy/blob/624217179aaf8d094e6ff75b7493ad1ee47599b0/sympy/functions/elementary/exponential.py#L369
That gives:
\`\`\`python
In [1]: x, y, z = symbols('x y z', real=True)

In [2]: sinh(Piecewise((x, y > x), (y, True)) * z**-1.0).is_zero
---------------------------------------------------------------------------
KeyError
\`\`\`
Before processing the assumptions query the value of the queried assumption is stored as `None` here:
https://github.com/sympy/sympy/blob/624217179aaf8d094e6ff75b7493ad1ee47599b0/sympy/core/assumptions.py#L491-L493
That `None` remains there if an exception is raised during the query:
\`\`\`python
In [1]: x, y, z = symbols('x y z', real=True)

In [2]: S = sinh(Piecewise((x, y > x), (y, True)) * z**-1.0)

In [3]: S._assumptions
Out[3]: {}

In [4]: try:
   ...:     S.is_zero
   ...: except Exception as e:
   ...:     print(e)
   ...: 
Piecewise generators do not make sense

In [5]: S._assumptions
Out[5]: 
{'zero': None,
 'extended_positive': None,
 'extended_real': None,
 'negative': None,
 'commutative': True,
 'extended_negative': None,
 'positive': None,
 'real': None}
\`\`\`
A subsequent call to create the same expression returns the same object due to the cache and the object still has `None` is its assumptions dict:
\`\`\`python
In [6]: S2 = sinh(Piecewise((x, y > x), (y, True)) * z**-1.0)

In [7]: S2 is S
Out[7]: True

In [8]: S2._assumptions
Out[8]: 
{'zero': None,
 'extended_positive': None,
 'extended_real': None,
 'negative': None,
 'commutative': True,
 'extended_negative': None,
 'positive': None,
 'real': None}

In [9]: S2.is_zero

In [10]: exp(sinh(Piecewise((x, y > x), (y, True)) * z**-1.0))
Out[10]: 
     ⎛ -1.0 ⎛⎧x  for x < y⎞⎞
 sinh⎜z    ⋅⎜⎨            ⎟⎟
     ⎝      ⎝⎩y  otherwise⎠⎠
ℯ  
\`\`\`
Subsequent `is_zero` checks just return `None` from the assumptions dict without calling the handlers so they pass without raising.

The reason the `is_zero` handler raises first time around is due to the `sinh.is_real` handler which does this:
https://github.com/sympy/sympy/blob/624217179aaf8d094e6ff75b7493ad1ee47599b0/sympy/functions/elementary/hyperbolic.py#L250-L251
The `%` leads to `Mod` with the Piecewise which calls `gcd` as @jksuom showed above.

There are a few separate issues here:

1. The old assumptions system stores `None` when running a query but doesn't remove that `None` when an exception is raised.
2. `Mod` calls `gcd` on the argument when it is a Piecewise and `gcd` without catching the possible exception..
3. The `gcd` function raises an exception when given a `Piecewise`.

The fix suggested by @jksuom is for 2. which seems reasonable and I think we can merge a PR for that to fix using `Piecewise` with `Mod`.

I wonder about 3. as well though. Should `gcd` with a `Piecewise` raise an exception? If so then maybe `Mod` shouldn't be calling `gcd` at all. Perhaps just something like `gcd_terms` or `factor_terms` should be used there.

For point 1. I think that really the best solution is not putting `None` into the assumptions dict at all as there are other ways that it can lead to non-deterministic behaviour. Removing that line leads to a lot of different examples of RecursionError though (personally I consider each of those to be a bug in the old assumptions system).
I'll put a PR together. And, ah I see, yes you are right - good point (regarding TF float exponents).

I cannot comment on 1 as I'm not really familiar with the assumptions systems. But, regarding 3, would this exception make more sense as a `NotImplementedError` in `gcd`? Consider the potential behavior where `gcd` is applied to each condition of a `Piecewise` expression:

\`\`\`python
In [1]: expr = Piecewise((x, x > 2), (2, True))

In [2]: expr
Out[2]: 
⎧x  for x > 2
⎨            
⎩2  otherwise

In [3]: gcd(x, x)
Out[3]: x

In [4]: gcd(2, x)
Out[4]: 1

In [5]: gcd(expr, x)  # current behavior
PolynomialError: Piecewise generators do not make sense

In [6]: gcd(expr, x)  # potential new behavior?
Out[6]: 
⎧x  for x > 2
⎨            
⎩1  otherwise
\`\`\`

That would be what I expect from `gcd` here. For the `gcd` of two `Piecewise` expressions, this gets messier and I think would involve intersecting sets of conditions.
```

## Patch

```diff
diff --git a/sympy/core/mod.py b/sympy/core/mod.py
--- a/sympy/core/mod.py
+++ b/sympy/core/mod.py
@@ -40,6 +40,7 @@ def eval(cls, p, q):
         from sympy.core.mul import Mul
         from sympy.core.singleton import S
         from sympy.core.exprtools import gcd_terms
+        from sympy.polys.polyerrors import PolynomialError
         from sympy.polys.polytools import gcd
 
         def doit(p, q):
@@ -166,10 +167,13 @@ def doit(p, q):
         # XXX other possibilities?
 
         # extract gcd; any further simplification should be done by the user
-        G = gcd(p, q)
-        if G != 1:
-            p, q = [
-                gcd_terms(i/G, clear=False, fraction=False) for i in (p, q)]
+        try:
+            G = gcd(p, q)
+            if G != 1:
+                p, q = [gcd_terms(i/G, clear=False, fraction=False)
+                        for i in (p, q)]
+        except PolynomialError:  # issue 21373
+            G = S.One
         pwas, qwas = p, q
 
         # simplify terms

```

## Test Patch

```diff
diff --git a/sympy/core/tests/test_arit.py b/sympy/core/tests/test_arit.py
--- a/sympy/core/tests/test_arit.py
+++ b/sympy/core/tests/test_arit.py
@@ -1913,6 +1913,16 @@ def test_Mod():
     assert Mod(x, y).rewrite(floor) == x - y*floor(x/y)
     assert ((x - Mod(x, y))/y).rewrite(floor) == floor(x/y)
 
+    # issue 21373
+    from sympy.functions.elementary.trigonometric import sinh
+    from sympy.functions.elementary.piecewise import Piecewise
+
+    x_r, y_r = symbols('x_r y_r', real=True)
+    (Piecewise((x_r, y_r > x_r), (y_r, True)) / z) % 1
+    expr = exp(sinh(Piecewise((x_r, y_r > x_r), (y_r, True)) / z))
+    expr.subs({1: 1.0})
+    sinh(Piecewise((x_r, y_r > x_r), (y_r, True)) * z ** -1.0).is_zero
+
 
 def test_Mod_Pow():
     # modular exponentiation

```


## Code snippets

### 1 - sympy/core/expr.py:

Start line: 3153, End line: 3211

```python
@sympify_method_args
class Expr(Basic, EvalfMixin):

    def aseries(self, x=None, n=6, bound=0, hir=False):
        # ... other code

        if x.is_positive is x.is_negative is None:
            xpos = Dummy('x', positive=True)
            return self.subs(x, xpos).aseries(xpos, n, bound, hir).subs(xpos, x)

        om, exps = mrv(self, x)

        # We move one level up by replacing `x` by `exp(x)`, and then
        # computing the asymptotic series for f(exp(x)). Then asymptotic series
        # can be obtained by moving one-step back, by replacing x by ln(x).

        if x in om:
            s = self.subs(x, exp(x)).aseries(x, n, bound, hir).subs(x, log(x))
            if s.getO():
                return s + Order(1/x**n, (x, S.Infinity))
            return s

        k = Dummy('k', positive=True)
        # f is rewritten in terms of omega
        func, logw = rewrite(exps, om, x, k)

        if self in om:
            if bound <= 0:
                return self
            s = (self.exp).aseries(x, n, bound=bound)
            s = s.func(*[t.removeO() for t in s.args])
            res = exp(s.subs(x, 1/x).as_leading_term(x).subs(x, 1/x))

            func = exp(self.args[0] - res.args[0]) / k
            logw = log(1/res)

        s = func.series(k, 0, n)

        # Hierarchical series
        if hir:
            return s.subs(k, exp(logw))

        o = s.getO()
        terms = sorted(Add.make_args(s.removeO()), key=lambda i: int(i.as_coeff_exponent(k)[1]))
        s = S.Zero
        has_ord = False

        # Then we recursively expand these coefficients one by one into
        # their asymptotic series in terms of their most rapidly varying subexpressions.
        for t in terms:
            coeff, expo = t.as_coeff_exponent(k)
            if coeff.has(x):
                # Recursive step
                snew = coeff.aseries(x, n, bound=bound-1)
                if has_ord and snew.getO():
                    break
                elif snew.getO():
                    has_ord = True
                s += (snew * k**expo)
            else:
                s += t

        if not o or has_ord:
            return s.subs(k, exp(logw))
        return (s + o).subs(k, exp(logw))
```
### 2 - sympy/core/power.py:

Start line: 851, End line: 908

```python
class Pow(Expr):

    def _eval_subs(self, old, new):
        # ... other code

        if old == self.base or (old == exp and self.base == S.Exp1):
            if new.is_Function and isinstance(new, Callable):
                return new(self.exp._subs(old, new))
            else:
                return new**self.exp._subs(old, new)

        # issue 10829: (4**x - 3*y + 2).subs(2**x, y) -> y**2 - 3*y + 2
        if isinstance(old, self.func) and self.exp == old.exp:
            l = log(self.base, old.base)
            if l.is_Number:
                return Pow(new, l)

        if isinstance(old, self.func) and self.base == old.base:
            if self.exp.is_Add is False:
                ct1 = self.exp.as_independent(Symbol, as_Add=False)
                ct2 = old.exp.as_independent(Symbol, as_Add=False)
                ok, pow, remainder_pow = _check(ct1, ct2, old)
                if ok:
                    # issue 5180: (x**(6*y)).subs(x**(3*y),z)->z**2
                    result = self.func(new, pow)
                    if remainder_pow is not None:
                        result = Mul(result, Pow(old.base, remainder_pow))
                    return result
            else:  # b**(6*x + a).subs(b**(3*x), y) -> y**2 * b**a
                # exp(exp(x) + exp(x**2)).subs(exp(exp(x)), w) -> w * exp(exp(x**2))
                oarg = old.exp
                new_l = []
                o_al = []
                ct2 = oarg.as_coeff_mul()
                for a in self.exp.args:
                    newa = a._subs(old, new)
                    ct1 = newa.as_coeff_mul()
                    ok, pow, remainder_pow = _check(ct1, ct2, old)
                    if ok:
                        new_l.append(new**pow)
                        if remainder_pow is not None:
                            o_al.append(remainder_pow)
                        continue
                    elif not old.is_commutative and not newa.is_integer:
                        # If any term in the exponent is non-integer,
                        # we do not do any substitutions in the noncommutative case
                        return
                    o_al.append(newa)
                if new_l:
                    expo = Add(*o_al)
                    new_l.append(Pow(self.base, expo, evaluate=False) if expo != 1 else self.base)
                    return Mul(*new_l)

        if (isinstance(old, exp) or (old.is_Pow and old.base is S.Exp1)) and self.exp.is_extended_real and self.base.is_positive:
            ct1 = old.exp.as_independent(Symbol, as_Add=False)
            ct2 = (self.exp*log(self.base)).as_independent(
                Symbol, as_Add=False)
            ok, pow, remainder_pow = _check(ct1, ct2, old)
            if ok:
                result = self.func(new, pow)  # (2**x).subs(exp(x*log(2)), z) -> z
                if remainder_pow is not None:
                    result = Mul(result, Pow(old.base, remainder_pow))
                return result
```
### 3 - sympy/solvers/solvers.py:

Start line: 2550, End line: 2677

```python
def _tsolve(eq, sym, **flags):
    # ... other code
    try:
        if lhs.is_Add:
            # it's time to try factoring; powdenest is used
            # to try get powers in standard form for better factoring
            f = factor(powdenest(lhs - rhs))
            if f.is_Mul:
                return _solve(f, sym, **flags)
            if rhs:
                f = logcombine(lhs, force=flags.get('force', True))
                if f.count(log) != lhs.count(log):
                    if isinstance(f, log):
                        return _solve(f.args[0] - exp(rhs), sym, **flags)
                    return _tsolve(f - rhs, sym, **flags)

        elif lhs.is_Pow:
            if lhs.exp.is_Integer:
                if lhs - rhs != eq:
                    return _solve(lhs - rhs, sym, **flags)

            if sym not in lhs.exp.free_symbols:
                return _solve(lhs.base - rhs**(1/lhs.exp), sym, **flags)

            # _tsolve calls this with Dummy before passing the actual number in.
            if any(t.is_Dummy for t in rhs.free_symbols):
                raise NotImplementedError # _tsolve will call here again...

            # a ** g(x) == 0
            if not rhs:
                # f(x)**g(x) only has solutions where f(x) == 0 and g(x) != 0 at
                # the same place
                sol_base = _solve(lhs.base, sym, **flags)
                return [s for s in sol_base if lhs.exp.subs(sym, s) != 0]

            # a ** g(x) == b
            if not lhs.base.has(sym):
                if lhs.base == 0:
                    return _solve(lhs.exp, sym, **flags) if rhs != 0 else []

                # Gets most solutions...
                if lhs.base == rhs.as_base_exp()[0]:
                    # handles case when bases are equal
                    sol = _solve(lhs.exp - rhs.as_base_exp()[1], sym, **flags)
                else:
                    # handles cases when bases are not equal and exp
                    # may or may not be equal
                    sol = _solve(exp(log(lhs.base)*lhs.exp)-exp(log(rhs)), sym, **flags)

                # Check for duplicate solutions
                def equal(expr1, expr2):
                    _ = Dummy()
                    eq = checksol(expr1 - _, _, expr2)
                    if eq is None:
                        if nsimplify(expr1) != nsimplify(expr2):
                            return False
                        # they might be coincidentally the same
                        # so check more rigorously
                        eq = expr1.equals(expr2)
                    return eq

                # Guess a rational exponent
                e_rat = nsimplify(log(abs(rhs))/log(abs(lhs.base)))
                e_rat = simplify(posify(e_rat)[0])
                n, d = fraction(e_rat)
                if expand(lhs.base**n - rhs**d) == 0:
                    sol = [s for s in sol if not equal(lhs.exp.subs(sym, s), e_rat)]
                    sol.extend(_solve(lhs.exp - e_rat, sym, **flags))

                return list(ordered(set(sol)))

            # f(x) ** g(x) == c
            else:
                sol = []
                logform = lhs.exp*log(lhs.base) - log(rhs)
                if logform != lhs - rhs:
                    try:
                        sol.extend(_solve(logform, sym, **flags))
                    except NotImplementedError:
                        pass

                # Collect possible solutions and check with substitution later.
                check = []
                if rhs == 1:
                    # f(x) ** g(x) = 1 -- g(x)=0 or f(x)=+-1
                    check.extend(_solve(lhs.exp, sym, **flags))
                    check.extend(_solve(lhs.base - 1, sym, **flags))
                    check.extend(_solve(lhs.base + 1, sym, **flags))
                elif rhs.is_Rational:
                    for d in (i for i in divisors(abs(rhs.p)) if i != 1):
                        e, t = integer_log(rhs.p, d)
                        if not t:
                            continue  # rhs.p != d**b
                        for s in divisors(abs(rhs.q)):
                            if s**e== rhs.q:
                                r = Rational(d, s)
                                check.extend(_solve(lhs.base - r, sym, **flags))
                                check.extend(_solve(lhs.base + r, sym, **flags))
                                check.extend(_solve(lhs.exp - e, sym, **flags))
                elif rhs.is_irrational:
                    b_l, e_l = lhs.base.as_base_exp()
                    n, d = (e_l*lhs.exp).as_numer_denom()
                    b, e = sqrtdenest(rhs).as_base_exp()
                    check = [sqrtdenest(i) for i in (_solve(lhs.base - b, sym, **flags))]
                    check.extend([sqrtdenest(i) for i in (_solve(lhs.exp - e, sym, **flags))])
                    if e_l*d != 1:
                        check.extend(_solve(b_l**n - rhs**(e_l*d), sym, **flags))
                for s in check:
                    ok = checksol(eq, sym, s)
                    if ok is None:
                        ok = eq.subs(sym, s).equals(0)
                    if ok:
                        sol.append(s)
                return list(ordered(set(sol)))

        elif lhs.is_Function and len(lhs.args) == 1:
            if lhs.func in multi_inverses:
                # sin(x) = 1/3 -> x - asin(1/3) & x - (pi - asin(1/3))
                soln = []
                for i in multi_inverses[lhs.func](rhs):
                    soln.extend(_solve(lhs.args[0] - i, sym, **flags))
                return list(ordered(soln))
            elif lhs.func == LambertW:
                return _solve(lhs.args[0] - rhs*exp(rhs), sym, **flags)

        rewrite = lhs.rewrite(exp)
        if rewrite != lhs:
            return _solve(rewrite - rhs, sym, **flags)
    except NotImplementedError:
        pass
  # ... other code
```
### 4 - sympy/solvers/solvers.py:

Start line: 380, End line: 832

```python
def solve(f, *symbols, **flags):
    r"""
    Algebraically solves equations and systems of equations.

    Explanation
    ===========

    Currently supported:
        - polynomial
        - transcendental
        - piecewise combinations of the above
        - systems of linear and polynomial equations
        - systems containing relational expressions

    Examples
    ========

    The output varies according to the input and can be seen by example:

        >>> from sympy import solve, Poly, Eq, Function, exp
        >>> from sympy.abc import x, y, z, a, b
        >>> f = Function('f')

    Boolean or univariate Relational:

        >>> solve(x < 3)
        (-oo < x) & (x < 3)


    To always get a list of solution mappings, use flag dict=True:

        >>> solve(x - 3, dict=True)
        [{x: 3}]
        >>> sol = solve([x - 3, y - 1], dict=True)
        >>> sol
        [{x: 3, y: 1}]
        >>> sol[0][x]
        3
        >>> sol[0][y]
        1


    To get a list of *symbols* and set of solution(s) use flag set=True:

        >>> solve([x**2 - 3, y - 1], set=True)
        ([x, y], {(-sqrt(3), 1), (sqrt(3), 1)})


    Single expression and single symbol that is in the expression:

        >>> solve(x - y, x)
        [y]
        >>> solve(x - 3, x)
        [3]
        >>> solve(Eq(x, 3), x)
        [3]
        >>> solve(Poly(x - 3), x)
        [3]
        >>> solve(x**2 - y**2, x, set=True)
        ([x], {(-y,), (y,)})
        >>> solve(x**4 - 1, x, set=True)
        ([x], {(-1,), (1,), (-I,), (I,)})

    Single expression with no symbol that is in the expression:

        >>> solve(3, x)
        []
        >>> solve(x - 3, y)
        []

    Single expression with no symbol given. In this case, all free *symbols*
    will be selected as potential *symbols* to solve for. If the equation is
    univariate then a list of solutions is returned; otherwise - as is the case
    when *symbols* are given as an iterable of length greater than 1 - a list of
    mappings will be returned:

        >>> solve(x - 3)
        [3]
        >>> solve(x**2 - y**2)
        [{x: -y}, {x: y}]
        >>> solve(z**2*x**2 - z**2*y**2)
        [{x: -y}, {x: y}, {z: 0}]
        >>> solve(z**2*x - z**2*y**2)
        [{x: y**2}, {z: 0}]

    When an object other than a Symbol is given as a symbol, it is
    isolated algebraically and an implicit solution may be obtained.
    This is mostly provided as a convenience to save you from replacing
    the object with a Symbol and solving for that Symbol. It will only
    work if the specified object can be replaced with a Symbol using the
    subs method:

    >>> solve(f(x) - x, f(x))
    [x]
    >>> solve(f(x).diff(x) - f(x) - x, f(x).diff(x))
    [x + f(x)]
    >>> solve(f(x).diff(x) - f(x) - x, f(x))
    [-x + Derivative(f(x), x)]
    >>> solve(x + exp(x)**2, exp(x), set=True)
    ([exp(x)], {(-sqrt(-x),), (sqrt(-x),)})

    >>> from sympy import Indexed, IndexedBase, Tuple, sqrt
    >>> A = IndexedBase('A')
    >>> eqs = Tuple(A[1] + A[2] - 3, A[1] - A[2] + 1)
    >>> solve(eqs, eqs.atoms(Indexed))
    {A[1]: 1, A[2]: 2}

        * To solve for a symbol implicitly, use implicit=True:

            >>> solve(x + exp(x), x)
            [-LambertW(1)]
            >>> solve(x + exp(x), x, implicit=True)
            [-exp(x)]

        * It is possible to solve for anything that can be targeted with
          subs:

            >>> solve(x + 2 + sqrt(3), x + 2)
            [-sqrt(3)]
            >>> solve((x + 2 + sqrt(3), x + 4 + y), y, x + 2)
            {y: -2 + sqrt(3), x + 2: -sqrt(3)}

        * Nothing heroic is done in this implicit solving so you may end up
          with a symbol still in the solution:

            >>> eqs = (x*y + 3*y + sqrt(3), x + 4 + y)
            >>> solve(eqs, y, x + 2)
            {y: -sqrt(3)/(x + 3), x + 2: -2*x/(x + 3) - 6/(x + 3) + sqrt(3)/(x + 3)}
            >>> solve(eqs, y*x, x)
            {x: -y - 4, x*y: -3*y - sqrt(3)}

        * If you attempt to solve for a number remember that the number
          you have obtained does not necessarily mean that the value is
          equivalent to the expression obtained:

            >>> solve(sqrt(2) - 1, 1)
            [sqrt(2)]
            >>> solve(x - y + 1, 1)  # /!\ -1 is targeted, too
            [x/(y - 1)]
            >>> [_.subs(z, -1) for _ in solve((x - y + 1).subs(-1, z), 1)]
            [-x + y]

        * To solve for a function within a derivative, use ``dsolve``.

    Single expression and more than one symbol:

        * When there is a linear solution:

            >>> solve(x - y**2, x, y)
            [(y**2, y)]
            >>> solve(x**2 - y, x, y)
            [(x, x**2)]
            >>> solve(x**2 - y, x, y, dict=True)
            [{y: x**2}]

        * When undetermined coefficients are identified:

            * That are linear:

                >>> solve((a + b)*x - b + 2, a, b)
                {a: -2, b: 2}

            * That are nonlinear:

                >>> solve((a + b)*x - b**2 + 2, a, b, set=True)
                ([a, b], {(-sqrt(2), sqrt(2)), (sqrt(2), -sqrt(2))})

        * If there is no linear solution, then the first successful
          attempt for a nonlinear solution will be returned:

            >>> solve(x**2 - y**2, x, y, dict=True)
            [{x: -y}, {x: y}]
            >>> solve(x**2 - y**2/exp(x), x, y, dict=True)
            [{x: 2*LambertW(-y/2)}, {x: 2*LambertW(y/2)}]
            >>> solve(x**2 - y**2/exp(x), y, x)
            [(-x*sqrt(exp(x)), x), (x*sqrt(exp(x)), x)]

    Iterable of one or more of the above:

        * Involving relationals or bools:

            >>> solve([x < 3, x - 2])
            Eq(x, 2)
            >>> solve([x > 3, x - 2])
            False

        * When the system is linear:

            * With a solution:

                >>> solve([x - 3], x)
                {x: 3}
                >>> solve((x + 5*y - 2, -3*x + 6*y - 15), x, y)
                {x: -3, y: 1}
                >>> solve((x + 5*y - 2, -3*x + 6*y - 15), x, y, z)
                {x: -3, y: 1}
                >>> solve((x + 5*y - 2, -3*x + 6*y - z), z, x, y)
                {x: 2 - 5*y, z: 21*y - 6}

            * Without a solution:

                >>> solve([x + 3, x - 3])
                []

        * When the system is not linear:

            >>> solve([x**2 + y -2, y**2 - 4], x, y, set=True)
            ([x, y], {(-2, -2), (0, 2), (2, -2)})

        * If no *symbols* are given, all free *symbols* will be selected and a
          list of mappings returned:

            >>> solve([x - 2, x**2 + y])
            [{x: 2, y: -4}]
            >>> solve([x - 2, x**2 + f(x)], {f(x), x})
            [{x: 2, f(x): -4}]

        * If any equation does not depend on the symbol(s) given, it will be
          eliminated from the equation set and an answer may be given
          implicitly in terms of variables that were not of interest:

            >>> solve([x - y, y - 3], x)
            {x: y}

    **Additional Examples**

    ``solve()`` with check=True (default) will run through the symbol tags to
    elimate unwanted solutions. If no assumptions are included, all possible
    solutions will be returned:

        >>> from sympy import Symbol, solve
        >>> x = Symbol("x")
        >>> solve(x**2 - 1)
        [-1, 1]

    By using the positive tag, only one solution will be returned:

        >>> pos = Symbol("pos", positive=True)
        >>> solve(pos**2 - 1)
        [1]

    Assumptions are not checked when ``solve()`` input involves
    relationals or bools.

    When the solutions are checked, those that make any denominator zero
    are automatically excluded. If you do not want to exclude such solutions,
    then use the check=False option:

        >>> from sympy import sin, limit
        >>> solve(sin(x)/x)  # 0 is excluded
        [pi]

    If check=False, then a solution to the numerator being zero is found: x = 0.
    In this case, this is a spurious solution since $\sin(x)/x$ has the well
    known limit (without dicontinuity) of 1 at x = 0:

        >>> solve(sin(x)/x, check=False)
        [0, pi]

    In the following case, however, the limit exists and is equal to the
    value of x = 0 that is excluded when check=True:

        >>> eq = x**2*(1/x - z**2/x)
        >>> solve(eq, x)
        []
        >>> solve(eq, x, check=False)
        [0]
        >>> limit(eq, x, 0, '-')
        0
        >>> limit(eq, x, 0, '+')
        0

    **Disabling High-Order Explicit Solutions**

    When solving polynomial expressions, you might not want explicit solutions
    (which can be quite long). If the expression is univariate, ``CRootOf``
    instances will be returned instead:

        >>> solve(x**3 - x + 1)
        [-1/((-1/2 - sqrt(3)*I/2)*(3*sqrt(69)/2 + 27/2)**(1/3)) - (-1/2 -
        sqrt(3)*I/2)*(3*sqrt(69)/2 + 27/2)**(1/3)/3, -(-1/2 +
        sqrt(3)*I/2)*(3*sqrt(69)/2 + 27/2)**(1/3)/3 - 1/((-1/2 +
        sqrt(3)*I/2)*(3*sqrt(69)/2 + 27/2)**(1/3)), -(3*sqrt(69)/2 +
        27/2)**(1/3)/3 - 1/(3*sqrt(69)/2 + 27/2)**(1/3)]
        >>> solve(x**3 - x + 1, cubics=False)
        [CRootOf(x**3 - x + 1, 0),
         CRootOf(x**3 - x + 1, 1),
         CRootOf(x**3 - x + 1, 2)]

    If the expression is multivariate, no solution might be returned:

        >>> solve(x**3 - x + a, x, cubics=False)
        []

    Sometimes solutions will be obtained even when a flag is False because the
    expression could be factored. In the following example, the equation can
    be factored as the product of a linear and a quadratic factor so explicit
    solutions (which did not require solving a cubic expression) are obtained:

        >>> eq = x**3 + 3*x**2 + x - 1
        >>> solve(eq, cubics=False)
        [-1, -1 + sqrt(2), -sqrt(2) - 1]

    **Solving Equations Involving Radicals**

    Because of SymPy's use of the principle root, some solutions
    to radical equations will be missed unless check=False:

        >>> from sympy import root
        >>> eq = root(x**3 - 3*x**2, 3) + 1 - x
        >>> solve(eq)
        []
        >>> solve(eq, check=False)
        [1/3]

    In the above example, there is only a single solution to the
    equation. Other expressions will yield spurious roots which
    must be checked manually; roots which give a negative argument
    to odd-powered radicals will also need special checking:

        >>> from sympy import real_root, S
        >>> eq = root(x, 3) - root(x, 5) + S(1)/7
        >>> solve(eq)  # this gives 2 solutions but misses a 3rd
        [CRootOf(7*x**5 - 7*x**3 + 1, 1)**15,
        CRootOf(7*x**5 - 7*x**3 + 1, 2)**15]
        >>> sol = solve(eq, check=False)
        >>> [abs(eq.subs(x,i).n(2)) for i in sol]
        [0.48, 0.e-110, 0.e-110, 0.052, 0.052]

    The first solution is negative so ``real_root`` must be used to see that it
    satisfies the expression:

        >>> abs(real_root(eq.subs(x, sol[0])).n(2))
        0.e-110

    If the roots of the equation are not real then more care will be
    necessary to find the roots, especially for higher order equations.
    Consider the following expression:

        >>> expr = root(x, 3) - root(x, 5)

    We will construct a known value for this expression at x = 3 by selecting
    the 1-th root for each radical:

        >>> expr1 = root(x, 3, 1) - root(x, 5, 1)
        >>> v = expr1.subs(x, -3)

    The ``solve`` function is unable to find any exact roots to this equation:

        >>> eq = Eq(expr, v); eq1 = Eq(expr1, v)
        >>> solve(eq, check=False), solve(eq1, check=False)
        ([], [])

    The function ``unrad``, however, can be used to get a form of the equation
    for which numerical roots can be found:

        >>> from sympy.solvers.solvers import unrad
        >>> from sympy import nroots
        >>> e, (p, cov) = unrad(eq)
        >>> pvals = nroots(e)
        >>> inversion = solve(cov, x)[0]
        >>> xvals = [inversion.subs(p, i) for i in pvals]

    Although ``eq`` or ``eq1`` could have been used to find ``xvals``, the
    solution can only be verified with ``expr1``:

        >>> z = expr - v
        >>> [xi.n(chop=1e-9) for xi in xvals if abs(z.subs(x, xi).n()) < 1e-9]
        []
        >>> z1 = expr1 - v
        >>> [xi.n(chop=1e-9) for xi in xvals if abs(z1.subs(x, xi).n()) < 1e-9]
        [-3.0]

    Parameters
    ==========

    f :
        - a single Expr or Poly that must be zero
        - an Equality
        - a Relational expression
        - a Boolean
        - iterable of one or more of the above

    symbols : (object(s) to solve for) specified as
        - none given (other non-numeric objects will be used)
        - single symbol
        - denested list of symbols
          (e.g., ``solve(f, x, y)``)
        - ordered iterable of symbols
          (e.g., ``solve(f, [x, y])``)

    flags :
        dict=True (default is False)
            Return list (perhaps empty) of solution mappings.
        set=True (default is False)
            Return list of symbols and set of tuple(s) of solution(s).
        exclude=[] (default)
            Do not try to solve for any of the free symbols in exclude;
            if expressions are given, the free symbols in them will
            be extracted automatically.
        check=True (default)
            If False, do not do any testing of solutions. This can be
            useful if you want to include solutions that make any
            denominator zero.
        numerical=True (default)
            Do a fast numerical check if *f* has only one symbol.
        minimal=True (default is False)
            A very fast, minimal testing.
        warn=True (default is False)
            Show a warning if ``checksol()`` could not conclude.
        simplify=True (default)
            Simplify all but polynomials of order 3 or greater before
            returning them and (if check is not False) use the
            general simplify function on the solutions and the
            expression obtained when they are substituted into the
            function which should be zero.
        force=True (default is False)
            Make positive all symbols without assumptions regarding sign.
        rational=True (default)
            Recast Floats as Rational; if this option is not used, the
            system containing Floats may fail to solve because of issues
            with polys. If rational=None, Floats will be recast as
            rationals but the answer will be recast as Floats. If the
            flag is False then nothing will be done to the Floats.
        manual=True (default is False)
            Do not use the polys/matrix method to solve a system of
            equations, solve them one at a time as you might "manually."
        implicit=True (default is False)
            Allows ``solve`` to return a solution for a pattern in terms of
            other functions that contain that pattern; this is only
            needed if the pattern is inside of some invertible function
            like cos, exp, ect.
        particular=True (default is False)
            Instructs ``solve`` to try to find a particular solution to a linear
            system with as many zeros as possible; this is very expensive.
        quick=True (default is False)
            When using particular=True, use a fast heuristic to find a
            solution with many zeros (instead of using the very slow method
            guaranteed to find the largest number of zeros possible).
        cubics=True (default)
            Return explicit solutions when cubic expressions are encountered.
        quartics=True (default)
            Return explicit solutions when quartic expressions are encountered.
        quintics=True (default)
            Return explicit solutions (if possible) when quintic expressions
            are encountered.

    See Also
    ========

    rsolve: For solving recurrence relationships
    dsolve: For solving differential equations

    """
    # ... other code
```
### 5 - sympy/physics/mechanics/functions.py:

Start line: 658, End line: 678

```python
def _smart_subs(expr, sub_dict):
    # ... other code

    def _recurser(expr, sub_dict):
        # Decompose the expression into num, den
        num, den = _fraction_decomp(expr)
        if den != 1:
            # If there is a non trivial denominator, we need to handle it
            denom_subbed = _recurser(den, sub_dict)
            if denom_subbed.evalf() == 0:
                # If denom is 0 after this, attempt to simplify the bad expr
                expr = simplify(expr)
            else:
                # Expression won't result in nan, find numerator
                num_subbed = _recurser(num, sub_dict)
                return num_subbed / denom_subbed
        # We have to crawl the tree manually, because `expr` may have been
        # modified in the simplify step. First, perform subs as normal:
        val = _sub_func(expr, sub_dict)
        if val is not None:
            return val
        new_args = (_recurser(arg, sub_dict) for arg in expr.args)
        return expr.func(*new_args)
    return _recurser(expr, sub_dict)
```
### 6 - sympy/physics/mechanics/functions.py:

Start line: 644, End line: 656

```python
def _smart_subs(expr, sub_dict):
    """Performs subs, checking for conditions that may result in `nan` or
    `oo`, and attempts to simplify them out.

    The expression tree is traversed twice, and the following steps are
    performed on each expression node:
    - First traverse:
        Replace all `tan` with `sin/cos`.
    - Second traverse:
        If node is a fraction, check if the denominator evaluates to 0.
        If so, attempt to simplify it out. Then if node is in sub_dict,
        sub in the corresponding value."""
    expr = _crawl(expr, _tan_repl_func)
    # ... other code
```
### 7 - sympy/solvers/bivariate.py:

Start line: 281, End line: 362

```python
def _solve_lambert(f, symbol, gens):
    # ... other code

    nrhs, lhs = f.as_independent(symbol, as_Add=True)
    rhs = -nrhs

    lamcheck = [tmp for tmp in gens
                if (tmp.func in [exp, log] or
                (tmp.is_Pow and symbol in tmp.exp.free_symbols))]
    if not lamcheck:
        raise NotImplementedError()

    if lhs.is_Add or lhs.is_Mul:
        # replacing all even_degrees of symbol with dummy variable t
        # since these will need special handling; non-Add/Mul do not
        # need this handling
        t = Dummy('t', **symbol.assumptions0)
        lhs = lhs.replace(
            lambda i:  # find symbol**even
                i.is_Pow and i.base == symbol and i.exp.is_even,
            lambda i:  # replace t**even
                t**i.exp)

        if lhs.is_Add and lhs.has(t):
            t_indep = lhs.subs(t, 0)
            t_term = lhs - t_indep
            _rhs = rhs - t_indep
            if not t_term.is_Add and _rhs and not (
                    t_term.has(S.ComplexInfinity, S.NaN)):
                eq = expand_log(log(t_term) - log(_rhs))
                return _solve_even_degree_expr(eq, t, symbol)
        elif lhs.is_Mul and rhs:
            # this needs to happen whether t is present or not
            lhs = expand_log(log(lhs), force=True)
            rhs = log(rhs)
            if lhs.has(t) and lhs.is_Add:
                # it expanded from Mul to Add
                eq = lhs - rhs
                return _solve_even_degree_expr(eq, t, symbol)

        # restore symbol in lhs
        lhs = lhs.xreplace({t: symbol})

    lhs = powsimp(factor(lhs, deep=True))

    # make sure we have inverted as completely as possible
    r = Dummy()
    i, lhs = _invert(lhs - r, symbol)
    rhs = i.xreplace({r: rhs})

    # For the first forms:
    #
    # 1a1) B**B = R will arrive here as B*log(B) = log(R)
    #      lhs is Mul so take log of both sides:
    #        log(B) + log(log(B)) = log(log(R))
    # 1a2) B*(b*log(B) + c)**a = R will arrive unchanged so
    #      lhs is Mul, so take log of both sides:
    #        log(B) + a*log(b*log(B) + c) = log(R)
    # 1b) d*log(a*B + b) + c*B = R will arrive unchanged so
    #      lhs is Add, so isolate c*B and expand log of both sides:
    #        log(c) + log(B) = log(R - d*log(a*B + b))

    soln = []
    if not soln:
        mainlog = _mostfunc(lhs, log, symbol)
        if mainlog:
            if lhs.is_Mul and rhs != 0:
                soln = _lambert(log(lhs) - log(rhs), symbol)
            elif lhs.is_Add:
                other = lhs.subs(mainlog, 0)
                if other and not other.is_Add and [
                        tmp for tmp in other.atoms(Pow)
                        if symbol in tmp.free_symbols]:
                    if not rhs:
                        diff = log(other) - log(other - lhs)
                    else:
                        diff = log(lhs - other) - log(rhs - other)
                    soln = _lambert(expand_log(diff), symbol)
                else:
                    #it's ready to go
                    soln = _lambert(lhs - rhs, symbol)

    # For the next forms,
    #
    #     collect on main exp
    # ... other code
```
### 8 - sympy/concrete/summations.py:

Start line: 1, End line: 29

```python
from sympy.calculus.singularities import is_decreasing
from sympy.calculus.util import AccumulationBounds
from sympy.concrete.expr_with_limits import AddWithLimits
from sympy.concrete.expr_with_intlimits import ExprWithIntLimits
from sympy.concrete.gosper import gosper_sum
from sympy.core.add import Add
from sympy.core.function import Derivative
from sympy.core.mul import Mul
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Wild, Symbol
from sympy.functions.special.zeta_functions import zeta
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import cot, csc
from sympy.logic.boolalg import And
from sympy.polys import apart, together
from sympy.polys.polyerrors import PolynomialError, PolificationFailed
from sympy.polys.polytools import parallel_poly_from_expr
from sympy.series.limitseq import limit_seq
from sympy.series.order import O
from sympy.series.residues import residue
from sympy.sets.sets import FiniteSet
from sympy.simplify import denom
from sympy.simplify.combsimp import combsimp
from sympy.simplify.powsimp import powsimp
from sympy.solvers import solve
from sympy.solvers.solveset import solveset
from sympy.utilities.iterables import sift
import itertools
```
### 9 - sympy/integrals/rubi/utility_function.py:

Start line: 6501, End line: 7009

```python
@doctest_depends_on(modules=('matchpy',))
def _TrigSimplifyAux():
    # ... other code
    rule13 = ReplacementRule(pattern13, lambda u, v, b, a : Mul(u, Add(Mul(S(1), Pow(a, S(-1))), Mul(S(-1), Mul(Sin(v), Pow(b, S(-1)))))))
    replacer.add(rule13)

    pattern14 = Pattern(UtilityOperator(Mul(WC('u', S(1)), Pow(tan(v_), WC('n', S(1))), Pow(Add(a_, Mul(WC('b', S(1)), Pow(tan(v_), WC('n', S(1))))), S(-1)))), CustomConstraint(lambda n: PositiveIntegerQ(n)), CustomConstraint(lambda a: NonsumQ(a)))
    rule14 = ReplacementRule(pattern14, lambda n, a, u, v, b : Mul(u, Pow(Add(b, Mul(a, Pow(Cot(v), n))), S(-1))))
    replacer.add(rule14)

    pattern15 = Pattern(UtilityOperator(Mul(Pow(cot(v_), WC('n', S(1))), WC('u', S(1)), Pow(Add(Mul(Pow(cot(v_), WC('n', S(1))), WC('b', S(1))), a_), S(-1)))), CustomConstraint(lambda n: PositiveIntegerQ(n)), CustomConstraint(lambda a: NonsumQ(a)))
    rule15 = ReplacementRule(pattern15, lambda n, a, u, v, b : Mul(u, Pow(Add(b, Mul(a, Pow(Tan(v), n))), S(-1))))
    replacer.add(rule15)

    pattern16 = Pattern(UtilityOperator(Mul(WC('u', S(1)), Pow(sec(v_), WC('n', S(1))), Pow(Add(a_, Mul(WC('b', S(1)), Pow(sec(v_), WC('n', S(1))))), S(-1)))), CustomConstraint(lambda n: PositiveIntegerQ(n)), CustomConstraint(lambda a: NonsumQ(a)))
    rule16 = ReplacementRule(pattern16, lambda n, a, u, v, b : Mul(u, Pow(Add(b, Mul(a, Pow(Cos(v), n))), S(-1))))
    replacer.add(rule16)

    pattern17 = Pattern(UtilityOperator(Mul(Pow(csc(v_), WC('n', S(1))), WC('u', S(1)), Pow(Add(Mul(Pow(csc(v_), WC('n', S(1))), WC('b', S(1))), a_), S(-1)))), CustomConstraint(lambda n: PositiveIntegerQ(n)), CustomConstraint(lambda a: NonsumQ(a)))
    rule17 = ReplacementRule(pattern17, lambda n, a, u, v, b : Mul(u, Pow(Add(b, Mul(a, Pow(Sin(v), n))), S(-1))))
    replacer.add(rule17)

    pattern18 = Pattern(UtilityOperator(Mul(WC('u', S(1)), Pow(Add(a_, Mul(WC('b', S(1)), Pow(sec(v_), WC('n', S(1))))), S(-1)), Pow(tan(v_), WC('n', S(1))))), CustomConstraint(lambda n: PositiveIntegerQ(n)), CustomConstraint(lambda a: NonsumQ(a)))
    rule18 = ReplacementRule(pattern18, lambda n, a, u, v, b : Mul(u, Mul(Pow(Sin(v), n), Pow(Add(b, Mul(a, Pow(Cos(v), n))), S(-1)))))
    replacer.add(rule18)

    pattern19 = Pattern(UtilityOperator(Mul(Pow(cot(v_), WC('n', S(1))), WC('u', S(1)), Pow(Add(Mul(Pow(csc(v_), WC('n', S(1))), WC('b', S(1))), a_), S(-1)))), CustomConstraint(lambda n: PositiveIntegerQ(n)), CustomConstraint(lambda a: NonsumQ(a)))
    rule19 = ReplacementRule(pattern19, lambda n, a, u, v, b : Mul(u, Mul(Pow(Cos(v), n), Pow(Add(b, Mul(a, Pow(Sin(v), n))), S(-1)))))
    replacer.add(rule19)

    pattern20 = Pattern(UtilityOperator(Mul(WC('u', S(1)), Pow(Add(Mul(WC('a', S(1)), Pow(sec(v_), WC('n', S(1)))), Mul(WC('b', S(1)), Pow(tan(v_), WC('n', S(1))))), WC('p', S(1))))), CustomConstraint(lambda n, p: IntegersQ(n, p)))
    rule20 = ReplacementRule(pattern20, lambda n, a, p, u, v, b : Mul(u, Pow(Sec(v), Mul(n, p)), Pow(Add(a, Mul(b, Pow(Sin(v), n))), p)))
    replacer.add(rule20)

    pattern21 = Pattern(UtilityOperator(Mul(Pow(Add(Mul(Pow(csc(v_), WC('n', S(1))), WC('a', S(1))), Mul(Pow(cot(v_), WC('n', S(1))), WC('b', S(1)))), WC('p', S(1))), WC('u', S(1)))), CustomConstraint(lambda n, p: IntegersQ(n, p)))
    rule21 = ReplacementRule(pattern21, lambda n, a, p, u, v, b : Mul(u, Pow(Csc(v), Mul(n, p)), Pow(Add(a, Mul(b, Pow(Cos(v), n))), p)))
    replacer.add(rule21)

    pattern22 = Pattern(UtilityOperator(Mul(WC('u', S(1)), Pow(Add(Mul(WC('b', S(1)), Pow(sin(v_), WC('n', S(1)))), Mul(WC('a', S(1)), Pow(tan(v_), WC('n', S(1))))), WC('p', S(1))))), CustomConstraint(lambda n, p: IntegersQ(n, p)))
    rule22 = ReplacementRule(pattern22, lambda n, a, p, u, v, b : Mul(u, Pow(Tan(v), Mul(n, p)), Pow(Add(a, Mul(b, Pow(Cos(v), n))), p)))
    replacer.add(rule22)

    pattern23 = Pattern(UtilityOperator(Mul(Pow(Add(Mul(Pow(cot(v_), WC('n', S(1))), WC('a', S(1))), Mul(Pow(cos(v_), WC('n', S(1))), WC('b', S(1)))), WC('p', S(1))), WC('u', S(1)))), CustomConstraint(lambda n, p: IntegersQ(n, p)))
    rule23 = ReplacementRule(pattern23, lambda n, a, p, u, v, b : Mul(u, Pow(Cot(v), Mul(n, p)), Pow(Add(a, Mul(b, Pow(Sin(v), n))), p)))
    replacer.add(rule23)

    pattern24 = Pattern(UtilityOperator(Mul(Pow(cos(v_), WC('m', S(1))), WC('u', S(1)), Pow(Add(WC('a', S(0)), Mul(WC('c', S(1)), Pow(sec(v_), WC('n', S(1)))), Mul(WC('b', S(1)), Pow(tan(v_), WC('n', S(1))))), WC('p', S(1))))), CustomConstraint(lambda n, p, m: IntegersQ(m, n, p)))
    rule24 = ReplacementRule(pattern24, lambda n, a, c, p, m, u, v, b : Mul(u, Pow(Cos(v), Add(m, Mul(S(-1), Mul(n, p)))), Pow(Add(c, Mul(b, Pow(Sin(v), n)), Mul(a, Pow(Cos(v), n))), p)))
    replacer.add(rule24)

    pattern25 = Pattern(UtilityOperator(Mul(WC('u', S(1)), Pow(sec(v_), WC('m', S(1))), Pow(Add(WC('a', S(0)), Mul(WC('c', S(1)), Pow(sec(v_), WC('n', S(1)))), Mul(WC('b', S(1)), Pow(tan(v_), WC('n', S(1))))), WC('p', S(1))))), CustomConstraint(lambda n, p, m: IntegersQ(m, n, p)))
    rule25 = ReplacementRule(pattern25, lambda n, a, c, p, m, u, v, b : Mul(u, Pow(Sec(v), Add(m, Mul(n, p))), Pow(Add(c, Mul(b, Pow(Sin(v), n)), Mul(a, Pow(Cos(v), n))), p)))
    replacer.add(rule25)

    pattern26 = Pattern(UtilityOperator(Mul(Pow(Add(WC('a', S(0)), Mul(Pow(cot(v_), WC('n', S(1))), WC('b', S(1))), Mul(Pow(csc(v_), WC('n', S(1))), WC('c', S(1)))), WC('p', S(1))), WC('u', S(1)), Pow(sin(v_), WC('m', S(1))))), CustomConstraint(lambda n, p, m: IntegersQ(m, n, p)))
    rule26 = ReplacementRule(pattern26, lambda n, a, c, p, m, u, v, b : Mul(u, Pow(Sin(v), Add(m, Mul(S(-1), Mul(n, p)))), Pow(Add(c, Mul(b, Pow(Cos(v), n)), Mul(a, Pow(Sin(v), n))), p)))
    replacer.add(rule26)

    pattern27 = Pattern(UtilityOperator(Mul(Pow(csc(v_), WC('m', S(1))), Pow(Add(WC('a', S(0)), Mul(Pow(cot(v_), WC('n', S(1))), WC('b', S(1))), Mul(Pow(csc(v_), WC('n', S(1))), WC('c', S(1)))), WC('p', S(1))), WC('u', S(1)))), CustomConstraint(lambda n, p, m: IntegersQ(m, n, p)))
    rule27 = ReplacementRule(pattern27, lambda n, a, c, p, m, u, v, b : Mul(u, Pow(Csc(v), Add(m, Mul(n, p))), Pow(Add(c, Mul(b, Pow(Cos(v), n)), Mul(a, Pow(Sin(v), n))), p)))
    replacer.add(rule27)

    pattern28 = Pattern(UtilityOperator(Mul(WC('u', S(1)), Pow(Add(Mul(Pow(csc(v_), WC('m', S(1))), WC('a', S(1))), Mul(WC('b', S(1)), Pow(sin(v_), WC('n', S(1))))), WC('p', S(1))))), CustomConstraint(lambda n, m: IntegersQ(m, n)))
    rule28 = ReplacementRule(pattern28, lambda n, a, p, m, u, v, b : If(And(ZeroQ(Add(m, n, S(-2))), ZeroQ(Add(a, b))), Mul(u, Pow(Mul(a, Mul(Pow(Cos(v), S('2')), Pow(Pow(Sin(v), m), S(-1)))), p)), Mul(u, Pow(Mul(Add(a, Mul(b, Pow(Sin(v), Add(m, n)))), Pow(Pow(Sin(v), m), S(-1))), p))))
    replacer.add(rule28)

    pattern29 = Pattern(UtilityOperator(Mul(WC('u', S(1)), Pow(Add(Mul(Pow(cos(v_), WC('n', S(1))), WC('b', S(1))), Mul(WC('a', S(1)), Pow(sec(v_), WC('m', S(1))))), WC('p', S(1))))), CustomConstraint(lambda n, m: IntegersQ(m, n)))
    rule29 = ReplacementRule(pattern29, lambda n, a, p, m, u, v, b : If(And(ZeroQ(Add(m, n, S(-2))), ZeroQ(Add(a, b))), Mul(u, Pow(Mul(a, Mul(Pow(Sin(v), S('2')), Pow(Pow(Cos(v), m), S(-1)))), p)), Mul(u, Pow(Mul(Add(a, Mul(b, Pow(Cos(v), Add(m, n)))), Pow(Pow(Cos(v), m), S(-1))), p))))
    replacer.add(rule29)

    pattern30 = Pattern(UtilityOperator(u_))
    rule30 = ReplacementRule(pattern30, lambda u : u)
    replacer.add(rule30)

    return replacer

@doctest_depends_on(modules=('matchpy',))
def TrigSimplifyAux(expr):
    return TrigSimplifyAux_replacer.replace(UtilityOperator(expr))

def Cancel(expr):
    return cancel(expr)

class Util_Part(Function):
    def doit(self):
        i = Simplify(self.args[0])
        if len(self.args) > 2 :
            lst = list(self.args[1:])
        else:
            lst = self.args[1]
        if isinstance(i, (int, Integer)):
            if isinstance(lst, list):
                return lst[i - 1]
            elif AtomQ(lst):
                return lst
            return lst.args[i - 1]
        else:
            return self

def Part(lst, i): #see i = -1
    if isinstance(lst, list):
        return Util_Part(i, *lst).doit()
    return Util_Part(i, lst).doit()

def PolyLog(n, p, z=None):
    return polylog(n, p)

def D(f, x):
    try:
        return f.diff(x)
    except ValueError:
        return Function('D')(f, x)

def IntegralFreeQ(u):
    return FreeQ(u, Integral)

def Dist(u, v, x):
    #Dist(u,v) returns the sum of u times each term of v, provided v is free of Int
    u = replace_pow_exp(u) # to replace back to sympy's exp
    v = replace_pow_exp(v)
    w = Simp(u*x**2, x)/x**2
    if u == 1:
        return v
    elif u == 0:
        return 0
    elif NumericFactor(u) < 0 and NumericFactor(-u) > 0:
        return -Dist(-u, v, x)
    elif SumQ(v):
        return Add(*[Dist(u, i, x) for i in v.args])
    elif IntegralFreeQ(v):
        return Simp(u*v, x)
    elif w != u and FreeQ(w, x) and w == Simp(w, x) and w == Simp(w*x**2, x)/x**2:
        return Dist(w, v, x)
    else:
        return Simp(u*v, x)

def PureFunctionOfCothQ(u, v, x):
    # If u is a pure function of Coth[v], PureFunctionOfCothQ[u,v,x] returns True;
    if AtomQ(u):
        return u != x
    elif CalculusQ(u):
        return False
    elif HyperbolicQ(u) and ZeroQ(u.args[0] - v):
        return CothQ(u)
    return all(PureFunctionOfCothQ(i, v, x) for i in u.args)

def LogIntegral(z):
    return li(z)

def ExpIntegralEi(z):
    return Ei(z)

def ExpIntegralE(a, b):
    return expint(a, b).evalf()

def SinIntegral(z):
    return Si(z)

def CosIntegral(z):
    return Ci(z)

def SinhIntegral(z):
    return Shi(z)

def CoshIntegral(z):
    return Chi(z)

class PolyGamma(Function):
    @classmethod
    def eval(cls, *args):
        if len(args) == 2:
            return polygamma(args[0], args[1])
        return digamma(args[0])

def LogGamma(z):
    return loggamma(z)

class ProductLog(Function):
    @classmethod
    def eval(cls, *args):
        if len(args) == 2:
            return LambertW(args[1], args[0]).evalf()
        return LambertW(args[0]).evalf()

def Factorial(a):
    return factorial(a)

def Zeta(*args):
    return zeta(*args)

def HypergeometricPFQ(a, b, c):
    return hyper(a, b, c)

def Sum_doit(exp, args):
    """
    This function perform summation using sympy's `Sum`.

    Examples
    ========

    >>> from sympy.integrals.rubi.utility_function import Sum_doit
    >>> from sympy.abc import x
    >>> Sum_doit(2*x + 2, [x, 0, 1.7])
    6

    """
    exp = replace_pow_exp(exp)
    if not isinstance(args[2], (int, Integer)):
        new_args = [args[0], args[1], Floor(args[2])]
        return Sum(exp, new_args).doit()

    return Sum(exp, args).doit()

def PolynomialQuotient(p, q, x):
    try:
        p = poly(p, x)
        q = poly(q, x)

    except:
        p = poly(p)
        q = poly(q)
    try:
        return quo(p, q).as_expr()
    except (PolynomialDivisionFailed, UnificationFailed):
        return p/q

def PolynomialRemainder(p, q, x):
    try:
        p = poly(p, x)
        q = poly(q, x)

    except:
        p = poly(p)
        q = poly(q)
    try:
        return rem(p, q).as_expr()
    except (PolynomialDivisionFailed, UnificationFailed):
        return S(0)

def Floor(x, a = None):
    if a is None:
        return floor(x)
    return a*floor(x/a)

def Factor(var):
    return factor(var)

def Rule(a, b):
    return {a: b}

def Distribute(expr, *args):
    if len(args) == 1:
        if isinstance(expr, args[0]):
            return expr
        else:
            return expr.expand()
    if len(args) == 2:
        if isinstance(expr, args[1]):
            return expr.expand()
        else:
            return expr
    return expr.expand()

def CoprimeQ(*args):
    args = S(args)
    g = gcd(*args)
    if g == 1:
        return True
    return False

def Discriminant(a, b):
    try:
        return discriminant(a, b)
    except PolynomialError:
        return Function('Discriminant')(a, b)

def Negative(x):
    return x < S(0)

def Quotient(m, n):
    return Floor(m/n)

def process_trig(expr):
    """
    This function processes trigonometric expressions such that all `cot` is
    rewritten in terms of `tan`, `sec` in terms of `cos`, `csc` in terms of `sin` and
    similarly for `coth`, `sech` and `csch`.

    Examples
    ========

    >>> from sympy.integrals.rubi.utility_function import process_trig
    >>> from sympy.abc import x
    >>> from sympy import coth, cot, csc
    >>> process_trig(x*cot(x))
    x/tan(x)
    >>> process_trig(coth(x)*csc(x))
    1/(sin(x)*tanh(x))

    """
    expr = expr.replace(lambda x: isinstance(x, cot), lambda x: 1/tan(x.args[0]))
    expr = expr.replace(lambda x: isinstance(x, sec), lambda x: 1/cos(x.args[0]))
    expr = expr.replace(lambda x: isinstance(x, csc), lambda x: 1/sin(x.args[0]))
    expr = expr.replace(lambda x: isinstance(x, coth), lambda x: 1/tanh(x.args[0]))
    expr = expr.replace(lambda x: isinstance(x, sech), lambda x: 1/cosh(x.args[0]))
    expr = expr.replace(lambda x: isinstance(x, csch), lambda x: 1/sinh(x.args[0]))
    return expr

def _ExpandIntegrand():
    Plus = Add
    Times = Mul
    def cons_f1(m):
        return PositiveIntegerQ(m)

    cons1 = CustomConstraint(cons_f1)
    def cons_f2(d, c, b, a):
        return ZeroQ(-a*d + b*c)

    cons2 = CustomConstraint(cons_f2)
    def cons_f3(a, x):
        return FreeQ(a, x)

    cons3 = CustomConstraint(cons_f3)
    def cons_f4(b, x):
        return FreeQ(b, x)

    cons4 = CustomConstraint(cons_f4)
    def cons_f5(c, x):
        return FreeQ(c, x)

    cons5 = CustomConstraint(cons_f5)
    def cons_f6(d, x):
        return FreeQ(d, x)

    cons6 = CustomConstraint(cons_f6)
    def cons_f7(e, x):
        return FreeQ(e, x)

    cons7 = CustomConstraint(cons_f7)
    def cons_f8(f, x):
        return FreeQ(f, x)

    cons8 = CustomConstraint(cons_f8)
    def cons_f9(g, x):
        return FreeQ(g, x)

    cons9 = CustomConstraint(cons_f9)
    def cons_f10(h, x):
        return FreeQ(h, x)

    cons10 = CustomConstraint(cons_f10)
    def cons_f11(e, b, c, f, n, p, F, x, d, m):
        if not isinstance(x, Symbol):
            return False
        return FreeQ(List(F, b, c, d, e, f, m, n, p), x)

    cons11 = CustomConstraint(cons_f11)
    def cons_f12(F, x):
        return FreeQ(F, x)

    cons12 = CustomConstraint(cons_f12)
    def cons_f13(m, x):
        return FreeQ(m, x)

    cons13 = CustomConstraint(cons_f13)
    def cons_f14(n, x):
        return FreeQ(n, x)

    cons14 = CustomConstraint(cons_f14)
    def cons_f15(p, x):
        return FreeQ(p, x)

    cons15 = CustomConstraint(cons_f15)
    def cons_f16(e, b, c, f, n, a, p, F, x, d, m):
        if not isinstance(x, Symbol):
            return False
        return FreeQ(List(F, a, b, c, d, e, f, m, n, p), x)

    cons16 = CustomConstraint(cons_f16)
    def cons_f17(n, m):
        return IntegersQ(m, n)

    cons17 = CustomConstraint(cons_f17)
    def cons_f18(n):
        return Less(n, S(0))

    cons18 = CustomConstraint(cons_f18)
    def cons_f19(x, u):
        if not isinstance(x, Symbol):
            return False
        return PolynomialQ(u, x)

    cons19 = CustomConstraint(cons_f19)
    def cons_f20(G, F, u):
        return SameQ(F(u)*G(u), S(1))

    cons20 = CustomConstraint(cons_f20)
    def cons_f21(q, x):
        return FreeQ(q, x)

    cons21 = CustomConstraint(cons_f21)
    def cons_f22(F):
        return MemberQ(List(ArcSin, ArcCos, ArcSinh, ArcCosh), F)

    cons22 = CustomConstraint(cons_f22)
    def cons_f23(j, n):
        return ZeroQ(j - S(2)*n)

    cons23 = CustomConstraint(cons_f23)
    def cons_f24(A, x):
        return FreeQ(A, x)

    cons24 = CustomConstraint(cons_f24)
    def cons_f25(B, x):
        return FreeQ(B, x)

    cons25 = CustomConstraint(cons_f25)
    def cons_f26(m, u, x):
        if not isinstance(x, Symbol):
            return False
        def _cons_f_u(d, w, c, p, x):
            return And(FreeQ(List(c, d), x), IntegerQ(p), Greater(p, m))
        cons_u = CustomConstraint(_cons_f_u)
        pat = Pattern(UtilityOperator((c_ + x_*WC('d', S(1)))**p_*WC('w', S(1)), x_), cons_u)
        result_matchq = is_match(UtilityOperator(u, x), pat)
        return Not(And(PositiveIntegerQ(m), result_matchq))

    cons26 = CustomConstraint(cons_f26)
    def cons_f27(b, v, n, a, x, u, m):
        if not isinstance(x, Symbol):
            return False
        return And(FreeQ(List(a, b, m), x), NegativeIntegerQ(n), Not(IntegerQ(m)), PolynomialQ(u, x), PolynomialQ(v, x),\
            RationalQ(m), Less(m, -1), GreaterEqual(Exponent(u, x), (-n - IntegerPart(m))*Exponent(v, x)))
    cons27 = CustomConstraint(cons_f27)
    def cons_f28(v, n, x, u, m):
        if not isinstance(x, Symbol):
            return False
        return And(FreeQ(List(a, b, m), x), NegativeIntegerQ(n), Not(IntegerQ(m)), PolynomialQ(u, x),\
            PolynomialQ(v, x), GreaterEqual(Exponent(u, x), -n*Exponent(v, x)))
    cons28 = CustomConstraint(cons_f28)
    def cons_f29(n):
        return PositiveIntegerQ(n/S(4))

    cons29 = CustomConstraint(cons_f29)
    def cons_f30(n):
        return IntegerQ(n)

    cons30 = CustomConstraint(cons_f30)
    def cons_f31(n):
        return Greater(n, S(1))

    cons31 = CustomConstraint(cons_f31)
    def cons_f32(n, m):
        return Less(S(0), m, n)

    cons32 = CustomConstraint(cons_f32)
    def cons_f33(n, m):
        return OddQ(n/GCD(m, n))

    cons33 = CustomConstraint(cons_f33)
    def cons_f34(a, b):
        return PosQ(a/b)

    cons34 = CustomConstraint(cons_f34)
    def cons_f35(n, m, p):
        return IntegersQ(m, n, p)

    cons35 = CustomConstraint(cons_f35)
    def cons_f36(n, m, p):
        return Less(S(0), m, p, n)

    cons36 = CustomConstraint(cons_f36)
    def cons_f37(q, n, m, p):
        return IntegersQ(m, n, p, q)

    cons37 = CustomConstraint(cons_f37)
    def cons_f38(n, q, m, p):
        return Less(S(0), m, p, q, n)

    cons38 = CustomConstraint(cons_f38)
    def cons_f39(n):
        return IntegerQ(n/S(2))

    cons39 = CustomConstraint(cons_f39)
    def cons_f40(p):
        return NegativeIntegerQ(p)

    cons40 = CustomConstraint(cons_f40)
    def cons_f41(n, m):
        return IntegersQ(m, n/S(2))

    cons41 = CustomConstraint(cons_f41)
    def cons_f42(n, m):
        return Unequal(m, n/S(2))

    cons42 = CustomConstraint(cons_f42)
    def cons_f43(c, b, a):
        return NonzeroQ(-S(4)*a*c + b**S(2))

    cons43 = CustomConstraint(cons_f43)
    def cons_f44(j, n, m):
        return IntegersQ(m, n, j)

    cons44 = CustomConstraint(cons_f44)
    def cons_f45(n, m):
        return Less(S(0), m, S(2)*n)

    cons45 = CustomConstraint(cons_f45)
    def cons_f46(n, m, p):
        return Not(And(Equal(m, n), Equal(p, S(-1))))

    cons46 = CustomConstraint(cons_f46)
    # ... other code
```
### 10 - sympy/codegen/rewriting.py:

Start line: 137, End line: 175

```python
exp2_opt = ReplaceOptim(
    lambda p: p.is_Pow and p.base == 2,
    lambda p: exp2(p.exp)
)


_d = Wild('d', properties=[lambda x: x.is_Dummy])
_u = Wild('u', properties=[lambda x: not x.is_number and not x.is_Add])
_v = Wild('v')
_w = Wild('w')
_n = Wild('n', properties=[lambda x: x.is_number])

sinc_opt1 = ReplaceOptim(
    sin(_w)/_w, sinc(_w)
)
sinc_opt2 = ReplaceOptim(
    sin(_n*_w)/_w, _n*sinc(_n*_w)
)
sinc_opts = (sinc_opt1, sinc_opt2)

log2_opt = ReplaceOptim(_v*log(_w)/log(2), _v*log2(_w), cost_function=lambda expr: expr.count(
    lambda e: (  # division & eval of transcendentals are expensive floating point operations...
        e.is_Pow and e.exp.is_negative  # division
        or (isinstance(e, (log, log2)) and not e.args[0].is_number))  # transcendental
    )
)

log2const_opt = ReplaceOptim(log(2)*log2(_w), log(_w))

logsumexp_2terms_opt = ReplaceOptim(
    lambda l: (isinstance(l, log)
               and l.args[0].is_Add
               and len(l.args[0].args) == 2
               and all(isinstance(t, exp) for t in l.args[0].args)),
    lambda l: (
        Max(*[e.args[0] for e in l.args[0].args]) +
        log1p(exp(Min(*[e.args[0] for e in l.args[0].args])))
    )
)
```
