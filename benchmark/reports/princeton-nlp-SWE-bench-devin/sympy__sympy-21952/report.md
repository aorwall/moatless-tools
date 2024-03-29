# sympy__sympy-21952

| **sympy/sympy** | `b8156f36f0f3144c5e3b66002b9e8fcbe2ee66c4` |
| ---- | ---- |
| **No of patches** | 2 |
| **All found context length** | - |
| **Any found context length** | 2004 |
| **Avg pos** | 6.0 |
| **Min pos** | 4 |
| **Max pos** | 4 |
| **Top file pos** | 2 |
| **Missing snippets** | 5 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sympy/core/mul.py b/sympy/core/mul.py
--- a/sympy/core/mul.py
+++ b/sympy/core/mul.py
@@ -1334,6 +1334,7 @@ def _eval_is_zero(self):
     #_eval_is_integer = lambda self: _fuzzy_group(
     #    (a.is_integer for a in self.args), quick_exit=True)
     def _eval_is_integer(self):
+        from sympy import trailing
         is_rational = self._eval_is_rational()
         if is_rational is False:
             return False
@@ -1342,11 +1343,14 @@ def _eval_is_integer(self):
         denominators = []
         for a in self.args:
             if a.is_integer:
-                numerators.append(a)
+                if abs(a) is not S.One:
+                    numerators.append(a)
             elif a.is_Rational:
                 n, d = a.as_numer_denom()
-                numerators.append(n)
-                denominators.append(d)
+                if abs(n) is not S.One:
+                    numerators.append(n)
+                if d is not S.One:
+                    denominators.append(d)
             elif a.is_Pow:
                 b, e = a.as_base_exp()
                 if not b.is_integer or not e.is_integer: return
@@ -1364,13 +1368,36 @@ def _eval_is_integer(self):
         if not denominators:
             return True
 
-        odd = lambda ints: all(i.is_odd for i in ints)
-        even = lambda ints: any(i.is_even for i in ints)
+        allodd = lambda x: all(i.is_odd for i in x)
+        alleven = lambda x: all(i.is_even for i in x)
+        anyeven = lambda x: any(i.is_even for i in x)
 
-        if odd(numerators) and even(denominators):
+        if allodd(numerators) and anyeven(denominators):
             return False
-        elif even(numerators) and denominators == [2]:
+        elif anyeven(numerators) and denominators == [2]:
             return True
+        elif alleven(numerators) and allodd(denominators
+                ) and (Mul(*denominators, evaluate=False) - 1
+                ).is_positive:
+            return False
+        if len(denominators) == 1:
+            d = denominators[0]
+            if d.is_Integer and d.is_even:
+                # if minimal power of 2 in num vs den is not
+                # negative then we have an integer
+                if (Add(*[i.as_base_exp()[1] for i in
+                        numerators if i.is_even]) - trailing(d.p)
+                        ).is_nonnegative:
+                    return True
+        if len(numerators) == 1:
+            n = numerators[0]
+            if n.is_Integer and n.is_even:
+                # if minimal power of 2 in den vs num is positive
+                # then we have have a non-integer
+                if (Add(*[i.as_base_exp()[1] for i in
+                        denominators if i.is_even]) - trailing(n.p)
+                        ).is_positive:
+                    return False
 
     def _eval_is_polar(self):
         has_polar = any(arg.is_polar for arg in self.args)
@@ -1545,37 +1572,54 @@ def _eval_is_extended_negative(self):
         return self._eval_pos_neg(-1)
 
     def _eval_is_odd(self):
+        from sympy import trailing, fraction
         is_integer = self.is_integer
-
         if is_integer:
+            if self.is_zero:
+                return False
+            n, d = fraction(self)
+            if d.is_Integer and d.is_even:
+                # if minimal power of 2 in num vs den is
+                # positive then we have an even number
+                if (Add(*[i.as_base_exp()[1] for i in
+                        Mul.make_args(n) if i.is_even]) - trailing(d.p)
+                        ).is_positive:
+                    return False
+                return
             r, acc = True, 1
             for t in self.args:
-                if not t.is_integer:
-                    return None
-                elif t.is_even:
+                if abs(t) is S.One:
+                    continue
+                assert t.is_integer
+                if t.is_even:
+                    return False
+                if r is False:
+                    pass
+                elif acc != 1 and (acc + t).is_odd:
                     r = False
-                elif t.is_integer:
-                    if r is False:
-                        pass
-                    elif acc != 1 and (acc + t).is_odd:
-                        r = False
-                    elif t.is_odd is None:
-                        r = None
+                elif t.is_even is None:
+                    r = None
                 acc = t
             return r
-
-        # !integer -> !odd
-        elif is_integer is False:
-            return False
+        return is_integer # !integer -> !odd
 
     def _eval_is_even(self):
+        from sympy import trailing, fraction
         is_integer = self.is_integer
 
         if is_integer:
             return fuzzy_not(self.is_odd)
 
-        elif is_integer is False:
-            return False
+        n, d = fraction(self)
+        if n.is_Integer and n.is_even:
+            # if minimal power of 2 in den vs num is not
+            # negative then this is not an integer and
+            # can't be even
+            if (Add(*[i.as_base_exp()[1] for i in
+                    Mul.make_args(d) if i.is_even]) - trailing(n.p)
+                    ).is_nonnegative:
+                return False
+        return is_integer
 
     def _eval_is_composite(self):
         """
diff --git a/sympy/core/numbers.py b/sympy/core/numbers.py
--- a/sympy/core/numbers.py
+++ b/sympy/core/numbers.py
@@ -1572,6 +1572,16 @@ class Rational(Number):
     >>> r.p/r.q
     0.75
 
+    If an unevaluated Rational is desired, ``gcd=1`` can be passed and
+    this will keep common divisors of the numerator and denominator
+    from being eliminated. It is not possible, however, to leave a
+    negative value in the denominator.
+
+    >>> Rational(2, 4, gcd=1)
+    2/4
+    >>> Rational(2, -4, gcd=1).q
+    4
+
     See Also
     ========
     sympy.core.sympify.sympify, sympy.simplify.simplify.nsimplify

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sympy/core/mul.py | 1337 | 1337 | 4 | 2 | 2004
| sympy/core/mul.py | 1345 | 1349 | 4 | 2 | 2004
| sympy/core/mul.py | 1367 | 1372 | 4 | 2 | 2004
| sympy/core/mul.py | 1548 | 1577 | - | 2 | -
| sympy/core/numbers.py | 1575 | 1575 | - | 5 | -


## Problem Statement

```
If n is even, n**2/2 should also be even
The following:

\`\`\` python
>>> n = Symbol('n', integer=True, even=True)
>>> (n**2/2).is_even
\`\`\`

should return `True`, but it returns `None` (of course, this is also an enhancement).

That makes me think that perhaps symbolic integers should keep a more complex "assumptions" method, which generalizes "is_even" and "is_odd" to instead contain a dictionary of primes that are known to divide that integer, mapping to minimum and maximum known multiplicity.

I would like to think about it and post a proposition/plan, but I am not sure what is the correct github way of doing this.

Updated _eval_is_odd to handle more complex inputs
Changed the function _eval_is_odd to handle integer inputs that have a denominator, such as 
\`\`\`
n = Symbol('n',integer=True,even=True)
m = Symbol('m',integer=true,even=True)
x = Mul(n,m,S.Half)
\`\`\`
The example expression x is recognized by SymPy as an integer, but can be decomposed into n,m,and 1/2.  My new function evaluates the oddness of each part and uses this to calculate the oddness of the entire integer.

Addresses issue #8648 

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 sympy/assumptions/handlers/ntheory.py | 144 | 183| 275 | 275 | 1628 | 
| 2 | 1 sympy/assumptions/handlers/ntheory.py | 185 | 268| 508 | 783 | 1628 | 
| 3 | 1 sympy/assumptions/handlers/ntheory.py | 1 | 142| 843 | 1626 | 1628 | 
| **-> 4 <-** | **2 sympy/core/mul.py** | 1336 | 1381| 378 | 2004 | 19021 | 
| 5 | **2 sympy/core/mul.py** | 1544 | 1569| 162 | 2166 | 19021 | 
| 6 | 3 sympy/assumptions/handlers/sets.py | 553 | 607| 463 | 2629 | 24566 | 
| 7 | **3 sympy/core/mul.py** | 1233 | 1335| 867 | 3496 | 24566 | 
| 8 | 4 sympy/functions/combinatorial/factorials.py | 442 | 490| 424 | 3920 | 33904 | 
| 9 | **5 sympy/core/numbers.py** | 2261 | 2297| 231 | 4151 | 64875 | 
| 10 | **5 sympy/core/numbers.py** | 2407 | 2500| 669 | 4820 | 64875 | 
| 11 | 6 sympy/ntheory/primetest.py | 556 | 601| 799 | 5619 | 71019 | 
| 12 | 7 sympy/ntheory/factor_.py | 2348 | 2425| 744 | 6363 | 91932 | 
| 13 | 8 sympy/core/power.py | 747 | 789| 339 | 6702 | 109073 | 
| 14 | 9 sympy/assumptions/handlers/order.py | 79 | 117| 255 | 6957 | 111894 | 
| 15 | 10 sympy/functions/combinatorial/numbers.py | 992 | 1029| 355 | 7312 | 130876 | 
| 16 | **10 sympy/core/mul.py** | 1571 | 1595| 177 | 7489 | 130876 | 
| 17 | 11 sympy/ntheory/residue_ntheory.py | 941 | 1026| 695 | 8184 | 142709 | 
| 18 | 12 sympy/functions/special/polynomials.py | 928 | 943| 265 | 8449 | 155701 | 
| 19 | 12 sympy/ntheory/factor_.py | 1 | 62| 727 | 9176 | 155701 | 
| 20 | 12 sympy/functions/combinatorial/numbers.py | 480 | 524| 386 | 9562 | 155701 | 
| 21 | 13 sympy/solvers/solveset.py | 1264 | 1328| 658 | 10220 | 189111 | 
| 22 | 14 sympy/concrete/summations.py | 1414 | 1420| 149 | 10369 | 203323 | 


### Hint

```
I have added some handling for this instance in this [PR](https://github.com/sympy/sympy/pull/12320).  I did put thought into generalizing this even more for any integer divisor, but because we don't have the factorizations for the symbols, this does not seem easily possible.
Can you add some tests (based on the other tests, it looks like `sympy/core/tests/test_arit.py` is the proper file). 
@asmeurer , [here](https://github.com/mikaylazgrace/sympy/blob/Working-on-issue-8648/sympy/core/mul.py) is the updated work (line 1266) incorporating some of the ideas we discussed above.  I'm unsure if I am calling the functions .fraction() and .as_coeff_mul() correctly.

An error raised by Travis in the original PR said that when I called the functions (around line 90 in mul.py), "self is not defined".  Any ideas on how to call the functions?  Below is how I did it (which didn't work):

\`\`\`
class Mul(Expr, AssocOp):

    __slots__ = []

    is_Mul = True
    is_odd = self._eval_is_odd()
    is_even = self._eval_is_even()
\`\`\`



self is only defined inside of methods (where self is the first argument). There's no need to define is_odd or is_even. SymPy defines those automatically from _eval_is_odd and _eval_is_even.
@asmeurer , Travis is giving me this error:       
\`\`\`
File "/home/travis/virtualenv/python2.7.9/lib/python2.7/site-packages/sympy/core/mul.py", line 1291, in _eval_is_odd
        symbols = list(chain.from_iterable(symbols)) #.as_coeff_mul() returns a tuple for arg[1] so we change it to a list
    TypeError: 'exp_polar' object is not iterable
\`\`\`

I'm a bit unsure how to approach/fix this.  The added symbols = list... was in response to the fact that .as_coeff_mul() returns a tuple in arg[1] which isn't iterable.
I'm not clear what you are trying to do on that line. You should be able to just have 

\`\`\`
coeff, args = self.as_coeff_mul()
for arg in args:
    ...
\`\`\`
I had some equivalent of that, but Travis said that I was trying to iterate over a tuple. I thought .as_coeff_mul() returned a tuple for the second part? Maybe I'm incorrect there. 
I just made a small change that might solve it:  I'm looping over symbols instead of symbols.arg[1].  

I also changed .as_coeff_mul() per your suggestion (around line 1288):
\`\`\`
        if is_integer:
            coeff,symbols = self.as_coeff_mul()
            if coeff == 1 or coeff == -1:
                r = True
                for symbol in symbols:
                ...
\`\`\`

@mikaylazgrace Are you still interested to work on it?
I will takeover and come up with a PR soon.
```

## Patch

```diff
diff --git a/sympy/core/mul.py b/sympy/core/mul.py
--- a/sympy/core/mul.py
+++ b/sympy/core/mul.py
@@ -1334,6 +1334,7 @@ def _eval_is_zero(self):
     #_eval_is_integer = lambda self: _fuzzy_group(
     #    (a.is_integer for a in self.args), quick_exit=True)
     def _eval_is_integer(self):
+        from sympy import trailing
         is_rational = self._eval_is_rational()
         if is_rational is False:
             return False
@@ -1342,11 +1343,14 @@ def _eval_is_integer(self):
         denominators = []
         for a in self.args:
             if a.is_integer:
-                numerators.append(a)
+                if abs(a) is not S.One:
+                    numerators.append(a)
             elif a.is_Rational:
                 n, d = a.as_numer_denom()
-                numerators.append(n)
-                denominators.append(d)
+                if abs(n) is not S.One:
+                    numerators.append(n)
+                if d is not S.One:
+                    denominators.append(d)
             elif a.is_Pow:
                 b, e = a.as_base_exp()
                 if not b.is_integer or not e.is_integer: return
@@ -1364,13 +1368,36 @@ def _eval_is_integer(self):
         if not denominators:
             return True
 
-        odd = lambda ints: all(i.is_odd for i in ints)
-        even = lambda ints: any(i.is_even for i in ints)
+        allodd = lambda x: all(i.is_odd for i in x)
+        alleven = lambda x: all(i.is_even for i in x)
+        anyeven = lambda x: any(i.is_even for i in x)
 
-        if odd(numerators) and even(denominators):
+        if allodd(numerators) and anyeven(denominators):
             return False
-        elif even(numerators) and denominators == [2]:
+        elif anyeven(numerators) and denominators == [2]:
             return True
+        elif alleven(numerators) and allodd(denominators
+                ) and (Mul(*denominators, evaluate=False) - 1
+                ).is_positive:
+            return False
+        if len(denominators) == 1:
+            d = denominators[0]
+            if d.is_Integer and d.is_even:
+                # if minimal power of 2 in num vs den is not
+                # negative then we have an integer
+                if (Add(*[i.as_base_exp()[1] for i in
+                        numerators if i.is_even]) - trailing(d.p)
+                        ).is_nonnegative:
+                    return True
+        if len(numerators) == 1:
+            n = numerators[0]
+            if n.is_Integer and n.is_even:
+                # if minimal power of 2 in den vs num is positive
+                # then we have have a non-integer
+                if (Add(*[i.as_base_exp()[1] for i in
+                        denominators if i.is_even]) - trailing(n.p)
+                        ).is_positive:
+                    return False
 
     def _eval_is_polar(self):
         has_polar = any(arg.is_polar for arg in self.args)
@@ -1545,37 +1572,54 @@ def _eval_is_extended_negative(self):
         return self._eval_pos_neg(-1)
 
     def _eval_is_odd(self):
+        from sympy import trailing, fraction
         is_integer = self.is_integer
-
         if is_integer:
+            if self.is_zero:
+                return False
+            n, d = fraction(self)
+            if d.is_Integer and d.is_even:
+                # if minimal power of 2 in num vs den is
+                # positive then we have an even number
+                if (Add(*[i.as_base_exp()[1] for i in
+                        Mul.make_args(n) if i.is_even]) - trailing(d.p)
+                        ).is_positive:
+                    return False
+                return
             r, acc = True, 1
             for t in self.args:
-                if not t.is_integer:
-                    return None
-                elif t.is_even:
+                if abs(t) is S.One:
+                    continue
+                assert t.is_integer
+                if t.is_even:
+                    return False
+                if r is False:
+                    pass
+                elif acc != 1 and (acc + t).is_odd:
                     r = False
-                elif t.is_integer:
-                    if r is False:
-                        pass
-                    elif acc != 1 and (acc + t).is_odd:
-                        r = False
-                    elif t.is_odd is None:
-                        r = None
+                elif t.is_even is None:
+                    r = None
                 acc = t
             return r
-
-        # !integer -> !odd
-        elif is_integer is False:
-            return False
+        return is_integer # !integer -> !odd
 
     def _eval_is_even(self):
+        from sympy import trailing, fraction
         is_integer = self.is_integer
 
         if is_integer:
             return fuzzy_not(self.is_odd)
 
-        elif is_integer is False:
-            return False
+        n, d = fraction(self)
+        if n.is_Integer and n.is_even:
+            # if minimal power of 2 in den vs num is not
+            # negative then this is not an integer and
+            # can't be even
+            if (Add(*[i.as_base_exp()[1] for i in
+                    Mul.make_args(d) if i.is_even]) - trailing(n.p)
+                    ).is_nonnegative:
+                return False
+        return is_integer
 
     def _eval_is_composite(self):
         """
diff --git a/sympy/core/numbers.py b/sympy/core/numbers.py
--- a/sympy/core/numbers.py
+++ b/sympy/core/numbers.py
@@ -1572,6 +1572,16 @@ class Rational(Number):
     >>> r.p/r.q
     0.75
 
+    If an unevaluated Rational is desired, ``gcd=1`` can be passed and
+    this will keep common divisors of the numerator and denominator
+    from being eliminated. It is not possible, however, to leave a
+    negative value in the denominator.
+
+    >>> Rational(2, 4, gcd=1)
+    2/4
+    >>> Rational(2, -4, gcd=1).q
+    4
+
     See Also
     ========
     sympy.core.sympify.sympify, sympy.simplify.simplify.nsimplify

```

## Test Patch

```diff
diff --git a/sympy/core/tests/test_arit.py b/sympy/core/tests/test_arit.py
--- a/sympy/core/tests/test_arit.py
+++ b/sympy/core/tests/test_arit.py
@@ -512,6 +512,12 @@ def test_Mul_is_even_odd():
     assert (x*(x + k)).is_odd is False
     assert (x*(x + m)).is_odd is None
 
+    # issue 8648
+    assert (m**2/2).is_even
+    assert (m**2/3).is_even is False
+    assert (2/m**2).is_odd is False
+    assert (2/m).is_odd is None
+
 
 @XFAIL
 def test_evenness_in_ternary_integer_product_with_odd():
@@ -1051,6 +1057,18 @@ def test_Pow_is_integer():
     assert (1/(x + 1)).is_integer is False
     assert (1/(-x - 1)).is_integer is False
 
+    # issue 8648-like
+    k = Symbol('k', even=True)
+    assert (k**3/2).is_integer
+    assert (k**3/8).is_integer
+    assert (k**3/16).is_integer is None
+    assert (2/k).is_integer is None
+    assert (2/k**2).is_integer is False
+    o = Symbol('o', odd=True)
+    assert (k/o).is_integer is None
+    o = Symbol('o', odd=True, prime=True)
+    assert (k/o).is_integer is False
+
 
 def test_Pow_is_real():
     x = Symbol('x', real=True)
diff --git a/sympy/core/tests/test_numbers.py b/sympy/core/tests/test_numbers.py
--- a/sympy/core/tests/test_numbers.py
+++ b/sympy/core/tests/test_numbers.py
@@ -343,6 +343,11 @@ def test_Rational_new():
     assert Rational(mpq(2, 6)) == Rational(1, 3)
     assert Rational(PythonRational(2, 6)) == Rational(1, 3)
 
+    assert Rational(2, 4, gcd=1).q == 4
+    n = Rational(2, -4, gcd=1)
+    assert n.q == 4
+    assert n.p == -2
+
 
 def test_Number_new():
     """"

```


## Code snippets

### 1 - sympy/assumptions/handlers/ntheory.py:

Start line: 144, End line: 183

```python
@EvenPredicate.register(Mul)
def _(expr, assumptions):
    """
    Even * Integer    -> Even
    Even * Odd        -> Even
    Integer * Odd     -> ?
    Odd * Odd         -> Odd
    Even * Even       -> Even
    Integer * Integer -> Even if Integer + Integer = Odd
    otherwise         -> ?
    """
    if expr.is_number:
        return _EvenPredicate_number(expr, assumptions)
    even, odd, irrational, acc = False, 0, False, 1
    for arg in expr.args:
        # check for all integers and at least one even
        if ask(Q.integer(arg), assumptions):
            if ask(Q.even(arg), assumptions):
                even = True
            elif ask(Q.odd(arg), assumptions):
                odd += 1
            elif not even and acc != 1:
                if ask(Q.odd(acc + arg), assumptions):
                    even = True
        elif ask(Q.irrational(arg), assumptions):
            # one irrational makes the result False
            # two makes it undefined
            if irrational:
                break
            irrational = True
        else:
            break
        acc = arg
    else:
        if irrational:
            return False
        if even:
            return True
        if odd == len(expr.args):
            return False
```
### 2 - sympy/assumptions/handlers/ntheory.py:

Start line: 185, End line: 268

```python
@EvenPredicate.register(Add)
def _(expr, assumptions):
    """
    Even + Odd  -> Odd
    Even + Even -> Even
    Odd  + Odd  -> Even

    """
    if expr.is_number:
        return _EvenPredicate_number(expr, assumptions)
    _result = True
    for arg in expr.args:
        if ask(Q.even(arg), assumptions):
            pass
        elif ask(Q.odd(arg), assumptions):
            _result = not _result
        else:
            break
    else:
        return _result

@EvenPredicate.register(Pow)
def _(expr, assumptions):
    if expr.is_number:
        return _EvenPredicate_number(expr, assumptions)
    if ask(Q.integer(expr.exp), assumptions):
        if ask(Q.positive(expr.exp), assumptions):
            return ask(Q.even(expr.base), assumptions)
        elif ask(~Q.negative(expr.exp) & Q.odd(expr.base), assumptions):
            return False
        elif expr.base is S.NegativeOne:
            return False

@EvenPredicate.register(Integer)
def _(expr, assumptions):
    return not bool(expr.p & 1)

@EvenPredicate.register_many(Rational, Infinity, NegativeInfinity, ImaginaryUnit)
def _(expr, assumptions):
    return False

@EvenPredicate.register(NumberSymbol)
def _(expr, assumptions):
    return _EvenPredicate_number(expr, assumptions)

@EvenPredicate.register(Abs)
def _(expr, assumptions):
    if ask(Q.real(expr.args[0]), assumptions):
        return ask(Q.even(expr.args[0]), assumptions)

@EvenPredicate.register(re)
def _(expr, assumptions):
    if ask(Q.real(expr.args[0]), assumptions):
        return ask(Q.even(expr.args[0]), assumptions)

@EvenPredicate.register(im)
def _(expr, assumptions):
    if ask(Q.real(expr.args[0]), assumptions):
        return True

@EvenPredicate.register(NaN)
def _(expr, assumptions):
    return None


# OddPredicate

@OddPredicate.register(Expr)
def _(expr, assumptions):
    ret = expr.is_odd
    if ret is None:
        raise MDNotImplementedError
    return ret

@OddPredicate.register(Basic)
def _(expr, assumptions):
    _integer = ask(Q.integer(expr), assumptions)
    if _integer:
        _even = ask(Q.even(expr), assumptions)
        if _even is None:
            return None
        return not _even
    return _integer
```
### 3 - sympy/assumptions/handlers/ntheory.py:

Start line: 1, End line: 142

```python
"""
Handlers for keys related to number theory: prime, even, odd, etc.
"""

from sympy.assumptions import Q, ask
from sympy.core import Add, Basic, Expr, Float, Mul, Pow, S
from sympy.core.numbers import (ImaginaryUnit, Infinity, Integer, NaN,
    NegativeInfinity, NumberSymbol, Rational)
from sympy.functions import Abs, im, re
from sympy.ntheory import isprime

from sympy.multipledispatch import MDNotImplementedError

from ..predicates.ntheory import (PrimePredicate, CompositePredicate,
    EvenPredicate, OddPredicate)


# PrimePredicate

def _PrimePredicate_number(expr, assumptions):
    # helper method
    exact = not expr.atoms(Float)
    try:
        i = int(expr.round())
        if (expr - i).equals(0) is False:
            raise TypeError
    except TypeError:
        return False
    if exact:
        return isprime(i)
    # when not exact, we won't give a True or False
    # since the number represents an approximate value

@PrimePredicate.register(Expr)
def _(expr, assumptions):
    ret = expr.is_prime
    if ret is None:
        raise MDNotImplementedError
    return ret

@PrimePredicate.register(Basic)
def _(expr, assumptions):
    if expr.is_number:
        return _PrimePredicate_number(expr, assumptions)

@PrimePredicate.register(Mul)
def _(expr, assumptions):
    if expr.is_number:
        return _PrimePredicate_number(expr, assumptions)
    for arg in expr.args:
        if not ask(Q.integer(arg), assumptions):
            return None
    for arg in expr.args:
        if arg.is_number and arg.is_composite:
            return False

@PrimePredicate.register(Pow)
def _(expr, assumptions):
    """
    Integer**Integer     -> !Prime
    """
    if expr.is_number:
        return _PrimePredicate_number(expr, assumptions)
    if ask(Q.integer(expr.exp), assumptions) and \
            ask(Q.integer(expr.base), assumptions):
        return False

@PrimePredicate.register(Integer)
def _(expr, assumptions):
    return isprime(expr)

@PrimePredicate.register_many(Rational, Infinity, NegativeInfinity, ImaginaryUnit)
def _(expr, assumptions):
    return False

@PrimePredicate.register(Float)
def _(expr, assumptions):
    return _PrimePredicate_number(expr, assumptions)

@PrimePredicate.register(NumberSymbol)
def _(expr, assumptions):
    return _PrimePredicate_number(expr, assumptions)

@PrimePredicate.register(NaN)
def _(expr, assumptions):
    return None


# CompositePredicate

@CompositePredicate.register(Expr)
def _(expr, assumptions):
    ret = expr.is_composite
    if ret is None:
        raise MDNotImplementedError
    return ret

@CompositePredicate.register(Basic)
def _(expr, assumptions):
    _positive = ask(Q.positive(expr), assumptions)
    if _positive:
        _integer = ask(Q.integer(expr), assumptions)
        if _integer:
            _prime = ask(Q.prime(expr), assumptions)
            if _prime is None:
                return
            # Positive integer which is not prime is not
            # necessarily composite
            if expr.equals(1):
                return False
            return not _prime
        else:
            return _integer
    else:
        return _positive


# EvenPredicate

def _EvenPredicate_number(expr, assumptions):
    # helper method
    try:
        i = int(expr.round())
        if not (expr - i).equals(0):
            raise TypeError
    except TypeError:
        return False
    if isinstance(expr, (float, Float)):
        return False
    return i % 2 == 0

@EvenPredicate.register(Expr)
def _(expr, assumptions):
    ret = expr.is_even
    if ret is None:
        raise MDNotImplementedError
    return ret

@EvenPredicate.register(Basic)
def _(expr, assumptions):
    if expr.is_number:
        return _EvenPredicate_number(expr, assumptions)
```
### 4 - sympy/core/mul.py:

Start line: 1336, End line: 1381

```python
class Mul(Expr, AssocOp):
    def _eval_is_integer(self):
        is_rational = self._eval_is_rational()
        if is_rational is False:
            return False

        numerators = []
        denominators = []
        for a in self.args:
            if a.is_integer:
                numerators.append(a)
            elif a.is_Rational:
                n, d = a.as_numer_denom()
                numerators.append(n)
                denominators.append(d)
            elif a.is_Pow:
                b, e = a.as_base_exp()
                if not b.is_integer or not e.is_integer: return
                if e.is_negative:
                    denominators.append(2 if a is S.Half else Pow(a, S.NegativeOne))
                else:
                    # for integer b and positive integer e: a = b**e would be integer
                    assert not e.is_positive
                    # for self being rational and e equal to zero: a = b**e would be 1
                    assert not e.is_zero
                    return # sign of e unknown -> self.is_integer cannot be decided
            else:
                return

        if not denominators:
            return True

        odd = lambda ints: all(i.is_odd for i in ints)
        even = lambda ints: any(i.is_even for i in ints)

        if odd(numerators) and even(denominators):
            return False
        elif even(numerators) and denominators == [2]:
            return True

    def _eval_is_polar(self):
        has_polar = any(arg.is_polar for arg in self.args)
        return has_polar and \
            all(arg.is_polar or arg.is_positive for arg in self.args)

    def _eval_is_extended_real(self):
        return self._eval_real_imag(True)
```
### 5 - sympy/core/mul.py:

Start line: 1544, End line: 1569

```python
class Mul(Expr, AssocOp):

    def _eval_is_extended_negative(self):
        return self._eval_pos_neg(-1)

    def _eval_is_odd(self):
        is_integer = self.is_integer

        if is_integer:
            r, acc = True, 1
            for t in self.args:
                if not t.is_integer:
                    return None
                elif t.is_even:
                    r = False
                elif t.is_integer:
                    if r is False:
                        pass
                    elif acc != 1 and (acc + t).is_odd:
                        r = False
                    elif t.is_odd is None:
                        r = None
                acc = t
            return r

        # !integer -> !odd
        elif is_integer is False:
            return False
```
### 6 - sympy/assumptions/handlers/sets.py:

Start line: 553, End line: 607

```python
@ImaginaryPredicate.register(Pow)
def _(expr, assumptions):
    """
    * Imaginary**Odd        -> Imaginary
    * Imaginary**Even       -> Real
    * b**Imaginary          -> !Imaginary if exponent is an integer
                               multiple of I*pi/log(b)
    * Imaginary**Real       -> ?
    * Positive**Real        -> Real
    * Negative**Integer     -> Real
    * Negative**(Integer/2) -> Imaginary
    * Negative**Real        -> not Imaginary if exponent is not Rational
    """
    if expr.is_number:
        return _Imaginary_number(expr, assumptions)

    if expr.base == E:
        a = expr.exp/I/pi
        return ask(Q.integer(2*a) & ~Q.integer(a), assumptions)

    if expr.base.func == exp or (expr.base.is_Pow and expr.base.base == E):
        if ask(Q.imaginary(expr.base.exp), assumptions):
            if ask(Q.imaginary(expr.exp), assumptions):
                return False
            i = expr.base.exp/I/pi
            if ask(Q.integer(2*i), assumptions):
                return ask(Q.imaginary(((-1)**i)**expr.exp), assumptions)

    if ask(Q.imaginary(expr.base), assumptions):
        if ask(Q.integer(expr.exp), assumptions):
            odd = ask(Q.odd(expr.exp), assumptions)
            if odd is not None:
                return odd
            return

    if ask(Q.imaginary(expr.exp), assumptions):
        imlog = ask(Q.imaginary(log(expr.base)), assumptions)
        if imlog is not None:
            # I**i -> real; (2*I)**i -> complex ==> not imaginary
            return False

    if ask(Q.real(expr.base) & Q.real(expr.exp), assumptions):
        if ask(Q.positive(expr.base), assumptions):
            return False
        else:
            rat = ask(Q.rational(expr.exp), assumptions)
            if not rat:
                return rat
            if ask(Q.integer(expr.exp), assumptions):
                return False
            else:
                half = ask(Q.integer(2*expr.exp), assumptions)
                if half:
                    return ask(Q.negative(expr.base), assumptions)
                return half
```
### 7 - sympy/core/mul.py:

Start line: 1233, End line: 1335

```python
class Mul(Expr, AssocOp):

    def as_powers_dict(self):
        d = defaultdict(int)
        for term in self.args:
            for b, e in term.as_powers_dict().items():
                d[b] += e
        return d

    def as_numer_denom(self):
        # don't use _from_args to rebuild the numerators and denominators
        # as the order is not guaranteed to be the same once they have
        # been separated from each other
        numers, denoms = list(zip(*[f.as_numer_denom() for f in self.args]))
        return self.func(*numers), self.func(*denoms)

    def as_base_exp(self):
        e1 = None
        bases = []
        nc = 0
        for m in self.args:
            b, e = m.as_base_exp()
            if not b.is_commutative:
                nc += 1
            if e1 is None:
                e1 = e
            elif e != e1 or nc > 1:
                return self, S.One
            bases.append(b)
        return self.func(*bases), e1

    def _eval_is_polynomial(self, syms):
        return all(term._eval_is_polynomial(syms) for term in self.args)

    def _eval_is_rational_function(self, syms):
        return all(term._eval_is_rational_function(syms) for term in self.args)

    def _eval_is_meromorphic(self, x, a):
        return _fuzzy_group((arg.is_meromorphic(x, a) for arg in self.args),
                            quick_exit=True)

    def _eval_is_algebraic_expr(self, syms):
        return all(term._eval_is_algebraic_expr(syms) for term in self.args)

    _eval_is_commutative = lambda self: _fuzzy_group(
        a.is_commutative for a in self.args)

    def _eval_is_complex(self):
        comp = _fuzzy_group(a.is_complex for a in self.args)
        if comp is False:
            if any(a.is_infinite for a in self.args):
                if any(a.is_zero is not False for a in self.args):
                    return None
                return False
        return comp

    def _eval_is_finite(self):
        if all(a.is_finite for a in self.args):
            return True
        if any(a.is_infinite for a in self.args):
            if all(a.is_zero is False for a in self.args):
                return False

    def _eval_is_infinite(self):
        if any(a.is_infinite for a in self.args):
            if any(a.is_zero for a in self.args):
                return S.NaN.is_infinite
            if any(a.is_zero is None for a in self.args):
                return None
            return True

    def _eval_is_rational(self):
        r = _fuzzy_group((a.is_rational for a in self.args), quick_exit=True)
        if r:
            return r
        elif r is False:
            return self.is_zero

    def _eval_is_algebraic(self):
        r = _fuzzy_group((a.is_algebraic for a in self.args), quick_exit=True)
        if r:
            return r
        elif r is False:
            return self.is_zero

    def _eval_is_zero(self):
        zero = infinite = False
        for a in self.args:
            z = a.is_zero
            if z:
                if infinite:
                    return  # 0*oo is nan and nan.is_zero is None
                zero = True
            else:
                if not a.is_finite:
                    if zero:
                        return  # 0*oo is nan and nan.is_zero is None
                    infinite = True
                if zero is False and z is None:  # trap None
                    zero = None
        return zero

    # without involving odd/even checks this code would suffice:
    #_eval_is_integer = lambda self: _fuzzy_group(
    #    (a.is_integer for a in self.args), quick_exit=True)
```
### 8 - sympy/functions/combinatorial/factorials.py:

Start line: 442, End line: 490

```python
class factorial2(CombinatorialFunction):


    def _eval_is_even(self):
        # Double factorial is even for every positive even input
        n = self.args[0]
        if n.is_integer:
            if n.is_odd:
                return False
            if n.is_even:
                if n.is_positive:
                    return True
                if n.is_zero:
                    return False

    def _eval_is_integer(self):
        # Double factorial is an integer for every nonnegative input, and for
        # -1 and -3
        n = self.args[0]
        if n.is_integer:
            if (n + 1).is_nonnegative:
                return True
            if n.is_odd:
                return (n + 3).is_nonnegative

    def _eval_is_odd(self):
        # Double factorial is odd for every odd input not smaller than -3, and
        # for 0
        n = self.args[0]
        if n.is_odd:
            return (n + 3).is_nonnegative
        if n.is_even:
            if n.is_positive:
                return False
            if n.is_zero:
                return True

    def _eval_is_positive(self):
        # Double factorial is positive for every nonnegative input, and for
        # every odd negative input which is of the form -1-4k for an
        # nonnegative integer k
        n = self.args[0]
        if n.is_integer:
            if (n + 1).is_nonnegative:
                return True
            if n.is_odd:
                return ((n + 1) / 2).is_even

    def _eval_rewrite_as_gamma(self, n, piecewise=True, **kwargs):
        from sympy import gamma, Piecewise, sqrt
        return 2**(n/2)*gamma(n/2 + 1) * Piecewise((1, Eq(Mod(n, 2), 0)),
                (sqrt(2/pi), Eq(Mod(n, 2), 1)))
```
### 9 - sympy/core/numbers.py:

Start line: 2261, End line: 2297

```python
class Integer(Rational):

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

Start line: 2407, End line: 2500

```python
class Integer(Rational):

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
    # for Integer only and not for general sympy expressions. This is to achieve
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
### 16 - sympy/core/mul.py:

Start line: 1571, End line: 1595

```python
class Mul(Expr, AssocOp):

    def _eval_is_even(self):
        is_integer = self.is_integer

        if is_integer:
            return fuzzy_not(self.is_odd)

        elif is_integer is False:
            return False

    def _eval_is_composite(self):
        """
        Here we count the number of arguments that have a minimum value
        greater than two.
        If there are more than one of such a symbol then the result is composite.
        Else, the result cannot be determined.
        """
        number_of_args = 0 # count of symbols with minimum value greater than one
        for arg in self.args:
            if not (arg.is_integer and arg.is_positive):
                return None
            if (arg-1).is_positive:
                number_of_args += 1

        if number_of_args > 1:
            return True
```
