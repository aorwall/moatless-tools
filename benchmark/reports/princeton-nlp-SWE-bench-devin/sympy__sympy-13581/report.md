# sympy__sympy-13581

| **sympy/sympy** | `a531dfdf2c536620fdaf080f7470dde08c257e92` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 944 |
| **Any found context length** | 944 |
| **Avg pos** | 2.0 |
| **Min pos** | 2 |
| **Max pos** | 2 |
| **Top file pos** | 2 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sympy/core/mod.py b/sympy/core/mod.py
--- a/sympy/core/mod.py
+++ b/sympy/core/mod.py
@@ -107,6 +107,38 @@ def doit(p, q):
             elif (qinner*(q + qinner)).is_nonpositive:
                 # |qinner| < |q| and have different sign
                 return p
+        elif isinstance(p, Add):
+            # separating into modulus and non modulus
+            both_l = non_mod_l, mod_l = [], []
+            for arg in p.args:
+                both_l[isinstance(arg, cls)].append(arg)
+            # if q same for all
+            if mod_l and all(inner.args[1] == q for inner in mod_l):
+                net = Add(*non_mod_l) + Add(*[i.args[0] for i in mod_l])
+                return cls(net, q)
+
+        elif isinstance(p, Mul):
+            # separating into modulus and non modulus
+            both_l = non_mod_l, mod_l = [], []
+            for arg in p.args:
+                both_l[isinstance(arg, cls)].append(arg)
+
+            if mod_l and all(inner.args[1] == q for inner in mod_l):
+                # finding distributive term
+                non_mod_l = [cls(x, q) for x in non_mod_l]
+                mod = []
+                non_mod = []
+                for j in non_mod_l:
+                    if isinstance(j, cls):
+                        mod.append(j.args[0])
+                    else:
+                        non_mod.append(j)
+                prod_mod = Mul(*mod)
+                prod_non_mod = Mul(*non_mod)
+                prod_mod1 = Mul(*[i.args[0] for i in mod_l])
+                net = prod_mod1*prod_mod
+                return prod_non_mod*cls(net, q)
+
         # XXX other possibilities?
 
         # extract gcd; any further simplification should be done by the user

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sympy/core/mod.py | 110 | 110 | 2 | 2 | 944


## Problem Statement

```
Mod(Mod(x + 1, 2) + 1, 2) should simplify to Mod(x, 2)
From [stackoverflow](https://stackoverflow.com/questions/46914006/modulo-computations-in-sympy-fail)

Also, something like `Mod(foo*Mod(x + 1, 2) + non_mod_terms + 1, 2)` could be simplified. Recursively.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 sympy/simplify/hyperexpand.py | 84 | 96| 160 | 160 | 24725 | 
| **-> 2 <-** | **2 sympy/core/mod.py** | 91 | 180| 784 | 944 | 26044 | 
| 3 | **2 sympy/core/mod.py** | 26 | 89| 434 | 1378 | 26044 | 
| 4 | **2 sympy/core/mod.py** | 1 | 24| 132 | 1510 | 26044 | 
| 5 | 3 sympy/core/expr.py | 1342 | 1428| 808 | 2318 | 54664 | 
| 6 | 4 sympy/ntheory/residue_ntheory.py | 479 | 581| 768 | 3086 | 64650 | 
| 7 | 5 sympy/integrals/meijerint.py | 1683 | 1717| 342 | 3428 | 88605 | 
| 8 | 6 sympy/functions/special/polynomials.py | 885 | 900| 265 | 3693 | 101363 | 
| 9 | 7 sympy/core/mul.py | 1604 | 1628| 237 | 3930 | 115652 | 
| 10 | 8 sympy/core/exprtools.py | 1052 | 1098| 441 | 4371 | 127600 | 
| 11 | 8 sympy/ntheory/residue_ntheory.py | 302 | 359| 424 | 4795 | 127600 | 
| 12 | 9 sympy/core/power.py | 637 | 691| 643 | 5438 | 141450 | 
| 13 | 9 sympy/core/mul.py | 474 | 562| 823 | 6261 | 141450 | 
| 14 | 10 sympy/simplify/fu.py | 607 | 662| 532 | 6793 | 159789 | 
| 15 | 11 sympy/polys/agca/modules.py | 825 | 849| 225 | 7018 | 171370 | 
| 16 | 12 sympy/polys/galoistools.py | 2247 | 2284| 377 | 7395 | 189673 | 
| 17 | 12 sympy/simplify/fu.py | 1227 | 1284| 527 | 7922 | 189673 | 
| 18 | 12 sympy/functions/special/polynomials.py | 917 | 926| 158 | 8080 | 189673 | 
| 19 | 12 sympy/core/mul.py | 235 | 364| 964 | 9044 | 189673 | 
| 20 | 12 sympy/core/expr.py | 1260 | 1301| 316 | 9360 | 189673 | 


## Patch

```diff
diff --git a/sympy/core/mod.py b/sympy/core/mod.py
--- a/sympy/core/mod.py
+++ b/sympy/core/mod.py
@@ -107,6 +107,38 @@ def doit(p, q):
             elif (qinner*(q + qinner)).is_nonpositive:
                 # |qinner| < |q| and have different sign
                 return p
+        elif isinstance(p, Add):
+            # separating into modulus and non modulus
+            both_l = non_mod_l, mod_l = [], []
+            for arg in p.args:
+                both_l[isinstance(arg, cls)].append(arg)
+            # if q same for all
+            if mod_l and all(inner.args[1] == q for inner in mod_l):
+                net = Add(*non_mod_l) + Add(*[i.args[0] for i in mod_l])
+                return cls(net, q)
+
+        elif isinstance(p, Mul):
+            # separating into modulus and non modulus
+            both_l = non_mod_l, mod_l = [], []
+            for arg in p.args:
+                both_l[isinstance(arg, cls)].append(arg)
+
+            if mod_l and all(inner.args[1] == q for inner in mod_l):
+                # finding distributive term
+                non_mod_l = [cls(x, q) for x in non_mod_l]
+                mod = []
+                non_mod = []
+                for j in non_mod_l:
+                    if isinstance(j, cls):
+                        mod.append(j.args[0])
+                    else:
+                        non_mod.append(j)
+                prod_mod = Mul(*mod)
+                prod_non_mod = Mul(*non_mod)
+                prod_mod1 = Mul(*[i.args[0] for i in mod_l])
+                net = prod_mod1*prod_mod
+                return prod_non_mod*cls(net, q)
+
         # XXX other possibilities?
 
         # extract gcd; any further simplification should be done by the user

```

## Test Patch

```diff
diff --git a/sympy/core/tests/test_arit.py b/sympy/core/tests/test_arit.py
--- a/sympy/core/tests/test_arit.py
+++ b/sympy/core/tests/test_arit.py
@@ -1655,6 +1655,12 @@ def test_Mod():
     # issue 10963
     assert (x**6000%400).args[1] == 400
 
+    #issue 13543
+    assert Mod(Mod(x + 1, 2) + 1 , 2) == Mod(x,2)
+
+    assert Mod(Mod(x + 2, 4)*(x + 4), 4) == Mod(x*(x + 2), 4)
+    assert Mod(Mod(x + 2, 4)*4, 4) == 0
+
 
 def test_Mod_is_integer():
     p = Symbol('p', integer=True)

```


## Code snippets

### 1 - sympy/simplify/hyperexpand.py:

Start line: 84, End line: 96

```python
# function to define "buckets"
def _mod1(x):
    # TODO see if this can work as Mod(x, 1); this will require
    # different handling of the "buckets" since these need to
    # be sorted and that fails when there is a mixture of
    # integers and expressions with parameters. With the current
    # Mod behavior, Mod(k, 1) == Mod(1, 1) == 0 if k is an integer.
    # Although the sorting can be done with Basic.compare, this may
    # still require different handling of the sorted buckets.
    if x.is_Number:
        return Mod(x, 1)
    c, x = x.as_coeff_Add()
    return Mod(c, 1) + x
```
### 2 - sympy/core/mod.py:

Start line: 91, End line: 180

```python
class Mod(Function):

    @classmethod
    def eval(cls, p, q):
        # ... other code

        rv = doit(p, q)
        if rv is not None:
            return rv

        # denest
        if isinstance(p, cls):
            qinner = p.args[1]
            if qinner % q == 0:
                return cls(p.args[0], q)
            elif (qinner*(q - qinner)).is_nonnegative:
                # |qinner| < |q| and have same sign
                return p
        elif isinstance(-p, cls):
            qinner = (-p).args[1]
            if qinner % q == 0:
                return cls(-(-p).args[0], q)
            elif (qinner*(q + qinner)).is_nonpositive:
                # |qinner| < |q| and have different sign
                return p
        # XXX other possibilities?

        # extract gcd; any further simplification should be done by the user
        G = gcd(p, q)
        if G != 1:
            p, q = [
                gcd_terms(i/G, clear=False, fraction=False) for i in (p, q)]
        pwas, qwas = p, q

        # simplify terms
        # (x + y + 2) % x -> Mod(y + 2, x)
        if p.is_Add:
            args = []
            for i in p.args:
                a = cls(i, q)
                if a.count(cls) > i.count(cls):
                    args.append(i)
                else:
                    args.append(a)
            if args != list(p.args):
                p = Add(*args)

        else:
            # handle coefficients if they are not Rational
            # since those are not handled by factor_terms
            # e.g. Mod(.6*x, .3*y) -> 0.3*Mod(2*x, y)
            cp, p = p.as_coeff_Mul()
            cq, q = q.as_coeff_Mul()
            ok = False
            if not cp.is_Rational or not cq.is_Rational:
                r = cp % cq
                if r == 0:
                    G *= cq
                    p *= int(cp/cq)
                    ok = True
            if not ok:
                p = cp*p
                q = cq*q

        # simple -1 extraction
        if p.could_extract_minus_sign() and q.could_extract_minus_sign():
            G, p, q = [-i for i in (G, p, q)]

        # check again to see if p and q can now be handled as numbers
        rv = doit(p, q)
        if rv is not None:
            return rv*G

        # put 1.0 from G on inside
        if G.is_Float and G == 1:
            p *= G
            return cls(p, q, evaluate=False)
        elif G.is_Mul and G.args[0].is_Float and G.args[0] == 1:
            p = G.args[0]*p
            G = Mul._from_args(G.args[1:])
        return G*cls(p, q, evaluate=(p, q) != (pwas, qwas))

    def _eval_is_integer(self):
        from sympy.core.logic import fuzzy_and, fuzzy_not
        p, q = self.args
        if fuzzy_and([p.is_integer, q.is_integer, fuzzy_not(q.is_zero)]):
            return True

    def _eval_is_nonnegative(self):
        if self.args[1].is_positive:
            return True

    def _eval_is_nonpositive(self):
        if self.args[1].is_negative:
            return True
```
### 3 - sympy/core/mod.py:

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
### 4 - sympy/core/mod.py:

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
### 5 - sympy/core/expr.py:

Start line: 1342, End line: 1428

```python
class Expr(Basic, EvalfMixin):

    def coeff(self, x, n=1, right=False):
        # ... other code

        if self_c:
            xargs = x.args_cnc(cset=True, warn=False)[0]
            for a in args:
                margs = a.args_cnc(cset=True, warn=False)[0]
                if len(xargs) > len(margs):
                    continue
                resid = margs.difference(xargs)
                if len(resid) + len(xargs) == len(margs):
                    co.append(Mul(*resid))
            if co == []:
                return S.Zero
            elif co:
                return Add(*co)
        elif x_c:
            xargs = x.args_cnc(cset=True, warn=False)[0]
            for a in args:
                margs, nc = a.args_cnc(cset=True)
                if len(xargs) > len(margs):
                    continue
                resid = margs.difference(xargs)
                if len(resid) + len(xargs) == len(margs):
                    co.append(Mul(*(list(resid) + nc)))
            if co == []:
                return S.Zero
            elif co:
                return Add(*co)
        else:  # both nc
            xargs, nx = x.args_cnc(cset=True)
            # find the parts that pass the commutative terms
            for a in args:
                margs, nc = a.args_cnc(cset=True)
                if len(xargs) > len(margs):
                    continue
                resid = margs.difference(xargs)
                if len(resid) + len(xargs) == len(margs):
                    co.append((resid, nc))
            # now check the non-comm parts
            if not co:
                return S.Zero
            if all(n == co[0][1] for r, n in co):
                ii = find(co[0][1], nx, right)
                if ii is not None:
                    if not right:
                        return Mul(Add(*[Mul(*r) for r, c in co]), Mul(*co[0][1][:ii]))
                    else:
                        return Mul(*co[0][1][ii + len(nx):])
            beg = reduce(incommon, (n[1] for n in co))
            if beg:
                ii = find(beg, nx, right)
                if ii is not None:
                    if not right:
                        gcdc = co[0][0]
                        for i in range(1, len(co)):
                            gcdc = gcdc.intersection(co[i][0])
                            if not gcdc:
                                break
                        return Mul(*(list(gcdc) + beg[:ii]))
                    else:
                        m = ii + len(nx)
                        return Add(*[Mul(*(list(r) + n[m:])) for r, n in co])
            end = list(reversed(
                reduce(incommon, (list(reversed(n[1])) for n in co))))
            if end:
                ii = find(end, nx, right)
                if ii is not None:
                    if not right:
                        return Add(*[Mul(*(list(r) + n[:-len(end) + ii])) for r, n in co])
                    else:
                        return Mul(*end[ii + len(nx):])
            # look for single match
            hit = None
            for i, (r, n) in enumerate(co):
                ii = find(n, nx, right)
                if ii is not None:
                    if not hit:
                        hit = ii, r, n
                    else:
                        break
            else:
                if hit:
                    ii, r, n = hit
                    if not right:
                        return Mul(*(list(r) + n[:ii]))
                    else:
                        return Mul(*n[ii + len(nx):])

            return S.Zero
```
### 6 - sympy/ntheory/residue_ntheory.py:

Start line: 479, End line: 581

```python
def _sqrt_mod1(a, p, n):
    """
    Find solution to ``x**2 == a mod p**n`` when ``a % p == 0``

    see http://www.numbertheory.org/php/squareroot.html
    """
    pn = p**n
    a = a % pn
    if a == 0:
        # case gcd(a, p**k) = p**n
        m = n // 2
        if n % 2 == 1:
            pm1 = p**(m + 1)
            def _iter0a():
                i = 0
                while i < pn:
                    yield i
                    i += pm1
            return _iter0a()
        else:
            pm = p**m
            def _iter0b():
                i = 0
                while i < pn:
                    yield i
                    i += pm
            return _iter0b()

    # case gcd(a, p**k) = p**r, r < n
    f = factorint(a)
    r = f[p]
    if r % 2 == 1:
        return None
    m = r // 2
    a1 = a >> r
    if p == 2:
        if n - r == 1:
            pnm1 = 1 << (n - m + 1)
            pm1 = 1 << (m + 1)
            def _iter1():
                k = 1 << (m + 2)
                i = 1 << m
                while i < pnm1:
                    j = i
                    while j < pn:
                        yield j
                        j += k
                    i += pm1
            return _iter1()
        if n - r == 2:
            res = _sqrt_mod_prime_power(a1, p, n - r)
            if res is None:
                return None
            pnm = 1 << (n - m)
            def _iter2():
                s = set()
                for r in res:
                    i = 0
                    while i < pn:
                        x = (r << m) + i
                        if x not in s:
                            s.add(x)
                            yield x
                        i += pnm
            return _iter2()
        if n - r > 2:
            res = _sqrt_mod_prime_power(a1, p, n - r)
            if res is None:
                return None
            pnm1 = 1 << (n - m - 1)
            def _iter3():
                s = set()
                for r in res:
                    i = 0
                    while i < pn:
                        x = ((r << m) + i) % pn
                        if x not in s:
                            s.add(x)
                            yield x
                        i += pnm1
            return _iter3()
    else:
        m = r // 2
        a1 = a // p**r
        res1 = _sqrt_mod_prime_power(a1, p, n - r)
        if res1 is None:
            return None
        pm = p**m
        pnr = p**(n-r)
        pnm = p**(n-m)

        def _iter4():
            s = set()
            pm = p**m
            for rx in res1:
                i = 0
                while i < pnm:
                    x = ((rx + i) % pn)
                    if x not in s:
                        s.add(x)
                        yield x*pm
                    i += pnr
        return _iter4()
```
### 7 - sympy/integrals/meijerint.py:

Start line: 1683, End line: 1717

```python
def _meijerint_indefinite_1(f, x):
    # ... other code

    def _clean(res):
        """This multiplies out superfluous powers of x we created, and chops off
        constants:

            >> _clean(x*(exp(x)/x - 1/x) + 3)
            exp(x)

        cancel is used before mul_expand since it is possible for an
        expression to have an additive constant that doesn't become isolated
        with simple expansion. Such a situation was identified in issue 6369:


        >>> from sympy import sqrt, cancel
        >>> from sympy.abc import x
        >>> a = sqrt(2*x + 1)
        >>> bad = (3*x*a**5 + 2*x - a**5 + 1)/a**2
        >>> bad.expand().as_independent(x)[0]
        0
        >>> cancel(bad).expand().as_independent(x)[0]
        1
        """
        from sympy import cancel
        res = expand_mul(cancel(res), deep=False)
        return Add._from_args(res.as_coeff_add(x)[1])

    res = piecewise_fold(res)
    if res.is_Piecewise:
        newargs = []
        for expr, cond in res.args:
            expr = _my_unpolarify(_clean(expr))
            newargs += [(expr, cond)]
        res = Piecewise(*newargs)
    else:
        res = _my_unpolarify(_clean(res))
    return Piecewise((res, _my_unpolarify(cond)), (Integral(f, x), True))
```
### 8 - sympy/functions/special/polynomials.py:

Start line: 885, End line: 900

```python
class assoc_legendre(Function):

    @classmethod
    def eval(cls, n, m, x):
        if m.could_extract_minus_sign():
            # P^{-m}_n  --->  F * P^m_n
            return S.NegativeOne**(-m) * (factorial(m + n)/factorial(n - m)) * assoc_legendre(n, -m, x)
        if m == 0:
            # P^0_n  --->  L_n
            return legendre(n, x)
        if x == 0:
            return 2**m*sqrt(S.Pi) / (gamma((1 - m - n)/2)*gamma(1 - (m - n)/2))
        if n.is_Number and m.is_Number and n.is_integer and m.is_integer:
            if n.is_negative:
                raise ValueError("%s : 1st index must be nonnegative integer (got %r)" % (cls, n))
            if abs(m) > n:
                raise ValueError("%s : abs('2nd index') must be <= '1st index' (got %r, %r)" % (cls, n, m))
            return cls._eval_at_order(int(n), abs(int(m))).subs(_x, x)
```
### 9 - sympy/core/mul.py:

Start line: 1604, End line: 1628

```python
class Mul(Expr, AssocOp):

    def _eval_nseries(self, x, n, logx):
        from sympy import Order, powsimp
        terms = [t.nseries(x, n=n, logx=logx) for t in self.args]
        res = powsimp(self.func(*terms).expand(), combine='exp', deep=True)
        if res.has(Order):
            res += Order(x**n, x)
        return res

    def _eval_as_leading_term(self, x):
        return self.func(*[t.as_leading_term(x) for t in self.args])

    def _eval_conjugate(self):
        return self.func(*[t.conjugate() for t in self.args])

    def _eval_transpose(self):
        return self.func(*[t.transpose() for t in self.args[::-1]])

    def _eval_adjoint(self):
        return self.func(*[t.adjoint() for t in self.args[::-1]])

    def _sage_(self):
        s = 1
        for x in self.args:
            s *= x._sage_()
        return s
```
### 10 - sympy/core/exprtools.py:

Start line: 1052, End line: 1098

```python
def gcd_terms(terms, isprimitive=False, clear=True, fraction=True):
    # ... other code

    isadd = isinstance(terms, Add)
    addlike = isadd or not isinstance(terms, Basic) and \
        is_sequence(terms, include=set) and \
        not isinstance(terms, Dict)

    if addlike:
        if isadd:  # i.e. an Add
            terms = list(terms.args)
        else:
            terms = sympify(terms)
        terms, reps = mask(terms)
        cont, numer, denom = _gcd_terms(terms, isprimitive, fraction)
        numer = numer.xreplace(reps)
        coeff, factors = cont.as_coeff_Mul()
        if not clear:
            c, _coeff = coeff.as_coeff_Mul()
            if not c.is_Integer and not clear and numer.is_Add:
                n, d = c.as_numer_denom()
                _numer = numer/d
                if any(a.as_coeff_Mul()[0].is_Integer
                        for a in _numer.args):
                    numer = _numer
                    coeff = n*_coeff
        return _keep_coeff(coeff, factors*numer/denom, clear=clear)

    if not isinstance(terms, Basic):
        return terms

    if terms.is_Atom:
        return terms

    if terms.is_Mul:
        c, args = terms.as_coeff_mul()
        return _keep_coeff(c, Mul(*[gcd_terms(i, isprimitive, clear, fraction)
            for i in args]), clear=clear)

    def handle(a):
        # don't treat internal args like terms of an Add
        if not isinstance(a, Expr):
            if isinstance(a, Basic):
                return a.func(*[handle(i) for i in a.args])
            return type(a)([handle(i) for i in a])
        return gcd_terms(a, isprimitive, clear, fraction)

    if isinstance(terms, Dict):
        return Dict(*[(k, handle(v)) for k, v in terms.args])
    return terms.func(*[handle(i) for i in terms.args])
```
