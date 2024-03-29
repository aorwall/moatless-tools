# sympy__sympy-13915

| **sympy/sympy** | `5c1644ff85e15752f9f8721bc142bfbf975e7805` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | - |
| **Avg pos** | - |
| **Min pos** | - |
| **Max pos** | - |
| **Top file pos** | - |
| **Missing snippets** | 1 |
| **Missing patch files** | 1 |


## Expected patch

```diff
diff --git a/sympy/core/mul.py b/sympy/core/mul.py
--- a/sympy/core/mul.py
+++ b/sympy/core/mul.py
@@ -423,6 +423,11 @@ def _gather(c_powers):
             changed = False
             for b, e in c_powers:
                 if e.is_zero:
+                    # canceling out infinities yields NaN
+                    if (b.is_Add or b.is_Mul) and any(infty in b.args
+                        for infty in (S.ComplexInfinity, S.Infinity,
+                                      S.NegativeInfinity)):
+                        return [S.NaN], [], None
                     continue
                 if e is S.One:
                     if b.is_Number:

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sympy/core/mul.py | 426 | 426 | - | - | -


## Problem Statement

```
Issue with a substitution that leads to an undefined expression
\`\`\`
Python 3.6.4 |Anaconda custom (64-bit)| (default, Dec 21 2017, 15:39:08) 
Type 'copyright', 'credits' or 'license' for more information
IPython 6.2.1 -- An enhanced Interactive Python. Type '?' for help.

In [1]: from sympy import *

In [2]: a,b = symbols('a,b')

In [3]: r = (1/(a+b) + 1/(a-b))/(1/(a+b) - 1/(a-b))

In [4]: r.subs(b,a)
Out[4]: 1

In [6]: import sympy

In [7]: sympy.__version__
Out[7]: '1.1.1'
\`\`\`

If b is substituted by a, r is undefined. It is possible to calculate the limit
`r.limit(b,a) # -1`

But whenever a subexpression of r is undefined, r itself is undefined.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 sympy/integrals/integrals.py | 243 | 320| 780 | 780 | 12172 | 
| 2 | 2 sympy/concrete/expr_with_limits.py | 292 | 367| 728 | 1508 | 15782 | 
| 3 | 2 sympy/integrals/integrals.py | 322 | 353| 296 | 1804 | 15782 | 
| 4 | 3 sympy/core/function.py | 1755 | 1805| 504 | 2308 | 39855 | 
| 5 | 4 sympy/solvers/solvers.py | 451 | 901| 4430 | 6738 | 70986 | 
| 6 | 5 sympy/core/basic.py | 727 | 839| 1037 | 7775 | 85491 | 
| 7 | 6 sympy/strategies/rl.py | 115 | 161| 273 | 8048 | 86630 | 
| 8 | 7 sympy/concrete/summations.py | 910 | 994| 683 | 8731 | 96482 | 
| 9 | 7 sympy/concrete/expr_with_limits.py | 238 | 290| 470 | 9201 | 96482 | 
| 10 | 7 sympy/core/function.py | 1712 | 1754| 393 | 9594 | 96482 | 
| 11 | 7 sympy/core/function.py | 1884 | 1908| 282 | 9876 | 96482 | 
| 12 | 8 sympy/strategies/tools.py | 1 | 23| 192 | 10068 | 96849 | 
| 13 | 9 sympy/core/sympify.py | 78 | 259| 1755 | 11823 | 100825 | 
| 14 | 9 sympy/core/function.py | 1869 | 1882| 216 | 12039 | 100825 | 
| 15 | 9 sympy/concrete/summations.py | 857 | 907| 469 | 12508 | 100825 | 
| 16 | 9 sympy/core/basic.py | 840 | 923| 691 | 13199 | 100825 | 
| 17 | 10 sympy/core/expr.py | 1948 | 1972| 159 | 13358 | 129509 | 
| 18 | 10 sympy/core/function.py | 1807 | 1867| 405 | 13763 | 129509 | 
| 19 | 11 sympy/functions/elementary/piecewise.py | 538 | 551| 212 | 13975 | 138907 | 
| 20 | 12 sympy/solvers/solveset.py | 1628 | 1652| 290 | 14265 | 159154 | 
| 21 | 13 sympy/stats/rv.py | 1078 | 1109| 190 | 14455 | 166626 | 
| 22 | 14 sympy/concrete/expr_with_intlimits.py | 1 | 102| 1134 | 15589 | 169446 | 
| 23 | 15 sympy/series/limits.py | 159 | 227| 546 | 16135 | 171292 | 
| 24 | 15 sympy/core/basic.py | 925 | 995| 637 | 16772 | 171292 | 
| 25 | 16 sympy/simplify/gammasimp.py | 476 | 513| 276 | 17048 | 175534 | 
| 26 | 17 sympy/physics/mechanics/functions.py | 549 | 569| 248 | 17296 | 180864 | 
| 27 | 17 sympy/core/expr.py | 3431 | 3450| 165 | 17461 | 180864 | 
| 28 | 17 sympy/series/limits.py | 111 | 156| 337 | 17798 | 180864 | 
| 29 | 17 sympy/concrete/summations.py | 1051 | 1103| 481 | 18279 | 180864 | 
| 30 | 18 sympy/core/power.py | 884 | 1004| 1059 | 19338 | 194699 | 


## Missing Patch Files

 * 1: sympy/core/mul.py

### Hint

```
In this regard, don't you think that `r.simplify()` is wrong? It returns `-a/b` which is not correct if b=a.
`simplify` works for the generic case. SymPy would be hard to use if getting a+b from `simplify((a**2-b**2)/(a-b))` required an explicit declaration that a is not equal to b. (Besides, there is currently no way to express that declaration to `simplify`, anyway). This is part of reason we avoid `simplify` in code:  it can change the outcome in edge cases. 

The fundamental issue here is: for what kind of expression `expr` do we want expr/expr to return 1? Current behavior:

zoo / zoo   # nan
(zoo + 3) / (zoo + 3)   # nan
(zoo + a) / (zoo + a)    # 1  
(zoo + a) / (a - zoo)   # 1 because -zoo is zoo  (zoo is complex infinity)  

The rules for combining an expression with its inverse in Mul appear to be too lax. 

There is a check of the form `if something is S.ComplexInfinity`... which returns nan in the first two cases, but this condition is not met by `zoo + a`. 

But using something like `numerator.is_finite` would not work either, because most of the time, we don't know if a symbolic expression is finite. E.g., `(a+b).is_finite` is None, unknown,  unless the symbols were explicitly declared to be finite.

My best idea so far is to have three cases for expr/expr: 

1. expr is infinite or 0: return nan
2. Otherwise, if expr contains infinities (how to check this efficiently? Mul needs to be really fast), return expr/expr without combining 
3. Otherwise, return 1
"But using something like numerator.is_finite would not work either"

I had thought of something like denom.is_zero. If in expr_1/expr_2 the denominator is zero, the fraction is undefined. The only way to get a value from this is to use limits. At least i would think so.

My first idea was that sympy first simplifies and then substitutes. But then, the result should be -1. 

(zoo+a)/(a-zoo) # 1
explains what happens, but i had expected, that
zoo/expr leads to nan and expr/zoo leads to nan as well.

I agree, that Mul needs to be really fast, but this is about subst. But i confess, i don't know much about symbolic math.
zoo/3 is zoo, and 4/zoo is 0. I think it's convenient, and not controversial, to have these. 

Substitution is not to blame: it replaces b by a as requested, evaluating 1/(a-a) as zoo.  This is how `r` becomes `(1/(2*a) + zoo) / (1/(2*a) - zoo)`. So far nothing wrong has happened. The problem is that (because of -zoo being same as zoo) both parts are identified as the same and then the `_gather` helper of Mul method combines the powers 1 and -1 into power 0. And anything to power 0 returns 1 in SymPy, hence the result. 

I think we should prevent combining powers when base contains Infinity or ComplexInfinity. For example, (x+zoo) / (x+zoo)**2  returning 1 / (x+zoo) isn't right either. 
I dont really understand what happens. How can i get the result zoo? 

In my example `r.subs(b,a)` returns ` 1`,  
but `r.subs(b,-a)` returns `(zoo + 1/(2*a))/(zoo - 1/(2*a))`

So how is zoo defined? Is it `(1/z).limit(z,0)`? I get `oo` as result, but how is this related to  `zoo`? As far as i know, `zoo` is ComplexInfinity. By playing around, i just found another confusing result:

`(zoo+z)/(zoo-z)` returns `(z + zoo)/(-z + zoo)`, 
but
`(z + zoo)/(z-zoo)` returns 1

I just found, `1/S.Zero` returns `zoo`, as well as `(1/S.Zero)**2`. To me, that would mean i should not divide by `zoo`.
There are three infinities: positive infinity oo, negative infinity -oo, and complex infinity zoo. Here is the difference:

- If z is a positive number that tends to zero, then 1/z tends to oo
- if z is a negative number than tends to zero, then 1/z tends to -oo
- If z is a complex number that tends to zero, then 1/z tends to zoo

The complex infinity zoo does not have a determined sign, so -zoo is taken to  be the same as zoo. So when you put `(z + zoo)/(z-zoo)` two things happen: first, z-zoo returns z+zoo (you can check this directly) and second, the two identical expressions are cancelled, leaving 1.

However, in (zoo+z)/(zoo-z) the terms are not identical, so they do not cancel. 

I am considering a solution that returns NaN when Mul cancels an expression with infinity of any kind. So for example (z+zoo)/(z+zoo) and (z-oo)/(z-oo) both return NaN. However, it changes the behavior in a couple of tests, so I have to investigate whether the tests are being wrong about infinities, or something else is. 
Ok. I think i got it. Thank you for your patient explanation. 
Maybe one last question. Should `z + zoo` result in `zoo`? I think that would be natural.
```

## Patch

```diff
diff --git a/sympy/core/mul.py b/sympy/core/mul.py
--- a/sympy/core/mul.py
+++ b/sympy/core/mul.py
@@ -423,6 +423,11 @@ def _gather(c_powers):
             changed = False
             for b, e in c_powers:
                 if e.is_zero:
+                    # canceling out infinities yields NaN
+                    if (b.is_Add or b.is_Mul) and any(infty in b.args
+                        for infty in (S.ComplexInfinity, S.Infinity,
+                                      S.NegativeInfinity)):
+                        return [S.NaN], [], None
                     continue
                 if e is S.One:
                     if b.is_Number:

```

## Test Patch

```diff
diff --git a/sympy/core/tests/test_arit.py b/sympy/core/tests/test_arit.py
--- a/sympy/core/tests/test_arit.py
+++ b/sympy/core/tests/test_arit.py
@@ -1,7 +1,7 @@
 from __future__ import division
 
 from sympy import (Basic, Symbol, sin, cos, exp, sqrt, Rational, Float, re, pi,
-        sympify, Add, Mul, Pow, Mod, I, log, S, Max, symbols, oo, Integer,
+        sympify, Add, Mul, Pow, Mod, I, log, S, Max, symbols, oo, zoo, Integer,
         sign, im, nan, Dummy, factorial, comp, refine
 )
 from sympy.core.compatibility import long, range
@@ -1937,6 +1937,14 @@ def test_Mul_with_zero_infinite():
     assert e.is_positive is None
     assert e.is_hermitian is None
 
+def test_Mul_does_not_cancel_infinities():
+    a, b = symbols('a b')
+    assert ((zoo + 3*a)/(3*a + zoo)) is nan
+    assert ((b - oo)/(b - oo)) is nan
+    # issue 13904
+    expr = (1/(a+b) + 1/(a-b))/(1/(a+b) - 1/(a-b))
+    assert expr.subs(b, a) is nan
+
 def test_issue_8247_8354():
     from sympy import tan
     z = sqrt(1 + sqrt(3)) + sqrt(3 + 3*sqrt(3)) - sqrt(10 + 6*sqrt(3))

```


## Code snippets

### 1 - sympy/integrals/integrals.py:

Start line: 243, End line: 320

```python
class Integral(AddWithLimits):

    def transform(self, x, u):
        from sympy.solvers.solvers import solve, posify
        d = Dummy('d')

        xfree = x.free_symbols.intersection(self.variables)
        if len(xfree) > 1:
            raise ValueError(
                'F(x) can only contain one of: %s' % self.variables)
        xvar = xfree.pop() if xfree else d

        if xvar not in self.variables:
            return self

        u = sympify(u)
        if isinstance(u, Expr):
            ufree = u.free_symbols
            if len(ufree) != 1:
                raise ValueError(filldedent('''
                When f(u) has more than one free symbol, the one replacing x
                must be identified: pass f(u) as (f(u), u)'''))
            uvar = ufree.pop()
        else:
            u, uvar = u
            if uvar not in u.free_symbols:
                raise ValueError(filldedent('''
                Expecting a tuple (expr, symbol) where symbol identified
                a free symbol in expr, but symbol is not in expr's free
                symbols.'''))
            if not isinstance(uvar, Symbol):
                raise ValueError(filldedent('''
                Expecting a tuple (expr, symbol) but didn't get
                a symbol; got %s''' % uvar))

        if x.is_Symbol and u.is_Symbol:
            return self.xreplace({x: u})

        if not x.is_Symbol and not u.is_Symbol:
            raise ValueError('either x or u must be a symbol')

        if uvar == xvar:
            return self.transform(x, (u.subs(uvar, d), d)).xreplace({d: uvar})

        if uvar in self.limits:
            raise ValueError(filldedent('''
            u must contain the same variable as in x
            or a variable that is not already an integration variable'''))

        if not x.is_Symbol:
            F = [x.subs(xvar, d)]
            soln = solve(u - x, xvar, check=False)
            if not soln:
                raise ValueError('no solution for solve(F(x) - f(u), x)')
            f = [fi.subs(uvar, d) for fi in soln]
        else:
            f = [u.subs(uvar, d)]
            pdiff, reps = posify(u - x)
            puvar = uvar.subs([(v, k) for k, v in reps.items()])
            soln = [s.subs(reps) for s in solve(pdiff, puvar)]
            if not soln:
                raise ValueError('no solution for solve(F(x) - f(u), u)')
            F = [fi.subs(xvar, d) for fi in soln]

        newfuncs = set([(self.function.subs(xvar, fi)*fi.diff(d)
                        ).subs(d, uvar) for fi in f])
        if len(newfuncs) > 1:
            raise ValueError(filldedent('''
            The mapping between F(x) and f(u) did not give
            a unique integrand.'''))
        newfunc = newfuncs.pop()

        def _calc_limit_1(F, a, b):
            """
            replace d with a, using subs if possible, otherwise limit
            where sign of b is considered
            """
            wok = F.subs(d, a)
            if wok is S.NaN or wok.is_finite is False and a.is_finite:
                return limit(sign(b)*F, d, a)
            return wok
        # ... other code
```
### 2 - sympy/concrete/expr_with_limits.py:

Start line: 292, End line: 367

```python
class ExprWithLimits(Expr):

    def _eval_subs(self, old, new):
        """
        Perform substitutions over non-dummy variables
        of an expression with limits.  Also, can be used
        to specify point-evaluation of an abstract antiderivative.

        Examples
        ========

        >>> from sympy import Sum, oo
        >>> from sympy.abc import s, n
        >>> Sum(1/n**s, (n, 1, oo)).subs(s, 2)
        Sum(n**(-2), (n, 1, oo))

        >>> from sympy import Integral
        >>> from sympy.abc import x, a
        >>> Integral(a*x**2, x).subs(x, 4)
        Integral(a*x**2, (x, 4))

        See Also
        ========

        variables : Lists the integration variables
        transform : Perform mapping on the dummy variable for integrals
        change_index : Perform mapping on the sum and product dummy variables

        """
        from sympy.core.function import AppliedUndef, UndefinedFunction
        func, limits = self.function, list(self.limits)

        # If one of the expressions we are replacing is used as a func index
        # one of two things happens.
        #   - the old variable first appears as a free variable
        #     so we perform all free substitutions before it becomes
        #     a func index.
        #   - the old variable first appears as a func index, in
        #     which case we ignore.  See change_index.

        # Reorder limits to match standard mathematical practice for scoping
        limits.reverse()

        if not isinstance(old, Symbol) or \
                old.free_symbols.intersection(self.free_symbols):
            sub_into_func = True
            for i, xab in enumerate(limits):
                if 1 == len(xab) and old == xab[0]:
                    xab = (old, old)
                limits[i] = Tuple(xab[0], *[l._subs(old, new) for l in xab[1:]])
                if len(xab[0].free_symbols.intersection(old.free_symbols)) != 0:
                    sub_into_func = False
                    break
            if isinstance(old, AppliedUndef) or isinstance(old, UndefinedFunction):
                sy2 = set(self.variables).intersection(set(new.atoms(Symbol)))
                sy1 = set(self.variables).intersection(set(old.args))
                if not sy2.issubset(sy1):
                    raise ValueError(
                        "substitution can not create dummy dependencies")
                sub_into_func = True
            if sub_into_func:
                func = func.subs(old, new)
        else:
            # old is a Symbol and a dummy variable of some limit
            for i, xab in enumerate(limits):
                if len(xab) == 3:
                    limits[i] = Tuple(xab[0], *[l._subs(old, new) for l in xab[1:]])
                    if old == xab[0]:
                        break
        # simplify redundant limits (x, x)  to (x, )
        for i, xab in enumerate(limits):
            if len(xab) == 2 and (xab[0] - xab[1]).is_zero:
                limits[i] = Tuple(xab[0], )

        # Reorder limits back to representation-form
        limits.reverse()

        return self.func(func, *limits)
```
### 3 - sympy/integrals/integrals.py:

Start line: 322, End line: 353

```python
class Integral(AddWithLimits):

    def transform(self, x, u):
        # ... other code

        def _calc_limit(a, b):
            """
            replace d with a, using subs if possible, otherwise limit
            where sign of b is considered
            """
            avals = list({_calc_limit_1(Fi, a, b) for Fi in F})
            if len(avals) > 1:
                raise ValueError(filldedent('''
                The mapping between F(x) and f(u) did not
                give a unique limit.'''))
            return avals[0]

        newlimits = []
        for xab in self.limits:
            sym = xab[0]
            if sym == xvar:
                if len(xab) == 3:
                    a, b = xab[1:]
                    a, b = _calc_limit(a, b), _calc_limit(b, a)
                    if a - b > 0:
                        a, b = b, a
                        newfunc = -newfunc
                    newlimits.append((uvar, a, b))
                elif len(xab) == 2:
                    a = _calc_limit(xab[1], 1)
                    newlimits.append((uvar, a))
                else:
                    newlimits.append(uvar)
            else:
                newlimits.append(xab)

        return self.func(newfunc, *newlimits)
```
### 4 - sympy/core/function.py:

Start line: 1755, End line: 1805

```python
class Subs(Expr):
    def __new__(cls, expr, variables, point, **assumptions):
        from sympy import Symbol
        if not is_sequence(variables, Tuple):
            variables = [variables]
        variables = list(sympify(variables))

        if list(uniq(variables)) != variables:
            repeated = [ v for v in set(variables) if variables.count(v) > 1 ]
            raise ValueError('cannot substitute expressions %s more than '
                             'once.' % repeated)

        point = Tuple(*(point if is_sequence(point, Tuple) else [point]))

        if len(point) != len(variables):
            raise ValueError('Number of point values must be the same as '
                             'the number of variables.')

        expr = sympify(expr)

        # use symbols with names equal to the point value (with preppended _)
        # to give a variable-independent expression
        pre = "_"
        pts = sorted(set(point), key=default_sort_key)
        from sympy.printing import StrPrinter
        class CustomStrPrinter(StrPrinter):
            def _print_Dummy(self, expr):
                return str(expr) + str(expr.dummy_index)
        def mystr(expr, **settings):
            p = CustomStrPrinter(settings)
            return p.doprint(expr)
        while 1:
            s_pts = {p: Symbol(pre + mystr(p)) for p in pts}
            reps = [(v, s_pts[p])
                for v, p in zip(variables, point)]
            # if any underscore-preppended symbol is already a free symbol
            # and is a variable with a different point value, then there
            # is a clash, e.g. _0 clashes in Subs(_0 + _1, (_0, _1), (1, 0))
            # because the new symbol that would be created is _1 but _1
            # is already mapped to 0 so __0 and __1 are used for the new
            # symbols
            if any(r in expr.free_symbols and
                   r in variables and
                   Symbol(pre + mystr(point[variables.index(r)])) != r
                   for _, r in reps):
                pre += "_"
                continue
            break

        obj = Expr.__new__(cls, expr, Tuple(*variables), point)
        obj._expr = expr.subs(reps)
        return obj
```
### 5 - sympy/solvers/solvers.py:

Start line: 451, End line: 901

```python
def solve(f, *symbols, **flags):
    r"""
    Algebraically solves equations and systems of equations.

    Currently supported are:
        - polynomial,
        - transcendental
        - piecewise combinations of the above
        - systems of linear and polynomial equations
        - systems containing relational expressions.

    Input is formed as:

    * f
        - a single Expr or Poly that must be zero,
        - an Equality
        - a Relational expression or boolean
        - iterable of one or more of the above

    * symbols (object(s) to solve for) specified as
        - none given (other non-numeric objects will be used)
        - single symbol
        - denested list of symbols
          e.g. solve(f, x, y)
        - ordered iterable of symbols
          e.g. solve(f, [x, y])

    * flags
        'dict'=True (default is False)
            return list (perhaps empty) of solution mappings
        'set'=True (default is False)
            return list of symbols and set of tuple(s) of solution(s)
        'exclude=[] (default)'
            don't try to solve for any of the free symbols in exclude;
            if expressions are given, the free symbols in them will
            be extracted automatically.
        'check=True (default)'
            If False, don't do any testing of solutions. This can be
            useful if one wants to include solutions that make any
            denominator zero.
        'numerical=True (default)'
            do a fast numerical check if ``f`` has only one symbol.
        'minimal=True (default is False)'
            a very fast, minimal testing.
        'warn=True (default is False)'
            show a warning if checksol() could not conclude.
        'simplify=True (default)'
            simplify all but polynomials of order 3 or greater before
            returning them and (if check is not False) use the
            general simplify function on the solutions and the
            expression obtained when they are substituted into the
            function which should be zero
        'force=True (default is False)'
            make positive all symbols without assumptions regarding sign.
        'rational=True (default)'
            recast Floats as Rational; if this option is not used, the
            system containing floats may fail to solve because of issues
            with polys. If rational=None, Floats will be recast as
            rationals but the answer will be recast as Floats. If the
            flag is False then nothing will be done to the Floats.
        'manual=True (default is False)'
            do not use the polys/matrix method to solve a system of
            equations, solve them one at a time as you might "manually"
        'implicit=True (default is False)'
            allows solve to return a solution for a pattern in terms of
            other functions that contain that pattern; this is only
            needed if the pattern is inside of some invertible function
            like cos, exp, ....
        'particular=True (default is False)'
            instructs solve to try to find a particular solution to a linear
            system with as many zeros as possible; this is very expensive
        'quick=True (default is False)'
            when using particular=True, use a fast heuristic instead to find a
            solution with many zeros (instead of using the very slow method
            guaranteed to find the largest number of zeros possible)
        'cubics=True (default)'
            return explicit solutions when cubic expressions are encountered
        'quartics=True (default)'
            return explicit solutions when quartic expressions are encountered
        'quintics=True (default)'
            return explicit solutions (if possible) when quintic expressions
            are encountered

    Examples
    ========

    The output varies according to the input and can be seen by example::

        >>> from sympy import solve, Poly, Eq, Function, exp
        >>> from sympy.abc import x, y, z, a, b
        >>> f = Function('f')

    * boolean or univariate Relational

        >>> solve(x < 3)
        (-oo < x) & (x < 3)


    * to always get a list of solution mappings, use flag dict=True

        >>> solve(x - 3, dict=True)
        [{x: 3}]
        >>> sol = solve([x - 3, y - 1], dict=True)
        >>> sol
        [{x: 3, y: 1}]
        >>> sol[0][x]
        3
        >>> sol[0][y]
        1


    * to get a list of symbols and set of solution(s) use flag set=True

        >>> solve([x**2 - 3, y - 1], set=True)
        ([x, y], {(-sqrt(3), 1), (sqrt(3), 1)})


    * single expression and single symbol that is in the expression

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

    * single expression with no symbol that is in the expression

        >>> solve(3, x)
        []
        >>> solve(x - 3, y)
        []

    * single expression with no symbol given

          In this case, all free symbols will be selected as potential
          symbols to solve for. If the equation is univariate then a list
          of solutions is returned; otherwise -- as is the case when symbols are
          given as an iterable of length > 1 -- a list of mappings will be returned.

            >>> solve(x - 3)
            [3]
            >>> solve(x**2 - y**2)
            [{x: -y}, {x: y}]
            >>> solve(z**2*x**2 - z**2*y**2)
            [{x: -y}, {x: y}, {z: 0}]
            >>> solve(z**2*x - z**2*y**2)
            [{x: y**2}, {z: 0}]

    * when an object other than a Symbol is given as a symbol, it is
      isolated algebraically and an implicit solution may be obtained.
      This is mostly provided as a convenience to save one from replacing
      the object with a Symbol and solving for that Symbol. It will only
      work if the specified object can be replaced with a Symbol using the
      subs method.

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

        * To solve for a *symbol* implicitly, use 'implicit=True':

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
            {y: -sqrt(3)/(x + 3), x + 2: (-2*x - 6 + sqrt(3))/(x + 3)}
            >>> solve(eqs, y*x, x)
            {x: -y - 4, x*y: -3*y - sqrt(3)}

        * if you attempt to solve for a number remember that the number
          you have obtained does not necessarily mean that the value is
          equivalent to the expression obtained:

            >>> solve(sqrt(2) - 1, 1)
            [sqrt(2)]
            >>> solve(x - y + 1, 1)  # /!\ -1 is targeted, too
            [x/(y - 1)]
            >>> [_.subs(z, -1) for _ in solve((x - y + 1).subs(-1, z), 1)]
            [-x + y]

        * To solve for a function within a derivative, use dsolve.

    * single expression and more than 1 symbol

        * when there is a linear solution

            >>> solve(x - y**2, x, y)
            [{x: y**2}]
            >>> solve(x**2 - y, x, y)
            [{y: x**2}]

        * when undetermined coefficients are identified

            * that are linear

                >>> solve((a + b)*x - b + 2, a, b)
                {a: -2, b: 2}

            * that are nonlinear

                >>> solve((a + b)*x - b**2 + 2, a, b, set=True)
                ([a, b], {(-sqrt(2), sqrt(2)), (sqrt(2), -sqrt(2))})

        * if there is no linear solution then the first successful
          attempt for a nonlinear solution will be returned

            >>> solve(x**2 - y**2, x, y)
            [{x: -y}, {x: y}]
            >>> solve(x**2 - y**2/exp(x), x, y)
            [{x: 2*LambertW(y/2)}]
            >>> solve(x**2 - y**2/exp(x), y, x)
            [{y: -x*sqrt(exp(x))}, {y: x*sqrt(exp(x))}]

    * iterable of one or more of the above

        * involving relationals or bools

            >>> solve([x < 3, x - 2])
            Eq(x, 2)
            >>> solve([x > 3, x - 2])
            False

        * when the system is linear

            * with a solution

                >>> solve([x - 3], x)
                {x: 3}
                >>> solve((x + 5*y - 2, -3*x + 6*y - 15), x, y)
                {x: -3, y: 1}
                >>> solve((x + 5*y - 2, -3*x + 6*y - 15), x, y, z)
                {x: -3, y: 1}
                >>> solve((x + 5*y - 2, -3*x + 6*y - z), z, x, y)
                {x: -5*y + 2, z: 21*y - 6}

            * without a solution

                >>> solve([x + 3, x - 3])
                []

        * when the system is not linear

            >>> solve([x**2 + y -2, y**2 - 4], x, y, set=True)
            ([x, y], {(-2, -2), (0, 2), (2, -2)})

        * if no symbols are given, all free symbols will be selected and a list
          of mappings returned

            >>> solve([x - 2, x**2 + y])
            [{x: 2, y: -4}]
            >>> solve([x - 2, x**2 + f(x)], {f(x), x})
            [{x: 2, f(x): -4}]

        * if any equation doesn't depend on the symbol(s) given it will be
          eliminated from the equation set and an answer may be given
          implicitly in terms of variables that were not of interest

            >>> solve([x - y, y - 3], x)
            {x: y}

    Notes
    =====

    solve() with check=True (default) will run through the symbol tags to
    elimate unwanted solutions.  If no assumptions are included all possible
    solutions will be returned.

        >>> from sympy import Symbol, solve
        >>> x = Symbol("x")
        >>> solve(x**2 - 1)
        [-1, 1]

    By using the positive tag only one solution will be returned:

        >>> pos = Symbol("pos", positive=True)
        >>> solve(pos**2 - 1)
        [1]


    Assumptions aren't checked when `solve()` input involves
    relationals or bools.

    When the solutions are checked, those that make any denominator zero
    are automatically excluded. If you do not want to exclude such solutions
    then use the check=False option:

        >>> from sympy import sin, limit
        >>> solve(sin(x)/x)  # 0 is excluded
        [pi]

    If check=False then a solution to the numerator being zero is found: x = 0.
    In this case, this is a spurious solution since sin(x)/x has the well known
    limit (without dicontinuity) of 1 at x = 0:

        >>> solve(sin(x)/x, check=False)
        [0, pi]

    In the following case, however, the limit exists and is equal to the the
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

    Disabling high-order, explicit solutions
    ----------------------------------------

    When solving polynomial expressions, one might not want explicit solutions
    (which can be quite long). If the expression is univariate, CRootOf
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

    Solving equations involving radicals
    ------------------------------------

    Because of SymPy's use of the principle root (issue #8789), some solutions
    to radical equations will be missed unless check=False:

        >>> from sympy import root
        >>> eq = root(x**3 - 3*x**2, 3) + 1 - x
        >>> solve(eq)
        []
        >>> solve(eq, check=False)
        [1/3]

    In the above example there is only a single solution to the equation. Other
    expressions will yield spurious roots which must be checked manually;
    roots which give a negative argument to odd-powered radicals will also need
    special checking:

        >>> from sympy import real_root, S
        >>> eq = root(x, 3) - root(x, 5) + S(1)/7
        >>> solve(eq)  # this gives 2 solutions but misses a 3rd
        [CRootOf(7*_p**5 - 7*_p**3 + 1, 1)**15,
        CRootOf(7*_p**5 - 7*_p**3 + 1, 2)**15]
        >>> sol = solve(eq, check=False)
        >>> [abs(eq.subs(x,i).n(2)) for i in sol]
        [0.48, 0.e-110, 0.e-110, 0.052, 0.052]

        The first solution is negative so real_root must be used to see that
        it satisfies the expression:

        >>> abs(real_root(eq.subs(x, sol[0])).n(2))
        0.e-110

    If the roots of the equation are not real then more care will be necessary
    to find the roots, especially for higher order equations. Consider the
    following expression:

        >>> expr = root(x, 3) - root(x, 5)

    We will construct a known value for this expression at x = 3 by selecting
    the 1-th root for each radical:

        >>> expr1 = root(x, 3, 1) - root(x, 5, 1)
        >>> v = expr1.subs(x, -3)

    The solve function is unable to find any exact roots to this equation:

        >>> eq = Eq(expr, v); eq1 = Eq(expr1, v)
        >>> solve(eq, check=False), solve(eq1, check=False)
        ([], [])

    The function unrad, however, can be used to get a form of the equation for
    which numerical roots can be found:

        >>> from sympy.solvers.solvers import unrad
        >>> from sympy import nroots
        >>> e, (p, cov) = unrad(eq)
        >>> pvals = nroots(e)
        >>> inversion = solve(cov, x)[0]
        >>> xvals = [inversion.subs(p, i) for i in pvals]

    Although eq or eq1 could have been used to find xvals, the solution can
    only be verified with expr1:

        >>> z = expr - v
        >>> [xi.n(chop=1e-9) for xi in xvals if abs(z.subs(x, xi).n()) < 1e-9]
        []
        >>> z1 = expr1 - v
        >>> [xi.n(chop=1e-9) for xi in xvals if abs(z1.subs(x, xi).n()) < 1e-9]
        [-3.0]

    See Also
    ========

        - rsolve() for solving recurrence relationships
        - dsolve() for solving differential equations

    """
    # ... other code
```
### 6 - sympy/core/basic.py:

Start line: 727, End line: 839

```python
class Basic(with_metaclass(ManagedProperties)):

    def subs(self, *args, **kwargs):
        """
        Substitutes old for new in an expression after sympifying args.

        `args` is either:
          - two arguments, e.g. foo.subs(old, new)
          - one iterable argument, e.g. foo.subs(iterable). The iterable may be
             o an iterable container with (old, new) pairs. In this case the
               replacements are processed in the order given with successive
               patterns possibly affecting replacements already made.
             o a dict or set whose key/value items correspond to old/new pairs.
               In this case the old/new pairs will be sorted by op count and in
               case of a tie, by number of args and the default_sort_key. The
               resulting sorted list is then processed as an iterable container
               (see previous).

        If the keyword ``simultaneous`` is True, the subexpressions will not be
        evaluated until all the substitutions have been made.

        Examples
        ========

        >>> from sympy import pi, exp, limit, oo
        >>> from sympy.abc import x, y
        >>> (1 + x*y).subs(x, pi)
        pi*y + 1
        >>> (1 + x*y).subs({x:pi, y:2})
        1 + 2*pi
        >>> (1 + x*y).subs([(x, pi), (y, 2)])
        1 + 2*pi
        >>> reps = [(y, x**2), (x, 2)]
        >>> (x + y).subs(reps)
        6
        >>> (x + y).subs(reversed(reps))
        x**2 + 2

        >>> (x**2 + x**4).subs(x**2, y)
        y**2 + y

        To replace only the x**2 but not the x**4, use xreplace:

        >>> (x**2 + x**4).xreplace({x**2: y})
        x**4 + y

        To delay evaluation until all substitutions have been made,
        set the keyword ``simultaneous`` to True:

        >>> (x/y).subs([(x, 0), (y, 0)])
        0
        >>> (x/y).subs([(x, 0), (y, 0)], simultaneous=True)
        nan

        This has the added feature of not allowing subsequent substitutions
        to affect those already made:

        >>> ((x + y)/y).subs({x + y: y, y: x + y})
        1
        >>> ((x + y)/y).subs({x + y: y, y: x + y}, simultaneous=True)
        y/(x + y)

        In order to obtain a canonical result, unordered iterables are
        sorted by count_op length, number of arguments and by the
        default_sort_key to break any ties. All other iterables are left
        unsorted.

        >>> from sympy import sqrt, sin, cos
        >>> from sympy.abc import a, b, c, d, e

        >>> A = (sqrt(sin(2*x)), a)
        >>> B = (sin(2*x), b)
        >>> C = (cos(2*x), c)
        >>> D = (x, d)
        >>> E = (exp(x), e)

        >>> expr = sqrt(sin(2*x))*sin(exp(x)*x)*cos(2*x) + sin(2*x)

        >>> expr.subs(dict([A, B, C, D, E]))
        a*c*sin(d*e) + b

        The resulting expression represents a literal replacement of the
        old arguments with the new arguments. This may not reflect the
        limiting behavior of the expression:

        >>> (x**3 - 3*x).subs({x: oo})
        nan

        >>> limit(x**3 - 3*x, x, oo)
        oo

        If the substitution will be followed by numerical
        evaluation, it is better to pass the substitution to
        evalf as

        >>> (1/x).evalf(subs={x: 3.0}, n=21)
        0.333333333333333333333

        rather than

        >>> (1/x).subs({x: 3.0}).evalf(21)
        0.333333333333333314830

        as the former will ensure that the desired level of precision is
        obtained.

        See Also
        ========
        replace: replacement capable of doing wildcard-like matching,
                 parsing of match, and conditional replacements
        xreplace: exact node replacement in expr tree; also capable of
                  using matching rules
        evalf: calculates the given formula to a desired level of precision

        """
        # ... other code
```
### 7 - sympy/strategies/rl.py:

Start line: 115, End line: 161

```python
def subs(a, b):
    """ Replace expressions exactly """
    def subs_rl(expr):
        if expr == a:
            return b
        else:
            return expr
    return subs_rl

# Functions that are rules

def unpack(expr):
    """ Rule to unpack singleton args

    >>> from sympy.strategies import unpack
    >>> from sympy import Basic
    >>> unpack(Basic(2))
    2
    """
    if len(expr.args) == 1:
        return expr.args[0]
    else:
        return expr

def flatten(expr, new=new):
    """ Flatten T(a, b, T(c, d), T2(e)) to T(a, b, c, d, T2(e)) """
    cls = expr.__class__
    args = []
    for arg in expr.args:
        if arg.__class__ == cls:
            args.extend(arg.args)
        else:
            args.append(arg)
    return new(expr.__class__, *args)

def rebuild(expr):
    """ Rebuild a SymPy tree

    This function recursively calls constructors in the expression tree.
    This forces canonicalization and removes ugliness introduced by the use of
    Basic.__new__
    """
    try:
        return type(expr)(*list(map(rebuild, expr.args)))
    except Exception:
        return expr
```
### 8 - sympy/concrete/summations.py:

Start line: 910, End line: 994

```python
def eval_sum_symbolic(f, limits):
    from sympy.functions import harmonic, bernoulli

    f_orig = f
    (i, a, b) = limits
    if not f.has(i):
        return f*(b - a + 1)

    # Linearity
    if f.is_Mul:
        L, R = f.as_two_terms()

        if not L.has(i):
            sR = eval_sum_symbolic(R, (i, a, b))
            if sR:
                return L*sR

        if not R.has(i):
            sL = eval_sum_symbolic(L, (i, a, b))
            if sL:
                return R*sL

        try:
            f = apart(f, i)  # see if it becomes an Add
        except PolynomialError:
            pass

    if f.is_Add:
        L, R = f.as_two_terms()
        lrsum = telescopic(L, R, (i, a, b))

        if lrsum:
            return lrsum

        lsum = eval_sum_symbolic(L, (i, a, b))
        rsum = eval_sum_symbolic(R, (i, a, b))

        if None not in (lsum, rsum):
            r = lsum + rsum
            if not r is S.NaN:
                return r

    # Polynomial terms with Faulhaber's formula
    n = Wild('n')
    result = f.match(i**n)

    if result is not None:
        n = result[n]

        if n.is_Integer:
            if n >= 0:
                if (b is S.Infinity and not a is S.NegativeInfinity) or \
                   (a is S.NegativeInfinity and not b is S.Infinity):
                    return S.Infinity
                return ((bernoulli(n + 1, b + 1) - bernoulli(n + 1, a))/(n + 1)).expand()
            elif a.is_Integer and a >= 1:
                if n == -1:
                    return harmonic(b) - harmonic(a - 1)
                else:
                    return harmonic(b, abs(n)) - harmonic(a - 1, abs(n))

    if not (a.has(S.Infinity, S.NegativeInfinity) or
            b.has(S.Infinity, S.NegativeInfinity)):
        # Geometric terms
        c1 = Wild('c1', exclude=[i])
        c2 = Wild('c2', exclude=[i])
        c3 = Wild('c3', exclude=[i])

        e = f.match(c1**(c2*i + c3))

        if e is not None:
            p = (c1**c3).subs(e)
            q = (c1**c2).subs(e)

            r = p*(q**a - q**(b + 1))/(1 - q)
            l = p*(b - a + 1)

            return Piecewise((l, Eq(q, S.One)), (r, True))

        r = gosper_sum(f, (i, a, b))

        if not r in (None, S.NaN):
            return r

    return eval_sum_hyper(f_orig, (i, a, b))
```
### 9 - sympy/concrete/expr_with_limits.py:

Start line: 238, End line: 290

```python
class ExprWithLimits(Expr):

    def as_dummy(self):
        """
        Replace instances of the given dummy variables with explicit dummy
        counterparts to make clear what are dummy variables and what
        are real-world symbols in an object.

        Examples
        ========

        >>> from sympy import Integral
        >>> from sympy.abc import x, y
        >>> Integral(x, (x, x, y), (y, x, y)).as_dummy()
        Integral(_x, (_x, x, _y), (_y, x, y))

        If the object supperts the "integral at" limit ``(x,)`` it
        is not treated as a dummy, but the explicit form, ``(x, x)``
        of length 2 does treat the variable as a dummy.

        >>> Integral(x, x).as_dummy()
        Integral(x, x)
        >>> Integral(x, (x, x)).as_dummy()
        Integral(_x, (_x, x))

        If there were no dummies in the original expression, then the
        the symbols which cannot be changed by subs() are clearly seen as
        those with an underscore prefix.

        See Also
        ========

        variables : Lists the integration variables
        transform : Perform mapping on the integration variable
        """
        reps = {}
        f = self.function
        limits = list(self.limits)
        for i in range(-1, -len(limits) - 1, -1):
            xab = list(limits[i])
            if len(xab) == 1:
                continue
            x = xab[0]
            xab[0] = x.as_dummy()
            for j in range(1, len(xab)):
                xab[j] = xab[j].subs(reps)
            reps[x] = xab[0]
            limits[i] = xab
        f = f.subs(reps)
        return self.func(f, *limits)

    def _eval_interval(self, x, a, b):
        limits = [(i if i[0] != x else (x, a, b)) for i in self.limits]
        integrand = self.function
        return self.func(integrand, *limits)
```
### 10 - sympy/core/function.py:

Start line: 1712, End line: 1754

```python
class Subs(Expr):
    """
    Represents unevaluated substitutions of an expression.

    ``Subs(expr, x, x0)`` receives 3 arguments: an expression, a variable or
    list of distinct variables and a point or list of evaluation points
    corresponding to those variables.

    ``Subs`` objects are generally useful to represent unevaluated derivatives
    calculated at a point.

    The variables may be expressions, but they are subjected to the limitations
    of subs(), so it is usually a good practice to use only symbols for
    variables, since in that case there can be no ambiguity.

    There's no automatic expansion - use the method .doit() to effect all
    possible substitutions of the object and also of objects inside the
    expression.

    When evaluating derivatives at a point that is not a symbol, a Subs object
    is returned. One is also able to calculate derivatives of Subs objects - in
    this case the expression is always expanded (for the unevaluated form, use
    Derivative()).

    A simple example:

    >>> from sympy import Subs, Function, sin
    >>> from sympy.abc import x, y, z
    >>> f = Function('f')
    >>> e = Subs(f(x).diff(x), x, y)
    >>> e.subs(y, 0)
    Subs(Derivative(f(x), x), (x,), (0,))
    >>> e.subs(f, sin).doit()
    cos(y)

    An example with several variables:

    >>> Subs(f(x)*sin(y) + z, (x, y), (0, 1))
    Subs(z + f(x)*sin(y), (x, y), (0, 1))
    >>> _.doit()
    z + f(0)*sin(1)

    """
```
