# sympy__sympy-21313

| **sympy/sympy** | `546e10799fe55b3e59dea8fa6b3a6d6e71843d33` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | 12962 |
| **Avg pos** | 26.0 |
| **Min pos** | 26 |
| **Max pos** | 26 |
| **Top file pos** | 2 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sympy/sets/handlers/functions.py b/sympy/sets/handlers/functions.py
--- a/sympy/sets/handlers/functions.py
+++ b/sympy/sets/handlers/functions.py
@@ -1,4 +1,4 @@
-from sympy import Set, symbols, exp, log, S, Wild, Dummy, oo
+from sympy import Set, symbols, exp, log, S, Wild, Dummy, oo, Float
 from sympy.core import Expr, Add
 from sympy.core.function import Lambda, _coeff_isneg, FunctionClass
 from sympy.logic.boolalg import true
@@ -192,7 +192,9 @@ def _set_function(f, self): # noqa:F811
     a = Wild('a', exclude=[n])
     b = Wild('b', exclude=[n])
     match = expr.match(a*n + b)
-    if match and match[a]:
+    if match and match[a] and (
+            not match[a].atoms(Float) and
+            not match[b].atoms(Float)):
         # canonical shift
         a, b = match[a], match[b]
         if a in [1, -1]:

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sympy/sets/handlers/functions.py | 1 | 1 | - | 2 | -
| sympy/sets/handlers/functions.py | 195 | 195 | 26 | 2 | 12962


## Problem Statement

```
don't canonicalize imageset based on Float
While writing this [answer](https://stackoverflow.com/a/67053708/1089161) about how to get something resembling a float-version for range to work, I tried to think about how I would do this in SymPy. Although Floats present their own difficulties, there is canonicalization being done with `imageset` expressions that makes this even worse:
\`\`\`
>>> a,b,c = 0.092, 0.433, 0.341
>>> a in imageset(x,a+c*x,Integers)
True
>>> a in imageset(x,b+c*x,Integers)
False  <- expected based on nature of floats
>>> b in imageset(x,b+c*x,Integers)  # this should not be expected
False <- not expected
\`\`\`
That last result should represent an error. The reason it happens is because `b` is replaced with `b%c`:
\`\`\`
>>> b, round(b%c,3), imageset(x,b+c*x,Integers)
(0.433, 0.092, ImageSet(Lambda(x, 0.341*x + 0.092), Integers))
\`\`\`
So while canonicalization is OK for Rationals, it should not be done to Floats.

Working around this issue, here is a version of `frange` that might work for SymPy:
\`\`\`python
def frange(A, a, step, rational=None, _str=True):
    """return all values between `a` and `A` that are separated by `step`
    and that include `A`.

    EXAMPLES
    ========

    >>> frange(1, 3, .6)
    FiniteSet(1.0, 1.6, 2.2, 2.8)
    >>> frange(3, 1, .6)
    FiniteSet(1.2, 1.8, 2.4, 3.0)
    >>> frange(1, 3, .6, rational=True)
    FiniteSet(1, 8/5, 11/5, 14/5)

    >>> a, b, c = 0.092, 0.433, 0.341
    >>> frange(a, b, c) == frange(b, a, c) == FiniteSet(0.092, 0.433)

    Input values are parsed in WYSIWYG fashion by using Rational
    equivalents of the `str` values of the input. Note the difference
    between the last example above and those below when this is
    disabled:

    >>> frange(a, b, c, _str=False)
    FiniteSet(0.092)
    >>> frange(b, a, c, _str=False)
    FiniteSet(0.433)

    The binary representations of `a` and `b` are such that the
    difference is not `c` unless the fraction corresponding to the
    `str` values is used:

    >>> b - a == c
    False
    >>> Rational(str(b)) - Rational(str(a)) == Rational(str(c))
    True
    """
    from sympy.functions.special.gamma_functions import intlike
    if A == a:
        return S.EmptySet
    v = A, a, step
    A, B, C = [Rational(str(i) if _str else i) for i in v]
    inf = min(A, B)
    sup = max(A, B)
    rv = Interval(inf, sup).intersection(
    imageset(x, A + x*C, Integers))
    if not rational and not all(intlike(i) for i in v):
        rv = rv.func(*[float(i) for i in rv])
    return rv
\`\`\`

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 sympy/sets/handlers/intersection.py | 105 | 220| 985 | 985 | 4183 | 
| 2 | **2 sympy/sets/handlers/functions.py** | 25 | 117| 822 | 1807 | 6469 | 
| 3 | 3 sympy/sets/fancysets.py | 579 | 641| 533 | 2340 | 17359 | 
| 4 | 3 sympy/sets/fancysets.py | 494 | 577| 674 | 3014 | 17359 | 
| 5 | 3 sympy/sets/fancysets.py | 746 | 874| 1098 | 4112 | 17359 | 
| 6 | 3 sympy/sets/handlers/intersection.py | 223 | 368| 1317 | 5429 | 17359 | 
| 7 | 3 sympy/sets/handlers/intersection.py | 425 | 477| 497 | 5926 | 17359 | 
| 8 | 3 sympy/sets/handlers/intersection.py | 77 | 103| 250 | 6176 | 17359 | 
| 9 | 3 sympy/sets/fancysets.py | 665 | 691| 206 | 6382 | 17359 | 
| 10 | 4 sympy/calculus/util.py | 173 | 222| 347 | 6729 | 29242 | 
| 11 | 5 sympy/sets/sets.py | 2178 | 2271| 772 | 7501 | 46863 | 
| 12 | 6 sympy/sets/handlers/issubset.py | 103 | 140| 357 | 7858 | 48241 | 
| 13 | 6 sympy/sets/handlers/issubset.py | 50 | 67| 197 | 8055 | 48241 | 
| 14 | 6 sympy/sets/fancysets.py | 876 | 908| 185 | 8240 | 48241 | 
| 15 | 6 sympy/sets/fancysets.py | 736 | 745| 151 | 8391 | 48241 | 
| 16 | 6 sympy/sets/fancysets.py | 268 | 324| 459 | 8850 | 48241 | 
| 17 | 7 sympy/simplify/radsimp.py | 750 | 825| 756 | 9606 | 58572 | 
| 18 | 7 sympy/sets/fancysets.py | 910 | 934| 218 | 9824 | 58572 | 
| 19 | 8 sympy/functions/elementary/piecewise.py | 530 | 543| 207 | 10031 | 69945 | 
| 20 | 8 sympy/sets/fancysets.py | 1160 | 1187| 209 | 10240 | 69945 | 
| 21 | 8 sympy/sets/fancysets.py | 693 | 734| 286 | 10526 | 69945 | 
| 22 | 8 sympy/sets/handlers/issubset.py | 69 | 101| 293 | 10819 | 69945 | 
| 23 | 9 sympy/concrete/guess.py | 110 | 169| 553 | 11372 | 74914 | 
| 24 | 9 sympy/sets/handlers/intersection.py | 1 | 28| 298 | 11670 | 74914 | 
| 25 | 9 sympy/functions/elementary/piecewise.py | 545 | 629| 854 | 12524 | 74914 | 
| **-> 26 <-** | **9 sympy/sets/handlers/functions.py** | 173 | 220| 438 | 12962 | 74914 | 
| 27 | 9 sympy/sets/fancysets.py | 20 | 64| 319 | 13281 | 74914 | 
| 28 | 10 sympy/polys/modulargcd.py | 1705 | 1766| 410 | 13691 | 93390 | 
| 29 | 10 sympy/functions/elementary/piecewise.py | 1139 | 1182| 404 | 14095 | 93390 | 
| 30 | 10 sympy/functions/elementary/piecewise.py | 657 | 676| 219 | 14314 | 93390 | 
| 31 | 10 sympy/simplify/radsimp.py | 1182 | 1206| 216 | 14530 | 93390 | 
| 32 | 10 sympy/sets/fancysets.py | 426 | 476| 444 | 14974 | 93390 | 
| 33 | 11 sympy/sets/__init__.py | 1 | 36| 287 | 15261 | 93677 | 
| 34 | 12 sympy/plotting/pygletplot/plot_interval.py | 164 | 180| 142 | 15403 | 94980 | 
| 35 | 12 sympy/sets/handlers/issubset.py | 15 | 32| 198 | 15601 | 94980 | 
| 36 | 13 sympy/solvers/solvers.py | 3366 | 3483| 1593 | 17194 | 126671 | 
| 37 | 13 sympy/polys/modulargcd.py | 1219 | 1282| 480 | 17674 | 126671 | 
| 38 | 14 sympy/ntheory/continued_fraction.py | 1 | 68| 630 | 18304 | 129417 | 
| 39 | 14 sympy/sets/sets.py | 2273 | 2304| 255 | 18559 | 129417 | 
| 40 | 14 sympy/sets/sets.py | 917 | 955| 339 | 18898 | 129417 | 
| 41 | 15 sympy/core/function.py | 3285 | 3374| 789 | 19687 | 157642 | 
| 42 | 15 sympy/sets/handlers/intersection.py | 371 | 423| 417 | 20104 | 157642 | 
| 43 | 16 sympy/plotting/intervalmath/interval_arithmetic.py | 280 | 319| 342 | 20446 | 160809 | 
| 44 | 17 sympy/series/formal.py | 109 | 165| 427 | 20873 | 175004 | 
| 46 | 18 sympy/sets/fancysets.py | 643 | 663| 143 | 21823 | 199407 | 


## Patch

```diff
diff --git a/sympy/sets/handlers/functions.py b/sympy/sets/handlers/functions.py
--- a/sympy/sets/handlers/functions.py
+++ b/sympy/sets/handlers/functions.py
@@ -1,4 +1,4 @@
-from sympy import Set, symbols, exp, log, S, Wild, Dummy, oo
+from sympy import Set, symbols, exp, log, S, Wild, Dummy, oo, Float
 from sympy.core import Expr, Add
 from sympy.core.function import Lambda, _coeff_isneg, FunctionClass
 from sympy.logic.boolalg import true
@@ -192,7 +192,9 @@ def _set_function(f, self): # noqa:F811
     a = Wild('a', exclude=[n])
     b = Wild('b', exclude=[n])
     match = expr.match(a*n + b)
-    if match and match[a]:
+    if match and match[a] and (
+            not match[a].atoms(Float) and
+            not match[b].atoms(Float)):
         # canonical shift
         a, b = match[a], match[b]
         if a in [1, -1]:

```

## Test Patch

```diff
diff --git a/sympy/sets/tests/test_fancysets.py b/sympy/sets/tests/test_fancysets.py
--- a/sympy/sets/tests/test_fancysets.py
+++ b/sympy/sets/tests/test_fancysets.py
@@ -480,6 +480,9 @@ def test_Integers_eval_imageset():
     y = Symbol('y')
     L = imageset(x, 2*x + y, S.Integers)
     assert y + 4 in L
+    a, b, c = 0.092, 0.433, 0.341
+    assert a in imageset(x, a + c*x, S.Integers)
+    assert b in imageset(x, b + c*x, S.Integers)
 
     _x = symbols('x', negative=True)
     eq = _x**2 - _x + 1

```


## Code snippets

### 1 - sympy/sets/handlers/intersection.py:

Start line: 105, End line: 220

```python
@dispatch(Range, Range)  # type: ignore # noqa:F811
def intersection_sets(a, b): # noqa:F811
    from sympy.solvers.diophantine.diophantine import diop_linear
    from sympy.core.numbers import ilcm
    from sympy import sign

    # non-overlap quick exits
    if not b:
        return S.EmptySet
    if not a:
        return S.EmptySet
    if b.sup < a.inf:
        return S.EmptySet
    if b.inf > a.sup:
        return S.EmptySet

    # work with finite end at the start
    r1 = a
    if r1.start.is_infinite:
        r1 = r1.reversed
    r2 = b
    if r2.start.is_infinite:
        r2 = r2.reversed

    # If both ends are infinite then it means that one Range is just the set
    # of all integers (the step must be 1).
    if r1.start.is_infinite:
        return b
    if r2.start.is_infinite:
        return a

    # this equation represents the values of the Range;
    # it's a linear equation
    eq = lambda r, i: r.start + i*r.step

    # we want to know when the two equations might
    # have integer solutions so we use the diophantine
    # solver
    va, vb = diop_linear(eq(r1, Dummy('a')) - eq(r2, Dummy('b')))

    # check for no solution
    no_solution = va is None and vb is None
    if no_solution:
        return S.EmptySet

    # there is a solution
    # -------------------

    # find the coincident point, c
    a0 = va.as_coeff_Add()[0]
    c = eq(r1, a0)

    # find the first point, if possible, in each range
    # since c may not be that point
    def _first_finite_point(r1, c):
        if c == r1.start:
            return c
        # st is the signed step we need to take to
        # get from c to r1.start
        st = sign(r1.start - c)*step
        # use Range to calculate the first point:
        # we want to get as close as possible to
        # r1.start; the Range will not be null since
        # it will at least contain c
        s1 = Range(c, r1.start + st, st)[-1]
        if s1 == r1.start:
            pass
        else:
            # if we didn't hit r1.start then, if the
            # sign of st didn't match the sign of r1.step
            # we are off by one and s1 is not in r1
            if sign(r1.step) != sign(st):
                s1 -= st
        if s1 not in r1:
            return
        return s1

    # calculate the step size of the new Range
    step = abs(ilcm(r1.step, r2.step))
    s1 = _first_finite_point(r1, c)
    if s1 is None:
        return S.EmptySet
    s2 = _first_finite_point(r2, c)
    if s2 is None:
        return S.EmptySet

    # replace the corresponding start or stop in
    # the original Ranges with these points; the
    # result must have at least one point since
    # we know that s1 and s2 are in the Ranges
    def _updated_range(r, first):
        st = sign(r.step)*step
        if r.start.is_finite:
            rv = Range(first, r.stop, st)
        else:
            rv = Range(r.start, first + st, st)
        return rv
    r1 = _updated_range(a, s1)
    r2 = _updated_range(b, s2)

    # work with them both in the increasing direction
    if sign(r1.step) < 0:
        r1 = r1.reversed
    if sign(r2.step) < 0:
        r2 = r2.reversed

    # return clipped Range with positive step; it
    # can't be empty at this point
    start = max(r1.start, r2.start)
    stop = min(r1.stop, r2.stop)
    return Range(start, stop, step)


@dispatch(Range, Integers)  # type: ignore # noqa:F811
def intersection_sets(a, b): # noqa:F811
    return a
```
### 2 - sympy/sets/handlers/functions.py:

Start line: 25, End line: 117

```python
@dispatch(Lambda, Interval)  # type: ignore # noqa:F811
def _set_function(f, x): # noqa:F811
    from sympy.functions.elementary.miscellaneous import Min, Max
    from sympy.solvers.solveset import solveset
    from sympy.core.function import diff, Lambda
    from sympy.series import limit
    from sympy.calculus.singularities import singularities
    from sympy.sets import Complement
    # TODO: handle functions with infinitely many solutions (eg, sin, tan)
    # TODO: handle multivariate functions

    expr = f.expr
    if len(expr.free_symbols) > 1 or len(f.variables) != 1:
        return
    var = f.variables[0]
    if not var.is_real:
        if expr.subs(var, Dummy(real=True)).is_real is False:
            return

    if expr.is_Piecewise:
        result = S.EmptySet
        domain_set = x
        for (p_expr, p_cond) in expr.args:
            if p_cond is true:
                intrvl = domain_set
            else:
                intrvl = p_cond.as_set()
                intrvl = Intersection(domain_set, intrvl)

            if p_expr.is_Number:
                image = FiniteSet(p_expr)
            else:
                image = imageset(Lambda(var, p_expr), intrvl)
            result = Union(result, image)

            # remove the part which has been `imaged`
            domain_set = Complement(domain_set, intrvl)
            if domain_set is S.EmptySet:
                break
        return result

    if not x.start.is_comparable or not x.end.is_comparable:
        return

    try:
        from sympy.polys.polyutils import _nsort
        sing = list(singularities(expr, var, x))
        if len(sing) > 1:
            sing = _nsort(sing)
    except NotImplementedError:
        return

    if x.left_open:
        _start = limit(expr, var, x.start, dir="+")
    elif x.start not in sing:
        _start = f(x.start)
    if x.right_open:
        _end = limit(expr, var, x.end, dir="-")
    elif x.end not in sing:
        _end = f(x.end)

    if len(sing) == 0:
        soln_expr = solveset(diff(expr, var), var)
        if not (isinstance(soln_expr, FiniteSet) or soln_expr is EmptySet):
            return
        solns = list(soln_expr)

        extr = [_start, _end] + [f(i) for i in solns
                                 if i.is_real and i in x]
        start, end = Min(*extr), Max(*extr)

        left_open, right_open = False, False
        if _start <= _end:
            # the minimum or maximum value can occur simultaneously
            # on both the edge of the interval and in some interior
            # point
            if start == _start and start not in solns:
                left_open = x.left_open
            if end == _end and end not in solns:
                right_open = x.right_open
        else:
            if start == _end and start not in solns:
                left_open = x.right_open
            if end == _start and end not in solns:
                right_open = x.left_open

        return Interval(start, end, left_open, right_open)
    else:
        return imageset(f, Interval(x.start, sing[0],
                                    x.left_open, True)) + \
            Union(*[imageset(f, Interval(sing[i], sing[i + 1], True, True))
                    for i in range(0, len(sing) - 1)]) + \
            imageset(f, Interval(sing[-1], x.end, True, x.right_open))
```
### 3 - sympy/sets/fancysets.py:

Start line: 579, End line: 641

```python
class Range(Set):

    def __new__(cls, *args):
        from sympy.functions.elementary.integers import ceiling
        if len(args) == 1:
            if isinstance(args[0], range):
                raise TypeError(
                    'use sympify(%s) to convert range to Range' % args[0])

        # expand range
        slc = slice(*args)

        if slc.step == 0:
            raise ValueError("step cannot be 0")

        start, stop, step = slc.start or 0, slc.stop, slc.step or 1
        try:
            ok = []
            for w in (start, stop, step):
                w = sympify(w)
                if w in [S.NegativeInfinity, S.Infinity] or (
                        w.has(Symbol) and w.is_integer != False):
                    ok.append(w)
                elif not w.is_Integer:
                    raise ValueError
                else:
                    ok.append(w)
        except ValueError:
            raise ValueError(filldedent('''
            rguments to Range must be integers; `imageset` can define
            ses, e.g. use `imageset(i, i/10, Range(3))` to give
            , 1/5].'''))
        start, stop, step = ok

        null = False
        if any(i.has(Symbol) for i in (start, stop, step)):
            if start == stop:
                null = True
            else:
                end = stop
        elif start.is_infinite:
            span = step*(stop - start)
            if span is S.NaN or span <= 0:
                null = True
            elif step.is_Integer and stop.is_infinite and abs(step) != 1:
                raise ValueError(filldedent('''
                    Step size must be %s in this case.''' % (1 if step > 0 else -1)))
            else:
                end = stop
        else:
            oostep = step.is_infinite
            if oostep:
                step = S.One if step > 0 else S.NegativeOne
            n = ceiling((stop - start)/step)
            if n <= 0:
                null = True
            elif oostep:
                end = start + 1
                step = S.One  # make it a canonical single step
            else:
                end = start + n*step
        if null:
            start = end = S.Zero
            step = S.One
        return Basic.__new__(cls, start, end, step)
```
### 4 - sympy/sets/fancysets.py:

Start line: 494, End line: 577

```python
class Range(Set):
    """
    Represents a range of integers. Can be called as Range(stop),
    Range(start, stop), or Range(start, stop, step); when step is
    not given it defaults to 1.

    `Range(stop)` is the same as `Range(0, stop, 1)` and the stop value
    (juse as for Python ranges) is not included in the Range values.

        >>> from sympy import Range
        >>> list(Range(3))
        [0, 1, 2]

    The step can also be negative:

        >>> list(Range(10, 0, -2))
        [10, 8, 6, 4, 2]

    The stop value is made canonical so equivalent ranges always
    have the same args:

        >>> Range(0, 10, 3)
        Range(0, 12, 3)

    Infinite ranges are allowed. ``oo`` and ``-oo`` are never included in the
    set (``Range`` is always a subset of ``Integers``). If the starting point
    is infinite, then the final value is ``stop - step``. To iterate such a
    range, it needs to be reversed:

        >>> from sympy import oo
        >>> r = Range(-oo, 1)
        >>> r[-1]
        0
        >>> next(iter(r))
        Traceback (most recent call last):
        ...
        TypeError: Cannot iterate over Range with infinite start
        >>> next(iter(r.reversed))
        0

    Although Range is a set (and supports the normal set
    operations) it maintains the order of the elements and can
    be used in contexts where `range` would be used.

        >>> from sympy import Interval
        >>> Range(0, 10, 2).intersect(Interval(3, 7))
        Range(4, 8, 2)
        >>> list(_)
        [4, 6]

    Although slicing of a Range will always return a Range -- possibly
    empty -- an empty set will be returned from any intersection that
    is empty:

        >>> Range(3)[:0]
        Range(0, 0, 1)
        >>> Range(3).intersect(Interval(4, oo))
        EmptySet
        >>> Range(3).intersect(Range(4, oo))
        EmptySet

    Range will accept symbolic arguments but has very limited support
    for doing anything other than displaying the Range:

        >>> from sympy import Symbol, pprint
        >>> from sympy.abc import i, j, k
        >>> Range(i, j, k).start
        i
        >>> Range(i, j, k).inf
        Traceback (most recent call last):
        ...
        ValueError: invalid method for symbolic range

    Better success will be had when using integer symbols:

        >>> n = Symbol('n', integer=True)
        >>> r = Range(n, n + 20, 3)
        >>> r.inf
        n
        >>> pprint(r)
        {n, n + 3, ..., n + 17}
    """

    is_iterable = True
```
### 5 - sympy/sets/fancysets.py:

Start line: 746, End line: 874

```python
class Range(Set):

    def __getitem__(self, i):
        # ... other code
        if isinstance(i, slice):
            if self.size.is_finite:  # validates, too
                start, stop, step = i.indices(self.size)
                n = ceiling((stop - start)/step)
                if n <= 0:
                    return Range(0)
                canonical_stop = start + n*step
                end = canonical_stop - step
                ss = step*self.step
                return Range(self[start], self[end] + ss, ss)
            else:  # infinite Range
                start = i.start
                stop = i.stop
                if i.step == 0:
                    raise ValueError(zerostep)
                step = i.step or 1
                ss = step*self.step
                #---------------------
                # handle infinite Range
                #   i.e. Range(-oo, oo) or Range(oo, -oo, -1)
                # --------------------
                if self.start.is_infinite and self.stop.is_infinite:
                    raise ValueError(infinite)
                #---------------------
                # handle infinite on right
                #   e.g. Range(0, oo) or Range(0, -oo, -1)
                # --------------------
                if self.stop.is_infinite:
                    # start and stop are not interdependent --
                    # they only depend on step --so we use the
                    # equivalent reversed values
                    return self.reversed[
                        stop if stop is None else -stop + 1:
                        start if start is None else -start:
                        step].reversed
                #---------------------
                # handle infinite on the left
                #   e.g. Range(oo, 0, -1) or Range(-oo, 0)
                # --------------------
                # consider combinations of
                # start/stop {== None, < 0, == 0, > 0} and
                # step {< 0, > 0}
                if start is None:
                    if stop is None:
                        if step < 0:
                            return Range(self[-1], self.start, ss)
                        elif step > 1:
                            raise ValueError(ambiguous)
                        else:  # == 1
                            return self
                    elif stop < 0:
                        if step < 0:
                            return Range(self[-1], self[stop], ss)
                        else:  # > 0
                            return Range(self.start, self[stop], ss)
                    elif stop == 0:
                        if step > 0:
                            return Range(0)
                        else:  # < 0
                            raise ValueError(ooslice)
                    elif stop == 1:
                        if step > 0:
                            raise ValueError(ooslice)  # infinite singleton
                        else:  # < 0
                            raise ValueError(ooslice)
                    else:  # > 1
                        raise ValueError(ooslice)
                elif start < 0:
                    if stop is None:
                        if step < 0:
                            return Range(self[start], self.start, ss)
                        else:  # > 0
                            return Range(self[start], self.stop, ss)
                    elif stop < 0:
                        return Range(self[start], self[stop], ss)
                    elif stop == 0:
                        if step < 0:
                            raise ValueError(ooslice)
                        else:  # > 0
                            return Range(0)
                    elif stop > 0:
                        raise ValueError(ooslice)
                elif start == 0:
                    if stop is None:
                        if step < 0:
                            raise ValueError(ooslice)  # infinite singleton
                        elif step > 1:
                            raise ValueError(ambiguous)
                        else:  # == 1
                            return self
                    elif stop < 0:
                        if step > 1:
                            raise ValueError(ambiguous)
                        elif step == 1:
                            return Range(self.start, self[stop], ss)
                        else:  # < 0
                            return Range(0)
                    else:  # >= 0
                        raise ValueError(ooslice)
                elif start > 0:
                    raise ValueError(ooslice)
        else:
            if not self:
                raise IndexError('Range index out of range')
            if i == 0:
                if self.start.is_infinite:
                    raise ValueError(ooslice)
                if self.has(Symbol):
                    if (self.stop > self.start) == self.step.is_positive and self.step.is_positive is not None:
                        pass
                    else:
                        _ = self.size  # validate
                return self.start
            if i == -1:
                if self.stop.is_infinite:
                    raise ValueError(ooslice)
                n = self.stop - self.step
                if n.is_Integer or (
                        n.is_integer and (
                            (n - self.start).is_nonnegative ==
                            self.step.is_positive)):
                    return n
            _ = self.size  # validate
            rv = (self.stop if i < 0 else self.start) + i*self.step
            if rv.is_infinite:
                raise ValueError(ooslice)
            if rv < self.inf or rv > self.sup:
                raise IndexError("Range index out of range")
            return rv
```
### 6 - sympy/sets/handlers/intersection.py:

Start line: 223, End line: 368

```python
@dispatch(ImageSet, Set)  # type: ignore # noqa:F811
def intersection_sets(self, other): # noqa:F811
    from sympy.solvers.diophantine import diophantine

    # Only handle the straight-forward univariate case
    if (len(self.lamda.variables) > 1
            or self.lamda.signature != self.lamda.variables):
        return None
    base_set = self.base_sets[0]

    # Intersection between ImageSets with Integers as base set
    # For {f(n) : n in Integers} & {g(m) : m in Integers} we solve the
    # diophantine equations f(n)=g(m).
    # If the solutions for n are {h(t) : t in Integers} then we return
    # {f(h(t)) : t in integers}.
    # If the solutions for n are {n_1, n_2, ..., n_k} then we return
    # {f(n_i) : 1 <= i <= k}.
    if base_set is S.Integers:
        gm = None
        if isinstance(other, ImageSet) and other.base_sets == (S.Integers,):
            gm = other.lamda.expr
            var = other.lamda.variables[0]
            # Symbol of second ImageSet lambda must be distinct from first
            m = Dummy('m')
            gm = gm.subs(var, m)
        elif other is S.Integers:
            m = gm = Dummy('m')
        if gm is not None:
            fn = self.lamda.expr
            n = self.lamda.variables[0]
            try:
                solns = list(diophantine(fn - gm, syms=(n, m), permute=True))
            except (TypeError, NotImplementedError):
                # TypeError if equation not polynomial with rational coeff.
                # NotImplementedError if correct format but no solver.
                return
            # 3 cases are possible for solns:
            # - empty set,
            # - one or more parametric (infinite) solutions,
            # - a finite number of (non-parametric) solution couples.
            # Among those, there is one type of solution set that is
            # not helpful here: multiple parametric solutions.
            if len(solns) == 0:
                return EmptySet
            elif any(not isinstance(s, int) and s.free_symbols
                     for tupl in solns for s in tupl):
                if len(solns) == 1:
                    soln, solm = solns[0]
                    (t,) = soln.free_symbols
                    expr = fn.subs(n, soln.subs(t, n)).expand()
                    return imageset(Lambda(n, expr), S.Integers)
                else:
                    return
            else:
                return FiniteSet(*(fn.subs(n, s[0]) for s in solns))

    if other == S.Reals:
        from sympy.solvers.solveset import solveset_real
        from sympy.core.function import expand_complex

        f = self.lamda.expr
        n = self.lamda.variables[0]

        n_ = Dummy(n.name, real=True)
        f_ = f.subs(n, n_)

        re, im = f_.as_real_imag()
        im = expand_complex(im)

        re = re.subs(n_, n)
        im = im.subs(n_, n)
        ifree = im.free_symbols
        lam = Lambda(n, re)
        if not im:
            # allow re-evaluation
            # of self in this case to make
            # the result canonical
            pass
        elif im.is_zero is False:
            return S.EmptySet
        elif ifree != {n}:
            return None
        else:
            # univarite imaginary part in same variable
            base_set = base_set.intersect(solveset_real(im, n))
        return imageset(lam, base_set)

    elif isinstance(other, Interval):
        from sympy.solvers.solveset import (invert_real, invert_complex,
                                            solveset)

        f = self.lamda.expr
        n = self.lamda.variables[0]
        new_inf, new_sup = None, None
        new_lopen, new_ropen = other.left_open, other.right_open

        if f.is_real:
            inverter = invert_real
        else:
            inverter = invert_complex

        g1, h1 = inverter(f, other.inf, n)
        g2, h2 = inverter(f, other.sup, n)

        if all(isinstance(i, FiniteSet) for i in (h1, h2)):
            if g1 == n:
                if len(h1) == 1:
                    new_inf = h1.args[0]
            if g2 == n:
                if len(h2) == 1:
                    new_sup = h2.args[0]
            # TODO: Design a technique to handle multiple-inverse
            # functions

            # Any of the new boundary values cannot be determined
            if any(i is None for i in (new_sup, new_inf)):
                return


            range_set = S.EmptySet

            if all(i.is_real for i in (new_sup, new_inf)):
                # this assumes continuity of underlying function
                # however fixes the case when it is decreasing
                if new_inf > new_sup:
                    new_inf, new_sup = new_sup, new_inf
                new_interval = Interval(new_inf, new_sup, new_lopen, new_ropen)
                range_set = base_set.intersect(new_interval)
            else:
                if other.is_subset(S.Reals):
                    solutions = solveset(f, n, S.Reals)
                    if not isinstance(range_set, (ImageSet, ConditionSet)):
                        range_set = solutions.intersect(other)
                    else:
                        return

            if range_set is S.EmptySet:
                return S.EmptySet
            elif isinstance(range_set, Range) and range_set.size is not S.Infinity:
                range_set = FiniteSet(*list(range_set))

            if range_set is not None:
                return imageset(Lambda(n, f), range_set)
            return
        else:
            return
```
### 7 - sympy/sets/handlers/intersection.py:

Start line: 425, End line: 477

```python
@dispatch(type(EmptySet), Set)  # type: ignore # noqa:F811
def intersection_sets(a, b): # noqa:F811
    return S.EmptySet

@dispatch(UniversalSet, Set)  # type: ignore # noqa:F811
def intersection_sets(a, b): # noqa:F811
    return b

@dispatch(FiniteSet, FiniteSet)  # type: ignore # noqa:F811
def intersection_sets(a, b): # noqa:F811
    return FiniteSet(*(a._elements & b._elements))

@dispatch(FiniteSet, Set)  # type: ignore # noqa:F811
def intersection_sets(a, b): # noqa:F811
    try:
        return FiniteSet(*[el for el in a if el in b])
    except TypeError:
        return None  # could not evaluate `el in b` due to symbolic ranges.

@dispatch(Set, Set)  # type: ignore # noqa:F811
def intersection_sets(a, b): # noqa:F811
    return None

@dispatch(Integers, Rationals)  # type: ignore # noqa:F811
def intersection_sets(a, b): # noqa:F811
    return a

@dispatch(Naturals, Rationals)  # type: ignore # noqa:F811
def intersection_sets(a, b): # noqa:F811
    return a

@dispatch(Rationals, Reals)  # type: ignore # noqa:F811
def intersection_sets(a, b): # noqa:F811
    return a

def _intlike_interval(a, b):
    try:
        from sympy.functions.elementary.integers import floor, ceiling
        if b._inf is S.NegativeInfinity and b._sup is S.Infinity:
            return a
        s = Range(max(a.inf, ceiling(b.left)), floor(b.right) + 1)
        return intersection_sets(s, b)  # take out endpoints if open interval
    except ValueError:
        return None

@dispatch(Integers, Interval)  # type: ignore # noqa:F811
def intersection_sets(a, b): # noqa:F811
    return _intlike_interval(a, b)

@dispatch(Naturals, Interval)  # type: ignore # noqa:F811
def intersection_sets(a, b): # noqa:F811
    return _intlike_interval(a, b)
```
### 8 - sympy/sets/handlers/intersection.py:

Start line: 77, End line: 103

```python
@dispatch(Integers, Reals)  # type: ignore # noqa:F811
def intersection_sets(a, b): # noqa:F811
    return a

@dispatch(Range, Interval)  # type: ignore # noqa:F811
def intersection_sets(a, b): # noqa:F811
    from sympy.functions.elementary.integers import floor, ceiling
    if not all(i.is_number for i in b.args[:2]):
        return

    # In case of null Range, return an EmptySet.
    if a.size == 0:
        return S.EmptySet

    # trim down to self's size, and represent
    # as a Range with step 1.
    start = ceiling(max(b.inf, a.inf))
    if start not in b:
        start += 1
    end = floor(min(b.sup, a.sup))
    if end not in b:
        end -= 1
    return intersection_sets(a, Range(start, end + 1))

@dispatch(Range, Naturals)  # type: ignore # noqa:F811
def intersection_sets(a, b): # noqa:F811
    return intersection_sets(a, Interval(b.inf, S.Infinity))
```
### 9 - sympy/sets/fancysets.py:

Start line: 665, End line: 691

```python
class Range(Set):

    def _contains(self, other):
        if not self:
            return S.false
        if other.is_infinite:
            return S.false
        if not other.is_integer:
            return other.is_integer
        if self.has(Symbol):
            try:
                _ = self.size  # validate
            except ValueError:
                return
        if self.start.is_finite:
            ref = self.start
        elif self.stop.is_finite:
            ref = self.stop
        else:  # both infinite; step is +/- 1 (enforced by __new__)
            return S.true
        if self.size == 1:
            return Eq(other, self[0])
        res = (ref - other) % self.step
        if res == S.Zero:
            return And(other >= self.inf, other <= self.sup)
        elif res.is_Integer:  # off sequence
            return S.false
        else:  # symbolic/unsimplified residue modulo step
            return None
```
### 10 - sympy/calculus/util.py:

Start line: 173, End line: 222

```python
def function_range(f, symbol, domain):
    # ... other code

    for interval in interval_iter:
        if isinstance(interval, FiniteSet):
            for singleton in interval:
                if singleton in domain:
                    range_int += FiniteSet(f.subs(symbol, singleton))
        elif isinstance(interval, Interval):
            vals = S.EmptySet
            critical_points = S.EmptySet
            critical_values = S.EmptySet
            bounds = ((interval.left_open, interval.inf, '+'),
                   (interval.right_open, interval.sup, '-'))

            for is_open, limit_point, direction in bounds:
                if is_open:
                    critical_values += FiniteSet(limit(f, symbol, limit_point, direction))
                    vals += critical_values

                else:
                    vals += FiniteSet(f.subs(symbol, limit_point))

            solution = solveset(f.diff(symbol), symbol, interval)

            if not iterable(solution):
                raise NotImplementedError(
                        'Unable to find critical points for {}'.format(f))
            if isinstance(solution, ImageSet):
                raise NotImplementedError(
                        'Infinite number of critical points for {}'.format(f))

            critical_points += solution

            for critical_point in critical_points:
                vals += FiniteSet(f.subs(symbol, critical_point))

            left_open, right_open = False, False

            if critical_values is not S.EmptySet:
                if critical_values.inf == vals.inf:
                    left_open = True

                if critical_values.sup == vals.sup:
                    right_open = True

            range_int += Interval(vals.inf, vals.sup, left_open, right_open)
        else:
            raise NotImplementedError(filldedent('''
                Unable to find range for the given domain.
                '''))

    return range_int
```
### 26 - sympy/sets/handlers/functions.py:

Start line: 173, End line: 220

```python
@dispatch(FunctionUnion, Integers)  # type: ignore # noqa:F811
def _set_function(f, self): # noqa:F811
    expr = f.expr
    if not isinstance(expr, Expr):
        return

    n = f.variables[0]
    if expr == abs(n):
        return S.Naturals0

    # f(x) + c and f(-x) + c cover the same integers
    # so choose the form that has the fewest negatives
    c = f(0)
    fx = f(n) - c
    f_x = f(-n) - c
    neg_count = lambda e: sum(_coeff_isneg(_) for _ in Add.make_args(e))
    if neg_count(f_x) < neg_count(fx):
        expr = f_x + c

    a = Wild('a', exclude=[n])
    b = Wild('b', exclude=[n])
    match = expr.match(a*n + b)
    if match and match[a]:
        # canonical shift
        a, b = match[a], match[b]
        if a in [1, -1]:
            # drop integer addends in b
            nonint = []
            for bi in Add.make_args(b):
                if not bi.is_integer:
                    nonint.append(bi)
            b = Add(*nonint)
        if b.is_number and a.is_real:
            # avoid Mod for complex numbers, #11391
            br, bi = match_real_imag(b)
            if br and br.is_comparable and a.is_comparable:
                br %= a
                b = br + S.ImaginaryUnit*bi
        elif b.is_number and a.is_imaginary:
            br, bi = match_real_imag(b)
            ai = a/S.ImaginaryUnit
            if bi and bi.is_comparable and ai.is_comparable:
                bi %= ai
                b = br + S.ImaginaryUnit*bi
        expr = a*n + b

    if expr != f.expr:
        return ImageSet(Lambda(n, expr), S.Integers)
```
