# sympy__sympy-11831

| **sympy/sympy** | `9ce74956ad542e069a9a7743bf0a751c5a26e727` |
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
diff --git a/sympy/sets/sets.py b/sympy/sets/sets.py
--- a/sympy/sets/sets.py
+++ b/sympy/sets/sets.py
@@ -665,6 +665,11 @@ def _measure(self):
     def __len__(self):
         return Mul(*[len(s) for s in self.args])
 
+    def __bool__(self):
+        return all([bool(s) for s in self.args])
+
+    __nonzero__ = __bool__
+
 
 class Interval(Set, EvalfMixin):
     """

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sympy/sets/sets.py | 668 | 668 | - | - | -


## Problem Statement

```
set intersection gives TypeError: object of type 'Naturals0' has no len()
This is from https://stackoverflow.com/questions/40441532/how-to-restrict-sympy-finiteset-containing-symbol

\`\`\`
In [47]: d = symbols("d")

In [48]: solution = sets.FiniteSet((d + 1, -d + 4, -d + 5, d))

In [49]: solution.intersect(S.Naturals0**4)
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-49-a152e62d0932> in <module>()
----> 1 solution.intersect(S.Naturals0**4)

/Users/aaronmeurer/Documents/Python/sympy/sympy/sympy/sets/sets.py in intersect(self, other)
    106
    107         """
--> 108         return Intersection(self, other)
    109
    110     def intersection(self, other):

/Users/aaronmeurer/Documents/Python/sympy/sympy/sympy/sets/sets.py in __new__(cls, *args, **kwargs)
   1401         # Reduce sets using known rules
   1402         if evaluate:
-> 1403             return Intersection.reduce(args)
   1404
   1405         return Basic.__new__(cls, *args)

/Users/aaronmeurer/Documents/Python/sympy/sympy/sympy/sets/sets.py in reduce(args)
   1525
   1526         # Handle Finite sets
-> 1527         rv = Intersection._handle_finite_sets(args)
   1528         if rv is not None:
   1529             return rv

/Users/aaronmeurer/Documents/Python/sympy/sympy/sympy/sets/sets.py in _handle_finite_sets(args)
   1499
   1500             other_sets = Intersection(*other)
-> 1501             if not other_sets:
   1502                 return S.EmptySet  # b/c we use evaluate=False below
   1503             res += Intersection(

/Users/aaronmeurer/Documents/Python/sympy/sympy/sympy/sets/sets.py in __len__(self)
    664
    665     def __len__(self):
--> 666         return Mul(*[len(s) for s in self.args])
    667
    668

/Users/aaronmeurer/Documents/Python/sympy/sympy/sympy/sets/sets.py in <listcomp>(.0)
    664
    665     def __len__(self):
--> 666         return Mul(*[len(s) for s in self.args])
    667
    668

TypeError: object of type 'Naturals0' has no len()
\`\`\`

Optimistically marking this as easy to fix (I could be wrong). 

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 sympy/sets/fancysets.py | 627 | 653| 215 | 215 | 11281 | 
| 2 | 2 sympy/solvers/solveset.py | 1457 | 1482| 273 | 488 | 30442 | 
| 3 | 2 sympy/sets/fancysets.py | 18 | 72| 346 | 834 | 30442 | 
| 4 | 2 sympy/sets/fancysets.py | 1 | 15| 165 | 999 | 30442 | 
| 5 | 2 sympy/sets/fancysets.py | 655 | 758| 881 | 1880 | 30442 | 
| 6 | 2 sympy/sets/fancysets.py | 418 | 491| 597 | 2477 | 30442 | 
| 7 | 2 sympy/solvers/solveset.py | 835 | 901| 478 | 2955 | 30442 | 
| 8 | 2 sympy/sets/fancysets.py | 760 | 805| 318 | 3273 | 30442 | 
| 9 | 2 sympy/sets/fancysets.py | 75 | 92| 123 | 3396 | 30442 | 
| 10 | 2 sympy/sets/fancysets.py | 171 | 210| 321 | 3717 | 30442 | 
| 11 | 2 sympy/solvers/solveset.py | 1237 | 1306| 457 | 4174 | 30442 | 
| 12 | 2 sympy/solvers/solveset.py | 655 | 734| 790 | 4964 | 30442 | 
| 13 | 2 sympy/solvers/solveset.py | 1785 | 1845| 585 | 5549 | 30442 | 
| 14 | 2 sympy/sets/fancysets.py | 295 | 381| 729 | 6278 | 30442 | 
| 15 | 2 sympy/sets/fancysets.py | 383 | 416| 339 | 6617 | 30442 | 
| 16 | 2 sympy/solvers/solveset.py | 1920 | 2074| 1939 | 8556 | 30442 | 
| 17 | 2 sympy/solvers/solveset.py | 1846 | 1871| 202 | 8758 | 30442 | 
| 18 | 2 sympy/sets/fancysets.py | 1390 | 1468| 571 | 9329 | 30442 | 
| 19 | 2 sympy/solvers/solveset.py | 1093 | 1235| 1361 | 10690 | 30442 | 
| 20 | 2 sympy/solvers/solveset.py | 1887 | 2131| 236 | 10926 | 30442 | 
| 21 | 3 sympy/solvers/solvers.py | 1125 | 1235| 834 | 11760 | 60025 | 
| 22 | 3 sympy/solvers/solveset.py | 1557 | 1580| 221 | 11981 | 60025 | 
| 23 | 3 sympy/solvers/solveset.py | 2075 | 2132| 464 | 12445 | 60025 | 
| 24 | 4 sympy/geometry/entity.py | 493 | 516| 202 | 12647 | 64577 | 
| 25 | 4 sympy/solvers/solvers.py | 962 | 1058| 795 | 13442 | 64577 | 
| 26 | 4 sympy/sets/fancysets.py | 816 | 923| 923 | 14365 | 64577 | 
| 27 | 5 sympy/sets/conditionset.py | 34 | 61| 253 | 14618 | 65066 | 
| 28 | 5 sympy/solvers/solveset.py | 1872 | 1886| 149 | 14767 | 65066 | 
| 29 | 5 sympy/sets/fancysets.py | 925 | 975| 313 | 15080 | 65066 | 
| 30 | 5 sympy/solvers/solvers.py | 358 | 800| 4409 | 19489 | 65066 | 
| 31 | 5 sympy/solvers/solvers.py | 888 | 961| 703 | 20192 | 65066 | 
| 32 | 5 sympy/sets/fancysets.py | 95 | 169| 441 | 20633 | 65066 | 
| 33 | 5 sympy/solvers/solvers.py | 1328 | 1383| 448 | 21081 | 65066 | 
| 34 | 6 sympy/calculus/util.py | 1129 | 1171| 316 | 21397 | 74276 | 
| 35 | 6 sympy/solvers/solveset.py | 737 | 834| 764 | 22161 | 74276 | 
| 36 | 7 sympy/core/numbers.py | 2813 | 2878| 481 | 22642 | 101155 | 
| 37 | 7 sympy/solvers/solveset.py | 1532 | 1556| 290 | 22932 | 101155 | 
| 38 | 7 sympy/sets/fancysets.py | 558 | 605| 406 | 23338 | 101155 | 
| 39 | 7 sympy/solvers/solveset.py | 282 | 310| 254 | 23592 | 101155 | 
| 40 | 7 sympy/solvers/solveset.py | 1632 | 1648| 174 | 23766 | 101155 | 
| 41 | 7 sympy/solvers/solvers.py | 801 | 887| 826 | 24592 | 101155 | 
| 42 | 7 sympy/core/numbers.py | 3125 | 3168| 336 | 24928 | 101155 | 
| 43 | 7 sympy/solvers/solveset.py | 1400 | 1455| 459 | 25387 | 101155 | 
| 44 | 7 sympy/solvers/solveset.py | 628 | 653| 263 | 25650 | 101155 | 
| 45 | 7 sympy/solvers/solvers.py | 1745 | 1837| 822 | 26472 | 101155 | 
| 46 | 7 sympy/solvers/solvers.py | 1059 | 1123| 597 | 27069 | 101155 | 
| 47 | 7 sympy/core/numbers.py | 2602 | 2667| 477 | 27546 | 101155 | 
| 48 | 7 sympy/solvers/solvers.py | 1577 | 1636| 515 | 28061 | 101155 | 
| 49 | 7 sympy/solvers/solveset.py | 1484 | 1531| 470 | 28531 | 101155 | 
| 50 | 7 sympy/core/numbers.py | 3064 | 3102| 231 | 28762 | 101155 | 
| 51 | 7 sympy/solvers/solvers.py | 64 | 76| 132 | 28894 | 101155 | 
| 52 | 7 sympy/solvers/solvers.py | 217 | 299| 688 | 29582 | 101155 | 
| 53 | 7 sympy/sets/fancysets.py | 807 | 815| 137 | 29719 | 101155 | 
| 54 | 7 sympy/calculus/util.py | 154 | 221| 563 | 30282 | 101155 | 
| 55 | 7 sympy/calculus/util.py | 779 | 833| 516 | 30798 | 101155 | 
| 56 | 8 sympy/combinatorics/tensor_can.py | 417 | 530| 1374 | 32172 | 114302 | 
| 57 | 8 sympy/solvers/solvers.py | 1639 | 1743| 729 | 32901 | 114302 | 
| 58 | 8 sympy/solvers/solveset.py | 1650 | 1784| 1342 | 34243 | 114302 | 
| 59 | 8 sympy/calculus/util.py | 864 | 933| 557 | 34800 | 114302 | 
| 60 | 8 sympy/core/numbers.py | 2746 | 2775| 191 | 34991 | 114302 | 
| 61 | 8 sympy/sets/fancysets.py | 213 | 293| 639 | 35630 | 114302 | 
| 62 | 9 sympy/diffgeom/diffgeom.py | 1 | 19| 171 | 35801 | 128763 | 
| 63 | 9 sympy/solvers/solveset.py | 1309 | 1398| 1009 | 36810 | 128763 | 


## Missing Patch Files

 * 1: sympy/sets/sets.py

## Patch

```diff
diff --git a/sympy/sets/sets.py b/sympy/sets/sets.py
--- a/sympy/sets/sets.py
+++ b/sympy/sets/sets.py
@@ -665,6 +665,11 @@ def _measure(self):
     def __len__(self):
         return Mul(*[len(s) for s in self.args])
 
+    def __bool__(self):
+        return all([bool(s) for s in self.args])
+
+    __nonzero__ = __bool__
+
 
 class Interval(Set, EvalfMixin):
     """

```

## Test Patch

```diff
diff --git a/sympy/sets/tests/test_sets.py b/sympy/sets/tests/test_sets.py
--- a/sympy/sets/tests/test_sets.py
+++ b/sympy/sets/tests/test_sets.py
@@ -963,6 +963,8 @@ def test_issue_Symbol_inter():
     assert Intersection(FiniteSet(x**2, 1, sin(x)), FiniteSet(x**2, 2, sin(x)), r) == \
         Intersection(r, FiniteSet(x**2, sin(x)))
 
+def test_issue_11827():
+    assert S.Naturals0**4
 
 def test_issue_10113():
     f = x**2/(x**2 - 4)

```


## Code snippets

### 1 - sympy/sets/fancysets.py:

Start line: 627, End line: 653

```python
class Range(Set):

    def _intersect(self, other):
        from sympy.functions.elementary.integers import ceiling, floor
        from sympy.functions.elementary.complexes import sign

        if other is S.Naturals:
            return self._intersect(Interval(1, S.Infinity))

        if other is S.Integers:
            return self

        if other.is_Interval:
            if not all(i.is_number for i in other.args[:2]):
                return

            # In case of null Range, return an EmptySet.
            if self.size == 0:
                return S.EmptySet

            # trim down to self's size, and represent
            # as a Range with step 1.
            start = ceiling(max(other.inf, self.inf))
            if start not in other:
                start += 1
            end = floor(min(other.sup, self.sup))
            if end not in other:
                end -= 1
            return self.intersect(Range(start, end + 1))
        # ... other code
```
### 2 - sympy/solvers/solveset.py:

Start line: 1457, End line: 1482

```python
def substitution(system, symbols, result=[{}], known_symbols=[],
                 exclude=[], all_symbols=None):
    # ... other code

    def add_intersection_complement(result, sym_set, **flags):
        # If solveset have returned some intersection/complement
        # for any symbol. It will be added in final solution.
        final_result = []
        for res in result:
            res_copy = res
            for key_res, value_res in res.items():
                # Intersection/complement is in Interval or Set.
                intersection_true = flags.get('Intersection', True)
                complements_true = flags.get('Complement', True)
                for key_sym, value_sym in sym_set.items():
                    if key_sym == key_res:
                        if intersection_true:
                            # testcase is not added for this line(intersection)
                            new_value = \
                                Intersection(FiniteSet(value_res), value_sym)
                            if new_value is not S.EmptySet:
                                res_copy[key_res] = new_value
                        if complements_true:
                            new_value = \
                                Complement(FiniteSet(value_res), value_sym)
                            if new_value is not S.EmptySet:
                                res_copy[key_res] = new_value
            final_result.append(res_copy)
        return final_result
    # end of def add_intersection_complement()
    # ... other code
```
### 3 - sympy/sets/fancysets.py:

Start line: 18, End line: 72

```python
class Naturals(with_metaclass(Singleton, Set)):
    """
    Represents the natural numbers (or counting numbers) which are all
    positive integers starting from 1. This set is also available as
    the Singleton, S.Naturals.

    Examples
    ========

    >>> from sympy import S, Interval, pprint
    >>> 5 in S.Naturals
    True
    >>> iterable = iter(S.Naturals)
    >>> next(iterable)
    1
    >>> next(iterable)
    2
    >>> next(iterable)
    3
    >>> pprint(S.Naturals.intersect(Interval(0, 10)))
    {1, 2, ..., 10}

    See Also
    ========
    Naturals0 : non-negative integers (i.e. includes 0, too)
    Integers : also includes negative integers
    """

    is_iterable = True
    _inf = S.One
    _sup = S.Infinity

    def _intersect(self, other):
        if other.is_Interval:
            return Intersection(
                S.Integers, other, Interval(self._inf, S.Infinity))
        return None

    def _contains(self, other):
        if not isinstance(other, Expr):
            return S.false
        elif other.is_positive and other.is_integer:
            return S.true
        elif other.is_integer is False or other.is_positive is False:
            return S.false

    def __iter__(self):
        i = self._inf
        while True:
            yield i
            i = i + 1

    @property
    def _boundary(self):
        return self
```
### 4 - sympy/sets/fancysets.py:

Start line: 1, End line: 15

```python
from __future__ import print_function, division

from sympy.logic.boolalg import And
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.compatibility import as_int, with_metaclass, range, PY3
from sympy.core.expr import Expr
from sympy.core.function import Lambda, _coeff_isneg
from sympy.core.singleton import Singleton, S
from sympy.core.symbol import Dummy, symbols, Wild
from sympy.core.sympify import _sympify, sympify, converter
from sympy.sets.sets import (Set, Interval, Intersection, EmptySet, Union,
                             FiniteSet, imageset)
from sympy.sets.conditionset import ConditionSet
from sympy.utilities.misc import filldedent, func_name
```
### 5 - sympy/sets/fancysets.py:

Start line: 655, End line: 758

```python
class Range(Set):

    def _intersect(self, other):
        # ... other code

        if isinstance(other, Range):
            from sympy.solvers.diophantine import diop_linear
            from sympy.core.numbers import ilcm

            # non-overlap quick exits
            if not other:
                return S.EmptySet
            if not self:
                return S.EmptySet
            if other.sup < self.inf:
                return S.EmptySet
            if other.inf > self.sup:
                return S.EmptySet

            # work with finite end at the start
            r1 = self
            if r1.start.is_infinite:
                r1 = r1.reversed
            r2 = other
            if r2.start.is_infinite:
                r2 = r2.reversed

            # this equation represents the values of the Range;
            # it's a linear equation
            eq = lambda r, i: r.start + i*r.step

            # we want to know when the two equations might
            # have integer solutions so we use the diophantine
            # solver
            a, b = diop_linear(eq(r1, Dummy()) - eq(r2, Dummy()))

            # check for no solution
            no_solution = a is None and b is None
            if no_solution:
                return S.EmptySet

            # there is a solution
            # -------------------

            # find the coincident point, c
            a0 = a.as_coeff_Add()[0]
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
            r1 = _updated_range(self, s1)
            r2 = _updated_range(other, s2)

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
        else:
            return
```
### 6 - sympy/sets/fancysets.py:

Start line: 418, End line: 491

```python
class ImageSet(Set):

    def _intersect(self, other):
        # ... other code

        if other == S.Reals:
            from sympy.solvers.solveset import solveset_real
            from sympy.core.function import expand_complex
            if len(self.lamda.variables) > 1:
                return None

            f = self.lamda.expr
            n = self.lamda.variables[0]

            n_ = Dummy(n.name, real=True)
            f_ = f.subs(n, n_)

            re, im = f_.as_real_imag()
            im = expand_complex(im)

            return imageset(Lambda(n_, re),
                            self.base_set.intersect(
                                solveset_real(im, n_)))

        elif isinstance(other, Interval):
            from sympy.solvers.solveset import (invert_real, invert_complex,
                                                solveset)

            f = self.lamda.expr
            n = self.lamda.variables[0]
            base_set = self.base_set
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
                    new_interval = Interval(new_inf, new_sup, new_lopen, new_ropen)
                    range_set = base_set._intersect(new_interval)
                else:
                    if other.is_subset(S.Reals):
                        solutions = solveset(f, n, S.Reals)
                        if not isinstance(range_set, (ImageSet, ConditionSet)):
                            range_set = solutions._intersect(other)
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
### 7 - sympy/solvers/solveset.py:

Start line: 835, End line: 901

```python
def solveset(f, symbol=None, domain=S.Complexes):
    f = sympify(f)

    if f is S.true:
        return domain

    if f is S.false:
        return S.EmptySet

    if not isinstance(f, (Expr, Number)):
        raise ValueError("%s is not a valid SymPy expression" % (f))

    free_symbols = f.free_symbols

    if not free_symbols:
        b = Eq(f, 0)
        if b is S.true:
            return domain
        elif b is S.false:
            return S.EmptySet
        else:
            raise NotImplementedError(filldedent('''
                relationship between value and 0 is unknown: %s''' % b))

    if symbol is None:
        if len(free_symbols) == 1:
            symbol = free_symbols.pop()
        else:
            raise ValueError(filldedent('''
                The independent variable must be specified for a
                multivariate equation.'''))
    elif not getattr(symbol, 'is_Symbol', False):
        raise ValueError('A Symbol must be given, not type %s: %s' %
            (type(symbol), symbol))

    if isinstance(f, Eq):
        from sympy.core import Add
        f = Add(f.lhs, - f.rhs, evaluate=False)
    elif f.is_Relational:
        if not domain.is_subset(S.Reals):
            raise NotImplementedError(filldedent('''
                Inequalities in the complex domain are
                not supported. Try the real domain by
                setting domain=S.Reals'''))
        try:
            result = solve_univariate_inequality(
            f, symbol, relational=False) - _invalid_solutions(
            f, symbol, domain)
        except NotImplementedError:
            result = ConditionSet(symbol, f, domain)
        return result

    return _solveset(f, symbol, domain, _check=True)


def _invalid_solutions(f, symbol, domain):
    bad = S.EmptySet
    for d in denoms(f):
        bad += _solveset(d, symbol, domain, _check=False)
    return bad


def solveset_real(f, symbol):
    return solveset(f, symbol, S.Reals)


def solveset_complex(f, symbol):
    return solveset(f, symbol, S.Complexes)
```
### 8 - sympy/sets/fancysets.py:

Start line: 760, End line: 805

```python
class Range(Set):

    def _contains(self, other):
        if not self:
            return S.false
        if other.is_infinite:
            return S.false
        if not other.is_integer:
            return other.is_integer
        ref = self.start if self.start.is_finite else self.stop
        if (ref - other) % self.step:  # off sequence
            return S.false
        return _sympify(other >= self.inf and other <= self.sup)

    def __iter__(self):
        if self.start in [S.NegativeInfinity, S.Infinity]:
            raise ValueError("Cannot iterate over Range with infinite start")
        elif self:
            i = self.start
            step = self.step

            while True:
                if (step > 0 and not (self.start <= i < self.stop)) or \
                   (step < 0 and not (self.stop < i <= self.start)):
                    break
                yield i
                i += step

    def __len__(self):
        if not self:
            return 0
        dif = self.stop - self.start
        if dif.is_infinite:
            raise ValueError(
                "Use .size to get the length of an infinite Range")
        return abs(dif//self.step)

    @property
    def size(self):
        try:
            return _sympify(len(self))
        except ValueError:
            return S.Infinity

    def __nonzero__(self):
        return self.start != self.stop

    __bool__ = __nonzero__
```
### 9 - sympy/sets/fancysets.py:

Start line: 75, End line: 92

```python
class Naturals0(Naturals):
    """Represents the whole numbers which are all the non-negative integers,
    inclusive of zero.

    See Also
    ========
    Naturals : positive integers; does not include 0
    Integers : also includes the negative integers
    """
    _inf = S.Zero

    def _contains(self, other):
        if not isinstance(other, Expr):
            return S.false
        elif other.is_integer and other.is_nonnegative:
            return S.true
        elif other.is_integer is False or other.is_nonnegative is False:
            return S.false
```
### 10 - sympy/sets/fancysets.py:

Start line: 171, End line: 210

```python
class Integers(with_metaclass(Singleton, Set)):

    def _eval_imageset(self, f):
        expr = f.expr
        if not isinstance(expr, Expr):
            return

        if len(f.variables) > 1:
            return

        n = f.variables[0]

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
            expr = match[a]*n + match[b] % match[a]

        if expr != f.expr:
            return ImageSet(Lambda(n, expr), S.Integers)


class Reals(with_metaclass(Singleton, Interval)):

    def __new__(cls):
        return Interval.__new__(cls, -S.Infinity, S.Infinity)

    def __eq__(self, other):
        return other == Interval(-S.Infinity, S.Infinity)

    def __hash__(self):
        return hash(Interval(-S.Infinity, S.Infinity))
```
