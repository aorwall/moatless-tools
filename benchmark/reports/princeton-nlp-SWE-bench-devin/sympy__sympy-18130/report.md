# sympy__sympy-18130

| **sympy/sympy** | `24fda38589c91044a4dca327bde11e69547ff6a6` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | - |
| **Avg pos** | - |
| **Min pos** | - |
| **Max pos** | - |
| **Top file pos** | 12 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sympy/solvers/diophantine.py b/sympy/solvers/diophantine.py
--- a/sympy/solvers/diophantine.py
+++ b/sympy/solvers/diophantine.py
@@ -1004,7 +1004,7 @@ def _diop_quadratic(var, coeff, t):
                 for z0 in range(0, abs(_c)):
                     # Check if the coefficients of y and x obtained are integers or not
                     if (divisible(sqa*g*z0**2 + D*z0 + sqa*F, _c) and
-                            divisible(e*sqc**g*z0**2 + E*z0 + e*sqc*F, _c)):
+                            divisible(e*sqc*g*z0**2 + E*z0 + e*sqc*F, _c)):
                         sol.add((solve_x(z0), solve_y(z0)))
 
     # (3) Method used when B**2 - 4*A*C is a square, is described in p. 6 of the below paper

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sympy/solvers/diophantine.py | 1007 | 1007 | - | 12 | -


## Problem Statement

```
ImageSet of n**2-1 returns EmptySet as intersection with Integers (diophantine bug)
\`\`\`
In [1]: ImageSet(Lambda(n, n**2 - 1), S.Integers).intersect(S.Integers)
Out[1]: ∅
\`\`\`

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 sympy/sets/handlers/intersection.py | 223 | 348| 1074 | 1074 | 3707 | 
| 2 | 2 sympy/sets/fancysets.py | 273 | 329| 459 | 1533 | 14359 | 
| 3 | 2 sympy/sets/fancysets.py | 330 | 357| 242 | 1775 | 14359 | 
| 4 | 2 sympy/sets/fancysets.py | 163 | 231| 397 | 2172 | 14359 | 
| 5 | 3 sympy/sets/sets.py | 1374 | 1410| 233 | 2405 | 30674 | 
| 6 | 3 sympy/sets/fancysets.py | 359 | 393| 320 | 2725 | 30674 | 
| 7 | 3 sympy/sets/sets.py | 2004 | 2095| 787 | 3512 | 30674 | 
| 8 | 3 sympy/sets/sets.py | 2097 | 2128| 255 | 3767 | 30674 | 
| 9 | 3 sympy/sets/fancysets.py | 431 | 481| 444 | 4211 | 30674 | 
| 10 | 4 sympy/sets/__init__.py | 1 | 34| 261 | 4472 | 30935 | 
| 11 | 4 sympy/sets/fancysets.py | 483 | 496| 146 | 4618 | 30935 | 
| 12 | 4 sympy/sets/fancysets.py | 395 | 429| 279 | 4897 | 30935 | 
| 13 | 4 sympy/sets/sets.py | 1349 | 1372| 134 | 5031 | 30935 | 
| 14 | 4 sympy/sets/handlers/intersection.py | 77 | 103| 220 | 5251 | 30935 | 
| 15 | 4 sympy/sets/sets.py | 1790 | 1827| 337 | 5588 | 30935 | 
| 16 | 4 sympy/sets/handlers/intersection.py | 1 | 28| 248 | 5836 | 30935 | 
| 17 | 4 sympy/sets/sets.py | 1296 | 1330| 176 | 6012 | 30935 | 
| 18 | 4 sympy/sets/sets.py | 178 | 214| 363 | 6375 | 30935 | 
| 19 | 4 sympy/sets/handlers/intersection.py | 405 | 457| 397 | 6772 | 30935 | 
| 20 | 4 sympy/sets/fancysets.py | 72 | 133| 406 | 7178 | 30935 | 
| 21 | 4 sympy/sets/sets.py | 1606 | 1668| 299 | 7477 | 30935 | 
| 22 | 4 sympy/sets/fancysets.py | 136 | 160| 167 | 7644 | 30935 | 
| 23 | 4 sympy/sets/handlers/intersection.py | 30 | 75| 407 | 8051 | 30935 | 
| 24 | 5 sympy/simplify/sqrtdenest.py | 76 | 102| 297 | 8348 | 37513 | 
| 25 | 5 sympy/sets/sets.py | 155 | 176| 152 | 8500 | 37513 | 
| 26 | 5 sympy/sets/sets.py | 889 | 928| 372 | 8872 | 37513 | 
| 27 | 5 sympy/sets/sets.py | 1412 | 1514| 968 | 9840 | 37513 | 
| 28 | 5 sympy/sets/sets.py | 1722 | 1766| 339 | 10179 | 37513 | 
| 29 | 5 sympy/sets/sets.py | 1049 | 1062| 122 | 10301 | 37513 | 
| 30 | 5 sympy/sets/handlers/intersection.py | 351 | 403| 397 | 10698 | 37513 | 
| 31 | 5 sympy/sets/sets.py | 111 | 127| 139 | 10837 | 37513 | 
| 32 | 5 sympy/sets/sets.py | 1332 | 1347| 128 | 10965 | 37513 | 
| 33 | 5 sympy/sets/sets.py | 1207 | 1249| 414 | 11379 | 37513 | 
| 34 | 5 sympy/sets/sets.py | 129 | 153| 153 | 11532 | 37513 | 
| 35 | 5 sympy/sets/handlers/intersection.py | 105 | 220| 962 | 12494 | 37513 | 
| 36 | 5 sympy/sets/sets.py | 1768 | 1788| 181 | 12675 | 37513 | 
| 37 | 5 sympy/sets/sets.py | 240 | 283| 239 | 12914 | 37513 | 
| 38 | 5 sympy/sets/fancysets.py | 234 | 270| 220 | 13134 | 37513 | 
| 39 | 5 sympy/sets/sets.py | 1176 | 1205| 225 | 13359 | 37513 | 
| 40 | 5 sympy/sets/sets.py | 38 | 82| 322 | 13681 | 37513 | 
| 41 | 5 sympy/sets/sets.py | 1251 | 1266| 142 | 13823 | 37513 | 
| 42 | 5 sympy/sets/fancysets.py | 23 | 69| 331 | 14154 | 37513 | 
| 43 | 5 sympy/sets/fancysets.py | 692 | 730| 283 | 14437 | 37513 | 
| 44 | 5 sympy/sets/fancysets.py | 741 | 863| 1041 | 15478 | 37513 | 
| 45 | 5 sympy/sets/fancysets.py | 584 | 646| 533 | 16011 | 37513 | 
| 46 | 6 sympy/functions/combinatorial/numbers.py | 1737 | 1771| 292 | 16303 | 55489 | 
| 47 | 6 sympy/sets/sets.py | 603 | 618| 112 | 16415 | 55489 | 
| 48 | 6 sympy/functions/combinatorial/numbers.py | 2008 | 2047| 346 | 16761 | 55489 | 
| 49 | 7 sympy/sets/setexpr.py | 1 | 97| 832 | 17593 | 56321 | 
| 50 | 8 sympy/solvers/solveset.py | 2023 | 2078| 493 | 18086 | 86995 | 
| 51 | 8 sympy/functions/combinatorial/numbers.py | 1718 | 1734| 136 | 18222 | 86995 | 
| 52 | 9 sympy/combinatorics/partitions.py | 301 | 327| 247 | 18469 | 92718 | 
| 53 | 9 sympy/sets/sets.py | 620 | 657| 248 | 18717 | 92718 | 
| 54 | 9 sympy/sets/fancysets.py | 1383 | 1418| 169 | 18886 | 92718 | 
| 55 | 9 sympy/sets/fancysets.py | 670 | 690| 146 | 19032 | 92718 | 
| 56 | 9 sympy/solvers/solveset.py | 1145 | 1209| 659 | 19691 | 92718 | 
| 57 | 9 sympy/sets/sets.py | 471 | 522| 306 | 19997 | 92718 | 
| 58 | 9 sympy/sets/fancysets.py | 499 | 582| 674 | 20671 | 92718 | 
| 59 | 9 sympy/sets/sets.py | 1122 | 1157| 201 | 20872 | 92718 | 
| 60 | 10 sympy/sets/handlers/functions.py | 25 | 112| 759 | 21631 | 94831 | 
| 61 | 11 sympy/combinatorics/subsets.py | 558 | 587| 223 | 21854 | 98899 | 
| 62 | 11 sympy/sets/handlers/functions.py | 168 | 215| 428 | 22282 | 98899 | 
| 63 | 11 sympy/sets/sets.py | 1911 | 1933| 206 | 22488 | 98899 | 
| 64 | **12 sympy/solvers/diophantine.py** | 3259 | 3304| 494 | 22982 | 130436 | 
| 65 | 13 sympy/sets/handlers/union.py | 1 | 14| 111 | 23093 | 131457 | 
| 66 | 13 sympy/sets/sets.py | 2290 | 2309| 204 | 23297 | 131457 | 
| 67 | 13 sympy/solvers/solveset.py | 1856 | 1953| 767 | 24064 | 131457 | 
| 68 | 14 sympy/sets/handlers/issubset.py | 1 | 32| 316 | 24380 | 132348 | 
| 69 | 14 sympy/sets/sets.py | 1936 | 2001| 388 | 24768 | 132348 | 
| 70 | 14 sympy/sets/sets.py | 1279 | 1293| 126 | 24894 | 132348 | 
| 71 | 14 sympy/sets/fancysets.py | 732 | 740| 137 | 25031 | 132348 | 
| 72 | 15 sympy/utilities/iterables.py | 1726 | 1804| 531 | 25562 | 154222 | 
| 73 | 15 sympy/sets/sets.py | 706 | 728| 206 | 25768 | 154222 | 
| 74 | 15 sympy/functions/combinatorial/numbers.py | 1700 | 1715| 168 | 25936 | 154222 | 
| 75 | 15 sympy/functions/combinatorial/numbers.py | 1373 | 1417| 218 | 26154 | 154222 | 
| 76 | 16 sympy/tensor/indexed.py | 577 | 639| 509 | 26663 | 160468 | 
| 77 | 17 sympy/core/numbers.py | 3789 | 3832| 242 | 26905 | 190199 | 
| 78 | 18 sympy/sets/conditionset.py | 1 | 18| 154 | 27059 | 192450 | 


### Hint

```
This one's a bug in `diophantine`:
\`\`\`
In [1]: diophantine(x**2 - 1 - y)
Out[1]: set()
\`\`\`
The equation has rather trivial integer solutions.
\`\`\`
In [14]: from sympy.solvers.diophantine import diop_quadratic                                                                  
In [15]: diop_quadratic(m**2 - n - 1, k)                                                                                       
Out[15]:
⎧⎛    2    ⎞⎫
⎨⎝k, k  - 1⎠⎬
⎩           ⎭

In [16]: diophantine(m**2 - n - 1, k)
Out[16]: set()
\`\`\`
The solutions are discarded somewhere inside `diophantine`.

Actually, when `diop_quadratic` is invoked via `diophantine` the signs of the coefficients are negated. So the issue can be reproduced as follows:
\`\`\`
In [1]: from sympy.solvers.diophantine import diop_quadratic

In [2]: diop_quadratic(m**2 - n - 1, k)
Out[2]:
⎧⎛    2    ⎞⎫
⎨⎝k, k  - 1⎠⎬
⎩           ⎭

In [3]: diop_quadratic(-m**2 + n + 1, k)
Out[3]: set()
\`\`\`
```

## Patch

```diff
diff --git a/sympy/solvers/diophantine.py b/sympy/solvers/diophantine.py
--- a/sympy/solvers/diophantine.py
+++ b/sympy/solvers/diophantine.py
@@ -1004,7 +1004,7 @@ def _diop_quadratic(var, coeff, t):
                 for z0 in range(0, abs(_c)):
                     # Check if the coefficients of y and x obtained are integers or not
                     if (divisible(sqa*g*z0**2 + D*z0 + sqa*F, _c) and
-                            divisible(e*sqc**g*z0**2 + E*z0 + e*sqc*F, _c)):
+                            divisible(e*sqc*g*z0**2 + E*z0 + e*sqc*F, _c)):
                         sol.add((solve_x(z0), solve_y(z0)))
 
     # (3) Method used when B**2 - 4*A*C is a square, is described in p. 6 of the below paper

```

## Test Patch

```diff
diff --git a/sympy/solvers/tests/test_diophantine.py b/sympy/solvers/tests/test_diophantine.py
--- a/sympy/solvers/tests/test_diophantine.py
+++ b/sympy/solvers/tests/test_diophantine.py
@@ -540,6 +540,12 @@ def test_diophantine():
     assert diophantine(x**2 + y**2 +3*x- 5, permute=True) == \
         set([(-1, 1), (-4, -1), (1, -1), (1, 1), (-4, 1), (-1, -1), (4, 1), (4, -1)])
 
+    # issue 18122
+    assert check_solutions(x**2-y)
+    assert check_solutions(y**2-x)
+    assert diophantine((x**2-y), t) == set([(t, t**2)])
+    assert diophantine((y**2-x), t) == set([(t**2, -t)])
+
 
 def test_general_pythagorean():
     from sympy.abc import a, b, c, d, e

```


## Code snippets

### 1 - sympy/sets/handlers/intersection.py:

Start line: 223, End line: 348

```python
@dispatch(ImageSet, Set)
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
    if base_set is S.Integers:
        gm = None
        if isinstance(other, ImageSet) and other.base_sets == (S.Integers,):
            gm = other.lamda.expr
            m = other.lamda.variables[0]
        elif other is S.Integers:
            m = gm = Dummy('x')
        if gm is not None:
            fn = self.lamda.expr
            n = self.lamda.variables[0]
            solns = list(diophantine(fn - gm, syms=(n, m)))
            if len(solns) == 0:
                return EmptySet
            elif len(solns) != 1:
                return
            else:
                soln, solm = solns[0]
                (t,) = soln.free_symbols
                expr = fn.subs(n, soln.subs(t, n))
                return imageset(Lambda(n, expr), S.Integers)

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
### 2 - sympy/sets/fancysets.py:

Start line: 273, End line: 329

```python
class ImageSet(Set):
    """
    Image of a set under a mathematical function. The transformation
    must be given as a Lambda function which has as many arguments
    as the elements of the set upon which it operates, e.g. 1 argument
    when acting on the set of integers or 2 arguments when acting on
    a complex region.

    This function is not normally called directly, but is called
    from `imageset`.


    Examples
    ========

    >>> from sympy import Symbol, S, pi, Dummy, Lambda
    >>> from sympy.sets.sets import FiniteSet, Interval
    >>> from sympy.sets.fancysets import ImageSet

    >>> x = Symbol('x')
    >>> N = S.Naturals
    >>> squares = ImageSet(Lambda(x, x**2), N) # {x**2 for x in N}
    >>> 4 in squares
    True
    >>> 5 in squares
    False

    >>> FiniteSet(0, 1, 2, 3, 4, 5, 6, 7, 9, 10).intersect(squares)
    FiniteSet(1, 4, 9)

    >>> square_iterable = iter(squares)
    >>> for i in range(4):
    ...     next(square_iterable)
    1
    4
    9
    16

    If you want to get value for `x` = 2, 1/2 etc. (Please check whether the
    `x` value is in `base_set` or not before passing it as args)

    >>> squares.lamda(2)
    4
    >>> squares.lamda(S(1)/2)
    1/4

    >>> n = Dummy('n')
    >>> solutions = ImageSet(Lambda(n, n*pi), S.Integers) # solutions of sin(x) = 0
    >>> dom = Interval(-1, 1)
    >>> dom.intersect(solutions)
    FiniteSet(0)

    See Also
    ========

    sympy.sets.sets.imageset
    """
```
### 3 - sympy/sets/fancysets.py:

Start line: 330, End line: 357

```python
class ImageSet(Set):
    def __new__(cls, flambda, *sets):
        if not isinstance(flambda, Lambda):
            raise ValueError('First argument must be a Lambda')

        signature = flambda.signature

        if len(signature) != len(sets):
            raise ValueError('Incompatible signature')

        sets = [_sympify(s) for s in sets]

        if not all(isinstance(s, Set) for s in sets):
            raise TypeError("Set arguments to ImageSet should of type Set")

        if not all(cls._check_sig(sg, st) for sg, st in zip(signature, sets)):
            raise ValueError("Signature %s does not match sets %s" % (signature, sets))

        if flambda is S.IdentityFunction and len(sets) == 1:
            return sets[0]

        if not set(flambda.variables) & flambda.expr.free_symbols:
            is_empty = fuzzy_or(s.is_empty for s in sets)
            if is_empty == True:
                return S.EmptySet
            elif is_empty == False:
                return FiniteSet(flambda.expr)

        return Basic.__new__(cls, flambda, *sets)
```
### 4 - sympy/sets/fancysets.py:

Start line: 163, End line: 231

```python
class Integers(with_metaclass(Singleton, Set)):
    """
    Represents all integers: positive, negative and zero. This set is also
    available as the Singleton, S.Integers.

    Examples
    ========

    >>> from sympy import S, Interval, pprint
    >>> 5 in S.Naturals
    True
    >>> iterable = iter(S.Integers)
    >>> next(iterable)
    0
    >>> next(iterable)
    1
    >>> next(iterable)
    -1
    >>> next(iterable)
    2

    >>> pprint(S.Integers.intersect(Interval(-4, 4)))
    {-4, -3, ..., 4}

    See Also
    ========

    Naturals0 : non-negative integers
    Integers : positive and negative integers and zero
    """

    is_iterable = True
    is_empty = False
    is_finite_set = False

    def _contains(self, other):
        if not isinstance(other, Expr):
            return S.false
        return other.is_integer

    def __iter__(self):
        yield S.Zero
        i = S.One
        while True:
            yield i
            yield -i
            i = i + 1

    @property
    def _inf(self):
        return S.NegativeInfinity

    @property
    def _sup(self):
        return S.Infinity

    @property
    def _boundary(self):
        return self

    def as_relational(self, x):
        from sympy.functions.elementary.integers import floor
        return And(Eq(floor(x), x), -oo < x, x < oo)

    def _eval_is_subset(self, other):
        return Range(-oo, oo).is_subset(other)

    def _eval_is_superset(self, other):
        return Range(-oo, oo).is_superset(other)
```
### 5 - sympy/sets/sets.py:

Start line: 1374, End line: 1410

```python
class Intersection(Set, LatticeOp):

    def __iter__(self):
        sets_sift = sift(self.args, lambda x: x.is_iterable)

        completed = False
        candidates = sets_sift[True] + sets_sift[None]

        finite_candidates, others = [], []
        for candidate in candidates:
            length = None
            try:
                length = len(candidate)
            except TypeError:
                others.append(candidate)

            if length is not None:
                finite_candidates.append(candidate)
        finite_candidates.sort(key=len)

        for s in finite_candidates + others:
            other_sets = set(self.args) - set((s,))
            other = Intersection(*other_sets, evaluate=False)
            completed = True
            for x in s:
                try:
                    if x in other:
                        yield x
                except TypeError:
                    completed = False
            if completed:
                return

        if not completed:
            if not candidates:
                raise TypeError("None of the constituent sets are iterable")
            raise TypeError(
                "The computation had not completed because of the "
                "undecidable set membership is found in every candidates.")
```
### 6 - sympy/sets/fancysets.py:

Start line: 359, End line: 393

```python
class ImageSet(Set):

    lamda = property(lambda self: self.args[0])
    base_sets = property(lambda self: self.args[1:])

    @property
    def base_set(self):
        # XXX: Maybe deprecate this? It is poorly defined in handling
        # the multivariate case...
        sets = self.base_sets
        if len(sets) == 1:
            return sets[0]
        else:
            return ProductSet(*sets).flatten()

    @property
    def base_pset(self):
        return ProductSet(*self.base_sets)

    @classmethod
    def _check_sig(cls, sig_i, set_i):
        if sig_i.is_symbol:
            return True
        elif isinstance(set_i, ProductSet):
            sets = set_i.sets
            if len(sig_i) != len(sets):
                return False
            # Recurse through the signature for nested tuples:
            return all(cls._check_sig(ts, ps) for ts, ps in zip(sig_i, sets))
        else:
            # XXX: Need a better way of checking whether a set is a set of
            # Tuples or not. For example a FiniteSet can contain Tuples
            # but so can an ImageSet or a ConditionSet. Others like
            # Integers, Reals etc can not contain Tuples. We could just
            # list the possibilities here... Current code for e.g.
            # _contains probably only works for ProductSet.
            return True # Give the benefit of the doubt
```
### 7 - sympy/sets/sets.py:

Start line: 2004, End line: 2095

```python
def imageset(*args):
    r"""
    Return an image of the set under transformation ``f``.

    If this function can't compute the image, it returns an
    unevaluated ImageSet object.

    .. math::
        \{ f(x) \mid x \in \mathrm{self} \}

    Examples
    ========

    >>> from sympy import S, Interval, Symbol, imageset, sin, Lambda
    >>> from sympy.abc import x, y

    >>> imageset(x, 2*x, Interval(0, 2))
    Interval(0, 4)

    >>> imageset(lambda x: 2*x, Interval(0, 2))
    Interval(0, 4)

    >>> imageset(Lambda(x, sin(x)), Interval(-2, 1))
    ImageSet(Lambda(x, sin(x)), Interval(-2, 1))

    >>> imageset(sin, Interval(-2, 1))
    ImageSet(Lambda(x, sin(x)), Interval(-2, 1))
    >>> imageset(lambda y: x + y, Interval(-2, 1))
    ImageSet(Lambda(y, x + y), Interval(-2, 1))

    Expressions applied to the set of Integers are simplified
    to show as few negatives as possible and linear expressions
    are converted to a canonical form. If this is not desirable
    then the unevaluated ImageSet should be used.

    >>> imageset(x, -2*x + 5, S.Integers)
    ImageSet(Lambda(x, 2*x + 1), Integers)

    See Also
    ========

    sympy.sets.fancysets.ImageSet

    """
    from sympy.core import Lambda
    from sympy.sets.fancysets import ImageSet
    from sympy.sets.setexpr import set_function

    if len(args) < 2:
        raise ValueError('imageset expects at least 2 args, got: %s' % len(args))

    if isinstance(args[0], (Symbol, tuple)) and len(args) > 2:
        f = Lambda(args[0], args[1])
        set_list = args[2:]
    else:
        f = args[0]
        set_list = args[1:]

    if isinstance(f, Lambda):
        pass
    elif callable(f):
        nargs = getattr(f, 'nargs', {})
        if nargs:
            if len(nargs) != 1:
                raise NotImplementedError(filldedent('''
                    This function can take more than 1 arg
                    but the potentially complicated set input
                    has not been analyzed at this point to
                    know its dimensions. TODO
                    '''))
            N = nargs.args[0]
            if N == 1:
                s = 'x'
            else:
                s = [Symbol('x%i' % i) for i in range(1, N + 1)]
        else:
            if PY3:
                s = inspect.signature(f).parameters
            else:
                s = inspect.getargspec(f).args
        dexpr = _sympify(f(*[Dummy() for i in s]))
        var = tuple(_uniquely_named_symbol(Symbol(i), dexpr) for i in s)
        f = Lambda(var, f(*var))
    else:
        raise TypeError(filldedent('''
            expecting lambda, Lambda, or FunctionClass,
            not \'%s\'.''' % func_name(f)))

    if any(not isinstance(s, Set) for s in set_list):
        name = [func_name(s) for s in set_list]
        raise ValueError(
            'arguments after mapping should be sets, not %s' % name)
    # ... other code
```
### 8 - sympy/sets/sets.py:

Start line: 2097, End line: 2128

```python
def imageset(*args):
    # ... other code

    if len(set_list) == 1:
        set = set_list[0]
        try:
            # TypeError if arg count != set dimensions
            r = set_function(f, set)
            if r is None:
                raise TypeError
            if not r:
                return r
        except TypeError:
            r = ImageSet(f, set)
        if isinstance(r, ImageSet):
            f, set = r.args

        if f.variables[0] == f.expr:
            return set

        if isinstance(set, ImageSet):
            # XXX: Maybe this should just be:
            # f2 = set.lambda
            # fun = Lambda(f2.signature, f(*f2.expr))
            # return imageset(fun, *set.base_sets)
            if len(set.lamda.variables) == 1 and len(f.variables) == 1:
                x = set.lamda.variables[0]
                y = f.variables[0]
                return imageset(
                    Lambda(x, f.expr.subs(y, set.lamda.expr)), *set.base_sets)

        if r is not None:
            return r

    return ImageSet(f, *set_list)
```
### 9 - sympy/sets/fancysets.py:

Start line: 431, End line: 481

```python
class ImageSet(Set):

    def _contains(self, other):
        # ... other code

        def get_equations(expr, candidate):
            '''Find the equations relating symbols in expr and candidate.'''
            queue = [(expr, candidate)]
            for e, c in queue:
                if not isinstance(e, Tuple):
                    yield Eq(e, c)
                elif not isinstance(c, Tuple) or len(e) != len(c):
                    yield False
                    return
                else:
                    queue.extend(zip(e, c))

        # Get the basic objects together:
        other = _sympify(other)
        expr = self.lamda.expr
        sig = self.lamda.signature
        variables = self.lamda.variables
        base_sets = self.base_sets

        # Use dummy symbols for ImageSet parameters so they don't match
        # anything in other
        rep = {v: Dummy(v.name) for v in variables}
        variables = [v.subs(rep) for v in variables]
        sig = sig.subs(rep)
        expr = expr.subs(rep)

        # Map the parts of other to those in the Lambda expr
        equations = []
        for eq in get_equations(expr, other):
            # Unsatisfiable equation?
            if eq is False:
                return False
            equations.append(eq)

        # Map the symbols in the signature to the corresponding domains
        symsetmap = get_symsetmap(sig, base_sets)
        if symsetmap is None:
            # Can't factor the base sets to a ProductSet
            return None

        # Which of the variables in the Lambda signature need to be solved for?
        symss = (eq.free_symbols for eq in equations)
        variables = set(variables) & reduce(set.union, symss, set())

        # Use internal multivariate solveset
        variables = tuple(variables)
        base_sets = [symsetmap[v] for v in variables]
        solnset = _solveset_multi(equations, variables, base_sets)
        if solnset is None:
            return None
        return fuzzy_not(solnset.is_empty)
```
### 10 - sympy/sets/__init__.py:

Start line: 1, End line: 34

```python
from .sets import (Set, Interval, Union, FiniteSet, ProductSet,
        Intersection, imageset, Complement, SymmetricDifference)
from .fancysets import ImageSet, Range, ComplexRegion
from .contains import Contains
from .conditionset import ConditionSet
from .ordinals import Ordinal, OmegaPower, ord0
from .powerset import PowerSet
from ..core.singleton import S

Reals = S.Reals
Naturals = S.Naturals
Naturals0 = S.Naturals0
UniversalSet = S.UniversalSet
EmptySet = S.EmptySet
Integers = S.Integers
Rationals = S.Rationals

__all__ = [
    'Set', 'Interval', 'Union', 'EmptySet', 'FiniteSet', 'ProductSet',
    'Intersection', 'imageset', 'Complement', 'SymmetricDifference',

    'ImageSet', 'Range', 'ComplexRegion', 'Reals',

    'Contains',

    'ConditionSet',

    'Ordinal', 'OmegaPower', 'ord0',

    'PowerSet',

    'Reals', 'Naturals', 'Naturals0', 'UniversalSet', 'Integers', 'Rationals',
]
```
### 64 - sympy/solvers/diophantine.py:

Start line: 3259, End line: 3304

```python
def sum_of_squares(n, k, zeros=False):
    """Return a generator that yields the k-tuples of nonnegative
    values, the squares of which sum to n. If zeros is False (default)
    then the solution will not contain zeros. The nonnegative
    elements of a tuple are sorted.

    * If k == 1 and n is square, (n,) is returned.

    * If k == 2 then n can only be written as a sum of squares if
      every prime in the factorization of n that has the form
      4*k + 3 has an even multiplicity. If n is prime then
      it can only be written as a sum of two squares if it is
      in the form 4*k + 1.

    * if k == 3 then n can be written as a sum of squares if it does
      not have the form 4**m*(8*k + 7).

    * all integers can be written as the sum of 4 squares.

    * if k > 4 then n can be partitioned and each partition can
      be written as a sum of 4 squares; if n is not evenly divisible
      by 4 then n can be written as a sum of squares only if the
      an additional partition can be written as sum of squares.
      For example, if k = 6 then n is partitioned into two parts,
      the first being written as a sum of 4 squares and the second
      being written as a sum of 2 squares -- which can only be
      done if the condition above for k = 2 can be met, so this will
      automatically reject certain partitions of n.

    Examples
    ========

    >>> from sympy.solvers.diophantine import sum_of_squares
    >>> list(sum_of_squares(25, 2))
    [(3, 4)]
    >>> list(sum_of_squares(25, 2, True))
    [(3, 4), (0, 5)]
    >>> list(sum_of_squares(25, 4))
    [(1, 2, 2, 4)]

    See Also
    ========
    sympy.utilities.iterables.signed_permutations
    """
    for t in power_representation(n, 2, k, zeros):
        yield t
```
