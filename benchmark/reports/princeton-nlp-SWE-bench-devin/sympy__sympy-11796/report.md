# sympy__sympy-11796

| **sympy/sympy** | `8e80c0be90728b915942d7953e4b2c5d56deb570` |
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
@@ -737,6 +737,8 @@ def __new__(cls, start, end, left_open=False, right_open=False):
         if end == start and (left_open or right_open):
             return S.EmptySet
         if end == start and not (left_open or right_open):
+            if start == S.Infinity or start == S.NegativeInfinity:
+                return S.EmptySet
             return FiniteSet(end)
 
         # Make sure infinite interval end points are open.

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sympy/sets/sets.py | 740 | 740 | - | - | -


## Problem Statement

```
Where oo belongs? (Concept)
Hi again, well, i'm little confuse of the conditions to take or not `oo` in some sets:

\`\`\` python
>>> Interval(-oo, oo)
(-oo, oo)
\`\`\`

First the means the interval is created excluding `oo` and `-oo`, and interval interpret it in that way, but now:

\`\`\` python
>>> Interval(oo, oo)
{oo}
\`\`\`

Here is a little conflict, in first place Interval show don't take `oo` but now it is there? in some way is fine to have a way to can represent the `oo` from Interval.

Now from this point we have some points:
How they will interpret the limit concept? basically two options, limit is:

\`\`\` python
[x, oo]
\`\`\`

or

\`\`\` python
[x, oo)
\`\`\`

?
This point is very important, because define the behavior for sets, and affects directly like this issue: https://github.com/sympy/sympy/issues/11174

so, for now only to match the math in all sets we can say the limit is calculated via

\`\`\` python
[x, oo)
\`\`\`

now, what is the effect of this in Sympy?, first this enable the limit concept in every unbounded set, for now i found this two issues:
https://github.com/sympy/sympy/issues/11688
https://github.com/sympy/sympy/issues/11640

for example, actually we have this:

\`\`\` python
>>> solveset(y/x, x)
EmptySet()
\`\`\`

this return should be something like... `nan`? because in the limit we don't know what is the proportion of `y` and `x`, so we can't calc it.

actually this concept is applied in some way like:

\`\`\` python
>>> solveset(y*x, x)
{0} 
\`\`\`

Now the next question, `oo` will represent the infinite, as a integer, real or what?
i know this question don't have sense, but let me try explain it:

\`\`\` python
>>> Interval(-oo, oo) in S.Reals
False
>>> Interval(-oo, oo) in S.Naturals
#can't be calculated for now
\`\`\`

if the oo represent the infinite without form, it can exist in S.Naturals, and S.Reals, but if you represent the infinite like the interval between it, `Interval(x, oo)` where is the limit of x to infinite while always `x < oo`, in other way `Interval(A, B)` where A go to `oo` and B do to `oo`, but it need always will respect this condition `A < B` so between `A` and `B` can exist any type of numbers, so `oo` can't exist in `S.Naturals` because `Interval(A, B)` can contains a real number for example, but the extension of that concept says `oo` can't exist in any set, because always will exist a bigger set, in sympy you have an approximation of it, is `UniversalSet`, but don't will be true completely, because, why is it the limit set?, `UniversalSet` can represent two things, the limit of the actually human knowledge (or applied to Sympy), or the 'master' set, thinking its like the perfection of the sets knowledge.
Obvs, to `oo` make some sense in the actual system the option is interpret `oo` without limit or form, and take the second interpretation of `UniversalSet` (if you take the first. `oo` can't exist in any place).
If you disagree you always can discuss and change the behavior.

Objetives of this issue:

Get a clear definitions in Sympy of:
- Infinite
- Limit
- UniversalSet

Then, clear the behavior of this concepts in Sympy, and to finish, set the behavior in Sympy.

Thx. Cya.


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 sympy/series/order.py | 1 | 122| 1162 | 1162 | 4079 | 
| 2 | 2 sympy/calculus/util.py | 457 | 608| 1496 | 2658 | 13289 | 
| 3 | 3 sympy/core/numbers.py | 2432 | 2531| 635 | 3293 | 40168 | 
| 4 | 3 sympy/calculus/util.py | 779 | 833| 516 | 3809 | 40168 | 
| 5 | 4 sympy/concrete/summations.py | 394 | 484| 829 | 4638 | 49933 | 
| 6 | 5 sympy/series/limits.py | 75 | 119| 327 | 4965 | 51399 | 
| 7 | 6 sympy/series/gruntz.py | 407 | 439| 329 | 5294 | 57605 | 
| 8 | 7 examples/beginner/limits_examples.py | 1 | 42| 306 | 5600 | 57911 | 
| 9 | 7 sympy/calculus/util.py | 864 | 933| 557 | 6157 | 57911 | 
| 10 | 7 sympy/calculus/util.py | 720 | 735| 174 | 6331 | 57911 | 
| 11 | 7 sympy/calculus/util.py | 702 | 718| 168 | 6499 | 57911 | 
| 12 | 7 sympy/calculus/util.py | 610 | 631| 181 | 6680 | 57911 | 
| 13 | 7 sympy/core/numbers.py | 2602 | 2667| 477 | 7157 | 57911 | 
| 14 | 7 sympy/calculus/util.py | 737 | 777| 353 | 7510 | 57911 | 
| 15 | 7 sympy/calculus/util.py | 935 | 979| 338 | 7848 | 57911 | 
| 16 | 7 sympy/calculus/util.py | 835 | 862| 237 | 8085 | 57911 | 
| 17 | 7 sympy/concrete/summations.py | 23 | 150| 1346 | 9431 | 57911 | 
| 18 | 7 sympy/core/numbers.py | 2993 | 3061| 352 | 9783 | 57911 | 
| 19 | 8 sympy/sets/fancysets.py | 816 | 923| 923 | 10706 | 69192 | 
| 20 | 8 sympy/calculus/util.py | 1096 | 1127| 219 | 10925 | 69192 | 
| 21 | 8 sympy/series/limits.py | 122 | 190| 569 | 11494 | 69192 | 
| 22 | 9 sympy/core/power.py | 102 | 182| 1032 | 12526 | 82383 | 
| 23 | 9 sympy/calculus/util.py | 1173 | 1189| 143 | 12669 | 82383 | 
| 24 | 9 sympy/series/gruntz.py | 1 | 119| 1345 | 14014 | 82383 | 
| 25 | 9 sympy/concrete/summations.py | 307 | 393| 759 | 14773 | 82383 | 
| 26 | 10 sympy/concrete/expr_with_limits.py | 74 | 118| 384 | 15157 | 86030 | 
| 27 | 10 sympy/concrete/expr_with_limits.py | 266 | 341| 729 | 15886 | 86030 | 
| 28 | 10 sympy/core/numbers.py | 2813 | 2878| 481 | 16367 | 86030 | 
| 29 | 11 sympy/plotting/intervalmath/interval_arithmetic.py | 315 | 354| 341 | 16708 | 89451 | 
| 30 | 11 sympy/sets/fancysets.py | 655 | 758| 881 | 17589 | 89451 | 
| 31 | 11 sympy/concrete/summations.py | 1 | 20| 198 | 17787 | 89451 | 
| 32 | 11 sympy/plotting/intervalmath/interval_arithmetic.py | 100 | 135| 249 | 18036 | 89451 | 
| 33 | 11 sympy/calculus/util.py | 1058 | 1094| 289 | 18325 | 89451 | 
| 34 | 11 sympy/concrete/summations.py | 485 | 518| 306 | 18631 | 89451 | 
| 35 | 11 sympy/plotting/intervalmath/interval_arithmetic.py | 278 | 313| 292 | 18923 | 89451 | 
| 36 | 12 sympy/polys/rootisolation.py | 728 | 782| 647 | 19570 | 108516 | 
| 37 | 12 sympy/sets/fancysets.py | 760 | 805| 318 | 19888 | 108516 | 
| 38 | 12 sympy/calculus/util.py | 981 | 1018| 299 | 20187 | 108516 | 
| 39 | 13 sympy/series/fourier.py | 39 | 90| 466 | 20653 | 111443 | 
| 40 | 13 sympy/plotting/intervalmath/interval_arithmetic.py | 186 | 219| 242 | 20895 | 111443 | 
| 41 | 14 sympy/solvers/solveset.py | 1920 | 2074| 1939 | 22834 | 130604 | 
| 42 | 14 sympy/sets/fancysets.py | 627 | 653| 215 | 23049 | 130604 | 
| 43 | 14 sympy/calculus/util.py | 1020 | 1056| 288 | 23337 | 130604 | 
| 44 | 14 sympy/plotting/intervalmath/interval_arithmetic.py | 242 | 276| 272 | 23609 | 130604 | 
| 45 | 15 sympy/calculus/singularities.py | 103 | 139| 360 | 23969 | 132609 | 
| 46 | 16 sympy/concrete/products.py | 15 | 187| 1768 | 25737 | 136984 | 
| 47 | 17 sympy/solvers/solvers.py | 358 | 800| 4409 | 30146 | 166567 | 
| 48 | 17 sympy/sets/fancysets.py | 171 | 210| 321 | 30467 | 166567 | 
| 49 | 17 sympy/sets/fancysets.py | 18 | 72| 346 | 30813 | 166567 | 
| 50 | 17 sympy/calculus/singularities.py | 59 | 100| 328 | 31141 | 166567 | 
| 51 | 18 sympy/integrals/integrals.py | 321 | 352| 296 | 31437 | 178376 | 
| 52 | 18 sympy/series/limits.py | 1 | 45| 389 | 31826 | 178376 | 
| 53 | 18 sympy/plotting/intervalmath/interval_arithmetic.py | 162 | 184| 179 | 32005 | 178376 | 
| 54 | 18 sympy/plotting/intervalmath/interval_arithmetic.py | 137 | 160| 182 | 32187 | 178376 | 
| 55 | 18 sympy/concrete/expr_with_limits.py | 212 | 264| 470 | 32657 | 178376 | 
| 56 | 18 sympy/concrete/expr_with_limits.py | 176 | 210| 281 | 32938 | 178376 | 


## Missing Patch Files

 * 1: sympy/sets/sets.py

### Hint

```
Interval should represent a real interval. I think we decided in another issue that Interval should always be open for infinite boundaries, because it should always be a subset of S.Reals. So 

\`\`\`
>>> Interval(oo, oo)
{oo}
\`\`\`

is wrong.  I'm going to modify the issue title to make this clearer. 

Regarding your other points, note that `in` means "is contained in", not "is subset of". So `<set of numbers> in <set of numbers>` will always give False. I'm really not following your other points, but note that both `S.Reals` and `S.Naturals` (the latter is a subset of the former) contain only _finite_ numbers, so `oo` is not contained in either). 

```

## Patch

```diff
diff --git a/sympy/sets/sets.py b/sympy/sets/sets.py
--- a/sympy/sets/sets.py
+++ b/sympy/sets/sets.py
@@ -737,6 +737,8 @@ def __new__(cls, start, end, left_open=False, right_open=False):
         if end == start and (left_open or right_open):
             return S.EmptySet
         if end == start and not (left_open or right_open):
+            if start == S.Infinity or start == S.NegativeInfinity:
+                return S.EmptySet
             return FiniteSet(end)
 
         # Make sure infinite interval end points are open.

```

## Test Patch

```diff
diff --git a/sympy/sets/tests/test_sets.py b/sympy/sets/tests/test_sets.py
--- a/sympy/sets/tests/test_sets.py
+++ b/sympy/sets/tests/test_sets.py
@@ -35,6 +35,8 @@ def test_interval_arguments():
     assert Interval(-oo, 0) == Interval(-oo, 0, True, False)
     assert Interval(-oo, 0).left_open is true
     assert Interval(oo, -oo) == S.EmptySet
+    assert Interval(oo, oo) == S.EmptySet
+    assert Interval(-oo, -oo) == S.EmptySet
 
     assert isinstance(Interval(1, 1), FiniteSet)
     e = Sum(x, (x, 1, 3))

```


## Code snippets

### 1 - sympy/series/order.py:

Start line: 1, End line: 122

```python
from __future__ import print_function, division

from sympy.core import S, sympify, Expr, Rational, Symbol, Dummy
from sympy.core import Add, Mul, expand_power_base, expand_log
from sympy.core.cache import cacheit
from sympy.core.compatibility import default_sort_key, is_sequence
from sympy.core.containers import Tuple
from sympy.utilities.iterables import uniq
from sympy.sets.sets import Complement


class Order(Expr):
    r""" Represents the limiting behavior of some function

    The order of a function characterizes the function based on the limiting
    behavior of the function as it goes to some limit. Only taking the limit
    point to be a number is currently supported. This is expressed in
    big O notation [1]_.

    The formal definition for the order of a function `g(x)` about a point `a`
    is such that `g(x) = O(f(x))` as `x \rightarrow a` if and only if for any
    `\delta > 0` there exists a `M > 0` such that `|g(x)| \leq M|f(x)|` for
    `|x-a| < \delta`.  This is equivalent to `\lim_{x \rightarrow a}
    \sup |g(x)/f(x)| < \infty`.

    Let's illustrate it on the following example by taking the expansion of
    `\sin(x)` about 0:

    .. math ::
        \sin(x) = x - x^3/3! + O(x^5)

    where in this case `O(x^5) = x^5/5! - x^7/7! + \cdots`. By the definition
    of `O`, for any `\delta > 0` there is an `M` such that:

    .. math ::
        |x^5/5! - x^7/7! + ....| <= M|x^5| \text{ for } |x| < \delta

    or by the alternate definition:

    .. math ::
        \lim_{x \rightarrow 0} | (x^5/5! - x^7/7! + ....) / x^5| < \infty

    which surely is true, because

    .. math ::
        \lim_{x \rightarrow 0} | (x^5/5! - x^7/7! + ....) / x^5| = 1/5!


    As it is usually used, the order of a function can be intuitively thought
    of representing all terms of powers greater than the one specified. For
    example, `O(x^3)` corresponds to any terms proportional to `x^3,
    x^4,\ldots` and any higher power. For a polynomial, this leaves terms
    proportional to `x^2`, `x` and constants.

    Examples
    ========

    >>> from sympy import O, oo, cos, pi
    >>> from sympy.abc import x, y

    >>> O(x + x**2)
    O(x)
    >>> O(x + x**2, (x, 0))
    O(x)
    >>> O(x + x**2, (x, oo))
    O(x**2, (x, oo))

    >>> O(1 + x*y)
    O(1, x, y)
    >>> O(1 + x*y, (x, 0), (y, 0))
    O(1, x, y)
    >>> O(1 + x*y, (x, oo), (y, oo))
    O(x*y, (x, oo), (y, oo))

    >>> O(1) in O(1, x)
    True
    >>> O(1, x) in O(1)
    False
    >>> O(x) in O(1, x)
    True
    >>> O(x**2) in O(x)
    True

    >>> O(x)*x
    O(x**2)
    >>> O(x) - O(x)
    O(x)
    >>> O(cos(x))
    O(1)
    >>> O(cos(x), (x, pi/2))
    O(x - pi/2, (x, pi/2))

    References
    ==========

    .. [1] `Big O notation <http://en.wikipedia.org/wiki/Big_O_notation>`_

    Notes
    =====

    In ``O(f(x), x)`` the expression ``f(x)`` is assumed to have a leading
    term.  ``O(f(x), x)`` is automatically transformed to
    ``O(f(x).as_leading_term(x),x)``.

        ``O(expr*f(x), x)`` is ``O(f(x), x)``

        ``O(expr, x)`` is ``O(1)``

        ``O(0, x)`` is 0.

    Multivariate O is also supported:

        ``O(f(x, y), x, y)`` is transformed to
        ``O(f(x, y).as_leading_term(x,y).as_leading_term(y), x, y)``

    In the multivariate case, it is assumed the limits w.r.t. the various
    symbols commute.

    If no symbols are passed then all symbols in the expression are used
    and the limit point is assumed to be zero.

    """
```
### 2 - sympy/calculus/util.py:

Start line: 457, End line: 608

```python
class AccumulationBounds(AtomicExpr):
    """
    # Note AccumulationBounds has an alias: AccumBounds

    AccumulationBounds represent an interval `[a, b]`, which is always closed
    at the ends. Here `a` and `b` can be any value from extended real numbers.

    The intended meaning of AccummulationBounds is to give an approximate
    location of the accumulation points of a real function at a limit point.

    Let `a` and `b` be reals such that a <= b.

    `\langle a, b\rangle = \{x \in \mathbb{R} \mid a \le x \le b\}`

    `\langle -\infty, b\rangle = \{x \in \mathbb{R} \mid x \le b\} \cup \{-\infty, \infty\}`

    `\langle a, \infty \rangle = \{x \in \mathbb{R} \mid a \le x\} \cup \{-\infty, \infty\}`

    `\langle -\infty, \infty \rangle = \mathbb{R} \cup \{-\infty, \infty\}`

    `oo` and `-oo` are added to the second and third definition respectively,
    since if either `-oo` or `oo` is an argument, then the other one should
    be included (though not as an end point). This is forced, since we have,
    for example, `1/AccumBounds(0, 1) = AccumBounds(1, oo)`, and the limit at
    `0` is not one-sided. As x tends to `0-`, then `1/x -> -oo`, so `-oo`
    should be interpreted as belonging to `AccumBounds(1, oo)` though it need
    not appear explicitly.

    In many cases it suffices to know that the limit set is bounded.
    However, in some other cases more exact information could be useful.
    For example, all accumulation values of cos(x) + 1 are non-negative.
    (AccumBounds(-1, 1) + 1 = AccumBounds(0, 2))

    A AccumulationBounds object is defined to be real AccumulationBounds,
    if its end points are finite reals.

    Let `X`, `Y` be real AccumulationBounds, then their sum, difference,
    product are defined to be the following sets:

    `X + Y = \{ x+y \mid x \in X \cap y \in Y\}`

    `X - Y = \{ x-y \mid x \in X \cap y \in Y\}`

    `X * Y = \{ x*y \mid x \in X \cap y \in Y\}`

    There is, however, no consensus on Interval division.

    `X / Y = \{ z \mid \exists x \in X, y \in Y \mid y \neq 0, z = x/y\}`

    Note: According to this definition the quotient of two AccumulationBounds
    may not be a AccumulationBounds object but rather a union of
    AccumulationBounds.

    Note
    ====

    The main focus in the interval arithmetic is on the simplest way to calculate
    upper and lower endpoints for the range of values of a function in one or more
    variables. These barriers are not necessarily the supremum or infimum, since
    the precise calculation of those values can be difficult or impossible.

    Examples
    ========

    >>> from sympy import AccumBounds, sin, exp, log, pi, E, S, oo
    >>> from sympy.abc import x

    >>> AccumBounds(0, 1) + AccumBounds(1, 2)
    <1, 3>

    >>> AccumBounds(0, 1) - AccumBounds(0, 2)
    <-2, 1>

    >>> AccumBounds(-2, 3)*AccumBounds(-1, 1)
    <-3, 3>

    >>> AccumBounds(1, 2)*AccumBounds(3, 5)
    <3, 10>

    The exponentiation of AccumulationBounds is defined
    as follows:

    If 0 does not belong to `X` or `n > 0` then

    `X^n = \{ x^n \mid x \in X\}`

    otherwise

    `X^n = \{ x^n \mid x \neq 0, x \in X\} \cup \{-\infty, \infty\}`

    Here for fractional `n`, the part of `X` resulting in a complex
    AccumulationBounds object is neglected.

    >>> AccumBounds(-1, 4)**(S(1)/2)
    <0, 2>

    >>> AccumBounds(1, 2)**2
    <1, 4>

    >>> AccumBounds(-1, oo)**(-1)
    <-oo, oo>

    Note: `<a, b>^2` is not same as `<a, b>*<a, b>`

    >>> AccumBounds(-1, 1)**2
    <0, 1>

    >>> AccumBounds(1, 3) < 4
    True

    >>> AccumBounds(1, 3) < -1
    False

    Some elementary functions can also take AccumulationBounds as input.
    A function `f` evaluated for some real AccumulationBounds `<a, b>`
    is defined as `f(\langle a, b\rangle) = \{ f(x) \mid a \le x \le b \}`

    >>> sin(AccumBounds(pi/6, pi/3))
    <1/2, sqrt(3)/2>

    >>> exp(AccumBounds(0, 1))
    <1, E>

    >>> log(AccumBounds(1, E))
    <0, 1>

    Some symbol in an expression can be substituted for a AccumulationBounds
    object. But it doesn't necessarily evaluate the AccumulationBounds for
    that expression.

    Same expression can be evaluated to different values depending upon
    the form it is used for substituion. For example:

    >>> (x**2 + 2*x + 1).subs(x, AccumBounds(-1, 1))
    <-1, 4>

    >>> ((x + 1)**2).subs(x, AccumBounds(-1, 1))
    <0, 4>

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Interval_arithmetic

    .. [2] http://fab.cba.mit.edu/classes/S62.12/docs/Hickey_interval.pdf

    Notes
    =====

    Do not use ``AccumulationBounds`` for floating point interval arithmetic
    calculations, use ``mpmath.iv`` instead.
    """
```
### 3 - sympy/core/numbers.py:

Start line: 2432, End line: 2531

```python
class Infinity(with_metaclass(Singleton, Number)):
    r"""Positive infinite quantity.

    In real analysis the symbol `\infty` denotes an unbounded
    limit: `x\to\infty` means that `x` grows without bound.

    Infinity is often used not only to define a limit but as a value
    in the affinely extended real number system.  Points labeled `+\infty`
    and `-\infty` can be added to the topological space of the real numbers,
    producing the two-point compactification of the real numbers.  Adding
    algebraic properties to this gives us the extended real numbers.

    Infinity is a singleton, and can be accessed by ``S.Infinity``,
    or can be imported as ``oo``.

    Examples
    ========

    >>> from sympy import oo, exp, limit, Symbol
    >>> 1 + oo
    oo
    >>> 42/oo
    0
    >>> x = Symbol('x')
    >>> limit(exp(x), x, oo)
    oo

    See Also
    ========

    NegativeInfinity, NaN

    References
    ==========

    .. [1] http://en.wikipedia.org/wiki/Infinity
    """

    is_commutative = True
    is_positive = True
    is_infinite = True
    is_number = True
    is_prime = False

    __slots__ = []

    def __new__(cls):
        return AtomicExpr.__new__(cls)

    def _latex(self, printer):
        return r"\infty"

    @_sympifyit('other', NotImplemented)
    def __add__(self, other):
        if isinstance(other, Number):
            if other is S.NegativeInfinity or other is S.NaN:
                return S.NaN
            elif other.is_Float:
                if other == Float('-inf'):
                    return S.NaN
                else:
                    return Float('inf')
            else:
                return S.Infinity
        return NotImplemented
    __radd__ = __add__

    @_sympifyit('other', NotImplemented)
    def __sub__(self, other):
        if isinstance(other, Number):
            if other is S.Infinity or other is S.NaN:
                return S.NaN
            elif other.is_Float:
                if other == Float('inf'):
                    return S.NaN
                else:
                    return Float('inf')
            else:
                return S.Infinity
        return NotImplemented

    @_sympifyit('other', NotImplemented)
    def __mul__(self, other):
        if isinstance(other, Number):
            if other is S.Zero or other is S.NaN:
                return S.NaN
            elif other.is_Float:
                if other == 0:
                    return S.NaN
                if other > 0:
                    return Float('inf')
                else:
                    return Float('-inf')
            else:
                if other > 0:
                    return S.Infinity
                else:
                    return S.NegativeInfinity
        return NotImplemented
    __rmul__ = __mul__
```
### 4 - sympy/calculus/util.py:

Start line: 779, End line: 833

```python
class AccumulationBounds(AtomicExpr):

    @_sympifyit('other', NotImplemented)
    def __div__(self, other):
        if isinstance(other, Expr):
            if isinstance(other, AccumBounds):
                if not S.Zero in other:
                    return self*AccumBounds(1/other.max, 1/other.min)

                if S.Zero in self and S.Zero in other:
                    if self.min.is_zero and other.min.is_zero:
                        return AccumBounds(0, oo)
                    if self.max.is_zero and other.min.is_zero:
                        return AccumBounds(-oo, 0)
                    return AccumBounds(-oo, oo)

                if self.max.is_negative:
                    if other.min.is_negative:
                        if other.max.is_zero:
                            return AccumBounds(self.max/other.min, oo)
                        if other.max.is_positive:
                            # the actual answer is a Union of AccumBounds,
                            # Union(AccumBounds(-oo, self.max/other.max),
                            #       AccumBounds(self.max/other.min, oo))
                            return AccumBounds(-oo, oo)

                    if other.min.is_zero and other.max.is_positive:
                        return AccumBounds(-oo, self.max/other.max)

                if self.min.is_positive:
                    if other.min.is_negative:
                        if other.max.is_zero:
                            return AccumBounds(-oo, self.min/other.min)
                        if other.max.is_positive:
                            # the actual answer is a Union of AccumBounds,
                            # Union(AccumBounds(-oo, self.min/other.min),
                            #       AccumBounds(self.min/other.max, oo))
                            return AccumBounds(-oo, oo)

                    if other.min.is_zero and other.max.is_positive:
                        return AccumBounds(self.min/other.max, oo)

            elif other.is_real:
                if other is S.Infinity or other is S.NegativeInfinity:
                    if self == AccumBounds(-oo, oo):
                        return AccumBounds(-oo, oo)
                    if self.max is S.Infinity:
                        return AccumBounds(Min(0, other), Max(0, other))
                    if self.min is S.NegativeInfinity:
                        return AccumBounds(Min(0, -other), Max(0, -other))
                if other.is_positive:
                    return AccumBounds(self.min/other, self.max/other)
                elif other.is_negative:
                    return AccumBounds(self.max/other, self.min/other)
            return Mul(self, 1/other, evaluate=False)

        return NotImplemented
```
### 5 - sympy/concrete/summations.py:

Start line: 394, End line: 484

```python
class Sum(AddWithLimits, ExprWithIntLimits):

    def is_convergent(self):
        # ... other code
        if lower_limit is S.NegativeInfinity:
            if upper_limit is S.Infinity:
                return Sum(sequence_term, (sym, 0, S.Infinity)).is_convergent() and \
                        Sum(sequence_term, (sym, S.NegativeInfinity, 0)).is_convergent()
            sequence_term = simplify(sequence_term.xreplace({sym: -sym}))
            lower_limit = -upper_limit
            upper_limit = S.Infinity

        interval = Interval(lower_limit, upper_limit)

        # Piecewise function handle
        if sequence_term.is_Piecewise:
            for func_cond in sequence_term.args:
                if func_cond[1].func is Ge or func_cond[1].func is Gt or func_cond[1] == True:
                    return Sum(func_cond[0], (sym, lower_limit, upper_limit)).is_convergent()
            return S.true

        ###  -------- Divergence test ----------- ###
        try:
            lim_val = limit(sequence_term, sym, upper_limit)
            if lim_val.is_number and lim_val is not S.Zero:
                return S.false
        except NotImplementedError:
            pass

        try:
            lim_val_abs = limit(abs(sequence_term), sym, upper_limit)
            if lim_val_abs.is_number and lim_val_abs is not S.Zero:
                return S.false
        except NotImplementedError:
            pass

        order = O(sequence_term, (sym, S.Infinity))

        ### --------- p-series test (1/n**p) ---------- ###
        p1_series_test = order.expr.match(sym**p)
        if p1_series_test is not None:
            if p1_series_test[p] < -1:
                return S.true
            if p1_series_test[p] > -1:
                return S.false

        p2_series_test = order.expr.match((1/sym)**p)
        if p2_series_test is not None:
            if p2_series_test[p] > 1:
                return S.true
            if p2_series_test[p] < 1:
                return S.false

        ### ----------- root test ---------------- ###
        lim = Limit(abs(sequence_term)**(1/sym), sym, S.Infinity)
        lim_evaluated = lim.doit()
        if lim_evaluated.is_number:
            if lim_evaluated < 1:
                return S.true
            if lim_evaluated > 1:
                return S.false

        ### ------------- alternating series test ----------- ###
        dict_val = sequence_term.match((-1)**(sym + p)*q)
        if not dict_val[p].has(sym) and is_decreasing(dict_val[q], interval):
            return S.true

        ### ------------- comparison test ------------- ###
        # (1/log(n)**p) comparison
        log_test = order.expr.match(1/(log(sym)**p))
        if log_test is not None:
            return S.false

        # (1/(n*log(n)**p)) comparison
        log_n_test = order.expr.match(1/(sym*(log(sym))**p))
        if log_n_test is not None:
            if log_n_test[p] > 1:
                return S.true
            return S.false

        # (1/(n*log(n)*log(log(n))*p)) comparison
        log_log_n_test = order.expr.match(1/(sym*(log(sym)*log(log(sym))**p)))
        if log_log_n_test is not None:
            if log_log_n_test[p] > 1:
                return S.true
            return S.false

        # (1/(n**p*log(n))) comparison
        n_log_test = order.expr.match(1/(sym**p*log(sym)))
        if n_log_test is not None:
            if n_log_test[p] > 1:
                return S.true
            return S.false

        ### ------------- integral test -------------- ###
        # ... other code
```
### 6 - sympy/series/limits.py:

Start line: 75, End line: 119

```python
class Limit(Expr):
    """Represents an unevaluated limit.

    Examples
    ========

    >>> from sympy import Limit, sin, Symbol
    >>> from sympy.abc import x
    >>> Limit(sin(x)/x, x, 0)
    Limit(sin(x)/x, x, 0)
    >>> Limit(1/x, x, 0, dir="-")
    Limit(1/x, x, 0, dir='-')

    """

    def __new__(cls, e, z, z0, dir="+"):
        e = sympify(e)
        z = sympify(z)
        z0 = sympify(z0)

        if z0 is S.Infinity:
            dir = "-"
        elif z0 is S.NegativeInfinity:
            dir = "+"

        if isinstance(dir, string_types):
            dir = Symbol(dir)
        elif not isinstance(dir, Symbol):
            raise TypeError("direction must be of type basestring or Symbol, not %s" % type(dir))
        if str(dir) not in ('+', '-'):
            raise ValueError(
                "direction must be either '+' or '-', not %s" % dir)

        obj = Expr.__new__(cls)
        obj._args = (e, z, z0, dir)
        return obj


    @property
    def free_symbols(self):
        e = self.args[0]
        isyms = e.free_symbols
        isyms.difference_update(self.args[1].free_symbols)
        isyms.update(self.args[2].free_symbols)
        return isyms
```
### 7 - sympy/series/gruntz.py:

Start line: 407, End line: 439

```python
@debug
@timeit
@cacheit
def limitinf(e, x):
    """Limit e(x) for x-> oo"""
    #rewrite e in terms of tractable functions only
    e = e.rewrite('tractable', deep=True)

    if not e.has(x):
        return e  # e is a constant
    if e.has(Order):
        e = e.expand().removeO()
    if not x.is_positive:
        # We make sure that x.is_positive is True so we
        # get all the correct mathematical behavior from the expression.
        # We need a fresh variable.
        p = Dummy('p', positive=True, finite=True)
        e = e.subs(x, p)
        x = p
    c0, e0 = mrv_leadterm(e, x)
    sig = sign(e0, x)
    if sig == 1:
        return S.Zero  # e0>0: lim f = 0
    elif sig == -1:  # e0<0: lim f = +-oo (the sign depends on the sign of c0)
        if c0.match(I*Wild("a", exclude=[I])):
            return c0*oo
        s = sign(c0, x)
        #the leading term shouldn't be 0:
        if s == 0:
            raise ValueError("Leading term should not be 0")
        return s*oo
    elif sig == 0:
        return limitinf(c0, x)  # e0=0: lim f = lim c0
```
### 8 - examples/beginner/limits_examples.py:

Start line: 1, End line: 42

```python
#!/usr/bin/env python

"""Limits Example

Demonstrates limits.
"""

from sympy import exp, log, Symbol, Rational, sin, limit, sqrt, oo


def sqrt3(x):
    return x**Rational(1, 3)


def show(computed, correct):
    print("computed:", computed, "correct:", correct)


def main():
    x = Symbol("x")
    a = Symbol("a")
    h = Symbol("h")

    show( limit(sqrt(x**2 - 5*x + 6) - x, x, oo), -Rational(5)/2 )

    show( limit(x*(sqrt(x**2 + 1) - x), x, oo), Rational(1)/2 )

    show( limit(x - sqrt3(x**3 - 1), x, oo), Rational(0) )

    show( limit(log(1 + exp(x))/x, x, -oo), Rational(0) )

    show( limit(log(1 + exp(x))/x, x, oo), Rational(1) )

    show( limit(sin(3*x)/x, x, 0), Rational(3) )

    show( limit(sin(5*x)/sin(2*x), x, 0), Rational(5)/2 )

    show( limit(((x - 1)/(x + 1))**x, x, oo), exp(-2))

if __name__ == "__main__":
    main()
```
### 9 - sympy/calculus/util.py:

Start line: 864, End line: 933

```python
class AccumulationBounds(AtomicExpr):

    @_sympifyit('other', NotImplemented)
    def __pow__(self, other):
        from sympy.functions.elementary.miscellaneous import real_root
        if isinstance(other, Expr):
            if other is S.Infinity:
                if self.min.is_nonnegative:
                    if self.max < 1:
                        return S.Zero
                    if self.min > 1:
                        return S.Infinity
                    return AccumBounds(0, oo)
                elif self.max.is_negative:
                    if self.min > -1:
                        return S.Zero
                    if self.max < -1:
                        return FiniteSet(-oo, oo)
                    return AccumBounds(-oo, oo)
                else:
                    if self.min > -1:
                        if self.max < 1:
                            return S.Zero
                        return AccumBounds(0, oo)
                    return AccumBounds(-oo, oo)

            if other is S.NegativeInfinity:
                return (1/self)**oo

            if other.is_real and other.is_number:
                if other.is_zero:
                    return S.One

                if other.is_Integer:
                    if self.min.is_positive:
                        return AccumBounds(Min(self.min**other, self.max**other),
                            Max(self.min**other, self.max**other))
                    elif self.max.is_negative:
                        return AccumBounds(Min(self.max**other, self.min**other),
                                    Max(self.max**other, self.min**other))

                    if other % 2 == 0:
                        if other.is_negative:
                            if self.min.is_zero:
                                return AccumBounds(self.max**other, oo)
                            if self.max.is_zero:
                                return AccumBounds(self.min**other, oo)
                            return AccumBounds(0, oo)
                        return AccumBounds(S.Zero,
                            Max(self.min**other, self.max**other))
                    else:
                        if other.is_negative:
                            if self.min.is_zero:
                                return AccumBounds(self.max**other, oo)
                            if self.max.is_zero:
                                return AccumBounds(-oo, self.min**other)
                            return AccumBounds(-oo, oo)
                        return AccumBounds(self.min**other, self.max**other)

                num, den = other.as_numer_denom()
                if num == S(1):
                    if den % 2 == 0:
                        if S.Zero in self:
                            if self.min.is_negative:
                                return AccumBounds(0, real_root(self.max, den))
                    return AccumBounds(real_root(self.min, den),
                                    real_root(self.max, den))
                num_pow = self**num
                return num_pow**(1/den)
            return Pow(self, other, evaluate=False)

        return NotImplemented
```
### 10 - sympy/calculus/util.py:

Start line: 720, End line: 735

```python
class AccumulationBounds(AtomicExpr):

    @_sympifyit('other', NotImplemented)
    def __sub__(self, other):
        if isinstance(other, Expr):
            if isinstance(other, AccumBounds):
                return AccumBounds(Add(self.min, -other.max), Add(self.max, -other.min))
            if other is S.NegativeInfinity and self.min is S.NegativeInfinity or \
                    other is S.Infinity and self.max is S.Infinity:
                return AccumBounds(-oo, oo)
            elif other.is_real:
                return AccumBounds(Add(self.min, -other), Add(self.max, - other))
            return Add(self, -other, evaluate=False)
        return NotImplemented

    @_sympifyit('other', NotImplemented)
    def __rsub__(self, other):
        return self.__neg__() + other
```
