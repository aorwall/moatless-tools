# sympy__sympy-20801

| **sympy/sympy** | `e11d3fed782146eebbffdc9ced0364b223b84b6c` |
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
diff --git a/sympy/core/numbers.py b/sympy/core/numbers.py
--- a/sympy/core/numbers.py
+++ b/sympy/core/numbers.py
@@ -1386,8 +1386,6 @@ def __eq__(self, other):
             other = _sympify(other)
         except SympifyError:
             return NotImplemented
-        if not self:
-            return not other
         if isinstance(other, Boolean):
             return False
         if other.is_NumberSymbol:
@@ -1408,6 +1406,8 @@ def __eq__(self, other):
             # the mpf tuples
             ompf = other._as_mpf_val(self._prec)
             return bool(mlib.mpf_eq(self._mpf_, ompf))
+        if not self:
+            return not other
         return False    # Float != non-Number
 
     def __ne__(self, other):

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sympy/core/numbers.py | 1389 | 1390 | - | - | -
| sympy/core/numbers.py | 1411 | 1411 | - | - | -


## Problem Statement

```
S(0.0) == S.false returns True
This issue is related to those listed in #20033. 

As shown by @sayandip18, comparing `S.false` to `S(0.0)` returns 2 different results depending on the order in which they are compared:

\`\`\`pycon
>>> from sympy import *
>>> S(0.0) == S.false
True
>>> S.false == S(0.0)
False
\`\`\`
Based on the results of comparison to `S(0)`:

\`\`\`pycon
>>> S(0) == S.false
False
>>> S.false == S(0)
False
\`\`\`
I assume we would want `S(0.0) == S.false` to return True as well?

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 sympy/logic/boolalg.py | 350 | 414| 355 | 355 | 24905 | 
| 2 | 1 sympy/logic/boolalg.py | 233 | 347| 937 | 1292 | 24905 | 
| 3 | 2 sympy/core/basic.py | 316 | 360| 364 | 1656 | 40758 | 
| 4 | 3 sympy/core/relational.py | 1361 | 1441| 758 | 2414 | 51913 | 
| 5 | 3 sympy/core/relational.py | 255 | 301| 375 | 2789 | 51913 | 
| 6 | 3 sympy/core/relational.py | 1253 | 1360| 786 | 3575 | 51913 | 
| 7 | 4 sympy/core/expr.py | 704 | 779| 715 | 4290 | 85607 | 
| 8 | 4 sympy/logic/boolalg.py | 417 | 453| 281 | 4571 | 85607 | 
| 9 | 4 sympy/core/relational.py | 1225 | 1250| 217 | 4788 | 85607 | 
| 10 | 4 sympy/logic/boolalg.py | 104 | 132| 242 | 5030 | 85607 | 
| 11 | 4 sympy/core/expr.py | 780 | 837| 554 | 5584 | 85607 | 
| 12 | 5 sympy/physics/control/lti.py | 729 | 752| 230 | 5814 | 98038 | 
| 13 | 5 sympy/core/relational.py | 1135 | 1223| 709 | 6523 | 98038 | 
| 14 | 6 sympy/core/kind.py | 162 | 187| 141 | 6664 | 100623 | 
| 15 | 6 sympy/physics/control/lti.py | 754 | 777| 231 | 6895 | 100623 | 
| 16 | 6 sympy/logic/boolalg.py | 1281 | 1313| 280 | 7175 | 100623 | 
| 17 | 6 sympy/core/basic.py | 1862 | 1898| 311 | 7486 | 100623 | 
| 18 | 7 sympy/sets/handlers/comparison.py | 1 | 54| 470 | 7956 | 101093 | 
| 19 | 7 sympy/physics/control/lti.py | 779 | 802| 204 | 8160 | 101093 | 
| 20 | 7 sympy/core/basic.py | 362 | 417| 349 | 8509 | 101093 | 
| 21 | 7 sympy/logic/boolalg.py | 1469 | 1509| 367 | 8876 | 101093 | 
| 22 | 7 sympy/logic/boolalg.py | 1233 | 1280| 343 | 9219 | 101093 | 
| 23 | 7 sympy/logic/boolalg.py | 1338 | 1375| 325 | 9544 | 101093 | 
| 24 | 7 sympy/core/expr.py | 532 | 605| 721 | 10265 | 101093 | 
| 25 | 8 sympy/integrals/transforms.py | 957 | 977| 215 | 10480 | 118105 | 
| 26 | 8 sympy/logic/boolalg.py | 455 | 475| 206 | 10686 | 118105 | 
| 27 | 8 sympy/core/relational.py | 552 | 586| 280 | 10966 | 118105 | 
| 28 | 9 sympy/parsing/sympy_parser.py | 1010 | 1047| 234 | 11200 | 126358 | 
| 29 | 9 sympy/core/basic.py | 189 | 233| 343 | 11543 | 126358 | 
| 30 | 10 sympy/core/assumptions.py | 283 | 316| 337 | 11880 | 130666 | 
| 31 | 11 sympy/sets/contains.py | 1 | 50| 323 | 12203 | 130989 | 
| 32 | 11 sympy/logic/boolalg.py | 1377 | 1395| 198 | 12401 | 130989 | 
| 33 | 11 sympy/core/basic.py | 639 | 684| 374 | 12775 | 130989 | 
| 34 | 12 sympy/series/order.py | 330 | 408| 697 | 13472 | 135066 | 
| 35 | 12 sympy/logic/boolalg.py | 1431 | 1467| 350 | 13822 | 135066 | 
| 36 | 12 sympy/core/expr.py | 2546 | 2617| 570 | 14392 | 135066 | 
| 37 | 13 sympy/core/logic.py | 115 | 148| 239 | 14631 | 137737 | 
| 38 | 14 sympy/sets/sets.py | 293 | 337| 347 | 14978 | 155362 | 
| 39 | 14 sympy/core/relational.py | 1078 | 1132| 334 | 15312 | 155362 | 
| 40 | 14 sympy/logic/boolalg.py | 720 | 793| 645 | 15957 | 155362 | 
| 41 | 15 sympy/assumptions/handlers/calculus.py | 112 | 165| 406 | 16363 | 157142 | 
| 42 | 15 sympy/core/expr.py | 607 | 702| 813 | 17176 | 157142 | 
| 43 | 15 sympy/logic/boolalg.py | 1316 | 1337| 117 | 17293 | 157142 | 
| 44 | 15 sympy/core/relational.py | 389 | 421| 274 | 17567 | 157142 | 
| 45 | 15 sympy/logic/boolalg.py | 937 | 972| 267 | 17834 | 157142 | 
| 46 | 15 sympy/core/relational.py | 768 | 998| 2058 | 19892 | 157142 | 
| 47 | 16 sympy/utilities/iterables.py | 21 | 54| 288 | 20180 | 178969 | 
| 48 | 16 sympy/core/expr.py | 2754 | 2814| 453 | 20633 | 178969 | 
| 49 | 16 sympy/core/expr.py | 398 | 449| 384 | 21017 | 178969 | 


## Missing Patch Files

 * 1: sympy/core/numbers.py

## Patch

```diff
diff --git a/sympy/core/numbers.py b/sympy/core/numbers.py
--- a/sympy/core/numbers.py
+++ b/sympy/core/numbers.py
@@ -1386,8 +1386,6 @@ def __eq__(self, other):
             other = _sympify(other)
         except SympifyError:
             return NotImplemented
-        if not self:
-            return not other
         if isinstance(other, Boolean):
             return False
         if other.is_NumberSymbol:
@@ -1408,6 +1406,8 @@ def __eq__(self, other):
             # the mpf tuples
             ompf = other._as_mpf_val(self._prec)
             return bool(mlib.mpf_eq(self._mpf_, ompf))
+        if not self:
+            return not other
         return False    # Float != non-Number
 
     def __ne__(self, other):

```

## Test Patch

```diff
diff --git a/sympy/core/tests/test_numbers.py b/sympy/core/tests/test_numbers.py
--- a/sympy/core/tests/test_numbers.py
+++ b/sympy/core/tests/test_numbers.py
@@ -436,6 +436,7 @@ def eq(a, b):
     a = Float(2) ** Float(4)
     assert eq(a.evalf(), Float(16))
     assert (S(.3) == S(.5)) is False
+
     mpf = (0, 5404319552844595, -52, 53)
     x_str =  Float((0, '13333333333333', -52, 53))
     x2_str = Float((0, '26666666666666', -53, 54))
@@ -582,7 +583,12 @@ def teq(a):
     for i, a in zip(u, v):
         assert Float(i) is a
 
-
+def test_zero_not_false():
+    # https://github.com/sympy/sympy/issues/20796
+    assert (S(0.0) == S.false) is False
+    assert (S.false == S(0.0)) is False
+    assert (S(0) == S.false) is False
+    assert (S.false == S(0)) is False
 
 @conserve_mpmath_dps
 def test_float_mpf():

```


## Code snippets

### 1 - sympy/logic/boolalg.py:

Start line: 350, End line: 414

```python
class BooleanFalse(BooleanAtom, metaclass=Singleton):
    """
    SymPy version of False, a singleton that can be accessed via S.false.

    This is the SymPy version of False, for use in the logic module. The
    primary advantage of using false instead of False is that shorthand boolean
    operations like ~ and >> will work as expected on this class, whereas with
    False they act bitwise on 0. Functions in the logic module will return this
    class when they evaluate to false.

    Notes
    ======

    See note in :py:class`sympy.logic.boolalg.BooleanTrue`

    Examples
    ========

    >>> from sympy import sympify, true, false, Or
    >>> sympify(False)
    False
    >>> _ is False, _ is false
    (False, True)

    >>> Or(true, false)
    True
    >>> _ is true
    True

    Python operators give a boolean result for false but a
    bitwise result for False

    >>> ~false, ~False
    (True, -1)
    >>> false >> false, False >> False
    (True, 0)

    See Also
    ========

    sympy.logic.boolalg.BooleanTrue

    """
    def __bool__(self):
        return False

    def __hash__(self):
        return hash(False)

    @property
    def negated(self):
        return S.true

    def as_set(self):
        """
        Rewrite logic operators and relationals in terms of real sets.

        Examples
        ========

        >>> from sympy import false
        >>> false.as_set()
        EmptySet
        """
        return S.EmptySet
```
### 2 - sympy/logic/boolalg.py:

Start line: 233, End line: 347

```python
class BooleanTrue(BooleanAtom, metaclass=Singleton):
    """
    SymPy version of True, a singleton that can be accessed via S.true.

    This is the SymPy version of True, for use in the logic module. The
    primary advantage of using true instead of True is that shorthand boolean
    operations like ~ and >> will work as expected on this class, whereas with
    True they act bitwise on 1. Functions in the logic module will return this
    class when they evaluate to true.

    Notes
    =====

    There is liable to be some confusion as to when ``True`` should
    be used and when ``S.true`` should be used in various contexts
    throughout SymPy. An important thing to remember is that
    ``sympify(True)`` returns ``S.true``. This means that for the most
    part, you can just use ``True`` and it will automatically be converted
    to ``S.true`` when necessary, similar to how you can generally use 1
    instead of ``S.One``.

    The rule of thumb is:

    "If the boolean in question can be replaced by an arbitrary symbolic
    ``Boolean``, like ``Or(x, y)`` or ``x > 1``, use ``S.true``.
    Otherwise, use ``True``"

    In other words, use ``S.true`` only on those contexts where the
    boolean is being used as a symbolic representation of truth.
    For example, if the object ends up in the ``.args`` of any expression,
    then it must necessarily be ``S.true`` instead of ``True``, as
    elements of ``.args`` must be ``Basic``. On the other hand,
    ``==`` is not a symbolic operation in SymPy, since it always returns
    ``True`` or ``False``, and does so in terms of structural equality
    rather than mathematical, so it should return ``True``. The assumptions
    system should use ``True`` and ``False``. Aside from not satisfying
    the above rule of thumb, the assumptions system uses a three-valued logic
    (``True``, ``False``, ``None``), whereas ``S.true`` and ``S.false``
    represent a two-valued logic. When in doubt, use ``True``.

    "``S.true == True is True``."

    While "``S.true is True``" is ``False``, "``S.true == True``"
    is ``True``, so if there is any doubt over whether a function or
    expression will return ``S.true`` or ``True``, just use ``==``
    instead of ``is`` to do the comparison, and it will work in either
    case.  Finally, for boolean flags, it's better to just use ``if x``
    instead of ``if x is True``. To quote PEP 8:

    Don't compare boolean values to ``True`` or ``False``
    using ``==``.

    * Yes:   ``if greeting:``
    * No:    ``if greeting == True:``
    * Worse: ``if greeting is True:``

    Examples
    ========

    >>> from sympy import sympify, true, false, Or
    >>> sympify(True)
    True
    >>> _ is True, _ is true
    (False, True)

    >>> Or(true, false)
    True
    >>> _ is true
    True

    Python operators give a boolean result for true but a
    bitwise result for True

    >>> ~true, ~True
    (False, -2)
    >>> true >> true, True >> True
    (True, 0)

    Python operators give a boolean result for true but a
    bitwise result for True

    >>> ~true, ~True
    (False, -2)
    >>> true >> true, True >> True
    (True, 0)

    See Also
    ========

    sympy.logic.boolalg.BooleanFalse

    """
    def __bool__(self):
        return True

    def __hash__(self):
        return hash(True)

    @property
    def negated(self):
        return S.false

    def as_set(self):
        """
        Rewrite logic operators and relationals in terms of real sets.

        Examples
        ========

        >>> from sympy import true
        >>> true.as_set()
        UniversalSet

        """
        return S.UniversalSet
```
### 3 - sympy/core/basic.py:

Start line: 316, End line: 360

```python
class Basic(Printable, metaclass=ManagedProperties):

    def __eq__(self, other):
        """Return a boolean indicating whether a == b on the basis of
        their symbolic trees.

        This is the same as a.compare(b) == 0 but faster.

        Notes
        =====

        If a class that overrides __eq__() needs to retain the
        implementation of __hash__() from a parent class, the
        interpreter must be told this explicitly by setting __hash__ =
        <ParentClass>.__hash__. Otherwise the inheritance of __hash__()
        will be blocked, just as if __hash__ had been explicitly set to
        None.

        References
        ==========

        from http://docs.python.org/dev/reference/datamodel.html#object.__hash__
        """
        if self is other:
            return True

        tself = type(self)
        tother = type(other)
        if tself is not tother:
            try:
                other = _sympify(other)
                tother = type(other)
            except SympifyError:
                return NotImplemented

            # As long as we have the ordering of classes (sympy.core),
            # comparing types will be slow in Python 2, because it uses
            # __cmp__. Until we can remove it
            # (https://github.com/sympy/sympy/issues/4269), we only compare
            # types in Python 2 directly if they actually have __ne__.
            if type(tself).__ne__ is not type.__ne__:
                if tself != tother:
                    return False
            elif tself is not tother:
                return False

        return self._hashable_content() == other._hashable_content()
```
### 4 - sympy/core/relational.py:

Start line: 1361, End line: 1441

```python
def is_eq(lhs, rhs):
    # ... other code
    if lhs == rhs:
        return True  # e.g. True == True
    elif all(isinstance(i, BooleanAtom) for i in (rhs, lhs)):
        return False  # True != False
    elif not (lhs.is_Symbol or rhs.is_Symbol) and (
        isinstance(lhs, Boolean) !=
        isinstance(rhs, Boolean)):
        return False  # only Booleans can equal Booleans

    if lhs.is_infinite or rhs.is_infinite:
        if fuzzy_xor([lhs.is_infinite, rhs.is_infinite]):
            return False
        if fuzzy_xor([lhs.is_extended_real, rhs.is_extended_real]):
            return False
        if fuzzy_and([lhs.is_extended_real, rhs.is_extended_real]):
            return fuzzy_xor([lhs.is_extended_positive, fuzzy_not(rhs.is_extended_positive)])

        # Try to split real/imaginary parts and equate them
        I = S.ImaginaryUnit

        def split_real_imag(expr):
            real_imag = lambda t: (
                'real' if t.is_extended_real else
                'imag' if (I * t).is_extended_real else None)
            return sift(Add.make_args(expr), real_imag)

        lhs_ri = split_real_imag(lhs)
        if not lhs_ri[None]:
            rhs_ri = split_real_imag(rhs)
            if not rhs_ri[None]:
                eq_real = Eq(Add(*lhs_ri['real']), Add(*rhs_ri['real']))
                eq_imag = Eq(I * Add(*lhs_ri['imag']), I * Add(*rhs_ri['imag']))
                return fuzzy_and(map(fuzzy_bool, [eq_real, eq_imag]))

        # Compare e.g. zoo with 1+I*oo by comparing args
        arglhs = arg(lhs)
        argrhs = arg(rhs)
        # Guard against Eq(nan, nan) -> Falsesymp
        if not (arglhs == S.NaN and argrhs == S.NaN):
            return fuzzy_bool(Eq(arglhs, argrhs))

    if all(isinstance(i, Expr) for i in (lhs, rhs)):
        # see if the difference evaluates
        dif = lhs - rhs
        z = dif.is_zero
        if z is not None:
            if z is False and dif.is_commutative:  # issue 10728
                return False
            if z:
                return True

        n2 = _n2(lhs, rhs)
        if n2 is not None:
            return _sympify(n2 == 0)

        # see if the ratio evaluates
        n, d = dif.as_numer_denom()
        rv = None
        if n.is_zero:
            rv = d.is_nonzero
        elif n.is_finite:
            if d.is_infinite:
                rv = True
            elif n.is_zero is False:
                rv = d.is_infinite
                if rv is None:
                    # if the condition that makes the denominator
                    # infinite does not make the original expression
                    # True then False can be returned
                    l, r = clear_coefficients(d, S.Infinity)
                    args = [_.subs(l, r) for _ in (lhs, rhs)]
                    if args != [lhs, rhs]:
                        rv = fuzzy_bool(Eq(*args))
                        if rv is True:
                            rv = None
        elif any(a.is_infinite for a in Add.make_args(n)):
            # (inf or nan)/x != 0
            rv = False
        if rv is not None:
            return rv
```
### 5 - sympy/core/relational.py:

Start line: 255, End line: 301

```python
class Relational(Boolean, EvalfMixin):

    def equals(self, other, failing_expression=False):
        """Return True if the sides of the relationship are mathematically
        identical and the type of relationship is the same.
        If failing_expression is True, return the expression whose truth value
        was unknown."""
        if isinstance(other, Relational):
            if self == other or self.reversed == other:
                return True
            a, b = self, other
            if a.func in (Eq, Ne) or b.func in (Eq, Ne):
                if a.func != b.func:
                    return False
                left, right = [i.equals(j,
                                        failing_expression=failing_expression)
                               for i, j in zip(a.args, b.args)]
                if left is True:
                    return right
                if right is True:
                    return left
                lr, rl = [i.equals(j, failing_expression=failing_expression)
                          for i, j in zip(a.args, b.reversed.args)]
                if lr is True:
                    return rl
                if rl is True:
                    return lr
                e = (left, right, lr, rl)
                if all(i is False for i in e):
                    return False
                for i in e:
                    if i not in (True, False):
                        return i
            else:
                if b.func != a.func:
                    b = b.reversed
                if a.func != b.func:
                    return False
                left = a.lhs.equals(b.lhs,
                                    failing_expression=failing_expression)
                if left is False:
                    return False
                right = a.rhs.equals(b.rhs,
                                     failing_expression=failing_expression)
                if right is False:
                    return False
                if left is True:
                    return right
                return left
```
### 6 - sympy/core/relational.py:

Start line: 1253, End line: 1360

```python
def is_eq(lhs, rhs):
    """
    Fuzzy bool representing mathematical equality between lhs and rhs.

    Parameters
    ==========

    lhs: Expr
        The left-hand side of the expression, must be sympified.

    rhs: Expr
        The right-hand side of the expression, must be sympified.

    Returns
    =======

    True if lhs is equal to rhs, false is lhs is not equal to rhs, and
    None if the comparison between lhs and rhs is indeterminate.

    Explanation
    ===========

    This function is intended to give a relatively fast determination and deliberately does not attempt slow
    calculations that might help in obtaining a determination of True or False in more difficult cases.

    InEquality classes, such as Lt, Gt, etc. Use one of is_ge, is_le, etc.
    To implement comparisons with ``Gt(a, b)`` or ``a > b`` etc for an ``Expr`` subclass
    it is only necessary to define a dispatcher method for ``_eval_is_ge`` like

    >>> from sympy.core.relational import is_eq
    >>> from sympy.core.relational import is_neq
    >>> from sympy import S, Basic, Eq, sympify
    >>> from sympy.abc import x
    >>> from sympy.multipledispatch import dispatch
    >>> class MyBasic(Basic):
    ...     def __new__(cls, arg):
    ...         return Basic.__new__(cls, sympify(arg))
    ...     @property
    ...     def value(self):
    ...         return self.args[0]
    ...
    >>> @dispatch(MyBasic, MyBasic)
    ... def _eval_is_eq(a, b):
    ...     return is_eq(a.value, b.value)
    ...
    >>> a = MyBasic(1)
    >>> b = MyBasic(1)
    >>> a == b
    True
    >>> Eq(a, b)
    True
    >>> a != b
    False
    >>> is_eq(a, b)
    True


    Examples
    ========



    >>> is_eq(S(0), S(0))
    True
    >>> Eq(0, 0)
    True
    >>> is_neq(S(0), S(0))
    False
    >>> is_eq(S(0), S(2))
    False
    >>> Eq(0, 2)
    False
    >>> is_neq(S(0), S(2))
    True
    >>> is_eq(S(0), x)

    >>> Eq(S(0), x)
    Eq(0, x)



    """
    from sympy.core.add import Add
    from sympy.functions.elementary.complexes import arg
    from sympy.simplify.simplify import clear_coefficients
    from sympy.utilities.iterables import sift

    # here, _eval_Eq is only called for backwards compatibility
    # new code should use is_eq with multiple dispatch as
    # outlined in the docstring
    for side1, side2 in (lhs, rhs), (rhs, lhs):
        eval_func = getattr(side1, '_eval_Eq', None)
        if eval_func is not None:
            retval = eval_func(side2)
            if retval is not None:
                return retval

    retval = _eval_is_eq(lhs, rhs)
    if retval is not None:
        return retval

    if dispatch(type(lhs), type(rhs)) != dispatch(type(rhs), type(lhs)):
        retval = _eval_is_eq(rhs, lhs)
        if retval is not None:
            return retval

    # retval is still None, so go through the equality logic
    # If expressions have the same structure, they must be equal.
    # ... other code
```
### 7 - sympy/core/expr.py:

Start line: 704, End line: 779

```python
@sympify_method_args
class Expr(Basic, EvalfMixin):

    def equals(self, other, failing_expression=False):
        """Return True if self == other, False if it doesn't, or None. If
        failing_expression is True then the expression which did not simplify
        to a 0 will be returned instead of None.

        Explanation
        ===========

        If ``self`` is a Number (or complex number) that is not zero, then
        the result is False.

        If ``self`` is a number and has not evaluated to zero, evalf will be
        used to test whether the expression evaluates to zero. If it does so
        and the result has significance (i.e. the precision is either -1, for
        a Rational result, or is greater than 1) then the evalf value will be
        used to return True or False.

        """
        from sympy.simplify.simplify import nsimplify, simplify
        from sympy.solvers.solvers import solve
        from sympy.polys.polyerrors import NotAlgebraic
        from sympy.polys.numberfields import minimal_polynomial

        other = sympify(other)
        if self == other:
            return True

        # they aren't the same so see if we can make the difference 0;
        # don't worry about doing simplification steps one at a time
        # because if the expression ever goes to 0 then the subsequent
        # simplification steps that are done will be very fast.
        diff = factor_terms(simplify(self - other), radical=True)

        if not diff:
            return True

        if not diff.has(Add, Mod):
            # if there is no expanding to be done after simplifying
            # then this can't be a zero
            return False

        constant = diff.is_constant(simplify=False, failing_number=True)

        if constant is False:
            return False

        if not diff.is_number:
            if constant is None:
                # e.g. unless the right simplification is done, a symbolic
                # zero is possible (see expression of issue 6829: without
                # simplification constant will be None).
                return

        if constant is True:
            # this gives a number whether there are free symbols or not
            ndiff = diff._random()
            # is_comparable will work whether the result is real
            # or complex; it could be None, however.
            if ndiff and ndiff.is_comparable:
                return False

        # sometimes we can use a simplified result to give a clue as to
        # what the expression should be; if the expression is *not* zero
        # then we should have been able to compute that and so now
        # we can just consider the cases where the approximation appears
        # to be zero -- we try to prove it via minimal_polynomial.
        #
        # removed
        # ns = nsimplify(diff)
        # if diff.is_number and (not ns or ns == diff):
        #
        # The thought was that if it nsimplifies to 0 that's a sure sign
        # to try the following to prove it; or if it changed but wasn't
        # zero that might be a sign that it's not going to be easy to
        # prove. But tests seem to be working without that logic.
        #
        # ... other code
```
### 8 - sympy/logic/boolalg.py:

Start line: 417, End line: 453

```python
true = BooleanTrue()
false = BooleanFalse()
# We want S.true and S.false to work, rather than S.BooleanTrue and
# S.BooleanFalse, but making the class and instance names the same causes some
# major issues (like the inability to import the class directly from this
# file).
S.true = true
S.false = false

converter[bool] = lambda x: S.true if x else S.false


class BooleanFunction(Application, Boolean):
    """Boolean function is a function that lives in a boolean space
    It is used as base class for And, Or, Not, etc.
    """
    is_Boolean = True

    def _eval_simplify(self, **kwargs):
        rv = self.func(*[
            a._eval_simplify(**kwargs) for a in self.args])
        return simplify_logic(rv)

    def simplify(self, **kwargs):
        from sympy.simplify.simplify import simplify
        return simplify(self, **kwargs)

    def __lt__(self, other):
        from sympy.utilities.misc import filldedent
        raise TypeError(filldedent('''
            A Boolean argument can only be used in
            Eq and Ne; all other relationals expect
            real expressions.
        '''))
    __le__ = __lt__
    __ge__ = __lt__
    __gt__ = __lt__
```
### 9 - sympy/core/relational.py:

Start line: 1225, End line: 1250

```python
def is_ge(lhs, rhs):
    # ... other code

    if retval is not None:
        return retval
    else:
        n2 = _n2(lhs, rhs)
        if n2 is not None:
            # use float comparison for infinity.
            # otherwise get stuck in infinite recursion
            if n2 in (S.Infinity, S.NegativeInfinity):
                n2 = float(n2)
            return _sympify(n2 >= 0)
        if lhs.is_extended_real and rhs.is_extended_real:
            if (lhs.is_infinite and lhs.is_extended_positive) or (rhs.is_infinite and rhs.is_extended_negative):
                return True
            diff = lhs - rhs
            if diff is not S.NaN:
                rv = diff.is_extended_nonnegative
                if rv is not None:
                    return rv


def is_neq(lhs, rhs):
    """Fuzzy bool for lhs does not equal rhs.

    See the docstring for is_eq for more
    """
    return fuzzy_not(is_eq(lhs, rhs))
```
### 10 - sympy/logic/boolalg.py:

Start line: 104, End line: 132

```python
@sympify_method_args
class Boolean(Basic):

    def equals(self, other):
        """
        Returns True if the given formulas have the same truth table.
        For two formulas to be equal they must have the same literals.

        Examples
        ========

        >>> from sympy.abc import A, B, C
        >>> from sympy.logic.boolalg import And, Or, Not
        >>> (A >> B).equals(~B >> ~A)
        True
        >>> Not(And(A, B, C)).equals(And(Not(A), Not(B), Not(C)))
        False
        >>> Not(And(A, Not(A))).equals(Or(B, Not(B)))
        False

        """
        from sympy.logic.inference import satisfiable
        from sympy.core.relational import Relational

        if self.has(Relational) or other.has(Relational):
            raise NotImplementedError('handling of relationals')
        return self.atoms() == other.atoms() and \
            not satisfiable(Not(Equivalent(self, other)))

    def to_nnf(self, simplify=True):
        # override where necessary
        return self
```
