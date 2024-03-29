# sympy__sympy-18057

| **sympy/sympy** | `62000f37b8821573ba00280524ffb4ac4a380875` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 1950 |
| **Any found context length** | 1950 |
| **Avg pos** | 2.0 |
| **Min pos** | 2 |
| **Max pos** | 2 |
| **Top file pos** | 2 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sympy/core/expr.py b/sympy/core/expr.py
--- a/sympy/core/expr.py
+++ b/sympy/core/expr.py
@@ -121,7 +121,7 @@ def _hashable_content(self):
 
     def __eq__(self, other):
         try:
-            other = sympify(other)
+            other = _sympify(other)
             if not isinstance(other, Expr):
                 return False
         except (SympifyError, SyntaxError):

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sympy/core/expr.py | 124 | 124 | 2 | 2 | 1950


## Problem Statement

```
Sympy incorrectly attempts to eval reprs in its __eq__ method
Passing strings produced by unknown objects into eval is **very bad**. It is especially surprising for an equality check to trigger that kind of behavior. This should be fixed ASAP.

Repro code:

\`\`\`
import sympy
class C:
    def __repr__(self):
        return 'x.y'
_ = sympy.Symbol('x') == C()
\`\`\`

Results in:

\`\`\`
E   AttributeError: 'Symbol' object has no attribute 'y'
\`\`\`

On the line:

\`\`\`
    expr = eval(
        code, global_dict, local_dict)  # take local objects in preference
\`\`\`

Where code is:

\`\`\`
Symbol ('x' ).y
\`\`\`

Full trace:

\`\`\`
FAILED                   [100%]
        class C:
            def __repr__(self):
                return 'x.y'
    
>       _ = sympy.Symbol('x') == C()

_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
sympy/core/expr.py:124: in __eq__
    other = sympify(other)
sympy/core/sympify.py:385: in sympify
    expr = parse_expr(a, local_dict=locals, transformations=transformations, evaluate=evaluate)
sympy/parsing/sympy_parser.py:1011: in parse_expr
    return eval_expr(code, local_dict, global_dict)
sympy/parsing/sympy_parser.py:906: in eval_expr
    code, global_dict, local_dict)  # take local objects in preference
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

>   ???
E   AttributeError: 'Symbol' object has no attribute 'y'

<string>:1: AttributeError
\`\`\`

Related issue: an unknown object whose repr is `x` will incorrectly compare as equal to a sympy symbol x:

\`\`\`
    class C:
        def __repr__(self):
            return 'x'

    assert sympy.Symbol('x') != C()  # fails
\`\`\`

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 sympy/core/sympify.py | 80 | 264| 1781 | 1781 | 4273 | 
| **-> 2 <-** | **2 sympy/core/expr.py** | 122 | 142| 169 | 1950 | 37400 | 
| 3 | 3 sympy/core/relational.py | 1 | 30| 250 | 2200 | 46154 | 
| 4 | 3 sympy/core/sympify.py | 1 | 23| 173 | 2373 | 46154 | 
| 5 | **3 sympy/core/expr.py** | 332 | 383| 477 | 2850 | 46154 | 
| 6 | 4 sympy/core/symbol.py | 261 | 316| 409 | 3259 | 52763 | 
| 7 | 5 sympy/core/function.py | 2144 | 2206| 564 | 3823 | 80325 | 
| 8 | 6 sympy/core/basic.py | 534 | 578| 355 | 4178 | 96422 | 
| 9 | **6 sympy/core/expr.py** | 792 | 849| 547 | 4725 | 96422 | 
| 10 | **6 sympy/core/expr.py** | 144 | 206| 536 | 5261 | 96422 | 
| 11 | **6 sympy/core/expr.py** | 851 | 904| 431 | 5692 | 96422 | 
| 12 | 6 sympy/core/relational.py | 397 | 454| 472 | 6164 | 96422 | 
| 13 | 6 sympy/core/sympify.py | 265 | 373| 870 | 7034 | 96422 | 
| 14 | **6 sympy/core/expr.py** | 385 | 405| 161 | 7195 | 96422 | 
| 15 | 7 sympy/__init__.py | 1 | 93| 620 | 7815 | 97042 | 
| 16 | 8 sympy/core/power.py | 1366 | 1397| 237 | 8052 | 112504 | 
| 17 | 9 sympy/plotting/experimental_lambdify.py | 1 | 76| 868 | 8920 | 118407 | 
| 18 | **9 sympy/core/expr.py** | 2483 | 2554| 565 | 9485 | 118407 | 
| 19 | 10 sympy/core/backend.py | 1 | 24| 357 | 9842 | 118765 | 
| 20 | 10 sympy/core/symbol.py | 237 | 259| 288 | 10130 | 118765 | 
| 21 | **10 sympy/core/expr.py** | 993 | 1036| 323 | 10453 | 118765 | 
| 22 | 11 sympy/abc.py | 72 | 112| 432 | 10885 | 119918 | 
| 23 | 11 sympy/core/relational.py | 794 | 1025| 2080 | 12965 | 119918 | 
| 24 | 12 sympy/integrals/transforms.py | 942 | 967| 231 | 13196 | 136738 | 
| 25 | **12 sympy/core/expr.py** | 1 | 13| 124 | 13320 | 136738 | 
| 26 | 12 sympy/core/sympify.py | 374 | 389| 147 | 13467 | 136738 | 
| 27 | 12 sympy/core/function.py | 2269 | 2315| 313 | 13780 | 136738 | 
| 28 | 12 sympy/core/relational.py | 615 | 648| 248 | 14028 | 136738 | 
| 29 | **12 sympy/core/expr.py** | 2626 | 2686| 448 | 14476 | 136738 | 
| 30 | 13 sympy/codegen/rewriting.py | 129 | 165| 398 | 14874 | 138819 | 
| 31 | **13 sympy/core/expr.py** | 622 | 717| 806 | 15680 | 138819 | 
| 32 | 13 sympy/core/relational.py | 479 | 577| 950 | 16630 | 138819 | 
| 33 | 14 sympy/printing/ccode.py | 757 | 893| 1518 | 18148 | 147005 | 
| 34 | **14 sympy/core/expr.py** | 314 | 330| 157 | 18305 | 147005 | 
| 35 | 14 sympy/core/power.py | 777 | 831| 645 | 18950 | 147005 | 
| 36 | 14 sympy/core/power.py | 1297 | 1315| 133 | 19083 | 147005 | 
| 37 | 14 sympy/core/symbol.py | 213 | 235| 199 | 19282 | 147005 | 
| 38 | **14 sympy/core/expr.py** | 278 | 313| 500 | 19782 | 147005 | 
| 39 | **14 sympy/core/expr.py** | 407 | 418| 124 | 19906 | 147005 | 
| 40 | **14 sympy/core/expr.py** | 225 | 276| 455 | 20361 | 147005 | 
| 41 | 14 sympy/core/basic.py | 345 | 400| 346 | 20707 | 147005 | 
| 42 | 15 sympy/this.py | 1 | 22| 119 | 20826 | 147124 | 
| 43 | 15 sympy/core/relational.py | 288 | 368| 756 | 21582 | 147124 | 
| 44 | 16 sympy/printing/repr.py | 210 | 221| 141 | 21723 | 149817 | 
| 45 | 16 sympy/core/power.py | 1270 | 1295| 213 | 21936 | 149817 | 
| 46 | 16 sympy/core/basic.py | 1130 | 1193| 520 | 22456 | 149817 | 
| 47 | 16 sympy/core/symbol.py | 475 | 506| 291 | 22747 | 149817 | 
| 48 | 17 sympy/printing/pretty/pretty.py | 1 | 29| 280 | 23027 | 173256 | 
| 49 | 18 sympy/functions/special/hyper.py | 1085 | 1095| 144 | 23171 | 183567 | 
| 50 | 19 sympy/parsing/sympy_parser.py | 1011 | 1048| 234 | 23405 | 191830 | 
| 51 | 19 sympy/core/sympify.py | 392 | 418| 180 | 23585 | 191830 | 
| 52 | 19 sympy/codegen/rewriting.py | 233 | 256| 197 | 23782 | 191830 | 


### Hint

```
See also #12524
Safe flag or no, == should call _sympify since an expression shouldn't equal a string. 

I also think we should deprecate the string fallback in sympify. It has led to serious performance issues in the past and clearly has security issues as well. 
Actually, it looks like we also have

\`\`\`
>>> x == 'x'
True
\`\`\`

which is a major regression since 1.4. 

I bisected it to 73caef3991ca5c4c6a0a2c16cc8853cf212db531. 

The bug in the issue doesn't exist in 1.4 either. So we could consider doing a 1.5.1 release fixing this. 
The thing is, I could have swore this behavior was tested. But I don't see anything in the test changes from https://github.com/sympy/sympy/pull/16924 about string comparisons. 
```

## Patch

```diff
diff --git a/sympy/core/expr.py b/sympy/core/expr.py
--- a/sympy/core/expr.py
+++ b/sympy/core/expr.py
@@ -121,7 +121,7 @@ def _hashable_content(self):
 
     def __eq__(self, other):
         try:
-            other = sympify(other)
+            other = _sympify(other)
             if not isinstance(other, Expr):
                 return False
         except (SympifyError, SyntaxError):

```

## Test Patch

```diff
diff --git a/sympy/core/tests/test_expr.py b/sympy/core/tests/test_expr.py
--- a/sympy/core/tests/test_expr.py
+++ b/sympy/core/tests/test_expr.py
@@ -1903,3 +1903,24 @@ def test_ExprBuilder():
     eb = ExprBuilder(Mul)
     eb.args.extend([x, x])
     assert eb.build() == x**2
+
+def test_non_string_equality():
+    # Expressions should not compare equal to strings
+    x = symbols('x')
+    one = sympify(1)
+    assert (x == 'x') is False
+    assert (x != 'x') is True
+    assert (one == '1') is False
+    assert (one != '1') is True
+    assert (x + 1 == 'x + 1') is False
+    assert (x + 1 != 'x + 1') is True
+
+    # Make sure == doesn't try to convert the resulting expression to a string
+    # (e.g., by calling sympify() instead of _sympify())
+
+    class BadRepr(object):
+        def __repr__(self):
+            raise RuntimeError
+
+    assert (x == BadRepr()) is False
+    assert (x != BadRepr()) is True
diff --git a/sympy/core/tests/test_var.py b/sympy/core/tests/test_var.py
--- a/sympy/core/tests/test_var.py
+++ b/sympy/core/tests/test_var.py
@@ -19,7 +19,8 @@ def test_var():
     assert ns['fg'] == Symbol('fg')
 
 # check return value
-    assert v == ['d', 'e', 'fg']
+    assert v != ['d', 'e', 'fg']
+    assert v == [Symbol('d'), Symbol('e'), Symbol('fg')]
 
 
 def test_var_return():

```


## Code snippets

### 1 - sympy/core/sympify.py:

Start line: 80, End line: 264

```python
def sympify(a, locals=None, convert_xor=True, strict=False, rational=False,
        evaluate=None):
    """Converts an arbitrary expression to a type that can be used inside SymPy.

    For example, it will convert Python ints into instances of sympy.Integer,
    floats into instances of sympy.Float, etc. It is also able to coerce symbolic
    expressions which inherit from Basic. This can be useful in cooperation
    with SAGE.

    It currently accepts as arguments:
       - any object defined in SymPy
       - standard numeric python types: int, long, float, Decimal
       - strings (like "0.09" or "2e-19")
       - booleans, including ``None`` (will leave ``None`` unchanged)
       - dict, lists, sets or tuples containing any of the above

    .. warning::
        Note that this function uses ``eval``, and thus shouldn't be used on
        unsanitized input.

    If the argument is already a type that SymPy understands, it will do
    nothing but return that value. This can be used at the beginning of a
    function to ensure you are working with the correct type.

    >>> from sympy import sympify

    >>> sympify(2).is_integer
    True
    >>> sympify(2).is_real
    True

    >>> sympify(2.0).is_real
    True
    >>> sympify("2.0").is_real
    True
    >>> sympify("2e-45").is_real
    True

    If the expression could not be converted, a SympifyError is raised.

    >>> sympify("x***2")
    Traceback (most recent call last):
    ...
    SympifyError: SympifyError: "could not parse u'x***2'"

    Locals
    ------

    The sympification happens with access to everything that is loaded
    by ``from sympy import *``; anything used in a string that is not
    defined by that import will be converted to a symbol. In the following,
    the ``bitcount`` function is treated as a symbol and the ``O`` is
    interpreted as the Order object (used with series) and it raises
    an error when used improperly:

    >>> s = 'bitcount(42)'
    >>> sympify(s)
    bitcount(42)
    >>> sympify("O(x)")
    O(x)
    >>> sympify("O + 1")
    Traceback (most recent call last):
    ...
    TypeError: unbound method...

    In order to have ``bitcount`` be recognized it can be imported into a
    namespace dictionary and passed as locals:

    >>> from sympy.core.compatibility import exec_
    >>> ns = {}
    >>> exec_('from sympy.core.evalf import bitcount', ns)
    >>> sympify(s, locals=ns)
    6

    In order to have the ``O`` interpreted as a Symbol, identify it as such
    in the namespace dictionary. This can be done in a variety of ways; all
    three of the following are possibilities:

    >>> from sympy import Symbol
    >>> ns["O"] = Symbol("O")  # method 1
    >>> exec_('from sympy.abc import O', ns)  # method 2
    >>> ns.update(dict(O=Symbol("O")))  # method 3
    >>> sympify("O + 1", locals=ns)
    O + 1

    If you want *all* single-letter and Greek-letter variables to be symbols
    then you can use the clashing-symbols dictionaries that have been defined
    there as private variables: _clash1 (single-letter variables), _clash2
    (the multi-letter Greek names) or _clash (both single and multi-letter
    names that are defined in abc).

    >>> from sympy.abc import _clash1
    >>> _clash1
    {'C': C, 'E': E, 'I': I, 'N': N, 'O': O, 'Q': Q, 'S': S}
    >>> sympify('I & Q', _clash1)
    I & Q

    Strict
    ------

    If the option ``strict`` is set to ``True``, only the types for which an
    explicit conversion has been defined are converted. In the other
    cases, a SympifyError is raised.

    >>> print(sympify(None))
    None
    >>> sympify(None, strict=True)
    Traceback (most recent call last):
    ...
    SympifyError: SympifyError: None

    Evaluation
    ----------

    If the option ``evaluate`` is set to ``False``, then arithmetic and
    operators will be converted into their SymPy equivalents and the
    ``evaluate=False`` option will be added. Nested ``Add`` or ``Mul`` will
    be denested first. This is done via an AST transformation that replaces
    operators with their SymPy equivalents, so if an operand redefines any
    of those operations, the redefined operators will not be used.

    >>> sympify('2**2 / 3 + 5')
    19/3
    >>> sympify('2**2 / 3 + 5', evaluate=False)
    2**2/3 + 5

    Extending
    ---------

    To extend ``sympify`` to convert custom objects (not derived from ``Basic``),
    just define a ``_sympy_`` method to your class. You can do that even to
    classes that you do not own by subclassing or adding the method at runtime.

    >>> from sympy import Matrix
    >>> class MyList1(object):
    ...     def __iter__(self):
    ...         yield 1
    ...         yield 2
    ...         return
    ...     def __getitem__(self, i): return list(self)[i]
    ...     def _sympy_(self): return Matrix(self)
    >>> sympify(MyList1())
    Matrix([
    [1],
    [2]])

    If you do not have control over the class definition you could also use the
    ``converter`` global dictionary. The key is the class and the value is a
    function that takes a single argument and returns the desired SymPy
    object, e.g. ``converter[MyList] = lambda x: Matrix(x)``.

    >>> class MyList2(object):   # XXX Do not do this if you control the class!
    ...     def __iter__(self):  #     Use _sympy_!
    ...         yield 1
    ...         yield 2
    ...         return
    ...     def __getitem__(self, i): return list(self)[i]
    >>> from sympy.core.sympify import converter
    >>> converter[MyList2] = lambda x: Matrix(x)
    >>> sympify(MyList2())
    Matrix([
    [1],
    [2]])

    Notes
    =====

    The keywords ``rational`` and ``convert_xor`` are only used
    when the input is a string.

    Sometimes autosimplification during sympification results in expressions
    that are very different in structure than what was entered. Until such
    autosimplification is no longer done, the ``kernS`` function might be of
    some use. In the example below you can see how an expression reduces to
    -1 by autosimplification, but does not do so when ``kernS`` is used.

    >>> from sympy.core.sympify import kernS
    >>> from sympy.abc import x
    >>> -2*(-(-x + 1/x)/(x*(x - 1/x)**2) - 1/(x*(x - 1/x))) - 1
    -1
    >>> s = '-2*(-(-x + 1/x)/(x*(x - 1/x)**2) - 1/(x*(x - 1/x))) - 1'
    >>> sympify(s)
    -1
    >>> kernS(s)
    -2*(-(-x + 1/x)/(x*(x - 1/x)**2) - 1/(x*(x - 1/x))) - 1

    """
    # ... other code
```
### 2 - sympy/core/expr.py:

Start line: 122, End line: 142

```python
class Expr(Basic, EvalfMixin):

    def __eq__(self, other):
        try:
            other = sympify(other)
            if not isinstance(other, Expr):
                return False
        except (SympifyError, SyntaxError):
            return False
        # check for pure number expr
        if  not (self.is_Number and other.is_Number) and (
                type(self) != type(other)):
            return False
        a, b = self._hashable_content(), other._hashable_content()
        if a != b:
            return False
        # check number *in* an expression
        for a, b in zip(a, b):
            if not isinstance(a, Expr):
                continue
            if a.is_Number and type(a) != type(b):
                return False
        return True
```
### 3 - sympy/core/relational.py:

Start line: 1, End line: 30

```python
from __future__ import print_function, division

from sympy.utilities.exceptions import SymPyDeprecationWarning
from .add import _unevaluated_Add, Add
from .basic import S
from .compatibility import ordered
from .expr import Expr
from .evalf import EvalfMixin
from .sympify import _sympify
from .evaluate import global_evaluate

from sympy.logic.boolalg import Boolean, BooleanAtom

__all__ = (
    'Rel', 'Eq', 'Ne', 'Lt', 'Le', 'Gt', 'Ge',
    'Relational', 'Equality', 'Unequality', 'StrictLessThan', 'LessThan',
    'StrictGreaterThan', 'GreaterThan',
)



# Note, see issue 4986.  Ideally, we wouldn't want to subclass both Boolean
# and Expr.

def _canonical(cond):
    # return a condition in which all relationals are canonical
    reps = {r: r.canonical for r in cond.atoms(Relational)}
    return cond.xreplace(reps)
    # XXX: AttributeError was being caught here but it wasn't triggered by any of
    # the tests so I've removed it...
```
### 4 - sympy/core/sympify.py:

Start line: 1, End line: 23

```python
"""sympify -- convert objects SymPy internal format"""

from __future__ import print_function, division

from inspect import getmro

from .core import all_classes as sympy_classes
from .compatibility import iterable, string_types, range
from .evaluate import global_evaluate


class SympifyError(ValueError):
    def __init__(self, expr, base_exc=None):
        self.expr = expr
        self.base_exc = base_exc

    def __str__(self):
        if self.base_exc is None:
            return "SympifyError: %r" % (self.expr,)

        return ("Sympify of expression '%s' failed, because of exception being "
            "raised:\n%s: %s" % (self.expr, self.base_exc.__class__.__name__,
            str(self.base_exc)))
```
### 5 - sympy/core/expr.py:

Start line: 332, End line: 383

```python
class Expr(Basic, EvalfMixin):

    def _cmp(self, other, op, cls):
        assert op in ("<", ">", "<=", ">=")
        try:
            other = _sympify(other)
        except SympifyError:
            raise TypeError("Invalid comparison %s %s %s" % (self, op, other))
        for me in (self, other):
            if me.is_extended_real is False:
                raise TypeError("Invalid comparison of non-real %s" % me)
            if me is S.NaN:
                raise TypeError("Invalid NaN comparison")

        n2 = _n2(self, other)
        if n2 is not None:
            # use float comparison for infinity.
            # otherwise get stuck in infinite recursion
            if n2 in (S.Infinity, S.NegativeInfinity):
                n2 = float(n2)
            if op == "<":
                return _sympify(n2 < 0)
            elif op == ">":
                return _sympify(n2 > 0)
            elif op == "<=":
                return _sympify(n2 <= 0)
            else: # >=
                return _sympify(n2 >= 0)

        if self.is_extended_real and other.is_extended_real:
            if op in ("<=", ">") \
                and ((self.is_infinite and self.is_extended_negative) \
                     or (other.is_infinite and other.is_extended_positive)):
                return S.true if op == "<=" else S.false
            if op in ("<", ">=") \
                and ((self.is_infinite and self.is_extended_positive) \
                     or (other.is_infinite and other.is_extended_negative)):
                return S.true if op == ">=" else S.false
            diff = self - other
            if diff is not S.NaN:
                if op == "<":
                    test = diff.is_extended_negative
                elif op == ">":
                    test = diff.is_extended_positive
                elif op == "<=":
                    test = diff.is_extended_nonpositive
                else: # >=
                    test = diff.is_extended_nonnegative

                if test is not None:
                    return S.true if test == True else S.false

        # return unevaluated comparison object
        return cls(self, other, evaluate=False)
```
### 6 - sympy/core/symbol.py:

Start line: 261, End line: 316

```python
class Symbol(AtomicExpr, Boolean):

    __xnew__ = staticmethod(
        __new_stage2__)            # never cached (e.g. dummy)
    __xnew_cached_ = staticmethod(
        cacheit(__new_stage2__))   # symbols are always cached

    def __getnewargs__(self):
        return (self.name,)

    def __getstate__(self):
        return {'_assumptions': self._assumptions}

    def _hashable_content(self):
        # Note: user-specified assumptions not hashed, just derived ones
        return (self.name,) + tuple(sorted(self.assumptions0.items()))

    def _eval_subs(self, old, new):
        from sympy.core.power import Pow
        if old.is_Pow:
            return Pow(self, S.One, evaluate=False)._eval_subs(old, new)

    @property
    def assumptions0(self):
        return dict((key, value) for key, value
                in self._assumptions.items() if value is not None)

    @cacheit
    def sort_key(self, order=None):
        return self.class_key(), (1, (str(self),)), S.One.sort_key(), S.One

    def as_dummy(self):
        return Dummy(self.name)

    def as_real_imag(self, deep=True, **hints):
        from sympy import im, re
        if hints.get('ignore') == self:
            return None
        else:
            return (re(self), im(self))

    def _sage_(self):
        import sage.all as sage
        return sage.var(self.name)

    def is_constant(self, *wrt, **flags):
        if not wrt:
            return False
        return not self in wrt

    @property
    def free_symbols(self):
        return {self}

    binary_symbols = free_symbols  # in this case, not always

    def as_set(self):
        return S.UniversalSet
```
### 7 - sympy/core/function.py:

Start line: 2144, End line: 2206

```python
class Subs(Expr):
    def __new__(cls, expr, variables, point, **assumptions):
        from sympy import Symbol

        if not is_sequence(variables, Tuple):
            variables = [variables]
        variables = Tuple(*variables)

        if has_dups(variables):
            repeated = [str(v) for v, i in Counter(variables).items() if i > 1]
            __ = ', '.join(repeated)
            raise ValueError(filldedent('''
                The following expressions appear more than once: %s
                ''' % __))

        point = Tuple(*(point if is_sequence(point, Tuple) else [point]))

        if len(point) != len(variables):
            raise ValueError('Number of point values must be the same as '
                             'the number of variables.')

        if not point:
            return sympify(expr)

        # denest
        if isinstance(expr, Subs):
            variables = expr.variables + variables
            point = expr.point + point
            expr = expr.expr
        else:
            expr = sympify(expr)

        # use symbols with names equal to the point value (with prepended _)
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
            # if any underscore-prepended symbol is already a free symbol
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
        obj._expr = expr.xreplace(dict(reps))
        return obj
```
### 8 - sympy/core/basic.py:

Start line: 534, End line: 578

```python
class Basic(with_metaclass(ManagedProperties)):

    def as_dummy(self):
        """Return the expression with any objects having structurally
        bound symbols replaced with unique, canonical symbols within
        the object in which they appear and having only the default
        assumption for commutativity being True.

        Examples
        ========

        >>> from sympy import Integral, Symbol
        >>> from sympy.abc import x, y
        >>> r = Symbol('r', real=True)
        >>> Integral(r, (r, x)).as_dummy()
        Integral(_0, (_0, x))
        >>> _.variables[0].is_real is None
        True

        Notes
        =====

        Any object that has structural dummy variables should have
        a property, `bound_symbols` that returns a list of structural
        dummy symbols of the object itself.

        Lambda and Subs have bound symbols, but because of how they
        are cached, they already compare the same regardless of their
        bound symbols:

        >>> from sympy import Lambda
        >>> Lambda(x, x + 1) == Lambda(y, y + 1)
        True
        """
        def can(x):
            d = {i: i.as_dummy() for i in x.bound_symbols}
            # mask free that shadow bound
            x = x.subs(d)
            c = x.canonical_variables
            # replace bound
            x = x.xreplace(c)
            # undo masking
            x = x.xreplace(dict((v, k) for k, v in d.items()))
            return x
        return self.replace(
            lambda x: hasattr(x, 'bound_symbols'),
            lambda x: can(x))
```
### 9 - sympy/core/expr.py:

Start line: 792, End line: 849

```python
class Expr(Basic, EvalfMixin):

    def equals(self, other, failing_expression=False):
        # ... other code
        if diff.is_number:
            # try to prove via self-consistency
            surds = [s for s in diff.atoms(Pow) if s.args[0].is_Integer]
            # it seems to work better to try big ones first
            surds.sort(key=lambda x: -x.args[0])
            for s in surds:
                try:
                    # simplify is False here -- this expression has already
                    # been identified as being hard to identify as zero;
                    # we will handle the checking ourselves using nsimplify
                    # to see if we are in the right ballpark or not and if so
                    # *then* the simplification will be attempted.
                    sol = solve(diff, s, simplify=False)
                    if sol:
                        if s in sol:
                            # the self-consistent result is present
                            return True
                        if all(si.is_Integer for si in sol):
                            # perfect powers are removed at instantiation
                            # so surd s cannot be an integer
                            return False
                        if all(i.is_algebraic is False for i in sol):
                            # a surd is algebraic
                            return False
                        if any(si in surds for si in sol):
                            # it wasn't equal to s but it is in surds
                            # and different surds are not equal
                            return False
                        if any(nsimplify(s - si) == 0 and
                                simplify(s - si) == 0 for si in sol):
                            return True
                        if s.is_real:
                            if any(nsimplify(si, [s]) == s and simplify(si) == s
                                    for si in sol):
                                return True
                except NotImplementedError:
                    pass

            # try to prove with minimal_polynomial but know when
            # *not* to use this or else it can take a long time. e.g. issue 8354
            if True:  # change True to condition that assures non-hang
                try:
                    mp = minimal_polynomial(diff)
                    if mp.is_Symbol:
                        return True
                    return False
                except (NotAlgebraic, NotImplementedError):
                    pass

        # diff has not simplified to zero; constant is either None, True
        # or the number with significance (is_comparable) that was randomly
        # calculated twice as the same value.
        if constant not in (True, None) and constant != 0:
            return False

        if failing_expression:
            return diff
        return None
```
### 10 - sympy/core/expr.py:

Start line: 144, End line: 206

```python
class Expr(Basic, EvalfMixin):

    # ***************
    # * Arithmetics *
    # ***************
    # Expr and its sublcasses use _op_priority to determine which object
    # passed to a binary special method (__mul__, etc.) will handle the
    # operation. In general, the 'call_highest_priority' decorator will choose
    # the object with the highest _op_priority to handle the call.
    # Custom subclasses that want to define their own binary special methods
    # should set an _op_priority value that is higher than the default.
    #
    # **NOTE**:
    # This is a temporary fix, and will eventually be replaced with
    # something better and more powerful.  See issue 5510.
    _op_priority = 10.0

    def __pos__(self):
        return self

    def __neg__(self):
        # Mul has its own __neg__ routine, so we just
        # create a 2-args Mul with the -1 in the canonical
        # slot 0.
        c = self.is_commutative
        return Mul._from_args((S.NegativeOne, self), c)

    def __abs__(self):
        from sympy import Abs
        return Abs(self)

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__radd__')
    def __add__(self, other):
        return Add(self, other)

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__add__')
    def __radd__(self, other):
        return Add(other, self)

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__rsub__')
    def __sub__(self, other):
        return Add(self, -other)

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__sub__')
    def __rsub__(self, other):
        return Add(other, -self)

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__rmul__')
    def __mul__(self, other):
        return Mul(self, other)

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__mul__')
    def __rmul__(self, other):
        return Mul(other, self)

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__rpow__')
    def _pow(self, other):
        return Pow(self, other)
```
### 11 - sympy/core/expr.py:

Start line: 851, End line: 904

```python
class Expr(Basic, EvalfMixin):

    def _eval_is_positive(self):
        finite = self.is_finite
        if finite is False:
            return False
        extended_positive = self.is_extended_positive
        if finite is True:
            return extended_positive
        if extended_positive is False:
            return False

    def _eval_is_negative(self):
        finite = self.is_finite
        if finite is False:
            return False
        extended_negative = self.is_extended_negative
        if finite is True:
            return extended_negative
        if extended_negative is False:
            return False

    def _eval_is_extended_positive_negative(self, positive):
        from sympy.polys.numberfields import minimal_polynomial
        from sympy.polys.polyerrors import NotAlgebraic
        if self.is_number:
            if self.is_extended_real is False:
                return False

            # check to see that we can get a value
            try:
                n2 = self._eval_evalf(2)
            # XXX: This shouldn't be caught here
            # Catches ValueError: hypsum() failed to converge to the requested
            # 34 bits of accuracy
            except ValueError:
                return None
            if n2 is None:
                return None
            if getattr(n2, '_prec', 1) == 1:  # no significance
                return None
            if n2 is S.NaN:
                return None

            r, i = self.evalf(2).as_real_imag()
            if not i.is_Number or not r.is_Number:
                return False
            if r._prec != 1 and i._prec != 1:
                return bool(not i and ((r > 0) if positive else (r < 0)))
            elif r._prec == 1 and (not i or i._prec == 1) and \
                    self.is_algebraic and not self.has(Function):
                try:
                    if minimal_polynomial(self).is_Symbol:
                        return False
                except (NotAlgebraic, NotImplementedError):
                    pass
```
### 14 - sympy/core/expr.py:

Start line: 385, End line: 405

```python
class Expr(Basic, EvalfMixin):

    def __ge__(self, other):
        from sympy import GreaterThan
        return self._cmp(other, ">=", GreaterThan)

    def __le__(self, other):
        from sympy import LessThan
        return self._cmp(other, "<=", LessThan)

    def __gt__(self, other):
        from sympy import StrictGreaterThan
        return self._cmp(other, ">", StrictGreaterThan)

    def __lt__(self, other):
        from sympy import StrictLessThan
        return self._cmp(other, "<", StrictLessThan)

    def __trunc__(self):
        if not self.is_number:
            raise TypeError("can't truncate symbols and expressions")
        else:
            return Integer(self)
```
### 18 - sympy/core/expr.py:

Start line: 2483, End line: 2554

```python
class Expr(Basic, EvalfMixin):

    def _eval_is_polynomial(self, syms):
        if self.free_symbols.intersection(syms) == set([]):
            return True
        return False

    def is_polynomial(self, *syms):
        r"""
        Return True if self is a polynomial in syms and False otherwise.

        This checks if self is an exact polynomial in syms.  This function
        returns False for expressions that are "polynomials" with symbolic
        exponents.  Thus, you should be able to apply polynomial algorithms to
        expressions for which this returns True, and Poly(expr, \*syms) should
        work if and only if expr.is_polynomial(\*syms) returns True. The
        polynomial does not have to be in expanded form.  If no symbols are
        given, all free symbols in the expression will be used.

        This is not part of the assumptions system.  You cannot do
        Symbol('z', polynomial=True).

        Examples
        ========

        >>> from sympy import Symbol
        >>> x = Symbol('x')
        >>> ((x**2 + 1)**4).is_polynomial(x)
        True
        >>> ((x**2 + 1)**4).is_polynomial()
        True
        >>> (2**x + 1).is_polynomial(x)
        False


        >>> n = Symbol('n', nonnegative=True, integer=True)
        >>> (x**n + 1).is_polynomial(x)
        False

        This function does not attempt any nontrivial simplifications that may
        result in an expression that does not appear to be a polynomial to
        become one.

        >>> from sympy import sqrt, factor, cancel
        >>> y = Symbol('y', positive=True)
        >>> a = sqrt(y**2 + 2*y + 1)
        >>> a.is_polynomial(y)
        False
        >>> factor(a)
        y + 1
        >>> factor(a).is_polynomial(y)
        True

        >>> b = (y**2 + 2*y + 1)/(y + 1)
        >>> b.is_polynomial(y)
        False
        >>> cancel(b)
        y + 1
        >>> cancel(b).is_polynomial(y)
        True

        See also .is_rational_function()

        """
        if syms:
            syms = set(map(sympify, syms))
        else:
            syms = self.free_symbols

        if syms.intersection(self.free_symbols) == set([]):
            # constant polynomial
            return True
        else:
            return self._eval_is_polynomial(syms)
```
### 21 - sympy/core/expr.py:

Start line: 993, End line: 1036

```python
class Expr(Basic, EvalfMixin):

    def _eval_power(self, other):
        # subclass to compute self**other for cases when
        # other is not NaN, 0, or 1
        return None

    def _eval_conjugate(self):
        if self.is_extended_real:
            return self
        elif self.is_imaginary:
            return -self

    def conjugate(self):
        from sympy.functions.elementary.complexes import conjugate as c
        return c(self)

    def _eval_transpose(self):
        from sympy.functions.elementary.complexes import conjugate
        if (self.is_complex or self.is_infinite):
            return self
        elif self.is_hermitian:
            return conjugate(self)
        elif self.is_antihermitian:
            return -conjugate(self)

    def transpose(self):
        from sympy.functions.elementary.complexes import transpose
        return transpose(self)

    def _eval_adjoint(self):
        from sympy.functions.elementary.complexes import conjugate, transpose
        if self.is_hermitian:
            return self
        elif self.is_antihermitian:
            return -self
        obj = self._eval_conjugate()
        if obj is not None:
            return transpose(obj)
        obj = self._eval_transpose()
        if obj is not None:
            return conjugate(obj)

    def adjoint(self):
        from sympy.functions.elementary.complexes import adjoint
        return adjoint(self)
```
### 25 - sympy/core/expr.py:

Start line: 1, End line: 13

```python
from __future__ import print_function, division

from .sympify import sympify, _sympify, SympifyError
from .basic import Basic, Atom
from .singleton import S
from .evalf import EvalfMixin, pure_complex
from .decorators import _sympifyit, call_highest_priority
from .cache import cacheit
from .compatibility import reduce, as_int, default_sort_key, range, Iterable
from sympy.utilities.misc import func_name
from mpmath.libmp import mpf_log, prec_to_dps

from collections import defaultdict
```
### 29 - sympy/core/expr.py:

Start line: 2626, End line: 2686

```python
class Expr(Basic, EvalfMixin):

    def _eval_is_algebraic_expr(self, syms):
        if self.free_symbols.intersection(syms) == set([]):
            return True
        return False

    def is_algebraic_expr(self, *syms):
        """
        This tests whether a given expression is algebraic or not, in the
        given symbols, syms. When syms is not given, all free symbols
        will be used. The rational function does not have to be in expanded
        or in any kind of canonical form.

        This function returns False for expressions that are "algebraic
        expressions" with symbolic exponents. This is a simple extension to the
        is_rational_function, including rational exponentiation.

        Examples
        ========

        >>> from sympy import Symbol, sqrt
        >>> x = Symbol('x', real=True)
        >>> sqrt(1 + x).is_rational_function()
        False
        >>> sqrt(1 + x).is_algebraic_expr()
        True

        This function does not attempt any nontrivial simplifications that may
        result in an expression that does not appear to be an algebraic
        expression to become one.

        >>> from sympy import exp, factor
        >>> a = sqrt(exp(x)**2 + 2*exp(x) + 1)/(exp(x) + 1)
        >>> a.is_algebraic_expr(x)
        False
        >>> factor(a).is_algebraic_expr()
        True

        See Also
        ========
        is_rational_function()

        References
        ==========

        - https://en.wikipedia.org/wiki/Algebraic_expression

        """
        if syms:
            syms = set(map(sympify, syms))
        else:
            syms = self.free_symbols

        if syms.intersection(self.free_symbols) == set([]):
            # constant algebraic expression
            return True
        else:
            return self._eval_is_algebraic_expr(syms)

    ###################################################################################
    ##################### SERIES, LEADING TERM, LIMIT, ORDER METHODS ##################
    ###################################################################################
```
### 31 - sympy/core/expr.py:

Start line: 622, End line: 717

```python
class Expr(Basic, EvalfMixin):

    def is_constant(self, *wrt, **flags):

        def check_denominator_zeros(expression):
            from sympy.solvers.solvers import denoms

            retNone = False
            for den in denoms(expression):
                z = den.is_zero
                if z is True:
                    return True
                if z is None:
                    retNone = True
            if retNone:
                return None
            return False

        simplify = flags.get('simplify', True)

        if self.is_number:
            return True
        free = self.free_symbols
        if not free:
            return True  # assume f(1) is some constant

        # if we are only interested in some symbols and they are not in the
        # free symbols then this expression is constant wrt those symbols
        wrt = set(wrt)
        if wrt and not wrt & free:
            return True
        wrt = wrt or free

        # simplify unless this has already been done
        expr = self
        if simplify:
            expr = expr.simplify()

        # is_zero should be a quick assumptions check; it can be wrong for
        # numbers (see test_is_not_constant test), giving False when it
        # shouldn't, but hopefully it will never give True unless it is sure.
        if expr.is_zero:
            return True

        # try numerical evaluation to see if we get two different values
        failing_number = None
        if wrt == free:
            # try 0 (for a) and 1 (for b)
            try:
                a = expr.subs(list(zip(free, [0]*len(free))),
                    simultaneous=True)
                if a is S.NaN:
                    # evaluation may succeed when substitution fails
                    a = expr._random(None, 0, 0, 0, 0)
            except ZeroDivisionError:
                a = None
            if a is not None and a is not S.NaN:
                try:
                    b = expr.subs(list(zip(free, [1]*len(free))),
                        simultaneous=True)
                    if b is S.NaN:
                        # evaluation may succeed when substitution fails
                        b = expr._random(None, 1, 0, 1, 0)
                except ZeroDivisionError:
                    b = None
                if b is not None and b is not S.NaN and b.equals(a) is False:
                    return False
                # try random real
                b = expr._random(None, -1, 0, 1, 0)
                if b is not None and b is not S.NaN and b.equals(a) is False:
                    return False
                # try random complex
                b = expr._random()
                if b is not None and b is not S.NaN:
                    if b.equals(a) is False:
                        return False
                    failing_number = a if a.is_number else b

        # now we will test each wrt symbol (or all free symbols) to see if the
        # expression depends on them or not using differentiation. This is
        # not sufficient for all expressions, however, so we don't return
        # False if we get a derivative other than 0 with free symbols.
        for w in wrt:
            deriv = expr.diff(w)
            if simplify:
                deriv = deriv.simplify()
            if deriv != 0:
                if not (pure_complex(deriv, or_real=True)):
                    if flags.get('failing_number', False):
                        return failing_number
                    elif deriv.free_symbols:
                        # dead line provided _random returns None in such cases
                        return None
                return False
        cd = check_denominator_zeros(self)
        if cd is True:
            return False
        elif cd is None:
            return None
        return True
```
### 34 - sympy/core/expr.py:

Start line: 314, End line: 330

```python
class Expr(Basic, EvalfMixin):
    __long__ = __int__

    def __float__(self):
        # Don't bother testing if it's a number; if it's not this is going
        # to fail, and if it is we still need to check that it evalf'ed to
        # a number.
        result = self.evalf()
        if result.is_Number:
            return float(result)
        if result.is_number and result.as_real_imag()[1]:
            raise TypeError("can't convert complex to float")
        raise TypeError("can't convert expression to float")

    def __complex__(self):
        result = self.evalf()
        re, im = result.as_real_imag()
        return complex(float(re), float(im))
```
### 38 - sympy/core/expr.py:

Start line: 278, End line: 313

```python
class Expr(Basic, EvalfMixin):

    def __int__(self):
        # Although we only need to round to the units position, we'll
        # get one more digit so the extra testing below can be avoided
        # unless the rounded value rounded to an integer, e.g. if an
        # expression were equal to 1.9 and we rounded to the unit position
        # we would get a 2 and would not know if this rounded up or not
        # without doing a test (as done below). But if we keep an extra
        # digit we know that 1.9 is not the same as 1 and there is no
        # need for further testing: our int value is correct. If the value
        # were 1.99, however, this would round to 2.0 and our int value is
        # off by one. So...if our round value is the same as the int value
        # (regardless of how much extra work we do to calculate extra decimal
        # places) we need to test whether we are off by one.
        from sympy import Dummy
        if not self.is_number:
            raise TypeError("can't convert symbols to int")
        r = self.round(2)
        if not r.is_Number:
            raise TypeError("can't convert complex to int")
        if r in (S.NaN, S.Infinity, S.NegativeInfinity):
            raise TypeError("can't convert %s to int" % r)
        i = int(r)
        if not i:
            return 0
        # off-by-one check
        if i == r and not (self - i).equals(0):
            isign = 1 if i > 0 else -1
            x = Dummy()
            # in the following (self - i).evalf(2) will not always work while
            # (self - r).evalf(2) and the use of subs does; if the test that
            # was added when this comment was added passes, it might be safe
            # to simply use sign to compute this rather than doing this by hand:
            diff_sign = 1 if (self - x).evalf(2, subs={x: i}) > 0 else -1
            if diff_sign != isign:
                i -= isign
        return i
```
### 39 - sympy/core/expr.py:

Start line: 407, End line: 418

```python
class Expr(Basic, EvalfMixin):

    @staticmethod
    def _from_mpmath(x, prec):
        from sympy import Float
        if hasattr(x, "_mpf_"):
            return Float._new(x._mpf_, prec)
        elif hasattr(x, "_mpc_"):
            re, im = x._mpc_
            re = Float._new(re, prec)
            im = Float._new(im, prec)*S.ImaginaryUnit
            return re + im
        else:
            raise TypeError("expected mpmath number (mpf or mpc)")
```
### 40 - sympy/core/expr.py:

Start line: 225, End line: 276

```python
class Expr(Basic, EvalfMixin):

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__pow__')
    def __rpow__(self, other):
        return Pow(other, self)

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__rdiv__')
    def __div__(self, other):
        return Mul(self, Pow(other, S.NegativeOne))

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__div__')
    def __rdiv__(self, other):
        return Mul(other, Pow(self, S.NegativeOne))

    __truediv__ = __div__
    __rtruediv__ = __rdiv__

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__rmod__')
    def __mod__(self, other):
        return Mod(self, other)

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__mod__')
    def __rmod__(self, other):
        return Mod(other, self)

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__rfloordiv__')
    def __floordiv__(self, other):
        from sympy.functions.elementary.integers import floor
        return floor(self / other)

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__floordiv__')
    def __rfloordiv__(self, other):
        from sympy.functions.elementary.integers import floor
        return floor(other / self)


    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__rdivmod__')
    def __divmod__(self, other):
        from sympy.functions.elementary.integers import floor
        return floor(self / other), Mod(self, other)

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__divmod__')
    def __rdivmod__(self, other):
        from sympy.functions.elementary.integers import floor
        return floor(other / self), Mod(other, self)
```
