# sympy__sympy-20590

| **sympy/sympy** | `cffd4e0f86fefd4802349a9f9b19ed70934ea354` |
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
diff --git a/sympy/core/_print_helpers.py b/sympy/core/_print_helpers.py
--- a/sympy/core/_print_helpers.py
+++ b/sympy/core/_print_helpers.py
@@ -17,6 +17,11 @@ class Printable:
     This also adds support for LaTeX printing in jupyter notebooks.
     """
 
+    # Since this class is used as a mixin we set empty slots. That means that
+    # instances of any subclasses that use slots will not need to have a
+    # __dict__.
+    __slots__ = ()
+
     # Note, we always use the default ordering (lex) in __str__ and __repr__,
     # regardless of the global setting. See issue 5487.
     def __str__(self):

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sympy/core/_print_helpers.py | 20 | 20 | - | - | -


## Problem Statement

```
Symbol instances have __dict__ since 1.7?
In version 1.6.2 Symbol instances had no `__dict__` attribute
\`\`\`python
>>> sympy.Symbol('s').__dict__
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
<ipython-input-3-e2060d5eec73> in <module>
----> 1 sympy.Symbol('s').__dict__

AttributeError: 'Symbol' object has no attribute '__dict__'
>>> sympy.Symbol('s').__slots__
('name',)
\`\`\`

This changes in 1.7 where `sympy.Symbol('s').__dict__` now exists (and returns an empty dict)
I may misinterpret this, but given the purpose of `__slots__`, I assume this is a bug, introduced because some parent class accidentally stopped defining `__slots__`.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 sympy/core/symbol.py | 389 | 416| 239 | 239 | 6838 | 
| 2 | 1 sympy/core/symbol.py | 291 | 348| 446 | 685 | 6838 | 
| 3 | 1 sympy/core/symbol.py | 242 | 265| 210 | 895 | 6838 | 
| 4 | 1 sympy/core/symbol.py | 267 | 289| 287 | 1182 | 6838 | 
| 5 | 1 sympy/core/symbol.py | 351 | 387| 289 | 1471 | 6838 | 
| 6 | 1 sympy/core/symbol.py | 18 | 55| 270 | 1741 | 6838 | 
| 7 | 2 sympy/stats/rv.py | 303 | 329| 203 | 1944 | 18768 | 
| 8 | 2 sympy/core/symbol.py | 545 | 660| 1075 | 3019 | 18768 | 
| 9 | 2 sympy/core/symbol.py | 661 | 754| 721 | 3740 | 18768 | 
| 10 | 2 sympy/core/symbol.py | 512 | 543| 287 | 4027 | 18768 | 
| 11 | 3 sympy/core/basic.py | 513 | 557| 355 | 4382 | 34603 | 
| 12 | 3 sympy/core/symbol.py | 180 | 220| 204 | 4586 | 34603 | 
| 13 | 4 sympy/core/containers.py | 187 | 224| 257 | 4843 | 36984 | 
| 14 | 5 sympy/deprecated/class_registry.py | 29 | 52| 228 | 5071 | 37429 | 
| 15 | 5 sympy/core/symbol.py | 1 | 16| 125 | 5196 | 37429 | 
| 16 | 6 sympy/abc.py | 74 | 114| 441 | 5637 | 38598 | 
| 17 | 7 sympy/core/singleton.py | 88 | 108| 192 | 5829 | 40366 | 
| 18 | 7 sympy/deprecated/class_registry.py | 1 | 27| 222 | 6051 | 40366 | 
| 19 | 8 sympy/core/core.py | 45 | 64| 126 | 6177 | 41153 | 
| 20 | 8 sympy/core/containers.py | 226 | 237| 168 | 6345 | 41153 | 
| 21 | 9 sympy/categories/baseclasses.py | 1 | 30| 250 | 6595 | 48467 | 
| 22 | 9 sympy/core/singleton.py | 110 | 134| 202 | 6797 | 48467 | 
| 23 | 10 sympy/printing/dot.py | 1 | 16| 110 | 6907 | 50733 | 
| 24 | 10 sympy/core/singleton.py | 1 | 87| 932 | 7839 | 50733 | 
| 25 | 11 sympy/polys/polyutils.py | 463 | 489| 204 | 8043 | 54181 | 
| 26 | 12 sympy/codegen/ast.py | 302 | 324| 167 | 8210 | 67594 | 
| 27 | 12 sympy/core/basic.py | 873 | 951| 695 | 8905 | 67594 | 
| 28 | 12 sympy/polys/polyutils.py | 423 | 461| 267 | 9172 | 67594 | 
| 29 | 13 sympy/__init__.py | 1 | 70| 687 | 9859 | 75735 | 
| 30 | 14 sympy/printing/pretty/pretty_symbology.py | 299 | 328| 230 | 10089 | 81687 | 
| 31 | 14 sympy/core/symbol.py | 57 | 121| 416 | 10505 | 81687 | 
| 32 | 15 sympy/integrals/rubi/symbol.py | 1 | 40| 391 | 10896 | 82078 | 
| 33 | 16 sympy/physics/vector/frame.py | 48 | 58| 136 | 11032 | 92340 | 
| 34 | 17 sympy/core/function.py | 2187 | 2249| 564 | 11596 | 120402 | 
| 35 | 18 sympy/diffgeom/diffgeom.py | 563 | 618| 353 | 11949 | 137004 | 
| 36 | 19 sympy/interactive/session.py | 205 | 230| 251 | 12200 | 140634 | 
| 37 | 19 sympy/__init__.py | 261 | 537| 224 | 12424 | 140634 | 
| 38 | 19 sympy/core/function.py | 943 | 961| 141 | 12565 | 140634 | 
| 39 | 19 sympy/core/basic.py | 28 | 141| 773 | 13338 | 140634 | 
| 40 | 20 sympy/utilities/exceptions.py | 133 | 187| 458 | 13796 | 142377 | 
| 41 | 21 sympy/core/backend.py | 1 | 34| 560 | 14356 | 142937 | 
| 42 | 21 sympy/codegen/ast.py | 850 | 893| 215 | 14571 | 142937 | 
| 43 | 21 sympy/printing/pretty/pretty_symbology.py | 107 | 183| 759 | 15330 | 142937 | 
| 44 | 21 sympy/core/symbol.py | 419 | 510| 720 | 16050 | 142937 | 
| 45 | 21 sympy/stats/rv.py | 264 | 274| 127 | 16177 | 142937 | 
| 46 | 21 sympy/abc.py | 1 | 73| 727 | 16904 | 142937 | 
| 47 | 21 sympy/stats/rv.py | 276 | 301| 158 | 17062 | 142937 | 
| 48 | 21 sympy/core/symbol.py | 820 | 884| 542 | 17604 | 142937 | 
| 49 | 21 sympy/codegen/ast.py | 1356 | 1412| 370 | 17974 | 142937 | 
| 50 | 22 sympy/core/operations.py | 1 | 16| 134 | 18108 | 148561 | 
| 51 | 23 sympy/core/relational.py | 1 | 41| 328 | 18436 | 159652 | 
| 52 | 23 sympy/printing/pretty/pretty_symbology.py | 216 | 256| 711 | 19147 | 159652 | 
| 53 | 23 sympy/printing/pretty/pretty_symbology.py | 257 | 297| 603 | 19750 | 159652 | 
| 54 | 23 sympy/interactive/session.py | 162 | 203| 332 | 20082 | 159652 | 
| 55 | 24 sympy/combinatorics/free_groups.py | 139 | 162| 207 | 20289 | 169738 | 
| 56 | 24 sympy/printing/pretty/pretty_symbology.py | 1 | 46| 270 | 20559 | 169738 | 
| 57 | 24 sympy/diffgeom/diffgeom.py | 2037 | 2076| 235 | 20794 | 169738 | 
| 58 | 25 sympy/functions/elementary/miscellaneous.py | 31 | 57| 121 | 20915 | 176497 | 
| 59 | 26 sympy/core/__init__.py | 38 | 92| 397 | 21312 | 177287 | 
| 60 | 26 sympy/core/singleton.py | 137 | 181| 317 | 21629 | 177287 | 
| 61 | 27 sympy/parsing/sympy_parser.py | 532 | 579| 386 | 22015 | 185550 | 
| 62 | 28 sympy/physics/mechanics/system.py | 9 | 207| 1885 | 23900 | 189530 | 
| 63 | 29 sympy/series/sequences.py | 86 | 101| 124 | 24024 | 198237 | 
| 64 | 29 sympy/categories/baseclasses.py | 171 | 221| 301 | 24325 | 198237 | 
| 65 | 29 sympy/core/function.py | 906 | 941| 372 | 24697 | 198237 | 
| 66 | 29 sympy/printing/pretty/pretty_symbology.py | 184 | 215| 314 | 25011 | 198237 | 
| 67 | 29 sympy/stats/rv.py | 238 | 262| 236 | 25247 | 198237 | 
| 68 | 29 sympy/physics/vector/frame.py | 1 | 46| 307 | 25554 | 198237 | 
| 69 | 29 sympy/utilities/exceptions.py | 1 | 131| 1294 | 26848 | 198237 | 


## Missing Patch Files

 * 1: sympy/core/_print_helpers.py

### Hint

```
I've bisected the change to 5644df199fdac0b7a44e85c97faff58dfd462a5a from #19425
It seems that Basic now inherits `DefaultPrinting` which I guess doesn't have slots. I'm not sure if it's a good idea to add `__slots__` to that class as it would then affect all subclasses.

@eric-wieser 
I'm not sure if this should count as a regression but it's certainly not an intended change.
Maybe we should just get rid of `__slots__`. The benchmark results from #19425 don't show any regression from not using `__slots__`.
Adding `__slots__` won't affect subclasses - if a subclass does not specify `__slots__`, then the default is to add a `__dict__` anyway.

I think adding it should be fine.
Using slots can break multiple inheritance but only if the slots are non-empty I guess. Maybe this means that any mixin should always declare empty slots or it won't work properly with subclasses that have slots...

I see that `EvalfMixin` has `__slots__ = ()`.
I guess we should add empty slots to DefaultPrinting then. Probably the intention of using slots with Basic classes is to enforce immutability so this could be considered a regression in that sense so it should go into 1.7.1 I think.
```

## Patch

```diff
diff --git a/sympy/core/_print_helpers.py b/sympy/core/_print_helpers.py
--- a/sympy/core/_print_helpers.py
+++ b/sympy/core/_print_helpers.py
@@ -17,6 +17,11 @@ class Printable:
     This also adds support for LaTeX printing in jupyter notebooks.
     """
 
+    # Since this class is used as a mixin we set empty slots. That means that
+    # instances of any subclasses that use slots will not need to have a
+    # __dict__.
+    __slots__ = ()
+
     # Note, we always use the default ordering (lex) in __str__ and __repr__,
     # regardless of the global setting. See issue 5487.
     def __str__(self):

```

## Test Patch

```diff
diff --git a/sympy/core/tests/test_basic.py b/sympy/core/tests/test_basic.py
--- a/sympy/core/tests/test_basic.py
+++ b/sympy/core/tests/test_basic.py
@@ -34,6 +34,12 @@ def test_structure():
     assert bool(b1)
 
 
+def test_immutable():
+    assert not hasattr(b1, '__dict__')
+    with raises(AttributeError):
+        b1.x = 1
+
+
 def test_equality():
     instances = [b1, b2, b3, b21, Basic(b1, b1, b1), Basic]
     for i, b_i in enumerate(instances):

```


## Code snippets

### 1 - sympy/core/symbol.py:

Start line: 389, End line: 416

```python
class Dummy(Symbol):

    def __new__(cls, name=None, dummy_index=None, **assumptions):
        if dummy_index is not None:
            assert name is not None, "If you specify a dummy_index, you must also provide a name"

        if name is None:
            name = "Dummy_" + str(Dummy._count)

        if dummy_index is None:
            dummy_index = Dummy._base_dummy_index + Dummy._count
            Dummy._count += 1

        cls._sanitize(assumptions, cls)
        obj = Symbol.__xnew__(cls, name, **assumptions)

        obj.dummy_index = dummy_index

        return obj

    def __getstate__(self):
        return {'_assumptions': self._assumptions, 'dummy_index': self.dummy_index}

    @cacheit
    def sort_key(self, order=None):
        return self.class_key(), (
            2, (self.name, self.dummy_index)), S.One.sort_key(), S.One

    def _hashable_content(self):
        return Symbol._hashable_content(self) + (self.dummy_index,)
```
### 2 - sympy/core/symbol.py:

Start line: 291, End line: 348

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
        return {key: value for key, value
                in self._assumptions.items() if value is not None}

    @cacheit
    def sort_key(self, order=None):
        return self.class_key(), (1, (self.name,)), S.One.sort_key(), S.One

    def as_dummy(self):
        # only put commutativity in explicitly if it is False
        return Dummy(self.name) if self.is_commutative is not False \
            else Dummy(self.name, commutative=self.is_commutative)

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
### 3 - sympy/core/symbol.py:

Start line: 242, End line: 265

```python
class Symbol(AtomicExpr, Boolean):

    def _merge(self, assumptions):
        base = self.assumptions0
        for k in set(assumptions) & set(base):
            if assumptions[k] != base[k]:
                from sympy.utilities.misc import filldedent
                raise ValueError(filldedent('''
                    non-matching assumptions for %s: existing value
                    is %s and new value is %s''' % (
                    k, base[k], assumptions[k])))
        base.update(assumptions)
        return base

    def __new__(cls, name, **assumptions):
        """Symbols are identified by name and assumptions::

        >>> from sympy import Symbol
        >>> Symbol("x") == Symbol("x")
        True
        >>> Symbol("x", real=True) == Symbol("x", real=False)
        False

        """
        cls._sanitize(assumptions, cls)
        return Symbol.__xnew_cached_(cls, name, **assumptions)
```
### 4 - sympy/core/symbol.py:

Start line: 267, End line: 289

```python
class Symbol(AtomicExpr, Boolean):

    def __new_stage2__(cls, name, **assumptions):
        if not isinstance(name, str):
            raise TypeError("name should be a string, not %s" % repr(type(name)))

        obj = Expr.__new__(cls)
        obj.name = name

        # TODO: Issue #8873: Forcing the commutative assumption here means
        # later code such as ``srepr()`` cannot tell whether the user
        # specified ``commutative=True`` or omitted it.  To workaround this,
        # we keep a copy of the assumptions dict, then create the StdFactKB,
        # and finally overwrite its ``._generator`` with the dict copy.  This
        # is a bit of a hack because we assume StdFactKB merely copies the
        # given dict as ``._generator``, but future modification might, e.g.,
        # compute a minimal equivalent assumption set.
        tmp_asm_copy = assumptions.copy()

        # be strict about commutativity
        is_commutative = fuzzy_bool(assumptions.get('commutative', True))
        assumptions['commutative'] = is_commutative
        obj._assumptions = StdFactKB(assumptions)
        obj._assumptions._generator = tmp_asm_copy  # Issue #8873
        return obj
```
### 5 - sympy/core/symbol.py:

Start line: 351, End line: 387

```python
class Dummy(Symbol):
    """Dummy symbols are each unique, even if they have the same name:

    Examples
    ========

    >>> from sympy import Dummy
    >>> Dummy("x") == Dummy("x")
    False

    If a name is not supplied then a string value of an internal count will be
    used. This is useful when a temporary variable is needed and the name
    of the variable used in the expression is not important.

    >>> Dummy() #doctest: +SKIP
    _Dummy_10

    """

    # In the rare event that a Dummy object needs to be recreated, both the
    # `name` and `dummy_index` should be passed.  This is used by `srepr` for
    # example:
    # >>> d1 = Dummy()
    # >>> d2 = eval(srepr(d1))
    # >>> d2 == d1
    # True
    #
    # If a new session is started between `srepr` and `eval`, there is a very
    # small chance that `d2` will be equal to a previously-created Dummy.

    _count = 0
    _prng = random.Random()
    _base_dummy_index = _prng.randint(10**6, 9*10**6)

    __slots__ = ('dummy_index',)

    is_Dummy = True
```
### 6 - sympy/core/symbol.py:

Start line: 18, End line: 55

```python
class Str(Atom):
    """
    Represents string in SymPy.

    Explanation
    ===========

    Previously, ``Symbol`` was used where string is needed in ``args`` of SymPy
    objects, e.g. denoting the name of the instance. However, since ``Symbol``
    represents mathematical scalar, this class should be used instead.

    """
    __slots__ = ('name',)

    def __new__(cls, name, **kwargs):
        if not isinstance(name, str):
            raise TypeError("name should be a string, not %s" % repr(type(name)))
        obj = Expr.__new__(cls, **kwargs)
        obj.name = name
        return obj

    def __getnewargs__(self):
        return (self.name,)

    def _hashable_content(self):
        return (self.name,)


def _filter_assumptions(kwargs):
    """Split the given dict into assumptions and non-assumptions.
    Keys are taken as assumptions if they correspond to an
    entry in ``_assume_defined``.
    """
    assumptions, nonassumptions = map(dict, sift(kwargs.items(),
        lambda i: i[0] in _assume_defined,
        binary=True))
    Symbol._sanitize(assumptions)
    return assumptions, nonassumptions
```
### 7 - sympy/stats/rv.py:

Start line: 303, End line: 329

```python
class RandomIndexedSymbol(RandomSymbol):

    def __new__(cls, idx_obj, pspace=None):
        if pspace is None:
            # Allow single arg, representing pspace == PSpace()
            pspace = PSpace()
        if not isinstance(idx_obj, (Indexed, Function)):
            raise TypeError("An Function or Indexed object is expected not %s"%(idx_obj))
        return Basic.__new__(cls, idx_obj, pspace)

    symbol = property(lambda self: self.args[0])
    name = property(lambda self: str(self.args[0]))

    @property
    def key(self):
        if isinstance(self.symbol, Indexed):
            return self.symbol.args[1]
        elif isinstance(self.symbol, Function):
            return self.symbol.args[0]

    @property
    def free_symbols(self):
        if self.key.free_symbols:
            free_syms = self.key.free_symbols
            free_syms.add(self)
            return free_syms
        return {self}
```
### 8 - sympy/core/symbol.py:

Start line: 545, End line: 660

```python
def symbols(names, *, cls=Symbol, **args):
    r"""
    Transform strings into instances of :class:`Symbol` class.

    :func:`symbols` function returns a sequence of symbols with names taken
    from ``names`` argument, which can be a comma or whitespace delimited
    string, or a sequence of strings::

        >>> from sympy import symbols, Function

        >>> x, y, z = symbols('x,y,z')
        >>> a, b, c = symbols('a b c')

    The type of output is dependent on the properties of input arguments::

        >>> symbols('x')
        x
        >>> symbols('x,')
        (x,)
        >>> symbols('x,y')
        (x, y)
        >>> symbols(('a', 'b', 'c'))
        (a, b, c)
        >>> symbols(['a', 'b', 'c'])
        [a, b, c]
        >>> symbols({'a', 'b', 'c'})
        {a, b, c}

    If an iterable container is needed for a single symbol, set the ``seq``
    argument to ``True`` or terminate the symbol name with a comma::

        >>> symbols('x', seq=True)
        (x,)

    To reduce typing, range syntax is supported to create indexed symbols.
    Ranges are indicated by a colon and the type of range is determined by
    the character to the right of the colon. If the character is a digit
    then all contiguous digits to the left are taken as the nonnegative
    starting value (or 0 if there is no digit left of the colon) and all
    contiguous digits to the right are taken as 1 greater than the ending
    value::

        >>> symbols('x:10')
        (x0, x1, x2, x3, x4, x5, x6, x7, x8, x9)

        >>> symbols('x5:10')
        (x5, x6, x7, x8, x9)
        >>> symbols('x5(:2)')
        (x50, x51)

        >>> symbols('x5:10,y:5')
        (x5, x6, x7, x8, x9, y0, y1, y2, y3, y4)

        >>> symbols(('x5:10', 'y:5'))
        ((x5, x6, x7, x8, x9), (y0, y1, y2, y3, y4))

    If the character to the right of the colon is a letter, then the single
    letter to the left (or 'a' if there is none) is taken as the start
    and all characters in the lexicographic range *through* the letter to
    the right are used as the range::

        >>> symbols('x:z')
        (x, y, z)
        >>> symbols('x:c')  # null range
        ()
        >>> symbols('x(:c)')
        (xa, xb, xc)

        >>> symbols(':c')
        (a, b, c)

        >>> symbols('a:d, x:z')
        (a, b, c, d, x, y, z)

        >>> symbols(('a:d', 'x:z'))
        ((a, b, c, d), (x, y, z))

    Multiple ranges are supported; contiguous numerical ranges should be
    separated by parentheses to disambiguate the ending number of one
    range from the starting number of the next::

        >>> symbols('x:2(1:3)')
        (x01, x02, x11, x12)
        >>> symbols(':3:2')  # parsing is from left to right
        (00, 01, 10, 11, 20, 21)

    Only one pair of parentheses surrounding ranges are removed, so to
    include parentheses around ranges, double them. And to include spaces,
    commas, or colons, escape them with a backslash::

        >>> symbols('x((a:b))')
        (x(a), x(b))
        >>> symbols(r'x(:1\,:2)')  # or r'x((:1)\,(:2))'
        (x(0,0), x(0,1))

    All newly created symbols have assumptions set according to ``args``::

        >>> a = symbols('a', integer=True)
        >>> a.is_integer
        True

        >>> x, y, z = symbols('x,y,z', real=True)
        >>> x.is_real and y.is_real and z.is_real
        True

    Despite its name, :func:`symbols` can create symbol-like objects like
    instances of Function or Wild classes. To achieve this, set ``cls``
    keyword argument to the desired type::

        >>> symbols('f,g,h', cls=Function)
        (f, g, h)

        >>> type(_[0])
        <class 'sympy.core.function.UndefinedFunction'>

    """
    # ... other code
```
### 9 - sympy/core/symbol.py:

Start line: 661, End line: 754

```python
def symbols(names, *, cls=Symbol, **args):
    result = []

    if isinstance(names, str):
        marker = 0
        literals = [r'\,', r'\:', r'\ ']
        for i in range(len(literals)):
            lit = literals.pop(0)
            if lit in names:
                while chr(marker) in names:
                    marker += 1
                lit_char = chr(marker)
                marker += 1
                names = names.replace(lit, lit_char)
                literals.append((lit_char, lit[1:]))
        def literal(s):
            if literals:
                for c, l in literals:
                    s = s.replace(c, l)
            return s

        names = names.strip()
        as_seq = names.endswith(',')
        if as_seq:
            names = names[:-1].rstrip()
        if not names:
            raise ValueError('no symbols given')

        # split on commas
        names = [n.strip() for n in names.split(',')]
        if not all(n for n in names):
            raise ValueError('missing symbol between commas')
        # split on spaces
        for i in range(len(names) - 1, -1, -1):
            names[i: i + 1] = names[i].split()

        seq = args.pop('seq', as_seq)

        for name in names:
            if not name:
                raise ValueError('missing symbol')

            if ':' not in name:
                symbol = cls(literal(name), **args)
                result.append(symbol)
                continue

            split = _range.split(name)
            # remove 1 layer of bounding parentheses around ranges
            for i in range(len(split) - 1):
                if i and ':' in split[i] and split[i] != ':' and \
                        split[i - 1].endswith('(') and \
                        split[i + 1].startswith(')'):
                    split[i - 1] = split[i - 1][:-1]
                    split[i + 1] = split[i + 1][1:]
            for i, s in enumerate(split):
                if ':' in s:
                    if s[-1].endswith(':'):
                        raise ValueError('missing end range')
                    a, b = s.split(':')
                    if b[-1] in string.digits:
                        a = 0 if not a else int(a)
                        b = int(b)
                        split[i] = [str(c) for c in range(a, b)]
                    else:
                        a = a or 'a'
                        split[i] = [string.ascii_letters[c] for c in range(
                            string.ascii_letters.index(a),
                            string.ascii_letters.index(b) + 1)]  # inclusive
                    if not split[i]:
                        break
                else:
                    split[i] = [s]
            else:
                seq = True
                if len(split) == 1:
                    names = split[0]
                else:
                    names = [''.join(s) for s in cartes(*split)]
                if literals:
                    result.extend([cls(literal(s), **args) for s in names])
                else:
                    result.extend([cls(s, **args) for s in names])

        if not seq and len(result) <= 1:
            if not result:
                return ()
            return result[0]

        return tuple(result)
    else:
        for name in names:
            result.append(symbols(name, **args))

        return type(names)(result)
```
### 10 - sympy/core/symbol.py:

Start line: 512, End line: 543

```python
class Wild(Symbol):

    def __new__(cls, name, exclude=(), properties=(), **assumptions):
        exclude = tuple([sympify(x) for x in exclude])
        properties = tuple(properties)
        cls._sanitize(assumptions, cls)
        return Wild.__xnew__(cls, name, exclude, properties, **assumptions)

    def __getnewargs__(self):
        return (self.name, self.exclude, self.properties)

    @staticmethod
    @cacheit
    def __xnew__(cls, name, exclude, properties, **assumptions):
        obj = Symbol.__xnew__(cls, name, **assumptions)
        obj.exclude = exclude
        obj.properties = properties
        return obj

    def _hashable_content(self):
        return super()._hashable_content() + (self.exclude, self.properties)

    # TODO add check against another Wild
    def matches(self, expr, repl_dict={}, old=False):
        if any(expr.has(x) for x in self.exclude):
            return None
        if any(not f(expr) for f in self.properties):
            return None
        repl_dict = repl_dict.copy()
        repl_dict[self] = expr
        return repl_dict


_range = _re.compile('([0-9]*:[0-9]+|[a-zA-Z]?:[a-zA-Z])')
```
