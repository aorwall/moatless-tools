# sympy__sympy-18198

| **sympy/sympy** | `74b8046b46c70b201fe118cc36b29ce6c0d3b9ec` |
| ---- | ---- |
| **No of patches** | 21 |
| **All found context length** | 17332 |
| **Any found context length** | 9579 |
| **Avg pos** | 2.9523809523809526 |
| **Min pos** | 12 |
| **Max pos** | 20 |
| **Top file pos** | 1 |
| **Missing snippets** | 76 |
| **Missing patch files** | 15 |


## Expected patch

```diff
diff --git a/sympy/combinatorics/permutations.py b/sympy/combinatorics/permutations.py
--- a/sympy/combinatorics/permutations.py
+++ b/sympy/combinatorics/permutations.py
@@ -4,7 +4,7 @@
 from collections import defaultdict
 
 from sympy.core.basic import Atom, Basic
-from sympy.core.evaluate import global_evaluate
+from sympy.core.parameters import global_parameters
 from sympy.core.expr import Expr
 from sympy.core.compatibility import \
     is_sequence, reduce, range, as_int, Iterable
@@ -3002,7 +3002,10 @@ class AppliedPermutation(Expr):
     >>> _.subs(x, 1)
     2
     """
-    def __new__(cls, perm, x, evaluate=global_evaluate[0]):
+    def __new__(cls, perm, x, evaluate=None):
+        if evaluate is None:
+            evaluate = global_parameters.evaluate
+
         perm = _sympify(perm)
         x = _sympify(x)
 
diff --git a/sympy/core/__init__.py b/sympy/core/__init__.py
--- a/sympy/core/__init__.py
+++ b/sympy/core/__init__.py
@@ -26,7 +26,7 @@
 from .evalf import PrecisionExhausted, N
 from .containers import Tuple, Dict
 from .exprtools import gcd_terms, factor_terms, factor_nc
-from .evaluate import evaluate
+from .parameters import evaluate
 
 # expose singletons
 Catalan = S.Catalan
diff --git a/sympy/core/add.py b/sympy/core/add.py
--- a/sympy/core/add.py
+++ b/sympy/core/add.py
@@ -5,7 +5,7 @@
 
 from .basic import Basic
 from .compatibility import reduce, is_sequence, range
-from .evaluate import global_distribute
+from .parameters import global_parameters
 from .logic import _fuzzy_group, fuzzy_or, fuzzy_not
 from .singleton import S
 from .operations import AssocOp
@@ -1085,7 +1085,7 @@ def _mpc_(self):
         return (Float(re_part)._mpf_, Float(im_part)._mpf_)
 
     def __neg__(self):
-        if not global_distribute[0]:
+        if not global_parameters.distribute:
             return super(Add, self).__neg__()
         return Add(*[-i for i in self.args])
 
diff --git a/sympy/core/evaluate.py b/sympy/core/evaluate.py
deleted file mode 100644
--- a/sympy/core/evaluate.py
+++ /dev/null
@@ -1,72 +0,0 @@
-from .cache import clear_cache
-from contextlib import contextmanager
-
-
-class _global_function(list):
-    """ The cache must be cleared whenever _global_function is changed. """
-
-    def __setitem__(self, key, value):
-        if (self[key] != value):
-            clear_cache()
-        super(_global_function, self).__setitem__(key, value)
-
-
-global_evaluate = _global_function([True])
-global_distribute = _global_function([True])
-
-
-@contextmanager
-def evaluate(x):
-    """ Control automatic evaluation
-
-    This context manager controls whether or not all SymPy functions evaluate
-    by default.
-
-    Note that much of SymPy expects evaluated expressions.  This functionality
-    is experimental and is unlikely to function as intended on large
-    expressions.
-
-    Examples
-    ========
-
-    >>> from sympy.abc import x
-    >>> from sympy.core.evaluate import evaluate
-    >>> print(x + x)
-    2*x
-    >>> with evaluate(False):
-    ...     print(x + x)
-    x + x
-    """
-
-    old = global_evaluate[0]
-
-    global_evaluate[0] = x
-    yield
-    global_evaluate[0] = old
-
-
-@contextmanager
-def distribute(x):
-    """ Control automatic distribution of Number over Add
-
-    This context manager controls whether or not Mul distribute Number over
-    Add. Plan is to avoid distributing Number over Add in all of sympy. Once
-    that is done, this contextmanager will be removed.
-
-    Examples
-    ========
-
-    >>> from sympy.abc import x
-    >>> from sympy.core.evaluate import distribute
-    >>> print(2*(x + 1))
-    2*x + 2
-    >>> with distribute(False):
-    ...     print(2*(x + 1))
-    2*(x + 1)
-    """
-
-    old = global_distribute[0]
-
-    global_distribute[0] = x
-    yield
-    global_distribute[0] = old
diff --git a/sympy/core/function.py b/sympy/core/function.py
--- a/sympy/core/function.py
+++ b/sympy/core/function.py
@@ -46,7 +46,7 @@
 
 from sympy.core.compatibility import string_types, with_metaclass, PY3, range
 from sympy.core.containers import Tuple, Dict
-from sympy.core.evaluate import global_evaluate
+from sympy.core.parameters import global_parameters
 from sympy.core.logic import fuzzy_and
 from sympy.utilities import default_sort_key
 from sympy.utilities.exceptions import SymPyDeprecationWarning
@@ -276,7 +276,7 @@ def __new__(cls, *args, **options):
         from sympy.sets.sets import FiniteSet
 
         args = list(map(sympify, args))
-        evaluate = options.pop('evaluate', global_evaluate[0])
+        evaluate = options.pop('evaluate', global_parameters.evaluate)
         # WildFunction (and anything else like it) may have nargs defined
         # and we throw that value away here
         options.pop('nargs', None)
@@ -469,7 +469,7 @@ def __new__(cls, *args, **options):
                 'plural': 's'*(min(cls.nargs) != 1),
                 'given': n})
 
-        evaluate = options.get('evaluate', global_evaluate[0])
+        evaluate = options.get('evaluate', global_parameters.evaluate)
         result = super(Function, cls).__new__(cls, *args, **options)
         if evaluate and isinstance(result, cls) and result.args:
             pr2 = min(cls._should_evalf(a) for a in result.args)
diff --git a/sympy/core/mul.py b/sympy/core/mul.py
--- a/sympy/core/mul.py
+++ b/sympy/core/mul.py
@@ -12,7 +12,7 @@
 from .logic import fuzzy_not, _fuzzy_group
 from .compatibility import reduce, range
 from .expr import Expr
-from .evaluate import global_distribute
+from .parameters import global_parameters
 
 
 
@@ -202,7 +202,7 @@ def flatten(cls, seq):
                     if r is not S.One:  # 2-arg hack
                         # leave the Mul as a Mul
                         rv = [cls(a*r, b, evaluate=False)], [], None
-                    elif global_distribute[0] and b.is_commutative:
+                    elif global_parameters.distribute and b.is_commutative:
                         r, b = b.as_coeff_Add()
                         bargs = [_keep_coeff(a, bi) for bi in Add.make_args(b)]
                         _addsort(bargs)
@@ -626,7 +626,7 @@ def _handle_for_oo(c_part, coeff_sign):
             c_part.insert(0, coeff)
 
         # we are done
-        if (global_distribute[0] and not nc_part and len(c_part) == 2 and
+        if (global_parameters.distribute and not nc_part and len(c_part) == 2 and
                 c_part[0].is_Number and c_part[0].is_finite and c_part[1].is_Add):
             # 2*(1+a) -> 2 + 2 * a
             coeff = c_part[0]
diff --git a/sympy/core/numbers.py b/sympy/core/numbers.py
--- a/sympy/core/numbers.py
+++ b/sympy/core/numbers.py
@@ -30,7 +30,7 @@
     fnan as _mpf_nan, fzero, _normalize as mpf_normalize,
     prec_to_dps, fone, fnone)
 from sympy.utilities.misc import debug, filldedent
-from .evaluate import global_evaluate
+from .parameters import global_parameters
 
 from sympy.utilities.exceptions import SymPyDeprecationWarning
 
@@ -711,7 +711,7 @@ def sort_key(self, order=None):
 
     @_sympifyit('other', NotImplemented)
     def __add__(self, other):
-        if isinstance(other, Number) and global_evaluate[0]:
+        if isinstance(other, Number) and global_parameters.evaluate:
             if other is S.NaN:
                 return S.NaN
             elif other is S.Infinity:
@@ -722,7 +722,7 @@ def __add__(self, other):
 
     @_sympifyit('other', NotImplemented)
     def __sub__(self, other):
-        if isinstance(other, Number) and global_evaluate[0]:
+        if isinstance(other, Number) and global_parameters.evaluate:
             if other is S.NaN:
                 return S.NaN
             elif other is S.Infinity:
@@ -733,7 +733,7 @@ def __sub__(self, other):
 
     @_sympifyit('other', NotImplemented)
     def __mul__(self, other):
-        if isinstance(other, Number) and global_evaluate[0]:
+        if isinstance(other, Number) and global_parameters.evaluate:
             if other is S.NaN:
                 return S.NaN
             elif other is S.Infinity:
@@ -756,7 +756,7 @@ def __mul__(self, other):
 
     @_sympifyit('other', NotImplemented)
     def __div__(self, other):
-        if isinstance(other, Number) and global_evaluate[0]:
+        if isinstance(other, Number) and global_parameters.evaluate:
             if other is S.NaN:
                 return S.NaN
             elif other is S.Infinity or other is S.NegativeInfinity:
@@ -1291,28 +1291,28 @@ def __neg__(self):
 
     @_sympifyit('other', NotImplemented)
     def __add__(self, other):
-        if isinstance(other, Number) and global_evaluate[0]:
+        if isinstance(other, Number) and global_parameters.evaluate:
             rhs, prec = other._as_mpf_op(self._prec)
             return Float._new(mlib.mpf_add(self._mpf_, rhs, prec, rnd), prec)
         return Number.__add__(self, other)
 
     @_sympifyit('other', NotImplemented)
     def __sub__(self, other):
-        if isinstance(other, Number) and global_evaluate[0]:
+        if isinstance(other, Number) and global_parameters.evaluate:
             rhs, prec = other._as_mpf_op(self._prec)
             return Float._new(mlib.mpf_sub(self._mpf_, rhs, prec, rnd), prec)
         return Number.__sub__(self, other)
 
     @_sympifyit('other', NotImplemented)
     def __mul__(self, other):
-        if isinstance(other, Number) and global_evaluate[0]:
+        if isinstance(other, Number) and global_parameters.evaluate:
             rhs, prec = other._as_mpf_op(self._prec)
             return Float._new(mlib.mpf_mul(self._mpf_, rhs, prec, rnd), prec)
         return Number.__mul__(self, other)
 
     @_sympifyit('other', NotImplemented)
     def __div__(self, other):
-        if isinstance(other, Number) and other != 0 and global_evaluate[0]:
+        if isinstance(other, Number) and other != 0 and global_parameters.evaluate:
             rhs, prec = other._as_mpf_op(self._prec)
             return Float._new(mlib.mpf_div(self._mpf_, rhs, prec, rnd), prec)
         return Number.__div__(self, other)
@@ -1321,24 +1321,24 @@ def __div__(self, other):
 
     @_sympifyit('other', NotImplemented)
     def __mod__(self, other):
-        if isinstance(other, Rational) and other.q != 1 and global_evaluate[0]:
+        if isinstance(other, Rational) and other.q != 1 and global_parameters.evaluate:
             # calculate mod with Rationals, *then* round the result
             return Float(Rational.__mod__(Rational(self), other),
                          precision=self._prec)
-        if isinstance(other, Float) and global_evaluate[0]:
+        if isinstance(other, Float) and global_parameters.evaluate:
             r = self/other
             if r == int(r):
                 return Float(0, precision=max(self._prec, other._prec))
-        if isinstance(other, Number) and global_evaluate[0]:
+        if isinstance(other, Number) and global_parameters.evaluate:
             rhs, prec = other._as_mpf_op(self._prec)
             return Float._new(mlib.mpf_mod(self._mpf_, rhs, prec, rnd), prec)
         return Number.__mod__(self, other)
 
     @_sympifyit('other', NotImplemented)
     def __rmod__(self, other):
-        if isinstance(other, Float) and global_evaluate[0]:
+        if isinstance(other, Float) and global_parameters.evaluate:
             return other.__mod__(self)
-        if isinstance(other, Number) and global_evaluate[0]:
+        if isinstance(other, Number) and global_parameters.evaluate:
             rhs, prec = other._as_mpf_op(self._prec)
             return Float._new(mlib.mpf_mod(rhs, self._mpf_, prec, rnd), prec)
         return Number.__rmod__(self, other)
@@ -1696,7 +1696,7 @@ def __neg__(self):
 
     @_sympifyit('other', NotImplemented)
     def __add__(self, other):
-        if global_evaluate[0]:
+        if global_parameters.evaluate:
             if isinstance(other, Integer):
                 return Rational(self.p + self.q*other.p, self.q, 1)
             elif isinstance(other, Rational):
@@ -1711,7 +1711,7 @@ def __add__(self, other):
 
     @_sympifyit('other', NotImplemented)
     def __sub__(self, other):
-        if global_evaluate[0]:
+        if global_parameters.evaluate:
             if isinstance(other, Integer):
                 return Rational(self.p - self.q*other.p, self.q, 1)
             elif isinstance(other, Rational):
@@ -1723,7 +1723,7 @@ def __sub__(self, other):
         return Number.__sub__(self, other)
     @_sympifyit('other', NotImplemented)
     def __rsub__(self, other):
-        if global_evaluate[0]:
+        if global_parameters.evaluate:
             if isinstance(other, Integer):
                 return Rational(self.q*other.p - self.p, self.q, 1)
             elif isinstance(other, Rational):
@@ -1735,7 +1735,7 @@ def __rsub__(self, other):
         return Number.__rsub__(self, other)
     @_sympifyit('other', NotImplemented)
     def __mul__(self, other):
-        if global_evaluate[0]:
+        if global_parameters.evaluate:
             if isinstance(other, Integer):
                 return Rational(self.p*other.p, self.q, igcd(other.p, self.q))
             elif isinstance(other, Rational):
@@ -1749,7 +1749,7 @@ def __mul__(self, other):
 
     @_sympifyit('other', NotImplemented)
     def __div__(self, other):
-        if global_evaluate[0]:
+        if global_parameters.evaluate:
             if isinstance(other, Integer):
                 if self.p and other.p == S.Zero:
                     return S.ComplexInfinity
@@ -1764,7 +1764,7 @@ def __div__(self, other):
         return Number.__div__(self, other)
     @_sympifyit('other', NotImplemented)
     def __rdiv__(self, other):
-        if global_evaluate[0]:
+        if global_parameters.evaluate:
             if isinstance(other, Integer):
                 return Rational(other.p*self.q, self.p, igcd(self.p, other.p))
             elif isinstance(other, Rational):
@@ -1778,7 +1778,7 @@ def __rdiv__(self, other):
 
     @_sympifyit('other', NotImplemented)
     def __mod__(self, other):
-        if global_evaluate[0]:
+        if global_parameters.evaluate:
             if isinstance(other, Rational):
                 n = (self.p*other.q) // (other.p*self.q)
                 return Rational(self.p*other.q - n*other.p*self.q, self.q*other.q)
@@ -2139,14 +2139,14 @@ def __abs__(self):
 
     def __divmod__(self, other):
         from .containers import Tuple
-        if isinstance(other, Integer) and global_evaluate[0]:
+        if isinstance(other, Integer) and global_parameters.evaluate:
             return Tuple(*(divmod(self.p, other.p)))
         else:
             return Number.__divmod__(self, other)
 
     def __rdivmod__(self, other):
         from .containers import Tuple
-        if isinstance(other, integer_types) and global_evaluate[0]:
+        if isinstance(other, integer_types) and global_parameters.evaluate:
             return Tuple(*(divmod(other, self.p)))
         else:
             try:
@@ -2160,7 +2160,7 @@ def __rdivmod__(self, other):
 
     # TODO make it decorator + bytecodehacks?
     def __add__(self, other):
-        if global_evaluate[0]:
+        if global_parameters.evaluate:
             if isinstance(other, integer_types):
                 return Integer(self.p + other)
             elif isinstance(other, Integer):
@@ -2172,7 +2172,7 @@ def __add__(self, other):
             return Add(self, other)
 
     def __radd__(self, other):
-        if global_evaluate[0]:
+        if global_parameters.evaluate:
             if isinstance(other, integer_types):
                 return Integer(other + self.p)
             elif isinstance(other, Rational):
@@ -2181,7 +2181,7 @@ def __radd__(self, other):
         return Rational.__radd__(self, other)
 
     def __sub__(self, other):
-        if global_evaluate[0]:
+        if global_parameters.evaluate:
             if isinstance(other, integer_types):
                 return Integer(self.p - other)
             elif isinstance(other, Integer):
@@ -2192,7 +2192,7 @@ def __sub__(self, other):
         return Rational.__sub__(self, other)
 
     def __rsub__(self, other):
-        if global_evaluate[0]:
+        if global_parameters.evaluate:
             if isinstance(other, integer_types):
                 return Integer(other - self.p)
             elif isinstance(other, Rational):
@@ -2201,7 +2201,7 @@ def __rsub__(self, other):
         return Rational.__rsub__(self, other)
 
     def __mul__(self, other):
-        if global_evaluate[0]:
+        if global_parameters.evaluate:
             if isinstance(other, integer_types):
                 return Integer(self.p*other)
             elif isinstance(other, Integer):
@@ -2212,7 +2212,7 @@ def __mul__(self, other):
         return Rational.__mul__(self, other)
 
     def __rmul__(self, other):
-        if global_evaluate[0]:
+        if global_parameters.evaluate:
             if isinstance(other, integer_types):
                 return Integer(other*self.p)
             elif isinstance(other, Rational):
@@ -2221,7 +2221,7 @@ def __rmul__(self, other):
         return Rational.__rmul__(self, other)
 
     def __mod__(self, other):
-        if global_evaluate[0]:
+        if global_parameters.evaluate:
             if isinstance(other, integer_types):
                 return Integer(self.p % other)
             elif isinstance(other, Integer):
@@ -2230,7 +2230,7 @@ def __mod__(self, other):
         return Rational.__mod__(self, other)
 
     def __rmod__(self, other):
-        if global_evaluate[0]:
+        if global_parameters.evaluate:
             if isinstance(other, integer_types):
                 return Integer(other % self.p)
             elif isinstance(other, Integer):
@@ -2845,7 +2845,7 @@ def evalf(self, prec=None, **options):
 
     @_sympifyit('other', NotImplemented)
     def __add__(self, other):
-        if isinstance(other, Number) and global_evaluate[0]:
+        if isinstance(other, Number) and global_parameters.evaluate:
             if other is S.NegativeInfinity or other is S.NaN:
                 return S.NaN
             return self
@@ -2854,7 +2854,7 @@ def __add__(self, other):
 
     @_sympifyit('other', NotImplemented)
     def __sub__(self, other):
-        if isinstance(other, Number) and global_evaluate[0]:
+        if isinstance(other, Number) and global_parameters.evaluate:
             if other is S.Infinity or other is S.NaN:
                 return S.NaN
             return self
@@ -2866,7 +2866,7 @@ def __rsub__(self, other):
 
     @_sympifyit('other', NotImplemented)
     def __mul__(self, other):
-        if isinstance(other, Number) and global_evaluate[0]:
+        if isinstance(other, Number) and global_parameters.evaluate:
             if other.is_zero or other is S.NaN:
                 return S.NaN
             if other.is_extended_positive:
@@ -2877,7 +2877,7 @@ def __mul__(self, other):
 
     @_sympifyit('other', NotImplemented)
     def __div__(self, other):
-        if isinstance(other, Number) and global_evaluate[0]:
+        if isinstance(other, Number) and global_parameters.evaluate:
             if other is S.Infinity or \
                 other is S.NegativeInfinity or \
                     other is S.NaN:
@@ -3013,7 +3013,7 @@ def evalf(self, prec=None, **options):
 
     @_sympifyit('other', NotImplemented)
     def __add__(self, other):
-        if isinstance(other, Number) and global_evaluate[0]:
+        if isinstance(other, Number) and global_parameters.evaluate:
             if other is S.Infinity or other is S.NaN:
                 return S.NaN
             return self
@@ -3022,7 +3022,7 @@ def __add__(self, other):
 
     @_sympifyit('other', NotImplemented)
     def __sub__(self, other):
-        if isinstance(other, Number) and global_evaluate[0]:
+        if isinstance(other, Number) and global_parameters.evaluate:
             if other is S.NegativeInfinity or other is S.NaN:
                 return S.NaN
             return self
@@ -3034,7 +3034,7 @@ def __rsub__(self, other):
 
     @_sympifyit('other', NotImplemented)
     def __mul__(self, other):
-        if isinstance(other, Number) and global_evaluate[0]:
+        if isinstance(other, Number) and global_parameters.evaluate:
             if other.is_zero or other is S.NaN:
                 return S.NaN
             if other.is_extended_positive:
@@ -3045,7 +3045,7 @@ def __mul__(self, other):
 
     @_sympifyit('other', NotImplemented)
     def __div__(self, other):
-        if isinstance(other, Number) and global_evaluate[0]:
+        if isinstance(other, Number) and global_parameters.evaluate:
             if other is S.Infinity or \
                 other is S.NegativeInfinity or \
                     other is S.NaN:
diff --git a/sympy/core/operations.py b/sympy/core/operations.py
--- a/sympy/core/operations.py
+++ b/sympy/core/operations.py
@@ -5,7 +5,7 @@
 from sympy.core.cache import cacheit
 from sympy.core.compatibility import ordered, range
 from sympy.core.logic import fuzzy_and
-from sympy.core.evaluate import global_evaluate
+from sympy.core.parameters import global_parameters
 from sympy.utilities.iterables import sift
 
 
@@ -38,7 +38,7 @@ def __new__(cls, *args, **options):
 
         evaluate = options.get('evaluate')
         if evaluate is None:
-            evaluate = global_evaluate[0]
+            evaluate = global_parameters.evaluate
         if not evaluate:
             obj = cls._from_args(args)
             obj = cls._exec_constructor_postprocessors(obj)
diff --git a/sympy/core/parameters.py b/sympy/core/parameters.py
new file mode 100644
--- /dev/null
+++ b/sympy/core/parameters.py
@@ -0,0 +1,128 @@
+"""Thread-safe global parameters"""
+
+from .cache import clear_cache
+from contextlib import contextmanager
+from threading import local
+
+class _global_parameters(local):
+    """
+    Thread-local global parameters.
+
+    Explanation
+    ===========
+
+    This class generates thread-local container for SymPy's global parameters.
+    Every global parameters must be passed as keyword argument when generating
+    its instance.
+    A variable, `global_parameters` is provided as default instance for this class.
+
+    WARNING! Although the global parameters are thread-local, SymPy's cache is not
+    by now.
+    This may lead to undesired result in multi-threading operations.
+
+    Examples
+    ========
+
+    >>> from sympy.abc import x
+    >>> from sympy.core.cache import clear_cache
+    >>> from sympy.core.parameters import global_parameters as gp
+
+    >>> gp.evaluate
+    True
+    >>> x+x
+    2*x
+
+    >>> log = []
+    >>> def f():
+    ...     clear_cache()
+    ...     gp.evaluate = False
+    ...     log.append(x+x)
+    ...     clear_cache()
+    >>> import threading
+    >>> thread = threading.Thread(target=f)
+    >>> thread.start()
+    >>> thread.join()
+
+    >>> print(log)
+    [x + x]
+
+    >>> gp.evaluate
+    True
+    >>> x+x
+    2*x
+
+    References
+    ==========
+
+    .. [1] https://docs.python.org/3/library/threading.html
+
+    """
+    def __init__(self, **kwargs):
+        self.__dict__.update(kwargs)
+
+    def __setattr__(self, name, value):
+        if getattr(self, name) != value:
+            clear_cache()
+        return super(_global_parameters, self).__setattr__(name, value)
+
+global_parameters = _global_parameters(evaluate=True, distribute=True)
+
+@contextmanager
+def evaluate(x):
+    """ Control automatic evaluation
+
+    This context manager controls whether or not all SymPy functions evaluate
+    by default.
+
+    Note that much of SymPy expects evaluated expressions.  This functionality
+    is experimental and is unlikely to function as intended on large
+    expressions.
+
+    Examples
+    ========
+
+    >>> from sympy.abc import x
+    >>> from sympy.core.parameters import evaluate
+    >>> print(x + x)
+    2*x
+    >>> with evaluate(False):
+    ...     print(x + x)
+    x + x
+    """
+
+    old = global_parameters.evaluate
+
+    try:
+        global_parameters.evaluate = x
+        yield
+    finally:
+        global_parameters.evaluate = old
+
+
+@contextmanager
+def distribute(x):
+    """ Control automatic distribution of Number over Add
+
+    This context manager controls whether or not Mul distribute Number over
+    Add. Plan is to avoid distributing Number over Add in all of sympy. Once
+    that is done, this contextmanager will be removed.
+
+    Examples
+    ========
+
+    >>> from sympy.abc import x
+    >>> from sympy.core.parameters import distribute
+    >>> print(2*(x + 1))
+    2*x + 2
+    >>> with distribute(False):
+    ...     print(2*(x + 1))
+    2*(x + 1)
+    """
+
+    old = global_parameters.distribute
+
+    try:
+        global_parameters.distribute = x
+        yield
+    finally:
+        global_parameters.distribute = old
diff --git a/sympy/core/power.py b/sympy/core/power.py
--- a/sympy/core/power.py
+++ b/sympy/core/power.py
@@ -11,7 +11,7 @@
     expand_mul)
 from .logic import fuzzy_bool, fuzzy_not, fuzzy_and
 from .compatibility import as_int, range
-from .evaluate import global_evaluate
+from .parameters import global_parameters
 from sympy.utilities.iterables import sift
 
 from mpmath.libmp import sqrtrem as mpmath_sqrtrem
@@ -257,7 +257,7 @@ class Pow(Expr):
     @cacheit
     def __new__(cls, b, e, evaluate=None):
         if evaluate is None:
-            evaluate = global_evaluate[0]
+            evaluate = global_parameters.evaluate
         from sympy.functions.elementary.exponential import exp_polar
 
         b = _sympify(b)
diff --git a/sympy/core/relational.py b/sympy/core/relational.py
--- a/sympy/core/relational.py
+++ b/sympy/core/relational.py
@@ -8,7 +8,7 @@
 from .expr import Expr
 from .evalf import EvalfMixin
 from .sympify import _sympify
-from .evaluate import global_evaluate
+from .parameters import global_parameters
 
 from sympy.logic.boolalg import Boolean, BooleanAtom
 
@@ -483,7 +483,7 @@ def __new__(cls, lhs, rhs=None, **options):
         lhs = _sympify(lhs)
         rhs = _sympify(rhs)
 
-        evaluate = options.pop('evaluate', global_evaluate[0])
+        evaluate = options.pop('evaluate', global_parameters.evaluate)
 
         if evaluate:
             # If one expression has an _eval_Eq, return its results.
@@ -713,7 +713,7 @@ def __new__(cls, lhs, rhs, **options):
         lhs = _sympify(lhs)
         rhs = _sympify(rhs)
 
-        evaluate = options.pop('evaluate', global_evaluate[0])
+        evaluate = options.pop('evaluate', global_parameters.evaluate)
 
         if evaluate:
             is_equal = Equality(lhs, rhs)
@@ -760,7 +760,7 @@ def __new__(cls, lhs, rhs, **options):
         lhs = _sympify(lhs)
         rhs = _sympify(rhs)
 
-        evaluate = options.pop('evaluate', global_evaluate[0])
+        evaluate = options.pop('evaluate', global_parameters.evaluate)
 
         if evaluate:
             # First we invoke the appropriate inequality method of `lhs`
diff --git a/sympy/core/sympify.py b/sympy/core/sympify.py
--- a/sympy/core/sympify.py
+++ b/sympy/core/sympify.py
@@ -6,7 +6,7 @@
 
 from .core import all_classes as sympy_classes
 from .compatibility import iterable, string_types, range
-from .evaluate import global_evaluate
+from .parameters import global_parameters
 
 
 class SympifyError(ValueError):
@@ -288,10 +288,7 @@ def sympify(a, locals=None, convert_xor=True, strict=False, rational=False,
             return a
 
     if evaluate is None:
-        if global_evaluate[0] is False:
-            evaluate = global_evaluate[0]
-        else:
-            evaluate = True
+        evaluate = global_parameters.evaluate
 
     # Support for basic numpy datatypes
     # Note that this check exists to avoid importing NumPy when not necessary
diff --git a/sympy/geometry/ellipse.py b/sympy/geometry/ellipse.py
--- a/sympy/geometry/ellipse.py
+++ b/sympy/geometry/ellipse.py
@@ -10,7 +10,7 @@
 
 from sympy import Expr, Eq
 from sympy.core import S, pi, sympify
-from sympy.core.evaluate import global_evaluate
+from sympy.core.parameters import global_parameters
 from sympy.core.logic import fuzzy_bool
 from sympy.core.numbers import Rational, oo
 from sympy.core.compatibility import ordered
@@ -1546,7 +1546,7 @@ class Circle(Ellipse):
     def __new__(cls, *args, **kwargs):
         from sympy.geometry.util import find
         from .polygon import Triangle
-        evaluate = kwargs.get('evaluate', global_evaluate[0])
+        evaluate = kwargs.get('evaluate', global_parameters.evaluate)
         if len(args) == 1 and isinstance(args[0], (Expr, Eq)):
             x = kwargs.get('x', 'x')
             y = kwargs.get('y', 'y')
diff --git a/sympy/geometry/point.py b/sympy/geometry/point.py
--- a/sympy/geometry/point.py
+++ b/sympy/geometry/point.py
@@ -30,7 +30,7 @@
 from sympy.functions.elementary.complexes import im
 from sympy.matrices import Matrix
 from sympy.core.numbers import Float
-from sympy.core.evaluate import global_evaluate
+from sympy.core.parameters import global_parameters
 from sympy.core.add import Add
 from sympy.utilities.iterables import uniq
 from sympy.utilities.misc import filldedent, func_name, Undecidable
@@ -106,7 +106,7 @@ class Point(GeometryEntity):
     is_Point = True
 
     def __new__(cls, *args, **kwargs):
-        evaluate = kwargs.get('evaluate', global_evaluate[0])
+        evaluate = kwargs.get('evaluate', global_parameters.evaluate)
         on_morph = kwargs.get('on_morph', 'ignore')
 
         # unpack into coords
diff --git a/sympy/series/sequences.py b/sympy/series/sequences.py
--- a/sympy/series/sequences.py
+++ b/sympy/series/sequences.py
@@ -6,7 +6,7 @@
                                       is_sequence, iterable, ordered)
 from sympy.core.containers import Tuple
 from sympy.core.decorators import call_highest_priority
-from sympy.core.evaluate import global_evaluate
+from sympy.core.parameters import global_parameters
 from sympy.core.function import UndefinedFunction
 from sympy.core.mul import Mul
 from sympy.core.numbers import Integer
@@ -1005,7 +1005,7 @@ class SeqAdd(SeqExprOp):
     """
 
     def __new__(cls, *args, **kwargs):
-        evaluate = kwargs.get('evaluate', global_evaluate[0])
+        evaluate = kwargs.get('evaluate', global_parameters.evaluate)
 
         # flatten inputs
         args = list(args)
@@ -1114,7 +1114,7 @@ class SeqMul(SeqExprOp):
     """
 
     def __new__(cls, *args, **kwargs):
-        evaluate = kwargs.get('evaluate', global_evaluate[0])
+        evaluate = kwargs.get('evaluate', global_parameters.evaluate)
 
         # flatten inputs
         args = list(args)
diff --git a/sympy/sets/powerset.py b/sympy/sets/powerset.py
--- a/sympy/sets/powerset.py
+++ b/sympy/sets/powerset.py
@@ -1,7 +1,7 @@
 from __future__ import print_function, division
 
 from sympy.core.decorators import _sympifyit
-from sympy.core.evaluate import global_evaluate
+from sympy.core.parameters import global_parameters
 from sympy.core.logic import fuzzy_bool
 from sympy.core.singleton import S
 from sympy.core.sympify import _sympify
@@ -71,7 +71,10 @@ class PowerSet(Set):
 
     .. [2] https://en.wikipedia.org/wiki/Axiom_of_power_set
     """
-    def __new__(cls, arg, evaluate=global_evaluate[0]):
+    def __new__(cls, arg, evaluate=None):
+        if evaluate is None:
+            evaluate=global_parameters.evaluate
+
         arg = _sympify(arg)
 
         if not isinstance(arg, Set):
diff --git a/sympy/sets/sets.py b/sympy/sets/sets.py
--- a/sympy/sets/sets.py
+++ b/sympy/sets/sets.py
@@ -11,7 +11,7 @@
 from sympy.core.decorators import (deprecated, sympify_method_args,
     sympify_return)
 from sympy.core.evalf import EvalfMixin
-from sympy.core.evaluate import global_evaluate
+from sympy.core.parameters import global_parameters
 from sympy.core.expr import Expr
 from sympy.core.logic import fuzzy_bool, fuzzy_or, fuzzy_and, fuzzy_not
 from sympy.core.numbers import Float
@@ -1172,7 +1172,7 @@ def zero(self):
         return S.UniversalSet
 
     def __new__(cls, *args, **kwargs):
-        evaluate = kwargs.get('evaluate', global_evaluate[0])
+        evaluate = kwargs.get('evaluate', global_parameters.evaluate)
 
         # flatten inputs to merge intersections and iterables
         args = _sympify(args)
@@ -1345,7 +1345,7 @@ def zero(self):
         return S.EmptySet
 
     def __new__(cls, *args, **kwargs):
-        evaluate = kwargs.get('evaluate', global_evaluate[0])
+        evaluate = kwargs.get('evaluate', global_parameters.evaluate)
 
         # flatten inputs to merge intersections and iterables
         args = list(ordered(set(_sympify(args))))
@@ -1767,7 +1767,7 @@ class FiniteSet(Set, EvalfMixin):
     is_finite_set = True
 
     def __new__(cls, *args, **kwargs):
-        evaluate = kwargs.get('evaluate', global_evaluate[0])
+        evaluate = kwargs.get('evaluate', global_parameters.evaluate)
         if evaluate:
             args = list(map(sympify, args))
 
diff --git a/sympy/simplify/radsimp.py b/sympy/simplify/radsimp.py
--- a/sympy/simplify/radsimp.py
+++ b/sympy/simplify/radsimp.py
@@ -7,7 +7,7 @@
 from sympy.core import expand_power_base, sympify, Add, S, Mul, Derivative, Pow, symbols, expand_mul
 from sympy.core.add import _unevaluated_Add
 from sympy.core.compatibility import iterable, ordered, default_sort_key
-from sympy.core.evaluate import global_evaluate
+from sympy.core.parameters import global_parameters
 from sympy.core.exprtools import Factors, gcd_terms
 from sympy.core.function import _mexpand
 from sympy.core.mul import _keep_coeff, _unevaluated_Mul
@@ -163,7 +163,7 @@ def collect(expr, syms, func=None, evaluate=None, exact=False, distribute_order_
     syms = list(syms) if iterable(syms) else [syms]
 
     if evaluate is None:
-        evaluate = global_evaluate[0]
+        evaluate = global_parameters.evaluate
 
     def make_expression(terms):
         product = []
@@ -496,7 +496,7 @@ def collect_sqrt(expr, evaluate=None):
     collect, collect_const, rcollect
     """
     if evaluate is None:
-        evaluate = global_evaluate[0]
+        evaluate = global_parameters.evaluate
     # this step will help to standardize any complex arguments
     # of sqrts
     coeff, expr = expr.as_content_primitive()
diff --git a/sympy/simplify/simplify.py b/sympy/simplify/simplify.py
--- a/sympy/simplify/simplify.py
+++ b/sympy/simplify/simplify.py
@@ -6,7 +6,7 @@
                         expand_func, Function, Dummy, Expr, factor_terms,
                         expand_power_exp, Eq)
 from sympy.core.compatibility import iterable, ordered, range, as_int
-from sympy.core.evaluate import global_evaluate
+from sympy.core.parameters import global_parameters
 from sympy.core.function import expand_log, count_ops, _mexpand, _coeff_isneg, \
     nfloat, expand_mul, expand_multinomial
 from sympy.core.numbers import Float, I, pi, Rational, Integer
@@ -378,7 +378,7 @@ def signsimp(expr, evaluate=None):
 
     """
     if evaluate is None:
-        evaluate = global_evaluate[0]
+        evaluate = global_parameters.evaluate
     expr = sympify(expr)
     if not isinstance(expr, (Expr, Relational)) or expr.is_Atom:
         return expr
diff --git a/sympy/stats/symbolic_probability.py b/sympy/stats/symbolic_probability.py
--- a/sympy/stats/symbolic_probability.py
+++ b/sympy/stats/symbolic_probability.py
@@ -2,7 +2,7 @@
 
 from sympy import Expr, Add, Mul, S, Integral, Eq, Sum, Symbol
 from sympy.core.compatibility import default_sort_key
-from sympy.core.evaluate import global_evaluate
+from sympy.core.parameters import global_parameters
 from sympy.core.sympify import _sympify
 from sympy.stats import variance, covariance
 from sympy.stats.rv import RandomSymbol, probability, expectation
@@ -324,7 +324,7 @@ def __new__(cls, arg1, arg2, condition=None, **kwargs):
         arg1 = _sympify(arg1)
         arg2 = _sympify(arg2)
 
-        if kwargs.pop('evaluate', global_evaluate[0]):
+        if kwargs.pop('evaluate', global_parameters.evaluate):
             arg1, arg2 = sorted([arg1, arg2], key=default_sort_key)
 
         if condition is None:
diff --git a/sympy/tensor/functions.py b/sympy/tensor/functions.py
--- a/sympy/tensor/functions.py
+++ b/sympy/tensor/functions.py
@@ -1,6 +1,6 @@
 from sympy import Expr, S, Mul, sympify
 from sympy.core.compatibility import Iterable
-from sympy.core.evaluate import global_evaluate
+from sympy.core.parameters import global_parameters
 
 
 class TensorProduct(Expr):
@@ -15,7 +15,7 @@ def __new__(cls, *args, **kwargs):
         from sympy.strategies import flatten
 
         args = [sympify(arg) for arg in args]
-        evaluate = kwargs.get("evaluate", global_evaluate[0])
+        evaluate = kwargs.get("evaluate", global_parameters.evaluate)
 
         if not evaluate:
             obj = Expr.__new__(cls, *args)

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sympy/combinatorics/permutations.py | 7 | 7 | - | - | -
| sympy/combinatorics/permutations.py | 3005 | 3005 | - | - | -
| sympy/core/__init__.py | 29 | 29 | 20 | 12 | 17332
| sympy/core/add.py | 8 | 8 | 17 | 10 | 16218
| sympy/core/add.py | 1088 | 1088 | - | 10 | -
| sympy/core/evaluate.py | 1 | 72 | - | 1 | -
| sympy/core/function.py | 49 | 49 | 12 | 2 | 9579
| sympy/core/function.py | 279 | 279 | - | 2 | -
| sympy/core/function.py | 472 | 472 | 13 | 2 | 9920
| sympy/core/mul.py | 15 | 15 | - | - | -
| sympy/core/mul.py | 205 | 205 | - | - | -
| sympy/core/mul.py | 629 | 629 | - | - | -
| sympy/core/numbers.py | 33 | 33 | - | - | -
| sympy/core/numbers.py | 714 | 714 | - | - | -
| sympy/core/numbers.py | 725 | 725 | - | - | -
| sympy/core/numbers.py | 736 | 736 | - | - | -
| sympy/core/numbers.py | 759 | 759 | - | - | -
| sympy/core/numbers.py | 1294 | 1315 | - | - | -
| sympy/core/numbers.py | 1324 | 1341 | - | - | -
| sympy/core/numbers.py | 1699 | 1699 | - | - | -
| sympy/core/numbers.py | 1714 | 1714 | - | - | -
| sympy/core/numbers.py | 1726 | 1726 | - | - | -
| sympy/core/numbers.py | 1738 | 1738 | - | - | -
| sympy/core/numbers.py | 1752 | 1752 | - | - | -
| sympy/core/numbers.py | 1767 | 1767 | - | - | -
| sympy/core/numbers.py | 1781 | 1781 | - | - | -
| sympy/core/numbers.py | 2142 | 2149 | - | - | -
| sympy/core/numbers.py | 2163 | 2163 | - | - | -
| sympy/core/numbers.py | 2175 | 2175 | - | - | -
| sympy/core/numbers.py | 2184 | 2184 | - | - | -
| sympy/core/numbers.py | 2195 | 2195 | - | - | -
| sympy/core/numbers.py | 2204 | 2204 | - | - | -
| sympy/core/numbers.py | 2215 | 2215 | - | - | -
| sympy/core/numbers.py | 2224 | 2224 | - | - | -
| sympy/core/numbers.py | 2233 | 2233 | - | - | -
| sympy/core/numbers.py | 2848 | 2848 | - | - | -
| sympy/core/numbers.py | 2857 | 2857 | - | - | -
| sympy/core/numbers.py | 2869 | 2869 | - | - | -
| sympy/core/numbers.py | 2880 | 2880 | - | - | -
| sympy/core/numbers.py | 3016 | 3016 | - | - | -
| sympy/core/numbers.py | 3025 | 3025 | - | - | -
| sympy/core/numbers.py | 3037 | 3037 | - | - | -
| sympy/core/numbers.py | 3048 | 3048 | - | - | -
| sympy/core/operations.py | 8 | 8 | - | - | -
| sympy/core/operations.py | 41 | 41 | - | - | -
| sympy/core/parameters.py | 0 | 0 | - | - | -
| sympy/core/power.py | 14 | 14 | - | 15 | -
| sympy/core/power.py | 260 | 260 | - | 15 | -
| sympy/core/relational.py | 11 | 11 | - | - | -
| sympy/core/relational.py | 486 | 486 | - | - | -
| sympy/core/relational.py | 716 | 716 | - | - | -
| sympy/core/relational.py | 763 | 763 | - | - | -
| sympy/core/sympify.py | 9 | 9 | - | 6 | -
| sympy/core/sympify.py | 291 | 294 | - | 6 | -
| sympy/geometry/ellipse.py | 13 | 13 | - | - | -
| sympy/geometry/ellipse.py | 1549 | 1549 | - | - | -
| sympy/geometry/point.py | 33 | 33 | - | - | -
| sympy/geometry/point.py | 109 | 109 | - | - | -
| sympy/series/sequences.py | 9 | 9 | - | - | -
| sympy/series/sequences.py | 1008 | 1008 | - | - | -
| sympy/series/sequences.py | 1117 | 1117 | - | - | -
| sympy/sets/powerset.py | 4 | 4 | - | - | -
| sympy/sets/powerset.py | 74 | 74 | - | - | -
| sympy/sets/sets.py | 14 | 14 | - | - | -
| sympy/sets/sets.py | 1175 | 1175 | - | - | -
| sympy/sets/sets.py | 1348 | 1348 | - | - | -
| sympy/sets/sets.py | 1770 | 1770 | - | - | -
| sympy/simplify/radsimp.py | 10 | 10 | - | - | -
| sympy/simplify/radsimp.py | 166 | 166 | - | - | -
| sympy/simplify/radsimp.py | 499 | 499 | - | - | -
| sympy/simplify/simplify.py | 9 | 9 | - | - | -
| sympy/simplify/simplify.py | 381 | 381 | - | - | -
| sympy/stats/symbolic_probability.py | 5 | 5 | - | - | -
| sympy/stats/symbolic_probability.py | 327 | 327 | - | - | -
| sympy/tensor/functions.py | 3 | 3 | - | - | -
| sympy/tensor/functions.py | 18 | 18 | - | - | -


## Problem Statement

```
Suggestion on `core.evaluate` module
As I understand, `core.evaluate` module is first developed to handle the global value of `evaluate` parameter. Then, it is extended to handle `distribute` parameter as well.
Since more global parameters might appear in the future, I think this module can be renamed to `core.parameters` for clarity.

Besides that, if more parameters are added, it will be annoying to have all `global_foo[0]`, `global_bar[0]`, and so on. I am thinking of a dict-like handler named `global_parameters` to manage every global parameters. It will behave like this:

1. Its `__getitem__()` method returns `global_foo` object.
\`\`\`
>>> global_parameters
{'evaluate': [True], 'distribute': [True]}
>>> global_parameters['evaluate']
[True]
\`\`\`

2. It has `foo` property that returns or sets the value of global `foo`.
\`\`\`
>>> global_parameters.evaluate
True
>>> global_parameters.evaluate = False
>>> global_parameters.evaluate
False
>>> global_parameters
{'evaluate': [False], 'distribute': [True]}
\`\`\`

3. Its properties are not `bool` - They are callable new classes so that they can be used as context manager.
\`\`\`
>>> from sympy.abc import x
>>> with global_parameters.evaluate(False):
         print(x + x)
x + x
\`\`\`

I have already written a code which satisfies suggestion 1 and 2. It seems to be working well. How does everyone think about it?

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 sympy/core/evaluate.py** | 1 | 45| 249 | 249 | 419 | 
| 2 | **2 sympy/core/function.py** | 2453 | 2778| 3102 | 3351 | 28016 | 
| 3 | 3 sympy/plotting/experimental_lambdify.py | 1 | 76| 868 | 4219 | 33919 | 
| 4 | **3 sympy/core/function.py** | 2269 | 2315| 313 | 4532 | 33919 | 
| 5 | 4 sympy/functions/special/hyper.py | 185 | 211| 321 | 4853 | 44230 | 
| 6 | **4 sympy/core/evaluate.py** | 48 | 73| 168 | 5021 | 44230 | 
| 7 | 5 sympy/core/mod.py | 165 | 209| 416 | 5437 | 45966 | 
| 8 | **6 sympy/core/sympify.py** | 80 | 264| 1781 | 7218 | 50239 | 
| 9 | 7 sympy/utilities/lambdify.py | 107 | 825| 513 | 7731 | 61739 | 
| 10 | 8 sympy/core/basic.py | 892 | 976| 726 | 8457 | 77683 | 
| 11 | 9 sympy/core/evalf.py | 1249 | 1305| 636 | 9093 | 91417 | 
| **-> 12 <-** | **9 sympy/core/function.py** | 1 | 60| 486 | 9579 | 91417 | 
| **-> 13 <-** | **9 sympy/core/function.py** | 450 | 480| 341 | 9920 | 91417 | 
| 14 | 9 sympy/utilities/lambdify.py | 173 | 682| 5180 | 15100 | 91417 | 
| 15 | 9 sympy/plotting/experimental_lambdify.py | 247 | 347| 832 | 15932 | 91417 | 
| 16 | 9 sympy/core/mod.py | 211 | 228| 141 | 16073 | 91417 | 
| **-> 17 <-** | **10 sympy/core/add.py** | 1 | 22| 145 | 16218 | 100433 | 
| 18 | 11 sympy/assumptions/sathandlers.py | 225 | 260| 278 | 16496 | 104033 | 
| 19 | 11 sympy/core/basic.py | 1727 | 1790| 467 | 16963 | 104033 | 
| **-> 20 <-** | **12 sympy/core/__init__.py** | 1 | 36| 369 | 17332 | 104403 | 
| 21 | 13 sympy/simplify/hyperexpand.py | 84 | 96| 160 | 17492 | 129230 | 
| 22 | 14 sympy/core/expr.py | 3399 | 3427| 245 | 17737 | 162581 | 
| 23 | **14 sympy/core/function.py** | 2208 | 2267| 482 | 18219 | 162581 | 
| 24 | 14 sympy/functions/special/hyper.py | 461 | 482| 271 | 18490 | 162581 | 
| 25 | **14 sympy/core/function.py** | 482 | 503| 186 | 18676 | 162581 | 
| 26 | **15 sympy/core/power.py** | 783 | 837| 645 | 19321 | 178093 | 
| 27 | 15 sympy/utilities/lambdify.py | 1 | 91| 658 | 19979 | 178093 | 
| 28 | 15 sympy/functions/special/hyper.py | 633 | 665| 383 | 20362 | 178093 | 
| 29 | **15 sympy/core/add.py** | 495 | 537| 503 | 20865 | 178093 | 
| 30 | 16 sympy/functions/elementary/piecewise.py | 148 | 208| 542 | 21407 | 189363 | 
| 31 | **16 sympy/core/function.py** | 2317 | 2339| 277 | 21684 | 189363 | 
| 32 | **16 sympy/core/function.py** | 318 | 356| 289 | 21973 | 189363 | 
| 33 | **16 sympy/core/function.py** | 2144 | 2206| 564 | 22537 | 189363 | 
| 34 | 16 sympy/core/mod.py | 92 | 164| 669 | 23206 | 189363 | 
| 35 | 17 sympy/core/symbol.py | 261 | 316| 409 | 23615 | 195972 | 
| 36 | 17 sympy/utilities/lambdify.py | 1046 | 1084| 373 | 23988 | 195972 | 
| 37 | 17 sympy/core/basic.py | 779 | 891| 1046 | 25034 | 195972 | 
| 38 | **17 sympy/core/function.py** | 605 | 628| 173 | 25207 | 195972 | 
| 39 | **17 sympy/core/function.py** | 704 | 746| 469 | 25676 | 195972 | 


## Missing Patch Files

 * 1: sympy/combinatorics/permutations.py
 * 2: sympy/core/__init__.py
 * 3: sympy/core/add.py
 * 4: ev/null
 * 5: sympy/core/function.py
 * 6: sympy/core/mul.py
 * 7: sympy/core/numbers.py
 * 8: sympy/core/operations.py
 * 9: sympy/core/parameters.py
 * 10: sympy/core/power.py
 * 11: sympy/core/relational.py
 * 12: sympy/core/sympify.py
 * 13: sympy/geometry/ellipse.py
 * 14: sympy/geometry/point.py
 * 15: sympy/series/sequences.py
 * 16: sympy/sets/powerset.py
 * 17: sympy/sets/sets.py
 * 18: sympy/simplify/radsimp.py
 * 19: sympy/simplify/simplify.py
 * 20: sympy/stats/symbolic_probability.py
 * 21: sympy/tensor/functions.py

### Hint

```
Is your code thread-safe?
> Is your code thread-safe?

I didn't check it.
`global_parameters` is singleton and relegates every operations to `global_foo` it contains, so hopefully it will cause no problem as long as `global_foo` does the job right.

Can you suggest the way to check its thread safety?
We should really use thread local storage rather than global variables. Mutable global variables that are not thread-local are not thread-safe.
> We should really use thread local storage rather than global variables. Mutable global variables that are not thread-local are not thread-safe.

Thanks. I didn't know about thread safety before. I'm curious: does the current approach in `core.evaluate` satisfies thread safety? How does it achieve this?
Also, then how can I ensure the thread safety of `global_parameters`? Will blocking suggestion 2 (make the global parameter mutable by implementing setter) do? 
I don't think that the current approach is thread-safe because it just uses a list. That list will be shared between threads so if one thread sets global evaluate to True then it will affect the other threads. To be thread-safe this global needs to use thread-local storage.
```

## Patch

```diff
diff --git a/sympy/combinatorics/permutations.py b/sympy/combinatorics/permutations.py
--- a/sympy/combinatorics/permutations.py
+++ b/sympy/combinatorics/permutations.py
@@ -4,7 +4,7 @@
 from collections import defaultdict
 
 from sympy.core.basic import Atom, Basic
-from sympy.core.evaluate import global_evaluate
+from sympy.core.parameters import global_parameters
 from sympy.core.expr import Expr
 from sympy.core.compatibility import \
     is_sequence, reduce, range, as_int, Iterable
@@ -3002,7 +3002,10 @@ class AppliedPermutation(Expr):
     >>> _.subs(x, 1)
     2
     """
-    def __new__(cls, perm, x, evaluate=global_evaluate[0]):
+    def __new__(cls, perm, x, evaluate=None):
+        if evaluate is None:
+            evaluate = global_parameters.evaluate
+
         perm = _sympify(perm)
         x = _sympify(x)
 
diff --git a/sympy/core/__init__.py b/sympy/core/__init__.py
--- a/sympy/core/__init__.py
+++ b/sympy/core/__init__.py
@@ -26,7 +26,7 @@
 from .evalf import PrecisionExhausted, N
 from .containers import Tuple, Dict
 from .exprtools import gcd_terms, factor_terms, factor_nc
-from .evaluate import evaluate
+from .parameters import evaluate
 
 # expose singletons
 Catalan = S.Catalan
diff --git a/sympy/core/add.py b/sympy/core/add.py
--- a/sympy/core/add.py
+++ b/sympy/core/add.py
@@ -5,7 +5,7 @@
 
 from .basic import Basic
 from .compatibility import reduce, is_sequence, range
-from .evaluate import global_distribute
+from .parameters import global_parameters
 from .logic import _fuzzy_group, fuzzy_or, fuzzy_not
 from .singleton import S
 from .operations import AssocOp
@@ -1085,7 +1085,7 @@ def _mpc_(self):
         return (Float(re_part)._mpf_, Float(im_part)._mpf_)
 
     def __neg__(self):
-        if not global_distribute[0]:
+        if not global_parameters.distribute:
             return super(Add, self).__neg__()
         return Add(*[-i for i in self.args])
 
diff --git a/sympy/core/evaluate.py b/sympy/core/evaluate.py
deleted file mode 100644
--- a/sympy/core/evaluate.py
+++ /dev/null
@@ -1,72 +0,0 @@
-from .cache import clear_cache
-from contextlib import contextmanager
-
-
-class _global_function(list):
-    """ The cache must be cleared whenever _global_function is changed. """
-
-    def __setitem__(self, key, value):
-        if (self[key] != value):
-            clear_cache()
-        super(_global_function, self).__setitem__(key, value)
-
-
-global_evaluate = _global_function([True])
-global_distribute = _global_function([True])
-
-
-@contextmanager
-def evaluate(x):
-    """ Control automatic evaluation
-
-    This context manager controls whether or not all SymPy functions evaluate
-    by default.
-
-    Note that much of SymPy expects evaluated expressions.  This functionality
-    is experimental and is unlikely to function as intended on large
-    expressions.
-
-    Examples
-    ========
-
-    >>> from sympy.abc import x
-    >>> from sympy.core.evaluate import evaluate
-    >>> print(x + x)
-    2*x
-    >>> with evaluate(False):
-    ...     print(x + x)
-    x + x
-    """
-
-    old = global_evaluate[0]
-
-    global_evaluate[0] = x
-    yield
-    global_evaluate[0] = old
-
-
-@contextmanager
-def distribute(x):
-    """ Control automatic distribution of Number over Add
-
-    This context manager controls whether or not Mul distribute Number over
-    Add. Plan is to avoid distributing Number over Add in all of sympy. Once
-    that is done, this contextmanager will be removed.
-
-    Examples
-    ========
-
-    >>> from sympy.abc import x
-    >>> from sympy.core.evaluate import distribute
-    >>> print(2*(x + 1))
-    2*x + 2
-    >>> with distribute(False):
-    ...     print(2*(x + 1))
-    2*(x + 1)
-    """
-
-    old = global_distribute[0]
-
-    global_distribute[0] = x
-    yield
-    global_distribute[0] = old
diff --git a/sympy/core/function.py b/sympy/core/function.py
--- a/sympy/core/function.py
+++ b/sympy/core/function.py
@@ -46,7 +46,7 @@
 
 from sympy.core.compatibility import string_types, with_metaclass, PY3, range
 from sympy.core.containers import Tuple, Dict
-from sympy.core.evaluate import global_evaluate
+from sympy.core.parameters import global_parameters
 from sympy.core.logic import fuzzy_and
 from sympy.utilities import default_sort_key
 from sympy.utilities.exceptions import SymPyDeprecationWarning
@@ -276,7 +276,7 @@ def __new__(cls, *args, **options):
         from sympy.sets.sets import FiniteSet
 
         args = list(map(sympify, args))
-        evaluate = options.pop('evaluate', global_evaluate[0])
+        evaluate = options.pop('evaluate', global_parameters.evaluate)
         # WildFunction (and anything else like it) may have nargs defined
         # and we throw that value away here
         options.pop('nargs', None)
@@ -469,7 +469,7 @@ def __new__(cls, *args, **options):
                 'plural': 's'*(min(cls.nargs) != 1),
                 'given': n})
 
-        evaluate = options.get('evaluate', global_evaluate[0])
+        evaluate = options.get('evaluate', global_parameters.evaluate)
         result = super(Function, cls).__new__(cls, *args, **options)
         if evaluate and isinstance(result, cls) and result.args:
             pr2 = min(cls._should_evalf(a) for a in result.args)
diff --git a/sympy/core/mul.py b/sympy/core/mul.py
--- a/sympy/core/mul.py
+++ b/sympy/core/mul.py
@@ -12,7 +12,7 @@
 from .logic import fuzzy_not, _fuzzy_group
 from .compatibility import reduce, range
 from .expr import Expr
-from .evaluate import global_distribute
+from .parameters import global_parameters
 
 
 
@@ -202,7 +202,7 @@ def flatten(cls, seq):
                     if r is not S.One:  # 2-arg hack
                         # leave the Mul as a Mul
                         rv = [cls(a*r, b, evaluate=False)], [], None
-                    elif global_distribute[0] and b.is_commutative:
+                    elif global_parameters.distribute and b.is_commutative:
                         r, b = b.as_coeff_Add()
                         bargs = [_keep_coeff(a, bi) for bi in Add.make_args(b)]
                         _addsort(bargs)
@@ -626,7 +626,7 @@ def _handle_for_oo(c_part, coeff_sign):
             c_part.insert(0, coeff)
 
         # we are done
-        if (global_distribute[0] and not nc_part and len(c_part) == 2 and
+        if (global_parameters.distribute and not nc_part and len(c_part) == 2 and
                 c_part[0].is_Number and c_part[0].is_finite and c_part[1].is_Add):
             # 2*(1+a) -> 2 + 2 * a
             coeff = c_part[0]
diff --git a/sympy/core/numbers.py b/sympy/core/numbers.py
--- a/sympy/core/numbers.py
+++ b/sympy/core/numbers.py
@@ -30,7 +30,7 @@
     fnan as _mpf_nan, fzero, _normalize as mpf_normalize,
     prec_to_dps, fone, fnone)
 from sympy.utilities.misc import debug, filldedent
-from .evaluate import global_evaluate
+from .parameters import global_parameters
 
 from sympy.utilities.exceptions import SymPyDeprecationWarning
 
@@ -711,7 +711,7 @@ def sort_key(self, order=None):
 
     @_sympifyit('other', NotImplemented)
     def __add__(self, other):
-        if isinstance(other, Number) and global_evaluate[0]:
+        if isinstance(other, Number) and global_parameters.evaluate:
             if other is S.NaN:
                 return S.NaN
             elif other is S.Infinity:
@@ -722,7 +722,7 @@ def __add__(self, other):
 
     @_sympifyit('other', NotImplemented)
     def __sub__(self, other):
-        if isinstance(other, Number) and global_evaluate[0]:
+        if isinstance(other, Number) and global_parameters.evaluate:
             if other is S.NaN:
                 return S.NaN
             elif other is S.Infinity:
@@ -733,7 +733,7 @@ def __sub__(self, other):
 
     @_sympifyit('other', NotImplemented)
     def __mul__(self, other):
-        if isinstance(other, Number) and global_evaluate[0]:
+        if isinstance(other, Number) and global_parameters.evaluate:
             if other is S.NaN:
                 return S.NaN
             elif other is S.Infinity:
@@ -756,7 +756,7 @@ def __mul__(self, other):
 
     @_sympifyit('other', NotImplemented)
     def __div__(self, other):
-        if isinstance(other, Number) and global_evaluate[0]:
+        if isinstance(other, Number) and global_parameters.evaluate:
             if other is S.NaN:
                 return S.NaN
             elif other is S.Infinity or other is S.NegativeInfinity:
@@ -1291,28 +1291,28 @@ def __neg__(self):
 
     @_sympifyit('other', NotImplemented)
     def __add__(self, other):
-        if isinstance(other, Number) and global_evaluate[0]:
+        if isinstance(other, Number) and global_parameters.evaluate:
             rhs, prec = other._as_mpf_op(self._prec)
             return Float._new(mlib.mpf_add(self._mpf_, rhs, prec, rnd), prec)
         return Number.__add__(self, other)
 
     @_sympifyit('other', NotImplemented)
     def __sub__(self, other):
-        if isinstance(other, Number) and global_evaluate[0]:
+        if isinstance(other, Number) and global_parameters.evaluate:
             rhs, prec = other._as_mpf_op(self._prec)
             return Float._new(mlib.mpf_sub(self._mpf_, rhs, prec, rnd), prec)
         return Number.__sub__(self, other)
 
     @_sympifyit('other', NotImplemented)
     def __mul__(self, other):
-        if isinstance(other, Number) and global_evaluate[0]:
+        if isinstance(other, Number) and global_parameters.evaluate:
             rhs, prec = other._as_mpf_op(self._prec)
             return Float._new(mlib.mpf_mul(self._mpf_, rhs, prec, rnd), prec)
         return Number.__mul__(self, other)
 
     @_sympifyit('other', NotImplemented)
     def __div__(self, other):
-        if isinstance(other, Number) and other != 0 and global_evaluate[0]:
+        if isinstance(other, Number) and other != 0 and global_parameters.evaluate:
             rhs, prec = other._as_mpf_op(self._prec)
             return Float._new(mlib.mpf_div(self._mpf_, rhs, prec, rnd), prec)
         return Number.__div__(self, other)
@@ -1321,24 +1321,24 @@ def __div__(self, other):
 
     @_sympifyit('other', NotImplemented)
     def __mod__(self, other):
-        if isinstance(other, Rational) and other.q != 1 and global_evaluate[0]:
+        if isinstance(other, Rational) and other.q != 1 and global_parameters.evaluate:
             # calculate mod with Rationals, *then* round the result
             return Float(Rational.__mod__(Rational(self), other),
                          precision=self._prec)
-        if isinstance(other, Float) and global_evaluate[0]:
+        if isinstance(other, Float) and global_parameters.evaluate:
             r = self/other
             if r == int(r):
                 return Float(0, precision=max(self._prec, other._prec))
-        if isinstance(other, Number) and global_evaluate[0]:
+        if isinstance(other, Number) and global_parameters.evaluate:
             rhs, prec = other._as_mpf_op(self._prec)
             return Float._new(mlib.mpf_mod(self._mpf_, rhs, prec, rnd), prec)
         return Number.__mod__(self, other)
 
     @_sympifyit('other', NotImplemented)
     def __rmod__(self, other):
-        if isinstance(other, Float) and global_evaluate[0]:
+        if isinstance(other, Float) and global_parameters.evaluate:
             return other.__mod__(self)
-        if isinstance(other, Number) and global_evaluate[0]:
+        if isinstance(other, Number) and global_parameters.evaluate:
             rhs, prec = other._as_mpf_op(self._prec)
             return Float._new(mlib.mpf_mod(rhs, self._mpf_, prec, rnd), prec)
         return Number.__rmod__(self, other)
@@ -1696,7 +1696,7 @@ def __neg__(self):
 
     @_sympifyit('other', NotImplemented)
     def __add__(self, other):
-        if global_evaluate[0]:
+        if global_parameters.evaluate:
             if isinstance(other, Integer):
                 return Rational(self.p + self.q*other.p, self.q, 1)
             elif isinstance(other, Rational):
@@ -1711,7 +1711,7 @@ def __add__(self, other):
 
     @_sympifyit('other', NotImplemented)
     def __sub__(self, other):
-        if global_evaluate[0]:
+        if global_parameters.evaluate:
             if isinstance(other, Integer):
                 return Rational(self.p - self.q*other.p, self.q, 1)
             elif isinstance(other, Rational):
@@ -1723,7 +1723,7 @@ def __sub__(self, other):
         return Number.__sub__(self, other)
     @_sympifyit('other', NotImplemented)
     def __rsub__(self, other):
-        if global_evaluate[0]:
+        if global_parameters.evaluate:
             if isinstance(other, Integer):
                 return Rational(self.q*other.p - self.p, self.q, 1)
             elif isinstance(other, Rational):
@@ -1735,7 +1735,7 @@ def __rsub__(self, other):
         return Number.__rsub__(self, other)
     @_sympifyit('other', NotImplemented)
     def __mul__(self, other):
-        if global_evaluate[0]:
+        if global_parameters.evaluate:
             if isinstance(other, Integer):
                 return Rational(self.p*other.p, self.q, igcd(other.p, self.q))
             elif isinstance(other, Rational):
@@ -1749,7 +1749,7 @@ def __mul__(self, other):
 
     @_sympifyit('other', NotImplemented)
     def __div__(self, other):
-        if global_evaluate[0]:
+        if global_parameters.evaluate:
             if isinstance(other, Integer):
                 if self.p and other.p == S.Zero:
                     return S.ComplexInfinity
@@ -1764,7 +1764,7 @@ def __div__(self, other):
         return Number.__div__(self, other)
     @_sympifyit('other', NotImplemented)
     def __rdiv__(self, other):
-        if global_evaluate[0]:
+        if global_parameters.evaluate:
             if isinstance(other, Integer):
                 return Rational(other.p*self.q, self.p, igcd(self.p, other.p))
             elif isinstance(other, Rational):
@@ -1778,7 +1778,7 @@ def __rdiv__(self, other):
 
     @_sympifyit('other', NotImplemented)
     def __mod__(self, other):
-        if global_evaluate[0]:
+        if global_parameters.evaluate:
             if isinstance(other, Rational):
                 n = (self.p*other.q) // (other.p*self.q)
                 return Rational(self.p*other.q - n*other.p*self.q, self.q*other.q)
@@ -2139,14 +2139,14 @@ def __abs__(self):
 
     def __divmod__(self, other):
         from .containers import Tuple
-        if isinstance(other, Integer) and global_evaluate[0]:
+        if isinstance(other, Integer) and global_parameters.evaluate:
             return Tuple(*(divmod(self.p, other.p)))
         else:
             return Number.__divmod__(self, other)
 
     def __rdivmod__(self, other):
         from .containers import Tuple
-        if isinstance(other, integer_types) and global_evaluate[0]:
+        if isinstance(other, integer_types) and global_parameters.evaluate:
             return Tuple(*(divmod(other, self.p)))
         else:
             try:
@@ -2160,7 +2160,7 @@ def __rdivmod__(self, other):
 
     # TODO make it decorator + bytecodehacks?
     def __add__(self, other):
-        if global_evaluate[0]:
+        if global_parameters.evaluate:
             if isinstance(other, integer_types):
                 return Integer(self.p + other)
             elif isinstance(other, Integer):
@@ -2172,7 +2172,7 @@ def __add__(self, other):
             return Add(self, other)
 
     def __radd__(self, other):
-        if global_evaluate[0]:
+        if global_parameters.evaluate:
             if isinstance(other, integer_types):
                 return Integer(other + self.p)
             elif isinstance(other, Rational):
@@ -2181,7 +2181,7 @@ def __radd__(self, other):
         return Rational.__radd__(self, other)
 
     def __sub__(self, other):
-        if global_evaluate[0]:
+        if global_parameters.evaluate:
             if isinstance(other, integer_types):
                 return Integer(self.p - other)
             elif isinstance(other, Integer):
@@ -2192,7 +2192,7 @@ def __sub__(self, other):
         return Rational.__sub__(self, other)
 
     def __rsub__(self, other):
-        if global_evaluate[0]:
+        if global_parameters.evaluate:
             if isinstance(other, integer_types):
                 return Integer(other - self.p)
             elif isinstance(other, Rational):
@@ -2201,7 +2201,7 @@ def __rsub__(self, other):
         return Rational.__rsub__(self, other)
 
     def __mul__(self, other):
-        if global_evaluate[0]:
+        if global_parameters.evaluate:
             if isinstance(other, integer_types):
                 return Integer(self.p*other)
             elif isinstance(other, Integer):
@@ -2212,7 +2212,7 @@ def __mul__(self, other):
         return Rational.__mul__(self, other)
 
     def __rmul__(self, other):
-        if global_evaluate[0]:
+        if global_parameters.evaluate:
             if isinstance(other, integer_types):
                 return Integer(other*self.p)
             elif isinstance(other, Rational):
@@ -2221,7 +2221,7 @@ def __rmul__(self, other):
         return Rational.__rmul__(self, other)
 
     def __mod__(self, other):
-        if global_evaluate[0]:
+        if global_parameters.evaluate:
             if isinstance(other, integer_types):
                 return Integer(self.p % other)
             elif isinstance(other, Integer):
@@ -2230,7 +2230,7 @@ def __mod__(self, other):
         return Rational.__mod__(self, other)
 
     def __rmod__(self, other):
-        if global_evaluate[0]:
+        if global_parameters.evaluate:
             if isinstance(other, integer_types):
                 return Integer(other % self.p)
             elif isinstance(other, Integer):
@@ -2845,7 +2845,7 @@ def evalf(self, prec=None, **options):
 
     @_sympifyit('other', NotImplemented)
     def __add__(self, other):
-        if isinstance(other, Number) and global_evaluate[0]:
+        if isinstance(other, Number) and global_parameters.evaluate:
             if other is S.NegativeInfinity or other is S.NaN:
                 return S.NaN
             return self
@@ -2854,7 +2854,7 @@ def __add__(self, other):
 
     @_sympifyit('other', NotImplemented)
     def __sub__(self, other):
-        if isinstance(other, Number) and global_evaluate[0]:
+        if isinstance(other, Number) and global_parameters.evaluate:
             if other is S.Infinity or other is S.NaN:
                 return S.NaN
             return self
@@ -2866,7 +2866,7 @@ def __rsub__(self, other):
 
     @_sympifyit('other', NotImplemented)
     def __mul__(self, other):
-        if isinstance(other, Number) and global_evaluate[0]:
+        if isinstance(other, Number) and global_parameters.evaluate:
             if other.is_zero or other is S.NaN:
                 return S.NaN
             if other.is_extended_positive:
@@ -2877,7 +2877,7 @@ def __mul__(self, other):
 
     @_sympifyit('other', NotImplemented)
     def __div__(self, other):
-        if isinstance(other, Number) and global_evaluate[0]:
+        if isinstance(other, Number) and global_parameters.evaluate:
             if other is S.Infinity or \
                 other is S.NegativeInfinity or \
                     other is S.NaN:
@@ -3013,7 +3013,7 @@ def evalf(self, prec=None, **options):
 
     @_sympifyit('other', NotImplemented)
     def __add__(self, other):
-        if isinstance(other, Number) and global_evaluate[0]:
+        if isinstance(other, Number) and global_parameters.evaluate:
             if other is S.Infinity or other is S.NaN:
                 return S.NaN
             return self
@@ -3022,7 +3022,7 @@ def __add__(self, other):
 
     @_sympifyit('other', NotImplemented)
     def __sub__(self, other):
-        if isinstance(other, Number) and global_evaluate[0]:
+        if isinstance(other, Number) and global_parameters.evaluate:
             if other is S.NegativeInfinity or other is S.NaN:
                 return S.NaN
             return self
@@ -3034,7 +3034,7 @@ def __rsub__(self, other):
 
     @_sympifyit('other', NotImplemented)
     def __mul__(self, other):
-        if isinstance(other, Number) and global_evaluate[0]:
+        if isinstance(other, Number) and global_parameters.evaluate:
             if other.is_zero or other is S.NaN:
                 return S.NaN
             if other.is_extended_positive:
@@ -3045,7 +3045,7 @@ def __mul__(self, other):
 
     @_sympifyit('other', NotImplemented)
     def __div__(self, other):
-        if isinstance(other, Number) and global_evaluate[0]:
+        if isinstance(other, Number) and global_parameters.evaluate:
             if other is S.Infinity or \
                 other is S.NegativeInfinity or \
                     other is S.NaN:
diff --git a/sympy/core/operations.py b/sympy/core/operations.py
--- a/sympy/core/operations.py
+++ b/sympy/core/operations.py
@@ -5,7 +5,7 @@
 from sympy.core.cache import cacheit
 from sympy.core.compatibility import ordered, range
 from sympy.core.logic import fuzzy_and
-from sympy.core.evaluate import global_evaluate
+from sympy.core.parameters import global_parameters
 from sympy.utilities.iterables import sift
 
 
@@ -38,7 +38,7 @@ def __new__(cls, *args, **options):
 
         evaluate = options.get('evaluate')
         if evaluate is None:
-            evaluate = global_evaluate[0]
+            evaluate = global_parameters.evaluate
         if not evaluate:
             obj = cls._from_args(args)
             obj = cls._exec_constructor_postprocessors(obj)
diff --git a/sympy/core/parameters.py b/sympy/core/parameters.py
new file mode 100644
--- /dev/null
+++ b/sympy/core/parameters.py
@@ -0,0 +1,128 @@
+"""Thread-safe global parameters"""
+
+from .cache import clear_cache
+from contextlib import contextmanager
+from threading import local
+
+class _global_parameters(local):
+    """
+    Thread-local global parameters.
+
+    Explanation
+    ===========
+
+    This class generates thread-local container for SymPy's global parameters.
+    Every global parameters must be passed as keyword argument when generating
+    its instance.
+    A variable, `global_parameters` is provided as default instance for this class.
+
+    WARNING! Although the global parameters are thread-local, SymPy's cache is not
+    by now.
+    This may lead to undesired result in multi-threading operations.
+
+    Examples
+    ========
+
+    >>> from sympy.abc import x
+    >>> from sympy.core.cache import clear_cache
+    >>> from sympy.core.parameters import global_parameters as gp
+
+    >>> gp.evaluate
+    True
+    >>> x+x
+    2*x
+
+    >>> log = []
+    >>> def f():
+    ...     clear_cache()
+    ...     gp.evaluate = False
+    ...     log.append(x+x)
+    ...     clear_cache()
+    >>> import threading
+    >>> thread = threading.Thread(target=f)
+    >>> thread.start()
+    >>> thread.join()
+
+    >>> print(log)
+    [x + x]
+
+    >>> gp.evaluate
+    True
+    >>> x+x
+    2*x
+
+    References
+    ==========
+
+    .. [1] https://docs.python.org/3/library/threading.html
+
+    """
+    def __init__(self, **kwargs):
+        self.__dict__.update(kwargs)
+
+    def __setattr__(self, name, value):
+        if getattr(self, name) != value:
+            clear_cache()
+        return super(_global_parameters, self).__setattr__(name, value)
+
+global_parameters = _global_parameters(evaluate=True, distribute=True)
+
+@contextmanager
+def evaluate(x):
+    """ Control automatic evaluation
+
+    This context manager controls whether or not all SymPy functions evaluate
+    by default.
+
+    Note that much of SymPy expects evaluated expressions.  This functionality
+    is experimental and is unlikely to function as intended on large
+    expressions.
+
+    Examples
+    ========
+
+    >>> from sympy.abc import x
+    >>> from sympy.core.parameters import evaluate
+    >>> print(x + x)
+    2*x
+    >>> with evaluate(False):
+    ...     print(x + x)
+    x + x
+    """
+
+    old = global_parameters.evaluate
+
+    try:
+        global_parameters.evaluate = x
+        yield
+    finally:
+        global_parameters.evaluate = old
+
+
+@contextmanager
+def distribute(x):
+    """ Control automatic distribution of Number over Add
+
+    This context manager controls whether or not Mul distribute Number over
+    Add. Plan is to avoid distributing Number over Add in all of sympy. Once
+    that is done, this contextmanager will be removed.
+
+    Examples
+    ========
+
+    >>> from sympy.abc import x
+    >>> from sympy.core.parameters import distribute
+    >>> print(2*(x + 1))
+    2*x + 2
+    >>> with distribute(False):
+    ...     print(2*(x + 1))
+    2*(x + 1)
+    """
+
+    old = global_parameters.distribute
+
+    try:
+        global_parameters.distribute = x
+        yield
+    finally:
+        global_parameters.distribute = old
diff --git a/sympy/core/power.py b/sympy/core/power.py
--- a/sympy/core/power.py
+++ b/sympy/core/power.py
@@ -11,7 +11,7 @@
     expand_mul)
 from .logic import fuzzy_bool, fuzzy_not, fuzzy_and
 from .compatibility import as_int, range
-from .evaluate import global_evaluate
+from .parameters import global_parameters
 from sympy.utilities.iterables import sift
 
 from mpmath.libmp import sqrtrem as mpmath_sqrtrem
@@ -257,7 +257,7 @@ class Pow(Expr):
     @cacheit
     def __new__(cls, b, e, evaluate=None):
         if evaluate is None:
-            evaluate = global_evaluate[0]
+            evaluate = global_parameters.evaluate
         from sympy.functions.elementary.exponential import exp_polar
 
         b = _sympify(b)
diff --git a/sympy/core/relational.py b/sympy/core/relational.py
--- a/sympy/core/relational.py
+++ b/sympy/core/relational.py
@@ -8,7 +8,7 @@
 from .expr import Expr
 from .evalf import EvalfMixin
 from .sympify import _sympify
-from .evaluate import global_evaluate
+from .parameters import global_parameters
 
 from sympy.logic.boolalg import Boolean, BooleanAtom
 
@@ -483,7 +483,7 @@ def __new__(cls, lhs, rhs=None, **options):
         lhs = _sympify(lhs)
         rhs = _sympify(rhs)
 
-        evaluate = options.pop('evaluate', global_evaluate[0])
+        evaluate = options.pop('evaluate', global_parameters.evaluate)
 
         if evaluate:
             # If one expression has an _eval_Eq, return its results.
@@ -713,7 +713,7 @@ def __new__(cls, lhs, rhs, **options):
         lhs = _sympify(lhs)
         rhs = _sympify(rhs)
 
-        evaluate = options.pop('evaluate', global_evaluate[0])
+        evaluate = options.pop('evaluate', global_parameters.evaluate)
 
         if evaluate:
             is_equal = Equality(lhs, rhs)
@@ -760,7 +760,7 @@ def __new__(cls, lhs, rhs, **options):
         lhs = _sympify(lhs)
         rhs = _sympify(rhs)
 
-        evaluate = options.pop('evaluate', global_evaluate[0])
+        evaluate = options.pop('evaluate', global_parameters.evaluate)
 
         if evaluate:
             # First we invoke the appropriate inequality method of `lhs`
diff --git a/sympy/core/sympify.py b/sympy/core/sympify.py
--- a/sympy/core/sympify.py
+++ b/sympy/core/sympify.py
@@ -6,7 +6,7 @@
 
 from .core import all_classes as sympy_classes
 from .compatibility import iterable, string_types, range
-from .evaluate import global_evaluate
+from .parameters import global_parameters
 
 
 class SympifyError(ValueError):
@@ -288,10 +288,7 @@ def sympify(a, locals=None, convert_xor=True, strict=False, rational=False,
             return a
 
     if evaluate is None:
-        if global_evaluate[0] is False:
-            evaluate = global_evaluate[0]
-        else:
-            evaluate = True
+        evaluate = global_parameters.evaluate
 
     # Support for basic numpy datatypes
     # Note that this check exists to avoid importing NumPy when not necessary
diff --git a/sympy/geometry/ellipse.py b/sympy/geometry/ellipse.py
--- a/sympy/geometry/ellipse.py
+++ b/sympy/geometry/ellipse.py
@@ -10,7 +10,7 @@
 
 from sympy import Expr, Eq
 from sympy.core import S, pi, sympify
-from sympy.core.evaluate import global_evaluate
+from sympy.core.parameters import global_parameters
 from sympy.core.logic import fuzzy_bool
 from sympy.core.numbers import Rational, oo
 from sympy.core.compatibility import ordered
@@ -1546,7 +1546,7 @@ class Circle(Ellipse):
     def __new__(cls, *args, **kwargs):
         from sympy.geometry.util import find
         from .polygon import Triangle
-        evaluate = kwargs.get('evaluate', global_evaluate[0])
+        evaluate = kwargs.get('evaluate', global_parameters.evaluate)
         if len(args) == 1 and isinstance(args[0], (Expr, Eq)):
             x = kwargs.get('x', 'x')
             y = kwargs.get('y', 'y')
diff --git a/sympy/geometry/point.py b/sympy/geometry/point.py
--- a/sympy/geometry/point.py
+++ b/sympy/geometry/point.py
@@ -30,7 +30,7 @@
 from sympy.functions.elementary.complexes import im
 from sympy.matrices import Matrix
 from sympy.core.numbers import Float
-from sympy.core.evaluate import global_evaluate
+from sympy.core.parameters import global_parameters
 from sympy.core.add import Add
 from sympy.utilities.iterables import uniq
 from sympy.utilities.misc import filldedent, func_name, Undecidable
@@ -106,7 +106,7 @@ class Point(GeometryEntity):
     is_Point = True
 
     def __new__(cls, *args, **kwargs):
-        evaluate = kwargs.get('evaluate', global_evaluate[0])
+        evaluate = kwargs.get('evaluate', global_parameters.evaluate)
         on_morph = kwargs.get('on_morph', 'ignore')
 
         # unpack into coords
diff --git a/sympy/series/sequences.py b/sympy/series/sequences.py
--- a/sympy/series/sequences.py
+++ b/sympy/series/sequences.py
@@ -6,7 +6,7 @@
                                       is_sequence, iterable, ordered)
 from sympy.core.containers import Tuple
 from sympy.core.decorators import call_highest_priority
-from sympy.core.evaluate import global_evaluate
+from sympy.core.parameters import global_parameters
 from sympy.core.function import UndefinedFunction
 from sympy.core.mul import Mul
 from sympy.core.numbers import Integer
@@ -1005,7 +1005,7 @@ class SeqAdd(SeqExprOp):
     """
 
     def __new__(cls, *args, **kwargs):
-        evaluate = kwargs.get('evaluate', global_evaluate[0])
+        evaluate = kwargs.get('evaluate', global_parameters.evaluate)
 
         # flatten inputs
         args = list(args)
@@ -1114,7 +1114,7 @@ class SeqMul(SeqExprOp):
     """
 
     def __new__(cls, *args, **kwargs):
-        evaluate = kwargs.get('evaluate', global_evaluate[0])
+        evaluate = kwargs.get('evaluate', global_parameters.evaluate)
 
         # flatten inputs
         args = list(args)
diff --git a/sympy/sets/powerset.py b/sympy/sets/powerset.py
--- a/sympy/sets/powerset.py
+++ b/sympy/sets/powerset.py
@@ -1,7 +1,7 @@
 from __future__ import print_function, division
 
 from sympy.core.decorators import _sympifyit
-from sympy.core.evaluate import global_evaluate
+from sympy.core.parameters import global_parameters
 from sympy.core.logic import fuzzy_bool
 from sympy.core.singleton import S
 from sympy.core.sympify import _sympify
@@ -71,7 +71,10 @@ class PowerSet(Set):
 
     .. [2] https://en.wikipedia.org/wiki/Axiom_of_power_set
     """
-    def __new__(cls, arg, evaluate=global_evaluate[0]):
+    def __new__(cls, arg, evaluate=None):
+        if evaluate is None:
+            evaluate=global_parameters.evaluate
+
         arg = _sympify(arg)
 
         if not isinstance(arg, Set):
diff --git a/sympy/sets/sets.py b/sympy/sets/sets.py
--- a/sympy/sets/sets.py
+++ b/sympy/sets/sets.py
@@ -11,7 +11,7 @@
 from sympy.core.decorators import (deprecated, sympify_method_args,
     sympify_return)
 from sympy.core.evalf import EvalfMixin
-from sympy.core.evaluate import global_evaluate
+from sympy.core.parameters import global_parameters
 from sympy.core.expr import Expr
 from sympy.core.logic import fuzzy_bool, fuzzy_or, fuzzy_and, fuzzy_not
 from sympy.core.numbers import Float
@@ -1172,7 +1172,7 @@ def zero(self):
         return S.UniversalSet
 
     def __new__(cls, *args, **kwargs):
-        evaluate = kwargs.get('evaluate', global_evaluate[0])
+        evaluate = kwargs.get('evaluate', global_parameters.evaluate)
 
         # flatten inputs to merge intersections and iterables
         args = _sympify(args)
@@ -1345,7 +1345,7 @@ def zero(self):
         return S.EmptySet
 
     def __new__(cls, *args, **kwargs):
-        evaluate = kwargs.get('evaluate', global_evaluate[0])
+        evaluate = kwargs.get('evaluate', global_parameters.evaluate)
 
         # flatten inputs to merge intersections and iterables
         args = list(ordered(set(_sympify(args))))
@@ -1767,7 +1767,7 @@ class FiniteSet(Set, EvalfMixin):
     is_finite_set = True
 
     def __new__(cls, *args, **kwargs):
-        evaluate = kwargs.get('evaluate', global_evaluate[0])
+        evaluate = kwargs.get('evaluate', global_parameters.evaluate)
         if evaluate:
             args = list(map(sympify, args))
 
diff --git a/sympy/simplify/radsimp.py b/sympy/simplify/radsimp.py
--- a/sympy/simplify/radsimp.py
+++ b/sympy/simplify/radsimp.py
@@ -7,7 +7,7 @@
 from sympy.core import expand_power_base, sympify, Add, S, Mul, Derivative, Pow, symbols, expand_mul
 from sympy.core.add import _unevaluated_Add
 from sympy.core.compatibility import iterable, ordered, default_sort_key
-from sympy.core.evaluate import global_evaluate
+from sympy.core.parameters import global_parameters
 from sympy.core.exprtools import Factors, gcd_terms
 from sympy.core.function import _mexpand
 from sympy.core.mul import _keep_coeff, _unevaluated_Mul
@@ -163,7 +163,7 @@ def collect(expr, syms, func=None, evaluate=None, exact=False, distribute_order_
     syms = list(syms) if iterable(syms) else [syms]
 
     if evaluate is None:
-        evaluate = global_evaluate[0]
+        evaluate = global_parameters.evaluate
 
     def make_expression(terms):
         product = []
@@ -496,7 +496,7 @@ def collect_sqrt(expr, evaluate=None):
     collect, collect_const, rcollect
     """
     if evaluate is None:
-        evaluate = global_evaluate[0]
+        evaluate = global_parameters.evaluate
     # this step will help to standardize any complex arguments
     # of sqrts
     coeff, expr = expr.as_content_primitive()
diff --git a/sympy/simplify/simplify.py b/sympy/simplify/simplify.py
--- a/sympy/simplify/simplify.py
+++ b/sympy/simplify/simplify.py
@@ -6,7 +6,7 @@
                         expand_func, Function, Dummy, Expr, factor_terms,
                         expand_power_exp, Eq)
 from sympy.core.compatibility import iterable, ordered, range, as_int
-from sympy.core.evaluate import global_evaluate
+from sympy.core.parameters import global_parameters
 from sympy.core.function import expand_log, count_ops, _mexpand, _coeff_isneg, \
     nfloat, expand_mul, expand_multinomial
 from sympy.core.numbers import Float, I, pi, Rational, Integer
@@ -378,7 +378,7 @@ def signsimp(expr, evaluate=None):
 
     """
     if evaluate is None:
-        evaluate = global_evaluate[0]
+        evaluate = global_parameters.evaluate
     expr = sympify(expr)
     if not isinstance(expr, (Expr, Relational)) or expr.is_Atom:
         return expr
diff --git a/sympy/stats/symbolic_probability.py b/sympy/stats/symbolic_probability.py
--- a/sympy/stats/symbolic_probability.py
+++ b/sympy/stats/symbolic_probability.py
@@ -2,7 +2,7 @@
 
 from sympy import Expr, Add, Mul, S, Integral, Eq, Sum, Symbol
 from sympy.core.compatibility import default_sort_key
-from sympy.core.evaluate import global_evaluate
+from sympy.core.parameters import global_parameters
 from sympy.core.sympify import _sympify
 from sympy.stats import variance, covariance
 from sympy.stats.rv import RandomSymbol, probability, expectation
@@ -324,7 +324,7 @@ def __new__(cls, arg1, arg2, condition=None, **kwargs):
         arg1 = _sympify(arg1)
         arg2 = _sympify(arg2)
 
-        if kwargs.pop('evaluate', global_evaluate[0]):
+        if kwargs.pop('evaluate', global_parameters.evaluate):
             arg1, arg2 = sorted([arg1, arg2], key=default_sort_key)
 
         if condition is None:
diff --git a/sympy/tensor/functions.py b/sympy/tensor/functions.py
--- a/sympy/tensor/functions.py
+++ b/sympy/tensor/functions.py
@@ -1,6 +1,6 @@
 from sympy import Expr, S, Mul, sympify
 from sympy.core.compatibility import Iterable
-from sympy.core.evaluate import global_evaluate
+from sympy.core.parameters import global_parameters
 
 
 class TensorProduct(Expr):
@@ -15,7 +15,7 @@ def __new__(cls, *args, **kwargs):
         from sympy.strategies import flatten
 
         args = [sympify(arg) for arg in args]
-        evaluate = kwargs.get("evaluate", global_evaluate[0])
+        evaluate = kwargs.get("evaluate", global_parameters.evaluate)
 
         if not evaluate:
             obj = Expr.__new__(cls, *args)

```

## Test Patch

```diff
diff --git a/sympy/core/tests/test_arit.py b/sympy/core/tests/test_arit.py
--- a/sympy/core/tests/test_arit.py
+++ b/sympy/core/tests/test_arit.py
@@ -3,7 +3,7 @@
         oo, zoo, Integer, sign, im, nan, Dummy, factorial, comp, floor
 )
 from sympy.core.compatibility import long, range
-from sympy.core.evaluate import distribute
+from sympy.core.parameters import distribute
 from sympy.core.expr import unchanged
 from sympy.utilities.iterables import cartes
 from sympy.utilities.pytest import XFAIL, raises
diff --git a/sympy/core/tests/test_evaluate.py b/sympy/core/tests/test_parameters.py
similarity index 98%
rename from sympy/core/tests/test_evaluate.py
rename to sympy/core/tests/test_parameters.py
--- a/sympy/core/tests/test_evaluate.py
+++ b/sympy/core/tests/test_parameters.py
@@ -1,5 +1,5 @@
 from sympy.abc import x, y
-from sympy.core.evaluate import evaluate
+from sympy.core.parameters import evaluate
 from sympy.core import Mul, Add, Pow, S
 from sympy import sqrt, oo
 

```


## Code snippets

### 1 - sympy/core/evaluate.py:

Start line: 1, End line: 45

```python
from .cache import clear_cache
from contextlib import contextmanager


class _global_function(list):
    """ The cache must be cleared whenever _global_function is changed. """

    def __setitem__(self, key, value):
        if (self[key] != value):
            clear_cache()
        super(_global_function, self).__setitem__(key, value)


global_evaluate = _global_function([True])
global_distribute = _global_function([True])


@contextmanager
def evaluate(x):
    """ Control automatic evaluation

    This context manager controls whether or not all SymPy functions evaluate
    by default.

    Note that much of SymPy expects evaluated expressions.  This functionality
    is experimental and is unlikely to function as intended on large
    expressions.

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy.core.evaluate import evaluate
    >>> print(x + x)
    2*x
    >>> with evaluate(False):
    ...     print(x + x)
    x + x
    """

    old = global_evaluate[0]

    global_evaluate[0] = x
    yield
    global_evaluate[0] = old
```
### 2 - sympy/core/function.py:

Start line: 2453, End line: 2778

```python
def expand(e, deep=True, modulus=None, power_base=True, power_exp=True,
        mul=True, log=True, multinomial=True, basic=True, **hints):
    r"""
    Expand an expression using methods given as hints.

    Hints evaluated unless explicitly set to False are:  ``basic``, ``log``,
    ``multinomial``, ``mul``, ``power_base``, and ``power_exp`` The following
    hints are supported but not applied unless set to True:  ``complex``,
    ``func``, and ``trig``.  In addition, the following meta-hints are
    supported by some or all of the other hints:  ``frac``, ``numer``,
    ``denom``, ``modulus``, and ``force``.  ``deep`` is supported by all
    hints.  Additionally, subclasses of Expr may define their own hints or
    meta-hints.

    The ``basic`` hint is used for any special rewriting of an object that
    should be done automatically (along with the other hints like ``mul``)
    when expand is called. This is a catch-all hint to handle any sort of
    expansion that may not be described by the existing hint names. To use
    this hint an object should override the ``_eval_expand_basic`` method.
    Objects may also define their own expand methods, which are not run by
    default.  See the API section below.

    If ``deep`` is set to ``True`` (the default), things like arguments of
    functions are recursively expanded.  Use ``deep=False`` to only expand on
    the top level.

    If the ``force`` hint is used, assumptions about variables will be ignored
    in making the expansion.

    Hints
    =====

    These hints are run by default

    mul
    ---

    Distributes multiplication over addition:

    >>> from sympy import cos, exp, sin
    >>> from sympy.abc import x, y, z
    >>> (y*(x + z)).expand(mul=True)
    x*y + y*z

    multinomial
    -----------

    Expand (x + y + ...)**n where n is a positive integer.

    >>> ((x + y + z)**2).expand(multinomial=True)
    x**2 + 2*x*y + 2*x*z + y**2 + 2*y*z + z**2

    power_exp
    ---------

    Expand addition in exponents into multiplied bases.

    >>> exp(x + y).expand(power_exp=True)
    exp(x)*exp(y)
    >>> (2**(x + y)).expand(power_exp=True)
    2**x*2**y

    power_base
    ----------

    Split powers of multiplied bases.

    This only happens by default if assumptions allow, or if the
    ``force`` meta-hint is used:

    >>> ((x*y)**z).expand(power_base=True)
    (x*y)**z
    >>> ((x*y)**z).expand(power_base=True, force=True)
    x**z*y**z
    >>> ((2*y)**z).expand(power_base=True)
    2**z*y**z

    Note that in some cases where this expansion always holds, SymPy performs
    it automatically:

    >>> (x*y)**2
    x**2*y**2

    log
    ---

    Pull out power of an argument as a coefficient and split logs products
    into sums of logs.

    Note that these only work if the arguments of the log function have the
    proper assumptions--the arguments must be positive and the exponents must
    be real--or else the ``force`` hint must be True:

    >>> from sympy import log, symbols
    >>> log(x**2*y).expand(log=True)
    log(x**2*y)
    >>> log(x**2*y).expand(log=True, force=True)
    2*log(x) + log(y)
    >>> x, y = symbols('x,y', positive=True)
    >>> log(x**2*y).expand(log=True)
    2*log(x) + log(y)

    basic
    -----

    This hint is intended primarily as a way for custom subclasses to enable
    expansion by default.

    These hints are not run by default:

    complex
    -------

    Split an expression into real and imaginary parts.

    >>> x, y = symbols('x,y')
    >>> (x + y).expand(complex=True)
    re(x) + re(y) + I*im(x) + I*im(y)
    >>> cos(x).expand(complex=True)
    -I*sin(re(x))*sinh(im(x)) + cos(re(x))*cosh(im(x))

    Note that this is just a wrapper around ``as_real_imag()``.  Most objects
    that wish to redefine ``_eval_expand_complex()`` should consider
    redefining ``as_real_imag()`` instead.

    func
    ----

    Expand other functions.

    >>> from sympy import gamma
    >>> gamma(x + 1).expand(func=True)
    x*gamma(x)

    trig
    ----

    Do trigonometric expansions.

    >>> cos(x + y).expand(trig=True)
    -sin(x)*sin(y) + cos(x)*cos(y)
    >>> sin(2*x).expand(trig=True)
    2*sin(x)*cos(x)

    Note that the forms of ``sin(n*x)`` and ``cos(n*x)`` in terms of ``sin(x)``
    and ``cos(x)`` are not unique, due to the identity `\sin^2(x) + \cos^2(x)
    = 1`.  The current implementation uses the form obtained from Chebyshev
    polynomials, but this may change.  See `this MathWorld article
    <http://mathworld.wolfram.com/Multiple-AngleFormulas.html>`_ for more
    information.

    Notes
    =====

    - You can shut off unwanted methods::

        >>> (exp(x + y)*(x + y)).expand()
        x*exp(x)*exp(y) + y*exp(x)*exp(y)
        >>> (exp(x + y)*(x + y)).expand(power_exp=False)
        x*exp(x + y) + y*exp(x + y)
        >>> (exp(x + y)*(x + y)).expand(mul=False)
        (x + y)*exp(x)*exp(y)

    - Use deep=False to only expand on the top level::

        >>> exp(x + exp(x + y)).expand()
        exp(x)*exp(exp(x)*exp(y))
        >>> exp(x + exp(x + y)).expand(deep=False)
        exp(x)*exp(exp(x + y))

    - Hints are applied in an arbitrary, but consistent order (in the current
      implementation, they are applied in alphabetical order, except
      multinomial comes before mul, but this may change).  Because of this,
      some hints may prevent expansion by other hints if they are applied
      first. For example, ``mul`` may distribute multiplications and prevent
      ``log`` and ``power_base`` from expanding them. Also, if ``mul`` is
      applied before ``multinomial`, the expression might not be fully
      distributed. The solution is to use the various ``expand_hint`` helper
      functions or to use ``hint=False`` to this function to finely control
      which hints are applied. Here are some examples::

        >>> from sympy import expand, expand_mul, expand_power_base
        >>> x, y, z = symbols('x,y,z', positive=True)

        >>> expand(log(x*(y + z)))
        log(x) + log(y + z)

      Here, we see that ``log`` was applied before ``mul``.  To get the mul
      expanded form, either of the following will work::

        >>> expand_mul(log(x*(y + z)))
        log(x*y + x*z)
        >>> expand(log(x*(y + z)), log=False)
        log(x*y + x*z)

      A similar thing can happen with the ``power_base`` hint::

        >>> expand((x*(y + z))**x)
        (x*y + x*z)**x

      To get the ``power_base`` expanded form, either of the following will
      work::

        >>> expand((x*(y + z))**x, mul=False)
        x**x*(y + z)**x
        >>> expand_power_base((x*(y + z))**x)
        x**x*(y + z)**x

        >>> expand((x + y)*y/x)
        y + y**2/x

      The parts of a rational expression can be targeted::

        >>> expand((x + y)*y/x/(x + 1), frac=True)
        (x*y + y**2)/(x**2 + x)
        >>> expand((x + y)*y/x/(x + 1), numer=True)
        (x*y + y**2)/(x*(x + 1))
        >>> expand((x + y)*y/x/(x + 1), denom=True)
        y*(x + y)/(x**2 + x)

    - The ``modulus`` meta-hint can be used to reduce the coefficients of an
      expression post-expansion::

        >>> expand((3*x + 1)**2)
        9*x**2 + 6*x + 1
        >>> expand((3*x + 1)**2, modulus=5)
        4*x**2 + x + 1

    - Either ``expand()`` the function or ``.expand()`` the method can be
      used.  Both are equivalent::

        >>> expand((x + 1)**2)
        x**2 + 2*x + 1
        >>> ((x + 1)**2).expand()
        x**2 + 2*x + 1

    API
    ===

    Objects can define their own expand hints by defining
    ``_eval_expand_hint()``.  The function should take the form::

        def _eval_expand_hint(self, **hints):
            # Only apply the method to the top-level expression
            ...

    See also the example below.  Objects should define ``_eval_expand_hint()``
    methods only if ``hint`` applies to that specific object.  The generic
    ``_eval_expand_hint()`` method defined in Expr will handle the no-op case.

    Each hint should be responsible for expanding that hint only.
    Furthermore, the expansion should be applied to the top-level expression
    only.  ``expand()`` takes care of the recursion that happens when
    ``deep=True``.

    You should only call ``_eval_expand_hint()`` methods directly if you are
    100% sure that the object has the method, as otherwise you are liable to
    get unexpected ``AttributeError``s.  Note, again, that you do not need to
    recursively apply the hint to args of your object: this is handled
    automatically by ``expand()``.  ``_eval_expand_hint()`` should
    generally not be used at all outside of an ``_eval_expand_hint()`` method.
    If you want to apply a specific expansion from within another method, use
    the public ``expand()`` function, method, or ``expand_hint()`` functions.

    In order for expand to work, objects must be rebuildable by their args,
    i.e., ``obj.func(*obj.args) == obj`` must hold.

    Expand methods are passed ``**hints`` so that expand hints may use
    'metahints'--hints that control how different expand methods are applied.
    For example, the ``force=True`` hint described above that causes
    ``expand(log=True)`` to ignore assumptions is such a metahint.  The
    ``deep`` meta-hint is handled exclusively by ``expand()`` and is not
    passed to ``_eval_expand_hint()`` methods.

    Note that expansion hints should generally be methods that perform some
    kind of 'expansion'.  For hints that simply rewrite an expression, use the
    .rewrite() API.

    Examples
    ========

    >>> from sympy import Expr, sympify
    >>> class MyClass(Expr):
    ...     def __new__(cls, *args):
    ...         args = sympify(args)
    ...         return Expr.__new__(cls, *args)
    ...
    ...     def _eval_expand_double(self, **hints):
    ...         '''
    ...         Doubles the args of MyClass.
    ...
    ...         If there more than four args, doubling is not performed,
    ...         unless force=True is also used (False by default).
    ...         '''
    ...         force = hints.pop('force', False)
    ...         if not force and len(self.args) > 4:
    ...             return self
    ...         return self.func(*(self.args + self.args))
    ...
    >>> a = MyClass(1, 2, MyClass(3, 4))
    >>> a
    MyClass(1, 2, MyClass(3, 4))
    >>> a.expand(double=True)
    MyClass(1, 2, MyClass(3, 4, 3, 4), 1, 2, MyClass(3, 4, 3, 4))
    >>> a.expand(double=True, deep=False)
    MyClass(1, 2, MyClass(3, 4), 1, 2, MyClass(3, 4))

    >>> b = MyClass(1, 2, 3, 4, 5)
    >>> b.expand(double=True)
    MyClass(1, 2, 3, 4, 5)
    >>> b.expand(double=True, force=True)
    MyClass(1, 2, 3, 4, 5, 1, 2, 3, 4, 5)

    See Also
    ========

    expand_log, expand_mul, expand_multinomial, expand_complex, expand_trig,
    expand_power_base, expand_power_exp, expand_func, sympy.simplify.hyperexpand.hyperexpand

    """
    # don't modify this; modify the Expr.expand method
    hints['power_base'] = power_base
    hints['power_exp'] = power_exp
    hints['mul'] = mul
    hints['log'] = log
    hints['multinomial'] = multinomial
    hints['basic'] = basic
    return sympify(e).expand(deep=deep, modulus=modulus, **hints)
```
### 3 - sympy/plotting/experimental_lambdify.py:

Start line: 1, End line: 76

```python
""" rewrite of lambdify - This stuff is not stable at all.

It is for internal use in the new plotting module.
It may (will! see the Q'n'A in the source) be rewritten.

It's completely self contained. Especially it does not use lambdarepr.

It does not aim to replace the current lambdify. Most importantly it will never
ever support anything else than sympy expressions (no Matrices, dictionaries
and so on).
"""

from __future__ import print_function, division

import re
from sympy import Symbol, NumberSymbol, I, zoo, oo
from sympy.core.compatibility import exec_, string_types
from sympy.utilities.iterables import numbered_symbols

#  We parse the expression string into a tree that identifies functions. Then
# we translate the names of the functions and we translate also some strings
# that are not names of functions (all this according to translation
# dictionaries).
#  If the translation goes to another module (like numpy) the
# module is imported and 'func' is translated to 'module.func'.
#  If a function can not be translated, the inner nodes of that part of the
# tree are not translated. So if we have Integral(sqrt(x)), sqrt is not
# translated to np.sqrt and the Integral does not crash.
#  A namespace for all this is generated by crawling the (func, args) tree of
# the expression. The creation of this namespace involves many ugly
# workarounds.
#  The namespace consists of all the names needed for the sympy expression and
# all the name of modules used for translation. Those modules are imported only
# as a name (import numpy as np) in order to keep the namespace small and
# manageable.

#  Please, if there is a bug, do not try to fix it here! Rewrite this by using
# the method proposed in the last Q'n'A below. That way the new function will
# work just as well, be just as simple, but it wont need any new workarounds.
#  If you insist on fixing it here, look at the workarounds in the function
# sympy_expression_namespace and in lambdify.

# Q: Why are you not using python abstract syntax tree?
# A: Because it is more complicated and not much more powerful in this case.

# Q: What if I have Symbol('sin') or g=Function('f')?
# A: You will break the algorithm. We should use srepr to defend against this?
#  The problem with Symbol('sin') is that it will be printed as 'sin'. The
# parser will distinguish it from the function 'sin' because functions are
# detected thanks to the opening parenthesis, but the lambda expression won't
# understand the difference if we have also the sin function.
# The solution (complicated) is to use srepr and maybe ast.
#  The problem with the g=Function('f') is that it will be printed as 'f' but in
# the global namespace we have only 'g'. But as the same printer is used in the
# constructor of the namespace there will be no problem.

# Q: What if some of the printers are not printing as expected?
# A: The algorithm wont work. You must use srepr for those cases. But even
# srepr may not print well. All problems with printers should be considered
# bugs.

# Q: What about _imp_ functions?
# A: Those are taken care for by evalf. A special case treatment will work
# faster but it's not worth the code complexity.

# Q: Will ast fix all possible problems?
# A: No. You will always have to use some printer. Even srepr may not work in
# some cases. But if the printer does not work, that should be considered a
# bug.

# Q: Is there same way to fix all possible problems?
# A: Probably by constructing our strings ourself by traversing the (func,
# args) tree and creating the namespace at the same time. That actually sounds
# good.

from sympy.external import import_module
```
### 4 - sympy/core/function.py:

Start line: 2269, End line: 2315

```python
class Subs(Expr):

    def evalf(self, prec=None, **options):
        return self.doit().evalf(prec, **options)

    n = evalf

    @property
    def variables(self):
        """The variables to be evaluated"""
        return self._args[1]

    bound_symbols = variables

    @property
    def expr(self):
        """The expression on which the substitution operates"""
        return self._args[0]

    @property
    def point(self):
        """The values for which the variables are to be substituted"""
        return self._args[2]

    @property
    def free_symbols(self):
        return (self.expr.free_symbols - set(self.variables) |
            set(self.point.free_symbols))

    @property
    def expr_free_symbols(self):
        return (self.expr.expr_free_symbols - set(self.variables) |
            set(self.point.expr_free_symbols))

    def __eq__(self, other):
        if not isinstance(other, Subs):
            return False
        return self._hashable_content() == other._hashable_content()

    def __ne__(self, other):
        return not(self == other)

    def __hash__(self):
        return super(Subs, self).__hash__()

    def _hashable_content(self):
        return (self._expr.xreplace(self.canonical_variables),
            ) + tuple(ordered([(v, p) for v, p in
            zip(self.variables, self.point) if not self.expr.has(v)]))
```
### 5 - sympy/functions/special/hyper.py:

Start line: 185, End line: 211

```python
class hyper(TupleParametersBase):


    def __new__(cls, ap, bq, z, **kwargs):
        # TODO should we check convergence conditions?
        return Function.__new__(cls, _prep_tuple(ap), _prep_tuple(bq), z, **kwargs)

    @classmethod
    def eval(cls, ap, bq, z):
        from sympy import unpolarify
        if len(ap) <= len(bq) or (len(ap) == len(bq) + 1 and (Abs(z) <= 1) == True):
            nz = unpolarify(z)
            if z != nz:
                return hyper(ap, bq, nz)

    def fdiff(self, argindex=3):
        if argindex != 3:
            raise ArgumentIndexError(self, argindex)
        nap = Tuple(*[a + 1 for a in self.ap])
        nbq = Tuple(*[b + 1 for b in self.bq])
        fac = Mul(*self.ap)/Mul(*self.bq)
        return fac*hyper(nap, nbq, self.argument)

    def _eval_expand_func(self, **hints):
        from sympy import gamma, hyperexpand
        if len(self.ap) == 2 and len(self.bq) == 1 and self.argument == 1:
            a, b = self.ap
            c = self.bq[0]
            return gamma(c)*gamma(c - a - b)/gamma(c - a)/gamma(c - b)
        return hyperexpand(self)
```
### 6 - sympy/core/evaluate.py:

Start line: 48, End line: 73

```python
@contextmanager
def distribute(x):
    """ Control automatic distribution of Number over Add

    This context manager controls whether or not Mul distribute Number over
    Add. Plan is to avoid distributing Number over Add in all of sympy. Once
    that is done, this contextmanager will be removed.

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy.core.evaluate import distribute
    >>> print(2*(x + 1))
    2*x + 2
    >>> with distribute(False):
    ...     print(2*(x + 1))
    2*(x + 1)
    """

    old = global_distribute[0]

    global_distribute[0] = x
    yield
    global_distribute[0] = old
```
### 7 - sympy/core/mod.py:

Start line: 165, End line: 209

```python
class Mod(Function):

    @classmethod
    def eval(cls, p, q):
        # ... other code
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
```
### 8 - sympy/core/sympify.py:

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
### 9 - sympy/utilities/lambdify.py:

Start line: 107, End line: 825

```python
def _import(module, reload=False):
    """
    Creates a global translation dictionary for module.

    The argument module has to be one of the following strings: "math",
    "mpmath", "numpy", "sympy", "tensorflow".
    These dictionaries map names of python functions to their equivalent in
    other modules.
    """
    # Required despite static analysis claiming it is not used
    from sympy.external import import_module
    try:
        namespace, namespace_default, translations, import_commands = MODULES[
            module]
    except KeyError:
        raise NameError(
            "'%s' module can't be used for lambdification" % module)

    # Clear namespace or exit
    if namespace != namespace_default:
        # The namespace was already generated, don't do it again if not forced.
        if reload:
            namespace.clear()
            namespace.update(namespace_default)
        else:
            return

    for import_command in import_commands:
        if import_command.startswith('import_module'):
            module = eval(import_command)

            if module is not None:
                namespace.update(module.__dict__)
                continue
        else:
            try:
                exec_(import_command, {}, namespace)
                continue
            except ImportError:
                pass

        raise ImportError(
            "can't import '%s' with '%s' command" % (module, import_command))

    # Add translated names to namespace
    for sympyname, translation in translations.items():
        namespace[sympyname] = namespace[translation]

    # For computing the modulus of a sympy expression we use the builtin abs
    # function, instead of the previously used fabs function for all
    # translation modules. This is because the fabs function in the math
    # module does not accept complex valued arguments. (see issue 9474). The
    # only exception, where we don't use the builtin abs function is the
    # mpmath translation module, because mpmath.fabs returns mpf objects in
    # contrast to abs().
    if 'Abs' not in namespace:
        namespace['Abs'] = abs


# Used for dynamically generated filenames that are inserted into the
# linecache.
_lambdify_generated_counter = 1

@doctest_depends_on(modules=('numpy', 'tensorflow', ), python_version=(3,))
def lambdify(args, expr, modules=None, printer=None, use_imps=True,
             dummify=False):
    # ... other code
```
### 10 - sympy/core/basic.py:

Start line: 892, End line: 976

```python
class Basic(with_metaclass(ManagedProperties)):

    def subs(self, *args, **kwargs):
        from sympy.core.containers import Dict
        from sympy.utilities import default_sort_key
        from sympy import Dummy, Symbol

        unordered = False
        if len(args) == 1:
            sequence = args[0]
            if isinstance(sequence, set):
                unordered = True
            elif isinstance(sequence, (Dict, Mapping)):
                unordered = True
                sequence = sequence.items()
            elif not iterable(sequence):
                from sympy.utilities.misc import filldedent
                raise ValueError(filldedent("""
                   When a single argument is passed to subs
                   it should be a dictionary of old: new pairs or an iterable
                   of (old, new) tuples."""))
        elif len(args) == 2:
            sequence = [args]
        else:
            raise ValueError("subs accepts either 1 or 2 arguments")

        sequence = list(sequence)
        for i, s in enumerate(sequence):
            if isinstance(s[0], string_types):
                # when old is a string we prefer Symbol
                s = Symbol(s[0]), s[1]
            try:
                s = [sympify(_, strict=not isinstance(_, string_types))
                     for _ in s]
            except SympifyError:
                # if it can't be sympified, skip it
                sequence[i] = None
                continue
            # skip if there is no change
            sequence[i] = None if _aresame(*s) else tuple(s)
        sequence = list(filter(None, sequence))

        if unordered:
            sequence = dict(sequence)
            if not all(k.is_Atom for k in sequence):
                d = {}
                for o, n in sequence.items():
                    try:
                        ops = o.count_ops(), len(o.args)
                    except TypeError:
                        ops = (0, 0)
                    d.setdefault(ops, []).append((o, n))
                newseq = []
                for k in sorted(d.keys(), reverse=True):
                    newseq.extend(
                        sorted([v[0] for v in d[k]], key=default_sort_key))
                sequence = [(k, sequence[k]) for k in newseq]
                del newseq, d
            else:
                sequence = sorted([(k, v) for (k, v) in sequence.items()],
                                  key=default_sort_key)

        if kwargs.pop('simultaneous', False):  # XXX should this be the default for dict subs?
            reps = {}
            rv = self
            kwargs['hack2'] = True
            m = Dummy('subs_m')
            for old, new in sequence:
                com = new.is_commutative
                if com is None:
                    com = True
                d = Dummy('subs_d', commutative=com)
                # using d*m so Subs will be used on dummy variables
                # in things like Derivative(f(x, y), x) in which x
                # is both free and bound
                rv = rv._subs(old, d*m, **kwargs)
                if not isinstance(rv, Basic):
                    break
                reps[d] = new
            reps[m] = S.One  # get rid of m
            return rv.xreplace(reps)
        else:
            rv = self
            for old, new in sequence:
                rv = rv._subs(old, new, **kwargs)
                if not isinstance(rv, Basic):
                    break
            return rv
```
### 12 - sympy/core/function.py:

Start line: 1, End line: 60

```python
"""
There are three types of functions implemented in SymPy:

    1) defined functions (in the sense that they can be evaluated) like
       exp or sin; they have a name and a body:
           f = exp
    2) undefined function which have a name but no body. Undefined
       functions can be defined using a Function class as follows:
           f = Function('f')
       (the result will be a Function instance)
    3) anonymous function (or lambda function) which have a body (defined
       with dummy variables) but have no name:
           f = Lambda(x, exp(x)*x)
           f = Lambda((x, y), exp(x)*y)
    The fourth type of functions are composites, like (sin + cos)(x); these work in
    SymPy core, but are not yet part of SymPy.

    Examples
    ========

    >>> import sympy
    >>> f = sympy.Function("f")
    >>> from sympy.abc import x
    >>> f(x)
    f(x)
    >>> print(sympy.srepr(f(x).func))
    Function('f')
    >>> f(x).args
    (x,)

"""
from __future__ import print_function, division

from .add import Add
from .assumptions import ManagedProperties
from .basic import Basic, _atomic
from .cache import cacheit
from .compatibility import iterable, is_sequence, as_int, ordered, Iterable
from .decorators import _sympifyit
from .expr import Expr, AtomicExpr
from .numbers import Rational, Float
from .operations import LatticeOp
from .rules import Transform
from .singleton import S
from .sympify import sympify

from sympy.core.compatibility import string_types, with_metaclass, PY3, range
from sympy.core.containers import Tuple, Dict
from sympy.core.evaluate import global_evaluate
from sympy.core.logic import fuzzy_and
from sympy.utilities import default_sort_key
from sympy.utilities.exceptions import SymPyDeprecationWarning
from sympy.utilities.iterables import has_dups, sift
from sympy.utilities.misc import filldedent

import mpmath
import mpmath.libmp as mlib

import inspect
from collections import Counter
```
### 13 - sympy/core/function.py:

Start line: 450, End line: 480

```python
class Function(Application, Expr):

    @cacheit
    def __new__(cls, *args, **options):
        # Handle calls like Function('f')
        if cls is Function:
            return UndefinedFunction(*args, **options)

        n = len(args)
        if n not in cls.nargs:
            # XXX: exception message must be in exactly this format to
            # make it work with NumPy's functions like vectorize(). See,
            # for example, https://github.com/numpy/numpy/issues/1697.
            # The ideal solution would be just to attach metadata to
            # the exception and change NumPy to take advantage of this.
            temp = ('%(name)s takes %(qual)s %(args)s '
                   'argument%(plural)s (%(given)s given)')
            raise TypeError(temp % {
                'name': cls,
                'qual': 'exactly' if len(cls.nargs) == 1 else 'at least',
                'args': min(cls.nargs),
                'plural': 's'*(min(cls.nargs) != 1),
                'given': n})

        evaluate = options.get('evaluate', global_evaluate[0])
        result = super(Function, cls).__new__(cls, *args, **options)
        if evaluate and isinstance(result, cls) and result.args:
            pr2 = min(cls._should_evalf(a) for a in result.args)
            if pr2 > 0:
                pr = max(cls._should_evalf(a) for a in result.args)
                result = result.evalf(mlib.libmpf.prec_to_dps(pr))

        return result
```
### 17 - sympy/core/add.py:

Start line: 1, End line: 22

```python
from __future__ import print_function, division

from collections import defaultdict
from functools import cmp_to_key

from .basic import Basic
from .compatibility import reduce, is_sequence, range
from .evaluate import global_distribute
from .logic import _fuzzy_group, fuzzy_or, fuzzy_not
from .singleton import S
from .operations import AssocOp
from .cache import cacheit
from .numbers import ilcm, igcd
from .expr import Expr

# Key for sorting commutative args in canonical order
_args_sortkey = cmp_to_key(Basic.compare)


def _addsort(args):
    # in-place sorting of args
    args.sort(key=_args_sortkey)
```
### 20 - sympy/core/__init__.py:

Start line: 1, End line: 36

```python
"""Core module. Provides the basic operations needed in sympy.
"""

from .sympify import sympify, SympifyError
from .cache import cacheit
from .basic import Basic, Atom, preorder_traversal
from .singleton import S
from .expr import Expr, AtomicExpr, UnevaluatedExpr
from .symbol import Symbol, Wild, Dummy, symbols, var
from .numbers import Number, Float, Rational, Integer, NumberSymbol, \
    RealNumber, igcd, ilcm, seterr, E, I, nan, oo, pi, zoo, \
    AlgebraicNumber, comp, mod_inverse
from .power import Pow, integer_nthroot, integer_log
from .mul import Mul, prod
from .add import Add
from .mod import Mod
from .relational import ( Rel, Eq, Ne, Lt, Le, Gt, Ge,
    Equality, GreaterThan, LessThan, Unequality, StrictGreaterThan,
    StrictLessThan )
from .multidimensional import vectorize
from .function import Lambda, WildFunction, Derivative, diff, FunctionClass, \
    Function, Subs, expand, PoleError, count_ops, \
    expand_mul, expand_log, expand_func, \
    expand_trig, expand_complex, expand_multinomial, nfloat, \
    expand_power_base, expand_power_exp, arity
from .evalf import PrecisionExhausted, N
from .containers import Tuple, Dict
from .exprtools import gcd_terms, factor_terms, factor_nc
from .evaluate import evaluate

# expose singletons
Catalan = S.Catalan
EulerGamma = S.EulerGamma
GoldenRatio = S.GoldenRatio
TribonacciConstant = S.TribonacciConstant
```
### 23 - sympy/core/function.py:

Start line: 2208, End line: 2267

```python
class Subs(Expr):

    def _eval_is_commutative(self):
        return self.expr.is_commutative

    def doit(self, **hints):
        e, v, p = self.args

        # remove self mappings
        for i, (vi, pi) in enumerate(zip(v, p)):
            if vi == pi:
                v = v[:i] + v[i + 1:]
                p = p[:i] + p[i + 1:]
        if not v:
            return self.expr

        if isinstance(e, Derivative):
            # apply functions first, e.g. f -> cos
            undone = []
            for i, vi in enumerate(v):
                if isinstance(vi, FunctionClass):
                    e = e.subs(vi, p[i])
                else:
                    undone.append((vi, p[i]))
            if not isinstance(e, Derivative):
                e = e.doit()
            if isinstance(e, Derivative):
                # do Subs that aren't related to differentiation
                undone2 = []
                D = Dummy()
                for vi, pi in undone:
                    if D not in e.xreplace({vi: D}).free_symbols:
                        e = e.subs(vi, pi)
                    else:
                        undone2.append((vi, pi))
                undone = undone2
                # differentiate wrt variables that are present
                wrt = []
                D = Dummy()
                expr = e.expr
                free = expr.free_symbols
                for vi, ci in e.variable_count:
                    if isinstance(vi, Symbol) and vi in free:
                        expr = expr.diff((vi, ci))
                    elif D in expr.subs(vi, D).free_symbols:
                        expr = expr.diff((vi, ci))
                    else:
                        wrt.append((vi, ci))
                # inject remaining subs
                rv = expr.subs(undone)
                # do remaining differentiation *in order given*
                for vc in wrt:
                    rv = rv.diff(vc)
            else:
                # inject remaining subs
                rv = e.subs(undone)
        else:
            rv = e.doit(**hints).subs(list(zip(v, p)))

        if hints.get('deep', True) and rv != self:
            rv = rv.doit(**hints)
        return rv
```
### 25 - sympy/core/function.py:

Start line: 482, End line: 503

```python
class Function(Application, Expr):

    @classmethod
    def _should_evalf(cls, arg):
        """
        Decide if the function should automatically evalf().

        By default (in this implementation), this happens if (and only if) the
        ARG is a floating point number.
        This function is used by __new__.

        Returns the precision to evalf to, or -1 if it shouldn't evalf.
        """
        from sympy.core.evalf import pure_complex
        if arg.is_Float:
            return arg._prec
        if not arg.is_Add:
            return -1
        m = pure_complex(arg)
        if m is None or not (m[0].is_Float or m[1].is_Float):
            return -1
        l = [i._prec for i in m if i.is_Float]
        l.append(-1)
        return max(l)
```
### 26 - sympy/core/power.py:

Start line: 783, End line: 837

```python
class Pow(Expr):

    def _eval_subs(self, old, new):
        # ... other code

        if old == self.base:
            return new**self.exp._subs(old, new)

        # issue 10829: (4**x - 3*y + 2).subs(2**x, y) -> y**2 - 3*y + 2
        if isinstance(old, self.func) and self.exp == old.exp:
            l = log(self.base, old.base)
            if l.is_Number:
                return Pow(new, l)

        if isinstance(old, self.func) and self.base == old.base:
            if self.exp.is_Add is False:
                ct1 = self.exp.as_independent(Symbol, as_Add=False)
                ct2 = old.exp.as_independent(Symbol, as_Add=False)
                ok, pow, remainder_pow = _check(ct1, ct2, old)
                if ok:
                    # issue 5180: (x**(6*y)).subs(x**(3*y),z)->z**2
                    result = self.func(new, pow)
                    if remainder_pow is not None:
                        result = Mul(result, Pow(old.base, remainder_pow))
                    return result
            else:  # b**(6*x + a).subs(b**(3*x), y) -> y**2 * b**a
                # exp(exp(x) + exp(x**2)).subs(exp(exp(x)), w) -> w * exp(exp(x**2))
                oarg = old.exp
                new_l = []
                o_al = []
                ct2 = oarg.as_coeff_mul()
                for a in self.exp.args:
                    newa = a._subs(old, new)
                    ct1 = newa.as_coeff_mul()
                    ok, pow, remainder_pow = _check(ct1, ct2, old)
                    if ok:
                        new_l.append(new**pow)
                        if remainder_pow is not None:
                            o_al.append(remainder_pow)
                        continue
                    elif not old.is_commutative and not newa.is_integer:
                        # If any term in the exponent is non-integer,
                        # we do not do any substitutions in the noncommutative case
                        return
                    o_al.append(newa)
                if new_l:
                    expo = Add(*o_al)
                    new_l.append(Pow(self.base, expo, evaluate=False) if expo != 1 else self.base)
                    return Mul(*new_l)

        if isinstance(old, exp) and self.exp.is_extended_real and self.base.is_positive:
            ct1 = old.args[0].as_independent(Symbol, as_Add=False)
            ct2 = (self.exp*log(self.base)).as_independent(
                Symbol, as_Add=False)
            ok, pow, remainder_pow = _check(ct1, ct2, old)
            if ok:
                result = self.func(new, pow)  # (2**x).subs(exp(x*log(2)), z) -> z
                if remainder_pow is not None:
                    result = Mul(result, Pow(old.base, remainder_pow))
                return result
```
### 29 - sympy/core/add.py:

Start line: 495, End line: 537

```python
class Add(Expr, AssocOp):

    def _eval_is_polynomial(self, syms):
        return all(term._eval_is_polynomial(syms) for term in self.args)

    def _eval_is_rational_function(self, syms):
        return all(term._eval_is_rational_function(syms) for term in self.args)

    def _eval_is_algebraic_expr(self, syms):
        return all(term._eval_is_algebraic_expr(syms) for term in self.args)

    # assumption methods
    _eval_is_real = lambda self: _fuzzy_group(
        (a.is_real for a in self.args), quick_exit=True)
    _eval_is_extended_real = lambda self: _fuzzy_group(
        (a.is_extended_real for a in self.args), quick_exit=True)
    _eval_is_complex = lambda self: _fuzzy_group(
        (a.is_complex for a in self.args), quick_exit=True)
    _eval_is_antihermitian = lambda self: _fuzzy_group(
        (a.is_antihermitian for a in self.args), quick_exit=True)
    _eval_is_finite = lambda self: _fuzzy_group(
        (a.is_finite for a in self.args), quick_exit=True)
    _eval_is_hermitian = lambda self: _fuzzy_group(
        (a.is_hermitian for a in self.args), quick_exit=True)
    _eval_is_integer = lambda self: _fuzzy_group(
        (a.is_integer for a in self.args), quick_exit=True)
    _eval_is_rational = lambda self: _fuzzy_group(
        (a.is_rational for a in self.args), quick_exit=True)
    _eval_is_algebraic = lambda self: _fuzzy_group(
        (a.is_algebraic for a in self.args), quick_exit=True)
    _eval_is_commutative = lambda self: _fuzzy_group(
        a.is_commutative for a in self.args)

    def _eval_is_infinite(self):
        sawinf = False
        for a in self.args:
            ainf = a.is_infinite
            if ainf is None:
                return None
            elif ainf is True:
                # infinite+infinite might not be infinite
                if sawinf is True:
                    return None
                sawinf = True
        return sawinf
```
### 31 - sympy/core/function.py:

Start line: 2317, End line: 2339

```python
class Subs(Expr):

    def _eval_subs(self, old, new):
        # Subs doit will do the variables in order; the semantics
        # of subs for Subs is have the following invariant for
        # Subs object foo:
        #    foo.doit().subs(reps) == foo.subs(reps).doit()
        pt = list(self.point)
        if old in self.variables:
            if _atomic(new) == set([new]) and not any(
                    i.has(new) for i in self.args):
                # the substitution is neutral
                return self.xreplace({old: new})
            # any occurrence of old before this point will get
            # handled by replacements from here on
            i = self.variables.index(old)
            for j in range(i, len(self.variables)):
                pt[j] = pt[j]._subs(old, new)
            return self.func(self.expr, self.variables, pt)
        v = [i._subs(old, new) for i in self.variables]
        if v != list(self.variables):
            return self.func(self.expr, self.variables + (old,), pt + [new])
        expr = self.expr._subs(old, new)
        pt = [i._subs(old, new) for i in self.point]
        return self.func(expr, v, pt)
```
### 32 - sympy/core/function.py:

Start line: 318, End line: 356

```python
class Application(with_metaclass(FunctionClass, Basic)):

    @classmethod
    def eval(cls, *args):
        """
        Returns a canonical form of cls applied to arguments args.

        The eval() method is called when the class cls is about to be
        instantiated and it should return either some simplified instance
        (possible of some other class), or if the class cls should be
        unmodified, return None.

        Examples of eval() for the function "sign"
        ---------------------------------------------

        .. code-block:: python

            @classmethod
            def eval(cls, arg):
                if arg is S.NaN:
                    return S.NaN
                if arg.is_zero: return S.Zero
                if arg.is_positive: return S.One
                if arg.is_negative: return S.NegativeOne
                if isinstance(arg, Mul):
                    coeff, terms = arg.as_coeff_Mul(rational=True)
                    if coeff is not S.One:
                        return cls(coeff) * cls(terms)

        """
        return

    @property
    def func(self):
        return self.__class__

    def _eval_subs(self, old, new):
        if (old.is_Function and new.is_Function and
            callable(old) and callable(new) and
            old == self.func and len(self.args) in new.nargs):
            return new(*[i._subs(old, new) for i in self.args])
```
### 33 - sympy/core/function.py:

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
### 38 - sympy/core/function.py:

Start line: 605, End line: 628

```python
class Function(Application, Expr):

    def _eval_derivative(self, s):
        # f(x).diff(s) -> x.diff(s) * f.fdiff(1)(s)
        i = 0
        l = []
        for a in self.args:
            i += 1
            da = a.diff(s)
            if da.is_zero:
                continue
            try:
                df = self.fdiff(i)
            except ArgumentIndexError:
                df = Function.fdiff(self, i)
            l.append(df * da)
        return Add(*l)

    def _eval_is_commutative(self):
        return fuzzy_and(a.is_commutative for a in self.args)

    def as_base_exp(self):
        """
        Returns the method as the 2-tuple (base, exponent).
        """
        return self, S.One
```
### 39 - sympy/core/function.py:

Start line: 704, End line: 746

```python
class Function(Application, Expr):

    def _eval_nseries(self, x, n, logx):
        # ... other code
        if (self.func.nargs is S.Naturals0
                or (self.func.nargs == FiniteSet(1) and args0[0])
                or any(c > 1 for c in self.func.nargs)):
            e = self
            e1 = e.expand()
            if e == e1:
                #for example when e = sin(x+1) or e = sin(cos(x))
                #let's try the general algorithm
                term = e.subs(x, S.Zero)
                if term.is_finite is False or term is S.NaN:
                    raise PoleError("Cannot expand %s around 0" % (self))
                series = term
                fact = S.One
                _x = Dummy('x')
                e = e.subs(x, _x)
                for i in range(n - 1):
                    i += 1
                    fact *= Rational(i)
                    e = e.diff(_x)
                    subs = e.subs(_x, S.Zero)
                    if subs is S.NaN:
                        # try to evaluate a limit if we have to
                        subs = e.limit(_x, S.Zero)
                    if subs.is_finite is False:
                        raise PoleError("Cannot expand %s around 0" % (self))
                    term = subs*(x**i)/fact
                    term = term.expand()
                    series += term
                return series + Order(x**n, x)
            return e1.nseries(x, n=n, logx=logx)
        arg = self.args[0]
        l = []
        g = None
        # try to predict a number of terms needed
        nterms = n + 2
        cf = Order(arg.as_leading_term(x), x).getn()
        if cf != 0:
            nterms = int(nterms / cf)
        for i in range(nterms):
            g = self.taylor_term(i, arg, g)
            g = g.nseries(x, n=n, logx=logx)
            l.append(g)
        return Add(*l) + Order(x**n, x)
```
