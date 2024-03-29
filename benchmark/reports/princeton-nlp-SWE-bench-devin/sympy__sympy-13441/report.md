# sympy__sympy-13441

| **sympy/sympy** | `e0cd7d65857a90376a9b49529840f96908dd774f` |
| ---- | ---- |
| **No of patches** | 3 |
| **All found context length** | - |
| **Any found context length** | - |
| **Avg pos** | - |
| **Min pos** | - |
| **Max pos** | - |
| **Top file pos** | 1 |
| **Missing snippets** | 4 |
| **Missing patch files** | 1 |


## Expected patch

```diff
diff --git a/sympy/core/add.py b/sympy/core/add.py
--- a/sympy/core/add.py
+++ b/sympy/core/add.py
@@ -396,13 +396,26 @@ def matches(self, expr, repl_dict={}, old=False):
     @staticmethod
     def _combine_inverse(lhs, rhs):
         """
-        Returns lhs - rhs, but treats arguments like symbols, so things like
-        oo - oo return 0, instead of a nan.
+        Returns lhs - rhs, but treats oo like a symbol so oo - oo
+        returns 0, instead of a nan.
         """
-        from sympy import oo, I, expand_mul
-        if lhs == oo and rhs == oo or lhs == oo*I and rhs == oo*I:
-            return S.Zero
-        return expand_mul(lhs - rhs)
+        from sympy.core.function import expand_mul
+        from sympy.core.symbol import Dummy
+        inf = (S.Infinity, S.NegativeInfinity)
+        if lhs.has(*inf) or rhs.has(*inf):
+            oo = Dummy('oo')
+            reps = {
+                S.Infinity: oo,
+                S.NegativeInfinity: -oo}
+            ireps = dict([(v, k) for k, v in reps.items()])
+            eq = expand_mul(lhs.xreplace(reps) - rhs.xreplace(reps))
+            if eq.has(oo):
+                eq = eq.replace(
+                    lambda x: x.is_Pow and x.base == oo,
+                    lambda x: x.base)
+            return eq.xreplace(ireps)
+        else:
+            return expand_mul(lhs - rhs)
 
     @cacheit
     def as_two_terms(self):
diff --git a/sympy/core/function.py b/sympy/core/function.py
--- a/sympy/core/function.py
+++ b/sympy/core/function.py
@@ -456,17 +456,15 @@ def _should_evalf(cls, arg):
 
         Returns the precision to evalf to, or -1 if it shouldn't evalf.
         """
-        from sympy.core.symbol import Wild
+        from sympy.core.evalf import pure_complex
         if arg.is_Float:
             return arg._prec
         if not arg.is_Add:
             return -1
-        # Don't use as_real_imag() here, that's too much work
-        a, b = Wild('a'), Wild('b')
-        m = arg.match(a + b*S.ImaginaryUnit)
-        if not m or not (m[a].is_Float or m[b].is_Float):
+        m = pure_complex(arg)
+        if m is None or not (m[0].is_Float or m[1].is_Float):
             return -1
-        l = [m[i]._prec for i in m if m[i].is_Float]
+        l = [i._prec for i in m if i.is_Float]
         l.append(-1)
         return max(l)
 
diff --git a/sympy/core/operations.py b/sympy/core/operations.py
--- a/sympy/core/operations.py
+++ b/sympy/core/operations.py
@@ -79,8 +79,8 @@ def _new_rawargs(self, *args, **kwargs):
 
            Note: use this with caution. There is no checking of arguments at
            all. This is best used when you are rebuilding an Add or Mul after
-           simply removing one or more terms. If modification which result,
-           for example, in extra 1s being inserted (as when collecting an
+           simply removing one or more args. If, for example, modifications,
+           result in extra 1s being inserted (as when collecting an
            expression's numerators and denominators) they will not show up in
            the result but a Mul will be returned nonetheless:
 
@@ -180,29 +180,26 @@ def _matches_commutative(self, expr, repl_dict={}, old=False):
         # eliminate exact part from pattern: (2+a+w1+w2).matches(expr) -> (w1+w2).matches(expr-a-2)
         from .function import WildFunction
         from .symbol import Wild
-        wild_part = []
-        exact_part = []
-        for p in ordered(self.args):
-            if p.has(Wild, WildFunction) and (not expr.has(p)):
-                # not all Wild should stay Wilds, for example:
-                # (w2+w3).matches(w1) -> (w1+w3).matches(w1) -> w3.matches(0)
-                wild_part.append(p)
-            else:
-                exact_part.append(p)
-
-        if exact_part:
-            exact = self.func(*exact_part)
+        from sympy.utilities.iterables import sift
+        sifted = sift(self.args, lambda p:
+            p.has(Wild, WildFunction) and not expr.has(p))
+        wild_part = sifted[True]
+        exact_part = sifted[False]
+        if not exact_part:
+            wild_part = list(ordered(wild_part))
+        else:
+            exact = self._new_rawargs(*exact_part)
             free = expr.free_symbols
             if free and (exact.free_symbols - free):
                 # there are symbols in the exact part that are not
                 # in the expr; but if there are no free symbols, let
                 # the matching continue
                 return None
-            newpattern = self.func(*wild_part)
             newexpr = self._combine_inverse(expr, exact)
             if not old and (expr.is_Add or expr.is_Mul):
                 if newexpr.count_ops() > expr.count_ops():
                     return None
+            newpattern = self._new_rawargs(*wild_part)
             return newpattern.matches(newexpr, repl_dict)
 
         # now to real work ;)

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sympy/core/add.py | 399 | 405 | - | - | -
| sympy/core/function.py | 459 | 469 | - | 1 | -
| sympy/core/operations.py | 82 | 83 | - | 4 | -
| sympy/core/operations.py | 183 | 201 | - | 4 | -


## Problem Statement

```
count_ops is slow for large expressions
It seems that this script was hanging inside `count_ops`:

\`\`\`
moorepants@garuda:pydy.wiki(master)$ SYMPY_CACHE_SIZE=10000 ipython
Python 3.5.1 |Continuum Analytics, Inc.| (default, Dec  7 2015, 11:16:01) 
Type "copyright", "credits" or "license" for more information.

IPython 4.1.2 -- An enhanced Interactive Python.
?         -> Introduction and overview of IPython's features.
%quickref -> Quick reference.
help      -> Python's own help system.
object?   -> Details about 'object', use 'object??' for extra details.

In [1]: %paste
   In [1]: from pydy.models import n_link_pendulum_on_cart

   In [2]: sys = n_link_pendulum_on_cart(3)

   In [3]: x_dot = sys.eom_method.rhs()

   In [4]: %time jac = x_dot.jacobian(sys.states)
## -- End pasted text --
CPU times: user 2.2 s, sys: 4 ms, total: 2.21 s
Wall time: 2.2 s

In [2]: %paste
   In [5]: sys = n_link_pendulum_on_cart(4)

   In [6]: x_dot = sys.eom_method.rhs()

   In [7]: %time jac = x_dot.jacobian(sys.states)
## -- End pasted text --
^C---------------------------------------------------------------------------
KeyboardInterrupt                         Traceback (most recent call last)
<ipython-input-2-1039ec729c05> in <module>()
      3 x_dot = sys.eom_method.rhs()
      4 
----> 5 get_ipython().magic('time jac = x_dot.jacobian(sys.states)')

/home/moorepants/miniconda3/lib/python3.5/site-packages/IPython/core/interactiveshell.py in magic(self, arg_s)
   2161         magic_name, _, magic_arg_s = arg_s.partition(' ')
   2162         magic_name = magic_name.lstrip(prefilter.ESC_MAGIC)
-> 2163         return self.run_line_magic(magic_name, magic_arg_s)
   2164 
   2165     #-------------------------------------------------------------------------

/home/moorepants/miniconda3/lib/python3.5/site-packages/IPython/core/interactiveshell.py in run_line_magic(self, magic_name, line)
   2082                 kwargs['local_ns'] = sys._getframe(stack_depth).f_locals
   2083             with self.builtin_trap:
-> 2084                 result = fn(*args,**kwargs)
   2085             return result
   2086 

<decorator-gen-60> in time(self, line, cell, local_ns)

/home/moorepants/miniconda3/lib/python3.5/site-packages/IPython/core/magic.py in <lambda>(f, *a, **k)
    191     # but it's overkill for just that one bit of state.
    192     def magic_deco(arg):
--> 193         call = lambda f, *a, **k: f(*a, **k)
    194 
    195         if callable(arg):

/home/moorepants/miniconda3/lib/python3.5/site-packages/IPython/core/magics/execution.py in time(self, line, cell, local_ns)
   1175         else:
   1176             st = clock2()
-> 1177             exec(code, glob, local_ns)
   1178             end = clock2()
   1179             out = None

<timed exec> in <module>()

/home/moorepants/miniconda3/lib/python3.5/site-packages/sympy/matrices/matrices.py in jacobian(self, X)
   1551         # m is the number of functions and n is the number of variables
   1552         # computing the Jacobian is now easy:
-> 1553         return self._new(m, n, lambda j, i: self[j].diff(X[i]))
   1554 
   1555     def QRdecomposition(self):

/home/moorepants/miniconda3/lib/python3.5/site-packages/sympy/matrices/dense.py in _new(cls, *args, **kwargs)
    601     @classmethod
    602     def _new(cls, *args, **kwargs):
--> 603         rows, cols, flat_list = cls._handle_creation_inputs(*args, **kwargs)
    604         self = object.__new__(cls)
    605         self.rows = rows

/home/moorepants/miniconda3/lib/python3.5/site-packages/sympy/matrices/matrices.py in _handle_creation_inputs(cls, *args, **kwargs)
    207                     flat_list.extend(
    208                         [cls._sympify(op(cls._sympify(i), cls._sympify(j)))
--> 209                         for j in range(cols)])
    210 
    211             # Matrix(2, 2, [1, 2, 3, 4])

/home/moorepants/miniconda3/lib/python3.5/site-packages/sympy/matrices/matrices.py in <listcomp>(.0)
    207                     flat_list.extend(
    208                         [cls._sympify(op(cls._sympify(i), cls._sympify(j)))
--> 209                         for j in range(cols)])
    210 
    211             # Matrix(2, 2, [1, 2, 3, 4])

/home/moorepants/miniconda3/lib/python3.5/site-packages/sympy/matrices/matrices.py in <lambda>(j, i)
   1551         # m is the number of functions and n is the number of variables
   1552         # computing the Jacobian is now easy:
-> 1553         return self._new(m, n, lambda j, i: self[j].diff(X[i]))
   1554 
   1555     def QRdecomposition(self):

/home/moorepants/miniconda3/lib/python3.5/site-packages/sympy/core/expr.py in diff(self, *symbols, **assumptions)
   2864         new_symbols = list(map(sympify, symbols))  # e.g. x, 2, y, z
   2865         assumptions.setdefault("evaluate", True)
-> 2866         return Derivative(self, *new_symbols, **assumptions)
   2867 
   2868     ###########################################################################

/home/moorepants/miniconda3/lib/python3.5/site-packages/sympy/core/function.py in __new__(cls, expr, *variables, **assumptions)
   1141                     old_v = v
   1142                     v = new_v
-> 1143                 obj = expr._eval_derivative(v)
   1144                 nderivs += 1
   1145                 if not is_symbol:

/home/moorepants/miniconda3/lib/python3.5/site-packages/sympy/core/mul.py in _eval_derivative(self, s)
    832         terms = []
    833         for i in range(len(args)):
--> 834             d = args[i].diff(s)
    835             if d:
    836                 terms.append(self.func(*(args[:i] + [d] + args[i + 1:])))

/home/moorepants/miniconda3/lib/python3.5/site-packages/sympy/core/expr.py in diff(self, *symbols, **assumptions)
   2864         new_symbols = list(map(sympify, symbols))  # e.g. x, 2, y, z
   2865         assumptions.setdefault("evaluate", True)
-> 2866         return Derivative(self, *new_symbols, **assumptions)
   2867 
   2868     ###########################################################################

/home/moorepants/miniconda3/lib/python3.5/site-packages/sympy/core/function.py in __new__(cls, expr, *variables, **assumptions)
   1141                     old_v = v
   1142                     v = new_v
-> 1143                 obj = expr._eval_derivative(v)
   1144                 nderivs += 1
   1145                 if not is_symbol:

/home/moorepants/miniconda3/lib/python3.5/site-packages/sympy/core/add.py in _eval_derivative(self, s)
    351     @cacheit
    352     def _eval_derivative(self, s):
--> 353         return self.func(*[a.diff(s) for a in self.args])
    354 
    355     def _eval_nseries(self, x, n, logx):

/home/moorepants/miniconda3/lib/python3.5/site-packages/sympy/core/add.py in <listcomp>(.0)
    351     @cacheit
    352     def _eval_derivative(self, s):
--> 353         return self.func(*[a.diff(s) for a in self.args])
    354 
    355     def _eval_nseries(self, x, n, logx):

/home/moorepants/miniconda3/lib/python3.5/site-packages/sympy/core/expr.py in diff(self, *symbols, **assumptions)
   2864         new_symbols = list(map(sympify, symbols))  # e.g. x, 2, y, z
   2865         assumptions.setdefault("evaluate", True)
-> 2866         return Derivative(self, *new_symbols, **assumptions)
   2867 
   2868     ###########################################################################

/home/moorepants/miniconda3/lib/python3.5/site-packages/sympy/core/function.py in __new__(cls, expr, *variables, **assumptions)
   1141                     old_v = v
   1142                     v = new_v
-> 1143                 obj = expr._eval_derivative(v)
   1144                 nderivs += 1
   1145                 if not is_symbol:

/home/moorepants/miniconda3/lib/python3.5/site-packages/sympy/core/mul.py in _eval_derivative(self, s)
    832         terms = []
    833         for i in range(len(args)):
--> 834             d = args[i].diff(s)
    835             if d:
    836                 terms.append(self.func(*(args[:i] + [d] + args[i + 1:])))

/home/moorepants/miniconda3/lib/python3.5/site-packages/sympy/core/expr.py in diff(self, *symbols, **assumptions)
   2864         new_symbols = list(map(sympify, symbols))  # e.g. x, 2, y, z
   2865         assumptions.setdefault("evaluate", True)
-> 2866         return Derivative(self, *new_symbols, **assumptions)
   2867 
   2868     ###########################################################################

/home/moorepants/miniconda3/lib/python3.5/site-packages/sympy/core/function.py in __new__(cls, expr, *variables, **assumptions)
   1141                     old_v = v
   1142                     v = new_v
-> 1143                 obj = expr._eval_derivative(v)
   1144                 nderivs += 1
   1145                 if not is_symbol:

/home/moorepants/miniconda3/lib/python3.5/site-packages/sympy/core/add.py in _eval_derivative(self, s)
    351     @cacheit
    352     def _eval_derivative(self, s):
--> 353         return self.func(*[a.diff(s) for a in self.args])
    354 
    355     def _eval_nseries(self, x, n, logx):

/home/moorepants/miniconda3/lib/python3.5/site-packages/sympy/core/add.py in <listcomp>(.0)
    351     @cacheit
    352     def _eval_derivative(self, s):
--> 353         return self.func(*[a.diff(s) for a in self.args])
    354 
    355     def _eval_nseries(self, x, n, logx):

/home/moorepants/miniconda3/lib/python3.5/site-packages/sympy/core/expr.py in diff(self, *symbols, **assumptions)
   2864         new_symbols = list(map(sympify, symbols))  # e.g. x, 2, y, z
   2865         assumptions.setdefault("evaluate", True)
-> 2866         return Derivative(self, *new_symbols, **assumptions)
   2867 
   2868     ###########################################################################

/home/moorepants/miniconda3/lib/python3.5/site-packages/sympy/core/function.py in __new__(cls, expr, *variables, **assumptions)
   1141                     old_v = v
   1142                     v = new_v
-> 1143                 obj = expr._eval_derivative(v)
   1144                 nderivs += 1
   1145                 if not is_symbol:

/home/moorepants/miniconda3/lib/python3.5/site-packages/sympy/core/mul.py in _eval_derivative(self, s)
    832         terms = []
    833         for i in range(len(args)):
--> 834             d = args[i].diff(s)
    835             if d:
    836                 terms.append(self.func(*(args[:i] + [d] + args[i + 1:])))

/home/moorepants/miniconda3/lib/python3.5/site-packages/sympy/core/expr.py in diff(self, *symbols, **assumptions)
   2864         new_symbols = list(map(sympify, symbols))  # e.g. x, 2, y, z
   2865         assumptions.setdefault("evaluate", True)
-> 2866         return Derivative(self, *new_symbols, **assumptions)
   2867 
   2868     ###########################################################################

/home/moorepants/miniconda3/lib/python3.5/site-packages/sympy/core/function.py in __new__(cls, expr, *variables, **assumptions)
   1141                     old_v = v
   1142                     v = new_v
-> 1143                 obj = expr._eval_derivative(v)
   1144                 nderivs += 1
   1145                 if not is_symbol:

/home/moorepants/miniconda3/lib/python3.5/site-packages/sympy/core/add.py in _eval_derivative(self, s)
    351     @cacheit
    352     def _eval_derivative(self, s):
--> 353         return self.func(*[a.diff(s) for a in self.args])
    354 
    355     def _eval_nseries(self, x, n, logx):

/home/moorepants/miniconda3/lib/python3.5/site-packages/sympy/core/add.py in <listcomp>(.0)
    351     @cacheit
    352     def _eval_derivative(self, s):
--> 353         return self.func(*[a.diff(s) for a in self.args])
    354 
    355     def _eval_nseries(self, x, n, logx):

/home/moorepants/miniconda3/lib/python3.5/site-packages/sympy/core/expr.py in diff(self, *symbols, **assumptions)
   2864         new_symbols = list(map(sympify, symbols))  # e.g. x, 2, y, z
   2865         assumptions.setdefault("evaluate", True)
-> 2866         return Derivative(self, *new_symbols, **assumptions)
   2867 
   2868     ###########################################################################

/home/moorepants/miniconda3/lib/python3.5/site-packages/sympy/core/function.py in __new__(cls, expr, *variables, **assumptions)
   1141                     old_v = v
   1142                     v = new_v
-> 1143                 obj = expr._eval_derivative(v)
   1144                 nderivs += 1
   1145                 if not is_symbol:

/home/moorepants/miniconda3/lib/python3.5/site-packages/sympy/core/mul.py in _eval_derivative(self, s)
    832         terms = []
    833         for i in range(len(args)):
--> 834             d = args[i].diff(s)
    835             if d:
    836                 terms.append(self.func(*(args[:i] + [d] + args[i + 1:])))

/home/moorepants/miniconda3/lib/python3.5/site-packages/sympy/core/expr.py in diff(self, *symbols, **assumptions)
   2864         new_symbols = list(map(sympify, symbols))  # e.g. x, 2, y, z
   2865         assumptions.setdefault("evaluate", True)
-> 2866         return Derivative(self, *new_symbols, **assumptions)
   2867 
   2868     ###########################################################################

/home/moorepants/miniconda3/lib/python3.5/site-packages/sympy/core/function.py in __new__(cls, expr, *variables, **assumptions)
   1141                     old_v = v
   1142                     v = new_v
-> 1143                 obj = expr._eval_derivative(v)
   1144                 nderivs += 1
   1145                 if not is_symbol:

/home/moorepants/miniconda3/lib/python3.5/site-packages/sympy/core/add.py in _eval_derivative(self, s)
    351     @cacheit
    352     def _eval_derivative(self, s):
--> 353         return self.func(*[a.diff(s) for a in self.args])
    354 
    355     def _eval_nseries(self, x, n, logx):

/home/moorepants/miniconda3/lib/python3.5/site-packages/sympy/core/add.py in <listcomp>(.0)
    351     @cacheit
    352     def _eval_derivative(self, s):
--> 353         return self.func(*[a.diff(s) for a in self.args])
    354 
    355     def _eval_nseries(self, x, n, logx):

/home/moorepants/miniconda3/lib/python3.5/site-packages/sympy/core/expr.py in diff(self, *symbols, **assumptions)
   2864         new_symbols = list(map(sympify, symbols))  # e.g. x, 2, y, z
   2865         assumptions.setdefault("evaluate", True)
-> 2866         return Derivative(self, *new_symbols, **assumptions)
   2867 
   2868     ###########################################################################

/home/moorepants/miniconda3/lib/python3.5/site-packages/sympy/core/function.py in __new__(cls, expr, *variables, **assumptions)
   1141                     old_v = v
   1142                     v = new_v
-> 1143                 obj = expr._eval_derivative(v)
   1144                 nderivs += 1
   1145                 if not is_symbol:

/home/moorepants/miniconda3/lib/python3.5/site-packages/sympy/core/mul.py in _eval_derivative(self, s)
    832         terms = []
    833         for i in range(len(args)):
--> 834             d = args[i].diff(s)
    835             if d:
    836                 terms.append(self.func(*(args[:i] + [d] + args[i + 1:])))

/home/moorepants/miniconda3/lib/python3.5/site-packages/sympy/core/expr.py in diff(self, *symbols, **assumptions)
   2864         new_symbols = list(map(sympify, symbols))  # e.g. x, 2, y, z
   2865         assumptions.setdefault("evaluate", True)
-> 2866         return Derivative(self, *new_symbols, **assumptions)
   2867 
   2868     ###########################################################################

/home/moorepants/miniconda3/lib/python3.5/site-packages/sympy/core/function.py in __new__(cls, expr, *variables, **assumptions)
   1141                     old_v = v
   1142                     v = new_v
-> 1143                 obj = expr._eval_derivative(v)
   1144                 nderivs += 1
   1145                 if not is_symbol:

/home/moorepants/miniconda3/lib/python3.5/site-packages/sympy/core/power.py in _eval_derivative(self, s)
    982     def _eval_derivative(self, s):
    983         from sympy import log
--> 984         dbase = self.base.diff(s)
    985         dexp = self.exp.diff(s)
    986         return self * (dexp * log(self.base) + dbase * self.exp/self.base)

/home/moorepants/miniconda3/lib/python3.5/site-packages/sympy/core/expr.py in diff(self, *symbols, **assumptions)
   2864         new_symbols = list(map(sympify, symbols))  # e.g. x, 2, y, z
   2865         assumptions.setdefault("evaluate", True)
-> 2866         return Derivative(self, *new_symbols, **assumptions)
   2867 
   2868     ###########################################################################

/home/moorepants/miniconda3/lib/python3.5/site-packages/sympy/core/function.py in __new__(cls, expr, *variables, **assumptions)
   1141                     old_v = v
   1142                     v = new_v
-> 1143                 obj = expr._eval_derivative(v)
   1144                 nderivs += 1
   1145                 if not is_symbol:

/home/moorepants/miniconda3/lib/python3.5/site-packages/sympy/core/add.py in _eval_derivative(self, s)
    351     @cacheit
    352     def _eval_derivative(self, s):
--> 353         return self.func(*[a.diff(s) for a in self.args])
    354 
    355     def _eval_nseries(self, x, n, logx):

/home/moorepants/miniconda3/lib/python3.5/site-packages/sympy/core/add.py in <listcomp>(.0)
    351     @cacheit
    352     def _eval_derivative(self, s):
--> 353         return self.func(*[a.diff(s) for a in self.args])
    354 
    355     def _eval_nseries(self, x, n, logx):

/home/moorepants/miniconda3/lib/python3.5/site-packages/sympy/core/expr.py in diff(self, *symbols, **assumptions)
   2864         new_symbols = list(map(sympify, symbols))  # e.g. x, 2, y, z
   2865         assumptions.setdefault("evaluate", True)
-> 2866         return Derivative(self, *new_symbols, **assumptions)
   2867 
   2868     ###########################################################################

/home/moorepants/miniconda3/lib/python3.5/site-packages/sympy/core/function.py in __new__(cls, expr, *variables, **assumptions)
   1141                     old_v = v
   1142                     v = new_v
-> 1143                 obj = expr._eval_derivative(v)
   1144                 nderivs += 1
   1145                 if not is_symbol:

/home/moorepants/miniconda3/lib/python3.5/site-packages/sympy/core/mul.py in _eval_derivative(self, s)
    832         terms = []
    833         for i in range(len(args)):
--> 834             d = args[i].diff(s)
    835             if d:
    836                 terms.append(self.func(*(args[:i] + [d] + args[i + 1:])))

/home/moorepants/miniconda3/lib/python3.5/site-packages/sympy/core/expr.py in diff(self, *symbols, **assumptions)
   2864         new_symbols = list(map(sympify, symbols))  # e.g. x, 2, y, z
   2865         assumptions.setdefault("evaluate", True)
-> 2866         return Derivative(self, *new_symbols, **assumptions)
   2867 
   2868     ###########################################################################

/home/moorepants/miniconda3/lib/python3.5/site-packages/sympy/core/function.py in __new__(cls, expr, *variables, **assumptions)
   1141                     old_v = v
   1142                     v = new_v
-> 1143                 obj = expr._eval_derivative(v)
   1144                 nderivs += 1
   1145                 if not is_symbol:

/home/moorepants/miniconda3/lib/python3.5/site-packages/sympy/core/power.py in _eval_derivative(self, s)
    984         dbase = self.base.diff(s)
    985         dexp = self.exp.diff(s)
--> 986         return self * (dexp * log(self.base) + dbase * self.exp/self.base)
    987 
    988     def _eval_evalf(self, prec):

/home/moorepants/miniconda3/lib/python3.5/site-packages/sympy/core/function.py in __new__(cls, *args, **options)
    388 
    389         pr = max(cls._should_evalf(a) for a in result.args)
--> 390         pr2 = min(cls._should_evalf(a) for a in result.args)
    391         if pr2 > 0:
    392             return result.evalf(mlib.libmpf.prec_to_dps(pr))

/home/moorepants/miniconda3/lib/python3.5/site-packages/sympy/core/function.py in <genexpr>(.0)
    388 
    389         pr = max(cls._should_evalf(a) for a in result.args)
--> 390         pr2 = min(cls._should_evalf(a) for a in result.args)
    391         if pr2 > 0:
    392             return result.evalf(mlib.libmpf.prec_to_dps(pr))

/home/moorepants/miniconda3/lib/python3.5/site-packages/sympy/core/function.py in _should_evalf(cls, arg)
    411         # Don't use as_real_imag() here, that's too much work
    412         a, b = Wild('a'), Wild('b')
--> 413         m = arg.match(a + b*S.ImaginaryUnit)
    414         if not m or not (m[a].is_Float or m[b].is_Float):
    415             return -1

/home/moorepants/miniconda3/lib/python3.5/site-packages/sympy/core/basic.py in match(self, pattern, old)
   1489         """
   1490         pattern = sympify(pattern)
-> 1491         return pattern.matches(self, old=old)
   1492 
   1493     def count_ops(self, visual=None):

/home/moorepants/miniconda3/lib/python3.5/site-packages/sympy/core/add.py in matches(self, expr, repl_dict, old)
    365 
    366     def matches(self, expr, repl_dict={}, old=False):
--> 367         return AssocOp._matches_commutative(self, expr, repl_dict, old)
    368 
    369     @staticmethod

/home/moorepants/miniconda3/lib/python3.5/site-packages/sympy/core/operations.py in _matches_commutative(self, expr, repl_dict, old)
    215                     d1 = w.matches(last_op, repl_dict)
    216                     if d1 is not None:
--> 217                         d2 = self.xreplace(d1).matches(expr, d1)
    218                         if d2 is not None:
    219                             return d2

/home/moorepants/miniconda3/lib/python3.5/site-packages/sympy/core/add.py in matches(self, expr, repl_dict, old)
    365 
    366     def matches(self, expr, repl_dict={}, old=False):
--> 367         return AssocOp._matches_commutative(self, expr, repl_dict, old)
    368 
    369     @staticmethod

/home/moorepants/miniconda3/lib/python3.5/site-packages/sympy/core/operations.py in _matches_commutative(self, expr, repl_dict, old)
    201             newexpr = self._combine_inverse(expr, exact)
    202             if not old and (expr.is_Add or expr.is_Mul):
--> 203                 if newexpr.count_ops() > expr.count_ops():
    204                     return None
    205             return newpattern.matches(newexpr, repl_dict)

/home/moorepants/miniconda3/lib/python3.5/site-packages/sympy/core/expr.py in count_ops(self, visual)
   1017         """wrapper for count_ops that returns the operation count."""
   1018         from .function import count_ops
-> 1019         return count_ops(self, visual)
   1020 
   1021     def args_cnc(self, cset=False, warn=True, split_1=True):

/home/moorepants/miniconda3/lib/python3.5/site-packages/sympy/core/function.py in count_ops(expr, visual)
   2378                 a.is_Pow or
   2379                 a.is_Function or
-> 2380                 isinstance(a, Derivative) or
   2381                     isinstance(a, Integral)):
   2382 

KeyboardInterrupt: 

In [3]: 
\`\`\`


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 sympy/core/function.py** | 2499 | 2626| 904 | 904 | 23089 | 
| 2 | **1 sympy/core/function.py** | 2426 | 2498| 615 | 1519 | 23089 | 
| 3 | 2 sympy/core/basic.py | 1518 | 1547| 241 | 1760 | 37510 | 
| 4 | 3 sympy/core/expr.py | 1034 | 1088| 400 | 2160 | 65963 | 
| 5 | **4 sympy/core/operations.py** | 211 | 260| 431 | 2591 | 69779 | 
| 6 | 5 sympy/integrals/rubi/rubi.py | 1 | 131| 1310 | 3901 | 72013 | 
| 7 | 6 sympy/integrals/rubi/utility_function.py | 1 | 158| 1722 | 5623 | 154032 | 
| 8 | 6 sympy/integrals/rubi/utility_function.py | 6644 | 6700| 908 | 6531 | 154032 | 
| 9 | 6 sympy/core/expr.py | 2001 | 2082| 690 | 7221 | 154032 | 
| 10 | 6 sympy/core/expr.py | 92 | 150| 483 | 7704 | 154032 | 


## Missing Patch Files

 * 1: sympy/core/add.py
 * 2: sympy/core/function.py
 * 3: sympy/core/operations.py

### Hint

```
My first thought is that the following

\`\`\`
202             if not old and (expr.is_Add or expr.is_Mul):
203                 if newexpr.count_ops() > expr.count_ops():
\`\`\`

should be

\`\`\`
if not old and (expr.is_Add or expr.is_Mul):
    len(expr.func.make_args(newexpr)) > len(expr.args):
\`\`\`

Here is a pyinstrument profile of count_ops:

https://rawgit.com/moorepants/b92b851bcc5236f71de1caf61de98e88/raw/8e5ce6255971c115d46fed3d65560f427d0a44aa/profile_count_ops.html

I've updated the script so that it also calls `jacobian()` and found that for n>3 there are wildly different results. It seems that count_ops is called somewhere in jacobian if n>3.

Profile from n=3:

https://rawgit.com/moorepants/b92b851bcc5236f71de1caf61de98e88/raw/77e5f6f162e370b3a35060bef0030333e5ba3926/profile_count_ops_n_3.html

Profile from n=4 (had to kill this one because it doesn't finish):

https://rawgit.com/moorepants/b92b851bcc5236f71de1caf61de98e88/raw/77e5f6f162e370b3a35060bef0030333e5ba3926/profile_count_ops_n_4.html

This gist: https://gist.github.com/moorepants/b92b851bcc5236f71de1caf61de98e88

I'm seeing that _matches_commutative sympy/core/operations.py:127 is whats taking so much time. This calls some simplification routines in the fraction function and count_ops which takes for ever. I'm not sure why `_matches_commutative` is getting called.

The use of matches in the core should be forbidden (mostly):
\`\`\`
>>> from timeit import timeit
>>> timeit('e.match(pat)','''
... from sympy import I, Wild
... a,b=Wild('a'),Wild('b')
... e=3+4*I
... pat=a+I*b
... ''',number=100)
0.2531449845839618
>>> timeit('''
... a, b = e.as_two_terms()
... b = b.as_two_terms()
... ''','''
... from sympy import I, Wild
... a,b=Wild('a'),Wild('b')
... e=3+4*I
... pat=a+I*b
... ''',number=100)
0.008118156473557292
>>> timeit('''
... pure_complex(e)''','''
... from sympy import I
... from sympy.core.evalf import pure_complex
... e = 3+4*I''',number=100)
0.001546217867016253
\`\`\`
Could you run this again on my `n` branch?
Much improved. It finishes in a tolerable time:

\`\`\`
In [1]: from pydy.models import n_link_pendulum_on_cart

In [2]: sys = n_link_pendulum_on_cart(3)

In [3]: x_dot = sys.eom_method.rhs()

In [5]: %time jac = x_dot.jacobian(sys.states)
CPU times: user 1.85 s, sys: 0 ns, total: 1.85 s
Wall time: 1.85 s

In [6]: sys = n_link_pendulum_on_cart(4)

In [7]: x_dot = sys.eom_method.rhs()

In [8]: %time jac = x_dot.jacobian(sys.states)
CPU times: user 22.6 s, sys: 8 ms, total: 22.6 s
Wall time: 22.6 s
\`\`\`
```

## Patch

```diff
diff --git a/sympy/core/add.py b/sympy/core/add.py
--- a/sympy/core/add.py
+++ b/sympy/core/add.py
@@ -396,13 +396,26 @@ def matches(self, expr, repl_dict={}, old=False):
     @staticmethod
     def _combine_inverse(lhs, rhs):
         """
-        Returns lhs - rhs, but treats arguments like symbols, so things like
-        oo - oo return 0, instead of a nan.
+        Returns lhs - rhs, but treats oo like a symbol so oo - oo
+        returns 0, instead of a nan.
         """
-        from sympy import oo, I, expand_mul
-        if lhs == oo and rhs == oo or lhs == oo*I and rhs == oo*I:
-            return S.Zero
-        return expand_mul(lhs - rhs)
+        from sympy.core.function import expand_mul
+        from sympy.core.symbol import Dummy
+        inf = (S.Infinity, S.NegativeInfinity)
+        if lhs.has(*inf) or rhs.has(*inf):
+            oo = Dummy('oo')
+            reps = {
+                S.Infinity: oo,
+                S.NegativeInfinity: -oo}
+            ireps = dict([(v, k) for k, v in reps.items()])
+            eq = expand_mul(lhs.xreplace(reps) - rhs.xreplace(reps))
+            if eq.has(oo):
+                eq = eq.replace(
+                    lambda x: x.is_Pow and x.base == oo,
+                    lambda x: x.base)
+            return eq.xreplace(ireps)
+        else:
+            return expand_mul(lhs - rhs)
 
     @cacheit
     def as_two_terms(self):
diff --git a/sympy/core/function.py b/sympy/core/function.py
--- a/sympy/core/function.py
+++ b/sympy/core/function.py
@@ -456,17 +456,15 @@ def _should_evalf(cls, arg):
 
         Returns the precision to evalf to, or -1 if it shouldn't evalf.
         """
-        from sympy.core.symbol import Wild
+        from sympy.core.evalf import pure_complex
         if arg.is_Float:
             return arg._prec
         if not arg.is_Add:
             return -1
-        # Don't use as_real_imag() here, that's too much work
-        a, b = Wild('a'), Wild('b')
-        m = arg.match(a + b*S.ImaginaryUnit)
-        if not m or not (m[a].is_Float or m[b].is_Float):
+        m = pure_complex(arg)
+        if m is None or not (m[0].is_Float or m[1].is_Float):
             return -1
-        l = [m[i]._prec for i in m if m[i].is_Float]
+        l = [i._prec for i in m if i.is_Float]
         l.append(-1)
         return max(l)
 
diff --git a/sympy/core/operations.py b/sympy/core/operations.py
--- a/sympy/core/operations.py
+++ b/sympy/core/operations.py
@@ -79,8 +79,8 @@ def _new_rawargs(self, *args, **kwargs):
 
            Note: use this with caution. There is no checking of arguments at
            all. This is best used when you are rebuilding an Add or Mul after
-           simply removing one or more terms. If modification which result,
-           for example, in extra 1s being inserted (as when collecting an
+           simply removing one or more args. If, for example, modifications,
+           result in extra 1s being inserted (as when collecting an
            expression's numerators and denominators) they will not show up in
            the result but a Mul will be returned nonetheless:
 
@@ -180,29 +180,26 @@ def _matches_commutative(self, expr, repl_dict={}, old=False):
         # eliminate exact part from pattern: (2+a+w1+w2).matches(expr) -> (w1+w2).matches(expr-a-2)
         from .function import WildFunction
         from .symbol import Wild
-        wild_part = []
-        exact_part = []
-        for p in ordered(self.args):
-            if p.has(Wild, WildFunction) and (not expr.has(p)):
-                # not all Wild should stay Wilds, for example:
-                # (w2+w3).matches(w1) -> (w1+w3).matches(w1) -> w3.matches(0)
-                wild_part.append(p)
-            else:
-                exact_part.append(p)
-
-        if exact_part:
-            exact = self.func(*exact_part)
+        from sympy.utilities.iterables import sift
+        sifted = sift(self.args, lambda p:
+            p.has(Wild, WildFunction) and not expr.has(p))
+        wild_part = sifted[True]
+        exact_part = sifted[False]
+        if not exact_part:
+            wild_part = list(ordered(wild_part))
+        else:
+            exact = self._new_rawargs(*exact_part)
             free = expr.free_symbols
             if free and (exact.free_symbols - free):
                 # there are symbols in the exact part that are not
                 # in the expr; but if there are no free symbols, let
                 # the matching continue
                 return None
-            newpattern = self.func(*wild_part)
             newexpr = self._combine_inverse(expr, exact)
             if not old and (expr.is_Add or expr.is_Mul):
                 if newexpr.count_ops() > expr.count_ops():
                     return None
+            newpattern = self._new_rawargs(*wild_part)
             return newpattern.matches(newexpr, repl_dict)
 
         # now to real work ;)

```

## Test Patch

```diff
diff --git a/sympy/core/tests/test_match.py b/sympy/core/tests/test_match.py
--- a/sympy/core/tests/test_match.py
+++ b/sympy/core/tests/test_match.py
@@ -396,7 +396,7 @@ def test_match_wild_wild():
     assert p.match(q*r) is None
 
 
-def test_combine_inverse():
+def test__combine_inverse():
     x, y = symbols("x y")
     assert Mul._combine_inverse(x*I*y, x*I) == y
     assert Mul._combine_inverse(x*I*y, y*I) == x
@@ -404,6 +404,9 @@ def test_combine_inverse():
     assert Mul._combine_inverse(oo*I*y, oo*I) == y
     assert Add._combine_inverse(oo, oo) == S(0)
     assert Add._combine_inverse(oo*I, oo*I) == S(0)
+    assert Add._combine_inverse(x*oo, x*oo) == S(0)
+    assert Add._combine_inverse(-x*oo, -x*oo) == S(0)
+    assert Add._combine_inverse((x - oo)*(x + oo), -oo)
 
 
 def test_issue_3773():

```


## Code snippets

### 1 - sympy/core/function.py:

Start line: 2499, End line: 2626

```python
def count_ops(expr, visual=False):
    # ... other code
    if isinstance(expr, Expr):

        ops = []
        args = [expr]
        NEG = Symbol('NEG')
        DIV = Symbol('DIV')
        SUB = Symbol('SUB')
        ADD = Symbol('ADD')
        while args:
            a = args.pop()

            # XXX: This is a hack to support non-Basic args
            if isinstance(a, string_types):
                continue

            if a.is_Rational:
                #-1/3 = NEG + DIV
                if a is not S.One:
                    if a.p < 0:
                        ops.append(NEG)
                    if a.q != 1:
                        ops.append(DIV)
                    continue
            elif a.is_Mul:
                if _coeff_isneg(a):
                    ops.append(NEG)
                    if a.args[0] is S.NegativeOne:
                        a = a.as_two_terms()[1]
                    else:
                        a = -a
                n, d = fraction(a)
                if n.is_Integer:
                    ops.append(DIV)
                    if n < 0:
                        ops.append(NEG)
                    args.append(d)
                    continue  # won't be -Mul but could be Add
                elif d is not S.One:
                    if not d.is_Integer:
                        args.append(d)
                    ops.append(DIV)
                    args.append(n)
                    continue  # could be -Mul
            elif a.is_Add:
                aargs = list(a.args)
                negs = 0
                for i, ai in enumerate(aargs):
                    if _coeff_isneg(ai):
                        negs += 1
                        args.append(-ai)
                        if i > 0:
                            ops.append(SUB)
                    else:
                        args.append(ai)
                        if i > 0:
                            ops.append(ADD)
                if negs == len(aargs):  # -x - y = NEG + SUB
                    ops.append(NEG)
                elif _coeff_isneg(aargs[0]):  # -x + y = SUB, but already recorded ADD
                    ops.append(SUB - ADD)
                continue
            if a.is_Pow and a.exp is S.NegativeOne:
                ops.append(DIV)
                args.append(a.base)  # won't be -Mul but could be Add
                continue
            if (a.is_Mul or
                a.is_Pow or
                a.is_Function or
                isinstance(a, Derivative) or
                    isinstance(a, Integral)):

                o = Symbol(a.func.__name__.upper())
                # count the args
                if (a.is_Mul or isinstance(a, LatticeOp)):
                    ops.append(o*(len(a.args) - 1))
                else:
                    ops.append(o)
            if not a.is_Symbol:
                args.extend(a.args)

    elif type(expr) is dict:
        ops = [count_ops(k, visual=visual) +
               count_ops(v, visual=visual) for k, v in expr.items()]
    elif iterable(expr):
        ops = [count_ops(i, visual=visual) for i in expr]
    elif isinstance(expr, BooleanFunction):
        ops = []
        for arg in expr.args:
            ops.append(count_ops(arg, visual=True))
        o = Symbol(expr.func.__name__.upper())
        ops.append(o)
    elif not isinstance(expr, Basic):
        ops = []
    else:  # it's Basic not isinstance(expr, Expr):
        if not isinstance(expr, Basic):
            raise TypeError("Invalid type of expr")
        else:
            ops = []
            args = [expr]
            while args:
                a = args.pop()

                # XXX: This is a hack to support non-Basic args
                if isinstance(a, string_types):
                    continue

                if a.args:
                    o = Symbol(a.func.__name__.upper())
                    if a.is_Boolean:
                        ops.append(o*(len(a.args)-1))
                    else:
                        ops.append(o)
                    args.extend(a.args)

    if not ops:
        if visual:
            return S.Zero
        return 0

    ops = Add(*ops)

    if visual:
        return ops

    if ops.is_Number:
        return int(ops)

    return sum(int((a.args or [1])[0]) for a in Add.make_args(ops))
```
### 2 - sympy/core/function.py:

Start line: 2426, End line: 2498

```python
def count_ops(expr, visual=False):
    """
    Return a representation (integer or expression) of the operations in expr.

    If ``visual`` is ``False`` (default) then the sum of the coefficients of the
    visual expression will be returned.

    If ``visual`` is ``True`` then the number of each type of operation is shown
    with the core class types (or their virtual equivalent) multiplied by the
    number of times they occur.

    If expr is an iterable, the sum of the op counts of the
    items will be returned.

    Examples
    ========

    >>> from sympy.abc import a, b, x, y
    >>> from sympy import sin, count_ops

    Although there isn't a SUB object, minus signs are interpreted as
    either negations or subtractions:

    >>> (x - y).count_ops(visual=True)
    SUB
    >>> (-x).count_ops(visual=True)
    NEG

    Here, there are two Adds and a Pow:

    >>> (1 + a + b**2).count_ops(visual=True)
    2*ADD + POW

    In the following, an Add, Mul, Pow and two functions:

    >>> (sin(x)*x + sin(x)**2).count_ops(visual=True)
    ADD + MUL + POW + 2*SIN

    for a total of 5:

    >>> (sin(x)*x + sin(x)**2).count_ops(visual=False)
    5

    Note that "what you type" is not always what you get. The expression
    1/x/y is translated by sympy into 1/(x*y) so it gives a DIV and MUL rather
    than two DIVs:

    >>> (1/x/y).count_ops(visual=True)
    DIV + MUL

    The visual option can be used to demonstrate the difference in
    operations for expressions in different forms. Here, the Horner
    representation is compared with the expanded form of a polynomial:

    >>> eq=x*(1 + x*(2 + x*(3 + x)))
    >>> count_ops(eq.expand(), visual=True) - count_ops(eq, visual=True)
    -MUL + 3*POW

    The count_ops function also handles iterables:

    >>> count_ops([x, sin(x), None, True, x + 2], visual=False)
    2
    >>> count_ops([x, sin(x), None, True, x + 2], visual=True)
    ADD + SIN
    >>> count_ops({x: sin(x), x + 2: y + 1}, visual=True)
    2*ADD + SIN

    """
    from sympy import Integral, Symbol
    from sympy.simplify.radsimp import fraction
    from sympy.logic.boolalg import BooleanFunction

    expr = sympify(expr)
    # ... other code
```
### 3 - sympy/core/basic.py:

Start line: 1518, End line: 1547

```python
class Basic(with_metaclass(ManagedProperties)):

    def count_ops(self, visual=None):
        """wrapper for count_ops that returns the operation count."""
        from sympy import count_ops
        return count_ops(self, visual)

    def doit(self, **hints):
        """Evaluate objects that are not evaluated by default like limits,
           integrals, sums and products. All objects of this kind will be
           evaluated recursively, unless some species were excluded via 'hints'
           or unless the 'deep' hint was set to 'False'.

           >>> from sympy import Integral
           >>> from sympy.abc import x

           >>> 2*Integral(x, x)
           2*Integral(x, x)

           >>> (2*Integral(x, x)).doit()
           x**2

           >>> (2*Integral(x, x)).doit(deep=False)
           2*Integral(x, x)

        """
        if hints.get('deep', True):
            terms = [term.doit(**hints) if isinstance(term, Basic) else term
                                         for term in self.args]
            return self.func(*terms)
        else:
            return self
```
### 4 - sympy/core/expr.py:

Start line: 1034, End line: 1088

```python
class Expr(Basic, EvalfMixin):

    def removeO(self):
        """Removes the additive O(..) symbol if there is one"""
        return self

    def getO(self):
        """Returns the additive O(..) symbol if there is one, else None."""
        return None

    def getn(self):
        """
        Returns the order of the expression.

        The order is determined either from the O(...) term. If there
        is no O(...) term, it returns None.

        Examples
        ========

        >>> from sympy import O
        >>> from sympy.abc import x
        >>> (1 + x + O(x**2)).getn()
        2
        >>> (1 + x).getn()

        """
        from sympy import Dummy, Symbol
        o = self.getO()
        if o is None:
            return None
        elif o.is_Order:
            o = o.expr
            if o is S.One:
                return S.Zero
            if o.is_Symbol:
                return S.One
            if o.is_Pow:
                return o.args[1]
            if o.is_Mul:  # x**n*log(x)**n or x**n/log(x)**n
                for oi in o.args:
                    if oi.is_Symbol:
                        return S.One
                    if oi.is_Pow:
                        syms = oi.atoms(Symbol)
                        if len(syms) == 1:
                            x = syms.pop()
                            oi = oi.subs(x, Dummy('x', positive=True))
                            if oi.base.is_Symbol and oi.exp.is_Rational:
                                return abs(oi.exp)

        raise NotImplementedError('not sure of order of %s' % o)

    def count_ops(self, visual=None):
        """wrapper for count_ops that returns the operation count."""
        from .function import count_ops
        return count_ops(self, visual)
```
### 5 - sympy/core/operations.py:

Start line: 211, End line: 260

```python
class AssocOp(Basic):

    def _matches_commutative(self, expr, repl_dict={}, old=False):
        # ... other code
        while expr not in saw:
            saw.add(expr)
            expr_list = (self.identity,) + tuple(ordered(self.make_args(expr)))
            for last_op in reversed(expr_list):
                for w in reversed(wild_part):
                    d1 = w.matches(last_op, repl_dict)
                    if d1 is not None:
                        d2 = self.xreplace(d1).matches(expr, d1)
                        if d2 is not None:
                            return d2

            if i == 0:
                if self.is_Mul:
                    # make e**i look like Mul
                    if expr.is_Pow and expr.exp.is_Integer:
                        if expr.exp > 0:
                            expr = Mul(*[expr.base, expr.base**(expr.exp - 1)], evaluate=False)
                        else:
                            expr = Mul(*[1/expr.base, expr.base**(expr.exp + 1)], evaluate=False)
                        i += 1
                        continue

                elif self.is_Add:
                    # make i*e look like Add
                    c, e = expr.as_coeff_Mul()
                    if abs(c) > 1:
                        if c > 0:
                            expr = Add(*[e, (c - 1)*e], evaluate=False)
                        else:
                            expr = Add(*[-e, (c + 1)*e], evaluate=False)
                        i += 1
                        continue

                    # try collection on non-Wild symbols
                    from sympy.simplify.radsimp import collect
                    was = expr
                    did = set()
                    for w in reversed(wild_part):
                        c, w = w.as_coeff_mul(Wild)
                        free = c.free_symbols - did
                        if free:
                            did.update(free)
                            expr = collect(expr, free)
                    if expr != was:
                        i += 0
                        continue

                break  # if we didn't continue, there is nothing more to do

        return
```
### 6 - sympy/integrals/rubi/rubi.py:

Start line: 1, End line: 131

```python
from sympy.external import import_module
matchpy = import_module("matchpy")
from sympy.utilities.decorator import doctest_depends_on
import inspect, re

if matchpy:
    from matchpy import (Operation, CommutativeOperation, AssociativeOperation,
        ManyToOneReplacer, OneIdentityOperation, CustomConstraint)
    from matchpy.expressions.functions import register_operation_iterator, register_operation_factory
    from sympy import Pow, Add, Integral, Basic, Mul, S
    from sympy.functions import (log, sin, cos, tan, cot, csc, sec, sqrt, erf,
        exp, log, gamma, acosh, asinh, atanh, acoth, acsch, asech, cosh, sinh,
        tanh, coth, sech, csch, atan, acsc, asin, acot, acos, asec, fresnels,
        fresnelc, erfc, erfi)

    Operation.register(Integral)
    register_operation_iterator(Integral, lambda a: (a._args[0],) + a._args[1], lambda a: len((a._args[0],) + a._args[1]))

    Operation.register(Pow)
    OneIdentityOperation.register(Pow)
    register_operation_iterator(Pow, lambda a: a._args, lambda a: len(a._args))

    Operation.register(Add)
    OneIdentityOperation.register(Add)
    CommutativeOperation.register(Add)
    AssociativeOperation.register(Add)
    register_operation_iterator(Add, lambda a: a._args, lambda a: len(a._args))

    Operation.register(Mul)
    OneIdentityOperation.register(Mul)
    CommutativeOperation.register(Mul)
    AssociativeOperation.register(Mul)
    register_operation_iterator(Mul, lambda a: a._args, lambda a: len(a._args))

    Operation.register(exp)
    register_operation_iterator(exp, lambda a: a._args, lambda a: len(a._args))

    Operation.register(log)
    register_operation_iterator(log, lambda a: a._args, lambda a: len(a._args))

    Operation.register(gamma)
    register_operation_iterator(gamma, lambda a: a._args, lambda a: len(a._args))

    Operation.register(fresnels)
    register_operation_iterator(fresnels, lambda a: a._args, lambda a: len(a._args))

    Operation.register(fresnelc)
    register_operation_iterator(fresnelc, lambda a: a._args, lambda a: len(a._args))

    Operation.register(erfc)
    register_operation_iterator(erfc, lambda a: a._args, lambda a: len(a._args))

    Operation.register(erfi)
    register_operation_iterator(erfi, lambda a: a._args, lambda a: len(a._args))

    Operation.register(sin)
    register_operation_iterator(sin, lambda a: a._args, lambda a: len(a._args))

    Operation.register(cos)
    register_operation_iterator(cos, lambda a: a._args, lambda a: len(a._args))

    Operation.register(tan)
    register_operation_iterator(tan, lambda a: a._args, lambda a: len(a._args))

    Operation.register(cot)
    register_operation_iterator(cot, lambda a: a._args, lambda a: len(a._args))

    Operation.register(csc)
    register_operation_iterator(csc, lambda a: a._args, lambda a: len(a._args))

    Operation.register(sec)
    register_operation_iterator(sec, lambda a: a._args, lambda a: len(a._args))

    Operation.register(sinh)
    register_operation_iterator(sinh, lambda a: a._args, lambda a: len(a._args))

    Operation.register(cosh)
    register_operation_iterator(cosh, lambda a: a._args, lambda a: len(a._args))

    Operation.register(tanh)
    register_operation_iterator(tanh, lambda a: a._args, lambda a: len(a._args))

    Operation.register(coth)
    register_operation_iterator(coth, lambda a: a._args, lambda a: len(a._args))

    Operation.register(csch)
    register_operation_iterator(csch, lambda a: a._args, lambda a: len(a._args))

    Operation.register(sech)
    register_operation_iterator(sech, lambda a: a._args, lambda a: len(a._args))

    Operation.register(asin)
    register_operation_iterator(asin, lambda a: a._args, lambda a: len(a._args))

    Operation.register(acos)
    register_operation_iterator(acos, lambda a: a._args, lambda a: len(a._args))

    Operation.register(atan)
    register_operation_iterator(atan, lambda a: a._args, lambda a: len(a._args))

    Operation.register(acot)
    register_operation_iterator(acot, lambda a: a._args, lambda a: len(a._args))

    Operation.register(acsc)
    register_operation_iterator(acsc, lambda a: a._args, lambda a: len(a._args))

    Operation.register(asec)
    register_operation_iterator(asec, lambda a: a._args, lambda a: len(a._args))

    Operation.register(asinh)
    register_operation_iterator(asinh, lambda a: a._args, lambda a: len(a._args))

    Operation.register(acosh)
    register_operation_iterator(acosh, lambda a: a._args, lambda a: len(a._args))

    Operation.register(atanh)
    register_operation_iterator(atanh, lambda a: a._args, lambda a: len(a._args))

    Operation.register(acoth)
    register_operation_iterator(acoth, lambda a: a._args, lambda a: len(a._args))

    Operation.register(acsch)
    register_operation_iterator(acsch, lambda a: a._args, lambda a: len(a._args))

    Operation.register(asech)
    register_operation_iterator(asech, lambda a: a._args, lambda a: len(a._args))

    def sympy_op_factory(old_operation, new_operands, variable_name):
         return type(old_operation)(*new_operands)

    register_operation_factory(Basic, sympy_op_factory)
    # ... other code
```
### 7 - sympy/integrals/rubi/utility_function.py:

Start line: 1, End line: 158

```python
'''
Utility functions for Rubi integration.

See: http://www.apmaths.uwo.ca/~arich/IntegrationRules/PortableDocumentFiles/Integration%20utility%20functions.pdf
'''
from sympy.external import import_module
matchpy = import_module("matchpy")
from sympy.utilities.decorator import doctest_depends_on
from sympy.functions.elementary.integers import floor, frac
from sympy.functions import (log, sin, cos, tan, cot, csc, sec, sqrt, erf, gamma)
from sympy.functions.elementary.hyperbolic import acosh, asinh, atanh, acoth, acsch, asech, cosh, sinh, tanh, coth, sech, csch
from sympy.functions.elementary.trigonometric import atan, acsc, asin, acot, acos, asec
from sympy.polys.polytools import degree, Poly, quo, rem
from sympy.simplify.simplify import fraction, simplify, cancel
from sympy.core.sympify import sympify
from sympy.utilities.iterables import postorder_traversal
from sympy.functions.special.error_functions import fresnelc, fresnels, erfc, erfi
from sympy.functions.elementary.complexes import im, re, Abs
from sympy.core.exprtools import factor_terms
from sympy import (Basic, exp, polylog, N, Wild, factor, gcd, Sum, S, I, Mul,
    Add, hyper, symbols, sqf_list, sqf, Max, factorint, Min, sign, E,
    expand_trig, poly, apart, lcm, And, Pow, pi, zoo, oo, Integral, UnevaluatedExpr)
from mpmath import appellf1
from sympy.functions.special.elliptic_integrals import elliptic_f, elliptic_e, elliptic_pi
from sympy.utilities.iterables import flatten
from random import randint
from sympy.logic.boolalg import Or

if matchpy:
    from matchpy import Arity, Operation, CommutativeOperation, AssociativeOperation, OneIdentityOperation, CustomConstraint, Pattern, ReplacementRule, ManyToOneReplacer
    from matchpy.expressions.functions import register_operation_iterator, register_operation_factory
    from sympy.integrals.rubi.symbol import WC

    class UtilityOperator(Operation):
        name = 'UtilityOperator'
        arity = Arity.variadic
        commutative=False
        associative=True

    Operation.register(Integral)
    register_operation_iterator(Integral, lambda a: (a._args[0],) + a._args[1], lambda a: len((a._args[0],) + a._args[1]))

    Operation.register(Pow)
    OneIdentityOperation.register(Pow)
    register_operation_iterator(Pow, lambda a: a._args, lambda a: len(a._args))

    Operation.register(Add)
    OneIdentityOperation.register(Add)
    CommutativeOperation.register(Add)
    AssociativeOperation.register(Add)
    register_operation_iterator(Add, lambda a: a._args, lambda a: len(a._args))

    Operation.register(Mul)
    OneIdentityOperation.register(Mul)
    CommutativeOperation.register(Mul)
    AssociativeOperation.register(Mul)
    register_operation_iterator(Mul, lambda a: a._args, lambda a: len(a._args))

    Operation.register(exp)
    register_operation_iterator(exp, lambda a: a._args, lambda a: len(a._args))

    Operation.register(log)
    register_operation_iterator(log, lambda a: a._args, lambda a: len(a._args))

    Operation.register(gamma)
    register_operation_iterator(gamma, lambda a: a._args, lambda a: len(a._args))

    Operation.register(fresnels)
    register_operation_iterator(fresnels, lambda a: a._args, lambda a: len(a._args))

    Operation.register(fresnelc)
    register_operation_iterator(fresnelc, lambda a: a._args, lambda a: len(a._args))

    Operation.register(erfc)
    register_operation_iterator(erfc, lambda a: a._args, lambda a: len(a._args))

    Operation.register(erfi)
    register_operation_iterator(erfi, lambda a: a._args, lambda a: len(a._args))

    Operation.register(sin)
    register_operation_iterator(sin, lambda a: a._args, lambda a: len(a._args))

    Operation.register(cos)
    register_operation_iterator(cos, lambda a: a._args, lambda a: len(a._args))

    Operation.register(tan)
    register_operation_iterator(tan, lambda a: a._args, lambda a: len(a._args))

    Operation.register(cot)
    register_operation_iterator(cot, lambda a: a._args, lambda a: len(a._args))

    Operation.register(csc)
    register_operation_iterator(csc, lambda a: a._args, lambda a: len(a._args))

    Operation.register(sec)
    register_operation_iterator(sec, lambda a: a._args, lambda a: len(a._args))

    Operation.register(sinh)
    register_operation_iterator(sinh, lambda a: a._args, lambda a: len(a._args))

    Operation.register(cosh)
    register_operation_iterator(cosh, lambda a: a._args, lambda a: len(a._args))

    Operation.register(tanh)
    register_operation_iterator(tanh, lambda a: a._args, lambda a: len(a._args))

    Operation.register(coth)
    register_operation_iterator(coth, lambda a: a._args, lambda a: len(a._args))

    Operation.register(csch)
    register_operation_iterator(csch, lambda a: a._args, lambda a: len(a._args))

    Operation.register(sech)
    register_operation_iterator(sech, lambda a: a._args, lambda a: len(a._args))

    Operation.register(asin)
    register_operation_iterator(asin, lambda a: a._args, lambda a: len(a._args))

    Operation.register(acos)
    register_operation_iterator(acos, lambda a: a._args, lambda a: len(a._args))

    Operation.register(atan)
    register_operation_iterator(atan, lambda a: a._args, lambda a: len(a._args))

    Operation.register(acot)
    register_operation_iterator(acot, lambda a: a._args, lambda a: len(a._args))

    Operation.register(acsc)
    register_operation_iterator(acsc, lambda a: a._args, lambda a: len(a._args))

    Operation.register(asec)
    register_operation_iterator(asec, lambda a: a._args, lambda a: len(a._args))

    Operation.register(asinh)
    register_operation_iterator(asinh, lambda a: a._args, lambda a: len(a._args))

    Operation.register(acosh)
    register_operation_iterator(acosh, lambda a: a._args, lambda a: len(a._args))

    Operation.register(atanh)
    register_operation_iterator(atanh, lambda a: a._args, lambda a: len(a._args))

    Operation.register(acoth)
    register_operation_iterator(acoth, lambda a: a._args, lambda a: len(a._args))

    Operation.register(acsch)
    register_operation_iterator(acsch, lambda a: a._args, lambda a: len(a._args))

    Operation.register(asech)
    register_operation_iterator(asech, lambda a: a._args, lambda a: len(a._args))

    def sympy_op_factory(old_operation, new_operands, variable_name):
         return type(old_operation)(*new_operands)

    register_operation_factory(Basic, sympy_op_factory)

    A_, B_, C_, F_, G_, a_, b_, c_, d_, e_, f_, g_, h_, i_, j_, k_, l_, m_, n_, p_, q_, r_, t_, u_, v_, s_, w_, x_, z_ = [WC(i) for i in 'ABCFGabcdefghijklmnpqrtuvswxz']
    a, b, c, d, e = symbols('a b c d e')
```
### 8 - sympy/integrals/rubi/utility_function.py:

Start line: 6644, End line: 6700

```python
@doctest_depends_on(modules=('matchpy',))
def _TrigSimplifyAux():
    # ... other code
    rule27 = ReplacementRule(pattern27, lambda n, a, c, p, m, u, v, b : Mul(u, Pow(Csc(v), Add(m, Mul(n, p))), Pow(Add(c, Mul(b, Pow(Cos(v), n)), Mul(a, Pow(Sin(v), n))), p)))
    replacer.add(rule27)

    pattern28 = Pattern(UtilityOperator(Mul(WC('u', S(1)), Pow(Add(Mul(Pow(csc(v_), WC('m', S(1))), WC('a', S(1))), Mul(WC('b', S(1)), Pow(sin(v_), WC('n', S(1))))), WC('p', S(1))))), CustomConstraint(lambda n, m: IntegersQ(m, n)))
    rule28 = ReplacementRule(pattern28, lambda n, a, p, m, u, v, b : If(And(ZeroQ(Add(m, n, S(-2))), ZeroQ(Add(a, b))), Mul(u, Pow(Mul(a, Mul(Pow(Cos(v), S('2')), Pow(Pow(Sin(v), m), S(-1)))), p)), Mul(u, Pow(Mul(Add(a, Mul(b, Pow(Sin(v), Add(m, n)))), Pow(Pow(Sin(v), m), S(-1))), p))))
    replacer.add(rule28)

    pattern29 = Pattern(UtilityOperator(Mul(WC('u', S(1)), Pow(Add(Mul(Pow(cos(v_), WC('n', S(1))), WC('b', S(1))), Mul(WC('a', S(1)), Pow(sec(v_), WC('m', S(1))))), WC('p', S(1))))), CustomConstraint(lambda n, m: IntegersQ(m, n)))
    rule29 = ReplacementRule(pattern29, lambda n, a, p, m, u, v, b : If(And(ZeroQ(Add(m, n, S(-2))), ZeroQ(Add(a, b))), Mul(u, Pow(Mul(a, Mul(Pow(Sin(v), S('2')), Pow(Pow(Cos(v), m), S(-1)))), p)), Mul(u, Pow(Mul(Add(a, Mul(b, Pow(Cos(v), Add(m, n)))), Pow(Pow(Cos(v), m), S(-1))), p))))
    replacer.add(rule29)

    pattern30 = Pattern(UtilityOperator(u_))
    rule30 = ReplacementRule(pattern30, lambda u : u)
    replacer.add(rule30)

    return replacer

@doctest_depends_on(modules=('matchpy',))
def TrigSimplifyAux(expr):
    return TrigSimplifyAux_replacer.replace(UtilityOperator(expr))

def Cancel(expr):
    return cancel(expr)

def Part(lst, i):
    if isinstance(lst, list):
        return lst[i - 1] # Python list indexing starts 1 unit below Mathematica
    elif AtomQ(lst):
        return lst
    return lst.args[i-1]

def PolyLog(n, p, z=None):
    return polylog(n, p)

def D(f, x):
    return f.diff(x)

def Dist(u, v, x):
    return Mul(u, v)

def PureFunctionOfCothQ(u, v, x):
    # If u is a pure function of Coth[v], PureFunctionOfCothQ[u,v,x] returns True;
    if AtomQ(u):
        return u != x
    elif CalculusQ(u):
        return False
    elif HyperbolicQ(u) and ZeroQ(u.args[0] - v):
        return CothQ(u)
    return all(PureFunctionOfCothQ(i, v, x) for i in u.args)

if matchpy:
    TrigSimplifyAux_replacer = _TrigSimplifyAux()
    SimplifyAntiderivative_replacer = _SimplifyAntiderivative()
    SimplifyAntiderivativeSum_replacer = _SimplifyAntiderivativeSum()
    FixSimplify_replacer = _FixSimplify()
    SimpFixFactor_replacer = _SimpFixFactor()
```
### 9 - sympy/core/expr.py:

Start line: 2001, End line: 2082

```python
class Expr(Basic, EvalfMixin):

    def extract_multiplicatively(self, c):
        # ... other code
        if self.is_Number:
            if self is S.Infinity:
                if c.is_positive:
                    return S.Infinity
            elif self is S.NegativeInfinity:
                if c.is_negative:
                    return S.Infinity
                elif c.is_positive:
                    return S.NegativeInfinity
            elif self is S.ComplexInfinity:
                if not c.is_zero:
                    return S.ComplexInfinity
            elif self.is_Integer:
                if not quotient.is_Integer:
                    return None
                elif self.is_positive and quotient.is_negative:
                    return None
                else:
                    return quotient
            elif self.is_Rational:
                if not quotient.is_Rational:
                    return None
                elif self.is_positive and quotient.is_negative:
                    return None
                else:
                    return quotient
            elif self.is_Float:
                if not quotient.is_Float:
                    return None
                elif self.is_positive and quotient.is_negative:
                    return None
                else:
                    return quotient
        elif self.is_NumberSymbol or self.is_Symbol or self is S.ImaginaryUnit:
            if quotient.is_Mul and len(quotient.args) == 2:
                if quotient.args[0].is_Integer and quotient.args[0].is_positive and quotient.args[1] == self:
                    return quotient
            elif quotient.is_Integer and c.is_Number:
                return quotient
        elif self.is_Add:
            cs, ps = self.primitive()
            # assert cs >= 1
            if c.is_Number and c is not S.NegativeOne:
                # assert c != 1 (handled at top)
                if cs is not S.One:
                    if c.is_negative:
                        xc = -(cs.extract_multiplicatively(-c))
                    else:
                        xc = cs.extract_multiplicatively(c)
                    if xc is not None:
                        return xc*ps  # rely on 2-arg Mul to restore Add
                return  # |c| != 1 can only be extracted from cs
            if c == ps:
                return cs
            # check args of ps
            newargs = []
            for arg in ps.args:
                newarg = arg.extract_multiplicatively(c)
                if newarg is None:
                    return  # all or nothing
                newargs.append(newarg)
            # args should be in same order so use unevaluated return
            if cs is not S.One:
                return Add._from_args([cs*t for t in newargs])
            else:
                return Add._from_args(newargs)
        elif self.is_Mul:
            args = list(self.args)
            for i, arg in enumerate(args):
                newarg = arg.extract_multiplicatively(c)
                if newarg is not None:
                    args[i] = newarg
                    return Mul(*args)
        elif self.is_Pow:
            if c.is_Pow and c.base == self.base:
                new_exp = self.exp.extract_additively(c.exp)
                if new_exp is not None:
                    return self.base ** (new_exp)
            elif c == self.base:
                new_exp = self.exp.extract_additively(1)
                if new_exp is not None:
                    return self.base ** (new_exp)
```
### 10 - sympy/core/expr.py:

Start line: 92, End line: 150

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
        return Mul(S.NegativeOne, self)

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
