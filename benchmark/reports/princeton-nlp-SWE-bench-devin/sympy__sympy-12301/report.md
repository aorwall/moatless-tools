# sympy__sympy-12301

| **sympy/sympy** | `5155b7641fa389e10aeb5cfebcbefba02cb9221c` |
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
diff --git a/sympy/simplify/cse_main.py b/sympy/simplify/cse_main.py
--- a/sympy/simplify/cse_main.py
+++ b/sympy/simplify/cse_main.py
@@ -310,10 +310,18 @@ def update(k):
             # remove it
             if Func is Add:
                 take = min(func_dicts[k][i] for i in com_dict)
-                com_func_take = Mul(take, from_dict(com_dict), evaluate=False)
+                _sum = from_dict(com_dict)
+                if take == 1:
+                    com_func_take = _sum
+                else:
+                    com_func_take = Mul(take, _sum, evaluate=False)
             else:
                 take = igcd(*[func_dicts[k][i] for i in com_dict])
-                com_func_take = Pow(from_dict(com_dict), take, evaluate=False)
+                base = from_dict(com_dict)
+                if take == 1:
+                    com_func_take = base
+                else:
+                    com_func_take = Pow(base, take, evaluate=False)
             for di in com_dict:
                 func_dicts[k][di] -= take*com_dict[di]
             # compute the remaining expression
@@ -546,23 +554,12 @@ def _rebuild(expr):
     #     R = [(x0, d + f), (x1, b + d)]
     #     C = [e + x0 + x1, g + x0 + x1, a + c + d + f + g]
     # but the args of C[-1] should not be `(a + c, d + f + g)`
-    nested = [[i for i in f.args if isinstance(i, f.func)] for f in exprs]
     for i in range(len(exprs)):
         F = reduced_exprs[i].func
         if not (F is Mul or F is Add):
             continue
-        nested = [a for a in exprs[i].args if isinstance(a, F)]
-        args = []
-        for a in reduced_exprs[i].args:
-            if isinstance(a, F):
-                for ai in a.args:
-                    if isinstance(ai, F) and ai not in nested:
-                        args.extend(ai.args)
-                    else:
-                        args.append(ai)
-            else:
-                args.append(a)
-        reduced_exprs[i] = F(*args)
+        if any(isinstance(a, F) for a in reduced_exprs[i].args):
+            reduced_exprs[i] = F(*reduced_exprs[i].args)
 
     return replacements, reduced_exprs
 

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sympy/simplify/cse_main.py | 313 | 316 | - | - | -
| sympy/simplify/cse_main.py | 549 | 565 | - | - | -


## Problem Statement

```
Test failure in Travis
\`\`\`
______________ sympy/simplify/tests/test_cse.py:test_issue_11230 _______________
  File "/home/travis/virtualenv/python3.5.2/lib/python3.5/site-packages/sympy-1.0.1.dev0-py3.5.egg/sympy/simplify/tests/test_cse.py", line 433, in test_issue_11230
    assert not any(i.is_Mul for a in C for i in a.args)
AssertionError
\`\`\`

I was able to reproduce this locally on a 64-bit system running Ubuntu 16.04.

How to reproduce:

\`\`\` bash
conda create -n test_sympy python=3.5 matplotlib numpy scipy pip llvmlite
source activate test_sympy
python
\`\`\`

\`\`\` Python
>>> import os
>>> os.environ['PYTHONHASHSEED'] = '736538842'
>>> import sympy
>>> sympy.test(split='4/4', seed=57601301)
\`\`\`


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 release/fabfile.py | 938 | 959| 193 | 193 | 11132 | 
| 2 | 2 sympy/utilities/runtests.py | 1028 | 1049| 268 | 461 | 29815 | 
| 3 | 3 sympy/conftest.py | 1 | 27| 189 | 650 | 30399 | 
| 4 | 3 sympy/utilities/runtests.py | 1020 | 1026| 130 | 780 | 30399 | 
| 5 | 3 sympy/utilities/runtests.py | 1051 | 1136| 637 | 1417 | 30399 | 
| 6 | 3 release/fabfile.py | 239 | 258| 222 | 1639 | 30399 | 
| 7 | 3 sympy/utilities/runtests.py | 1138 | 1180| 367 | 2006 | 30399 | 
| 8 | 3 sympy/utilities/runtests.py | 1243 | 1325| 756 | 2762 | 30399 | 
| 9 | 4 setup.py | 331 | 371| 366 | 3128 | 33340 | 
| 10 | 4 sympy/utilities/runtests.py | 1 | 49| 254 | 3382 | 33340 | 
| 11 | 4 sympy/utilities/runtests.py | 435 | 459| 235 | 3617 | 33340 | 
| 12 | 4 sympy/utilities/runtests.py | 67 | 82| 110 | 3727 | 33340 | 
| 13 | 4 setup.py | 261 | 329| 655 | 4382 | 33340 | 
| 14 | 5 bin/get_sympy.py | 1 | 18| 106 | 4488 | 33447 | 
| 15 | 5 setup.py | 69 | 138| 607 | 5095 | 33447 | 
| 16 | 6 examples/all.py | 1 | 87| 549 | 5644 | 34985 | 
| 17 | 7 sympy/__init__.py | 1 | 93| 665 | 6309 | 35650 | 
| 18 | 7 sympy/utilities/runtests.py | 104 | 133| 232 | 6541 | 35650 | 
| 19 | 7 sympy/utilities/runtests.py | 997 | 1018| 156 | 6697 | 35650 | 
| 20 | 7 sympy/utilities/runtests.py | 704 | 806| 879 | 7576 | 35650 | 
| 21 | 7 sympy/utilities/runtests.py | 1527 | 1555| 291 | 7867 | 35650 | 
| 22 | 7 sympy/utilities/runtests.py | 970 | 995| 239 | 8106 | 35650 | 
| 23 | 7 sympy/utilities/runtests.py | 921 | 967| 405 | 8511 | 35650 | 
| 24 | 7 sympy/utilities/runtests.py | 1360 | 1445| 630 | 9141 | 35650 | 
| 25 | 7 setup.py | 212 | 231| 148 | 9289 | 35650 | 
| 26 | 8 bin/coverage_doctest.py | 536 | 558| 286 | 9575 | 41122 | 
| 27 | 8 bin/coverage_doctest.py | 1 | 69| 437 | 10012 | 41122 | 
| 28 | 8 bin/coverage_doctest.py | 560 | 651| 920 | 10932 | 41122 | 
| 29 | 8 sympy/utilities/runtests.py | 2148 | 2231| 650 | 11582 | 41122 | 
| 30 | 8 sympy/utilities/runtests.py | 208 | 224| 188 | 11770 | 41122 | 
| 31 | 8 sympy/utilities/runtests.py | 540 | 612| 651 | 12421 | 41122 | 
| 32 | 8 sympy/utilities/runtests.py | 2027 | 2062| 408 | 12829 | 41122 | 
| 33 | 9 bin/generate_test_list.py | 36 | 72| 313 | 13142 | 41656 | 
| 34 | 10 sympy/core/backend.py | 1 | 24| 341 | 13483 | 41998 | 
| 35 | 11 sympy/core/benchmarks/bench_basic.py | 1 | 18| 0 | 13483 | 42071 | 
| 36 | 12 sympy/series/benchmarks/bench_limit.py | 1 | 10| 0 | 13483 | 42115 | 
| 37 | 12 sympy/utilities/runtests.py | 1218 | 1241| 148 | 13631 | 42115 | 
| 38 | 12 sympy/utilities/runtests.py | 615 | 703| 768 | 14399 | 42115 | 
| 39 | 12 sympy/utilities/runtests.py | 300 | 434| 1131 | 15530 | 42115 | 
| 40 | 13 sympy/core/__init__.py | 1 | 35| 353 | 15883 | 42469 | 
| 41 | 13 sympy/conftest.py | 30 | 44| 152 | 16035 | 42469 | 
| 42 | 14 sympy/core/numbers.py | 1753 | 1778| 157 | 16192 | 69929 | 
| 43 | 15 sympy/release.py | 1 | 2| 0 | 16192 | 69941 | 
| 44 | 15 bin/generate_test_list.py | 1 | 33| 220 | 16412 | 69941 | 
| 45 | 15 sympy/utilities/runtests.py | 1204 | 1215| 129 | 16541 | 69941 | 
| 46 | 15 sympy/utilities/runtests.py | 1182 | 1202| 158 | 16699 | 69941 | 
| 47 | 15 release/fabfile.py | 360 | 435| 767 | 17466 | 69941 | 
| 48 | 16 sympy/assumptions/__init__.py | 1 | 4| 0 | 17466 | 69978 | 
| 49 | 17 sympy/series/sequences.py | 1 | 19| 175 | 17641 | 77291 | 
| 50 | 17 release/fabfile.py | 220 | 237| 128 | 17769 | 77291 | 
| 51 | 18 sympy/sets/__init__.py | 1 | 6| 0 | 17769 | 77352 | 
| 52 | 19 sympy/core/benchmarks/bench_assumptions.py | 1 | 15| 0 | 17769 | 77413 | 
| 53 | 19 sympy/utilities/runtests.py | 462 | 537| 634 | 18403 | 77413 | 
| 54 | 20 sympy/core/expr.py | 1 | 12| 113 | 18516 | 105376 | 
| 55 | 21 sympy/core/benchmarks/bench_expand.py | 1 | 26| 162 | 18678 | 105538 | 
| 56 | 22 sympy/series/benchmarks/bench_order.py | 1 | 11| 0 | 18678 | 105601 | 
| 57 | 22 sympy/utilities/runtests.py | 227 | 297| 587 | 19265 | 105601 | 
| 58 | 23 sympy/functions/special/benchmarks/bench_special.py | 1 | 11| 0 | 19265 | 105661 | 
| 59 | 24 sympy/solvers/benchmarks/bench_solvers.py | 1 | 14| 0 | 19265 | 105754 | 
| 60 | 25 sympy/core/benchmarks/bench_sympify.py | 1 | 14| 0 | 19265 | 105808 | 
| 61 | 26 sympy/codegen/__init__.py | 1 | 2| 0 | 19265 | 105822 | 
| 62 | 27 sympy/functions/elementary/miscellaneous.py | 1 | 19| 182 | 19447 | 110974 | 
| 63 | 27 sympy/core/numbers.py | 1 | 35| 294 | 19741 | 110974 | 
| 64 | 27 sympy/utilities/runtests.py | 1704 | 1737| 423 | 20164 | 110974 | 
| 65 | 28 sympy/simplify/trigsimp.py | 1 | 24| 239 | 20403 | 123036 | 
| 66 | 29 sympy/functions/elementary/benchmarks/bench_exp.py | 1 | 13| 0 | 20403 | 123091 | 
| 67 | 30 sympy/unify/__init__.py | 1 | 10| 0 | 20403 | 123155 | 


## Missing Patch Files

 * 1: sympy/simplify/cse_main.py

### Hint

```
Log is here, https://travis-ci.org/sympy/sympy/jobs/163790187
Permanent link, https://gist.github.com/isuruf/9410c21df1be658d168727018007a63a

ping @smichr 

Seeing jobs failing frequently now
https://travis-ci.org/sympy/sympy/jobs/164977570
https://travis-ci.org/sympy/sympy/jobs/164880234

It seems that the failure is generated in the following way.

\`\`\`
>>> from sympy import cse
>>> from sympy.abc import a, c, i, g, l, m
>>> p = [c*g*i**2*m, a*c*i*l*m, g*i**2*l*m]
>>> cse(p)
([(x0, g*i), (x1, i*m)], [c*x0*x1, a*l*(c*i*m), l*x0*x1])
\`\`\`

`c*i*m` is recognized as a common factor of the first two expressions, and is marked as a candidate by writing it in the form `(c*i*m)**1` by [`update`](https://github.com/sympy/sympy/blob/master/sympy/simplify/cse_main.py#L316). It is later abandoned as a part of `c*g*i**2*m` and is left alone in `a*c*i*l*m`. When its expression tree is rebuilt, the following results.

\`\`\`
>>> from sympy import Mul, Pow
>>> Mul(a, l, Pow(c*i*m, 1, evaluate=False))
a*l*(c*i*m)
\`\`\`

(`x0*x1 = g*i**2*m` is not recognized as a common subexpression for some reason.)

Do we have a fix for this? If not, let's revert the PR. @smichr 

@smichr, ping.

Just seeing this now. I thought that there was a line of code to remove exponents of "`1". I may be able to look at this tomorrow, but it's more realistic to expect a delay up until Friday. I'll see what I can do but if this is really a hassle for tests it can be reverted and I'll try again later.

Thanks for looking into this. We can wait for a few more days.

I am working on this...perhaps can finish before Monday. I am getting 3 different possibilities, @jksuom for the test expression you gave:

\`\`\`
============================= test process starts =============================
executable:         C:\Python27\python.exe  (2.7.7-final-0) [CPython]
architecture:       32-bit
cache:              yes
ground types:       python
random seed:        36014997
hash randomization: on (PYTHONHASHSEED=643787914)

sympy\simplify\tests\test_cse.py[1] ([(x0, g*i), (x1, i*m)], [c*x0*x1, a*c*l*x1, l*x0*x1])
.                                      [OK]

================== tests finished: 1 passed, in 0.48 seconds ==================
rerun 3
============================= test process starts =============================
executable:         C:\Python27\python.exe  (2.7.7-final-0) [CPython]
architecture:       32-bit
cache:              yes
ground types:       python
random seed:        30131441
hash randomization: on (PYTHONHASHSEED=2864749239)

sympy\simplify\tests\test_cse.py[1] ([(x0, g*i), (x1, c*i*m)], [x0*x1, a*l*x1, i*l*m*x0])
.                                      [OK]

================== tests finished: 1 passed, in 0.37 seconds ==================
rerun 4
============================= test process starts =============================
executable:         C:\Python27\python.exe  (2.7.7-final-0) [CPython]
architecture:       32-bit
cache:              yes
ground types:       python
random seed:        20393357
hash randomization: on (PYTHONHASHSEED=1323273449)

sympy\simplify\tests\test_cse.py[1] ([(x0, g*i**2*m)], [c*x0, (c*i)*(a*l*m), l*x0])
.                                      [OK]
\`\`\`

So it looks like I have to make this canonical, too.
```

## Patch

```diff
diff --git a/sympy/simplify/cse_main.py b/sympy/simplify/cse_main.py
--- a/sympy/simplify/cse_main.py
+++ b/sympy/simplify/cse_main.py
@@ -310,10 +310,18 @@ def update(k):
             # remove it
             if Func is Add:
                 take = min(func_dicts[k][i] for i in com_dict)
-                com_func_take = Mul(take, from_dict(com_dict), evaluate=False)
+                _sum = from_dict(com_dict)
+                if take == 1:
+                    com_func_take = _sum
+                else:
+                    com_func_take = Mul(take, _sum, evaluate=False)
             else:
                 take = igcd(*[func_dicts[k][i] for i in com_dict])
-                com_func_take = Pow(from_dict(com_dict), take, evaluate=False)
+                base = from_dict(com_dict)
+                if take == 1:
+                    com_func_take = base
+                else:
+                    com_func_take = Pow(base, take, evaluate=False)
             for di in com_dict:
                 func_dicts[k][di] -= take*com_dict[di]
             # compute the remaining expression
@@ -546,23 +554,12 @@ def _rebuild(expr):
     #     R = [(x0, d + f), (x1, b + d)]
     #     C = [e + x0 + x1, g + x0 + x1, a + c + d + f + g]
     # but the args of C[-1] should not be `(a + c, d + f + g)`
-    nested = [[i for i in f.args if isinstance(i, f.func)] for f in exprs]
     for i in range(len(exprs)):
         F = reduced_exprs[i].func
         if not (F is Mul or F is Add):
             continue
-        nested = [a for a in exprs[i].args if isinstance(a, F)]
-        args = []
-        for a in reduced_exprs[i].args:
-            if isinstance(a, F):
-                for ai in a.args:
-                    if isinstance(ai, F) and ai not in nested:
-                        args.extend(ai.args)
-                    else:
-                        args.append(ai)
-            else:
-                args.append(a)
-        reduced_exprs[i] = F(*args)
+        if any(isinstance(a, F) for a in reduced_exprs[i].args):
+            reduced_exprs[i] = F(*reduced_exprs[i].args)
 
     return replacements, reduced_exprs
 

```

## Test Patch

```diff
diff --git a/sympy/simplify/tests/test_cse.py b/sympy/simplify/tests/test_cse.py
--- a/sympy/simplify/tests/test_cse.py
+++ b/sympy/simplify/tests/test_cse.py
@@ -422,6 +422,13 @@ def test_issue_8891():
 
 
 def test_issue_11230():
+    # a specific test that always failed
+    a, b, f, k, l, i = symbols('a b f k l i')
+    p = [a*b*f*k*l, a*i*k**2*l, f*i*k**2*l]
+    R, C = cse(p)
+    assert not any(i.is_Mul for a in C for i in a.args)
+
+    # random tests for the issue
     from random import choice
     from sympy.core.function import expand_mul
     s = symbols('a:m')

```


## Code snippets

### 1 - release/fabfile.py:

Start line: 938, End line: 959

```python
@task
def test_pypi(release='2'):
    """
    Test that the sympy can be pip installed, and that sympy imports in the
    install.
    """
    # This function is similar to test_tarball()

    version = get_sympy_version()

    release = str(release)

    if release not in {'2', '3'}: # TODO: Add win32
        raise ValueError("release must be one of '2', '3', not %s" % release)

    venv = "/home/vagrant/repos/test-{release}-pip-virtualenv".format(release=release)

    with use_venv(release):
        make_virtualenv(venv)
        with virtualenv(venv):
            run("pip install sympy")
            run('python -c "import sympy; assert sympy.__version__ == \'{version}\'"'.format(version=version))
```
### 2 - sympy/utilities/runtests.py:

Start line: 1028, End line: 1049

```python
class SymPyTests(object):

    def _enhance_asserts(self, source):
        # ... other code

        class Transform(NodeTransformer):
            def visit_Assert(self, stmt):
                if isinstance(stmt.test, Compare):
                    compare = stmt.test
                    values = [compare.left] + compare.comparators
                    names = [ "_%s" % i for i, _ in enumerate(values) ]
                    names_store = [ Name(n, Store()) for n in names ]
                    names_load = [ Name(n, Load()) for n in names ]
                    target = Tuple(names_store, Store())
                    value = Tuple(values, Load())
                    assign = Assign([target], value)
                    new_compare = Compare(names_load[0], compare.ops, names_load[1:])
                    msg_format = "\n%s " + "\n%s ".join([ ops[op.__class__.__name__] for op in compare.ops ]) + "\n%s"
                    msg = BinOp(Str(msg_format), Mod(), Tuple(names_load, Load()))
                    test = Assert(new_compare, msg, lineno=stmt.lineno, col_offset=stmt.col_offset)
                    return [assign, test]
                else:
                    return stmt

        tree = parse(source)
        new_tree = Transform().visit(tree)
        return fix_missing_locations(new_tree)
```
### 3 - sympy/conftest.py:

Start line: 1, End line: 27

```python
from __future__ import print_function, division

import sys
sys._running_pytest = True
from distutils.version import LooseVersion as V

import pytest
from sympy.core.cache import clear_cache
import re

sp = re.compile(r'([0-9]+)/([1-9][0-9]*)')

def process_split(session, config, items):
    split = config.getoption("--split")
    if not split:
        return
    m = sp.match(split)
    if not m:
        raise ValueError("split must be a string of the form a/b "
                         "where a and b are ints.")
    i, t = map(int, m.groups())
    start, end = (i-1)*len(items)//t, i*len(items)//t

    if i < t:
        # remove elements from end of list first
        del items[end:]
    del items[:start]
```
### 4 - sympy/utilities/runtests.py:

Start line: 1020, End line: 1026

```python
class SymPyTests(object):

    def _enhance_asserts(self, source):
        from ast import (NodeTransformer, Compare, Name, Store, Load, Tuple,
            Assign, BinOp, Str, Mod, Assert, parse, fix_missing_locations)

        ops = {"Eq": '==', "NotEq": '!=', "Lt": '<', "LtE": '<=',
                "Gt": '>', "GtE": '>=', "Is": 'is', "IsNot": 'is not',
                "In": 'in', "NotIn": 'not in'}
        # ... other code
```
### 5 - sympy/utilities/runtests.py:

Start line: 1051, End line: 1136

```python
class SymPyTests(object):

    def test_file(self, filename, sort=True, timeout=False, slow=False, enhance_asserts=False):
        reporter = self._reporter
        funcs = []
        try:
            gl = {'__file__': filename}
            try:
                if PY3:
                    open_file = lambda: open(filename, encoding="utf8")
                else:
                    open_file = lambda: open(filename)

                with open_file() as f:
                    source = f.read()
                    if self._kw:
                        for l in source.splitlines():
                            if l.lstrip().startswith('def '):
                                if any(l.find(k) != -1 for k in self._kw):
                                    break
                        else:
                            return

                if enhance_asserts:
                    try:
                        source = self._enhance_asserts(source)
                    except ImportError:
                        pass

                code = compile(source, filename, "exec")
                exec_(code, gl)
            except (SystemExit, KeyboardInterrupt):
                raise
            except ImportError:
                reporter.import_error(filename, sys.exc_info())
                return
            except Exception:
                reporter.test_exception(sys.exc_info())

            clear_cache()
            self._count += 1
            random.seed(self._seed)
            disabled = gl.get("disabled", False)
            if not disabled:
                # we need to filter only those functions that begin with 'test_'
                # We have to be careful about decorated functions. As long as
                # the decorator uses functools.wraps, we can detect it.
                funcs = []
                for f in gl:
                    if (f.startswith("test_") and (inspect.isfunction(gl[f])
                        or inspect.ismethod(gl[f]))):
                        func = gl[f]
                        # Handle multiple decorators
                        while hasattr(func, '__wrapped__'):
                            func = func.__wrapped__

                        if inspect.getsourcefile(func) == filename:
                            funcs.append(gl[f])
                if slow:
                    funcs = [f for f in funcs if getattr(f, '_slow', False)]
                # Sorting of XFAILed functions isn't fixed yet :-(
                funcs.sort(key=lambda x: inspect.getsourcelines(x)[1])
                i = 0
                while i < len(funcs):
                    if inspect.isgeneratorfunction(funcs[i]):
                    # some tests can be generators, that return the actual
                    # test functions. We unpack it below:
                        f = funcs.pop(i)
                        for fg in f():
                            func = fg[0]
                            args = fg[1:]
                            fgw = lambda: func(*args)
                            funcs.insert(i, fgw)
                            i += 1
                    else:
                        i += 1
                # drop functions that are not selected with the keyword expression:
                funcs = [x for x in funcs if self.matches(x)]

            if not funcs:
                return
        except Exception:
            reporter.entering_filename(filename, len(funcs))
            raise

        reporter.entering_filename(filename, len(funcs))
        if not sort:
            random.shuffle(funcs)
        # ... other code
```
### 6 - release/fabfile.py:

Start line: 239, End line: 258

```python
@task
def test_tarball(release='2'):
    """
    Test that the tarball can be unpacked and installed, and that sympy
    imports in the install.
    """
    if release not in {'2', '3'}: # TODO: Add win32
        raise ValueError("release must be one of '2', '3', not %s" % release)

    venv = "/home/vagrant/repos/test-{release}-virtualenv".format(release=release)
    tarball_formatter_dict = tarball_formatter()

    with use_venv(release):
        make_virtualenv(venv)
        with virtualenv(venv):
            run("cp /vagrant/release/{source} releasetar.tar".format(**tarball_formatter_dict))
            run("tar xvf releasetar.tar")
            with cd("/home/vagrant/{source-orig-notar}".format(**tarball_formatter_dict)):
                run("python setup.py install")
                run('python -c "import sympy; print(sympy.__version__)"')
```
### 7 - sympy/utilities/runtests.py:

Start line: 1138, End line: 1180

```python
class SymPyTests(object):

    def test_file(self, filename, sort=True, timeout=False, slow=False, enhance_asserts=False):
        # ... other code

        for f in funcs:
            start = time.time()
            reporter.entering_test(f)
            try:
                if getattr(f, '_slow', False) and not slow:
                    raise Skipped("Slow")
                if timeout:
                    self._timeout(f, timeout)
                else:
                    random.seed(self._seed)
                    f()
            except KeyboardInterrupt:
                if getattr(f, '_slow', False):
                    reporter.test_skip("KeyboardInterrupt")
                else:
                    raise
            except Exception:
                if timeout:
                    signal.alarm(0)  # Disable the alarm. It could not be handled before.
                t, v, tr = sys.exc_info()
                if t is AssertionError:
                    reporter.test_fail((t, v, tr))
                    if self._post_mortem:
                        pdb.post_mortem(tr)
                elif t.__name__ == "Skipped":
                    reporter.test_skip(v)
                elif t.__name__ == "XFail":
                    reporter.test_xfail()
                elif t.__name__ == "XPass":
                    reporter.test_xpass(v)
                else:
                    reporter.test_exception((t, v, tr))
                    if self._post_mortem:
                        pdb.post_mortem(tr)
            else:
                reporter.test_pass()
            taken = time.time() - start
            if taken > self._slow_threshold:
                reporter.slow_test_functions.append((f.__name__, taken))
            if getattr(f, '_slow', False) and slow:
                if taken < self._fast_threshold:
                    reporter.fast_test_functions.append((f.__name__, taken))
        reporter.leaving_filename()
```
### 8 - sympy/utilities/runtests.py:

Start line: 1243, End line: 1325

```python
class SymPyDocTests(object):

    def test_file(self, filename):
        clear_cache()

        from sympy.core.compatibility import StringIO

        rel_name = filename[len(self._root_dir) + 1:]
        dirname, file = os.path.split(filename)
        module = rel_name.replace(os.sep, '.')[:-3]

        if rel_name.startswith("examples"):
            # Examples files do not have __init__.py files,
            # So we have to temporarily extend sys.path to import them
            sys.path.insert(0, dirname)
            module = file[:-3]  # remove ".py"
        setup_pprint()
        try:
            module = pdoctest._normalize_module(module)
            tests = SymPyDocTestFinder().find(module)
        except (SystemExit, KeyboardInterrupt):
            raise
        except ImportError:
            self._reporter.import_error(filename, sys.exc_info())
            return
        finally:
            if rel_name.startswith("examples"):
                del sys.path[0]

        tests = [test for test in tests if len(test.examples) > 0]
        # By default tests are sorted by alphabetical order by function name.
        # We sort by line number so one can edit the file sequentially from
        # bottom to top. However, if there are decorated functions, their line
        # numbers will be too large and for now one must just search for these
        # by text and function name.
        tests.sort(key=lambda x: -x.lineno)

        if not tests:
            return
        self._reporter.entering_filename(filename, len(tests))
        for test in tests:
            assert len(test.examples) != 0

            # check if there are external dependencies which need to be met
            if '_doctest_depends_on' in test.globs:
                has_dependencies = self._process_dependencies(test.globs['_doctest_depends_on'])
                if has_dependencies is not True:
                    # has_dependencies is either True or a message
                    self._reporter.test_skip(v="\n" + has_dependencies)
                    continue

            if self._reporter._verbose:
                self._reporter.write("\n{} ".format(test.name))

            runner = SymPyDocTestRunner(optionflags=pdoctest.ELLIPSIS |
                    pdoctest.NORMALIZE_WHITESPACE |
                    pdoctest.IGNORE_EXCEPTION_DETAIL)
            runner._checker = SymPyOutputChecker()
            old = sys.stdout
            new = StringIO()
            sys.stdout = new
            # If the testing is normal, the doctests get importing magic to
            # provide the global namespace. If not normal (the default) then
            # then must run on their own; all imports must be explicit within
            # a function's docstring. Once imported that import will be
            # available to the rest of the tests in a given function's
            # docstring (unless clear_globs=True below).
            if not self._normal:
                test.globs = {}
                # if this is uncommented then all the test would get is what
                # comes by default with a "from sympy import *"
                #exec('from sympy import *') in test.globs
            test.globs['print_function'] = print_function
            try:
                f, t = runner.run(test, compileflags=future_flags,
                                  out=new.write, clear_globs=False)
            except KeyboardInterrupt:
                raise
            finally:
                sys.stdout = old
            if f > 0:
                self._reporter.doctest_fail(test.name, new.getvalue())
            else:
                self._reporter.test_pass()
        self._reporter.leaving_filename()
```
### 9 - setup.py:

Start line: 331, End line: 371

```python
if __name__ == '__main__':
    setup(name='sympy',
          version=__version__,
          description='Computer algebra system (CAS) in Python',
          long_description=long_description,
          author='SymPy development team',
          author_email='sympy@googlegroups.com',
          license='BSD',
          keywords="Math CAS",
          url='http://sympy.org',
          packages=['sympy'] + modules + tests,
          scripts=['bin/isympy'],
          ext_modules=[],
          package_data={
              'sympy.utilities.mathml': ['data/*.xsl'],
              'sympy.logic.benchmarks': ['input/*.cnf'],
              },
          data_files=[('share/man/man1', ['doc/man/isympy.1'])],
          cmdclass={'test': test_sympy,
                    'bench': run_benchmarks,
                    'clean': clean,
                    'audit': audit},
          classifiers=[
            'License :: OSI Approved :: BSD License',
            'Operating System :: OS Independent',
            'Programming Language :: Python',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Mathematics',
            'Topic :: Scientific/Engineering :: Physics',
            'Programming Language :: Python :: 2',
            'Programming Language :: Python :: 2.6',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.2',
            'Programming Language :: Python :: 3.3',
            'Programming Language :: Python :: 3.4',
            'Programming Language :: Python :: 3.5',
            ],
          install_requires=['mpmath>=%s' % mpmath_version]
          )
```
### 10 - sympy/utilities/runtests.py:

Start line: 1, End line: 49

```python
"""
This is our testing framework.

Goals:

* it should be compatible with py.test and operate very similarly
  (or identically)
* doesn't require any external dependencies
* preferably all the functionality should be in this file only
* no magic, just import the test file and execute the test functions, that's it
* portable

"""

from __future__ import print_function, division

import os
import sys
import platform
import inspect
import traceback
import pdb
import re
import linecache
import time
from fnmatch import fnmatch
from timeit import default_timer as clock
import doctest as pdoctest  # avoid clashing with our doctest() function
from doctest import DocTestFinder, DocTestRunner
import random
import subprocess
import signal
import stat

from sympy.core.cache import clear_cache
from sympy.core.compatibility import exec_, PY3, string_types, range
from sympy.utilities.misc import find_executable
from sympy.external import import_module
from sympy.utilities.exceptions import SymPyDeprecationWarning

IS_WINDOWS = (os.name == 'nt')


class Skipped(Exception):
    pass


# add more flags ??
future_flags = division.compiler_flag
```
