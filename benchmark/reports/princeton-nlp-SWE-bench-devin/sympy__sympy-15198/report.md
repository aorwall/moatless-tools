# sympy__sympy-15198

| **sympy/sympy** | `115dd821a4b9ec94ca1bd339a8c0d63f31a12167` |
| ---- | ---- |
| **No of patches** | 11 |
| **All found context length** | 8122 |
| **Any found context length** | 8122 |
| **Avg pos** | 12.545454545454545 |
| **Min pos** | 25 |
| **Max pos** | 71 |
| **Top file pos** | 2 |
| **Missing snippets** | 20 |
| **Missing patch files** | 5 |


## Expected patch

```diff
diff --git a/sympy/combinatorics/homomorphisms.py b/sympy/combinatorics/homomorphisms.py
--- a/sympy/combinatorics/homomorphisms.py
+++ b/sympy/combinatorics/homomorphisms.py
@@ -445,6 +445,7 @@ def group_isomorphism(G, H, isomorphism=True):
     ========
 
     >>> from sympy.combinatorics import Permutation
+    >>> Permutation.print_cyclic = True
     >>> from sympy.combinatorics.perm_groups import PermutationGroup
     >>> from sympy.combinatorics.free_groups import free_group
     >>> from sympy.combinatorics.fp_groups import FpGroup
diff --git a/sympy/printing/ccode.py b/sympy/printing/ccode.py
--- a/sympy/printing/ccode.py
+++ b/sympy/printing/ccode.py
@@ -168,7 +168,7 @@ class C89CodePrinter(CodePrinter):
         'precision': 17,
         'user_functions': {},
         'human': True,
-        'allow_unknown_functions': True,
+        'allow_unknown_functions': False,
         'contract': True,
         'dereference': set(),
         'error_on_reserved': False,
diff --git a/sympy/printing/codeprinter.py b/sympy/printing/codeprinter.py
--- a/sympy/printing/codeprinter.py
+++ b/sympy/printing/codeprinter.py
@@ -54,7 +54,7 @@ class CodePrinter(StrPrinter):
         'reserved_word_suffix': '_',
         'human': True,
         'inline': False,
-        'allow_unknown_functions': True,
+        'allow_unknown_functions': False,
     }
 
     def __init__(self, settings=None):
@@ -382,7 +382,7 @@ def _print_Function(self, expr):
         elif hasattr(expr, '_imp_') and isinstance(expr._imp_, Lambda):
             # inlined function
             return self._print(expr._imp_(*expr.args))
-        elif expr.is_Function and self._settings.get('allow_unknown_functions', True):
+        elif expr.is_Function and self._settings.get('allow_unknown_functions', False):
             return '%s(%s)' % (self._print(expr.func), ', '.join(map(self._print, expr.args)))
         else:
             return self._print_not_supported(expr)
diff --git a/sympy/printing/fcode.py b/sympy/printing/fcode.py
--- a/sympy/printing/fcode.py
+++ b/sympy/printing/fcode.py
@@ -98,7 +98,7 @@ class FCodePrinter(CodePrinter):
         'precision': 17,
         'user_functions': {},
         'human': True,
-        'allow_unknown_functions': True,
+        'allow_unknown_functions': False,
         'source_format': 'fixed',
         'contract': True,
         'standard': 77,
diff --git a/sympy/printing/glsl.py b/sympy/printing/glsl.py
--- a/sympy/printing/glsl.py
+++ b/sympy/printing/glsl.py
@@ -50,7 +50,7 @@ class GLSLPrinter(CodePrinter):
         'precision': 9,
         'user_functions': {},
         'human': True,
-        'allow_unknown_functions': True,
+        'allow_unknown_functions': False,
         'contract': True,
         'error_on_reserved': False,
         'reserved_word_suffix': '_'
diff --git a/sympy/printing/jscode.py b/sympy/printing/jscode.py
--- a/sympy/printing/jscode.py
+++ b/sympy/printing/jscode.py
@@ -55,7 +55,7 @@ class JavascriptCodePrinter(CodePrinter):
         'precision': 17,
         'user_functions': {},
         'human': True,
-        'allow_unknown_functions': True,
+        'allow_unknown_functions': False,
         'contract': True
     }
 
diff --git a/sympy/printing/julia.py b/sympy/printing/julia.py
--- a/sympy/printing/julia.py
+++ b/sympy/printing/julia.py
@@ -62,7 +62,7 @@ class JuliaCodePrinter(CodePrinter):
         'precision': 17,
         'user_functions': {},
         'human': True,
-        'allow_unknown_functions': True,
+        'allow_unknown_functions': False,
         'contract': True,
         'inline': True,
     }
diff --git a/sympy/printing/mathematica.py b/sympy/printing/mathematica.py
--- a/sympy/printing/mathematica.py
+++ b/sympy/printing/mathematica.py
@@ -47,7 +47,7 @@ class MCodePrinter(CodePrinter):
         'precision': 15,
         'user_functions': {},
         'human': True,
-        'allow_unknown_functions': True,
+        'allow_unknown_functions': False,
     }
 
     _number_symbols = set()
diff --git a/sympy/printing/octave.py b/sympy/printing/octave.py
--- a/sympy/printing/octave.py
+++ b/sympy/printing/octave.py
@@ -78,7 +78,7 @@ class OctaveCodePrinter(CodePrinter):
         'precision': 17,
         'user_functions': {},
         'human': True,
-        'allow_unknown_functions': True,
+        'allow_unknown_functions': False,
         'contract': True,
         'inline': True,
     }
diff --git a/sympy/utilities/lambdify.py b/sympy/utilities/lambdify.py
--- a/sympy/utilities/lambdify.py
+++ b/sympy/utilities/lambdify.py
@@ -425,6 +425,7 @@ def lambdify(args, expr, modules=None, printer=None, use_imps=True,
                 for k in m:
                     user_functions[k] = k
         printer = Printer({'fully_qualified_modules': False, 'inline': True,
+                           'allow_unknown_functions': True,
                            'user_functions': user_functions})
 
     # Get the names of the args, for creating a docstring
diff --git a/sympy/utilities/runtests.py b/sympy/utilities/runtests.py
--- a/sympy/utilities/runtests.py
+++ b/sympy/utilities/runtests.py
@@ -145,13 +145,14 @@ def setup_pprint():
     import sympy.interactive.printing as interactive_printing
 
     # force pprint to be in ascii mode in doctests
-    pprint_use_unicode(False)
+    use_unicode_prev = pprint_use_unicode(False)
 
     # hook our nice, hash-stable strprinter
     init_printing(pretty_print=False)
 
     # Prevent init_printing() in doctests from affecting other doctests
     interactive_printing.NO_GLOBAL = True
+    return use_unicode_prev
 
 def run_in_subprocess_with_hash_randomization(
         function, function_args=(),
@@ -657,6 +658,8 @@ def _doctest(*paths, **kwargs):
     Returns 0 if tests passed and 1 if they failed.  See the docstrings of
     ``doctest()`` and ``test()`` for more information.
     """
+    from sympy import pprint_use_unicode
+
     normal = kwargs.get("normal", False)
     verbose = kwargs.get("verbose", False)
     colors = kwargs.get("colors", True)
@@ -822,7 +825,7 @@ def _doctest(*paths, **kwargs):
             continue
         old_displayhook = sys.displayhook
         try:
-            setup_pprint()
+            use_unicode_prev = setup_pprint()
             out = sympytestfile(
                 rst_file, module_relative=False, encoding='utf-8',
                 optionflags=pdoctest.ELLIPSIS | pdoctest.NORMALIZE_WHITESPACE |
@@ -835,6 +838,7 @@ def _doctest(*paths, **kwargs):
             # if True
             import sympy.interactive.printing as interactive_printing
             interactive_printing.NO_GLOBAL = False
+            pprint_use_unicode(use_unicode_prev)
 
         rstfailed, tested = out
         if tested:
@@ -1344,6 +1348,7 @@ def test_file(self, filename):
 
         from sympy.core.compatibility import StringIO
         import sympy.interactive.printing as interactive_printing
+        from sympy import pprint_use_unicode
 
         rel_name = filename[len(self._root_dir) + 1:]
         dirname, file = os.path.split(filename)
@@ -1354,7 +1359,6 @@ def test_file(self, filename):
             # So we have to temporarily extend sys.path to import them
             sys.path.insert(0, dirname)
             module = file[:-3]  # remove ".py"
-        setup_pprint()
         try:
             module = pdoctest._normalize_module(module)
             tests = SymPyDocTestFinder().find(module)
@@ -1366,7 +1370,6 @@ def test_file(self, filename):
         finally:
             if rel_name.startswith("examples"):
                 del sys.path[0]
-            interactive_printing.NO_GLOBAL = False
 
         tests = [test for test in tests if len(test.examples) > 0]
         # By default tests are sorted by alphabetical order by function name.
@@ -1412,6 +1415,10 @@ def test_file(self, filename):
                 # comes by default with a "from sympy import *"
                 #exec('from sympy import *') in test.globs
             test.globs['print_function'] = print_function
+
+            old_displayhook = sys.displayhook
+            use_unicode_prev = setup_pprint()
+
             try:
                 f, t = runner.run(test, compileflags=future_flags,
                                   out=new.write, clear_globs=False)
@@ -1423,6 +1430,10 @@ def test_file(self, filename):
                 self._reporter.doctest_fail(test.name, new.getvalue())
             else:
                 self._reporter.test_pass()
+                sys.displayhook = old_displayhook
+                interactive_printing.NO_GLOBAL = False
+                pprint_use_unicode(use_unicode_prev)
+
         self._reporter.leaving_filename()
 
     def get_test_files(self, dir, pat='*.py', init_only=True):

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sympy/combinatorics/homomorphisms.py | 448 | 448 | - | - | -
| sympy/printing/ccode.py | 171 | 171 | - | 5 | -
| sympy/printing/codeprinter.py | 57 | 57 | - | 12 | -
| sympy/printing/codeprinter.py | 385 | 385 | - | 12 | -
| sympy/printing/fcode.py | 101 | 101 | - | 11 | -
| sympy/printing/glsl.py | 53 | 53 | - | - | -
| sympy/printing/jscode.py | 58 | 58 | 25 | 2 | 8122
| sympy/printing/julia.py | 65 | 65 | 71 | 4 | 24203
| sympy/printing/mathematica.py | 50 | 50 | - | - | -
| sympy/printing/octave.py | 81 | 81 | 42 | 3 | 13178
| sympy/utilities/lambdify.py | 428 | 428 | - | - | -
| sympy/utilities/runtests.py | 148 | 148 | - | - | -
| sympy/utilities/runtests.py | 660 | 660 | - | - | -
| sympy/utilities/runtests.py | 825 | 825 | - | - | -
| sympy/utilities/runtests.py | 838 | 838 | - | - | -
| sympy/utilities/runtests.py | 1347 | 1347 | - | - | -
| sympy/utilities/runtests.py | 1357 | 1357 | - | - | -
| sympy/utilities/runtests.py | 1369 | 1369 | - | - | -
| sympy/utilities/runtests.py | 1415 | 1415 | - | - | -
| sympy/utilities/runtests.py | 1426 | 1426 | - | - | -


## Problem Statement

```
1.3rc1 codegen regression in octave/julia/jscode
@asmeurer @bjodah I have a (minor?) regression in codeprinting from e99b756df3291a666ee2d2288daec4253014df40
Can one of you double-check that commit before 1.3?

Octave codegen prints `laguerre` but is supposed to error on `assoc_laguerre` (untested, apparently).  The above commit breaks that.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 sympy/printing/llvmjitcode.py | 438 | 475| 346 | 346 | 4016 | 
| 2 | **2 sympy/printing/jscode.py** | 100 | 110| 133 | 479 | 6831 | 
| 3 | **2 sympy/printing/jscode.py** | 87 | 98| 138 | 617 | 6831 | 
| 4 | **2 sympy/printing/jscode.py** | 112 | 139| 215 | 832 | 6831 | 
| 5 | 2 sympy/printing/llvmjitcode.py | 148 | 171| 202 | 1034 | 6831 | 
| 6 | **3 sympy/printing/octave.py** | 1 | 59| 662 | 1696 | 13344 | 
| 7 | **3 sympy/printing/octave.py** | 381 | 475| 813 | 2509 | 13344 | 
| 8 | 3 sympy/printing/llvmjitcode.py | 263 | 286| 161 | 2670 | 13344 | 
| 9 | 3 sympy/printing/llvmjitcode.py | 245 | 261| 150 | 2820 | 13344 | 
| 10 | **3 sympy/printing/octave.py** | 341 | 357| 183 | 3003 | 13344 | 
| 11 | 3 sympy/printing/llvmjitcode.py | 210 | 243| 312 | 3315 | 13344 | 
| 12 | **3 sympy/printing/jscode.py** | 141 | 171| 358 | 3673 | 13344 | 
| 13 | 3 sympy/printing/llvmjitcode.py | 124 | 145| 178 | 3851 | 13344 | 
| 14 | **4 sympy/printing/julia.py** | 353 | 369| 181 | 4032 | 19103 | 
| 15 | **4 sympy/printing/julia.py** | 1 | 43| 496 | 4528 | 19103 | 
| 16 | **4 sympy/printing/julia.py** | 214 | 256| 290 | 4818 | 19103 | 
| 17 | **4 sympy/printing/octave.py** | 231 | 252| 172 | 4990 | 19103 | 
| 18 | **4 sympy/printing/octave.py** | 255 | 280| 282 | 5272 | 19103 | 
| 19 | 4 sympy/printing/llvmjitcode.py | 358 | 436| 827 | 6099 | 19103 | 
| 20 | **4 sympy/printing/jscode.py** | 173 | 204| 259 | 6358 | 19103 | 
| 21 | 4 sympy/printing/llvmjitcode.py | 57 | 77| 242 | 6600 | 19103 | 
| 22 | **4 sympy/printing/octave.py** | 211 | 228| 199 | 6799 | 19103 | 
| 23 | **4 sympy/printing/octave.py** | 525 | 701| 333 | 7132 | 19103 | 
| 24 | **4 sympy/printing/octave.py** | 136 | 208| 704 | 7836 | 19103 | 
| **-> 25 <-** | **4 sympy/printing/jscode.py** | 46 | 85| 286 | 8122 | 19103 | 
| 26 | 4 sympy/printing/llvmjitcode.py | 79 | 107| 267 | 8389 | 19103 | 
| 27 | **4 sympy/printing/julia.py** | 194 | 211| 198 | 8587 | 19103 | 
| 28 | **5 sympy/printing/ccode.py** | 482 | 507| 236 | 8823 | 27100 | 
| 29 | **5 sympy/printing/julia.py** | 393 | 418| 238 | 9061 | 27100 | 
| 30 | **5 sympy/printing/julia.py** | 259 | 284| 281 | 9342 | 27100 | 
| 31 | 5 sympy/printing/llvmjitcode.py | 110 | 122| 155 | 9497 | 27100 | 
| 32 | **5 sympy/printing/ccode.py** | 302 | 342| 344 | 9841 | 27100 | 
| 33 | **5 sympy/printing/ccode.py** | 622 | 635| 157 | 9998 | 27100 | 
| 34 | 6 sympy/printing/cxxcode.py | 140 | 156| 130 | 10128 | 28730 | 
| 35 | **6 sympy/printing/julia.py** | 119 | 191| 701 | 10829 | 28730 | 
| 36 | **6 sympy/printing/jscode.py** | 1 | 43| 314 | 11143 | 28730 | 
| 37 | **6 sympy/printing/octave.py** | 478 | 522| 496 | 11639 | 28730 | 
| 38 | 7 sympy/integrals/meijerint.py | 987 | 1010| 442 | 12081 | 53012 | 
| 39 | **7 sympy/printing/ccode.py** | 719 | 737| 179 | 12260 | 53012 | 
| 40 | **7 sympy/printing/ccode.py** | 466 | 480| 163 | 12423 | 53012 | 
| 41 | **7 sympy/printing/ccode.py** | 418 | 437| 251 | 12674 | 53012 | 
| **-> 42 <-** | **7 sympy/printing/octave.py** | 62 | 133| 504 | 13178 | 53012 | 
| 43 | **7 sympy/printing/julia.py** | 458 | 624| 269 | 13447 | 53012 | 
| 44 | 8 sympy/printing/rcode.py | 230 | 264| 371 | 13818 | 56736 | 
| 45 | **8 sympy/printing/ccode.py** | 407 | 416| 126 | 13944 | 56736 | 
| 46 | **8 sympy/printing/ccode.py** | 377 | 405| 314 | 14258 | 56736 | 
| 47 | 8 sympy/printing/cxxcode.py | 1 | 61| 654 | 14912 | 56736 | 
| 48 | **8 sympy/printing/ccode.py** | 286 | 300| 204 | 15116 | 56736 | 
| 49 | **8 sympy/printing/octave.py** | 330 | 338| 140 | 15256 | 56736 | 
| 50 | 8 sympy/integrals/meijerint.py | 1011 | 1052| 628 | 15884 | 56736 | 
| 51 | 9 sympy/printing/pycode.py | 337 | 363| 272 | 16156 | 61856 | 
| 52 | 10 sympy/printing/rust.py | 468 | 480| 145 | 16301 | 67293 | 
| 53 | **10 sympy/printing/ccode.py** | 683 | 716| 292 | 16593 | 67293 | 
| 54 | **10 sympy/printing/octave.py** | 360 | 378| 189 | 16782 | 67293 | 
| 55 | 10 sympy/printing/rust.py | 1 | 56| 581 | 17363 | 67293 | 
| 56 | **10 sympy/printing/jscode.py** | 207 | 321| 1145 | 18508 | 67293 | 
| 57 | 10 sympy/printing/llvmjitcode.py | 1 | 22| 155 | 18663 | 67293 | 
| 58 | **10 sympy/printing/ccode.py** | 638 | 649| 107 | 18770 | 67293 | 
| 59 | 10 sympy/printing/cxxcode.py | 111 | 138| 285 | 19055 | 67293 | 
| 60 | **10 sympy/printing/octave.py** | 566 | 710| 1626 | 20681 | 67293 | 
| 61 | **10 sympy/printing/octave.py** | 317 | 327| 157 | 20838 | 67293 | 
| 62 | **11 sympy/printing/fcode.py** | 342 | 376| 355 | 21193 | 75056 | 
| 63 | **11 sympy/printing/julia.py** | 342 | 350| 139 | 21332 | 75056 | 
| 64 | **12 sympy/printing/codeprinter.py** | 331 | 362| 257 | 21589 | 79493 | 
| 65 | 12 sympy/integrals/meijerint.py | 1053 | 1095| 741 | 22330 | 79493 | 
| 66 | **12 sympy/printing/ccode.py** | 344 | 375| 360 | 22690 | 79493 | 
| 67 | **12 sympy/printing/fcode.py** | 323 | 340| 171 | 22861 | 79493 | 
| 68 | 12 sympy/printing/rcode.py | 181 | 216| 384 | 23245 | 79493 | 
| 69 | 12 sympy/printing/rcode.py | 145 | 155| 130 | 23375 | 79493 | 
| 70 | **12 sympy/printing/fcode.py** | 439 | 469| 330 | 23705 | 79493 | 
| **-> 71 <-** | **12 sympy/printing/julia.py** | 46 | 116| 498 | 24203 | 79493 | 
| 72 | 12 sympy/printing/llvmjitcode.py | 25 | 55| 267 | 24470 | 79493 | 
| 73 | **12 sympy/printing/fcode.py** | 254 | 295| 352 | 24822 | 79493 | 
| 74 | 12 sympy/printing/rust.py | 263 | 285| 228 | 25050 | 79493 | 
| 75 | 12 sympy/printing/rcode.py | 158 | 179| 177 | 25227 | 79493 | 
| 76 | 12 sympy/integrals/meijerint.py | 1096 | 1133| 808 | 26035 | 79493 | 
| 77 | 13 sympy/parsing/latex/_parse_latex_antlr.py | 1 | 59| 471 | 26506 | 83885 | 
| 78 | **13 sympy/printing/julia.py** | 372 | 390| 187 | 26693 | 83885 | 
| 79 | 14 bin/mailmap_update.py | 1 | 104| 792 | 27485 | 85342 | 
| 80 | **14 sympy/printing/fcode.py** | 395 | 411| 187 | 27672 | 85342 | 
| 81 | **14 sympy/printing/fcode.py** | 378 | 393| 124 | 27796 | 85342 | 
| 82 | **14 sympy/printing/octave.py** | 283 | 314| 175 | 27971 | 85342 | 
| 83 | **14 sympy/printing/julia.py** | 421 | 455| 399 | 28370 | 85342 | 
| 84 | 14 sympy/printing/llvmjitcode.py | 289 | 324| 314 | 28684 | 85342 | 
| 85 | 14 sympy/printing/rcode.py | 266 | 275| 125 | 28809 | 85342 | 
| 86 | 14 sympy/printing/rcode.py | 218 | 228| 158 | 28967 | 85342 | 
| 87 | 15 sympy/codegen/ast.py | 630 | 642| 166 | 29133 | 98692 | 
| 88 | **15 sympy/printing/ccode.py** | 92 | 100| 132 | 29265 | 98692 | 
| 89 | 16 sympy/physics/quantum/cg.py | 684 | 701| 229 | 29494 | 105227 | 
| 90 | **16 sympy/printing/fcode.py** | 413 | 437| 226 | 29720 | 105227 | 
| 91 | 17 sympy/parsing/autolev/_listener_autolev_antlr.py | 1741 | 1860| 1531 | 31251 | 128338 | 
| 92 | 17 sympy/physics/quantum/cg.py | 499 | 507| 171 | 31422 | 128338 | 
| 93 | **17 sympy/printing/fcode.py** | 297 | 321| 219 | 31641 | 128338 | 
| 94 | **17 sympy/printing/fcode.py** | 189 | 198| 117 | 31758 | 128338 | 
| 95 | **17 sympy/printing/julia.py** | 491 | 633| 1597 | 33355 | 128338 | 
| 96 | **17 sympy/printing/ccode.py** | 1 | 90| 752 | 34107 | 128338 | 
| 97 | 17 sympy/physics/quantum/cg.py | 673 | 681| 112 | 34219 | 128338 | 
| 98 | 17 sympy/printing/cxxcode.py | 79 | 108| 285 | 34504 | 128338 | 
| 99 | 18 sympy/codegen/rewriting.py | 165 | 195| 248 | 34752 | 130192 | 
| 100 | 18 sympy/integrals/meijerint.py | 1175 | 1212| 760 | 35512 | 130192 | 
| 101 | **18 sympy/printing/codeprinter.py** | 124 | 201| 718 | 36230 | 130192 | 
| 102 | **18 sympy/printing/codeprinter.py** | 390 | 441| 507 | 36737 | 130192 | 
| 103 | 19 sympy/polys/modulargcd.py | 790 | 804| 171 | 36908 | 148687 | 
| 104 | 20 sympy/codegen/cxxnodes.py | 1 | 15| 0 | 36908 | 148768 | 
| 105 | 20 sympy/physics/quantum/cg.py | 510 | 518| 176 | 37084 | 148768 | 
| 106 | **20 sympy/printing/fcode.py** | 164 | 187| 207 | 37291 | 148768 | 
| 107 | 21 sympy/physics/wigner.py | 205 | 264| 290 | 37581 | 157002 | 
| 108 | **21 sympy/printing/fcode.py** | 654 | 674| 178 | 37759 | 157002 | 
| 109 | **21 sympy/printing/ccode.py** | 522 | 545| 190 | 37949 | 157002 | 
| 110 | 21 sympy/integrals/meijerint.py | 799 | 841| 489 | 38438 | 157002 | 
| 111 | 21 sympy/printing/rust.py | 57 | 162| 1065 | 39503 | 157002 | 
| 112 | 22 sympy/parsing/latex/_antlr/latexparser.py | 54 | 85| 1527 | 41030 | 187277 | 
| 113 | **22 sympy/printing/codeprinter.py** | 203 | 238| 257 | 41287 | 187277 | 
| 114 | **22 sympy/printing/ccode.py** | 652 | 681| 312 | 41599 | 187277 | 
| 115 | 23 sympy/plotting/experimental_lambdify.py | 1 | 76| 865 | 42464 | 193129 | 
| 116 | **23 sympy/printing/fcode.py** | 615 | 639| 204 | 42668 | 193129 | 
| 117 | 23 sympy/integrals/meijerint.py | 1213 | 1239| 453 | 43121 | 193129 | 
| 118 | 23 sympy/printing/rust.py | 338 | 400| 468 | 43589 | 193129 | 
| 119 | **23 sympy/printing/julia.py** | 287 | 324| 215 | 43804 | 193129 | 
| 120 | **23 sympy/printing/ccode.py** | 547 | 618| 611 | 44415 | 193129 | 
| 121 | 23 sympy/printing/rust.py | 429 | 466| 349 | 44764 | 193129 | 
| 122 | 24 sympy/codegen/cutils.py | 1 | 9| 0 | 44764 | 193223 | 
| 123 | **24 sympy/printing/julia.py** | 327 | 339| 181 | 44945 | 193223 | 


## Missing Patch Files

 * 1: sympy/combinatorics/homomorphisms.py
 * 2: sympy/printing/ccode.py
 * 3: sympy/printing/codeprinter.py
 * 4: sympy/printing/fcode.py
 * 5: sympy/printing/glsl.py
 * 6: sympy/printing/jscode.py
 * 7: sympy/printing/julia.py
 * 8: sympy/printing/mathematica.py
 * 9: sympy/printing/octave.py
 * 10: sympy/utilities/lambdify.py
 * 11: sympy/utilities/runtests.py

## Patch

```diff
diff --git a/sympy/combinatorics/homomorphisms.py b/sympy/combinatorics/homomorphisms.py
--- a/sympy/combinatorics/homomorphisms.py
+++ b/sympy/combinatorics/homomorphisms.py
@@ -445,6 +445,7 @@ def group_isomorphism(G, H, isomorphism=True):
     ========
 
     >>> from sympy.combinatorics import Permutation
+    >>> Permutation.print_cyclic = True
     >>> from sympy.combinatorics.perm_groups import PermutationGroup
     >>> from sympy.combinatorics.free_groups import free_group
     >>> from sympy.combinatorics.fp_groups import FpGroup
diff --git a/sympy/printing/ccode.py b/sympy/printing/ccode.py
--- a/sympy/printing/ccode.py
+++ b/sympy/printing/ccode.py
@@ -168,7 +168,7 @@ class C89CodePrinter(CodePrinter):
         'precision': 17,
         'user_functions': {},
         'human': True,
-        'allow_unknown_functions': True,
+        'allow_unknown_functions': False,
         'contract': True,
         'dereference': set(),
         'error_on_reserved': False,
diff --git a/sympy/printing/codeprinter.py b/sympy/printing/codeprinter.py
--- a/sympy/printing/codeprinter.py
+++ b/sympy/printing/codeprinter.py
@@ -54,7 +54,7 @@ class CodePrinter(StrPrinter):
         'reserved_word_suffix': '_',
         'human': True,
         'inline': False,
-        'allow_unknown_functions': True,
+        'allow_unknown_functions': False,
     }
 
     def __init__(self, settings=None):
@@ -382,7 +382,7 @@ def _print_Function(self, expr):
         elif hasattr(expr, '_imp_') and isinstance(expr._imp_, Lambda):
             # inlined function
             return self._print(expr._imp_(*expr.args))
-        elif expr.is_Function and self._settings.get('allow_unknown_functions', True):
+        elif expr.is_Function and self._settings.get('allow_unknown_functions', False):
             return '%s(%s)' % (self._print(expr.func), ', '.join(map(self._print, expr.args)))
         else:
             return self._print_not_supported(expr)
diff --git a/sympy/printing/fcode.py b/sympy/printing/fcode.py
--- a/sympy/printing/fcode.py
+++ b/sympy/printing/fcode.py
@@ -98,7 +98,7 @@ class FCodePrinter(CodePrinter):
         'precision': 17,
         'user_functions': {},
         'human': True,
-        'allow_unknown_functions': True,
+        'allow_unknown_functions': False,
         'source_format': 'fixed',
         'contract': True,
         'standard': 77,
diff --git a/sympy/printing/glsl.py b/sympy/printing/glsl.py
--- a/sympy/printing/glsl.py
+++ b/sympy/printing/glsl.py
@@ -50,7 +50,7 @@ class GLSLPrinter(CodePrinter):
         'precision': 9,
         'user_functions': {},
         'human': True,
-        'allow_unknown_functions': True,
+        'allow_unknown_functions': False,
         'contract': True,
         'error_on_reserved': False,
         'reserved_word_suffix': '_'
diff --git a/sympy/printing/jscode.py b/sympy/printing/jscode.py
--- a/sympy/printing/jscode.py
+++ b/sympy/printing/jscode.py
@@ -55,7 +55,7 @@ class JavascriptCodePrinter(CodePrinter):
         'precision': 17,
         'user_functions': {},
         'human': True,
-        'allow_unknown_functions': True,
+        'allow_unknown_functions': False,
         'contract': True
     }
 
diff --git a/sympy/printing/julia.py b/sympy/printing/julia.py
--- a/sympy/printing/julia.py
+++ b/sympy/printing/julia.py
@@ -62,7 +62,7 @@ class JuliaCodePrinter(CodePrinter):
         'precision': 17,
         'user_functions': {},
         'human': True,
-        'allow_unknown_functions': True,
+        'allow_unknown_functions': False,
         'contract': True,
         'inline': True,
     }
diff --git a/sympy/printing/mathematica.py b/sympy/printing/mathematica.py
--- a/sympy/printing/mathematica.py
+++ b/sympy/printing/mathematica.py
@@ -47,7 +47,7 @@ class MCodePrinter(CodePrinter):
         'precision': 15,
         'user_functions': {},
         'human': True,
-        'allow_unknown_functions': True,
+        'allow_unknown_functions': False,
     }
 
     _number_symbols = set()
diff --git a/sympy/printing/octave.py b/sympy/printing/octave.py
--- a/sympy/printing/octave.py
+++ b/sympy/printing/octave.py
@@ -78,7 +78,7 @@ class OctaveCodePrinter(CodePrinter):
         'precision': 17,
         'user_functions': {},
         'human': True,
-        'allow_unknown_functions': True,
+        'allow_unknown_functions': False,
         'contract': True,
         'inline': True,
     }
diff --git a/sympy/utilities/lambdify.py b/sympy/utilities/lambdify.py
--- a/sympy/utilities/lambdify.py
+++ b/sympy/utilities/lambdify.py
@@ -425,6 +425,7 @@ def lambdify(args, expr, modules=None, printer=None, use_imps=True,
                 for k in m:
                     user_functions[k] = k
         printer = Printer({'fully_qualified_modules': False, 'inline': True,
+                           'allow_unknown_functions': True,
                            'user_functions': user_functions})
 
     # Get the names of the args, for creating a docstring
diff --git a/sympy/utilities/runtests.py b/sympy/utilities/runtests.py
--- a/sympy/utilities/runtests.py
+++ b/sympy/utilities/runtests.py
@@ -145,13 +145,14 @@ def setup_pprint():
     import sympy.interactive.printing as interactive_printing
 
     # force pprint to be in ascii mode in doctests
-    pprint_use_unicode(False)
+    use_unicode_prev = pprint_use_unicode(False)
 
     # hook our nice, hash-stable strprinter
     init_printing(pretty_print=False)
 
     # Prevent init_printing() in doctests from affecting other doctests
     interactive_printing.NO_GLOBAL = True
+    return use_unicode_prev
 
 def run_in_subprocess_with_hash_randomization(
         function, function_args=(),
@@ -657,6 +658,8 @@ def _doctest(*paths, **kwargs):
     Returns 0 if tests passed and 1 if they failed.  See the docstrings of
     ``doctest()`` and ``test()`` for more information.
     """
+    from sympy import pprint_use_unicode
+
     normal = kwargs.get("normal", False)
     verbose = kwargs.get("verbose", False)
     colors = kwargs.get("colors", True)
@@ -822,7 +825,7 @@ def _doctest(*paths, **kwargs):
             continue
         old_displayhook = sys.displayhook
         try:
-            setup_pprint()
+            use_unicode_prev = setup_pprint()
             out = sympytestfile(
                 rst_file, module_relative=False, encoding='utf-8',
                 optionflags=pdoctest.ELLIPSIS | pdoctest.NORMALIZE_WHITESPACE |
@@ -835,6 +838,7 @@ def _doctest(*paths, **kwargs):
             # if True
             import sympy.interactive.printing as interactive_printing
             interactive_printing.NO_GLOBAL = False
+            pprint_use_unicode(use_unicode_prev)
 
         rstfailed, tested = out
         if tested:
@@ -1344,6 +1348,7 @@ def test_file(self, filename):
 
         from sympy.core.compatibility import StringIO
         import sympy.interactive.printing as interactive_printing
+        from sympy import pprint_use_unicode
 
         rel_name = filename[len(self._root_dir) + 1:]
         dirname, file = os.path.split(filename)
@@ -1354,7 +1359,6 @@ def test_file(self, filename):
             # So we have to temporarily extend sys.path to import them
             sys.path.insert(0, dirname)
             module = file[:-3]  # remove ".py"
-        setup_pprint()
         try:
             module = pdoctest._normalize_module(module)
             tests = SymPyDocTestFinder().find(module)
@@ -1366,7 +1370,6 @@ def test_file(self, filename):
         finally:
             if rel_name.startswith("examples"):
                 del sys.path[0]
-            interactive_printing.NO_GLOBAL = False
 
         tests = [test for test in tests if len(test.examples) > 0]
         # By default tests are sorted by alphabetical order by function name.
@@ -1412,6 +1415,10 @@ def test_file(self, filename):
                 # comes by default with a "from sympy import *"
                 #exec('from sympy import *') in test.globs
             test.globs['print_function'] = print_function
+
+            old_displayhook = sys.displayhook
+            use_unicode_prev = setup_pprint()
+
             try:
                 f, t = runner.run(test, compileflags=future_flags,
                                   out=new.write, clear_globs=False)
@@ -1423,6 +1430,10 @@ def test_file(self, filename):
                 self._reporter.doctest_fail(test.name, new.getvalue())
             else:
                 self._reporter.test_pass()
+                sys.displayhook = old_displayhook
+                interactive_printing.NO_GLOBAL = False
+                pprint_use_unicode(use_unicode_prev)
+
         self._reporter.leaving_filename()
 
     def get_test_files(self, dir, pat='*.py', init_only=True):

```

## Test Patch

```diff
diff --git a/sympy/printing/tests/test_ccode.py b/sympy/printing/tests/test_ccode.py
--- a/sympy/printing/tests/test_ccode.py
+++ b/sympy/printing/tests/test_ccode.py
@@ -133,8 +133,12 @@ def test_ccode_inline_function():
 
 def test_ccode_exceptions():
     assert ccode(gamma(x), standard='C99') == "tgamma(x)"
+    gamma_c89 = ccode(gamma(x), standard='C89')
+    assert 'not supported in c' in gamma_c89.lower()
     gamma_c89 = ccode(gamma(x), standard='C89', allow_unknown_functions=False)
     assert 'not supported in c' in gamma_c89.lower()
+    gamma_c89 = ccode(gamma(x), standard='C89', allow_unknown_functions=True)
+    assert not 'not supported in c' in gamma_c89.lower()
     assert ccode(ceiling(x)) == "ceil(x)"
     assert ccode(Abs(x)) == "fabs(x)"
     assert ccode(gamma(x)) == "tgamma(x)"
diff --git a/sympy/printing/tests/test_fcode.py b/sympy/printing/tests/test_fcode.py
--- a/sympy/printing/tests/test_fcode.py
+++ b/sympy/printing/tests/test_fcode.py
@@ -168,10 +168,10 @@ def test_implicit():
 def test_not_fortran():
     x = symbols('x')
     g = Function('g')
-    gamma_f = fcode(gamma(x), allow_unknown_functions=False)
+    gamma_f = fcode(gamma(x))
     assert gamma_f == "C     Not supported in Fortran:\nC     gamma\n      gamma(x)"
     assert fcode(Integral(sin(x))) == "C     Not supported in Fortran:\nC     Integral\n      Integral(sin(x), x)"
-    assert fcode(g(x), allow_unknown_functions=False) == "C     Not supported in Fortran:\nC     g\n      g(x)"
+    assert fcode(g(x)) == "C     Not supported in Fortran:\nC     g\n      g(x)"
 
 
 def test_user_functions():
diff --git a/sympy/printing/tests/test_octave.py b/sympy/printing/tests/test_octave.py
--- a/sympy/printing/tests/test_octave.py
+++ b/sympy/printing/tests/test_octave.py
@@ -374,6 +374,15 @@ def test_octave_not_supported():
     )
 
 
+def test_octave_not_supported_not_on_whitelist():
+    from sympy import assoc_laguerre
+    assert mcode(assoc_laguerre(x, y, z)) == (
+        "% Not supported in Octave:\n"
+        "% assoc_laguerre\n"
+        "assoc_laguerre(x, y, z)"
+    )
+
+
 def test_octave_expint():
     assert mcode(expint(1, x)) == "expint(x)"
     assert mcode(expint(2, x)) == (

```


## Code snippets

### 1 - sympy/printing/llvmjitcode.py:

Start line: 438, End line: 475

```python
@doctest_depends_on(modules=('llvmlite', 'scipy'))
def llvm_callable(args, expr, callback_type=None):

    if not llvmlite:
        raise ImportError("llvmlite is required for llvmjitcode")

    signature = CodeSignature(ctypes.py_object)

    arg_ctypes = []
    if callback_type is None:
        for arg in args:
            arg_ctype = ctypes.c_double
            arg_ctypes.append(arg_ctype)
    elif callback_type == 'scipy.integrate' or callback_type == 'scipy.integrate.test':
        signature.ret_type = ctypes.c_double
        arg_ctypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
        arg_ctypes_formal = [ctypes.c_int, ctypes.c_double]
        signature.input_arg = 1
    elif callback_type == 'cubature':
        arg_ctypes = [ctypes.c_int,
                      ctypes.POINTER(ctypes.c_double),
                      ctypes.c_void_p,
                      ctypes.c_int,
                      ctypes.POINTER(ctypes.c_double)
                      ]
        signature.ret_type = ctypes.c_int
        signature.input_arg = 1
        signature.ret_arg = 4
    else:
        raise ValueError("Unknown callback type: %s" % callback_type)

    signature.arg_ctypes = arg_ctypes

    fptr = _llvm_jit_code(args, expr, signature, callback_type)

    if callback_type and callback_type == 'scipy.integrate':
        arg_ctypes = arg_ctypes_formal

    cfunc = ctypes.CFUNCTYPE(signature.ret_type, *arg_ctypes)(fptr)
    return cfunc
```
### 2 - sympy/printing/jscode.py:

Start line: 100, End line: 110

```python
class JavascriptCodePrinter(CodePrinter):

    def _print_Pow(self, expr):
        PREC = precedence(expr)
        if expr.exp == -1:
            return '1/%s' % (self.parenthesize(expr.base, PREC))
        elif expr.exp == 0.5:
            return 'Math.sqrt(%s)' % self._print(expr.base)
        elif expr.exp == S(1)/3:
            return 'Math.cbrt(%s)' % self._print(expr.base)
        else:
            return 'Math.pow(%s, %s)' % (self._print(expr.base),
                                 self._print(expr.exp))
```
### 3 - sympy/printing/jscode.py:

Start line: 87, End line: 98

```python
class JavascriptCodePrinter(CodePrinter):

    def _get_loop_opening_ending(self, indices):
        open_lines = []
        close_lines = []
        loopstart = "for (var %(varble)s=%(start)s; %(varble)s<%(end)s; %(varble)s++){"
        for i in indices:
            # Javascript arrays start at 0 and end at dimension-1
            open_lines.append(loopstart % {
                'varble': self._print(i.label),
                'start': self._print(i.lower),
                'end': self._print(i.upper + 1)})
            close_lines.append("}")
        return open_lines, close_lines
```
### 4 - sympy/printing/jscode.py:

Start line: 112, End line: 139

```python
class JavascriptCodePrinter(CodePrinter):

    def _print_Rational(self, expr):
        p, q = int(expr.p), int(expr.q)
        return '%d/%d' % (p, q)

    def _print_Indexed(self, expr):
        # calculate index for 1d array
        dims = expr.shape
        elem = S.Zero
        offset = S.One
        for i in reversed(range(expr.rank)):
            elem += expr.indices[i]*offset
            offset *= dims[i]
        return "%s[%s]" % (self._print(expr.base.label), self._print(elem))

    def _print_Idx(self, expr):
        return self._print(expr.label)

    def _print_Exp1(self, expr):
        return "Math.E"

    def _print_Pi(self, expr):
        return 'Math.PI'

    def _print_Infinity(self, expr):
        return 'Number.POSITIVE_INFINITY'

    def _print_NegativeInfinity(self, expr):
        return 'Number.NEGATIVE_INFINITY'
```
### 5 - sympy/printing/llvmjitcode.py:

Start line: 148, End line: 171

```python
class LLVMJitCode(object):
    def __init__(self, signature):
        self.signature = signature
        self.fp_type = ll.DoubleType()
        self.module = ll.Module('mod1')
        self.fn = None
        self.llvm_arg_types = []
        self.llvm_ret_type = self.fp_type
        self.param_dict = {}  # map symbol name to LLVM function argument
        self.link_name = ''

    def _from_ctype(self, ctype):
        if ctype == ctypes.c_int:
            return ll.IntType(32)
        if ctype == ctypes.c_double:
            return self.fp_type
        if ctype == ctypes.POINTER(ctypes.c_double):
            return ll.PointerType(self.fp_type)
        if ctype == ctypes.c_void_p:
            return ll.PointerType(ll.IntType(32))
        if ctype == ctypes.py_object:
            return ll.PointerType(ll.IntType(32))

        print("Unhandled ctype = %s" % str(ctype))
```
### 6 - sympy/printing/octave.py:

Start line: 1, End line: 59

```python
"""
Octave (and Matlab) code printer

The `OctaveCodePrinter` converts SymPy expressions into Octave expressions.
It uses a subset of the Octave language for Matlab compatibility.

A complete code generator, which uses `octave_code` extensively, can be found
in `sympy.utilities.codegen`.  The `codegen` module can be used to generate
complete source code files.

"""

from __future__ import print_function, division
from sympy.core import Mul, Pow, S, Rational
from sympy.core.compatibility import string_types, range
from sympy.core.mul import _keep_coeff
from sympy.codegen.ast import Assignment
from sympy.printing.codeprinter import CodePrinter
from sympy.printing.precedence import precedence, PRECEDENCE
from re import search

# List of known functions.  First, those that have the same name in
# SymPy and Octave.   This is almost certainly incomplete!
known_fcns_src1 = ["sin", "cos", "tan", "cot", "sec", "csc",
                   "asin", "acos", "acot", "atan", "atan2", "asec", "acsc",
                   "sinh", "cosh", "tanh", "coth", "csch", "sech",
                   "asinh", "acosh", "atanh", "acoth", "asech", "acsch",
                   "erfc", "erfi", "erf", "erfinv", "erfcinv",
                   "besseli", "besselj", "besselk", "bessely",
                   "bernoulli", "beta", "euler", "exp", "factorial", "floor",
                   "fresnelc", "fresnels", "gamma", "harmonic", "log",
                   "polylog", "sign", "zeta"]

# These functions have different names ("Sympy": "Octave"), more
# generally a mapping to (argument_conditions, octave_function).
known_fcns_src2 = {
    "Abs": "abs",
    "arg": "angle",  # arg/angle ok in Octave but only angle in Matlab
    "ceiling": "ceil",
    "chebyshevu": "chebyshevU",
    "chebyshevt": "chebyshevT",
    "Chi": "coshint",
    "Ci": "cosint",
    "conjugate": "conj",
    "DiracDelta": "dirac",
    "Heaviside": "heaviside",
    "im": "imag",
    "laguerre": "laguerreL",
    "LambertW": "lambertw",
    "li": "logint",
    "loggamma": "gammaln",
    "Max": "max",
    "Min": "min",
    "polygamma": "psi",
    "re": "real",
    "RisingFactorial": "pochhammer",
    "Shi": "sinhint",
    "Si": "sinint",
}
```
### 7 - sympy/printing/octave.py:

Start line: 381, End line: 475

```python
class OctaveCodePrinter(CodePrinter):


    def _print_Indexed(self, expr):
        inds = [ self._print(i) for i in expr.indices ]
        return "%s(%s)" % (self._print(expr.base.label), ", ".join(inds))


    def _print_Idx(self, expr):
        return self._print(expr.label)


    def _print_KroneckerDelta(self, expr):
        prec = PRECEDENCE["Pow"]
        return "double(%s == %s)" % tuple(self.parenthesize(x, prec)
                                          for x in expr.args)


    def _print_Identity(self, expr):
        shape = expr.shape
        if len(shape) == 2 and shape[0] == shape[1]:
            shape = [shape[0]]
        s = ", ".join(self._print(n) for n in shape)
        return "eye(" + s + ")"


    def _print_uppergamma(self, expr):
        return "gammainc(%s, %s, 'upper')" % (self._print(expr.args[1]),
                                              self._print(expr.args[0]))


    def _print_lowergamma(self, expr):
        return "gammainc(%s, %s, 'lower')" % (self._print(expr.args[1]),
                                              self._print(expr.args[0]))


    def _print_sinc(self, expr):
        #Note: Divide by pi because Octave implements normalized sinc function.
        return "sinc(%s)" % self._print(expr.args[0]/S.Pi)


    def _print_hankel1(self, expr):
        return "besselh(%s, 1, %s)" % (self._print(expr.order),
                                       self._print(expr.argument))


    def _print_hankel2(self, expr):
        return "besselh(%s, 2, %s)" % (self._print(expr.order),
                                       self._print(expr.argument))


    # Note: as of 2015, Octave doesn't have spherical Bessel functions
    def _print_jn(self, expr):
        from sympy.functions import sqrt, besselj
        x = expr.argument
        expr2 = sqrt(S.Pi/(2*x))*besselj(expr.order + S.Half, x)
        return self._print(expr2)


    def _print_yn(self, expr):
        from sympy.functions import sqrt, bessely
        x = expr.argument
        expr2 = sqrt(S.Pi/(2*x))*bessely(expr.order + S.Half, x)
        return self._print(expr2)


    def _print_airyai(self, expr):
        return "airy(0, %s)" % self._print(expr.args[0])


    def _print_airyaiprime(self, expr):
        return "airy(1, %s)" % self._print(expr.args[0])


    def _print_airybi(self, expr):
        return "airy(2, %s)" % self._print(expr.args[0])


    def _print_airybiprime(self, expr):
        return "airy(3, %s)" % self._print(expr.args[0])


    def _print_expint(self, expr):
        mu, x = expr.args
        if mu != 1:
            return self._print_not_supported(expr)
        return "expint(%s)" % self._print(x)


    def _one_or_two_reversed_args(self, expr):
        assert len(expr.args) <= 2
        return '{name}({args})'.format(
            name=self.known_functions[expr.__class__.__name__],
            args=", ".join([self._print(x) for x in reversed(expr.args)])
        )


    _print_DiracDelta = _print_LambertW = _one_or_two_reversed_args
```
### 8 - sympy/printing/llvmjitcode.py:

Start line: 263, End line: 286

```python
class LLVMJitCode(object):

    def _compile_function(self, strmod):
        global exe_engines
        llmod = llvm.parse_assembly(strmod)

        pmb = llvm.create_pass_manager_builder()
        pmb.opt_level = 2
        pass_manager = llvm.create_module_pass_manager()
        pmb.populate(pass_manager)

        pass_manager.run(llmod)

        target_machine = \
            llvm.Target.from_default_triple().create_target_machine()
        exe_eng = llvm.create_mcjit_compiler(llmod, target_machine)
        exe_eng.finalize_object()
        exe_engines.append(exe_eng)

        if False:
            print("Assembly")
            print(target_machine.emit_assembly(llmod))

        fptr = exe_eng.get_function_address(self.link_name)

        return fptr
```
### 9 - sympy/printing/llvmjitcode.py:

Start line: 245, End line: 261

```python
class LLVMJitCode(object):

    def _convert_expr(self, lj, expr):
        try:
            # Match CSE return data structure.
            if len(expr) == 2:
                tmp_exprs = expr[0]
                final_exprs = expr[1]
                if len(final_exprs) != 1 and self.signature.ret_type == ctypes.c_double:
                    raise NotImplementedError("Return of multiple expressions not supported for this callback")
                for name, e in tmp_exprs:
                    val = lj._print(e)
                    lj._add_tmp_var(name, val)
        except TypeError:
            final_exprs = [expr]

        vals = [lj._print(e) for e in final_exprs]

        return vals
```
### 10 - sympy/printing/octave.py:

Start line: 341, End line: 357

```python
class OctaveCodePrinter(CodePrinter):


    # FIXME: Str/CodePrinter could define each of these to call the _print
    # method from higher up the class hierarchy (see _print_NumberSymbol).
    # Then subclasses like us would not need to repeat all this.
    _print_Matrix = \
        _print_DenseMatrix = \
        _print_MutableDenseMatrix = \
        _print_ImmutableMatrix = \
        _print_ImmutableDenseMatrix = \
        _print_MatrixBase
    _print_MutableSparseMatrix = \
        _print_ImmutableSparseMatrix = \
        _print_SparseMatrix


    def _print_MatrixElement(self, expr):
        return self.parenthesize(expr.parent, PRECEDENCE["Atom"], strict=True) \
            + '(%s, %s)' % (expr.i + 1, expr.j + 1)
```
### 12 - sympy/printing/jscode.py:

Start line: 141, End line: 171

```python
class JavascriptCodePrinter(CodePrinter):

    def _print_Piecewise(self, expr):
        if expr.args[-1].cond != True:
            # We need the last conditional to be a True, otherwise the resulting
            # function may not return a result.
            raise ValueError("All Piecewise expressions must contain an "
                             "(expr, True) statement to be used as a default "
                             "condition. Without one, the generated "
                             "expression may not evaluate to anything under "
                             "some condition.")
        lines = []
        if expr.has(Assignment):
            for i, (e, c) in enumerate(expr.args):
                if i == 0:
                    lines.append("if (%s) {" % self._print(c))
                elif i == len(expr.args) - 1 and c == True:
                    lines.append("else {")
                else:
                    lines.append("else if (%s) {" % self._print(c))
                code0 = self._print(e)
                lines.append(code0)
                lines.append("}")
            return "\n".join(lines)
        else:
            # The piecewise was used in an expression, need to do inline
            # operators. This has the downside that inline operators will
            # not work for statements that span multiple lines (Matrix or
            # Indexed expressions).
            ecpairs = ["((%s) ? (\n%s\n)\n" % (self._print(c), self._print(e))
                    for e, c in expr.args[:-1]]
            last_line = ": (\n%s\n)" % self._print(expr.args[-1].expr)
            return ": ".join(ecpairs) + last_line + " ".join([")"*len(ecpairs)])
```
### 14 - sympy/printing/julia.py:

Start line: 353, End line: 369

```python
class JuliaCodePrinter(CodePrinter):


    # FIXME: Str/CodePrinter could define each of these to call the _print
    # method from higher up the class hierarchy (see _print_NumberSymbol).
    # Then subclasses like us would not need to repeat all this.
    _print_Matrix = \
        _print_DenseMatrix = \
        _print_MutableDenseMatrix = \
        _print_ImmutableMatrix = \
        _print_ImmutableDenseMatrix = \
        _print_MatrixBase
    _print_MutableSparseMatrix = \
        _print_ImmutableSparseMatrix = \
        _print_SparseMatrix


    def _print_MatrixElement(self, expr):
        return self.parenthesize(expr.parent, PRECEDENCE["Atom"], strict=True) \
            + '[%s,%s]' % (expr.i + 1, expr.j + 1)
```
### 15 - sympy/printing/julia.py:

Start line: 1, End line: 43

```python
"""
Julia code printer

The `JuliaCodePrinter` converts SymPy expressions into Julia expressions.

A complete code generator, which uses `julia_code` extensively, can be found
in `sympy.utilities.codegen`.  The `codegen` module can be used to generate
complete source code files.

"""

from __future__ import print_function, division
from sympy.core import Mul, Pow, S, Rational
from sympy.core.compatibility import string_types, range
from sympy.core.mul import _keep_coeff
from sympy.printing.codeprinter import CodePrinter, Assignment
from sympy.printing.precedence import precedence, PRECEDENCE
from re import search

# List of known functions.  First, those that have the same name in
# SymPy and Julia. This is almost certainly incomplete!
known_fcns_src1 = ["sin", "cos", "tan", "cot", "sec", "csc",
                   "asin", "acos", "atan", "acot", "asec", "acsc",
                   "sinh", "cosh", "tanh", "coth", "sech", "csch",
                   "asinh", "acosh", "atanh", "acoth", "asech", "acsch"
                   "sinc", "atan2", "sign", "floor", "log", "exp",
                   "cbrt", "sqrt", "erf", "erfc", "erfi",
                   "factorial", "gamma", "digamma", "trigamma",
                   "polygamma", "beta",
                   "airyai", "airyaiprime", "airybi", "airybiprime",
                   "besselj", "bessely", "besseli", "besselk",
                   "erfinv", "erfcinv"]
# These functions have different names ("Sympy": "Julia"), more
# generally a mapping to (argument_conditions, julia_function).
known_fcns_src2 = {
    "Abs": "abs",
    "ceiling": "ceil",
    "conjugate": "conj",
    "hankel1": "hankelh1",
    "hankel2": "hankelh2",
    "im": "imag",
    "re": "real"
}
```
### 16 - sympy/printing/julia.py:

Start line: 214, End line: 256

```python
class JuliaCodePrinter(CodePrinter):


    def _print_MatPow(self, expr):
        PREC = precedence(expr)
        return '%s^%s' % (self.parenthesize(expr.base, PREC),
                          self.parenthesize(expr.exp, PREC))


    def _print_Pi(self, expr):
        if self._settings["inline"]:
            return "pi"
        else:
            return super(JuliaCodePrinter, self)._print_NumberSymbol(expr)


    def _print_ImaginaryUnit(self, expr):
        return "im"


    def _print_Exp1(self, expr):
        if self._settings["inline"]:
            return "e"
        else:
            return super(JuliaCodePrinter, self)._print_NumberSymbol(expr)


    def _print_EulerGamma(self, expr):
        if self._settings["inline"]:
            return "eulergamma"
        else:
            return super(JuliaCodePrinter, self)._print_NumberSymbol(expr)


    def _print_Catalan(self, expr):
        if self._settings["inline"]:
            return "catalan"
        else:
            return super(JuliaCodePrinter, self)._print_NumberSymbol(expr)


    def _print_GoldenRatio(self, expr):
        if self._settings["inline"]:
            return "golden"
        else:
            return super(JuliaCodePrinter, self)._print_NumberSymbol(expr)
```
### 17 - sympy/printing/octave.py:

Start line: 231, End line: 252

```python
class OctaveCodePrinter(CodePrinter):


    def _print_MatPow(self, expr):
        PREC = precedence(expr)
        return '%s^%s' % (self.parenthesize(expr.base, PREC),
                          self.parenthesize(expr.exp, PREC))


    def _print_Pi(self, expr):
        return 'pi'


    def _print_ImaginaryUnit(self, expr):
        return "1i"


    def _print_Exp1(self, expr):
        return "exp(1)"


    def _print_GoldenRatio(self, expr):
        # FIXME: how to do better, e.g., for octave_code(2*GoldenRatio)?
        #return self._print((1+sqrt(S(5)))/2)
        return "(1+sqrt(5))/2"
```
### 18 - sympy/printing/octave.py:

Start line: 255, End line: 280

```python
class OctaveCodePrinter(CodePrinter):


    def _print_Assignment(self, expr):
        from sympy.functions.elementary.piecewise import Piecewise
        from sympy.tensor.indexed import IndexedBase
        # Copied from codeprinter, but remove special MatrixSymbol treatment
        lhs = expr.lhs
        rhs = expr.rhs
        # We special case assignments that take multiple lines
        if not self._settings["inline"] and isinstance(expr.rhs, Piecewise):
            # Here we modify Piecewise so each expression is now
            # an Assignment, and then continue on the print.
            expressions = []
            conditions = []
            for (e, c) in rhs.args:
                expressions.append(Assignment(lhs, e))
                conditions.append(c)
            temp = Piecewise(*zip(expressions, conditions))
            return self._print(temp)
        if self._settings["contract"] and (lhs.has(IndexedBase) or
                rhs.has(IndexedBase)):
            # Here we check if there is looping to be done, and if so
            # print the required loops.
            return self._doprint_loops(rhs, lhs)
        else:
            lhs_code = self._print(lhs)
            rhs_code = self._print(rhs)
            return self._get_statement("%s = %s" % (lhs_code, rhs_code))
```
### 20 - sympy/printing/jscode.py:

Start line: 173, End line: 204

```python
class JavascriptCodePrinter(CodePrinter):

    def _print_MatrixElement(self, expr):
        return "{0}[{1}]".format(self.parenthesize(expr.parent,
            PRECEDENCE["Atom"], strict=True),
            expr.j + expr.i*expr.parent.shape[1])

    def indent_code(self, code):
        """Accepts a string of code or a list of code lines"""

        if isinstance(code, string_types):
            code_lines = self.indent_code(code.splitlines(True))
            return ''.join(code_lines)

        tab = "   "
        inc_token = ('{', '(', '{\n', '(\n')
        dec_token = ('}', ')')

        code = [ line.lstrip(' \t') for line in code ]

        increase = [ int(any(map(line.endswith, inc_token))) for line in code ]
        decrease = [ int(any(map(line.startswith, dec_token)))
                     for line in code ]

        pretty = []
        level = 0
        for n, line in enumerate(code):
            if line == '' or line == '\n':
                pretty.append(line)
                continue
            level -= decrease[n]
            pretty.append("%s%s" % (tab*level, line))
            level += increase[n]
        return pretty
```
### 22 - sympy/printing/octave.py:

Start line: 211, End line: 228

```python
class OctaveCodePrinter(CodePrinter):


    def _print_Pow(self, expr):
        powsymbol = '^' if all([x.is_number for x in expr.args]) else '.^'

        PREC = precedence(expr)

        if expr.exp == S.Half:
            return "sqrt(%s)" % self._print(expr.base)

        if expr.is_commutative:
            if expr.exp == -S.Half:
                sym = '/' if expr.base.is_number else './'
                return "1" + sym + "sqrt(%s)" % self._print(expr.base)
            if expr.exp == -S.One:
                sym = '/' if expr.base.is_number else './'
                return "1" + sym + "%s" % self.parenthesize(expr.base, PREC)

        return '%s%s%s' % (self.parenthesize(expr.base, PREC), powsymbol,
                           self.parenthesize(expr.exp, PREC))
```
### 23 - sympy/printing/octave.py:

Start line: 525, End line: 701

```python
class OctaveCodePrinter(CodePrinter):


    def _print_zeta(self, expr):
        if len(expr.args) == 1:
            return "zeta(%s)" % self._print(expr.args[0])
        else:
            # Matlab two argument zeta is not equivalent to SymPy's
            return self._print_not_supported(expr)


    def indent_code(self, code):
        """Accepts a string of code or a list of code lines"""

        # code mostly copied from ccode
        if isinstance(code, string_types):
            code_lines = self.indent_code(code.splitlines(True))
            return ''.join(code_lines)

        tab = "  "
        inc_regex = ('^function ', '^if ', '^elseif ', '^else$', '^for ')
        dec_regex = ('^end$', '^elseif ', '^else$')

        # pre-strip left-space from the code
        code = [ line.lstrip(' \t') for line in code ]

        increase = [ int(any([search(re, line) for re in inc_regex]))
                     for line in code ]
        decrease = [ int(any([search(re, line) for re in dec_regex]))
                     for line in code ]

        pretty = []
        level = 0
        for n, line in enumerate(code):
            if line == '' or line == '\n':
                pretty.append(line)
                continue
            level -= decrease[n]
            pretty.append("%s%s" % (tab*level, line))
            level += increase[n]
        return pretty


def octave_code(expr, assign_to=None, **settings):
    # ... other code
```
### 24 - sympy/printing/octave.py:

Start line: 136, End line: 208

```python
class OctaveCodePrinter(CodePrinter):


    def _print_Mul(self, expr):
        # print complex numbers nicely in Octave
        if (expr.is_number and expr.is_imaginary and
                (S.ImaginaryUnit*expr).is_Integer):
            return "%si" % self._print(-S.ImaginaryUnit*expr)

        # cribbed from str.py
        prec = precedence(expr)

        c, e = expr.as_coeff_Mul()
        if c < 0:
            expr = _keep_coeff(-c, e)
            sign = "-"
        else:
            sign = ""

        a = []  # items in the numerator
        b = []  # items that are in the denominator (if any)

        pow_paren = []  # Will collect all pow with more than one base element and exp = -1

        if self.order not in ('old', 'none'):
            args = expr.as_ordered_factors()
        else:
            # use make_args in case expr was something like -x -> x
            args = Mul.make_args(expr)

        # Gather args for numerator/denominator
        for item in args:
            if (item.is_commutative and item.is_Pow and item.exp.is_Rational
                    and item.exp.is_negative):
                if item.exp != -1:
                    b.append(Pow(item.base, -item.exp, evaluate=False))
                else:
                    if len(item.args[0].args) != 1 and isinstance(item.base, Mul):   # To avoid situations like #14160
                        pow_paren.append(item)
                    b.append(Pow(item.base, -item.exp))
            elif item.is_Rational and item is not S.Infinity:
                if item.p != 1:
                    a.append(Rational(item.p))
                if item.q != 1:
                    b.append(Rational(item.q))
            else:
                a.append(item)

        a = a or [S.One]

        a_str = [self.parenthesize(x, prec) for x in a]
        b_str = [self.parenthesize(x, prec) for x in b]

        # To parenthesize Pow with exp = -1 and having more than one Symbol
        for item in pow_paren:
            if item.base in b:
                b_str[b.index(item.base)] = "(%s)" % b_str[b.index(item.base)]

        # from here it differs from str.py to deal with "*" and ".*"
        def multjoin(a, a_str):
            # here we probably are assuming the constants will come first
            r = a_str[0]
            for i in range(1, len(a)):
                mulsym = '*' if a[i-1].is_number else '.*'
                r = r + mulsym + a_str[i]
            return r

        if len(b) == 0:
            return sign + multjoin(a, a_str)
        elif len(b) == 1:
            divsym = '/' if b[0].is_number else './'
            return sign + multjoin(a, a_str) + divsym + b_str[0]
        else:
            divsym = '/' if all([bi.is_number for bi in b]) else './'
            return (sign + multjoin(a, a_str) +
                    divsym + "(%s)" % multjoin(b, b_str))
```
### 25 - sympy/printing/jscode.py:

Start line: 46, End line: 85

```python
class JavascriptCodePrinter(CodePrinter):
    """"A Printer to convert python expressions to strings of javascript code
    """
    printmethod = '_javascript'
    language = 'Javascript'

    _default_settings = {
        'order': None,
        'full_prec': 'auto',
        'precision': 17,
        'user_functions': {},
        'human': True,
        'allow_unknown_functions': True,
        'contract': True
    }

    def __init__(self, settings={}):
        CodePrinter.__init__(self, settings)
        self.known_functions = dict(known_functions)
        userfuncs = settings.get('user_functions', {})
        self.known_functions.update(userfuncs)

    def _rate_index_position(self, p):
        return p*5

    def _get_statement(self, codestring):
        return "%s;" % codestring

    def _get_comment(self, text):
        return "// {0}".format(text)

    def _declare_number_const(self, name, value):
        return "var {0} = {1};".format(name, value.evalf(self._settings['precision']))

    def _format_code(self, lines):
        return self.indent_code(lines)

    def _traverse_matrix_indices(self, mat):
        rows, cols = mat.shape
        return ((i, j) for i in range(rows) for j in range(cols))
```
### 27 - sympy/printing/julia.py:

Start line: 194, End line: 211

```python
class JuliaCodePrinter(CodePrinter):


    def _print_Pow(self, expr):
        powsymbol = '^' if all([x.is_number for x in expr.args]) else '.^'

        PREC = precedence(expr)

        if expr.exp == S.Half:
            return "sqrt(%s)" % self._print(expr.base)

        if expr.is_commutative:
            if expr.exp == -S.Half:
                sym = '/' if expr.base.is_number else './'
                return "1" + sym + "sqrt(%s)" % self._print(expr.base)
            if expr.exp == -S.One:
                sym = '/' if expr.base.is_number else './'
                return "1" + sym + "%s" % self.parenthesize(expr.base, PREC)

        return '%s%s%s' % (self.parenthesize(expr.base, PREC), powsymbol,
                           self.parenthesize(expr.exp, PREC))
```
### 28 - sympy/printing/ccode.py:

Start line: 482, End line: 507

```python
class C89CodePrinter(CodePrinter):

    def _print_Declaration(self, decl):
        from sympy.codegen.cnodes import restrict
        var = decl.variable
        val = var.value
        if var.type == untyped:
            raise ValueError("C does not support untyped variables")

        if isinstance(var, Pointer):
            result = '{vc}{t} *{pc} {r}{s}'.format(
                vc='const ' if value_const in var.attrs else '',
                t=self._print(var.type),
                pc=' const' if pointer_const in var.attrs else '',
                r='restrict ' if restrict in var.attrs else '',
                s=self._print(var.symbol)
            )
        elif isinstance(var, Variable):
            result = '{vc}{t} {s}'.format(
                vc='const ' if value_const in var.attrs else '',
                t=self._print(var.type),
                s=self._print(var.symbol)
            )
        else:
            raise NotImplementedError("Unknown type of var: %s" % type(var))
        if val != None:
            result += ' = %s' % self._print(val)
        return result
```
### 29 - sympy/printing/julia.py:

Start line: 393, End line: 418

```python
class JuliaCodePrinter(CodePrinter):


    def _print_Indexed(self, expr):
        inds = [ self._print(i) for i in expr.indices ]
        return "%s[%s]" % (self._print(expr.base.label), ",".join(inds))


    def _print_Idx(self, expr):
        return self._print(expr.label)


    def _print_Identity(self, expr):
        return "eye(%s)" % self._print(expr.shape[0])


    # Note: as of 2015, Julia doesn't have spherical Bessel functions
    def _print_jn(self, expr):
        from sympy.functions import sqrt, besselj
        x = expr.argument
        expr2 = sqrt(S.Pi/(2*x))*besselj(expr.order + S.Half, x)
        return self._print(expr2)


    def _print_yn(self, expr):
        from sympy.functions import sqrt, bessely
        x = expr.argument
        expr2 = sqrt(S.Pi/(2*x))*bessely(expr.order + S.Half, x)
        return self._print(expr2)
```
### 30 - sympy/printing/julia.py:

Start line: 259, End line: 284

```python
class JuliaCodePrinter(CodePrinter):


    def _print_Assignment(self, expr):
        from sympy.functions.elementary.piecewise import Piecewise
        from sympy.tensor.indexed import IndexedBase
        # Copied from codeprinter, but remove special MatrixSymbol treatment
        lhs = expr.lhs
        rhs = expr.rhs
        # We special case assignments that take multiple lines
        if not self._settings["inline"] and isinstance(expr.rhs, Piecewise):
            # Here we modify Piecewise so each expression is now
            # an Assignment, and then continue on the print.
            expressions = []
            conditions = []
            for (e, c) in rhs.args:
                expressions.append(Assignment(lhs, e))
                conditions.append(c)
            temp = Piecewise(*zip(expressions, conditions))
            return self._print(temp)
        if self._settings["contract"] and (lhs.has(IndexedBase) or
                rhs.has(IndexedBase)):
            # Here we check if there is looping to be done, and if so
            # print the required loops.
            return self._doprint_loops(rhs, lhs)
        else:
            lhs_code = self._print(lhs)
            rhs_code = self._print(rhs)
            return self._get_statement("%s = %s" % (lhs_code, rhs_code))
```
### 32 - sympy/printing/ccode.py:

Start line: 302, End line: 342

```python
class C89CodePrinter(CodePrinter):

    def _print_Rational(self, expr):
        p, q = int(expr.p), int(expr.q)
        suffix = self._get_literal_suffix(real)
        return '%d.0%s/%d.0%s' % (p, suffix, q, suffix)

    def _print_Indexed(self, expr):
        # calculate index for 1d array
        offset = getattr(expr.base, 'offset', S.Zero)
        strides = getattr(expr.base, 'strides', None)
        indices = expr.indices

        if strides is None or isinstance(strides, string_types):
            dims = expr.shape
            shift = S.One
            temp = tuple()
            if strides == 'C' or strides is None:
                traversal = reversed(range(expr.rank))
                indices = indices[::-1]
            elif strides == 'F':
                traversal = range(expr.rank)

            for i in traversal:
                temp += (shift,)
                shift *= dims[i]
            strides = temp
        flat_index = sum([x[0]*x[1] for x in zip(indices, strides)]) + offset
        return "%s[%s]" % (self._print(expr.base.label),
                           self._print(flat_index))

    def _print_Idx(self, expr):
        return self._print(expr.label)

    @_as_macro_if_defined
    def _print_NumberSymbol(self, expr):
        return super(C89CodePrinter, self)._print_NumberSymbol(expr)

    def _print_Infinity(self, expr):
        return 'HUGE_VAL'

    def _print_NegativeInfinity(self, expr):
        return '-HUGE_VAL'
```
### 33 - sympy/printing/ccode.py:

Start line: 622, End line: 635

```python
class _C9XCodePrinter(object):
    # Move these methods to C99CodePrinter when removing CCodePrinter
    def _get_loop_opening_ending(self, indices):
        open_lines = []
        close_lines = []
        loopstart = "for (int %(var)s=%(start)s; %(var)s<%(end)s; %(var)s++){"  # C99
        for i in indices:
            # C arrays start at 0 and end at dimension-1
            open_lines.append(loopstart % {
                'var': self._print(i.label),
                'start': self._print(i.lower),
                'end': self._print(i.upper + 1)})
            close_lines.append("}")
        return open_lines, close_lines
```
### 35 - sympy/printing/julia.py:

Start line: 119, End line: 191

```python
class JuliaCodePrinter(CodePrinter):


    def _print_Mul(self, expr):
        # print complex numbers nicely in Julia
        if (expr.is_number and expr.is_imaginary and
                expr.as_coeff_Mul()[0].is_integer):
            return "%sim" % self._print(-S.ImaginaryUnit*expr)

        # cribbed from str.py
        prec = precedence(expr)

        c, e = expr.as_coeff_Mul()
        if c < 0:
            expr = _keep_coeff(-c, e)
            sign = "-"
        else:
            sign = ""

        a = []  # items in the numerator
        b = []  # items that are in the denominator (if any)

        pow_paren = []  # Will collect all pow with more than one base element and exp = -1

        if self.order not in ('old', 'none'):
            args = expr.as_ordered_factors()
        else:
            # use make_args in case expr was something like -x -> x
            args = Mul.make_args(expr)

        # Gather args for numerator/denominator
        for item in args:
            if (item.is_commutative and item.is_Pow and item.exp.is_Rational
                    and item.exp.is_negative):
                if item.exp != -1:
                    b.append(Pow(item.base, -item.exp, evaluate=False))
                else:
                    if len(item.args[0].args) != 1 and isinstance(item.base, Mul):   # To avoid situations like #14160
                        pow_paren.append(item)
                    b.append(Pow(item.base, -item.exp))
            elif item.is_Rational and item is not S.Infinity:
                if item.p != 1:
                    a.append(Rational(item.p))
                if item.q != 1:
                    b.append(Rational(item.q))
            else:
                a.append(item)

        a = a or [S.One]

        a_str = [self.parenthesize(x, prec) for x in a]
        b_str = [self.parenthesize(x, prec) for x in b]

        # To parenthesize Pow with exp = -1 and having more than one Symbol
        for item in pow_paren:
            if item.base in b:
                b_str[b.index(item.base)] = "(%s)" % b_str[b.index(item.base)]

        # from here it differs from str.py to deal with "*" and ".*"
        def multjoin(a, a_str):
            # here we probably are assuming the constants will come first
            r = a_str[0]
            for i in range(1, len(a)):
                mulsym = '*' if a[i-1].is_number else '.*'
                r = r + mulsym + a_str[i]
            return r

        if len(b) == 0:
            return sign + multjoin(a, a_str)
        elif len(b) == 1:
            divsym = '/' if b[0].is_number else './'
            return sign + multjoin(a, a_str) + divsym + b_str[0]
        else:
            divsym = '/' if all([bi.is_number for bi in b]) else './'
            return (sign + multjoin(a, a_str) +
                    divsym + "(%s)" % multjoin(b, b_str))
```
### 36 - sympy/printing/jscode.py:

Start line: 1, End line: 43

```python
"""
Javascript code printer

The JavascriptCodePrinter converts single sympy expressions into single
Javascript expressions, using the functions defined in the Javascript
Math object where possible.

"""

from __future__ import print_function, division

from sympy.core import S
from sympy.codegen.ast import Assignment
from sympy.printing.codeprinter import CodePrinter
from sympy.printing.precedence import precedence, PRECEDENCE
from sympy.core.compatibility import string_types, range


# dictionary mapping sympy function to (argument_conditions, Javascript_function).
# Used in JavascriptCodePrinter._print_Function(self)
known_functions = {
    'Abs': 'Math.abs',
    'acos': 'Math.acos',
    'acosh': 'Math.acosh',
    'asin': 'Math.asin',
    'asinh': 'Math.asinh',
    'atan': 'Math.atan',
    'atan2': 'Math.atan2',
    'atanh': 'Math.atanh',
    'ceiling': 'Math.ceil',
    'cos': 'Math.cos',
    'cosh': 'Math.cosh',
    'exp': 'Math.exp',
    'floor': 'Math.floor',
    'log': 'Math.log',
    'Max': 'Math.max',
    'Min': 'Math.min',
    'sign': 'Math.sign',
    'sin': 'Math.sin',
    'sinh': 'Math.sinh',
    'tan': 'Math.tan',
    'tanh': 'Math.tanh',
}
```
### 37 - sympy/printing/octave.py:

Start line: 478, End line: 522

```python
class OctaveCodePrinter(CodePrinter):


    def _nested_binary_math_func(self, expr):
        return '{name}({arg1}, {arg2})'.format(
            name=self.known_functions[expr.__class__.__name__],
            arg1=self._print(expr.args[0]),
            arg2=self._print(expr.func(*expr.args[1:]))
            )

    _print_Max = _print_Min = _nested_binary_math_func


    def _print_Piecewise(self, expr):
        if expr.args[-1].cond != True:
            # We need the last conditional to be a True, otherwise the resulting
            # function may not return a result.
            raise ValueError("All Piecewise expressions must contain an "
                             "(expr, True) statement to be used as a default "
                             "condition. Without one, the generated "
                             "expression may not evaluate to anything under "
                             "some condition.")
        lines = []
        if self._settings["inline"]:
            # Express each (cond, expr) pair in a nested Horner form:
            #   (condition) .* (expr) + (not cond) .* (<others>)
            # Expressions that result in multiple statements won't work here.
            ecpairs = ["({0}).*({1}) + (~({0})).*(".format
                       (self._print(c), self._print(e))
                       for e, c in expr.args[:-1]]
            elast = "%s" % self._print(expr.args[-1].expr)
            pw = " ...\n".join(ecpairs) + elast + ")"*len(ecpairs)
            # Note: current need these outer brackets for 2*pw.  Would be
            # nicer to teach parenthesize() to do this for us when needed!
            return "(" + pw + ")"
        else:
            for i, (e, c) in enumerate(expr.args):
                if i == 0:
                    lines.append("if (%s)" % self._print(c))
                elif i == len(expr.args) - 1 and c == True:
                    lines.append("else")
                else:
                    lines.append("elseif (%s)" % self._print(c))
                code0 = self._print(e)
                lines.append(code0)
                if i == len(expr.args) - 1:
                    lines.append("end")
            return "\n".join(lines)
```
### 39 - sympy/printing/ccode.py:

Start line: 719, End line: 737

```python
for k in ('Abs Sqrt exp exp2 expm1 log log10 log2 log1p Cbrt hypot fma Mod'
          ' loggamma sin cos tan asin acos atan atan2 sinh cosh tanh asinh acosh '
          'atanh erf erfc loggamma gamma ceiling floor').split():
    setattr(C99CodePrinter, '_print_%s' % k, C99CodePrinter._print_math_func)


class C11CodePrinter(C99CodePrinter):

    @requires(headers={'stdalign.h'})
    def _print_alignof(self, expr):
        arg, = expr.args
        return 'alignof(%s)' % self._print(arg)


c_code_printers = {
    'c89': C89CodePrinter,
    'c99': C99CodePrinter,
    'c11': C11CodePrinter
}
```
### 40 - sympy/printing/ccode.py:

Start line: 466, End line: 480

```python
class C89CodePrinter(CodePrinter):

    def _get_func_suffix(self, type_):
        return self.type_func_suffixes[self.type_aliases.get(type_, type_)]

    def _get_literal_suffix(self, type_):
        return self.type_literal_suffixes[self.type_aliases.get(type_, type_)]

    def _get_math_macro_suffix(self, type_):
        alias = self.type_aliases.get(type_, type_)
        dflt = self.type_math_macro_suffixes.get(alias, '')
        return self.type_math_macro_suffixes.get(type_, dflt)

    def _print_Type(self, type_):
        self.headers.update(self.type_headers.get(type_, set()))
        self.macros.update(self.type_macros.get(type_, set()))
        return self._print(self.type_mappings.get(type_, type_.name))
```
### 41 - sympy/printing/ccode.py:

Start line: 418, End line: 437

```python
class C89CodePrinter(CodePrinter):

    def _print_sign(self, func):
        return '((({0}) > 0) - (({0}) < 0))'.format(self._print(func.args[0]))

    def _print_Max(self, expr):
        if "Max" in self.known_functions:
            return self._print_Function(expr)
        from sympy import Max
        if len(expr.args) == 1:
            return self._print(expr.args[0])
        return "((%(a)s > %(b)s) ? %(a)s : %(b)s)" % {
            'a': expr.args[0], 'b': self._print(Max(*expr.args[1:]))}

    def _print_Min(self, expr):
        if "Min" in self.known_functions:
            return self._print_Function(expr)
        from sympy import Min
        if len(expr.args) == 1:
            return self._print(expr.args[0])
        return "((%(a)s < %(b)s) ? %(a)s : %(b)s)" % {
            'a': expr.args[0], 'b': self._print(Min(*expr.args[1:]))}
```
### 42 - sympy/printing/octave.py:

Start line: 62, End line: 133

```python
class OctaveCodePrinter(CodePrinter):
    """
    A printer to convert expressions to strings of Octave/Matlab code.
    """
    printmethod = "_octave"
    language = "Octave"

    _operators = {
        'and': '&',
        'or': '|',
        'not': '~',
    }

    _default_settings = {
        'order': None,
        'full_prec': 'auto',
        'precision': 17,
        'user_functions': {},
        'human': True,
        'allow_unknown_functions': True,
        'contract': True,
        'inline': True,
    }
    # Note: contract is for expressing tensors as loops (if True), or just
    # assignment (if False).  FIXME: this should be looked a more carefully
    # for Octave.


    def __init__(self, settings={}):
        super(OctaveCodePrinter, self).__init__(settings)
        self.known_functions = dict(zip(known_fcns_src1, known_fcns_src1))
        self.known_functions.update(dict(known_fcns_src2))
        userfuncs = settings.get('user_functions', {})
        self.known_functions.update(userfuncs)


    def _rate_index_position(self, p):
        return p*5


    def _get_statement(self, codestring):
        return "%s;" % codestring


    def _get_comment(self, text):
        return "% {0}".format(text)


    def _declare_number_const(self, name, value):
        return "{0} = {1};".format(name, value)


    def _format_code(self, lines):
        return self.indent_code(lines)


    def _traverse_matrix_indices(self, mat):
        # Octave uses Fortran order (column-major)
        rows, cols = mat.shape
        return ((i, j) for j in range(cols) for i in range(rows))


    def _get_loop_opening_ending(self, indices):
        open_lines = []
        close_lines = []
        for i in indices:
            # Octave arrays start at 1 and end at dimension
            var, start, stop = map(self._print,
                    [i.label, i.lower + 1, i.upper + 1])
            open_lines.append("for %s = %s:%s" % (var, start, stop))
            close_lines.append("end")
        return open_lines, close_lines
```
### 43 - sympy/printing/julia.py:

Start line: 458, End line: 624

```python
class JuliaCodePrinter(CodePrinter):


    def indent_code(self, code):
        """Accepts a string of code or a list of code lines"""

        # code mostly copied from ccode
        if isinstance(code, string_types):
            code_lines = self.indent_code(code.splitlines(True))
            return ''.join(code_lines)

        tab = "    "
        inc_regex = ('^function ', '^if ', '^elseif ', '^else$', '^for ')
        dec_regex = ('^end$', '^elseif ', '^else$')

        # pre-strip left-space from the code
        code = [ line.lstrip(' \t') for line in code ]

        increase = [ int(any([search(re, line) for re in inc_regex]))
                     for line in code ]
        decrease = [ int(any([search(re, line) for re in dec_regex]))
                     for line in code ]

        pretty = []
        level = 0
        for n, line in enumerate(code):
            if line == '' or line == '\n':
                pretty.append(line)
                continue
            level -= decrease[n]
            pretty.append("%s%s" % (tab*level, line))
            level += increase[n]
        return pretty


def julia_code(expr, assign_to=None, **settings):
    # ... other code
```
### 45 - sympy/printing/ccode.py:

Start line: 407, End line: 416

```python
class C89CodePrinter(CodePrinter):

    def _print_For(self, expr):
        target = self._print(expr.target)
        if isinstance(expr.iterable, Range):
            start, stop, step = expr.iterable.args
        else:
            raise NotImplementedError("Only iterable currently supported is Range")
        body = self._print(expr.body)
        return ('for ({target} = {start}; {target} < {stop}; {target} += '
                '{step}) {{\n{body}\n}}').format(target=target, start=start,
                stop=stop, step=step, body=body)
```
### 46 - sympy/printing/ccode.py:

Start line: 377, End line: 405

```python
class C89CodePrinter(CodePrinter):

    def _print_ITE(self, expr):
        from sympy.functions import Piecewise
        _piecewise = Piecewise((expr.args[1], expr.args[0]), (expr.args[2], True))
        return self._print(_piecewise)

    def _print_MatrixElement(self, expr):
        return "{0}[{1}]".format(self.parenthesize(expr.parent, PRECEDENCE["Atom"],
            strict=True), expr.j + expr.i*expr.parent.shape[1])

    def _print_Symbol(self, expr):
        name = super(C89CodePrinter, self)._print_Symbol(expr)
        if expr in self._settings['dereference']:
            return '(*{0})'.format(name)
        else:
            return name

    def _print_Relational(self, expr):
        lhs_code = self._print(expr.lhs)
        rhs_code = self._print(expr.rhs)
        op = expr.rel_op
        return ("{0} {1} {2}").format(lhs_code, op, rhs_code)

    def _print_sinc(self, expr):
        from sympy.functions.elementary.trigonometric import sin
        from sympy.core.relational import Ne
        from sympy.functions import Piecewise
        _piecewise = Piecewise(
            (sin(expr.args[0]) / expr.args[0], Ne(expr.args[0], 0)), (1, True))
        return self._print(_piecewise)
```
### 48 - sympy/printing/ccode.py:

Start line: 286, End line: 300

```python
class C89CodePrinter(CodePrinter):

    @_as_macro_if_defined
    def _print_Pow(self, expr):
        if "Pow" in self.known_functions:
            return self._print_Function(expr)
        PREC = precedence(expr)
        suffix = self._get_func_suffix(real)
        if expr.exp == -1:
            return '1.0%s/%s' % (suffix.upper(), self.parenthesize(expr.base, PREC))
        elif expr.exp == 0.5:
            return '%ssqrt%s(%s)' % (self._ns, suffix, self._print(expr.base))
        elif expr.exp == S.One/3 and self.standard != 'C89':
            return '%scbrt%s(%s)' % (self._ns, suffix, self._print(expr.base))
        else:
            return '%spow%s(%s, %s)' % (self._ns, suffix, self._print(expr.base),
                                   self._print(expr.exp))
```
### 49 - sympy/printing/octave.py:

Start line: 330, End line: 338

```python
class OctaveCodePrinter(CodePrinter):


    def _print_SparseMatrix(self, A):
        from sympy.matrices import Matrix
        L = A.col_list();
        # make row vectors of the indices and entries
        I = Matrix([[k[0] + 1 for k in L]])
        J = Matrix([[k[1] + 1 for k in L]])
        AIJ = Matrix([[k[2] for k in L]])
        return "sparse(%s, %s, %s, %s, %s)" % (self._print(I), self._print(J),
                                            self._print(AIJ), A.rows, A.cols)
```
### 53 - sympy/printing/ccode.py:

Start line: 683, End line: 716

```python
class C99CodePrinter(_C9XCodePrinter, C89CodePrinter):

    @requires(headers={'math.h'}, libraries={'m'})
    @_as_macro_if_defined
    def _print_math_func(self, expr, nest=False):
        known = self.known_functions[expr.__class__.__name__]
        if not isinstance(known, string_types):
            for cb, name in known:
                if cb(*expr.args):
                    known = name
                    break
            else:
                raise ValueError("No matching printer")
        try:
            return known(self, *expr.args)
        except TypeError:
            suffix = self._get_func_suffix(real) if self._ns + known in self._prec_funcs else ''

        if nest:
            args = self._print(expr.args[0])
            if len(expr.args) > 1:
                args += ', %s' % self._print(expr.func(*expr.args[1:]))
        else:
            args = ', '.join(map(lambda arg: self._print(arg), expr.args))
        return '{ns}{name}{suffix}({args})'.format(
            ns=self._ns,
            name=known,
            suffix=suffix,
            args=args
        )

    def _print_Max(self, expr):
        return self._print_math_func(expr, nest=True)

    def _print_Min(self, expr):
        return self._print_math_func(expr, nest=True)
```
### 54 - sympy/printing/octave.py:

Start line: 360, End line: 378

```python
class OctaveCodePrinter(CodePrinter):


    def _print_MatrixSlice(self, expr):
        def strslice(x, lim):
            l = x[0] + 1
            h = x[1]
            step = x[2]
            lstr = self._print(l)
            hstr = 'end' if h == lim else self._print(h)
            if step == 1:
                if l == 1 and h == lim:
                    return ':'
                if l == h:
                    return lstr
                else:
                    return lstr + ':' + hstr
            else:
                return ':'.join((lstr, self._print(step), hstr))
        return (self._print(expr.parent) + '(' +
                strslice(expr.rowslice, expr.parent.shape[0]) + ', ' +
                strslice(expr.colslice, expr.parent.shape[1]) + ')')
```
### 56 - sympy/printing/jscode.py:

Start line: 207, End line: 321

```python
def jscode(expr, assign_to=None, **settings):
    """Converts an expr to a string of javascript code

    Parameters
    ==========

    expr : Expr
        A sympy expression to be converted.
    assign_to : optional
        When given, the argument is used as the name of the variable to which
        the expression is assigned. Can be a string, ``Symbol``,
        ``MatrixSymbol``, or ``Indexed`` type. This is helpful in case of
        line-wrapping, or for expressions that generate multi-line statements.
    precision : integer, optional
        The precision for numbers such as pi [default=15].
    user_functions : dict, optional
        A dictionary where keys are ``FunctionClass`` instances and values are
        their string representations. Alternatively, the dictionary value can
        be a list of tuples i.e. [(argument_test, js_function_string)]. See
        below for examples.
    human : bool, optional
        If True, the result is a single string that may contain some constant
        declarations for the number symbols. If False, the same information is
        returned in a tuple of (symbols_to_declare, not_supported_functions,
        code_text). [default=True].
    contract: bool, optional
        If True, ``Indexed`` instances are assumed to obey tensor contraction
        rules and the corresponding nested loops over indices are generated.
        Setting contract=False will not generate loops, instead the user is
        responsible to provide values for the indices in the code.
        [default=True].

    Examples
    ========

    >>> from sympy import jscode, symbols, Rational, sin, ceiling, Abs
    >>> x, tau = symbols("x, tau")
    >>> jscode((2*tau)**Rational(7, 2))
    '8*Math.sqrt(2)*Math.pow(tau, 7/2)'
    >>> jscode(sin(x), assign_to="s")
    's = Math.sin(x);'

    Custom printing can be defined for certain types by passing a dictionary of
    "type" : "function" to the ``user_functions`` kwarg. Alternatively, the
    dictionary value can be a list of tuples i.e. [(argument_test,
    js_function_string)].

    >>> custom_functions = {
    ...   "ceiling": "CEIL",
    ...   "Abs": [(lambda x: not x.is_integer, "fabs"),
    ...           (lambda x: x.is_integer, "ABS")]
    ... }
    >>> jscode(Abs(x) + ceiling(x), user_functions=custom_functions)
    'fabs(x) + CEIL(x)'

    ``Piecewise`` expressions are converted into conditionals. If an
    ``assign_to`` variable is provided an if statement is created, otherwise
    the ternary operator is used. Note that if the ``Piecewise`` lacks a
    default term, represented by ``(expr, True)`` then an error will be thrown.
    This is to prevent generating an expression that may not evaluate to
    anything.

    >>> from sympy import Piecewise
    >>> expr = Piecewise((x + 1, x > 0), (x, True))
    >>> print(jscode(expr, tau))
    if (x > 0) {
       tau = x + 1;
    }
    else {
       tau = x;
    }

    Support for loops is provided through ``Indexed`` types. With
    ``contract=True`` these expressions will be turned into loops, whereas
    ``contract=False`` will just print the assignment expression that should be
    looped over:

    >>> from sympy import Eq, IndexedBase, Idx
    >>> len_y = 5
    >>> y = IndexedBase('y', shape=(len_y,))
    >>> t = IndexedBase('t', shape=(len_y,))
    >>> Dy = IndexedBase('Dy', shape=(len_y-1,))
    >>> i = Idx('i', len_y-1)
    >>> e=Eq(Dy[i], (y[i+1]-y[i])/(t[i+1]-t[i]))
    >>> jscode(e.rhs, assign_to=e.lhs, contract=False)
    'Dy[i] = (y[i + 1] - y[i])/(t[i + 1] - t[i]);'

    Matrices are also supported, but a ``MatrixSymbol`` of the same dimensions
    must be provided to ``assign_to``. Note that any expression that can be
    generated normally can also exist inside a Matrix:

    >>> from sympy import Matrix, MatrixSymbol
    >>> mat = Matrix([x**2, Piecewise((x + 1, x > 0), (x, True)), sin(x)])
    >>> A = MatrixSymbol('A', 3, 1)
    >>> print(jscode(mat, A))
    A[0] = Math.pow(x, 2);
    if (x > 0) {
       A[1] = x + 1;
    }
    else {
       A[1] = x;
    }
    A[2] = Math.sin(x);
    """

    return JavascriptCodePrinter(settings).doprint(expr, assign_to)


def print_jscode(expr, **settings):
    """Prints the Javascript representation of the given expression.

       See jscode for the meaning of the optional arguments.
    """
    print(jscode(expr, **settings))
```
### 58 - sympy/printing/ccode.py:

Start line: 638, End line: 649

```python
@deprecated(
    last_supported_version='1.0',
    useinstead="C89CodePrinter or C99CodePrinter, e.g. ccode(..., standard='C99')",
    issue=12220,
    deprecated_since_version='1.1')
class CCodePrinter(_C9XCodePrinter, C89CodePrinter):
    """
    Deprecated.

    Alias for C89CodePrinter, for backwards compatibility.
    """
    _kf = _known_functions_C9X  # known_functions-dict to copy
```
### 60 - sympy/printing/octave.py:

Start line: 566, End line: 710

```python
def octave_code(expr, assign_to=None, **settings):
    r"""Converts `expr` to a string of Octave (or Matlab) code.

    The string uses a subset of the Octave language for Matlab compatibility.

    Parameters
    ==========

    expr : Expr
        A sympy expression to be converted.
    assign_to : optional
        When given, the argument is used as the name of the variable to which
        the expression is assigned.  Can be a string, ``Symbol``,
        ``MatrixSymbol``, or ``Indexed`` type.  This can be helpful for
        expressions that generate multi-line statements.
    precision : integer, optional
        The precision for numbers such as pi  [default=16].
    user_functions : dict, optional
        A dictionary where keys are ``FunctionClass`` instances and values are
        their string representations.  Alternatively, the dictionary value can
        be a list of tuples i.e. [(argument_test, cfunction_string)].  See
        below for examples.
    human : bool, optional
        If True, the result is a single string that may contain some constant
        declarations for the number symbols.  If False, the same information is
        returned in a tuple of (symbols_to_declare, not_supported_functions,
        code_text).  [default=True].
    contract: bool, optional
        If True, ``Indexed`` instances are assumed to obey tensor contraction
        rules and the corresponding nested loops over indices are generated.
        Setting contract=False will not generate loops, instead the user is
        responsible to provide values for the indices in the code.
        [default=True].
    inline: bool, optional
        If True, we try to create single-statement code instead of multiple
        statements.  [default=True].

    Examples
    ========

    >>> from sympy import octave_code, symbols, sin, pi
    >>> x = symbols('x')
    >>> octave_code(sin(x).series(x).removeO())
    'x.^5/120 - x.^3/6 + x'

    >>> from sympy import Rational, ceiling, Abs
    >>> x, y, tau = symbols("x, y, tau")
    >>> octave_code((2*tau)**Rational(7, 2))
    '8*sqrt(2)*tau.^(7/2)'

    Note that element-wise (Hadamard) operations are used by default between
    symbols.  This is because its very common in Octave to write "vectorized"
    code.  It is harmless if the values are scalars.

    >>> octave_code(sin(pi*x*y), assign_to="s")
    's = sin(pi*x.*y);'

    If you need a matrix product "*" or matrix power "^", you can specify the
    symbol as a ``MatrixSymbol``.

    >>> from sympy import Symbol, MatrixSymbol
    >>> n = Symbol('n', integer=True, positive=True)
    >>> A = MatrixSymbol('A', n, n)
    >>> octave_code(3*pi*A**3)
    '(3*pi)*A^3'

    This class uses several rules to decide which symbol to use a product.
    Pure numbers use "*", Symbols use ".*" and MatrixSymbols use "*".
    A HadamardProduct can be used to specify componentwise multiplication ".*"
    of two MatrixSymbols.  There is currently there is no easy way to specify
    scalar symbols, so sometimes the code might have some minor cosmetic
    issues.  For example, suppose x and y are scalars and A is a Matrix, then
    while a human programmer might write "(x^2*y)*A^3", we generate:

    >>> octave_code(x**2*y*A**3)
    '(x.^2.*y)*A^3'

    Matrices are supported using Octave inline notation.  When using
    ``assign_to`` with matrices, the name can be specified either as a string
    or as a ``MatrixSymbol``.  The dimensions must align in the latter case.

    >>> from sympy import Matrix, MatrixSymbol
    >>> mat = Matrix([[x**2, sin(x), ceiling(x)]])
    >>> octave_code(mat, assign_to='A')
    'A = [x.^2 sin(x) ceil(x)];'

    ``Piecewise`` expressions are implemented with logical masking by default.
    Alternatively, you can pass "inline=False" to use if-else conditionals.
    Note that if the ``Piecewise`` lacks a default term, represented by
    ``(expr, True)`` then an error will be thrown.  This is to prevent
    generating an expression that may not evaluate to anything.

    >>> from sympy import Piecewise
    >>> pw = Piecewise((x + 1, x > 0), (x, True))
    >>> octave_code(pw, assign_to=tau)
    'tau = ((x > 0).*(x + 1) + (~(x > 0)).*(x));'

    Note that any expression that can be generated normally can also exist
    inside a Matrix:

    >>> mat = Matrix([[x**2, pw, sin(x)]])
    >>> octave_code(mat, assign_to='A')
    'A = [x.^2 ((x > 0).*(x + 1) + (~(x > 0)).*(x)) sin(x)];'

    Custom printing can be defined for certain types by passing a dictionary of
    "type" : "function" to the ``user_functions`` kwarg.  Alternatively, the
    dictionary value can be a list of tuples i.e., [(argument_test,
    cfunction_string)].  This can be used to call a custom Octave function.

    >>> from sympy import Function
    >>> f = Function('f')
    >>> g = Function('g')
    >>> custom_functions = {
    ...   "f": "existing_octave_fcn",
    ...   "g": [(lambda x: x.is_Matrix, "my_mat_fcn"),
    ...         (lambda x: not x.is_Matrix, "my_fcn")]
    ... }
    >>> mat = Matrix([[1, x]])
    >>> octave_code(f(x) + g(x) + g(mat), user_functions=custom_functions)
    'existing_octave_fcn(x) + my_fcn(x) + my_mat_fcn([1 x])'

    Support for loops is provided through ``Indexed`` types. With
    ``contract=True`` these expressions will be turned into loops, whereas
    ``contract=False`` will just print the assignment expression that should be
    looped over:

    >>> from sympy import Eq, IndexedBase, Idx, ccode
    >>> len_y = 5
    >>> y = IndexedBase('y', shape=(len_y,))
    >>> t = IndexedBase('t', shape=(len_y,))
    >>> Dy = IndexedBase('Dy', shape=(len_y-1,))
    >>> i = Idx('i', len_y-1)
    >>> e = Eq(Dy[i], (y[i+1]-y[i])/(t[i+1]-t[i]))
    >>> octave_code(e.rhs, assign_to=e.lhs, contract=False)
    'Dy(i) = (y(i + 1) - y(i))./(t(i + 1) - t(i));'
    """
    return OctaveCodePrinter(settings).doprint(expr, assign_to)


def print_octave_code(expr, **settings):
    """Prints the Octave (or Matlab) representation of the given expression.

    See `octave_code` for the meaning of the optional arguments.
    """
    print(octave_code(expr, **settings))
```
### 61 - sympy/printing/octave.py:

Start line: 317, End line: 327

```python
class OctaveCodePrinter(CodePrinter):


    def _print_MatrixBase(self, A):
        # Handle zero dimensions:
        if (A.rows, A.cols) == (0, 0):
            return '[]'
        elif A.rows == 0 or A.cols == 0:
            return 'zeros(%s, %s)' % (A.rows, A.cols)
        elif (A.rows, A.cols) == (1, 1):
            # Octave does not distinguish between scalars and 1x1 matrices
            return self._print(A[0, 0])
        return "[%s]" % "; ".join(" ".join([self._print(a) for a in A[r, :]])
                                  for r in range(A.rows))
```
### 62 - sympy/printing/fcode.py:

Start line: 342, End line: 376

```python
class FCodePrinter(CodePrinter):

    def _print_Rational(self, expr):
        p, q = int(expr.p), int(expr.q)
        return "%d.0d0/%d.0d0" % (p, q)

    def _print_Float(self, expr):
        printed = CodePrinter._print_Float(self, expr)
        e = printed.find('e')
        if e > -1:
            return "%sd%s" % (printed[:e], printed[e + 1:])
        return "%sd0" % printed

    def _print_Indexed(self, expr):
        inds = [ self._print(i) for i in expr.indices ]
        return "%s(%s)" % (self._print(expr.base.label), ", ".join(inds))

    def _print_Idx(self, expr):
        return self._print(expr.label)

    def _print_AugmentedAssignment(self, expr):
        lhs_code = self._print(expr.lhs)
        rhs_code = self._print(expr.rhs)
        return self._get_statement("{0} = {0} {1} {2}".format(
            *map(lambda arg: self._print(arg),
                 [lhs_code, expr.binop, rhs_code])))

    def _print_sum_(self, sm):
        params = self._print(sm.array)
        if sm.dim != None:
            params += ', ' + self._print(sm.dim)
        if sm.mask != None:
            params += ', mask=' + self._print(sm.mask)
        return '%s(%s)' % (sm.__class__.__name__.rstrip('_'), params)

    def _print_product_(self, prod):
        return self._print_sum_(prod)
```
### 63 - sympy/printing/julia.py:

Start line: 342, End line: 350

```python
class JuliaCodePrinter(CodePrinter):


    def _print_SparseMatrix(self, A):
        from sympy.matrices import Matrix
        L = A.col_list();
        # make row vectors of the indices and entries
        I = Matrix([k[0] + 1 for k in L])
        J = Matrix([k[1] + 1 for k in L])
        AIJ = Matrix([k[2] for k in L])
        return "sparse(%s, %s, %s, %s, %s)" % (self._print(I), self._print(J),
                                            self._print(AIJ), A.rows, A.cols)
```
### 64 - sympy/printing/codeprinter.py:

Start line: 331, End line: 362

```python
class CodePrinter(StrPrinter):

    def _print_AugmentedAssignment(self, expr):
        lhs_code = self._print(expr.lhs)
        rhs_code = self._print(expr.rhs)
        return self._get_statement("{0} {1} {2}".format(
            *map(lambda arg: self._print(arg),
                 [lhs_code, expr.op, rhs_code])))

    def _print_FunctionCall(self, expr):
        return '%s(%s)' % (
            expr.name,
            ', '.join(map(lambda arg: self._print(arg),
                          expr.function_args)))

    def _print_Variable(self, expr):
        return self._print(expr.symbol)

    def _print_Statement(self, expr):
        arg, = expr.args
        return self._get_statement(self._print(arg))

    def _print_Symbol(self, expr):

        name = super(CodePrinter, self)._print_Symbol(expr)

        if name in self.reserved_words:
            if self._settings['error_on_reserved']:
                msg = ('This expression includes the symbol "{}" which is a '
                       'reserved keyword in this language.')
                raise ValueError(msg.format(name))
            return name + self._settings['reserved_word_suffix']
        else:
            return name
```
### 66 - sympy/printing/ccode.py:

Start line: 344, End line: 375

```python
class C89CodePrinter(CodePrinter):

    def _print_Piecewise(self, expr):
        if expr.args[-1].cond != True:
            # We need the last conditional to be a True, otherwise the resulting
            # function may not return a result.
            raise ValueError("All Piecewise expressions must contain an "
                             "(expr, True) statement to be used as a default "
                             "condition. Without one, the generated "
                             "expression may not evaluate to anything under "
                             "some condition.")
        lines = []
        if expr.has(Assignment):
            for i, (e, c) in enumerate(expr.args):
                if i == 0:
                    lines.append("if (%s) {" % self._print(c))
                elif i == len(expr.args) - 1 and c == True:
                    lines.append("else {")
                else:
                    lines.append("else if (%s) {" % self._print(c))
                code0 = self._print(e)
                lines.append(code0)
                lines.append("}")
            return "\n".join(lines)
        else:
            # The piecewise was used in an expression, need to do inline
            # operators. This has the downside that inline operators will
            # not work for statements that span multiple lines (Matrix or
            # Indexed expressions).
            ecpairs = ["((%s) ? (\n%s\n)\n" % (self._print(c),
                                               self._print(e))
                    for e, c in expr.args[:-1]]
            last_line = ": (\n%s\n)" % self._print(expr.args[-1].expr)
            return ": ".join(ecpairs) + last_line + " ".join([")"*len(ecpairs)])
```
### 67 - sympy/printing/fcode.py:

Start line: 323, End line: 340

```python
class FCodePrinter(CodePrinter):

    def _print_Pow(self, expr):
        PREC = precedence(expr)
        if expr.exp == -1:
            return '%s/%s' % (
                self._print(literal_dp(1)),
                self.parenthesize(expr.base, PREC)
            )
        elif expr.exp == 0.5:
            if expr.base.is_integer:
                # Fortran intrinsic sqrt() does not accept integer argument
                if expr.base.is_Number:
                    return 'sqrt(%s.0d0)' % self._print(expr.base)
                else:
                    return 'sqrt(dble(%s))' % self._print(expr.base)
            else:
                return 'sqrt(%s)' % self._print(expr.base)
        else:
            return CodePrinter._print_Pow(self, expr)
```
### 70 - sympy/printing/fcode.py:

Start line: 439, End line: 469

```python
class FCodePrinter(CodePrinter):

    def _print_Declaration(self, expr):
        var = expr.variable
        val = var.value
        dim = var.attr_params('dimension')
        intents = [intent in var.attrs for intent in (intent_in, intent_out, intent_inout)]
        if intents.count(True) == 0:
            intent = ''
        elif intents.count(True) == 1:
            intent = ', intent(%s)' % ['in', 'out', 'inout'][intents.index(True)]
        else:
            raise ValueError("Multiple intents specified for %s" % self)

        if isinstance(var, Pointer):
            raise NotImplementedError("Pointers are not available by default in Fortran.")
        if self._settings["standard"] >= 90:
            result = '{t}{vc}{dim}{intent}{alloc} :: {s}'.format(
                t=self._print(var.type),
                vc=', parameter' if value_const in var.attrs else '',
                dim=', dimension(%s)' % ', '.join(map(lambda arg: self._print(arg), dim)) if dim else '',
                intent=intent,
                alloc=', allocatable' if allocatable in var.attrs else '',
                s=self._print(var.symbol)
            )
            if val != None:
                result += ' = %s' % self._print(val)
        else:
            if value_const in var.attrs or val:
                raise NotImplementedError("F77 init./parameter statem. req. multiple lines.")
            result = ' '.join(map(lambda arg: self._print(arg), [var.type, var.symbol]))

        return result
```
### 71 - sympy/printing/julia.py:

Start line: 46, End line: 116

```python
class JuliaCodePrinter(CodePrinter):
    """
    A printer to convert expressions to strings of Julia code.
    """
    printmethod = "_julia"
    language = "Julia"

    _operators = {
        'and': '&&',
        'or': '||',
        'not': '!',
    }

    _default_settings = {
        'order': None,
        'full_prec': 'auto',
        'precision': 17,
        'user_functions': {},
        'human': True,
        'allow_unknown_functions': True,
        'contract': True,
        'inline': True,
    }
    # Note: contract is for expressing tensors as loops (if True), or just
    # assignment (if False).  FIXME: this should be looked a more carefully
    # for Julia.

    def __init__(self, settings={}):
        super(JuliaCodePrinter, self).__init__(settings)
        self.known_functions = dict(zip(known_fcns_src1, known_fcns_src1))
        self.known_functions.update(dict(known_fcns_src2))
        userfuncs = settings.get('user_functions', {})
        self.known_functions.update(userfuncs)


    def _rate_index_position(self, p):
        return p*5


    def _get_statement(self, codestring):
        return "%s" % codestring


    def _get_comment(self, text):
        return "# {0}".format(text)


    def _declare_number_const(self, name, value):
        return "const {0} = {1}".format(name, value)


    def _format_code(self, lines):
        return self.indent_code(lines)


    def _traverse_matrix_indices(self, mat):
        # Julia uses Fortran order (column-major)
        rows, cols = mat.shape
        return ((i, j) for j in range(cols) for i in range(rows))


    def _get_loop_opening_ending(self, indices):
        open_lines = []
        close_lines = []
        for i in indices:
            # Julia arrays start at 1 and end at dimension
            var, start, stop = map(self._print,
                    [i.label, i.lower + 1, i.upper + 1])
            open_lines.append("for %s = %s:%s" % (var, start, stop))
            close_lines.append("end")
        return open_lines, close_lines
```
### 73 - sympy/printing/fcode.py:

Start line: 254, End line: 295

```python
class FCodePrinter(CodePrinter):

    def _print_MatrixElement(self, expr):
        return "{0}({1}, {2})".format(self.parenthesize(expr.parent,
                PRECEDENCE["Atom"], strict=True), expr.i + 1, expr.j + 1)

    def _print_Add(self, expr):
        # purpose: print complex numbers nicely in Fortran.
        # collect the purely real and purely imaginary parts:
        pure_real = []
        pure_imaginary = []
        mixed = []
        for arg in expr.args:
            if arg.is_number and arg.is_real:
                pure_real.append(arg)
            elif arg.is_number and arg.is_imaginary:
                pure_imaginary.append(arg)
            else:
                mixed.append(arg)
        if len(pure_imaginary) > 0:
            if len(mixed) > 0:
                PREC = precedence(expr)
                term = Add(*mixed)
                t = self._print(term)
                if t.startswith('-'):
                    sign = "-"
                    t = t[1:]
                else:
                    sign = "+"
                if precedence(term) < PREC:
                    t = "(%s)" % t

                return "cmplx(%s,%s) %s %s" % (
                    self._print(Add(*pure_real)),
                    self._print(-S.ImaginaryUnit*Add(*pure_imaginary)),
                    sign, t,
                )
            else:
                return "cmplx(%s,%s)" % (
                    self._print(Add(*pure_real)),
                    self._print(-S.ImaginaryUnit*Add(*pure_imaginary)),
                )
        else:
            return CodePrinter._print_Add(self, expr)
```
### 78 - sympy/printing/julia.py:

Start line: 372, End line: 390

```python
class JuliaCodePrinter(CodePrinter):


    def _print_MatrixSlice(self, expr):
        def strslice(x, lim):
            l = x[0] + 1
            h = x[1]
            step = x[2]
            lstr = self._print(l)
            hstr = 'end' if h == lim else self._print(h)
            if step == 1:
                if l == 1 and h == lim:
                    return ':'
                if l == h:
                    return lstr
                else:
                    return lstr + ':' + hstr
            else:
                return ':'.join((lstr, self._print(step), hstr))
        return (self._print(expr.parent) + '[' +
                strslice(expr.rowslice, expr.parent.shape[0]) + ',' +
                strslice(expr.colslice, expr.parent.shape[1]) + ']')
```
### 80 - sympy/printing/fcode.py:

Start line: 395, End line: 411

```python
class FCodePrinter(CodePrinter):

    def _print_ImpliedDoLoop(self, idl):
        step = '' if idl.step == 1 else ', {step}'
        return ('({expr}, {counter} = {first}, {last}'+step+')').format(
            **idl.kwargs(apply=lambda arg: self._print(arg))
        )

    def _print_For(self, expr):
        target = self._print(expr.target)
        if isinstance(expr.iterable, Range):
            start, stop, step = expr.iterable.args
        else:
            raise NotImplementedError("Only iterable currently supported is Range")
        body = self._print(expr.body)
        return ('do {target} = {start}, {stop}, {step}\n'
                '{body}\n'
                'end do').format(target=target, start=start, stop=stop,
                        step=step, body=body)
```
### 81 - sympy/printing/fcode.py:

Start line: 378, End line: 393

```python
class FCodePrinter(CodePrinter):

    def _print_Do(self, do):
        excl = ['concurrent']
        if do.step == 1:
            excl.append('step')
            step = ''
        else:
            step = ', {step}'

        return (
            'do {concurrent}{counter} = {first}, {last}'+step+'\n'
            '{body}\n'
            'end do\n'
        ).format(
            concurrent='concurrent ' if do.concurrent else '',
            **do.kwargs(apply=lambda arg: self._print(arg), exclude=excl)
        )
```
### 82 - sympy/printing/octave.py:

Start line: 283, End line: 314

```python
class OctaveCodePrinter(CodePrinter):


    def _print_Infinity(self, expr):
        return 'inf'


    def _print_NegativeInfinity(self, expr):
        return '-inf'


    def _print_NaN(self, expr):
        return 'NaN'


    def _print_list(self, expr):
        return '{' + ', '.join(self._print(a) for a in expr) + '}'
    _print_tuple = _print_list
    _print_Tuple = _print_list


    def _print_BooleanTrue(self, expr):
        return "true"


    def _print_BooleanFalse(self, expr):
        return "false"


    def _print_bool(self, expr):
        return str(expr).lower()


    # Could generate quadrature code for definite Integrals?
    #_print_Integral = _print_not_supported
```
### 83 - sympy/printing/julia.py:

Start line: 421, End line: 455

```python
class JuliaCodePrinter(CodePrinter):


    def _print_Piecewise(self, expr):
        if expr.args[-1].cond != True:
            # We need the last conditional to be a True, otherwise the resulting
            # function may not return a result.
            raise ValueError("All Piecewise expressions must contain an "
                             "(expr, True) statement to be used as a default "
                             "condition. Without one, the generated "
                             "expression may not evaluate to anything under "
                             "some condition.")
        lines = []
        if self._settings["inline"]:
            # Express each (cond, expr) pair in a nested Horner form:
            #   (condition) .* (expr) + (not cond) .* (<others>)
            # Expressions that result in multiple statements won't work here.
            ecpairs = ["({0}) ? ({1}) :".format
                       (self._print(c), self._print(e))
                       for e, c in expr.args[:-1]]
            elast = " (%s)" % self._print(expr.args[-1].expr)
            pw = "\n".join(ecpairs) + elast
            # Note: current need these outer brackets for 2*pw.  Would be
            # nicer to teach parenthesize() to do this for us when needed!
            return "(" + pw + ")"
        else:
            for i, (e, c) in enumerate(expr.args):
                if i == 0:
                    lines.append("if (%s)" % self._print(c))
                elif i == len(expr.args) - 1 and c == True:
                    lines.append("else")
                else:
                    lines.append("elseif (%s)" % self._print(c))
                code0 = self._print(e)
                lines.append(code0)
                if i == len(expr.args) - 1:
                    lines.append("end")
            return "\n".join(lines)
```
### 88 - sympy/printing/ccode.py:

Start line: 92, End line: 100

```python
reserved_words = [
    'auto', 'break', 'case', 'char', 'const', 'continue', 'default', 'do',
    'double', 'else', 'enum', 'extern', 'float', 'for', 'goto', 'if', 'int',
    'long', 'register', 'return', 'short', 'signed', 'sizeof', 'static',
    'struct', 'entry',  # never standardized, we'll leave it here anyway
    'switch', 'typedef', 'union', 'unsigned', 'void', 'volatile', 'while'
]

reserved_words_c99 = ['inline', 'restrict']
```
### 90 - sympy/printing/fcode.py:

Start line: 413, End line: 437

```python
class FCodePrinter(CodePrinter):

    def _print_Equality(self, expr):
        lhs, rhs = expr.args
        return ' == '.join(map(lambda arg: self._print(arg), (lhs, rhs)))

    def _print_Unequality(self, expr):
        lhs, rhs = expr.args
        return ' /= '.join(map(lambda arg: self._print(arg), (lhs, rhs)))

    def _print_Type(self, type_):
        type_ = self.type_aliases.get(type_, type_)
        type_str = self.type_mappings.get(type_, type_.name)
        module_uses = self.type_modules.get(type_)
        if module_uses:
            for k, v in module_uses:
                self.module_uses[k].add(v)
        return type_str

    def _print_Element(self, elem):
        return '{symbol}({idxs})'.format(
            symbol=self._print(elem.symbol),
            idxs=', '.join(map(lambda arg: self._print(arg), elem.indices))
        )

    def _print_Extent(self, ext):
        return str(ext)
```
### 93 - sympy/printing/fcode.py:

Start line: 297, End line: 321

```python
class FCodePrinter(CodePrinter):

    def _print_Function(self, expr):
        # All constant function args are evaluated as floats
        prec =  self._settings['precision']
        args = [N(a, prec) for a in expr.args]
        eval_expr = expr.func(*args)
        if not isinstance(eval_expr, Function):
            return self._print(eval_expr)
        else:
            return CodePrinter._print_Function(self, expr.func(*args))

    def _print_ImaginaryUnit(self, expr):
        # purpose: print complex numbers nicely in Fortran.
        return "cmplx(0,1)"

    def _print_int(self, expr):
        return str(expr)

    def _print_Mul(self, expr):
        # purpose: print complex numbers nicely in Fortran.
        if expr.is_number and expr.is_imaginary:
            return "cmplx(0,%s)" % (
                self._print(-S.ImaginaryUnit*expr)
            )
        else:
            return CodePrinter._print_Mul(self, expr)
```
### 94 - sympy/printing/fcode.py:

Start line: 189, End line: 198

```python
class FCodePrinter(CodePrinter):

    def _get_loop_opening_ending(self, indices):
        open_lines = []
        close_lines = []
        for i in indices:
            # fortran arrays start at 1 and end at dimension
            var, start, stop = map(self._print,
                    [i.label, i.lower + 1, i.upper + 1])
            open_lines.append("do %s = %s, %s" % (var, start, stop))
            close_lines.append("end do")
        return open_lines, close_lines
```
### 95 - sympy/printing/julia.py:

Start line: 491, End line: 633

```python
def julia_code(expr, assign_to=None, **settings):
    r"""Converts `expr` to a string of Julia code.

    Parameters
    ==========

    expr : Expr
        A sympy expression to be converted.
    assign_to : optional
        When given, the argument is used as the name of the variable to which
        the expression is assigned.  Can be a string, ``Symbol``,
        ``MatrixSymbol``, or ``Indexed`` type.  This can be helpful for
        expressions that generate multi-line statements.
    precision : integer, optional
        The precision for numbers such as pi  [default=16].
    user_functions : dict, optional
        A dictionary where keys are ``FunctionClass`` instances and values are
        their string representations.  Alternatively, the dictionary value can
        be a list of tuples i.e. [(argument_test, cfunction_string)].  See
        below for examples.
    human : bool, optional
        If True, the result is a single string that may contain some constant
        declarations for the number symbols.  If False, the same information is
        returned in a tuple of (symbols_to_declare, not_supported_functions,
        code_text).  [default=True].
    contract: bool, optional
        If True, ``Indexed`` instances are assumed to obey tensor contraction
        rules and the corresponding nested loops over indices are generated.
        Setting contract=False will not generate loops, instead the user is
        responsible to provide values for the indices in the code.
        [default=True].
    inline: bool, optional
        If True, we try to create single-statement code instead of multiple
        statements.  [default=True].

    Examples
    ========

    >>> from sympy import julia_code, symbols, sin, pi
    >>> x = symbols('x')
    >>> julia_code(sin(x).series(x).removeO())
    'x.^5/120 - x.^3/6 + x'

    >>> from sympy import Rational, ceiling, Abs
    >>> x, y, tau = symbols("x, y, tau")
    >>> julia_code((2*tau)**Rational(7, 2))
    '8*sqrt(2)*tau.^(7/2)'

    Note that element-wise (Hadamard) operations are used by default between
    symbols.  This is because its possible in Julia to write "vectorized"
    code.  It is harmless if the values are scalars.

    >>> julia_code(sin(pi*x*y), assign_to="s")
    's = sin(pi*x.*y)'

    If you need a matrix product "*" or matrix power "^", you can specify the
    symbol as a ``MatrixSymbol``.

    >>> from sympy import Symbol, MatrixSymbol
    >>> n = Symbol('n', integer=True, positive=True)
    >>> A = MatrixSymbol('A', n, n)
    >>> julia_code(3*pi*A**3)
    '(3*pi)*A^3'

    This class uses several rules to decide which symbol to use a product.
    Pure numbers use "*", Symbols use ".*" and MatrixSymbols use "*".
    A HadamardProduct can be used to specify componentwise multiplication ".*"
    of two MatrixSymbols.  There is currently there is no easy way to specify
    scalar symbols, so sometimes the code might have some minor cosmetic
    issues.  For example, suppose x and y are scalars and A is a Matrix, then
    while a human programmer might write "(x^2*y)*A^3", we generate:

    >>> julia_code(x**2*y*A**3)
    '(x.^2.*y)*A^3'

    Matrices are supported using Julia inline notation.  When using
    ``assign_to`` with matrices, the name can be specified either as a string
    or as a ``MatrixSymbol``.  The dimensions must align in the latter case.

    >>> from sympy import Matrix, MatrixSymbol
    >>> mat = Matrix([[x**2, sin(x), ceiling(x)]])
    >>> julia_code(mat, assign_to='A')
    'A = [x.^2 sin(x) ceil(x)]'

    ``Piecewise`` expressions are implemented with logical masking by default.
    Alternatively, you can pass "inline=False" to use if-else conditionals.
    Note that if the ``Piecewise`` lacks a default term, represented by
    ``(expr, True)`` then an error will be thrown.  This is to prevent
    generating an expression that may not evaluate to anything.

    >>> from sympy import Piecewise
    >>> pw = Piecewise((x + 1, x > 0), (x, True))
    >>> julia_code(pw, assign_to=tau)
    'tau = ((x > 0) ? (x + 1) : (x))'

    Note that any expression that can be generated normally can also exist
    inside a Matrix:

    >>> mat = Matrix([[x**2, pw, sin(x)]])
    >>> julia_code(mat, assign_to='A')
    'A = [x.^2 ((x > 0) ? (x + 1) : (x)) sin(x)]'

    Custom printing can be defined for certain types by passing a dictionary of
    "type" : "function" to the ``user_functions`` kwarg.  Alternatively, the
    dictionary value can be a list of tuples i.e., [(argument_test,
    cfunction_string)].  This can be used to call a custom Julia function.

    >>> from sympy import Function
    >>> f = Function('f')
    >>> g = Function('g')
    >>> custom_functions = {
    ...   "f": "existing_julia_fcn",
    ...   "g": [(lambda x: x.is_Matrix, "my_mat_fcn"),
    ...         (lambda x: not x.is_Matrix, "my_fcn")]
    ... }
    >>> mat = Matrix([[1, x]])
    >>> julia_code(f(x) + g(x) + g(mat), user_functions=custom_functions)
    'existing_julia_fcn(x) + my_fcn(x) + my_mat_fcn([1 x])'

    Support for loops is provided through ``Indexed`` types. With
    ``contract=True`` these expressions will be turned into loops, whereas
    ``contract=False`` will just print the assignment expression that should be
    looped over:

    >>> from sympy import Eq, IndexedBase, Idx, ccode
    >>> len_y = 5
    >>> y = IndexedBase('y', shape=(len_y,))
    >>> t = IndexedBase('t', shape=(len_y,))
    >>> Dy = IndexedBase('Dy', shape=(len_y-1,))
    >>> i = Idx('i', len_y-1)
    >>> e = Eq(Dy[i], (y[i+1]-y[i])/(t[i+1]-t[i]))
    >>> julia_code(e.rhs, assign_to=e.lhs, contract=False)
    'Dy[i] = (y[i + 1] - y[i])./(t[i + 1] - t[i])'
    """
    return JuliaCodePrinter(settings).doprint(expr, assign_to)


def print_julia_code(expr, **settings):
    """Prints the Julia representation of the given expression.

    See `julia_code` for the meaning of the optional arguments.
    """
    print(julia_code(expr, **settings))
```
### 96 - sympy/printing/ccode.py:

Start line: 1, End line: 90

```python
"""
C code printer

The C89CodePrinter & C99CodePrinter converts single sympy expressions into
single C expressions, using the functions defined in math.h where possible.

A complete code generator, which uses ccode extensively, can be found in
sympy.utilities.codegen. The codegen module can be used to generate complete
source code files that are compilable without further modifications.


"""

from __future__ import print_function, division

from functools import wraps
from itertools import chain

from sympy.core import S
from sympy.core.compatibility import string_types, range
from sympy.core.decorators import deprecated
from sympy.codegen.ast import (
    Assignment, Pointer, Type, Variable, Declaration,
    real, complex_, integer, bool_, float32, float64, float80,
    complex64, complex128, intc, value_const, pointer_const,
    int8, int16, int32, int64, uint8, uint16, uint32, uint64, untyped
)
from sympy.printing.codeprinter import CodePrinter, requires
from sympy.printing.precedence import precedence, PRECEDENCE
from sympy.sets.fancysets import Range

# dictionary mapping sympy function to (argument_conditions, C_function).
# Used in C89CodePrinter._print_Function(self)
known_functions_C89 = {
    "Abs": [(lambda x: not x.is_integer, "fabs"), (lambda x: x.is_integer, "abs")],
    "Mod": [
        (
            lambda numer, denom: numer.is_integer and denom.is_integer,
            lambda printer, numer, denom, *args: "((%s) %% (%s))" % (
                printer._print(numer), printer._print(denom))
        ),
        (
            lambda numer, denom: not numer.is_integer or not denom.is_integer,
            "fmod"
        )
    ],
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "asin": "asin",
    "acos": "acos",
    "atan": "atan",
    "atan2": "atan2",
    "exp": "exp",
    "log": "log",
    "sinh": "sinh",
    "cosh": "cosh",
    "tanh": "tanh",
    "floor": "floor",
    "ceiling": "ceil",
}

# move to C99 once CCodePrinter is removed:
_known_functions_C9X = dict(known_functions_C89, **{
    "asinh": "asinh",
    "acosh": "acosh",
    "atanh": "atanh",
    "erf": "erf",
    "gamma": "tgamma",
})
known_functions = _known_functions_C9X

known_functions_C99 = dict(_known_functions_C9X, **{
    'exp2': 'exp2',
    'expm1': 'expm1',
    'expm1': 'expm1',
    'log10': 'log10',
    'log2': 'log2',
    'log1p': 'log1p',
    'Cbrt': 'cbrt',
    'hypot': 'hypot',
    'fma': 'fma',
    'loggamma': 'lgamma',
    'erfc': 'erfc',
    'Max': 'fmax',
    'Min': 'fmin'
})

# These are the core reserved words in the C language. Taken from:
# http://en.cppreference.com/w/c/keyword
```
### 101 - sympy/printing/codeprinter.py:

Start line: 124, End line: 201

```python
class CodePrinter(StrPrinter):

    def _doprint_loops(self, expr, assign_to=None):
        # Here we print an expression that contains Indexed objects, they
        # correspond to arrays in the generated code.  The low-level implementation
        # involves looping over array elements and possibly storing results in temporary
        # variables or accumulate it in the assign_to object.

        if self._settings.get('contract', True):
            from sympy.tensor import get_contraction_structure
            # Setup loops over non-dummy indices  --  all terms need these
            indices = self._get_expression_indices(expr, assign_to)
            # Setup loops over dummy indices  --  each term needs separate treatment
            dummies = get_contraction_structure(expr)
        else:
            indices = []
            dummies = {None: (expr,)}
        openloop, closeloop = self._get_loop_opening_ending(indices)

        # terms with no summations first
        if None in dummies:
            text = StrPrinter.doprint(self, Add(*dummies[None]))
        else:
            # If all terms have summations we must initialize array to Zero
            text = StrPrinter.doprint(self, 0)

        # skip redundant assignments (where lhs == rhs)
        lhs_printed = self._print(assign_to)
        lines = []
        if text != lhs_printed:
            lines.extend(openloop)
            if assign_to is not None:
                text = self._get_statement("%s = %s" % (lhs_printed, text))
            lines.append(text)
            lines.extend(closeloop)

        # then terms with summations
        for d in dummies:
            if isinstance(d, tuple):
                indices = self._sort_optimized(d, expr)
                openloop_d, closeloop_d = self._get_loop_opening_ending(
                    indices)

                for term in dummies[d]:
                    if term in dummies and not ([list(f.keys()) for f in dummies[term]]
                            == [[None] for f in dummies[term]]):
                        # If one factor in the term has it's own internal
                        # contractions, those must be computed first.
                        # (temporary variables?)
                        raise NotImplementedError(
                            "FIXME: no support for contractions in factor yet")
                    else:

                        # We need the lhs expression as an accumulator for
                        # the loops, i.e
                        #
                        # for (int d=0; d < dim; d++){
                        #    lhs[] = lhs[] + term[][d]
                        # }           ^.................. the accumulator
                        #
                        # We check if the expression already contains the
                        # lhs, and raise an exception if it does, as that
                        # syntax is currently undefined.  FIXME: What would be
                        # a good interpretation?
                        if assign_to is None:
                            raise AssignmentError(
                                "need assignment variable for loops")
                        if term.has(assign_to):
                            raise ValueError("FIXME: lhs present in rhs,\
                                this is undefined in CodePrinter")

                        lines.extend(openloop)
                        lines.extend(openloop_d)
                        text = "%s = %s" % (lhs_printed, StrPrinter.doprint(
                            self, assign_to + term))
                        lines.append(self._get_statement(text))
                        lines.extend(closeloop_d)
                        lines.extend(closeloop)

        return "\n".join(lines)
```
### 102 - sympy/printing/codeprinter.py:

Start line: 390, End line: 441

```python
class CodePrinter(StrPrinter):

    _print_Expr = _print_Function

    def _print_NumberSymbol(self, expr):
        if self._settings.get("inline", False):
            return self._print(Float(expr.evalf(self._settings["precision"])))
        else:
            # A Number symbol that is not implemented here or with _printmethod
            # is registered and evaluated
            self._number_symbols.add((expr,
                Float(expr.evalf(self._settings["precision"]))))
            return str(expr)

    def _print_Catalan(self, expr):
        return self._print_NumberSymbol(expr)
    def _print_EulerGamma(self, expr):
        return self._print_NumberSymbol(expr)
    def _print_GoldenRatio(self, expr):
        return self._print_NumberSymbol(expr)
    def _print_TribonacciConstant(self, expr):
        return self._print_NumberSymbol(expr)
    def _print_Exp1(self, expr):
        return self._print_NumberSymbol(expr)
    def _print_Pi(self, expr):
        return self._print_NumberSymbol(expr)

    def _print_And(self, expr):
        PREC = precedence(expr)
        return (" %s " % self._operators['and']).join(self.parenthesize(a, PREC)
                for a in sorted(expr.args, key=default_sort_key))

    def _print_Or(self, expr):
        PREC = precedence(expr)
        return (" %s " % self._operators['or']).join(self.parenthesize(a, PREC)
                for a in sorted(expr.args, key=default_sort_key))

    def _print_Xor(self, expr):
        if self._operators.get('xor') is None:
            return self._print_not_supported(expr)
        PREC = precedence(expr)
        return (" %s " % self._operators['xor']).join(self.parenthesize(a, PREC)
                for a in expr.args)

    def _print_Equivalent(self, expr):
        if self._operators.get('equivalent') is None:
            return self._print_not_supported(expr)
        PREC = precedence(expr)
        return (" %s " % self._operators['equivalent']).join(self.parenthesize(a, PREC)
                for a in expr.args)

    def _print_Not(self, expr):
        PREC = precedence(expr)
        return self._operators['not'] + self.parenthesize(expr.args[0], PREC)
```
### 106 - sympy/printing/fcode.py:

Start line: 164, End line: 187

```python
class FCodePrinter(CodePrinter):

    def _rate_index_position(self, p):
        return -p*5

    def _get_statement(self, codestring):
        return codestring

    def _get_comment(self, text):
        return "! {0}".format(text)

    def _declare_number_const(self, name, value):
        return "parameter ({0} = {1})".format(name, self._print(value))

    def _print_NumberSymbol(self, expr):
        # A Number symbol that is not implemented here or with _printmethod
        # is registered and evaluated
        self._number_symbols.add((expr, Float(expr.evalf(self._settings['precision']))))
        return str(expr)

    def _format_code(self, lines):
        return self._wrap_fortran(self.indent_code(lines))

    def _traverse_matrix_indices(self, mat):
        rows, cols = mat.shape
        return ((i, j) for j in range(cols) for i in range(rows))
```
### 108 - sympy/printing/fcode.py:

Start line: 654, End line: 674

```python
class FCodePrinter(CodePrinter):

    def _print_Print(self, ps):
        if ps.format_string != None:
            fmt = self._print(ps.format_string)
        else:
            fmt = "*"
        return "print {fmt}, {iolist}".format(fmt=fmt, iolist=', '.join(
            map(lambda arg: self._print(arg), ps.print_args)))

    def _print_Return(self, rs):
        arg, = rs.args
        return "{result_name} = {arg}".format(
            result_name=self._context.get('result_name', 'sympy_result'),
            arg=self._print(arg)
        )

    def _print_FortranReturn(self, frs):
        arg, = frs.args
        if arg:
            return 'return %s' % self._print(arg)
        else:
            return 'return'
```
### 109 - sympy/printing/ccode.py:

Start line: 522, End line: 545

```python
class C89CodePrinter(CodePrinter):

    @requires(headers={'stdbool.h'})
    def _print_BooleanTrue(self, expr):
        return 'true'

    @requires(headers={'stdbool.h'})
    def _print_BooleanFalse(self, expr):
        return 'false'

    def _print_Element(self, elem):
        if elem.strides == None:
            if elem.offset != None:
                raise ValueError("Expected strides when offset is given")
            idxs = ']['.join(map(lambda arg: self._print(arg),
                                 elem.indices))
        else:
            global_idx = sum([i*s for i, s in zip(elem.indices, elem.strides)])
            if elem.offset != None:
                global_idx += elem.offset
            idxs = self._print(global_idx)

        return "{symb}[{idxs}]".format(
            symb=self._print(elem.symbol),
            idxs=idxs
        )
```
### 113 - sympy/printing/codeprinter.py:

Start line: 203, End line: 238

```python
class CodePrinter(StrPrinter):

    def _get_expression_indices(self, expr, assign_to):
        from sympy.tensor import get_indices
        rinds, junk = get_indices(expr)
        linds, junk = get_indices(assign_to)

        # support broadcast of scalar
        if linds and not rinds:
            rinds = linds
        if rinds != linds:
            raise ValueError("lhs indices must match non-dummy"
                    " rhs indices in %s" % expr)

        return self._sort_optimized(rinds, assign_to)

    def _sort_optimized(self, indices, expr):

        from sympy.tensor.indexed import Indexed

        if not indices:
            return []

        # determine optimized loop order by giving a score to each index
        # the index with the highest score are put in the innermost loop.
        score_table = {}
        for i in indices:
            score_table[i] = 0

        arrays = expr.atoms(Indexed)
        for arr in arrays:
            for p, ind in enumerate(arr.indices):
                try:
                    score_table[ind] += self._rate_index_position(p)
                except KeyError:
                    pass

        return sorted(indices, key=lambda x: score_table[x])
```
### 114 - sympy/printing/ccode.py:

Start line: 652, End line: 681

```python
class C99CodePrinter(_C9XCodePrinter, C89CodePrinter):
    standard = 'C99'
    reserved_words = set(reserved_words + reserved_words_c99)
    type_mappings=dict(chain(C89CodePrinter.type_mappings.items(), {
        complex64: 'float complex',
        complex128: 'double complex',
    }.items()))
    type_headers = dict(chain(C89CodePrinter.type_headers.items(), {
        complex64: {'complex.h'},
        complex128: {'complex.h'}
    }.items()))
    _kf = known_functions_C99  # known_functions-dict to copy

    # functions with versions with 'f' and 'l' suffixes:
    _prec_funcs = ('fabs fmod remainder remquo fma fmax fmin fdim nan exp exp2'
                   ' expm1 log log10 log2 log1p pow sqrt cbrt hypot sin cos tan'
                   ' asin acos atan atan2 sinh cosh tanh asinh acosh atanh erf'
                   ' erfc tgamma lgamma ceil floor trunc round nearbyint rint'
                   ' frexp ldexp modf scalbn ilogb logb nextafter copysign').split()

    def _print_Infinity(self, expr):
        return 'INFINITY'

    def _print_NegativeInfinity(self, expr):
        return '-INFINITY'

    def _print_NaN(self, expr):
        return 'NAN'

    # tgamma was already covered by 'known_functions' dict
```
### 116 - sympy/printing/fcode.py:

Start line: 615, End line: 639

```python
class FCodePrinter(CodePrinter):

    def _print_GoTo(self, goto):
        if goto.expr:  # computed goto
            return "go to ({labels}), {expr}".format(
                labels=', '.join(map(lambda arg: self._print(arg), goto.labels)),
                expr=self._print(goto.expr)
            )
        else:
            lbl, = goto.labels
            return "go to %s" % self._print(lbl)

    def _print_Program(self, prog):
        return (
            "program {name}\n"
            "{body}\n"
            "end program\n"
        ).format(**prog.kwargs(apply=lambda arg: self._print(arg)))

    def _print_Module(self, mod):
        return (
            "module {name}\n"
            "{declarations}\n"
            "\ncontains\n\n"
            "{definitions}\n"
            "end module\n"
        ).format(**mod.kwargs(apply=lambda arg: self._print(arg)))
```
### 119 - sympy/printing/julia.py:

Start line: 287, End line: 324

```python
class JuliaCodePrinter(CodePrinter):


    def _print_Infinity(self, expr):
        return 'Inf'


    def _print_NegativeInfinity(self, expr):
        return '-Inf'


    def _print_NaN(self, expr):
        return 'NaN'


    def _print_list(self, expr):
        return 'Any[' + ', '.join(self._print(a) for a in expr) + ']'


    def _print_tuple(self, expr):
        if len(expr) == 1:
            return "(%s,)" % self._print(expr[0])
        else:
            return "(%s)" % self.stringify(expr, ", ")
    _print_Tuple = _print_tuple


    def _print_BooleanTrue(self, expr):
        return "true"


    def _print_BooleanFalse(self, expr):
        return "false"


    def _print_bool(self, expr):
        return str(expr).lower()


    # Could generate quadrature code for definite Integrals?
    #_print_Integral = _print_not_supported
```
### 120 - sympy/printing/ccode.py:

Start line: 547, End line: 618

```python
class C89CodePrinter(CodePrinter):

    def _print_CodeBlock(self, expr):
        """ Elements of code blocks printed as statements. """
        return '\n'.join([self._get_statement(self._print(i)) for i in expr.args])

    def _print_While(self, expr):
        return 'while ({condition}) {{\n{body}\n}}'.format(**expr.kwargs(
            apply=lambda arg: self._print(arg)))

    def _print_Scope(self, expr):
        return '{\n%s\n}' % self._print_CodeBlock(expr.body)

    @requires(headers={'stdio.h'})
    def _print_Print(self, expr):
        return 'printf({fmt}, {pargs})'.format(
            fmt=self._print(expr.format_string),
            pargs=', '.join(map(lambda arg: self._print(arg), expr.print_args))
        )

    def _print_FunctionPrototype(self, expr):
        pars = ', '.join(map(lambda arg: self._print(Declaration(arg)),
                             expr.parameters))
        return "%s %s(%s)" % (
            tuple(map(lambda arg: self._print(arg),
                      (expr.return_type, expr.name))) + (pars,)
        )

    def _print_FunctionDefinition(self, expr):
        return "%s%s" % (self._print_FunctionPrototype(expr),
                         self._print_Scope(expr))

    def _print_Return(self, expr):
        arg, = expr.args
        return 'return %s' % self._print(arg)

    def _print_CommaOperator(self, expr):
        return '(%s)' % ', '.join(map(lambda arg: self._print(arg), expr.args))

    def _print_Label(self, expr):
        return '%s:' % str(expr)

    def _print_goto(self, expr):
        return 'goto %s' % expr.label

    def _print_PreIncrement(self, expr):
        arg, = expr.args
        return '++(%s)' % self._print(arg)

    def _print_PostIncrement(self, expr):
        arg, = expr.args
        return '(%s)++' % self._print(arg)

    def _print_PreDecrement(self, expr):
        arg, = expr.args
        return '--(%s)' % self._print(arg)

    def _print_PostDecrement(self, expr):
        arg, = expr.args
        return '(%s)--' % self._print(arg)

    def _print_struct(self, expr):
        return "%(keyword)s %(name)s {\n%(lines)s}" % dict(
            keyword=expr.__class__.__name__, name=expr.name, lines=';\n'.join(
                [self._print(decl) for decl in expr.declarations] + [''])
        )

    def _print_BreakToken(self, _):
        return 'break'

    def _print_ContinueToken(self, _):
        return 'continue'

    _print_union = _print_struct
```
### 123 - sympy/printing/julia.py:

Start line: 327, End line: 339

```python
class JuliaCodePrinter(CodePrinter):


    def _print_MatrixBase(self, A):
        # Handle zero dimensions:
        if A.rows == 0 or A.cols == 0:
            return 'zeros(%s, %s)' % (A.rows, A.cols)
        elif (A.rows, A.cols) == (1, 1):
            return "[%s]" % A[0, 0]
        elif A.rows == 1:
            return "[%s]" % A.table(self, rowstart='', rowend='', colsep=' ')
        elif A.cols == 1:
            # note .table would unnecessarily equispace the rows
            return "[%s]" % ", ".join([self._print(a) for a in A])
        return "[%s]" % A.table(self, rowstart='', rowend='',
                                rowsep=';\n', colsep=' ')
```
