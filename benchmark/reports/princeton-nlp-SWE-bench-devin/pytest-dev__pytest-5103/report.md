# pytest-dev__pytest-5103

| **pytest-dev/pytest** | `10ca84ffc56c2dd2d9dc4bd71b7b898e083500cd` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 70317 |
| **Any found context length** | 62268 |
| **Avg pos** | 425.0 |
| **Min pos** | 132 |
| **Max pos** | 161 |
| **Top file pos** | 7 |
| **Missing snippets** | 3 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/src/_pytest/assertion/rewrite.py b/src/_pytest/assertion/rewrite.py
--- a/src/_pytest/assertion/rewrite.py
+++ b/src/_pytest/assertion/rewrite.py
@@ -964,6 +964,8 @@ def visit_Call_35(self, call):
         """
         visit `ast.Call` nodes on Python3.5 and after
         """
+        if isinstance(call.func, ast.Name) and call.func.id == "all":
+            return self._visit_all(call)
         new_func, func_expl = self.visit(call.func)
         arg_expls = []
         new_args = []
@@ -987,6 +989,27 @@ def visit_Call_35(self, call):
         outer_expl = "%s\n{%s = %s\n}" % (res_expl, res_expl, expl)
         return res, outer_expl
 
+    def _visit_all(self, call):
+        """Special rewrite for the builtin all function, see #5062"""
+        if not isinstance(call.args[0], (ast.GeneratorExp, ast.ListComp)):
+            return
+        gen_exp = call.args[0]
+        assertion_module = ast.Module(
+            body=[ast.Assert(test=gen_exp.elt, lineno=1, msg="", col_offset=1)]
+        )
+        AssertionRewriter(module_path=None, config=None).run(assertion_module)
+        for_loop = ast.For(
+            iter=gen_exp.generators[0].iter,
+            target=gen_exp.generators[0].target,
+            body=assertion_module.body,
+            orelse=[],
+        )
+        self.statements.append(for_loop)
+        return (
+            ast.Num(n=1),
+            "",
+        )  # Return an empty expression, all the asserts are in the for_loop
+
     def visit_Starred(self, starred):
         # From Python 3.5, a Starred node can appear in a function call
         res, expl = self.visit(starred.value)
@@ -997,6 +1020,8 @@ def visit_Call_legacy(self, call):
         """
         visit `ast.Call nodes on 3.4 and below`
         """
+        if isinstance(call.func, ast.Name) and call.func.id == "all":
+            return self._visit_all(call)
         new_func, func_expl = self.visit(call.func)
         arg_expls = []
         new_args = []

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| src/_pytest/assertion/rewrite.py | 967 | 967 | 132 | 7 | 62268
| src/_pytest/assertion/rewrite.py | 990 | 990 | 132 | 7 | 62268
| src/_pytest/assertion/rewrite.py | 1000 | 1000 | 161 | 7 | 70317


## Problem Statement

```
Unroll the iterable for all/any calls to get better reports
Sometime I need to assert some predicate on all of an iterable, and for that the builtin functions `all`/`any` are great - but the failure messages aren't useful at all!
For example - the same test written in three ways:

- A generator expression
\`\`\`sh                                                                                                                                                                                                                         
    def test_all_even():
        even_stevens = list(range(1,100,2))
>       assert all(is_even(number) for number in even_stevens)
E       assert False
E        +  where False = all(<generator object test_all_even.<locals>.<genexpr> at 0x101f82ed0>)
\`\`\`
- A list comprehension
\`\`\`sh
    def test_all_even():
        even_stevens = list(range(1,100,2))
>       assert all([is_even(number) for number in even_stevens])
E       assert False
E        +  where False = all([False, False, False, False, False, False, ...])
\`\`\`
- A for loop
\`\`\`sh
    def test_all_even():
        even_stevens = list(range(1,100,2))
        for number in even_stevens:
>           assert is_even(number)
E           assert False
E            +  where False = is_even(1)

test_all_any.py:7: AssertionError
\`\`\`
The only one that gives a meaningful report is the for loop - but it's way more wordy, and `all` asserts don't translate to a for loop nicely (I'll have to write a `break` or a helper function - yuck)
I propose the assertion re-writer "unrolls" the iterator to the third form, and then uses the already existing reports.

- [x] Include a detailed description of the bug or suggestion
- [x] `pip list` of the virtual environment you are using
\`\`\`
Package        Version
-------------- -------
atomicwrites   1.3.0  
attrs          19.1.0 
more-itertools 7.0.0  
pip            19.0.3 
pluggy         0.9.0  
py             1.8.0  
pytest         4.4.0  
setuptools     40.8.0 
six            1.12.0 
\`\`\`
- [x] pytest and operating system versions
`platform darwin -- Python 3.7.3, pytest-4.4.0, py-1.8.0, pluggy-0.9.0`
- [x] Minimal example if possible


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 src/pytest.py | 1 | 107| 723 | 723 | 723 | 
| 2 | 2 src/_pytest/runner.py | 156 | 198| 338 | 1061 | 3494 | 
| 3 | 3 src/_pytest/assertion/__init__.py | 1 | 31| 167 | 1228 | 4576 | 
| 4 | 4 src/_pytest/skipping.py | 125 | 187| 583 | 1811 | 6132 | 
| 5 | 4 src/_pytest/runner.py | 247 | 274| 237 | 2048 | 6132 | 
| 6 | 5 testing/python/collect.py | 548 | 647| 663 | 2711 | 15719 | 
| 7 | 5 testing/python/collect.py | 268 | 333| 472 | 3183 | 15719 | 
| 8 | 6 doc/en/example/xfail_demo.py | 1 | 39| 143 | 3326 | 15863 | 
| 9 | **7 src/_pytest/assertion/rewrite.py** | 1 | 55| 367 | 3693 | 25267 | 
| 10 | 8 src/_pytest/unittest.py | 248 | 289| 286 | 3979 | 27314 | 
| 11 | 9 src/_pytest/doctest.py | 1 | 35| 253 | 4232 | 31477 | 
| 12 | 10 src/_pytest/assertion/util.py | 99 | 135| 159 | 4391 | 34735 | 
| 13 | 11 testing/example_scripts/issue_519.py | 1 | 31| 350 | 4741 | 35201 | 
| 14 | 12 src/_pytest/reports.py | 34 | 127| 567 | 5308 | 38178 | 
| 15 | 13 doc/en/example/assertion/failure_demo.py | 164 | 202| 256 | 5564 | 39833 | 
| 16 | 13 doc/en/example/assertion/failure_demo.py | 205 | 252| 229 | 5793 | 39833 | 
| 17 | 14 testing/python/fixtures.py | 2018 | 2180| 969 | 6762 | 64632 | 
| 18 | 15 src/_pytest/pytester.py | 414 | 438| 219 | 6981 | 75196 | 
| 19 | 16 src/_pytest/_code/code.py | 578 | 621| 308 | 7289 | 83187 | 
| 20 | 16 src/_pytest/assertion/util.py | 1 | 28| 179 | 7468 | 83187 | 
| 21 | 16 testing/python/collect.py | 1008 | 1023| 134 | 7602 | 83187 | 
| 22 | 17 src/_pytest/nose.py | 1 | 70| 508 | 8110 | 83696 | 
| 23 | 17 testing/python/fixtures.py | 2182 | 3108| 6112 | 14222 | 83696 | 
| 24 | 17 testing/python/collect.py | 1025 | 1044| 168 | 14390 | 83696 | 
| 25 | 17 src/_pytest/reports.py | 1 | 31| 228 | 14618 | 83696 | 
| 26 | 18 src/_pytest/compat.py | 321 | 458| 804 | 15422 | 86846 | 
| 27 | 19 src/_pytest/terminal.py | 624 | 657| 338 | 15760 | 95325 | 
| 28 | 19 src/_pytest/reports.py | 190 | 205| 159 | 15919 | 95325 | 
| 29 | **19 src/_pytest/assertion/rewrite.py** | 665 | 728| 455 | 16374 | 95325 | 
| 30 | 19 doc/en/example/assertion/failure_demo.py | 1 | 40| 170 | 16544 | 95325 | 
| 31 | 19 src/_pytest/reports.py | 280 | 342| 421 | 16965 | 95325 | 
| 32 | 19 src/_pytest/unittest.py | 1 | 28| 183 | 17148 | 95325 | 
| 33 | **19 src/_pytest/assertion/rewrite.py** | 814 | 879| 575 | 17723 | 95325 | 
| 34 | 20 src/_pytest/_io/saferepr.py | 1 | 21| 130 | 17853 | 96000 | 
| 35 | 20 src/_pytest/_code/code.py | 1 | 29| 163 | 18016 | 96000 | 
| 36 | 20 src/_pytest/assertion/util.py | 138 | 185| 452 | 18468 | 96000 | 
| 37 | 21 extra/get_issues.py | 55 | 86| 231 | 18699 | 96543 | 
| 38 | 22 testing/python/raises.py | 1 | 67| 447 | 19146 | 98372 | 
| 39 | 23 src/_pytest/junitxml.py | 1 | 88| 687 | 19833 | 103385 | 
| 40 | 23 src/_pytest/doctest.py | 220 | 276| 505 | 20338 | 103385 | 
| 41 | 23 src/_pytest/runner.py | 38 | 64| 232 | 20570 | 103385 | 
| 42 | 23 src/_pytest/runner.py | 1 | 35| 191 | 20761 | 103385 | 
| 43 | 23 src/_pytest/terminal.py | 823 | 844| 181 | 20942 | 103385 | 
| 44 | 23 src/_pytest/terminal.py | 396 | 445| 421 | 21363 | 103385 | 
| 45 | 23 src/_pytest/doctest.py | 88 | 129| 319 | 21682 | 103385 | 
| 46 | 24 src/_pytest/python_api.py | 552 | 671| 1035 | 22717 | 109814 | 
| 47 | 24 testing/python/fixtures.py | 47 | 1011| 6204 | 28921 | 109814 | 
| 48 | 24 src/_pytest/reports.py | 268 | 277| 134 | 29055 | 109814 | 
| 49 | **24 src/_pytest/assertion/rewrite.py** | 466 | 490| 232 | 29287 | 109814 | 
| 50 | 24 src/_pytest/compat.py | 1 | 98| 634 | 29921 | 109814 | 
| 51 | 24 testing/python/collect.py | 1296 | 1306| 110 | 30031 | 109814 | 
| 52 | 24 testing/python/collect.py | 1046 | 1064| 155 | 30186 | 109814 | 
| 53 | 24 testing/python/collect.py | 1097 | 1134| 209 | 30395 | 109814 | 
| 54 | 24 src/_pytest/terminal.py | 885 | 915| 260 | 30655 | 109814 | 
| 55 | 24 testing/python/fixtures.py | 3110 | 3953| 5054 | 35709 | 109814 | 
| 56 | 24 src/_pytest/assertion/util.py | 251 | 283| 312 | 36021 | 109814 | 
| 57 | 24 testing/python/fixtures.py | 1013 | 2016| 6226 | 42247 | 109814 | 
| 58 | 24 doc/en/example/assertion/failure_demo.py | 124 | 161| 148 | 42395 | 109814 | 
| 59 | 24 src/_pytest/reports.py | 157 | 188| 238 | 42633 | 109814 | 
| 60 | 24 src/_pytest/junitxml.py | 253 | 281| 223 | 42856 | 109814 | 
| 61 | 24 src/_pytest/junitxml.py | 218 | 235| 164 | 43020 | 109814 | 
| 62 | 24 src/_pytest/pytester.py | 299 | 330| 248 | 43268 | 109814 | 
| 63 | 24 src/_pytest/terminal.py | 797 | 821| 204 | 43472 | 109814 | 
| 64 | 24 src/_pytest/reports.py | 129 | 155| 159 | 43631 | 109814 | 
| 65 | 24 src/_pytest/terminal.py | 917 | 932| 188 | 43819 | 109814 | 
| 66 | 25 src/_pytest/python.py | 1245 | 1271| 208 | 44027 | 121630 | 
| 67 | **25 src/_pytest/assertion/rewrite.py** | 1048 | 1085| 422 | 44449 | 121630 | 
| 68 | 26 testing/python/integration.py | 1 | 35| 240 | 44689 | 124403 | 
| 69 | 26 src/_pytest/terminal.py | 516 | 554| 324 | 45013 | 124403 | 
| 70 | 27 src/_pytest/stepwise.py | 78 | 115| 311 | 45324 | 125169 | 
| 71 | 28 src/_pytest/freeze_support.py | 1 | 48| 273 | 45597 | 125443 | 
| 72 | 28 src/_pytest/junitxml.py | 125 | 166| 340 | 45937 | 125443 | 
| 73 | 28 src/_pytest/junitxml.py | 237 | 251| 133 | 46070 | 125443 | 
| 74 | 28 src/_pytest/reports.py | 388 | 408| 144 | 46214 | 125443 | 
| 75 | 28 src/_pytest/doctest.py | 132 | 178| 342 | 46556 | 125443 | 
| 76 | 28 src/_pytest/python.py | 364 | 391| 249 | 46805 | 125443 | 
| 77 | 29 bench/bench_argcomplete.py | 1 | 20| 179 | 46984 | 125622 | 
| 78 | 29 src/_pytest/python.py | 206 | 240| 334 | 47318 | 125622 | 
| 79 | 29 src/_pytest/reports.py | 207 | 265| 481 | 47799 | 125622 | 
| 80 | **29 src/_pytest/assertion/rewrite.py** | 343 | 371| 351 | 48150 | 125622 | 
| 81 | 29 src/_pytest/doctest.py | 468 | 504| 343 | 48493 | 125622 | 
| 82 | 29 doc/en/example/assertion/failure_demo.py | 43 | 121| 683 | 49176 | 125622 | 
| 83 | 29 src/_pytest/reports.py | 411 | 435| 171 | 49347 | 125622 | 
| 84 | 30 src/_pytest/config/__init__.py | 1030 | 1054| 149 | 49496 | 134041 | 
| 85 | 30 src/_pytest/unittest.py | 219 | 245| 196 | 49692 | 134041 | 
| 86 | **30 src/_pytest/assertion/rewrite.py** | 953 | 965| 174 | 49866 | 134041 | 
| 87 | 31 testing/python/metafunc.py | 80 | 124| 450 | 50316 | 146503 | 
| 88 | 31 src/_pytest/terminal.py | 759 | 795| 300 | 50616 | 146503 | 
| 89 | 32 src/_pytest/warnings.py | 138 | 161| 159 | 50775 | 147820 | 
| 90 | 32 src/_pytest/pytester.py | 1 | 37| 226 | 51001 | 147820 | 
| 91 | 32 src/_pytest/junitxml.py | 168 | 216| 317 | 51318 | 147820 | 
| 92 | 32 extra/get_issues.py | 33 | 52| 143 | 51461 | 147820 | 
| 93 | 32 src/_pytest/python_api.py | 49 | 105| 411 | 51872 | 147820 | 
| 94 | 32 doc/en/example/assertion/failure_demo.py | 255 | 282| 165 | 52037 | 147820 | 
| 95 | 32 src/_pytest/terminal.py | 934 | 959| 200 | 52237 | 147820 | 
| 96 | 32 src/_pytest/python.py | 1273 | 1300| 217 | 52454 | 147820 | 
| 97 | 32 src/_pytest/terminal.py | 1 | 31| 160 | 52614 | 147820 | 
| 98 | 33 src/_pytest/outcomes.py | 33 | 75| 264 | 52878 | 149183 | 
| 99 | 33 src/_pytest/doctest.py | 343 | 361| 137 | 53015 | 149183 | 
| 100 | 33 src/_pytest/reports.py | 344 | 385| 315 | 53330 | 149183 | 
| 101 | **33 src/_pytest/assertion/rewrite.py** | 918 | 951| 353 | 53683 | 149183 | 
| 102 | 33 src/_pytest/_code/code.py | 1039 | 1066| 231 | 53914 | 149183 | 
| 103 | 34 doc/en/example/multipython.py | 48 | 73| 176 | 54090 | 149630 | 
| 104 | 34 src/_pytest/assertion/util.py | 286 | 327| 329 | 54419 | 149630 | 
| 105 | **34 src/_pytest/assertion/rewrite.py** | 730 | 760| 262 | 54681 | 149630 | 
| 106 | 35 src/_pytest/deprecated.py | 1 | 68| 732 | 55413 | 150599 | 
| 107 | 35 testing/python/fixtures.py | 1 | 44| 259 | 55672 | 150599 | 
| 108 | 35 src/_pytest/junitxml.py | 503 | 528| 179 | 55851 | 150599 | 
| 109 | 35 testing/python/collect.py | 872 | 909| 322 | 56173 | 150599 | 
| 110 | 35 testing/python/integration.py | 37 | 68| 227 | 56400 | 150599 | 
| 111 | 35 src/_pytest/pytester.py | 272 | 297| 212 | 56612 | 150599 | 
| 112 | 35 src/_pytest/pytester.py | 155 | 200| 323 | 56935 | 150599 | 
| 113 | 35 src/_pytest/terminal.py | 266 | 290| 162 | 57097 | 150599 | 
| 114 | 35 src/_pytest/terminal.py | 846 | 859| 130 | 57227 | 150599 | 
| 115 | 35 testing/python/collect.py | 1 | 34| 239 | 57466 | 150599 | 
| 116 | 36 src/_pytest/fixtures.py | 779 | 808| 245 | 57711 | 161358 | 
| 117 | 36 testing/example_scripts/issue_519.py | 34 | 52| 115 | 57826 | 161358 | 
| 118 | 36 testing/python/metafunc.py | 1193 | 1231| 244 | 58070 | 161358 | 
| 119 | 36 src/_pytest/skipping.py | 33 | 70| 363 | 58433 | 161358 | 
| 120 | 37 src/_pytest/pastebin.py | 93 | 114| 185 | 58618 | 162204 | 
| 121 | 37 testing/python/collect.py | 1137 | 1180| 332 | 58950 | 162204 | 
| 122 | 37 src/_pytest/junitxml.py | 530 | 614| 626 | 59576 | 162204 | 
| 123 | 37 src/_pytest/terminal.py | 585 | 605| 188 | 59764 | 162204 | 
| 124 | 37 testing/python/raises.py | 145 | 183| 265 | 60029 | 162204 | 
| 125 | 37 src/_pytest/terminal.py | 734 | 757| 143 | 60172 | 162204 | 
| 126 | 37 src/_pytest/terminal.py | 680 | 696| 123 | 60295 | 162204 | 
| 127 | 37 testing/python/collect.py | 1183 | 1209| 182 | 60477 | 162204 | 
| 128 | 37 src/_pytest/python.py | 1 | 61| 422 | 60899 | 162204 | 
| 129 | **37 src/_pytest/assertion/rewrite.py** | 1030 | 1046| 179 | 61078 | 162204 | 
| 130 | 37 testing/python/collect.py | 185 | 238| 321 | 61399 | 162204 | 
| 131 | **37 src/_pytest/assertion/rewrite.py** | 608 | 663| 542 | 61941 | 162204 | 
| **-> 132 <-** | **37 src/_pytest/assertion/rewrite.py** | 967 | 998| 327 | 62268 | 162204 | 
| 133 | 37 testing/python/collect.py | 1349 | 1382| 239 | 62507 | 162204 | 
| 134 | 37 src/_pytest/python_api.py | 189 | 220| 253 | 62760 | 162204 | 
| 135 | 37 src/_pytest/assertion/util.py | 64 | 96| 276 | 63036 | 162204 | 
| 136 | 37 testing/python/collect.py | 529 | 546| 183 | 63219 | 162204 | 
| 137 | 37 src/_pytest/runner.py | 118 | 131| 129 | 63348 | 162204 | 
| 138 | 37 src/_pytest/python.py | 299 | 313| 141 | 63489 | 162204 | 
| 139 | 37 testing/python/collect.py | 1067 | 1094| 183 | 63672 | 162204 | 
| 140 | 37 src/_pytest/unittest.py | 138 | 169| 250 | 63922 | 162204 | 
| 141 | **37 src/_pytest/assertion/rewrite.py** | 374 | 427| 538 | 64460 | 162204 | 
| 142 | 37 src/_pytest/doctest.py | 388 | 407| 159 | 64619 | 162204 | 
| 143 | 37 src/_pytest/terminal.py | 1015 | 1034| 171 | 64790 | 162204 | 
| 144 | 37 src/_pytest/python.py | 1303 | 1372| 493 | 65283 | 162204 | 
| 145 | 37 src/_pytest/runner.py | 102 | 115| 119 | 65402 | 162204 | 
| 146 | 38 testing/python/approx.py | 429 | 445| 138 | 65540 | 168033 | 
| 147 | 38 src/_pytest/python.py | 813 | 849| 354 | 65894 | 168033 | 
| 148 | 38 src/_pytest/assertion/util.py | 330 | 366| 365 | 66259 | 168033 | 
| 149 | 38 src/_pytest/terminal.py | 1037 | 1086| 347 | 66606 | 168033 | 
| 150 | 38 src/_pytest/python.py | 165 | 180| 209 | 66815 | 168033 | 
| 151 | 39 setup.py | 1 | 18| 194 | 67009 | 168356 | 
| 152 | 39 testing/python/collect.py | 1309 | 1346| 320 | 67329 | 168356 | 
| 153 | 40 src/_pytest/assertion/truncate.py | 1 | 40| 235 | 67564 | 169109 | 
| 154 | 41 src/_pytest/resultlog.py | 66 | 83| 145 | 67709 | 169858 | 
| 155 | 41 src/_pytest/doctest.py | 38 | 85| 321 | 68030 | 169858 | 
| 156 | 41 src/_pytest/doctest.py | 313 | 340| 211 | 68241 | 169858 | 
| 157 | 41 testing/python/approx.py | 28 | 61| 500 | 68741 | 169858 | 
| 158 | 41 src/_pytest/_code/code.py | 896 | 937| 294 | 69035 | 169858 | 
| 159 | 42 doc/en/conf.py | 1 | 105| 775 | 69810 | 172343 | 
| 160 | 42 src/_pytest/terminal.py | 861 | 883| 210 | 70020 | 172343 | 
| **-> 161 <-** | **42 src/_pytest/assertion/rewrite.py** | 1000 | 1028| 297 | 70317 | 172343 | 
| 162 | 43 src/_pytest/pathlib.py | 1 | 94| 489 | 70806 | 174486 | 
| 163 | 44 src/_pytest/main.py | 36 | 144| 757 | 71563 | 180261 | 
| 164 | 44 testing/python/metafunc.py | 422 | 461| 330 | 71893 | 180261 | 
| 165 | 44 src/_pytest/junitxml.py | 387 | 425| 295 | 72188 | 180261 | 
| 166 | 44 src/_pytest/python_api.py | 672 | 715| 379 | 72567 | 180261 | 
| 167 | 44 testing/python/approx.py | 1 | 25| 134 | 72701 | 180261 | 
| 168 | 44 src/_pytest/python.py | 64 | 121| 375 | 73076 | 180261 | 
| 169 | **44 src/_pytest/assertion/rewrite.py** | 908 | 916| 136 | 73212 | 180261 | 
| 170 | **44 src/_pytest/assertion/rewrite.py** | 430 | 463| 285 | 73497 | 180261 | 
| 171 | 44 src/_pytest/terminal.py | 59 | 147| 632 | 74129 | 180261 | 
| 172 | 44 testing/python/metafunc.py | 1135 | 1167| 200 | 74329 | 180261 | 
| 173 | 44 src/_pytest/terminal.py | 962 | 1012| 485 | 74814 | 180261 | 
| 174 | 45 src/_pytest/helpconfig.py | 228 | 247| 145 | 74959 | 181989 | 
| 175 | 45 testing/python/collect.py | 650 | 679| 257 | 75216 | 181989 | 
| 176 | 46 src/_pytest/logging.py | 592 | 611| 167 | 75383 | 186719 | 
| 177 | 46 src/_pytest/python.py | 393 | 438| 394 | 75777 | 186719 | 
| 178 | 47 src/_pytest/_code/__init__.py | 1 | 15| 123 | 75900 | 186843 | 
| 179 | 48 doc/en/example/nonpython/conftest.py | 1 | 47| 321 | 76221 | 187164 | 
| 180 | 48 testing/python/collect.py | 80 | 114| 256 | 76477 | 187164 | 
| 181 | 48 src/_pytest/doctest.py | 181 | 218| 309 | 76786 | 187164 | 
| 182 | 48 src/_pytest/assertion/util.py | 369 | 398| 258 | 77044 | 187164 | 
| 183 | 48 testing/python/approx.py | 297 | 313| 207 | 77251 | 187164 | 
| 184 | 49 testing/python/setup_only.py | 251 | 270| 125 | 77376 | 188697 | 
| 185 | 49 src/_pytest/python.py | 316 | 345| 272 | 77648 | 188697 | 
| 186 | 49 src/_pytest/terminal.py | 346 | 359| 126 | 77774 | 188697 | 
| 187 | 49 src/_pytest/junitxml.py | 465 | 501| 292 | 78066 | 188697 | 
| 188 | 50 src/_pytest/mark/evaluate.py | 50 | 71| 192 | 78258 | 189490 | 
| 189 | 50 src/_pytest/_io/saferepr.py | 50 | 67| 181 | 78439 | 189490 | 
| 190 | 50 src/_pytest/main.py | 295 | 341| 339 | 78778 | 189490 | 
| 191 | 50 testing/python/raises.py | 212 | 227| 135 | 78913 | 189490 | 
| 192 | **50 src/_pytest/assertion/rewrite.py** | 493 | 516| 229 | 79142 | 189490 | 
| 193 | 50 testing/python/collect.py | 837 | 869| 304 | 79446 | 189490 | 
| 194 | 50 setup.py | 21 | 43| 129 | 79575 | 189490 | 
| 195 | 50 src/_pytest/unittest.py | 171 | 197| 173 | 79748 | 189490 | 
| 196 | **50 src/_pytest/assertion/rewrite.py** | 84 | 163| 795 | 80543 | 189490 | 
| 197 | 50 testing/python/metafunc.py | 705 | 732| 211 | 80754 | 189490 | 
| 198 | 50 src/_pytest/pytester.py | 131 | 152| 244 | 80998 | 189490 | 
| 199 | 50 testing/python/approx.py | 447 | 476| 226 | 81224 | 189490 | 
| 200 | 51 testing/example_scripts/collect/package_infinite_recursion/conftest.py | 1 | 3| 0 | 81224 | 189500 | 
| 201 | 51 src/_pytest/doctest.py | 507 | 538| 226 | 81450 | 189500 | 
| 202 | 52 src/_pytest/_code/source.py | 234 | 262| 162 | 81612 | 191857 | 
| 203 | 52 src/_pytest/terminal.py | 150 | 177| 216 | 81828 | 191857 | 
| 204 | 52 testing/python/collect.py | 484 | 511| 208 | 82036 | 191857 | 
| 205 | 52 src/_pytest/pytester.py | 333 | 366| 157 | 82193 | 191857 | 
| 206 | 52 src/_pytest/terminal.py | 228 | 251| 204 | 82397 | 191857 | 
| 207 | 52 testing/python/metafunc.py | 1 | 37| 212 | 82609 | 191857 | 
| 208 | 52 testing/python/approx.py | 489 | 510| 208 | 82817 | 191857 | 
| 209 | 52 testing/python/approx.py | 79 | 97| 242 | 83059 | 191857 | 
| 210 | 52 src/_pytest/assertion/__init__.py | 104 | 156| 407 | 83466 | 191857 | 
| 211 | 52 src/_pytest/_code/code.py | 867 | 893| 246 | 83712 | 191857 | 
| 212 | 52 src/_pytest/terminal.py | 607 | 622| 129 | 83841 | 191857 | 
| 213 | 52 testing/python/setup_only.py | 33 | 59| 167 | 84008 | 191857 | 
| 214 | 52 src/_pytest/pytester.py | 982 | 1017| 308 | 84316 | 191857 | 
| 215 | 52 testing/python/collect.py | 1242 | 1273| 203 | 84519 | 191857 | 
| 216 | 52 src/_pytest/main.py | 504 | 549| 379 | 84898 | 191857 | 
| 217 | 52 src/_pytest/terminal.py | 305 | 326| 170 | 85068 | 191857 | 
| 218 | 52 src/_pytest/resultlog.py | 85 | 102| 160 | 85228 | 191857 | 
| 219 | 52 src/_pytest/_code/source.py | 289 | 329| 369 | 85597 | 191857 | 
| 220 | 52 testing/python/approx.py | 412 | 427| 124 | 85721 | 191857 | 
| 221 | 52 testing/python/integration.py | 71 | 86| 109 | 85830 | 191857 | 
| 222 | 52 src/_pytest/main.py | 724 | 774| 430 | 86260 | 191857 | 
| 223 | 52 src/_pytest/mark/evaluate.py | 73 | 126| 348 | 86608 | 191857 | 
| 224 | 52 testing/python/metafunc.py | 776 | 795| 130 | 86738 | 191857 | 
| 225 | 52 src/_pytest/assertion/__init__.py | 58 | 101| 317 | 87055 | 191857 | 
| 226 | 52 src/_pytest/skipping.py | 1 | 30| 190 | 87245 | 191857 | 
| 227 | 52 src/_pytest/_code/code.py | 835 | 864| 208 | 87453 | 191857 | 
| 228 | **52 src/_pytest/assertion/rewrite.py** | 308 | 340| 218 | 87671 | 191857 | 
| 229 | 52 src/_pytest/_code/code.py | 387 | 424| 277 | 87948 | 191857 | 
| 230 | 52 testing/python/collect.py | 36 | 58| 188 | 88136 | 191857 | 
| 231 | 52 src/_pytest/skipping.py | 73 | 92| 167 | 88303 | 191857 | 
| 232 | 52 src/_pytest/pytester.py | 252 | 270| 166 | 88469 | 191857 | 
| 233 | 53 testing/example_scripts/perf_examples/collect_stats/generate_folders.py | 1 | 28| 197 | 88666 | 192054 | 
| 234 | 54 src/_pytest/hookspec.py | 471 | 485| 122 | 88788 | 196458 | 
| 235 | 54 testing/python/approx.py | 63 | 77| 250 | 89038 | 196458 | 
| 236 | 55 src/_pytest/recwarn.py | 220 | 251| 260 | 89298 | 198389 | 
| 237 | 56 doc/en/example/pythoncollection.py | 1 | 15| 0 | 89298 | 198437 | 
| 238 | 56 src/_pytest/helpconfig.py | 143 | 210| 554 | 89852 | 198437 | 
| 239 | 56 testing/python/collect.py | 513 | 527| 151 | 90003 | 198437 | 
| 240 | 57 doc/en/example/py2py3/conftest.py | 1 | 17| 0 | 90003 | 198523 | 
| 241 | 57 testing/python/integration.py | 223 | 245| 184 | 90187 | 198523 | 
| 242 | 57 testing/python/collect.py | 779 | 801| 196 | 90383 | 198523 | 
| 243 | 57 testing/python/collect.py | 1276 | 1293| 130 | 90513 | 198523 | 
| 244 | 57 testing/python/collect.py | 241 | 266| 182 | 90695 | 198523 | 
| 245 | 58 src/_pytest/setuponly.py | 53 | 89| 265 | 90960 | 199110 | 
| 246 | **58 src/_pytest/assertion/rewrite.py** | 519 | 605| 598 | 91558 | 199110 | 
| 247 | 58 src/_pytest/pastebin.py | 27 | 45| 177 | 91735 | 199110 | 
| 248 | 58 src/_pytest/hookspec.py | 382 | 397| 132 | 91867 | 199110 | 
| 249 | 58 src/_pytest/terminal.py | 468 | 481| 150 | 92017 | 199110 | 
| 250 | 58 src/_pytest/unittest.py | 82 | 105| 157 | 92174 | 199110 | 


### Hint

```
Hello, I am new here and would be interested in working on this issue if that is possible.
@danielx123 
Sure!  But I don't think this is an easy issue, since it involved the assertion rewriting - but if you're familar with Python's AST and pytest's internals feel free to pick this up.
We also have a tag "easy" for issues that are probably easier for starting contributors: https://github.com/pytest-dev/pytest/issues?q=is%3Aopen+is%3Aissue+label%3A%22status%3A+easy%22
I was planning on starting a pr today, but probably won't be able to finish it until next week - @danielx123 maybe we could collaborate? 
```

## Patch

```diff
diff --git a/src/_pytest/assertion/rewrite.py b/src/_pytest/assertion/rewrite.py
--- a/src/_pytest/assertion/rewrite.py
+++ b/src/_pytest/assertion/rewrite.py
@@ -964,6 +964,8 @@ def visit_Call_35(self, call):
         """
         visit `ast.Call` nodes on Python3.5 and after
         """
+        if isinstance(call.func, ast.Name) and call.func.id == "all":
+            return self._visit_all(call)
         new_func, func_expl = self.visit(call.func)
         arg_expls = []
         new_args = []
@@ -987,6 +989,27 @@ def visit_Call_35(self, call):
         outer_expl = "%s\n{%s = %s\n}" % (res_expl, res_expl, expl)
         return res, outer_expl
 
+    def _visit_all(self, call):
+        """Special rewrite for the builtin all function, see #5062"""
+        if not isinstance(call.args[0], (ast.GeneratorExp, ast.ListComp)):
+            return
+        gen_exp = call.args[0]
+        assertion_module = ast.Module(
+            body=[ast.Assert(test=gen_exp.elt, lineno=1, msg="", col_offset=1)]
+        )
+        AssertionRewriter(module_path=None, config=None).run(assertion_module)
+        for_loop = ast.For(
+            iter=gen_exp.generators[0].iter,
+            target=gen_exp.generators[0].target,
+            body=assertion_module.body,
+            orelse=[],
+        )
+        self.statements.append(for_loop)
+        return (
+            ast.Num(n=1),
+            "",
+        )  # Return an empty expression, all the asserts are in the for_loop
+
     def visit_Starred(self, starred):
         # From Python 3.5, a Starred node can appear in a function call
         res, expl = self.visit(starred.value)
@@ -997,6 +1020,8 @@ def visit_Call_legacy(self, call):
         """
         visit `ast.Call nodes on 3.4 and below`
         """
+        if isinstance(call.func, ast.Name) and call.func.id == "all":
+            return self._visit_all(call)
         new_func, func_expl = self.visit(call.func)
         arg_expls = []
         new_args = []

```

## Test Patch

```diff
diff --git a/testing/test_assertrewrite.py b/testing/test_assertrewrite.py
--- a/testing/test_assertrewrite.py
+++ b/testing/test_assertrewrite.py
@@ -656,6 +656,12 @@ def __repr__(self):
         else:
             assert lines == ["assert 0 == 1\n +  where 1 = \\n{ \\n~ \\n}.a"]
 
+    def test_unroll_expression(self):
+        def f():
+            assert all(x == 1 for x in range(10))
+
+        assert "0 == 1" in getmsg(f)
+
     def test_custom_repr_non_ascii(self):
         def f():
             class A(object):
@@ -671,6 +677,53 @@ def __repr__(self):
         assert "UnicodeDecodeError" not in msg
         assert "UnicodeEncodeError" not in msg
 
+    def test_unroll_generator(self, testdir):
+        testdir.makepyfile(
+            """
+            def check_even(num):
+                if num % 2 == 0:
+                    return True
+                return False
+
+            def test_generator():
+                odd_list = list(range(1,9,2))
+                assert all(check_even(num) for num in odd_list)"""
+        )
+        result = testdir.runpytest()
+        result.stdout.fnmatch_lines(["*assert False*", "*where False = check_even(1)*"])
+
+    def test_unroll_list_comprehension(self, testdir):
+        testdir.makepyfile(
+            """
+            def check_even(num):
+                if num % 2 == 0:
+                    return True
+                return False
+
+            def test_list_comprehension():
+                odd_list = list(range(1,9,2))
+                assert all([check_even(num) for num in odd_list])"""
+        )
+        result = testdir.runpytest()
+        result.stdout.fnmatch_lines(["*assert False*", "*where False = check_even(1)*"])
+
+    def test_for_loop(self, testdir):
+        testdir.makepyfile(
+            """
+            def check_even(num):
+                if num % 2 == 0:
+                    return True
+                return False
+
+            def test_for_loop():
+                odd_list = list(range(1,9,2))
+                for num in odd_list:
+                    assert check_even(num)
+        """
+        )
+        result = testdir.runpytest()
+        result.stdout.fnmatch_lines(["*assert False*", "*where False = check_even(1)*"])
+
 
 class TestRewriteOnImport(object):
     def test_pycache_is_a_file(self, testdir):

```


## Code snippets

### 1 - src/pytest.py:

Start line: 1, End line: 107

```python
# PYTHON_ARGCOMPLETE_OK
"""
pytest: unit and functional testing with Python.
"""
# else we are imported
from _pytest import __version__
from _pytest.assertion import register_assert_rewrite
from _pytest.config import cmdline
from _pytest.config import hookimpl
from _pytest.config import hookspec
from _pytest.config import main
from _pytest.config import UsageError
from _pytest.debugging import pytestPDB as __pytestPDB
from _pytest.fixtures import fillfixtures as _fillfuncargs
from _pytest.fixtures import fixture
from _pytest.fixtures import yield_fixture
from _pytest.freeze_support import freeze_includes
from _pytest.main import Session
from _pytest.mark import MARK_GEN as mark
from _pytest.mark import param
from _pytest.nodes import Collector
from _pytest.nodes import File
from _pytest.nodes import Item
from _pytest.outcomes import exit
from _pytest.outcomes import fail
from _pytest.outcomes import importorskip
from _pytest.outcomes import skip
from _pytest.outcomes import xfail
from _pytest.python import Class
from _pytest.python import Function
from _pytest.python import Instance
from _pytest.python import Module
from _pytest.python import Package
from _pytest.python_api import approx
from _pytest.python_api import raises
from _pytest.recwarn import deprecated_call
from _pytest.recwarn import warns
from _pytest.warning_types import PytestAssertRewriteWarning
from _pytest.warning_types import PytestCacheWarning
from _pytest.warning_types import PytestCollectionWarning
from _pytest.warning_types import PytestConfigWarning
from _pytest.warning_types import PytestDeprecationWarning
from _pytest.warning_types import PytestExperimentalApiWarning
from _pytest.warning_types import PytestUnhandledCoroutineWarning
from _pytest.warning_types import PytestUnknownMarkWarning
from _pytest.warning_types import PytestWarning
from _pytest.warning_types import RemovedInPytest4Warning

set_trace = __pytestPDB.set_trace

__all__ = [
    "__version__",
    "_fillfuncargs",
    "approx",
    "Class",
    "cmdline",
    "Collector",
    "deprecated_call",
    "exit",
    "fail",
    "File",
    "fixture",
    "freeze_includes",
    "Function",
    "hookimpl",
    "hookspec",
    "importorskip",
    "Instance",
    "Item",
    "main",
    "mark",
    "Module",
    "Package",
    "param",
    "PytestAssertRewriteWarning",
    "PytestCacheWarning",
    "PytestCollectionWarning",
    "PytestConfigWarning",
    "PytestDeprecationWarning",
    "PytestExperimentalApiWarning",
    "PytestUnhandledCoroutineWarning",
    "PytestUnknownMarkWarning",
    "PytestWarning",
    "raises",
    "register_assert_rewrite",
    "RemovedInPytest4Warning",
    "Session",
    "set_trace",
    "skip",
    "UsageError",
    "warns",
    "xfail",
    "yield_fixture",
]

if __name__ == "__main__":
    # if run as a script or by 'python -m pytest'
    # we trigger the below "else" condition by the following import
    import pytest

    raise SystemExit(pytest.main())
else:

    from _pytest.compat import _setup_collect_fakemodule

    _setup_collect_fakemodule()
```
### 2 - src/_pytest/runner.py:

Start line: 156, End line: 198

```python
def pytest_report_teststatus(report):
    if report.when in ("setup", "teardown"):
        if report.failed:
            #      category, shortletter, verbose-word
            return "error", "E", "ERROR"
        elif report.skipped:
            return "skipped", "s", "SKIPPED"
        else:
            return "", "", ""


#
# Implementation


def call_and_report(item, when, log=True, **kwds):
    call = call_runtest_hook(item, when, **kwds)
    hook = item.ihook
    report = hook.pytest_runtest_makereport(item=item, call=call)
    if log:
        hook.pytest_runtest_logreport(report=report)
    if check_interactive_exception(call, report):
        hook.pytest_exception_interact(node=item, call=call, report=report)
    return report


def check_interactive_exception(call, report):
    return call.excinfo and not (
        hasattr(report, "wasxfail")
        or call.excinfo.errisinstance(Skipped)
        or call.excinfo.errisinstance(bdb.BdbQuit)
    )


def call_runtest_hook(item, when, **kwds):
    hookname = "pytest_runtest_" + when
    ihook = getattr(item.ihook, hookname)
    reraise = (Exit,)
    if not item.config.getoption("usepdb", False):
        reraise += (KeyboardInterrupt,)
    return CallInfo.from_call(
        lambda: ihook(item=item, **kwds), when=when, reraise=reraise
    )
```
### 3 - src/_pytest/assertion/__init__.py:

Start line: 1, End line: 31

```python
"""
support for presenting detailed information in failing assertions.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import six

from _pytest.assertion import rewrite
from _pytest.assertion import truncate
from _pytest.assertion import util


def pytest_addoption(parser):
    group = parser.getgroup("debugconfig")
    group.addoption(
        "--assert",
        action="store",
        dest="assertmode",
        choices=("rewrite", "plain"),
        default="rewrite",
        metavar="MODE",
        help="""Control assertion debugging tools.  'plain'
                            performs no assertion debugging.  'rewrite'
                            (the default) rewrites assert statements in
                            test modules on import to provide assert
                            expression information.""",
    )
```
### 4 - src/_pytest/skipping.py:

Start line: 125, End line: 187

```python
@hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    evalxfail = getattr(item, "_evalxfail", None)
    # unitttest special case, see setting of _unexpectedsuccess
    if hasattr(item, "_unexpectedsuccess") and rep.when == "call":
        from _pytest.compat import _is_unittest_unexpected_success_a_failure

        if item._unexpectedsuccess:
            rep.longrepr = "Unexpected success: {}".format(item._unexpectedsuccess)
        else:
            rep.longrepr = "Unexpected success"
        if _is_unittest_unexpected_success_a_failure():
            rep.outcome = "failed"
        else:
            rep.outcome = "passed"
            rep.wasxfail = rep.longrepr
    elif item.config.option.runxfail:
        pass  # don't interefere
    elif call.excinfo and call.excinfo.errisinstance(xfail.Exception):
        rep.wasxfail = "reason: " + call.excinfo.value.msg
        rep.outcome = "skipped"
    elif evalxfail and not rep.skipped and evalxfail.wasvalid() and evalxfail.istrue():
        if call.excinfo:
            if evalxfail.invalidraise(call.excinfo.value):
                rep.outcome = "failed"
            else:
                rep.outcome = "skipped"
                rep.wasxfail = evalxfail.getexplanation()
        elif call.when == "call":
            strict_default = item.config.getini("xfail_strict")
            is_strict_xfail = evalxfail.get("strict", strict_default)
            explanation = evalxfail.getexplanation()
            if is_strict_xfail:
                rep.outcome = "failed"
                rep.longrepr = "[XPASS(strict)] {}".format(explanation)
            else:
                rep.outcome = "passed"
                rep.wasxfail = explanation
    elif (
        getattr(item, "_skipped_by_mark", False)
        and rep.skipped
        and type(rep.longrepr) is tuple
    ):
        # skipped by mark.skipif; change the location of the failure
        # to point to the item definition, otherwise it will display
        # the location of where the skip exception was raised within pytest
        filename, line, reason = rep.longrepr
        filename, line = item.location[:2]
        rep.longrepr = filename, line, reason


# called by terminalreporter progress reporting


def pytest_report_teststatus(report):
    if hasattr(report, "wasxfail"):
        if report.skipped:
            return "xfailed", "x", "XFAIL"
        elif report.passed:
            return "xpassed", "X", "XPASS"
```
### 5 - src/_pytest/runner.py:

Start line: 247, End line: 274

```python
def pytest_runtest_makereport(item, call):
    return TestReport.from_item_and_call(item, call)


def pytest_make_collect_report(collector):
    call = CallInfo.from_call(lambda: list(collector.collect()), "collect")
    longrepr = None
    if not call.excinfo:
        outcome = "passed"
    else:
        from _pytest import nose

        skip_exceptions = (Skipped,) + nose.get_skip_exceptions()
        if call.excinfo.errisinstance(skip_exceptions):
            outcome = "skipped"
            r = collector._repr_failure_py(call.excinfo, "line").reprcrash
            longrepr = (str(r.path), r.lineno, r.message)
        else:
            outcome = "failed"
            errorinfo = collector.repr_failure(call.excinfo)
            if not hasattr(errorinfo, "toterminal"):
                errorinfo = CollectErrorRepr(errorinfo)
            longrepr = errorinfo
    rep = CollectReport(
        collector.nodeid, outcome, longrepr, getattr(call, "result", None)
    )
    rep.call = call  # see collect_one_node
    return rep
```
### 6 - testing/python/collect.py:

Start line: 548, End line: 647

```python
class TestFunction(object):

    def test_parametrize_skipif(self, testdir):
        testdir.makepyfile(
            """
            import pytest

            m = pytest.mark.skipif('True')

            @pytest.mark.parametrize('x', [0, 1, pytest.param(2, marks=m)])
            def test_skip_if(x):
                assert x < 2
        """
        )
        result = testdir.runpytest()
        result.stdout.fnmatch_lines(["* 2 passed, 1 skipped in *"])

    def test_parametrize_skip(self, testdir):
        testdir.makepyfile(
            """
            import pytest

            m = pytest.mark.skip('')

            @pytest.mark.parametrize('x', [0, 1, pytest.param(2, marks=m)])
            def test_skip(x):
                assert x < 2
        """
        )
        result = testdir.runpytest()
        result.stdout.fnmatch_lines(["* 2 passed, 1 skipped in *"])

    def test_parametrize_skipif_no_skip(self, testdir):
        testdir.makepyfile(
            """
            import pytest

            m = pytest.mark.skipif('False')

            @pytest.mark.parametrize('x', [0, 1, m(2)])
            def test_skipif_no_skip(x):
                assert x < 2
        """
        )
        result = testdir.runpytest()
        result.stdout.fnmatch_lines(["* 1 failed, 2 passed in *"])

    def test_parametrize_xfail(self, testdir):
        testdir.makepyfile(
            """
            import pytest

            m = pytest.mark.xfail('True')

            @pytest.mark.parametrize('x', [0, 1, pytest.param(2, marks=m)])
            def test_xfail(x):
                assert x < 2
        """
        )
        result = testdir.runpytest()
        result.stdout.fnmatch_lines(["* 2 passed, 1 xfailed in *"])

    def test_parametrize_passed(self, testdir):
        testdir.makepyfile(
            """
            import pytest

            m = pytest.mark.xfail('True')

            @pytest.mark.parametrize('x', [0, 1, pytest.param(2, marks=m)])
            def test_xfail(x):
                pass
        """
        )
        result = testdir.runpytest()
        result.stdout.fnmatch_lines(["* 2 passed, 1 xpassed in *"])

    def test_parametrize_xfail_passed(self, testdir):
        testdir.makepyfile(
            """
            import pytest

            m = pytest.mark.xfail('False')

            @pytest.mark.parametrize('x', [0, 1, m(2)])
            def test_passed(x):
                pass
        """
        )
        result = testdir.runpytest()
        result.stdout.fnmatch_lines(["* 3 passed in *"])

    def test_function_original_name(self, testdir):
        items = testdir.getitems(
            """
            import pytest
            @pytest.mark.parametrize('arg', [1,2])
            def test_func(arg):
                pass
        """
        )
        assert [x.originalname for x in items] == ["test_func", "test_func"]
```
### 7 - testing/python/collect.py:

Start line: 268, End line: 333

```python
class TestFunction(object):

    @staticmethod
    def make_function(testdir, **kwargs):
        from _pytest.fixtures import FixtureManager

        config = testdir.parseconfigure()
        session = testdir.Session(config)
        session._fixturemanager = FixtureManager(session)

        return pytest.Function(config=config, parent=session, **kwargs)

    def test_function_equality(self, testdir, tmpdir):
        def func1():
            pass

        def func2():
            pass

        f1 = self.make_function(testdir, name="name", args=(1,), callobj=func1)
        assert f1 == f1
        f2 = self.make_function(testdir, name="name", callobj=func2)
        assert f1 != f2

    def test_repr_produces_actual_test_id(self, testdir):
        f = self.make_function(
            testdir, name=r"test[\xe5]", callobj=self.test_repr_produces_actual_test_id
        )
        assert repr(f) == r"<Function test[\xe5]>"

    def test_issue197_parametrize_emptyset(self, testdir):
        testdir.makepyfile(
            """
            import pytest
            @pytest.mark.parametrize('arg', [])
            def test_function(arg):
                pass
        """
        )
        reprec = testdir.inline_run()
        reprec.assertoutcome(skipped=1)

    def test_single_tuple_unwraps_values(self, testdir):
        testdir.makepyfile(
            """
            import pytest
            @pytest.mark.parametrize(('arg',), [(1,)])
            def test_function(arg):
                assert arg == 1
        """
        )
        reprec = testdir.inline_run()
        reprec.assertoutcome(passed=1)

    def test_issue213_parametrize_value_no_equal(self, testdir):
        testdir.makepyfile(
            """
            import pytest
            class A(object):
                def __eq__(self, other):
                    raise ValueError("not possible")
            @pytest.mark.parametrize('arg', [A()])
            def test_function(arg):
                assert arg.__class__.__name__ == "A"
        """
        )
        reprec = testdir.inline_run("--fulltrace")
        reprec.assertoutcome(passed=1)
```
### 8 - doc/en/example/xfail_demo.py:

Start line: 1, End line: 39

```python
import pytest

xfail = pytest.mark.xfail


@xfail
def test_hello():
    assert 0


@xfail(run=False)
def test_hello2():
    assert 0


@xfail("hasattr(os, 'sep')")
def test_hello3():
    assert 0


@xfail(reason="bug 110")
def test_hello4():
    assert 0


@xfail('pytest.__version__[0] != "17"')
def test_hello5():
    assert 0


def test_hello6():
    pytest.xfail("reason")


@xfail(raises=IndexError)
def test_hello7():
    x = []
    x[1] = 1
```
### 9 - src/_pytest/assertion/rewrite.py:

Start line: 1, End line: 55

```python
"""Rewrite assertion AST to produce nice error messages"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ast
import errno
import imp
import itertools
import marshal
import os
import re
import string
import struct
import sys
import types

import atomicwrites
import py
import six

from _pytest._io.saferepr import saferepr
from _pytest.assertion import util
from _pytest.assertion.util import (  # noqa: F401
    format_explanation as _format_explanation,
)
from _pytest.compat import spec_from_file_location
from _pytest.pathlib import fnmatch_ex
from _pytest.pathlib import PurePath

# pytest caches rewritten pycs in __pycache__.
if hasattr(imp, "get_tag"):
    PYTEST_TAG = imp.get_tag() + "-PYTEST"
else:
    if hasattr(sys, "pypy_version_info"):
        impl = "pypy"
    elif sys.platform == "java":
        impl = "jython"
    else:
        impl = "cpython"
    ver = sys.version_info
    PYTEST_TAG = "%s-%s%s-PYTEST" % (impl, ver[0], ver[1])
    del ver, impl

PYC_EXT = ".py" + (__debug__ and "c" or "o")
PYC_TAIL = "." + PYTEST_TAG + PYC_EXT

ASCII_IS_DEFAULT_ENCODING = sys.version_info[0] < 3

if sys.version_info >= (3, 5):
    ast_Call = ast.Call
else:

    def ast_Call(a, b, c):
        return ast.Call(a, b, c, None, None)
```
### 10 - src/_pytest/unittest.py:

Start line: 248, End line: 289

```python
# twisted trial support


@hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item):
    if isinstance(item, TestCaseFunction) and "twisted.trial.unittest" in sys.modules:
        ut = sys.modules["twisted.python.failure"]
        Failure__init__ = ut.Failure.__init__
        check_testcase_implements_trial_reporter()

        def excstore(
            self, exc_value=None, exc_type=None, exc_tb=None, captureVars=None
        ):
            if exc_value is None:
                self._rawexcinfo = sys.exc_info()
            else:
                if exc_type is None:
                    exc_type = type(exc_value)
                self._rawexcinfo = (exc_type, exc_value, exc_tb)
            try:
                Failure__init__(
                    self, exc_value, exc_type, exc_tb, captureVars=captureVars
                )
            except TypeError:
                Failure__init__(self, exc_value, exc_type, exc_tb)

        ut.Failure.__init__ = excstore
        yield
        ut.Failure.__init__ = Failure__init__
    else:
        yield


def check_testcase_implements_trial_reporter(done=[]):
    if done:
        return
    from zope.interface import classImplements
    from twisted.trial.itrial import IReporter

    classImplements(TestCaseFunction, IReporter)
    done.append(1)
```
### 29 - src/_pytest/assertion/rewrite.py:

Start line: 665, End line: 728

```python
class AssertionRewriter(ast.NodeVisitor):

    def run(self, mod):
        """Find all assert statements in *mod* and rewrite them."""
        if not mod.body:
            # Nothing to do.
            return
        # Insert some special imports at the top of the module but after any
        # docstrings and __future__ imports.
        aliases = [
            ast.alias(six.moves.builtins.__name__, "@py_builtins"),
            ast.alias("_pytest.assertion.rewrite", "@pytest_ar"),
        ]
        doc = getattr(mod, "docstring", None)
        expect_docstring = doc is None
        if doc is not None and self.is_rewrite_disabled(doc):
            return
        pos = 0
        lineno = 1
        for item in mod.body:
            if (
                expect_docstring
                and isinstance(item, ast.Expr)
                and isinstance(item.value, ast.Str)
            ):
                doc = item.value.s
                if self.is_rewrite_disabled(doc):
                    return
                expect_docstring = False
            elif (
                not isinstance(item, ast.ImportFrom)
                or item.level > 0
                or item.module != "__future__"
            ):
                lineno = item.lineno
                break
            pos += 1
        else:
            lineno = item.lineno
        imports = [
            ast.Import([alias], lineno=lineno, col_offset=0) for alias in aliases
        ]
        mod.body[pos:pos] = imports
        # Collect asserts.
        nodes = [mod]
        while nodes:
            node = nodes.pop()
            for name, field in ast.iter_fields(node):
                if isinstance(field, list):
                    new = []
                    for i, child in enumerate(field):
                        if isinstance(child, ast.Assert):
                            # Transform assert.
                            new.extend(self.visit(child))
                        else:
                            new.append(child)
                            if isinstance(child, ast.AST):
                                nodes.append(child)
                    setattr(node, name, new)
                elif (
                    isinstance(field, ast.AST)
                    # Don't recurse into expressions as they can't contain
                    # asserts.
                    and not isinstance(field, ast.expr)
                ):
                    nodes.append(field)
```
### 33 - src/_pytest/assertion/rewrite.py:

Start line: 814, End line: 879

```python
class AssertionRewriter(ast.NodeVisitor):

    def visit_Assert(self, assert_):
        """Return the AST statements to replace the ast.Assert instance.

        This rewrites the test of an assertion to provide
        intermediate values and replace it with an if statement which
        raises an assertion error with a detailed explanation in case
        the expression is false.

        """
        if isinstance(assert_.test, ast.Tuple) and len(assert_.test.elts) >= 1:
            from _pytest.warning_types import PytestAssertRewriteWarning
            import warnings

            warnings.warn_explicit(
                PytestAssertRewriteWarning(
                    "assertion is always true, perhaps remove parentheses?"
                ),
                category=None,
                filename=str(self.module_path),
                lineno=assert_.lineno,
            )

        self.statements = []
        self.variables = []
        self.variable_counter = itertools.count()
        self.stack = []
        self.on_failure = []
        self.push_format_context()
        # Rewrite assert into a bunch of statements.
        top_condition, explanation = self.visit(assert_.test)
        # If in a test module, check if directly asserting None, in order to warn [Issue #3191]
        if self.module_path is not None:
            self.statements.append(
                self.warn_about_none_ast(
                    top_condition, module_path=self.module_path, lineno=assert_.lineno
                )
            )
        # Create failure message.
        body = self.on_failure
        negation = ast.UnaryOp(ast.Not(), top_condition)
        self.statements.append(ast.If(negation, body, []))
        if assert_.msg:
            assertmsg = self.helper("_format_assertmsg", assert_.msg)
            explanation = "\n>assert " + explanation
        else:
            assertmsg = ast.Str("")
            explanation = "assert " + explanation
        template = ast.BinOp(assertmsg, ast.Add(), ast.Str(explanation))
        msg = self.pop_format_context(template)
        fmt = self.helper("_format_explanation", msg)
        err_name = ast.Name("AssertionError", ast.Load())
        exc = ast_Call(err_name, [fmt], [])
        if sys.version_info[0] >= 3:
            raise_ = ast.Raise(exc, None)
        else:
            raise_ = ast.Raise(exc, None, None)
        body.append(raise_)
        # Clear temporary variables by setting them to None.
        if self.variables:
            variables = [ast.Name(name, ast.Store()) for name in self.variables]
            clear = ast.Assign(variables, _NameConstant(None))
            self.statements.append(clear)
        # Fix line numbers.
        for stmt in self.statements:
            set_location(stmt, assert_.lineno, assert_.col_offset)
        return self.statements
```
### 49 - src/_pytest/assertion/rewrite.py:

Start line: 466, End line: 490

```python
def rewrite_asserts(mod, module_path=None, config=None):
    """Rewrite the assert statements in mod."""
    AssertionRewriter(module_path, config).run(mod)


def _saferepr(obj):
    """Get a safe repr of an object for assertion error messages.

    The assertion formatting (util.format_explanation()) requires
    newlines to be escaped since they are a special character for it.
    Normally assertion.util.format_explanation() does this but for a
    custom repr it is possible to contain one of the special escape
    sequences, especially '\n{' and '\n}' are likely to be present in
    JSON reprs.

    """
    r = saferepr(obj)
    # only occurs in python2.x, repr must return text in python3+
    if isinstance(r, bytes):
        # Represent unprintable bytes as `\x##`
        r = u"".join(
            u"\\x{:x}".format(ord(c)) if c not in string.printable else c.decode()
            for c in r
        )
    return r.replace(u"\n", u"\\n")
```
### 67 - src/_pytest/assertion/rewrite.py:

Start line: 1048, End line: 1085

```python
class AssertionRewriter(ast.NodeVisitor):

    def visit_Compare(self, comp):
        self.push_format_context()
        left_res, left_expl = self.visit(comp.left)
        if isinstance(comp.left, (ast.Compare, ast.BoolOp)):
            left_expl = "({})".format(left_expl)
        res_variables = [self.variable() for i in range(len(comp.ops))]
        load_names = [ast.Name(v, ast.Load()) for v in res_variables]
        store_names = [ast.Name(v, ast.Store()) for v in res_variables]
        it = zip(range(len(comp.ops)), comp.ops, comp.comparators)
        expls = []
        syms = []
        results = [left_res]
        for i, op, next_operand in it:
            next_res, next_expl = self.visit(next_operand)
            if isinstance(next_operand, (ast.Compare, ast.BoolOp)):
                next_expl = "({})".format(next_expl)
            results.append(next_res)
            sym = binop_map[op.__class__]
            syms.append(ast.Str(sym))
            expl = "%s %s %s" % (left_expl, sym, next_expl)
            expls.append(ast.Str(expl))
            res_expr = ast.Compare(left_res, [op], [next_res])
            self.statements.append(ast.Assign([store_names[i]], res_expr))
            left_res, left_expl = next_res, next_expl
        # Use pytest.assertion.util._reprcompare if that's available.
        expl_call = self.helper(
            "_call_reprcompare",
            ast.Tuple(syms, ast.Load()),
            ast.Tuple(load_names, ast.Load()),
            ast.Tuple(expls, ast.Load()),
            ast.Tuple(results, ast.Load()),
        )
        if len(comp.ops) > 1:
            res = ast.BoolOp(ast.And(), load_names)
        else:
            res = load_names[0]
        return res, self.explanation_param(self.pop_format_context(expl_call))
```
### 80 - src/_pytest/assertion/rewrite.py:

Start line: 343, End line: 371

```python
def _write_pyc(state, co, source_stat, pyc):
    # Technically, we don't have to have the same pyc format as
    # (C)Python, since these "pycs" should never be seen by builtin
    # import. However, there's little reason deviate, and I hope
    # sometime to be able to use imp.load_compiled to load them. (See
    # the comment in load_module above.)
    try:
        with atomicwrites.atomic_write(pyc, mode="wb", overwrite=True) as fp:
            fp.write(imp.get_magic())
            # as of now, bytecode header expects 32-bit numbers for size and mtime (#4903)
            mtime = int(source_stat.mtime) & 0xFFFFFFFF
            size = source_stat.size & 0xFFFFFFFF
            # "<LL" stands for 2 unsigned longs, little-ending
            fp.write(struct.pack("<LL", mtime, size))
            fp.write(marshal.dumps(co))
    except EnvironmentError as e:
        state.trace("error writing pyc file at %s: errno=%s" % (pyc, e.errno))
        # we ignore any failure to write the cache file
        # there are many reasons, permission-denied, __pycache__ being a
        # file etc.
        return False
    return True


RN = "\r\n".encode("utf-8")
N = "\n".encode("utf-8")

cookie_re = re.compile(r"^[ \t\f]*#.*coding[:=][ \t]*[-\w.]+")
BOM_UTF8 = "\xef\xbb\xbf"
```
### 86 - src/_pytest/assertion/rewrite.py:

Start line: 953, End line: 965

```python
class AssertionRewriter(ast.NodeVisitor):

    def visit_UnaryOp(self, unary):
        pattern = unary_map[unary.op.__class__]
        operand_res, operand_expl = self.visit(unary.operand)
        res = self.assign(ast.UnaryOp(unary.op, operand_res))
        return res, pattern % (operand_expl,)

    def visit_BinOp(self, binop):
        symbol = binop_map[binop.op.__class__]
        left_expr, left_expl = self.visit(binop.left)
        right_expr, right_expl = self.visit(binop.right)
        explanation = "(%s %s %s)" % (left_expl, symbol, right_expl)
        res = self.assign(ast.BinOp(left_expr, binop.op, right_expr))
        return res, explanation
```
### 101 - src/_pytest/assertion/rewrite.py:

Start line: 918, End line: 951

```python
class AssertionRewriter(ast.NodeVisitor):

    def visit_BoolOp(self, boolop):
        res_var = self.variable()
        expl_list = self.assign(ast.List([], ast.Load()))
        app = ast.Attribute(expl_list, "append", ast.Load())
        is_or = int(isinstance(boolop.op, ast.Or))
        body = save = self.statements
        fail_save = self.on_failure
        levels = len(boolop.values) - 1
        self.push_format_context()
        # Process each operand, short-circuting if needed.
        for i, v in enumerate(boolop.values):
            if i:
                fail_inner = []
                # cond is set in a prior loop iteration below
                self.on_failure.append(ast.If(cond, fail_inner, []))  # noqa
                self.on_failure = fail_inner
            self.push_format_context()
            res, expl = self.visit(v)
            body.append(ast.Assign([ast.Name(res_var, ast.Store())], res))
            expl_format = self.pop_format_context(ast.Str(expl))
            call = ast_Call(app, [expl_format], [])
            self.on_failure.append(ast.Expr(call))
            if i < levels:
                cond = res
                if is_or:
                    cond = ast.UnaryOp(ast.Not(), cond)
                inner = []
                self.statements.append(ast.If(cond, inner, []))
                self.statements = body = inner
        self.statements = save
        self.on_failure = fail_save
        expl_template = self.helper("_format_boolop", expl_list, ast.Num(is_or))
        expl = self.pop_format_context(expl_template)
        return ast.Name(res_var, ast.Load()), self.explanation_param(expl)
```
### 105 - src/_pytest/assertion/rewrite.py:

Start line: 730, End line: 760

```python
class AssertionRewriter(ast.NodeVisitor):

    @staticmethod
    def is_rewrite_disabled(docstring):
        return "PYTEST_DONT_REWRITE" in docstring

    def variable(self):
        """Get a new variable."""
        # Use a character invalid in python identifiers to avoid clashing.
        name = "@py_assert" + str(next(self.variable_counter))
        self.variables.append(name)
        return name

    def assign(self, expr):
        """Give *expr* a name."""
        name = self.variable()
        self.statements.append(ast.Assign([ast.Name(name, ast.Store())], expr))
        return ast.Name(name, ast.Load())

    def display(self, expr):
        """Call saferepr on the expression."""
        return self.helper("_saferepr", expr)

    def helper(self, name, *args):
        """Call a helper in this module."""
        py_name = ast.Name("@pytest_ar", ast.Load())
        attr = ast.Attribute(py_name, name, ast.Load())
        return ast_Call(attr, list(args), [])

    def builtin(self, name):
        """Return the builtin called *name*."""
        builtin_name = ast.Name("@py_builtins", ast.Load())
        return ast.Attribute(builtin_name, name, ast.Load())
```
### 129 - src/_pytest/assertion/rewrite.py:

Start line: 1030, End line: 1046

```python
class AssertionRewriter(ast.NodeVisitor):

    # ast.Call signature changed on 3.5,
    # conditionally change  which methods is named
    # visit_Call depending on Python version
    if sys.version_info >= (3, 5):
        visit_Call = visit_Call_35
    else:
        visit_Call = visit_Call_legacy

    def visit_Attribute(self, attr):
        if not isinstance(attr.ctx, ast.Load):
            return self.generic_visit(attr)
        value, value_expl = self.visit(attr.value)
        res = self.assign(ast.Attribute(value, attr.attr, ast.Load()))
        res_expl = self.explanation_param(self.display(res))
        pat = "%s\n{%s = %s.%s\n}"
        expl = pat % (res_expl, res_expl, value_expl, attr.attr)
        return res, expl
```
### 131 - src/_pytest/assertion/rewrite.py:

Start line: 608, End line: 663

```python
class AssertionRewriter(ast.NodeVisitor):
    """Assertion rewriting implementation.

    The main entrypoint is to call .run() with an ast.Module instance,
    this will then find all the assert statements and rewrite them to
    provide intermediate values and a detailed assertion error.  See
    http://pybites.blogspot.be/2011/07/behind-scenes-of-pytests-new-assertion.html
    for an overview of how this works.

    The entry point here is .run() which will iterate over all the
    statements in an ast.Module and for each ast.Assert statement it
    finds call .visit() with it.  Then .visit_Assert() takes over and
    is responsible for creating new ast statements to replace the
    original assert statement: it rewrites the test of an assertion
    to provide intermediate values and replace it with an if statement
    which raises an assertion error with a detailed explanation in
    case the expression is false.

    For this .visit_Assert() uses the visitor pattern to visit all the
    AST nodes of the ast.Assert.test field, each visit call returning
    an AST node and the corresponding explanation string.  During this
    state is kept in several instance attributes:

    :statements: All the AST statements which will replace the assert
       statement.

    :variables: This is populated by .variable() with each variable
       used by the statements so that they can all be set to None at
       the end of the statements.

    :variable_counter: Counter to create new unique variables needed
       by statements.  Variables are created using .variable() and
       have the form of "@py_assert0".

    :on_failure: The AST statements which will be executed if the
       assertion test fails.  This is the code which will construct
       the failure message and raises the AssertionError.

    :explanation_specifiers: A dict filled by .explanation_param()
       with %-formatting placeholders and their corresponding
       expressions to use in the building of an assertion message.
       This is used by .pop_format_context() to build a message.

    :stack: A stack of the explanation_specifiers dicts maintained by
       .push_format_context() and .pop_format_context() which allows
       to build another %-formatted string while already building one.

    This state is reset on every new assert statement visited and used
    by the other visitors.

    """

    def __init__(self, module_path, config):
        super(AssertionRewriter, self).__init__()
        self.module_path = module_path
        self.config = config
```
### 132 - src/_pytest/assertion/rewrite.py:

Start line: 967, End line: 998

```python
class AssertionRewriter(ast.NodeVisitor):

    def visit_Call_35(self, call):
        """
        visit `ast.Call` nodes on Python3.5 and after
        """
        new_func, func_expl = self.visit(call.func)
        arg_expls = []
        new_args = []
        new_kwargs = []
        for arg in call.args:
            res, expl = self.visit(arg)
            arg_expls.append(expl)
            new_args.append(res)
        for keyword in call.keywords:
            res, expl = self.visit(keyword.value)
            new_kwargs.append(ast.keyword(keyword.arg, res))
            if keyword.arg:
                arg_expls.append(keyword.arg + "=" + expl)
            else:  # **args have `arg` keywords with an .arg of None
                arg_expls.append("**" + expl)

        expl = "%s(%s)" % (func_expl, ", ".join(arg_expls))
        new_call = ast.Call(new_func, new_args, new_kwargs)
        res = self.assign(new_call)
        res_expl = self.explanation_param(self.display(res))
        outer_expl = "%s\n{%s = %s\n}" % (res_expl, res_expl, expl)
        return res, outer_expl

    def visit_Starred(self, starred):
        # From Python 3.5, a Starred node can appear in a function call
        res, expl = self.visit(starred.value)
        new_starred = ast.Starred(res, starred.ctx)
        return new_starred, "*" + expl
```
### 141 - src/_pytest/assertion/rewrite.py:

Start line: 374, End line: 427

```python
def _rewrite_test(config, fn):
    """Try to read and rewrite *fn* and return the code object."""
    state = config._assertstate
    try:
        stat = fn.stat()
        source = fn.read("rb")
    except EnvironmentError:
        return None, None
    if ASCII_IS_DEFAULT_ENCODING:
        # ASCII is the default encoding in Python 2. Without a coding
        # declaration, Python 2 will complain about any bytes in the file
        # outside the ASCII range. Sadly, this behavior does not extend to
        # compile() or ast.parse(), which prefer to interpret the bytes as
        # latin-1. (At least they properly handle explicit coding cookies.) To
        # preserve this error behavior, we could force ast.parse() to use ASCII
        # as the encoding by inserting a coding cookie. Unfortunately, that
        # messes up line numbers. Thus, we have to check ourselves if anything
        # is outside the ASCII range in the case no encoding is explicitly
        # declared. For more context, see issue #269. Yay for Python 3 which
        # gets this right.
        end1 = source.find("\n")
        end2 = source.find("\n", end1 + 1)
        if (
            not source.startswith(BOM_UTF8)
            and cookie_re.match(source[0:end1]) is None
            and cookie_re.match(source[end1 + 1 : end2]) is None
        ):
            if hasattr(state, "_indecode"):
                # encodings imported us again, so don't rewrite.
                return None, None
            state._indecode = True
            try:
                try:
                    source.decode("ascii")
                except UnicodeDecodeError:
                    # Let it fail in real import.
                    return None, None
            finally:
                del state._indecode
    try:
        tree = ast.parse(source, filename=fn.strpath)
    except SyntaxError:
        # Let this pop up again in the real import.
        state.trace("failed to parse: %r" % (fn,))
        return None, None
    rewrite_asserts(tree, fn, config)
    try:
        co = compile(tree, fn.strpath, "exec", dont_inherit=True)
    except SyntaxError:
        # It's possible that this error is from some bug in the
        # assertion rewriting, but I don't know of a fast way to tell.
        state.trace("failed to compile: %r" % (fn,))
        return None, None
    return stat, co
```
### 161 - src/_pytest/assertion/rewrite.py:

Start line: 1000, End line: 1028

```python
class AssertionRewriter(ast.NodeVisitor):

    def visit_Call_legacy(self, call):
        """
        visit `ast.Call nodes on 3.4 and below`
        """
        new_func, func_expl = self.visit(call.func)
        arg_expls = []
        new_args = []
        new_kwargs = []
        new_star = new_kwarg = None
        for arg in call.args:
            res, expl = self.visit(arg)
            new_args.append(res)
            arg_expls.append(expl)
        for keyword in call.keywords:
            res, expl = self.visit(keyword.value)
            new_kwargs.append(ast.keyword(keyword.arg, res))
            arg_expls.append(keyword.arg + "=" + expl)
        if call.starargs:
            new_star, expl = self.visit(call.starargs)
            arg_expls.append("*" + expl)
        if call.kwargs:
            new_kwarg, expl = self.visit(call.kwargs)
            arg_expls.append("**" + expl)
        expl = "%s(%s)" % (func_expl, ", ".join(arg_expls))
        new_call = ast.Call(new_func, new_args, new_kwargs, new_star, new_kwarg)
        res = self.assign(new_call)
        res_expl = self.explanation_param(self.display(res))
        outer_expl = "%s\n{%s = %s\n}" % (res_expl, res_expl, expl)
        return res, outer_expl
```
### 169 - src/_pytest/assertion/rewrite.py:

Start line: 908, End line: 916

```python
class AssertionRewriter(ast.NodeVisitor):

    def visit_Name(self, name):
        # Display the repr of the name if it's a local variable or
        # _should_repr_global_name() thinks it's acceptable.
        locs = ast_Call(self.builtin("locals"), [], [])
        inlocs = ast.Compare(ast.Str(name.id), [ast.In()], [locs])
        dorepr = self.helper("_should_repr_global_name", name)
        test = ast.BoolOp(ast.Or(), [inlocs, dorepr])
        expr = ast.IfExp(test, self.display(name), ast.Str(name.id))
        return name, self.explanation_param(expr)
```
### 170 - src/_pytest/assertion/rewrite.py:

Start line: 430, End line: 463

```python
def _read_pyc(source, pyc, trace=lambda x: None):
    """Possibly read a pytest pyc containing rewritten code.

    Return rewritten code if successful or None if not.
    """
    try:
        fp = open(pyc, "rb")
    except IOError:
        return None
    with fp:
        try:
            mtime = int(source.mtime())
            size = source.size()
            data = fp.read(12)
        except EnvironmentError as e:
            trace("_read_pyc(%s): EnvironmentError %s" % (source, e))
            return None
        # Check for invalid or out of date pyc file.
        if (
            len(data) != 12
            or data[:4] != imp.get_magic()
            or struct.unpack("<LL", data[4:]) != (mtime & 0xFFFFFFFF, size & 0xFFFFFFFF)
        ):
            trace("_read_pyc(%s): invalid or out of date pyc" % source)
            return None
        try:
            co = marshal.load(fp)
        except Exception as e:
            trace("_read_pyc(%s): marshal.load error %s" % (source, e))
            return None
        if not isinstance(co, types.CodeType):
            trace("_read_pyc(%s): not a code object" % source)
            return None
        return co
```
### 192 - src/_pytest/assertion/rewrite.py:

Start line: 493, End line: 516

```python
def _format_assertmsg(obj):
    """Format the custom assertion message given.

    For strings this simply replaces newlines with '\n~' so that
    util.format_explanation() will preserve them instead of escaping
    newlines.  For other objects saferepr() is used first.

    """
    # reprlib appears to have a bug which means that if a string
    # contains a newline it gets escaped, however if an object has a
    # .__repr__() which contains newlines it does not get escaped.
    # However in either case we want to preserve the newline.
    replaces = [(u"\n", u"\n~"), (u"%", u"%%")]
    if not isinstance(obj, six.string_types):
        obj = saferepr(obj)
        replaces.append((u"\\n", u"\n~"))

    if isinstance(obj, bytes):
        replaces = [(r1.encode(), r2.encode()) for r1, r2 in replaces]

    for r1, r2 in replaces:
        obj = obj.replace(r1, r2)

    return obj
```
### 196 - src/_pytest/assertion/rewrite.py:

Start line: 84, End line: 163

```python
class AssertionRewritingHook(object):

    def find_module(self, name, path=None):
        if self._writing_pyc:
            return None
        state = self.config._assertstate
        if self._early_rewrite_bailout(name, state):
            return None
        state.trace("find_module called for: %s" % name)
        names = name.rsplit(".", 1)
        lastname = names[-1]
        pth = None
        if path is not None:
            # Starting with Python 3.3, path is a _NamespacePath(), which
            # causes problems if not converted to list.
            path = list(path)
            if len(path) == 1:
                pth = path[0]
        if pth is None:
            try:
                fd, fn, desc = self._imp_find_module(lastname, path)
            except ImportError:
                return None
            if fd is not None:
                fd.close()
            tp = desc[2]
            if tp == imp.PY_COMPILED:
                if hasattr(imp, "source_from_cache"):
                    try:
                        fn = imp.source_from_cache(fn)
                    except ValueError:
                        # Python 3 doesn't like orphaned but still-importable
                        # .pyc files.
                        fn = fn[:-1]
                else:
                    fn = fn[:-1]
            elif tp != imp.PY_SOURCE:
                # Don't know what this is.
                return None
        else:
            fn = os.path.join(pth, name.rpartition(".")[2] + ".py")

        fn_pypath = py.path.local(fn)
        if not self._should_rewrite(name, fn_pypath, state):
            return None

        self._rewritten_names.add(name)

        # The requested module looks like a test file, so rewrite it. This is
        # the most magical part of the process: load the source, rewrite the
        # asserts, and load the rewritten source. We also cache the rewritten
        # module code in a special pyc. We must be aware of the possibility of
        # concurrent pytest processes rewriting and loading pycs. To avoid
        # tricky race conditions, we maintain the following invariant: The
        # cached pyc is always a complete, valid pyc. Operations on it must be
        # atomic. POSIX's atomic rename comes in handy.
        write = not sys.dont_write_bytecode
        cache_dir = os.path.join(fn_pypath.dirname, "__pycache__")
        if write:
            try:
                os.mkdir(cache_dir)
            except OSError:
                e = sys.exc_info()[1].errno
                if e == errno.EEXIST:
                    # Either the __pycache__ directory already exists (the
                    # common case) or it's blocked by a non-dir node. In the
                    # latter case, we'll ignore it in _write_pyc.
                    pass
                elif e in [errno.ENOENT, errno.ENOTDIR]:
                    # One of the path components was not a directory, likely
                    # because we're in a zip file.
                    write = False
                elif e in [errno.EACCES, errno.EROFS, errno.EPERM]:
                    state.trace("read only directory: %r" % fn_pypath.dirname)
                    write = False
                else:
                    raise
        cache_name = fn_pypath.basename[:-3] + PYC_TAIL
        pyc = os.path.join(cache_dir, cache_name)
        # Notice that even if we're in a read-only directory, I'm going
        # to check for a cached pyc. This may not be optimal...
        co = _read_pyc(fn_pypath, pyc, state.trace)
        # ... other code
```
### 228 - src/_pytest/assertion/rewrite.py:

Start line: 308, End line: 340

```python
class AssertionRewritingHook(object):

    def is_package(self, name):
        try:
            fd, fn, desc = self._imp_find_module(name)
        except ImportError:
            return False
        if fd is not None:
            fd.close()
        tp = desc[2]
        return tp == imp.PKG_DIRECTORY

    @classmethod
    def _register_with_pkg_resources(cls):
        """
        Ensure package resources can be loaded from this loader. May be called
        multiple times, as the operation is idempotent.
        """
        try:
            import pkg_resources

            # access an attribute in case a deferred importer is present
            pkg_resources.__name__
        except ImportError:
            return

        # Since pytest tests are always located in the file system, the
        #  DefaultProvider is appropriate.
        pkg_resources.register_loader_type(cls, pkg_resources.DefaultProvider)

    def get_data(self, pathname):
        """Optional PEP302 get_data API.
        """
        with open(pathname, "rb") as f:
            return f.read()
```
### 246 - src/_pytest/assertion/rewrite.py:

Start line: 519, End line: 605

```python
def _should_repr_global_name(obj):
    if callable(obj):
        return False

    try:
        return not hasattr(obj, "__name__")
    except Exception:
        return True


def _format_boolop(explanations, is_or):
    explanation = "(" + (is_or and " or " or " and ").join(explanations) + ")"
    if isinstance(explanation, six.text_type):
        return explanation.replace(u"%", u"%%")
    else:
        return explanation.replace(b"%", b"%%")


def _call_reprcompare(ops, results, expls, each_obj):
    for i, res, expl in zip(range(len(ops)), results, expls):
        try:
            done = not res
        except Exception:
            done = True
        if done:
            break
    if util._reprcompare is not None:
        custom = util._reprcompare(ops[i], each_obj[i], each_obj[i + 1])
        if custom is not None:
            return custom
    return expl


unary_map = {ast.Not: "not %s", ast.Invert: "~%s", ast.USub: "-%s", ast.UAdd: "+%s"}

binop_map = {
    ast.BitOr: "|",
    ast.BitXor: "^",
    ast.BitAnd: "&",
    ast.LShift: "<<",
    ast.RShift: ">>",
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
    ast.FloorDiv: "//",
    ast.Mod: "%%",  # escaped for string formatting
    ast.Eq: "==",
    ast.NotEq: "!=",
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Gt: ">",
    ast.GtE: ">=",
    ast.Pow: "**",
    ast.Is: "is",
    ast.IsNot: "is not",
    ast.In: "in",
    ast.NotIn: "not in",
}
# Python 3.5+ compatibility
try:
    binop_map[ast.MatMult] = "@"
except AttributeError:
    pass

# Python 3.4+ compatibility
if hasattr(ast, "NameConstant"):
    _NameConstant = ast.NameConstant
else:

    def _NameConstant(c):
        return ast.Name(str(c), ast.Load())


def set_location(node, lineno, col_offset):
    """Set node location information recursively."""

    def _fix(node, lineno, col_offset):
        if "lineno" in node._attributes:
            node.lineno = lineno
        if "col_offset" in node._attributes:
            node.col_offset = col_offset
        for child in ast.iter_child_nodes(node):
            _fix(child, lineno, col_offset)

    _fix(node, lineno, col_offset)
    return node
```
