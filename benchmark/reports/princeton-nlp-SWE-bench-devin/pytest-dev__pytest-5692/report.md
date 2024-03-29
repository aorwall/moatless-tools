# pytest-dev__pytest-5692

| **pytest-dev/pytest** | `29e336bd9bf87eaef8e2683196ee1975f1ad4088` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 3453 |
| **Any found context length** | 1101 |
| **Avg pos** | 16.0 |
| **Min pos** | 3 |
| **Max pos** | 13 |
| **Top file pos** | 1 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/src/_pytest/junitxml.py b/src/_pytest/junitxml.py
--- a/src/_pytest/junitxml.py
+++ b/src/_pytest/junitxml.py
@@ -10,9 +10,11 @@
 """
 import functools
 import os
+import platform
 import re
 import sys
 import time
+from datetime import datetime
 
 import py
 
@@ -666,6 +668,8 @@ def pytest_sessionfinish(self):
             skipped=self.stats["skipped"],
             tests=numtests,
             time="%.3f" % suite_time_delta,
+            timestamp=datetime.fromtimestamp(self.suite_start_time).isoformat(),
+            hostname=platform.node(),
         )
         logfile.write(Junit.testsuites([suite_node]).unicode(indent=0))
         logfile.close()

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| src/_pytest/junitxml.py | 13 | 13 | 3 | 1 | 1101
| src/_pytest/junitxml.py | 669 | 669 | 13 | 1 | 3453


## Problem Statement

```
Hostname and timestamp properties in generated JUnit XML reports
Pytest enables generating JUnit XML reports of the tests.

However, there are some properties missing, specifically `hostname` and `timestamp` from the `testsuite` XML element. Is there an option to include them?

Example of a pytest XML report:
\`\`\`xml
<?xml version="1.0" encoding="utf-8"?>
<testsuite errors="0" failures="2" name="check" skipped="0" tests="4" time="0.049">
	<testcase classname="test_sample.TestClass" file="test_sample.py" line="3" name="test_addOne_normal" time="0.001"></testcase>
	<testcase classname="test_sample.TestClass" file="test_sample.py" line="6" name="test_addOne_edge" time="0.001"></testcase>
</testsuite>
\`\`\`

Example of a junit XML report:
\`\`\`xml
<?xml version="1.0" encoding="UTF-8"?>
<testsuite name="location.GeoLocationTest" tests="2" skipped="0" failures="0" errors="0" timestamp="2019-04-22T10:32:27" hostname="Anass-MacBook-Pro.local" time="0.048">
  <properties/>
  <testcase name="testIoException()" classname="location.GeoLocationTest" time="0.044"/>
  <testcase name="testJsonDeserialization()" classname="location.GeoLocationTest" time="0.003"/>
  <system-out><![CDATA[]]></system-out>
  <system-err><![CDATA[]]></system-err>
</testsuite>
\`\`\`

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 src/_pytest/junitxml.py** | 116 | 157| 339 | 339 | 5027 | 
| 2 | **1 src/_pytest/junitxml.py** | 673 | 692| 146 | 485 | 5027 | 
| **-> 3 <-** | **1 src/_pytest/junitxml.py** | 1 | 79| 616 | 1101 | 5027 | 
| 4 | **1 src/_pytest/junitxml.py** | 506 | 531| 178 | 1279 | 5027 | 
| 5 | **1 src/_pytest/junitxml.py** | 381 | 425| 334 | 1613 | 5027 | 
| 6 | **1 src/_pytest/junitxml.py** | 82 | 114| 207 | 1820 | 5027 | 
| 7 | **1 src/_pytest/junitxml.py** | 159 | 210| 333 | 2153 | 5027 | 
| 8 | **1 src/_pytest/junitxml.py** | 212 | 229| 161 | 2314 | 5027 | 
| 9 | **1 src/_pytest/junitxml.py** | 351 | 378| 247 | 2561 | 5027 | 
| 10 | **1 src/_pytest/junitxml.py** | 466 | 504| 310 | 2871 | 5027 | 
| 11 | **1 src/_pytest/junitxml.py** | 247 | 275| 217 | 3088 | 5027 | 
| 12 | **1 src/_pytest/junitxml.py** | 293 | 311| 124 | 3212 | 5027 | 
| **-> 13 <-** | **1 src/_pytest/junitxml.py** | 643 | 671| 241 | 3453 | 5027 | 
| 14 | **1 src/_pytest/junitxml.py** | 533 | 617| 625 | 4078 | 5027 | 
| 15 | **1 src/_pytest/junitxml.py** | 231 | 245| 132 | 4210 | 5027 | 
| 16 | 2 src/_pytest/reports.py | 31 | 124| 572 | 4782 | 7995 | 
| 17 | **2 src/_pytest/junitxml.py** | 619 | 641| 204 | 4986 | 7995 | 
| 18 | **2 src/_pytest/junitxml.py** | 428 | 441| 119 | 5105 | 7995 | 
| 19 | 2 src/_pytest/reports.py | 187 | 202| 156 | 5261 | 7995 | 
| 20 | 2 src/_pytest/reports.py | 277 | 336| 420 | 5681 | 7995 | 
| 21 | **2 src/_pytest/junitxml.py** | 314 | 348| 258 | 5939 | 7995 | 
| 22 | 3 testing/python/collect.py | 1017 | 1032| 133 | 6072 | 17604 | 
| 23 | 3 src/_pytest/reports.py | 382 | 400| 143 | 6215 | 17604 | 
| 24 | 3 testing/python/collect.py | 1034 | 1053| 167 | 6382 | 17604 | 
| 25 | 4 src/_pytest/terminal.py | 385 | 434| 420 | 6802 | 25766 | 
| 26 | 5 src/_pytest/skipping.py | 120 | 178| 533 | 7335 | 27244 | 
| 27 | 5 src/_pytest/reports.py | 1 | 28| 222 | 7557 | 27244 | 
| 28 | 5 testing/python/collect.py | 1055 | 1073| 154 | 7711 | 27244 | 
| 29 | 5 src/_pytest/terminal.py | 921 | 946| 199 | 7910 | 27244 | 
| 30 | 5 src/_pytest/terminal.py | 872 | 902| 250 | 8160 | 27244 | 
| 31 | 6 src/_pytest/python.py | 1209 | 1235| 208 | 8368 | 38706 | 
| 32 | 7 src/pytest.py | 1 | 107| 708 | 9076 | 39414 | 
| 33 | 8 src/_pytest/helpconfig.py | 209 | 245| 252 | 9328 | 41103 | 
| 34 | 8 src/_pytest/terminal.py | 255 | 279| 161 | 9489 | 41103 | 
| 35 | 8 src/_pytest/terminal.py | 904 | 919| 187 | 9676 | 41103 | 
| 36 | 9 src/_pytest/runner.py | 151 | 193| 338 | 10014 | 43831 | 
| 37 | 9 src/_pytest/terminal.py | 611 | 644| 331 | 10345 | 43831 | 
| 38 | 9 src/_pytest/python.py | 1237 | 1264| 217 | 10562 | 43831 | 
| 39 | 9 src/_pytest/runner.py | 1 | 30| 165 | 10727 | 43831 | 
| 40 | 9 src/_pytest/runner.py | 33 | 59| 232 | 10959 | 43831 | 
| 41 | 9 src/_pytest/python.py | 55 | 107| 354 | 11313 | 43831 | 
| 42 | 9 src/_pytest/terminal.py | 784 | 808| 203 | 11516 | 43831 | 
| 43 | 9 src/_pytest/reports.py | 126 | 152| 158 | 11674 | 43831 | 
| 44 | 10 src/_pytest/unittest.py | 242 | 283| 286 | 11960 | 45828 | 
| 45 | 10 src/_pytest/terminal.py | 810 | 831| 180 | 12140 | 45828 | 
| 46 | 11 src/_pytest/pastebin.py | 78 | 99| 181 | 12321 | 46593 | 
| 47 | 11 src/_pytest/reports.py | 154 | 185| 237 | 12558 | 46593 | 
| 48 | 11 src/_pytest/python.py | 1267 | 1336| 487 | 13045 | 46593 | 
| 49 | 11 src/_pytest/runner.py | 242 | 270| 249 | 13294 | 46593 | 
| 50 | 12 src/_pytest/faulthandler.py | 1 | 36| 249 | 13543 | 47168 | 
| 51 | 13 src/_pytest/doctest.py | 219 | 275| 495 | 14038 | 51668 | 
| 52 | 13 src/_pytest/terminal.py | 436 | 455| 198 | 14236 | 51668 | 
| 53 | 13 src/_pytest/python.py | 284 | 298| 141 | 14377 | 51668 | 
| 54 | 13 src/_pytest/terminal.py | 833 | 846| 125 | 14502 | 51668 | 
| 55 | 13 src/_pytest/doctest.py | 37 | 84| 321 | 14823 | 51668 | 
| 56 | 13 src/_pytest/terminal.py | 49 | 136| 630 | 15453 | 51668 | 
| 57 | 13 src/_pytest/reports.py | 265 | 274| 134 | 15587 | 51668 | 
| 58 | 14 testing/python/fixtures.py | 1 | 42| 238 | 15825 | 76589 | 
| 59 | **14 src/_pytest/junitxml.py** | 444 | 463| 143 | 15968 | 76589 | 
| 60 | 15 src/_pytest/resultlog.py | 62 | 79| 144 | 16112 | 77310 | 
| 61 | 15 src/_pytest/terminal.py | 721 | 744| 142 | 16254 | 77310 | 
| 62 | 15 src/_pytest/terminal.py | 848 | 870| 208 | 16462 | 77310 | 
| 63 | 16 doc/en/conf.py | 118 | 240| 811 | 17273 | 79769 | 
| 64 | 16 doc/en/conf.py | 1 | 116| 789 | 18062 | 79769 | 
| 65 | 16 src/_pytest/unittest.py | 106 | 155| 349 | 18411 | 79769 | 
| 66 | 16 src/_pytest/terminal.py | 505 | 543| 323 | 18734 | 79769 | 
| 67 | 16 src/_pytest/helpconfig.py | 139 | 206| 550 | 19284 | 79769 | 
| 68 | 17 extra/get_issues.py | 55 | 86| 231 | 19515 | 80312 | 
| 69 | 17 testing/python/collect.py | 196 | 249| 320 | 19835 | 80312 | 
| 70 | 17 src/_pytest/pastebin.py | 1 | 38| 268 | 20103 | 80312 | 
| 71 | 17 src/_pytest/unittest.py | 223 | 239| 138 | 20241 | 80312 | 
| 72 | 17 src/_pytest/terminal.py | 667 | 683| 122 | 20363 | 80312 | 
| 73 | 17 testing/python/collect.py | 279 | 344| 471 | 20834 | 80312 | 
| 74 | 17 src/_pytest/unittest.py | 80 | 103| 157 | 20991 | 80312 | 
| 75 | 17 src/_pytest/terminal.py | 594 | 609| 128 | 21119 | 80312 | 
| 76 | 17 src/_pytest/doctest.py | 180 | 217| 303 | 21422 | 80312 | 
| 77 | 17 doc/en/conf.py | 241 | 341| 656 | 22078 | 80312 | 
| 78 | 17 src/_pytest/terminal.py | 699 | 719| 191 | 22269 | 80312 | 
| 79 | 18 src/_pytest/fixtures.py | 1237 | 1269| 246 | 22515 | 91285 | 
| 80 | 18 src/_pytest/fixtures.py | 1 | 105| 672 | 23187 | 91285 | 
| 81 | 18 testing/python/collect.py | 559 | 658| 662 | 23849 | 91285 | 
| 82 | 19 src/_pytest/main.py | 151 | 177| 175 | 24024 | 96526 | 
| 83 | 19 src/_pytest/terminal.py | 335 | 348| 123 | 24147 | 96526 | 
| 84 | 19 src/_pytest/terminal.py | 1009 | 1058| 347 | 24494 | 96526 | 
| 85 | 20 doc/en/example/xfail_demo.py | 1 | 39| 143 | 24637 | 96670 | 
| 86 | 20 src/_pytest/terminal.py | 217 | 240| 203 | 24840 | 96670 | 
| 87 | 20 src/_pytest/python.py | 1187 | 1206| 183 | 25023 | 96670 | 
| 88 | 20 src/_pytest/fixtures.py | 1159 | 1187| 249 | 25272 | 96670 | 
| 89 | 20 src/_pytest/python.py | 349 | 376| 249 | 25521 | 96670 | 
| 90 | 20 src/_pytest/terminal.py | 572 | 592| 187 | 25708 | 96670 | 
| 91 | 20 src/_pytest/terminal.py | 545 | 570| 258 | 25966 | 96670 | 
| 92 | 20 src/_pytest/unittest.py | 29 | 77| 397 | 26363 | 96670 | 
| 93 | 20 src/_pytest/main.py | 651 | 700| 425 | 26788 | 96670 | 
| 94 | 20 src/_pytest/terminal.py | 139 | 166| 216 | 27004 | 96670 | 
| 95 | 20 src/_pytest/runner.py | 97 | 110| 119 | 27123 | 96670 | 
| 96 | 21 src/_pytest/stepwise.py | 73 | 109| 290 | 27413 | 97384 | 
| 97 | **21 src/_pytest/junitxml.py** | 278 | 290| 132 | 27545 | 97384 | 
| 98 | 21 src/_pytest/terminal.py | 457 | 470| 149 | 27694 | 97384 | 
| 99 | 21 src/_pytest/resultlog.py | 1 | 35| 235 | 27929 | 97384 | 
| 100 | 22 src/_pytest/outcomes.py | 39 | 54| 124 | 28053 | 98862 | 
| 101 | 22 testing/python/fixtures.py | 2152 | 3078| 6102 | 34155 | 98862 | 
| 102 | 22 testing/python/fixtures.py | 2033 | 2150| 714 | 34869 | 98862 | 
| 103 | 23 testing/python/metafunc.py | 1295 | 1333| 243 | 35112 | 111963 | 
| 104 | 24 doc/en/example/nonpython/conftest.py | 1 | 47| 315 | 35427 | 112278 | 
| 105 | 24 src/_pytest/terminal.py | 364 | 383| 196 | 35623 | 112278 | 
| 106 | 25 src/_pytest/mark/__init__.py | 36 | 74| 341 | 35964 | 113449 | 
| 107 | 25 src/_pytest/main.py | 42 | 150| 757 | 36721 | 113449 | 
| 108 | 25 src/_pytest/doctest.py | 313 | 339| 198 | 36919 | 113449 | 
| 109 | 26 src/_pytest/cacheprovider.py | 201 | 255| 493 | 37412 | 116821 | 
| 110 | 26 src/_pytest/reports.py | 338 | 379| 311 | 37723 | 116821 | 
| 111 | 26 src/_pytest/fixtures.py | 594 | 621| 274 | 37997 | 116821 | 
| 112 | 27 src/_pytest/pytester.py | 247 | 274| 213 | 38210 | 127040 | 
| 113 | 27 src/_pytest/reports.py | 403 | 427| 171 | 38381 | 127040 | 
| 114 | 27 src/_pytest/resultlog.py | 81 | 98| 159 | 38540 | 127040 | 
| 115 | 27 src/_pytest/python.py | 110 | 128| 176 | 38716 | 127040 | 
| 116 | 27 src/_pytest/reports.py | 204 | 262| 480 | 39196 | 127040 | 
| 117 | 27 src/_pytest/fixtures.py | 854 | 883| 272 | 39468 | 127040 | 
| 118 | 27 src/_pytest/cacheprovider.py | 292 | 347| 410 | 39878 | 127040 | 
| 119 | 28 testing/python/integration.py | 37 | 68| 226 | 40104 | 129965 | 
| 120 | 29 doc/en/conftest.py | 1 | 2| 0 | 40104 | 129972 | 
| 121 | 29 src/_pytest/skipping.py | 28 | 65| 363 | 40467 | 129972 | 
| 122 | 29 src/_pytest/python.py | 301 | 330| 272 | 40739 | 129972 | 
| 123 | 29 src/_pytest/terminal.py | 294 | 315| 165 | 40904 | 129972 | 
| 124 | 30 src/_pytest/assertion/__init__.py | 1 | 32| 191 | 41095 | 131127 | 
| 125 | 30 src/_pytest/terminal.py | 746 | 782| 299 | 41394 | 131127 | 
| 126 | 30 src/_pytest/terminal.py | 281 | 292| 165 | 41559 | 131127 | 
| 127 | 30 testing/python/collect.py | 788 | 810| 195 | 41754 | 131127 | 
| 128 | 30 src/_pytest/main.py | 309 | 360| 303 | 42057 | 131127 | 
| 129 | 30 src/_pytest/fixtures.py | 1143 | 1157| 161 | 42218 | 131127 | 
| 130 | 31 src/_pytest/assertion/rewrite.py | 378 | 463| 622 | 42840 | 139912 | 
| 131 | 31 src/_pytest/unittest.py | 157 | 188| 205 | 43045 | 139912 | 
| 132 | 32 src/_pytest/logging.py | 278 | 297| 130 | 43175 | 144883 | 
| 133 | 32 src/_pytest/logging.py | 104 | 185| 534 | 43709 | 144883 | 
| 134 | 33 testing/python/approx.py | 26 | 59| 483 | 44192 | 150684 | 
| 135 | 33 src/_pytest/terminal.py | 646 | 665| 153 | 44345 | 150684 | 
| 136 | 33 src/_pytest/python.py | 192 | 228| 355 | 44700 | 150684 | 
| 137 | 33 src/_pytest/fixtures.py | 1271 | 1325| 434 | 45134 | 150684 | 
| 138 | 34 testing/example_scripts/issue_519.py | 1 | 31| 350 | 45484 | 151150 | 
| 139 | 34 src/_pytest/pytester.py | 1 | 29| 170 | 45654 | 151150 | 
| 140 | 34 src/_pytest/doctest.py | 87 | 128| 311 | 45965 | 151150 | 
| 141 | 34 src/_pytest/main.py | 447 | 491| 372 | 46337 | 151150 | 
| 142 | 35 src/_pytest/_code/code.py | 857 | 880| 172 | 46509 | 159241 | 
| 143 | 35 src/_pytest/pytester.py | 32 | 54| 144 | 46653 | 159241 | 
| 144 | 36 src/_pytest/compat.py | 140 | 178| 244 | 46897 | 161605 | 
| 145 | 36 testing/python/collect.py | 1106 | 1143| 209 | 47106 | 161605 | 
| 146 | 37 src/_pytest/setuponly.py | 49 | 85| 265 | 47371 | 162169 | 
| 147 | 37 src/_pytest/logging.py | 563 | 593| 233 | 47604 | 162169 | 
| 148 | 37 src/_pytest/fixtures.py | 384 | 473| 727 | 48331 | 162169 | 
| 149 | 37 src/_pytest/main.py | 363 | 418| 473 | 48804 | 162169 | 
| 150 | 38 src/_pytest/setupplan.py | 1 | 28| 163 | 48967 | 162333 | 
| 151 | 38 testing/python/collect.py | 755 | 786| 204 | 49171 | 162333 | 
| 152 | 38 testing/python/metafunc.py | 900 | 929| 221 | 49392 | 162333 | 
| 153 | 38 testing/python/integration.py | 364 | 381| 134 | 49526 | 162333 | 
| 154 | 38 testing/python/fixtures.py | 1026 | 2031| 6201 | 55727 | 162333 | 
| 155 | 38 src/_pytest/helpconfig.py | 39 | 83| 297 | 56024 | 162333 | 
| 156 | 39 src/_pytest/hookspec.py | 381 | 396| 132 | 56156 | 166812 | 
| 157 | 39 src/_pytest/doctest.py | 438 | 453| 111 | 56267 | 166812 | 
| 158 | 39 src/_pytest/terminal.py | 472 | 503| 279 | 56546 | 166812 | 
| 159 | 39 src/_pytest/fixtures.py | 475 | 502| 208 | 56754 | 166812 | 
| 160 | 39 src/_pytest/terminal.py | 1 | 46| 262 | 57016 | 166812 | 
| 161 | 39 src/_pytest/logging.py | 447 | 460| 129 | 57145 | 166812 | 
| 162 | 39 testing/python/fixtures.py | 45 | 1024| 6303 | 63448 | 166812 | 
| 163 | 39 testing/python/metafunc.py | 291 | 312| 229 | 63677 | 166812 | 
| 164 | 39 src/_pytest/fixtures.py | 829 | 852| 188 | 63865 | 166812 | 
| 165 | 39 src/_pytest/pytester.py | 310 | 343| 157 | 64022 | 166812 | 
| 166 | 39 src/_pytest/python.py | 1 | 52| 363 | 64385 | 166812 | 
| 167 | 39 src/_pytest/unittest.py | 205 | 220| 133 | 64518 | 166812 | 
| 168 | 39 testing/python/collect.py | 1076 | 1103| 183 | 64701 | 166812 | 
| 169 | 39 testing/python/metafunc.py | 365 | 383| 160 | 64861 | 166812 | 
| 170 | 39 testing/python/metafunc.py | 878 | 897| 129 | 64990 | 166812 | 
| 171 | 39 extra/get_issues.py | 33 | 52| 143 | 65133 | 166812 | 
| 172 | 39 src/_pytest/fixtures.py | 574 | 592| 183 | 65316 | 166812 | 
| 173 | 39 testing/python/metafunc.py | 1237 | 1269| 199 | 65515 | 166812 | 
| 174 | 40 doc/en/example/conftest.py | 1 | 2| 0 | 65515 | 166819 | 
| 175 | 40 src/_pytest/logging.py | 534 | 561| 211 | 65726 | 166819 | 
| 176 | 40 src/_pytest/cacheprovider.py | 145 | 199| 500 | 66226 | 166819 | 
| 177 | 40 src/_pytest/fixtures.py | 271 | 294| 191 | 66417 | 166819 | 
| 178 | 40 src/_pytest/logging.py | 353 | 366| 113 | 66530 | 166819 | 
| 179 | 40 src/_pytest/python.py | 131 | 148| 212 | 66742 | 166819 | 
| 180 | 40 src/_pytest/python.py | 378 | 419| 367 | 67109 | 166819 | 
| 181 | 41 bench/empty.py | 1 | 3| 0 | 67109 | 166841 | 
| 182 | 42 src/_pytest/config/__init__.py | 736 | 748| 139 | 67248 | 175380 | 
| 183 | 43 doc/en/example/py2py3/conftest.py | 1 | 17| 0 | 67248 | 175466 | 
| 184 | 43 testing/python/integration.py | 1 | 35| 239 | 67487 | 175466 | 
| 185 | 43 src/_pytest/cacheprovider.py | 394 | 432| 337 | 67824 | 175466 | 
| 186 | 43 src/_pytest/_code/code.py | 608 | 651| 307 | 68131 | 175466 | 
| 187 | 44 doc/en/example/assertion/failure_demo.py | 43 | 121| 680 | 68811 | 177125 | 
| 188 | 45 testing/conftest.py | 1 | 58| 333 | 69144 | 177459 | 
| 189 | 45 testing/python/integration.py | 325 | 362| 224 | 69368 | 177459 | 
| 190 | 45 testing/python/metafunc.py | 513 | 552| 329 | 69697 | 177459 | 
| 191 | 45 testing/python/metafunc.py | 349 | 363| 144 | 69841 | 177459 | 
| 192 | 45 src/_pytest/runner.py | 62 | 94| 270 | 70111 | 177459 | 
| 193 | 45 src/_pytest/runner.py | 113 | 126| 129 | 70240 | 177459 | 
| 194 | 45 src/_pytest/resultlog.py | 38 | 60| 191 | 70431 | 177459 | 
| 195 | 45 src/_pytest/terminal.py | 685 | 697| 119 | 70550 | 177459 | 
| 196 | 45 src/_pytest/python.py | 1143 | 1171| 266 | 70816 | 177459 | 
| 197 | 45 src/_pytest/unittest.py | 1 | 26| 176 | 70992 | 177459 | 
| 198 | 45 testing/python/integration.py | 439 | 466| 160 | 71152 | 177459 | 
| 199 | 46 src/_pytest/nodes.py | 241 | 287| 365 | 71517 | 180555 | 
| 200 | 46 src/_pytest/skipping.py | 1 | 25| 162 | 71679 | 180555 | 
| 201 | 47 bench/bench_argcomplete.py | 1 | 20| 179 | 71858 | 180734 | 
| 202 | 47 src/_pytest/nodes.py | 394 | 426| 244 | 72102 | 180734 | 
| 203 | 47 testing/python/integration.py | 415 | 437| 140 | 72242 | 180734 | 
| 204 | 48 src/_pytest/__init__.py | 1 | 9| 0 | 72242 | 180790 | 
| 205 | 48 src/_pytest/setuponly.py | 1 | 46| 297 | 72539 | 180790 | 
| 206 | 49 src/_pytest/mark/structures.py | 36 | 59| 202 | 72741 | 183609 | 
| 207 | 50 setup.py | 1 | 16| 147 | 72888 | 183878 | 
| 208 | 51 testing/python/setup_only.py | 123 | 153| 183 | 73071 | 185411 | 
| 209 | 51 src/_pytest/helpconfig.py | 117 | 136| 125 | 73196 | 185411 | 
| 210 | 51 src/_pytest/pytester.py | 276 | 307| 247 | 73443 | 185411 | 
| 211 | 51 testing/python/integration.py | 242 | 264| 183 | 73626 | 185411 | 
| 212 | 51 testing/python/setup_only.py | 156 | 182| 167 | 73793 | 185411 | 
| 213 | 51 testing/python/fixtures.py | 3080 | 3972| 5385 | 79178 | 185411 | 
| 214 | 51 src/_pytest/config/__init__.py | 266 | 293| 257 | 79435 | 185411 | 
| 215 | 51 src/_pytest/fixtures.py | 368 | 382| 198 | 79633 | 185411 | 
| 216 | 51 testing/python/metafunc.py | 955 | 981| 196 | 79829 | 185411 | 
| 217 | 51 src/_pytest/mark/structures.py | 1 | 33| 179 | 80008 | 185411 | 
| 218 | 51 src/_pytest/runner.py | 129 | 148| 171 | 80179 | 185411 | 
| 219 | 51 src/_pytest/cacheprovider.py | 350 | 391| 287 | 80466 | 185411 | 
| 220 | 51 src/_pytest/doctest.py | 294 | 310| 133 | 80599 | 185411 | 
| 221 | 51 src/_pytest/terminal.py | 350 | 362| 121 | 80720 | 185411 | 
| 222 | 51 testing/python/setup_only.py | 62 | 93| 190 | 80910 | 185411 | 
| 223 | 51 src/_pytest/_code/code.py | 912 | 953| 294 | 81204 | 185411 | 
| 224 | 52 extra/setup-py.test/setup.py | 1 | 12| 0 | 81204 | 185489 | 
| 225 | 52 testing/python/metafunc.py | 1665 | 1704| 260 | 81464 | 185489 | 
| 226 | 52 testing/python/metafunc.py | 1271 | 1293| 224 | 81688 | 185489 | 
| 227 | 52 doc/en/example/assertion/failure_demo.py | 256 | 283| 161 | 81849 | 185489 | 
| 228 | 52 src/_pytest/main.py | 227 | 261| 245 | 82094 | 185489 | 


## Patch

```diff
diff --git a/src/_pytest/junitxml.py b/src/_pytest/junitxml.py
--- a/src/_pytest/junitxml.py
+++ b/src/_pytest/junitxml.py
@@ -10,9 +10,11 @@
 """
 import functools
 import os
+import platform
 import re
 import sys
 import time
+from datetime import datetime
 
 import py
 
@@ -666,6 +668,8 @@ def pytest_sessionfinish(self):
             skipped=self.stats["skipped"],
             tests=numtests,
             time="%.3f" % suite_time_delta,
+            timestamp=datetime.fromtimestamp(self.suite_start_time).isoformat(),
+            hostname=platform.node(),
         )
         logfile.write(Junit.testsuites([suite_node]).unicode(indent=0))
         logfile.close()

```

## Test Patch

```diff
diff --git a/testing/test_junitxml.py b/testing/test_junitxml.py
--- a/testing/test_junitxml.py
+++ b/testing/test_junitxml.py
@@ -1,4 +1,6 @@
 import os
+import platform
+from datetime import datetime
 from xml.dom import minidom
 
 import py
@@ -139,6 +141,30 @@ def test_xpass():
         node = dom.find_first_by_tag("testsuite")
         node.assert_attr(name="pytest", errors=1, failures=2, skipped=1, tests=5)
 
+    def test_hostname_in_xml(self, testdir):
+        testdir.makepyfile(
+            """
+            def test_pass():
+                pass
+        """
+        )
+        result, dom = runandparse(testdir)
+        node = dom.find_first_by_tag("testsuite")
+        node.assert_attr(hostname=platform.node())
+
+    def test_timestamp_in_xml(self, testdir):
+        testdir.makepyfile(
+            """
+            def test_pass():
+                pass
+        """
+        )
+        start_time = datetime.now()
+        result, dom = runandparse(testdir)
+        node = dom.find_first_by_tag("testsuite")
+        timestamp = datetime.strptime(node["timestamp"], "%Y-%m-%dT%H:%M:%S.%f")
+        assert start_time <= timestamp < datetime.now()
+
     def test_timing_function(self, testdir):
         testdir.makepyfile(
             """

```


## Code snippets

### 1 - src/_pytest/junitxml.py:

Start line: 116, End line: 157

```python
class _NodeReporter:

    def record_testreport(self, testreport):
        assert not self.testcase
        names = mangle_test_address(testreport.nodeid)
        existing_attrs = self.attrs
        classnames = names[:-1]
        if self.xml.prefix:
            classnames.insert(0, self.xml.prefix)
        attrs = {
            "classname": ".".join(classnames),
            "name": bin_xml_escape(names[-1]),
            "file": testreport.location[0],
        }
        if testreport.location[1] is not None:
            attrs["line"] = testreport.location[1]
        if hasattr(testreport, "url"):
            attrs["url"] = testreport.url
        self.attrs = attrs
        self.attrs.update(existing_attrs)  # restore any user-defined attributes

        # Preserve legacy testcase behavior
        if self.family == "xunit1":
            return

        # Filter out attributes not permitted by this test family.
        # Including custom attributes because they are not valid here.
        temp_attrs = {}
        for key in self.attrs.keys():
            if key in families[self.family]["testcase"]:
                temp_attrs[key] = self.attrs[key]
        self.attrs = temp_attrs

    def to_xml(self):
        testcase = Junit.testcase(time="%.3f" % self.duration, **self.attrs)
        testcase.append(self.make_properties_node())
        for node in self.nodes:
            testcase.append(node)
        return testcase

    def _add_simple(self, kind, message, data=None):
        data = bin_xml_escape(data)
        node = kind(data, message=message)
        self.append(node)
```
### 2 - src/_pytest/junitxml.py:

Start line: 673, End line: 692

```python
class LogXML:

    def pytest_terminal_summary(self, terminalreporter):
        terminalreporter.write_sep("-", "generated xml file: %s" % (self.logfile))

    def add_global_property(self, name, value):
        __tracebackhide__ = True
        _check_record_param_type("name", name)
        self.global_properties.append((name, bin_xml_escape(value)))

    def _get_global_properties_node(self):
        """Return a Junit node containing custom properties, if any.
        """
        if self.global_properties:
            return Junit.properties(
                [
                    Junit.property(name=name, value=value)
                    for name, value in self.global_properties
                ]
            )
        return ""
```
### 3 - src/_pytest/junitxml.py:

Start line: 1, End line: 79

```python
"""
    report test results in JUnit-XML format,
    for use with Jenkins and build integration servers.


Based on initial code from Ross Lawley.

Output conforms to https://github.com/jenkinsci/xunit-plugin/blob/master/
src/main/resources/org/jenkinsci/plugins/xunit/types/model/xsd/junit-10.xsd
"""
import functools
import os
import re
import sys
import time

import py

import pytest
from _pytest import nodes
from _pytest.config import filename_arg


class Junit(py.xml.Namespace):
    pass


# We need to get the subset of the invalid unicode ranges according to
# XML 1.0 which are valid in this python build.  Hence we calculate
# this dynamically instead of hardcoding it.  The spec range of valid
# chars is: Char ::= #x9 | #xA | #xD | [#x20-#xD7FF] | [#xE000-#xFFFD]
#                    | [#x10000-#x10FFFF]
_legal_chars = (0x09, 0x0A, 0x0D)
_legal_ranges = ((0x20, 0x7E), (0x80, 0xD7FF), (0xE000, 0xFFFD), (0x10000, 0x10FFFF))
_legal_xml_re = [
    "{}-{}".format(chr(low), chr(high))
    for (low, high) in _legal_ranges
    if low < sys.maxunicode
]
_legal_xml_re = [chr(x) for x in _legal_chars] + _legal_xml_re
illegal_xml_re = re.compile("[^%s]" % "".join(_legal_xml_re))
del _legal_chars
del _legal_ranges
del _legal_xml_re

_py_ext_re = re.compile(r"\.py$")


def bin_xml_escape(arg):
    def repl(matchobj):
        i = ord(matchobj.group())
        if i <= 0xFF:
            return "#x%02X" % i
        else:
            return "#x%04X" % i

    return py.xml.raw(illegal_xml_re.sub(repl, py.xml.escape(arg)))


def merge_family(left, right):
    result = {}
    for kl, vl in left.items():
        for kr, vr in right.items():
            if not isinstance(vl, list):
                raise TypeError(type(vl))
            result[kl] = vl + vr
    left.update(result)


families = {}
families["_base"] = {"testcase": ["classname", "name"]}
families["_base_legacy"] = {"testcase": ["file", "line", "url"]}

# xUnit 1.x inherits legacy attributes
families["xunit1"] = families["_base"].copy()
merge_family(families["xunit1"], families["_base_legacy"])

# xUnit 2.x uses strict base attributes
families["xunit2"] = families["_base"]
```
### 4 - src/_pytest/junitxml.py:

Start line: 506, End line: 531

```python
class LogXML:

    def node_reporter(self, report):
        nodeid = getattr(report, "nodeid", report)
        # local hack to handle xdist report order
        slavenode = getattr(report, "node", None)

        key = nodeid, slavenode

        if key in self.node_reporters:
            # TODO: breasks for --dist=each
            return self.node_reporters[key]

        reporter = _NodeReporter(nodeid, self)

        self.node_reporters[key] = reporter
        self.node_reporters_ordered.append(reporter)

        return reporter

    def add_stats(self, key):
        if key in self.stats:
            self.stats[key] += 1

    def _opentestcase(self, report):
        reporter = self.node_reporter(report)
        reporter.record_testreport(report)
        return reporter
```
### 5 - src/_pytest/junitxml.py:

Start line: 381, End line: 425

```python
def pytest_addoption(parser):
    group = parser.getgroup("terminal reporting")
    group.addoption(
        "--junitxml",
        "--junit-xml",
        action="store",
        dest="xmlpath",
        metavar="path",
        type=functools.partial(filename_arg, optname="--junitxml"),
        default=None,
        help="create junit-xml style report file at given path.",
    )
    group.addoption(
        "--junitprefix",
        "--junit-prefix",
        action="store",
        metavar="str",
        default=None,
        help="prepend prefix to classnames in junit-xml output",
    )
    parser.addini(
        "junit_suite_name", "Test suite name for JUnit report", default="pytest"
    )
    parser.addini(
        "junit_logging",
        "Write captured log messages to JUnit report: "
        "one of no|system-out|system-err",
        default="no",
    )  # choices=['no', 'stdout', 'stderr'])
    parser.addini(
        "junit_log_passing_tests",
        "Capture log information for passing tests to JUnit report: ",
        type="bool",
        default=True,
    )
    parser.addini(
        "junit_duration_report",
        "Duration time to report: one of total|call",
        default="total",
    )  # choices=['total', 'call'])
    parser.addini(
        "junit_family",
        "Emit XML for schema: one of legacy|xunit1|xunit2",
        default="xunit1",
    )
```
### 6 - src/_pytest/junitxml.py:

Start line: 82, End line: 114

```python
class _NodeReporter:
    def __init__(self, nodeid, xml):
        self.id = nodeid
        self.xml = xml
        self.add_stats = self.xml.add_stats
        self.family = self.xml.family
        self.duration = 0
        self.properties = []
        self.nodes = []
        self.testcase = None
        self.attrs = {}

    def append(self, node):
        self.xml.add_stats(type(node).__name__)
        self.nodes.append(node)

    def add_property(self, name, value):
        self.properties.append((str(name), bin_xml_escape(value)))

    def add_attribute(self, name, value):
        self.attrs[str(name)] = bin_xml_escape(value)

    def make_properties_node(self):
        """Return a Junit node containing custom properties, if any.
        """
        if self.properties:
            return Junit.properties(
                [
                    Junit.property(name=name, value=value)
                    for name, value in self.properties
                ]
            )
        return ""
```
### 7 - src/_pytest/junitxml.py:

Start line: 159, End line: 210

```python
class _NodeReporter:

    def write_captured_output(self, report):
        if not self.xml.log_passing_tests and report.passed:
            return

        content_out = report.capstdout
        content_log = report.caplog
        content_err = report.capstderr

        if content_log or content_out:
            if content_log and self.xml.logging == "system-out":
                if content_out:
                    # syncing stdout and the log-output is not done yet. It's
                    # probably not worth the effort. Therefore, first the captured
                    # stdout is shown and then the captured logs.
                    content = "\n".join(
                        [
                            " Captured Stdout ".center(80, "-"),
                            content_out,
                            "",
                            " Captured Log ".center(80, "-"),
                            content_log,
                        ]
                    )
                else:
                    content = content_log
            else:
                content = content_out

            if content:
                tag = getattr(Junit, "system-out")
                self.append(tag(bin_xml_escape(content)))

        if content_log or content_err:
            if content_log and self.xml.logging == "system-err":
                if content_err:
                    content = "\n".join(
                        [
                            " Captured Stderr ".center(80, "-"),
                            content_err,
                            "",
                            " Captured Log ".center(80, "-"),
                            content_log,
                        ]
                    )
                else:
                    content = content_log
            else:
                content = content_err

            if content:
                tag = getattr(Junit, "system-err")
                self.append(tag(bin_xml_escape(content)))
```
### 8 - src/_pytest/junitxml.py:

Start line: 212, End line: 229

```python
class _NodeReporter:

    def append_pass(self, report):
        self.add_stats("passed")

    def append_failure(self, report):
        # msg = str(report.longrepr.reprtraceback.extraline)
        if hasattr(report, "wasxfail"):
            self._add_simple(Junit.skipped, "xfail-marked test passes unexpectedly")
        else:
            if hasattr(report.longrepr, "reprcrash"):
                message = report.longrepr.reprcrash.message
            elif isinstance(report.longrepr, str):
                message = report.longrepr
            else:
                message = str(report.longrepr)
            message = bin_xml_escape(message)
            fail = Junit.failure(message=message)
            fail.append(bin_xml_escape(report.longrepr))
            self.append(fail)
```
### 9 - src/_pytest/junitxml.py:

Start line: 351, End line: 378

```python
@pytest.fixture(scope="session")
def record_testsuite_property(request):
    """
    Records a new ``<property>`` tag as child of the root ``<testsuite>``. This is suitable to
    writing global information regarding the entire test suite, and is compatible with ``xunit2`` JUnit family.

    This is a ``session``-scoped fixture which is called with ``(name, value)``. Example:

    .. code-block:: python

        def test_foo(record_testsuite_property):
            record_testsuite_property("ARCH", "PPC")
            record_testsuite_property("STORAGE_TYPE", "CEPH")

    ``name`` must be a string, ``value`` will be converted to a string and properly xml-escaped.
    """

    __tracebackhide__ = True

    def record_func(name, value):
        """noop function in case --junitxml was not passed in the command-line"""
        __tracebackhide__ = True
        _check_record_param_type("name", name)

    xml = getattr(request.config, "_xml", None)
    if xml is not None:
        record_func = xml.add_global_property  # noqa
    return record_func
```
### 10 - src/_pytest/junitxml.py:

Start line: 466, End line: 504

```python
class LogXML:
    def __init__(
        self,
        logfile,
        prefix,
        suite_name="pytest",
        logging="no",
        report_duration="total",
        family="xunit1",
        log_passing_tests=True,
    ):
        logfile = os.path.expanduser(os.path.expandvars(logfile))
        self.logfile = os.path.normpath(os.path.abspath(logfile))
        self.prefix = prefix
        self.suite_name = suite_name
        self.logging = logging
        self.log_passing_tests = log_passing_tests
        self.report_duration = report_duration
        self.family = family
        self.stats = dict.fromkeys(["error", "passed", "failure", "skipped"], 0)
        self.node_reporters = {}  # nodeid -> _NodeReporter
        self.node_reporters_ordered = []
        self.global_properties = []

        # List of reports that failed on call but teardown is pending.
        self.open_reports = []
        self.cnt_double_fail_tests = 0

        # Replaces convenience family with real family
        if self.family == "legacy":
            self.family = "xunit1"

    def finalize(self, report):
        nodeid = getattr(report, "nodeid", report)
        # local hack to handle xdist report order
        slavenode = getattr(report, "node", None)
        reporter = self.node_reporters.pop((nodeid, slavenode))
        if reporter is not None:
            reporter.finalize()
```
### 11 - src/_pytest/junitxml.py:

Start line: 247, End line: 275

```python
class _NodeReporter:

    def append_skipped(self, report):
        if hasattr(report, "wasxfail"):
            xfailreason = report.wasxfail
            if xfailreason.startswith("reason: "):
                xfailreason = xfailreason[8:]
            self.append(
                Junit.skipped(
                    "", type="pytest.xfail", message=bin_xml_escape(xfailreason)
                )
            )
        else:
            filename, lineno, skipreason = report.longrepr
            if skipreason.startswith("Skipped: "):
                skipreason = skipreason[9:]
            details = "{}:{}: {}".format(filename, lineno, skipreason)

            self.append(
                Junit.skipped(
                    bin_xml_escape(details),
                    type="pytest.skip",
                    message=bin_xml_escape(skipreason),
                )
            )
            self.write_captured_output(report)

    def finalize(self):
        data = self.to_xml().unicode(indent=0)
        self.__dict__.clear()
        self.to_xml = lambda: py.xml.raw(data)
```
### 12 - src/_pytest/junitxml.py:

Start line: 293, End line: 311

```python
@pytest.fixture
def record_property(request):
    """Add an extra properties the calling test.
    User properties become part of the test report and are available to the
    configured reporters, like JUnit XML.
    The fixture is callable with ``(name, value)``, with value being automatically
    xml-encoded.

    Example::

        def test_function(record_property):
            record_property("example_key", 1)
    """
    _warn_incompatibility_with_xunit2(request, "record_property")

    def append_property(name, value):
        request.node.user_properties.append((name, value))

    return append_property
```
### 13 - src/_pytest/junitxml.py:

Start line: 643, End line: 671

```python
class LogXML:

    def pytest_sessionfinish(self):
        dirname = os.path.dirname(os.path.abspath(self.logfile))
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        logfile = open(self.logfile, "w", encoding="utf-8")
        suite_stop_time = time.time()
        suite_time_delta = suite_stop_time - self.suite_start_time

        numtests = (
            self.stats["passed"]
            + self.stats["failure"]
            + self.stats["skipped"]
            + self.stats["error"]
            - self.cnt_double_fail_tests
        )
        logfile.write('<?xml version="1.0" encoding="utf-8"?>')

        suite_node = Junit.testsuite(
            self._get_global_properties_node(),
            [x.to_xml() for x in self.node_reporters_ordered],
            name=self.suite_name,
            errors=self.stats["error"],
            failures=self.stats["failure"],
            skipped=self.stats["skipped"],
            tests=numtests,
            time="%.3f" % suite_time_delta,
        )
        logfile.write(Junit.testsuites([suite_node]).unicode(indent=0))
        logfile.close()
```
### 14 - src/_pytest/junitxml.py:

Start line: 533, End line: 617

```python
class LogXML:

    def pytest_runtest_logreport(self, report):
        """handle a setup/call/teardown report, generating the appropriate
        xml tags as necessary.

        note: due to plugins like xdist, this hook may be called in interlaced
        order with reports from other nodes. for example:

        usual call order:
            -> setup node1
            -> call node1
            -> teardown node1
            -> setup node2
            -> call node2
            -> teardown node2

        possible call order in xdist:
            -> setup node1
            -> call node1
            -> setup node2
            -> call node2
            -> teardown node2
            -> teardown node1
        """
        close_report = None
        if report.passed:
            if report.when == "call":  # ignore setup/teardown
                reporter = self._opentestcase(report)
                reporter.append_pass(report)
        elif report.failed:
            if report.when == "teardown":
                # The following vars are needed when xdist plugin is used
                report_wid = getattr(report, "worker_id", None)
                report_ii = getattr(report, "item_index", None)
                close_report = next(
                    (
                        rep
                        for rep in self.open_reports
                        if (
                            rep.nodeid == report.nodeid
                            and getattr(rep, "item_index", None) == report_ii
                            and getattr(rep, "worker_id", None) == report_wid
                        )
                    ),
                    None,
                )
                if close_report:
                    # We need to open new testcase in case we have failure in
                    # call and error in teardown in order to follow junit
                    # schema
                    self.finalize(close_report)
                    self.cnt_double_fail_tests += 1
            reporter = self._opentestcase(report)
            if report.when == "call":
                reporter.append_failure(report)
                self.open_reports.append(report)
            else:
                reporter.append_error(report)
        elif report.skipped:
            reporter = self._opentestcase(report)
            reporter.append_skipped(report)
        self.update_testcase_duration(report)
        if report.when == "teardown":
            reporter = self._opentestcase(report)
            reporter.write_captured_output(report)

            for propname, propvalue in report.user_properties:
                reporter.add_property(propname, propvalue)

            self.finalize(report)
            report_wid = getattr(report, "worker_id", None)
            report_ii = getattr(report, "item_index", None)
            close_report = next(
                (
                    rep
                    for rep in self.open_reports
                    if (
                        rep.nodeid == report.nodeid
                        and getattr(rep, "item_index", None) == report_ii
                        and getattr(rep, "worker_id", None) == report_wid
                    )
                ),
                None,
            )
            if close_report:
                self.open_reports.remove(close_report)
```
### 15 - src/_pytest/junitxml.py:

Start line: 231, End line: 245

```python
class _NodeReporter:

    def append_collect_error(self, report):
        # msg = str(report.longrepr.reprtraceback.extraline)
        self.append(
            Junit.error(bin_xml_escape(report.longrepr), message="collection failure")
        )

    def append_collect_skipped(self, report):
        self._add_simple(Junit.skipped, "collection skipped", report.longrepr)

    def append_error(self, report):
        if report.when == "teardown":
            msg = "test teardown failure"
        else:
            msg = "test setup failure"
        self._add_simple(Junit.error, msg, report.longrepr)
```
### 17 - src/_pytest/junitxml.py:

Start line: 619, End line: 641

```python
class LogXML:

    def update_testcase_duration(self, report):
        """accumulates total duration for nodeid from given report and updates
        the Junit.testcase with the new total if already created.
        """
        if self.report_duration == "total" or report.when == self.report_duration:
            reporter = self.node_reporter(report)
            reporter.duration += getattr(report, "duration", 0.0)

    def pytest_collectreport(self, report):
        if not report.passed:
            reporter = self._opentestcase(report)
            if report.failed:
                reporter.append_collect_error(report)
            else:
                reporter.append_collect_skipped(report)

    def pytest_internalerror(self, excrepr):
        reporter = self.node_reporter("internal")
        reporter.attrs.update(classname="pytest", name="internal")
        reporter._add_simple(Junit.error, "internal error", excrepr)

    def pytest_sessionstart(self):
        self.suite_start_time = time.time()
```
### 18 - src/_pytest/junitxml.py:

Start line: 428, End line: 441

```python
def pytest_configure(config):
    xmlpath = config.option.xmlpath
    # prevent opening xmllog on slave nodes (xdist)
    if xmlpath and not hasattr(config, "slaveinput"):
        config._xml = LogXML(
            xmlpath,
            config.option.junitprefix,
            config.getini("junit_suite_name"),
            config.getini("junit_logging"),
            config.getini("junit_duration_report"),
            config.getini("junit_family"),
            config.getini("junit_log_passing_tests"),
        )
        config.pluginmanager.register(config._xml)
```
### 21 - src/_pytest/junitxml.py:

Start line: 314, End line: 348

```python
@pytest.fixture
def record_xml_attribute(request):
    """Add extra xml attributes to the tag for the calling test.
    The fixture is callable with ``(name, value)``, with value being
    automatically xml-encoded
    """
    from _pytest.warning_types import PytestExperimentalApiWarning

    request.node.warn(
        PytestExperimentalApiWarning("record_xml_attribute is an experimental feature")
    )

    _warn_incompatibility_with_xunit2(request, "record_xml_attribute")

    # Declare noop
    def add_attr_noop(name, value):
        pass

    attr_func = add_attr_noop

    xml = getattr(request.config, "_xml", None)
    if xml is not None:
        node_reporter = xml.node_reporter(request.node.nodeid)
        attr_func = node_reporter.add_attribute

    return attr_func


def _check_record_param_type(param, v):
    """Used by record_testsuite_property to check that the given parameter name is of the proper
    type"""
    __tracebackhide__ = True
    if not isinstance(v, str):
        msg = "{param} parameter needs to be a string, but {g} given"
        raise TypeError(msg.format(param=param, g=type(v).__name__))
```
### 59 - src/_pytest/junitxml.py:

Start line: 444, End line: 463

```python
def pytest_unconfigure(config):
    xml = getattr(config, "_xml", None)
    if xml:
        del config._xml
        config.pluginmanager.unregister(xml)


def mangle_test_address(address):
    path, possible_open_bracket, params = address.partition("[")
    names = path.split("::")
    try:
        names.remove("()")
    except ValueError:
        pass
    # convert file path to dotted path
    names[0] = names[0].replace(nodes.SEP, ".")
    names[0] = _py_ext_re.sub("", names[0])
    # put any params back
    names[-1] += possible_open_bracket + params
    return names
```
### 97 - src/_pytest/junitxml.py:

Start line: 278, End line: 290

```python
def _warn_incompatibility_with_xunit2(request, fixture_name):
    """Emits a PytestWarning about the given fixture being incompatible with newer xunit revisions"""
    from _pytest.warning_types import PytestWarning

    xml = getattr(request.config, "_xml", None)
    if xml is not None and xml.family not in ("xunit1", "legacy"):
        request.node.warn(
            PytestWarning(
                "{fixture_name} is incompatible with junit_family '{family}' (use 'legacy' or 'xunit1')".format(
                    fixture_name=fixture_name, family=xml.family
                )
            )
        )
```
