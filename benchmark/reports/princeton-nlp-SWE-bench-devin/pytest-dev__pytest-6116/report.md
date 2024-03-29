# pytest-dev__pytest-6116

| **pytest-dev/pytest** | `e670ff76cbad80108bde9bab616b66771b8653cf` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 757 |
| **Any found context length** | 757 |
| **Avg pos** | 1.0 |
| **Min pos** | 1 |
| **Max pos** | 1 |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/src/_pytest/main.py b/src/_pytest/main.py
--- a/src/_pytest/main.py
+++ b/src/_pytest/main.py
@@ -109,6 +109,7 @@ def pytest_addoption(parser):
     group.addoption(
         "--collectonly",
         "--collect-only",
+        "--co",
         action="store_true",
         help="only collect tests, don't execute them.",
     ),

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| src/_pytest/main.py | 112 | 112 | 1 | 1 | 757


## Problem Statement

```
pytest --collect-only needs a one char shortcut command
I find myself needing to run `--collect-only` very often and that cli argument is a very long to type one. 

I do think that it would be great to allocate a character for it, not sure which one yet. Please use up/down thumbs to vote if you would find it useful or not and eventually proposing which char should be used. 

Clearly this is a change very easy to implement but first I want to see if others would find it useful or not.
pytest --collect-only needs a one char shortcut command
I find myself needing to run `--collect-only` very often and that cli argument is a very long to type one. 

I do think that it would be great to allocate a character for it, not sure which one yet. Please use up/down thumbs to vote if you would find it useful or not and eventually proposing which char should be used. 

Clearly this is a change very easy to implement but first I want to see if others would find it useful or not.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 src/_pytest/main.py** | 46 | 154| 757 | 757 | 5363 | 
| 2 | 2 src/_pytest/terminal.py | 57 | 148| 639 | 1396 | 14296 | 
| 3 | 2 src/_pytest/terminal.py | 32 | 54| 174 | 1570 | 14296 | 
| 4 | **2 src/_pytest/main.py** | 155 | 181| 175 | 1745 | 14296 | 
| 5 | 3 src/_pytest/mark/__init__.py | 37 | 75| 341 | 2086 | 15452 | 
| 6 | 4 src/_pytest/config/argparsing.py | 424 | 464| 427 | 2513 | 19388 | 
| 7 | 4 src/_pytest/config/argparsing.py | 408 | 422| 155 | 2668 | 19388 | 
| 8 | 5 src/_pytest/debugging.py | 22 | 43| 155 | 2823 | 21829 | 
| 9 | 6 src/_pytest/junitxml.py | 383 | 427| 334 | 3157 | 26883 | 
| 10 | 6 src/_pytest/terminal.py | 1061 | 1080| 160 | 3317 | 26883 | 
| 11 | 7 src/_pytest/python.py | 57 | 109| 354 | 3671 | 38295 | 
| 12 | 8 src/_pytest/doctest.py | 48 | 98| 334 | 4005 | 43130 | 
| 13 | 9 src/_pytest/helpconfig.py | 39 | 83| 297 | 4302 | 44819 | 
| 14 | 9 src/_pytest/config/argparsing.py | 253 | 279| 223 | 4525 | 44819 | 
| 15 | 9 src/_pytest/config/argparsing.py | 371 | 405| 359 | 4884 | 44819 | 
| 16 | 10 src/pytest.py | 1 | 107| 708 | 5592 | 45527 | 
| 17 | 11 src/_pytest/stepwise.py | 1 | 23| 131 | 5723 | 46241 | 
| 18 | 12 src/_pytest/assertion/truncate.py | 76 | 96| 175 | 5898 | 46963 | 
| 19 | 12 src/_pytest/python.py | 112 | 130| 176 | 6074 | 46963 | 
| 20 | 13 src/_pytest/setuponly.py | 1 | 43| 289 | 6363 | 47496 | 
| 21 | **13 src/_pytest/main.py** | 316 | 367| 314 | 6677 | 47496 | 
| 22 | 13 src/_pytest/terminal.py | 527 | 565| 335 | 7012 | 47496 | 
| 23 | 14 bench/bench_argcomplete.py | 1 | 20| 179 | 7191 | 47675 | 
| 24 | 14 src/_pytest/config/argparsing.py | 295 | 329| 315 | 7506 | 47675 | 
| 25 | 15 src/_pytest/faulthandler.py | 1 | 36| 249 | 7755 | 48250 | 
| 26 | 16 src/_pytest/cacheprovider.py | 302 | 357| 410 | 8165 | 51704 | 
| 27 | 16 src/_pytest/config/argparsing.py | 163 | 234| 605 | 8770 | 51704 | 
| 28 | 16 src/_pytest/terminal.py | 273 | 297| 161 | 8931 | 51704 | 
| 29 | 17 testing/python/collect.py | 881 | 918| 321 | 9252 | 60980 | 
| 30 | 18 src/_pytest/pytester.py | 47 | 69| 144 | 9396 | 72237 | 
| 31 | 19 src/_pytest/capture.py | 1 | 36| 198 | 9594 | 78227 | 
| 32 | 19 src/_pytest/config/argparsing.py | 64 | 86| 223 | 9817 | 78227 | 
| 33 | 19 testing/python/collect.py | 788 | 810| 195 | 10012 | 78227 | 
| 34 | 19 src/_pytest/config/argparsing.py | 236 | 251| 127 | 10139 | 78227 | 
| 35 | **19 src/_pytest/main.py** | 288 | 313| 226 | 10365 | 78227 | 
| 36 | 20 src/_pytest/skipping.py | 28 | 65| 363 | 10728 | 79702 | 
| 37 | 20 src/_pytest/debugging.py | 165 | 207| 351 | 11079 | 79702 | 
| 38 | 20 src/_pytest/helpconfig.py | 86 | 114| 219 | 11298 | 79702 | 
| 39 | 20 src/_pytest/terminal.py | 973 | 998| 223 | 11521 | 79702 | 
| 40 | 20 src/_pytest/helpconfig.py | 139 | 206| 550 | 12071 | 79702 | 
| 41 | 20 src/_pytest/setuponly.py | 46 | 79| 242 | 12313 | 79702 | 
| 42 | 20 src/_pytest/python.py | 357 | 384| 249 | 12562 | 79702 | 
| 43 | 20 src/_pytest/mark/__init__.py | 78 | 93| 123 | 12685 | 79702 | 
| 44 | 20 src/_pytest/terminal.py | 616 | 631| 128 | 12813 | 79702 | 
| 45 | 20 testing/python/collect.py | 1205 | 1230| 186 | 12999 | 79702 | 
| 46 | 20 src/_pytest/mark/__init__.py | 123 | 161| 242 | 13241 | 79702 | 
| 47 | 21 testing/example_scripts/config/collect_pytest_prefix/conftest.py | 1 | 3| 0 | 13241 | 79710 | 
| 48 | **21 src/_pytest/main.py** | 231 | 268| 264 | 13505 | 79710 | 
| 49 | 22 src/_pytest/_argcomplete.py | 62 | 107| 294 | 13799 | 80547 | 
| 50 | 23 src/_pytest/setupplan.py | 1 | 28| 163 | 13962 | 80711 | 
| 51 | 23 testing/python/collect.py | 846 | 878| 304 | 14266 | 80711 | 
| 52 | 23 testing/python/collect.py | 524 | 538| 150 | 14416 | 80711 | 
| 53 | 24 src/_pytest/config/__init__.py | 107 | 171| 356 | 14772 | 89562 | 
| 54 | 24 testing/python/collect.py | 1106 | 1143| 209 | 14981 | 89562 | 
| 55 | 24 src/_pytest/cacheprovider.py | 149 | 203| 500 | 15481 | 89562 | 
| 56 | 24 src/_pytest/skipping.py | 1 | 25| 162 | 15643 | 89562 | 
| 57 | 25 testing/conftest.py | 1 | 60| 359 | 16002 | 90240 | 
| 58 | 25 src/_pytest/config/__init__.py | 741 | 763| 172 | 16174 | 90240 | 
| 59 | 26 src/_pytest/warnings.py | 32 | 54| 156 | 16330 | 91370 | 
| 60 | 26 src/_pytest/config/argparsing.py | 281 | 292| 135 | 16465 | 91370 | 
| 61 | 26 src/_pytest/doctest.py | 632 | 654| 201 | 16666 | 91370 | 
| 62 | 26 testing/python/collect.py | 1076 | 1103| 183 | 16849 | 91370 | 
| 63 | 27 src/_pytest/runner.py | 42 | 68| 232 | 17081 | 94199 | 
| 64 | 27 src/_pytest/terminal.py | 924 | 954| 271 | 17352 | 94199 | 
| 65 | 27 src/_pytest/runner.py | 1 | 39| 212 | 17564 | 94199 | 
| 66 | 27 src/_pytest/terminal.py | 707 | 719| 119 | 17683 | 94199 | 
| 67 | 27 testing/python/collect.py | 279 | 344| 471 | 18154 | 94199 | 
| 68 | 27 testing/python/collect.py | 540 | 557| 182 | 18336 | 94199 | 
| 69 | 27 testing/python/collect.py | 713 | 728| 143 | 18479 | 94199 | 
| 70 | 28 src/_pytest/mark/structures.py | 37 | 60| 202 | 18681 | 97062 | 
| 71 | 28 src/_pytest/pytester.py | 1 | 44| 246 | 18927 | 97062 | 
| 72 | 28 src/_pytest/mark/__init__.py | 1 | 14| 116 | 19043 | 97062 | 
| 73 | **28 src/_pytest/main.py** | 457 | 500| 369 | 19412 | 97062 | 
| 74 | 29 doc/en/example/xfail_demo.py | 1 | 39| 143 | 19555 | 97206 | 
| 75 | 29 src/_pytest/terminal.py | 633 | 666| 340 | 19895 | 97206 | 
| 76 | 29 src/_pytest/terminal.py | 151 | 178| 216 | 20111 | 97206 | 
| 77 | 29 src/_pytest/python.py | 386 | 427| 367 | 20478 | 97206 | 
| 78 | 29 src/_pytest/helpconfig.py | 117 | 136| 125 | 20603 | 97206 | 
| 79 | 29 src/_pytest/config/argparsing.py | 88 | 106| 183 | 20786 | 97206 | 
| 80 | 29 src/_pytest/_argcomplete.py | 1 | 59| 541 | 21327 | 97206 | 
| 81 | 29 src/_pytest/assertion/truncate.py | 1 | 34| 209 | 21536 | 97206 | 
| 82 | 29 src/_pytest/helpconfig.py | 1 | 36| 242 | 21778 | 97206 | 
| 83 | 29 src/_pytest/pytester.py | 810 | 820| 121 | 21899 | 97206 | 
| 84 | 29 testing/python/collect.py | 920 | 939| 178 | 22077 | 97206 | 
| 85 | 29 src/_pytest/terminal.py | 181 | 197| 125 | 22202 | 97206 | 
| 86 | 29 testing/python/collect.py | 559 | 658| 662 | 22864 | 97206 | 
| 87 | 29 src/_pytest/terminal.py | 1039 | 1058| 171 | 23035 | 97206 | 
| 88 | 29 src/_pytest/doctest.py | 312 | 325| 136 | 23171 | 97206 | 
| 89 | 30 src/_pytest/assertion/__init__.py | 1 | 33| 196 | 23367 | 98436 | 
| 90 | 30 src/_pytest/config/__init__.py | 1045 | 1075| 261 | 23628 | 98436 | 
| 91 | 30 src/_pytest/doctest.py | 347 | 373| 197 | 23825 | 98436 | 
| 92 | 30 src/_pytest/python.py | 200 | 236| 354 | 24179 | 98436 | 
| 93 | **30 src/_pytest/main.py** | 184 | 228| 366 | 24545 | 98436 | 
| 94 | 30 src/_pytest/python.py | 133 | 150| 212 | 24757 | 98436 | 
| 95 | 30 src/_pytest/config/__init__.py | 1 | 57| 349 | 25106 | 98436 | 
| 96 | 30 src/_pytest/runner.py | 276 | 343| 533 | 25639 | 98436 | 
| 97 | 31 testing/python/metafunc.py | 1579 | 1609| 228 | 25867 | 111534 | 
| 98 | 31 testing/python/collect.py | 755 | 786| 204 | 26071 | 111534 | 
| 99 | 31 src/_pytest/doctest.py | 448 | 470| 207 | 26278 | 111534 | 
| 100 | 31 testing/python/collect.py | 1233 | 1250| 130 | 26408 | 111534 | 
| 101 | 31 src/_pytest/cacheprovider.py | 205 | 259| 493 | 26901 | 111534 | 
| 102 | 31 src/_pytest/terminal.py | 956 | 971| 193 | 27094 | 111534 | 
| 103 | 31 testing/python/collect.py | 196 | 249| 320 | 27414 | 111534 | 
| 104 | 31 src/_pytest/terminal.py | 235 | 258| 210 | 27624 | 111534 | 
| 105 | 31 testing/python/collect.py | 730 | 753| 170 | 27794 | 111534 | 
| 106 | 31 testing/python/collect.py | 174 | 194| 127 | 27921 | 111534 | 
| 107 | 32 testing/example_scripts/issue_519.py | 34 | 52| 115 | 28036 | 112000 | 
| 108 | 32 src/_pytest/terminal.py | 403 | 452| 420 | 28456 | 112000 | 
| 109 | 33 bench/manyparam.py | 1 | 15| 0 | 28456 | 112041 | 
| 110 | 33 src/_pytest/cacheprovider.py | 262 | 299| 320 | 28776 | 112041 | 
| 111 | 34 testing/python/fixtures.py | 1067 | 2072| 6201 | 34977 | 138347 | 
| 112 | 35 setup.py | 1 | 16| 156 | 35133 | 138618 | 
| 113 | 36 src/_pytest/logging.py | 172 | 259| 584 | 35717 | 144253 | 
| 114 | 36 src/_pytest/terminal.py | 492 | 525| 302 | 36019 | 144253 | 
| 115 | 36 testing/python/collect.py | 1146 | 1172| 181 | 36200 | 144253 | 
| 116 | 36 testing/python/metafunc.py | 1750 | 1787| 253 | 36453 | 144253 | 
| 117 | 36 testing/python/collect.py | 1 | 33| 225 | 36678 | 144253 | 
| 118 | 36 src/_pytest/capture.py | 242 | 264| 209 | 36887 | 144253 | 
| 119 | 37 src/_pytest/compat.py | 85 | 101| 128 | 37015 | 146823 | 
| 120 | 38 doc/en/example/pythoncollection.py | 1 | 15| 0 | 37015 | 146870 | 
| 121 | 38 setup.py | 19 | 40| 115 | 37130 | 146870 | 
| 122 | 38 src/_pytest/python.py | 1205 | 1231| 208 | 37338 | 146870 | 
| 123 | 39 doc/en/example/nonpython/conftest.py | 1 | 47| 314 | 37652 | 147184 | 
| 124 | 39 src/_pytest/python.py | 1183 | 1202| 183 | 37835 | 147184 | 
| 125 | 39 src/_pytest/doctest.py | 210 | 247| 315 | 38150 | 147184 | 
| 126 | 39 testing/python/metafunc.py | 1541 | 1560| 168 | 38318 | 147184 | 
| 127 | 40 src/_pytest/_code/code.py | 1 | 32| 164 | 38482 | 155509 | 
| 128 | 41 testing/python/integration.py | 1 | 35| 239 | 38721 | 158434 | 
| 129 | 41 src/_pytest/config/argparsing.py | 332 | 356| 201 | 38922 | 158434 | 
| 130 | 41 src/_pytest/warnings.py | 1 | 29| 212 | 39134 | 158434 | 
| 131 | **41 src/_pytest/main.py** | 584 | 604| 181 | 39315 | 158434 | 
| 132 | 41 testing/python/collect.py | 474 | 493| 156 | 39471 | 158434 | 
| 133 | 41 src/_pytest/assertion/truncate.py | 37 | 73| 336 | 39807 | 158434 | 
| 134 | 41 src/_pytest/config/__init__.py | 683 | 739| 472 | 40279 | 158434 | 
| 135 | 41 testing/example_scripts/issue_519.py | 1 | 31| 350 | 40629 | 158434 | 
| 136 | 41 src/_pytest/doctest.py | 101 | 149| 336 | 40965 | 158434 | 
| 137 | 42 src/_pytest/nodes.py | 166 | 190| 134 | 41099 | 161873 | 
| 138 | 42 testing/python/metafunc.py | 349 | 363| 144 | 41243 | 161873 | 
| 139 | 42 testing/python/collect.py | 79 | 113| 255 | 41498 | 161873 | 
| 140 | 42 src/_pytest/mark/__init__.py | 96 | 120| 159 | 41657 | 161873 | 
| 141 | 43 src/_pytest/reports.py | 329 | 352| 174 | 41831 | 165343 | 
| 142 | 43 src/_pytest/junitxml.py | 233 | 247| 132 | 41963 | 165343 | 
| 143 | 43 src/_pytest/python.py | 739 | 780| 334 | 42297 | 165343 | 
| 144 | 43 src/_pytest/terminal.py | 477 | 490| 149 | 42446 | 165343 | 
| 145 | 43 src/_pytest/terminal.py | 1083 | 1104| 171 | 42617 | 165343 | 
| 146 | 43 src/_pytest/runner.py | 244 | 273| 269 | 42886 | 165343 | 
| 147 | 43 src/_pytest/doctest.py | 328 | 344| 133 | 43019 | 165343 | 
| 148 | 43 src/_pytest/config/argparsing.py | 358 | 369| 126 | 43145 | 165343 | 
| 149 | 43 testing/python/collect.py | 346 | 364| 152 | 43297 | 165343 | 
| 150 | 43 testing/python/fixtures.py | 85 | 1065| 6303 | 49600 | 165343 | 
| 151 | 44 src/_pytest/pastebin.py | 1 | 38| 268 | 49868 | 166148 | 
| 152 | 44 src/_pytest/skipping.py | 68 | 87| 167 | 50035 | 166148 | 
| 153 | 44 src/_pytest/compat.py | 282 | 377| 598 | 50633 | 166148 | 
| 154 | 44 src/_pytest/nodes.py | 1 | 33| 222 | 50855 | 166148 | 
| 155 | 44 testing/python/fixtures.py | 2074 | 2473| 2558 | 53413 | 166148 | 
| 156 | 45 src/_pytest/hookspec.py | 280 | 311| 229 | 53642 | 170541 | 
| 157 | 45 src/_pytest/terminal.py | 689 | 705| 122 | 53764 | 170541 | 
| 158 | 45 src/_pytest/capture.py | 740 | 762| 220 | 53984 | 170541 | 
| 159 | 45 testing/python/metafunc.py | 1522 | 1539| 152 | 54136 | 170541 | 
| 160 | 45 src/_pytest/reports.py | 37 | 133| 613 | 54749 | 170541 | 
| 161 | 46 testing/example_scripts/fixtures/fill_fixtures/test_conftest_funcargs_only_available_in_subdir/sub1/conftest.py | 1 | 8| 0 | 54749 | 170567 | 
| 162 | 46 testing/python/metafunc.py | 1562 | 1577| 129 | 54878 | 170567 | 
| 163 | 47 bench/skip.py | 1 | 10| 0 | 54878 | 170602 | 
| 164 | 47 testing/conftest.py | 96 | 121| 162 | 55040 | 170602 | 
| 165 | 47 src/_pytest/mark/structures.py | 1 | 34| 188 | 55228 | 170602 | 
| 166 | 47 src/_pytest/terminal.py | 1 | 29| 125 | 55353 | 170602 | 
| 167 | 47 src/_pytest/pytester.py | 344 | 380| 237 | 55590 | 170602 | 
| 168 | 48 doc/en/example/costlysetup/conftest.py | 1 | 21| 0 | 55590 | 170682 | 
| 169 | 49 src/_pytest/fixtures.py | 169 | 203| 327 | 55917 | 182430 | 
| 170 | 49 src/_pytest/_code/code.py | 1029 | 1049| 153 | 56070 | 182430 | 
| 171 | 49 src/_pytest/runner.py | 122 | 136| 136 | 56206 | 182430 | 
| 172 | 49 src/_pytest/fixtures.py | 478 | 505| 208 | 56414 | 182430 | 
| 173 | 49 testing/python/metafunc.py | 1611 | 1625| 127 | 56541 | 182430 | 
| 174 | 49 src/_pytest/compat.py | 161 | 199| 244 | 56785 | 182430 | 
| 175 | 49 src/_pytest/capture.py | 280 | 294| 140 | 56925 | 182430 | 
| 176 | 49 testing/python/collect.py | 1253 | 1263| 110 | 57035 | 182430 | 
| 177 | 49 src/_pytest/python.py | 1139 | 1167| 266 | 57301 | 182430 | 
| 178 | 49 testing/python/fixtures.py | 2475 | 3503| 6208 | 63509 | 182430 | 
| 179 | 50 src/_pytest/report_log.py | 1 | 29| 167 | 63676 | 182918 | 
| 180 | 50 testing/python/integration.py | 415 | 437| 140 | 63816 | 182918 | 
| 181 | 50 src/_pytest/pastebin.py | 83 | 104| 181 | 63997 | 182918 | 
| 182 | 50 src/_pytest/python.py | 1170 | 1180| 122 | 64119 | 182918 | 
| 183 | 50 testing/python/fixtures.py | 3505 | 4210| 4568 | 68687 | 182918 | 
| 184 | 50 src/_pytest/capture.py | 297 | 311| 144 | 68831 | 182918 | 
| 185 | 50 src/_pytest/python.py | 340 | 355| 146 | 68977 | 182918 | 
| 186 | 50 src/_pytest/pytester.py | 937 | 956| 191 | 69168 | 182918 | 
| 187 | 50 testing/python/metafunc.py | 1627 | 1641| 132 | 69300 | 182918 | 
| 188 | 50 src/_pytest/terminal.py | 312 | 333| 165 | 69465 | 182918 | 
| 189 | 50 src/_pytest/hookspec.py | 262 | 277| 119 | 69584 | 182918 | 
| 190 | 50 src/_pytest/python.py | 715 | 737| 217 | 69801 | 182918 | 
| 191 | 50 src/_pytest/reports.py | 306 | 326| 155 | 69956 | 182918 | 
| 192 | 50 testing/python/metafunc.py | 836 | 876| 313 | 70269 | 182918 | 
| 193 | 50 src/_pytest/capture.py | 39 | 60| 193 | 70462 | 182918 | 
| 194 | 50 src/_pytest/doctest.py | 190 | 207| 172 | 70634 | 182918 | 
| 195 | 50 testing/python/metafunc.py | 1706 | 1729| 222 | 70856 | 182918 | 
| 196 | 50 src/_pytest/skipping.py | 120 | 178| 530 | 71386 | 182918 | 
| 197 | 50 src/_pytest/terminal.py | 887 | 922| 304 | 71690 | 182918 | 
| 198 | 50 src/_pytest/pytester.py | 154 | 195| 293 | 71983 | 182918 | 
| 199 | **50 src/_pytest/main.py** | 502 | 582| 723 | 72706 | 182918 | 
| 200 | 50 src/_pytest/debugging.py | 127 | 163| 277 | 72983 | 182918 | 
| 201 | 50 src/_pytest/pytester.py | 496 | 584| 735 | 73718 | 182918 | 
| 202 | 50 src/_pytest/terminal.py | 454 | 475| 219 | 73937 | 182918 | 
| 203 | 51 src/_pytest/assertion/util.py | 408 | 444| 289 | 74226 | 186608 | 
| 204 | 51 src/_pytest/doctest.py | 152 | 187| 279 | 74505 | 186608 | 
| 205 | 51 src/_pytest/terminal.py | 353 | 366| 123 | 74628 | 186608 | 
| 206 | 51 src/_pytest/capture.py | 684 | 737| 298 | 74926 | 186608 | 
| 207 | **51 src/_pytest/main.py** | 666 | 715| 425 | 75351 | 186608 | 
| 208 | 51 testing/python/metafunc.py | 1731 | 1748| 127 | 75478 | 186608 | 
| 209 | 51 src/_pytest/mark/structures.py | 113 | 146| 309 | 75787 | 186608 | 
| 210 | 52 bench/empty.py | 1 | 3| 0 | 75787 | 186630 | 
| 211 | 53 src/_pytest/config/findpaths.py | 1 | 59| 396 | 76183 | 187788 | 
| 212 | 53 testing/python/metafunc.py | 178 | 203| 234 | 76417 | 187788 | 
| 213 | 53 src/_pytest/terminal.py | 810 | 834| 203 | 76620 | 187788 | 
| 214 | 53 src/_pytest/config/__init__.py | 231 | 241| 154 | 76774 | 187788 | 
| 215 | 53 src/_pytest/python.py | 603 | 629| 215 | 76989 | 187788 | 
| 216 | 53 testing/python/metafunc.py | 807 | 834| 210 | 77199 | 187788 | 
| 217 | 53 src/_pytest/assertion/util.py | 350 | 364| 118 | 77317 | 187788 | 
| 218 | 53 testing/python/collect.py | 131 | 172| 251 | 77568 | 187788 | 
| 219 | 53 testing/python/metafunc.py | 1295 | 1333| 243 | 77811 | 187788 | 
| 220 | 53 src/_pytest/terminal.py | 668 | 687| 153 | 77964 | 187788 | 
| 221 | 53 src/_pytest/terminal.py | 859 | 885| 228 | 78192 | 187788 | 
| 222 | 53 src/_pytest/doctest.py | 527 | 552| 248 | 78440 | 187788 | 
| 223 | 54 src/_pytest/unittest.py | 242 | 283| 286 | 78726 | 189785 | 
| 224 | 54 testing/python/collect.py | 59 | 77| 189 | 78915 | 189785 | 
| 225 | 54 testing/python/collect.py | 690 | 710| 133 | 79048 | 189785 | 
| 226 | 55 testing/example_scripts/fixtures/fill_fixtures/test_conftest_funcargs_only_available_in_subdir/sub2/conftest.py | 1 | 7| 0 | 79048 | 189810 | 
| 227 | 56 doc/en/_themes/flask_theme_support.py | 1 | 88| 1273 | 80321 | 191083 | 
| 228 | 56 src/_pytest/_code/code.py | 925 | 966| 300 | 80621 | 191083 | 
| 229 | 56 src/_pytest/_code/code.py | 1069 | 1079| 138 | 80759 | 191083 | 
| 230 | 56 src/_pytest/doctest.py | 554 | 581| 284 | 81043 | 191083 | 
| 231 | 56 src/_pytest/config/argparsing.py | 108 | 128| 209 | 81252 | 191083 | 
| 232 | 56 src/_pytest/config/__init__.py | 925 | 964| 300 | 81552 | 191083 | 
| 233 | 56 src/_pytest/faulthandler.py | 54 | 87| 205 | 81757 | 191083 | 
| 234 | 56 src/_pytest/config/__init__.py | 315 | 330| 158 | 81915 | 191083 | 


### Hint

```
Agreed, it's probably the option I use most which doesn't have a shortcut.

Both `-c` and `-o` are taken. I guess `-n` (as in "no action", compare `-n`/`--dry-run` for e.g. `git clean`) could work? 

Maybe `--co` (for either "**co**llect" or "**c**ollect **o**nly), similar to other two-character shortcuts we already have (`--sw`, `--lf`, `--ff`, `--nf`)?
I like `--co`, and it doesn't seem to be used by any plugins as far as I can search:

https://github.com/search?utf8=%E2%9C%93&q=--co+language%3APython+pytest+language%3APython+language%3APython&type=Code&ref=advsearch&l=Python&l=Python
> I find myself needing to run `--collect-only` very often and that cli argument is a very long to type one.

Just out of curiosity: Why?  (i.e. what's your use case?)

+0 for `--co`.

But in general you can easily also have an alias "alias pco='pytest --collect-only'" - (or "alias pco='p --collect-only" if you have a shortcut for pytest already.. :))
I routinely use `--collect-only` when I switch to a different development branch or start working on a different area of our code base. I think `--co` is fine.
Agreed, it's probably the option I use most which doesn't have a shortcut.

Both `-c` and `-o` are taken. I guess `-n` (as in "no action", compare `-n`/`--dry-run` for e.g. `git clean`) could work? 

Maybe `--co` (for either "**co**llect" or "**c**ollect **o**nly), similar to other two-character shortcuts we already have (`--sw`, `--lf`, `--ff`, `--nf`)?
I like `--co`, and it doesn't seem to be used by any plugins as far as I can search:

https://github.com/search?utf8=%E2%9C%93&q=--co+language%3APython+pytest+language%3APython+language%3APython&type=Code&ref=advsearch&l=Python&l=Python
> I find myself needing to run `--collect-only` very often and that cli argument is a very long to type one.

Just out of curiosity: Why?  (i.e. what's your use case?)

+0 for `--co`.

But in general you can easily also have an alias "alias pco='pytest --collect-only'" - (or "alias pco='p --collect-only" if you have a shortcut for pytest already.. :))
I routinely use `--collect-only` when I switch to a different development branch or start working on a different area of our code base. I think `--co` is fine.
```

## Patch

```diff
diff --git a/src/_pytest/main.py b/src/_pytest/main.py
--- a/src/_pytest/main.py
+++ b/src/_pytest/main.py
@@ -109,6 +109,7 @@ def pytest_addoption(parser):
     group.addoption(
         "--collectonly",
         "--collect-only",
+        "--co",
         action="store_true",
         help="only collect tests, don't execute them.",
     ),

```

## Test Patch

```diff
diff --git a/testing/test_collection.py b/testing/test_collection.py
--- a/testing/test_collection.py
+++ b/testing/test_collection.py
@@ -402,7 +402,7 @@ def pytest_collect_file(path, parent):
         )
         testdir.mkdir("sub")
         testdir.makepyfile("def test_x(): pass")
-        result = testdir.runpytest("--collect-only")
+        result = testdir.runpytest("--co")
         result.stdout.fnmatch_lines(["*MyModule*", "*test_x*"])
 
     def test_pytest_collect_file_from_sister_dir(self, testdir):
@@ -433,7 +433,7 @@ def pytest_collect_file(path, parent):
         p = testdir.makepyfile("def test_x(): pass")
         p.copy(sub1.join(p.basename))
         p.copy(sub2.join(p.basename))
-        result = testdir.runpytest("--collect-only")
+        result = testdir.runpytest("--co")
         result.stdout.fnmatch_lines(["*MyModule1*", "*MyModule2*", "*test_x*"])
 
 

```


## Code snippets

### 1 - src/_pytest/main.py:

Start line: 46, End line: 154

```python
def pytest_addoption(parser):
    parser.addini(
        "norecursedirs",
        "directory patterns to avoid for recursion",
        type="args",
        default=[".*", "build", "dist", "CVS", "_darcs", "{arch}", "*.egg", "venv"],
    )
    parser.addini(
        "testpaths",
        "directories to search for tests when no files or directories are given in the "
        "command line.",
        type="args",
        default=[],
    )
    group = parser.getgroup("general", "running and selection options")
    group._addoption(
        "-x",
        "--exitfirst",
        action="store_const",
        dest="maxfail",
        const=1,
        help="exit instantly on first error or failed test.",
    ),
    group._addoption(
        "--maxfail",
        metavar="num",
        action="store",
        type=int,
        dest="maxfail",
        default=0,
        help="exit after first num failures or errors.",
    )
    group._addoption(
        "--strict-markers",
        "--strict",
        action="store_true",
        help="markers not registered in the `markers` section of the configuration file raise errors.",
    )
    group._addoption(
        "-c",
        metavar="file",
        type=str,
        dest="inifilename",
        help="load configuration from `file` instead of trying to locate one of the implicit "
        "configuration files.",
    )
    group._addoption(
        "--continue-on-collection-errors",
        action="store_true",
        default=False,
        dest="continue_on_collection_errors",
        help="Force test execution even if collection errors occur.",
    )
    group._addoption(
        "--rootdir",
        action="store",
        dest="rootdir",
        help="Define root directory for tests. Can be relative path: 'root_dir', './root_dir', "
        "'root_dir/another_dir/'; absolute path: '/home/user/root_dir'; path with variables: "
        "'$HOME/root_dir'.",
    )

    group = parser.getgroup("collect", "collection")
    group.addoption(
        "--collectonly",
        "--collect-only",
        action="store_true",
        help="only collect tests, don't execute them.",
    ),
    group.addoption(
        "--pyargs",
        action="store_true",
        help="try to interpret all arguments as python packages.",
    )
    group.addoption(
        "--ignore",
        action="append",
        metavar="path",
        help="ignore path during collection (multi-allowed).",
    )
    group.addoption(
        "--ignore-glob",
        action="append",
        metavar="path",
        help="ignore path pattern during collection (multi-allowed).",
    )
    group.addoption(
        "--deselect",
        action="append",
        metavar="nodeid_prefix",
        help="deselect item during collection (multi-allowed).",
    )
    # when changing this to --conf-cut-dir, config.py Conftest.setinitial
    # needs upgrading as well
    group.addoption(
        "--confcutdir",
        dest="confcutdir",
        default=None,
        metavar="dir",
        type=functools.partial(directory_arg, optname="--confcutdir"),
        help="only load conftest.py's relative to specified dir.",
    )
    group.addoption(
        "--noconftest",
        action="store_true",
        dest="noconftest",
        default=False,
        help="Don't load any conftest.py files.",
    )
    # ... other code
```
### 2 - src/_pytest/terminal.py:

Start line: 57, End line: 148

```python
def pytest_addoption(parser):
    group = parser.getgroup("terminal reporting", "reporting", after="general")
    group._addoption(
        "-v",
        "--verbose",
        action="count",
        default=0,
        dest="verbose",
        help="increase verbosity.",
    ),
    group._addoption(
        "-q",
        "--quiet",
        action=MoreQuietAction,
        default=0,
        dest="verbose",
        help="decrease verbosity.",
    ),
    group._addoption(
        "--verbosity",
        dest="verbose",
        type=int,
        default=0,
        help="set verbosity. Default is 0.",
    )
    group._addoption(
        "-r",
        action="store",
        dest="reportchars",
        default="",
        metavar="chars",
        help="show extra test summary info as specified by chars: (f)ailed, "
        "(E)rror, (s)kipped, (x)failed, (X)passed, "
        "(p)assed, (P)assed with output, (a)ll except passed (p/P), or (A)ll. "
        "(w)arnings are enabled by default (see --disable-warnings).",
    )
    group._addoption(
        "--disable-warnings",
        "--disable-pytest-warnings",
        default=False,
        dest="disable_warnings",
        action="store_true",
        help="disable warnings summary",
    )
    group._addoption(
        "-l",
        "--showlocals",
        action="store_true",
        dest="showlocals",
        default=False,
        help="show locals in tracebacks (disabled by default).",
    )
    group._addoption(
        "--tb",
        metavar="style",
        action="store",
        dest="tbstyle",
        default="auto",
        choices=["auto", "long", "short", "no", "line", "native"],
        help="traceback print mode (auto/long/short/line/native/no).",
    )
    group._addoption(
        "--show-capture",
        action="store",
        dest="showcapture",
        choices=["no", "stdout", "stderr", "log", "all"],
        default="all",
        help="Controls how captured stdout/stderr/log is shown on failed tests. "
        "Default is 'all'.",
    )
    group._addoption(
        "--fulltrace",
        "--full-trace",
        action="store_true",
        default=False,
        help="don't cut any tracebacks (default is to cut).",
    )
    group._addoption(
        "--color",
        metavar="color",
        action="store",
        dest="color",
        default="auto",
        choices=["yes", "no", "auto"],
        help="color terminal output (yes/no/auto).",
    )

    parser.addini(
        "console_output_style",
        help='console output: "classic", or with additional progress information ("progress" (percentage) | "count").',
        default="progress",
    )
```
### 3 - src/_pytest/terminal.py:

Start line: 32, End line: 54

```python
class MoreQuietAction(argparse.Action):
    """
    a modified copy of the argparse count action which counts down and updates
    the legacy quiet attribute at the same time

    used to unify verbosity handling
    """

    def __init__(self, option_strings, dest, default=None, required=False, help=None):
        super().__init__(
            option_strings=option_strings,
            dest=dest,
            nargs=0,
            default=default,
            required=required,
            help=help,
        )

    def __call__(self, parser, namespace, values, option_string=None):
        new_count = getattr(namespace, self.dest, 0) - 1
        setattr(namespace, self.dest, new_count)
        # todo Deprecate config.quiet
        namespace.quiet = getattr(namespace, "quiet", 0) + 1
```
### 4 - src/_pytest/main.py:

Start line: 155, End line: 181

```python
def pytest_addoption(parser):
    # ... other code
    group.addoption(
        "--keepduplicates",
        "--keep-duplicates",
        action="store_true",
        dest="keepduplicates",
        default=False,
        help="Keep duplicate tests.",
    )
    group.addoption(
        "--collect-in-virtualenv",
        action="store_true",
        dest="collect_in_virtualenv",
        default=False,
        help="Don't ignore tests in a local virtualenv directory",
    )

    group = parser.getgroup("debugconfig", "test session debugging and configuration")
    group.addoption(
        "--basetemp",
        dest="basetemp",
        default=None,
        metavar="dir",
        help=(
            "base temporary directory for this test run."
            "(warning: this directory is removed if it exists)"
        ),
    )
```
### 5 - src/_pytest/mark/__init__.py:

Start line: 37, End line: 75

```python
def pytest_addoption(parser):
    group = parser.getgroup("general")
    group._addoption(
        "-k",
        action="store",
        dest="keyword",
        default="",
        metavar="EXPRESSION",
        help="only run tests which match the given substring expression. "
        "An expression is a python evaluatable expression "
        "where all names are substring-matched against test names "
        "and their parent classes. Example: -k 'test_method or test_"
        "other' matches all test functions and classes whose name "
        "contains 'test_method' or 'test_other', while -k 'not test_method' "
        "matches those that don't contain 'test_method' in their names. "
        "-k 'not test_method and not test_other' will eliminate the matches. "
        "Additionally keywords are matched to classes and functions "
        "containing extra names in their 'extra_keyword_matches' set, "
        "as well as functions which have names assigned directly to them.",
    )

    group._addoption(
        "-m",
        action="store",
        dest="markexpr",
        default="",
        metavar="MARKEXPR",
        help="only run tests matching given mark expression.  "
        "example: -m 'mark1 and not mark2'.",
    )

    group.addoption(
        "--markers",
        action="store_true",
        help="show markers (builtin, plugin and per-project ones).",
    )

    parser.addini("markers", "markers for test functions", "linelist")
    parser.addini(EMPTY_PARAMETERSET_OPTION, "default marker for empty parametersets")
```
### 6 - src/_pytest/config/argparsing.py:

Start line: 424, End line: 464

```python
class DropShorterLongHelpFormatter(argparse.HelpFormatter):

    def _format_action_invocation(self, action):
        orgstr = argparse.HelpFormatter._format_action_invocation(self, action)
        if orgstr and orgstr[0] != "-":  # only optional arguments
            return orgstr
        res = getattr(action, "_formatted_action_invocation", None)
        if res:
            return res
        options = orgstr.split(", ")
        if len(options) == 2 and (len(options[0]) == 2 or len(options[1]) == 2):
            # a shortcut for '-h, --help' or '--abc', '-a'
            action._formatted_action_invocation = orgstr
            return orgstr
        return_list = []
        option_map = getattr(action, "map_long_option", {})
        if option_map is None:
            option_map = {}
        short_long = {}  # type: Dict[str, str]
        for option in options:
            if len(option) == 2 or option[2] == " ":
                continue
            if not option.startswith("--"):
                raise ArgumentError(
                    'long optional argument without "--": [%s]' % (option), self
                )
            xxoption = option[2:]
            if xxoption.split()[0] not in option_map:
                shortened = xxoption.replace("-", "")
                if shortened not in short_long or len(short_long[shortened]) < len(
                    xxoption
                ):
                    short_long[shortened] = xxoption
        # now short_long has been filled out to the longest with dashes
        # **and** we keep the right option ordering from add_argument
        for option in options:
            if len(option) == 2 or option[2] == " ":
                return_list.append(option)
            if option[2:] == short_long.get(option.replace("-", "")):
                return_list.append(option.replace(" ", "=", 1))
        action._formatted_action_invocation = ", ".join(return_list)
        return action._formatted_action_invocation
```
### 7 - src/_pytest/config/argparsing.py:

Start line: 408, End line: 422

```python
class DropShorterLongHelpFormatter(argparse.HelpFormatter):
    """shorten help for long options that differ only in extra hyphens

    - collapse **long** options that are the same except for extra hyphens
    - special action attribute map_long_option allows suppressing additional
      long options
    - shortcut if there are only two options and one of them is a short one
    - cache result on action object as this is called at least 2 times
    """

    def __init__(self, *args, **kwargs):
        """Use more accurate terminal width via pylib."""
        if "width" not in kwargs:
            kwargs["width"] = py.io.get_terminal_width()
        super().__init__(*args, **kwargs)
```
### 8 - src/_pytest/debugging.py:

Start line: 22, End line: 43

```python
def pytest_addoption(parser):
    group = parser.getgroup("general")
    group._addoption(
        "--pdb",
        dest="usepdb",
        action="store_true",
        help="start the interactive Python debugger on errors or KeyboardInterrupt.",
    )
    group._addoption(
        "--pdbcls",
        dest="usepdb_cls",
        metavar="modulename:classname",
        type=_validate_usepdb_cls,
        help="start a custom interactive Python debugger on errors. "
        "For example: --pdbcls=IPython.terminal.debugger:TerminalPdb",
    )
    group._addoption(
        "--trace",
        dest="trace",
        action="store_true",
        help="Immediately break when running each test.",
    )
```
### 9 - src/_pytest/junitxml.py:

Start line: 383, End line: 427

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
### 10 - src/_pytest/terminal.py:

Start line: 1061, End line: 1080

```python
_color_for_type = {
    "failed": "red",
    "error": "red",
    "warnings": "yellow",
    "passed": "green",
}
_color_for_type_default = "yellow"


def _make_plural(count, noun):
    # No need to pluralize words such as `failed` or `passed`.
    if noun not in ["error", "warnings"]:
        return count, noun

    # The `warnings` key is plural. To avoid API breakage, we keep it that way but
    # set it to singular here so we can determine plurality in the same way as we do
    # for `error`.
    noun = noun.replace("warnings", "warning")

    return count, noun + "s" if count != 1 else noun
```
### 21 - src/_pytest/main.py:

Start line: 316, End line: 367

```python
def pytest_collection_modifyitems(items, config):
    deselect_prefixes = tuple(config.getoption("deselect") or [])
    if not deselect_prefixes:
        return

    remaining = []
    deselected = []
    for colitem in items:
        if colitem.nodeid.startswith(deselect_prefixes):
            deselected.append(colitem)
        else:
            remaining.append(colitem)

    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = remaining


class FSHookProxy:
    def __init__(self, fspath, pm, remove_mods):
        self.fspath = fspath
        self.pm = pm
        self.remove_mods = remove_mods

    def __getattr__(self, name):
        x = self.pm.subset_hook_caller(name, remove_plugins=self.remove_mods)
        self.__dict__[name] = x
        return x


class NoMatch(Exception):
    """ raised if matching cannot locate a matching names. """


class Interrupted(KeyboardInterrupt):
    """ signals an interrupted test run. """

    __module__ = "builtins"  # for py3


class Failed(Exception):
    """ signals a stop as failed test run. """


@attr.s
class _bestrelpath_cache(dict):
    path = attr.ib()

    def __missing__(self, path: str) -> str:
        r = self.path.bestrelpath(path)  # type: str
        self[path] = r
        return r
```
### 35 - src/_pytest/main.py:

Start line: 288, End line: 313

```python
def pytest_ignore_collect(path, config):
    ignore_paths = config._getconftest_pathlist("collect_ignore", path=path.dirpath())
    ignore_paths = ignore_paths or []
    excludeopt = config.getoption("ignore")
    if excludeopt:
        ignore_paths.extend([py.path.local(x) for x in excludeopt])

    if py.path.local(path) in ignore_paths:
        return True

    ignore_globs = config._getconftest_pathlist(
        "collect_ignore_glob", path=path.dirpath()
    )
    ignore_globs = ignore_globs or []
    excludeglobopt = config.getoption("ignore_glob")
    if excludeglobopt:
        ignore_globs.extend([py.path.local(x) for x in excludeglobopt])

    if any(fnmatch.fnmatch(str(path), str(glob)) for glob in ignore_globs):
        return True

    allow_in_venv = config.getoption("collect_in_virtualenv")
    if not allow_in_venv and _in_venv(path):
        return True

    return False
```
### 48 - src/_pytest/main.py:

Start line: 231, End line: 268

```python
def pytest_cmdline_main(config):
    return wrap_session(config, _main)


def _main(config, session):
    """ default command line protocol for initialization, session,
    running tests and reporting. """
    config.hook.pytest_collection(session=session)
    config.hook.pytest_runtestloop(session=session)

    if session.testsfailed:
        return ExitCode.TESTS_FAILED
    elif session.testscollected == 0:
        return ExitCode.NO_TESTS_COLLECTED


def pytest_collection(session):
    return session.perform_collect()


def pytest_runtestloop(session):
    if session.testsfailed and not session.config.option.continue_on_collection_errors:
        raise session.Interrupted(
            "%d error%s during collection"
            % (session.testsfailed, "s" if session.testsfailed != 1 else "")
        )

    if session.config.option.collectonly:
        return True

    for i, item in enumerate(session.items):
        nextitem = session.items[i + 1] if i + 1 < len(session.items) else None
        item.config.hook.pytest_runtest_protocol(item=item, nextitem=nextitem)
        if session.shouldfail:
            raise session.Failed(session.shouldfail)
        if session.shouldstop:
            raise session.Interrupted(session.shouldstop)
    return True
```
### 73 - src/_pytest/main.py:

Start line: 457, End line: 500

```python
class Session(nodes.FSCollector):

    def _perform_collect(self, args, genitems):
        if args is None:
            args = self.config.args
        self.trace("perform_collect", self, args)
        self.trace.root.indent += 1
        self._notfound = []
        initialpaths = []
        self._initialparts = []
        self.items = items = []
        for arg in args:
            parts = self._parsearg(arg)
            self._initialparts.append(parts)
            initialpaths.append(parts[0])
        self._initialpaths = frozenset(initialpaths)
        rep = collect_one_node(self)
        self.ihook.pytest_collectreport(report=rep)
        self.trace.root.indent -= 1
        if self._notfound:
            errors = []
            for arg, exc in self._notfound:
                line = "(no name {!r} in any of {!r})".format(arg, exc.args[0])
                errors.append("not found: {}\n{}".format(arg, line))
            raise UsageError(*errors)
        if not genitems:
            return rep.result
        else:
            if rep.passed:
                for node in rep.result:
                    self.items.extend(self.genitems(node))
            return items

    def collect(self):
        for initialpart in self._initialparts:
            self.trace("processing argument", initialpart)
            self.trace.root.indent += 1
            try:
                yield from self._collect(initialpart)
            except NoMatch:
                report_arg = "::".join(map(str, initialpart))
                # we are inside a make_report hook so
                # we cannot directly pass through the exception
                self._notfound.append((report_arg, sys.exc_info()[1]))

            self.trace.root.indent -= 1
```
### 93 - src/_pytest/main.py:

Start line: 184, End line: 228

```python
def wrap_session(config, doit):
    """Skeleton command line program"""
    session = Session(config)
    session.exitstatus = ExitCode.OK
    initstate = 0
    try:
        try:
            config._do_configure()
            initstate = 1
            config.hook.pytest_sessionstart(session=session)
            initstate = 2
            session.exitstatus = doit(config, session) or 0
        except UsageError:
            session.exitstatus = ExitCode.USAGE_ERROR
            raise
        except Failed:
            session.exitstatus = ExitCode.TESTS_FAILED
        except (KeyboardInterrupt, exit.Exception):
            excinfo = _pytest._code.ExceptionInfo.from_current()
            exitstatus = ExitCode.INTERRUPTED
            if isinstance(excinfo.value, exit.Exception):
                if excinfo.value.returncode is not None:
                    exitstatus = excinfo.value.returncode
                if initstate < 2:
                    sys.stderr.write(
                        "{}: {}\n".format(excinfo.typename, excinfo.value.msg)
                    )
            config.hook.pytest_keyboard_interrupt(excinfo=excinfo)
            session.exitstatus = exitstatus
        except:  # noqa
            excinfo = _pytest._code.ExceptionInfo.from_current()
            config.notify_exception(excinfo, config.option)
            session.exitstatus = ExitCode.INTERNAL_ERROR
            if excinfo.errisinstance(SystemExit):
                sys.stderr.write("mainloop: caught unexpected SystemExit!\n")

    finally:
        excinfo = None  # Explicitly break reference cycle.
        session.startdir.chdir()
        if initstate >= 2:
            config.hook.pytest_sessionfinish(
                session=session, exitstatus=session.exitstatus
            )
        config._ensure_unconfigure()
    return session.exitstatus
```
### 131 - src/_pytest/main.py:

Start line: 584, End line: 604

```python
class Session(nodes.FSCollector):

    def _collectfile(self, path, handle_dupes=True):
        assert (
            path.isfile()
        ), "{!r} is not a file (isdir={!r}, exists={!r}, islink={!r})".format(
            path, path.isdir(), path.exists(), path.islink()
        )
        ihook = self.gethookproxy(path)
        if not self.isinitpath(path):
            if ihook.pytest_ignore_collect(path=path, config=self.config):
                return ()

        if handle_dupes:
            keepduplicates = self.config.getoption("keepduplicates")
            if not keepduplicates:
                duplicate_paths = self.config.pluginmanager._duplicatepaths
                if path in duplicate_paths:
                    return ()
                else:
                    duplicate_paths.add(path)

        return ihook.pytest_collect_file(path=path, parent=self)
```
### 199 - src/_pytest/main.py:

Start line: 502, End line: 582

```python
class Session(nodes.FSCollector):

    def _collect(self, arg):
        from _pytest.python import Package

        names = arg[:]
        argpath = names.pop(0)

        # Start with a Session root, and delve to argpath item (dir or file)
        # and stack all Packages found on the way.
        # No point in finding packages when collecting doctests
        if not self.config.getoption("doctestmodules", False):
            pm = self.config.pluginmanager
            for parent in reversed(argpath.parts()):
                if pm._confcutdir and pm._confcutdir.relto(parent):
                    break

                if parent.isdir():
                    pkginit = parent.join("__init__.py")
                    if pkginit.isfile():
                        if pkginit not in self._node_cache:
                            col = self._collectfile(pkginit, handle_dupes=False)
                            if col:
                                if isinstance(col[0], Package):
                                    self._pkg_roots[parent] = col[0]
                                # always store a list in the cache, matchnodes expects it
                                self._node_cache[col[0].fspath] = [col[0]]

        # If it's a directory argument, recurse and look for any Subpackages.
        # Let the Package collector deal with subnodes, don't collect here.
        if argpath.check(dir=1):
            assert not names, "invalid arg {!r}".format(arg)

            seen_dirs = set()
            for path in argpath.visit(
                fil=self._visit_filter, rec=self._recurse, bf=True, sort=True
            ):
                dirpath = path.dirpath()
                if dirpath not in seen_dirs:
                    # Collect packages first.
                    seen_dirs.add(dirpath)
                    pkginit = dirpath.join("__init__.py")
                    if pkginit.exists():
                        for x in self._collectfile(pkginit):
                            yield x
                            if isinstance(x, Package):
                                self._pkg_roots[dirpath] = x
                if dirpath in self._pkg_roots:
                    # Do not collect packages here.
                    continue

                for x in self._collectfile(path):
                    key = (type(x), x.fspath)
                    if key in self._node_cache:
                        yield self._node_cache[key]
                    else:
                        self._node_cache[key] = x
                        yield x
        else:
            assert argpath.check(file=1)

            if argpath in self._node_cache:
                col = self._node_cache[argpath]
            else:
                collect_root = self._pkg_roots.get(argpath.dirname, self)
                col = collect_root._collectfile(argpath, handle_dupes=False)
                if col:
                    self._node_cache[argpath] = col
            m = self.matchnodes(col, names)
            # If __init__.py was the only file requested, then the matched node will be
            # the corresponding Package, and the first yielded item will be the __init__
            # Module itself, so just use that. If this special case isn't taken, then all
            # the files in the package will be yielded.
            if argpath.basename == "__init__.py":
                try:
                    yield next(m[0].collect())
                except StopIteration:
                    # The package collects nothing with only an __init__.py
                    # file in it, which gets ignored by the default
                    # "python_files" option.
                    pass
                return
            yield from m
```
### 207 - src/_pytest/main.py:

Start line: 666, End line: 715

```python
class Session(nodes.FSCollector):

    def _matchnodes(self, matching, names):
        if not matching or not names:
            return matching
        name = names[0]
        assert name
        nextnames = names[1:]
        resultnodes = []
        for node in matching:
            if isinstance(node, nodes.Item):
                if not names:
                    resultnodes.append(node)
                continue
            assert isinstance(node, nodes.Collector)
            key = (type(node), node.nodeid)
            if key in self._node_cache:
                rep = self._node_cache[key]
            else:
                rep = collect_one_node(node)
                self._node_cache[key] = rep
            if rep.passed:
                has_matched = False
                for x in rep.result:
                    # TODO: remove parametrized workaround once collection structure contains parametrization
                    if x.name == name or x.name.split("[")[0] == name:
                        resultnodes.extend(self.matchnodes([x], nextnames))
                        has_matched = True
                # XXX accept IDs that don't have "()" for class instances
                if not has_matched and len(rep.result) == 1 and x.name == "()":
                    nextnames.insert(0, name)
                    resultnodes.extend(self.matchnodes([x], nextnames))
            else:
                # report collection failures here to avoid failing to run some test
                # specified in the command line because the module could not be
                # imported (#134)
                node.ihook.pytest_collectreport(report=rep)
        return resultnodes

    def genitems(self, node):
        self.trace("genitems", node)
        if isinstance(node, nodes.Item):
            node.ihook.pytest_itemcollected(item=node)
            yield node
        else:
            assert isinstance(node, nodes.Collector)
            rep = collect_one_node(node)
            if rep.passed:
                for subnode in rep.result:
                    yield from self.genitems(subnode)
            node.ihook.pytest_collectreport(report=rep)
```
