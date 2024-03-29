# pytest-dev__pytest-8022

| **pytest-dev/pytest** | `e986d84466dfa98dbbc55cc1bf5fcb99075f4ac3` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 2174 |
| **Any found context length** | 2174 |
| **Avg pos** | 5.0 |
| **Min pos** | 5 |
| **Max pos** | 5 |
| **Top file pos** | 3 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/src/_pytest/main.py b/src/_pytest/main.py
--- a/src/_pytest/main.py
+++ b/src/_pytest/main.py
@@ -765,12 +765,14 @@ def collect(self) -> Iterator[Union[nodes.Item, nodes.Collector]]:
                     self._notfound.append((report_arg, col))
                     continue
 
-                # If __init__.py was the only file requested, then the matched node will be
-                # the corresponding Package, and the first yielded item will be the __init__
-                # Module itself, so just use that. If this special case isn't taken, then all
-                # the files in the package will be yielded.
-                if argpath.basename == "__init__.py":
-                    assert isinstance(matching[0], nodes.Collector)
+                # If __init__.py was the only file requested, then the matched
+                # node will be the corresponding Package (by default), and the
+                # first yielded item will be the __init__ Module itself, so
+                # just use that. If this special case isn't taken, then all the
+                # files in the package will be yielded.
+                if argpath.basename == "__init__.py" and isinstance(
+                    matching[0], Package
+                ):
                     try:
                         yield next(iter(matching[0].collect()))
                     except StopIteration:

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| src/_pytest/main.py | 768 | 773 | 5 | 3 | 2174


## Problem Statement

```
Doctest collection only returns single test for __init__.py
<!--
Thanks for submitting an issue!

Quick check-list while reporting bugs:
-->

`pytest --doctest-modules __init__.py` will only collect a single doctest because of this:

https://github.com/pytest-dev/pytest/blob/e986d84466dfa98dbbc55cc1bf5fcb99075f4ac3/src/_pytest/main.py#L768-L781

Introduced a while back by @kchmck here: https://github.com/pytest-dev/pytest/commit/5ac4eff09b8514a5b46bdff464605a60051abc83

See failing tests: https://github.com/pytest-dev/pytest/pull/8015

Failing doctest collection
When the module is an __init__.py the doctest collection only picks up 1 doctest.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 src/_pytest/doctest.py | 526 | 552| 237 | 237 | 5620 | 
| 2 | 1 src/_pytest/doctest.py | 410 | 438| 215 | 452 | 5620 | 
| 3 | 2 testing/python/collect.py | 1 | 35| 241 | 693 | 15468 | 
| 4 | 2 testing/python/collect.py | 1351 | 1388| 320 | 1013 | 15468 | 
| **-> 5 <-** | **3 src/_pytest/main.py** | 652 | 785| 1161 | 2174 | 22191 | 
| 6 | 3 src/_pytest/doctest.py | 1 | 62| 411 | 2585 | 22191 | 
| 7 | 3 testing/python/collect.py | 253 | 278| 181 | 2766 | 22191 | 
| 8 | 3 testing/python/collect.py | 1213 | 1239| 179 | 2945 | 22191 | 
| 9 | 3 testing/python/collect.py | 556 | 573| 182 | 3127 | 22191 | 
| 10 | 3 testing/python/collect.py | 197 | 250| 318 | 3445 | 22191 | 
| 11 | 3 testing/python/collect.py | 943 | 982| 352 | 3797 | 22191 | 
| 12 | 3 testing/python/collect.py | 81 | 114| 241 | 4038 | 22191 | 
| 13 | 3 testing/python/collect.py | 908 | 940| 304 | 4342 | 22191 | 
| 14 | 3 testing/python/collect.py | 1318 | 1335| 130 | 4472 | 22191 | 
| 15 | 3 testing/python/collect.py | 1338 | 1348| 110 | 4582 | 22191 | 
| 16 | 3 testing/python/collect.py | 1298 | 1315| 119 | 4701 | 22191 | 
| 17 | 4 src/_pytest/python.py | 496 | 538| 337 | 5038 | 36116 | 
| 18 | 4 src/_pytest/doctest.py | 490 | 524| 297 | 5335 | 36116 | 
| 19 | 4 testing/python/collect.py | 756 | 771| 149 | 5484 | 36116 | 
| 20 | 4 src/_pytest/doctest.py | 269 | 289| 211 | 5695 | 36116 | 
| 21 | 4 testing/python/collect.py | 540 | 554| 150 | 5845 | 36116 | 
| 22 | 4 testing/python/collect.py | 1005 | 1029| 168 | 6013 | 36116 | 
| 23 | 5 src/_pytest/unittest.py | 46 | 59| 127 | 6140 | 38960 | 
| 24 | 5 src/_pytest/doctest.py | 135 | 168| 278 | 6418 | 38960 | 
| 25 | 5 testing/python/collect.py | 61 | 79| 187 | 6605 | 38960 | 
| 26 | 5 src/_pytest/python.py | 700 | 729| 287 | 6892 | 38960 | 
| 27 | 5 testing/python/collect.py | 1242 | 1269| 183 | 7075 | 38960 | 
| 28 | 5 testing/python/collect.py | 132 | 173| 247 | 7322 | 38960 | 
| 29 | 6 testing/python/integration.py | 1 | 40| 274 | 7596 | 41973 | 
| 30 | 6 src/_pytest/python.py | 218 | 250| 342 | 7938 | 41973 | 
| 31 | 6 testing/python/collect.py | 575 | 663| 591 | 8529 | 41973 | 
| 32 | 6 testing/python/collect.py | 280 | 347| 474 | 9003 | 41973 | 
| 33 | 6 src/_pytest/python.py | 412 | 448| 308 | 9311 | 41973 | 
| 34 | 6 testing/python/collect.py | 773 | 796| 170 | 9481 | 41973 | 
| 35 | 6 testing/python/collect.py | 1173 | 1210| 209 | 9690 | 41973 | 
| 36 | **6 src/_pytest/main.py** | 326 | 343| 162 | 9852 | 41973 | 
| 37 | 6 src/_pytest/doctest.py | 303 | 372| 611 | 10463 | 41973 | 
| 38 | 6 testing/python/collect.py | 1391 | 1424| 239 | 10702 | 41973 | 
| 39 | 7 doc/en/example/pythoncollection.py | 1 | 15| 0 | 10702 | 42020 | 
| 40 | 8 src/pytest/collect.py | 1 | 40| 220 | 10922 | 42240 | 
| 41 | 8 testing/python/collect.py | 831 | 844| 122 | 11044 | 42240 | 
| 42 | 9 src/pytest/__init__.py | 1 | 100| 682 | 11726 | 42922 | 
| 43 | 9 src/_pytest/doctest.py | 171 | 200| 230 | 11956 | 42922 | 
| 44 | 9 src/_pytest/python.py | 450 | 493| 389 | 12345 | 42922 | 
| 45 | 10 doc/en/conf.py | 1 | 109| 785 | 13130 | 46076 | 
| 46 | 10 testing/python/collect.py | 116 | 129| 110 | 13240 | 46076 | 
| 47 | 10 testing/python/integration.py | 42 | 74| 246 | 13486 | 46076 | 
| 48 | 10 testing/python/collect.py | 175 | 195| 127 | 13613 | 46076 | 
| 49 | 10 testing/python/collect.py | 1143 | 1170| 183 | 13796 | 46076 | 
| 50 | 10 src/_pytest/python.py | 676 | 698| 208 | 14004 | 46076 | 
| 51 | 10 testing/python/collect.py | 37 | 59| 187 | 14191 | 46076 | 
| 52 | 10 src/_pytest/python.py | 574 | 619| 453 | 14644 | 46076 | 
| 53 | **10 src/_pytest/main.py** | 363 | 387| 237 | 14881 | 46076 | 
| 54 | 10 testing/python/collect.py | 704 | 731| 258 | 15139 | 46076 | 
| 55 | **10 src/_pytest/main.py** | 390 | 405| 112 | 15251 | 46076 | 
| 56 | 10 src/_pytest/doctest.py | 242 | 267| 216 | 15467 | 46076 | 
| 57 | 11 testing/example_scripts/issue_519.py | 1 | 33| 370 | 15837 | 46558 | 
| 58 | 11 src/_pytest/python.py | 346 | 362| 142 | 15979 | 46558 | 
| 59 | 11 testing/python/collect.py | 1101 | 1120| 167 | 16146 | 46558 | 
| 60 | 11 src/_pytest/python.py | 850 | 882| 242 | 16388 | 46558 | 
| 61 | 12 src/_pytest/pytester.py | 1008 | 1017| 133 | 16521 | 61938 | 
| 62 | 12 src/_pytest/python.py | 189 | 202| 119 | 16640 | 61938 | 
| 63 | 12 testing/python/collect.py | 733 | 753| 133 | 16773 | 61938 | 
| 64 | **12 src/_pytest/main.py** | 533 | 567| 300 | 17073 | 61938 | 
| 65 | 13 testing/python/fixtures.py | 4077 | 4367| 1868 | 18941 | 89215 | 
| 66 | 13 testing/python/collect.py | 1272 | 1295| 195 | 19136 | 89215 | 
| 67 | 13 testing/python/collect.py | 984 | 1003| 178 | 19314 | 89215 | 
| 68 | 13 src/_pytest/python.py | 1398 | 1430| 260 | 19574 | 89215 | 
| 69 | 13 src/_pytest/unittest.py | 1 | 43| 270 | 19844 | 89215 | 
| 70 | **13 src/_pytest/main.py** | 569 | 634| 569 | 20413 | 89215 | 
| 71 | 13 src/_pytest/unittest.py | 322 | 368| 347 | 20760 | 89215 | 
| 72 | 14 doc/en/example/xfail_demo.py | 1 | 39| 143 | 20903 | 89359 | 
| 73 | 14 src/_pytest/pytester.py | 1 | 71| 432 | 21335 | 89359 | 
| 74 | 14 testing/python/collect.py | 688 | 701| 124 | 21459 | 89359 | 
| 75 | 14 src/_pytest/python.py | 1432 | 1457| 259 | 21718 | 89359 | 
| 76 | 14 src/_pytest/doctest.py | 202 | 219| 167 | 21885 | 89359 | 
| 77 | 14 src/_pytest/python.py | 753 | 788| 251 | 22136 | 89359 | 
| 78 | 14 src/_pytest/python.py | 802 | 817| 159 | 22295 | 89359 | 
| 79 | 15 testing/python/metafunc.py | 1356 | 1378| 230 | 22525 | 103948 | 
| 80 | **15 src/_pytest/main.py** | 305 | 323| 139 | 22664 | 103948 | 
| 81 | 15 src/_pytest/python.py | 831 | 847| 166 | 22830 | 103948 | 
| 82 | 16 src/_pytest/runner.py | 340 | 368| 316 | 23146 | 107679 | 
| 83 | 17 src/_pytest/reports.py | 342 | 385| 306 | 23452 | 111929 | 
| 84 | 17 testing/python/collect.py | 846 | 877| 270 | 23722 | 111929 | 
| 85 | 17 src/_pytest/python.py | 363 | 395| 306 | 24028 | 111929 | 
| 86 | **17 src/_pytest/main.py** | 636 | 650| 187 | 24215 | 111929 | 
| 87 | 17 src/_pytest/doctest.py | 115 | 132| 144 | 24359 | 111929 | 
| 88 | 17 testing/python/collect.py | 879 | 905| 173 | 24532 | 111929 | 
| 89 | **17 src/_pytest/main.py** | 1 | 48| 284 | 24816 | 111929 | 
| 90 | 17 src/_pytest/python.py | 1460 | 1519| 451 | 25267 | 111929 | 
| 91 | 17 testing/python/collect.py | 1084 | 1099| 133 | 25400 | 111929 | 
| 92 | 18 src/_pytest/config/__init__.py | 1203 | 1227| 212 | 25612 | 124454 | 
| 93 | 18 testing/python/collect.py | 798 | 829| 204 | 25816 | 124454 | 
| 94 | 18 testing/python/integration.py | 371 | 388| 134 | 25950 | 124454 | 
| 95 | 19 testing/conftest.py | 1 | 64| 384 | 26334 | 125888 | 
| 96 | 20 src/_pytest/outcomes.py | 48 | 63| 124 | 26458 | 127603 | 
| 97 | 20 src/_pytest/doctest.py | 441 | 457| 148 | 26606 | 127603 | 
| 98 | 20 testing/python/integration.py | 331 | 369| 237 | 26843 | 127603 | 
| 99 | 21 doc/en/example/nonpython/conftest.py | 1 | 17| 116 | 26959 | 127925 | 
| 100 | 21 src/_pytest/doctest.py | 65 | 112| 326 | 27285 | 127925 | 
| 101 | 22 src/_pytest/hookspec.py | 202 | 241| 335 | 27620 | 134655 | 
| 102 | 23 doc/en/example/assertion/failure_demo.py | 163 | 202| 270 | 27890 | 136304 | 
| 103 | 23 src/_pytest/hookspec.py | 337 | 356| 168 | 28058 | 136304 | 
| 104 | 23 testing/python/collect.py | 498 | 538| 294 | 28352 | 136304 | 
| 105 | 23 testing/python/fixtures.py | 1 | 84| 498 | 28850 | 136304 | 
| 106 | 24 testing/example_scripts/config/collect_pytest_prefix/conftest.py | 1 | 3| 0 | 28850 | 136312 | 
| 107 | 24 testing/python/integration.py | 272 | 288| 121 | 28971 | 136312 | 
| 108 | 25 extra/get_issues.py | 33 | 52| 152 | 29123 | 136866 | 
| 109 | 25 testing/example_scripts/issue_519.py | 36 | 54| 111 | 29234 | 136866 | 
| 110 | 25 testing/python/collect.py | 665 | 686| 121 | 29355 | 136866 | 
| 111 | 25 src/_pytest/unittest.py | 62 | 111| 445 | 29800 | 136866 | 
| 112 | **25 src/_pytest/main.py** | 51 | 158| 753 | 30553 | 136866 | 
| 113 | 25 src/_pytest/pytester.py | 1251 | 1267| 158 | 30711 | 136866 | 
| 114 | 25 src/_pytest/python.py | 1 | 77| 550 | 31261 | 136866 | 
| 115 | 26 src/_pytest/cacheprovider.py | 295 | 355| 541 | 31802 | 141245 | 
| 116 | 26 testing/python/integration.py | 291 | 328| 219 | 32021 | 141245 | 
| 117 | 27 src/_pytest/nodes.py | 529 | 560| 208 | 32229 | 145676 | 
| 118 | 27 testing/python/fixtures.py | 1065 | 2075| 6139 | 38368 | 145676 | 
| 119 | 27 src/_pytest/python.py | 656 | 674| 218 | 38586 | 145676 | 
| 120 | 27 testing/python/collect.py | 1031 | 1052| 189 | 38775 | 145676 | 
| 121 | 28 testing/example_scripts/issue88_initial_file_multinodes/conftest.py | 1 | 15| 0 | 38775 | 145738 | 
| 122 | 28 src/_pytest/hookspec.py | 377 | 400| 193 | 38968 | 145738 | 
| 123 | 28 src/_pytest/pytester.py | 1221 | 1249| 250 | 39218 | 145738 | 
| 124 | 28 testing/python/fixtures.py | 2997 | 3076| 516 | 39734 | 145738 | 
| 125 | 28 src/_pytest/pytester.py | 348 | 380| 218 | 39952 | 145738 | 
| 126 | 28 testing/python/metafunc.py | 1322 | 1354| 211 | 40163 | 145738 | 
| 127 | 29 src/_pytest/skipping.py | 241 | 255| 152 | 40315 | 148186 | 
| 128 | 29 src/_pytest/unittest.py | 298 | 319| 174 | 40489 | 148186 | 
| 129 | 29 testing/python/integration.py | 248 | 270| 183 | 40672 | 148186 | 
| 130 | 29 src/_pytest/runner.py | 153 | 176| 186 | 40858 | 148186 | 
| 131 | 29 src/_pytest/doctest.py | 703 | 725| 204 | 41062 | 148186 | 
| 132 | 29 testing/python/collect.py | 1122 | 1140| 154 | 41216 | 148186 | 
| 133 | 30 src/_pytest/setuponly.py | 59 | 95| 288 | 41504 | 148882 | 
| 134 | 30 testing/python/metafunc.py | 1398 | 1422| 214 | 41718 | 148882 | 
| 135 | 30 testing/python/fixtures.py | 87 | 1063| 6247 | 47965 | 148882 | 
| 136 | 30 src/_pytest/doctest.py | 222 | 239| 172 | 48137 | 148882 | 
| 137 | 30 src/_pytest/unittest.py | 147 | 169| 220 | 48357 | 148882 | 
| 138 | 30 testing/python/metafunc.py | 139 | 186| 471 | 48828 | 148882 | 
| 139 | 30 src/_pytest/runner.py | 196 | 222| 226 | 49054 | 148882 | 
| 140 | 31 src/_pytest/mark/__init__.py | 240 | 265| 177 | 49231 | 150907 | 
| 141 | 31 src/_pytest/hookspec.py | 244 | 260| 129 | 49360 | 150907 | 
| 142 | 31 src/_pytest/runner.py | 1 | 40| 243 | 49603 | 150907 | 
| 143 | 31 src/_pytest/unittest.py | 257 | 295| 398 | 50001 | 150907 | 
| 144 | 31 src/_pytest/python.py | 554 | 572| 184 | 50185 | 150907 | 
| 145 | 31 src/_pytest/skipping.py | 258 | 314| 523 | 50708 | 150907 | 
| 146 | 31 src/_pytest/pytester.py | 964 | 989| 243 | 50951 | 150907 | 
| 147 | 32 src/_pytest/terminal.py | 753 | 785| 329 | 51280 | 161908 | 
| 148 | 33 testing/example_scripts/collect/package_infinite_recursion/conftest.py | 1 | 3| 0 | 51280 | 161918 | 
| 149 | 33 src/_pytest/config/__init__.py | 751 | 816| 308 | 51588 | 161918 | 
| 150 | 34 doc/en/example/multipython.py | 25 | 46| 152 | 51740 | 162355 | 
| 151 | 34 src/_pytest/python.py | 397 | 410| 150 | 51890 | 162355 | 
| 152 | **34 src/_pytest/main.py** | 787 | 800| 130 | 52020 | 162355 | 
| 153 | 34 src/_pytest/reports.py | 388 | 418| 246 | 52266 | 162355 | 
| 154 | 35 testing/example_scripts/fixtures/fill_fixtures/test_conftest_funcargs_only_available_in_subdir/sub1/conftest.py | 1 | 8| 0 | 52266 | 162386 | 
| 155 | 35 testing/python/metafunc.py | 887 | 914| 222 | 52488 | 162386 | 
| 156 | 35 doc/en/example/assertion/failure_demo.py | 123 | 160| 143 | 52631 | 162386 | 
| 157 | 35 src/_pytest/terminal.py | 733 | 751| 149 | 52780 | 162386 | 
| 158 | 35 extra/get_issues.py | 55 | 86| 231 | 53011 | 162386 | 
| 159 | 35 src/_pytest/unittest.py | 204 | 255| 395 | 53406 | 162386 | 
| 160 | **35 src/_pytest/main.py** | 159 | 215| 400 | 53806 | 162386 | 
| 161 | 35 src/_pytest/python.py | 640 | 654| 142 | 53948 | 162386 | 
| 162 | **35 src/_pytest/main.py** | 439 | 503| 557 | 54505 | 162386 | 
| 163 | 35 src/_pytest/skipping.py | 45 | 81| 381 | 54886 | 162386 | 
| 164 | 35 testing/python/collect.py | 398 | 425| 193 | 55079 | 162386 | 
| 165 | 36 src/_pytest/capture.py | 150 | 171| 191 | 55270 | 169758 | 
| 166 | 37 testing/python/raises.py | 1 | 51| 332 | 55602 | 171962 | 
| 167 | 37 src/_pytest/python.py | 622 | 638| 141 | 55743 | 171962 | 
| 168 | 37 src/_pytest/unittest.py | 171 | 202| 266 | 56009 | 171962 | 
| 169 | 37 src/_pytest/doctest.py | 571 | 659| 800 | 56809 | 171962 | 
| 170 | 38 src/_pytest/junitxml.py | 212 | 234| 230 | 57039 | 177708 | 
| 171 | 38 testing/python/metafunc.py | 1380 | 1396| 110 | 57149 | 177708 | 
| 172 | 38 src/_pytest/cacheprovider.py | 181 | 226| 363 | 57512 | 177708 | 
| 173 | 38 doc/en/example/multipython.py | 1 | 22| 115 | 57627 | 177708 | 
| 174 | 38 testing/python/metafunc.py | 1080 | 1094| 118 | 57745 | 177708 | 
| 175 | 38 testing/python/raises.py | 129 | 156| 223 | 57968 | 177708 | 
| 176 | 38 testing/python/fixtures.py | 3078 | 4075| 6022 | 63990 | 177708 | 
| 177 | 38 testing/python/collect.py | 427 | 444| 123 | 64113 | 177708 | 
| 178 | 39 src/_pytest/assertion/__init__.py | 154 | 180| 256 | 64369 | 179142 | 
| 179 | 39 testing/python/collect.py | 477 | 496| 156 | 64525 | 179142 | 
| 180 | 39 testing/python/metafunc.py | 1011 | 1033| 176 | 64701 | 179142 | 
| 181 | 39 src/_pytest/cacheprovider.py | 229 | 247| 176 | 64877 | 179142 | 
| 182 | 39 doc/en/example/assertion/failure_demo.py | 1 | 39| 163 | 65040 | 179142 | 
| 183 | 39 src/_pytest/doctest.py | 555 | 568| 135 | 65175 | 179142 | 
| 184 | 39 src/_pytest/hookspec.py | 318 | 334| 129 | 65304 | 179142 | 
| 185 | 39 testing/python/metafunc.py | 1035 | 1061| 208 | 65512 | 179142 | 
| 186 | 39 src/_pytest/runner.py | 114 | 133| 221 | 65733 | 179142 | 
| 187 | 39 src/_pytest/pytester.py | 1582 | 1661| 779 | 66512 | 179142 | 
| 188 | **39 src/_pytest/main.py** | 521 | 531| 131 | 66643 | 179142 | 
| 189 | 39 src/_pytest/nodes.py | 480 | 526| 392 | 67035 | 179142 | 


### Hint

```


```

## Patch

```diff
diff --git a/src/_pytest/main.py b/src/_pytest/main.py
--- a/src/_pytest/main.py
+++ b/src/_pytest/main.py
@@ -765,12 +765,14 @@ def collect(self) -> Iterator[Union[nodes.Item, nodes.Collector]]:
                     self._notfound.append((report_arg, col))
                     continue
 
-                # If __init__.py was the only file requested, then the matched node will be
-                # the corresponding Package, and the first yielded item will be the __init__
-                # Module itself, so just use that. If this special case isn't taken, then all
-                # the files in the package will be yielded.
-                if argpath.basename == "__init__.py":
-                    assert isinstance(matching[0], nodes.Collector)
+                # If __init__.py was the only file requested, then the matched
+                # node will be the corresponding Package (by default), and the
+                # first yielded item will be the __init__ Module itself, so
+                # just use that. If this special case isn't taken, then all the
+                # files in the package will be yielded.
+                if argpath.basename == "__init__.py" and isinstance(
+                    matching[0], Package
+                ):
                     try:
                         yield next(iter(matching[0].collect()))
                     except StopIteration:

```

## Test Patch

```diff
diff --git a/testing/test_doctest.py b/testing/test_doctest.py
--- a/testing/test_doctest.py
+++ b/testing/test_doctest.py
@@ -68,9 +68,13 @@ def my_func():
             assert isinstance(items[0].parent, DoctestModule)
             assert items[0].parent is items[1].parent
 
-    def test_collect_module_two_doctest_no_modulelevel(self, pytester: Pytester):
+    @pytest.mark.parametrize("filename", ["__init__", "whatever"])
+    def test_collect_module_two_doctest_no_modulelevel(
+        self, pytester: Pytester, filename: str,
+    ) -> None:
         path = pytester.makepyfile(
-            whatever="""
+            **{
+                filename: """
             '# Empty'
             def my_func():
                 ">>> magic = 42 "
@@ -84,7 +88,8 @@ def another():
                 # This is another function
                 >>> import os # this one does have a doctest
                 '''
-        """
+            """,
+            },
         )
         for p in (path, pytester.path):
             items, reprec = pytester.inline_genitems(p, "--doctest-modules")

```


## Code snippets

### 1 - src/_pytest/doctest.py:

Start line: 526, End line: 552

```python
class DoctestModule(pytest.Module):
    def collect(self) -> Iterable[DoctestItem]:
        # ... other code

        if self.fspath.basename == "conftest.py":
            module = self.config.pluginmanager._importconftest(
                self.fspath, self.config.getoption("importmode")
            )
        else:
            try:
                module = import_path(self.fspath)
            except ImportError:
                if self.config.getvalue("doctest_ignore_import_errors"):
                    pytest.skip("unable to import module %r" % self.fspath)
                else:
                    raise
        # Uses internal doctest module parsing mechanism.
        finder = MockAwareDocTestFinder()
        optionflags = get_optionflags(self)
        runner = _get_runner(
            verbose=False,
            optionflags=optionflags,
            checker=_get_checker(),
            continue_on_failure=_get_continue_on_failure(self.config),
        )

        for test in finder.find(module, module.__name__):
            if test.examples:  # skip empty doctests
                yield DoctestItem.from_parent(
                    self, name=test.name, runner=runner, dtest=test
                )
```
### 2 - src/_pytest/doctest.py:

Start line: 410, End line: 438

```python
class DoctestTextfile(pytest.Module):
    obj = None

    def collect(self) -> Iterable[DoctestItem]:
        import doctest

        # Inspired by doctest.testfile; ideally we would use it directly,
        # but it doesn't support passing a custom checker.
        encoding = self.config.getini("doctest_encoding")
        text = self.fspath.read_text(encoding)
        filename = str(self.fspath)
        name = self.fspath.basename
        globs = {"__name__": "__main__"}

        optionflags = get_optionflags(self)

        runner = _get_runner(
            verbose=False,
            optionflags=optionflags,
            checker=_get_checker(),
            continue_on_failure=_get_continue_on_failure(self.config),
        )

        parser = doctest.DocTestParser()
        test = parser.get_doctest(text, globs, name, filename, 0)
        if test.examples:
            yield DoctestItem.from_parent(
                self, name=test.name, runner=runner, dtest=test
            )
```
### 3 - testing/python/collect.py:

Start line: 1, End line: 35

```python
import sys
import textwrap
from typing import Any
from typing import Dict

import _pytest._code
import pytest
from _pytest.config import ExitCode
from _pytest.nodes import Collector
from _pytest.pytester import Testdir


class TestModule:
    def test_failing_import(self, testdir):
        modcol = testdir.getmodulecol("import alksdjalskdjalkjals")
        pytest.raises(Collector.CollectError, modcol.collect)

    def test_import_duplicate(self, testdir):
        a = testdir.mkdir("a")
        b = testdir.mkdir("b")
        p = a.ensure("test_whatever.py")
        p.pyimport()
        del sys.modules["test_whatever"]
        b.ensure("test_whatever.py")
        result = testdir.runpytest()
        result.stdout.fnmatch_lines(
            [
                "*import*mismatch*",
                "*imported*test_whatever*",
                "*%s*" % a.join("test_whatever.py"),
                "*not the same*",
                "*%s*" % b.join("test_whatever.py"),
                "*HINT*",
            ]
        )
```
### 4 - testing/python/collect.py:

Start line: 1351, End line: 1388

```python
def test_package_with_modules(testdir):
    """
    .
    └── root
        ├── __init__.py
        ├── sub1
        │   ├── __init__.py
        │   └── sub1_1
        │       ├── __init__.py
        │       └── test_in_sub1.py
        └── sub2
            └── test
                └── test_in_sub2.py

    """
    root = testdir.mkpydir("root")
    sub1 = root.mkdir("sub1")
    sub1.ensure("__init__.py")
    sub1_test = sub1.mkdir("sub1_1")
    sub1_test.ensure("__init__.py")
    sub2 = root.mkdir("sub2")
    sub2_test = sub2.mkdir("sub2")

    sub1_test.join("test_in_sub1.py").write("def test_1(): pass")
    sub2_test.join("test_in_sub2.py").write("def test_2(): pass")

    # Execute from .
    result = testdir.runpytest("-v", "-s")
    result.assert_outcomes(passed=2)

    # Execute from . with one argument "root"
    result = testdir.runpytest("-v", "-s", "root")
    result.assert_outcomes(passed=2)

    # Chdir into package's root and execute with no args
    root.chdir()
    result = testdir.runpytest("-v", "-s")
    result.assert_outcomes(passed=2)
```
### 5 - src/_pytest/main.py:

Start line: 652, End line: 785

```python
@final
class Session(nodes.FSCollector):

    def collect(self) -> Iterator[Union[nodes.Item, nodes.Collector]]:
        # ... other code

        for argpath, names in self._initial_parts:
            self.trace("processing argument", (argpath, names))
            self.trace.root.indent += 1

            # Start with a Session root, and delve to argpath item (dir or file)
            # and stack all Packages found on the way.
            # No point in finding packages when collecting doctests.
            if not self.config.getoption("doctestmodules", False):
                pm = self.config.pluginmanager
                for parent in reversed(argpath.parts()):
                    if pm._confcutdir and pm._confcutdir.relto(parent):
                        break

                    if parent.isdir():
                        pkginit = parent.join("__init__.py")
                        if pkginit.isfile() and pkginit not in node_cache1:
                            col = self._collectfile(pkginit, handle_dupes=False)
                            if col:
                                if isinstance(col[0], Package):
                                    pkg_roots[str(parent)] = col[0]
                                node_cache1[col[0].fspath] = [col[0]]

            # If it's a directory argument, recurse and look for any Subpackages.
            # Let the Package collector deal with subnodes, don't collect here.
            if argpath.check(dir=1):
                assert not names, "invalid arg {!r}".format((argpath, names))

                seen_dirs: Set[py.path.local] = set()
                for direntry in visit(str(argpath), self._recurse):
                    if not direntry.is_file():
                        continue

                    path = py.path.local(direntry.path)
                    dirpath = path.dirpath()

                    if dirpath not in seen_dirs:
                        # Collect packages first.
                        seen_dirs.add(dirpath)
                        pkginit = dirpath.join("__init__.py")
                        if pkginit.exists():
                            for x in self._collectfile(pkginit):
                                yield x
                                if isinstance(x, Package):
                                    pkg_roots[str(dirpath)] = x
                    if str(dirpath) in pkg_roots:
                        # Do not collect packages here.
                        continue

                    for x in self._collectfile(path):
                        key = (type(x), x.fspath)
                        if key in node_cache2:
                            yield node_cache2[key]
                        else:
                            node_cache2[key] = x
                            yield x
            else:
                assert argpath.check(file=1)

                if argpath in node_cache1:
                    col = node_cache1[argpath]
                else:
                    collect_root = pkg_roots.get(argpath.dirname, self)
                    col = collect_root._collectfile(argpath, handle_dupes=False)
                    if col:
                        node_cache1[argpath] = col

                matching = []
                work: List[
                    Tuple[Sequence[Union[nodes.Item, nodes.Collector]], Sequence[str]]
                ] = [(col, names)]
                while work:
                    self.trace("matchnodes", col, names)
                    self.trace.root.indent += 1

                    matchnodes, matchnames = work.pop()
                    for node in matchnodes:
                        if not matchnames:
                            matching.append(node)
                            continue
                        if not isinstance(node, nodes.Collector):
                            continue
                        key = (type(node), node.nodeid)
                        if key in matchnodes_cache:
                            rep = matchnodes_cache[key]
                        else:
                            rep = collect_one_node(node)
                            matchnodes_cache[key] = rep
                        if rep.passed:
                            submatchnodes = []
                            for r in rep.result:
                                # TODO: Remove parametrized workaround once collection structure contains
                                # parametrization.
                                if (
                                    r.name == matchnames[0]
                                    or r.name.split("[")[0] == matchnames[0]
                                ):
                                    submatchnodes.append(r)
                            if submatchnodes:
                                work.append((submatchnodes, matchnames[1:]))
                            # XXX Accept IDs that don't have "()" for class instances.
                            elif len(rep.result) == 1 and rep.result[0].name == "()":
                                work.append((rep.result, matchnames))
                        else:
                            # Report collection failures here to avoid failing to run some test
                            # specified in the command line because the module could not be
                            # imported (#134).
                            node.ihook.pytest_collectreport(report=rep)

                    self.trace("matchnodes finished -> ", len(matching), "nodes")
                    self.trace.root.indent -= 1

                if not matching:
                    report_arg = "::".join((str(argpath), *names))
                    self._notfound.append((report_arg, col))
                    continue

                # If __init__.py was the only file requested, then the matched node will be
                # the corresponding Package, and the first yielded item will be the __init__
                # Module itself, so just use that. If this special case isn't taken, then all
                # the files in the package will be yielded.
                if argpath.basename == "__init__.py":
                    assert isinstance(matching[0], nodes.Collector)
                    try:
                        yield next(iter(matching[0].collect()))
                    except StopIteration:
                        # The package collects nothing with only an __init__.py
                        # file in it, which gets ignored by the default
                        # "python_files" option.
                        pass
                    continue

                yield from matching

            self.trace.root.indent -= 1
```
### 6 - src/_pytest/doctest.py:

Start line: 1, End line: 62

```python
"""Discover and run doctests in modules and test files."""
import bdb
import inspect
import platform
import sys
import traceback
import types
import warnings
from contextlib import contextmanager
from typing import Any
from typing import Callable
from typing import Dict
from typing import Generator
from typing import Iterable
from typing import List
from typing import Optional
from typing import Pattern
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import Union

import py.path

import pytest
from _pytest import outcomes
from _pytest._code.code import ExceptionInfo
from _pytest._code.code import ReprFileLocation
from _pytest._code.code import TerminalRepr
from _pytest._io import TerminalWriter
from _pytest.compat import safe_getattr
from _pytest.config import Config
from _pytest.config.argparsing import Parser
from _pytest.fixtures import FixtureRequest
from _pytest.nodes import Collector
from _pytest.outcomes import OutcomeException
from _pytest.pathlib import import_path
from _pytest.python_api import approx
from _pytest.warning_types import PytestWarning

if TYPE_CHECKING:
    import doctest

DOCTEST_REPORT_CHOICE_NONE = "none"
DOCTEST_REPORT_CHOICE_CDIFF = "cdiff"
DOCTEST_REPORT_CHOICE_NDIFF = "ndiff"
DOCTEST_REPORT_CHOICE_UDIFF = "udiff"
DOCTEST_REPORT_CHOICE_ONLY_FIRST_FAILURE = "only_first_failure"

DOCTEST_REPORT_CHOICES = (
    DOCTEST_REPORT_CHOICE_NONE,
    DOCTEST_REPORT_CHOICE_CDIFF,
    DOCTEST_REPORT_CHOICE_NDIFF,
    DOCTEST_REPORT_CHOICE_UDIFF,
    DOCTEST_REPORT_CHOICE_ONLY_FIRST_FAILURE,
)

# Lazy definition of runner class
RUNNER_CLASS = None
# Lazy definition of output checker class
CHECKER_CLASS: Optional[Type["doctest.OutputChecker"]] = None
```
### 7 - testing/python/collect.py:

Start line: 253, End line: 278

```python
class TestFunction:
    def test_getmodulecollector(self, testdir):
        item = testdir.getitem("def test_func(): pass")
        modcol = item.getparent(pytest.Module)
        assert isinstance(modcol, pytest.Module)
        assert hasattr(modcol.obj, "test_func")

    @pytest.mark.filterwarnings("default")
    def test_function_as_object_instance_ignored(self, testdir):
        testdir.makepyfile(
            """
            class A(object):
                def __call__(self, tmpdir):
                    0/0

            test_a = A()
        """
        )
        result = testdir.runpytest()
        result.stdout.fnmatch_lines(
            [
                "collected 0 items",
                "*test_function_as_object_instance_ignored.py:2: "
                "*cannot collect 'test_a' because it is not a function.",
            ]
        )
```
### 8 - testing/python/collect.py:

Start line: 1213, End line: 1239

```python
@pytest.mark.filterwarnings("default")
def test_dont_collect_non_function_callable(testdir):
    """Test for issue https://github.com/pytest-dev/pytest/issues/331

    In this case an INTERNALERROR occurred trying to report the failure of
    a test like this one because pytest failed to get the source lines.
    """
    testdir.makepyfile(
        """
        class Oh(object):
            def __call__(self):
                pass

        test_a = Oh()

        def test_real():
            pass
    """
    )
    result = testdir.runpytest()
    result.stdout.fnmatch_lines(
        [
            "*collected 1 item*",
            "*test_dont_collect_non_function_callable.py:2: *cannot collect 'test_a' because it is not a function*",
            "*1 passed, 1 warning in *",
        ]
    )
```
### 9 - testing/python/collect.py:

Start line: 556, End line: 573

```python
class TestFunction:

    def test_issue751_multiple_parametrize_with_ids(self, testdir):
        modcol = testdir.getmodulecol(
            """
            import pytest
            @pytest.mark.parametrize('x', [0], ids=['c'])
            @pytest.mark.parametrize('y', [0, 1], ids=['a', 'b'])
            class Test(object):
                def test1(self, x, y):
                    pass
                def test2(self, x, y):
                    pass
        """
        )
        colitems = modcol.collect()[0].collect()[0].collect()
        assert colitems[0].name == "test1[a-c]"
        assert colitems[1].name == "test1[b-c]"
        assert colitems[2].name == "test2[a-c]"
        assert colitems[3].name == "test2[b-c]"
```
### 10 - testing/python/collect.py:

Start line: 197, End line: 250

```python
class TestClass:

    def test_setup_teardown_class_as_classmethod(self, testdir):
        testdir.makepyfile(
            test_mod1="""
            class TestClassMethod(object):
                @classmethod
                def setup_class(cls):
                    pass
                def test_1(self):
                    pass
                @classmethod
                def teardown_class(cls):
                    pass
        """
        )
        result = testdir.runpytest()
        result.stdout.fnmatch_lines(["*1 passed*"])

    def test_issue1035_obj_has_getattr(self, testdir):
        modcol = testdir.getmodulecol(
            """
            class Chameleon(object):
                def __getattr__(self, name):
                    return True
            chameleon = Chameleon()
        """
        )
        colitems = modcol.collect()
        assert len(colitems) == 0

    def test_issue1579_namedtuple(self, testdir):
        testdir.makepyfile(
            """
            import collections

            TestCase = collections.namedtuple('TestCase', ['a'])
        """
        )
        result = testdir.runpytest()
        result.stdout.fnmatch_lines(
            "*cannot collect test class 'TestCase' "
            "because it has a __new__ constructor*"
        )

    def test_issue2234_property(self, testdir):
        testdir.makepyfile(
            """
            class TestCase(object):
                @property
                def prop(self):
                    raise NotImplementedError()
        """
        )
        result = testdir.runpytest()
        assert result.ret == ExitCode.NO_TESTS_COLLECTED
```
### 36 - src/_pytest/main.py:

Start line: 326, End line: 343

```python
def pytest_runtestloop(session: "Session") -> bool:
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
### 53 - src/_pytest/main.py:

Start line: 363, End line: 387

```python
def pytest_ignore_collect(path: py.path.local, config: Config) -> Optional[bool]:
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
    return None
```
### 55 - src/_pytest/main.py:

Start line: 390, End line: 405

```python
def pytest_collection_modifyitems(items: List[nodes.Item], config: Config) -> None:
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
```
### 64 - src/_pytest/main.py:

Start line: 533, End line: 567

```python
@final
class Session(nodes.FSCollector):

    def _collectfile(
        self, path: py.path.local, handle_dupes: bool = True
    ) -> Sequence[nodes.Collector]:
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

        return ihook.pytest_collect_file(path=path, parent=self)  # type: ignore[no-any-return]

    @overload
    def perform_collect(
        self, args: Optional[Sequence[str]] = ..., genitems: "Literal[True]" = ...
    ) -> Sequence[nodes.Item]:
        ...

    @overload
    def perform_collect(
        self, args: Optional[Sequence[str]] = ..., genitems: bool = ...
    ) -> Sequence[Union[nodes.Item, nodes.Collector]]:
        ...
```
### 70 - src/_pytest/main.py:

Start line: 569, End line: 634

```python
@final
class Session(nodes.FSCollector):

    def perform_collect(
        self, args: Optional[Sequence[str]] = None, genitems: bool = True
    ) -> Sequence[Union[nodes.Item, nodes.Collector]]:
        """Perform the collection phase for this session.

        This is called by the default
        :func:`pytest_collection <_pytest.hookspec.pytest_collection>` hook
        implementation; see the documentation of this hook for more details.
        For testing purposes, it may also be called directly on a fresh
        ``Session``.

        This function normally recursively expands any collectors collected
        from the session to their items, and only items are returned. For
        testing purposes, this may be suppressed by passing ``genitems=False``,
        in which case the return value contains these collectors unexpanded,
        and ``session.items`` is empty.
        """
        if args is None:
            args = self.config.args

        self.trace("perform_collect", self, args)
        self.trace.root.indent += 1

        self._notfound: List[Tuple[str, Sequence[nodes.Collector]]] = []
        self._initial_parts: List[Tuple[py.path.local, List[str]]] = []
        self.items: List[nodes.Item] = []

        hook = self.config.hook

        items: Sequence[Union[nodes.Item, nodes.Collector]] = self.items
        try:
            initialpaths: List[py.path.local] = []
            for arg in args:
                fspath, parts = resolve_collection_argument(
                    self.config.invocation_params.dir,
                    arg,
                    as_pypath=self.config.option.pyargs,
                )
                self._initial_parts.append((fspath, parts))
                initialpaths.append(fspath)
            self._initialpaths = frozenset(initialpaths)
            rep = collect_one_node(self)
            self.ihook.pytest_collectreport(report=rep)
            self.trace.root.indent -= 1
            if self._notfound:
                errors = []
                for arg, cols in self._notfound:
                    line = f"(no name {arg!r} in any of {cols!r})"
                    errors.append(f"not found: {arg}\n{line}")
                raise UsageError(*errors)
            if not genitems:
                items = rep.result
            else:
                if rep.passed:
                    for node in rep.result:
                        self.items.extend(self.genitems(node))

            self.config.pluginmanager.check_pending()
            hook.pytest_collection_modifyitems(
                session=self, config=self.config, items=items
            )
        finally:
            hook.pytest_collection_finish(session=self)

        self.testscollected = len(items)
        return items
```
### 80 - src/_pytest/main.py:

Start line: 305, End line: 323

```python
def pytest_cmdline_main(config: Config) -> Union[int, ExitCode]:
    return wrap_session(config, _main)


def _main(config: Config, session: "Session") -> Optional[Union[int, ExitCode]]:
    """Default command line protocol for initialization, session,
    running tests and reporting."""
    config.hook.pytest_collection(session=session)
    config.hook.pytest_runtestloop(session=session)

    if session.testsfailed:
        return ExitCode.TESTS_FAILED
    elif session.testscollected == 0:
        return ExitCode.NO_TESTS_COLLECTED
    return None


def pytest_collection(session: "Session") -> None:
    session.perform_collect()
```
### 86 - src/_pytest/main.py:

Start line: 636, End line: 650

```python
@final
class Session(nodes.FSCollector):

    def collect(self) -> Iterator[Union[nodes.Item, nodes.Collector]]:
        from _pytest.python import Package

        # Keep track of any collected nodes in here, so we don't duplicate fixtures.
        node_cache1: Dict[py.path.local, Sequence[nodes.Collector]] = {}
        node_cache2: Dict[
            Tuple[Type[nodes.Collector], py.path.local], nodes.Collector
        ] = ({})

        # Keep track of any collected collectors in matchnodes paths, so they
        # are not collected more than once.
        matchnodes_cache: Dict[Tuple[Type[nodes.Collector], str], CollectReport] = ({})

        # Dirnames of pkgs with dunder-init files.
        pkg_roots: Dict[str, Package] = {}
        # ... other code
```
### 89 - src/_pytest/main.py:

Start line: 1, End line: 48

```python
"""Core implementation of the testing process: init, session, runtest loop."""
import argparse
import fnmatch
import functools
import importlib
import os
import sys
from pathlib import Path
from typing import Callable
from typing import Dict
from typing import FrozenSet
from typing import Iterator
from typing import List
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import Union

import attr
import py

import _pytest._code
from _pytest import nodes
from _pytest.compat import final
from _pytest.config import Config
from _pytest.config import directory_arg
from _pytest.config import ExitCode
from _pytest.config import hookimpl
from _pytest.config import PytestPluginManager
from _pytest.config import UsageError
from _pytest.config.argparsing import Parser
from _pytest.fixtures import FixtureManager
from _pytest.outcomes import exit
from _pytest.pathlib import absolutepath
from _pytest.pathlib import bestrelpath
from _pytest.pathlib import visit
from _pytest.reports import CollectReport
from _pytest.reports import TestReport
from _pytest.runner import collect_one_node
from _pytest.runner import SetupState


if TYPE_CHECKING:
    from typing_extensions import Literal
```
### 112 - src/_pytest/main.py:

Start line: 51, End line: 158

```python
def pytest_addoption(parser: Parser) -> None:
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
    )
    group = parser.getgroup("pytest-warnings")
    group.addoption(
        "-W",
        "--pythonwarnings",
        action="append",
        help="set which warnings to report, see -W option of python itself.",
    )
    parser.addini(
        "filterwarnings",
        type="linelist",
        help="Each line specifies a pattern for "
        "warnings.filterwarnings. "
        "Processed after -W/--pythonwarnings.",
    )
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
        "--strict-config",
        action="store_true",
        help="any warnings encountered while parsing the `pytest` section of the configuration file raise errors.",
    )
    group._addoption(
        "--strict-markers",
        action="store_true",
        help="markers not registered in the `markers` section of the configuration file raise errors.",
    )
    group._addoption(
        "--strict", action="store_true", help="(deprecated) alias to --strict-markers.",
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
        "--co",
        action="store_true",
        help="only collect tests, don't execute them.",
    )
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
    # ... other code
```
### 152 - src/_pytest/main.py:

Start line: 787, End line: 800

```python
@final
class Session(nodes.FSCollector):

    def genitems(
        self, node: Union[nodes.Item, nodes.Collector]
    ) -> Iterator[nodes.Item]:
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
### 160 - src/_pytest/main.py:

Start line: 159, End line: 215

```python
def pytest_addoption(parser: Parser) -> None:
    # ... other code
    group.addoption(
        "--deselect",
        action="append",
        metavar="nodeid_prefix",
        help="deselect item (via node id prefix) during collection (multi-allowed).",
    )
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
    group.addoption(
        "--import-mode",
        default="prepend",
        choices=["prepend", "append", "importlib"],
        dest="importmode",
        help="prepend/append to sys.path when importing test modules and conftest files, "
        "default is to prepend.",
    )

    group = parser.getgroup("debugconfig", "test session debugging and configuration")
    group.addoption(
        "--basetemp",
        dest="basetemp",
        default=None,
        type=validate_basetemp,
        metavar="dir",
        help=(
            "base temporary directory for this test run."
            "(warning: this directory is removed if it exists)"
        ),
    )
```
### 162 - src/_pytest/main.py:

Start line: 439, End line: 503

```python
@final
class Session(nodes.FSCollector):
    Interrupted = Interrupted
    Failed = Failed
    # Set on the session by runner.pytest_sessionstart.
    _setupstate: SetupState
    # Set on the session by fixtures.pytest_sessionstart.
    _fixturemanager: FixtureManager
    exitstatus: Union[int, ExitCode]

    def __init__(self, config: Config) -> None:
        super().__init__(
            config.rootdir, parent=None, config=config, session=self, nodeid=""
        )
        self.testsfailed = 0
        self.testscollected = 0
        self.shouldstop: Union[bool, str] = False
        self.shouldfail: Union[bool, str] = False
        self.trace = config.trace.root.get("collection")
        self.startdir = config.invocation_dir
        self._initialpaths: FrozenSet[py.path.local] = frozenset()

        self._bestrelpathcache: Dict[Path, str] = _bestrelpath_cache(config.rootpath)

        self.config.pluginmanager.register(self, name="session")

    @classmethod
    def from_config(cls, config: Config) -> "Session":
        session: Session = cls._create(config)
        return session

    def __repr__(self) -> str:
        return "<%s %s exitstatus=%r testsfailed=%d testscollected=%d>" % (
            self.__class__.__name__,
            self.name,
            getattr(self, "exitstatus", "<UNSET>"),
            self.testsfailed,
            self.testscollected,
        )

    def _node_location_to_relpath(self, node_path: Path) -> str:
        # bestrelpath is a quite slow function.
        return self._bestrelpathcache[node_path]

    @hookimpl(tryfirst=True)
    def pytest_collectstart(self) -> None:
        if self.shouldfail:
            raise self.Failed(self.shouldfail)
        if self.shouldstop:
            raise self.Interrupted(self.shouldstop)

    @hookimpl(tryfirst=True)
    def pytest_runtest_logreport(
        self, report: Union[TestReport, CollectReport]
    ) -> None:
        if report.failed and not hasattr(report, "wasxfail"):
            self.testsfailed += 1
            maxfail = self.config.getvalue("maxfail")
            if maxfail and self.testsfailed >= maxfail:
                self.shouldfail = "stopping after %d failures" % (self.testsfailed)

    pytest_collectreport = pytest_runtest_logreport

    def isinitpath(self, path: py.path.local) -> bool:
        return path in self._initialpaths
```
### 188 - src/_pytest/main.py:

Start line: 521, End line: 531

```python
@final
class Session(nodes.FSCollector):

    def _recurse(self, direntry: "os.DirEntry[str]") -> bool:
        if direntry.name == "__pycache__":
            return False
        path = py.path.local(direntry.path)
        ihook = self.gethookproxy(path.dirpath())
        if ihook.pytest_ignore_collect(path=path, config=self.config):
            return False
        norecursepatterns = self.config.getini("norecursedirs")
        if any(path.check(fnmatch=pat) for pat in norecursepatterns):
            return False
        return True
```
