# pytest-dev__pytest-6680

| **pytest-dev/pytest** | `194b52145b98fda8ad1c62ebacf96b9e2916309c` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 460 |
| **Any found context length** | 460 |
| **Avg pos** | 1.0 |
| **Min pos** | 1 |
| **Max pos** | 1 |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/src/_pytest/deprecated.py b/src/_pytest/deprecated.py
--- a/src/_pytest/deprecated.py
+++ b/src/_pytest/deprecated.py
@@ -36,7 +36,10 @@
 
 NODE_USE_FROM_PARENT = UnformattedWarning(
     PytestDeprecationWarning,
-    "direct construction of {name} has been deprecated, please use {name}.from_parent",
+    "Direct construction of {name} has been deprecated, please use {name}.from_parent.\n"
+    "See "
+    "https://docs.pytest.org/en/latest/deprecations.html#node-construction-changed-to-node-from-parent"
+    " for more details.",
 )
 
 JUNIT_XML_DEFAULT_FAMILY = PytestDeprecationWarning(

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| src/_pytest/deprecated.py | 39 | 39 | 1 | 1 | 460


## Problem Statement

```
Improve deprecation docs for Node.from_parent
In the "Node Construction changed to Node.from_parent" section in the deprecation docs, we definitely need to add:

* [x] An example of the warning that users will see (so they can find the session on google).
* [x] The warning `NODE_USE_FROM_PARENT` should point to the deprecation docs.
* [x] Show a "before -> after" example.
* [x] ensure from_parent will not support config/session

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 src/_pytest/deprecated.py** | 1 | 52| 460 | 460 | 460 | 
| 2 | 2 src/_pytest/nodes.py | 180 | 210| 203 | 663 | 4949 | 
| 3 | 2 src/_pytest/nodes.py | 1 | 40| 280 | 943 | 4949 | 
| 4 | 3 src/_pytest/python.py | 162 | 171| 118 | 1061 | 17304 | 
| 5 | 3 src/_pytest/nodes.py | 320 | 367| 407 | 1468 | 17304 | 
| 6 | 4 src/_pytest/fixtures.py | 1 | 111| 725 | 2193 | 29223 | 
| 7 | 4 src/_pytest/nodes.py | 154 | 178| 206 | 2399 | 29223 | 
| 8 | 5 src/_pytest/main.py | 368 | 460| 802 | 3201 | 34512 | 
| 9 | 6 src/_pytest/junitxml.py | 263 | 275| 133 | 3334 | 39624 | 
| 10 | 7 src/_pytest/warning_types.py | 1 | 70| 328 | 3662 | 40353 | 
| 11 | 7 src/_pytest/warning_types.py | 73 | 89| 127 | 3789 | 40353 | 
| 12 | 7 src/_pytest/main.py | 637 | 686| 431 | 4220 | 40353 | 
| 13 | 7 src/_pytest/python.py | 1525 | 1564| 365 | 4585 | 40353 | 
| 14 | 7 src/_pytest/nodes.py | 430 | 447| 148 | 4733 | 40353 | 
| 15 | 8 src/_pytest/warnings.py | 148 | 170| 252 | 4985 | 41595 | 
| 16 | 8 src/_pytest/python.py | 1353 | 1412| 434 | 5419 | 41595 | 
| 17 | 8 src/_pytest/nodes.py | 262 | 318| 451 | 5870 | 41595 | 
| 18 | 9 src/_pytest/assertion/rewrite.py | 438 | 492| 427 | 6297 | 50697 | 
| 19 | 9 src/_pytest/main.py | 590 | 608| 151 | 6448 | 50697 | 
| 20 | 9 src/_pytest/fixtures.py | 1402 | 1456| 434 | 6882 | 50697 | 
| 21 | 9 src/_pytest/junitxml.py | 190 | 214| 223 | 7105 | 50697 | 
| 22 | 9 src/_pytest/main.py | 493 | 509| 163 | 7268 | 50697 | 
| 23 | 9 src/_pytest/junitxml.py | 232 | 260| 217 | 7485 | 50697 | 
| 24 | 10 src/_pytest/mark/structures.py | 357 | 395| 215 | 7700 | 53606 | 
| 25 | 11 src/_pytest/recwarn.py | 36 | 55| 201 | 7901 | 55612 | 
| 26 | 11 src/_pytest/main.py | 462 | 491| 291 | 8192 | 55612 | 
| 27 | 11 src/_pytest/junitxml.py | 216 | 230| 132 | 8324 | 55612 | 
| 28 | 12 src/_pytest/terminal.py | 409 | 420| 116 | 8440 | 65257 | 
| 29 | 12 src/_pytest/terminal.py | 241 | 257| 142 | 8582 | 65257 | 
| 30 | 12 src/_pytest/warnings.py | 1 | 31| 224 | 8806 | 65257 | 
| 31 | 13 doc/en/conf.py | 347 | 370| 215 | 9021 | 68142 | 
| 32 | 13 src/_pytest/junitxml.py | 124 | 153| 251 | 9272 | 68142 | 
| 33 | 14 testing/python/fixtures.py | 1067 | 2059| 6151 | 15423 | 95066 | 
| 34 | 14 src/_pytest/main.py | 511 | 588| 736 | 16159 | 95066 | 
| 35 | 14 src/_pytest/nodes.py | 574 | 587| 135 | 16294 | 95066 | 
| 36 | 15 src/_pytest/config/findpaths.py | 89 | 111| 157 | 16451 | 96292 | 
| 37 | 15 src/_pytest/terminal.py | 209 | 239| 241 | 16692 | 96292 | 
| 38 | 15 src/_pytest/recwarn.py | 58 | 75| 135 | 16827 | 96292 | 
| 39 | 15 src/_pytest/nodes.py | 450 | 481| 248 | 17075 | 96292 | 
| 40 | 15 src/_pytest/warnings.py | 100 | 145| 286 | 17361 | 96292 | 
| 41 | 16 src/_pytest/config/__init__.py | 772 | 836| 540 | 17901 | 105804 | 
| 42 | 16 src/_pytest/nodes.py | 212 | 236| 147 | 18048 | 105804 | 
| 43 | 16 src/_pytest/nodes.py | 90 | 152| 474 | 18522 | 105804 | 
| 44 | 16 src/_pytest/fixtures.py | 1274 | 1286| 149 | 18671 | 105804 | 
| 45 | 16 testing/python/fixtures.py | 85 | 1065| 6306 | 24977 | 105804 | 
| 46 | 17 src/_pytest/hookspec.py | 336 | 441| 739 | 25716 | 110276 | 
| 47 | 18 src/pytest/__init__.py | 1 | 100| 661 | 26377 | 110938 | 
| 48 | 18 doc/en/conf.py | 114 | 210| 774 | 27151 | 110938 | 
| 49 | 18 src/_pytest/terminal.py | 770 | 790| 191 | 27342 | 110938 | 
| 50 | 18 src/_pytest/python.py | 1289 | 1321| 239 | 27581 | 110938 | 
| 51 | 18 src/_pytest/recwarn.py | 203 | 227| 186 | 27767 | 110938 | 
| 52 | 18 testing/python/fixtures.py | 3537 | 4294| 4978 | 32745 | 110938 | 
| 53 | 18 src/_pytest/warning_types.py | 118 | 134| 134 | 32879 | 110938 | 
| 54 | 18 src/_pytest/fixtures.py | 1255 | 1272| 156 | 33035 | 110938 | 
| 55 | 18 src/_pytest/warning_types.py | 92 | 115| 140 | 33175 | 110938 | 
| 56 | 18 src/_pytest/fixtures.py | 1212 | 1253| 348 | 33523 | 110938 | 
| 57 | 18 src/_pytest/python.py | 558 | 575| 136 | 33659 | 110938 | 
| 58 | 18 src/_pytest/fixtures.py | 114 | 169| 607 | 34266 | 110938 | 
| 59 | 18 src/_pytest/fixtures.py | 1458 | 1487| 222 | 34488 | 110938 | 
| 60 | 19 scripts/release.py | 68 | 85| 168 | 34656 | 111862 | 
| 61 | 19 scripts/release.py | 88 | 103| 124 | 34780 | 111862 | 
| 62 | 19 testing/python/fixtures.py | 2061 | 2505| 2812 | 37592 | 111862 | 
| 63 | 19 src/_pytest/warnings.py | 34 | 56| 156 | 37748 | 111862 | 
| 64 | 19 src/_pytest/nodes.py | 43 | 66| 224 | 37972 | 111862 | 
| 65 | 19 src/_pytest/fixtures.py | 1320 | 1366| 401 | 38373 | 111862 | 
| 66 | 20 src/_pytest/stepwise.py | 26 | 71| 298 | 38671 | 112576 | 
| 67 | 20 src/_pytest/config/__init__.py | 876 | 903| 255 | 38926 | 112576 | 
| 68 | 20 src/_pytest/recwarn.py | 78 | 130| 474 | 39400 | 112576 | 
| 69 | 21 testing/python/collect.py | 35 | 57| 187 | 39587 | 122115 | 
| 70 | 21 src/_pytest/junitxml.py | 155 | 188| 315 | 39902 | 122115 | 
| 71 | 21 testing/python/fixtures.py | 2507 | 3535| 6208 | 46110 | 122115 | 
| 72 | 21 src/_pytest/terminal.py | 817 | 873| 454 | 46564 | 122115 | 
| 73 | 22 src/_pytest/_code/source.py | 348 | 369| 219 | 46783 | 125299 | 
| 74 | 22 src/_pytest/assertion/rewrite.py | 619 | 682| 472 | 47255 | 125299 | 
| 75 | 22 src/_pytest/fixtures.py | 587 | 605| 178 | 47433 | 125299 | 
| 76 | 23 doc/en/_themes/flask_theme_support.py | 1 | 88| 1273 | 48706 | 126572 | 
| 77 | 23 src/_pytest/nodes.py | 238 | 260| 195 | 48901 | 126572 | 
| 78 | 23 src/_pytest/fixtures.py | 607 | 634| 269 | 49170 | 126572 | 
| 79 | 24 src/_pytest/cacheprovider.py | 471 | 509| 340 | 49510 | 130545 | 
| 80 | 24 src/_pytest/python.py | 1323 | 1350| 217 | 49727 | 130545 | 
| 81 | 24 src/_pytest/terminal.py | 334 | 345| 165 | 49892 | 130545 | 
| 82 | 24 src/_pytest/assertion/rewrite.py | 971 | 985| 181 | 50073 | 130545 | 
| 83 | 24 src/_pytest/nodes.py | 69 | 87| 186 | 50259 | 130545 | 
| 84 | 24 src/_pytest/cacheprovider.py | 272 | 326| 494 | 50753 | 130545 | 
| 85 | 25 doc/en/example/xfail_demo.py | 1 | 39| 143 | 50896 | 130689 | 
| 86 | 26 src/_pytest/helpconfig.py | 141 | 208| 550 | 51446 | 132376 | 
| 87 | 27 scripts/publish-gh-release-notes.py | 41 | 66| 227 | 51673 | 133126 | 
| 88 | 27 src/_pytest/assertion/rewrite.py | 807 | 883| 677 | 52350 | 133126 | 
| 89 | 28 src/_pytest/debugging.py | 294 | 320| 219 | 52569 | 135567 | 
| 90 | 28 doc/en/conf.py | 373 | 391| 128 | 52697 | 135567 | 
| 91 | 29 src/_pytest/runner.py | 75 | 107| 270 | 52967 | 138546 | 
| 92 | 29 src/_pytest/assertion/rewrite.py | 240 | 255| 139 | 53106 | 138546 | 
| 93 | 29 src/_pytest/junitxml.py | 90 | 122| 207 | 53313 | 138546 | 
| 94 | 30 src/_pytest/setuponly.py | 46 | 79| 242 | 53555 | 139083 | 
| 95 | 30 src/_pytest/fixtures.py | 864 | 886| 180 | 53735 | 139083 | 
| 96 | 30 src/_pytest/junitxml.py | 1 | 87| 660 | 54395 | 139083 | 
| 97 | 30 src/_pytest/recwarn.py | 229 | 265| 299 | 54694 | 139083 | 
| 98 | 31 src/_pytest/_code/code.py | 651 | 695| 361 | 55055 | 148456 | 
| 99 | 32 src/_pytest/skipping.py | 34 | 71| 363 | 55418 | 150002 | 
| 100 | 32 src/_pytest/config/__init__.py | 265 | 275| 154 | 55572 | 150002 | 
| 101 | 33 src/_pytest/assertion/__init__.py | 1 | 38| 226 | 55798 | 151302 | 
| 102 | 33 src/_pytest/fixtures.py | 303 | 339| 375 | 56173 | 151302 | 
| 103 | 34 src/_pytest/doctest.py | 214 | 261| 406 | 56579 | 156388 | 
| 104 | 34 src/_pytest/python.py | 511 | 555| 449 | 57028 | 156388 | 
| 105 | 34 src/_pytest/fixtures.py | 888 | 921| 316 | 57344 | 156388 | 
| 106 | 34 testing/python/collect.py | 131 | 172| 247 | 57591 | 156388 | 
| 107 | 35 src/_pytest/reports.py | 1 | 38| 287 | 57878 | 159995 | 
| 108 | 35 src/_pytest/cacheprovider.py | 49 | 87| 288 | 58166 | 159995 | 
| 109 | 36 src/_pytest/mark/legacy.py | 1 | 25| 124 | 58290 | 160744 | 
| 110 | 36 src/_pytest/doctest.py | 105 | 153| 347 | 58637 | 160744 | 
| 111 | 36 src/_pytest/assertion/rewrite.py | 885 | 893| 136 | 58773 | 160744 | 
| 112 | 36 src/_pytest/warnings.py | 59 | 97| 323 | 59096 | 160744 | 
| 113 | 36 src/_pytest/junitxml.py | 520 | 606| 642 | 59738 | 160744 | 
| 114 | 36 src/_pytest/fixtures.py | 380 | 394| 193 | 59931 | 160744 | 
| 115 | 37 testing/python/integration.py | 325 | 362| 224 | 60155 | 163673 | 
| 116 | 37 src/_pytest/hookspec.py | 444 | 473| 215 | 60370 | 163673 | 
| 117 | 37 src/_pytest/runner.py | 365 | 393| 238 | 60608 | 163673 | 
| 118 | 38 testing/example_scripts/issue88_initial_file_multinodes/conftest.py | 1 | 15| 0 | 60608 | 163726 | 
| 119 | 38 src/_pytest/config/__init__.py | 976 | 1022| 427 | 61035 | 163726 | 
| 120 | 38 src/_pytest/main.py | 610 | 635| 262 | 61297 | 163726 | 
| 121 | 38 src/_pytest/fixtures.py | 672 | 692| 132 | 61429 | 163726 | 
| 122 | 38 src/_pytest/hookspec.py | 163 | 249| 531 | 61960 | 163726 | 
| 123 | 38 src/_pytest/assertion/rewrite.py | 944 | 969| 249 | 62209 | 163726 | 
| 124 | 39 doc/en/example/nonpython/conftest.py | 1 | 16| 115 | 62324 | 164049 | 
| 125 | 39 src/_pytest/cacheprovider.py | 329 | 366| 320 | 62644 | 164049 | 
| 126 | 39 src/_pytest/assertion/rewrite.py | 770 | 805| 305 | 62949 | 164049 | 
| 127 | 39 src/_pytest/assertion/rewrite.py | 729 | 741| 127 | 63076 | 164049 | 
| 128 | 39 src/_pytest/junitxml.py | 336 | 363| 248 | 63324 | 164049 | 
| 129 | 40 src/_pytest/config/argparsing.py | 472 | 511| 425 | 63749 | 168450 | 
| 130 | 40 src/_pytest/doctest.py | 345 | 361| 133 | 63882 | 168450 | 
| 131 | 41 src/_pytest/logging.py | 264 | 289| 165 | 64047 | 174294 | 
| 132 | 41 src/_pytest/assertion/rewrite.py | 39 | 63| 264 | 64311 | 174294 | 
| 133 | 41 src/_pytest/mark/structures.py | 29 | 52| 202 | 64513 | 174294 | 
| 134 | 41 src/_pytest/doctest.py | 445 | 477| 269 | 64782 | 174294 | 
| 135 | 41 src/_pytest/skipping.py | 1 | 31| 201 | 64983 | 174294 | 
| 136 | 41 src/_pytest/runner.py | 148 | 167| 171 | 65154 | 174294 | 
| 137 | 41 src/_pytest/fixtures.py | 277 | 300| 191 | 65345 | 174294 | 
| 138 | 41 src/_pytest/fixtures.py | 488 | 515| 203 | 65548 | 174294 | 
| 139 | 41 src/_pytest/junitxml.py | 453 | 491| 310 | 65858 | 174294 | 
| 140 | 41 src/_pytest/main.py | 41 | 148| 741 | 66599 | 174294 | 
| 141 | 41 src/_pytest/recwarn.py | 186 | 200| 132 | 66731 | 174294 | 
| 142 | 42 src/_pytest/pastebin.py | 25 | 43| 183 | 66914 | 175145 | 
| 143 | 42 src/_pytest/hookspec.py | 568 | 593| 249 | 67163 | 175145 | 
| 144 | 42 src/_pytest/assertion/rewrite.py | 271 | 322| 429 | 67592 | 175145 | 
| 145 | 42 src/_pytest/nodes.py | 555 | 572| 167 | 67759 | 175145 | 
| 146 | 42 src/_pytest/fixtures.py | 1288 | 1318| 256 | 68015 | 175145 | 
| 147 | 42 src/_pytest/pastebin.py | 46 | 60| 157 | 68172 | 175145 | 
| 148 | 42 src/_pytest/nodes.py | 483 | 495| 155 | 68327 | 175145 | 
| 149 | 43 src/_pytest/compat.py | 1 | 80| 405 | 68732 | 178075 | 
| 150 | 43 src/_pytest/hookspec.py | 122 | 133| 112 | 68844 | 178075 | 
| 151 | 43 src/_pytest/debugging.py | 323 | 340| 125 | 68969 | 178075 | 
| 152 | 44 testing/python/metafunc.py | 110 | 132| 182 | 69151 | 193006 | 
| 153 | 44 src/_pytest/config/argparsing.py | 371 | 401| 259 | 69410 | 193006 | 
| 154 | 44 src/_pytest/terminal.py | 875 | 911| 310 | 69720 | 193006 | 
| 155 | 44 src/_pytest/assertion/rewrite.py | 1 | 36| 241 | 69961 | 193006 | 
| 156 | 44 src/_pytest/junitxml.py | 493 | 518| 177 | 70138 | 193006 | 
| 157 | 44 src/_pytest/assertion/rewrite.py | 743 | 768| 260 | 70398 | 193006 | 
| 158 | 44 src/_pytest/nodes.py | 497 | 508| 130 | 70528 | 193006 | 
| 159 | 44 src/_pytest/helpconfig.py | 39 | 84| 301 | 70829 | 193006 | 
| 160 | 44 doc/en/conf.py | 211 | 346| 790 | 71619 | 193006 | 
| 161 | 44 src/_pytest/assertion/rewrite.py | 362 | 378| 155 | 71774 | 193006 | 
| 162 | 44 src/_pytest/config/argparsing.py | 199 | 272| 641 | 72415 | 193006 | 
| 163 | 44 scripts/release.py | 106 | 125| 126 | 72541 | 193006 | 
| 164 | 44 src/_pytest/terminal.py | 1155 | 1174| 171 | 72712 | 193006 | 
| 165 | 45 src/_pytest/monkeypatch.py | 251 | 285| 310 | 73022 | 195436 | 
| 166 | 45 src/_pytest/recwarn.py | 1 | 33| 179 | 73201 | 195436 | 
| 167 | 45 src/_pytest/cacheprovider.py | 157 | 167| 129 | 73330 | 195436 | 
| 168 | 45 src/_pytest/setuponly.py | 1 | 43| 293 | 73623 | 195436 | 
| 169 | 45 src/_pytest/cacheprovider.py | 226 | 270| 436 | 74059 | 195436 | 
| 170 | 45 src/_pytest/assertion/rewrite.py | 684 | 714| 262 | 74321 | 195436 | 
| 171 | 45 src/_pytest/assertion/rewrite.py | 895 | 928| 389 | 74710 | 195436 | 
| 172 | 46 testing/example_scripts/issue_519.py | 34 | 52| 115 | 74825 | 195902 | 
| 173 | 47 doc/en/example/assertion/failure_demo.py | 191 | 203| 114 | 74939 | 197561 | 
| 174 | 47 src/_pytest/python.py | 399 | 440| 364 | 75303 | 197561 | 
| 175 | 47 src/_pytest/terminal.py | 494 | 517| 227 | 75530 | 197561 | 
| 176 | 47 testing/python/collect.py | 1220 | 1247| 219 | 75749 | 197561 | 
| 177 | 47 testing/python/collect.py | 572 | 671| 662 | 76411 | 197561 | 
| 178 | 47 src/_pytest/monkeypatch.py | 1 | 37| 250 | 76661 | 197561 | 
| 179 | 48 src/_pytest/faulthandler.py | 46 | 114| 550 | 77211 | 198402 | 
| 180 | 48 src/_pytest/junitxml.py | 411 | 428| 169 | 77380 | 198402 | 
| 181 | 48 src/_pytest/assertion/rewrite.py | 987 | 1023| 427 | 77807 | 198402 | 
| 182 | 48 src/_pytest/nodes.py | 510 | 530| 182 | 77989 | 198402 | 
| 183 | 48 src/_pytest/doctest.py | 263 | 326| 564 | 78553 | 198402 | 
| 184 | 48 src/_pytest/assertion/rewrite.py | 207 | 238| 271 | 78824 | 198402 | 
| 185 | 48 src/_pytest/terminal.py | 77 | 169| 663 | 79487 | 198402 | 
| 186 | 48 src/_pytest/python.py | 72 | 124| 354 | 79841 | 198402 | 
| 187 | 48 src/_pytest/recwarn.py | 133 | 184| 413 | 80254 | 198402 | 
| 188 | 48 src/_pytest/cacheprovider.py | 369 | 424| 410 | 80664 | 198402 | 
| 189 | 48 src/_pytest/assertion/rewrite.py | 65 | 103| 312 | 80976 | 198402 | 
| 190 | 48 src/_pytest/config/__init__.py | 942 | 974| 242 | 81218 | 198402 | 
| 191 | 48 testing/python/collect.py | 1070 | 1088| 154 | 81372 | 198402 | 
| 192 | 48 src/_pytest/stepwise.py | 73 | 109| 290 | 81662 | 198402 | 
| 193 | 48 src/_pytest/monkeypatch.py | 40 | 66| 180 | 81842 | 198402 | 
| 194 | 48 src/_pytest/nodes.py | 391 | 427| 290 | 82132 | 198402 | 
| 195 | 48 src/_pytest/_code/code.py | 1006 | 1020| 130 | 82262 | 198402 | 
| 196 | 48 src/_pytest/terminal.py | 792 | 815| 142 | 82404 | 198402 | 
| 197 | 48 src/_pytest/junitxml.py | 431 | 450| 147 | 82551 | 198402 | 
| 198 | 48 src/_pytest/terminal.py | 260 | 293| 333 | 82884 | 198402 | 
| 199 | 48 src/_pytest/assertion/rewrite.py | 258 | 268| 160 | 83044 | 198402 | 
| 200 | 48 src/_pytest/python.py | 482 | 509| 251 | 83295 | 198402 | 


## Patch

```diff
diff --git a/src/_pytest/deprecated.py b/src/_pytest/deprecated.py
--- a/src/_pytest/deprecated.py
+++ b/src/_pytest/deprecated.py
@@ -36,7 +36,10 @@
 
 NODE_USE_FROM_PARENT = UnformattedWarning(
     PytestDeprecationWarning,
-    "direct construction of {name} has been deprecated, please use {name}.from_parent",
+    "Direct construction of {name} has been deprecated, please use {name}.from_parent.\n"
+    "See "
+    "https://docs.pytest.org/en/latest/deprecations.html#node-construction-changed-to-node-from-parent"
+    " for more details.",
 )
 
 JUNIT_XML_DEFAULT_FAMILY = PytestDeprecationWarning(

```

## Test Patch

```diff
diff --git a/testing/deprecated_test.py b/testing/deprecated_test.py
--- a/testing/deprecated_test.py
+++ b/testing/deprecated_test.py
@@ -86,7 +86,7 @@ class MockConfig:
     ms = MockConfig()
     with pytest.warns(
         DeprecationWarning,
-        match="direct construction of .* has been deprecated, please use .*.from_parent",
+        match="Direct construction of .* has been deprecated, please use .*.from_parent.*",
     ) as w:
         nodes.Node(name="test", config=ms, session=ms, nodeid="None")
     assert w[0].lineno == inspect.currentframe().f_lineno - 1

```


## Code snippets

### 1 - src/_pytest/deprecated.py:

Start line: 1, End line: 52

```python
"""
This module contains deprecation messages and bits of code used elsewhere in the codebase
that is planned to be removed in the next pytest release.

Keeping it in a central location makes it easy to track what is deprecated and should
be removed when the time comes.

All constants defined in this module should be either PytestWarning instances or UnformattedWarning
in case of warnings which need to format their messages.
"""
from _pytest.warning_types import PytestDeprecationWarning
from _pytest.warning_types import UnformattedWarning

# set of plugins which have been integrated into the core; we use this list to ignore
# them during registration to avoid conflicts
DEPRECATED_EXTERNAL_PLUGINS = {
    "pytest_catchlog",
    "pytest_capturelog",
    "pytest_faulthandler",
}

FUNCARGNAMES = PytestDeprecationWarning(
    "The `funcargnames` attribute was an alias for `fixturenames`, "
    "since pytest 2.3 - use the newer attribute instead."
)

RESULT_LOG = PytestDeprecationWarning(
    "--result-log is deprecated, please try the new pytest-reportlog plugin.\n"
    "See https://docs.pytest.org/en/latest/deprecations.html#result-log-result-log for more information."
)

FIXTURE_POSITIONAL_ARGUMENTS = PytestDeprecationWarning(
    "Passing arguments to pytest.fixture() as positional arguments is deprecated - pass them "
    "as a keyword argument instead."
)

NODE_USE_FROM_PARENT = UnformattedWarning(
    PytestDeprecationWarning,
    "direct construction of {name} has been deprecated, please use {name}.from_parent",
)

JUNIT_XML_DEFAULT_FAMILY = PytestDeprecationWarning(
    "The 'junit_family' default value will change to 'xunit2' in pytest 6.0.\n"
    "Add 'junit_family=xunit1' to your pytest.ini file to keep the current format "
    "in future versions of pytest and silence this warning."
)

NO_PRINT_LOGS = PytestDeprecationWarning(
    "--no-print-logs is deprecated and scheduled for removal in pytest 6.0.\n"
    "Please use --show-capture instead."
)
```
### 2 - src/_pytest/nodes.py:

Start line: 180, End line: 210

```python
class Node(metaclass=NodeMeta):

    def warn(self, warning):
        """Issue a warning for this item.

        Warnings will be displayed after the test session, unless explicitly suppressed

        :param Warning warning: the warning instance to issue. Must be a subclass of PytestWarning.

        :raise ValueError: if ``warning`` instance is not a subclass of PytestWarning.

        Example usage:

        .. code-block:: python

            node.warn(PytestWarning("some message"))

        """
        from _pytest.warning_types import PytestWarning

        if not isinstance(warning, PytestWarning):
            raise ValueError(
                "warning must be an instance of PytestWarning or subclass, got {!r}".format(
                    warning
                )
            )
        path, lineno = get_fslocation_from_item(self)
        warnings.warn_explicit(
            warning,
            category=None,
            filename=str(path),
            lineno=lineno + 1 if lineno is not None else None,
        )
```
### 3 - src/_pytest/nodes.py:

Start line: 1, End line: 40

```python
import os
import warnings
from functools import lru_cache
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Union

import py

import _pytest._code
from _pytest._code.code import ExceptionChainRepr
from _pytest._code.code import ExceptionInfo
from _pytest._code.code import ReprExceptionInfo
from _pytest._code.source import getfslineno
from _pytest.compat import cached_property
from _pytest.compat import TYPE_CHECKING
from _pytest.config import Config
from _pytest.config import PytestPluginManager
from _pytest.deprecated import NODE_USE_FROM_PARENT
from _pytest.fixtures import FixtureDef
from _pytest.fixtures import FixtureLookupError
from _pytest.fixtures import FixtureLookupErrorRepr
from _pytest.mark.structures import Mark
from _pytest.mark.structures import MarkDecorator
from _pytest.mark.structures import NodeKeywords
from _pytest.outcomes import fail
from _pytest.outcomes import Failed
from _pytest.store import Store

if TYPE_CHECKING:
    # Imported here due to circular import.
    from _pytest.main import Session  # noqa: F401

SEP = "/"

tracebackcutdir = py.path.local(_pytest.__file__).dirpath()
```
### 4 - src/_pytest/python.py:

Start line: 162, End line: 171

```python
def async_warn(nodeid: str) -> None:
    msg = "async def functions are not natively supported and have been skipped.\n"
    msg += (
        "You need to install a suitable plugin for your async framework, for example:\n"
    )
    msg += "  - pytest-asyncio\n"
    msg += "  - pytest-trio\n"
    msg += "  - pytest-tornasync"
    warnings.warn(PytestUnhandledCoroutineWarning(msg.format(nodeid)))
    skip(msg="async def function and no async plugin installed (see warnings)")
```
### 5 - src/_pytest/nodes.py:

Start line: 320, End line: 367

```python
class Node(metaclass=NodeMeta):

    def _repr_failure_py(
        self, excinfo: ExceptionInfo[Union[Failed, FixtureLookupError]], style=None
    ) -> Union[str, ReprExceptionInfo, ExceptionChainRepr, FixtureLookupErrorRepr]:
        if isinstance(excinfo.value, fail.Exception):
            if not excinfo.value.pytrace:
                return str(excinfo.value)
        if isinstance(excinfo.value, FixtureLookupError):
            return excinfo.value.formatrepr()
        if self.config.getoption("fulltrace", False):
            style = "long"
        else:
            tb = _pytest._code.Traceback([excinfo.traceback[-1]])
            self._prunetraceback(excinfo)
            if len(excinfo.traceback) == 0:
                excinfo.traceback = tb
            if style == "auto":
                style = "long"
        # XXX should excinfo.getrepr record all data and toterminal() process it?
        if style is None:
            if self.config.getoption("tbstyle", "auto") == "short":
                style = "short"
            else:
                style = "long"

        if self.config.getoption("verbose", 0) > 1:
            truncate_locals = False
        else:
            truncate_locals = True

        try:
            os.getcwd()
            abspath = False
        except OSError:
            abspath = True

        return excinfo.getrepr(
            funcargs=True,
            abspath=abspath,
            showlocals=self.config.getoption("showlocals", False),
            style=style,
            tbfilter=False,  # pruned already, or in --fulltrace mode.
            truncate_locals=truncate_locals,
        )

    def repr_failure(
        self, excinfo, style=None
    ) -> Union[str, ReprExceptionInfo, ExceptionChainRepr, FixtureLookupErrorRepr]:
        return self._repr_failure_py(excinfo, style)
```
### 6 - src/_pytest/fixtures.py:

Start line: 1, End line: 111

```python
import functools
import inspect
import sys
import warnings
from collections import defaultdict
from collections import deque
from collections import OrderedDict
from typing import Dict
from typing import List
from typing import Tuple

import attr
import py

import _pytest
from _pytest._code.code import FormattedExcinfo
from _pytest._code.code import TerminalRepr
from _pytest._code.source import getfslineno
from _pytest._io import TerminalWriter
from _pytest.compat import _format_args
from _pytest.compat import _PytestWrapper
from _pytest.compat import get_real_func
from _pytest.compat import get_real_method
from _pytest.compat import getfuncargnames
from _pytest.compat import getimfunc
from _pytest.compat import getlocation
from _pytest.compat import is_generator
from _pytest.compat import NOTSET
from _pytest.compat import safe_getattr
from _pytest.compat import TYPE_CHECKING
from _pytest.deprecated import FIXTURE_POSITIONAL_ARGUMENTS
from _pytest.deprecated import FUNCARGNAMES
from _pytest.mark import ParameterSet
from _pytest.outcomes import fail
from _pytest.outcomes import TEST_OUTCOME

if TYPE_CHECKING:
    from typing import Type

    from _pytest import nodes
    from _pytest.main import Session


@attr.s(frozen=True)
class PseudoFixtureDef:
    cached_result = attr.ib()
    scope = attr.ib()


def pytest_sessionstart(session: "Session"):
    import _pytest.python
    import _pytest.nodes

    scopename2class.update(
        {
            "package": _pytest.python.Package,
            "class": _pytest.python.Class,
            "module": _pytest.python.Module,
            "function": _pytest.nodes.Item,
            "session": _pytest.main.Session,
        }
    )
    session._fixturemanager = FixtureManager(session)


scopename2class = {}  # type: Dict[str, Type[nodes.Node]]

scope2props = dict(session=())  # type: Dict[str, Tuple[str, ...]]
scope2props["package"] = ("fspath",)
scope2props["module"] = ("fspath", "module")
scope2props["class"] = scope2props["module"] + ("cls",)
scope2props["instance"] = scope2props["class"] + ("instance",)
scope2props["function"] = scope2props["instance"] + ("function", "keywords")


def scopeproperty(name=None, doc=None):
    def decoratescope(func):
        scopename = name or func.__name__

        def provide(self):
            if func.__name__ in scope2props[self.scope]:
                return func(self)
            raise AttributeError(
                "{} not available in {}-scoped context".format(scopename, self.scope)
            )

        return property(provide, None, None, func.__doc__)

    return decoratescope


def get_scope_package(node, fixturedef):
    import pytest

    cls = pytest.Package
    current = node
    fixture_package_name = "{}/{}".format(fixturedef.baseid, "__init__.py")
    while current and (
        type(current) is not cls or fixture_package_name != current.nodeid
    ):
        current = current.parent
    if current is None:
        return node.session
    return current


def get_scope_node(node, scope):
    cls = scopename2class.get(scope)
    if cls is None:
        raise ValueError("unknown scope")
    return node.getparent(cls)
```
### 7 - src/_pytest/nodes.py:

Start line: 154, End line: 178

```python
class Node(metaclass=NodeMeta):

    @classmethod
    def from_parent(cls, parent: "Node", **kw):
        """
        Public Constructor for Nodes

        This indirection got introduced in order to enable removing
        the fragile logic from the node constructors.

        Subclasses can use ``super().from_parent(...)`` when overriding the construction

        :param parent: the parent node of this test Node
        """
        if "config" in kw:
            raise TypeError("config is not a valid argument for from_parent")
        if "session" in kw:
            raise TypeError("session is not a valid argument for from_parent")
        return cls._create(parent=parent, **kw)

    @property
    def ihook(self):
        """ fspath sensitive hook proxy used to call pytest hooks"""
        return self.session.gethookproxy(self.fspath)

    def __repr__(self):
        return "<{} {}>".format(self.__class__.__name__, getattr(self, "name", None))
```
### 8 - src/_pytest/main.py:

Start line: 368, End line: 460

```python
class Session(nodes.FSCollector):
    Interrupted = Interrupted
    Failed = Failed
    # Set on the session by runner.pytest_sessionstart.
    _setupstate = None  # type: SetupState
    # Set on the session by fixtures.pytest_sessionstart.
    _fixturemanager = None  # type: FixtureManager
    exitstatus = None  # type: Union[int, ExitCode]

    def __init__(self, config: Config) -> None:
        nodes.FSCollector.__init__(
            self, config.rootdir, parent=None, config=config, session=self, nodeid=""
        )
        self.testsfailed = 0
        self.testscollected = 0
        self.shouldstop = False
        self.shouldfail = False
        self.trace = config.trace.root.get("collection")
        self.startdir = config.invocation_dir
        self._initialpaths = frozenset()  # type: FrozenSet[py.path.local]

        # Keep track of any collected nodes in here, so we don't duplicate fixtures
        self._collection_node_cache1 = (
            {}
        )  # type: Dict[py.path.local, Sequence[nodes.Collector]]
        self._collection_node_cache2 = (
            {}
        )  # type: Dict[Tuple[Type[nodes.Collector], py.path.local], nodes.Collector]
        self._collection_node_cache3 = (
            {}
        )  # type: Dict[Tuple[Type[nodes.Collector], str], CollectReport]

        # Dirnames of pkgs with dunder-init files.
        self._collection_pkg_roots = {}  # type: Dict[py.path.local, Package]

        self._bestrelpathcache = _bestrelpath_cache(
            config.rootdir
        )  # type: Dict[py.path.local, str]

        self.config.pluginmanager.register(self, name="session")

    @classmethod
    def from_config(cls, config):
        return cls._create(config)

    def __repr__(self):
        return "<%s %s exitstatus=%r testsfailed=%d testscollected=%d>" % (
            self.__class__.__name__,
            self.name,
            getattr(self, "exitstatus", "<UNSET>"),
            self.testsfailed,
            self.testscollected,
        )

    def _node_location_to_relpath(self, node_path: py.path.local) -> str:
        # bestrelpath is a quite slow function
        return self._bestrelpathcache[node_path]

    @hookimpl(tryfirst=True)
    def pytest_collectstart(self):
        if self.shouldfail:
            raise self.Failed(self.shouldfail)
        if self.shouldstop:
            raise self.Interrupted(self.shouldstop)

    @hookimpl(tryfirst=True)
    def pytest_runtest_logreport(self, report):
        if report.failed and not hasattr(report, "wasxfail"):
            self.testsfailed += 1
            maxfail = self.config.getvalue("maxfail")
            if maxfail and self.testsfailed >= maxfail:
                self.shouldfail = "stopping after %d failures" % (self.testsfailed)

    pytest_collectreport = pytest_runtest_logreport

    def isinitpath(self, path):
        return path in self._initialpaths

    def gethookproxy(self, fspath: py.path.local):
        return super()._gethookproxy(fspath)

    def perform_collect(self, args=None, genitems=True):
        hook = self.config.hook
        try:
            items = self._perform_collect(args, genitems)
            self.config.pluginmanager.check_pending()
            hook.pytest_collection_modifyitems(
                session=self, config=self.config, items=items
            )
        finally:
            hook.pytest_collection_finish(session=self)
        self.testscollected = len(items)
        return items
```
### 9 - src/_pytest/junitxml.py:

Start line: 263, End line: 275

```python
def _warn_incompatibility_with_xunit2(request, fixture_name):
    """Emits a PytestWarning about the given fixture being incompatible with newer xunit revisions"""
    from _pytest.warning_types import PytestWarning

    xml = request.config._store.get(xml_key, None)
    if xml is not None and xml.family not in ("xunit1", "legacy"):
        request.node.warn(
            PytestWarning(
                "{fixture_name} is incompatible with junit_family '{family}' (use 'legacy' or 'xunit1')".format(
                    fixture_name=fixture_name, family=xml.family
                )
            )
        )
```
### 10 - src/_pytest/warning_types.py:

Start line: 1, End line: 70

```python
from typing import Any
from typing import Generic
from typing import TypeVar

import attr

from _pytest.compat import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Type  # noqa: F401 (used in type string)


class PytestWarning(UserWarning):
    """
    Bases: :class:`UserWarning`.

    Base class for all warnings emitted by pytest.
    """

    __module__ = "pytest"


class PytestAssertRewriteWarning(PytestWarning):
    """
    Bases: :class:`PytestWarning`.

    Warning emitted by the pytest assert rewrite module.
    """

    __module__ = "pytest"


class PytestCacheWarning(PytestWarning):
    """
    Bases: :class:`PytestWarning`.

    Warning emitted by the cache plugin in various situations.
    """

    __module__ = "pytest"


class PytestConfigWarning(PytestWarning):
    """
    Bases: :class:`PytestWarning`.

    Warning emitted for configuration issues.
    """

    __module__ = "pytest"


class PytestCollectionWarning(PytestWarning):
    """
    Bases: :class:`PytestWarning`.

    Warning emitted when pytest is not able to collect a file or symbol in a module.
    """

    __module__ = "pytest"


class PytestDeprecationWarning(PytestWarning, DeprecationWarning):
    """
    Bases: :class:`pytest.PytestWarning`, :class:`DeprecationWarning`.

    Warning class for features that will be removed in a future version.
    """

    __module__ = "pytest"
```
