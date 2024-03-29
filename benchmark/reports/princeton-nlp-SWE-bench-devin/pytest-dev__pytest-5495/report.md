# pytest-dev__pytest-5495

| **pytest-dev/pytest** | `1aefb24b37c30fba8fd79a744829ca16e252f340` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 1592 |
| **Any found context length** | 1592 |
| **Avg pos** | 6.0 |
| **Min pos** | 6 |
| **Max pos** | 6 |
| **Top file pos** | 2 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/src/_pytest/assertion/util.py b/src/_pytest/assertion/util.py
--- a/src/_pytest/assertion/util.py
+++ b/src/_pytest/assertion/util.py
@@ -254,17 +254,38 @@ def _compare_eq_iterable(left, right, verbose=0):
 
 
 def _compare_eq_sequence(left, right, verbose=0):
+    comparing_bytes = isinstance(left, bytes) and isinstance(right, bytes)
     explanation = []
     len_left = len(left)
     len_right = len(right)
     for i in range(min(len_left, len_right)):
         if left[i] != right[i]:
+            if comparing_bytes:
+                # when comparing bytes, we want to see their ascii representation
+                # instead of their numeric values (#5260)
+                # using a slice gives us the ascii representation:
+                # >>> s = b'foo'
+                # >>> s[0]
+                # 102
+                # >>> s[0:1]
+                # b'f'
+                left_value = left[i : i + 1]
+                right_value = right[i : i + 1]
+            else:
+                left_value = left[i]
+                right_value = right[i]
+
             explanation += [
-                "At index {} diff: {!r} != {!r}".format(i, left[i], right[i])
+                "At index {} diff: {!r} != {!r}".format(i, left_value, right_value)
             ]
             break
-    len_diff = len_left - len_right
 
+    if comparing_bytes:
+        # when comparing bytes, it doesn't help to show the "sides contain one or more items"
+        # longer explanation, so skip it
+        return explanation
+
+    len_diff = len_left - len_right
     if len_diff:
         if len_diff > 0:
             dir_with_more = "Left"

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| src/_pytest/assertion/util.py | 257 | 260 | 6 | 2 | 1592


## Problem Statement

```
Confusing assertion rewriting message with byte strings
The comparison with assertion rewriting for byte strings is confusing: 
\`\`\`
    def test_b():
>       assert b"" == b"42"
E       AssertionError: assert b'' == b'42'
E         Right contains more items, first extra item: 52
E         Full diff:
E         - b''
E         + b'42'
E         ?   ++
\`\`\`

52 is the ASCII ordinal of "4" here.

It became clear to me when using another example:

\`\`\`
    def test_b():
>       assert b"" == b"1"
E       AssertionError: assert b'' == b'1'
E         Right contains more items, first extra item: 49
E         Full diff:
E         - b''
E         + b'1'
E         ?   +
\`\`\`

Not sure what should/could be done here.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 src/_pytest/assertion/rewrite.py | 857 | 894| 417 | 417 | 7629 | 
| 2 | 1 src/_pytest/assertion/rewrite.py | 765 | 798| 353 | 770 | 7629 | 
| 3 | 1 src/_pytest/assertion/rewrite.py | 800 | 812| 171 | 941 | 7629 | 
| 4 | 1 src/_pytest/assertion/rewrite.py | 755 | 763| 136 | 1077 | 7629 | 
| 5 | 1 src/_pytest/assertion/rewrite.py | 1 | 27| 189 | 1266 | 7629 | 
| **-> 6 <-** | **2 src/_pytest/assertion/util.py** | 256 | 301| 326 | 1592 | 10593 | 
| 7 | 2 src/_pytest/assertion/rewrite.py | 663 | 726| 547 | 2139 | 10593 | 
| 8 | **2 src/_pytest/assertion/util.py** | 304 | 340| 358 | 2497 | 10593 | 
| 9 | 3 doc/en/example/assertion/failure_demo.py | 43 | 121| 680 | 3177 | 12233 | 
| 10 | 3 src/_pytest/assertion/rewrite.py | 841 | 855| 181 | 3358 | 12233 | 
| 11 | 3 src/_pytest/assertion/rewrite.py | 579 | 609| 262 | 3620 | 12233 | 
| 12 | 3 src/_pytest/assertion/rewrite.py | 346 | 366| 191 | 3811 | 12233 | 
| 13 | **3 src/_pytest/assertion/util.py** | 166 | 226| 492 | 4303 | 12233 | 
| 14 | **3 src/_pytest/assertion/util.py** | 116 | 163| 438 | 4741 | 12233 | 
| 15 | 3 src/_pytest/assertion/rewrite.py | 514 | 577| 450 | 5191 | 12233 | 
| 16 | 3 src/_pytest/assertion/rewrite.py | 458 | 512| 526 | 5717 | 12233 | 
| 17 | 3 doc/en/example/assertion/failure_demo.py | 255 | 282| 161 | 5878 | 12233 | 
| 18 | 3 src/_pytest/assertion/rewrite.py | 327 | 343| 146 | 6024 | 12233 | 
| 19 | 3 src/_pytest/assertion/rewrite.py | 638 | 661| 241 | 6265 | 12233 | 
| 20 | **3 src/_pytest/assertion/util.py** | 229 | 253| 184 | 6449 | 12233 | 
| 21 | 4 testing/python/metafunc.py | 206 | 226| 279 | 6728 | 25348 | 
| 22 | 4 src/_pytest/assertion/rewrite.py | 183 | 204| 190 | 6918 | 25348 | 
| 23 | 5 src/_pytest/doctest.py | 473 | 509| 343 | 7261 | 29359 | 
| 24 | 5 src/_pytest/assertion/rewrite.py | 93 | 144| 561 | 7822 | 29359 | 
| 25 | **5 src/_pytest/assertion/util.py** | 343 | 372| 252 | 8074 | 29359 | 
| 26 | 5 doc/en/example/assertion/failure_demo.py | 124 | 161| 143 | 8217 | 29359 | 
| 27 | 5 src/_pytest/assertion/rewrite.py | 257 | 287| 340 | 8557 | 29359 | 
| 28 | 6 src/_pytest/compat.py | 170 | 204| 287 | 8844 | 31645 | 
| 29 | 7 src/_pytest/assertion/__init__.py | 1 | 25| 141 | 8985 | 32663 | 
| 30 | 7 testing/python/metafunc.py | 258 | 274| 193 | 9178 | 32663 | 
| 31 | 8 testing/python/approx.py | 427 | 443| 137 | 9315 | 38464 | 
| 32 | 8 src/_pytest/assertion/rewrite.py | 56 | 91| 271 | 9586 | 38464 | 
| 33 | 8 src/_pytest/assertion/rewrite.py | 206 | 237| 251 | 9837 | 38464 | 
| 34 | 8 doc/en/example/assertion/failure_demo.py | 205 | 252| 228 | 10065 | 38464 | 
| 35 | 9 src/_pytest/assertion/truncate.py | 1 | 34| 209 | 10274 | 39186 | 
| 36 | 9 src/_pytest/assertion/rewrite.py | 814 | 839| 249 | 10523 | 39186 | 
| 37 | 10 src/_pytest/hookspec.py | 471 | 485| 122 | 10645 | 43590 | 
| 38 | **10 src/_pytest/assertion/util.py** | 1 | 29| 227 | 10872 | 43590 | 
| 39 | 10 src/_pytest/assertion/rewrite.py | 30 | 54| 228 | 11100 | 43590 | 
| 40 | 11 testing/python/collect.py | 1251 | 1276| 186 | 11286 | 53199 | 
| 41 | 11 testing/python/metafunc.py | 292 | 313| 229 | 11515 | 53199 | 
| 42 | 11 src/_pytest/assertion/rewrite.py | 624 | 636| 117 | 11632 | 53199 | 
| 43 | 11 src/_pytest/assertion/rewrite.py | 728 | 753| 224 | 11856 | 53199 | 
| 44 | 12 src/_pytest/pytester.py | 1203 | 1227| 165 | 12021 | 63418 | 
| 45 | 12 src/_pytest/assertion/truncate.py | 37 | 73| 336 | 12357 | 63418 | 
| 46 | 12 src/_pytest/assertion/rewrite.py | 611 | 622| 127 | 12484 | 63418 | 
| 47 | 12 testing/python/metafunc.py | 315 | 348| 263 | 12747 | 63418 | 
| 48 | 12 src/_pytest/assertion/rewrite.py | 290 | 324| 291 | 13038 | 63418 | 
| 49 | 12 doc/en/example/assertion/failure_demo.py | 164 | 202| 255 | 13293 | 63418 | 
| 50 | 12 src/_pytest/assertion/rewrite.py | 146 | 181| 377 | 13670 | 63418 | 
| 51 | 12 testing/python/collect.py | 279 | 344| 471 | 14141 | 63418 | 
| 52 | 12 src/_pytest/assertion/rewrite.py | 369 | 455| 594 | 14735 | 63418 | 
| 53 | 12 testing/python/approx.py | 313 | 328| 259 | 14994 | 63418 | 
| 54 | 12 testing/python/metafunc.py | 122 | 177| 398 | 15392 | 63418 | 
| 55 | 12 src/_pytest/compat.py | 250 | 329| 493 | 15885 | 63418 | 
| 56 | 12 testing/python/approx.py | 26 | 59| 483 | 16368 | 63418 | 
| 57 | 13 testing/python/fixtures.py | 1012 | 2013| 6212 | 22580 | 88448 | 
| 58 | 13 testing/python/fixtures.py | 2015 | 2177| 968 | 23548 | 88448 | 
| 59 | 13 doc/en/example/assertion/failure_demo.py | 1 | 40| 169 | 23717 | 88448 | 
| 60 | 13 src/_pytest/assertion/__init__.py | 52 | 91| 282 | 23999 | 88448 | 
| 61 | 13 src/_pytest/doctest.py | 457 | 471| 120 | 24119 | 88448 | 
| 62 | 13 testing/python/fixtures.py | 1 | 43| 250 | 24369 | 88448 | 
| 63 | 13 testing/python/approx.py | 61 | 75| 249 | 24618 | 88448 | 
| 64 | 13 testing/python/fixtures.py | 2179 | 3105| 6108 | 30726 | 88448 | 
| 65 | 14 testing/python/raises.py | 1 | 65| 442 | 31168 | 90188 | 
| 66 | 14 testing/python/fixtures.py | 46 | 1010| 6200 | 37368 | 90188 | 
| 67 | 14 testing/python/collect.py | 559 | 658| 662 | 38030 | 90188 | 
| 68 | 15 src/_pytest/python_api.py | 324 | 474| 1903 | 39933 | 96404 | 
| 69 | 15 testing/python/metafunc.py | 350 | 364| 144 | 40077 | 96404 | 
| 70 | 15 src/_pytest/compat.py | 129 | 167| 244 | 40321 | 96404 | 
| 71 | 16 doc/en/_themes/flask_theme_support.py | 1 | 88| 1273 | 41594 | 97677 | 
| 72 | 17 src/_pytest/config/__init__.py | 998 | 1022| 149 | 41743 | 105818 | 
| 73 | 18 src/_pytest/capture.py | 413 | 446| 210 | 41953 | 111776 | 
| 74 | 18 src/_pytest/assertion/rewrite.py | 239 | 254| 126 | 42079 | 111776 | 
| 75 | 18 testing/python/approx.py | 295 | 311| 206 | 42285 | 111776 | 
| 76 | 18 src/_pytest/python_api.py | 200 | 230| 255 | 42540 | 111776 | 
| 77 | 18 testing/python/collect.py | 881 | 918| 321 | 42861 | 111776 | 
| 78 | 18 testing/python/metafunc.py | 1285 | 1323| 243 | 43104 | 111776 | 
| 79 | 18 testing/python/metafunc.py | 366 | 384| 160 | 43264 | 111776 | 
| 80 | 18 src/_pytest/assertion/__init__.py | 94 | 146| 404 | 43668 | 111776 | 
| 81 | 19 src/_pytest/_code/code.py | 800 | 823| 172 | 43840 | 119300 | 
| 82 | 19 testing/python/metafunc.py | 228 | 256| 214 | 44054 | 119300 | 
| 83 | 19 src/_pytest/python_api.py | 1 | 29| 155 | 44209 | 119300 | 
| 84 | 20 bench/bench_argcomplete.py | 1 | 20| 179 | 44388 | 119479 | 
| 85 | 20 testing/python/raises.py | 210 | 225| 134 | 44522 | 119479 | 
| 86 | **20 src/_pytest/assertion/util.py** | 84 | 113| 130 | 44652 | 119479 | 
| 87 | 20 testing/python/fixtures.py | 3107 | 3988| 5314 | 49966 | 119479 | 
| 88 | 20 src/_pytest/_code/code.py | 899 | 930| 277 | 50243 | 119479 | 
| 89 | 20 src/_pytest/pytester.py | 391 | 415| 218 | 50461 | 119479 | 
| 90 | 20 src/_pytest/config/__init__.py | 1060 | 1076| 173 | 50634 | 119479 | 
| 91 | 20 src/_pytest/_code/code.py | 452 | 476| 239 | 50873 | 119479 | 
| 92 | 20 src/_pytest/_code/code.py | 959 | 979| 150 | 51023 | 119479 | 
| 93 | 20 src/_pytest/python_api.py | 526 | 645| 1035 | 52058 | 119479 | 
| 94 | 20 testing/python/collect.py | 661 | 688| 227 | 52285 | 119479 | 
| 95 | 20 testing/python/approx.py | 77 | 95| 241 | 52526 | 119479 | 
| 96 | 20 src/_pytest/python_api.py | 166 | 197| 253 | 52779 | 119479 | 
| 97 | 21 src/_pytest/deprecated.py | 1 | 62| 650 | 53429 | 120487 | 
| 98 | 21 src/_pytest/compat.py | 1 | 59| 349 | 53778 | 120487 | 
| 99 | 21 src/_pytest/_code/code.py | 855 | 896| 294 | 54072 | 120487 | 
| 100 | 22 src/_pytest/_io/saferepr.py | 1 | 17| 124 | 54196 | 121153 | 
| 101 | 23 src/_pytest/fixtures.py | 229 | 266| 361 | 54557 | 132017 | 
| 102 | 23 testing/python/metafunc.py | 868 | 887| 129 | 54686 | 132017 | 
| 103 | 24 testing/example_scripts/issue_519.py | 1 | 31| 350 | 55036 | 132483 | 
| 104 | 24 testing/python/approx.py | 249 | 259| 150 | 55186 | 132483 | 
| 105 | 24 src/_pytest/assertion/truncate.py | 76 | 96| 175 | 55361 | 132483 | 
| 106 | 24 testing/python/raises.py | 227 | 266| 282 | 55643 | 132483 | 
| 107 | 25 testing/python/integration.py | 89 | 126| 197 | 55840 | 135251 | 
| 108 | 25 src/_pytest/_io/saferepr.py | 46 | 63| 181 | 56021 | 135251 | 
| 109 | 25 testing/python/collect.py | 1106 | 1143| 209 | 56230 | 135251 | 
| 110 | **25 src/_pytest/assertion/util.py** | 375 | 392| 137 | 56367 | 135251 | 
| 111 | 25 testing/python/raises.py | 143 | 181| 264 | 56631 | 135251 | 
| 112 | 25 testing/python/metafunc.py | 514 | 553| 329 | 56960 | 135251 | 
| 113 | 25 testing/python/metafunc.py | 179 | 204| 234 | 57194 | 135251 | 
| 114 | 25 src/_pytest/capture.py | 766 | 818| 450 | 57644 | 135251 | 
| 115 | 25 testing/python/approx.py | 348 | 376| 332 | 57976 | 135251 | 
| 116 | 25 src/_pytest/_code/code.py | 632 | 656| 236 | 58212 | 135251 | 
| 117 | 26 src/_pytest/config/argparsing.py | 243 | 269| 223 | 58435 | 138981 | 
| 118 | 27 doc/en/example/multipython.py | 48 | 73| 175 | 58610 | 139427 | 
| 119 | 27 testing/python/integration.py | 71 | 86| 109 | 58719 | 139427 | 
| 120 | 27 src/_pytest/python_api.py | 135 | 163| 242 | 58961 | 139427 | 
| 121 | 27 src/_pytest/doctest.py | 277 | 289| 123 | 59084 | 139427 | 
| 122 | 27 testing/example_scripts/issue_519.py | 34 | 52| 115 | 59199 | 139427 | 
| 123 | 27 src/_pytest/config/argparsing.py | 271 | 282| 125 | 59324 | 139427 | 
| 124 | 27 src/_pytest/_code/code.py | 826 | 852| 220 | 59544 | 139427 | 
| 125 | 27 testing/python/collect.py | 495 | 522| 205 | 59749 | 139427 | 
| 126 | 28 src/pytest.py | 1 | 109| 736 | 60485 | 140163 | 
| 127 | 28 src/_pytest/doctest.py | 218 | 274| 495 | 60980 | 140163 | 
| 128 | 29 src/_pytest/mark/evaluate.py | 48 | 69| 191 | 61171 | 140950 | 
| 129 | **29 src/_pytest/assertion/util.py** | 49 | 81| 271 | 61442 | 140950 | 
| 130 | 29 testing/python/metafunc.py | 405 | 421| 117 | 61559 | 140950 | 
| 131 | 29 testing/python/metafunc.py | 973 | 988| 144 | 61703 | 140950 | 
| 132 | 29 src/_pytest/assertion/__init__.py | 28 | 49| 190 | 61893 | 140950 | 
| 133 | 29 src/_pytest/_io/saferepr.py | 20 | 44| 230 | 62123 | 140950 | 
| 134 | 29 src/_pytest/capture.py | 297 | 311| 144 | 62267 | 140950 | 
| 135 | 29 src/_pytest/_code/code.py | 551 | 594| 307 | 62574 | 140950 | 
| 136 | 29 testing/python/metafunc.py | 1 | 33| 185 | 62759 | 140950 | 
| 137 | 29 testing/python/approx.py | 172 | 186| 411 | 63170 | 140950 | 
| 138 | 30 doc/en/example/xfail_demo.py | 1 | 39| 143 | 63313 | 141094 | 
| 139 | 31 src/_pytest/monkeypatch.py | 1 | 36| 245 | 63558 | 143509 | 
| 140 | 31 src/_pytest/doctest.py | 512 | 552| 268 | 63826 | 143509 | 
| 141 | 31 testing/python/collect.py | 968 | 987| 170 | 63996 | 143509 | 
| 142 | 31 testing/python/approx.py | 241 | 247| 122 | 64118 | 143509 | 
| 143 | 31 testing/python/collect.py | 690 | 710| 133 | 64251 | 143509 | 
| 144 | 31 testing/python/collect.py | 730 | 753| 170 | 64421 | 143509 | 
| 145 | 31 testing/python/collect.py | 920 | 939| 178 | 64599 | 143509 | 
| 146 | 32 src/_pytest/python.py | 1163 | 1191| 271 | 64870 | 155203 | 
| 147 | 33 doc/en/example/assertion/global_testmodule_config/conftest.py | 1 | 15| 0 | 64870 | 155284 | 
| 148 | 33 testing/python/collect.py | 346 | 364| 152 | 65022 | 155284 | 
| 149 | 33 testing/python/approx.py | 200 | 213| 183 | 65205 | 155284 | 
| 150 | 33 testing/python/collect.py | 1 | 33| 225 | 65430 | 155284 | 
| 151 | 34 src/_pytest/reports.py | 30 | 123| 566 | 65996 | 158241 | 
| 152 | 34 testing/python/collect.py | 755 | 786| 204 | 66200 | 158241 | 
| 153 | 34 testing/python/approx.py | 410 | 425| 123 | 66323 | 158241 | 
| 154 | 34 testing/python/approx.py | 445 | 474| 225 | 66548 | 158241 | 
| 155 | 34 src/_pytest/config/__init__.py | 743 | 759| 152 | 66700 | 158241 | 
| 156 | 34 testing/python/collect.py | 788 | 810| 195 | 66895 | 158241 | 
| 157 | 34 testing/python/approx.py | 272 | 283| 201 | 67096 | 158241 | 
| 158 | 34 src/_pytest/python.py | 1229 | 1255| 208 | 67304 | 158241 | 
| 159 | 34 testing/python/collect.py | 989 | 1014| 204 | 67508 | 158241 | 
| 160 | 34 testing/python/integration.py | 306 | 343| 224 | 67732 | 158241 | 
| 161 | 34 testing/python/metafunc.py | 386 | 403| 125 | 67857 | 158241 | 
| 162 | 34 src/_pytest/_code/code.py | 756 | 797| 338 | 68195 | 158241 | 
| 163 | 34 src/_pytest/python.py | 1257 | 1284| 217 | 68412 | 158241 | 
| 164 | 34 src/_pytest/config/argparsing.py | 361 | 395| 359 | 68771 | 158241 | 
| 165 | 34 testing/python/metafunc.py | 1633 | 1653| 201 | 68972 | 158241 | 
| 166 | 34 src/_pytest/deprecated.py | 64 | 100| 358 | 69330 | 158241 | 
| 167 | 35 src/_pytest/skipping.py | 108 | 117| 123 | 69453 | 159719 | 
| 168 | 35 testing/python/metafunc.py | 1740 | 1777| 253 | 69706 | 159719 | 
| 169 | 35 src/_pytest/mark/evaluate.py | 71 | 124| 345 | 70051 | 159719 | 
| 170 | 35 src/_pytest/python_api.py | 32 | 85| 387 | 70438 | 159719 | 
| 171 | 35 testing/python/metafunc.py | 771 | 786| 212 | 70650 | 159719 | 
| 172 | 35 testing/python/collect.py | 196 | 249| 320 | 70970 | 159719 | 
| 173 | 35 testing/python/collect.py | 540 | 557| 182 | 71152 | 159719 | 
| 174 | 35 testing/python/metafunc.py | 788 | 795| 136 | 71288 | 159719 | 
| 175 | 35 testing/python/metafunc.py | 423 | 454| 226 | 71514 | 159719 | 
| 176 | 35 testing/python/metafunc.py | 1165 | 1181| 149 | 71663 | 159719 | 
| 177 | 35 testing/python/metafunc.py | 76 | 120| 448 | 72111 | 159719 | 
| 178 | 35 testing/python/integration.py | 223 | 245| 183 | 72294 | 159719 | 
| 179 | 35 testing/python/metafunc.py | 570 | 601| 270 | 72564 | 159719 | 
| 180 | 35 src/_pytest/_code/code.py | 478 | 535| 368 | 72932 | 159719 | 
| 181 | 35 testing/python/integration.py | 364 | 393| 199 | 73131 | 159719 | 
| 182 | 35 testing/python/approx.py | 487 | 508| 206 | 73337 | 159719 | 
| 183 | 35 src/_pytest/capture.py | 741 | 763| 220 | 73557 | 159719 | 
| 184 | 35 src/_pytest/_code/code.py | 933 | 956| 174 | 73731 | 159719 | 
| 185 | 35 src/_pytest/python.py | 1207 | 1226| 189 | 73920 | 159719 | 
| 186 | 36 testing/python/setup_only.py | 123 | 153| 183 | 74103 | 161252 | 
| 187 | 36 src/_pytest/_code/code.py | 658 | 689| 303 | 74406 | 161252 | 
| 188 | 36 testing/python/metafunc.py | 456 | 487| 221 | 74627 | 161252 | 
| 189 | 36 testing/python/metafunc.py | 1696 | 1719| 222 | 74849 | 161252 | 
| 190 | 36 src/_pytest/config/argparsing.py | 226 | 241| 127 | 74976 | 161252 | 
| 191 | 36 testing/python/approx.py | 97 | 110| 233 | 75209 | 161252 | 
| 192 | 36 testing/python/collect.py | 35 | 57| 187 | 75396 | 161252 | 
| 193 | 36 testing/python/setup_only.py | 156 | 182| 167 | 75563 | 161252 | 
| 194 | 37 src/_pytest/terminal.py | 921 | 946| 199 | 75762 | 169413 | 
| 195 | 37 testing/python/collect.py | 59 | 77| 189 | 75951 | 169413 | 
| 196 | 37 testing/python/metafunc.py | 1112 | 1132| 162 | 76113 | 169413 | 
| 197 | 37 src/_pytest/python_api.py | 88 | 118| 203 | 76316 | 169413 | 
| 198 | 38 testing/example_scripts/issue88_initial_file_multinodes/conftest.py | 1 | 15| 0 | 76316 | 169466 | 
| 199 | 38 src/_pytest/terminal.py | 255 | 279| 161 | 76477 | 169466 | 
| 200 | 38 testing/python/setup_only.py | 33 | 59| 167 | 76644 | 169466 | 
| 201 | 38 src/_pytest/_code/code.py | 1 | 86| 558 | 77202 | 169466 | 
| 202 | 38 src/_pytest/_io/saferepr.py | 66 | 79| 140 | 77342 | 169466 | 
| 203 | 38 testing/python/metafunc.py | 890 | 919| 227 | 77569 | 169466 | 
| 204 | 39 doc/en/example/costlysetup/sub_b/__init__.py | 1 | 2| 0 | 77569 | 169467 | 
| 205 | 39 src/_pytest/terminal.py | 872 | 902| 250 | 77819 | 169467 | 
| 206 | 39 testing/python/collect.py | 474 | 493| 156 | 77975 | 169467 | 
| 207 | 39 src/_pytest/terminal.py | 987 | 1006| 171 | 78146 | 169467 | 
| 208 | 39 testing/python/integration.py | 1 | 35| 239 | 78385 | 169467 | 
| 209 | 39 testing/python/setup_only.py | 185 | 200| 119 | 78504 | 169467 | 
| 210 | 39 src/_pytest/_code/code.py | 691 | 717| 210 | 78714 | 169467 | 
| 211 | 39 testing/python/metafunc.py | 921 | 943| 176 | 78890 | 169467 | 
| 212 | 39 testing/python/integration.py | 345 | 362| 134 | 79024 | 169467 | 
| 213 | 39 testing/python/approx.py | 476 | 485| 127 | 79151 | 169467 | 
| 214 | 39 testing/python/setup_only.py | 62 | 93| 190 | 79341 | 169467 | 


### Hint

```
hmmm yes, this ~kinda makes sense as `bytes` objects are sequences of integers -- we should maybe just omit the "contains more items" messaging for bytes objects?
```

## Patch

```diff
diff --git a/src/_pytest/assertion/util.py b/src/_pytest/assertion/util.py
--- a/src/_pytest/assertion/util.py
+++ b/src/_pytest/assertion/util.py
@@ -254,17 +254,38 @@ def _compare_eq_iterable(left, right, verbose=0):
 
 
 def _compare_eq_sequence(left, right, verbose=0):
+    comparing_bytes = isinstance(left, bytes) and isinstance(right, bytes)
     explanation = []
     len_left = len(left)
     len_right = len(right)
     for i in range(min(len_left, len_right)):
         if left[i] != right[i]:
+            if comparing_bytes:
+                # when comparing bytes, we want to see their ascii representation
+                # instead of their numeric values (#5260)
+                # using a slice gives us the ascii representation:
+                # >>> s = b'foo'
+                # >>> s[0]
+                # 102
+                # >>> s[0:1]
+                # b'f'
+                left_value = left[i : i + 1]
+                right_value = right[i : i + 1]
+            else:
+                left_value = left[i]
+                right_value = right[i]
+
             explanation += [
-                "At index {} diff: {!r} != {!r}".format(i, left[i], right[i])
+                "At index {} diff: {!r} != {!r}".format(i, left_value, right_value)
             ]
             break
-    len_diff = len_left - len_right
 
+    if comparing_bytes:
+        # when comparing bytes, it doesn't help to show the "sides contain one or more items"
+        # longer explanation, so skip it
+        return explanation
+
+    len_diff = len_left - len_right
     if len_diff:
         if len_diff > 0:
             dir_with_more = "Left"

```

## Test Patch

```diff
diff --git a/testing/test_assertion.py b/testing/test_assertion.py
--- a/testing/test_assertion.py
+++ b/testing/test_assertion.py
@@ -331,6 +331,27 @@ def test_multiline_text_diff(self):
         assert "- spam" in diff
         assert "+ eggs" in diff
 
+    def test_bytes_diff_normal(self):
+        """Check special handling for bytes diff (#5260)"""
+        diff = callequal(b"spam", b"eggs")
+
+        assert diff == [
+            "b'spam' == b'eggs'",
+            "At index 0 diff: b's' != b'e'",
+            "Use -v to get the full diff",
+        ]
+
+    def test_bytes_diff_verbose(self):
+        """Check special handling for bytes diff (#5260)"""
+        diff = callequal(b"spam", b"eggs", verbose=True)
+        assert diff == [
+            "b'spam' == b'eggs'",
+            "At index 0 diff: b's' != b'e'",
+            "Full diff:",
+            "- b'spam'",
+            "+ b'eggs'",
+        ]
+
     def test_list(self):
         expl = callequal([0, 1], [0, 2])
         assert len(expl) > 1

```


## Code snippets

### 1 - src/_pytest/assertion/rewrite.py:

Start line: 857, End line: 894

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
            expl = "{} {} {}".format(left_expl, sym, next_expl)
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
### 2 - src/_pytest/assertion/rewrite.py:

Start line: 765, End line: 798

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
            call = ast.Call(app, [expl_format], [])
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
### 3 - src/_pytest/assertion/rewrite.py:

Start line: 800, End line: 812

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
        explanation = "({} {} {})".format(left_expl, symbol, right_expl)
        res = self.assign(ast.BinOp(left_expr, binop.op, right_expr))
        return res, explanation
```
### 4 - src/_pytest/assertion/rewrite.py:

Start line: 755, End line: 763

```python
class AssertionRewriter(ast.NodeVisitor):

    def visit_Name(self, name):
        # Display the repr of the name if it's a local variable or
        # _should_repr_global_name() thinks it's acceptable.
        locs = ast.Call(self.builtin("locals"), [], [])
        inlocs = ast.Compare(ast.Str(name.id), [ast.In()], [locs])
        dorepr = self.helper("_should_repr_global_name", name)
        test = ast.BoolOp(ast.Or(), [inlocs, dorepr])
        expr = ast.IfExp(test, self.display(name), ast.Str(name.id))
        return name, self.explanation_param(expr)
```
### 5 - src/_pytest/assertion/rewrite.py:

Start line: 1, End line: 27

```python
"""Rewrite assertion AST to produce nice error messages"""
import ast
import errno
import importlib.machinery
import importlib.util
import itertools
import marshal
import os
import struct
import sys
import types

import atomicwrites

from _pytest._io.saferepr import saferepr
from _pytest._version import version
from _pytest.assertion import util
from _pytest.assertion.util import (  # noqa: F401
    format_explanation as _format_explanation,
)
from _pytest.pathlib import fnmatch_ex
from _pytest.pathlib import PurePath

# pytest caches rewritten pycs in __pycache__.
PYTEST_TAG = "{}-pytest-{}".format(sys.implementation.cache_tag, version)
PYC_EXT = ".py" + (__debug__ and "c" or "o")
PYC_TAIL = "." + PYTEST_TAG + PYC_EXT
```
### 6 - src/_pytest/assertion/util.py:

Start line: 256, End line: 301

```python
def _compare_eq_sequence(left, right, verbose=0):
    explanation = []
    len_left = len(left)
    len_right = len(right)
    for i in range(min(len_left, len_right)):
        if left[i] != right[i]:
            explanation += [
                "At index {} diff: {!r} != {!r}".format(i, left[i], right[i])
            ]
            break
    len_diff = len_left - len_right

    if len_diff:
        if len_diff > 0:
            dir_with_more = "Left"
            extra = saferepr(left[len_right])
        else:
            len_diff = 0 - len_diff
            dir_with_more = "Right"
            extra = saferepr(right[len_left])

        if len_diff == 1:
            explanation += [
                "{} contains one more item: {}".format(dir_with_more, extra)
            ]
        else:
            explanation += [
                "%s contains %d more items, first extra item: %s"
                % (dir_with_more, len_diff, extra)
            ]
    return explanation


def _compare_eq_set(left, right, verbose=0):
    explanation = []
    diff_left = left - right
    diff_right = right - left
    if diff_left:
        explanation.append("Extra items in the left set:")
        for item in diff_left:
            explanation.append(saferepr(item))
    if diff_right:
        explanation.append("Extra items in the right set:")
        for item in diff_right:
            explanation.append(saferepr(item))
    return explanation
```
### 7 - src/_pytest/assertion/rewrite.py:

Start line: 663, End line: 726

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
                filename=self.module_path,
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
        exc = ast.Call(err_name, [fmt], [])
        raise_ = ast.Raise(exc, None)

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
### 8 - src/_pytest/assertion/util.py:

Start line: 304, End line: 340

```python
def _compare_eq_dict(left, right, verbose=0):
    explanation = []
    set_left = set(left)
    set_right = set(right)
    common = set_left.intersection(set_right)
    same = {k: left[k] for k in common if left[k] == right[k]}
    if same and verbose < 2:
        explanation += ["Omitting %s identical items, use -vv to show" % len(same)]
    elif same:
        explanation += ["Common items:"]
        explanation += pprint.pformat(same).splitlines()
    diff = {k for k in common if left[k] != right[k]}
    if diff:
        explanation += ["Differing items:"]
        for k in diff:
            explanation += [saferepr({k: left[k]}) + " != " + saferepr({k: right[k]})]
    extra_left = set_left - set_right
    len_extra_left = len(extra_left)
    if len_extra_left:
        explanation.append(
            "Left contains %d more item%s:"
            % (len_extra_left, "" if len_extra_left == 1 else "s")
        )
        explanation.extend(
            pprint.pformat({k: left[k] for k in extra_left}).splitlines()
        )
    extra_right = set_right - set_left
    len_extra_right = len(extra_right)
    if len_extra_right:
        explanation.append(
            "Right contains %d more item%s:"
            % (len_extra_right, "" if len_extra_right == 1 else "s")
        )
        explanation.extend(
            pprint.pformat({k: right[k] for k in extra_right}).splitlines()
        )
    return explanation
```
### 9 - doc/en/example/assertion/failure_demo.py:

Start line: 43, End line: 121

```python
class TestSpecialisedExplanations:
    def test_eq_text(self):
        assert "spam" == "eggs"

    def test_eq_similar_text(self):
        assert "foo 1 bar" == "foo 2 bar"

    def test_eq_multiline_text(self):
        assert "foo\nspam\nbar" == "foo\neggs\nbar"

    def test_eq_long_text(self):
        a = "1" * 100 + "a" + "2" * 100
        b = "1" * 100 + "b" + "2" * 100
        assert a == b

    def test_eq_long_text_multiline(self):
        a = "1\n" * 100 + "a" + "2\n" * 100
        b = "1\n" * 100 + "b" + "2\n" * 100
        assert a == b

    def test_eq_list(self):
        assert [0, 1, 2] == [0, 1, 3]

    def test_eq_list_long(self):
        a = [0] * 100 + [1] + [3] * 100
        b = [0] * 100 + [2] + [3] * 100
        assert a == b

    def test_eq_dict(self):
        assert {"a": 0, "b": 1, "c": 0} == {"a": 0, "b": 2, "d": 0}

    def test_eq_set(self):
        assert {0, 10, 11, 12} == {0, 20, 21}

    def test_eq_longer_list(self):
        assert [1, 2] == [1, 2, 3]

    def test_in_list(self):
        assert 1 in [0, 2, 3, 4, 5]

    def test_not_in_text_multiline(self):
        text = "some multiline\ntext\nwhich\nincludes foo\nand a\ntail"
        assert "foo" not in text

    def test_not_in_text_single(self):
        text = "single foo line"
        assert "foo" not in text

    def test_not_in_text_single_long(self):
        text = "head " * 50 + "foo " + "tail " * 20
        assert "foo" not in text

    def test_not_in_text_single_long_term(self):
        text = "head " * 50 + "f" * 70 + "tail " * 20
        assert "f" * 70 not in text

    def test_eq_dataclass(self):
        from dataclasses import dataclass

        @dataclass
        class Foo:
            a: int
            b: str

        left = Foo(1, "b")
        right = Foo(1, "c")
        assert left == right

    def test_eq_attrs(self):
        import attr

        @attr.s
        class Foo:
            a = attr.ib()
            b = attr.ib()

        left = Foo(1, "b")
        right = Foo(1, "c")
        assert left == right
```
### 10 - src/_pytest/assertion/rewrite.py:

Start line: 841, End line: 855

```python
class AssertionRewriter(ast.NodeVisitor):

    def visit_Starred(self, starred):
        # From Python 3.5, a Starred node can appear in a function call
        res, expl = self.visit(starred.value)
        new_starred = ast.Starred(res, starred.ctx)
        return new_starred, "*" + expl

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
### 13 - src/_pytest/assertion/util.py:

Start line: 166, End line: 226

```python
def _diff_text(left, right, verbose=0):
    """Return the explanation for the diff between text or bytes.

    Unless --verbose is used this will skip leading and trailing
    characters which are identical to keep the diff minimal.

    If the input are bytes they will be safely converted to text.
    """
    from difflib import ndiff

    explanation = []

    def escape_for_readable_diff(binary_text):
        """
        Ensures that the internal string is always valid unicode, converting any bytes safely to valid unicode.
        This is done using repr() which then needs post-processing to fix the encompassing quotes and un-escape
        newlines and carriage returns (#429).
        """
        r = str(repr(binary_text)[1:-1])
        r = r.replace(r"\n", "\n")
        r = r.replace(r"\r", "\r")
        return r

    if isinstance(left, bytes):
        left = escape_for_readable_diff(left)
    if isinstance(right, bytes):
        right = escape_for_readable_diff(right)
    if verbose < 1:
        i = 0  # just in case left or right has zero length
        for i in range(min(len(left), len(right))):
            if left[i] != right[i]:
                break
        if i > 42:
            i -= 10  # Provide some context
            explanation = [
                "Skipping %s identical leading characters in diff, use -v to show" % i
            ]
            left = left[i:]
            right = right[i:]
        if len(left) == len(right):
            for i in range(len(left)):
                if left[-i] != right[-i]:
                    break
            if i > 42:
                i -= 10  # Provide some context
                explanation += [
                    "Skipping {} identical trailing "
                    "characters in diff, use -v to show".format(i)
                ]
                left = left[:-i]
                right = right[:-i]
    keepends = True
    if left.isspace() or right.isspace():
        left = repr(str(left))
        right = repr(str(right))
        explanation += ["Strings contain only whitespace, escaping them using repr()"]
    explanation += [
        line.strip("\n")
        for line in ndiff(left.splitlines(keepends), right.splitlines(keepends))
    ]
    return explanation
```
### 14 - src/_pytest/assertion/util.py:

Start line: 116, End line: 163

```python
def assertrepr_compare(config, op, left, right):
    """Return specialised explanations for some operators/operands"""
    width = 80 - 15 - len(op) - 2  # 15 chars indentation, 1 space around op
    left_repr = saferepr(left, maxsize=int(width // 2))
    right_repr = saferepr(right, maxsize=width - len(left_repr))

    summary = "{} {} {}".format(left_repr, op, right_repr)

    verbose = config.getoption("verbose")
    explanation = None
    try:
        if op == "==":
            if istext(left) and istext(right):
                explanation = _diff_text(left, right, verbose)
            else:
                if issequence(left) and issequence(right):
                    explanation = _compare_eq_sequence(left, right, verbose)
                elif isset(left) and isset(right):
                    explanation = _compare_eq_set(left, right, verbose)
                elif isdict(left) and isdict(right):
                    explanation = _compare_eq_dict(left, right, verbose)
                elif type(left) == type(right) and (isdatacls(left) or isattrs(left)):
                    type_fn = (isdatacls, isattrs)
                    explanation = _compare_eq_cls(left, right, verbose, type_fn)
                elif verbose > 0:
                    explanation = _compare_eq_verbose(left, right)
                if isiterable(left) and isiterable(right):
                    expl = _compare_eq_iterable(left, right, verbose)
                    if explanation is not None:
                        explanation.extend(expl)
                    else:
                        explanation = expl
        elif op == "not in":
            if istext(left) and istext(right):
                explanation = _notin_text(left, right, verbose)
    except outcomes.Exit:
        raise
    except Exception:
        explanation = [
            "(pytest_assertion plugin: representation of details failed.  "
            "Probably an object has a faulty __repr__.)",
            str(_pytest._code.ExceptionInfo.from_current()),
        ]

    if not explanation:
        return None

    return [summary] + explanation
```
### 20 - src/_pytest/assertion/util.py:

Start line: 229, End line: 253

```python
def _compare_eq_verbose(left, right):
    keepends = True
    left_lines = repr(left).splitlines(keepends)
    right_lines = repr(right).splitlines(keepends)

    explanation = []
    explanation += ["-" + line for line in left_lines]
    explanation += ["+" + line for line in right_lines]

    return explanation


def _compare_eq_iterable(left, right, verbose=0):
    if not verbose:
        return ["Use -v to get the full diff"]
    # dynamic import to speedup pytest
    import difflib

    left_formatting = pprint.pformat(left).splitlines()
    right_formatting = pprint.pformat(right).splitlines()
    explanation = ["Full diff:"]
    explanation.extend(
        line.strip() for line in difflib.ndiff(left_formatting, right_formatting)
    )
    return explanation
```
### 25 - src/_pytest/assertion/util.py:

Start line: 343, End line: 372

```python
def _compare_eq_cls(left, right, verbose, type_fns):
    isdatacls, isattrs = type_fns
    if isdatacls(left):
        all_fields = left.__dataclass_fields__
        fields_to_check = [field for field, info in all_fields.items() if info.compare]
    elif isattrs(left):
        all_fields = left.__attrs_attrs__
        fields_to_check = [field.name for field in all_fields if field.cmp]

    same = []
    diff = []
    for field in fields_to_check:
        if getattr(left, field) == getattr(right, field):
            same.append(field)
        else:
            diff.append(field)

    explanation = []
    if same and verbose < 2:
        explanation.append("Omitting %s identical items, use -vv to show" % len(same))
    elif same:
        explanation += ["Matching attributes:"]
        explanation += pprint.pformat(same).splitlines()
    if diff:
        explanation += ["Differing attributes:"]
        for field in diff:
            explanation += [
                ("%s: %r != %r") % (field, getattr(left, field), getattr(right, field))
            ]
    return explanation
```
### 38 - src/_pytest/assertion/util.py:

Start line: 1, End line: 29

```python
"""Utilities for assertion debugging"""
import pprint
from collections.abc import Sequence

import _pytest._code
from _pytest import outcomes
from _pytest._io.saferepr import saferepr

# The _reprcompare attribute on the util module is used by the new assertion
# interpretation code and assertion rewriter to detect this plugin was
# loaded and in turn call the hooks defined here as part of the
# DebugInterpreter.
_reprcompare = None


def format_explanation(explanation):
    """This formats an explanation

    Normally all embedded newlines are escaped, however there are
    three exceptions: \n{, \n} and \n~.  The first two are intended
    cover nested explanations, see function and attribute explanations
    for examples (.visit_Call(), visit_Attribute()).  The last one is
    for when one explanation needs to span multiple lines, e.g. when
    displaying diffs.
    """
    explanation = explanation
    lines = _split_explanation(explanation)
    result = _format_lines(lines)
    return "\n".join(result)
```
### 86 - src/_pytest/assertion/util.py:

Start line: 84, End line: 113

```python
def issequence(x):
    return isinstance(x, Sequence) and not isinstance(x, str)


def istext(x):
    return isinstance(x, str)


def isdict(x):
    return isinstance(x, dict)


def isset(x):
    return isinstance(x, (set, frozenset))


def isdatacls(obj):
    return getattr(obj, "__dataclass_fields__", None) is not None


def isattrs(obj):
    return getattr(obj, "__attrs_attrs__", None) is not None


def isiterable(obj):
    try:
        iter(obj)
        return not istext(obj)
    except TypeError:
        return False
```
### 110 - src/_pytest/assertion/util.py:

Start line: 375, End line: 392

```python
def _notin_text(term, text, verbose=0):
    index = text.find(term)
    head = text[:index]
    tail = text[index + len(term) :]
    correct_text = head + tail
    diff = _diff_text(correct_text, text, verbose)
    newdiff = ["%s is contained here:" % saferepr(term, maxsize=42)]
    for line in diff:
        if line.startswith("Skipping"):
            continue
        if line.startswith("- "):
            continue
        if line.startswith("+ "):
            newdiff.append("  " + line[2:])
        else:
            newdiff.append(line)
    return newdiff
```
### 129 - src/_pytest/assertion/util.py:

Start line: 49, End line: 81

```python
def _format_lines(lines):
    """Format the individual lines

    This will replace the '{', '}' and '~' characters of our mini
    formatting language with the proper 'where ...', 'and ...' and ' +
    ...' text, taking care of indentation along the way.

    Return a list of formatted lines.
    """
    result = lines[:1]
    stack = [0]
    stackcnt = [0]
    for line in lines[1:]:
        if line.startswith("{"):
            if stackcnt[-1]:
                s = "and   "
            else:
                s = "where "
            stack.append(len(result))
            stackcnt[-1] += 1
            stackcnt.append(0)
            result.append(" +" + "  " * (len(stack) - 1) + s + line[1:])
        elif line.startswith("}"):
            stack.pop()
            stackcnt.pop()
            result[stack[-1]] += line[1:]
        else:
            assert line[0] in ["~", ">"]
            stack[-1] += 1
            indent = len(stack) if line.startswith("~") else len(stack) - 1
            result.append("  " * indent + line[1:])
    assert len(stack) == 1
    return result
```
