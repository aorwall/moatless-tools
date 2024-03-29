# pytest-dev__pytest-7673

| **pytest-dev/pytest** | `75af2bfa06436752165df884d4666402529b1d6a` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 195 |
| **Any found context length** | 195 |
| **Avg pos** | 1.0 |
| **Min pos** | 1 |
| **Max pos** | 1 |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/src/_pytest/logging.py b/src/_pytest/logging.py
--- a/src/_pytest/logging.py
+++ b/src/_pytest/logging.py
@@ -439,7 +439,8 @@ def set_level(self, level: Union[int, str], logger: Optional[str] = None) -> Non
         # Save the original log-level to restore it during teardown.
         self._initial_logger_levels.setdefault(logger, logger_obj.level)
         logger_obj.setLevel(level)
-        self._initial_handler_level = self.handler.level
+        if self._initial_handler_level is None:
+            self._initial_handler_level = self.handler.level
         self.handler.setLevel(level)
 
     @contextmanager

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| src/_pytest/logging.py | 442 | 442 | 1 | 1 | 195


## Problem Statement

```
logging: handler level restored incorrectly if caplog.set_level is called more than once
pytest version: 6.0.1

The fix in #7571 (backported to 6.0.1) has a bug where it does a "set" instead of "setdefault" to the `_initial_handler_level`. So if there are multiple calls to `caplog.set_level`, the level will be restored to that of the one-before-last call, instead of the value before the test.

Will submit a fix for this shortly.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 src/_pytest/logging.py** | 424 | 443| 195 | 195 | 6354 | 
| 2 | **1 src/_pytest/logging.py** | 445 | 465| 167 | 362 | 6354 | 
| 3 | **1 src/_pytest/logging.py** | 342 | 369| 215 | 577 | 6354 | 
| 4 | **1 src/_pytest/logging.py** | 736 | 777| 334 | 911 | 6354 | 
| 5 | **1 src/_pytest/logging.py** | 779 | 815| 272 | 1183 | 6354 | 
| 6 | **1 src/_pytest/logging.py** | 485 | 511| 213 | 1396 | 6354 | 
| 7 | **1 src/_pytest/logging.py** | 468 | 482| 143 | 1539 | 6354 | 
| 8 | **1 src/_pytest/logging.py** | 284 | 311| 204 | 1743 | 6354 | 
| 9 | 2 src/_pytest/faulthandler.py | 72 | 84| 132 | 1875 | 7252 | 
| 10 | **2 src/_pytest/logging.py** | 695 | 733| 376 | 2251 | 7252 | 
| 11 | 3 src/_pytest/capture.py | 151 | 172| 192 | 2443 | 14570 | 
| 12 | **3 src/_pytest/logging.py** | 314 | 339| 210 | 2653 | 14570 | 
| 13 | 4 testing/python/fixtures.py | 1046 | 2038| 6156 | 8809 | 41228 | 
| 14 | **4 src/_pytest/logging.py** | 1 | 42| 303 | 9112 | 41228 | 
| 15 | **4 src/_pytest/logging.py** | 655 | 676| 200 | 9312 | 41228 | 
| 16 | 5 src/_pytest/debugging.py | 112 | 149| 298 | 9610 | 44219 | 
| 17 | 6 src/_pytest/pytester.py | 256 | 302| 363 | 9973 | 57211 | 
| 18 | **6 src/_pytest/logging.py** | 45 | 88| 407 | 10380 | 57211 | 
| 19 | 7 src/_pytest/setuponly.py | 59 | 95| 288 | 10668 | 57907 | 
| 20 | 7 src/_pytest/faulthandler.py | 86 | 115| 238 | 10906 | 57907 | 
| 21 | 8 testing/conftest.py | 1 | 64| 383 | 11289 | 59516 | 
| 22 | 9 src/_pytest/cacheprovider.py | 291 | 351| 543 | 11832 | 63861 | 
| 23 | 9 testing/python/fixtures.py | 3513 | 4257| 4871 | 16703 | 63861 | 
| 24 | 10 src/pytest/__init__.py | 1 | 98| 670 | 17373 | 64531 | 
| 25 | 10 src/_pytest/faulthandler.py | 49 | 70| 217 | 17590 | 64531 | 
| 26 | 10 src/_pytest/debugging.py | 151 | 235| 641 | 18231 | 64531 | 
| 27 | 11 src/_pytest/main.py | 394 | 426| 220 | 18451 | 71164 | 
| 28 | **11 src/_pytest/logging.py** | 198 | 281| 582 | 19033 | 71164 | 
| 29 | 11 src/_pytest/capture.py | 736 | 818| 616 | 19649 | 71164 | 
| 30 | 11 src/_pytest/pytester.py | 140 | 162| 248 | 19897 | 71164 | 
| 31 | 12 testing/python/collect.py | 398 | 425| 193 | 20090 | 80929 | 
| 32 | 13 src/_pytest/python.py | 1323 | 1355| 260 | 20350 | 94371 | 
| 33 | 13 testing/python/collect.py | 924 | 963| 352 | 20702 | 94371 | 
| 34 | 13 src/_pytest/faulthandler.py | 25 | 46| 196 | 20898 | 94371 | 
| 35 | 14 src/_pytest/fixtures.py | 1228 | 1243| 127 | 21025 | 108355 | 
| 36 | 14 src/_pytest/main.py | 429 | 508| 746 | 21771 | 108355 | 
| 37 | 14 src/_pytest/faulthandler.py | 1 | 22| 125 | 21896 | 108355 | 
| 38 | 14 src/_pytest/fixtures.py | 684 | 704| 201 | 22097 | 108355 | 
| 39 | 14 src/_pytest/pytester.py | 336 | 368| 251 | 22348 | 108355 | 
| 40 | 14 src/_pytest/fixtures.py | 1 | 122| 819 | 23167 | 108355 | 
| 41 | **14 src/_pytest/logging.py** | 384 | 403| 147 | 23314 | 108355 | 
| 42 | 14 testing/python/fixtures.py | 2486 | 3511| 6204 | 29518 | 108355 | 
| 43 | 14 src/_pytest/fixtures.py | 1210 | 1225| 113 | 29631 | 108355 | 
| 44 | 14 src/_pytest/pytester.py | 235 | 254| 199 | 29830 | 108355 | 
| 45 | 14 src/_pytest/pytester.py | 394 | 411| 165 | 29995 | 108355 | 
| 46 | 14 src/_pytest/capture.py | 882 | 899| 152 | 30147 | 108355 | 
| 47 | 14 testing/python/collect.py | 427 | 444| 123 | 30270 | 108355 | 
| 48 | 15 src/_pytest/nodes.py | 1 | 49| 321 | 30591 | 113387 | 
| 49 | 15 testing/python/fixtures.py | 2040 | 2484| 2812 | 33403 | 113387 | 
| 50 | 15 src/_pytest/main.py | 526 | 551| 232 | 33635 | 113387 | 
| 51 | 15 src/_pytest/capture.py | 919 | 933| 149 | 33784 | 113387 | 
| 52 | 16 testing/python/integration.py | 331 | 369| 237 | 34021 | 116408 | 
| 53 | 16 src/_pytest/pytester.py | 370 | 392| 151 | 34172 | 116408 | 
| 54 | 17 src/_pytest/hookspec.py | 541 | 576| 245 | 34417 | 123025 | 
| 55 | 18 testing/python/metafunc.py | 1215 | 1244| 211 | 34628 | 137551 | 
| 56 | 18 src/_pytest/cacheprovider.py | 505 | 545| 357 | 34985 | 137551 | 
| 57 | 18 src/_pytest/capture.py | 33 | 64| 222 | 35207 | 137551 | 
| 58 | 18 testing/python/metafunc.py | 114 | 136| 190 | 35397 | 137551 | 
| 59 | 18 src/_pytest/setuponly.py | 30 | 56| 242 | 35639 | 137551 | 
| 60 | 18 testing/python/fixtures.py | 86 | 1044| 6146 | 41785 | 137551 | 
| 61 | 18 src/_pytest/cacheprovider.py | 400 | 455| 416 | 42201 | 137551 | 
| 62 | 18 src/_pytest/capture.py | 936 | 951| 153 | 42354 | 137551 | 
| 63 | 19 scripts/release.py | 108 | 127| 126 | 42480 | 138504 | 
| 64 | 19 src/_pytest/capture.py | 642 | 734| 792 | 43272 | 138504 | 
| 65 | 19 src/_pytest/python.py | 1385 | 1444| 454 | 43726 | 138504 | 
| 66 | 19 src/_pytest/debugging.py | 367 | 389| 205 | 43931 | 138504 | 
| 67 | **19 src/_pytest/logging.py** | 624 | 653| 254 | 44185 | 138504 | 
| 68 | 19 src/_pytest/capture.py | 821 | 845| 246 | 44431 | 138504 | 
| 69 | 20 src/_pytest/doctest.py | 489 | 523| 297 | 44728 | 144150 | 
| 70 | 20 src/_pytest/main.py | 565 | 598| 371 | 45099 | 144150 | 
| 71 | 20 src/_pytest/pytester.py | 1 | 59| 355 | 45454 | 144150 | 
| 72 | 21 doc/en/example/xfail_demo.py | 1 | 39| 143 | 45597 | 144294 | 
| 73 | 21 src/_pytest/doctest.py | 390 | 406| 133 | 45730 | 144294 | 
| 74 | 21 src/_pytest/python.py | 216 | 248| 342 | 46072 | 144294 | 
| 75 | 21 src/_pytest/capture.py | 552 | 625| 514 | 46586 | 144294 | 
| 76 | 22 src/_pytest/stepwise.py | 84 | 109| 202 | 46788 | 145144 | 
| 77 | 22 testing/python/collect.py | 81 | 114| 241 | 47029 | 145144 | 
| 78 | **22 src/_pytest/logging.py** | 678 | 693| 157 | 47186 | 145144 | 
| 79 | 22 testing/python/collect.py | 986 | 1010| 168 | 47354 | 145144 | 
| 80 | **22 src/_pytest/logging.py** | 576 | 592| 149 | 47503 | 145144 | 
| 81 | 23 testing/example_scripts/issue_519.py | 36 | 54| 111 | 47614 | 145629 | 
| 82 | 23 testing/python/collect.py | 197 | 250| 318 | 47932 | 145629 | 
| 83 | 23 src/_pytest/main.py | 600 | 616| 174 | 48106 | 145629 | 
| 84 | 23 src/_pytest/capture.py | 253 | 347| 729 | 48835 | 145629 | 
| 85 | 24 src/_pytest/warnings.py | 192 | 219| 275 | 49110 | 147259 | 
| 86 | 24 src/_pytest/main.py | 291 | 309| 139 | 49249 | 147259 | 
| 87 | 24 src/_pytest/python.py | 1357 | 1382| 262 | 49511 | 147259 | 
| 88 | 24 src/_pytest/cacheprovider.py | 226 | 241| 130 | 49641 | 147259 | 
| 89 | 24 src/_pytest/main.py | 349 | 373| 237 | 49878 | 147259 | 
| 90 | 24 src/_pytest/debugging.py | 95 | 110| 152 | 50030 | 147259 | 
| 91 | 25 src/_pytest/config/exceptions.py | 1 | 8| 0 | 50030 | 147301 | 
| 92 | 25 testing/python/fixtures.py | 1 | 83| 492 | 50522 | 147301 | 
| 93 | 25 src/_pytest/main.py | 716 | 753| 379 | 50901 | 147301 | 
| 94 | 25 testing/python/collect.py | 1194 | 1220| 179 | 51080 | 147301 | 
| 95 | 26 src/_pytest/runner.py | 187 | 213| 230 | 51310 | 150982 | 
| 96 | **26 src/_pytest/logging.py** | 371 | 382| 121 | 51431 | 150982 | 
| 97 | 26 src/_pytest/cacheprovider.py | 178 | 223| 369 | 51800 | 150982 | 
| 98 | 26 src/_pytest/fixtures.py | 1007 | 1028| 186 | 51986 | 150982 | 
| 99 | 26 src/_pytest/main.py | 51 | 156| 740 | 52726 | 150982 | 
| 100 | 26 src/_pytest/debugging.py | 285 | 307| 198 | 52924 | 150982 | 
| 101 | 26 src/_pytest/runner.py | 144 | 167| 186 | 53110 | 150982 | 
| 102 | 26 testing/python/integration.py | 42 | 74| 250 | 53360 | 150982 | 
| 103 | 26 src/_pytest/main.py | 553 | 563| 112 | 53472 | 150982 | 
| 104 | 26 testing/example_scripts/issue_519.py | 1 | 33| 373 | 53845 | 150982 | 
| 105 | 26 src/_pytest/capture.py | 864 | 879| 146 | 53991 | 150982 | 
| 106 | 27 src/_pytest/outcomes.py | 48 | 63| 124 | 54115 | 152719 | 
| 107 | 27 testing/python/collect.py | 1319 | 1329| 110 | 54225 | 152719 | 
| 108 | 27 testing/python/collect.py | 1253 | 1276| 195 | 54420 | 152719 | 
| 109 | 27 src/_pytest/fixtures.py | 894 | 920| 239 | 54659 | 152719 | 
| 110 | 27 src/_pytest/fixtures.py | 123 | 163| 281 | 54940 | 152719 | 
| 111 | 27 src/_pytest/main.py | 312 | 329| 162 | 55102 | 152719 | 
| 112 | 27 src/_pytest/debugging.py | 237 | 282| 384 | 55486 | 152719 | 
| 113 | 27 src/_pytest/fixtures.py | 306 | 313| 106 | 55592 | 152719 | 
| 114 | 27 src/_pytest/fixtures.py | 781 | 801| 178 | 55770 | 152719 | 
| 115 | 27 src/_pytest/python.py | 79 | 122| 306 | 56076 | 152719 | 
| 116 | 27 src/_pytest/stepwise.py | 1 | 31| 190 | 56266 | 152719 | 
| 117 | 28 src/_pytest/assertion/__init__.py | 154 | 180| 256 | 56522 | 154157 | 
| 118 | 29 src/_pytest/setupplan.py | 1 | 41| 269 | 56791 | 154427 | 
| 119 | 29 src/_pytest/pytester.py | 570 | 597| 188 | 56979 | 154427 | 
| 120 | 29 src/_pytest/capture.py | 67 | 88| 222 | 57201 | 154427 | 
| 121 | **29 src/_pytest/logging.py** | 594 | 622| 203 | 57404 | 154427 | 
| 122 | 29 src/_pytest/fixtures.py | 706 | 735| 312 | 57716 | 154427 | 
| 123 | 30 src/_pytest/config/__init__.py | 957 | 973| 170 | 57886 | 165729 | 
| 124 | 30 testing/python/collect.py | 575 | 663| 591 | 58477 | 165729 | 
| 125 | 30 src/_pytest/runner.py | 89 | 102| 123 | 58600 | 165729 | 
| 126 | 30 testing/python/collect.py | 965 | 984| 178 | 58778 | 165729 | 
| 127 | 30 src/_pytest/fixtures.py | 1532 | 1564| 262 | 59040 | 165729 | 
| 128 | 31 testing/python/raises.py | 158 | 186| 236 | 59276 | 167997 | 
| 129 | 31 testing/python/collect.py | 1154 | 1191| 209 | 59485 | 167997 | 
| 130 | 31 src/_pytest/capture.py | 902 | 916| 152 | 59637 | 167997 | 
| 131 | 31 src/_pytest/fixtures.py | 482 | 563| 698 | 60335 | 167997 | 
| 132 | 32 src/_pytest/skipping.py | 261 | 317| 523 | 60858 | 170460 | 
| 133 | 32 testing/python/metafunc.py | 138 | 185| 471 | 61329 | 170460 | 
| 134 | 32 src/_pytest/doctest.py | 61 | 111| 339 | 61668 | 170460 | 
| 135 | 32 testing/python/collect.py | 831 | 853| 200 | 61868 | 170460 | 
| 136 | 32 src/_pytest/capture.py | 91 | 148| 519 | 62387 | 170460 | 
| 137 | 33 src/_pytest/__init__.py | 1 | 9| 0 | 62387 | 170516 | 
| 138 | 33 testing/python/collect.py | 1103 | 1121| 154 | 62541 | 170516 | 
| 139 | 33 src/_pytest/capture.py | 1 | 30| 168 | 62709 | 170516 | 
| 140 | 33 src/_pytest/config/__init__.py | 1131 | 1152| 191 | 62900 | 170516 | 
| 141 | 33 src/_pytest/python.py | 777 | 809| 250 | 63150 | 170516 | 
| 142 | 34 doc/en/conf.py | 1 | 107| 779 | 63929 | 173472 | 
| 143 | 35 src/_pytest/helpconfig.py | 97 | 125| 228 | 64157 | 175323 | 
| 144 | 36 src/_pytest/unittest.py | 166 | 197| 266 | 64423 | 178151 | 
| 145 | 36 testing/python/collect.py | 280 | 347| 474 | 64897 | 178151 | 
| 146 | 36 src/_pytest/pytester.py | 62 | 97| 237 | 65134 | 178151 | 
| 147 | 36 testing/python/collect.py | 1035 | 1062| 217 | 65351 | 178151 | 
| 148 | 36 src/_pytest/runner.py | 105 | 124| 221 | 65572 | 178151 | 
| 149 | 36 testing/python/metafunc.py | 241 | 285| 339 | 65911 | 178151 | 
| 150 | 36 testing/python/metafunc.py | 1389 | 1413| 210 | 66121 | 178151 | 
| 151 | 36 src/_pytest/pytester.py | 1457 | 1470| 126 | 66247 | 178151 | 
| 152 | 37 testing/example_scripts/issue88_initial_file_multinodes/conftest.py | 1 | 15| 0 | 66247 | 178213 | 
| 153 | 37 testing/python/integration.py | 1 | 40| 278 | 66525 | 178213 | 
| 154 | 38 src/_pytest/mark/__init__.py | 270 | 285| 143 | 66668 | 180244 | 
| 155 | 38 testing/python/integration.py | 272 | 288| 121 | 66789 | 180244 | 
| 156 | 38 src/_pytest/pytester.py | 414 | 462| 385 | 67174 | 180244 | 
| 157 | 39 src/_pytest/terminal.py | 508 | 557| 427 | 67601 | 190838 | 
| 158 | 39 src/_pytest/pytester.py | 304 | 334| 244 | 67845 | 190838 | 
| 159 | 39 src/_pytest/python.py | 433 | 476| 390 | 68235 | 190838 | 
| 160 | 39 testing/python/collect.py | 446 | 475| 192 | 68427 | 190838 | 
| 161 | 40 src/_pytest/_io/__init__.py | 1 | 9| 0 | 68427 | 190872 | 
| 162 | 40 src/_pytest/terminal.py | 258 | 271| 117 | 68544 | 190872 | 
| 163 | 40 testing/python/raises.py | 188 | 217| 261 | 68805 | 190872 | 
| 164 | 40 src/_pytest/python.py | 174 | 184| 127 | 68932 | 190872 | 
| 165 | 40 src/_pytest/main.py | 510 | 524| 162 | 69094 | 190872 | 
| 166 | 40 testing/python/metafunc.py | 1465 | 1480| 118 | 69212 | 190872 | 
| 167 | 40 src/_pytest/hookspec.py | 232 | 248| 134 | 69346 | 190872 | 
| 168 | 40 testing/python/integration.py | 291 | 328| 219 | 69565 | 190872 | 
| 169 | 41 src/_pytest/pastebin.py | 52 | 66| 162 | 69727 | 191786 | 
| 170 | 41 doc/en/conf.py | 347 | 373| 236 | 69963 | 191786 | 
| 171 | 41 src/_pytest/capture.py | 213 | 250| 192 | 70155 | 191786 | 
| 172 | 41 src/_pytest/python.py | 141 | 158| 219 | 70374 | 191786 | 
| 173 | 42 src/_pytest/reports.py | 163 | 182| 124 | 70498 | 196072 | 


## Patch

```diff
diff --git a/src/_pytest/logging.py b/src/_pytest/logging.py
--- a/src/_pytest/logging.py
+++ b/src/_pytest/logging.py
@@ -439,7 +439,8 @@ def set_level(self, level: Union[int, str], logger: Optional[str] = None) -> Non
         # Save the original log-level to restore it during teardown.
         self._initial_logger_levels.setdefault(logger, logger_obj.level)
         logger_obj.setLevel(level)
-        self._initial_handler_level = self.handler.level
+        if self._initial_handler_level is None:
+            self._initial_handler_level = self.handler.level
         self.handler.setLevel(level)
 
     @contextmanager

```

## Test Patch

```diff
diff --git a/testing/logging/test_fixture.py b/testing/logging/test_fixture.py
--- a/testing/logging/test_fixture.py
+++ b/testing/logging/test_fixture.py
@@ -65,6 +65,7 @@ def test_change_level_undos_handler_level(testdir: Testdir) -> None:
 
         def test1(caplog):
             assert caplog.handler.level == 0
+            caplog.set_level(9999)
             caplog.set_level(41)
             assert caplog.handler.level == 41
 

```


## Code snippets

### 1 - src/_pytest/logging.py:

Start line: 424, End line: 443

```python
class LogCaptureFixture:

    def clear(self) -> None:
        """Reset the list of log records and the captured log text."""
        self.handler.reset()

    def set_level(self, level: Union[int, str], logger: Optional[str] = None) -> None:
        """Set the level of a logger for the duration of a test.

        .. versionchanged:: 3.4
            The levels of the loggers changed by this function will be
            restored to their initial values at the end of the test.

        :param int level: The level.
        :param str logger: The logger to update. If not given, the root logger.
        """
        logger_obj = logging.getLogger(logger)
        # Save the original log-level to restore it during teardown.
        self._initial_logger_levels.setdefault(logger, logger_obj.level)
        logger_obj.setLevel(level)
        self._initial_handler_level = self.handler.level
        self.handler.setLevel(level)
```
### 2 - src/_pytest/logging.py:

Start line: 445, End line: 465

```python
class LogCaptureFixture:

    @contextmanager
    def at_level(
        self, level: int, logger: Optional[str] = None
    ) -> Generator[None, None, None]:
        """Context manager that sets the level for capturing of logs. After
        the end of the 'with' statement the level is restored to its original
        value.

        :param int level: The level.
        :param str logger: The logger to update. If not given, the root logger.
        """
        logger_obj = logging.getLogger(logger)
        orig_level = logger_obj.level
        logger_obj.setLevel(level)
        handler_orig_level = self.handler.level
        self.handler.setLevel(level)
        try:
            yield
        finally:
            logger_obj.setLevel(orig_level)
            self.handler.setLevel(handler_orig_level)
```
### 3 - src/_pytest/logging.py:

Start line: 342, End line: 369

```python
class LogCaptureFixture:
    """Provides access and control of log capturing."""

    def __init__(self, item: nodes.Node) -> None:
        self._item = item
        self._initial_handler_level = None  # type: Optional[int]
        # Dict of log name -> log level.
        self._initial_logger_levels = {}  # type: Dict[Optional[str], int]

    def _finalize(self) -> None:
        """Finalize the fixture.

        This restores the log levels changed by :meth:`set_level`.
        """
        # Restore log levels.
        if self._initial_handler_level is not None:
            self.handler.setLevel(self._initial_handler_level)
        for logger_name, level in self._initial_logger_levels.items():
            logger = logging.getLogger(logger_name)
            logger.setLevel(level)

    @property
    def handler(self) -> LogCaptureHandler:
        """Get the logging handler used by the fixture.

        :rtype: LogCaptureHandler
        """
        return self._item._store[caplog_handler_key]
```
### 4 - src/_pytest/logging.py:

Start line: 736, End line: 777

```python
class _FileHandler(logging.FileHandler):
    """A logging FileHandler with pytest tweaks."""

    def handleError(self, record: logging.LogRecord) -> None:
        # Handled by LogCaptureHandler.
        pass


class _LiveLoggingStreamHandler(logging.StreamHandler):
    """A logging StreamHandler used by the live logging feature: it will
    write a newline before the first log message in each test.

    During live logging we must also explicitly disable stdout/stderr
    capturing otherwise it will get captured and won't appear in the
    terminal.
    """

    # Officially stream needs to be a IO[str], but TerminalReporter
    # isn't. So force it.
    stream = None  # type: TerminalReporter # type: ignore

    def __init__(
        self,
        terminal_reporter: TerminalReporter,
        capture_manager: Optional[CaptureManager],
    ) -> None:
        logging.StreamHandler.__init__(self, stream=terminal_reporter)  # type: ignore[arg-type]
        self.capture_manager = capture_manager
        self.reset()
        self.set_when(None)
        self._test_outcome_written = False

    def reset(self) -> None:
        """Reset the handler; should be called before the start of each test."""
        self._first_record_emitted = False

    def set_when(self, when: Optional[str]) -> None:
        """Prepare for the given test phase (setup/call/teardown)."""
        self._when = when
        self._section_name_shown = False
        if when == "start":
            self._test_outcome_written = False
```
### 5 - src/_pytest/logging.py:

Start line: 779, End line: 815

```python
class _LiveLoggingStreamHandler(logging.StreamHandler):

    def emit(self, record: logging.LogRecord) -> None:
        ctx_manager = (
            self.capture_manager.global_and_fixture_disabled()
            if self.capture_manager
            else nullcontext()
        )
        with ctx_manager:
            if not self._first_record_emitted:
                self.stream.write("\n")
                self._first_record_emitted = True
            elif self._when in ("teardown", "finish"):
                if not self._test_outcome_written:
                    self._test_outcome_written = True
                    self.stream.write("\n")
            if not self._section_name_shown and self._when:
                self.stream.section("live log " + self._when, sep="-", bold=True)
                self._section_name_shown = True
            super().emit(record)

    def handleError(self, record: logging.LogRecord) -> None:
        # Handled by LogCaptureHandler.
        pass


class _LiveLoggingNullHandler(logging.NullHandler):
    """A logging handler used when live logging is disabled."""

    def reset(self) -> None:
        pass

    def set_when(self, when: str) -> None:
        pass

    def handleError(self, record: logging.LogRecord) -> None:
        # Handled by LogCaptureHandler.
        pass
```
### 6 - src/_pytest/logging.py:

Start line: 485, End line: 511

```python
def get_log_level_for_setting(config: Config, *setting_names: str) -> Optional[int]:
    for setting_name in setting_names:
        log_level = config.getoption(setting_name)
        if log_level is None:
            log_level = config.getini(setting_name)
        if log_level:
            break
    else:
        return None

    if isinstance(log_level, str):
        log_level = log_level.upper()
    try:
        return int(getattr(logging, log_level, log_level))
    except ValueError as e:
        # Python logging does not recognise this as a logging level
        raise pytest.UsageError(
            "'{}' is not recognized as a logging level name for "
            "'{}'. Please consider passing the "
            "logging level num instead.".format(log_level, setting_name)
        ) from e


# run after terminalreporter/capturemanager are configured
@pytest.hookimpl(trylast=True)
def pytest_configure(config: Config) -> None:
    config.pluginmanager.register(LoggingPlugin(config), "logging-plugin")
```
### 7 - src/_pytest/logging.py:

Start line: 468, End line: 482

```python
@pytest.fixture
def caplog(request: FixtureRequest) -> Generator[LogCaptureFixture, None, None]:
    """Access and control log capturing.

    Captured logs are available through the following properties/methods::

    * caplog.messages        -> list of format-interpolated log messages
    * caplog.text            -> string containing formatted log output
    * caplog.records         -> list of logging.LogRecord instances
    * caplog.record_tuples   -> list of (logger_name, level, message) tuples
    * caplog.clear()         -> clear captured records and formatted log output string
    """
    result = LogCaptureFixture(request.node)
    yield result
    result._finalize()
```
### 8 - src/_pytest/logging.py:

Start line: 284, End line: 311

```python
_HandlerType = TypeVar("_HandlerType", bound=logging.Handler)


# Not using @contextmanager for performance reasons.
class catching_logs:
    """Context manager that prepares the whole logging machinery properly."""

    __slots__ = ("handler", "level", "orig_level")

    def __init__(self, handler: _HandlerType, level: Optional[int] = None) -> None:
        self.handler = handler
        self.level = level

    def __enter__(self):
        root_logger = logging.getLogger()
        if self.level is not None:
            self.handler.setLevel(self.level)
        root_logger.addHandler(self.handler)
        if self.level is not None:
            self.orig_level = root_logger.level
            root_logger.setLevel(min(self.orig_level, self.level))
        return self.handler

    def __exit__(self, type, value, traceback):
        root_logger = logging.getLogger()
        if self.level is not None:
            root_logger.setLevel(self.orig_level)
        root_logger.removeHandler(self.handler)
```
### 9 - src/_pytest/faulthandler.py:

Start line: 72, End line: 84

```python
class FaultHandlerHooks:

    @staticmethod
    def _get_stderr_fileno():
        try:
            return sys.stderr.fileno()
        except (AttributeError, io.UnsupportedOperation):
            # pytest-xdist monkeypatches sys.stderr with an object that is not an actual file.
            # https://docs.python.org/3/library/faulthandler.html#issue-with-file-descriptors
            # This is potentially dangerous, but the best we can do.
            return sys.__stderr__.fileno()

    @staticmethod
    def get_timeout_config_value(config):
        return float(config.getini("faulthandler_timeout") or 0.0)
```
### 10 - src/_pytest/logging.py:

Start line: 695, End line: 733

```python
class LoggingPlugin:

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_setup(self, item: nodes.Item) -> Generator[None, None, None]:
        self.log_cli_handler.set_when("setup")

        empty = {}  # type: Dict[str, List[logging.LogRecord]]
        item._store[caplog_records_key] = empty
        yield from self._runtest_for(item, "setup")

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_call(self, item: nodes.Item) -> Generator[None, None, None]:
        self.log_cli_handler.set_when("call")

        yield from self._runtest_for(item, "call")

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_teardown(self, item: nodes.Item) -> Generator[None, None, None]:
        self.log_cli_handler.set_when("teardown")

        yield from self._runtest_for(item, "teardown")
        del item._store[caplog_records_key]
        del item._store[caplog_handler_key]

    @pytest.hookimpl
    def pytest_runtest_logfinish(self) -> None:
        self.log_cli_handler.set_when("finish")

    @pytest.hookimpl(hookwrapper=True, tryfirst=True)
    def pytest_sessionfinish(self) -> Generator[None, None, None]:
        self.log_cli_handler.set_when("sessionfinish")

        with catching_logs(self.log_cli_handler, level=self.log_cli_level):
            with catching_logs(self.log_file_handler, level=self.log_file_level):
                yield

    @pytest.hookimpl
    def pytest_unconfigure(self) -> None:
        # Close the FileHandler explicitly.
        # (logging.shutdown might have lost the weakref?!)
        self.log_file_handler.close()
```
### 12 - src/_pytest/logging.py:

Start line: 314, End line: 339

```python
class LogCaptureHandler(logging.StreamHandler):
    """A logging handler that stores log records and the log text."""

    stream = None  # type: StringIO

    def __init__(self) -> None:
        """Create a new log handler."""
        super().__init__(StringIO())
        self.records = []  # type: List[logging.LogRecord]

    def emit(self, record: logging.LogRecord) -> None:
        """Keep the log records in a list in addition to the log text."""
        self.records.append(record)
        super().emit(record)

    def reset(self) -> None:
        self.records = []
        self.stream = StringIO()

    def handleError(self, record: logging.LogRecord) -> None:
        if logging.raiseExceptions:
            # Fail the test if the log message is bad (emit failed).
            # The default behavior of logging is to print "Logging error"
            # to stderr with the call stack and some extra details.
            # pytest wants to make such mistakes visible during testing.
            raise
```
### 14 - src/_pytest/logging.py:

Start line: 1, End line: 42

```python
"""Access and control log capturing."""
import logging
import os
import re
import sys
from contextlib import contextmanager
from io import StringIO
from typing import AbstractSet
from typing import Dict
from typing import Generator
from typing import List
from typing import Mapping
from typing import Optional
from typing import Tuple
from typing import TypeVar
from typing import Union

import pytest
from _pytest import nodes
from _pytest._io import TerminalWriter
from _pytest.capture import CaptureManager
from _pytest.compat import nullcontext
from _pytest.config import _strtobool
from _pytest.config import Config
from _pytest.config import create_terminal_writer
from _pytest.config.argparsing import Parser
from _pytest.fixtures import FixtureRequest
from _pytest.main import Session
from _pytest.pathlib import Path
from _pytest.store import StoreKey
from _pytest.terminal import TerminalReporter


DEFAULT_LOG_FORMAT = "%(levelname)-8s %(name)s:%(filename)s:%(lineno)d %(message)s"
DEFAULT_LOG_DATE_FORMAT = "%H:%M:%S"
_ANSI_ESCAPE_SEQ = re.compile(r"\x1b\[[\d;]+m")
caplog_handler_key = StoreKey["LogCaptureHandler"]()
caplog_records_key = StoreKey[Dict[str, List[logging.LogRecord]]]()


def _remove_ansi_escape_sequences(text: str) -> str:
    return _ANSI_ESCAPE_SEQ.sub("", text)
```
### 15 - src/_pytest/logging.py:

Start line: 655, End line: 676

```python
class LoggingPlugin:

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtestloop(self, session: Session) -> Generator[None, None, None]:
        if session.config.option.collectonly:
            yield
            return

        if self._log_cli_enabled() and self._config.getoption("verbose") < 1:
            # The verbose flag is needed to avoid messy test progress output.
            self._config.option.verbose = 1

        with catching_logs(self.log_cli_handler, level=self.log_cli_level):
            with catching_logs(self.log_file_handler, level=self.log_file_level):
                yield  # Run all the tests.

    @pytest.hookimpl
    def pytest_runtest_logstart(self) -> None:
        self.log_cli_handler.reset()
        self.log_cli_handler.set_when("start")

    @pytest.hookimpl
    def pytest_runtest_logreport(self) -> None:
        self.log_cli_handler.set_when("logreport")
```
### 18 - src/_pytest/logging.py:

Start line: 45, End line: 88

```python
class ColoredLevelFormatter(logging.Formatter):
    """A logging formatter which colorizes the %(levelname)..s part of the
    log format passed to __init__."""

    LOGLEVEL_COLOROPTS = {
        logging.CRITICAL: {"red"},
        logging.ERROR: {"red", "bold"},
        logging.WARNING: {"yellow"},
        logging.WARN: {"yellow"},
        logging.INFO: {"green"},
        logging.DEBUG: {"purple"},
        logging.NOTSET: set(),
    }  # type: Mapping[int, AbstractSet[str]]
    LEVELNAME_FMT_REGEX = re.compile(r"%\(levelname\)([+-.]?\d*s)")

    def __init__(self, terminalwriter: TerminalWriter, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._original_fmt = self._style._fmt
        self._level_to_fmt_mapping = {}  # type: Dict[int, str]

        assert self._fmt is not None
        levelname_fmt_match = self.LEVELNAME_FMT_REGEX.search(self._fmt)
        if not levelname_fmt_match:
            return
        levelname_fmt = levelname_fmt_match.group()

        for level, color_opts in self.LOGLEVEL_COLOROPTS.items():
            formatted_levelname = levelname_fmt % {
                "levelname": logging.getLevelName(level)
            }

            # add ANSI escape sequences around the formatted levelname
            color_kwargs = {name: True for name in color_opts}
            colorized_formatted_levelname = terminalwriter.markup(
                formatted_levelname, **color_kwargs
            )
            self._level_to_fmt_mapping[level] = self.LEVELNAME_FMT_REGEX.sub(
                colorized_formatted_levelname, self._fmt
            )

    def format(self, record: logging.LogRecord) -> str:
        fmt = self._level_to_fmt_mapping.get(record.levelno, self._original_fmt)
        self._style._fmt = fmt
        return super().format(record)
```
### 28 - src/_pytest/logging.py:

Start line: 198, End line: 281

```python
def pytest_addoption(parser: Parser) -> None:
    """Add options to control log capturing."""
    group = parser.getgroup("logging")

    def add_option_ini(option, dest, default=None, type=None, **kwargs):
        parser.addini(
            dest, default=default, type=type, help="default value for " + option
        )
        group.addoption(option, dest=dest, **kwargs)

    add_option_ini(
        "--log-level",
        dest="log_level",
        default=None,
        metavar="LEVEL",
        help=(
            "level of messages to catch/display.\n"
            "Not set by default, so it depends on the root/parent log handler's"
            ' effective level, where it is "WARNING" by default.'
        ),
    )
    add_option_ini(
        "--log-format",
        dest="log_format",
        default=DEFAULT_LOG_FORMAT,
        help="log format as used by the logging module.",
    )
    add_option_ini(
        "--log-date-format",
        dest="log_date_format",
        default=DEFAULT_LOG_DATE_FORMAT,
        help="log date format as used by the logging module.",
    )
    parser.addini(
        "log_cli",
        default=False,
        type="bool",
        help='enable log display during test run (also known as "live logging").',
    )
    add_option_ini(
        "--log-cli-level", dest="log_cli_level", default=None, help="cli logging level."
    )
    add_option_ini(
        "--log-cli-format",
        dest="log_cli_format",
        default=None,
        help="log format as used by the logging module.",
    )
    add_option_ini(
        "--log-cli-date-format",
        dest="log_cli_date_format",
        default=None,
        help="log date format as used by the logging module.",
    )
    add_option_ini(
        "--log-file",
        dest="log_file",
        default=None,
        help="path to a file when logging will be written to.",
    )
    add_option_ini(
        "--log-file-level",
        dest="log_file_level",
        default=None,
        help="log file logging level.",
    )
    add_option_ini(
        "--log-file-format",
        dest="log_file_format",
        default=DEFAULT_LOG_FORMAT,
        help="log format as used by the logging module.",
    )
    add_option_ini(
        "--log-file-date-format",
        dest="log_file_date_format",
        default=DEFAULT_LOG_DATE_FORMAT,
        help="log date format as used by the logging module.",
    )
    add_option_ini(
        "--log-auto-indent",
        dest="log_auto_indent",
        default=None,
        help="Auto-indent multiline messages passed to the logging module. Accepts true|on, false|off or an integer.",
    )
```
### 41 - src/_pytest/logging.py:

Start line: 384, End line: 403

```python
class LogCaptureFixture:

    @property
    def text(self) -> str:
        """The formatted log text."""
        return _remove_ansi_escape_sequences(self.handler.stream.getvalue())

    @property
    def records(self) -> List[logging.LogRecord]:
        """The list of log records."""
        return self.handler.records

    @property
    def record_tuples(self) -> List[Tuple[str, int, str]]:
        """A list of a stripped down version of log records intended
        for use in assertion comparison.

        The format of the tuple is:

            (logger_name, log_level, message)
        """
        return [(r.name, r.levelno, r.getMessage()) for r in self.records]
```
### 67 - src/_pytest/logging.py:

Start line: 624, End line: 653

```python
class LoggingPlugin:

    def _log_cli_enabled(self):
        """Return whether live logging is enabled."""
        enabled = self._config.getoption(
            "--log-cli-level"
        ) is not None or self._config.getini("log_cli")
        if not enabled:
            return False

        terminal_reporter = self._config.pluginmanager.get_plugin("terminalreporter")
        if terminal_reporter is None:
            # terminal reporter is disabled e.g. by pytest-xdist.
            return False

        return True

    @pytest.hookimpl(hookwrapper=True, tryfirst=True)
    def pytest_sessionstart(self) -> Generator[None, None, None]:
        self.log_cli_handler.set_when("sessionstart")

        with catching_logs(self.log_cli_handler, level=self.log_cli_level):
            with catching_logs(self.log_file_handler, level=self.log_file_level):
                yield

    @pytest.hookimpl(hookwrapper=True, tryfirst=True)
    def pytest_collection(self) -> Generator[None, None, None]:
        self.log_cli_handler.set_when("collection")

        with catching_logs(self.log_cli_handler, level=self.log_cli_level):
            with catching_logs(self.log_file_handler, level=self.log_file_level):
                yield
```
### 78 - src/_pytest/logging.py:

Start line: 678, End line: 693

```python
class LoggingPlugin:

    def _runtest_for(self, item: nodes.Item, when: str) -> Generator[None, None, None]:
        """Implement the internals of the pytest_runtest_xxx() hooks."""
        with catching_logs(
            self.caplog_handler, level=self.log_level,
        ) as caplog_handler, catching_logs(
            self.report_handler, level=self.log_level,
        ) as report_handler:
            caplog_handler.reset()
            report_handler.reset()
            item._store[caplog_records_key][when] = caplog_handler.records
            item._store[caplog_handler_key] = caplog_handler

            yield

            log = report_handler.stream.getvalue().strip()
            item.add_report_section(when, "log", log)
```
### 80 - src/_pytest/logging.py:

Start line: 576, End line: 592

```python
class LoggingPlugin:

    def _create_formatter(self, log_format, log_date_format, auto_indent):
        # Color option doesn't exist if terminal plugin is disabled.
        color = getattr(self._config.option, "color", "no")
        if color != "no" and ColoredLevelFormatter.LEVELNAME_FMT_REGEX.search(
            log_format
        ):
            formatter = ColoredLevelFormatter(
                create_terminal_writer(self._config), log_format, log_date_format
            )  # type: logging.Formatter
        else:
            formatter = logging.Formatter(log_format, log_date_format)

        formatter._style = PercentStyleMultiline(
            formatter._style._fmt, auto_indent=auto_indent
        )

        return formatter
```
### 96 - src/_pytest/logging.py:

Start line: 371, End line: 382

```python
class LogCaptureFixture:

    def get_records(self, when: str) -> List[logging.LogRecord]:
        """Get the logging records for one of the possible test phases.

        :param str when:
            Which test phase to obtain the records from. Valid values are: "setup", "call" and "teardown".

        :returns: The list of captured records at the given stage.
        :rtype: List[logging.LogRecord]

        .. versionadded:: 3.4
        """
        return self._item._store[caplog_records_key].get(when, [])
```
### 121 - src/_pytest/logging.py:

Start line: 594, End line: 622

```python
class LoggingPlugin:

    def set_log_path(self, fname: str) -> None:
        """Set the filename parameter for Logging.FileHandler().

        Creates parent directory if it does not exist.

        .. warning::
            This is an experimental API.
        """
        fpath = Path(fname)

        if not fpath.is_absolute():
            fpath = Path(str(self._config.rootdir), fpath)

        if not fpath.parent.exists():
            fpath.parent.mkdir(exist_ok=True, parents=True)

        stream = fpath.open(mode="w", encoding="UTF-8")
        if sys.version_info >= (3, 7):
            old_stream = self.log_file_handler.setStream(stream)
        else:
            old_stream = self.log_file_handler.stream
            self.log_file_handler.acquire()
            try:
                self.log_file_handler.flush()
                self.log_file_handler.stream = stream
            finally:
                self.log_file_handler.release()
        if old_stream:
            old_stream.close()
```
