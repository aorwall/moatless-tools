# pytest-dev__pytest-5808

| **pytest-dev/pytest** | `404cf0c872880f2aac8214bb490b26c9a659548e` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 479 |
| **Any found context length** | 479 |
| **Avg pos** | 2.0 |
| **Min pos** | 2 |
| **Max pos** | 2 |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/src/_pytest/pastebin.py b/src/_pytest/pastebin.py
--- a/src/_pytest/pastebin.py
+++ b/src/_pytest/pastebin.py
@@ -65,7 +65,7 @@ def create_new_paste(contents):
     from urllib.request import urlopen
     from urllib.parse import urlencode
 
-    params = {"code": contents, "lexer": "python3", "expiry": "1week"}
+    params = {"code": contents, "lexer": "text", "expiry": "1week"}
     url = "https://bpaste.net"
     try:
         response = (

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| src/_pytest/pastebin.py | 68 | 68 | 2 | 1 | 479


## Problem Statement

```
Lexer "python3" in --pastebin feature causes HTTP errors
The `--pastebin` option currently submits the output of `pytest` to `bpaste.net` using `lexer=python3`: https://github.com/pytest-dev/pytest/blob/d47b9d04d4cf824150caef46c9c888779c1b3f58/src/_pytest/pastebin.py#L68-L73

For some `contents`, this will raise a "HTTP Error 400: Bad Request".

As an example:
~~~
>>> from urllib.request import urlopen
>>> with open("data.txt", "rb") as in_fh:
...     data = in_fh.read()
>>> url = "https://bpaste.net"
>>> urlopen(url, data=data)
HTTPError: Bad Request
~~~
with the attached [data.txt](https://github.com/pytest-dev/pytest/files/3561212/data.txt).

This is the underlying cause for the problems mentioned in #5764.

The call goes through fine if `lexer` is changed from `python3` to `text`. This would seem like the right thing to do in any case: the console output of a `pytest` run that is being uploaded is not Python code, but arbitrary text.


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 src/_pytest/pastebin.py** | 1 | 38| 268 | 268 | 806 | 
| **-> 2 <-** | **1 src/_pytest/pastebin.py** | 57 | 80| 211 | 479 | 806 | 
| 3 | **1 src/_pytest/pastebin.py** | 83 | 104| 181 | 660 | 806 | 
| 4 | **1 src/_pytest/pastebin.py** | 41 | 54| 145 | 805 | 806 | 
| 5 | 2 src/_pytest/faulthandler.py | 1 | 36| 249 | 1054 | 1381 | 
| 6 | 3 src/pytest.py | 1 | 107| 708 | 1762 | 2089 | 
| 7 | 4 src/_pytest/skipping.py | 28 | 65| 363 | 2125 | 3567 | 
| 8 | 5 src/_pytest/pytester.py | 32 | 54| 144 | 2269 | 13894 | 
| 9 | 6 src/_pytest/python.py | 56 | 108| 354 | 2623 | 25295 | 
| 10 | 7 src/_pytest/_io/saferepr.py | 1 | 21| 159 | 2782 | 25787 | 
| 11 | 8 doc/en/example/py2py3/conftest.py | 1 | 17| 0 | 2782 | 25873 | 
| 12 | 9 doc/en/conf.py | 241 | 341| 656 | 3438 | 28332 | 
| 13 | 10 src/_pytest/debugging.py | 23 | 44| 155 | 3593 | 30742 | 
| 14 | 10 src/_pytest/pytester.py | 1 | 29| 170 | 3763 | 30742 | 
| 15 | 11 src/_pytest/config/argparsing.py | 371 | 405| 359 | 4122 | 34678 | 
| 16 | 12 src/_pytest/python_api.py | 720 | 737| 149 | 4271 | 41248 | 
| 17 | 13 src/_pytest/config/exceptions.py | 1 | 10| 0 | 4271 | 41293 | 
| 18 | 14 bench/skip.py | 1 | 10| 0 | 4271 | 41328 | 
| 19 | 15 src/_pytest/_code/code.py | 1063 | 1073| 138 | 4409 | 49545 | 
| 20 | 16 testing/python/collect.py | 1205 | 1230| 186 | 4595 | 58822 | 
| 21 | 17 src/_pytest/main.py | 42 | 150| 757 | 5352 | 64109 | 
| 22 | 18 src/_pytest/capture.py | 413 | 446| 210 | 5562 | 70130 | 
| 23 | 19 src/_pytest/doctest.py | 37 | 84| 321 | 5883 | 74630 | 
| 24 | 20 src/_pytest/terminal.py | 50 | 137| 630 | 6513 | 82862 | 
| 25 | 21 src/_pytest/cacheprovider.py | 292 | 347| 410 | 6923 | 86234 | 
| 26 | 22 doc/en/example/assertion/failure_demo.py | 191 | 203| 114 | 7037 | 87893 | 
| 27 | 22 src/_pytest/python_api.py | 687 | 717| 318 | 7355 | 87893 | 
| 28 | 23 doc/en/example/conftest.py | 1 | 2| 0 | 7355 | 87900 | 
| 29 | 23 src/_pytest/main.py | 151 | 177| 175 | 7530 | 87900 | 
| 30 | 24 src/_pytest/reports.py | 265 | 274| 134 | 7664 | 90900 | 
| 31 | 25 src/_pytest/warnings.py | 32 | 54| 157 | 7821 | 91979 | 
| 32 | 26 doc/en/conftest.py | 1 | 2| 0 | 7821 | 91986 | 
| 33 | 27 bench/empty.py | 1 | 3| 0 | 7821 | 92008 | 
| 34 | 28 src/_pytest/config/__init__.py | 228 | 238| 154 | 7975 | 100773 | 
| 35 | 29 src/_pytest/junitxml.py | 383 | 427| 334 | 8309 | 105828 | 
| 36 | 30 testing/example_scripts/conftest_usageerror/conftest.py | 1 | 9| 0 | 8309 | 105859 | 
| 37 | 30 src/_pytest/python.py | 111 | 129| 176 | 8485 | 105859 | 
| 38 | 31 doc/en/example/xfail_demo.py | 1 | 39| 143 | 8628 | 106003 | 
| 39 | 31 src/_pytest/capture.py | 770 | 822| 450 | 9078 | 106003 | 
| 40 | 31 src/_pytest/pytester.py | 57 | 82| 201 | 9279 | 106003 | 
| 41 | 32 src/_pytest/setupplan.py | 1 | 28| 163 | 9442 | 106167 | 
| 42 | 33 bench/manyparam.py | 1 | 15| 0 | 9442 | 106208 | 
| 43 | 34 src/_pytest/nodes.py | 1 | 26| 148 | 9590 | 109514 | 
| 44 | 35 src/_pytest/outcomes.py | 39 | 54| 124 | 9714 | 111028 | 
| 45 | 36 src/_pytest/compat.py | 147 | 185| 244 | 9958 | 113496 | 
| 46 | 37 src/_pytest/resultlog.py | 1 | 35| 235 | 10193 | 114217 | 
| 47 | 37 src/_pytest/python.py | 152 | 173| 241 | 10434 | 114217 | 
| 48 | 37 src/_pytest/reports.py | 1 | 29| 227 | 10661 | 114217 | 
| 49 | 37 src/_pytest/config/argparsing.py | 332 | 356| 201 | 10862 | 114217 | 
| 50 | 38 src/_pytest/helpconfig.py | 39 | 83| 297 | 11159 | 115906 | 
| 51 | 38 src/_pytest/pytester.py | 115 | 136| 243 | 11402 | 115906 | 
| 52 | 39 src/_pytest/assertion/__init__.py | 1 | 33| 196 | 11598 | 117136 | 
| 53 | 39 src/_pytest/doctest.py | 313 | 339| 198 | 11796 | 117136 | 
| 54 | 40 extra/get_issues.py | 55 | 86| 231 | 12027 | 117679 | 
| 55 | 40 src/_pytest/doctest.py | 509 | 534| 237 | 12264 | 117679 | 
| 56 | 40 src/_pytest/_code/code.py | 1 | 31| 159 | 12423 | 117679 | 
| 57 | 41 src/_pytest/mark/__init__.py | 36 | 74| 341 | 12764 | 118850 | 
| 58 | 41 src/_pytest/config/argparsing.py | 88 | 106| 183 | 12947 | 118850 | 
| 59 | 41 src/_pytest/helpconfig.py | 86 | 114| 219 | 13166 | 118850 | 
| 60 | 42 doc/en/example/costlysetup/conftest.py | 1 | 21| 0 | 13166 | 118929 | 
| 61 | 42 src/_pytest/python_api.py | 1 | 41| 229 | 13395 | 118929 | 
| 62 | 42 src/_pytest/cacheprovider.py | 201 | 255| 493 | 13888 | 118929 | 
| 63 | 42 src/_pytest/pytester.py | 84 | 113| 193 | 14081 | 118929 | 
| 64 | 43 src/_pytest/assertion/rewrite.py | 1 | 36| 229 | 14310 | 127960 | 
| 65 | 43 src/_pytest/compat.py | 1 | 46| 225 | 14535 | 127960 | 
| 66 | 43 src/_pytest/python.py | 1204 | 1230| 208 | 14743 | 127960 | 
| 67 | 44 src/_pytest/runner.py | 160 | 191| 238 | 14981 | 130808 | 
| 68 | 44 src/_pytest/mark/__init__.py | 77 | 95| 141 | 15122 | 130808 | 
| 69 | 44 testing/python/collect.py | 1146 | 1172| 182 | 15304 | 130808 | 
| 70 | 44 extra/get_issues.py | 1 | 30| 168 | 15472 | 130808 | 
| 71 | 45 setup.py | 1 | 16| 147 | 15619 | 131070 | 
| 72 | 45 doc/en/conf.py | 118 | 240| 811 | 16430 | 131070 | 
| 73 | 46 src/_pytest/__init__.py | 1 | 9| 0 | 16430 | 131126 | 
| 74 | 46 src/_pytest/terminal.py | 386 | 435| 420 | 16850 | 131126 | 
| 75 | 46 src/_pytest/junitxml.py | 446 | 465| 143 | 16993 | 131126 | 
| 76 | 46 src/_pytest/outcomes.py | 139 | 154| 130 | 17123 | 131126 | 
| 77 | 47 testing/python/integration.py | 1 | 35| 239 | 17362 | 134051 | 
| 78 | 47 src/_pytest/terminal.py | 686 | 698| 119 | 17481 | 134051 | 
| 79 | 47 src/_pytest/python.py | 1262 | 1331| 487 | 17968 | 134051 | 
| 80 | 47 setup.py | 19 | 40| 115 | 18083 | 134051 | 
| 81 | 48 testing/python/metafunc.py | 1295 | 1333| 243 | 18326 | 147152 | 
| 82 | 48 src/_pytest/skipping.py | 1 | 25| 162 | 18488 | 147152 | 
| 83 | 48 doc/en/example/assertion/failure_demo.py | 43 | 121| 680 | 19168 | 147152 | 
| 84 | 49 src/_pytest/_code/__init__.py | 1 | 11| 0 | 19168 | 147253 | 
| 85 | 49 src/_pytest/debugging.py | 1 | 20| 122 | 19290 | 147253 | 
| 86 | 49 src/_pytest/junitxml.py | 214 | 231| 161 | 19451 | 147253 | 
| 87 | 49 doc/en/example/assertion/failure_demo.py | 1 | 40| 169 | 19620 | 147253 | 
| 88 | 50 doc/en/example/costlysetup/sub_b/__init__.py | 1 | 2| 0 | 19620 | 147254 | 
| 89 | 50 src/_pytest/config/__init__.py | 755 | 767| 139 | 19759 | 147254 | 
| 90 | 50 src/_pytest/faulthandler.py | 39 | 51| 121 | 19880 | 147254 | 
| 91 | 50 src/_pytest/junitxml.py | 233 | 247| 132 | 20012 | 147254 | 
| 92 | 50 src/_pytest/junitxml.py | 161 | 212| 333 | 20345 | 147254 | 
| 93 | 50 src/_pytest/python.py | 132 | 149| 212 | 20557 | 147254 | 
| 94 | 50 src/_pytest/capture.py | 1 | 35| 189 | 20746 | 147254 | 
| 95 | 50 src/_pytest/terminal.py | 140 | 167| 216 | 20962 | 147254 | 
| 96 | 50 testing/python/collect.py | 920 | 939| 178 | 21140 | 147254 | 
| 97 | 50 doc/en/conf.py | 1 | 116| 789 | 21929 | 147254 | 
| 98 | 50 src/_pytest/terminal.py | 834 | 847| 125 | 22054 | 147254 | 
| 99 | 50 src/_pytest/debugging.py | 286 | 312| 219 | 22273 | 147254 | 
| 100 | 50 src/_pytest/skipping.py | 120 | 178| 533 | 22806 | 147254 | 
| 101 | 50 src/_pytest/faulthandler.py | 54 | 87| 205 | 23011 | 147254 | 
| 102 | 51 src/_pytest/fixtures.py | 682 | 733| 470 | 23481 | 158236 | 
| 103 | 52 src/_pytest/stepwise.py | 1 | 23| 131 | 23612 | 158950 | 
| 104 | 52 testing/python/metafunc.py | 52 | 73| 211 | 23823 | 158950 | 
| 105 | 52 src/_pytest/_code/code.py | 613 | 656| 307 | 24130 | 158950 | 
| 106 | 52 src/_pytest/junitxml.py | 430 | 443| 119 | 24249 | 158950 | 
| 107 | 52 src/_pytest/reports.py | 188 | 203| 156 | 24405 | 158950 | 
| 108 | 52 src/_pytest/terminal.py | 811 | 832| 180 | 24585 | 158950 | 
| 109 | 52 src/_pytest/helpconfig.py | 1 | 36| 242 | 24827 | 158950 | 
| 110 | 52 src/_pytest/terminal.py | 722 | 745| 142 | 24969 | 158950 | 
| 111 | 52 src/_pytest/config/argparsing.py | 253 | 279| 223 | 25192 | 158950 | 
| 112 | 53 testing/freeze/tox_run.py | 1 | 13| 0 | 25192 | 159035 | 
| 113 | 53 src/_pytest/helpconfig.py | 209 | 245| 252 | 25444 | 159035 | 
| 114 | 53 doc/en/example/assertion/failure_demo.py | 206 | 253| 228 | 25672 | 159035 | 
| 115 | 53 src/_pytest/pytester.py | 310 | 346| 210 | 25882 | 159035 | 
| 116 | 53 testing/python/integration.py | 71 | 86| 109 | 25991 | 159035 | 
| 117 | 53 src/_pytest/terminal.py | 873 | 903| 250 | 26241 | 159035 | 
| 118 | 53 src/_pytest/helpconfig.py | 139 | 206| 550 | 26791 | 159035 | 
| 119 | 53 src/_pytest/junitxml.py | 249 | 277| 217 | 27008 | 159035 | 
| 120 | 53 src/_pytest/outcomes.py | 115 | 136| 174 | 27182 | 159035 | 
| 121 | 53 src/_pytest/doctest.py | 477 | 507| 291 | 27473 | 159035 | 
| 122 | 53 doc/en/example/assertion/failure_demo.py | 164 | 188| 159 | 27632 | 159035 | 
| 123 | 53 testing/python/collect.py | 559 | 658| 662 | 28294 | 159035 | 
| 124 | 54 src/_pytest/_code/source.py | 231 | 259| 162 | 28456 | 161417 | 
| 125 | 54 src/_pytest/config/argparsing.py | 281 | 292| 135 | 28591 | 161417 | 
| 126 | 55 src/_pytest/logging.py | 104 | 185| 534 | 29125 | 166388 | 
| 127 | 55 src/_pytest/fixtures.py | 763 | 792| 242 | 29367 | 166388 | 
| 128 | 55 src/_pytest/config/__init__.py | 100 | 163| 351 | 29718 | 166388 | 
| 129 | 55 src/_pytest/config/argparsing.py | 236 | 251| 127 | 29845 | 166388 | 
| 130 | 55 testing/python/metafunc.py | 642 | 670| 230 | 30075 | 166388 | 
| 131 | 55 src/_pytest/outcomes.py | 57 | 89| 198 | 30273 | 166388 | 
| 132 | 55 src/_pytest/doctest.py | 87 | 128| 311 | 30584 | 166388 | 
| 133 | 55 src/_pytest/reports.py | 127 | 153| 158 | 30742 | 166388 | 
| 134 | 56 testing/example_scripts/issue_519.py | 1 | 31| 350 | 31092 | 166854 | 
| 135 | 56 src/_pytest/cacheprovider.py | 145 | 199| 500 | 31592 | 166854 | 
| 136 | 57 doc/en/example/costlysetup/sub_a/__init__.py | 1 | 2| 0 | 31592 | 166855 | 
| 137 | 57 src/_pytest/debugging.py | 274 | 283| 132 | 31724 | 166855 | 
| 138 | 57 src/_pytest/config/__init__.py | 1069 | 1093| 149 | 31873 | 166855 | 
| 139 | 57 src/_pytest/_code/code.py | 919 | 960| 294 | 32167 | 166855 | 
| 140 | 57 src/_pytest/python_api.py | 739 | 759| 176 | 32343 | 166855 | 
| 141 | 57 src/_pytest/terminal.py | 785 | 809| 203 | 32546 | 166855 | 
| 142 | 57 src/_pytest/debugging.py | 252 | 271| 156 | 32702 | 166855 | 
| 143 | 58 doc/en/example/pythoncollection.py | 1 | 15| 0 | 32702 | 166902 | 
| 144 | 58 src/_pytest/runner.py | 121 | 135| 136 | 32838 | 166902 | 
| 145 | 58 testing/python/metafunc.py | 121 | 176| 398 | 33236 | 166902 | 
| 146 | 58 src/_pytest/junitxml.py | 1 | 81| 624 | 33860 | 166902 | 
| 147 | 58 src/_pytest/fixtures.py | 736 | 760| 214 | 34074 | 166902 | 
| 148 | 58 src/_pytest/main.py | 309 | 360| 303 | 34377 | 166902 | 
| 149 | 58 src/_pytest/junitxml.py | 280 | 292| 132 | 34509 | 166902 | 
| 150 | 58 src/_pytest/assertion/rewrite.py | 237 | 252| 133 | 34642 | 166902 | 
| 151 | 59 testing/python/raises.py | 187 | 202| 135 | 34777 | 168624 | 
| 152 | 59 src/_pytest/resultlog.py | 62 | 79| 144 | 34921 | 168624 | 
| 153 | 59 src/_pytest/config/__init__.py | 867 | 913| 414 | 35335 | 168624 | 
| 154 | 59 src/_pytest/_code/code.py | 864 | 887| 188 | 35523 | 168624 | 
| 155 | 59 src/_pytest/capture.py | 254 | 264| 107 | 35630 | 168624 | 
| 156 | 59 src/_pytest/terminal.py | 336 | 349| 123 | 35753 | 168624 | 
| 157 | 59 testing/python/collect.py | 59 | 77| 189 | 35942 | 168624 | 
| 158 | 59 src/_pytest/resultlog.py | 81 | 98| 159 | 36101 | 168624 | 
| 159 | 59 src/_pytest/terminal.py | 922 | 947| 199 | 36300 | 168624 | 
| 160 | 59 src/_pytest/debugging.py | 315 | 330| 117 | 36417 | 168624 | 
| 161 | 59 testing/python/collect.py | 968 | 987| 170 | 36587 | 168624 | 
| 162 | 60 src/_pytest/setuponly.py | 1 | 46| 297 | 36884 | 169188 | 
| 163 | 60 src/_pytest/pytester.py | 1211 | 1235| 165 | 37049 | 169188 | 
| 164 | 60 src/_pytest/logging.py | 675 | 693| 168 | 37217 | 169188 | 
| 165 | 60 testing/python/metafunc.py | 1144 | 1173| 199 | 37416 | 169188 | 
| 166 | 60 src/_pytest/config/__init__.py | 1 | 56| 344 | 37760 | 169188 | 
| 167 | 61 testing/python/fixtures.py | 2073 | 2190| 714 | 38474 | 194363 | 
| 168 | 62 extra/setup-py.test/setup.py | 1 | 12| 0 | 38474 | 194441 | 
| 169 | 63 scripts/publish_gh_release_notes.py | 61 | 96| 266 | 38740 | 195141 | 
| 170 | 63 src/_pytest/nodes.py | 258 | 305| 384 | 39124 | 195141 | 
| 171 | 63 src/_pytest/config/__init__.py | 731 | 753| 172 | 39296 | 195141 | 
| 172 | 63 testing/python/collect.py | 1106 | 1143| 209 | 39505 | 195141 | 
| 173 | 64 src/_pytest/assertion/util.py | 20 | 36| 147 | 39652 | 198383 | 
| 174 | 64 src/_pytest/terminal.py | 295 | 316| 165 | 39817 | 198383 | 
| 175 | 64 src/_pytest/python_api.py | 567 | 686| 1043 | 40860 | 198383 | 
| 176 | 64 src/_pytest/logging.py | 563 | 593| 233 | 41093 | 198383 | 
| 177 | 65 bench/bench_argcomplete.py | 1 | 20| 179 | 41272 | 198562 | 
| 178 | 65 src/_pytest/reports.py | 404 | 428| 171 | 41443 | 198562 | 
| 179 | 65 src/_pytest/debugging.py | 160 | 202| 351 | 41794 | 198562 | 
| 180 | 65 testing/python/collect.py | 79 | 113| 255 | 42049 | 198562 | 
| 181 | 66 doc/en/example/nonpython/conftest.py | 1 | 47| 314 | 42363 | 198876 | 
| 182 | 66 src/_pytest/terminal.py | 256 | 280| 161 | 42524 | 198876 | 
| 183 | 66 testing/python/collect.py | 812 | 843| 264 | 42788 | 198876 | 
| 184 | 66 src/_pytest/junitxml.py | 508 | 533| 178 | 42966 | 198876 | 
| 185 | 66 src/_pytest/skipping.py | 90 | 105| 130 | 43096 | 198876 | 
| 186 | 66 testing/python/integration.py | 37 | 68| 226 | 43322 | 198876 | 
| 187 | 67 doc/en/example/assertion/global_testmodule_config/conftest.py | 1 | 15| 0 | 43322 | 198957 | 
| 188 | 67 src/_pytest/_code/code.py | 694 | 718| 236 | 43558 | 198957 | 


## Patch

```diff
diff --git a/src/_pytest/pastebin.py b/src/_pytest/pastebin.py
--- a/src/_pytest/pastebin.py
+++ b/src/_pytest/pastebin.py
@@ -65,7 +65,7 @@ def create_new_paste(contents):
     from urllib.request import urlopen
     from urllib.parse import urlencode
 
-    params = {"code": contents, "lexer": "python3", "expiry": "1week"}
+    params = {"code": contents, "lexer": "text", "expiry": "1week"}
     url = "https://bpaste.net"
     try:
         response = (

```

## Test Patch

```diff
diff --git a/testing/test_pastebin.py b/testing/test_pastebin.py
--- a/testing/test_pastebin.py
+++ b/testing/test_pastebin.py
@@ -165,7 +165,7 @@ def test_create_new_paste(self, pastebin, mocked_urlopen):
         assert len(mocked_urlopen) == 1
         url, data = mocked_urlopen[0]
         assert type(data) is bytes
-        lexer = "python3"
+        lexer = "text"
         assert url == "https://bpaste.net"
         assert "lexer=%s" % lexer in data.decode()
         assert "code=full-paste-contents" in data.decode()

```


## Code snippets

### 1 - src/_pytest/pastebin.py:

Start line: 1, End line: 38

```python
""" submit failure or test session information to a pastebin service. """
import tempfile

import pytest


def pytest_addoption(parser):
    group = parser.getgroup("terminal reporting")
    group._addoption(
        "--pastebin",
        metavar="mode",
        action="store",
        dest="pastebin",
        default=None,
        choices=["failed", "all"],
        help="send failed|all info to bpaste.net pastebin service.",
    )


@pytest.hookimpl(trylast=True)
def pytest_configure(config):
    if config.option.pastebin == "all":
        tr = config.pluginmanager.getplugin("terminalreporter")
        # if no terminal reporter plugin is present, nothing we can do here;
        # this can happen when this function executes in a slave node
        # when using pytest-xdist, for example
        if tr is not None:
            # pastebin file will be utf-8 encoded binary file
            config._pastebinfile = tempfile.TemporaryFile("w+b")
            oldwrite = tr._tw.write

            def tee_write(s, **kwargs):
                oldwrite(s, **kwargs)
                if isinstance(s, str):
                    s = s.encode("utf-8")
                config._pastebinfile.write(s)

            tr._tw.write = tee_write
```
### 2 - src/_pytest/pastebin.py:

Start line: 57, End line: 80

```python
def create_new_paste(contents):
    """
    Creates a new paste using bpaste.net service.

    :contents: paste contents as utf-8 encoded bytes
    :returns: url to the pasted contents or error message
    """
    import re
    from urllib.request import urlopen
    from urllib.parse import urlencode

    params = {"code": contents, "lexer": "python3", "expiry": "1week"}
    url = "https://bpaste.net"
    try:
        response = (
            urlopen(url, data=urlencode(params).encode("ascii")).read().decode("utf-8")
        )
    except OSError as exc_info:  # urllib errors
        return "bad response: %s" % exc_info
    m = re.search(r'href="/raw/(\w+)"', response)
    if m:
        return "{}/show/{}".format(url, m.group(1))
    else:
        return "bad response: invalid format ('" + response + "')"
```
### 3 - src/_pytest/pastebin.py:

Start line: 83, End line: 104

```python
def pytest_terminal_summary(terminalreporter):
    import _pytest.config

    if terminalreporter.config.option.pastebin != "failed":
        return
    tr = terminalreporter
    if "failed" in tr.stats:
        terminalreporter.write_sep("=", "Sending information to Paste Service")
        for rep in terminalreporter.stats.get("failed"):
            try:
                msg = rep.longrepr.reprtraceback.reprentries[-1].reprfileloc
            except AttributeError:
                msg = tr._getfailureheadline(rep)
            tw = _pytest.config.create_terminal_writer(
                terminalreporter.config, stringio=True
            )
            rep.toterminal(tw)
            s = tw.stringio.getvalue()
            assert len(s)
            pastebinurl = create_new_paste(s)
            tr.write_line("{} --> {}".format(msg, pastebinurl))
```
### 4 - src/_pytest/pastebin.py:

Start line: 41, End line: 54

```python
def pytest_unconfigure(config):
    if hasattr(config, "_pastebinfile"):
        # get terminal contents and delete file
        config._pastebinfile.seek(0)
        sessionlog = config._pastebinfile.read()
        config._pastebinfile.close()
        del config._pastebinfile
        # undo our patching in the terminal reporter
        tr = config.pluginmanager.getplugin("terminalreporter")
        del tr._tw.__dict__["write"]
        # write summary
        tr.write_sep("=", "Sending information to Paste Service")
        pastebinurl = create_new_paste(sessionlog)
        tr.write_line("pastebin session-log: %s\n" % pastebinurl)
```
### 5 - src/_pytest/faulthandler.py:

Start line: 1, End line: 36

```python
import io
import os
import sys

import pytest


def pytest_addoption(parser):
    help = (
        "Dump the traceback of all threads if a test takes "
        "more than TIMEOUT seconds to finish.\n"
        "Not available on Windows."
    )
    parser.addini("faulthandler_timeout", help, default=0.0)


def pytest_configure(config):
    import faulthandler

    # avoid trying to dup sys.stderr if faulthandler is already enabled
    if faulthandler.is_enabled():
        return

    stderr_fd_copy = os.dup(_get_stderr_fileno())
    config.fault_handler_stderr = os.fdopen(stderr_fd_copy, "w")
    faulthandler.enable(file=config.fault_handler_stderr)


def _get_stderr_fileno():
    try:
        return sys.stderr.fileno()
    except (AttributeError, io.UnsupportedOperation):
        # python-xdist monkeypatches sys.stderr with an object that is not an actual file.
        # https://docs.python.org/3/library/faulthandler.html#issue-with-file-descriptors
        # This is potentially dangerous, but the best we can do.
        return sys.__stderr__.fileno()
```
### 6 - src/pytest.py:

Start line: 1, End line: 107

```python
# PYTHON_ARGCOMPLETE_OK
"""
pytest: unit and functional testing with Python.
"""
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
from _pytest.main import ExitCode
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
    "ExitCode",
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
### 7 - src/_pytest/skipping.py:

Start line: 28, End line: 65

```python
def pytest_configure(config):
    if config.option.runxfail:
        # yay a hack
        import pytest

        old = pytest.xfail
        config._cleanup.append(lambda: setattr(pytest, "xfail", old))

        def nop(*args, **kwargs):
            pass

        nop.Exception = xfail.Exception
        setattr(pytest, "xfail", nop)

    config.addinivalue_line(
        "markers",
        "skip(reason=None): skip the given test function with an optional reason. "
        'Example: skip(reason="no way of currently testing this") skips the '
        "test.",
    )
    config.addinivalue_line(
        "markers",
        "skipif(condition): skip the given test function if eval(condition) "
        "results in a True value.  Evaluation happens within the "
        "module global context. Example: skipif('sys.platform == \"win32\"') "
        "skips the test if we are on the win32 platform. see "
        "https://docs.pytest.org/en/latest/skipping.html",
    )
    config.addinivalue_line(
        "markers",
        "xfail(condition, reason=None, run=True, raises=None, strict=False): "
        "mark the test function as an expected failure if eval(condition) "
        "has a True value. Optionally specify a reason for better reporting "
        "and run=False if you don't even want to execute the test function. "
        "If only specific exception(s) are expected, you can list them in "
        "raises, and if the test fails in other ways, it will be reported as "
        "a true failure. See https://docs.pytest.org/en/latest/skipping.html",
    )
```
### 8 - src/_pytest/pytester.py:

Start line: 32, End line: 54

```python
def pytest_addoption(parser):
    parser.addoption(
        "--lsof",
        action="store_true",
        dest="lsof",
        default=False,
        help="run FD checks if lsof is available",
    )

    parser.addoption(
        "--runpytest",
        default="inprocess",
        dest="runpytest",
        choices=("inprocess", "subprocess"),
        help=(
            "run pytest sub runs in tests using an 'inprocess' "
            "or 'subprocess' (python -m main) method"
        ),
    )

    parser.addini(
        "pytester_example_dir", help="directory to take the pytester example files from"
    )
```
### 9 - src/_pytest/python.py:

Start line: 56, End line: 108

```python
def pytest_addoption(parser):
    group = parser.getgroup("general")
    group.addoption(
        "--fixtures",
        "--funcargs",
        action="store_true",
        dest="showfixtures",
        default=False,
        help="show available fixtures, sorted by plugin appearance "
        "(fixtures with leading '_' are only shown with '-v')",
    )
    group.addoption(
        "--fixtures-per-test",
        action="store_true",
        dest="show_fixtures_per_test",
        default=False,
        help="show fixtures per test",
    )
    parser.addini(
        "python_files",
        type="args",
        # NOTE: default is also used in AssertionRewritingHook.
        default=["test_*.py", "*_test.py"],
        help="glob-style file patterns for Python test module discovery",
    )
    parser.addini(
        "python_classes",
        type="args",
        default=["Test"],
        help="prefixes or glob names for Python test class discovery",
    )
    parser.addini(
        "python_functions",
        type="args",
        default=["test"],
        help="prefixes or glob names for Python test function and method discovery",
    )
    parser.addini(
        "disable_test_id_escaping_and_forfeit_all_rights_to_community_support",
        type="bool",
        default=False,
        help="disable string escape non-ascii characters, might cause unwanted "
        "side effects(use at your own risk)",
    )

    group.addoption(
        "--import-mode",
        default="prepend",
        choices=["prepend", "append"],
        dest="importmode",
        help="prepend/append to sys.path when importing test modules, "
        "default is to prepend.",
    )
```
### 10 - src/_pytest/_io/saferepr.py:

Start line: 1, End line: 21

```python
import pprint
import reprlib


def _format_repr_exception(exc, obj):
    exc_name = type(exc).__name__
    try:
        exc_info = str(exc)
    except Exception:
        exc_info = "unknown"
    return '<[{}("{}") raised in repr()] {} object at 0x{:x}>'.format(
        exc_name, exc_info, obj.__class__.__name__, id(obj)
    )


def _ellipsize(s, maxsize):
    if len(s) > maxsize:
        i = max(0, (maxsize - 3) // 2)
        j = max(0, maxsize - 3 - i)
        return s[:i] + "..." + s[len(s) - j :]
    return s
```
