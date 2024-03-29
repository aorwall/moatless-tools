# pytest-dev__pytest-5809

| **pytest-dev/pytest** | `8aba863a634f40560e25055d179220f0eefabe9a` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 345 |
| **Any found context length** | 345 |
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
@@ -77,11 +77,7 @@ def create_new_paste(contents):
         from urllib.request import urlopen
         from urllib.parse import urlencode
 
-    params = {
-        "code": contents,
-        "lexer": "python3" if sys.version_info[0] >= 3 else "python",
-        "expiry": "1week",
-    }
+    params = {"code": contents, "lexer": "text", "expiry": "1week"}
     url = "https://bpaste.net"
     response = urlopen(url, data=urlencode(params).encode("ascii")).read()
     m = re.search(r'href="/raw/(\w+)"', response.decode("utf-8"))

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| src/_pytest/pastebin.py | 80 | 84 | 2 | 1 | 345


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
| 1 | **1 src/_pytest/pastebin.py** | 1 | 25| 130 | 130 | 854 | 
| **-> 2 <-** | **1 src/_pytest/pastebin.py** | 65 | 91| 215 | 345 | 854 | 
| 3 | **1 src/_pytest/pastebin.py** | 28 | 46| 177 | 522 | 854 | 
| 4 | **1 src/_pytest/pastebin.py** | 94 | 115| 185 | 707 | 854 | 
| 5 | **1 src/_pytest/pastebin.py** | 49 | 62| 145 | 852 | 854 | 
| 6 | 2 src/pytest.py | 1 | 108| 731 | 1583 | 1585 | 
| 7 | 3 src/_pytest/python_api.py | 673 | 716| 379 | 1962 | 8022 | 
| 8 | 4 src/_pytest/skipping.py | 33 | 70| 363 | 2325 | 9581 | 
| 9 | 5 src/_pytest/pytester.py | 41 | 63| 144 | 2469 | 20110 | 
| 10 | 6 src/_pytest/python.py | 64 | 116| 354 | 2823 | 31935 | 
| 11 | 7 doc/en/example/py2py3/conftest.py | 1 | 18| 0 | 2823 | 32029 | 
| 12 | 8 src/_pytest/_code/code.py | 1 | 30| 171 | 2994 | 40061 | 
| 13 | 9 doc/en/conf.py | 244 | 344| 663 | 3657 | 42546 | 
| 14 | 10 src/_pytest/debugging.py | 28 | 49| 155 | 3812 | 45027 | 
| 15 | 10 src/_pytest/pytester.py | 1 | 38| 234 | 4046 | 45027 | 
| 16 | 11 src/_pytest/config/exceptions.py | 1 | 11| 0 | 4046 | 45080 | 
| 17 | 12 testing/python/raises.py | 284 | 326| 348 | 4394 | 47275 | 
| 18 | 12 src/_pytest/_code/code.py | 1046 | 1073| 231 | 4625 | 47275 | 
| 19 | 13 bench/skip.py | 1 | 13| 0 | 4625 | 47324 | 
| 20 | 14 src/_pytest/deprecated.py | 71 | 97| 237 | 4862 | 48307 | 
| 21 | 15 src/_pytest/doctest.py | 479 | 515| 343 | 5205 | 52574 | 
| 22 | 16 src/_pytest/assertion/__init__.py | 1 | 32| 175 | 5380 | 53664 | 
| 23 | 17 src/_pytest/_io/saferepr.py | 1 | 22| 138 | 5518 | 54347 | 
| 24 | 18 testing/python/collect.py | 1259 | 1290| 203 | 5721 | 64042 | 
| 25 | 18 src/_pytest/python_api.py | 719 | 744| 185 | 5906 | 64042 | 
| 26 | 19 src/_pytest/main.py | 37 | 145| 757 | 6663 | 69871 | 
| 27 | 20 src/_pytest/capture.py | 427 | 460| 216 | 6879 | 76038 | 
| 28 | 20 src/_pytest/doctest.py | 41 | 88| 321 | 7200 | 76038 | 
| 29 | 21 src/_pytest/setupplan.py | 1 | 33| 194 | 7394 | 76233 | 
| 30 | 21 src/_pytest/pytester.py | 66 | 84| 129 | 7523 | 76233 | 
| 31 | 22 src/_pytest/cacheprovider.py | 299 | 354| 410 | 7933 | 79669 | 
| 32 | 22 src/_pytest/capture.py | 788 | 851| 508 | 8441 | 79669 | 
| 33 | 23 doc/en/example/conftest.py | 1 | 3| 0 | 8441 | 79684 | 
| 34 | 24 src/_pytest/terminal.py | 59 | 147| 632 | 9073 | 88164 | 
| 35 | 24 src/_pytest/main.py | 146 | 172| 175 | 9248 | 88164 | 
| 36 | 24 src/_pytest/doctest.py | 316 | 343| 211 | 9459 | 88164 | 
| 37 | 25 doc/en/example/xfail_demo.py | 1 | 40| 151 | 9610 | 88316 | 
| 38 | 26 src/_pytest/reports.py | 269 | 278| 134 | 9744 | 91301 | 
| 39 | 27 src/_pytest/warnings.py | 40 | 62| 157 | 9901 | 92626 | 
| 40 | 28 doc/en/conftest.py | 1 | 3| 0 | 9901 | 92641 | 
| 41 | 29 bench/empty.py | 1 | 4| 0 | 9901 | 92671 | 
| 42 | 30 src/_pytest/junitxml.py | 391 | 435| 334 | 10235 | 97779 | 
| 43 | 31 testing/example_scripts/conftest_usageerror/conftest.py | 1 | 10| 0 | 10235 | 97818 | 
| 44 | 32 src/_pytest/resultlog.py | 1 | 23| 119 | 10354 | 98575 | 
| 45 | 32 src/_pytest/python.py | 119 | 137| 176 | 10530 | 98575 | 
| 46 | 33 bench/manyparam.py | 1 | 16| 0 | 10530 | 98624 | 
| 47 | 33 doc/en/conf.py | 112 | 243| 843 | 11373 | 98624 | 
| 48 | 34 src/_pytest/config/argparsing.py | 322 | 345| 201 | 11574 | 102020 | 
| 49 | 35 src/_pytest/runner.py | 157 | 199| 338 | 11912 | 104799 | 
| 50 | 35 src/_pytest/config/argparsing.py | 271 | 282| 126 | 12038 | 104799 | 
| 51 | 36 src/_pytest/helpconfig.py | 44 | 88| 297 | 12335 | 106535 | 
| 52 | 36 src/_pytest/reports.py | 1 | 32| 236 | 12571 | 106535 | 
| 53 | 37 setup.py | 1 | 20| 207 | 12778 | 106871 | 
| 54 | 38 src/_pytest/assertion/rewrite.py | 1 | 56| 375 | 13153 | 116185 | 
| 55 | 39 extra/get_issues.py | 56 | 87| 231 | 13384 | 116736 | 
| 56 | 40 src/_pytest/mark/__init__.py | 41 | 79| 341 | 13725 | 117916 | 
| 57 | 40 src/_pytest/helpconfig.py | 91 | 119| 218 | 13943 | 117916 | 
| 58 | 41 doc/en/example/costlysetup/conftest.py | 1 | 22| 0 | 13943 | 118004 | 
| 59 | 41 src/_pytest/cacheprovider.py | 208 | 262| 494 | 14437 | 118004 | 
| 60 | 42 src/_pytest/compat.py | 1 | 98| 636 | 15073 | 121158 | 
| 61 | 43 testing/python/integration.py | 1 | 36| 248 | 15321 | 123939 | 
| 62 | 44 src/_pytest/outcomes.py | 36 | 78| 264 | 15585 | 125257 | 
| 63 | 44 src/_pytest/pytester.py | 132 | 153| 244 | 15829 | 125257 | 
| 64 | 44 src/_pytest/helpconfig.py | 229 | 248| 145 | 15974 | 125257 | 
| 65 | 44 src/_pytest/main.py | 385 | 418| 201 | 16175 | 125257 | 
| 66 | 44 src/_pytest/python.py | 160 | 175| 209 | 16384 | 125257 | 
| 67 | 44 src/_pytest/python.py | 1242 | 1268| 208 | 16592 | 125257 | 
| 68 | 44 src/_pytest/deprecated.py | 1 | 69| 746 | 17338 | 125257 | 
| 69 | 44 src/_pytest/mark/__init__.py | 82 | 99| 124 | 17462 | 125257 | 
| 70 | 44 testing/python/collect.py | 1200 | 1226| 182 | 17644 | 125257 | 
| 71 | 44 src/_pytest/assertion/rewrite.py | 328 | 356| 351 | 17995 | 125257 | 
| 72 | 45 doc/en/example/assertion/failure_demo.py | 1 | 41| 178 | 18173 | 126920 | 
| 73 | 46 src/_pytest/setuponly.py | 1 | 24| 122 | 18295 | 127515 | 
| 74 | 47 src/_pytest/__init__.py | 1 | 10| 0 | 18295 | 127579 | 
| 75 | 47 setup.py | 23 | 45| 129 | 18424 | 127579 | 
| 76 | 48 src/_pytest/stepwise.py | 1 | 24| 139 | 18563 | 128301 | 
| 77 | 48 src/_pytest/junitxml.py | 454 | 473| 143 | 18706 | 128301 | 
| 78 | 49 src/_pytest/assertion/util.py | 100 | 136| 159 | 18865 | 131567 | 
| 79 | 49 src/_pytest/python.py | 1300 | 1369| 493 | 19358 | 131567 | 
| 80 | 50 testing/python/metafunc.py | 1289 | 1327| 244 | 19602 | 144760 | 
| 81 | 50 src/_pytest/terminal.py | 698 | 710| 119 | 19721 | 144760 | 
| 82 | 51 testing/python/fixtures.py | 1 | 44| 259 | 19980 | 169867 | 
| 83 | 51 src/_pytest/terminal.py | 396 | 445| 421 | 20401 | 169867 | 
| 84 | 51 extra/get_issues.py | 1 | 31| 176 | 20577 | 169867 | 
| 85 | 51 src/_pytest/skipping.py | 1 | 30| 193 | 20770 | 169867 | 
| 86 | 51 src/_pytest/config/argparsing.py | 226 | 241| 128 | 20898 | 169867 | 
| 87 | 51 src/_pytest/pytester.py | 101 | 130| 194 | 21092 | 169867 | 
| 88 | 51 src/_pytest/debugging.py | 1 | 25| 153 | 21245 | 169867 | 
| 89 | 51 src/_pytest/junitxml.py | 222 | 239| 164 | 21409 | 169867 | 
| 90 | 51 src/_pytest/doctest.py | 552 | 584| 230 | 21639 | 169867 | 
| 91 | 52 doc/en/example/costlysetup/sub_b/__init__.py | 1 | 3| 0 | 21639 | 169876 | 
| 92 | 52 src/_pytest/junitxml.py | 241 | 255| 133 | 21772 | 169876 | 
| 93 | 52 src/_pytest/resultlog.py | 26 | 48| 193 | 21965 | 169876 | 
| 94 | 52 src/_pytest/junitxml.py | 169 | 220| 334 | 22299 | 169876 | 
| 95 | 52 src/_pytest/python.py | 140 | 157| 212 | 22511 | 169876 | 
| 96 | 52 src/_pytest/terminal.py | 150 | 177| 216 | 22727 | 169876 | 
| 97 | 53 doc/en/_themes/flask_theme_support.py | 1 | 15| 117 | 22844 | 171157 | 
| 98 | 54 src/_pytest/config/__init__.py | 709 | 721| 140 | 22984 | 179513 | 
| 99 | 54 testing/python/collect.py | 928 | 947| 179 | 23163 | 179513 | 
| 100 | 54 doc/en/example/assertion/failure_demo.py | 165 | 203| 256 | 23419 | 179513 | 
| 101 | 54 src/_pytest/terminal.py | 846 | 859| 130 | 23549 | 179513 | 
| 102 | 54 src/_pytest/debugging.py | 291 | 317| 219 | 23768 | 179513 | 
| 103 | 54 src/_pytest/skipping.py | 125 | 187| 583 | 24351 | 179513 | 
| 104 | 54 doc/en/example/assertion/failure_demo.py | 44 | 122| 683 | 25034 | 179513 | 
| 105 | 55 src/_pytest/fixtures.py | 699 | 750| 475 | 25509 | 190501 | 
| 106 | 55 src/_pytest/terminal.py | 823 | 844| 181 | 25690 | 190501 | 
| 107 | 55 src/_pytest/main.py | 345 | 382| 333 | 26023 | 190501 | 
| 108 | 55 src/_pytest/config/argparsing.py | 75 | 98| 229 | 26252 | 190501 | 
| 109 | 55 src/_pytest/junitxml.py | 438 | 451| 119 | 26371 | 190501 | 
| 110 | 55 src/_pytest/capture.py | 1 | 24| 114 | 26485 | 190501 | 
| 111 | 55 src/_pytest/terminal.py | 734 | 757| 143 | 26628 | 190501 | 
| 112 | 55 src/_pytest/helpconfig.py | 1 | 41| 278 | 26906 | 190501 | 
| 113 | 55 testing/python/metafunc.py | 57 | 78| 212 | 27118 | 190501 | 
| 114 | 55 src/_pytest/reports.py | 191 | 206| 159 | 27277 | 190501 | 
| 115 | 55 src/_pytest/compat.py | 321 | 458| 806 | 28083 | 190501 | 
| 116 | 55 src/_pytest/capture.py | 27 | 43| 117 | 28200 | 190501 | 
| 117 | 55 src/_pytest/config/argparsing.py | 243 | 269| 224 | 28424 | 190501 | 
| 118 | 56 testing/freeze/tox_run.py | 1 | 14| 0 | 28424 | 190594 | 
| 119 | 56 src/_pytest/python_api.py | 553 | 672| 1035 | 29459 | 190594 | 
| 120 | 56 src/_pytest/doctest.py | 91 | 132| 319 | 29778 | 190594 | 
| 121 | 56 doc/en/example/assertion/failure_demo.py | 206 | 253| 229 | 30007 | 190594 | 
| 122 | 56 src/_pytest/pytester.py | 325 | 358| 157 | 30164 | 190594 | 
| 123 | 56 src/_pytest/runner.py | 1 | 36| 199 | 30363 | 190594 | 
| 124 | 56 testing/python/integration.py | 72 | 87| 109 | 30472 | 190594 | 
| 125 | 56 src/_pytest/junitxml.py | 257 | 285| 223 | 30695 | 190594 | 
| 126 | 57 testing/example_scripts/issue_519.py | 1 | 32| 358 | 31053 | 191068 | 
| 127 | 58 src/_pytest/_code/source.py | 235 | 263| 162 | 31215 | 193433 | 
| 128 | 58 src/_pytest/helpconfig.py | 144 | 211| 554 | 31769 | 193433 | 
| 129 | 59 src/_pytest/logging.py | 119 | 200| 534 | 32303 | 198524 | 
| 130 | 59 src/_pytest/fixtures.py | 780 | 809| 245 | 32548 | 198524 | 
| 131 | 59 src/_pytest/terminal.py | 885 | 915| 260 | 32808 | 198524 | 
| 132 | 59 testing/python/metafunc.py | 636 | 664| 231 | 33039 | 198524 | 
| 133 | 59 src/_pytest/terminal.py | 346 | 359| 126 | 33165 | 198524 | 
| 134 | 59 src/_pytest/terminal.py | 797 | 821| 204 | 33369 | 198524 | 
| 135 | 59 src/_pytest/_code/code.py | 585 | 628| 308 | 33677 | 198524 | 
| 136 | 59 src/_pytest/config/__init__.py | 1 | 48| 298 | 33975 | 198524 | 
| 137 | 59 testing/python/metafunc.py | 126 | 181| 406 | 34381 | 198524 | 
| 138 | 59 src/_pytest/cacheprovider.py | 152 | 206| 501 | 34882 | 198524 | 
| 139 | 59 src/_pytest/reports.py | 130 | 156| 159 | 35041 | 198524 | 
| 140 | 60 doc/en/example/costlysetup/sub_a/__init__.py | 1 | 3| 0 | 35041 | 198533 | 
| 141 | 60 src/_pytest/debugging.py | 279 | 288| 132 | 35173 | 198533 | 
| 142 | 60 src/_pytest/config/__init__.py | 1024 | 1048| 149 | 35322 | 198533 | 
| 143 | 60 testing/python/collect.py | 565 | 664| 663 | 35985 | 198533 | 
| 144 | 60 src/_pytest/_code/code.py | 903 | 944| 294 | 36279 | 198533 | 
| 145 | 60 testing/python/raises.py | 1 | 69| 464 | 36743 | 198533 | 
| 146 | 61 doc/en/example/pythoncollection.py | 1 | 16| 0 | 36743 | 198589 | 
| 147 | 61 src/_pytest/runner.py | 119 | 132| 129 | 36872 | 198589 | 
| 148 | 61 src/_pytest/terminal.py | 305 | 326| 170 | 37042 | 198589 | 


## Patch

```diff
diff --git a/src/_pytest/pastebin.py b/src/_pytest/pastebin.py
--- a/src/_pytest/pastebin.py
+++ b/src/_pytest/pastebin.py
@@ -77,11 +77,7 @@ def create_new_paste(contents):
         from urllib.request import urlopen
         from urllib.parse import urlencode
 
-    params = {
-        "code": contents,
-        "lexer": "python3" if sys.version_info[0] >= 3 else "python",
-        "expiry": "1week",
-    }
+    params = {"code": contents, "lexer": "text", "expiry": "1week"}
     url = "https://bpaste.net"
     response = urlopen(url, data=urlencode(params).encode("ascii")).read()
     m = re.search(r'href="/raw/(\w+)"', response.decode("utf-8"))

```

## Test Patch

```diff
diff --git a/testing/test_pastebin.py b/testing/test_pastebin.py
--- a/testing/test_pastebin.py
+++ b/testing/test_pastebin.py
@@ -126,7 +126,7 @@ def test_create_new_paste(self, pastebin, mocked_urlopen):
         assert len(mocked_urlopen) == 1
         url, data = mocked_urlopen[0]
         assert type(data) is bytes
-        lexer = "python3" if sys.version_info[0] >= 3 else "python"
+        lexer = "text"
         assert url == "https://bpaste.net"
         assert "lexer=%s" % lexer in data.decode()
         assert "code=full-paste-contents" in data.decode()

```


## Code snippets

### 1 - src/_pytest/pastebin.py:

Start line: 1, End line: 25

```python
# -*- coding: utf-8 -*-
""" submit failure or test session information to a pastebin service. """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import tempfile

import six

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
```
### 2 - src/_pytest/pastebin.py:

Start line: 65, End line: 91

```python
def create_new_paste(contents):
    """
    Creates a new paste using bpaste.net service.

    :contents: paste contents as utf-8 encoded bytes
    :returns: url to the pasted contents
    """
    import re

    if sys.version_info < (3, 0):
        from urllib import urlopen, urlencode
    else:
        from urllib.request import urlopen
        from urllib.parse import urlencode

    params = {
        "code": contents,
        "lexer": "python3" if sys.version_info[0] >= 3 else "python",
        "expiry": "1week",
    }
    url = "https://bpaste.net"
    response = urlopen(url, data=urlencode(params).encode("ascii")).read()
    m = re.search(r'href="/raw/(\w+)"', response.decode("utf-8"))
    if m:
        return "%s/show/%s" % (url, m.group(1))
    else:
        return "bad response: " + response
```
### 3 - src/_pytest/pastebin.py:

Start line: 28, End line: 46

```python
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
                if isinstance(s, six.text_type):
                    s = s.encode("utf-8")
                config._pastebinfile.write(s)

            tr._tw.write = tee_write
```
### 4 - src/_pytest/pastebin.py:

Start line: 94, End line: 115

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
            tr.write_line("%s --> %s" % (msg, pastebinurl))
```
### 5 - src/_pytest/pastebin.py:

Start line: 49, End line: 62

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
### 6 - src/pytest.py:

Start line: 1, End line: 108

```python
# -*- coding: utf-8 -*-
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
### 7 - src/_pytest/python_api.py:

Start line: 673, End line: 716

```python
def raises(expected_exception, *args, **kwargs):
    __tracebackhide__ = True
    for exc in filterfalse(isclass, always_iterable(expected_exception, BASE_TYPE)):
        msg = (
            "exceptions must be old-style classes or"
            " derived from BaseException, not %s"
        )
        raise TypeError(msg % type(exc))

    message = "DID NOT RAISE {}".format(expected_exception)
    match_expr = None

    if not args:
        if "message" in kwargs:
            message = kwargs.pop("message")
            warnings.warn(deprecated.RAISES_MESSAGE_PARAMETER, stacklevel=2)
        if "match" in kwargs:
            match_expr = kwargs.pop("match")
        if kwargs:
            msg = "Unexpected keyword arguments passed to pytest.raises: "
            msg += ", ".join(sorted(kwargs))
            raise TypeError(msg)
        return RaisesContext(expected_exception, message, match_expr)
    elif isinstance(args[0], str):
        warnings.warn(deprecated.RAISES_EXEC, stacklevel=2)
        code, = args
        assert isinstance(code, str)
        frame = sys._getframe(1)
        loc = frame.f_locals.copy()
        loc.update(kwargs)
        # print "raises frame scope: %r" % frame.f_locals
        try:
            code = _pytest._code.Source(code).compile(_genframe=frame)
            exec(code, frame.f_globals, loc)
            # XXX didn't mean f_globals == f_locals something special?
            #     this is destroyed here ...
        except expected_exception:
            return _pytest._code.ExceptionInfo.from_current()
    else:
        func = args[0]
        try:
            func(*args[1:], **kwargs)
        except expected_exception:
            return _pytest._code.ExceptionInfo.from_current()
    fail(message)
```
### 8 - src/_pytest/skipping.py:

Start line: 33, End line: 70

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
### 9 - src/_pytest/pytester.py:

Start line: 41, End line: 63

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
### 10 - src/_pytest/python.py:

Start line: 64, End line: 116

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
