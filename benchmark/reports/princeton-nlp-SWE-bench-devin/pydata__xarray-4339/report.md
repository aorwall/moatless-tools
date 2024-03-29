# pydata__xarray-4339

| **pydata/xarray** | `3b5a8ee46be7fd00d7ea9093d1941cb6c3be191c` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | 13107 |
| **Avg pos** | 301.0 |
| **Min pos** | 38 |
| **Max pos** | 59 |
| **Top file pos** | 9 |
| **Missing snippets** | 15 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/xarray/core/accessor_str.py b/xarray/core/accessor_str.py
--- a/xarray/core/accessor_str.py
+++ b/xarray/core/accessor_str.py
@@ -90,7 +90,7 @@ def _apply(self, f, dtype=None):
 
     def len(self):
         """
-        Compute the length of each element in the array.
+        Compute the length of each string in the array.
 
         Returns
         -------
@@ -104,9 +104,9 @@ def __getitem__(self, key):
         else:
             return self.get(key)
 
-    def get(self, i):
+    def get(self, i, default=""):
         """
-        Extract element from indexable in each element in the array.
+        Extract character number `i` from each string in the array.
 
         Parameters
         ----------
@@ -120,12 +120,18 @@ def get(self, i):
         -------
         items : array of objects
         """
-        obj = slice(-1, None) if i == -1 else slice(i, i + 1)
-        return self._apply(lambda x: x[obj])
+        s = slice(-1, None) if i == -1 else slice(i, i + 1)
+
+        def f(x):
+            item = x[s]
+
+            return item if item else default
+
+        return self._apply(f)
 
     def slice(self, start=None, stop=None, step=None):
         """
-        Slice substrings from each element in the array.
+        Slice substrings from each string in the array.
 
         Parameters
         ----------
@@ -359,7 +365,7 @@ def count(self, pat, flags=0):
 
     def startswith(self, pat):
         """
-        Test if the start of each string element matches a pattern.
+        Test if the start of each string in the array matches a pattern.
 
         Parameters
         ----------
@@ -378,7 +384,7 @@ def startswith(self, pat):
 
     def endswith(self, pat):
         """
-        Test if the end of each string element matches a pattern.
+        Test if the end of each string in the array matches a pattern.
 
         Parameters
         ----------
@@ -432,8 +438,7 @@ def pad(self, width, side="left", fillchar=" "):
 
     def center(self, width, fillchar=" "):
         """
-        Filling left and right side of strings in the array with an
-        additional character.
+        Pad left and right side of each string in the array.
 
         Parameters
         ----------
@@ -451,8 +456,7 @@ def center(self, width, fillchar=" "):
 
     def ljust(self, width, fillchar=" "):
         """
-        Filling right side of strings in the array with an additional
-        character.
+        Pad right side of each string in the array.
 
         Parameters
         ----------
@@ -470,7 +474,7 @@ def ljust(self, width, fillchar=" "):
 
     def rjust(self, width, fillchar=" "):
         """
-        Filling left side of strings in the array with an additional character.
+        Pad left side of each string in the array.
 
         Parameters
         ----------
@@ -488,7 +492,7 @@ def rjust(self, width, fillchar=" "):
 
     def zfill(self, width):
         """
-        Pad strings in the array by prepending '0' characters.
+        Pad each string in the array by prepending '0' characters.
 
         Strings in the array are padded with '0' characters on the
         left of the string to reach a total string length  `width`. Strings
@@ -508,7 +512,7 @@ def zfill(self, width):
 
     def contains(self, pat, case=True, flags=0, regex=True):
         """
-        Test if pattern or regex is contained within a string of the array.
+        Test if pattern or regex is contained within each string of the array.
 
         Return boolean array based on whether a given pattern or regex is
         contained within a string of the array.
@@ -554,7 +558,7 @@ def contains(self, pat, case=True, flags=0, regex=True):
 
     def match(self, pat, case=True, flags=0):
         """
-        Determine if each string matches a regular expression.
+        Determine if each string in the array matches a regular expression.
 
         Parameters
         ----------
@@ -613,7 +617,7 @@ def strip(self, to_strip=None, side="both"):
 
     def lstrip(self, to_strip=None):
         """
-        Remove leading and trailing characters.
+        Remove leading characters.
 
         Strip whitespaces (including newlines) or a set of specified characters
         from each string in the array from the left side.
@@ -633,7 +637,7 @@ def lstrip(self, to_strip=None):
 
     def rstrip(self, to_strip=None):
         """
-        Remove leading and trailing characters.
+        Remove trailing characters.
 
         Strip whitespaces (including newlines) or a set of specified characters
         from each string in the array from the right side.
@@ -653,8 +657,7 @@ def rstrip(self, to_strip=None):
 
     def wrap(self, width, **kwargs):
         """
-        Wrap long strings in the array to be formatted in paragraphs with
-        length less than a given width.
+        Wrap long strings in the array in paragraphs with length less than `width`.
 
         This method has the same keyword parameters and defaults as
         :class:`textwrap.TextWrapper`.
@@ -663,38 +666,20 @@ def wrap(self, width, **kwargs):
         ----------
         width : int
             Maximum line-width
-        expand_tabs : bool, optional
-            If true, tab characters will be expanded to spaces (default: True)
-        replace_whitespace : bool, optional
-            If true, each whitespace character (as defined by
-            string.whitespace) remaining after tab expansion will be replaced
-            by a single space (default: True)
-        drop_whitespace : bool, optional
-            If true, whitespace that, after wrapping, happens to end up at the
-            beginning or end of a line is dropped (default: True)
-        break_long_words : bool, optional
-            If true, then words longer than width will be broken in order to
-            ensure that no lines are longer than width. If it is false, long
-            words will not be broken, and some lines may be longer than width.
-            (default: True)
-        break_on_hyphens : bool, optional
-            If true, wrapping will occur preferably on whitespace and right
-            after hyphens in compound words, as it is customary in English. If
-            false, only whitespaces will be considered as potentially good
-            places for line breaks, but you need to set break_long_words to
-            false if you want truly insecable words. (default: True)
+        **kwargs
+            keyword arguments passed into :class:`textwrap.TextWrapper`.
 
         Returns
         -------
         wrapped : same type as values
         """
-        tw = textwrap.TextWrapper(width=width)
+        tw = textwrap.TextWrapper(width=width, **kwargs)
         f = lambda x: "\n".join(tw.wrap(x))
         return self._apply(f)
 
     def translate(self, table):
         """
-        Map all characters in the string through the given mapping table.
+        Map characters of each string through the given mapping table.
 
         Parameters
         ----------

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| xarray/core/accessor_str.py | 93 | 93 | 58 | 9 | 19240
| xarray/core/accessor_str.py | 107 | 109 | 59 | 9 | 19358
| xarray/core/accessor_str.py | 123 | 128 | - | 9 | -
| xarray/core/accessor_str.py | 362 | 362 | - | 9 | -
| xarray/core/accessor_str.py | 381 | 381 | - | 9 | -
| xarray/core/accessor_str.py | 435 | 436 | 53 | 9 | 17025
| xarray/core/accessor_str.py | 454 | 455 | 43 | 9 | 13977
| xarray/core/accessor_str.py | 473 | 473 | 38 | 9 | 13107
| xarray/core/accessor_str.py | 491 | 491 | 50 | 9 | 15815
| xarray/core/accessor_str.py | 511 | 511 | - | 9 | -
| xarray/core/accessor_str.py | 557 | 557 | - | 9 | -
| xarray/core/accessor_str.py | 616 | 616 | - | 9 | -
| xarray/core/accessor_str.py | 636 | 636 | - | 9 | -
| xarray/core/accessor_str.py | 656 | 657 | - | 9 | -
| xarray/core/accessor_str.py | 666 | 697 | - | 9 | -


## Problem Statement

```
missing parameter in DataArray.str.get
While working on #4286 I noticed that the docstring of `DataArray.str.get` claims to allow passing a default value in addition to the index, but the python code doesn't have that parameter at all.
I think the default value is a good idea and that we should make the code match the docstring.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 xarray/core/dataarray.py | 361 | 402| 384 | 384 | 34142 | 
| 2 | 2 xarray/conventions.py | 106 | 127| 168 | 552 | 39996 | 
| 3 | 2 xarray/core/dataarray.py | 2109 | 2134| 199 | 751 | 39996 | 
| 4 | 2 xarray/core/dataarray.py | 3485 | 3652| 161 | 912 | 39996 | 
| 5 | 2 xarray/core/dataarray.py | 265 | 359| 853 | 1765 | 39996 | 
| 6 | 2 xarray/core/dataarray.py | 404 | 430| 251 | 2016 | 39996 | 
| 7 | 2 xarray/core/dataarray.py | 2254 | 2276| 168 | 2184 | 39996 | 
| 8 | 2 xarray/core/dataarray.py | 1005 | 1066| 543 | 2727 | 39996 | 
| 9 | 2 xarray/core/dataarray.py | 645 | 660| 165 | 2892 | 39996 | 
| 10 | 2 xarray/core/dataarray.py | 2230 | 2252| 169 | 3061 | 39996 | 
| 11 | 2 xarray/core/dataarray.py | 2729 | 2755| 268 | 3329 | 39996 | 
| 12 | 3 xarray/core/variable.py | 1164 | 1174| 129 | 3458 | 60816 | 
| 13 | 3 xarray/core/dataarray.py | 1 | 81| 433 | 3891 | 60816 | 
| 14 | 3 xarray/core/dataarray.py | 2085 | 2107| 196 | 4087 | 60816 | 
| 15 | 3 xarray/core/dataarray.py | 625 | 643| 156 | 4243 | 60816 | 
| 16 | 3 xarray/core/dataarray.py | 3501 | 3652| 1804 | 6047 | 60816 | 
| 17 | 4 xarray/core/duck_array_ops.py | 327 | 352| 270 | 6317 | 65855 | 
| 18 | 4 xarray/core/dataarray.py | 938 | 955| 158 | 6475 | 65855 | 
| 19 | 5 xarray/core/ops.py | 109 | 134| 213 | 6688 | 68515 | 
| 20 | 5 xarray/core/dataarray.py | 2034 | 2052| 133 | 6821 | 68515 | 
| 21 | 5 xarray/core/dataarray.py | 2757 | 2789| 218 | 7039 | 68515 | 
| 22 | 5 xarray/conventions.py | 176 | 218| 400 | 7439 | 68515 | 
| 23 | 5 xarray/core/dataarray.py | 1158 | 1173| 134 | 7573 | 68515 | 
| 24 | 5 xarray/core/dataarray.py | 2700 | 2727| 232 | 7805 | 68515 | 
| 25 | 6 xarray/core/indexes.py | 70 | 87| 121 | 7926 | 69513 | 
| 26 | 6 xarray/conventions.py | 82 | 103| 204 | 8130 | 69513 | 
| 27 | 6 xarray/core/dataarray.py | 796 | 818| 185 | 8315 | 69513 | 
| 28 | 6 xarray/core/variable.py | 1249 | 1272| 371 | 8686 | 69513 | 
| 29 | 6 xarray/core/dataarray.py | 3848 | 3948| 1053 | 9739 | 69513 | 
| 30 | 6 xarray/core/dataarray.py | 1192 | 1207| 125 | 9864 | 69513 | 
| 31 | 6 xarray/core/dataarray.py | 759 | 794| 307 | 10171 | 69513 | 
| 32 | 6 xarray/core/variable.py | 1101 | 1135| 315 | 10486 | 69513 | 
| 33 | 6 xarray/core/dataarray.py | 2054 | 2083| 238 | 10724 | 69513 | 
| 34 | 6 xarray/core/variable.py | 727 | 765| 444 | 11168 | 69513 | 
| 35 | 7 xarray/core/dataset.py | 1 | 136| 647 | 11815 | 123569 | 
| 36 | 7 xarray/core/dataarray.py | 3654 | 3749| 931 | 12746 | 123569 | 
| 37 | 8 xarray/core/weighted.py | 227 | 263| 251 | 12997 | 125568 | 
| **-> 38 <-** | **9 xarray/core/accessor_str.py** | 471 | 487| 110 | 13107 | 131895 | 
| 39 | 9 xarray/core/dataarray.py | 187 | 212| 236 | 13343 | 131895 | 
| 40 | 10 xarray/core/dask_array_compat.py | 133 | 146| 180 | 13523 | 133632 | 
| 41 | 10 xarray/core/dataarray.py | 820 | 840| 168 | 13691 | 133632 | 
| 42 | 11 xarray/core/common.py | 376 | 391| 174 | 13865 | 147049 | 
| **-> 43 <-** | **11 xarray/core/accessor_str.py** | 452 | 469| 112 | 13977 | 147049 | 
| 44 | 11 xarray/core/dataarray.py | 2008 | 2032| 208 | 14185 | 147049 | 
| 45 | 11 xarray/core/variable.py | 1 | 71| 418 | 14603 | 147049 | 
| 46 | 11 xarray/core/dataarray.py | 1175 | 1190| 134 | 14737 | 147049 | 
| 47 | 11 xarray/core/dataarray.py | 2838 | 2885| 353 | 15090 | 147049 | 
| 48 | 12 xarray/core/formatting.py | 455 | 468| 134 | 15224 | 152332 | 
| 49 | 12 xarray/core/dataarray.py | 662 | 722| 448 | 15672 | 152332 | 
| **-> 50 <-** | **12 xarray/core/accessor_str.py** | 489 | 507| 143 | 15815 | 152332 | 
| 51 | 12 xarray/core/dataarray.py | 1273 | 1331| 524 | 16339 | 152332 | 
| 52 | 12 xarray/core/dataarray.py | 1333 | 1395| 573 | 16912 | 152332 | 
| **-> 53 <-** | **12 xarray/core/accessor_str.py** | 433 | 450| 113 | 17025 | 152332 | 
| 54 | 12 xarray/core/duck_array_ops.py | 82 | 105| 245 | 17270 | 152332 | 
| 55 | 13 xarray/backends/api.py | 1 | 59| 310 | 17580 | 163422 | 
| 56 | 13 xarray/core/dataarray.py | 1068 | 1156| 811 | 18391 | 163422 | 
| 57 | 13 xarray/core/dataarray.py | 519 | 608| 562 | 18953 | 163422 | 
| **-> 58 <-** | **13 xarray/core/accessor_str.py** | 64 | 105| 287 | 19240 | 163422 | 
| **-> 59 <-** | **13 xarray/core/accessor_str.py** | 107 | 124| 118 | 19358 | 163422 | 
| 60 | 13 xarray/core/dataarray.py | 449 | 475| 233 | 19591 | 163422 | 
| 61 | 14 asv_bench/benchmarks/indexing.py | 1 | 57| 730 | 20321 | 164834 | 
| 62 | 14 xarray/core/variable.py | 1274 | 1293| 202 | 20523 | 164834 | 
| 63 | 14 xarray/core/dataarray.py | 84 | 166| 691 | 21214 | 164834 | 
| 64 | 15 xarray/core/coordinates.py | 284 | 302| 174 | 21388 | 167806 | 
| 65 | 15 xarray/core/weighted.py | 183 | 224| 347 | 21735 | 167806 | 
| 66 | **15 xarray/core/accessor_str.py** | 398 | 431| 263 | 21998 | 167806 | 
| 67 | 16 xarray/__init__.py | 1 | 93| 603 | 22601 | 168409 | 
| 68 | 17 xarray/plot/plot.py | 116 | 200| 587 | 23188 | 176619 | 
| 69 | 17 xarray/core/duck_array_ops.py | 108 | 132| 275 | 23463 | 176619 | 
| 70 | 18 xarray/core/alignment.py | 606 | 652| 337 | 23800 | 182581 | 
| 71 | 18 xarray/core/dataarray.py | 3950 | 4060| 1122 | 24922 | 182581 | 
| 72 | 18 xarray/core/dataarray.py | 1558 | 1607| 375 | 25297 | 182581 | 
| 73 | 19 xarray/testing.py | 268 | 285| 213 | 25510 | 185567 | 
| 74 | 20 xarray/core/indexing.py | 666 | 687| 160 | 25670 | 197230 | 
| 75 | 20 xarray/core/ops.py | 205 | 232| 208 | 25878 | 197230 | 
| 76 | 20 xarray/core/variable.py | 579 | 616| 334 | 26212 | 197230 | 
| 77 | 20 xarray/core/variable.py | 2257 | 2324| 612 | 26824 | 197230 | 


### Hint

```
Similarly `str.wrap` does not pass on its `kwargs`

https://github.com/pydata/xarray/blob/7daad4fce3bf8ad9b9bc8e7baa104c476437e68d/xarray/core/accessor_str.py#L654
```

## Patch

```diff
diff --git a/xarray/core/accessor_str.py b/xarray/core/accessor_str.py
--- a/xarray/core/accessor_str.py
+++ b/xarray/core/accessor_str.py
@@ -90,7 +90,7 @@ def _apply(self, f, dtype=None):
 
     def len(self):
         """
-        Compute the length of each element in the array.
+        Compute the length of each string in the array.
 
         Returns
         -------
@@ -104,9 +104,9 @@ def __getitem__(self, key):
         else:
             return self.get(key)
 
-    def get(self, i):
+    def get(self, i, default=""):
         """
-        Extract element from indexable in each element in the array.
+        Extract character number `i` from each string in the array.
 
         Parameters
         ----------
@@ -120,12 +120,18 @@ def get(self, i):
         -------
         items : array of objects
         """
-        obj = slice(-1, None) if i == -1 else slice(i, i + 1)
-        return self._apply(lambda x: x[obj])
+        s = slice(-1, None) if i == -1 else slice(i, i + 1)
+
+        def f(x):
+            item = x[s]
+
+            return item if item else default
+
+        return self._apply(f)
 
     def slice(self, start=None, stop=None, step=None):
         """
-        Slice substrings from each element in the array.
+        Slice substrings from each string in the array.
 
         Parameters
         ----------
@@ -359,7 +365,7 @@ def count(self, pat, flags=0):
 
     def startswith(self, pat):
         """
-        Test if the start of each string element matches a pattern.
+        Test if the start of each string in the array matches a pattern.
 
         Parameters
         ----------
@@ -378,7 +384,7 @@ def startswith(self, pat):
 
     def endswith(self, pat):
         """
-        Test if the end of each string element matches a pattern.
+        Test if the end of each string in the array matches a pattern.
 
         Parameters
         ----------
@@ -432,8 +438,7 @@ def pad(self, width, side="left", fillchar=" "):
 
     def center(self, width, fillchar=" "):
         """
-        Filling left and right side of strings in the array with an
-        additional character.
+        Pad left and right side of each string in the array.
 
         Parameters
         ----------
@@ -451,8 +456,7 @@ def center(self, width, fillchar=" "):
 
     def ljust(self, width, fillchar=" "):
         """
-        Filling right side of strings in the array with an additional
-        character.
+        Pad right side of each string in the array.
 
         Parameters
         ----------
@@ -470,7 +474,7 @@ def ljust(self, width, fillchar=" "):
 
     def rjust(self, width, fillchar=" "):
         """
-        Filling left side of strings in the array with an additional character.
+        Pad left side of each string in the array.
 
         Parameters
         ----------
@@ -488,7 +492,7 @@ def rjust(self, width, fillchar=" "):
 
     def zfill(self, width):
         """
-        Pad strings in the array by prepending '0' characters.
+        Pad each string in the array by prepending '0' characters.
 
         Strings in the array are padded with '0' characters on the
         left of the string to reach a total string length  `width`. Strings
@@ -508,7 +512,7 @@ def zfill(self, width):
 
     def contains(self, pat, case=True, flags=0, regex=True):
         """
-        Test if pattern or regex is contained within a string of the array.
+        Test if pattern or regex is contained within each string of the array.
 
         Return boolean array based on whether a given pattern or regex is
         contained within a string of the array.
@@ -554,7 +558,7 @@ def contains(self, pat, case=True, flags=0, regex=True):
 
     def match(self, pat, case=True, flags=0):
         """
-        Determine if each string matches a regular expression.
+        Determine if each string in the array matches a regular expression.
 
         Parameters
         ----------
@@ -613,7 +617,7 @@ def strip(self, to_strip=None, side="both"):
 
     def lstrip(self, to_strip=None):
         """
-        Remove leading and trailing characters.
+        Remove leading characters.
 
         Strip whitespaces (including newlines) or a set of specified characters
         from each string in the array from the left side.
@@ -633,7 +637,7 @@ def lstrip(self, to_strip=None):
 
     def rstrip(self, to_strip=None):
         """
-        Remove leading and trailing characters.
+        Remove trailing characters.
 
         Strip whitespaces (including newlines) or a set of specified characters
         from each string in the array from the right side.
@@ -653,8 +657,7 @@ def rstrip(self, to_strip=None):
 
     def wrap(self, width, **kwargs):
         """
-        Wrap long strings in the array to be formatted in paragraphs with
-        length less than a given width.
+        Wrap long strings in the array in paragraphs with length less than `width`.
 
         This method has the same keyword parameters and defaults as
         :class:`textwrap.TextWrapper`.
@@ -663,38 +666,20 @@ def wrap(self, width, **kwargs):
         ----------
         width : int
             Maximum line-width
-        expand_tabs : bool, optional
-            If true, tab characters will be expanded to spaces (default: True)
-        replace_whitespace : bool, optional
-            If true, each whitespace character (as defined by
-            string.whitespace) remaining after tab expansion will be replaced
-            by a single space (default: True)
-        drop_whitespace : bool, optional
-            If true, whitespace that, after wrapping, happens to end up at the
-            beginning or end of a line is dropped (default: True)
-        break_long_words : bool, optional
-            If true, then words longer than width will be broken in order to
-            ensure that no lines are longer than width. If it is false, long
-            words will not be broken, and some lines may be longer than width.
-            (default: True)
-        break_on_hyphens : bool, optional
-            If true, wrapping will occur preferably on whitespace and right
-            after hyphens in compound words, as it is customary in English. If
-            false, only whitespaces will be considered as potentially good
-            places for line breaks, but you need to set break_long_words to
-            false if you want truly insecable words. (default: True)
+        **kwargs
+            keyword arguments passed into :class:`textwrap.TextWrapper`.
 
         Returns
         -------
         wrapped : same type as values
         """
-        tw = textwrap.TextWrapper(width=width)
+        tw = textwrap.TextWrapper(width=width, **kwargs)
         f = lambda x: "\n".join(tw.wrap(x))
         return self._apply(f)
 
     def translate(self, table):
         """
-        Map all characters in the string through the given mapping table.
+        Map characters of each string through the given mapping table.
 
         Parameters
         ----------

```

## Test Patch

```diff
diff --git a/xarray/tests/test_accessor_str.py b/xarray/tests/test_accessor_str.py
--- a/xarray/tests/test_accessor_str.py
+++ b/xarray/tests/test_accessor_str.py
@@ -596,7 +596,7 @@ def test_wrap():
     )
 
     # expected values
-    xp = xr.DataArray(
+    expected = xr.DataArray(
         [
             "hello world",
             "hello world!",
@@ -610,15 +610,29 @@ def test_wrap():
         ]
     )
 
-    rs = values.str.wrap(12, break_long_words=True)
-    assert_equal(rs, xp)
+    result = values.str.wrap(12, break_long_words=True)
+    assert_equal(result, expected)
 
     # test with pre and post whitespace (non-unicode), NaN, and non-ascii
     # Unicode
     values = xr.DataArray(["  pre  ", "\xac\u20ac\U00008000 abadcafe"])
-    xp = xr.DataArray(["  pre", "\xac\u20ac\U00008000 ab\nadcafe"])
-    rs = values.str.wrap(6)
-    assert_equal(rs, xp)
+    expected = xr.DataArray(["  pre", "\xac\u20ac\U00008000 ab\nadcafe"])
+    result = values.str.wrap(6)
+    assert_equal(result, expected)
+
+
+def test_wrap_kwargs_passed():
+    # GH4334
+
+    values = xr.DataArray("  hello world  ")
+
+    result = values.str.wrap(7)
+    expected = xr.DataArray("  hello\nworld")
+    assert_equal(result, expected)
+
+    result = values.str.wrap(7, drop_whitespace=False)
+    expected = xr.DataArray("  hello\n world\n  ")
+    assert_equal(result, expected)
 
 
 def test_get(dtype):
@@ -642,6 +656,15 @@ def test_get(dtype):
     assert_equal(result, expected)
 
 
+def test_get_default(dtype):
+    # GH4334
+    values = xr.DataArray(["a_b", "c", ""]).astype(dtype)
+
+    result = values.str.get(2, "default")
+    expected = xr.DataArray(["b", "default", "default"]).astype(dtype)
+    assert_equal(result, expected)
+
+
 def test_encode_decode():
     data = xr.DataArray(["a", "b", "a\xe4"])
     encoded = data.str.encode("utf-8")

```


## Code snippets

### 1 - xarray/core/dataarray.py:

Start line: 361, End line: 402

```python
class DataArray(AbstractArray, DataWithCoords):

    def _replace(
        self,
        variable: Variable = None,
        coords=None,
        name: Union[Hashable, None, Default] = _default,
        indexes=None,
    ) -> "DataArray":
        if variable is None:
            variable = self.variable
        if coords is None:
            coords = self._coords
        if name is _default:
            name = self.name
        return type(self)(variable, coords, name=name, fastpath=True, indexes=indexes)

    def _replace_maybe_drop_dims(
        self, variable: Variable, name: Union[Hashable, None, Default] = _default
    ) -> "DataArray":
        if variable.dims == self.dims and variable.shape == self.shape:
            coords = self._coords.copy()
            indexes = self._indexes
        elif variable.dims == self.dims:
            # Shape has changed (e.g. from reduce(..., keepdims=True)
            new_sizes = dict(zip(self.dims, variable.shape))
            coords = {
                k: v
                for k, v in self._coords.items()
                if v.shape == tuple(new_sizes[d] for d in v.dims)
            }
            changed_dims = [
                k for k in variable.dims if variable.sizes[k] != self.sizes[k]
            ]
            indexes = propagate_indexes(self._indexes, exclude=changed_dims)
        else:
            allowed_dims = set(variable.dims)
            coords = {
                k: v for k, v in self._coords.items() if set(v.dims) <= allowed_dims
            }
            indexes = propagate_indexes(
                self._indexes, exclude=(set(self.dims) - allowed_dims)
            )
        return self._replace(variable, coords, name, indexes=indexes)
```
### 2 - xarray/conventions.py:

Start line: 106, End line: 127

```python
def maybe_default_fill_value(var):
    # make NaN the fill value for float types:
    if (
        "_FillValue" not in var.attrs
        and "_FillValue" not in var.encoding
        and np.issubdtype(var.dtype, np.floating)
    ):
        var.attrs["_FillValue"] = var.dtype.type(np.nan)
    return var


def maybe_encode_bools(var):
    if (
        (var.dtype == bool)
        and ("dtype" not in var.encoding)
        and ("dtype" not in var.attrs)
    ):
        dims, data, attrs, encoding = _var_as_tuple(var)
        attrs["dtype"] = "bool"
        data = data.astype(dtype="i1", copy=True)
        var = Variable(dims, data, attrs, encoding)
    return var
```
### 3 - xarray/core/dataarray.py:

Start line: 2109, End line: 2134

```python
class DataArray(AbstractArray, DataWithCoords):

    def fillna(self, value: Any) -> "DataArray":
        """Fill missing values in this object.

        This operation follows the normal broadcasting and alignment rules that
        xarray uses for binary arithmetic, except the result is aligned to this
        object (``join='left'``) instead of aligned to the intersection of
        index coordinates (``join='inner'``).

        Parameters
        ----------
        value : scalar, ndarray or DataArray
            Used to fill all matching missing values in this array. If the
            argument is a DataArray, it is first aligned with (reindexed to)
            this array.

        Returns
        -------
        DataArray
        """
        if utils.is_dict_like(value):
            raise TypeError(
                "cannot provide fill value as a dictionary with "
                "fillna on a DataArray"
            )
        out = ops.fillna(self, value)
        return out
```
### 4 - xarray/core/dataarray.py:

Start line: 3485, End line: 3652

```python
class DataArray(AbstractArray, DataWithCoords):

    def pad(
        self,
        pad_width: Mapping[Hashable, Union[int, Tuple[int, int]]] = None,
        mode: str = "constant",
        stat_length: Union[
            int, Tuple[int, int], Mapping[Hashable, Tuple[int, int]]
        ] = None,
        constant_values: Union[
            int, Tuple[int, int], Mapping[Hashable, Tuple[int, int]]
        ] = None,
        end_values: Union[
            int, Tuple[int, int], Mapping[Hashable, Tuple[int, int]]
        ] = None,
        reflect_type: str = None,
        **pad_width_kwargs: Any,
    ) -> "DataArray":
        # ... other code
```
### 5 - xarray/core/dataarray.py:

Start line: 265, End line: 359

```python
class DataArray(AbstractArray, DataWithCoords):

    def __init__(
        self,
        data: Any = dtypes.NA,
        coords: Union[Sequence[Tuple], Mapping[Hashable, Any], None] = None,
        dims: Union[Hashable, Sequence[Hashable], None] = None,
        name: Hashable = None,
        attrs: Mapping = None,
        # internal parameters
        indexes: Dict[Hashable, pd.Index] = None,
        fastpath: bool = False,
    ):
        """
        Parameters
        ----------
        data : array_like
            Values for this array. Must be an ``numpy.ndarray``, ndarray like,
            or castable to an ``ndarray``. If a self-described xarray or pandas
            object, attempts are made to use this array's metadata to fill in
            other unspecified arguments. A view of the array's data is used
            instead of a copy if possible.
        coords : sequence or dict of array_like objects, optional
            Coordinates (tick labels) to use for indexing along each dimension.
            The following notations are accepted:

            - mapping {dimension name: array-like}
            - sequence of tuples that are valid arguments for xarray.Variable()
              - (dims, data)
              - (dims, data, attrs)
              - (dims, data, attrs, encoding)

            Additionally, it is possible to define a coord whose name
            does not match the dimension name, or a coord based on multiple
            dimensions, with one of the following notations:

            - mapping {coord name: DataArray}
            - mapping {coord name: Variable}
            - mapping {coord name: (dimension name, array-like)}
            - mapping {coord name: (tuple of dimension names, array-like)}

        dims : hashable or sequence of hashable, optional
            Name(s) of the data dimension(s). Must be either a hashable (only
            for 1D data) or a sequence of hashables with length equal to the
            number of dimensions. If this argument is omitted, dimension names
            default to ``['dim_0', ... 'dim_n']``.
        name : str or None, optional
            Name of this array.
        attrs : dict_like or None, optional
            Attributes to assign to the new instance. By default, an empty
            attribute dictionary is initialized.
        """
        if fastpath:
            variable = data
            assert dims is None
            assert attrs is None
        else:
            # try to fill in arguments from data if they weren't supplied
            if coords is None:

                if isinstance(data, DataArray):
                    coords = data.coords
                elif isinstance(data, pd.Series):
                    coords = [data.index]
                elif isinstance(data, pd.DataFrame):
                    coords = [data.index, data.columns]
                elif isinstance(data, (pd.Index, IndexVariable)):
                    coords = [data]
                elif isinstance(data, pdcompat.Panel):
                    coords = [data.items, data.major_axis, data.minor_axis]

            if dims is None:
                dims = getattr(data, "dims", getattr(coords, "dims", None))
            if name is None:
                name = getattr(data, "name", None)
            if attrs is None and not isinstance(data, PANDAS_TYPES):
                attrs = getattr(data, "attrs", None)

            data = _check_data_shape(data, coords, dims)
            data = as_compatible_data(data)
            coords, dims = _infer_coords_and_dims(data.shape, coords, dims)
            variable = Variable(dims, data, attrs, fastpath=True)
            indexes = dict(
                _extract_indexes_from_coords(coords)
            )  # needed for to_dataset

        # These fully describe a DataArray
        self._variable = variable
        assert isinstance(coords, dict)
        self._coords = coords
        self._name = name

        # TODO(shoyer): document this argument, once it becomes part of the
        # public interface.
        self._indexes = indexes

        self._file_obj = None
```
### 6 - xarray/core/dataarray.py:

Start line: 404, End line: 430

```python
class DataArray(AbstractArray, DataWithCoords):

    def _overwrite_indexes(self, indexes: Mapping[Hashable, Any]) -> "DataArray":
        if not len(indexes):
            return self
        coords = self._coords.copy()
        for name, idx in indexes.items():
            coords[name] = IndexVariable(name, idx)
        obj = self._replace(coords=coords)

        # switch from dimension to level names, if necessary
        dim_names: Dict[Any, str] = {}
        for dim, idx in indexes.items():
            if not isinstance(idx, pd.MultiIndex) and idx.name != dim:
                dim_names[dim] = idx.name
        if dim_names:
            obj = obj.rename(dim_names)
        return obj

    def _to_temp_dataset(self) -> Dataset:
        return self._to_dataset_whole(name=_THIS_ARRAY, shallow_copy=False)

    def _from_temp_dataset(
        self, dataset: Dataset, name: Hashable = _default
    ) -> "DataArray":
        variable = dataset._variables.pop(_THIS_ARRAY)
        coords = dataset._variables
        indexes = dataset._indexes
        return self._replace(variable, coords, name, indexes=indexes)
```
### 7 - xarray/core/dataarray.py:

Start line: 2254, End line: 2276

```python
class DataArray(AbstractArray, DataWithCoords):

    def bfill(self, dim: Hashable, limit: int = None) -> "DataArray":
        """Fill NaN values by propogating values backward

        *Requires bottleneck.*

        Parameters
        ----------
        dim : str
            Specifies the dimension along which to propagate values when
            filling.
        limit : int, default None
            The maximum number of consecutive NaN values to backward fill. In
            other words, if there is a gap with more than this number of
            consecutive NaNs, it will only be partially filled. Must be greater
            than 0 or None for no limit.

        Returns
        -------
        DataArray
        """
        from .missing import bfill

        return bfill(self, dim, limit=limit)
```
### 8 - xarray/core/dataarray.py:

Start line: 1005, End line: 1066

```python
class DataArray(AbstractArray, DataWithCoords):

    def isel(
        self,
        indexers: Mapping[Hashable, Any] = None,
        drop: bool = False,
        missing_dims: str = "raise",
        **indexers_kwargs: Any,
    ) -> "DataArray":
        """Return a new DataArray whose data is given by integer indexing
        along the specified dimension(s).

        Parameters
        ----------
        indexers : dict, optional
            A dict with keys matching dimensions and values given
            by integers, slice objects or arrays.
            indexer can be a integer, slice, array-like or DataArray.
            If DataArrays are passed as indexers, xarray-style indexing will be
            carried out. See :ref:`indexing` for the details.
            One of indexers or indexers_kwargs must be provided.
        drop : bool, optional
            If ``drop=True``, drop coordinates variables indexed by integers
            instead of making them scalar.
        missing_dims : {"raise", "warn", "ignore"}, default "raise"
            What to do if dimensions that should be selected from are not present in the
            DataArray:
            - "raise": raise an exception
            - "warning": raise a warning, and ignore the missing dimensions
            - "ignore": ignore the missing dimensions
        **indexers_kwargs : {dim: indexer, ...}, optional
            The keyword arguments form of ``indexers``.

        See Also
        --------
        Dataset.isel
        DataArray.sel
        """

        indexers = either_dict_or_kwargs(indexers, indexers_kwargs, "isel")

        if any(is_fancy_indexer(idx) for idx in indexers.values()):
            ds = self._to_temp_dataset()._isel_fancy(
                indexers, drop=drop, missing_dims=missing_dims
            )
            return self._from_temp_dataset(ds)

        # Much faster algorithm for when all indexers are ints, slices, one-dimensional
        # lists, or zero or one-dimensional np.ndarray's

        variable = self._variable.isel(indexers, missing_dims=missing_dims)

        coords = {}
        for coord_name, coord_value in self._coords.items():
            coord_indexers = {
                k: v for k, v in indexers.items() if k in coord_value.dims
            }
            if coord_indexers:
                coord_value = coord_value.isel(coord_indexers)
                if drop and coord_value.ndim == 0:
                    continue
            coords[coord_name] = coord_value

        return self._replace(variable=variable, coords=coords)
```
### 9 - xarray/core/dataarray.py:

Start line: 645, End line: 660

```python
class DataArray(AbstractArray, DataWithCoords):

    def __setitem__(self, key: Any, value: Any) -> None:
        if isinstance(key, str):
            self.coords[key] = value
        else:
            # Coordinates in key, value and self[key] should be consistent.
            # TODO Coordinate consistency in key is checked here, but it
            # causes unnecessary indexing. It should be optimized.
            obj = self[key]
            if isinstance(value, DataArray):
                assert_coordinate_consistent(value, obj.coords.variables)
            # DataArray key -> Variable key
            key = {
                k: v.variable if isinstance(v, DataArray) else v
                for k, v in self._item_key_to_dict(key).items()
            }
            self.variable[key] = value
```
### 10 - xarray/core/dataarray.py:

Start line: 2230, End line: 2252

```python
class DataArray(AbstractArray, DataWithCoords):

    def ffill(self, dim: Hashable, limit: int = None) -> "DataArray":
        """Fill NaN values by propogating values forward

        *Requires bottleneck.*

        Parameters
        ----------
        dim : hashable
            Specifies the dimension along which to propagate values when
            filling.
        limit : int, default None
            The maximum number of consecutive NaN values to forward fill. In
            other words, if there is a gap with more than this number of
            consecutive NaNs, it will only be partially filled. Must be greater
            than 0 or None for no limit.

        Returns
        -------
        DataArray
        """
        from .missing import ffill

        return ffill(self, dim, limit=limit)
```
### 38 - xarray/core/accessor_str.py:

Start line: 471, End line: 487

```python
class StringAccessor:

    def rjust(self, width, fillchar=" "):
        """
        Filling left side of strings in the array with an additional character.

        Parameters
        ----------
        width : int
            Minimum width of resulting string; additional characters will be
            filled with ``fillchar``
        fillchar : str
            Additional character for filling, default is whitespace

        Returns
        -------
        filled : same type as values
        """
        return self.pad(width, side="left", fillchar=fillchar)
```
### 43 - xarray/core/accessor_str.py:

Start line: 452, End line: 469

```python
class StringAccessor:

    def ljust(self, width, fillchar=" "):
        """
        Filling right side of strings in the array with an additional
        character.

        Parameters
        ----------
        width : int
            Minimum width of resulting string; additional characters will be
            filled with ``fillchar``
        fillchar : str
            Additional character for filling, default is whitespace

        Returns
        -------
        filled : same type as values
        """
        return self.pad(width, side="right", fillchar=fillchar)
```
### 50 - xarray/core/accessor_str.py:

Start line: 489, End line: 507

```python
class StringAccessor:

    def zfill(self, width):
        """
        Pad strings in the array by prepending '0' characters.

        Strings in the array are padded with '0' characters on the
        left of the string to reach a total string length  `width`. Strings
        in the array  with length greater or equal to `width` are unchanged.

        Parameters
        ----------
        width : int
            Minimum length of resulting string; strings with length less
            than `width` be prepended with '0' characters.

        Returns
        -------
        filled : same type as values
        """
        return self.pad(width, side="left", fillchar="0")
```
### 53 - xarray/core/accessor_str.py:

Start line: 433, End line: 450

```python
class StringAccessor:

    def center(self, width, fillchar=" "):
        """
        Filling left and right side of strings in the array with an
        additional character.

        Parameters
        ----------
        width : int
            Minimum width of resulting string; additional characters will be
            filled with ``fillchar``
        fillchar : str
            Additional character for filling, default is whitespace

        Returns
        -------
        filled : same type as values
        """
        return self.pad(width, side="both", fillchar=fillchar)
```
### 58 - xarray/core/accessor_str.py:

Start line: 64, End line: 105

```python
class StringAccessor:
    """Vectorized string functions for string-like arrays.

    Similar to pandas, fields can be accessed through the `.str` attribute
    for applicable DataArrays.

        >>> da = xr.DataArray(["some", "text", "in", "an", "array"])
        >>> ds.str.len()
        <xarray.DataArray (dim_0: 5)>
        array([4, 4, 2, 2, 5])
        Dimensions without coordinates: dim_0

    """

    __slots__ = ("_obj",)

    def __init__(self, obj):
        self._obj = obj

    def _apply(self, f, dtype=None):
        # TODO handling of na values ?
        if dtype is None:
            dtype = self._obj.dtype

        g = np.vectorize(f, otypes=[dtype])
        return apply_ufunc(g, self._obj, dask="parallelized", output_dtypes=[dtype])

    def len(self):
        """
        Compute the length of each element in the array.

        Returns
        -------
        lengths array : array of int
        """
        return self._apply(len, dtype=int)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return self.slice(start=key.start, stop=key.stop, step=key.step)
        else:
            return self.get(key)
```
### 59 - xarray/core/accessor_str.py:

Start line: 107, End line: 124

```python
class StringAccessor:

    def get(self, i):
        """
        Extract element from indexable in each element in the array.

        Parameters
        ----------
        i : int
            Position of element to extract.
        default : optional
            Value for out-of-range index. If not specified (None) defaults to
            an empty string.

        Returns
        -------
        items : array of objects
        """
        obj = slice(-1, None) if i == -1 else slice(i, i + 1)
        return self._apply(lambda x: x[obj])
```
### 66 - xarray/core/accessor_str.py:

Start line: 398, End line: 431

```python
class StringAccessor:

    def pad(self, width, side="left", fillchar=" "):
        """
        Pad strings in the array up to width.

        Parameters
        ----------
        width : int
            Minimum width of resulting string; additional characters will be
            filled with character defined in `fillchar`.
        side : {'left', 'right', 'both'}, default 'left'
            Side from which to fill resulting string.
        fillchar : str, default ' '
            Additional character for filling, default is whitespace.

        Returns
        -------
        filled : same type as values
            Array with a minimum number of char in each element.
        """
        width = int(width)
        fillchar = self._obj.dtype.type(fillchar)
        if len(fillchar) != 1:
            raise TypeError("fillchar must be a character, not str")

        if side == "left":
            f = lambda s: s.rjust(width, fillchar)
        elif side == "right":
            f = lambda s: s.ljust(width, fillchar)
        elif side == "both":
            f = lambda s: s.center(width, fillchar)
        else:  # pragma: no cover
            raise ValueError("Invalid side")

        return self._apply(f)
```
