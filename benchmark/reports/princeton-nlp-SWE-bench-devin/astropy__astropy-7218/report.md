# astropy__astropy-7218

| **astropy/astropy** | `9626265d77b8a21c113615c08bc6782deb52eaed` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 10681 |
| **Any found context length** | 10681 |
| **Avg pos** | 24.0 |
| **Min pos** | 24 |
| **Max pos** | 24 |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/astropy/io/fits/hdu/hdulist.py b/astropy/io/fits/hdu/hdulist.py
--- a/astropy/io/fits/hdu/hdulist.py
+++ b/astropy/io/fits/hdu/hdulist.py
@@ -510,6 +510,25 @@ def fileinfo(self, index):
 
         return output
 
+    def __copy__(self):
+        """
+        Return a shallow copy of an HDUList.
+
+        Returns
+        -------
+        copy : `HDUList`
+            A shallow copy of this `HDUList` object.
+
+        """
+
+        return self[:]
+
+    # Syntactic sugar for `__copy__()` magic method
+    copy = __copy__
+
+    def __deepcopy__(self, memo=None):
+        return HDUList([hdu.copy() for hdu in self])
+
     def pop(self, index=-1):
         """ Remove an item from the list and return it.
 

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| astropy/io/fits/hdu/hdulist.py | 513 | 513 | 24 | 1 | 10681


## Problem Statement

```
HDUList.copy() returns a list
Currently ``HDUList.copy()`` returns a list rather than an ``HDUList``:

\`\`\`python
In [1]: from astropy.io.fits import HDUList

In [2]: hdulist = HDUList()

In [3]: hdulist.copy()
Out[3]: []

In [4]: type(_)
Out[4]: list
\`\`\`

This is with Python 3.6.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 astropy/io/fits/hdu/hdulist.py** | 386 | 400| 157 | 157 | 11238 | 
| 2 | **1 astropy/io/fits/hdu/hdulist.py** | 844 | 906| 614 | 771 | 11238 | 
| 3 | **1 astropy/io/fits/hdu/hdulist.py** | 610 | 655| 408 | 1179 | 11238 | 
| 4 | **1 astropy/io/fits/hdu/hdulist.py** | 167 | 262| 758 | 1937 | 11238 | 
| 5 | **1 astropy/io/fits/hdu/hdulist.py** | 1001 | 1076| 622 | 2559 | 11238 | 
| 6 | **1 astropy/io/fits/hdu/hdulist.py** | 264 | 321| 511 | 3070 | 11238 | 
| 7 | **1 astropy/io/fits/hdu/hdulist.py** | 402 | 444| 385 | 3455 | 11238 | 
| 8 | **1 astropy/io/fits/hdu/hdulist.py** | 747 | 810| 550 | 4005 | 11238 | 
| 9 | **1 astropy/io/fits/hdu/hdulist.py** | 360 | 384| 174 | 4179 | 11238 | 
| 10 | **1 astropy/io/fits/hdu/hdulist.py** | 1239 | 1346| 1019 | 5198 | 11238 | 
| 11 | 2 astropy/io/fits/hdu/nonstandard.py | 57 | 125| 536 | 5734 | 12191 | 
| 12 | 3 astropy/nddata/ccddata.py | 225 | 293| 737 | 6471 | 17153 | 
| 13 | **3 astropy/io/fits/hdu/hdulist.py** | 1164 | 1210| 444 | 6915 | 17153 | 
| 14 | **3 astropy/io/fits/hdu/hdulist.py** | 323 | 358| 270 | 7185 | 17153 | 
| 15 | **3 astropy/io/fits/hdu/hdulist.py** | 1212 | 1237| 217 | 7402 | 17153 | 
| 16 | **3 astropy/io/fits/hdu/hdulist.py** | 1078 | 1162| 639 | 8041 | 17153 | 
| 17 | **3 astropy/io/fits/hdu/hdulist.py** | 714 | 745| 211 | 8252 | 17153 | 
| 18 | 4 astropy/io/fits/hdu/base.py | 631 | 642| 139 | 8391 | 30184 | 
| 19 | **4 astropy/io/fits/hdu/hdulist.py** | 545 | 608| 571 | 8962 | 30184 | 
| 20 | 4 astropy/io/fits/hdu/base.py | 924 | 969| 341 | 9303 | 30184 | 
| 21 | 5 astropy/io/fits/hdu/__init__.py | 3 | 17| 195 | 9498 | 30398 | 
| 22 | 6 astropy/io/fits/hdu/groups.py | 449 | 499| 402 | 9900 | 35352 | 
| 23 | **6 astropy/io/fits/hdu/hdulist.py** | 446 | 511| 477 | 10377 | 35352 | 
| **-> 24 <-** | **6 astropy/io/fits/hdu/hdulist.py** | 513 | 543| 304 | 10681 | 35352 | 
| 25 | 6 astropy/io/fits/hdu/base.py | 644 | 716| 720 | 11401 | 35352 | 
| 26 | **6 astropy/io/fits/hdu/hdulist.py** | 657 | 712| 444 | 11845 | 35352 | 
| 27 | 7 examples/io/create-mef.py | 1 | 55| 270 | 12115 | 35683 | 
| 28 | **7 astropy/io/fits/hdu/hdulist.py** | 940 | 999| 427 | 12542 | 35683 | 
| 29 | **7 astropy/io/fits/hdu/hdulist.py** | 1348 | 1390| 271 | 12813 | 35683 | 
| 30 | 7 astropy/io/fits/hdu/base.py | 335 | 373| 393 | 13206 | 35683 | 
| 31 | **7 astropy/io/fits/hdu/hdulist.py** | 812 | 842| 230 | 13436 | 35683 | 
| 32 | **7 astropy/io/fits/hdu/hdulist.py** | 28 | 141| 1091 | 14527 | 35683 | 
| 33 | 7 astropy/io/fits/hdu/base.py | 563 | 599| 354 | 14881 | 35683 | 
| 34 | 7 astropy/io/fits/hdu/base.py | 462 | 487| 211 | 15092 | 35683 | 
| 35 | 7 astropy/io/fits/hdu/base.py | 121 | 151| 266 | 15358 | 35683 | 
| 36 | **7 astropy/io/fits/hdu/hdulist.py** | 4 | 25| 165 | 15523 | 35683 | 
| 37 | **7 astropy/io/fits/hdu/hdulist.py** | 143 | 164| 197 | 15720 | 35683 | 
| 38 | 7 astropy/io/fits/hdu/base.py | 153 | 256| 806 | 16526 | 35683 | 
| 39 | 8 astropy/wcs/wcs.py | 2385 | 2426| 317 | 16843 | 64082 | 
| 40 | 8 astropy/io/fits/hdu/groups.py | 306 | 343| 366 | 17209 | 64082 | 
| 41 | 8 astropy/io/fits/hdu/base.py | 297 | 333| 366 | 17575 | 64082 | 
| 42 | 8 astropy/io/fits/hdu/base.py | 258 | 295| 364 | 17939 | 64082 | 
| 43 | 8 astropy/io/fits/hdu/nonstandard.py | 3 | 55| 409 | 18348 | 64082 | 
| 44 | 9 astropy/io/fits/hdu/table.py | 414 | 468| 467 | 18815 | 77173 | 
| 45 | 9 astropy/io/fits/hdu/base.py | 767 | 782| 176 | 18991 | 77173 | 
| 46 | 9 astropy/io/fits/hdu/groups.py | 569 | 600| 231 | 19222 | 77173 | 
| 47 | 10 astropy/io/fits/convenience.py | 1016 | 1030| 152 | 19374 | 86299 | 
| 48 | 10 astropy/io/fits/hdu/base.py | 537 | 561| 195 | 19569 | 86299 | 
| 49 | 10 astropy/io/fits/hdu/table.py | 470 | 505| 324 | 19893 | 86299 | 
| 50 | 10 astropy/io/fits/hdu/base.py | 1506 | 1543| 268 | 20161 | 86299 | 
| 51 | 10 astropy/io/fits/hdu/table.py | 187 | 208| 200 | 20361 | 86299 | 
| 52 | 11 astropy/io/fits/hdu/image.py | 600 | 653| 411 | 20772 | 96283 | 
| 53 | 11 astropy/io/fits/hdu/base.py | 90 | 118| 328 | 21100 | 96283 | 
| 54 | 12 astropy/io/fits/hdu/compressed.py | 1584 | 1614| 248 | 21348 | 115081 | 
| 55 | 12 astropy/io/fits/hdu/compressed.py | 1365 | 1406| 377 | 21725 | 115081 | 
| 56 | 12 astropy/io/fits/hdu/compressed.py | 373 | 394| 218 | 21943 | 115081 | 
| 57 | 12 astropy/io/fits/hdu/table.py | 48 | 70| 184 | 22127 | 115081 | 
| 58 | 12 astropy/io/fits/hdu/compressed.py | 1793 | 1826| 364 | 22491 | 115081 | 
| 59 | 12 astropy/io/fits/hdu/table.py | 154 | 185| 313 | 22804 | 115081 | 
| 60 | 12 astropy/io/fits/hdu/compressed.py | 608 | 672| 747 | 23551 | 115081 | 
| 61 | 12 astropy/io/fits/hdu/table.py | 746 | 776| 313 | 23864 | 115081 | 
| 62 | 12 astropy/io/fits/hdu/image.py | 773 | 804| 268 | 24132 | 115081 | 
| 63 | 13 astropy/io/fits/hdu/streaming.py | 3 | 34| 173 | 24305 | 116809 | 
| 64 | 13 astropy/io/fits/hdu/image.py | 248 | 308| 574 | 24879 | 116809 | 
| 65 | 13 astropy/io/fits/hdu/base.py | 719 | 764| 377 | 25256 | 116809 | 
| 66 | 13 astropy/io/fits/hdu/compressed.py | 1408 | 1433| 248 | 25504 | 116809 | 
| 67 | 13 astropy/io/fits/hdu/base.py | 375 | 460| 742 | 26246 | 116809 | 
| 68 | 13 astropy/io/fits/hdu/base.py | 56 | 87| 248 | 26494 | 116809 | 
| 69 | **13 astropy/io/fits/hdu/hdulist.py** | 908 | 938| 284 | 26778 | 116809 | 
| 70 | 13 astropy/io/fits/hdu/compressed.py | 1775 | 1791| 238 | 27016 | 116809 | 
| 71 | 13 astropy/io/fits/hdu/compressed.py | 1435 | 1462| 256 | 27272 | 116809 | 
| 72 | 13 astropy/io/fits/hdu/table.py | 1406 | 1440| 306 | 27578 | 116809 | 
| 73 | 13 astropy/io/fits/hdu/table.py | 555 | 590| 281 | 27859 | 116809 | 
| 74 | 13 astropy/io/fits/hdu/base.py | 1564 | 1576| 129 | 27988 | 116809 | 
| 75 | 14 astropy/io/fits/diff.py | 465 | 497| 337 | 28325 | 128847 | 
| 76 | 14 astropy/io/fits/diff.py | 199 | 211| 140 | 28465 | 128847 | 
| 77 | 14 astropy/io/fits/hdu/table.py | 1233 | 1273| 367 | 28832 | 128847 | 
| 78 | 15 astropy/table/column.py | 260 | 302| 386 | 29218 | 139612 | 
| 79 | 15 astropy/io/fits/hdu/base.py | 601 | 629| 263 | 29481 | 139612 | 
| 80 | 15 astropy/io/fits/hdu/base.py | 815 | 860| 297 | 29778 | 139612 | 
| 81 | 15 astropy/io/fits/hdu/compressed.py | 1853 | 1863| 123 | 29901 | 139612 | 
| 82 | 15 astropy/io/fits/hdu/image.py | 176 | 204| 219 | 30120 | 139612 | 
| 83 | 15 astropy/nddata/ccddata.py | 295 | 332| 431 | 30551 | 139612 | 
| 84 | 15 astropy/io/fits/hdu/table.py | 72 | 131| 588 | 31139 | 139612 | 
| 85 | 15 astropy/io/fits/hdu/image.py | 949 | 1028| 678 | 31817 | 139612 | 
| 86 | 15 astropy/io/fits/hdu/groups.py | 252 | 304| 479 | 32296 | 139612 | 
| 87 | 16 astropy/io/misc/asdf/tags/fits/fits.py | 42 | 71| 206 | 32502 | 140212 | 
| 88 | 17 astropy/table/table.py | 2562 | 2597| 245 | 32747 | 163103 | 
| 89 | 17 astropy/io/fits/hdu/base.py | 784 | 813| 218 | 32965 | 163103 | 
| 90 | 17 astropy/io/fits/hdu/base.py | 863 | 894| 247 | 33212 | 163103 | 
| 91 | 17 astropy/io/fits/hdu/table.py | 385 | 412| 196 | 33408 | 163103 | 
| 92 | 17 astropy/io/fits/hdu/table.py | 816 | 866| 429 | 33837 | 163103 | 
| 93 | 17 astropy/io/fits/hdu/base.py | 1578 | 1613| 292 | 34129 | 163103 | 
| 94 | 17 astropy/wcs/wcs.py | 826 | 870| 564 | 34693 | 163103 | 
| 95 | 17 astropy/io/fits/hdu/compressed.py | 396 | 672| 119 | 34812 | 163103 | 
| 96 | 17 astropy/io/fits/hdu/table.py | 916 | 964| 393 | 35205 | 163103 | 
| 97 | 17 astropy/io/fits/hdu/compressed.py | 1828 | 1851| 203 | 35408 | 163103 | 
| 98 | 18 astropy/io/fits/fitsrec.py | 552 | 579| 220 | 35628 | 174614 | 
| 99 | 18 astropy/io/fits/hdu/image.py | 3 | 36| 205 | 35833 | 174614 | 
| 100 | 18 astropy/io/fits/hdu/table.py | 1045 | 1114| 574 | 36407 | 174614 | 
| 101 | 18 astropy/io/fits/hdu/image.py | 1044 | 1111| 580 | 36987 | 174614 | 
| 102 | 18 astropy/io/fits/hdu/table.py | 276 | 383| 1054 | 38041 | 174614 | 
| 103 | 18 astropy/io/fits/hdu/image.py | 461 | 564| 978 | 39019 | 174614 | 
| 104 | 18 astropy/io/fits/hdu/table.py | 133 | 152| 128 | 39147 | 174614 | 
| 105 | 18 astropy/io/fits/hdu/image.py | 566 | 598| 303 | 39450 | 174614 | 
| 106 | 18 astropy/io/fits/hdu/groups.py | 345 | 381| 314 | 39764 | 174614 | 
| 107 | 18 astropy/io/fits/diff.py | 402 | 463| 622 | 40386 | 174614 | 
| 108 | 18 astropy/io/fits/hdu/image.py | 38 | 174| 1353 | 41739 | 174614 | 
| 109 | 19 astropy/io/fits/header.py | 743 | 776| 199 | 41938 | 189985 | 
| 110 | 19 astropy/wcs/wcs.py | 535 | 563| 189 | 42127 | 189985 | 
| 111 | 19 astropy/io/fits/hdu/compressed.py | 674 | 700| 201 | 42328 | 189985 | 
| 112 | 19 astropy/io/fits/hdu/compressed.py | 1348 | 1363| 226 | 42554 | 189985 | 
| 113 | 19 astropy/io/fits/hdu/table.py | 1220 | 1231| 161 | 42715 | 189985 | 
| 114 | 19 astropy/io/fits/hdu/base.py | 896 | 922| 243 | 42958 | 189985 | 
| 115 | 19 astropy/io/fits/hdu/table.py | 507 | 536| 345 | 43303 | 189985 | 
| 116 | 19 astropy/io/fits/hdu/compressed.py | 1885 | 1906| 234 | 43537 | 189985 | 
| 117 | 19 astropy/io/fits/hdu/base.py | 1441 | 1449| 153 | 43690 | 189985 | 
| 118 | 19 astropy/io/fits/hdu/groups.py | 383 | 447| 644 | 44334 | 189985 | 
| 119 | 19 astropy/io/fits/hdu/compressed.py | 1616 | 1683| 625 | 44959 | 189985 | 
| 120 | 19 astropy/io/fits/hdu/table.py | 966 | 990| 241 | 45200 | 189985 | 
| 121 | 19 astropy/io/fits/hdu/base.py | 5 | 53| 370 | 45570 | 189985 | 
| 122 | 19 astropy/io/fits/hdu/table.py | 1393 | 1404| 163 | 45733 | 189985 | 
| 123 | 19 astropy/io/fits/convenience.py | 426 | 500| 701 | 46434 | 189985 | 
| 124 | 19 astropy/io/fits/hdu/base.py | 1397 | 1439| 363 | 46797 | 189985 | 
| 125 | 19 astropy/io/fits/diff.py | 499 | 524| 270 | 47067 | 189985 | 
| 126 | 19 astropy/io/fits/hdu/table.py | 1194 | 1219| 211 | 47278 | 189985 | 
| 127 | 19 astropy/io/fits/hdu/groups.py | 501 | 518| 213 | 47491 | 189985 | 
| 128 | 19 astropy/io/fits/convenience.py | 945 | 1013| 686 | 48177 | 189985 | 
| 129 | 19 astropy/io/fits/convenience.py | 58 | 81| 218 | 48395 | 189985 | 
| 130 | 19 astropy/io/fits/hdu/base.py | 971 | 1039| 639 | 49034 | 189985 | 
| 131 | 19 astropy/io/fits/hdu/image.py | 872 | 925| 530 | 49564 | 189985 | 
| 132 | 19 astropy/io/fits/hdu/base.py | 1194 | 1247| 482 | 50046 | 189985 | 
| 133 | 20 astropy/io/fits/scripts/fitsheader.py | 143 | 197| 410 | 50456 | 192487 | 
| 134 | 20 astropy/io/fits/hdu/compressed.py | 1464 | 1582| 1216 | 51672 | 192487 | 
| 135 | 20 astropy/io/fits/hdu/table.py | 992 | 1043| 676 | 52348 | 192487 | 
| 136 | 20 astropy/nddata/ccddata.py | 334 | 375| 347 | 52695 | 192487 | 
| 137 | 20 astropy/io/fits/hdu/compressed.py | 1266 | 1346| 808 | 53503 | 192487 | 
| 138 | 20 astropy/io/fits/hdu/compressed.py | 1208 | 1264| 620 | 54123 | 192487 | 
| 139 | 20 astropy/io/fits/hdu/image.py | 927 | 946| 201 | 54324 | 192487 | 
| 140 | 20 astropy/io/fits/hdu/image.py | 1030 | 1041| 150 | 54474 | 192487 | 
| 141 | 20 astropy/table/column.py | 68 | 107| 270 | 54744 | 192487 | 
| 142 | 21 astropy/io/fits/util.py | 824 | 844| 167 | 54911 | 199423 | 
| 143 | 21 astropy/io/fits/hdu/table.py | 210 | 234| 224 | 55135 | 199423 | 


### Hint

```
This might be related to another issue reported in #7185 where adding two `HDUList`s also produces a `list` instead of another `HDUList`.
We should be able to fix this specific case by overriding `list.copy()` method with:
\`\`\`python
class HDUList(list, _Verify):
    ...
    def copy(self):
        return self[:]
    ...
\`\`\`

And the result:
\`\`\`python
>>> type(HDUList().copy())
astropy.io.fits.hdu.hdulist.HDUList
\`\`\`
```

## Patch

```diff
diff --git a/astropy/io/fits/hdu/hdulist.py b/astropy/io/fits/hdu/hdulist.py
--- a/astropy/io/fits/hdu/hdulist.py
+++ b/astropy/io/fits/hdu/hdulist.py
@@ -510,6 +510,25 @@ def fileinfo(self, index):
 
         return output
 
+    def __copy__(self):
+        """
+        Return a shallow copy of an HDUList.
+
+        Returns
+        -------
+        copy : `HDUList`
+            A shallow copy of this `HDUList` object.
+
+        """
+
+        return self[:]
+
+    # Syntactic sugar for `__copy__()` magic method
+    copy = __copy__
+
+    def __deepcopy__(self, memo=None):
+        return HDUList([hdu.copy() for hdu in self])
+
     def pop(self, index=-1):
         """ Remove an item from the list and return it.
 

```

## Test Patch

```diff
diff --git a/astropy/io/fits/tests/test_hdulist.py b/astropy/io/fits/tests/test_hdulist.py
--- a/astropy/io/fits/tests/test_hdulist.py
+++ b/astropy/io/fits/tests/test_hdulist.py
@@ -5,6 +5,7 @@
 import os
 import platform
 import sys
+import copy
 
 import pytest
 import numpy as np
@@ -376,6 +377,43 @@ def test_file_like_3(self):
         info = [(0, 'PRIMARY', 1, 'PrimaryHDU', 5, (100,), 'int32', '')]
         assert fits.info(self.temp('tmpfile.fits'), output=False) == info
 
+    def test_shallow_copy(self):
+        """
+        Tests that `HDUList.__copy__()` and `HDUList.copy()` return a
+        shallow copy (regression test for #7211).
+        """
+
+        n = np.arange(10.0)
+        primary_hdu = fits.PrimaryHDU(n)
+        hdu = fits.ImageHDU(n)
+        hdul = fits.HDUList([primary_hdu, hdu])
+
+        for hdulcopy in (hdul.copy(), copy.copy(hdul)):
+            assert isinstance(hdulcopy, fits.HDUList)
+            assert hdulcopy is not hdul
+            assert hdulcopy[0] is hdul[0]
+            assert hdulcopy[1] is hdul[1]
+
+    def test_deep_copy(self):
+        """
+        Tests that `HDUList.__deepcopy__()` returns a deep copy.
+        """
+
+        n = np.arange(10.0)
+        primary_hdu = fits.PrimaryHDU(n)
+        hdu = fits.ImageHDU(n)
+        hdul = fits.HDUList([primary_hdu, hdu])
+
+        hdulcopy = copy.deepcopy(hdul)
+
+        assert isinstance(hdulcopy, fits.HDUList)
+        assert hdulcopy is not hdul
+
+        for index in range(len(hdul)):
+            assert hdulcopy[index] is not hdul[index]
+            assert hdulcopy[index].header == hdul[index].header
+            np.testing.assert_array_equal(hdulcopy[index].data, hdul[index].data)
+
     def test_new_hdu_extname(self):
         """
         Tests that new extension HDUs that are added to an HDUList can be

```


## Code snippets

### 1 - astropy/io/fits/hdu/hdulist.py:

Start line: 386, End line: 400

```python
class HDUList(list, _Verify):

    @classmethod
    def fromfile(cls, fileobj, mode=None, memmap=None,
                 save_backup=False, cache=True, lazy_load_hdus=True,
                 **kwargs):
        """
        Creates an `HDUList` instance from a file-like object.

        The actual implementation of ``fitsopen()``, and generally shouldn't
        be used directly.  Use :func:`open` instead (and see its
        documentation for details of the parameters accepted by this method).
        """

        return cls._readfrom(fileobj=fileobj, mode=mode, memmap=memmap,
                             save_backup=save_backup, cache=cache,
                             lazy_load_hdus=lazy_load_hdus, **kwargs)
```
### 2 - astropy/io/fits/hdu/hdulist.py:

Start line: 844, End line: 906

```python
class HDUList(list, _Verify):

    @deprecated_renamed_argument('clobber', 'overwrite', '2.0')
    def writeto(self, fileobj, output_verify='exception', overwrite=False,
                checksum=False):
        """
        Write the `HDUList` to a new file.

        Parameters
        ----------
        fileobj : file path, file object or file-like object
            File to write to.  If a file object, must be opened in a
            writeable mode.

        output_verify : str
            Output verification option.  Must be one of ``"fix"``,
            ``"silentfix"``, ``"ignore"``, ``"warn"``, or
            ``"exception"``.  May also be any combination of ``"fix"`` or
            ``"silentfix"`` with ``"+ignore"``, ``+warn``, or ``+exception"
            (e.g. ``"fix+warn"``).  See :ref:`verify` for more info.

        overwrite : bool, optional
            If ``True``, overwrite the output file if it exists. Raises an
            ``OSError`` if ``False`` and the output file exists. Default is
            ``False``.

            .. versionchanged:: 1.3
               ``overwrite`` replaces the deprecated ``clobber`` argument.

        checksum : bool
            When `True` adds both ``DATASUM`` and ``CHECKSUM`` cards
            to the headers of all HDU's written to the file.
        """

        if (len(self) == 0):
            warnings.warn("There is nothing to write.", AstropyUserWarning)
            return

        self.verify(option=output_verify)

        # make sure the EXTEND keyword is there if there is extension
        self.update_extend()

        # make note of whether the input file object is already open, in which
        # case we should not close it after writing (that should be the job
        # of the caller)
        closed = isinstance(fileobj, str) or fileobj_closed(fileobj)

        # writeto is only for writing a new file from scratch, so the most
        # sensible mode to require is 'ostream'.  This can accept an open
        # file object that's open to write only, or in append/update modes
        # but only if the file doesn't exist.
        fileobj = _File(fileobj, mode='ostream', overwrite=overwrite)
        hdulist = self.fromfile(fileobj)
        try:
            dirname = os.path.dirname(hdulist._file.name)
        except AttributeError:
            dirname = None

        with _free_space_check(self, dirname=dirname):
            for hdu in self:
                hdu._prewriteto(checksum=checksum)
                hdu._writeto(hdulist._file)
                hdu._postwriteto()
        hdulist.close(output_verify=output_verify, closed=closed)
```
### 3 - astropy/io/fits/hdu/hdulist.py:

Start line: 610, End line: 655

```python
class HDUList(list, _Verify):

    def append(self, hdu):
        """
        Append a new HDU to the `HDUList`.

        Parameters
        ----------
        hdu : HDU object
            HDU to add to the `HDUList`.
        """

        if not isinstance(hdu, _BaseHDU):
            raise ValueError('HDUList can only append an HDU.')

        if len(self) > 0:
            if isinstance(hdu, GroupsHDU):
                raise ValueError(
                    "Can't append a GroupsHDU to a non-empty HDUList")

            if isinstance(hdu, PrimaryHDU):
                # You passed a Primary HDU but we need an Extension HDU
                # so create an Extension HDU from the input Primary HDU.
                # TODO: This isn't necessarily sufficient to copy the HDU;
                # _header_offset and friends need to be copied too.
                hdu = ImageHDU(hdu.data, hdu.header)
        else:
            if not isinstance(hdu, (PrimaryHDU, _NonstandardHDU)):
                # You passed in an Extension HDU but we need a Primary
                # HDU.
                # If you provided an ImageHDU then we can convert it to
                # a primary HDU and use that.
                if isinstance(hdu, ImageHDU):
                    hdu = PrimaryHDU(hdu.data, hdu.header)
                else:
                    # You didn't provide an ImageHDU so we create a
                    # simple Primary HDU and append that first before
                    # we append the new Extension HDU.
                    phdu = PrimaryHDU()
                    super().append(phdu)

        super().append(hdu)
        hdu._new = True
        self._resize = True
        self._truncate = False

        # make sure the EXTEND keyword is in primary HDU if there is extension
        self.update_extend()
```
### 4 - astropy/io/fits/hdu/hdulist.py:

Start line: 167, End line: 262

```python
class HDUList(list, _Verify):
    """
    HDU list class.  This is the top-level FITS object.  When a FITS
    file is opened, a `HDUList` object is returned.
    """

    def __init__(self, hdus=[], file=None):
        """
        Construct a `HDUList` object.

        Parameters
        ----------
        hdus : sequence of HDU objects or single HDU, optional
            The HDU object(s) to comprise the `HDUList`.  Should be
            instances of HDU classes like `ImageHDU` or `BinTableHDU`.

        file : file object, bytes, optional
            The opened physical file associated with the `HDUList`
            or a bytes object containing the contents of the FITS
            file.
        """

        if isinstance(file, bytes):
            self._data = file
            self._file = None
        else:
            self._file = file
            self._data = None

        self._save_backup = False

        # For internal use only--the keyword args passed to fitsopen /
        # HDUList.fromfile/string when opening the file
        self._open_kwargs = {}
        self._in_read_next_hdu = False

        # If we have read all the HDUs from the file or not
        # The assumes that all HDUs have been written when we first opened the
        # file; we do not currently support loading additional HDUs from a file
        # while it is being streamed to.  In the future that might be supported
        # but for now this is only used for the purpose of lazy-loading of
        # existing HDUs.
        if file is None:
            self._read_all = True
        elif self._file is not None:
            # Should never attempt to read HDUs in ostream mode
            self._read_all = self._file.mode == 'ostream'
        else:
            self._read_all = False

        if hdus is None:
            hdus = []

        # can take one HDU, as well as a list of HDU's as input
        if isinstance(hdus, _ValidHDU):
            hdus = [hdus]
        elif not isinstance(hdus, (HDUList, list)):
            raise TypeError("Invalid input for HDUList.")

        for idx, hdu in enumerate(hdus):
            if not isinstance(hdu, _BaseHDU):
                raise TypeError("Element {} in the HDUList input is "
                                "not an HDU.".format(idx))

        super().__init__(hdus)

        if file is None:
            # Only do this when initializing from an existing list of HDUs
            # When initalizing from a file, this will be handled by the
            # append method after the first HDU is read
            self.update_extend()

    def __len__(self):
        if not self._in_read_next_hdu:
            self.readall()

        return super().__len__()

    def __repr__(self):
        # In order to correctly repr an HDUList we need to load all the
        # HDUs as well
        self.readall()

        return super().__repr__()

    def __iter__(self):
        # While effectively this does the same as:
        # for idx in range(len(self)):
        #     yield self[idx]
        # the more complicated structure is here to prevent the use of len(),
        # which would break the lazy loading
        for idx in itertools.count():
            try:
                yield self[idx]
            except IndexError:
                break
```
### 5 - astropy/io/fits/hdu/hdulist.py:

Start line: 1001, End line: 1076

```python
class HDUList(list, _Verify):

    @classmethod
    def _readfrom(cls, fileobj=None, data=None, mode=None,
                  memmap=None, save_backup=False, cache=True,
                  lazy_load_hdus=True, **kwargs):
        """
        Provides the implementations from HDUList.fromfile and
        HDUList.fromstring, both of which wrap this method, as their
        implementations are largely the same.
        """

        if fileobj is not None:
            if not isinstance(fileobj, _File):
                # instantiate a FITS file object (ffo)
                fileobj = _File(fileobj, mode=mode, memmap=memmap, cache=cache)
            # The Astropy mode is determined by the _File initializer if the
            # supplied mode was None
            mode = fileobj.mode
            hdulist = cls(file=fileobj)
        else:
            if mode is None:
                # The default mode
                mode = 'readonly'

            hdulist = cls(file=data)
            # This method is currently only called from HDUList.fromstring and
            # HDUList.fromfile.  If fileobj is None then this must be the
            # fromstring case; the data type of ``data`` will be checked in the
            # _BaseHDU.fromstring call.

        hdulist._save_backup = save_backup
        hdulist._open_kwargs = kwargs

        if fileobj is not None and fileobj.writeonly:
            # Output stream--not interested in reading/parsing
            # the HDUs--just writing to the output file
            return hdulist

        # Make sure at least the PRIMARY HDU can be read
        read_one = hdulist._read_next_hdu()

        # If we're trying to read only and no header units were found,
        # raise an exception
        if not read_one and mode in ('readonly', 'denywrite'):
            # Close the file if necessary (issue #6168)
            if hdulist._file.close_on_error:
                hdulist._file.close()

            raise OSError('Empty or corrupt FITS file')

        if not lazy_load_hdus:
            # Go ahead and load all HDUs
            while hdulist._read_next_hdu():
                pass

        # initialize/reset attributes to be used in "update/append" mode
        hdulist._resize = False
        hdulist._truncate = False

        return hdulist

    def _try_while_unread_hdus(self, func, *args, **kwargs):
        """
        Attempt an operation that accesses an HDU by index/name
        that can fail if not all HDUs have been read yet.  Keep
        reading HDUs until the operation succeeds or there are no
        more HDUs to read.
        """

        while True:
            try:
                return func(*args, **kwargs)
            except Exception:
                if self._read_next_hdu():
                    continue
                else:
                    raise
```
### 6 - astropy/io/fits/hdu/hdulist.py:

Start line: 264, End line: 321

```python
class HDUList(list, _Verify):

    def __getitem__(self, key):
        """
        Get an HDU from the `HDUList`, indexed by number or name.
        """

        # If the key is a slice we need to make sure the necessary HDUs
        # have been loaded before passing the slice on to super.
        if isinstance(key, slice):
            max_idx = key.stop
            # Check for and handle the case when no maximum was
            # specified (e.g. [1:]).
            if max_idx is None:
                # We need all of the HDUs, so load them
                # and reset the maximum to the actual length.
                max_idx = len(self)

            # Just in case the max_idx is negative...
            max_idx = self._positive_index_of(max_idx)

            number_loaded = super().__len__()

            if max_idx >= number_loaded:
                # We need more than we have, try loading up to and including
                # max_idx. Note we do not try to be clever about skipping HDUs
                # even though key.step might conceivably allow it.
                for i in range(number_loaded, max_idx):
                    # Read until max_idx or to the end of the file, whichever
                    # comes first.
                    if not self._read_next_hdu():
                        break

            try:
                hdus = super().__getitem__(key)
            except IndexError as e:
                # Raise a more helpful IndexError if the file was not fully read.
                if self._read_all:
                    raise e
                else:
                    raise IndexError('HDU not found, possibly because the index '
                                     'is out of range, or because the file was '
                                     'closed before all HDUs were read')
            else:
                return HDUList(hdus)

        # Originally this used recursion, but hypothetically an HDU with
        # a very large number of HDUs could blow the stack, so use a loop
        # instead
        try:
            return self._try_while_unread_hdus(super().__getitem__,
                                               self._positive_index_of(key))
        except IndexError as e:
            # Raise a more helpful IndexError if the file was not fully read.
            if self._read_all:
                raise e
            else:
                raise IndexError('HDU not found, possibly because the index '
                                 'is out of range, or because the file was '
                                 'closed before all HDUs were read')
```
### 7 - astropy/io/fits/hdu/hdulist.py:

Start line: 402, End line: 444

```python
class HDUList(list, _Verify):

    @classmethod
    def fromstring(cls, data, **kwargs):
        """
        Creates an `HDUList` instance from a string or other in-memory data
        buffer containing an entire FITS file.  Similar to
        :meth:`HDUList.fromfile`, but does not accept the mode or memmap
        arguments, as they are only relevant to reading from a file on disk.

        This is useful for interfacing with other libraries such as CFITSIO,
        and may also be useful for streaming applications.

        Parameters
        ----------
        data : str, buffer, memoryview, etc.
            A string or other memory buffer containing an entire FITS file.  It
            should be noted that if that memory is read-only (such as a Python
            string) the returned :class:`HDUList`'s data portions will also be
            read-only.

        kwargs : dict
            Optional keyword arguments.  See
            :func:`astropy.io.fits.open` for details.

        Returns
        -------
        hdul : HDUList
            An :class:`HDUList` object representing the in-memory FITS file.
        """

        try:
            # Test that the given object supports the buffer interface by
            # ensuring an ndarray can be created from it
            np.ndarray((), dtype='ubyte', buffer=data)
        except TypeError:
            raise TypeError(
                'The provided object {} does not contain an underlying '
                'memory buffer.  fromstring() requires an object that '
                'supports the buffer interface such as bytes, buffer, '
                'memoryview, ndarray, etc.  This restriction is to ensure '
                'that efficient access to the array/table data is possible.'
                ''.format(data))

        return cls._readfrom(data=data, **kwargs)
```
### 8 - astropy/io/fits/hdu/hdulist.py:

Start line: 747, End line: 810

```python
class HDUList(list, _Verify):

    @ignore_sigint
    def flush(self, output_verify='fix', verbose=False):
        """
        Force a write of the `HDUList` back to the file (for append and
        update modes only).

        Parameters
        ----------
        output_verify : str
            Output verification option.  Must be one of ``"fix"``,
            ``"silentfix"``, ``"ignore"``, ``"warn"``, or
            ``"exception"``.  May also be any combination of ``"fix"`` or
            ``"silentfix"`` with ``"+ignore"``, ``+warn``, or ``+exception"
            (e.g. ``"fix+warn"``).  See :ref:`verify` for more info.

        verbose : bool
            When `True`, print verbose messages
        """

        if self._file.mode not in ('append', 'update', 'ostream'):
            warnings.warn("Flush for '{}' mode is not supported."
                         .format(self._file.mode), AstropyUserWarning)
            return

        if self._save_backup and self._file.mode in ('append', 'update'):
            filename = self._file.name
            if os.path.exists(filename):
                # The the file doesn't actually exist anymore for some reason
                # then there's no point in trying to make a backup
                backup = filename + '.bak'
                idx = 1
                while os.path.exists(backup):
                    backup = filename + '.bak.' + str(idx)
                    idx += 1
                warnings.warn('Saving a backup of {} to {}.'.format(
                        filename, backup), AstropyUserWarning)
                try:
                    shutil.copy(filename, backup)
                except OSError as exc:
                    raise OSError('Failed to save backup to destination {}: '
                                  '{}'.format(filename, exc))

        self.verify(option=output_verify)

        if self._file.mode in ('append', 'ostream'):
            for hdu in self:
                if verbose:
                    try:
                        extver = str(hdu._header['extver'])
                    except KeyError:
                        extver = ''

                # only append HDU's which are "new"
                if hdu._new:
                    hdu._prewriteto(checksum=hdu._output_checksum)
                    with _free_space_check(self):
                        hdu._writeto(self._file)
                        if verbose:
                            print('append HDU', hdu.name, extver)
                        hdu._new = False
                    hdu._postwriteto()

        elif self._file.mode == 'update':
            self._flush_update()
```
### 9 - astropy/io/fits/hdu/hdulist.py:

Start line: 360, End line: 384

```python
class HDUList(list, _Verify):

    def __delitem__(self, key):
        """
        Delete an HDU from the `HDUList`, indexed by number or name.
        """

        if isinstance(key, slice):
            end_index = len(self)
        else:
            key = self._positive_index_of(key)
            end_index = len(self) - 1

        self._try_while_unread_hdus(super().__delitem__, key)

        if (key == end_index or key == -1 and not self._resize):
            self._truncate = True
        else:
            self._truncate = False
            self._resize = True

    # Support the 'with' statement
    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()
```
### 10 - astropy/io/fits/hdu/hdulist.py:

Start line: 1239, End line: 1346

```python
class HDUList(list, _Verify):

    def _flush_resize(self):
        """
        Implements flushing changes in update mode when parts of one or more HDU
        need to be resized.
        """

        old_name = self._file.name
        old_memmap = self._file.memmap
        name = _tmp_name(old_name)

        if not self._file.file_like:
            old_mode = os.stat(old_name).st_mode
            # The underlying file is an actual file object.  The HDUList is
            # resized, so we need to write it to a tmp file, delete the
            # original file, and rename the tmp file to the original file.
            if self._file.compression == 'gzip':
                new_file = gzip.GzipFile(name, mode='ab+')
            elif self._file.compression == 'bzip2':
                new_file = bz2.BZ2File(name, mode='w')
            else:
                new_file = name

            with self.fromfile(new_file, mode='append') as hdulist:

                for hdu in self:
                    hdu._writeto(hdulist._file, inplace=True, copy=True)
                if sys.platform.startswith('win'):
                    # Collect a list of open mmaps to the data; this well be
                    # used later.  See below.
                    mmaps = [(idx, _get_array_mmap(hdu.data), hdu.data)
                             for idx, hdu in enumerate(self) if hdu._has_data]

                hdulist._file.close()
                self._file.close()
            if sys.platform.startswith('win'):
                # Close all open mmaps to the data.  This is only necessary on
                # Windows, which will not allow a file to be renamed or deleted
                # until all handles to that file have been closed.
                for idx, mmap, arr in mmaps:
                    if mmap is not None:
                        mmap.close()

            os.remove(self._file.name)

            # reopen the renamed new file with "update" mode
            os.rename(name, old_name)
            os.chmod(old_name, old_mode)

            if isinstance(new_file, gzip.GzipFile):
                old_file = gzip.GzipFile(old_name, mode='rb+')
            else:
                old_file = old_name

            ffo = _File(old_file, mode='update', memmap=old_memmap)

            self._file = ffo

            for hdu in self:
                # Need to update the _file attribute and close any open mmaps
                # on each HDU
                if hdu._has_data and _get_array_mmap(hdu.data) is not None:
                    del hdu.data
                hdu._file = ffo

            if sys.platform.startswith('win'):
                # On Windows, all the original data mmaps were closed above.
                # However, it's possible that the user still has references to
                # the old data which would no longer work (possibly even cause
                # a segfault if they try to access it).  This replaces the
                # buffers used by the original arrays with the buffers of mmap
                # arrays created from the new file.  This seems to work, but
                # it's a flaming hack and carries no guarantees that it won't
                # lead to odd behavior in practice.  Better to just not keep
                # references to data from files that had to be resized upon
                # flushing (on Windows--again, this is no problem on Linux).
                for idx, mmap, arr in mmaps:
                    if mmap is not None:
                        arr.data = self[idx].data.data
                del mmaps  # Just to be sure

        else:
            # The underlying file is not a file object, it is a file like
            # object.  We can't write out to a file, we must update the file
            # like object in place.  To do this, we write out to a temporary
            # file, then delete the contents in our file like object, then
            # write the contents of the temporary file to the now empty file
            # like object.
            self.writeto(name)
            hdulist = self.fromfile(name)
            ffo = self._file

            ffo.truncate(0)
            ffo.seek(0)

            for hdu in hdulist:
                hdu._writeto(ffo, inplace=True, copy=True)

            # Close the temporary file and delete it.
            hdulist.close()
            os.remove(hdulist._file.name)

        # reset the resize attributes after updating
        self._resize = False
        self._truncate = False
        for hdu in self:
            hdu._header._modified = False
            hdu._new = False
            hdu._file = ffo
```
### 13 - astropy/io/fits/hdu/hdulist.py:

Start line: 1164, End line: 1210

```python
class HDUList(list, _Verify):

    def _verify(self, option='warn'):
        errs = _ErrList([], unit='HDU')

        # the first (0th) element must be a primary HDU
        if len(self) > 0 and (not isinstance(self[0], PrimaryHDU)) and \
                             (not isinstance(self[0], _NonstandardHDU)):
            err_text = "HDUList's 0th element is not a primary HDU."
            fix_text = 'Fixed by inserting one as 0th HDU.'

            def fix(self=self):
                self.insert(0, PrimaryHDU())

            err = self.run_option(option, err_text=err_text,
                                  fix_text=fix_text, fix=fix)
            errs.append(err)

        if len(self) > 1 and ('EXTEND' not in self[0].header or
                              self[0].header['EXTEND'] is not True):
            err_text = ('Primary HDU does not contain an EXTEND keyword '
                        'equal to T even though there are extension HDUs.')
            fix_text = 'Fixed by inserting or updating the EXTEND keyword.'

            def fix(header=self[0].header):
                naxis = header['NAXIS']
                if naxis == 0:
                    after = 'NAXIS'
                else:
                    after = 'NAXIS' + str(naxis)
                header.set('EXTEND', value=True, after=after)

            errs.append(self.run_option(option, err_text=err_text,
                                        fix_text=fix_text, fix=fix))

        # each element calls their own verify
        for idx, hdu in enumerate(self):
            if idx > 0 and (not isinstance(hdu, ExtensionHDU)):
                err_text = ("HDUList's element {} is not an "
                            "extension HDU.".format(str(idx)))

                err = self.run_option(option, err_text=err_text, fixable=False)
                errs.append(err)

            else:
                result = hdu._verify(option)
                if result:
                    errs.append(result)
        return errs
```
### 14 - astropy/io/fits/hdu/hdulist.py:

Start line: 323, End line: 358

```python
class HDUList(list, _Verify):

    def __contains__(self, item):
        """
        Returns `True` if ``HDUList.index_of(item)`` succeeds.
        """

        try:
            self._try_while_unread_hdus(self.index_of, item)
        except KeyError:
            return False

        return True

    def __setitem__(self, key, hdu):
        """
        Set an HDU to the `HDUList`, indexed by number or name.
        """

        _key = self._positive_index_of(key)
        if isinstance(hdu, (slice, list)):
            if _is_int(_key):
                raise ValueError('An element in the HDUList must be an HDU.')
            for item in hdu:
                if not isinstance(item, _BaseHDU):
                    raise ValueError('{} is not an HDU.'.format(item))
        else:
            if not isinstance(hdu, _BaseHDU):
                raise ValueError('{} is not an HDU.'.format(hdu))

        try:
            self._try_while_unread_hdus(super().__setitem__, _key, hdu)
        except IndexError:
            raise IndexError('Extension {} is out of bound or not found.'
                            .format(key))

        self._resize = True
        self._truncate = False
```
### 15 - astropy/io/fits/hdu/hdulist.py:

Start line: 1212, End line: 1237

```python
class HDUList(list, _Verify):

    def _flush_update(self):
        """Implements flushing changes to a file in update mode."""

        for hdu in self:
            # Need to all _prewriteto() for each HDU first to determine if
            # resizing will be necessary
            hdu._prewriteto(checksum=hdu._output_checksum, inplace=True)

        try:
            self._wasresized()

            # if the HDUList is resized, need to write out the entire contents of
            # the hdulist to the file.
            if self._resize or self._file.compression:
                self._flush_resize()
            else:
                # if not resized, update in place
                for hdu in self:
                    hdu._writeto(self._file, inplace=True)

            # reset the modification attributes after updating
            for hdu in self:
                hdu._header._modified = False
        finally:
            for hdu in self:
                hdu._postwriteto()
```
### 16 - astropy/io/fits/hdu/hdulist.py:

Start line: 1078, End line: 1162

```python
class HDUList(list, _Verify):

    def _read_next_hdu(self):
        """
        Lazily load a single HDU from the fileobj or data string the `HDUList`
        was opened from, unless no further HDUs are found.

        Returns True if a new HDU was loaded, or False otherwise.
        """

        if self._read_all:
            return False

        saved_compression_enabled = compressed.COMPRESSION_ENABLED
        fileobj, data, kwargs = self._file, self._data, self._open_kwargs

        if fileobj is not None and fileobj.closed:
            return False

        try:
            self._in_read_next_hdu = True

            if ('disable_image_compression' in kwargs and
                kwargs['disable_image_compression']):
                compressed.COMPRESSION_ENABLED = False

            # read all HDUs
            try:
                if fileobj is not None:
                    try:
                        # Make sure we're back to the end of the last read
                        # HDU
                        if len(self) > 0:
                            last = self[len(self) - 1]
                            if last._data_offset is not None:
                                offset = last._data_offset + last._data_size
                                fileobj.seek(offset, os.SEEK_SET)

                        hdu = _BaseHDU.readfrom(fileobj, **kwargs)
                    except EOFError:
                        self._read_all = True
                        return False
                    except OSError:
                        # Close the file: see
                        # https://github.com/astropy/astropy/issues/6168
                        #
                        if self._file.close_on_error:
                            self._file.close()

                        if fileobj.writeonly:
                            self._read_all = True
                            return False
                        else:
                            raise
                else:
                    if not data:
                        self._read_all = True
                        return False
                    hdu = _BaseHDU.fromstring(data, **kwargs)
                    self._data = data[hdu._data_offset + hdu._data_size:]

                super().append(hdu)
                if len(self) == 1:
                    # Check for an extension HDU and update the EXTEND
                    # keyword of the primary HDU accordingly
                    self.update_extend()

                hdu._new = False
                if 'checksum' in kwargs:
                    hdu._output_checksum = kwargs['checksum']
            # check in the case there is extra space after the last HDU or
            # corrupted HDU
            except (VerifyError, ValueError) as exc:
                warnings.warn(
                    'Error validating header for HDU #{} (note: Astropy '
                    'uses zero-based indexing).\n{}\n'
                    'There may be extra bytes after the last HDU or the '
                    'file is corrupted.'.format(
                        len(self), indent(str(exc))), VerifyWarning)
                del exc
                self._read_all = True
                return False
        finally:
            compressed.COMPRESSION_ENABLED = saved_compression_enabled
            self._in_read_next_hdu = False

        return True
```
### 17 - astropy/io/fits/hdu/hdulist.py:

Start line: 714, End line: 745

```python
class HDUList(list, _Verify):

    def _positive_index_of(self, key):
        """
        Same as index_of, but ensures always returning a positive index
        or zero.

        (Really this should be called non_negative_index_of but it felt
        too long.)

        This means that if the key is a negative integer, we have to
        convert it to the corresponding positive index.  This means
        knowing the length of the HDUList, which in turn means loading
        all HDUs.  Therefore using negative indices on HDULists is inherently
        inefficient.
        """

        index = self.index_of(key)

        if index >= 0:
            return index

        if abs(index) > len(self):
            raise IndexError(
                'Extension {} is out of bound or not found.'.format(index))

        return len(self) + index

    def readall(self):
        """
        Read data of all HDUs into memory.
        """
        while self._read_next_hdu():
            pass
```
### 19 - astropy/io/fits/hdu/hdulist.py:

Start line: 545, End line: 608

```python
class HDUList(list, _Verify):

    def insert(self, index, hdu):
        """
        Insert an HDU into the `HDUList` at the given ``index``.

        Parameters
        ----------
        index : int
            Index before which to insert the new HDU.

        hdu : HDU object
            The HDU object to insert
        """

        if not isinstance(hdu, _BaseHDU):
            raise ValueError('{} is not an HDU.'.format(hdu))

        num_hdus = len(self)

        if index == 0 or num_hdus == 0:
            if num_hdus != 0:
                # We are inserting a new Primary HDU so we need to
                # make the current Primary HDU into an extension HDU.
                if isinstance(self[0], GroupsHDU):
                    raise ValueError(
                        "The current Primary HDU is a GroupsHDU.  "
                        "It can't be made into an extension HDU, "
                        "so another HDU cannot be inserted before it.")

                hdu1 = ImageHDU(self[0].data, self[0].header)

                # Insert it into position 1, then delete HDU at position 0.
                super().insert(1, hdu1)
                super().__delitem__(0)

            if not isinstance(hdu, (PrimaryHDU, _NonstandardHDU)):
                # You passed in an Extension HDU but we need a Primary HDU.
                # If you provided an ImageHDU then we can convert it to
                # a primary HDU and use that.
                if isinstance(hdu, ImageHDU):
                    hdu = PrimaryHDU(hdu.data, hdu.header)
                else:
                    # You didn't provide an ImageHDU so we create a
                    # simple Primary HDU and append that first before
                    # we append the new Extension HDU.
                    phdu = PrimaryHDU()

                    super().insert(0, phdu)
                    index = 1
        else:
            if isinstance(hdu, GroupsHDU):
                raise ValueError('A GroupsHDU must be inserted as a '
                                 'Primary HDU.')

            if isinstance(hdu, PrimaryHDU):
                # You passed a Primary HDU but we need an Extension HDU
                # so create an Extension HDU from the input Primary HDU.
                hdu = ImageHDU(hdu.data, hdu.header)

        super().insert(index, hdu)
        hdu._new = True
        self._resize = True
        self._truncate = False
        # make sure the EXTEND keyword is in primary HDU if there is extension
        self.update_extend()
```
### 23 - astropy/io/fits/hdu/hdulist.py:

Start line: 446, End line: 511

```python
class HDUList(list, _Verify):

    def fileinfo(self, index):
        """
        Returns a dictionary detailing information about the locations
        of the indexed HDU within any associated file.  The values are
        only valid after a read or write of the associated file with
        no intervening changes to the `HDUList`.

        Parameters
        ----------
        index : int
            Index of HDU for which info is to be returned.

        Returns
        -------
        fileinfo : dict or None

            The dictionary details information about the locations of
            the indexed HDU within an associated file.  Returns `None`
            when the HDU is not associated with a file.

            Dictionary contents:

            ========== ========================================================
            Key        Value
            ========== ========================================================
            file       File object associated with the HDU
            filename   Name of associated file object
            filemode   Mode in which the file was opened (readonly,
                       update, append, denywrite, ostream)
            resized    Flag that when `True` indicates that the data has been
                       resized since the last read/write so the returned values
                       may not be valid.
            hdrLoc     Starting byte location of header in file
            datLoc     Starting byte location of data block in file
            datSpan    Data size including padding
            ========== ========================================================

        """

        if self._file is not None:
            output = self[index].fileinfo()

            if not output:
                # OK, the HDU associated with this index is not yet
                # tied to the file associated with the HDUList.  The only way
                # to get the file object is to check each of the HDU's in the
                # list until we find the one associated with the file.
                f = None

                for hdu in self:
                    info = hdu.fileinfo()

                    if info:
                        f = info['file']
                        fm = info['filemode']
                        break

                output = {'file': f, 'filemode': fm, 'hdrLoc': None,
                          'datLoc': None, 'datSpan': None}

            output['filename'] = self._file.name
            output['resized'] = self._wasresized()
        else:
            output = None

        return output
```
### 24 - astropy/io/fits/hdu/hdulist.py:

Start line: 513, End line: 543

```python
class HDUList(list, _Verify):

    def pop(self, index=-1):
        """ Remove an item from the list and return it.

        Parameters
        ----------
        index : int, str, tuple of (string, int), optional
            An integer value of ``index`` indicates the position from which
            ``pop()`` removes and returns an HDU. A string value or a tuple
            of ``(string, int)`` functions as a key for identifying the
            HDU to be removed and returned. If ``key`` is a tuple, it is
            of the form ``(key, ver)`` where ``ver`` is an ``EXTVER``
            value that must match the HDU being searched for.

            If the key is ambiguous (e.g. there are multiple 'SCI' extensions)
            the first match is returned.  For a more precise match use the
            ``(name, ver)`` pair.

            If even the ``(name, ver)`` pair is ambiguous the numeric index
            must be used to index the duplicate HDU.

        Returns
        -------
        hdu : HDU object
            The HDU object at position indicated by ``index`` or having name
            and version specified by ``index``.
        """

        # Make sure that HDUs are loaded before attempting to pop
        self.readall()
        list_index = self.index_of(index)
        return super(HDUList, self).pop(list_index)
```
### 26 - astropy/io/fits/hdu/hdulist.py:

Start line: 657, End line: 712

```python
class HDUList(list, _Verify):

    def index_of(self, key):
        """
        Get the index of an HDU from the `HDUList`.

        Parameters
        ----------
        key : int, str or tuple of (string, int)
           The key identifying the HDU.  If ``key`` is a tuple, it is of the
           form ``(key, ver)`` where ``ver`` is an ``EXTVER`` value that must
           match the HDU being searched for.

           If the key is ambiguous (e.g. there are multiple 'SCI' extensions)
           the first match is returned.  For a more precise match use the
           ``(name, ver)`` pair.

           If even the ``(name, ver)`` pair is ambiguous (it shouldn't be
           but it's not impossible) the numeric index must be used to index
           the duplicate HDU.

        Returns
        -------
        index : int
           The index of the HDU in the `HDUList`.
        """

        if _is_int(key):
            return key
        elif isinstance(key, tuple):
            _key, _ver = key
        else:
            _key = key
            _ver = None

        if not isinstance(_key, str):
            raise KeyError(
                '{} indices must be integers, extension names as strings, '
                'or (extname, version) tuples; got {}'
                ''.format(self.__class__.__name__, _key))

        _key = (_key.strip()).upper()

        found = None
        for idx, hdu in enumerate(self):
            name = hdu.name
            if isinstance(name, str):
                name = name.strip().upper()
            # 'PRIMARY' should always work as a reference to the first HDU
            if ((name == _key or (_key == 'PRIMARY' and idx == 0)) and
                (_ver is None or _ver == hdu.ver)):
                found = idx
                break

        if (found is None):
            raise KeyError('Extension {!r} not found.'.format(key))
        else:
            return found
```
### 28 - astropy/io/fits/hdu/hdulist.py:

Start line: 940, End line: 999

```python
class HDUList(list, _Verify):

    def info(self, output=None):
        """
        Summarize the info of the HDUs in this `HDUList`.

        Note that this function prints its results to the console---it
        does not return a value.

        Parameters
        ----------
        output : file, bool, optional
            A file-like object to write the output to.  If `False`, does not
            output to a file and instead returns a list of tuples representing
            the HDU info.  Writes to ``sys.stdout`` by default.
        """

        if output is None:
            output = sys.stdout

        if self._file is None:
            name = '(No file associated with this HDUList)'
        else:
            name = self._file.name

        results = ['Filename: {}'.format(name),
                   'No.    Name      Ver    Type      Cards   Dimensions   Format']

        format = '{:3d}  {:10}  {:3} {:11}  {:5d}   {}   {}   {}'
        default = ('', '', '', 0, (), '', '')
        for idx, hdu in enumerate(self):
            summary = hdu._summary()
            if len(summary) < len(default):
                summary += default[len(summary):]
            summary = (idx,) + summary
            if output:
                results.append(format.format(*summary))
            else:
                results.append(summary)

        if output:
            output.write('\n'.join(results))
            output.write('\n')
            output.flush()
        else:
            return results[2:]

    def filename(self):
        """
        Return the file name associated with the HDUList object if one exists.
        Otherwise returns None.

        Returns
        -------
        filename : a string containing the file name associated with the
                   HDUList object if an association exists.  Otherwise returns
                   None.
        """
        if self._file is not None:
            if hasattr(self._file, 'name'):
                return self._file.name
        return None
```
### 29 - astropy/io/fits/hdu/hdulist.py:

Start line: 1348, End line: 1390

```python
class HDUList(list, _Verify):

    def _wasresized(self, verbose=False):
        """
        Determine if any changes to the HDUList will require a file resize
        when flushing the file.

        Side effect of setting the objects _resize attribute.
        """

        if not self._resize:

            # determine if any of the HDU is resized
            for hdu in self:
                # Header:
                nbytes = len(str(hdu._header))
                if nbytes != (hdu._data_offset - hdu._header_offset):
                    self._resize = True
                    self._truncate = False
                    if verbose:
                        print('One or more header is resized.')
                    break

                # Data:
                if not hdu._has_data:
                    continue

                nbytes = hdu.size
                nbytes = nbytes + _pad_length(nbytes)
                if nbytes != hdu._data_size:
                    self._resize = True
                    self._truncate = False
                    if verbose:
                        print('One or more data area is resized.')
                    break

            if self._truncate:
                try:
                    self._file.truncate(hdu._data_offset + hdu._data_size)
                except OSError:
                    self._resize = True
                self._truncate = False

        return self._resize
```
### 31 - astropy/io/fits/hdu/hdulist.py:

Start line: 812, End line: 842

```python
class HDUList(list, _Verify):

    def update_extend(self):
        """
        Make sure that if the primary header needs the keyword ``EXTEND`` that
        it has it and it is correct.
        """

        if not len(self):
            return

        if not isinstance(self[0], PrimaryHDU):
            # A PrimaryHDU will be automatically inserted at some point, but it
            # might not have been added yet
            return

        hdr = self[0].header

        def get_first_ext():
            try:
                return self[1]
            except IndexError:
                return None

        if 'EXTEND' in hdr:
            if not hdr['EXTEND'] and get_first_ext() is not None:
                hdr['EXTEND'] = True
        elif get_first_ext() is not None:
            if hdr['NAXIS'] == 0:
                hdr.set('EXTEND', True, after='NAXIS')
            else:
                n = hdr['NAXIS']
                hdr.set('EXTEND', True, after='NAXIS' + str(n))
```
### 32 - astropy/io/fits/hdu/hdulist.py:

Start line: 28, End line: 141

```python
def fitsopen(name, mode='readonly', memmap=None, save_backup=False,
             cache=True, lazy_load_hdus=None, **kwargs):
    """Factory function to open a FITS file and return an `HDUList` object.

    Parameters
    ----------
    name : file path, file object, file-like object or pathlib.Path object
        File to be opened.

    mode : str, optional
        Open mode, 'readonly' (default), 'update', 'append', 'denywrite', or
        'ostream'.

        If ``name`` is a file object that is already opened, ``mode`` must
        match the mode the file was opened with, readonly (rb), update (rb+),
        append (ab+), ostream (w), denywrite (rb)).

    memmap : bool, optional
        Is memory mapping to be used?

    save_backup : bool, optional
        If the file was opened in update or append mode, this ensures that a
        backup of the original file is saved before any changes are flushed.
        The backup has the same name as the original file with ".bak" appended.
        If "file.bak" already exists then "file.bak.1" is used, and so on.

    cache : bool, optional
        If the file name is a URL, `~astropy.utils.data.download_file` is used
        to open the file.  This specifies whether or not to save the file
        locally in Astropy's download cache (default: `True`).

    lazy_load_hdus : bool, option
        By default `~astropy.io.fits.open` will not read all the HDUs and
        headers in a FITS file immediately upon opening.  This is an
        optimization especially useful for large files, as FITS has no way
        of determining the number and offsets of all the HDUs in a file
        without scanning through the file and reading all the headers.

        To disable lazy loading and read all HDUs immediately (the old
        behavior) use ``lazy_load_hdus=False``.  This can lead to fewer
        surprises--for example with lazy loading enabled, ``len(hdul)``
        can be slow, as it means the entire FITS file needs to be read in
        order to determine the number of HDUs.  ``lazy_load_hdus=False``
        ensures that all HDUs have already been loaded after the file has
        been opened.

        .. versionadded:: 1.3

    kwargs : dict, optional
        additional optional keyword arguments, possible values are:

        - **uint** : bool

            Interpret signed integer data where ``BZERO`` is the
            central value and ``BSCALE == 1`` as unsigned integer
            data.  For example, ``int16`` data with ``BZERO = 32768``
            and ``BSCALE = 1`` would be treated as ``uint16`` data.
            This is enabled by default so that the pseudo-unsigned
            integer convention is assumed.

            Note, for backward compatibility, the kwarg **uint16** may
            be used instead.  The kwarg was renamed when support was
            added for integers of any size.

        - **ignore_missing_end** : bool

            Do not issue an exception when opening a file that is
            missing an ``END`` card in the last header.

        - **checksum** : bool, str

            If `True`, verifies that both ``DATASUM`` and
            ``CHECKSUM`` card values (when present in the HDU header)
            match the header and data of all HDU's in the file.  Updates to a
            file that already has a checksum will preserve and update the
            existing checksums unless this argument is given a value of
            'remove', in which case the CHECKSUM and DATASUM values are not
            checked, and are removed when saving changes to the file.

        - **disable_image_compression** : bool

            If `True`, treats compressed image HDU's like normal
            binary table HDU's.

        - **do_not_scale_image_data** : bool

            If `True`, image data is not scaled using BSCALE/BZERO values
            when read.

        - **character_as_bytes** : bool

            Whether to return bytes for string columns. By default this is `False`
            and (unicode) strings are returned, but this does not respect memory
            mapping and loads the whole column in memory when accessed.

        - **ignore_blank** : bool

            If `True`, the BLANK keyword is ignored if present.

        - **scale_back** : bool

            If `True`, when saving changes to a file that contained scaled
            image data, restore the data to the original type and reapply the
            original BSCALE/BZERO values.  This could lead to loss of accuracy
            if scaling back to integer values after performing floating point
            operations on the data.

    Returns
    -------
        hdulist : an `HDUList` object
            `HDUList` containing all of the header data units in the
            file.

    """
    # ... other code
```
### 36 - astropy/io/fits/hdu/hdulist.py:

Start line: 4, End line: 25

```python
import bz2
import gzip
import itertools
import os
import shutil
import sys
import warnings

import numpy as np

from . import compressed
from .base import _BaseHDU, _ValidHDU, _NonstandardHDU, ExtensionHDU
from .groups import GroupsHDU
from .image import PrimaryHDU, ImageHDU
from ..file import _File
from ..header import _pad_length
from ..util import (_is_int, _tmp_name, fileobj_closed, ignore_sigint,
                    _get_array_mmap, _free_space_check)
from ..verify import _Verify, _ErrList, VerifyError, VerifyWarning
from ....utils import indent
from ....utils.exceptions import AstropyUserWarning
from ....utils.decorators import deprecated_renamed_argument
```
### 37 - astropy/io/fits/hdu/hdulist.py:

Start line: 143, End line: 164

```python
def fitsopen(name, mode='readonly', memmap=None, save_backup=False,
             cache=True, lazy_load_hdus=None, **kwargs):

    from .. import conf

    if memmap is None:
        # distinguish between True (kwarg explicitly set)
        # and None (preference for memmap in config, might be ignored)
        memmap = None if conf.use_memmap else False
    else:
        memmap = bool(memmap)

    if lazy_load_hdus is None:
        lazy_load_hdus = conf.lazy_load_hdus
    else:
        lazy_load_hdus = bool(lazy_load_hdus)

    if 'uint' not in kwargs:
        kwargs['uint'] = conf.enable_uint

    if not name:
        raise ValueError('Empty filename: {!r}'.format(name))

    return HDUList.fromfile(name, mode, memmap, save_backup, cache,
                            lazy_load_hdus, **kwargs)
```
### 69 - astropy/io/fits/hdu/hdulist.py:

Start line: 908, End line: 938

```python
class HDUList(list, _Verify):

    def close(self, output_verify='exception', verbose=False, closed=True):
        """
        Close the associated FITS file and memmap object, if any.

        Parameters
        ----------
        output_verify : str
            Output verification option.  Must be one of ``"fix"``,
            ``"silentfix"``, ``"ignore"``, ``"warn"``, or
            ``"exception"``.  May also be any combination of ``"fix"`` or
            ``"silentfix"`` with ``"+ignore"``, ``+warn``, or ``+exception"
            (e.g. ``"fix+warn"``).  See :ref:`verify` for more info.

        verbose : bool
            When `True`, print out verbose messages.

        closed : bool
            When `True`, close the underlying file object.
        """

        try:
            if (self._file and self._file.mode in ('append', 'update')
                    and not self._file.closed):
                self.flush(output_verify=output_verify, verbose=verbose)
        finally:
            if self._file and closed and hasattr(self._file, 'close'):
                self._file.close()

            # Give individual HDUs an opportunity to do on-close cleanup
            for hdu in self:
                hdu._close(closed=closed)
```
