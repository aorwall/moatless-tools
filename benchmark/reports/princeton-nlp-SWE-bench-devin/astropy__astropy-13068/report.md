# astropy__astropy-13068

| **astropy/astropy** | `2288ecd4e9c4d3722d72b7f4a6555a34f4f04fc7` |
| ---- | ---- |
| **No of patches** | 2 |
| **All found context length** | 14984 |
| **Any found context length** | 14984 |
| **Avg pos** | 45.5 |
| **Min pos** | 42 |
| **Max pos** | 49 |
| **Top file pos** | 1 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/astropy/time/core.py b/astropy/time/core.py
--- a/astropy/time/core.py
+++ b/astropy/time/core.py
@@ -655,9 +655,6 @@ def precision(self):
     @precision.setter
     def precision(self, val):
         del self.cache
-        if not isinstance(val, int) or val < 0 or val > 9:
-            raise ValueError('precision attribute must be an int between '
-                             '0 and 9')
         self._time.precision = val
 
     @property
diff --git a/astropy/time/formats.py b/astropy/time/formats.py
--- a/astropy/time/formats.py
+++ b/astropy/time/formats.py
@@ -230,6 +230,18 @@ def masked(self):
     def jd2_filled(self):
         return np.nan_to_num(self.jd2) if self.masked else self.jd2
 
+    @property
+    def precision(self):
+        return self._precision
+
+    @precision.setter
+    def precision(self, val):
+        #Verify precision is 0-9 (inclusive)
+        if not isinstance(val, int) or val < 0 or val > 9:
+            raise ValueError('precision attribute must be an int between '
+                             '0 and 9')
+        self._precision = val
+
     @lazyproperty
     def cache(self):
         """

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| astropy/time/core.py | 658 | 660 | 49 | 2 | 17141
| astropy/time/formats.py | 233 | 233 | 42 | 1 | 14984


## Problem Statement

```
Time from astropy.time not precise
Hello,

I encounter difficulties with Time. I'm working on a package to perform photometry and occultation. 

For this last case, data need times values accurately estimated. Of course, data coming from different camera will will have different time format in the header.

to manage this without passing long time to build a time parser, i decided to use Time object which do exactly what i need. The problem is, i dont arrive to make accurate conversion between different format using Time.

let's do an exemple:

\`\`\`
t1 = '2022-03-24T23:13:41.390999'
t1 = Time(t1, format = 'isot', precision = len(t1.split('.')[-1]))
t2 = t1.to_value('jd')
# result is 2459663.4678401737
\`\`\`
now let's do reverse

\`\`\`
t2 = Time(t2, format = 'jd', precision = len(str(t2).split('.')[-1]))
t3 = t2.to_value('isot')
# result is 2022-03-24T23:13:41.0551352177
\`\`\`
as you can see i don't fall back on the same value and the difference is quite high. I would like to fall back on the original one.

thank you in advance


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 astropy/time/formats.py** | 651 | 681| 291 | 291 | 19170 | 
| 2 | **2 astropy/time/core.py** | 2127 | 2163| 454 | 745 | 45811 | 
| 3 | **2 astropy/time/formats.py** | 1601 | 1630| 268 | 1013 | 45811 | 
| 4 | **2 astropy/time/formats.py** | 1363 | 1392| 356 | 1369 | 45811 | 
| 5 | **2 astropy/time/core.py** | 452 | 509| 555 | 1924 | 45811 | 
| 6 | **2 astropy/time/formats.py** | 1340 | 1361| 302 | 2226 | 45811 | 
| 7 | **2 astropy/time/core.py** | 2177 | 2233| 509 | 2735 | 45811 | 
| 8 | **2 astropy/time/formats.py** | 1277 | 1315| 359 | 3094 | 45811 | 
| 9 | **2 astropy/time/formats.py** | 1140 | 1160| 213 | 3307 | 45811 | 
| 10 | 3 astropy/io/misc/asdf/tags/time/time.py | 95 | 131| 330 | 3637 | 46805 | 
| 11 | **3 astropy/time/core.py** | 1552 | 1595| 381 | 4018 | 46805 | 
| 12 | **3 astropy/time/core.py** | 783 | 824| 504 | 4522 | 46805 | 
| 13 | **3 astropy/time/formats.py** | 1255 | 1275| 228 | 4750 | 46805 | 
| 14 | **3 astropy/time/formats.py** | 1 | 45| 443 | 5193 | 46805 | 
| 15 | **3 astropy/time/core.py** | 1658 | 1707| 459 | 5652 | 46805 | 
| 16 | 4 astropy/visualization/time.py | 150 | 185| 409 | 6061 | 48882 | 
| 17 | **4 astropy/time/formats.py** | 552 | 576| 333 | 6394 | 48882 | 
| 18 | **4 astropy/time/core.py** | 825 | 874| 470 | 6864 | 48882 | 
| 19 | **4 astropy/time/core.py** | 2685 | 2714| 320 | 7184 | 48882 | 
| 20 | **4 astropy/time/core.py** | 1486 | 1550| 684 | 7868 | 48882 | 
| 21 | **4 astropy/time/core.py** | 1597 | 1632| 310 | 8178 | 48882 | 
| 22 | **4 astropy/time/formats.py** | 1894 | 1905| 128 | 8306 | 48882 | 
| 23 | **4 astropy/time/formats.py** | 1123 | 1138| 158 | 8464 | 48882 | 
| 24 | **4 astropy/time/formats.py** | 1838 | 1864| 228 | 8692 | 48882 | 
| 25 | **4 astropy/time/formats.py** | 1317 | 1338| 245 | 8937 | 48882 | 
| 26 | 4 astropy/io/misc/asdf/tags/time/time.py | 33 | 93| 454 | 9391 | 48882 | 
| 27 | 4 astropy/visualization/time.py | 187 | 256| 473 | 9864 | 48882 | 
| 28 | **4 astropy/time/core.py** | 2235 | 2277| 365 | 10229 | 48882 | 
| 29 | 5 astropy/extern/_strptime.py | 127 | 170| 642 | 10871 | 54403 | 
| 30 | **5 astropy/time/formats.py** | 1585 | 1599| 124 | 10995 | 54403 | 
| 31 | **5 astropy/time/formats.py** | 1708 | 1717| 140 | 11135 | 54403 | 
| 32 | **5 astropy/time/formats.py** | 1502 | 1535| 414 | 11549 | 54403 | 
| 33 | **5 astropy/time/core.py** | 1709 | 1743| 283 | 11832 | 54403 | 
| 34 | **5 astropy/time/formats.py** | 946 | 967| 262 | 12094 | 54403 | 
| 35 | **5 astropy/time/formats.py** | 1451 | 1499| 593 | 12687 | 54403 | 
| 36 | **5 astropy/time/formats.py** | 513 | 550| 419 | 13106 | 54403 | 
| 37 | 6 astropy/io/fits/fitstime.py | 331 | 362| 208 | 13314 | 59772 | 
| 38 | **6 astropy/time/core.py** | 299 | 323| 247 | 13561 | 59772 | 
| 39 | **6 astropy/time/formats.py** | 398 | 407| 110 | 13671 | 59772 | 
| 40 | **6 astropy/time/core.py** | 1158 | 1173| 171 | 13842 | 59772 | 
| 41 | 6 astropy/io/fits/fitstime.py | 365 | 423| 573 | 14415 | 59772 | 
| **-> 42 <-** | **6 astropy/time/formats.py** | 157 | 238| 569 | 14984 | 59772 | 
| 43 | **6 astropy/time/formats.py** | 817 | 836| 183 | 15167 | 59772 | 
| 44 | **6 astropy/time/formats.py** | 861 | 918| 519 | 15686 | 59772 | 
| 45 | **6 astropy/time/core.py** | 1815 | 1825| 158 | 15844 | 59772 | 
| 46 | **6 astropy/time/formats.py** | 599 | 649| 672 | 16516 | 59772 | 
| 47 | 6 astropy/extern/_strptime.py | 172 | 187| 168 | 16684 | 59772 | 
| 48 | **6 astropy/time/core.py** | 537 | 554| 188 | 16872 | 59772 | 
| **-> 49 <-** | **6 astropy/time/core.py** | 647 | 687| 269 | 17141 | 59772 | 
| 50 | **6 astropy/time/core.py** | 588 | 645| 604 | 17745 | 59772 | 
| 51 | **6 astropy/time/formats.py** | 122 | 141| 247 | 17992 | 59772 | 
| 52 | **6 astropy/time/core.py** | 381 | 450| 657 | 18649 | 59772 | 
| 53 | **6 astropy/time/core.py** | 511 | 535| 261 | 18910 | 59772 | 
| 54 | **6 astropy/time/core.py** | 2165 | 2175| 130 | 19040 | 59772 | 
| 55 | **6 astropy/time/core.py** | 2114 | 2126| 144 | 19184 | 59772 | 
| 56 | **6 astropy/time/formats.py** | 1672 | 1706| 405 | 19589 | 59772 | 
| 57 | **6 astropy/time/core.py** | 2393 | 2425| 250 | 19839 | 59772 | 
| 58 | **6 astropy/time/core.py** | 742 | 765| 197 | 20036 | 59772 | 
| 59 | **6 astropy/time/formats.py** | 1867 | 1892| 252 | 20288 | 59772 | 
| 60 | **6 astropy/time/core.py** | 208 | 267| 570 | 20858 | 59772 | 
| 61 | **6 astropy/time/core.py** | 189 | 206| 170 | 21028 | 59772 | 
| 62 | **6 astropy/time/formats.py** | 240 | 296| 644 | 21672 | 59772 | 
| 63 | **6 astropy/time/formats.py** | 79 | 120| 350 | 22022 | 59772 | 
| 64 | 7 astropy/io/misc/asdf/tags/time/timedelta.py | 3 | 16| 134 | 22156 | 60073 | 
| 65 | 7 astropy/visualization/time.py | 52 | 148| 825 | 22981 | 60073 | 
| 66 | **7 astropy/time/formats.py** | 448 | 474| 270 | 23251 | 60073 | 
| 67 | 7 astropy/io/fits/fitstime.py | 549 | 605| 612 | 23863 | 60073 | 
| 68 | **7 astropy/time/core.py** | 2279 | 2321| 350 | 24213 | 60073 | 
| 69 | 8 astropy/time/utils.py | 203 | 247| 375 | 24588 | 62279 | 
| 70 | **8 astropy/time/formats.py** | 1423 | 1448| 241 | 24829 | 62279 | 
| 71 | 8 astropy/extern/_strptime.py | 115 | 125| 140 | 24969 | 62279 | 
| 72 | **8 astropy/time/core.py** | 2017 | 2076| 579 | 25548 | 62279 | 
| 73 | **8 astropy/time/formats.py** | 491 | 510| 220 | 25768 | 62279 | 
| 74 | **8 astropy/time/core.py** | 950 | 978| 258 | 26026 | 62279 | 
| 75 | **8 astropy/time/core.py** | 556 | 586| 224 | 26250 | 62279 | 
| 76 | 8 astropy/io/misc/asdf/tags/time/timedelta.py | 16 | 37| 141 | 26391 | 62279 | 
| 77 | **8 astropy/time/core.py** | 2078 | 2112| 438 | 26829 | 62279 | 
| 78 | **8 astropy/time/core.py** | 1374 | 1407| 255 | 27084 | 62279 | 
| 79 | 8 astropy/extern/_strptime.py | 310 | 368| 572 | 27656 | 62279 | 
| 80 | **8 astropy/time/formats.py** | 704 | 753| 576 | 28232 | 62279 | 
| 81 | **8 astropy/time/formats.py** | 921 | 944| 222 | 28454 | 62279 | 
| 82 | 8 astropy/extern/_strptime.py | 509 | 530| 213 | 28667 | 62279 | 
| 83 | **8 astropy/time/core.py** | 1 | 79| 956 | 29623 | 62279 | 
| 84 | 8 astropy/time/utils.py | 186 | 200| 138 | 29761 | 62279 | 
| 85 | **8 astropy/time/formats.py** | 839 | 858| 211 | 29972 | 62279 | 
| 86 | 8 astropy/visualization/time.py | 1 | 17| 109 | 30081 | 62279 | 
| 87 | **8 astropy/time/formats.py** | 579 | 597| 163 | 30244 | 62279 | 
| 88 | 9 astropy/time/time_helper/__init__.py | 1 | 5| 12 | 30256 | 62292 | 
| 89 | **9 astropy/time/formats.py** | 409 | 446| 396 | 30652 | 62292 | 
| 90 | **9 astropy/time/formats.py** | 298 | 326| 213 | 30865 | 62292 | 
| 91 | **9 astropy/time/formats.py** | 477 | 488| 106 | 30971 | 62292 | 
| 92 | **9 astropy/time/formats.py** | 1740 | 1761| 251 | 31222 | 62292 | 
| 93 | **9 astropy/time/core.py** | 1019 | 1042| 166 | 31388 | 62292 | 
| 94 | 9 astropy/extern/_strptime.py | 461 | 507| 563 | 31951 | 62292 | 
| 95 | **9 astropy/time/formats.py** | 969 | 1022| 555 | 32506 | 62292 | 
| 96 | **9 astropy/time/core.py** | 2427 | 2450| 246 | 32752 | 62292 | 
| 97 | **9 astropy/time/formats.py** | 769 | 788| 250 | 33002 | 62292 | 
| 98 | **9 astropy/time/formats.py** | 791 | 815| 273 | 33275 | 62292 | 
| 99 | 10 astropy/coordinates/attributes.py | 134 | 191| 330 | 33605 | 65816 | 
| 100 | **10 astropy/time/formats.py** | 1720 | 1737| 198 | 33803 | 65816 | 
| 101 | 11 astropy/io/votable/tree.py | 1807 | 1853| 337 | 34140 | 94959 | 
| 102 | **11 astropy/time/core.py** | 2324 | 2391| 699 | 34839 | 94959 | 
| 103 | **11 astropy/time/core.py** | 1044 | 1073| 256 | 35095 | 94959 | 
| 104 | 12 astropy/wcs/docstrings.py | 2823 | 2960| 735 | 35830 | 116412 | 
| 105 | 12 astropy/io/votable/tree.py | 1787 | 1805| 196 | 36026 | 116412 | 
| 106 | **12 astropy/time/core.py** | 2616 | 2684| 747 | 36773 | 116412 | 
| 107 | **12 astropy/time/formats.py** | 1538 | 1582| 544 | 37317 | 116412 | 
| 108 | 12 astropy/io/misc/asdf/tags/time/time.py | 3 | 30| 194 | 37511 | 116412 | 
| 109 | **12 astropy/time/formats.py** | 328 | 364| 319 | 37830 | 116412 | 
| 110 | **12 astropy/time/core.py** | 712 | 740| 254 | 38084 | 116412 | 
| 111 | 12 astropy/io/fits/fitstime.py | 279 | 306| 251 | 38335 | 116412 | 
| 112 | **12 astropy/time/core.py** | 1457 | 1483| 206 | 38541 | 116412 | 
| 113 | 12 astropy/extern/_strptime.py | 369 | 460| 933 | 39474 | 116412 | 
| 114 | 13 astropy/timeseries/sampled.py | 61 | 142| 701 | 40175 | 119811 | 
| 115 | **13 astropy/time/formats.py** | 1058 | 1121| 559 | 40734 | 119811 | 
| 116 | **13 astropy/time/formats.py** | 684 | 701| 205 | 40939 | 119811 | 
| 117 | **13 astropy/time/core.py** | 767 | 781| 126 | 41065 | 119811 | 
| 118 | **13 astropy/time/core.py** | 2584 | 2614| 250 | 41315 | 119811 | 
| 119 | **13 astropy/time/core.py** | 1075 | 1157| 793 | 42108 | 119811 | 
| 120 | **13 astropy/time/core.py** | 1634 | 1656| 180 | 42288 | 119811 | 
| 121 | **13 astropy/time/core.py** | 2482 | 2517| 302 | 42590 | 119811 | 
| 122 | **13 astropy/time/core.py** | 1409 | 1431| 204 | 42794 | 119811 | 
| 123 | 13 astropy/extern/_strptime.py | 190 | 234| 671 | 43465 | 119811 | 
| 124 | **13 astropy/time/core.py** | 1279 | 1295| 188 | 43653 | 119811 | 
| 125 | **13 astropy/time/core.py** | 1954 | 2015| 628 | 44281 | 119811 | 
| 126 | 13 astropy/io/fits/fitstime.py | 426 | 498| 603 | 44884 | 119811 | 
| 127 | **13 astropy/time/formats.py** | 1764 | 1802| 426 | 45310 | 119811 | 
| 128 | **13 astropy/time/formats.py** | 756 | 766| 129 | 45439 | 119811 | 
| 129 | 14 astropy/coordinates/builtin_frames/intermediate_rotation_transforms.py | 23 | 42| 254 | 45693 | 123190 | 
| 130 | 14 astropy/visualization/time.py | 20 | 50| 319 | 46012 | 123190 | 
| 131 | **14 astropy/time/formats.py** | 1214 | 1253| 426 | 46438 | 123190 | 
| 132 | **14 astropy/time/core.py** | 80 | 111| 467 | 46905 | 123190 | 
| 133 | 14 astropy/io/fits/fitstime.py | 309 | 328| 139 | 47044 | 123190 | 
| 134 | **14 astropy/time/core.py** | 689 | 710| 212 | 47256 | 123190 | 
| 135 | **14 astropy/time/formats.py** | 1394 | 1421| 317 | 47573 | 123190 | 
| 136 | 15 astropy/time/time_helper/function_helpers.py | 1 | 31| 235 | 47808 | 123426 | 
| 137 | 16 astropy/coordinates/builtin_frames/utils.py | 98 | 140| 258 | 48066 | 127258 | 
| 138 | **16 astropy/time/core.py** | 2551 | 2582| 309 | 48375 | 127258 | 
| 139 | 16 astropy/extern/_strptime.py | 1 | 39| 205 | 48580 | 127258 | 
| 140 | 17 astropy/timeseries/periodograms/bls/core.py | 359 | 373| 169 | 48749 | 134862 | 
| 141 | **17 astropy/time/formats.py** | 1633 | 1670| 580 | 49329 | 134862 | 
| 142 | **17 astropy/time/formats.py** | 1908 | 1922| 158 | 49487 | 134862 | 
| 143 | **17 astropy/time/formats.py** | 1925 | 1949| 248 | 49735 | 134862 | 
| 144 | 18 astropy/timeseries/periodograms/lombscargle/core.py | 363 | 386| 196 | 49931 | 140795 | 
| 145 | **18 astropy/time/formats.py** | 1805 | 1835| 238 | 50169 | 140795 | 
| 146 | **18 astropy/time/core.py** | 1878 | 1952| 753 | 50922 | 140795 | 
| 147 | 19 astropy/io/votable/exceptions.py | 1460 | 1472| 145 | 51067 | 154127 | 
| 148 | 19 astropy/timeseries/sampled.py | 208 | 245| 384 | 51451 | 154127 | 
| 149 | 19 astropy/io/fits/fitstime.py | 501 | 547| 431 | 51882 | 154127 | 
| 150 | **19 astropy/time/core.py** | 1433 | 1455| 236 | 52118 | 154127 | 
| 151 | **19 astropy/time/formats.py** | 1163 | 1211| 476 | 52594 | 154127 | 
| 152 | 20 astropy/coordinates/angle_formats.py | 562 | 649| 832 | 53426 | 159330 | 
| 153 | 21 astropy/timeseries/__init__.py | 1 | 13| 106 | 53532 | 159437 | 
| 154 | 21 astropy/coordinates/builtin_frames/utils.py | 1 | 39| 322 | 53854 | 159437 | 
| 155 | 21 astropy/timeseries/sampled.py | 278 | 316| 283 | 54137 | 159437 | 
| 156 | **21 astropy/time/core.py** | 1329 | 1344| 193 | 54330 | 159437 | 
| 157 | **21 astropy/time/formats.py** | 1025 | 1056| 348 | 54678 | 159437 | 
| 158 | **21 astropy/time/core.py** | 2519 | 2549| 301 | 54979 | 159437 | 
| 159 | **21 astropy/time/formats.py** | 143 | 155| 131 | 55110 | 159437 | 
| 160 | **21 astropy/time/core.py** | 876 | 948| 685 | 55795 | 159437 | 
| 161 | **21 astropy/time/core.py** | 1175 | 1189| 128 | 55923 | 159437 | 
| 162 | 22 astropy/timeseries/core.py | 46 | 101| 443 | 56366 | 160151 | 
| 163 | 23 astropy/coordinates/builtin_frames/equatorial.py | 1 | 43| 271 | 56637 | 161217 | 
| 164 | 24 astropy/utils/iers/iers.py | 12 | 78| 762 | 57399 | 172830 | 
| 165 | 24 astropy/io/fits/fitstime.py | 3 | 66| 581 | 57980 | 172830 | 
| 166 | 25 astropy/coordinates/builtin_frames/altaz.py | 38 | 73| 428 | 58408 | 174031 | 
| 167 | 25 astropy/coordinates/builtin_frames/utils.py | 334 | 385| 501 | 58909 | 174031 | 
| 168 | 25 astropy/time/utils.py | 75 | 115| 471 | 59380 | 174031 | 
| 169 | 26 astropy/coordinates/sky_coordinate.py | 752 | 820| 779 | 60159 | 193641 | 
| 170 | 26 astropy/coordinates/angle_formats.py | 514 | 526| 123 | 60282 | 193641 | 
| 171 | **26 astropy/time/core.py** | 1745 | 1813| 787 | 61069 | 193641 | 
| 172 | 27 astropy/timeseries/io/__init__.py | 1 | 4| 23 | 61092 | 193664 | 
| 173 | **27 astropy/time/core.py** | 326 | 378| 447 | 61539 | 193664 | 
| 174 | **27 astropy/time/core.py** | 1827 | 1876| 465 | 62004 | 193664 | 
| 175 | 27 astropy/timeseries/periodograms/bls/core.py | 332 | 357| 210 | 62214 | 193664 | 
| 176 | 27 astropy/utils/iers/iers.py | 391 | 449| 661 | 62875 | 193664 | 
| 177 | 27 astropy/extern/_strptime.py | 255 | 287| 311 | 63186 | 193664 | 
| 178 | 27 astropy/coordinates/builtin_frames/intermediate_rotation_transforms.py | 45 | 62| 205 | 63391 | 193664 | 


### Hint

```
Welcome to Astropy 👋 and thank you for your first issue!

A project member will respond to you as soon as possible; in the meantime, please double-check the [guidelines for submitting issues](https://github.com/astropy/astropy/blob/main/CONTRIBUTING.md#reporting-issues) and make sure you've provided the requested details.

GitHub issues in the Astropy repository are used to track bug reports and feature requests; If your issue poses a question about how to use Astropy, please instead raise your question in the [Astropy Discourse user forum](https://community.openastronomy.org/c/astropy/8) and close this issue.

If you feel that this issue has not been responded to in a timely manner, please leave a comment mentioning our software support engineer @embray, or send a message directly to the [development mailing list](http://groups.google.com/group/astropy-dev).  If the issue is urgent or sensitive in nature (e.g., a security vulnerability) please send an e-mail directly to the private e-mail feedback@astropy.org.
@mhvk will have the answer I guess, but it seems the issue comes from the use of `precision`, which probably does not do what you expect. And should be <= 9 :

> precision: int between 0 and 9 inclusive
    Decimal precision when outputting seconds as floating point.

The interesting thing is that when precision is > 9 the results are incorrect:

\`\`\`
In [52]: for p in range(15):
    ...:     print(f'{p:2d}', Time(t2, format = 'jd', precision = p).to_value('isot'))
    ...: 
 0 2022-03-24T23:13:41
 1 2022-03-24T23:13:41.4
 2 2022-03-24T23:13:41.39
 3 2022-03-24T23:13:41.391
 4 2022-03-24T23:13:41.3910
 5 2022-03-24T23:13:41.39101
 6 2022-03-24T23:13:41.391012
 7 2022-03-24T23:13:41.3910118
 8 2022-03-24T23:13:41.39101177
 9 2022-03-24T23:13:41.391011775
10 2022-03-24T23:13:41.0551352177
11 2022-03-24T23:13:41.00475373422
12 2022-03-24T23:13:41.-00284414132
13 2022-03-24T23:13:41.0000514624247
14 2022-03-24T23:13:41.00000108094123
\`\`\`

To get a better precision you can use `.to_value('jd', 'long')`: (and the weird results with `precision > 9` remain)

\`\`\`
In [53]: t2 = t1.to_value('jd', 'long'); t2
Out[53]: 2459663.4678401735996

In [54]: for p in range(15):
    ...:     print(f'{p:2d}', Time(t2, format = 'jd', precision = p).to_value('isot'))
    ...: 
 0 2022-03-24T23:13:41
 1 2022-03-24T23:13:41.4
 2 2022-03-24T23:13:41.39
 3 2022-03-24T23:13:41.391
 4 2022-03-24T23:13:41.3910
 5 2022-03-24T23:13:41.39100
 6 2022-03-24T23:13:41.390999
 7 2022-03-24T23:13:41.3909990
 8 2022-03-24T23:13:41.39099901
 9 2022-03-24T23:13:41.390999005
10 2022-03-24T23:13:41.0551334172
11 2022-03-24T23:13:41.00475357898
12 2022-03-24T23:13:41.-00284404844
13 2022-03-24T23:13:41.0000514607441
14 2022-03-24T23:13:41.00000108090593
\`\`\`
`astropy.time.Time` uses two float 64 to obtain very high precision, from the docs:

> All time manipulations and arithmetic operations are done internally using two 64-bit floats to represent time. Floating point algorithms from [1](https://docs.astropy.org/en/stable/time/index.html#id2) are used so that the [Time](https://docs.astropy.org/en/stable/api/astropy.time.Time.html#astropy.time.Time) object maintains sub-nanosecond precision over times spanning the age of the universe.

https://docs.astropy.org/en/stable/time/index.html

By doing `t1.to_value('jd')` you combine the two floats into a single float, loosing precision. However, the difference should not be 2 seconds, rather in the microsecond range.

When I leave out the precision argument or setting it to 9 for nanosecond precision, I get a difference of 12µs when going through the single jd float, which is expected:

\`\`\`
from astropy.time import Time
import astropy.units as u


isot = '2022-03-24T23:13:41.390999'

t1 = Time(isot, format = 'isot', precision=9)
jd = t1.to_value('jd')
t2 = Time(jd, format='jd', precision=9)

print(f"Original:       {t1.isot}")
print(f"Converted back: {t2.isot}")
print(f"Difference:     {(t2 - t1).to(u.us):.2f}")

t3 = Time(t1.jd1, t1.jd2, format='jd', precision=9)
print(f"Using jd1+jd2:  {t3.isot}")
print(f"Difference:     {(t3 - t1).to(u.ns):.2f}")
\`\`\`

prints:

\`\`\`
Original:       2022-03-24T23:13:41.390999000
Converted back: 2022-03-24T23:13:41.391011775
Difference:     12.77 us
Using jd1+jd2:  2022-03-24T23:13:41.390999000
Difference:     0.00 ns
\`\`\`
Thank you for your answers.

do they are a way to have access to this two floats? if i use jd tranformation it's because it's more easy for me to manipulate numbers. 
@antoinech13 See my example, it accesses `t1.jd1` and `t1.jd2`.
oh yes thank you.
Probably we should keep this open to address the issue with precsion > 9 that @saimn found?
sorry. yes indeed
Hello, I'm not familiar with this repository, but from my quick skimming it seems that using a precision outside of the range 0-9 (inclusive) is intended to trigger an exception. (see [here](https://github.com/astropy/astropy/blob/main/astropy/time/core.py#L610-L611), note that this line is part of the `TimeBase` class which `Time` inherits from). Though someone more familiar with the repository can correct me if I'm wrong.

Edit:
It seems the exception was only written for the setter and not for the case where `Time()` is initialized with the precision. Thus:
\`\`\`
>>> from astropy.time import Time
>>> t1 = Time(123, fromat="jd")
>>> t1.precision = 10
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/home/brett/env/lib/python3.8/site-packages/astropy/time/core.py", line 610, in precision
    raise ValueError('precision attribute must be an int between '
ValueError: precision attribute must be an int between 0 and 9
\`\`\`
produces the exception, but 
\`\`\`
>>> t2 = Time(123, format="jd", precision=10)
>>> 
\`\`\`
does not.

@saimn - good catch on the precision issue, this is not expected but seems to be the cause of the original problem.

This precision is just being passed straight through to ERFA, which clearly is not doing any validation on that value. It looks like giving a value > 9 actually causes a bug in the output, yikes.
FYI @antoinech13 - the `precision` argument only impacts the precision of the seconds output in string formats like `isot`. So setting the precision for a `jd` format `Time` object is generally not necessary.
@taldcroft - I looked and indeed there is no specific check in https://github.com/liberfa/erfa/blob/master/src/d2tf.c, though the comment notes:
\`\`\`
**  2) The largest positive useful value for ndp is determined by the
**     size of days, the format of double on the target platform, and
**     the risk of overflowing ihmsf[3].  On a typical platform, for
**     days up to 1.0, the available floating-point precision might
**     correspond to ndp=12.  However, the practical limit is typically
**     ndp=9, set by the capacity of a 32-bit int, or ndp=4 if int is
**     only 16 bits.
\`\`\`
This is actually a bit misleading, since the fraction of the second is stored in a 32-bit int, so it cannot possibly store more than 9 digits. Indeed,
\`\`\`
In [31]: from erfa import d2tf

In [32]: d2tf(9, 1-2**-47)
Out[32]: (b'+', (23, 59, 59, 999999999))

In [33]: d2tf(10, 1-2**-47)
Out[33]: (b'+', (23, 59, 59, 1410065407))

In [34]: np.int32('9'*10)
Out[34]: 1410065407

In [36]: np.int32('9'*9)
Out[36]: 999999999
\`\`\`
As for how to fix this, right now we do check `precision` as a property, but not on input:
\`\`\`
In [42]: t = Time('J2000')

In [43]: t = Time('J2000', precision=10)

In [44]: t.precision
Out[44]: 10

In [45]: t.precision = 10
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<ipython-input-45-59f84a57d617> in <module>
----> 1 t.precision = 10

/usr/lib/python3/dist-packages/astropy/time/core.py in precision(self, val)
    608         del self.cache
    609         if not isinstance(val, int) or val < 0 or val > 9:
--> 610             raise ValueError('precision attribute must be an int between '
    611                              '0 and 9')
    612         self._time.precision = val

ValueError: precision attribute must be an int between 0 and 9
\`\`\`
Seems reasonable to check on input as well.
```

## Patch

```diff
diff --git a/astropy/time/core.py b/astropy/time/core.py
--- a/astropy/time/core.py
+++ b/astropy/time/core.py
@@ -655,9 +655,6 @@ def precision(self):
     @precision.setter
     def precision(self, val):
         del self.cache
-        if not isinstance(val, int) or val < 0 or val > 9:
-            raise ValueError('precision attribute must be an int between '
-                             '0 and 9')
         self._time.precision = val
 
     @property
diff --git a/astropy/time/formats.py b/astropy/time/formats.py
--- a/astropy/time/formats.py
+++ b/astropy/time/formats.py
@@ -230,6 +230,18 @@ def masked(self):
     def jd2_filled(self):
         return np.nan_to_num(self.jd2) if self.masked else self.jd2
 
+    @property
+    def precision(self):
+        return self._precision
+
+    @precision.setter
+    def precision(self, val):
+        #Verify precision is 0-9 (inclusive)
+        if not isinstance(val, int) or val < 0 or val > 9:
+            raise ValueError('precision attribute must be an int between '
+                             '0 and 9')
+        self._precision = val
+
     @lazyproperty
     def cache(self):
         """

```

## Test Patch

```diff
diff --git a/astropy/time/tests/test_basic.py b/astropy/time/tests/test_basic.py
--- a/astropy/time/tests/test_basic.py
+++ b/astropy/time/tests/test_basic.py
@@ -259,6 +259,20 @@ def test_precision(self):
         assert t.iso == '2010-01-01 00:00:00.000000000'
         assert t.tai.utc.iso == '2010-01-01 00:00:00.000000000'
 
+    def test_precision_input(self):
+        """Verifies that precision can only be 0-9 (inclusive). Any other
+        value should raise a ValueError exception."""
+
+        err_message = 'precision attribute must be an int'
+
+        with pytest.raises(ValueError, match=err_message):
+            t = Time('2010-01-01 00:00:00', format='iso', scale='utc',
+                     precision=10)
+
+        with pytest.raises(ValueError, match=err_message):
+            t = Time('2010-01-01 00:00:00', format='iso', scale='utc')
+            t.precision = -1
+
     def test_transforms(self):
         """Transform from UTC to all supported time scales (TAI, TCB, TCG,
         TDB, TT, UT1, UTC).  This requires auxiliary information (latitude and

```


## Code snippets

### 1 - astropy/time/formats.py:

Start line: 651, End line: 681

```python
class TimeFromEpoch(TimeNumeric):

    def to_value(self, parent=None, **kwargs):
        # Make sure that scale is the same as epoch scale so we can just
        # subtract the epoch and convert
        if self.scale != self.epoch_scale:
            if parent is None:
                raise ValueError('cannot compute value without parent Time object')
            try:
                tm = getattr(parent, self.epoch_scale)
            except Exception as err:
                raise ScaleValueError("Cannot convert from '{}' epoch scale '{}'"
                                      "to specified scale '{}', got error:\n{}"
                                      .format(self.name, self.epoch_scale,
                                              self.scale, err)) from err

            jd1, jd2 = tm._time.jd1, tm._time.jd2
        else:
            jd1, jd2 = self.jd1, self.jd2

        # This factor is guaranteed to be exactly representable, which
        # means time_from_epoch1 is calculated exactly.
        factor = 1. / self.unit
        time_from_epoch1 = (jd1 - self.epoch.jd1) * factor
        time_from_epoch2 = (jd2 - self.epoch.jd2) * factor

        return super().to_value(jd1=time_from_epoch1, jd2=time_from_epoch2, **kwargs)

    value = property(to_value)

    @property
    def _default_scale(self):
        return self.epoch_scale
```
### 2 - astropy/time/core.py:

Start line: 2127, End line: 2163

```python
class Time(TimeBase):
    def _get_delta_tdb_tt(self, jd1=None, jd2=None):
        if not hasattr(self, '_delta_tdb_tt'):
            # If jd1 and jd2 are not provided (which is the case for property
            # attribute access) then require that the time scale is TT or TDB.
            # Otherwise the computations here are not correct.
            if jd1 is None or jd2 is None:
                if self.scale not in ('tt', 'tdb'):
                    raise ValueError('Accessing the delta_tdb_tt attribute '
                                     'is only possible for TT or TDB time '
                                     'scales')
                else:
                    jd1 = self._time.jd1
                    jd2 = self._time.jd2_filled

            # First go from the current input time (which is either
            # TDB or TT) to an approximate UT1.  Since TT and TDB are
            # pretty close (few msec?), assume TT.  Similarly, since the
            # UT1 terms are very small, use UTC instead of UT1.
            njd1, njd2 = erfa.tttai(jd1, jd2)
            njd1, njd2 = erfa.taiutc(njd1, njd2)
            # subtract 0.5, so UT is fraction of the day from midnight
            ut = day_frac(njd1 - 0.5, njd2)[1]

            if self.location is None:
                # Assume geocentric.
                self._delta_tdb_tt = erfa.dtdb(jd1, jd2, ut, 0., 0., 0.)
            else:
                location = self.location
                # Geodetic params needed for d_tdb_tt()
                lon = location.lon
                rxy = np.hypot(location.x, location.y)
                z = location.z
                self._delta_tdb_tt = erfa.dtdb(
                    jd1, jd2, ut, lon.to_value(u.radian),
                    rxy.to_value(u.km), z.to_value(u.km))

        return self._delta_tdb_tt
```
### 3 - astropy/time/formats.py:

Start line: 1601, End line: 1630

```python
class TimeDatetime64(TimeISOT):

    def set_jds(self, val1, val2):
        # If there are any masked values in the ``val1`` datetime64 array
        # ('NaT') then stub them with a valid date so downstream parse_string
        # will work.  The value under the mask is arbitrary but a "modern" date
        # is good.
        mask = np.isnat(val1)
        masked = np.any(mask)
        if masked:
            val1 = val1.copy()
            val1[mask] = '2000'

        # Make sure M(onth) and Y(ear) dates will parse and convert to bytestring
        if val1.dtype.name in ['datetime64[M]', 'datetime64[Y]']:
            val1 = val1.astype('datetime64[D]')
        val1 = val1.astype('S')

        # Standard ISO string parsing now
        super().set_jds(val1, val2)

        # Finally apply mask if necessary
        if masked:
            self.jd2[mask] = np.nan

    @property
    def value(self):
        precision = self.precision
        self.precision = 9
        ret = super().value
        self.precision = precision
        return ret.astype('datetime64')
```
### 4 - astropy/time/formats.py:

Start line: 1363, End line: 1392

```python
class TimeString(TimeUnique):

    def get_jds_fast(self, val1, val2):
        """Use fast C parser to parse time strings in val1 and get jd1, jd2"""
        # Handle bytes or str input and convert to uint8.  We need to the
        # dtype _parse_times.dt_u1 instead of uint8, since otherwise it is
        # not possible to create a gufunc with structured dtype output.
        # See note about ufunc type resolver in pyerfa/erfa/ufunc.c.templ.
        if val1.dtype.kind == 'U':
            # Note: val1.astype('S') is *very* slow, so we check ourselves
            # that the input is pure ASCII.
            val1_uint32 = val1.view((np.uint32, val1.dtype.itemsize // 4))
            if np.any(val1_uint32 > 127):
                raise ValueError('input is not pure ASCII')

            # It might be possible to avoid making a copy via astype with
            # cleverness in parse_times.c but leave that for another day.
            chars = val1_uint32.astype(_parse_times.dt_u1)

        else:
            chars = val1.view((_parse_times.dt_u1, val1.dtype.itemsize))

        # Call the fast parsing ufunc.
        time_struct = self._fast_parser(chars)
        jd1, jd2 = erfa.dtf2d(self.scale.upper().encode('ascii'),
                              time_struct['year'],
                              time_struct['month'],
                              time_struct['day'],
                              time_struct['hour'],
                              time_struct['minute'],
                              time_struct['second'])
        return day_frac(jd1, jd2)
```
### 5 - astropy/time/core.py:

Start line: 452, End line: 509

```python
class TimeBase(ShapedLikeNDArray):

    def _get_time_fmt(self, val, val2, format, scale,
                      precision, in_subfmt, out_subfmt):
        """
        Given the supplied val, val2, format and scale try to instantiate
        the corresponding TimeFormat class to convert the input values into
        the internal jd1 and jd2.

        If format is `None` and the input is a string-type or object array then
        guess available formats and stop when one matches.
        """

        if (format is None
                and (val.dtype.kind in ('S', 'U', 'O', 'M') or val.dtype.names)):
            # Input is a string, object, datetime, or a table-like ndarray
            # (structured array, recarray). These input types can be
            # uniquely identified by the format classes.
            formats = [(name, cls) for name, cls in self.FORMATS.items()
                       if issubclass(cls, TimeUnique)]

            # AstropyTime is a pseudo-format that isn't in the TIME_FORMATS registry,
            # but try to guess it at the end.
            formats.append(('astropy_time', TimeAstropyTime))

        elif not (isinstance(format, str)
                  and format.lower() in self.FORMATS):
            if format is None:
                raise ValueError("No time format was given, and the input is "
                                 "not unique")
            else:
                raise ValueError("Format {!r} is not one of the allowed "
                                 "formats {}".format(format,
                                                     sorted(self.FORMATS)))
        else:
            formats = [(format, self.FORMATS[format])]

        assert formats
        problems = {}
        for name, cls in formats:
            try:
                return cls(val, val2, scale, precision, in_subfmt, out_subfmt)
            except UnitConversionError:
                raise
            except (ValueError, TypeError) as err:
                # If ``format`` specified then there is only one possibility, so raise
                # immediately and include the upstream exception message to make it
                # easier for user to see what is wrong.
                if len(formats) == 1:
                    raise ValueError(
                        f'Input values did not match the format class {format}:'
                        + os.linesep
                        + f'{err.__class__.__name__}: {err}'
                    ) from err
                else:
                    problems[name] = err
        else:
            raise ValueError(f'Input values did not match any of the formats '
                             f'where the format keyword is optional: '
                             f'{problems}') from problems[formats[0][0]]
```
### 6 - astropy/time/formats.py:

Start line: 1340, End line: 1361

```python
class TimeString(TimeUnique):

    def get_jds_python(self, val1, val2):
        """Parse the time strings contained in val1 and get jd1, jd2"""
        # Select subformats based on current self.in_subfmt
        subfmts = self._select_subfmts(self.in_subfmt)
        # Be liberal in what we accept: convert bytes to ascii.
        # Here .item() is needed for arrays with entries of unequal length,
        # to strip trailing 0 bytes.
        to_string = (str if val1.dtype.kind == 'U' else
                     lambda x: str(x.item(), encoding='ascii'))
        iterator = np.nditer([val1, None, None, None, None, None, None],
                             flags=['zerosize_ok'],
                             op_dtypes=[None] + 5 * [np.intc] + [np.double])
        for val, iy, im, id, ihr, imin, dsec in iterator:
            val = to_string(val)
            iy[...], im[...], id[...], ihr[...], imin[...], dsec[...] = (
                self.parse_string(val, subfmts))

        jd1, jd2 = erfa.dtf2d(self.scale.upper().encode('ascii'),
                              *iterator.operands[1:])
        jd1, jd2 = day_frac(jd1, jd2)

        return jd1, jd2
```
### 7 - astropy/time/core.py:

Start line: 2177, End line: 2233

```python
class Time(TimeBase):

    def __sub__(self, other):
        # T      - Tdelta = T
        # T      - T      = Tdelta
        other_is_delta = not isinstance(other, Time)
        if other_is_delta:  # T - Tdelta
            # Check other is really a TimeDelta or something that can initialize.
            if not isinstance(other, TimeDelta):
                try:
                    other = TimeDelta(other)
                except Exception:
                    return NotImplemented

            # we need a constant scale to calculate, which is guaranteed for
            # TimeDelta, but not for Time (which can be UTC)
            out = self.replicate()
            if self.scale in other.SCALES:
                if other.scale not in (out.scale, None):
                    other = getattr(other, out.scale)
            else:
                if other.scale is None:
                    out._set_scale('tai')
                else:
                    if self.scale not in TIME_TYPES[other.scale]:
                        raise TypeError("Cannot subtract Time and TimeDelta instances "
                                        "with scales '{}' and '{}'"
                                        .format(self.scale, other.scale))
                    out._set_scale(other.scale)
            # remove attributes that are invalidated by changing time
            for attr in ('_delta_ut1_utc', '_delta_tdb_tt'):
                if hasattr(out, attr):
                    delattr(out, attr)

        else:  # T - T
            # the scales should be compatible (e.g., cannot convert TDB to LOCAL)
            if other.scale not in self.SCALES:
                raise TypeError("Cannot subtract Time instances "
                                "with scales '{}' and '{}'"
                                .format(self.scale, other.scale))
            self_time = (self._time if self.scale in TIME_DELTA_SCALES
                         else self.tai._time)
            # set up TimeDelta, subtraction to be done shortly
            out = TimeDelta(self_time.jd1, self_time.jd2, format='jd',
                            scale=self_time.scale)

            if other.scale != out.scale:
                other = getattr(other, out.scale)

        jd1 = out._time.jd1 - other._time.jd1
        jd2 = out._time.jd2 - other._time.jd2

        out._time.jd1, out._time.jd2 = day_frac(jd1, jd2)

        if other_is_delta:
            # Go back to left-side scale if needed
            out._set_scale(self.scale)

        return out
```
### 8 - astropy/time/formats.py:

Start line: 1277, End line: 1315

```python
class TimeString(TimeUnique):

    def parse_string(self, timestr, subfmts):
        """Read time from a single string, using a set of possible formats."""
        # Datetime components required for conversion to JD by ERFA, along
        # with the default values.
        components = ('year', 'mon', 'mday', 'hour', 'min', 'sec')
        defaults = (None, 1, 1, 0, 0, 0)
        # Assume that anything following "." on the right side is a
        # floating fraction of a second.
        try:
            idot = timestr.rindex('.')
        except Exception:
            fracsec = 0.0
        else:
            timestr, fracsec = timestr[:idot], timestr[idot:]
            fracsec = float(fracsec)

        for _, strptime_fmt_or_regex, _ in subfmts:
            if isinstance(strptime_fmt_or_regex, str):
                try:
                    tm = time.strptime(timestr, strptime_fmt_or_regex)
                except ValueError:
                    continue
                else:
                    vals = [getattr(tm, 'tm_' + component)
                            for component in components]

            else:
                tm = re.match(strptime_fmt_or_regex, timestr)
                if tm is None:
                    continue
                tm = tm.groupdict()
                vals = [int(tm.get(component, default)) for component, default
                        in zip(components, defaults)]

            # Add fractional seconds
            vals[-1] = vals[-1] + fracsec
            return vals
        else:
            raise ValueError(f'Time {timestr} does not match {self.name} format')
```
### 9 - astropy/time/formats.py:

Start line: 1140, End line: 1160

```python
class TimeYMDHMS(TimeUnique):

    @property
    def value(self):
        scale = self.scale.upper().encode('ascii')
        iys, ims, ids, ihmsfs = erfa.d2dtf(scale, 9,
                                           self.jd1, self.jd2_filled)

        out = np.empty(self.jd1.shape, dtype=[('year', 'i4'),
                                              ('month', 'i4'),
                                              ('day', 'i4'),
                                              ('hour', 'i4'),
                                              ('minute', 'i4'),
                                              ('second', 'f8')])
        out['year'] = iys
        out['month'] = ims
        out['day'] = ids
        out['hour'] = ihmsfs['h']
        out['minute'] = ihmsfs['m']
        out['second'] = ihmsfs['s'] + ihmsfs['f'] * 10**(-9)
        out = out.view(np.recarray)

        return self.mask_if_needed(out)
```
### 10 - astropy/io/misc/asdf/tags/time/time.py:

Start line: 95, End line: 131

```python
class TimeType(AstropyAsdfType):

    @classmethod
    def from_tree(cls, node, ctx):
        if isinstance(node, (str, list, np.ndarray)):
            t = time.Time(node)
            fmt = _astropy_format_to_asdf_format.get(t.format, t.format)
            if fmt not in _guessable_formats:
                raise ValueError(f"Invalid time '{node}'")
            return t

        value = node['value']
        fmt = node.get('format')
        scale = node.get('scale')
        location = node.get('location')
        if location is not None:
            unit = location.get('unit', u.m)
            # This ensures that we can read the v.1.0.0 schema and convert it
            # to the new EarthLocation object, which expects Quantity components
            for comp in ['x', 'y', 'z']:
                if not isinstance(location[comp], Quantity):
                    location[comp] = Quantity(location[comp], unit=unit)
            location = EarthLocation.from_geocentric(
                location['x'], location['y'], location['z'])

        return time.Time(value, format=fmt, scale=scale, location=location)

    @classmethod
    def assert_equal(cls, old, new):
        assert old.format == new.format
        assert old.scale == new.scale
        if isinstance(old.location, EarthLocation):
            assert isinstance(new.location, EarthLocation)
            _assert_earthlocation_equal(old.location, new.location)
        else:
            assert old.location == new.location

        assert_array_equal(old, new)
```
### 11 - astropy/time/core.py:

Start line: 1552, End line: 1595

```python
class Time(TimeBase):

    def __init__(self, val, val2=None, format=None, scale=None,
                 precision=None, in_subfmt=None, out_subfmt=None,
                 location=None, copy=False):

        if location is not None:
            from astropy.coordinates import EarthLocation
            if isinstance(location, EarthLocation):
                self.location = location
            else:
                self.location = EarthLocation(*location)
            if self.location.size == 1:
                self.location = self.location.squeeze()
        else:
            if not hasattr(self, 'location'):
                self.location = None

        if isinstance(val, Time):
            # Update _time formatting parameters if explicitly specified
            if precision is not None:
                self._time.precision = precision
            if in_subfmt is not None:
                self._time.in_subfmt = in_subfmt
            if out_subfmt is not None:
                self._time.out_subfmt = out_subfmt
            self.SCALES = TIME_TYPES[self.scale]
            if scale is not None:
                self._set_scale(scale)
        else:
            self._init_from_vals(val, val2, format, scale, copy,
                                 precision, in_subfmt, out_subfmt)
            self.SCALES = TIME_TYPES[self.scale]

        if self.location is not None and (self.location.size > 1
                                          and self.location.shape != self.shape):
            try:
                # check the location can be broadcast to self's shape.
                self.location = np.broadcast_to(self.location, self.shape,
                                                subok=True)
            except Exception as err:
                raise ValueError('The location with shape {} cannot be '
                                 'broadcast against time with shape {}. '
                                 'Typically, either give a single location or '
                                 'one for each time.'
                                 .format(self.location.shape, self.shape)) from err
```
### 12 - astropy/time/core.py:

Start line: 783, End line: 824

```python
class TimeBase(ShapedLikeNDArray):

    def to_value(self, format, subfmt='*'):
        """Get time values expressed in specified output format.

        This method allows representing the ``Time`` object in the desired
        output ``format`` and optional sub-format ``subfmt``.  Available
        built-in formats include ``jd``, ``mjd``, ``iso``, and so forth. Each
        format can have its own sub-formats

        For built-in numerical formats like ``jd`` or ``unix``, ``subfmt`` can
        be one of 'float', 'long', 'decimal', 'str', or 'bytes'.  Here, 'long'
        uses ``numpy.longdouble`` for somewhat enhanced precision (with
        the enhancement depending on platform), and 'decimal'
        :class:`decimal.Decimal` for full precision.  For 'str' and 'bytes', the
        number of digits is also chosen such that time values are represented
        accurately.

        For built-in date-like string formats, one of 'date_hms', 'date_hm', or
        'date' (or 'longdate_hms', etc., for 5-digit years in
        `~astropy.time.TimeFITS`).  For sub-formats including seconds, the
        number of digits used for the fractional seconds is as set by
        `~astropy.time.Time.precision`.

        Parameters
        ----------
        format : str
            The format in which one wants the time values. Default: the current
            format.
        subfmt : str or None, optional
            Value or wildcard pattern to select the sub-format in which the
            values should be given.  The default of '*' picks the first
            available for a given format, i.e., 'float' or 'date_hms'.
            If `None`, use the instance's ``out_subfmt``.

        """
        # TODO: add a precision argument (but ensure it is keyword argument
        # only, to make life easier for TimeDelta.to_value()).
        if format not in self.FORMATS:
            raise ValueError(f'format must be one of {list(self.FORMATS)}')

        cache = self.cache['format']
        # Try to keep cache behaviour like it was in astropy < 4.0.
        key = format if subfmt is None else (format, subfmt)
        # ... other code
```
### 13 - astropy/time/formats.py:

Start line: 1255, End line: 1275

```python
class TimeString(TimeUnique):

    def __init_subclass__(cls, **kwargs):
        if 'fast_parser_pars' in cls.__dict__:
            fpp = cls.fast_parser_pars
            fpp = np.array(list(zip(map(chr, fpp['delims']),
                                    fpp['starts'],
                                    fpp['stops'],
                                    fpp['break_allowed'])),
                           _parse_times.dt_pars)
            if cls.fast_parser_pars['has_day_of_year']:
                fpp['start'][1] = fpp['stop'][1] = -1
            cls._fast_parser = _parse_times.create_parser(fpp)

        super().__init_subclass__(**kwargs)

    def _check_val_type(self, val1, val2):
        if val1.dtype.kind not in ('S', 'U') and val1.size:
            raise TypeError(f'Input values for {self.name} class must be strings')
        if val2 is not None:
            raise ValueError(
                f'{self.name} objects do not accept a val2 but you provided {val2}')
        return val1, None
```
### 14 - astropy/time/formats.py:

Start line: 1, End line: 45

```python
# -*- coding: utf-8 -*-
import fnmatch
import time
import re
import datetime
import warnings
from decimal import Decimal
from collections import OrderedDict, defaultdict

import numpy as np
import erfa

from astropy.utils.decorators import lazyproperty, classproperty
from astropy.utils.exceptions import AstropyDeprecationWarning
import astropy.units as u

from . import _parse_times
from . import utils
from .utils import day_frac, quantity_day_frac, two_sum, two_product
from . import conf

__all__ = ['TimeFormat', 'TimeJD', 'TimeMJD', 'TimeFromEpoch', 'TimeUnix',
           'TimeUnixTai', 'TimeCxcSec', 'TimeGPS', 'TimeDecimalYear',
           'TimePlotDate', 'TimeUnique', 'TimeDatetime', 'TimeString',
           'TimeISO', 'TimeISOT', 'TimeFITS', 'TimeYearDayTime',
           'TimeEpochDate', 'TimeBesselianEpoch', 'TimeJulianEpoch',
           'TimeDeltaFormat', 'TimeDeltaSec', 'TimeDeltaJD',
           'TimeEpochDateString', 'TimeBesselianEpochString',
           'TimeJulianEpochString', 'TIME_FORMATS', 'TIME_DELTA_FORMATS',
           'TimezoneInfo', 'TimeDeltaDatetime', 'TimeDatetime64', 'TimeYMDHMS',
           'TimeNumeric', 'TimeDeltaNumeric']

__doctest_skip__ = ['TimePlotDate']

# These both get filled in at end after TimeFormat subclasses defined.
# Use an OrderedDict to fix the order in which formats are tried.
# This ensures, e.g., that 'isot' gets tried before 'fits'.
TIME_FORMATS = OrderedDict()
TIME_DELTA_FORMATS = OrderedDict()

# Translations between deprecated FITS timescales defined by
# Rots et al. 2015, A&A 574:A36, and timescales used here.
FITS_DEPRECATED_SCALES = {'TDT': 'tt', 'ET': 'tt',
                          'GMT': 'utc', 'UT': 'utc', 'IAT': 'tai'}
```
### 15 - astropy/time/core.py:

Start line: 1658, End line: 1707

```python
class Time(TimeBase):

    @classmethod
    def strptime(cls, time_string, format_string, **kwargs):
        """
        Parse a string to a Time according to a format specification.
        See `time.strptime` documentation for format specification.

        >>> Time.strptime('2012-Jun-30 23:59:60', '%Y-%b-%d %H:%M:%S')
        <Time object: scale='utc' format='isot' value=2012-06-30T23:59:60.000>

        Parameters
        ----------
        time_string : str, sequence, or ndarray
            Objects containing time data of type string
        format_string : str
            String specifying format of time_string.
        kwargs : dict
            Any keyword arguments for ``Time``.  If the ``format`` keyword
            argument is present, this will be used as the Time format.

        Returns
        -------
        time_obj : `~astropy.time.Time`
            A new `~astropy.time.Time` object corresponding to the input
            ``time_string``.

        """
        time_array = np.asarray(time_string)

        if time_array.dtype.kind not in ('U', 'S'):
            err = "Expected type is string, a bytes-like object or a sequence"\
                  " of these. Got dtype '{}'".format(time_array.dtype.kind)
            raise TypeError(err)

        to_string = (str if time_array.dtype.kind == 'U' else
                     lambda x: str(x.item(), encoding='ascii'))
        iterator = np.nditer([time_array, None],
                             op_dtypes=[time_array.dtype, 'U30'])

        for time, formatted in iterator:
            tt, fraction = _strptime._strptime(to_string(time), format_string)
            time_tuple = tt[:6] + (fraction,)
            formatted[...] = '{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:06}'\
                .format(*time_tuple)

        format = kwargs.pop('format', None)
        out = cls(*iterator.operands[1:], format='isot', **kwargs)
        if format is not None:
            out.format = format

        return out
```
### 17 - astropy/time/formats.py:

Start line: 552, End line: 576

```python
class TimeDecimalYear(TimeNumeric):

    def to_value(self, **kwargs):
        scale = self.scale.upper().encode('ascii')
        iy_start, ims, ids, ihmsfs = erfa.d2dtf(scale, 0,  # precision=0
                                                self.jd1, self.jd2_filled)
        imon = np.ones_like(iy_start)
        iday = np.ones_like(iy_start)
        ihr = np.zeros_like(iy_start)
        imin = np.zeros_like(iy_start)
        isec = np.zeros_like(self.jd1)

        # Possible enhancement: use np.unique to only compute start, stop
        # for unique values of iy_start.
        scale = self.scale.upper().encode('ascii')
        jd1_start, jd2_start = erfa.dtf2d(scale, iy_start, imon, iday,
                                          ihr, imin, isec)
        jd1_end, jd2_end = erfa.dtf2d(scale, iy_start + 1, imon, iday,
                                      ihr, imin, isec)
        # Trying to be precise, but more than float64 not useful.
        dt = (self.jd1 - jd1_start) + (self.jd2 - jd2_start)
        dt_end = (jd1_end - jd1_start) + (jd2_end - jd2_start)
        decimalyear = iy_start + dt / dt_end

        return super().to_value(jd1=decimalyear, jd2=np.float64(0.0), **kwargs)

    value = property(to_value)
```
### 18 - astropy/time/core.py:

Start line: 825, End line: 874

```python
class TimeBase(ShapedLikeNDArray):

    def to_value(self, format, subfmt='*'):
        # ... other code
        if key not in cache:
            if format == self.format:
                tm = self
            else:
                tm = self.replicate(format=format)

            # Some TimeFormat subclasses may not be able to handle being passes
            # on a out_subfmt. This includes some core classes like
            # TimeBesselianEpochString that do not have any allowed subfmts. But
            # those do deal with `self.out_subfmt` internally, so if subfmt is
            # the same, we do not pass it on.
            kwargs = {}
            if subfmt is not None and subfmt != tm.out_subfmt:
                kwargs['out_subfmt'] = subfmt
            try:
                value = tm._time.to_value(parent=tm, **kwargs)
            except TypeError as exc:
                # Try validating subfmt, e.g. for formats like 'jyear_str' that
                # do not implement out_subfmt in to_value() (because there are
                # no allowed subformats).  If subfmt is not valid this gives the
                # same exception as would have occurred if the call to
                # `to_value()` had succeeded.
                tm._time._select_subfmts(subfmt)

                # Subfmt was valid, so fall back to the original exception to see
                # if it was lack of support for out_subfmt as a call arg.
                if "unexpected keyword argument 'out_subfmt'" in str(exc):
                    raise ValueError(
                        f"to_value() method for format {format!r} does not "
                        f"support passing a 'subfmt' argument") from None
                else:
                    # Some unforeseen exception so raise.
                    raise

            value = tm._shaped_like_input(value)
            cache[key] = value
        return cache[key]

    @property
    def value(self):
        """Time value(s) in current format"""
        return self.to_value(self.format, None)

    @property
    def masked(self):
        return self._time.masked

    @property
    def mask(self):
        return self._time.mask
```
### 19 - astropy/time/core.py:

Start line: 2685, End line: 2714

```python
class TimeDelta(TimeBase):

    def to_value(self, *args, **kwargs):
        # (effectively, by doing the reverse of quantity_day_frac)?
        # This way, only equivalencies could lead to possible precision loss.
        if ('format' in kwargs
                or (args != () and (args[0] is None or args[0] in self.FORMATS))):
            # Super-class will error with duplicate arguments, etc.
            return super().to_value(*args, **kwargs)

        # With positional arguments, we try parsing the first one as a unit,
        # so that on failure we can give a more informative exception.
        if args:
            try:
                unit = u.Unit(args[0])
            except ValueError as exc:
                raise ValueError("first argument is not one of the known "
                                 "formats ({}) and failed to parse as a unit."
                                 .format(list(self.FORMATS))) from exc
            args = (unit,) + args[1:]

        return u.Quantity(self._time.jd1 + self._time.jd2,
                          u.day).to_value(*args, **kwargs)

    def _make_value_equivalent(self, item, value):
        """Coerce setitem value into an equivalent TimeDelta object"""
        if not isinstance(value, TimeDelta):
            try:
                value = self.__class__(value, scale=self.scale, format=self.format)
            except Exception as err:
                raise ValueError('cannot convert value to a compatible TimeDelta '
                                 'object: {}'.format(err))
        return value
```
### 20 - astropy/time/core.py:

Start line: 1486, End line: 1550

```python
class Time(TimeBase):
    """
    Represent and manipulate times and dates for astronomy.

    A `Time` object is initialized with one or more times in the ``val``
    argument.  The input times in ``val`` must conform to the specified
    ``format`` and must correspond to the specified time ``scale``.  The
    optional ``val2`` time input should be supplied only for numeric input
    formats (e.g. JD) where very high precision (better than 64-bit precision)
    is required.

    The allowed values for ``format`` can be listed with::

      >>> list(Time.FORMATS)
      ['jd', 'mjd', 'decimalyear', 'unix', 'unix_tai', 'cxcsec', 'gps', 'plot_date',
       'stardate', 'datetime', 'ymdhms', 'iso', 'isot', 'yday', 'datetime64',
       'fits', 'byear', 'jyear', 'byear_str', 'jyear_str']

    See also: http://docs.astropy.org/en/stable/time/

    Parameters
    ----------
    val : sequence, ndarray, number, str, bytes, or `~astropy.time.Time` object
        Value(s) to initialize the time or times.  Bytes are decoded as ascii.
    val2 : sequence, ndarray, or number; optional
        Value(s) to initialize the time or times.  Only used for numerical
        input, to help preserve precision.
    format : str, optional
        Format of input value(s)
    scale : str, optional
        Time scale of input value(s), must be one of the following:
        ('tai', 'tcb', 'tcg', 'tdb', 'tt', 'ut1', 'utc')
    precision : int, optional
        Digits of precision in string representation of time
    in_subfmt : str, optional
        Unix glob to select subformats for parsing input times
    out_subfmt : str, optional
        Unix glob to select subformat for outputting times
    location : `~astropy.coordinates.EarthLocation` or tuple, optional
        If given as an tuple, it should be able to initialize an
        an EarthLocation instance, i.e., either contain 3 items with units of
        length for geocentric coordinates, or contain a longitude, latitude,
        and an optional height for geodetic coordinates.
        Can be a single location, or one for each input time.
        If not given, assumed to be the center of the Earth for time scale
        transformations to and from the solar-system barycenter.
    copy : bool, optional
        Make a copy of the input values
    """
    SCALES = TIME_SCALES
    """List of time scales"""

    FORMATS = TIME_FORMATS
    """Dict of time formats"""

    def __new__(cls, val, val2=None, format=None, scale=None,
                precision=None, in_subfmt=None, out_subfmt=None,
                location=None, copy=False):

        if isinstance(val, Time):
            self = val.replicate(format=format, copy=copy, cls=cls)
        else:
            self = super().__new__(cls)

        return self
```
### 21 - astropy/time/core.py:

Start line: 1597, End line: 1632

```python
class Time(TimeBase):

    def _make_value_equivalent(self, item, value):
        """Coerce setitem value into an equivalent Time object"""

        # If there is a vector location then broadcast to the Time shape
        # and then select with ``item``
        if self.location is not None and self.location.shape:
            self_location = np.broadcast_to(self.location, self.shape, subok=True)[item]
        else:
            self_location = self.location

        if isinstance(value, Time):
            # Make sure locations are compatible.  Location can be either None or
            # a Location object.
            if self_location is None and value.location is None:
                match = True
            elif ((self_location is None and value.location is not None)
                  or (self_location is not None and value.location is None)):
                match = False
            else:
                match = np.all(self_location == value.location)
            if not match:
                raise ValueError('cannot set to Time with different location: '
                                 'expected location={} and '
                                 'got location={}'
                                 .format(self_location, value.location))
        else:
            try:
                value = self.__class__(value, scale=self.scale, location=self_location)
            except Exception:
                try:
                    value = self.__class__(value, scale=self.scale, format=self.format,
                                           location=self_location)
                except Exception as err:
                    raise ValueError('cannot convert value to a compatible Time object: {}'
                                     .format(err))
        return value
```
### 22 - astropy/time/formats.py:

Start line: 1894, End line: 1905

```python
class TimeDeltaDatetime(TimeDeltaFormat, TimeUnique):

    @property
    def value(self):
        iterator = np.nditer([self.jd1, self.jd2, None],
                             flags=['refs_ok', 'zerosize_ok'],
                             op_dtypes=[None, None, object])

        for jd1, jd2, out in iterator:
            jd1_, jd2_ = day_frac(jd1, jd2)
            out[...] = datetime.timedelta(days=jd1_,
                                          microseconds=jd2_ * 86400 * 1e6)

        return self.mask_if_needed(iterator.operands[-1])
```
### 23 - astropy/time/formats.py:

Start line: 1123, End line: 1138

```python
class TimeYMDHMS(TimeUnique):

    def set_jds(self, val1, val2):
        if val1 is None:
            # Input was empty list []
            jd1 = np.array([], dtype=np.float64)
            jd2 = np.array([], dtype=np.float64)

        else:
            jd1, jd2 = erfa.dtf2d(self.scale.upper().encode('ascii'),
                                  val1['year'],
                                  val1.get('month', 1),
                                  val1.get('day', 1),
                                  val1.get('hour', 0),
                                  val1.get('minute', 0),
                                  val1.get('second', 0))

        self.jd1, self.jd2 = day_frac(jd1, jd2)
```
### 24 - astropy/time/formats.py:

Start line: 1838, End line: 1864

```python
class TimeDeltaNumeric(TimeDeltaFormat, TimeNumeric):

    def set_jds(self, val1, val2):
        self._check_scale(self._scale)  # Validate scale.
        self.jd1, self.jd2 = day_frac(val1, val2, divisor=1. / self.unit)

    def to_value(self, **kwargs):
        # Note that 1/unit is always exactly representable, so the
        # following multiplications are exact.
        factor = 1. / self.unit
        jd1 = self.jd1 * factor
        jd2 = self.jd2 * factor
        return super().to_value(jd1=jd1, jd2=jd2, **kwargs)

    value = property(to_value)


class TimeDeltaSec(TimeDeltaNumeric):
    """Time delta in SI seconds"""
    name = 'sec'
    unit = 1. / erfa.DAYSEC  # for quantity input


class TimeDeltaJD(TimeDeltaNumeric):
    """Time delta in Julian days (86400 SI seconds)"""
    name = 'jd'
    unit = 1.
```
### 25 - astropy/time/formats.py:

Start line: 1317, End line: 1338

```python
class TimeString(TimeUnique):

    def set_jds(self, val1, val2):
        """Parse the time strings contained in val1 and set jd1, jd2"""
        # If specific input subformat is required then use the Python parser.
        # Also do this if Time format class does not define `use_fast_parser` or
        # if the fast parser is entirely disabled. Note that `use_fast_parser`
        # is ignored for format classes that don't have a fast parser.
        if (self.in_subfmt != '*'
                or '_fast_parser' not in self.__class__.__dict__
                or conf.use_fast_parser == 'False'):
            jd1, jd2 = self.get_jds_python(val1, val2)
        else:
            try:
                jd1, jd2 = self.get_jds_fast(val1, val2)
            except Exception:
                # Fall through to the Python parser unless fast is forced.
                if conf.use_fast_parser == 'force':
                    raise
                else:
                    jd1, jd2 = self.get_jds_python(val1, val2)

        self.jd1 = jd1
        self.jd2 = jd2
```
### 28 - astropy/time/core.py:

Start line: 2235, End line: 2277

```python
class Time(TimeBase):

    def __add__(self, other):
        # T      + Tdelta = T
        # T      + T      = error
        if isinstance(other, Time):
            raise OperandTypeError(self, other, '+')

        # Check other is really a TimeDelta or something that can initialize.
        if not isinstance(other, TimeDelta):
            try:
                other = TimeDelta(other)
            except Exception:
                return NotImplemented

        # ideally, we calculate in the scale of the Time item, since that is
        # what we want the output in, but this may not be possible, since
        # TimeDelta cannot be converted arbitrarily
        out = self.replicate()
        if self.scale in other.SCALES:
            if other.scale not in (out.scale, None):
                other = getattr(other, out.scale)
        else:
            if other.scale is None:
                out._set_scale('tai')
            else:
                if self.scale not in TIME_TYPES[other.scale]:
                    raise TypeError("Cannot add Time and TimeDelta instances "
                                    "with scales '{}' and '{}'"
                                    .format(self.scale, other.scale))
                out._set_scale(other.scale)
        # remove attributes that are invalidated by changing time
        for attr in ('_delta_ut1_utc', '_delta_tdb_tt'):
            if hasattr(out, attr):
                delattr(out, attr)

        jd1 = out._time.jd1 + other._time.jd1
        jd2 = out._time.jd2 + other._time.jd2

        out._time.jd1, out._time.jd2 = day_frac(jd1, jd2)

        # Go back to left-side scale if needed
        out._set_scale(self.scale)

        return out
```
### 30 - astropy/time/formats.py:

Start line: 1585, End line: 1599

```python
class TimeDatetime64(TimeISOT):
    name = 'datetime64'

    def _check_val_type(self, val1, val2):
        if not val1.dtype.kind == 'M':
            if val1.size > 0:
                raise TypeError('Input values for {} class must be '
                                'datetime64 objects'.format(self.name))
            else:
                val1 = np.array([], 'datetime64[D]')
        if val2 is not None:
            raise ValueError(
                f'{self.name} objects do not accept a val2 but you provided {val2}')

        return val1, None
```
### 31 - astropy/time/formats.py:

Start line: 1708, End line: 1717

```python
class TimeFITS(TimeString):

    @property
    def value(self):
        """Convert times to strings, using signed 5 digit if necessary."""
        if 'long' not in self.out_subfmt:
            # If we have times before year 0 or after year 9999, we can
            # output only in a "long" format, using signed 5-digit years.
            jd = self.jd1 + self.jd2
            if jd.size and (jd.min() < 1721425.5 or jd.max() >= 5373484.5):
                self.out_subfmt = 'long' + self.out_subfmt
        return super().value
```
### 32 - astropy/time/formats.py:

Start line: 1502, End line: 1535

```python
class TimeISOT(TimeISO):
    """
    ISO 8601 compliant date-time format "YYYY-MM-DDTHH:MM:SS.sss...".
    This is the same as TimeISO except for a "T" instead of space between
    the date and time.
    For example, 2000-01-01T00:00:00.000 is midnight on January 1, 2000.

    The allowed subformats are:

    - 'date_hms': date + hours, mins, secs (and optional fractional secs)
    - 'date_hm': date + hours, mins
    - 'date': date
    """

    name = 'isot'
    subfmts = (('date_hms',
                '%Y-%m-%dT%H:%M:%S',
                '{year:d}-{mon:02d}-{day:02d}T{hour:02d}:{min:02d}:{sec:02d}'),
               ('date_hm',
                '%Y-%m-%dT%H:%M',
                '{year:d}-{mon:02d}-{day:02d}T{hour:02d}:{min:02d}'),
               ('date',
                '%Y-%m-%d',
                '{year:d}-{mon:02d}-{day:02d}'))

    # See TimeISO for explanation
    fast_parser_pars = dict(
        delims=(0, ord('-'), ord('-'), ord('T'), ord(':'), ord(':'), ord('.')),
        starts=(0, 4, 7, 10, 13, 16, 19),
        stops=(3, 6, 9, 12, 15, 18, -1),
        # Break allowed *before*
        #              y  m  d  h  m  s  f
        break_allowed=(0, 0, 0, 1, 0, 1, 1),
        has_day_of_year=0)
```
### 33 - astropy/time/core.py:

Start line: 1709, End line: 1743

```python
class Time(TimeBase):

    def strftime(self, format_spec):
        """
        Convert Time to a string or a numpy.array of strings according to a
        format specification.
        See `time.strftime` documentation for format specification.

        Parameters
        ----------
        format_spec : str
            Format definition of return string.

        Returns
        -------
        formatted : str or numpy.array
            String or numpy.array of strings formatted according to the given
            format string.

        """
        formatted_strings = []
        for sk in self.replicate('iso')._time.str_kwargs():
            date_tuple = date(sk['year'], sk['mon'], sk['day']).timetuple()
            datetime_tuple = (sk['year'], sk['mon'], sk['day'],
                              sk['hour'], sk['min'], sk['sec'],
                              date_tuple[6], date_tuple[7], -1)
            fmtd_str = format_spec
            if '%f' in fmtd_str:
                fmtd_str = fmtd_str.replace('%f', '{frac:0{precision}}'.format(
                    frac=sk['fracsec'], precision=self.precision))
            fmtd_str = strftime(fmtd_str, datetime_tuple)
            formatted_strings.append(fmtd_str)

        if self.isscalar:
            return formatted_strings[0]
        else:
            return np.array(formatted_strings).reshape(self.shape)
```
### 34 - astropy/time/formats.py:

Start line: 946, End line: 967

```python
class TimeDatetime(TimeUnique):

    def set_jds(self, val1, val2):
        """Convert datetime object contained in val1 to jd1, jd2"""
        # Iterate through the datetime objects, getting year, month, etc.
        iterator = np.nditer([val1, None, None, None, None, None, None],
                             flags=['refs_ok', 'zerosize_ok'],
                             op_dtypes=[None] + 5*[np.intc] + [np.double])
        for val, iy, im, id, ihr, imin, dsec in iterator:
            dt = val.item()

            if dt.tzinfo is not None:
                dt = (dt - dt.utcoffset()).replace(tzinfo=None)

            iy[...] = dt.year
            im[...] = dt.month
            id[...] = dt.day
            ihr[...] = dt.hour
            imin[...] = dt.minute
            dsec[...] = dt.second + dt.microsecond / 1e6

        jd1, jd2 = erfa.dtf2d(self.scale.upper().encode('ascii'),
                              *iterator.operands[1:])
        self.jd1, self.jd2 = day_frac(jd1, jd2)
```
### 35 - astropy/time/formats.py:

Start line: 1451, End line: 1499

```python
class TimeISO(TimeString):
    """
    ISO 8601 compliant date-time format "YYYY-MM-DD HH:MM:SS.sss...".
    For example, 2000-01-01 00:00:00.000 is midnight on January 1, 2000.

    The allowed subformats are:

    - 'date_hms': date + hours, mins, secs (and optional fractional secs)
    - 'date_hm': date + hours, mins
    - 'date': date
    """

    name = 'iso'
    subfmts = (('date_hms',
                '%Y-%m-%d %H:%M:%S',
                # XXX To Do - use strftime for output ??
                '{year:d}-{mon:02d}-{day:02d} {hour:02d}:{min:02d}:{sec:02d}'),
               ('date_hm',
                '%Y-%m-%d %H:%M',
                '{year:d}-{mon:02d}-{day:02d} {hour:02d}:{min:02d}'),
               ('date',
                '%Y-%m-%d',
                '{year:d}-{mon:02d}-{day:02d}'))

    # Define positions and starting delimiter for year, month, day, hour,
    # minute, seconds components of an ISO time. This is used by the fast
    # C-parser parse_ymdhms_times()
    #
    #  "2000-01-12 13:14:15.678"
    #   01234567890123456789012
    #   yyyy-mm-dd hh:mm:ss.fff
    # Parsed as ('yyyy', '-mm', '-dd', ' hh', ':mm', ':ss', '.fff')
    fast_parser_pars = dict(
        delims=(0, ord('-'), ord('-'), ord(' '), ord(':'), ord(':'), ord('.')),
        starts=(0, 4, 7, 10, 13, 16, 19),
        stops=(3, 6, 9, 12, 15, 18, -1),
        # Break allowed *before*
        #              y  m  d  h  m  s  f
        break_allowed=(0, 0, 0, 1, 0, 1, 1),
        has_day_of_year=0)

    def parse_string(self, timestr, subfmts):
        # Handle trailing 'Z' for UTC time
        if timestr.endswith('Z'):
            if self.scale != 'utc':
                raise ValueError("Time input terminating in 'Z' must have "
                                 "scale='UTC'")
            timestr = timestr[:-1]
        return super().parse_string(timestr, subfmts)
```
### 36 - astropy/time/formats.py:

Start line: 513, End line: 550

```python
class TimeDecimalYear(TimeNumeric):
    """
    Time as a decimal year, with integer values corresponding to midnight
    of the first day of each year.  For example 2000.5 corresponds to the
    ISO time '2000-07-02 00:00:00'.
    """
    name = 'decimalyear'

    def set_jds(self, val1, val2):
        self._check_scale(self._scale)  # Validate scale.

        sum12, err12 = two_sum(val1, val2)
        iy_start = np.trunc(sum12).astype(int)
        extra, y_frac = two_sum(sum12, -iy_start)
        y_frac += extra + err12

        val = (val1 + val2).astype(np.double)
        iy_start = np.trunc(val).astype(int)

        imon = np.ones_like(iy_start)
        iday = np.ones_like(iy_start)
        ihr = np.zeros_like(iy_start)
        imin = np.zeros_like(iy_start)
        isec = np.zeros_like(y_frac)

        # Possible enhancement: use np.unique to only compute start, stop
        # for unique values of iy_start.
        scale = self.scale.upper().encode('ascii')
        jd1_start, jd2_start = erfa.dtf2d(scale, iy_start, imon, iday,
                                          ihr, imin, isec)
        jd1_end, jd2_end = erfa.dtf2d(scale, iy_start + 1, imon, iday,
                                      ihr, imin, isec)

        t_start = Time(jd1_start, jd2_start, scale=self.scale, format='jd')
        t_end = Time(jd1_end, jd2_end, scale=self.scale, format='jd')
        t_frac = t_start + (t_end - t_start) * y_frac

        self.jd1, self.jd2 = day_frac(t_frac.jd1, t_frac.jd2)
```
### 38 - astropy/time/core.py:

Start line: 299, End line: 323

```python
class TimeInfo(TimeInfoBase):

    def _construct_from_dict(self, map):
        # See comment above. May need to convert string back to datetime64.
        # Note that _serialize_context is not set here so we just look for the
        # string value directly.
        if (map['format'] == 'datetime64'
                and 'value' in map
                and map['value'].dtype.kind == 'U'):
            map['value'] = map['value'].astype('datetime64')

        # Convert back to datetime objects for datetime format.
        if map['format'] == 'datetime' and 'value' in map:
            from datetime import datetime
            map['value'] = np.vectorize(datetime.fromisoformat)(map['value'])

        delta_ut1_utc = map.pop('_delta_ut1_utc', None)
        delta_tdb_tt = map.pop('_delta_tdb_tt', None)

        out = super()._construct_from_dict(map)

        if delta_ut1_utc is not None:
            out._delta_ut1_utc = delta_ut1_utc
        if delta_tdb_tt is not None:
            out._delta_tdb_tt = delta_tdb_tt

        return out
```
### 39 - astropy/time/formats.py:

Start line: 398, End line: 407

```python
class TimeNumeric(TimeFormat):
    subfmts = (
        ('float', np.float64, None, np.add),
        ('long', np.longdouble, utils.longdouble_to_twoval,
         utils.twoval_to_longdouble),
        ('decimal', np.object_, utils.decimal_to_twoval,
         utils.twoval_to_decimal),
        ('str', np.str_, utils.decimal_to_twoval, utils.twoval_to_string),
        ('bytes', np.bytes_, utils.bytes_to_twoval, utils.twoval_to_bytes),
    )
```
### 40 - astropy/time/core.py:

Start line: 1158, End line: 1173

```python
class TimeBase(ShapedLikeNDArray):

    def _apply(self, method, *args, format=None, cls=None, **kwargs):
        # ... other code
        if new_format not in tm.FORMATS:
            raise ValueError(f'format must be one of {list(tm.FORMATS)}')

        NewFormat = tm.FORMATS[new_format]

        tm._time = NewFormat(
            tm._time.jd1, tm._time.jd2,
            tm._time._scale,
            precision=self.precision,
            in_subfmt=NewFormat._get_allowed_subfmt(self.in_subfmt),
            out_subfmt=NewFormat._get_allowed_subfmt(self.out_subfmt),
            from_jd=True)
        tm._format = new_format
        tm.SCALES = self.SCALES

        return tm
```
### 42 - astropy/time/formats.py:

Start line: 157, End line: 238

```python
class TimeFormat:

    @property
    def in_subfmt(self):
        return self._in_subfmt

    @in_subfmt.setter
    def in_subfmt(self, subfmt):
        # Validate subfmt value for this class, raises ValueError if not.
        self._select_subfmts(subfmt)
        self._in_subfmt = subfmt

    @property
    def out_subfmt(self):
        return self._out_subfmt

    @out_subfmt.setter
    def out_subfmt(self, subfmt):
        # Validate subfmt value for this class, raises ValueError if not.
        self._select_subfmts(subfmt)
        self._out_subfmt = subfmt

    @property
    def jd1(self):
        return self._jd1

    @jd1.setter
    def jd1(self, jd1):
        self._jd1 = _validate_jd_for_storage(jd1)
        if self._jd2 is not None:
            self._jd1, self._jd2 = _broadcast_writeable(self._jd1, self._jd2)

    @property
    def jd2(self):
        return self._jd2

    @jd2.setter
    def jd2(self, jd2):
        self._jd2 = _validate_jd_for_storage(jd2)
        if self._jd1 is not None:
            self._jd1, self._jd2 = _broadcast_writeable(self._jd1, self._jd2)

    def __len__(self):
        return len(self.jd1)

    @property
    def scale(self):
        """Time scale"""
        self._scale = self._check_scale(self._scale)
        return self._scale

    @scale.setter
    def scale(self, val):
        self._scale = val

    def mask_if_needed(self, value):
        if self.masked:
            value = np.ma.array(value, mask=self.mask, copy=False)
        return value

    @property
    def mask(self):
        if 'mask' not in self.cache:
            self.cache['mask'] = np.isnan(self.jd2)
            if self.cache['mask'].shape:
                self.cache['mask'].flags.writeable = False
        return self.cache['mask']

    @property
    def masked(self):
        if 'masked' not in self.cache:
            self.cache['masked'] = bool(np.any(self.mask))
        return self.cache['masked']

    @property
    def jd2_filled(self):
        return np.nan_to_num(self.jd2) if self.masked else self.jd2

    @lazyproperty
    def cache(self):
        """
        Return the cache associated with this instance.
        """
        return defaultdict(dict)
```
### 43 - astropy/time/formats.py:

Start line: 817, End line: 836

```python
class TimePlotDate(TimeFromEpoch):

    @lazyproperty
    def epoch(self):
        """Reference epoch time from which the time interval is measured"""
        try:
            # Matplotlib >= 3.3 has a get_epoch() function
            from matplotlib.dates import get_epoch
        except ImportError:
            # If no get_epoch() then the epoch is '0001-01-01'
            _epoch = self._epoch
        else:
            # Get the matplotlib date epoch as an ISOT string in UTC
            epoch_utc = get_epoch()
            from erfa import ErfaWarning
            with warnings.catch_warnings():
                # Catch possible dubious year warnings from erfa
                warnings.filterwarnings('ignore', category=ErfaWarning)
                _epoch = Time(epoch_utc, scale='utc', format='isot')
            _epoch.format = 'jd'

        return _epoch
```
### 44 - astropy/time/formats.py:

Start line: 861, End line: 918

```python
class TimeAstropyTime(TimeUnique):
    """
    Instantiate date from an Astropy Time object (or list thereof).

    This is purely for instantiating from a Time object.  The output
    format is the same as the first time instance.
    """
    name = 'astropy_time'

    def __new__(cls, val1, val2, scale, precision,
                in_subfmt, out_subfmt, from_jd=False):
        """
        Use __new__ instead of __init__ to output a class instance that
        is the same as the class of the first Time object in the list.
        """
        val1_0 = val1.flat[0]
        if not (isinstance(val1_0, Time) and all(type(val) is type(val1_0)
                                                 for val in val1.flat)):
            raise TypeError('Input values for {} class must all be same '
                            'astropy Time type.'.format(cls.name))

        if scale is None:
            scale = val1_0.scale

        if val1.shape:
            vals = [getattr(val, scale)._time for val in val1]
            jd1 = np.concatenate([np.atleast_1d(val.jd1) for val in vals])
            jd2 = np.concatenate([np.atleast_1d(val.jd2) for val in vals])

            # Collect individual location values and merge into a single location.
            if any(tm.location is not None for tm in val1):
                if any(tm.location is None for tm in val1):
                    raise ValueError('cannot concatenate times unless all locations '
                                     'are set or no locations are set')
                locations = []
                for tm in val1:
                    location = np.broadcast_to(tm.location, tm._time.jd1.shape,
                                               subok=True)
                    locations.append(np.atleast_1d(location))

                location = np.concatenate(locations)

            else:
                location = None
        else:
            val = getattr(val1_0, scale)._time
            jd1, jd2 = val.jd1, val.jd2
            location = val1_0.location

        OutTimeFormat = val1_0._time.__class__
        self = OutTimeFormat(jd1, jd2, scale, precision, in_subfmt, out_subfmt,
                             from_jd=True)

        # Make a temporary hidden attribute to transfer location back to the
        # parent Time object where it needs to live.
        self._location = location

        return self
```
### 45 - astropy/time/core.py:

Start line: 1815, End line: 1825

```python
class Time(TimeBase):

    def light_travel_time(self, skycoord, kind='barycentric', location=None, ephemeris=None):

        # get unit ICRS vector to star
        spos = (skycoord.icrs.represent_as(UnitSphericalRepresentation).
                represent_as(CartesianRepresentation).xyz)

        # Move X,Y,Z to last dimension, to enable possible broadcasting below.
        cpos = np.rollaxis(cpos, 0, cpos.ndim)
        spos = np.rollaxis(spos, 0, spos.ndim)

        # calculate light travel time correction
        tcor_val = (spos * cpos).sum(axis=-1) / const.c
        return TimeDelta(tcor_val, scale='tdb')
```
### 46 - astropy/time/formats.py:

Start line: 599, End line: 649

```python
class TimeFromEpoch(TimeNumeric):

    def set_jds(self, val1, val2):
        """
        Initialize the internal jd1 and jd2 attributes given val1 and val2.
        For an TimeFromEpoch subclass like TimeUnix these will be floats giving
        the effective seconds since an epoch time (e.g. 1970-01-01 00:00:00).
        """
        # Form new JDs based on epoch time + time from epoch (converted to JD).
        # One subtlety that might not be obvious is that 1.000 Julian days in
        # UTC can be 86400 or 86401 seconds.  For the TimeUnix format the
        # assumption is that every day is exactly 86400 seconds, so this is, in
        # principle, doing the math incorrectly, *except* that it matches the
        # definition of Unix time which does not include leap seconds.

        # note: use divisor=1./self.unit, since this is either 1 or 1/86400,
        # and 1/86400 is not exactly representable as a float64, so multiplying
        # by that will cause rounding errors. (But inverting it as a float64
        # recovers the exact number)
        day, frac = day_frac(val1, val2, divisor=1. / self.unit)

        jd1 = self.epoch.jd1 + day
        jd2 = self.epoch.jd2 + frac

        # For the usual case that scale is the same as epoch_scale, we only need
        # to ensure that abs(jd2) <= 0.5. Since abs(self.epoch.jd2) <= 0.5 and
        # abs(frac) <= 0.5, we can do simple (fast) checks and arithmetic here
        # without another call to day_frac(). Note also that `round(jd2.item())`
        # is about 10x faster than `np.round(jd2)`` for a scalar.
        if self.epoch.scale == self.scale:
            jd1_extra = np.round(jd2) if jd2.shape else round(jd2.item())
            jd1 += jd1_extra
            jd2 -= jd1_extra

            self.jd1, self.jd2 = jd1, jd2
            return

        # Create a temporary Time object corresponding to the new (jd1, jd2) in
        # the epoch scale (e.g. UTC for TimeUnix) then convert that to the
        # desired time scale for this object.
        #
        # A known limitation is that the transform from self.epoch_scale to
        # self.scale cannot involve any metadata like lat or lon.
        try:
            tm = getattr(Time(jd1, jd2, scale=self.epoch_scale,
                              format='jd'), self.scale)
        except Exception as err:
            raise ScaleValueError("Cannot convert from '{}' epoch scale '{}'"
                                  "to specified scale '{}', got error:\n{}"
                                  .format(self.name, self.epoch_scale,
                                          self.scale, err)) from err

        self.jd1, self.jd2 = day_frac(tm._time.jd1, tm._time.jd2)
```
### 48 - astropy/time/core.py:

Start line: 537, End line: 554

```python
class TimeBase(ShapedLikeNDArray):

    @format.setter
    def format(self, format):
        """Set time format"""
        if format not in self.FORMATS:
            raise ValueError(f'format must be one of {list(self.FORMATS)}')
        format_cls = self.FORMATS[format]

        # Get the new TimeFormat object to contain time in new format.  Possibly
        # coerce in/out_subfmt to '*' (default) if existing subfmt values are
        # not valid in the new format.
        self._time = format_cls(
            self._time.jd1, self._time.jd2,
            self._time._scale, self.precision,
            in_subfmt=format_cls._get_allowed_subfmt(self.in_subfmt),
            out_subfmt=format_cls._get_allowed_subfmt(self.out_subfmt),
            from_jd=True)

        self._format = format
```
### 49 - astropy/time/core.py:

Start line: 647, End line: 687

```python
class TimeBase(ShapedLikeNDArray):

    @property
    def precision(self):
        """
        Decimal precision when outputting seconds as floating point (int
        value between 0 and 9 inclusive).
        """
        return self._time.precision

    @precision.setter
    def precision(self, val):
        del self.cache
        if not isinstance(val, int) or val < 0 or val > 9:
            raise ValueError('precision attribute must be an int between '
                             '0 and 9')
        self._time.precision = val

    @property
    def in_subfmt(self):
        """
        Unix wildcard pattern to select subformats for parsing string input
        times.
        """
        return self._time.in_subfmt

    @in_subfmt.setter
    def in_subfmt(self, val):
        self._time.in_subfmt = val
        del self.cache

    @property
    def out_subfmt(self):
        """
        Unix wildcard pattern to select subformats for outputting times.
        """
        return self._time.out_subfmt

    @out_subfmt.setter
    def out_subfmt(self, val):
        # Setting the out_subfmt property here does validation of ``val``
        self._time.out_subfmt = val
        del self.cache
```
### 50 - astropy/time/core.py:

Start line: 588, End line: 645

```python
class TimeBase(ShapedLikeNDArray):

    def _set_scale(self, scale):
        """
        This is the key routine that actually does time scale conversions.
        This is not public and not connected to the read-only scale property.
        """

        if scale == self.scale:
            return
        if scale not in self.SCALES:
            raise ValueError("Scale {!r} is not in the allowed scales {}"
                             .format(scale, sorted(self.SCALES)))

        if scale == 'utc' or self.scale == 'utc':
            # If doing a transform involving UTC then check that the leap
            # seconds table is up to date.
            _check_leapsec()

        # Determine the chain of scale transformations to get from the current
        # scale to the new scale.  MULTI_HOPS contains a dict of all
        # transformations (xforms) that require intermediate xforms.
        # The MULTI_HOPS dict is keyed by (sys1, sys2) in alphabetical order.
        xform = (self.scale, scale)
        xform_sort = tuple(sorted(xform))
        multi = MULTI_HOPS.get(xform_sort, ())
        xforms = xform_sort[:1] + multi + xform_sort[-1:]
        # If we made the reverse xform then reverse it now.
        if xform_sort != xform:
            xforms = tuple(reversed(xforms))

        # Transform the jd1,2 pairs through the chain of scale xforms.
        jd1, jd2 = self._time.jd1, self._time.jd2_filled
        for sys1, sys2 in zip(xforms[:-1], xforms[1:]):
            # Some xforms require an additional delta_ argument that is
            # provided through Time methods.  These values may be supplied by
            # the user or computed based on available approximations.  The
            # get_delta_ methods are available for only one combination of
            # sys1, sys2 though the property applies for both xform directions.
            args = [jd1, jd2]
            for sys12 in ((sys1, sys2), (sys2, sys1)):
                dt_method = '_get_delta_{}_{}'.format(*sys12)
                try:
                    get_dt = getattr(self, dt_method)
                except AttributeError:
                    pass
                else:
                    args.append(get_dt(jd1, jd2))
                    break

            conv_func = getattr(erfa, sys1 + sys2)
            jd1, jd2 = conv_func(*args)

        jd1, jd2 = day_frac(jd1, jd2)
        if self.masked:
            jd2[self.mask] = np.nan

        self._time = self.FORMATS[self.format](jd1, jd2, scale, self.precision,
                                               self.in_subfmt, self.out_subfmt,
                                               from_jd=True)
```
### 51 - astropy/time/formats.py:

Start line: 122, End line: 141

```python
class TimeFormat:

    def __init_subclass__(cls, **kwargs):
        # Register time formats that define a name, but leave out astropy_time since
        # it is not a user-accessible format and is only used for initialization into
        # a different format.
        if 'name' in cls.__dict__ and cls.name != 'astropy_time':
            # FIXME: check here that we're not introducing a collision with
            # an existing method or attribute; problem is it could be either
            # astropy.time.Time or astropy.time.TimeDelta, and at the point
            # where this is run neither of those classes have necessarily been
            # constructed yet.
            if 'value' in cls.__dict__ and not hasattr(cls.value, "fget"):
                raise ValueError("If defined, 'value' must be a property")

            cls._registry[cls.name] = cls

        # If this class defines its own subfmts, preprocess the definitions.
        if 'subfmts' in cls.__dict__:
            cls.subfmts = _regexify_subfmts(cls.subfmts)

        return super().__init_subclass__(**kwargs)
```
### 52 - astropy/time/core.py:

Start line: 381, End line: 450

```python
class TimeBase(ShapedLikeNDArray):
    """Base time class from which Time and TimeDelta inherit."""

    # Make sure that reverse arithmetic (e.g., TimeDelta.__rmul__)
    # gets called over the __mul__ of Numpy arrays.
    __array_priority__ = 20000

    # Declare that Time can be used as a Table column by defining the
    # attribute where column attributes will be stored.
    _astropy_column_attrs = None

    def __getnewargs__(self):
        return (self._time,)

    def _init_from_vals(self, val, val2, format, scale, copy,
                        precision=None, in_subfmt=None, out_subfmt=None):
        """
        Set the internal _format, scale, and _time attrs from user
        inputs.  This handles coercion into the correct shapes and
        some basic input validation.
        """
        if precision is None:
            precision = 3
        if in_subfmt is None:
            in_subfmt = '*'
        if out_subfmt is None:
            out_subfmt = '*'

        # Coerce val into an array
        val = _make_array(val, copy)

        # If val2 is not None, ensure consistency
        if val2 is not None:
            val2 = _make_array(val2, copy)
            try:
                np.broadcast(val, val2)
            except ValueError:
                raise ValueError('Input val and val2 have inconsistent shape; '
                                 'they cannot be broadcast together.')

        if scale is not None:
            if not (isinstance(scale, str)
                    and scale.lower() in self.SCALES):
                raise ScaleValueError("Scale {!r} is not in the allowed scales "
                                      "{}".format(scale,
                                                  sorted(self.SCALES)))

        # If either of the input val, val2 are masked arrays then
        # find the masked elements and fill them.
        mask, val, val2 = _check_for_masked_and_fill(val, val2)

        # Parse / convert input values into internal jd1, jd2 based on format
        self._time = self._get_time_fmt(val, val2, format, scale,
                                        precision, in_subfmt, out_subfmt)
        self._format = self._time.name

        # Hack from #9969 to allow passing the location value that has been
        # collected by the TimeAstropyTime format class up to the Time level.
        # TODO: find a nicer way.
        if hasattr(self._time, '_location'):
            self.location = self._time._location
            del self._time._location

        # If any inputs were masked then masked jd2 accordingly.  From above
        # routine ``mask`` must be either Python bool False or an bool ndarray
        # with shape broadcastable to jd2.
        if mask is not False:
            mask = np.broadcast_to(mask, self._time.jd2.shape)
            self._time.jd1[mask] = 2451544.5  # Set to JD for 2000-01-01
            self._time.jd2[mask] = np.nan
```
### 53 - astropy/time/core.py:

Start line: 511, End line: 535

```python
class TimeBase(ShapedLikeNDArray):

    @property
    def writeable(self):
        return self._time.jd1.flags.writeable & self._time.jd2.flags.writeable

    @writeable.setter
    def writeable(self, value):
        self._time.jd1.flags.writeable = value
        self._time.jd2.flags.writeable = value

    @property
    def format(self):
        """
        Get or set time format.

        The format defines the way times are represented when accessed via the
        ``.value`` attribute.  By default it is the same as the format used for
        initializing the `Time` instance, but it can be set to any other value
        that could be used for initialization.  These can be listed with::

          >>> list(Time.FORMATS)
          ['jd', 'mjd', 'decimalyear', 'unix', 'unix_tai', 'cxcsec', 'gps', 'plot_date',
           'stardate', 'datetime', 'ymdhms', 'iso', 'isot', 'yday', 'datetime64',
           'fits', 'byear', 'jyear', 'byear_str', 'jyear_str']
        """
        return self._format
```
### 54 - astropy/time/core.py:

Start line: 2165, End line: 2175

```python
class Time(TimeBase):

    def _set_delta_tdb_tt(self, val):
        del self.cache
        if hasattr(val, 'to'):  # Matches Quantity but also TimeDelta.
            val = val.to(u.second).value
        val = self._match_shape(val)
        self._delta_tdb_tt = val

    # Note can't use @property because _get_delta_tdb_tt is explicitly
    # called with the optional jd1 and jd2 args.
    delta_tdb_tt = property(_get_delta_tdb_tt, _set_delta_tdb_tt)
    """TDB - TT time scale offset"""
```
### 55 - astropy/time/core.py:

Start line: 2114, End line: 2126

```python
class Time(TimeBase):

    def _set_delta_ut1_utc(self, val):
        del self.cache
        if hasattr(val, 'to'):  # Matches Quantity but also TimeDelta.
            val = val.to(u.second).value
        val = self._match_shape(val)
        self._delta_ut1_utc = val

    # Note can't use @property because _get_delta_tdb_tt is explicitly
    # called with the optional jd1 and jd2 args.
    delta_ut1_utc = property(_get_delta_ut1_utc, _set_delta_ut1_utc)
    """UT1 - UTC time scale offset"""

    # Property for ERFA DTR arg = TDB - TT
```
### 56 - astropy/time/formats.py:

Start line: 1672, End line: 1706

```python
class TimeFITS(TimeString):

    def parse_string(self, timestr, subfmts):
        """Read time and deprecated scale if present"""
        # Try parsing with any of the allowed sub-formats.
        for _, regex, _ in subfmts:
            tm = re.match(regex, timestr)
            if tm:
                break
        else:
            raise ValueError(f'Time {timestr} does not match {self.name} format')
        tm = tm.groupdict()
        # Scale and realization are deprecated and strings in this form
        # are no longer created.  We issue a warning but still use the value.
        if tm['scale'] is not None:
            warnings.warn("FITS time strings should no longer have embedded time scale.",
                          AstropyDeprecationWarning)
            # If a scale was given, translate from a possible deprecated
            # timescale identifier to the scale used by Time.
            fits_scale = tm['scale'].upper()
            scale = FITS_DEPRECATED_SCALES.get(fits_scale, fits_scale.lower())
            if scale not in TIME_SCALES:
                raise ValueError("Scale {!r} is not in the allowed scales {}"
                                 .format(scale, sorted(TIME_SCALES)))
            # If no scale was given in the initialiser, set the scale to
            # that given in the string.  Realization is ignored
            # and is only supported to allow old-style strings to be
            # parsed.
            if self._scale is None:
                self._scale = scale
            if scale != self.scale:
                raise ValueError("Input strings for {} class must all "
                                 "have consistent time scales."
                                 .format(self.name))
        return [int(tm['year']), int(tm['mon']), int(tm['mday']),
                int(tm.get('hour', 0)), int(tm.get('min', 0)),
                float(tm.get('sec', 0.))]
```
### 57 - astropy/time/core.py:

Start line: 2393, End line: 2425

```python
class TimeDelta(TimeBase):

    def __init__(self, val, val2=None, format=None, scale=None, copy=False):
        if isinstance(val, TimeDelta):
            if scale is not None:
                self._set_scale(scale)
        else:
            format = format or self._get_format(val)
            self._init_from_vals(val, val2, format, scale, copy)

            if scale is not None:
                self.SCALES = TIME_DELTA_TYPES[scale]

    @staticmethod
    def _get_format(val):
        if isinstance(val, timedelta):
            return 'datetime'

        if getattr(val, 'unit', None) is None:
            warn('Numerical value without unit or explicit format passed to'
                 ' TimeDelta, assuming days', TimeDeltaMissingUnitWarning)

        return 'jd'

    def replicate(self, *args, **kwargs):
        out = super().replicate(*args, **kwargs)
        out.SCALES = self.SCALES
        return out

    def to_datetime(self):
        """
        Convert to ``datetime.timedelta`` object.
        """
        tm = self.replicate(format='datetime')
        return tm._shaped_like_input(tm._time.value)
```
### 58 - astropy/time/core.py:

Start line: 742, End line: 765

```python
class TimeBase(ShapedLikeNDArray):

    def _shaped_like_input(self, value):
        if self._time.jd1.shape:
            if isinstance(value, np.ndarray):
                return value
            else:
                raise TypeError(
                    f"JD is an array ({self._time.jd1!r}) but value "
                    f"is not ({value!r})")
        else:
            # zero-dimensional array, is it safe to unbox?
            if (isinstance(value, np.ndarray)
                    and not value.shape
                    and not np.ma.is_masked(value)):
                if value.dtype.kind == 'M':
                    # existing test doesn't want datetime64 converted
                    return value[()]
                elif value.dtype.fields:
                    # Unpack but keep field names; .item() doesn't
                    # Still don't get python types in the fields
                    return value[()]
                else:
                    return value.item()
            else:
                return value
```
### 59 - astropy/time/formats.py:

Start line: 1867, End line: 1892

```python
class TimeDeltaDatetime(TimeDeltaFormat, TimeUnique):
    """Time delta in datetime.timedelta"""
    name = 'datetime'

    def _check_val_type(self, val1, val2):
        if not all(isinstance(val, datetime.timedelta) for val in val1.flat):
            raise TypeError('Input values for {} class must be '
                            'datetime.timedelta objects'.format(self.name))
        if val2 is not None:
            raise ValueError(
                f'{self.name} objects do not accept a val2 but you provided {val2}')
        return val1, None

    def set_jds(self, val1, val2):
        self._check_scale(self._scale)  # Validate scale.
        iterator = np.nditer([val1, None, None],
                             flags=['refs_ok', 'zerosize_ok'],
                             op_dtypes=[None, np.double, np.double])

        day = datetime.timedelta(days=1)
        for val, jd1, jd2 in iterator:
            jd1[...], other = divmod(val.item(), day)
            jd2[...] = other / day

        self.jd1, self.jd2 = day_frac(iterator.operands[-2],
                                      iterator.operands[-1])
```
### 60 - astropy/time/core.py:

Start line: 208, End line: 267

```python
class TimeInfoBase(MixinInfo):

    def new_like(self, cols, length, metadata_conflicts='warn', name=None):
        """
        Return a new Time instance which is consistent with the input Time objects
        ``cols`` and has ``length`` rows.

        This is intended for creating an empty Time instance whose elements can
        be set in-place for table operations like join or vstack.  It checks
        that the input locations and attributes are consistent.  This is used
        when a Time object is used as a mixin column in an astropy Table.

        Parameters
        ----------
        cols : list
            List of input columns (Time objects)
        length : int
            Length of the output column object
        metadata_conflicts : str ('warn'|'error'|'silent')
            How to handle metadata conflicts
        name : str
            Output column name

        Returns
        -------
        col : Time (or subclass)
            Empty instance of this class consistent with ``cols``

        """
        # Get merged info attributes like shape, dtype, format, description, etc.
        attrs = self.merge_cols_attributes(cols, metadata_conflicts, name,
                                           ('meta', 'description'))
        attrs.pop('dtype')  # Not relevant for Time
        col0 = cols[0]

        # Check that location is consistent for all Time objects
        for col in cols[1:]:
            # This is the method used by __setitem__ to ensure that the right side
            # has a consistent location (and coerce data if necessary, but that does
            # not happen in this case since `col` is already a Time object).  If this
            # passes then any subsequent table operations via setitem will work.
            try:
                col0._make_value_equivalent(slice(None), col)
            except ValueError:
                raise ValueError('input columns have inconsistent locations')

        # Make a new Time object with the desired shape and attributes
        shape = (length,) + attrs.pop('shape')
        jd2000 = 2451544.5  # Arbitrary JD value J2000.0 that will work with ERFA
        jd1 = np.full(shape, jd2000, dtype='f8')
        jd2 = np.zeros(shape, dtype='f8')
        tm_attrs = {attr: getattr(col0, attr)
                    for attr in ('scale', 'location',
                                 'precision', 'in_subfmt', 'out_subfmt')}
        out = self._parent_cls(jd1, jd2, format='jd', **tm_attrs)
        out.format = col0.format

        # Set remaining info attributes
        for attr, value in attrs.items():
            setattr(out.info, attr, value)

        return out
```
### 61 - astropy/time/core.py:

Start line: 189, End line: 206

```python
class TimeInfoBase(MixinInfo):

    def _construct_from_dict(self, map):
        if 'jd1' in map and 'jd2' in map:
            # Initialize as JD but revert to desired format and out_subfmt (if needed)
            format = map.pop('format')
            out_subfmt = map.pop('out_subfmt', None)
            map['format'] = 'jd'
            map['val'] = map.pop('jd1')
            map['val2'] = map.pop('jd2')
            out = self._parent_cls(**map)
            out.format = format
            if out_subfmt is not None:
                out.out_subfmt = out_subfmt

        else:
            map['val'] = map.pop('value')
            out = self._parent_cls(**map)

        return out
```
### 62 - astropy/time/formats.py:

Start line: 240, End line: 296

```python
class TimeFormat:

    def _check_val_type(self, val1, val2):
        """Input value validation, typically overridden by derived classes"""
        # val1 cannot contain nan, but val2 can contain nan
        isfinite1 = np.isfinite(val1)
        if val1.size > 1:  # Calling .all() on a scalar is surprisingly slow
            isfinite1 = isfinite1.all()  # Note: arr.all() about 3x faster than np.all(arr)
        elif val1.size == 0:
            isfinite1 = False
        ok1 = (val1.dtype.kind == 'f' and val1.dtype.itemsize >= 8
               and isfinite1 or val1.size == 0)
        ok2 = val2 is None or (
            val2.dtype.kind == 'f' and val2.dtype.itemsize >= 8
            and not np.any(np.isinf(val2))) or val2.size == 0
        if not (ok1 and ok2):
            raise TypeError('Input values for {} class must be finite doubles'
                            .format(self.name))

        if getattr(val1, 'unit', None) is not None:
            # Convert any quantity-likes to days first, attempting to be
            # careful with the conversion, so that, e.g., large numbers of
            # seconds get converted without losing precision because
            # 1/86400 is not exactly representable as a float.
            val1 = u.Quantity(val1, copy=False)
            if val2 is not None:
                val2 = u.Quantity(val2, copy=False)

            try:
                val1, val2 = quantity_day_frac(val1, val2)
            except u.UnitsError:
                raise u.UnitConversionError(
                    "only quantities with time units can be "
                    "used to instantiate Time instances.")
            # We now have days, but the format may expect another unit.
            # On purpose, multiply with 1./day_unit because typically it is
            # 1./erfa.DAYSEC, and inverting it recovers the integer.
            # (This conversion will get undone in format's set_jds, hence
            # there may be room for optimizing this.)
            factor = 1. / getattr(self, 'unit', 1.)
            if factor != 1.:
                val1, carry = two_product(val1, factor)
                carry += val2 * factor
                val1, val2 = two_sum(val1, carry)

        elif getattr(val2, 'unit', None) is not None:
            raise TypeError('Cannot mix float and Quantity inputs')

        if val2 is None:
            val2 = np.array(0, dtype=val1.dtype)

        def asarray_or_scalar(val):
            """
            Remove ndarray subclasses since for jd1/jd2 we want a pure ndarray
            or a Python or numpy scalar.
            """
            return np.asarray(val) if isinstance(val, np.ndarray) else val

        return asarray_or_scalar(val1), asarray_or_scalar(val2)
```
### 63 - astropy/time/formats.py:

Start line: 79, End line: 120

```python
class TimeFormat:
    """
    Base class for time representations.

    Parameters
    ----------
    val1 : numpy ndarray, list, number, str, or bytes
        Values to initialize the time or times.  Bytes are decoded as ascii.
    val2 : numpy ndarray, list, or number; optional
        Value(s) to initialize the time or times.  Only used for numerical
        input, to help preserve precision.
    scale : str
        Time scale of input value(s)
    precision : int
        Precision for seconds as floating point
    in_subfmt : str
        Select subformat for inputting string times
    out_subfmt : str
        Select subformat for outputting string times
    from_jd : bool
        If true then val1, val2 are jd1, jd2
    """

    _default_scale = 'utc'  # As of astropy 0.4
    subfmts = ()
    _registry = TIME_FORMATS

    def __init__(self, val1, val2, scale, precision,
                 in_subfmt, out_subfmt, from_jd=False):
        self.scale = scale  # validation of scale done later with _check_scale
        self.precision = precision
        self.in_subfmt = in_subfmt
        self.out_subfmt = out_subfmt

        self._jd1, self._jd2 = None, None

        if from_jd:
            self.jd1 = val1
            self.jd2 = val2
        else:
            val1, val2 = self._check_val_type(val1, val2)
            self.set_jds(val1, val2)
```
### 66 - astropy/time/formats.py:

Start line: 448, End line: 474

```python
class TimeNumeric(TimeFormat):

    def to_value(self, jd1=None, jd2=None, parent=None, out_subfmt=None):
        """
        Return time representation from internal jd1 and jd2.
        Subclasses that require ``parent`` or to adjust the jds should
        override this method.
        """
        # TODO: do this in __init_subclass__?
        if self.__class__.value.fget is not self.__class__.to_value:
            return self.value

        if jd1 is None:
            jd1 = self.jd1
        if jd2 is None:
            jd2 = self.jd2
        if out_subfmt is None:
            out_subfmt = self.out_subfmt
        subfmt = self._select_subfmts(out_subfmt)[0]
        kwargs = {}
        if subfmt[0] in ('str', 'bytes'):
            unit = getattr(self, 'unit', 1)
            digits = int(np.ceil(np.log10(unit / np.finfo(float).eps)))
            # TODO: allow a way to override the format.
            kwargs['fmt'] = f'.{digits}f'
        value = subfmt[3](jd1, jd2, **kwargs)
        return self.mask_if_needed(value)

    value = property(to_value)
```
### 68 - astropy/time/core.py:

Start line: 2279, End line: 2321

```python
class Time(TimeBase):

    # Reverse addition is possible: <something-Tdelta-ish> + T
    # but there is no case of <something> - T, so no __rsub__.
    def __radd__(self, other):
        return self.__add__(other)

    def __array_function__(self, function, types, args, kwargs):
        """
        Wrap numpy functions.

        Parameters
        ----------
        function : callable
            Numpy function to wrap
        types : iterable of classes
            Classes that provide an ``__array_function__`` override. Can
            in principle be used to interact with other classes. Below,
            mostly passed on to `~numpy.ndarray`, which can only interact
            with subclasses.
        args : tuple
            Positional arguments provided in the function call.
        kwargs : dict
            Keyword arguments provided in the function call.
        """
        if function in CUSTOM_FUNCTIONS:
            f = CUSTOM_FUNCTIONS[function]
            return f(*args, **kwargs)
        elif function in UNSUPPORTED_FUNCTIONS:
            return NotImplemented
        else:
            return super().__array_function__(function, types, args, kwargs)

    def to_datetime(self, timezone=None):
        # TODO: this could likely go through to_value, as long as that
        # had an **kwargs part that was just passed on to _time.
        tm = self.replicate(format='datetime')
        return tm._shaped_like_input(tm._time.to_value(timezone))

    to_datetime.__doc__ = TimeDatetime.to_value.__doc__


class TimeDeltaMissingUnitWarning(AstropyDeprecationWarning):
    """Warning for missing unit or format in TimeDelta"""
    pass
```
### 70 - astropy/time/formats.py:

Start line: 1423, End line: 1448

```python
class TimeString(TimeUnique):

    def format_string(self, str_fmt, **kwargs):
        """Write time to a string using a given format.

        By default, just interprets str_fmt as a format string,
        but subclasses can add to this.
        """
        return str_fmt.format(**kwargs)

    @property
    def value(self):
        # Select the first available subformat based on current
        # self.out_subfmt
        subfmts = self._select_subfmts(self.out_subfmt)
        _, _, str_fmt = subfmts[0]

        # TODO: fix this ugly hack
        if self.precision > 0 and str_fmt.endswith('{sec:02d}'):
            str_fmt += '.{fracsec:0' + str(self.precision) + 'd}'

        # Try to optimize this later.  Can't pre-allocate because length of
        # output could change, e.g. year rolls from 999 to 1000.
        outs = []
        for kwargs in self.str_kwargs():
            outs.append(str(self.format_string(str_fmt, **kwargs)))

        return np.array(outs).reshape(self.jd1.shape)
```
### 72 - astropy/time/core.py:

Start line: 2017, End line: 2076

```python
class Time(TimeBase):

    def _call_erfa(self, function, scales):
        # TODO: allow erfa functions to be used on Time with __array_ufunc__.
        erfa_parameters = [getattr(getattr(self, scale)._time, jd_part)
                           for scale in scales
                           for jd_part in ('jd1', 'jd2_filled')]

        result = function(*erfa_parameters)

        if self.masked:
            result[self.mask] = np.nan

        return result

    def get_delta_ut1_utc(self, iers_table=None, return_status=False):
        """Find UT1 - UTC differences by interpolating in IERS Table.

        Parameters
        ----------
        iers_table : `~astropy.utils.iers.IERS`, optional
            Table containing UT1-UTC differences from IERS Bulletins A
            and/or B.  Default: `~astropy.utils.iers.earth_orientation_table`
            (which in turn defaults to the combined version provided by
            `~astropy.utils.iers.IERS_Auto`).
        return_status : bool
            Whether to return status values.  If `False` (default), iers
            raises `IndexError` if any time is out of the range
            covered by the IERS table.

        Returns
        -------
        ut1_utc : float or float array
            UT1-UTC, interpolated in IERS Table
        status : int or int array
            Status values (if ``return_status=`True```)::
            ``astropy.utils.iers.FROM_IERS_B``
            ``astropy.utils.iers.FROM_IERS_A``
            ``astropy.utils.iers.FROM_IERS_A_PREDICTION``
            ``astropy.utils.iers.TIME_BEFORE_IERS_RANGE``
            ``astropy.utils.iers.TIME_BEYOND_IERS_RANGE``

        Notes
        -----
        In normal usage, UT1-UTC differences are calculated automatically
        on the first instance ut1 is needed.

        Examples
        --------
        To check in code whether any times are before the IERS table range::

            >>> from astropy.utils.iers import TIME_BEFORE_IERS_RANGE
            >>> t = Time(['1961-01-01', '2000-01-01'], scale='utc')
            >>> delta, status = t.get_delta_ut1_utc(return_status=True)  # doctest: +REMOTE_DATA
            >>> status == TIME_BEFORE_IERS_RANGE  # doctest: +REMOTE_DATA
            array([ True, False]...)
        """
        if iers_table is None:
            from astropy.utils.iers import earth_orientation_table
            iers_table = earth_orientation_table.get()

        return iers_table.ut1_utc(self.utc, return_status=return_status)
```
### 73 - astropy/time/formats.py:

Start line: 491, End line: 510

```python
class TimeMJD(TimeNumeric):
    """
    Modified Julian Date time format.
    This represents the number of days since midnight on November 17, 1858.
    For example, 51544.0 in MJD is midnight on January 1, 2000.
    """
    name = 'mjd'

    def set_jds(self, val1, val2):
        self._check_scale(self._scale)  # Validate scale.
        jd1, jd2 = day_frac(val1, val2)
        jd1 += erfa.DJM0  # erfa.DJM0=2400000.5 (from erfam.h).
        self.jd1, self.jd2 = day_frac(jd1, jd2)

    def to_value(self, **kwargs):
        jd1 = self.jd1 - erfa.DJM0  # This cannot lose precision.
        jd2 = self.jd2
        return super().to_value(jd1=jd1, jd2=jd2, **kwargs)

    value = property(to_value)
```
### 74 - astropy/time/core.py:

Start line: 950, End line: 978

```python
class TimeBase(ShapedLikeNDArray):

    def __setitem__(self, item, value):
        if not self.writeable:
            if self.shape:
                raise ValueError('{} object is read-only. Make a '
                                 'copy() or set "writeable" attribute to True.'
                                 .format(self.__class__.__name__))
            else:
                raise ValueError('scalar {} object is read-only.'
                                 .format(self.__class__.__name__))

        # Any use of setitem results in immediate cache invalidation
        del self.cache

        # Setting invalidates transform deltas
        for attr in ('_delta_tdb_tt', '_delta_ut1_utc'):
            if hasattr(self, attr):
                delattr(self, attr)

        if value is np.ma.masked or value is np.nan:
            self._time.jd2[item] = np.nan
            return

        value = self._make_value_equivalent(item, value)

        # Finally directly set the jd1/2 values.  Locations are known to match.
        if self.scale is not None:
            value = getattr(value, self.scale)
        self._time.jd1[item] = value._time.jd1
        self._time.jd2[item] = value._time.jd2
```
### 75 - astropy/time/core.py:

Start line: 556, End line: 586

```python
class TimeBase(ShapedLikeNDArray):

    def __repr__(self):
        return ("<{} object: scale='{}' format='{}' value={}>"
                .format(self.__class__.__name__, self.scale, self.format,
                        getattr(self, self.format)))

    def __str__(self):
        return str(getattr(self, self.format))

    def __hash__(self):

        try:
            loc = getattr(self, 'location', None)
            if loc is not None:
                loc = loc.x.to_value(u.m), loc.y.to_value(u.m), loc.z.to_value(u.m)

            return hash((self.jd1, self.jd2, self.scale, loc))

        except TypeError:
            if self.ndim != 0:
                reason = '(must be scalar)'
            elif self.masked:
                reason = '(value is masked)'
            else:
                raise

            raise TypeError(f"unhashable type: '{self.__class__.__name__}' {reason}")

    @property
    def scale(self):
        """Time scale"""
        return self._time.scale
```
### 77 - astropy/time/core.py:

Start line: 2078, End line: 2112

```python
class Time(TimeBase):

    # Property for ERFA DUT arg = UT1 - UTC
    def _get_delta_ut1_utc(self, jd1=None, jd2=None):
        """
        Get ERFA DUT arg = UT1 - UTC.  This getter takes optional jd1 and
        jd2 args because it gets called that way when converting time scales.
        If delta_ut1_utc is not yet set, this will interpolate them from the
        the IERS table.
        """
        # Sec. 4.3.1: the arg DUT is the quantity delta_UT1 = UT1 - UTC in
        # seconds. It is obtained from tables published by the IERS.
        if not hasattr(self, '_delta_ut1_utc'):
            from astropy.utils.iers import earth_orientation_table
            iers_table = earth_orientation_table.get()
            # jd1, jd2 are normally set (see above), except if delta_ut1_utc
            # is access directly; ensure we behave as expected for that case
            if jd1 is None:
                self_utc = self.utc
                jd1, jd2 = self_utc._time.jd1, self_utc._time.jd2_filled
                scale = 'utc'
            else:
                scale = self.scale
            # interpolate UT1-UTC in IERS table
            delta = iers_table.ut1_utc(jd1, jd2)
            # if we interpolated using UT1 jds, we may be off by one
            # second near leap seconds (and very slightly off elsewhere)
            if scale == 'ut1':
                # calculate UTC using the offset we got; the ERFA routine
                # is tolerant of leap seconds, so will do this right
                jd1_utc, jd2_utc = erfa.ut1utc(jd1, jd2, delta.to_value(u.s))
                # calculate a better estimate using the nearly correct UTC
                delta = iers_table.ut1_utc(jd1_utc, jd2_utc)

            self._set_delta_ut1_utc(delta)

        return self._delta_ut1_utc
```
### 78 - astropy/time/core.py:

Start line: 1374, End line: 1407

```python
class TimeBase(ShapedLikeNDArray):

    def __getattr__(self, attr):
        """
        Get dynamic attributes to output format or do timescale conversion.
        """
        if attr in self.SCALES and self.scale is not None:
            cache = self.cache['scale']
            if attr not in cache:
                if attr == self.scale:
                    tm = self
                else:
                    tm = self.replicate()
                    tm._set_scale(attr)
                    if tm.shape:
                        # Prevent future modification of cached array-like object
                        tm.writeable = False
                cache[attr] = tm
            return cache[attr]

        elif attr in self.FORMATS:
            return self.to_value(attr, subfmt=None)

        elif attr in TIME_SCALES:  # allowed ones done above (self.SCALES)
            if self.scale is None:
                raise ScaleValueError("Cannot convert TimeDelta with "
                                      "undefined scale to any defined scale.")
            else:
                raise ScaleValueError("Cannot convert {} with scale "
                                      "'{}' to scale '{}'"
                                      .format(self.__class__.__name__,
                                              self.scale, attr))

        else:
            # Should raise AttributeError
            return self.__getattribute__(attr)
```
### 80 - astropy/time/formats.py:

Start line: 704, End line: 753

```python
class TimeUnixTai(TimeUnix):
    """
    Unix time (TAI): SI seconds elapsed since 1970-01-01 00:00:00 TAI (see caveats).

    This will generally differ from standard (UTC) Unix time by the cumulative
    integral number of leap seconds introduced into UTC since 1972-01-01 UTC
    plus the initial offset of 10 seconds at that date.

    This convention matches the definition of linux CLOCK_TAI
    (https://www.cl.cam.ac.uk/~mgk25/posix-clocks.html),
    and the Precision Time Protocol
    (https://en.wikipedia.org/wiki/Precision_Time_Protocol), which
    is also used by the White Rabbit protocol in High Energy Physics:
    https://white-rabbit.web.cern.ch.

    Caveats:

    - Before 1972, fractional adjustments to UTC were made, so the difference
      between ``unix`` and ``unix_tai`` time is no longer an integer.
    - Because of the fractional adjustments, to be very precise, ``unix_tai``
      is the number of seconds since ``1970-01-01 00:00:00 TAI`` or equivalently
      ``1969-12-31 23:59:51.999918 UTC``.  The difference between TAI and UTC
      at that epoch was 8.000082 sec.
    - On the day of a positive leap second the difference between ``unix`` and
      ``unix_tai`` times increases linearly through the day by 1.0. See also the
      documentation for the `~astropy.time.TimeUnix` class.
    - Negative leap seconds are possible, though none have been needed to date.

    Examples
    --------

      >>> # get the current offset between TAI and UTC
      >>> from astropy.time import Time
      >>> t = Time('2020-01-01', scale='utc')
      >>> t.unix_tai - t.unix
      37.0

      >>> # Before 1972, the offset between TAI and UTC was not integer
      >>> t = Time('1970-01-01', scale='utc')
      >>> t.unix_tai - t.unix  # doctest: +FLOAT_CMP
      8.000082

      >>> # Initial offset of 10 seconds in 1972
      >>> t = Time('1972-01-01', scale='utc')
      >>> t.unix_tai - t.unix
      10.0
    """
    name = 'unix_tai'
    epoch_val = '1970-01-01 00:00:00'
    epoch_scale = 'tai'
```
### 81 - astropy/time/formats.py:

Start line: 921, End line: 944

```python
class TimeDatetime(TimeUnique):
    """
    Represent date as Python standard library `~datetime.datetime` object

    Example::

      >>> from astropy.time import Time
      >>> from datetime import datetime
      >>> t = Time(datetime(2000, 1, 2, 12, 0, 0), scale='utc')
      >>> t.iso
      '2000-01-02 12:00:00.000'
      >>> t.tt.datetime
      datetime.datetime(2000, 1, 2, 12, 1, 4, 184000)
    """
    name = 'datetime'

    def _check_val_type(self, val1, val2):
        if not all(isinstance(val, datetime.datetime) for val in val1.flat):
            raise TypeError('Input values for {} class must be '
                            'datetime objects'.format(self.name))
        if val2 is not None:
            raise ValueError(
                f'{self.name} objects do not accept a val2 but you provided {val2}')
        return val1, None
```
### 83 - astropy/time/core.py:

Start line: 1, End line: 79

```python
# -*- coding: utf-8 -*-

import os
import copy
import enum
import operator
import threading
from datetime import datetime, date, timedelta
from time import strftime
from warnings import warn

import numpy as np
import erfa

from astropy import units as u, constants as const
from astropy.units import UnitConversionError
from astropy.utils import ShapedLikeNDArray
from astropy.utils.compat.misc import override__dir__
from astropy.utils.data_info import MixinInfo, data_info_factory
from astropy.utils.exceptions import AstropyDeprecationWarning, AstropyWarning
from .utils import day_frac
from .formats import (TIME_FORMATS, TIME_DELTA_FORMATS,
                      TimeJD, TimeUnique, TimeAstropyTime, TimeDatetime)
# Import TimeFromEpoch to avoid breaking code that followed the old example of
# making a custom timescale in the documentation.
from .formats import TimeFromEpoch  # noqa
from .time_helper.function_helpers import CUSTOM_FUNCTIONS, UNSUPPORTED_FUNCTIONS

from astropy.extern import _strptime

__all__ = ['TimeBase', 'Time', 'TimeDelta', 'TimeInfo', 'TimeInfoBase', 'update_leap_seconds',
           'TIME_SCALES', 'STANDARD_TIME_SCALES', 'TIME_DELTA_SCALES',
           'ScaleValueError', 'OperandTypeError', 'TimeDeltaMissingUnitWarning']


STANDARD_TIME_SCALES = ('tai', 'tcb', 'tcg', 'tdb', 'tt', 'ut1', 'utc')
LOCAL_SCALES = ('local',)
TIME_TYPES = dict((scale, scales) for scales in (STANDARD_TIME_SCALES, LOCAL_SCALES)
                  for scale in scales)
TIME_SCALES = STANDARD_TIME_SCALES + LOCAL_SCALES
MULTI_HOPS = {('tai', 'tcb'): ('tt', 'tdb'),
              ('tai', 'tcg'): ('tt',),
              ('tai', 'ut1'): ('utc',),
              ('tai', 'tdb'): ('tt',),
              ('tcb', 'tcg'): ('tdb', 'tt'),
              ('tcb', 'tt'): ('tdb',),
              ('tcb', 'ut1'): ('tdb', 'tt', 'tai', 'utc'),
              ('tcb', 'utc'): ('tdb', 'tt', 'tai'),
              ('tcg', 'tdb'): ('tt',),
              ('tcg', 'ut1'): ('tt', 'tai', 'utc'),
              ('tcg', 'utc'): ('tt', 'tai'),
              ('tdb', 'ut1'): ('tt', 'tai', 'utc'),
              ('tdb', 'utc'): ('tt', 'tai'),
              ('tt', 'ut1'): ('tai', 'utc'),
              ('tt', 'utc'): ('tai',),
              }
GEOCENTRIC_SCALES = ('tai', 'tt', 'tcg')
BARYCENTRIC_SCALES = ('tcb', 'tdb')
ROTATIONAL_SCALES = ('ut1',)
TIME_DELTA_TYPES = dict((scale, scales)
                        for scales in (GEOCENTRIC_SCALES, BARYCENTRIC_SCALES,
                                       ROTATIONAL_SCALES, LOCAL_SCALES) for scale in scales)
TIME_DELTA_SCALES = GEOCENTRIC_SCALES + BARYCENTRIC_SCALES + ROTATIONAL_SCALES + LOCAL_SCALES
# For time scale changes, we need L_G and L_B, which are stored in erfam.h as
#   /* L_G = 1 - d(TT)/d(TCG) */
#   define ERFA_ELG (6.969290134e-10)
#   /* L_B = 1 - d(TDB)/d(TCB), and TDB (s) at TAI 1977/1/1.0 */
#   define ERFA_ELB (1.550519768e-8)
# These are exposed in erfa as erfa.ELG and erfa.ELB.
# Implied: d(TT)/d(TCG) = 1-L_G
# and      d(TCG)/d(TT) = 1/(1-L_G) = 1 + (1-(1-L_G))/(1-L_G) = 1 + L_G/(1-L_G)
# scale offsets as second = first + first * scale_offset[(first,second)]
```
### 85 - astropy/time/formats.py:

Start line: 839, End line: 858

```python
class TimeStardate(TimeFromEpoch):
    """
    Stardate: date units from 2318-07-05 12:00:00 UTC.
    For example, stardate 41153.7 is 00:52 on April 30, 2363.
    See http://trekguide.com/Stardates.htm#TNG for calculations and reference points
    """
    name = 'stardate'
    unit = 0.397766856  # Stardate units per day
    epoch_val = '2318-07-05 11:00:00'  # Date and time of stardate 00000.00
    epoch_val2 = None
    epoch_scale = 'tai'
    epoch_format = 'iso'


class TimeUnique(TimeFormat):
    """
    Base class for time formats that can uniquely create a time object
    without requiring an explicit format specifier.  This class does
    nothing but provide inheritance to identify a class as unique.
    """
```
### 87 - astropy/time/formats.py:

Start line: 579, End line: 597

```python
class TimeFromEpoch(TimeNumeric):
    """
    Base class for times that represent the interval from a particular
    epoch as a floating point multiple of a unit time interval (e.g. seconds
    or days).
    """

    @classproperty(lazy=True)
    def _epoch(cls):
        # Ideally we would use `def epoch(cls)` here and not have the instance
        # property below. However, this breaks the sphinx API docs generation
        # in a way that was not resolved. See #10406 for details.
        return Time(cls.epoch_val, cls.epoch_val2, scale=cls.epoch_scale,
                    format=cls.epoch_format)

    @property
    def epoch(self):
        """Reference epoch time from which the time interval is measured"""
        return self._epoch
```
### 89 - astropy/time/formats.py:

Start line: 409, End line: 446

```python
class TimeNumeric(TimeFormat):

    def _check_val_type(self, val1, val2):
        """Input value validation, typically overridden by derived classes"""
        # Save original state of val2 because the super()._check_val_type below
        # may change val2 from None to np.array(0). The value is saved in order
        # to prevent a useless and slow call to np.result_type() below in the
        # most common use-case of providing only val1.
        orig_val2_is_none = val2 is None

        if val1.dtype.kind == 'f':
            val1, val2 = super()._check_val_type(val1, val2)
        elif (not orig_val2_is_none
              or not (val1.dtype.kind in 'US'
                      or (val1.dtype.kind == 'O'
                          and all(isinstance(v, Decimal) for v in val1.flat)))):
            raise TypeError(
                'for {} class, input should be doubles, string, or Decimal, '
                'and second values are only allowed for doubles.'
                .format(self.name))

        val_dtype = (val1.dtype if orig_val2_is_none else
                     np.result_type(val1.dtype, val2.dtype))
        subfmts = self._select_subfmts(self.in_subfmt)
        for subfmt, dtype, convert, _ in subfmts:
            if np.issubdtype(val_dtype, dtype):
                break
        else:
            raise ValueError('input type not among selected sub-formats.')

        if convert is not None:
            try:
                val1, val2 = convert(val1, val2)
            except Exception:
                raise TypeError(
                    'for {} class, input should be (long) doubles, string, '
                    'or Decimal, and second values are only allowed for '
                    '(long) doubles.'.format(self.name))

        return val1, val2
```
### 90 - astropy/time/formats.py:

Start line: 298, End line: 326

```python
class TimeFormat:

    def _check_scale(self, scale):
        """
        Return a validated scale value.

        If there is a class attribute 'scale' then that defines the default /
        required time scale for this format.  In this case if a scale value was
        provided that needs to match the class default, otherwise return
        the class default.

        Otherwise just make sure that scale is in the allowed list of
        scales.  Provide a different error message if `None` (no value) was
        supplied.
        """
        if scale is None:
            scale = self._default_scale

        if scale not in TIME_SCALES:
            raise ScaleValueError("Scale value '{}' not in "
                                  "allowed values {}"
                                  .format(scale, TIME_SCALES))

        return scale

    def set_jds(self, val1, val2):
        """
        Set internal jd1 and jd2 from val1 and val2.  Must be provided
        by derived classes.
        """
        raise NotImplementedError
```
### 91 - astropy/time/formats.py:

Start line: 477, End line: 488

```python
class TimeJD(TimeNumeric):
    """
    Julian Date time format.
    This represents the number of days since the beginning of
    the Julian Period.
    For example, 2451544.5 in JD is midnight on January 1, 2000.
    """
    name = 'jd'

    def set_jds(self, val1, val2):
        self._check_scale(self._scale)  # Validate scale.
        self.jd1, self.jd2 = day_frac(val1, val2)
```
### 92 - astropy/time/formats.py:

Start line: 1740, End line: 1761

```python
class TimeBesselianEpoch(TimeEpochDate):
    """Besselian Epoch year as floating point value(s) like 1950.0"""
    name = 'byear'
    epoch_to_jd = 'epb2jd'
    jd_to_epoch = 'epb'

    def _check_val_type(self, val1, val2):
        """Input value validation, typically overridden by derived classes"""
        if hasattr(val1, 'to') and hasattr(val1, 'unit') and val1.unit is not None:
            raise ValueError("Cannot use Quantities for 'byear' format, "
                             "as the interpretation would be ambiguous. "
                             "Use float with Besselian year instead. ")
        # FIXME: is val2 really okay here?
        return super()._check_val_type(val1, val2)


class TimeJulianEpoch(TimeEpochDate):
    """Julian Epoch year as floating point value(s) like 2000.0"""
    name = 'jyear'
    unit = erfa.DJY  # 365.25, the Julian year, for conversion to quantities
    epoch_to_jd = 'epj2jd'
    jd_to_epoch = 'epj'
```
### 93 - astropy/time/core.py:

Start line: 1019, End line: 1042

```python
class TimeBase(ShapedLikeNDArray):

    def copy(self, format=None):
        """
        Return a fully independent copy the Time object, optionally changing
        the format.

        If ``format`` is supplied then the time format of the returned Time
        object will be set accordingly, otherwise it will be unchanged from the
        original.

        In this method a full copy of the internal time arrays will be made.
        The internal time arrays are normally not changeable by the user so in
        most cases the ``replicate()`` method should be used.

        Parameters
        ----------
        format : str, optional
            Time format of the copy.

        Returns
        -------
        tm : Time object
            Copy of this object
        """
        return self._apply('copy', format=format)
```
### 95 - astropy/time/formats.py:

Start line: 969, End line: 1022

```python
class TimeDatetime(TimeUnique):

    def to_value(self, timezone=None, parent=None, out_subfmt=None):
        """
        Convert to (potentially timezone-aware) `~datetime.datetime` object.

        If ``timezone`` is not ``None``, return a timezone-aware datetime
        object.

        Parameters
        ----------
        timezone : {`~datetime.tzinfo`, None}, optional
            If not `None`, return timezone-aware datetime.

        Returns
        -------
        `~datetime.datetime`
            If ``timezone`` is not ``None``, output will be timezone-aware.
        """
        if out_subfmt is not None:
            # Out_subfmt not allowed for this format, so raise the standard
            # exception by trying to validate the value.
            self._select_subfmts(out_subfmt)

        if timezone is not None:
            if self._scale != 'utc':
                raise ScaleValueError("scale is {}, must be 'utc' when timezone "
                                      "is supplied.".format(self._scale))

        # Rather than define a value property directly, we have a function,
        # since we want to be able to pass in timezone information.
        scale = self.scale.upper().encode('ascii')
        iys, ims, ids, ihmsfs = erfa.d2dtf(scale, 6,  # 6 for microsec
                                           self.jd1, self.jd2_filled)
        ihrs = ihmsfs['h']
        imins = ihmsfs['m']
        isecs = ihmsfs['s']
        ifracs = ihmsfs['f']
        iterator = np.nditer([iys, ims, ids, ihrs, imins, isecs, ifracs, None],
                             flags=['refs_ok', 'zerosize_ok'],
                             op_dtypes=7*[None] + [object])

        for iy, im, id, ihr, imin, isec, ifracsec, out in iterator:
            if isec >= 60:
                raise ValueError('Time {} is within a leap second but datetime '
                                 'does not support leap seconds'
                                 .format((iy, im, id, ihr, imin, isec, ifracsec)))
            if timezone is not None:
                out[...] = datetime.datetime(iy, im, id, ihr, imin, isec, ifracsec,
                                             tzinfo=TimezoneInfo()).astimezone(timezone)
            else:
                out[...] = datetime.datetime(iy, im, id, ihr, imin, isec, ifracsec)

        return self.mask_if_needed(iterator.operands[-1])

    value = property(to_value)
```
### 96 - astropy/time/core.py:

Start line: 2427, End line: 2450

```python
class TimeDelta(TimeBase):

    def _set_scale(self, scale):
        """
        This is the key routine that actually does time scale conversions.
        This is not public and not connected to the read-only scale property.
        """

        if scale == self.scale:
            return
        if scale not in self.SCALES:
            raise ValueError("Scale {!r} is not in the allowed scales {}"
                             .format(scale, sorted(self.SCALES)))

        # For TimeDelta, there can only be a change in scale factor,
        # which is written as time2 - time1 = scale_offset * time1
        scale_offset = SCALE_OFFSETS[(self.scale, scale)]
        if scale_offset is None:
            self._time.scale = scale
        else:
            jd1, jd2 = self._time.jd1, self._time.jd2
            offset1, offset2 = day_frac(jd1, jd2, factor=scale_offset)
            self._time = self.FORMATS[self.format](
                jd1 + offset1, jd2 + offset2, scale,
                self.precision, self.in_subfmt,
                self.out_subfmt, from_jd=True)
```
### 97 - astropy/time/formats.py:

Start line: 769, End line: 788

```python
class TimeGPS(TimeFromEpoch):
    """GPS time: seconds from 1980-01-06 00:00:00 UTC
    For example, 630720013.0 is midnight on January 1, 2000.

    Notes
    =====
    This implementation is strictly a representation of the number of seconds
    (including leap seconds) since midnight UTC on 1980-01-06.  GPS can also be
    considered as a time scale which is ahead of TAI by a fixed offset
    (to within about 100 nanoseconds).

    For details, see https://www.usno.navy.mil/USNO/time/gps/usno-gps-time-transfer
    """
    name = 'gps'
    unit = 1.0 / erfa.DAYSEC  # in days (1 day == 86400 seconds)
    epoch_val = '1980-01-06 00:00:19'
    # above epoch is the same as Time('1980-01-06 00:00:00', scale='utc').tai
    epoch_val2 = None
    epoch_scale = 'tai'
    epoch_format = 'iso'
```
### 98 - astropy/time/formats.py:

Start line: 791, End line: 815

```python
class TimePlotDate(TimeFromEpoch):
    """
    Matplotlib `~matplotlib.pyplot.plot_date` input:
    1 + number of days from 0001-01-01 00:00:00 UTC

    This can be used directly in the matplotlib `~matplotlib.pyplot.plot_date`
    function::

      >>> import matplotlib.pyplot as plt
      >>> jyear = np.linspace(2000, 2001, 20)
      >>> t = Time(jyear, format='jyear', scale='utc')
      >>> plt.plot_date(t.plot_date, jyear)
      >>> plt.gcf().autofmt_xdate()  # orient date labels at a slant
      >>> plt.draw()

    For example, 730120.0003703703 is midnight on January 1, 2000.
    """
    # This corresponds to the zero reference time for matplotlib plot_date().
    # Note that TAI and UTC are equivalent at the reference time.
    name = 'plot_date'
    unit = 1.0
    epoch_val = 1721424.5  # Time('0001-01-01 00:00:00', scale='tai').jd - 1
    epoch_val2 = None
    epoch_scale = 'utc'
    epoch_format = 'jd'
```
### 100 - astropy/time/formats.py:

Start line: 1720, End line: 1737

```python
class TimeEpochDate(TimeNumeric):
    """
    Base class for support floating point Besselian and Julian epoch dates
    """
    _default_scale = 'tt'  # As of astropy 3.2, this is no longer 'utc'.

    def set_jds(self, val1, val2):
        self._check_scale(self._scale)  # validate scale.
        epoch_to_jd = getattr(erfa, self.epoch_to_jd)
        jd1, jd2 = epoch_to_jd(val1 + val2)
        self.jd1, self.jd2 = day_frac(jd1, jd2)

    def to_value(self, **kwargs):
        jd_to_epoch = getattr(erfa, self.jd_to_epoch)
        value = jd_to_epoch(self.jd1, self.jd2)
        return super().to_value(jd1=value, jd2=np.float64(0.0), **kwargs)

    value = property(to_value)
```
### 102 - astropy/time/core.py:

Start line: 2324, End line: 2391

```python
class TimeDelta(TimeBase):
    """
    Represent the time difference between two times.

    A TimeDelta object is initialized with one or more times in the ``val``
    argument.  The input times in ``val`` must conform to the specified
    ``format``.  The optional ``val2`` time input should be supplied only for
    numeric input formats (e.g. JD) where very high precision (better than
    64-bit precision) is required.

    The allowed values for ``format`` can be listed with::

      >>> list(TimeDelta.FORMATS)
      ['sec', 'jd', 'datetime']

    Note that for time differences, the scale can be among three groups:
    geocentric ('tai', 'tt', 'tcg'), barycentric ('tcb', 'tdb'), and rotational
    ('ut1'). Within each of these, the scales for time differences are the
    same. Conversion between geocentric and barycentric is possible, as there
    is only a scale factor change, but one cannot convert to or from 'ut1', as
    this requires knowledge of the actual times, not just their difference. For
    a similar reason, 'utc' is not a valid scale for a time difference: a UTC
    day is not always 86400 seconds.

    See also:

    - https://docs.astropy.org/en/stable/time/
    - https://docs.astropy.org/en/stable/time/index.html#time-deltas

    Parameters
    ----------
    val : sequence, ndarray, number, `~astropy.units.Quantity` or `~astropy.time.TimeDelta` object
        Value(s) to initialize the time difference(s). Any quantities will
        be converted appropriately (with care taken to avoid rounding
        errors for regular time units).
    val2 : sequence, ndarray, number, or `~astropy.units.Quantity`; optional
        Additional values, as needed to preserve precision.
    format : str, optional
        Format of input value(s). For numerical inputs without units,
        "jd" is assumed and values are interpreted as days.
        A deprecation warning is raised in this case. To avoid the warning,
        either specify the format or add units to the input values.
    scale : str, optional
        Time scale of input value(s), must be one of the following values:
        ('tdb', 'tt', 'ut1', 'tcg', 'tcb', 'tai'). If not given (or
        ``None``), the scale is arbitrary; when added or subtracted from a
        ``Time`` instance, it will be used without conversion.
    copy : bool, optional
        Make a copy of the input values
    """
    SCALES = TIME_DELTA_SCALES
    """List of time delta scales."""

    FORMATS = TIME_DELTA_FORMATS
    """Dict of time delta formats."""

    info = TimeDeltaInfo()

    def __new__(cls, val, val2=None, format=None, scale=None,
                precision=None, in_subfmt=None, out_subfmt=None,
                location=None, copy=False):

        if isinstance(val, TimeDelta):
            self = val.replicate(format=format, copy=copy, cls=cls)
        else:
            self = super().__new__(cls)

        return self
```
### 103 - astropy/time/core.py:

Start line: 1044, End line: 1073

```python
class TimeBase(ShapedLikeNDArray):

    def replicate(self, format=None, copy=False, cls=None):
        """
        Return a replica of the Time object, optionally changing the format.

        If ``format`` is supplied then the time format of the returned Time
        object will be set accordingly, otherwise it will be unchanged from the
        original.

        If ``copy`` is set to `True` then a full copy of the internal time arrays
        will be made.  By default the replica will use a reference to the
        original arrays when possible to save memory.  The internal time arrays
        are normally not changeable by the user so in most cases it should not
        be necessary to set ``copy`` to `True`.

        The convenience method copy() is available in which ``copy`` is `True`
        by default.

        Parameters
        ----------
        format : str, optional
            Time format of the replica.
        copy : bool, optional
            Return a true copy instead of using references where possible.

        Returns
        -------
        tm : Time object
            Replica of this object
        """
        return self._apply('copy' if copy else 'replicate', format=format, cls=cls)
```
### 106 - astropy/time/core.py:

Start line: 2616, End line: 2684

```python
class TimeDelta(TimeBase):

    def to_value(self, *args, **kwargs):
        """Get time delta values expressed in specified output format or unit.

        This method is flexible and handles both conversion to a specified
        ``TimeDelta`` format / sub-format AND conversion to a specified unit.
        If positional argument(s) are provided then the first one is checked
        to see if it is a valid ``TimeDelta`` format, and next it is checked
        to see if it is a valid unit or unit string.

        To convert to a ``TimeDelta`` format and optional sub-format the options
        are::

          tm = TimeDelta(1.0 * u.s)
          tm.to_value('jd')  # equivalent of tm.jd
          tm.to_value('jd', 'decimal')  # convert to 'jd' as a Decimal object
          tm.to_value('jd', subfmt='decimal')
          tm.to_value(format='jd', subfmt='decimal')

        To convert to a unit with optional equivalencies, the options are::

          tm.to_value('hr')  # convert to u.hr (hours)
          tm.to_value('hr', [])  # specify equivalencies as a positional arg
          tm.to_value('hr', equivalencies=[])
          tm.to_value(unit='hr', equivalencies=[])

        The built-in `~astropy.time.TimeDelta` options for ``format`` are:
        {'jd', 'sec', 'datetime'}.

        For the two numerical formats 'jd' and 'sec', the available ``subfmt``
        options are: {'float', 'long', 'decimal', 'str', 'bytes'}. Here, 'long'
        uses ``numpy.longdouble`` for somewhat enhanced precision (with the
        enhancement depending on platform), and 'decimal' instances of
        :class:`decimal.Decimal` for full precision.  For the 'str' and 'bytes'
        sub-formats, the number of digits is also chosen such that time values
        are represented accurately.  Default: as set by ``out_subfmt`` (which by
        default picks the first available for a given format, i.e., 'float').

        Parameters
        ----------
        format : str, optional
            The format in which one wants the `~astropy.time.TimeDelta` values.
            Default: the current format.
        subfmt : str, optional
            Possible sub-format in which the values should be given. Default: as
            set by ``out_subfmt`` (which by default picks the first available
            for a given format, i.e., 'float' or 'date_hms').
        unit : `~astropy.units.UnitBase` instance or str, optional
            The unit in which the value should be given.
        equivalencies : list of tuple
            A list of equivalence pairs to try if the units are not directly
            convertible (see :ref:`astropy:unit_equivalencies`). If `None`, no
            equivalencies will be applied at all, not even any set globally or
            within a context.

        Returns
        -------
        value : ndarray or scalar
            The value in the format or units specified.

        See also
        --------
        to : Convert to a `~astropy.units.Quantity` instance in a given unit.
        value : The time value in the current format.

        """
        if not (args or kwargs):
            raise TypeError('to_value() missing required format or unit argument')

        # TODO: maybe allow 'subfmt' also for units, keeping full precision
        # ... other code
```
### 107 - astropy/time/formats.py:

Start line: 1538, End line: 1582

```python
class TimeYearDayTime(TimeISO):
    """
    Year, day-of-year and time as "YYYY:DOY:HH:MM:SS.sss...".
    The day-of-year (DOY) goes from 001 to 365 (366 in leap years).
    For example, 2000:001:00:00:00.000 is midnight on January 1, 2000.

    The allowed subformats are:

    - 'date_hms': date + hours, mins, secs (and optional fractional secs)
    - 'date_hm': date + hours, mins
    - 'date': date
    """

    name = 'yday'
    subfmts = (('date_hms',
                '%Y:%j:%H:%M:%S',
                '{year:d}:{yday:03d}:{hour:02d}:{min:02d}:{sec:02d}'),
               ('date_hm',
                '%Y:%j:%H:%M',
                '{year:d}:{yday:03d}:{hour:02d}:{min:02d}'),
               ('date',
                '%Y:%j',
                '{year:d}:{yday:03d}'))

    # Define positions and starting delimiter for year, month, day, hour,
    # minute, seconds components of an ISO time. This is used by the fast
    # C-parser parse_ymdhms_times()
    #
    #  "2000:123:13:14:15.678"
    #   012345678901234567890
    #   yyyy:ddd:hh:mm:ss.fff
    # Parsed as ('yyyy', ':ddd', ':hh', ':mm', ':ss', '.fff')
    #
    # delims: character at corresponding `starts` position (0 => no character)
    # starts: position where component starts (including delimiter if present)
    # stops: position where component ends (-1 => continue to end of string)

    fast_parser_pars = dict(
        delims=(0, 0, ord(':'), ord(':'), ord(':'), ord(':'), ord('.')),
        starts=(0, -1, 4, 8, 11, 14, 17),
        stops=(3, -1, 7, 10, 13, 16, -1),
        # Break allowed before:
        #              y  m  d  h  m  s  f
        break_allowed=(0, 0, 0, 1, 0, 1, 1),
        has_day_of_year=1)
```
### 109 - astropy/time/formats.py:

Start line: 328, End line: 364

```python
class TimeFormat:

    def to_value(self, parent=None, out_subfmt=None):
        """
        Return time representation from internal jd1 and jd2 in specified
        ``out_subfmt``.

        This is the base method that ignores ``parent`` and uses the ``value``
        property to compute the output. This is done by temporarily setting
        ``self.out_subfmt`` and calling ``self.value``. This is required for
        legacy Format subclasses prior to astropy 4.0  New code should instead
        implement the value functionality in ``to_value()`` and then make the
        ``value`` property be a simple call to ``self.to_value()``.

        Parameters
        ----------
        parent : object
            Parent `~astropy.time.Time` object associated with this
            `~astropy.time.TimeFormat` object
        out_subfmt : str or None
            Output subformt (use existing self.out_subfmt if `None`)

        Returns
        -------
        value : numpy.array, numpy.ma.array
            Array or masked array of formatted time representation values
        """
        # Get value via ``value`` property, overriding out_subfmt temporarily if needed.
        if out_subfmt is not None:
            out_subfmt_orig = self.out_subfmt
            try:
                self.out_subfmt = out_subfmt
                value = self.value
            finally:
                self.out_subfmt = out_subfmt_orig
        else:
            value = self.value

        return self.mask_if_needed(value)
```
### 110 - astropy/time/core.py:

Start line: 712, End line: 740

```python
class TimeBase(ShapedLikeNDArray):

    @shape.setter
    def shape(self, shape):
        del self.cache

        # We have to keep track of arrays that were already reshaped,
        # since we may have to return those to their original shape if a later
        # shape-setting fails.
        reshaped = []
        oldshape = self.shape

        # In-place reshape of data/attributes.  Need to access _time.jd1/2 not
        # self.jd1/2 because the latter are not guaranteed to be the actual
        # data, and in fact should not be directly changeable from the public
        # API.
        for obj, attr in ((self._time, 'jd1'),
                          (self._time, 'jd2'),
                          (self, '_delta_ut1_utc'),
                          (self, '_delta_tdb_tt'),
                          (self, 'location')):
            val = getattr(obj, attr, None)
            if val is not None and val.size > 1:
                try:
                    val.shape = shape
                except Exception:
                    for val2 in reshaped:
                        val2.shape = oldshape
                    raise
                else:
                    reshaped.append(val)
```
### 112 - astropy/time/core.py:

Start line: 1457, End line: 1483

```python
class TimeBase(ShapedLikeNDArray):

    def __lt__(self, other):
        return self._time_comparison(other, operator.lt)

    def __le__(self, other):
        return self._time_comparison(other, operator.le)

    def __eq__(self, other):
        """
        If other is an incompatible object for comparison, return `False`.
        Otherwise, return `True` if the time difference between self and
        other is zero.
        """
        return self._time_comparison(other, operator.eq)

    def __ne__(self, other):
        """
        If other is an incompatible object for comparison, return `True`.
        Otherwise, return `False` if the time difference between self and
        other is zero.
        """
        return self._time_comparison(other, operator.ne)

    def __gt__(self, other):
        return self._time_comparison(other, operator.gt)

    def __ge__(self, other):
        return self._time_comparison(other, operator.ge)
```
### 115 - astropy/time/formats.py:

Start line: 1058, End line: 1121

```python
class TimeYMDHMS(TimeUnique):

    def _check_val_type(self, val1, val2):
        """
        This checks inputs for the YMDHMS format.

        It is bit more complex than most format checkers because of the flexible
        input that is allowed.  Also, it actually coerces ``val1`` into an appropriate
        dict of ndarrays that can be used easily by ``set_jds()``.  This is useful
        because it makes it easy to get default values in that routine.

        Parameters
        ----------
        val1 : ndarray or None
        val2 : ndarray or None

        Returns
        -------
        val1_as_dict, val2 : val1 as dict or None, val2 is always None

        """
        if val2 is not None:
            raise ValueError('val2 must be None for ymdhms format')

        ymdhms = ['year', 'month', 'day', 'hour', 'minute', 'second']

        if val1.dtype.names:
            # Convert to a dict of ndarray
            val1_as_dict = {name: val1[name] for name in val1.dtype.names}

        elif val1.shape == (0,):
            # Input was empty list [], so set to None and set_jds will handle this
            return None, None

        elif (val1.dtype.kind == 'O'
              and val1.shape == ()
              and isinstance(val1.item(), dict)):
            # Code gets here for input as a dict.  The dict input
            # can be either scalar values or N-d arrays.

            # Extract the item (which is a dict) and broadcast values to the
            # same shape here.
            names = val1.item().keys()
            values = val1.item().values()
            val1_as_dict = {name: value for name, value
                            in zip(names, np.broadcast_arrays(*values))}

        else:
            raise ValueError('input must be dict or table-like')

        # Check that the key names now are good.
        names = val1_as_dict.keys()
        required_names = ymdhms[:len(names)]

        def comma_repr(vals):
            return ', '.join(repr(val) for val in vals)

        bad_names = set(names) - set(ymdhms)
        if bad_names:
            raise ValueError(f'{comma_repr(bad_names)} not allowed as YMDHMS key name(s)')

        if set(names) != set(required_names):
            raise ValueError(f'for {len(names)} input key names '
                             f'you must supply {comma_repr(required_names)}')

        return val1_as_dict, val2
```
### 116 - astropy/time/formats.py:

Start line: 684, End line: 701

```python
class TimeUnix(TimeFromEpoch):
    """
    Unix time (UTC): seconds from 1970-01-01 00:00:00 UTC, ignoring leap seconds.

    For example, 946684800.0 in Unix time is midnight on January 1, 2000.

    NOTE: this quantity is not exactly unix time and differs from the strict
    POSIX definition by up to 1 second on days with a leap second.  POSIX
    unix time actually jumps backward by 1 second at midnight on leap second
    days while this class value is monotonically increasing at 86400 seconds
    per UTC day.
    """
    name = 'unix'
    unit = 1.0 / erfa.DAYSEC  # in days (1 day == 86400 seconds)
    epoch_val = '1970-01-01 00:00:00'
    epoch_val2 = None
    epoch_scale = 'utc'
    epoch_format = 'iso'
```
### 117 - astropy/time/core.py:

Start line: 767, End line: 781

```python
class TimeBase(ShapedLikeNDArray):

    @property
    def jd1(self):
        """
        First of the two doubles that internally store time value(s) in JD.
        """
        jd1 = self._time.mask_if_needed(self._time.jd1)
        return self._shaped_like_input(jd1)

    @property
    def jd2(self):
        """
        Second of the two doubles that internally store time value(s) in JD.
        """
        jd2 = self._time.mask_if_needed(self._time.jd2)
        return self._shaped_like_input(jd2)
```
### 118 - astropy/time/core.py:

Start line: 2584, End line: 2614

```python
class TimeDelta(TimeBase):

    def __rtruediv__(self, other):
        """Division by `TimeDelta` objects of numbers/arrays."""
        # Here, we do not have to worry about returning NotImplemented,
        # since other has already had a chance to look at us.
        return other / self.to(u.day)

    def to(self, unit, equivalencies=[]):
        """
        Convert to a quantity in the specified unit.

        Parameters
        ----------
        unit : unit-like
            The unit to convert to.
        equivalencies : list of tuple
            A list of equivalence pairs to try if the units are not directly
            convertible (see :ref:`astropy:unit_equivalencies`). If `None`, no
            equivalencies will be applied at all, not even any set globallyq
            or within a context.

        Returns
        -------
        quantity : `~astropy.units.Quantity`
            The quantity in the units specified.

        See also
        --------
        to_value : get the numerical value in a given unit.
        """
        return u.Quantity(self._time.jd1 + self._time.jd2,
                          u.day).to(unit, equivalencies=equivalencies)
```
### 119 - astropy/time/core.py:

Start line: 1075, End line: 1157

```python
class TimeBase(ShapedLikeNDArray):

    def _apply(self, method, *args, format=None, cls=None, **kwargs):
        """Create a new time object, possibly applying a method to the arrays.

        Parameters
        ----------
        method : str or callable
            If string, can be 'replicate'  or the name of a relevant
            `~numpy.ndarray` method. In the former case, a new time instance
            with unchanged internal data is created, while in the latter the
            method is applied to the internal ``jd1`` and ``jd2`` arrays, as
            well as to possible ``location``, ``_delta_ut1_utc``, and
            ``_delta_tdb_tt`` arrays.
            If a callable, it is directly applied to the above arrays.
            Examples: 'copy', '__getitem__', 'reshape', `~numpy.broadcast_to`.
        args : tuple
            Any positional arguments for ``method``.
        kwargs : dict
            Any keyword arguments for ``method``.  If the ``format`` keyword
            argument is present, this will be used as the Time format of the
            replica.

        Examples
        --------
        Some ways this is used internally::

            copy : ``_apply('copy')``
            replicate : ``_apply('replicate')``
            reshape : ``_apply('reshape', new_shape)``
            index or slice : ``_apply('__getitem__', item)``
            broadcast : ``_apply(np.broadcast, shape=new_shape)``
        """
        new_format = self.format if format is None else format

        if callable(method):
            apply_method = lambda array: method(array, *args, **kwargs)

        else:
            if method == 'replicate':
                apply_method = None
            else:
                apply_method = operator.methodcaller(method, *args, **kwargs)

        jd1, jd2 = self._time.jd1, self._time.jd2
        if apply_method:
            jd1 = apply_method(jd1)
            jd2 = apply_method(jd2)

        # Get a new instance of our class and set its attributes directly.
        tm = super().__new__(cls or self.__class__)
        tm._time = TimeJD(jd1, jd2, self.scale, precision=0,
                          in_subfmt='*', out_subfmt='*', from_jd=True)

        # Optional ndarray attributes.
        for attr in ('_delta_ut1_utc', '_delta_tdb_tt', 'location'):
            try:
                val = getattr(self, attr)
            except AttributeError:
                continue

            if apply_method:
                # Apply the method to any value arrays (though skip if there is
                # only an array scalar and the method would return a view,
                # since in that case nothing would change).
                if getattr(val, 'shape', ()):
                    val = apply_method(val)
                elif method == 'copy' or method == 'flatten':
                    # flatten should copy also for a single element array, but
                    # we cannot use it directly for array scalars, since it
                    # always returns a one-dimensional array. So, just copy.
                    val = copy.copy(val)

            setattr(tm, attr, val)

        # Copy other 'info' attr only if it has actually been defined and the
        # time object is not a scalar (issue #10688).
        # See PR #3898 for further explanation and justification, along
        # with Quantity.__array_finalize__
        if 'info' in self.__dict__:
            tm.info = self.info

        # Make the new internal _time object corresponding to the format
        # in the copy.  If the format is unchanged this process is lightweight
        # and does not create any new arrays.
        # ... other code
```
### 120 - astropy/time/core.py:

Start line: 1634, End line: 1656

```python
class Time(TimeBase):

    @classmethod
    def now(cls):
        """
        Creates a new object corresponding to the instant in time this
        method is called.

        .. note::
            "Now" is determined using the `~datetime.datetime.utcnow`
            function, so its accuracy and precision is determined by that
            function.  Generally that means it is set by the accuracy of
            your system clock.

        Returns
        -------
        nowtime : :class:`~astropy.time.Time`
            A new `Time` object (or a subclass of `Time` if this is called from
            such a subclass) at the current time.
        """
        # call `utcnow` immediately to be sure it's ASAP
        dtnow = datetime.utcnow()
        return cls(val=dtnow, format='datetime', scale='utc')

    info = TimeInfo()
```
### 121 - astropy/time/core.py:

Start line: 2482, End line: 2517

```python
class TimeDelta(TimeBase):

    def __add__(self, other):
        # If other is a Time then use Time.__add__ to do the calculation.
        if isinstance(other, Time):
            return other.__add__(self)

        return self._add_sub(other, operator.add)

    def __sub__(self, other):
        # TimeDelta - Time is an error
        if isinstance(other, Time):
            raise OperandTypeError(self, other, '-')

        return self._add_sub(other, operator.sub)

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        out = self.__sub__(other)
        return -out

    def __neg__(self):
        """Negation of a `TimeDelta` object."""
        new = self.copy()
        new._time.jd1 = -self._time.jd1
        new._time.jd2 = -self._time.jd2
        return new

    def __abs__(self):
        """Absolute value of a `TimeDelta` object."""
        jd1, jd2 = self._time.jd1, self._time.jd2
        negative = jd1 + jd2 < 0
        new = self.copy()
        new._time.jd1 = np.where(negative, -jd1, jd1)
        new._time.jd2 = np.where(negative, -jd2, jd2)
        return new
```
### 122 - astropy/time/core.py:

Start line: 1409, End line: 1431

```python
class TimeBase(ShapedLikeNDArray):

    @override__dir__
    def __dir__(self):
        result = set(self.SCALES)
        result.update(self.FORMATS)
        return result

    def _match_shape(self, val):
        """
        Ensure that `val` is matched to length of self.  If val has length 1
        then broadcast, otherwise cast to double and make sure shape matches.
        """
        val = _make_array(val, copy=True)  # be conservative and copy
        if val.size > 1 and val.shape != self.shape:
            try:
                # check the value can be broadcast to the shape of self.
                val = np.broadcast_to(val, self.shape, subok=True)
            except Exception:
                raise ValueError('Attribute shape must match or be '
                                 'broadcastable to that of Time object. '
                                 'Typically, give either a single value or '
                                 'one for each time.')

        return val
```
### 124 - astropy/time/core.py:

Start line: 1279, End line: 1295

```python
class TimeBase(ShapedLikeNDArray):

    def argsort(self, axis=-1):
        """Returns the indices that would sort the time array.

        This is similar to :meth:`~numpy.ndarray.argsort`, but adapted to ensure
        that the full precision given by the two doubles ``jd1`` and ``jd2``
        is used, and that corresponding attributes are copied.  Internally,
        it uses :func:`~numpy.lexsort`, and hence no sort method can be chosen.
        """
        # For procedure, see comment on argmin.
        jd1, jd2 = self.jd1, self.jd2
        approx = jd1 + jd2
        remainder = (jd1 - approx) + jd2

        if axis is None:
            return np.lexsort((remainder.ravel(), approx.ravel()))
        else:
            return np.lexsort(keys=(remainder, approx), axis=axis)
```
### 125 - astropy/time/core.py:

Start line: 1954, End line: 2015

```python
class Time(TimeBase):

    if isinstance(sidereal_time.__doc__, str):
        sidereal_time.__doc__ = sidereal_time.__doc__.format(
            'apparent', sorted(SIDEREAL_TIME_MODELS['apparent'].keys()),
            'mean', sorted(SIDEREAL_TIME_MODELS['mean'].keys()))

    def _sid_time_or_earth_rot_ang(self, longitude, function, scales, include_tio=True):
        """Calculate a local sidereal time or Earth rotation angle.

        Parameters
        ----------
        longitude : `~astropy.units.Quantity`, `~astropy.coordinates.EarthLocation`, str, or None; optional
            The longitude on the Earth at which to compute the Earth rotation
            angle (taken from a location as needed).  If `None` (default), taken
            from the ``location`` attribute of the Time instance.
        function : callable
            The ERFA function to use.
        scales : tuple of str
            The time scales that the function requires on input.
        include_tio : bool, optional
            Whether to includes the TIO locator corrected for polar motion.
            Should be `False` for pre-2000 IAU models.  Default: `True`.

        Returns
        -------
        `~astropy.coordinates.Longitude`
            Local sidereal time or Earth rotation angle, with units of hourangle.

        """  # noqa
        from astropy.coordinates import Longitude, EarthLocation
        from astropy.coordinates.builtin_frames.utils import get_polar_motion
        from astropy.coordinates.matrix_utilities import rotation_matrix

        if longitude is None:
            if self.location is None:
                raise ValueError('No longitude is given but the location for '
                                 'the Time object is not set.')
            longitude = self.location.lon
        elif isinstance(longitude, EarthLocation):
            longitude = longitude.lon
        else:
            # Sanity check on input; default unit is degree.
            longitude = Longitude(longitude, u.degree, copy=False)

        theta = self._call_erfa(function, scales)

        if include_tio:
            # TODO: this duplicates part of coordinates.erfa_astrom.ErfaAstrom.apio;
            # maybe posisble to factor out to one or the other.
            sp = self._call_erfa(erfa.sp00, ('tt',))
            xp, yp = get_polar_motion(self)
            # Form the rotation matrix, CIRS to apparent [HA,Dec].
            r = (rotation_matrix(longitude, 'z')
                 @ rotation_matrix(-yp, 'x', unit=u.radian)
                 @ rotation_matrix(-xp, 'y', unit=u.radian)
                 @ rotation_matrix(theta + sp, 'z', unit=u.radian))
            # Solve for angle.
            angle = np.arctan2(r[..., 0, 1], r[..., 0, 0]) << u.radian

        else:
            angle = longitude + (theta << u.radian)

        return Longitude(angle, u.hourangle)
```
### 127 - astropy/time/formats.py:

Start line: 1764, End line: 1802

```python
class TimeEpochDateString(TimeString):
    """
    Base class to support string Besselian and Julian epoch dates
    such as 'B1950.0' or 'J2000.0' respectively.
    """
    _default_scale = 'tt'  # As of astropy 3.2, this is no longer 'utc'.

    def set_jds(self, val1, val2):
        epoch_prefix = self.epoch_prefix
        # Be liberal in what we accept: convert bytes to ascii.
        to_string = (str if val1.dtype.kind == 'U' else
                     lambda x: str(x.item(), encoding='ascii'))
        iterator = np.nditer([val1, None], op_dtypes=[val1.dtype, np.double],
                             flags=['zerosize_ok'])
        for val, years in iterator:
            try:
                time_str = to_string(val)
                epoch_type, year_str = time_str[0], time_str[1:]
                year = float(year_str)
                if epoch_type.upper() != epoch_prefix:
                    raise ValueError
            except (IndexError, ValueError, UnicodeEncodeError):
                raise ValueError(f'Time {val} does not match {self.name} format')
            else:
                years[...] = year

        self._check_scale(self._scale)  # validate scale.
        epoch_to_jd = getattr(erfa, self.epoch_to_jd)
        jd1, jd2 = epoch_to_jd(iterator.operands[-1])
        self.jd1, self.jd2 = day_frac(jd1, jd2)

    @property
    def value(self):
        jd_to_epoch = getattr(erfa, self.jd_to_epoch)
        years = jd_to_epoch(self.jd1, self.jd2)
        # Use old-style format since it is a factor of 2 faster
        str_fmt = self.epoch_prefix + '%.' + str(self.precision) + 'f'
        outs = [str_fmt % year for year in years.flat]
        return np.array(outs).reshape(self.jd1.shape)
```
### 128 - astropy/time/formats.py:

Start line: 756, End line: 766

```python
class TimeCxcSec(TimeFromEpoch):
    """
    Chandra X-ray Center seconds from 1998-01-01 00:00:00 TT.
    For example, 63072064.184 is midnight on January 1, 2000.
    """
    name = 'cxcsec'
    unit = 1.0 / erfa.DAYSEC  # in days (1 day == 86400 seconds)
    epoch_val = '1998-01-01 00:00:00'
    epoch_val2 = None
    epoch_scale = 'tt'
    epoch_format = 'iso'
```
### 131 - astropy/time/formats.py:

Start line: 1214, End line: 1253

```python
class TimeString(TimeUnique):
    """
    Base class for string-like time representations.

    This class assumes that anything following the last decimal point to the
    right is a fraction of a second.

    **Fast C-based parser**

    Time format classes can take advantage of a fast C-based parser if the times
    are represented as fixed-format strings with year, month, day-of-month,
    hour, minute, second, OR year, day-of-year, hour, minute, second. This can
    be a factor of 20 or more faster than the pure Python parser.

    Fixed format means that the components always have the same number of
    characters. The Python parser will accept ``2001-9-2`` as a date, but the C
    parser would require ``2001-09-02``.

    A subclass in this case must define a class attribute ``fast_parser_pars``
    which is a `dict` with all of the keys below. An inherited attribute is not
    checked, only an attribute in the class ``__dict__``.

    - ``delims`` (tuple of int): ASCII code for character at corresponding
      ``starts`` position (0 => no character)

    - ``starts`` (tuple of int): position where component starts (including
      delimiter if present). Use -1 for the month component for format that use
      day of year.

    - ``stops`` (tuple of int): position where component ends. Use -1 to
      continue to end of string, or for the month component for formats that use
      day of year.

    - ``break_allowed`` (tuple of int): if true (1) then the time string can
          legally end just before the corresponding component (e.g. "2000-01-01"
          is a valid time but "2000-01-01 12" is not).

    - ``has_day_of_year`` (int): 0 if dates have year, month, day; 1 if year,
      day-of-year
    """
```
### 132 - astropy/time/core.py:

Start line: 80, End line: 111

```python
SCALE_OFFSETS = {('tt', 'tai'): None,
                 ('tai', 'tt'): None,
                 ('tcg', 'tt'): -erfa.ELG,
                 ('tt', 'tcg'): erfa.ELG / (1. - erfa.ELG),
                 ('tcg', 'tai'): -erfa.ELG,
                 ('tai', 'tcg'): erfa.ELG / (1. - erfa.ELG),
                 ('tcb', 'tdb'): -erfa.ELB,
                 ('tdb', 'tcb'): erfa.ELB / (1. - erfa.ELB)}

# triple-level dictionary, yay!
SIDEREAL_TIME_MODELS = {
    'mean': {
        'IAU2006': {'function': erfa.gmst06, 'scales': ('ut1', 'tt')},
        'IAU2000': {'function': erfa.gmst00, 'scales': ('ut1', 'tt')},
        'IAU1982': {'function': erfa.gmst82, 'scales': ('ut1',), 'include_tio': False}
    },
    'apparent': {
        'IAU2006A': {'function': erfa.gst06a, 'scales': ('ut1', 'tt')},
        'IAU2000A': {'function': erfa.gst00a, 'scales': ('ut1', 'tt')},
        'IAU2000B': {'function': erfa.gst00b, 'scales': ('ut1',)},
        'IAU1994': {'function': erfa.gst94, 'scales': ('ut1',), 'include_tio': False}
    }}


class _LeapSecondsCheck(enum.Enum):
    NOT_STARTED = 0     # No thread has reached the check
    RUNNING = 1         # A thread is running update_leap_seconds (_LEAP_SECONDS_LOCK is held)
    DONE = 2            # update_leap_seconds has completed


_LEAP_SECONDS_CHECK = _LeapSecondsCheck.NOT_STARTED
_LEAP_SECONDS_LOCK = threading.RLock()
```
### 134 - astropy/time/core.py:

Start line: 689, End line: 710

```python
class TimeBase(ShapedLikeNDArray):

    @property
    def shape(self):
        """The shape of the time instances.

        Like `~numpy.ndarray.shape`, can be set to a new shape by assigning a
        tuple.  Note that if different instances share some but not all
        underlying data, setting the shape of one instance can make the other
        instance unusable.  Hence, it is strongly recommended to get new,
        reshaped instances with the ``reshape`` method.

        Raises
        ------
        ValueError
            If the new shape has the wrong total number of elements.
        AttributeError
            If the shape of the ``jd1``, ``jd2``, ``location``,
            ``delta_ut1_utc``, or ``delta_tdb_tt`` attributes cannot be changed
            without the arrays being copied.  For these cases, use the
            `Time.reshape` method (which copies any arrays that cannot be
            reshaped in-place).
        """
        return self._time.jd1.shape
```
### 135 - astropy/time/formats.py:

Start line: 1394, End line: 1421

```python
class TimeString(TimeUnique):

    def str_kwargs(self):
        """
        Generator that yields a dict of values corresponding to the
        calendar date and time for the internal JD values.
        """
        scale = self.scale.upper().encode('ascii'),
        iys, ims, ids, ihmsfs = erfa.d2dtf(scale, self.precision,
                                           self.jd1, self.jd2_filled)

        # Get the str_fmt element of the first allowed output subformat
        _, _, str_fmt = self._select_subfmts(self.out_subfmt)[0]

        yday = None
        has_yday = '{yday:' in str_fmt

        ihrs = ihmsfs['h']
        imins = ihmsfs['m']
        isecs = ihmsfs['s']
        ifracs = ihmsfs['f']
        for iy, im, id, ihr, imin, isec, ifracsec in np.nditer(
                [iys, ims, ids, ihrs, imins, isecs, ifracs],
                flags=['zerosize_ok']):
            if has_yday:
                yday = datetime.datetime(iy, im, id).timetuple().tm_yday

            yield {'year': int(iy), 'mon': int(im), 'day': int(id),
                   'hour': int(ihr), 'min': int(imin), 'sec': int(isec),
                   'fracsec': int(ifracsec), 'yday': yday}
```
### 138 - astropy/time/core.py:

Start line: 2551, End line: 2582

```python
class TimeDelta(TimeBase):

    def __rmul__(self, other):
        """Multiplication of numbers/arrays with `TimeDelta` objects."""
        return self.__mul__(other)

    def __truediv__(self, other):
        """Division of `TimeDelta` objects by numbers/arrays."""
        # Cannot do __mul__(1./other) as that looses precision
        if ((isinstance(other, u.UnitBase)
             and other == u.dimensionless_unscaled)
                or (isinstance(other, str) and other == '')):
            return self.copy()

        # If other is something consistent with a dimensionless quantity
        # (could just be a float or an array), then we can just divide in.
        try:
            other = u.Quantity(other, u.dimensionless_unscaled, copy=False)
        except Exception:
            # If not consistent with a dimensionless quantity, try downgrading
            # self to a quantity and see if things work.
            try:
                return self.to(u.day) / other
            except Exception:
                # The various ways we could divide all failed;
                # returning NotImplemented to give other a final chance.
                return NotImplemented

        jd1, jd2 = day_frac(self.jd1, self.jd2, divisor=other.value)
        out = TimeDelta(jd1, jd2, format='jd', scale=self.scale)

        if self.format != 'jd':
            out = out.replicate(format=self.format)
        return out
```
### 141 - astropy/time/formats.py:

Start line: 1633, End line: 1670

```python
class TimeFITS(TimeString):
    """
    FITS format: "[±Y]YYYY-MM-DD[THH:MM:SS[.sss]]".

    ISOT but can give signed five-digit year (mostly for negative years);

    The allowed subformats are:

    - 'date_hms': date + hours, mins, secs (and optional fractional secs)
    - 'date': date
    - 'longdate_hms': as 'date_hms', but with signed 5-digit year
    - 'longdate': as 'date', but with signed 5-digit year

    See Rots et al., 2015, A&A 574:A36 (arXiv:1409.7583).
    """
    name = 'fits'
    subfmts = (
        ('date_hms',
         (r'(?P<year>\d{4})-(?P<mon>\d\d)-(?P<mday>\d\d)T'
          r'(?P<hour>\d\d):(?P<min>\d\d):(?P<sec>\d\d(\.\d*)?)'),
         '{year:04d}-{mon:02d}-{day:02d}T{hour:02d}:{min:02d}:{sec:02d}'),
        ('date',
         r'(?P<year>\d{4})-(?P<mon>\d\d)-(?P<mday>\d\d)',
         '{year:04d}-{mon:02d}-{day:02d}'),
        ('longdate_hms',
         (r'(?P<year>[+-]\d{5})-(?P<mon>\d\d)-(?P<mday>\d\d)T'
          r'(?P<hour>\d\d):(?P<min>\d\d):(?P<sec>\d\d(\.\d*)?)'),
         '{year:+06d}-{mon:02d}-{day:02d}T{hour:02d}:{min:02d}:{sec:02d}'),
        ('longdate',
         r'(?P<year>[+-]\d{5})-(?P<mon>\d\d)-(?P<mday>\d\d)',
         '{year:+06d}-{mon:02d}-{day:02d}'))
    # Add the regex that parses the scale and possible realization.
    # Support for this is deprecated.  Read old style but no longer write
    # in this style.
    subfmts = tuple(
        (subfmt[0],
         subfmt[1] + r'(\((?P<scale>\w+)(\((?P<realization>\w+)\))?\))?',
         subfmt[2]) for subfmt in subfmts)
```
### 142 - astropy/time/formats.py:

Start line: 1908, End line: 1922

```python
def _validate_jd_for_storage(jd):
    if isinstance(jd, (float, int)):
        return np.array(jd, dtype=np.float_)
    if (isinstance(jd, np.generic)
        and (jd.dtype.kind == 'f' and jd.dtype.itemsize <= 8
             or jd.dtype.kind in 'iu')):
        return np.array(jd, dtype=np.float_)
    elif (isinstance(jd, np.ndarray)
          and jd.dtype.kind == 'f'
          and jd.dtype.itemsize == 8):
        return jd
    else:
        raise TypeError(
            f"JD values must be arrays (possibly zero-dimensional) "
            f"of floats but we got {jd!r} of type {type(jd)}")
```
### 143 - astropy/time/formats.py:

Start line: 1925, End line: 1949

```python
def _broadcast_writeable(jd1, jd2):
    if jd1.shape == jd2.shape:
        return jd1, jd2
    # When using broadcast_arrays, *both* are flagged with
    # warn-on-write, even the one that wasn't modified, and
    # require "C" only clears the flag if it actually copied
    # anything.
    shape = np.broadcast(jd1, jd2).shape
    if jd1.shape == shape:
        s_jd1 = jd1
    else:
        s_jd1 = np.require(np.broadcast_to(jd1, shape),
                           requirements=["C", "W"])
    if jd2.shape == shape:
        s_jd2 = jd2
    else:
        s_jd2 = np.require(np.broadcast_to(jd2, shape),
                           requirements=["C", "W"])
    return s_jd1, s_jd2


# Import symbols from core.py that are used in this module. This succeeds
# because __init__.py imports format.py just before core.py.
from .core import Time, TIME_SCALES, TIME_DELTA_SCALES, ScaleValueError  # noqa
```
### 145 - astropy/time/formats.py:

Start line: 1805, End line: 1835

```python
class TimeBesselianEpochString(TimeEpochDateString):
    """Besselian Epoch year as string value(s) like 'B1950.0'"""
    name = 'byear_str'
    epoch_to_jd = 'epb2jd'
    jd_to_epoch = 'epb'
    epoch_prefix = 'B'


class TimeJulianEpochString(TimeEpochDateString):
    """Julian Epoch year as string value(s) like 'J2000.0'"""
    name = 'jyear_str'
    epoch_to_jd = 'epj2jd'
    jd_to_epoch = 'epj'
    epoch_prefix = 'J'


class TimeDeltaFormat(TimeFormat):
    """Base class for time delta representations"""

    _registry = TIME_DELTA_FORMATS

    def _check_scale(self, scale):
        """
        Check that the scale is in the allowed list of scales, or is `None`
        """
        if scale is not None and scale not in TIME_DELTA_SCALES:
            raise ScaleValueError("Scale value '{}' not in "
                                  "allowed values {}"
                                  .format(scale, TIME_DELTA_SCALES))

        return scale
```
### 146 - astropy/time/core.py:

Start line: 1878, End line: 1952

```python
class Time(TimeBase):

    def sidereal_time(self, kind, longitude=None, model=None):
        """Calculate sidereal time.

        Parameters
        ----------
        kind : str
            ``'mean'`` or ``'apparent'``, i.e., accounting for precession
            only, or also for nutation.
        longitude : `~astropy.units.Quantity`, `~astropy.coordinates.EarthLocation`, str, or None; optional
            The longitude on the Earth at which to compute the Earth rotation
            angle (taken from a location as needed).  If `None` (default), taken
            from the ``location`` attribute of the Time instance. If the special
            string  'greenwich' or 'tio', the result will be relative to longitude
            0 for models before 2000, and relative to the Terrestrial Intermediate
            Origin (TIO) for later ones (i.e., the output of the relevant ERFA
            function that calculates greenwich sidereal time).
        model : str or None; optional
            Precession (and nutation) model to use.  The available ones are:
            - {0}: {1}
            - {2}: {3}
            If `None` (default), the last (most recent) one from the appropriate
            list above is used.

        Returns
        -------
        `~astropy.coordinates.Longitude`
            Local sidereal time, with units of hourangle.

        See Also
        --------
        astropy.time.Time.earth_rotation_angle

        References
        ----------
        IAU 2006 NFA Glossary
        (currently located at: https://syrte.obspm.fr/iauWGnfa/NFA_Glossary.html)

        Notes
        -----
        The difference between apparent sidereal time and Earth rotation angle
        is the equation of the origins, which is the angle between the Celestial
        Intermediate Origin (CIO) and the equinox. Applying apparent sidereal
        time to the hour angle yields the true apparent Right Ascension with
        respect to the equinox, while applying the Earth rotation angle yields
        the intermediate (CIRS) Right Ascension with respect to the CIO.

        For the IAU precession models from 2000 onwards, the result includes the
        TIO locator (s'), which positions the Terrestrial Intermediate Origin on
        the equator of the Celestial Intermediate Pole (CIP) and is rigorously
        corrected for polar motion (except when ``longitude='tio'`` or ``'greenwich'``).

        """  # noqa (docstring is formatted below)

        if kind.lower() not in SIDEREAL_TIME_MODELS.keys():
            raise ValueError('The kind of sidereal time has to be {}'.format(
                ' or '.join(sorted(SIDEREAL_TIME_MODELS.keys()))))

        available_models = SIDEREAL_TIME_MODELS[kind.lower()]

        if model is None:
            model = sorted(available_models.keys())[-1]
        elif model.upper() not in available_models:
            raise ValueError(
                'Model {} not implemented for {} sidereal time; '
                'available models are {}'
                .format(model, kind, sorted(available_models.keys())))

        model_kwargs = available_models[model.upper()]

        if isinstance(longitude, str) and longitude in ('tio', 'greenwich'):
            longitude = 0
            model_kwargs = model_kwargs.copy()
            model_kwargs['include_tio'] = False

        return self._sid_time_or_earth_rot_ang(longitude=longitude, **model_kwargs)
```
### 150 - astropy/time/core.py:

Start line: 1433, End line: 1455

```python
class TimeBase(ShapedLikeNDArray):

    def _time_comparison(self, other, op):
        """If other is of same class as self, compare difference in self.scale.
        Otherwise, return NotImplemented
        """
        if other.__class__ is not self.__class__:
            try:
                other = self.__class__(other, scale=self.scale)
            except Exception:
                # Let other have a go.
                return NotImplemented

        if(self.scale is not None and self.scale not in other.SCALES
           or other.scale is not None and other.scale not in self.SCALES):
            # Other will also not be able to do it, so raise a TypeError
            # immediately, allowing us to explain why it doesn't work.
            raise TypeError("Cannot compare {} instances with scales "
                            "'{}' and '{}'".format(self.__class__.__name__,
                                                   self.scale, other.scale))

        if self.scale is not None and other.scale is not None:
            other = getattr(other, self.scale)

        return op((self.jd1 - other.jd1) + (self.jd2 - other.jd2), 0.)
```
### 151 - astropy/time/formats.py:

Start line: 1163, End line: 1211

```python
class TimezoneInfo(datetime.tzinfo):
    """
    Subclass of the `~datetime.tzinfo` object, used in the
    to_datetime method to specify timezones.

    It may be safer in most cases to use a timezone database package like
    pytz rather than defining your own timezones - this class is mainly
    a workaround for users without pytz.
    """
    @u.quantity_input(utc_offset=u.day, dst=u.day)
    def __init__(self, utc_offset=0 * u.day, dst=0 * u.day, tzname=None):
        """
        Parameters
        ----------
        utc_offset : `~astropy.units.Quantity`, optional
            Offset from UTC in days. Defaults to zero.
        dst : `~astropy.units.Quantity`, optional
            Daylight Savings Time offset in days. Defaults to zero
            (no daylight savings).
        tzname : str or None, optional
            Name of timezone

        Examples
        --------
        >>> from datetime import datetime
        >>> from astropy.time import TimezoneInfo  # Specifies a timezone
        >>> import astropy.units as u
        >>> utc = TimezoneInfo()    # Defaults to UTC
        >>> utc_plus_one_hour = TimezoneInfo(utc_offset=1*u.hour)  # UTC+1
        >>> dt_aware = datetime(2000, 1, 1, 0, 0, 0, tzinfo=utc_plus_one_hour)
        >>> print(dt_aware)
        2000-01-01 00:00:00+01:00
        >>> print(dt_aware.astimezone(utc))
        1999-12-31 23:00:00+00:00
        """
        if utc_offset == 0 and dst == 0 and tzname is None:
            tzname = 'UTC'
        self._utcoffset = datetime.timedelta(utc_offset.to_value(u.day))
        self._tzname = tzname
        self._dst = datetime.timedelta(dst.to_value(u.day))

    def utcoffset(self, dt):
        return self._utcoffset

    def tzname(self, dt):
        return str(self._tzname)

    def dst(self, dt):
        return self._dst
```
### 156 - astropy/time/core.py:

Start line: 1329, End line: 1344

```python
class TimeBase(ShapedLikeNDArray):

    def ptp(self, axis=None, out=None, keepdims=False):
        """Peak to peak (maximum - minimum) along a given axis.

        This is similar to :meth:`~numpy.ndarray.ptp`, but adapted to ensure
        that the full precision given by the two doubles ``jd1`` and ``jd2``
        is used.

        Note that the ``out`` argument is present only for compatibility with
        `~numpy.ptp`; since `Time` instances are immutable, it is not possible
        to have an actual ``out`` to store the result in.
        """
        if out is not None:
            raise ValueError("Since `Time` instances are immutable, ``out`` "
                             "cannot be set to anything but ``None``.")
        return (self.max(axis, keepdims=keepdims)
                - self.min(axis, keepdims=keepdims))
```
### 157 - astropy/time/formats.py:

Start line: 1025, End line: 1056

```python
class TimeYMDHMS(TimeUnique):
    """
    ymdhms: A Time format to represent Time as year, month, day, hour,
    minute, second (thus the name ymdhms).

    Acceptable inputs must have keys or column names in the "YMDHMS" set of
    ``year``, ``month``, ``day`` ``hour``, ``minute``, ``second``:

    - Dict with keys in the YMDHMS set
    - NumPy structured array, record array or astropy Table, or single row
      of those types, with column names in the YMDHMS set

    One can supply a subset of the YMDHMS values, for instance only 'year',
    'month', and 'day'.  Inputs have the following defaults::

      'month': 1, 'day': 1, 'hour': 0, 'minute': 0, 'second': 0

    When the input is supplied as a ``dict`` then each value can be either a
    scalar value or an array.  The values will be broadcast to a common shape.

    Example::

      >>> from astropy.time import Time
      >>> t = Time({'year': 2015, 'month': 2, 'day': 3,
      ...           'hour': 12, 'minute': 13, 'second': 14.567},
      ...           scale='utc')
      >>> t.iso
      '2015-02-03 12:13:14.567'
      >>> t.ymdhms.year
      2015
    """
    name = 'ymdhms'
```
### 158 - astropy/time/core.py:

Start line: 2519, End line: 2549

```python
class TimeDelta(TimeBase):

    def __mul__(self, other):
        """Multiplication of `TimeDelta` objects by numbers/arrays."""
        # Check needed since otherwise the self.jd1 * other multiplication
        # would enter here again (via __rmul__)
        if isinstance(other, Time):
            raise OperandTypeError(self, other, '*')
        elif ((isinstance(other, u.UnitBase)
               and other == u.dimensionless_unscaled)
                or (isinstance(other, str) and other == '')):
            return self.copy()

        # If other is something consistent with a dimensionless quantity
        # (could just be a float or an array), then we can just multiple in.
        try:
            other = u.Quantity(other, u.dimensionless_unscaled, copy=False)
        except Exception:
            # If not consistent with a dimensionless quantity, try downgrading
            # self to a quantity and see if things work.
            try:
                return self.to(u.day) * other
            except Exception:
                # The various ways we could multiply all failed;
                # returning NotImplemented to give other a final chance.
                return NotImplemented

        jd1, jd2 = day_frac(self.jd1, self.jd2, factor=other.value)
        out = TimeDelta(jd1, jd2, format='jd', scale=self.scale)

        if self.format != 'jd':
            out = out.replicate(format=self.format)
        return out
```
### 159 - astropy/time/formats.py:

Start line: 143, End line: 155

```python
class TimeFormat:

    @classmethod
    def _get_allowed_subfmt(cls, subfmt):
        """Get an allowed subfmt for this class, either the input ``subfmt``
        if this is valid or '*' as a default.  This method gets used in situations
        where the format of an existing Time object is changing and so the
        out_ or in_subfmt may need to be coerced to the default '*' if that
        ``subfmt`` is no longer valid.
        """
        try:
            cls._select_subfmts(subfmt)
        except ValueError:
            subfmt = '*'
        return subfmt
```
### 160 - astropy/time/core.py:

Start line: 876, End line: 948

```python
class TimeBase(ShapedLikeNDArray):

    def insert(self, obj, values, axis=0):
        """
        Insert values before the given indices in the column and return
        a new `~astropy.time.Time` or  `~astropy.time.TimeDelta` object.

        The values to be inserted must conform to the rules for in-place setting
        of ``Time`` objects (see ``Get and set values`` in the ``Time``
        documentation).

        The API signature matches the ``np.insert`` API, but is more limited.
        The specification of insert index ``obj`` must be a single integer,
        and the ``axis`` must be ``0`` for simple row insertion before the
        index.

        Parameters
        ----------
        obj : int
            Integer index before which ``values`` is inserted.
        values : array-like
            Value(s) to insert.  If the type of ``values`` is different
            from that of quantity, ``values`` is converted to the matching type.
        axis : int, optional
            Axis along which to insert ``values``.  Default is 0, which is the
            only allowed value and will insert a row.

        Returns
        -------
        out : `~astropy.time.Time` subclass
            New time object with inserted value(s)

        """
        # Validate inputs: obj arg is integer, axis=0, self is not a scalar, and
        # input index is in bounds.
        try:
            idx0 = operator.index(obj)
        except TypeError:
            raise TypeError('obj arg must be an integer')

        if axis != 0:
            raise ValueError('axis must be 0')

        if not self.shape:
            raise TypeError('cannot insert into scalar {} object'
                            .format(self.__class__.__name__))

        if abs(idx0) > len(self):
            raise IndexError('index {} is out of bounds for axis 0 with size {}'
                             .format(idx0, len(self)))

        # Turn negative index into positive
        if idx0 < 0:
            idx0 = len(self) + idx0

        # For non-Time object, use numpy to help figure out the length.  (Note annoying
        # case of a string input that has a length which is not the length we want).
        if not isinstance(values, self.__class__):
            values = np.asarray(values)
        n_values = len(values) if values.shape else 1

        # Finally make the new object with the correct length and set values for the
        # three sections, before insert, the insert, and after the insert.
        out = self.__class__.info.new_like([self], len(self) + n_values, name=self.info.name)

        out._time.jd1[:idx0] = self._time.jd1[:idx0]
        out._time.jd2[:idx0] = self._time.jd2[:idx0]

        # This uses the Time setting machinery to coerce and validate as necessary.
        out[idx0:idx0 + n_values] = values

        out._time.jd1[idx0 + n_values:] = self._time.jd1[idx0:]
        out._time.jd2[idx0 + n_values:] = self._time.jd2[idx0:]

        return out
```
### 161 - astropy/time/core.py:

Start line: 1175, End line: 1189

```python
class TimeBase(ShapedLikeNDArray):

    def __copy__(self):
        """
        Overrides the default behavior of the `copy.copy` function in
        the python stdlib to behave like `Time.copy`. Does *not* make a
        copy of the JD arrays - only copies by reference.
        """
        return self.replicate()

    def __deepcopy__(self, memo):
        """
        Overrides the default behavior of the `copy.deepcopy` function
        in the python stdlib to behave like `Time.copy`. Does make a
        copy of the JD arrays.
        """
        return self.copy()
```
### 171 - astropy/time/core.py:

Start line: 1745, End line: 1813

```python
class Time(TimeBase):

    def light_travel_time(self, skycoord, kind='barycentric', location=None, ephemeris=None):
        """Light travel time correction to the barycentre or heliocentre.

        The frame transformations used to calculate the location of the solar
        system barycentre and the heliocentre rely on the erfa routine epv00,
        which is consistent with the JPL DE405 ephemeris to an accuracy of
        11.2 km, corresponding to a light travel time of 4 microseconds.

        The routine assumes the source(s) are at large distance, i.e., neglects
        finite-distance effects.

        Parameters
        ----------
        skycoord : `~astropy.coordinates.SkyCoord`
            The sky location to calculate the correction for.
        kind : str, optional
            ``'barycentric'`` (default) or ``'heliocentric'``
        location : `~astropy.coordinates.EarthLocation`, optional
            The location of the observatory to calculate the correction for.
            If no location is given, the ``location`` attribute of the Time
            object is used
        ephemeris : str, optional
            Solar system ephemeris to use (e.g., 'builtin', 'jpl'). By default,
            use the one set with ``astropy.coordinates.solar_system_ephemeris.set``.
            For more information, see `~astropy.coordinates.solar_system_ephemeris`.

        Returns
        -------
        time_offset : `~astropy.time.TimeDelta`
            The time offset between the barycentre or Heliocentre and Earth,
            in TDB seconds.  Should be added to the original time to get the
            time in the Solar system barycentre or the Heliocentre.
            Also, the time conversion to BJD will then include the relativistic correction as well.
        """

        if kind.lower() not in ('barycentric', 'heliocentric'):
            raise ValueError("'kind' parameter must be one of 'heliocentric' "
                             "or 'barycentric'")

        if location is None:
            if self.location is None:
                raise ValueError('An EarthLocation needs to be set or passed '
                                 'in to calculate bary- or heliocentric '
                                 'corrections')
            location = self.location

        from astropy.coordinates import (UnitSphericalRepresentation, CartesianRepresentation,
                                         HCRS, ICRS, GCRS, solar_system_ephemeris)

        # ensure sky location is ICRS compatible
        if not skycoord.is_transformable_to(ICRS()):
            raise ValueError("Given skycoord is not transformable to the ICRS")

        # get location of observatory in ITRS coordinates at this Time
        try:
            itrs = location.get_itrs(obstime=self)
        except Exception:
            raise ValueError("Supplied location does not have a valid `get_itrs` method")

        with solar_system_ephemeris.set(ephemeris):
            if kind.lower() == 'heliocentric':
                # convert to heliocentric coordinates, aligned with ICRS
                cpos = itrs.transform_to(HCRS(obstime=self)).cartesian.xyz
            else:
                # first we need to convert to GCRS coordinates with the correct
                # obstime, since ICRS coordinates have no frame time
                gcrs_coo = itrs.transform_to(GCRS(obstime=self))
                # convert to barycentric (BCRS) coordinates, aligned with ICRS
                cpos = gcrs_coo.transform_to(ICRS()).cartesian.xyz
        # ... other code
```
### 173 - astropy/time/core.py:

Start line: 326, End line: 378

```python
class TimeDeltaInfo(TimeInfoBase):
    """
    Container for meta information like name, description, format.  This is
    required when the object is used as a mixin column within a table, but can
    be used as a general way to store meta information.
    """
    _represent_as_dict_extra_attrs = ('format', 'scale')

    def new_like(self, cols, length, metadata_conflicts='warn', name=None):
        """
        Return a new TimeDelta instance which is consistent with the input Time objects
        ``cols`` and has ``length`` rows.

        This is intended for creating an empty Time instance whose elements can
        be set in-place for table operations like join or vstack.  It checks
        that the input locations and attributes are consistent.  This is used
        when a Time object is used as a mixin column in an astropy Table.

        Parameters
        ----------
        cols : list
            List of input columns (Time objects)
        length : int
            Length of the output column object
        metadata_conflicts : str ('warn'|'error'|'silent')
            How to handle metadata conflicts
        name : str
            Output column name

        Returns
        -------
        col : Time (or subclass)
            Empty instance of this class consistent with ``cols``

        """
        # Get merged info attributes like shape, dtype, format, description, etc.
        attrs = self.merge_cols_attributes(cols, metadata_conflicts, name,
                                           ('meta', 'description'))
        attrs.pop('dtype')  # Not relevant for Time
        col0 = cols[0]

        # Make a new Time object with the desired shape and attributes
        shape = (length,) + attrs.pop('shape')
        jd1 = np.zeros(shape, dtype='f8')
        jd2 = np.zeros(shape, dtype='f8')
        out = self._parent_cls(jd1, jd2, format='jd', scale=col0.scale)
        out.format = col0.format

        # Set remaining info attributes
        for attr, value in attrs.items():
            setattr(out.info, attr, value)

        return out
```
### 174 - astropy/time/core.py:

Start line: 1827, End line: 1876

```python
class Time(TimeBase):

    def earth_rotation_angle(self, longitude=None):
        """Calculate local Earth rotation angle.

        Parameters
        ----------
        longitude : `~astropy.units.Quantity`, `~astropy.coordinates.EarthLocation`, str, or None; optional
            The longitude on the Earth at which to compute the Earth rotation
            angle (taken from a location as needed).  If `None` (default), taken
            from the ``location`` attribute of the Time instance. If the special
            string 'tio', the result will be relative to the Terrestrial
            Intermediate Origin (TIO) (i.e., the output of `~erfa.era00`).

        Returns
        -------
        `~astropy.coordinates.Longitude`
            Local Earth rotation angle with units of hourangle.

        See Also
        --------
        astropy.time.Time.sidereal_time

        References
        ----------
        IAU 2006 NFA Glossary
        (currently located at: https://syrte.obspm.fr/iauWGnfa/NFA_Glossary.html)

        Notes
        -----
        The difference between apparent sidereal time and Earth rotation angle
        is the equation of the origins, which is the angle between the Celestial
        Intermediate Origin (CIO) and the equinox. Applying apparent sidereal
        time to the hour angle yields the true apparent Right Ascension with
        respect to the equinox, while applying the Earth rotation angle yields
        the intermediate (CIRS) Right Ascension with respect to the CIO.

        The result includes the TIO locator (s'), which positions the Terrestrial
        Intermediate Origin on the equator of the Celestial Intermediate Pole (CIP)
        and is rigorously corrected for polar motion.
        (except when ``longitude='tio'``).

        """  # noqa
        if isinstance(longitude, str) and longitude == 'tio':
            longitude = 0
            include_tio = False
        else:
            include_tio = True

        return self._sid_time_or_earth_rot_ang(longitude=longitude,
                                               function=erfa.era00, scales=('ut1',),
                                               include_tio=include_tio)
```
