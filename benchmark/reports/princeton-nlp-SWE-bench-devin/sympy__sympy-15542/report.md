# sympy__sympy-15542

| **sympy/sympy** | `495e749818bbcd55dc0d9ee7101cb36646e4277a` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 824 |
| **Any found context length** | 824 |
| **Avg pos** | 6.0 |
| **Min pos** | 3 |
| **Max pos** | 3 |
| **Top file pos** | 2 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sympy/geometry/point.py b/sympy/geometry/point.py
--- a/sympy/geometry/point.py
+++ b/sympy/geometry/point.py
@@ -380,19 +380,20 @@ def are_coplanar(cls, *points):
         points = list(uniq(points))
         return Point.affine_rank(*points) <= 2
 
-    def distance(self, p):
-        """The Euclidean distance from self to point p.
-
-        Parameters
-        ==========
-
-        p : Point
+    def distance(self, other):
+        """The Euclidean distance between self and another GeometricEntity.
 
         Returns
         =======
 
         distance : number or symbolic expression.
 
+        Raises
+        ======
+        AttributeError : if other is a GeometricEntity for which
+                         distance is not defined.
+        TypeError : if other is not recognized as a GeometricEntity.
+
         See Also
         ========
 
@@ -402,19 +403,34 @@ def distance(self, p):
         Examples
         ========
 
-        >>> from sympy.geometry import Point
+        >>> from sympy.geometry import Point, Line
         >>> p1, p2 = Point(1, 1), Point(4, 5)
+        >>> l = Line((3, 1), (2, 2))
         >>> p1.distance(p2)
         5
+        >>> p1.distance(l)
+        sqrt(2)
+
+        The computed distance may be symbolic, too:
 
         >>> from sympy.abc import x, y
         >>> p3 = Point(x, y)
-        >>> p3.distance(Point(0, 0))
+        >>> p3.distance((0, 0))
         sqrt(x**2 + y**2)
 
         """
-        s, p = Point._normalize_dimension(self, Point(p))
-        return sqrt(Add(*((a - b)**2 for a, b in zip(s, p))))
+        if not isinstance(other , GeometryEntity) :
+            try :
+                other = Point(other, dim=self.ambient_dimension)
+            except TypeError :
+                raise TypeError("not recognized as a GeometricEntity: %s" % type(other))
+        if isinstance(other , Point) :
+            s, p = Point._normalize_dimension(self, Point(other))
+            return sqrt(Add(*((a - b)**2 for a, b in zip(s, p))))
+        try :
+            return other.distance(self)
+        except AttributeError :
+            raise AttributeError("distance between Point and %s is not defined" % type(other))
 
     def dot(self, p):
         """Return dot product of self with another Point."""

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sympy/geometry/point.py | 383 | 389 | 3 | 2 | 824
| sympy/geometry/point.py | 405 | 417 | 3 | 2 | 824


## Problem Statement

```
Should Point.distance(Line) return distance?
In Geometry module, `Line.distance(Point)` can be used to compute distance, but `Point.distance(Line)` cannot. Should this be made symmetric? 
\`\`\`
>>> L = Line((1, 1), (2, 2))
>>> P = Point(1, 0)
>>> L.distance(P)
sqrt(2)/2
>>> P.distance(L)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/home/k3/sympy/sympy/geometry/point.py", line 416, in distance
    s, p = Point._normalize_dimension(self, Point(p))
  File "/home/k3/sympy/sympy/geometry/point.py", line 129, in __new__
    .format(func_name(coords))))
TypeError: 
Expecting sequence of coordinates, not `Line2D`
\`\`\`

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 sympy/geometry/line.py | 1182 | 1213| 261 | 261 | 20788 | 
| 2 | **2 sympy/geometry/point.py** | 183 | 225| 352 | 613 | 29986 | 
| **-> 3 <-** | **2 sympy/geometry/point.py** | 383 | 417| 211 | 824 | 29986 | 
| 4 | 2 sympy/geometry/line.py | 1605 | 1651| 428 | 1252 | 29986 | 
| 5 | 2 sympy/geometry/line.py | 1931 | 1960| 318 | 1570 | 29986 | 
| 6 | 2 sympy/geometry/line.py | 1215 | 1253| 288 | 1858 | 29986 | 
| 7 | 2 sympy/geometry/line.py | 1394 | 1432| 323 | 2181 | 29986 | 
| 8 | **2 sympy/geometry/point.py** | 753 | 784| 185 | 2366 | 29986 | 
| 9 | 2 sympy/geometry/line.py | 2446 | 2462| 208 | 2574 | 29986 | 
| 10 | 2 sympy/geometry/line.py | 644 | 659| 139 | 2713 | 29986 | 
| 11 | **2 sympy/geometry/point.py** | 1 | 41| 293 | 3006 | 29986 | 
| 12 | **2 sympy/geometry/point.py** | 786 | 845| 350 | 3356 | 29986 | 
| 13 | 2 sympy/geometry/line.py | 1653 | 1677| 186 | 3542 | 29986 | 
| 14 | 3 sympy/geometry/plane.py | 247 | 306| 529 | 4071 | 37597 | 
| 15 | **3 sympy/geometry/point.py** | 419 | 430| 133 | 4204 | 37597 | 
| 16 | **3 sympy/geometry/point.py** | 688 | 712| 258 | 4462 | 37597 | 
| 17 | 3 sympy/geometry/line.py | 868 | 892| 147 | 4609 | 37597 | 
| 18 | 3 sympy/geometry/line.py | 1102 | 1145| 399 | 5008 | 37597 | 
| 19 | **3 sympy/geometry/point.py** | 504 | 540| 269 | 5277 | 37597 | 
| 20 | 4 sympy/integrals/intpoly.py | 360 | 387| 282 | 5559 | 49387 | 
| 21 | 5 sympy/geometry/polygon.py | 953 | 1012| 609 | 6168 | 67712 | 
| 22 | 5 sympy/geometry/line.py | 2390 | 2411| 185 | 6353 | 67712 | 
| 23 | **5 sympy/geometry/point.py** | 111 | 181| 621 | 6974 | 67712 | 
| 24 | 5 sympy/geometry/line.py | 545 | 594| 426 | 7400 | 67712 | 
| 25 | 5 sympy/geometry/line.py | 1495 | 1560| 566 | 7966 | 67712 | 
| 26 | **5 sympy/geometry/point.py** | 626 | 650| 158 | 8124 | 67712 | 
| 27 | 5 sympy/geometry/line.py | 480 | 543| 608 | 8732 | 67712 | 
| 28 | 5 sympy/geometry/line.py | 596 | 642| 391 | 9123 | 67712 | 
| 29 | 5 sympy/geometry/line.py | 245 | 296| 405 | 9528 | 67712 | 
| 30 | 5 sympy/geometry/line.py | 2267 | 2313| 329 | 9857 | 67712 | 
| 31 | **5 sympy/geometry/point.py** | 1172 | 1194| 133 | 9990 | 67712 | 
| 32 | 6 sympy/physics/vector/point.py | 218 | 244| 187 | 10177 | 71464 | 
| 33 | 6 sympy/geometry/line.py | 661 | 696| 205 | 10382 | 71464 | 
| 34 | 6 sympy/geometry/line.py | 2112 | 2154| 511 | 10893 | 71464 | 
| 35 | 6 sympy/geometry/line.py | 698 | 717| 119 | 11012 | 71464 | 
| 36 | 6 sympy/geometry/line.py | 1147 | 1180| 277 | 11289 | 71464 | 
| 37 | 6 sympy/geometry/line.py | 719 | 760| 301 | 11590 | 71464 | 
| 38 | **6 sympy/geometry/point.py** | 1145 | 1170| 179 | 11769 | 71464 | 
| 39 | **6 sympy/geometry/point.py** | 284 | 310| 264 | 12033 | 71464 | 
| 40 | 6 sympy/geometry/line.py | 1310 | 1320| 124 | 12157 | 71464 | 
| 41 | **6 sympy/geometry/point.py** | 847 | 915| 417 | 12574 | 71464 | 
| 42 | 6 sympy/geometry/polygon.py | 903 | 952| 516 | 13090 | 71464 | 
| 43 | 6 sympy/geometry/line.py | 2369 | 2388| 153 | 13243 | 71464 | 
| 44 | 6 sympy/geometry/line.py | 1679 | 1703| 210 | 13453 | 71464 | 
| 45 | **6 sympy/geometry/point.py** | 598 | 624| 269 | 13722 | 71464 | 
| 46 | 7 sympy/geometry/__init__.py | 1 | 24| 195 | 13917 | 71660 | 
| 47 | 7 sympy/geometry/line.py | 83 | 107| 237 | 14154 | 71660 | 
| 48 | **7 sympy/geometry/point.py** | 44 | 109| 493 | 14647 | 71660 | 
| 49 | 7 sympy/geometry/line.py | 142 | 204| 516 | 15163 | 71660 | 
| 50 | **7 sympy/geometry/point.py** | 1023 | 1051| 142 | 15305 | 71660 | 
| 51 | 7 sympy/geometry/line.py | 762 | 805| 327 | 15632 | 71660 | 
| 52 | 7 sympy/geometry/line.py | 959 | 980| 244 | 15876 | 71660 | 
| 53 | **7 sympy/geometry/point.py** | 714 | 751| 226 | 16102 | 71660 | 
| 54 | 7 sympy/geometry/line.py | 1814 | 1849| 230 | 16332 | 71660 | 
| 55 | **7 sympy/geometry/point.py** | 227 | 251| 169 | 16501 | 71660 | 
| 56 | 8 sympy/vector/point.py | 123 | 157| 240 | 16741 | 72731 | 
| 57 | 8 sympy/geometry/line.py | 1705 | 1746| 300 | 17041 | 72731 | 
| 58 | **8 sympy/geometry/point.py** | 652 | 686| 202 | 17243 | 72731 | 
| 59 | 8 sympy/geometry/line.py | 390 | 438| 495 | 17738 | 72731 | 
| 60 | 8 sympy/geometry/line.py | 807 | 866| 498 | 18236 | 72731 | 
| 61 | 8 sympy/geometry/line.py | 1 | 42| 275 | 18511 | 72731 | 
| 62 | 8 sympy/geometry/polygon.py | 786 | 813| 218 | 18729 | 72731 | 
| 63 | 8 sympy/geometry/line.py | 464 | 478| 185 | 18914 | 72731 | 
| 64 | **8 sympy/geometry/point.py** | 1308 | 1352| 219 | 19133 | 72731 | 
| 65 | 8 sympy/geometry/line.py | 439 | 451| 155 | 19288 | 72731 | 
| 66 | **8 sympy/geometry/point.py** | 1196 | 1232| 225 | 19513 | 72731 | 
| 67 | 8 sympy/geometry/line.py | 1886 | 1930| 302 | 19815 | 72731 | 
| 68 | 8 sympy/geometry/line.py | 109 | 140| 192 | 20007 | 72731 | 
| 69 | 8 sympy/geometry/line.py | 2683 | 2729| 373 | 20380 | 72731 | 
| 70 | 8 sympy/geometry/plane.py | 636 | 700| 667 | 21047 | 72731 | 
| 71 | **8 sympy/geometry/point.py** | 1109 | 1143| 263 | 21310 | 72731 | 
| 72 | 8 sympy/geometry/line.py | 206 | 243| 267 | 21577 | 72731 | 
| 73 | **8 sympy/geometry/point.py** | 337 | 381| 337 | 21914 | 72731 | 
| 74 | 9 sympy/geometry/util.py | 503 | 525| 171 | 22085 | 78067 | 
| 75 | 9 sympy/geometry/line.py | 453 | 462| 148 | 22233 | 78067 | 
| 76 | 9 sympy/geometry/line.py | 894 | 957| 588 | 22821 | 78067 | 
| 77 | 9 sympy/geometry/line.py | 1031 | 1100| 558 | 23379 | 78067 | 
| 78 | 10 sympy/geometry/ellipse.py | 861 | 940| 845 | 24224 | 89941 | 
| 79 | **10 sympy/geometry/point.py** | 463 | 502| 234 | 24458 | 89941 | 
| 80 | **10 sympy/geometry/point.py** | 1053 | 1107| 363 | 24821 | 89941 | 
| 81 | 10 sympy/vector/point.py | 45 | 93| 354 | 25175 | 89941 | 
| 82 | 10 sympy/geometry/util.py | 478 | 501| 267 | 25442 | 89941 | 
| 83 | 10 sympy/geometry/line.py | 1986 | 2019| 292 | 25734 | 89941 | 
| 84 | 10 sympy/geometry/polygon.py | 815 | 901| 781 | 26515 | 89941 | 
| 85 | 10 sympy/geometry/line.py | 2564 | 2582| 224 | 26739 | 89941 | 
| 86 | 10 sympy/geometry/line.py | 2464 | 2513| 522 | 27261 | 89941 | 
| 87 | 10 sympy/geometry/util.py | 252 | 328| 601 | 27862 | 89941 | 
| 88 | **10 sympy/geometry/point.py** | 951 | 976| 212 | 28074 | 89941 | 
| 89 | 10 sympy/geometry/plane.py | 464 | 502| 247 | 28321 | 89941 | 
| 90 | 10 sympy/geometry/line.py | 350 | 388| 243 | 28564 | 89941 | 
| 91 | **10 sympy/geometry/point.py** | 542 | 596| 430 | 28994 | 89941 | 
| 92 | 10 sympy/geometry/line.py | 2021 | 2061| 276 | 29270 | 89941 | 
| 93 | 11 sympy/diffgeom/diffgeom.py | 349 | 406| 445 | 29715 | 104384 | 
| 94 | 11 sympy/geometry/plane.py | 53 | 73| 248 | 29963 | 104384 | 
| 95 | 11 sympy/vector/point.py | 95 | 121| 176 | 30139 | 104384 | 
| 96 | 11 sympy/geometry/util.py | 442 | 476| 280 | 30419 | 104384 | 
| 97 | 12 sympy/geometry/entity.py | 357 | 414| 591 | 31010 | 109571 | 
| 98 | 12 sympy/geometry/plane.py | 611 | 634| 164 | 31174 | 109571 | 
| 99 | 12 sympy/geometry/line.py | 2064 | 2111| 292 | 31466 | 109571 | 
| 100 | 12 sympy/geometry/plane.py | 702 | 752| 509 | 31975 | 109571 | 
| 101 | **12 sympy/geometry/point.py** | 432 | 461| 194 | 32169 | 109571 | 
| 102 | 12 sympy/geometry/polygon.py | 2447 | 2471| 195 | 32364 | 109571 | 
| 103 | 12 sympy/geometry/polygon.py | 1 | 24| 201 | 32565 | 109571 | 
| 104 | 12 sympy/geometry/line.py | 1434 | 1467| 225 | 32790 | 109571 | 
| 105 | 12 sympy/geometry/line.py | 45 | 81| 238 | 33028 | 109571 | 
| 106 | 12 sympy/geometry/line.py | 1562 | 1603| 434 | 33462 | 109571 | 
| 107 | 12 sympy/geometry/line.py | 2414 | 2444| 236 | 33698 | 109571 | 
| 108 | 12 sympy/geometry/line.py | 1469 | 1492| 174 | 33872 | 109571 | 
| 109 | 12 sympy/geometry/plane.py | 828 | 891| 595 | 34467 | 109571 | 
| 110 | 12 sympy/geometry/plane.py | 505 | 544| 251 | 34718 | 109571 | 
| 111 | 12 sympy/integrals/intpoly.py | 1062 | 1087| 198 | 34916 | 109571 | 
| 112 | 12 sympy/geometry/ellipse.py | 667 | 690| 244 | 35160 | 109571 | 
| 113 | 12 sympy/geometry/plane.py | 546 | 562| 155 | 35315 | 109571 | 
| 114 | 12 sympy/geometry/line.py | 298 | 348| 453 | 35768 | 109571 | 
| 115 | **12 sympy/geometry/point.py** | 1234 | 1259| 229 | 35997 | 109571 | 
| 116 | 12 sympy/geometry/ellipse.py | 1181 | 1258| 645 | 36642 | 109571 | 
| 117 | 13 sympy/geometry/parabola.py | 102 | 125| 157 | 36799 | 112320 | 
| 118 | 13 sympy/integrals/intpoly.py | 1090 | 1146| 689 | 37488 | 112320 | 
| 119 | 13 sympy/geometry/line.py | 1256 | 1309| 338 | 37826 | 112320 | 
| 120 | 13 sympy/geometry/line.py | 982 | 1028| 337 | 38163 | 112320 | 
| 121 | 13 sympy/geometry/polygon.py | 1795 | 1835| 404 | 38567 | 112320 | 
| 122 | 13 sympy/geometry/line.py | 2220 | 2264| 357 | 38924 | 112320 | 
| 123 | 13 sympy/geometry/line.py | 2516 | 2562| 295 | 39219 | 112320 | 
| 124 | 13 sympy/vector/point.py | 1 | 43| 314 | 39533 | 112320 | 
| 125 | **13 sympy/geometry/point.py** | 1000 | 1021| 154 | 39687 | 112320 | 
| 126 | 14 sympy/vector/functions.py | 324 | 390| 590 | 40277 | 116183 | 
| 127 | 14 sympy/geometry/plane.py | 139 | 194| 693 | 40970 | 116183 | 
| 128 | 14 sympy/geometry/polygon.py | 1036 | 1055| 198 | 41168 | 116183 | 
| 129 | 14 sympy/geometry/line.py | 2338 | 2367| 194 | 41362 | 116183 | 
| 130 | 14 sympy/geometry/polygon.py | 679 | 700| 209 | 41571 | 116183 | 
| 131 | 14 sympy/geometry/parabola.py | 127 | 151| 154 | 41725 | 116183 | 
| 132 | 14 sympy/geometry/util.py | 114 | 173| 392 | 42117 | 116183 | 
| 133 | 14 sympy/geometry/plane.py | 309 | 332| 199 | 42316 | 116183 | 
| 134 | 14 sympy/geometry/ellipse.py | 331 | 350| 114 | 42430 | 116183 | 
| 135 | **14 sympy/geometry/point.py** | 253 | 282| 229 | 42659 | 116183 | 
| 136 | **14 sympy/geometry/point.py** | 312 | 335| 228 | 42887 | 116183 | 
| 137 | 14 sympy/geometry/polygon.py | 1863 | 1915| 506 | 43393 | 116183 | 
| 138 | 14 sympy/geometry/parabola.py | 1 | 16| 118 | 43511 | 116183 | 
| 139 | 14 sympy/geometry/util.py | 1 | 40| 259 | 43770 | 116183 | 
| 140 | 14 sympy/geometry/util.py | 176 | 249| 708 | 44478 | 116183 | 
| 141 | 14 sympy/geometry/line.py | 2188 | 2218| 218 | 44696 | 116183 | 
| 142 | 14 sympy/geometry/line.py | 1962 | 1984| 239 | 44935 | 116183 | 
| 143 | 15 sympy/plotting/plot.py | 660 | 707| 470 | 45405 | 133237 | 
| 144 | 15 sympy/geometry/entity.py | 505 | 533| 279 | 45684 | 133237 | 
| 145 | 15 sympy/geometry/parabola.py | 227 | 260| 194 | 45878 | 133237 | 
| 146 | 15 sympy/geometry/plane.py | 1 | 23| 177 | 46055 | 133237 | 
| 147 | 15 sympy/geometry/parabola.py | 153 | 186| 210 | 46265 | 133237 | 
| 148 | 15 sympy/plotting/plot.py | 598 | 657| 719 | 46984 | 133237 | 
| 149 | 15 sympy/geometry/polygon.py | 114 | 156| 384 | 47368 | 133237 | 
| 150 | 15 sympy/geometry/util.py | 43 | 111| 603 | 47971 | 133237 | 
| 151 | 15 sympy/geometry/polygon.py | 2495 | 2512| 191 | 48162 | 133237 | 
| 152 | 15 sympy/geometry/plane.py | 585 | 609| 158 | 48320 | 133237 | 
| 153 | 15 sympy/geometry/line.py | 1851 | 1883| 198 | 48518 | 133237 | 
| 154 | 15 sympy/plotting/plot.py | 541 | 575| 360 | 48878 | 133237 | 
| 155 | 15 sympy/geometry/plane.py | 75 | 89| 167 | 49045 | 133237 | 
| 156 | 15 sympy/geometry/line.py | 1346 | 1392| 421 | 49466 | 133237 | 
| 157 | 15 sympy/geometry/ellipse.py | 558 | 584| 137 | 49603 | 133237 | 
| 158 | 15 sympy/geometry/polygon.py | 1398 | 1438| 220 | 49823 | 133237 | 
| 159 | 15 sympy/geometry/plane.py | 754 | 791| 310 | 50133 | 133237 | 
| 160 | 15 sympy/geometry/plane.py | 26 | 52| 297 | 50430 | 133237 | 
| 161 | 15 sympy/geometry/plane.py | 335 | 354| 243 | 50673 | 133237 | 
| 162 | 15 sympy/geometry/polygon.py | 378 | 438| 679 | 51352 | 133237 | 
| 163 | 15 sympy/geometry/line.py | 1748 | 1775| 176 | 51528 | 133237 | 
| 164 | 15 sympy/geometry/polygon.py | 1947 | 1969| 136 | 51664 | 133237 | 
| 165 | 15 sympy/geometry/line.py | 2156 | 2186| 217 | 51881 | 133237 | 
| 166 | 15 sympy/diffgeom/diffgeom.py | 246 | 346| 768 | 52649 | 133237 | 
| 167 | 15 sympy/geometry/polygon.py | 1971 | 1993| 133 | 52782 | 133237 | 
| 168 | **15 sympy/geometry/point.py** | 1285 | 1306| 180 | 52962 | 133237 | 
| 169 | 15 sympy/geometry/polygon.py | 2059 | 2088| 198 | 53160 | 133237 | 
| 170 | 15 sympy/geometry/parabola.py | 81 | 100| 121 | 53281 | 133237 | 
| 171 | 15 sympy/geometry/polygon.py | 1917 | 1945| 195 | 53476 | 133237 | 
| 172 | 15 sympy/geometry/plane.py | 357 | 435| 772 | 54248 | 133237 | 
| 173 | 15 sympy/geometry/polygon.py | 1995 | 2021| 174 | 54422 | 133237 | 
| 174 | 15 sympy/geometry/polygon.py | 2120 | 2145| 176 | 54598 | 133237 | 
| 175 | 15 sympy/geometry/polygon.py | 342 | 375| 265 | 54863 | 133237 | 
| 176 | 15 sympy/physics/vector/point.py | 275 | 304| 210 | 55073 | 133237 | 
| 177 | 15 sympy/geometry/line.py | 2616 | 2646| 241 | 55314 | 133237 | 
| 178 | 15 sympy/physics/vector/point.py | 458 | 499| 300 | 55614 | 133237 | 
| 179 | 15 sympy/geometry/ellipse.py | 123 | 155| 327 | 55941 | 133237 | 
| 180 | 15 sympy/plotting/plot.py | 424 | 438| 112 | 56053 | 133237 | 
| 181 | 15 sympy/plotting/plot.py | 373 | 401| 237 | 56290 | 133237 | 
| 182 | 15 sympy/geometry/polygon.py | 1666 | 1692| 238 | 56528 | 133237 | 
| 183 | 16 sympy/combinatorics/permutations.py | 2663 | 2688| 214 | 56742 | 155755 | 
| 184 | 16 sympy/geometry/polygon.py | 620 | 677| 458 | 57200 | 155755 | 
| 185 | 16 sympy/geometry/polygon.py | 1837 | 1861| 141 | 57341 | 155755 | 
| 186 | 16 sympy/physics/vector/point.py | 1 | 33| 212 | 57553 | 155755 | 
| 187 | 16 sympy/geometry/polygon.py | 1295 | 1339| 232 | 57785 | 155755 | 
| 188 | 16 sympy/geometry/plane.py | 564 | 583| 144 | 57929 | 155755 | 
| 189 | 16 sympy/integrals/intpoly.py | 1026 | 1059| 382 | 58311 | 155755 | 
| 190 | 16 sympy/geometry/plane.py | 438 | 461| 249 | 58560 | 155755 | 
| 191 | 16 sympy/geometry/polygon.py | 441 | 484| 367 | 58927 | 155755 | 
| 192 | 16 sympy/geometry/entity.py | 102 | 126| 233 | 59160 | 155755 | 
| 193 | 16 sympy/geometry/line.py | 1778 | 1812| 177 | 59337 | 155755 | 
| 194 | 16 sympy/geometry/line.py | 2648 | 2680| 252 | 59589 | 155755 | 
| 195 | 16 sympy/geometry/ellipse.py | 1499 | 1529| 219 | 59808 | 155755 | 
| 196 | 16 sympy/geometry/ellipse.py | 352 | 403| 340 | 60148 | 155755 | 
| 197 | 16 sympy/geometry/ellipse.py | 1092 | 1134| 438 | 60586 | 155755 | 
| 198 | 16 sympy/geometry/plane.py | 793 | 826| 272 | 60858 | 155755 | 
| 199 | 16 sympy/geometry/polygon.py | 2023 | 2057| 279 | 61137 | 155755 | 
| 200 | 16 sympy/geometry/parabola.py | 64 | 79| 141 | 61278 | 155755 | 
| 201 | 16 sympy/geometry/entity.py | 593 | 614| 184 | 61462 | 155755 | 
| 202 | 16 sympy/geometry/line.py | 2584 | 2614| 240 | 61702 | 155755 | 
| 203 | 16 sympy/geometry/polygon.py | 2245 | 2269| 148 | 61850 | 155755 | 
| 204 | 16 sympy/geometry/ellipse.py | 1400 | 1446| 418 | 62268 | 155755 | 
| 205 | 16 sympy/geometry/polygon.py | 27 | 112| 711 | 62979 | 155755 | 
| 206 | 16 sympy/geometry/polygon.py | 729 | 783| 492 | 63471 | 155755 | 
| 207 | 16 sympy/geometry/parabola.py | 338 | 384| 317 | 63788 | 155755 | 
| 208 | **16 sympy/geometry/point.py** | 1261 | 1283| 200 | 63988 | 155755 | 
| 209 | 16 sympy/diffgeom/diffgeom.py | 470 | 490| 202 | 64190 | 155755 | 
| 210 | 16 sympy/geometry/polygon.py | 2386 | 2413| 207 | 64397 | 155755 | 
| 211 | 16 sympy/geometry/ellipse.py | 1 | 34| 289 | 64686 | 155755 | 
| 212 | 16 sympy/geometry/line.py | 2315 | 2335| 219 | 64905 | 155755 | 
| 213 | 16 sympy/geometry/polygon.py | 1252 | 1293| 218 | 65123 | 155755 | 
| 214 | 17 sympy/physics/units/dimensions.py | 447 | 461| 142 | 65265 | 160737 | 
| 215 | 17 sympy/geometry/polygon.py | 532 | 618| 644 | 65909 | 160737 | 
| 216 | 17 sympy/geometry/polygon.py | 1537 | 1593| 360 | 66269 | 160737 | 
| 217 | 17 sympy/geometry/polygon.py | 2415 | 2445| 225 | 66494 | 160737 | 
| 218 | **17 sympy/geometry/point.py** | 978 | 998| 175 | 66669 | 160737 | 
| 219 | 18 sympy/vector/coordsysrect.py | 256 | 298| 361 | 67030 | 169008 | 
| 220 | 18 sympy/geometry/parabola.py | 386 | 416| 189 | 67219 | 169008 | 
| 221 | 18 sympy/vector/coordsysrect.py | 376 | 399| 264 | 67483 | 169008 | 
| 222 | 18 sympy/geometry/polygon.py | 2173 | 2208| 325 | 67808 | 169008 | 
| 223 | 18 sympy/plotting/plot.py | 1127 | 1140| 181 | 67989 | 169008 | 
| 224 | 18 sympy/geometry/polygon.py | 2350 | 2384| 285 | 68274 | 169008 | 
| 225 | 18 sympy/geometry/polygon.py | 1057 | 1107| 264 | 68538 | 169008 | 
| 226 | 19 sympy/polys/compatibility.py | 1 | 68| 808 | 69346 | 185044 | 
| 227 | 19 sympy/geometry/polygon.py | 1737 | 1793| 418 | 69764 | 185044 | 
| 228 | 19 sympy/geometry/polygon.py | 158 | 205| 381 | 70145 | 185044 | 
| 229 | 19 sympy/geometry/ellipse.py | 1448 | 1497| 283 | 70428 | 185044 | 
| 230 | 19 sympy/geometry/ellipse.py | 1556 | 1569| 112 | 70540 | 185044 | 
| 231 | 19 sympy/geometry/ellipse.py | 1286 | 1332| 402 | 70942 | 185044 | 
| 232 | 19 sympy/geometry/polygon.py | 273 | 304| 205 | 71147 | 185044 | 
| 233 | 19 sympy/geometry/ellipse.py | 37 | 121| 601 | 71748 | 185044 | 
| 234 | 19 sympy/geometry/parabola.py | 288 | 336| 520 | 72268 | 185044 | 
| 235 | 19 sympy/geometry/ellipse.py | 612 | 665| 601 | 72869 | 185044 | 
| 236 | 20 sympy/physics/vector/fieldfunctions.py | 250 | 314| 554 | 73423 | 187366 | 
| 237 | 20 sympy/geometry/ellipse.py | 1531 | 1554| 118 | 73541 | 187366 | 


### Hint

```
It would be natural that `distance` be symmetric. That is not even hard to implement.
I think it is right for `distance` to be symmetric . I would like to give this a try  . 
```

## Patch

```diff
diff --git a/sympy/geometry/point.py b/sympy/geometry/point.py
--- a/sympy/geometry/point.py
+++ b/sympy/geometry/point.py
@@ -380,19 +380,20 @@ def are_coplanar(cls, *points):
         points = list(uniq(points))
         return Point.affine_rank(*points) <= 2
 
-    def distance(self, p):
-        """The Euclidean distance from self to point p.
-
-        Parameters
-        ==========
-
-        p : Point
+    def distance(self, other):
+        """The Euclidean distance between self and another GeometricEntity.
 
         Returns
         =======
 
         distance : number or symbolic expression.
 
+        Raises
+        ======
+        AttributeError : if other is a GeometricEntity for which
+                         distance is not defined.
+        TypeError : if other is not recognized as a GeometricEntity.
+
         See Also
         ========
 
@@ -402,19 +403,34 @@ def distance(self, p):
         Examples
         ========
 
-        >>> from sympy.geometry import Point
+        >>> from sympy.geometry import Point, Line
         >>> p1, p2 = Point(1, 1), Point(4, 5)
+        >>> l = Line((3, 1), (2, 2))
         >>> p1.distance(p2)
         5
+        >>> p1.distance(l)
+        sqrt(2)
+
+        The computed distance may be symbolic, too:
 
         >>> from sympy.abc import x, y
         >>> p3 = Point(x, y)
-        >>> p3.distance(Point(0, 0))
+        >>> p3.distance((0, 0))
         sqrt(x**2 + y**2)
 
         """
-        s, p = Point._normalize_dimension(self, Point(p))
-        return sqrt(Add(*((a - b)**2 for a, b in zip(s, p))))
+        if not isinstance(other , GeometryEntity) :
+            try :
+                other = Point(other, dim=self.ambient_dimension)
+            except TypeError :
+                raise TypeError("not recognized as a GeometricEntity: %s" % type(other))
+        if isinstance(other , Point) :
+            s, p = Point._normalize_dimension(self, Point(other))
+            return sqrt(Add(*((a - b)**2 for a, b in zip(s, p))))
+        try :
+            return other.distance(self)
+        except AttributeError :
+            raise AttributeError("distance between Point and %s is not defined" % type(other))
 
     def dot(self, p):
         """Return dot product of self with another Point."""

```

## Test Patch

```diff
diff --git a/sympy/geometry/tests/test_point.py b/sympy/geometry/tests/test_point.py
--- a/sympy/geometry/tests/test_point.py
+++ b/sympy/geometry/tests/test_point.py
@@ -33,6 +33,7 @@ def test_point():
     p3 = Point(0, 0)
     p4 = Point(1, 1)
     p5 = Point(0, 1)
+    line = Line(Point(1,0), slope = 1)
 
     assert p1 in p1
     assert p1 not in p2
@@ -55,6 +56,10 @@ def test_point():
     assert Point.distance(p1, p1) == 0
     assert Point.distance(p3, p2) == sqrt(p2.x**2 + p2.y**2)
 
+    # distance should be symmetric
+    assert p1.distance(line) == line.distance(p1)
+    assert p4.distance(line) == line.distance(p4)
+
     assert Point.taxicab_distance(p4, p3) == 2
 
     assert Point.canberra_distance(p4, p5) == 1
@@ -72,7 +77,7 @@ def test_point():
     assert Point.is_collinear(p3, p4, p1_1, p1_2)
     assert Point.is_collinear(p3, p4, p1_1, p1_3) is False
     assert Point.is_collinear(p3, p3, p4, p5) is False
-    line = Line(Point(1,0), slope = 1)
+
     raises(TypeError, lambda: Point.is_collinear(line))
     raises(TypeError, lambda: p1_1.is_collinear(line))
 

```


## Code snippets

### 1 - sympy/geometry/line.py:

Start line: 1182, End line: 1213

```python
class Line(LinearEntity):

    def distance(self, other):
        """
        Finds the shortest distance between a line and a point.

        Raises
        ======

        NotImplementedError is raised if `other` is not a Point

        Examples
        ========

        >>> from sympy import Point, Line
        >>> p1, p2 = Point(0, 0), Point(1, 1)
        >>> s = Line(p1, p2)
        >>> s.distance(Point(-1, 1))
        sqrt(2)
        >>> s.distance((-1, 2))
        3*sqrt(2)/2
        >>> p1, p2 = Point(0, 0, 0), Point(1, 1, 1)
        >>> s = Line(p1, p2)
        >>> s.distance(Point(-1, 1, 1))
        2*sqrt(6)/3
        >>> s.distance((-1, 1, 1))
        2*sqrt(6)/3

        """
        if not isinstance(other, GeometryEntity):
            other = Point(other, dim=self.ambient_dimension)
        if self.contains(other):
            return S.Zero
        return self.perpendicular_segment(other).length
```
### 2 - sympy/geometry/point.py:

Start line: 183, End line: 225

```python
class Point(GeometryEntity):

    def __abs__(self):
        """Returns the distance between this point and the origin."""
        origin = Point([0]*len(self))
        return Point.distance(origin, self)

    def __add__(self, other):
        """Add other to self by incrementing self's coordinates by
        those of other.

        Notes
        =====

        >>> from sympy.geometry.point import Point

        When sequences of coordinates are passed to Point methods, they
        are converted to a Point internally. This __add__ method does
        not do that so if floating point values are used, a floating
        point result (in terms of SymPy Floats) will be returned.

        >>> Point(1, 2) + (.1, .2)
        Point2D(1.1, 2.2)

        If this is not desired, the `translate` method can be used or
        another Point can be added:

        >>> Point(1, 2).translate(.1, .2)
        Point2D(11/10, 11/5)
        >>> Point(1, 2) + Point(.1, .2)
        Point2D(11/10, 11/5)

        See Also
        ========

        sympy.geometry.point.Point.translate

        """
        try:
            s, o = Point._normalize_dimension(self, Point(other, evaluate=False))
        except TypeError:
            raise GeometryError("Don't know how to add {} and a Point object".format(other))

        coords = [simplify(a + b) for a, b in zip(s, o)]
        return Point(coords, evaluate=False)
```
### 3 - sympy/geometry/point.py:

Start line: 383, End line: 417

```python
class Point(GeometryEntity):

    def distance(self, p):
        """The Euclidean distance from self to point p.

        Parameters
        ==========

        p : Point

        Returns
        =======

        distance : number or symbolic expression.

        See Also
        ========

        sympy.geometry.line.Segment.length
        sympy.geometry.point.Point.taxicab_distance

        Examples
        ========

        >>> from sympy.geometry import Point
        >>> p1, p2 = Point(1, 1), Point(4, 5)
        >>> p1.distance(p2)
        5

        >>> from sympy.abc import x, y
        >>> p3 = Point(x, y)
        >>> p3.distance(Point(0, 0))
        sqrt(x**2 + y**2)

        """
        s, p = Point._normalize_dimension(self, Point(p))
        return sqrt(Add(*((a - b)**2 for a, b in zip(s, p))))
```
### 4 - sympy/geometry/line.py:

Start line: 1605, End line: 1651

```python
class Segment(LinearEntity):

    def equals(self, other):
        """Returns True if self and other are the same mathematical entities"""
        return isinstance(other, self.func) and list(
            ordered(self.args)) == list(ordered(other.args))

    def distance(self, other):
        """
        Finds the shortest distance between a line segment and a point.

        Raises
        ======

        NotImplementedError is raised if `other` is not a Point

        Examples
        ========

        >>> from sympy import Point, Segment
        >>> p1, p2 = Point(0, 1), Point(3, 4)
        >>> s = Segment(p1, p2)
        >>> s.distance(Point(10, 15))
        sqrt(170)
        >>> s.distance((0, 12))
        sqrt(73)
        >>> from sympy import Point3D, Segment3D
        >>> p1, p2 = Point3D(0, 0, 3), Point3D(1, 1, 4)
        >>> s = Segment3D(p1, p2)
        >>> s.distance(Point3D(10, 15, 12))
        sqrt(341)
        >>> s.distance((10, 15, 12))
        sqrt(341)
        """
        if not isinstance(other, GeometryEntity):
            other = Point(other, dim=self.ambient_dimension)
        if isinstance(other, Point):
            vp1 = other - self.p1
            vp2 = other - self.p2

            dot_prod_sign_1 = self.direction.dot(vp1) >= 0
            dot_prod_sign_2 = self.direction.dot(vp2) <= 0
            if dot_prod_sign_1 and dot_prod_sign_2:
                return Line(self.p1, self.p2).distance(other)
            if dot_prod_sign_1 and not dot_prod_sign_2:
                return abs(vp2)
            if not dot_prod_sign_1 and dot_prod_sign_2:
                return abs(vp1)
        raise NotImplementedError()
```
### 5 - sympy/geometry/line.py:

Start line: 1931, End line: 1960

```python
class Line2D(LinearEntity2D, Line):
    def __new__(cls, p1, pt=None, slope=None, **kwargs):
        if isinstance(p1, LinearEntity):
            if pt is not None:
                raise ValueError('When p1 is a LinearEntity, pt should be None')
            p1, pt = Point._normalize_dimension(*p1.args, dim=2)
        else:
            p1 = Point(p1, dim=2)
        if pt is not None and slope is None:
            try:
                p2 = Point(pt, dim=2)
            except (NotImplementedError, TypeError, ValueError):
                raise ValueError(filldedent('''
                    The 2nd argument was not a valid Point.
                    If it was a slope, enter it with keyword "slope".
                    '''))
        elif slope is not None and pt is None:
            slope = sympify(slope)
            if slope.is_finite is False:
                # when infinite slope, don't change x
                dx = 0
                dy = 1
            else:
                # go over 1 up slope
                dx = 1
                dy = slope
            # XXX avoiding simplification by adding to coords directly
            p2 = Point(p1.x + dx, p1.y + dy, evaluate=False)
        else:
            raise ValueError('A 2nd Point or keyword "slope" must be used.')
        return LinearEntity2D.__new__(cls, p1, p2, **kwargs)
```
### 6 - sympy/geometry/line.py:

Start line: 1215, End line: 1253

```python
class Line(LinearEntity):

    @deprecated(useinstead="equals", issue=12860, deprecated_since_version="1.0")
    def equal(self, other):
        return self.equals(other)

    def equals(self, other):
        """Returns True if self and other are the same mathematical entities"""
        if not isinstance(other, Line):
            return False
        return Point.is_collinear(self.p1, other.p1, self.p2, other.p2)

    def plot_interval(self, parameter='t'):
        """The plot interval for the default geometric plot of line. Gives
        values that will produce a line that is +/- 5 units long (where a
        unit is the distance between the two points that define the line).

        Parameters
        ==========

        parameter : str, optional
            Default value is 't'.

        Returns
        =======

        plot_interval : list (plot interval)
            [parameter, lower_bound, upper_bound]

        Examples
        ========

        >>> from sympy import Point, Line
        >>> p1, p2 = Point(0, 0), Point(5, 3)
        >>> l1 = Line(p1, p2)
        >>> l1.plot_interval()
        [t, -5, 5]

        """
        t = _symbol(parameter, real=True)
        return [t, -5, 5]
```
### 7 - sympy/geometry/line.py:

Start line: 1394, End line: 1432

```python
class Ray(LinearEntity):

    def distance(self, other):
        """
        Finds the shortest distance between the ray and a point.

        Raises
        ======

        NotImplementedError is raised if `other` is not a Point

        Examples
        ========

        >>> from sympy import Point, Ray
        >>> p1, p2 = Point(0, 0), Point(1, 1)
        >>> s = Ray(p1, p2)
        >>> s.distance(Point(-1, -1))
        sqrt(2)
        >>> s.distance((-1, 2))
        3*sqrt(2)/2
        >>> p1, p2 = Point(0, 0, 0), Point(1, 1, 2)
        >>> s = Ray(p1, p2)
        >>> s
        Ray3D(Point3D(0, 0, 0), Point3D(1, 1, 2))
        >>> s.distance(Point(-1, -1, 2))
        4*sqrt(3)/3
        >>> s.distance((-1, -1, 2))
        4*sqrt(3)/3

        """
        if not isinstance(other, GeometryEntity):
            other = Point(other, dim=self.ambient_dimension)
        if self.contains(other):
            return S.Zero

        proj = Line(self.p1, self.p2).projection(other)
        if self.contains(proj):
            return abs(other - proj)
        else:
            return abs(other - self.source)
```
### 8 - sympy/geometry/point.py:

Start line: 753, End line: 784

```python
class Point(GeometryEntity):

    def taxicab_distance(self, p):
        """The Taxicab Distance from self to point p.

        Returns the sum of the horizontal and vertical distances to point p.

        Parameters
        ==========

        p : Point

        Returns
        =======

        taxicab_distance : The sum of the horizontal
        and vertical distances to point p.

        See Also
        ========

        sympy.geometry.point.Point.distance

        Examples
        ========

        >>> from sympy.geometry import Point
        >>> p1, p2 = Point(1, 1), Point(4, 5)
        >>> p1.taxicab_distance(p2)
        7

        """
        s, p = Point._normalize_dimension(self, Point(p))
        return Add(*(abs(a - b) for a, b in zip(s, p)))
```
### 9 - sympy/geometry/line.py:

Start line: 2446, End line: 2462

```python
class Line3D(LinearEntity3D, Line):

    def __new__(cls, p1, pt=None, direction_ratio=[], **kwargs):
        if isinstance(p1, LinearEntity3D):
            if pt is not None:
                raise ValueError('if p1 is a LinearEntity, pt must be None.')
            p1, pt = p1.args
        else:
            p1 = Point(p1, dim=3)
        if pt is not None and len(direction_ratio) == 0:
            pt = Point(pt, dim=3)
        elif len(direction_ratio) == 3 and pt is None:
            pt = Point3D(p1.x + direction_ratio[0], p1.y + direction_ratio[1],
                         p1.z + direction_ratio[2])
        else:
            raise ValueError('A 2nd Point or keyword "direction_ratio" must '
                             'be used.')

        return LinearEntity3D.__new__(cls, p1, pt, **kwargs)
```
### 10 - sympy/geometry/line.py:

Start line: 644, End line: 659

```python
class LinearEntity(GeometrySet):

    def is_similar(self, other):
        """
        Return True if self and other are contained in the same line.

        Examples
        ========

        >>> from sympy import Point, Line
        >>> p1, p2, p3 = Point(0, 1), Point(3, 4), Point(2, 3)
        >>> l1 = Line(p1, p2)
        >>> l2 = Line(p1, p3)
        >>> l1.is_similar(l2)
        True
        """
        l = Line(self.p1, self.p2)
        return l.contains(other)
```
### 11 - sympy/geometry/point.py:

Start line: 1, End line: 41

```python
"""Geometrical Points.

Contains
========
Point
Point2D
Point3D

When methods of Point require 1 or more points as arguments, they
can be passed as a sequence of coordinates or Points:

>>> from sympy.geometry.point import Point
>>> Point(1, 1).is_collinear((2, 2), (3, 4))
False
>>> Point(1, 1).is_collinear(Point(2, 2), Point(3, 4))
False

"""

from __future__ import division, print_function

import warnings

from sympy.core import S, sympify, Expr
from sympy.core.numbers import Number
from sympy.core.compatibility import iterable, is_sequence, as_int
from sympy.core.containers import Tuple
from sympy.simplify import nsimplify, simplify
from sympy.geometry.exceptions import GeometryError
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.complexes import im
from sympy.matrices import Matrix
from sympy.core.relational import Eq
from sympy.core.numbers import Float
from sympy.core.evaluate import global_evaluate
from sympy.core.add import Add
from sympy.sets import FiniteSet
from sympy.utilities.iterables import uniq
from sympy.utilities.misc import filldedent, func_name, Undecidable

from .entity import GeometryEntity
```
### 12 - sympy/geometry/point.py:

Start line: 786, End line: 845

```python
class Point(GeometryEntity):

    def canberra_distance(self, p):
        """The Canberra Distance from self to point p.

        Returns the weighted sum of horizontal and vertical distances to
        point p.

        Parameters
        ==========

        p : Point

        Returns
        =======

        canberra_distance : The weighted sum of horizontal and vertical
        distances to point p. The weight used is the sum of absolute values
        of the coordinates.

        See Also
        ========

        sympy.geometry.point.Point.distance

        Examples
        ========

        >>> from sympy.geometry import Point
        >>> p1, p2 = Point(1, 1), Point(3, 3)
        >>> p1.canberra_distance(p2)
        1
        >>> p1, p2 = Point(0, 0), Point(3, 3)
        >>> p1.canberra_distance(p2)
        2

        Raises
        ======

        ValueError when both vectors are zero.

        See Also
        ========

        sympy.geometry.point.Point.distance

        """

        s, p = Point._normalize_dimension(self, Point(p))
        if self.is_zero and p.is_zero:
            raise ValueError("Cannot project to the zero vector.")
        return Add(*((abs(a - b)/(abs(a) + abs(b))) for a, b in zip(s, p)))

    @property
    def unit(self):
        """Return the Point that is in the same direction as `self`
        and a distance of 1 from the origin"""
        return self / abs(self)

    n = evalf

    __truediv__ = __div__
```
### 15 - sympy/geometry/point.py:

Start line: 419, End line: 430

```python
class Point(GeometryEntity):

    def dot(self, p):
        """Return dot product of self with another Point."""
        if not is_sequence(p):
            p = Point(p)  # raise the error via Point
        return Add(*(a*b for a, b in zip(self, p)))

    def equals(self, other):
        """Returns whether the coordinates of self and other agree."""
        # a point is equal to another point if all its components are equal
        if not isinstance(other, Point) or len(self) != len(other):
            return False
        return all(a.equals(b) for a,b in zip(self, other))
```
### 16 - sympy/geometry/point.py:

Start line: 688, End line: 712

```python
class Point(GeometryEntity):

    @property
    def orthogonal_direction(self):
        """Returns a non-zero point that is orthogonal to the
        line containing `self` and the origin.

        Examples
        ========

        >>> from sympy.geometry import Line, Point
        >>> a = Point(1, 2, 3)
        >>> a.orthogonal_direction
        Point3D(-2, 1, 0)
        >>> b = _
        >>> Line(b, b.origin).is_perpendicular(Line(a, a.origin))
        True
        """
        dim = self.ambient_dimension
        # if a coordinate is zero, we can put a 1 there and zeros elsewhere
        if self[0] == S.Zero:
            return Point([1] + (dim - 1)*[0])
        if self[1] == S.Zero:
            return Point([0,1] + (dim - 2)*[0])
        # if the first two coordinates aren't zero, we can create a non-zero
        # orthogonal vector by swapping them, negating one, and padding with zeros
        return Point([-self[1], self[0]] + (dim - 2)*[0])
```
### 19 - sympy/geometry/point.py:

Start line: 504, End line: 540

```python
class Point(GeometryEntity):

    def is_collinear(self, *args):
        """Returns `True` if there exists a line
        that contains `self` and `points`.  Returns `False` otherwise.
        A trivially True value is returned if no points are given.

        Parameters
        ==========

        args : sequence of Points

        Returns
        =======

        is_collinear : boolean

        See Also
        ========

        sympy.geometry.line.Line

        Examples
        ========

        >>> from sympy import Point
        >>> from sympy.abc import x
        >>> p1, p2 = Point(0, 0), Point(1, 1)
        >>> p3, p4, p5 = Point(2, 2), Point(x, x), Point(1, 2)
        >>> Point.is_collinear(p1, p2, p3, p4)
        True
        >>> Point.is_collinear(p1, p2, p3, p5)
        False

        """
        points = (self,) + args
        points = Point._normalize_dimension(*[Point(i) for i in points])
        points = list(uniq(points))
        return Point.affine_rank(*points) <= 1
```
### 23 - sympy/geometry/point.py:

Start line: 111, End line: 181

```python
class Point(GeometryEntity):

    def __new__(cls, *args, **kwargs):
        evaluate = kwargs.get('evaluate', global_evaluate[0])
        on_morph = kwargs.get('on_morph', 'ignore')

        # unpack into coords
        coords = args[0] if len(args) == 1 else args

        # check args and handle quickly handle Point instances
        if isinstance(coords, Point):
            # even if we're mutating the dimension of a point, we
            # don't reevaluate its coordinates
            evaluate = False
            if len(coords) == kwargs.get('dim', len(coords)):
                return coords

        if not is_sequence(coords):
            raise TypeError(filldedent('''
                Expecting sequence of coordinates, not `{}`'''
                                       .format(func_name(coords))))
        # A point where only `dim` is specified is initialized
        # to zeros.
        if len(coords) == 0 and kwargs.get('dim', None):
            coords = (S.Zero,)*kwargs.get('dim')

        coords = Tuple(*coords)
        dim = kwargs.get('dim', len(coords))

        if len(coords) < 2:
            raise ValueError(filldedent('''
                Point requires 2 or more coordinates or
                keyword `dim` > 1.'''))
        if len(coords) != dim:
            message = ("Dimension of {} needs to be changed"
                       "from {} to {}.").format(coords, len(coords), dim)
            if on_morph == 'ignore':
                pass
            elif on_morph == "error":
                raise ValueError(message)
            elif on_morph == 'warn':
                warnings.warn(message)
            else:
                raise ValueError(filldedent('''
                        on_morph value should be 'error',
                        'warn' or 'ignore'.'''))
        if any(i for i in coords[dim:]):
            raise ValueError('Nonzero coordinates cannot be removed.')
        if any(a.is_number and im(a) for a in coords):
            raise ValueError('Imaginary coordinates are not permitted.')
        if not all(isinstance(a, Expr) for a in coords):
            raise TypeError('Coordinates must be valid SymPy expressions.')

        # pad with zeros appropriately
        coords = coords[:dim] + (S.Zero,)*(dim - len(coords))

        # Turn any Floats into rationals and simplify
        # any expressions before we instantiate
        if evaluate:
            coords = coords.xreplace(dict(
                [(f, simplify(nsimplify(f, rational=True)))
                 for f in coords.atoms(Float)]))

        # return 2D or 3D instances
        if len(coords) == 2:
            kwargs['_nocheck'] = True
            return Point2D(*coords, **kwargs)
        elif len(coords) == 3:
            kwargs['_nocheck'] = True
            return Point3D(*coords, **kwargs)

        # the general Point
        return GeometryEntity.__new__(cls, *coords)
```
### 26 - sympy/geometry/point.py:

Start line: 626, End line: 650

```python
class Point(GeometryEntity):

    @property
    def is_zero(self):
        """True if every coordinate is zero, False if any coordinate is not zero,
        and None if it cannot be determined."""
        nonzero = [x.is_nonzero for x in self.args]
        if any(nonzero):
            return False
        if any(x is None for x in nonzero):
            return None
        return True

    @property
    def length(self):
        """
        Treating a Point as a Line, this returns 0 for the length of a Point.

        Examples
        ========

        >>> from sympy import Point
        >>> p = Point(0, 1)
        >>> p.length
        0
        """
        return S.Zero
```
### 31 - sympy/geometry/point.py:

Start line: 1172, End line: 1194

```python
class Point3D(Point):

    def direction_ratio(self, point):
        """
        Gives the direction ratio between 2 points

        Parameters
        ==========

        p : Point3D

        Returns
        =======

        list

        Examples
        ========

        >>> from sympy import Point3D
        >>> p1 = Point3D(1, 2, 3)
        >>> p1.direction_ratio(Point3D(2, 3, 5))
        [1, 1, 2]
        """
        return [(point.x - self.x),(point.y - self.y),(point.z - self.z)]
```
### 38 - sympy/geometry/point.py:

Start line: 1145, End line: 1170

```python
class Point3D(Point):

    def direction_cosine(self, point):
        """
        Gives the direction cosine between 2 points

        Parameters
        ==========

        p : Point3D

        Returns
        =======

        list

        Examples
        ========

        >>> from sympy import Point3D
        >>> p1 = Point3D(1, 2, 3)
        >>> p1.direction_cosine(Point3D(2, 3, 5))
        [sqrt(6)/6, sqrt(6)/6, sqrt(6)/3]
        """
        a = self.direction_ratio(point)
        b = sqrt(Add(*(i**2 for i in a)))
        return [(point.x - self.x) / b,(point.y - self.y) / b,
                (point.z - self.z) / b]
```
### 39 - sympy/geometry/point.py:

Start line: 284, End line: 310

```python
class Point(GeometryEntity):

    def __neg__(self):
        """Negate the point."""
        coords = [-x for x in self.args]
        return Point(coords, evaluate=False)

    def __sub__(self, other):
        """Subtract two points, or subtract a factor from this point's
        coordinates."""
        return self + [-x for x in other]

    @classmethod
    def _normalize_dimension(cls, *points, **kwargs):
        """Ensure that points have the same dimension.
        By default `on_morph='warn'` is passed to the
        `Point` constructor."""
        # if we have a built-in ambient dimension, use it
        dim = getattr(cls, '_ambient_dimension', None)
        # override if we specified it
        dim = kwargs.get('dim', dim)
        # if no dim was given, use the highest dimensional point
        if dim is None:
            dim = max(i.ambient_dimension for i in points)
        if all(i.ambient_dimension == dim for i in points):
            return list(points)
        kwargs['dim'] = dim
        kwargs['on_morph'] = kwargs.get('on_morph', 'warn')
        return [Point(i, **kwargs) for i in points]
```
### 41 - sympy/geometry/point.py:

Start line: 847, End line: 915

```python
class Point2D(Point):
    """A point in a 2-dimensional Euclidean space.

    Parameters
    ==========

    coords : sequence of 2 coordinate values.

    Attributes
    ==========

    x
    y
    length

    Raises
    ======

    TypeError
        When trying to add or subtract points with different dimensions.
        When trying to create a point with more than two dimensions.
        When `intersection` is called with object other than a Point.

    See Also
    ========

    sympy.geometry.line.Segment : Connects two Points

    Examples
    ========

    >>> from sympy.geometry import Point2D
    >>> from sympy.abc import x
    >>> Point2D(1, 2)
    Point2D(1, 2)
    >>> Point2D([1, 2])
    Point2D(1, 2)
    >>> Point2D(0, x)
    Point2D(0, x)

    Floats are automatically converted to Rational unless the
    evaluate flag is False:

    >>> Point2D(0.5, 0.25)
    Point2D(1/2, 1/4)
    >>> Point2D(0.5, 0.25, evaluate=False)
    Point2D(0.5, 0.25)

    """

    _ambient_dimension = 2

    def __new__(cls, *args, **kwargs):
        if not kwargs.pop('_nocheck', False):
            kwargs['dim'] = 2
            args = Point(*args, **kwargs)
        return GeometryEntity.__new__(cls, *args)

    def __contains__(self, item):
        return item == self

    @property
    def bounds(self):
        """Return a tuple (xmin, ymin, xmax, ymax) representing the bounding
        rectangle for the geometric figure.

        """

        return (self.x, self.y, self.x, self.y)
```
### 45 - sympy/geometry/point.py:

Start line: 598, End line: 624

```python
class Point(GeometryEntity):

    @property
    def is_nonzero(self):
        """True if any coordinate is nonzero, False if every coordinate is zero,
        and None if it cannot be determined."""
        is_zero = self.is_zero
        if is_zero is None:
            return None
        return not is_zero

    def is_scalar_multiple(self, p):
        """Returns whether each coordinate of `self` is a scalar
        multiple of the corresponding coordinate in point p.
        """
        s, o = Point._normalize_dimension(self, Point(p))
        # 2d points happen a lot, so optimize this function call
        if s.ambient_dimension == 2:
            (x1, y1), (x2, y2) = s.args, o.args
            rv = (x1*y2 - x2*y1).equals(0)
            if rv is None:
                raise Undecidable(filldedent(
                    '''can't determine if %s is a scalar multiple of
                    %s''' % (s, o)))

        # if the vectors p1 and p2 are linearly dependent, then they must
        # be scalar multiples of each other
        m = Matrix([s.args, o.args])
        return m.rank() < 2
```
### 48 - sympy/geometry/point.py:

Start line: 44, End line: 109

```python
class Point(GeometryEntity):
    """A point in a n-dimensional Euclidean space.

    Parameters
    ==========

    coords : sequence of n-coordinate values. In the special
        case where n=2 or 3, a Point2D or Point3D will be created
        as appropriate.
    evaluate : if `True` (default), all floats are turn into
        exact types.
    dim : number of coordinates the point should have.  If coordinates
        are unspecified, they are padded with zeros.
    on_morph : indicates what should happen when the number of
        coordinates of a point need to be changed by adding or
        removing zeros.  Possible values are `'warn'`, `'error'`, or
        `ignore` (default).  No warning or error is given when `*args`
        is empty and `dim` is given. An error is always raised when
        trying to remove nonzero coordinates.


    Attributes
    ==========

    length
    origin: A `Point` representing the origin of the
        appropriately-dimensioned space.

    Raises
    ======

    TypeError : When instantiating with anything but a Point or sequence
    ValueError : when instantiating with a sequence with length < 2 or
        when trying to reduce dimensions if keyword `on_morph='error'` is
        set.

    See Also
    ========

    sympy.geometry.line.Segment : Connects two Points

    Examples
    ========

    >>> from sympy.geometry import Point
    >>> from sympy.abc import x
    >>> Point(1, 2, 3)
    Point3D(1, 2, 3)
    >>> Point([1, 2])
    Point2D(1, 2)
    >>> Point(0, x)
    Point2D(0, x)
    >>> Point(dim=4)
    Point(0, 0, 0, 0)

    Floats are automatically converted to Rational unless the
    evaluate flag is False:

    >>> Point(0.5, 0.25)
    Point2D(1/2, 1/4)
    >>> Point(0.5, 0.25, evaluate=False)
    Point2D(0.5, 0.25)

    """

    is_Point = True
```
### 50 - sympy/geometry/point.py:

Start line: 1023, End line: 1051

```python
class Point2D(Point):

    @property
    def x(self):
        """
        Returns the X coordinate of the Point.

        Examples
        ========

        >>> from sympy import Point2D
        >>> p = Point2D(0, 1)
        >>> p.x
        0
        """
        return self.args[0]

    @property
    def y(self):
        """
        Returns the Y coordinate of the Point.

        Examples
        ========

        >>> from sympy import Point2D
        >>> p = Point2D(0, 1)
        >>> p.y
        1
        """
        return self.args[1]
```
### 53 - sympy/geometry/point.py:

Start line: 714, End line: 751

```python
class Point(GeometryEntity):

    @staticmethod
    def project(a, b):
        """Project the point `a` onto the line between the origin
        and point `b` along the normal direction.

        Parameters
        ==========

        a : Point
        b : Point

        Returns
        =======

        p : Point

        See Also
        ========

        sympy.geometry.line.LinearEntity.projection

        Examples
        ========

        >>> from sympy.geometry import Line, Point
        >>> a = Point(1, 2)
        >>> b = Point(2, 5)
        >>> z = a.origin
        >>> p = Point.project(a, b)
        >>> Line(p, a).is_perpendicular(Line(p, b))
        True
        >>> Point.is_collinear(z, p, b)
        True
        """
        a, b = Point._normalize_dimension(Point(a), Point(b))
        if b.is_zero:
            raise ValueError("Cannot project to the zero vector.")
        return b*(a.dot(b) / b.dot(b))
```
### 55 - sympy/geometry/point.py:

Start line: 227, End line: 251

```python
class Point(GeometryEntity):

    def __contains__(self, item):
        return item in self.args

    def __div__(self, divisor):
        """Divide point's coordinates by a factor."""
        divisor = sympify(divisor)
        coords = [simplify(x/divisor) for x in self.args]
        return Point(coords, evaluate=False)

    def __eq__(self, other):
        if not isinstance(other, Point) or len(self.args) != len(other.args):
            return False
        return self.args == other.args

    def __getitem__(self, key):
        return self.args[key]

    def __hash__(self):
        return hash(self.args)

    def __iter__(self):
        return self.args.__iter__()

    def __len__(self):
        return len(self.args)
```
### 58 - sympy/geometry/point.py:

Start line: 652, End line: 686

```python
class Point(GeometryEntity):

    def midpoint(self, p):
        """The midpoint between self and point p.

        Parameters
        ==========

        p : Point

        Returns
        =======

        midpoint : Point

        See Also
        ========

        sympy.geometry.line.Segment.midpoint

        Examples
        ========

        >>> from sympy.geometry import Point
        >>> p1, p2 = Point(1, 1), Point(13, 5)
        >>> p1.midpoint(p2)
        Point2D(7, 3)

        """
        s, p = Point._normalize_dimension(self, Point(p))
        return Point([simplify((a + b)*S.Half) for a, b in zip(s, p)])

    @property
    def origin(self):
        """A point of all zeros of the same ambient dimension
        as the current point"""
        return Point([0]*len(self), evaluate=False)
```
### 64 - sympy/geometry/point.py:

Start line: 1308, End line: 1352

```python
class Point3D(Point):

    @property
    def x(self):
        """
        Returns the X coordinate of the Point.

        Examples
        ========

        >>> from sympy import Point3D
        >>> p = Point3D(0, 1, 3)
        >>> p.x
        0
        """
        return self.args[0]

    @property
    def y(self):
        """
        Returns the Y coordinate of the Point.

        Examples
        ========

        >>> from sympy import Point3D
        >>> p = Point3D(0, 1, 2)
        >>> p.y
        1
        """
        return self.args[1]

    @property
    def z(self):
        """
        Returns the Z coordinate of the Point.

        Examples
        ========

        >>> from sympy import Point3D
        >>> p = Point3D(0, 1, 1)
        >>> p.z
        1
        """
        return self.args[2]
```
### 66 - sympy/geometry/point.py:

Start line: 1196, End line: 1232

```python
class Point3D(Point):

    def intersection(self, other):
        """The intersection between this point and another point.

        Parameters
        ==========

        other : Point

        Returns
        =======

        intersection : list of Points

        Notes
        =====

        The return value will either be an empty list if there is no
        intersection, otherwise it will contain this point.

        Examples
        ========

        >>> from sympy import Point3D
        >>> p1, p2, p3 = Point3D(0, 0, 0), Point3D(1, 1, 1), Point3D(0, 0, 0)
        >>> p1.intersection(p2)
        []
        >>> p1.intersection(p3)
        [Point3D(0, 0, 0)]

        """
        if not isinstance(other, GeometryEntity):
            other = Point(other, dim=3)
        if isinstance(other, Point3D):
            if self == other:
                return [self]
            return []
        return other.intersection(self)
```
### 71 - sympy/geometry/point.py:

Start line: 1109, End line: 1143

```python
class Point3D(Point):

    @staticmethod
    def are_collinear(*points):
        """Is a sequence of points collinear?

        Test whether or not a set of points are collinear. Returns True if
        the set of points are collinear, or False otherwise.

        Parameters
        ==========

        points : sequence of Point

        Returns
        =======

        are_collinear : boolean

        See Also
        ========

        sympy.geometry.line.Line3D

        Examples
        ========

        >>> from sympy import Point3D, Matrix
        >>> from sympy.abc import x
        >>> p1, p2 = Point3D(0, 0, 0), Point3D(1, 1, 1)
        >>> p3, p4, p5 = Point3D(2, 2, 2), Point3D(x, x, x), Point3D(1, 2, 6)
        >>> Point3D.are_collinear(p1, p2, p3, p4)
        True
        >>> Point3D.are_collinear(p1, p2, p3, p5)
        False
        """
        return Point.is_collinear(*points)
```
### 73 - sympy/geometry/point.py:

Start line: 337, End line: 381

```python
class Point(GeometryEntity):

    @classmethod
    def are_coplanar(cls, *points):
        """Return True if there exists a plane in which all the points
        lie.  A trivial True value is returned if `len(points) < 3` or
        all Points are 2-dimensional.

        Parameters
        ==========

        A set of points

        Raises
        ======

        ValueError : if less than 3 unique points are given

        Returns
        =======

        boolean

        Examples
        ========

        >>> from sympy import Point3D
        >>> p1 = Point3D(1, 2, 2)
        >>> p2 = Point3D(2, 7, 2)
        >>> p3 = Point3D(0, 0, 2)
        >>> p4 = Point3D(1, 1, 2)
        >>> Point3D.are_coplanar(p1, p2, p3, p4)
        True
        >>> p5 = Point3D(0, 1, 3)
        >>> Point3D.are_coplanar(p1, p2, p3, p5)
        False

        """
        if len(points) <= 1:
            return True

        points = cls._normalize_dimension(*[Point(i) for i in points])
        # quick exit if we are in 2D
        if points[0].ambient_dimension == 2:
            return True
        points = list(uniq(points))
        return Point.affine_rank(*points) <= 2
```
### 79 - sympy/geometry/point.py:

Start line: 463, End line: 502

```python
class Point(GeometryEntity):

    def intersection(self, other):
        """The intersection between this point and another GeometryEntity.

        Parameters
        ==========

        other : Point

        Returns
        =======

        intersection : list of Points

        Notes
        =====

        The return value will either be an empty list if there is no
        intersection, otherwise it will contain this point.

        Examples
        ========

        >>> from sympy import Point
        >>> p1, p2, p3 = Point(0, 0), Point(1, 1), Point(0, 0)
        >>> p1.intersection(p2)
        []
        >>> p1.intersection(p3)
        [Point2D(0, 0)]

        """
        if not isinstance(other, GeometryEntity):
            other = Point(other)
        if isinstance(other, Point):
            if self == other:
                return [self]
            p1, p2 = Point._normalize_dimension(self, other)
            if p1 == self and p1 == p2:
                return [self]
            return []
        return other.intersection(self)
```
### 80 - sympy/geometry/point.py:

Start line: 1053, End line: 1107

```python
class Point3D(Point):
    """A point in a 3-dimensional Euclidean space.

    Parameters
    ==========

    coords : sequence of 3 coordinate values.

    Attributes
    ==========

    x
    y
    z
    length

    Raises
    ======

    TypeError
        When trying to add or subtract points with different dimensions.
        When `intersection` is called with object other than a Point.

    Examples
    ========

    >>> from sympy import Point3D
    >>> from sympy.abc import x
    >>> Point3D(1, 2, 3)
    Point3D(1, 2, 3)
    >>> Point3D([1, 2, 3])
    Point3D(1, 2, 3)
    >>> Point3D(0, x, 3)
    Point3D(0, x, 3)

    Floats are automatically converted to Rational unless the
    evaluate flag is False:

    >>> Point3D(0.5, 0.25, 2)
    Point3D(1/2, 1/4, 2)
    >>> Point3D(0.5, 0.25, 3, evaluate=False)
    Point3D(0.5, 0.25, 3)

    """

    _ambient_dimension = 3

    def __new__(cls, *args, **kwargs):
        if not kwargs.pop('_nocheck', False):
            kwargs['dim'] = 3
            args = Point(*args, **kwargs)
        return GeometryEntity.__new__(cls, *args)

    def __contains__(self, item):
        return item == self
```
### 88 - sympy/geometry/point.py:

Start line: 951, End line: 976

```python
class Point2D(Point):

    def scale(self, x=1, y=1, pt=None):
        """Scale the coordinates of the Point by multiplying by
        ``x`` and ``y`` after subtracting ``pt`` -- default is (0, 0) --
        and then adding ``pt`` back again (i.e. ``pt`` is the point of
        reference for the scaling).

        See Also
        ========

        rotate, translate

        Examples
        ========

        >>> from sympy import Point2D
        >>> t = Point2D(1, 1)
        >>> t.scale(2)
        Point2D(2, 1)
        >>> t.scale(2, 2)
        Point2D(2, 2)

        """
        if pt:
            pt = Point(pt, dim=2)
            return self.translate(*(-pt).args).scale(x, y).translate(*pt.args)
        return Point(self.x*x, self.y*y)
```
### 91 - sympy/geometry/point.py:

Start line: 542, End line: 596

```python
class Point(GeometryEntity):

    def is_concyclic(self, *args):
        """Do `self` and the given sequence of points lie in a circle?

        Returns True if the set of points are concyclic and
        False otherwise. A trivial value of True is returned
        if there are fewer than 2 other points.

        Parameters
        ==========

        args : sequence of Points

        Returns
        =======

        is_concyclic : boolean


        Examples
        ========

        >>> from sympy import Point

        Define 4 points that are on the unit circle:

        >>> p1, p2, p3, p4 = Point(1, 0), (0, 1), (-1, 0), (0, -1)

        >>> p1.is_concyclic() == p1.is_concyclic(p2, p3, p4) == True
        True

        Define a point not on that circle:

        >>> p = Point(1, 1)

        >>> p.is_concyclic(p1, p2, p3)
        False

        """
        points = (self,) + args
        points = Point._normalize_dimension(*[Point(i) for i in points])
        points = list(uniq(points))
        if not Point.affine_rank(*points) <= 2:
            return False
        origin = points[0]
        points = [p - origin for p in points]
        # points are concyclic if they are coplanar and
        # there is a point c so that ||p_i-c|| == ||p_j-c|| for all
        # i and j.  Rearranging this equation gives us the following
        # condition: the matrix `mat` must not a pivot in the last
        # column.
        mat = Matrix([list(i) + [i.dot(i)] for i in points])
        rref, pivots = mat.rref()
        if len(origin) not in pivots:
            return True
        return False
```
### 101 - sympy/geometry/point.py:

Start line: 432, End line: 461

```python
class Point(GeometryEntity):

    def evalf(self, prec=None, **options):
        """Evaluate the coordinates of the point.

        This method will, where possible, create and return a new Point
        where the coordinates are evaluated as floating point numbers to
        the precision indicated (default=15).

        Parameters
        ==========

        prec : int

        Returns
        =======

        point : Point

        Examples
        ========

        >>> from sympy import Point, Rational
        >>> p1 = Point(Rational(1, 2), Rational(3, 2))
        >>> p1
        Point2D(1/2, 3/2)
        >>> p1.evalf()
        Point2D(0.5, 1.5)

        """
        coords = [x.evalf(prec, **options) for x in self.args]
        return Point(*coords, evaluate=False)
```
### 115 - sympy/geometry/point.py:

Start line: 1234, End line: 1259

```python
class Point3D(Point):

    def scale(self, x=1, y=1, z=1, pt=None):
        """Scale the coordinates of the Point by multiplying by
        ``x`` and ``y`` after subtracting ``pt`` -- default is (0, 0) --
        and then adding ``pt`` back again (i.e. ``pt`` is the point of
        reference for the scaling).

        See Also
        ========

        translate

        Examples
        ========

        >>> from sympy import Point3D
        >>> t = Point3D(1, 1, 1)
        >>> t.scale(2)
        Point3D(2, 1, 1)
        >>> t.scale(2, 2)
        Point3D(2, 2, 1)

        """
        if pt:
            pt = Point3D(pt)
            return self.translate(*(-pt).args).scale(x, y, z).translate(*pt.args)
        return Point3D(self.x*x, self.y*y, self.z*z)
```
### 125 - sympy/geometry/point.py:

Start line: 1000, End line: 1021

```python
class Point2D(Point):

    def translate(self, x=0, y=0):
        """Shift the Point by adding x and y to the coordinates of the Point.

        See Also
        ========

        rotate, scale

        Examples
        ========

        >>> from sympy import Point2D
        >>> t = Point2D(0, 1)
        >>> t.translate(2)
        Point2D(2, 1)
        >>> t.translate(2, 2)
        Point2D(2, 3)
        >>> t + Point2D(2, 2)
        Point2D(2, 3)

        """
        return Point(self.x + x, self.y + y)
```
### 135 - sympy/geometry/point.py:

Start line: 253, End line: 282

```python
class Point(GeometryEntity):

    def __mul__(self, factor):
        """Multiply point's coordinates by a factor.

        Notes
        =====

        >>> from sympy.geometry.point import Point

        When multiplying a Point by a floating point number,
        the coordinates of the Point will be changed to Floats:

        >>> Point(1, 2)*0.1
        Point2D(0.1, 0.2)

        If this is not desired, the `scale` method can be used or
        else only multiply or divide by integers:

        >>> Point(1, 2).scale(1.1, 1.1)
        Point2D(11/10, 11/5)
        >>> Point(1, 2)*11/10
        Point2D(11/10, 11/5)

        See Also
        ========

        sympy.geometry.point.Point.scale
        """
        factor = sympify(factor)
        coords = [simplify(x*factor) for x in self.args]
        return Point(coords, evaluate=False)
```
### 136 - sympy/geometry/point.py:

Start line: 312, End line: 335

```python
class Point(GeometryEntity):

    @staticmethod
    def affine_rank(*args):
        """The affine rank of a set of points is the dimension
        of the smallest affine space containing all the points.
        For example, if the points lie on a line (and are not all
        the same) their affine rank is 1.  If the points lie on a plane
        but not a line, their affine rank is 2.  By convention, the empty
        set has affine rank -1."""

        if len(args) == 0:
            return -1
        # make sure we're genuinely points
        # and translate every point to the origin
        points = Point._normalize_dimension(*[Point(i) for i in args])
        origin = points[0]
        points = [i - origin for i in points[1:]]

        m = Matrix([i.args for i in points])
        return m.rank()

    @property
    def ambient_dimension(self):
        """Number of components this point has."""
        return getattr(self, '_ambient_dimension', len(self))
```
### 168 - sympy/geometry/point.py:

Start line: 1285, End line: 1306

```python
class Point3D(Point):

    def translate(self, x=0, y=0, z=0):
        """Shift the Point by adding x and y to the coordinates of the Point.

        See Also
        ========

        rotate, scale

        Examples
        ========

        >>> from sympy import Point3D
        >>> t = Point3D(0, 1, 1)
        >>> t.translate(2)
        Point3D(2, 1, 1)
        >>> t.translate(2, 2)
        Point3D(2, 3, 1)
        >>> t + Point3D(2, 2, 2)
        Point3D(2, 3, 3)

        """
        return Point3D(self.x + x, self.y + y, self.z + z)
```
### 208 - sympy/geometry/point.py:

Start line: 1261, End line: 1283

```python
class Point3D(Point):

    def transform(self, matrix):
        """Return the point after applying the transformation described
        by the 4x4 Matrix, ``matrix``.

        See Also
        ========
        geometry.entity.rotate
        geometry.entity.scale
        geometry.entity.translate
        """
        try:
            col, row = matrix.shape
            valid_matrix = matrix.is_square and col == 4
        except AttributeError:
            # We hit this block if matrix argument is not actually a Matrix.
            valid_matrix = False
        if not valid_matrix:
            raise ValueError("The argument to the transform function must be " \
            + "a 4x4 matrix")
        from sympy.matrices.expressions import Transpose
        x, y, z = self.args
        m = Transpose(matrix)
        return Point3D(*(Matrix(1, 4, [x, y, z, 1])*m).tolist()[0][:3])
```
### 218 - sympy/geometry/point.py:

Start line: 978, End line: 998

```python
class Point2D(Point):

    def transform(self, matrix):
        """Return the point after applying the transformation described
        by the 3x3 Matrix, ``matrix``.

        See Also
        ========
        geometry.entity.rotate
        geometry.entity.scale
        geometry.entity.translate
        """
        try:
            col, row = matrix.shape
            valid_matrix = matrix.is_square and col == 3
        except AttributeError:
            # We hit this block if matrix argument is not actually a Matrix.
            valid_matrix = False
        if not valid_matrix:
            raise ValueError("The argument to the transform function must be " \
            + "a 3x3 matrix")
        x, y = self.args
        return Point(*(Matrix(1, 3, [x, y, 1])*matrix).tolist()[0][:2])
```
