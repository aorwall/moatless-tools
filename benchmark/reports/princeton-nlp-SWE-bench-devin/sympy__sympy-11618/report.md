# sympy__sympy-11618

| **sympy/sympy** | `360290c4c401e386db60723ddb0109ed499c9f6e` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 1378 |
| **Any found context length** | 1378 |
| **Avg pos** | 7.0 |
| **Min pos** | 7 |
| **Max pos** | 7 |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sympy/geometry/point.py b/sympy/geometry/point.py
--- a/sympy/geometry/point.py
+++ b/sympy/geometry/point.py
@@ -266,6 +266,20 @@ def distance(self, p):
         sqrt(x**2 + y**2)
 
         """
+        if type(p) is not type(self):
+            if len(p) == len(self):
+                return sqrt(sum([(a - b)**2 for a, b in zip(
+                    self.args, p.args if isinstance(p, Point) else p)]))
+            else:
+                p1 = [0] * max(len(p), len(self))
+                p2 = p.args if len(p.args) > len(self.args) else self.args
+
+                for i in range(min(len(p), len(self))):
+                    p1[i] = p.args[i] if len(p) < len(self) else self.args[i]
+
+                return sqrt(sum([(a - b)**2 for a, b in zip(
+                    p1, p2)]))
+
         return sqrt(sum([(a - b)**2 for a, b in zip(
             self.args, p.args if isinstance(p, Point) else p)]))
 

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sympy/geometry/point.py | 269 | 269 | 7 | 1 | 1378


## Problem Statement

```
distance calculation wrong
\`\`\` python
>>> Point(2,0).distance(Point(1,0,2))
1
\`\`\`

The 3rd dimension is being ignored when the Points are zipped together to calculate the distance so `sqrt((2-1)**2 + (0-0)**2)` is being computed instead of `sqrt(5)`.


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 sympy/geometry/point.py** | 765 | 813| 306 | 306 | 7361 | 
| 2 | **1 sympy/geometry/point.py** | 272 | 303| 177 | 483 | 7361 | 
| 3 | **1 sympy/geometry/point.py** | 883 | 905| 133 | 616 | 7361 | 
| 4 | **1 sympy/geometry/point.py** | 526 | 548| 196 | 812 | 7361 | 
| 5 | **1 sympy/geometry/point.py** | 814 | 833| 182 | 994 | 7361 | 
| 6 | 2 sympy/physics/vector/point.py | 218 | 244| 187 | 1181 | 11113 | 
| **-> 7 <-** | **2 sympy/geometry/point.py** | 237 | 270| 197 | 1378 | 11113 | 
| 8 | **2 sympy/geometry/point.py** | 204 | 235| 201 | 1579 | 11113 | 
| 9 | **2 sympy/geometry/point.py** | 835 | 881| 234 | 1813 | 11113 | 
| 10 | 3 sympy/geometry/line3d.py | 955 | 987| 278 | 2091 | 22064 | 
| 11 | **3 sympy/geometry/point.py** | 1056 | 1081| 229 | 2320 | 22064 | 
| 12 | 4 sympy/vector/point.py | 1 | 43| 314 | 2634 | 23135 | 
| 13 | 5 sympy/geometry/plane.py | 305 | 364| 531 | 3165 | 29569 | 
| 14 | **5 sympy/geometry/point.py** | 907 | 932| 177 | 3342 | 29569 | 
| 15 | 5 sympy/geometry/line3d.py | 1174 | 1205| 271 | 3613 | 29569 | 
| 16 | **5 sympy/geometry/point.py** | 970 | 1017| 393 | 4006 | 29569 | 
| 17 | 6 sympy/geometry/util.py | 415 | 449| 281 | 4287 | 35061 | 
| 18 | 7 sympy/geometry/line.py | 1138 | 1177| 307 | 4594 | 47700 | 
| 19 | 7 sympy/geometry/util.py | 476 | 498| 171 | 4765 | 47700 | 
| 20 | **7 sympy/geometry/point.py** | 477 | 525| 288 | 5053 | 47700 | 
| 21 | 7 sympy/geometry/line3d.py | 1406 | 1441| 305 | 5358 | 47700 | 
| 22 | **7 sympy/geometry/point.py** | 1083 | 1104| 180 | 5538 | 47700 | 
| 23 | **7 sympy/geometry/point.py** | 406 | 475| 507 | 6045 | 47700 | 
| 24 | **7 sympy/geometry/point.py** | 28 | 77| 312 | 6357 | 47700 | 
| 25 | **7 sympy/geometry/point.py** | 693 | 718| 208 | 6565 | 47700 | 
| 26 | **7 sympy/geometry/point.py** | 78 | 118| 339 | 6904 | 47700 | 
| 27 | 7 sympy/geometry/line.py | 1412 | 1445| 257 | 7161 | 47700 | 
| 28 | 7 sympy/geometry/util.py | 451 | 474| 267 | 7428 | 47700 | 
| 29 | **7 sympy/geometry/point.py** | 743 | 763| 175 | 7603 | 47700 | 
| 30 | **7 sympy/geometry/point.py** | 1106 | 1129| 200 | 7803 | 47700 | 
| 31 | **7 sympy/geometry/point.py** | 550 | 590| 208 | 8011 | 47700 | 
| 32 | **7 sympy/geometry/point.py** | 720 | 741| 154 | 8165 | 47700 | 
| 33 | 7 sympy/geometry/line.py | 1743 | 1777| 256 | 8421 | 47700 | 
| 34 | **7 sympy/geometry/point.py** | 1 | 25| 142 | 8563 | 47700 | 
| 35 | 7 sympy/geometry/line3d.py | 884 | 916| 308 | 8871 | 47700 | 
| 36 | 7 sympy/geometry/util.py | 336 | 412| 602 | 9473 | 47700 | 
| 37 | 7 sympy/vector/point.py | 45 | 93| 354 | 9827 | 47700 | 
| 38 | 8 sympy/geometry/polygon.py | 694 | 721| 217 | 10044 | 64487 | 
| 39 | 8 sympy/vector/point.py | 123 | 157| 240 | 10284 | 64487 | 
| 40 | **8 sympy/geometry/point.py** | 334 | 365| 199 | 10483 | 64487 | 
| 41 | 9 sympy/plotting/plot.py | 1097 | 1110| 181 | 10664 | 80286 | 
| 42 | **9 sympy/geometry/point.py** | 193 | 202| 120 | 10784 | 80286 | 
| 43 | 9 sympy/geometry/line3d.py | 838 | 852| 178 | 10962 | 80286 | 
| 44 | 9 sympy/geometry/line3d.py | 1364 | 1383| 140 | 11102 | 80286 | 
| 45 | 9 sympy/geometry/line3d.py | 140 | 180| 275 | 11377 | 80286 | 
| 46 | 9 sympy/geometry/polygon.py | 861 | 920| 609 | 11986 | 80286 | 
| 47 | 9 sympy/geometry/polygon.py | 811 | 860| 517 | 12503 | 80286 | 
| 48 | 9 sympy/geometry/line3d.py | 1320 | 1333| 214 | 12717 | 80286 | 
| 49 | 9 sympy/geometry/polygon.py | 723 | 809| 781 | 13498 | 80286 | 
| 50 | 9 sympy/geometry/polygon.py | 1685 | 1725| 397 | 13895 | 80286 | 
| 51 | 9 sympy/geometry/line3d.py | 1039 | 1053| 178 | 14073 | 80286 | 
| 52 | 9 sympy/geometry/plane.py | 514 | 578| 665 | 14738 | 80286 | 
| 53 | **9 sympy/geometry/point.py** | 934 | 968| 265 | 15003 | 80286 | 
| 54 | 9 sympy/geometry/line3d.py | 1140 | 1172| 250 | 15253 | 80286 | 
| 55 | **9 sympy/geometry/point.py** | 1019 | 1054| 206 | 15459 | 80286 | 
| 56 | 9 sympy/geometry/plane.py | 51 | 69| 217 | 15676 | 80286 | 
| 57 | 9 sympy/vector/point.py | 95 | 121| 176 | 15852 | 80286 | 
| 58 | 9 sympy/geometry/util.py | 634 | 708| 708 | 16560 | 80286 | 
| 59 | 10 sympy/vector/functions.py | 337 | 403| 590 | 17150 | 84109 | 
| 60 | 11 sympy/geometry/ellipse.py | 1286 | 1332| 307 | 17457 | 95326 | 
| 61 | 12 sympy/diffgeom/diffgeom.py | 353 | 410| 445 | 17902 | 109787 | 
| 62 | 12 sympy/geometry/plane.py | 110 | 129| 242 | 18144 | 109787 | 
| 63 | 13 sympy/diffgeom/rn.py | 61 | 102| 602 | 18746 | 111105 | 
| 64 | 14 sympy/combinatorics/permutations.py | 2640 | 2665| 214 | 18960 | 133397 | 
| 65 | **14 sympy/geometry/point.py** | 305 | 332| 150 | 19110 | 133397 | 
| 66 | 14 sympy/geometry/plane.py | 92 | 108| 155 | 19265 | 133397 | 
| 67 | 14 sympy/geometry/ellipse.py | 444 | 470| 138 | 19403 | 133397 | 
| 68 | 14 sympy/geometry/line3d.py | 1279 | 1318| 318 | 19721 | 133397 | 
| 69 | 14 sympy/geometry/polygon.py | 1628 | 1683| 413 | 20134 | 133397 | 
| 70 | 14 sympy/geometry/line3d.py | 1385 | 1404| 155 | 20289 | 133397 | 
| 71 | 14 sympy/geometry/plane.py | 131 | 168| 306 | 20595 | 133397 | 
| 72 | 14 sympy/physics/vector/point.py | 1 | 33| 212 | 20807 | 133397 | 
| 73 | 15 sympy/physics/vector/fieldfunctions.py | 250 | 314| 554 | 21361 | 135721 | 
| 74 | 15 sympy/geometry/line.py | 522 | 558| 228 | 21589 | 135721 | 
| 75 | 15 sympy/geometry/polygon.py | 1727 | 1751| 141 | 21730 | 135721 | 
| 76 | 15 sympy/physics/vector/point.py | 275 | 304| 210 | 21940 | 135721 | 
| 77 | 15 sympy/geometry/line3d.py | 96 | 115| 154 | 22094 | 135721 | 
| 78 | 15 sympy/geometry/line.py | 1261 | 1303| 488 | 22582 | 135721 | 
| 79 | 15 sympy/diffgeom/diffgeom.py | 474 | 494| 201 | 22783 | 135721 | 
| 80 | **15 sympy/geometry/point.py** | 120 | 191| 573 | 23356 | 135721 | 
| 81 | 15 sympy/physics/vector/point.py | 458 | 499| 300 | 23656 | 135721 | 
| 82 | 15 sympy/geometry/line3d.py | 631 | 718| 759 | 24415 | 135721 | 
| 83 | 15 sympy/geometry/plane.py | 580 | 632| 440 | 24855 | 135721 | 
| 84 | 16 sympy/geometry/entity.py | 510 | 531| 180 | 25035 | 140199 | 
| 85 | **16 sympy/geometry/point.py** | 659 | 691| 207 | 25242 | 140199 | 
| 86 | 16 sympy/geometry/polygon.py | 1913 | 1947| 277 | 25519 | 140199 | 
| 87 | 16 sympy/geometry/line.py | 1701 | 1720| 115 | 25634 | 140199 | 
| 88 | 16 sympy/physics/vector/point.py | 431 | 456| 171 | 25805 | 140199 | 
| 89 | 16 sympy/geometry/ellipse.py | 367 | 386| 114 | 25919 | 140199 | 
| 90 | **16 sympy/geometry/point.py** | 367 | 404| 209 | 26128 | 140199 | 
| 91 | 16 sympy/geometry/polygon.py | 2008 | 2033| 176 | 26304 | 140199 | 
| 92 | 16 sympy/geometry/line.py | 1567 | 1614| 299 | 26603 | 140199 | 
| 93 | 16 sympy/geometry/line3d.py | 117 | 138| 186 | 26789 | 140199 | 
| 94 | 16 sympy/geometry/line3d.py | 1055 | 1074| 140 | 26929 | 140199 | 
| 95 | 16 sympy/geometry/util.py | 1 | 23| 115 | 27044 | 140199 | 
| 96 | 17 sympy/geometry/__init__.py | 1 | 30| 213 | 27257 | 140413 | 
| 97 | 18 sympy/physics/unitsystems/dimensions.py | 1 | 21| 159 | 27416 | 144282 | 
| 98 | 18 sympy/physics/vector/point.py | 112 | 156| 418 | 27834 | 144282 | 
| 99 | 18 sympy/geometry/polygon.py | 2133 | 2157| 148 | 27982 | 144282 | 
| 100 | 18 sympy/physics/vector/point.py | 57 | 110| 538 | 28520 | 144282 | 
| 101 | 18 sympy/geometry/line3d.py | 1076 | 1106| 238 | 28758 | 144282 | 
| 102 | 18 sympy/geometry/line.py | 1616 | 1628| 188 | 28946 | 144282 | 
| 103 | 18 sympy/geometry/line3d.py | 75 | 94| 142 | 29088 | 144282 | 
| 104 | **18 sympy/geometry/point.py** | 592 | 657| 459 | 29547 | 144282 | 
| 105 | 18 sympy/geometry/line.py | 1002 | 1028| 254 | 29801 | 144282 | 
| 106 | 18 sympy/geometry/polygon.py | 1949 | 1978| 198 | 29999 | 144282 | 
| 107 | 18 sympy/physics/vector/point.py | 158 | 185| 176 | 30175 | 144282 | 
| 108 | 18 sympy/geometry/ellipse.py | 1334 | 1353| 184 | 30359 | 144282 | 
| 109 | 18 sympy/physics/vector/point.py | 335 | 384| 458 | 30817 | 144282 | 
| 110 | 18 sympy/physics/vector/point.py | 386 | 429| 375 | 31192 | 144282 | 
| 111 | 18 sympy/physics/unitsystems/dimensions.py | 232 | 250| 188 | 31380 | 144282 | 
| 112 | 18 sympy/geometry/line3d.py | 989 | 1037| 311 | 31691 | 144282 | 
| 113 | 18 sympy/diffgeom/diffgeom.py | 259 | 350| 672 | 32363 | 144282 | 
| 114 | 18 sympy/geometry/polygon.py | 2188 | 2222| 283 | 32646 | 144282 | 
| 115 | 18 sympy/geometry/plane.py | 698 | 723| 333 | 32979 | 144282 | 
| 116 | 18 sympy/geometry/polygon.py | 355 | 388| 265 | 33244 | 144282 | 
| 117 | 18 sympy/geometry/line3d.py | 720 | 765| 372 | 33616 | 144282 | 
| 118 | 18 sympy/geometry/line3d.py | 338 | 380| 323 | 33939 | 144282 | 
| 119 | 18 sympy/geometry/line.py | 472 | 520| 283 | 34222 | 144282 | 
| 120 | 18 sympy/physics/unitsystems/dimensions.py | 540 | 563| 123 | 34345 | 144282 | 
| 121 | 18 sympy/geometry/line.py | 1722 | 1741| 126 | 34471 | 144282 | 
| 122 | 18 sympy/geometry/polygon.py | 2224 | 2251| 207 | 34678 | 144282 | 
| 123 | 18 sympy/geometry/line.py | 1305 | 1324| 115 | 34793 | 144282 | 
| 124 | 19 sympy/geometry/parabola.py | 166 | 199| 194 | 34987 | 146395 | 
| 125 | 19 sympy/geometry/polygon.py | 2333 | 2350| 191 | 35178 | 146395 | 
| 126 | 19 sympy/geometry/ellipse.py | 1355 | 1378| 118 | 35296 | 146395 | 
| 127 | 19 sympy/geometry/line3d.py | 854 | 882| 216 | 35512 | 146395 | 
| 128 | 19 sympy/geometry/plane.py | 24 | 50| 297 | 35809 | 146395 | 
| 129 | 19 sympy/geometry/polygon.py | 1203 | 1230| 144 | 35953 | 146395 | 
| 130 | 19 sympy/geometry/line.py | 1326 | 1345| 119 | 36072 | 146395 | 
| 131 | 19 sympy/geometry/plane.py | 71 | 90| 144 | 36216 | 146395 | 
| 132 | 19 sympy/geometry/line.py | 1347 | 1377| 211 | 36427 | 146395 | 
| 133 | 20 sympy/core/numbers.py | 964 | 997| 368 | 36795 | 172883 | 
| 134 | 20 sympy/geometry/polygon.py | 1980 | 2006| 181 | 36976 | 172883 | 
| 135 | 20 sympy/geometry/line3d.py | 1207 | 1234| 199 | 37175 | 172883 | 
| 136 | 20 sympy/geometry/polygon.py | 1298 | 1338| 220 | 37395 | 172883 | 
| 137 | 20 sympy/geometry/polygon.py | 24 | 113| 766 | 38161 | 172883 | 
| 138 | 20 sympy/geometry/line3d.py | 1108 | 1138| 239 | 38400 | 172883 | 
| 139 | 21 sympy/physics/optics/gaussopt.py | 200 | 226| 129 | 38529 | 178578 | 
| 140 | 21 sympy/physics/unitsystems/dimensions.py | 208 | 230| 152 | 38681 | 178578 | 
| 141 | 21 sympy/geometry/ellipse.py | 204 | 228| 121 | 38802 | 178578 | 
| 142 | 21 sympy/core/numbers.py | 34 | 71| 478 | 39280 | 178578 | 
| 143 | 21 sympy/geometry/line3d.py | 806 | 836| 234 | 39514 | 178578 | 
| 144 | 21 sympy/geometry/line3d.py | 513 | 591| 647 | 40161 | 178578 | 
| 145 | 22 sympy/functions/special/hyper.py | 944 | 955| 156 | 40317 | 188371 | 
| 146 | 23 sympy/physics/units.py | 130 | 212| 791 | 41108 | 191114 | 
| 147 | 23 sympy/geometry/entity.py | 131 | 159| 323 | 41431 | 191114 | 
| 148 | 24 sympy/physics/unitsystems/__init__.py | 1 | 28| 238 | 41669 | 191353 | 
| 149 | 24 sympy/geometry/plane.py | 488 | 512| 158 | 41827 | 191353 | 
| 150 | 24 sympy/physics/vector/point.py | 35 | 55| 210 | 42037 | 191353 | 
| 151 | 24 sympy/geometry/line3d.py | 250 | 295| 354 | 42391 | 191353 | 


## Patch

```diff
diff --git a/sympy/geometry/point.py b/sympy/geometry/point.py
--- a/sympy/geometry/point.py
+++ b/sympy/geometry/point.py
@@ -266,6 +266,20 @@ def distance(self, p):
         sqrt(x**2 + y**2)
 
         """
+        if type(p) is not type(self):
+            if len(p) == len(self):
+                return sqrt(sum([(a - b)**2 for a, b in zip(
+                    self.args, p.args if isinstance(p, Point) else p)]))
+            else:
+                p1 = [0] * max(len(p), len(self))
+                p2 = p.args if len(p.args) > len(self.args) else self.args
+
+                for i in range(min(len(p), len(self))):
+                    p1[i] = p.args[i] if len(p) < len(self) else self.args[i]
+
+                return sqrt(sum([(a - b)**2 for a, b in zip(
+                    p1, p2)]))
+
         return sqrt(sum([(a - b)**2 for a, b in zip(
             self.args, p.args if isinstance(p, Point) else p)]))
 

```

## Test Patch

```diff
diff --git a/sympy/geometry/tests/test_point.py b/sympy/geometry/tests/test_point.py
--- a/sympy/geometry/tests/test_point.py
+++ b/sympy/geometry/tests/test_point.py
@@ -243,6 +243,11 @@ def test_issue_9214():
 
     assert Point3D.are_collinear(p1, p2, p3) is False
 
+def test_issue_11617():
+    p1 = Point3D(1,0,2)
+    p2 = Point2D(2,0)
+
+    assert p1.distance(p2) == sqrt(5)
 
 def test_transform():
     p = Point(1, 1)

```


## Code snippets

### 1 - sympy/geometry/point.py:

Start line: 765, End line: 813

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

    Notes
    =====

    Currently only 2-dimensional and 3-dimensional points are supported.

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
```
### 2 - sympy/geometry/point.py:

Start line: 272, End line: 303

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

        sympy.geometry.Point.distance

        Examples
        ========

        >>> from sympy.geometry import Point
        >>> p1, p2 = Point(1, 1), Point(4, 5)
        >>> p1.taxicab_distance(p2)
        7

        """
        p = Point(p)
        return sum(abs(a - b) for a, b in zip(self.args, p.args))
```
### 3 - sympy/geometry/point.py:

Start line: 883, End line: 905

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
### 4 - sympy/geometry/point.py:

Start line: 526, End line: 548

```python
class Point2D(Point):
    def __new__(cls, *args, **kwargs):
        eval = kwargs.get('evaluate', global_evaluate[0])
        check = True
        if isinstance(args[0], Point2D):
            if not eval:
                return args[0]
            args = args[0].args
            check = False
        else:
            if iterable(args[0]):
                args = args[0]
            if len(args) != 2:
                raise ValueError(
                    "Only two dimensional points currently supported")
        coords = Tuple(*args)
        if check:
            if any(a.is_number and im(a) for a in coords):
                raise ValueError('Imaginary args not permitted.')
        if eval:
            coords = coords.xreplace(dict(
                [(f, simplify(nsimplify(f, rational=True)))
                for f in coords.atoms(Float)]))
        return GeometryEntity.__new__(cls, *coords)
```
### 5 - sympy/geometry/point.py:

Start line: 814, End line: 833

```python
class Point3D(Point):
    def __new__(cls, *args, **kwargs):
        eval = kwargs.get('evaluate', global_evaluate[0])
        if isinstance(args[0], (Point, Point3D)):
            if not eval:
                return args[0]
            args = args[0].args
        else:
            if iterable(args[0]):
                args = args[0]
            if len(args) not in (2, 3):
                raise TypeError(
                    "Enter a 2 or 3 dimensional point")
        coords = Tuple(*args)
        if len(coords) == 2:
            coords += (S.Zero,)
        if eval:
            coords = coords.xreplace(dict(
                [(f, simplify(nsimplify(f, rational=True)))
                for f in coords.atoms(Float)]))
        return GeometryEntity.__new__(cls, *coords)
```
### 6 - sympy/physics/vector/point.py:

Start line: 218, End line: 244

```python
class Point(object):

    def pos_from(self, otherpoint):
        """Returns a Vector distance between this Point and the other Point.

        Parameters
        ==========

        otherpoint : Point
            The otherpoint we are locating this one relative to

        Examples
        ========

        >>> from sympy.physics.vector import Point, ReferenceFrame
        >>> N = ReferenceFrame('N')
        >>> p1 = Point('p1')
        >>> p2 = Point('p2')
        >>> p1.set_pos(p2, 10 * N.x)
        >>> p1.pos_from(p2)
        10*N.x

        """

        outvec = Vector(0)
        plist = self._pdict_list(otherpoint, 0)
        for i in range(len(plist) - 1):
            outvec += plist[i]._pos_dict[plist[i + 1]]
        return outvec
```
### 7 - sympy/geometry/point.py:

Start line: 237, End line: 270

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
        return sqrt(sum([(a - b)**2 for a, b in zip(
            self.args, p.args if isinstance(p, Point) else p)]))
```
### 8 - sympy/geometry/point.py:

Start line: 204, End line: 235

```python
class Point(GeometryEntity):

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

    @property
    def origin(self):
        """A point of all zeros of the same ambient dimension
        as the current point"""
        return Point([0]*len(self))

    @property
    def is_zero(self):
        """True if every coordinate is zero, otherwise False."""
        return all(x == S.Zero for x in self.args)

    @property
    def ambient_dimension(self):
        """The dimension of the ambient space the point is in.
        I.e., if the point is in R^n, the ambient dimension
        will be n"""
        return len(self)
```
### 9 - sympy/geometry/point.py:

Start line: 835, End line: 881

```python
class Point3D(Point):

    def __contains__(self, item):
        return item == self

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
### 10 - sympy/geometry/line3d.py:

Start line: 955, End line: 987

```python
class Line3D(LinearEntity3D):

    def distance(self, o):
        """
        Finds the shortest distance between a line and a point.

        Raises
        ======

        NotImplementedError is raised if o is not an instance of Point3D

        Examples
        ========

        >>> from sympy import Point3D, Line3D
        >>> p1, p2 = Point3D(0, 0, 0), Point3D(1, 1, 1)
        >>> s = Line3D(p1, p2)
        >>> s.distance(Point3D(-1, 1, 1))
        2*sqrt(6)/3
        >>> s.distance((-1, 1, 1))
        2*sqrt(6)/3
        """
        if not isinstance(o, Point3D):
            if is_sequence(o):
                o = Point3D(o)
        if o in self:
            return S.Zero
        a = self.perpendicular_segment(o).length
        return a

    def equals(self, other):
        """Returns True if self and other are the same mathematical entities"""
        if not isinstance(other, Line3D):
            return False
        return Point3D.are_collinear(self.p1, other.p1, self.p2, other.p2)
```
### 11 - sympy/geometry/point.py:

Start line: 1056, End line: 1081

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
### 14 - sympy/geometry/point.py:

Start line: 907, End line: 932

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
        b = sqrt(sum(i**2 for i in a))
        return [(point.x - self.x) / b,(point.y - self.y) / b,
                (point.z - self.z) / b]
```
### 16 - sympy/geometry/point.py:

Start line: 970, End line: 1017

```python
class Point3D(Point):

    @staticmethod
    def are_coplanar(*points):
        """

        This function tests whether passed points are coplanar or not.
        It uses the fact that the triple scalar product of three vectors
        vanishes if the vectors are coplanar. Which means that the volume
        of the solid described by them will have to be zero for coplanarity.

        Parameters
        ==========

        A set of points 3D points

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
        from sympy.geometry.plane import Plane
        points = list(set(points))
        if len(points) < 3:
            raise ValueError('At least 3 points are needed to define a plane.')
        a, b = points[:2]
        for i, c in enumerate(points[2:]):
            try:
                p = Plane(a, b, c)
                for j in (0, 1, i):
                    points.pop(j)
                return all(p.is_coplanar(i) for i in points)
            except ValueError:
                pass
        raise ValueError('At least 3 non-collinear points needed to define plane.')
```
### 20 - sympy/geometry/point.py:

Start line: 477, End line: 525

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
```
### 22 - sympy/geometry/point.py:

Start line: 1083, End line: 1104

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
### 23 - sympy/geometry/point.py:

Start line: 406, End line: 475

```python
class Point(GeometryEntity):

    def dot(self, p2):
        """Return dot product of self with another Point."""
        p2 = Point(p2)
        return Add(*[a*b for a,b in zip(self, p2)])

    def equals(self, other):
        """Returns whether the coordinates of self and other agree."""
        # a point is equal to another point if all its components are equal
        if not isinstance(other, Point) or len(self.args) != len(other.args):
            return False
        return all(a.equals(b) for a,b in zip(self.args, other.args))

    def __len__(self):
        return len(self.args)

    def __iter__(self):
        return self.args.__iter__()

    def __eq__(self, other):
        if not isinstance(other, Point) or len(self.args) != len(other.args):
            return False
        return self.args == other.args

    def __hash__(self):
        return hash(self.args)

    def __getitem__(self, key):
        return self.args[key]

    def __add__(self, other):
        """Add other to self by incrementing self's coordinates by those of other.

        See Also
        ========

        sympy.geometry.entity.translate

        """

        if iterable(other) and len(other) == len(self):
            return Point([simplify(a + b) for a, b in zip(self, other)])
        else:
            raise ValueError(
                "Points must have the same number of dimensions")

    def __sub__(self, other):
        """Subtract two points, or subtract a factor from this point's
        coordinates."""
        return self + (-other)

    def __mul__(self, factor):
        """Multiply point's coordinates by a factor."""
        factor = sympify(factor)
        return Point([simplify(x*factor) for x in self.args])

    def __div__(self, divisor):
        """Divide point's coordinates by a factor."""
        divisor = sympify(divisor)
        return Point([simplify(x/divisor) for x in self.args])

    __truediv__ = __div__

    def __neg__(self):
        """Negate the point."""
        return Point([-x for x in self.args])

    def __abs__(self):
        """Returns the distance between this point and the origin."""
        origin = Point([0]*len(self))
        return Point.distance(origin, self)
```
### 24 - sympy/geometry/point.py:

Start line: 28, End line: 77

```python
class Point(GeometryEntity):
    """A point in a n-dimensional Euclidean space.

    Parameters
    ==========

    coords : sequence of n-coordinate values. In the special
    case where n=2 or 3, a Point2D or Point3D will be created
    as appropriate.

    Attributes
    ==========

    length
    origin: A `Point` representing the origin of the
        appropriately-dimensioned space.

    Raises
    ======

    TypeError
        When trying to add or subtract points with different dimensions.
        When `intersection` is called with object other than a Point.

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

    Floats are automatically converted to Rational unless the
    evaluate flag is False:

    >>> Point(0.5, 0.25)
    Point2D(1/2, 1/4)
    >>> Point(0.5, 0.25, evaluate=False)
    Point2D(0.5, 0.25)

    """
```
### 25 - sympy/geometry/point.py:

Start line: 693, End line: 718

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
            pt = Point(pt)
            return self.translate(*(-pt).args).scale(x, y).translate(*pt.args)
        return Point(self.x*x, self.y*y)
```
### 26 - sympy/geometry/point.py:

Start line: 78, End line: 118

```python
class Point(GeometryEntity):
    def __new__(cls, *args, **kwargs):
        evaluate = kwargs.get('evaluate', global_evaluate[0])

        if iterable(args[0]):
            if isinstance(args[0], Point) and not evaluate:
                return args[0]
            args = args[0]

        # unpack the arguments into a friendly Tuple
        # if we were already a Point, we're doing an excess
        # iteration, but we'll worry about efficiency later
        coords = Tuple(*args)
        if any(a.is_number and im(a) for a in coords):
            raise ValueError('Imaginary coordinates not permitted.')

        # Turn any Floats into rationals and simplify
        # any expressions before we instantiate
        if evaluate:
            coords = coords.xreplace(dict(
                [(f, simplify(nsimplify(f, rational=True)))
                for f in coords.atoms(Float)]))
        if len(coords) == 2:
            return Point2D(coords, **kwargs)
        if len(coords) == 3:
            return Point3D(coords, **kwargs)

        return GeometryEntity.__new__(cls, *coords)

    is_Point = True

    def __contains__(self, item):
        return item in self.args

    def is_concyclic(*args):
        # Coincident points are irrelevant and can confuse this algorithm.
        # Use only unique points.
        args = list(set(args))
        if not all(isinstance(p, Point) for p in args):
            raise TypeError('Must pass only Point objects')

        return args[0].is_concyclic(*args[1:])
```
### 29 - sympy/geometry/point.py:

Start line: 743, End line: 763

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
### 30 - sympy/geometry/point.py:

Start line: 1106, End line: 1129

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
### 31 - sympy/geometry/point.py:

Start line: 550, End line: 590

```python
class Point2D(Point):

    def __contains__(self, item):
        return item == self

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

    @property
    def bounds(self):
        """Return a tuple (xmin, ymin, xmax, ymax) representing the bounding
        rectangle for the geometric figure.

        """

        return (self.x, self.y, self.x, self.y)
```
### 32 - sympy/geometry/point.py:

Start line: 720, End line: 741

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
### 34 - sympy/geometry/point.py:

Start line: 1, End line: 25

```python
"""Geometrical Points.

Contains
========
Point
Point2D
Point3D

"""

from __future__ import division, print_function

from sympy.core import S, sympify
from sympy.core.compatibility import iterable
from sympy.core.containers import Tuple
from sympy.simplify import nsimplify, simplify
from sympy.geometry.exceptions import GeometryError
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.complexes import im
from sympy.matrices import Matrix
from sympy.core.numbers import Float
from sympy.core.evaluate import global_evaluate
from sympy.core.add import Add

from .entity import GeometryEntity
```
### 40 - sympy/geometry/point.py:

Start line: 334, End line: 365

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

    n = evalf
```
### 42 - sympy/geometry/point.py:

Start line: 193, End line: 202

```python
class Point(GeometryEntity):

    def is_scalar_multiple(p1, p2):
        """Returns whether `p1` and `p2` are scalar multiples
        of eachother.
        """
        # if the vectors p1 and p2 are linearly dependent, then they must
        # be scalar multiples of eachother
        m = Matrix([p1.args, p2.args])
        # XXX: issue #9480 we need `simplify=True` otherwise the
        # rank may be computed incorrectly
        return m.rank(simplify=True) < 2
```
### 53 - sympy/geometry/point.py:

Start line: 934, End line: 968

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

        sympy.geometry.line3d.Line3D

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
### 55 - sympy/geometry/point.py:

Start line: 1019, End line: 1054

```python
class Point3D(Point):

    def intersection(self, o):
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
        if isinstance(o, Point3D):
            if self == o:
                return [self]
            return []

        return o.intersection(self)
```
### 65 - sympy/geometry/point.py:

Start line: 305, End line: 332

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
        return Point([simplify((a + b)*S.Half) for a, b in zip(self.args, p.args)])
```
### 80 - sympy/geometry/point.py:

Start line: 120, End line: 191

```python
class Point(GeometryEntity):

    def is_collinear(*args):
        """Is a sequence of points collinear?

        Test whether or not a set of points are collinear. Returns True if
        the set of points are collinear, or False otherwise.

        Parameters
        ==========

        points : sequence of Point

        Returns
        =======

        is_collinear : boolean

        Notes
        =====

        Slope is preserved everywhere on a line, so the slope between
        any two points on the line should be the same. Take the first
        two points, p1 and p2, and create a translated point v1
        with p1 as the origin. Now for every other point we create
        a translated point, vi with p1 also as the origin. Note that
        these translations preserve slope since everything is
        consistently translated to a new origin of p1. Since slope
        is preserved then we have the following equality:

              * v1_slope = vi_slope
              * v1.y/v1.x = vi.y/vi.x (due to translation)
              * v1.y*vi.x = vi.y*v1.x
              * v1.y*vi.x - vi.y*v1.x = 0           (*)

        Hence, if we have a vi such that the equality in (*) is False
        then the points are not collinear. We do this test for every
        point in the list, and if all pass then they are collinear.

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

        # Coincident points are irrelevant; use only unique points.
        args = list(set(args))
        if not all(isinstance(p, Point) for p in args):
            raise TypeError('Must pass only Point objects')

        if len(args) == 0:
            return False
        if len(args) <= 2:
            return True

        # translate our points
        points = [p - args[0] for p in args[1:]]
        for p in points[1:]:
            if not Point.is_scalar_multiple(points[0], p):
                return False
        return True
```
### 85 - sympy/geometry/point.py:

Start line: 659, End line: 691

```python
class Point2D(Point):

    def rotate(self, angle, pt=None):
        """Rotate ``angle`` radians counterclockwise about Point ``pt``.

        See Also
        ========

        rotate, scale

        Examples
        ========

        >>> from sympy import Point2D, pi
        >>> t = Point2D(1, 0)
        >>> t.rotate(pi/2)
        Point2D(0, 1)
        >>> t.rotate(pi/2, (2, 0))
        Point2D(2, -1)

        """
        from sympy import cos, sin, Point

        c = cos(angle)
        s = sin(angle)

        rv = self
        if pt is not None:
            pt = Point(pt)
            rv -= pt
        x, y = rv.args
        rv = Point(c*x - s*y, s*x + c*y)
        if pt is not None:
            rv += pt
        return rv
```
### 90 - sympy/geometry/point.py:

Start line: 367, End line: 404

```python
class Point(GeometryEntity):

    def intersection(self, o):
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
        if isinstance(o, Point):
            if len(self) != len(o):
                raise ValueError("Points must be of the same dimension to intersect")
            if self == o:
                return [self]
            return []

        return o.intersection(self)
```
### 104 - sympy/geometry/point.py:

Start line: 592, End line: 657

```python
class Point2D(Point):

    def is_concyclic(*points):
        """Is a sequence of points concyclic?

        Test whether or not a sequence of points are concyclic (i.e., they lie
        on a circle).

        Parameters
        ==========

        points : sequence of Points

        Returns
        =======

        is_concyclic : boolean
            True if points are concyclic, False otherwise.

        See Also
        ========

        sympy.geometry.ellipse.Circle

        Notes
        =====

        No points are not considered to be concyclic. One or two points
        are definitely concyclic and three points are conyclic iff they
        are not collinear.

        For more than three points, create a circle from the first three
        points. If the circle cannot be created (i.e., they are collinear)
        then all of the points cannot be concyclic. If the circle is created
        successfully then simply check the remaining points for containment
        in the circle.

        Examples
        ========

        >>> from sympy.geometry import Point
        >>> p1, p2 = Point(-1, 0), Point(1, 0)
        >>> p3, p4 = Point(0, 1), Point(-1, 2)
        >>> Point.is_concyclic(p1, p2, p3)
        True
        >>> Point.is_concyclic(p1, p2, p3, p4)
        False

        """
        if len(points) == 0:
            return False
        if len(points) <= 2:
            return True
        points = [Point(p) for p in points]
        if len(points) == 3:
            return (not Point.is_collinear(*points))

        try:
            from .ellipse import Circle
            c = Circle(points[0], points[1], points[2])
            for point in points[3:]:
                if point not in c:
                    return False
            return True
        except GeometryError:
            # Circle could not be created, because of collinearity of the
            # three points passed in, hence they are not concyclic.
            return False
```
