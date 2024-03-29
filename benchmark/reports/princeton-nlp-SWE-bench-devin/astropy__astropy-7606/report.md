# astropy__astropy-7606

| **astropy/astropy** | `3cedd79e6c121910220f8e6df77c54a0b344ea94` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 994 |
| **Any found context length** | 454 |
| **Avg pos** | 4.0 |
| **Min pos** | 1 |
| **Max pos** | 3 |
| **Top file pos** | 1 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/astropy/units/core.py b/astropy/units/core.py
--- a/astropy/units/core.py
+++ b/astropy/units/core.py
@@ -728,7 +728,7 @@ def __eq__(self, other):
         try:
             other = Unit(other, parse_strict='silent')
         except (ValueError, UnitsError, TypeError):
-            return False
+            return NotImplemented
 
         # Other is Unit-like, but the test below requires it is a UnitBase
         # instance; if it is not, give up (so that other can try).
@@ -1710,8 +1710,12 @@ def _unrecognized_operator(self, *args, **kwargs):
         _unrecognized_operator
 
     def __eq__(self, other):
-        other = Unit(other, parse_strict='silent')
-        return isinstance(other, UnrecognizedUnit) and self.name == other.name
+        try:
+            other = Unit(other, parse_strict='silent')
+        except (ValueError, UnitsError, TypeError):
+            return NotImplemented
+
+        return isinstance(other, type(self)) and self.name == other.name
 
     def __ne__(self, other):
         return not (self == other)

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| astropy/units/core.py | 731 | 731 | 3 | 1 | 994
| astropy/units/core.py | 1713 | 1714 | 1 | 1 | 454


## Problem Statement

```
Unit equality comparison with None raises TypeError for UnrecognizedUnit
\`\`\`
In [12]: x = u.Unit('asdf', parse_strict='silent')

In [13]: x == None  # Should be False
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-13-2486f2ccf928> in <module>()
----> 1 x == None  # Should be False

/Users/aldcroft/anaconda3/lib/python3.5/site-packages/astropy/units/core.py in __eq__(self, other)
   1699 
   1700     def __eq__(self, other):
-> 1701         other = Unit(other, parse_strict='silent')
   1702         return isinstance(other, UnrecognizedUnit) and self.name == other.name
   1703 

/Users/aldcroft/anaconda3/lib/python3.5/site-packages/astropy/units/core.py in __call__(self, s, represents, format, namespace, doc, parse_strict)
   1808 
   1809         elif s is None:
-> 1810             raise TypeError("None is not a valid Unit")
   1811 
   1812         else:

TypeError: None is not a valid Unit
\`\`\`

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 astropy/units/core.py** | 1675 | 1733| 454 | 454 | 16841 | 
| 2 | **1 astropy/units/core.py** | 1396 | 1436| 357 | 811 | 16841 | 
| **-> 3 <-** | **1 astropy/units/core.py** | 719 | 741| 183 | 994 | 16841 | 
| 4 | **1 astropy/units/core.py** | 794 | 823| 256 | 1250 | 16841 | 
| 5 | **1 astropy/units/core.py** | 763 | 792| 242 | 1492 | 16841 | 
| 6 | **1 astropy/units/core.py** | 743 | 761| 143 | 1635 | 16841 | 
| 7 | 2 astropy/units/function/core.py | 259 | 283| 209 | 1844 | 22068 | 
| 8 | 2 astropy/units/function/core.py | 170 | 201| 276 | 2120 | 22068 | 
| 9 | **2 astropy/units/core.py** | 1183 | 1198| 183 | 2303 | 22068 | 
| 10 | 2 astropy/units/function/core.py | 572 | 592| 164 | 2467 | 22068 | 
| 11 | 3 astropy/units/format/vounit.py | 98 | 122| 185 | 2652 | 23945 | 
| 12 | **3 astropy/units/core.py** | 874 | 902| 244 | 2896 | 23945 | 
| 13 | 4 astropy/units/quantity_helper.py | 288 | 368| 809 | 3705 | 31138 | 
| 14 | **4 astropy/units/core.py** | 1828 | 1946| 788 | 4493 | 31138 | 
| 15 | 4 astropy/units/quantity_helper.py | 586 | 693| 1104 | 5597 | 31138 | 
| 16 | 5 astropy/io/misc/asdf/tags/unit/unit.py | 4 | 26| 164 | 5761 | 31327 | 
| 17 | **5 astropy/units/core.py** | 1330 | 1362| 293 | 6054 | 31327 | 
| 18 | **5 astropy/units/core.py** | 1 | 32| 196 | 6250 | 31327 | 
| 19 | 5 astropy/units/format/vounit.py | 82 | 96| 131 | 6381 | 31327 | 
| 20 | **5 astropy/units/core.py** | 614 | 639| 184 | 6565 | 31327 | 
| 21 | 5 astropy/units/format/vounit.py | 124 | 152| 249 | 6814 | 31327 | 
| 22 | 5 astropy/units/format/vounit.py | 186 | 236| 362 | 7176 | 31327 | 
| 23 | **5 astropy/units/core.py** | 701 | 717| 140 | 7316 | 31327 | 
| 24 | 5 astropy/units/quantity_helper.py | 1 | 40| 218 | 7534 | 31327 | 
| 25 | 5 astropy/units/format/vounit.py | 154 | 184| 230 | 7764 | 31327 | 
| 26 | **5 astropy/units/core.py** | 211 | 243| 214 | 7978 | 31327 | 
| 27 | **5 astropy/units/core.py** | 661 | 677| 144 | 8122 | 31327 | 
| 28 | 6 astropy/units/physical.py | 69 | 134| 838 | 8960 | 32533 | 
| 29 | 7 astropy/units/format/generic.py | 440 | 495| 381 | 9341 | 36128 | 
| 30 | **7 astropy/units/core.py** | 679 | 699| 154 | 9495 | 36128 | 
| 31 | **7 astropy/units/core.py** | 1364 | 1394| 277 | 9772 | 36128 | 
| 32 | **7 astropy/units/core.py** | 1736 | 1825| 663 | 10435 | 36128 | 
| 33 | 8 astropy/units/si.py | 1 | 82| 733 | 11168 | 38236 | 
| 34 | 8 astropy/units/function/core.py | 303 | 332| 235 | 11403 | 38236 | 
| 35 | 8 astropy/units/quantity_helper.py | 85 | 99| 104 | 11507 | 38236 | 
| 36 | **8 astropy/units/core.py** | 1646 | 1672| 160 | 11667 | 38236 | 
| 37 | **8 astropy/units/core.py** | 1225 | 1251| 276 | 11943 | 38236 | 
| 38 | 8 astropy/units/physical.py | 48 | 66| 117 | 12060 | 38236 | 
| 39 | 8 astropy/units/format/vounit.py | 28 | 80| 592 | 12652 | 38236 | 
| 40 | **8 astropy/units/core.py** | 1000 | 1121| 947 | 13599 | 38236 | 
| 41 | 8 astropy/units/quantity_helper.py | 102 | 226| 881 | 14480 | 38236 | 
| 42 | **8 astropy/units/core.py** | 937 | 971| 260 | 14740 | 38236 | 
| 43 | 9 astropy/units/imperial.py | 1 | 107| 754 | 15494 | 39527 | 
| 44 | 9 astropy/units/function/core.py | 285 | 301| 155 | 15649 | 39527 | 
| 45 | 10 astropy/coordinates/errors.py | 154 | 176| 159 | 15808 | 40616 | 
| 46 | 11 astropy/units/astrophys.py | 78 | 154| 752 | 16560 | 42327 | 
| 47 | **11 astropy/units/core.py** | 1200 | 1223| 229 | 16789 | 42327 | 
| 48 | **11 astropy/units/core.py** | 973 | 998| 191 | 16980 | 42327 | 
| 49 | 11 astropy/units/function/core.py | 594 | 618| 169 | 17149 | 42327 | 
| 50 | 12 astropy/units/format/fits.py | 82 | 108| 199 | 17348 | 43595 | 
| 51 | **12 astropy/units/core.py** | 641 | 659| 155 | 17503 | 43595 | 
| 52 | 13 astropy/units/format/ogip.py | 402 | 423| 165 | 17668 | 47228 | 
| 53 | 13 astropy/units/function/core.py | 1 | 25| 221 | 17889 | 47228 | 
| 54 | **13 astropy/units/core.py** | 1298 | 1328| 168 | 18057 | 47228 | 
| 55 | **13 astropy/units/core.py** | 904 | 935| 336 | 18393 | 47228 | 
| 56 | 14 astropy/nddata/nduncertainty.py | 129 | 145| 141 | 18534 | 52524 | 
| 57 | 14 astropy/units/imperial.py | 108 | 168| 386 | 18920 | 52524 | 
| 58 | 15 astropy/coordinates/attributes.py | 285 | 319| 270 | 19190 | 56091 | 
| 59 | **15 astropy/units/core.py** | 531 | 543| 129 | 19319 | 56091 | 
| 60 | 15 astropy/units/format/generic.py | 497 | 528| 215 | 19534 | 56091 | 
| 61 | **15 astropy/units/core.py** | 2084 | 2111| 224 | 19758 | 56091 | 
| 62 | **15 astropy/units/core.py** | 285 | 334| 502 | 20260 | 56091 | 
| 63 | 16 astropy/units/function/magnitude_zero_points.py | 1 | 67| 446 | 20706 | 56616 | 
| 64 | 16 astropy/units/quantity_helper.py | 432 | 493| 601 | 21307 | 56616 | 
| 65 | **16 astropy/units/core.py** | 1625 | 1644| 194 | 21501 | 56616 | 
| 66 | **16 astropy/units/core.py** | 1253 | 1296| 328 | 21829 | 56616 | 
| 67 | 17 astropy/io/votable/exceptions.py | 1023 | 1034| 104 | 21933 | 69084 | 
| 68 | 18 astropy/units/format/utils.py | 191 | 219| 212 | 22145 | 70462 | 
| 69 | 18 astropy/units/astrophys.py | 1 | 77| 732 | 22877 | 70462 | 
| 70 | 18 astropy/units/astrophys.py | 155 | 181| 184 | 23061 | 70462 | 
| 71 | 18 astropy/units/function/core.py | 536 | 548| 141 | 23202 | 70462 | 
| 72 | 18 astropy/units/si.py | 84 | 176| 741 | 23943 | 70462 | 
| 73 | **18 astropy/units/core.py** | 825 | 872| 369 | 24312 | 70462 | 
| 74 | 19 astropy/visualization/units.py | 1 | 51| 329 | 24641 | 71117 | 
| 75 | 20 astropy/units/utils.py | 163 | 188| 239 | 24880 | 72974 | 
| 76 | 21 astropy/units/required_by_vounit.py | 1 | 60| 337 | 25217 | 73477 | 
| 77 | **21 astropy/units/core.py** | 488 | 529| 288 | 25505 | 73477 | 
| 78 | 21 astropy/coordinates/attributes.py | 397 | 430| 225 | 25730 | 73477 | 
| 79 | 21 astropy/units/si.py | 177 | 242| 592 | 26322 | 73477 | 
| 80 | 22 astropy/units/equivalencies.py | 5 | 51| 339 | 26661 | 80193 | 
| 81 | 22 astropy/units/format/generic.py | 17 | 52| 232 | 26893 | 80193 | 
| 82 | 23 astropy/modeling/parameters.py | 436 | 455| 136 | 27029 | 87235 | 
| 83 | 23 astropy/units/function/core.py | 203 | 257| 456 | 27485 | 87235 | 
| 84 | 23 astropy/units/quantity_helper.py | 228 | 272| 357 | 27842 | 87235 | 
| 85 | 23 astropy/units/quantity_helper.py | 369 | 431| 792 | 28634 | 87235 | 
| 86 | 23 astropy/units/quantity_helper.py | 43 | 82| 346 | 28980 | 87235 | 
| 87 | **23 astropy/units/core.py** | 452 | 485| 171 | 29151 | 87235 | 
| 88 | 24 astropy/units/quantity.py | 720 | 764| 276 | 29427 | 100842 | 
| 89 | 25 astropy/table/column.py | 594 | 614| 150 | 29577 | 112229 | 
| 90 | 25 astropy/coordinates/attributes.py | 216 | 258| 321 | 29898 | 112229 | 
| 91 | 26 astropy/units/cgs.py | 1 | 136| 850 | 30748 | 113123 | 
| 92 | 27 astropy/io/votable/tree.py | 1376 | 1398| 207 | 30955 | 140001 | 
| 93 | 27 astropy/units/quantity.py | 1 | 55| 370 | 31325 | 140001 | 
| 94 | 27 astropy/units/quantity.py | 342 | 388| 477 | 31802 | 140001 | 
| 95 | **27 astropy/units/core.py** | 2206 | 2274| 559 | 32361 | 140001 | 
| 96 | 27 astropy/units/format/utils.py | 152 | 188| 235 | 32596 | 140001 | 
| 97 | 27 astropy/units/quantity.py | 834 | 941| 803 | 33399 | 140001 | 
| 98 | **27 astropy/units/core.py** | 337 | 389| 582 | 33981 | 140001 | 
| 99 | 27 astropy/units/quantity.py | 991 | 1064| 628 | 34609 | 140001 | 
| 100 | 27 astropy/units/function/core.py | 130 | 168| 293 | 34902 | 140001 | 
| 101 | 27 astropy/units/function/core.py | 28 | 128| 830 | 35732 | 140001 | 
| 102 | 27 astropy/io/votable/tree.py | 769 | 806| 330 | 36062 | 140001 | 
| 103 | 27 astropy/units/format/generic.py | 212 | 235| 176 | 36238 | 140001 | 
| 104 | **27 astropy/units/core.py** | 64 | 103| 285 | 36523 | 140001 | 
| 105 | 27 astropy/units/quantity_helper.py | 543 | 584| 419 | 36942 | 140001 | 
| 106 | 27 astropy/units/quantity_helper.py | 275 | 285| 113 | 37055 | 140001 | 
| 107 | 27 astropy/units/format/vounit.py | 7 | 26| 139 | 37194 | 140001 | 
| 108 | 27 astropy/units/format/fits.py | 139 | 158| 145 | 37339 | 140001 | 
| 109 | 27 astropy/units/quantity.py | 270 | 340| 669 | 38008 | 140001 | 
| 110 | 28 astropy/units/deprecated.py | 40 | 69| 199 | 38207 | 140515 | 
| 111 | 28 astropy/units/format/ogip.py | 375 | 400| 200 | 38407 | 140515 | 
| 112 | 29 astropy/stats/bls/core.py | 1 | 22| 134 | 38541 | 147273 | 
| 113 | 29 astropy/coordinates/attributes.py | 338 | 373| 236 | 38777 | 147273 | 
| 114 | 30 astropy/units/format/cds.py | 303 | 337| 240 | 39017 | 149692 | 
| 115 | 30 astropy/units/format/cds.py | 1 | 78| 365 | 39382 | 149692 | 
| 116 | **30 astropy/units/core.py** | 1123 | 1181| 525 | 39907 | 149692 | 
| 117 | 30 astropy/units/format/cds.py | 280 | 301| 146 | 40053 | 149692 | 
| 118 | **30 astropy/units/core.py** | 1439 | 1527| 697 | 40750 | 149692 | 
| 119 | 31 astropy/units/decorators.py | 82 | 153| 472 | 41222 | 151357 | 
| 120 | 32 astropy/modeling/core.py | 1491 | 1565| 662 | 41884 | 179335 | 
| 121 | 32 astropy/units/function/core.py | 334 | 360| 272 | 42156 | 179335 | 
| 122 | 33 astropy/io/votable/ucd.py | 163 | 195| 216 | 42372 | 180651 | 
| 123 | 33 astropy/units/format/generic.py | 191 | 210| 181 | 42553 | 180651 | 
| 124 | 33 astropy/units/physical.py | 1 | 45| 193 | 42746 | 180651 | 
| 125 | 33 astropy/units/quantity.py | 802 | 832| 202 | 42948 | 180651 | 
| 126 | **33 astropy/units/core.py** | 392 | 425| 286 | 43234 | 180651 | 
| 127 | 34 astropy/units/function/logarithmic.py | 87 | 104| 206 | 43440 | 183609 | 
| 128 | 35 astropy/units/__init__.py | 13 | 40| 140 | 43580 | 183836 | 
| 129 | 35 astropy/units/quantity.py | 592 | 611| 232 | 43812 | 183836 | 
| 130 | **35 astropy/units/core.py** | 2277 | 2317| 256 | 44068 | 183836 | 
| 131 | 36 astropy/coordinates/angles.py | 242 | 338| 797 | 44865 | 189720 | 
| 132 | 36 astropy/units/format/cds.py | 339 | 372| 242 | 45107 | 189720 | 
| 133 | **36 astropy/units/core.py** | 106 | 152| 332 | 45439 | 189720 | 
| 134 | 36 astropy/visualization/units.py | 79 | 102| 157 | 45596 | 189720 | 
| 135 | 37 astropy/units/function/units.py | 1 | 47| 277 | 45873 | 190093 | 
| 136 | 37 astropy/units/format/generic.py | 296 | 407| 692 | 46565 | 190093 | 
| 137 | 37 astropy/units/format/ogip.py | 52 | 110| 588 | 47153 | 190093 | 
| 138 | 37 astropy/units/format/fits.py | 110 | 137| 228 | 47381 | 190093 | 
| 139 | 37 astropy/units/format/generic.py | 55 | 103| 270 | 47651 | 190093 | 
| 140 | 37 astropy/units/quantity.py | 671 | 718| 425 | 48076 | 190093 | 
| 141 | **37 astropy/units/core.py** | 2043 | 2082| 337 | 48413 | 190093 | 
| 142 | 37 astropy/units/function/core.py | 362 | 394| 263 | 48676 | 190093 | 
| 143 | 38 astropy/units/format/unicode_format.py | 1 | 45| 199 | 48875 | 190491 | 
| 144 | 38 astropy/units/format/utils.py | 78 | 110| 235 | 49110 | 190491 | 
| 145 | 38 astropy/modeling/core.py | 1406 | 1428| 203 | 49313 | 190491 | 
| 146 | 39 astropy/coordinates/distances.py | 8 | 93| 777 | 50090 | 192293 | 
| 147 | **39 astropy/units/core.py** | 1949 | 2041| 613 | 50703 | 192293 | 
| 148 | 39 astropy/units/quantity.py | 419 | 470| 538 | 51241 | 192293 | 
| 149 | 39 astropy/visualization/units.py | 53 | 77| 196 | 51437 | 192293 | 
| 150 | 39 astropy/coordinates/angles.py | 83 | 109| 231 | 51668 | 192293 | 
| 151 | 39 astropy/units/quantity.py | 766 | 784| 194 | 51862 | 192293 | 
| 152 | 40 astropy/units/cds.py | 173 | 189| 140 | 52002 | 194430 | 
| 153 | **40 astropy/units/core.py** | 1584 | 1602| 150 | 52152 | 194430 | 
| 154 | 40 astropy/units/quantity.py | 196 | 268| 741 | 52893 | 194430 | 
| 155 | 41 astropy/io/misc/asdf/tags/time/time.py | 4 | 32| 188 | 53081 | 195472 | 


### Hint

```
`x is None` works fine. Is there a reason why `==` is needed here?
`x is None` would indeed be preferred, but `==` should never fail, so this is still a bug.
```

## Patch

```diff
diff --git a/astropy/units/core.py b/astropy/units/core.py
--- a/astropy/units/core.py
+++ b/astropy/units/core.py
@@ -728,7 +728,7 @@ def __eq__(self, other):
         try:
             other = Unit(other, parse_strict='silent')
         except (ValueError, UnitsError, TypeError):
-            return False
+            return NotImplemented
 
         # Other is Unit-like, but the test below requires it is a UnitBase
         # instance; if it is not, give up (so that other can try).
@@ -1710,8 +1710,12 @@ def _unrecognized_operator(self, *args, **kwargs):
         _unrecognized_operator
 
     def __eq__(self, other):
-        other = Unit(other, parse_strict='silent')
-        return isinstance(other, UnrecognizedUnit) and self.name == other.name
+        try:
+            other = Unit(other, parse_strict='silent')
+        except (ValueError, UnitsError, TypeError):
+            return NotImplemented
+
+        return isinstance(other, type(self)) and self.name == other.name
 
     def __ne__(self, other):
         return not (self == other)

```

## Test Patch

```diff
diff --git a/astropy/units/tests/test_units.py b/astropy/units/tests/test_units.py
--- a/astropy/units/tests/test_units.py
+++ b/astropy/units/tests/test_units.py
@@ -185,6 +185,13 @@ def test_unknown_unit3():
     assert unit != unit3
     assert not unit.is_equivalent(unit3)
 
+    # Also test basic (in)equalities.
+    assert unit == "FOO"
+    assert unit != u.m
+    # next two from gh-7603.
+    assert unit != None  # noqa
+    assert unit not in (None, u.m)
+
     with pytest.raises(ValueError):
         unit._get_converter(unit3)
 

```


## Code snippets

### 1 - astropy/units/core.py:

Start line: 1675, End line: 1733

```python
class UnrecognizedUnit(IrreducibleUnit):
    """
    A unit that did not parse correctly.  This allows for
    roundtripping it as a string, but no unit operations actually work
    on it.

    Parameters
    ----------
    st : str
        The name of the unit.
    """
    # For UnrecognizedUnits, we want to use "standard" Python
    # pickling, not the special case that is used for
    # IrreducibleUnits.
    __reduce__ = object.__reduce__

    def __repr__(self):
        return "UnrecognizedUnit({0})".format(str(self))

    def __bytes__(self):
        return self.name.encode('ascii', 'replace')

    def __str__(self):
        return self.name

    def to_string(self, format=None):
        return self.name

    def _unrecognized_operator(self, *args, **kwargs):
        raise ValueError(
            "The unit {0!r} is unrecognized, so all arithmetic operations "
            "with it are invalid.".format(self.name))

    __pow__ = __div__ = __rdiv__ = __truediv__ = __rtruediv__ = __mul__ = \
        __rmul__ = __lt__ = __gt__ = __le__ = __ge__ = __neg__ = \
        _unrecognized_operator

    def __eq__(self, other):
        other = Unit(other, parse_strict='silent')
        return isinstance(other, UnrecognizedUnit) and self.name == other.name

    def __ne__(self, other):
        return not (self == other)

    def is_equivalent(self, other, equivalencies=None):
        self._normalize_equivalencies(equivalencies)
        return self == other

    def _get_converter(self, other, equivalencies=None):
        self._normalize_equivalencies(equivalencies)
        raise ValueError(
            "The unit {0!r} is unrecognized.  It can not be converted "
            "to other units.".format(self.name))

    def get_format_name(self, format):
        return self.name

    def is_unity(self):
        return False
```
### 2 - astropy/units/core.py:

Start line: 1396, End line: 1436

```python
class UnitBase(metaclass=InheritDocstrings):

    def find_equivalent_units(self, equivalencies=[], units=None,
                              include_prefix_units=False):
        """
        Return a list of all the units that are the same type as ``self``.

        Parameters
        ----------
        equivalencies : list of equivalence pairs, optional
            A list of equivalence pairs to also list.  See
            :ref:`unit_equivalencies`.
            Any list given, including an empty one, supercedes global defaults
            that may be in effect (as set by `set_enabled_equivalencies`)

        units : set of units to search in, optional
            If not provided, all defined units will be searched for
            equivalencies.  Otherwise, may be a dict, module or
            sequence containing the units to search for equivalencies.

        include_prefix_units : bool, optional
            When `True`, include prefixed units in the result.
            Default is `False`.

        Returns
        -------
        units : list of `UnitBase`
            A list of unit objects that match ``u``.  A subclass of
            `list` (``EquivalentUnitsList``) is returned that
            pretty-prints the list of units when output.
        """
        results = self.compose(
            equivalencies=equivalencies, units=units, max_depth=1,
            include_prefix_units=include_prefix_units)
        results = set(
            x.bases[0] for x in results if len(x.bases) == 1)
        return self.EquivalentUnitsList(results)

    def is_unity(self):
        """
        Returns `True` if the unit is unscaled and dimensionless.
        """
        return False
```
### 3 - astropy/units/core.py:

Start line: 719, End line: 741

```python
class UnitBase(metaclass=InheritDocstrings):

    def __hash__(self):
        # This must match the hash used in CompositeUnit for a unit
        # with only one base and no scale or power.
        return hash((str(self.scale), self.name, str('1')))

    def __eq__(self, other):
        if self is other:
            return True

        try:
            other = Unit(other, parse_strict='silent')
        except (ValueError, UnitsError, TypeError):
            return False

        # Other is Unit-like, but the test below requires it is a UnitBase
        # instance; if it is not, give up (so that other can try).
        if not isinstance(other, UnitBase):
            return NotImplemented

        try:
            return is_effectively_unity(self._to(other))
        except UnitsError:
            return False
```
### 4 - astropy/units/core.py:

Start line: 794, End line: 823

```python
class UnitBase(metaclass=InheritDocstrings):

    def _is_equivalent(self, other, equivalencies=[]):
        """Returns `True` if this unit is equivalent to `other`.
        See `is_equivalent`, except that a proper Unit object should be
        given (i.e., no string) and that the equivalency list should be
        normalized using `_normalize_equivalencies`.
        """
        if isinstance(other, UnrecognizedUnit):
            return False

        if (self._get_physical_type_id() ==
                other._get_physical_type_id()):
            return True
        elif len(equivalencies):
            unit = self.decompose()
            other = other.decompose()
            for a, b, forward, backward in equivalencies:
                if b is None:
                    # after canceling, is what's left convertible
                    # to dimensionless (according to the equivalency)?
                    try:
                        (other/unit).decompose([a])
                        return True
                    except Exception:
                        pass
                else:
                    if(a._is_equivalent(unit) and b._is_equivalent(other) or
                       b._is_equivalent(unit) and a._is_equivalent(other)):
                        return True

        return False
```
### 5 - astropy/units/core.py:

Start line: 763, End line: 792

```python
class UnitBase(metaclass=InheritDocstrings):

    def is_equivalent(self, other, equivalencies=[]):
        """
        Returns `True` if this unit is equivalent to ``other``.

        Parameters
        ----------
        other : unit object or string or tuple
            The unit to convert to. If a tuple of units is specified, this
            method returns true if the unit matches any of those in the tuple.

        equivalencies : list of equivalence pairs, optional
            A list of equivalence pairs to try if the units are not
            directly convertible.  See :ref:`unit_equivalencies`.
            This list is in addition to possible global defaults set by, e.g.,
            `set_enabled_equivalencies`.
            Use `None` to turn off all equivalencies.

        Returns
        -------
        bool
        """
        equivalencies = self._normalize_equivalencies(equivalencies)

        if isinstance(other, tuple):
            return any(self.is_equivalent(u, equivalencies=equivalencies)
                       for u in other)

        other = Unit(other, parse_strict='silent')

        return self._is_equivalent(other, equivalencies)
```
### 6 - astropy/units/core.py:

Start line: 743, End line: 761

```python
class UnitBase(metaclass=InheritDocstrings):

    def __ne__(self, other):
        return not (self == other)

    def __le__(self, other):
        scale = self._to(Unit(other))
        return scale <= 1. or is_effectively_unity(scale)

    def __ge__(self, other):
        scale = self._to(Unit(other))
        return scale >= 1. or is_effectively_unity(scale)

    def __lt__(self, other):
        return not (self >= other)

    def __gt__(self, other):
        return not (self <= other)

    def __neg__(self):
        return self * -1.
```
### 7 - astropy/units/function/core.py:

Start line: 259, End line: 283

```python
class FunctionUnitBase(metaclass=ABCMeta):

    def is_unity(self):
        return False

    def __eq__(self, other):
        return (self.physical_unit == getattr(other, 'physical_unit',
                                              dimensionless_unscaled) and
                self.function_unit == getattr(other, 'function_unit', other))

    def __ne__(self, other):
        return not self.__eq__(other)

    def __mul__(self, other):
        if isinstance(other, (str, UnitBase, FunctionUnitBase)):
            if self.physical_unit == dimensionless_unscaled:
                # If dimensionless, drop back to normal unit and retry.
                return self.function_unit * other
            else:
                raise UnitsError("Cannot multiply a function unit "
                                 "with a physical dimension with any unit.")
        else:
            # Anything not like a unit, try initialising as a function quantity.
            try:
                return self._quantity_class(other, unit=self)
            except Exception:
                return NotImplemented
```
### 8 - astropy/units/function/core.py:

Start line: 170, End line: 201

```python
class FunctionUnitBase(metaclass=ABCMeta):

    def is_equivalent(self, other, equivalencies=[]):
        """
        Returns `True` if this unit is equivalent to ``other``.

        Parameters
        ----------
        other : unit object or string or tuple
            The unit to convert to. If a tuple of units is specified, this
            method returns true if the unit matches any of those in the tuple.

        equivalencies : list of equivalence pairs, optional
            A list of equivalence pairs to try if the units are not
            directly convertible.  See :ref:`unit_equivalencies`.
            This list is in addition to the built-in equivalencies between the
            function unit and the physical one, as well as possible global
            defaults set by, e.g., `~astropy.units.set_enabled_equivalencies`.
            Use `None` to turn off any global equivalencies.

        Returns
        -------
        bool
        """
        if isinstance(other, tuple):
            return any(self.is_equivalent(u, equivalencies=equivalencies)
                       for u in other)

        other_physical_unit = getattr(other, 'physical_unit', (
            dimensionless_unscaled if self.function_unit.is_equivalent(other)
            else other))

        return self.physical_unit.is_equivalent(other_physical_unit,
                                                equivalencies)
```
### 9 - astropy/units/core.py:

Start line: 1183, End line: 1198

```python
class UnitBase(metaclass=InheritDocstrings):

    def compose(self, equivalencies=[], units=None, max_depth=2,
                include_prefix_units=None):
        # ... other code

        def has_bases_in_common_with_equiv(unit, other):
            if has_bases_in_common(unit, other):
                return True
            for funit, tunit, a, b in equivalencies:
                if tunit is not None:
                    if unit._is_equivalent(funit):
                        if has_bases_in_common(tunit.decompose(), other):
                            return True
                    elif unit._is_equivalent(tunit):
                        if has_bases_in_common(funit.decompose(), other):
                            return True
                else:
                    if unit._is_equivalent(funit):
                        if has_bases_in_common(dimensionless_unscaled, other):
                            return True
            return False
        # ... other code
```
### 10 - astropy/units/function/core.py:

Start line: 572, End line: 592

```python
class FunctionQuantity(Quantity):

    def _comparison(self, other, comparison_func):
        """Do a comparison between self and other, raising UnitsError when
        other cannot be converted to self because it has different physical
        unit, and returning NotImplemented when there are other errors."""
        try:
            # will raise a UnitsError if physical units not equivalent
            other_in_own_unit = self._to_own_unit(other, check_precision=False)
        except UnitsError as exc:
            if self.unit.physical_unit != dimensionless_unscaled:
                raise exc

            try:
                other_in_own_unit = self._function_view._to_own_unit(
                    other, check_precision=False)
            except Exception:
                raise exc

        except Exception:
            return NotImplemented

        return comparison_func(other_in_own_unit)
```
### 12 - astropy/units/core.py:

Start line: 874, End line: 902

```python
class UnitBase(metaclass=InheritDocstrings):

    def _get_converter(self, other, equivalencies=[]):
        other = Unit(other)

        # First see if it is just a scaling.
        try:
            scale = self._to(other)
        except UnitsError:
            pass
        else:
            return lambda val: scale * _condition_arg(val)

        # if that doesn't work, maybe we can do it with equivalencies?
        try:
            return self._apply_equivalencies(
                self, other, self._normalize_equivalencies(equivalencies))
        except UnitsError as exc:
            # Last hope: maybe other knows how to do it?
            # We assume the equivalencies have the unit itself as first item.
            # TODO: maybe better for other to have a `_back_converter` method?
            if hasattr(other, 'equivalencies'):
                for funit, tunit, a, b in other.equivalencies:
                    if other is funit:
                        try:
                            return lambda v: b(self._get_converter(
                                tunit, equivalencies=equivalencies)(v))
                        except Exception:
                            pass

            raise exc
```
### 14 - astropy/units/core.py:

Start line: 1828, End line: 1946

```python
class Unit(NamedUnit, metaclass=_UnitMetaClass):
    """
    The main unit class.

    There are a number of different ways to construct a Unit, but
    always returns a `UnitBase` instance.  If the arguments refer to
    an already-existing unit, that existing unit instance is returned,
    rather than a new one.

    - From a string::

        Unit(s, format=None, parse_strict='silent')

      Construct from a string representing a (possibly compound) unit.

      The optional `format` keyword argument specifies the format the
      string is in, by default ``"generic"``.  For a description of
      the available formats, see `astropy.units.format`.

      The optional ``parse_strict`` keyword controls what happens when an
      unrecognized unit string is passed in.  It may be one of the following:

         - ``'raise'``: (default) raise a ValueError exception.

         - ``'warn'``: emit a Warning, and return an
           `UnrecognizedUnit` instance.

         - ``'silent'``: return an `UnrecognizedUnit` instance.

    - From a number::

        Unit(number)

      Creates a dimensionless unit.

    - From a `UnitBase` instance::

        Unit(unit)

      Returns the given unit unchanged.

    - From `None`::

        Unit()

      Returns the null unit.

    - The last form, which creates a new `Unit` is described in detail
      below.

    Parameters
    ----------
    st : str or list of str
        The name of the unit.  If a list, the first element is the
        canonical (short) name, and the rest of the elements are
        aliases.

    represents : UnitBase instance
        The unit that this named unit represents.

    doc : str, optional
        A docstring describing the unit.

    format : dict, optional
        A mapping to format-specific representations of this unit.
        For example, for the ``Ohm`` unit, it might be nice to have it
        displayed as ``\\Omega`` by the ``latex`` formatter.  In that
        case, `format` argument should be set to::

            {'latex': r'\\Omega'}

    namespace : dictionary, optional
        When provided, inject the unit (and all of its aliases) into
        the given namespace.

    Raises
    ------
    ValueError
        If any of the given unit names are already in the registry.

    ValueError
        If any of the given unit names are not valid Python tokens.
    """

    def __init__(self, st, represents=None, doc=None,
                 format=None, namespace=None):

        represents = Unit(represents)
        self._represents = represents

        NamedUnit.__init__(self, st, namespace=namespace, doc=doc,
                           format=format)

    @property
    def represents(self):
        """The unit that this named unit represents."""
        return self._represents

    def decompose(self, bases=set()):
        return self._represents.decompose(bases=bases)

    def is_unity(self):
        return self._represents.is_unity()

    def __hash__(self):
        return hash(self.name) + hash(self._represents)

    @classmethod
    def _from_physical_type_id(cls, physical_type_id):
        # get string bases and powers from the ID tuple
        bases = [cls(base) for base, _ in physical_type_id]
        powers = [power for _, power in physical_type_id]

        if len(physical_type_id) == 1 and powers[0] == 1:
            unit = bases[0]
        else:
            unit = CompositeUnit(1, bases, powers)

        return unit
```
### 17 - astropy/units/core.py:

Start line: 1330, End line: 1362

```python
class UnitBase(metaclass=InheritDocstrings):

    def _get_units_with_same_physical_type(self, equivalencies=[]):
        """
        Return a list of registered units with the same physical type
        as this unit.

        This function is used by Quantity to add its built-in
        conversions to equivalent units.

        This is a private method, since end users should be encouraged
        to use the more powerful `compose` and `find_equivalent_units`
        methods (which use this under the hood).

        Parameters
        ----------
        equivalencies : list of equivalence pairs, optional
            A list of equivalence pairs to also pull options from.
            See :ref:`unit_equivalencies`.  It must already be
            normalized using `_normalize_equivalencies`.
        """
        unit_registry = get_current_unit_registry()
        units = set(unit_registry.get_units_with_physical_type(self))
        for funit, tunit, a, b in equivalencies:
            if tunit is not None:
                if self.is_equivalent(funit) and tunit not in units:
                    units.update(
                        unit_registry.get_units_with_physical_type(tunit))
                if self._is_equivalent(tunit) and funit not in units:
                    units.update(
                        unit_registry.get_units_with_physical_type(funit))
            else:
                if self.is_equivalent(funit):
                    units.add(dimensionless_unscaled)
        return units
```
### 18 - astropy/units/core.py:

Start line: 1, End line: 32

```python
# -*- coding: utf-8 -*-


import inspect
import operator
import textwrap
import warnings

import numpy as np

from ..utils.decorators import lazyproperty
from ..utils.exceptions import AstropyWarning
from ..utils.misc import isiterable, InheritDocstrings
from .utils import (is_effectively_unity, sanitize_scale, validate_power,
                    resolve_fractions)
from . import format as unit_format


__all__ = [
    'UnitsError', 'UnitsWarning', 'UnitConversionError', 'UnitTypeError',
    'UnitBase', 'NamedUnit', 'IrreducibleUnit', 'Unit', 'CompositeUnit',
    'PrefixUnit', 'UnrecognizedUnit', 'def_unit', 'get_current_unit_registry',
    'set_enabled_units', 'add_enabled_units',
    'set_enabled_equivalencies', 'add_enabled_equivalencies',
    'dimensionless_unscaled', 'one']

UNITY = 1.0
```
### 20 - astropy/units/core.py:

Start line: 614, End line: 639

```python
class UnitBase(metaclass=InheritDocstrings):

    @staticmethod
    def _normalize_equivalencies(equivalencies):
        """
        Normalizes equivalencies, ensuring each is a 4-tuple of the form::

        (from_unit, to_unit, forward_func, backward_func)

        Parameters
        ----------
        equivalencies : list of equivalency pairs, or `None`

        Returns
        -------
        A normalized list, including possible global defaults set by, e.g.,
        `set_enabled_equivalencies`, except when `equivalencies`=`None`,
        in which case the returned list is always empty.

        Raises
        ------
        ValueError if an equivalency cannot be interpreted
        """
        normalized = _normalize_equivalencies(equivalencies)
        if equivalencies is not None:
            normalized += get_current_unit_registry().equivalencies

        return normalized
```
### 23 - astropy/units/core.py:

Start line: 701, End line: 717

```python
class UnitBase(metaclass=InheritDocstrings):

    def __rmul__(self, m):
        if isinstance(m, (bytes, str)):
            return Unit(m) * self

        # Cannot handle this as Unit.  Here, m cannot be a Quantity,
        # so we make it into one, fasttracking when it does not have a unit
        # for the common case of <array> * <unit>.
        try:
            from .quantity import Quantity
            if hasattr(m, 'unit'):
                result = Quantity(m)
                result *= self
                return result
            else:
                return Quantity(m, self)
        except TypeError:
            return NotImplemented
```
### 26 - astropy/units/core.py:

Start line: 211, End line: 243

```python
class _UnitRegistry:

    def get_units_with_physical_type(self, unit):
        """
        Get all units in the registry with the same physical type as
        the given unit.

        Parameters
        ----------
        unit : UnitBase instance
        """
        return self._by_physical_type.get(unit._get_physical_type_id(), set())

    @property
    def equivalencies(self):
        return list(self._equivalencies)

    def set_enabled_equivalencies(self, equivalencies):
        """
        Sets the equivalencies enabled in the unit registry.

        These equivalencies are used if no explicit equivalencies are given,
        both in unit conversion and in finding equivalent units.

        This is meant in particular for allowing angles to be dimensionless.
        Use with care.

        Parameters
        ----------
        equivalencies : list of equivalent pairs
            E.g., as returned by
            `~astropy.units.equivalencies.dimensionless_angles`.
        """
        self._reset_equivalencies()
        return self.add_enabled_equivalencies(equivalencies)
```
### 27 - astropy/units/core.py:

Start line: 661, End line: 677

```python
class UnitBase(metaclass=InheritDocstrings):

    def __rdiv__(self, m):
        if isinstance(m, (bytes, str)):
            return Unit(m) / self

        try:
            # Cannot handle this as Unit.  Here, m cannot be a Quantity,
            # so we make it into one, fasttracking when it does not have a
            # unit, for the common case of <array> / <unit>.
            from .quantity import Quantity
            if hasattr(m, 'unit'):
                result = Quantity(m)
                result /= self
                return result
            else:
                return Quantity(m, self**(-1))
        except TypeError:
            return NotImplemented
```
### 30 - astropy/units/core.py:

Start line: 679, End line: 699

```python
class UnitBase(metaclass=InheritDocstrings):

    __truediv__ = __div__

    __rtruediv__ = __rdiv__

    def __mul__(self, m):
        if isinstance(m, (bytes, str)):
            m = Unit(m)

        if isinstance(m, UnitBase):
            if m.is_unity():
                return self
            elif self.is_unity():
                return m
            return CompositeUnit(1, [self, m], [1, 1], _error_check=False)

        # Cannot handle this as Unit, re-try as Quantity.
        try:
            from .quantity import Quantity
            return Quantity(1, self) * m
        except TypeError:
            return NotImplemented
```
### 31 - astropy/units/core.py:

Start line: 1364, End line: 1394

```python
class UnitBase(metaclass=InheritDocstrings):

    class EquivalentUnitsList(list):
        """
        A class to handle pretty-printing the result of
        `find_equivalent_units`.
        """

        def __repr__(self):
            if len(self) == 0:
                return "[]"
            else:
                lines = []
                for u in self:
                    irred = u.decompose().to_string()
                    if irred == u.name:
                        irred = "irreducible"
                    lines.append((u.name, irred, ', '.join(u.aliases)))

                lines.sort()
                lines.insert(0, ('Primary name', 'Unit definition', 'Aliases'))
                widths = [0, 0, 0]
                for line in lines:
                    for i, col in enumerate(line):
                        widths[i] = max(widths[i], len(col))

                f = "  {{0:<{0}s}} | {{1:<{1}s}} | {{2:<{2}s}}".format(*widths)
                lines = [f.format(*line) for line in lines]
                lines = (lines[0:1] +
                         ['['] +
                         ['{0} ,'.format(x) for x in lines[1:]] +
                         [']'])
                return '\n'.join(lines)
```
### 32 - astropy/units/core.py:

Start line: 1736, End line: 1825

```python
class _UnitMetaClass(InheritDocstrings):
    """
    This metaclass exists because the Unit constructor should
    sometimes return instances that already exist.  This "overrides"
    the constructor before the new instance is actually created, so we
    can return an existing one.
    """

    def __call__(self, s, represents=None, format=None, namespace=None,
                 doc=None, parse_strict='raise'):

        # Short-circuit if we're already a unit
        if hasattr(s, '_get_physical_type_id'):
            return s

        # turn possible Quantity input for s or represents into a Unit
        from .quantity import Quantity

        if isinstance(represents, Quantity):
            if is_effectively_unity(represents.value):
                represents = represents.unit
            else:
                # cannot use _error_check=False: scale may be effectively unity
                represents = CompositeUnit(represents.value *
                                           represents.unit.scale,
                                           bases=represents.unit.bases,
                                           powers=represents.unit.powers)

        if isinstance(s, Quantity):
            if is_effectively_unity(s.value):
                s = s.unit
            else:
                s = CompositeUnit(s.value * s.unit.scale,
                                  bases=s.unit.bases,
                                  powers=s.unit.powers)

        # now decide what we really need to do; define derived Unit?
        if isinstance(represents, UnitBase):
            # This has the effect of calling the real __new__ and
            # __init__ on the Unit class.
            return super().__call__(
                s, represents, format=format, namespace=namespace, doc=doc)

        # or interpret a Quantity (now became unit), string or number?
        if isinstance(s, UnitBase):
            return s

        elif isinstance(s, (bytes, str)):
            if len(s.strip()) == 0:
                # Return the NULL unit
                return dimensionless_unscaled

            if format is None:
                format = unit_format.Generic

            f = unit_format.get_format(format)
            if isinstance(s, bytes):
                s = s.decode('ascii')

            try:
                return f.parse(s)
            except Exception as e:
                if parse_strict == 'silent':
                    pass
                else:
                    # Deliberately not issubclass here. Subclasses
                    # should use their name.
                    if f is not unit_format.Generic:
                        format_clause = f.name + ' '
                    else:
                        format_clause = ''
                    msg = ("'{0}' did not parse as {1}unit: {2}"
                           .format(s, format_clause, str(e)))
                    if parse_strict == 'raise':
                        raise ValueError(msg)
                    elif parse_strict == 'warn':
                        warnings.warn(msg, UnitsWarning)
                    else:
                        raise ValueError("'parse_strict' must be 'warn', "
                                         "'raise' or 'silent'")
                return UnrecognizedUnit(s)

        elif isinstance(s, (int, float, np.floating, np.integer)):
            return CompositeUnit(s, [], [])

        elif s is None:
            raise TypeError("None is not a valid Unit")

        else:
            raise TypeError("{0} can not be converted to a Unit".format(s))
```
### 36 - astropy/units/core.py:

Start line: 1646, End line: 1672

```python
class IrreducibleUnit(NamedUnit):

    @property
    def represents(self):
        """The unit that this named unit represents.

        For an irreducible unit, that is always itself.
        """
        return self

    def decompose(self, bases=set()):
        if len(bases) and self not in bases:
            for base in bases:
                try:
                    scale = self._to(base)
                except UnitsError:
                    pass
                else:
                    if is_effectively_unity(scale):
                        return base
                    else:
                        return CompositeUnit(scale, [base], [1],
                                             _error_check=False)

            raise UnitConversionError(
                "Unit {0} can not be decomposed into the requested "
                "bases".format(self))

        return self
```
### 37 - astropy/units/core.py:

Start line: 1225, End line: 1251

```python
class UnitBase(metaclass=InheritDocstrings):

    def compose(self, equivalencies=[], units=None, max_depth=2,
                include_prefix_units=None):
        # ... other code

        def sort_results(results):
            if not len(results):
                return []

            # Sort the results so the simplest ones appear first.
            # Simplest is defined as "the minimum sum of absolute
            # powers" (i.e. the fewest bases), and preference should
            # be given to results where the sum of powers is positive
            # and the scale is exactly equal to 1.0
            results = list(results)
            results.sort(key=lambda x: np.abs(x.scale))
            results.sort(key=lambda x: np.sum(np.abs(x.powers)))
            results.sort(key=lambda x: np.sum(x.powers) < 0.0)
            results.sort(key=lambda x: not is_effectively_unity(x.scale))

            last_result = results[0]
            filtered = [last_result]
            for result in results[1:]:
                if str(result) != str(last_result):
                    filtered.append(result)
                last_result = result

            return filtered

        return sort_results(self._compose(
            equivalencies=equivalencies, namespace=units,
            max_depth=max_depth, depth=0, cached_results={}))
```
### 40 - astropy/units/core.py:

Start line: 1000, End line: 1121

```python
class UnitBase(metaclass=InheritDocstrings):

    def _compose(self, equivalencies=[], namespace=[], max_depth=2, depth=0,
                 cached_results=None):
        def is_final_result(unit):
            # Returns True if this result contains only the expected
            # units
            for base in unit.bases:
                if base not in namespace:
                    return False
            return True

        unit = self.decompose()
        key = hash(unit)

        cached = cached_results.get(key)
        if cached is not None:
            if isinstance(cached, Exception):
                raise cached
            return cached

        # Prevent too many levels of recursion
        # And special case for dimensionless unit
        if depth >= max_depth:
            cached_results[key] = [unit]
            return [unit]

        # Make a list including all of the equivalent units
        units = [unit]
        for funit, tunit, a, b in equivalencies:
            if tunit is not None:
                if self._is_equivalent(funit):
                    scale = funit.decompose().scale / unit.scale
                    units.append(Unit(a(1.0 / scale) * tunit).decompose())
                elif self._is_equivalent(tunit):
                    scale = tunit.decompose().scale / unit.scale
                    units.append(Unit(b(1.0 / scale) * funit).decompose())
            else:
                if self._is_equivalent(funit):
                    units.append(Unit(unit.scale))

        # Store partial results
        partial_results = []
        # Store final results that reduce to a single unit or pair of
        # units
        if len(unit.bases) == 0:
            final_results = [set([unit]), set()]
        else:
            final_results = [set(), set()]

        for tunit in namespace:
            tunit_decomposed = tunit.decompose()
            for u in units:
                # If the unit is a base unit, look for an exact match
                # to one of the bases of the target unit.  If found,
                # factor by the same power as the target unit's base.
                # This allows us to factor out fractional powers
                # without needing to do an exhaustive search.
                if len(tunit_decomposed.bases) == 1:
                    for base, power in zip(u.bases, u.powers):
                        if tunit_decomposed._is_equivalent(base):
                            tunit = tunit ** power
                            tunit_decomposed = tunit_decomposed ** power
                            break

                composed = (u / tunit_decomposed).decompose()
                factored = composed * tunit
                len_bases = len(composed.bases)
                if is_final_result(factored) and len_bases <= 1:
                    final_results[len_bases].add(factored)
                else:
                    partial_results.append(
                        (len_bases, composed, tunit))

        # Do we have any minimal results?
        for final_result in final_results:
            if len(final_result):
                results = final_results[0].union(final_results[1])
                cached_results[key] = results
                return results

        partial_results.sort(key=operator.itemgetter(0))

        # ...we have to recurse and try to further compose
        results = []
        for len_bases, composed, tunit in partial_results:
            try:
                composed_list = composed._compose(
                    equivalencies=equivalencies,
                    namespace=namespace,
                    max_depth=max_depth, depth=depth + 1,
                    cached_results=cached_results)
            except UnitsError:
                composed_list = []
            for subcomposed in composed_list:
                results.append(
                    (len(subcomposed.bases), subcomposed, tunit))

        if len(results):
            results.sort(key=operator.itemgetter(0))

            min_length = results[0][0]
            subresults = set()
            for len_bases, composed, tunit in results:
                if len_bases > min_length:
                    break
                else:
                    factored = composed * tunit
                    if is_final_result(factored):
                        subresults.add(factored)

            if len(subresults):
                cached_results[key] = subresults
                return subresults

        if not is_final_result(self):
            result = UnitsError(
                "Cannot represent unit {0} in terms of the given "
                "units".format(self))
            cached_results[key] = result
            raise result

        cached_results[key] = [self]
        return [self]
```
### 42 - astropy/units/core.py:

Start line: 937, End line: 971

```python
class UnitBase(metaclass=InheritDocstrings):

    def to(self, other, value=UNITY, equivalencies=[]):
        """
        Return the converted values in the specified unit.

        Parameters
        ----------
        other : unit object or string
            The unit to convert to.

        value : scalar int or float, or sequence convertible to array, optional
            Value(s) in the current unit to be converted to the
            specified unit.  If not provided, defaults to 1.0

        equivalencies : list of equivalence pairs, optional
            A list of equivalence pairs to try if the units are not
            directly convertible.  See :ref:`unit_equivalencies`.
            This list is in addition to possible global defaults set by, e.g.,
            `set_enabled_equivalencies`.
            Use `None` to turn off all equivalencies.

        Returns
        -------
        values : scalar or array
            Converted value(s). Input value sequences are returned as
            numpy arrays.

        Raises
        ------
        UnitsError
            If units are inconsistent
        """
        if other is self and value is UNITY:
            return UNITY
        else:
            return self._get_converter(other, equivalencies=equivalencies)(value)
```
### 47 - astropy/units/core.py:

Start line: 1200, End line: 1223

```python
class UnitBase(metaclass=InheritDocstrings):

    def compose(self, equivalencies=[], units=None, max_depth=2,
                include_prefix_units=None):
        # ... other code

        def filter_units(units):
            filtered_namespace = set()
            for tunit in units:
                if (isinstance(tunit, UnitBase) and
                    (include_prefix_units or
                     not isinstance(tunit, PrefixUnit)) and
                    has_bases_in_common_with_equiv(
                        decomposed, tunit.decompose())):
                    filtered_namespace.add(tunit)
            return filtered_namespace

        decomposed = self.decompose()

        if units is None:
            units = filter_units(self._get_units_with_same_physical_type(
                equivalencies=equivalencies))
            if len(units) == 0:
                units = get_current_unit_registry().non_prefix_units
        elif isinstance(units, dict):
            units = set(filter_units(units.values()))
        elif inspect.ismodule(units):
            units = filter_units(vars(units).values())
        else:
            units = filter_units(_flatten_units_collection(units))
        # ... other code
```
### 48 - astropy/units/core.py:

Start line: 973, End line: 998

```python
class UnitBase(metaclass=InheritDocstrings):

    def in_units(self, other, value=1.0, equivalencies=[]):
        """
        Alias for `to` for backward compatibility with pynbody.
        """
        return self.to(
            other, value=value, equivalencies=equivalencies)

    def decompose(self, bases=set()):
        """
        Return a unit object composed of only irreducible units.

        Parameters
        ----------
        bases : sequence of UnitBase, optional
            The bases to decompose into.  When not provided,
            decomposes down to any irreducible units.  When provided,
            the decomposed result will only contain the given units.
            This will raises a `UnitsError` if it's not possible
            to do so.

        Returns
        -------
        unit : CompositeUnit object
            New object containing only irreducible unit objects.
        """
        raise NotImplementedError()
```
### 51 - astropy/units/core.py:

Start line: 641, End line: 659

```python
class UnitBase(metaclass=InheritDocstrings):

    def __pow__(self, p):
        p = validate_power(p)
        return CompositeUnit(1, [self], [p], _error_check=False)

    def __div__(self, m):
        if isinstance(m, (bytes, str)):
            m = Unit(m)

        if isinstance(m, UnitBase):
            if m.is_unity():
                return self
            return CompositeUnit(1, [self, m], [1, -1], _error_check=False)

        try:
            # Cannot handle this as Unit, re-try as Quantity
            from .quantity import Quantity
            return Quantity(1, self) / m
        except TypeError:
            return NotImplemented
```
### 54 - astropy/units/core.py:

Start line: 1298, End line: 1328

```python
class UnitBase(metaclass=InheritDocstrings):

    @lazyproperty
    def si(self):
        """
        Returns a copy of the current `Unit` instance in SI units.
        """

        from . import si
        return self.to_system(si)[0]

    @lazyproperty
    def cgs(self):
        """
        Returns a copy of the current `Unit` instance with CGS units.
        """
        from . import cgs
        return self.to_system(cgs)[0]

    @property
    def physical_type(self):
        """
        Return the physical type on the unit.

        Examples
        --------
        >>> from astropy import units as u
        >>> print(u.m.physical_type)
        length

        """
        from . import physical
        return physical.get_physical_type(self)
```
### 55 - astropy/units/core.py:

Start line: 904, End line: 935

```python
class UnitBase(metaclass=InheritDocstrings):

    def _to(self, other):
        """
        Returns the scale to the specified unit.

        See `to`, except that a Unit object should be given (i.e., no
        string), and that all defaults are used, i.e., no
        equivalencies and value=1.
        """
        # There are many cases where we just want to ensure a Quantity is
        # of a particular unit, without checking whether it's already in
        # a particular unit.  If we're being asked to convert from a unit
        # to itself, we can short-circuit all of this.
        if self is other:
            return 1.0

        # Don't presume decomposition is possible; e.g.,
        # conversion to function units is through equivalencies.
        if isinstance(other, UnitBase):
            self_decomposed = self.decompose()
            other_decomposed = other.decompose()

            # Check quickly whether equivalent.  This is faster than
            # `is_equivalent`, because it doesn't generate the entire
            # physical type list of both units.  In other words it "fails
            # fast".
            if(self_decomposed.powers == other_decomposed.powers and
               all(self_base is other_base for (self_base, other_base)
                   in zip(self_decomposed.bases, other_decomposed.bases))):
                return self_decomposed.scale / other_decomposed.scale

        raise UnitConversionError(
            "'{0!r}' is not a scaled version of '{1!r}'".format(self, other))
```
### 59 - astropy/units/core.py:

Start line: 531, End line: 543

```python
class UnitBase(metaclass=InheritDocstrings):

    def _get_physical_type_id(self):
        """
        Returns an identifier that uniquely identifies the physical
        type of this unit.  It is comprised of the bases and powers of
        this unit, without the scale.  Since it is hashable, it is
        useful as a dictionary key.
        """
        unit = self.decompose()
        r = zip([x.name for x in unit.bases], unit.powers)
        # bases and powers are already sorted in a unique way
        # r.sort()
        r = tuple(r)
        return r
```
### 61 - astropy/units/core.py:

Start line: 2084, End line: 2111

```python
class CompositeUnit(UnitBase):

    def __copy__(self):
        """
        For compatibility with python copy module.
        """
        return CompositeUnit(self._scale, self._bases[:], self._powers[:])

    def decompose(self, bases=set()):
        if len(bases) == 0 and self._decomposed_cache is not None:
            return self._decomposed_cache

        for base in self.bases:
            if (not isinstance(base, IrreducibleUnit) or
                    (len(bases) and base not in bases)):
                break
        else:
            if len(bases) == 0:
                self._decomposed_cache = self
            return self

        x = CompositeUnit(self.scale, self.bases, self.powers, decompose=True,
                          decompose_bases=bases)
        if len(bases) == 0:
            self._decomposed_cache = x
        return x

    def is_unity(self):
        unit = self.decompose()
        return len(unit.bases) == 0 and unit.scale == 1.0
```
### 62 - astropy/units/core.py:

Start line: 285, End line: 334

```python
def set_enabled_units(units):
    """
    Sets the units enabled in the unit registry.

    These units are searched when using
    `UnitBase.find_equivalent_units`, for example.

    This may be used either permanently, or as a context manager using
    the ``with`` statement (see example below).

    Parameters
    ----------
    units : list of sequences, dicts, or modules containing units, or units
        This is a list of things in which units may be found
        (sequences, dicts or modules), or units themselves.  The
        entire set will be "enabled" for searching through by methods
        like `UnitBase.find_equivalent_units` and `UnitBase.compose`.

    Examples
    --------

    >>> from astropy import units as u
    >>> with u.set_enabled_units([u.pc]):
    ...     u.m.find_equivalent_units()
    ...
      Primary name | Unit definition | Aliases
    [
      pc           | 3.08568e+16 m   | parsec  ,
    ]
    >>> u.m.find_equivalent_units()
      Primary name | Unit definition | Aliases
    [
      AU           | 1.49598e+11 m   | au, astronomical_unit ,
      Angstrom     | 1e-10 m         | AA, angstrom          ,
      cm           | 0.01 m          | centimeter            ,
      earthRad     | 6.3781e+06 m    | R_earth, Rearth       ,
      jupiterRad   | 7.1492e+07 m    | R_jup, Rjup, R_jupiter, Rjupiter ,
      lyr          | 9.46073e+15 m   | lightyear             ,
      m            | irreducible     | meter                 ,
      micron       | 1e-06 m         |                       ,
      pc           | 3.08568e+16 m   | parsec                ,
      solRad       | 6.957e+08 m     | R_sun, Rsun           ,
    ]
    """
    # get a context with a new registry, using equivalencies of the current one
    context = _UnitContext(
        equivalencies=get_current_unit_registry().equivalencies)
    # in this new current registry, enable the units requested
    get_current_unit_registry().set_enabled_units(units)
    return context
```
### 65 - astropy/units/core.py:

Start line: 1625, End line: 1644

```python
class IrreducibleUnit(NamedUnit):
    """
    Irreducible units are the units that all other units are defined
    in terms of.

    Examples are meters, seconds, kilograms, amperes, etc.  There is
    only once instance of such a unit per type.
    """

    def __reduce__(self):
        # When IrreducibleUnit objects are passed to other processes
        # over multiprocessing, they need to be recreated to be the
        # ones already in the subprocesses' namespace, not new
        # objects, or they will be considered "unconvertible".
        # Therefore, we have a custom pickler/unpickler that
        # understands how to recreate the Unit on the other side.
        registry = get_current_unit_registry().registry
        return (_recreate_irreducible_unit,
                (self.__class__, list(self.names), self.name in registry),
                self.__dict__)
```
### 66 - astropy/units/core.py:

Start line: 1253, End line: 1296

```python
class UnitBase(metaclass=InheritDocstrings):

    def to_system(self, system):
        """
        Converts this unit into ones belonging to the given system.
        Since more than one result may be possible, a list is always
        returned.

        Parameters
        ----------
        system : module
            The module that defines the unit system.  Commonly used
            ones include `astropy.units.si` and `astropy.units.cgs`.

            To use your own module it must contain unit objects and a
            sequence member named ``bases`` containing the base units of
            the system.

        Returns
        -------
        units : list of `CompositeUnit`
            The list is ranked so that units containing only the base
            units of that system will appear first.
        """
        bases = set(system.bases)

        def score(compose):
            # In case that compose._bases has no elements we return
            # 'np.inf' as 'score value'.  It does not really matter which
            # number we would return. This case occurs for instance for
            # dimensionless quantities:
            compose_bases = compose.bases
            if len(compose_bases) == 0:
                return np.inf
            else:
                sum = 0
                for base in compose_bases:
                    if base in bases:
                        sum += 1

                return sum / float(len(compose_bases))

        x = self.decompose(bases=bases)
        composed = x.compose(units=system)
        composed = sorted(composed, key=score, reverse=True)
        return composed
```
### 73 - astropy/units/core.py:

Start line: 825, End line: 872

```python
class UnitBase(metaclass=InheritDocstrings):

    def _apply_equivalencies(self, unit, other, equivalencies):
        """
        Internal function (used from `_get_converter`) to apply
        equivalence pairs.
        """
        def make_converter(scale1, func, scale2):
            def convert(v):
                return func(_condition_arg(v) / scale1) * scale2
            return convert

        for funit, tunit, a, b in equivalencies:
            if tunit is None:
                try:
                    ratio_in_funit = (other.decompose() /
                                      unit.decompose()).decompose([funit])
                    return make_converter(ratio_in_funit.scale, a, 1.)
                except UnitsError:
                    pass
            else:
                try:
                    scale1 = funit._to(unit)
                    scale2 = tunit._to(other)
                    return make_converter(scale1, a, scale2)
                except UnitsError:
                    pass
                try:
                    scale1 = tunit._to(unit)
                    scale2 = funit._to(other)
                    return make_converter(scale1, b, scale2)
                except UnitsError:
                    pass

        def get_err_str(unit):
            unit_str = unit.to_string('unscaled')
            physical_type = unit.physical_type
            if physical_type != 'unknown':
                unit_str = "'{0}' ({1})".format(
                    unit_str, physical_type)
            else:
                unit_str = "'{0}'".format(unit_str)
            return unit_str

        unit_str = get_err_str(unit)
        other_str = get_err_str(other)

        raise UnitConversionError(
            "{0} and {1} are not convertible".format(
                unit_str, other_str))
```
### 77 - astropy/units/core.py:

Start line: 488, End line: 529

```python
class UnitBase(metaclass=InheritDocstrings):
    """
    Abstract base class for units.

    Most of the arithmetic operations on units are defined in this
    base class.

    Should not be instantiated by users directly.
    """
    # Make sure that __rmul__ of units gets called over the __mul__ of Numpy
    # arrays to avoid element-wise multiplication.
    __array_priority__ = 1000

    def __deepcopy__(self, memo):
        # This may look odd, but the units conversion will be very
        # broken after deep-copying if we don't guarantee that a given
        # physical unit corresponds to only one instance
        return self

    def _repr_latex_(self):
        """
        Generate latex representation of unit name.  This is used by
        the IPython notebook to print a unit with a nice layout.

        Returns
        -------
        Latex string
        """
        return unit_format.Latex.to_string(self)

    def __bytes__(self):
        """Return string representation for unit"""
        return unit_format.Generic.to_string(self).encode('unicode_escape')

    def __str__(self):
        """Return string representation for unit"""
        return unit_format.Generic.to_string(self)

    def __repr__(self):
        string = unit_format.Generic.to_string(self)

        return 'Unit("{0}")'.format(string)
```
### 87 - astropy/units/core.py:

Start line: 452, End line: 485

```python
class UnitsError(Exception):
    """
    The base class for unit-specific exceptions.
    """


class UnitScaleError(UnitsError, ValueError):
    """
    Used to catch the errors involving scaled units,
    which are not recognized by FITS format.
    """
    pass


class UnitConversionError(UnitsError, ValueError):
    """
    Used specifically for errors related to converting between units or
    interpreting units in terms of other units.
    """


class UnitTypeError(UnitsError, TypeError):
    """
    Used specifically for errors in setting to units not allowed by a class.

    E.g., would be raised if the unit of an `~astropy.coordinates.Angle`
    instances were set to a non-angular unit.
    """


class UnitsWarning(AstropyWarning):
    """
    The base class for unit-specific warnings.
    """
```
### 95 - astropy/units/core.py:

Start line: 2206, End line: 2274

```python
def def_unit(s, represents=None, doc=None, format=None, prefixes=False,
             exclude_prefixes=[], namespace=None):
    """
    Factory function for defining new units.

    Parameters
    ----------
    s : str or list of str
        The name of the unit.  If a list, the first element is the
        canonical (short) name, and the rest of the elements are
        aliases.

    represents : UnitBase instance, optional
        The unit that this named unit represents.  If not provided,
        a new `IrreducibleUnit` is created.

    doc : str, optional
        A docstring describing the unit.

    format : dict, optional
        A mapping to format-specific representations of this unit.
        For example, for the ``Ohm`` unit, it might be nice to
        have it displayed as ``\\Omega`` by the ``latex``
        formatter.  In that case, `format` argument should be set
        to::

            {'latex': r'\\Omega'}

    prefixes : bool or list, optional
        When `True`, generate all of the SI prefixed versions of the
        unit as well.  For example, for a given unit ``m``, will
        generate ``mm``, ``cm``, ``km``, etc.  When a list, it is a list of
        prefix definitions of the form:

            (short_names, long_tables, factor)

        Default is `False`.  This function always returns the base
        unit object, even if multiple scaled versions of the unit were
        created.

    exclude_prefixes : list of str, optional
        If any of the SI prefixes need to be excluded, they may be
        listed here.  For example, ``Pa`` can be interpreted either as
        "petaannum" or "Pascal".  Therefore, when defining the
        prefixes for ``a``, ``exclude_prefixes`` should be set to
        ``["P"]``.

    namespace : dict, optional
        When provided, inject the unit (and all of its aliases and
        prefixes), into the given namespace dictionary.

    Returns
    -------
    unit : `UnitBase` object
        The newly-defined unit, or a matching unit that was already
        defined.
    """

    if represents is not None:
        result = Unit(s, represents, namespace=namespace, doc=doc,
                      format=format)
    else:
        result = IrreducibleUnit(
            s, namespace=namespace, doc=doc, format=format)

    if prefixes:
        _add_prefixes(result, excludes=exclude_prefixes, namespace=namespace,
                      prefixes=prefixes)
    return result
```
### 98 - astropy/units/core.py:

Start line: 337, End line: 389

```python
def add_enabled_units(units):
    """
    Adds to the set of units enabled in the unit registry.

    These units are searched when using
    `UnitBase.find_equivalent_units`, for example.

    This may be used either permanently, or as a context manager using
    the ``with`` statement (see example below).

    Parameters
    ----------
    units : list of sequences, dicts, or modules containing units, or units
        This is a list of things in which units may be found
        (sequences, dicts or modules), or units themselves.  The
        entire set will be added to the "enabled" set for searching
        through by methods like `UnitBase.find_equivalent_units` and
        `UnitBase.compose`.

    Examples
    --------

    >>> from astropy import units as u
    >>> from astropy.units import imperial
    >>> with u.add_enabled_units(imperial):
    ...     u.m.find_equivalent_units()
    ...
      Primary name | Unit definition | Aliases
    [
      AU           | 1.49598e+11 m   | au, astronomical_unit ,
      Angstrom     | 1e-10 m         | AA, angstrom          ,
      cm           | 0.01 m          | centimeter            ,
      earthRad     | 6.3781e+06 m    | R_earth, Rearth       ,
      ft           | 0.3048 m        | foot                  ,
      fur          | 201.168 m       | furlong               ,
      inch         | 0.0254 m        |                       ,
      jupiterRad   | 7.1492e+07 m    | R_jup, Rjup, R_jupiter, Rjupiter ,
      lyr          | 9.46073e+15 m   | lightyear             ,
      m            | irreducible     | meter                 ,
      mi           | 1609.34 m       | mile                  ,
      micron       | 1e-06 m         |                       ,
      mil          | 2.54e-05 m      | thou                  ,
      nmi          | 1852 m          | nauticalmile, NM      ,
      pc           | 3.08568e+16 m   | parsec                ,
      solRad       | 6.957e+08 m     | R_sun, Rsun           ,
      yd           | 0.9144 m        | yard                  ,
    ]
    """
    # get a context with a new registry, which is a copy of the current one
    context = _UnitContext(get_current_unit_registry())
    # in this new current registry, enable the further units requested
    get_current_unit_registry().add_enabled_units(units)
    return context
```
### 104 - astropy/units/core.py:

Start line: 64, End line: 103

```python
def _normalize_equivalencies(equivalencies):
    """
    Normalizes equivalencies, ensuring each is a 4-tuple of the form::

    (from_unit, to_unit, forward_func, backward_func)

    Parameters
    ----------
    equivalencies : list of equivalency pairs

    Raises
    ------
    ValueError if an equivalency cannot be interpreted
    """
    if equivalencies is None:
        return []

    normalized = []

    for i, equiv in enumerate(equivalencies):
        if len(equiv) == 2:
            funit, tunit = equiv
            a = b = lambda x: x
        elif len(equiv) == 3:
            funit, tunit, a = equiv
            b = a
        elif len(equiv) == 4:
            funit, tunit, a, b = equiv
        else:
            raise ValueError(
                "Invalid equivalence entry {0}: {1!r}".format(i, equiv))
        if not (funit is Unit(funit) and
                (tunit is None or tunit is Unit(tunit)) and
                callable(a) and
                callable(b)):
            raise ValueError(
                "Invalid equivalence entry {0}: {1!r}".format(i, equiv))
        normalized.append((funit, tunit, a, b))

    return normalized
```
### 116 - astropy/units/core.py:

Start line: 1123, End line: 1181

```python
class UnitBase(metaclass=InheritDocstrings):

    def compose(self, equivalencies=[], units=None, max_depth=2,
                include_prefix_units=None):
        """
        Return the simplest possible composite unit(s) that represent
        the given unit.  Since there may be multiple equally simple
        compositions of the unit, a list of units is always returned.

        Parameters
        ----------
        equivalencies : list of equivalence pairs, optional
            A list of equivalence pairs to also list.  See
            :ref:`unit_equivalencies`.
            This list is in addition to possible global defaults set by, e.g.,
            `set_enabled_equivalencies`.
            Use `None` to turn off all equivalencies.

        units : set of units to compose to, optional
            If not provided, any known units may be used to compose
            into.  Otherwise, ``units`` is a dict, module or sequence
            containing the units to compose into.

        max_depth : int, optional
            The maximum recursion depth to use when composing into
            composite units.

        include_prefix_units : bool, optional
            When `True`, include prefixed units in the result.
            Default is `True` if a sequence is passed in to ``units``,
            `False` otherwise.

        Returns
        -------
        units : list of `CompositeUnit`
            A list of candidate compositions.  These will all be
            equally simple, but it may not be possible to
            automatically determine which of the candidates are
            better.
        """
        # if units parameter is specified and is a sequence (list|tuple),
        # include_prefix_units is turned on by default.  Ex: units=[u.kpc]
        if include_prefix_units is None:
            include_prefix_units = isinstance(units, (list, tuple))

        # Pre-normalize the equivalencies list
        equivalencies = self._normalize_equivalencies(equivalencies)

        # The namespace of units to compose into should be filtered to
        # only include units with bases in common with self, otherwise
        # they can't possibly provide useful results.  Having too many
        # destination units greatly increases the search space.

        def has_bases_in_common(a, b):
            if len(a.bases) == 0 and len(b.bases) == 0:
                return True
            for ab in a.bases:
                for bb in b.bases:
                    if ab == bb:
                        return True
            return False
        # ... other code
```
### 118 - astropy/units/core.py:

Start line: 1439, End line: 1527

```python
class NamedUnit(UnitBase):
    """
    The base class of units that have a name.

    Parameters
    ----------
    st : str, list of str, 2-tuple
        The name of the unit.  If a list of strings, the first element
        is the canonical (short) name, and the rest of the elements
        are aliases.  If a tuple of lists, the first element is a list
        of short names, and the second element is a list of long
        names; all but the first short name are considered "aliases".
        Each name *should* be a valid Python identifier to make it
        easy to access, but this is not required.

    namespace : dict, optional
        When provided, inject the unit, and all of its aliases, in the
        given namespace dictionary.  If a unit by the same name is
        already in the namespace, a ValueError is raised.

    doc : str, optional
        A docstring describing the unit.

    format : dict, optional
        A mapping to format-specific representations of this unit.
        For example, for the ``Ohm`` unit, it might be nice to have it
        displayed as ``\\Omega`` by the ``latex`` formatter.  In that
        case, `format` argument should be set to::

            {'latex': r'\\Omega'}

    Raises
    ------
    ValueError
        If any of the given unit names are already in the registry.

    ValueError
        If any of the given unit names are not valid Python tokens.
    """

    def __init__(self, st, doc=None, format=None, namespace=None):

        UnitBase.__init__(self)

        if isinstance(st, (bytes, str)):
            self._names = [st]
            self._short_names = [st]
            self._long_names = []
        elif isinstance(st, tuple):
            if not len(st) == 2:
                raise ValueError("st must be string, list or 2-tuple")
            self._names = st[0] + [n for n in st[1] if n not in st[0]]
            if not len(self._names):
                raise ValueError("must provide at least one name")
            self._short_names = st[0][:]
            self._long_names = st[1][:]
        else:
            if len(st) == 0:
                raise ValueError(
                    "st list must have at least one entry")
            self._names = st[:]
            self._short_names = [st[0]]
            self._long_names = st[1:]

        if format is None:
            format = {}
        self._format = format

        if doc is None:
            doc = self._generate_doc()
        else:
            doc = textwrap.dedent(doc)
            doc = textwrap.fill(doc)

        self.__doc__ = doc

        self._inject(namespace)

    def _generate_doc(self):
        """
        Generate a docstring for the unit if the user didn't supply
        one.  This is only used from the constructor and may be
        overridden in subclasses.
        """
        names = self.names
        if len(self.names) > 1:
            return "{1} ({0})".format(*names[:2])
        else:
            return names[0]
```
### 126 - astropy/units/core.py:

Start line: 392, End line: 425

```python
def set_enabled_equivalencies(equivalencies):
    """
    Sets the equivalencies enabled in the unit registry.

    These equivalencies are used if no explicit equivalencies are given,
    both in unit conversion and in finding equivalent units.

    This is meant in particular for allowing angles to be dimensionless.
    Use with care.

    Parameters
    ----------
    equivalencies : list of equivalent pairs
        E.g., as returned by
        `~astropy.units.equivalencies.dimensionless_angles`.

    Examples
    --------
    Exponentiation normally requires dimensionless quantities.  To avoid
    problems with complex phases::

        >>> from astropy import units as u
        >>> with u.set_enabled_equivalencies(u.dimensionless_angles()):
        ...     phase = 0.5 * u.cycle
        ...     np.exp(1j*phase)  # doctest: +SKIP
        <Quantity  -1. +1.22464680e-16j>
    """
    # doctest skipped as the complex number formatting changed in numpy 1.14.
    #
    # get a context with a new registry, using all units of the current one
    context = _UnitContext(get_current_unit_registry())
    # in this new current registry, enable the equivalencies requested
    get_current_unit_registry().set_enabled_equivalencies(equivalencies)
    return context
```
### 130 - astropy/units/core.py:

Start line: 2277, End line: 2317

```python
def _condition_arg(value):
    """
    Validate value is acceptable for conversion purposes.

    Will convert into an array if not a scalar, and can be converted
    into an array

    Parameters
    ----------
    value : int or float value, or sequence of such values

    Returns
    -------
    Scalar value or numpy array

    Raises
    ------
    ValueError
        If value is not as expected
    """
    if isinstance(value, (float, int, complex)):
        return value

    if isinstance(value, np.ndarray) and value.dtype.kind in ['i', 'f', 'c']:
        return value

    avalue = np.array(value)
    if avalue.dtype.kind not in ['i', 'f', 'c']:
        raise ValueError("Value not scalar compatible or convertible to "
                         "an int, float, or complex array")
    return avalue


dimensionless_unscaled = CompositeUnit(1, [], [], _error_check=False)
# Abbreviation of the above, see #1980
one = dimensionless_unscaled

# Maintain error in old location for backward compatibility
# TODO: Is this still needed? Should there be a deprecation warning?
unit_format.fits.UnitScaleError = UnitScaleError
```
### 133 - astropy/units/core.py:

Start line: 106, End line: 152

```python
class _UnitRegistry:
    """
    Manages a registry of the enabled units.
    """

    def __init__(self, init=[], equivalencies=[]):

        if isinstance(init, _UnitRegistry):
            # If passed another registry we don't need to rebuild everything.
            # but because these are mutable types we don't want to create
            # conflicts so everything needs to be copied.
            self._equivalencies = init._equivalencies.copy()
            self._all_units = init._all_units.copy()
            self._registry = init._registry.copy()
            self._non_prefix_units = init._non_prefix_units.copy()
            # The physical type is a dictionary containing sets as values.
            # All of these must be copied otherwise we could alter the old
            # registry.
            self._by_physical_type = {k: v.copy() for k, v in
                                      init._by_physical_type.items()}

        else:
            self._reset_units()
            self._reset_equivalencies()
            self.add_enabled_units(init)
            self.add_enabled_equivalencies(equivalencies)

    def _reset_units(self):
        self._all_units = set()
        self._non_prefix_units = set()
        self._registry = {}
        self._by_physical_type = {}

    def _reset_equivalencies(self):
        self._equivalencies = set()

    @property
    def registry(self):
        return self._registry

    @property
    def all_units(self):
        return self._all_units

    @property
    def non_prefix_units(self):
        return self._non_prefix_units
```
### 141 - astropy/units/core.py:

Start line: 2043, End line: 2082

```python
class CompositeUnit(UnitBase):

    def _expand_and_gather(self, decompose=False, bases=set()):
        def add_unit(unit, power, scale):
            if bases and unit not in bases:
                for base in bases:
                    try:
                        scale *= unit._to(base) ** power
                    except UnitsError:
                        pass
                    else:
                        unit = base
                        break

            if unit in new_parts:
                a, b = resolve_fractions(new_parts[unit], power)
                new_parts[unit] = a + b
            else:
                new_parts[unit] = power
            return scale

        new_parts = {}
        scale = self._scale

        for b, p in zip(self._bases, self._powers):
            if decompose and b not in bases:
                b = b.decompose(bases=bases)

            if isinstance(b, CompositeUnit):
                scale *= b._scale ** p
                for b_sub, p_sub in zip(b._bases, b._powers):
                    a, b = resolve_fractions(p_sub, p)
                    scale = add_unit(b_sub, a * b, scale)
            else:
                scale = add_unit(b, p, scale)

        new_parts = [x for x in new_parts.items() if x[1] != 0]
        new_parts.sort(key=lambda x: (-x[1], getattr(x[0], 'name', '')))

        self._bases = [x[0] for x in new_parts]
        self._powers = [x[1] for x in new_parts]
        self._scale = sanitize_scale(scale)
```
### 147 - astropy/units/core.py:

Start line: 1949, End line: 2041

```python
class PrefixUnit(Unit):
    """
    A unit that is simply a SI-prefixed version of another unit.

    For example, ``mm`` is a `PrefixUnit` of ``.001 * m``.

    The constructor is the same as for `Unit`.
    """


class CompositeUnit(UnitBase):
    """
    Create a composite unit using expressions of previously defined
    units.

    Direct use of this class is not recommended. Instead use the
    factory function `Unit` and arithmetic operators to compose
    units.

    Parameters
    ----------
    scale : number
        A scaling factor for the unit.

    bases : sequence of `UnitBase`
        A sequence of units this unit is composed of.

    powers : sequence of numbers
        A sequence of powers (in parallel with ``bases``) for each
        of the base units.
    """

    def __init__(self, scale, bases, powers, decompose=False,
                 decompose_bases=set(), _error_check=True):
        # There are many cases internal to astropy.units where we
        # already know that all the bases are Unit objects, and the
        # powers have been validated.  In those cases, we can skip the
        # error checking for performance reasons.  When the private
        # kwarg `_error_check` is False, the error checking is turned
        # off.
        if _error_check:
            scale = sanitize_scale(scale)
            for base in bases:
                if not isinstance(base, UnitBase):
                    raise TypeError(
                        "bases must be sequence of UnitBase instances")
            powers = [validate_power(p) for p in powers]

        self._scale = scale
        self._bases = bases
        self._powers = powers
        self._decomposed_cache = None
        self._expand_and_gather(decompose=decompose, bases=decompose_bases)
        self._hash = None

    def __repr__(self):
        if len(self._bases):
            return super().__repr__()
        else:
            if self._scale != 1.0:
                return 'Unit(dimensionless with a scale of {0})'.format(
                    self._scale)
            else:
                return 'Unit(dimensionless)'

    def __hash__(self):
        if self._hash is None:
            parts = ([str(self._scale)] +
                     [x.name for x in self._bases] +
                     [str(x) for x in self._powers])
            self._hash = hash(tuple(parts))
        return self._hash

    @property
    def scale(self):
        """
        Return the scale of the composite unit.
        """
        return self._scale

    @property
    def bases(self):
        """
        Return the bases of the composite unit.
        """
        return self._bases

    @property
    def powers(self):
        """
        Return the powers of the composite unit.
        """
        return self._powers
```
### 153 - astropy/units/core.py:

Start line: 1584, End line: 1602

```python
class NamedUnit(UnitBase):

    def _inject(self, namespace=None):
        """
        Injects the unit, and all of its aliases, in the given
        namespace dictionary.
        """
        if namespace is None:
            return

        # Loop through all of the names first, to ensure all of them
        # are new, then add them all as a single "transaction" below.
        for name in self._names:
            if name in namespace and self != namespace[name]:
                raise ValueError(
                    "Object with name {0!r} already exists in "
                    "given namespace ({1!r}).".format(
                        name, namespace[name]))

        for name in self._names:
            namespace[name] = self
```
