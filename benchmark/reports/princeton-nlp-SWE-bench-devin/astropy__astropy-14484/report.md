# astropy__astropy-14484

| **astropy/astropy** | `09e54670e4a46ed510e32d8206e4853920684952` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 10262 |
| **Any found context length** | 10262 |
| **Avg pos** | 31.0 |
| **Min pos** | 31 |
| **Max pos** | 31 |
| **Top file pos** | 4 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/astropy/units/quantity_helper/function_helpers.py b/astropy/units/quantity_helper/function_helpers.py
--- a/astropy/units/quantity_helper/function_helpers.py
+++ b/astropy/units/quantity_helper/function_helpers.py
@@ -75,9 +75,10 @@
     np.put, np.fill_diagonal, np.tile, np.repeat,
     np.split, np.array_split, np.hsplit, np.vsplit, np.dsplit,
     np.stack, np.column_stack, np.hstack, np.vstack, np.dstack,
-    np.amax, np.amin, np.ptp, np.sum, np.cumsum,
+    np.max, np.min, np.amax, np.amin, np.ptp, np.sum, np.cumsum,
     np.prod, np.product, np.cumprod, np.cumproduct,
     np.round, np.around,
+    np.round_,  # Alias for np.round in NUMPY_LT_1_25, but deprecated since.
     np.fix, np.angle, np.i0, np.clip,
     np.isposinf, np.isneginf, np.isreal, np.iscomplex,
     np.average, np.mean, np.std, np.var, np.median, np.trace,

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| astropy/units/quantity_helper/function_helpers.py | 78 | 78 | 31 | 4 | 10262


## Problem Statement

```
New Quantity warning starting with yesterday's numpy-dev
### Description

Starting today, `photutils` CI tests with `astropy-dev` and `numpy-dev` started failing due a new warning.  I've extracted a MWE showing the warning:

\`\`\`python
import astropy.units as u
import pytest
from numpy.testing import assert_equal

a = [78, 78, 81] * u.pix**2
b = [78.5, 78.5, 78.625] * u.pix**2
with pytest.raises(AssertionError):
    assert_equal(a, b)
\`\`\`
The warning is:
\`\`\`
WARNING: function 'max' is not known to astropy's Quantity. Will run it anyway, hoping it will treat ndarray subclasses correctly. Please raise an issue at https://github.com/astropy/astropy/issues. [astropy.units.quantity]
\`\`\`

The warning is not emitted with `astropy-dev` and `numpy` stable (1.24.2).

CC: @mhvk 

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 astropy/units/quantity.py | 1133 | 1161| 211 | 211 | 18636 | 
| 2 | 1 astropy/units/quantity.py | 9 | 60| 315 | 526 | 18636 | 
| 3 | 1 astropy/units/quantity.py | 1891 | 1911| 245 | 771 | 18636 | 
| 4 | 1 astropy/units/quantity.py | 1789 | 1803| 148 | 919 | 18636 | 
| 5 | 1 astropy/units/quantity.py | 526 | 582| 550 | 1469 | 18636 | 
| 6 | 1 astropy/units/quantity.py | 1957 | 2036| 658 | 2127 | 18636 | 
| 7 | 2 astropy/coordinates/spectral_coordinate.py | 1 | 36| 213 | 2340 | 25221 | 
| 8 | 2 astropy/units/quantity.py | 584 | 606| 233 | 2573 | 25221 | 
| 9 | 2 astropy/units/quantity.py | 1163 | 1186| 176 | 2749 | 25221 | 
| 10 | 2 astropy/units/quantity.py | 1330 | 1391| 411 | 3160 | 25221 | 
| 11 | 2 astropy/units/quantity.py | 434 | 525| 780 | 3940 | 25221 | 
| 12 | 2 astropy/units/quantity.py | 1188 | 1290| 750 | 4690 | 25221 | 
| 13 | 3 astropy/units/function/core.py | 773 | 791| 214 | 4904 | 31333 | 
| 14 | 3 astropy/units/quantity.py | 333 | 432| 823 | 5727 | 31333 | 
| 15 | 3 astropy/units/quantity.py | 1316 | 1328| 159 | 5886 | 31333 | 
| 16 | 3 astropy/units/quantity.py | 866 | 887| 201 | 6087 | 31333 | 
| 17 | 3 astropy/units/quantity.py | 1292 | 1314| 187 | 6274 | 31333 | 
| 18 | **4 astropy/units/quantity_helper/function_helpers.py** | 101 | 132| 366 | 6640 | 41398 | 
| 19 | 4 astropy/units/quantity.py | 608 | 618| 116 | 6756 | 41398 | 
| 20 | 4 astropy/units/function/core.py | 631 | 644| 141 | 6897 | 41398 | 
| 21 | 5 astropy/visualization/units.py | 3 | 48| 315 | 7212 | 42077 | 
| 22 | 6 astropy/units/equivalencies.py | 4 | 39| 217 | 7429 | 51125 | 
| 23 | 7 astropy/coordinates/spectral_quantity.py | 1 | 22| 172 | 7601 | 53805 | 
| 24 | 8 astropy/io/misc/asdf/tags/unit/quantity.py | 2 | 33| 200 | 7801 | 54022 | 
| 25 | 8 astropy/units/quantity.py | 63 | 78| 109 | 7910 | 54022 | 
| 26 | 8 astropy/units/quantity.py | 1805 | 1889| 747 | 8657 | 54022 | 
| 27 | **8 astropy/units/quantity_helper/function_helpers.py** | 301 | 321| 153 | 8810 | 54022 | 
| 28 | 9 astropy/coordinates/attributes.py | 309 | 356| 313 | 9123 | 57605 | 
| 29 | 9 astropy/units/function/core.py | 551 | 590| 247 | 9370 | 57605 | 
| 30 | 9 astropy/units/function/core.py | 646 | 656| 137 | 9507 | 57605 | 
| **-> 31 <-** | **9 astropy/units/quantity_helper/function_helpers.py** | 37 | 100| 755 | 10262 | 57605 | 
| 32 | 9 astropy/units/quantity.py | 260 | 332| 755 | 11017 | 57605 | 
| 33 | **9 astropy/units/quantity_helper/function_helpers.py** | 516 | 542| 205 | 11222 | 57605 | 
| 34 | 9 astropy/units/quantity.py | 1002 | 1064| 405 | 11627 | 57605 | 
| 35 | 9 astropy/units/quantity.py | 1535 | 1550| 154 | 11781 | 57605 | 
| 36 | 9 astropy/units/quantity.py | 1086 | 1099| 131 | 11912 | 57605 | 
| 37 | 9 astropy/units/quantity.py | 620 | 699| 773 | 12685 | 57605 | 
| 38 | 9 astropy/units/quantity.py | 1483 | 1511| 251 | 12936 | 57605 | 
| 39 | 10 astropy/units/core.py | 8 | 51| 215 | 13151 | 76794 | 
| 40 | 10 astropy/units/quantity.py | 839 | 864| 263 | 13414 | 76794 | 
| 41 | 11 astropy/units/physical.py | 5 | 131| 128 | 13542 | 82414 | 
| 42 | **11 astropy/units/quantity_helper/function_helpers.py** | 545 | 565| 200 | 13742 | 82414 | 
| 43 | 11 astropy/units/quantity.py | 1699 | 1787| 792 | 14534 | 82414 | 
| 44 | **11 astropy/units/quantity_helper/function_helpers.py** | 185 | 271| 771 | 15305 | 82414 | 
| 45 | 12 astropy/table/column.py | 3 | 65| 397 | 15702 | 97243 | 
| 46 | 12 astropy/units/quantity.py | 1574 | 1609| 327 | 16029 | 97243 | 
| 47 | 12 astropy/coordinates/spectral_quantity.py | 86 | 113| 222 | 16251 | 97243 | 
| 48 | 12 astropy/units/quantity.py | 1101 | 1131| 189 | 16440 | 97243 | 
| 49 | 13 astropy/modeling/fitting.py | 25 | 78| 357 | 16797 | 114979 | 
| 50 | 13 astropy/units/function/core.py | 721 | 754| 229 | 17026 | 114979 | 
| 51 | **13 astropy/units/quantity_helper/function_helpers.py** | 568 | 614| 345 | 17371 | 114979 | 
| 52 | 13 astropy/units/quantity.py | 909 | 946| 372 | 17743 | 114979 | 
| 53 | 13 astropy/units/function/core.py | 4 | 47| 237 | 17980 | 114979 | 
| 54 | 13 astropy/visualization/units.py | 50 | 73| 209 | 18189 | 114979 | 
| 55 | **13 astropy/units/quantity_helper/function_helpers.py** | 274 | 298| 216 | 18405 | 114979 | 
| 56 | 14 astropy/units/quantity_helper/converters.py | 330 | 393| 500 | 18905 | 118267 | 
| 57 | 14 astropy/units/function/core.py | 755 | 771| 173 | 19078 | 118267 | 
| 58 | **14 astropy/units/quantity_helper/function_helpers.py** | 362 | 383| 193 | 19271 | 118267 | 
| 59 | 15 astropy/units/function/logarithmic.py | 382 | 407| 333 | 19604 | 121877 | 
| 60 | **15 astropy/units/quantity_helper/function_helpers.py** | 164 | 182| 188 | 19792 | 121877 | 
| 61 | 16 astropy/stats/sigma_clipping.py | 3 | 19| 114 | 19906 | 130704 | 
| 62 | 16 astropy/table/column.py | 1076 | 1087| 131 | 20037 | 130704 | 
| 63 | **16 astropy/units/quantity_helper/function_helpers.py** | 1004 | 1117| 795 | 20832 | 130704 | 
| 64 | 16 astropy/visualization/units.py | 75 | 100| 182 | 21014 | 130704 | 
| 65 | 16 astropy/coordinates/spectral_quantity.py | 71 | 84| 159 | 21173 | 130704 | 
| 66 | 17 astropy/io/votable/exceptions.py | 411 | 428| 211 | 21384 | 144116 | 
| 67 | 17 astropy/units/quantity.py | 1513 | 1533| 157 | 21541 | 144116 | 
| 68 | 18 astropy/coordinates/errors.py | 6 | 31| 123 | 21664 | 145223 | 
| 69 | **18 astropy/units/quantity_helper/function_helpers.py** | 913 | 940| 240 | 21904 | 145223 | 
| 70 | **18 astropy/units/quantity_helper/function_helpers.py** | 637 | 668| 202 | 22106 | 145223 | 
| 71 | 18 astropy/units/quantity.py | 1552 | 1572| 182 | 22288 | 145223 | 
| 72 | 18 astropy/units/quantity.py | 701 | 764| 506 | 22794 | 145223 | 
| 73 | 19 astropy/nddata/nduncertainty.py | 3 | 27| 141 | 22935 | 154825 | 
| 74 | 19 astropy/coordinates/spectral_quantity.py | 54 | 69| 161 | 23096 | 154825 | 
| 75 | 20 astropy/units/quantity_helper/helpers.py | 338 | 424| 758 | 23854 | 158680 | 
| 76 | 20 astropy/units/function/logarithmic.py | 268 | 355| 817 | 24671 | 158680 | 
| 77 | 20 astropy/nddata/nduncertainty.py | 186 | 216| 240 | 24911 | 158680 | 
| 78 | 20 astropy/units/quantity.py | 1066 | 1084| 186 | 25097 | 158680 | 
| 79 | **20 astropy/units/quantity_helper/function_helpers.py** | 844 | 910| 607 | 25704 | 158680 | 
| 80 | 20 astropy/units/quantity_helper/converters.py | 174 | 289| 1146 | 26850 | 158680 | 
| 81 | 20 astropy/units/core.py | 906 | 938| 233 | 27083 | 158680 | 
| 82 | 20 astropy/units/quantity.py | 158 | 181| 224 | 27307 | 158680 | 
| 83 | 21 astropy/constants/astropyconst40.py | 7 | 70| 512 | 27819 | 159247 | 
| 84 | 22 astropy/wcs/wcsapi/conftest.py | 1 | 34| 267 | 28086 | 160730 | 
| 85 | 22 astropy/units/function/core.py | 592 | 629| 290 | 28376 | 160730 | 
| 86 | 23 astropy/constants/astropyconst20.py | 7 | 70| 512 | 28888 | 161297 | 
| 87 | 23 astropy/units/function/logarithmic.py | 357 | 380| 148 | 29036 | 161297 | 
| 88 | 23 astropy/coordinates/attributes.py | 270 | 307| 324 | 29360 | 161297 | 
| 89 | 24 astropy/units/format/utils.py | 193 | 220| 205 | 29565 | 162688 | 
| 90 | 24 astropy/coordinates/errors.py | 157 | 176| 133 | 29698 | 162688 | 
| 91 | 24 astropy/units/function/core.py | 477 | 550| 756 | 30454 | 162688 | 
| 92 | 25 astropy/constants/constant.py | 123 | 159| 305 | 30759 | 164663 | 
| 93 | 25 astropy/units/quantity.py | 766 | 837| 744 | 31503 | 164663 | 
| 94 | 25 astropy/io/votable/exceptions.py | 85 | 95| 119 | 31622 | 164663 | 
| 95 | 26 astropy/coordinates/builtin_frames/utils.py | 7 | 37| 323 | 31945 | 168500 | 
| 96 | 26 astropy/units/quantity.py | 889 | 907| 182 | 32127 | 168500 | 
| 97 | 26 astropy/units/function/core.py | 658 | 695| 297 | 32424 | 168500 | 
| 98 | 26 astropy/units/function/core.py | 697 | 719| 168 | 32592 | 168500 | 
| 99 | 27 astropy/units/decorators.py | 138 | 219| 577 | 33169 | 170889 | 
| 100 | **27 astropy/units/quantity_helper/function_helpers.py** | 671 | 708| 252 | 33421 | 170889 | 
| 101 | 28 astropy/units/quantity_helper/__init__.py | 8 | 17| 82 | 33503 | 171020 | 
| 102 | 28 astropy/io/votable/exceptions.py | 145 | 156| 126 | 33629 | 171020 | 
| 103 | 29 astropy/utils/compat/numpycompat.py | 7 | 29| 246 | 33875 | 171307 | 
| 104 | 29 astropy/units/quantity.py | 1912 | 1955| 383 | 34258 | 171307 | 
| 105 | 30 astropy/config/configuration.py | 12 | 88| 505 | 34763 | 177738 | 
| 106 | 31 astropy/units/quantity_helper/erfa.py | 88 | 117| 236 | 34999 | 181660 | 
| 107 | 31 astropy/coordinates/errors.py | 60 | 79| 134 | 35133 | 181660 | 
| 108 | 32 astropy/utils/exceptions.py | 11 | 81| 385 | 35518 | 182147 | 
| 109 | 32 astropy/units/equivalencies.py | 889 | 909| 144 | 35662 | 182147 | 
| 110 | 32 astropy/units/decorators.py | 221 | 345| 926 | 36588 | 182147 | 
| 111 | 32 astropy/units/quantity.py | 1611 | 1628| 142 | 36730 | 182147 | 


### Hint

```
We saw this downstream in Jdaviz too. cc @bmorris3 
```

## Patch

```diff
diff --git a/astropy/units/quantity_helper/function_helpers.py b/astropy/units/quantity_helper/function_helpers.py
--- a/astropy/units/quantity_helper/function_helpers.py
+++ b/astropy/units/quantity_helper/function_helpers.py
@@ -75,9 +75,10 @@
     np.put, np.fill_diagonal, np.tile, np.repeat,
     np.split, np.array_split, np.hsplit, np.vsplit, np.dsplit,
     np.stack, np.column_stack, np.hstack, np.vstack, np.dstack,
-    np.amax, np.amin, np.ptp, np.sum, np.cumsum,
+    np.max, np.min, np.amax, np.amin, np.ptp, np.sum, np.cumsum,
     np.prod, np.product, np.cumprod, np.cumproduct,
     np.round, np.around,
+    np.round_,  # Alias for np.round in NUMPY_LT_1_25, but deprecated since.
     np.fix, np.angle, np.i0, np.clip,
     np.isposinf, np.isneginf, np.isreal, np.iscomplex,
     np.average, np.mean, np.std, np.var, np.median, np.trace,

```

## Test Patch

```diff
diff --git a/astropy/units/tests/test_quantity_non_ufuncs.py b/astropy/units/tests/test_quantity_non_ufuncs.py
--- a/astropy/units/tests/test_quantity_non_ufuncs.py
+++ b/astropy/units/tests/test_quantity_non_ufuncs.py
@@ -17,7 +17,7 @@
     TBD_FUNCTIONS,
     UNSUPPORTED_FUNCTIONS,
 )
-from astropy.utils.compat import NUMPY_LT_1_23, NUMPY_LT_1_24
+from astropy.utils.compat import NUMPY_LT_1_23, NUMPY_LT_1_24, NUMPY_LT_1_25
 
 needs_array_function = pytest.mark.xfail(
     not ARRAY_FUNCTION_ENABLED, reason="Needs __array_function__ support"
@@ -608,6 +608,12 @@ def test_dsplit(self):
 
 
 class TestUfuncReductions(InvariantUnitTestSetup):
+    def test_max(self):
+        self.check(np.max)
+
+    def test_min(self):
+        self.check(np.min)
+
     def test_amax(self):
         self.check(np.amax)
 
@@ -658,8 +664,17 @@ def test_ptp(self):
         self.check(np.ptp)
         self.check(np.ptp, axis=0)
 
+    def test_round(self):
+        self.check(np.round)
+
     def test_round_(self):
-        self.check(np.round_)
+        if NUMPY_LT_1_25:
+            self.check(np.round_)
+        else:
+            with pytest.warns(
+                DeprecationWarning, match="`round_` is deprecated as of NumPy 1.25.0"
+            ):
+                self.check(np.round_)
 
     def test_around(self):
         self.check(np.around)
diff --git a/astropy/utils/masked/tests/test_function_helpers.py b/astropy/utils/masked/tests/test_function_helpers.py
--- a/astropy/utils/masked/tests/test_function_helpers.py
+++ b/astropy/utils/masked/tests/test_function_helpers.py
@@ -579,6 +579,12 @@ def check(self, function, *args, method=None, **kwargs):
         x = getattr(self.ma, method)(*args, **kwargs)
         assert_masked_equal(o, x)
 
+    def test_max(self):
+        self.check(np.max, method="max")
+
+    def test_min(self):
+        self.check(np.min, method="min")
+
     def test_amax(self):
         self.check(np.amax, method="max")
 
@@ -619,8 +625,17 @@ def test_ptp(self):
         self.check(np.ptp)
         self.check(np.ptp, axis=0)
 
+    def test_round(self):
+        self.check(np.round, method="round")
+
     def test_round_(self):
-        self.check(np.round_, method="round")
+        if NUMPY_LT_1_25:
+            self.check(np.round_, method="round")
+        else:
+            with pytest.warns(
+                DeprecationWarning, match="`round_` is deprecated as of NumPy 1.25.0"
+            ):
+                self.check(np.round_, method="round")
 
     def test_around(self):
         self.check(np.around, method="round")

```


## Code snippets

### 1 - astropy/units/quantity.py:

Start line: 1133, End line: 1161

```python
class Quantity(np.ndarray):

    # Equality needs to be handled explicitly as ndarray.__eq__ gives
    # DeprecationWarnings on any error, which is distracting, and does not
    # deal well with structured arrays (nor does the ufunc).
    def __eq__(self, other):
        try:
            other_value = self._to_own_unit(other)
        except UnitsError:
            return False
        except Exception:
            return NotImplemented
        return self.value.__eq__(other_value)

    def __ne__(self, other):
        try:
            other_value = self._to_own_unit(other)
        except UnitsError:
            return True
        except Exception:
            return NotImplemented
        return self.value.__ne__(other_value)

    # Unit conversion operator (<<).
    def __lshift__(self, other):
        try:
            other = Unit(other, parse_strict="silent")
        except UnitTypeError:
            return NotImplemented

        return self.__class__(self, other, copy=False, subok=True)
```
### 2 - astropy/units/quantity.py:

Start line: 9, End line: 60

```python
import numbers
import operator
import re
import warnings
from fractions import Fraction

# THIRD PARTY
import numpy as np

# LOCAL
from astropy import config as _config
from astropy.utils.compat import NUMPY_LT_1_22
from astropy.utils.data_info import ParentDtypeInfo
from astropy.utils.decorators import deprecated
from astropy.utils.exceptions import AstropyWarning
from astropy.utils.misc import isiterable

from .core import (
    Unit,
    UnitBase,
    UnitConversionError,
    UnitsError,
    UnitTypeError,
    dimensionless_unscaled,
    get_current_unit_registry,
)
from .format import Base, Latex
from .quantity_helper import can_have_arbitrary_unit, check_output, converters_and_unit
from .quantity_helper.function_helpers import (
    DISPATCHED_FUNCTIONS,
    FUNCTION_HELPERS,
    SUBCLASS_SAFE_FUNCTIONS,
    UNSUPPORTED_FUNCTIONS,
)
from .structured import StructuredUnit, _structured_unit_like_dtype
from .utils import is_effectively_unity

__all__ = [
    "Quantity",
    "SpecificTypeQuantity",
    "QuantityInfoBase",
    "QuantityInfo",
    "allclose",
    "isclose",
]


# We don't want to run doctests in the docstrings we inherit from Numpy
__doctest_skip__ = ["Quantity.*"]

_UNIT_NOT_INITIALISED = "(Unit not initialised)"
_UFUNCS_FILTER_WARNINGS = {np.arcsin, np.arccos, np.arccosh, np.arctanh}
```
### 3 - astropy/units/quantity.py:

Start line: 1891, End line: 1911

```python
class Quantity(np.ndarray):

    def _not_implemented_or_raise(self, function, types):
        # Our function helper or dispatcher found that the function does not
        # work with Quantity.  In principle, there may be another class that
        # knows what to do with us, for which we should return NotImplemented.
        # But if there is ndarray (or a non-Quantity subclass of it) around,
        # it quite likely coerces, so we should just break.
        if any(
            issubclass(t, np.ndarray) and not issubclass(t, Quantity) for t in types
        ):
            raise TypeError(
                f"the Quantity implementation cannot handle {function} "
                "with the given arguments."
            ) from None
        else:
            return NotImplemented

    # Calculation -- override ndarray methods to take into account units.
    # We use the corresponding numpy functions to evaluate the results, since
    # the methods do not always allow calling with keyword arguments.
    # For instance, np.array([0.,2.]).clip(a_min=0., a_max=1.) gives
    # TypeError: 'a_max' is an invalid keyword argument for this function.
```
### 4 - astropy/units/quantity.py:

Start line: 1789, End line: 1803

```python
class Quantity(np.ndarray):

    if NUMPY_LT_1_22:

        def argmax(self, axis=None, out=None):
            return self.view(np.ndarray).argmax(axis, out=out)

        def argmin(self, axis=None, out=None):
            return self.view(np.ndarray).argmin(axis, out=out)

    else:

        def argmax(self, axis=None, out=None, *, keepdims=False):
            return self.view(np.ndarray).argmax(axis=axis, out=out, keepdims=keepdims)

        def argmin(self, axis=None, out=None, *, keepdims=False):
            return self.view(np.ndarray).argmin(axis=axis, out=out, keepdims=keepdims)
```
### 5 - astropy/units/quantity.py:

Start line: 526, End line: 582

```python
class Quantity(np.ndarray):

    def __new__(
        cls,
        value,
        unit=None,
        dtype=np.inexact,
        copy=True,
        order=None,
        subok=False,
        ndmin=0,
    ):
        # ... other code
        if value_unit is None:
            # If the value has a `unit` attribute and if not None
            # (for Columns with uninitialized unit), treat it like a quantity.
            value_unit = getattr(value, "unit", None)
            if value_unit is None:
                # Default to dimensionless for no (initialized) unit attribute.
                if unit is None:
                    using_default_unit = True
                    unit = cls._default_unit
                value_unit = unit  # signal below that no conversion is needed
            else:
                try:
                    value_unit = Unit(value_unit)
                except Exception as exc:
                    raise TypeError(
                        f"The unit attribute {value.unit!r} of the input could "
                        "not be parsed as an astropy Unit."
                    ) from exc

                if unit is None:
                    unit = value_unit
                elif unit is not value_unit:
                    copy = False  # copy will be made in conversion at end

        value = np.array(
            value, dtype=dtype, copy=copy, order=order, subok=True, ndmin=ndmin
        )

        # For no-user-input unit, make sure the constructed unit matches the
        # structure of the data.
        if using_default_unit and value.dtype.names is not None:
            unit = value_unit = _structured_unit_like_dtype(value_unit, value.dtype)

        # check that array contains numbers or long int objects
        if value.dtype.kind in "OSU" and not (
            value.dtype.kind == "O" and isinstance(value.item(0), numbers.Number)
        ):
            raise TypeError("The value must be a valid Python or Numpy numeric type.")

        # by default, cast any integer, boolean, etc., to float
        if float_default and value.dtype.kind in "iuO":
            value = value.astype(float)

        # if we allow subclasses, allow a class from the unit.
        if subok:
            qcls = getattr(unit, "_quantity_class", cls)
            if issubclass(qcls, cls):
                cls = qcls

        value = value.view(cls)
        value._set_unit(value_unit)
        if unit is value_unit:
            return value
        else:
            # here we had non-Quantity input that had a "unit" attribute
            # with a unit different from the desired one.  So, convert.
            return value.to(unit)
```
### 6 - astropy/units/quantity.py:

Start line: 1957, End line: 2036

```python
class Quantity(np.ndarray):

    def trace(self, offset=0, axis1=0, axis2=1, dtype=None, out=None):
        return self._wrap_function(np.trace, offset, axis1, axis2, dtype, out=out)

    def var(
        self, axis=None, dtype=None, out=None, ddof=0, keepdims=False, *, where=True
    ):
        return self._wrap_function(
            np.var,
            axis,
            dtype,
            out=out,
            ddof=ddof,
            keepdims=keepdims,
            where=where,
            unit=self.unit**2,
        )

    def std(
        self, axis=None, dtype=None, out=None, ddof=0, keepdims=False, *, where=True
    ):
        return self._wrap_function(
            np.std, axis, dtype, out=out, ddof=ddof, keepdims=keepdims, where=where
        )

    def mean(self, axis=None, dtype=None, out=None, keepdims=False, *, where=True):
        return self._wrap_function(
            np.mean, axis, dtype, out=out, keepdims=keepdims, where=where
        )

    def round(self, decimals=0, out=None):
        return self._wrap_function(np.round, decimals, out=out)

    def dot(self, b, out=None):
        result_unit = self.unit * getattr(b, "unit", dimensionless_unscaled)
        return self._wrap_function(np.dot, b, out=out, unit=result_unit)

    # Calculation: override methods that do not make sense.

    def all(self, axis=None, out=None):
        raise TypeError(
            "cannot evaluate truth value of quantities. "
            "Evaluate array with q.value.all(...)"
        )

    def any(self, axis=None, out=None):
        raise TypeError(
            "cannot evaluate truth value of quantities. "
            "Evaluate array with q.value.any(...)"
        )

    # Calculation: numpy functions that can be overridden with methods.

    def diff(self, n=1, axis=-1):
        return self._wrap_function(np.diff, n, axis)

    def ediff1d(self, to_end=None, to_begin=None):
        return self._wrap_function(np.ediff1d, to_end, to_begin)

    if NUMPY_LT_1_22:

        @deprecated("5.3", alternative="np.nansum", obj_type="method")
        def nansum(self, axis=None, out=None, keepdims=False):
            return self._wrap_function(np.nansum, axis, out=out, keepdims=keepdims)

    else:

        @deprecated("5.3", alternative="np.nansum", obj_type="method")
        def nansum(
            self, axis=None, out=None, keepdims=False, *, initial=None, where=True
        ):
            if initial is not None:
                initial = self._to_own_unit(initial)
            return self._wrap_function(
                np.nansum,
                axis,
                out=out,
                keepdims=keepdims,
                initial=initial,
                where=where,
            )
```
### 7 - astropy/coordinates/spectral_coordinate.py:

Start line: 1, End line: 36

```python
import warnings
from textwrap import indent

import numpy as np

import astropy.units as u
from astropy.constants import c
from astropy.coordinates import (
    ICRS,
    CartesianDifferential,
    CartesianRepresentation,
    SkyCoord,
)
from astropy.coordinates.baseframe import BaseCoordinateFrame, frame_transform_graph
from astropy.coordinates.spectral_quantity import SpectralQuantity
from astropy.utils.exceptions import AstropyUserWarning

__all__ = ["SpectralCoord"]


class NoVelocityWarning(AstropyUserWarning):
    pass


class NoDistanceWarning(AstropyUserWarning):
    pass


KMS = u.km / u.s
ZERO_VELOCITIES = CartesianDifferential([0, 0, 0] * KMS)

# Default distance to use for target when none is provided
DEFAULT_DISTANCE = 1e6 * u.kpc

# We don't want to run doctests in the docstrings we inherit from Quantity
__doctest_skip__ = ["SpectralCoord.*"]
```
### 8 - astropy/units/quantity.py:

Start line: 584, End line: 606

```python
class Quantity(np.ndarray):

    def __array_finalize__(self, obj):
        # Check whether super().__array_finalize should be called
        # (sadly, ndarray.__array_finalize__ is None; we cannot be sure
        # what is above us).
        super_array_finalize = super().__array_finalize__
        if super_array_finalize is not None:
            super_array_finalize(obj)

        # If we're a new object or viewing an ndarray, nothing has to be done.
        if obj is None or obj.__class__ is np.ndarray:
            return

        # If our unit is not set and obj has a valid one, use it.
        if self._unit is None:
            unit = getattr(obj, "_unit", None)
            if unit is not None:
                self._set_unit(unit)

        # Copy info if the original had `info` defined.  Because of the way the
        # DataInfo works, `'info' in obj.__dict__` is False until the
        # `info` attribute is accessed or set.
        if "info" in obj.__dict__:
            self.info = obj.info
```
### 9 - astropy/units/quantity.py:

Start line: 1163, End line: 1186

```python
class Quantity(np.ndarray):

    def __ilshift__(self, other):
        try:
            other = Unit(other, parse_strict="silent")
        except UnitTypeError:
            return NotImplemented  # try other.__rlshift__(self)

        try:
            factor = self.unit._to(other)
        except UnitConversionError:  # incompatible, or requires an Equivalency
            return NotImplemented
        except AttributeError:  # StructuredUnit does not have `_to`
            # In principle, in-place might be possible.
            return NotImplemented

        view = self.view(np.ndarray)
        try:
            view *= factor  # operates on view
        except TypeError:
            # The error is `numpy.core._exceptions._UFuncOutputCastingError`,
            # which inherits from `TypeError`.
            return NotImplemented

        self._set_unit(other)
        return self
```
### 10 - astropy/units/quantity.py:

Start line: 1330, End line: 1391

```python
class Quantity(np.ndarray):

    # __contains__ is OK

    def __bool__(self):
        """This method raises ValueError, since truthiness of quantities is ambiguous,
        especially for logarithmic units and temperatures. Use explicit comparisons.
        """
        raise ValueError(
            f"{type(self).__name__} truthiness is ambiguous, especially for logarithmic units"
            " and temperatures. Use explicit comparisons."
        )

    def __len__(self):
        if self.isscalar:
            raise TypeError(
                f"'{self.__class__.__name__}' object with a scalar value has no len()"
            )
        else:
            return len(self.value)

    # Numerical types
    def __float__(self):
        try:
            return float(self.to_value(dimensionless_unscaled))
        except (UnitsError, TypeError):
            raise TypeError(
                "only dimensionless scalar quantities can be "
                "converted to Python scalars"
            )

    def __int__(self):
        try:
            return int(self.to_value(dimensionless_unscaled))
        except (UnitsError, TypeError):
            raise TypeError(
                "only dimensionless scalar quantities can be "
                "converted to Python scalars"
            )

    def __index__(self):
        # for indices, we do not want to mess around with scaling at all,
        # so unlike for float, int, we insist here on unscaled dimensionless
        try:
            assert self.unit.is_unity()
            return self.value.__index__()
        except Exception:
            raise TypeError(
                "only integer dimensionless scalar quantities "
                "can be converted to a Python index"
            )

    # TODO: we may want to add a hook for dimensionless quantities?
    @property
    def _unitstr(self):
        if self.unit is None:
            unitstr = _UNIT_NOT_INITIALISED
        else:
            unitstr = str(self.unit)

        if unitstr:
            unitstr = " " + unitstr

        return unitstr
```
### 18 - astropy/units/quantity_helper/function_helpers.py:

Start line: 101, End line: 132

```python
UNSUPPORTED_FUNCTIONS |= {
    np.packbits, np.unpackbits, np.unravel_index,
    np.ravel_multi_index, np.ix_, np.cov, np.corrcoef,
    np.busday_count, np.busday_offset, np.datetime_as_string,
    np.is_busday, np.all, np.any, np.sometrue, np.alltrue,
}  # fmt: skip

# Could be supported if we had a natural logarithm unit.
UNSUPPORTED_FUNCTIONS |= {np.linalg.slogdet}
TBD_FUNCTIONS = {
    rfn.drop_fields, rfn.rename_fields, rfn.append_fields, rfn.join_by,
    rfn.apply_along_fields, rfn.assign_fields_by_name,
    rfn.find_duplicates, rfn.recursive_fill_fields, rfn.require_fields,
    rfn.repack_fields, rfn.stack_arrays,
}  # fmt: skip
UNSUPPORTED_FUNCTIONS |= TBD_FUNCTIONS
IGNORED_FUNCTIONS = {
    # I/O - useless for Quantity, since no way to store the unit.
    np.save, np.savez, np.savetxt, np.savez_compressed,
    # Polynomials
    np.poly, np.polyadd, np.polyder, np.polydiv, np.polyfit, np.polyint,
    np.polymul, np.polysub, np.polyval, np.roots, np.vander,
    # functions taking record arrays (which are deprecated)
    rfn.rec_append_fields, rfn.rec_drop_fields, rfn.rec_join,
}  # fmt: skip
if NUMPY_LT_1_23:
    IGNORED_FUNCTIONS |= {
        # Deprecated, removed in numpy 1.23
        np.asscalar,
        np.alen,
    }
UNSUPPORTED_FUNCTIONS |= IGNORED_FUNCTIONS
```
### 27 - astropy/units/quantity_helper/function_helpers.py:

Start line: 301, End line: 321

```python
def _as_quantity(a):
    """Convert argument to a Quantity (or raise NotImplementedError)."""
    from astropy.units import Quantity

    try:
        return Quantity(a, copy=False, subok=True)
    except Exception:
        # If we cannot convert to Quantity, we should just bail.
        raise NotImplementedError


def _as_quantities(*args):
    """Convert arguments to Quantity (or raise NotImplentedError)."""
    from astropy.units import Quantity

    try:
        # Note: this should keep the dtype the same
        return tuple(Quantity(a, copy=False, subok=True, dtype=None) for a in args)
    except Exception:
        # If we cannot convert to Quantity, we should just bail.
        raise NotImplementedError
```
### 31 - astropy/units/quantity_helper/function_helpers.py:

Start line: 37, End line: 100

```python
import functools
import operator

import numpy as np
from numpy.lib import recfunctions as rfn

from astropy.units.core import (
    UnitConversionError,
    UnitsError,
    UnitTypeError,
    dimensionless_unscaled,
)
from astropy.utils import isiterable
from astropy.utils.compat import NUMPY_LT_1_23

# In 1.17, overrides are enabled by default, but it is still possible to
# turn them off using an environment variable.  We use getattr since it
# is planned to remove that possibility in later numpy versions.
ARRAY_FUNCTION_ENABLED = getattr(np.core.overrides, "ENABLE_ARRAY_FUNCTION", True)
SUBCLASS_SAFE_FUNCTIONS = set()
"""Functions with implementations supporting subclasses like Quantity."""
FUNCTION_HELPERS = {}
"""Functions with implementations usable with proper unit conversion."""
DISPATCHED_FUNCTIONS = {}
"""Functions for which we provide our own implementation."""
UNSUPPORTED_FUNCTIONS = set()
"""Functions that cannot sensibly be used with quantities."""
SUBCLASS_SAFE_FUNCTIONS |= {
    np.shape, np.size, np.ndim,
    np.reshape, np.ravel, np.moveaxis, np.rollaxis, np.swapaxes,
    np.transpose, np.atleast_1d, np.atleast_2d, np.atleast_3d,
    np.expand_dims, np.squeeze, np.broadcast_to, np.broadcast_arrays,
    np.flip, np.fliplr, np.flipud, np.rot90,
    np.argmin, np.argmax, np.argsort, np.lexsort, np.searchsorted,
    np.nonzero, np.argwhere, np.flatnonzero,
    np.diag_indices_from, np.triu_indices_from, np.tril_indices_from,
    np.real, np.imag, np.diagonal, np.diagflat, np.empty_like,
    np.compress, np.extract, np.delete, np.trim_zeros, np.roll, np.take,
    np.put, np.fill_diagonal, np.tile, np.repeat,
    np.split, np.array_split, np.hsplit, np.vsplit, np.dsplit,
    np.stack, np.column_stack, np.hstack, np.vstack, np.dstack,
    np.amax, np.amin, np.ptp, np.sum, np.cumsum,
    np.prod, np.product, np.cumprod, np.cumproduct,
    np.round, np.around,
    np.fix, np.angle, np.i0, np.clip,
    np.isposinf, np.isneginf, np.isreal, np.iscomplex,
    np.average, np.mean, np.std, np.var, np.median, np.trace,
    np.nanmax, np.nanmin, np.nanargmin, np.nanargmax, np.nanmean,
    np.nanmedian, np.nansum, np.nancumsum, np.nanstd, np.nanvar,
    np.nanprod, np.nancumprod,
    np.einsum_path, np.trapz, np.linspace,
    np.sort, np.msort, np.partition, np.meshgrid,
    np.common_type, np.result_type, np.can_cast, np.min_scalar_type,
    np.iscomplexobj, np.isrealobj,
    np.shares_memory, np.may_share_memory,
    np.apply_along_axis, np.take_along_axis, np.put_along_axis,
    np.linalg.cond, np.linalg.multi_dot,
}  # fmt: skip

# Implemented as methods on Quantity:
# np.ediff1d is from setops, but we support it anyway; the others
# currently return NotImplementedError.
# TODO: move latter to UNSUPPORTED? Would raise TypeError instead.
SUBCLASS_SAFE_FUNCTIONS |= {np.ediff1d}
```
### 33 - astropy/units/quantity_helper/function_helpers.py:

Start line: 516, End line: 542

```python
@function_helper
def where(condition, *args):
    from astropy.units import Quantity

    if isinstance(condition, Quantity) or len(args) != 2:
        raise NotImplementedError

    args, unit = _quantities2arrays(*args)
    return (condition,) + args, {}, unit, None


@function_helper(helps=({np.quantile, np.nanquantile}))
def quantile(a, q, *args, _q_unit=dimensionless_unscaled, **kwargs):
    if len(args) >= 2:
        out = args[1]
        args = args[:1] + args[2:]
    else:
        out = kwargs.pop("out", None)

    from astropy.units import Quantity

    if isinstance(q, Quantity):
        q = q.to_value(_q_unit)

    (a,), kwargs, unit, out = _iterable_helper(a, out=out, **kwargs)

    return (a, q) + args, kwargs, unit, out
```
### 42 - astropy/units/quantity_helper/function_helpers.py:

Start line: 545, End line: 565

```python
@function_helper(helps={np.percentile, np.nanpercentile})
def percentile(a, q, *args, **kwargs):
    from astropy.units import percent

    return quantile(a, q, *args, _q_unit=percent, **kwargs)


@function_helper
def count_nonzero(a, *args, **kwargs):
    return (a.value,) + args, kwargs, None, None


@function_helper(helps={np.isclose, np.allclose})
def close(a, b, rtol=1e-05, atol=1e-08, *args, **kwargs):
    from astropy.units import Quantity

    (a, b), unit = _quantities2arrays(a, b, unit_from_first=True)
    # Allow number without a unit as having the unit.
    atol = Quantity(atol, unit).value

    return (a, b, rtol, atol) + args, kwargs, None, None
```
### 44 - astropy/units/quantity_helper/function_helpers.py:

Start line: 185, End line: 271

```python
@function_helper(helps={np.tril, np.triu})
def invariant_m_helper(m, *args, **kwargs):
    return (m.view(np.ndarray),) + args, kwargs, m.unit, None


@function_helper(helps={np.fft.fftshift, np.fft.ifftshift})
def invariant_x_helper(x, *args, **kwargs):
    return (x.view(np.ndarray),) + args, kwargs, x.unit, None


# Note that ones_like does *not* work by default since if one creates an empty
# array with a unit, one cannot just fill it with unity.  Indeed, in this
# respect, it is a bit of an odd function for Quantity. On the other hand, it
# matches the idea that a unit is the same as the quantity with that unit and
# value of 1. Also, it used to work without __array_function__.
# zeros_like does work by default for regular quantities, because numpy first
# creates an empty array with the unit and then fills it with 0 (which can have
# any unit), but for structured dtype this fails (0 cannot have an arbitrary
# structured unit), so we include it here too.
@function_helper(helps={np.ones_like, np.zeros_like})
def like_helper(a, *args, **kwargs):
    subok = args[2] if len(args) > 2 else kwargs.pop("subok", True)
    unit = a.unit if subok else None
    return (a.view(np.ndarray),) + args, kwargs, unit, None


@function_helper
def sinc(x):
    from astropy.units.si import radian

    try:
        x = x.to_value(radian)
    except UnitsError:
        raise UnitTypeError(
            "Can only apply 'sinc' function to quantities with angle units"
        )
    return (x,), {}, dimensionless_unscaled, None


@dispatched_function
def unwrap(p, discont=None, axis=-1):
    from astropy.units.si import radian

    if discont is None:
        discont = np.pi << radian

    p, discont = _as_quantities(p, discont)
    result = np.unwrap.__wrapped__(
        p.to_value(radian), discont.to_value(radian), axis=axis
    )
    result = radian.to(p.unit, result)
    return result, p.unit, None


@function_helper
def argpartition(a, *args, **kwargs):
    return (a.view(np.ndarray),) + args, kwargs, None, None


@function_helper
def full_like(a, fill_value, *args, **kwargs):
    unit = a.unit if kwargs.get("subok", True) else None
    return (a.view(np.ndarray), a._to_own_unit(fill_value)) + args, kwargs, unit, None


@function_helper
def putmask(a, mask, values):
    from astropy.units import Quantity

    if isinstance(a, Quantity):
        return (a.view(np.ndarray), mask, a._to_own_unit(values)), {}, a.unit, None
    elif isinstance(values, Quantity):
        return (a, mask, values.to_value(dimensionless_unscaled)), {}, None, None
    else:
        raise NotImplementedError


@function_helper
def place(arr, mask, vals):
    from astropy.units import Quantity

    if isinstance(arr, Quantity):
        return (arr.view(np.ndarray), mask, arr._to_own_unit(vals)), {}, arr.unit, None
    elif isinstance(vals, Quantity):
        return (arr, mask, vals.to_value(dimensionless_unscaled)), {}, None, None
    else:
        raise NotImplementedError
```
### 51 - astropy/units/quantity_helper/function_helpers.py:

Start line: 568, End line: 614

```python
@dispatched_function
def array_equal(a1, a2, equal_nan=False):
    try:
        args, unit = _quantities2arrays(a1, a2)
    except UnitConversionError:
        return False, None, None
    return np.array_equal(*args, equal_nan=equal_nan), None, None


@dispatched_function
def array_equiv(a1, a2):
    try:
        args, unit = _quantities2arrays(a1, a2)
    except UnitConversionError:
        return False, None, None
    return np.array_equiv(*args), None, None


@function_helper(helps={np.dot, np.outer})
def dot_like(a, b, out=None):
    from astropy.units import Quantity

    a, b = _as_quantities(a, b)
    unit = a.unit * b.unit
    if out is not None:
        if not isinstance(out, Quantity):
            raise NotImplementedError
        return tuple(x.view(np.ndarray) for x in (a, b, out)), {}, unit, out
    else:
        return (a.view(np.ndarray), b.view(np.ndarray)), {}, unit, None


@function_helper(
    helps={
        np.cross,
        np.inner,
        np.vdot,
        np.tensordot,
        np.kron,
        np.correlate,
        np.convolve,
    }
)
def cross_like(a, b, *args, **kwargs):
    a, b = _as_quantities(a, b)
    unit = a.unit * b.unit
    return (a.view(np.ndarray), b.view(np.ndarray)) + args, kwargs, unit, None
```
### 55 - astropy/units/quantity_helper/function_helpers.py:

Start line: 274, End line: 298

```python
@function_helper
def copyto(dst, src, *args, **kwargs):
    from astropy.units import Quantity

    if isinstance(dst, Quantity):
        return (dst.view(np.ndarray), dst._to_own_unit(src)) + args, kwargs, None, None
    elif isinstance(src, Quantity):
        return (dst, src.to_value(dimensionless_unscaled)) + args, kwargs, None, None
    else:
        raise NotImplementedError


@function_helper
def nan_to_num(x, copy=True, nan=0.0, posinf=None, neginf=None):
    nan = x._to_own_unit(nan)
    if posinf is not None:
        posinf = x._to_own_unit(posinf)
    if neginf is not None:
        neginf = x._to_own_unit(neginf)
    return (
        (x.view(np.ndarray),),
        dict(copy=True, nan=nan, posinf=posinf, neginf=neginf),
        x.unit,
        None,
    )
```
### 58 - astropy/units/quantity_helper/function_helpers.py:

Start line: 362, End line: 383

```python
def _iterable_helper(*args, out=None, **kwargs):
    """Convert arguments to Quantity, and treat possible 'out'."""
    from astropy.units import Quantity

    if out is not None:
        if isinstance(out, Quantity):
            kwargs["out"] = out.view(np.ndarray)
        else:
            # TODO: for an ndarray output, we could in principle
            # try converting all Quantity to dimensionless.
            raise NotImplementedError

    arrays, unit = _quantities2arrays(*args)
    return arrays, kwargs, unit, out


@function_helper
def concatenate(arrays, axis=0, out=None, **kwargs):
    # TODO: make this smarter by creating an appropriately shaped
    # empty output array and just filling it.
    arrays, kwargs, unit, out = _iterable_helper(*arrays, out=out, axis=axis, **kwargs)
    return (arrays,), kwargs, unit, out
```
### 60 - astropy/units/quantity_helper/function_helpers.py:

Start line: 164, End line: 182

```python
function_helper = FunctionAssigner(FUNCTION_HELPERS)

dispatched_function = FunctionAssigner(DISPATCHED_FUNCTIONS)


# fmt: off
@function_helper(
    helps={
        np.copy, np.asfarray, np.real_if_close, np.sort_complex, np.resize,
        np.fft.fft, np.fft.ifft, np.fft.rfft, np.fft.irfft,
        np.fft.fft2, np.fft.ifft2, np.fft.rfft2, np.fft.irfft2,
        np.fft.fftn, np.fft.ifftn, np.fft.rfftn, np.fft.irfftn,
        np.fft.hfft, np.fft.ihfft,
        np.linalg.eigvals, np.linalg.eigvalsh,
    }
)
# fmt: on
def invariant_a_helper(a, *args, **kwargs):
    return (a.view(np.ndarray),) + args, kwargs, a.unit, None
```
### 63 - astropy/units/quantity_helper/function_helpers.py:

Start line: 1004, End line: 1117

```python
@function_helper
def diag(v, *args, **kwargs):
    # Function works for *getting* the diagonal, but not *setting*.
    # So, override always.
    return (v.value,) + args, kwargs, v.unit, None


@function_helper(module=np.linalg)
def svd(a, full_matrices=True, compute_uv=True, hermitian=False):
    unit = a.unit
    if compute_uv:
        unit = (None, unit, None)

    return ((a.view(np.ndarray), full_matrices, compute_uv, hermitian), {}, unit, None)


def _interpret_tol(tol, unit):
    from astropy.units import Quantity

    return Quantity(tol, unit).value


@function_helper(module=np.linalg)
def matrix_rank(M, tol=None, *args, **kwargs):
    if tol is not None:
        tol = _interpret_tol(tol, M.unit)

    return (M.view(np.ndarray), tol) + args, kwargs, None, None


@function_helper(helps={np.linalg.inv, np.linalg.tensorinv})
def inv(a, *args, **kwargs):
    return (a.view(np.ndarray),) + args, kwargs, 1 / a.unit, None


@function_helper(module=np.linalg)
def pinv(a, rcond=1e-15, *args, **kwargs):
    rcond = _interpret_tol(rcond, a.unit)

    return (a.view(np.ndarray), rcond) + args, kwargs, 1 / a.unit, None


@function_helper(module=np.linalg)
def det(a):
    return (a.view(np.ndarray),), {}, a.unit ** a.shape[-1], None


@function_helper(helps={np.linalg.solve, np.linalg.tensorsolve})
def solve(a, b, *args, **kwargs):
    a, b = _as_quantities(a, b)

    return (
        (a.view(np.ndarray), b.view(np.ndarray)) + args,
        kwargs,
        b.unit / a.unit,
        None,
    )


@function_helper(module=np.linalg)
def lstsq(a, b, rcond="warn"):
    a, b = _as_quantities(a, b)

    if rcond not in (None, "warn", -1):
        rcond = _interpret_tol(rcond, a.unit)

    return (
        (a.view(np.ndarray), b.view(np.ndarray), rcond),
        {},
        (b.unit / a.unit, b.unit**2, None, a.unit),
        None,
    )


@function_helper(module=np.linalg)
def norm(x, ord=None, *args, **kwargs):
    if ord == 0:
        from astropy.units import dimensionless_unscaled

        unit = dimensionless_unscaled
    else:
        unit = x.unit
    return (x.view(np.ndarray), ord) + args, kwargs, unit, None


@function_helper(module=np.linalg)
def matrix_power(a, n):
    return (a.value, n), {}, a.unit**n, None


@function_helper(module=np.linalg)
def cholesky(a):
    return (a.value,), {}, a.unit**0.5, None


@function_helper(module=np.linalg)
def qr(a, mode="reduced"):
    if mode.startswith("e"):
        units = None
    elif mode == "r":
        units = a.unit
    else:
        from astropy.units import dimensionless_unscaled

        units = (dimensionless_unscaled, a.unit)

    return (a.value, mode), {}, units, None


@function_helper(helps={np.linalg.eig, np.linalg.eigh})
def eig(a, *args, **kwargs):
    from astropy.units import dimensionless_unscaled

    return (a.value,) + args, kwargs, (a.unit, dimensionless_unscaled), None
```
### 69 - astropy/units/quantity_helper/function_helpers.py:

Start line: 913, End line: 940

```python
@dispatched_function
def apply_over_axes(func, a, axes):
    # Copied straight from numpy/lib/shape_base, just to omit its
    # val = asarray(a); if only it had been asanyarray, or just not there
    # since a is assumed to an an array in the next line...
    # Which is what we do here - we can only get here if it is a Quantity.
    val = a
    N = a.ndim
    if np.array(axes).ndim == 0:
        axes = (axes,)
    for axis in axes:
        if axis < 0:
            axis = N + axis
        args = (val, axis)
        res = func(*args)
        if res.ndim == val.ndim:
            val = res
        else:
            res = np.expand_dims(res, axis)
            if res.ndim == val.ndim:
                val = res
            else:
                raise ValueError(
                    "function is not returning an array of the correct shape"
                )
    # Returning unit is None to signal nothing should happen to
    # the output.
    return val, None, None
```
### 70 - astropy/units/quantity_helper/function_helpers.py:

Start line: 637, End line: 668

```python
@function_helper
def bincount(x, weights=None, minlength=0):
    from astropy.units import Quantity

    if isinstance(x, Quantity):
        raise NotImplementedError
    return (x, weights.value, minlength), {}, weights.unit, None


@function_helper
def digitize(x, bins, *args, **kwargs):
    arrays, unit = _quantities2arrays(x, bins, unit_from_first=True)
    return arrays + args, kwargs, None, None


def _check_bins(bins, unit):
    from astropy.units import Quantity

    check = _as_quantity(bins)
    if check.ndim > 0:
        return check.to_value(unit)
    elif isinstance(bins, Quantity):
        # bins should be an integer (or at least definitely not a Quantity).
        raise NotImplementedError
    else:
        return bins


def _check_range(range, unit):
    range = _as_quantity(range)
    range = range.to_value(unit)
    return range
```
### 79 - astropy/units/quantity_helper/function_helpers.py:

Start line: 844, End line: 910

```python
@function_helper
def logspace(start, stop, *args, **kwargs):
    from astropy.units import LogQuantity, dex

    if not isinstance(start, LogQuantity) or not isinstance(stop, LogQuantity):
        raise NotImplementedError

    # Get unit from end point as for linspace.
    stop = stop.to(dex(stop.unit.physical_unit))
    start = start.to(stop.unit)
    unit = stop.unit.physical_unit
    return (start.value, stop.value) + args, kwargs, unit, None


@function_helper
def geomspace(start, stop, *args, **kwargs):
    # Get unit from end point as for linspace.
    (stop, start), unit = _quantities2arrays(stop, start)
    return (start, stop) + args, kwargs, unit, None


@function_helper
def interp(x, xp, fp, *args, **kwargs):
    from astropy.units import Quantity

    (x, xp), _ = _quantities2arrays(x, xp)
    if isinstance(fp, Quantity):
        unit = fp.unit
        fp = fp.value
    else:
        unit = None

    return (x, xp, fp) + args, kwargs, unit, None


@function_helper
def unique(
    ar, return_index=False, return_inverse=False, return_counts=False, axis=None
):
    unit = ar.unit
    n_index = sum(bool(i) for i in (return_index, return_inverse, return_counts))
    if n_index:
        unit = [unit] + n_index * [None]

    return (ar.value, return_index, return_inverse, return_counts, axis), {}, unit, None


@function_helper
def intersect1d(ar1, ar2, assume_unique=False, return_indices=False):
    (ar1, ar2), unit = _quantities2arrays(ar1, ar2)
    if return_indices:
        unit = [unit, None, None]
    return (ar1, ar2, assume_unique, return_indices), {}, unit, None


@function_helper(helps=(np.setxor1d, np.union1d, np.setdiff1d))
def twosetop(ar1, ar2, *args, **kwargs):
    (ar1, ar2), unit = _quantities2arrays(ar1, ar2)
    return (ar1, ar2) + args, kwargs, unit, None


@function_helper(helps=(np.isin, np.in1d))
def setcheckop(ar1, ar2, *args, **kwargs):
    # This tests whether ar1 is in ar2, so we should change the unit of
    # a1 to that of a2.
    (ar2, ar1), unit = _quantities2arrays(ar2, ar1)
    return (ar1, ar2) + args, kwargs, None, None
```
### 100 - astropy/units/quantity_helper/function_helpers.py:

Start line: 671, End line: 708

```python
@function_helper
def histogram(a, bins=10, range=None, weights=None, density=None):
    if weights is not None:
        weights = _as_quantity(weights)
        unit = weights.unit
        weights = weights.value
    else:
        unit = None

    a = _as_quantity(a)
    if not isinstance(bins, str):
        bins = _check_bins(bins, a.unit)

    if range is not None:
        range = _check_range(range, a.unit)

    if density:
        unit = (unit or 1) / a.unit

    return (
        (a.value, bins, range),
        {"weights": weights, "density": density},
        (unit, a.unit),
        None,
    )


@function_helper(helps=np.histogram_bin_edges)
def histogram_bin_edges(a, bins=10, range=None, weights=None):
    # weights is currently unused
    a = _as_quantity(a)
    if not isinstance(bins, str):
        bins = _check_bins(bins, a.unit)

    if range is not None:
        range = _check_range(range, a.unit)

    return (a.value, bins, range, weights), {}, a.unit, None
```
