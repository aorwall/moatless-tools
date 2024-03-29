# astropy__astropy-12057

| **astropy/astropy** | `b6769c18c0881b6d290e543e9334c25043018b3f` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | 1495 |
| **Avg pos** | 5.0 |
| **Min pos** | 2 |
| **Max pos** | 3 |
| **Top file pos** | 1 |
| **Missing snippets** | 4 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/astropy/nddata/nduncertainty.py b/astropy/nddata/nduncertainty.py
--- a/astropy/nddata/nduncertainty.py
+++ b/astropy/nddata/nduncertainty.py
@@ -395,6 +395,40 @@ def _propagate_multiply(self, other_uncert, result_data, correlation):
     def _propagate_divide(self, other_uncert, result_data, correlation):
         return None
 
+    def represent_as(self, other_uncert):
+        """Convert this uncertainty to a different uncertainty type.
+
+        Parameters
+        ----------
+        other_uncert : `NDUncertainty` subclass
+            The `NDUncertainty` subclass to convert to.
+
+        Returns
+        -------
+        resulting_uncertainty : `NDUncertainty` instance
+            An instance of ``other_uncert`` subclass containing the uncertainty
+            converted to the new uncertainty type.
+
+        Raises
+        ------
+        TypeError
+            If either the initial or final subclasses do not support
+            conversion, a `TypeError` is raised.
+        """
+        as_variance = getattr(self, "_convert_to_variance", None)
+        if as_variance is None:
+            raise TypeError(
+                f"{type(self)} does not support conversion to another "
+                "uncertainty type."
+            )
+        from_variance = getattr(other_uncert, "_convert_from_variance", None)
+        if from_variance is None:
+            raise TypeError(
+                f"{other_uncert.__name__} does not support conversion from "
+                "another uncertainty type."
+            )
+        return from_variance(as_variance())
+
 
 class UnknownUncertainty(NDUncertainty):
     """This class implements any unknown uncertainty type.
@@ -748,6 +782,17 @@ def _propagate_divide(self, other_uncert, result_data, correlation):
     def _data_unit_to_uncertainty_unit(self, value):
         return value
 
+    def _convert_to_variance(self):
+        new_array = None if self.array is None else self.array ** 2
+        new_unit = None if self.unit is None else self.unit ** 2
+        return VarianceUncertainty(new_array, unit=new_unit)
+
+    @classmethod
+    def _convert_from_variance(cls, var_uncert):
+        new_array = None if var_uncert.array is None else var_uncert.array ** (1 / 2)
+        new_unit = None if var_uncert.unit is None else var_uncert.unit ** (1 / 2)
+        return cls(new_array, unit=new_unit)
+
 
 class VarianceUncertainty(_VariancePropagationMixin, NDUncertainty):
     """
@@ -834,6 +879,13 @@ def _propagate_divide(self, other_uncert, result_data, correlation):
     def _data_unit_to_uncertainty_unit(self, value):
         return value ** 2
 
+    def _convert_to_variance(self):
+        return self
+
+    @classmethod
+    def _convert_from_variance(cls, var_uncert):
+        return var_uncert
+
 
 def _inverse(x):
     """Just a simple inverse for use in the InverseVariance"""
@@ -933,3 +985,14 @@ def _propagate_divide(self, other_uncert, result_data, correlation):
 
     def _data_unit_to_uncertainty_unit(self, value):
         return 1 / value ** 2
+
+    def _convert_to_variance(self):
+        new_array = None if self.array is None else 1 / self.array
+        new_unit = None if self.unit is None else 1 / self.unit
+        return VarianceUncertainty(new_array, unit=new_unit)
+
+    @classmethod
+    def _convert_from_variance(cls, var_uncert):
+        new_array = None if var_uncert.array is None else 1 / var_uncert.array
+        new_unit = None if var_uncert.unit is None else 1 / var_uncert.unit
+        return cls(new_array, unit=new_unit)

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| astropy/nddata/nduncertainty.py | 398 | 398 | - | 1 | -
| astropy/nddata/nduncertainty.py | 751 | 751 | - | 1 | -
| astropy/nddata/nduncertainty.py | 837 | 837 | 2 | 1 | 1495
| astropy/nddata/nduncertainty.py | 936 | 936 | 3 | 1 | 2245


## Problem Statement

```
Add helpers to convert between different types of uncertainties
Currently there no easy way to convert from an arbitrary uncertainty class to a different uncertainty class. This would be useful to be able to pass NDData objects to external libraries/tools which assume, for example, that uncertainties will always stored as variances. Here's some really scrappy code I bunged together quickly for my purposes (probably buggy, I need to properly test it), but what are peoples opinions on what's the best API/design/framework for such a system?

\`\`\`python
from astropy.nddata import (
    VarianceUncertainty, StdDevUncertainty, InverseVariance,
)

def std_to_var(obj):
    return VarianceUncertainty(obj.array ** 2, unit=obj.unit ** 2)


def var_to_invvar(obj):
    return InverseVariance(obj.array ** -1, unit=obj.unit ** -1)


def invvar_to_var(obj):
    return VarianceUncertainty(obj.array ** -1, unit=obj.unit ** -1)


def var_to_std(obj):
    return VarianceUncertainty(obj.array ** 1/2, unit=obj.unit ** 1/2)


FUNC_MAP = {
    (StdDevUncertainty, VarianceUncertainty): std_to_var,
    (StdDevUncertainty, InverseVariance): lambda x: var_to_invvar(
        std_to_var(x)
    ),
    (VarianceUncertainty, StdDevUncertainty): var_to_std,
    (VarianceUncertainty, InverseVariance): var_to_invvar,
    (InverseVariance, StdDevUncertainty): lambda x: var_to_std(
        invvar_to_var(x)
    ),
    (InverseVariance, VarianceUncertainty): invvar_to_var,
    (StdDevUncertainty, StdDevUncertainty): lambda x: x,
    (VarianceUncertainty, VarianceUncertainty): lambda x: x,
    (InverseVariance, InverseVariance): lambda x: x,
}


def convert_uncertainties(obj, new_class):
    return FUNC_MAP[(type(obj), new_class)](obj)
\`\`\`

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 astropy/nddata/nduncertainty.py** | 656 | 749| 758 | 758 | 7385 | 
| **-> 2 <-** | **1 astropy/nddata/nduncertainty.py** | 752 | 840| 737 | 1495 | 7385 | 
| **-> 3 <-** | **1 astropy/nddata/nduncertainty.py** | 843 | 936| 750 | 2245 | 7385 | 
| 4 | **1 astropy/nddata/nduncertainty.py** | 399 | 449| 309 | 2554 | 7385 | 
| 5 | **1 astropy/nddata/nduncertainty.py** | 350 | 396| 339 | 2893 | 7385 | 
| 6 | **1 astropy/nddata/nduncertainty.py** | 31 | 134| 779 | 3672 | 7385 | 
| 7 | 2 astropy/nddata/nddata.py | 310 | 334| 257 | 3929 | 10126 | 
| 8 | **2 astropy/nddata/nduncertainty.py** | 204 | 235| 348 | 4277 | 10126 | 
| 9 | 2 astropy/nddata/nddata.py | 285 | 308| 225 | 4502 | 10126 | 
| 10 | **2 astropy/nddata/nduncertainty.py** | 452 | 538| 680 | 5182 | 10126 | 
| 11 | **2 astropy/nddata/nduncertainty.py** | 136 | 165| 239 | 5421 | 10126 | 
| 12 | 3 astropy/nddata/ccddata.py | 4 | 27| 213 | 5634 | 16717 | 
| 13 | **3 astropy/nddata/nduncertainty.py** | 3 | 28| 178 | 5812 | 16717 | 
| 14 | 4 astropy/nddata/compat.py | 113 | 129| 166 | 5978 | 18820 | 
| 15 | **4 astropy/nddata/nduncertainty.py** | 237 | 277| 339 | 6317 | 18820 | 
| 16 | 4 astropy/nddata/nddata.py | 119 | 235| 997 | 7314 | 18820 | 
| 17 | **4 astropy/nddata/nduncertainty.py** | 540 | 628| 781 | 8095 | 18820 | 
| 18 | **4 astropy/nddata/nduncertainty.py** | 279 | 348| 573 | 8668 | 18820 | 
| 19 | 4 astropy/nddata/ccddata.py | 251 | 270| 187 | 8855 | 18820 | 
| 20 | **4 astropy/nddata/nduncertainty.py** | 629 | 653| 349 | 9204 | 18820 | 
| 21 | 4 astropy/nddata/ccddata.py | 50 | 71| 229 | 9433 | 18820 | 
| 22 | 5 astropy/modeling/fitting.py | 101 | 136| 315 | 9748 | 35989 | 
| 23 | **5 astropy/nddata/nduncertainty.py** | 167 | 202| 338 | 10086 | 35989 | 
| 24 | 6 astropy/nddata/mixins/ndarithmetic.py | 248 | 284| 357 | 10443 | 41089 | 
| 25 | 6 astropy/nddata/compat.py | 84 | 111| 237 | 10680 | 41089 | 
| 26 | 7 astropy/uncertainty/__init__.py | 1 | 13| 72 | 10752 | 41161 | 
| 27 | 7 astropy/nddata/compat.py | 5 | 82| 697 | 11449 | 41161 | 
| 28 | 7 astropy/nddata/nddata.py | 21 | 117| 843 | 12292 | 41161 | 
| 29 | 7 astropy/nddata/mixins/ndarithmetic.py | 5 | 98| 907 | 13199 | 41161 | 
| 30 | 7 astropy/nddata/mixins/ndarithmetic.py | 101 | 162| 559 | 13758 | 41161 | 
| 31 | 7 astropy/nddata/compat.py | 245 | 291| 349 | 14107 | 41161 | 
| 32 | 8 astropy/nddata/__init__.py | 11 | 29| 102 | 14209 | 41516 | 
| 33 | 8 astropy/nddata/mixins/ndarithmetic.py | 325 | 395| 612 | 14821 | 41516 | 
| 34 | 8 astropy/nddata/nddata.py | 5 | 18| 119 | 14940 | 41516 | 
| 35 | 8 astropy/nddata/__init__.py | 32 | 53| 155 | 15095 | 41516 | 
| 36 | 8 astropy/nddata/ccddata.py | 337 | 404| 730 | 15825 | 41516 | 
| 37 | 9 astropy/units/quantity_helper/helpers.py | 323 | 382| 785 | 16610 | 45329 | 
| 38 | 10 astropy/nddata/mixins/ndslicing.py | 98 | 133| 291 | 16901 | 46382 | 
| 39 | 10 astropy/nddata/compat.py | 131 | 154| 185 | 17086 | 46382 | 
| 40 | 11 astropy/time/core.py | 1 | 79| 956 | 18042 | 73023 | 
| 41 | 12 astropy/units/quantity_helper/converters.py | 1 | 29| 189 | 18231 | 76278 | 
| 42 | 13 astropy/constants/codata2010.py | 7 | 70| 754 | 18985 | 77633 | 
| 43 | 14 astropy/units/quantity_helper/function_helpers.py | 101 | 141| 482 | 19467 | 87072 | 
| 44 | 14 astropy/units/quantity_helper/helpers.py | 383 | 437| 665 | 20132 | 87072 | 
| 45 | 15 astropy/time/formats.py | 1 | 45| 443 | 20575 | 106242 | 
| 46 | 16 astropy/uncertainty/core.py | 329 | 349| 158 | 20733 | 108941 | 
| 47 | 16 astropy/units/quantity_helper/function_helpers.py | 1 | 100| 764 | 21497 | 108941 | 
| 48 | 17 astropy/nddata/nddata_base.py | 5 | 77| 376 | 21873 | 109345 | 
| 49 | 18 astropy/constants/codata2014.py | 69 | 105| 510 | 22383 | 110641 | 
| 50 | 19 astropy/io/fits/column.py | 3 | 77| 761 | 23144 | 132978 | 
| 51 | 20 astropy/units/cds.py | 50 | 136| 1282 | 24426 | 135126 | 
| 52 | 21 astropy/constants/astropyconst40.py | 7 | 50| 483 | 24909 | 135664 | 
| 53 | 22 astropy/units/core.py | 1025 | 1061| 290 | 25199 | 154449 | 
| 54 | 22 astropy/nddata/ccddata.py | 74 | 180| 1005 | 26204 | 154449 | 
| 55 | 23 astropy/units/equivalencies.py | 4 | 24| 201 | 26405 | 163670 | 
| 56 | 24 astropy/constants/astropyconst20.py | 7 | 50| 483 | 26888 | 164208 | 
| 57 | 24 astropy/nddata/nddata.py | 237 | 283| 296 | 27184 | 164208 | 
| 58 | 24 astropy/units/quantity_helper/converters.py | 97 | 112| 143 | 27327 | 164208 | 
| 59 | 24 astropy/constants/codata2010.py | 72 | 110| 551 | 27878 | 164208 | 
| 60 | 25 astropy/nddata/decorators.py | 145 | 279| 1217 | 29095 | 166467 | 
| 61 | 26 astropy/units/quantity_helper/scipy_special.py | 1 | 61| 719 | 29814 | 167488 | 
| 62 | 27 astropy/units/quantity_helper/__init__.py | 8 | 17| 82 | 29896 | 167619 | 
| 63 | 28 astropy/constants/iau2012.py | 7 | 77| 769 | 30665 | 168438 | 
| 64 | 28 astropy/time/core.py | 452 | 509| 555 | 31220 | 168438 | 
| 65 | 29 astropy/utils/iers/iers.py | 12 | 78| 762 | 31982 | 180051 | 
| 66 | 30 astropy/modeling/math_functions.py | 5 | 42| 359 | 32341 | 180741 | 
| 67 | 31 astropy/cosmology/core.py | 3 | 44| 213 | 32554 | 185256 | 


### Hint

```
See also #10128 which is maybe not exactly the same need but related in the sense that there is currently no easy way to get uncertainties in a specific format (variance, std).
Very much from the left field, but in coordinate representations, we deal with this by insisting every representation can be transformed to/from cartesian, and then have a `represent_as` method that by default goes through cartesian. A similar scheme (probably going through variance) might well be possible here.
It sounds like the `represent_as` method via variance would be reasonable, I'll see if I can spend some time coding something up (but if someone else wants to have a go, don't let me stop you).
```

## Patch

```diff
diff --git a/astropy/nddata/nduncertainty.py b/astropy/nddata/nduncertainty.py
--- a/astropy/nddata/nduncertainty.py
+++ b/astropy/nddata/nduncertainty.py
@@ -395,6 +395,40 @@ def _propagate_multiply(self, other_uncert, result_data, correlation):
     def _propagate_divide(self, other_uncert, result_data, correlation):
         return None
 
+    def represent_as(self, other_uncert):
+        """Convert this uncertainty to a different uncertainty type.
+
+        Parameters
+        ----------
+        other_uncert : `NDUncertainty` subclass
+            The `NDUncertainty` subclass to convert to.
+
+        Returns
+        -------
+        resulting_uncertainty : `NDUncertainty` instance
+            An instance of ``other_uncert`` subclass containing the uncertainty
+            converted to the new uncertainty type.
+
+        Raises
+        ------
+        TypeError
+            If either the initial or final subclasses do not support
+            conversion, a `TypeError` is raised.
+        """
+        as_variance = getattr(self, "_convert_to_variance", None)
+        if as_variance is None:
+            raise TypeError(
+                f"{type(self)} does not support conversion to another "
+                "uncertainty type."
+            )
+        from_variance = getattr(other_uncert, "_convert_from_variance", None)
+        if from_variance is None:
+            raise TypeError(
+                f"{other_uncert.__name__} does not support conversion from "
+                "another uncertainty type."
+            )
+        return from_variance(as_variance())
+
 
 class UnknownUncertainty(NDUncertainty):
     """This class implements any unknown uncertainty type.
@@ -748,6 +782,17 @@ def _propagate_divide(self, other_uncert, result_data, correlation):
     def _data_unit_to_uncertainty_unit(self, value):
         return value
 
+    def _convert_to_variance(self):
+        new_array = None if self.array is None else self.array ** 2
+        new_unit = None if self.unit is None else self.unit ** 2
+        return VarianceUncertainty(new_array, unit=new_unit)
+
+    @classmethod
+    def _convert_from_variance(cls, var_uncert):
+        new_array = None if var_uncert.array is None else var_uncert.array ** (1 / 2)
+        new_unit = None if var_uncert.unit is None else var_uncert.unit ** (1 / 2)
+        return cls(new_array, unit=new_unit)
+
 
 class VarianceUncertainty(_VariancePropagationMixin, NDUncertainty):
     """
@@ -834,6 +879,13 @@ def _propagate_divide(self, other_uncert, result_data, correlation):
     def _data_unit_to_uncertainty_unit(self, value):
         return value ** 2
 
+    def _convert_to_variance(self):
+        return self
+
+    @classmethod
+    def _convert_from_variance(cls, var_uncert):
+        return var_uncert
+
 
 def _inverse(x):
     """Just a simple inverse for use in the InverseVariance"""
@@ -933,3 +985,14 @@ def _propagate_divide(self, other_uncert, result_data, correlation):
 
     def _data_unit_to_uncertainty_unit(self, value):
         return 1 / value ** 2
+
+    def _convert_to_variance(self):
+        new_array = None if self.array is None else 1 / self.array
+        new_unit = None if self.unit is None else 1 / self.unit
+        return VarianceUncertainty(new_array, unit=new_unit)
+
+    @classmethod
+    def _convert_from_variance(cls, var_uncert):
+        new_array = None if var_uncert.array is None else 1 / var_uncert.array
+        new_unit = None if var_uncert.unit is None else 1 / var_uncert.unit
+        return cls(new_array, unit=new_unit)

```

## Test Patch

```diff
diff --git a/astropy/nddata/tests/test_nduncertainty.py b/astropy/nddata/tests/test_nduncertainty.py
--- a/astropy/nddata/tests/test_nduncertainty.py
+++ b/astropy/nddata/tests/test_nduncertainty.py
@@ -4,7 +4,7 @@
 
 import pytest
 import numpy as np
-from numpy.testing import assert_array_equal
+from numpy.testing import assert_array_equal, assert_allclose
 
 from astropy.nddata.nduncertainty import (StdDevUncertainty,
                              VarianceUncertainty,
@@ -73,6 +73,11 @@ def _propagate_divide(self, data, final_data):
     UnknownUncertainty
 ]
 
+uncertainty_types_with_conversion_support = (
+    StdDevUncertainty, VarianceUncertainty, InverseVariance)
+uncertainty_types_without_conversion_support = (
+    FakeUncertainty, UnknownUncertainty)
+
 
 @pytest.mark.parametrize(('UncertClass'), uncertainty_types_to_be_tested)
 def test_init_fake_with_list(UncertClass):
@@ -354,3 +359,35 @@ def test_assigning_uncertainty_with_bad_unit_to_parent_fails(NDClass,
     v = UncertClass([1, 1], unit=u.second)
     with pytest.raises(u.UnitConversionError):
         ndd.uncertainty = v
+
+
+@pytest.mark.parametrize('UncertClass', uncertainty_types_with_conversion_support)
+def test_self_conversion_via_variance_supported(UncertClass):
+    uncert = np.arange(1, 11).reshape(2, 5) * u.adu
+    start_uncert = UncertClass(uncert)
+    final_uncert = start_uncert.represent_as(UncertClass)
+    assert_array_equal(start_uncert.array, final_uncert.array)
+    assert start_uncert.unit == final_uncert.unit
+
+
+@pytest.mark.parametrize(
+    'UncertClass,to_variance_func',
+    zip(uncertainty_types_with_conversion_support,
+    (lambda x: x ** 2, lambda x: x, lambda x: 1 / x))
+)
+def test_conversion_to_from_variance_supported(UncertClass, to_variance_func):
+    uncert = np.arange(1, 11).reshape(2, 5) * u.adu
+    start_uncert = UncertClass(uncert)
+    var_uncert = start_uncert.represent_as(VarianceUncertainty)
+    final_uncert = var_uncert.represent_as(UncertClass)
+    assert_allclose(to_variance_func(start_uncert.array), var_uncert.array)
+    assert_array_equal(start_uncert.array, final_uncert.array)
+    assert start_uncert.unit == final_uncert.unit
+
+
+@pytest.mark.parametrize('UncertClass', uncertainty_types_without_conversion_support)
+def test_self_conversion_via_variance_not_supported(UncertClass):
+    uncert = np.arange(1, 11).reshape(2, 5) * u.adu
+    start_uncert = UncertClass(uncert)
+    with pytest.raises(TypeError):
+        final_uncert = start_uncert.represent_as(UncertClass)

```


## Code snippets

### 1 - astropy/nddata/nduncertainty.py:

Start line: 656, End line: 749

```python
class StdDevUncertainty(_VariancePropagationMixin, NDUncertainty):
    """Standard deviation uncertainty assuming first order gaussian error
    propagation.

    This class implements uncertainty propagation for ``addition``,
    ``subtraction``, ``multiplication`` and ``division`` with other instances
    of `StdDevUncertainty`. The class can handle if the uncertainty has a
    unit that differs from (but is convertible to) the parents `NDData` unit.
    The unit of the resulting uncertainty will have the same unit as the
    resulting data. Also support for correlation is possible but requires the
    correlation as input. It cannot handle correlation determination itself.

    Parameters
    ----------
    args, kwargs :
        see `NDUncertainty`

    Examples
    --------
    `StdDevUncertainty` should always be associated with an `NDData`-like
    instance, either by creating it during initialization::

        >>> from astropy.nddata import NDData, StdDevUncertainty
        >>> ndd = NDData([1,2,3], unit='m',
        ...              uncertainty=StdDevUncertainty([0.1, 0.1, 0.1]))
        >>> ndd.uncertainty  # doctest: +FLOAT_CMP
        StdDevUncertainty([0.1, 0.1, 0.1])

    or by setting it manually on the `NDData` instance::

        >>> ndd.uncertainty = StdDevUncertainty([0.2], unit='m', copy=True)
        >>> ndd.uncertainty  # doctest: +FLOAT_CMP
        StdDevUncertainty([0.2])

    the uncertainty ``array`` can also be set directly::

        >>> ndd.uncertainty.array = 2
        >>> ndd.uncertainty
        StdDevUncertainty(2)

    .. note::
        The unit will not be displayed.
    """

    @property
    def supports_correlated(self):
        """`True` : `StdDevUncertainty` allows to propagate correlated \
                    uncertainties.

        ``correlation`` must be given, this class does not implement computing
        it by itself.
        """
        return True

    @property
    def uncertainty_type(self):
        """``"std"`` : `StdDevUncertainty` implements standard deviation.
        """
        return 'std'

    def _convert_uncertainty(self, other_uncert):
        if isinstance(other_uncert, StdDevUncertainty):
            return other_uncert
        else:
            raise IncompatibleUncertaintiesException

    def _propagate_add(self, other_uncert, result_data, correlation):
        return super()._propagate_add_sub(other_uncert, result_data,
                                          correlation, subtract=False,
                                          to_variance=np.square,
                                          from_variance=np.sqrt)

    def _propagate_subtract(self, other_uncert, result_data, correlation):
        return super()._propagate_add_sub(other_uncert, result_data,
                                          correlation, subtract=True,
                                          to_variance=np.square,
                                          from_variance=np.sqrt)

    def _propagate_multiply(self, other_uncert, result_data, correlation):
        return super()._propagate_multiply_divide(other_uncert,
                                                  result_data, correlation,
                                                  divide=False,
                                                  to_variance=np.square,
                                                  from_variance=np.sqrt)

    def _propagate_divide(self, other_uncert, result_data, correlation):
        return super()._propagate_multiply_divide(other_uncert,
                                                  result_data, correlation,
                                                  divide=True,
                                                  to_variance=np.square,
                                                  from_variance=np.sqrt)

    def _data_unit_to_uncertainty_unit(self, value):
        return value
```
### 2 - astropy/nddata/nduncertainty.py:

Start line: 752, End line: 840

```python
class VarianceUncertainty(_VariancePropagationMixin, NDUncertainty):
    """
    Variance uncertainty assuming first order Gaussian error
    propagation.

    This class implements uncertainty propagation for ``addition``,
    ``subtraction``, ``multiplication`` and ``division`` with other instances
    of `VarianceUncertainty`. The class can handle if the uncertainty has a
    unit that differs from (but is convertible to) the parents `NDData` unit.
    The unit of the resulting uncertainty will be the square of the unit of the
    resulting data. Also support for correlation is possible but requires the
    correlation as input. It cannot handle correlation determination itself.

    Parameters
    ----------
    args, kwargs :
        see `NDUncertainty`

    Examples
    --------
    Compare this example to that in `StdDevUncertainty`; the uncertainties
    in the examples below are equivalent to the uncertainties in
    `StdDevUncertainty`.

    `VarianceUncertainty` should always be associated with an `NDData`-like
    instance, either by creating it during initialization::

        >>> from astropy.nddata import NDData, VarianceUncertainty
        >>> ndd = NDData([1,2,3], unit='m',
        ...              uncertainty=VarianceUncertainty([0.01, 0.01, 0.01]))
        >>> ndd.uncertainty  # doctest: +FLOAT_CMP
        VarianceUncertainty([0.01, 0.01, 0.01])

    or by setting it manually on the `NDData` instance::

        >>> ndd.uncertainty = VarianceUncertainty([0.04], unit='m^2', copy=True)
        >>> ndd.uncertainty  # doctest: +FLOAT_CMP
        VarianceUncertainty([0.04])

    the uncertainty ``array`` can also be set directly::

        >>> ndd.uncertainty.array = 4
        >>> ndd.uncertainty
        VarianceUncertainty(4)

    .. note::
        The unit will not be displayed.
    """
    @property
    def uncertainty_type(self):
        """``"var"`` : `VarianceUncertainty` implements variance.
        """
        return 'var'

    @property
    def supports_correlated(self):
        """`True` : `VarianceUncertainty` allows to propagate correlated \
                    uncertainties.

        ``correlation`` must be given, this class does not implement computing
        it by itself.
        """
        return True

    def _propagate_add(self, other_uncert, result_data, correlation):
        return super()._propagate_add_sub(other_uncert, result_data,
                                          correlation, subtract=False)

    def _propagate_subtract(self, other_uncert, result_data, correlation):
        return super()._propagate_add_sub(other_uncert, result_data,
                                          correlation, subtract=True)

    def _propagate_multiply(self, other_uncert, result_data, correlation):
        return super()._propagate_multiply_divide(other_uncert,
                                                  result_data, correlation,
                                                  divide=False)

    def _propagate_divide(self, other_uncert, result_data, correlation):
        return super()._propagate_multiply_divide(other_uncert,
                                                  result_data, correlation,
                                                  divide=True)

    def _data_unit_to_uncertainty_unit(self, value):
        return value ** 2


def _inverse(x):
    """Just a simple inverse for use in the InverseVariance"""
    return 1 / x
```
### 3 - astropy/nddata/nduncertainty.py:

Start line: 843, End line: 936

```python
class InverseVariance(_VariancePropagationMixin, NDUncertainty):
    """
    Inverse variance uncertainty assuming first order Gaussian error
    propagation.

    This class implements uncertainty propagation for ``addition``,
    ``subtraction``, ``multiplication`` and ``division`` with other instances
    of `InverseVariance`. The class can handle if the uncertainty has a unit
    that differs from (but is convertible to) the parents `NDData` unit. The
    unit of the resulting uncertainty will the inverse square of the unit of
    the resulting data. Also support for correlation is possible but requires
    the correlation as input. It cannot handle correlation determination
    itself.

    Parameters
    ----------
    args, kwargs :
        see `NDUncertainty`

    Examples
    --------
    Compare this example to that in `StdDevUncertainty`; the uncertainties
    in the examples below are equivalent to the uncertainties in
    `StdDevUncertainty`.

    `InverseVariance` should always be associated with an `NDData`-like
    instance, either by creating it during initialization::

        >>> from astropy.nddata import NDData, InverseVariance
        >>> ndd = NDData([1,2,3], unit='m',
        ...              uncertainty=InverseVariance([100, 100, 100]))
        >>> ndd.uncertainty  # doctest: +FLOAT_CMP
        InverseVariance([100, 100, 100])

    or by setting it manually on the `NDData` instance::

        >>> ndd.uncertainty = InverseVariance([25], unit='1/m^2', copy=True)
        >>> ndd.uncertainty  # doctest: +FLOAT_CMP
        InverseVariance([25])

    the uncertainty ``array`` can also be set directly::

        >>> ndd.uncertainty.array = 0.25
        >>> ndd.uncertainty
        InverseVariance(0.25)

    .. note::
        The unit will not be displayed.
    """
    @property
    def uncertainty_type(self):
        """``"ivar"`` : `InverseVariance` implements inverse variance.
        """
        return 'ivar'

    @property
    def supports_correlated(self):
        """`True` : `InverseVariance` allows to propagate correlated \
                    uncertainties.

        ``correlation`` must be given, this class does not implement computing
        it by itself.
        """
        return True

    def _propagate_add(self, other_uncert, result_data, correlation):
        return super()._propagate_add_sub(other_uncert, result_data,
                                          correlation, subtract=False,
                                          to_variance=_inverse,
                                          from_variance=_inverse)

    def _propagate_subtract(self, other_uncert, result_data, correlation):
        return super()._propagate_add_sub(other_uncert, result_data,
                                          correlation, subtract=True,
                                          to_variance=_inverse,
                                          from_variance=_inverse)

    def _propagate_multiply(self, other_uncert, result_data, correlation):
        return super()._propagate_multiply_divide(other_uncert,
                                                  result_data, correlation,
                                                  divide=False,
                                                  to_variance=_inverse,
                                                  from_variance=_inverse)

    def _propagate_divide(self, other_uncert, result_data, correlation):
        return super()._propagate_multiply_divide(other_uncert,
                                                  result_data, correlation,
                                                  divide=True,
                                                  to_variance=_inverse,
                                                  from_variance=_inverse)

    def _data_unit_to_uncertainty_unit(self, value):
        return 1 / value ** 2
```
### 4 - astropy/nddata/nduncertainty.py:

Start line: 399, End line: 449

```python
class UnknownUncertainty(NDUncertainty):
    """This class implements any unknown uncertainty type.

    The main purpose of having an unknown uncertainty class is to prevent
    uncertainty propagation.

    Parameters
    ----------
    args, kwargs :
        see `NDUncertainty`
    """

    @property
    def supports_correlated(self):
        """`False` : Uncertainty propagation is *not* possible for this class.
        """
        return False

    @property
    def uncertainty_type(self):
        """``"unknown"`` : `UnknownUncertainty` implements any unknown \
                           uncertainty type.
        """
        return 'unknown'

    def _data_unit_to_uncertainty_unit(self, value):
        """
        No way to convert if uncertainty is unknown.
        """
        return None

    def _convert_uncertainty(self, other_uncert):
        """Raise an Exception because unknown uncertainty types cannot
        implement propagation.
        """
        msg = "Uncertainties of unknown type cannot be propagated."
        raise IncompatibleUncertaintiesException(msg)

    def _propagate_add(self, other_uncert, result_data, correlation):
        """Not possible for unknown uncertainty types.
        """
        return None

    def _propagate_subtract(self, other_uncert, result_data, correlation):
        return None

    def _propagate_multiply(self, other_uncert, result_data, correlation):
        return None

    def _propagate_divide(self, other_uncert, result_data, correlation):
        return None
```
### 5 - astropy/nddata/nduncertainty.py:

Start line: 350, End line: 396

```python
class NDUncertainty(metaclass=ABCMeta):

    def _convert_uncertainty(self, other_uncert):
        """Checks if the uncertainties are compatible for propagation.

        Checks if the other uncertainty is `NDUncertainty`-like and if so
        verify that the uncertainty_type is equal. If the latter is not the
        case try returning ``self.__class__(other_uncert)``.

        Parameters
        ----------
        other_uncert : `NDUncertainty` subclass
            The other uncertainty.

        Returns
        -------
        other_uncert : `NDUncertainty` subclass
            but converted to a compatible `NDUncertainty` subclass if
            possible and necessary.

        Raises
        ------
        IncompatibleUncertaintiesException:
            If the other uncertainty cannot be converted to a compatible
            `NDUncertainty` subclass.
        """
        if isinstance(other_uncert, NDUncertainty):
            if self.uncertainty_type == other_uncert.uncertainty_type:
                return other_uncert
            else:
                return self.__class__(other_uncert)
        else:
            raise IncompatibleUncertaintiesException

    @abstractmethod
    def _propagate_add(self, other_uncert, result_data, correlation):
        return None

    @abstractmethod
    def _propagate_subtract(self, other_uncert, result_data, correlation):
        return None

    @abstractmethod
    def _propagate_multiply(self, other_uncert, result_data, correlation):
        return None

    @abstractmethod
    def _propagate_divide(self, other_uncert, result_data, correlation):
        return None
```
### 6 - astropy/nddata/nduncertainty.py:

Start line: 31, End line: 134

```python
class NDUncertainty(metaclass=ABCMeta):
    """This is the metaclass for uncertainty classes used with `NDData`.

    Parameters
    ----------
    array : any type, optional
        The array or value (the parameter name is due to historical reasons) of
        the uncertainty. `numpy.ndarray`, `~astropy.units.Quantity` or
        `NDUncertainty` subclasses are recommended.
        If the `array` is `list`-like or `numpy.ndarray`-like it will be cast
        to a plain `numpy.ndarray`.
        Default is ``None``.

    unit : unit-like, optional
        Unit for the uncertainty ``array``. Strings that can be converted to a
        `~astropy.units.Unit` are allowed.
        Default is ``None``.

    copy : `bool`, optional
        Indicates whether to save the `array` as a copy. ``True`` copies it
        before saving, while ``False`` tries to save every parameter as
        reference. Note however that it is not always possible to save the
        input as reference.
        Default is ``True``.

    Raises
    ------
    IncompatibleUncertaintiesException
        If given another `NDUncertainty`-like class as ``array`` if their
        ``uncertainty_type`` is different.
    """

    def __init__(self, array=None, copy=True, unit=None):
        if isinstance(array, NDUncertainty):
            # Given an NDUncertainty class or subclass check that the type
            # is the same.
            if array.uncertainty_type != self.uncertainty_type:
                raise IncompatibleUncertaintiesException
            # Check if two units are given and take the explicit one then.
            if (unit is not None and unit != array._unit):
                # TODO : Clarify it (see NDData.init for same problem)?
                log.info("overwriting Uncertainty's current "
                         "unit with specified unit.")
            elif array._unit is not None:
                unit = array.unit
            array = array.array

        elif isinstance(array, Quantity):
            # Check if two units are given and take the explicit one then.
            if (unit is not None and array.unit is not None and
                    unit != array.unit):
                log.info("overwriting Quantity's current "
                         "unit with specified unit.")
            elif array.unit is not None:
                unit = array.unit
            array = array.value

        if unit is None:
            self._unit = None
        else:
            self._unit = Unit(unit)

        if copy:
            array = deepcopy(array)
            unit = deepcopy(unit)

        self.array = array
        self.parent_nddata = None  # no associated NDData - until it is set!

    @property
    @abstractmethod
    def uncertainty_type(self):
        """`str` : Short description of the type of uncertainty.

        Defined as abstract property so subclasses *have* to override this.
        """
        return None

    @property
    def supports_correlated(self):
        """`bool` : Supports uncertainty propagation with correlated \
                 uncertainties?

        .. versionadded:: 1.2
        """
        return False

    @property
    def array(self):
        """`numpy.ndarray` : the uncertainty's value.
        """
        return self._array

    @array.setter
    def array(self, value):
        if isinstance(value, (list, np.ndarray)):
            value = np.array(value, subok=False, copy=False)
        self._array = value

    @property
    def unit(self):
        """`~astropy.units.Unit` : The unit of the uncertainty, if any.
        """
        return self._unit
```
### 7 - astropy/nddata/nddata.py:

Start line: 310, End line: 334

```python
class NDData(NDDataBase):

    @uncertainty.setter
    def uncertainty(self, value):
        if value is not None:
            # There is one requirements on the uncertainty: That
            # it has an attribute 'uncertainty_type'.
            # If it does not match this requirement convert it to an unknown
            # uncertainty.
            if not hasattr(value, 'uncertainty_type'):
                log.info('uncertainty should have attribute uncertainty_type.')
                value = UnknownUncertainty(value, copy=False)

            # If it is a subclass of NDUncertainty we must set the
            # parent_nddata attribute. (#4152)
            if isinstance(value, NDUncertainty):
                # In case the uncertainty already has a parent create a new
                # instance because we need to assume that we don't want to
                # steal the uncertainty from another NDData object
                if value._parent_nddata is not None:
                    value = value.__class__(value, copy=False)
                # Then link it to this NDData instance (internally this needs
                # to be saved as weakref but that's done by NDUncertainty
                # setter).
                value.parent_nddata = self
        self._uncertainty = value
```
### 8 - astropy/nddata/nduncertainty.py:

Start line: 204, End line: 235

```python
class NDUncertainty(metaclass=ABCMeta):

    @parent_nddata.setter
    def parent_nddata(self, value):
        if value is not None and not isinstance(value, weakref.ref):
            # Save a weak reference on the uncertainty that points to this
            # instance of NDData. Direct references should NOT be used:
            # https://github.com/astropy/astropy/pull/4799#discussion_r61236832
            value = weakref.ref(value)
        # Set _parent_nddata here and access below with the property because value
        # is a weakref
        self._parent_nddata = value
        # set uncertainty unit to that of the parent if it was not already set, unless initializing
        # with empty parent (Value=None)
        if value is not None:
            parent_unit = self.parent_nddata.unit
            if self.unit is None:
                if parent_unit is None:
                    self.unit = None
                else:
                    # Set the uncertainty's unit to the appropriate value
                    self.unit = self._data_unit_to_uncertainty_unit(parent_unit)
            else:
                # Check that units of uncertainty are compatible with those of
                # the parent. If they are, no need to change units of the
                # uncertainty or the data. If they are not, let the user know.
                unit_from_data = self._data_unit_to_uncertainty_unit(parent_unit)
                try:
                    unit_from_data.to(self.unit)
                except UnitConversionError:
                    raise UnitConversionError("Unit {} of uncertainty "
                                              "incompatible with unit {} of "
                                              "data".format(self.unit,
                                                            parent_unit))
```
### 9 - astropy/nddata/nddata.py:

Start line: 285, End line: 308

```python
class NDData(NDDataBase):

    @wcs.setter
    def wcs(self, wcs):
        if self._wcs is not None and wcs is not None:
            raise ValueError("You can only set the wcs attribute with a WCS if no WCS is present.")

        if wcs is None or isinstance(wcs, BaseHighLevelWCS):
            self._wcs = wcs
        elif isinstance(wcs, BaseLowLevelWCS):
            self._wcs = HighLevelWCSWrapper(wcs)
        else:
            raise TypeError("The wcs argument must implement either the high or"
                            " low level WCS API.")

    @property
    def uncertainty(self):
        """
        any type : Uncertainty in the dataset, if any.

        Should have an attribute ``uncertainty_type`` that defines what kind of
        uncertainty is stored, such as ``'std'`` for standard deviation or
        ``'var'`` for variance. A metaclass defining such an interface is
        `~astropy.nddata.NDUncertainty` but isn't mandatory.
        """
        return self._uncertainty
```
### 10 - astropy/nddata/nduncertainty.py:

Start line: 452, End line: 538

```python
class _VariancePropagationMixin:
    """
    Propagation of uncertainties for variances, also used to perform error
    propagation for variance-like uncertainties (standard deviation and inverse
    variance).
    """

    def _propagate_add_sub(self, other_uncert, result_data, correlation,
                           subtract=False,
                           to_variance=lambda x: x, from_variance=lambda x: x):
        """
        Error propagation for addition or subtraction of variance or
        variance-like uncertainties. Uncertainties are calculated using the
        formulae for variance but can be used for uncertainty convertible to
        a variance.

        Parameters
        ----------

        other_uncert : `~astropy.nddata.NDUncertainty` instance
            The uncertainty, if any, of the other operand.

        result_data : `~astropy.nddata.NDData` instance
            The results of the operation on the data.

        correlation : float or array-like
            Correlation of the uncertainties.

        subtract : bool, optional
            If ``True``, propagate for subtraction, otherwise propagate for
            addition.

        to_variance : function, optional
            Function that will transform the input uncertainties to variance.
            The default assumes the uncertainty is the variance.

        from_variance : function, optional
            Function that will convert from variance to the input uncertainty.
            The default assumes the uncertainty is the variance.
        """
        if subtract:
            correlation_sign = -1
        else:
            correlation_sign = 1

        try:
            result_unit_sq = result_data.unit ** 2
        except AttributeError:
            result_unit_sq = None

        if other_uncert.array is not None:
            # Formula: sigma**2 = dB
            if (other_uncert.unit is not None and
                    result_unit_sq != to_variance(other_uncert.unit)):
                # If the other uncertainty has a unit and this unit differs
                # from the unit of the result convert it to the results unit
                other = to_variance(other_uncert.array <<
                                    other_uncert.unit).to(result_unit_sq).value
            else:
                other = to_variance(other_uncert.array)
        else:
            other = 0

        if self.array is not None:
            # Formula: sigma**2 = dA

            if self.unit is not None and to_variance(self.unit) != self.parent_nddata.unit**2:
                # If the uncertainty has a different unit than the result we
                # need to convert it to the results unit.
                this = to_variance(self.array << self.unit).to(result_unit_sq).value
            else:
                this = to_variance(self.array)
        else:
            this = 0

        # Formula: sigma**2 = dA + dB +/- 2*cor*sqrt(dA*dB)
        # Formula: sigma**2 = sigma_other + sigma_self +/- 2*cor*sqrt(dA*dB)
        #     (sign depends on whether addition or subtraction)

        # Determine the result depending on the correlation
        if isinstance(correlation, np.ndarray) or correlation != 0:
            corr = 2 * correlation * np.sqrt(this * other)
            result = this + other + correlation_sign * corr
        else:
            result = this + other

        return from_variance(result)
```
### 11 - astropy/nddata/nduncertainty.py:

Start line: 136, End line: 165

```python
class NDUncertainty(metaclass=ABCMeta):

    @unit.setter
    def unit(self, value):
        """
        The unit should be set to a value consistent with the parent NDData
        unit and the uncertainty type.
        """
        if value is not None:
            # Check the hidden attribute below, not the property. The property
            # raises an exception if there is no parent_nddata.
            if self._parent_nddata is not None:
                parent_unit = self.parent_nddata.unit
                try:
                    # Check for consistency with the unit of the parent_nddata
                    self._data_unit_to_uncertainty_unit(parent_unit).to(value)
                except UnitConversionError:
                    raise UnitConversionError("Unit {} is incompatible "
                                              "with unit {} of parent "
                                              "nddata".format(value,
                                                              parent_unit))

            self._unit = Unit(value)
        else:
            self._unit = value

    @property
    def quantity(self):
        """
        This uncertainty as an `~astropy.units.Quantity` object.
        """
        return Quantity(self.array, self.unit, copy=False, dtype=self.array.dtype)
```
### 13 - astropy/nddata/nduncertainty.py:

Start line: 3, End line: 28

```python
import numpy as np
from abc import ABCMeta, abstractmethod
from copy import deepcopy
import weakref


# from astropy.utils.compat import ignored
from astropy import log
from astropy.units import Unit, Quantity, UnitConversionError

__all__ = ['MissingDataAssociationException',
           'IncompatibleUncertaintiesException', 'NDUncertainty',
           'StdDevUncertainty', 'UnknownUncertainty',
           'VarianceUncertainty', 'InverseVariance']


class IncompatibleUncertaintiesException(Exception):
    """This exception should be used to indicate cases in which uncertainties
    with two different classes can not be propagated.
    """


class MissingDataAssociationException(Exception):
    """This exception should be used to indicate that an uncertainty instance
    has not been associated with a parent `~astropy.nddata.NDData` object.
    """
```
### 15 - astropy/nddata/nduncertainty.py:

Start line: 237, End line: 277

```python
class NDUncertainty(metaclass=ABCMeta):

    @abstractmethod
    def _data_unit_to_uncertainty_unit(self, value):
        """
        Subclasses must override this property. It should take in a data unit
        and return the correct unit for the uncertainty given the uncertainty
        type.
        """
        return None

    def __repr__(self):
        prefix = self.__class__.__name__ + '('
        try:
            body = np.array2string(self.array, separator=', ', prefix=prefix)
        except AttributeError:
            # In case it wasn't possible to use array2string
            body = str(self.array)
        return ''.join([prefix, body, ')'])

    def __getstate__(self):
        # Because of the weak reference the class wouldn't be picklable.
        try:
            return self._array, self._unit, self.parent_nddata
        except MissingDataAssociationException:
            # In case there's no parent
            return self._array, self._unit, None

    def __setstate__(self, state):
        if len(state) != 3:
            raise TypeError('The state should contain 3 items.')
        self._array = state[0]
        self._unit = state[1]

        parent = state[2]
        if parent is not None:
            parent = weakref.ref(parent)
        self._parent_nddata = parent

    def __getitem__(self, item):
        """Normal slicing on the array, keep the unit and return a reference.
        """
        return self.__class__(self.array[item], unit=self.unit, copy=False)
```
### 17 - astropy/nddata/nduncertainty.py:

Start line: 540, End line: 628

```python
class _VariancePropagationMixin:

    def _propagate_multiply_divide(self, other_uncert, result_data,
                                   correlation,
                                   divide=False,
                                   to_variance=lambda x: x,
                                   from_variance=lambda x: x):
        """
        Error propagation for multiplication or division of variance or
        variance-like uncertainties. Uncertainties are calculated using the
        formulae for variance but can be used for uncertainty convertible to
        a variance.

        Parameters
        ----------

        other_uncert : `~astropy.nddata.NDUncertainty` instance
            The uncertainty, if any, of the other operand.

        result_data : `~astropy.nddata.NDData` instance
            The results of the operation on the data.

        correlation : float or array-like
            Correlation of the uncertainties.

        divide : bool, optional
            If ``True``, propagate for division, otherwise propagate for
            multiplication.

        to_variance : function, optional
            Function that will transform the input uncertainties to variance.
            The default assumes the uncertainty is the variance.

        from_variance : function, optional
            Function that will convert from variance to the input uncertainty.
            The default assumes the uncertainty is the variance.
        """
        # For multiplication we don't need the result as quantity
        if isinstance(result_data, Quantity):
            result_data = result_data.value

        if divide:
            correlation_sign = -1
        else:
            correlation_sign = 1

        if other_uncert.array is not None:
            # We want the result to have a unit consistent with the parent, so
            # we only need to convert the unit of the other uncertainty if it
            # is different from its data's unit.
            if (other_uncert.unit and
                to_variance(1 * other_uncert.unit) !=
                    ((1 * other_uncert.parent_nddata.unit)**2).unit):
                d_b = to_variance(other_uncert.array << other_uncert.unit).to(
                    (1 * other_uncert.parent_nddata.unit)**2).value
            else:
                d_b = to_variance(other_uncert.array)
            # Formula: sigma**2 = |A|**2 * d_b
            right = np.abs(self.parent_nddata.data**2 * d_b)
        else:
            right = 0

        if self.array is not None:
            # Just the reversed case
            if (self.unit and
                to_variance(1 * self.unit) !=
                    ((1 * self.parent_nddata.unit)**2).unit):
                d_a = to_variance(self.array << self.unit).to(
                    (1 * self.parent_nddata.unit)**2).value
            else:
                d_a = to_variance(self.array)
            # Formula: sigma**2 = |B|**2 * d_a
            left = np.abs(other_uncert.parent_nddata.data**2 * d_a)
        else:
            left = 0

        # Multiplication
        #
        # The fundamental formula is:
        #   sigma**2 = |AB|**2*(d_a/A**2+d_b/B**2+2*sqrt(d_a)/A*sqrt(d_b)/B*cor)
        #
        # This formula is not very handy since it generates NaNs for every
        # zero in A and B. So we rewrite it:
        #
        # Multiplication Formula:
        #   sigma**2 = (d_a*B**2 + d_b*A**2 + (2 * cor * ABsqrt(dAdB)))
        #   sigma**2 = (left + right + (2 * cor * ABsqrt(dAdB)))
        #
        # Division
        #
        # The fundamental formula for division is:
        # ... other code
```
### 18 - astropy/nddata/nduncertainty.py:

Start line: 279, End line: 348

```python
class NDUncertainty(metaclass=ABCMeta):

    def propagate(self, operation, other_nddata, result_data, correlation):
        """Calculate the resulting uncertainty given an operation on the data.

        .. versionadded:: 1.2

        Parameters
        ----------
        operation : callable
            The operation that is performed on the `NDData`. Supported are
            `numpy.add`, `numpy.subtract`, `numpy.multiply` and
            `numpy.true_divide` (or `numpy.divide`).

        other_nddata : `NDData` instance
            The second operand in the arithmetic operation.

        result_data : `~astropy.units.Quantity` or ndarray
            The result of the arithmetic operations on the data.

        correlation : `numpy.ndarray` or number
            The correlation (rho) is defined between the uncertainties in
            sigma_AB = sigma_A * sigma_B * rho. A value of ``0`` means
            uncorrelated operands.

        Returns
        -------
        resulting_uncertainty : `NDUncertainty` instance
            Another instance of the same `NDUncertainty` subclass containing
            the uncertainty of the result.

        Raises
        ------
        ValueError
            If the ``operation`` is not supported or if correlation is not zero
            but the subclass does not support correlated uncertainties.

        Notes
        -----
        First this method checks if a correlation is given and the subclass
        implements propagation with correlated uncertainties.
        Then the second uncertainty is converted (or an Exception is raised)
        to the same class in order to do the propagation.
        Then the appropriate propagation method is invoked and the result is
        returned.
        """
        # Check if the subclass supports correlation
        if not self.supports_correlated:
            if isinstance(correlation, np.ndarray) or correlation != 0:
                raise ValueError("{} does not support uncertainty propagation"
                                 " with correlation."
                                 "".format(self.__class__.__name__))

        # Get the other uncertainty (and convert it to a matching one)
        other_uncert = self._convert_uncertainty(other_nddata.uncertainty)

        if operation.__name__ == 'add':
            result = self._propagate_add(other_uncert, result_data,
                                         correlation)
        elif operation.__name__ == 'subtract':
            result = self._propagate_subtract(other_uncert, result_data,
                                              correlation)
        elif operation.__name__ == 'multiply':
            result = self._propagate_multiply(other_uncert, result_data,
                                              correlation)
        elif operation.__name__ in ['true_divide', 'divide']:
            result = self._propagate_divide(other_uncert, result_data,
                                            correlation)
        else:
            raise ValueError('unsupported operation')

        return self.__class__(result, copy=False)
```
### 20 - astropy/nddata/nduncertainty.py:

Start line: 629, End line: 653

```python
class _VariancePropagationMixin:

    def _propagate_multiply_divide(self, other_uncert, result_data,
                                   correlation,
                                   divide=False,
                                   to_variance=lambda x: x,
                                   from_variance=lambda x: x):
        #   sigma**2 = |A/B|**2*(d_a/A**2+d_b/B**2-2*sqrt(d_a)/A*sqrt(d_b)/B*cor)
        #
        # As with multiplication, it is convenient to rewrite this to avoid
        # nans where A is zero.
        #
        # Division formula (rewritten):
        #   sigma**2 = d_a/B**2 + (A/B)**2 * d_b/B**2
        #                   - 2 * cor * A *sqrt(dAdB) / B**3
        #   sigma**2 = d_a/B**2 + (A/B)**2 * d_b/B**2
        #                   - 2*cor * sqrt(d_a)/B**2  * sqrt(d_b) * A / B
        #   sigma**2 = multiplication formula/B**4 (and sign change in
        #               the correlation)

        if isinstance(correlation, np.ndarray) or correlation != 0:
            corr = (2 * correlation * np.sqrt(d_a * d_b) *
                    self.parent_nddata.data *
                    other_uncert.parent_nddata.data)
        else:
            corr = 0

        if divide:
            return from_variance((left + right + correlation_sign * corr) /
                                 other_uncert.parent_nddata.data**4)
        else:
            return from_variance(left + right + correlation_sign * corr)
```
### 23 - astropy/nddata/nduncertainty.py:

Start line: 167, End line: 202

```python
class NDUncertainty(metaclass=ABCMeta):

    @property
    def parent_nddata(self):
        """`NDData` : reference to `NDData` instance with this uncertainty.

        In case the reference is not set uncertainty propagation will not be
        possible since propagation might need the uncertain data besides the
        uncertainty.
        """
        no_parent_message = "uncertainty is not associated with an NDData object"
        parent_lost_message = (
            "the associated NDData object was deleted and cannot be accessed "
            "anymore. You can prevent the NDData object from being deleted by "
            "assigning it to a variable. If this happened after unpickling "
            "make sure you pickle the parent not the uncertainty directly."
        )
        try:
            parent = self._parent_nddata
        except AttributeError:
            raise MissingDataAssociationException(no_parent_message)
        else:
            if parent is None:
                raise MissingDataAssociationException(no_parent_message)
            else:
                # The NDData is saved as weak reference so we must call it
                # to get the object the reference points to. However because
                # we have a weak reference here it's possible that the parent
                # was deleted because its reference count dropped to zero.
                if isinstance(self._parent_nddata, weakref.ref):
                    resolved_parent = self._parent_nddata()
                    if resolved_parent is None:
                        log.info(parent_lost_message)
                    return resolved_parent
                else:
                    log.info("parent_nddata should be a weakref to an NDData "
                             "object.")
                    return self._parent_nddata
```
