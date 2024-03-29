# astropy__astropy-13745

| **astropy/astropy** | `0446f168dc6e34996482394f00770b52756b8f9c` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 276 |
| **Any found context length** | 276 |
| **Avg pos** | 1.0 |
| **Min pos** | 1 |
| **Max pos** | 1 |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/astropy/coordinates/angles.py b/astropy/coordinates/angles.py
--- a/astropy/coordinates/angles.py
+++ b/astropy/coordinates/angles.py
@@ -587,7 +587,7 @@ def _validate_angles(self, angles=None):
         if angles.unit is u.deg:
             limit = 90
         elif angles.unit is u.rad:
-            limit = 0.5 * np.pi
+            limit = self.dtype.type(0.5 * np.pi)
         else:
             limit = u.degree.to(angles.unit, 90.0)
 

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| astropy/coordinates/angles.py | 590 | 590 | 1 | 1 | 276


## Problem Statement

```
float32 representation of pi/2 is rejected by `Latitude`
<!-- This comments are hidden when you submit the issue,
so you do not need to remove them! -->

<!-- Please be sure to check out our contributing guidelines,
https://github.com/astropy/astropy/blob/main/CONTRIBUTING.md .
Please be sure to check out our code of conduct,
https://github.com/astropy/astropy/blob/main/CODE_OF_CONDUCT.md . -->

<!-- Please have a search on our GitHub repository to see if a similar
issue has already been posted.
If a similar issue is closed, have a quick look to see if you are satisfied
by the resolution.
If not please go ahead and open an issue! -->

<!-- Please check that the development version still produces the same bug.
You can install development version with
pip install git+https://github.com/astropy/astropy
command. -->

### Description

The closest float32 value to pi/2 is by accident slightly larger than pi/2:

\`\`\`
In [5]: np.pi/2
Out[5]: 1.5707963267948966

In [6]: np.float32(np.pi/2)
Out[6]: 1.5707964
\`\`\`

Astropy checks using float64 precision, rejecting "valid" alt values (e.g. float32 values read from files):

\`\`\`

In [1]: from astropy.coordinates import Latitude

In [2]: import numpy as np

In [3]: lat = np.float32(np.pi/2)

In [4]: Latitude(lat, 'rad')
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
Cell In [4], line 1
----> 1 Latitude(lat, 'rad')

File ~/.local/lib/python3.10/site-packages/astropy/coordinates/angles.py:564, in Latitude.__new__(cls, angle, unit, **kwargs)
    562     raise TypeError("A Latitude angle cannot be created from a Longitude angle")
    563 self = super().__new__(cls, angle, unit=unit, **kwargs)
--> 564 self._validate_angles()
    565 return self

File ~/.local/lib/python3.10/site-packages/astropy/coordinates/angles.py:585, in Latitude._validate_angles(self, angles)
    582     invalid_angles = (np.any(angles.value < lower) or
    583                       np.any(angles.value > upper))
    584 if invalid_angles:
--> 585     raise ValueError('Latitude angle(s) must be within -90 deg <= angle <= 90 deg, '
    586                      'got {}'.format(angles.to(u.degree)))

ValueError: Latitude angle(s) must be within -90 deg <= angle <= 90 deg, got 90.00000250447816 deg
\`\`\`

### Expected behavior

Be lenient? E.g. only make the comparison up to float 32 precision?

### Actual behavior
See error above

### Steps to Reproduce

See snippet above.

### System Details
<!-- Even if you do not think this is necessary, it is useful information for the maintainers.
Please run the following snippet and paste the output below:
import platform; print(platform.platform())
import sys; print("Python", sys.version)
import numpy; print("Numpy", numpy.__version__)
import erfa; print("pyerfa", erfa.__version__)
import astropy; print("astropy", astropy.__version__)
import scipy; print("Scipy", scipy.__version__)
import matplotlib; print("Matplotlib", matplotlib.__version__)
-->
\`\`\`
Linux-5.15.65-1-MANJARO-x86_64-with-glibc2.36
Python 3.10.7 (main, Sep  6 2022, 21:22:27) [GCC 12.2.0]
Numpy 1.23.3
pyerfa 2.0.0.1
astropy 5.0.1
Scipy 1.9.1
Matplotlib 3.5.2
\`\`\`

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 astropy/coordinates/angles.py** | 576 | 601| 276 | 276 | 6346 | 
| 2 | **1 astropy/coordinates/angles.py** | 603 | 619| 160 | 436 | 6346 | 
| 3 | 2 astropy/utils/compat/numpycompat.py | 7 | 26| 296 | 732 | 6683 | 
| 4 | **2 astropy/coordinates/angles.py** | 522 | 574| 494 | 1226 | 6683 | 
| 5 | **2 astropy/coordinates/angles.py** | 691 | 716| 208 | 1434 | 6683 | 
| 6 | 3 astropy/coordinates/orbital_elements.py | 8 | 86| 1363 | 2797 | 10419 | 
| 7 | 4 astropy/coordinates/angle_formats.py | 328 | 337| 122 | 2919 | 15614 | 
| 8 | 5 astropy/coordinates/builtin_frames/utils.py | 7 | 38| 314 | 3233 | 19434 | 
| 9 | 6 astropy/coordinates/earth.py | 3 | 55| 470 | 3703 | 27814 | 
| 10 | 7 astropy/coordinates/builtin_frames/altaz.py | 3 | 70| 749 | 4452 | 29004 | 
| 11 | 7 astropy/coordinates/angle_formats.py | 301 | 325| 219 | 4671 | 29004 | 
| 12 | 7 astropy/coordinates/angle_formats.py | 340 | 362| 186 | 4857 | 29004 | 
| 13 | 8 astropy/coordinates/angle_parsetab.py | 39 | 81| 944 | 5801 | 31689 | 
| 14 | 8 astropy/coordinates/angle_parsetab.py | 22 | 22| 937 | 6738 | 31689 | 
| 15 | 9 astropy/coordinates/errors.py | 6 | 23| 114 | 6852 | 32753 | 
| 16 | 9 astropy/coordinates/earth.py | 849 | 866| 175 | 7027 | 32753 | 
| 17 | 10 astropy/coordinates/earth_orientation.py | 12 | 47| 225 | 7252 | 34602 | 
| 18 | 11 astropy/coordinates/builtin_frames/itrs_observed_transforms.py | 1 | 49| 387 | 7639 | 36237 | 
| 19 | 11 astropy/coordinates/errors.py | 111 | 131| 125 | 7764 | 36237 | 
| 20 | 12 astropy/time/core.py | 9 | 77| 937 | 8701 | 62831 | 
| 21 | 12 astropy/coordinates/errors.py | 134 | 150| 129 | 8830 | 62831 | 
| 22 | 13 astropy/units/equivalencies.py | 59 | 74| 146 | 8976 | 72052 | 
| 23 | 14 astropy/constants/iau2012.py | 7 | 77| 769 | 9745 | 72871 | 
| 24 | **14 astropy/coordinates/angles.py** | 680 | 689| 121 | 9866 | 72871 | 
| 25 | **14 astropy/coordinates/angles.py** | 8 | 24| 130 | 9996 | 72871 | 
| 26 | 15 astropy/units/si.py | 8 | 83| 754 | 10750 | 74978 | 
| 27 | 15 astropy/coordinates/errors.py | 26 | 46| 122 | 10872 | 74978 | 
| 28 | 16 astropy/units/quantity_helper/function_helpers.py | 101 | 141| 477 | 11349 | 84906 | 
| 29 | 17 astropy/utils/iers/iers.py | 12 | 78| 762 | 12111 | 96519 | 
| 30 | 18 astropy/time/utils.py | 188 | 202| 138 | 12249 | 98784 | 
| 31 | **18 astropy/coordinates/angles.py** | 145 | 157| 123 | 12372 | 98784 | 
| 32 | 19 astropy/coordinates/angle_utilities.py | 7 | 56| 418 | 12790 | 101023 | 
| 33 | 19 astropy/coordinates/angle_formats.py | 413 | 440| 251 | 13041 | 101023 | 
| 34 | 19 astropy/coordinates/angle_parsetab.py | 1 | 20| 363 | 13404 | 101023 | 
| 35 | 19 astropy/coordinates/errors.py | 68 | 89| 123 | 13527 | 101023 | 
| 36 | 20 examples/coordinates/plot_sgr-coordinate-frame.py | 184 | 243| 607 | 14134 | 103415 | 
| 37 | 21 astropy/visualization/wcsaxes/coordinate_helpers.py | 8 | 46| 306 | 14440 | 112412 | 
| 38 | 22 astropy/io/fits/card.py | 1250 | 1299| 375 | 14815 | 123480 | 
| 39 | 23 astropy/wcs/docstrings.py | 343 | 392| 738 | 15553 | 144933 | 
| 40 | 23 astropy/coordinates/angle_formats.py | 562 | 649| 832 | 16385 | 144933 | 
| 41 | **23 astropy/coordinates/angles.py** | 277 | 379| 884 | 17269 | 144933 | 
| 42 | **23 astropy/coordinates/angles.py** | 454 | 519| 552 | 17821 | 144933 | 
| 43 | **23 astropy/coordinates/angles.py** | 105 | 143| 338 | 18159 | 144933 | 
| 44 | 24 astropy/visualization/wcsaxes/formatter_locator.py | 11 | 48| 343 | 18502 | 149325 | 
| 45 | 25 astropy/io/fits/util.py | 719 | 752| 208 | 18710 | 156362 | 
| 46 | 25 astropy/coordinates/errors.py | 49 | 65| 130 | 18840 | 156362 | 
| 47 | 26 astropy/coordinates/attributes.py | 213 | 256| 313 | 19153 | 159902 | 
| 48 | 27 astropy/io/votable/exceptions.py | 742 | 755| 176 | 19329 | 173226 | 
| 49 | 28 astropy/extern/configobj/validate.py | 839 | 888| 509 | 19838 | 185858 | 
| 50 | 28 astropy/coordinates/angle_utilities.py | 227 | 245| 217 | 20055 | 185858 | 
| 51 | 28 astropy/coordinates/angle_utilities.py | 59 | 85| 246 | 20301 | 185858 | 
| 52 | 28 astropy/coordinates/earth_orientation.py | 50 | 74| 193 | 20494 | 185858 | 
| 53 | 29 astropy/convolution/convolve.py | 56 | 76| 745 | 21239 | 197190 | 


### Hint

```
> Be lenient? E.g. only make the comparison up to float 32 precision?

Instead, we could make the comparison based on the precision of the ``dtype``, using something like https://numpy.org/doc/stable/reference/generated/numpy.finfo.html?highlight=finfo#numpy.finfo
That's a funny one! I think @nstarman's suggestion would work: would just need to change the dtype of `limit` to `self.dtype` in `_validate_angles`.
That wouldn't solve the case where the value is read from a float32 into a float64, which can happen pretty fast due to the places where casting can happen. Better than nothing, but...
Do we want to simply let it pass with that value, or rather "round" the input value down to the float64 representation of `pi/2`? Just wondering what may happen with larger values in any calculations down the line; probably nothing really terrible (like ending up with inverse Longitude), but...
This is what I did to fix it on our end:
https://github.com/cta-observatory/ctapipe/pull/2077/files#diff-d2022785b8c35b2f43d3b9d43c3721efaa9339d98dbff39c864172f1ba2f4f6f
\`\`\`python
_half_pi = 0.5 * np.pi
_half_pi_maxval = (1 + 1e-6) * _half_pi




def _clip_altitude_if_close(altitude):
    """
    Round absolute values slightly larger than pi/2 in float64 to pi/2

    These can come from simtel_array because float32(pi/2) > float64(pi/2)
    and simtel using float32.

    Astropy complains about these values, so we fix them here.
    """
    if altitude > _half_pi and altitude < _half_pi_maxval:
        return _half_pi


    if altitude < -_half_pi and altitude > -_half_pi_maxval:
        return -_half_pi


    return altitude
\`\`\`

Would that be an acceptable solution also here?
Does this keep the numpy dtype of the input?
No, the point is that this casts to float64.
So ``Latitude(pi/2, unit=u.deg, dtype=float32)``  can become a float64?
> Does this keep the numpy dtype of the input?

If `limit` is cast to `self.dtype` (is that identical to `self.angle.dtype`?) as per your suggestion above, it should.
But that modification should already catch the cases of `angle` still passed as float32, since both are compared at the same resolution. I'd vote to do this and only implement the more lenient comparison (for float32 that had already been upcast to float64)  as a fallback, i.e. if still `invalid_angles`, set something like
`_half_pi_maxval = (0.5 + np.finfo(np.float32).eps)) * np.pi` and do a second comparison to that, if that passes, set to  `limit * np.sign(self.angle)`. Have to remember that `self.angle` is an array in general...
> So `Latitude(pi/2, unit=u.deg, dtype=float32)` can become a float64?

`Latitude(pi/2, unit=u.rad, dtype=float32)` would in that approach, as it currently raises the `ValueError`.
I'll open a PR with unit test cases and then we can decide about the wanted behaviour for each of them
```

## Patch

```diff
diff --git a/astropy/coordinates/angles.py b/astropy/coordinates/angles.py
--- a/astropy/coordinates/angles.py
+++ b/astropy/coordinates/angles.py
@@ -587,7 +587,7 @@ def _validate_angles(self, angles=None):
         if angles.unit is u.deg:
             limit = 90
         elif angles.unit is u.rad:
-            limit = 0.5 * np.pi
+            limit = self.dtype.type(0.5 * np.pi)
         else:
             limit = u.degree.to(angles.unit, 90.0)
 

```

## Test Patch

```diff
diff --git a/astropy/coordinates/tests/test_angles.py b/astropy/coordinates/tests/test_angles.py
--- a/astropy/coordinates/tests/test_angles.py
+++ b/astropy/coordinates/tests/test_angles.py
@@ -1092,3 +1092,54 @@ def test_str_repr_angles_nan(cls, input, expstr, exprepr):
     # Deleting whitespaces since repr appears to be adding them for some values
     # making the test fail.
     assert repr(q).replace(" ", "") == f'<{cls.__name__}{exprepr}>'.replace(" ","")
+
+
+@pytest.mark.parametrize("sign", (-1, 1))
+@pytest.mark.parametrize(
+    "value,expected_value,dtype,expected_dtype",
+    [
+        (np.pi / 2, np.pi / 2, None, np.float64),
+        (np.pi / 2, np.pi / 2, np.float64, np.float64),
+        (np.float32(np.pi / 2), np.float32(np.pi / 2), None, np.float32),
+        (np.float32(np.pi / 2), np.float32(np.pi / 2), np.float32, np.float32),
+        # these cases would require coercing the float32 value to the float64 value
+        # making validate have side effects, so it's not implemented for now
+        # (np.float32(np.pi / 2), np.pi / 2, np.float64, np.float64),
+        # (np.float32(-np.pi / 2), -np.pi / 2, np.float64, np.float64),
+    ]
+)
+def test_latitude_limits(value, expected_value, dtype, expected_dtype, sign):
+    """
+    Test that the validation of the Latitude value range in radians works
+    in both float32 and float64.
+
+    As discussed in issue #13708, before, the float32 represenation of pi/2
+    was rejected as invalid because the comparison always used the float64
+    representation.
+    """
+    # this prevents upcasting to float64 as sign * value would do
+    if sign < 0:
+        value = -value
+        expected_value = -expected_value
+
+    result = Latitude(value, u.rad, dtype=dtype)
+    assert result.value == expected_value
+    assert result.dtype == expected_dtype
+    assert result.unit == u.rad
+
+
+@pytest.mark.parametrize(
+    "value,dtype",
+    [
+        (0.50001 * np.pi, np.float32),
+        (np.float32(0.50001 * np.pi), np.float32),
+        (0.50001 * np.pi, np.float64),
+    ]
+)
+def test_latitude_out_of_limits(value, dtype):
+    """
+    Test that values slightly larger than pi/2 are rejected for different dtypes.
+    Test cases for issue #13708
+    """
+    with pytest.raises(ValueError, match=r"Latitude angle\(s\) must be within.*"):
+        Latitude(value, u.rad, dtype=dtype)

```


## Code snippets

### 1 - astropy/coordinates/angles.py:

Start line: 576, End line: 601

```python
class Latitude(Angle):

    def _validate_angles(self, angles=None):
        """Check that angles are between -90 and 90 degrees.
        If not given, the check is done on the object itself"""
        # Convert the lower and upper bounds to the "native" unit of
        # this angle.  This limits multiplication to two values,
        # rather than the N values in `self.value`.  Also, the
        # comparison is performed on raw arrays, rather than Quantity
        # objects, for speed.
        if angles is None:
            angles = self

        if angles.unit is u.deg:
            limit = 90
        elif angles.unit is u.rad:
            limit = 0.5 * np.pi
        else:
            limit = u.degree.to(angles.unit, 90.0)

        # This invalid catch block can be removed when the minimum numpy
        # version is >= 1.19 (NUMPY_LT_1_19)
        with np.errstate(invalid='ignore'):
            invalid_angles = (np.any(angles.value < -limit) or
                              np.any(angles.value > limit))
        if invalid_angles:
            raise ValueError('Latitude angle(s) must be within -90 deg <= angle <= 90 deg, '
                             'got {}'.format(angles.to(u.degree)))
```
### 2 - astropy/coordinates/angles.py:

Start line: 603, End line: 619

```python
class Latitude(Angle):

    def __setitem__(self, item, value):
        # Forbid assigning a Long to a Lat.
        if isinstance(value, Longitude):
            raise TypeError("A Longitude angle cannot be assigned to a Latitude angle")
        # first check bounds
        if value is not np.ma.masked:
            self._validate_angles(value)
        super().__setitem__(item, value)

    # Any calculation should drop to Angle
    def __array_ufunc__(self, *args, **kwargs):
        results = super().__array_ufunc__(*args, **kwargs)
        return _no_angle_subclass(results)


class LongitudeInfo(u.QuantityInfo):
    _represent_as_dict_attrs = u.QuantityInfo._represent_as_dict_attrs + ('wrap_angle',)
```
### 3 - astropy/utils/compat/numpycompat.py:

Start line: 7, End line: 26

```python
import numpy as np

from astropy.utils import minversion

__all__ = ['NUMPY_LT_1_19', 'NUMPY_LT_1_19_1', 'NUMPY_LT_1_20',
           'NUMPY_LT_1_21_1', 'NUMPY_LT_1_22', 'NUMPY_LT_1_22_1',
           'NUMPY_LT_1_23', 'NUMPY_LT_1_24']

# TODO: It might also be nice to have aliases to these named for specific
# features/bugs we're checking for (ex:
# astropy.table.table._BROKEN_UNICODE_TABLE_SORT)
NUMPY_LT_1_19 = not minversion(np, '1.19')
NUMPY_LT_1_19_1 = not minversion(np, '1.19.1')
NUMPY_LT_1_20 = not minversion(np, '1.20')
NUMPY_LT_1_21_1 = not minversion(np, '1.21.1')
NUMPY_LT_1_22 = not minversion(np, '1.22')
NUMPY_LT_1_22_1 = not minversion(np, '1.22.1')
NUMPY_LT_1_23 = not minversion(np, '1.23')
NUMPY_LT_1_24 = not minversion(np, '1.24dev0')
```
### 4 - astropy/coordinates/angles.py:

Start line: 522, End line: 574

```python
class Latitude(Angle):
    """
    Latitude-like angle(s) which must be in the range -90 to +90 deg.

    A Latitude object is distinguished from a pure
    :class:`~astropy.coordinates.Angle` by virtue of being constrained
    so that::

      -90.0 * u.deg <= angle(s) <= +90.0 * u.deg

    Any attempt to set a value outside that range will result in a
    `ValueError`.

    The input angle(s) can be specified either as an array, list,
    scalar, tuple (see below), string,
    :class:`~astropy.units.Quantity` or another
    :class:`~astropy.coordinates.Angle`.

    The input parser is flexible and supports all of the input formats
    supported by :class:`~astropy.coordinates.Angle`.

    Parameters
    ----------
    angle : array, list, scalar, `~astropy.units.Quantity`, `~astropy.coordinates.Angle`
        The angle value(s). If a tuple, will be interpreted as ``(h, m, s)``
        or ``(d, m, s)`` depending on ``unit``. If a string, it will be
        interpreted following the rules described for
        :class:`~astropy.coordinates.Angle`.

        If ``angle`` is a sequence or array of strings, the resulting
        values will be in the given ``unit``, or if `None` is provided,
        the unit will be taken from the first given value.

    unit : unit-like, optional
        The unit of the value specified for the angle.  This may be
        any string that `~astropy.units.Unit` understands, but it is
        better to give an actual unit object.  Must be an angular
        unit.

    Raises
    ------
    `~astropy.units.UnitsError`
        If a unit is not provided or it is not an angular unit.
    `TypeError`
        If the angle parameter is an instance of :class:`~astropy.coordinates.Longitude`.
    """
    def __new__(cls, angle, unit=None, **kwargs):
        # Forbid creating a Lat from a Long.
        if isinstance(angle, Longitude):
            raise TypeError("A Latitude angle cannot be created from a Longitude angle")
        self = super().__new__(cls, angle, unit=unit, **kwargs)
        self._validate_angles()
        return self
```
### 5 - astropy/coordinates/angles.py:

Start line: 691, End line: 716

```python
class Longitude(Angle):

    def __setitem__(self, item, value):
        # Forbid assigning a Lat to a Long.
        if isinstance(value, Latitude):
            raise TypeError("A Latitude angle cannot be assigned to a Longitude angle")
        super().__setitem__(item, value)
        self._wrap_at(self.wrap_angle)

    @property
    def wrap_angle(self):
        return self._wrap_angle

    @wrap_angle.setter
    def wrap_angle(self, value):
        self._wrap_angle = Angle(value, copy=False)
        self._wrap_at(self.wrap_angle)

    def __array_finalize__(self, obj):
        super().__array_finalize__(obj)
        self._wrap_angle = getattr(obj, '_wrap_angle',
                                   self._default_wrap_angle)

    # Any calculation should drop to Angle
    def __array_ufunc__(self, *args, **kwargs):
        results = super().__array_ufunc__(*args, **kwargs)
        return _no_angle_subclass(results)
```
### 6 - astropy/coordinates/orbital_elements.py:

Start line: 8, End line: 86

```python
import erfa
import numpy as np
from numpy.polynomial.polynomial import polyval

from astropy import units as u
from astropy.utils import deprecated

from . import ICRS, GeocentricTrueEcliptic, SkyCoord
from .builtin_frames.utils import get_jd12

__all__ = ["calc_moon"]

# Meeus 1998: table 47.A
#   D   M   M'  F   l    r
_MOON_L_R = (
    (0, 0, 1, 0, 6288774, -20905355),
    (2, 0, -1, 0, 1274027, -3699111),
    (2, 0, 0, 0, 658314, -2955968),
    (0, 0, 2, 0, 213618, -569925),
    (0, 1, 0, 0, -185116, 48888),
    (0, 0, 0, 2, -114332, -3149),
    (2, 0, -2, 0, 58793, 246158),
    (2, -1, -1, 0, 57066, -152138),
    (2, 0, 1, 0, 53322, -170733),
    (2, -1, 0, 0, 45758, -204586),
    (0, 1, -1, 0, -40923, -129620),
    (1, 0, 0, 0, -34720, 108743),
    (0, 1, 1, 0, -30383, 104755),
    (2, 0, 0, -2, 15327, 10321),
    (0, 0, 1, 2, -12528, 0),
    (0, 0, 1, -2, 10980, 79661),
    (4, 0, -1, 0, 10675, -34782),
    (0, 0, 3, 0, 10034, -23210),
    (4, 0, -2, 0, 8548, -21636),
    (2, 1, -1, 0, -7888, 24208),
    (2, 1, 0, 0, -6766, 30824),
    (1, 0, -1, 0, -5163, -8379),
    (1, 1, 0, 0, 4987, -16675),
    (2, -1, 1, 0, 4036, -12831),
    (2, 0, 2, 0, 3994, -10445),
    (4, 0, 0, 0, 3861, -11650),
    (2, 0, -3, 0, 3665, 14403),
    (0, 1, -2, 0, -2689, -7003),
    (2, 0, -1, 2, -2602, 0),
    (2, -1, -2, 0, 2390, 10056),
    (1, 0, 1, 0, -2348, 6322),
    (2, -2, 0, 0, 2236, -9884),
    (0, 1, 2, 0, -2120, 5751),
    (0, 2, 0, 0, -2069, 0),
    (2, -2, -1, 0, 2048, -4950),
    (2, 0, 1, -2, -1773, 4130),
    (2, 0, 0, 2, -1595, 0),
    (4, -1, -1, 0, 1215, -3958),
    (0, 0, 2, 2, -1110, 0),
    (3, 0, -1, 0, -892, 3258),
    (2, 1, 1, 0, -810, 2616),
    (4, -1, -2, 0, 759, -1897),
    (0, 2, -1, 0, -713, -2117),
    (2, 2, -1, 0, -700, 2354),
    (2, 1, -2, 0, 691, 0),
    (2, -1, 0, -2, 596, 0),
    (4, 0, 1, 0, 549, -1423),
    (0, 0, 4, 0, 537, -1117),
    (4, -1, 0, 0, 520, -1571),
    (1, 0, -2, 0, -487, -1739),
    (2, 1, 0, -2, -399, 0),
    (0, 0, 2, -2, -381, -4421),
    (1, 1, 1, 0, 351, 0),
    (3, 0, -2, 0, -340, 0),
    (4, 0, -3, 0, 330, 0),
    (2, -1, 2, 0, 327, 0),
    (0, 2, 1, 0, -323, 1165),
    (1, 1, -1, 0, 299, 0),
    (2, 0, 3, 0, 294, 0),
    (2, 0, -1, -2, 0, 8752)
)

# Meeus 1998: table 47.B
#   D   M   M'  F   b
```
### 7 - astropy/coordinates/angle_formats.py:

Start line: 328, End line: 337

```python
def _check_minute_range(m):
    """
    Checks that the given value is in the range [0,60].  If the value
    is equal to 60, then a warning is raised.
    """
    if np.any(m == 60.):
        warn(IllegalMinuteWarning(m, 'Treating as 0 min, +1 hr/deg'))
    elif np.any(m < -60.) or np.any(m > 60.):
        # "Error: minutes not in range [-60,60) ({0}).".format(min))
        raise IllegalMinuteError(m)
```
### 8 - astropy/coordinates/builtin_frames/utils.py:

Start line: 7, End line: 38

```python
import warnings

import erfa
import numpy as np

from astropy import units as u
from astropy.coordinates.earth import EarthLocation
from astropy.time import Time
from astropy.utils import iers
from astropy.utils.exceptions import AstropyWarning

from ..representation import CartesianDifferential

# We use tt as the time scale for this equinoxes, primarily because it is the
# convention for J2000 (it is unclear if there is any "right answer" for B1950)
# while #8600 makes this the default behavior, we show it here to ensure it's
# clear which is used here
EQUINOX_J2000 = Time('J2000', scale='tt')
EQUINOX_B1950 = Time('B1950', scale='tt')

# This is a time object that is the default "obstime" when such an attribute is
# necessary.  Currently, we use J2000.
DEFAULT_OBSTIME = Time('J2000', scale='tt')

# This is an EarthLocation that is the default "location" when such an attribute is
# necessary. It is the centre of the Earth.
EARTH_CENTER = EarthLocation(0*u.km, 0*u.km, 0*u.km)

PIOVER2 = np.pi / 2.

# comes from the mean of the 1962-2014 IERS B data
_DEFAULT_PM = (0.035, 0.29)*u.arcsec
```
### 9 - astropy/coordinates/earth.py:

Start line: 3, End line: 55

```python
import collections
import json
import socket
import urllib.error
import urllib.parse
import urllib.request
from warnings import warn

import erfa
import numpy as np

from astropy import constants as consts
from astropy import units as u
from astropy.units.quantity import QuantityInfoBase
from astropy.utils import data
from astropy.utils.decorators import format_doc
from astropy.utils.exceptions import AstropyUserWarning

from .angles import Angle, Latitude, Longitude
from .errors import UnknownSiteException
from .matrix_utilities import matrix_transpose
from .representation import BaseRepresentation, CartesianDifferential, CartesianRepresentation

__all__ = ['EarthLocation', 'BaseGeodeticRepresentation',
           'WGS84GeodeticRepresentation', 'WGS72GeodeticRepresentation',
           'GRS80GeodeticRepresentation']

GeodeticLocation = collections.namedtuple('GeodeticLocation', ['lon', 'lat', 'height'])

ELLIPSOIDS = {}
"""Available ellipsoids (defined in erfam.h, with numbers exposed in erfa)."""
# Note: they get filled by the creation of the geodetic classes.

OMEGA_EARTH = ((1.002_737_811_911_354_48 * u.cycle/u.day)
               .to(1/u.s, u.dimensionless_angles()))
"""
Rotational velocity of Earth, following SOFA's pvtob.

In UT1 seconds, this would be 2 pi / (24 * 3600), but we need the value
in SI seconds, so multiply by the ratio of stellar to solar day.
See Explanatory Supplement to the Astronomical Almanac, ed. P. Kenneth
Seidelmann (1992), University Science Books. The constant is the
conventional, exact one (IERS conventions 2003); see
http://hpiers.obspm.fr/eop-pc/index.php?index=constants.
"""


def _check_ellipsoid(ellipsoid=None, default='WGS84'):
    if ellipsoid is None:
        ellipsoid = default
    if ellipsoid not in ELLIPSOIDS:
        raise ValueError(f'Ellipsoid {ellipsoid} not among known ones ({ELLIPSOIDS})')
    return ellipsoid
```
### 10 - astropy/coordinates/builtin_frames/altaz.py:

Start line: 3, End line: 70

```python
import numpy as np

from astropy import units as u
from astropy.coordinates import representation as r
from astropy.coordinates.attributes import EarthLocationAttribute, QuantityAttribute, TimeAttribute
from astropy.coordinates.baseframe import BaseCoordinateFrame, RepresentationMapping, base_doc
from astropy.utils.decorators import format_doc

__all__ = ['AltAz']


_90DEG = 90*u.deg

doc_components = """
    az : `~astropy.coordinates.Angle`, optional, keyword-only
        The Azimuth for this object (``alt`` must also be given and
        ``representation`` must be None).
    alt : `~astropy.coordinates.Angle`, optional, keyword-only
        The Altitude for this object (``az`` must also be given and
        ``representation`` must be None).
    distance : `~astropy.units.Quantity` ['length'], optional, keyword-only
        The Distance for this object along the line-of-sight.

    pm_az_cosalt : `~astropy.units.Quantity` ['angular speed'], optional, keyword-only
        The proper motion in azimuth (including the ``cos(alt)`` factor) for
        this object (``pm_alt`` must also be given).
    pm_alt : `~astropy.units.Quantity` ['angular speed'], optional, keyword-only
        The proper motion in altitude for this object (``pm_az_cosalt`` must
        also be given).
    radial_velocity : `~astropy.units.Quantity` ['speed'], optional, keyword-only
        The radial velocity of this object."""

doc_footer = """
    Other parameters
    ----------------
    obstime : `~astropy.time.Time`
        The time at which the observation is taken.  Used for determining the
        position and orientation of the Earth.
    location : `~astropy.coordinates.EarthLocation`
        The location on the Earth.  This can be specified either as an
        `~astropy.coordinates.EarthLocation` object or as anything that can be
        transformed to an `~astropy.coordinates.ITRS` frame.
    pressure : `~astropy.units.Quantity` ['pressure']
        The atmospheric pressure as an `~astropy.units.Quantity` with pressure
        units.  This is necessary for performing refraction corrections.
        Setting this to 0 (the default) will disable refraction calculations
        when transforming to/from this frame.
    temperature : `~astropy.units.Quantity` ['temperature']
        The ground-level temperature as an `~astropy.units.Quantity` in
        deg C.  This is necessary for performing refraction corrections.
    relative_humidity : `~astropy.units.Quantity` ['dimensionless'] or number
        The relative humidity as a dimensionless quantity between 0 to 1.
        This is necessary for performing refraction corrections.
    obswl : `~astropy.units.Quantity` ['length']
        The average wavelength of observations as an `~astropy.units.Quantity`
         with length units.  This is necessary for performing refraction
         corrections.

    Notes
    -----
    The refraction model is based on that implemented in ERFA, which is fast
    but becomes inaccurate for altitudes below about 5 degrees.  Near and below
    altitudes of 0, it can even give meaningless answers, and in this case
    transforming to AltAz and back to another frame can give highly discrepant
    results.  For much better numerical stability, leave the ``pressure`` at
    ``0`` (the default), thereby disabling the refraction correction and
    yielding "topocentric" horizontal coordinates.
    """
```
### 24 - astropy/coordinates/angles.py:

Start line: 680, End line: 689

```python
class Longitude(Angle):

    def __new__(cls, angle, unit=None, wrap_angle=None, **kwargs):
        # Forbid creating a Long from a Lat.
        if isinstance(angle, Latitude):
            raise TypeError("A Longitude angle cannot be created from "
                            "a Latitude angle.")
        self = super().__new__(cls, angle, unit=unit, **kwargs)
        if wrap_angle is None:
            wrap_angle = getattr(angle, 'wrap_angle', self._default_wrap_angle)
        self.wrap_angle = wrap_angle  # angle-like b/c property setter
        return self
```
### 25 - astropy/coordinates/angles.py:

Start line: 8, End line: 24

```python
import warnings
from collections import namedtuple

import numpy as np

from astropy import units as u
from astropy.utils import isiterable

from . import angle_formats as form

__all__ = ['Angle', 'Latitude', 'Longitude']


# these are used by the `hms` and `dms` attributes
hms_tuple = namedtuple('hms_tuple', ('h', 'm', 's'))
dms_tuple = namedtuple('dms_tuple', ('d', 'm', 's'))
signed_dms_tuple = namedtuple('signed_dms_tuple', ('sign', 'd', 'm', 's'))
```
### 31 - astropy/coordinates/angles.py:

Start line: 145, End line: 157

```python
class Angle(u.SpecificTypeQuantity):

    @staticmethod
    def _tuple_to_float(angle, unit):
        """
        Converts an angle represented as a 3-tuple or 2-tuple into a floating
        point number in the given unit.
        """
        # TODO: Numpy array of tuples?
        if unit == u.hourangle:
            return form.hms_to_hours(*angle)
        elif unit == u.degree:
            return form.dms_to_degrees(*angle)
        else:
            raise u.UnitsError(f"Can not parse '{angle}' as unit '{unit}'")
```
### 41 - astropy/coordinates/angles.py:

Start line: 277, End line: 379

```python
class Angle(u.SpecificTypeQuantity):

    def to_string(self, unit=None, decimal=False, sep='fromunit',
                  precision=None, alwayssign=False, pad=False,
                  fields=3, format=None):
        # ... other code

        separators = {
            None: {
                u.degree: 'dms',
                u.hourangle: 'hms'},
            'latex': {
                u.degree: [r'^\circ', r'{}^\prime', r'{}^{\prime\prime}'],
                u.hourangle: [r'^{\mathrm{h}}', r'^{\mathrm{m}}', r'^{\mathrm{s}}']},
            'unicode': {
                u.degree: '°′″',
                u.hourangle: 'ʰᵐˢ'}
        }
        # 'latex_inline' provides no functionality beyond what 'latex' offers,
        # but it should be implemented to avoid ValueErrors in user code.
        separators['latex_inline'] = separators['latex']

        if sep == 'fromunit':
            if format not in separators:
                raise ValueError(f"Unknown format '{format}'")
            seps = separators[format]
            if unit in seps:
                sep = seps[unit]

        # Create an iterator so we can format each element of what
        # might be an array.
        if unit is u.degree:
            if decimal:
                values = self.degree
                if precision is not None:
                    func = ("{0:0." + str(precision) + "f}").format
                else:
                    func = '{:g}'.format
            else:
                if sep == 'fromunit':
                    sep = 'dms'
                values = self.degree
                func = lambda x: form.degrees_to_string(
                    x, precision=precision, sep=sep, pad=pad,
                    fields=fields)

        elif unit is u.hourangle:
            if decimal:
                values = self.hour
                if precision is not None:
                    func = ("{0:0." + str(precision) + "f}").format
                else:
                    func = '{:g}'.format
            else:
                if sep == 'fromunit':
                    sep = 'hms'
                values = self.hour
                func = lambda x: form.hours_to_string(
                    x, precision=precision, sep=sep, pad=pad,
                    fields=fields)

        elif unit.is_equivalent(u.radian):
            if decimal:
                values = self.to_value(unit)
                if precision is not None:
                    func = ("{0:1." + str(precision) + "f}").format
                else:
                    func = "{:g}".format
            elif sep == 'fromunit':
                values = self.to_value(unit)
                unit_string = unit.to_string(format=format)
                if format == 'latex' or format == 'latex_inline':
                    unit_string = unit_string[1:-1]

                if precision is not None:
                    def plain_unit_format(val):
                        return ("{0:0." + str(precision) + "f}{1}").format(
                            val, unit_string)
                    func = plain_unit_format
                else:
                    def plain_unit_format(val):
                        return f"{val:g}{unit_string}"
                    func = plain_unit_format
            else:
                raise ValueError(
                    f"'{unit.name}' can not be represented in sexagesimal notation")

        else:
            raise u.UnitsError(
                "The unit value provided is not an angular unit.")

        def do_format(val):
            # Check if value is not nan to avoid ValueErrors when turning it into
            # a hexagesimal string.
            if not np.isnan(val):
                s = func(float(val))
                if alwayssign and not s.startswith('-'):
                    s = '+' + s
                if format == 'latex' or format == 'latex_inline':
                    s = f'${s}$'
                return s
            s = f"{val}"
            return s

        format_ufunc = np.vectorize(do_format, otypes=['U'])
        result = format_ufunc(values)

        if result.ndim == 0:
            result = result[()]
        return result
```
### 42 - astropy/coordinates/angles.py:

Start line: 454, End line: 519

```python
class Angle(u.SpecificTypeQuantity):

    def is_within_bounds(self, lower=None, upper=None):
        """
        Check if all angle(s) satisfy ``lower <= angle < upper``

        If ``lower`` is not specified (or `None`) then no lower bounds check is
        performed.  Likewise ``upper`` can be left unspecified.  For example::

          >>> from astropy.coordinates import Angle
          >>> import astropy.units as u
          >>> a = Angle([-20, 150, 350] * u.deg)
          >>> a.is_within_bounds('0d', '360d')
          False
          >>> a.is_within_bounds(None, '360d')
          True
          >>> a.is_within_bounds(-30 * u.deg, None)
          True

        Parameters
        ----------
        lower : angle-like or None
            Specifies lower bound for checking.  This can be any object
            that can initialize an `~astropy.coordinates.Angle` object, e.g. ``'180d'``,
            ``180 * u.deg``, or ``Angle(180, unit=u.deg)``.
        upper : angle-like or None
            Specifies upper bound for checking.  This can be any object
            that can initialize an `~astropy.coordinates.Angle` object, e.g. ``'180d'``,
            ``180 * u.deg``, or ``Angle(180, unit=u.deg)``.

        Returns
        -------
        is_within_bounds : bool
            `True` if all angles satisfy ``lower <= angle < upper``
        """
        ok = True
        if lower is not None:
            ok &= np.all(Angle(lower) <= self)
        if ok and upper is not None:
            ok &= np.all(self < Angle(upper))
        return bool(ok)

    def _str_helper(self, format=None):
        if self.isscalar:
            return self.to_string(format=format)

        def formatter(x):
            return x.to_string(format=format)

        return np.array2string(self, formatter={'all': formatter})

    def __str__(self):
        return self._str_helper()

    def _repr_latex_(self):
        return self._str_helper(format='latex')


def _no_angle_subclass(obj):
    """Return any Angle subclass objects as an Angle objects.

    This is used to ensure that Latitude and Longitude change to Angle
    objects when they are used in calculations (such as lon/2.)
    """
    if isinstance(obj, tuple):
        return tuple(_no_angle_subclass(_obj) for _obj in obj)

    return obj.view(Angle) if isinstance(obj, (Latitude, Longitude)) else obj
```
### 43 - astropy/coordinates/angles.py:

Start line: 105, End line: 143

```python
class Angle(u.SpecificTypeQuantity):
    _equivalent_unit = u.radian
    _include_easy_conversion_members = True

    def __new__(cls, angle, unit=None, dtype=np.inexact, copy=True, **kwargs):

        if not isinstance(angle, u.Quantity):
            if unit is not None:
                unit = cls._convert_unit_to_angle_unit(u.Unit(unit))

            if isinstance(angle, tuple):
                angle = cls._tuple_to_float(angle, unit)

            elif isinstance(angle, str):
                angle, angle_unit = form.parse_angle(angle, unit)
                if angle_unit is None:
                    angle_unit = unit

                if isinstance(angle, tuple):
                    if angle_unit == u.hourangle:
                        form._check_hour_range(angle[0])
                    form._check_minute_range(angle[1])
                    a = np.abs(angle[0]) + angle[1] / 60.
                    if len(angle) == 3:
                        form._check_second_range(angle[2])
                        a += angle[2] / 3600.

                    angle = np.copysign(a, angle[0])

                if angle_unit is not unit:
                    # Possible conversion to `unit` will be done below.
                    angle = u.Quantity(angle, angle_unit, copy=False)

            elif (isiterable(angle) and
                  not (isinstance(angle, np.ndarray) and
                       angle.dtype.kind not in 'SUVO')):
                angle = [Angle(x, unit, copy=False) for x in angle]

        return super().__new__(cls, angle, unit, dtype=dtype, copy=copy,
                               **kwargs)
```
