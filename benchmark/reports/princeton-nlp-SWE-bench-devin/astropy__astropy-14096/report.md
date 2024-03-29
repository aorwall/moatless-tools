# astropy__astropy-14096

| **astropy/astropy** | `1a4462d72eb03f30dc83a879b1dd57aac8b2c18b` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 3135 |
| **Any found context length** | 3135 |
| **Avg pos** | 40.0 |
| **Min pos** | 10 |
| **Max pos** | 30 |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/astropy/coordinates/sky_coordinate.py b/astropy/coordinates/sky_coordinate.py
--- a/astropy/coordinates/sky_coordinate.py
+++ b/astropy/coordinates/sky_coordinate.py
@@ -894,10 +894,8 @@ def __getattr__(self, attr):
             if frame_cls is not None and self.frame.is_transformable_to(frame_cls):
                 return self.transform_to(attr)
 
-        # Fail
-        raise AttributeError(
-            f"'{self.__class__.__name__}' object has no attribute '{attr}'"
-        )
+        # Call __getattribute__; this will give correct exception.
+        return self.__getattribute__(attr)
 
     def __setattr__(self, attr, val):
         # This is to make anything available through __getattr__ immutable

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| astropy/coordinates/sky_coordinate.py | 897 | 900 | 30 | 1 | 11097


## Problem Statement

```
Subclassed SkyCoord gives misleading attribute access message
I'm trying to subclass `SkyCoord`, and add some custom properties. This all seems to be working fine, but when I have a custom property (`prop` below) that tries to access a non-existent attribute (`random_attr`) below, the error message is misleading because it says `prop` doesn't exist, where it should say `random_attr` doesn't exist.

\`\`\`python
import astropy.coordinates as coord


class custom_coord(coord.SkyCoord):
    @property
    def prop(self):
        return self.random_attr


c = custom_coord('00h42m30s', '+41d12m00s', frame='icrs')
c.prop
\`\`\`

raises
\`\`\`
Traceback (most recent call last):
  File "test.py", line 11, in <module>
    c.prop
  File "/Users/dstansby/miniconda3/lib/python3.7/site-packages/astropy/coordinates/sky_coordinate.py", line 600, in __getattr__
    .format(self.__class__.__name__, attr))
AttributeError: 'custom_coord' object has no attribute 'prop'
\`\`\`

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 astropy/coordinates/sky_coordinate.py** | 929 | 952| 225 | 225 | 19772 | 
| 2 | **1 astropy/coordinates/sky_coordinate.py** | 902 | 927| 249 | 474 | 19772 | 
| 3 | **1 astropy/coordinates/sky_coordinate.py** | 289 | 386| 751 | 1225 | 19772 | 
| 4 | **1 astropy/coordinates/sky_coordinate.py** | 706 | 718| 169 | 1394 | 19772 | 
| 5 | 2 astropy/coordinates/attributes.py | 441 | 473| 215 | 1609 | 23343 | 
| 6 | 3 astropy/io/misc/asdf/tags/coordinates/skycoord.py | 2 | 23| 138 | 1747 | 23498 | 
| 7 | **3 astropy/coordinates/sky_coordinate.py** | 2132 | 2180| 468 | 2215 | 23498 | 
| 8 | **3 astropy/coordinates/sky_coordinate.py** | 76 | 107| 271 | 2486 | 23498 | 
| 9 | 3 astropy/coordinates/attributes.py | 100 | 139| 298 | 2784 | 23498 | 
| **-> 10 <-** | **3 astropy/coordinates/sky_coordinate.py** | 861 | 900| 351 | 3135 | 23498 | 
| 11 | 3 astropy/coordinates/attributes.py | 417 | 439| 212 | 3347 | 23498 | 
| 12 | **3 astropy/coordinates/sky_coordinate.py** | 954 | 989| 266 | 3613 | 23498 | 
| 13 | **3 astropy/coordinates/sky_coordinate.py** | 2077 | 2131| 546 | 4159 | 23498 | 
| 14 | **3 astropy/coordinates/sky_coordinate.py** | 780 | 859| 805 | 4964 | 23498 | 
| 15 | **3 astropy/coordinates/sky_coordinate.py** | 169 | 287| 1599 | 6563 | 23498 | 
| 16 | 4 astropy/io/misc/asdf/tags/coordinates/frames.py | 114 | 130| 139 | 6702 | 24638 | 
| 17 | 5 astropy/coordinates/baseframe.py | 1713 | 1767| 458 | 7160 | 41142 | 
| 18 | 6 astropy/coordinates/builtin_frames/skyoffset.py | 145 | 167| 269 | 7429 | 42900 | 
| 19 | 7 astropy/coordinates/spectral_coordinate.py | 245 | 306| 515 | 7944 | 49485 | 
| 20 | 8 astropy/coordinates/errors.py | 182 | 206| 157 | 8101 | 50595 | 
| 21 | 9 astropy/utils/decorators.py | 743 | 758| 160 | 8261 | 59236 | 
| 22 | **9 astropy/coordinates/sky_coordinate.py** | 1 | 34| 203 | 8464 | 59236 | 
| 23 | **9 astropy/coordinates/sky_coordinate.py** | 37 | 74| 292 | 8756 | 59236 | 
| 24 | 9 astropy/coordinates/attributes.py | 221 | 269| 319 | 9075 | 59236 | 
| 25 | **9 astropy/coordinates/sky_coordinate.py** | 478 | 509| 303 | 9378 | 59236 | 
| 26 | **9 astropy/coordinates/sky_coordinate.py** | 2034 | 2076| 503 | 9881 | 59236 | 
| 27 | 10 astropy/utils/data_info.py | 211 | 245| 212 | 10093 | 65260 | 
| 28 | 11 astropy/cosmology/core.py | 592 | 613| 153 | 10246 | 70253 | 
| 29 | **11 astropy/coordinates/sky_coordinate.py** | 580 | 616| 332 | 10578 | 70253 | 
| **-> 30 <-** | **11 astropy/coordinates/sky_coordinate.py** | 109 | 2223| 519 | 11097 | 70253 | 
| 31 | 11 astropy/io/misc/asdf/tags/coordinates/frames.py | 84 | 111| 210 | 11307 | 70253 | 
| 32 | 12 astropy/coordinates/sky_coordinate_parsers.py | 150 | 223| 667 | 11974 | 76562 | 
| 33 | 13 astropy/coordinates/builtin_frames/galactocentric.py | 573 | 591| 161 | 12135 | 82828 | 
| 34 | 14 astropy/utils/metadata.py | 526 | 544| 147 | 12282 | 86843 | 
| 35 | 14 astropy/coordinates/baseframe.py | 225 | 304| 769 | 13051 | 86843 | 
| 36 | 15 astropy/units/astrophys.py | 237 | 257| 141 | 13192 | 88465 | 
| 37 | 16 astropy/io/misc/asdf/tags/coordinates/spectralcoord.py | 2 | 45| 340 | 13532 | 88822 | 
| 38 | 16 astropy/coordinates/attributes.py | 4 | 64| 429 | 13961 | 88822 | 
| 39 | **16 astropy/coordinates/sky_coordinate.py** | 1961 | 2033| 795 | 14756 | 88822 | 
| 40 | 16 astropy/utils/decorators.py | 760 | 787| 200 | 14956 | 88822 | 
| 41 | **16 astropy/coordinates/sky_coordinate.py** | 1103 | 1146| 313 | 15269 | 88822 | 
| 42 | 16 astropy/coordinates/sky_coordinate_parsers.py | 623 | 671| 463 | 15732 | 88822 | 
| 43 | 16 astropy/utils/decorators.py | 614 | 724| 793 | 16525 | 88822 | 
| 44 | 16 astropy/coordinates/sky_coordinate_parsers.py | 289 | 348| 518 | 17043 | 88822 | 
| 45 | 16 astropy/coordinates/spectral_coordinate.py | 184 | 243| 585 | 17628 | 88822 | 
| 46 | 16 astropy/coordinates/baseframe.py | 598 | 677| 686 | 18314 | 88822 | 
| 47 | 16 astropy/utils/decorators.py | 726 | 741| 159 | 18473 | 88822 | 
| 48 | 16 astropy/coordinates/sky_coordinate_parsers.py | 415 | 503| 777 | 19250 | 88822 | 
| 49 | 16 astropy/io/misc/asdf/tags/coordinates/frames.py | 52 | 82| 212 | 19462 | 88822 | 
| 50 | 16 astropy/coordinates/baseframe.py | 2002 | 2036| 238 | 19700 | 88822 | 
| 51 | 16 astropy/coordinates/builtin_frames/skyoffset.py | 169 | 197| 238 | 19938 | 88822 | 
| 52 | 16 astropy/coordinates/sky_coordinate_parsers.py | 74 | 149| 738 | 20676 | 88822 | 
| 53 | 16 astropy/coordinates/baseframe.py | 711 | 754| 320 | 20996 | 88822 | 
| 54 | 16 astropy/coordinates/sky_coordinate_parsers.py | 384 | 413| 240 | 21236 | 88822 | 
| 55 | 16 astropy/coordinates/builtin_frames/galactocentric.py | 3 | 37| 261 | 21497 | 88822 | 
| 56 | 17 astropy/io/fits/column.py | 911 | 979| 516 | 22013 | 111326 | 
| 57 | **17 astropy/coordinates/sky_coordinate.py** | 388 | 418| 233 | 22246 | 111326 | 
| 58 | 17 astropy/coordinates/sky_coordinate_parsers.py | 607 | 620| 156 | 22402 | 111326 | 
| 59 | 18 astropy/units/equivalencies.py | 893 | 913| 144 | 22546 | 120388 | 
| 60 | 18 astropy/utils/metadata.py | 406 | 449| 264 | 22810 | 120388 | 
| 61 | 18 astropy/coordinates/baseframe.py | 835 | 887| 551 | 23361 | 120388 | 
| 62 | 18 astropy/coordinates/builtin_frames/skyoffset.py | 2 | 58| 437 | 23798 | 120388 | 
| 63 | 18 astropy/coordinates/baseframe.py | 306 | 392| 707 | 24505 | 120388 | 
| 64 | 19 astropy/cosmology/flrw/__init__.py | 22 | 56| 255 | 24760 | 120808 | 
| 65 | 19 astropy/coordinates/baseframe.py | 1144 | 1243| 943 | 25703 | 120808 | 
| 66 | 19 astropy/coordinates/spectral_coordinate.py | 308 | 401| 773 | 26476 | 120808 | 
| 67 | **19 astropy/coordinates/sky_coordinate.py** | 2181 | 2224| 417 | 26893 | 120808 | 
| 68 | 19 astropy/coordinates/attributes.py | 201 | 219| 142 | 27035 | 120808 | 
| 69 | 19 astropy/coordinates/builtin_frames/galactocentric.py | 149 | 227| 750 | 27785 | 120808 | 
| 70 | **19 astropy/coordinates/sky_coordinate.py** | 1696 | 1714| 202 | 27987 | 120808 | 
| 71 | 20 astropy/extern/configobj/configobj.py | 194 | 280| 462 | 28449 | 139233 | 
| 72 | 20 astropy/coordinates/baseframe.py | 9 | 38| 152 | 28601 | 139233 | 
| 73 | 21 astropy/coordinates/matching.py | 177 | 197| 263 | 28864 | 144417 | 
| 74 | 21 astropy/coordinates/baseframe.py | 1660 | 1711| 478 | 29342 | 144417 | 
| 75 | **21 astropy/coordinates/sky_coordinate.py** | 1791 | 1822| 278 | 29620 | 144417 | 
| 76 | **21 astropy/coordinates/sky_coordinate.py** | 618 | 704| 796 | 30416 | 144417 | 
| 77 | 21 astropy/cosmology/core.py | 3 | 52| 247 | 30663 | 144417 | 
| 78 | 21 astropy/coordinates/baseframe.py | 116 | 134| 186 | 30849 | 144417 | 
| 79 | 21 astropy/coordinates/matching.py | 438 | 521| 777 | 31626 | 144417 | 
| 80 | 21 astropy/coordinates/baseframe.py | 889 | 923| 295 | 31921 | 144417 | 
| 81 | 21 astropy/coordinates/sky_coordinate_parsers.py | 40 | 71| 227 | 32148 | 144417 | 
| 82 | 21 astropy/utils/metadata.py | 485 | 509| 210 | 32358 | 144417 | 
| 83 | 21 astropy/io/fits/column.py | 1230 | 1300| 663 | 33021 | 144417 | 
| 84 | 21 astropy/coordinates/baseframe.py | 1115 | 1143| 287 | 33308 | 144417 | 
| 85 | 21 astropy/coordinates/baseframe.py | 756 | 769| 140 | 33448 | 144417 | 
| 86 | 22 astropy/visualization/wcsaxes/core.py | 331 | 358| 244 | 33692 | 151732 | 
| 87 | 23 astropy/io/misc/asdf/tags/coordinates/__init__.py | 1 | 2| 0 | 33692 | 151749 | 
| 88 | 24 astropy/io/votable/tree.py | 412 | 435| 191 | 33883 | 181109 | 
| 89 | 24 astropy/utils/decorators.py | 234 | 322| 529 | 34412 | 181109 | 
| 90 | 24 astropy/visualization/wcsaxes/core.py | 3 | 27| 216 | 34628 | 181109 | 
| 91 | 24 astropy/coordinates/sky_coordinate_parsers.py | 226 | 288| 620 | 35248 | 181109 | 
| 92 | 24 astropy/extern/configobj/configobj.py | 106 | 136| 191 | 35439 | 181109 | 
| 93 | 24 astropy/coordinates/attributes.py | 476 | 500| 193 | 35632 | 181109 | 
| 94 | 24 astropy/io/votable/tree.py | 438 | 464| 205 | 35837 | 181109 | 
| 95 | 25 astropy/nddata/utils.py | 6 | 37| 157 | 35994 | 188646 | 
| 96 | 25 astropy/coordinates/sky_coordinate_parsers.py | 3 | 37| 248 | 36242 | 188646 | 


### Hint

```
This is because the property raises an `AttributeError`, which causes Python to call `__getattr__`. You can catch the `AttributeError` in the property and raise another exception with a better message.
The problem is it's a nightmare for debugging at the moment. If I had a bunch of different attributes in `prop(self)`, and only one didn't exist, there's no way of knowing which one doesn't exist. Would it possible modify the `__getattr__` method in `SkyCoord` to raise the original `AttributeError`?
No, `__getattr__` does not know about the other errors. So the only way is to catch the AttributeError and raise it as another error...
https://stackoverflow.com/questions/36575068/attributeerrors-undesired-interaction-between-property-and-getattr
@adrn , since you added the milestone, what is its status for feature freeze tomorrow?
Milestone is removed as there hasn't been any updates on this, and the issue hasn't been resolved on master.
```

## Patch

```diff
diff --git a/astropy/coordinates/sky_coordinate.py b/astropy/coordinates/sky_coordinate.py
--- a/astropy/coordinates/sky_coordinate.py
+++ b/astropy/coordinates/sky_coordinate.py
@@ -894,10 +894,8 @@ def __getattr__(self, attr):
             if frame_cls is not None and self.frame.is_transformable_to(frame_cls):
                 return self.transform_to(attr)
 
-        # Fail
-        raise AttributeError(
-            f"'{self.__class__.__name__}' object has no attribute '{attr}'"
-        )
+        # Call __getattribute__; this will give correct exception.
+        return self.__getattribute__(attr)
 
     def __setattr__(self, attr, val):
         # This is to make anything available through __getattr__ immutable

```

## Test Patch

```diff
diff --git a/astropy/coordinates/tests/test_sky_coord.py b/astropy/coordinates/tests/test_sky_coord.py
--- a/astropy/coordinates/tests/test_sky_coord.py
+++ b/astropy/coordinates/tests/test_sky_coord.py
@@ -2165,3 +2165,21 @@ def test_match_to_catalog_3d_and_sky():
     npt.assert_array_equal(idx, [0, 1, 2, 3])
     assert_allclose(angle, 0 * u.deg, atol=1e-14 * u.deg, rtol=0)
     assert_allclose(distance, 0 * u.kpc, atol=1e-14 * u.kpc, rtol=0)
+
+
+def test_subclass_property_exception_error():
+    """Regression test for gh-8340.
+
+    Non-existing attribute access inside a property should give attribute
+    error for the attribute, not for the property.
+    """
+
+    class custom_coord(SkyCoord):
+        @property
+        def prop(self):
+            return self.random_attr
+
+    c = custom_coord("00h42m30s", "+41d12m00s", frame="icrs")
+    with pytest.raises(AttributeError, match="random_attr"):
+        # Before this matched "prop" rather than "random_attr"
+        c.prop

```


## Code snippets

### 1 - astropy/coordinates/sky_coordinate.py:

Start line: 929, End line: 952

```python
class SkyCoord(ShapedLikeNDArray):

    def __delattr__(self, attr):
        # mirror __setattr__ above
        if "_sky_coord_frame" in self.__dict__:
            if self._is_name(attr):
                raise AttributeError(f"'{attr}' is immutable")

            if not attr.startswith("_") and hasattr(self._sky_coord_frame, attr):
                delattr(self._sky_coord_frame, attr)
                return

            frame_cls = frame_transform_graph.lookup_name(attr)
            if frame_cls is not None and self.frame.is_transformable_to(frame_cls):
                raise AttributeError(f"'{attr}' is immutable")

        if attr in frame_transform_graph.frame_attributes:
            # All possible frame attributes can be deleted, but need to remove
            # the corresponding private variable.  See __getattr__ above.
            super().__delattr__("_" + attr)
            # Also remove it from the set of extra attributes
            self._extra_frameattr_names -= {attr}

        else:
            # Otherwise, do the standard Python attribute setting
            super().__delattr__(attr)
```
### 2 - astropy/coordinates/sky_coordinate.py:

Start line: 902, End line: 927

```python
class SkyCoord(ShapedLikeNDArray):

    def __setattr__(self, attr, val):
        # This is to make anything available through __getattr__ immutable
        if "_sky_coord_frame" in self.__dict__:
            if self._is_name(attr):
                raise AttributeError(f"'{attr}' is immutable")

            if not attr.startswith("_") and hasattr(self._sky_coord_frame, attr):
                setattr(self._sky_coord_frame, attr, val)
                return

            frame_cls = frame_transform_graph.lookup_name(attr)
            if frame_cls is not None and self.frame.is_transformable_to(frame_cls):
                raise AttributeError(f"'{attr}' is immutable")

        if attr in frame_transform_graph.frame_attributes:
            # All possible frame attributes can be set, but only via a private
            # variable.  See __getattr__ above.
            super().__setattr__("_" + attr, val)
            # Validate it
            frame_transform_graph.frame_attributes[attr].__get__(self)
            # And add to set of extra attributes
            self._extra_frameattr_names |= {attr}

        else:
            # Otherwise, do the standard Python attribute setting
            super().__setattr__(attr, val)
```
### 3 - astropy/coordinates/sky_coordinate.py:

Start line: 289, End line: 386

```python
class SkyCoord(ShapedLikeNDArray):

    # Declare that SkyCoord can be used as a Table column by defining the
    # info property.
    info = SkyCoordInfo()

    def __init__(self, *args, copy=True, **kwargs):
        # these are frame attributes set on this SkyCoord but *not* a part of
        # the frame object this SkyCoord contains
        self._extra_frameattr_names = set()

        # If all that is passed in is a frame instance that already has data,
        # we should bypass all of the parsing and logic below. This is here
        # to make this the fastest way to create a SkyCoord instance. Many of
        # the classmethods implemented for performance enhancements will use
        # this as the initialization path
        if (
            len(args) == 1
            and len(kwargs) == 0
            and isinstance(args[0], (BaseCoordinateFrame, SkyCoord))
        ):
            coords = args[0]
            if isinstance(coords, SkyCoord):
                self._extra_frameattr_names = coords._extra_frameattr_names
                self.info = coords.info

                # Copy over any extra frame attributes
                for attr_name in self._extra_frameattr_names:
                    # Setting it will also validate it.
                    setattr(self, attr_name, getattr(coords, attr_name))

                coords = coords.frame

            if not coords.has_data:
                raise ValueError(
                    "Cannot initialize from a coordinate frame "
                    "instance without coordinate data"
                )

            if copy:
                self._sky_coord_frame = coords.copy()
            else:
                self._sky_coord_frame = coords

        else:
            # Get the frame instance without coordinate data but with all frame
            # attributes set - these could either have been passed in with the
            # frame as an instance, or passed in as kwargs here
            frame_cls, frame_kwargs = _get_frame_without_data(args, kwargs)

            # Parse the args and kwargs to assemble a sanitized and validated
            # kwargs dict for initializing attributes for this object and for
            # creating the internal self._sky_coord_frame object
            args = list(args)  # Make it mutable
            skycoord_kwargs, components, info = _parse_coordinate_data(
                frame_cls(**frame_kwargs), args, kwargs
            )

            # In the above two parsing functions, these kwargs were identified
            # as valid frame attributes for *some* frame, but not the frame that
            # this SkyCoord will have. We keep these attributes as special
            # skycoord frame attributes:
            for attr in skycoord_kwargs:
                # Setting it will also validate it.
                setattr(self, attr, skycoord_kwargs[attr])

            if info is not None:
                self.info = info

            # Finally make the internal coordinate object.
            frame_kwargs.update(components)
            self._sky_coord_frame = frame_cls(copy=copy, **frame_kwargs)

            if not self._sky_coord_frame.has_data:
                raise ValueError("Cannot create a SkyCoord without data")

    @property
    def frame(self):
        return self._sky_coord_frame

    @property
    def representation_type(self):
        return self.frame.representation_type

    @representation_type.setter
    def representation_type(self, value):
        self.frame.representation_type = value

    # TODO: remove these in future
    @property
    def representation(self):
        return self.frame.representation

    @representation.setter
    def representation(self, value):
        self.frame.representation = value

    @property
    def shape(self):
        return self.frame.shape
```
### 4 - astropy/coordinates/sky_coordinate.py:

Start line: 706, End line: 718

```python
class SkyCoord(ShapedLikeNDArray):

    def transform_to(self, frame, merge_attributes=True):

        # Finally make the new SkyCoord object from the `new_coord` and
        # remaining frame_kwargs that are not frame_attributes in `new_coord`.
        for attr in set(new_coord.frame_attributes) & set(frame_kwargs.keys()):
            frame_kwargs.pop(attr)

        # Always remove the origin frame attribute, as that attribute only makes
        # sense with a SkyOffsetFrame (in which case it will be stored on the frame).
        # See gh-11277.
        # TODO: Should it be a property of the frame attribute that it can
        # or cannot be stored on a SkyCoord?
        frame_kwargs.pop("origin", None)

        return self.__class__(new_coord, **frame_kwargs)
```
### 5 - astropy/coordinates/attributes.py:

Start line: 441, End line: 473

```python
class CoordinateAttribute(Attribute):

    def convert_input(self, value):
        """
        Checks that the input is a SkyCoord with the necessary units (or the
        special value ``None``).

        Parameters
        ----------
        value : object
            Input value to be converted.

        Returns
        -------
        out, converted : correctly-typed object, boolean
            Tuple consisting of the correctly-typed object and a boolean which
            indicates if conversion was actually performed.

        Raises
        ------
        ValueError
            If the input is not valid for this attribute.
        """
        from .sky_coordinate import SkyCoord

        if value is None:
            return None, False
        elif isinstance(value, SkyCoord) and isinstance(value.frame, self._frame):
            return value.frame, True
        elif isinstance(value, self._frame):
            return value, False
        else:
            value = SkyCoord(value)  # always make the value a SkyCoord
            transformedobj = value.transform_to(self._frame)
            return transformedobj.frame, True
```
### 6 - astropy/io/misc/asdf/tags/coordinates/skycoord.py:

Start line: 2, End line: 23

```python
from astropy.coordinates import SkyCoord
from astropy.coordinates.tests.helper import skycoord_equal
from astropy.io.misc.asdf.types import AstropyType


class SkyCoordType(AstropyType):
    name = "coordinates/skycoord"
    types = [SkyCoord]
    version = "1.0.0"

    @classmethod
    def to_tree(cls, obj, ctx):
        return obj.info._represent_as_dict()

    @classmethod
    def from_tree(cls, tree, ctx):
        return SkyCoord.info._construct_from_dict(tree)

    @classmethod
    def assert_equal(cls, old, new):
        assert skycoord_equal(old, new)
```
### 7 - astropy/coordinates/sky_coordinate.py:

Start line: 2132, End line: 2180

```python
class SkyCoord(ShapedLikeNDArray):
    @classmethod
    def guess_from_table(cls, table, **coord_kwargs):
        # ... other code
        for comp_name in representation_component_names:
            # this matches things like 'ra[...]'' but *not* 'rad'.
            # note that the "_" must be in there explicitly, because
            # "alphanumeric" usually includes underscores.
            starts_with_comp = comp_name + r"(\W|\b|_)"
            # this part matches stuff like 'center_ra', but *not*
            # 'aura'
            ends_with_comp = r".*(\W|\b|_)" + comp_name + r"\b"
            # the final regex ORs together the two patterns
            rex = re.compile(
                rf"({starts_with_comp})|({ends_with_comp})", re.IGNORECASE | re.UNICODE
            )

            # find all matches
            matches = {col_name for col_name in table.colnames if rex.match(col_name)}

            # now need to select among matches, also making sure we don't have
            # an exact match with another component
            if len(matches) == 0:  # no matches
                continue
            elif len(matches) == 1:  # only one match
                col_name = matches.pop()
            else:  # more than 1 match
                # try to sieve out other components
                matches -= representation_component_names - {comp_name}
                # if there's only one remaining match, it worked.
                if len(matches) == 1:
                    col_name = matches.pop()
                else:
                    raise ValueError(
                        f'Found at least two matches for component "{comp_name}":'
                        f' "{matches}". Cannot guess coordinates from a table with this'
                        " ambiguity."
                    )

            comp_kwargs[comp_name] = table[col_name]

        for k, v in comp_kwargs.items():
            if k in coord_kwargs:
                raise ValueError(
                    f'Found column "{v.name}" in table, but it was already provided as'
                    ' "{k}" keyword to guess_from_table function.'
                )
            else:
                coord_kwargs[k] = v

        return cls(**coord_kwargs)

    # Name resolve
```
### 8 - astropy/coordinates/sky_coordinate.py:

Start line: 76, End line: 107

```python
class SkyCoordInfo(MixinInfo):

    def _represent_as_dict(self):
        sc = self._parent
        attrs = list(sc.representation_component_names)

        # Don't output distance unless it's actually distance.
        if isinstance(sc.data, UnitSphericalRepresentation):
            attrs = attrs[:-1]

        diff = sc.data.differentials.get("s")
        if diff is not None:
            diff_attrs = list(sc.get_representation_component_names("s"))
            # Don't output proper motions if they haven't been specified.
            if isinstance(diff, RadialDifferential):
                diff_attrs = diff_attrs[2:]
            # Don't output radial velocity unless it's actually velocity.
            elif isinstance(
                diff, (UnitSphericalDifferential, UnitSphericalCosLatDifferential)
            ):
                diff_attrs = diff_attrs[:-1]
            attrs.extend(diff_attrs)

        attrs.extend(frame_transform_graph.frame_attributes.keys())

        out = super()._represent_as_dict(attrs)

        out["representation_type"] = sc.representation_type.get_name()
        out["frame"] = sc.frame.name
        # Note that sc.info.unit is a fake composite unit (e.g. 'deg,deg,None'
        # or None,None,m) and is not stored.  The individual attributes have
        # units.

        return out
```
### 9 - astropy/coordinates/attributes.py:

Start line: 100, End line: 139

```python
class Attribute:

    def __get__(self, instance, frame_cls=None):
        if instance is None:
            out = self.default
        else:
            out = getattr(instance, "_" + self.name, self.default)
            if out is None:
                out = getattr(instance, self.secondary_attribute, self.default)

        out, converted = self.convert_input(out)
        if instance is not None:
            # None if instance (frame) has no data!
            instance_shape = getattr(instance, "shape", None)
            if instance_shape is not None and (
                getattr(out, "shape", ()) and out.shape != instance_shape
            ):
                # If the shapes do not match, try broadcasting.
                try:
                    if isinstance(out, ShapedLikeNDArray):
                        out = out._apply(
                            np.broadcast_to, shape=instance_shape, subok=True
                        )
                    else:
                        out = np.broadcast_to(out, instance_shape, subok=True)
                except ValueError:
                    # raise more informative exception.
                    raise ValueError(
                        f"attribute {self.name} should be scalar or have shape"
                        f" {instance_shape}, but it has shape {out.shape} and could not"
                        " be broadcast."
                    )

                converted = True

            if converted:
                setattr(instance, "_" + self.name, out)

        return out

    def __set__(self, instance, val):
        raise AttributeError("Cannot set frame attribute")
```
### 10 - astropy/coordinates/sky_coordinate.py:

Start line: 861, End line: 900

```python
class SkyCoord(ShapedLikeNDArray):

    def _is_name(self, string):
        """
        Returns whether a string is one of the aliases for the frame.
        """
        return self.frame.name == string or (
            isinstance(self.frame.name, list) and string in self.frame.name
        )

    def __getattr__(self, attr):
        """
        Overrides getattr to return coordinates that this can be transformed
        to, based on the alias attr in the primary transform graph.
        """
        if "_sky_coord_frame" in self.__dict__:
            if self._is_name(attr):
                return self  # Should this be a deepcopy of self?

            # Anything in the set of all possible frame_attr_names is handled
            # here. If the attr is relevant for the current frame then delegate
            # to self.frame otherwise get it from self._<attr>.
            if attr in frame_transform_graph.frame_attributes:
                if attr in self.frame.frame_attributes:
                    return getattr(self.frame, attr)
                else:
                    return getattr(self, "_" + attr, None)

            # Some attributes might not fall in the above category but still
            # are available through self._sky_coord_frame.
            if not attr.startswith("_") and hasattr(self._sky_coord_frame, attr):
                return getattr(self._sky_coord_frame, attr)

            # Try to interpret as a new frame for transforming.
            frame_cls = frame_transform_graph.lookup_name(attr)
            if frame_cls is not None and self.frame.is_transformable_to(frame_cls):
                return self.transform_to(attr)

        # Fail
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{attr}'"
        )
```
### 12 - astropy/coordinates/sky_coordinate.py:

Start line: 954, End line: 989

```python
class SkyCoord(ShapedLikeNDArray):

    def __dir__(self):
        """
        Override the builtin `dir` behavior to include:
        - Transforms available by aliases
        - Attribute / methods of the underlying self.frame object
        """
        dir_values = set(super().__dir__())

        # determine the aliases that this can be transformed to.
        for name in frame_transform_graph.get_names():
            frame_cls = frame_transform_graph.lookup_name(name)
            if self.frame.is_transformable_to(frame_cls):
                dir_values.add(name)

        # Add public attributes of self.frame
        dir_values.update(
            {attr for attr in dir(self.frame) if not attr.startswith("_")}
        )

        # Add all possible frame attributes
        dir_values.update(frame_transform_graph.frame_attributes.keys())

        return sorted(dir_values)

    def __repr__(self):
        clsnm = self.__class__.__name__
        coonm = self.frame.__class__.__name__
        frameattrs = self.frame._frame_attrs_repr()
        if frameattrs:
            frameattrs = ": " + frameattrs

        data = self.frame._data_repr()
        if data:
            data = ": " + data

        return f"<{clsnm} ({coonm}{frameattrs}){data}>"
```
### 13 - astropy/coordinates/sky_coordinate.py:

Start line: 2077, End line: 2131

```python
class SkyCoord(ShapedLikeNDArray):
    @classmethod
    def guess_from_table(cls, table, **coord_kwargs):
        r"""
        A convenience method to create and return a new `SkyCoord` from the data
        in an astropy Table.

        This method matches table columns that start with the case-insensitive
        names of the the components of the requested frames (including
        differentials), if they are also followed by a non-alphanumeric
        character. It will also match columns that *end* with the component name
        if a non-alphanumeric character is *before* it.

        For example, the first rule means columns with names like
        ``'RA[J2000]'`` or ``'ra'`` will be interpreted as ``ra`` attributes for
        `~astropy.coordinates.ICRS` frames, but ``'RAJ2000'`` or ``'radius'``
        are *not*. Similarly, the second rule applied to the
        `~astropy.coordinates.Galactic` frame means that a column named
        ``'gal_l'`` will be used as the the ``l`` component, but ``gall`` or
        ``'fill'`` will not.

        The definition of alphanumeric here is based on Unicode's definition
        of alphanumeric, except without ``_`` (which is normally considered
        alphanumeric).  So for ASCII, this means the non-alphanumeric characters
        are ``<space>_!"#$%&'()*+,-./\:;<=>?@[]^`{|}~``).

        Parameters
        ----------
        table : `~astropy.table.Table` or subclass
            The table to load data from.
        **coord_kwargs
            Any additional keyword arguments are passed directly to this class's
            constructor.

        Returns
        -------
        newsc : `~astropy.coordinates.SkyCoord` or subclass
            The new `SkyCoord` (or subclass) object.

        Raises
        ------
        ValueError
            If more than one match is found in the table for a component,
            unless the additional matches are also valid frame component names.
            If a "coord_kwargs" is provided for a value also found in the table.

        """
        _frame_cls, _frame_kwargs = _get_frame_without_data([], coord_kwargs)
        frame = _frame_cls(**_frame_kwargs)
        coord_kwargs["frame"] = coord_kwargs.get("frame", frame)

        representation_component_names = set(
            frame.get_representation_component_names()
        ).union(set(frame.get_representation_component_names("s")))

        comp_kwargs = {}
        # ... other code
```
### 14 - astropy/coordinates/sky_coordinate.py:

Start line: 780, End line: 859

```python
class SkyCoord(ShapedLikeNDArray):

    def apply_space_motion(self, new_obstime=None, dt=None):
        # ... other code
        if dt is None:
            # self.obstime is not None and new_obstime is not None b/c of above
            # checks
            t2 = new_obstime
        else:
            # new_obstime is definitely None b/c of the above checks
            if t1 is None:
                # MAGIC NUMBER: if the current SkyCoord object has no obstime,
                # assume J2000 to do the dt offset. This is not actually used
                # for anything except a delta-t in starpm, so it's OK that it's
                # not necessarily the "real" obstime
                t1 = Time("J2000")
                new_obstime = None  # we don't actually know the initial obstime
                t2 = t1 + dt
            else:
                t2 = t1 + dt
                new_obstime = t2
        # starpm wants tdb time
        t1 = t1.tdb
        t2 = t2.tdb

        # proper motion in RA should not include the cos(dec) term, see the
        # erfa function eraStarpv, comment (4).  So we convert to the regular
        # spherical differentials.
        icrsrep = self.icrs.represent_as(SphericalRepresentation, SphericalDifferential)
        icrsvel = icrsrep.differentials["s"]

        parallax_zero = False
        try:
            plx = icrsrep.distance.to_value(u.arcsecond, u.parallax())
        except u.UnitConversionError:  # No distance: set to 0 by convention
            plx = 0.0
            parallax_zero = True

        try:
            rv = icrsvel.d_distance.to_value(u.km / u.s)
        except u.UnitConversionError:  # No RV
            rv = 0.0

        starpm = erfa.pmsafe(
            icrsrep.lon.radian,
            icrsrep.lat.radian,
            icrsvel.d_lon.to_value(u.radian / u.yr),
            icrsvel.d_lat.to_value(u.radian / u.yr),
            plx,
            rv,
            t1.jd1,
            t1.jd2,
            t2.jd1,
            t2.jd2,
        )

        if parallax_zero:
            new_distance = None
        else:
            new_distance = Distance(parallax=starpm[4] << u.arcsec)

        icrs2 = ICRS(
            ra=u.Quantity(starpm[0], u.radian, copy=False),
            dec=u.Quantity(starpm[1], u.radian, copy=False),
            pm_ra=u.Quantity(starpm[2], u.radian / u.yr, copy=False),
            pm_dec=u.Quantity(starpm[3], u.radian / u.yr, copy=False),
            distance=new_distance,
            radial_velocity=u.Quantity(starpm[5], u.km / u.s, copy=False),
            differential_type=SphericalDifferential,
        )

        # Update the obstime of the returned SkyCoord, and need to carry along
        # the frame attributes
        frattrs = {
            attrnm: getattr(self, attrnm) for attrnm in self._extra_frameattr_names
        }
        frattrs["obstime"] = new_obstime
        result = self.__class__(icrs2, **frattrs).transform_to(self.frame)

        # Without this the output might not have the right differential type.
        # Not sure if this fixes the problem or just hides it.  See #11932
        result.differential_type = self.differential_type

        return result
```
### 15 - astropy/coordinates/sky_coordinate.py:

Start line: 169, End line: 287

```python
class SkyCoord(ShapedLikeNDArray):
    """High-level object providing a flexible interface for celestial coordinate
    representation, manipulation, and transformation between systems.

    The `SkyCoord` class accepts a wide variety of inputs for initialization. At
    a minimum these must provide one or more celestial coordinate values with
    unambiguous units.  Inputs may be scalars or lists/tuples/arrays, yielding
    scalar or array coordinates (can be checked via ``SkyCoord.isscalar``).
    Typically one also specifies the coordinate frame, though this is not
    required. The general pattern for spherical representations is::

      SkyCoord(COORD, [FRAME], keyword_args ...)
      SkyCoord(LON, LAT, [FRAME], keyword_args ...)
      SkyCoord(LON, LAT, [DISTANCE], frame=FRAME, unit=UNIT, keyword_args ...)
      SkyCoord([FRAME], <lon_attr>=LON, <lat_attr>=LAT, keyword_args ...)

    It is also possible to input coordinate values in other representations
    such as cartesian or cylindrical.  In this case one includes the keyword
    argument ``representation_type='cartesian'`` (for example) along with data
    in ``x``, ``y``, and ``z``.

    See also: https://docs.astropy.org/en/stable/coordinates/

    Examples
    --------
    The examples below illustrate common ways of initializing a `SkyCoord`
    object.  For a complete description of the allowed syntax see the
    full coordinates documentation.  First some imports::

      >>> from astropy.coordinates import SkyCoord  # High-level coordinates
      >>> from astropy.coordinates import ICRS, Galactic, FK4, FK5  # Low-level frames
      >>> from astropy.coordinates import Angle, Latitude, Longitude  # Angles
      >>> import astropy.units as u

    The coordinate values and frame specification can now be provided using
    positional and keyword arguments::

      >>> c = SkyCoord(10, 20, unit="deg")  # defaults to ICRS frame
      >>> c = SkyCoord([1, 2, 3], [-30, 45, 8], frame="icrs", unit="deg")  # 3 coords

      >>> coords = ["1:12:43.2 +31:12:43", "1 12 43.2 +31 12 43"]
      >>> c = SkyCoord(coords, frame=FK4, unit=(u.hourangle, u.deg), obstime="J1992.21")

      >>> c = SkyCoord("1h12m43.2s +1d12m43s", frame=Galactic)  # Units from string
      >>> c = SkyCoord(frame="galactic", l="1h12m43.2s", b="+1d12m43s")

      >>> ra = Longitude([1, 2, 3], unit=u.deg)  # Could also use Angle
      >>> dec = np.array([4.5, 5.2, 6.3]) * u.deg  # Astropy Quantity
      >>> c = SkyCoord(ra, dec, frame='icrs')
      >>> c = SkyCoord(frame=ICRS, ra=ra, dec=dec, obstime='2001-01-02T12:34:56')

      >>> c = FK4(1 * u.deg, 2 * u.deg)  # Uses defaults for obstime, equinox
      >>> c = SkyCoord(c, obstime='J2010.11', equinox='B1965')  # Override defaults

      >>> c = SkyCoord(w=0, u=1, v=2, unit='kpc', frame='galactic',
      ...              representation_type='cartesian')

      >>> c = SkyCoord([ICRS(ra=1*u.deg, dec=2*u.deg), ICRS(ra=3*u.deg, dec=4*u.deg)])

    Velocity components (proper motions or radial velocities) can also be
    provided in a similar manner::

      >>> c = SkyCoord(ra=1*u.deg, dec=2*u.deg, radial_velocity=10*u.km/u.s)

      >>> c = SkyCoord(ra=1*u.deg, dec=2*u.deg, pm_ra_cosdec=2*u.mas/u.yr, pm_dec=1*u.mas/u.yr)

    As shown, the frame can be a `~astropy.coordinates.BaseCoordinateFrame`
    class or the corresponding string alias.  The frame classes that are built in
    to astropy are `ICRS`, `FK5`, `FK4`, `FK4NoETerms`, and `Galactic`.
    The string aliases are simply lower-case versions of the class name, and
    allow for creating a `SkyCoord` object and transforming frames without
    explicitly importing the frame classes.

    Parameters
    ----------
    frame : `~astropy.coordinates.BaseCoordinateFrame` class or string, optional
        Type of coordinate frame this `SkyCoord` should represent. Defaults to
        to ICRS if not given or given as None.
    unit : `~astropy.units.Unit`, string, or tuple of :class:`~astropy.units.Unit` or str, optional
        Units for supplied coordinate values.
        If only one unit is supplied then it applies to all values.
        Note that passing only one unit might lead to unit conversion errors
        if the coordinate values are expected to have mixed physical meanings
        (e.g., angles and distances).
    obstime : time-like, optional
        Time(s) of observation.
    equinox : time-like, optional
        Coordinate frame equinox time.
    representation_type : str or Representation class
        Specifies the representation, e.g. 'spherical', 'cartesian', or
        'cylindrical'.  This affects the positional args and other keyword args
        which must correspond to the given representation.
    copy : bool, optional
        If `True` (default), a copy of any coordinate data is made.  This
        argument can only be passed in as a keyword argument.
    **keyword_args
        Other keyword arguments as applicable for user-defined coordinate frames.
        Common options include:

        ra, dec : angle-like, optional
            RA and Dec for frames where ``ra`` and ``dec`` are keys in the
            frame's ``representation_component_names``, including `ICRS`,
            `FK5`, `FK4`, and `FK4NoETerms`.
        pm_ra_cosdec, pm_dec  : `~astropy.units.Quantity` ['angular speed'], optional
            Proper motion components, in angle per time units.
        l, b : angle-like, optional
            Galactic ``l`` and ``b`` for for frames where ``l`` and ``b`` are
            keys in the frame's ``representation_component_names``, including
            the `Galactic` frame.
        pm_l_cosb, pm_b : `~astropy.units.Quantity` ['angular speed'], optional
            Proper motion components in the `Galactic` frame, in angle per time
            units.
        x, y, z : float or `~astropy.units.Quantity` ['length'], optional
            Cartesian coordinates values
        u, v, w : float or `~astropy.units.Quantity` ['length'], optional
            Cartesian coordinates values for the Galactic frame.
        radial_velocity : `~astropy.units.Quantity` ['speed'], optional
            The component of the velocity along the line-of-sight (i.e., the
            radial direction), in velocity units.
    """
```
### 22 - astropy/coordinates/sky_coordinate.py:

Start line: 1, End line: 34

```python
import copy
import operator
import re
import warnings

import erfa
import numpy as np

from astropy import units as u
from astropy.constants import c as speed_of_light
from astropy.table import QTable
from astropy.time import Time
from astropy.utils import ShapedLikeNDArray
from astropy.utils.data_info import MixinInfo
from astropy.utils.exceptions import AstropyUserWarning

from .angles import Angle
from .baseframe import BaseCoordinateFrame, GenericFrame, frame_transform_graph
from .distances import Distance
from .representation import (
    RadialDifferential,
    SphericalDifferential,
    SphericalRepresentation,
    UnitSphericalCosLatDifferential,
    UnitSphericalDifferential,
    UnitSphericalRepresentation,
)
from .sky_coordinate_parsers import (
    _get_frame_class,
    _get_frame_without_data,
    _parse_coordinate_data,
)

__all__ = ["SkyCoord", "SkyCoordInfo"]
```
### 23 - astropy/coordinates/sky_coordinate.py:

Start line: 37, End line: 74

```python
class SkyCoordInfo(MixinInfo):
    """
    Container for meta information like name, description, format.  This is
    required when the object is used as a mixin column within a table, but can
    be used as a general way to store meta information.
    """

    attrs_from_parent = {"unit"}  # Unit is read-only
    _supports_indexing = False

    @staticmethod
    def default_format(val):
        repr_data = val.info._repr_data
        formats = ["{0." + compname + ".value:}" for compname in repr_data.components]
        return ",".join(formats).format(repr_data)

    @property
    def unit(self):
        repr_data = self._repr_data
        unit = ",".join(
            str(getattr(repr_data, comp).unit) or "None"
            for comp in repr_data.components
        )
        return unit

    @property
    def _repr_data(self):
        if self._parent is None:
            return None

        sc = self._parent
        if issubclass(sc.representation_type, SphericalRepresentation) and isinstance(
            sc.data, UnitSphericalRepresentation
        ):
            repr_data = sc.represent_as(sc.data.__class__, in_frame_units=True)
        else:
            repr_data = sc.represent_as(sc.representation_type, in_frame_units=True)
        return repr_data
```
### 25 - astropy/coordinates/sky_coordinate.py:

Start line: 478, End line: 509

```python
class SkyCoord(ShapedLikeNDArray):

    def __setitem__(self, item, value):
        """Implement self[item] = value for SkyCoord

        The right hand ``value`` must be strictly consistent with self:
        - Identical class
        - Equivalent frames
        - Identical representation_types
        - Identical representation differentials keys
        - Identical frame attributes
        - Identical "extra" frame attributes (e.g. obstime for an ICRS coord)

        With these caveats the setitem ends up as effectively a setitem on
        the representation data.

          self.frame.data[item] = value.frame.data
        """
        if self.__class__ is not value.__class__:
            raise TypeError(
                "can only set from object of same class: "
                f"{self.__class__.__name__} vs. {value.__class__.__name__}"
            )

        # Make sure that any extra frame attribute names are equivalent.
        for attr in self._extra_frameattr_names | value._extra_frameattr_names:
            if not self.frame._frameattr_equiv(
                getattr(self, attr), getattr(value, attr)
            ):
                raise ValueError(f"attribute {attr} is not equivalent")

        # Set the frame values.  This checks frame equivalence and also clears
        # the cache to ensure that the object is not in an inconsistent state.
        self._sky_coord_frame[item] = value._sky_coord_frame
```
### 26 - astropy/coordinates/sky_coordinate.py:

Start line: 2034, End line: 2076

```python
class SkyCoord(ShapedLikeNDArray):

    def radial_velocity_correction(
        self, kind="barycentric", obstime=None, location=None
    ):
        # ... other code
        if self.data.__class__ is UnitSphericalRepresentation:
            targcart = icrs_cart_novel
        else:
            # skycoord has distances so apply parallax
            obs_icrs_cart = pos_earth + gcrs_p
            targcart = icrs_cart_novel - obs_icrs_cart
            targcart /= targcart.norm()

        if kind == "barycentric":
            beta_obs = (v_origin_to_earth + gcrs_v) / speed_of_light
            gamma_obs = 1 / np.sqrt(1 - beta_obs.norm() ** 2)
            gr = location.gravitational_redshift(obstime)
            # barycentric redshift according to eq 28 in Wright & Eastmann (2014),
            # neglecting Shapiro delay and effects of the star's own motion
            zb = gamma_obs * (1 + beta_obs.dot(targcart)) / (1 + gr / speed_of_light)
            # try and get terms corresponding to stellar motion.
            if icrs_cart.differentials:
                try:
                    ro = self.icrs.cartesian
                    beta_star = ro.differentials["s"].to_cartesian() / speed_of_light
                    # ICRS unit vector at coordinate epoch
                    ro = ro.without_differentials()
                    ro /= ro.norm()
                    zb *= (1 + beta_star.dot(ro)) / (1 + beta_star.dot(targcart))
                except u.UnitConversionError:
                    warnings.warn(
                        "SkyCoord contains some velocity information, but not enough to"
                        " calculate the full space motion of the source, and so this"
                        " has been ignored for the purposes of calculating the radial"
                        " velocity correction. This can lead to errors on the order of"
                        " metres/second.",
                        AstropyUserWarning,
                    )

            zb = zb - 1
            return zb * speed_of_light
        else:
            # do a simpler correction ignoring time dilation and gravitational redshift
            # this is adequate since Heliocentric corrections shouldn't be used if
            # cm/s precision is required.
            return targcart.dot(v_origin_to_earth + gcrs_v)

    # Table interactions
```
### 29 - astropy/coordinates/sky_coordinate.py:

Start line: 580, End line: 616

```python
class SkyCoord(ShapedLikeNDArray):

    def is_transformable_to(self, new_frame):
        """
        Determines if this coordinate frame can be transformed to another
        given frame.

        Parameters
        ----------
        new_frame : frame class, frame object, or str
            The proposed frame to transform into.

        Returns
        -------
        transformable : bool or str
            `True` if this can be transformed to ``new_frame``, `False` if
            not, or the string 'same' if ``new_frame`` is the same system as
            this object but no transformation is defined.

        Notes
        -----
        A return value of 'same' means the transformation will work, but it will
        just give back a copy of this object.  The intended usage is::

            if coord.is_transformable_to(some_unknown_frame):
                coord2 = coord.transform_to(some_unknown_frame)

        This will work even if ``some_unknown_frame``  turns out to be the same
        frame class as ``coord``.  This is intended for cases where the frame
        is the same regardless of the frame attributes (e.g. ICRS), but be
        aware that it *might* also indicate that someone forgot to define the
        transformation between two objects of the same frame class but with
        different attributes.
        """
        # TODO! like matplotlib, do string overrides for modified methods
        new_frame = (
            _get_frame_class(new_frame) if isinstance(new_frame, str) else new_frame
        )
        return self.frame.is_transformable_to(new_frame)
```
### 30 - astropy/coordinates/sky_coordinate.py:

Start line: 109, End line: 2223

```python
class SkyCoordInfo(MixinInfo):

    def new_like(self, skycoords, length, metadata_conflicts="warn", name=None):
        """
        Return a new SkyCoord instance which is consistent with the input
        SkyCoord objects ``skycoords`` and has ``length`` rows.  Being
        "consistent" is defined as being able to set an item from one to each of
        the rest without any exception being raised.

        This is intended for creating a new SkyCoord instance whose elements can
        be set in-place for table operations like join or vstack.  This is used
        when a SkyCoord object is used as a mixin column in an astropy Table.

        The data values are not predictable and it is expected that the consumer
        of the object will fill in all values.

        Parameters
        ----------
        skycoords : list
            List of input SkyCoord objects
        length : int
            Length of the output skycoord object
        metadata_conflicts : str ('warn'|'error'|'silent')
            How to handle metadata conflicts
        name : str
            Output name (sets output skycoord.info.name)

        Returns
        -------
        skycoord : SkyCoord (or subclass)
            Instance of this class consistent with ``skycoords``

        """
        # Get merged info attributes like shape, dtype, format, description, etc.
        attrs = self.merge_cols_attributes(
            skycoords, metadata_conflicts, name, ("meta", "description")
        )
        skycoord0 = skycoords[0]

        # Make a new SkyCoord object with the desired length and attributes
        # by using the _apply / __getitem__ machinery to effectively return
        # skycoord0[[0, 0, ..., 0, 0]]. This will have the all the right frame
        # attributes with the right shape.
        indexes = np.zeros(length, dtype=np.int64)
        out = skycoord0[indexes]

        # Use __setitem__ machinery to check for consistency of all skycoords
        for skycoord in skycoords[1:]:
            try:
                out[0] = skycoord[0]
            except Exception as err:
                raise ValueError("Input skycoords are inconsistent.") from err

        # Set (merged) info attributes
        for attr in ("name", "meta", "description"):
            if attr in attrs:
                setattr(out.info, attr, attrs[attr])

        return out


class SkyCoord(ShapedLikeNDArray):
```
### 39 - astropy/coordinates/sky_coordinate.py:

Start line: 1961, End line: 2033

```python
class SkyCoord(ShapedLikeNDArray):

    def radial_velocity_correction(
        self, kind="barycentric", obstime=None, location=None
    ):
        # has to be here to prevent circular imports
        from .solar_system import get_body_barycentric_posvel

        # location validation
        timeloc = getattr(obstime, "location", None)
        if location is None:
            if self.location is not None:
                location = self.location
                if timeloc is not None:
                    raise ValueError(
                        "`location` cannot be in both the passed-in `obstime` and this"
                        " `SkyCoord` because it is ambiguous which is meant for the"
                        " radial_velocity_correction."
                    )
            elif timeloc is not None:
                location = timeloc
            else:
                raise TypeError(
                    "Must provide a `location` to radial_velocity_correction, either as"
                    " a SkyCoord frame attribute, as an attribute on the passed in"
                    " `obstime`, or in the method call."
                )

        elif self.location is not None or timeloc is not None:
            raise ValueError(
                "Cannot compute radial velocity correction if `location` argument is"
                " passed in and there is also a  `location` attribute on this SkyCoord"
                " or the passed-in `obstime`."
            )

        # obstime validation
        coo_at_rv_obstime = self  # assume we need no space motion for now
        if obstime is None:
            obstime = self.obstime
            if obstime is None:
                raise TypeError(
                    "Must provide an `obstime` to radial_velocity_correction, either as"
                    " a SkyCoord frame attribute or in the method call."
                )
        elif self.obstime is not None and self.frame.data.differentials:
            # we do need space motion after all
            coo_at_rv_obstime = self.apply_space_motion(obstime)
        elif self.obstime is None:
            # warn the user if the object has differentials set
            if "s" in self.data.differentials:
                warnings.warn(
                    "SkyCoord has space motion, and therefore the specified "
                    "position of the SkyCoord may not be the same as "
                    "the `obstime` for the radial velocity measurement. "
                    "This may affect the rv correction at the order of km/s"
                    "for very high proper motions sources. If you wish to "
                    "apply space motion of the SkyCoord to correct for this"
                    "the `obstime` attribute of the SkyCoord must be set",
                    AstropyUserWarning,
                )

        pos_earth, v_earth = get_body_barycentric_posvel("earth", obstime)
        if kind == "barycentric":
            v_origin_to_earth = v_earth
        elif kind == "heliocentric":
            v_sun = get_body_barycentric_posvel("sun", obstime)[1]
            v_origin_to_earth = v_earth - v_sun
        else:
            raise ValueError(
                "`kind` argument to radial_velocity_correction must "
                f"be 'barycentric' or 'heliocentric', but got '{kind}'"
            )

        gcrs_p, gcrs_v = location.get_gcrs_posvel(obstime)
        # transforming to GCRS is not the correct thing to do here, since we don't want to
        # include aberration (or light deflection)? Instead, only apply parallax if necessary
        icrs_cart = coo_at_rv_obstime.icrs.cartesian
        icrs_cart_novel = icrs_cart.without_differentials()
        # ... other code
```
### 41 - astropy/coordinates/sky_coordinate.py:

Start line: 1103, End line: 1146

```python
class SkyCoord(ShapedLikeNDArray):

    def is_equivalent_frame(self, other):
        """
        Checks if this object's frame as the same as that of the ``other``
        object.

        To be the same frame, two objects must be the same frame class and have
        the same frame attributes. For two `SkyCoord` objects, *all* of the
        frame attributes have to match, not just those relevant for the object's
        frame.

        Parameters
        ----------
        other : SkyCoord or BaseCoordinateFrame
            The other object to check.

        Returns
        -------
        isequiv : bool
            True if the frames are the same, False if not.

        Raises
        ------
        TypeError
            If ``other`` isn't a `SkyCoord` or a `BaseCoordinateFrame` or subclass.
        """
        if isinstance(other, BaseCoordinateFrame):
            return self.frame.is_equivalent_frame(other)
        elif isinstance(other, SkyCoord):
            if other.frame.name != self.frame.name:
                return False

            for fattrnm in frame_transform_graph.frame_attributes:
                if not BaseCoordinateFrame._frameattr_equiv(
                    getattr(self, fattrnm), getattr(other, fattrnm)
                ):
                    return False
            return True
        else:
            # not a BaseCoordinateFrame nor a SkyCoord object
            raise TypeError(
                "Tried to do is_equivalent_frame on something that isn't frame-like"
            )

    # High-level convenience methods
```
### 57 - astropy/coordinates/sky_coordinate.py:

Start line: 388, End line: 418

```python
class SkyCoord(ShapedLikeNDArray):

    def __eq__(self, value):
        """Equality operator for SkyCoord

        This implements strict equality and requires that the frames are
        equivalent, extra frame attributes are equivalent, and that the
        representation data are exactly equal.
        """

        if isinstance(value, BaseCoordinateFrame):
            if value._data is None:
                raise ValueError("Can only compare SkyCoord to Frame with data")

            return self.frame == value

        if not isinstance(value, SkyCoord):
            return NotImplemented

        # Make sure that any extra frame attribute names are equivalent.
        for attr in self._extra_frameattr_names | value._extra_frameattr_names:
            if not self.frame._frameattr_equiv(
                getattr(self, attr), getattr(value, attr)
            ):
                raise ValueError(
                    f"cannot compare: extra frame attribute '{attr}' is not equivalent"
                    " (perhaps compare the frames directly to avoid this exception)"
                )

        return self._sky_coord_frame == value._sky_coord_frame

    def __ne__(self, value):
        return np.logical_not(self == value)
```
### 67 - astropy/coordinates/sky_coordinate.py:

Start line: 2181, End line: 2224

```python
class SkyCoord(ShapedLikeNDArray):
    @classmethod
    def from_name(cls, name, frame="icrs", parse=False, cache=True):
        """
        Given a name, query the CDS name resolver to attempt to retrieve
        coordinate information for that object. The search database, sesame
        url, and  query timeout can be set through configuration items in
        ``astropy.coordinates.name_resolve`` -- see docstring for
        `~astropy.coordinates.get_icrs_coordinates` for more
        information.

        Parameters
        ----------
        name : str
            The name of the object to get coordinates for, e.g. ``'M42'``.
        frame : str or `BaseCoordinateFrame` class or instance
            The frame to transform the object to.
        parse : bool
            Whether to attempt extracting the coordinates from the name by
            parsing with a regex. For objects catalog names that have
            J-coordinates embedded in their names, e.g.,
            'CRTS SSS100805 J194428-420209', this may be much faster than a
            Sesame query for the same object name. The coordinates extracted
            in this way may differ from the database coordinates by a few
            deci-arcseconds, so only use this option if you do not need
            sub-arcsecond accuracy for coordinates.
        cache : bool, optional
            Determines whether to cache the results or not. To update or
            overwrite an existing value, pass ``cache='update'``.

        Returns
        -------
        coord : SkyCoord
            Instance of the SkyCoord class.
        """

        from .name_resolve import get_icrs_coordinates

        icrs_coord = get_icrs_coordinates(name, parse, cache=cache)
        icrs_sky_coord = cls(icrs_coord)
        if frame in ("icrs", icrs_coord.__class__):
            return icrs_sky_coord
        else:
            return icrs_sky_coord.transform_to(frame)
```
### 70 - astropy/coordinates/sky_coordinate.py:

Start line: 1696, End line: 1714

```python
class SkyCoord(ShapedLikeNDArray):

    def skyoffset_frame(self, rotation=None):
        """
        Returns the sky offset frame with this `SkyCoord` at the origin.

        Returns
        -------
        astrframe : `~astropy.coordinates.SkyOffsetFrame`
            A sky offset frame of the same type as this `SkyCoord` (e.g., if
            this object has an ICRS coordinate, the resulting frame is
            SkyOffsetICRS, with the origin set to this object)
        rotation : angle-like
            The final rotation of the frame about the ``origin``. The sign of
            the rotation is the left-hand rule. That is, an object at a
            particular position angle in the un-rotated system will be sent to
            the positive latitude (z) direction in the final frame.
        """
        from .builtin_frames.skyoffset import SkyOffsetFrame

        return SkyOffsetFrame(origin=self, rotation=rotation)
```
### 75 - astropy/coordinates/sky_coordinate.py:

Start line: 1791, End line: 1822

```python
class SkyCoord(ShapedLikeNDArray):

    @classmethod
    def from_pixel(cls, xp, yp, wcs, origin=0, mode="all"):
        """
        Create a new `SkyCoord` from pixel coordinates using an
        `~astropy.wcs.WCS` object.

        Parameters
        ----------
        xp, yp : float or ndarray
            The coordinates to convert.
        wcs : `~astropy.wcs.WCS`
            The WCS to use for convert
        origin : int
            Whether to return 0 or 1-based pixel coordinates.
        mode : 'all' or 'wcs'
            Whether to do the transformation including distortions (``'all'``) or
            only including only the core WCS transformation (``'wcs'``).

        Returns
        -------
        coord : `~astropy.coordinates.SkyCoord`
            A new object with sky coordinates corresponding to the input ``xp``
            and ``yp``.

        See Also
        --------
        to_pixel : to do the inverse operation
        astropy.wcs.utils.pixel_to_skycoord : the implementation of this method
        """
        from astropy.wcs.utils import pixel_to_skycoord

        return pixel_to_skycoord(xp, yp, wcs=wcs, origin=origin, mode=mode, cls=cls)
```
### 76 - astropy/coordinates/sky_coordinate.py:

Start line: 618, End line: 704

```python
class SkyCoord(ShapedLikeNDArray):

    def transform_to(self, frame, merge_attributes=True):
        """Transform this coordinate to a new frame.

        The precise frame transformed to depends on ``merge_attributes``.
        If `False`, the destination frame is used exactly as passed in.
        But this is often not quite what one wants.  E.g., suppose one wants to
        transform an ICRS coordinate that has an obstime attribute to FK4; in
        this case, one likely would want to use this information. Thus, the
        default for ``merge_attributes`` is `True`, in which the precedence is
        as follows: (1) explicitly set (i.e., non-default) values in the
        destination frame; (2) explicitly set values in the source; (3) default
        value in the destination frame.

        Note that in either case, any explicitly set attributes on the source
        `SkyCoord` that are not part of the destination frame's definition are
        kept (stored on the resulting `SkyCoord`), and thus one can round-trip
        (e.g., from FK4 to ICRS to FK4 without losing obstime).

        Parameters
        ----------
        frame : str, `BaseCoordinateFrame` class or instance, or `SkyCoord` instance
            The frame to transform this coordinate into.  If a `SkyCoord`, the
            underlying frame is extracted, and all other information ignored.
        merge_attributes : bool, optional
            Whether the default attributes in the destination frame are allowed
            to be overridden by explicitly set attributes in the source
            (see note above; default: `True`).

        Returns
        -------
        coord : `SkyCoord`
            A new object with this coordinate represented in the `frame` frame.

        Raises
        ------
        ValueError
            If there is no possible transformation route.

        """
        from astropy.coordinates.errors import ConvertError

        frame_kwargs = {}

        # Frame name (string) or frame class?  Coerce into an instance.
        try:
            frame = _get_frame_class(frame)()
        except Exception:
            pass

        if isinstance(frame, SkyCoord):
            frame = frame.frame  # Change to underlying coord frame instance

        if isinstance(frame, BaseCoordinateFrame):
            new_frame_cls = frame.__class__
            # Get frame attributes, allowing defaults to be overridden by
            # explicitly set attributes of the source if ``merge_attributes``.
            for attr in frame_transform_graph.frame_attributes:
                self_val = getattr(self, attr, None)
                frame_val = getattr(frame, attr, None)
                if frame_val is not None and not (
                    merge_attributes and frame.is_frame_attr_default(attr)
                ):
                    frame_kwargs[attr] = frame_val
                elif self_val is not None and not self.is_frame_attr_default(attr):
                    frame_kwargs[attr] = self_val
                elif frame_val is not None:
                    frame_kwargs[attr] = frame_val
        else:
            raise ValueError(
                "Transform `frame` must be a frame name, class, or instance"
            )

        # Get the composite transform to the new frame
        trans = frame_transform_graph.get_transform(self.frame.__class__, new_frame_cls)
        if trans is None:
            raise ConvertError(
                f"Cannot transform from {self.frame.__class__} to {new_frame_cls}"
            )

        # Make a generic frame which will accept all the frame kwargs that
        # are provided and allow for transforming through intermediate frames
        # which may require one or more of those kwargs.
        generic_frame = GenericFrame(frame_kwargs)

        # Do the transformation, returning a coordinate frame of the desired
        # final type (not generic).
        new_coord = trans(self.frame, generic_frame)
        # ... other code
```
